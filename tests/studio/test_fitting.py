"""Fitting engine: backends, bounds, constraints, priors, several data sets.

The model is a small analytic function (no SASfit library needed), written like a ScatteringContribution so that the
fit addresses ("0.ff.A", "scale.0", "background") work unchanged.  Its result carries a deterministic jitter of
1e-5, like the adaptive integrals of SASfit, so that the optimisers are tested under realistic noise.
"""

import threading

import numpy as np
import pytest

from pysasfit.studio import fitting
from pysasfit.studio.fitting import FitProblem
from pysasfit.studio.model import ModelFunction


class _Function:
    parameter_names = ["A", "k"]


class Toy:
    """I(q) = A exp(-k q) with a relative jitter of 1e-5 (an 'adaptive integral')."""

    def __init__(self, A=2.0, k=1.0, jitter=1e-5):
        self._v = {"ff": np.array([A, k], float)}
        self._f = _Function()
        self.jitter = jitter

    def function_for(self, role):
        return self._f if role == "ff" else None

    def vector(self, role):
        return self._v.get(role)

    def intensity(self, q):
        A, k = self._v["ff"]
        h = np.sin(1e7 * (A * 1.234 + k * 2.345) + np.arange(q.size)) * self.jitter
        return A * np.exp(-k * q) * (1 + h)


Q = np.linspace(0.1, 3, 60)


def make_data(A=5.0, k=0.8, bg=0.3, seed=1):
    rng = np.random.default_rng(seed)
    I = A * np.exp(-k * Q) + bg
    s = 0.01 * (I + 0.05)
    return Q, I + rng.normal(0, s), s


def make_model(A=2.0, k=1.0, scale=1.0, bg=0.0):
    return ModelFunction([Toy(A, k)], scales=[scale], background=bg)


def problem(data=None, upper=True, start=(2.0, 1.0)):
    p = FitProblem()
    p.add_dataset(data or make_data(), make_model(*start))
    p.add_parameter("0.ff.A", lower=0, upper=20 if upper else np.inf)
    p.add_parameter("0.ff.k", lower=0, upper=5 if upper else np.inf)
    p.add_parameter("background", lower=-1, upper=2 if upper else np.inf)
    return p


def installed(module):
    from importlib.util import find_spec

    return find_spec(module) is not None


CASES = [
    ("scipy", "trf", "scipy"),
    ("scipy", "dogbox", "scipy"),
    ("scipy-global", "differential_evolution", "scipy"),
    ("scipy-global", "basinhopping", "scipy"),
    ("nlopt", "LN_BOBYQA", "nlopt"),
    ("nlopt", "LD_SLSQP", "nlopt"),
    ("iminuit", "migrad", "iminuit"),
    ("iminuit", "simplex", "iminuit"),
    ("bumps", "lm", "bumps"),
    ("bumps", "amoeba", "bumps"),
    ("cma", "", "cma"),
]


def _case(backend, algorithm, module):
    return pytest.param(backend, algorithm, id=f"{backend}-{algorithm or 'default'}",
                        marks=pytest.mark.skipif(not installed(module), reason=f"{module} is not installed"))


@pytest.mark.parametrize("backend,algorithm", [_case(*c) for c in CASES])
def test_recovers_parameters(backend, algorithm):
    options = {"polish": True} if backend != "scipy" else {}
    r = problem().fit(backend, algorithm, **options)
    v = r.values
    assert abs(v["ff.A"] - 5) < 0.4 and abs(v["ff.k"] - 0.8) < 0.06 and abs(v["background"] - 0.3) < 0.06
    assert r.chi2_red < 2.0
    assert r.errors["ff.A"] > 0 and r.errors["ff.k"] > 0
    assert r.nfev > 0 and r.dof == Q.size - 3


def test_bounds_are_respected():
    p = FitProblem()
    p.add_dataset(make_data(), make_model())
    p.add_parameter("0.ff.A", lower=0, upper=3)  # the optimum (5) is outside
    p.add_parameter("0.ff.k", lower=0)
    r = p.fit("scipy")
    assert r.values["ff.A"] <= 3 + 1e-9
    assert "ff.A" in r.at_bounds


def test_fixed_parameter_is_not_fitted():
    p = problem()
    p.parameter("background").fixed = True
    p.parameter("background").value = 0.3
    r = p.fit("scipy")
    assert r.values["background"] == 0.3 and "background" not in r.errors
    assert r.dof == Q.size - 2


def test_unknown_backend_and_algorithm():
    with pytest.raises(ValueError):
        problem().fit("no-such-backend")
    with pytest.raises(ValueError):
        problem().fit("scipy-global", "no-such-method")


def test_global_methods_need_finite_bounds():
    with pytest.raises(ValueError, match="bound"):
        problem(upper=False).fit("scipy-global", "differential_evolution")


def test_register_backend():
    calls = []

    def runner(ctx, algorithm, options):
        calls.append(algorithm)
        return ctx.u0, "nothing done", True

    fitting.register_backend("dummy-test", runner)
    try:
        assert "dummy-test" in fitting.available_backends()
        r = problem().fit("dummy-test", "x")
        assert calls == ["x"] and r.message == "nothing done"
    finally:
        fitting.BACKENDS.pop("dummy-test")


def test_available_backends_lists_installed_packages_only():
    av = fitting.available_backends()
    assert "scipy" in av and "scipy-global" in av
    for key, module in (("nlopt", "nlopt"), ("iminuit", "iminuit"), ("bumps", "bumps"), ("cma", "cma")):
        assert (key in av) == installed(module)
    label, algs = av["scipy"]
    assert set(algs) == set(fitting.SCIPY_ALGORITHMS)


@pytest.mark.parametrize("backend,algorithm", [_case("scipy", "trf", "scipy"), _case("nlopt", "LN_COBYLA", "nlopt"),
                                               _case("iminuit", "migrad", "iminuit"), _case("cma", "", "cma")])
def test_inequality_constraint(backend, algorithm):
    p = problem()
    p.add_constraint("ff_A + 10*ff_k < 10")  # the unconstrained optimum gives 13
    r = p.fit(backend, algorithm, polish=backend != "scipy")
    g = r.values["ff.A"] + 10 * r.values["ff.k"] - 10
    assert g < 0.1  # the penalty leaves a small violation
    assert g > -2.0  # and the constraint is active
    assert r.constraints[0][2] in (True, False)


def test_constraint_text_is_checked():
    p = problem()
    for bad in ("__import__('os').system('x') < 1", "ff_A.real < 1", "foo < 1", "ff_A"):
        with pytest.raises(ValueError):
            p.add_constraint(bad)


def test_equality_constraint_in_a_mixture():
    qq = Q
    true = 0.3 * 5 * np.exp(-0.5 * qq) + 0.7 * np.exp(-3.0 * qq)
    s = 0.01 * (true + 0.05)
    obs = true + np.random.default_rng(2).normal(0, s)
    m = ModelFunction([Toy(5, 0.5), Toy(1, 3.0)], scales=[0.5, 0.5])
    p = FitProblem()
    p.add_dataset((qq, obs, s), m)
    p.add_parameter("scale.0", lower=0, upper=1, label="c1.scale")
    p.add_parameter("scale.1", lower=0, upper=1, label="c2.scale")
    p.add_constraint("c1_scale + c2_scale == 1")
    r = p.fit("scipy")
    v = r.values
    assert abs(v["c1.scale"] + v["c2.scale"] - 1) < 2e-3
    assert abs(v["c1.scale"] - 0.3) < 0.02


def test_gaussian_prior_pulls_the_parameter():
    p = problem()
    p.add_prior("ff.k", 1.2, 1e-4)
    r = p.fit("scipy")
    assert r.values["ff.k"] > 1.15
    assert r.dof == Q.size + 1 - 3  # the prior counts as a data point


def test_two_data_sets_common_and_local_parameters():
    model = make_model()  # one model object shared by both data sets
    p = FitProblem()
    p.add_dataset(make_data(5, 0.8, 0.3, seed=1), model, name="a")
    p.add_dataset(make_data(2, 0.8, 0.1, seed=2), model, name="b")
    p.add_common("0.ff.k", lower=0)
    p.add_local("0.ff.A", lower=0)
    p.add_local("background")
    r = p.fit("scipy")
    v = r.values
    assert abs(v["ff.k"] - 0.8) < 0.03
    assert abs(v["ff.A[a]"] - 5) < 0.3 and abs(v["ff.A[b]"] - 2) < 0.2
    assert abs(v["background[b]"] - 0.1) < 0.05
    for k, name in enumerate("ab"):  # every curve uses its own local values
        expect = v[f"ff.A[{name}]"] * np.exp(-v["ff.k"] * Q[0]) + v[f"background[{name}]"]
        assert abs(r.curves[k][1][0] - expect) < 1e-3 * expect


def test_fit_q_range_selects_points():
    p = FitProblem()
    p.add_dataset(make_data(), make_model(), q_range=(0.5, 2.0))
    p.add_parameter("0.ff.A")
    p.add_parameter("0.ff.k")
    p.add_parameter("background")
    n = int(np.sum((Q >= 0.5) & (Q <= 2.0)))
    assert p.fit("scipy").dof == n - 3


@pytest.mark.parametrize("backend,algorithm", [_case("scipy", None, "scipy"), _case("nlopt", "LN_NELDERMEAD", "nlopt")])
def test_cancel(backend, algorithm):
    stop = threading.Event()

    def callback(nfev, chi2):
        if nfev >= 15:
            stop.set()

    r = problem().fit(backend, algorithm, callback=callback, cancel=stop)
    assert r.message == "cancelled" and not r.success


@pytest.mark.skipif(not installed("iminuit"), reason="iminuit is not installed")
def test_minos_gives_asymmetric_errors():
    r = problem().fit("iminuit", "migrad+minos")
    assert set(r.asymmetric) == {"ff.A", "ff.k", "background"}
    for lo, hi in r.asymmetric.values():
        assert lo < 0 < hi
    assert "/" in r.text()  # printed as (-a / +b)


@pytest.mark.skipif(not installed("bumps"), reason="bumps is not installed")
def test_dream_reports_posterior_errors():
    r = problem().fit("bumps", "dream", samples=3000, burn=30)
    assert r.errors["ff.k"] > 0
    lo, hi = r.asymmetric["ff.k"]
    assert lo < 0 < hi
    assert abs(r.values["ff.k"] - 0.8) < 0.15


@pytest.mark.skipif(not installed("bumps"), reason="bumps is not installed")
def test_bumps_keeps_the_order_of_the_parameters():
    # bumps sorts parameters by name ('background' < 'ff.A' < 'ff.k'), which differs from the order of the fit
    r = problem().fit("bumps", "lm")
    assert abs(r.values["ff.A"] - 5) < 0.4 and abs(r.values["background"] - 0.3) < 0.06


def test_polish_reaches_the_least_squares_minimum():
    ref = problem().fit("scipy")
    r = problem().fit("scipy-global", "differential_evolution", maxeval=600, polish=True)
    assert abs(r.chi2_red - ref.chi2_red) < 0.02
    assert "refined" in r.message


def test_maxeval_limits_the_work():
    r = problem().fit("scipy-global", "differential_evolution", maxeval=200)
    assert r.nfev <= 400
