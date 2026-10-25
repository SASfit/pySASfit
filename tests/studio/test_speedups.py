"""DFO-LS backend and the speed-ups of the fit (no SASfit library needed): variable projection, form-factor cache,
coarse to fine, fixed size grid, model profiling and the node check."""

import numpy as np
import pytest

from pysasfit.studio import fitting
from pysasfit.studio.fitting import FitProblem
from pysasfit.studio.integration import ContributionCache, format_node_check, suggest_n_x
from pysasfit.studio.model import ModelFunction, ScatteringContribution
from pysasfit.studio.profiling import format_profile, profile_model

try:  # tests/studio may or may not be a package
    from .test_fitting import Q, installed, make_data, make_model, problem
except ImportError:
    from test_fitting import Q, installed, make_data, make_model, problem


# ----------------------------------------------------------------------------- DFO-LS
@pytest.mark.skipif(not installed("dfols"), reason="DFO-LS is not installed")
@pytest.mark.parametrize("algorithm", ["", "noisy"])
def test_dfols_recovers_parameters(algorithm):
    r = problem().fit("dfols", algorithm, polish=True)
    v = r.values
    assert abs(v["ff.A"] - 5) < 0.4 and abs(v["ff.k"] - 0.8) < 0.06 and abs(v["background"] - 0.3) < 0.06
    assert r.chi2_red < 2.0 and r.errors["ff.k"] > 0


@pytest.mark.skipif(not installed("dfols"), reason="DFO-LS is not installed")
def test_dfols_respects_bounds_and_constraints():
    p = problem()
    p.add_constraint("ff_A + 10*ff_k < 10")
    r = p.fit("dfols", polish=True)
    assert r.values["ff.A"] + 10 * r.values["ff.k"] - 10 < 0.1
    assert 0 <= r.values["ff.k"] <= 5


def test_dfols_is_listed_only_when_installed():
    assert ("dfols" in fitting.available_backends()) == installed("dfols")


# ------------------------------------------------------------------ variable projection
def linear_problem():
    """I = scale * exp(-k q) + background: scale and background are linear, k is not (A is not fitted)."""
    p = FitProblem()
    p.add_dataset(make_data(), make_model(1.0, 1.0))
    p.add_parameter("scale.0", lower=0, upper=50, value=1.0)
    p.add_parameter("0.ff.k", lower=0, upper=5)
    p.add_parameter("background", lower=-1, upper=2)
    return p


def test_variable_projection_matches_the_full_fit():
    full = linear_problem().fit("scipy")
    vp = linear_problem().fit("scipy", vp=True)
    for label in full.values:
        assert vp.values[label] == pytest.approx(full.values[label], rel=2e-3, abs=2e-3)
    assert vp.chi2 == pytest.approx(full.chi2, rel=1e-3)
    assert "variable projection (2 linear" in vp.message
    assert vp.dof == full.dof
    for label in full.errors:  # errors come from the full problem
        assert vp.errors[label] == pytest.approx(full.errors[label], rel=0.1)


def test_variable_projection_needs_fewer_evaluations():
    full = linear_problem().fit("scipy")
    vp = linear_problem().fit("scipy", vp=True)
    assert vp.nfev < full.nfev


def test_variable_projection_respects_bounds_of_linear_parameters():
    p = linear_problem()
    p.parameter("scale").upper = 3.0  # the optimum (5) is outside
    r = p.fit("scipy", vp=True)
    assert r.values["scale"] <= 3.0 + 1e-9 and "scale" in r.at_bounds


def test_variable_projection_keeps_constrained_parameters_nonlinear():
    p = linear_problem()
    p.add_constraint("scale < 4")
    lin = p.linear_parameters(range(len(p.parameters)))
    assert [p.parameters[i].label for i in lin] == ["background"]


def test_variable_projection_with_only_linear_parameters():
    p = FitProblem()
    p.add_dataset(make_data(5.0, 0.8, 0.3), make_model(1.0, 0.8))
    p.add_parameter("scale.0", lower=0)
    p.add_parameter("background")
    r = p.fit("scipy", vp=True)
    assert "linear least squares" in r.message
    assert abs(r.values["scale"] - 5) < 0.2 and abs(r.values["background"] - 0.3) < 0.05


@pytest.mark.skipif(not installed("nlopt"), reason="nlopt is not installed")
def test_variable_projection_works_with_other_backends():
    r = linear_problem().fit("nlopt", "LN_BOBYQA", vp=True, polish=True)
    assert abs(r.values["scale"] - 5) < 0.4 and abs(r.values["ff.k"] - 0.8) < 0.06


# --------------------------------------------------------------------------- cache
class _Sphere:
    """Form factor of one parameter R; counts its evaluations."""

    parameter_names = ["R"]

    def __init__(self):
        self.calls = 0

    def evaluate_p(self, q, p, kind="intensity"):
        self.calls += 1
        x = q * p[0]
        return (np.sin(x) - x * np.cos(x)) ** 2 / x**6 * p[0] ** 6


class _TwoParam(_Sphere):
    parameter_names = ["R", "contrast"]

    def evaluate_p(self, q, p, kind="intensity"):
        return p[1] ** 2 * super().evaluate_p(q, p, kind)


class _Gauss:
    """N(x) = N exp(-(x - mu)^2 / 2 s^2)."""

    parameter_names = ["N", "mu", "s"]

    def evaluate_p(self, x, p):
        return p[0] * np.exp(-0.5 * ((np.asarray(x, float) - p[1]) / p[2]) ** 2)


def distributed_contribution():
    c = object.__new__(ScatteringContribution)
    c.ff, c.sd, c.sq = _Sphere(), _Gauss(), None
    c.ff_p, c.sd_p, c.sq_p = np.array([5.0]), np.array([1.0, 5.0, 0.5]), None
    c.distributed, c.approach, c.x_range = 0, "monodisperse", None
    c.n_x, c.method, c.rtol, c.atol, c.limit = 24, "gauss-legendre", 1e-6, 0.0, 200
    c.x_power, c.tail = 6.0, 1e-10
    c.last_integration, c.cache, c.snap = None, None, 0.0
    c.fixed_grid, c._grid, c.grid_margin = False, None, 1.6
    return c


QQ = np.linspace(0.05, 1.0, 25)


def test_amplitude_scales_the_intensity_without_recomputing_the_form_factor():
    c = distributed_contribution()
    c.enable_cache(snap=0.05)
    I1 = c.intensity(QQ)
    calls = c.ff.calls
    assert calls > 0
    c.sd_p[0] = 3.0  # only the amplitude N changes
    I2 = c.intensity(QQ)
    assert c.ff.calls == calls  # the nodes are reused
    np.testing.assert_allclose(I2, 3.0 * I1, rtol=1e-12)
    assert c.cache.node_hits > 0


def test_width_change_reuses_the_form_factor_on_unchanged_nodes():
    c = distributed_contribution()
    c.enable_cache(snap=0.5)  # a coarse grid: a tiny change of the width keeps the integration range
    c.intensity(QQ)
    calls = c.ff.calls
    c.sd_p[2] = 0.5005
    c.intensity(QQ)
    assert c.ff.calls == calls


def test_form_factor_parameter_change_invalidates_the_nodes():
    c = distributed_contribution()
    c.ff = _TwoParam()
    c.ff_p = np.array([5.0, 1.0])
    c.enable_cache(snap=0.05)
    c.intensity(QQ)
    calls = c.ff.calls
    c.ff_p[1] = 2.0  # a form-factor parameter other than the distributed one
    c.intensity(QQ)
    assert c.ff.calls > calls


def test_cached_intensity_equals_uncached_up_to_the_snapped_range():
    plain, cached = distributed_contribution(), distributed_contribution()
    cached.enable_cache(snap=0.02)
    np.testing.assert_allclose(cached.intensity(QQ), plain.intensity(QQ), rtol=1e-6)


def test_identical_call_is_served_from_the_result_cache():
    c = distributed_contribution()
    c.enable_cache(snap=0.05)
    I1 = c.intensity(QQ)
    calls = c.ff.calls
    I2 = c.intensity(QQ)
    assert c.ff.calls == calls and c.cache.result_hits == 1
    I2[0] = -1  # the cache hands out copies
    np.testing.assert_array_equal(c.intensity(QQ), I1)


def test_snap_rounds_the_range_outwards():
    c = distributed_contribution()
    lo0, hi0 = c.support()
    c.snap = 0.05
    lo, hi = c.support()
    assert lo <= lo0 * (1 + 1e-12) and hi >= hi0 * (1 - 1e-12)
    assert hi / hi0 < 1.06 and lo0 / lo < 1.06


def test_amplitude_is_set_to_one_in_components():
    c = distributed_contribution()
    m = ModelFunction([c], scales=[2.0])
    c.sd_p[0] = 4.0
    ref = distributed_contribution().intensity(QQ)  # N = 1
    parts, _bg = m.components(QQ, {0})
    np.testing.assert_allclose(parts[0], ref, rtol=1e-12)  # N is set to 1 while evaluating ...
    assert c.sd_p[0] == 4.0  # ... and restored afterwards


def test_fit_restores_the_cache_state_of_the_contributions():
    c = distributed_contribution()
    obs = c.intensity(QQ) * 2.0
    p = FitProblem()
    p.add_dataset((QQ, obs, 0.01 * obs), ModelFunction([c], scales=[1.0]))
    p.add_parameter("scale.0", lower=0, upper=10)
    r = p.fit("scipy", vp=True, cache=True)
    assert c.cache is None and c.snap == 0.0  # back to the state before the fit
    assert abs(r.values["scale"] - 2.0) < 1e-3
    assert "cache:" in r.message


def test_cache_lru_is_bounded():
    cache = ContributionCache(max_results=3, max_scopes=2)
    for i in range(10):
        cache.put(i, np.zeros(1))
        cache.scope(i)
    assert len(cache._results) == 3 and len(cache._scopes) == 2
    assert cache.get(9) is not None and cache.get(0) is None


# ------------------------------------------------------------------ coarse to fine
def size_fit_problem(n_x=48):
    c = distributed_contribution()
    c.n_x = n_x
    obs = c.intensity(QQ)
    p = FitProblem()
    c2 = distributed_contribution()
    c2.n_x = n_x
    c2.ff_p[0], c2.sd_p[1], c2.sd_p[2] = 5.0, 5.4, 0.7  # start away from the truth
    p.add_dataset((QQ, obs, 0.01 * obs), ModelFunction([c2], scales=[1.0]))
    p.add_parameter("0.sd.mu", lower=3, upper=8)
    p.add_parameter("0.sd.s", lower=0.1, upper=2)
    p.add_parameter("scale.0", lower=0, upper=10)
    return p, c2


def test_coarse_to_fine_reaches_the_same_minimum_with_cheaper_evaluations():
    p, c = size_fit_problem()
    fine = p.fit("scipy", vp=True)
    ff_calls_fine = c.ff.calls
    p2, c2 = size_fit_problem()
    r = p2.fit("scipy", vp=True, coarse=True)
    assert abs(r.values["sd.mu"] - 5.0) < 0.02 and abs(r.values["sd.s"] - 0.5) < 0.02
    assert r.values["sd.mu"] == pytest.approx(fine.values["sd.mu"], abs=0.02)
    assert c2.ff.calls < ff_calls_fine  # fewer form-factor evaluations in total
    assert r.message.startswith("coarse to fine")
    assert c2.n_x == 48 and c2.rtol == 1e-6  # settings restored


def test_coarse_to_fine_uses_a_local_second_stage_after_a_global_search():
    p, _c = size_fit_problem()
    r = p.fit("scipy-global", "differential_evolution", coarse=True, maxeval=300)
    assert "stage 2" in r.message and r.backend == "scipy"


def test_coarse_without_size_distribution_is_a_plain_fit():
    r = problem().fit("scipy", coarse=True)
    assert not r.message.startswith("coarse")


# ------------------------------------------------------------------ fixed grid
class _SQ:
    parameter_names = ["Rhs"]

    def evaluate_p(self, q, p, kind="intensity"):
        return 1.0 + 0.5 * np.exp(-np.asarray(q, float) * p[0])


def test_fixed_grid_agrees_with_gauss_legendre():
    plain, grid = distributed_contribution(), distributed_contribution()
    grid.fixed_grid = True
    np.testing.assert_allclose(grid.intensity(QQ), plain.intensity(QQ), rtol=2e-4)
    assert grid.last_integration["method"] == "fixed grid"


def test_fixed_grid_decoupling_agrees_with_gauss_legendre():
    plain, grid = distributed_contribution(), distributed_contribution()
    for c in (plain, grid):
        c.sq, c.sq_p, c.approach = _SQ(), np.array([5.0]), "decoupling"
    grid.fixed_grid = True
    np.testing.assert_allclose(grid.intensity(QQ), plain.intensity(QQ), rtol=2e-4)


def test_fixed_grid_reuses_the_form_factor_when_the_distribution_changes():
    c = distributed_contribution()
    c.fixed_grid = True
    c.enable_cache(snap=0.05)
    c.intensity(QQ)
    calls = c.ff.calls
    c.sd_p[1], c.sd_p[2] = 5.1, 0.52  # mu and width change
    I = c.intensity(QQ)
    assert c.ff.calls == calls  # no new form-factor evaluation
    ref = distributed_contribution()
    ref.sd_p[1], ref.sd_p[2] = 5.1, 0.52
    np.testing.assert_allclose(I, ref.intensity(QQ), rtol=2e-4)


def test_fixed_grid_is_rebuilt_when_the_distribution_leaves_it():
    c = distributed_contribution()
    c.fixed_grid = True
    c.intensity(QQ)
    first = c._grid
    c.sd_p[1], c.sd_p[2] = 12.0, 1.0
    I = c.intensity(QQ)
    assert c._grid is not first
    ref = distributed_contribution()
    ref.sd_p[1], ref.sd_p[2] = 12.0, 1.0
    np.testing.assert_allclose(I, ref.intensity(QQ), rtol=2e-4)


def test_fit_with_the_fixed_grid_needs_far_fewer_form_factor_calls():
    p, c = size_fit_problem()
    plain = p.fit("scipy", vp=True)
    calls_plain = c.ff.calls
    p2, c2 = size_fit_problem()
    r = p2.fit("scipy", vp=True, grid=True)
    assert abs(r.values["sd.mu"] - plain.values["sd.mu"]) < 0.02 and abs(r.values["sd.s"] - plain.values["sd.s"]) < 0.02
    assert c2.ff.calls < 0.5 * calls_plain
    assert c2.fixed_grid is False and c2.cache is None and c2._grid is None  # state restored


# ------------------------------------------------------------------ profiler
def test_profile_splits_the_time_and_restores_the_functions():
    c = distributed_contribution()
    c.n_x = 24
    res = profile_model(ModelFunction([c], scales=[1.0]), QQ, repeat=2)
    entry = res["contributions"][0]
    assert entry["ff"]["calls"] == 4 * 24 and entry["sd"]["calls"] >= 4 * 24
    assert res["total"] > 0 and res["overhead"] >= 0 and res["q_points"] == QQ.size
    text = format_profile(res, n_free=3)
    assert "form factor" in text and "size distribution" in text and "fit with 3 free parameters" in text
    assert "evaluate_p" not in vars(c.ff) and "evaluate_p" not in vars(c.sd)  # instrumentation removed


# ------------------------------------------------------------------ node check
def test_suggest_n_x_finds_a_smaller_sufficient_number():
    c = distributed_contribution()
    c.n_x = 96
    r = suggest_n_x(c, QQ, tol=1e-4)
    assert r["n_x"] is not None and r["n_x"] < 96
    assert c.n_x == 96 and c.cache is None  # restored
    assert all(e <= 1e-4 for n, e in r["table"] if n >= r["n_x"])
    assert "is enough" in format_node_check([("contribution 1", r)])


def test_suggest_n_x_reports_adaptive_methods():
    c = distributed_contribution()
    c.method = "quad_vec"
    r = suggest_n_x(c, QQ)
    assert r["n_x"] is None and "chooses its nodes itself" in format_node_check([("c1", r)])


# ------------------------------------------------------------------ correlations in the result text
def test_result_text_has_a_correlation_matrix():
    r = problem().fit("scipy")
    t = r.text()
    assert "correlation matrix of the free parameters" in t
    assert "ff.A" in t.split("correlation matrix")[1]


def test_strongly_correlated_parameters_are_reported_with_advice():
    r = problem().fit("scipy")
    r.covariance = np.array([[1.0, -0.99, 0.2], [-0.99, 1.0, 0.1], [0.2, 0.1, 1.0]])  # ff.A / ff.k strongly correlated
    t = r.text()
    assert "strongly correlated" in t and "ff.A / ff.k: r = -0.990" in t
    assert "ff.k / background" not in t  # only pairs above the threshold
    assert "Priors tab" in t and "fix one of them" in t
    r.covariance = np.eye(3)
    assert "strongly correlated" not in r.text()
