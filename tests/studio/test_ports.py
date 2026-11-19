"""Python ports of SASfit plugins (pysasfit.studio.ports) against the unchanged C sources and closed forms.

The C reference is built from tests/studio/port_reference/sasfit_src (copies of the SASfit plugin sources) with
small shim headers; it needs a C compiler and GSL, otherwise those tests are skipped.
"""

import ctypes
import importlib.util
import math
import os

import numpy as np
import pytest
from scipy import special

from pysasfit.studio import functions, ports
from pysasfit.studio.ports import check
from pysasfit.studio.ports import deformed_polymers as dp
from pysasfit.studio.ports.quadrature import integrate_01, tanh_sinh_01

HERE = os.path.dirname(__file__)


class _Param(ctypes.Structure):
    _fields_ = [
        ("p", ctypes.c_double * 50),
        ("errStatus", ctypes.c_int),
        ("errStr", ctypes.c_char * 256),
        ("kernelSelector", ctypes.c_int),  # port_reference shim: sasfit_param.kernelSelector
    ]


@pytest.fixture(scope="module")
def portref(tmp_path_factory):
    # tests/studio/port_reference/build.py, under its own module name (mock_sasfit/build.py is 'build')
    spec = importlib.util.spec_from_file_location(
        "port_reference_build", os.path.join(HERE, "port_reference", "build.py")
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    lib = mod.build(str(tmp_path_factory.mktemp("portref")))
    if lib is None:
        pytest.skip("no C compiler or GSL for the C reference of the ports")
    return ctypes.CDLL(lib)


def c_reference(lib, name):
    f = getattr(lib, "sasfit_" + name)
    f.restype = ctypes.c_double
    f.argtypes = [ctypes.c_double, ctypes.POINTER(_Param)]

    def ref(q, p):
        s = _Param()
        s.p[:] = [float(v) for v in p]
        out = np.empty(len(q))
        for i, qi in enumerate(q):
            out[i] = f(float(qi), ctypes.byref(s))
            if s.errStatus:
                raise RuntimeError(s.errStr.decode())
        return out

    return ref


# --------------------------------------------------------------------------------------------- against C
_PILOT = [
    "ff_deformed_generalized_gaussian_chain",
    "ff_deformed_ring_polymer__bmb_",
    "ff_deformed_ring_polymer__bz_",
    "ff_ringpolymerBMB",
    "ff_ringpolymerBZ",
]


@pytest.mark.parametrize("name", _PILOT)
def test_port_matches_c_source(portref, name):
    r = check.compare(name, reference=c_reference(portref, name), n_sets=12, seed=3)
    assert r.passed, str(r)
    assert r.n_ref_zero == 0


def test_ggc_matches_c_everywhere(portref):
    """Since the C fix of 2026-11-17 (series for u < 1 instead of the Guinier branch) C and port agree everywhere."""
    ref = c_reference(portref, "ff_deformed_generalized_gaussian_chain")
    spec = ports.get("ff_deformed_generalized_gaussian_chain")
    for p in ports.sample_parameters(spec, 10, seed=1):
        q = np.logspace(-4, 1, 300)
        np.testing.assert_allclose(ref(q, p), spec(q, p), rtol=1e-12)


@pytest.mark.parametrize(
    "name",
    [
        "ff_ringpolymerBMB",
        "ff_ringpolymerBZ",
        "ff_deformed_ring_polymer__bmb_",
        "ff_deformed_ring_polymer__bz_",
    ],
)
def test_c_rings_independent_of_the_integrator(portref, name):
    """With plain adaptive QAGS (rel. 1e-8, as a SASfit strategy) the fixed C rings stay correct up to q Rg ~ 1e3;
    before the fix they returned 0 or wrong values from q Rg ~ 30 on."""
    portref.shim_set_integrator.argtypes = [ctypes.c_int]
    portref.shim_set_integrator(1)
    try:
        r = check.compare(
            name, reference=c_reference(portref, name), n_sets=12, seed=5, rtol=1e-7
        )
    finally:
        portref.shim_set_integrator(0)
    assert r.passed, str(r)


def test_c_psi_override_equals_port_psi_argument(portref):
    """SASfit's ψ override (radians) on the C side == the vectorised psi argument of the port."""
    portref.shim_set_psi.argtypes = [ctypes.c_double]
    name = "ff_deformed_ring_polymer__bmb_"
    ref = c_reference(portref, name)
    spec = ports.get(name)
    p = np.zeros(50)
    p[:6] = [30.0, 8.0, 0.55, 2.0, 20.0, 75.0]
    q = np.logspace(-2, 0.5, 40)
    psis = np.linspace(0.1, 3.0, 5)
    grid = spec(q[None, :], p, psi=psis[:, None])
    try:
        for row, ps in zip(grid, psis):
            portref.shim_set_psi(float(ps))
            np.testing.assert_allclose(row, ref(q, p), rtol=1e-12)
    finally:
        portref.shim_clear_psi()


# --------------------------------------------------------------------------------------------- closed forms
def test_ggc_nu_half_is_debye():
    beta = np.logspace(-12, 8, 300)
    debye = np.where(
        beta < 1e-4, 1 - beta / 3 + beta**2 / 12, 2 * (np.expm1(-beta) + beta) / beta**2
    )
    np.testing.assert_allclose(dp.ggc_iso(beta, 0.5), debye, rtol=1e-12)


def test_ring_bmb_gaussian_ring_is_dawson():
    """ν = 1/2 (ε = 0): P = 2 D(√μ/2)/√μ with Dawson's integral D and μ = 2β (K_0 = 1/12)."""
    beta = np.logspace(-10, 10, 200)
    mu = 2.0 * beta
    exact = 2.0 * special.dawsn(np.sqrt(mu) / 2) / np.sqrt(mu)
    np.testing.assert_allclose(dp.ring_bmb_iso(beta, 0.5), exact, rtol=1e-12)
    assert dp.ring_bmb_iso(np.array([0.0]), 0.5)[0] == pytest.approx(1.0, abs=1e-14)


@pytest.mark.parametrize("nu", [0.26, 0.4, 0.5, 0.6, 0.9])
def test_ring_large_q_scaling(nu):
    """Large q: P -> 2 Γ(1+1/a) μ^{-1/a} (end-point layer, a = 2ν); BZ has s(x) ~ x^a too."""
    a = 2 * nu
    beta = np.array([1e12, 1e14])
    eps = a - 1
    k = math.exp(
        special.gammaln(2 + eps)
        + special.gammaln(3 + eps)
        - special.gammaln(5 + 2 * eps)
    )
    mu = beta / (6 * k)
    asym = 2 * math.gamma(1 + 1 / a) * mu ** (-1 / a)
    np.testing.assert_allclose(dp.ring_bmb_iso(beta, nu), asym, rtol=1e-4)
    assert np.all(np.isfinite(dp.ring_bz_iso(beta, nu))) and np.all(
        dp.ring_bz_iso(beta, nu) > 0
    )


def test_deformed_isotropic_limits():
    q = np.logspace(-3, 1, 50)
    p = np.zeros(50)
    p[:6] = [12.0, 12.0, 0.45, 3.0, 33.0, 71.0]  # Rg_ll = Rg_perp: no ψ dependence
    iso = np.zeros(50)
    iso[[0, 2, 3]] = [12.0, 0.45, 3.0]
    np.testing.assert_allclose(
        dp.ff_deformed_ring_polymer__bz_(q, p), dp.ff_ringpolymerBZ(q, iso), rtol=1e-13
    )
    np.testing.assert_allclose(
        dp.ff_deformed_ring_polymer__bmb_(q, p),
        dp.ff_ringpolymerBMB(q, iso),
        rtol=1e-13,
    )
    # ψ = θ0: the stretching direction, Rg_ll counts
    p[:6] = [20.0, 5.0, 0.45, 3.0, 40.0, 40.0]
    iso[0] = 20.0
    np.testing.assert_allclose(
        dp.ff_deformed_ring_polymer__bmb_(q, p),
        dp.ff_ringpolymerBMB(q, iso),
        rtol=1e-13,
    )
    p[5] = 130.0  # perpendicular: Rg_perp
    iso[0] = 5.0
    np.testing.assert_allclose(
        dp.ff_deformed_ring_polymer__bmb_(q, p),
        dp.ff_ringpolymerBMB(q, iso),
        rtol=1e-12,
    )


def test_invalid_parameters_raise():
    p = np.zeros(50)
    p[:6] = [10.0, 10.0, 0.0, 1.0, 0.0, 0.0]
    with pytest.raises(ValueError):
        dp.ff_deformed_generalized_gaussian_chain(np.array([0.1]), p)


def test_tanh_sinh_nodes():
    x, xc, _ = tanh_sinh_01()
    np.testing.assert_allclose(x + xc, 1.0, rtol=0, atol=1e-15)
    assert integrate_01(lambda x, xc: np.ones_like(x)) == pytest.approx(1.0, rel=1e-14)
    assert integrate_01(lambda x, xc: x ** (-0.5)) == pytest.approx(
        2.0, rel=1e-12
    )  # end-point singularity
    assert integrate_01(lambda x, xc: np.log(xc)) == pytest.approx(-1.0, rel=1e-12)


# --------------------------------------------------------------------------------------------- registry
def test_enable_and_disable_ports():
    name = "ff_ringpolymerBZ"
    ports.enable([name])
    try:
        f = functions.SASfitFunction(name)
        assert f.language == "python"
        q = np.logspace(-2, 0, 10)
        p = f.parameter_vector()
        p[[0, 2, 3]] = [10.0, 0.5, 2.0]
        np.testing.assert_allclose(f.evaluate_p(q, p), dp.ff_ringpolymerBZ(q, p))
        assert name in ports.enabled()
    finally:
        ports.disable()
    assert name not in functions._PYTHON_FUNCTIONS and not ports.enabled()


def test_language_c_skips_python_functions():
    functions.register_python_function(
        "ff_only_python_xyz", lambda x, p: 1.0, parameters=["a"]
    )
    try:
        assert functions.SASfitFunction("ff_only_python_xyz").language == "python"
        with pytest.raises(KeyError):
            functions.SASfitFunction("ff_only_python_xyz", language="c")
    finally:
        functions.unregister_python_function("ff_only_python_xyz")


def test_compare_and_report_with_own_reference():
    name = "ff_deformed_generalized_gaussian_chain"
    exact = ports.get(name)
    r = check.compare(name, reference=lambda q, p: exact(q, p) * (1 + 1e-9), n_sets=3)
    assert r.passed and 0.5e-9 < r.max_rel < 2e-9 and r.psi_consistent
    bad = check.compare(name, reference=lambda q, p: exact(q, p) * 1.01, n_sets=2)
    assert not bad.passed
    zero = check.compare(name, reference=lambda q, p: np.zeros_like(q), n_sets=1)
    assert (
        zero.n_ref_zero == check.DEFAULT_Q.size
        and not zero.passed
        and zero.zero_at["q"] == check.DEFAULT_Q[0]
    )
    half = check.compare(
        name, reference=lambda q, p: np.where(q > 1, 0.0, exact(q, p)), n_sets=2
    )
    assert (
        half.max_rel < 1e-14
        and half.n_ref_zero > 0
        and not half.passed
        and half.zero_at["q"] > 1
    )
    assert (
        half.x_label == "q Rg"
        and half.x_first_bad > 1.0
        and "| OK |" in check.report([half])
    )
    assert r.x_first_bad is None
    table = check.report([r, bad])
    assert "| " + name + " | PASS" in table and "FAIL" in table


# --------------------------------------------------------------------------------------------- networks, reptation, shear flow
_NEW = [
    "ff_we__deformed_polym__netw_",
    "ff_hsh__deformed_polym__netw_",
    "ff_we__deformed_ringpolym__netw_",
    "ff_hsh__deformed_ringpolym__netw_",
    "ff_reptating_chain",
    "ff_polymer_under_shearflow",
    "ff_polymer_under_shearflow1",
    "ff_polymer_under_shearflow2",
    "ff_ring_under_shearflow",
    "ff_ring_under_shearflow1",
    "ff_ring_under_shearflow2",
    "ff_shearflow_lin",
    "ff_shearflow_lin_g1",
    "ff_shearflow_lin_g2",
]


@pytest.mark.parametrize("name", _NEW)
def test_new_ports_match_c_source(portref, name):
    """Against the C sources as fixed on 2026-11-17 (normal mode of the ports)."""
    q = np.logspace(-1.5, 1, 40)
    r = check.compare(name, reference=c_reference(portref, name), q=q, n_sets=3, seed=7)
    assert r.passed, str(r)


def test_two_stage_is_exact():
    from scipy.integrate import quad

    from pysasfit.studio.ports.shearflow import two_stage

    for b, c, f in [
        (1e-9, 3e-9, 0.3),
        (1e-3, 2e-3, 0.5),
        (0.5, 2.0, 0.3),
        (40.0, 7.0, 0.8),
        (1e4, 1e3, 0.1),
    ]:

        def g(x, b=b, c=c, f=f):
            return b * x if x < f else b * f + c * (x - f)

        ref = quad(
            lambda x: 2 * (1 - x) * np.exp(-g(x)),
            0,
            1,
            points=[f],
            epsabs=0,
            epsrel=1e-13,
            limit=200,
        )[0]
        assert two_stage(b, c, f) == pytest.approx(ref, rel=1e-12)
    assert two_stage(0.0, 0.0, 0.4) == pytest.approx(1.0, abs=1e-15)


def test_debye_small_x():
    mp = pytest.importorskip("mpmath")
    from pysasfit.studio.ports.reptation import debye

    mp.mp.dps = 50
    x = np.logspace(-12, 2, 120)
    exact = np.array(
        [float(2 * (mp.mpf(v) - 1 + mp.exp(-mp.mpf(v))) / mp.mpf(v) ** 2) for v in x]
    )
    np.testing.assert_allclose(debye(x), exact, rtol=2e-15)


def test_reptating_chain_unstretched_is_debye():
    from pysasfit.studio.ports.reptation import debye

    q = np.logspace(-3, 1, 50)
    p = np.zeros(50)
    p[:6] = [2.0, 10.0, 1.0, 0.5, 30.0, 70.0]
    np.testing.assert_allclose(
        ports.get("ff_reptating_chain")(q, p), 2.0 * debye((q * 10.0) ** 2), rtol=1e-14
    )


def test_sector_average_conventions():
    """Port: mean over theta_0 ± Delta/2 with I0; c_compat reproduces the C normalisations."""
    q = np.logspace(-2, 0, 20)
    spec = ports.get("ff_polymer_under_shearflow")
    p = np.zeros(50)
    p[:7] = [20.0, 2.0, 0.5, 3.0, 10.0, 40.0, 0.0]
    single = spec(q, p)
    p[6] = 1e-6  # a tiny sector gives the value without averaging
    np.testing.assert_allclose(spec(q, p), single, rtol=1e-10)
    np.testing.assert_allclose(
        spec.intensity(q, p, None, c_compat=True), 0.5 * single, rtol=1e-10
    )
    p[6] = 0.0
    assert np.all(np.isinf(spec.intensity(q, p, None, c_compat=True)))  # C: 1/0
    ring2 = ports.get("ff_ring_under_shearflow2")
    p2 = np.zeros(50)
    p2[:10] = [20.0, 0.3, 1.0, 2.0, 0.5, 1.5, 2.0, 10.0, 40.0, 30.0]
    ratio = ring2.intensity(q, p2, None, c_compat=True) / ring2(q, p2)
    np.testing.assert_allclose(
        ratio, np.radians(30.0), rtol=1e-12
    )  # C: not divided by the sector width


@pytest.mark.parametrize(
    "name", ["ff_polymer_under_shearflow1", "ff_ring_under_shearflow1"]
)
def test_shearflow1_forward_scattering_is_i0(name):
    spec = ports.get(name)
    p = [10.0, 3.0, 30.0, 1.5, 0.5, 0.3]
    assert spec(np.array([1e-8]), p)[0] == pytest.approx(3.0, rel=1e-10)
    # old C (before 2026-11-17): no I0, ring with P(0) = 1/2
    old = spec.intensity(np.array([1e-8]), p, None, c_compat=True)[0]
    assert old == pytest.approx(0.5 if "ring" in name else 1.0, rel=1e-10)


# --------------------------------------------------------------------------- ff_py_* aliases and menu tree
def test_aliases_are_python_functions_next_to_c():
    from pysasfit.studio import groups

    names = ports.register_aliases()
    assert len(names) == len(ports.names())
    assert "ff_py_sheared_cylinders__legendre_" in names
    f = functions.SASfitFunction("ff_py_reptating_chain")
    assert f.language == "python" and f.info.collection == ports.PORTS_COLLECTION
    assert ports.get("ff_py_reptating_chain") is ports.get("ff_reptating_chain")
    assert ports.original("ff_py_reptating_chain") == "ff_reptating_chain"
    q = np.array([0.01, 0.1])
    p = f.parameter_vector([1.0, 10.0, 1.5, 0.1, 30.0, 60.0])
    np.testing.assert_allclose(
        f.evaluate_p(q, p), ports.get("ff_reptating_chain")(q, p[:6])
    )
    entries = [functions._PYTHON_FUNCTIONS[n][0] for n in names]
    top = groups.build_menu(entries, ("ff",), groups.load_groups())
    assert [c.label for c in top.children] == ["py plugin ports"]
    # peaks_py_ribbon_w_sd sits in the peaks menu, like the C peaks_ribbon_w_sd
    assert top.children[0].count() == sum(n.startswith("ff_") for n in names)
