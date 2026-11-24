"""Batch 8 of the isotropic ports (2026-11-24): triax_ellip_shell, thin_objects (ellsh/cylsh with size distribution),
hyperbranched_polymer, helix and pcs_fuzzy_cyl.

C comparison against the (fixed) C code with σ = 0 where the C code integrates a log-normal distribution (it
truncates the range with find_LogNorm_int_range: deviations 1e-6 ... 1e-4 for σ > 0), independent references
(scipy quadrature, mpmath, explicit geometry) and forward scattering I(0) = (Δη V)²."""

from __future__ import annotations

import ctypes
import importlib.util
import math
import os

import mpmath as mp
import numpy as np
import pytest
from scipy import integrate, special

from pysasfit.studio.ports import (
    check,
    fuzzy_cyl,
    get,
    helix,
    hyperbranched,
    sample_parameters,
    thin_objects,
    triax_ellip_shell,
)

HERE = os.path.dirname(os.path.abspath(__file__))


class _Param(ctypes.Structure):
    _fields_ = [
        ("p", ctypes.c_double * 50),
        ("errStatus", ctypes.c_int),
        ("errStr", ctypes.c_char * 256),
        ("kernelSelector", ctypes.c_int),
    ]


@pytest.fixture(scope="module")
def portref(tmp_path_factory):
    spec = importlib.util.spec_from_file_location("port_reference_build_b8", os.path.join(HERE, "port_reference", "build.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    lib = mod.build(str(tmp_path_factory.mktemp("portref_b8")))
    if lib is None:
        pytest.skip("no C compiler or GSL for the C reference of the ports")
    return ctypes.CDLL(lib)


def _ref(lib, name, kind=""):
    f = getattr(lib, "sasfit_" + name + kind)
    f.restype = ctypes.c_double
    if kind == "_v":
        f.argtypes = [ctypes.c_double, ctypes.POINTER(_Param), ctypes.c_int]

        def vol(x, p, dist):
            s = _Param()
            s.p[: len(p)] = [float(v) for v in p[:50]]
            return f(float(x), ctypes.byref(s), int(dist))

        return vol
    f.argtypes = [ctypes.c_double, ctypes.POINTER(_Param)]

    def ref(q, p):
        out = []
        for x in q:
            s = _Param()
            s.p[: len(p)] = [float(v) for v in p[:50]]
            out.append(f(float(x), ctypes.byref(s)))
            if s.errStatus:
                raise RuntimeError(s.errStr.decode())
        return np.array(out)

    return ref


def _vec(p):
    v = np.zeros(50)
    v[: len(p)] = p
    return v


def _ln(x, x0, s):
    return math.exp(-0.5 * (math.log(x / x0) / s) ** 2) / (math.sqrt(2.0 * math.pi) * s * x)


# ------------------------------------------------------------------ triax_ellip_shell
_TRIAX = [d["name"] for d in triax_ellip_shell.SPECS if d["name"] != "ff_triax_ellip_shell_syl"]


@pytest.mark.parametrize("name", _TRIAX)
def test_triax_against_c_sigma0(portref, name):
    spec = get(name)
    q = np.logspace(-3, 0.5, 10)
    for p in sample_parameters(spec, 2, seed=5):
        p[4] = 0.0
        np.testing.assert_allclose(spec.intensity(q, p), _ref(portref, name)(q, p), rtol=1e-10)
        np.testing.assert_allclose(spec.amplitude(q, p), _ref(portref, name, "_f")(q, p), rtol=1e-10,
                                   atol=1e-12 * abs(spec.amplitude(np.array([0.0]), p)[0]))
        assert spec.volume(np.array(p[0]), p, 0) == pytest.approx(_ref(portref, name, "_v")(p[0], p, 0), rel=1e-13)


def test_triax_syl_against_c(portref):
    r = check.compare("ff_triax_ellip_shell_syl", reference=_ref(portref, "ff_triax_ellip_shell_syl"),
                      q=np.logspace(-3, 0.5, 10), n_sets=3, seed=7)
    assert r.passed, str(r)


@pytest.mark.parametrize("name", ["ff_triax_ellip_shell", "ff_triax_ellip_shell_2t"])
def test_triax_sigma_loosely_against_c(portref, name):
    q = np.array([1e-3, 0.1, 0.6])
    p = _vec([10, 6, 4, 2, 0.1, 1.0, 0.4, 0.1])
    np.testing.assert_allclose(get(name).intensity(q, p), _ref(portref, name)(q, p), rtol=2e-4)


@pytest.mark.parametrize("qi", [0.05, 0.6])
def test_triax_sigma_against_nquad(qi):
    a, b, c, t0, s, ec, es, ev = 10.0, 6.0, 4.0, 2.0, 0.12, 1.0, 0.4, 0.1

    def phi(u):
        return 3.0 * (math.sin(u) - u * math.cos(u)) / u**3

    def F(nu, x, m):  # _3t: all lengths scaled by ν
        A, B, C, T = a * nu, b * nu, c * nu, t0 * nu
        cph, sph = math.cos(x * math.pi / 2) ** 2, math.sin(x * math.pi / 2) ** 2
        rc = math.sqrt((A * A * cph + B * B * sph) * (1 - m * m) + (C * m) ** 2)
        rt = math.sqrt(((A + T) ** 2 * cph + (B + T) ** 2 * sph) * (1 - m * m) + ((C + T) * m) ** 2)
        f = (ec - es) * 4 / 3 * math.pi * A * B * C * phi(qi * rc) + (es - ev) * 4 / 3 * math.pi * (A + T) * (B + T) * (
            C + T) * phi(qi * rt)
        return f * f * _ln(nu, 1.0, s)

    ref = integrate.nquad(F, [[math.exp(-8 * s), math.exp(8 * s + 6 * s * s)], [0, 1], [0, 1]],
                          opts={"epsrel": 1e-10, "epsabs": 0, "limit": 200})[0]
    got = triax_ellip_shell.triax_ellip_shell_3t(np.array([qi]), _vec([a, b, c, t0, s, ec, es, ev]))[0]
    assert got == pytest.approx(ref, rel=1e-9)


def test_triax_forward_scattering():
    a, b, c, t, ec, es, ev = 10.0, 6.0, 4.0, 2.0, 1.0, 0.4, 0.1
    i0 = triax_ellip_shell.triax_ellip_shell(np.array([0.0]), _vec([a, b, c, t, 0, ec, es, ev]))[0]
    v = 4 / 3 * math.pi
    assert i0 == pytest.approx(((ec - es) * v * a * b * c + (es - ev) * v * (a + t) * (b + t) * (c + t)) ** 2, rel=1e-13)


# ------------------------------------------------------------------ thin_objects: shells with size distribution
_THIN = ["ff_ellsh_sd_homoxs", "ff_ellsh_sd_bilayergauss", "ff_cylsh_sd_homoxs", "ff_cylsh_sd_bilayergauss"]


@pytest.mark.parametrize("name", _THIN)
def test_thin_shells_against_c(portref, name):
    r = check.compare(name, reference=_ref(portref, name), q=np.logspace(-3, 0.5, 16), n_sets=4, seed=3)
    assert r.passed, str(r)


@pytest.mark.parametrize("name", ["ff_ellsh_sd_homoxs", "ff_ellsh_sd_bilayergauss"])
def test_thin_ellsh_amplitude_and_volume_against_c(portref, name):
    spec = get(name)
    q = np.logspace(-3, 0.5, 8)
    for p in sample_parameters(spec, 2, seed=11):
        f = spec.amplitude(q, p)
        np.testing.assert_allclose(f, _ref(portref, name, "_f")(q, p), rtol=1e-9, atol=1e-12 * np.max(np.abs(f)))
        assert spec.volume(np.array(p[0]), p, 0) == pytest.approx(_ref(portref, name, "_v")(p[0], p, 0), rel=1e-12)


def test_pcs_amplitudes_against_c(portref):
    q = np.logspace(-3, 0.5, 8)
    for name, p in (("ff_pcs_bilayergauss", [0.4, 1.0, 0.5, -0.5, 4.0, 0.0]),
                    ("ff_pcs_homogeneousplate", [4.0, 0.0, 0, 0, 1.0, 0.2])):
        f = get(name).amplitude(q, _vec(p))
        np.testing.assert_allclose(f, _ref(portref, name, "_f")(q, _vec(p)), rtol=1e-10, atol=1e-13 * np.max(np.abs(f)))


def test_thin_ellsh_sigma_against_quad():
    """Size average of the σ_R = 0 function by scipy quad = the port's log-normal quadrature."""
    p0 = _vec([4.0, 0.0, 50.0, 0.0, 1.5, 0, 1.0, 0.0])
    s = 0.15
    for qi in (0.02, 0.3):
        def f(R):
            p = p0.copy()
            p[2] = R
            return _ln(R, 50.0, s) * thin_objects.ff_ellsh_sd_homoxs(np.array([qi]), p)[0]

        ref = integrate.quad(f, 50 * math.exp(-9 * s), 50 * math.exp(9 * s), epsabs=0, epsrel=1e-12, limit=400)[0]
        p = p0.copy()
        p[3] = s
        assert thin_objects.ff_ellsh_sd_homoxs(np.array([qi]), p)[0] == pytest.approx(ref, rel=1e-9)


# ------------------------------------------------------------------ hyperbranched_polymer
@pytest.mark.parametrize("name", [d["name"] for d in hyperbranched.SPECS])
def test_hyperbranched_against_c(portref, name):
    r = check.compare(name, reference=_ref(portref, name), q=np.logspace(-3, 1, 24), n_sets=6, seed=5)
    assert r.passed, str(r)


@pytest.mark.parametrize("g", [1, 4, 9])
def test_dendrimer_small_q_against_mpmath(g):
    mp.mp.dps = 50
    terms = hyperbranched._DENDRIMER[g]
    a = 3.0
    for qi in (1e-4, 1e-2, 0.3):
        u = mp.mpf(qi * a) ** 2
        ref = 2 / u**2 * mp.fsum(mp.mpf(c) * (mp.exp(-mp.mpf(n) / d * u) - 1 + mp.mpf(n) / d * u) for c, n, d in terms)
        got = hyperbranched.ideal_dendrimer(np.array([qi]), _vec([a, 0, g, 1.0]))[0]
        assert got == pytest.approx(float(ref), rel=1e-13)


def test_combs_limits():
    q = np.array([0.0, 1e-6, 0.1, 1.0])
    for f in (hyperbranched.comb_hetero, hyperbranched.comb_random, hyperbranched.comb_regular):
        for K in (0.0, 0.5, 1.0):
            v = f(q, _vec([5.0, 8.0, K, 2.0]))
            assert np.all(np.isfinite(v)) and v[0] == pytest.approx(2.0) and v[1] == pytest.approx(2.0, rel=1e-9)
    # K = 1: linear chain (Debye)
    x2 = (0.3 * 5.0) ** 2
    debye = 2 * (math.exp(-x2) - 1 + x2) / x2**2
    assert hyperbranched.comb_random(np.array([0.3]), _vec([5.0, 8.0, 1.0, 1.0]))[0] == pytest.approx(debye, rel=1e-12)


@pytest.mark.parametrize("typ, ex", [(2, 2.0), (3, 2.18)])
def test_f_funct_excluded_volume_against_quad(typ, ex):
    sig, f, al, i0 = 5.0, 3.0, 0.3, 2.0
    nu = 0.588
    a_s = al * (1 - f)
    for qi in (0.01, 0.2):
        e = 6.0 if typ == 2 else 6.2
        g = lambda n: (1 - a_s) * math.exp(-(1 - a_s) * n - (sig * qi) ** 2 * n ** (ex * nu) / e)  # noqa: E731
        ref = i0 * integrate.quad(g, 0, np.inf, epsabs=0, epsrel=1e-12, limit=400)[0]
        got = hyperbranched.f_funct_homo_polym(np.array([qi]), _vec([sig, f, al, i0, typ]))[0]
        assert got == pytest.approx(ref, rel=1e-8)


# ------------------------------------------------------------------ helix
@pytest.mark.parametrize("name", ["ff_round_helix", "ff_beads_helix", "ff_fan_helix", "ff_superhelices_straight"])
def test_helix_against_c(portref, name):
    r = check.compare(name, reference=_ref(portref, name), q=np.logspace(-3, 0.3, 16), n_sets=4, seed=5)
    assert r.passed, str(r)
    spec = get(name)
    for p in sample_parameters(spec, 2, seed=1):
        assert spec.volume(np.array(p[0]), p, 0) == pytest.approx(_ref(portref, name, "_v")(p[0], p, 0), rel=1e-12)


def test_coiled_superhelix_against_c(portref):
    p = _vec([3, 15, 0, 3, 0, 40, 1.0])
    q = np.array([0.0, 0.03, 0.15])
    np.testing.assert_allclose(helix.superhelices_coiled(q, p), _ref(portref, "ff_superhelices_coiled")(q, p), rtol=1e-9)
    L = helix.superhelices_coiled_v(np.array(3.0), p, 0)
    assert L == pytest.approx(_ref(portref, "ff_superhelices_coiled", "_v")(3.0, p, 0), rel=1e-12)
    assert helix.superhelices_coiled(np.array([0.0]), p)[0] == pytest.approx(L * L, rel=1e-12)


def test_coiled_superhelix_N0_is_plain_helix():
    """N = 0, R1 = 0: the coiled helix is the plain helix of radius R2 (superhelices_straight with R1 = R2)."""
    R2, P, T = 15.0, 40.0, 1.5
    q = np.array([0.01, 0.1, 0.3])
    a = helix.superhelices_coiled(q, _vec([0, R2, 0, 0, 0, P, T]))
    b = helix.superhelices_straight(q, _vec([R2, 0, 0, 0, 0, P, P * T]))
    np.testing.assert_allclose(a, b, rtol=1e-10)


def test_helix_forward_scattering():
    p = _vec([3, 10, 4, 12, 60, 40, 300, 1.0, 0.8, 0.1])
    i0 = helix.round_helix(np.array([0.0]), p)[0]
    assert i0 == pytest.approx(((0.9 * 9 + 0.7 * 16) * math.pi * 300) ** 2, rel=1e-13)
    p = _vec([3, 20, 8, 0, 0, 40, 300, 1.0, 0, 0.1])
    assert helix.beads_helix(np.array([0.0]), p)[0] == pytest.approx((0.9 * helix.beads_helix_v(3.0, p, 0)) ** 2, rel=1e-13)
    p = _vec([20, 0.3, 60, 90, 0, 40, 300, 1.0, 0, 0.1])
    assert helix.fan_helix(np.array([0.0]), p)[0] == pytest.approx((0.9 * helix.fan_helix_v(20.0, p, 0)) ** 2, rel=1e-13)
    p = _vec([15, 0, 0, 0, 0, 50, 300])
    L = helix.superhelices_straight_v(15.0, p, 0)
    assert helix.superhelices_straight(np.array([0.0]), p)[0] == pytest.approx(L * L, rel=1e-13)


def test_thinrod_helix_series():
    for u in (1e-4, 9e-4, 1.1e-3, 0.5):
        si = special.sici(u)[0]
        ref = float(2 * mp.si(mp.mpf(u)) / u - mp.sinc(mp.mpf(u) / 2) ** 2) if u < 1e-2 else 2 * si / u - np.sinc(u / 2 / np.pi) ** 2
        assert helix.thinrod_helix(np.array([u]), 1.0)[0] == pytest.approx(ref, rel=1e-14)


# ------------------------------------------------------------------ pcs_fuzzy_cyl
_PCS = [d["name"] for d in fuzzy_cyl.SPECS]


@pytest.mark.parametrize("name", _PCS)
def test_pcs_fuzzy_cyl_against_c_sigma0(portref, name):
    spec = get(name)
    profile = "profile" in name
    x = np.linspace(0.0, 60.0, 25) if profile else np.logspace(-3, 0.5, 14)
    rtol, atol = 1e-11, 1e-14
    for p in sample_parameters(spec, 3, seed=9):
        if "sigma_R0" in spec.parameters:
            p[1] = 0.0
        i = spec.intensity(x, p)
        np.testing.assert_allclose(i, _ref(portref, name)(x, p), rtol=rtol, atol=atol * float(np.max(np.abs(i))))
        if spec.amplitude is not None:
            f = spec.amplitude(x, p)
            np.testing.assert_allclose(f, _ref(portref, name, "_f")(x, p), rtol=max(rtol, 1e-10),
                                       atol=max(atol, 1e-12) * np.max(np.abs(f)))
        if spec.volume is not None:
            assert spec.volume(np.array(p[0]), p, 0) == pytest.approx(_ref(portref, name, "_v")(p[0], p, 0), rel=1e-13)


@pytest.mark.parametrize("name, p", [("boucher_cyl", [20, 0.12, 0, 4.5, 0, 0, 0, 1.3]),
                                     ("boucher2_cyl", [20, 0.12, 0, 3.0, 0, 0, 0, 1.3]),
                                     ("fuzzy_cyl", [20, 0.12, 0, 3, 0, 0, 0, 1.2, 0.1]),
                                     ("linsh_cyl", [20, 0.12, 8, 0, 0, 1.0, 0.7, 0.2, 0.1])])
def test_pcs_fuzzy_cyl_sigma_against_quad(portref, name, p):
    p = _vec(p)
    f = getattr(fuzzy_cyl, name)
    for qi in (1e-3, 0.08, 0.5):
        def g(R):
            pp = p.copy()
            pp[0], pp[1] = R, 0.0
            return _ln(R, 20.0, p[1]) * f(np.array([qi]), pp)[0]

        ref = integrate.quad(g, 20 * math.exp(-9 * p[1]), 20 * math.exp(9 * p[1]), epsabs=0, epsrel=1e-12, limit=400)[0]
        assert f(np.array([qi]), p)[0] == pytest.approx(ref, rel=1e-9)
        assert _ref(portref, "ff_" + name)([qi], p)[0] == pytest.approx(ref, rel=1e-4)  # truncated range in C


@pytest.mark.parametrize("pair", [
    ("profile_boucher_cyl", "boucher_cyl_f", [20, 0, 0, 4.5, 0, 0, 0, 1.3], 0.0),
    ("profile_boucher2_cyl", "boucher2_cyl_f", [20, 0, 0, 5.0, 0, 0, 0, 1.3], 0.0),
    ("profile_fuzzy_cyl", "fuzzy_cyl_f", [20, 0, 0, 3, 0, 0, 0, 1.2, 0.1], 0.1),
    ("profile_linsh_cyl", "linsh_cyl_f", [20, 0, 8, 0, 0, 1.0, 0.7, 0.2, 0.1], 0.1),
    ("rc_exp_sh_cyl_profile", "rc_exp_sh_cyl_f", [20, 10, -2, 1, 1.0, 0.8, 0.2, 0.1], 0.1),
    ("rc_exp_sh_cyl_profile", "rc_exp_sh_cyl_f", [20, 10, 1.5, 0.5, 1.0, 0.8, 0.2, 0.1], 0.1),
])
def test_profiles_are_hankel_pairs_of_the_amplitudes(pair):
    prof, amp, p, eta_sol = pair
    p = _vec(p)
    rho = getattr(fuzzy_cyl, prof)
    F = getattr(fuzzy_cyl, amp)
    breaks = [20.0, 28.0, 30.0]
    for qi in (1e-3, 0.1, 0.4):
        g = lambda r: 2 * math.pi * r * special.j0(qi * r) * (float(rho(np.array([r]), p)[0]) - eta_sol)  # noqa: E731
        ref = sum(integrate.quad(g, a, b, epsabs=0, epsrel=1e-12, limit=400)[0]
                  for a, b in zip([0.0] + breaks, breaks + [90.0]))
        assert F(np.array([qi]), p)[0] == pytest.approx(ref, rel=1e-8, abs=1e-10 * abs(F(np.array([0.0]), p)[0]))


def test_fuzzy_profile_limits():
    p = _vec([20, 0, 0, 1e-9, 0, 0, 0, 1.2, 0.1])
    r = np.array([0.0, 10.0, 19.0, 21.0, 40.0])
    np.testing.assert_allclose(fuzzy_cyl.profile_fuzzy_cyl(r, p), [1.2, 1.2, 1.2, 0.1, 0.1], rtol=1e-12)
    # large σ: the convolution conserves the integral 2π ∫ (ρ - η_sol) r dr = Δη π R²
    p = _vec([20, 0, 0, 8, 0, 0, 0, 1.2, 0.1])
    tot = integrate.quad(lambda r: 2 * math.pi * r * (fuzzy_cyl.profile_fuzzy_cyl(np.array([r]), p)[0] - 0.1), 0, 150,
                         epsabs=0, epsrel=1e-11, limit=400)[0]
    assert tot == pytest.approx(1.1 * math.pi * 400, rel=1e-9)
