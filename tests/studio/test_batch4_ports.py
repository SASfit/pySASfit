"""Batch 4 of the isotropic ports (2026-11-23): fuzzysphere, gen_exp_sphere, stochastic_models.

C comparisons of fuzzysphere and gen_exp_sphere run in test_isotropic_ports.py; here the stochastic models against
the (fixed) C code on a small grid, and checks independent of C: forward scattering, mpmath, γ(0) = 1.
"""

from __future__ import annotations

import ctypes
import importlib.util
import math
import os

import mpmath as mp
import numpy as np
import pytest
from scipy import integrate

from pysasfit.studio import ports
from pysasfit.studio.ports import check, fuzzy, genexp, stochastic

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
    spec = importlib.util.spec_from_file_location(
        "port_reference_build_b4", os.path.join(HERE, "port_reference", "build.py")
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    lib = mod.build(str(tmp_path_factory.mktemp("portref_b4")))
    if lib is None:
        pytest.skip("no C compiler or GSL for the C reference of the ports")
    return ctypes.CDLL(lib)


def _ref(lib, name, suffix=""):
    f = getattr(lib, "sasfit_" + name + suffix)
    f.restype = ctypes.c_double
    f.argtypes = [ctypes.c_double, ctypes.POINTER(_Param)]

    def ref(q, p):
        s = _Param()
        s.p[: len(p)] = [float(v) for v in p]
        out = np.array([f(float(x), ctypes.byref(s)) for x in q])
        if s.errStatus:
            raise RuntimeError(s.errStr.decode())
        return out

    return ref


# ------------------------------------------------------------------------------------------- stochastic vs C
_ST_C = [
    d["name"]
    for d in stochastic.SPECS
    if d.get("c_reference", True) and d["name"].endswith(("_1", "_2", "_4", "_5", "model"))
]


@pytest.mark.parametrize("name", _ST_C)
def test_stochastic_against_c(portref, name):
    r = check.compare(
        name, reference=_ref(portref, name), q=np.logspace(-2.5, 0.5, 14), n_sets=1, seed=5
    )
    assert r.passed, str(r)


def test_stochastic_against_scipy_quadrature():
    """I(q) = ∫ 4π r γ(r) sin(qr)/q dr with QAWO: independent of the Gauss-Legendre panels of the port."""
    p = [1.3, 0.3, 40.0, 12.0, 3.0]
    for kind, mode in ((2, "one"), (3, "two"), (5, "inter"), (4, "one")):
        gam = stochastic._crw_gamma(kind, mode, p[1], p[2], p[3], p[4])
        f = stochastic._crw(kind, mode)
        for q in (0.02, 0.3, 2.0):
            v, _ = integrate.quad(
                lambda r, gam=gam, q=q: 4 * math.pi * r * gam(np.array([r]))[0] / q,
                0,
                45 * p[3],
                weight="sin",
                wvar=q,
                limit=2000,
            )
            assert f(np.array([q]), p)[0] == pytest.approx(p[0] * v, rel=1e-8)


@pytest.mark.parametrize("mode", ["one", "two", "inter"])
@pytest.mark.parametrize("kind", [1, 2, 3, 4, 5])
def test_crw_gamma_is_normalised(kind, mode):
    for fp in (0.15, 0.5, 0.8):
        gam = stochastic._crw_gamma(kind, mode, fp, 50.0, 10.0, 2.5)
        g = gam(np.array([0.0, 1e-9, 2000.0]))
        assert g[0] == pytest.approx(1.0, abs=1e-12)
        assert g[1] == pytest.approx(1.0, abs=1e-6)
        assert abs(g[2]) < 1e-12


def test_level_cut_normalisation_against_mpmath():
    """∫_0^{π/2}: φ(1-φ) for one cut (1/2π) and two cuts (1/π) - the normalisation of γ."""
    mp.mp.dps = 30
    for fp in (0.2, 0.5, 0.7):
        a1 = math.sqrt(2) * float(mp.erfinv(1 - 2 * fp))
        one = mp.quad(lambda t, a1=a1: mp.exp(-(a1**2) / (1 + mp.sin(t))), [0, mp.pi / 2]) / (2 * mp.pi)
        assert float(one) == pytest.approx(fp * (1 - fp), rel=1e-12)
        a2 = math.sqrt(2) * float(mp.erfinv(fp))
        two = mp.quad(
            lambda t, a2=a2: mp.exp(-(a2**2) / (1 + mp.sin(t)))
            - (mp.exp(-(a2**2) / (1 - mp.sin(t))) if mp.sin(t) < 1 else 0),
            [0, mp.pi / 2],
        ) / mp.pi
        assert float(two) == pytest.approx(fp * (1 - fp), rel=1e-12)
        x = stochastic._x_cov(np.array([1.0]), a1, False)[0]
        assert x == pytest.approx(fp * (1 - fp), rel=1e-12)


def test_boolean_gamma_limits_and_covariogram():
    fp, mu, sig = 0.4, 10.0, 0.3
    k0 = stochastic.theta_k(np.array([0.0]), fp, mu, sig)[0]
    assert k0 == pytest.approx(-math.log(fp), rel=1e-12)
    # ⟨V k(r/R)⟩/⟨V⟩ with the sphere covariogram k(x) = 1 - 3x/4 + x³/16 (x = r/R < 2), log-normal R
    r = 7.0
    num = integrate.quad(
        lambda R: (
            math.exp(-0.5 * (math.log(R / mu) / sig) ** 2) / (R * sig * math.sqrt(2 * math.pi))
            * R**3 * (1 - 0.75 * r / R + (r / R) ** 3 / 16 if r < 2 * R else 0.0)
        ),
        r / 2, mu * math.exp(12 * sig), limit=200,
    )[0]  # fmt: skip
    den = mu**3 * math.exp(4.5 * sig * sig)
    assert stochastic.theta_k(np.array([r]), fp, mu, sig)[0] == pytest.approx(
        -math.log(fp) * num / den, rel=1e-9
    )


# ------------------------------------------------------------------------------------------------ fuzzy shells
def _forward(prof, p, rmax):
    return 4 * math.pi * integrate.quad(lambda r: prof(np.array([r]), p)[0] * r * r, 0, rmax, limit=400, epsabs=0)[0]


@pytest.mark.parametrize(
    "name,prof,p,rmax",
    [
        ("ff_linshell", fuzzy.profile_linshell, [12.0, 6.0, 1.3, 0.9, 0.1, 0.8, 0.2], 18.0),
        ("ff_linshell2", fuzzy.profile_linshell2, [30.0, 6.0, 1.3, 0.9, 0.1, 0.8, 0.2], 30.0),
        ("ff_expshell", fuzzy.profile_expshell, [12.0, 6.0, 1.3, 0.9, 0.1, 0.8, -2.0, 0.2], 18.0),
        ("ff_expshell", fuzzy.profile_expshell, [12.0, 6.0, 1.3, 0.9, 0.1, 0.8, 1.5, 0.2], 18.0),
        ("ff_coreshellmicrogel", fuzzy.profile_coreshellmicrogel, [15.0, 2.0, 10.0, 2.0, 3.0, 3.0, 1.0, 0.6, 0.05], 50.0),
    ],
)  # fmt: skip
def test_shell_forward_scattering_and_small_q(name, prof, p, rmax):
    """I(q -> 0) = (∫ Δη dV)², stable down to q = 1e-8 (the C closed forms lost all digits there)."""
    es = p[-1]
    f0 = _forward(lambda r, pp: prof(r, pp) - es, p, rmax)
    spec = ports.get(name)
    q = np.array([1e-8, 1e-7, 1e-6])
    np.testing.assert_allclose(spec(q, p), f0 * f0, rtol=1e-9)


def test_linshell2_is_linshell_with_outer_radius(portref):
    """R_tot - dR as core radius (the C macro R1 = RTOT-DR was not parenthesised: q*R1 = q*RTOT - DR)."""
    q = np.logspace(-3, 0.5, 30)
    p2 = [30.0, 8.0, 1.3, 0.9, 0.1, 0.8, 0.2]
    p1 = [22.0] + p2[1:]
    np.testing.assert_allclose(ports.get("ff_linshell2")(q, p2), ports.get("ff_linshell")(q, p1), rtol=1e-13)
    np.testing.assert_allclose(
        _ref(portref, "ff_linshell2")(q, p2), _ref(portref, "ff_linshell")(q, p1), rtol=1e-13
    )


def test_microgel_against_mpmath():
    mp.mp.dps = 30
    p = [21.6, 2.97, 19.66, 1.22, 5.54, 2.68, 1.03, 0.67, 0.02]
    rcore, sc, rshin, ssh, rout, so = fuzzy._mg_radii(p)
    pts = sorted({0.0, rcore - sc, rcore, rcore + sc, rshin - ssh, rshin, rshin + ssh, rout - so, rout, rout + so})
    for q in (0.05, 0.5, 2.9, 10.0):
        f = lambda r, q=q: (fuzzy.profile_coreshellmicrogel(np.array([float(r)]), p)[0] - p[8]) * 4 * mp.pi * r * mp.sin(q * r) / q
        ex = float(mp.quad(f, pts) ** 2)
        assert fuzzy.coreshellmicrogel(np.array([q]), p)[0] == pytest.approx(ex, rel=1e-11)


@pytest.mark.parametrize("name", ["ff_fuzzysphere", "ff_linshell", "ff_linshell2", "ff_expshell", "ff_coreshellmicrogel"])
def test_volumes_against_c(portref, name):
    spec = ports.get(name)
    f = getattr(portref, "sasfit_" + name + "_v")
    f.restype = ctypes.c_double
    f.argtypes = [ctypes.c_double, ctypes.POINTER(_Param), ctypes.c_int]
    for p in ports.sample_parameters(spec, 3, seed=4):
        s = _Param()
        s.p[: len(p)] = [float(v) for v in p]
        assert spec.volume(np.array([1.0]), p, 0)[0] == pytest.approx(f(1.0, ctypes.byref(s), 0), rel=1e-12)


# ------------------------------------------------------------------------------------------- gen. exp sphere
def test_gen_gauss_sphere_against_mpmath():
    mp.mp.dps = 40
    for rg, a in ((5.0, -2.5), (5.0, 0.0), (8.0, 1.7), (3.0, 6.0)):
        b = 3 + a
        for q in (1e-4, 0.3, 2.0, 8.0):
            ex = float((4 * mp.pi * rg**3 * mp.hyp1f1(mp.mpf(b) / 2, 1.5, -((q * rg) ** 2) / (2 * b))) ** 2)
            got = genexp.gen_gauss_sphere(np.array([q]), [rg, a, 1.0])[0]
            assert got == pytest.approx(ex, rel=1e-12, abs=1e-300)
