"""Batch 6 of the isotropic ports (2026-11-24): Koyama worm (polymer_semi_flexible) and the thin_objects functions
with a cross-section form factor: elliptical core-shell cylinders (rod, worms), the Pcs family, thin discs.

C comparison against the (fixed) C code without size distributions (the C code integrates log-normal
distributions only over SASfit's find_LogNorm_int_range, which drops ~1e-4 (up to 1e-3 at large q) of the weight); distributions
against scipy quadrature and loosely against C; limits and forward scattering."""

from __future__ import annotations

import ctypes
import importlib.util
import math
import os

import mpmath as mp
import numpy as np
import pytest
from scipy import integrate, special

from pysasfit.studio.ports import check, thin_objects, worms

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
    spec = importlib.util.spec_from_file_location("port_reference_build_b6", os.path.join(HERE, "port_reference", "build.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    lib = mod.build(str(tmp_path_factory.mktemp("portref_b6")))
    if lib is None:
        pytest.skip("no C compiler or GSL for the C reference of the ports")
    return ctypes.CDLL(lib)


def _ref(lib, name):
    f = getattr(lib, "sasfit_" + name)
    f.restype = ctypes.c_double
    f.argtypes = [ctypes.c_double, ctypes.POINTER(_Param)]

    def ref(q, p):
        out = []
        for x in q:
            s = _Param()
            s.p[: len(p)] = [float(v) for v in p]
            out.append(f(float(x), ctypes.byref(s)))
            if s.errStatus:
                raise RuntimeError(s.errStr.decode())
        return np.array(out)

    return ref


def _vec(p):
    v = np.zeros(50)
    v[: len(p)] = p
    return v


_NAMES = [d["name"] for d in thin_objects.SPECS] + ["ff_koyama_worm"]


@pytest.mark.parametrize("name", _NAMES)
def test_against_c(portref, name):
    r = check.compare(name, reference=_ref(portref, name), q=np.logspace(-3, 0.8, 30), n_sets=5, seed=6)
    assert r.passed, str(r)


_DIST = [
    ("ff_pcs_ellcylsh", [5, 0.15, 0.6, 2, 0.1, 0, 0, 1, 0.5, 0]),
    ("ff_rod_sd_ellcylsh", [5, 0.1, 1.3, 2, 0.1, 100, 0.2, 0, 1, 0.5, 0]),
    ("ff_worm_ps1_sd_ellcylsh", [5, 0.1, 1.3, 2, 20, 500, 0, 1, 0.5, 0]),
    ("ff_worm_kholodenko_sd_ellcylsh", [5, 0.1, 1.0, 2, 0.1, 20, 500, 0, 1, 0.5, 0]),
    ("ff_pcs_homogeneouscyl", [5, 0.2, 0, 0, 0, 0, 0, 1.2, 0, 0.3]),
    ("ff_pcs_homogeneousplate", [4, 0.2, 0, 0, 1, 0.2]),
    ("ff_pcs_twoinfinitelythinlayers", [4, 0.15]),
    ("ff_pcs_bilayergauss", [0.5, 1, 0.4, -0.6, 4, 0.1]),
    ("ff_pcs_layeredcentrosymmetricxs", [4, 0.1, 2, 0.15, 1, -0.3, 0.2]),
    ("ff_disc_sd_homoxs", [4, 0.1, 50, 0.2, 0, 0, 1, 0.2]),
    ("ff_disc_sd_bilayergauss", [4, 0.1, 50, 0.15, 0, 0, 0.4, -0.6, 0.5, 1]),
]


@pytest.mark.parametrize("name, p", _DIST)
def test_distributions_loosely_against_c(portref, name, p):
    q = np.logspace(-3, 0.5, 8)
    py = getattr(thin_objects, name)(q, _vec(p))
    np.testing.assert_allclose(py, _ref(portref, name)(q, p), rtol=2e-3)


# ------------------------------------------------------------------ independent references
def _lam(u):
    return 1.0 if u == 0 else 2.0 * special.j1(u) / u


def _gauss(t, s):
    return math.exp(-0.5 * (t / s) ** 2) / (math.sqrt(2 * math.pi) * s)


@pytest.mark.parametrize("q", [1e-3, 0.4, 1.5])
def test_pcs_ellcylsh_against_tplquad(q):
    R0, sR, eps, t0, st, ec, es, ev = 5, 0.15, 0.6, 2, 0.1, 1, 0.5, 0

    def f(u2, u1, al):
        R, T = R0 * math.exp(u1), t0 * math.exp(u2)
        ri = math.hypot(R * math.sin(al), R * eps * math.cos(al))
        ro = math.hypot((R + T) * math.sin(al), (R * eps + T) * math.cos(al))
        A = math.pi * R * R * eps * (ec - es) * _lam(q * ri) + math.pi * (R + T) * (R * eps + T) * (es - ev) * _lam(q * ro)
        return A * A * _gauss(u1, sR) * _gauss(u2, st) * 2 / math.pi

    ref, _ = integrate.tplquad(f, 0, math.pi / 2, -12 * sR, 12 * sR, -12 * st, 12 * st, epsabs=0, epsrel=1e-11)
    got = thin_objects.pcs_ellcylsh(np.array([q]), R0, sR, eps, t0, st, ec, es, ev)[0]
    assert got == pytest.approx(ref, rel=1e-9)


def test_pcs_ellcylsh_continuous_at_circle_and_forward():
    a = thin_objects.pcs_ellcylsh(np.array([0.3, 1.0]), 5, 0, 1.0, 2, 0, 1, 0.5, 0)
    b = thin_objects.pcs_ellcylsh(np.array([0.3, 1.0]), 5, 0, 1.0 + 1e-9, 2, 0, 1, 0.5, 0)
    np.testing.assert_allclose(a, b, rtol=1e-8)
    eps = 0.7
    i0 = thin_objects.pcs_ellcylsh(np.array([0.0]), 5, 0, eps, 2, 0, 1, 0.5, 0)[0]
    assert i0 == pytest.approx((math.pi * 25 * eps * 0.5 + math.pi * 7 * (5 * eps + 2) * 0.5) ** 2, rel=1e-13)
    w = thin_objects.ff_worm_ps3_sd_ellcylsh(np.array([0.0]), _vec([5, 0, eps, 2, 20, 500, 0, 1, 0.5, 0]))[0]
    assert w == pytest.approx(i0 * 500 ** 2, rel=1e-13)


def _lognorm_quad(f, x0, s, k):
    v, _ = integrate.quad(lambda t: _gauss(t, s) * f(x0 * math.exp(t)), k * s * s - 12 * s, k * s * s + 12 * s,
                          epsabs=0, epsrel=1e-12, limit=2000)
    return v


@pytest.mark.parametrize("q", [1e-3, 0.05, 0.4])
def test_pprimes_against_quad(q):
    def rod(x):
        u = q * x
        return x * x * (2 * special.sici(u)[0] / u - (math.sin(u / 2) / (u / 2)) ** 2)

    assert thin_objects.rod_pprime(np.array([q]), 100, 0.2)[0] == pytest.approx(_lognorm_quad(rod, 100, 0.2, 2), rel=1e-10)

    def disc(x):
        u = q * x
        return 2 * (math.pi * x / q) ** 2 * (1 - _lam(2 * u)) if u > 1e-2 else (math.pi * x * x) ** 2 * (1 - u * u / 6 + u ** 4 / 72)

    assert thin_objects.disc_pprime(np.array([q]), 50, 0.2)[0] == pytest.approx(_lognorm_quad(disc, 50, 0.2, 4), rel=1e-10)


def test_disc_pprime_small_q():
    # the closed form cancels; mpmath reference of 2 (pi R/q)^2 (1 - 2 J1(2u)/(2u))
    mp.mp.dps = 40
    for u in (1e-8, 1e-4, 0.05, 0.0999, 0.1001, 3.0):
        R = 10.0
        q = u / R
        ref = 2 * (mp.pi * R / q) ** 2 * (1 - 2 * mp.besselj(1, 2 * mp.mpf(u)) / (2 * mp.mpf(u)))
        assert thin_objects.disc_pprime(np.array([q]), R, 0.0)[0] == pytest.approx(float(ref), rel=1e-13)
    assert thin_objects.disc_pprime(np.array([0.0]), 10.0, 0.0)[0] == pytest.approx((math.pi * 100) ** 2)


@pytest.mark.parametrize("q", [1e-3, 0.3, 1.2])
def test_pcs_family_against_quad(q):
    cyl = lambda x: (math.pi * x * x * 0.9 * _lam(q * x)) ** 2  # noqa: E731
    got = thin_objects.ff_pcs_homogeneouscyl(np.array([q]), _vec([5, 0.2, 0, 0, 0, 0, 0, 1.2, 0, 0.3]))[0]
    assert got == pytest.approx(_lognorm_quad(cyl, 5, 0.2, 4), rel=1e-10)
    plate = lambda x: (0.8 * x * np.sinc(q * x / 2 / math.pi)) ** 2  # noqa: E731
    got = thin_objects.ff_pcs_homogeneousplate(np.array([q]), _vec([4, 0.2, 0, 0, 1, 0.2]))[0]
    assert got == pytest.approx(_lognorm_quad(plate, 4, 0.2, 2), rel=1e-10)
    bil = lambda nu: (math.sqrt(2 * math.pi) * 0.5 * math.exp(-0.5 * (q * 0.5) ** 2)  # noqa: E731
                      - 2 * math.sqrt(2 * math.pi) * 0.4 * 0.6 * math.exp(-0.5 * (q * 0.4) ** 2) * math.cos(q * 2 * nu)) ** 2
    got = thin_objects.ff_pcs_bilayergauss(np.array([q]), _vec([0.5, 1, 0.4, -0.6, 4, 0.1]))[0]
    assert got == pytest.approx(_lognorm_quad(bil, 1, 0.1, 0), rel=1e-10)

    def lay(L, S):
        j0 = lambda x: np.sinc(x / math.pi)  # noqa: E731
        return ((-0.3 - 0.2) * (L + 2 * S) * j0(q * (L + 2 * S) / 2) - (-0.3 - 1) * L * j0(q * L / 2)) ** 2

    ref, _ = integrate.dblquad(lambda u2, u1: _gauss(u1, 0.1) * _gauss(u2, 0.15) * lay(4 * math.exp(u1), 2 * math.exp(u2)),
                               -1.2, 1.2, -1.8, 1.8, epsabs=0, epsrel=1e-11)
    got = thin_objects.ff_pcs_layeredcentrosymmetricxs(np.array([q]), _vec([4, 0.1, 2, 0.15, 1, -0.3, 0.2]))[0]
    assert got == pytest.approx(ref, rel=1e-9)


# ------------------------------------------------------------------ Koyama worm
def _koyama_mp(x):
    x = mp.mpf(x)
    a = x - (1 - mp.e ** (-x))
    b = mp.mpf(5) / 3 * x ** 2 - mp.mpf(52) / 9 * x - mp.mpf(2) / 27 * (1 - mp.e ** (-3 * x)) + 8 * (1 - mp.e ** (-x)) - 2 * x * mp.e ** (-x)
    K = b / a ** 2
    root = mp.sqrt(10 * abs(1 - mp.mpf(3) / 5 * K))
    return a - a * root / 2, a * root


def test_koyama_distribution_against_mpmath():
    mp.mp.dps = 60
    xs = np.array([1e-7, 1e-4, 0.03, 0.4, 0.999, 1.001, 3.0, 30.0, 3000.0])
    f, g2 = worms._koyama_f(xs)
    for x, fi, gi in zip(xs, f, g2):
        fr, gr = _koyama_mp(x)
        assert fi == pytest.approx(float(fr), rel=1e-12)
        assert gi == pytest.approx(float(gr), rel=1e-12)


@pytest.mark.parametrize("lb, L", [(2.0, 50.0), (10.0, 30.0), (20.0, 5.0)])
def test_koyama_against_quad(lb, L):
    q = np.array([1e-3, 0.05, 0.3, 2.0])
    X = 2 * L / lb
    got = worms.koyama(q, lb, L)
    for qi, gi in zip(q, got):
        s = qi * lb / 2

        def phi(x):
            f, g2 = worms._koyama_f(np.array([x]))
            sg = s * math.sqrt(max(g2[0], 0.0))
            return math.exp(-s * s * f[0] / 3) * np.sinc(sg / math.pi) * 2 * (X - x) / X ** 2

        ref, _ = integrate.quad(phi, 0, X, limit=5000, epsabs=0, epsrel=1e-12)
        assert gi == pytest.approx(ref, rel=1e-10)


def test_koyama_limits():
    q = np.array([0.01, 0.1, 1.0])
    rod = worms.koyama(q, 1e4, 10.0)                       # L << lb: rigid rod
    u = q * 10.0
    np.testing.assert_allclose(rod, 2 * special.sici(u)[0] / u - (np.sin(u / 2) / (u / 2)) ** 2, rtol=1e-3)
    gauss = worms.koyama(q[:2], 1.0, 1e5)                  # L >> lb: Gaussian chain, Rg² = L lb/6
    x = (q[:2] ** 2) * 1e5 / 6
    np.testing.assert_allclose(gauss, 2 * (np.expm1(-x) + x) / x ** 2, rtol=2e-3)
    assert worms.ff_koyama(np.array([0.0]), _vec([2.0, 50.0, 0, 3.0]))[0] == 3.0
