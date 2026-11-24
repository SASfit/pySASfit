"""Batch 5 of the isotropic ports (2026-11-23): worm-like chains (polymer_semi_flexible + the worm functions of
thin_objects) and the spherical micelles with worm / SAW chains (spherical_ellipsoidal_micelles).

C comparison against the (fixed) C code; checks independent of C: direct sums (freely jointed chain), scipy
quadrature (Kholodenko, corona profile), mpmath (fRg2), continuity of method 3 at q0, forward scattering."""

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
from pysasfit.studio.ports import check, micelles_worm, worms

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
    spec = importlib.util.spec_from_file_location("port_reference_build_b5", os.path.join(HERE, "port_reference", "build.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    lib = mod.build(str(tmp_path_factory.mktemp("portref_b5")))
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


_NAMES = [d["name"] for d in worms.SPECS + micelles_worm.SPECS]


@pytest.mark.parametrize("name", _NAMES)
def test_against_c(portref, name):
    r = check.compare(name, reference=_ref(portref, name), q=np.logspace(-4, 1.2, 40), n_sets=6, seed=11)
    assert r.passed, str(r)


@pytest.mark.parametrize("exvol", [0, 1])
@pytest.mark.parametrize("lb, L", [(10.0, 30.0), (10.0, 100.0), (10.0, 2000.0), (3.0, 11.0)])
def test_worm_ps3_continuous_at_q0(lb, L, exvol):
    if L > 4 * lb:
        q0 = 3.1
    else:
        q0 = max(2.02 * lb / math.sqrt(worms.frg2(L, lb)), 4.0)
    x = q0 / lb
    s = worms.worm_ps3(np.array([x * (1 - 1e-10), x * (1 + 1e-10)]), lb, L, exvol)
    assert abs(s[1] / s[0] - 1) < 1e-8
    d = worms.worm_ps3(x * np.array([1 - 2e-5, 1 - 1e-5, 1 + 1e-5, 1 + 2e-5]), lb, L, exvol)
    sl, sr = (d[1] - d[0]) / (x * 1e-5), (d[3] - d[2]) / (x * 1e-5)
    assert abs(sr / sl - 1) < 5e-4        # smooth match of value and slope


@pytest.mark.parametrize("fn", [worms.worm_ps1, worms.worm_ps3])
def test_limits(fn):
    for ex in (0, 1):
        s = fn(np.array([0.0, 1e-9, 1e4]), 2.0, 400.0, ex)
        # exvol: the tanh switch of the excluded-volume chain is 1e-9 at q -> 0 (formula of the paper, as in C)
        assert s[0] == 1.0 and s[1] == pytest.approx(1.0, abs=1e-12 if ex == 0 else 2e-9)
        assert s[2] == pytest.approx(math.pi / (1e4 * 400.0), rel=2e-2)   # rod-like π/(qL)


def test_frg2_series_against_mpmath():
    for n in (1e-6, 1e-3, 0.3, 0.999, 1.0, 1.5, 20.0):
        with mp.workdps(80):
            nn = mp.mpf(n)
            g = 1 - mp.mpf(3) / 2 / nn + mp.mpf(3) / 2 / nn**2 - mp.mpf(3) / 4 / nn**3 * (1 - mp.e ** (-2 * nn))
        assert worms.frg2(n * 7.0, 7.0) == pytest.approx(float(7.0 * n * 7.0 / 6 * g), rel=1e-13)
    assert worms.frg2(1e-4, 1.0) == pytest.approx(1e-8 / 12, rel=1e-3)          # rod limit L²/12


@pytest.mark.parametrize("n", [1, 3, 10, 57])
def test_freely_jointed_against_direct_sum(n):
    b = 2.0
    q = np.logspace(-6, 1.3, 60)
    x = q * b
    with mp.workdps(40):
        ref = []
        for xi in x:
            xi = mp.mpf(xi)
            si = mp.si(xi)
            prod = 2 * si / xi - (mp.sin(xi / 2) / (xi / 2)) ** 2
            G, g = si / xi, mp.sin(xi) / xi
            pair = sum((n - k) * g ** (k - 1) for k in range(1, n))
            ref.append(float((n * prod + 2 * G * G * pair) / n**2))
    np.testing.assert_allclose(worms.freely_jointed(q, b, n * b), ref, rtol=1e-12)


def test_freely_jointed_negative_gamma_noninteger():
    s = worms.freely_jointed(np.array([2.0]), 1.0, 7.5)       # j0(2) > 0
    t = worms.freely_jointed(np.array([4.0]), 1.0, 7.5)       # j0(4) < 0: finite (was NaN in C)
    assert np.isfinite(s).all() and np.isfinite(t).all()
    a = worms.freely_jointed(np.array([4.0]), 1.0, 7.0)
    c = worms.freely_jointed(np.array([4.0]), 1.0, 8.0)
    assert min(a, c) - 0.05 < t < max(a, c) + 0.05


def test_kholodenko_against_scipy():
    for lb, L in ((2.0, 50.0), (5.0, 3.0), (1.0, 2000.0)):
        N = L / lb
        for q in (1e-4, 0.3, 1.0, 2.9 / lb, 3.0 / lb, 3.3 / lb, 8.0 / lb):
            qlb3 = q * lb / 3.0

            def fy(x):
                if x == 0:
                    return 1.0
                if qlb3 < 1:
                    E = math.sqrt(1 - qlb3**2)
                    return math.exp(E * x - x) * (-math.expm1(-2 * E * x)) / (-math.expm1(-2 * x)) / E
                if qlb3 == 1:
                    return x / math.sinh(x) if x < 700 else 0.0
                F = math.sqrt(qlb3**2 - 1)
                return math.sin(F * x) / F * 2 * math.exp(-x) / (-math.expm1(-2 * x))

            X = 3 * N
            v, _ = integrate.quad(lambda x: 2 / X * (1 - x / X) * fy(x), 0, X, limit=2000, epsabs=0, epsrel=1e-12)
            assert worms.kholodenko(np.array([q]), lb, L)[0] == pytest.approx(v, rel=1e-9), (lb, L, q)


def test_corona_against_scipy():
    for rc, t, alpha in ((0.0, 5.0, 1.3), (2.0, 5.0, 0.0), (2.0, 5.0, 3.0), (3.0, 8.0, 4.2)):
        for q in (1e-3, 0.7, 3.0):
            num, _ = integrate.quad(lambda r: r ** (2 - alpha) * (np.sinc(q * r / np.pi)), rc, rc + t, epsrel=1e-13,
                                    epsabs=0, limit=500)
            den, _ = integrate.quad(lambda r: r ** (2 - alpha), rc, rc + t, epsrel=1e-13)
            assert micelles_worm.corona_amplitude(np.array([q]), rc, t, alpha)[0] == pytest.approx(num / den, rel=1e-10)


def test_forward_scattering():
    """I(0) = (Δη V)² of the whole micelle."""
    vb, ec, eb, es, x = 400.0, 6.0, 1.0, 0.5, 0.2
    # sphere+chains(SAW)_Rc: Nagg = V (1 - x)/V_core
    rcore, vcore = 8.0, 2.0
    nagg = 4 / 3 * math.pi * rcore**3 * (1 - x) / vcore
    total = nagg * vcore * (ec - es) + nagg * vb * (eb - es)
    p = [rcore, vcore, vb, ec, eb, es, x, 40.0, 4.0]
    assert micelles_worm.sphere_chains_saw_rc(np.array([1e-8]), p)[0] == pytest.approx(total**2, rel=1e-8)
    # the R_core/n_agg variant with the same Nagg gives the same (C before 2026-11-23: core volume / (1 - x))
    n_agg = nagg / (4 * math.pi * rcore**2)
    p2 = [rcore, n_agg, vb, ec, eb, es, x, 40.0, 4.0]
    q = np.logspace(-3, 0.5, 20)
    np.testing.assert_allclose(micelles_worm.sphere_chains_saw(q, p2), micelles_worm.sphere_chains_saw_rc(q, p),
                               rtol=1e-12)
    # sphere+R^-a, all three variants
    for ex in (0, 1, 2):
        p = [rcore, vcore, vb, ec, eb, es, 1.5, 5.0, 40.0 if ex < 2 else 3.0, 4.0 if ex < 2 else 0.6, x, ex]
        assert micelles_worm.sphere_r_a_rc(np.array([1e-8]), p)[0] == pytest.approx(total**2, rel=1e-8)
        pa = [rcore, n_agg, *p[2:]]
        np.testing.assert_allclose(micelles_worm.sphere_r_a(q, pa), micelles_worm.sphere_r_a_rc(q, p), rtol=1e-12)
        pn = [nagg, vcore, *p[2:]]
        np.testing.assert_allclose(micelles_worm.sphere_r_a_nagg(q, pn), micelles_worm.sphere_r_a_rc(q, p), rtol=1e-12)
    # no chains: the core alone (C returned Φ² without volume and contrast)
    p0 = [rcore, 0.0, vb, ec, eb, es, 1.5, 5.0, 40.0, 4.0, x, 0]
    v = 4 / 3 * math.pi * rcore**3 * (1 - x) * (ec - es)
    assert micelles_worm.sphere_r_a(np.array([1e-8]), p0)[0] == pytest.approx(v * v, rel=1e-8)


def test_registered():
    ports.register_aliases()
    for n in _NAMES:
        assert ports.get(n).name == n
