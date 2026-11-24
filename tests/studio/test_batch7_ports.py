"""Batch 7 of the isotropic ports (2026-11-24): plugins cylinders and ellipsoidal_shell.

C comparison against the (fixed) C code (ellipsoidal shells with σ = 0: the C code integrates the log-normal only
over SASfit's truncated range), independent references (scipy quadrature, mpmath) and forward scattering."""

from __future__ import annotations

import ctypes
import importlib.util
import math
import os

import mpmath as mp
import numpy as np
import pytest
from scipy import integrate, special

from pysasfit.studio.ports import check, cylinders, ellipsoidal_shell, get, sample_parameters

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
    spec = importlib.util.spec_from_file_location("port_reference_build_b7", os.path.join(HERE, "port_reference", "build.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    lib = mod.build(str(tmp_path_factory.mktemp("portref_b7")))
    if lib is None:
        pytest.skip("no C compiler or GSL for the C reference of the ports")
    return ctypes.CDLL(lib)


def _ref(lib, name, kind=""):
    f = getattr(lib, "sasfit_" + name + kind)
    f.restype = ctypes.c_double
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


_CYL = [d["name"] for d in cylinders.SPECS if d["name"] != "ff_torus"]


@pytest.mark.parametrize("name", _CYL)
def test_cylinders_against_c(portref, name):
    r = check.compare(name, reference=_ref(portref, name), q=np.logspace(-3, 0.8, 30), n_sets=5, seed=7)
    assert r.passed, str(r)


def test_torus_against_c(portref):
    q = np.logspace(-3, 0.8, 7)
    for p in ([10, 3, 1, 1, 1, 0.5, 0], [10, 3, 0, 1.5, 1, 0.5, 0], [5, 2, 1, 0.7, 2, 2, 0.3]):
        np.testing.assert_allclose(cylinders.torus(q, _vec(p)), _ref(portref, "ff_torus")(q, p), rtol=1e-11)
    p = [10, 3, 1, 1, 1, 0.5, 0]
    np.testing.assert_allclose(cylinders.torus_f(q, _vec(p)), _ref(portref, "ff_torus", "_f")(q, p), rtol=1e-11,
                               atol=1e-12 * abs(cylinders.torus_f(np.array([0.0]), _vec(p))[0]))


@pytest.mark.parametrize("name", [d["name"] for d in ellipsoidal_shell.SPECS])
def test_ellipsoidal_against_c(portref, name):
    spec = get(name)
    q = np.logspace(-3, 0.8, 24)
    for p in sample_parameters(spec, 4, seed=5):
        if "spheroid" not in name:
            p[4] = 0.0
        np.testing.assert_allclose(spec.intensity(q, p), _ref(portref, name)(q, p), rtol=1e-11)


@pytest.mark.parametrize("name", ["ff_ellipsoidal_shell_2t", "ff_ellipsoidal_shell_re", "ff_ellipsoidal_shell_0t"])
def test_ellipsoidal_sigma_loosely_against_c(portref, name):
    q = np.logspace(-3, 0.5, 8)
    p = _vec([10, 5, 0, 1.5, 0.15, 1, 0.5, 0, 0.1, 0.2])
    np.testing.assert_allclose(get(name).intensity(q, p), _ref(portref, name)(q, p), rtol=2e-5)


_AMP = [("ff_cylinder", [5, 0, 40, 1]), ("ff_disc", [5, 0, 0, 1]), ("ff_ellcylshell1", [5, 1, 1.6, 40, 1, 0.5, 0]),
        ("ff_cylshell2", [5, 1, 1, 40, 1, 0.5, 0]), ("ff_ellipsoidal_shell_rp_t", [10, 5, 0, 1.5, 0, 1, 0.5, 0, 0, 0]),
        ("ff_spheroid_V", [800, 5, 0, 0, 0, 1, 0, 0.2]), ("ff_spheroid_nu", [2.5, 5, 0, 0, 0, 1, 0, 0.2])]


@pytest.mark.parametrize("name, p", _AMP)
def test_amplitudes_against_c(portref, name, p):
    q = np.logspace(-3, 0.6, 10)
    f = get(name).amplitude(q, _vec(p))
    ref = _ref(portref, name, "_f")(q, p)
    np.testing.assert_allclose(f, ref, rtol=1e-10, atol=1e-11 * np.max(np.abs(ref)))


# ------------------------------------------------------------------ independent references
def _lam(u):
    return 1.0 if u == 0 else 2.0 * special.j1(u) / u


def _j0(x):
    return 1.0 if x == 0 else math.sin(x) / x


@pytest.mark.parametrize("q", [1e-3, 0.2, 1.5])
def test_cylinder_against_quad(q):
    R, L = 5.0, 40.0
    f = lambda a: math.sin(a) * (math.pi * R * R * L * _lam(q * R * math.sin(a)) * _j0(q * L * math.cos(a) / 2)) ** 2  # noqa: E731
    ref, _ = integrate.quad(f, 0, math.pi / 2, epsabs=0, epsrel=1e-13, limit=500)
    assert cylinders.cylinder(np.array([q]), _vec([R, 0, L, 1]))[0] == pytest.approx(ref, rel=1e-11)


@pytest.mark.parametrize("q", [1e-3, 0.3, 1.2])
def test_ellcylshell_against_dblquad_and_continuity(q):
    R, t, eps, L, ec, es, ev = 5.0, 1.0, 1.6, 30.0, 1.0, 0.4, 0.0

    def A(al, ph, typ):
        ri = math.hypot(R * math.sin(ph), eps * R * math.cos(ph))
        ro = math.hypot((R + t) * math.sin(ph), (eps * R + t) * math.cos(ph))
        b1 = math.pi * R * R * eps * L * (ec - es)
        b2 = math.pi * (R + t) * (eps * R + t) * (L + 2 * t * typ) * (es - ev)
        return (b1 * _j0(q * L / 2 * math.cos(al)) * _lam(q * ri * math.sin(al))
                + b2 * _j0(q * (L / 2 + t * typ) * math.cos(al)) * _lam(q * ro * math.sin(al)))

    for typ, f in ((0, cylinders.ellcylshell1), (1, cylinders.ellcylshell2)):
        ref, _ = integrate.dblquad(lambda al, ph: A(al, ph, typ) ** 2 * math.sin(al) * 2 / math.pi, 0, math.pi / 2,
                                   0, math.pi / 2, epsabs=0, epsrel=1e-11)
        assert f(np.array([q]), _vec([R, t, eps, L, ec, es, ev]))[0] == pytest.approx(ref, rel=1e-9)
    a = cylinders.ellcylshell2(np.array([q]), _vec([R, t, 1.0, L, ec, es, ev]))[0]
    b = cylinders.ellcylshell2(np.array([q]), _vec([R, t, 1.0 + 1e-9, L, ec, es, ev]))[0]
    assert a == pytest.approx(b, rel=1e-7)


def test_forward_scattering_and_volumes():
    R, t, L, ec, es, ev = 5.0, 1.0, 30.0, 1.0, 0.4, 0.1
    i0 = cylinders.cylshell2(np.array([0.0]), _vec([R, t, 1, L, ec, es, ev]))[0]
    b1 = math.pi * R * R * L * (ec - es)
    b2 = math.pi * (R + t) ** 2 * (L + 2 * t) * (es - ev)
    assert i0 == pytest.approx((b1 + b2) ** 2, rel=1e-13)
    v = get("ff_cylshell2").volume
    assert v(np.array(R), _vec([R, t, 1, L]), 0) == pytest.approx(math.pi * (R + t) ** 2 * (L + 2 * t))
    # spheroid_V: I(0) = (Δη V)²
    s = ellipsoidal_shell.spheroid_V(np.array([0.0]), _vec([800, 5, 0, 0, 0, 1.2, 0, 0.2]))[0]
    assert s == pytest.approx((800 * 1.0) ** 2, rel=1e-13)
    # torus (ring torus, a < R): V = 2 π² R a² ν
    R, a, nu = 10.0, 3.0, 1.3
    t0 = cylinders.torus(np.array([0.0]), _vec([R, a, 0, nu, 1, 1, 0]))[0]
    assert t0 == pytest.approx((2 * math.pi ** 2 * R * a * a * nu) ** 2, rel=1e-12)
    # mean outer volume of the ellipsoidal shells
    p = _vec([10, 5, 0, 1.5, 0.2, 1, 0.5, 0, 0, 0])
    for name, sc in (("ff_ellipsoidal_shell_2t", (1, 1, 1)), ("ff_ellipsoidal_shell_re", (0, 1, 0)),
                     ("ff_ellipsoidal_shell_rp_t", (1, 0, 1)), ("ff_ellipsoidal_shell_0", (0, 0, 0))):
        s = 0.2 if any(sc) else 0.0
        ref, _ = integrate.quad(lambda u: math.exp(-0.5 * (u / max(s, 1e-30)) ** 2) / (math.sqrt(2 * math.pi) * max(s, 1e-30))
                                * 4 / 3 * math.pi * (10 * math.exp(u) ** sc[0] + 1.5 * math.exp(u) ** sc[2])
                                * (5 * math.exp(u) ** sc[1] + 1.5 * math.exp(u) ** sc[2]) ** 2, -3, 3, epsrel=1e-12) if s \
            else (4 / 3 * math.pi * 11.5 * 6.5 ** 2, 0)
        assert get(name).volume(np.array(10.0), p, 0) == pytest.approx(ref, rel=1e-10)


@pytest.mark.parametrize("q", [1e-3, 0.1, 0.5])
def test_ellipsoidal_shell_sigma_against_dblquad(q):
    Rp, Re, t, s, ec, es, ev, xc, xs = 10.0, 5.0, 1.5, 0.15, 1.0, 0.5, 0.0, 0.1, 0.2
    ecp, esp = ec * (1 - xc) + ev * xc, es * (1 - xs) + ev * xs

    def F(mu, A, B, T):
        rc = math.sqrt((A * mu) ** 2 + B * B * (1 - mu * mu))
        rt = math.sqrt(((A + T) * mu) ** 2 + (B + T) ** 2 * (1 - mu * mu))
        ph = lambda x: 1.0 if x == 0 else 3 * special.spherical_jn(1, x) / x  # noqa: E731
        return ((ecp - esp) * 4 / 3 * math.pi * A * B * B * ph(q * rc)
                + (esp - ev) * 4 / 3 * math.pi * (A + T) * (B + T) ** 2 * ph(q * rt))

    g = lambda u: math.exp(-0.5 * (u / s) ** 2) / (math.sqrt(2 * math.pi) * s)  # noqa: E731
    ref, _ = integrate.dblquad(lambda mu, u: g(u) * F(mu, Rp, Re * math.exp(u), t * math.exp(u)) ** 2,
                               -10 * s, 10 * s, 0, 1, epsabs=0, epsrel=1e-11)
    got = ellipsoidal_shell.ellipsoidal_shell_re_t(np.array([q]), _vec([Rp, Re, 0, t, s, ec, es, ev, xc, xs]))[0]
    assert got == pytest.approx(ref, rel=1e-9)


def test_porod_cylinders_small_q_against_mpmath():
    mp.mp.dps = 40

    def long_mp(q, R, L):
        q = mp.mpf(q)
        mu, sig = L * q, 2 * R * q
        pi_mu = mp.si(mu) + mp.cos(mu) / mu + mp.sin(mu) / mu ** 2
        G1 = 2 / (sig / 2) * mp.besselj(1, sig / 2)
        om = 8 / sig ** 2 * (3 * mp.besselj(2, sig) + mp.besselj(0, sig) - 1)
        return 2 / mu * (pi_mu * G1 ** 2 - om / mu - mp.sin(mu) / mu ** 2)

    def flat_mp(q, R, L):
        q = mp.mpf(q)
        sig, mu = L * q, 2 * R * q
        J0, J1 = (lambda x: mp.besselj(0, x)), (lambda x: mp.besselj(1, x))
        I1 = lambda x: 2 * x * J0(x) - 2 * J1(x) - mp.pi * x * (J0(x) * mp.struveh(1, x) - J1(x) * mp.struveh(0, x))  # noqa: E731
        I0s = (I1(sig) + sig * 2 / sig * J1(sig)) / 2
        Om = 2 / sig * (I0s - 2 * J1(sig))
        xi = (mp.sin(sig / 2) / (sig / 2)) ** 2
        return 8 / mu ** 2 * (xi + I1(mu) * Om / mu - 2 / mu * J1(mu))

    for f, ref, (R, L) in ((cylinders.longcylinder, long_mp, (2.0, 100.0)), (cylinders.longcylinder, long_mp, (1.0, 300.0)),
                           (cylinders.flatcylinder, flat_mp, (40.0, 2.0)), (cylinders.flatcylinder, flat_mp, (10.0, 5.0))):
        V = math.pi * R * R * L
        for x in (1e-6, 1e-3, 0.1, 0.299, 0.301, 0.6, 1.0, 3.0):
            q = x / max(L, 2 * R)
            assert f(np.array([q]), _vec([R, 0, L, 1]))[0] / V ** 2 == pytest.approx(float(ref(q, R, L)), rel=1e-9)
    assert cylinders.longcylinder(np.array([0.0]), _vec([2, 0, 100, 1.5]))[0] == pytest.approx((1.5 * math.pi * 4 * 100) ** 2)


def test_disc_small_q_against_mpmath():
    mp.mp.dps = 40
    R = 10.0
    for u in (1e-8, 1e-4, 0.1, 0.2499, 0.2501, 3.0):
        q = u / R
        ref = (mp.pi * R * R) ** 2 * 2 / mp.mpf(u) ** 2 * (1 - mp.besselj(1, 2 * mp.mpf(u)) / mp.mpf(u))
        assert cylinders.disc(np.array([q]), _vec([R, 0, 0, 1]))[0] == pytest.approx(float(ref), rel=1e-13)
        fref = mp.pi * R * R * 2 * (1 - mp.cos(mp.mpf(u))) / mp.mpf(u) ** 2
        assert cylinders.disc_f(np.array([q]), _vec([R, 0, 0, 1]))[0] == pytest.approx(float(fref), rel=1e-13)
