"""Python ports of the SASfit plugin sheared_objects: independent references and the corrected C sources."""

from __future__ import annotations

import ctypes
import importlib.util
import math
import os

import numpy as np
import pytest
from scipy import integrate, stats

from pysasfit.studio import ports
from pysasfit.studio.ports import sheared as S

P = [15.0, 4.0, 60.0, 1.0, 2.0, 0.3, 40.0, 0.0, 0.0, 0.0, 0.0]


def _iso(shape, q, p):
    amp = S._AMP[shape]
    p = np.asarray(p, float)
    return integrate.quad(
        lambda c: amp(q, np.array(c), p, 1.0) ** 2, 0, 1, limit=500, epsrel=1e-11
    )[0]


def _sphere_ref(shape, kind, q, p):
    """Full-sphere double integral without the symmetry reductions of the port."""
    p = np.asarray(p, float)
    amp, psi, k = S._AMP[shape], math.radians(p[6]), p[8]
    if kind == "hayterpenfold":

        def f(ph, th):
            cg = math.cos(psi) * math.sin(th) * math.cos(ph) + math.sin(psi) * math.cos(
                th
            )
            return (
                float(S.hayter_penfold(th, ph, k))
                * amp(q, np.array(cg), p, 1.0) ** 2
                * math.sin(th)
            )

        z = 1.0
    else:

        def dens(th):
            return float(S._shape(kind, np.array(min(th, math.pi - th)), p))

        z = (
            2
            * math.pi
            * integrate.quad(lambda th: dens(th) * math.sin(th), 0, math.pi, limit=400)[
                0
            ]
        )

        def f(ph, th):
            cg = math.cos(psi) * math.cos(th) + math.sin(psi) * math.sin(th) * math.cos(
                ph
            )
            return dens(th) * amp(q, np.array(cg), p, 1.0) ** 2 * math.sin(th)

    v = integrate.nquad(
        f, [[0, 2 * math.pi], [0, math.pi]], opts={"limit": 200, "epsrel": 1e-9}
    )[0]
    return v / z


def test_registered():
    names = [n for n in ports.names() if n.startswith("ff_sheared_")]
    plain = [n for n in names if not n.endswith("__schulz_")]
    assert len(plain) == 13 and len(names) == 26
    assert all(ports.get(n).c_reference for n in plain)
    assert not any(ports.get(n).c_reference for n in names if n.endswith("__schulz_"))


@pytest.mark.parametrize("shape", ["cylinder", "spheroid"])
@pytest.mark.parametrize(
    "kind", ["maier_saupe", "gauss", "boltzmann", "onsager", "heavyside", "legendre"]
)
def test_uniform_orientation_is_the_isotropic_average(shape, kind):
    if shape == "spheroid" and kind == "legendre":
        pytest.skip("no spheroid Legendre function")
    q = np.array([1e-5, 0.02, 0.1, 0.4])
    got = S.sheared_intensity(
        q, P, shape=shape, kind=kind
    )  # kappa = 0 (Legendre: S_l = 0)
    ref = np.array([_iso(shape, qi, P) for qi in q])
    np.testing.assert_allclose(got, ref, rtol=1e-9)


@pytest.mark.parametrize(
    ("shape", "kind", "kappa", "psi", "q"),
    [
        ("cylinder", "maier_saupe", 3.0, 40.0, 0.1),
        ("cylinder", "gauss", -2.0, 70.0, 0.15),
        ("spheroid", "boltzmann", 2.0, 20.0, 0.1),
        ("cylinder", "onsager", -3.0, 55.0, 0.08),
        ("cylinder", "hayterpenfold", 3.0, 30.0, 0.1),
        ("spheroid", "hayterpenfold", 8.0, 60.0, 0.12),
    ],
)
def test_against_full_sphere_integral(shape, kind, kappa, psi, q):
    p = list(P)
    p[6], p[8] = psi, kappa
    got = S.sheared_intensity(np.array([q]), p, shape=shape, kind=kind)[0]
    assert got == pytest.approx(_sphere_ref(shape, kind, q, p), rel=1e-9)


def test_forward_scattering():
    p = np.asarray(P, float)
    r, t, length = p[0], p[1], p[2]
    dv = (p[3] - p[4]) * math.pi * r * r * length + (p[4] - p[5]) * math.pi * (
        r + t
    ) ** 2 * length
    for kind in ("maier_saupe", "hayterpenfold"):
        p[8] = 2.0
        assert S.sheared_intensity(np.array([0.0]), p, kind=kind)[0] == pytest.approx(
            dv * dv, rel=1e-12
        )


def test_narrow_cone_is_the_aligned_particle():
    p = list(P)
    p[8] = 200.0  # Heaviside cone of half-width 1/200 around the axis
    q = np.array([0.05, 0.1])
    psi = math.radians(p[6])
    aligned = S._amp_cylinder(q, math.cos(psi), np.asarray(p), 1.0) ** 2
    np.testing.assert_allclose(
        S.sheared_intensity(q, p, kind="heavyside"), aligned, rtol=2e-3
    )


def test_hayter_penfold_is_normalised():
    for k in (0.5, 3.0, 30.0):
        v = integrate.dblquad(
            lambda ph, th, k=k: S.hayter_penfold(th, ph, k) * math.sin(th),
            0,
            math.pi,
            0,
            2 * math.pi,
        )[0]
        assert v == pytest.approx(1.0, rel=1e-8)


def test_lognormal_size_average():
    p = list(P)
    p[7], p[8] = 0.2, 3.0

    def at(nu):
        pp = list(p)
        pp[0] *= nu
        pp[1] *= nu
        pp[2] *= nu
        pp[7] = 0.0
        return S.sheared_intensity(np.array([0.1]), pp, kind="maier_saupe")[0]

    ref = integrate.quad(
        lambda nu: stats.lognorm.pdf(nu, 0.2) * at(nu), 0.2, 4, limit=200, epsrel=1e-10
    )[0]
    assert S.sheared_intensity(np.array([0.1]), p, kind="maier_saupe")[
        0
    ] == pytest.approx(ref, rel=1e-8)


def test_psi_broadcast_and_convergence():
    p = list(P)
    p[8] = 5.0
    q = np.array([0.05, 0.5, 2.0])
    psi = np.radians([0.0, 45.0, 90.0])
    a = S.sheared_intensity(q[:, None], p, psi[None, :], kind="gauss")
    assert a.shape == (3, 3)
    b = S.sheared_intensity(q[:, None], p, psi[None, :], kind="gauss", accuracy=3)
    np.testing.assert_allclose(a, b, rtol=1e-8)


# ------------------------------------------------------------------------------- corrected C (Linux build)
@pytest.fixture(scope="module")
def portref(tmp_path_factory):
    here = os.path.dirname(os.path.abspath(__file__))
    spec = importlib.util.spec_from_file_location(
        "port_reference_build_sheared", os.path.join(here, "port_reference", "build.py")
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    lib = mod.build(str(tmp_path_factory.mktemp("portref_sheared")))
    if lib is None:
        pytest.skip("no C compiler or GSL for the C reference of the ports")
    return ctypes.CDLL(lib)


class _Param(ctypes.Structure):
    _fields_ = [
        ("p", ctypes.c_double * 50),
        ("errStatus", ctypes.c_int),
        ("errStr", ctypes.c_char * 256),
        (
            "kernelSelector",
            ctypes.c_int,
        ),  # port_reference shim: sasfit_param.kernelSelector
    ]


_C_CASES = {
    "maier_saupe": (-3.0, 4.0),
    "gauss": (3.0,),
    "boltzmann": (-2.0, 3.0),
    "onsager": (-3.0, 15.0),
    "heavyside": (-2.5, 1.5),
    "hayterpenfold": (5.0,),
    "legendre": (0.4,),
}


@pytest.mark.parametrize("shape", ["cylinder", "spheroid"])
@pytest.mark.parametrize("kind", list(_C_CASES))
def test_against_corrected_c(portref, shape, kind):
    if shape == "spheroid" and kind == "legendre":
        pytest.skip("no spheroid Legendre function")
    name = f"ff_sheared_{shape}s__{kind}_"
    f = getattr(portref, "sasfit_" + name)
    f.restype = ctypes.c_double
    f.argtypes = [ctypes.c_double, ctypes.POINTER(_Param)]
    q = np.array([0.003, 0.05, 0.25])
    for kappa in _C_CASES[kind]:
        p = [12.0, 3.0, 80.0, 1.0, 2.0, 0.3, 35.0, 0.0, kappa, 0.1, 0.05]
        s = _Param()
        s.p[:11] = p
        c = np.array([f(float(x), ctypes.byref(s)) for x in q])
        np.testing.assert_allclose(
            S.sheared_intensity(q, p, shape=shape, kind=kind), c, rtol=1e-8
        )


# ------------------------------------------------------------------ compiled backend and size average (2026-11-18)
_needs_numba = pytest.mark.skipif(
    not S._sheared_numba.AVAILABLE, reason="Numba not installed"
)


@_needs_numba
@pytest.mark.parametrize("shape,kind", [("cylinder", k) for k in S.DISTRIBUTIONS] + [("spheroid", "maier_saupe"),
                                        ("spheroid", "hayterpenfold"), ("spheroid", "heavyside")])  # fmt: skip
@pytest.mark.parametrize("sigma", [0.0, 0.15])
def test_numba_backend_matches_numpy(shape, kind, sigma):
    p = np.zeros(11)
    p[:9] = [
        10.0,
        2.0,
        60.0 if shape == "cylinder" else 25.0,
        1.0,
        2.0,
        0.3,
        30.0,
        sigma,
        2.5,
    ]
    if kind == "legendre":
        p[8:11] = [0.3, 0.1, 0.05]
    q = np.logspace(-2, np.log10(0.4), 12)
    psi = np.linspace(0.0, 1.5, 12)
    a = S.sheared_intensity(
        q, p, psi, shape=shape, kind=kind, backend="numpy", method="quadrature"
    )
    b = S.sheared_intensity(
        q, p, psi, shape=shape, kind=kind, backend="numba", method="quadrature"
    )
    np.testing.assert_allclose(b, a, rtol=1e-12)


@_needs_numba
def test_cephes_j1_is_scipy_j1():
    from scipy import special

    x = np.concatenate([np.linspace(-30.0, 30.0, 6001), np.logspace(-8, 4, 2001)])
    v = np.array([S._sheared_numba.j1(t) for t in x])
    np.testing.assert_allclose(v, special.j1(x), rtol=1e-15, atol=1e-300)


def test_size_average_by_scaled_q():
    # I(q) = Σ w ν⁶ I₁(q ν) must equal the direct average over the amplitudes A(q, γ; ν) (A includes ν³)
    p = np.zeros(11)
    p[:9] = [8.0, 2.0, 40.0, 1.0, 2.0, 0.0, 40.0, 0.25, 3.0]
    q = np.array([0.02, 0.1, 0.3])
    psi = math.radians(40.0)
    # fine Gauss-Legendre in z = ln ν (24-node Gauss-Hermite, used before 2026-11-18, is off by 4e-4 at q = 0.3)
    z, w = np.polynomial.legendre.leggauss(240)
    z, w = 9.0 * z, 9.0 * w * np.exp(-0.5 * (9.0 * z) ** 2) / math.sqrt(2.0 * math.pi)
    nus = np.exp(0.25 * z)
    th = np.linspace(0.0, math.pi, 801)
    ph = np.linspace(0.0, 2.0 * math.pi, 801)
    T, F = np.meshgrid(th, ph, indexing="ij")
    dens = (
        S._shape("maier_saupe", T, p) / S.orientation_norm("maier_saupe", p) * np.sin(T)
    )
    cosg = math.cos(psi) * np.cos(T) + math.sin(psi) * np.sin(T) * np.cos(F)
    ref = []
    for qi in q:
        a2 = sum(wk * S._amp_cylinder(qi, cosg, p, nk) ** 2 for wk, nk in zip(w, nus))
        ref.append(integrate.simpson(integrate.simpson(a2 * dens, x=ph, axis=1), x=th))
    np.testing.assert_allclose(
        S.sheared_intensity(q, p, psi, kind="maier_saupe"), ref, rtol=2e-6
    )


# ------------------------------------------------------------------ Legendre method, size averages (2026-11-18)
@pytest.mark.parametrize("shape,kind,kappa", [("cylinder", "maier_saupe", 8.0), ("cylinder", "gauss", -3.0),
                         ("cylinder", "heavyside", 2.0), ("cylinder", "legendre", 0.3), ("spheroid", "onsager", -20.0),
                         ("spheroid", "boltzmann", 4.0), ("cylinder", "hayterpenfold", 5.0),
                         ("spheroid", "hayterpenfold", 0.5)])  # fmt: skip
def test_legendre_method_matches_quadrature(shape, kind, kappa):
    p = np.zeros(11)
    p[:9] = [
        10.0,
        2.0,
        80.0 if shape == "cylinder" else 30.0,
        1.0,
        2.0,
        0.3,
        0.0,
        0.0,
        kappa,
    ]
    if kind == "legendre":
        p[8:11] = [0.3, 0.15, 0.05]
    q = np.repeat(np.logspace(-2, np.log10(0.8), 6), 3)
    psi = np.tile(np.radians([5.0, 40.0, 85.0]), 6)
    a = S.sheared_intensity(
        q, p, psi, shape=shape, kind=kind, method="quadrature", accuracy=1.5
    )
    b = S.sheared_intensity(q, p, psi, shape=shape, kind=kind, method="legendre")
    np.testing.assert_allclose(b, a, rtol=1e-10)


def _fine_size_average(q, psi, p, shape, kind, dist, n=500):
    from scipy import special

    sig = p[7]
    if dist == "lognormal":
        x, w = np.polynomial.legendre.leggauss(n)
        z, w = (
            9.0 * x,
            9.0 * w * np.exp(-0.5 * (9.0 * x) ** 2) / math.sqrt(2.0 * math.pi),
        )
        nu = np.exp(sig * z)
    else:
        zz = (1.0 - sig**2) / sig**2
        hi = special.gammainccinv(zz + 7.0, 1e-18) / (zz + 1.0)
        x, w = np.polynomial.legendre.leggauss(n)
        nu, w = 1e-4 + (hi - 1e-4) * (x + 1.0) / 2.0, w * (hi - 1e-4) / 2.0
        w = w * np.exp(
            zz * np.log(nu)
            - (zz + 1.0) * nu
            - (special.gammaln(zz + 1.0) - (zz + 1.0) * math.log(zz + 1.0))
        )
    p1 = np.array(p, float)
    p1[7] = 0.0
    i1 = S.sheared_intensity(
        np.outer(q, nu), p1, np.outer(psi, np.ones_like(nu)), shape=shape, kind=kind
    )
    return i1 @ (w * nu**6)


@pytest.mark.parametrize("shape,kind,kappa,sigma,dist", [("cylinder", "gauss", -2.0, 0.3, "lognormal"),
                         ("spheroid", "heavyside", 2.0, 0.2, "lognormal"), ("cylinder", "maier_saupe", 5.0, 0.3, "schulz"),
                         ("spheroid", "gauss", 3.0, 0.3, "schulz"), ("spheroid", "maier_saupe", 5.0, 0.1, "schulz")])  # fmt: skip
def test_size_average_at_large_q(shape, kind, kappa, sigma, dist):
    # the size average must resolve the oscillations of I₁(q ν) in ν (24 Gauss-Hermite nodes did not: errors up to
    # 50 % at q D ~ 100 before 2026-11-18); spheroid + schulz uses the analytic averages of pysasfit.scatter
    p = np.zeros(11)
    p[:9] = [
        5.0,
        2.0,
        40.0 if shape == "cylinder" else 30.0,
        1.0,
        2.0,
        0.3,
        0.0,
        sigma,
        kappa,
    ]
    q = np.array([0.05, 0.3, 0.6])
    psi = np.radians([15.0, 75.0, 40.0])
    ref = _fine_size_average(q, psi, p, shape, kind, dist)
    got = S.sheared_intensity(q, p, psi, shape=shape, kind=kind, size_dist=dist)
    np.testing.assert_allclose(got, ref, rtol=5e-8)


def test_schulz_spheroid_analytic_vs_nodes():
    p = np.zeros(11)
    p[:9] = [10.0, 2.0, 30.0, 1.0, 2.0, 0.3, 0.0, 0.2, 4.0]
    q = np.array([0.02, 0.2, 0.7])
    psi = np.radians([20.0, 60.0, 80.0])
    a = S.sheared_intensity(
        q, p, psi, shape="spheroid", kind="maier_saupe", size_dist="schulz"
    )
    b = S.sheared_intensity(
        q,
        p,
        psi,
        shape="spheroid",
        kind="maier_saupe",
        size_dist="schulz",
        method="quadrature",
    )
    np.testing.assert_allclose(a, b, rtol=1e-8)


@pytest.mark.parametrize("kappa", [0.3, 2.0, 30.0])
def test_hayter_penfold_fourier_legendre_matches_sphere_quadrature(kappa):
    p = np.zeros(11)
    p[8] = kappa
    psi = np.radians([0.0, 5.0, 40.0, 85.0, 90.0, 120.0])
    a = S.orientation_legendre("hayterpenfold", p, 80, psi, method="sphere")
    b = S.orientation_legendre("hayterpenfold", p, 80, psi, method="fourier")
    np.testing.assert_allclose(b, a, atol=1e-14)
    np.testing.assert_allclose(b[0], 1.0, rtol=1e-13)  # normalisation
