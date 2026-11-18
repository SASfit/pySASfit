"""Python ports of isotropic SASfit plugins (spheres, polymers, misc) against the C sources."""

from __future__ import annotations

import ctypes
import importlib.util
import os

import mpmath as mp
import numpy as np
import pytest

from pysasfit.studio import ports
from pysasfit.studio.ports import (
    check,
    common,
    fractals,
    guwelop,
    micelles,
    misc,
    nonparticulate,
    pearls,
    polymers,
    spheres,
)

HERE = os.path.dirname(os.path.abspath(__file__))
NAMES = [
    d["name"]
    for m in (
        spheres,
        polymers,
        misc,
        nonparticulate,
        pearls,
        micelles,
        fractals,
        guwelop,
    )
    for d in m.SPECS
    if d.get("c_reference", True)
    and d["name"] != "ff_two_attached_spheres_w_distr"  # slow C cubature: below
]


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


@pytest.fixture(scope="module")
def portref(tmp_path_factory):
    spec = importlib.util.spec_from_file_location(
        "port_reference_build_iso", os.path.join(HERE, "port_reference", "build.py")
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    lib = mod.build(str(tmp_path_factory.mktemp("portref_iso")))
    if lib is None:
        pytest.skip("no C compiler or GSL for the C reference of the ports")
    return ctypes.CDLL(lib)


def _ref(lib, name, suffix=""):
    f = getattr(lib, "sasfit_" + name + suffix)
    f.restype = ctypes.c_double
    f.argtypes = [ctypes.c_double, ctypes.POINTER(_Param)]

    def ref(q, p):
        s = _Param()
        s.p[:] = [float(v) for v in p]
        out = np.array([f(float(x), ctypes.byref(s)) for x in q])
        if s.errStatus:
            raise RuntimeError(s.errStr.decode())
        return out

    return ref


@pytest.mark.parametrize("name", NAMES)
def test_against_c(portref, name):
    r = check.compare(
        name, reference=_ref(portref, name), q=np.logspace(-3, 1, 60), n_sets=4, seed=11
    )
    assert r.passed, str(r)


@pytest.mark.parametrize(
    "name",
    [
        n
        for n in NAMES
        if ports.get(n).amplitude is not None and not n.startswith("ff_gauss")
    ],
)
def test_amplitude_against_c(portref, name):
    spec = ports.get(name)
    q = np.logspace(-1.5, 1, 40)
    f = _ref(portref, name, "_f")
    for p in ports.sample_parameters(spec, 3, seed=2):
        c, py = f(q, p), spec.amplitude(q, p)
        m = spec.c_valid(q, p) if spec.c_valid is not None else np.ones(q.shape, bool)
        np.testing.assert_allclose(py[m], c[m], rtol=1e-8, atol=1e-300)


def test_amplitude_squared_is_intensity():
    for name in (
        "ff_sphere",
        "ff_spherical_shell_iii",
        "ff_sphere_with_3_shells",
        "ff_boucher_sphere",
    ):
        spec = ports.get(name)
        q = np.logspace(-2, 1, 30)
        for p in ports.sample_parameters(spec, 2, seed=1):
            np.testing.assert_allclose(
                spec.amplitude(q, p) ** 2, spec(q, p), rtol=1e-12
            )


def test_gauss_amplitude_is_the_debye_amplitude():  # C before 2026-11-18: (1 - e^{+u})/u
    q = np.array([0.0, 0.1, 1.0])
    f = ports.get("ff_gauss").amplitude(q, [10.0, 0, 4.0])
    u = (q * 10.0) ** 2
    np.testing.assert_allclose(
        f,
        2.0 * np.where(u == 0, 1.0, -np.expm1(-u) / np.where(u == 0, 1, u)),
        rtol=1e-14,
    )


def test_stable_helpers_against_mpmath():
    mp.mp.dps = 40
    x = np.logspace(-5, 2, 80)
    phi = np.array([float(3 * (mp.sin(t) - t * mp.cos(t)) / mp.mpf(t) ** 3) for t in x])
    np.testing.assert_allclose(common.phi_sphere(x), phi, rtol=1e-13)
    deb = np.array(
        [float(2 * (mp.mpf(t) - 1 + mp.exp(-mp.mpf(t))) / mp.mpf(t) ** 2) for t in x]
    )
    np.testing.assert_allclose(common.debye(x), deb, rtol=1e-13)
    for a in (0.6, 1.0, 3.5):
        ref = np.array([float(mp.gammainc(a, 0, t) / mp.mpf(t) ** a) for t in x])
        np.testing.assert_allclose(common.lower_gamma_ratio(a, x), ref, rtol=1e-12)


def test_gauspoly_and_fract_branched_against_mpmath():
    mp.mp.dps = 40
    q = np.logspace(-3, 1.5, 60)

    def gp(qq, mw):
        w = mp.mpf(mw) / 1e4 - 1
        k = (mp.mpf(qq)) ** 2 / (1 + 2 * w)
        lead = mp.exp(-k) if w == 0 else (1 + w * k) ** (-1 / w)
        return float(2 * (lead + k - 1) / ((1 + w) * k * k))

    for mw in (1e4, 1.3e4, 5e4):
        np.testing.assert_allclose(
            polymers.gauspoly(q, [1.0, mw, 1e4, 1.0]),
            [gp(t, mw) for t in q],
            rtol=1e-12,
        )

    def fb(qq, rg, vm, c):
        ub = (mp.mpf(qq) * rg) ** 2 * (2 * vm + c) * (2 * vm + c + 1) / 6
        g = lambda a: mp.gammainc(a, 0, ub) / ub**a
        return float((c * c + c) / 2 * (g(c / (2 * vm)) - g((c + 1) / (2 * vm))) / vm)

    np.testing.assert_allclose(
        polymers.fract_branched(q, [2.0, 0.4, 2.6, 1.0]),
        [fb(t, 2.0, 0.4, 2.6) for t in q],
        rtol=1e-11,
    )


def test_volumes_against_c(portref):
    for name in ("ff_multilamellar_vesicle", "ff_sphere", "ff_spherical_shell_iii"):
        spec = ports.get(name)
        f = getattr(portref, "sasfit_" + name + "_v")
        f.restype = ctypes.c_double
        f.argtypes = [ctypes.c_double, ctypes.POINTER(_Param), ctypes.c_int]
        p = spec.defaults()
        s = _Param()
        s.p[:] = [float(v) for v in p]
        x = np.array([5.0, 20.0])
        for dist in (0, 1):
            c = np.array([f(float(v), ctypes.byref(s), dist) for v in x])
            np.testing.assert_allclose(spec.volume(x, p, dist), c, rtol=1e-12)


def test_kratky_without_smearing_matches_c(portref):
    spec = ports.get("ff_kratkysphere")
    p = spec.defaults()
    p[1] = 0.0
    q = np.array([0.01, 0.1, 1.0])
    np.testing.assert_allclose(
        spec(q, p), _ref(portref, "ff_kratkysphere")(q, p), rtol=1e-12
    )


def test_generalized_form_factor_with_sigma_against_c(portref):
    spec = ports.get("ff_generalized_form_factor")
    p = spec.defaults()
    p[1] = 0.1
    q = np.array([0.02, 0.1, 0.3])
    # C cuts the log-normal integral at the 99.9 % point (find_LogNorm_int_range), the port uses Gauss-Hermite
    np.testing.assert_allclose(
        spec(q, p), _ref(portref, "ff_generalized_form_factor")(q, p), rtol=1e-3
    )


# ---------------------------------------------------------------- batch 2 (2026-11-18)


def test_volumes_batch2_against_c(portref):
    f = portref.sasfit_ff_linear_pearls_v
    f.restype = ctypes.c_double
    f.argtypes = [ctypes.c_double, ctypes.POINTER(_Param), ctypes.c_int]
    spec = ports.get("ff_linear_pearls")
    p = spec.defaults()
    s = _Param()
    s.p[:] = [float(v) for v in p]
    np.testing.assert_allclose(
        spec.volume(np.array([1.0]), p, 0), [f(1.0, ctypes.byref(s), 0)], rtol=1e-13
    )


@pytest.mark.parametrize("model", [1, 2, 3])
def test_spinodal2_models_against_c(portref, model):
    p = np.zeros(50)
    p[:5] = [1.0, 0.2, 0.1, 0.1, model]
    q = np.logspace(-3, 1, 40)
    np.testing.assert_allclose(
        nonparticulate.spinodal2(q, p), _ref(portref, "ff_spinodal2")(q, p), rtol=1e-12
    )


def test_q_zero_limits():
    q0 = np.array([0.0])
    p = np.zeros(50)
    p[:9] = [1.0, 0.1, 2.0, 0.1, 10.0, 1.0, 1.0, 2.0, 3.0]
    assert nonparticulate.beaucage_exppowlaw(q0, p)[0] == pytest.approx(
        3.0
    )  # G + Gs (was NaN in C)
    p = np.zeros(50)
    p[[0, 3, 4, 5, 6, 7]] = [2.0, 10.0, 1.0, 3.0, 7.4, 2.6]
    # I(0) = (m β_r + N β_s)² (SASfit convention since 2026-11-18; 0/0 in C before); integer N, g interpolated
    br, bs, n, g = 1.0, 3.0, 7.4, 2.6
    i0 = lambda n, g: (g * (n - 1) * br + n * bs) ** 2  # noqa: E731
    w, wg = n - 7, g - 2
    assert pearls.pearl_necklace(q0, p)[0] == pytest.approx((1 - w) * i0(7, 1) + w * i0(8, 1), rel=1e-14)
    ext = (1 - w) * ((1 - wg) * i0(7, 2) + wg * i0(7, 3)) + w * ((1 - wg) * i0(8, 2) + wg * i0(8, 3))
    assert pearls.pearl_necklace_ext(q0, p)[0] == pytest.approx(ext, rel=1e-14)
    assert pearls.pearl_necklace_mod(q0, p)[0] == pytest.approx(i0(n, 1), rel=1e-14)


def test_pearl_necklace_against_mpmath():
    mp.mp.dps = 40
    R, l, br, bs, NP = 1.2925, 4.2862, 1.6224, 3.1197, 2.3178
    p = np.zeros(50)
    p[[0, 3, 4, 5, 6]] = [R, l, br, bs, NP]

    def ref(q, mod):
        q = mp.mpf(q)
        A = l + 2 * R
        x = q * R
        F = 3 * (mp.sin(x) - x * mp.cos(x)) / x**3
        g = mp.sin(q * A) / (q * A)
        j = mp.sin(q * l / 2) / (q * l / 2)
        LA = mp.si(q * l) / (q * l)
        beta = (mp.si(q * R + q * l) - mp.si(q * R)) / (q * l)

        def one(n):
            m = n - 1
            sss = (
                2
                * bs**2
                * F**2
                * (
                    mp.mpf(n) / 2
                    + n * g / (1 - g)
                    - (1 - abs(g) ** n) * g / (1 - g) ** 2
                )
            )
            srr = br**2 * (
                m * (2 * LA - j * j)
                + 2 * beta**2 * (m / (1 - g) - (1 - abs(g) ** m) / (1 - g) ** 2)
            )
            srs = (
                4
                * bs
                * beta
                * br
                * F
                * (m / (1 - g) - (1 - abs(g) ** m) * g / (1 - g) ** 2)
            )
            return sss + srr + srs  # I(0) = (m β_r + n β_s)²

        if mod:
            return one(mp.mpf(NP))
        n = int(NP)
        return (1 - (NP - n)) * one(n) + (NP - n) * one(n + 1)

    for q in (1e-4, 1e-3, 0.3, 2.0):
        assert pearls.pearl_necklace(np.array([q]), p)[0] == pytest.approx(
            float(ref(q, False)), rel=1e-13
        )
        assert pearls.pearl_necklace_mod(np.array([q]), p)[0] == pytest.approx(
            float(ref(q, True)), rel=1e-13
        )


@pytest.mark.parametrize("eps", [0.2, 0.7, 1.0, 1.6, 5.0])
def test_spheroid_surface(eps):
    # numerical surface of revolution: 2π ∫ x(t) sqrt(x'² + z'²) dt, x = r sin t, z = eps r cos t
    t = np.linspace(0.0, np.pi, 200001)
    integrand = 2 * np.pi * np.sin(t) * np.sqrt(np.cos(t) ** 2 + (eps * np.sin(t)) ** 2)
    num = np.trapezoid(integrand, t) * 3.0**2
    assert micelles.spheroid_surface(3.0, eps) == pytest.approx(num, rel=1e-8)


def test_ell_chains_nagg_core_radius():
    # Nagg V_core/(1-xsolv) is the spheroid volume 4/3 π R³ epsilon (C used 4/3 = 1 before 2026-11-18)
    p = np.zeros(50)
    p[:10] = [50.0, 5.0, 5.0, 1.0, 0.0, 0.0, 0.2, 2.0, 1.0, 1.5]
    q = np.logspace(-2, 0, 5)
    r = (50.0 * 5.0 / 0.8 / (4.0 / 3.0 * np.pi * 1.5)) ** (1.0 / 3.0)
    p_rc = p.copy()
    p_rc[0] = r
    np.testing.assert_allclose(
        micelles.ell_chains_rw_nagg(q, p),
        micelles.ell_chains_rw_rc(q, p_rc),
        rtol=1e-12,
    )


# ---------------------------------------------------------------- batch 3 (2026-11-18)


def test_power_law_sphere_against_mpmath():
    # F = 4π R³ [Φ(qR)/3 - 1F2((α+3)/2; 3/2, (α+5)/2; -(qR)²/4)/(α+3)] (C: sasfit_pfq, not in the reference lib)
    mp.mp.dps = 30
    for r, al, q in (
        (10.0, 1.5, 0.3),
        (10.0, -2.5, 2.0),
        (5.0, 0.5, 0.001),
        (20.0, 4.0, 1.0),
    ):
        qr = mp.mpf(q * r)
        f = 4 * mp.pi * (r**3 * (mp.sin(qr) - qr * mp.cos(qr)) / qr**3
                         - r**3 * mp.hyp1f2((al + 3) / 2, 1.5, (al + 5) / 2, -(qr**2) / 4) / (al + 3))  # fmt: skip
        p = np.zeros(50)
        p[:3] = [r, al, 1.0]
        assert fractals.power_law(np.array([q]), p)[0] == pytest.approx(
            float(f**2), rel=1e-11
        )


def test_fractal_series_against_mpmath():
    # C before 2026-11-18: three-term series for q xmax < 0.32, errors up to 7e-3
    mp.mp.dps = 25
    n, xmin, fd, xmax = 0.5, 2.86, 1.3952, 41.2
    p = np.zeros(50)
    p[:4] = [n, xmin, fd, xmax]
    for q in (0.0075, 0.1, 3.0):
        f = lambda x, q=q: (
            n
            * fd
            * x ** (-1 - fd)
            / (xmin**-fd - xmax**-fd)
            * x**6
            / (1 + (q * x) ** 2) ** 2
        )
        ref = mp.quad(f, [xmin, min(max(1 / q, xmin * 1.01), xmax * 0.99), xmax])
        assert fractals.fractal_series1(np.array([q]), p)[0] == pytest.approx(
            float(ref), rel=1e-12
        )


def test_massfractgauss_large_q_against_mpmath():
    # C before 2026-11-18: e^{-x} 1F1(...; x) overflowed, asymptotic branch off by 4e-4
    mp.mp.dps = 30
    rg, d = 17.5, 1.71
    p = np.zeros(50)
    p[:3] = [rg, d, 1.0]
    for q in (0.05, 2.1):
        x = mp.mpf(q * rg) ** 2 / d
        ref = mp.exp(-x) * mp.hyp1f1((3 - d) / 2, 1.5, x)
        assert fractals.massfractgauss(np.array([q]), p)[0] == pytest.approx(
            float(ref), rel=1e-12
        )


def test_doubleshellchain_is_continuous_in_n():
    p = np.zeros(50)
    p[:10] = [5.0, 1.0, 1.0, 2.0, 1.0, 2.0, 1.5, 0.0, 0.0, 3.0]
    q = np.logspace(-2, 0, 7)
    a = fractals.doubleshellchain(q, p)
    p[9] = 3.0 - 1e-9
    b = fractals.doubleshellchain(
        q, p
    )  # C before 2026-11-18: jumped (only the last row of the n+1 sum)
    np.testing.assert_allclose(b, a, rtol=1e-7)


def test_two_attached_spheres_volume():
    p = np.zeros(50)
    p[[0, 2]] = [5.0, 3.0]
    assert fractals.two_attached_spheres_v(np.array([1.0]), p, 0)[0] == pytest.approx(
        4 / 3 * np.pi * (125 + 27)
    )


def test_two_attached_spheres_w_distr_against_c(portref):
    p = np.zeros(50)
    p[:7] = [8.0, 0.15, 5.0, 0.1, 0.0, 1.0, 2.0]
    q = np.array([0.01, 0.2, 1.0])
    c = _ref(portref, "ff_two_attached_spheres_w_distr")(q, p)
    np.testing.assert_allclose(
        fractals.two_attached_spheres_w_distr(q, p), c, rtol=1e-4
    )  # C: 99.9 % range


def test_aggregate_exp_cutoff_against_mpmath():
    mp.mp.dps = 25
    for xi, d, a, q in ((2.0, 1.5, 0.7, 3.0), (3.0, 1.3, 1.6, 0.2)):
        f = lambda x, xi=xi, d=d, a=a, q=q: (
            d * x ** (d - 1) * mp.exp(-((x / xi) ** a)) * mp.sin(q * x) / (q * x)
        )
        pts = (
            [0]
            + [xi * 2.0 ** (-k) for k in range(30, 0, -1)]
            + list(np.linspace(xi, xi * 60 ** (1 / a), 200))
        )
        ref = mp.quad(f, pts) / (xi**d * mp.gamma(d / a) * d / a)
        p = np.zeros(50)
        p[:4] = [xi, d, 1.0, a]
        assert fractals.aggregate_exp_cutoff(np.array([q]), p)[0] == pytest.approx(
            float(ref), rel=1e-11
        )


# ---------------------------------------------------- forward scattering / geometry checks (2026-11-18)


@pytest.mark.parametrize("n", [1.0, 2.0, 3.0, 5.0])
def test_linear_pearls_forward_scattering(n):
    """SASfit convention I(0) = (Δη V_total)² with V_total = the volume function N V (integer N; non-integer N
    interpolates the intensities of floor(N) and floor(N)+1 pearls)."""
    spec = ports.get("ff_linear_pearls")
    p = np.zeros(spec.n_params if hasattr(spec, "n_params") else 5)
    p[:5] = [10.0, 5.0, n, 1.3, 0.3]
    i0 = spec(np.array([1e-8]), p)[0]
    vtot = spec.volume(np.array([1.0]), p, 0)[0]
    np.testing.assert_allclose(i0, (1.0 * vtot) ** 2, rtol=1e-10)


def test_tetrahedrondoubleshell_geometry_and_forward_scattering():
    """The five centre distances must be realisable in 3D (centred Gram matrix of rank 3), I(0) = n² (Σ Δη_i V_i)²."""
    import math

    from pysasfit.studio.ports import fractals

    d = np.ones((5, 5))
    np.fill_diagonal(d, 0.0)
    d[3, 4] = d[4, 3] = math.sqrt(8.0 / 3.0)
    j = np.eye(5) - 0.2
    ev = np.linalg.eigvalsh(-0.5 * j @ d**2 @ j)
    assert np.sum(ev > 1e-12) == 3 and ev.min() > -1e-12
    # the port's √(8/3) R12 for pair (4, 5): compare with the explicit sum over the bipyramid coordinates
    p = [20.0, 5.0, 5.0, 10.0, 1.0, 0.5, 0.3, 0.1, 0.2, 5.0]
    r12 = 2 * (20 + 5 + 5) + 10
    h = math.sqrt(2.0 / 3.0) * r12
    base = np.array([[1, 0, 0], [-0.5, math.sqrt(3) / 2, 0], [-0.5, -math.sqrt(3) / 2, 0]]) * r12 / math.sqrt(3)
    x = np.vstack([base, [0, 0, h], [0, 0, -h]])
    q = np.array([1e-8, 0.01, 0.05, 0.1])
    rij = np.linalg.norm(x[:, None] - x[None], axis=-1)
    s = np.array([np.sum(np.sinc(qi * rij / np.pi)) for qi in q])
    a = fractals._shell_amp(q, p)
    np.testing.assert_allclose(fractals.tetrahedrondoubleshell(q, p), a**2 * s, rtol=1e-12)
    vol = lambda r: 4.0 / 3.0 * math.pi * r**3  # noqa: E731
    av = vol(20) * (1.0 - 0.5) + vol(25) * (0.5 - 0.1 - 0.8 * 0.2) + vol(30) * 0.8 * 0.2
    np.testing.assert_allclose(fractals.tetrahedrondoubleshell(q[:1], p)[0], 25 * av**2, rtol=1e-9)


def _pearl_necklace_rg2(r, l, br, bs, n, g):
    """Exact Rg² of N pearls joined by g freely jointed rods (end rods radial): <d²> between two points is the sum of
    the squared lengths of the independent bond pieces between them (+ 3R²/5 per pearl point)."""
    bonds = [l + 2 * r] if g == 1 else [r + l] + [l] * (g - 2) + [l + r]
    starts = [r] if g == 1 else [r] + [0.0] * (g - 1)
    m2 = lambda a, b: (b**3 - a**3) / (3 * (b - a))  # noqa: E731  (mean of x² for x uniform in [a, b])
    elems = []  # (weight, gap, bond) ; pearls: gap index k at the bond boundary, bond = None
    for k in range(n):
        elems.append((bs, k, None))
        if k < n - 1:
            elems += [(br, k, j) for j in range(g)]

    def d2(a, b):
        (wa, ka, ja), (wb, kb, jb) = a, b
        if (ka, ja) == (kb, jb):
            return 1.2 * r * r if ja is None else l * l / 6.0
        # order: a before b along the chain
        pos = lambda k, j: (k, -1) if j is None else (k, j)  # noqa: E731
        if pos(ka, ja) > pos(kb, jb):
            (ka, ja), (kb, jb) = (kb, jb), (ka, ja)
        out = 0.0
        # piece of a's bond after the point (or 3R²/5 for a pearl)
        if ja is None:
            out += 0.6 * r * r
            k0, j0 = ka, 0
        else:
            s0 = starts[ja]
            out += m2(bonds[ja] - s0 - l, bonds[ja] - s0)
            k0, j0 = (ka, ja + 1) if ja + 1 < g else (ka + 1, -1)
        if jb is None:
            out += 0.6 * r * r
            k1, j1 = kb, 0  # full bonds up to pearl kb
        else:
            out += m2(starts[jb], starts[jb] + l)
            k1, j1 = kb, jb
        # full bonds strictly between
        k, j = k0, j0
        if j == -1:
            k, j = k, 0  # start at the pearl centre k0 (a's bond ended there)
        while (k, j) < (k1, j1):
            out += bonds[j] ** 2
            j += 1
            if j == g:
                k, j = k + 1, 0
        return out

    w = sum(e[0] for e in elems)
    return sum(a[0] * b[0] * d2(a, b) for a in elems for b in elems) / (2 * w * w)


@pytest.mark.parametrize("n,g", [(1, 3), (2, 1), (3, 1), (2, 2), (3, 2), (4, 3), (5, 4)])
def test_pearl_necklace_ext_radius_of_gyration(n, g):
    """g freely jointed rods (length l) between the pearls, end rods radial; g = 1 is pearl_necklace."""
    r, l, br, bs = 10.0, 30.0, 0.5, 2.0
    p = [r, 0, 0, l, br, bs, n, g]
    q = np.array([1e-4, 2e-4]) / (l + 2 * r)
    pq = pearls.pearl_necklace_ext(q, p) / ((g * (n - 1)) * br + n * bs) ** 2
    rg2 = (4 * (1 - pq[0]) / q[0] ** 2 - (1 - pq[1]) / q[1] ** 2)  # Richardson, = Rg²/3 * 3
    np.testing.assert_allclose(rg2, _pearl_necklace_rg2(r, l, br, bs, n, g), rtol=1e-6)
    if g == 1:
        qq = np.geomspace(1e-3, 2, 20)
        np.testing.assert_allclose(pearls.pearl_necklace_ext(qq, p), pearls.pearl_necklace(qq, p), rtol=1e-13)
