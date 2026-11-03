"""pySASmorph numerical core: transforms, extrapolation, level-cut relation, round trips."""

import os

import numpy as np
import pytest
from scipy.integrate import quad

from pysasfit.morph import (LevelCut, MorphParameters, correlations, extrapolate, levelcut_intensity,
                            porosity_from_invariant, reconstruct, specific_surface_voxels, voxel_intensity)
from pysasfit.morph.grf import field_fft, grid_distances
from pysasfit.morph.transforms import moments_pl, porod_tail, sine_transform_pl


def dab(q, A=1.0, xi=50.0):
    """Debye-Anderson-Brumberger: gamma(r) = exp(-r/xi)."""
    return A / (1 + (q * xi) ** 2) ** 2


# ------------------------------------------------------------------ transforms
@pytest.mark.parametrize("r", [1e-5, 1e-2, 0.7, 13.0])
def test_sine_transform_exact_for_piecewise_linear(r):
    x = np.linspace(0.0, 3.0, 301)
    u = x * np.exp(-x)
    ref = sum(quad(lambda t: np.interp(t, x, u) * np.sin(t * r), x[i], x[i + 1])[0] for i in range(300))
    assert sine_transform_pl(x, u, [r])[0] == pytest.approx(ref, rel=1e-8, abs=1e-14)


def test_moments():
    x = np.linspace(0, 2, 11)
    assert moments_pl(x, x, 1) == pytest.approx(8 / 3)


@pytest.mark.parametrize("r", [0.0, 0.1, 5.0, 100.0])
def test_porod_tail(r):
    K, qN = 2.0, 0.5
    ref = K / qN if r == 0 else quad(lambda q: K * q ** -3 / r, qN, np.inf, weight="sin", wvar=r)[0]
    assert porod_tail(K, qN, [r])[0] == pytest.approx(ref, rel=1e-7)


# ------------------------------------------------------------------ extrapolation, invariant, gamma
def test_dab_invariant_porod_and_gamma():
    xi = 50.0
    q = np.geomspace(2e-3, 0.5, 300)
    I = dab(q, 1.0, xi)
    c = extrapolate(q, I, low_q="guinier", porod_points=40)
    assert c.K == pytest.approx(xi ** -4, rel=2e-2)           # I q^4 -> 1/xi^4
    assert c.invariant == pytest.approx(np.pi / (4 * xi ** 3), rel=2e-3)
    r = np.linspace(0, 400, 81)
    assert np.max(np.abs(c.gamma(r) - np.exp(-r / xi))) < 5e-3
    assert c.gamma([0.0])[0] == pytest.approx(1.0)


def test_porod_background_is_fitted_and_removed():
    q = np.geomspace(2e-3, 0.5, 300)
    I = np.where(q < 0.1, dab(q), 1.6e-7 * q ** -4) + 3e-6
    c = extrapolate(q, I, porod_qmin=0.1, fit_background=True)
    assert c.B == pytest.approx(3e-6, rel=1e-6)
    assert c.K == pytest.approx(1.6e-7, rel=1e-6)


# ------------------------------------------------------------------ level cut
@pytest.mark.parametrize("phi", [0.5, 0.3, 0.1])
def test_levelcut_limits(phi):
    lc = LevelCut(phi)
    assert lc.F(1.0) == pytest.approx(phi * (1 - phi), rel=1e-6)
    assert lc.F(-1.0) == pytest.approx(-phi ** 2 if phi <= 0.5 else -(1 - phi) ** 2, rel=1e-4)
    assert lc.F(0.0) == pytest.approx(0.0, abs=1e-12)
    g = np.linspace(-0.8, 1.0, 19)
    assert np.allclose(lc.g_from_gamma(lc.gamma_from_g(g)), g, atol=1e-6)


def test_levelcut_symmetric_case_is_arcsine():
    lc = LevelCut(0.5)
    g = np.linspace(-0.95, 0.95, 21)
    assert np.allclose(lc.F(g), np.arcsin(g) / (2 * np.pi), atol=1e-12)


def test_porosity_from_invariant():
    phi, drho = 0.2, 3e10
    Q = 2 * np.pi ** 2 * drho ** 2 * phi * (1 - phi) / 1e24
    assert porosity_from_invariant(Q, drho) == pytest.approx(phi)


# ------------------------------------------------------------------ round trip g -> I -> g
def _synthetic(phi=0.3, xi=60.0, d=150.0):
    """Exact scattering of a level-cut field with g(r) = exp(-(r/xi)^2) sin(2 pi r/d)/(2 pi r/d).

    g must be smooth at r = 0 (g = 1 - b r^2): a cusp 1 - r/xi would give gamma ~ 1 - c sqrt(r)
    and I ~ q^-3.5 instead of Porod's q^-4."""
    r = np.linspace(0, 3000, 30001)
    kk = 2 * np.pi / d
    with np.errstate(invalid="ignore"):
        g = np.exp(-(r / xi) ** 2) * np.where(r > 0, np.sin(kk * r) / (kk * r), 1.0)
    lc = LevelCut(phi)
    q = np.geomspace(1e-3, 0.6, 400)
    I = levelcut_intensity(q, r, lc.F(g), drho2=1.0)
    return q, I, r, g


def test_round_trip_recovers_g():
    phi = 0.3
    q, I, r0, g0 = _synthetic(phi)
    res = correlations(q, I, 0.01 * I, MorphParameters(porosity=phi, box_size=1200, n=128, porod_points=60))
    gi = np.interp(res.r, r0, g0)
    m = res.r < 500
    assert np.max(np.abs(res.g[m] - gi[m])) < 0.02
    assert res.levelcut.clipped == 0
    assert res.chi2_levelcut < 2.0                       # model reproduces the data to about 1 %
    assert res.drho2 == pytest.approx(1.0, rel=0.02)     # contrast recovered from the invariant


# ------------------------------------------------------------------ fields and voxels
def test_fft_field_has_prescribed_correlation():
    n, a = 64, 4.0
    r = np.linspace(0, 1000, 2001)
    g = np.exp(-(r / 20.0) ** 2)
    psi, info = field_fft(r, g, n, a, rng=1)
    assert info["negative_power"] < 1e-3
    F = np.fft.rfftn(psi.astype(float))
    corr = np.fft.irfftn(np.abs(F) ** 2, s=psi.shape, axes=(0, 1, 2)) / psi.size
    d = grid_distances(n, a)
    for rr in (0.0, 8.0, 20.0, 40.0):
        sel = np.isclose(d, rr)
        assert corr[sel].mean() == pytest.approx(np.exp(-(rr / 20.0) ** 2), abs=0.05)


def test_voxel_intensity_parseval():
    """integral I d^3q / (2 pi)^3 = drho2 phi (1 - phi)."""
    rng = np.random.default_rng(0)
    mat = rng.random((32, 32, 32)) < 0.3
    a = 1.0
    eta = mat - mat.mean()
    F = np.fft.fftn(eta)
    total = np.sum(np.abs(F) ** 2) * a ** 3 / mat.size / (mat.size * a ** 3)
    assert total == pytest.approx(mat.mean() * (1 - mat.mean()), rel=1e-10)
    q, I, cnt = voxel_intensity(mat, a)
    assert np.all(I > 0) and q[0] == pytest.approx(2 * np.pi / 32)


def test_voxel_surface_of_a_slab():
    mat = np.zeros((40, 40, 40), bool)
    mat[:, :, 10:20] = True                      # two planes of area L^2 in a periodic box
    sv = specific_surface_voxels(mat, 1.0) * 1.5   # undo the isotropic 2/3 factor
    assert sv == pytest.approx(2 * 40 * 40 / 40 ** 3)


@pytest.mark.parametrize("method", ["fft", "cosine"])
def test_reconstruct_end_to_end(method):
    phi = 0.3
    q, I, _, _ = _synthetic(phi)
    p = MorphParameters(porosity=phi, box_size=1200, n=64, method=method, n_waves=1500, seed=3, porod_points=60)
    res = reconstruct(q, I, 0.01 * I, p)
    assert res.matrix.shape == (64, 64, 64)
    assert res.statistics["porosity_voxels"] == pytest.approx(phi, abs=1e-3)
    # the voxel structure scatters like the data in the resolved range (speckle: generous tolerance)
    qv, Iv = res.q_voxel, res.I_voxel
    sel = (qv > 0.02) & (qv < 0.5 * np.pi / res.voxel_size)
    ratio = Iv[sel] / np.interp(qv[sel], q, I)
    assert 0.6 < np.median(ratio) < 1.4
    assert np.isfinite(res.S_over_V_porod)
    assert "SASmorph reconstruction" in res.report()


# ------------------------------------------------------------------ export
def test_exports(tmp_path):
    phi = 0.3
    q, I, _, _ = _synthetic(phi)
    res = reconstruct(q, I, 0.01 * I, MorphParameters(porosity=phi, box_size=1200, n=32, seed=1, porod_points=60))
    from pysasfit.morph.export import write_curves, write_matrix

    files = write_curves(res, str(tmp_path / "s"))
    assert all(os.path.isfile(f) for f in files)
    g = np.loadtxt(tmp_path / "s_gr.txt")
    assert g.shape[1] == 2 and g[0, 1] == pytest.approx(1.0)
    p = write_matrix(res, str(tmp_path / "s.vtk"))
    raw = open(p, "rb").read()
    assert b"DIMENSIONS 32 32 32" in raw
    body = np.frombuffer(raw.split(b"LOOKUP_TABLE default\n", 1)[1][:32 ** 3], np.uint8)
    assert np.array_equal(body.reshape((32, 32, 32), order="F"), res.matrix.astype(np.uint8))
    np.testing.assert_array_equal(np.load(write_matrix(res, str(tmp_path / "s.npy"))), res.matrix)
    tifffile = pytest.importorskip("tifffile")
    t = tifffile.imread(write_matrix(res, str(tmp_path / "s.tif")))
    assert t.shape == (32, 32, 32) and np.array_equal(t[5] > 0, res.matrix[:, :, 5])
    h5py = pytest.importorskip("h5py")
    with h5py.File(write_matrix(res, str(tmp_path / "s.h5"))) as h:
        assert np.array_equal(h["matrix"][()] > 0, res.matrix)
        assert "gamma" in h["curves"] and h.attrs["parameters"]


def test_surface_mesh_of_a_sphere(tmp_path):
    pytest.importorskip("pyvista")
    from pysasfit.morph.view3d import surface_mesh

    n, R = 48, 15.0
    x = np.arange(n) - n / 2 + 0.5
    mask = (x[:, None, None] ** 2 + x[None, :, None] ** 2 + x[None, None, :] ** 2) < R ** 2
    mesh = surface_mesh(mask, a=2.0)          # marching cubes on a binary mask, Taubin-smoothed
    assert mesh.area == pytest.approx(4 * np.pi * (2 * R) ** 2, rel=0.1)
    c = np.asarray(mesh.center)
    assert np.allclose(c, 2.0 * (n / 2 - 0.5), atol=1.0)   # voxel centres at i * a
    mesh.save(str(tmp_path / "sphere.stl"))
    assert (tmp_path / "sphere.stl").stat().st_size > 0


def test_fixed_background_is_subtracted():
    q = np.geomspace(2e-3, 0.5, 300)
    I = np.where(q < 0.1, dab(q), 1.6e-7 * q ** -4) + 3e-6
    c = extrapolate(q, I, porod_qmin=0.1, background=3e-6)
    assert (c.B, c.B_mode) == (3e-6, "fixed")
    assert c.K == pytest.approx(1.6e-7, rel=1e-6)
    fitted = extrapolate(q, I, porod_qmin=0.1, fit_background=True)
    assert fitted.B_mode == "fitted" and fitted.invariant == pytest.approx(c.invariant, rel=1e-6)
    assert extrapolate(q, I, porod_qmin=0.1).B_mode == "none"
