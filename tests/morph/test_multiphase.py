"""Multi-phase SASmorph: multi-level cut of one Gaussian field, contrast series."""

import numpy as np
import pytest
from scipy.integrate import quad

from pysasfit.morph import LevelCut, MorphParameters, correlations
from pysasfit.morph.grf import field_fft, grid_distances
from pysasfit.morph.multiphase import (DataSet, MultiCut, MultiParameters, PhaseSystem, correlations_multi,
                                       interface_areas, reconstruct_multi)
from pysasfit.morph.scattering import levelcut_intensity


def _g(r, xi=60.0, d=150.0):
    kk = 2 * np.pi / d
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.exp(-(r / xi) ** 2) * np.where(r > 0, np.sin(kk * r) / (kk * r), 1.0)


def _data(phases, sld, q=None, scale=1.0, noise=0.01):
    r = np.linspace(0, 3000, 30001)
    mc = MultiCut(phases)
    q = np.geomspace(1e-3, 0.6, 400) if q is None else q
    I = levelcut_intensity(q, r, mc.covariance(sld, _g(r)), scale)
    return q, I, noise * I


@pytest.mark.parametrize("phi", [0.5, 0.3, 0.1])
def test_two_phases_reduce_to_levelcut(phi):
    mc = MultiCut(PhaseSystem(["solid", "pore"], [1 - phi, phi]))
    lc = LevelCut(phi)
    g = np.linspace(-0.99, 1.0, 101)
    assert np.allclose(mc.G([0.0, 1.0], g), lc.F(g) / lc.variance, atol=1e-6)


def test_two_cut_phase_is_roberts_formula():
    ps = PhaseSystem(list("abc"), [0.4, 0.2, 0.4])
    mc = MultiCut(ps)
    a, b = ps.levels

    def f(t):
        return (np.exp(-a * a / (1 + t)) - 2 * np.exp(-(a * a - 2 * a * b * t + b * b) / (2 * (1 - t * t)))
                + np.exp(-b * b / (1 + t))) / np.sqrt(1 - t * t)

    for g in (0.95, 0.3, -0.5):
        assert mc.covariance([0, 1, 0], [g])[0] == pytest.approx(quad(f, 0, g)[0] / (2 * np.pi), rel=1e-5)


def test_end_values_and_variance():
    ps = PhaseSystem(list("abcd"), [0.3, 0.15, 0.25, 0.3])
    mc = MultiCut(ps)
    sld = [1.0, -0.5, 2.0, 0.3]
    assert mc.covariance(sld, [1.0])[0] == pytest.approx(mc.variance(sld), rel=1e-9)
    assert mc.covariance(sld, [0.0])[0] == pytest.approx(0.0, abs=1e-12)


def test_covariance_matches_simulated_three_phase_field():
    n, a = 96, 4.0
    r = np.linspace(0, 1000, 4001)
    g = np.exp(-(r / 16.0) ** 2)
    psi, _ = field_fft(r, g, n, a, rng=7)
    ps = PhaseSystem(["oil", "film", "water"], [0.45, 0.15, 0.40])
    labels = np.digitize(psi, np.quantile(psi, np.cumsum(ps.fractions)[:-1]))
    sld = np.array([-0.4, 4.0, 6.3])
    rho = sld[labels] - sld[labels].mean()
    F = np.fft.rfftn(rho)
    corr = np.fft.irfftn(np.abs(F) ** 2, s=rho.shape, axes=(0, 1, 2)) / rho.size
    d = grid_distances(n, a)
    mc = MultiCut(ps)
    for rr in (0.0, 4.0, 8.0, 16.0, 24.0):
        sel = np.isclose(d, rr)
        assert corr[sel].mean() == pytest.approx(mc.covariance(sld, [np.exp(-(rr / 16.0) ** 2)])[0],
                                                 abs=0.03 * mc.variance(sld))


def test_single_curve_two_phase_equals_two_phase_pipeline():
    phi = 0.3
    ps = PhaseSystem(["solid", "pore"], [1 - phi, phi])
    q, I, s = _data(ps, [0.0, 1.0])
    p = MultiParameters(names=ps.names, fractions=list(ps.fractions), box_size=1200, n=64, porod_points=60)
    mres = correlations_multi([DataSet(q, I, s, [0.0, 1.0])], p)
    ores = correlations(q, I, s, MorphParameters(porosity=phi, box_size=1200, n=64, porod_points=60))
    assert np.max(np.abs(mres.g - ores.g)) < 2e-3
    assert mres.curves[0].scale == pytest.approx(ores.drho2, rel=1e-3)


def test_contrast_series_recovers_g_and_fractions():
    """Microemulsion-like oil | film | water, three SANS contrasts (bulk, film, mixed)."""
    ps = PhaseSystem(["oil", "film", "water"], [0.42, 0.16, 0.42])
    contrasts = {"bulk": [0.0, 0.0, 6.3], "film": [0.0, 4.0, 0.0], "mixed": [-0.4, 1.0, 6.3]}
    sets = [DataSet(*_data(ps, sld), sld=np.array(sld, float), name=k) for k, sld in contrasts.items()]
    p = MultiParameters(names=ps.names, fractions=[0.33, 0.34, 0.33], box_size=1200, n=64, porod_points=60,
                        fit_fractions=True)
    res = correlations_multi(sets, p)
    # the fractions are only weakly determined (the film contrast has not reached its Porod regime at
    # q = 0.6, so its invariant is 5 % off): a few hundredths
    assert np.allclose(res.multicut.phases.fractions, ps.fractions, atol=0.04)
    assert res.fit_info["end"] < 0.01 * res.fit_info["start"]
    m = res.r < 400
    assert np.max(np.abs(res.g[m] - _g(res.r[m]))) < 0.03
    assert all(c.misfit < 0.01 for c in res.curves)


def test_contrast_series_with_known_fractions():
    ps = PhaseSystem(["oil", "film", "water"], [0.42, 0.16, 0.42])
    contrasts = {"bulk": [0.0, 0.0, 6.3], "film": [0.0, 4.0, 0.0], "mixed": [-0.4, 1.0, 6.3]}
    sets = [DataSet(*_data(ps, sld), sld=np.array(sld, float), name=k) for k, sld in contrasts.items()]
    p = MultiParameters(names=ps.names, fractions=list(ps.fractions), box_size=1200, n=64, porod_points=60)
    res = correlations_multi(sets, p)
    m = res.r < 400
    assert np.max(np.abs(res.g[m] - _g(res.r[m]))) < 0.01
    assert all(c.misfit < 1e-3 for c in res.curves)
    assert all(c.chi2 < 6 for c in res.curves)            # 1 % errors; one common g for three contrasts
    assert all(abs(c.norm - 1) < 0.03 for c in res.curves)


def test_absolute_invariant_check():
    ps = PhaseSystem(["solid", "liquid", "gas"], [0.5, 0.3, 0.2])
    sld = np.array([3.5, 1.0, 0.0])            # 1e-6 / A^2
    mc = MultiCut(ps)
    abs_scale = 1e-4 * 1.0                    # I in 1/cm: C0 [1e-12 A^-4] -> 1e20 cm^-4; Q factor 1e-4 for 1/A
    q, I, s = _data(ps, sld, scale=abs_scale)
    p = MultiParameters(names=ps.names, fractions=list(ps.fractions), box_size=1200, n=64, porod_points=60,
                        sld_absolute=True, q_unit="A")
    res = correlations_multi([DataSet(q, I, s, sld)], p)
    c = res.curves[0]
    assert c.invariant / c.Q_model_abs == pytest.approx(1.0, rel=0.03)
    assert c.Q_model_abs == pytest.approx(1e-4 * 2 * np.pi ** 2 * mc.variance(sld))


def test_reconstruct_three_phases():
    ps = PhaseSystem(["oil", "film", "water"], [0.42, 0.16, 0.42])
    sets = [DataSet(*_data(ps, sld), sld=np.array(sld, float), name=nm)
            for nm, sld in (("bulk", [0, 0, 6.3]), ("film", [0, 4, 0]))]
    p = MultiParameters(names=ps.names, fractions=list(ps.fractions), box_size=600, n=96, seed=2,
                        porod_points=60)          # voxel 6.25 A resolves the film (about 13 A thick)
    res = reconstruct_multi(sets, p)
    assert res.labels.shape == (96, 96, 96) and res.labels.max() == 2
    vf = np.bincount(res.labels.ravel(), minlength=3) / res.labels.size
    assert np.allclose(vf, ps.fractions, atol=2e-3)
    # the film separates oil and water: hardly any direct oil|water contact
    S = res.statistics["interfaces"]
    assert S.get((0, 2), 0.0) < 0.15 * min(S[(0, 1)], S[(1, 2)])
    for c in res.curves:
        sel = (c.q_voxel > 0.02) & (c.q_voxel < 0.5 * np.pi / res.voxel_size)
        ratio = c.I_voxel[sel] / np.interp(c.q_voxel[sel], c.q_data, c.I_data)
        assert 0.6 < np.median(ratio) < 1.4
    assert "multi-phase" in res.report()


def test_interface_areas_of_slabs():
    lab = np.zeros((40, 40, 40), np.int8)
    lab[:, :, 10:15] = 1
    lab[:, :, 15:25] = 2
    S = interface_areas(lab, 1.0)
    assert set(S) == {(0, 1), (1, 2), (0, 2)}
    assert all(v * 1.5 == pytest.approx(40 * 40 / 40 ** 3) for v in S.values())


def test_multi_exports(tmp_path):
    import os

    from pysasfit.morph.export import write_curves_multi, write_matrix

    ps = PhaseSystem(["a", "b", "c"], [0.4, 0.2, 0.4])
    sets = [DataSet(*_data(ps, [0, 1, 2]), sld=np.array([0, 1, 2.0]), name="x/1"),
            DataSet(*_data(ps, [0, 2, 0]), sld=np.array([0, 2, 0.0]), name="film")]
    res = reconstruct_multi(sets, MultiParameters(names=ps.names, fractions=list(ps.fractions), box_size=600,
                                                  n=24, seed=1, porod_points=60))
    files = write_curves_multi(res, str(tmp_path / "m"))
    assert all(os.path.isfile(f) for f in files) and any("x_1_gamma" in f for f in files)
    raw = open(write_matrix(res, str(tmp_path / "m.vtk")), "rb").read()
    body = np.frombuffer(raw.split(b"LOOKUP_TABLE default\n", 1)[1][:24 ** 3], np.uint8)
    assert np.array_equal(body.reshape((24, 24, 24), order="F"), res.labels)
    assert np.array_equal(np.load(write_matrix(res, str(tmp_path / "m.npy"))), res.labels)
    h5py = pytest.importorskip("h5py")
    with h5py.File(write_matrix(res, str(tmp_path / "m.h5"))) as h:
        assert np.array_equal(h["labels"][()], res.labels) and "film" in h["curves"]


def test_per_curve_background():
    ps = PhaseSystem(["solid", "pore"], [0.7, 0.3])
    q, I, s = _data(ps, [0.0, 1.0])
    B = 0.5 * I[-1]
    ref = correlations_multi([DataSet(q, I, s, [0.0, 1.0])],
                             MultiParameters(names=ps.names, fractions=[0.7, 0.3], box_size=1200, n=64,
                                             porod_points=60))
    p = MultiParameters(names=ps.names, fractions=[0.7, 0.3], box_size=1200, n=64, porod_points=60)
    fixed = correlations_multi([DataSet(q, I + B, s, [0.0, 1.0], background=B)], p)
    assert fixed.curves[0].curve.B_mode == "fixed"
    assert np.allclose(fixed.g, ref.g, atol=1e-6)
    fit = correlations_multi([DataSet(q, I + B, s, [0.0, 1.0], fit_background=True)], p)
    assert fit.curves[0].curve.B == pytest.approx(B, rel=0.1)
    assert "background B" in fit.report() and "(fitted" in fit.report()
