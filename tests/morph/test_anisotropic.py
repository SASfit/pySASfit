"""Anisotropic (uniaxial) SASmorph: models A (affine) and B (Legendre) on exact synthetic 2D data."""
import os

import numpy as np
import pytest

from pysasfit.morph.anisotropic import (AnisoParameters, Data2D, _J_tail, correlations_aniso, estimate_axis,
                                        load_2d, reconstruct_aniso, spherical_transform)
from pysasfit.morph.correlation import LevelCut
from pysasfit.morph.scattering import levelcut_intensity

PHI, ELL, LAM, AXIS = 0.3, 20.0, 2.0, 30.0


def _iso():
    lc = LevelCut(PHI)
    r = np.linspace(0, 600, 12001)
    c = lc.F(np.exp(-r ** 2 / (2 * ELL ** 2)))
    qg = np.geomspace(1e-4, 1.2, 3000)
    return lc, r, c, qg, levelcut_intensity(qg, r, c, 1.0)


def _data(lam=LAM, axis=AXIS, n=140, qm=0.4):
    lc, r, c, qg, Ig = _iso()
    x = np.linspace(-qm, qm, n)
    QX, QY = np.meshgrid(x, x)
    t = np.radians(axis)
    qp, qs = QX * np.cos(t) + QY * np.sin(t), -QX * np.sin(t) + QY * np.cos(t)
    I = lam * np.interp(np.sqrt((lam * qp) ** 2 + qs ** 2), qg, Ig)
    s = 0.01 * I + 1e-6 * I.max()
    m = np.hypot(QX, QY) > 0.006
    return Data2D(QX[m], QY[m], I[m], s[m]), lc, r, c


@pytest.fixture(scope="module")
def synth():
    return _data()


def _gamma_true(lc, r, c, rp, rs, lam=LAM):
    return np.interp(np.sqrt((np.asarray(rp) / lam) ** 2 + np.asarray(rs) ** 2), r, c / lc.variance)


def test_tail_constant_and_transform():
    from scipy.special import gamma as G
    assert _J_tail(2, 0.0)[0] == pytest.approx(np.pi / 16, rel=1e-12)
    assert _J_tail(4, 0.0)[0] == pytest.approx(np.sqrt(np.pi) / 8 * G(1.5) / G(4), rel=1e-12)
    # the tail of a pure Porod l = 2 component: transform from q_N on equals the analytic tail
    qn = np.array([0.0, 1e-3])
    r = np.array([5.0, 20.0])
    t = spherical_transform(qn, np.zeros(2), 2, r, K=1.0)
    from scipy.integrate import quad
    from scipy.special import spherical_jn
    for k, rr in enumerate(r):
        ref = quad(lambda x: spherical_jn(2, x) / x ** 2, 1e-3 * rr, 400, limit=4000)[0] + 400 ** -2 * 0  # tail beyond 400 ~ 1e-6
        assert t[k] == pytest.approx(-rr * ref / (2 * np.pi ** 2), rel=2e-3)


def test_axis_estimate(synth):
    d, *_ = synth
    phi_max, an = estimate_axis(d)
    assert abs(((phi_max - (AXIS + 90)) + 90) % 180 - 90) < 4 and an > 0.3


@pytest.mark.parametrize("mode", ["A", "B"])
def test_gamma_recovered(synth, mode):
    d, lc, r, c = synth
    p = AnisoParameters(mode=mode, axis_deg=AXIS + (4 if mode == "A" else 0), porosity=PHI, box_size=600, n=64,
                        lmax=8, seed=1)
    res = correlations_aniso(d, p)
    if mode == "A":
        assert res.stretch == pytest.approx(LAM, rel=2e-3) and res.axis_deg == pytest.approx(AXIS, abs=0.2)
    pts = [(0, 10), (10, 0), (30, 0), (0, 30), (40, 40), (80, 10)]
    for rp, rs in pts:
        assert res.gamma(np.array([rp]), np.array([rs]))[0] == pytest.approx(_gamma_true(lc, r, c, rp, rs), abs=0.025)
    assert res.drho2 == pytest.approx(1.0, rel=0.03)
    assert res.clipped == 0 and res.chi2_levelcut < 20
    assert "uniaxial" in res.report() or "affine" in res.report()


def test_isotropic_data_have_no_higher_orders():
    d, lc, r, c = _data(lam=1.0)
    res = correlations_aniso(d, AnisoParameters(mode="B", axis_deg=0.0, porosity=PHI, box_size=600, n=64, lmax=6))
    assert np.max(np.abs(res.C_l[1:])) < 0.01
    assert np.max(np.abs(res.I_l[:, 1:] / res.I_l[:, :1])) < 0.02


def test_reconstruction_is_stretched(synth):
    d, *_ = synth
    res = reconstruct_aniso(d, AnisoParameters(mode="B", axis_deg=AXIS, porosity=PHI, box_size=600, n=64, lmax=8,
                                               seed=3))
    assert res.matrix.shape == (64, 64, 64) and res.matrix.mean() == pytest.approx(PHI, abs=1e-3)
    ch = res.statistics["chords"]["pore"]
    assert ch[0] / (0.5 * (ch[1] + ch[2])) == pytest.approx(LAM, rel=0.2)       # stretched along the axis (x)
    ff = res.statistics["face_fractions"]
    assert ff[0] < 0.27 < ff[1] and ff[1] == pytest.approx(ff[2], rel=0.1)
    assert res.negative_power < 0.02
    assert res.statistics["S_over_V_voxels"] == pytest.approx(res.S_over_V_porod, rel=0.15)
    assert res.q_voxel is not None and res.I_l_voxel.shape[1] == len(res.ls)


def test_load_2d_formats(tmp_path, synth):
    d, *_ = synth
    p = tmp_path / "a.npz"
    np.savez(p, qx=d.qx, qy=d.qy, I=d.I, sigma=d.sigma)
    e = load_2d(p)
    assert e.I.size == d.I.size and np.allclose(e.qx, d.qx)
    t = tmp_path / "a.dat"
    np.savetxt(t, np.column_stack([d.qx, d.qy, d.I, d.sigma]))
    assert load_2d(t).I.size == d.I.size
    import h5py
    h = tmp_path / "a.h5"
    n = 40
    x = np.linspace(-0.2, 0.2, n)
    QX, QY = np.meshgrid(x, x)
    with h5py.File(h, "w") as f:
        g = f.create_group("sasentry01/sasdata01")
        g["I"] = np.ones((n, n))
        g["Idev"] = 0.1 * np.ones((n, n))
        g["Qx"], g["Qy"] = QX, QY
        g["Qx"].attrs["units"] = "1/nm"
        mask = np.zeros((n, n), np.int8)
        mask[0, :] = 1
        g["Mask"] = mask
    e = load_2d(h)
    assert e.I.size == n * n - n - (1 if (np.hypot(QX, QY)[1:] == 0).any() else 0) and e.q_unit == "nm"


def test_gui_window(synth, tmp_path):
    pytest.importorskip("PySide6")
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6 import QtWidgets
    QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    from pysasfit.morph.aniso_gui import AnisoWindow
    from pysasfit.morph.gui import MainWindow
    d, *_ = synth
    p = tmp_path / "synth.npz"
    np.savez(p, qx=d.qx, qy=d.qy, I=d.I, sigma=d.sigma)
    main = MainWindow()
    w = main.on_anisotropic()
    assert isinstance(w, AnisoWindow)
    w.load_file(str(p))
    assert w.data is not None and abs(((w.axis.value() - AXIS) + 90) % 180 - 90) < 4
    w.n.setValue(48)
    w.box_size.setText("600")
    w.porosity.setValue(PHI)
    w.mode.setCurrentIndex(w.mode.findData("A"))
    res = w.run(True)
    assert res is not None and res.matrix.shape == (48, 48, 48) and res.stretch == pytest.approx(LAM, rel=0.01)
    assert "affine" in w.results_text.toPlainText()
    for f in (w.fig_2d, w.fig_l, w.fig_c, w.fig_s):
        assert f.axes
    w.close()
    main.close()
