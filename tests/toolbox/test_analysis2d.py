"""2D analysis (radial, sector, azimuthal, box, A + B sin²ψ, sums, rocking) and 1D arithmetic."""
import numpy as np
import pytest

from pysasfit.toolbox import analysis2d as an
from pysasfit.toolbox import data1d as d1
from pysasfit.toolbox.image2d import Image2D


def image(A=1.0, B=2.0, psi0=30.0, n=96, noise=0.0, seed=1, omega=None):
    im = Image2D(np.zeros((n, n)), np.full((n, n), 0.05), np.zeros((n, n), bool), "synthetic.h5", "test",
                 {"sample": "S", **({} if omega is None else {"hdf:sample/omega": omega})}, unit="1/cm/sr",
                 centre_px=(47.5, 47.5), pixel_m=(0.0075, 0.0075), SD_m=4.0, wavelength_A=6.0)
    psi = np.radians(im.psi_deg() - psi0)
    rng = np.random.default_rng(seed)
    im.I = A + B * np.sin(psi) ** 2 + noise * rng.normal(size=(n, n))
    im.mask[44:52, 44:52] = True                       # beam stop
    return im


def test_sin2_and_averages():
    im = image()
    a, b, chi = an.sin2_decomposition(im, n=10, spacing="lin", psi0=30.0)
    assert np.allclose(a.y, 1.0, atol=1e-6) and np.allclose(b.y, 2.0, atol=1e-6) and np.all(a.dy > 0)
    r = an.radial_average(im, n=20, spacing="lin")
    assert r.is_q and np.allclose(r.y, 2.0, atol=0.05)              # <sin²> = 1/2 over full rings
    s0 = an.radial_average(im, n=20, spacing="lin", sector=(30.0, 5.0, True))   # along ψ0: sin² ≈ 0
    s90 = an.radial_average(im, n=20, spacing="lin", sector=(120.0, 5.0, True))
    assert np.all(s0.y < 1.05) and np.all(s90.y > 2.9)
    az = an.azimuthal_average(im, 0.3, 0.8, n=36)
    assert az.xlabel == "ψ" and az.y[np.argmin(np.abs(az.x - 30))] < 1.05 and az.y[np.argmin(np.abs(az.x - 120))] > 2.9
    bx = an.box_profile(im, 30.0, 0.8, 0.05, n=20)                   # along ψ0: I ≈ A
    assert np.allclose(bx.y, 1.0, atol=0.05) and bx.x.min() < 0 < bx.x.max()
    assert np.all(np.isfinite(a.dy)) and a.dy[0] < 0.02


def test_sums_rocking_shapes():
    ims = [image(A=1.0 + k, omega=0.1 * k) for k in range(3)]
    s = an.sum_images(ims)
    assert np.allclose(s.I[~s.mask], sum(im.I for im in ims)[~s.mask]) and s.dI[0, 0] == pytest.approx(0.05 * 3 ** 0.5)
    m = an.sum_images(ims, mean=True)
    assert np.allclose(m.I, s.I / 3)
    rc = an.rocking_curve(ims[::-1], "hdf:sample/omega", q1=0.2, q2=0.6)
    assert np.allclose(rc.x, [0, 0.1, 0.2]) and np.allclose(np.diff(rc.y), 1.0, atol=1e-6)
    im = ims[0]
    sel = an.pixels_in_shape(im, "pixels", "rectangle", (0, 9.5, 0, 9.5))
    assert sel.sum() == 100
    sel = an.pixels_in_shape(im, "polar", "rectangle", (0, 90, 0, 10))       # first quadrant
    assert 0.2 < sel.mean() < 0.3
    sel = an.pixels_in_shape(im, "pixels", "polygon", [(0, 0), (10, 0), (0, 10)])
    assert 40 < sel.sum() < 70
    assert an.pixels_in_shape(im, "pixels", "above", 2.5).sum() > 0
    g = np.zeros((50, 60))
    g[20:25, 30:35] = 100
    assert an.beam_centre_of_gravity(g) == pytest.approx((32.0, 22.0))


def test_data1d_arithmetic(tmp_path):
    x = np.linspace(0.1, 1, 10)
    a = d1.Data1D(x, 2 * x, 0.1 * np.ones(10), name="a", yunit="1/cm")
    b = d1.Data1D(x, x, 0.1 * np.ones(10), name="b", yunit="1/cm")
    assert np.allclose(a.scale(3).y, 6 * x) and np.allclose(a.scale(-3).dy, 0.3)
    assert np.allclose(a.divide(2).y, x) and np.allclose(a.offset(1).y, 2 * x + 1)
    c = d1.combine(a, b, "-")
    assert np.allclose(c.y, x) and np.allclose(c.dy, np.hypot(0.1, 0.1))
    r = d1.combine(a, b, "/")
    assert np.allclose(r.y, 2.0) and r.yunit == ""
    f = d1.combine(a, b, "a-f*b", f=2.0)
    assert np.allclose(f.y, 0.0)
    p = d1.write_ascii(tmp_path / "c.dat", c)
    back = d1.read_data1d(p)[0]
    assert np.allclose(back.y, c.y) and back.name == c.name
    az = d1.Data1D(np.arange(5.0), np.ones(5), None, name="psi", xlabel="ψ", xunit="deg")
    with pytest.raises(ValueError):
        d1.combine(a, az)


RAW = __import__("os").environ.get("PYSASRED_TEST_RAW", "/mnt/user-data/uploads/Sept2026/raw")


@pytest.mark.skipif(not __import__("os").path.isdir(RAW), reason="SANS-I raw data not available")
def test_gui_viewer_to_2d_to_1d(tmp_path):
    import os
    pytest.importorskip("PySide6")
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6 import QtWidgets
    QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    from pysasfit.toolbox.gui import MainWindow
    w = MainWindow()
    v, t, d = w.viewer, w.images, w.data1d
    # HDF5 viewer → 2D data: geometry from the SANS-I raw file, centre of gravity of a direct beam
    v.select_path(os.path.join(RAW, "sans2026n082173.hdf"), "/entry1/SANS/detector/counts")
    v.on_to_2d()
    assert w.tabs.currentWidget() is t and len(t.images) == 1
    eb = t.images[0]
    assert eb.has_q and eb.SD_m > 8.0 and eb.centre_px == pytest.approx((61.7, 63.9), abs=0.4)
    v.select_path(os.path.join(RAW, "sans2026n082130.hdf"), "/entry1/SANS/detector/counts")
    v.on_to_2d()
    im = t.images[-1]
    assert im.meta["sample"] == "EXP-005370-A_1" and im.unit == "counts"
    t.list.selectRow(1)
    t._set_centre(*eb.centre_px)
    # mask: the beam stop by a rectangle in pixels, then undo / redo by hand
    t.view.setCurrentIndex(t.view.findData("pixels"))
    t.replot()
    t._apply_shape("rectangle", (55, 69, 57, 71))
    assert im.mask.sum() == 15 * 15
    t.on_mask_undo()
    assert im.mask.sum() == 0
    t._apply_shape("rectangle", (55, 69, 57, 71))
    # every average → 1D data
    for kind, n in (("radial", 1), ("sector", 1), ("azimuthal", 1), ("box", 1), ("sin2", 2)):
        t.avg_kind.setCurrentIndex(t.avg_kind.findData(kind))
        t.show_region.setChecked(True)
        t.view.setCurrentIndex(t.view.findData("q"))
        t.replot()
        out = t.on_average()
        assert len(out) == n and all(len(o.x) > 3 for o in out)
    assert w.tabs.currentWidget() is d and len(d.data) == 6
    # sum of the two images, rocking curve over a header item
    t.list.selectAll()
    s = t.on_sum()
    assert s is not None and np.allclose(s.I, eb.I + im.I)
    t.list.selectRow(0)
    t.list.selectAll()
    t.rock_key.setCurrentText("counting_time_s")
    rc = t.on_rocking()
    assert rc is not None and len(rc.x) >= 2
    # 1D arithmetic in the GUI
    d.table.selectRow(0)
    d.value.setText("2")
    d.on_scalar("scale")
    assert np.allclose(d.data[-1].y, 2 * d.data[0].y)
    d.a_combo.setCurrentIndex(0)
    d.b_combo.setCurrentIndex(1)
    d.op.setCurrentIndex(d.op.findData("/"))
    d.on_binary()
    assert d.data[-1].name.startswith("(")
    d.table.selectRow(len(d.data) - 1)
    files = d.on_save(folder=str(tmp_path))
    assert len(files) == 1 and os.path.isfile(files[0])
    w.close()
