"""Masks: standard border, suggested beamstop / hot pixels, shapes, files, and the editor."""
import os

import numpy as np
import pytest

from pysasfit.sasred.core import mask as mk

RAW = os.environ.get("PYSASRED_TEST_RAW", "/mnt/user-data/uploads/Sept2026/raw")
need = pytest.mark.skipif(not os.path.isdir(RAW), reason="Vifor raw data not available")


def _image(seed=3):
    """Synthetic SANS image: smooth scattering, a 9 x 7 beamstop shadow at (40, 30), one hot pixel."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[:64, :80]
    r = np.hypot(xx - 40.3, yy - 30.6)
    lam = 2000.0 / (1 + (r / 6) ** 2) + 5
    shadow = (np.abs(xx - 40) <= 4) & (np.abs(yy - 30) <= 3)
    lam[shadow] = 8.0
    c = rng.poisson(lam)
    c[50, 10] = 5000
    return c, shadow


def test_border():
    m = mk.border_mask((10, 12), 2)
    assert m[0].all() and m[1].all() and m[-2:].all() and m[:, :2].all() and m[:, -2:].all()
    assert not m[2:-2, 2:-2].any() and mk.summary(m)["border"] == 10 * 12 - 6 * 8


def test_suggest_beamstop_and_hot():
    c, shadow = _image()
    m, rep = mk.suggest_mask(c, (40.3, 30.6))
    bs = (m & mk.BEAMSTOP) != 0
    assert np.all(bs[shadow])                                # the whole shadow
    assert bs.sum() <= (9 + 2) * (7 + 2) + 4                 # plus at most the 1-pixel penumbra
    assert m[50, 10] & mk.HOT and rep["hot"] == 1
    assert (m[:2] & mk.BORDER).all()
    # direct beam (no shadow): no beamstop, and nothing flagged near the beam centre
    yy, xx = np.mgrid[:64, :80]
    db = np.random.default_rng(1).poisson(1e4 * np.exp(-((xx - 40) ** 2 + (yy - 30) ** 2) / 4) + 0.5)
    m2, rep2 = mk.suggest_mask(db, (40, 30))
    assert not (m2 & (mk.BEAMSTOP | mk.HOT)).any() and "no sudden drop" in rep2["beamstop"]["reason"]


def test_shapes():
    s = (20, 30)
    r = mk.pixels_in_rectangle(s, 2.4, 5.6, 7.0, 3.0)
    assert r.sum() == 5 * 5 and r[3:8, 2:7].all()                  # every pixel touched
    d = mk.pixels_in_disc(s, 10, 10, 0.5)
    assert d.sum() == 1 and d[10, 10]
    e = mk.pixels_in_ellipse(s, 0, 20, 5, 15)
    assert e[10, 10] and not e[5, 0]
    p = mk.pixels_in_polygon(s, [(0.5, 0.5), (4.5, 0.5), (4.5, 2.5), (0.5, 2.5)])
    assert p[1:3, 1:5].all() and p.sum() == 5 * 3                  # inside and under the edges


@pytest.mark.parametrize("ext", [".h5", ".npy", ".txt"])
def test_mask_files(tmp_path, ext):
    pytest.importorskip("h5py")
    m = mk.border_mask((16, 20))
    m[5:8, 6:9] |= mk.USER
    m[10, 10] = mk.HOT
    p = mk.save_mask(str(tmp_path / f"m{ext}"), m, beam_center_px=(9.5, 8.25), run=82130)
    back, info = mk.load_mask(p, shape=(16, 20))
    if ext == ".txt":
        np.testing.assert_array_equal(back != 0, m != 0)       # 0/1 only
    else:
        np.testing.assert_array_equal(back, m)
    if ext == ".h5":
        assert info["beam_center_px"] == (9.5, 8.25) and info["run"] == "82130"
    with pytest.raises(ValueError):
        mk.load_mask(p, shape=(16, 21))


# ---------------------------------------------------------------- editor in the GUI
def _app():
    pytest.importorskip("PySide6")
    pytest.importorskip("h5py")
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6 import QtWidgets
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


@need
def test_editor_suggest_edit_undo_files(tmp_path):
    app = _app()
    from pysasfit.sasred.gui import MainWindow
    w = MainWindow()
    ed = w.mask_editor
    w.set_folder(RAW)
    w.select_runs([82130])
    app.processEvents()
    s0 = mk.summary(ed.mask)
    assert ed.mask is not None and ed.pristine
    r = w.bs_radius.value()
    guess = mk.beamstop_guess(ed.mask.shape, (w.xc.value(), w.yc.value()), r, "square")
    assert s0["beamstop"] == guess.sum() >= (2 * int(r)) ** 2      # border + square first guess
    w.xc.setValue(61.73)
    w.yc.setValue(63.93)
    ys, xs = np.nonzero(ed.mask & mk.BEAMSTOP)                      # the untouched guess follows the centre
    assert abs(xs.mean() - 61.73) < 1 and abs(ys.mean() - 63.93) < 1
    ed.apply(mk.pixels_in_rectangle(ed.mask.shape, 100, 104, 100, 104), add=True)   # user pixels first
    ed.suggest()
    s = mk.summary(ed.mask)
    assert 120 < s["beamstop"] < 300 and s["user"] == 25            # shadow found, user pixels kept
    ys, xs = np.nonzero(ed.mask & mk.BEAMSTOP)                      # suggested as a rectangle
    assert s["beamstop"] == (xs.max() - xs.min() + 1) * (ys.max() - ys.min() + 1)
    assert not ed.pristine
    w.xc.setValue(62.5)                                             # an edited mask does not move
    assert (np.nonzero(ed.mask & mk.BEAMSTOP)[1] == xs).all()
    full = w.full_mask(ed.mask.shape)
    assert (full == ed.mask).all()
    w.update_preview()
    assert w.preview is not None
    ed.apply(mk.pixels_in_rectangle(ed.mask.shape, 0, 127, 60, 67), add=False)     # remove a band
    assert not ed.mask[60:68].any()
    ed.undo()
    assert ed.mask[60:68].any()
    ed.redo()
    assert not ed.mask[60:68].any()
    ed.undo()
    p = mk.save_mask(str(tmp_path / "m.h5"), ed.mask)
    saved = ed.mask.copy()
    ed.reset()
    guess = mk.beamstop_guess(ed.mask.shape, (w.xc.value(), w.yc.value()), r, "square")
    assert mk.summary(ed.mask)["beamstop"] == guess.sum() and ed.pristine
    assert ed.load(p) and (ed.mask == saved).all()
    # every tool can be activated on the detector plot
    for i in range(ed.tool.count()):
        ed.tool.setCurrentIndex(i)
        app.processEvents()
    # brush: simulated drag
    ed.tool.setCurrentIndex(ed.tool.findData("+brush"))

    class E:
        def __init__(self, x, y):
            self.inaxes, self.xdata, self.ydata, self.button = ed._ax, x, y, 1
    before = int(np.count_nonzero(ed.mask & mk.USER))
    ed._paint(E(20.0, 100.0), True, start=True)
    ed._paint(E(24.0, 100.0), True)
    ed._paint_end(None)
    assert np.count_nonzero(ed.mask & mk.USER) > before
    w.det_view.setCurrentIndex(w.det_view.findData("polar"))      # mask shown in the polar view
    app.processEvents()
    w.close()


@need
def test_editor_mouse_rectangle_and_polygon():
    """Real matplotlib mouse events on the detector plot."""
    app = _app()
    from matplotlib.backend_bases import MouseEvent
    from pysasfit.sasred.gui import MainWindow
    w = MainWindow()
    w.resize(1200, 800)
    w.show()
    w.set_folder(RAW)
    w.select_runs([82140])
    ed = w.mask_editor
    c = w.det_canvas

    def ev(kind, xd, yd):
        x, y = ed._ax.transData.transform((xd, yd))
        MouseEvent(kind, c, x, y, 1)._process()

    ed.tool.setCurrentIndex(ed.tool.findData("+rect"))
    app.processEvents()
    c.draw()
    for kind, p in (("button_press_event", (100, 20)), ("motion_notify_event", (110, 30)),
                    ("button_release_event", (110, 30))):
        ev(kind, *p)
    n_rect = mk.summary(ed.mask)["user"]
    assert 100 <= n_rect <= 121
    ed.tool.setCurrentIndex(ed.tool.findData("+polygon"))
    app.processEvents()
    c.draw()
    for p in [(20, 100), (40, 100), (40, 115), (20, 100)]:
        ev("motion_notify_event", *p)
        ev("button_press_event", *p)
        ev("button_release_event", *p)
    app.processEvents()
    assert mk.summary(ed.mask)["user"] > n_rect + 100
    ed.tool.setCurrentIndex(ed.tool.findData("-rect"))
    app.processEvents()
    c.draw()
    for kind, p in (("button_press_event", (95, 15)), ("motion_notify_event", (115, 35)),
                    ("button_release_event", (115, 35))):
        ev(kind, *p)
    assert not ed.mask[20:31, 100:111].any()
    w.close()



def test_beamstop_guess_and_rectangle_suggestion():
    sq = mk.beamstop_guess((20, 30), (10.2, 9.7), 3, "square")
    assert sq.sum() == 36 and sq[9, 10] and not sq[9, 14]          # pixel centres within ±3 px
    disc = mk.beamstop_guess((20, 30), (10, 10), 3, "disc")
    assert disc.sum() == 29
    assert not mk.beamstop_guess((20, 30), (10, 10), 0).any()
    # a round shadow: rectangle = its bounding box, shadow = the pixels themselves
    yy, xx = np.mgrid[:80, :80]
    img = np.full((80, 80), 1000.0)
    img[np.hypot(xx - 40, yy - 40) < 6] = 5.0
    rect, info = mk.suggest_beamstop(img, (40, 40), grow_px=0)
    shad, _ = mk.suggest_beamstop(img, (40, 40), grow_px=0, shape="shadow")
    ys, xs = np.nonzero(shad)
    assert info["shape"] == "rectangle" and rect.sum() == (np.ptp(xs) + 1) * (np.ptp(ys) + 1) > shad.sum()


def test_colour_range_from_unmasked_pixels():
    from pysasfit.core.colorscale import make_norm
    img = np.ones((10, 10))
    img[5, 5] = 1e6                                 # bright pixel next to a badly placed beamstop
    valid = np.ones((10, 10), bool)
    valid[5, 5] = False
    assert make_norm("linear", img).vmax == 1e6
    assert make_norm("linear", img, valid=valid).vmax <= 2
    assert make_norm("log1p", img, valid=~valid).vmin == 1e6   # the masked pixels on their own scale


@need
def test_masked_pixel_display_modes():
    app = _app()
    from pysasfit.sasred.gui import MainWindow
    w = MainWindow()
    w.set_folder(RAW)
    w.select_runs([82130])
    app.processEvents()
    ed = w.mask_editor
    for mode in ("colours", "scale", "hidden"):
        ed.display.setCurrentIndex(ed.display.findData(mode))
        for view in ("cartesian", "polar"):
            w.det_view.setCurrentIndex(w.det_view.findData(view))
            w._draw_detector()
            axes = [c for a in w.det_canvas.figure.axes for c in [a, *a.child_axes]]
            labels = [a.get_ylabel() for a in axes]
            assert ("masked pixels" in labels) == (mode == "scale")
    w.close()


def test_shapes_include_start_and_end_pixels():
    """Drawing selects every pixel touched, including the pixel where the drag starts."""
    s = (20, 30)
    r = mk.pixels_in_rectangle(s, 10.3, 20.2, 5.4, 7.6)            # from inside pixel (10, 5) to (20, 8)
    assert r[5, 10] and r[8, 20] and r.sum() == 11 * 4
    line = mk.pixels_in_rectangle(s, 3.2, 15.4, 9.1, 9.3)          # a horizontal line within row 9
    assert line.sum() == 13 and line[9].sum() == 13
    one = mk.pixels_in_rectangle(s, 4.4, 4.4, 2.6, 2.6)            # a click
    assert one.sum() == 1 and one[3, 4]
    assert mk.pixels_in_rectangle(s, 20.2, 10.3, 7.6, 5.4).sum() == 44   # dragged the other way
    e = mk.pixels_in_ellipse(s, 3.2, 15.4, 9.1, 9.3)               # a flat ellipse in one row
    assert e[9].sum() == 13 and e.sum() == 13
    e2 = mk.pixels_in_ellipse(s, 5, 15, 2, 12)
    assert e2[7, 5] and e2[7, 15] and e2[2, 10] and e2[12, 10] and not e2[2, 5]
    p = mk.pixels_in_polygon(s, [(2.3, 4.2), (12.4, 4.4), (12.4, 4.45)])   # a thin polygon along row 4
    assert p[4, 2] and p[4, 12] and p[4].sum() == 11
