"""Scale bar and size box of the 3D panel (pyATsas, pyDENsity): geometry without OpenGL, widget state."""

import os

import numpy as np
import pytest

from pysasfit.atsas.view3d import nice_length, scale_guides


def cuboid(a, b, c, n=9):
    """Surface points of an a × b × c box (Å), centred at the origin."""
    g = np.linspace(-0.5, 0.5, n)
    X, Y, Z = np.meshgrid(g, g, g, indexing="ij")
    p = np.c_[X.ravel(), Y.ravel(), Z.ravel()]
    p = p[np.any(np.abs(p) == 0.5, axis=1)]
    return p * [a, b, c]


def rotation(ax, ay):
    cx, sx, cy, sy = np.cos(ax), np.sin(ax), np.cos(ay), np.sin(ay)
    return np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]]) @ np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])


def test_nice_length():
    assert [nice_length(x) for x in (0.7, 1, 1.9, 3, 7, 23, 480, 1000)] == [0.5, 1, 1, 2, 5, 20, 200, 1000]
    assert nice_length(0) == 0 and nice_length(np.nan) == 0


def test_scale_bar_has_a_round_length_in_the_shown_unit():
    v = cuboid(2000, 600, 300)                         # 200 nm long
    segs, labels, info = scale_guides(v, bar=True, box=False)
    assert segs.shape == (6, 3)                        # bar + 2 end ticks
    assert np.linalg.norm(segs[1] - segs[0]) == pytest.approx(500.0)   # 0.4 × 200 nm -> 50 nm
    assert labels[-1][1] == "50 nm" and info == "scale bar 50 nm"
    assert segs[:, 2].max() < v[:, 2].min()            # below the object (z is up)
    _s, labels, _i = scale_guides(v, bar=True, unit="Å")
    assert labels[-1][1] == "500 Å"
    _s, labels, _i = scale_guides(cuboid(40, 30, 20), bar=True)         # small: auto gives Å
    assert labels[-1][1] == "10 Å"


def test_box_along_the_axes_gives_the_extents():
    segs, labels, info = scale_guides(cuboid(2000, 600, 300) + 50, bar=False, box=True)
    assert segs.shape == (24, 3)                       # 12 edges
    assert [t for _p, t in labels] == ["200 nm", "60 nm", "30 nm"]
    assert info == "x 200 nm, y 60 nm, z 30 nm"
    assert segs.min(axis=0) == pytest.approx([-950, -250, -100])


def test_principal_box_does_not_depend_on_the_orientation():
    v = cuboid(2000, 600, 300, n=15) @ rotation(0.6, -0.4).T + [10, -20, 30]
    _s, labels, info = scale_guides(v, bar=False, box=True, box_axes="principal")
    assert [t for _p, t in labels] == ["200 nm", "60 nm", "30 nm"]
    assert info.startswith("length 200 nm, width 60 nm, thickness 30 nm")
    _s, labels, _i = scale_guides(v, bar=False, box=True, box_axes="xyz")
    assert labels[0][1] != "200 nm"                    # the rotated box is larger along x


def test_nothing_to_show():
    segs, labels, info = scale_guides(np.zeros((0, 3)), bar=True, box=True)
    assert segs.shape == (0, 3) and labels == [] and info == ""


def test_model_view_guides_and_state():
    pytest.importorskip("PySide6")
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6 import QtWidgets

    QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    from pysasfit.atsas.view3d import ModelView

    w = ModelView()
    v = cuboid(2000, 600, 300)
    faces = np.array([[0, 1, 2]], np.int32)
    w.add_mesh("map", v, faces)
    assert w.guides()[1] == []                         # off by default
    w.set_guide_state({"bar": True, "box": True, "box_axes": "principal", "unit": "nm"})
    assert w.guide_state() == {"bar": True, "box": True, "box_axes": "principal", "unit": "nm"}
    segs, labels, info = w.guides()
    if w._overlay_cls() is None:                       # no OpenGL / old pyqtgraph: the bar is drawn in 3D
        assert len(segs) == 30 and "scale bar 50 nm" in info
    else:                                              # the bar is an overlay in the lower left corner
        assert len(segs) == 24 and len(labels) == 3
        assert w.overlay_bar is not None and "scale bar: lower left" in w.info.text()
        L, text, px = w.overlay_bar.bar(800, 0.5)      # 800 px wide, 0.5 px/Å: 20 % = 320 Å -> 20 nm
        assert (L, text, px) == (200.0, "20 nm", 100.0)
    w.set_guide_state(None)                            # an old session: everything off
    assert w.guide_state()["bar"] is False
    w.deleteLater()


def test_bar_lies_along_the_screen_and_labels_at_the_near_corner():
    from pysasfit.atsas.view3d import camera_direction

    v = cuboid(2000, 600, 300)
    for az in (35.0, 120.0, -60.0):
        d = camera_direction(az, 20.0)
        segs, labels, _i = scale_guides(v, bar=True, box=True, view_dir=d)
        bar = segs[-6:]
        along = bar[1] - bar[0]
        assert abs(along @ d) < 1e-9 and abs(along[2]) < 1e-9      # horizontal, across the line of sight
        for pos, _t in labels[:3]:                                   # box labels on the side facing the camera
            assert pos @ d > 0
        assert bar[:, 2].max() < segs[:-6, 2].min()                  # the bar below the box
