"""Lattice sketch (latview): block geometry, reciprocal lattice, reflections on the detector vs. the 2D pattern."""
import numpy as np
import pytest

from pysasfit.scatter import engine, latview, params


def _p(**kw):
    p = params.defaults()
    p.update(ComboBoxParticle=0, EditRadius=3, EditSigma=0.08, uca=20, ucb=20, ucc=20, ucalpha=90, ucbeta=90,
             ucgamma=90, HKLmax=3, EditDomainSize=150, Ordis=6, ucn1=1, ucn2=1, ucn3=1, ucpsi=0,
             EditQmax=1.5, EditQmaxPreset=True, EditDebyeWaller=0.5)
    p.update(kw)
    return p


def _nn(c):
    d = np.linalg.norm(c[:, None] - c[None], axis=-1)
    return d[d > 1e-9].min()


@pytest.mark.parametrize("lt, n, nn", [(7, 27, 20.0), (4, 35, 20 * np.sqrt(3) / 2), (5, 63, 20 / np.sqrt(2)),
                                       (17, 95, 20 * np.sqrt(3) / 4)])
def test_block_counts_and_neighbour_distance(lt, n, nn):
    sc = latview.build_scene(_p(LType=lt), ncell=2)
    assert len(sc.centres) == n
    assert _nn(sc.centres) == pytest.approx(nn, rel=1e-9)
    assert np.allclose(sc.centres.mean(0), 0, atol=1e-9)           # block centred


def test_zone_axis_111_reflections_in_detector_plane():
    sc = latview.build_scene(_p(LType=5))
    on = sc.hkl[:, sc.on_detector]
    assert on.shape[1] > 0 and np.all(on.sum(0) == 0)              # [111] along the beam: h + k + l = 0
    assert np.allclose(sc.recip.T @ sc.cell, 2 * np.pi * np.eye(3))


def test_ewald_sphere_excitation():
    sc = latview.build_scene(_p(LType=7, EditQmaxPreset=False, EditWavelength=0.6))
    k = 2 * np.pi / 0.6
    G = sc.G
    assert np.allclose(sc.excitation, np.linalg.norm(G + np.array([[k], [0], [0]]), axis=0) - k)


def test_predicted_spots_are_peaks_of_the_pattern():
    p = _p(LType=5, GridPoints=96)
    sc = latview.build_scene(p)
    refl = latview.detector_reflections(sc)
    assert len(refl) == 6
    qx, qy, qz = engine.q_grid_2d(p)
    I = engine.intensity_2d(p, qx, qy, qz)
    step = qy[1, 0] - qy[0, 0]
    for r in refl:
        i, j = int(round((r["qh"] - qy[0, 0]) / step)), int(round((r["qv"] - qz[0, 0]) / step))
        win = I[i - 3:i + 4, j - 3:j + 4]
        ring = np.median(I[i - 8:i + 9, j - 8:j + 9])
        assert win.max() > 20 * ring                                 # a Bragg peak sits at the predicted spot


def test_powder_rings_and_other_kinds():
    sc = latview.build_scene(_p(LType=5, Ordis=7))
    rings = latview.detector_reflections(sc)
    assert not sc.oriented and rings[0]["hkl"] == (1, 1, 1)
    assert rings[0]["q"] == pytest.approx(2 * np.pi * np.sqrt(3) / 20)
    hexc = latview.build_scene(_p(LType=1, ComboBoxParticle=1, Length=40, ucgamma=120), axes_mode="lattice")
    assert hexc.dim == 2 and np.allclose(np.abs(hexc.axes[:, 2]), 1)       # rods vertical, as in the engine
    lam = latview.build_scene(_p(LType=0, ComboBoxParticle=2, EditRadius=2, Length=50, uca=12))
    assert lam.slab is not None and len(lam.slab["positions"]) >= 3
    gy = latview.build_scene(_p(LType=9, uca=30, ucb=30, ucc=30))
    F, to_lab = latview.bicontinuous_field(gy, 8)
    assert F.shape == (17, 17, 17) and F.min() < 0 < F.max()


def test_meshes():
    for spec in ({"type": "sphere", "radius": 2, "shell": 3}, {"type": "cylinder", "radius": 1, "half_length": 5},
                 {"type": "ellipsoid", "semi": (1, 2, 3)}, {"type": "box", "half": (1, 1, 1)},
                 {"type": "superquadric", "semi": (1, 1, 1), "exponent": 4}):
        V, F = latview.particle_mesh(spec)
        assert V.ndim == 2 and F.max() < len(V)
    V, F = latview.place(latview.particle_mesh({"type": "cylinder", "radius": 1, "half_length": 5}),
                         np.zeros((2, 3)), np.array([[1.0, 0, 0], [0, 0, 1.0]]))
    assert np.ptp(V[:len(V) // 2, 0]) == pytest.approx(10) and np.ptp(V[len(V) // 2:, 2]) == pytest.approx(10)


def test_panel_detector_view_and_overlay():
    pytest.importorskip("PySide6")
    from PySide6 import QtWidgets
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    from pysasfit.scatter.latview_gui import LatticeView, draw_reflections
    w = LatticeView()
    w.view_mode.setCurrentIndex(w.view_mode.findData("det"))
    w.set_params(_p(LType=5))
    w.rebuild()
    app.processEvents()
    assert "6 reflections on the detector" in w.info.text()
    assert w.state()["view"] == "det"
    from matplotlib.figure import Figure
    ax = Figure().add_subplot(111)
    assert len(draw_reflections(ax, latview.build_scene(_p(LType=5, Ordis=7)), colour="w")) > 3


def test_marks_do_not_depend_on_radial_width():
    """CrystalScatter form: the radial width smears a reflection along its own direction only; whether it reaches
    the detector is set by the azimuthal width (EditAzi), and it appears at |q| = g along the projection of G."""
    base = dict(LType=17, uca=30, ucb=30, ucc=42, ucalpha=94, ucbeta=94, ComboBoxPeak=0, HKLmax=6, EditQmax=2,
                EditRadius=1, EditDebyeWaller=1)
    sharp = latview.build_scene(_p(**base, RadButDebyeScherrer=True, EditDomainSize=140))
    broad = latview.build_scene(_p(**base, RadButDebyeScherrer=False))                  # fixed width 1 1/nm
    assert np.array_equal(sharp.on_detector, broad.on_detector)
    r = np.hypot(*sharp.det[:, sharp.on_detector])
    assert np.allclose(r, np.linalg.norm(sharp.G[:, sharp.on_detector], axis=0), rtol=1e-9)
    wide = latview.build_scene(_p(**base, EditDomainSize=140, EditAzi=10))              # arc width 0.4 1/nm
    assert wide.on_detector.sum() > sharp.on_detector.sum()
    assert latview.build_scene(_p(**base, EditAzi=0)).rings
