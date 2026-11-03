"""pySASmorph window: load, correlations, reconstruction, slices, session (offscreen)."""

import os

import numpy as np
import pytest

pytest.importorskip("PySide6")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6 import QtWidgets  # noqa: E402

from pysasfit.morph import LevelCut, levelcut_intensity  # noqa: E402


@pytest.fixture(scope="module")
def app():
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


def _write_data(path, three_columns=True):
    r = np.linspace(0, 3000, 20001)
    kk = 2 * np.pi / 150.0
    g = np.exp(-(r / 60.0) ** 2) * np.where(r > 0, np.sin(kk * r) / (kk * np.where(r > 0, r, 1)), 1.0)
    q = np.geomspace(1e-3, 0.4, 250)
    I = levelcut_intensity(q, r, LevelCut(0.3).F(g))
    cols = [q, I, 0.02 * I] if three_columns else [q, I]
    np.savetxt(path, np.column_stack(cols))
    return path


def test_window_full_run(app, tmp_path):
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT

    from pysasfit.morph.gui import MainWindow

    w = MainWindow()
    assert w.load_file(_write_data(tmp_path / "s.dat"))
    assert w.result is not None and w.result.labels is None          # correlations after loading
    w.set_phases(["solid", "pore"], [0.7, 0.3])
    w.box.setText("1200")
    w.n.setValue(32)
    w.seed.setValue(5)
    w.on_reconstruct()
    res = w.result
    assert res.labels.shape == (32, 32, 32)
    assert "percolates" in w.results_text.toPlainText()
    assert w.btn_3d.isEnabled() and w.act_export_matrix.isEnabled()
    w.slice_axis.setCurrentIndex(2)
    w.slice_pos.setValue(3)
    w.show_field.setChecked(True)
    assert w.slice_label.text().startswith("x =")
    # every plot has its toolbar
    bars = {id(t.canvas) for t in w.findChildren(NavigationToolbar2QT)}
    assert all(id(c) in bars for c in w.findChildren(FigureCanvasQTAgg))
    # session round trip
    w.save_session()
    w2 = MainWindow()
    assert w2.restore_session()
    assert w2.n.value() == 32 and w2.box.text() == "1200" and w2.data is not None
    w.close()
    w2.close()


def test_two_column_file(app, tmp_path):
    from pysasfit.morph.gui import MainWindow

    w = MainWindow()
    assert w.load_file(_write_data(tmp_path / "two.txt", three_columns=False))
    assert w.data[2] is None and w.result is not None
    w.close()


def test_3d_panel(app, tmp_path):
    pytest.importorskip("pyqtgraph")
    from pysasfit.morph.gui import MainWindow

    w = MainWindow()
    w.load_file(_write_data(tmp_path / "s.dat"))
    w.set_phases(["solid", "pore"], [0.7, 0.3])
    w.box.setText("1200")
    w.n.setValue(32)
    w.seed.setValue(1)
    w.on_reconstruct()
    v = w.view3d
    assert v.view is not None and w.result.levels is not None
    assert v.source.currentData() == "field"
    for phase in ("pore", "solid", "all phases"):
        v.phase.setCurrentText(phase)
        meshes = v.meshes()
        assert len(meshes) == (2 if phase == "all phases" else 1) and all(len(m) > 100 for m, _ in meshes)
    v.phase.setCurrentText("pore")
    v.largest.setChecked(True)
    (m, cols), = v.meshes()
    assert cols.shape == (len(m.vertices), 4)
    v.clip_axis.setCurrentIndex(3)
    v.clip_pos.setValue(10)
    assert all(mm.vertices[:, 2].max() <= 10.6 * w.result.voxel_size for mm, _ in v.meshes())
    w.show_3d_panel()
    app.processEvents()
    v.redraw()                       # draws when OpenGL is available, else only logs Qt warnings
    w.close()


def test_contrast_series_three_phases(app, tmp_path):
    """Two curves (contrasts) of an oil | film | water structure, three phases, fractions fitted."""
    from pysasfit.morph.gui import MainWindow
    from pysasfit.morph.multiphase import MultiCut, PhaseSystem

    ps = PhaseSystem(["oil", "film", "water"], [0.42, 0.16, 0.42])
    r = np.linspace(0, 3000, 20001)
    kk = 2 * np.pi / 150.0
    g = np.exp(-(r / 60.0) ** 2) * np.where(r > 0, np.sin(kk * r) / (kk * np.where(r > 0, r, 1)), 1.0)
    q = np.geomspace(1e-3, 0.6, 250)
    paths = []
    for name, sld in (("bulk", [0, 0, 6.3]), ("film", [0, 4, 0])):
        I = levelcut_intensity(q, r, MultiCut(ps).covariance(sld, g))
        p = tmp_path / f"{name}.dat"
        np.savetxt(p, np.column_stack([q, I, 0.01 * I]))
        paths.append(p)
    w = MainWindow()
    w.load_file(paths[0], correlations=False)
    w.load_file(paths[1], add=True, correlations=False)
    assert w.phase_table.columnCount() == 4
    w.set_phases(ps.names, [0.4, 0.2, 0.4], [[0, 0, 6.3], [0, 4, 0]])
    w.fit_fractions.setChecked(True)
    w.box.setText("600")
    w.n.setValue(48)
    w.seed.setValue(3)
    w.on_reconstruct()
    res = w.result
    assert len(res.curves) == 2 and res.labels.max() == 2
    assert np.allclose(w._fractions(), res.params.fractions, atol=1e-3)      # fitted fractions in the table
    assert all(c.misfit < 0.01 for c in res.curves)
    assert "S/V oil|film" in w.results_text.toPlainText()
    w.view3d.phase.setCurrentText("film")
    (m, _), = w.view3d.meshes()
    assert len(m) > 100
    w.on_move_phase(1)                                                 # nothing selected: no change
    w.phase_table.selectRow(1)
    w.on_add_phase()
    assert w.phase_table.rowCount() == 4
    w.save_session()
    w2 = MainWindow()
    assert w2.restore_session()
    assert len(w2.datasets) == 2 and w2.phase_table.rowCount() == 4
    assert w2._sld_columns()[1][:2] == [0.0, 4.0]
    w.close()
    w2.close()


def test_background_in_gui(app, tmp_path):
    from PySide6 import QtCore

    from pysasfit.morph.gui import MainWindow

    path = _write_data(tmp_path / "b.dat")
    q, I, s = np.loadtxt(path, unpack=True)
    B = 0.3 * I[-1]
    np.savetxt(path, np.column_stack([q, I + B, s]))
    w = MainWindow()
    w.load_file(path)
    w.set_phases(["solid", "pore"], [0.7, 0.3])
    w.curve_table.item(0, 3).setCheckState(QtCore.Qt.CheckState.Checked)       # fit B
    assert w.datasets[0]["fit_bg"] and w.result.curves[0].curve.B_mode == "fitted"
    fitted = w.datasets[0]["B"]
    assert fitted == pytest.approx(B, rel=0.15)
    assert float(w.curve_table.item(0, 4).text()) == pytest.approx(fitted, rel=1e-5)
    assert w.b_box.value() == pytest.approx(fitted, rel=1e-5)
    w.curve_table.item(0, 4).setText(f"{B:.8g}")                              # manual value: fixed
    assert not w.datasets[0]["fit_bg"] and w.result.curves[0].curve.B_mode == "fixed"
    assert w.result.curves[0].curve.B == pytest.approx(B, rel=1e-6)
    w.b_box.setValue(1.1 * B)                                                   # spin box: fixed, delayed run
    w._b_timer.timeout.emit()
    assert w.result.curves[0].curve.B == pytest.approx(1.1 * B, rel=1e-6)
    w.iq_mode.setCurrentIndex(1)                                                # Porod plot
    assert w.ax_iq.get_ylabel() == "I q⁴ / K"
    w.iq_mode.setCurrentIndex(0)
    assert any(t.get_text().startswith(" B = ") for t in w.ax_iq.texts)
    w._on_b_fit()
    assert w.datasets[0]["fit_bg"]
    w.save_session()
    w2 = MainWindow()
    w2.restore_session()
    assert w2.datasets[0]["fit_bg"] and w2.result.curves[0].curve.B_mode == "fitted"
    w.close()
    w2.close()
