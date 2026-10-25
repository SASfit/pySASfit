"""The SANS-reduction tool (pysasfit-sasred and the launcher tab)."""
import os

import numpy as np
import pytest

pytest.importorskip("PySide6")
pytest.importorskip("h5py")
pytest.importorskip("yaml")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6 import QtWidgets  # noqa: E402

from pysasfit.sasred.gui import MainWindow  # noqa: E402

RAW = os.environ.get("PYSASRED_TEST_RAW", "/mnt/user-data/uploads/Sept2026/raw")
need = pytest.mark.skipif(not os.path.isdir(RAW), reason="Vifor raw data not available")


def _app():
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


def test_window_without_data():
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
    _app()
    w = MainWindow()
    assert set(w.docks) == {"runs", "reduction", "detector", "iq", "budget", "log"}
    canvases = w.findChildren(FigureCanvasQTAgg)
    bars = {id(tb.canvas) for tb in w.findChildren(NavigationToolbar2QT)}
    assert len(canvases) == 3 and all(id(c) in bars for c in canvases)
    w.clear_results()      # nothing to clear, must not fail
    w.close()


def test_preferences_registered():
    from pysasfit.core import settings as prefs
    assert ("sasred", "SANS reduction") in prefs.tools()
    assert prefs.get("sasred.qbins.n") == 60


@need
def test_folder_beam_centre_reduce_and_save(tmp_path):
    from pysasfit.sasred.io.sinq import read_sans_i
    from pysasfit.sasred.reduce import reduce_run_1d
    from pysasfit.sasred.core.averaging import q_edges
    import h5py

    app = _app()
    w = MainWindow()
    w.set_folder(RAW)
    assert len(w.entries) > 100
    w.select_runs([82173])                           # attenuated direct beam, 8 m
    w.on_beam_centre()
    assert abs(w.xc.value() + 1 - 62.73) < 0.1 and abs(w.yc.value() - 63.94) < 0.1
    w.select_runs([82130, 82131])
    w.set_units("A")                                 # q limits below in 1/Å
    w.qmin.setValue(0.002)
    w.qmax.setValue(0.08)
    w.nbins.setValue(40)
    w.on_reduce()
    app.processEvents()
    assert set(w.results) == {82130, 82131}
    # the GUI result is the library result with the same parameters
    raw = read_sans_i(os.path.join(RAW, "sans2026n082130.hdf"))
    ref = reduce_run_1d(raw, (w.xc.value(), w.yc.value()), q_edges(0.002, 0.08, 40), mask=w._mask(raw.counts.shape))
    np.testing.assert_allclose(w.results[82130].I, ref.I, rtol=1e-14)
    np.testing.assert_allclose(w.results[82130].idev(), ref.idev(), rtol=1e-14)
    paths = w.save_all(str(tmp_path))
    assert len(paths) == 4 and all(os.path.isfile(p) for p in paths)
    with h5py.File([p for p in paths if p.endswith(".h5")][0]) as f:
        assert "I_budget" in f["sasentry01/sasdata01"]
    d = np.loadtxt([p for p in paths if p.endswith(".dat")][0])
    assert d.shape[1] == 5
    w.close()


def test_launcher_has_sans_reduction_tab():
    from pysasfit.gui.launcher import Launcher
    _app()
    w = Launcher(tools=[("pySASred", "pysasfit.sasred.gui", "MainWindow")])
    assert w.tabs.tabText(0) == "pySASred"
    assert isinstance(w.window_for("pySASred"), MainWindow)
    w.close()


def test_colour_scales():
    from pysasfit.core.colorscale import SCALES, make_norm
    img = np.array([[0.0, 1.0], [10.0, 1000.0]])
    for key in SCALES:
        n = make_norm(key, img)
        v = np.asarray(n(img))
        assert v[0, 0] == pytest.approx(0) and v[1, 1] == pytest.approx(1) and np.all(np.diff(v.ravel()) > 0), key
    np.testing.assert_allclose(make_norm("log1p", img)(img), np.log1p(img) / np.log1p(1000.0))
    with pytest.raises(ValueError):
        make_norm("nonsense", img)


@need
def test_detector_colour_scale_selector():
    app = _app()
    w = MainWindow()
    w.set_folder(RAW)
    w.select_runs([82130])
    for key in ("linear", "log1p", "symlog", "asinh"):
        w.det_scale.setCurrentIndex(w.det_scale.findData(key))
        app.processEvents()
        im = w.det_canvas.figure.axes[0].images[0]
        assert type(im.norm).__name__ == {"linear": "Normalize", "log1p": "FuncNorm", "symlog": "SymLogNorm",
                                          "asinh": "AsinhNorm"}[key]
    w.close()



@need
def test_units_switch_and_live_preview(tmp_path):
    import h5py
    app = _app()
    w = MainWindow()
    assert w.unit == "nm" and w.qmin.suffix() == " 1/nm"     # default: nm and 1/nm
    w.set_folder(RAW)
    assert w.table.horizontalHeaderItem(2).text() == "λ [nm]"
    w.xc.setValue(61.73)
    w.yc.setValue(63.93)
    w.select_runs([82130])
    w.update_preview()                                     # (normally after a short delay)
    p = w.preview
    assert p is not None and p.run == 82130 and 82130 not in w.results   # shown, not kept
    line = w.iq_canvas.figure.axes[0].get_lines()
    xs = [l.get_xdata() for l in line if l.get_label() == "" or True]
    assert w.iq_canvas.figure.axes[0].get_xlabel() == "q [1/nm]"
    assert any(np.allclose(x, 10 * p.q) for x in xs if len(x) == len(p.q))   # plotted in 1/nm
    w.qmin.setValue(0.1)                                   # 0.1 1/nm = 0.01 1/Å
    w.qmax.setValue(0.8)
    w.update_preview()
    assert w.preview.q.min() >= 0.01 - 1e-12 and w.preview.q.max() <= 0.08 + 1e-12
    w.set_units("A")                                       # the limits are converted
    assert w.qmin.value() == pytest.approx(0.01) and w.qmax.value() == pytest.approx(0.08)
    assert w.iq_canvas.figure.axes[0].get_xlabel() == "q [1/Å]"
    assert w.table.horizontalHeaderItem(2).text() == "λ [Å]"
    w.set_units("nm")
    w.on_reduce()
    paths = w.save_all(str(tmp_path))
    with h5py.File([x for x in paths if x.endswith(".h5")][0]) as f:
        q = f["sasentry01/sasdata01/Q"]
        assert q.attrs["units"] == b"1/nm"
        np.testing.assert_allclose(q[()], 10 * w.results[82130].q)
    txt = open([x for x in paths if x.endswith(".dat")][0]).read()
    assert "q [1/nm]" in txt and "wavelength = 0.600" in txt and " nm\n" in txt
    w.live.setChecked(False)
    w.update_preview()
    assert w.preview is None
    w.close()


@need
def test_detector_polar_view():
    app = _app()
    w = MainWindow()
    w.set_folder(RAW)
    w.select_runs([82130])
    w.sector.setChecked(True)
    w.det_view.setCurrentIndex(w.det_view.findData("polar"))
    app.processEvents()
    ax = w.det_canvas.figure.axes[0]
    assert ax.get_xlim() == (0, 360) and ax.get_ylabel() == "q [1/nm]"
    assert len(ax.collections) >= 1                 # the (phi, q) mesh (+ mask colours)
    w.set_units("A")
    assert w.det_canvas.figure.axes[0].get_ylabel() == "q [1/Å]"
    w.det_view.setCurrentIndex(w.det_view.findData("cartesian"))
    assert w.det_canvas.figure.axes[0].images
    w.close()
