import os
import time

import pytest

pytest.importorskip("PySide6")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6 import QtWidgets  # noqa: E402

from pysasfit.sldprofile.gui import MainWindow  # noqa: E402

DATA = os.path.join(os.path.dirname(__file__), "..", "..", "examples", "sldprofile", "data", "inhomogeneous_sphere_r5nm.dat")


def test_gui_load_and_fit():
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    w = MainWindow()
    w.load_file(DATA)
    assert w.data is not None and len(w.data[0]) == 50
    w.sp_n.setValue(30)
    w.ck_fix.setChecked(True)
    w.ck_auto.setChecked(False)
    w.sp_rmax.setValue(5.0)
    w.on_run()
    t = time.time()
    while (w.worker is not None or w.result is None) and time.time() - t < 120:
        app.processEvents()
        time.sleep(0.02)
    assert w.result is not None
    assert 0.5 < w.result.chi2_red < 1.5
    w.close()


def test_gui_two_step_method(tmp_path):
    """Glatter's two-step route from the GUI: IFT with D_max by evidence, deconvolution, save."""
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    w = MainWindow()
    w.load_file(DATA)
    w.sp_n.setValue(30)
    w.cb_method.setCurrentIndex(1)
    assert w.cb_perr.isEnabled() and w.sp_nr.isEnabled()
    w.ck_auto.setChecked(False)
    w.sp_rmax.setValue(5.0)
    w.sp_nc.setValue(7)
    w.on_run()
    t = time.time()
    while (w.worker is not None or w.result is None) and time.time() - t < 120:
        app.processEvents()
        time.sleep(0.02)
    res = w.result
    assert res is not None and hasattr(res, "twostep")
    assert 0.5 < res.chi2_red < 1.5                       # IFT fit of I(q)
    assert abs(res.r_max[0] - 5.0) < 0.5                  # D_max / 2 from the evidence
    assert "Two-step method" in w.summary.toPlainText()
    w.cb_over.setCurrentIndex(1)                          # p(r) overlay: IFT p(r) + deconvolution fit
    w.redraw()
    labels = [ln.get_label() for ax in w.fig.axes for ln in ax.get_lines()]
    assert "from deconvolved ρ(r)" in labels and "p(r) of deconvolved ρ" in labels
    prefix = str(tmp_path / "ts")
    w.result.twostep.save(prefix)
    assert (tmp_path / "ts_ift_pr.dat").exists() and (tmp_path / "ts_I_from_rho.dat").exists()
    w.close()
