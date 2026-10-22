import os
import time

import pytest

pytest.importorskip("PySide6")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6 import QtCore, QtWidgets  # noqa: E402

from pysasfit.sldprofile.gui import MainWindow  # noqa: E402

DATA = os.path.join(os.path.dirname(__file__), "..", "..", "examples", "sldprofile", "data", "inhomogeneous_sphere_r5nm.dat")


def test_gui_load_and_fit():
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    w = MainWindow()
    w.load_file(DATA)
    assert w.data is not None and len(w.data[0]) == 50
    # the folder of the file is shown below its name
    assert w.lbl_path.isVisibleTo(w)
    assert w.lbl_path.text().replace("\u200b", "") == os.path.dirname(os.path.abspath(DATA))
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
    labels = [ln.get_label() for f in w.figs.values() for ax in f.axes for ln in ax.get_lines()]
    assert "from deconvolved ρ(r)" in labels and "p(r) of deconvolved ρ" in labels
    prefix = str(tmp_path / "ts")
    w.result.twostep.save(prefix)
    assert (tmp_path / "ts_ift_pr.dat").exists() and (tmp_path / "ts_I_from_rho.dat").exists()
    w.close()


def test_gui_bspline_representation_adaptive():
    """Cubic B-spline representation with adaptive knots from the GUI."""
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    w = MainWindow()
    w.load_file(DATA)
    w.cb_rep.setCurrentIndex(1)            # cubic B-spline
    w.sp_n.setValue(15)
    w.cb_grid.setCurrentIndex(2)           # adaptive knots
    w.ck_fix.setChecked(True)
    w.ck_auto.setChecked(False)
    w.sp_rmax.setValue(5.0)
    w.ck_ticks.setChecked(True)
    w.on_run()
    t = time.time()
    while (w.worker is not None or w.result is None) and time.time() - t < 120:
        app.processEvents()
        time.sleep(0.02)
    assert w.result is not None
    assert getattr(w.result.best.grid, "is_bspline", False) and w.result.best.grid.degree == 3
    assert 0.5 < w.result.chi2_red < 1.5
    assert "cubic B-spline" in w.log.toPlainText() and "adaptive knots" in w.log.toPlainText()
    w.redraw()
    w.close()


def test_gui_detachable_panels_and_layout_in_session():
    """Input and every plot are dock widgets; a detached panel stays detached in the
    last session; View -> Reset layout docks everything again."""
    from pysasfit.core import project as prj
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    w = MainWindow()
    assert set(w.docks) == {"input", "iq", "profile", "rmax", "evidence", "text", "alpha2"}
    assert [a.text() for a in w.view_menu.actions()][:7] == [d.windowTitle() for d in w.docks.values()]
    w.load_file(DATA)
    w.show()
    app.processEvents()
    # the detach button of the title bar (clicked(bool) must not be taken as the panel name)
    w.docks["rmax"].titleBarWidget().detach_button.click()
    app.processEvents()
    assert "rmax" in w.detached
    w.dock_panel("rmax")
    pw = w.detach("profile")
    # an independent window: no parent (not kept above the main window), not a tool window
    assert pw.parent() is None and pw.isWindow()
    assert (pw.windowFlags() & QtCore.Qt.WindowType.Tool) != QtCore.Qt.WindowType.Tool
    assert not (pw.windowFlags() & QtCore.Qt.WindowType.WindowStaysOnTopHint)
    assert w.canvases["profile"].window() is pw
    w.set_panel_visible("evidence", False)
    app.processEvents()
    assert w.view_actions["profile"].isChecked() and not w.view_actions["evidence"].isChecked()
    w.close()                                           # saves the session incl. the layout
    assert not pw.isVisible()                           # detached windows close with the main window
    w2 = MainWindow()
    w2.show()
    assert w2.restore_session()
    app.processEvents()
    assert "profile" in w2.detached and w2.detached["profile"].isVisible()
    assert w2.docks["evidence"].isHidden()
    w2.detached["profile"].close()                      # closing the window docks the panel again
    app.processEvents()
    assert "profile" not in w2.detached and w2.canvases["profile"].window() is w2
    w2.detach("iq")
    w2.reset_layout()
    app.processEvents()
    assert not w2.detached and not w2.docks["evidence"].isHidden()
    prj.forget_session("sldprofile")
    w2.close()


def test_gui_ift_points_follow_shannon_number():
    """IFT points default to N_s = q_max D_max / pi; > 25 % away -> bold + comment."""
    import numpy as np
    from pysasfit.sldprofile.twostep import shannon_points
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    w = MainWindow()
    w.load_file(DATA)
    w.cb_method.setCurrentIndex(1)
    w.ck_auto.setChecked(False)
    w.ck_fix.setChecked(False)
    w.sp_rmax.setValue(20.0)
    w.sp_hi.setValue(1.5)
    ns = shannon_points(np.max(w.data[0]), 2 * 20.0 * 1.5)
    assert w.sp_nr.value() == ns and not w.sp_nr.font().bold()
    w.sp_rmax.setValue(30.0)                             # follows while not changed by hand
    ns2 = shannon_points(np.max(w.data[0]), 2 * 30.0 * 1.5)
    assert w.sp_nr.value() == ns2
    w.sp_nr.setValue(round(0.8 * ns2))                   # chosen by hand, 20 % below: tolerated
    assert not w.sp_nr.font().bold() and "under-sampled" not in w.lbl_nr.text()
    w.sp_nr.setValue(int(0.7 * ns2))                     # 30 % below
    assert w.sp_nr.font().bold() and "under-sampled" in w.lbl_nr.text()
    w.sp_rmax.setValue(35.0)                             # no longer overwritten
    assert w.sp_nr.value() == int(0.7 * ns2)
    w.sp_nr.setValue(int(1.5 * w._nr_suggested))
    assert w.sp_nr.font().bold() and "above" in w.lbl_nr.text()
    w.sp_nr.setValue(w._nr_suggested)
    assert not w.sp_nr.font().bold()
    w.close()


def test_gui_alpha2_panel_hidden_by_default():
    """Two-step route: the chi2(alpha) panel is optional (View menu), marks the alphas of
    both criteria and is hidden again by View -> Reset layout."""
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    w = MainWindow()
    w.show()
    app.processEvents()
    assert w.docks["alpha2"].isHidden() and not w.view_actions["alpha2"].isChecked()
    w.load_file(DATA)
    w.sp_n.setValue(20)
    w.cb_method.setCurrentIndex(1)
    w.ck_auto.setChecked(False)
    w.sp_rmax.setValue(5.0)
    w.ck_fix.setChecked(True)
    w.on_run()
    t = time.time()
    while (w.worker is not None or w.result is None) and time.time() - t < 120:
        app.processEvents()
        time.sleep(0.02)
    sc = w.result.twostep.alpha_scan
    assert sc["alpha_from"] == "intensity"
    w.view_actions["alpha2"].trigger()                   # View -> show the panel
    app.processEvents()
    assert not w.docks["alpha2"].isHidden() and w.view_actions["alpha2"].isChecked()
    labels = [ln.get_label() for ax in w.figs["alpha2"].axes for ln in ax.get_lines()]
    assert "χ²/M of I(q) from ρ(r)" in labels
    assert "α from the fit of I(q) (used)" in labels and "α of max. evidence of p(r)" in labels
    w.reset_layout()
    assert w.docks["alpha2"].isHidden() and not w.view_actions["alpha2"].isChecked()
    assert not w.docks["evidence"].isHidden()
    w.close()


def test_gui_panel_borders_and_wrapped_title():
    """Every detachable panel has a border (View -> Panel border), kept with the
    layout; the I(q) title is broken into lines that fit the panel width."""
    from PySide6 import QtWidgets as W
    from matplotlib.figure import Figure
    from pysasfit.core.docking import DEFAULT_PANEL_BORDER
    from pysasfit.sldprofile.gui import _wrap_to_width
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    w = MainWindow()
    assert DEFAULT_PANEL_BORDER == "sunken" and w.border_actions["sunken"].isChecked()
    assert set(w.panel_frames) == set(w.docks)
    assert all(f.frameShape() == W.QFrame.Shape.Panel and f.frameShadow() == W.QFrame.Shadow.Sunken
               for f in w.panel_frames.values())
    w.border_actions["line"].trigger()
    assert w.panel_border == "line"
    assert all(f.frameShape() == W.QFrame.Shape.Box and f.frameShadow() == W.QFrame.Shadow.Plain
               for f in w.panel_frames.values())
    st = w.dock_state()
    w.set_panel_border("groove")
    w.restore_dock_state(st)
    assert w.panel_border == "line" and w.border_actions["line"].isChecked()
    pw = w.detach("profile")                              # the border goes with a detached panel
    assert w.panel_frames["profile"].window() is pw
    w.dock_panel("profile")
    w.close()
    fig = Figure(figsize=(3, 2), dpi=100)
    txt = _wrap_to_width(["two-step (Glatter), sphere, smooth:", "r_max = 5.175 ± 0.15,", "Rg = 4.01 ± 0.014,",
                          "χ²/M = 0.951"], fig, 9)
    assert txt.count("\n") >= 2 and all(len(t) <= 300 / (0.6 * 9 * 100 / 72) + 1 or "  " not in t
                                        for t in txt.split("\n"))


def _run(w, app):
    w.on_run()
    t = time.time()
    while (w.worker is not None or w.result is None) and time.time() - t < 120:
        app.processEvents()
        time.sleep(0.02)


def test_gui_manual_alpha_is_an_add_on():
    """Manual ln alpha (click or spin box) shows a second solution; the automatic result
    stays; 'automatic' removes it; a new fit resets it."""
    from types import SimpleNamespace
    import numpy as np
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    w = MainWindow()
    w.load_file(DATA)
    w.sp_n.setValue(20)
    w.ck_fix.setChecked(True)
    w.ck_auto.setChecked(False)
    w.sp_rmax.setValue(5.0)
    assert not w.ck_man.isEnabled()
    _run(w, app)
    auto = w.result
    la0 = float(np.log(auto.best.alpha))
    assert w.ck_man.isEnabled() and abs(w.sp_man.value() - la0) < 0.01
    # a click on the evidence plot at ln alpha = la0 - 4
    ax = w.figs["evidence"].axes[0]
    w._on_alpha_click(SimpleNamespace(button=1, inaxes=ax, xdata=la0 - 4, canvas=w.canvases["evidence"]))
    m = w._manual
    assert m is not None and abs(m["la"] - (la0 - 4)) < 0.006 and w.result is auto
    assert m["sol"].alpha < auto.best.alpha and "Δln Ev" in w.lbl_man.text()
    labels = [ln.get_label() for f in w.figs.values() for a in f.axes for ln in a.get_lines()]
    assert any("(manual)" in lab for lab in labels)
    w.sp_man.setValue(la0 + 2)                            # typed value
    assert abs(w._manual["la"] - (la0 + 2)) < 0.006
    w.btn_auto.click()
    assert w._manual is None and not w.ck_man.isChecked() and abs(w.sp_man.value() - la0) < 0.01
    # two-step: step 2 at another alpha; a new fit switches the manual solution off
    w.set_manual_alpha(la0)
    w.cb_method.setCurrentIndex(1)
    w.result = None
    _run(w, app)
    assert w._manual is None and not w.ck_man.isChecked()
    ts = w.result.twostep
    w.set_manual_alpha(float(np.log(ts.alpha_scan["evidence_alpha"])))
    assert w._manual is not None and "χ²/M of p(r)" in w.lbl_man.text()
    w.close()
