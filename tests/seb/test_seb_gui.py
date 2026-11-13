"""pySEBpolymers panel: templates, build, parameters, fit, Python code, session."""
import os
import sys

import numpy as np
import pytest

pytest.importorskip("PySide6")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6 import QtWidgets  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pycode_util import run_script  # noqa: E402

from pysasfit.seb import templates as T  # noqa: E402
from pysasfit.seb.gui import MainWindow  # noqa: E402


@pytest.fixture
def win():
    QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    w = MainWindow()
    yield w
    w._timer.stop()
    w.close()


def _micelle_data(tmp_path, Rc=35.0, Rg=15.0, noise=0.02, seed=3):
    code = T.code_for("Micelle (core + grafted chains)", {"n": 12})
    s = T.build(code)
    p = {"beta_core": 4.0, "R_core": Rc, "beta_chain": 0.5, "Rg_chain": Rg}
    q = np.geomspace(0.003, 0.3, 120)
    I = 2.0 * s.form_factor(q, p) + 1e-3
    sig = noise * I
    I = I + np.random.default_rng(seed).normal(0, sig)
    path = tmp_path / "micelle.dat"
    np.savetxt(path, np.column_stack([q, I, sig]))
    return path


def test_every_template_builds_in_the_panel(win):
    for name in T.TEMPLATES:
        win.template.setCurrentText(name)
        assert win.structure is not None and win.code_built == win.editor.toPlainText(), name
        if win.curve[0] is None:              # the start template is built, not calculated (Calculate button)
            win.recalc()
        q, I = win.curve
        assert np.all(np.isfinite(I)) and I[0] > 0.9, name
        names = [win.table.item(r, 0).text() for r in range(win.table.rowCount())]
        assert names[:2] == ["scale", "background"] and set(names[2:]) == set(win.structure.parameters())


def test_options_and_edited_code(win):
    win.template.setCurrentText("Star polymer")
    win.opt_widgets["n"].setValue(9)
    assert len(win.structure.units()) == 10                      # 9 arms on a Point
    assert win.designer.root.count_units() == 10
    code = win.editor.toPlainText().replace("range(9)", "range(3)")
    assert win.build(code) and len(win.structure.units()) == 4
    assert len(win.designer.root.nodes) == 4                     # typed code is drawn, one node per sub-unit
    assert not win.build("s = 3")                                   # error is shown, structure kept
    assert len(win.structure.units()) == 4 and "Structure" in win.struct_status.text()


def test_designer_edits_rebuild_the_structure(win):
    win.template.setCurrentText("Micelle (core + grafted chains)")
    win.opt_widgets["n"].setValue(5)
    d = win.designer.root
    core = d.by_name("core")
    # a second array of rods on the same core, at the centre
    def add():
        arr = d.add_pattern("Array", "ThinRod", name="rods", tag="rod", n=3, in_ref="end1")
        d.link(core.id, "center", arr.id, "anchor")
    assert win.designer.edit(add)
    assert len(win.structure.units()) == 1 + 5 + 3
    assert {"L_rod", "beta_rod"} <= set(win.structure.parameters())
    names = [win.table.item(r, 0).text() for r in range(win.table.rowCount())]
    assert "L_rod" in names
    win.designer.undo()
    assert len(win.structure.units()) == 6
    win.designer.redo()
    assert len(win.structure.units()) == 9
    # an incomplete design does not replace the structure (undo/redo made a new Design object)
    d = win.designer.root
    assert win.designer.edit(lambda: d.add_unit("SolidSphere", "lonely"))
    assert len(win.structure.units()) == 9 and "not connected" in win.struct_status.text()


def test_sketch_and_design_files(win, tmp_path):
    win.template.setCurrentText("Bottle brush")
    win.draw_sketch(new=True)
    ax = win.sketch_canvas.figure.axes[0]
    assert ax.name == "3d" and len(ax.lines) >= 60
    path = tmp_path / "brush.sebdesign"
    win.save_design(str(path))
    win.template.setCurrentText("Star polymer")
    assert win.open_design(str(path))
    assert len(win.structure.units()) == 60 and "segment" in win.designer.root.components


def test_fit_recovers_the_micelle(win, tmp_path):
    win.template.setCurrentText("Micelle (core + grafted chains)")
    win.opt_widgets["n"].setValue(12)
    assert win.load_file(str(_micelle_data(tmp_path)))
    win.set_param("beta_core", 4.0)
    win.set_param("beta_chain", 0.5)
    win.set_param("R_core", 30.0, fit=True, lo=5, hi=100)
    win.set_param("Rg_chain", 20.0, fit=True, lo=2, hi=60)
    win.set_param("background", 0.0, fit=True)
    res = win.fit(wait=True)
    assert res is not None and res["success"]
    assert res["values"]["R_core"] == pytest.approx(35.0, rel=0.02)
    assert res["values"]["Rg_chain"] == pytest.approx(15.0, rel=0.03)
    assert res["values"]["scale"] == pytest.approx(2.0, rel=0.03)
    assert 0.5 < res["chi2"] < 2.0
    assert win.values()["R_core"] == pytest.approx(res["values"]["R_core"], rel=1e-5)   # written back
    assert win.table.item(win._row_of()["R_core"], 5).text()                            # error shown


def test_python_code_repeats_the_calculation(win, tmp_path):
    win.template.setCurrentText("Dendrimer")
    path = _micelle_data(tmp_path)
    win.load_file(str(path))
    win.set_param("Rg_g1", 7.5)
    win.recalc()
    code = win.python_code()
    ns = run_script(code, tmp_path)
    q, I = win.curve
    np.testing.assert_allclose(ns["q"], q)
    np.testing.assert_allclose(ns["I"], I, rtol=1e-12)
    assert "chi2_red" in ns


def test_session_roundtrip(win, tmp_path, monkeypatch):
    monkeypatch.setattr(MainWindow, "_session_path", staticmethod(lambda: str(tmp_path / "seb_session.json")))
    win.template.setCurrentText("Bottle brush")
    win.opt_widgets["n"].setValue(7)
    win.set_param("Rg_sc", 12.5, fit=True, lo=1, hi=99)
    win.mode.setCurrentIndex(1)
    win.load_file(str(_micelle_data(tmp_path)))
    win.save_session()
    w2 = MainWindow()
    try:
        assert w2.restore_session()
        assert w2.template.currentText() == "Bottle brush" and w2.options()["n"] == 7
        assert len(w2.structure.units()) == 14 and w2.mode.currentIndex() == 1
        r = w2.param_rows()["Rg_sc"]
        assert r == {"value": 12.5, "fit": True, "lo": 1.0, "hi": 99.0}
        assert w2.param_rows()["scale"]["hi"] == np.inf
        assert w2.data is not None and w2.data["name"] == "micelle.dat"
    finally:
        w2._timer.stop()
        w2.close()


def test_launcher_hosts_the_tab():
    from pysasfit.gui.launcher import Launcher
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    w = Launcher(tools=[("pySEBpolymers", "pysasfit.seb.gui", "MainWindow")])
    assert w.tabs.tabText(0) == "pySEBpolymers"
    win = w.window_for("pySEBpolymers")
    fmenu = w.menuBar().actions()[0].menu()
    fmenu.aboutToShow.emit()
    assert win.copy_python_action in fmenu.actions()
    app.processEvents()
    w.close()
