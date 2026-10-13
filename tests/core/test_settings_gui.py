import json
import os

import pytest

pytest.importorskip("PySide6")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6 import QtWidgets  # noqa: E402

from pysasfit.core import settings as prefs  # noqa: E402

DATA = os.path.join(os.path.dirname(__file__), "..", "..", "examples", "sldprofile", "data",
                    "inhomogeneous_sphere_r5nm.dat")


@pytest.fixture
def app(tmp_path, monkeypatch):
    monkeypatch.setenv("PYSASFIT_SETTINGS", str(tmp_path / "settings.json"))
    prefs.load()
    a = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    yield a
    prefs.reset()
    prefs.load(str(tmp_path / "none.json"))


def test_dialog_edit_save_and_reset(app, tmp_path, monkeypatch):
    from pysasfit.core.settings_dialog import PreferencesDialog

    warned = []
    monkeypatch.setattr(QtWidgets.QMessageBox, "warning", staticmethod(lambda *a, **k: warned.append(a)))
    d = PreferencesDialog(tool="inversion")
    assert d.tabs.tabText(d.tabs.currentIndex()).startswith("Size distribution")
    rows = {r.s.key: r for r in d.rows}
    rows["sldprofile.opt.alpha_step"].w.setText("0.25")
    rows["inversion.opt.evidence_lambda_scale"].w.setCurrentIndex(1)
    rows["oz.generic.grid_exponent"].w.setValue(13)
    assert rows["sldprofile.opt.alpha_step"].label.font().bold()
    assert d.save(), warned
    data = json.loads((tmp_path / "settings.json").read_text())["values"]
    assert data == {"sldprofile.opt.alpha_step": 0.25, "inversion.opt.evidence_lambda_scale": "median",
                    "oz.generic.grid_exponent": 13}
    # invalid input is refused and nothing is written
    d2 = PreferencesDialog()
    {r.s.key: r for r in d2.rows}["sldprofile.opt.alpha_step"].w.setText("abc")
    assert not d2.save() and warned
    d3 = PreferencesDialog()
    d3.btn_reset_all.click()
    # every factory value survives the editor round trip unchanged
    assert all(r.value() == r.s.default for r in d3.rows)
    assert d3.save() and json.loads((tmp_path / "settings.json").read_text())["values"] == {}


def test_tools_read_preferences(app):
    from pysasfit.sldprofile.gui import MainWindow
    from pysasfit.oz.gui.generic_polydisperse_tab import GenericPolydisperseTab

    prefs.set_value("sldprofile.opt.alpha_step", 0.25)
    prefs.set_value("oz.generic.grid_exponent", 13)
    prefs.set_value("oz.generic.transform_type", 4)
    prefs.set_value("oz.generic.fit_max_nfev", 77)
    w = MainWindow()
    w.load_file(DATA)
    assert w._fitter().opt["alpha_step"] == 0.25
    assert any(a.text() == "&Settings" for a in w.menuBar().actions())
    w.close()
    t = GenericPolydisperseTab()
    assert t.SETTINGS_DEFAULTS["gridN"] == 2**13 and t.SETTINGS_DEFAULTS["transformType"] == 4
    assert t.FIT_MAX_NFEV == 77
    t.FIT_MAX_NFEV = 5                      # a per-tab override still wins
    assert t.FIT_MAX_NFEV == 5
    t.deleteLater()


def test_launcher_has_one_settings_menu(app):
    from pysasfit.gui.launcher import Launcher

    w = Launcher()
    assert [a.text() for a in w.menuBar().actions()] == ["&Settings"]
    for win in w.windows.values():
        if isinstance(win, QtWidgets.QMainWindow):
            assert not win.menuBar().isVisible()
    w.close()


def test_explanation_panel(qtbot=None):
    from PySide6 import QtWidgets
    from pysasfit.core.settings_dialog import PreferencesDialog
    QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    dlg = PreferencesDialog()
    r = dlg.rows[0]
    dlg.explain(r)
    assert r.s.key in dlg.explain_box.text()
    assert r.s.help[:30].replace("&", "&amp;").replace("<", "&lt;") in dlg.explain_box.text()
    dlg.close()
