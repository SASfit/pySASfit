"""pySLDcalculator window (offscreen)."""
import os

import pytest

pytest.importorskip("periodictable")
pytest.importorskip("PySide6")
pytest.importorskip("matplotlib")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6 import QtWidgets  # noqa: E402


@pytest.fixture()
def win(tmp_path, monkeypatch):
    monkeypatch.setenv("PYSASFIT_SETTINGS", str(tmp_path / "settings.json"))
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    from pysasfit.sld.gui import MainWindow
    w = MainWindow()
    yield w
    w.close()
    app.processEvents()


def test_table_follows_the_inputs(win):
    from pysasfit.sld import entries as E
    assert win.table.rowCount() > 40 and win._columns[:2] == ["silica", "polystyrene"]
    win.add_material(E.new_entry("lys", "KVFGRCELAAAMKR", "aa", "residues", 0.0))
    assert win._columns[2] == "lys" and win.matrix.count() == 5
    before = dict(((g, q), v) for g, q, u, v in win._rows)[("Neutrons", "SLD")][2]
    win.d2o.setValue(1.0)
    after = dict(((g, q), v) for g, q, u, v in win._rows)[("Neutrons", "SLD")][2]
    assert after > before + 0.5
    win.text.setPlainText("SiO2 (")
    win._editor_changed()
    assert win.error.text()
    win.text.setPlainText("AAAA")
    win._editor_changed()
    assert not win.error.text()
    for i in range(win.tabs.count()):
        win.tabs.setCurrentIndex(i)
    assert win.residues.rowCount() >= 20 and win.elements.rowCount() > 0


def test_session_and_python_code(win):
    from pysasfit.sld.gui import MainWindow
    win.matrix.setCurrentIndex(win.matrix.findData(1))
    win.x_table.setCurrentIndex(1)
    win.save_session()
    w2 = MainWindow()
    assert w2.restore_session()
    assert w2.matrix.currentData() == 1 and w2.x_table.currentData() == "henke"
    ns = {}
    exec(win.python_code(), ns)
    assert ns["columns"] == ["silica", "polystyrene"] and not ns["errors"]
    win.remove_material()
    assert len(win.entries) == 1


def test_protonated_density_checkbox(win):
    win.list.setCurrentCell(1, 0)
    win.text.setPlainText("C8D8")
    win._editor_changed()
    sld = lambda: dict(((g, q), v) for g, q, u, v in win._rows)[("Neutrons", "SLD")][1]  # noqa: E731
    low = sld()
    win.natural.setChecked(True)
    assert win.entries[1]["natural_density"] and sld() > low + 0.3
    assert sld() == pytest.approx(6.47, abs=0.02)
    win.size_mode.setCurrentIndex(win.size_mode.findData("volume"))
    assert not win.natural.isEnabled() and not win.entries[1]["natural_density"]


def test_search_dialog(win, monkeypatch):
    from pysasfit.sld import lookup as L
    monkeypatch.setattr(L, "_get_json", lambda url, timeout=0: (_ for _ in ()).throw(OSError("offline")))
    dlg = win.search_material()
    dlg.query.setText("toluene")
    for key, box in dlg.boxes.items():
        box.setChecked(key in ("builtin", "orso"))
    dlg.start()
    dlg.wait()
    assert dlg.table.rowCount() == 2 and "ORSO SLD database: no connection" in dlg.status.text()
    n = len(win.entries)
    dlg.table.selectRow(1)
    dlg.add_selected()
    assert len(win.entries) == n + 1 and win.entries[-1]["text"] == "C7D8" and win.entries[-1]["size"] == 0.943
    assert win.state()["search_sources"] == ["builtin", "orso"]
    dlg.close()
