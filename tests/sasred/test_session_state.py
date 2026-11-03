"""What pySASred keeps: centres set by hand when the folder is read again or names are corrected,
hand settings of the reduction tables when they are refilled, and the tables in the session."""
import os

import pytest

RAW = os.environ.get("PYSASRED_TEST_RAW", "/mnt/user-data/uploads/Sept2026/raw")
C8 = "L2 8.0 m, λ 6.00 Å, L1 8.0 m"
K8 = (8.0, 6.0, 8.0)

pytestmark = pytest.mark.skipif(not os.path.isdir(RAW), reason="Vifor raw data not available")


@pytest.fixture
def win(tmp_path, monkeypatch):
    pytest.importorskip("PySide6")
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    monkeypatch.setenv("PYSASFIT_SETTINGS", str(tmp_path / "settings.json"))
    from PySide6 import QtWidgets
    QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    from pysasfit.core import settings as prefs
    from pysasfit.sasred import gui as g
    monkeypatch.setattr(g, "_session_path", lambda: str(tmp_path / "sasred_session.json"))
    old = prefs.get("sasred.table.autofill")
    prefs.set_value("sasred.table.autofill", True)
    made = []

    def make():
        w = g.MainWindow()
        made.append(w)
        return w
    yield make
    for w in made:
        w.close()
    prefs.set_value("sasred.table.autofill", old)


def _hand_centre(w, x=60.25, y=61.75):
    from pysasfit.sasred.catalogue import GeometryCentre
    w.centres[K8] = GeometryCentre(K8, x, y, 0.0, 0.0, [], "by hand")


def test_hand_centre_survives_reload_and_renaming(win):
    w = win()
    w.set_folder(RAW)
    auto = w.centres[K8]
    assert auto.source == "direct beam"
    _hand_centre(w)
    w.set_folder(RAW)                                   # folder read again (new runs)
    assert (w.centres[K8].x, w.centres[K8].source) == (60.25, "by hand")
    assert "(kept: set before)" in w.log_view.toPlainText()
    w._update_centres(log=False)                        # what correcting names does (set_names)
    assert w.centres[K8].x == 60.25
    assert w.centres[(2.0, 6.0, 8.0)].source != "by hand"


def test_table_settings_survive_refill(win):
    w = win()
    w.set_folder(RAW)
    pages = {p.table.config: p for p in w.table_panel.pages()}
    t = pages[C8].table
    t.scale = "direct_beam"
    t.direct_beam_attenuator = 0.0123
    t.ignored = ["empty_cell"]
    s0 = t.samples[0]
    s0.runs[0].use = False
    s0.scaling = 0.123
    s0.transmission, s0.transmission_source = 0.77, "by hand"
    w.table_panel.on_fill()                             # refill (what reading the folder again does)
    t2 = {p.table.config: p.table for p in w.table_panel.pages()}[C8]
    assert t2 is not t
    assert (t2.scale, t2.direct_beam_attenuator, t2.ignored) == ("direct_beam", 0.0123, ["empty_cell"])
    n = next(r for r in t2.samples if r.info == s0.info)
    assert n.runs[0].use is False and n.scaling == 0.123 and (n.transmission, n.transmission_source) == (0.77, "by hand")
    assert "kept from the table before" in w.log_view.toPlainText()


def test_session_restores_tables_and_centres(win):
    w = win()
    w.set_folder(RAW)
    _hand_centre(w, 59.5, 62.5)
    t = {p.table.config: p.table for p in w.table_panel.pages()}[C8]
    t.scale = "direct_beam"
    t.samples[0].runs[0].use = False
    info = t.samples[0].info
    w.close()                                           # writes the session
    w2 = win()
    assert w2.restore_session()
    assert w2.folder == RAW and (w2.centres[K8].x, w2.centres[K8].y) == (59.5, 62.5)
    t2 = {p.table.config: p.table for p in w2.table_panel.pages()}[C8]
    assert t2.scale == "direct_beam"
    assert next(r for r in t2.samples if r.info == info).runs[0].use is False
    assert "reduction tables of the last session restored" in w2.log_view.toPlainText()
