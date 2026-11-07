"""Panel title bars coloured like window title bars (preferences appearance.panel_title.*)."""
import os

import pytest

pytest.importorskip("PySide6")


@pytest.fixture
def win(tmp_path, monkeypatch):
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    monkeypatch.setenv("PYSASFIT_SETTINGS", str(tmp_path / "settings.json"))
    from PySide6 import QtWidgets
    QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    from pysasfit.sasred import gui as g
    monkeypatch.setattr(g, "_session_path", lambda: str(tmp_path / "sasred_session.json"))
    w = g.MainWindow()
    yield w
    w.close()


def test_active_blue_others_pale(win):
    win._mark_active("iq")
    assert win.docks["iq"].titleBarWidget().colours == ("#0078d7", "#ffffff")
    bg, fg = win.docks["log"].titleBarWidget().colours
    assert bg != "#0078d7" and fg == "#1a1a1a"
    win._mark_active("log")
    assert win.docks["log"].titleBarWidget().colours[0] == "#0078d7"
    assert win.docks["iq"].titleBarWidget().colours[0] != "#0078d7"


def test_preferences_change(win):
    from pysasfit.core import settings as prefs
    prefs.set_value("appearance.panel_title.colour", "#2b579a")
    prefs.set_value("appearance.panel_title.mode", "all")
    win.refresh_title_bars()
    assert all(d.titleBarWidget().colours[0] == "#2b579a" for d in win.docks.values())
    prefs.set_value("appearance.panel_title.mode", "off")
    win.refresh_title_bars()
    assert all(d.titleBarWidget().colours is None for d in win.docks.values())
    prefs.set_value("appearance.panel_title.mode", "pale")
    prefs.set_value("appearance.panel_title.colour", "no colour")      # invalid: default blue
    win._mark_active("iq", force=True)
    assert win.docks["iq"].titleBarWidget().colours[0] == "#0078d7"
