"""I(q) plot: the axis scale is kept when other curves are shown."""
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


def test_scale_kept_on_redraw(win):
    win.iq_yscale.setCurrentIndex(win.iq_yscale.findData("linear"))
    win._draw_iq()
    assert win.iq_yscale.currentData() == "linear"
    win.iq_yscale.setCurrentIndex(win.iq_yscale.findData("log"))
    win._draw_iq()                                   # empty plot is linear on screen: must not flip the selector
    win._draw_iq()
    assert win.iq_yscale.currentData() == "log" and win.iq_xscale.currentData() == "log"


def test_toolbar_change_taken_over(win):
    win._draw_iq()
    win._iq_drawn_scales = ("log", "log")
    win._iq_ax.set_xscale("linear")                  # as the toolbar / key 'k' would do
    win._draw_iq()
    assert win.iq_xscale.currentData() == "linear"


def test_session_roundtrip(win):
    win.iq_xscale.setCurrentIndex(win.iq_xscale.findData("linear"))
    s = win._session()
    assert s["iq_xscale"] == "linear"
