"""Scroll bars instead of a minimum window size (pysasfit.core.scrolling)."""
import os

import pytest

pytest.importorskip("PySide6")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6 import QtWidgets  # noqa: E402

from pysasfit.core.scrolling import MIN_PAGE, scroll_inside, scrolled  # noqa: E402


def _wide_page():
    p = QtWidgets.QWidget()
    lay = QtWidgets.QHBoxLayout(p)
    buttons = [QtWidgets.QPushButton("x" * 30) for _ in range(6)]
    for b in buttons:
        lay.addWidget(b)
    return p, buttons


def test_scroll_inside_keeps_the_page_and_removes_its_minimum_size():
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    p, buttons = _wide_page()
    assert p.minimumSizeHint().width() > 800
    sc = scroll_inside(p)
    assert p.minimumSizeHint() == MIN_PAGE
    assert scroll_inside(p) is sc                       # once only
    p.resize(250, 120)
    p.show()
    app.processEvents()
    assert sc.horizontalScrollBar().isVisible() and buttons[0].isVisible()
    assert buttons[0].window() is p                     # the controls are still in the page
    p.close()


def test_scrolled_widget_has_a_small_minimum():
    QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    p, _b = _wide_page()
    sc = scrolled(p)
    assert sc.widget() is p and sc.widgetResizable() and sc.minimumSizeHint() == MIN_PAGE


def test_dock_panels_scroll():
    """A wide panel of a tool (DockMixin) does not fix the smallest size of its window."""
    from pysasfit.core.docking import DockMixin
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    class Win(DockMixin, QtWidgets.QMainWindow):
        def __init__(self):
            super().__init__()
            self.menuBar().addMenu("&View")
            self._add_dock("wide", "Wide", _wide_page()[0])
            self._add_dock("text", "Text", QtWidgets.QPlainTextEdit())
            self._finish_docks()

    w = Win()
    w.show()
    app.processEvents()
    assert w.minimumSizeHint().width() < 500
    w.close()
