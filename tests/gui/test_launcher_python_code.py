"""Launcher: every tool has "Copy Python code" (Ctrl+Shift+C), listed in the launcher's File menu."""
import os

import pytest

pytest.importorskip("PySide6")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6 import QtWidgets  # noqa: E402

from pysasfit.gui.launcher import Launcher  # noqa: E402


def test_every_tool_copies_python_code():
    """Every tool has "Copy Python code" (Ctrl+Shift+C), listed in the launcher's File menu."""
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    w = Launcher()
    app.processEvents()
    for title, win in w.windows.items():
        act = getattr(win, "copy_python_action", None)
        assert act is not None, title
        assert act.shortcut().toString() == "Ctrl+Shift+C"
        assert callable(getattr(win, "python_code", None)), title
        w.tabs.show_page(win)
        fmenu = w.menuBar().actions()[0].menu()
        fmenu.aboutToShow.emit()
        assert act in fmenu.actions(), title
    w.close()
