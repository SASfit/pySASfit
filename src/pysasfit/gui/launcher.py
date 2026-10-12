"""One window for all pysasfit tools.

    pysasfit-gui [datafile]
    python -m pysasfit.gui.launcher [datafile]

Every tool keeps its own main window; the launcher hosts them as tabs. A tool
whose import fails (e.g. a missing optional dependency) gets a tab with the
error message instead of breaking the whole launcher. A data file given on the
command line is loaded into the SLD-profile tab.

To add a tool, append ``(title, "module.path", "ClassName")`` to ``TOOLS``.
"""

from __future__ import annotations

import importlib
import os
import sys
import traceback

from PySide6 import QtCore, QtWidgets

TOOLS = [
    ("SLD profile", "pysasfit.sldprofile.gui", "MainWindow"),
    ("Size distribution / p(r)", "pysasfit.inversion.gui.app", "MainWindow"),
    ("Ornstein-Zernike", "pysasfit.oz.gui.main_window", "OZWidget"),
]


def _error_tab(title, exc_text):
    w = QtWidgets.QPlainTextEdit()
    w.setReadOnly(True)
    w.setPlainText(f"The tool '{title}' could not be loaded:\n\n{exc_text}")
    return w


class Launcher(QtWidgets.QMainWindow):
    def __init__(self, tools=None):
        super().__init__()
        self.setWindowTitle("pySASfit")
        self.tabs = QtWidgets.QTabWidget()
        self.tabs.setDocumentMode(True)
        self.setCentralWidget(self.tabs)
        self.windows = {}
        for title, module, cls in tools or TOOLS:
            try:
                win = getattr(importlib.import_module(module), cls)()
                win.setWindowFlags(QtCore.Qt.WindowType.Widget)  # embed the QMainWindow
                self.windows[title] = win
                self.tabs.addTab(win, title)
            except Exception:  # noqa: BLE001 -- show, don't crash
                self.tabs.addTab(_error_tab(title, traceback.format_exc()), title + " (error)")
        self.resize(1500, 950)

    def window_for(self, title):
        return self.windows.get(title)

    def closeEvent(self, event):
        for w in self.windows.values():
            shutdown = getattr(w, "shutdown", None)  # e.g. the OZ tool restores stdout/stderr
            if callable(shutdown):
                shutdown()
        super().closeEvent(event)


def main(argv=None):
    argv = sys.argv if argv is None else argv
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(argv)
    w = Launcher()
    w.show()
    if len(argv) > 1 and os.path.isfile(argv[1]):
        sld = w.window_for("SLD profile")
        if sld is not None:
            sld.load_file(argv[1])
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
