"""One window for all pysasfit tools.

    pysasfit-gui [datafile | project.pysasfit]
    python -m pysasfit.gui.launcher [datafile | project.pysasfit]

Every tool keeps its own main window; the launcher hosts them as tabs. A tool
whose import fails (e.g. a missing optional dependency) gets a tab with the
error message instead of breaking the whole launcher. A data file given on the
command line is loaded into the SLD-profile tab, a project file into its tool.
The other tools reopen their last session (see :mod:`pysasfit.core.project`).

To add a tool, append ``(title, "module.path", "ClassName")`` to ``TOOLS``.
"""

from __future__ import annotations

import importlib
import os
import sys
import traceback

from PySide6 import QtCore, QtGui, QtWidgets

TOOLS = [
    ("SLD profile", "pysasfit.sldprofile.gui", "MainWindow"),
    ("Size distribution / p(r)", "pysasfit.inversion.gui.app", "MainWindow"),
    ("Ornstein-Zernike", "pysasfit.oz.gui.main_window", "OZWidget"),
]

# tab title -> preferences page
_TOOL_KEYS = {"SLD profile": "sldprofile", "Size distribution / p(r)": "inversion", "Ornstein-Zernike": "oz"}


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
                # one Settings menu for the whole launcher (below), not one per tab
                if isinstance(win, QtWidgets.QMainWindow) and win.menuBar() is not None:
                    win.menuBar().setVisible(False)
                self.windows[title] = win
                self.tabs.addTab(win, title)
            except Exception:  # noqa: BLE001 -- show, don't crash
                self.tabs.addTab(_error_tab(title, traceback.format_exc()), title + " (error)")
        self.resize(1500, 950)
        from ..core.settings_dialog import open_preferences
        # projects: act on the tool in front
        fmenu = self.menuBar().addMenu("&File")
        for text, meth, key in (("&Open project…", "on_open_project", QtGui.QKeySequence.StandardKey.Open),
                                ("&Save project…", "on_save_project", QtGui.QKeySequence.StandardKey.Save)):
            a = QtGui.QAction(text, self)
            a.setShortcut(key)
            a.triggered.connect(lambda _=False, m=meth: self._call_current(m))
            fmenu.addAction(a)
        # panels of the tool in front (input, plots): show / hide, reset layout
        self.view_menu = self.menuBar().addMenu("&View")
        self.view_menu.aboutToShow.connect(self._fill_view_menu)
        menu = self.menuBar().addMenu("&Settings")
        act = QtGui.QAction("&Preferences…", self)
        act.setShortcut(QtGui.QKeySequence.StandardKey.Preferences)
        # opens on the page of the tool that is in front
        act.triggered.connect(lambda: open_preferences(
            self, _TOOL_KEYS.get(self.tabs.tabText(self.tabs.currentIndex()))))
        menu.addAction(act)
        self.preferences_action = act
        forget = QtGui.QAction("Forget last session (all tools)", self)
        forget.triggered.connect(self.forget_sessions)
        menu.addAction(forget)

    def _fill_view_menu(self):
        m = self.view_menu
        m.clear()
        w = self.tabs.currentWidget()
        docks = getattr(w, "docks", None)
        if not docks:
            a = m.addAction("(no detachable panels in this tool)")
            a.setEnabled(False)
            return
        for a in w.view_actions.values():
            m.addAction(a)
        m.addSeparator()
        if getattr(w, "border_actions", None):
            sub = m.addMenu("Panel border")
            for a in w.border_actions.values():
                sub.addAction(a)
        m.addAction("Dock all panels", w.dock_all)
        m.addAction("Reset layout", w.reset_layout)

    def _call_current(self, method):
        w = self.tabs.currentWidget()
        f = getattr(w, method, None)
        if callable(f):
            f()
        else:
            QtWidgets.QMessageBox.information(self, "Projects", "This tool has no project files.")

    def forget_sessions(self):
        for w in self.windows.values():
            f = getattr(w, "forget_session", None)
            if callable(f):
                f()
        self.statusBar().showMessage("last sessions deleted", 8000)

    def restore_sessions(self, skip=()):
        for title, w in self.windows.items():
            f = getattr(w, "restore_session", None)
            if callable(f) and title not in skip:
                f()

    def open_project(self, path):
        """Open a project file in the tool it belongs to."""
        import json
        import zipfile
        try:
            with zipfile.ZipFile(path) as z:
                tool = json.loads(z.read("project.json").decode("utf-8")).get("tool")
        except Exception as e:  # noqa: BLE001
            QtWidgets.QMessageBox.critical(self, "Open project", f"{path}: not a pySASfit project ({e})")
            return None
        for title, key in _TOOL_KEYS.items():
            w = self.windows.get(title)
            if key == tool and hasattr(w, "open_project"):
                self.tabs.setCurrentWidget(w)
                w.open_project(path)
                return title
        QtWidgets.QMessageBox.critical(self, "Open project", f"{path}: no tool '{tool}' in this window")
        return None

    def window_for(self, title):
        return self.windows.get(title)

    def closeEvent(self, event):
        for w in self.windows.values():
            shutdown = getattr(w, "shutdown", None)  # e.g. the OZ tool restores stdout/stderr
            if callable(shutdown):
                shutdown()
            close_detached = getattr(w, "close_detached", None)   # independent panel windows
            if callable(close_detached):
                close_detached()
        super().closeEvent(event)


def main(argv=None):
    argv = sys.argv if argv is None else argv
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(argv)
    w = Launcher()
    w.show()
    skip = set()
    if len(argv) > 1 and os.path.isfile(argv[1]):
        if argv[1].endswith(".pysasfit"):
            t = w.open_project(argv[1])
            if t:
                skip.add(t)
        else:
            sld = w.window_for("SLD profile")
            if sld is not None:
                sld.load_file(argv[1])
                skip.add("SLD profile")
    w.restore_sessions(skip)
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
