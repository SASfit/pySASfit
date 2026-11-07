"""One window for all pysasfit tools.

    pysasfit-gui [datafile | project.pysasfit]
    python -m pysasfit.gui.launcher [datafile | project.pysasfit]

Every tool keeps its own main window; the launcher hosts them as tabs. A tool
whose import fails (e.g. a missing optional dependency) gets a tab with the
error message instead of breaking the whole launcher. A data file given on the
command line is loaded into the pySLDprofile tab, a project file into its tool.
The other tools reopen their last session (see :mod:`pysasfit.core.project`).

To add a tool, append ``(title, "module.path", "ClassName")`` to ``TOOLS``.

The tools' own menu bars are hidden; the launcher's File, View and Settings menus act on the
tool in front and also list that tool's own entries the launcher does not provide itself.

Every tool tab can be detached into its own window (drag the tab out of the tab bar,
double-click it, right-click it, or View → Detach): the tool then shows its own menu
bar again. Closing that window or its "Attach" button puts the tab back
(:mod:`pysasfit.core.tabs`). Detached tools and their window positions are part of the
last session.
"""

from __future__ import annotations

import importlib
import os
import sys
import traceback

from PySide6 import QtCore, QtGui, QtWidgets

from ..core.tabs import DetachableTabWidget
from ..resources import ALIASES, tool_icon, wordmark

TOOLS = [
    ("pySASred", "pysasfit.sasred.gui", "MainWindow"),
    ("pySASstudio", "pysasfit.studio.explorer", "ModelExplorer"),
    ("pySLDprofile", "pysasfit.sldprofile.gui", "MainWindow"),
    ("pyINVersion", "pysasfit.inversion.gui.app", "MainWindow"),
    ("pyOZsolver", "pysasfit.oz.gui.main_window", "OZWidget"),
    ("pySASmorph", "pysasfit.morph.gui", "MainWindow"),
    ("pySCAtter", "pysasfit.scatter.gui", "MainWindow"),
    ("pyTOOLbox", "pysasfit.toolbox.gui", "MainWindow"),
]

# tab title -> preferences page
_TOOL_KEYS = {"pySLDprofile": "sldprofile", "pyINVersion": "inversion", "pyOZsolver": "oz",
             "pySASred": "sasred", "pyTOOLbox": "toolbox", "pySCAtter": "scatter"}

# tab title -> logo bundle name (pysasfit.resources)
# (the tab titles are the tool names of the logo bundle; other titles may name their logo here)
_TOOL_LOGOS = {}

# tab title -> what the tool does (tab tooltip)
_TOOL_INFO = {
    "pySASred": "SANS data reduction (SINQ SANS-I) with full error propagation",
    "pySASstudio": "SASfit model explorer and fitting",
    "pySLDprofile": "radial scattering-length-density profiles",
    "pyINVersion": "size distributions and p(r) by inversion",
    "pyOZsolver": "Ornstein-Zernike structure factors",
    "pySASmorph": "representative 3D two-phase morphologies from SAS data (SAXSMorph method)",
    "pySCAtter": "scattering patterns of particles, lattices and orientation distributions (hypergeometric "
                 "algorithms after Wagener & Förster), fits to 1D and 2D data",
    "pyTOOLbox": "small helpful tools: HDF5 viewer, metadata lister, merging of reduced curves, 2D data (masks, averages, sums), 1D data arithmetic",
}

# launcher menus; a tool's own menu entries that the launcher does not provide itself are
# appended to them (e.g. "Save project as…" of pySASstudio, "Save NXcanSAS…" of pySASred)
_MENUS = ("File", "View", "Settings")
_COVERED = {"File": ("open project", "save project"), "Settings": ("preferences", "forget last session")}


def _plain(text):
    return text.replace("&", "").replace("…", "").replace("...", "").strip().lower()


def _tool_menus(win):
    """{menu name: [QAction]} of the (hidden) menu bar of an embedded tool window."""
    out = {}
    mb = win.menuBar() if isinstance(win, QtWidgets.QMainWindow) else None
    if mb is None:
        return out
    for a in mb.actions():
        if a.menu() is not None:
            out.setdefault(a.text().replace("&", ""), []).extend(a.menu().actions())
    return out


def _same_key(action, standard_key):
    return any(action.shortcut() == k for k in QtGui.QKeySequence.keyBindings(standard_key)) \
        and not action.shortcut().isEmpty()


def _error_tab(title, exc_text):
    w = QtWidgets.QPlainTextEdit()
    w.setReadOnly(True)
    w.setPlainText(f"The tool '{title}' could not be loaded:\n\n{exc_text}")
    return w


class Launcher(QtWidgets.QMainWindow):
    def __init__(self, tools=None):
        super().__init__()
        self.setWindowTitle("pySASfit")
        self.setWindowIcon(tool_icon("pySASfit"))
        self.tabs = DetachableTabWidget()
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
                self._register_tool_shortcuts(win)
                self.windows[title] = win
                logo = _TOOL_LOGOS.get(title, title)
                i = self.tabs.addTab(win, tool_icon(logo), title)
                self.tabs.setTabToolTip(i, _TOOL_INFO.get(title, title))
                self.tabs.set_wordmark(i, wordmark(logo))     # coloured label as in the logo
            except Exception:  # noqa: BLE001 -- show, don't crash
                self.tabs.addTab(_error_tab(title, traceback.format_exc()), title + " (error)")
        self.resize(1500, 950)
        # one entry per tool (View menu, Alt+1 ... Alt+n): brings the tool to the front, also when detached
        self.tool_actions = []
        for n, (title, win) in enumerate(self.windows.items(), start=1):
            a = QtGui.QAction(tool_icon(_TOOL_LOGOS.get(title, title)), title, self)
            if n <= 9:
                a.setShortcut(QtGui.QKeySequence(f"Alt+{n}"))
            a.setToolTip(_TOOL_INFO.get(title, title))
            a.triggered.connect(lambda _=False, w=win: self.tabs.show_page(w))
            self.addAction(a)                          # shortcut works without opening the menu
            self.tool_actions.append(a)
        from ..core.settings_dialog import open_preferences
        # projects: act on the tool in front
        fmenu = self.menuBar().addMenu("&File")
        self.file_actions = []
        for text, meth, key in (("&Open project…", "on_open_project", QtGui.QKeySequence.StandardKey.Open),
                                ("&Save project…", "on_save_project", QtGui.QKeySequence.StandardKey.Save)):
            a = QtGui.QAction(text, self)
            a.setShortcut(key)
            a.triggered.connect(lambda _=False, m=meth, k=key: self._call_current(m, k))
            fmenu.addAction(a)
            self.file_actions.append(a)
        fmenu.aboutToShow.connect(lambda: self._fill_menu(fmenu, "File", self.file_actions))
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
        self.settings_actions = [act, forget]
        menu.aboutToShow.connect(lambda: self._fill_menu(menu, "Settings", self.settings_actions))

    # ---------------------------------------------------------------- tool menus
    @staticmethod
    def _extra_actions(win, name):
        """Entries of the tool's own menu ``name`` that the launcher menu does not already have."""
        covered = _COVERED.get(name, ())
        out = []
        for a in _tool_menus(win).get(name, []):
            if a.isSeparator() and out and not out[-1].isSeparator():
                out.append(a)
                continue
            if a.isSeparator() or not a.isVisible() or _plain(a.text()) in covered:
                continue
            if any(_same_key(a, k) for k in (QtGui.QKeySequence.StandardKey.Open, QtGui.QKeySequence.StandardKey.Save,
                                             QtGui.QKeySequence.StandardKey.Preferences)):
                continue
            out.append(a)
        while out and out[-1].isSeparator():
            out.pop()
        return out

    def _register_tool_shortcuts(self, win):
        """The tool's menu bar is hidden, so its keyboard shortcuts would be dead: attach the
        actions the launcher menus do not cover to the tool window itself."""
        for name in _MENUS:
            for a in self._extra_actions(win, name):
                if not a.isSeparator() and not a.shortcut().isEmpty():
                    win.addAction(a)

    def _fill_menu(self, menu, name, own):
        menu.clear()
        for a in own:
            menu.addAction(a)
        extra = self._extra_actions(self.tabs.currentWidget(), name)
        if extra:
            menu.addSeparator()
            for a in extra:
                menu.addAction(a)

    def _fill_view_menu(self):
        m = self.view_menu
        m.clear()
        self._fill_tool_view(m, self.tabs.currentWidget())
        # tool tabs <-> windows
        m.addSeparator()
        i = self.tabs.currentIndex()
        if i >= 0:
            t = self.tabs.tabText(i)
            m.addAction(self._icon(t), f"Detach '{t}' into a window", lambda: self.tabs.detach(i))
        for p in self.tabs.detached:
            t = self.tabs.title_of(p)
            m.addAction(self._icon(t), f"Attach '{t}'", lambda pp=p: self.tabs.attach(pp))
        if self.tabs.detached:
            m.addAction("Attach all tools", self.tabs.attach_all)
        # go to a tool
        m.addSection("Tools")
        for a in self.tool_actions:
            m.addAction(a)

    @staticmethod
    def _icon(title):
        return tool_icon(_TOOL_LOGOS.get(title, title))

    def _fill_tool_view(self, m, w):
        docks = getattr(w, "docks", None)
        if not docks:
            own = self._extra_actions(w, "View")      # e.g. the model explorer's own View menu
            for a in own:
                m.addAction(a)
            if own:
                return
            a = m.addAction("(no detachable panels in this tool)")
            a.setEnabled(False)
            return
        for a in getattr(w, "extra_view_actions", ()):   # the tool's own View entries (model explorer: sliders)
            m.addAction(a)
        for a in w.view_actions.values():
            m.addAction(a)
        m.addSeparator()
        if getattr(w, "border_actions", None):
            sub = m.addMenu("Panel border")
            for a in w.border_actions.values():
                sub.addAction(a)
        m.addAction("Dock all panels", w.dock_all)
        m.addAction("Reset layout", w.reset_layout)

    def _call_current(self, method, key=None):
        w = self.tabs.currentWidget()
        f = getattr(w, method, None)
        if callable(f):
            f()
            return
        # a tool with its own File menu (model explorer): the entry with the same shortcut or name
        want = _plain(method.replace("on_", "").replace("_", " "))
        for a in _tool_menus(w).get("File", []):
            if (key is not None and _same_key(a, key)) or _plain(a.text()) == want:
                a.trigger()
                return
        QtWidgets.QMessageBox.information(self, "Projects", "This tool has no project files.")

    def forget_sessions(self):
        for w in self.windows.values():
            f = getattr(w, "forget_session", None)
            if callable(f):
                f()
        try:
            os.remove(self._layout_path())          # detached tools, window positions
        except OSError:
            pass
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
                self.tabs.show_page(w)
                w.open_project(path)
                return title
        QtWidgets.QMessageBox.critical(self, "Open project", f"{path}: no tool '{tool}' in this window")
        return None

    # ---------------------------------------------------------------- detached tools (last session)
    @staticmethod
    def _layout_path():
        from ..core import settings as prefs
        return os.path.join(os.path.dirname(prefs.settings_path()), "launcher_session.json")

    def save_layout(self):
        import json
        try:
            p = self._layout_path()
            os.makedirs(os.path.dirname(p), exist_ok=True)
            g = self.geometry()
            with open(p, "w", encoding="utf-8") as fh:
                json.dump({"detached": self.tabs.state(), "geometry": [g.x(), g.y(), g.width(), g.height()],
                           "current": self.tabs.tabText(self.tabs.currentIndex())}, fh, indent=1)
        except OSError:
            pass

    def restore_layout(self):
        import json
        try:
            with open(self._layout_path(), encoding="utf-8") as fh:
                st = json.load(fh)
        except (OSError, ValueError):
            return False
        g = st.get("geometry")
        if isinstance(g, list) and len(g) == 4:
            self.setGeometry(*[int(v) for v in g])
        # tools that were renamed (pySASbox -> pyTOOLbox) keep their place
        detached = {ALIASES.get(k, k): v for k, v in (st.get("detached") or {}).items()}
        self.tabs.restore_state(detached)
        current = st.get("current") or ""
        cur = self.tabs.page_for(ALIASES.get(current, current))
        if cur is not None and not self.tabs.is_detached(cur):
            self.tabs.setCurrentWidget(cur)
        return True

    def window_for(self, title):
        return self.windows.get(title)

    def closeEvent(self, event):
        for w in self.windows.values():
            may_close = getattr(w, "may_close", None)    # e.g. pyTOOLbox: edited values not yet written
            if callable(may_close) and not may_close():
                event.ignore()
                return
        self.save_layout()
        self.tabs.close_detached()
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
    app.setWindowIcon(tool_icon("pySASfit"))
    from ..core.tabstyle import apply_tab_style
    apply_tab_style(app)
    w = Launcher()
    w.show()
    skip = set()
    if len(argv) > 1 and os.path.isfile(argv[1]):
        if argv[1].endswith(".pysasfit"):
            t = w.open_project(argv[1])
            if t:
                skip.add(t)
        else:
            sld = w.window_for("pySLDprofile")
            if sld is not None:
                sld.load_file(argv[1])
                skip.add("pySLDprofile")
    w.restore_sessions(skip)
    w.restore_layout()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
