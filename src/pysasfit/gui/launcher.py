"""One window for all pysasfit tools.

    pysasfit-gui [datafile | project.pysasfit]
    python -m pysasfit.gui.launcher [datafile | project.pysasfit]

Every tool keeps its own main window; the launcher hosts them as tabs. A tool
whose import fails (e.g. a missing optional dependency) gets a tab with the
error message instead of breaking the whole launcher. A data file given on the
command line is loaded into the pySLDprofile tab, a project file into its tool.
The other tools reopen their last session (see :mod:`pysasfit.core.project`).

Fast start: the launcher only creates the tabs. A tool is imported and built when it is first
needed -- its tab is opened, it is detached, a project of it is opened, Alt+n -- and its last
session is restored then. Nothing is recalculated or rescanned at the start: the tools show the
inputs of the last session and calculate when their button is pressed (pySASred reads the last
raw-data folder on Rescan, pySCAtter on Calculate 2D, pySEBpolymers on Calculate, pySASmorph on
Correlations, pySASstudio on Calculate, pySASscout on Search); pyATsas and pyDENsity show the installation found by the
last search (Search again searches, see :mod:`pysasfit.core.installcache`).

To add a tool, append ``(title, "module.path", "ClassName")`` to ``TOOLS``.

The tools' own menu bars are hidden; the launcher's File, View and Settings menus act on the
tool in front and also list that tool's own entries the launcher does not provide itself.

The tools are chosen in a column of icons at the left edge (:mod:`pysasfit.gui.sidebar`): a row of
tabs does not fit the screen with this many tools. Hovering an icon names the tool; the button at the
top of the column shows the names next to the icons.

Every tool can be detached into its own window (drag its icon out of the side bar,
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
from ..resources import ALIASES, TOOL_COLOURS, tool_icon, wordmark
from .sidebar import ToolSideBar

TOOLS = [
    ("pySASred", "pysasfit.sasred.gui", "MainWindow"),
    ("pySASstudio", "pysasfit.studio.explorer", "ModelExplorer"),
    ("pySLDprofile", "pysasfit.sldprofile.gui", "MainWindow"),
    ("pyINVersion", "pysasfit.inversion.gui.app", "MainWindow"),
    ("pyOZsolver", "pysasfit.oz.gui.main_window", "OZWidget"),
    ("pySASmorph", "pysasfit.morph.gui", "MainWindow"),
    ("pySCAtter", "pysasfit.scatter.gui", "MainWindow"),
    ("pySEBpolymers", "pysasfit.seb.gui", "MainWindow"),
    ("pyATsas", "pysasfit.atsas.gui", "MainWindow"),
    ("pyDENsity", "pysasfit.density.gui", "MainWindow"),
    ("pyDENSSmulti", "pysasfit.densitymulti.gui", "MainWindow"),
    ("pySLDcalculator", "pysasfit.sld.gui", "MainWindow"),
    ("pySASscout", "pysasfit.scout.gui", "MainWindow"),
    ("pyTOOLbox", "pysasfit.toolbox.gui", "MainWindow"),
]

# tab title -> preferences page
_TOOL_KEYS = {"pySLDprofile": "sldprofile", "pyINVersion": "inversion", "pyOZsolver": "oz",
             "pySASred": "sasred", "pyTOOLbox": "toolbox", "pySCAtter": "scatter", "pySEBpolymers": "seb", "pyATsas": "atsas",
             "pyDENsity": "density", "pyDENSSmulti": "densitymulti", "pySASscout": "scout"}

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
    "pySEBpolymers": "form factors of structures of linked sub-units (micelles, stars, dendrimers, bottle brushes, ...; SEB / Svaneborg-Pedersen formalism), fits to 1D data",
    "pyATsas": "the ATSAS programs (EMBL Hamburg / BIOSAXS, installed separately): Guinier analysis, p(r), model "
               "curves, bead models; starts PRIMUS, CHROMIXS, PEAK, ...",
    "pyDENsity": "electron-density maps from SAXS/SANS curves with DENSS (Grant 2018; installed separately): single "
                 "and averaged maps, isosurfaces, slices",
    "pyDENSSmulti": "3D maps from several SAS contrasts (SANS contrast variation, SANS + SAXS): volume "
                    "fractions of the components (match points, H/D exchange) or a DENSS-Multiple-type SLD "
                    "map; own DENSS-type iteration, no installation needed",
    "pySLDcalculator": "scattering length densities of materials and biomolecules for neutrons and X-rays "
                       "(energy dependent across absorption edges): molecular volume, contrast, excess "
                       "scattering length, H/D exchange and match points",
    "pySASscout": "model search: proposes and ranks form factor, size distribution and structure factor (also "
                  "mixtures) for a measured curve, with AICc/BIC, Akaike weights and warnings",
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


class _Pending(QtWidgets.QWidget):
    """Tab page of a tool that is not built yet (the tools are built when they are first shown)."""

    def __init__(self, title):
        super().__init__()
        self.title = title
        lay = QtWidgets.QVBoxLayout(self)
        lab = QtWidgets.QLabel(f"{title} is started when its tab is opened …")
        lab.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        lab.setStyleSheet("color: gray;")
        lay.addWidget(lab)


class Launcher(QtWidgets.QMainWindow):
    """The tools are imported and built when they are first needed (their tab is opened, detached, a
    project of theirs is opened, ...), and their last session is restored then: the launcher itself starts
    at once. ``lazy=False`` builds all of them immediately."""

    def __init__(self, tools=None, lazy=True):
        super().__init__()
        self.setWindowTitle("pySASfit")
        self.setWindowIcon(tool_icon("pySASfit"))
        self.tabs = DetachableTabWidget()
        self.tabs.setDocumentMode(True)
        # the tools are chosen in a column of icons at the left (a row of tabs does not fit the screen any
        # more); the tab widget only holds the pages, its tab bar is hidden
        self.tabs.tabBar().hide()
        self.sidebar = ToolSideBar()
        central = QtWidgets.QWidget()
        cl = QtWidgets.QHBoxLayout(central)
        cl.setContentsMargins(0, 0, 0, 0)
        cl.setSpacing(0)
        cl.addWidget(self.sidebar)
        line = QtWidgets.QFrame()
        line.setFrameShape(QtWidgets.QFrame.Shape.VLine)
        line.setFrameShadow(QtWidgets.QFrame.Shadow.Sunken)
        cl.addWidget(line)
        cl.addWidget(self.tabs, 1)
        self.setCentralWidget(central)
        self.windows = {}                 # title -> tool window (only the tools built so far)
        self._specs = {}                  # title -> (module, class)
        self._pending = {}                # title -> placeholder page
        self._restore = False             # restore the last session of a tool when it is built
        self._skip = set()                # ... except these (a project / data file was opened in them)
        for title, module, cls in tools or TOOLS:
            self._specs[title] = (module, cls)
            page = _Pending(title)
            self._pending[title] = page
            logo = _TOOL_LOGOS.get(title, title)
            i = self.tabs.addTab(page, tool_icon(logo), title)
            self.tabs.setTabToolTip(i, _TOOL_INFO.get(title, title))
            self.tabs.set_wordmark(i, wordmark(logo))     # coloured label as in the logo
            n = len(self._specs)
            tip = f"<b>{title}</b><br>{_TOOL_INFO.get(title, '')}" + (f"<br><i>Alt+{n}</i>" if n <= 9 else "")
            mark = wordmark(logo)
            self.sidebar.add_tool(title, tool_icon(logo), TOOL_COLOURS.get(ALIASES.get(logo, logo), ("#5E7388",))[0],
                                  tip, letters=mark[1][0] if mark and len(mark) > 1 else title[2:5],
                                  segments=mark)
        self.tabs.resolve_page = self._resolve_page
        self.tabs.currentChanged.connect(self._on_current_changed)
        self.tabs.currentChanged.connect(lambda _i: self._sync_sidebar())
        self.tabs.detached_changed.connect(self._on_detached_changed)
        self.sidebar.activated.connect(self._sidebar_clicked)
        self.sidebar.detach_requested.connect(self._sidebar_detach)
        self.sidebar.context_requested.connect(self._sidebar_menu)
        self._sync_sidebar()
        self.apply_preferences()
        import weakref
        from ..core.settings_dialog import on_saved
        ref = weakref.ref(self)
        on_saved(lambda: ref() is not None and ref().apply_preferences())
        if not lazy:
            for title in list(self._specs):
                self.tool(title)
        self.resize(1500, 950)
        # one entry per tool (View menu, Alt+1 ... Alt+n): brings the tool to the front, also when detached
        self.tool_actions = []
        for n, title in enumerate(self._specs, start=1):
            a = QtGui.QAction(tool_icon(_TOOL_LOGOS.get(title, title)), title, self)
            if n <= 9:
                a.setShortcut(QtGui.QKeySequence(f"Alt+{n}"))
            a.setToolTip(_TOOL_INFO.get(title, title))
            a.triggered.connect(lambda _=False, t=title: self.show_tool(t))
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

    # ---------------------------------------------------------------- building the tools when needed
    def is_built(self, title):
        return title in self.windows

    def tool(self, title):
        """The window of the tool ``title``, built now if it is not yet (None: unknown tool or it could not be
        built; its tab then shows the error)."""
        if title in self.windows:
            return self.windows[title]
        page = self._pending.get(title)
        if page is None:
            return None
        module, cls = self._specs[title]
        QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.CursorShape.WaitCursor)
        try:
            try:
                win = getattr(importlib.import_module(module), cls)()
                win.setWindowFlags(QtCore.Qt.WindowType.Widget)  # embed the QMainWindow
                # one Settings menu for the whole launcher (below), not one per tab
                if isinstance(win, QtWidgets.QMainWindow) and win.menuBar() is not None:
                    win.menuBar().setVisible(False)
                self._register_tool_shortcuts(win)
            except Exception:  # noqa: BLE001 -- show, don't crash
                err = _error_tab(title, traceback.format_exc())
                del self._pending[title]
                self.tabs.replace_page(page, err)
                self.tabs.setTabText(self.tabs.indexOf(err), title + " (error)")
                page.deleteLater()
                return None
            del self._pending[title]
            self.windows[title] = win
            self.tabs.replace_page(page, win)
            page.deleteLater()
            self._sync_sidebar()
            if self._restore and title not in self._skip:
                f = getattr(win, "restore_session", None)
                if callable(f):
                    try:
                        f()
                    except Exception:  # noqa: BLE001 -- the last session is a convenience
                        traceback.print_exc()
        finally:
            QtWidgets.QApplication.restoreOverrideCursor()
        return win

    def show_tool(self, title):
        w = self.tool(title)
        if w is not None:
            self.tabs.show_page(w)
        else:                                        # could not be built: its tab shows the error
            p = self._page(title)
            if p is not None:
                self.tabs.show_page(p)
        self._sync_sidebar()
        return w

    def apply_preferences(self):
        """Preferences of the window itself (page 'pySASfit window'): colours of the side bar."""
        from ..core import settings as prefs
        try:
            self.sidebar.set_selected_colours(prefs.get("launcher.sidebar.selected"),
                                              prefs.get("launcher.sidebar.selected_dark"))
        except Exception:  # noqa: BLE001 -- a colour, not worth an error
            pass

    # ---------------------------------------------------------------- side bar (tool icons at the left)
    def _page(self, title):
        """The page of ``title``: its window, the placeholder, or the error page."""
        return (self.windows.get(title) or self._pending.get(title)
                or self.tabs.page_for(title + " (error)") or self.tabs.page_for(title))

    def _title(self, page):
        if isinstance(page, _Pending):
            return page.title
        for t, w in self.windows.items():
            if w is page:
                return t
        t = self.tabs.title_of(page) if page is not None else ""
        return t[:-len(" (error)")] if t.endswith(" (error)") else t

    def _sync_sidebar(self):
        self.sidebar.set_current(self._title(self.tabs.currentWidget()) if self.tabs.count() else None)
        for title in self._specs:
            w = self.windows.get(title)
            self.sidebar.set_state(title, "" if w is None else
                                   "detached" if self.tabs.is_detached(w) else "loaded")

    def _on_detached_changed(self, _page, detached):
        # a tool left the window: show a tool that is loaded already, not one that would have to be built
        if detached and isinstance(self.tabs.currentWidget(), _Pending):
            for w in self.windows.values():
                if self.tabs.indexOf(w) >= 0:
                    self.tabs.setCurrentWidget(w)
                    break
        self._sync_sidebar()

    def _sidebar_clicked(self, title):
        page = self._pending.get(title)
        if page is not None and self.tabs.indexOf(page) >= 0:
            self.tabs.setCurrentWidget(page)         # the placeholder is shown, then the tool is built
            self._on_current_changed(self.tabs.currentIndex())
            self._sync_sidebar()
        else:
            self.show_tool(title)

    def _sidebar_detach(self, title, gpos):
        w = self.tool(title)
        if w is None:
            return
        if self.tabs.is_detached(w):
            self.tabs.show_page(w)
        else:
            self.tabs.detach(w, pos=None if gpos is None or gpos.isNull() else gpos)

    def _sidebar_menu(self, title, gpos):
        m = QtWidgets.QMenu(self)
        w = self.windows.get(title)
        m.addAction(self._icon(title), f"Show {title}", lambda: self._sidebar_clicked(title))
        if w is not None and self.tabs.is_detached(w):
            m.addAction(f"Attach '{title}'", lambda: self.tabs.attach(w))
        else:
            m.addAction(f"Detach '{title}' into a window", lambda: self._sidebar_detach(title, QtCore.QPoint()))
        if self.tabs.detached:
            m.addAction("Attach all tools", self.tabs.attach_all)
        m.addSeparator()
        a = m.addAction("Tool names next to the icons")
        a.setCheckable(True)
        a.setChecked(self.sidebar.expanded())
        a.toggled.connect(self.sidebar.set_expanded)
        m.exec(gpos)

    def _resolve_page(self, page):
        if isinstance(page, _Pending):
            return self.tool(page.title) or self.tabs.page_for(page.title + " (error)") or page
        return page

    def _on_current_changed(self, i):
        page = self.tabs.widget(i)
        if isinstance(page, _Pending):
            # let the window paint first (the placeholder says what is happening), then build
            QtCore.QTimer.singleShot(0, lambda t=page.title: self._build_if_current(t))

    def _build_if_current(self, title):
        page = self._pending.get(title)
        if page is not None and self.tabs.currentWidget() is page:
            self.tool(title)

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
        cur = self._current()
        extra = self._extra_actions(cur, name)
        # "Copy Python code" of a tool without a menu bar (the OZ solver tabs)
        cp = getattr(cur, "copy_python_action", None)
        if name == "File" and cp is not None and cp not in extra:
            extra.append(cp)
        if extra:
            menu.addSeparator()
            for a in extra:
                menu.addAction(a)

    def _fill_view_menu(self):
        m = self.view_menu
        m.clear()
        self._fill_tool_view(m, self._current())
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
        m.addSeparator()
        a = m.addAction("Tool names next to the icons (side bar)")
        a.setCheckable(True)
        a.setChecked(self.sidebar.expanded())
        a.toggled.connect(self.sidebar.set_expanded)

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

    def _current(self):
        """The tool in front (built now if it is not yet)."""
        return self._resolve_page(self.tabs.currentWidget())

    def _call_current(self, method, key=None):
        w = self._current()
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
        restore, self._restore = self._restore, False
        try:
            for title in list(self._specs):
                w = self.tool(title)        # a tool not built yet: built now (without its session) to forget it
                f = getattr(w, "forget_session", None)
                if callable(f):
                    f()
        finally:
            self._restore = restore
        try:
            os.remove(self._layout_path())          # detached tools, window positions
        except OSError:
            pass
        self.statusBar().showMessage("last sessions deleted", 8000)

    def restore_sessions(self, skip=()):
        """Restore the last session of the tools built so far, and of every other tool when it is built
        (not of the tools in ``skip``)."""
        self._skip = set(skip)
        self._restore = True
        for title, w in list(self.windows.items()):
            f = getattr(w, "restore_session", None)
            if callable(f) and title not in self._skip:
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
            if key != tool or title not in self._specs:
                continue
            self._skip.add(title)                    # the project, not the last session
            w = self.tool(title)
            if hasattr(w, "open_project"):
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
                           "current": self._title(self.tabs.currentWidget()),
                           "sidebar_names": self.sidebar.expanded()}, fh, indent=1)
        except OSError:
            pass

    def restore_layout(self):
        import json
        try:
            with open(self._layout_path(), encoding="utf-8") as fh:
                st = json.load(fh)
        except (OSError, ValueError):
            return False
        self.sidebar.set_expanded(bool(st.get("sidebar_names", False)))
        g = st.get("geometry")
        if isinstance(g, list) and len(g) == 4:
            self.setGeometry(*[int(v) for v in g])
        # tools that were renamed (pySASbox -> pyTOOLbox) keep their place
        detached = {ALIASES.get(k, k): v for k, v in (st.get("detached") or {}).items()}
        self.tabs.restore_state(detached)            # the detached tools are built now
        current = st.get("current") or ""
        cur = self.tabs.page_for(ALIASES.get(current, current))
        if cur is not None and not self.tabs.is_detached(cur):
            self.tabs.setCurrentWidget(cur)
        return True

    def window_for(self, title):
        """The tool window ``title`` (built now if needed)."""
        return self.tool(title)

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
    w._on_current_changed(w.tabs.currentIndex())     # build the tool in front (after the window is shown)
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
