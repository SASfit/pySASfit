import os

import pytest

pytest.importorskip("PySide6")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6 import QtWidgets  # noqa: E402

from pysasfit.gui.launcher import Launcher  # noqa: E402

DATA = os.path.join(os.path.dirname(__file__), "..", "..", "examples", "sldprofile", "data",
                    "inhomogeneous_sphere_r5nm.dat")


def test_launcher_hosts_all_tools():
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    from pysasfit.gui.launcher import TOOLS
    w = Launcher(lazy=False)
    titles = [w.tabs.tabText(i) for i in range(w.tabs.count())]
    assert titles == [t for t, _m, _c in TOOLS], titles  # no "(error)" tabs
    oz = w.window_for("pyOZsolver")
    assert oz.count() >= 4 and not any("(error)" in oz.tabText(i) for i in range(oz.count()))
    w.window_for("pySLDprofile").load_file(DATA)
    w.tabs.setCurrentIndex(3)
    app.processEvents()
    w.close()


def test_broken_tool_gets_error_tab():
    QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    w = Launcher(tools=[("broken", "pysasfit.does_not_exist", "MainWindow")])
    assert w.tabs.tabText(0) == "broken"          # not built yet
    assert w.tool("broken") is None
    assert w.tabs.tabText(0) == "broken (error)"
    w.close()


def test_top_level_lazy_import():
    import pysasfit

    assert pysasfit.core.load_data is pysasfit.sldprofile.load_data
    assert pysasfit.core.Resolution is pysasfit.sldprofile.Resolution


def test_every_plot_has_a_navigation_toolbar():
    """Home/back/forward/pan/zoom/subplots/edit/save on every figure of every tool."""
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    w = Launcher(lazy=False)
    app.processEvents()
    canvases = w.findChildren(FigureCanvasQTAgg)
    toolbars = w.findChildren(NavigationToolbar2QT)
    assert len(canvases) >= 10
    withBar = {id(tb.canvas) for tb in toolbars}
    missing = [type(c.parent()).__name__ for c in canvases if id(c) not in withBar]
    assert not missing, missing
    # ... and the toolbar sits ABOVE its plot everywhere
    for tb in toolbars:
        lay = tb.parentWidget().layout()
        if lay is not None and lay.indexOf(tb.canvas) >= 0:
            assert lay.indexOf(tb) < lay.indexOf(tb.canvas), type(tb.parentWidget()).__name__
    w.close()


def test_tool_menus_are_merged_into_the_launcher():
    """The tools' menu bars are hidden in the launcher: their own entries (model explorer:
    Save project as, SASfit installation, Sliders; SANS reduction: Save NXcanSAS) appear in
    the launcher's File / Settings / View menus, without duplicating Open/Save/Preferences."""
    pytest.importorskip("h5py")
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    w = Launcher(tools=[("pySASred", "pysasfit.sasred.gui", "MainWindow"),
                        ("pySASstudio", "pysasfit.studio.explorer", "ModelExplorer")])
    assert [w.tabs.tabText(i) for i in range(w.tabs.count())] == ["pySASred", "pySASstudio"]
    menus = {a.text().replace("&", ""): a.menu() for a in w.menuBar().actions()}

    def texts(name):
        m = menus[name]
        m.aboutToShow.emit()
        return [a.text().replace("&", "") for a in m.actions() if not a.isSeparator()]

    w.tabs.setCurrentIndex(0)
    f = texts("File")
    assert "Save NXcanSAS…" in f and sum("Open project" in t for t in f) == 1
    s = texts("Settings")
    assert sum(t.startswith("Preferences") for t in s) == 1
    w.tabs.setCurrentIndex(1)
    app.processEvents()
    f = texts("File")
    assert "Save project as…" in f and sum("Save project" == t.rstrip("…") for t in f) == 1
    assert any("SASfit installation" in t for t in texts("Settings"))
    assert "Sliders" in texts("View")
    w.close()


def test_launcher_starts_without_building_the_tools(monkeypatch):
    """Fast start: no tool is imported or built, no installation searched, until its tab is opened."""
    import subprocess
    import sys

    calls = []
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: calls.append(a) or None)
    monkeypatch.setattr(subprocess, "Popen", lambda *a, **k: calls.append(a) or None)
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    w = Launcher(tools=[("pySEBpolymers", "pysasfit.seb.gui", "MainWindow"),
                        ("pyDENsity", "pysasfit.density.gui", "MainWindow"),
                        ("pyATsas", "pysasfit.atsas.gui", "MainWindow")])
    assert w.windows == {} and not calls
    w.restore_sessions()
    w.tabs.setCurrentIndex(1)                 # opening the tab builds the tool (after the window painted)
    for _ in range(3):
        app.processEvents()
    assert list(w.windows) == ["pyDENsity"] and not calls     # no DENSS search at the start
    assert w.tabs.currentWidget() is w.windows["pyDENsity"]
    assert w.windows["pyDENsity"].installed in (None, True, False)
    a = w.window_for("pyATsas")              # built on request
    assert a is w.windows["pyATsas"] and w.tabs.tabText(w.tabs.indexOf(a)) == "pyATsas"
    assert not calls
    # the tool's own keys, detach / attach work with a built tool
    win = w.tabs.detach(w.tabs.page_for("pySEBpolymers"))
    assert win is w.windows["pySEBpolymers"]
    w.tabs.attach(win)
    w.close()
    del sys


def test_lazy_tool_restores_its_session_when_built(monkeypatch):
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    from pysasfit.seb.gui import MainWindow
    seen = []
    monkeypatch.setattr(MainWindow, "restore_session", lambda self: seen.append(self) or True)
    w = Launcher(tools=[("pySEBpolymers", "pysasfit.seb.gui", "MainWindow")])
    w.restore_sessions()
    assert not seen
    win = w.tool("pySEBpolymers")
    assert seen == [win]
    w.close()
    w2 = Launcher(tools=[("pySEBpolymers", "pysasfit.seb.gui", "MainWindow")])
    w2.restore_sessions(skip={"pySEBpolymers"})
    w2.tool("pySEBpolymers")
    assert len(seen) == 1                       # skipped (a project was opened in it)
    w2.close()
    app.processEvents()


def test_side_bar_replaces_the_row_of_tabs():
    """One icon per tool in a column at the left; hovering names the tool; the row of tabs is hidden."""
    from PySide6 import QtCore

    from pysasfit.gui.launcher import TOOLS
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    w = Launcher(tools=[("pySEBpolymers", "pysasfit.seb.gui", "MainWindow"),
                        ("pyDENsity", "pysasfit.density.gui", "MainWindow"),
                        ("pySCAtter", "pysasfit.scatter.gui", "MainWindow")])
    w.show()
    app.processEvents()
    sb = w.sidebar
    assert not w.tabs.tabBar().isVisible()
    assert list(sb.buttons) == ["pySEBpolymers", "pyDENsity", "pySCAtter"]
    for t, b in sb.buttons.items():
        assert t in b.toolTip() and not b.icon().isNull()            # the name when the mouse is over it
        assert b.toolButtonStyle() == QtCore.Qt.ToolButtonStyle.ToolButtonIconOnly
    assert sb.width() < 60 and not w.windows
    sb.buttons["pyDENsity"].click()                                   # a click loads and shows the tool
    for _ in range(3):
        app.processEvents()
    assert list(w.windows) == ["pyDENsity"] and sb.current() == "pyDENsity"
    assert sb.buttons["pyDENsity"].state == "loaded" and sb.buttons["pySCAtter"].state == ""
    assert not w.tabs.tabBar().isVisible()
    # detach (double click / drag out / right mouse button) and attach
    w._sidebar_detach("pyDENsity", QtCore.QPoint())
    assert sb.buttons["pyDENsity"].state == "detached" and w.tabs.is_detached(w.windows["pyDENsity"])
    assert list(w.windows) == ["pyDENsity"]                           # detaching loads no other tool
    w.tabs.attach(w.windows["pyDENsity"])
    assert sb.buttons["pyDENsity"].state == "loaded" and sb.current() == "pyDENsity"
    # names next to the icons, kept for the next start
    sb.toggle.click()
    assert sb.expanded() and sb.width() > 100
    assert sb.buttons["pySCAtter"].toolButtonStyle() == QtCore.Qt.ToolButtonStyle.ToolButtonTextBesideIcon
    # the names as in the logos (grey py, bold core in the tool's colour, grey ending), flush left
    seg = sb.buttons["pySCAtter"].segments
    assert [t for t, *_ in seg] == ["py", "SCA", "tter"] and seg[1][3] and not seg[0][3]
    assert sb.width() >= sb.buttons["pySEBpolymers"].name_width() + 40
    w.save_layout()
    w.close()
    w2 = Launcher(tools=[("pySEBpolymers", "pysasfit.seb.gui", "MainWindow")])
    w2.restore_layout()
    assert w2.sidebar.expanded()
    w2.sidebar.set_expanded(False)
    w2.save_layout()
    w2.close()
    assert len(TOOLS) >= 12


def test_the_window_can_be_made_small():
    """No tool fixes a large minimum size of the pySASfit window: their panels scroll
    (pysasfit.core.scrolling). Before: 1227 x 662 and more, set by the widest / tallest tool."""
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    w = Launcher(lazy=False)
    w.show()
    for _ in range(5):
        app.processEvents()
    big = {t: (x.minimumSizeHint().width(), x.minimumSizeHint().height()) for t, x in w.windows.items()
           if x.minimumSizeHint().width() > 520 or x.minimumSizeHint().height() > 480}
    assert not big, big
    w.resize(640, 520)
    app.processEvents()
    assert (w.width(), w.height()) == (640, 520)
    w.close()


def test_selected_tool_background_is_a_preference_with_good_contrast():
    """The background of the tool in front (Preferences → pySASfit window) keeps every name readable:
    contrast >= 3 : 1 for the tool colours and >= 4.5 : 1 for the grey 'py' / ending, on both themes."""
    from pysasfit.core import settings as prefs
    from pysasfit.gui.sidebar import contrast
    from pysasfit.resources import COLOURS, TOOL_COLOURS, WORDMARKS

    light, dark = prefs.get("launcher.sidebar.selected"), prefs.get("launcher.sidebar.selected_dark")
    for k, bg in enumerate((light, dark)):
        assert contrast(COLOURS["py"][k], bg) >= 4.5
        for t in WORDMARKS:
            if t != "pySASfit":
                assert contrast(TOOL_COLOURS[t][k], bg) >= 3.0, (t, bg)
    assert contrast(TOOL_COLOURS["pySASstudio"][0], "#0078D4") < 1.5      # why not the Windows selection blue
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    w = Launcher(tools=[("pySEBpolymers", "pysasfit.seb.gui", "MainWindow")])
    assert w.sidebar.buttons["pySEBpolymers"].selected == (light, dark)
    old = prefs.get("launcher.sidebar.selected")
    try:
        prefs.set_value("launcher.sidebar.selected", "#eaf7ea")
        assert prefs.get("launcher.sidebar.selected") == "#EAF7EA"
        w.apply_preferences()
        assert w.sidebar.buttons["pySEBpolymers"].selected[0] == "#EAF7EA"
        with pytest.raises(ValueError):
            prefs.set_value("launcher.sidebar.selected", "yellow")
    finally:
        prefs.set_value("launcher.sidebar.selected", old)
    from pysasfit.core.settings_dialog import PreferencesDialog
    d = PreferencesDialog(w, "launcher")                     # the colour field with its chooser button
    row = next(r for r in d.rows if r.s.key == "launcher.sidebar.selected")
    assert row.value() == light and row.swatch is not None
    row.w.setText("nonsense")
    with pytest.raises(ValueError):
        row.value()
    d.close()
    w.close()
