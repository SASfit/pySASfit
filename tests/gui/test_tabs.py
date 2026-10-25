"""Detachable tabs (pysasfit.core.tabs) and their use in the launcher."""
import os

import pytest

pytest.importorskip("PySide6")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6 import QtCore, QtGui, QtWidgets  # noqa: E402

from pysasfit.core.tabs import DetachableTabWidget  # noqa: E402


def _app():
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


def _titles(t):
    return [t.tabText(i) for i in range(t.count())]


def _tabs():
    t = DetachableTabWidget()
    pages = {}
    for name in "ABCD":
        if name == "C":                                       # a QMainWindow page with a hidden menu bar
            p = QtWidgets.QMainWindow()
            p.menuBar().addMenu("&File").addAction("Do something")
            p.setWindowFlags(QtCore.Qt.WindowType.Widget)
            p.menuBar().setVisible(False)
        else:
            p = QtWidgets.QLabel(name)
        pages[name] = p
        t.addTab(p, name)
    t.resize(600, 400)
    t.show()
    return t, pages


def test_detach_and_attach_keep_the_order():
    app = _app()
    t, pages = _tabs()
    wb = t.detach(1)
    assert _titles(t) == ["A", "C", "D"] and t.is_detached(pages["B"]) and wb.isVisible()
    wc = t.detach(pages["C"])
    assert wc is pages["C"] and wc.isWindow() and not wc.menuBar().isHidden()   # own menu bar shown
    assert wc.menuBar().cornerWidget(QtCore.Qt.Corner.TopRightCorner).text() == "Attach"
    t.attach(pages["C"])
    assert _titles(t) == ["A", "C", "D"] and pages["C"].menuBar().isHidden()     # hidden again
    assert not pages["C"].isWindow()
    wb.close()                                                # closing the window attaches it
    app.processEvents()
    assert _titles(t) == ["A", "B", "C", "D"] and not t.detached
    t.close()


def test_reordered_tabs_and_state_roundtrip():
    _app()
    t, pages = _tabs()
    t.tabBar().moveTab(3, 0)                                  # D A B C
    t.detach(pages["A"])
    st = t.state()
    assert list(st) == ["A"] and len(st["A"]) == 4
    t.attach_all()
    assert _titles(t) == ["D", "A", "B", "C"]                 # back to its place in the new order
    t.restore_state(st)
    assert t.is_detached(pages["A"])
    t.show_page(pages["A"])                                   # raises its window
    t.close_detached()
    assert not t.detached[pages["A"]].isVisible()
    t.close()


def test_double_click_and_drag_out_detach():
    app = _app()
    t, pages = _tabs()
    bar = t.tabBar()
    r = bar.tabRect(2)
    QtCore.QCoreApplication.sendEvent(bar, QtGui.QMouseEvent(
        QtCore.QEvent.Type.MouseButtonDblClick, QtCore.QPointF(r.center()), QtCore.QPointF(bar.mapToGlobal(r.center())),
        QtCore.Qt.MouseButton.LeftButton, QtCore.Qt.MouseButton.LeftButton, QtCore.Qt.KeyboardModifier.NoModifier))
    assert t.is_detached(pages["C"])
    r = bar.tabRect(0)
    c = QtCore.QPointF(r.center())
    for typ, pos, btns in ((QtCore.QEvent.Type.MouseButtonPress, c, QtCore.Qt.MouseButton.LeftButton),
                           (QtCore.QEvent.Type.MouseMove, c + QtCore.QPointF(0, 200), QtCore.Qt.MouseButton.LeftButton)):
        QtCore.QCoreApplication.sendEvent(bar, QtGui.QMouseEvent(
            typ, pos, QtCore.QPointF(bar.mapToGlobal(pos.toPoint())), QtCore.Qt.MouseButton.LeftButton, btns,
            QtCore.Qt.KeyboardModifier.NoModifier))
    for _ in range(3):
        app.processEvents()
    assert t.is_detached(pages["A"]) and _titles(t) == ["B", "D"]
    t.attach_all()
    assert _titles(t) == ["A", "B", "C", "D"]
    t.close()


def test_launcher_tool_detached_and_restored(tmp_path, monkeypatch):
    pytest.importorskip("h5py")
    _app()
    from pysasfit.gui.launcher import Launcher
    tools = [("pySASred", "pysasfit.sasred.gui", "MainWindow"), ("pyTOOLbox", "pysasfit.toolbox.gui", "MainWindow")]
    w = Launcher(tools=tools)
    w.show()
    box = w.window_for("pyTOOLbox")
    win = w.tabs.detach(w.tabs.indexOf(box))
    assert win is box and box.menuBar().isVisible()           # pyTOOLbox in its own window, own menus
    assert [w.tabs.tabText(i) for i in range(w.tabs.count())] == ["pySASred"]
    w.view_menu.aboutToShow.emit()
    assert "Attach 'pyTOOLbox'" in [a.text() for a in w.view_menu.actions()]
    w.close()                                                 # saves the layout, hides the window
    assert not box.isVisible()
    w2 = Launcher(tools=tools)
    w2.restore_layout()
    assert w2.tabs.is_detached(w2.window_for("pyTOOLbox"))
    w2.tabs.attach_all()
    w2.forget_sessions()
    w2.close()


def test_tool_tabs_carry_the_coloured_wordmark():
    pytest.importorskip("h5py")
    _app()
    from pysasfit.gui.launcher import Launcher
    from pysasfit.resources import COLOURS
    w = Launcher(tools=[("pySASred", "pysasfit.sasred.gui", "MainWindow"), ("pyTOOLbox", "pysasfit.toolbox.gui", "MainWindow")])
    bar = w.tabs.tabBar()
    segs = bar.tabData(0)
    assert [s[0] for s in segs] == ["py", "SAS", "red"] and segs[1][3] is True
    assert segs[2][1] == COLOURS["suffix"][0] and segs[0][1] == COLOURS["py"][0]
    assert w.tabs.tabToolTip(0).startswith("SANS data reduction")
    w.show()
    w.grab()                                                  # paints the wordmarks
    box = w.window_for("pyTOOLbox")
    w.tabs.detach(w.tabs.indexOf(box))
    w.tabs.attach(box)                                        # the wordmark survives detach / attach
    assert [s[0] for s in bar.tabData(w.tabs.indexOf(box))] == ["py", "TOOL", "box"]
    assert not w.tabs.tabIcon(0).isNull()                     # menu icon of the tool
    # View menu: one entry per tool with its icon; Alt+2 brings pyTOOLbox to the front
    assert [a.text() for a in w.tool_actions] == ["pySASred", "pyTOOLbox"]
    assert w.tool_actions[1].shortcut().toString() == "Alt+2" and not w.tool_actions[1].icon().isNull()
    w.tool_actions[1].trigger()
    assert w.tabs.currentWidget() is box
    w._fill_view_menu()
    assert "pySASred" in [a.text() for a in w.view_menu.actions()]
    w.close()
