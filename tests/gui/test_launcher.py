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
    w = Launcher()
    titles = [w.tabs.tabText(i) for i in range(w.tabs.count())]
    assert titles == ["SLD profile", "Size distribution / p(r)", "Ornstein-Zernike"], titles  # no "(error)" tabs
    oz = w.window_for("Ornstein-Zernike")
    assert oz.count() >= 4 and not any("(error)" in oz.tabText(i) for i in range(oz.count()))
    w.window_for("SLD profile").load_file(DATA)
    w.tabs.setCurrentIndex(1)
    app.processEvents()
    w.close()


def test_broken_tool_gets_error_tab():
    QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    w = Launcher(tools=[("broken", "pysasfit.does_not_exist", "MainWindow")])
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
    w = Launcher()
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
