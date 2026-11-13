"""pySASscout window and its launcher tab (offscreen, Python reference functions)."""
import json
import os
import time

import numpy as np
import pytest

pytest.importorskip("PySide6")
pytest.importorskip("matplotlib")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6 import QtWidgets  # noqa: E402

from pysasfit.scout import reference  # noqa: E402


@pytest.fixture
def app():
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


@pytest.fixture
def curve(tmp_path):
    from pysasfit.studio.model import ModelFunction, ScatteringContribution

    reference.register()
    c = ScatteringContribution("ff_sphere", {"R": 60, "eta": 1}, size_distribution="sd_LogNorm",
                               sd_params={"N": 1e-12, "s": 0.2, "p": 1, "mu": 60}, distributed="R")  # fmt: skip
    q = np.geomspace(0.005, 0.4, 60)
    I = ModelFunction([c], background=0.002)(q)
    s = 0.03 * I + 0.0015 * I[-1]
    fn = tmp_path / "spheres.dat"
    np.savetxt(fn, np.column_stack([q, I + np.random.default_rng(0).normal(0, s), s]))
    yield str(fn)
    reference.unregister()


@pytest.fixture
def window(app, tmp_path, monkeypatch):
    from pysasfit.scout.gui import MainWindow

    from pysasfit.core import settings as prefs

    monkeypatch.setattr(MainWindow, "_session_path", staticmethod(lambda: str(tmp_path / "scout_session.json")))
    prefs.set_value("scout.store", str(tmp_path / "store"))          # every test has its own, empty results store
    w = MainWindow()
    yield w
    w.close()
    prefs.set_value("scout.store", "")


def _wait(app, w, seconds=120):
    t0 = time.time()
    while w._busy and time.time() - t0 < seconds:
        app.processEvents()
        time.sleep(0.02)
    app.processEvents()
    assert not w._busy


def _fast(w):
    for k, b in w.form_boxes.items():
        b.setChecked(k in ("sphere", "rod"))
    w.starts.setValue(3)
    w.max_starts.setValue(8)
    w.reference.setChecked(True)


def test_nothing_is_calculated_before_a_button_is_pressed(window, curve):
    w = window
    assert not w.search_btn.isEnabled() and not w.stop_btn.isEnabled()
    assert w.load_file(curve)
    assert w.search_btn.isEnabled() and w.result is None and w.features is None and w.table.rowCount() == 0
    assert "60 points" in w.file_label.text()
    w.run_features()
    assert w.features.guinier_ok and "Guinier" in w.features_view.toPlainText() and w.result is None


def test_search_fills_ranking_details_and_plot(app, window, curve, tmp_path):
    w = window
    w.load_file(curve)
    _fast(w)
    w.run_search()
    assert w._busy and w.stop_btn.isEnabled() and not w.search_btn.isEnabled()
    _wait(app, w)
    assert w.result.best.label == "sphere + LogNorm" and w.table.rowCount() == 3
    assert w.table.item(0, 1).text() == "sphere + LogNorm" and w.table.currentRow() == 0
    assert "0.sd.mu" in w.details.toPlainText() and "ranking" not in w.log_view.toPlainText()
    assert "candidates 3/3" in w.log_view.toPlainText()
    assert len(w.ax.lines) >= 1 and len(w.ax_res.lines) >= 1
    w.table.selectRow(2)                                    # another candidate: its curve and details
    assert w.selected().label == "thin rod" and "thin rod" in w.details.toPlainText()
    out = tmp_path / "p.json"
    w.result.save(str(out))
    assert json.load(open(out))["candidates"][0]["label"] == "sphere + LogNorm"


def test_stop_ranks_what_was_fitted(app, window, curve):
    w = window
    w.load_file(curve)
    _fast(w)
    w.run_search()
    w.stop()
    _wait(app, w)
    assert w.result is not None and any("cancelled" in n for n in w.result.notes)
    assert any(c.message == "skipped: cancelled" for c in w.result.candidates)


def test_results_store_is_shown_and_reused(app, window, curve, tmp_path):
    from pysasfit.core import settings as prefs

    w = window
    root = str(tmp_path / "store")
    assert w.store_dir() == os.path.abspath(root) and w.store_label.text() == os.path.abspath(root)
    assert not os.path.exists(root)                              # nothing is created before a search
    w.load_file(curve)
    assert not w.stored_btn.isEnabled() and not w.load_stored()
    _fast(w)
    w.run_search()
    _wait(app, w)
    assert w.result.store["folder"] == os.path.abspath(root) and w.table.item(0, 7).text() == "fit"
    assert "1 curve, 3 fits, 1 search" in w.store_summary.text() and w.stored_btn.isEnabled()
    w.run_search()                                               # again: everything comes from the store
    _wait(app, w)
    assert all(c.stored for c in w.result.candidates) and w.table.item(0, 7).text() == "store"
    assert "from the store" in w.statusBar().currentMessage()
    w.reuse.setChecked(False)                                    # ... unless reuse is switched off
    w.run_search()
    _wait(app, w)
    assert not any(c.stored for c in w.result.candidates)
    box = w.show_store_info(show=False)
    assert os.path.abspath(root) in box.informativeText() and "spheres.dat" in box.detailedText()
    assert any(b.text() == "Open folder" for b in box.buttons())
    # a new window: the stored search is offered and loaded without fitting
    from pysasfit.scout.gui import MainWindow

    w2 = MainWindow()
    w2.load_file(curve)
    assert w2.result is None and w2.stored_btn.isEnabled() and "stored search" in w2.statusBar().currentMessage()
    assert w2.load_stored() and w2.result.best.label == "sphere + LogNorm" and w2.table.rowCount() == 3
    assert "0.sd.mu" in w2.details.toPlainText() and len(w2.ax.lines) >= 1 and not w2._busy
    w2.q_max.setText("0.2")                                      # another q range is another curve
    w2.q_max.editingFinished.emit()
    assert not w2.stored_btn.isEnabled()
    # the folder follows the preferences
    prefs.set_value("scout.store", str(tmp_path / "elsewhere"))
    w2.refresh_store()
    assert w2.store_label.text() == os.path.abspath(str(tmp_path / "elsewhere"))
    w2.close()


def test_preferences_have_a_scout_page(app):
    from pysasfit.core import settings as prefs
    from pysasfit.gui.launcher import _TOOL_KEYS

    assert _TOOL_KEYS["pySASscout"] == "scout" and ("scout", "pySASscout") in prefs.tools()
    assert prefs.find("scout.store").effective_placeholder().endswith("scout_store")
    assert os.path.dirname(prefs.find("scout.store").effective_placeholder()) == os.path.dirname(prefs.settings_path())


def test_q_range_and_python_code(app, window, curve, tmp_path):
    try:
        from tests.pycode_util import run_script
    except ImportError:  # tests/ is not a package
        import sys
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
        from pycode_util import run_script

    w = window
    with pytest.raises(ValueError):
        w.python_code()
    w.load_file(curve)
    _fast(w)
    w.q_max.setText("0.2")
    q = w._selection()[0]
    assert q.max() <= 0.2 and q.size < 60
    code = w.python_code()
    assert "scout.search" in code and "q_range" in code and "reference=True" in code and "'rod'" in code
    ns = run_script(code, tmp_path)
    assert ns["result"].best.label == "sphere + LogNorm" and ns["result"].data["q_max"] <= 0.2


def test_session_restores_inputs_only(app, window, curve):
    from pysasfit.scout.gui import MainWindow

    w = window
    w.load_file(curve)
    _fast(w)
    w.q_min.setText("0.006")
    w.components.setValue(2)
    w.criterion.setCurrentIndex(1)
    w.save_session()
    w2 = MainWindow()
    assert w2.restore_session()
    assert w2.path == w.path and w2.q_min.text() == "0.006" and w2.components.value() == 2
    assert w2.criterion.currentData() == "bic" and w2.starts.value() == 3
    assert [k for k, b in w2.form_boxes.items() if b.isChecked()] == ["sphere", "rod"]
    assert w2.result is None and w2.features is None            # nothing calculated
    w2.forget_session()
    assert not MainWindow().restore_session()
    w2.close()


def test_launcher_has_the_scout_tab(app):
    from pysasfit.gui.launcher import TOOLS, Launcher
    from pysasfit.resources import tool_icon, wordmark

    assert ("pySASscout", "pysasfit.scout.gui", "MainWindow") in TOOLS
    assert not tool_icon("pySASscout").isNull() and wordmark("pySASscout")[1][0] == "SAS"
    w = Launcher(tools=[t for t in TOOLS if t[0] == "pySASscout"])
    assert w.tabs.tabText(0) == "pySASscout"
    win = w.tool("pySASscout")
    assert win is not None and w.tabs.tabText(0) == "pySASscout"          # no "(error)"
    assert win.copy_python_action.shortcut().toString() == "Ctrl+Shift+C"
    fmenu = w.menuBar().actions()[0].menu()
    fmenu.aboutToShow.emit()
    assert win.copy_python_action in fmenu.actions()
    assert any("protocol" in a.text() for a in fmenu.actions())
    w.close()
