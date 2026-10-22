"""The model explorer window (PySide6, offscreen) with the real SASfit library.

NOT RUN by the author of these tests: they need libsasfit and its plugins, which the test machine did not have.
They are skipped automatically when the explorer cannot be created (no PySide6, no SASfit library, no ff_sphere).
If one of them fails on your machine, send the output of  pytest tests/studio/test_explorer_gui.py -x -q -rs
"""

import json
import os

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")

from PySide6 import QtCore, QtWidgets  # noqa: E402


@pytest.fixture(scope="module")
def app():
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


@pytest.fixture()
def ex(app):
    try:
        from pysasfit.studio.explorer import ModelExplorer

        w = ModelExplorer("ff_sphere", sd="sd_LogNorm")
    except Exception as exc:  # noqa: BLE001 -- no library, no functions, no display
        pytest.skip(f"explorer cannot be created here: {type(exc).__name__}: {exc}")
    # the defaults of sd_LogNorm give a zero distribution: use a sensible one (N, s, p, mu)
    c = w.models()[0].contrib
    names = list(c.sd.parameter_names)
    for name, value in (("N", 1.0), ("s", 0.2), ("p", 1.0), ("mu", 50.0)):
        if name in names:
            c.sd_p[names.index(name)] = value
    ff = w.models()[0].contrib.vector("ff")
    ffnames = list(w.models()[0].contrib.ff.parameter_names)
    if "eta" in ffnames:
        ff[ffnames.index("eta")] = 1.0  # contrast: with 0 the intensity is 0
    w.logy.setChecked(False)  # _redraw stops before the other plots if no intensity is positive on a log axis
    ff_names = list(c.ff.parameter_names)
    if "eta" in ff_names:  # contrast: the default may be 0, which gives I(q) = 0
        c.ff_p[ff_names.index("eta")] = 1.0
    w.logy.setChecked(False)  # the distribution plots must not depend on the sign of I(q)
    ff = w.models()[0].contrib.vector("ff")
    ffnames = list(w.models()[0].contrib.ff.parameter_names)
    if "eta" in ffnames:
        ff[ffnames.index("eta")] = 1.0  # contrast: with 0 the intensity is 0
    w.logy.setChecked(False)  # _redraw stops before the other plots if no intensity is positive on a log axis
    w.after_fit()  # the rows follow the vectors
    w.show()
    yield w
    for dock in list(getattr(w, "_detached", [])):
        dock.close()
        dock.deleteLater()
    w.close()
    w.deleteLater()
    QtWidgets.QApplication.processEvents()


def settle(ex):
    ex.dist_dock.show()  # the distribution plot is only drawn while its dock is visible
    QtWidgets.QApplication.processEvents()
    ex._redraw()
    QtWidgets.QApplication.processEvents()


def why(ex):
    return f"dock visible: {ex.dist_dock.isVisible()}; status: {ex.status.text()}"


# ------------------------------------------------------------------ background rows
def test_the_background_has_a_constant_and_a_power_law_row(ex):
    assert ex.bg_row.value == 0.0 and ex.bgA_row.value == 0.0 and ex.bgalpha_row.value == 4.0


def test_parameter_entries_list_the_background_parameters_first(ex):
    entries = ex.parameter_entries()
    assert [e[0] for e in entries[:3]] == ["background", "bg.A", "bg.alpha"]
    assert entries[1][2] == "0" and entries[2][3] == "8"  # default bounds


def test_value_of_and_row_resolve_the_new_addresses(ex):
    ex.bgA_row.set_value(0.3, emit=False)
    ex.bgalpha_row.set_value(2.5, emit=False)
    assert ex.value_of("bg.A") == 0.3 and ex.value_of("bg.alpha") == 2.5
    assert ex._row("bg.A") is ex.bgA_row and ex._row("bg.alpha") is ex.bgalpha_row


def test_build_model_carries_the_power_law(ex):
    ex.bg_row.set_value(0.1, emit=False)
    ex.bgA_row.set_value(0.3, emit=False)
    ex.bgalpha_row.set_value(3.0, emit=False)
    m = ex.build_model()
    assert m.background == 0.1 and m.bg_A == 0.3 and m.bg_alpha == 3.0


def test_the_curve_shown_contains_the_power_law(ex):
    q = ex._q()
    base, _ = ex._compute(q)
    ex.bgA_row.set_value(2.0, emit=False)
    ex.bgalpha_row.set_value(3.0, emit=False)
    with_bg, _ = ex._compute(q)
    assert np.allclose(with_bg - base, 2.0 * q**-3.0)
    assert np.allclose(ex._power_term(q), 2.0 * q**-3.0)


def test_the_background_line_is_drawn_only_when_there_is_a_background(ex):
    settle(ex)
    assert not ex.bg_line.get_visible()
    ex.bg_row.set_value(0.05, emit=False)
    settle(ex)
    assert ex.bg_line.get_visible()
    x, y = ex.bg_line.get_data()
    assert np.allclose(y, 0.05)
    ex.bg_row.set_value(0.0, emit=False)
    ex.bgA_row.set_value(1.0, emit=False)
    settle(ex)
    assert ex.bg_line.get_visible() and np.allclose(ex.bg_line.get_data()[1], ex._q() ** -4.0)


def test_snapshot_and_restore_include_the_power_law(ex):
    ex.bgA_row.set_value(0.5, emit=False)
    ex.bgalpha_row.set_value(2.0, emit=False)
    snap = ex.snapshot()
    ex.bgA_row.set_value(9.0, emit=False)
    ex.bgalpha_row.set_value(1.0, emit=False)
    ex.restore(snap)
    assert ex.bgA_row.value == 0.5 and ex.bgalpha_row.value == 2.0


def test_apply_model_values_takes_the_fitted_power_law(ex):
    m = ex.build_model()
    m.bg_A, m.bg_alpha, m.background = 0.7, 3.5, 0.2
    ex.apply_model_values(m)
    assert ex.bgA_row.value == 0.7 and ex.bgalpha_row.value == 3.5 and ex.bg_row.value == 0.2


def test_reset_parameters_resets_the_power_law(ex):
    ex.bgA_row.set_value(0.5, emit=False)
    ex.bgalpha_row.set_value(2.0, emit=False)
    ex._reset()
    assert ex.bgA_row.value == 0.0 and ex.bgalpha_row.value == 4.0


def test_copy_python_mentions_the_power_law_only_when_it_is_used(ex):
    assert "bg_A" not in ex._python_snippet()
    ex.bgA_row.set_value(0.5, emit=False)
    assert "bg_A=0.5" in ex._python_snippet() and "bg_alpha=4.0" in ex._python_snippet()


def test_fit_boxes_of_the_power_law_rows_reach_the_fit_panel(ex):
    ex.bgA_row.set_fit(True, emit=True)
    labels = [s[1] for s in ex.fit_selection()]
    assert "bg.A" in labels


# ------------------------------------------------------------------ projects
def test_project_round_trip_keeps_the_power_law(ex, tmp_path):
    from pysasfit.studio import project

    ex.bgA_row.set_value(0.25, emit=False)
    ex.bgalpha_row.set_value(3.0, emit=False)
    ex.bgA_row.set_fit(True, emit=False)
    path = project.save(ex, tmp_path / "p.sasstudio")
    d = json.load(open(path, encoding="utf-8"))
    assert d["bg_A"]["value"] == 0.25 and d["bg_alpha"]["value"] == 3.0 and d["bg_A"]["fit"] is True
    ex.bgA_row.set_value(0.0, emit=False)
    ex.bgalpha_row.set_value(4.0, emit=False)
    ex.bgA_row.set_fit(False, emit=False)
    project.load(ex, path)
    assert ex.bgA_row.value == 0.25 and ex.bgalpha_row.value == 3.0 and ex.bgA_row.is_fit()


def test_an_old_project_without_the_power_law_still_loads(ex, tmp_path):
    from pysasfit.studio import project

    path = project.save(ex, tmp_path / "old.sasstudio")
    d = json.load(open(path, encoding="utf-8"))
    d.pop("bg_A")
    d.pop("bg_alpha")
    json.dump(d, open(path, "w", encoding="utf-8"))
    project.load(ex, path)  # must not raise
    assert ex.bgalpha_row.value == 4.0


# ------------------------------------------------------------------ docks
def test_the_three_docks_exist(ex):
    assert ex.model_dock.windowTitle() == "Model" and ex.fit_dock.windowTitle() == "Fit"
    assert ex.dist_dock.windowTitle() == "Size distributions"
    assert ex.model_dock.widget() is ex._left


def test_the_view_menu_has_a_detach_entry_per_dock(ex):
    view = [m for m in ex.menuBar().findChildren(QtWidgets.QMenu) if m.title().replace("&", "") == "View"][0]
    texts = [a.text().strip() for a in view.actions()]
    for title in ("Model", "Fit", "Size distributions"):
        assert f"detach '{title}' as a separate window" in texts


@pytest.mark.parametrize("name", ["model_dock", "fit_dock", "dist_dock"])
def test_detached_dock_is_a_window_without_an_owner_and_can_be_docked_again(ex, name):
    dock = getattr(ex, name)
    area = ex.dockWidgetArea(dock)
    ex._set_detached(dock, area, True)
    QtWidgets.QApplication.processEvents()
    assert dock.parent() is None and dock.isWindow() and dock in ex._detached
    ex._set_detached(dock, area, False)
    QtWidgets.QApplication.processEvents()
    assert dock.parent() is ex and not dock.isWindow() and dock not in ex._detached
    assert ex.dockWidgetArea(dock) == area


def test_closing_the_main_window_closes_detached_docks(ex):
    dock = ex.fit_dock
    ex._set_detached(dock, QtCore.Qt.DockWidgetArea.BottomDockWidgetArea, True)
    assert dock.isVisible()
    ex.close()
    assert not dock.isVisible()


def test_a_detached_fit_dock_still_works(ex):
    ex._set_detached(ex.fit_dock, QtCore.Qt.DockWidgetArea.BottomDockWidgetArea, True)
    assert ex.fit_panel.isVisible() or ex.fit_dock.isVisible()
    ex.fit_panel.refresh_table()  # the panel is alive and still talks to the explorer


# ------------------------------------------------------------------ size distributions
def test_the_distribution_dock_defaults(ex):
    assert ex.dist_xpow.value() == 3.0 and not ex.dist_logx.isChecked()


def test_the_weight_buttons_set_the_exponent(ex):
    buttons = {b.text(): b for b in ex.dist_dock.widget().findChildren(QtWidgets.QPushButton)}
    for text, value in (("p=0", 0.0), ("p=3", 3.0), ("p=6", 6.0)):
        buttons[text].click()
        assert ex.dist_xpow.value() == value


def test_the_distribution_is_plotted_with_the_chosen_weight(ex):
    settle(ex)
    ex.dist_xpow.setValue(0.0)
    settle(ex)
    lines = ex.dist_ax.get_lines()
    assert len(lines) == 1, why(ex)
    x0, y0 = lines[0].get_data()
    ex.dist_xpow.setValue(6.0)
    settle(ex)
    x6, y6 = ex.dist_ax.get_lines()[0].get_data()
    assert np.allclose(x0, x6) and np.allclose(y6, y0 * x0**6, rtol=1e-9)
    assert "x^6" in ex.dist_ax.get_ylabel()


def test_the_distribution_axis_can_be_logarithmic(ex):
    ex.dist_logx.setChecked(True)
    settle(ex)
    assert ex.dist_ax.get_xscale() == "log", why(ex)


def test_the_plot_range_function_gives_a_usable_range(ex):
    c = ex.models()[0].contrib
    lo, hi = ex._sd_range(c)
    assert 0 < lo < hi and np.isfinite([lo, hi]).all()
