"""The Fit panel (PySide6, offscreen): fitting in a thread, priors, export and clipboard buttons, speed-up boxes.

The panel is tested with a stand-in explorer (a small model, no SASfit library), so these tests also run where the
SASfit libraries are not installed. They are skipped when PySide6 is missing.
"""

import os
import time

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")

from PySide6 import QtWidgets  # noqa: E402

from pysasfit.studio import export, fit_panel  # noqa: E402
from pysasfit.studio.model import ModelFunction  # noqa: E402

try:
    from .support import Q, Toy, installed, make_data  # noqa: E402
except ImportError:
    from support import Q, Toy, installed, make_data  # noqa: E402


@pytest.fixture(scope="module")
def app():
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


class StubExplorer:
    """What the Fit panel needs from the explorer (see the docstring of fit_panel)."""

    def __init__(self):
        self.model = ModelFunction([Toy(2.0, 1.0, 0.0)], scales=[1.0], background=0.0, bg_A=0.0, bg_alpha=4.0)
        self.data = None
        self.integration = {"n_x": 96}
        self.ticked = ["0.ff.A", "0.ff.k", "background"]
        self.busy_calls = []
        self.restored = 0

    def has_model(self):
        return True

    def build_model(self):
        return self.model

    def parameter_entries(self):
        return [("background", "background", "-1", "2"), ("bg.A", "bg.A", "0", "5"), ("bg.alpha", "bg.alpha", "0", "8"),
                ("0.ff.A", "ff.A", "0", "20"), ("0.ff.k", "ff.k", "0", "5")]  # fmt: skip

    def fit_selection(self):
        entries = {a: (a, lab, lo, hi) for a, lab, lo, hi in self.parameter_entries()}
        return [entries[a] for a in self.ticked]

    def set_data(self, q, I, sigma, name="data"):
        self.data = {"q": np.asarray(q), "I": np.asarray(I), "sigma": np.asarray(sigma), "name": name}

    def clear_data(self):
        self.data = None

    def snapshot(self):
        m = self.model
        return {"v": m.contributions[0].vector("ff").copy(), "bg": m.background, "A": m.bg_A, "alpha": m.bg_alpha}

    def restore(self, snap):
        m = self.model
        m.contributions[0].vector("ff")[:] = snap["v"]
        m.background, m.bg_A, m.bg_alpha = snap["bg"], snap["A"], snap["alpha"]
        self.restored += 1

    def apply_model_values(self, model):
        pass

    def set_busy(self, busy):
        self.busy_calls.append(busy)

    def panels(self):
        return []

    def _schedule(self):
        pass

    def _q(self):
        return Q


@pytest.fixture()
def panel(app):
    ex = StubExplorer()
    ex.set_data(*make_data(), name="sample.dat")
    p = fit_panel.FitPanel(ex)
    p.refresh_table()
    yield p
    p.deleteLater()


def run_fit(p, timeout=60):
    p._fit_or_stop()
    t0 = time.time()
    while p._thread is not None and time.time() - t0 < timeout:
        QtWidgets.QApplication.processEvents()
        time.sleep(0.01)
    QtWidgets.QApplication.processEvents()
    assert p._thread is None, "the fit did not finish"


def choose(p, key):
    """Select an optimiser by its key (e.g. 'scipy-global') in the panel."""
    i = p.backend.findData(key)
    assert i >= 0, key
    p.backend.setCurrentIndex(i)


# ------------------------------------------------------------------ construction
def test_buttons_before_any_fit(app):
    ex = StubExplorer()
    p = fit_panel.FitPanel(ex)
    assert not p.export_btn.isEnabled() and not p.copy_btn.isEnabled()
    assert not p.fit_btn.isEnabled()  # no data
    assert p.tabs.indexOf(p.result_page) >= 0 and p.tabs.tabText(p.tabs.indexOf(p.result_page)) == "Result"


def test_fit_button_is_enabled_with_data(panel):
    assert panel.fit_btn.isEnabled() and panel.fit_btn.text() == "Fit"


def test_parameter_table_lists_the_ticked_parameters(panel):
    text = panel.summary.toPlainText()
    assert "ff.A" in text and "ff.k" in text and "background" in text and "bg.A" not in text


def test_priors_tab_explains_all_prior_kinds(panel):
    t = panel.priors.placeholderText()
    for word in ("Gaussian", "lognormal", "halfnormal", "loguniform"):
        assert word in t


def test_backends_include_multistart_in_scipy_global(panel):
    choose(panel, "scipy-global")
    names = [panel.algorithm.itemData(i) for i in range(panel.algorithm.count())]
    assert "multistart" in names and "differential_evolution" in names


def test_polish_box_is_hidden_for_scipy_only(panel):
    choose(panel, "scipy")
    assert not panel.polish.isVisibleTo(panel)
    choose(panel, "scipy-global")
    assert panel.polish.isVisibleTo(panel)


# ------------------------------------------------------------------ fitting
def test_fit_runs_in_a_thread_and_enables_the_export(panel):
    run_fit(panel)
    assert panel._summary is not None and panel._summary.success
    assert panel.export_btn.isEnabled() and panel.undo_btn.isEnabled()
    assert panel.tabs.currentWidget() is panel.result_page
    assert "chi" in panel.result.toPlainText().lower() and "converged" in panel.status.text()
    assert panel.explorer.busy_calls == [True, False]


def test_variable_projection_box_reaches_the_fit(panel):
    panel.vp.setChecked(True)
    run_fit(panel)
    assert "variable projection" in panel._summary.message


def test_fit_of_the_power_law_background_through_the_panel(app):
    ex = StubExplorer()
    q = Q
    I = 5.0 * np.exp(-0.8 * q) + 0.3 + 0.02 * q**-4.0
    s = 0.01 * (I + 0.05)
    ex.set_data(q, I + np.random.default_rng(1).normal(0, s), s, name="pl.dat")
    ex.model.bg_A = 0.05
    ex.ticked = ["0.ff.A", "0.ff.k", "background", "bg.A"]
    p = fit_panel.FitPanel(ex)
    p.refresh_table()
    p.vp.setChecked(True)
    run_fit(p)
    assert abs(p._summary.values["bg.A"] - 0.02) < 0.005


def test_multistart_through_the_panel(panel):
    choose(panel, "scipy-global")
    panel.algorithm.setCurrentIndex(panel.algorithm.findData("multistart"))
    run_fit(panel)
    assert "multistart" in panel._summary.message


def test_a_prior_line_is_used_by_the_fit(panel):
    panel.priors.setPlainText("# comment\nff.k 0.5 0.001\n\nbackground halfnormal 1")
    run_fit(panel)
    assert abs(panel._summary.values["ff.k"] - 0.5) < 0.05
    assert panel._summary.dof == Q.size + 2 - 3


def test_a_bad_prior_line_is_reported_without_starting_a_fit(panel):
    panel.priors.setPlainText("ff.k lognormal 1")
    panel._fit_or_stop()
    assert panel._thread is None
    assert "write" in panel.status.text() and "lognormal" in panel.status.text()
    assert panel.explorer.busy_calls == []


def test_an_unknown_parameter_in_a_prior_is_reported(panel):
    panel.priors.setPlainText("nothing 1 2")
    panel._fit_or_stop()
    assert panel._thread is None and "unknown parameter" in panel.status.text()


def test_loguniform_with_least_squares_shows_the_refusal_and_restores_the_values(panel):
    panel.explorer.ticked = ["0.ff.A", "0.ff.k", "background"]
    panel.explorer.parameter_entries = lambda: [("background", "background", "0.001", ""), ("0.ff.A", "ff.A", "0", "20"),
                                                ("0.ff.k", "ff.k", "0", "5")]  # fmt: skip
    panel.priors.setPlainText("background loguniform")
    before = panel.explorer.snapshot()
    run_fit(panel)
    assert "log-uniform" in panel.status.text() and panel._summary is None
    assert panel.explorer.restored >= 1
    assert np.allclose(panel.explorer.snapshot()["v"], before["v"])


def test_constraints_are_passed_to_the_problem(panel):
    run_fit(panel)
    free_k = panel._summary.values["ff.k"]
    panel.constraints.setPlainText("# comment\nff_k < 0.5")
    run_fit(panel)
    k = panel._summary.values["ff.k"]
    assert free_k > 0.6 and k < 0.52  # the penalty of the least-squares backend allows a slight excess


def test_no_parameter_ticked_is_reported(panel):
    panel.explorer.ticked = []
    panel._fit_or_stop()
    assert panel._thread is None and "tick the fit boxes" in panel.status.text()


def test_bounds_must_be_ordered_and_numeric(panel):
    panel.explorer.parameter_entries = lambda: [("background", "background", "5", "1"), ("0.ff.A", "ff.A", "0", "20"),
                                                ("0.ff.k", "ff.k", "0", "5")]  # fmt: skip
    panel._fit_or_stop()
    assert "lower must be smaller than upper" in panel.status.text()
    panel.explorer.parameter_entries = lambda: [("background", "background", "x", "1"), ("0.ff.A", "ff.A", "0", "20"),
                                                ("0.ff.k", "ff.k", "0", "5")]  # fmt: skip
    panel._fit_or_stop()
    assert "must be numbers" in panel.status.text()


def test_undo_restores_the_values(panel):
    before = panel.explorer.snapshot()["v"].copy()
    panel.explorer.model.contributions[0].vector("ff")[:] = [9.0, 9.0]  # as if a fit had moved them
    panel._before = {"v": before, "bg": 0.0, "A": 0.0, "alpha": 4.0}
    panel._undo()
    assert np.allclose(panel.explorer.model.contributions[0].vector("ff"), before)
    assert not panel.undo_btn.isEnabled()


def test_fit_q_range_limits_the_points(panel):
    panel.fit_qmin.setText("0.5")
    panel.fit_qmax.setText("2,5")  # a decimal comma is accepted
    assert panel.q_range() == (0.5, 2.5)
    run_fit(panel)
    n = int(np.sum((Q >= 0.5) & (Q <= 2.5)))
    assert panel._summary.dof == n - 3


# ------------------------------------------------------------------ speed-up boxes
def test_fixed_grid_ticks_and_locks_the_cache_box(panel):
    panel.cache.setChecked(False)
    panel.grid.setChecked(True)
    assert panel.cache.isChecked() and not panel.cache.isEnabled()
    assert panel.cache_requested() is False  # what the user set, for the project file
    panel.grid.setChecked(False)
    assert panel.cache.isEnabled() and not panel.cache.isChecked()


def test_the_cache_box_choice_survives_the_grid(panel):
    panel.cache.setChecked(True)
    panel.grid.setChecked(True)
    panel.grid.setChecked(False)
    assert panel.cache.isChecked() and panel.cache_requested() is True


def test_coarse_with_grid_adds_a_note(panel):
    panel.coarse.setChecked(True)
    panel.grid.setChecked(True)
    assert "only helps" in panel.coarse.toolTip() and "fixed grid" in panel.status.text()
    panel.grid.setChecked(False)
    assert "only helps" not in panel.coarse.toolTip()


def test_profile_model_button_writes_the_result_tab(panel):
    panel._profile()
    assert "one model evaluation" in panel.result.toPlainText()
    assert panel.tabs.currentWidget() is panel.result_page


# ------------------------------------------------------------------ export and clipboard
def fake_dialog(monkeypatch, path, chosen=""):
    monkeypatch.setattr(QtWidgets.QFileDialog, "getSaveFileName", staticmethod(lambda *a, **k: (str(path), chosen)))


@pytest.mark.skipif(not installed("openpyxl"), reason="openpyxl is not installed")
def test_export_to_xlsx(panel, tmp_path, monkeypatch):
    run_fit(panel)
    fake_dialog(monkeypatch, tmp_path / "out.xlsx")
    panel._export()
    assert (tmp_path / "out.xlsx").stat().st_size > 1000 and "saved" in panel.status.text()


def test_export_to_csv_writes_the_files(panel, tmp_path, monkeypatch):
    run_fit(panel)
    fake_dialog(monkeypatch, tmp_path / "out.csv")
    panel._export()
    for name in ("parameters", "correlation", "covariance", "curve"):
        assert (tmp_path / f"out_{name}.csv").exists()
    assert "out_parameters.csv" in panel.status.text()


@pytest.mark.skipif(not installed("matplotlib"), reason="matplotlib is not installed")
def test_export_to_pdf_png_and_svg(panel, tmp_path, monkeypatch):
    run_fit(panel)
    for suffix, magic in ((".pdf", b"%PDF"), (".png", b"\x89PNG"), (".svg", b"<?xml")):
        fake_dialog(monkeypatch, tmp_path / f"out{suffix}")
        panel._export()
        assert (tmp_path / f"out{suffix}").read_bytes()[:5].startswith(magic), suffix


def test_export_adds_the_suffix_of_the_chosen_filter(panel, tmp_path, monkeypatch):
    run_fit(panel)
    fake_dialog(monkeypatch, tmp_path / "noext", chosen="CSV files (*.csv)")
    panel._export()
    assert (tmp_path / "noext.csv_parameters.csv").exists() or (tmp_path / "noext_parameters.csv").exists()


def test_a_cancelled_dialog_writes_nothing(panel, tmp_path, monkeypatch):
    run_fit(panel)
    fake_dialog(monkeypatch, "")
    panel._export()
    assert list(tmp_path.iterdir()) == []


def test_export_errors_are_shown_in_the_status_line(panel, tmp_path, monkeypatch):
    run_fit(panel)
    fake_dialog(monkeypatch, tmp_path / "missing_folder" / "out.csv")
    panel._export()
    assert "Error" in panel.status.text() or "No such" in panel.status.text() or "FileNotFound" in panel.status.text()


def test_export_without_a_fit_does_nothing(panel, tmp_path, monkeypatch):
    fake_dialog(monkeypatch, tmp_path / "out.csv")
    panel._export()
    assert list(tmp_path.iterdir()) == []


@pytest.mark.skipif(not installed("matplotlib"), reason="matplotlib is not installed")
def test_copy_summary_image_puts_a_picture_on_the_clipboard(panel, app):
    run_fit(panel)
    QtWidgets.QApplication.clipboard().clear()
    panel._copy_image()
    img = QtWidgets.QApplication.clipboard().image()
    assert "clipboard" in panel.status.text()
    if img.isNull():
        pytest.skip("this Qt platform plugin has no image clipboard")
    assert img.width() > 500 and img.height() > 300


def test_copy_button_needs_matplotlib_and_a_fit(panel):
    assert not panel.copy_btn.isEnabled()
    run_fit(panel)
    assert panel.copy_btn.isEnabled() == export.have_matplotlib()


# ------------------------------------------------------------------ size distribution weight p from the explorer
def _with_distribution(ex):
    import numpy as _np

    class _Dist:
        name = "sd_Fake"
        parameter_names = ["N", "mu", "s"]

        def evaluate_p(self, x, p):
            x = _np.asarray(x, float)
            return p[0] * _np.exp(-0.5 * (_np.log(x / p[1]) / p[2]) ** 2) / x

    c = ex.model.contributions[0]
    c.sd, c.sd_p, c.support = _Dist(), _np.array([2.0, 50.0, 0.3]), (lambda: (20.0, 120.0))


def test_the_weight_p_defaults_to_3_without_a_distribution_dock(panel):
    assert panel._dist_power() == 3.0


def test_the_weight_p_is_read_from_the_explorer(panel):
    panel.explorer.dist_xpow = QtWidgets.QDoubleSpinBox()
    panel.explorer.dist_xpow.setRange(-6, 12)
    panel.explorer.dist_xpow.setValue(6.0)
    assert panel._dist_power() == 6.0


def test_csv_export_documents_the_weight_from_the_explorer(panel, tmp_path, monkeypatch):
    _with_distribution(panel.explorer)
    panel.explorer.dist_xpow = QtWidgets.QDoubleSpinBox()
    panel.explorer.dist_xpow.setRange(-6, 12)
    panel.explorer.dist_xpow.setValue(6.0)
    run_fit(panel)
    fake_dialog(monkeypatch, tmp_path / "out.csv")
    panel._export()
    header = open(tmp_path / "out_distributions.csv", encoding="utf-8").readline().strip()
    assert header == "distribution,x,N(x),N(x) x^6"


@pytest.mark.skipif(not installed("matplotlib"), reason="matplotlib is not installed")
def test_the_copied_picture_contains_the_distribution_panels(panel, app, monkeypatch):
    from pysasfit.studio import export

    seen = {}
    real = export.figure_bytes
    monkeypatch.setattr(export, "figure_bytes", lambda *a, **k: seen.setdefault("p", k.get("dist_power")) and real(*a, **k))
    _with_distribution(panel.explorer)
    panel.explorer.dist_xpow = QtWidgets.QDoubleSpinBox()
    panel.explorer.dist_xpow.setValue(3.0)
    run_fit(panel)
    panel._copy_image()
    assert seen["p"] == 3.0
