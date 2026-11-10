"""pyINVersion "Copy Python code": the script repeats Solve for several solvers and settings."""
import os
import sys

import numpy as np
import pytest

pytest.importorskip("PySide6")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
from PySide6.QtWidgets import QApplication  # noqa: E402

from pysasfit.inversion.gui.app import MainWindow  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from pycode_util import run_plot, run_script  # noqa: E402

DATA = os.path.join(os.path.dirname(__file__), "data", "test.dat")


@pytest.fixture
def win():
    app = QApplication.instance() or QApplication([])
    w = MainWindow()
    w.load_file(DATA)
    yield w
    w.close()


def _check(w):
    w._on_solve()
    ns = run_script(w.python_code())
    assert np.allclose(ns["result"].x, w._last_result.x, rtol=1e-7, atol=1e-10 * np.abs(w._last_result.x).max())
    assert np.allclose(ns["r_grid"], w._last_r_grid)
    return ns


def test_default_solver(win):
    _check(win)
    run_plot(win.python_code())


def test_solvers(win, solver):
    win.solver_combo.setCurrentIndex(win.solver_combo.findData(solver))
    win.kernel_combo.setCurrentIndex(win.kernel_combo.findData("sphere_rg"))
    win.r_max_spin.setValue(300.0)
    win.r_n_spin.setValue(40)
    _check(win)


def pytest_generate_tests(metafunc):
    if "solver" in metafunc.fixturenames:
        metafunc.parametrize("solver", ["bayesian_evidence", "general_tikhonov_gcv", "tsvd_gcv", "maxent_em", "arlsnn"])


def test_fitted_background_spline_and_fixed_bg(win):
    win._on_fit_background()
    assert win.bg_result is not None
    win.spline_check.setChecked(True)
    code = win.python_code()
    assert "fit_background(" in code and "SplineOptions(" in code
    _check(win)
    win.spline_check.setChecked(False)
    win.use_fixed_bg.setChecked(True)
    win.fixed_bg_spin.setValue(0.1)
    win.error_model_combo.setCurrentIndex(win.error_model_combo.findData("relative"))
    _check(win)


def test_monte_carlo_block(win):
    win.solver_combo.setCurrentIndex(win.solver_combo.findData("general_tikhonov_gcv"))
    win.mc_uncertainty_check.setChecked(True)
    win.n_boot_spin.setValue(5)
    ns = run_script(win.python_code())
    assert ns["boot"].n_boot == win.n_boot_spin.value() and ns["boot"].x_mean.shape == ns["result"].x.shape
