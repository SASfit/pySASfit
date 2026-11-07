"""The Monte-Carlo estimate runs in a background thread: the solve returns at once, the result arrives later;
a new solve stops a running estimate."""
import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
import time

from PySide6.QtWidgets import QApplication

from pysasfit.inversion.gui.app import MainWindow

DATA = os.path.join(os.path.dirname(__file__), "data", "test.dat")


def _wait(app, cond, t=120):
    t0 = time.time()
    while not cond() and time.time() - t0 < t:
        app.processEvents()
        time.sleep(0.02)


def test_bootstrap_in_background():
    app = QApplication.instance() or QApplication([])
    win = MainWindow()
    win.load_file(DATA)
    win.use_fixed_bg.setChecked(True)
    win.fixed_bg_spin.setValue(0.1)
    i = win.solver_combo.findData("bayesian_evidence")
    win.solver_combo.setCurrentIndex(i)
    win.mc_uncertainty_check.setChecked(True)
    win.n_boot_spin.setValue(12)
    win._on_solve()
    assert win._last_result is not None
    _wait(app, lambda: win._last_boot is not None)
    assert win._last_boot is not None and win._last_boot.n_boot + win._last_boot.n_failed == 12
    # a second solve supersedes the first estimate; its own estimate arrives
    win.n_boot_spin.setValue(10)
    win._on_solve()
    first = win._last_result
    _wait(app, lambda: win._last_boot is not None)
    assert win._last_result is first and win._last_boot.n_boot + win._last_boot.n_failed == 10
    win.shutdown()
    win.close()
