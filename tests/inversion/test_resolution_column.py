"""Optional 4th data column sigma_Q: loaded, and used to smear the kernel."""
import os

import numpy as np
import pytest

from pysasfit.inversion.io_utils import load_sas_data

DATA = os.path.join(os.path.dirname(__file__), "data", "test.dat")


def _four_columns(tmp_path, rel=0.08):
    d = load_sas_data(DATA)
    path = tmp_path / "test4.dat"
    np.savetxt(path, np.c_[d.q, d.I, d.dI, rel * d.q], header="q I dI sigma_Q")
    return str(path), d


def test_loader_reads_optional_sigma_q(tmp_path):
    path, d3 = _four_columns(tmp_path)
    d4 = load_sas_data(path)
    assert d4.q.size == d3.q.size and d3.dQ is None
    np.testing.assert_allclose(d4.dQ, 0.08 * d4.q)
    np.testing.assert_allclose(d4.dI, d3.dI)


def test_gui_smears_kernel_with_sigma_q(tmp_path):
    pytest.importorskip("PySide6")
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication
    from pysasfit.inversion.gui.app import MainWindow

    app = QApplication.instance() or QApplication([])
    path, _ = _four_columns(tmp_path, rel=0.15)
    w = MainWindow()
    w.load_file(path)
    assert w.res_check.isEnabled() and w.res_check.isChecked()
    w.solver_combo.setCurrentIndex(w.solver_combo.findData("bayesian_evidence"))
    w._on_solve()
    smeared = w._last_result
    assert "smeared with σ_Q" in w._last_result_info["representation"]
    w.res_check.setChecked(False)
    w._on_solve()
    plain = w._last_result
    assert "smeared" not in w._last_result_info["representation"]
    # the smeared kernel describes the same data with a sharper (less smoothed) p(r)
    assert not np.allclose(smeared.x, plain.x)
    assert np.isfinite(smeared.chi2_r_history[-1])
    w.close()
    w3 = MainWindow()
    w3.load_file(DATA)                       # 3 columns: option not available
    assert not w3.res_check.isEnabled() and not w3.res_check.isChecked()
    w3.close()
