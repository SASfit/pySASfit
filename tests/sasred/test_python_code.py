"""pySASred "Copy Python code": the script repeats "Reduce selected"."""
import os
import sys

import numpy as np
import pytest

pytest.importorskip("PySide6")
pytest.importorskip("h5py")
pytest.importorskip("yaml")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
from PySide6 import QtWidgets  # noqa: E402

from pysasfit.sasred.gui import MainWindow  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from pycode_util import run_plot, run_script  # noqa: E402

RAW = os.environ.get("PYSASRED_TEST_RAW", "/mnt/user-data/uploads/Sept2026/raw")
need = pytest.mark.skipif(not os.path.isdir(RAW), reason="Vifor raw data not available")


def test_needs_a_selection():
    QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    w = MainWindow()
    with pytest.raises(ValueError):
        w.python_code()
    w.close()


@need
def test_script_repeats_reduce_selected(tmp_path, monkeypatch):
    monkeypatch.setenv("PYSASFIT_SETTINGS", str(tmp_path / "settings.json"))
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    w = MainWindow()
    w.set_folder(RAW)
    w.select_runs([82130, 82131])
    w.set_units("A")
    w.qmin.setValue(0.002)
    w.qmax.setValue(0.08)
    w.nbins.setValue(40)
    w.sector.setChecked(True)
    w.on_reduce()
    app.processEvents()
    code = w.python_code()
    assert "unpack_mask(" in code and "reduce_run_1d(" in code
    ns = run_script(code)
    for run in (82130, 82131):
        r, g = ns["results"][run], w.results[run]
        np.testing.assert_allclose(r.I, g.I, rtol=1e-13)
        np.testing.assert_allclose(r.idev(), g.idev(), rtol=1e-13)
        if g.q_sigma is not None:
            np.testing.assert_allclose(r.q_sigma, g.q_sigma, rtol=1e-12)
    run_plot(code)
    w.close()


@need
def test_script_repeats_table_reduction(tmp_path, monkeypatch):
    monkeypatch.setenv("PYSASFIT_SETTINGS", str(tmp_path / "settings.json"))
    QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    w = MainWindow()
    w.set_folder(RAW)
    page = w.table_panel.page
    page.table.samples = page.table.samples[:2]
    page.refresh()
    page.on_reduce(all_rows=True)
    code = page.python_code()
    assert "ReductionTable.from_dict(" in code and "reduce_row_2d(" in code
    ns = run_script(code)
    assert set(ns["results"]) == {k for k in w.results}
    for k, g in w.results.items():
        np.testing.assert_allclose(ns["results"][k].I, g.I, rtol=1e-12)
        np.testing.assert_allclose(ns["results"][k].idev(), g.idev(), rtol=1e-12)
    w.close()
