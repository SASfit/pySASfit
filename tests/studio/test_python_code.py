"""pySASstudio "Copy Python code": curve, data sets, global parameters with links, the fit."""
import os
import sys
import time

import numpy as np
import pytest

pytest.importorskip("PySide6")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from pycode_util import run_plot, run_script  # noqa: E402


def _sph(q, p):
    R, eta = p[0], p[1]
    x = q * R
    with np.errstate(all="ignore"):
        f = np.where(x > 1e-6, 3 * (np.sin(x) - x * np.cos(x)) / x**3, 1.0)
    return (eta * 4 / 3 * np.pi * R**3 * f) ** 2


@pytest.fixture
def explorer(tmp_path, monkeypatch):
    from PySide6 import QtWidgets
    from pysasfit.studio import register_python_function, unregister_python_function
    monkeypatch.setenv("PYSASFIT_SETTINGS", str(tmp_path / "settings.json"))
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    register_python_function("ff_pysphere", _sph, parameters=("R", "eta"), defaults=(5.0, 1.0))
    from pysasfit.studio.explorer import ModelExplorer
    e = ModelExplorer(ff="ff_pysphere")
    yield e, app
    e.close()
    unregister_python_function("ff_pysphere")


def _data(R, sc, bg, seed):
    q = np.geomspace(0.05, 1.5, 70)
    rng = np.random.default_rng(seed)
    I = sc * _sph(q, [R, 1.0]) + bg
    s = 0.01 * I
    return q, I + rng.normal(0, s), s


def test_curve_only(explorer):
    e, app = explorer
    code = e.python_code()
    assert "register_python_function" in code
    ns = run_script(code)
    q = ns["q"]
    assert np.allclose(ns["I"], e._compute(q)[0], rtol=1e-12)
    run_plot(code)
    e.mode_combo.setCurrentIndex(e.mode_combo.findData("P"))
    ns = run_script(e.python_code())
    assert np.allclose(ns["I"], e._compute(ns["q"])[0], rtol=1e-12)


def test_two_data_sets_global_radius_fit(explorer, tmp_path):
    e, app = explorer
    e.add_contribution("ff_pysphere")
    p1, p2 = e.panels()
    p1.set_datasets_text("1")
    p2.set_datasets_text("2")
    for k, (R, sc, bg) in enumerate(((6.0, 1e-6, 0.01), (12.0, 1e-7, 0.02))):
        np.savetxt(tmp_path / f"d{k}.dat", np.column_stack(_data(R, sc, bg, k + 1)))
    e.fit_panel.load_files([str(tmp_path / "d0.dat"), str(tmp_path / "d1.dat")])
    fp = e.fit_panel
    fp.add_global(value=5.5, lower="1", upper="20")
    fp.links_edit.setPlainText("c1.ff.R = P1\nc2.ff.R = 2 * P1")
    for r in (e.bg_row, p1.scale_row, p2.scale_row):
        r.set_fit(True, emit=True)
    p1.scale_row.set_value(1e-6)
    p2.scale_row.set_value(1e-7)
    fp._local.add("background")
    fp.refresh_table()
    e._redraw()
    code = e.python_code()                 # before the fit: the script fits from the same start values
    assert "problem.link('P1', " in code and "load_data(" in code
    fp._fit_or_stop()
    t0 = time.time()
    while fp._thread is not None and time.time() - t0 < 120:
        app.processEvents()
        time.sleep(0.02)
    gui = fp._summary
    ns = run_script(code)
    s = ns["summary"]
    assert s.success
    assert s.values["P1"] == pytest.approx(gui.values["P1"], rel=1e-6)
    assert s.chi2_red == pytest.approx(gui.chi2_red, rel=1e-6)
    run_plot(code)
