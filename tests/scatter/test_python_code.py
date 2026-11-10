"""pySCAtter "Copy Python code": the script repeats the calculation and the fit."""
import os
import sys

import numpy as np
import pytest

pytest.importorskip("PySide6")
pytest.importorskip("h5py")
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0, os.path.dirname(__file__))
from pycode_util import run_plot, run_script  # noqa: E402
from test_scatter_gui import app, wait, win, write_2d  # noqa: E402,F401


def test_calculation_script(app, win):  # noqa: F811
    p = win.params()
    p.update(ComboBoxParticle=1, EditRadius=4, Length=40, Ordis=0, EditDbeta=15, theta=90, phi=90, GridPoints=24)
    win.params_panel.set_params(p, {})
    win.calculate(two_d=True)
    wait(app, win)
    code = win.python_code()
    ns = run_script(code)
    assert np.allclose(ns["I2"], win.result2d[3], rtol=1e-12)
    assert np.allclose(ns["I"], win.result1d[1], rtol=1e-12)
    run_plot(code)


def test_fit_1d_script(app, win, tmp_path):  # noqa: F811
    from pysasfit.scatter import engine
    p = win.params()
    p.update(EditRadius=8, EditSigma=0.08, I0=100, Base=0.05, GridPoints=16)
    q = np.geomspace(0.05, 2, 80)
    dq = 0.03 * q + 0.004
    I = engine.smeared_1d(p, q, dq)
    dI = 0.02 * I
    path = tmp_path / "c.dat"
    np.savetxt(path, np.column_stack([q, I + np.random.default_rng(3).normal(0, dI), dI, dq]), header="q I Idev Qdev")
    win.on_load_1d(str(path))
    wait(app, win)
    p0 = dict(p, EditRadius=7.0, EditSigma=0.12, I0=50.0, Base=0.0)
    win.params_panel.set_params(p0, {"EditRadius": True, "EditSigma": True, "I0": True, "Base": True})
    code = win.python_code()
    assert "res = fit(" in code and "read_curves(" in code
    win.on_fit()
    wait(app, win, "fit_worker")
    wait(app, win)
    ns = run_script(code)
    for k in ("EditRadius", "EditSigma", "I0", "Base"):
        assert ns["p"][k] == pytest.approx(win.params()[k], rel=1e-6, abs=1e-9)
    run_plot(code)
