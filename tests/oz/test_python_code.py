"""pyOZsolver "Copy Python code": every tab gives a script that repeats its calculation."""
import os
import sys

import numpy as np
import pytest

pytest.importorskip("PySide6")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
from PySide6 import QtWidgets  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from pycode_util import run_plot, run_script  # noqa: E402


@pytest.fixture(scope="module")
def oz():
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    from pysasfit.oz.gui.main_window import OZWidget
    w = OZWidget(redirect_output=False)
    yield w
    w.shutdown()
    w.close()


def test_oz_solver_tab(oz):
    from pysasfit.oz import ozLib
    t = oz.ozTab
    t.verifyVar.set(False)
    kw = t._solveArguments()
    ref = ozLib.solve(**kw)
    oz.setCurrentWidget(t)
    code = oz.python_code()
    assert "ozLib.solve(" in code
    ns = run_script(code)
    assert np.allclose(ns["result"].Sq, ref.Sq, rtol=1e-10, atol=1e-12)
    run_plot(code)


def test_generic_polydisperse_tab(oz):
    g = oz.tools["Polydisperse (any potential)"]
    g._settings = {"gridN": 1023, "pointsPerSigma": 40, "maxIterations": 8000, "transformType": 1}
    g.nQVar.set("30")
    oz.setCurrentWidget(g)
    ns = run_script(oz.python_code())
    g._onCompute()
    g.worker.join(300)
    g._poll()
    res = g.runs[-1] if g.runs else g.result
    assert np.allclose(ns["I"], res.I_exact, rtol=1e-8)
    assert np.allclose(ns["S_number"], res.S_number, rtol=1e-8)


def test_ry_tab(oz):
    t = oz.tools["RY Polydisperse Yukawa"]
    t.nbinsVar.set("2")
    t.nQVar.set("20")
    code = t.python_code()
    compile(code, "x", "exec")
    ns = run_script(code)
    assert ns["S_number"].shape == (20,) and np.all(np.isfinite(ns["S_number"]))


def test_mixture_validation_tab(oz):
    t = oz.tools["Mixture validation"]
    t.nbinsVar.set("2")
    t.gridKVar.set("10")
    t.refineVar.set("2")
    t.ppsVar.set("40")
    t.nQVar.set("20")
    ns = run_script(t.python_code())
    c = ns["curves"]
    assert np.max(np.abs(c["numerical (fine)"] - c["Gazzillo MSA (neutral = PY)"])) < 0.05
