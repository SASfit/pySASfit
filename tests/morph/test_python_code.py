"""pySASmorph "Copy Python code": the script repeats correlations and the reconstruction."""
import os
import sys

import numpy as np
import pytest

pytest.importorskip("PySide6")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
from PySide6 import QtWidgets  # noqa: E402

from pysasfit.morph import LevelCut, levelcut_intensity  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from pycode_util import run_plot, run_script  # noqa: E402


def _write(path, scale=1.0):
    r = np.linspace(0, 3000, 20001)
    kk = 2 * np.pi / 150.0
    g = np.exp(-(r / 60.0) ** 2) * np.where(r > 0, np.sin(kk * r) / (kk * np.where(r > 0, r, 1)), 1.0)
    q = np.geomspace(1e-3, 0.4, 200)
    I = scale * levelcut_intensity(q, r, LevelCut(0.3).F(g))
    np.savetxt(path, np.column_stack([q, I, 0.02 * I]))
    return str(path)


def test_reconstruction_script(tmp_path):
    QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    from pysasfit.morph.gui import MainWindow
    w = MainWindow()
    w.load_file(_write(tmp_path / "a.dat"))
    code = w.python_code()
    ns = run_script(code)
    assert "correlations_multi" in code
    assert np.allclose(ns["res"].g, w.result.g)
    w.set_phases(["solid", "pore"], [0.7, 0.3])
    w.n.setValue(32)
    w.seed.setValue(7)
    w.on_reconstruct()
    assert w.result.labels is not None
    code = w.python_code()
    assert "reconstruct_multi" in code
    ns = run_script(code)
    assert np.array_equal(ns["res"].labels, w.result.labels)
    run_plot(code)
    w.close()


def test_contrast_series_script(tmp_path):
    QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    from pysasfit.morph.gui import MainWindow
    w = MainWindow()
    w.load_file(_write(tmp_path / "a.dat"))
    w.load_file(_write(tmp_path / "b.dat", 0.5), add=True)
    ns = run_script(w.python_code())
    assert len(ns["sets"]) == 2
    assert np.allclose(ns["res"].g, w.result.g)
    w.close()


def test_anisotropic_window_script(tmp_path):
    QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    sys.path.insert(0, os.path.dirname(__file__))
    from test_anisotropic import _data
    from pysasfit.morph.aniso_gui import AnisoWindow
    d, *_ = _data(n=90)
    p = tmp_path / "synth.npz"
    np.savez(p, qx=d.qx, qy=d.qy, I=d.I, sigma=d.sigma)
    w = AnisoWindow()
    w.load_file(str(p))
    w.n.setValue(32)
    w.box_size.setText("600")
    w.seed.setText("3")
    w.mode.setCurrentIndex(w.mode.findData("A"))
    w.result = type("R", (), {"matrix": np.zeros(1)})()    # as after a reconstruction: the script reconstructs
    code = w.python_code()
    assert "reconstruct_aniso(data, params)" in code
    res = w.run(True)
    ns = run_script(code)
    assert ns["res"].stretch == pytest.approx(res.stretch, rel=1e-9)
    assert np.array_equal(ns["res"].matrix, res.matrix)
    w.close()
