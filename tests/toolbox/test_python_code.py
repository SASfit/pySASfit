"""pyTOOLbox "Copy Python code": merge, 2D averages, 1D arithmetic, HDF5 viewer, metadata lister."""
import os
import sys

import numpy as np
import pytest

pytest.importorskip("PySide6")
h5py = pytest.importorskip("h5py")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0, os.path.dirname(__file__))
from pycode_util import run_plot, run_script  # noqa: E402
from test_image2d import bersans2d  # noqa: E402
from test_merge import folder  # noqa: E402,F401
from test_toolbox import nxfile  # noqa: E402,F401


@pytest.fixture
def win():
    from PySide6 import QtWidgets
    QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    from pysasfit.toolbox.gui import MainWindow
    w = MainWindow()
    yield w
    w.close()


def test_merge_script(win, folder):  # noqa: F811
    t = win.merger
    t.add_folder(str(folder))
    t.regex.setText(r"EXP-1-(\w)_1")
    t.add_item("Sample/Temperature", "0.5")
    t.on_merge()
    win.tabs.setCurrentWidget(t)
    code = win.python_code()
    ns = run_script(code)
    assert len(ns["merged"]) == len(t.merged)
    for a, b in zip(ns["merged"], t.merged):
        np.testing.assert_allclose(a.I, b.I, rtol=1e-12)
    run_plot(code)


def test_image_average_script(win, tmp_path):
    bersans2d(tmp_path / "D0000001.008")
    t = win.images
    t.add_paths([str(tmp_path / "D0000001.008")])
    t.cx.setValue(7.0)
    t.cy.setValue(8.0)
    t.on_apply_geometry()
    t.current().mask[0, :] = True                     # an edited mask travels with the script
    for kind in ("radial", "sector", "azimuthal", "sin2"):
        t.avg_kind.setCurrentIndex(t.avg_kind.findData(kind))
        if kind == "azimuthal":
            t.q1.setValue(0.01)
            t.q2.setValue(0.05)
        gui = t.average()
        ns = run_script(t.python_code())
        assert len(ns["results"]) == len(gui)
        for a, b in zip(ns["results"], gui):
            np.testing.assert_allclose(a.y, b.y, rtol=1e-12, equal_nan=True)


def test_data1d_recipes(win, tmp_path):
    q = np.geomspace(0.01, 1, 30)
    for n, s in (("a.dat", 1.0), ("b.dat", 0.5)):
        np.savetxt(tmp_path / n, np.column_stack([q, s / (1 + (5 * q) ** 2), 0.01 * np.ones_like(q)]))
    t = win.data1d
    from PySide6 import QtWidgets
    orig = QtWidgets.QFileDialog.getOpenFileNames
    QtWidgets.QFileDialog.getOpenFileNames = staticmethod(lambda *a, **k: ([str(tmp_path / "a.dat"),
                                                                             str(tmp_path / "b.dat")], ""))
    try:
        t.on_add_files()
    finally:
        QtWidgets.QFileDialog.getOpenFileNames = orig
    assert len(t.data) == 2
    t.a_combo.setCurrentIndex(0)
    t.b_combo.setCurrentIndex(1)
    t.op.setCurrentIndex(t.op.findData("a-f*b"))
    t.f.setText("0.8")
    t.on_binary()                                       # a - 0.8 b
    t.table.selectRow(2)
    t.value.setText("3")
    t.on_scalar("scale")                                # (a - 0.8 b) * 3
    t.table.selectRow(3)
    win.tabs.setCurrentWidget(t)
    code = win.python_code()
    assert "d1.combine(ds1, ds2, 'a-f*b', 0.8)" in code and ".scale(3.0)" in code
    ns = run_script(code)
    np.testing.assert_allclose(ns["results"][0].y, t.data[3].y, rtol=1e-14)
    run_plot(code)


def test_viewer_and_lister_scripts(win, nxfile, tmp_path):  # noqa: F811
    v = win.viewer
    v.open_file(nxfile)
    v.select_path(nxfile, "/entry1/SANS/counts")
    v.index_spins[0][1].setValue(3)
    ns = run_script(v.python_code())
    np.testing.assert_array_equal(ns["a"], np.arange(192).reshape(4, 6, 8)[3])
    L = win.lister
    L.folder_edit.setText(str(tmp_path))
    L.instr_edit.setText("sans")
    L.year_edit.setText("2026")
    L.from_edit.setText("1")
    L.to_edit.setText("10")
    L.scan()
    for k in L.entry_state:
        L.entry_state[k] = k in ("/entry1/SANS/temperature", "/entry1/SANS/name")
    ns = {"__name__": "x"}
    exec(L.python_code(), ns)
    assert len(ns["rows"]) == 1 and ns["rows"][0][0] == "sans2026n000001.hdf"
