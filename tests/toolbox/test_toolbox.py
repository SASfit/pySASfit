"""pyTOOLbox: HDF5 helpers, viewer, metadata lister (synthetic files; real SANS-I files if present)."""
import os

import numpy as np
import pytest

h5py = pytest.importorskip("h5py")

from pysasfit.toolbox import hdf5  # noqa: E402

RAW = os.environ.get("PYSASRED_TEST_RAW", "/mnt/user-data/uploads/Sept2026/raw")


@pytest.fixture
def nxfile(tmp_path):
    """A small NeXus/NXcanSAS-like file with every kind of member."""
    p = tmp_path / "sans2026n000001.hdf"
    ext = tmp_path / "other.h5"
    with h5py.File(ext, "w") as f:
        f["data"] = np.arange(5.0)
    with h5py.File(p, "w") as f:
        f.attrs["default"] = "entry1"
        e = f.create_group("entry1")
        e.attrs["NX_class"] = "NXentry"
        e.attrs["default"] = "sasdata"
        d = e.create_group("sasdata")
        d.attrs.update({"NX_class": "NXdata", "signal": "I", "axes": "Q"})
        q = np.geomspace(0.01, 0.3, 20)
        d["Q"] = q
        d["Q"].attrs["units"] = "1/angstrom"
        d["I"] = 1 / q ** 2
        d["I"].attrs["uncertainties"] = "Idev"
        d["Idev"] = 0.1 / q ** 2
        s = e.create_group("SANS")
        s["counts"] = np.arange(4 * 6 * 8, dtype=np.int32).reshape(4, 6, 8)
        s.create_dataset("big", data=np.ones((300, 300)), chunks=(50, 50), compression="gzip")
        s["lambda"] = np.array([0.6])
        s["lambda"].attrs["units"] = "nm"
        s["name"] = b"water"
        s["temperature"] = 25.123456
        s.attrs["comment"] = "test"
        rec = np.zeros(3, dtype=[("a", "i4"), ("b", "f8")])
        s["table"] = rec
        e["link_to_counts"] = h5py.SoftLink("/entry1/SANS/counts")
        e["dangling"] = h5py.SoftLink("/nowhere")
        e["external"] = h5py.ExternalLink(str(ext), "/data")
    return str(p)


def test_children_kinds(nxfile):
    kids = {c.name: c for c in hdf5.children(nxfile, "/entry1")}
    assert kids["sasdata"].kind == "group" and kids["sasdata"].is_group and kids["sasdata"].info == "NXdata"
    assert kids["link_to_counts"].kind == "softlink" and "4×6×8" in kids["link_to_counts"].info
    assert kids["dangling"].kind == "broken"
    assert kids["external"].kind == "externallink"
    sans = {c.name: c for c in hdf5.children(nxfile, "/entry1/SANS")}
    assert sans["counts"].info == "4×6×8  int32"


def test_describe_dataset_and_attributes(nxfile):
    meta, attrs = hdf5.describe(nxfile, "/entry1/SANS/big")
    m = dict(meta)
    assert m["Shape"] == "(300, 300)" and m["Chunks"] == "(50, 50)" and m["Filters"].startswith("gzip")
    _meta, attrs = hdf5.describe(nxfile, "/entry1/SANS")
    assert ("comment", "test") in [(a, v) for a, v, _t in attrs]


def test_read_slice_orders_and_strides(nxfile):
    full = np.arange(4 * 6 * 8).reshape(4, 6, 8)
    a, st = hdf5.read_slice(nxfile, "/entry1/SANS/counts", x_dim=2, y_dim=1, index=(3, 0, 0))
    np.testing.assert_array_equal(a, full[3])                 # rows = y = dim 1
    a, st = hdf5.read_slice(nxfile, "/entry1/SANS/counts", x_dim=1, y_dim=2, index=(2, 0, 0))
    np.testing.assert_array_equal(a, full[2].T)               # rows = y = dim 2
    a, st = hdf5.read_slice(nxfile, "/entry1/SANS/counts", x_dim=0, index=(0, 5, 7))
    np.testing.assert_array_equal(a, full[:, 5, 7])
    a, st = hdf5.read_slice(nxfile, "/entry1/SANS/big", x_dim=1, y_dim=0, max_points=10_000)
    assert st == (3, 3) and a.shape == (100, 100)


def test_nxdata_follows_default(nxfile):
    nx = hdf5.nxdata(nxfile, "/")
    assert nx.signal == "/entry1/sasdata/I" and nx.axes == ["/entry1/sasdata/Q"]
    assert nx.errors == "/entry1/sasdata/Idev"
    assert hdf5.nxdata(nxfile, "/entry1/SANS") is None


def test_find(nxfile):
    assert "/entry1/SANS/counts" in hdf5.find(nxfile, "COUNTS")


def test_metadata_roundtrip_keeps_original(nxfile):
    from pysasfit.toolbox.meta_lister import collect_entries, write_values
    e = collect_entries(nxfile)
    assert e["/entry1/SANS/temperature"] == 25.123            # 3 digits (preferences default)
    assert e["/entry1/SANS/name"] == "water"
    assert "/entry1/SANS/counts" not in e                     # 192 elements: not metadata
    assert e["/entry1/SANS@comment"] == "test"
    err = write_values(nxfile, {"/entry1/SANS/name": "heavy water, long name", "/entry1/SANS@comment": "new",
                                "/entry1/SANS/temperature": 30.0})
    assert err == {}
    e = collect_entries(nxfile)
    assert e["/entry1/SANS/name"] == "heavy water, long name" and e["/entry1/SANS/old_name"] == "water"
    assert e["/entry1/SANS@comment"] == "new" and e["/entry1/SANS@old_comment"] == "test"
    assert e["/entry1/SANS/temperature"] == 30.0


# ---------------------------------------------------------------- GUI
def _app():
    pytest.importorskip("PySide6")
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6 import QtWidgets
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


def test_viewer_modes(nxfile):
    app = _app()
    from pysasfit.toolbox.gui import MainWindow
    w = MainWindow()
    v = w.viewer
    v.open_file(nxfile)
    v.select_path(nxfile, "/entry1/SANS/counts")
    app.processEvents()
    assert v.current == (nxfile, "/entry1/SANS/counts")
    assert v._effective_mode() == "Heatmap" and v.stack.currentIndex() == 0
    ax0 = v.canvas.figure.axes[0]
    assert len(ax0.images) == 1 and len(ax0.child_axes) == 1    # image and its attached colour bar
    assert ax0.get_anchor() == "W"                            # flush left
    v.index_spins[0][1].setValue(3)                          # slice 3 of dim 0
    img = v.canvas.figure.axes[0].images[0].get_array()
    np.testing.assert_array_equal(np.asarray(img), np.arange(192).reshape(4, 6, 8)[3])
    v.mode.setCurrentText("Matrix")
    assert v.stack.currentIndex() == 1 and v.matrix_model.rowCount() == 6 and v.matrix_model.columnCount() == 8
    v.mode.setCurrentText("Line")
    assert v.canvas.figure.axes[0].lines[0].get_ydata().size == 8
    v.mode.setCurrentText("Auto")
    v.select_path(nxfile, "/entry1/SANS/name")
    assert v.stack.currentIndex() == 2 and "water" in v.text.toPlainText()
    v.select_path(nxfile, "/entry1/SANS/table")
    assert v._effective_mode() == "Matrix"
    v.select_path(nxfile, "/entry1/sasdata")                 # NXdata group: I(Q) with error bars
    assert v._effective_mode() == "NX" and v.canvas.figure.axes[0].get_xlabel() == "Q [1/angstrom]"
    v.select_path(nxfile, "/entry1/external")
    assert v.current == (nxfile, "/entry1/external")
    v.search.setText("lambda")
    v._search()
    assert v.results.count() == 1
    v.results.itemClicked.emit(v.results.item(0))            # shows /entry1/SANS/lambda
    assert v.current == (nxfile, "/entry1/SANS/lambda")
    assert v.attr_table.item(0, 0).text() == "units" and v.attr_table.item(0, 1).text() == "nm"
    w.close()


def test_lister_tab_lists_run_range(nxfile, tmp_path):
    _app()
    from pysasfit.toolbox.gui import MainWindow
    w = MainWindow()
    L = w.lister
    L.folder_edit.setText(str(tmp_path))
    L.instr_edit.setText("sans")
    L.year_edit.setText("2026")
    L.from_edit.setText("1")
    L.to_edit.setText("10")
    L.scan()
    assert "/entry1/SANS/temperature" in L.entry_state
    for k in L.entry_state:
        L.entry_state[k] = k in ("/entry1/SANS/temperature", "/entry1/SANS/name")
    L.generate()
    assert L.table.rowCount() == 1 and sorted(L.columns) == ["file", "name", "temperature"]
    opened = []
    L.open_file_requested.connect(opened.append)
    L._on_double_click(0, 0)
    assert opened == [nxfile] and w.viewer.current[0] == nxfile
    w.close()


@pytest.mark.skipif(not os.path.isdir(RAW), reason="SANS-I raw data not available")
def test_viewer_on_real_sans_i_file():
    app = _app()
    from pysasfit.toolbox.viewer import HDF5Viewer
    v = HDF5Viewer()
    p = os.path.join(RAW, "sans2026n082130.hdf")
    v.select_path(p, "/entry1/SANS/detector/counts")
    app.processEvents()
    assert v._effective_mode() in ("Heatmap", "Line")
    assert v.canvas.figure.axes


def _detector_file(tmp_path):
    """NXdetector with rectangular pixels (x 5 mm, y 0.8 cm) and a frame stack, linked into NXdata."""
    p = str(tmp_path / "det.h5")
    with h5py.File(p, "w") as f:
        d = f.create_group("entry/instrument/detector")
        d.attrs["NX_class"] = "NXdetector"
        d["data"] = np.arange(3 * 4 * 6, dtype=float).reshape(3, 4, 6) + 1
        d["x_pixel_size"] = 5.0
        d["x_pixel_size"].attrs["units"] = "mm"
        d["y_pixel_size"] = np.array([0.8])
        d["y_pixel_size"].attrs["units"] = "cm"
        g = f.create_group("entry/data")
        g.attrs.update({"NX_class": "NXdata", "signal": "data"})
        g["data"] = d["data"]                                   # hard link
    return p


def test_pixel_size_and_aspect(tmp_path):
    p = _detector_file(tmp_path)
    assert hdf5.pixel_size(p, "/entry/instrument/detector/data") == pytest.approx((0.005, 0.008))
    assert hdf5.pixel_size(p, "/entry/data/data") == pytest.approx((0.005, 0.008))   # found via the link
    assert hdf5.pixel_aspect((0.005, 0.008), 3, 2, 1) == pytest.approx(1.6)
    assert hdf5.pixel_aspect((0.005, 0.008), 3, 1, 2) == pytest.approx(0.625)        # axes swapped
    assert hdf5.pixel_aspect((0.005, 0.008), 3, 2, 0) is None                        # frame axis shown


def test_viewer_keeps_pixel_aspect(tmp_path):
    _app()
    from pysasfit.toolbox.viewer import HDF5Viewer
    p = _detector_file(tmp_path)
    v = HDF5Viewer()
    v.select_path(p, "/entry/data/data")
    ax = v.canvas.figure.axes[0]
    assert v._effective_mode() == "Heatmap" and ax.get_aspect() == pytest.approx(1.6)
    v.keep_aspect.setChecked(False)
    assert v.canvas.figure.axes[0].get_aspect() == "auto"
    v.keep_aspect.setChecked(True)
    v.select_path(p, "/entry/data")                             # NXdata group
    assert v._effective_mode() == "NX" and v.canvas.figure.axes[0].get_aspect() == pytest.approx(1.6)


def test_nxcansas_i_axes(tmp_path):
    """NXcanSAS names the axis in @I_axes (not @axes): the default plot must still use Q."""
    p = str(tmp_path / "red.h5")
    with h5py.File(p, "w") as f:
        f.attrs["default"] = "sasentry01"
        e = f.create_group("sasentry01")
        e.attrs["default"] = "sasdata01"
        d = e.create_group("sasdata01")
        d.attrs.update({"NX_class": "NXdata", "canSAS_class": "SASdata", "signal": "I", "I_axes": "Q"})
        d["Q"] = np.linspace(0.1, 1, 5)
        d["I"] = np.ones(5)
        d["I"].attrs["uncertainties"] = "Idev"
        d["Idev"] = np.full(5, 0.1)
    nx = hdf5.nxdata(p, "/")
    assert nx.axes == ["/sasentry01/sasdata01/Q"] and nx.errors == "/sasentry01/sasdata01/Idev"


def test_old_package_name_still_imports():
    """pysasfit.sasbox (pySASbox) is the same code as pysasfit.toolbox."""
    import warnings

    from pysasfit.toolbox import hdf5 as new
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        from pysasfit.sasbox import hdf5 as old
    assert old is new


def test_tab_style_marks_the_selected_tab():
    pytest.importorskip("PySide6")
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6 import QtWidgets
    from pysasfit.core.tabstyle import apply_tab_style, tab_style_sheet
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    before = app.styleSheet()
    try:
        apply_tab_style(app)
        apply_tab_style(app)                                   # only once
        assert app.styleSheet().count("pysasfit-tab-style") == 1
        assert "QTabBar::tab:selected" in tab_style_sheet() and "#d9d9d9" in tab_style_sheet()
    finally:
        app.setStyleSheet(before)
