"""2D data in pyTOOLbox: readers, geometry, display."""
import os

import numpy as np
import pytest

from pysasfit.toolbox import image2d as i2

RAW = os.environ.get("PYSASRED_TEST_RAW", "/mnt/user-data/uploads/Sept2026/raw")
BERSANS2D = "/mnt/user-data/uploads/Vifor/1stJuly2024/analysed/D0007613.008"


def bersans2d(path, n=16, centre=(7.0, 8.0)):
    iy, ix = np.mgrid[0:n, 0:n]
    I = np.exp(-((ix - centre[0]) ** 2 + (iy - centre[1]) ** 2) / 20.0) + 0.01 * ix   # ix: orientation check
    lines = ["%File", "Type=SANSDAni", f"DataSizeX={n}", f"DataSizeY={n}", "%Sample", "SampleName=S1",
             "%Setup", "Lambda=0.6", "SD=8.0", "%Counts"]
    v = I.ravel()
    lines += [" ".join(f"{x:.6e}" for x in v[k:k + 8]) for k in range(0, v.size, 8)]
    lines.append("%Errors")
    lines += [" ".join(f"{0.1:.6e}" for _ in range(8)) for _ in range(0, v.size, 8)]
    with open(path, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    return I


def test_bersans_2d_and_geometry(tmp_path):
    I = bersans2d(tmp_path / "D0000001.008")
    im = i2.read_images(tmp_path / "D0000001.008")[0]
    assert np.allclose(im.I, I, rtol=1e-6) and im.dI[0, 0] == pytest.approx(0.1)
    assert im.sample == "S1" and im.SD_m == 8.0 and im.wavelength_A == pytest.approx(6.0) and not im.has_q
    im.set_geometry((7.0, 8.0))
    assert im.has_q and im.q()[8, 7] == 0
    psi = im.psi_deg()
    assert psi[8, 10] == pytest.approx(0) and psi[11, 7] == pytest.approx(90) and psi[8, 4] == pytest.approx(180)
    qx, qy = im.qxy()
    q1 = 4 * np.pi / 0.6 * np.sin(np.arctan(3 * 0.0075 / 8.0) / 2)
    assert qx[8, 10] == pytest.approx(q1) and qy[11, 7] == pytest.approx(q1)


@pytest.mark.skipif(not os.path.isfile(BERSANS2D), reason="BerSANS 2D file not available")
def test_real_bersans_2d():
    im = i2.read_images(BERSANS2D)[0]
    assert im.I.shape == (128, 128) and im.dI is not None and im.sample == "EXP-001763_2"


@pytest.mark.skipif(not os.path.isdir(RAW), reason="Vifor raw data not available")
def test_pysasred_2d_memory_file_and_gui(tmp_path):
    pytest.importorskip("PySide6")
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6 import QtWidgets
    QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    from pysasfit.sasred.gui import MainWindow as Red
    from pysasfit.sasred.io.nxcansas import write_nxcansas_2d
    from pysasfit.toolbox.gui import MainWindow as Box
    red = Red()
    red.set_folder(RAW)
    page = red.table_panel.pages()[0]
    red.reduce_table_rows(page.table, page.table.samples[:1])
    key, r = next(iter(red.results.items()))
    m = i2.image_from_reduced(r, key)
    assert m.I.shape == (128, 128) and m.has_q and m.mask.any() and "1/cm/sr" in m.unit
    assert np.allclose(m.q()[~m.mask], 10 * r.geometry.q[~m.mask], rtol=1e-3)      # same q as pySASred
    p = write_nxcansas_2d(tmp_path / "x_2d.h5", r)
    f = i2.read_images(p)[0]
    assert np.allclose(f.I, m.I) and np.array_equal(f.mask, m.mask) and f.centre_px == pytest.approx(m.centre_px)
    assert np.allclose(f.q(), m.q()) and f.sample == m.sample and f.meta.get("SD_m") == r.raw.SD_m
    # GUI: images from pySASred, from the merge tool, all views
    box = Box()
    t = box.images
    t.add_reduced({key: r})
    t.list.selectRow(0)
    for v in ("q", "pixels", "polar"):
        t.view.setCurrentIndex(t.view.findData(v))
        t.rings.setText("0.2 0.4")
        t.replot()
        assert t.canvas.figure.axes
    mt = box.merger
    mt.add_reduced({key: r})
    mt.files.selectRow(0)
    mt.on_show_2d()
    assert box.tabs.currentWidget() is t and len(t.images) == 1          # same key: replaced
    # files written by pySASred's save_all carry the 2D next to the 1D
    paths = red.save_all(str(tmp_path))
    assert any(x.endswith("_2d.h5") for x in paths)
    box.close()
    red.close()
