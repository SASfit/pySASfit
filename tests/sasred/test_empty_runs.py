"""Runs without usable data (measuring time 0, 0 counts, monitor 0) are skipped and marked."""
import copy
import os

import numpy as np
import pytest

VIFOR = os.environ.get("PYSASRED_TEST_RAW", "/mnt/user-data/uploads/Sept2026/raw")
MAREN = "/mnt/user-data/uploads/Maren"


def _one(folder, run):
    from pysasfit.sasred.catalogue import scan_folder
    cat = scan_folder(folder)
    return cat, next(e for e in cat if e.run == run)


def test_empty_reason_and_combined_counts_skip():
    if not os.path.isdir(VIFOR):
        pytest.skip("Vifor raw data not available")
    from pysasfit.sasred.io.sinq import read_sans_i
    from pysasfit.sasred.reduce import combined_normalised_counts, load_instrument
    r = read_sans_i(os.path.join(VIFOR, "sans2026n082187.hdf"), checksum=False)
    assert r.empty_reason == ""
    z = copy.deepcopy(r)
    z.run = 1
    z.monitors = dict(z.monitors, monitor1=0.0)
    assert "monitor1 0" in z.empty_reason
    t = copy.deepcopy(r)
    t.run = 2
    t.counting_time_s = 0.0
    c = copy.deepcopy(r)
    c.run = 3
    c.counts = np.zeros_like(c.counts)
    assert t.empty_reason == "measuring time 0 s" and c.empty_reason == "0 counts"
    inst = load_instrument(r)
    notes = []
    x, _ = combined_normalised_counts([r, z, t, c], inst, notes)
    y, _ = combined_normalised_counts([r], inst, [])
    assert np.allclose(x.value, y.value)
    assert any("run 1 skipped: monitor1 0" in n for n in notes) and any("run 3 skipped: 0 counts" in n for n in notes)
    with pytest.raises(ValueError, match="no usable runs"):
        combined_normalised_counts([z, c], inst, [])


def test_empty_llb_run_marked_and_left_out_of_the_plan():
    if not os.path.isdir(MAREN):
        pytest.skip("SANS-LLB data not available")
    from pysasfit.sasred import plan as pl
    from pysasfit.sasred.catalogue import COLUMNS
    cat, e = _one(MAREN, 5279)
    assert e.skip == "measuring time 0 s, 0 counts, monitor2 0"
    row = e.row()
    assert len(row) == len(COLUMNS) and row[-1].startswith("skipped: measuring time 0 s")
    p = pl.build_plan(cat)
    assert all(5279 not in x.runs for x in p.entries)
    assert (5279, e.skip) in p.skipped and "Skipped runs" in p.markdown()


def test_gui_run_table_comment():
    if not os.path.isdir(MAREN):
        pytest.skip("SANS-LLB data not available")
    pytest.importorskip("PySide6")
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6 import QtWidgets
    from pysasfit.sasred.catalogue import COLUMNS
    from pysasfit.sasred.gui import MainWindow
    QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    w = MainWindow()
    w.set_folder(MAREN)
    t = w.table
    i = next(i for i in range(t.rowCount()) if t.item(i, 0).data(0) == 5279)
    assert t.item(i, len(COLUMNS) - 1).text().startswith("skipped:")
    assert t.item(i, 1).font().strikeOut()
    for tab in w.table_panel.tables():
        for r in tab.rows():
            assert 5279 not in r.used
    w.close()
