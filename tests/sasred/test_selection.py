"""Select many samples: run range, pattern, header conditions; grouping with ± tolerances."""
import os
from types import SimpleNamespace

import pytest

from pysasfit.sasred.selection import Criterion, Metadata, group_selected, select_runs

RAW = os.environ.get("PYSASRED_TEST_RAW", "/mnt/user-data/uploads/Sept2026/raw")
need = pytest.mark.skipif(not os.path.isdir(RAW), reason="Vifor raw data not available")


def _runs():
    T = [20.0, 20.4, 25.1, 24.8, 20.2, 40.0, 25.0, 39.6]
    return [SimpleNamespace(run=100 + k, sample_name="S" if k < 6 else "R", T=t, kind="scattering",
                            start_time=f"2026-09-29 1{k}:00:00", counting_time_s=60.0, path="")
            for k, t in enumerate(T)]


def test_criteria():
    assert Criterion("sample", text="EXP-.*").test("EXP-005370-A_1")
    assert not Criterion("sample", text="EXP-.*").test("H2O")
    assert not Criterion("sample", text="EXP-.*", regex=False).test("EXP-1")
    assert Criterion("attenuator", text="0").test("0") and Criterion("attenuator", text="0").test(0.0)
    assert Criterion("L2_m", "range", lo=1.99, hi=2.01).test(2.0007)
    assert not Criterion("L2_m", "range", lo=1.99, hi=2.01).test(8.0)
    assert Criterion("x", "range", lo=None, hi=5).test(-3) and not Criterion("x", "range", lo=1).test(None)


def test_select_and_group_with_tolerance():
    meta = Metadata(_runs(), extra={"temperature": lambda r: r.T})
    assert "temperature" in meta.fields() and meta.is_numeric("temperature")
    sel = select_runs(meta, (101, 107))
    assert [r.run for r in sel] == list(range(101, 108))
    sel = select_runs(meta, (100, 107), pattern=(1, 1))                     # use 1, skip 1
    assert [r.run for r in sel] == [100, 102, 104, 106]
    sel = select_runs(meta, criteria=[Criterion("sample", text="S"), Criterion("temperature", "range", hi=30)])
    assert [r.run for r in sel] == [100, 101, 102, 103, 104]
    g = group_selected(meta, meta.items, {"sample": None, "temperature": 0.5})   # ± 0.5 K
    got = sorted((x.values["sample"], tuple(x.run_numbers)) for x in g)
    assert got == [("R", (106,)), ("R", (107,)), ("S", (100, 101, 104)), ("S", (102, 103)), ("S", (105,))]


@need
def test_vifor_like_bersans():
    """EXP-.*, λ 0.59–0.61 nm, attenuator 0, collimation 8 m, SD 1.99–2.01 m: 52 runs as in BerSANS."""
    from pysasfit.sasred.catalogue import scan_folder
    from pysasfit.sasred.reduction_table import ReductionTable
    meta = Metadata(scan_folder(RAW))
    assert "hdf:SANS/Dornier-VS/lambda" in meta.fields() and "hdf:sample/name" in meta.fields()
    crit = [Criterion("sample", text="EXP-.*"), Criterion("hdf:SANS/Dornier-VS/lambda", "range", lo=0.59, hi=0.61),
            Criterion("attenuator", text="0"), Criterion("L1_m", "range", lo=8, hi=8),
            Criterion("L2_m", "range", lo=1.99, hi=2.01)]
    sel = select_runs(meta, (82059, 82294), criteria=crit)
    assert len(sel) == 52 and sel[0].run == 82096
    groups = group_selected(meta, sel, {"sample": None})
    assert len(groups) == 30 and all(len(g.runs) <= 2 for g in groups)
    groups = group_selected(meta, sel, {"sample": None, "sample_position_mm": 0.25})
    assert len(groups) == 31                        # A_2 measured at two changer positions
    t = ReductionTable()
    rows = t.add_groups(groups)
    assert len(t.samples) == 31 and rows[0].info.startswith("EXP-005370-A_1 · sample_position_mm")


@need
def test_add_many_samples_dialog():
    pytest.importorskip("PySide6")
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6 import QtWidgets
    from pysasfit.sasred.gui import MainWindow
    from pysasfit.sasred.selection_gui import AddManySamplesDialog
    QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    w = MainWindow()
    w.set_folder(RAW)
    d = AddManySamplesDialog(w.entries, w)
    d.search.set_values("sample", "EXP-.*")
    d.search.set_values("hdf:SANS/Dornier-VS/lambda", 0.59, 0.61)
    d.search.set_values("attenuator", "0")
    d.search.set_values("L2_m", 1.99, 2.01)
    d.on_search()
    assert len(d.selected) == 52 and d.tabs.currentIndex() == 1 and d.sel_list.count() == 52
    assert len(d.groups()) == 30 and d.preview.rowCount() == 30
    d.group.set_values("sample_position_mm", 0.25)
    assert len(d.groups()) == 31
    d.run_lo.setValue(82200)
    d.on_search()
    assert all(r.run >= 82200 for r in d.selected)
    p = w.table_panel
    p.on_clear()                                    # a new tab filled by hand
    p.add_groups(d.groups(), check=False)
    assert len(p.table.samples) == len(d.groups()) and p.samp.rowCount() == len(p.table.samples)
    assert p.table.samples[0].trans_runs                    # transmission runs of the same sample
    w.close()
