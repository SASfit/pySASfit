"""Missing inputs of a reduction: listed (panel 'Missing input', log) and the cells coloured in the table."""
import os

import pytest

RAW = os.environ.get("PYSASRED_TEST_RAW", "/mnt/user-data/uploads/Sept2026/raw")
C8 = "L2 8.0 m, λ 6.00 Å, L1 8.0 m"


def test_missing_inputs_of_a_table():
    from pysasfit.sasred import reduction_table as rt
    t = rt.ReductionTable(config="manual")
    s = t.add_sample("S")
    s.set_runs([10])
    m = {(x.role, x.index, x.field) for x in t.missing_inputs()}
    assert ("blocked_beam", None, "runs") in m and ("sample_background", None, "runs") in m
    assert ("water", None, "runs") in m                              # no absolute scale
    assert ("sample", 0, "transmission") in m and ("sample", 0, "thickness") in m
    s.transmission, s.scaling = 0.8, 0.1
    t.corrections["water"].set_runs([1])
    t.corrections["water"].aperture_cm2 = 0.5
    m = {(x.role, x.index, x.field): x for x in t.missing_inputs()}
    assert ("water", None, "transmission") in m and ("empty_cell", None, "runs") in m
    assert m[("sample", 0, "aperture")].severity == "check"
    assert ("sample", 0, "transmission") not in m
    t.scale = "direct_beam"
    m = {(x.role, x.field) for x in t.missing_inputs()}
    assert ("direct_beam", "runs") in m and ("direct_beam", "efficiency") in m and ("water", "transmission") not in m
    t.scale = "relative"
    assert not any(x.role == "water" for x in t.missing_inputs())   # chosen on purpose: not missing
    s.runs[0].use = False
    assert any(x.index == 0 and x.field == "runs" for x in t.missing_inputs())


def test_gui_missing_panel_and_red_cells():
    if not os.path.isdir(RAW):
        pytest.skip("Vifor raw data not available")
    pytest.importorskip("PySide6")
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6 import QtWidgets
    from pysasfit.sasred.gui import MainWindow
    from pysasfit.sasred.table_gui import COLS_SAMPLE, MISSING_BG
    QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    w = MainWindow()
    w.set_folder(RAW)
    page = next(p for p in w.table_panel.pages() if p.table.config == C8)
    t = page.table
    t.samples = t.samples[:2]
    t.samples[1].scaling = None                       # thickness missing
    t.samples[0].runs[0].use = False
    t.samples[0].runs[1:] = []                        # all runs off
    page.refresh()
    j = COLS_SAMPLE.index("Thickness [cm]")
    assert page.samp.item(1, j).background().color() == MISSING_BG
    assert "d = 1 cm" in page.samp.item(1, j).toolTip()
    k = COLS_SAMPLE.index("Summed runs")
    assert page.samp.item(0, k).background().color() == MISSING_BG
    w.reduce_table_rows(t, [t.samples[1]])
    items = [w.missing_view.topLevelItem(i) for i in range(w.missing_view.topLevelItemCount())]
    texts = [(it.text(1), it.text(2)) for it in items]
    assert any(f == "thickness" for _, f in texts) and any(f == "runs" and lab == "Cadmium" for lab, f in texts)
    assert w.docks["missing"].windowTitle().startswith("Missing input (")
    assert "missing: " in w.log_view.toPlainText()
    it = next(it for it in items if it.text(2) == "thickness")
    w._on_missing_clicked(it)
    assert w.table_panel.page is page and page.samp.currentRow() == 1 and page.samp.currentColumn() == j
    w.close()


def test_optional_corrections_can_be_left_out_or_switched_off(tmp_path):
    """Cd, water background and sample background are optional: no runs or switched off in the table."""
    from pysasfit.sasred import reduction_table as rt
    t = rt.ReductionTable(config="manual")
    s = t.add_sample("S")
    s.set_runs([10])
    s.transmission, s.scaling = 0.8, 0.1
    t.corrections["water"].set_runs([1])
    t.corrections["water"].transmission = 0.5
    sev = {(x.role, x.field): x.severity for x in t.missing_inputs()}
    assert sev[("blocked_beam", "runs")] == "info" and sev[("sample_background", "runs")] == "info"
    assert sev[("empty_cell", "runs")] == "check"
    assert not any("Cadmium" in p for p in t.problems())
    for role in ("blocked_beam", "empty_cell", "sample_background"):
        t.corrections[role].set_runs([2])
        t.corrections[role].transmission = 0.9
    assert not t.missing_inputs()
    t.ignored = ["blocked_beam", "sample_background", "empty_cell"]
    sev = {(x.role, x.field): x.severity for x in t.missing_inputs()}
    assert set(sev.values()) == {"info"} and ("blocked_beam", "use") in sev
    assert not t.uses("blocked_beam") and t.uses("water")
    t2 = rt.ReductionTable.load(t.save(tmp_path / "t.json"))
    assert t2.ignored == t.ignored
    # without water the water background is not asked for
    t.corrections["water"].runs = []
    t.ignored = []
    t.corrections["empty_cell"].runs = []
    assert not any(x.role == "empty_cell" for x in t.missing_inputs())


def test_reduction_with_corrections_switched_off():
    if not os.path.isdir(RAW):
        pytest.skip("Vifor raw data not available")
    import copy

    import numpy as np
    from pysasfit.sasred import absolute as ab
    from pysasfit.sasred import plan as pl, reduction_table as rt
    from pysasfit.sasred.catalogue import beam_centres, scan_folder
    from pysasfit.sasred.core.averaging import q_edges
    from pysasfit.sasred.io.sinq import read_sans_i
    cat = scan_folder(RAW)
    raws = {e.run: e.raw for e in cat if e.raw}
    t = rt.from_plan(pl.build_plan(cat), C8, raws)
    c = beam_centres(list(raws.values()))[(8.0, 6.0, 8.0)]
    read = (lambda n: read_sans_i(os.path.join(RAW, f"sans2026n{n:06d}.hdf"), checksum=False))
    e = q_edges(0.006, 0.09, 30)
    row = t.samples[2]
    _, full = ab.reduce_row(t, row, read, (c.x, c.y), e)
    t2 = copy.deepcopy(t)
    t2.ignored = ["sample_background", "empty_cell"]
    r2, off = ab.reduce_row(t2, t2.samples[2], read, (c.x, c.y), e)
    assert any("sample background: switched off" in n for n in off.notes)
    assert any("water background: switched off" in n for n in off.notes)
    assert not np.allclose(off.I, full.I)
    miss = {(m["role"], m["field"]): m["severity"] for m in r2.parameters["missing"]}
    assert miss[("sample_background", "use")] == "info" and miss[("empty_cell", "use")] == "info"
