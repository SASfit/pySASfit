"""Correction matrix and sample table (BerSANS-like), filled from the plan or by hand."""
import os

import pytest

from pysasfit.sasred import reduction_table as rt
from pysasfit.sasred.plan import Entry, Plan

RAW = os.environ.get("PYSASRED_TEST_RAW", "/mnt/user-data/uploads/Sept2026/raw")
need = pytest.mark.skipif(not os.path.isdir(RAW), reason="Vifor raw data not available")
CFG = "L2 8.0 m, λ 6.00 Å, L1 8.0 m"


def _plan():
    E = lambda role, name, kind, runs, summed, outl, verdict, notes=None, cfg=CFG: Entry(
        cfg, role, name, kind, runs, summed, outl, verdict, run_notes=notes or {})
    return Plan([
        E("water", "H2O", "scattering", [1, 2, 3], [2, 3], [1], "outlier", {"1": "outlier: χ²/ndf 8"}),
        E("empty_cell", "EC", "scattering", [4, 5], [4, 5], [], "equivalent"),
        E("sample", "S1", "scattering", [6, 7], [], [], "shape drift"),
        E("sample", "S1", "transmission", [8, 9], [8, 9], [], "equivalent"),
        E("sample", "S2", "scattering", [10], [10], [], "single"),
        E("sample", "S2", "scattering", [11], [11], [], "single", cfg="L2 2.0 m, λ 6.00 Å, L1 8.0 m"),
    ], [])


def test_fill_from_plan():
    from pysasfit.sasred.io.transmission import Transmission
    p = _plan()
    assert rt.configurations(p) == [CFG, "L2 2.0 m, λ 6.00 Å, L1 8.0 m"]
    t = rt.from_plan(p, CFG, transmissions=[Transmission("D1", 8, "S1", 0.42, 0.001),
                                            Transmission("D2", 3, "H2O", 0.24, 0.001)])
    w = t.corrections["water"]
    assert w.used == [2, 3] and w.number == "2 of 3" and w.runs[0].comment.startswith("outlier")
    assert w.transmission == 0.24 and "trans.txt" in w.transmission_source
    assert t.corrections["sample_background"].used == [4, 5]           # copied from the empty cell
    assert "Water-BG" in t.corrections["sample_background"].notes
    s1, s2 = t.samples
    assert s1.label == "No. 1" and s1.used == [6, 7] and s1.notes.startswith("decision needed: shape drift")
    assert [c.run for c in s1.trans_runs] == [8, 9] and s1.transmission == 0.42
    assert s2.used == [10] and s2.transmission is None
    probs = t.problems()
    assert any("Cadmium" in x for x in probs) and any("No. 2 (S2): no transmission" in x for x in probs)


def test_manual_table_and_file(tmp_path):
    t = rt.ReductionTable(config="manual")
    assert [r.label for r in t.rows()] == ["Cadmium", "Water", "Water-BG", "Sample-BG"]
    t.corrections["water"].set_runs(rt.parse_runs("82187, 82190-82192"))
    assert t.corrections["water"].used == [82187, 82190, 82191, 82192]
    assert t.corrections["water"].runs_text() == "82187, 82190–82192"
    for k in range(3):
        r = t.add_sample(f"S{k}")
        r.set_runs([100 + k])
    t.samples[0].transmission = 0.5
    t.copy_value(0, "transmission", "below")
    assert [r.transmission for r in t.samples] == [0.5, 0.5, 0.5]
    t.remove_sample(1)
    assert [r.label for r in t.samples] == ["No. 1", "No. 2"] and t.samples[1].info == "S2"
    t.samples[1].runs[0].use = False
    assert any("all runs switched off" in x for x in t.problems())
    t2 = rt.ReductionTable.load(t.save(tmp_path / "table.json"))
    assert t2.to_dict() == t.to_dict()
    assert rt.parse_runs("5-3 7") == [3, 4, 5, 7]


@need
def test_gui_table_panel(tmp_path):
    pytest.importorskip("PySide6")
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6 import QtCore, QtWidgets
    from pysasfit.sasred.gui import MainWindow
    from pysasfit.sasred.table_gui import RunSetDialog
    QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    w = MainWindow()
    w.set_folder(RAW)                             # filled automatically (preference, default on)
    p = w.table_panel
    titles = [p.tabs.tabText(i) for i in range(p.tabs.count())]
    assert len(titles) == 3 and titles[0].startswith("L2 8.0 m") and titles[1].startswith("L2 2.0 m")
    assert titles[2].startswith("L2 3.9 m") and titles[2].endswith("?") and "(1 run)" in titles[2]
    t = p.table
    assert t.corrections["water"].info == "H2O" and len(t.samples) >= 30
    assert p.corr.cellWidget(1, 0).text() == "Water" and p.samp.rowCount() == len(t.samples)
    assert 82187 not in t.corrections["water"].used                    # outlier off (check on)
    water = t.corrections["water"]
    d = RunSetDialog(water, p.raws(), p)
    assert d.scatt.rowCount() == len(water.runs)
    d.scatt.item(3, 0).setCheckState(QtCore.Qt.CheckState.Unchecked)
    d.add_edit.setText("82189")
    d.on_add()
    d.apply()
    assert water.runs[3].use is False and 82189 in water.used
    item = p.samp.item(0, 4)
    item.setText("0.5")
    assert t.samples[0].transmission == 0.5 and t.samples[0].transmission_source == "by hand"
    p.samp.setCurrentCell(0, 4)
    p.on_copy("below")
    assert all(r.transmission == 0.5 for r in t.samples)
    pg = p.page                                                           # water box: g(λ) and scaling
    assert pg.g_auto.text().startswith("1.28") and "0.600 nm" in pg.g_auto.text()
    assert pg.w_den.text().startswith("T_W") and float(pg.w_scale.text().split()[0]) > 0
    pg.g_edit.setText("1.3")
    pg._on_g_edit()
    assert t.water_g == 1.3 and type(t).from_dict(t.to_dict()).water_g == 1.3
    pg.g_edit.setText("")
    pg._on_g_edit()
    assert t.water_g is None
    # the geometry of the tab is preset in Add many samples
    pre = dict(p.page.presets())
    assert pre["L2_m"][0] < 8.0 < pre["L2_m"][1] and pre["kind"] == ("scattering",)
    # a test geometry removed completely, also when filling again; restored on demand
    p.remove_geometry(2, confirm=False)
    assert p.tabs.count() == 2 and p.excluded == ["L2 3.9 m, λ 6.00 Å, L1 8.0 m"]
    p.check.setChecked(False)
    p.on_fill()
    assert p.tabs.count() == 2
    f = p.save(tmp_path / "tables.json")
    p.on_restore()
    assert p.tabs.count() == 3 and p.excluded == []
    p.load(f)
    assert p.tabs.count() == 2 and len(p.excluded) == 1
    p.on_clear()
    assert p.tabs.count() == 3 and p.table.samples == [] and p.table.config.startswith("by hand")
    w.close()


@need
def test_beam_centre_per_geometry():
    from pysasfit.sasred import catalogue as cg
    cat = cg.scan_folder(RAW)
    c = cg.beam_centres([e.raw for e in cat])
    c8, c2 = c[(8.0, 6.0, 8.0)], c[(2.0, 6.0, 8.0)]
    assert c8.source == "direct beam" and len(c8.runs) > 50
    assert c8.x == pytest.approx(61.73, abs=0.05) and c8.y == pytest.approx(63.96, abs=0.05)
    assert c2.source.startswith("borrowed") and c2.runs == []          # no direct beam at 2 m


@need
def test_gui_keeps_a_centre_per_geometry():
    pytest.importorskip("PySide6")
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6 import QtWidgets
    from pysasfit.sasred.gui import MainWindow
    QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    w = MainWindow()
    w.set_folder(RAW)
    w.select_runs([82203])                       # 2 m water
    assert "borrowed" in w.bc_label.text()
    w.xc.setValue(62.2)                          # set by hand for 2 m
    w.select_runs([82280])                       # 8 m water: its own centre comes back
    assert w.xc.value() == pytest.approx(61.73, abs=0.05)
    raw2 = [e.raw for e in w.entries if e.run == 82203][0]
    assert w.centre_for(raw2)[0] == pytest.approx(62.2)
    w.select_runs([82203, 82280])
    assert len({w.centre_for(e.raw) for e in w.selected_entries()}) == 2
    w.close()


def test_aperture_saved_and_old_files(tmp_path):
    from pysasfit.sasred.reduction_table import ReductionTable
    t = ReductionTable("cfg")
    t.corrections["water"].aperture_cm2 = 1.0
    d = t.to_dict()
    t2 = ReductionTable.from_dict(d)
    assert t2.corrections["water"].aperture_cm2 == 1.0
    del d["corrections"]["water"]["aperture_cm2"]               # file written before the aperture column
    d["corrections"]["water"]["unknown_future_key"] = 3
    assert ReductionTable.from_dict(d).corrections["water"].aperture_cm2 is None
