"""Absolute scale from the direct beam and an efficiency map (instead of water at long distances)."""
import copy
import os

import numpy as np
import pytest

RAW = os.environ.get("PYSASRED_TEST_RAW", "/mnt/user-data/uploads/Sept2026/raw")
C8 = "SD 8.0 m, λ 6.00 Å, coll 8.0 m"
C2 = "SD 2.0 m, λ 6.00 Å, coll 8.0 m"


def test_scale_method_and_file_roundtrip(tmp_path):
    from pysasfit.sasred import reduction_table as rt
    t = rt.ReductionTable(config="manual")
    assert t.scale_method() == "relative"
    t.corrections["direct_beam"].set_runs([5])
    assert t.scale_method() == "relative"          # no efficiency map yet
    t.efficiency = "map.h5"
    assert t.scale_method() == "direct_beam"
    assert not any("Water" in p for p in t.problems())
    t.corrections["water"].set_runs([4])
    assert t.scale_method() == "direct_beam"        # a map chosen: the water is not used
    assert not any(x.role in ("water", "empty_cell") for x in t.missing_inputs())
    t.efficiency = ""
    assert t.scale_method() == "water"
    t.efficiency = "map.h5"
    t.scale, t.direct_beam_radius_mm, t.direct_beam_attenuator, t.efficiency_mask_filled = "direct_beam", 40.0, 0.002, True
    assert t.scale_method() == "direct_beam"
    t2 = rt.ReductionTable.load(t.save(tmp_path / "t.json"))
    assert (t2.scale, t2.efficiency, t2.direct_beam_radius_mm, t2.direct_beam_attenuator, t2.efficiency_mask_filled) == \
        ("direct_beam", "map.h5", 40.0, 0.002, True)
    old = rt.ReductionTable.from_dict({"config": "old", "corrections": {}})      # tables saved before
    assert old.scale == "auto" and "direct_beam" in old.corrections
    t.scale = "nonsense"
    with pytest.raises(ValueError):
        t.scale_method()


def test_attenuator_of_the_direct_beam():
    from pysasfit.sasred.absolute import direct_beam_attenuator
    from pysasfit.sasred.reduce import load_instrument
    inst = load_instrument("sans_i")
    v, rel, text = direct_beam_attenuator("1", 6.0, inst)
    assert v == pytest.approx(0.00164) and rel == 0.02 and "water calibration" in text
    v, rel, text = direct_beam_attenuator("1", 10.0, inst)            # calibration at 0.6 nm only
    assert v == pytest.approx(0.002061856) and "nominal" in text
    assert direct_beam_attenuator("1", 6.0, inst, 0.0017)[0] == 0.0017
    assert direct_beam_attenuator("0", 6.0, inst)[0] == 1.0


@pytest.fixture(scope="module")
def setup():
    if not os.path.isdir(RAW):
        pytest.skip("Vifor raw data not available")
    from pysasfit.sasred import plan as pl, reduction_table as rt
    from pysasfit.sasred.catalogue import beam_centres, scan_folder
    from pysasfit.sasred.io.sinq import read_sans_i
    from pysasfit.sasred.io.transmission import read_transmission_summary
    cat = scan_folder(RAW)
    raws = {e.run: e.raw for e in cat if e.raw}
    plan = pl.build_plan(cat)
    tr = read_transmission_summary(os.path.join(RAW, "trans.txt"))
    tabs = {c: rt.from_plan(plan, c, raws, tr) for c in (C8, C2)}
    cs = beam_centres(list(raws.values()))
    read = (lambda n: read_sans_i(os.path.join(RAW, f"sans2026n{n:06d}.hdf"), checksum=False))
    return tabs, cs, read


def _eff(t, c, read):
    from pysasfit.sasred.reduce import efficiency_from_runs
    w, ec, cd = (t.corrections[k] for k in ("water", "empty_cell", "blocked_beam"))
    return efficiency_from_runs([read(n) for n in w.used], [read(n) for n in ec.used] or None, (c.x, c.y),
                                (w.transmission, w.transmission_sigma),
                                (ec.transmission, ec.transmission_sigma) if ec.used else None,
                                blocked=[read(n) for n in cd.used] or None)


def test_direct_beam_agrees_with_water(setup, tmp_path):
    """Water at 8 m reduced as a sample: with its own water and with the direct beam of 8 m + the efficiency map
    of the 2 m water. The attenuator of the direct beam comes from the water calibration (2 %)."""
    from pysasfit.sasred import absolute as ab
    from pysasfit.sasred.core.averaging import q_edges
    from pysasfit.sasred.reduction_table import RowSet
    tabs, cs, read = setup
    t8 = tabs[C8]
    c8 = cs[(8.0, 6.0, 8.0)]
    db = t8.corrections["direct_beam"]
    assert db.used == [82173, 82221, 82269] and "attenuator 1" in db.notes      # empty beam of this geometry
    w = t8.corrections["water"]
    row = RowSet("sample", "W", "H2O", runs=w.runs, transmission=w.transmission,
                 transmission_sigma=w.transmission_sigma, scaling=0.2)
    e = q_edges(0.006, 0.09, 30)
    _, rw = ab.reduce_row(t8, row, read, (c8.x, c8.y), e)
    eff2 = _eff(tabs[C2], cs[(2.0, 6.0, 8.0)], read)
    path = eff2.save(tmp_path / "eff2m.h5")
    t = copy.deepcopy(t8)
    t.scale, t.efficiency = "direct_beam", str(path)
    eff = ab.load_efficiency(t.efficiency)
    r2, rd = ab.reduce_row(t, row, read, (c8.x, c8.y), e, efficiency=eff)
    assert r2.absolute and rd.intensity.unit == "1/cm/sr" and r2.parameters["scale"] == "direct_beam"
    ratio = rd.I / rw.I
    assert np.all(np.abs(ratio[1:] - 1) < 0.05), ratio                 # beyond the beam-stop edge of the 2 m map
    assert abs(np.median(ratio) - 1) < 0.025
    assert any(n.startswith("direct beam Φ") for n in rd.notes)
    assert any("water calibration" in n for n in rd.notes)
    assert any("pixels of the efficiency map are estimates" in n for n in rd.notes)
    budget_keys = set(rd.intensity.terms)
    assert any(k.startswith("flood:") for k in budget_keys) and "attenuator_db:1" in budget_keys
    # by-hand attenuator transmission scales Φ inversely
    t.direct_beam_attenuator = 0.00164 * 1.1
    _, rd2 = ab.reduce_row(t, row, read, (c8.x, c8.y), e, efficiency=eff)
    assert np.allclose(rd2.I / rd.I, 1.1, rtol=1e-6)
    # masking the estimated pixels removes the lowest q bins
    t.direct_beam_attenuator, t.efficiency_mask_filled = None, True
    r2m, rdm = ab.reduce_row(t, row, read, (c8.x, c8.y), e, efficiency=eff)
    assert (r2m.mask != 0).sum() > (r2.mask != 0).sum()
    # without a map: a clear error
    with pytest.raises(ValueError, match="efficiency map"):
        ab.reduce_row(t, row, read, (c8.x, c8.y), e)


def test_direct_beam_flux_radius(setup):
    from pysasfit.sasred import absolute as ab
    from pysasfit.sasred.reduce import load_instrument
    tabs, cs, read = setup
    runs = [read(n) for n in tabs[C8].corrections["direct_beam"].used]
    inst = load_instrument(runs[0])
    auto = ab.direct_beam_flux(runs, inst)
    assert 0.99 < auto.fraction <= 1.0 and 20 < auto.radius_mm < 120
    small = ab.direct_beam_flux(runs, inst, radius_mm=10.0)
    assert small.radius_mm == 10.0 and float(small.flux.value) < float(auto.flux.value)


def test_gui_efficiency_map_and_display(setup):
    pytest.importorskip("PySide6")
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6 import QtWidgets
    from pysasfit.sasred.gui import MainWindow
    QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    w = MainWindow()
    w.set_folder(RAW)
    pages = {p.table.config: p for p in w.table_panel.pages()}
    p8 = pages[C8]
    # the 8 m geometry by the direct beam with the map of the 2 m water
    p8.table.scale, p8.table.efficiency = "direct_beam", f"geometry:{C2}"
    p8.refresh()
    assert p8.eff_combo.currentData() == f"geometry:{C2}" and "direct beam" in p8.scale_used.text()
    p8.table.samples = p8.table.samples[:1]
    tag = w.reduce_table_rows(p8.table, p8.table.samples)
    assert tag is not None and w.results[tag].intensity.unit == "1/cm/sr"
    assert w.results[tag].parameters["scale"] == "direct_beam"
    # show the map
    p8._on_show_eff()
    assert w.det_view.currentData() == "efficiency" and w.shown_efficiency is not None
    assert "efficiency" in w.det_canvas.figure.axes[0].get_title()
    # raw image ÷ efficiency ÷ solid angle
    w.det_view.setCurrentIndex(w.det_view.findData("cartesian"))
    w.select_runs([82187])
    raw = w.displayed_raw()
    if True:
        c0, u0, _ = w.display_image(raw, raw.counts.astype(float))
        w.det_eff.setChecked(True)
        w.det_domega.setChecked(True)
        c1, u1, note = w.display_image(raw, raw.counts.astype(float))
        assert u0 == "counts" and u1 == "counts / efficiency / sr" and note == ""
        assert np.nanmedian(c1) > 1e3 * np.nanmedian(c0 + 1)
    w.close()


def test_gui_make_efficiency_menu(setup):
    """'efficiency map → Make from the water of': one entry per geometry with water; without water a clear message."""
    pytest.importorskip("PySide6")
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6 import QtWidgets
    from pysasfit.sasred.gui import MainWindow
    QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    w = MainWindow()
    w._quiet = True                                    # no message boxes in the test
    w.set_folder(RAW)
    m = QtWidgets.QMenu()
    w.fill_make_menu(m)
    texts = [a.text() for a in m.actions()]
    assert len(texts) == 2 and texts[1].startswith(C2)
    m.actions()[1].trigger()
    assert w.shown_efficiency is not None and w.det_view.currentData() == "efficiency"
    assert w.shown_efficiency_name == f"geometry:{C2}"
    # a run of the 3.9 m geometry (no water) shown: no map, the message names the geometries with water
    w.shown_efficiency = None
    w.select_runs([e.run for e in w.entries if e.raw is not None and abs(e.raw.SD_m - 3.9) < 0.2][:1])
    w.log_view.clear()
    assert w.on_make_efficiency() is None and w.shown_efficiency is None
    assert "has no water" in w.log_view.toPlainText() and C2 in w.log_view.toPlainText()
    # nothing selected: the geometry of the reduction-table tab shown
    w.select_runs([])
    assert w.on_make_efficiency() is not None
    w.close()


def test_gui_map_uses_water_background_cd_and_transmissions(setup):
    """The map from a geometry: [(W − Cd)/T_W − (EC − Cd)/T_EC]/ΔΩ; switched-off rows are left out."""
    pytest.importorskip("PySide6")
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    import numpy as np
    from PySide6 import QtWidgets
    from pysasfit.sasred.core.efficiency import efficiency_from_water, water_intensity
    from pysasfit.sasred.core.geometry import run_geometry
    from pysasfit.sasred.reduce import combined_normalised_counts, load_instrument
    QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    from pysasfit.sasred.gui import MainWindow
    w = MainWindow()
    w._quiet = True
    w.set_folder(RAW)
    t = next(p.table for p in w.table_panel.pages() if p.table.config == C2)
    t.corrections["blocked_beam"].set_runs([t.corrections["empty_cell"].used[0]])   # a stand-in "Cd" run
    t.corrections["blocked_beam"].runs[0].use = True
    eff = w.efficiency_from_table(t)
    text = eff.meta["corrections"]
    assert "(W − Cd)/T_W − (EC − Cd)/T_EC" in text and "subtracted from both" in text
    # by hand
    read = w.run_reader("")
    corr = t.corrections
    W = [read(n) for n in corr["water"].used]
    inst = load_instrument(W[0])
    mask, _ = w.mask_for(W[0])
    mask = np.asarray(mask, np.uint32) & ~np.uint32(8)
    Wn, m = combined_normalised_counts(W, inst, [], mask)
    EC, m = combined_normalised_counts([read(n) for n in corr["empty_cell"].used], inst, [], m)
    B, m = combined_normalised_counts([read(n) for n in corr["blocked_beam"].used], inst, [], m)
    geo = run_geometry(W[0], w.centre_for(W[0]))
    I = water_intensity(Wn, EC, corr["water"].transmission, corr["empty_cell"].transmission, geo.d_omega, B)
    ok = eff.measured
    ref = I.value / np.mean(I.value[(m == 0) & np.isfinite(I.value) & (I.value > 0)])
    assert ok.sum() > 1000
    assert np.nanmax(np.abs(eff.value.value[ok] / ref[ok] - 1)) < 0.05      # same up to rejected pixels / mean
    # water background and Cd switched off: left out
    t.ignored = ["empty_cell", "blocked_beam"]
    eff2 = w.efficiency_from_table(t)
    assert "no water background (switched off)" in eff2.meta["corrections"]
    assert "no Cd (switched off)" in eff2.meta["corrections"]
    assert not np.allclose(eff2.value.value, eff.value.value)
    w.close()


def test_gui_choosing_a_map_disables_water(setup):
    """Choosing an efficiency map for a geometry that has water: direct-beam scale, water rows grey and unused."""
    pytest.importorskip("PySide6")
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6 import QtWidgets
    from pysasfit.sasred.gui import MainWindow
    from pysasfit.sasred.reduction_table import CORRECTIONS
    from pysasfit.sasred.table_gui import COLS_CORR, _GREY
    QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    w = MainWindow()
    w._quiet = True
    w.set_folder(RAW)
    p8 = next(p for p in w.table_panel.pages() if p.table.config == C8)
    assert p8.table.scale_method() == "water"
    p8.eff_combo.setCurrentIndex(p8.eff_combo.findData(f"geometry:{C2}"))
    p8._on_eff(0)
    assert p8.table.scale_method() == "direct_beam" and "direct beam" in p8.scale_used.text()
    i = list(CORRECTIONS).index("water")
    it = p8.corr.item(i, COLS_CORR.index("Summed runs"))
    assert it.foreground().color() == _GREY and "not used" in it.toolTip()
    p8.table.samples = p8.table.samples[:1]
    tag = w.reduce_table_rows(p8.table, p8.table.samples)
    r = w.results[tag]
    assert r.parameters["scale"] == "direct_beam"
    assert any("water of this geometry not used" in n for n in r.notes)
    # map removed again: back to water
    p8.eff_combo.setCurrentIndex(p8.eff_combo.findData(""))
    p8._on_eff(0)
    assert p8.table.scale_method() == "water"
    w.close()
