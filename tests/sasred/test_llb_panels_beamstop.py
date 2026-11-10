"""SANS-LLB: the distance of the wide-angle panels belongs to the instrument configuration; transmission
region of a semi-transparent beam stop (second mask layer: pixels used, not left out)."""
import os
import sys

import h5py
import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(__file__))
from test_sans_llb import write_llb  # noqa: E402


def _panels_at(path, mm):
    with h5py.File(path, "r+") as f:
        for b in ("left", "bottom"):
            f[f"entry0/SANS-LLB/{b}_detector/distance"][...] = mm
    return path


def test_panel_distance_in_configuration(tmp_path):
    from pysasfit.sasred.catalogue import legacy_mask_key, mask_key, scan_folder, COLUMNS
    from pysasfit.sasred.io.raw import read_raw
    from pysasfit.sasred.plan import _config, build_plan, config_base
    from pysasfit.sasred.reduction_table import configurations
    a = _panels_at(write_llb(tmp_path / "sans-llb2026n005262.hdf", "S", seed=1), 1020.0)
    b = _panels_at(write_llb(tmp_path / "sans-llb2026n005263.hdf", "S", seed=2), 2500.0)
    ra, rb = read_raw(a, checksum=False), read_raw(b, checksum=False)
    assert ra.panel_SD_m == pytest.approx(1.02) and rb.panel_SD_m == pytest.approx(2.5)
    assert _config(ra) != _config(rb) and config_base(_config(ra)) == config_base(_config(rb))
    assert _config(ra).endswith("panels 1.02 m")
    assert mask_key(ra) != mask_key(rb) and legacy_mask_key(mask_key(ra)) == legacy_mask_key(mask_key(rb))
    plan = build_plan([ra, rb])
    assert len(configurations(plan)) == 2                    # same central distance, two panel distances
    es = scan_folder(str(tmp_path))
    row = es[0].row()
    assert row[COLUMNS.index("SD [m] (panels)")] == "4.00 (1.02)"


def test_panels_at_different_distances_flagged(tmp_path):
    from pysasfit.sasred.io.raw import read_raw
    p = write_llb(tmp_path / "sans-llb2026n005262.hdf", "S", seed=1)
    with h5py.File(p, "r+") as f:
        f["entry0/SANS-LLB/bottom_detector/distance"][...] = 1100.0
    r = read_raw(p, checksum=False)
    assert r.panel_SD_spread_m == pytest.approx(0.08)


def test_region_intensity_and_transmission(tmp_path):
    from pysasfit.sasred.core.beamstop_transmission import beamstop_transmission, compare, region_intensity
    from pysasfit.sasred.io.raw import read_raw
    eb = read_raw(write_llb(tmp_path / "sans-llb2026n005270.hdf", "EB", seed=1, counts_scale=1.0), checksum=False)
    s = read_raw(write_llb(tmp_path / "sans-llb2026n005271.hdf", "S", seed=2, counts_scale=0.5), checksum=False)
    reg = np.zeros(eb.counts.shape, bool)
    reg[58:68, 58:68] = True
    i = region_intensity([eb], reg)
    assert i.n_pixels == 100 and i.counts == pytest.approx(float(eb.counts[reg].sum()))
    t = beamstop_transmission([s], [eb], reg)
    assert t.value == pytest.approx(0.5, rel=0.02) and 0 < t.sigma < 0.01
    two = region_intensity([eb, eb], reg)                    # the same run twice: the same intensity
    assert two.I == pytest.approx(i.I)
    ratio, s_ratio, z = compare(t.value, t.sigma, 0.5, 0.0)
    assert ratio == pytest.approx(1.0, abs=0.02) and abs(z) < 4
    with pytest.raises(ValueError):
        region_intensity([eb], np.zeros_like(reg))


def test_gui_region_layer_and_check(tmp_path, monkeypatch):
    pytest.importorskip("PySide6")
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    monkeypatch.setenv("PYSASFIT_SETTINGS", str(tmp_path / "settings.json"))
    from PySide6 import QtWidgets
    from pysasfit.sasred.core import mask as mk
    from pysasfit.sasred.gui import MainWindow
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    d = tmp_path / "raw"
    d.mkdir()
    write_llb(d / "sans-llb2026n005270.hdf", "EB", seed=1, counts_scale=1.0)
    write_llb(d / "sans-llb2026n005271.hdf", "EC", seed=2, counts_scale=0.9)
    write_llb(d / "sans-llb2026n005272.hdf", "H2O", seed=3, counts_scale=0.5)
    write_llb(d / "sans-llb2026n005273.hdf", "S", seed=4, counts_scale=0.6)
    w = MainWindow()
    try:
        w.set_folder(str(d))
        w.select_runs([5273])
        app.processEvents()
        ed = w.mask_editor
        before = ed.mask.copy()
        ed.layer.setCurrentIndex(ed.layer.findData("region"))
        ed.apply(mk.pixels_in_rectangle(ed.mask.shape, 58, 67, 58, 67), True)
        assert np.array_equal(ed.mask, before)                # the region does not mask anything
        assert ed.region.sum() == 100 and "transmission region 100 px" in ed.stats.text()
        raw = w.displayed_raw()
        assert w.region_for(raw).sum() == 100
        ed.apply(mk.pixels_in_rectangle(ed.mask.shape, 58, 59, 58, 67), False)
        assert ed.region.sum() == 80
        ed.undo()
        assert ed.region.sum() == 100 and np.array_equal(ed.mask, before)
        ed.reset()                                            # on the region layer: clears the region only
        assert ed.region is None and np.array_equal(ed.mask, before)
        ed.undo()
        assert ed.region.sum() == 100
        ed.suggest_region()
        assert ed.region is not None and ed.region[63, 62]
        ed.apply(mk.pixels_in_rectangle(ed.mask.shape, 58, 67, 58, 67), True)
        # masks file keeps the region
        p = w.save_masks(str(tmp_path / "masks.h5"))
        w.tregions.clear()
        w.mask_editor.region = None
        assert w.load_masks(p, replace=True) >= 1 and w.region_for(raw).sum() >= 100
        # check dialog against the empty beam
        w.table_panel.on_fill()
        dlg = w.on_beamstop_transmission()
        t = w.table_panel.page.table
        rows = t.rows()
        ref = next(i for i, r in enumerate(rows) if r.used and 5270 in r.used) if any(
            5270 in r.used for r in rows) else next(i for i, r in enumerate(rows) if r.used)
        dlg.ref.setCurrentIndex(dlg.ref.findData(ref))
        dlg.calculate()
        assert dlg.results and all("T" in r for r in dlg.results if not r["note"].startswith(("ValueError", "KeyError")))
        got = {r["runs"]: r for r in dlg.results}
        for runs, r in got.items():
            if "5273" in runs and "T" in r:
                assert r["T"] == pytest.approx(rows[ref].used and 0.6 / {5270: 1.0, 5271: 0.9, 5272: 0.5,
                                                                          5273: 0.6}[rows[ref].used[0]], rel=0.03)
        assert dlg.text().startswith("row\tsample")
        assert os.path.isfile(dlg.save(str(tmp_path / "bs.csv")))
        dlg.close()
    finally:
        w.close()
