"""SANS-LLB: reader, wide-angle panel geometry and reduction of the banks (synthetic files)."""
import os

import h5py
import numpy as np
import pytest

LAM_NM = 0.5


def _ds(g, name, value, units=None):
    d = g.create_dataset(name, data=np.atleast_1d(value) if not isinstance(value, np.ndarray) else value)
    if units is not None:
        d.attrs["units"] = np.bytes_(units)
    return d


def write_llb(path, sample="Protein", att="0", seed=0, bsy=-7.0, flat=1.0, counts_scale=1.0):
    """A small SANS-LLB file: central 128 x 128 at 4 m, panels at 1.02 m with the NICOS positions."""
    rng = np.random.default_rng(seed)
    with h5py.File(path, "w") as f:
        e = f.create_group("entry0")
        S = e.create_group("SANS-LLB")
        _ds(S, "attenuator/selection", np.bytes_(att))
        _ds(S, "beam_stop/x", 6.0, "mm")
        _ds(S, "beam_stop/y", bsy, "mm")
        _ds(S, "beam_stop/size", 70.0, "mm")
        _ds(S, "velocity_selector/wavelength", LAM_NM, "nm")
        _ds(S, "velocity_selector/wavelength_spread", 11.6, "%")
        _ds(S, "collimator/geometry/size", 8000.0, "mm")
        c = S.create_group("central_detector")
        _ds(c, "distance", 4000.0, "mm")
        _ds(c, "x", 0.0, "mm")
        _ds(c, "x_pixel_size", 5.0, "mm")
        _ds(c, "y_pixel_size", 5.0, "mm")
        _ds(c, "deadtime", 3.5e-6)
        yy, xx = np.mgrid[0:128, 0:128]
        r = np.hypot(xx - 62.5, yy - 63.0)
        img = rng.poisson(counts_scale * flat * (50 + 2000 * np.exp(-r / 15))).astype(np.uint32)
        c.create_dataset("data", data=img)
        cd = e.create_group("central_data")
        cd.create_dataset("posx", data=(63.5 - np.arange(128)) * 5.0)
        cd.create_dataset("posy", data=(np.arange(128) - 63.5) * 5.0)
        for b, shape, posx, posy, px, py in (
                ("left", (136, 16), 363.7 - 13.4 * np.arange(16), 513.5 - 5.0 * np.arange(136), 13.4, 5.0),
                ("bottom", (16, 136), 366.7 - 5.0 * np.arange(136), -342.5 + 13.4 * np.arange(16), 5.0, 13.4)):
            d = S.create_group(f"{b}_detector")
            _ds(d, "distance", 1020.0, "mm")
            _ds(d, "x_pixel_size", px, "mm")
            _ds(d, "y_pixel_size", py, "mm")
            _ds(d, "deadtime", 3.5e-6)
            X, Y = np.meshgrid(posx, posy)
            R = np.sqrt(1020.0 ** 2 + X ** 2 + Y ** 2)
            lam = counts_scale * flat * 2e9 * 1020.0 / R ** 3          # flat scatterer: counts ∝ solid angle
            d.create_dataset("data", data=rng.poisson(lam).astype(np.uint32))
            dd = e.create_group(f"{b}_data")
            dd.create_dataset("posx", data=posx)
            dd.create_dataset("posy", data=posy)
        for k, v in (("monitor0", 2.1e7), ("monitor1", 2.05e7), ("monitor2", 2e7)):
            _ds(e, f"{k}/integral", v, "counts")
        _ds(e, "control/count_time", 600.0)
        _ds(e, "control/preset", 2e7)
        _ds(e, "control/mode", np.bytes_("monitor"))
        _ds(e, "sample/name", np.bytes_(sample))
        _ds(e, "sample/x", 0.0, "mm")
        _ds(e, "sample/thickness", 2.0, "mm")
        _ds(e, "start_time", np.bytes_("2026-08-17 16:19:09.714"))
        _ds(e, "end_time", np.bytes_("2026-08-17 16:29:09.714"))
    return str(path)


@pytest.fixture
def llb(tmp_path):
    return write_llb(tmp_path / "sans-llb2026n005262.hdf")


def test_reader_and_dispatch(llb):
    from pysasfit.sasred.io.raw import instrument_of, read_raw
    from pysasfit.sasred.reduce import load_instrument
    assert instrument_of(llb) == "sans_llb"
    r = read_raw(llb)
    assert r.instrument == "sans_llb" and r.run == 5262 and r.counts.shape == (128, 128)
    assert r.SD_m == pytest.approx(4.0) and r.coll_m == pytest.approx(8.0) and r.wavelength_A == pytest.approx(5.0)
    assert r.dlam_over_lam_fwhm == pytest.approx(0.116) and r.sample_thickness_cm == pytest.approx(0.2)
    assert r.preset_monitor == "monitor2" and not r.aborted and r.beamstop_type == "70 mm"
    assert set(r.panels) == {"left", "bottom"} and r.panels["left"].tube_axis == 0 and r.panels["bottom"].tube_axis == 1
    inst = load_instrument(r)
    assert inst["monitor"]["normaliser"] == "monitor2" and inst["detector"]["deadtime"]["tau_s"] == 3.5e-6
    b = read_raw(llb, bank="left")
    assert b.bank == "left" and b.counts.shape == (136, 16) and b.SD_m == r.SD_m      # config key unchanged


def test_panel_geometry_and_offsets(llb):
    from pysasfit.sasred.core.geometry import run_geometry
    from pysasfit.sasred.io.raw import read_raw
    from pysasfit.sasred.io.sans_llb import beam_position_mm, panel_geometry
    b = read_raw(llb, bank="left")
    assert beam_position_mm(b, (63.5, 63.5)) == pytest.approx((0.0, 0.0))
    assert beam_position_mm(b, (62.5, 64.5)) == pytest.approx((5.0, 5.0))
    g = run_geometry(b, (63.5, 63.5))
    # pixel (row 0, tube 0): posx 363.7, posy 513.5 at 1.020 m
    x, y, L = 0.3637, 0.5135, 1.020
    tt = np.arctan(np.hypot(x, y) / L)
    assert g.q[0, 0] == pytest.approx(4 * np.pi / 5.0 * np.sin(tt / 2), rel=1e-9)
    assert g.d_omega[0, 0] == pytest.approx(0.0134 * 0.005 * L / (L * L + x * x + y * y) ** 1.5, rel=1e-9)
    assert g.x[0, 0] < 0 and g.y[0, 0] > 0           # x of the central pixel_geometry (against posx)
    # offsets: dy shifts along the tubes, dz the distance
    g2 = panel_geometry(b, (63.5, 63.5), offsets_mm=(0.0, -100.0, 50.0))
    assert g2.y[0, 0] == pytest.approx(0.4135) and g2.z[0, 0] == pytest.approx(1.070)


def test_panel_mask_and_reduction(llb, tmp_path):
    from pysasfit.sasred.core import mask as mk
    from pysasfit.sasred.core.averaging import q_edges
    from pysasfit.sasred.io.raw import read_raw
    from pysasfit.sasred.reduce import reduce_run_1d
    b = read_raw(llb, bank="bottom")
    m, info = mk.default_mask(b, (62.5, 63.0))
    assert (m[:, :10] != 0).all() and (m[:, -10:] != 0).all() and (m[:, 10:-10] == 0).all()
    r = reduce_run_1d(b, (63.5, 63.5), q_edges(0.17, 0.5, 15, "lin"), mask=m)
    # flat scatterer: per steradian constant (counts ∝ solid angle by construction)
    v = r.I[np.isfinite(r.I)]
    assert np.std(v) / np.mean(v) < 0.02
    assert any("bottom detector" in n for n in r.notes)


def test_row_reduction_banks_and_merge_label(tmp_path):
    """Sample = water (flat) on all banks: absolute 2D per pixel works on the panels as on the central detector."""
    from pysasfit.sasred import absolute as ab
    from pysasfit.sasred.core.averaging import q_edges
    from pysasfit.sasred.io.raw import read_raw
    from pysasfit.sasred.reduction_table import ReductionTable, RowSet, RunChoice
    from pysasfit.toolbox.merge import curve_from_reduced
    paths = {5260: write_llb(tmp_path / "sans-llb2026n005260.hdf", "EC", seed=1, flat=0.2),
             5261: write_llb(tmp_path / "sans-llb2026n005261.hdf", "H2O", seed=2, flat=1.0),
             5262: write_llb(tmp_path / "sans-llb2026n005262.hdf", "S", seed=3, flat=1.0)}
    t = ReductionTable(config="SD 4.0 m, λ 5.00 Å, coll 8.0 m")
    t.corrections["water"].runs = [RunChoice(5261)]
    t.corrections["water"].transmission = 0.5
    t.corrections["empty_cell"].runs = [RunChoice(5260)]
    t.corrections["empty_cell"].transmission = 0.9
    row = RowSet("sample", "No. 1", "S", runs=[RunChoice(5262)], transmission=0.5, scaling=0.2)
    out = {}
    for bank in ("", "left", "bottom"):
        r2, r1 = ab.reduce_row(t, row, lambda n, b=bank: read_raw(paths[n], checksum=False, bank=b), (62.5, 63.0),
                               lambda r2: q_edges(float(r2.geometry.q[r2.mask == 0].min()),
                                                  float(r2.geometry.q[r2.mask == 0].max()), 10))
        out[bank] = r1
        assert r2.absolute and r1.intensity.unit == "1/cm/sr"
    # sample = water: the same absolute level on every bank
    lv = [np.nanmedian(out[b].I) for b in out]
    assert max(lv) / min(lv) < 1.03
    c = curve_from_reduced(out["left"])
    assert "left panel 1.02 m" in c.geometry and "left" not in curve_from_reduced(out[""]).geometry


def test_catalogue_and_sans_i_untouched(tmp_path):
    from pysasfit.sasred.catalogue import COLUMNS, scan_folder
    for n, name, bsy in ((5274, "EB", -90.0), (5262, "S", -7.0)):
        write_llb(tmp_path / f"sans-llb2026n{n:06d}.hdf", name, seed=n, bsy=bsy)
    cat = scan_folder(str(tmp_path))
    assert [e.run for e in cat] == [5262, 5274] and all(e.raw.instrument == "sans_llb" for e in cat)
    row = cat[0].row()
    assert row[COLUMNS.index("monitor1")] == "2e+07"                          # the preset monitor (monitor2)


MAREN = os.environ.get("PYSASRED_LLB_RAW", "/mnt/user-data/uploads/Maren")


@pytest.mark.skipif(not os.path.isdir(MAREN), reason="SANS-LLB data not available")
def test_real_file_positions_match_nicos():
    from pysasfit.sasred.core.geometry import run_geometry
    from pysasfit.sasred.io.raw import read_raw
    p = os.path.join(MAREN, "sans-llb2026n005262.hdf")
    with h5py.File(p) as f:
        Qx, Qy = (f["entry0/left_data/Qx"][()], f["entry0/left_data/Qy"][()])
    g = run_geometry(read_raw(p, checksum=False, bank="left"), (63.5, 63.5))
    # along the horizontal through the beam (Qy ≈ 0) NICOS' Qx is the full q
    j = int(np.argmin(np.abs(Qy)))
    assert np.allclose(g.q[j] * 10, np.abs(Qx), rtol=0.01)


def test_name_and_role_corrections(tmp_path, monkeypatch):
    """Sample names / roles corrected by hand: kept per folder, used by the plan and the reduction table."""
    from pysasfit.sasred import names as nm
    from pysasfit.sasred import plan as pl
    from pysasfit.sasred import reduction_table as rt
    from pysasfit.sasred.catalogue import scan_folder
    monkeypatch.setattr(nm, "store_path", lambda folder: str(tmp_path / "names.json"))
    d = tmp_path / "raw"
    d.mkdir()
    for n, name in ((5260, "EC"), (5261, "H2O"), (5262, "Protein"), (5280, "real H2O, the other one is D2O")):
        write_llb(d / f"sans-llb2026n{n:06d}.hdf", name, seed=n)
    c = nm.Corrections(str(d))
    c.set([5261], "D2O", "sample", "cells swapped")
    c.set([5280], "H2O", "water")
    nm.save(c)
    c2 = nm.load(str(d))
    assert c2[5261]["name"] == "D2O" and c2[5280]["role"] == "water"
    cat = scan_folder(str(d))
    for e in cat:
        nm.apply(e.raw, c2)
    by = {e.run: e.raw for e in cat}
    assert by[5261].sample_name == "D2O" and by[5261].sample_name_in_file == "H2O" and by[5262].role == ""
    p = pl.build_plan(cat)
    t = rt.from_plan(p, rt.configurations(p)[0], {e.run: e.raw for e in cat})
    assert [x.run for x in t.corrections["water"].runs] == [5280]
    assert {s.info for s in t.samples} == {"D2O", "Protein"}
    c2.reset([5261])
    nm.apply(by[5261], c2)
    assert by[5261].sample_name == "H2O" and by[5261].role == ""


def test_gui_panel_masks(tmp_path, monkeypatch):
    """Each SANS-LLB bank has its own editable mask; the reduction of a bank uses it."""
    pytest.importorskip("PySide6")
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    monkeypatch.setenv("PYSASFIT_SETTINGS", str(tmp_path / "settings.json"))
    from PySide6 import QtWidgets
    from pysasfit.sasred.core import mask as mk
    from pysasfit.sasred.gui import MainWindow
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    d = tmp_path / "raw"
    d.mkdir()
    write_llb(d / "sans-llb2026n005262.hdf", "S", seed=1)
    w = MainWindow()
    try:
        w.set_folder(str(d))
        w.select_runs([5262])
        app.processEvents()
        ed = w.mask_editor
        w.det_bank.setCurrentIndex(w.det_bank.findData("left"))
        app.processEvents()
        assert ed.mask.shape == (136, 16) and int(np.count_nonzero(ed.mask & mk.BORDER)) == 2 * 10 * 16
        ed.apply(mk.pixels_in_rectangle(ed.mask.shape, 0, 15, 50, 59), True)
        w.det_bank.setCurrentIndex(0)
        app.processEvents()
        assert ed.mask.shape == (128, 128)
        assert any(k[3].endswith("left panel") for k in w.masks)
        w.bank_combo.setCurrentIndex(w.bank_combo.findData("left"))
        w.live.setChecked(False)
        w.on_reduce()
        r = w.results["5262 left"]
        assert int((r.flags != 0).sum()) == 2 * 10 * 16 + 16 * 10
    finally:
        w.close()
