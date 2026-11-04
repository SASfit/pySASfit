"""Merging reduced curves of several geometries, grouped by sample name and header items."""
import json
import os

import numpy as np
import pytest

from pysasfit.toolbox import merge as mg

BERSANS = "/mnt/user-data/uploads/Vifor/1stJuly2024/analysed"


def model(q):
    return 10.0 / (1 + (5 * q) ** 2) + 0.1


def bersans_file(path, sample, sd, temp, q, scale=1.0, noise=0.01, seed=0):
    rng = np.random.default_rng(seed)
    I = model(q) * scale
    dI = noise * I
    I = I + rng.normal(0, 1, len(q)) * dI
    lines = ["%File", f"FileName={os.path.basename(path)}", "Type=SANSDIso", "ContentX=Q", "%Sample",
             f"SampleName={sample}", f"Temperature={temp}", "%Setup", "Lambda=0.600", "Collimation=8",
             f"SD={sd:.3f}", "%History", "Sample-BG=EC", "%Counts"]
    lines += [f" {a:.4e}, {b:.4e}, {c:.4e},  0.000e+00" for a, b, c in zip(q, I, dI)]
    with open(path, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    return path


@pytest.fixture
def folder(tmp_path):
    q8, q2 = np.linspace(0.05, 0.8, 60), np.linspace(0.3, 3.0, 60)
    k = 0
    for s in ("A", "B"):
        for rep in (1, 2):
            for T in (20.0, 20.2, 40.0):
                if rep == 2 and T != 20.0:
                    continue
                k += 1
                name = f"EXP-1-{s}_{rep}"
                bersans_file(tmp_path / f"D{k:07d}.080", name, 8.0, T, q8, seed=k)
                k += 1
                bersans_file(tmp_path / f"D{k:07d}.020", name, 2.0, T, q2, seed=k,
                             scale=1.03 if s == "B" else 1.0)
    return tmp_path


def test_read_group_merge(folder):
    cs, bad = mg.read_folder(str(folder))
    assert not bad and len(cs) == 16 and cs[0].fmt == "bersans"
    c = cs[0]
    assert c.meta["Sample/Temperature"] == 20.0 and c.geometry.startswith("SD 8.00 m, λ 0.600 nm")
    # sample letter only; temperature within ± 0.5 K: 20.0 and 20.2 together, 40 apart
    groups, rest = mg.group_curves(cs, r"EXP-1-(\w)_\d", {"Sample/Temperature": 0.5})
    assert not rest and [g.label for g in groups] == ["A, Temperature 20–20.2", "A, Temperature 40",
                                                      "B, Temperature 20–20.2", "B, Temperature 40"]
    assert len(groups[0].curves) == 6 and "3 curves of geometry" in groups[0].notes[0]
    # exact temperatures and repeat number in the key
    g2, _ = mg.group_curves(cs, r"EXP-1-(\w)_(\d)", {"Sample/Temperature": None})
    assert len(g2) == 8 and g2[0].key == "A_1"
    # default: keep every point with its own dI and dq, geometry of each point in source
    ma0 = mg.merge(g2[0].curves, group=g2[0])
    assert ma0.method == "all" and len(ma0.q) == sum(len(c.q) for c in g2[0].curves)
    assert np.all(np.diff(ma0.q) >= 0) and set(np.unique(ma0.source)) == {0, 1}
    m = mg.merge(g2[0].curves, group=g2[0], method="weighted")
    assert len(m.overlaps) == 1 and abs(m.overlaps[0].factor - 1) < 0.01
    assert np.all(np.diff(m.q) > 0) and (m.source == -1).any()
    assert np.allclose(m.I, model(m.q), rtol=0.05)
    # B is 3 % too high at 2 m: seen in the overlap, removed with scale_to
    gb = [g for g in g2 if g.key == "B_1"][0]
    mb = mg.merge(gb.curves, group=gb, method="weighted")
    assert mb.overlaps[0].factor == pytest.approx(1.03, abs=0.01) and mb.overlaps[0].z > 3
    ms = mg.merge(gb.curves, group=gb, scale_to=gb.curves[0].geometry, method="weighted")
    assert ms.scales[gb.curves[1].geometry] == pytest.approx(1 / 1.03, abs=0.01)
    assert np.allclose(ms.I, model(ms.q), rtol=0.05)
    # scaling: one factor per geometry, the same for all groups (A is right, B 3 % too high at 2 m)
    g8, g2m = gb.curves[0].geometry, gb.curves[1].geometry
    cs = mg.common_scales(g2, g8, method="weighted")
    f, sf, n = cs[g2m]
    assert n == 8 and 0.97 < f < 1.0 and sf > 0
    mall = mg.merge_groups(g2, scale_to=g8, method="all")
    assert all(m.scales[g2m] == f for m in mall if g2m in m.scales)
    assert all(m.notes[0].startswith("one scale factor") for m in mall)
    own = mg.merge_groups(g2, scale_to=g8, common=False)
    assert len({round(m.scales[g2m], 6) for m in own}) > 1
    # keep all points, cuts and dropped points
    g8 = gb.curves[0].geometry
    ma = mg.merge(gb.curves, method="all", cuts={g8: (None, 0.6)}, drop={gb.curves[1].geometry: (2, 3)})
    assert len(ma.q) == np.sum(gb.curves[0].q <= 0.6) + 60 - 5 and ma.q.max() < 3.0


def test_write_read_series(folder, tmp_path_factory):
    out = tmp_path_factory.mktemp("out")
    cs, _ = mg.read_folder(str(folder))
    groups, _ = mg.group_curves(cs, r"EXP-1-(\w)_1", {"Sample/Temperature": 0.5})
    ms = mg.merge_groups(groups)
    a = mg.write_ascii(out / (mg.file_stem(ms[0]) + ".dat"), ms[0])
    b = mg.write_nxcansas(out / "one.h5", ms[0], units="A")
    allf = mg.write_nxcansas(out / "all.h5", ms)
    ra, rb, rall = mg.read_curves(a)[0], mg.read_curves(b)[0], mg.read_curves(allf)
    assert ra.fmt == "merged" and ra.sample == "A" and ra.meta["Sample/Temperature"] == pytest.approx(20.1)
    assert np.loadtxt(a).shape[1] == 5 and "geometry index: 0 = " in open(a).read()
    assert np.allclose(ra.q, ms[0].q, rtol=1e-6) and np.allclose(ra.dq, ms[0].dq, rtol=1e-6) and np.allclose(rb.q, ms[0].q) and np.allclose(rb.I, ms[0].I)
    assert len(rall) == 4 and rall[1].meta["Sample/Temperature"] == 40.0
    # series: both temperatures of A, ordered
    s = mg.series(ms, "Sample/Temperature")
    assert [v for v, _ in s["A"]] == [pytest.approx(20.1), 40.0]
    # the merged curves themselves can be grouped again (e.g. a series from several merged files)
    g, _ = mg.group_curves(rall, None, {"Sample/Temperature": 1})
    assert len(g) == 4


def test_pysasred_files(tmp_path):
    """pySASred writers carry the header items; the reader picks them up (synthetic reduced run)."""
    from types import SimpleNamespace
    from pysasfit.sasred.data.uarray import Source, UArray
    from pysasfit.sasred.io.ascii import write_ascii_1d
    from pysasfit.sasred.io.nxcansas import write_nxcansas_1d
    q = np.linspace(0.01, 0.1, 20)                              # 1/Å
    v = model(q * 10)
    I = UArray.from_source(v, Source("counting:42", (0.01 * v) ** 2, "counting", False), "1/cm/sr")
    raw = SimpleNamespace(run=42, sample_name="EXP-9-C_1", path=str(tmp_path / "none.hdf"), sha256="",
                          wavelength_A=6.0, SD_m=8.0, coll_m=8.0, dlam_over_lam_fwhm=0.1, attenuator="0",
                          detector_offset_m=(0, 0), detector_tilt_deg=0.0, beamstop_xy_mm=(0, 0),
                          sample_position_mm=0.0, sample_thickness_cm=0.1, counting_time_s=10.0,
                          start_time="2026-09-30T01:00:00")
    red = SimpleNamespace(run=42, raw=raw, q=q, q_sigma=q * 0.05, intensity=I, I=I.value,
                          idev=lambda: I.std(systematic=False), systematic_std=lambda: np.zeros_like(q),
                          parameters={"sample": "EXP-9-C_1"}, notes=["test"])
    write_nxcansas_1d(tmp_path / "r.h5", red)
    write_ascii_1d(tmp_path / "r.dat", red)
    for p in ("r.h5", "r.dat"):
        c = mg.read_curves(tmp_path / p)[0]
        assert c.sample == "EXP-9-C_1" and c.meta["SD_m"] == 8.0 and c.meta["wavelength_A"] == 6.0
        assert np.allclose(c.q, q * 10) and c.geometry == "SD 8.00 m, λ 0.600 nm, coll 8.0 m"


@pytest.mark.skipif(not os.path.isfile(os.path.join(BERSANS, "D0007625.021")), reason="BerSANS files not available")
def test_real_bersans_merge():
    a = mg.read_curves(os.path.join(BERSANS, "D0007652.080"))[0]
    b = mg.read_curves(os.path.join(BERSANS, "D0007625.020"))[0]
    ref = mg.read_curves(os.path.join(BERSANS, "D0007625.021"))[0]
    assert a.sample == b.sample == "EXP-001763_1" and len(ref.q) == len(a.q) + len(b.q)
    m = mg.merge([a, b], method="all")
    assert np.allclose(np.sort(m.q), np.sort(ref.q))                # BerSANS keeps all points too
    assert len(mg.merge([a, b]).overlaps) == 1


def test_gui(folder, tmp_path_factory):
    pytest.importorskip("PySide6")
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6 import QtWidgets
    QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    from pysasfit.toolbox.gui import MainWindow
    w = MainWindow()
    t = w.merger
    t.add_folder(str(folder))
    assert t.files.rowCount() == 16 and t.cuts.rowCount() == 2 and t.scale.count() == 3
    t.regex.setText(r"EXP-1-(\w)_1")
    t.add_item("Sample/Temperature", "0.5")
    t.on_merge()
    assert t.table.rowCount() == 4 and "Temperature 20" in t.table.item(0, 0).text()
    assert "+3.0 %" in t.table.item(2, 4).text() or "+2.9 %" in t.table.item(2, 4).text() \
        or "+3.1 %" in t.table.item(2, 4).text()
    assert t.table.item(0, 1).text() == "EXP-1-A_1"
    t.table.selectRow(0)
    assert t.sources.rowCount() == 4 and t.sources.item(0, 1).text() == "EXP-1-A_1"
    t.mode.setCurrentIndex(1)
    t.series_key.setCurrentText("Sample/Temperature")
    t.replot()
    assert t.canvas.figure.axes and len(t.canvas.figure.axes[0].lines) >= 2
    out = tmp_path_factory.mktemp("gui")
    t.out_all.setChecked(True)
    files = t.on_save(selected=False, folder=str(out))
    assert len(files) == 9 and os.path.isfile(out / "merged_all.h5")
    st = t.state()
    assert st["regex"] == r"EXP-1-(\w)_1" and st["items"] == [("Sample/Temperature", "0.5")]
    json.dumps(st)
    w.close()


RAW = os.environ.get("PYSASRED_TEST_RAW", "/mnt/user-data/uploads/Sept2026/raw")


@pytest.mark.skipif(not os.path.isdir(RAW), reason="Vifor raw data not available")
def test_from_sasred_in_memory():
    """Table rows reduced in pySASred reach the merge tool without files, both directions."""
    pytest.importorskip("PySide6")
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6 import QtWidgets
    QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    from pysasfit.sasred.gui import MainWindow as Red
    from pysasfit.toolbox import merge_gui
    from pysasfit.toolbox.gui import MainWindow as Box
    red = Red()
    red.set_folder(RAW)
    p = red.table_panel
    for page in p.pages()[:2]:                                   # 8 m and 2 m, first two samples
        red.reduce_table_rows(page.table, page.table.samples[:2])
    assert len(red.results) == 4
    box = Box()
    t = box.merger
    t.on_from_sasred()                                           # every open pySASred window
    assert len(t.curves) >= 4 and all(c.fmt == "pySASred (memory)" for c in t.curves)
    t.on_clear()
    t.add_reduced(red.results)
    assert len(t.curves) == 4
    assert t.curves[0].sample.startswith("EXP-005370-A") and "hdf:sample/thickness" in t.curves[0].meta
    t.add_reduced(red.results)                                   # same keys: replaced, not doubled
    assert len(t.curves) == 4
    t.regex.setText(r"EXP-005370-(\w_\d)")
    t.on_merge()
    assert t.table.rowCount() == 2 and len(t.merged[0].overlaps) == 1
    assert abs(t.merged[0].overlaps[0].factor - 1) < 0.03
    t.table.selectRow(0)                                         # underlying curves, back to pySASred
    assert t.sources.rowCount() == 2 and t.sources.item(0, 2).text().startswith("pySASred: SD ")
    t.sources.selectRow(1)
    t.on_show_source()
    pg = red.table_panel.page
    sel = pg.selected_rows()
    assert len(sel) == 1 and sel[0].info.startswith("EXP-005370-A_1")
    # pySASred → open merge tool (send)
    merge_gui.send_reduced(red.results)
    assert len(t.curves) == 4 or any(len(x.curves) == 4 for x in merge_gui._TOOLS)
    box.close()
    red.close()
