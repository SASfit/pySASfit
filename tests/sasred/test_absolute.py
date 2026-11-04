"""Reduction of table rows: 2D per pixel, then the average; missing corrections are allowed."""
import os

import numpy as np
import pytest

RAW = os.environ.get("PYSASRED_TEST_RAW", "/mnt/user-data/uploads/Sept2026/raw")
need = pytest.mark.skipif(not os.path.isdir(RAW), reason="Vifor raw data not available")


@pytest.fixture(scope="module")
def vifor():
    if not os.path.isdir(RAW):
        pytest.skip("Vifor raw data not available")
    from pysasfit.sasred import plan as pl, reduction_table as rt
    from pysasfit.sasred.catalogue import beam_centres, config_key, scan_folder
    from pysasfit.sasred.io.sinq import read_sans_i
    from pysasfit.sasred.io.transmission import read_transmission_summary
    cat = scan_folder(RAW)
    raws = {e.run: e.raw for e in cat if e.raw}
    t = rt.from_plan(pl.build_plan(cat), "SD 8.0 m, λ 6.00 Å, coll 8.0 m", raws,
                     read_transmission_summary(os.path.join(RAW, "trans.txt")))
    c = beam_centres(list(raws.values()))[(8.0, 6.0, 8.0)]
    return t, (c.x, c.y), (lambda n: read_sans_i(os.path.join(RAW, f"sans2026n{n:06d}.hdf"), checksum=False))


def test_water_as_sample_is_flat_and_absolute(vifor):
    from pysasfit.sasred import absolute as ab
    from pysasfit.sasred.core.averaging import q_edges
    from pysasfit.sasred.reduction_table import RowSet
    t, c, read = vifor
    w = t.corrections["water"]
    row = RowSet("sample", "W", "H2O", runs=w.runs, transmission=w.transmission,
                 transmission_sigma=w.transmission_sigma, scaling=0.2)
    r2, r1 = ab.reduce_row(t, row, read, c, q_edges(0.006, 0.09, 30))
    tau = w.transmission / t.corrections["empty_cell"].transmission
    assert r2.absolute and r1.intensity.unit == "1/cm/sr"
    g = 1.36992 + 0.31389 * 0.6 - 1.07706 * 0.6**2 + 0.54662 * 0.6**3   # g(λ) of the water at 0.6 nm
    t_w = w.transmission                                                          # BerSANS: denominator T_W
    assert ab.water_denominator() == "water_and_cell"
    assert np.allclose(r1.I, g * (1 - tau) / (4 * np.pi * t_w * 0.2), rtol=1e-3)   # sample == water (λ ≈ 0.6 nm)
    assert r2.parameters["k0"] == pytest.approx(g, rel=1e-3) and any("g(λ" in n for n in r1.notes)
    assert "water_g:6.000" in r1.intensity.budget() or any(k.startswith("water_g") for k in r1.intensity.terms)
    _, r1b = ab.reduce_row(t, row, read, c, q_edges(0.006, 0.09, 30), k0=1.0)
    assert np.allclose(r1.I / r1b.I, r2.parameters["k0"], rtol=1e-9)
    v, rel, note = ab.water_g(12.0)
    assert v == pytest.approx(1.1402, abs=2e-4) and rel == 0.006 and "extrapolated" not in note
    assert "extrapolated" in ab.water_g(15.0)[2]
    assert r2.image.shape == (128, 128) and (r2.mask != 0).any()


def test_sample_with_and_without_corrections(vifor, tmp_path):
    import copy

    from pysasfit.sasred import absolute as ab
    from pysasfit.sasred.core.averaging import q_edges
    t, c, read = vifor
    row = t.samples[2]
    e = q_edges(0.006, 0.09, 30)
    _, full = ab.reduce_row(t, row, read, c, e)
    assert any("no Cd" in n for n in full.notes) and full.intensity.unit == "1/cm/sr"
    assert np.all(full.I > 0) and np.all(full.idev() / full.I < 0.1)
    # without sample background and water background: still reduced, said so
    t2 = copy.deepcopy(t)
    t2.corrections["sample_background"].runs = []
    t2.corrections["empty_cell"].runs = []
    _, nobg = ab.reduce_row(t2, t2.samples[2], read, c, e)
    assert any("no sample background" in n for n in nobg.notes)
    assert any("no water background" in n for n in nobg.notes)
    # without water: relative scale per sr and cm
    t2.corrections["water"].runs = []
    r2, rel = ab.reduce_row(t2, t2.samples[2], read, c, e)
    assert not r2.absolute and rel.intensity.unit == "counts/monitor/sr/cm"
    assert any("no water" in n for n in rel.notes)
    # no transmission, no thickness
    t2.samples[2].transmission, t2.samples[2].scaling = None, None
    _, bare = ab.reduce_row(t2, t2.samples[2], read, c, e)
    assert any("T = 1" in n for n in bare.notes) and any("d = 1 cm" in n for n in bare.notes)


@need
def test_gui_reduce_geometry(tmp_path):
    pytest.importorskip("PySide6")
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6 import QtWidgets
    from pysasfit.sasred.gui import MainWindow
    QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    w = MainWindow()
    w.set_folder(RAW)
    page = w.table_panel.page
    page.table.samples = page.table.samples[:3]
    page.refresh()
    tag = page.on_reduce(all_rows=True)
    assert tag == "SD 8.0 m No. 3" and len(w.results) == 3
    r = w.results[tag]
    assert r.intensity.unit == "1/cm/sr" and r.parameters["runs"] == page.table.samples[2].used
    i = w.det_view.findData("reduced2d")
    w.det_view.setCurrentIndex(i)
    assert "reduced 2D" in w.det_canvas.figure.axes[0].get_title()
    paths = w.save_all(str(tmp_path))
    assert len(paths) == 9 and all(os.path.isfile(p) for p in paths)
    w.close()


def test_water_scaling_factor_as_bersans():
    from pysasfit.sasred.absolute import water_scaling_factor
    # BerSANS "Calculate" (T_W 0.5, T_EC 0.95, slit 1 cm²) shows 7.5389e-02: denominator T_W
    assert water_scaling_factor(0.5, 0.95, 1.0, 1.0, "water_and_cell") == pytest.approx(7.5389e-02, rel=1e-5)
    tau = 0.5 / 0.95
    assert water_scaling_factor(0.5, 0.95, 2.0, 1.3, "tau") == pytest.approx(1.3 * 2 * (1 - tau) / (4 * np.pi * tau))


def test_apertures_and_denominator(vifor):
    import copy
    from pysasfit.sasred import absolute as ab
    from pysasfit.sasred.reduce import load_instrument
    from pysasfit.sasred.reduction_table import RowSet
    t, c, read = vifor
    t = copy.deepcopy(t)
    w = t.corrections["water"]
    row = RowSet("sample", "W", "H2O", runs=w.runs, transmission=w.transmission, scaling=0.2)
    ref = ab.reduce_row_2d(t, row, read, c)
    row.aperture_cm2 = 0.5                                       # only one area given: ignored
    one = ab.reduce_row_2d(t, row, read, c)
    assert np.allclose(one.I, ref.I) and any("only one" in n for n in one.notes)
    w.aperture_cm2 = 1.0                                         # water with twice the area of the sample
    two = ab.reduce_row_2d(t, row, read, c)
    ok = (two.mask == 0)
    assert np.allclose(two.I[ok], 2 * ref.I[ok], rtol=1e-9) and two.parameters["aperture_cm2"] == 0.5
    assert any("BerSANS" in n for n in two.notes)
    inst = copy.deepcopy(load_instrument())
    inst["water"]["denominator"] = "tau"
    b = ab.reduce_row_2d(t, row, read, c, instrument=inst)       # Strunz: × T_EC
    assert np.allclose(b.I[ok], two.I[ok] * t.corrections["empty_cell"].transmission, rtol=1e-9)


def test_attenuator_water_check_note():
    import numpy as np
    from types import SimpleNamespace
    from pysasfit.sasred.reduce import load_instrument, normalised_counts
    inst = load_instrument()
    assert inst["attenuator"]["1"]["water_check"]["transmission"] == pytest.approx(1.64e-3)
    raw = SimpleNamespace(run=1, counts=np.full((4, 4), 100.0), counting_time_s=10.0, attenuator="1",
                          monitors={"monitor1": 1000.0})
    notes = []
    normalised_counts(raw, inst, notes)
    assert any("water calibration" in n and "+26 %" in n for n in notes)
