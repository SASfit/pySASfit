"""Detector efficiency from water: separable fill of masked pixels, uncertainty, files, use in the
reduction (synthetic data; the Vifor water runs when available)."""
import os

import numpy as np
import pytest

from pysasfit.sasred.core import mask as mk
from pysasfit.sasred.core.counting import counts_uarray
from pysasfit.sasred.core.efficiency import Efficiency, efficiency_from_water, separable_fill
from pysasfit.sasred.data.uarray import constant

RAW = os.environ.get("PYSASRED_TEST_RAW", "/mnt/user-data/uploads/Sept2026/raw")
need = pytest.mark.skipif(not os.path.isdir(RAW), reason="Vifor raw data not available")


def _setup(n=64, spread=0.1, seed=1):
    rng = np.random.default_rng(seed)
    a = 1 + spread * rng.standard_normal(n)          # y wires (rows)
    b = 1 + spread * rng.standard_normal(n)          # x wires (columns)
    m = mk.border_mask((n, n))
    m[mk.pixels_in_disc((n, n), 30, 33, 5)] |= mk.BEAMSTOP
    return np.outer(a, b), m


def test_masked_pixels_are_row_times_column():
    eps, m = _setup()
    valid = m == 0
    e = efficiency_from_water(constant(eps), valid, reject_sigma=None)
    v = e.value.value
    assert v[valid].mean() == pytest.approx(1.0)
    assert e.filled.sum() == np.count_nonzero(m == mk.BEAMSTOP)          # the beamstop disc
    assert np.all(e.undefined == (m & mk.BORDER).astype(bool))            # whole rows / columns masked
    assert np.all(v[e.undefined] == 1.0) and np.all(e.mask_bits()[e.undefined] == mk.EFFICIENCY)
    true = eps / eps[valid].mean()
    # x/y wire model: exact for a separable detector apart from the row/column sets (< 1 %)
    assert np.max(np.abs(v[e.filled] / true[e.filled] - 1)) < 0.01
    # the formula itself: R_i C_j / m with the means over the valid pixels
    w = np.where(valid, true, np.nan)
    i, j = np.argwhere(e.filled)[0]
    assert v[i, j] == pytest.approx(np.nanmean(w[i]) * np.nanmean(w[:, j]) / np.nanmean(w))


def test_partially_masked_row_and_column_filled_only_where_defined():
    v = np.ones((6, 6), bool)
    v[2, :] = False          # whole row
    v[4, 3] = False
    e, filled, undefined = separable_fill(constant(np.ones((6, 6))), v)
    assert undefined[2].all() and filled[4, 3] and filled.sum() == 1
    assert np.allclose(e.value, 1.0)


def test_uncertainty_against_monte_carlo():
    eps, m = _setup(n=32, spread=0.05, seed=3)
    m = mk.border_mask((32, 32))
    m[mk.pixels_in_disc((32, 32), 15, 16, 3)] |= mk.BEAMSTOP
    valid = m == 0
    lam = np.round(400.0 * eps)
    e = efficiency_from_water(counts_uarray(np.round(lam), "counting:w"), valid, reject_sigma=None)
    sig = e.std()
    rng = np.random.default_rng(7)
    sims = np.array([efficiency_from_water(constant(rng.poisson(lam).astype(float)), valid,
                                               reject_sigma=None).value.value
                     for _ in range(400)])
    mc = sims.std(axis=0)
    for sel in (e.measured, e.filled):
        ratio = sig[sel] / mc[sel]
        assert 0.85 < np.median(ratio) < 1.15
    # a filled pixel depends on its whole row and column: smaller error than a measured one
    assert np.median(sig[e.filled]) < 0.5 * np.median(sig[e.measured])


def test_beamstop_edge_pixels_are_rejected():
    eps, m = _setup()
    valid = m == 0
    counts = np.round(2000 * eps)
    counts[33, 36] = np.round(counts[33, 36] * 0.25)  # half shadow at the beamstop disc
    counts[10, 10] = np.round(counts[10, 10] * 0.25)  # far from the beamstop: real defect, kept
    e = efficiency_from_water(counts_uarray(counts, "counting:w"), valid, border=(m & mk.BORDER).astype(bool))
    assert e.rejected[33, 36] and not e.rejected[10, 10] and e.rejected.sum() == 1
    assert e.filled[33, 36] and abs(e.value.value[33, 36] - 1) < 0.4 and e.value.value[10, 10] < 0.4
    assert e.meta["n_rejected"] == 1


def test_save_and_load(tmp_path):
    eps, m = _setup()
    e = efficiency_from_water(counts_uarray(np.round(500 * eps), "counting:w"), m == 0, meta={"water_run": 1})
    p = e.save(tmp_path / "eff.h5")
    e2 = Efficiency.load(p)
    assert np.allclose(e2.value.value, e.value.value) and np.allclose(e2.std(), e.std())
    assert np.array_equal(e2.filled, e.filled) and np.array_equal(e2.undefined, e.undefined)
    assert np.array_equal(e2.rejected, e.rejected)
    assert e2.meta["water_run"] == 1 and list(e2.value.terms) == ["flood:eff"]


def test_transmission_table(tmp_path):
    from pysasfit.sasred.io.transmission import find_transmission, read_transmission_summary
    p = tmp_path / "trans.txt"
    p.write_text("Summary of results:\n-------------------\n   Data file,    Sample name, Transmission, +/- Error\n"
                 "D0082267.001,            H2O,       0.2399,    0.0008\n"
                 "D0082268.001,             EC,       0.8835,    0.0019\n")
    t = read_transmission_summary(p)
    assert [x.run for x in t] == [82267, 82268]
    w = find_transmission(t, "H2O")
    assert (w.value, w.sigma) == (0.2399, 0.0008)
    with pytest.raises(KeyError):
        find_transmission(t, "D2O")


@need
def test_water_efficiency_on_real_data(tmp_path):
    from pysasfit.sasred.core.averaging import q_edges
    from pysasfit.sasred.core.beamcentre import masked_centre_of_gravity
    from pysasfit.sasred.io.sinq import read_sans_i
    from pysasfit.sasred.io.transmission import find_transmission, read_transmission_summary
    from pysasfit.sasred.reduce import efficiency_from_runs, reduce_run_1d

    def run(n):
        return read_sans_i(os.path.join(RAW, f"sans2026n{n:06d}.hdf"))

    tt = read_transmission_summary(os.path.join(RAW, "trans.txt"))
    tw, te = find_transmission(tt, "H2O"), find_transmission(tt, "EC")
    bc = masked_centre_of_gravity(run(82173).counts)
    c = (bc.x, bc.y)
    eff = efficiency_from_runs([run(82280), run(82282)], [run(82281), run(82283)], c,
                               (tw.value, tw.sigma), (te.value, te.sigma))
    v = eff.value.value
    assert v[eff.measured].mean() == pytest.approx(1.0)
    assert 50 < eff.filled.sum() < 400 and eff.meta["water_runs"] == [82280, 82282]
    assert 0.8 < v[eff.filled].min() and v[eff.filled].max() < 1.3
    assert {"counting:82280", "counting:82283", "transmission:82280"} <= set(eff.value.terms)
    raw = run(82140)
    m, _ = mk.suggest_mask(raw.counts, c)
    edges = q_edges(0.006, 0.09, 40)
    r0 = reduce_run_1d(raw, c, edges, mask=m)
    r1 = reduce_run_1d(raw, c, edges, mask=m, efficiency=eff)
    assert np.all(np.abs(r1.I / r0.I - 1) < 0.2)
    assert eff.meta['n_rejected'] >= 0               # (the rectangular beamstop mask leaves few half shadows)
    assert np.all(r1.idev() / r1.I >= 0.999 * r0.idev() / r0.I)       # the flood adds its (relative) noise
    assert "counting:82280" in r1.intensity.budget()
    assert any("efficiency" in n for n in r1.notes)
    # the same map from its file: one source per pixel
    e2 = Efficiency.load(eff.save(tmp_path / "flood_8m.h5"))
    r2 = reduce_run_1d(raw, c, edges, mask=m, efficiency=e2)
    assert np.allclose(r2.I, r1.I) and "flood:flood_8m" in r2.intensity.budget()


def test_archive_in_one_file(tmp_path):
    """Maps with date stamps in one HDF5 archive; listing, loading, comparison over time."""
    import h5py

    from pysasfit.sasred import efficiency_log as log
    eps, m = _setup()
    valid = m == 0
    arch = str(tmp_path / "eff_archive.h5")
    rng = np.random.default_rng(5)
    names = []
    for date, gain in (("2026-09-29 14:00:00", 1.0), ("2026-03-02 09:00:00", 1.0), ("2026-10-20 08:00:00", 0.9)):
        e_true = eps.copy()
        e_true[:, 40] *= gain                                            # a weaker x wire later on
        counts = rng.poisson(20000 * e_true).astype(float)
        e = efficiency_from_water(counts_uarray(counts, "counting:w"), valid,
                                  meta=dict(measured_start=date, SD_m=8.0, wavelength_A=6.0, water_runs=[1],
                                            detector_tilt_deg=0.025, dlam_over_lam_fwhm=0.1))
        names.append(log.archive_efficiency(e, arch, comment=f"gain {gain}"))
    assert names[0].startswith("2026-09-29_SD-8.0m_6.00A_run1")
    with h5py.File(arch) as f:
        assert len(f) == 3 and f.attrs["default"] == names[2]          # newest measurement
        assert f[names[0]].attrs["NX_class"] == "NXentry" and "efficiency" in f[names[0]]
        det = f[names[0]]["instrument/detector"]
        assert det["distance"][()] == 8.0 and det["distance"].attrs["units"] == "m"
        assert det["tilt_angle"][()] == 0.025 and det["tilt_angle"].attrs["units"] == "deg"
        assert f[names[0]]["instrument/monochromator/wavelength"][()] == 6.0
    maps = log.list_maps(arch)
    assert [r["measured"] for r in maps] == ["2026-03-02", "2026-09-29", "2026-10-20"]
    assert maps[2]["comment"] == "gain 0.9" and maps[0]["n_filled"] > 0
    assert maps[0]["SD_m"] == 8.0 and maps[0]["wavelength_A"] == 6.0 and maps[0]["detector_tilt_deg"] == 0.025
    newest = log.load_map(path=arch)
    assert newest.meta["entry"] == names[2]
    h = log.history(arch)
    assert h[0]["rms_change"] == 0 and h[1]["changed_columns"] == []
    assert 0.7 < h[1]["chi2"] < 1.3 and h[2]["chi2"] > h[1]["chi2"]       # noise only, then a real change
    assert 40 in h[2]["changed_columns"] and h[2]["changed_rows"] == []
    assert h[2]["n_changed"] > 10
    fig = log.plot_history(arch)
    assert len(fig.axes) == 3
    # other configurations have their own reference
    e2m = efficiency_from_water(counts_uarray(rng.poisson(20000 * eps).astype(float), "counting:w"), valid,
                                meta=dict(measured_start="2026-10-21 08:00:00", SD_m=2.0, wavelength_A=6.0))
    n2 = log.archive_efficiency(e2m, arch)
    h = {r["entry"]: r for r in log.history(arch)}
    assert h[n2]["reference"] == n2 and h[names[2]]["reference"] == names[1]
    # the same name twice does not overwrite
    again = log.archive_efficiency(log.load_map(names[0], arch), arch)
    assert again == names[0] + "_2" and len(log.list_maps(arch)) == 5
