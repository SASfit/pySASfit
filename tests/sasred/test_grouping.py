"""Grouping runs by metadata with precisions and the equivalence check before summing."""
import os
from types import SimpleNamespace

import numpy as np
import pytest

from pysasfit.sasred import grouping as gr

RAW = os.environ.get("PYSASRED_TEST_RAW", "/mnt/user-data/uploads/Sept2026/raw")
need = pytest.mark.skipif(not os.path.isdir(RAW), reason="Vifor raw data not available")


def _raw(run, sample="S", T=None, SD=8.0, start="2026-09-29 14:00:00"):
    return SimpleNamespace(run=run, sample_name=sample, SD_m=SD, wavelength_A=6.0, attenuator="0",
                           start_time=start, counting_time_s=600.0, temperature=T, path="")


def test_group_by_metadata_with_precision():
    T = [20.0, 20.3, 20.4, 25.0, 25.2, 40.0]
    runs = [_raw(i + 1, T=t, start=f"2026-09-29 1{i}:00:00") for i, t in enumerate(T)]
    runs.append(_raw(9, sample="other", T=20.1))
    f = {"temperature": lambda r: r.temperature}
    g = gr.group_runs(runs, {"sample": None, "temperature": 0.5}, fields=f)
    assert [x.run_numbers for x in g if x.values["sample"] == "S"] == [[1, 2, 3], [4, 5], [6]]
    assert g[0].values["temperature"] == (20.0, 20.4) and "temperature=20..20.4" in g[0].label()
    # a group spans at most the precision (no chaining 20.0 -> 20.4 -> 20.8 ...)
    chain = [_raw(i, T=20 + 0.3 * i) for i in range(5)]
    assert [len(x.runs) for x in gr.group_runs(chain, {"temperature": 0.5}, fields=f)] == [2, 2, 1]
    # relative precision and min_runs
    L = [_raw(i, SD=v) for i, v in enumerate((8.0, 8.05, 2.0))]
    assert [x.run_numbers for x in gr.group_runs(L, {"SD_m": "1%"}, min_runs=2)] == [[0, 1]]
    with pytest.raises(KeyError):
        gr.group_runs(runs, {"pressure": 1.0})


def _curves(n=4, nq=40, scale=None, shape=None, outlier=None, counts=2e4, seed=0):
    rng = np.random.default_rng(seed)
    q = np.geomspace(0.005, 0.1, nq)
    model = 1e3 / (1 + (q / 0.01)**3) + 5
    t = 3600.0 * np.arange(n)
    I, s = [], []
    for k in range(n):
        m = model.copy()
        if scale is not None:
            m *= 1 + scale * k
        if shape is not None:
            m *= 1 + shape * k * np.exp(-q / 0.01)        # low q changes most
        if outlier == k:
            m *= 1.05
        c = rng.poisson(m / model[0] * counts * (q / q[0])**0.5)   # counts per bin
        f = m / (m / model[0] * counts * (q / q[0])**0.5)
        I.append(c * f)
        s.append(np.sqrt(np.maximum(c, 1)) * f)
    return q, np.array(I), np.array(s), t


@pytest.mark.parametrize("seed", range(3))
def test_equivalent_runs(seed):
    q, I, s, t = _curves(seed=seed)
    e = gr.equivalence(q, I, s, t, runs=[10, 11, 12, 13])
    assert e.verdict == "equivalent" and e.accepted == [10, 11, 12, 13]
    assert 0.6 < e.chi2_ndf < 1.4 and e.dof == 40 * 3
    assert np.allclose(e.scale, 1, atol=0.01) and not e.scale_trend.significant
    assert e.pair_chi2_ndf.shape == (4, 4) and e.pair_chi2_ndf[0, 0] == 0
    assert "equivalent" in e.summary()


def test_outlier_run():
    q, I, s, t = _curves(n=5, outlier=2)
    e = gr.equivalence(q, I, s, t, runs=[1, 2, 3, 4, 5])
    assert e.verdict == "outlier" and e.outliers == [3] and e.accepted == [1, 2, 4, 5]
    assert np.argmax(e.run_chi2) == 2


def test_sedimentation_as_scale_drift():
    q, I, s, t = _curves(n=4, scale=-0.03)            # 3 % fewer scatterers per hour
    e = gr.equivalence(q, I, s, t)
    assert e.verdict == "scale drift" and e.accepted == []
    assert e.scale_p > 0.01 and e.scale_trend.slope_per_h == pytest.approx(-0.03, rel=0.2)
    assert all(tr.slope_per_h < 0 for tr in e.band_trends)


def test_size_selective_change_as_shape_drift():
    q, I, s, t = _curves(n=4, shape=-0.05)
    e = gr.equivalence(q, I, s, t)
    assert e.verdict == "shape drift" and e.scale_p < 0.01
    low, high = e.band_trends[0], e.band_trends[-1]
    assert low.significant and low.slope_per_h < high.slope_per_h < 0.01


def test_two_runs_cannot_tell_step_from_trend():
    q, I, s, t = _curves(n=2, scale=-0.05)
    e = gr.equivalence(q, I, s, t)
    assert e.verdict == "scale change" and "two runs" in e.message


def test_relative_floor_and_full_covariance():
    q, I, s, t = _curves(n=3, scale=0.002, counts=1e6)  # 0.2 % beam drift, very good statistics
    assert gr.equivalence(q, I, s, t).verdict != "equivalent"
    assert gr.equivalence(q, I, s, t, rel_floor=0.003).verdict == "equivalent"
    cov = np.array([np.diag(x**2) for x in s])
    a, b = gr.equivalence(q, I, s, t, cov=cov), gr.equivalence(q, I, s, t)
    assert np.allclose(a.run_chi2, b.run_chi2)
    fig = gr.plot_equivalence(b)
    assert len(fig.axes) == 3


@need
def test_real_data_misplaced_sample():
    """EXP-005370-A_2 at 2 m: the first run sat at changer position 3.0 mm, the repeat at 29.3 mm."""
    from pysasfit.sasred.catalogue import scan_folder
    from pysasfit.sasred.core import mask as mk
    from pysasfit.sasred.core.averaging import q_edges
    from pysasfit.sasred.core.beamcentre import masked_centre_of_gravity
    from pysasfit.sasred.io.sinq import read_sans_i
    cat = scan_folder(RAW)
    match = {"sample": None, "SD_m": "1%", "wavelength_A": "2%", "attenuator": None}
    g = {x.label(): x for x in gr.group_runs(cat, match, min_runs=2)}
    a2 = next(x for k, x in g.items() if "A_2" in k and "SD_m=2" in k)
    assert a2.run_numbers == [82097, 82148]
    split = gr.group_runs(a2.runs, dict(match, sample_position_mm=0.5))
    assert [x.run_numbers for x in split] == [[82097], [82148]]
    bc = masked_centre_of_gravity(read_sans_i(os.path.join(RAW, "sans2026n082173.hdf")).counts)
    c = (bc.x, bc.y)
    m, _ = mk.suggest_mask(a2.runs[0].counts, c)
    e = gr.check_runs(a2, c, q_edges(0.02, 0.35, 30), mask=m)
    assert e.verdict.startswith("shape") and e.chi2_ndf > 100
    ok = next(x for k, x in g.items() if "A_4" in k and "SD_m=2" in k)
    assert gr.check_runs(ok, c, q_edges(0.02, 0.35, 30), mask=m).verdict == "equivalent"


def _image(kind, rate, rng, centre=(20.3, 18.7), shadow=(21.5, 18.5), n=40):
    yy, xx = np.mgrid[0:n, 0:n]
    if kind == "direct":
        lam = np.exp(-((xx - centre[0])**2 + (yy - centre[1])**2) / (2 * 1.5**2))
    else:
        lam = 1.0 / (1 + ((xx - centre[0])**2 + (yy - centre[1])**2) / 30.0)
        lam[(abs(xx - shadow[0]) <= 3) & (abs(yy - shadow[1]) <= 3)] = 0.002
    return rng.poisson(lam / lam.sum() * rate * 1e5).astype(float)


def _run(run, sample, kind, bs, rate=1.0, att="1", rng=None, SD=8.0):
    r = _raw(run, sample=sample, SD=SD, start=f"2026-09-29 1{run % 10}:00:00")
    r.counts = _image(kind, rate, rng)
    r.monitors = {"monitor1": 1e5}
    r.attenuator, r.coll_m, r.end_time, r.sample_position_mm = att, 8.0, None, 1.0
    r.beamstop_xy_mm, r.pixel_size_m = bs, (0.0075, 0.0075)
    return r


def test_beamstop_distance_to_beam_centre():
    from pysasfit.sasred import catalogue as cg
    rng = np.random.default_rng(4)
    IN = (-27.0, 1.0)
    runs = [_run(1, "S", "scatt", IN, rng=rng), _run(2, "H2O", "scatt", IN, rng=rng),
            _run(3, "S", "direct", (-27.0, -300.0), rng=rng),
            _run(4, "EB", "direct", (223.0, 1.0), att="0", rng=rng),        # moved in x, no attenuator
            _run(5, "S", "scatt", (-27.0, -300.0), rng=rng),                # moved, but no direct beam
            _run(6, "S", "scatt", (-27.0, -60.0), rng=rng)]                 # 60 mm: still "in"
    geo = cg.beamstop_geometry(runs)
    g = geo[(8.0, 6.0, 8.0)]
    assert g.beam_centre_px == pytest.approx((20.3, 18.7), abs=0.15)
    assert g.shadow_px == pytest.approx((21.5, 18.5), abs=0.1) and g.ref_mm == IN
    assert g.off_centre_mm == pytest.approx(np.hypot(1.2, 0.2) * 7.5, abs=1.5)   # shadow beside the beam
    assert cg.beamstop_distance(runs[2], geo) == pytest.approx(np.hypot(9.0, 302.5), abs=2)
    assert cg.beamstop_distance(runs[3], geo) == pytest.approx(259.0, abs=2)
    kinds, odd, _ = cg.classify(runs)
    assert [kinds[i] for i in range(1, 7)] == ["scattering", "scattering", "transmission", "transmission",
                                               "transmission", "scattering"]
    assert [o[0] for o in odd] == [5] and "no direct beam" in odd[0][1]
    assert not cg.is_transmission(runs[5], geo) and cg.is_transmission(runs[5], geo, limit=50)
    assert "off the beam centre" in g.describe()
    assert cg.looks_like_direct_beam(runs[2]) and not cg.looks_like_direct_beam(runs[0])


def test_plan_roles_and_rates(tmp_path):
    from pysasfit.sasred import plan as pl
    assert [pl.role_of(n) for n in ("H2O", "EC", "EB", "Cd", "B4C", "empty cell", "EXP-1")] == \
        ["water", "empty_cell", "empty_beam", "blocked_beam", "blocked_beam", "empty_cell", "sample"]
    assert pl.role_of("EC", {"EC": "sample_background"}) == "sample_background"
    assert pl._runs([5, 1, 2, 3, 7]) == "1–3, 5, 7"
    rng = np.random.default_rng(2)
    OUT, IN = (-27.0, -300.0), (-27.0, 1.0)
    runs = [_run(1, "S", "direct", OUT, 1.0, rng=rng), _run(2, "S", "direct", OUT, 1.0, rng=rng),
            _run(3, "S", "direct", OUT, 1.0, rng=rng), _run(4, "S", "direct", OUT, 1.2, rng=rng),
            _run(5, "EB", "direct", OUT, rng=rng), _run(6, "H2O", "scatt", IN, att="0", rng=rng)]
    v, outl, _ = pl.check_rates(runs[:4])
    assert v == "outlier" and outl == [4]
    p = pl.build_plan(runs)
    s = p.rows(role="sample")[0]
    assert s.kind == "transmission" and s.summed == [1, 2, 3] and s.outliers == [4]
    assert p.rows(role="water")[0].kind == "scattering" and p.rows(role="water")[0].verdict == "single"
    assert any("no empty_cell (scattering)" in m for m in p.missing)
    md = p.markdown()
    assert "| sample | S | transmission | 1–4 | 1–3 | 4 |" in md and "Beam stop: beam stop in the beam" in md
    q = pl.Plan.load(p.save(tmp_path / "plan.json"))
    assert q.entries[0] == p.entries[0] and q.missing == p.missing and q.beamstop == p.beamstop
