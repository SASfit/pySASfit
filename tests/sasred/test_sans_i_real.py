"""Tests on real SANS-I data (Vifor, September 2026). Skipped when the raw
files are not available."""
import os

import h5py
import numpy as np
import pytest

from pysasfit.sasred.core.averaging import q_edges
from pysasfit.sasred.core.beamcentre import masked_centre_of_gravity
from pysasfit.sasred.io.nxcansas import write_nxcansas_1d
from pysasfit.sasred.io.sinq import MissingField, RawRun, read_sans_i
from pysasfit.sasred.reduce import load_instrument, reduce_run_1d

RAW = os.environ.get("PYSASRED_TEST_RAW", "/mnt/user-data/uploads/Sept2026/raw")
need = pytest.mark.skipif(not os.path.isdir(RAW), reason="Vifor raw data not available")


def path(run):
    return os.path.join(RAW, f"sans2026n{run:06d}.hdf")


@need
def test_reader_units_and_fields():
    r = read_sans_i(path(82130))
    assert r.run == 82130 and r.counts.shape == (128, 128)
    assert abs(r.wavelength_A - 6.0) < 0.01          # stored in nm
    assert abs(r.L2_m - 8.0) < 0.01                  # stored in mm
    assert r.pixel_size_m == (0.0075, 0.0075)
    assert r.dlam_over_lam_fwhm == pytest.approx(0.09832)   # selector tilt 0° (table), not the header 0.1
    assert r.monitors["monitor1"] == 8e6 and not r.aborted


@need
def test_missing_required_field_raises(tmp_path):
    src = path(82130)
    dst = tmp_path / "sans2026n082130.hdf"
    with h5py.File(src) as a, h5py.File(dst, "w") as b:
        a.copy("entry1", b)
        del b["entry1/SANS/Dornier-VS/lambda"]
    with pytest.raises(MissingField):
        read_sans_i(dst)


@need
def test_beam_centre_matches_bersans():
    bc = masked_centre_of_gravity(read_sans_i(path(82173)).counts)
    assert abs(bc.x + 1 - 62.73) < 0.1               # BerSANS 8m.scr, 1-based x
    assert abs(bc.y - 63.94) < 0.1                   # BerSANS y (offset to be clarified)


def _reduce(raw, bc, edges, inst):
    yy, xx = np.mgrid[:128, :128]
    mask = np.where((xx - bc[0]) ** 2 + (yy - bc[1]) ** 2 < 36, 2, 0).astype(np.uint32)
    return reduce_run_1d(raw, bc, edges, instrument=inst, mask=mask)


@need
def test_independent_recalculation():
    """I(q) recomputed directly with numpy, without the UArray engine."""
    raw = read_sans_i(path(82130))
    inst = load_instrument()
    bc = (61.728, 63.931)
    edges = q_edges(0.002, 0.08, 40)
    r = _reduce(raw, bc, edges, inst)
    N = raw.counts.astype(float)
    tau = inst["detector"]["deadtime"]["tau_s"]
    f = 1 / (1 - tau * N.sum() / raw.counting_time_s)
    yy, xx = np.mgrid[:128, :128]
    x, y = (xx - bc[0]) * 0.0075, (yy - bc[1]) * 0.0075
    L2 = raw.L2_effective_m                       # recorded distance + offset (preference)
    d = np.sqrt(x * x + y * y + L2 ** 2)
    dom = 0.0075 ** 2 * L2 / d ** 3
    q = 4 * np.pi / raw.wavelength_A * np.sin(np.arccos(L2 / d) / 2)
    img = N * f / raw.monitors["monitor1"] / dom
    ok = (xx - bc[0]) ** 2 + (yy - bc[1]) ** 2 >= 36
    b = np.searchsorted(edges, q, side="right") - 1
    ref = [img[ok & (b == k)].mean() for k in range(len(edges) - 1) if np.any(ok & (b == k))]
    np.testing.assert_allclose(r.intensity.value, ref, rtol=1e-12)


@need
def test_monte_carlo_covariance_on_real_counts():
    """Poisson-resample the real counts and monitor; the sample covariance of
    I(q) must agree with the propagated one. A large dead time (30x the real
    value) makes the rank-1 dead-time coupling visible."""
    raw = read_sans_i(path(82130))
    inst = load_instrument()
    inst["detector"]["deadtime"]["tau_s"] = 2.0e-5
    bc = (61.728, 63.931)
    edges = q_edges(0.003, 0.08, 25)
    r = _reduce(raw, bc, edges, inst)
    C = r.intensity.covariance()
    rng = np.random.default_rng(11)
    sims = []
    for _ in range(3000):
        rr = RawRun(**{**raw.__dict__, "counts": rng.poisson(raw.counts),
                       "monitors": {**raw.monitors, "monitor1": float(rng.poisson(raw.monitors["monitor1"]))}})
        sims.append(_reduce(rr, bc, edges, inst).intensity.value)
    S = np.cov(np.array(sims).T)
    d = np.sqrt(np.diag(C))
    # variances within the Monte Carlo error (sqrt(2/n) ~ 2.6 %) and correlations within 0.06
    np.testing.assert_allclose(np.diag(S), np.diag(C), rtol=0.1)
    np.testing.assert_allclose(S / np.outer(d, d), C / np.outer(d, d), atol=0.07)
    # the off-diagonal (dead-time and monitor) correlation is real and reproduced
    iu = np.triu_indices_from(C, 1)
    corr_p = (C / np.outer(d, d))[iu]
    corr_mc = (S / np.outer(np.sqrt(np.diag(S)), np.sqrt(np.diag(S))))[iu]
    assert np.median(corr_p) > 0.02
    assert abs(np.median(corr_mc) - np.median(corr_p)) < 0.01


@need
def test_nxcansas_output(tmp_path):
    raw = read_sans_i(path(82130))
    r = _reduce(raw, (61.728, 63.931), q_edges(0.002, 0.08, 40), load_instrument())
    out = tmp_path / "a1.h5"
    write_nxcansas_1d(out, r)
    with h5py.File(out) as f:
        d = f["sasentry01/sasdata01"]
        assert d.attrs["signal"] == b"I"
        np.testing.assert_allclose(d["I"][()], r.intensity.value)
        np.testing.assert_allclose(d["Idev"][()] ** 2, np.diag(d["I_covariance_statistical"][()]))
        assert "monitor1:82130" in d["I_systematic"]
        assert f["sasentry01/sasprocess01/raw_sha256"][()].decode() == raw.sha256
