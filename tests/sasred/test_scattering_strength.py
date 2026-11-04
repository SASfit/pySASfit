"""Scattering strength: transmission modes, p_cS, regimes (Strunz et al. 2000)."""
import os

import numpy as np
import pytest

from pysasfit.sasred.core import scattering_strength as ss

RAW = os.environ.get("PYSASRED_TEST_RAW", "/mnt/user-data/uploads/Sept2026/raw")
need = pytest.mark.skipif(not os.path.isdir(RAW), reason="Vifor raw data not available")


def test_probability_of_a_guinier_curve():
    """p = d λ²/(2π) ∫ q I dq = d λ² I0 · 3 / (4π Rg²) in the small-angle limit."""
    lam, d, I0, rg = 6.0, 0.1, 50.0, 60.0
    exact = d * lam**2 * I0 * 3 / (4 * np.pi * rg**2)
    q = np.geomspace(0.025, 0.08, 60)             # misses most of the forward peak
    I = I0 * np.exp(-(q * rg)**2 / 3)
    p = ss.scattering_probability(q, I, d, lam, high="none")
    assert p.guinier_rg_A == pytest.approx(rg, rel=1e-3)
    assert p.p == pytest.approx(exact, rel=0.01)
    assert p.low_q > p.measured and p.extrapolated_fraction > 0.3 and p.notes
    # whole range measured: no extrapolation needed
    q2 = np.linspace(1e-4, 0.2, 4000)
    p2 = ss.scattering_probability(q2, I0 * np.exp(-(q2 * rg)**2 / 3), d, lam, low="none", high="none")
    assert p2.p == pytest.approx(exact, rel=1e-3)


def test_porod_tail_and_background():
    lam, d, B = 6.0, 0.2, 1e-6
    q = np.geomspace(0.02, 0.2, 50)
    p = ss.scattering_probability(q, B * q**-4 + 0.5, d, lam, background=0.5, low="none")
    tail = d * lam**2 / (2 * np.pi) * B / (2 * q[-1]**2)        # small-angle form of ∫ B q^-3 dq
    assert p.porod_B == pytest.approx(B, rel=1e-6)
    assert p.high_q == pytest.approx(tail, rel=0.01)


def test_scaling_and_thickness():
    assert ss.scale_probability(0.2, 0.1, 0.05, 6, 12) == pytest.approx(0.4)
    assert ss.thickness_for(0.1, 0.4, 0.2) == pytest.approx(0.05)
    assert ss.thickness_for(0.1, 0.1, 0.2, lam=6, lam_to=12) == pytest.approx(0.05)


def test_regimes():
    assert ss.assess(0.01).regime == "weak"
    m = ss.assess(0.15, thickness_cm=0.2, wavelength_A=8.0, background_fraction=0.5)
    assert m.regime == "moderate" and "eq. 8" in m.formula
    assert any("1.33 mm" in a for a in m.actions)                 # 0.2 cm × 0.1/0.15
    assert any("shape" in a for a in m.actions)
    s = ss.assess(0.6, plateau_reached=False)
    assert s.regime == "strong" and any("multiple scattering" in a for a in s.actions)
    assert any("lower limit" in a for a in s.actions)


def test_self_consistent_background_factor():
    """Sample on a strongly scattering substrate: S = I_true + B (1 - p_true); reducing with a
    trial p gives I_true + B (p - p_true); the iteration finds p_true."""
    lam, d = 6.0, 0.1
    q = np.geomspace(2e-3, 0.1, 80)
    I_true = 2000.0 * np.exp(-(q * 80)**2 / 3) + 1e-5 * q**-4 * (1 - np.exp(-(q * 80)**2 / 3))
    p_true = ss.scattering_probability(q, I_true, d, lam).p
    B = 300.0 * np.exp(-(q * 40)**2 / 3)
    S = I_true + B * (1 - p_true)
    p, pr, hist = ss.self_consistent_probability(lambda p: (q, S - B * (1 - p), None), d, lam, p0=0.0, tol=1e-6)
    assert 0.05 < p_true < 0.9 and p == pytest.approx(p_true, rel=1e-4) and len(hist) > 2


def _beams(n=128, T_si=0.8, p_cs=0.3, seed=0):
    yy, xx = np.mgrid[0:n, 0:n]
    r2 = (xx - 64.0)**2 + (yy - 64.0)**2
    beam = np.exp(-r2 / (2 * 2.0**2))
    beam *= 1e6 / beam.sum()
    halo = np.exp(-r2 / (2 * 15.0**2))
    halo *= 1e6 / halo.sum()                   # all scattered neutrons stay on the detector
    rng = np.random.default_rng(seed)
    E = rng.poisson(beam).astype(float)
    S = rng.poisson(T_si * ((1 - p_cs) * beam + p_cs * halo)).astype(float)
    return S, E


def test_aperture_transmission_modes():
    S, E = _beams()
    t = ss.aperture_transmission(S, E, (64.0, 64.0), 7.5e-3, 8.0)
    assert t.plateau_reached and t.T_si == pytest.approx(0.8, abs=0.003)
    assert t.tau_ul > 0.8 * 0.7                    # contaminated by scattering into the beam spot
    assert 0.25 < t.p_cs_lower <= 0.3 and t.p_cs_lower_sigma < 0.005
    assert np.all(np.diff(t.alpha_sr) >= 0) and t.T[0] < t.T[-1]
    # detector too small for the scattering: no plateau
    small = ss.aperture_transmission(S[44:84, 44:84], E[44:84, 44:84], (20.0, 20.0), 7.5e-3, 8.0)
    assert not small.plateau_reached and small.notes


@need
def test_real_transmission_runs():
    from pysasfit.sasred.core.beamcentre import masked_centre_of_gravity
    from pysasfit.sasred.io.sinq import read_sans_i

    def run(n):
        return read_sans_i(os.path.join(RAW, f"sans2026n{n:06d}.hdf"), checksum=False)
    eb = run(82173)
    bc = masked_centre_of_gravity(eb.counts)
    c = (bc.x, bc.y)
    w = ss.aperture_transmission_runs(run(82171), eb, c)          # water
    assert w.T_si == pytest.approx(0.2399, abs=0.006) and w.plateau_reached
    assert any("incoherent" in n for n in w.notes)               # no coherent scattering to see
    assert ss.aperture_transmission_runs(run(82172), eb, c).p_cs_lower < 0.002   # empty cell
    with pytest.raises(ValueError, match="SD"):
        ss.aperture_transmission_runs(run(82113), eb, c)          # A_1 transmission at 7.63 m
    with pytest.raises(ValueError, match="beam stop"):
        ss.aperture_transmission_runs(run(82079), eb, c)          # a scattering run (bsy = 1 mm)
    assert any("dead-time factor" in n for n in w.notes)
