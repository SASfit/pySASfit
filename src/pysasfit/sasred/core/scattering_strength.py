# SPDX-License-Identifier: BSD-3-Clause
"""How strongly does a sample scatter? Transmission modes, the coherent scattering probability
p_cS and the regime of the reduction formula (Strunz, Šaroun, Keiderling, Wiedenmann &
Przenioslo, J. Appl. Cryst. 33 (2000) 829).

Symbols of the paper:

* t_S     transmission for absorption and large-angle coherent (Bragg) scattering;
* T_Si    = t_S (1 - p_iS): absorption and incoherent scattering (what a transmission measured
  over a wide solid angle gives once all small-angle scattering is collected: the plateau);
* τ_S     = T_Si (1 - p_cS): total removal transmission (what a transmission through the direct
  beam spot only would give; measurable only as an upper limit τ_UL > τ_S because neutrons
  scattered into the beam spot are always counted);
* p_cS    = d_S ∫ dΣ/dΩ dΩ: probability of at least one coherent small-angle scattering event.

The cross-section from a sample in a container or on a substrate (EB) is (their eq. 3/8)

    [(S - Cd)/(T_EB T_Si) - (EB - Cd)/T_EB · (1 - p_cS)] / (d_S I_prim η ΔΩ)

so (1 - p_cS) multiplies the background, and T_Si (not τ_S) divides the sample. With τ_S in the
first term (their eq. 9, aqueous samples where incoherent attenuation dominates) the whole curve
is shifted by (1 - p_cS). BerSANS implements the general form with a user-given "scattering
probability" p(S) (default 0) and a transmission whose meaning depends on the detector area used.

This module provides

* :func:`aperture_transmission`: T(α) from a direct-beam run with and without sample, summed over
  growing discs; the plateau gives T_Si, the beam spot τ_UL, and 1 - τ_UL/T_Si ≤ p_cS;
* :func:`scattering_probability`: p_cS from a cross-section in absolute units, with Guinier and
  Porod extrapolations outside the measured q range;
* :func:`scale_probability` / :func:`thickness_for`: p_cS ∝ d λ² for single scattering;
* :func:`assess`: the regime (weak / moderate / strong) and what to do;
* :func:`self_consistent_probability`: the formula is an integral equation in p_cS; iterate
  reduce -> p_cS -> reduce until p_cS is stable.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, List, Optional, Sequence

import numpy as np

__all__ = ["TransmissionCurve", "aperture_transmission", "aperture_transmission_runs", "Probability",
           "scattering_probability", "scale_probability", "thickness_for", "Assessment", "assess",
           "self_consistent_probability", "solid_angle_of_cone", "plot_transmission"]


def solid_angle_of_cone(two_theta_max) -> np.ndarray:
    """α = 2π (1 - cos 2θ_max), the solid angle of the cone up to 2θ_max (sr)."""
    return 2 * np.pi * (1 - np.cos(np.asarray(two_theta_max, float)))


# ---------------------------------------------------------------------------- transmission modes
@dataclass
class TransmissionCurve:
    """Transmission summed over discs of radius R around the beam (Strunz et al. fig. 1)."""
    radius_px: np.ndarray
    alpha_sr: np.ndarray
    T: np.ndarray
    sigma: np.ndarray
    beam_radius_px: float          # disc holding (beam_fraction) of the empty-beam counts
    T_si: float                    # plateau (largest disc)
    T_si_sigma: float
    tau_ul: float                  # beam spot only
    tau_ul_sigma: float
    plateau_reached: bool          # T no longer changes over the outer part of the detector
    notes: List[str] = field(default_factory=list)
    ratio_sigma: float = 0.0       # σ of τ_UL / T_Si (the beam disc is part of the full sum)
    ring_edges_px: Optional[np.ndarray] = None   # radial average of the two runs per ring
    ring_sample: Optional[np.ndarray] = None     # mean per pixel, background subtracted
    ring_empty: Optional[np.ndarray] = None
    ring_sample_sigma: Optional[np.ndarray] = None
    ring_background: Optional[np.ndarray] = None  # subtracted background per pixel

    def scattered(self):
        """Radially averaged scattering outside the transmitted beam: S - τ_UL E per pixel and ring
        (≈ 0 inside the beam spot, the small-angle and incoherent halo outside)."""
        return self.ring_sample - self.tau_ul * self.ring_empty, self.ring_sample_sigma

    @property
    def p_cs_lower(self) -> float:
        """Lower limit of the coherent scattering probability: 1 - τ_UL / T_Si (≥ 0)."""
        return max(0.0, 1.0 - self.tau_ul / self.T_si)

    @property
    def p_cs_lower_sigma(self) -> float:
        return float(self.ratio_sigma)

    @property
    def incoherent_share(self) -> float:
        """Rough part of 1 - τ_UL/T_Si caused by incoherent scattering into the detector: the
        removed fraction 1 - T_Si spread isotropically over 4π, seen in the detector solid angle α
        (Strunz et al. section 5). Water in a flat cell scatters more forward (K0 ≈ 1.4 at 6 Å)."""
        return float((1 - self.T_si) / self.T_si * self.alpha_sr[-1] / (4 * np.pi))


def aperture_transmission(sample, empty, centre, pixel_size_m, SD_m, sample_var=None, empty_var=None,
                          cd=None, cd_var=None, valid=None, radii_px: Optional[Sequence[float]] = None,
                          beam_fraction: float = 0.99, flat_background="auto",
                          n_rings: int = 12) -> TransmissionCurve:
    """T(α) = Σ_disc(S - B) / Σ_disc(E - B) for discs of growing radius around ``centre``, and the
    radially averaged sample/empty ratio per ring.

    ``sample``, ``empty`` (and ``cd``) are images normalised to the same monitor (direct beam with
    and without the sample, beam stop out, attenuated as needed); the variances default to the
    values themselves (Poisson counts). ``pixel_size_m`` is (x, y) or one number.

    Background B: the Cd run when given. Without one, ``flat_background="auto"`` takes the mean of
    the empty-beam run in the outer half of the detector as a flat, beam-independent background per
    pixel (a number can be given instead; ``None`` = no background). Without it, the large discs
    collect background in both runs and T(α) drifts towards 1 by about (1 - T) · B_total / E_total."""
    S = np.asarray(sample, float)
    E = np.asarray(empty, float)
    vS = S.copy() if sample_var is None else np.asarray(sample_var, float)
    vE = E.copy() if empty_var is None else np.asarray(empty_var, float)
    ok = np.ones(S.shape, bool) if valid is None else np.asarray(valid, bool)
    px = np.broadcast_to(np.asarray(pixel_size_m, float), (2,))
    ny, nx = S.shape
    yy, xx = np.mgrid[0:ny, 0:nx]
    r_px = np.hypot(xx - centre[0], yy - centre[1])
    r_m = np.hypot((xx - centre[0]) * px[0], (yy - centre[1]) * px[1])
    rmax = float(r_px[ok].max())
    notes: List[str] = []
    b, vb = 0.0, 0.0                 # flat background per pixel and the variance of its mean
    if cd is not None:
        C = np.asarray(cd, float)
        vC = C.copy() if cd_var is None else np.asarray(cd_var, float)
    else:
        C, vC = np.zeros_like(S), np.zeros_like(S)
        if isinstance(flat_background, str) and flat_background == "auto":
            m = ok & (r_px >= 0.5 * rmax)
            b = float(E[m].mean())
            vb = float(vE[m].sum()) / m.sum()**2
            notes.append(f"no Cd run: flat background {b:.3g} per pixel from the outer half of the empty beam")
        elif flat_background is not None:
            b = float(flat_background)
    # beam spot from the empty beam
    e = np.where(ok, E - C - b, 0.0)
    order = np.argsort(r_px[ok])
    cum = np.cumsum(e[ok][order])
    k = int(np.searchsorted(cum, beam_fraction * cum[-1]))
    beam_r = float(max(1.0, r_px[ok][order][min(k, order.size - 1)]))
    if radii_px is None:
        radii_px = np.unique(np.concatenate([np.geomspace(max(1.0, 0.5 * beam_r), rmax, 40), [beam_r]]))
    radii = np.asarray(sorted(radii_px), float)

    def disc(m):
        n = m.sum()
        num = S[m].sum() - C[m].sum() - n * b
        den = E[m].sum() - C[m].sum() - n * b
        t = num / den
        var = (vS[m].sum() + t**2 * vE[m].sum() + (1 - t)**2 * (vC[m].sum() + (n**2) * vb)) / den**2
        return t, var, num, den

    T, sT, alpha = [], [], []
    for R in radii:
        m = ok & (r_px <= R)
        t, var, _, _ = disc(m)
        T.append(t)
        sT.append(np.sqrt(var))
        rm = r_m[m].max() if m.any() else 0.0
        alpha.append(solid_angle_of_cone(np.arctan(rm / SD_m)))
    T, sT, alpha = np.array(T), np.array(sT), np.array(alpha)
    ib = int(np.argmin(np.abs(radii - beam_r)))
    # τ_UL / T_Si with the correlation of the two sums (inner disc + rest); background neglected here
    m_in = ok & (r_px <= radii[ib])
    m_out = ok & (r_px <= radii[-1]) & ~m_in
    _, _, Si, Ei = disc(m_in)
    _, _, So, Eo = disc(m_out)
    Sa, Ea = Si + So, Ei + Eo
    rel2 = ((1 / Si - 1 / Sa)**2 * vS[m_in].sum() + (1 / Sa)**2 * vS[m_out].sum()
            + (1 / Ei - 1 / Ea)**2 * vE[m_in].sum() + (1 / Ea)**2 * vE[m_out].sum())
    ratio_sigma = float((Si / Ei) / (Sa / Ea) * np.sqrt(rel2))
    # radial average per ring: where do the scattered neutrons land?
    ring_edges = np.unique(np.concatenate([[0.0], np.geomspace(max(1.0, 0.5 * beam_r), rmax + 1e-9, n_rings)]))
    rs, re_, rsig, rbg = [], [], [], []
    for lo, hi in zip(ring_edges[:-1], ring_edges[1:]):
        m = ok & (r_px >= lo) & (r_px < hi)
        n = max(int(m.sum()), 1)
        sm = (S[m].sum() - C[m].sum()) / n - b
        em = (E[m].sum() - C[m].sum()) / n - b
        rs.append(sm)
        re_.append(em)
        rsig.append(np.sqrt(vS[m].sum() + vC[m].sum()) / n)
        rbg.append(b + C[m].sum() / n)
    outer = radii >= 0.7 * radii[-1]
    plateau = bool(abs(T[-1] - T[outer][0]) < 3 * np.hypot(sT[-1], sT[outer][0]) + 1e-12)
    if not plateau:
        notes.append("T still rises at the detector edge: not all small-angle scattering is collected, "
                     "T_Si is a lower limit (shorter distance or longer counting)")
    out = TransmissionCurve(radii, alpha, T, sT, beam_r, float(T[-1]), float(sT[-1]),
                            float(T[ib]), float(sT[ib]), plateau, notes, ratio_sigma,
                            ring_edges, np.array(rs), np.array(re_), np.array(rsig), np.array(rbg))
    if out.p_cs_lower < 3 * max(out.incoherent_share, ratio_sigma):
        out.notes.append(f"1 - τ_UL/T_Si = {out.p_cs_lower:.4f} is of the order of its uncertainty or of the "
                         f"incoherent scattering into the detector ({out.incoherent_share:.4f} if isotropic; "
                         "more forward for flat water-like samples): no measurable coherent scattering "
                         "probability")
    return out


def aperture_transmission_runs(sample, empty, centre, cd=None, instrument=None, valid=None,
                               check_beamstop: bool = True, **kw) -> TransmissionCurve:
    """:func:`aperture_transmission` from raw runs. Each run is corrected like a scattering run
    (counting error, dead time, attenuator, monitor: :func:`reduce.normalised_counts`), so sample
    and empty beam may use different attenuators and count rates.

    Both runs must show the direct beam (beam stop out of the beam,
    :func:`catalogue.looks_like_direct_beam`) and be at the same distance. The dead-time factor of each run is
    reported; above the preference ``sasred.transmission.deadtime_warn`` (1.05) the run is close to
    saturation and a (stronger) attenuator is recommended."""
    from ..catalogue import is_transmission
    from ..reduce import load_instrument, normalised_counts
    inst = instrument or load_instrument(sample)
    if abs(sample.SD_m - empty.SD_m) > 0.01 * empty.SD_m:
        raise ValueError(f"sample at SD = {sample.SD_m:.2f} m, empty beam at {empty.SD_m:.2f} m")
    if check_beamstop:
        inbeam = [r.run for r in (sample, empty, cd) if r is not None and not is_transmission(r)]
        if inbeam and cd is not None and inbeam == [cd.run]:
            inbeam = []                       # a Cd run may be measured with the beam stop in place
        if inbeam:
            raise ValueError(f"run(s) {inbeam}: the image shows no direct beam (beam stop in the beam?): "
                             "not a transmission run")
    try:
        from ...core import settings as prefs
        warn = float(prefs.effective("sasred.transmission.deadtime_warn"))
    except Exception:  # noqa: BLE001
        warn = 1.05
    notes: List[str] = []
    images = []
    for r in (sample, empty, cd):
        if r is None:
            images.append((None, None))
            continue
        x, factor, _, _ = normalised_counts(r, inst, notes)
        f = float(np.max(factor.value)) if hasattr(factor, "value") else float(np.max(factor))
        if f > warn:
            notes.append(f"run {r.run}: dead-time factor {f:.3f} (attenuator {r.attenuator}): close to "
                         "saturation, use a (stronger) attenuator")
        else:
            notes.append(f"run {r.run}: dead-time factor {f:.4f}, attenuator {r.attenuator}")
        images.append((x.value, x.std(systematic=False)**2))
    (s, vs), (e, ve), (c, vc) = images
    if sample.attenuator != empty.attenuator:
        notes.append(f"attenuator {sample.attenuator} (sample) vs {empty.attenuator} (empty beam): their "
                     "factors enter T directly (uncertainty not known)")
    t = aperture_transmission(s, e, centre, sample.pixel_size_m, getattr(sample, "SD_effective_m", sample.SD_m),
                              vs, ve, c, vc, valid, **kw)
    t.notes = list(dict.fromkeys(notes + t.notes))
    return t


def plot_transmission(t: TransmissionCurve, fig=None):
    """T(α) with τ_UL and T_Si, and the radially averaged runs per ring."""
    import matplotlib.pyplot as plt
    if fig is None:
        fig = plt.figure(figsize=(9, 3.6))
    a1, a2 = fig.subplots(1, 2)
    a1.errorbar(t.alpha_sr, t.T, t.sigma, fmt="o-", ms=3, lw=0.8)
    a1.axhline(t.T_si, color="#2E8B57", ls="--", lw=0.8, label=f"T_Si = {t.T_si:.4f}")
    a1.axhline(t.tau_ul, color="#C8283C", ls=":", lw=0.8, label=f"τ_UL = {t.tau_ul:.4f}")
    a1.set_xscale("log")
    a1.set_xlabel("α [sr]")
    a1.set_ylabel("T(α)")
    a1.legend(fontsize=7, title=f"p_cS ≥ {t.p_cs_lower:.4f}", title_fontsize=7)
    rc = 0.5 * (t.ring_edges_px[:-1] + t.ring_edges_px[1:])
    a2.plot(rc, t.ring_empty, "s-", ms=3, lw=0.8, label="empty beam")
    a2.errorbar(rc, t.ring_sample, t.ring_sample_sigma, fmt="o-", ms=3, lw=0.8, label="sample")
    sc, ss_ = t.scattered()
    a2.errorbar(rc, np.abs(sc), ss_, fmt=".", ms=3, lw=0.6, label="|sample - τ_UL · empty|")
    a2.plot(rc, t.ring_background, "k--", lw=0.6, label="background subtracted")
    a2.set_xscale("log")
    a2.set_yscale("log")
    a2.xaxis.set_minor_formatter(plt.NullFormatter())
    a2.set_xlabel("ring radius [px]")
    a2.set_ylabel("per pixel and monitor")
    a2.legend(fontsize=7)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------- p_cS from a curve
@dataclass
class Probability:
    """p_cS = d ∫ dΣ/dΩ dΩ split into the measured part and the two extrapolations."""
    p: float
    sigma: float                   # counting statistics of the measured part only
    measured: float
    low_q: float                   # 0 .. q_min (Guinier or constant)
    high_q: float                  # q_max .. ∞ (Porod q^-4)
    guinier_rg_A: Optional[float]
    porod_B: Optional[float]
    notes: List[str] = field(default_factory=list)

    @property
    def extrapolated_fraction(self) -> float:
        return (self.low_q + self.high_q) / self.p if self.p > 0 else 0.0


def _domega_dq(q, lam):
    """dΩ/dq for a ring at q (exact, elastic): 2π sin 2θ · d2θ/dq."""
    x = np.clip(q * lam / (4 * np.pi), 0, 1 - 1e-12)
    tt = 2 * np.arcsin(x)
    return 2 * np.pi * np.sin(tt) * (lam / (2 * np.pi)) / np.sqrt(1 - x**2)


def scattering_probability(q, dsdo, thickness_cm: float, wavelength_A: float, sigma=None,
                           background: float = 0.0, n_low: int = 5, n_high: int = 5,
                           low: str = "guinier", high: str = "porod") -> Probability:
    """Coherent small-angle scattering probability of a sample from its cross-section.

    ``q`` in 1/Å, ``dsdo`` in cm⁻¹ sr⁻¹ (coherent part; give the flat incoherent level as
    ``background`` to remove it), thickness in cm. Extrapolations: ``low`` = "guinier" (fit of the
    first ``n_low`` points, constant if the fit fails) or "constant" or "none"; ``high`` = "porod"
    (B q⁻⁴ from the last ``n_high`` points) or "none"."""
    q = np.asarray(q, float)
    I = np.asarray(dsdo, float) - background
    s = np.zeros_like(I) if sigma is None else np.asarray(sigma, float)
    o = np.argsort(q)
    q, I, s = q[o], I[o], s[o]
    good = np.isfinite(I) & np.isfinite(q) & (q > 0)
    q, I, s = q[good], I[good], s[good]
    lam = float(wavelength_A)
    w = _domega_dq(q, lam)
    # trapezoid with its variance
    dq = np.diff(q)
    c = np.zeros_like(q)
    c[:-1] += 0.5 * dq
    c[1:] += 0.5 * dq
    measured = float(np.sum(c * w * I))
    var = float(np.sum((c * w * s)**2))
    notes = []
    low_q, rg = 0.0, None
    if low != "none":
        qq, Il = q[:n_low], I[:n_low]
        I0, rg2 = float(Il[0]), 0.0
        if low == "guinier" and np.all(Il > 0) and qq.size >= 3:
            b, a = np.polyfit(qq**2, np.log(Il), 1)
            if b < 0:
                I0, rg2 = float(np.exp(a)), -3 * b
                rg = float(np.sqrt(rg2))
            else:
                notes.append("no Guinier region at the lowest q (rising or flat): constant extrapolation")
        g = np.linspace(0, q[0], 400)
        f = I0 * np.exp(-g**2 * rg2 / 3) if rg2 > 0 else np.full_like(g, I[0])
        low_q = float(np.trapezoid(f * _domega_dq(g, lam), g))
    high_q, B = 0.0, None
    if high == "porod":
        qq, Ih = q[-n_high:], I[-n_high:]
        if np.all(Ih > 0):
            B = float(np.mean(Ih * qq**4))
            qmax_geom = 4 * np.pi / lam * (1 - 1e-9)
            g = np.geomspace(q[-1], qmax_geom, 2000)
            high_q = float(np.trapezoid(B * g**-4 * _domega_dq(g, lam), g))
        else:
            notes.append("negative intensities at the highest q: no Porod tail (background too high?)")
    p = thickness_cm * (measured + low_q + high_q)
    out = Probability(p, thickness_cm * np.sqrt(var), thickness_cm * measured, thickness_cm * low_q,
                      thickness_cm * high_q, rg, B, notes)
    if out.extrapolated_fraction > 0.3:
        out.notes.append(f"{100 * out.extrapolated_fraction:.0f} % of p_cS is extrapolated: "
                         "measure a wider q range (lower q at a longer distance)")
    return out


def scale_probability(p: float, d_from: float, d_to: float, lam_from: float, lam_to: float) -> float:
    """p_cS at another thickness and wavelength, single scattering: p ∝ d λ² (dΣ/dΩ(q) fixed)."""
    return p * (d_to / d_from) * (lam_to / lam_from)**2


def thickness_for(p_target: float, p: float, d: float, lam: float = 1.0, lam_to: Optional[float] = None) -> float:
    """Thickness that gives ``p_target`` (e.g. 0.1) at wavelength ``lam_to`` (default: the same)."""
    lam_to = lam if lam_to is None else lam_to
    return d * p_target / p * (lam / lam_to)**2


# ---------------------------------------------------------------------------- regime
@dataclass
class Assessment:
    regime: str                    # weak | moderate | strong
    p_cs: float
    formula: str                   # which form of the reduction formula is adequate
    actions: List[str]


def assess(p_cs: float, thickness_cm: Optional[float] = None, wavelength_A: Optional[float] = None,
           background_fraction: Optional[float] = None, plateau_reached: Optional[bool] = None,
           weak: float = 0.05, strong: float = 0.3, target: float = 0.1) -> Assessment:
    """Regime of a sample and the actions for the reduction and the measurement.

    ``background_fraction``: (EB - Cd) / (S - Cd) summed near the beam stop; when the
    background is comparable to the sample signal the (1 - p_cS) factor changes the shape, not only
    the scale (Strunz et al., fig. 2)."""
    acts: List[str] = []
    if plateau_reached is False:
        acts.append("the wide-angle transmission has not reached its plateau: T_Si is a lower limit")
    if p_cs < weak:
        regime = "weak"
        formula = "eq. 9 with p_cS ≈ 0 (BerSANS default p(S) = 0): T_Si ≈ τ_S"
        acts.append("multiple scattering negligible; transmission through the beam spot is adequate")
    elif p_cs < strong:
        regime = "moderate"
        formula = "eq. 8: sample / (T_EB T_Si), background × (1 - p_cS); BerSANS: p(S) = p_cS"
        acts.append(f"use p_cS = {p_cs:.3f} for the background term; neglecting it shifts the absolute "
                    f"scale by up to {100 * p_cs:.0f} %")
        acts.append("measure the transmission over a wide solid angle (T_Si, plateau), not only the beam spot")
        acts.append("iterate: reduce -> p_cS from the curve -> reduce (self_consistent_probability)")
    else:
        regime = "strong"
        formula = "eq. 8 with modelled p_cS; the measured curve is a multiple-scattering curve"
        acts.append("the shape is broadened by multiple scattering (Schelten & Schmatz 1980): I(0), Rg and "
                    "the invariant are not those of single scattering")
        acts.append("τ_S cannot be measured (beam spot contaminated): take p_cS from the model of the "
                    "data analysis or from 1 - τ_UL/T_Si as a lower limit")
        acts.append("check with a thickness or wavelength series: the curves divided by d must agree "
                    "(grouping.equivalence on I/d); broadening that grows with d λ² is multiple scattering")
        acts.append("analyse with multiple scattering in the model, or measure a thinner sample")
    if p_cs >= weak and thickness_cm is not None:
        d_new = thickness_for(target, p_cs, thickness_cm)
        acts.append(f"p_cS = {target:g} needs d ≈ {10 * d_new:.2f} mm at the same wavelength")
        if wavelength_A is not None:
            lam_new = wavelength_A * np.sqrt(target / p_cs)
            acts.append(f"or λ ≈ {lam_new:.1f} Å at the same thickness (p ∝ d λ²)")
    if background_fraction is not None and background_fraction > 0.2 and p_cs >= weak:
        acts.append(f"the background is {100 * background_fraction:.0f} % of the signal near the beam stop: "
                    "a wrong (1 - p_cS) changes the curve shape, not only the scale")
    return Assessment(regime, float(p_cs), formula, acts)


def self_consistent_probability(reduce: Callable[[float], tuple], thickness_cm: float, wavelength_A: float,
                                p0: float = 0.0, tol: float = 1e-3, max_iter: int = 30, **kw):
    """Solve the integral equation for p_cS by iteration.

    ``reduce(p)`` returns (q, dΣ/dΩ, σ) in absolute units for a given p_cS. Returns
    (p_cS, Probability of the last curve, list of iterates). Raises if it does not converge."""
    p, hist = float(p0), [float(p0)]
    for _ in range(max_iter):
        q, I, s = reduce(p)
        pr = scattering_probability(q, I, thickness_cm, wavelength_A, sigma=s, **kw)
        p_new = min(max(pr.p, 0.0), 1.0)
        hist.append(p_new)
        if abs(p_new - p) < tol:
            return p_new, pr, hist
        p = p_new
    raise RuntimeError(f"p_cS did not converge: {hist[-5:]}")
