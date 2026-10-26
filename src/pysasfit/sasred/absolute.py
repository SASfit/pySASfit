# SPDX-License-Identifier: BSD-3-Clause
"""Reduction of a row of the reduction table: first the 2D image, pixel by pixel, then the radial
(or sector) average.

Per pixel ij, every run corrected for dead time and attenuator and normalised to the monitor (or
time), the runs of a row summed (:func:`reduce.combined_normalised_counts`):

    N_ij = A · (S - Cd)/T_S  -  (1 - p_cS) · (SB - Cd)/T_SB                    (sample - background)

with water (Strunz et al. 2000, eq. 8; BerSANS SANSAniDo eq. 2):

    W_ij = (W - Cd)/T_W  -  (WB - Cd)/T_WB
    dΣ/dΩ_ij = N_ij / W_ij · K0 (1 - τ_W) / (4π τ_W d_S)        [1/cm/sr],  τ_W = T_W / T_WB

The water has the same geometry, so solid angle and pixel efficiency cancel. Without water the
image is put on a relative scale per steradian and thickness:

    I_ij = N_ij / (ΔΩ_ij · d_S [· η_ij])                        [counts/monitor/sr/cm]

Missing measurements do not stop the reduction; each one is stated in the notes: no Cd → nothing
subtracted, no sample background → no background subtracted, no water background → water not
corrected for its cell, no transmission → T = 1, no thickness → d = 1 cm.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

import numpy as np

from .core import mask as mk
from .core.averaging import radial_binning
from .core.geometry import PixelGeometry, pixel_geometry
from .core.resolution import sigma_q_phase1
from .data.uarray import UArray, constant
from .reduce import ReducedRun, _L2, _transmission, combined_normalised_counts, load_instrument

__all__ = ["Reduced2D", "reduce_row_2d", "average", "reduce_row", "water_g", "water_denominator",
           "water_scaling_factor"]


def water_g(wavelength_A: float, instrument=None):
    """Forward excess g(λ) = K0 of the water (instrument file ``water.g_lambda``, λ in nm):
    (value, relative uncertainty, note)."""
    inst = instrument or load_instrument()
    g = (inst.get("water") or {}).get("g_lambda")
    if not g:
        return 1.0, None, "no g(λ) for the water in the instrument file: K0 = 1"
    lam = wavelength_A / 10.0
    c = g["formula"]
    val = float(sum(ci * lam**k for k, ci in enumerate(c)))
    lo, hi = g.get("range_nm", (None, None))
    note = f"water g(λ = {lam:.3f} nm) = {val:.4f}"
    if lo is not None and not (lo - 1e-9 <= lam <= hi + 1e-9):
        note += f" (extrapolated: measured {lo:g}–{hi:g} nm)"
    return val, g.get("uncertainty_rel"), note


def water_denominator(instrument=None) -> str:
    """Transmission in the denominator of the water factor: "water_and_cell" (T_W, BerSANS' scaling
    factor, the SANS-I setting) or "tau" (τ_W = T_W/T_WB, Strunz et al. 2000, eq. 8; used when the
    instrument file has no ``water.denominator``)."""
    inst = instrument or load_instrument()
    v = str((inst.get("water") or {}).get("denominator", "tau"))
    if v not in ("tau", "water_and_cell"):
        raise ValueError(f"water.denominator = {v!r}: 'tau' or 'water_and_cell'")
    return v


def water_scaling_factor(t_water: float, t_cell: Optional[float], aperture_cm2: Optional[float] = None,
                         g: float = 1.0, denominator: str = "tau") -> float:
    """g A_W (1 - τ_W)/(4π D), τ_W = T_W/T_WB, D = τ_W ("tau") or T_W ("water_and_cell"): the BerSANS
    "ScalingFactor(W)" (BerSANS' Calculate gives it for g = 1 and D = T_W)."""
    tau = t_water / t_cell if t_cell else t_water
    den = tau if denominator == "tau" else t_water
    return g * (aperture_cm2 or 1.0) * (1.0 - tau) / (4 * np.pi * den)


@dataclass
class Reduced2D:
    """2D result of one sample row."""
    label: str
    sample: str
    runs: List[int]
    image: UArray                 # per pixel
    mask: np.ndarray              # uint32 reason bits, 0 = valid
    geometry: PixelGeometry
    unit: str
    raw: object                   # first sample run
    absolute: bool
    notes: List[str] = field(default_factory=list)
    parameters: Dict[str, object] = field(default_factory=dict)

    @property
    def I(self):
        return self.image.value

    def idev(self):
        return self.image.std(systematic=False)


def _runs(rows_runs, read):
    return [read(r) for r in rows_runs]


def _trans(row, notes, what):
    if row.transmission is None:
        notes.append(f"{what} ({row.info or row.label}): no transmission, T = 1")
        return constant(1.0)
    key = f"transmission:{row.info or row.label}:{row.transmission:.6g}"
    return _transmission((row.transmission, row.transmission_sigma) if row.transmission_sigma else
                         row.transmission, key)


def reduce_row_2d(table, row, read: Callable[[int], object], beam_center_px, mask=None, instrument=None,
                  normalise_by: str = "monitor", k0: Optional[float] = None, efficiency=None) -> Reduced2D:
    """2D reduction of a sample row of a :class:`~pysasfit.sasred.reduction_table.ReductionTable`.

    ``read(run)``: the raw run (e.g. :func:`~pysasfit.sasred.io.sinq.read_sans_i` with checksum);
    ``mask``: uint32 mask of the geometry (default: suggested from the summed sample counts);
    ``k0``: forward excess of the water (Strunz et al. eq. 6); None: g(λ) of the instrument file with its
    uncertainty as a systematic source (:func:`water_g`); a number: that value; ``efficiency``: used
    only without water."""
    inst = instrument or load_instrument()
    notes: List[str] = []
    if not row.used:
        raise ValueError(f"{row.label} ({row.info}): no runs switched on")
    corr = table.corrections
    S_runs = _runs(row.used, read)
    raw0 = S_runs[0]
    if mask is None:
        mask, _ = mk.suggest_mask(np.sum([r.counts for r in S_runs], axis=0), beam_center_px)
        notes.append("mask suggested from the summed sample counts")
    mask = np.asarray(mask, np.uint32)
    S, mask = combined_normalised_counts(S_runs, inst, notes, mask, normalise_by)
    unit = "counts/monitor" if normalise_by == "monitor" else "counts/s"

    def combined(r, what):
        nonlocal mask
        if not r.used:
            return None
        rs = _runs(r.used, read)
        x, mask = combined_normalised_counts(rs, inst, notes, mask, normalise_by)
        other = [q.run for q in rs if abs(q.L2_m - raw0.L2_m) > 0.01 * raw0.L2_m
                 or abs(q.wavelength_A - raw0.wavelength_A) > 0.02 * raw0.wavelength_A]
        if other:
            raise ValueError(f"{what}: runs {other} have another distance or wavelength than the sample "
                             f"(L2 {raw0.L2_m:.2f} m, λ {raw0.wavelength_A:.2f} Å)")
        return x
    B = combined(corr["blocked_beam"], "Cd")
    if B is None:
        notes.append("no Cd (blocked beam): nothing subtracted for the beam-independent background")
    sub = (lambda x: x) if B is None else (lambda x: x - B)

    A = float(row.attenuator or 1.0)
    N = sub(S) / _trans(row, notes, "sample") * A
    if A != 1.0:
        notes.append(f"attenuation factor A = {A:g} applied to the sample")
    SB = combined(corr["sample_background"], "sample background")
    p = float(row.probability or 0.0)
    if SB is None:
        notes.append("no sample background: none subtracted")
    else:
        N = N - sub(SB) / _trans(corr["sample_background"], notes, "sample background") * (1.0 - p)
        if p:
            notes.append(f"background × (1 - p_cS), p_cS = {p:g}")

    geo = pixel_geometry(raw0.counts.shape, raw0.pixel_size_m, beam_center_px, _L2(raw0), raw0.wavelength_A)
    if getattr(raw0, "L2_offset_m", 0.0):
        notes.append(f"detector distance {raw0.L2_effective_m:.4f} m = recorded {raw0.L2_m:.4f} m "
                     f"{1000 * raw0.L2_offset_m:+.1f} mm offset")
    d = row.scaling
    if not d:
        notes.append("no sample thickness: d = 1 cm")
        d = 1.0
    W = combined(corr["water"], "water")
    absolute = W is not None
    k0_used = None
    if absolute:
        Wt = sub(W) / _trans(corr["water"], notes, "water")
        WB = combined(corr["empty_cell"], "water background")
        t_w = float(corr["water"].transmission or 1.0)
        t_ec = corr["empty_cell"].transmission if WB is not None else None
        if WB is None:
            notes.append("no water background (empty cell): the water is not corrected for its cell")
        else:
            Wt = Wt - sub(WB) / _trans(corr["empty_cell"], notes, "water background")
        tau = t_w / float(t_ec) if t_ec else t_w
        bad = (mask != 0) | ~np.isfinite(Wt.value) | (Wt.value <= 0)
        mask = mask | np.where((mask == 0) & bad, mk.EFFICIENCY, 0).astype(np.uint32)
        Wt = Wt + constant(np.where(bad, 1.0 - Wt.value, 0.0))          # finite where masked
        a_s, a_w = row.aperture_cm2, corr["water"].aperture_cm2
        if a_s and a_w:
            area = float(a_w) / float(a_s)
            notes.append(f"illuminated areas: water {a_w:g} cm², sample {a_s:g} cm² → × {area:.4g}")
        else:
            area = 1.0
            if a_s or a_w:
                notes.append("aperture given for only one of sample and water: same illuminated area assumed")
        if not (0 < tau < 1):
            notes.append(f"water transmission τ_W = {tau:g}: no absolute factor, result in units of water")
            factor = area / d
            unit = "1/cm (relative to water)"
        else:
            if k0 is None:
                gv, grel, gnote = water_g(raw0.wavelength_A, inst)
                notes.append(gnote)
                K = UArray.scalar(gv, gv * grel if grel else None, f"water_g:{raw0.wavelength_A:.3f}", "water",
                                  systematic=True) if grel else constant(gv)
                k0_used = gv
            else:
                K, k0_used = constant(float(k0)), float(k0)
            den_kind = water_denominator(inst)
            den = tau if den_kind == "tau" else t_w
            factor = K * ((1.0 - tau) * area / (4 * np.pi * den * d))
            unit = "1/cm/sr"
            dtxt = "τ_W" if den_kind == "tau" else "T_W (BerSANS)"
            notes.append(f"absolute scale from water: K0 (1 - τ_W)/(4π {dtxt} d) with τ_W = {tau:.4f}, "
                         f"T_W = {t_w:.4f}, d = {d:g} cm, K0 = {k0_used:.4f}")
            notes.append("BerSANS scaling factors: water "
                         f"{water_scaling_factor(t_w, t_ec, a_w, k0_used, den_kind):.5g}, "
                         f"sample {d * (float(a_s) if a_s and a_w else 1.0):.5g}")
        img = N / Wt * factor
    else:
        notes.append("no water: relative scale per steradian and cm of sample, not absolute")
        img = N / geo.d_omega / d
        unit = unit + "/sr/cm"
        if efficiency is not None:
            img = img / efficiency.value
            mask = mask | efficiency.mask_bits()
            notes.append("divided by the detector efficiency")
    img.unit = unit
    notes = list(dict.fromkeys(notes))
    return Reduced2D(row.label, row.info, list(row.used), img, mask, geo, unit, raw0, absolute, notes,
                     parameters=dict(beam_center_px=tuple(map(float, beam_center_px)), normalise_by=normalise_by,
                                     k0=k0_used, thickness_cm=d, attenuator_factor=A, p_cs=p,
                                     aperture_cm2=row.aperture_cm2,
                                     config=table.config))


def average(r2: Reduced2D, edges, sector_deg=None, dlam_over_lam_fwhm=None, instrument=None) -> ReducedRun:
    """Radial (or sector) average of a 2D result."""
    inst = instrument or load_instrument()
    b = radial_binning(r2.geometry.q, r2.mask == 0, edges, phi=r2.geometry.phi, sector_deg=sector_deg)
    if b.A.shape[0] == 0:
        raise ValueError("no valid pixels inside the q range")
    I = b.apply(r2.image)
    I.unit = r2.unit
    raw = r2.raw
    dl = raw.dlam_over_lam_fwhm if raw.dlam_over_lam_fwhm is not None else dlam_over_lam_fwhm
    if dl is None:
        raise ValueError("dlambda/lambda not in the raw file: give dlam_over_lam_fwhm")
    qs = sigma_q_phase1(b.q_mean, b.q_var_in_bin, dl, inst["wavelength"]["distribution"])
    notes = list(r2.notes) + ["Qdev: wavelength and in-bin pixel spread only; collimation term not yet included"]
    return ReducedRun(raw.run, b.q_mean, I, qs, b, r2.geometry, r2.image, 1.0, r2.mask, raw,
                      parameters=dict(r2.parameters, runs=r2.runs, label=r2.label, sample=r2.sample,
                                      sector_deg=sector_deg, dlam_over_lam_fwhm=dl),
                      notes=notes)


def reduce_row(table, row, read, beam_center_px, edges, mask=None, sector_deg=None, **kw):
    """2D reduction, then the average: (Reduced2D, ReducedRun)."""
    r2 = reduce_row_2d(table, row, read, beam_center_px, mask=mask, **kw)
    e = edges(r2) if callable(edges) else edges
    return r2, average(r2, e, sector_deg=sector_deg)
