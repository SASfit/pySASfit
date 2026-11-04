# SPDX-License-Identifier: BSD-3-Clause
"""Reduction of a row of the reduction table: first the 2D image, pixel by pixel, then the radial
(or sector) average.

Per pixel ij, every run corrected for dead time and attenuator and normalised to the monitor (or
time), the runs of a row summed (:func:`reduce.combined_normalised_counts`):

    N_ij = A · (S - Cd)/T_S  -  (1 - p_cS) · (SB - Cd)/T_SB                    (sample - background)

with water (Strunz et al. 2000, eq. 8; BerSANS SANSAniDo eq. 2):

    W_ij = (W - Cd)/T_W  -  (WB - Cd)/T_WB
    dΣ/dΩ_ij = N_ij / W_ij · K0 (1 - τ_W) / (4π τ_W d_S)        [1/cm/sr],  τ_W = T_W / T_WB

The water has the same geometry, so solid angle and pixel efficiency cancel.

Where water takes too long (long distances, long wavelengths) the absolute scale comes from the
**direct beam** of the same geometry (beam stop out, usually attenuated) and a detector
**efficiency map** ε_ij (mean 1; usually from the water of a shorter distance,
:func:`~pysasfit.sasred.reduce.efficiency_from_runs`):

    Φ = Σ_beam (DB - Cd)_kl / ε_kl                    [counts/monitor, attenuator corrected]
    dΣ/dΩ_ij = N_ij / (ε_ij ΔΩ_ij d_S Φ) · a_DB/a_S       [1/cm/sr]

The absolute efficiency of the detector and its λ dependence cancel between Φ and N; only the
relative pixel efficiency is taken from the map. The attenuator of the direct beam enters with its
factor from the instrument file (its uncertainty is propagated when the file gives one).

Without water and direct beam the image is put on a relative scale per steradian and thickness:

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
from .core.geometry import PixelGeometry, run_geometry
from .core.resolution import resolution_2d
from .data.uarray import UArray, constant
from .reduce import ReducedRun, _SD, _transmission, combined_normalised_counts, load_instrument

__all__ = ["Reduced2D", "reduce_row_2d", "average", "reduce_row", "water_g", "water_denominator",
           "water_scaling_factor", "load_efficiency", "direct_beam_flux", "direct_beam_attenuator", "DirectBeam"]


def load_efficiency(spec: str):
    """Efficiency map from ``"file.h5"`` (a map of :meth:`Efficiency.save`, or the newest map of an
    efficiency archive) or ``"file.h5::entry"`` (one map of an archive)."""
    from .core.efficiency import Efficiency
    path, _, entry = str(spec).partition("::")
    return Efficiency.load(path.strip(), entry.strip() or None)


@dataclass
class DirectBeam:
    """The integrated direct beam of :func:`direct_beam_flux`."""
    flux: UArray                  # scalar, counts/monitor (or counts/s), attenuator and efficiency corrected
    centre_px: tuple              # (x, y) centre of gravity of the beam
    radius_mm: float
    fraction: float               # share of the counts inside the radius among those inside the largest disc
    runs: List[int]
    attenuators: List[str]
    notes: List[str] = field(default_factory=list)

    def describe(self) -> str:
        f = self.flux
        s = float(np.sqrt(f.variance().sum())) if f.terms else 0.0
        return (f"direct beam Φ = {float(f.value):.6g} ± {s:.2g} {f.unit or ''} (runs {self.runs}, attenuator "
                f"{', '.join(self.attenuators)}), summed within {self.radius_mm:.1f} mm of "
                f"({self.centre_px[0]:.2f}, {self.centre_px[1]:.2f}) px")


def direct_beam_attenuator(attenuator: str, wavelength_A: float, inst: dict, by_hand: Optional[float] = None):
    """Transmission of the attenuator used for the direct beam: (value, relative σ or None, text).
    ``by_hand``: that value; else the water calibration of the instrument file (``water_check``) when it
    was made within 5 % of this wavelength; else the nominal factor."""
    att = (inst.get("attenuator") or {}).get(str(attenuator)) or {}
    nominal = float(att.get("transmission", 1.0))
    if str(attenuator) == "0":
        return 1.0, None, "no attenuator"
    if by_hand:
        return float(by_hand), None, f"attenuator {attenuator}: transmission {float(by_hand):.5g} given by hand"
    chk = att.get("water_check")
    if chk and abs(10 * float(chk["wavelength_nm"]) - wavelength_A) <= 0.05 * wavelength_A:
        v = float(chk["transmission"])
        return v, chk.get("uncertainty_rel"), (f"attenuator {attenuator}: transmission {v:.5g} from the water "
                                               f"calibration at {chk['wavelength_nm']} nm (nominal {nominal:.5g})")
    rel = att.get("uncertainty")
    return nominal, (float(rel) / nominal if rel else None), (
        f"attenuator {attenuator}: nominal transmission {nominal:.5g} of the instrument file" +
        ("" if rel else " (uncertainty unknown – the absolute scale is only as good as this factor)"))


def direct_beam_flux(runs, inst, centre_px=None, efficiency=None, blocked: Optional[UArray] = None,
                     radius_mm: Optional[float] = None, normalise_by: str = "monitor",
                     enclosed: float = 0.995, attenuator_transmission: Optional[float] = None) -> DirectBeam:
    """Integrated direct beam Φ = Σ (DB - Cd)/ε over a disc around the beam.

    ``runs``: raw runs of the direct beam (beam stop out; summed with their monitors, each corrected for
    dead time and attenuator); ``blocked``: the normalised Cd image (subtracted); ``efficiency``: the map
    (pixels divided by it); ``centre_px``: centre of the disc (default: centre of gravity of the beam);
    ``radius_mm``: radius of the disc, default automatic: the radius that holds ``enclosed`` of the counts
    inside the largest disc that fits on the detector, plus two pixels; ``attenuator_transmission``: the
    transmission of the attenuator by hand (default: :func:`direct_beam_attenuator`)."""
    from .core.beamcentre import masked_centre_of_gravity
    notes: List[str] = []
    runs = list(runs)
    raw0 = runs[0]
    from .reduce import normalised_counts
    m = np.zeros(raw0.counts.shape, np.uint32)
    mname = inst["monitor"]["normaliser"]
    total_w, DB = 0.0, None
    for q in runs:
        x, _, m, _ = normalised_counts(q, inst, [], m, normalise_by)    # divided by the nominal attenuator factor
        nominal = float(inst["attenuator"][str(q.attenuator)]["transmission"]) if str(q.attenuator) != "0" else 1.0
        a, rel, text = direct_beam_attenuator(q.attenuator, q.wavelength_A, inst, attenuator_transmission)
        notes.append(text)
        if a != nominal:
            x = x * (nominal / a)
        if rel:
            x = x / UArray.scalar(1.0, float(rel), f"attenuator_db:{q.attenuator}", "attenuator", systematic=True)
        w = float(q.monitors[mname]) if normalise_by == "monitor" else float(q.counting_time_s)
        DB = x * w if DB is None else DB + x * w
        total_w += w
    DB = DB / total_w
    if blocked is not None:
        DB = DB - blocked
    if efficiency is not None:
        if efficiency.shape != DB.shape:
            raise ValueError(f"efficiency map {efficiency.shape} and detector {DB.shape} differ in shape")
        DB = DB / efficiency.value
    v = np.nan_to_num(np.asarray(DB.value, float))
    if centre_px is None:
        c = masked_centre_of_gravity(np.clip(v, 0, None), radius_px=8.0)
        centre_px = (c.x, c.y)
    xc, yc = map(float, centre_px)
    ny, nx = v.shape
    px, py = (raw0.pixel_size_m, raw0.pixel_size_m) if np.isscalar(raw0.pixel_size_m) else raw0.pixel_size_m
    px, py = 1e3 * float(px), 1e3 * float(py)
    iy, ix = np.mgrid[0:ny, 0:nx].astype(float)
    r = np.hypot((ix - xc) * px, (iy - yc) * py)
    r_fit = min((xc + 0.5) * px, (nx - 0.5 - xc) * px, (yc + 0.5) * py, (ny - 0.5 - yc) * py)
    if r_fit <= 0:
        raise ValueError(f"direct beam centre ({xc:.1f}, {yc:.1f}) px is not on the detector")
    inside_max = r <= r_fit
    total = v[inside_max].sum()
    if total <= 0:
        raise ValueError(f"direct beam runs {[q.run for q in runs]}: no counts around the beam")
    if radius_mm is None:
        order = np.argsort(r[inside_max])
        cum = np.cumsum(v[inside_max][order])
        k = int(np.searchsorted(cum, enclosed * total))
        radius_mm = float(r[inside_max][order][min(k, order.size - 1)]) + 2 * max(px, py)
        how = f"automatic: {100 * enclosed:.1f} % of the counts within {r_fit:.0f} mm, + 2 pixels"
    else:
        how = "given"
    disc = r <= radius_mm
    frac = float(v[disc & inside_max].sum() / total)
    if radius_mm > r_fit:
        notes.append(f"direct beam: the disc of {radius_mm:.1f} mm reaches the detector edge "
                     f"(largest disc on the detector {r_fit:.0f} mm)")
    if np.any(m[disc] & 8):
        notes.append("direct beam: dead-time factor above the limit in the beam – measure with a stronger "
                     "attenuator or a shorter time")
    flux = DB.sum(disc)
    flux.unit = "counts/monitor" if normalise_by == "monitor" else "counts/s"
    atts = sorted({str(q.attenuator) for q in runs})
    notes.append(f"direct beam radius {radius_mm:.1f} mm ({how}), {100 * frac:.2f} % of the counts within "
                 f"{r_fit:.0f} mm")
    return DirectBeam(flux, (xc, yc), float(radius_mm), frac, [q.run for q in runs], atts, notes)


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
    beam_profile: object = None   # direct-beam profile of the geometry (None: no resolution)
    resolution: Optional[dict] = None   # 2D resolution per pixel (resolution_2d), None without beam profile

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


TILT_TOL_DEG = 0.25        # without a dλ/λ(tilt) table
DLAM_TOL_REL = 0.02        # with it: dλ/λ, and so roughly the flux, differs by more than 2 %


def _tilt_checks(table, row, tilts, normalise_by, notes, inst=None) -> list:
    """Runs of one reduction at different selector tilts. The tilt sets dλ/λ, and the flux grows roughly
    in proportion to dλ/λ: with the monitor normalisation the flux cancels (the monitor sees the same
    beam), with the time normalisation it does not. Returns Missing-input items (dicts); adds notes."""
    if not tilts:
        return []
    out = []
    s_t = [tilts[n] for n in row.used if n in tilts]
    if not s_t:
        return []
    t0 = s_t[0]
    table_dl = ((inst or {}).get("wavelength") or {}).get("selector_tilt")

    def differs(a, b):
        # with the calibration table: dλ/λ (and so the flux) differs by more than 2 %; else the tilt by 0.25°
        if table_dl:
            from .core.resolution import dlam_from_tilt
            return abs(dlam_from_tilt(a, table_dl)[0] / dlam_from_tilt(b, table_dl)[0] - 1) > DLAM_TOL_REL
        return abs(a - b) > TILT_TOL_DEG
    if any(differs(t, t0) for t in s_t):
        txt = (f"sample runs at different selector tilts ({', '.join(f'{t:+.1f}°' for t in sorted(set(s_t)))}): "
               f"summed; dλ/λ of the first run ({t0:+.1f}°) used for the resolution")
        notes.append(txt)
        out.append(dict(role="sample", index=table.samples.index(row) if row in table.samples else None,
                        label=f"{row.label} ({row.info})", field="selector tilt", text=txt, severity="check"))
    for role, r in table.corrections.items():
        if role == "blocked_beam" or not r.used or role in (getattr(table, "ignored", ()) or ()):
            continue
        other = sorted({round(tilts[n], 1) for n in r.used if n in tilts and differs(tilts[n], t0)})
        if not other:
            continue
        ts = ", ".join(f"{t:+.1f}°" for t in other)
        mon = (inst or {}).get("monitor") or {}
        after = mon.get("normaliser_after_selector")
        mname = mon.get("normaliser", "monitor")
        if normalise_by == "monitor" and after:
            txt = (f"{r.label} measured at selector tilt {ts}, the sample at {t0:+.1f}°: the different flux "
                   f"cancels in the normalisation by {mname} (behind the selector); dλ/λ of the sample used")
            sev = "info"
        elif normalise_by == "monitor":
            where = ("is in front of the selector" if after is False else
                     "may be in front of the selector (not known: needs runs at two wavelengths or tilts)")
            txt = (f"{r.label} measured at selector tilt {ts}, the sample at {t0:+.1f}°: the flux changes with "
                   f"the tilt (≈ ∝ dλ/λ) and {mname} {where}, so the normalisation may not correct it – "
                   "use runs of the same tilt")
            sev = "check"
        else:
            txt = (f"{r.label} measured at selector tilt {ts}, the sample at {t0:+.1f}°: the flux changes with "
                   "the tilt (≈ ∝ dλ/λ) and the time normalisation does not correct it – normalise by monitor "
                   "or use runs of the same tilt")
            sev = "check"
        notes.append(txt)
        out.append(dict(role=role, index=None, label=r.label, field="selector tilt", text=txt, severity=sev))
    return out


def reduce_row_2d(table, row, read: Callable[[int], object], beam_center_px, mask=None, instrument=None,
                  normalise_by: str = "monitor", k0: Optional[float] = None, efficiency=None,
                  scale: Optional[str] = None, beam_profile=None) -> Reduced2D:
    """2D reduction of a sample row of a :class:`~pysasfit.sasred.reduction_table.ReductionTable`.

    ``read(run)``: the raw run (e.g. :func:`~pysasfit.sasred.io.sinq.read_sans_i` with checksum);
    ``mask``: uint32 mask of the geometry (default: suggested from the summed sample counts);
    ``k0``: forward excess of the water (Strunz et al. eq. 6); None: g(λ) of the instrument file with its
    uncertainty as a systematic source (:func:`water_g`); a number: that value; ``efficiency``: the
    detector efficiency map (:class:`~pysasfit.sasred.core.efficiency.Efficiency`), needed for the
    direct-beam scale, used also for a relative scale, not with water; ``scale``: "water",
    "direct_beam" or "relative" (default: :meth:`ReductionTable.scale_method`); ``beam_profile``: the
    direct-beam profile of the geometry for the resolution (:mod:`pysasfit.sasred.core.beamprofile`);
    without it no resolution is determined (noted)."""
    notes: List[str] = []
    if not row.used:
        raise ValueError(f"{row.label} ({row.info}): no runs switched on")
    corr = table.corrections
    tilts: dict = {}                    # run -> selector tilt of every run read (flux and dλ/λ depend on it)
    _read_raw = read

    def read(n):
        r = _read_raw(n)
        t = getattr(r, "selector_tilt_deg", None)
        if t is not None:
            tilts[getattr(r, "run", n)] = float(t)
        return r
    S_runs = _runs(row.used, read)
    raw0 = S_runs[0]
    inst = instrument or load_instrument(raw0)
    if mask is None:
        if raw0.panel is not None:
            mask, _ = mk.default_mask(raw0, beam_center_px)
            notes.append(f"{raw0.bank} detector: tube ends and dead tubes masked")
        else:
            mask, _ = mk.suggest_mask(np.sum([r.counts for r in S_runs], axis=0), beam_center_px)
            notes.append("mask suggested from the summed sample counts")
    mask = np.asarray(mask, np.uint32)
    S, mask = combined_normalised_counts(S_runs, inst, notes, mask, normalise_by)
    unit = "counts/monitor" if normalise_by == "monitor" else "counts/s"

    ignored = set(getattr(table, "ignored", ()) or ())

    def combined(r, what):
        nonlocal mask
        if not r.used:
            return None
        if r.role in ignored:
            notes.append(f"{what}: switched off in the table, not used")
            return None
        rs = _runs(r.used, read)
        x, mask = combined_normalised_counts(rs, inst, notes, mask, normalise_by)
        other = [q.run for q in rs if abs(q.SD_m - raw0.SD_m) > 0.01 * raw0.SD_m
                 or abs(q.wavelength_A - raw0.wavelength_A) > 0.02 * raw0.wavelength_A]
        if other:
            raise ValueError(f"{what}: runs {other} have another distance or wavelength than the sample "
                             f"(SD {raw0.SD_m:.2f} m, λ {raw0.wavelength_A:.2f} Å)")
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

    geo = run_geometry(raw0, beam_center_px)
    if raw0.bank:
        from .io.sans_llb import panel_offsets
        notes.append(f"SANS-LLB {raw0.bank} detector at {geo.SD_m:.4f} m: geometry from the pixel positions + "
                     f"offsets (dx, dy, dz) = {tuple(round(v, 2) for v in panel_offsets(raw0.bank))} mm")
        if corr["water"].used:
            notes.append(f"{raw0.bank} detector: water per pixel at 2θ up to "
                         f"{np.degrees(geo.two_theta.max()):.0f}°; its angular dependence (inelastic, "
                         f"self-absorption) is not corrected")
    if getattr(raw0, "SD_offset_m", 0.0):
        notes.append(f"detector distance {raw0.SD_effective_m:.4f} m = recorded {raw0.SD_m:.4f} m "
                     f"{1000 * raw0.SD_offset_m:+.1f} mm offset")
    d = row.scaling
    if not d:
        notes.append("no sample thickness: d = 1 cm")
        d = 1.0
    method = scale or (table.scale_method() if hasattr(table, "scale_method") else "water")
    W = combined(corr["water"], "water") if method == "water" else None
    if method == "direct_beam" and corr["water"].used:
        notes.append("water of this geometry not used: absolute scale from the direct beam and the efficiency map")
    absolute = W is not None
    k0_used = None
    db_info = None
    if method == "direct_beam":
        if raw0.panel is not None:
            raise ValueError(f"{raw0.bank} detector: the direct-beam scale needs the efficiency of the wide-angle "
                             "panels relative to the central detector – not available yet; use water")
        if efficiency is None:
            raise ValueError("the direct-beam scale needs a detector efficiency map (Absolute scale → efficiency map)")
        if efficiency.shape != S.shape:
            raise ValueError(f"efficiency map {efficiency.shape} and detector {S.shape} differ in shape")
        dbrow = corr["direct_beam"]
        if not dbrow.used:
            raise ValueError("the direct-beam scale needs direct-beam runs (row Direct beam)")
        db_runs = _runs(dbrow.used, read)
        other = [q.run for q in db_runs if abs(q.SD_m - raw0.SD_m) > 0.01 * raw0.SD_m
                 or abs(q.wavelength_A - raw0.wavelength_A) > 0.02 * raw0.wavelength_A
                 or abs(getattr(q, "coll_m", 0) - getattr(raw0, "coll_m", 0)) > 0.01 * max(getattr(raw0, "coll_m", 0), 1e-9)]
        if other:
            raise ValueError(f"direct beam: runs {other} have another distance, wavelength or collimation than the "
                             f"sample (SD {raw0.SD_m:.2f} m, λ {raw0.wavelength_A:.2f} Å)")
        db = direct_beam_flux(db_runs, inst, efficiency=efficiency, blocked=B, normalise_by=normalise_by,
                              radius_mm=getattr(table, "direct_beam_radius_mm", None),
                              attenuator_transmission=getattr(table, "direct_beam_attenuator", None))
        notes.extend(db.notes)
        notes.append(db.describe())
        em = efficiency.meta or {}
        src = f"efficiency map {em.get('file', '')}" + (f" (water runs {em.get('water_runs')}, SD "
                                                        f"{em.get('SD_m', float('nan')):.2f} m, λ "
                                                        f"{em.get('wavelength_A', float('nan')):.2f} Å)"
                                                        if em.get("water_runs") else "")
        notes.append(src)
        lam_e = em.get("wavelength_A")
        if lam_e and abs(float(lam_e) - raw0.wavelength_A) > 0.02 * raw0.wavelength_A:
            notes.append(f"efficiency map measured at λ = {float(lam_e):.2f} Å, used at {raw0.wavelength_A:.2f} Å: "
                         "the relative pixel efficiency is assumed independent of λ")
        mask = mask | efficiency.mask_bits()
        if getattr(table, "efficiency_mask_filled", False):
            mask = mask | (efficiency.filled.astype(np.uint32) * mk.EFFICIENCY)
            notes.append(f"{int(efficiency.filled.sum())} pixels only estimated in the efficiency map (masked or "
                         "rejected in its water run) are masked")
        elif efficiency.filled.any():
            notes.append(f"{int(efficiency.filled.sum())} pixels of the efficiency map are estimates (row × column; "
                         "masked or rejected in its water run, e.g. behind its beam stop)")
        a_s, a_db = row.aperture_cm2, dbrow.aperture_cm2
        if a_s and a_db:
            area = float(a_db) / float(a_s)
            notes.append(f"illuminated areas: direct beam {a_db:g} cm², sample {a_s:g} cm² → × {area:.4g}")
        else:
            area = 1.0
            if a_s or a_db:
                notes.append("aperture given for only one of sample and direct beam: same area assumed")
        img = N / efficiency.value / geo.d_omega / db.flux * (area / d)
        unit = "1/cm/sr"
        absolute = True
        db_info = dict(flux=float(db.flux.value), radius_mm=db.radius_mm, centre_px=db.centre_px, runs=db.runs,
                       attenuators=db.attenuators, efficiency=em.get("file", ""))
        notes.append(f"absolute scale from the direct beam: N/(ε ΔΩ d Φ), d = {d:g} cm")
    elif absolute:
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
        notes.append("no water" + (" and no direct beam" if method != "relative" else "") +
                     ": relative scale per steradian and cm of sample, not absolute")
        img = N / geo.d_omega / d
        unit = unit + "/sr/cm"
        if efficiency is not None:
            img = img / efficiency.value
            mask = mask | efficiency.mask_bits()
            notes.append("divided by the detector efficiency")
    img.unit = unit
    notes = list(dict.fromkeys(notes))
    missing = []
    if hasattr(table, "missing_inputs"):
        from dataclasses import asdict
        missing = [asdict(x) for x in table.missing_inputs([row])]
    for x in _tilt_checks(table, row, tilts, normalise_by, notes, inst):
        missing.append(x)
    resolution = None
    dl = raw0.dlam_over_lam_fwhm
    if beam_profile is None:
        from .reduce import NO_RESOLUTION
        notes.append(NO_RESOLUTION)
    elif dl is None:
        notes.append("dλ/λ not in the raw file: resolution (Qdev) not determined")
    else:
        resolution = resolution_2d(geo, dl, inst["wavelength"]["distribution"], beam_profile, raw0.pixel_size_m,
                                   _SD(raw0))
    return Reduced2D(row.label, row.info, list(row.used), img, mask, geo, unit, raw0, absolute, notes,
                     parameters=dict(beam_center_px=tuple(map(float, beam_center_px)), normalise_by=normalise_by,
                                     k0=k0_used, thickness_cm=d, attenuator_factor=A, p_cs=p,
                                     aperture_cm2=row.aperture_cm2, missing=missing, scale=method if absolute or method == "relative"
                                     else "relative", direct_beam=db_info,
                                     beam_profile=None if beam_profile is None else beam_profile.to_dict(),
                                     config=table.config),
                     beam_profile=beam_profile, resolution=resolution)


def average(r2: Reduced2D, edges, sector_deg=None, dlam_over_lam_fwhm=None, instrument=None) -> ReducedRun:
    """Radial (or sector) average of a 2D result."""
    inst = instrument or load_instrument(r2.raw)
    b = radial_binning(r2.geometry.q, r2.mask == 0, edges, phi=r2.geometry.phi, sector_deg=sector_deg)
    if b.A.shape[0] == 0:
        raise ValueError("no valid pixels inside the q range")
    I = b.apply(r2.image)
    I.unit = r2.unit
    raw = r2.raw
    dl = raw.dlam_over_lam_fwhm if raw.dlam_over_lam_fwhm is not None else dlam_over_lam_fwhm
    from .reduce import NO_RESOLUTION, q_resolution_1d
    qs, res_note = q_resolution_1d(b, r2.geometry, dl, inst, r2.beam_profile,
                                   getattr(raw, "dlam_source", "") if raw.dlam_over_lam_fwhm is not None else "given")
    notes = [n for n in r2.notes if n != NO_RESOLUTION] + [res_note]
    return ReducedRun(raw.run, b.q_mean, I, qs, b, r2.geometry, r2.image, 1.0, r2.mask, raw,
                      parameters=dict(r2.parameters, runs=r2.runs, label=r2.label, sample=r2.sample,
                                      sector_deg=sector_deg, dlam_over_lam_fwhm=dl, selector_tilt_deg=getattr(raw, "selector_tilt_deg", None)),
                      notes=notes)


def reduce_row(table, row, read, beam_center_px, edges, mask=None, sector_deg=None, **kw):
    """2D reduction, then the average: (Reduced2D, ReducedRun)."""
    r2 = reduce_row_2d(table, row, read, beam_center_px, mask=mask, **kw)
    e = edges(r2) if callable(edges) else edges
    return r2, average(r2, e, sector_deg=sector_deg)
