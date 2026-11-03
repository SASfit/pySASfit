# SPDX-License-Identifier: BSD-3-Clause
"""Reduction of a single run to I(q):

    read -> counting error -> dead time -> attenuator -> monitor (or time)
         -> geometry and solid angle -> [detector efficiency] -> radial or sector average

Result: monitor-normalised intensity per steradian, I(q) in units of
counts / monitor / sr, with the exact first-order propagation of every
source. The detector efficiency comes from a water run
(:func:`efficiency_from_runs`); transmission and background subtraction of the
sample, thickness and absolute scale are phase 3.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from importlib import resources
from typing import Dict, List, Optional

import numpy as np
import yaml

from .core.averaging import Binning, radial_binning
from .core.counting import counts_uarray
from .core.deadtime import monitor_deadtime, nonparalysable
from .core.efficiency import Efficiency, efficiency_from_water, water_intensity
from .core.geometry import PixelGeometry, pixel_geometry, run_geometry
from .core.normalisation import attenuator, by_monitor, by_time
from .core.resolution import sigma_q_phase1
from .data.uarray import Source, UArray, constant
from .io.sinq import RawRun

__all__ = ["load_instrument", "ReducedRun", "reduce_run_1d", "uncertain_parameter", "normalised_counts",
           "combined_normalised_counts", "efficiency_from_runs"]


def _L2(raw) -> float:
    """Distance for the geometry: the effective distance (recorded + offset) when known."""
    return float(getattr(raw, "L2_effective_m", raw.L2_m))


_INSTRUMENTS: Dict[str, dict] = {}


def load_instrument(name="sans_i") -> dict:
    """The instrument file ``instruments/sinq/<name>.yaml``; ``name`` may also be a raw run (its
    ``instrument``, SANS-I by default)."""
    if not isinstance(name, str):
        name = getattr(name, "instrument", None) or "sans_i"
    if name not in _INSTRUMENTS:
        with resources.files(__package__ + ".instruments.sinq").joinpath(f"{name}.yaml").open() as fh:
            _INSTRUMENTS[name] = yaml.safe_load(fh)
    import copy
    return copy.deepcopy(_INSTRUMENTS[name])


def _panel_offsets(raw):
    from .io.sans_llb import panel_offsets
    return panel_offsets(raw.bank)


def uncertain_parameter(value: float, sigma: Optional[float], key: str, group: str,
                        notes: List[str]) -> UArray:
    """A scalar parameter. sigma None means 'uncertainty not known': it is not
    propagated and a note says so."""
    if sigma is None:
        notes.append(f"{key}: uncertainty unknown, not propagated")
        return constant(value)
    return UArray.scalar(value, sigma, key, group, systematic=True)


@dataclass
class ReducedRun:
    run: int
    q: np.ndarray
    intensity: UArray            # 1D, counts / monitor / sr
    q_sigma: np.ndarray
    binning: Binning
    geometry: PixelGeometry
    image: UArray                # 2D, counts / monitor / sr, before averaging
    deadtime_factor: float
    flags: np.ndarray
    raw: RawRun
    parameters: Dict[str, object] = field(default_factory=dict)
    notes: List[str] = field(default_factory=list)

    @property
    def I(self):
        return self.intensity.value

    def idev(self):
        """Statistical standard deviation (non-systematic sources): Idev."""
        return self.intensity.std(systematic=False)

    def systematic_std(self):
        return self.intensity.std(systematic=True)


def normalised_counts(raw: RawRun, inst: dict, notes: List[str], mask=None, normalise_by: str = "monitor"):
    """Steps 2 to 6 of one run: counting error, mask, dead time, attenuator and normalisation.
    Returns (UArray image in counts / monitor or counts / s, dead-time factor, uint32 mask, unit)."""
    run = raw.run

    # step 3: counting error on the raw counts, before anything else
    c = counts_uarray(raw.counts, f"counting:{run}")

    # step 2: static mask (reason bits; 0 = valid)
    mask = np.zeros(raw.counts.shape, np.uint32) if mask is None else np.asarray(mask, np.uint32)

    # step 4: dead time
    dt = inst["detector"]["deadtime"]
    tau = uncertain_parameter(dt["tau_s"], dt.get("uncertainty_s"), "deadtime:detector", "deadtime", notes)
    axis = None if dt.get("rate", "global") == "global" else int(dt["rate"].split(":")[1])
    c, factor, dtflags = nonparalysable(c, raw.counting_time_s, tau, axis=axis,
                                        flag_limit=dt.get("flag_factor", 1.2))
    if np.any(dtflags):
        mask = mask | (dtflags.astype(np.uint32) * 8)
        notes.append(f"dead-time factor above {dt.get('flag_factor', 1.2)}: pixels flagged (bit 8)")

    # step 5: attenuator
    att = inst["attenuator"].get(raw.attenuator)
    if att is None:
        raise KeyError(f"attenuator selection '{raw.attenuator}' not in the instrument file")
    a = uncertain_parameter(att["transmission"], att.get("uncertainty"),
                            f"attenuator:{raw.attenuator}", "attenuator", notes) \
        if raw.attenuator != "0" else constant(1.0)
    c = attenuator(c, a)
    chk = (att or {}).get("water_check") if raw.attenuator != "0" else None
    if chk:
        ratio = float(chk["transmission"]) / float(att["transmission"])
        notes.append(f"attenuator {raw.attenuator}: nominal transmission {float(att['transmission']):.4g}, the water "
                     f"calibration at {chk['wavelength_nm']} nm gives {float(chk['transmission']):.4g} "
                     f"(× {ratio:.3f}): intensities scaled with this attenuator against unattenuated runs (e.g. an "
                     f"absolute scale from the direct beam) may be off by {100 * (1 / ratio - 1):+.0f} %")

    # step 6: normalisation
    if normalise_by == "monitor":
        mname = inst["monitor"]["normaliser"]
        M = raw.monitors[mname]
        mon = UArray.from_source(M, Source(f"{mname}:{run}", np.array([M]), "monitor", True))
        mdt = inst["monitor"]["deadtime"]
        if mdt["tau_s"]:
            taum = uncertain_parameter(mdt["tau_s"], mdt.get("uncertainty_s"), f"deadtime:{mname}",
                                       "deadtime", notes)
            mon = monitor_deadtime(mon, raw.counting_time_s, taum)
        else:
            notes.append(f"{mname}: no monitor dead-time correction (tau = 0)")
        c = by_monitor(c, mon)
        unit = "counts/monitor/sr"
    elif normalise_by == "time":
        c = by_time(c, raw.counting_time_s)
        unit = "counts/s/sr"
    else:
        raise ValueError("normalise_by must be 'monitor' or 'time'")

    return c, factor, mask, unit


def reduce_run_1d(raw: RawRun, beam_center_px, edges, instrument: Optional[dict] = None,
                  mask: Optional[np.ndarray] = None, normalise_by: str = "monitor",
                  sector_deg: Optional[tuple] = None,
                  dlam_over_lam_fwhm: Optional[float] = None,
                  efficiency: Optional[Efficiency] = None) -> ReducedRun:
    inst = instrument or load_instrument(raw)
    notes: List[str] = []
    run = raw.run
    c, factor, mask, unit = normalised_counts(raw, inst, notes, mask, normalise_by)
    dt = inst["detector"]["deadtime"]

    # step 7 and 11: geometry and solid angle
    if getattr(raw, "L2_offset_m", 0.0):
        notes.append(f"detector distance {raw.L2_effective_m:.4f} m = recorded {raw.L2_m:.4f} m "
                     f"{1000 * raw.L2_offset_m:+.1f} mm offset (uncertainty not propagated)")
    geo = run_geometry(raw, beam_center_px)
    if raw.bank:
        notes.append(f"SANS-LLB {raw.bank} detector: geometry from the pixel positions + offsets "
                     f"(dx, dy, dz) = {tuple(round(v, 2) for v in _panel_offsets(raw))} mm")
    if abs(raw.detector_tilt_deg) > 0.01:
        notes.append(f"chi_position = {raw.detector_tilt_deg} deg not applied as tilt (to confirm)")
    img = c / geo.d_omega
    if efficiency is not None:
        if efficiency.shape != img.shape:
            raise ValueError(f"efficiency map {efficiency.shape} does not fit the detector {img.shape}")
        img = img / efficiency.value
        mask = mask | efficiency.mask_bits()
        src = efficiency.meta.get("water_runs") or efficiency.meta.get("water_run") or efficiency.meta.get("file", "?")
        notes.append(f"divided by the detector efficiency from {src}")
        L2 = efficiency.meta.get("L2_m")
        if L2 and abs(float(L2) - raw.L2_m) > 0.01 * raw.L2_m:
            notes.append(f"efficiency measured at L2 = {float(L2):.2f} m, this run at {raw.L2_m:.2f} m")
        lam = efficiency.meta.get("wavelength_A")
        if lam and abs(float(lam) - raw.wavelength_A) > 0.02 * raw.wavelength_A:
            notes.append(f"efficiency measured at λ = {float(lam):.2f} Å, this run at {raw.wavelength_A:.2f} Å")
        tilt = efficiency.meta.get("detector_tilt_deg")
        if tilt is not None and abs(float(tilt) - raw.detector_tilt_deg) > 0.05:
            notes.append(f"efficiency measured at detector tilt {float(tilt):.2f}°, this run at "
                         f"{raw.detector_tilt_deg:.2f}°")
    img.unit = unit

    # step 14: averaging
    if not np.any(mask == 0):
        raise ValueError("no valid pixels left (all masked or flagged, e.g. dead-time factor above the flag limit)")
    b = radial_binning(geo.q, mask == 0, edges, phi=geo.phi, sector_deg=sector_deg)
    if b.A.shape[0] == 0:
        raise ValueError("no valid pixels inside the q range")
    I = b.apply(img)
    I.unit = unit

    dl = raw.dlam_over_lam_fwhm if raw.dlam_over_lam_fwhm is not None else dlam_over_lam_fwhm
    if dl is None:
        raise ValueError("dlambda/lambda not in the raw file: give dlam_over_lam_fwhm")
    qs = sigma_q_phase1(b.q_mean, b.q_var_in_bin, dl, inst["wavelength"]["distribution"])
    notes.append("Qdev: wavelength and in-bin pixel spread only; collimation term not yet included")
    return ReducedRun(run, b.q_mean, I, qs, b, geo, img, float(np.max(factor.value)),
                      mask, raw,
                      parameters=dict(beam_center_px=tuple(beam_center_px), normalise_by=normalise_by,
                                      tau_s=dt["tau_s"], attenuator=raw.attenuator,
                                      dlam_over_lam_fwhm=dl, sector_deg=sector_deg,
                                      efficiency=None if efficiency is None else dict(efficiency.meta)),
                      notes=notes)


def _transmission(t, key: str) -> UArray:
    """(value, sigma), a UArray or a number."""
    if isinstance(t, UArray):
        return t
    if isinstance(t, (tuple, list)):
        value, sigma = t
        return UArray.scalar(float(value), sigma, key, "transmission", systematic=True)
    return UArray.scalar(float(t), None, key, "transmission")


def _as_list(runs):
    if runs is None:
        return []
    return list(runs) if isinstance(runs, (list, tuple)) else [runs]


def combined_normalised_counts(runs, inst: dict, notes: List[str], mask=None, normalise_by: str = "monitor"):
    """Several runs of the same sample and configuration as one image: each run corrected and
    normalised by :func:`normalised_counts`, then averaged with the monitor counts (or counting
    times) as weights, i.e. the summed corrected counts over the summed monitor. Returns
    (UArray, uint32 mask)."""
    runs = _as_list(runs)
    if not runs:
        raise ValueError("no runs")
    bad = [(r.run, r.empty_reason) for r in runs if getattr(r, "empty_reason", "")]
    for x, why in bad:
        notes.append(f"run {x} skipped: {why}")
    runs = [r for r in runs if not getattr(r, "empty_reason", "")]
    if not runs:
        raise ValueError("no usable runs: " + "; ".join(f"{x}: {why}" for x, why in bad))
    total, wsum = None, 0.0
    mname = inst["monitor"]["normaliser"]
    for r in runs:
        x, _, mask, _ = normalised_counts(r, inst, notes, mask, normalise_by)
        w = float(r.monitors[mname]) if normalise_by == "monitor" else float(r.counting_time_s)
        total = x * w if total is None else total + x * w
        wsum += w
    return total / wsum, mask


def efficiency_from_runs(water, empty_cell, beam_center_px, t_water, t_empty=None,
                         mask=None, blocked=None, instrument: Optional[dict] = None,
                         normalise_by: str = "monitor", reject_sigma: Optional[float] = 5.0,
                         water_k=None) -> Efficiency:
    """Detector efficiency from water (see :mod:`pysasfit.sasred.core.efficiency`).

    ``water``, ``empty_cell``, ``blocked``: a raw run or a list of runs (summed) at the same
    detector distance and wavelength (``empty_cell`` and ``blocked`` may be None); ``t_water``,
    ``t_empty``: transmissions as (value, sigma) or plain numbers; ``mask``: the mask of the water
    data (default: the suggested mask of the summed water counts, border + beamstop + hot pixels).
    Every run is corrected for dead time and attenuator and normalised (monitor or time) as in
    :func:`reduce_run_1d`. ``reject_sigma``: see
    :func:`~pysasfit.sasred.core.efficiency.efficiency_from_water` (beamstop edge). ``water_k``:
    the angular dependence K(2θ)/K(0) of the water (a function of 2θ in rad, e.g.
    :meth:`~pysasfit.sasred.core.water.WaterComparison.K` from water at two distances); the water
    image is divided by it so that the map holds the detector efficiency only."""
    from .core.geometry import pixel_geometry
    from .core import mask as mk
    waters, cells, blocks = _as_list(water), _as_list(empty_cell), _as_list(blocked)
    if not waters:
        raise ValueError("no water run")
    water = waters[0]
    inst = instrument or load_instrument(water)
    notes: List[str] = []
    if mask is None:
        if water.panel is not None:
            mask, _ = mk.default_mask(water, beam_center_px)
        else:
            mask, _ = mk.suggest_mask(np.sum([w.counts for w in waters], axis=0), beam_center_px)
    W, mask = combined_normalised_counts(waters, inst, notes, mask, normalise_by)
    EC = B = None
    if cells:
        EC, mask = combined_normalised_counts(cells, inst, notes, mask, normalise_by)
        if t_empty is None:
            raise ValueError("the empty-cell transmission t_empty is needed with an empty cell")
    if blocks:
        B, mask = combined_normalised_counts(blocks, inst, notes, mask, normalise_by)
    for r in waters[1:] + cells + blocks:
        if (abs(r.L2_m - water.L2_m) > 0.01 * water.L2_m
                or abs(r.wavelength_A - water.wavelength_A) > 0.02 * water.wavelength_A
                or abs(r.detector_tilt_deg - water.detector_tilt_deg) > 0.05):
            notes.append(f"run {r.run}: other distance, wavelength or detector tilt than the water run {water.run}")
    empty_cell = cells[0] if cells else None
    tw = _transmission(t_water, f"transmission:{water.run}")
    te = None if empty_cell is None else _transmission(t_empty, f"transmission:{empty_cell.run}")
    geo = run_geometry(water, beam_center_px)
    I = water_intensity(W, EC, tw, te if te is not None else 1.0, geo.d_omega, B)
    if water_k is not None:
        I = I / np.asarray(water_k(geo.two_theta), float)
        notes.append("water divided by its angular dependence K(2θ)/K(0)")
    valid = (np.asarray(mask) == 0) & np.isfinite(I.value) & (I.value > 0)
    meta = dict(water_run=water.run, water_runs=[r.run for r in waters], empty_cell_runs=[r.run for r in cells],
                blocked_runs=[r.run for r in blocks],
                t_water=[float(tw.value), float(tw.std())],
                t_empty=None if te is None else [float(te.value), float(te.std())],
                L2_m=water.L2_m, wavelength_A=water.wavelength_A, beam_center_px=list(map(float, beam_center_px)),
                dlam_over_lam_fwhm=water.dlam_over_lam_fwhm, detector_tilt_deg=water.detector_tilt_deg,
                L1_m=water.L1_m, pixel_size_m=list(water.pixel_size_m), water_k_applied=water_k is not None,
                detector_offset_m=list(water.detector_offset_m), attenuator=water.attenuator,
                measured_start=min(str(r.start_time) for r in waters),
                measured_end=max(str(r.end_time or r.start_time) for r in waters),
                normalise_by=normalise_by, sample_name=water.sample_name, notes=list(dict.fromkeys(notes)),
                method="water, empty-cell and transmission corrected, divided by the solid angle, mean 1 over "
                       "the valid pixels; masked pixels = row mean x column mean / mean (x and y wire)")
    eff = efficiency_from_water(I, valid, meta, reject_sigma=reject_sigma,
                                border=(np.asarray(mask) & mk.BORDER).astype(bool))
    eff.meta["n_filled"] = int(eff.filled.sum())
    eff.meta["n_undefined"] = int(eff.undefined.sum())
    return eff
