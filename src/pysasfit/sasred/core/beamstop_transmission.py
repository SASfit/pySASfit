# SPDX-License-Identifier: BSD-3-Clause
"""Transmission from the beam behind a semi-transparent beam stop (SANS-LLB).

A semi-transparent beam stop lets a known, small fraction of the direct beam through. Its counts, in a
*transmission region* of pixels drawn around the beam centre (the second layer of the mask editor: these
pixels are used, not left out), measure the beam transmitted by the sample during the scattering run itself:

    I_bs = sum over the region of the counts, corrected for dead time and attenuator, per monitor (or s)
    T_bs = I_bs(sample run) / I_bs(reference run)

with a reference run of the same configuration and beam stop: the empty beam, or the empty cell (then
T_bs is the transmission of the sample relative to the cell). Scattered neutrons that land in the region
are counted too, so T_bs is slightly too high for strong scatterers; how reliable it is, is what
:func:`compare` shows against the transmission from the transmission runs (beam stop out).

The uncertainties come from the same propagation as the reduction (counting, monitor, dead time,
attenuator; :mod:`pysasfit.sasred.data.uarray`): ``sigma`` statistical, ``sigma_total`` including the
systematic sources that do not cancel in the ratio (e.g. different attenuators).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence

import numpy as np

__all__ = ["RegionIntensity", "region_intensity", "BeamstopTransmission", "beamstop_transmission", "compare"]


@dataclass
class RegionIntensity:
    """Intensity in the transmission region of one run or of runs summed (monitor-weighted mean)."""
    runs: List[int]
    value: object                     # UArray scalar: counts / monitor (or / s), dead time and attenuator corrected
    counts: float                     # raw counts in the region, all runs
    n_pixels: int
    unit: str
    notes: List[str] = field(default_factory=list)

    @property
    def I(self) -> float:
        return float(np.asarray(self.value.value).ravel()[0])

    @property
    def sigma(self) -> float:
        return float(np.asarray(self.value.std(systematic=False)).ravel()[0])


def region_intensity(raws: Sequence, region, instrument: Optional[dict] = None,
                     normalise_by: str = "monitor") -> RegionIntensity:
    """Sum of the corrected counts of every run of ``raws`` in ``region`` (bool image, True = used), per
    monitor (``normalise_by="monitor"``) or per second; several runs: weighted by their monitor (or
    time), i.e. as if summed."""
    from ..reduce import load_instrument, normalised_counts
    raws = [r for r in raws if r is not None]
    if not raws:
        raise ValueError("no runs")
    region = np.asarray(region, bool)
    if not region.any():
        raise ValueError("the transmission region is empty")
    inst = instrument or load_instrument(raws[0])
    notes: List[str] = []
    total, wsum, counts = None, 0.0, 0.0
    unit = ""
    for r in raws:
        if r.counts.shape != region.shape:
            raise ValueError(f"run {r.run}: detector {r.counts.shape} does not fit the region {region.shape}")
        c, _factor, _mask, unit = normalised_counts(r, inst, notes, None, normalise_by)
        s = c.sum(region)
        if normalise_by == "monitor":
            w = float(r.monitors[inst["monitor"]["normaliser"]])
        else:
            w = float(r.counting_time_s)
        total = s * w if total is None else total + s * w
        wsum += w
        counts += float(np.sum(r.counts[region]))
    unit = unit.replace("/sr", "")
    return RegionIntensity([r.run for r in raws], total / wsum, counts, int(region.sum()), unit,
                           list(dict.fromkeys(notes)))


@dataclass
class BeamstopTransmission:
    sample: RegionIntensity
    reference: RegionIntensity
    value: float
    sigma: float                      # statistical
    sigma_total: float                # with the systematic sources that do not cancel

    def text(self) -> str:
        return (f"T_bs = {self.value:.4f} ± {self.sigma:.4f} (total ± {self.sigma_total:.4f}): runs "
                f"{self.sample.runs} / reference {self.reference.runs}, region {self.sample.n_pixels} px, "
                f"{self.sample.counts:.0f} / {self.reference.counts:.0f} counts")


def beamstop_transmission(sample_raws: Sequence, reference_raws: Sequence, region, instrument=None,
                          normalise_by: str = "monitor") -> BeamstopTransmission:
    """T_bs = I_bs(sample runs) / I_bs(reference runs) in the transmission region."""
    s = region_intensity(sample_raws, region, instrument, normalise_by)
    ref = region_intensity(reference_raws, region, instrument, normalise_by)
    if ref.I <= 0:
        raise ValueError(f"reference runs {ref.runs}: no counts in the transmission region")
    t = s.value / ref.value
    v = float(np.asarray(t.value).ravel()[0])
    return BeamstopTransmission(s, ref, v, float(np.asarray(t.std(systematic=False)).ravel()[0]),
                                float(np.asarray(t.std()).ravel()[0]))


def compare(t_bs: float, s_bs: float, t_runs: Optional[float], s_runs: Optional[float]):
    """(ratio T_bs / T_runs, its σ, z = (T_bs - T_runs) / σ_diff) or (None, None, None)."""
    if t_runs is None or not np.isfinite(t_runs) or t_runs <= 0:
        return None, None, None
    s_runs = 0.0 if s_runs is None or not np.isfinite(s_runs) else float(s_runs)
    ratio = t_bs / t_runs
    s_ratio = ratio * np.hypot(s_bs / t_bs if t_bs else 0.0, s_runs / t_runs)
    sd = np.hypot(s_bs, s_runs)
    z = (t_bs - t_runs) / sd if sd > 0 else None
    return float(ratio), float(s_ratio), (None if z is None else float(z))
