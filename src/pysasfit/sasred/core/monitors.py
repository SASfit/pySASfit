# SPDX-License-Identifier: BSD-3-Clause
"""Which neutron monitors sit before and which after the velocity selector.

A monitor in front of the selector sees the white beam: its ratio to the proton beam
(``integrated_beam``, the proton charge) does not change with the wavelength or the selector tilt.
A monitor behind the selector sees what the selector passes, and its ratio changes with λ (source
spectrum, selector transmission) and with the tilt (flux roughly ∝ dλ/λ). Only a monitor behind the
selector cancels a change of the selector setting when the data are normalised by it.

:func:`selector_test` compares the ratio monitor / proton beam between groups of runs at different
(λ, tilt). It needs runs at two or more settings; with one setting it says so.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

__all__ = ["MonitorTest", "selector_test"]


@dataclass
class MonitorTest:
    monitor: str
    position: str                     # "before selector" | "after selector" | "undetermined" | "proton beam"
    ratios: Dict[tuple, float] = field(default_factory=dict)    # (λ Å, tilt °) -> median monitor/proton
    change: float = np.nan            # max/min − 1 of the group medians
    noise: float = np.nan             # typical relative scatter within a group
    reason: str = ""

    @property
    def after_selector(self) -> Optional[bool]:
        return {"before selector": False, "after selector": True}.get(self.position)

    def describe(self) -> str:
        r = ", ".join(f"{k[0]:g} Å/{k[1]:+g}°: {v:.4g}" for k, v in sorted(self.ratios.items()))
        return f"{self.monitor}: {self.position} ({self.reason}){'; ' + r if r else ''}"


def _setting(r):
    t = getattr(r, "selector_tilt_deg", None)
    return (round(float(r.wavelength_A), 1), round(2 * float(t or 0.0)) / 2 + 0.0)


def selector_test(raws, proton: str = "integrated_beam", min_runs: int = 2,
                  monitors: Optional[List[str]] = None, min_counts: float = 100.0) -> Dict[str, MonitorTest]:
    """Position of every monitor relative to the selector from runs at different (λ, tilt).

    For each setting the median of monitor/proton is taken (runs with proton charge and at least
    ``min_counts`` monitor counts, at least ``min_runs`` runs; a frame of a selector speed scan counts
    alone); the scatter within a setting (median relative deviation) is the
    noise. The monitor is *before the selector* when the medians of all settings agree within
    max(3·noise, 5 %), *after the selector* when they differ by more than max(5·noise, 15 %);
    in between, or with fewer than two settings, *undetermined*."""
    groups: Dict[tuple, list] = {}       # setting -> list of monitor dicts (runs, or frames of a scan)
    for r in raws:
        if r is None or getattr(r, "excluded", ""):
            continue
        fr = getattr(r, "frames", None) or {}
        if getattr(r, "scan", "") and "wavelength_A" in fr:          # selector speed scan: every frame
            t = getattr(r, "selector_tilt_deg", None)
            tilt = round(2 * float(t or 0.0)) / 2 + 0.0
            fm = fr.get("monitors") or {}
            for i, lam in enumerate(fr["wavelength_A"]):
                m = {k: float(v[i]) for k, v in fm.items() if len(v) > i}
                m["_frame"] = True                                     # a long frame counts on its own
                if m.get(proton, 0) > 0:
                    groups.setdefault((round(float(lam), 1), tilt), []).append(m)
            continue
        if getattr(r, "empty_reason", ""):
            continue
        p = float((r.monitors or {}).get(proton, 0) or 0)
        if p <= 0:
            continue
        groups.setdefault(_setting(r), []).append(dict(r.monitors))
    names = monitors or sorted({k for ms in groups.values() for m in ms for k in m if k not in (proton, "_frame")})
    out: Dict[str, MonitorTest] = {}
    for m in names:
        if all(mm.get(m) == mm[proton] for ms in groups.values() for mm in ms):
            out[m] = MonitorTest(m, "proton beam", reason=f"same counts as {proton}")
            continue
        med, scat = {}, []
        for k, rs in groups.items():
            x = np.array([float(r.get(m, 0) or 0) / float(r[proton]) for r in rs
                          if float(r.get(m, 0) or 0) >= min_counts])          # a few counts: noise
            if len(x) < min_runs and not (len(x) and any(r.get("_frame") for r in rs)):
                continue
            c = float(np.median(x))
            med[k] = c
            if len(x) > 1:
                scat.append(float(np.median(np.abs(x / c - 1))))
        if not med:
            out[m] = MonitorTest(m, "undetermined", reason=f"fewer than {min_counts:g} counts")
            continue
        noise = float(np.median(scat)) if scat else 0.0
        if len(med) < 2:
            out[m] = MonitorTest(m, "undetermined", med, np.nan, noise,
                                 "all runs at one wavelength and tilt: needs runs at two settings")
            continue
        change = max(med.values()) / min(med.values()) - 1
        if change < max(3 * noise, 0.05):
            pos, why = "before selector", f"ratio to the proton beam constant within {100 * change:.1f} %"
        elif change > max(5 * noise, 0.15):
            pos, why = "after selector", f"ratio to the proton beam changes by {100 * change:.0f} %"
        else:
            pos, why = "undetermined", f"change {100 * change:.1f} % vs scatter {100 * noise:.1f} %"
        out[m] = MonitorTest(m, pos, med, change, noise, why)
    return out
