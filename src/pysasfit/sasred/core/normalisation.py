# SPDX-License-Identifier: BSD-3-Clause
"""Attenuator and normalisation (correction steps 5 and 6)."""
from __future__ import annotations

from ..data.uarray import UArray, constant

__all__ = ["attenuator", "by_monitor", "by_time"]


def attenuator(x: UArray, transmission: UArray | float) -> UArray:
    """Correct for an attenuator of the given transmission (factor <= 1):
    x / transmission. The tabulated factor may carry its own uncertainty."""
    t = transmission if isinstance(transmission, UArray) else constant(transmission)
    if t.value <= 0:
        raise ValueError("attenuator transmission must be positive")
    return x / t


def by_monitor(x: UArray, monitor: UArray) -> UArray:
    """Divide by the monitor count (a scalar Poisson source of the run)."""
    if monitor.value <= 0:
        raise ValueError("monitor count must be positive")
    return x / monitor


def by_time(x: UArray, time_s: float) -> UArray:
    """Divide by the counting time (taken as exact)."""
    if time_s <= 0:
        raise ValueError("counting time must be positive")
    return x / time_s
