# SPDX-License-Identifier: BSD-3-Clause
"""Display and output units of wavelength and q.

The reduction works internally in Å and 1/Å (``RawRun.wavelength_A``, ``ReducedRun.q``).
What the user sees and what is written to files is nm and 1/nm by default, or Å and 1/Å
on request (Preferences → SANS reduction, or the unit selector on the Reduction panel).
"""
from __future__ import annotations

from dataclasses import dataclass

__all__ = ["Units", "UNITS", "units"]


@dataclass(frozen=True)
class Units:
    key: str
    length: str          # label of the wavelength unit
    q: str               # label of the q unit
    nexus_q: str         # NeXus/NXcanSAS units attribute of Q
    nexus_length: str
    to_length: float     # Å -> this unit
    to_q: float          # 1/Å -> this unit

    def lam(self, lam_A):
        return lam_A * self.to_length

    def q_out(self, q_invA):
        """1/Å (internal) -> this unit."""
        return q_invA * self.to_q

    def q_in(self, q):
        """this unit -> 1/Å (internal)."""
        return q / self.to_q


UNITS = {
    "nm": Units("nm", "nm", "1/nm", "1/nm", "nm", 0.1, 10.0),
    "A": Units("A", "Å", "1/Å", "1/angstrom", "angstrom", 1.0, 1.0),
}


def units(key="nm") -> Units:
    try:
        return UNITS[key]
    except KeyError:
        raise ValueError(f"unknown units '{key}' (one of {list(UNITS)})") from None
