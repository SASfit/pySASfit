# SPDX-License-Identifier: BSD-3-Clause
"""Header table of a folder of SANS-I raw files (``sans<year>n<run>.hdf``)."""
from __future__ import annotations

import glob
import os
import re
from dataclasses import dataclass
from typing import List, Optional

from .io.sinq import RawRun, read_sans_i

__all__ = ["CatalogueEntry", "scan_folder", "raw_files"]

_RUN = re.compile(r"n0*(\d+)\.hdf$", re.IGNORECASE)


@dataclass
class CatalogueEntry:
    path: str
    run: int
    raw: Optional[RawRun] = None        # header and counts, without checksum
    error: Optional[str] = None         # file that could not be read

    def row(self, units="nm"):
        """Values for the run table (wavelength in nm or Å)."""
        from .units import units as _units
        r = self.raw
        if r is None:
            return [self.run, f"(unreadable: {self.error})", "", "", "", "", "", "", ""]
        tot = int(r.counts.sum())
        lam = f"{_units(units).lam(r.wavelength_A):.{3 if units == 'nm' else 2}f}"
        return [r.run, r.sample_name, lam, f"{r.L2_m:.2f}", r.attenuator,
                f"{r.monitors.get('monitor1', 0):.4g}", f"{r.counting_time_s:.1f}", f"{tot:d}",
                "yes" if r.aborted else ""]


COLUMNS = ["run", "sample", "λ [nm]", "L2 [m]", "att.", "monitor1", "time [s]", "counts", "aborted"]


def raw_files(folder) -> List[str]:
    files = [p for p in glob.glob(os.path.join(folder, "*.hdf")) if _RUN.search(os.path.basename(p))]
    return sorted(files, key=lambda p: int(_RUN.search(os.path.basename(p)).group(1)))


def scan_folder(folder, progress=None) -> List[CatalogueEntry]:
    """Read the header of every raw file in ``folder`` (sorted by run number).
    ``progress(i, n)`` is called after each file. Unreadable files are kept with
    their error message instead of stopping the scan."""
    files = raw_files(folder)
    out = []
    for i, p in enumerate(files):
        run = int(_RUN.search(os.path.basename(p)).group(1))
        try:
            out.append(CatalogueEntry(p, run, read_sans_i(p, checksum=False)))
        except Exception as e:  # noqa: BLE001 -- listed, not fatal
            out.append(CatalogueEntry(p, run, None, f"{type(e).__name__}: {e}"))
        if progress is not None:
            progress(i + 1, len(files))
    return out
