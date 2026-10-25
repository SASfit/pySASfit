# SPDX-License-Identifier: BSD-3-Clause
"""Transmission tables written during the beam time (``trans.txt``)::

    Summary of results:
    -------------------
       Data file,    Sample name, Transmission, +/- Error
    D0082267.001,            H2O,       0.2399,    0.0008

The data file names the transmission run (``D0082267.001`` = run 82267); the sample name links
it to the scattering runs of the same sample, which have their own run numbers.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Optional

__all__ = ["Transmission", "read_transmission_summary", "find_transmission"]

_NUM = r"[-+0-9.eE]+"
_LINE = re.compile(rf"^\s*(?P<file>[^,]+?)\s*,\s*(?P<sample>[^,]*?)\s*,\s*(?P<t>{_NUM})\s*,\s*(?P<s>{_NUM})\s*$")


@dataclass(frozen=True)
class Transmission:
    file: str
    run: Optional[int]
    sample: str
    value: float
    sigma: float


def read_transmission_summary(path) -> List[Transmission]:
    """All rows of a transmission table, in file order."""
    out = []
    with open(path, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            m = _LINE.match(line)
            if not m:
                continue
            digits = re.findall(r"\d+", m["file"])
            run = int(digits[0]) if digits else None
            out.append(Transmission(m["file"], run, m["sample"], float(m["t"]), float(m["s"])))
    return out


def find_transmission(entries, sample: Optional[str] = None, run: Optional[int] = None) -> Transmission:
    """The row of a sample name (the last one if the sample was measured more than once) or of a
    transmission run. KeyError if there is none."""
    hits = [e for e in entries if (sample is None or e.sample == sample) and (run is None or e.run == run)]
    if not hits:
        raise KeyError(f"no transmission for sample={sample!r} run={run!r}")
    return hits[-1]
