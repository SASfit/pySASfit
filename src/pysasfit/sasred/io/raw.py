# SPDX-License-Identifier: BSD-3-Clause
"""Raw files of the SINQ SANS instruments: the reader is chosen from the file (SANS-I ``entry1/SANS``,
SANS-LLB ``entry0/SANS-LLB``)."""
from __future__ import annotations

from typing import Optional

import h5py

from .sinq import RawRun, read_sans_i

__all__ = ["read_raw", "instrument_of"]


def instrument_of(path) -> str:
    """"sans_i" or "sans_llb" (from the content of the file)."""
    with h5py.File(str(path), "r") as f:
        if "entry0/SANS-LLB" in f:
            return "sans_llb"
        if "entry1/SANS" in f:
            return "sans_i"
    raise ValueError(f"{path}: neither a SANS-I nor a SANS-LLB raw file")


def read_raw(path, checksum: bool = True, distance_offset_m: Optional[float] = None,
             bank: str = "") -> RawRun:
    """Read a raw file of SANS-I or SANS-LLB; ``bank`` ("left", "bottom") returns the run of a
    SANS-LLB wide-angle panel (:func:`~.sans_llb.bank_run`)."""
    if instrument_of(path) == "sans_llb":
        from .sans_llb import bank_run, read_sans_llb
        r = read_sans_llb(path, checksum=checksum, distance_offset_m=distance_offset_m)
        return bank_run(r, bank) if bank else r
    if bank:
        raise KeyError(f"{path}: SANS-I has no detector bank {bank!r}")
    return read_sans_i(path, checksum=checksum, distance_offset_m=distance_offset_m)
