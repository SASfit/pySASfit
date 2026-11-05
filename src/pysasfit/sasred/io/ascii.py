# SPDX-License-Identifier: BSD-3-Clause
"""Plain-text output: q, I, Idev (statistical), Qdev, and the systematic standard
deviation as a fifth column. The header lists the run, parameters and notes. Without a resolution
(no direct-beam measurement of the geometry) Qdev is written as 0, as readers take a fourth column as
the resolution, and the header says so."""
from __future__ import annotations

import numpy as np

from .. import __version__

__all__ = ["write_ascii_1d"]


def write_ascii_1d(path, reduced, units="nm"):
    """``units``: "nm" (q, Qdev in 1/nm, default) or "A" (1/Å)."""
    from ..units import units as _units
    u = _units(units)
    r = reduced
    head = [f"pySASred {__version__}", f"run {r.run}  sample {r.raw.sample_name}",
            f"raw {r.raw.path}", f"unit of I: {r.intensity.unit}"]
    head += [f"{k} = {v}" for k, v in r.parameters.items()]
    head += [f"note: {n}" for n in r.notes]
    head.append(f"wavelength = {u.lam(r.raw.wavelength_A):.5g} {u.length}")
    from .metadata import metadata_json
    head.append(f"metadata_json = {metadata_json(r.raw, getattr(r, 'parameters', None))}")
    qu = "1/A" if u.key == "A" else u.q
    qs = r.q_sigma
    if qs is None:
        head.append("Qdev = 0: resolution not determined (no direct-beam measurement of this geometry)")
        qs = np.zeros(len(r.q))
    head.append(f"q [{qu}]  I  Idev(statistical)  Qdev [{qu}]  sigma_systematic")
    data = np.column_stack([u.q_out(r.q), r.I, r.idev(), u.q_out(qs), r.systematic_std()])
    np.savetxt(path, data, fmt="%.8e", header="\n".join(head))
    return path
