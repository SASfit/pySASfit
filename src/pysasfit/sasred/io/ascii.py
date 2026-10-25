# SPDX-License-Identifier: BSD-3-Clause
"""Plain-text output: q, I, Idev (statistical), Qdev, and the systematic standard
deviation as a fifth column. The header lists the run, parameters and notes."""
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
    qu = "1/A" if u.key == "A" else u.q
    head.append(f"q [{qu}]  I  Idev(statistical)  Qdev [{qu}]  sigma_systematic")
    data = np.column_stack([u.q_out(r.q), r.I, r.idev(), u.q_out(r.q_sigma), r.systematic_std()])
    np.savetxt(path, data, fmt="%.8e", header="\n".join(head))
    return path
