# SPDX-License-Identifier: BSD-3-Clause
"""NXcanSAS 1D output with full uncertainty information.

Standard fields: I with @uncertainties = "Idev" (statistical part only),
Q with @resolutions = "Qdev". Non-standard additions, all inside SASdata::

  I_covariance_statistical   (n x n)  covariance of the non-systematic sources
  I_systematic/<source>      sensitivity dI/ds of each systematic source,
                             with @sigma, @group (fully correlated scale terms)
  I_budget/<source>          variance contribution of every source

Provenance goes to SASprocess: software, version, raw file paths and SHA-256,
parameters and notes (for example "uncertainty unknown, not propagated").
"""
from __future__ import annotations

import json
from datetime import datetime, timezone

import h5py
import numpy as np

from .. import __version__

__all__ = ["write_nxcansas_1d"]


def _s(x):
    return np.bytes_(str(x))


def write_nxcansas_1d(path, reduced, title=None, i_unit="arbitrary", units="nm"):
    """``units``: "nm" (Q in 1/nm, default) or "A" (Q in 1/angstrom)."""
    from ..units import units as _units
    u = _units(units)
    I = reduced.intensity
    with h5py.File(path, "w") as f:
        f.attrs["default"] = "sasentry01"
        e = f.create_group("sasentry01")
        e.attrs.update({"NX_class": _s("NXentry"), "canSAS_class": _s("SASentry"),
                        "version": _s("1.1"), "default": _s("sasdata01")})
        e["definition"] = _s("NXcanSAS")
        e["title"] = _s(title or reduced.raw.sample_name)
        e["run"] = _s(reduced.run)
        d = e.create_group("sasdata01")
        d.attrs.update({"NX_class": _s("NXdata"), "canSAS_class": _s("SASdata"),
                        "signal": _s("I"), "I_axes": _s("Q"), "Q_indices": np.int32(0)})
        q = d.create_dataset("Q", data=u.q_out(reduced.q))
        q.attrs.update({"units": _s(u.nexus_q), "resolutions": _s("Qdev")})
        d.create_dataset("Qdev", data=u.q_out(reduced.q_sigma)).attrs["units"] = _s(u.nexus_q)
        i = d.create_dataset("I", data=I.value)
        i.attrs.update({"units": _s(i_unit), "uncertainties": _s("Idev"),
                        "sasred_unit": _s(I.unit)})
        d.create_dataset("Idev", data=I.std(systematic=False)).attrs["units"] = _s(i_unit)
        d.create_dataset("I_covariance_statistical", data=I.covariance(systematic=False))
        g = d.create_group("I_systematic")
        for k in I._select(None, True, None):
            src = I.sources[k]
            ds = g.create_dataset(k.replace("/", "_"), data=I.jacobian(k).ravel()
                                  if src.n == 1 else I.jacobian(k))
            ds.attrs.update({"sigma": np.sqrt(np.asarray(src.variance, float)), "group": _s(src.group)})
        b = d.create_group("I_budget")
        for k, v in I.budget().items():
            b.create_dataset(k.replace("/", "_"), data=v).attrs["group"] = _s(I.sources[k].group)
        p = e.create_group("sasprocess01")
        p.attrs.update({"NX_class": _s("NXprocess"), "canSAS_class": _s("SASprocess")})
        p["name"] = _s("pySASred")
        p["version"] = _s(__version__)
        p["date"] = _s(datetime.now(timezone.utc).isoformat())
        p["raw_file"] = _s(reduced.raw.path)
        p["raw_sha256"] = _s(reduced.raw.sha256)
        p["parameters"] = _s(json.dumps(reduced.parameters, default=str))
        p["notes"] = _s("\n".join(reduced.notes))
