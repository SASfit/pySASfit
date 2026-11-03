# SPDX-License-Identifier: BSD-3-Clause
"""Header items of a raw run, written into the reduced files so that later tools (pyTOOLbox,
*Merge geometries*) can group curves by any of them (sample, temperature, field, pressure, ...).

Named items (:data:`pysasfit.sasred.grouping.FIELDS`, e.g. ``L2_m``, ``wavelength_A``) and every
scalar entry of the raw file as ``hdf:<path>`` (values as stored); a numeric 1-D array (a log, e.g.
a temperature during the run) gives its mean.
"""
from __future__ import annotations

import json
from typing import Dict

import numpy as np

__all__ = ["run_metadata", "metadata_json"]


def _plain(v):
    if isinstance(v, bytes):
        return v.decode(errors="replace").strip()
    if isinstance(v, np.generic):
        v = v.item()
    if isinstance(v, float) and not np.isfinite(v):
        return None
    return v


def run_metadata(raw, max_log: int = 100000) -> Dict[str, object]:
    from ..grouping import FIELDS
    out: Dict[str, object] = {}
    for k, get in FIELDS.items():
        try:
            v = get(raw)
        except Exception:  # noqa: BLE001
            continue
        if k == "start":
            v = getattr(raw, "start_time", None)
        out[k] = _plain(v)
    out["run"] = int(getattr(raw, "run", 0) or 0)
    out["instrument"] = getattr(raw, "instrument", "sans_i")
    if getattr(raw, "bank", ""):
        out["bank"] = raw.bank
        g = getattr(raw, "panel", None)
        if g is not None:
            out["bank_distance_m"] = float(g.distance_m)
    try:
        import h5py
        with h5py.File(raw.path, "r") as f:
            def visit(name, o):
                if not isinstance(o, h5py.Dataset):
                    return
                key = "hdf:" + (name.split("/", 1)[1] if name.startswith(("entry1/", "entry0/")) else name)
                try:
                    if o.size == 1:
                        out[key] = _plain(np.asarray(o[()]).ravel()[0])
                    elif o.ndim == 1 and 1 < o.size <= max_log and o.dtype.kind in "fiu":
                        out[key] = _plain(float(np.nanmean(o[()])))
                except Exception:  # noqa: BLE001
                    pass
            f.visititems(visit)
    except Exception:  # noqa: BLE001 -- no file: named items only
        pass
    return out


def metadata_json(raw) -> str:
    return json.dumps(run_metadata(raw), default=str)
