# SPDX-License-Identifier: BSD-3-Clause
"""NXcanSAS 1D output with full uncertainty information.

Standard fields: I with @uncertainties = "Idev" (statistical part only),
Q with @resolutions = "Qdev" (left out when the resolution was not determined: no direct-beam
measurement of the geometry). 2D: Qx, Qy with @resolutions = "Qxdev" / "Qydev" (per pixel). Non-standard
additions, all inside SASdata::

  I_covariance_statistical   (n x n)  covariance of the non-systematic sources
  I_systematic/<source>      sensitivity dI/ds of each systematic source,
                             with @sigma, @group (fully correlated scale terms)
  I_budget/<source>          variance contribution of every source

Provenance goes to SASprocess: software, version, raw file paths and SHA-256,
parameters and notes (for example "uncertainty unknown, not propagated"), and ``metadata``: the
header items of the raw run as JSON (:mod:`.metadata`), used to group curves later.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone

import h5py
import numpy as np

from .. import __version__

__all__ = ["write_nxcansas_1d", "write_nxcansas_2d"]


def _s(x):
    return np.bytes_(str(x).encode("utf-8"))


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
        q.attrs["units"] = _s(u.nexus_q)
        if reduced.q_sigma is not None:
            q.attrs["resolutions"] = _s("Qdev")
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
        from .metadata import metadata_json
        p["metadata"] = _s(metadata_json(reduced.raw))          # header items of the (first) raw run
        sm = e.create_group("sassample")
        sm.attrs.update({"NX_class": _s("NXsample"), "canSAS_class": _s("SASsample")})
        sm["name"] = _s(reduced.raw.sample_name)


def write_nxcansas_2d(path, reduced, title=None, units="nm"):
    """The 2D image of a reduction (:class:`~pysasfit.sasred.reduce.ReducedRun` or
    :class:`~pysasfit.sasred.absolute.Reduced2D`) as NXcanSAS: SASdata with I (ny, nx), Idev
    (statistical), Qx, Qy (ny, nx; ``I_axes`` = "Qy,Qx"), Mask (1 = not valid) and the uint32 reason
    bits of pySASred (``mask_bits``); SASinstrument/SASdetector with distance, beam centre (pixels)
    and pixel size so that q and the azimuth can be recomputed; SASprocess as in 1D with the
    header items (``metadata``)."""
    from ..units import units as _units
    from .metadata import metadata_json
    u = _units(units)
    r = reduced
    img = r.image
    mask = np.asarray(getattr(r, "mask", None) if getattr(r, "mask", None) is not None else r.flags, np.uint32)
    geo, raw = r.geometry, r.raw
    par = dict(getattr(r, "parameters", {}) or {})
    with h5py.File(path, "w") as f:
        f.attrs["default"] = "sasentry01"
        e = f.create_group("sasentry01")
        e.attrs.update({"NX_class": _s("NXentry"), "canSAS_class": _s("SASentry"),
                        "version": _s("1.1"), "default": _s("sasdata01")})
        e["definition"] = _s("NXcanSAS")
        e["title"] = _s(title or par.get("sample") or raw.sample_name)
        e["run"] = _s(getattr(r, "run", "") or (getattr(r, "runs", None) or [""])[0])
        d = e.create_group("sasdata01")
        d.attrs.update({"NX_class": _s("NXdata"), "canSAS_class": _s("SASdata"), "signal": _s("I"),
                        "I_axes": _s("Qy,Qx"), "Qy_indices": np.int32(0), "Qx_indices": np.int32(1)})
        unit = getattr(img, "unit", "") or getattr(r, "unit", "")
        i = d.create_dataset("I", data=np.asarray(img.value, float), compression="gzip")
        i.attrs.update({"units": _s("1/cm" if "cm" in unit and "sr" in unit and "monitor" not in unit else
                                    "arbitrary"), "uncertainties": _s("Idev"), "sasred_unit": _s(unit)})
        d.create_dataset("Idev", data=np.asarray(img.std(systematic=False), float), compression="gzip")
        res = getattr(r, "resolution", None)
        for k, a in (("Qx", geo.qx), ("Qy", geo.qy)):
            ds = d.create_dataset(k, data=u.q_out(np.asarray(a, float)), compression="gzip")
            ds.attrs["units"] = _s(u.nexus_q)
            if res is not None:
                ds.attrs["resolutions"] = _s(k + "dev")
        if res is not None:
            # per pixel: sigma along the detector axes (standard), along / across q and the q_x-q_y
            # correlation (non-standard, for 2D smearing with the full Gaussian)
            for k, key in (("Qxdev", "qx_dev"), ("Qydev", "qy_dev"), ("Qdev_parallel", "q_parallel_dev"),
                           ("Qdev_perpendicular", "q_perpendicular_dev")):
                d.create_dataset(k, data=u.q_out(np.asarray(res[key], float)), compression="gzip").attrs["units"] = \
                    _s(u.nexus_q)
            d.create_dataset("Qxy_correlation", data=np.asarray(res["rho_xy"], float), compression="gzip")
        d.create_dataset("Mask", data=(mask != 0).astype(np.int8), compression="gzip")
        d.create_dataset("mask_bits", data=mask, compression="gzip")
        ins = e.create_group("sasinstrument")
        ins.attrs.update({"NX_class": _s("NXinstrument"), "canSAS_class": _s("SASinstrument")})
        det = ins.create_group("sasdetector")
        det.attrs.update({"NX_class": _s("NXdetector"), "canSAS_class": _s("SASdetector")})
        det["SDD"] = float(geo.L2_m)
        det["SDD"].attrs["units"] = _s("m")
        bc = par.get("beam_center_px") or (np.nan, np.nan)
        det["beam_center_x"] = float(bc[0])
        det["beam_center_y"] = float(bc[1])
        det["beam_center_x"].attrs["units"] = det["beam_center_y"].attrs["units"] = _s("pixel")
        ps = raw.pixel_size_m
        px, py = (ps, ps) if np.isscalar(ps) else ps
        det["x_pixel_size"], det["y_pixel_size"] = float(px), float(py)
        det["x_pixel_size"].attrs["units"] = det["y_pixel_size"].attrs["units"] = _s("m")
        det["wavelength"] = float(raw.wavelength_A)
        det["wavelength"].attrs["units"] = _s("angstrom")
        sm = e.create_group("sassample")
        sm.attrs.update({"NX_class": _s("NXsample"), "canSAS_class": _s("SASsample")})
        sm["name"] = _s(par.get("sample") or raw.sample_name)
        p = e.create_group("sasprocess01")
        p.attrs.update({"NX_class": _s("NXprocess"), "canSAS_class": _s("SASprocess")})
        p["name"] = _s("pySASred")
        p["version"] = _s(__version__)
        p["date"] = _s(datetime.now(timezone.utc).isoformat())
        p["raw_file"] = _s(raw.path)
        p["parameters"] = _s(json.dumps(par, default=str))
        p["notes"] = _s("\n".join(getattr(r, "notes", []) or []))
        p["metadata"] = _s(metadata_json(raw))
    return str(path)
