# SPDX-License-Identifier: BSD-3-Clause
"""Detector efficiency (flood) from a water measurement.

Water scatters almost isotropically, so after the usual corrections its image is flat apart
from the efficiency of the detector pixels:

    I_water = [ (W - B) - (T_w / T_ec) (EC - B) ] / (T_w ΔΩ)

with W, EC and B the monitor-normalised images of water, empty cell and (optionally) the
blocked beam, T_w and T_ec the transmissions and ΔΩ the solid angle of each pixel. The
efficiency is this image divided by its mean over the valid (unmasked) pixels.

**Masked pixels** (behind the beamstop, hot pixels, ...) have no water data. The SANS-I
detector is a multiwire detector read out in coincidence: a pixel is the crossing of an x wire
and a y wire, so its efficiency is approximately the product of a row and a column factor,
ε(i, j) ≈ a_i · b_j. The masked pixels are therefore filled with

    ε(i, j) = R_i · C_j / m,

R_i the mean efficiency of the valid pixels of row i (same y wire), C_j that of column j (same
x wire) and m the mean of all valid pixels (1 by construction). A pixel whose whole row or
whole column is masked (the standard border) has no estimate: its efficiency is set to 1 and
it gets the mask bit :data:`~pysasfit.sasred.core.mask.EFFICIENCY`.

**Edge of the beamstop**: pixels in the half shadow of the beamstop are not always in the
mask of the water run, and their low "efficiency" would then be applied to the sample. Valid
pixels within ``edge_px`` pixels of the masked area (not counting the border) whose value
differs from the row x column estimate by more than ``reject_sigma`` statistical standard
deviations are therefore *rejected* and filled like the masked ones (iterated until no new
pixel is rejected). Elsewhere a deviating pixel is kept: it may be a real local defect.

The efficiency is a :class:`~pysasfit.sasred.data.uarray.UArray`: it keeps its exact
sensitivities to the counts and monitors of the water and empty-cell runs and to the
transmissions, so dividing a sample image by it propagates all of them (also the
correlations it causes between the q points). Saved to a file (:meth:`Efficiency.save`), only
the standard deviation per pixel is kept; :meth:`Efficiency.load` makes it one independent
source per pixel (``flood:<name>``), which drops the (small) correlations between pixels.
"""
from __future__ import annotations

import json
import os
import warnings
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, Optional

import numpy as np
import scipy.sparse as sp

from ..data.uarray import Source, UArray
from .mask import EFFICIENCY

__all__ = ["Efficiency", "water_intensity", "separable_fill", "efficiency_from_water"]


@dataclass
class Efficiency:
    """Detector efficiency map (mean 1 over the measured pixels).

    Fields::

        value     : UArray (detector shape) with its uncertainty sources
        measured  : bool, pixels taken from the water data
        filled    : bool, masked or rejected pixels approximated by row x column (x and y wire)
        undefined : bool, no estimate (whole row or column masked); value 1, mask bit EFFICIENCY
        rejected  : bool, valid pixels at the beamstop edge replaced by the estimate (part of filled)
        meta      : provenance (runs, transmissions, distance, wavelength, ...)
    """
    value: UArray
    measured: np.ndarray
    filled: np.ndarray
    undefined: np.ndarray
    rejected: Optional[np.ndarray] = None
    meta: Dict[str, object] = field(default_factory=dict)

    def __post_init__(self):
        if self.rejected is None:
            self.rejected = np.zeros(self.filled.shape, bool)

    @property
    def shape(self):
        return self.value.shape

    def std(self) -> np.ndarray:
        return self.value.std()

    def profiles(self):
        """(row profile, column profile): mean efficiency of the measured pixels of every row (y wire)
        and every column (x wire); NaN where a row or column has none. Useful to follow single wires."""
        v = np.where(self.measured, self.value.value, np.nan)
        with np.errstate(invalid="ignore"), warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            return np.nanmean(v, axis=1), np.nanmean(v, axis=0)

    def mask_bits(self) -> np.ndarray:
        """uint32 mask with the bit EFFICIENCY on the pixels without an estimate."""
        return self.undefined.astype(np.uint32) * EFFICIENCY

    # ------------------------------------------------------------------ files
    def write_entry(self, entry, description: str = ""):
        """Write the map into an open h5py group ``entry`` (made an NXentry): ``efficiency`` (NXdata:
        ``efficiency``, ``efficiency_errors``, ``filled``, ``undefined``, ``rejected``, the wire
        profiles) and ``process`` with the provenance."""
        from ... import __version__
        e = entry
        e.attrs["NX_class"] = "NXentry"
        e.attrs["default"] = "efficiency"
        d = e.create_group("efficiency")
        d.attrs["NX_class"] = "NXdata"
        d.attrs["signal"] = "efficiency"
        d.create_dataset("efficiency", data=self.value.value, compression="gzip")
        d.create_dataset("efficiency_errors", data=self.std(), compression="gzip")
        for key, arr, meaning in (
                ("filled", self.filled, "1 = masked in the water run, approximated by row x column"),
                ("undefined", self.undefined, "1 = no estimate (whole row or column masked); value 1"),
                ("rejected", self.rejected, "1 = valid in the water run but replaced (beamstop edge)")):
            d.create_dataset(key, data=arr.astype(np.uint8), compression="gzip")
            d[key].attrs["meaning"] = meaning
        rows, cols = self.profiles()
        d.create_dataset("row_profile", data=rows)
        d.create_dataset("column_profile", data=cols)
        d["row_profile"].attrs["meaning"] = "mean efficiency of the measured pixels of each row (y wire)"
        d["column_profile"].attrs["meaning"] = "mean efficiency of the measured pixels of each column (x wire)"
        self._write_instrument(e)
        p = e.create_group("process")
        p.attrs["NX_class"] = "NXprocess"
        p["program"] = f"pySASred {__version__}"
        p["date"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
        p["meta"] = json.dumps(self.meta, default=str)
        if description:
            p["description"] = description

    #: configuration of the water measurement written to ``instrument`` (meta key, group, name, unit)
    _CONFIG = (("L2_m", "detector", "distance", "m"),
               ("detector_tilt_deg", "detector", "tilt_angle", "deg"),
               ("pixel_size_m", "detector", "pixel_size", "m"),
               ("detector_offset_m", "detector", "offset", "m"),
               ("beam_center_px", "detector", "beam_center", "pixel"),
               ("wavelength_A", "monochromator", "wavelength", "angstrom"),
               ("dlam_over_lam_fwhm", "monochromator", "wavelength_spread", ""),
               ("L1_m", "collimator", "length", "m"),
               ("attenuator", "attenuator", "selection", ""))

    def _write_instrument(self, entry):
        """Configuration of the water measurement (detector distance and tilt, wavelength, ...) as
        NeXus fields under ``instrument``, so that every archived map says what it was measured at."""
        classes = {"detector": "NXdetector", "monochromator": "NXmonochromator",
                   "collimator": "NXcollimator", "attenuator": "NXattenuator"}
        ins = None
        for key, grp, name, unit in self._CONFIG:
            v = self.meta.get(key)
            if v is None:
                continue
            if ins is None:
                ins = entry.create_group("instrument")
                ins.attrs["NX_class"] = "NXinstrument"
            g = ins.require_group(grp)
            g.attrs["NX_class"] = classes[grp]
            ds = g.create_dataset(name, data=v if isinstance(v, str) else np.asarray(v, float))
            if unit:
                ds.attrs["units"] = unit

    def save(self, path, description: str = "") -> str:
        """HDF5 file with one map: ``/entry`` (see :meth:`write_entry`). For a dated collection of
        maps in one file see :mod:`pysasfit.sasred.efficiency_log`."""
        import h5py
        path = str(path)
        with h5py.File(path, "w") as f:
            f.attrs["default"] = "entry"
            self.write_entry(f.create_group("entry"), description)
        return path

    @classmethod
    def read_entry(cls, entry, name: str) -> "Efficiency":
        """Map from an open h5py group written by :meth:`write_entry`; its uncertainty becomes one
        independent source per pixel, ``flood:<name>``."""
        d = entry["efficiency"]
        v = np.asarray(d["efficiency"][()], float)
        s = np.asarray(d["efficiency_errors"][()], float)
        filled = np.asarray(d["filled"][()], bool)
        undefined = np.asarray(d["undefined"][()], bool)
        rejected = np.asarray(d["rejected"][()], bool) if "rejected" in d else np.zeros(filled.shape, bool)
        meta = {}
        if "process/meta" in entry:
            raw = entry["process/meta"][()]
            meta = json.loads(raw.decode() if isinstance(raw, bytes) else raw)
        var = np.where(np.isfinite(s), s, 0.0) ** 2
        value = UArray.from_source(v, Source(f"flood:{name}", var.ravel(), "flood", False))
        return cls(value, ~(filled | undefined), filled, undefined, rejected, meta)

    @classmethod
    def load(cls, path, entry: Optional[str] = None, name: Optional[str] = None) -> "Efficiency":
        """Map from a file of :meth:`save` or from an entry of an efficiency archive
        (``entry``: its name; default ``entry``, or the newest map of an archive)."""
        import h5py
        path = str(path)
        with h5py.File(path, "r") as f:
            if entry is None:
                entry = "entry" if "entry" in f else str(f.attrs.get("default", ""))
            if entry not in f:
                raise KeyError(f"{path}: no efficiency map '{entry}'")
            stem = os.path.splitext(os.path.basename(path))[0]
            eff = cls.read_entry(f[entry], name or (stem if entry == "entry" else f"{stem}/{entry}"))
        eff.meta = dict(eff.meta, file=path, entry=entry)
        return eff


def water_intensity(water: UArray, empty_cell: Optional[UArray], t_water, t_empty, d_omega,
                    blocked: Optional[UArray] = None) -> UArray:
    """Corrected water image [(W - B) - (T_w/T_ec)(EC - B)] / (T_w ΔΩ).

    ``water``, ``empty_cell``, ``blocked``: monitor-normalised images (UArray); ``t_water``,
    ``t_empty``: transmissions (UArray scalars or numbers); ``d_omega``: solid angle per pixel."""
    w = water if blocked is None else water - blocked
    if empty_cell is not None:
        ec = empty_cell if blocked is None else empty_cell - blocked
        w = w - ec * (t_water / t_empty)
    return w / t_water / np.asarray(d_omega, float)


def separable_fill(e: UArray, valid) -> tuple:
    """Masked pixels of ``e`` (2D) replaced by R_i C_j / m (row and column means of the valid
    pixels, m their overall mean). Returns (UArray, filled, undefined); undefined pixels (whole
    row or column masked) get the value 1."""
    valid = np.asarray(valid, bool)
    ny, nx = valid.shape
    N = ny * nx
    if e.shape != valid.shape:
        raise ValueError("efficiency and mask shapes differ")
    n_row = valid.sum(axis=1)
    n_col = valid.sum(axis=0)
    if valid.sum() == 0:
        raise ValueError("no valid pixels")
    flat = np.flatnonzero(valid.ravel())
    iy, ix = np.divmod(flat, nx)
    A_R = sp.csr_matrix((1.0 / n_row[iy], (iy, flat)), shape=(ny, N))
    A_C = sp.csr_matrix((1.0 / n_col[ix], (ix, flat)), shape=(nx, N))
    A_m = sp.csr_matrix((np.full(flat.size, 1.0 / flat.size), (np.zeros(flat.size, int), flat)), shape=(1, N))

    row_ok = n_row > 0
    col_ok = n_col > 0
    target = ~valid & row_ok[:, None] & col_ok[None, :]
    undefined = ~valid & ~target
    keep = sp.diags(valid.ravel().astype(float), format="csr")
    out = e.linear(keep, out_shape=e.shape)
    k = np.flatnonzero(target.ravel())
    if k.size:
        ky, kx = np.divmod(k, nx)
        sel_r = sp.csr_matrix((np.ones(k.size), (np.arange(k.size), ky)), shape=(k.size, ny))
        sel_c = sp.csr_matrix((np.ones(k.size), (np.arange(k.size), kx)), shape=(k.size, nx))
        R = e.linear(sel_r @ A_R)
        C = e.linear(sel_c @ A_C)
        m = e.linear(A_m, out_shape=())
        f = R * C / m
        embed = sp.csr_matrix((np.ones(k.size), (k, np.arange(k.size))), shape=(N, k.size))
        out = out + f.linear(embed, out_shape=e.shape)
    if undefined.any():
        out = out + np.where(undefined, 1.0, 0.0)
    out.unit = ""
    return out, target, undefined


def efficiency_from_water(I_water: UArray, valid, meta: Optional[dict] = None, reject_sigma: Optional[float] = 5.0,
                          edge_px: int = 3, border=None, max_iter: int = 5) -> Efficiency:
    """Efficiency map from the corrected water image (:func:`water_intensity`) and the valid
    pixels of the water run: normalised to mean 1 over the valid pixels, masked pixels filled by
    :func:`separable_fill`. Pixels at the beamstop edge that deviate from the fill by more than
    ``reject_sigma`` standard deviations are rejected and filled too (None: no rejection).
    ``border``: bool image of the border mask, not counted as "masked area" for the edge."""
    from scipy import ndimage
    valid0 = np.asarray(valid, bool)
    valid = valid0.copy()
    rejected = np.zeros(valid.shape, bool)
    stat_var = None
    for _ in range(max_iter if reject_sigma else 1):
        mean = I_water.sum(valid) / float(valid.sum())
        e = I_water / mean
        filled_value, filled, undefined = separable_fill(e, valid)
        if not reject_sigma:
            break
        if stat_var is None:
            stat_var = e.variance(systematic=False)
        # row x column estimate for every pixel (from the current valid set)
        ev = np.where(valid, e.value, np.nan)
        with np.errstate(invalid="ignore"), warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)          # rows / columns without valid pixel
            pred = np.outer(np.nanmean(ev, 1), np.nanmean(ev, 0)) / np.nanmean(ev)
        inner = ~valid0 if border is None else (~valid0 & ~np.asarray(border, bool))
        near = ndimage.binary_dilation(inner, iterations=int(edge_px)) & valid
        z = np.abs(e.value - pred) / np.sqrt(np.maximum(stat_var, 1e-300))
        new = near & np.isfinite(z) & (z > reject_sigma)
        if not new.any():
            break
        rejected |= new
        valid &= ~new
    m = dict(meta or {})
    m["n_rejected"] = int(rejected.sum())
    return Efficiency(filled_value, valid.copy(), filled, undefined, rejected, m)
