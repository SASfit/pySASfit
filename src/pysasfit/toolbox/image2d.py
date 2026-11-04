# SPDX-License-Identifier: BSD-3-Clause
"""Reduced 2D scattering images for pyTOOLbox: display now, analysis later (sector and azimuthal
averages, A(q) + B(q) sin²ψ decomposition, box projections, sums of several images).

An :class:`Image2D` holds the image, its statistical uncertainty, the mask and the geometry
(beam centre in pixels, pixel size, detector distance, wavelength), from which q, qx, qy and the
azimuth ψ of every pixel follow. ψ = 0 is the horizontal direction (+x, axis 1), increasing
counter-clockwise (towards +y, axis 0), as in pySASred.

Sources (:func:`read_images`, :func:`image_from_reduced`):

* pySASred 2D NXcanSAS (``*_2d.h5``: I, Idev, Qx, Qy, Mask, SASdetector with the beam centre)
* other NXcanSAS 2D files with Qx, Qy (beam centre estimated from q = 0)
* BerSANS 2D files (``Type=SANSDAni``, e.g. ``D0082130.008``): no beam centre in the file; give it
  with :meth:`Image2D.set_geometry` (pixel 7.5 mm of SANS-I by default), else pixel axes only
* pySASred results in memory (single runs and rows of the reduction table)
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

__all__ = ["Image2D", "read_images", "image_from_reduced", "SANS_I_PIXEL_M"]

SANS_I_PIXEL_M = 0.0075


@dataclass
class Image2D:
    I: np.ndarray                              # (ny, nx)
    dI: Optional[np.ndarray]
    mask: np.ndarray                           # bool, True = not valid
    path: str
    fmt: str
    meta: Dict[str, object] = field(default_factory=dict)
    unit: str = ""
    centre_px: Optional[Tuple[float, float]] = None     # (x, y) 0-based
    pixel_m: Optional[Tuple[float, float]] = None
    SD_m: Optional[float] = None
    wavelength_A: Optional[float] = None
    entry: str = ""
    _qxy: Optional[Tuple[np.ndarray, np.ndarray]] = None   # given q (1/nm) when the geometry is unknown

    def __post_init__(self):
        from ..sasred.legacy import rename_keys
        new = rename_keys(self.meta)                # files written before coll / SD (L1_m, L2_m)
        if new != self.meta:                        # in place: readers keep filling the same dict
            self.meta.clear()
            self.meta.update(new)

    # ------------------------------------------------------------------ naming
    @property
    def sample(self) -> str:
        return str(self.meta.get("sample", "") or "")

    @property
    def name(self) -> str:
        b = os.path.basename(self.path)
        return f"{b}:{self.entry}" if self.entry else b

    @property
    def geometry_label(self) -> str:
        s = []
        coll = _num(self.meta.get("coll_m")) if self.meta.get("coll_m") is not None else _num(self.meta.get("Setup/Collimation"))
        if coll is not None:
            s.append(f"coll {coll:.1f} m")
        if self.SD_m is not None:
            s.append(f"SD {self.SD_m:.2f} m")
        if self.wavelength_A is not None:
            s.append(f"λ {self.wavelength_A / 10:.3f} nm")
        return ", ".join(s) or "pixels"

    # ------------------------------------------------------------------ geometry
    @property
    def has_q(self) -> bool:
        return self._qxy is not None or None not in (self.centre_px, self.pixel_m, self.SD_m, self.wavelength_A)

    def set_geometry(self, centre_px=None, pixel_m=None, SD_m=None, wavelength_A=None):
        """Set (or correct) the geometry; q and ψ then follow from it."""
        if centre_px is not None:
            self.centre_px = (float(centre_px[0]), float(centre_px[1]))
        if pixel_m is not None:
            self.pixel_m = (float(pixel_m), float(pixel_m)) if np.isscalar(pixel_m) else tuple(map(float, pixel_m))
        if SD_m is not None:
            self.SD_m = float(SD_m)
        if wavelength_A is not None:
            self.wavelength_A = float(wavelength_A)
        if None not in (self.centre_px, self.pixel_m, self.SD_m, self.wavelength_A):
            self._qxy = None

    def qxy(self) -> Tuple[np.ndarray, np.ndarray]:
        """qx, qy (1/nm) of every pixel (flat detector perpendicular to the beam)."""
        if self._qxy is not None:
            return self._qxy
        if not self.has_q:
            raise ValueError("geometry unknown: give beam centre, pixel size, distance and wavelength")
        ny, nx = self.I.shape
        iy, ix = np.mgrid[0:ny, 0:nx].astype(float)
        x = (ix - self.centre_px[0]) * self.pixel_m[0]
        y = (iy - self.centre_px[1]) * self.pixel_m[1]
        r = np.hypot(x, y)
        tth = np.arctan2(r, self.SD_m)
        q = 4 * np.pi / (self.wavelength_A / 10.0) * np.sin(tth / 2)          # 1/nm
        with np.errstate(invalid="ignore", divide="ignore"):
            c = np.where(r > 0, x / r, 0.0)
            s = np.where(r > 0, y / r, 0.0)
        return q * c, q * s

    def q(self) -> np.ndarray:
        qx, qy = self.qxy()
        return np.hypot(qx, qy)

    def psi_deg(self) -> np.ndarray:
        """Azimuth ψ in degrees, 0…360, 0 = +x (horizontal), counter-clockwise."""
        qx, qy = self.qxy()
        return np.degrees(np.arctan2(qy, qx)) % 360.0

    def q_extent(self):
        """(qx min, qx max, qy min, qy max) of the pixel edges, 1/nm, for imshow."""
        qx, qy = self.qxy()
        ny, nx = self.I.shape
        dx = (qx[:, -1] - qx[:, 0]).mean() / max(1, nx - 1)
        dy = (qy[-1, :] - qy[0, :]).mean() / max(1, ny - 1)
        return (qx[:, 0].mean() - dx / 2, qx[:, -1].mean() + dx / 2, qy[0, :].mean() - dy / 2, qy[-1, :].mean() + dy / 2)

    def valid(self) -> np.ndarray:
        return ~self.mask & np.isfinite(self.I)


# ------------------------------------------------------------------------------------- readers
def _dec(v):
    if isinstance(v, bytes):
        return v.decode(errors="replace")
    if isinstance(v, np.ndarray):
        v = v.ravel()
        return _dec(v[0]) if v.size == 1 else v
    if isinstance(v, np.generic):
        return v.item()
    return v


def _num(v):
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None
    return x if np.isfinite(x) else None


def _read_nxcansas(path) -> List[Image2D]:
    import h5py
    out = []
    with h5py.File(path, "r") as f:
        for en, e in f.items():
            if not isinstance(e, h5py.Group):
                continue
            for dn, d in e.items():
                if not (isinstance(d, h5py.Group) and str(_dec(d.attrs.get("canSAS_class", b""))) == "SASdata"):
                    continue
                if "I" not in d or d["I"].ndim != 2:
                    continue
                I = np.asarray(d["I"][()], float)
                unc = str(_dec(d["I"].attrs.get("uncertainties", b"Idev")))
                dI = np.asarray(d[unc][()], float) if unc in d else None
                if "mask_bits" in d:
                    mask = np.asarray(d["mask_bits"][()]) != 0
                elif "Mask" in d:
                    mask = np.asarray(d["Mask"][()]) != 0
                else:
                    mask = np.zeros(I.shape, bool)
                meta: Dict[str, object] = {}
                if "title" in e:
                    meta["title"] = str(_dec(e["title"][()]))
                img = Image2D(I, dI, mask, str(path), "nxcansas-2d", meta, entry=en,
                              unit=str(_dec(d["I"].attrs.get("sasred_unit", d["I"].attrs.get("units", b"")))))
                for k, g in e.items():
                    if not isinstance(g, h5py.Group):
                        continue
                    c = str(_dec(g.attrs.get("canSAS_class", b"")))
                    if c == "SASprocess":
                        for key in ("metadata",):
                            if key in g:
                                try:
                                    meta.update(json.loads(_dec(g[key][()])))
                                except (ValueError, TypeError):
                                    pass
                        if "parameters" in g:
                            try:
                                p = json.loads(_dec(g["parameters"][()]))
                                if p.get("sample"):
                                    meta["sample"] = p["sample"]
                            except (ValueError, TypeError):
                                pass
                    elif c == "SASsample" and "name" in g:
                        meta.setdefault("sample", str(_dec(g["name"][()])))
                    elif c == "SASinstrument":
                        det = next((x for x in g.values() if isinstance(x, h5py.Group) and
                                    str(_dec(x.attrs.get("canSAS_class", b""))) == "SASdetector"), None)
                        if det is not None:
                            gv = {k2: _num(_dec(det[k2][()])) for k2 in det if isinstance(det[k2], h5py.Dataset)}
                            if gv.get("beam_center_x") is not None and gv.get("beam_center_y") is not None:
                                img.centre_px = (gv["beam_center_x"], gv["beam_center_y"])
                            if gv.get("x_pixel_size"):
                                img.pixel_m = (gv["x_pixel_size"], gv.get("y_pixel_size") or gv["x_pixel_size"])
                            img.SD_m = gv.get("SDD")
                            img.wavelength_A = gv.get("wavelength")
                from ..sasred.legacy import rename_keys
                meta.update(rename_keys(meta))
                meta.setdefault("sample", meta.get("title", ""))
                if img.SD_m is None and _num(meta.get("SD_m")) is not None:
                    img.SD_m = _num(meta["SD_m"])
                if img.wavelength_A is None and _num(meta.get("wavelength_A")) is not None:
                    img.wavelength_A = _num(meta["wavelength_A"])
                if not img.has_q and "Qx" in d and "Qy" in d:
                    u = str(_dec(d["Qx"].attrs.get("units", b"1/nm"))).lower()
                    f = 10.0 if "ang" in u or u in ("1/a", "a^-1") else 1.0
                    qx, qy = np.asarray(d["Qx"][()], float) * f, np.asarray(d["Qy"][()], float) * f
                    if qx.shape == I.shape:
                        img._qxy = (qx, qy)
                out.append(img)
    return out


_FLOAT = re.compile(r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?")


def _read_bersans(path) -> Image2D:
    with open(path, encoding="utf-8", errors="replace") as fh:
        text = fh.read()
    meta: Dict[str, object] = {}
    sec = ""
    blocks: Dict[str, List[str]] = {}
    for ln in text.splitlines():
        s = ln.strip()
        if s.startswith("%"):
            sec = s[1:]
            continue
        if sec in ("Counts", "Errors"):
            blocks.setdefault(sec, []).append(s)
        elif "=" in s:
            k, v = s.split("=", 1)
            v = v.lstrip("=").strip()
            x = _num(v)
            meta[f"{sec}/{k.strip()}"] = x if x is not None and not re.search(r"[A-Za-z]", v) else v
    nx = int(_num(meta.get("File/DataSizeX")) or 128)
    ny = int(_num(meta.get("File/DataSizeY")) or 128)

    def arr(lines):
        # fixed-width numbers may touch: "4.329e-01-2.141e-01"
        v = np.array([float(x) for x in _FLOAT.findall(" ".join(lines))])
        return v[: nx * ny].reshape(ny, nx)          # same orientation as the SANS-I HDF counts
    I = arr(blocks["Counts"])
    dI = arr(blocks["Errors"]) if "Errors" in blocks else None
    meta["sample"] = meta.get("Sample/SampleName", os.path.basename(path))
    SD = _num(meta.get("Setup/SD"))
    lam = _num(meta.get("Setup/Lambda"))
    img = Image2D(I, dI, np.zeros(I.shape, bool), str(path), "bersans-2d", meta,
                  unit="1/cm/sr (BerSANS)", pixel_m=(SANS_I_PIXEL_M, SANS_I_PIXEL_M),
                  SD_m=SD, wavelength_A=10 * lam if lam else None)
    for kx, ky in (("History/BeamcenterX", "History/BeamcenterY"), ("Setup/BeamcenterX", "Setup/BeamcenterY")):
        if _num(meta.get(kx)) is not None and _num(meta.get(ky)) is not None:
            img.centre_px = (_num(meta[kx]) - 1.0, _num(meta[ky]) - 1.0)      # BerSANS counts from 1
            break
    return img


def read_images(path) -> List[Image2D]:
    path = str(path)
    with open(path, "rb") as fh:
        sig = fh.read(8)
    if sig.startswith(b"\x89HDF"):
        return _read_nxcansas(path)
    with open(path, encoding="utf-8", errors="replace") as fh:
        head = fh.read(2000)
    if "%File" in head and "SANSDAni" in head:
        return [_read_bersans(path)]
    raise ValueError(f"{os.path.basename(path)}: not a 2D NXcanSAS or BerSANS 2D file")


def image_from_reduced(reduced, key=None) -> Image2D:
    """2D image of a pySASred result in memory (:class:`~pysasfit.sasred.reduce.ReducedRun`,
    single run or table row, or :class:`~pysasfit.sasred.absolute.Reduced2D`)."""
    from ..sasred.io.metadata import run_metadata
    r = reduced
    raw = r.raw
    mask = getattr(r, "mask", None)
    if mask is None:
        mask = r.flags
    par = dict(getattr(r, "parameters", {}) or {})
    meta = run_metadata(raw)
    meta["sample"] = par.get("sample") or raw.sample_name
    if par.get("label"):
        meta["label"] = par["label"]
    ps = raw.pixel_size_m
    tag = key if key is not None else (par.get("label") or getattr(r, "run", ""))
    img = r.image
    return Image2D(np.asarray(img.value, float), np.asarray(img.std(systematic=False), float),
                   np.asarray(mask) != 0, f"pySASred:{tag}", "pySASred (memory)", meta,
                   unit=getattr(img, "unit", "") or getattr(r, "unit", ""),
                   centre_px=tuple(map(float, par.get("beam_center_px", (np.nan, np.nan)))),
                   pixel_m=(ps, ps) if np.isscalar(ps) else tuple(ps), SD_m=float(r.geometry.SD_m),
                   wavelength_A=float(raw.wavelength_A))
