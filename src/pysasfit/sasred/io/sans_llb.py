# SPDX-License-Identifier: BSD-3-Clause
"""SINQ SANS-LLB raw files (``sans-llb<year>n<run>.hdf``, NeXus ``entry0``).

SANS-LLB has three detector banks:

* ``central``: 128 x 128 monoblock (5 mm pixels) on the detector carriage (``central_detector``);
* ``left`` and ``bottom``: wide-angle panels of 16 3He tubes (13.4 mm apart, 5 mm pixels along the
  tubes) on a second carriage closer to the sample (``left_detector``, ``bottom_detector``). The direct
  beam never reaches them.

:func:`read_sans_llb` returns the run of the central detector as a :class:`~.sinq.RawRun` (everything
else in pySASred works on it unchanged: beam centre, masks, transmissions, catalogue, reduction table).
The panels travel along in ``RawRun.panels``; :func:`bank_run` turns one of them into a run of its own
whose geometry comes from the pixel positions (:func:`panel_geometry`), so that the same reduction
(dead time, attenuator, monitor, transmission, background, water, absolute scale per pixel, averaging)
works on it.

Coordinates: NICOS writes, for every bank, the pixel positions ``posx``/``posy`` [mm] in one frame
perpendicular to the beam (``<bank>_data/posx|posy``); the central detector has posx = 0, posy = 0 at
pixel (63.5, 63.5) and posx decreases with the column. The beam position in that frame follows from
the beam centre on the central detector (pixel indices). The positions of the panels as written are
nominal: their offsets (dx, dy along posx, posy and dz along the beam, in mm) are calibration values,
set under Settings → Preferences → pySASred → SANS-LLB wide-angle panels (or given to
:func:`panel_geometry`).
"""
from __future__ import annotations

import re
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import h5py
import numpy as np

from .sinq import MissingField, RawRun, _length_m, _read, sha256_of

__all__ = ["Panel", "read_sans_llb", "bank_run", "panel_geometry", "panel_offsets", "beam_position_mm",
           "panel_mask", "is_sans_llb", "BANKS"]

BANKS = ("left", "bottom")


@dataclass
class Panel:
    name: str
    counts: np.ndarray          # (ny, nx) as stored
    posx_mm: np.ndarray         # (nx,) nominal pixel centres
    posy_mm: np.ndarray         # (ny,)
    distance_m: float           # recorded sample-panel distance
    pixel_mm: Tuple[float, float]   # (x, y) pixel size
    tube_axis: int              # array axis along the tubes (left: 0, bottom: 1)
    deadtime_s: Optional[float] = None


def is_sans_llb(path) -> bool:
    try:
        with h5py.File(str(path), "r") as f:
            return "entry0/SANS-LLB" in f
    except OSError:
        return False


def _f(v):
    return float(np.asarray(v).ravel()[0])


def _llb_offset(key: str, default: float = 0.0) -> float:
    try:
        from ...core import settings as prefs
        return float(prefs.effective(key))
    except Exception:  # noqa: BLE001 -- no settings (scripts, tests)
        return default


def read_sans_llb(path, checksum: bool = True, distance_offset_m: Optional[float] = None) -> RawRun:
    """Read a SANS-LLB file: the central detector as the run, the wide-angle panels in ``panels``.
    ``distance_offset_m``: offset of the central detection layer from the recorded distance (None: the
    preference ``sasred.llb.central_dz_mm``)."""
    path = str(path)
    m = re.search(r"n0*(\d+)\.hdf$", Path(path).name)
    if not m:
        raise ValueError(f"cannot get the run number from '{path}'")
    missing: List[str] = []
    with h5py.File(path, "r") as f:
        g = f["entry0"]
        S = "SANS-LLB/"
        counts, _ = _read(g, S + "central_detector/data")
        counts = np.asarray(counts)
        n_frames = 1
        if counts.ndim == 3:
            n_frames = counts.shape[0]
            counts = counts.sum(axis=0)
        t, _ = _read(g, "control/count_time")
        lam, lu = _read(g, S + "velocity_selector/wavelength")
        lf = {"nm": 10.0, "angstrom": 1.0, "a": 1.0, "å": 1.0}.get(str(lu).lower()) if lu is not None else None
        if lf is None:
            raise ValueError(f"wavelength unit '{lu}' unknown or missing")
        dlam, dlu = _read(g, S + "velocity_selector/wavelength_spread", False, missing)
        if dlam is not None and str(dlu).strip() == "%":
            dlam = float(dlam) / 100.0
        SD, SDu = _read(g, S + "central_detector/distance")
        coll, collu = _read(g, S + "collimator/geometry/size")
        pxs, pxu = _read(g, S + "central_detector/x_pixel_size")
        pys, pyu = _read(g, S + "central_detector/y_pixel_size")
        cx, cxu = _read(g, S + "central_detector/x", False, missing)
        monitors = {}
        for k in ("monitor0", "monitor1", "monitor2", "monitor3", "monitor4"):
            v, _ = _read(g, f"{k}/integral", False, missing)
            if v is not None:
                monitors[k] = float(v)
        pc, _ = _read(g, "protoncount/integral", False, missing)
        if pc is not None:
            monitors["protoncount"] = float(pc)
        if not monitors:
            raise MissingField("no monitor")
        preset, _ = _read(g, "control/preset")
        mode, _ = _read(g, "control/mode")
        att, _ = _read(g, S + "attenuator/selection")
        bsx, _ = _read(g, S + "beam_stop/x")
        bsy, _ = _read(g, S + "beam_stop/y")
        bss, bsu = _read(g, S + "beam_stop/size", False, missing)
        name, _ = _read(g, "sample/name")
        pos, _ = _read(g, "sample/x", False, missing)
        thk, thu = _read(g, "sample/thickness", False, missing)
        start, _ = _read(g, "start_time")
        end, _ = _read(g, "end_time", False, missing)
        cpos = (np.asarray(g["central_data/posx"][()], float), np.asarray(g["central_data/posy"][()], float))
        panels: Dict[str, Panel] = {}
        for b in BANKS:
            try:
                d = g[S + f"{b}_detector"]
                c = np.asarray(d["data"][()])
                if c.ndim == 3:
                    c = c.sum(axis=0)
                dist, du = _read(d, "distance")
                ppx, ppxu = _read(d, "x_pixel_size")
                ppy, ppyu = _read(d, "y_pixel_size")
                dtv, _ = _read(d, "deadtime", False, [])
                px = np.asarray(g[f"{b}_data/posx"][()], float)
                py = np.asarray(g[f"{b}_data/posy"][()], float)
            except KeyError:
                missing.append(f"{b}_detector")
                continue
            if c.shape != (len(py), len(px)):
                raise ValueError(f"{b} detector: data {c.shape} does not fit posy {len(py)} x posx {len(px)}")
            axis = 0 if c.shape[0] > c.shape[1] else 1
            panels[b] = Panel(b, c, px, py, _length_m(dist, du, f"{b}_detector/distance"),
                              (_length_m(ppx, ppxu, "x_pixel_size") * 1e3, _length_m(ppy, ppyu, "y_pixel_size") * 1e3),
                              axis, None if dtv is None else float(dtv))
    if thk is not None and thu is not None and str(thu).lower() != "cm":
        thk = _length_m(thk, thu, "sample/thickness") * 100.0
    mon_preset = "monitor2" if "monitor2" in monitors else sorted(monitors)[0]
    run = RawRun(
        path=path, run=int(m.group(1)), sha256=sha256_of(path) if checksum else "",
        counts=counts, n_frames=n_frames, counting_time_s=float(t), monitors=monitors,
        preset=float(preset), count_mode=str(mode), wavelength_A=float(lam) * lf,
        dlam_over_lam_fwhm=None if dlam is None else float(dlam),
        coll_m=_length_m(coll, collu, "collimator/geometry/size"),
        SD_m=_length_m(SD, SDu, "central_detector/distance"),
        pixel_size_m=(_length_m(pxs, pxu, "x_pixel_size"), _length_m(pys, pyu, "y_pixel_size")),
        detector_offset_m=(None if cx is None else _length_m(cx, cxu, "central_detector/x"), None),
        detector_tilt_deg=0.0, attenuator=str(att).strip(), beamstop_xy_mm=(_f(bsx), _f(bsy)),
        beamstop_type="" if bss is None else f"{_f(bss):g} mm",
        sample_name=str(name), sample_position_mm=None if pos is None else _f(pos),
        sample_thickness_cm=None if thk is None else float(thk),
        start_time=str(start), end_time=end, missing_optional=missing,
        instrument="sans_llb", preset_monitor=mon_preset, panels=panels or None, central_pos_mm=cpos)
    off = _llb_offset("sasred.llb.central_dz_mm") / 1000.0 if distance_offset_m is None else float(distance_offset_m)
    if off:
        run.SD_offset_m = off
    return run


def beam_position_mm(raw: RawRun, beam_center_px) -> Tuple[float, float]:
    """Beam position (posx, posy) [mm] in the frame of the panels, from the beam centre (column, row)
    on the central detector."""
    px, py = raw.central_pos_mm
    cx, cy = beam_center_px
    ix, iy = np.arange(len(px), dtype=float), np.arange(len(py), dtype=float)
    # linear inside and beyond (constant pixel pitch)
    def lin(i, pos, v):
        k = (pos[-1] - pos[0]) / (i[-1] - i[0])
        return float(pos[0] + k * (v - i[0]))
    return lin(ix, px, cx), lin(iy, py, cy)


def panel_offsets(name: str) -> Tuple[float, float, float]:
    """(dx, dy, dz) [mm] of a panel from the preferences (0 without settings)."""
    return tuple(_llb_offset(f"sasred.llb.{name}.{k}_mm") for k in ("dx", "dy", "dz"))


def bank_run(raw: RawRun, name: str) -> RawRun:
    """The run of one wide-angle panel: counts of the panel, everything else of the run. ``SD_m`` stays
    the distance of the central detector (configuration key of beam centre, mask and grouping); the
    geometry comes from :func:`panel_geometry`."""
    if name in ("", "central"):
        return raw
    if not raw.panels or name not in raw.panels:
        raise KeyError(f"run {raw.run}: no {name} detector")
    p = raw.panels[name]
    return replace(raw, counts=p.counts, pixel_size_m=(p.pixel_mm[0] / 1e3, p.pixel_mm[1] / 1e3),
                   bank=name, panel=p, panels=None)


def panel_geometry(raw: RawRun, beam_center_px, offsets_mm: Optional[Tuple[float, float, float]] = None):
    """:class:`~pysasfit.sasred.core.geometry.PixelGeometry` of a panel run. Pixel centre in the frame of
    the central detector's pixel_geometry (x grows with the central column, i.e. against posx; y with
    posy), panel plane perpendicular to the beam at the recorded distance + dz:

        x = -(posx + dx - xb),  y = posy + dy - yb,  z = L + dz,  ΔΩ = A z / |r|³."""
    from ..core.geometry import PixelGeometry
    p: Panel = raw.panel
    dx, dy, dz = panel_offsets(p.name) if offsets_mm is None else offsets_mm
    xb, yb = beam_position_mm(raw, beam_center_px)
    X, Y = np.meshgrid(p.posx_mm, p.posy_mm)
    x = -(X + dx - xb) / 1e3
    y = (Y + dy - yb) / 1e3
    z = np.full_like(x, p.distance_m + dz / 1e3)
    d = np.sqrt(x * x + y * y + z * z)
    two_theta = np.arctan2(np.hypot(x, y), z)
    k = 2.0 * np.pi / raw.wavelength_A
    qx, qy, qz = k * x / d, k * y / d, k * (z / d - 1.0)
    q = 2.0 * k * np.sin(0.5 * two_theta)
    A = p.pixel_mm[0] * p.pixel_mm[1] / 1e6
    cos_psi = z / d
    return PixelGeometry(x, y, z, two_theta, np.arctan2(y, x), np.arccos(cos_psi), q, qx, qy, qz,
                         A * cos_psi / d ** 2, float(raw.wavelength_A), float(p.distance_m + dz / 1e3), 0.0)


def panel_mask(raw: RawRun, end_px: Optional[int] = None) -> np.ndarray:
    """Mask (reason bits) of a panel run: the ``end_px`` pixels at both ends of every tube (border bit;
    preference ``sasred.llb.end_px``) and tubes without counts (efficiency bit)."""
    from ..core import mask as mk
    p: Panel = raw.panel
    n = int(_llb_offset("sasred.llb.end_px", 10)) if end_px is None else int(end_px)
    m = np.zeros(p.counts.shape, np.uint32)
    if n > 0:
        if p.tube_axis == 0:
            m[:n, :] |= mk.BORDER
            m[-n:, :] |= mk.BORDER
        else:
            m[:, :n] |= mk.BORDER
            m[:, -n:] |= mk.BORDER
    tube_sum = p.counts.sum(axis=p.tube_axis)
    dead = tube_sum <= 0
    if dead.any():
        if p.tube_axis == 0:
            m[:, dead] |= mk.EFFICIENCY
        else:
            m[dead, :] |= mk.EFFICIENCY
    return m
