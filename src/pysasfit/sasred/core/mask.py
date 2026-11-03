# SPDX-License-Identifier: BSD-3-Clause
"""Detector masks: reason bits, the standard border, a suggested beamstop and hot-pixel
mask from the data, pixel selection by shapes, and mask files.

A mask is a ``uint32`` array of the detector shape; 0 = pixel used, any set bit = excluded,
the bit says why:

=====  ==========  =====================================================================
bit    name        meaning
=====  ==========  =====================================================================
1      border      the standard mask: outer rows and columns of the detector
2      beamstop    beamstop shadow (suggested from the data, or the disc on the panel)
4      user        drawn or loaded by the user
8      deadtime    dead-time factor above the flag limit (set by the reduction)
16     hot         isolated pixels far above their neighbourhood (suggested)
32     efficiency  no detector-efficiency estimate (whole row or column masked in the water run)
=====  ==========  =====================================================================

Mask files (:func:`save_mask` / :func:`load_mask`):

* ``.h5``: NeXus-like HDF5, ``/entry/instrument/detector/pixel_mask`` (uint32, these bits,
  the bit names as attribute) with the beam centre, run and date; the default
* ``.npy``: the uint32 array
* ``.txt``: 0/1 matrix (1 = masked; loaded as *user*), e.g. for other programs
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Dict, Optional

import numpy as np

__all__ = ["BORDER", "BEAMSTOP", "USER", "DEADTIME", "HOT", "EFFICIENCY", "BITS", "border_mask", "beamstop_guess",
           "suggest_beamstop",
           "suggest_hot_pixels", "pixels_in_rectangle", "pixels_in_ellipse", "pixels_in_polygon",
           "pixels_in_disc", "save_mask", "load_mask", "summary", "suggest_mask"]

BORDER, BEAMSTOP, USER, DEADTIME, HOT, EFFICIENCY = 1, 2, 4, 8, 16, 32
BITS = {BORDER: "border", BEAMSTOP: "beamstop", USER: "user", DEADTIME: "deadtime", HOT: "hot",
        EFFICIENCY: "efficiency"}


def border_mask(shape, width: int = 2) -> np.ndarray:
    """The standard mask: ``width`` outer rows and columns on every side."""
    m = np.zeros(shape, np.uint32)
    if width > 0:
        m[:width, :] = BORDER
        m[-width:, :] = BORDER
        m[:, :width] = BORDER
        m[:, -width:] = BORDER
    return m


def beamstop_guess(shape, beam_center_px, size_px: float, kind: str = "square") -> np.ndarray:
    """First guess of the beamstop shadow before the user adjusts it: a square of half width
    ``size_px`` (the SANS-I beamstops are square) or a disc of radius ``size_px`` around the beam
    centre. Pixel centres within the shape are selected."""
    yy, xx = np.mgrid[:shape[0], :shape[1]]
    x, y = float(beam_center_px[0]), float(beam_center_px[1])
    if size_px <= 0:
        return np.zeros(shape, bool)
    if kind == "square":
        return (np.abs(xx - x) <= size_px) & (np.abs(yy - y) <= size_px)
    if kind == "disc":
        return (xx - x) ** 2 + (yy - y) ** 2 <= size_px ** 2
    raise ValueError(f"beamstop shape '{kind}' (square or disc)")


def suggest_beamstop(counts, beam_center_px, valid: Optional[np.ndarray] = None, search_radius_px: float = 20.0,
                     drop: float = 0.3, smooth: int = 3, grow_px: int = 1, shape: str = "rectangle"):
    """Beamstop shadow: the region around the beam centre where the intensity drops suddenly.

    The image is median-filtered (``smooth`` pixels) against counting noise. The *rim level*
    is the 90th percentile within ``search_radius_px`` of the beam centre (the bright
    scattering around the beamstop); every pixel below ``drop`` × rim level is "low". The
    shadow is the connected region of low pixels that reaches to within 3 pixels of the beam
    centre, with holes filled and grown by ``grow_px`` pixels (the penumbra). With ``shape``
    "rectangle" (default; the SANS-I beamstops are square) the whole bounding rectangle of that
    region is returned, with "shadow" the region itself.

    Returns (bool mask, info dict); the mask is empty with ``info["reason"]`` when there is
    no shadow at the beam centre (e.g. a direct-beam run without beamstop)."""
    from scipy import ndimage
    c = np.asarray(counts, float)
    ny, nx = c.shape
    ok = np.ones(c.shape, bool) if valid is None else np.asarray(valid, bool)
    s = ndimage.median_filter(c, size=smooth, mode="nearest") if smooth and smooth > 1 else c
    yy, xx = np.mgrid[:ny, :nx]
    xc, yc = beam_center_px
    r = np.hypot(xx - xc, yy - yc)
    near = ok & (r <= search_radius_px)
    info = {"search_radius_px": search_radius_px, "drop": drop, "smooth": smooth, "grow_px": grow_px}
    empty = np.zeros(c.shape, bool)
    if not near.any():
        info["reason"] = "beam centre outside the valid detector area"
        return empty, info
    rim = float(np.percentile(s[near], 90))
    info["rim_level"] = rim
    if rim <= 0:
        info["reason"] = "no counts around the beam centre"
        return empty, info
    thr = drop * rim
    info["threshold"] = thr
    low = ok & (s < thr)
    seed = low & (r <= 3.0)
    if not seed.any():
        info["reason"] = "no sudden drop at the beam centre (no beamstop in the beam?)"
        return empty, info
    lbl, _n = ndimage.label(low)
    ids = np.unique(lbl[seed])
    ids = ids[ids > 0]
    region = np.isin(lbl, ids)
    region = ndimage.binary_fill_holes(region)
    if grow_px > 0:
        region = ndimage.binary_dilation(region, iterations=int(grow_px))
    region &= ok
    if shape == "rectangle":
        ys, xs = np.nonzero(region)
        rect = np.zeros(c.shape, bool)
        rect[ys.min():ys.max() + 1, xs.min():xs.max() + 1] = True
        region = rect & ok
    elif shape != "shadow":
        raise ValueError(f"beamstop shape '{shape}' (rectangle or shadow)")
    info["shape"] = shape
    n = int(region.sum())
    ys, xs = np.nonzero(region)
    info.update(n_pixels=n, bbox=(int(xs.min()), int(xs.max()), int(ys.min()), int(ys.max())),
                equivalent_radius_px=float(np.sqrt(n / np.pi)),
                shadow_median=float(np.median(c[region])) if n else float("nan"))
    if n > 0.25 * ok.sum():
        info["warning"] = "the suggested beamstop covers more than 25 % of the detector: lower the drop fraction"
    return region, info


def suggest_hot_pixels(counts, valid: Optional[np.ndarray] = None, factor: float = 5.0, n_sigma: float = 10.0,
                       size: int = 5, exclude: Optional[np.ndarray] = None):
    """Isolated pixels far above their neighbourhood: counts > ``factor`` × local median and
    more than ``n_sigma`` Poisson standard deviations above it (local median over
    ``size`` × ``size`` pixels). ``exclude``: pixels never flagged (e.g. near the beam centre)."""
    from scipy import ndimage
    c = np.asarray(counts, float)
    ok = np.ones(c.shape, bool) if valid is None else np.asarray(valid, bool)
    med = ndimage.median_filter(c, size=size, mode="nearest")
    base = np.maximum(med, 1.0)
    hot = ok & (c > factor * base) & (c - med > n_sigma * np.sqrt(base))
    if exclude is not None:
        hot &= ~np.asarray(exclude, bool)
    return hot


def suggest_mask(counts, beam_center_px, border_width: int = 2, search_radius_px: float = 20.0,
                 drop: float = 0.3, grow_px: int = 1, hot: bool = True, beamstop_shape: str = "rectangle"):
    """The mask suggested for one image: standard border + beamstop shadow + hot pixels.

    Hot pixels next to the beamstop (within ``grow_px`` + 2 pixels) are usually beam leaking past
    its edge; they are counted separately in the report. Without a beamstop shadow (direct
    beam), no hot pixels are searched within ``search_radius_px`` of the beam centre.
    Returns (uint32 mask, report dict)."""
    from scipy import ndimage
    c = np.asarray(counts, float)
    m = border_mask(c.shape, border_width)
    valid = m == 0
    bs, info = suggest_beamstop(c, beam_center_px, valid=valid, search_radius_px=search_radius_px, drop=drop,
                                grow_px=grow_px, shape=beamstop_shape)
    m[bs] |= BEAMSTOP
    report = {"border": int(np.count_nonzero(m & BORDER)), "beamstop": info, "hot": 0, "leak": 0}
    if hot:
        if bs.any():
            exclude = bs
        else:
            yy, xx = np.mgrid[:c.shape[0], :c.shape[1]]
            exclude = np.hypot(xx - beam_center_px[0], yy - beam_center_px[1]) <= search_radius_px
        h = suggest_hot_pixels(c, valid=valid, exclude=exclude)
        if bs.any():
            near = ndimage.binary_dilation(bs, iterations=int(grow_px) + 2)
            report["leak"] = int(np.count_nonzero(h & near))
        report["hot"] = int(np.count_nonzero(h))
        m[h] |= HOT
    return m, report


# ------------------------------------------------------------------ shapes -> pixels (pixel centres)
def _grid(shape):
    yy, xx = np.mgrid[:shape[0], :shape[1]]
    return xx.astype(float), yy.astype(float)


def _pixel(v):
    """Index of the pixel that contains the coordinate ``v`` (pixel i covers i − 0.5 … i + 0.5)."""
    return int(np.floor(float(v) + 0.5))


def pixels_in_rectangle(shape, x0, x1, y0, y1):
    """Every pixel touched by the rectangle (x0, y0)-(x1, y1), including the pixels of both corners:
    a drag within one row (or a single click) gives that row (or pixel)."""
    xx, yy = _grid(shape)
    xa, xb = sorted((_pixel(x0), _pixel(x1)))
    ya, yb = sorted((_pixel(y0), _pixel(y1)))
    return (xx >= xa) & (xx <= xb) & (yy >= ya) & (yy <= yb)


def pixels_in_ellipse(shape, x0, x1, y0, y1):
    """Ellipse inscribed in the pixels touched by the rectangle (x0, y0)-(x1, y1): it reaches the
    pixels of the four sides, so a flat ellipse within one row still selects that row."""
    xx, yy = _grid(shape)
    xa, xb = sorted((_pixel(x0), _pixel(x1)))
    ya, yb = sorted((_pixel(y0), _pixel(y1)))
    cx, cy = 0.5 * (xa + xb), 0.5 * (ya + yb)
    ax, ay = 0.5 * (xb - xa) + 0.5, 0.5 * (yb - ya) + 0.5
    return ((xx - cx) / ax) ** 2 + ((yy - cy) / ay) ** 2 <= 1.0 + 1e-9


def pixels_in_disc(shape, x, y, radius):
    xx, yy = _grid(shape)
    return (xx - x) ** 2 + (yy - y) ** 2 <= max(radius, 0.5) ** 2


def pixels_in_polygon(shape, vertices):
    """Pixels whose centre is inside the polygon, plus every pixel its edges pass through (so the
    corner pixels and thin polygons along a row are included)."""
    from matplotlib.path import Path
    xx, yy = _grid(shape)
    v = np.asarray(vertices, float)
    pts = np.column_stack([xx.ravel(), yy.ravel()])
    sel = Path(v).contains_points(pts, radius=1e-9).reshape(shape)
    closed = np.vstack([v, v[:1]])
    for (xa, ya), (xb, yb) in zip(closed[:-1], closed[1:]):
        n = int(np.ceil(4 * max(abs(xb - xa), abs(yb - ya)))) + 1     # 4 samples per pixel
        t = np.linspace(0.0, 1.0, n + 1)
        ix = np.floor(xa + t * (xb - xa) + 0.5).astype(int)
        iy = np.floor(ya + t * (yb - ya) + 0.5).astype(int)
        ok = (ix >= 0) & (ix < shape[1]) & (iy >= 0) & (iy < shape[0])
        sel[iy[ok], ix[ok]] = True
    return sel


# ------------------------------------------------------------------ files
def summary(mask) -> Dict[str, int]:
    """Number of pixels per reason (a pixel can have several) and in total."""
    m = np.asarray(mask, np.uint32)
    out = {name: int(np.count_nonzero(m & bit)) for bit, name in BITS.items()}
    out["total"] = int(np.count_nonzero(m))
    return out


def save_mask(path, mask, beam_center_px=None, run=None, description=""):
    path = str(path)
    m = np.asarray(mask, np.uint32)
    if path.lower().endswith(".npy"):
        np.save(path, m)
        return path
    if path.lower().endswith(".txt"):
        np.savetxt(path, (m != 0).astype(int), fmt="%d",
                   header="pySASred mask: 1 = masked, 0 = used; row = y (axis 0), column = x (axis 1)")
        return path
    import h5py
    from .. import __version__
    with h5py.File(path, "w") as f:
        e = f.create_group("entry")
        e.attrs["NX_class"] = "NXentry"
        d = e.create_group("instrument/detector")
        e["instrument"].attrs["NX_class"] = "NXinstrument"
        d.attrs["NX_class"] = "NXdetector"
        ds = d.create_dataset("pixel_mask", data=m, compression="gzip")
        ds.attrs["bits"] = json.dumps({str(k): v for k, v in BITS.items()})
        ds.attrs["convention"] = "0 = pixel used; set bits give the reasons for excluding it"
        if beam_center_px is not None:
            d["beam_center_px"] = np.asarray(beam_center_px, float)
        p = e.create_group("process")
        p["program"] = f"pySASred {__version__}"
        p["date"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
        if run is not None:
            p["run"] = str(run)
        if description:
            p["description"] = description
    return path


def load_mask(path, shape=None):
    """Mask from a file written by :func:`save_mask` (or any .npy / 0-1 text matrix).
    Returns (uint32 mask, info dict). ``shape`` is checked when given."""
    path = str(path)
    info = {"path": path}
    if path.lower().endswith(".npy"):
        m = np.asarray(np.load(path), np.uint32)
    elif path.lower().endswith((".txt", ".dat", ".msk")):
        m = (np.loadtxt(path) != 0).astype(np.uint32) * USER
    else:
        import h5py
        with h5py.File(path, "r") as f:
            m = np.asarray(f["entry/instrument/detector/pixel_mask"][()], np.uint32)
            if "entry/instrument/detector/beam_center_px" in f:
                info["beam_center_px"] = tuple(float(v) for v in f["entry/instrument/detector/beam_center_px"][()])
            if "entry/process/run" in f:
                info["run"] = f["entry/process/run"][()].decode() if isinstance(f["entry/process/run"][()], bytes) \
                    else str(f["entry/process/run"][()])
    if shape is not None and tuple(m.shape) != tuple(shape):
        raise ValueError(f"mask shape {m.shape} does not match the detector {tuple(shape)}")
    return m, info


def default_mask(raw, beam_center_px, **kw):
    """Suggested mask of a raw run (:func:`suggest_mask` of its counts); a SANS-LLB wide-angle panel
    gets the tube ends and dead tubes (:func:`pysasfit.sasred.io.sans_llb.panel_mask`). Returns
    (uint32 mask, report dict)."""
    if getattr(raw, "panel", None) is not None:
        from ..io.sans_llb import panel_mask
        m = panel_mask(raw)
        return m, {"panel": raw.bank, "masked": int((m != 0).sum())}
    return suggest_mask(raw.counts, beam_center_px, **kw)
