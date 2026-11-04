# SPDX-License-Identifier: BSD-3-Clause
"""Analysis of reduced 2D images (:class:`~pysasfit.toolbox.image2d.Image2D`), in the spirit of GRASP
and BerSANS. Every result is a :class:`~pysasfit.toolbox.data1d.Data1D` (or a new image).

Averages over the valid (unmasked) pixels of each bin: I = mean of the pixel values, σ =
sqrt(Σσ_j²)/n (statistical, independent pixels), x = mean of the pixel coordinate, dx = its spread.

* :func:`radial_average` – I(q), optionally in a sector ψ0 ± Δψ (and the opposite sector)
* :func:`azimuthal_average` – I(ψ) in a ring q1 ≤ q ≤ q2
* :func:`box_profile` – projection on the long axis of a box through a centre, tilted by an angle
* :func:`sin2_decomposition` – I(q, ψ) = A(q) + B(q) sin²(ψ − ψ0) per q ring (weighted least squares
  over the pixels); for magnetic SANS with the field along ψ0: A = nuclear (+ magnetic ∥), B = magnetic ⊥
* :func:`sum_images` – sum or mean of images of the same detector
* :func:`region_integral` / :func:`rocking_curve` – mean intensity in a region (q range, sector) of
  each image versus a header item (e.g. a rocking angle)

Angles in degrees, ψ = 0 along +x (horizontal), counter-clockwise; q in 1/nm.

Mask helpers select pixels inside shapes drawn in any of the views (pixels, qx–qy, ψ–q):
:func:`pixels_in_shape`.
"""
from __future__ import annotations

from typing import Callable, Optional, Sequence, Tuple

import numpy as np

from .data1d import Data1D
from .image2d import Image2D

__all__ = ["radial_average", "azimuthal_average", "box_profile", "sin2_decomposition", "sum_images",
           "region_integral", "rocking_curve", "view_coordinates", "pixels_in_shape", "q_edges",
           "sector_mask", "beam_centre_of_gravity"]


def q_edges(qmin: float, qmax: float, n: int, spacing: str = "lin") -> np.ndarray:
    return np.geomspace(qmin, qmax, n + 1) if spacing == "log" else np.linspace(qmin, qmax, n + 1)


def _pixels(img: Image2D):
    v = img.valid()
    s = img.dI if img.dI is not None else np.full(img.I.shape, np.nan)
    return v, np.asarray(img.I, float), np.asarray(s, float)


def _binned(coord, edges, valid, I, s):
    """Mean of the valid pixels per bin of ``coord``: x, dx, y, dy, n (empty bins dropped)."""
    c = np.asarray(coord, float).ravel()
    v = valid.ravel() & np.isfinite(c)
    b = np.searchsorted(edges, c, side="right") - 1
    v &= (b >= 0) & (b < len(edges) - 1)
    nb = len(edges) - 1
    b, cv, Iv, sv = b[v], c[v], I.ravel()[v], s.ravel()[v]
    n = np.bincount(b, minlength=nb).astype(float)
    with np.errstate(invalid="ignore", divide="ignore"):
        x = np.bincount(b, cv, nb) / n
        x2 = np.bincount(b, cv * cv, nb) / n
        y = np.bincount(b, Iv, nb) / n
        dy = np.sqrt(np.bincount(b, np.nan_to_num(sv) ** 2, nb)) / n
        if np.all(np.isnan(sv)):
            dy = np.full(nb, np.nan)
    keep = n > 0
    return x[keep], np.sqrt(np.clip(x2 - x * x, 0, None))[keep], y[keep], dy[keep], n[keep]


def sector_mask(img: Image2D, psi0: float, half_width: float, both: bool = True) -> np.ndarray:
    """Pixels with ψ within ψ0 ± half width (and ψ0 + 180° ± half width with ``both``)."""
    psi = img.psi_deg()
    d = (psi - psi0 + 180.0) % 360.0 - 180.0
    m = np.abs(d) <= half_width
    if both:
        d2 = (psi - psi0) % 360.0 - 180.0
        m |= np.abs(d2) <= half_width
    return m


def radial_average(img: Image2D, edges: Optional[np.ndarray] = None, n: int = 60, spacing: str = "log",
                   sector: Optional[Tuple[float, float, bool]] = None) -> Data1D:
    """I(q) of the valid pixels; ``sector`` = (ψ0, half width, both sides)."""
    v, I, s = _pixels(img)
    q = img.q()
    if sector is not None:
        v = v & sector_mask(img, *sector)
    if edges is None:
        qv = q[v & (q > 0)]
        edges = q_edges(float(qv.min()), float(qv.max()) * (1 + 1e-9), n, spacing)
    x, dx, y, dy, cnt = _binned(q, edges, v, I, s)
    what = "radial average" if sector is None else \
        f"sector {sector[0]:g}° ± {sector[1]:g}°" + (" (both sides)" if (len(sector) < 3 or sector[2]) else "")
    return Data1D(x, y, dy, dx, name=f"{img.sample} {what}", yunit=img.unit,
                  meta=dict(img.meta, pixels=int(cnt.sum())), source=f"2D {what}: {img.name}")


def azimuthal_average(img: Image2D, q1: float, q2: float, n: int = 36) -> Data1D:
    """I(ψ) of the valid pixels with q1 ≤ q ≤ q2, ``n`` bins over 0…360°."""
    v, I, s = _pixels(img)
    q = img.q()
    v = v & (q >= q1) & (q <= q2)
    x, dx, y, dy, _ = _binned(img.psi_deg(), np.linspace(0, 360, n + 1), v, I, s)
    return Data1D(x, y, dy, dx, name=f"{img.sample} I(ψ), q {q1:g}–{q2:g}", xlabel="ψ", xunit="deg",
                  yunit=img.unit, meta=dict(img.meta), source=f"2D azimuthal average: {img.name}")


def box_profile(img: Image2D, angle: float, half_length: float, width: float, n: int = 60,
                centre: Tuple[float, float] = (0.0, 0.0)) -> Data1D:
    """Pixels in a box of length 2·half_length and ``width`` (q units) through ``centre`` (qx, qy),
    its long axis at ``angle`` to +x, projected on the long axis: I(u), u from −half_length to
    +half_length."""
    v, I, s = _pixels(img)
    qx, qy = img.qxy()
    a = np.radians(angle)
    dxq, dyq = qx - centre[0], qy - centre[1]
    u = dxq * np.cos(a) + dyq * np.sin(a)
    w = -dxq * np.sin(a) + dyq * np.cos(a)
    v = v & (np.abs(w) <= width / 2) & (np.abs(u) <= half_length)
    x, dx, y, dy, _ = _binned(u, np.linspace(-half_length, half_length, n + 1), v, I, s)
    return Data1D(x, y, dy, dx, name=f"{img.sample} box {angle:g}°, width {width:g}", xlabel="u", xunit="1/nm",
                  yunit=img.unit, meta=dict(img.meta), source=f"2D box projection: {img.name}")


def sin2_decomposition(img: Image2D, edges: Optional[np.ndarray] = None, n: int = 40, spacing: str = "log",
                       psi0: float = 0.0) -> Tuple[Data1D, Data1D, np.ndarray]:
    """I(q, ψ) = A(q) + B(q) sin²(ψ − ψ0), fitted per q bin to the valid pixels (weights 1/σ², equal
    weights without σ). Returns (A, B, χ²/ndf per bin)."""
    v, I, s = _pixels(img)
    q = img.q()
    if edges is None:
        qv = q[v & (q > 0)]
        edges = q_edges(float(qv.min()), float(qv.max()) * (1 + 1e-9), n, spacing)
    S = np.sin(np.radians(img.psi_deg() - psi0)) ** 2
    b = np.searchsorted(edges, q, side="right") - 1
    nb = len(edges) - 1
    qs, A, sA, B, sB, chi = [], [], [], [], [], []
    for k in range(nb):
        m = v & (b == k)
        if m.sum() < 3:
            continue
        y, x, e = I[m], S[m], s[m]
        w = 1.0 / e ** 2 if np.all(np.isfinite(e) & (e > 0)) else np.ones_like(y)
        M = np.array([[w.sum(), (w * x).sum()], [(w * x).sum(), (w * x * x).sum()]])
        if abs(np.linalg.det(M)) < 1e-12 * max(1.0, abs(M).max()) ** 2:
            continue                                     # no spread of ψ in this ring (sector of the detector)
        C = np.linalg.inv(M)
        a, bb = C @ np.array([(w * y).sum(), (w * x * y).sum()])
        r = y - a - bb * x
        ndf = max(1, m.sum() - 2)
        c2 = float((w * r * r).sum() / ndf)
        scale = 1.0 if np.all(np.isfinite(e)) else c2      # unknown σ: from the scatter
        qs.append(q[m].mean())
        A.append(a)
        B.append(bb)
        sA.append(np.sqrt(C[0, 0] * scale))
        sB.append(np.sqrt(C[1, 1] * scale))
        chi.append(c2)
    qs = np.array(qs)
    base = dict(img.meta, psi0=psi0)
    dA = Data1D(qs, np.array(A), np.array(sA), None, name=f"{img.sample} A(q) (ψ0 {psi0:g}°)", yunit=img.unit,
                meta=base, source=f"2D A + B sin²(ψ−{psi0:g}°): {img.name}")
    dB = Data1D(qs, np.array(B), np.array(sB), None, name=f"{img.sample} B(q) (ψ0 {psi0:g}°)", yunit=img.unit,
                meta=base, source=f"2D A + B sin²(ψ−{psi0:g}°): {img.name}")
    return dA, dB, np.array(chi)


def sum_images(imgs: Sequence[Image2D], mean: bool = False) -> Image2D:
    """Sum (or mean) of images of the same detector shape; a pixel masked in any image is masked."""
    if not imgs:
        raise ValueError("no images")
    shape = imgs[0].I.shape
    if any(im.I.shape != shape for im in imgs):
        raise ValueError("the images have different shapes")
    I = np.sum([im.I for im in imgs], axis=0)
    have = all(im.dI is not None for im in imgs)
    dI = np.sqrt(np.sum([im.dI ** 2 for im in imgs], axis=0)) if have else None
    if mean:
        I = I / len(imgs)
        dI = dI / len(imgs) if dI is not None else None
    mask = np.any([im.mask for im in imgs], axis=0)
    im0 = imgs[0]
    names = ", ".join(im.sample or im.name for im in imgs[:4]) + (" …" if len(imgs) > 4 else "")
    out = Image2D(I, dI, mask, f"{'mean' if mean else 'sum'} of {len(imgs)}", "sum",
                  dict(im0.meta, sample=f"{'mean' if mean else 'sum'} of {len(imgs)}: {names}"),
                  unit=im0.unit, centre_px=im0.centre_px, pixel_m=im0.pixel_m, SD_m=im0.SD_m,
                  wavelength_A=im0.wavelength_A, entry=str(id(imgs)))
    out._qxy = im0._qxy
    return out


def region_integral(img: Image2D, q1: Optional[float] = None, q2: Optional[float] = None,
                    sector: Optional[Tuple[float, float, bool]] = None) -> Tuple[float, float, int]:
    """Mean intensity (± σ) of the valid pixels in a q range (and sector); whole detector without them."""
    v, I, s = _pixels(img)
    if img.has_q and (q1 is not None or q2 is not None or sector is not None):
        q = img.q()
        if q1 is not None:
            v = v & (q >= q1)
        if q2 is not None:
            v = v & (q <= q2)
        if sector is not None:
            v = v & sector_mask(img, *sector)
    n = int(v.sum())
    if n == 0:
        return np.nan, np.nan, 0
    return float(I[v].mean()), float(np.sqrt(np.nansum(s[v] ** 2)) / n), n


def rocking_curve(imgs: Sequence[Image2D], key: str, **region) -> Data1D:
    """Mean intensity in a region of every image versus the header item ``key`` (sorted)."""
    xs, ys, ss = [], [], []
    for im in imgs:
        try:
            x = float(im.meta.get(key))
        except (TypeError, ValueError):
            continue
        y, s, _ = region_integral(im, **region)
        xs.append(x)
        ys.append(y)
        ss.append(s)
    if not xs:
        raise ValueError(f"no image has a number for {key!r}")
    o = np.argsort(xs)
    lab = key.split("/")[-1]
    return Data1D(np.array(xs)[o], np.array(ys)[o], np.array(ss)[o], None, name=f"rocking: I vs {lab}",
                  xlabel=lab, xunit="", yunit=imgs[0].unit, source=f"2D rocking curve over {len(xs)} images")


# ------------------------------------------------------------------------------------- shapes / masks
def view_coordinates(img: Image2D, view: str):
    """Coordinates of every pixel in a view: 'pixels' (x, y), 'q' (qx, qy) or 'polar' (ψ in deg, q)."""
    if view == "pixels":
        iy, ix = np.mgrid[0:img.I.shape[0], 0:img.I.shape[1]]
        return ix.astype(float), iy.astype(float)
    if view == "q":
        return img.qxy()
    return img.psi_deg(), img.q()


def pixels_in_shape(img: Image2D, view: str, kind: str, pts) -> np.ndarray:
    """Pixels whose centre lies in a shape drawn in ``view``: 'rectangle' or 'ellipse' with
    pts = (x0, x1, y0, y1), 'polygon' with pts = [(x, y), …], 'above'/'below' with pts = value."""
    if kind in ("above", "below"):
        I = np.asarray(img.I, float)
        return (I > pts) if kind == "above" else (I < pts)
    X, Y = view_coordinates(img, view)
    if kind == "rectangle":
        x0, x1, y0, y1 = pts
        return (X >= min(x0, x1)) & (X <= max(x0, x1)) & (Y >= min(y0, y1)) & (Y <= max(y0, y1))
    if kind == "ellipse":
        x0, x1, y0, y1 = pts
        cx, cy, rx, ry = (x0 + x1) / 2, (y0 + y1) / 2, abs(x1 - x0) / 2, abs(y1 - y0) / 2
        if rx == 0 or ry == 0:
            return np.zeros(img.I.shape, bool)
        return ((X - cx) / rx) ** 2 + ((Y - cy) / ry) ** 2 <= 1.0
    if kind == "polygon":
        from matplotlib.path import Path
        p = Path(np.asarray(pts, float))
        return p.contains_points(np.column_stack([X.ravel(), Y.ravel()])).reshape(img.I.shape)
    raise ValueError(f"unknown shape {kind!r}")


def beam_centre_of_gravity(I: np.ndarray, radius: float = 15.0, iterations: int = 20):
    """Centre of gravity of the brightest region (direct beam): iterated in a disc of ``radius`` pixels."""
    a = np.nan_to_num(np.asarray(I, float))
    a = np.clip(a - np.median(a), 0, None)
    iy, ix = np.mgrid[0:a.shape[0], 0:a.shape[1]]
    y0, x0 = np.unravel_index(np.argmax(a), a.shape)
    x, y = float(x0), float(y0)
    for _ in range(iterations):
        m = (ix - x) ** 2 + (iy - y) ** 2 <= radius ** 2
        w = a * m
        if w.sum() <= 0:
            break
        xn, yn = float((w * ix).sum() / w.sum()), float((w * iy).sum() / w.sum())
        if abs(xn - x) < 1e-3 and abs(yn - y) < 1e-3:
            x, y = xn, yn
            break
        x, y = xn, yn
    return x, y
