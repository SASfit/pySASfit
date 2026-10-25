# -*- coding: utf-8 -*-
"""Colour scales for 2D detector images.

    norm = make_norm("asinh", image)
    ax.imshow(image, norm=norm)

The scales::

    linear   I
    log1p    log(1 + I)       counts: zero stays finite, small counts are spread out
    symlog   linear for |I| < 1, logarithmic above (also negative values, e.g. after subtraction)
    asinh    asinh(I)         like symlog but smooth at the transition

The threshold of symlog / asinh (``linear_width``) is 1 in the units of the image,
i.e. one count for raw data.
"""
from __future__ import annotations

import numpy as np
from matplotlib import colors

__all__ = ["SCALES", "make_norm", "decorate_colorbar"]

#: key -> label of the colour scales offered on the panels
SCALES = {"linear": "linear", "log1p": "log(1+I)", "symlog": "symlog", "asinh": "asinh"}


def make_norm(scale, data, linear_width=1.0, valid=None):
    """``matplotlib.colors.Normalize`` for ``scale`` covering the finite values of ``data``; with
    ``valid`` (bool array) only those pixels set the range, e.g. the unmasked pixels of a detector
    image, so that a few bright pixels at a badly placed beamstop do not dominate the colours."""
    a = np.asarray(data, float)
    if valid is not None:
        v = np.asarray(valid, bool)
        if v.shape == a.shape and v.any():
            a = a[v]
    fin = a[np.isfinite(a)]
    vmin, vmax = (float(fin.min()), float(fin.max())) if fin.size else (0.0, 1.0)
    if vmax <= vmin:
        vmax = vmin + 1.0
    if scale == "linear":
        return colors.Normalize(vmin=vmin, vmax=vmax)
    if scale == "log1p":
        lo = max(vmin, 0.0)                               # log1p is defined for I > -1; images are >= 0
        return colors.FuncNorm((np.log1p, np.expm1), vmin=lo, vmax=max(vmax, lo + 1.0))
    if scale == "symlog":
        return colors.SymLogNorm(linthresh=linear_width, vmin=vmin, vmax=vmax, base=10)
    if scale == "asinh":
        return colors.AsinhNorm(linear_width=linear_width, vmin=vmin, vmax=vmax)
    raise ValueError(f"unknown colour scale '{scale}' (one of {list(SCALES)})")


def _decade_label(v, _pos=None):
    if v == 0:
        return "0"
    e = int(round(np.log10(abs(v))))
    sign = "-" if v < 0 else ""
    return f"${sign}10^{{{e}}}$" if e not in (0, 1) else f"{sign}{10 ** e}"


def decorate_colorbar(cb, scale):
    """Readable ticks for the non-linear scales: 0 and the decades (also negative ones for
    symlog / asinh) inside the colour range; matplotlib's own ticks crowd for log1p and asinh."""
    from matplotlib.ticker import FixedLocator, FuncFormatter
    if scale == "linear":
        return cb
    n = cb.norm
    lo, hi = float(n.vmin), float(n.vmax)
    top = max(abs(lo), abs(hi), 1.0)
    dec = [10.0 ** k for k in range(0, int(np.floor(np.log10(top))) + 1)]
    ticks = [0.0] + dec + [-d for d in dec]
    ticks = sorted(t for t in ticks if lo - 1e-12 <= t <= hi + 1e-12)
    if len(ticks) >= 2:
        cb.ax.yaxis.set_major_locator(FixedLocator(ticks))
        cb.ax.yaxis.set_major_formatter(FuncFormatter(_decade_label))
        cb.ax.yaxis.set_minor_locator(FixedLocator([]))
    return cb
