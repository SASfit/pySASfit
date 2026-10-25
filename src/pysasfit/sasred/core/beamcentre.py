# SPDX-License-Identifier: BSD-3-Clause
"""Beam centre from an attenuated direct-beam run: masked centre of gravity.

Procedure (doc: "Beam centre and direct-beam profile"): exclude a rough mask
(for example everything far from the beam), take the maximum pixel, then
iterate the centre of gravity inside a disc of radius `radius_px` around the
current estimate until it moves less than `tol_px` (default 0.1 pixel). No
background is subtracted. The statistical uncertainty of the centre follows
exactly from the counts: var(x_c) = sum (x - x_c)^2 N / (sum N)^2.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

__all__ = ["BeamCentre", "masked_centre_of_gravity"]


@dataclass
class BeamCentre:
    x: float          # pixel, axis 1, 0-based
    y: float          # pixel, axis 0, 0-based
    sigma_x: float
    sigma_y: float
    second_moment_x: float   # pixel^2, for the divergence analysis
    second_moment_y: float
    counts_used: float
    iterations: int
    edge_fraction: float     # predicted fraction of the disc outside the detector


def masked_centre_of_gravity(counts, valid: Optional[np.ndarray] = None, radius_px: float = 8.0,
                             tol_px: float = 0.1, max_iter: int = 50) -> BeamCentre:
    c = np.asarray(counts, float)
    ny, nx = c.shape
    ok = np.ones_like(c, bool) if valid is None else np.asarray(valid, bool)
    w0 = np.where(ok, c, 0.0)
    iy, ix = np.unravel_index(np.argmax(w0), c.shape)
    xc, yc = float(ix), float(iy)
    yy, xx = np.mgrid[0:ny, 0:nx].astype(float)
    for it in range(1, max_iter + 1):
        disc = (xx - xc) ** 2 + (yy - yc) ** 2 <= radius_px ** 2
        w = np.where(disc & ok, c, 0.0)
        s = w.sum()
        if s <= 0:
            raise ValueError("no counts inside the beam-centre disc")
        xn, yn = (w * xx).sum() / s, (w * yy).sum() / s
        moved = np.hypot(xn - xc, yn - yc)
        xc, yc = xn, yn
        if moved < tol_px:
            break
    disc = (xx - xc) ** 2 + (yy - yc) ** 2 <= radius_px ** 2
    w = np.where(disc & ok, c, 0.0)
    s = w.sum()
    mx = (w * (xx - xc) ** 2).sum() / s
    my = (w * (yy - yc) ** 2).sum() / s
    # area of the disc that lies outside the detector (edge safeguard)
    g = np.linspace(-radius_px, radius_px, 201)
    gx, gy = np.meshgrid(g + xc, g + yc)
    inside_disc = (gx - xc) ** 2 + (gy - yc) ** 2 <= radius_px ** 2
    off = inside_disc & ((gx < -0.5) | (gx > nx - 0.5) | (gy < -0.5) | (gy > ny - 0.5))
    return BeamCentre(xc, yc, float(np.sqrt(mx / s)), float(np.sqrt(my / s)), float(mx), float(my),
                      float(s), it, float(off.sum() / inside_disc.sum()))
