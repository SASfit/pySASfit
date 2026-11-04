# SPDX-License-Identifier: BSD-3-Clause
"""Detector image in polar coordinates (azimuth φ, scattering vector q) for display.

φ = 0 is the horizontal direction (+x, axis 1 of the image), increasing counter-clockwise
(towards +y, axis 0), as the azimuth ``phi`` of :func:`pysasred.core.geometry.pixel_geometry`.
Every cell of the (q, φ) grid takes the value of the detector pixel that contains its centre
(nearest-pixel resampling, no interpolation: counts are not mixed); cells outside the
detector are NaN. Pixel sizes may differ in x and y.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

__all__ = ["PolarImage", "polar_image"]


@dataclass
class PolarImage:
    image: np.ndarray        # (n_r, n_phi)
    phi_edges_deg: np.ndarray
    r_edges_m: np.ndarray
    q_edges: np.ndarray      # 1/Å, from r through 2θ = atan(r / SD)

    def q_of_r(self, r_m):
        return np.interp(r_m, self.r_edges_m, self.q_edges)


def polar_image(values, beam_center_px, pixel_size_m, SD_m, wavelength_A, n_phi: int = 360,
                r_step_m: Optional[float] = None, invalid: Optional[np.ndarray] = None) -> PolarImage:
    """Resample ``values`` (ny, nx) onto a polar grid around the beam centre.

    ``invalid``: pixels shown as NaN (e.g. the beamstop mask)."""
    v = np.asarray(values, float)
    ny, nx = v.shape
    px, py = (pixel_size_m, pixel_size_m) if np.isscalar(pixel_size_m) else pixel_size_m
    xc, yc = beam_center_px
    corners = np.array([[-0.5, -0.5], [nx - 0.5, -0.5], [-0.5, ny - 0.5], [nx - 0.5, ny - 0.5]])
    rmax = float(np.max(np.hypot((corners[:, 0] - xc) * px, (corners[:, 1] - yc) * py)))
    step = r_step_m or 0.5 * min(px, py)
    n_r = max(2, int(np.ceil(rmax / step)))
    r_edges = np.linspace(0.0, n_r * step, n_r + 1)
    phi_edges = np.linspace(0.0, 360.0, n_phi + 1)
    rc = 0.5 * (r_edges[1:] + r_edges[:-1])
    pc = np.deg2rad(0.5 * (phi_edges[1:] + phi_edges[:-1]))
    R, P = np.meshgrid(rc, pc, indexing="ij")
    ix = np.rint(xc + R * np.cos(P) / px).astype(int)
    iy = np.rint(yc + R * np.sin(P) / py).astype(int)
    inside = (ix >= 0) & (ix < nx) & (iy >= 0) & (iy < ny)
    out = np.full(R.shape, np.nan)
    src = v.copy()
    if invalid is not None:
        src[np.asarray(invalid, bool)] = np.nan
    out[inside] = src[iy[inside], ix[inside]]
    q_edges = 4 * np.pi / wavelength_A * np.sin(0.5 * np.arctan(r_edges / SD_m))
    return PolarImage(out, phi_edges, r_edges, q_edges)
