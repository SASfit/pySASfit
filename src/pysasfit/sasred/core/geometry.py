# SPDX-License-Identifier: BSD-3-Clause
"""Pixel geometry (correction step 7) and solid angle (step 11).

Coordinates: pixel indices (iy, ix) are 0-based array indices of the stored
image, axis 1 = x (horizontal), axis 0 = y (vertical), counted from 0 as in
Python and GRASP (pixel centres at integers, pixel edges at half-integers).

The detector rotates about the vertical axis by alpha through the beam spot
(the pixel position xc, yc where the undeflected beam hits it); L2 is the
distance from the sample to that spot along the beam. In the lab frame
(z along the beam)::

    u = (ix - xc) px,  v = (iy - yc) py                  (in-plane offsets)
    r = (u cos(alpha), v, L2 + u sin(alpha))
    n = (-sin(alpha), 0, cos(alpha))                     (detector normal)

so that cos(psi) = n.r / ``|r|`` = L2 cos(alpha) / ``|r|`` and the exact solid angle
of a pixel of area A, to first order in A, is::

    dOmega = A cos(psi) / |r|^2 = A L2 cos(alpha) / |r|^3 ,

which reduces to A cos^3(2 theta) / L2^2 for alpha = 0.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = ["PixelGeometry", "pixel_geometry"]


@dataclass
class PixelGeometry:
    x: np.ndarray        # lab-frame position [m]
    y: np.ndarray
    z: np.ndarray
    two_theta: np.ndarray    # rad
    phi: np.ndarray          # azimuth, atan2(y, x), rad
    psi: np.ndarray          # incidence angle on the detector, rad
    q: np.ndarray            # 1/Angstrom
    qx: np.ndarray
    qy: np.ndarray
    qz: np.ndarray
    d_omega: np.ndarray      # sr
    wavelength_A: float
    L2_m: float
    alpha_rad: float


def pixel_geometry(shape, pixel_size_m, beam_center_px, L2_m, wavelength_A,
                   alpha_deg: float = 0.0) -> PixelGeometry:
    ny, nx = shape
    px, py = (pixel_size_m, pixel_size_m) if np.isscalar(pixel_size_m) else pixel_size_m
    xc, yc = beam_center_px
    a = np.deg2rad(alpha_deg)
    iy, ix = np.mgrid[0:ny, 0:nx].astype(float)
    u = (ix - xc) * px
    v = (iy - yc) * py
    x = u * np.cos(a)
    y = v
    z = L2_m + u * np.sin(a)
    d = np.sqrt(x * x + y * y + z * z)
    two_theta = np.arctan2(np.hypot(x, y), z)      # accurate at small angles
    cos_psi = L2_m * np.cos(a) / d
    k = 2.0 * np.pi / wavelength_A
    qx, qy, qz = k * x / d, k * y / d, k * (z / d - 1.0)
    q = 2.0 * k * np.sin(0.5 * two_theta)
    d_omega = px * py * cos_psi / d ** 2
    return PixelGeometry(x, y, z, two_theta, np.arctan2(y, x), np.arccos(np.clip(cos_psi, -1, 1)),
                         q, qx, qy, qz, d_omega, float(wavelength_A), float(L2_m), float(a))
