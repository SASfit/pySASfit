# SPDX-License-Identifier: BSD-3-Clause
"""Pixel geometry (correction step 7) and solid angle (step 11).

Coordinates: pixel indices (iy, ix) are 0-based array indices of the stored
image, axis 1 = columns (horizontal), axis 0 = rows (vertical), counted from 0 as in
Python and GRASP (pixel centres at integers, pixel edges at half-integers).

Lab frame of pySASfit (docs/conventions.md, right-handed, as NXcanSAS/McStas): z along the
beam (downstream), y vertical up, x horizontal = y × z, i.e. +x points to the LEFT as seen from
the sample looking downstream and to the RIGHT in the views of pySASfit, which show the detector
as seen looking towards the sample. The azimuth phi = atan2(y, x) is counter-clockwise in these
views. The raw arrays of SANS-I and SANS-LLB have the column index growing to the right as seen
from the sample, i.e. towards -x: ``column_sign`` = -1 (default).

The detector rotates about the vertical axis by alpha through the beam spot
(the pixel position xc, yc where the undeflected beam hits it); SD is the
distance from the sample to that spot along the beam. With s = column_sign::

    u = (ix - xc) px,  v = (iy - yc) py                  (in-plane offsets along columns, rows)
    r = (s u cos(alpha), v, SD + u sin(alpha))
    n = (-s sin(alpha), 0, cos(alpha))                   (detector normal)

so that cos(psi) = n.r / ``|r|`` = SD cos(alpha) / ``|r|`` and the exact solid angle
of a pixel of area A, to first order in A, is::

    dOmega = A cos(psi) / |r|^2 = A SD cos(alpha) / |r|^3 ,

which reduces to A cos^3(2 theta) / SD^2 for alpha = 0.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = ["PixelGeometry", "pixel_geometry", "run_geometry", "COLUMN_SIGN"]

#: lab x per +1 column of the raw arrays of the SINQ instruments (columns grow to the right as seen
#: from the sample, +x points to the left): -1
COLUMN_SIGN = -1.0


@dataclass
class PixelGeometry:
    x: np.ndarray        # lab-frame position [m]
    y: np.ndarray
    z: np.ndarray
    two_theta: np.ndarray    # rad
    phi: np.ndarray          # azimuth, atan2(y, x), rad (counter-clockwise in the view towards the sample)
    psi: np.ndarray          # incidence angle on the detector, rad
    q: np.ndarray            # 1/Angstrom
    qx: np.ndarray
    qy: np.ndarray
    qz: np.ndarray
    d_omega: np.ndarray      # sr
    wavelength_A: float
    SD_m: float
    alpha_rad: float


def pixel_geometry(shape, pixel_size_m, beam_center_px, SD_m, wavelength_A,
                   alpha_deg: float = 0.0, column_sign: float = COLUMN_SIGN) -> PixelGeometry:
    ny, nx = shape
    px, py = (pixel_size_m, pixel_size_m) if np.isscalar(pixel_size_m) else pixel_size_m
    xc, yc = beam_center_px
    a = np.deg2rad(alpha_deg)
    iy, ix = np.mgrid[0:ny, 0:nx].astype(float)
    u = (ix - xc) * px
    v = (iy - yc) * py
    x = float(np.sign(column_sign) or 1.0) * u * np.cos(a)
    y = v
    z = SD_m + u * np.sin(a)
    d = np.sqrt(x * x + y * y + z * z)
    two_theta = np.arctan2(np.hypot(x, y), z)      # accurate at small angles
    cos_psi = SD_m * np.cos(a) / d
    k = 2.0 * np.pi / wavelength_A
    qx, qy, qz = k * x / d, k * y / d, k * (z / d - 1.0)
    q = 2.0 * k * np.sin(0.5 * two_theta)
    d_omega = px * py * cos_psi / d ** 2
    return PixelGeometry(x, y, z, two_theta, np.arctan2(y, x), np.arccos(np.clip(cos_psi, -1, 1)),
                         q, qx, qy, qz, d_omega, float(wavelength_A), float(SD_m), float(a))


def run_geometry(raw, beam_center_px, alpha_deg: float = 0.0) -> PixelGeometry:
    """Geometry of a raw run: the area detector from its pixel size, distance and beam centre; a
    wide-angle panel of SANS-LLB (``raw.panel``) from its pixel positions and offsets
    (:func:`pysasfit.sasred.io.sans_llb.panel_geometry`; the beam centre is that of the central detector)."""
    if getattr(raw, "panel", None) is not None:
        from ..io.sans_llb import panel_geometry
        return panel_geometry(raw, beam_center_px)
    return pixel_geometry(raw.counts.shape, raw.pixel_size_m, beam_center_px,
                          float(getattr(raw, "SD_effective_m", raw.SD_m)), raw.wavelength_A, alpha_deg=alpha_deg,
                          column_sign=float(getattr(raw, "column_sign", COLUMN_SIGN)))


VIEW_NOTE = "view towards the sample"


def orient_pixel_axes(ax, raw=None, column_sign: float = None) -> None:
    """Orient a matplotlib image of raw detector pixels (``imshow(..., origin="lower")`` with column/row
    data coordinates) as pySASfit shows every detector plane: as seen looking towards the sample, lab +x
    to the right, +y up (docs/conventions.md). The data coordinates stay pixel indices (mask tools,
    beam centre). A SANS-LLB wide-angle panel (``raw.panel``) is oriented by its posx/posy (= lab x, y)."""
    p = getattr(raw, "panel", None) if raw is not None else None
    if p is not None:
        flip_x = p.posx_mm[0] > p.posx_mm[-1]
        flip_y = p.posy_mm[0] > p.posy_mm[-1]
    else:
        s = column_sign if column_sign is not None else float(getattr(raw, "column_sign", COLUMN_SIGN))
        flip_x, flip_y = s < 0, False
    if flip_x and not ax.xaxis_inverted():
        ax.invert_xaxis()
    if flip_y and not ax.yaxis_inverted():
        ax.invert_yaxis()
