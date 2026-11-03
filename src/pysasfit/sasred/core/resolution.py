# SPDX-License-Identifier: BSD-3-Clause
"""q resolution: one Gaussian sigma_q per point (1D) or a 2D Gaussian per pixel.

Three contributions, added in quadrature:

* **wavelength**: sigma_q = q sigma_lambda/lambda, along q (the velocity selector's triangular
  distribution: sigma = FWHM/sqrt 6);
* **geometry**: the direct-beam profile of the geometry (:mod:`pysasfit.sasred.core.beamprofile`),
  angular widths sigma_theta_x, sigma_theta_y: in the direction phi on the detector
  sigma_q^2 = k^2 cos^2(theta) (sigma_theta_x^2 cos^2 phi + sigma_theta_y^2 sin^2 phi), k = 2 pi/lambda,
  dq/d(2theta) = k cos theta;
* **pixels**: in 1D the spread of the pixel q inside each bin (which contains the pixel size); in 2D the
  pixel width, (k cos theta p/L2)^2/12 per detector axis.

Without a direct-beam measurement of the geometry the resolution is not determined (the reduction goes
on without it and says so).
"""
from __future__ import annotations

import numpy as np

__all__ = ["dlam_from_tilt", "sigma_lambda_rel", "sigma_q_phase1", "sigma_q_1d", "resolution_2d"]


def dlam_from_tilt(tilt_deg: float, table: dict):
    """dλ/λ (FWHM) of a velocity selector at the tilt angle ``tilt_deg`` from the calibration ``table``
    ({"tilt_deg": [...], "dlam_over_lam_fwhm": [...]}, the ``wavelength.selector_tilt`` entry of the
    instrument file): monotone cubic (PCHIP) interpolation. A tilt outside the table is clipped to its
    end (within 0.05 deg silently: read-back noise). Returns (dl, note); note is "" or says what was clipped."""
    from scipy.interpolate import PchipInterpolator
    t = np.asarray(table["tilt_deg"], float)
    d = np.asarray(table["dlam_over_lam_fwhm"], float)
    o = np.argsort(t)
    t, d = t[o], d[o]
    x = float(tilt_deg)
    note = ""
    if x < t[0] or x > t[-1]:
        xc = float(np.clip(x, t[0], t[-1]))
        if abs(xc - x) > 0.05:
            note = (f"selector tilt {x:.2f}° outside the calibration table ({t[0]:g}…{t[-1]:g}°): "
                    f"dλ/λ of {xc:g}° used")
        x = xc
    return float(PchipInterpolator(t, d)(x)), note


def sigma_lambda_rel(dlam_over_lam_fwhm: float, shape: str = "triangular") -> float:
    """Relative standard deviation of the wavelength distribution.
    triangular (velocity selector): sigma = FWHM / sqrt(6);
    gaussian: sigma = FWHM / (2 sqrt(2 ln 2))."""
    if shape == "triangular":
        return dlam_over_lam_fwhm / np.sqrt(6.0)
    if shape == "gaussian":
        return dlam_over_lam_fwhm / (2.0 * np.sqrt(2.0 * np.log(2.0)))
    raise ValueError(f"unknown wavelength distribution '{shape}'")


def sigma_q_phase1(q_mean, q_var_in_bin, dlam_over_lam_fwhm, shape="triangular"):
    """Wavelength term and in-bin spread only (no collimation term)."""
    s_l = sigma_lambda_rel(dlam_over_lam_fwhm, shape)
    return np.sqrt((q_mean * s_l) ** 2 + q_var_in_bin)


def _geometric_var(geo, profile):
    """Per pixel: k^2 cos^2(theta) (sx^2 cos^2 phi, sy^2 sin^2 phi) of the beam divergence."""
    tx, ty = profile.divergence_rad
    k = 2 * np.pi / geo.wavelength_A
    c2 = (k * np.cos(0.5 * geo.two_theta)) ** 2
    cp, sp_ = np.cos(geo.phi), np.sin(geo.phi)
    return c2 * tx ** 2, c2 * ty ** 2, cp, sp_


def sigma_q_1d(binning, geo, dlam_over_lam_fwhm, shape, profile):
    """sigma_q per bin: wavelength + in-bin spread + the geometric term averaged over the bin's pixels
    (radial direction of each pixel). Returns (sigma_q, dict of the three variances per bin)."""
    s_l = sigma_lambda_rel(dlam_over_lam_fwhm, shape)
    gx, gy, cp, sp_ = _geometric_var(geo, profile)
    g = (gx * cp ** 2 + gy * sp_ ** 2).ravel()
    v_geo = np.asarray(binning.A @ g).ravel()
    v_lam = (binning.q_mean * s_l) ** 2
    v_bin = binning.q_var_in_bin
    return np.sqrt(v_lam + v_bin + v_geo), {"wavelength": v_lam, "bin": v_bin, "geometry": v_geo}


def resolution_2d(geo, dlam_over_lam_fwhm, shape, profile, pixel_size_m, L2_m):
    """2D Gaussian resolution of every pixel: standard deviations along the detector axes (q_x, q_y) and
    along / across q (parallel = radial, perpendicular = azimuthal), and the q_x-q_y correlation."""
    s_l = sigma_lambda_rel(dlam_over_lam_fwhm, shape)
    gx, gy, cp, sp_ = _geometric_var(geo, profile)
    px, py = (pixel_size_m, pixel_size_m) if np.isscalar(pixel_size_m) else pixel_size_m
    k2c2 = (2 * np.pi / geo.wavelength_A * np.cos(0.5 * geo.two_theta)) ** 2
    gx = gx + k2c2 * (float(px) / L2_m) ** 2 / 12.0
    gy = gy + k2c2 * (float(py) / L2_m) ** 2 / 12.0
    w = (geo.q * s_l) ** 2
    sxx = gx + w * cp ** 2
    syy = gy + w * sp_ ** 2
    sxy = w * cp * sp_
    par = gx * cp ** 2 + gy * sp_ ** 2 + w
    perp = gx * sp_ ** 2 + gy * cp ** 2
    return {"qx_dev": np.sqrt(sxx), "qy_dev": np.sqrt(syy), "rho_xy": sxy / np.sqrt(sxx * syy),
            "q_parallel_dev": np.sqrt(par), "q_perpendicular_dev": np.sqrt(perp)}
