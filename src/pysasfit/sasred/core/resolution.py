# SPDX-License-Identifier: BSD-3-Clause
"""q resolution as one Gaussian width per point (doc: "Scope: the resolution
is given as one Gaussian sigma_q").

Phase 1 includes the wavelength term and the spread of pixel q inside each
bin (which contains the pixel size). The collimation term (apertures and L1,
Mildner-Carpenter, or the measured direct-beam width) belongs to the
resolution step of phase 3 and is reported as missing until then.
"""
from __future__ import annotations

import numpy as np

__all__ = ["sigma_lambda_rel", "sigma_q_phase1"]


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
    s_l = sigma_lambda_rel(dlam_over_lam_fwhm, shape)
    return np.sqrt((q_mean * s_l) ** 2 + q_var_in_bin)
