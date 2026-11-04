# SPDX-License-Identifier: BSD-3-Clause
"""Physical model of the SANS-I ³He detector: transmission of the entrance layers and absorption
probability in the detection gas, as a function of wavelength λ and of the angle ψ between the
neutron path and the detector normal.

Layers (instrument file ``detector.construction``): Al dome window (7 mm, Ø 1300 mm, radius of
curvature 930 mm), ⁴He at 2.7 bar, a flat 2 mm Al membrane, then ³He at 2.7 bar with CF₄ as stopping
gas. Values still unknown
(active depth, membrane material, ⁴He path, CF₄ pressure) are parameters.

Cross sections (thermal, 1.798 Å; absorption ∝ λ):

* ³He: σ_a = 5333 b;
* Al: σ_a = 0.231 b, σ_inc = 0.0082 b; coherent (Bragg) scattering only below the cut-off
  λ = 2 d_111 = 4.68 Å; above it a small inelastic (phonon) part, taken as ``al_inelastic_b``;
* ⁴He: σ_s = 1.34 b, no absorption.

The mean conversion depth below the start of the active volume, ⟨z⟩, is where the neutron is
detected on average: it is part of the effective sample-detector distance and shifts with λ.
"""
from __future__ import annotations

import numpy as np

__all__ = ["dome_sag", "dome_crossing_angle", "dome_transmission", "window_transmission", "number_density", "sigma_3he", "absorption_probability", "mean_conversion_depth",
           "al_transmission", "efficiency", "BARN"]

BARN = 1e-28          # m²
K_B = 1.380649e-23
LAMBDA_TH = 1.798     # Å
N_AL = 6.026e28       # Al atoms / m³


def number_density(pressure_bar: float, temperature_K: float = 293.0) -> float:
    """Atoms per m³ of an ideal gas."""
    return pressure_bar * 1e5 / (K_B * temperature_K)


def sigma_3he(wavelength_A) -> np.ndarray:
    """Macroscopic absorption cross section of ³He per bar at 293 K [1/m]: n σ_a(λ)."""
    return number_density(1.0) * 5333.0 * BARN * np.asarray(wavelength_A, float) / LAMBDA_TH


def absorption_probability(wavelength_A, depth_m, pressure_bar=2.7, psi_rad=0.0, temperature_K=293.0):
    """1 - exp(-Σ D / cos ψ): probability that the neutron is absorbed in the active depth D."""
    S = sigma_3he(wavelength_A) * pressure_bar * 293.0 / temperature_K
    return 1.0 - np.exp(-S * depth_m / np.cos(psi_rad))


def mean_conversion_depth(wavelength_A, depth_m=np.inf, pressure_bar=2.7, psi_rad=0.0, temperature_K=293.0):
    """Mean depth (along the detector normal) of the absorbed neutrons below the start of the
    active volume [m]: 1/Σ' - D e^{-Σ'D}/(1 - e^{-Σ'D}), Σ' = Σ / cos ψ (1/Σ' for D → ∞)."""
    S = sigma_3he(wavelength_A) * pressure_bar * 293.0 / temperature_K / np.cos(psi_rad)
    if np.isinf(depth_m):
        return 1.0 / S
    e = np.exp(-S * depth_m)
    return 1.0 / S - depth_m * e / (1.0 - e)


def al_transmission(wavelength_A, thickness_m, psi_rad=0.0, al_inelastic_b: float = 0.1):
    """Transmission of an Al layer for λ above the Bragg cut-off (4.68 Å): absorption, incoherent
    and an approximate inelastic part; the path is thickness / cos ψ."""
    lam = np.asarray(wavelength_A, float)
    if np.any(lam < 4.68):
        raise ValueError("below the Al Bragg cut-off (4.68 Å) the coherent scattering is not modelled")
    sig = (0.231 * lam / LAMBDA_TH + 0.0082 + al_inelastic_b) * BARN
    return np.exp(-N_AL * sig * thickness_m / np.cos(psi_rad))


def efficiency(wavelength_A, psi_rad=0.0, depth_m=0.03, window_m=0.007, membrane_m=0.002,
               membrane_al: bool = True, pressure_bar=2.7, he4_bar=2.7, he4_path_m=0.0, al_inelastic_b=0.1):
    """Detection probability: transmission of window, ⁴He and membrane times absorption in ³He.
    Relative to ψ = 0 this is the angular efficiency that water normalisation removes."""
    T = al_transmission(wavelength_A, window_m, psi_rad, al_inelastic_b)
    if membrane_al:
        T = T * al_transmission(wavelength_A, membrane_m, psi_rad, al_inelastic_b)
    T = T * np.exp(-number_density(he4_bar) * 1.34 * BARN * he4_path_m / np.cos(psi_rad))
    return T * absorption_probability(wavelength_A, depth_m, pressure_bar, psi_rad)


def dome_sag(radius_m: float = 0.93, diameter_m: float = 1.30) -> float:
    """Depth of the spherical dome cap: R - sqrt(R² - (D/2)²) (0.265 m for R 0.93 m, Ø 1.30 m)."""
    return radius_m - np.sqrt(radius_m**2 - (diameter_m / 2) ** 2)


def dome_crossing_angle(x_m, y_m, SD_m: float, radius_m: float = 0.93, apex_to_detector_m=None,
                        diameter_m: float = 1.30):
    """Angle γ [rad] between the neutron path (sample at the origin -> pixel (x, y) in the detection
    plane at SD) and the normal of the spherical dome window where the path crosses it.

    The dome is convex towards the sample, its centre on the beam axis; its apex lies
    ``apex_to_detector_m`` in front of the detection plane (default: the sag, i.e. the rim of the
    dome in the detection plane)."""
    a = dome_sag(radius_m, diameter_m) if apex_to_detector_m is None else apex_to_detector_m
    x, y = np.asarray(x_m, float), np.asarray(y_m, float)
    d = np.sqrt(x * x + y * y + SD_m**2)
    ux, uy, uz = x / d, y / d, SD_m / d
    zc = SD_m - a + radius_m                      # centre of the sphere on the axis
    b = uz * zc
    disc = b * b - zc * zc + radius_m**2
    t = b - np.sqrt(np.maximum(disc, 0.0))         # first crossing (the cap facing the sample)
    px, py, pz = t * ux, t * uy, t * uz - zc
    cosg = np.abs(ux * px + uy * py + uz * pz) / radius_m
    return np.arccos(np.clip(cosg, -1.0, 1.0))


def dome_transmission(wavelength_A, x_m, y_m, SD_m, thickness_m: float = 0.007, radius_m: float = 0.93,
                      apex_to_detector_m=None, al_inelastic_b: float = 0.1, relative: bool = True):
    """Transmission of the Al dome per pixel (path thickness / cos γ); with ``relative`` divided by
    its value on the axis, i.e. the part that varies over the detector."""
    g = dome_crossing_angle(x_m, y_m, SD_m, radius_m, apex_to_detector_m)
    T = al_transmission(wavelength_A, thickness_m, g, al_inelastic_b)
    return T / al_transmission(wavelength_A, thickness_m, 0.0, al_inelastic_b) if relative else T


def window_transmission(wavelength_A, x_m, y_m, SD_m, dome_m: float = 0.007, membrane_m: float = 0.002,
                        radius_m: float = 0.93, apex_to_detector_m=None, al_inelastic_b: float = 0.1,
                        relative: bool = True):
    """Dome (curved, crossing angle γ) and flat Al membrane (incidence angle ψ on the detector plane)
    per pixel; with ``relative`` divided by the value on the axis."""
    x, y = np.asarray(x_m, float), np.asarray(y_m, float)
    psi = np.arctan(np.hypot(x, y) / SD_m)
    T = dome_transmission(wavelength_A, x, y, SD_m, dome_m, radius_m, apex_to_detector_m, al_inelastic_b, relative)
    M = al_transmission(wavelength_A, membrane_m, psi, al_inelastic_b)
    return T * (M / al_transmission(wavelength_A, membrane_m, 0.0, al_inelastic_b) if relative else M)
