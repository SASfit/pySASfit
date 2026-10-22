"""
Named, pluggable Fredholm kernels for interactive selection (e.g. a GUI
dropdown), on top of the generic build_size_distribution_kernel machinery
in kernels.py.

Each kernel function has signature f(q, r) -> array over q, for a single
scalar r -- exactly what build_size_distribution_kernel's
`form_factor_squared` argument expects, so these plug in directly:

    A = build_size_distribution_kernel(q, r_grid, KERNEL_REGISTRY["sphere_rg"].func, alpha=0.0)

KERNEL_REGISTRY is deliberately a plain dict of small dataclasses (not a
plugin/entry-point system) so adding a new kernel later is just adding one
function + one registry entry -- matching Joachim's "or something else
defined later" requirement without over-engineering a plugin framework
before there's a second real use case for one.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np


@dataclass
class KernelSpec:
    label: str                 # display name, e.g. for a GUI dropdown
    func: Callable[[np.ndarray, float], np.ndarray]
    description: str
    default_alpha: float = 0.0  # default size-weighting exponent (see kernels.py eq. 32-34)
    # False for kernels whose unknown is not a size distribution (e.g. the
    # PDDF p(r) with the j0 kernel): no size weighting, alpha is fixed at 0
    # and the GUI hides the alpha field.
    uses_alpha: bool = True
    # y-axis label for the recovered function when uses_alpha is False
    result_label: str = "p(r)"
    # Symmetry of the PDDF-type kernels (Hansen 2016, eqs. 1-3):
    # "sphere" (3D), "cylinder" (cross-section of a long rod, per unit
    # length) or "lamella" (thickness of a flat sheet, per unit area).
    # None for size-distribution kernels.
    geometry: str | None = None
    # Forward value of the reduced intensity q^k I(q) for q -> 0, as a
    # multiple of int p(r) dr: k = 0, 1, 2 for sphere, cylinder, lamella.
    forward_factor: float = 4.0 * np.pi
    forward_label: str = "I(0) = 4π∫p dr"
    # name of the radius of gyration obtained from int p r^2 / (2 int p)
    rg_label: str = "R<sub>g</sub>"
    # True when p(0) != 0 (lamella: p(r) = gamma(r), gamma(0) = int rho^2 dx);
    # the GUI then preselects a smoothness penalty that does not pin p(0) = 0.
    p0_free: bool = False


def sinc_kernel(q: np.ndarray, r: float) -> np.ndarray:
    """
    K(q, r) = 4 pi r^2 j0(qr), where j0(x) = sin(x)/x is the zeroth-order
    spherical Bessel function of the first kind.

    This is the kernel relating a radial (pair-distance-style) distribution
    to I(q) via I(q) = integral 4 pi r^2 p(r) j0(qr) dr -- the
    spherically-averaged Fourier-sine transform kernel used in the
    indirect Fourier transform / Debye formula. j0 shows up here because
    it is precisely the radial part of a 3D Fourier transform of a
    spherically symmetric function (Joachim, 2026-09-25: correctly pointed
    out this is j0, not just "a sinc-like function" -- using
    scipy.special.spherical_jn(0, ...) directly now instead of the
    np.sinc(x/pi) reindexing trick, for both clarity and because the
    dedicated special function is the numerically appropriate way to
    evaluate it, including at qr=0 where j0(0)=1.
    """
    from scipy.special import spherical_jn

    q = np.asarray(q, dtype=float)
    qr = q * r
    return 4.0 * np.pi * spherical_jn(0, qr)


def sphere_rayleigh_gans_intensity(q: np.ndarray, r: float) -> np.ndarray:
    """
    Scattered intensity of a homogeneous sphere of radius r, Rayleigh-Gans
    (= exact form factor for a sphere in the Born/RG approximation, which
    for a sphere coincides with the standard exact form factor)::

        |F(qr)|^2 = r^6 * [3 (sin(qr) - qr cos(qr)) / (qr)^3]^2

    BUG FIX (2026-09-25, caught by comparing a blind inversion of the real
    JAC-2022-paper benchmark dataset against the paper's own published
    Figure 5): earlier versions of this function returned ONLY the
    normalized shape P(qr) = [3(sin(qr)-qr cos(qr))/(qr)^3]^2, P(0)=1,
    omitting the r^6 (Volume^2) prefactor. That is NOT just a missing
    "overall amplitude" -- omitting it changes the *relative* weighting
    between different particle sizes in a polydisperse sum, because real
    particles scatter with intensity proportional to Volume^2, not with
    uniform per-particle efficiency regardless of size. Without r^6, a
    blind EM inversion on that benchmark data (known bilognormal spheres
    at 60nm and 180nm modes) recovered a spurious single broad peak
    dominated by ~300nm content and negligible signal near 60nm --
    because the kernel made large particles look like weak scatterers, so
    the solver had to pile up unrealistic amounts of N(r) at large r to
    explain the data. With r^6 restored, a direct two-population linear
    fit against the real data recovers a volume ratio of ~1.9:1 in favor
    of the 60nm population, qualitatively matching the paper's Fig. 5
    (dominant sharp peak near 60-85nm, smaller broad hump near 180-220nm)
    for the first time.

    An overall physical prefactor ((4pi/3)^2 * (contrast)^2) is still
    omitted -- that IS a genuine free amplitude/scale (concentration,
    contrast), safe to fold into an overall fit scale factor if an
    absolute (not just relative) intensity match is needed. The r^6 term
    is not optional in the same way.
    """
    q = np.asarray(q, dtype=float)
    qr = q * r
    qr_safe = np.where(qr == 0, np.finfo(float).eps, qr)
    F = 3.0 * (np.sin(qr_safe) - qr_safe * np.cos(qr_safe)) / qr_safe**3
    F = np.where(qr == 0, 1.0, F)
    return (r**6) * F**2


def cylinder_pddf_kernel(q: np.ndarray, r: float) -> np.ndarray:
    """
    Kernel of the cross-section PDDF of a long cylinder (rod), Hansen
    (2016) eq. 2 per unit length:

        I(q) = (2 pi^2 / q) * integral p_c(r) J0(qr) dr,

    with p_c(r) = r * gamma_c(r) the pair-distance distribution of the
    cross-section (2D), J0 the cylindrical Bessel function of order 0.
    The 1/q factor is the 1/(qL) of the orientational average of a
    long rod (valid for q >> 2 pi/L).  Same normalization as
    pysasfit.sldprofile.twostep.ift_kernel(geometry="cylinder").
    """
    from scipy.special import j0

    q = np.asarray(q, dtype=float)
    q_safe = np.where(q == 0, np.finfo(float).tiny, q)
    return 2.0 * np.pi**2 * j0(q * r) / q_safe


def lamella_pddf_kernel(q: np.ndarray, r: float) -> np.ndarray:
    """
    Kernel of the thickness PDDF of a flat lamella (sheet), Hansen (2016)
    eq. 3 per unit area:

        I(q) = (4 pi / q^2) * integral p_t(r) cos(qr) dr,

    with p_t(r) = gamma_t(r) the 1D autocorrelation of the density profile
    across the sheet (p_t(0) = int rho^2 dx > 0, i.e. NOT zero at r = 0).
    The 1/q^2 factor is the Lorentz factor of the orientational average of a
    large flat sheet (valid for q >> 2 pi/sqrt(A)).  Same normalization as
    pysasfit.sldprofile.twostep.ift_kernel(geometry="lamella").
    """
    q = np.asarray(q, dtype=float)
    q_safe = np.where(q == 0, np.finfo(float).tiny, q)
    return 4.0 * np.pi * np.cos(q * r) / q_safe**2


KERNEL_REGISTRY: dict[str, KernelSpec] = {
    "sinc_4pi": KernelSpec(
        label="Sphere: 4\u03c0 j\u2080(qr)  [j\u2080(x)=sin(x)/x, spherical Bessel]",
        func=sinc_kernel,
        description="Fourier-sine (Debye/IFT-style) kernel relating a radial "
                     "distribution p(r) to I(q), via the zeroth-order spherical "
                     "Bessel function j0(qr).",
        default_alpha=0.0,
        uses_alpha=False,
        result_label="p(r)",
        geometry="sphere",
    ),
    "cylinder_pddf": KernelSpec(
        label="Cylinder cross-section: 2\u03c0\u00b2 J\u2080(qr)/q  [Hansen 2016]",
        func=cylinder_pddf_kernel,
        description="Cross-section pair-distance distribution p_c(r) of a long "
                     "rod (per unit length): I(q) = 2 pi^2/q int p_c(r) J0(qr) dr.",
        default_alpha=0.0,
        uses_alpha=False,
        result_label="$p_c(r)$",
        geometry="cylinder",
        forward_factor=2.0 * np.pi**2,
        forward_label="lim q·I(q) = 2π²∫p<sub>c</sub> dr",
        rg_label="R<sub>c</sub> (cross-section)",
    ),
    "lamella_pddf": KernelSpec(
        label="Lamella thickness: 4\u03c0 cos(qr)/q\u00b2  [Hansen 2016]",
        func=lamella_pddf_kernel,
        description="Thickness pair-distance distribution p_t(r) = gamma_t(r) of a "
                     "flat sheet (per unit area): I(q) = 4 pi/q^2 int p_t(r) cos(qr) dr; "
                     "p_t(0) > 0.",
        default_alpha=0.0,
        uses_alpha=False,
        result_label="$p_t(r)$",
        geometry="lamella",
        forward_factor=4.0 * np.pi,
        forward_label="lim q²·I(q) = 4π∫p<sub>t</sub> dr",
        rg_label="R<sub>t</sub> (thickness)",
        p0_free=True,
    ),
    "sphere_rg": KernelSpec(
        label="Sphere (Rayleigh-Gans)",
        func=sphere_rayleigh_gans_intensity,
        description="Homogeneous sphere form factor (with the physical r^6 "
                     "Volume^2 scaling). default_alpha=6 is not just a "
                     "convenience default here -- alpha=6 is what divides "
                     "the r^6 back out of the kernel for numerical "
                     "conditioning (see the function's own docstring); "
                     "using alpha=0 with this kernel produces a ~1e15 "
                     "dynamic range that breaks the EM solver.",
        default_alpha=6.0,
    ),
}


def list_kernels() -> list[tuple[str, str]]:
    """(key, label) pairs, e.g. for populating a GUI dropdown."""
    return [(key, spec.label) for key, spec in KERNEL_REGISTRY.items()]
