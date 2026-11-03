"""Two-point statistics of the clipped (level-cut) Gaussian random field.

A Gaussian random field psi(r) with unit variance and correlation function
g(r) = <psi(0) psi(r)>, g(0) = 1, is cut at the level alpha: phase 2 ("pore") is
psi > alpha, so its volume fraction is phi = P(psi > alpha) = erfc(alpha/sqrt 2)/2.
The indicator correlation of the cut field (Berk 1987/1991; Quintanilla 2007) is

    S2(r) - phi^2 = F(g(r)),
    F(g) = 1/(2 pi) integral_0^g exp(-alpha^2/(1+t)) / sqrt(1-t^2) dt
         = 1/(2 pi) integral_0^{arcsin g} exp(-alpha^2/(1+sin theta)) d theta,

with F(1) = phi(1-phi). For a two-phase sample the scattering gives
S2(r) - phi^2 = phi(1-phi) gamma(r) (Debye), hence

    g(r) = F^-1( phi(1-phi) gamma(r) ).

F is tabulated in theta (smooth integrand, no endpoint singularity) and
inverted by monotone interpolation. Values of gamma outside the range a
level-cut field can reach are clipped and counted (:attr:`LevelCut.clipped`).

The spectral function is the 3D Fourier transform of g,

    f(k) = 4 pi integral r^2 g(r) sin(kr)/(kr) dr,

and the distribution of wave-vector magnitudes of the field is
P(k) = k^2 f(k) / (2 pi^2) (integral P dk = g(0) = 1).
"""

from __future__ import annotations

import numpy as np
from scipy.integrate import cumulative_simpson
from scipy.special import erfcinv

from .transforms import sine_transform_pl

__all__ = ["LevelCut", "spectral_function", "wavevector_distribution"]


class LevelCut:
    """Relation between g(r) of the Gaussian field and the two-point function of the cut field."""

    def __init__(self, porosity, n=20001):
        phi = float(porosity)
        if not 0.0 < phi < 1.0:
            raise ValueError("porosity (volume fraction of phase 2) must lie strictly between 0 and 1")
        self.phi = phi
        self.alpha = float(np.sqrt(2.0) * erfcinv(2.0 * phi))
        th = np.linspace(-0.5 * np.pi, 0.5 * np.pi, n)
        with np.errstate(divide="ignore", over="ignore", under="ignore", invalid="ignore"):
            f = np.exp(-self.alpha ** 2 / (1.0 + np.sin(th)))
        f[0] = 0.0 if self.alpha != 0 else 1.0
        F = cumulative_simpson(f, x=th, initial=0.0) / (2 * np.pi)
        F -= np.interp(0.0, th, F)             # F(g = 0) = 0
        self._th, self._F = th, F
        self.clipped = 0

    @property
    def variance(self):
        """phi (1 - phi): F at g = 1."""
        return self.phi * (1 - self.phi)

    @property
    def F_range(self):
        return float(self._F[0]), float(self._F[-1])

    def F(self, g):
        """S2 - phi^2 of the cut field for field correlation g."""
        th = np.arcsin(np.clip(np.asarray(g, float), -1.0, 1.0))
        return np.interp(th, self._th, self._F)

    def g_from_gamma(self, gamma):
        """Field correlation g(r) from the normalised scattering correlation gamma(r)."""
        target = self.variance * np.asarray(gamma, float)
        lo, hi = self.F_range
        tol = 1e-6 * self.variance            # round-off at gamma(0) = 1 is not clipping
        self.clipped = int(np.sum((target < lo - tol) | (target > hi + tol)))
        th = np.interp(np.clip(target, lo, hi), self._F, self._th)
        return np.sin(th)

    def gamma_from_g(self, g):
        return self.F(g) / self.variance


def spectral_function(r, g, k):
    """f(k) = 4 pi integral r^2 g(r) sin(kr)/(kr) dr (g piecewise linear on r)."""
    r = np.asarray(r, float)
    g = np.asarray(g, float)
    k = np.atleast_1d(np.asarray(k, float))
    out = np.empty_like(k)
    nz = k > 0
    from .transforms import moments_pl
    out[~nz] = 4 * np.pi * moments_pl(r, r * g, 1)
    out[nz] = 4 * np.pi * sine_transform_pl(r, r * g, k[nz]) / k[nz]
    return out


def wavevector_distribution(k, f):
    """P(k) = k^2 f(k) / (2 pi^2), negative values set to zero."""
    return np.clip(np.asarray(k) ** 2 * np.asarray(f) / (2 * np.pi ** 2), 0.0, None)
