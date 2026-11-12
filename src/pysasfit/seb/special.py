"""Numerically safe special functions used by the sub-unit library.

All functions accept numpy arrays and are accurate at x -> 0, where the
closed forms suffer from cancellation; a short Taylor series is used below
a cut-off.
"""
import numpy as np
from scipy import special as sp


def _blend(x, exact, series, cut):
    x = np.asarray(x, dtype=float)
    small = np.abs(x) < cut
    out = np.empty_like(x)
    if np.any(small):
        out[small] = series(x[small])
    if np.any(~small):
        out[~small] = exact(x[~small])
    return out


def sinc(x):
    """sin(x)/x (unnormalised)."""
    return np.sinc(np.asarray(x, dtype=float) / np.pi)


def debye(x):
    """Debye function 2(exp(-x) - 1 + x)/x^2, x = (q Rg)^2."""
    return _blend(x, lambda x: 2.0 * (np.expm1(-x) + x) / x**2,
                  lambda x: 1 - x / 3 + x**2 / 12 - x**3 / 60 + x**4 / 360, 1e-3)


def one_minus_exp_over_x(x):
    """(1 - exp(-x))/x."""
    return _blend(x, lambda x: -np.expm1(-x) / x,
                  lambda x: 1 - x / 2 + x**2 / 6 - x**3 / 24 + x**4 / 120, 1e-4)


def sphere_amp(x):
    """3 (sin x - x cos x)/x^3, x = q R."""
    return _blend(x, lambda x: 3.0 * (np.sin(x) - x * np.cos(x)) / x**3,
                  lambda x: 1 - x**2 / 10 + x**4 / 280 - x**6 / 15120, 1e-2)


def si_over_x(x):
    """Si(x)/x."""
    return _blend(x, lambda x: sp.sici(x)[0] / x,
                  lambda x: 1 - x**2 / 18 + x**4 / 600, 1e-3)


def jinc(x):
    """2 J1(x)/x."""
    return _blend(x, lambda x: 2.0 * sp.j1(x) / x,
                  lambda x: 1 - x**2 / 8 + x**4 / 192, 1e-3)


def ring_debye(x):
    """Gaussian ring (loop) form factor, x = (q Rg)^2 with Rg^2 = N b^2/12.

    P = int_0^1 exp(-2 x t(1-t)) dt = sqrt(2/x) D(sqrt(x/2)), D = Dawson.
    """
    return _blend(x, lambda x: np.sqrt(2.0 / x) * sp.dawsn(np.sqrt(x / 2.0)),
                  lambda x: 1 - x / 3 + x**2 / 15 - x**3 / 105, 1e-3)


def thin_rod_ff(x):
    """Thin rod form factor 2 Si(x)/x - (sin(x/2)/(x/2))^2, x = q L."""
    return _blend(x, lambda x: 2.0 * sp.sici(x)[0] / x - sinc(x / 2) ** 2,
                  lambda x: 1 - x**2 / 36 + x**4 / 1800, 1e-2)


def int_j0_over_z(z):
    """(1/z) int_0^z J0(t) dt  (orientational average for points on a circle)."""
    return _blend(z, lambda z: sp.itj0y0(z)[0] / z,
                  lambda z: 1 - z**2 / 12 + z**4 / 320, 1e-3)


def thin_disk_ff(x):
    """Infinitely thin disk, 2/x^2 (1 - J1(2x)/x), x = q R."""
    return _blend(x, lambda x: 2.0 / x**2 * (1.0 - sp.j1(2 * x) / x),
                  lambda x: 1 - x**2 / 6 + x**4 / 72, 2e-2)


def disk_center_amp(x):
    """<sin(q r)/(q r)> over a uniform disk relative to its centre: 2(1-cos x)/x^2."""
    return _blend(x, lambda x: 2.0 * (1.0 - np.cos(x)) / x**2,
                  lambda x: 1 - x**2 / 12 + x**4 / 360, 1e-2)
