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


# ---------------------------------------------------------------------- chains with a Flory exponent
def _series_E(U, nu, kmax=60):
    """sum_k (-U)^k / (k! (2 nu k + 1))  -- small U."""
    out = np.zeros_like(U)
    term = np.ones_like(U)
    for k in range(kmax):
        out += term / (2 * nu * k + 1)
        term = term * (-U) / (k + 1)
        if np.all(np.abs(term) < 1e-17):
            break
    return out


def _series_F(U, nu, kmax=60):
    """sum_k (-U)^k / k! * 2 / ((2 nu k + 1)(2 nu k + 2))  -- small U."""
    out = np.zeros_like(U)
    term = np.ones_like(U)
    for k in range(kmax):
        out += term * 2.0 / ((2 * nu * k + 1) * (2 * nu * k + 2))
        term = term * (-U) / (k + 1)
        if np.all(np.abs(term) < 1e-17):
            break
    return out


def flory_end(U, nu):
    """int_0^1 exp(-U t^(2 nu)) dt = Gamma(a+1) P(a, U) / U^a, a = 1/(2 nu): the form-factor amplitude of a chain
    with Flory exponent nu relative to an end (Hammouda's generalised Gaussian chain), U = q^2 <R^2>/6."""
    a = 1.0 / (2.0 * nu)
    lg = sp.gammaln(a + 1)
    return _blend(U, lambda U: np.exp(lg + np.log(sp.gammainc(a, U)) - a * np.log(U)),
                  lambda U: _series_E(U, nu), 0.5)


def flory_ff(U, nu):
    """Form factor of a chain with Flory exponent nu (Hammouda 1993):
    (1/nu) [U^-a gamma(a, U) - U^-2a gamma(2a, U)], a = 1/(2 nu), gamma the lower incomplete gamma function."""
    a = 1.0 / (2.0 * nu)
    l1, l2 = sp.gammaln(a), sp.gammaln(2 * a)

    def exact(U):
        g1 = np.exp(l1 + np.log(sp.gammainc(a, U)) - a * np.log(U))
        g2 = np.exp(l2 + np.log(sp.gammainc(2 * a, U)) - 2 * a * np.log(U))
        return (g1 - g2) / nu
    return _blend(U, exact, lambda U: _series_F(U, nu), 0.5)


_LOOP_T, _LOOP_W = np.polynomial.legendre.leggauss(128)
_LOOP_T = 0.5 * (_LOOP_T + 1.0)
_LOOP_W = 0.5 * _LOOP_W


def flory_loop_ff(W, nu):
    """Form factor of a ring with Flory exponent nu: int_0^1 exp(-W [s(1-s)]^(2 nu)) ds (= 2 int_0^1/2),
    W = q^2 b^2 N^(2 nu)/6. Gauss-Legendre in t with s = t^3/2 (points crowd at the contact s -> 0)."""
    W = np.asarray(W, dtype=float)
    s = 0.5 * _LOOP_T ** 3
    jac = 1.5 * _LOOP_T ** 2
    g = (s * (1 - s)) ** (2 * nu)
    return 2.0 * (np.exp(-W[..., None] * g) * jac) @ _LOOP_W
