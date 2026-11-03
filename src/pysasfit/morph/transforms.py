"""Exact sine transforms of piecewise-linear functions (Filon-type quadrature).

All isotropic Fourier transforms in :mod:`pysasfit.morph` reduce to

    T(r) = integral_{x_0}^{x_N} u(x) sin(x r) dx

with ``u`` given on nodes ``x_0 < ... < x_N`` and linear in between. The integral
is evaluated *exactly* for that interpolant, so it stays accurate for any r: the
quadrature does not need many points per oscillation (a plain trapezoid rule on
the data grid fails as soon as r * dq ~ 1, which is where SAXSMorph's
high-frequency ripple in Gamma(r) and f(k) comes from).

For small ``r * x_N`` the closed form suffers from cancellation; there the
Taylor series in r with exact moments of the interpolant is used instead.

The Porod tail u(q) = K q^-3 beyond q_N (i.e. I(q) = K q^-4) is integrated
analytically with the sine integral, see :func:`porod_tail`.
"""

from __future__ import annotations

import numpy as np
from scipy.special import sici

__all__ = ["sine_transform_pl", "moments_pl", "porod_tail"]

#: below this value of r * x_N the Taylor series (to r^7) is used
SERIES_LIMIT = 0.05


def moments_pl(x, u, n):
    """Exact moment ``integral x^n u(x) dx`` of the piecewise-linear interpolant of (x, u)."""
    x = np.asarray(x, float)
    u = np.asarray(u, float)
    a, b = x[:-1], x[1:]
    s = (u[1:] - u[:-1]) / (b - a)
    alpha = u[:-1] - s * a
    return float(np.sum(alpha * (b ** (n + 1) - a ** (n + 1)) / (n + 1)
                        + s * (b ** (n + 2) - a ** (n + 2)) / (n + 2)))


def sine_transform_pl(x, u, r, chunk=256):
    """``integral_{x_0}^{x_N} u(x) sin(x r) dx`` for every r, u piecewise linear on the nodes x.

    Per segment [a, b] with slope s:  u_a cos(ar)/r - u_b cos(br)/r + s (sin br - sin ar)/r^2;
    the first two terms telescope over the segments.
    """
    x = np.asarray(x, float)
    u = np.asarray(u, float)
    r = np.atleast_1d(np.asarray(r, float))
    if x.ndim != 1 or x.shape != u.shape or x.size < 2:
        raise ValueError("x and u must be 1D arrays of the same length >= 2")
    if np.any(np.diff(x) <= 0):
        raise ValueError("x must be strictly increasing")
    s = np.diff(u) / np.diff(x)
    out = np.empty_like(r)
    xN = x[-1]
    small = np.abs(r) * max(abs(x[0]), abs(xN)) < SERIES_LIMIT
    if np.any(small):
        m1, m3, m5, m7 = (moments_pl(x, u, n) for n in (1, 3, 5, 7))
        rs = r[small]
        out[small] = rs * m1 - rs ** 3 * m3 / 6.0 + rs ** 5 * m5 / 120.0 - rs ** 7 * m7 / 5040.0
    big = np.flatnonzero(~small)
    for i0 in range(0, big.size, chunk):
        idx = big[i0:i0 + chunk]
        rr = r[idx][:, None]
        sx = np.sin(x[None, :] * rr)                       # (nr, nx)
        term = (sx[:, 1:] - sx[:, :-1]) @ s                 # sum_i s_i (sin b r - sin a r)
        rv = r[idx]
        out[idx] = (u[0] * np.cos(x[0] * rv) - u[-1] * np.cos(xN * rv)) / rv + term / rv ** 2
    return out


def _j3(a):
    """integral_a^inf sin(x) x^-3 dx  (a > 0)."""
    si, _ci = sici(a)
    return np.sin(a) / (2 * a * a) + np.cos(a) / (2 * a) - 0.5 * (0.5 * np.pi - si)


def porod_tail(K, qN, r):
    """Contribution of I(q) = K q^-4 for q > qN to ``integral q^2 I(q) sin(qr)/(qr) dq``.

    Equals K r J3(qN r) with J3(a) = integral_a^inf sin(x)/x^3 dx; the limit at r = 0
    is K / qN (the Porod part of the invariant).
    """
    r = np.atleast_1d(np.asarray(r, float))
    out = np.full_like(r, K / qN)
    nz = r > 0
    out[nz] = K * r[nz] * _j3(qN * r[nz])
    return out
