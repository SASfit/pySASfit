"""Python ports of the SASfit plugin guwelop: scattering of radial profiles η(x) = N (1 - F(x)), F the cumulative
GUWELOP distribution (generalised Weibull-Lomax-Poisson type), for spherical, cylindrical and planar cross-sections.

F(x) = 1 - expm1(-expm1(-β / expm1((x/λ)^α)^(1/γ)) θ) / expm1(θ); the profile is cut at bu (and at the
quantile 1 - 1e-14 of F, C: 1 - nriq). The Fourier integrals use Gauss-Legendre panels at quantiles of F, graded
towards 0, with nodes that resolve the oscillations of the kernel.
"""

from __future__ import annotations

import itertools
import math
from functools import lru_cache

import numpy as np
from scipy import special

from .common import make_spec

__all__ = ["SPECS"]

_EPS = 1e-14


@lru_cache(maxsize=64)
def _gl(n):
    return np.polynomial.legendre.leggauss(n)


def _pars(p):
    n, bu, theta = p[0], max(p[1], p[2]), p[4]
    alpha, beta, gamma, lam = abs(p[5]), abs(p[6]), abs(p[7]), abs(p[8])
    if lam <= 0 or alpha <= 0 or beta <= 0 or gamma <= 0:
        raise ValueError("lambda, alpha, beta, gamma > 0 required")
    if theta == 0:
        raise ValueError("theta != 0 required")
    return n, bu, theta, alpha, beta, gamma, lam


def cdf(x, p):
    """Cumulative GUWELOP distribution F(x) (0 for x <= 0, 1 where the formula overflows)."""
    _, _, theta, alpha, beta, gamma, lam = _pars(p)
    x = np.asarray(x, float)
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        e = np.expm1(np.power(np.maximum(x, 0.0) / lam, alpha))
        f = 1.0 - np.expm1(
            -np.expm1(-beta / np.power(e, 1.0 / gamma)) * theta
        ) / math.expm1(theta)
    f = np.where(np.isnan(f), 1.0, f)
    return np.where(x <= 0, 0.0, f)


def quantile(y, p):
    """Inverse of F (C formula)."""
    _, _, theta, alpha, beta, gamma, lam = _pars(p)
    y = np.asarray(y, float)
    with np.errstate(all="ignore"):
        inner = np.log(
            (1.0 - math.exp(theta)) * (-1.0 + 1.0 / (1.0 - math.exp(theta)) + y)
        )
        v = lam * np.power(
            np.log1p(np.power(-np.log(-(-theta + inner) / theta) / beta, -gamma)),
            1.0 / alpha,
        )
    return np.where(np.isnan(v), np.inf, v)


def _nodes(p, qmax):
    _, bu, *_ = _pars(p)
    xend = min(bu, float(quantile(1.0 - _EPS, p)))
    if not np.isfinite(xend) or xend <= 0:
        return np.zeros(0), np.zeros(0)
    qs = quantile(np.array([1e-12, 1e-8, 1e-5, 1e-3, 0.01, 0.05, 0.1, 0.2, 0.35, 0.5, 0.65, 0.8, 0.9, 0.95,
                            0.99, 0.999, 1 - 1e-5, 1 - 1e-8, 1 - 1e-11]), p)  # fmt: skip
    edges = {
        0.0,
        xend,
        *(xend * 2.0 ** -np.arange(1, 41)),
        *(x for x in qs if np.isfinite(x) and 0 < x < xend),
    }
    edges = np.array(sorted(edges))
    xs, ws = [], []
    for a, b in itertools.pairwise(edges):
        n = 24 + math.ceil(3.0 * qmax * (b - a) / np.pi)
        x, w = _gl(min(n, 2000))
        xs.append(a + 0.5 * (b - a) * (x + 1.0))
        ws.append(0.5 * (b - a) * w)
    return np.concatenate(xs), np.concatenate(ws)


def _amp(q, p, kernel):
    q = np.asarray(q, float)
    n = p[0]
    x, w = _nodes(p, float(np.max(q, initial=0.0)))
    if x.size == 0:
        return np.zeros(q.shape)
    eta = n * (1.0 - cdf(x, p))
    return (kernel(q[..., None], x) * (w * eta)).sum(-1)


def _k_sphere(q, x):
    return 4.0 * np.pi * x * x * np.sinc(q * x / np.pi)


def _k_cyl(q, x):
    return 2.0 * np.pi * x * special.j0(q * x)


def _k_planar(q, x):
    return 2.0 * np.cos(q * x)


def sphere_f(q, p):
    return _amp(q, p, _k_sphere)


def sphere(q, p):
    """p = [eta, dummy, bu, dummy, theta, alpha, beta, gamma, lambda]: (∫_0^bu η(x) 4πx² sin(qx)/(qx) dx)²."""
    return sphere_f(q, p) ** 2


def cyl_f(q, p):
    return _amp(q, p, _k_cyl)


def cyl(q, p):
    """p as sphere: P_cs = (∫_0^bu η(x) 2πx J0(qx) dx)² (cylindrical cross-section)."""
    return cyl_f(q, p) ** 2


def cyl_v(x, p, dist):
    return float(cyl_f(np.array([0.0]), p)[0]) + 0.0 * np.asarray(x, float)


def planar_f(q, p):
    return _amp(q, p, _k_planar)


def planar(q, p):
    """p as sphere: P_cs = (∫_0^bu η(x) 2 cos(qx) dx)² (planar cross-section)."""
    return planar_f(q, p) ** 2


def planar_v(x, p, dist):
    return float(planar_f(np.array([0.0]), p)[0]) + 0.0 * np.asarray(x, float)


def profile(x, p):
    """p as sphere: η(x) = eta (1 - F(x)) for 0 <= x <= bu, 0 beyond."""
    x = np.asarray(x, float)
    if np.any(x < 0):
        raise ValueError("x >= 0 required")
    _, bu, *_ = _pars(p)
    return np.where(x > bu, 0.0, p[0] * (1.0 - cdf(x, p)))


_G = "plugins/guwelop/sasfit_ff_guwelop_"
_PAR = ["eta", "dummy", "bu", "dummy", "theta", "alpha", "beta", "gamma", "lambda"]
_S = {0: (0.5, 2.0, "lin"), 2: (20.0, 100.0, "log"), 4: (0.5, 3.0, "lin"), 5: (1.0, 3.0, "lin"), 6: (0.5, 2.0, "lin"),
      7: (0.7, 2.0, "lin"), 8: (2.0, 10.0, "log")}  # fmt: skip


def _red(q, p):
    return np.asarray(q, float) * abs(p[8])


def _valid(q, p):
    """C comparison: q λ < 15 (beyond, I/I(0) < ~1e-13 and the adaptive C integral loses relative accuracy;
    checked against mpmath: the port stays at ~1e-11 there, C at 1e-8 ... 1e-4)."""
    return _red(q, p) < 15.0


SPECS = [
    make_spec("ff_guwelop_sphere", sphere, _PAR, _S, amplitude=sphere_f, source=_G + "sphere.c", reduced=_red,
              reduced_label="q λ", rtol=3e-7, c_integrates=True, c_valid=_valid,
              notes="C cuts η at the quantile 1 - nriq of F (SASfit setting), the port at 1 - 1e-14"),
    make_spec("ff_guwelop_cyl_Pcs", cyl, _PAR, _S, amplitude=cyl_f, volume=cyl_v, source=_G + "cyl_Pcs.c", reduced=_red,
              reduced_label="q λ", rtol=3e-7, c_integrates=True, c_valid=_valid),
    make_spec("ff_guwelop_planar_Pcs", planar, _PAR, _S, amplitude=planar_f, volume=planar_v,
              source=_G + "planar_Pcs.c", reduced=_red, reduced_label="q λ", rtol=1e-7, c_integrates=True,
              c_valid=_valid),
    make_spec("ff_guwelop_profile", profile, _PAR, _S, source=_G + "profile.c", rtol=1e-12),
]  # fmt: skip
