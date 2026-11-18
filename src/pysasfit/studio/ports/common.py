"""Numerically stable building blocks shared by the Python ports (vectorised over q)."""

from __future__ import annotations

import math

import numpy as np
from scipy import special

__all__ = [
    "debye",
    "debye_amplitude",
    "hyp0f1_neg",
    "lower_gamma_ratio",
    "make_spec",
    "phi_sphere",
    "sphere_amplitude",
    "zeros_volume",
]


def phi_sphere(x):
    """3 (sin x - x cos x)/x³ (1 at x = 0), accurate for all x."""
    x = np.asarray(x, float)
    small = np.abs(x) < 0.05
    xs = np.where(small, 1.0, x)
    x2 = x * x
    series = 1.0 - x2 / 10.0 * (1.0 - x2 / 28.0 * (1.0 - x2 / 54.0 * (1.0 - x2 / 88.0)))
    return np.where(small, series, 3.0 * (np.sin(xs) - xs * np.cos(xs)) / xs**3)


def sphere_amplitude(q, r, eta):
    """η 4π/3 R³ Φ(qR): SASfit's ``sasfit_ff_sphere_f`` (sphericalshells)."""
    r = np.asarray(r, float)
    return eta * 4.0 / 3.0 * np.pi * r**3 * phi_sphere(np.asarray(q, float) * r)


def debye(u):
    """2 (u - 1 + e^-u)/u² (1 at u = 0), Debye function of u = (q Rg)²."""
    u = np.asarray(u, float)
    small = np.abs(u) < 0.05
    us = np.where(small, 1.0, u)
    series = 1.0 - u / 3.0 * (
        1.0
        - u
        / 4.0
        * (1.0 - u / 5.0 * (1.0 - u / 6.0 * (1.0 - u / 7.0 * (1.0 - u / 8.0))))
    )
    return np.where(small, series, 2.0 * (us + np.expm1(-us)) / us**2)


def debye_amplitude(u):
    """(1 - e^-u)/u (1 at u = 0)."""
    u = np.asarray(u, float)
    small = np.abs(u) < 1e-8
    us = np.where(small, 1.0, u)
    return np.where(small, 1.0 - u / 2.0, -np.expm1(-us) / us)


def lower_gamma_ratio(a, u, nmax=60):
    """u^-a γ(a, u) = ∫_0^1 ... = Σ_n (-u)^n / (n! (a+n)) for u ≥ 0, a > 0 (series for u < 2, else via the
    regularised incomplete gamma function)."""
    u = np.asarray(u, float)
    out = np.empty(u.shape)
    small = u < 2.0
    if np.any(small):
        us = u[small]
        term = np.ones_like(us)
        s = term / a
        for n in range(1, nmax):
            term = term * (-us) / n
            s = s + term / (a + n)
        out[small] = s
    big = ~small
    if np.any(big):
        ub = u[big]
        out[big] = np.exp(special.gammaln(a) - a * np.log(ub)) * special.gammainc(a, ub)
    return out


def hyp0f1_neg(b, x):
    """0F1(; b; -x²/4) = Γ(b) (x/2)^(1-b) J_{b-1}(x) for x ≥ 0 (1 at x = 0)."""
    x = np.asarray(x, float)
    small = x < 1e-3
    xs = np.where(small, 1.0, x)
    z = -0.25 * x * x
    series = 1.0 + z / b * (1.0 + z / (2.0 * (b + 1.0)) * (1.0 + z / (3.0 * (b + 2.0))))
    with np.errstate(over="ignore", invalid="ignore"):
        big = np.exp(special.gammaln(b) + (1.0 - b) * np.log(xs / 2.0)) * special.jv(
            b - 1.0, xs
        )
        if b <= 0:
            big = special.hyp0f1(b, -0.25 * xs * xs)
    return np.where(small, series, big)


def c_valid_reduced(xmin, k=0):
    """c_valid: q·|p[k]| >= xmin (the C function loses digits by cancellation below)."""

    def valid(q, p):
        return np.asarray(q, float) * abs(p[k]) >= xmin

    return valid


def zeros_volume(x, p, dist=0):
    return np.zeros(np.shape(x))


def make_spec(name, intensity, parameters, sample, *, amplitude=None, volume=None, rtol=1e-9, source="",
              notes="", c_integrates=False, c_compat=False, reduced=None, reduced_label="q R", label="",
              group="", c_reference=True, c_valid=None):  # fmt: skip
    """Keyword dict for :class:`pysasfit.studio.ports.PortSpec` of an isotropic port."""

    def _vec(f):
        if f is None:
            return None

        def g(x, p, dist=0):
            return np.asarray(
                f(np.asarray(x, float), np.asarray(p, float), int(dist)), float
            )

        g.vectorized = True
        return g

    def _amp(f):
        if f is None:
            return None

        def g(q, p, psi=None):
            return np.asarray(f(np.asarray(q, float), np.asarray(p, float)), float)

        return g

    def _int(q, p, psi=None, c_compat=False):
        if c_compat:
            return intensity(np.asarray(q, float), np.asarray(p, float), c_compat=True)
        return intensity(np.asarray(q, float), np.asarray(p, float))

    _int.__name__ = name
    _int.__doc__ = intensity.__doc__
    return {
        "name": name,
        "intensity": _int,
        "parameters": list(parameters),
        "psi_index": None,
        "amplitude": _amp(amplitude),
        "volume": _vec(volume),
        "sample": dict(sample),
        "rtol": rtol,
        "source": source,
        "notes": notes,
        "c_integrates": c_integrates,
        "c_compat": c_compat,
        "reduced": reduced,
        "reduced_label": reduced_label,
        "label": label,
        "group": group,
        "c_reference": c_reference,
        "c_valid": c_valid,
    }


def lnorm_nodes(sigma, n=32):
    """Nodes ν and weights of the normalised log-normal distribution (median 1) by Gauss-Hermite in ln ν."""
    z, w = np.polynomial.hermite_e.hermegauss(n)
    return np.exp(sigma * z), w / math.sqrt(2.0 * math.pi)
