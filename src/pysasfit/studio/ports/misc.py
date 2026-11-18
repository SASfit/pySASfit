"""Python ports of small SASfit plugins: polynom, cv, generalized_formfactor.

C deviations: generalized_form_factor left its result uninitialised for sigma_a != 0 (fixed on 2026-11-18: log-normal
average over a, as the lesser generalized form factors do); the lesser generalized form factors
cut the log-normal integral at a finite range (find_LogNorm_int_range), the port uses Gauss-Hermite quadrature.
"""

from __future__ import annotations

import numpy as np
from scipy import special

from .common import hyp0f1_neg, lnorm_nodes, make_spec

__all__ = ["SPECS"]


def polynom(q, p):
    """p = [p0 ... p4]: p0 + p1 q + p2 q² + p3 q³ + p4 q⁴."""
    q = np.asarray(q, float)
    return p[0] + q * (p[1] + q * (p[2] + q * (p[3] + q * p[4])))


def cv2(x, p):
    """p = [I_bckgr, x_min, I_scaling]: I_bckgr + I_scaling (x - x_min)²."""
    return p[0] + p[2] * (np.asarray(x, float) - p[1]) ** 2


def cv(q, p):
    """p = [m, x0]: |m (q - x0)|."""
    return np.abs(p[0] * (np.asarray(q, float) - p[1]))


def _lesser_term(q, x, eta, alpha, gama, nu, power):
    """(q^(-γ/2) η x^α 2^(-ν) 0F1(;ν+1;-(qx)²/4)/Γ(ν+1))^power (0 at x = 0)."""
    with np.errstate(divide="ignore", invalid="ignore"):
        base = (
            np.power(q, -gama / 2.0)
            * eta
            * np.power(x, alpha)
            * 2.0**-nu
            * hyp0f1_neg(nu + 1.0, q * x)
            / special.gamma(nu + 1.0)
        )
    return np.where(x == 0, 0.0, base**power)


def _lesser(q, p, alpha, dim, power):
    q = np.asarray(q, float)
    a, sigma, s, pp, eta = p[0], abs(p[1]), p[4], p[5], p[6]
    if a < 0 or s < 0 or s > 3:
        raise ValueError("a >= 0 and 0 <= s <= 3 required")
    gama, beta = s, pp - s - 1.0
    nu = beta / 2.0
    if sigma == 0:
        return _lesser_term(q, a, eta, alpha, gama, nu, power)
    nus, w = lnorm_nodes(sigma, 48)
    x = a * nus
    return np.sum(
        w * _lesser_term(q[..., None], x, eta, alpha, gama, nu, power), axis=-1
    )


def lesser1(q, p):
    """p = [a, sigma_a, dummy, dim, s, p, eta]: α = dim - s, β = p - s - 1, γ = s, ν = β/2."""
    return _lesser(q, p, p[3] - p[4], p[3], 2)


def lesser1_f(q, p):
    return _lesser(q, p, p[3] - p[4], p[3], 1)


def lesser1_v(x, p, dist):
    return p[0] ** (p[3] - p[4]) + 0.0 * np.asarray(x, float)


def lesser2(q, p):
    """p = [a, sigma_a, alpha, dummy, s, p, eta]: β = p - s - 1, γ = s, ν = β/2."""
    return _lesser(q, p, p[2], p[2] + p[4], 2)


def lesser2_f(q, p):
    return _lesser(q, p, p[2], p[2] + p[4], 1)


def lesser2_v(x, p, dist):
    return p[0] ** p[2] + 0.0 * np.asarray(x, float)


def gff(q, p):
    """p = [a, sigma_a, alpha, beta, gamma, nu, eta]:
    η² a^(2α) (a q)^(2ν-β) 4^(-ν) (0F1(;ν+1;-(qa)²/4)/Γ(ν+1))² (sigma_a: log-normal average over a)."""
    q = np.asarray(q, float)
    a, sigma, alpha, beta, nu, eta = p[0], abs(p[1]), p[2], p[3], p[5], p[6]
    if a < 0 or alpha <= 0 or beta < 0 or nu < 0:
        raise ValueError("a >= 0, alpha > 0, beta >= 0, nu >= 0 required")
    if a == 0:
        return np.zeros(q.shape)

    def one(x, qq):
        with np.errstate(divide="ignore", invalid="ignore"):
            return (np.power(x * qq, 2 * nu - beta) * 4.0**-nu * (hyp0f1_neg(nu + 1.0, qq * x) / special.gamma(nu + 1.0)) ** 2
                    * np.power(x, 2 * alpha) * eta**2)  # fmt: skip

    if sigma == 0:
        return one(a, q)
    nus, w = lnorm_nodes(sigma, 48)
    return np.sum(w * one(a * nus, q[..., None]), axis=-1)


def _qa(q, p):
    return np.asarray(q, float) * abs(p[0])


_GF = "plugins/generalized_formfactor/sasfit_"
SPECS = [
    make_spec("ff_polynom", polynom, ["p0", "p1", "p2", "p3", "p4"], {k: (-2.0, 2.0, "lin") for k in range(5)},
              source="plugins/polynom/sasfit_ff_polynom.c", rtol=1e-13),
    make_spec("ff_cv2", cv2, ["I_bckgr", "x_min", "I_scaling"], {0: (0.0, 2.0, "lin"), 1: (0.0, 2.0, "lin"), 2: (-2.0, 2.0, "lin")},
              source="plugins/polynom/sasfit_ff_cv2.c", rtol=1e-13),
    make_spec("ff_cv", cv, ["m", "x0"], {0: (-2.0, 2.0, "lin"), 1: (0.0, 2.0, "lin")}, source="plugins/cv/sasfit_ff_cv.c",
              rtol=1e-13),
    make_spec("ff_generalized_form_factor", gff, ["a", "sigma_a", "alpha", "beta", "gamma", "nu", "eta"],
              {0: (1.0, 50.0, "log"), 2: (0.5, 3.0, "lin"), 3: (0.0, 2.0, "lin"), 5: (0.0, 2.0, "lin"), 6: (0.1, 3.0, "lin")},
              source=_GF + "ff_generalized_form_factor.c", reduced=_qa, reduced_label="q a", rtol=1e-8,
              notes="compared with sigma_a = 0 (C cuts the log-normal integral)"),
    make_spec("ff_lesser_generalized_form_factor", lesser1, ["a", "sigma_a", "dummy", "dim", "s", "p", "eta"],
              {0: (1.0, 50.0, "log"), 3: (1.0, 3.0, "lin"), 4: (0.0, 1.0, "lin"), 5: (2.0, 4.0, "lin"), 6: (0.1, 3.0, "lin")},
              amplitude=lesser1_f, volume=lesser1_v, source=_GF + "ff_lesser_generalized_form_factor.c", reduced=_qa,
              reduced_label="q a", rtol=1e-8, notes="compared with sigma_a = 0 (C cuts the log-normal integral)"),
    make_spec("ff_lesser_generalized_form_factor2", lesser2, ["a", "sigma_a", "alpha", "dummy", "s", "p", "eta"],
              {0: (1.0, 50.0, "log"), 2: (0.5, 3.0, "lin"), 4: (0.0, 1.0, "lin"), 5: (2.0, 4.0, "lin"), 6: (0.1, 3.0, "lin")},
              amplitude=lesser2_f, volume=lesser2_v, source=_GF + "ff_lesser_generalized_form_factor2.c", reduced=_qa,
              reduced_label="q a", rtol=1e-8, notes="compared with sigma_a = 0 (C cuts the log-normal integral)"),
]  # fmt: skip
