"""Python ports of the SASfit plugin gen_exp_sphere (radial profiles r^α e^{-√β r/Rg} and r^α e^{-β r²/2Rg²}).

Normalised so that ∫ρ dV = Δη V with V = 4π Rg³; the parameter Rg is the radius of gyration of the profile.
- generalised exponential sphere (β = (3+α)(4+α)): F = Δη V β^{(3+α)/2} (β+u²)^{-(1+α/2)} sin((2+α)θ)/((2+α)u),
  θ = atan(u/√β), u = q Rg; written as (θ/u) sinc((2+α)θ) (stable for u → 0 and α → -2).
- generalised Gaussian sphere (β = 3+α): F = Δη V 1F1(β/2; 3/2; -u²/(2β)). scipy's hyp1f1 agrees with mpmath to
  6e-15 (α in [-2.5, 8], u up to 1e3); the C code replaced failed GSL calls by an incorrect expression.
"""

from __future__ import annotations

import math

import numpy as np
from scipy import special

from .common import make_spec

__all__ = ["SPECS"]


def _check(p):
    if p[0] < 0:
        raise ValueError("Rg >= 0 required")
    if p[1] <= -3:
        raise ValueError("alpha > -3 required")


def gen_exp_sphere_f(q, p):
    """p = [Rg, alpha, Delta eta]."""
    q = np.asarray(q, float)
    _check(p)
    rg, al, de = p[0], p[1], p[2]
    beta = (3.0 + al) * (4.0 + al)
    u = q * rg
    x = u / math.sqrt(beta)
    th = np.arctan(x)
    with np.errstate(invalid="ignore", divide="ignore"):
        th_u = np.where(x < 1e-8, 1.0 - x * x / 3.0, th / np.where(x == 0, 1.0, x)) / math.sqrt(beta)  # θ/u
    pre = np.exp(0.5 * (3.0 + al) * math.log(beta) - (1.0 + 0.5 * al) * np.log(beta + u * u))
    return de * 4.0 * math.pi * rg**3 * pre * th_u * np.sinc((2.0 + al) * th / math.pi)


def gen_exp_sphere(q, p):
    return gen_exp_sphere_f(q, p) ** 2


def gen_sphere_v(x, p, dist):
    return 4.0 * math.pi * p[0] ** 3 + 0.0 * np.asarray(x, float)


def profile_gen_exp_sphere(r, p):
    r = np.abs(np.asarray(r, float))
    _check(p)
    rg, al, de = p[0], p[1], p[2]
    beta = (3.0 + al) * (4.0 + al)
    x = r / rg
    with np.errstate(divide="ignore", invalid="ignore"):
        lx = np.where(x > 0, np.log(np.where(x > 0, x, 1.0)), -np.inf)
        val = np.exp(0.5 * (3.0 + al) * math.log(beta) - x * math.sqrt(beta) + al * lx - special.gammaln(3.0 + al))
    if al == 0:
        val = np.exp(0.5 * 3.0 * math.log(beta) - x * math.sqrt(beta) - special.gammaln(3.0))
    return de * val


def gen_gauss_sphere_f(q, p):
    """p = [Rg, alpha, Delta eta]."""
    q = np.asarray(q, float)
    _check(p)
    rg, al, de = p[0], p[1], p[2]
    beta = 3.0 + al
    u = q * rg
    return de * 4.0 * math.pi * rg**3 * special.hyp1f1(0.5 * beta, 1.5, -(u * u) / (2.0 * beta))


def gen_gauss_sphere(q, p):
    return gen_gauss_sphere_f(q, p) ** 2


def profile_gen_gauss_sphere(r, p):
    r = np.abs(np.asarray(r, float))
    _check(p)
    rg, al, de = p[0], p[1], p[2]
    beta = 3.0 + al
    x = r / rg
    with np.errstate(divide="ignore", invalid="ignore"):
        lx = np.where(x > 0, np.log(np.where(x > 0, x, 1.0)), -np.inf)
        if al == 0:
            lx = np.zeros(x.shape)
        val = np.exp(-0.5 * (1.0 + al) * math.log(2.0) + 0.5 * beta * math.log(beta) - 0.5 * beta * x * x + al * lx
                     - special.gammaln(0.5 * beta))  # fmt: skip
    return de * val


_GE = "plugins/gen_exp_sphere/sasfit_ff_"
_P = ["Rg", "alpha", "Delta eta"]
_S = {0: (2.0, 50.0, "log"), 1: (-2.5, 6.0, "lin"), 2: (0.5, 2.0, "lin")}


def _qr(q, p):
    return np.asarray(q, float) * p[0]


SPECS = [
    make_spec("ff_gen__exp__sphere", gen_exp_sphere, _P, _S, amplitude=gen_exp_sphere_f, volume=gen_sphere_v,
              source=_GE + "gen__exp__sphere.c", reduced=_qr, reduced_label="q Rg", rtol=1e-11),
    make_spec("ff_profile__gen__exp__sphere", profile_gen_exp_sphere, _P, _S, source=_GE + "profile__gen__exp__sphere.c",
              rtol=1e-12),
    make_spec("ff_gen__gauss__sphere", gen_gauss_sphere, _P, _S, amplitude=gen_gauss_sphere_f, volume=gen_sphere_v,
              source=_GE + "gen__gauss__sphere.c", reduced=_qr, reduced_label="q Rg", rtol=1e-10),
    make_spec("ff_profile__gen__gauss__sphere", profile_gen_gauss_sphere, _P, _S,
              source=_GE + "profile__gen__gauss__sphere.c", rtol=1e-12),
]  # fmt: skip
