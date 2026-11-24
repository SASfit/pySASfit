"""Python ports of the SASfit plugin hyperbranched_polymer (batch 8, 2026-11-24): combs, ideal dendrimers,
fractal hyperbranched polymers and f-functional homopolymers (Gaussian statistics, I(0) = I0).

Combs (x = q Rg, s = x², backbone fraction K = lambda, number of side chains per backbone f = zeta; a = s(1 - K)/f
side-chain and b = s K backbone Debye variables): written with the positive auxiliary functions
g(z) = (e^{-z} - 1 + z)/z² and h(z) = (1 - e^{-z})/z, so that no terms cancel,

* ff_comb_hetero:  I = 2 I0 [f a² g(a) + g(b)(b + f a h(a))²] / (x⁴ (1 + (1 - K)²/f)),
* ff_comb_random:  I = 2 I0 [f a² g(a) + g(b)(b² + 2 f a b h(a) + f(f - 1) a² h(a)²)] / x⁴,
* ff_comb_regular: I = 2 I0 [f a² g(a) + b² g(b) + 2 f a h(a) c (G₊(c)/H₊(c) + f g(f c)/H₊(c))
  + (f - 1) a² h(a)² (G₊(c)/H₊(c)² + (f - 1) g((f - 1) c)/H₊(c)²)] / x⁴, c = b/(f + 1),
  G₊(c) = (e^c - 1 - c)/c², H₊(c) = (e^c - 1)/c;

algebraically identical to the C formulas (which subtract O(x²) terms to get O(x⁴) and returned 2 I0/x² or
2 I0/(x⁴(1 + (1-K)²/f)) where their 0/0 gave NaN, e.g. for K = 0).

* ff_ideal_dendrimer: closed forms of SASfit for generations 1-9 (0: Debye), I = (2/u²) Σ_k c_k (e^{-λ_k u}
  - 1 + λ_k u) = 2 Σ_k c_k λ_k² g(λ_k u), u = (q a)²; non-integer g interpolates linearly between floor(g) and
  floor(g) + 1. The C code summed the exponentials directly (coefficients up to 786432 cancel: 2e-4 relative error
  at u = 1e-3 for g = 9), rounded g to the nearest integer (extrapolating for fractional parts > 0.5) and read
  past its table for g = 9.
* ff_fract_hyperbranched_polym: I0 P_DB(q ξ_b)/P_lin(q ξ_l), P_lin(u) = sin((d - 1) atan u)/((d - 1) u
  (1 + u²)^{(d-1)/2}), P_DB = P_lin², ξ_b² = (C + 1) Rg²/(d(d + 1)), ξ_l² = 2C Rg²/(d(d + 1)).
* ff_f_funct_homo_polym: f-functional homopolymer (branching probability alpha, segment length sigma), type
  0: I0/(1 + x²/3), 1: I0 α f φ/(1 - α(f - 1)φ) (1 - α(f - 1))/(α f), φ = e^{-x²/6}, x² = q² σ²/(2(1 - (f-1)α)),
  2 and 3: I0 ∫_0^∞ (1 - α_s) exp(-(1 - α_s) n - (σq)² n^{2ν}/6) dn (Gaussian-like, ν = 0.588) and with the
  exponent 2.18 ν and 6.2 (α_s = α(1 - f)). C before 2026-11-24: types 2 and 3 were not multiplied by I0.
"""

from __future__ import annotations

import math

import numpy as np
from scipy import integrate

from .common import make_spec

__all__ = ["SPECS", "g_aux", "h_aux"]


def g_aux(z):
    """(e^{-z} - 1 + z)/z² (1/2 at 0), series below 0.1."""
    z = np.asarray(z, float)
    small = np.abs(z) < 0.1
    zs = np.where(small, 1.0, z)
    ser = np.zeros_like(z)
    t = np.full_like(z, 0.5)
    for n in range(1, 14):
        ser = ser + t
        t = -t * z / (n + 2)
    with np.errstate(over="ignore", invalid="ignore"):
        direct = (np.expm1(-zs) + zs) / (zs * zs)
    return np.where(small, ser, direct)


def h_aux(z):
    """(1 - e^{-z})/z (1 at 0)."""
    z = np.asarray(z, float)
    zs = np.where(z == 0, 1.0, z)
    return np.where(z == 0, 1.0, -np.expm1(-zs) / zs)


def _gplus(c):
    """G₊(c)/H₊(c) and G₊(c)/H₊(c)², G₊ = (e^c - 1 - c)/c², H₊ = (e^c - 1)/c, without overflow."""
    c = np.asarray(c, float)
    small = c < 0.1
    cs = np.where(small, 0.1, c)
    ser = np.zeros_like(c)
    t = np.full_like(c, 0.5)
    for n in range(1, 14):
        ser = ser + t
        t = t * c / (n + 2)
    hp = np.where(c == 0, 1.0, np.expm1(np.where(small, c, 0.0)) / np.where(c == 0, 1.0, c))
    em = np.exp(-cs)
    hp = np.where(small, hp, 1.0)
    r1 = np.where(small, ser / hp, (1.0 - (1.0 + cs) * em) / (cs * (-np.expm1(-cs))))
    r2 = np.where(small, ser / (hp * hp), (em - (1.0 + cs) * em * em) / np.expm1(-cs) ** 2)
    inv = np.where(small, 1.0 / hp, cs * em / (-np.expm1(-cs)))
    return r1, r2, inv


def _comb_vars(q, p):
    x2 = (np.asarray(q, float) * p[0]) ** 2
    f, K = p[1], p[2]
    return x2, f, K, x2 * (1.0 - K) / f, x2 * K


def _comb_check(p, name):
    if p[0] <= 0 or p[1] <= 1 or not 0 <= p[2] <= 1:
        raise ValueError(f"{name}: Rg > 0, zeta > 1 and 0 <= lambda <= 1 required")


def comb_hetero(q, p):
    """p = [a (Rg), zeta (f), lambda (K), I0]."""
    _comb_check(p, "comb_hetero")
    x2, f, K, a, b = _comb_vars(q, p)
    T = f * a * a * g_aux(a) + g_aux(b) * (b + f * a * h_aux(a)) ** 2
    x4 = np.where(x2 == 0, 1.0, x2 * x2)
    return np.where(x2 == 0, p[3], 2.0 * p[3] * T / (x4 * (1.0 + (1.0 - K) ** 2 / f)))


def comb_random(q, p):
    _comb_check(p, "comb_random")
    x2, f, K, a, b = _comb_vars(q, p)
    ha = h_aux(a)
    T = f * a * a * g_aux(a) + g_aux(b) * (b * b + 2.0 * f * a * b * ha + f * (f - 1.0) * (a * ha) ** 2)
    x4 = np.where(x2 == 0, 1.0, x2 * x2)
    return np.where(x2 == 0, p[3], 2.0 * p[3] * T / x4)


def comb_regular(q, p):
    _comb_check(p, "comb_regular")
    x2, f, K, a, b = _comb_vars(q, p)
    c = b / (f + 1.0)
    ha = h_aux(a)
    r1, r2, inv = _gplus(c)
    T = (f * a * a * g_aux(a) + b * b * g_aux(b)
         + 2.0 * f * a * ha * c * (r1 + f * g_aux(f * c) * inv)
         + (f - 1.0) * (a * ha) ** 2 * (r2 + (f - 1.0) * g_aux((f - 1.0) * c) * inv * inv))  # fmt: skip
    x4 = np.where(x2 == 0, 1.0, x2 * x2)
    return np.where(x2 == 0, p[3], 2.0 * p[3] * T / x4)


# generation -> [(c_k, numerator, denominator of λ_k)] of I = 2 Σ c_k λ_k² g(λ_k u)
_DENDRIMER = {
    0: [(-1, 1, 1)],
    1: [(-3, 1, 3), (3, 2, 3)],
    2: [(-3, 1, 9), (-12, 1, 3), (12, 4, 9)],
    3: [(-3, 1, 21), (6, 2, 21), (-12, 1, 7), (-48, 5, 21), (48, 2, 7)],
    4: [(-3, 1, 93), (42, 2, 93), (-12, 1, 31), (72, 4, 93), (-48, 5, 93), (96, 2, 31), (-192, 7, 93), (-768, 3, 31), (768, 10, 93)],
    5: [(-3, 1, 189), (90, 2, 189), (-12, 1, 63), (168, 4, 189), (-48, 5, 189), (288, 2, 63), (-192, 1, 27), (384, 8, 189), (-768, 1, 21), (-3072, 11, 189), (3072, 4, 63)],
    6: [(672, 2, 127), (360, 4, 381), (-12, 1, 127), (-3, 1, 381), (-3072, 11, 381), (-12288, 13, 381), (1152, 8, 381), (-48, 5, 381), (186, 2, 381), (1536, 10, 381), (-768, 3, 127), (-192, 7, 381), (12288, 14, 381)],
    7: [(6144, 4, 255), (-768, 1, 85), (2688, 8, 765), (-48, 1, 153), (-12, 1, 255), (378, 2, 765), (-3, 1, 765), (-192, 7, 765), (-3072, 11, 765), (-12288, 13, 765), (-49152, 1, 51), (1440, 2, 255), (4608, 2, 153), (744, 4, 765), (49152, 16, 765)],
    8: [(-196608, 17, 1533), (24576, 2, 219), (-12288, 13, 1533), (10752, 10, 1533), (5760, 8, 1533), (-192, 1, 219), (-48, 5, 1533), (762, 2, 1533), (-3, 1, 1533), (-3072, 11, 1533), (-49152, 5, 511), (1512, 4, 1533), (18432, 4, 511), (-768, 3, 511), (2976, 2, 511), (-12, 1, 511), (196608, 6, 511)],
    9: [(-48, 5, 3069), (-49152, 5, 1023), (-12288, 13, 3069), (43008, 4, 1023), (23040, 10, 3069), (-192, 7, 3069), (6048, 2, 1023), (3048, 4, 3069), (-12, 1, 1023), (-768, 1, 341), (-3, 1, 3069), (-3072, 1, 279), (1530, 2, 3069), (-196608, 17, 3069), (-786432, 19, 3069), (73728, 14, 3069), (11904, 8, 3069), (98304, 16, 3069), (786432, 20, 3069)],
}


def _dendrimer_gen(u, gen):
    if gen == 0:
        return 2.0 * g_aux(u)
    out = np.zeros_like(u)
    for c, n, d in _DENDRIMER[gen]:
        lam = n / d
        out = out + c * lam * lam * g_aux(lam * u)
    return 2.0 * out


def ideal_dendrimer(q, p):
    """p = [a (Rg), dummy, g (generation 0 ... 9), I0]."""
    if p[0] <= 0 or not 0 <= p[2] <= 9:
        raise ValueError("ideal_dendrimer: a > 0 and 0 <= g <= 9 required")
    u = (np.asarray(q, float) * p[0]) ** 2
    gl = min(int(math.floor(p[2])), 9)
    w = p[2] - gl
    res = _dendrimer_gen(u, gl)
    if w > 0:
        res = (1.0 - w) * res + w * _dendrimer_gen(u, gl + 1)
    return p[3] * res


def _plin(u, d):
    u = np.asarray(u, float)
    th = np.arctan(u)
    us = np.where(u == 0, 1.0, u)
    val = np.sinc((d - 1.0) * th / math.pi) * th / us * (1.0 + u * u) ** (-(d - 1.0) / 2.0)
    return np.where(u == 0, 1.0, val)


def fract_hyperbranched_polym(q, p):
    """p = [Rg, d, C, I0]."""
    Rg, d, C, I0 = p[0], p[1], p[2], p[3]
    if Rg <= 0 or not 1 <= d <= 3 or not 0 <= C <= 1 or I0 < 0:
        raise ValueError("fract_hyperbranched_polym: Rg > 0, 1 <= d <= 3, 0 <= C <= 1, I0 >= 0 required")
    q = np.asarray(q, float)
    xb = math.sqrt((C + 1.0) / (d * (d + 1.0))) * Rg
    xl = math.sqrt(2.0 * C / (d * (d + 1.0))) * Rg
    return I0 * _plin(q * xb, d) ** 2 / _plin(q * xl, d)


def f_funct_homo_polym(q, p):
    """p = [sigma, f, alpha, I0, type]."""
    sig, f, al, I0, typ = p[0], p[1], p[2], p[3], int(round(p[4]))
    if sig <= 0 or f <= 1 or not 0 <= al <= 1 or (al > 0 and f >= (1.0 + al) / al):
        raise ValueError("f_funct_homo_polym: sigma > 0, f > 1, 0 <= alpha <= 1, f < (1+alpha)/alpha required")
    if al >= 1.0 / (f - 1.0):
        raise ValueError("f_funct_homo_polym: alpha < 1/(f-1) required")
    q = np.atleast_1d(np.asarray(q, float))
    s2z = sig * sig / (2.0 * (1.0 - (f - 1.0) * al))
    x2 = q * q * s2z
    if typ == 0:
        out = I0 / (1.0 + x2 / 3.0)
    elif typ == 1:
        phi = np.exp(-x2 / 6.0)
        out = I0 * al * f * phi / (1.0 - al * (f - 1.0) * phi) * (1.0 - al * (f - 1.0)) / (al * f)
    elif typ in (2, 3):
        a_s = al * (1.0 - f)
        e, den = (2.0 * 0.588, 6.0) if typ == 2 else (2.18 * 0.588, 6.2)
        k = 1.0 - a_s
        out = np.empty(q.shape)
        for i, qi in enumerate(q):
            beta = (sig * qi) ** 2 / den / k ** e        # n = t/k
            fn = lambda t: math.exp(-t - beta * t ** e)  # noqa: E731
            out[i] = I0 * integrate.quad(fn, 0.0, np.inf, epsabs=0.0, epsrel=1e-13, limit=200)[0]
    else:
        raise ValueError(f"f_funct_homo_polym: type {typ} unknown (0 ... 3)")
    return out


# ------------------------------------------------------------------------------------------------ specs
_SRC = "plugins/hyperbranched_polymer/sasfit_ff_"
_GROUP = "ff_polymer_branched"
_CS = {0: (1.0, 30.0, "log"), 1: (1.5, 20.0, "log"), 2: (0.05, 0.95, "lin"), 3: (0.5, 2.0, "lin")}


def _red(q, p):
    return np.asarray(q, float) * abs(p[0])


SPECS = [
    make_spec("ff_comb_hetero", comb_hetero, ["a", "zeta", "lambda", "I0"], _CS, source=_SRC + "comb_hetero.c",
              reduced=_red, reduced_label="q a", rtol=1e-9, group=_GROUP),
    make_spec("ff_comb_random", comb_random, ["a", "zeta", "lambda", "I0"], _CS, source=_SRC + "comb_random.c",
              reduced=_red, reduced_label="q a", rtol=1e-9, group=_GROUP),
    make_spec("ff_comb_regular", comb_regular, ["Rg", "zeta", "lambda", "I0"], _CS, source=_SRC + "comb_regular.c",
              reduced=_red, reduced_label="q Rg", rtol=1e-9, group=_GROUP),
    make_spec("ff_ideal_dendrimer", ideal_dendrimer, ["a", "dummy", "g", "I0"],
              {0: (1.0, 30.0, "log"), 2: (0.0, 9.0, "int"), 3: (0.5, 2.0, "lin")}, source=_SRC + "ideal_dendrimer.c",
              reduced=_red, reduced_label="q a", rtol=1e-9, group=_GROUP),
    make_spec("ff_fract_hyperbranched_polym", fract_hyperbranched_polym, ["Rg", "d", "C", "I0"],
              {0: (1.0, 30.0, "log"), 1: (1.1, 2.9, "lin"), 2: (0.05, 1.0, "lin"), 3: (0.5, 2.0, "lin")},
              source=_SRC + "fract_hyperbranched_polym.c", reduced=_red, reduced_label="q Rg", rtol=1e-9,
              group=_GROUP),
    make_spec("ff_f_funct_homo_polym", f_funct_homo_polym, ["sigma", "f", "alpha", "I0", "type"],
              {0: (1.0, 20.0, "log"), 1: (2.5, 4.0, "lin"), 2: (0.05, 0.3, "lin"), 3: (0.5, 2.0, "lin"),
               4: (0.0, 3.0, "int")}, source=_SRC + "f_funct_homo_polym.c",
              reduced=lambda q, p: np.asarray(q, float) * abs(p[0]), reduced_label="q sigma", rtol=1e-9,
              group=_GROUP),
]
