"""Python ports of the SASfit plugin ellipsoidal_shell (batch 7, 2026-11-24): randomly oriented core-shell spheroids
(polar radius R_p, equatorial radius R_e, shell thickness t, solvent fractions x of core and shell) and homogeneous
spheroids, optionally with a log-normal distribution (median 1, width σ) of a scale factor ν applied to some of
the lengths.

    F(q, μ) = (η_c' - η_sh') V_c Φ(q r_c) + (η_sh' - η_sol) V_t Φ(q r_t),   Φ(x) = 3 j1(x)/x,
    r_c = √((A μ)² + B²(1 - μ²)),  r_t = √(((A + t) μ)² + (B + t)²(1 - μ²)),  V_c = 4π/3 A B², V_t = 4π/3 (A+t)(B+t)²,
    η' = η (1 - x) + η_sol x,      P(q) = ∫_0^1 dμ <F²>_ν

| function | scaled by ν | parameters |
|---|---|---|
| ``ff_ellipsoidal_shell_0`` | – | R_polar, R_equatorial, dummy, t, dummy, η_core, η_shell, η_solv, x_core, x_shell |
| ``ff_ellipsoidal_shell_0t`` | t | ..., t, sigma, ... |
| ``ff_ellipsoidal_shell_rp`` | R_p | |
| ``ff_ellipsoidal_shell_rp_t`` | R_p, t | |
| ``ff_ellipsoidal_shell_re`` | R_e | |
| ``ff_ellipsoidal_shell_re_t`` | R_e, t | |
| ``ff_ellipsoidal_shell_2`` | R_p, R_e | |
| ``ff_ellipsoidal_shell_2t`` | R_p, R_e, t | |
| ``ff_spheroid_R`` | – | R_polar, R_equatorial, dummy x3, η_core, dummy, η_solv |
| ``ff_spheroid_V`` | – | V, R_equatorial (R_polar = 3V/(4π R_e²)), ... |
| ``ff_spheroid_nu`` | – | ν, R_equatorial (R_polar = ν R_e), ... |

Amplitudes (``_f``) are <F> with the same averages; volumes (``_v``) the mean outer volume over the distribution.
The μ integral uses Gauss-Legendre on panels of at most 2 rad of phase, the ν integral Gauss-Legendre in ln ν over
[-8σ, 6σ² + 8σ] (exact to rounding). The spheroids and ``ellipsoidal_shell_0`` do not depend on ν: their p[4]
(dummy) has no effect (C integrated the normalised log-normal over a truncated range when it was not 0).

C fixes of 2026-11-24 (ellipsoidal_shell), followed here: ``_re`` and ``_re_t`` scaled the polar instead of the
equatorial radius (their volumes were written for R_e); ``ellipsoidal_shell_re_v`` had wrong coefficients;
``ellipsoidal_shell_2t_v`` returned 0; ``ellipsoidal_shell_0_v`` returned the core volume (the others the outer
volume); ``spheroid_V`` put R_equatorial into the polar axis and computed the equatorial one from V
(volume ≠ V: I(0) was not (Δη V)²).
"""

from __future__ import annotations

import math

import numpy as np
from scipy import special

from .common import make_spec
from .quadrature import gauss_legendre

__all__ = ["SPECS"]

_GN = 16


def _phi(x):
    """3 j1(x)/x (1 at 0)."""
    x = np.asarray(x, float)
    small = np.abs(x) < 1e-3
    xs = np.where(small, 1.0, x)
    x2 = x * x
    return np.where(small, 1.0 - x2 / 10.0 + x2 * x2 / 280.0, 3.0 * special.spherical_jn(1, xs) / xs)


def _nu_axis(sigma, phase):
    """ν nodes and weights of ∫ LN(ν; 1, σ) dν (σ = 0: ν = 1)."""
    if sigma == 0:
        return np.ones(1), np.ones(1)
    lo, hi = -8.0 * sigma, 6.0 * sigma * sigma + 8.0 * sigma
    n = max(int(math.ceil((hi - lo) / sigma)), math.ceil(phase * (math.exp(hi) - math.exp(lo)) / 5.0), 8)
    n = min(n, 4000)
    e = np.linspace(lo, hi, n + 1)
    t, w = gauss_legendre(_GN, e[:-1], e[1:])
    t, w = t.ravel(), w.ravel()
    w = w * np.exp(-0.5 * (t / sigma) ** 2) / (math.sqrt(2.0 * math.pi) * sigma)
    return np.exp(t), w


def _avg(q, A0, B0, t0, sigma, scale, eta, power):
    """∫_0^1 dμ <F^power>_ν; ``scale`` = (A, B, t) flags scaled by ν; eta = (η_c', η_sh', η_sol)."""
    q = np.atleast_1d(np.asarray(q, float))
    ec, es, ev = eta
    sa, sb, st = scale
    out = np.empty(q.shape)
    size = max(A0, B0) + t0
    for i, qi in enumerate(q):
        nu, wn = _nu_axis(sigma if any(scale) else 0.0, qi * size)
        mu_size = qi * size * float(np.max(nu))
        m, wm = gauss_legendre(_GN, *_edges(0.0, 1.0, mu_size))
        m, wm = m.ravel()[:, None], wm.ravel()
        A = A0 * (nu if sa else 1.0)
        B = B0 * (nu if sb else 1.0)
        T = t0 * (nu if st else 1.0)
        rc = np.sqrt((A * m) ** 2 + B ** 2 * (1.0 - m * m))
        rt = np.sqrt(((A + T) * m) ** 2 + (B + T) ** 2 * (1.0 - m * m))
        Vc = 4.0 / 3.0 * math.pi * A * B * B
        Vt = 4.0 / 3.0 * math.pi * (A + T) * (B + T) ** 2
        F = (ec - es) * Vc * _phi(qi * rc) + (es - ev) * Vt * _phi(qi * rt)              # (μ, ν)
        out[i] = wm @ (F ** power) @ wn
    return out


def _edges(a, b, phase):
    n = max(2, math.ceil(phase / 1.0))
    e = np.linspace(a, b, n + 1)
    return e[:-1], e[1:]


def _etas(p):
    ev = p[7]
    return (p[5] * (1.0 - p[8]) + ev * p[8], p[6] * (1.0 - p[9]) + ev * p[9], ev)


def _check(p, name):
    if p[0] < 0 or p[1] < 0 or p[3] < 0 or p[4] < 0:
        raise ValueError(f"{name}: R_polar, R_equatorial, t >= 0 and sigma >= 0 required")


def _shell(scale, name):
    def make(power):
        def f(q, p):
            _check(p, name)
            sig = abs(p[4]) if any(scale) else 0.0
            return _avg(q, abs(p[0]), abs(p[1]), abs(p[3]), sig, scale, _etas(p), power).reshape(np.shape(q))
        return f
    return make(2), make(1)


ellipsoidal_shell_0, ellipsoidal_shell_0_f = _shell((False, False, False), "ellipsoidal_shell_0")
ellipsoidal_shell_0t, ellipsoidal_shell_0t_f = _shell((False, False, True), "ellipsoidal_shell_0t")
ellipsoidal_shell_rp, ellipsoidal_shell_rp_f = _shell((True, False, False), "ellipsoidal_shell_rp")
ellipsoidal_shell_rp_t, ellipsoidal_shell_rp_t_f = _shell((True, False, True), "ellipsoidal_shell_rp_t")
ellipsoidal_shell_re, ellipsoidal_shell_re_f = _shell((False, True, False), "ellipsoidal_shell_re")
ellipsoidal_shell_re_t, ellipsoidal_shell_re_t_f = _shell((False, True, True), "ellipsoidal_shell_re_t")
ellipsoidal_shell_2, ellipsoidal_shell_2_f = _shell((True, True, False), "ellipsoidal_shell_2")
ellipsoidal_shell_2t, ellipsoidal_shell_2t_f = _shell((True, True, True), "ellipsoidal_shell_2t")


def _shell_vol(scale):
    """Mean outer volume 4π/3 <(A + t)(B + t)²> over ν (<ν^k> = exp(k² σ²/2))."""
    sa, sb, st = scale

    def v(x, p, dist=0):
        pp = np.zeros(max(50, len(p)))
        pp[: len(p)] = p
        A = np.abs(np.where(dist == 0, x, pp[0]))
        B = np.abs(np.where(dist == 1, x, pp[1]))
        T = np.abs(np.where(dist == 3, x, pp[3]))
        s = np.abs(np.where(dist == 4, x, pp[4])) if any(scale) else 0.0
        m = lambda k: np.exp(0.5 * s * s * k * k)  # noqa: E731
        # expand (A ν^a + T ν^c)(B ν^b + T ν^c)², a, b, c in {0, 1}
        a, b, c = int(sa), int(sb), int(st)
        terms = [(A * B * B, a + 2 * b), (2 * A * B * T, a + b + c), (A * T * T, a + 2 * c),
                 (T * B * B, c + 2 * b), (2 * T * B * T, b + 2 * c), (T ** 3, 3 * c)]
        return 4.0 / 3.0 * math.pi * sum(coef * m(k) for coef, k in terms)
    return v


def _spheroid(kind):
    def make(power):
        def f(q, p):
            if kind == "R":
                A, B = p[0], p[1]
            elif kind == "V":
                B = p[1]
                A = 3.0 * p[0] / (4.0 * math.pi * B * B) if B != 0 else 0.0
            else:
                B = p[1]
                A = p[0] * B
            if A < 0 or B < 0:
                raise ValueError(f"spheroid_{kind}: lengths >= 0 required")
            return _avg(q, A, B, 0.0, 0.0, (False, False, False), (p[5], p[5], p[7]), power).reshape(np.shape(q))
        return f
    return make(2), make(1)


spheroid_R, spheroid_R_f = _spheroid("R")
spheroid_V, spheroid_V_f = _spheroid("V")
spheroid_nu, spheroid_nu_f = _spheroid("nu")


def _vol_spheroid(kind):
    def v(x, p, dist=0):
        x = np.asarray(x, float)
        a = np.where(dist == 0, x, p[0])
        b = np.where(dist == 1, x, p[1])
        if kind == "R":
            return 4.0 / 3.0 * math.pi * a * b * b
        if kind == "V":
            return a + 0.0 * b
        return 4.0 / 3.0 * math.pi * b ** 3 * a
    return v


# ------------------------------------------------------------------------------------------------ specs
_SRC = "plugins/ellipsoidal_shell/sasfit_ff_"
_GROUP = "ff_plugins_ellipsoidal_obj"
_ES = ["R_polar", "R_equatorial", "dummy", "t", "sigma", "eta_core", "eta_shell", "eta_solv", "x_core", "x_shell"]
_ES0 = ["R_polar", "R_equatorial", "dummy", "t", "dummy", "eta_core", "eta_shell", "eta_solv", "x_core", "x_shell"]
_ESS = {0: (2.0, 30.0, "log"), 1: (2.0, 30.0, "log"), 3: (0.5, 5.0, "log"), 4: (0.0, 0.3, "lin"),
        5: (-1.0, 3.0, "lin"), 6: (0.0, 2.0, "lin"), 7: (-0.5, 0.5, "lin"), 8: (0.0, 0.5, "lin"), 9: (0.0, 0.5, "lin")}
_ESS0 = {k: v for k, v in _ESS.items() if k != 4}
_SP = ["R_polar", "R_equatorial", "dummy", "dummy", "dummy", "eta_core", "dummy", "eta_solv"]
_SPS = {0: (2.0, 40.0, "log"), 1: (2.0, 40.0, "log"), 5: (-1.0, 3.0, "lin"), 7: (-0.5, 0.5, "lin")}


def _red(q, p):
    return np.asarray(q, float) * (max(abs(p[0]), abs(p[1])) + abs(p[3]))


def _shell_spec(name, f, fa, scale, pars, sample):
    return make_spec(name, f, pars, sample, amplitude=fa, volume=_shell_vol(scale), source=_SRC + name[3:] + ".c",
                     reduced=_red, reduced_label="q (R_max + t)", rtol=1e-10, group=_GROUP)


SPECS = [
    _shell_spec("ff_ellipsoidal_shell_0", ellipsoidal_shell_0, ellipsoidal_shell_0_f, (False,) * 3, _ES0, _ESS0),
    _shell_spec("ff_ellipsoidal_shell_0t", ellipsoidal_shell_0t, ellipsoidal_shell_0t_f, (False, False, True), _ES, _ESS),
    _shell_spec("ff_ellipsoidal_shell_rp", ellipsoidal_shell_rp, ellipsoidal_shell_rp_f, (True, False, False), _ES, _ESS),
    _shell_spec("ff_ellipsoidal_shell_rp_t", ellipsoidal_shell_rp_t, ellipsoidal_shell_rp_t_f, (True, False, True), _ES,
                _ESS),
    _shell_spec("ff_ellipsoidal_shell_re", ellipsoidal_shell_re, ellipsoidal_shell_re_f, (False, True, False), _ES, _ESS),
    _shell_spec("ff_ellipsoidal_shell_re_t", ellipsoidal_shell_re_t, ellipsoidal_shell_re_t_f, (False, True, True), _ES,
                _ESS),
    _shell_spec("ff_ellipsoidal_shell_2", ellipsoidal_shell_2, ellipsoidal_shell_2_f, (True, True, False), _ES, _ESS),
    _shell_spec("ff_ellipsoidal_shell_2t", ellipsoidal_shell_2t, ellipsoidal_shell_2t_f, (True, True, True), _ES, _ESS),
    make_spec("ff_spheroid_R", spheroid_R, _SP, _SPS, amplitude=spheroid_R_f, volume=_vol_spheroid("R"),
              source=_SRC + "spheroid_R.c", reduced=lambda q, p: np.asarray(q) * max(abs(p[0]), abs(p[1])),
              reduced_label="q R_max", rtol=1e-10, group=_GROUP),
    make_spec("ff_spheroid_V", spheroid_V, ["V"] + _SP[1:], {0: (50.0, 5e4, "log"), 1: (2.0, 30.0, "log"),
                                                               5: (-1.0, 3.0, "lin"), 7: (-0.5, 0.5, "lin")},
              amplitude=spheroid_V_f, volume=_vol_spheroid("V"), source=_SRC + "spheroid_V.c",
              reduced=lambda q, p: np.asarray(q) * max(abs(p[1]), 3 * abs(p[0]) / (4 * math.pi * p[1] ** 2)),
              reduced_label="q R_max", rtol=1e-10, group=_GROUP),
    make_spec("ff_spheroid_nu", spheroid_nu, ["nu"] + _SP[1:], {0: (0.2, 5.0, "log"), 1: (2.0, 30.0, "log"),
                                                                 5: (-1.0, 3.0, "lin"), 7: (-0.5, 0.5, "lin")},
              amplitude=spheroid_nu_f, volume=_vol_spheroid("nu"), source=_SRC + "spheroid_nu.c",
              reduced=lambda q, p: np.asarray(q) * abs(p[1]) * max(1.0, abs(p[0])), reduced_label="q R_max",
              rtol=1e-10, group=_GROUP),
]
