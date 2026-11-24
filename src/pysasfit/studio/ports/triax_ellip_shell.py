"""Python ports of the SASfit plugin triax_ellip_shell (batch 8, 2026-11-24): randomly oriented core-shell triaxial
ellipsoids (semi-axes a, b, c of the core, shell thickness t), optionally with a log-normal distribution (median 1,
width σ) of a scale factor ν applied to some of the lengths.

    F(q; φ, μ) = (η_c - η_sh) V_c Φ(q r_c) + (η_sh - η_sol) V_t Φ(q r_t),   Φ(x) = 3 j1(x)/x,
    r_c² = (a² cos²φ + b² sin²φ)(1 - μ²) + c² μ²,   r_t: a, b, c -> a + t, b + t, c + t,
    V_c = 4π/3 a b c,  V_t = 4π/3 (a + t)(b + t)(c + t),
    P(q) = ∫_0^1 dx ∫_0^1 dμ <F²>_ν   (φ = x π/2)

| function | scaled by ν |
|---|---|
| ``ff_triax_ellip_shell``, ``ff_triax_ellip_shell_t`` | t |
| ``ff_triax_ellip_shell_1`` | a |
| ``ff_triax_ellip_shell_1t`` | a, t |
| ``ff_triax_ellip_shell_2`` | a, b |
| ``ff_triax_ellip_shell_2t`` | a, b, t |
| ``ff_triax_ellip_shell_3`` | a, b, c |
| ``ff_triax_ellip_shell_3t`` | a, b, c, t |

Parameters: a, b, c, t, sigma, eta_c, eta_sh, eta_sol. Amplitudes (``_f``) are <F> with the same averages;
volumes (``_v``) the mean outer volume 4π/3 <(a + t)(b + t)(c + t)> over ν (<ν^k> = exp(k²σ²/2)), with the
distributed parameter (``dist`` = 0 … 4) replaced by x.

``ff_triax_ellip_shell_syl``: a, delta_b, delta_c, t, eta_c, eta_sh_dry, eta_sol, phi, v_pol_ap; b = a + Δb,
c = b + Δc, the shell contains the polymer volume v_pol_ap · V_core (solvent fraction w = 1 - v V_c/(V_t - V_c),
η_sh = (1 - w) η_sh,dry + w η_sol) and the result is the macroscopic cross section n P(q) with
n = φ / ((1 + v) V_c).

The φ and μ integrals use 16-point Gauss-Legendre on panels of at most 6 rad of phase (quadrature error of a
sinusoid < 1e-20), the ν integral Gauss-Legendre in ln ν over [-8σ, 6σ² + 8σ] on panels of at most 4σ and 5 rad
(exact to rounding; checked against scipy nquad).

C fixes of 2026-11-24 (triax_ellip_shell), followed here: ``triax_ellip_shell`` and its ``_f`` read the third
cubature coordinate also for σ = 0 (2-dimensional cubature: ν from beyond the coordinate array) and integrated over
ν without the log-normal weight for σ > 0 — they now use the same kernel as ``_t``; the volume functions used the
ν of the last cubature point (the axis macros contain ν), ignored the distributed parameter, and ``_t_v`` had the
formula of ``_1`` (a scaled instead of t).
Not ported: ``carved_triax_ellip_shell`` (undocumented work in progress), ``triax_ellip_shell_sq_baba_ahmed_1/2``
(need the Baba-Ahmed structure factor of another plugin).
"""

from __future__ import annotations

import math

import numpy as np

from .common import make_spec
from .ellipsoidal_shell import _phi
from .quadrature import gauss_legendre

__all__ = ["SPECS", "triax_average", "triax_volume"]

_GN = 16


def _panels(a, b, phase, per=1.0):
    n = max(2, math.ceil(phase / per))
    e = np.linspace(a, b, n + 1)
    t, w = gauss_legendre(_GN, e[:-1], e[1:])
    return t.ravel(), w.ravel()


def _nu_axis_triax(sigma, phase):
    """ν nodes and weights of ∫ LN(ν; 1, σ) dν: 16-point Gauss-Legendre in t = ln ν on [-8σ, 8σ + 6σ²], panels at
    most 4σ wide and at most 5 rad of ``phase``·ν (the Gaussian weight is resolved to rounding by 16 nodes per 4σ)."""
    if sigma == 0:
        return np.ones(1), np.ones(1)
    lo, hi = -8.0 * sigma, 6.0 * sigma * sigma + 8.0 * sigma
    n = max(int(math.ceil((hi - lo) / (4.0 * sigma))), math.ceil(phase * (math.exp(hi) - math.exp(lo)) / 5.0))
    n = min(n, 4000)
    e = np.linspace(lo, hi, n + 1)
    t, w = gauss_legendre(_GN, e[:-1], e[1:])
    t, w = t.ravel(), w.ravel()
    w = w * np.exp(-0.5 * (t / sigma) ** 2) / (math.sqrt(2.0 * math.pi) * sigma)
    return np.exp(t), w


def triax_average(q, a0, b0, c0, t0, sigma, scale, eta, power):
    """∫_0^1 dx ∫_0^1 dμ <F^power>_ν; ``scale`` = flags (a, b, c, t) scaled by ν; eta = (η_c, η_sh, η_sol)."""
    q = np.atleast_1d(np.asarray(q, float))
    ec, es, ev = eta
    sa, sb, sc, st = scale
    size = max(a0, b0, c0) + t0
    out = np.empty(q.shape)
    for i, qi in enumerate(q):
        nu, wn = _nu_axis_triax(sigma if any(scale) else 0.0, qi * size)
        numax = float(np.max(nu))
        x, wx = _panels(0.0, 1.0, qi * abs(a0 - b0) * numax, 6.0)
        m, wm = _panels(0.0, 1.0, qi * size * numax, 6.0)
        M, N = m[:, None], nu[None, :]
        A = a0 * (N if sa else 1.0)
        B = b0 * (N if sb else 1.0)
        C = c0 * (N if sc else 1.0)
        T = t0 * (N if st else 1.0)
        Vc = 4.0 / 3.0 * math.pi * A * B * C
        Vt = 4.0 / 3.0 * math.pi * (A + T) * (B + T) * (C + T)
        s_m = 1.0 - M * M
        cz, czt = (C * M) ** 2, ((C + T) * M) ** 2
        acc = 0.0
        for xk, wk in zip(x, wx):          # one φ node at a time: arrays (μ, ν)
            cph, sph = math.cos(xk * math.pi / 2.0) ** 2, math.sin(xk * math.pi / 2.0) ** 2
            rc = np.sqrt((A * A * cph + B * B * sph) * s_m + cz)
            rt = np.sqrt(((A + T) ** 2 * cph + (B + T) ** 2 * sph) * s_m + czt)
            F = (ec - es) * Vc * _phi(qi * rc) + (es - ev) * Vt * _phi(qi * rt)
            acc += wk * (wm @ (F ** power) @ wn)
        out[i] = acc
    return out


def triax_volume(a, b, c, t, sigma, scale):
    """4π/3 <(a ν^ka + t ν^kt)(b ν^kb + t ν^kt)(c ν^kc + t ν^kt)>, <ν^k> = exp(k²σ²/2)."""
    ka, kb, kc, kt = (int(bool(s)) for s in scale)
    m = lambda k: np.exp(0.5 * sigma * sigma * k * k)  # noqa: E731
    total = 0.0
    for fa in ((a, ka), (t, kt)):
        for fb in ((b, kb), (t, kt)):
            for fc in ((c, kc), (t, kt)):
                total = total + fa[0] * fb[0] * fc[0] * m(fa[1] + fb[1] + fc[1])
    return 4.0 / 3.0 * math.pi * total


def _check(p, name):
    if min(p[0], p[1], p[2], p[3], p[4]) < 0:
        raise ValueError(f"{name}: a, b, c, t >= 0 and sigma >= 0 required")


def _family(scale, name):
    def make(power):
        def f(q, p):
            p = np.asarray(p, float)
            _check(p, name)
            return triax_average(q, p[0], p[1], p[2], p[3], p[4], scale, (p[5], p[6], p[7]), power).reshape(np.shape(q))
        f.__name__ = name + ("" if power == 2 else "_f")
        return f

    def vol(x, p, dist=0):
        x = np.asarray(x, float)
        pp = np.zeros(max(50, len(p)))
        pp[: len(p)] = p
        v = [np.where(dist == k, x, pp[k]) for k in range(5)]
        return triax_volume(v[0], v[1], v[2], v[3], np.abs(v[4]), scale)

    return make(2), make(1), vol


_SCALES = {"": (0, 0, 0, 1), "_t": (0, 0, 0, 1), "_1": (1, 0, 0, 0), "_1t": (1, 0, 0, 1), "_2": (1, 1, 0, 0),
           "_2t": (1, 1, 0, 1), "_3": (1, 1, 1, 0), "_3t": (1, 1, 1, 1)}


def syl(q, p):
    """p = [a, delta_b, delta_c, t, eta_c, eta_sh_dry, eta_sol, phi, v_pol_ap]."""
    p = np.asarray(p, float)
    a, db, dc, t = p[0], p[1], p[2], p[3]
    if min(a, db, dc, t, p[7], p[8]) < 0:
        raise ValueError("triax_ellip_shell_syl: a, delta_b, delta_c, t, phi, v_pol_ap >= 0 required")
    b = a + db
    c = b + dc
    vc = a * b * c
    w = 1.0 - p[8] * vc / ((a + t) * (b + t) * (c + t) - vc)
    if w < 0:
        raise ValueError("triax_ellip_shell_syl: w < 0 (polymer volume larger than the shell)")
    eta_sh = (1.0 - w) * p[5] + w * p[6]
    n = p[7] / ((1.0 + p[8]) * 4.0 / 3.0 * math.pi * vc)
    return (n * triax_average(q, a, b, c, t, 0.0, (0, 0, 0, 0), (p[4], eta_sh, p[6]), 2)).reshape(np.shape(q))


def syl_v(x, p, dist=0):
    x = np.asarray(x, float)
    a, t = p[0], p[3]
    b = a + p[1]
    c = b + p[2]
    v = np.where(dist == 0, (x + t) * (b + t) * (c + t),
        np.where(dist == 1, (a + t) * (x + t) * (c + t),
        np.where(dist == 2, (a + t) * (b + t) * (x + t),
        np.where(dist == 3, (a + x) * (b + x) * (c + x), (a + t) * (b + t) * (c + t)))))  # fmt: skip
    return 4.0 / 3.0 * math.pi * v


# ------------------------------------------------------------------------------------------------ specs
_SRC = "plugins/triax_ellip_shell/sasfit_ff_"
_GROUP = "ff_plugins_triax_ellshells"
_TP = ["a", "b", "c", "t", "sigma", "eta_c", "eta_sh", "eta_sol"]
_TS = {0: (2.0, 20.0, "log"), 1: (2.0, 20.0, "log"), 2: (2.0, 20.0, "log"), 3: (0.5, 5.0, "log"), 4: (0.0, 0.25, "lin"),
       5: (-1.0, 3.0, "lin"), 6: (0.0, 2.0, "lin"), 7: (-0.5, 0.5, "lin")}


def _red(q, p):
    return np.asarray(q, float) * (max(abs(p[0]), abs(p[1]), abs(p[2])) + abs(p[3]))


SPECS = []
for _suf, _sc in _SCALES.items():
    _nm = "triax_ellip_shell" + _suf
    _i, _a, _v = _family(_sc, _nm)
    globals()[_nm], globals()[_nm + "_f"] = _i, _a
    SPECS.append(make_spec("ff_" + _nm, _i, _TP, _TS, amplitude=_a, volume=_v, source=_SRC + _nm + ".c",
                           reduced=_red, reduced_label="q (max(a, b, c) + t)", rtol=1e-9, group=_GROUP))
SPECS.append(make_spec("ff_triax_ellip_shell_syl", syl,
                       ["a", "delta_b", "delta_c", "t", "eta_c", "eta_sh_dry", "eta_sol", "phi", "v_pol_ap"],
                       {0: (2.0, 15.0, "log"), 1: (0.0, 5.0, "lin"), 2: (0.0, 5.0, "lin"), 3: (2.0, 6.0, "lin"),
                        4: (-1.0, 3.0, "lin"), 5: (0.0, 2.0, "lin"), 6: (-0.5, 0.5, "lin"), 7: (0.01, 0.3, "lin"),
                        8: (0.0, 0.5, "lin")},
                       volume=syl_v, source=_SRC + "triax_ellip_shell_syl.c", reduced=_red,
                       reduced_label="q (c + t)", rtol=1e-9, group=_GROUP))
