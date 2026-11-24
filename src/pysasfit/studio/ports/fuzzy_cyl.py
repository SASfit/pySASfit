"""Python ports of the SASfit plugin pcs_fuzzy_cyl (cross-sections P_cs(q) of cylinders with radial profiles).

All P_cs are 2D: F(q) = 2π ∫ Δη(r) r J0(qr) dr, P_cs = F², log-normal radius R (median R0, width σ_R) averaged by
Gauss-Legendre in ln R (as ``thin_objects.lognorm_average``; C truncates the range with find_LogNorm_int_range).

* Boucher: Δη(r) = Δη (1 - r²/R²)^{α/2-2} (α > 2) → F = Δη 2πR² 0F1(α/2; -(qR)²/4)/(α - 2); Boucher2 has the
  profile scaled by (α/2 - 1) → F = Δη πR² 0F1(α/2; -(qR)²/4).
* fuzzy cylinder: disc ⊗ 2D Gaussian (σ): F = Δη πR² 2J1(qR)/(qR) e^{-q²σ²/2}; its profile is the exact convolution
  η_sol + Δη ∫_0^R (s/σ²) e^{-(r-s)²/2σ²} I0e(rs/σ²) ds.
* linear shell: core R (η_c) plus a shell of thickness DR whose SLD goes linearly from η_sh,i to η_sh,o; the core in
  closed form, the shell by Gauss-Legendre (panels ≤ 2 rad) (C: closed form with Struve functions / (DR q²),
  which stays accurate at small q).
* rc_exp_sh_cyl (doi:10.1021/ma102624b): core R_c (η_c) and an exponential shell of thickness t with a Fermi cut-off
  of width σ_s, truncated at R_c + t; F by the closed core term plus Gauss-Legendre over the shell.

C fixes 2026-11-24: boucher_cyl_f did not set q (it used a stale q, 0 for a fresh parameter block); the Boucher
profiles used integer abs(r) (r truncated to an int); profile_fuzzy_cyl read σ from
p[2] instead of p[3], had the unparenthesised DETA = ETA_CYL-ETA_SOL (η_sol dropped), multiplied the Hankel inverse by
Δη a second time and clobbered r (param->p[MAXPAR-1]) by calling fuzzy_cyl_f; it now integrates the exact convolution
for every σ (it used an erf planar approximation for σ <= R/10); profile_linsh_cyl used the indices p[3]..p[6] for the
SLDs while the header (and linsh_cyl) have p[5]..p[8]; linsh_cyl_v was 0 (now π(R+DR)²); rc_exp_sh_cyl_v replaced
Rc for dist = 1 and t for dist = 2 (now 0 and 1, the parameter indices); rc_exp_sh_cyl_f integrated over the SLD
step at Rc in one adaptive integral (up to 1e-5 relative error; now split at Rc).
The volume functions return the cross-section area (C convention of this plugin); in the ports x replaces the
distributed parameter (C boucher/fuzzy _v ignore it).
"""

from __future__ import annotations

import math

import numpy as np
from scipy import special

from .common import make_spec
from .quadrature import gauss_legendre
from .thin_objects import lognorm_average

__all__ = ["SPECS"]

_GN = 16


def _jinc2(u):
    """2 J1(u)/u, stable at u -> 0."""
    u = np.asarray(u, float)
    small = np.abs(u) < 1e-4
    us = np.where(small, 1.0, u)
    return np.where(small, 1.0 - u * u / 8.0, 2.0 * special.j1(us) / us)


def _power(f, power):
    return f if power == 1 else f**power


def _avg(q, amp, p, power, wpow):
    """<amp(q, R)^power> over log-normal R (p[0] median, p[1] width)."""
    if p[0] < 0:
        raise ValueError("R >= 0 required")
    f = lambda qi, x: _power(amp(qi, x), power)  # noqa: E731
    return lognorm_average(q, f, p[0], abs(p[1]), wpow).reshape(np.shape(q))


# ---------------------------------------------------------------------------------------------- Boucher
def _boucher_amp(p, half):
    a, deta = p[3], p[7]
    if half:
        return lambda qi, x: deta * math.pi * x * x * special.hyp0f1(0.5 * a, -0.25 * (qi * x) ** 2)
    return lambda qi, x: deta * 2.0 * math.pi * x * x * special.hyp0f1(0.5 * a, -0.25 * (qi * x) ** 2) / (a - 2.0)


def boucher_cyl(q, p):
    """p = [R0, sigma_R0, dummy, alpha, dummy, dummy, dummy, Delta eta] (α > 2)."""
    if p[3] <= 2:
        raise ValueError("boucher_cyl: alpha > 2 required")
    return _avg(q, _boucher_amp(p, False), p, 2, 4.0)


def boucher_cyl_f(q, p):
    if p[3] <= 2:
        raise ValueError("boucher_cyl: alpha > 2 required")
    return _avg(q, _boucher_amp(p, False), p, 1, 2.0)


def boucher2_cyl(q, p):
    """Same parameters; profile scaled by (α/2 - 1) (α > 0)."""
    if p[3] <= 0:
        raise ValueError("boucher2_cyl: alpha > 0 required")
    return _avg(q, _boucher_amp(p, True), p, 2, 4.0)


def boucher2_cyl_f(q, p):
    if p[3] <= 0:
        raise ValueError("boucher2_cyl: alpha > 0 required")
    return _avg(q, _boucher_amp(p, True), p, 1, 2.0)


def cyl_area_v(x, p, dist):
    """Cross-section area π R0² (x replaces R0 for dist = 0)."""
    x = np.asarray(x, float)
    return math.pi * (x if dist == 0 else p[0] + 0.0 * x) ** 2


def _boucher_profile(r, p, scale):
    if p[0] < 0 or p[3] <= 2:
        raise ValueError("profile_boucher: R >= 0 and alpha > 2 required")
    r = np.abs(np.asarray(r, float))
    R, a, deta = p[0], p[3], p[7]
    inside = r < R
    x = np.where(inside, r / R if R > 0 else 0.0, 0.0)
    return np.where(inside, deta * (1.0 - x * x) ** (0.5 * a - 2.0) * scale(a), 0.0)


def profile_boucher_cyl(r, p):
    """p = [R, dummy, dummy, alpha, dummy, dummy, dummy, Delta eta]: Δη (1 - r²/R²)^{α/2-2} (0 for |r| > R)."""
    return _boucher_profile(r, p, lambda a: 1.0)


def profile_boucher2_cyl(r, p):
    """As profile_boucher_cyl times (α/2 - 1)."""
    return _boucher_profile(r, p, lambda a: 0.5 * a - 1.0)


# ---------------------------------------------------------------------------------------------- fuzzy cylinder
def _fuzzy_amp(p):
    s, deta = abs(p[3]), p[7] - p[8]
    return lambda qi, x: deta * math.pi * x * x * _jinc2(qi * x) * math.exp(-0.5 * (qi * s) ** 2)


def fuzzy_cyl(q, p):
    """p = [R0, sigma_R0, dummy, sigma, dummy, dummy, dummy, eta_cyl, eta_sol]."""
    return _avg(q, _fuzzy_amp(p), p, 2, 4.0)


def fuzzy_cyl_f(q, p):
    return _avg(q, _fuzzy_amp(p), p, 1, 2.0)


def profile_fuzzy_cyl(r, p):
    """p = [R, dummy, dummy, sigma, dummy, dummy, dummy, eta_cyl, eta_sol]: disc ⊗ 2D Gaussian."""
    if p[0] < 0:
        raise ValueError("profile_fuzzy_cyl: R >= 0 required")
    r = np.abs(np.atleast_1d(np.asarray(r, float)))
    R, s, ec, es = abs(p[0]), abs(p[3]), p[7], p[8]
    if s == 0:
        return np.where(r <= R, ec, es)
    out = np.empty(r.shape)
    for i, ri in enumerate(r):
        a, b = max(0.0, ri - 12.0 * s), min(R, ri + 12.0 * s)
        if not b > a:
            out[i] = 0.0
            continue
        n = max(1, math.ceil((b - a) / s))
        e = np.linspace(a, b, n + 1)
        x, w = gauss_legendre(_GN, e[:-1], e[1:])
        x, w = x.ravel(), w.ravel()
        out[i] = np.sum(w * x / s**2 * np.exp(-0.5 * ((ri - x) / s) ** 2) * special.i0e(ri * x / s**2))
    return es + (ec - es) * out


# ---------------------------------------------------------------------------------------------- linear shell
def _edges(qi, a, b):
    n = max(1, math.ceil(qi * (b - a) / 2.0))
    return np.linspace(a, b, n + 1)


def _shell_hankel(qi, a, b, rho, edges=None):
    """2π ∫_a^b rho(r) r J0(q r) dr (Gauss-Legendre, panels at most 2 rad, or the given panel edges)."""
    if not b > a:
        return 0.0
    e = _edges(qi, a, b) if edges is None else edges
    x, w = gauss_legendre(_GN, e[:-1], e[1:])
    x, w = x.ravel(), w.ravel()
    return 2.0 * math.pi * float(np.sum(w * rho(x) * x * special.j0(qi * x)))


def _linsh_amp(p):
    dr, ec, ei, eo, es = p[2], p[5], p[6], p[7], p[8]

    def amp(qi, x):
        x = np.atleast_1d(x)
        out = np.empty(x.shape)
        for k, xk in enumerate(x):
            core = (ec - es) * math.pi * xk * xk * float(_jinc2(qi * xk))
            if dr == 0:
                out[k] = core
            else:
                m = (eo - ei) / dr
                out[k] = core + _shell_hankel(qi, xk, xk + dr, lambda r: ei + m * (r - xk) - es)
        return out

    return amp


def _linsh_check(p):
    if p[0] < 0 or p[2] < 0:
        raise ValueError("linsh_cyl: R >= 0 and DR >= 0 required")


def linsh_cyl(q, p):
    """p = [R0, sigma_R0, DR, dummy, dummy, eta_c, eta_sh_i, eta_sh_o, eta_sol]."""
    _linsh_check(p)
    return _avg(q, _linsh_amp(p), p, 2, 4.0)


def linsh_cyl_f(q, p):
    _linsh_check(p)
    return _avg(q, _linsh_amp(p), p, 1, 2.0)


def linsh_cyl_v(x, p, dist):
    """π (R0 + DR)² (x replaces R0 for dist = 0, DR for dist = 2; C before 2026-11-24: 0)."""
    x = np.asarray(x, float)
    if dist == 0:
        return math.pi * (x + p[2]) ** 2
    if dist == 2:
        return math.pi * (p[0] + x) ** 2
    return math.pi * (p[0] + p[2]) ** 2 + 0.0 * x


def profile_linsh_cyl(r, p):
    """Same parameters as linsh_cyl (σ_R unused)."""
    if p[0] < 0 or p[2] < 0:
        raise ValueError("profile_linsh_cyl: R >= 0 and DR >= 0 required")
    r = np.abs(np.asarray(r, float))
    R, dr, ec, ei, eo, es = p[0], p[2], p[5], p[6], p[7], p[8]
    if dr == 0:
        return np.where(r < R, ec, es)
    m = (eo - ei) / dr
    return np.where(r < R, ec, np.where(r < R + dr, ei + m * (r - R), es))


# ---------------------------------------------------------------------------------------------- rc_exp_sh_cyl
def _rc_shell_profile(r, p):
    """Profile on R_c <= r <= R_c + t (vectorised)."""
    rc, t, alpha, ss, ei, eo = p[0], p[1], p[2], p[3], p[5], p[6]
    x = (r - rc) / t
    if alpha < 0:
        prof = ei + (eo - ei) * x * np.exp((1.0 - x) * alpha)
    else:
        prof = eo + (ei - eo) * (1.0 - x) * np.exp(-x * alpha)
    if ss != 0:
        prof = prof / (1.0 + np.exp((r - rc - t) / ss))
    return prof


def rc_exp_sh_cyl_profile(r, p):
    """p = [Rc, t, alpha, sigma_s, eta_c, eta_sh_in, eta_sh_out, eta_solv]."""
    if p[0] < 0 or p[1] < 0:
        raise ValueError("rc_exp_sh_cyl: Rc >= 0 and t >= 0 required")
    r = np.abs(np.asarray(r, float))
    rc, t = p[0], p[1]
    inner = r < rc
    shell = (~inner) & (r <= rc + t)
    with np.errstate(divide="ignore", invalid="ignore"):
        sh = _rc_shell_profile(np.where(shell, r, rc + 0.5 * t), p) if t > 0 else np.zeros(r.shape)
    return np.where(inner, p[4], np.where(shell, sh, p[7]))


def rc_exp_sh_cyl_f(q, p):
    if p[0] < 0 or p[1] < 0:
        raise ValueError("rc_exp_sh_cyl: Rc >= 0 and t >= 0 required")
    q = np.atleast_1d(np.asarray(q, float))
    rc, t, ec, es = p[0], p[1], p[4], p[7]
    out = np.empty(q.shape)
    for i, qi in enumerate(q):
        core = (ec - es) * math.pi * rc * rc * float(_jinc2(qi * rc))
        sh = 0.0
        if t > 0:
            # the Fermi cut-off (width σ_s) at the outer edge: panels of at most σ_s over its last 30 σ_s
            ss = abs(p[3])
            e = _edges(qi, rc, rc + t)
            if 0 < ss < t / 8.0:
                m = max(rc, rc + t - 30.0 * ss)
                e = np.unique(np.concatenate([e[e < m], [m], np.linspace(m, rc + t, math.ceil((rc + t - m) / ss) + 1)]))
            sh = _shell_hankel(qi, rc, rc + t, lambda r: _rc_shell_profile(r, p) - es, edges=e)
        out[i] = core + sh
    return out.reshape(np.shape(np.asarray(q)))


def rc_exp_sh_cyl(q, p):
    return rc_exp_sh_cyl_f(q, p) ** 2


def rc_exp_sh_cyl_v(x, p, dist):
    """π (Rc + t)², with x replacing Rc (dist = 0) or t (dist = 1) (C before 2026-11-24: shifted by one index)."""
    x = np.asarray(x, float)
    if dist == 0:
        return math.pi * (x + p[1]) ** 2
    if dist == 1:
        return math.pi * (p[0] + x) ** 2
    return math.pi * (p[0] + p[1]) ** 2 + 0.0 * x


# ---------------------------------------------------------------------------------------------- specs
_SRC = "plugins/pcs_fuzzy_cyl/sasfit_ff_"
_G = "ff_plugins_pcslocalcylindrical"
_B = ["R0", "sigma_R0", "dummy", "alpha", "dummy", "dummy", "dummy", "Delta eta"]
_BS = {0: (5.0, 50.0, "log"), 1: (0.0, 0.2, "lin"), 3: (2.5, 8.0, "lin"), 7: (0.5, 2.0, "lin")}
_BP = ["R", "dummy", "dummy", "alpha", "dummy", "dummy", "dummy", "Delta eta"]
_BPS = {0: (5.0, 50.0, "log"), 3: (4.5, 8.0, "lin"), 7: (0.5, 2.0, "lin")}
_F = ["R0", "sigma_R0", "dummy", "sigma", "dummy", "dummy", "dummy", "eta_cyl", "eta_sol"]
_FS = {0: (5.0, 50.0, "log"), 1: (0.0, 0.2, "lin"), 3: (0.5, 5.0, "lin"), 7: (0.5, 2.0, "lin"), 8: (0.0, 0.3, "lin")}
_FP = ["R", "dummy", "dummy", "sigma", "dummy", "dummy", "dummy", "eta_cyl", "eta_sol"]
_FPS = {0: (5.0, 50.0, "log"), 3: (0.5, 5.0, "lin"), 7: (0.5, 2.0, "lin"), 8: (0.0, 0.3, "lin")}
_L = ["R0", "sigma_R0", "t", "dummy", "dummy", "eta_c", "eta_sh_i", "eta_sh_o", "eta_sol"]
_LS = {0: (5.0, 50.0, "log"), 1: (0.0, 0.2, "lin"), 2: (1.0, 20.0, "lin"), 5: (0.5, 2.0, "lin"), 6: (0.5, 2.0, "lin"),
       7: (0.0, 1.0, "lin"), 8: (0.0, 0.3, "lin")}  # fmt: skip
_LP = ["R0", "dummy"] + _L[2:]
_LPS = {k: v for k, v in _LS.items() if k != 1}
_RC = ["Rc", "t", "alpha", "sigma_s", "eta_c", "eta_sh_in", "eta_sh_out", "eta_solv"]
_RCS = {0: (5.0, 50.0, "log"), 1: (2.0, 30.0, "lin"), 2: (-4.0, 4.0, "lin"), 3: (0.1, 3.0, "lin"), 4: (0.5, 2.0, "lin"),
        5: (0.5, 2.0, "lin"), 6: (0.0, 1.0, "lin"), 7: (0.0, 0.3, "lin")}  # fmt: skip


def _qr(i):
    return lambda q, p: np.asarray(q, float) * abs(p[i])


SPECS = [
    make_spec("ff_boucher_cyl", boucher_cyl, _B, _BS, amplitude=boucher_cyl_f, volume=cyl_area_v,
              source=_SRC + "boucher_cyl.c", reduced=_qr(0), reduced_label="q R0", rtol=1e-8, c_integrates=True, group=_G),
    make_spec("ff_profile_boucher_cyl", profile_boucher_cyl, _BP, _BPS, source=_SRC + "profile_boucher_cyl.c",
              rtol=1e-13, group=_G),
    make_spec("ff_boucher2_cyl", boucher2_cyl, _B, _BS, amplitude=boucher2_cyl_f, volume=cyl_area_v,
              source=_SRC + "boucher2_cyl.c", reduced=_qr(0), reduced_label="q R0", rtol=1e-8, c_integrates=True,
              group=_G),
    make_spec("ff_profile_boucher2_cyl", profile_boucher2_cyl, _BP, _BPS, source=_SRC + "profile_boucher2_cyl.c",
              rtol=1e-13, group=_G),
    make_spec("ff_fuzzy_cyl", fuzzy_cyl, _F, _FS, amplitude=fuzzy_cyl_f, volume=cyl_area_v, source=_SRC + "fuzzy_cyl.c",
              reduced=_qr(0), reduced_label="q R0", rtol=1e-8, c_integrates=True, group=_G),
    make_spec("ff_profile_fuzzy_cyl", profile_fuzzy_cyl, _FP, _FPS, source=_SRC + "profile_fuzzy_cyl.c", rtol=1e-8,
              c_integrates=True, group=_G),
    make_spec("ff_linsh_cyl", linsh_cyl, _L, _LS, amplitude=linsh_cyl_f, volume=linsh_cyl_v, source=_SRC + "linsh_cyl.c",
              reduced=_qr(0), reduced_label="q R0", rtol=1e-8, c_integrates=True, group=_G),
    make_spec("ff_profile_linsh_cyl", profile_linsh_cyl, _LP, _LPS, source=_SRC + "profile_linsh_cyl.c", rtol=1e-13,
              group=_G),
    make_spec("ff_rc_exp_sh_cyl", rc_exp_sh_cyl, _RC, _RCS, amplitude=rc_exp_sh_cyl_f, volume=rc_exp_sh_cyl_v,
              source=_SRC + "rc_exp_sh_cyl.c", reduced=_qr(0), reduced_label="q Rc", rtol=1e-8, c_integrates=True,
              group="ff_plugins_fuzzy_cyl"),
    make_spec("ff_rc_exp_sh_cyl_profile", rc_exp_sh_cyl_profile, _RC, _RCS, source=_SRC + "rc_exp_sh_cyl_profile.c",
              rtol=1e-13, group="ff_plugins_fuzzy_profiles"),
]  # fmt: skip
