"""Python ports of the SASfit plugin fuzzysphere (spheres with smooth or graded radial profiles).

Amplitudes F(q) = 4π ∫ Δη(r) r² sin(qr)/(qr) dr: homogeneous inner spheres in closed form (stable φ(qR)), the graded
linear shells in closed form with the stable φ and χ(x) = ∫_0^1 t³ sinc(xt) dt, the exponential and parabolic zones by
Gauss-Legendre quadrature on panels of at most 2 rad of phase (exact to rounding, no cancellation at small q).
C before 2026-11-23: the closed forms of LinShell/LinShell2/ExpShell/coreshellmicrogel divided by q³ … q⁵ and lost
all digits for small q (fuzzysphere: 1e-5 at q R = 1e-6), LinShell2 had an unparenthesised macro (R1 = R_tot - DR used
as q*R_tot - DR), the LinShell volumes were 0. The fixed C uses the same stable forms (series, Gauss-Legendre).
"""

from __future__ import annotations

import math

import numpy as np
from scipy import special

from .common import make_spec, phi_sphere

__all__ = ["SPECS"]

_GX, _GW = np.polynomial.legendre.leggauss(16)


def _sphere_amp(q, r, contrast):
    """contrast · 4π/3 r³ φ(qr) (r >= 0)."""
    r = max(float(r), 0.0)
    return contrast * 4.0 / 3.0 * math.pi * r**3 * phi_sphere(np.asarray(q, float) * r)


def _zone_amp(q, a, b, rho):
    """4π ∫_a^b rho(r) r² sinc(qr) dr by Gauss-Legendre (rho: contrast profile, vectorised in r)."""
    q = np.atleast_1d(np.asarray(q, float))
    a, b = max(float(a), 0.0), float(b)
    if not b > a:
        return np.zeros(q.shape)
    nsub = max(1, math.ceil(float(np.max(q)) * (b - a) / 2.0))
    e = np.linspace(a, b, nsub + 1)
    c, h = 0.5 * (e[1:] + e[:-1]), 0.5 * (e[1:] - e[:-1])
    r = (c[:, None] + h[:, None] * _GX).ravel()
    w = (h[:, None] * _GW).ravel() * 4.0 * math.pi * r * r * rho(r)
    out = np.empty(q.shape)
    for i0 in range(0, q.size, 256):
        qq = q[i0 : i0 + 256]
        out[i0 : i0 + 256] = np.sinc(qq[:, None] * r[None, :] / np.pi) @ w
    return out


def _solvent_mix(eta_sh, x, eta_sol):
    return (1.0 - x) * eta_sh + x * eta_sol


# ---------------------------------------------------------------------------------------------- fuzzy sphere
def fuzzysphere_f(q, p):
    """p = [R, sigma, eta_sph, eta_sol]: sphere convolved with a Gaussian of width σ: V Δη φ(qR) e^{-q²σ²/2}."""
    q = np.asarray(q, float)
    r, s, e1, e0 = abs(p[0]), abs(p[1]), p[2], p[3]
    return _sphere_amp(q, r, e1 - e0) * np.exp(-0.5 * (q * s) ** 2)


def fuzzysphere(q, p):
    return fuzzysphere_f(q, p) ** 2


def fuzzysphere_v(x, p, dist):
    return 4.0 / 3.0 * math.pi * abs(p[0]) ** 3 + 0.0 * np.asarray(x, float)


def profile_fuzzysphere(r, p):
    """Radial SLD profile of the fuzzy sphere (sphere ⊗ 3D Gaussian)."""
    r = np.abs(np.asarray(r, float))
    R, s, e1, e0 = abs(p[0]), abs(p[1]), p[2], p[3]
    if s == 0:
        return np.where(r <= R, e1, e0)
    sq2 = math.sqrt(2.0) * s
    with np.errstate(divide="ignore", invalid="ignore"):
        gen = (np.exp(-0.5 * ((r + R) / s) ** 2) - np.exp(-0.5 * ((r - R) / s) ** 2)) * s / (
            math.sqrt(2.0 * math.pi) * r
        ) + 0.5 * special.erf((r + R) / sq2) - 0.5 * special.erf((r - R) / sq2)
    at0 = special.erf(R / sq2) - math.exp(-0.5 * (R / s) ** 2) * math.sqrt(2.0 / math.pi) * R / s
    return (e1 - e0) * np.where(r == 0, at0, gen) + e0


# ------------------------------------------------------------------------------------- linear shell profiles
def _linshell_parts(p, total):
    r0, dr = abs(p[0]), abs(p[1])
    ec, esh, xin, xout, es = p[2], p[3], p[4], p[5], p[6]
    rc = max(r0 - dr, 0.0) if total else r0
    ein, eout = _solvent_mix(esh, xin, es), _solvent_mix(esh, xout, es)
    return rc, dr, ec, ein, eout, es


def _chi(x):
    """(2x sin x - (x²-2) cos x - 2)/x⁴ = ∫_0^1 t³ sinc(xt) dt (series below x = 1)."""
    x = np.abs(np.asarray(x, float))
    out = np.empty(x.shape)
    lo = x < 1.0
    xl = x[lo]
    t, s = np.ones(xl.shape), np.full(xl.shape, 0.25)
    for k in range(1, 12):
        t = t * (-xl * xl / ((2.0 * k) * (2.0 * k + 1.0)))
        s = s + t / (2.0 * k + 4.0)
    out[lo] = s
    xh = x[~lo]
    out[~lo] = (2.0 * xh * np.sin(xh) - (xh * xh - 2.0) * np.cos(xh) - 2.0) / xh**4
    return out


def _linshell_f(q, p, total):
    """Core plus the linear shell rho(r) = m r + b: b (V(R_o)φ(qR_o) - V(R)φ(qR)) + 4π m (R_o⁴χ(qR_o) - R⁴χ(qR))."""
    q = np.atleast_1d(np.asarray(q, float))
    rc, dr, ec, ein, eout, es = _linshell_parts(p, total)
    out = _sphere_amp(q, rc, ec - es)
    if dr > 0:
        ro = rc + dr
        m = (eout - ein) / dr
        b = ein - es - m * rc
        out = out + _sphere_amp(q, ro, b) - _sphere_amp(q, rc, b)
        out = out + 4.0 * math.pi * m * (ro**4 * _chi(q * ro) - rc**4 * _chi(q * rc))
    return out


def linshell_f(q, p):
    """p = [R_core, dR, eta_core, eta_sh, x_in_solv, x_out_solv, eta_solv]: core R_core, shell dR with an SLD
    linear between (1-x_in) eta_sh + x_in eta_solv at R_core and the same with x_out at R_core + dR."""
    return _linshell_f(np.asarray(q, float), p, False)


def linshell(q, p):
    return linshell_f(q, p) ** 2


def linshell_v(x, p, dist):
    return 4.0 / 3.0 * math.pi * (abs(p[0]) + abs(p[1])) ** 3 + 0.0 * np.asarray(x, float)


def linshell2_f(q, p):
    """p = [R_tot, dR, ...]: as linshell with the outer radius R_tot (core R_tot - dR)."""
    return _linshell_f(np.asarray(q, float), p, True)


def linshell2(q, p):
    return linshell2_f(q, p) ** 2


def linshell2_v(x, p, dist):
    return 4.0 / 3.0 * math.pi * abs(p[0]) ** 3 + 0.0 * np.asarray(x, float)


def _profile_linshell(r, p, total):
    r = np.abs(np.asarray(r, float))
    rc, dr, ec, ein, eout, es = _linshell_parts(p, total)
    m = (eout - ein) / dr if dr != 0 else 1e100
    return np.where(r <= rc, ec, np.where(r < rc + dr, m * r + (-m * rc + ein), es))


def profile_linshell(r, p):
    return _profile_linshell(r, p, False)


def profile_linshell2(r, p):
    return _profile_linshell(r, p, True)


# ---------------------------------------------------------------------------------------- exponential shell
def _exp_profile(x, ein, eout, alpha):
    if alpha < 0:
        return ein + (eout - ein) * x * np.exp((1.0 - x) * alpha)
    return eout + (ein - eout) * (1.0 - x) * np.exp(-x * alpha)


def expshell_f(q, p):
    """p = [R_core, dR, eta_core, eta_sh, x_in_solv, x_out_solv, alpha, eta_solv]: shell profile
    eta_in + (eta_out - eta_in) x e^{(1-x)α} (α < 0) or eta_out + (eta_in - eta_out)(1-x) e^{-xα} (α >= 0),
    x = (r - R_core)/dR."""
    q = np.asarray(q, float)
    rc, dr = p[0], p[1]
    ec, esh, xin, xout, alpha, es = p[2], p[3], p[4], p[5], p[6], p[7]
    out = _sphere_amp(q, rc, ec - es)
    if dr == 0:
        return out
    ein, eout = _solvent_mix(esh, xin, es), _solvent_mix(esh, xout, es)
    return out + _zone_amp(q, rc, rc + dr, lambda r: _exp_profile((r - rc) / dr, ein, eout, alpha) - es)


def expshell(q, p):
    return expshell_f(q, p) ** 2


def expshell_v(x, p, dist):
    """As C: core + shell with the solvent-fraction profile integrated."""
    R, dR, xin, xout, al = p[0], p[1], p[4], p[5], p[6]
    vc = 4.0 / 3.0 * math.pi * R**3
    vss = 4.0 / 3.0 * math.pi * (R + dR) ** 3 - vc
    e = math.exp(al)
    if al < 0:
        vsh = (-4 * dR * math.pi * (dR**2 * (6 + al * (6 + al * (3 + al))) - 6 * dR**2 * e
               + 2 * al * dR * (2 + al * (2 + al) - 2 * e) * R + al**2 * (1 + al - e) * R**2)) / al**4  # fmt: skip
        if xin < xout:
            v = vc + vss + (xout - xin) / (1.0 - xin) * vsh
        else:
            v = vc + vss + (xout - xin) / (1.0 - xout) * vsh
    else:
        vsh = (4 * dR * math.pi * (dR**2 * (6 + 4 * al + al**2 + 2 * (-3 + al) * e) + 2 * al * dR * (2 + al + (-2 + al) * e) * R
               + al**2 * (1 + (-1 + al) * e) * R**2)) / (al**4 * e)  # fmt: skip
        v = vc + vss * (1 - xout) + vsh * (xout - xin)
    return v + 0.0 * np.asarray(x, float)


def profile_expshell(r, p):
    r = np.abs(np.asarray(r, float))
    rc, dr = abs(p[0]), abs(p[1])
    ec, esh, xin, xout, alpha, es = p[2], p[3], p[4], p[5], p[6], p[7]
    ein, eout = _solvent_mix(esh, xin, es), _solvent_mix(esh, xout, es)
    with np.errstate(divide="ignore", invalid="ignore"):
        x = (r - rc) / dr
        sh = _exp_profile(x, ein, eout, alpha)
    return np.where(r < rc, ec, np.where(r < rc + dr, sh, es))


# ----------------------------------------------------------------------------------- core-shell microgel
def _mg_radii(p):
    wc, sc, wsh, ssh, d, so = p[0], p[1], p[2], p[3], p[4], p[5]
    rcore = wc + sc
    rshin = rcore + d
    rout = rcore + d + ssh + wsh + so
    return rcore, sc, rshin, ssh, rout, so


def _mg_prof(r, rad, sig):
    """1 inside rad-σ, parabolic transitions to 0 at rad+σ (C prof)."""
    r = np.abs(r)
    rad, sig = abs(rad), abs(sig)
    out = np.zeros(np.shape(r))
    if sig == 0:
        return np.where(r <= rad, 1.0, 0.0)
    out = np.where(r <= rad - sig, 1.0, out)
    out = np.where((r > rad - sig) & (r <= rad), 1.0 - 0.5 * ((r - rad + sig) / sig) ** 2, out)
    out = np.where((r > rad) & (r <= rad + sig), 0.5 * ((rad - r + sig) / sig) ** 2, out)
    return out


def _mg_amp(q, rad, sig, contrast):
    rad, sig = abs(rad), abs(sig)
    out = _sphere_amp(q, rad - sig, contrast) if rad > sig else np.zeros(np.shape(q))
    rho = lambda r: contrast * _mg_prof(r, rad, sig)
    # two zones: the second derivative of the profile jumps at r = rad
    return out + _zone_amp(q, max(rad - sig, 0.0), rad, rho) + _zone_amp(q, max(rad, 0.0), rad + sig, rho)


def coreshellmicrogel_f(q, p):
    """p = [W_core, sigma_core, W_sh, sigma_sh,in, D, sigma_out, eta_core, eta_shell, eta_sol]: core and shell with
    parabolic transitions of half widths σ (R_core = W_core + σ_core, R_sh,in = R_core + D,
    R_out = R_sh,in + σ_sh,in + W_sh + σ_out)."""
    q = np.asarray(q, float)
    rcore, sc, rshin, ssh, rout, so = _mg_radii(p)
    ec, esh, es = p[6], p[7], p[8]
    return _mg_amp(q, rout, so, esh - es) - _mg_amp(q, rshin, ssh, esh - es) + _mg_amp(q, rcore, sc, ec - es)


def coreshellmicrogel(q, p):
    return coreshellmicrogel_f(q, p) ** 2


def coreshellmicrogel_v(x, p, dist):
    *_, rout, so = _mg_radii(p)
    return 4.0 * math.pi * (rout + so) ** 3 / 3.0 + 0.0 * np.asarray(x, float)


def profile_coreshellmicrogel(r, p):
    r = np.asarray(r, float)
    rcore, sc, rshin, ssh, rout, so = _mg_radii(p)
    ec, esh, es = p[6], p[7], p[8]
    return es + (esh - es) * _mg_prof(r, rout, so) - (esh - es) * _mg_prof(r, rshin, ssh) + (ec - es) * _mg_prof(r, rcore, sc)


_FS = "plugins/fuzzysphere/sasfit_ff_"
_SH7 = ["R_core", "dR", "eta_core", "eta_sh", "x_in_solv", "x_out_solv", "eta_solv"]
_SH7S = {0: (5.0, 50.0, "log"), 1: (1.0, 20.0, "log"), 2: (0.5, 2.0, "lin"), 3: (0.5, 2.0, "lin"), 4: (0.0, 0.5, "lin"),
         5: (0.5, 1.0, "lin"), 6: (0.0, 0.3, "lin")}  # fmt: skip
_MG = ["W_core", "sigma_core", "W_sh", "sigma_sh,in", "D", "sigma_out", "eta_core", "eta_shell", "eta_sol"]
_MGS = {0: (5.0, 30.0, "log"), 1: (0.5, 5.0, "lin"), 2: (2.0, 20.0, "lin"), 3: (0.5, 4.0, "lin"), 4: (0.0, 10.0, "lin"),
        5: (0.5, 5.0, "lin"), 6: (0.5, 2.0, "lin"), 7: (0.2, 1.0, "lin"), 8: (0.0, 0.1, "lin")}  # fmt: skip
_EX = ["R_core", "dR", "eta_core", "eta_sh", "x_in_solv", "x_out_solv", "alpha", "eta_solv"]
_EXS = {**_SH7S, 6: (-4.0, 4.0, "lin"), 7: (0.0, 0.3, "lin")}
_FZ = ["R", "sigma", "eta_sph", "eta_sol"]
_FZS = {0: (5.0, 50.0, "log"), 1: (0.5, 5.0, "lin"), 2: (0.5, 2.0, "lin"), 3: (0.0, 0.3, "lin")}


def _qr(i):
    return lambda q, p: np.asarray(q, float) * abs(p[i])


SPECS = [
    make_spec("ff_fuzzysphere", fuzzysphere, _FZ, _FZS, amplitude=fuzzysphere_f, volume=fuzzysphere_v,
              source=_FS + "fuzzysphere.c", reduced=_qr(0), reduced_label="q R", rtol=1e-12),
    make_spec("ff_radial_profile_of_fuzzysphere", profile_fuzzysphere, _FZ, _FZS, source=_FS + "radial_profile_of_fuzzysphere.c",
              rtol=1e-12),
    make_spec("ff_linshell", linshell, _SH7, _SH7S, amplitude=linshell_f, volume=linshell_v, source=_FS + "LinShell.c",
              reduced=_qr(0), reduced_label="q R_core", rtol=1e-9),
    make_spec("ff_radial_profile_of_linshell", profile_linshell, _SH7, _SH7S, source=_FS + "radial_profile_of_LinShell.c",
              rtol=1e-13),
    make_spec("ff_linshell2", linshell2, ["R_tot"] + _SH7[1:], {**_SH7S, 0: (25.0, 60.0, "log")}, amplitude=linshell2_f,
              volume=linshell2_v, source=_FS + "LinShell2.c", reduced=_qr(0), reduced_label="q R_tot", rtol=1e-9),
    make_spec("ff_radial_profile_of_linshell2", profile_linshell2, ["R_tot"] + _SH7[1:], {**_SH7S, 0: (25.0, 60.0, "log")},
              source=_FS + "radial_profile_of_LinShell2.c", rtol=1e-13),
    make_spec("ff_expshell", expshell, _EX, _EXS, amplitude=expshell_f, volume=expshell_v, source=_FS + "ExpShell.c",
              reduced=_qr(0), reduced_label="q R_core", rtol=1e-9),
    make_spec("ff_radial_profile_of_expshell", profile_expshell, _EX, _EXS, source=_FS + "radial_profile_of_ExpShell.c",
              rtol=1e-13),
    make_spec("ff_coreshellmicrogel", coreshellmicrogel, _MG, _MGS, amplitude=coreshellmicrogel_f,
              volume=coreshellmicrogel_v, source=_FS + "coreshellmicrogel.c", reduced=_qr(0), reduced_label="q W_core",
              rtol=1e-9),
    make_spec("ff_radial_profile_of_coreshellmicrogel", profile_coreshellmicrogel, _MG, _MGS,
              source=_FS + "radial_profile_of_coreshellmicrogel.c", rtol=1e-13),
]  # fmt: skip
