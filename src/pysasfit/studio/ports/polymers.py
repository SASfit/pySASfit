"""Python ports of the isotropic polymer SASfit plugins: polymer_chains, polymer_stars, blob, hammouda_branched.

Stable forms: the Debye function and the amplitude (1 - e^-u)/u via series/expm1, the generalized Gaussian coil
P = (1/ν)[u^-a γ(a,u) - u^-2a γ(2a,u)], a = 1/(2ν), via :func:`.common.lower_gamma_ratio` (series for small u),
the Zimm-Schulz coil by its Taylor series for small k.

C deviations, all fixed in the SASfit sources on 2026-11-18 (``c_compat=True`` reproduces the old intensity of the
Guinier branches; documented in docs/studio_manual/python_ports.md):

* generalized_gaussian_coil(_2, _3): exp(-q²Rg²/3) for q Rg < 0.1 (relative error up to ~3e-6);
  fract_branched_polym: the same for q Rg < 0.01.
* gauss, gauss2, gauss3 ``_f``: (1 - e^{+u})/u instead of (1 - e^{-u})/u. ``_v``: 8/3/√π is evaluated with integer
  division (ρ = 2/√π), still as in C (to be checked).
* gauss(2,3), gauspoly, benoitstar lost digits by cancellation for small q; fract_branched_polym lost all digits for
  small q Rg (Γ - Γ_inc).
* dozierstar: for ν = 1 the Gaussian term I0 exp(-q²Rg²/3) lost its exp factor.
* gauss_exvol_star overwrote its parameter p[3] ("dummy") with 0.
"""

from __future__ import annotations

import math

import numpy as np
from scipy import special

from .common import (
    debye,
    debye_amplitude,
    lower_gamma_ratio,
    make_spec,
)

__all__ = ["SPECS", "generalized_gaussian_coil"]

NA = 6.0221415e23
_PC = "plugins/polymer_chains/sasfit_"
_PS = "plugins/polymer_stars/sasfit_"


def _qrg(q, p, k=0):
    return np.asarray(q, float) * abs(p[k])


def _qrg1(q, p):
    return np.asarray(q, float) * abs(p[1])


# ------------------------------------------------------------------------------------------------ Gaussian chains
def _gauss_v(x, p, dist):
    rho = 2.0 / math.sqrt(math.pi)  # C: 8/3/M_SQRTPI with integer division 8/3 = 2
    return 4.0 / 3.0 * math.pi * (p[0] / rho) ** 3 + 0.0 * np.asarray(x, float)


def gauss(q, p):
    """p = [Rg, dummy, I0]: I0 · 2(u - 1 + e^-u)/u², u = (q Rg)²."""
    return p[2] * debye((np.asarray(q, float) * p[0]) ** 2)


def gauss_f(q, p):
    return math.sqrt(p[2]) * debye_amplitude((np.asarray(q, float) * p[0]) ** 2)


def _beta2(p):
    return p[2] - p[3] * p[5]


def gauss2(q, p):
    """p = [Rg, dummy, b_p, V, dummy, eta_s]: (b_p - V eta_s)² Debye."""
    if p[3] < 0:
        raise ValueError("V >= 0 required")
    return _beta2(p) ** 2 * debye((np.asarray(q, float) * p[0]) ** 2)


def gauss2_f(q, p):
    return abs(_beta2(p)) * debye_amplitude((np.asarray(q, float) * p[0]) ** 2)


def _beta3(p):
    return p[2] - p[3] / p[4] / NA * p[5]


def gauss3(q, p):
    """p = [Rg, dummy, b_p, Mw, rho_p, eta_s]: V = Mw/(rho_p N_A)."""
    if p[0] < 0 or p[1] < 0 or p[3] < 0 or p[4] < 0:
        raise ValueError("Rg, Mw, rho_p >= 0 required")
    return _beta3(p) ** 2 * debye((np.asarray(q, float) * p[0]) ** 2)


def gauss3_f(q, p):
    return abs(_beta3(p)) * debye_amplitude((np.asarray(q, float) * p[0]) ** 2)


def gauspoly(q, p):
    """p = [Rg, Mw, Mn, I0]: polydisperse (Zimm-Schulz) Gaussian coil, w = Mw/Mn - 1:
    2[(1 + w k)^(-1/w) + k - 1]/((1 + w) k²), k = u/(1 + 2w)."""
    if min(p[0], p[1], p[2]) < 0:
        raise ValueError("Rg, Mw, Mn >= 0 required")
    q = np.asarray(q, float)
    w = p[1] / p[2] - 1.0
    u = (q * p[0]) ** 2
    k = u / (1.0 + 2.0 * w)
    out = np.empty(q.shape)
    small = k * (1.0 + w) < 0.25  # the series converges like (w k)^n
    if np.any(small):  # Σ_{n>=2} (-1)^n Π_{j<n}(1 + j w) k^n / n!, divided by k²
        ks = k[small]
        coef = 1.0 + w  # Π_{j<2}(1 + j w)
        term = coef / 2.0 * np.ones_like(ks)
        s = term.copy()
        for n in range(3, 60):
            term = term * (-(1.0 + (n - 1) * w)) * ks / n
            s = s + term
        out[small] = 2.0 * s / (1.0 + w)
    big = ~small
    if np.any(big):
        kb = k[big]
        lead = np.exp(-kb) if w == 0 else np.exp(-np.log1p(w * kb) / w)
        out[big] = 2.0 * (lead + kb - 1.0) / ((1.0 + w) * kb * kb)
    return p[3] * out


def generalized_gaussian_coil(q, rg, nu, c_compat=False):
    """P(q) of the generalized Gaussian coil (P(0) = 1)."""
    q = np.asarray(q, float)
    a = 1.0 / (2.0 * nu)
    u = (2.0 * nu + 1.0) * (2.0 * nu + 2.0) * (q * rg) ** 2 / 6.0
    P = (lower_gamma_ratio(a, u) - lower_gamma_ratio(2.0 * a, u)) / nu
    if c_compat:
        P = np.where(q * rg < 0.1, np.exp(-((q * rg) ** 2) / 3.0), P)
    return P


def generalized_gaussian_coil_amplitude(q, rg, nu):
    q = np.asarray(q, float)
    a = 1.0 / (2.0 * nu)
    u = (2.0 * nu + 1.0) * (2.0 * nu + 2.0) * (q * rg) ** 2 / 6.0
    return lower_gamma_ratio(a, u) / (2.0 * nu)


def _check_ggc(p):
    if p[0] < 0 or p[1] <= 0:
        raise ValueError("Rg >= 0 and nu > 0 required")


def ggc(q, p, c_compat=False):
    """p = [Rg, nu, I0]: generalized Gaussian coil (Flory exponent ν), |I0| P(q)."""
    _check_ggc(p)
    return abs(p[2]) * generalized_gaussian_coil(q, p[0], p[1], c_compat)


def ggc_f(q, p):
    return math.sqrt(abs(p[2])) * generalized_gaussian_coil_amplitude(q, p[0], p[1])


def _cube_v(x, p, dist):
    return np.asarray(x, float) ** 3


def ggc2(q, p, c_compat=False):
    """p = [Rg, nu, b_p, V, dummy, eta_s]: (b_p - V eta_s)² P(q)."""
    _check_ggc(p)
    if p[3] < 0:
        raise ValueError("V >= 0 required")
    return _beta2(p) ** 2 * generalized_gaussian_coil(q, p[0], p[1], c_compat)


def ggc2_f(q, p):
    return abs(_beta2(p)) * generalized_gaussian_coil_amplitude(q, p[0], p[1])


def ggc3(q, p, c_compat=False):
    """p = [Rg, nu, b_p, Mw, rho_p, eta_s]: V = Mw/(rho_p N_A)."""
    _check_ggc(p)
    if p[3] < 0 or p[4] < 0:
        raise ValueError("Mw, rho_p >= 0 required")
    return _beta3(p) ** 2 * generalized_gaussian_coil(q, p[0], p[1], c_compat)


# ------------------------------------------------------------------------------------------------ stars
def benoitstar(q, p):
    """p = [I0, Rg, dummy, dummy, f]: Benoit star of f Gaussian arms, v = u f/(3f - 2)."""
    f = p[4]
    if f < 2.0 / 3.0 or p[1] < 0:
        raise ValueError("f >= 2/3 and Rg >= 0 required")
    v = (np.asarray(q, float) * p[1]) ** 2 * f / (3.0 * f - 2.0)
    vs = np.where(v == 0, 1.0, v)
    a = 0.5 * debye(v) * v * v  # v - 1 + e^-v, accurate
    b = np.expm1(-vs) ** 2
    out = 2.0 / (f * vs * vs) * (a + (f - 1.0) * b / 2.0)
    return p[0] * np.where(v == 0, 1.0, out)


def broken_rods_star(q, p):
    """p = [I0, L, dummy, dummy, f]: star of f rigid rods of length L."""
    L, f = p[1], p[4]
    if L < 0 or f < 1:
        raise ValueError("L >= 0 and f >= 1 required")
    u = np.asarray(q, float) * L
    small = np.abs(u) < 1e-3
    us = np.where(small, 1.0, u)
    g = np.where(small, 1.0 - u**2 / 18.0 + u**4 / 600.0, special.sici(us)[0] / us)
    j0 = np.where(small, 1.0 - (u / 2.0) ** 2 / 6.0, np.sin(us / 2.0) / (us / 2.0))
    return p[0] / f * (2.0 * g - j0**2 + (f - 1.0) * g * g)


def _dozier_term(q, xi, mu):
    """Γ(μ+1) sin(μ atan(qξ))/(μ qξ) / (1+q²ξ²)^(μ/2), continuous at μ = 0 and q = 0."""
    x = np.asarray(q, float) * xi
    xs = np.where(x == 0, 1.0, x)
    t = np.arctan(xs)
    s = t if mu == 0 else np.sin(mu * t) / mu
    val = special.gamma(mu + 1.0) * s / xs / np.power(1.0 + xs * xs, 0.5 * mu)
    return np.where(x == 0, special.gamma(mu + 1.0), val)


def dozierstar(q, p):
    """p = [I0, Rg, alpha, nu, f]: I0 exp(-q²Rg²/3) + 4πα Γ(μ+1) sin(μ atan qξ)/(μ qξ (1+q²ξ²)^(μ/2)),
    μ = 1/ν - 1, ξ = 2Rg/√f."""
    i0, rg, alpha, nu, f = p[0], p[1], p[2], p[3], p[4]
    if i0 < 0 or rg < 0 or alpha < 0 or f < 1 or nu <= 0:
        raise ValueError("I0, Rg, alpha >= 0, f >= 1 and nu > 0 required")
    q = np.asarray(q, float)
    mu = 1.0 / nu - 1.0
    xi = 2.0 * rg / math.sqrt(f)
    return i0 * np.exp(-((rg * q) ** 2) / 3.0) + 4.0 * np.pi * alpha * _dozier_term(
        q, xi, mu
    )


def dozierstar2(q, p):
    """p = [I0, Rg, dummy, nu, f]: I0/f [(f - 1) exp(-q²Rg²/3) + Γ(μ) sin(μ atan qξ)/(qξ (1+q²ξ²)^(μ/2))]."""
    i0, rg, nu, f = p[0], p[1], p[3], p[4]
    if i0 < 0 or rg < 0 or f < 1 or nu < 0:
        raise ValueError("I0, Rg, nu >= 0 and f >= 1 required")
    q = np.asarray(q, float)
    mu = 1.0 / nu - 1.0
    xi = 2.0 * rg / math.sqrt(f)
    out = (
        i0 / f * ((f - 1.0) * np.exp(-((rg * q) ** 2) / 3.0) + _dozier_term(q, xi, mu))
    )
    return np.where(q == 0, i0, out)


def _gauss_exvol_star(q, p, c_compat=False):
    i0, rg, nu, f = p[0], p[1], p[2], p[4]
    if f < 1 or rg < 0:
        raise ValueError("f >= 1 and Rg >= 0 required")
    p1 = generalized_gaussian_coil(q, rg, nu, c_compat)
    p2 = generalized_gaussian_coil(q, 2.0**nu * rg, nu, c_compat)
    return i0 / f * (p1 + (f - 1.0) * (2.0 * p2 - p1))


def polydispersestar(q, p):
    """p = [I0, Rg, dummy, dummy, f]: I0 (1 + u/(3f))/(1 + u(1+f)/(6f))², u = (q Rg)²."""
    if p[1] < 0 or p[4] < 0:
        raise ValueError("Rg, f >= 0 required")
    u = (np.asarray(q, float) * p[1]) ** 2
    f = p[4]
    return p[0] * (1.0 + u / (3.0 * f)) / (1.0 + u * (1.0 + f) / (6.0 * f)) ** 2


# ------------------------------------------------------------------------------------------------ blobs
def _e3(q, rg):
    return special.erf(q * rg / math.sqrt(6.0)) ** 3


def blob_beaucage(q, p):
    """p = [Rg, P, dummy, c_P]: c_P (erf³(q Rg/√6)/q)^P (0 at q = 0)."""
    if p[0] <= 0:
        raise ValueError("Rg > 0 required")
    q = np.asarray(q, float)
    qs = np.where(q == 0, 1.0, q)
    return np.where(q == 0, 0.0, p[3] * np.power(_e3(qs, p[0]) / qs, p[1]))


def blob_dendrimer(q, p):
    """p = [Rg, nu, xi, c_nu]: blob scattering of a dendrimer, q_b = q ξ / erf³(q Rg/√6)."""
    rg, nu, xi, alpha = p[0], p[1], p[2], p[3]
    if rg <= 0 or xi <= 0 or alpha < 0 or not (1.0 / 3.0 < nu <= 1.0):
        raise ValueError("Rg, xi > 0, c_nu >= 0 and 1/3 < nu <= 1 required")
    q = np.asarray(q, float)
    qs = np.where(q == 0, 1.0, q)
    e3 = _e3(qs, rg)
    if nu == 1:
        out = 2.0 * alpha * np.arctan(qs * xi / e3) * e3 / (np.pi * qs)
    else:
        mu = 1.0 / nu - 1.0
        qb = qs * xi / e3
        c = abs(
            4.0
            * np.pi
            * xi ** (-1.0 / nu)
            * math.cos(np.pi / (2.0 * nu))
            * special.gamma(mu)
        )
        out = alpha * np.abs(4.0 * np.pi * special.gamma(mu) / qb * np.sin(mu * np.arctan(qb))
                             / np.power(1.0 + qb * qb, mu / 2.0) / c)  # fmt: skip
    return np.where(q == 0, 0.0, out)


def blob_star(q, p):
    """p = [Rg, nu, f, c_nu]: blob scattering of a star, q_b = 2 q Rg/√f."""
    rg, nu, f, alpha = p[0], p[1], p[2], p[3]
    if rg < 0 or f <= 0 or not (1.0 / 3.0 < nu <= 1.0):
        raise ValueError("Rg >= 0, f > 0 and 1/3 < nu <= 1 required")
    q = np.asarray(q, float)
    qs = np.where(q == 0, 1.0, q)
    q0 = alpha * abs(
        2.0 ** (1.0 / nu)
        * f ** (-0.5 / nu)
        * (-1.0 + 1.0 / nu)
        * rg ** (1.0 / nu)
        / math.cos(np.pi / (2.0 * nu))
    )
    if nu == 1:
        out = 2.0 * alpha * np.arctan(2.0 * qs * rg / math.sqrt(f)) / (np.pi * qs)
    else:
        mu = 1.0 / nu - 1.0
        qb = qs * 2.0 * rg / math.sqrt(f)
        c = abs(2.0 ** (2.0 - 1.0 / nu) * f ** (0.5 / nu) * np.pi * rg ** (-1.0 / nu) * math.cos(np.pi / (2.0 * nu))
                * special.gamma(mu))  # fmt: skip
        out = alpha * np.abs(4.0 * np.pi * special.gamma(mu) / qb * np.sin(mu * np.arctan(qb))
                             / np.power(1.0 + qb * qb, mu / 2.0) / c)  # fmt: skip
    return np.where(q == 0, q0, out)


# ------------------------------------------------------------------------------------------------ branched
def fract_branched(q, p, c_compat=False):
    """p = [Rg, vm, c, I0]: fractal branched polymer (Hammouda), I(0) = I0."""
    rg, vm, c, i0 = p[0], p[1], p[2], p[3]
    if rg < 0 or vm <= 0 or c <= 0:
        raise ValueError("Rg >= 0, vm > 0 and c > 0 required")
    q = np.asarray(q, float)
    ub = (q * rg) ** 2 * (2.0 * vm + c) * (2.0 * vm + c + 1.0) / 6.0
    a1, a2 = 0.5 * c / vm, 0.5 * (c + 1.0) / vm
    out = (
        i0
        * 0.5
        * (c * c + c)
        * (lower_gamma_ratio(a1, ub) - lower_gamma_ratio(a2, ub))
        / vm
    )
    if c_compat:
        out = np.where(q * rg < 0.01, i0 * np.exp(-((q * rg) ** 2) / 3.0), out)
    return out


# ------------------------------------------------------------------------------------------------ registry
_RG = (1.0, 100.0, "log")
_GGC_NOTE = "C before 2026-11-18: exp(-q²Rg²/3) for q Rg < 0.1 (c_compat)"
SPECS = [
    make_spec("ff_gauss", gauss, ["Rg", "dummy", "I0"], {0: _RG, 2: (0.1, 10.0, "log")}, amplitude=gauss_f,
              volume=_gauss_v, source=_PC + "ff_gauss.c", reduced=_qrg, reduced_label="q Rg",
              notes="C before 2026-11-18: _f with e^{+u}, digits lost for small q"),
    make_spec("ff_gauss2", gauss2, ["Rg", "dummy", "b_p", "V", "dummy", "eta_s"],
              {0: _RG, 2: (1.0, 10.0, "lin"), 3: (0.1, 5.0, "lin"), 5: (0.0, 1.0, "lin")}, amplitude=gauss2_f,
              volume=_gauss_v, source=_PC + "ff_gauss2.c", reduced=_qrg, reduced_label="q Rg"),
    make_spec("ff_gauss3", gauss3, ["Rg", "dummy", "b_p", "Mw", "rho_p", "eta_s"],
              {0: _RG, 2: (1e-12, 1e-11, "lin"), 3: (1e4, 1e6, "log"), 4: (0.8, 1.5, "lin"), 5: (0.0, 1e10, "lin")},
              amplitude=gauss3_f, volume=_gauss_v, source=_PC + "ff_gauss3.c", reduced=_qrg, reduced_label="q Rg"),
    make_spec("ff_gauspoly", gauspoly, ["Rg", "Mw", "Mn", "I0"],
              {0: _RG, 1: (1.05e4, 3e4, "log"), 2: (1e4, 1e4, "lin"), 3: (0.1, 10.0, "log")},
              source=_PC + "ff_gauspoly.c", reduced=_qrg, reduced_label="q Rg", rtol=1e-7,
              notes="C before 2026-11-18: digits lost for small q"),
    make_spec("ff_generalized_gaussian_coil", ggc, ["Rg", "nu", "I0"], {0: _RG, 1: (0.3, 0.8, "lin"), 2: (0.1, 10.0, "log")},
              amplitude=ggc_f, volume=_cube_v, source=_PC + "ff_generalized_gaussian_coil.c", reduced=_qrg,
              reduced_label="q Rg", c_compat=False, notes=_GGC_NOTE),
    make_spec("ff_generalized_gaussian_coil_2", ggc2, ["Rg", "nu", "b_p", "V", "dummy", "eta_s"],
              {0: _RG, 1: (0.3, 0.8, "lin"), 2: (1.0, 10.0, "lin"), 3: (0.1, 5.0, "lin"), 5: (0.0, 1.0, "lin")},
              amplitude=ggc2_f, volume=_cube_v, source=_PC + "ff_generalized_gaussian_coil_2.c", reduced=_qrg,
              reduced_label="q Rg", c_compat=False, notes=_GGC_NOTE),
    make_spec("ff_generalized_gaussian_coil_3", ggc3, ["Rg", "nu", "b_p", "Mw", "rho_p", "eta_s"],
              {0: _RG, 1: (0.3, 0.8, "lin"), 2: (1e-12, 1e-11, "lin"), 3: (1e4, 1e6, "log"), 4: (0.8, 1.5, "lin"),
               5: (0.0, 1e10, "lin")},
              source=_PC + "ff_generalized_gaussian_coil_3.c", reduced=_qrg, reduced_label="q Rg", c_compat=False,
              notes=_GGC_NOTE),
    make_spec("ff_benoitstar", benoitstar, ["I0", "Rg", "dummy", "dummy", "f"],
              {0: (0.1, 10.0, "log"), 1: _RG, 4: (1.0, 30.0, "lin")}, source=_PS + "ff_benoitstar.c", reduced=_qrg1, reduced_label="q Rg", rtol=1e-7, notes="C before 2026-11-18: digits lost for small q"),
    make_spec("ff_broken_rods_star", broken_rods_star, ["I0", "L", "dummy", "dummy", "f"],
              {0: (0.1, 10.0, "log"), 1: _RG, 4: (1.0, 30.0, "lin")}, source=_PS + "ff_broken_rods_star.c",
              reduced=_qrg1, reduced_label="q L"),
    make_spec("ff_dozierstar", dozierstar, ["I0", "Rg", "alpha", "nu", "f"],
              {0: (0.1, 10.0, "log"), 1: _RG, 2: (0.0, 1.0, "lin"), 3: (0.4, 0.95, "lin"), 4: (1.0, 30.0, "lin")},
              source=_PS + "ff_dozierstar.c", reduced=_qrg1, reduced_label="q Rg",
              notes="C before 2026-11-18: no exp(-q²Rg²/3) on I0 for nu = 1"),
    make_spec("ff_dozierstar2", dozierstar2, ["I0", "Rg", "dummy", "nu", "f"],
              {0: (0.1, 10.0, "log"), 1: _RG, 3: (0.4, 0.95, "lin"), 4: (1.0, 30.0, "lin")},
              source=_PS + "ff_dozierstar2.c", reduced=_qrg1, reduced_label="q Rg"),
    make_spec("ff_gauss_exvol_star", _gauss_exvol_star, ["I0", "Rg", "nu", "dummy", "f"],
              {0: (0.1, 10.0, "log"), 1: _RG, 2: (0.3, 0.8, "lin"), 4: (1.0, 30.0, "lin")},
              source=_PS + "ff_gauss_exvol_star.c", reduced=_qrg1, reduced_label="q Rg", rtol=1e-8, c_compat=False,
              notes=_GGC_NOTE + "; C overwrites p[3]"),
    make_spec("ff_polydispersestar", polydispersestar, ["I0", "Rg", "dummy", "dummy", "f"],
              {0: (0.1, 10.0, "log"), 1: _RG, 4: (1.0, 30.0, "lin")}, source=_PS + "ff_polydispersestar.c",
              reduced=_qrg1, reduced_label="q Rg"),
    make_spec("ff_blob__beaucage_", blob_beaucage, ["Rg", "P", "dummy", "c_P"],
              {0: _RG, 1: (1.0, 3.0, "lin"), 3: (0.1, 10.0, "log")}, source="plugins/blob/sasfit_ff_blob__beaucage_.c",
              reduced=_qrg, reduced_label="q Rg"),
    make_spec("ff_blob__dendrimer_", blob_dendrimer, ["Rg", "nu", "xi", "c_nu"],
              {0: _RG, 1: (0.4, 0.95, "lin"), 2: (1.0, 20.0, "log"), 3: (0.1, 10.0, "log")},
              source="plugins/blob/sasfit_ff_blob__dendrimer_.c", reduced=_qrg, reduced_label="q Rg"),
    make_spec("ff_blob__star_", blob_star, ["Rg", "nu", "f", "c_nu"],
              {0: _RG, 1: (0.4, 0.95, "lin"), 2: (1.0, 30.0, "lin"), 3: (0.1, 10.0, "log")},
              source="plugins/blob/sasfit_ff_blob__star_.c", reduced=_qrg, reduced_label="q Rg"),
    make_spec("ff_fract_branched_polym", fract_branched, ["Rg", "vm", "c", "I0"],
              {0: _RG, 1: (0.3, 0.8, "lin"), 2: (1.0, 3.0, "lin"), 3: (0.1, 10.0, "log")},
              source="plugins/hammouda_branched/sasfit_ff_fract_branched_polym.c", reduced=_qrg, reduced_label="q Rg",
              c_compat=False, notes="C before 2026-11-18: Guinier for q Rg < 0.01 (c_compat), all digits lost above"),
]  # fmt: skip
