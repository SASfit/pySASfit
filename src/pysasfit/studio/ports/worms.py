"""Python ports of the SASfit plugin polymer_semi_flexible (worm-like chains) and of the worm functions of the
thin_objects plugin they call.

* Pedersen & Schurtenberger, Macromolecules 29 (1996) 7602: methods 1, 2 and 3 (``ff_worms_ps1_``,
  ``ff_worms_ps2_``, ``ff_worms_ps3_``), without (exvol = 0) and with excluded volume (exvol >= 1, methods 1, 3);
* Kholodenko worm (``ff_kholodenko_worm``, integral over the contour by Gauss-Legendre on graded panels);
* freely jointed chain of rods (``ff_freely_joined_chain_of_rods``);
* Koyama worm (``ff_koyama_worm``, Koyama, J. Phys. Soc. Jpn. 34 (1973) 1029): the pair distribution of a
  contour separation t = x lb/2 is approximated from <R²> and <R⁴> of the worm-like chain,
  φ(x) = exp(-s² f(x)/3) j0(s g(x)), s = q lb/2, P = (2/X²) ∫_0^X (X - x) φ(x) dx, X = 2L/lb
  (rod limit φ = j0(qt), Gaussian limit exp(-q²<R²>/6)). Batch 6, 2026-11-24.

All are I0 · S(q) with S(0) = 1; :func:`worm_ps3` (S(q) of method 3, normalised to 1) is also used by the micelle
ports (:mod:`.micelles_worm`).

C fixes of 2026-11-23 (thin_objects, spherical_ellipsoidal_micelles), followed here:
fRg2 lost all digits for L << b (1 - 1.5/n + 1.5/n² - 0.75/n³(1 - e^{-2n}), series below n = 1); IDebye used
1 - u/3 below u = 0.01 (error u²/12 up to 8e-6); the Sharp-Bloomfield term 7/(15u)(1 - e^{-u}) cancelled for small u
(expm1); method 3 with excluded volume took dS/dq at q0 by a central difference with the absolute step 1e-8
(relative error ~1e-8 of the large-q coefficients): step 1e-3 q0 now (both use the GSL five-point rule);
freely jointed chain: (1 - γ)⁻¹ - n⁻¹(1 - γⁿ)(1 - γ)⁻² cancelled for small qb (100 % at qb = 1e-5, the series was
used only below 1e-6), and γⁿ was NaN for γ = j0(qb) < 0 and non-integer n = L/b (now |γ|ⁿ cos(πn), equal for
integer n); method 2: 1 - e^{-x} by expm1.
"""

from __future__ import annotations

import math

import numpy as np
from scipy import special

from .common import debye, make_spec
from .quadrature import gauss_legendre

__all__ = ["SPECS", "worm_ps1", "worm_ps2", "worm_ps3", "kholodenko", "freely_jointed", "koyama", "frg2", "falpha2"]

_EPS = np.finfo(float).eps


# ------------------------------------------------------------------------------------------------ helpers
def frg2(L, b):
    """Rg² of an ideal semiflexible chain (contour L, Kuhn length b): L b/6 g(n), n = L/b,
    g = 1 - 1.5/n + 1.5/n² - 0.75/n³ (1 - e^{-2n}) = 0.75 Σ_{m>=4} (-2)^m n^(m-3)/m! (series for n < 1)."""
    n = L / b
    if n < 1.0:
        # t_m = 0.75 (-2)^m n^(m-3)/m!, from m = 4
        g, t = 0.0, 0.5 * n
        m = 4
        while True:
            g += t
            m += 1
            t *= -2.0 * n / m
            if abs(t) < 1e-17 * abs(g) or m > 80:
                break
    else:
        g = 1.0 - 1.5 / n + 1.5 / n**2 - 0.75 / n**3 * (-math.expm1(-2.0 * n))
    return L * b / 6.0 * g


def falpha2(L, b):
    """Excluded-volume expansion α²(n) = (1 + (n/3.12)² + (n/8.67)³)^(ε/3), ε = 0.170."""
    n = L / b
    return (1.0 + (n / 3.12) ** 2 + (n / 8.67) ** 3) ** (0.170 / 3.0)


def idebye(q, rg):
    return debye((np.asarray(q, float) * rg) ** 2)


def _sb_term(u):
    """4/15 + 7/(15u) - (11/15 + 7/(15u)) e^{-u} = 4/15 - 11/15 e^{-u} - 7/15 expm1(-u)/u (-> 0 at u = 0)."""
    u = np.asarray(u, float)
    us = np.where(u == 0, 1.0, u)
    return np.where(u == 0, 0.0, 4.0 / 15.0 - 11.0 / 15.0 * np.exp(-us) - 7.0 / 15.0 * np.expm1(-us) / us)


def s_sb(q, rg, L, b):
    """Daniels approximation (Sharp and Bloomfield)."""
    u = (np.asarray(q, float) * rg) ** 2
    return idebye(q, rg) + _sb_term(u) * b / L


_C1, _C2, _C3, _EPSV = 1.2220, 0.4288, -1.651, 0.170
_NU = (1.0 + _EPSV) / 2.0
_C4, _C5 = 1.523, 0.1477


def _fcorr(q, rg, clb, b, L):
    """1 where dS/dq <= 0 of the interpolated chain function, else 0 (Chen, Butler, Magid 2006); the C formula."""
    q = np.asarray(q, float)
    th = np.tanh((q * rg - _C4) / _C5)
    e = np.exp(-q * q * rg * rg)
    x = q * rg
    dS = (-(1.0 - th**2) / rg**3 / _C5 * (2.0 * e + 2.0 * q * q * rg * rg - 2.0) / q**4 / 2.0
          + (0.5 - th / 2.0) * (-4.0 * q * rg * rg * e + 4.0 * q * rg * rg) / q**4 / rg**4
          - 4.0 * (0.5 - th / 2.0) * (2.0 * e + 2.0 * q * q * rg * rg - 2.0) / q**5 / rg**4
          + (1.0 - th**2) * rg / _C5 * (_C1 * x ** (-1.0 / _NU) + _C2 * x ** (-2.0 / _NU) + _C3 * x ** (-3.0 / _NU)) / 2.0
          + (0.5 + th / 2.0) * (-_C1 * x ** (-1.0 / _NU) / _NU / q - 2.0 * _C2 * x ** (-2.0 / _NU) / _NU / q
                                - 3.0 * _C3 * x ** (-3.0 / _NU) / _NU / q)
          + clb * (-14.0 / 15.0 / q**3 / rg**2 + 14.0 / 15.0 / q**3 / rg**2 * e
                   + 2.0 * (11.0 / 15.0 + 7.0 / 15.0 / (q * q) / (rg * rg)) * q * rg * rg * e) * b / L)  # fmt: skip
    return np.where(dS <= 0, 1.0, 0.0)


def s_exv(q, rg, clb, b, L):
    """Chain with excluded volume (Utiyama et al., approximation of Pedersen and Schurtenberger)."""
    q = np.asarray(q, float)
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        x = q * rg
        w = 0.5 * (1.0 + np.tanh((x - _C4) / _C5))
        xs = np.where(x > 0, x, 1.0)
        tail = (_C1 * xs ** (-1.0 / _NU) + _C2 * xs ** (-2.0 / _NU) + _C3 * xs ** (-3.0 / _NU)) * _fcorr(
            np.where(q > 0, q, 1.0), rg, clb, b, L)
        return np.where(w > 0, (1.0 - w) * idebye(q, rg) + w * tail, idebye(q, rg))


def _central_deriv(f, x, h):
    fm1, fp1, fmh, fph = f(x - h), f(x + h), f(x - h / 2.0), f(x + h / 2.0)
    r3 = 0.5 * (fp1 - fm1)
    r5 = 4.0 / 3.0 * (fph - fmh) - r3 / 3.0
    e3 = (abs(fp1) + abs(fm1)) * _EPS
    e5 = 2.0 * (abs(fph) + abs(fmh)) * _EPS + e3
    dy = max(abs(r3 / h), abs(r5 / h)) * (abs(x) / h) * _EPS
    return r5 / h, abs(e5 / h) + dy, abs((r5 - r3) / h)


def gsl_deriv_central(f, x, h):
    """GSL's gsl_deriv_central (five-point rule with one step optimisation)."""
    r0, rnd, trc = _central_deriv(f, x, h)
    err = rnd + trc
    if rnd < trc and rnd > 0 and trc > 0:
        hopt = h * (rnd / (2.0 * trc)) ** (1.0 / 3.0)
        ropt, rnd2, trc2 = _central_deriv(f, x, hopt)
        err2 = rnd2 + trc2
        if err2 < err and abs(ropt - r0) < 4.0 * err:
            r0 = ropt
    return r0


# ------------------------------------------------------------------------------------------------ method 3
def _maple_sb(q0, RG, B, L, clb, p1, p2):
    """Large-q coefficients a1, a2 of method 3 without excluded volume, L > 4b (C, Maple)."""
    E = math.exp(-q0 * q0 / (B * B) * RG * RG)
    PI = math.pi
    a1 = -q0 ** (p1 - 4.0) * (
        -60.0 * q0**2 * RG**2 * B**3 * L + 30.0 * B**5 * L * E * p2 - 30.0 * B**5 * L * p2
        - 60.0 * q0**2 * RG**2 * B**3 * L * E + 14.0 * clb * q0**2 * RG**2 * E * B**4
        + 22.0 * clb * q0**6 * RG**6 * E + 14.0 * clb * q0**4 * RG**4 * E * B**2 + 15.0 * B**2 * q0**3 * RG**4 * PI
        - 14.0 * clb * q0**2 * RG**2 * B**4 + 120.0 * B**5 * L - 120.0 * B**5 * L * E + 30.0 * q0**2 * RG**2 * B**3 * L * p2
        - 15.0 * B**2 * q0**3 * RG**4 * PI * p2 - 11.0 * clb * q0**4 * RG**4 * E * B**2 * p2
        + 4.0 * B**2 * clb * q0**4 * RG**4 * p2 + 7.0 * clb * q0**2 * RG**2 * B**4 * p2
        - 7.0 * clb * q0**2 * RG**2 * E * B**4 * p2) / L / RG**4 / B / (-p2 + p1) / 15.0  # fmt: skip
    a2 = q0 ** (p2 - 4.0) * (
        -60.0 * q0**2 * RG**2 * B**3 * L * E - 60.0 * q0**2 * RG**2 * B**3 * L - 120.0 * B**5 * L * E + 120.0 * B**5 * L
        - 14.0 * clb * q0**2 * RG**2 * B**4 + 14.0 * clb * q0**2 * RG**2 * E * B**4 + 22.0 * clb * q0**6 * RG**6 * E
        + 14.0 * clb * q0**4 * RG**4 * E * B**2 - 15.0 * B**2 * p1 * PI * q0**3 * RG**4 + 15.0 * B**2 * q0**3 * RG**4 * PI
        + 30.0 * B**3 * p1 * L * q0**2 * RG**2 - 30.0 * L * p1 * B**5 + 4.0 * B**2 * p1 * clb * q0**4 * RG**4
        + 30.0 * B**5 * p1 * L * E - 11.0 * B**2 * p1 * clb * q0**4 * RG**4 * E + 7.0 * B**4 * p1 * clb * q0**2 * RG**2
        - 7.0 * B**4 * p1 * clb * q0**2 * RG**2 * E) / B / RG**4 / L / (-p2 + p1) / 15.0  # fmt: skip
    return a1, a2


def _maple_debye(q0, RG, B, L, p1, p2):
    """Large-q coefficients for L <= 4b (Debye function at low q)."""
    E = math.exp(-q0 * q0 / (B * B) * RG * RG)
    PI = math.pi
    a1 = B * q0 ** (p1 - 4.0) * (
        -2.0 * B * L * q0**2 * RG**2 * p2 + 4.0 * B * L * q0**2 * RG**2 * E + 4.0 * B * L * q0**2 * RG**2
        + 8.0 * B**3 * L * E - 8.0 * B**3 * L - PI * q0**3 * RG**4 + PI * q0**3 * RG**4 * p2
        - 2.0 * B**3 * L * E * p2 + 2.0 * B**3 * L * p2) / L / RG**4 / (p1 - p2)  # fmt: skip
    a2 = -B * q0 ** (p2 - 4.0) * (
        4.0 * B * L * q0**2 * RG**2 * E + 4.0 * B * L * q0**2 * RG**2 + 8.0 * B**3 * L * E - 8.0 * B**3 * L
        + p1 * PI * q0**3 * RG**4 - PI * q0**3 * RG**4 - 2.0 * p1 * B * L * q0**2 * RG**2
        - 2.0 * p1 * B**3 * L * E + 2.0 * p1 * B**3 * L) / RG**4 / L / (p1 - p2)  # fmt: skip
    return a1, a2


def _tail(q, a1, a2, p1, p2, B, L):
    with np.errstate(divide="ignore"):
        return a1 / (q * B) ** p1 + a2 / (q * B) ** p2 + math.pi / (q * L)


def worm_ps3(q, b, L, exvol):
    """S(q) of method 3 of Pedersen and Schurtenberger (S(0) = 1); exvol < 1 without, >= 1 with excluded volume."""
    q = np.atleast_1d(np.asarray(q, float))
    B = float(b)
    L = float(L)
    out = np.empty(q.shape)
    zero = np.abs(L * q) <= 1e-16
    a3, a4, p3 = 2.02, 3.06, 0.44
    if L > 4.0 * B:
        q0 = 3.1
    elif exvol < 1:
        q0 = max(a3 * B / math.sqrt(frg2(L, B)), 4.0)
    else:
        q0 = max(a3 * B / math.sqrt(frg2(L, B) * falpha2(L, B)), 3.0)
    qs = np.where(zero, 1.0, q)
    if exvol < 1:
        if L > 4.0 * B:
            RG = math.sqrt(L * B / 6.0)
            lo = qs * B < q0
            a1, a2 = _maple_sb(q0, RG, B, L, 1.0, 4.95, 5.29)
            S = np.where(lo, s_sb(qs, RG, L, B), _tail(qs, a1, a2, 4.95, 5.29, B, L))
        else:
            RG = math.sqrt(frg2(L, B))
            lo = qs * B < q0
            a1, a2 = _maple_debye(q0, RG, B, L, 5.13, 7.47)
            S = np.where(lo, idebye(qs, RG), _tail(qs, a1, a2, 5.13, 7.47, B, L))
    else:
        if L > 4.0 * B:
            RG = math.sqrt(falpha2(L, B) * frg2(L, B))
            clb = a4 / (L / B) ** p3 if L > 10.0 * B else 1.0

            def low(x):
                x = np.asarray(x, float)
                return s_exv(x, RG, clb, B, L) + clb * _sb_term((RG * x) ** 2) * B / L

            lo = qs * B <= q0
            p1, p2 = 4.12, 4.42
            x0 = q0 / B
            Sf = L * L * float(low(x0))
            Sdf = gsl_deriv_central(lambda x: L * L * float(low(x)), x0, 1e-3 * x0)
            a1 = -q0 ** (p1 - 1.0) * (-B * B * L * p2 * math.pi + B * B * L * math.pi + B * Sf * p2 * q0
                                      + Sdf * q0 * q0) / (L * L) / B / (p1 - p2)  # fmt: skip
            a2 = q0 ** (p2 - 1.0) * (-B * B * L * p1 * math.pi + B * B * L * math.pi + B * Sf * p1 * q0
                                     + Sdf * q0 * q0) / (L * L) / B / (p1 - p2)  # fmt: skip
            S = np.where(lo, low(qs), _tail(qs, a1, a2, p1, p2, B, L))
        else:
            RG = math.sqrt(frg2(L, B) * falpha2(L, B))
            q0 = max(a3 * B / math.sqrt(frg2(L, B)), 4.0)
            lo = qs * B <= q0
            a1, a2 = _maple_debye(q0, RG, B, L, 5.36, 5.62)
            S = np.where(lo, idebye(qs, RG), _tail(qs, a1, a2, 5.36, 5.62, B, L))
    out[:] = np.where(zero, 1.0, S)
    return out


# ------------------------------------------------------------------------------------------------ method 2
def worm_ps2(q, b, L):
    q = np.atleast_1d(np.asarray(q, float))
    B, L = float(b), float(L)
    nb = L / B
    qs = np.where(q == 0, 1.0, q)
    slocal = (1.0 if nb > 2 else 0.0625) / (L * B * qs * qs) + math.pi / (L * qs)
    if nb > 2:
        rg = math.sqrt(L * B / 6.0)
        s0 = s_sb(qs, rg, L, B)
        y = (qs * B / 5.53) ** 5.33
    else:
        rg = math.sqrt(frg2(L, B))
        s0 = idebye(qs, rg)
        y = (qs * B / (11.7 / nb)) ** 3.95
    S = s0 * np.exp(-y) + slocal * (-np.expm1(-y))
    return np.where(q == 0, 1.0, S)


# ------------------------------------------------------------------------------------------------ method 1
_NA1 = np.array([[0, 0, 0], [0, 0, 0], [0.3054, 0.2316, -22.779], [0.05777, 2.6531, 23.2457],
                 [-0.00604, 0.3706, 8.1092], [-0.03902, -1.0081, -3.3603]])  # fmt: skip
_NA2 = np.array([[0, 0, 0], [0, 0, 0], [0.0, -0.4963, -0.4678], [0.0, 0.03688, 0.3365], [0.0, 0.30570, 0.4290],
                 [0.0, 0.39013, 0.3737]])  # fmt: skip
_NB1 = np.array([[-0.0162, -0.3565, -0.3078], [0.09046, 0.1909, 0.05176], [0.1213, 0.15634, 0.01568]])
_NB2 = np.array([[0.0, -0.3946, 1.1361], [0.0, -0.2231, -0.01615], [0.0, -0.2546, -0.07606]])
_XA1 = np.array([[0, 0, 0], [0, 0, 0], [-0.1222, 1.761, -26.04], [0.3051, 2.252, 20.00], [-0.0711, -1.291, 4.382],
                 [0.0584, 0.6994, 1.594]])  # fmt: skip
_XA2 = np.array([[0, 0, 0], [0, 0, 0], [0.0, 0.1212, 0.0170], [0.0, -0.4169, -0.4731], [0.0, 0.1988, 0.1869],
                 [0.0, 0.3435, 0.3350]])  # fmt: skip
_XB1 = np.array([[-0.0699, 0.1342, -0.2020], [-0.0900, 0.0138, -0.0114], [0.2677, 0.1898, 0.0123]])
_XB2 = np.array([[0.0, -0.5171, 0.6950], [0.0, -0.2028, -0.3238], [0.0, -0.3112, -0.5403]])


def irod(q, H):
    x = np.asarray(q, float) * H
    si, _ = special.sici(x)
    return 2.0 * si / x - (np.sin(x / 2.0) / (x / 2.0)) ** 2


def worm_ps1(q, b, L, exvol):
    q_mem = np.atleast_1d(np.asarray(q, float))
    B, L = float(b), float(L)
    ex = exvol >= 1
    a1, a2, b1, b2 = (_XA1, _XA2, _XB1, _XB2) if ex else (_NA1, _NA2, _NB1, _NB2)
    nb = L / B
    qm = np.where(q_mem == 0, 1.0, q_mem)
    qq = np.where(qm * B > 10.0, 10.0 / B, qm)
    if not ex:
        rg2 = frg2(L, B)
        xi = qq * math.pi / (2.0 * abs(L)) * rg2
    else:
        rg2 = frg2(L, B) * falpha2(L, B)
        xi = qq * B * (math.pi * B / (1.103 * abs(L))) ** 1.5 * (rg2 / (B * B)) ** 1.282
    x5 = xi**5
    with np.errstate(divide="ignore", over="ignore"):
        chi = np.where(x5 < 0.01, 0.0, np.exp(-1.0 / np.where(x5 < 0.01, 1.0, x5)))
    e1 = 0.0 if 10.0 / nb > 100 else math.exp(-10.0 / nb)
    e2 = 0.0 if 2.0 * nb > 100 else math.exp(-2.0 * nb)
    AA = np.zeros(6)
    for i in range(2, 6):
        for j in range(3):
            AA[i] += a1[i, j] * nb ** (-j) * e1 + (a2[i, j] * nb**j * e2 if j > 0 else 0.0)
    BB = np.zeros(3)
    for i in range(3):
        for j in range(3):
            BB[i] += b1[i, j] * nb ** (-j) + (b2[i, j] * nb**j * e2 if j > 0 else 0.0)
    wg = np.ones(qq.shape)
    for i in range(2, 6):
        wg = wg + (1.0 - chi) * AA[i] * xi**i
    for i in range(3):
        wg = wg + chi * BB[i] * xi ** (-i)
    if not ex:
        schain = idebye(qq, math.sqrt(rg2))
    else:
        schain = s_exv(qq, math.sqrt(rg2), 1.0, B, L)
    srod = irod(qq, L)
    core = ((1.0 - chi) * schain + chi * srod) * wg
    big = qm * B > 10.0
    swc = np.where(big, math.pi / (qm * L) + 100.0 * (core - math.pi / (10.0 * nb)) / (qm * qm * B * B), core)
    return np.where(q_mem == 0, 1.0, swc)


# ------------------------------------------------------------------------------------------------ Kholodenko
def _kholodenko_fy(x, qlb3):
    """f(y) of the Kholodenko worm (x > 0), qlb3 = q lb/3."""
    if qlb3 == 1.0:
        return x / np.sinh(x) if np.max(x) < 250 else np.exp(np.log(x) - _lnsinh(x))
    if qlb3 < 1.0:
        E = math.sqrt(1.0 - qlb3 * qlb3)
        return np.exp(_lnsinh(E * x) - _lnsinh(x)) / E
    F = math.sqrt(qlb3 * qlb3 - 1.0)
    return np.sin(F * x) * np.exp(-_lnsinh(x)) / F


def _lnsinh(x):
    x = np.asarray(x, float)
    return x + np.log(-np.expm1(-2.0 * x) / 2.0)


def kholodenko(q, b, L):
    """S(q) of the Kholodenko worm: ∫_0^{3N} 2/(3N) (1 - x/(3N)) f(x) dx, N = L/b; π/(qL) beyond 10π/b (N > 1)
    or 10π/L (N <= 1), as in C."""
    q = np.atleast_1d(np.asarray(q, float))
    B, L = float(b), float(L)
    N = L / B
    out = np.empty(q.shape)
    X = 3.0 * N
    for i, qi in enumerate(q):
        cut = 10.0 * math.pi / B if N > 1 else 10.0 * math.pi / L
        if qi == 0:
            out[i] = 1.0
            continue
        if qi > cut:
            out[i] = math.pi / (L * qi)
            continue
        qlb3 = qi * B / 3.0
        if qlb3 > 1.0:  # oscillating, decays like e^{-x}
            F = math.sqrt(qlb3 * qlb3 - 1.0)
            top = min(X, 45.0)
            n = max(1, math.ceil(top / min(1.0, 1.5 / F)))
            edges = np.linspace(0.0, top, n + 1)
        else:  # monotone, slow decay for small q: geometric panels
            edges = [0.0]
            e = 0.5
            while e < X:
                edges.append(e)
                e *= 2.0
            edges.append(X)
            edges = np.array(edges)
        xg, wg = gauss_legendre(24, edges[:-1], edges[1:])
        xg, wg = xg.ravel(), wg.ravel()
        f = _kholodenko_fy(xg, qlb3)
        out[i] = np.sum(wg * 2.0 / X * (1.0 - xg / X) * f)
    return out


# ------------------------------------------------------------------------------------------------ freely jointed chain
def _one_minus_sinc(x):
    """1 - sin(x)/x, series below 0.5."""
    x = np.asarray(x, float)
    small = np.abs(x) < 0.5
    x2 = x * x
    t = x2 / 6.0
    ser = np.zeros_like(x)
    for k in range(1, 12):
        ser = ser + t
        t = -t * x2 / ((2 * k + 2) * (2 * k + 3))
    xs = np.where(small, 1.0, x)
    return np.where(small, ser, 1.0 - np.sin(xs) / xs)


def _h(delta, gamma, n):
    """[n δ - 1 + γⁿ]/(n δ²), δ = 1 - γ (series in δ for small n δ; γⁿ = |γ|ⁿ cos(π n) for γ < 0)."""
    delta = np.asarray(delta, float)
    gamma = np.asarray(gamma, float)
    small = (np.abs(delta) * max(n, 1.0) < 0.5) & (np.abs(delta) < 0.5)
    # series (1/n) Σ_{k>=2} C(n,k) (-δ)^(k-2)
    c = n * (n - 1.0) / 2.0
    ser = np.zeros_like(delta)
    pw = np.ones_like(delta)
    for k in range(2, 70):
        ser = ser + c * pw
        c = c * (n - k) / (k + 1.0)
        pw = pw * (-delta)
    ser = ser / n
    with np.errstate(divide="ignore", invalid="ignore"):
        gn = np.where(gamma > 0, np.exp(n * np.log(np.where(gamma > 0, gamma, 1.0))),
                      np.abs(gamma) ** n * math.cos(math.pi * n))
        direct = (n * delta - 1.0 + gn) / (n * delta * delta)
    return np.where(small, ser, direct)


def freely_jointed(q, b, L):
    """S(q) of a freely jointed chain of N = L/b rods of length b."""
    q = np.atleast_1d(np.asarray(q, float))
    B, L = float(b), float(L)
    n = L / B
    x = q * B
    xs = np.where(x == 0, 1.0, x)
    si, _ = special.sici(xs)
    G = np.where(x == 0, 1.0, si / xs)
    gamma = np.where(x == 0, 1.0, np.sin(xs) / xs)
    delta = _one_minus_sinc(x)
    cB = 2.0 / n * G * G * _h(delta, gamma, n)
    cC = 0.5 * np.where(x == 0, 1.0, np.sinc(x / 2.0 / np.pi)) ** 2
    return 2.0 / n * (G - cC) + cB


# ------------------------------------------------------------------------------------------------ Koyama worm
def _koyama_a(x):
    """2<R²>/lb² = x - 1 + e^{-x} (series below 0.5)."""
    x = np.asarray(x, float)
    small = x < 0.5
    xs = np.where(small, x, 0.0)
    ser = np.zeros_like(x)
    t = xs * xs / 2.0
    for k in range(2, 26):
        ser = ser + t
        t = -t * xs / (k + 1)
    return np.where(small, ser, x + np.expm1(-np.where(small, 1.0, x)))


# Taylor coefficients of 4<R⁴>/lb⁴ = 5/3 x² - 52/9 x - 2/27 (1 - e^{-3x}) + 8 (1 - e^{-x}) - 2x e^{-x}
_KB = np.zeros(40)
for _k in range(1, 40):
    _KB[_k] = (2.0 / 27.0 * (-3.0) ** _k - 8.0 * (-1.0) ** _k + 2.0 * _k * (-1.0) ** _k) / math.factorial(_k)
_KB[1] += -52.0 / 9.0
_KB[2] += 5.0 / 3.0
_KB[:4] = 0.0  # exact cancellation: the series starts with x⁴/4


def _koyama_b(x):
    """4<R⁴>/lb⁴ (series below 1: the closed form cancels like x⁻³ ε for small x)."""
    x = np.asarray(x, float)
    small = x < 1.0
    xs = np.where(small, x, 0.0)
    ser = np.polynomial.polynomial.polyval(xs, _KB)
    xl = np.where(small, 1.0, x)
    direct = (5.0 / 3.0 * xl * xl - 52.0 / 9.0 * xl + 2.0 / 27.0 * np.expm1(-3.0 * xl) - 8.0 * np.expm1(-xl)
              - 2.0 * xl * np.exp(-xl))
    return np.where(small, ser, direct)


def _koyama_fg(x):
    """(f, g²) of Koyama's distribution in units of lb²/4·q² ... : g² = a sqrt(10 |1 - 3K/5|), f = a - g²/2,
    K = <R⁴>/<R²>² = b/a² (1 for a = 0)."""
    x = np.asarray(x, float)
    a = _koyama_a(x)
    b = _koyama_b(x)
    with np.errstate(divide="ignore", invalid="ignore"):
        K = np.where(a > 0, b / (a * a), 1.0)
    root = np.sqrt(10.0 * np.abs(1.0 - 0.6 * K))
    g2 = a * root
    # f = a (1 - root/2); root -> 2 in the rod limit: 1 - root/2 = (4 - root²)/(2 (2 + root)), 4 - root² = 6K - 6
    # (for K < 5/3) without cancellation: K - 1 from the series of b - a²
    return a, g2, K


def _koyama_f(x):
    """f(x) = a - g²/2 with K - 1 = (b - a²)/a² computed without cancellation."""
    x = np.asarray(x, float)
    a, g2, K = _koyama_fg(x)
    # b - a² by series below 1 (both start with x⁴/4)
    small = x < 1.0
    xs = np.where(small, x, 0.0)
    ca = np.zeros(40)
    for k in range(2, 26):
        ca[k] = (-1.0) ** k / math.factorial(k)
    ca2 = np.polynomial.polynomial.polymul(ca, ca)[:40]
    dser = np.polynomial.polynomial.polyval(xs, _KB - ca2)
    with np.errstate(divide="ignore", invalid="ignore"):
        km1 = np.where(small, np.where(a > 0, dser / (a * a), 0.0), K - 1.0)
    root = np.sqrt(10.0 * np.abs(1.0 - 0.6 * K))
    # for 1 - 0.6 K >= 0: 4 - root² = 4 - 10 + 6K = 6 (K - 1)
    num = np.where(1.0 - 0.6 * K >= 0.0, 6.0 * km1, 4.0 - root * root)
    return a * num / (2.0 * (2.0 + root)), g2


def koyama(q, b, L):
    """S(q) of the Koyama worm (S(0) = 1): (2/X²) ∫_0^X (X - x) exp(-s² f/3) j0(s g) dx, s = q lb/2,
    X = 2L/lb. Gauss-Legendre on panels of at most one unit of the phase s g plus the exponent s² f/3, cut where
    the exponent exceeds 80."""
    q = np.atleast_1d(np.asarray(q, float))
    B, L = float(b), float(L)
    X = 2.0 * L / B
    out = np.empty(q.shape)
    # dense grid for the panel layout (shared by all q)
    xg0 = np.concatenate([[0.0], np.geomspace(min(1e-6, X * 1e-6), X, 600)])
    f0, g20 = _koyama_f(xg0)
    g0 = np.sqrt(np.maximum(g20, 0.0))
    for i, qi in enumerate(q):
        if qi == 0:
            out[i] = 1.0
            continue
        s = qi * B / 2.0
        expo = s * s * f0 / 3.0
        top = X
        if expo[-1] > 80.0:
            j = int(np.argmax(expo > 80.0))
            top = xg0[j]
        u = s * g0 + expo + np.log1p(xg0)          # activity measure (phase, decay, slow growth of the range)
        keep = xg0 <= top
        umax = float(np.interp(top, xg0[keep], u[keep])) if top < X else float(u[-1])
        npan = max(4, math.ceil(umax / 0.75))
        levels = np.linspace(0.0, umax, npan + 1)
        edges = np.interp(levels, u[keep] if top < X else u, xg0[keep] if top < X else xg0)
        edges[0], edges[-1] = 0.0, top
        edges = np.unique(edges)
        xq, wq = gauss_legendre(20, edges[:-1], edges[1:])
        xq, wq = xq.ravel(), wq.ravel()
        f, g2 = _koyama_f(xq)
        sg = s * np.sqrt(np.maximum(g2, 0.0))
        j0 = np.where(sg < 1e-8, 1.0 - sg * sg / 6.0, np.sin(sg) / np.where(sg == 0, 1.0, sg))
        phi = np.exp(-s * s * f / 3.0) * j0
        out[i] = np.sum(wq * 2.0 * (X - xq) / (X * X) * phi)
    return out


# ------------------------------------------------------------------------------------------------ form factors
def _check(p, name):
    if p[0] <= 0 or p[1] <= 0:
        raise ValueError(f"{name}: lb > 0 and L > 0 required")


def _ff(fn, needs_exvol):
    def intensity(q, p):
        q = np.asarray(q, float)
        _check(p, fn.__name__)
        S = fn(q, p[0], p[1], p[2]) if needs_exvol else fn(q, p[0], p[1])
        return p[3] * np.where(q == 0, 1.0, S).reshape(q.shape)

    intensity.__doc__ = f"p = [lb, L, {'exvol' if needs_exvol else 'dummy'}, I0]: I0 S(q), {fn.__doc__ or ''}"
    return intensity


ff_worms_ps1 = _ff(worm_ps1, True)
ff_worms_ps2 = _ff(worm_ps2, False)
ff_worms_ps3 = _ff(worm_ps3, True)
ff_kholodenko = _ff(kholodenko, False)
ff_fjc = _ff(freely_jointed, False)
ff_koyama = _ff(koyama, False)


def _red(q, p):
    return np.asarray(q, float) * abs(p[0])


_SF = "plugins/polymer_semi_flexible/sasfit_ff_"
_S_EX = {0: (1.0, 30.0, "log"), 1: (5.0, 3000.0, "log"), 2: (0.0, 1.0, "int"), 3: (0.5, 2.0, "lin")}
_S_NO = {0: (1.0, 30.0, "log"), 1: (5.0, 3000.0, "log"), 3: (0.5, 2.0, "lin")}

SPECS = [
    make_spec("ff_worms_ps1_", ff_worms_ps1, ["lb", "L", "exvol", "I0"], _S_EX, source=_SF + "worms_ps1_.c",
              reduced=_red, reduced_label="q lb", rtol=1e-9, group="ff_polymer_worm"),
    make_spec("ff_worms_ps2_", ff_worms_ps2, ["lb", "L", "dummy", "I0"], _S_NO, source=_SF + "worms_ps2_.c",
              reduced=_red, reduced_label="q lb", rtol=1e-9, group="ff_polymer_worm"),
    make_spec("ff_worms_ps3_", ff_worms_ps3, ["lb", "L", "exvol", "I0"], _S_EX, source=_SF + "worms_ps3_.c",
              reduced=_red, reduced_label="q lb", rtol=1e-8, group="ff_polymer_worm"),
    make_spec("ff_kholodenko_worm", ff_kholodenko, ["lb", "L", "dummy", "I0"], _S_NO,
              source=_SF + "kholodenko_worm.c", reduced=_red, reduced_label="q lb", rtol=1e-9, group="ff_polymer_worm"),
    make_spec("ff_freely_joined_chain_of_rods", ff_fjc, ["lb", "L", "dummy", "I0"], _S_NO,
              source=_SF + "freely_joined_chain_of_rods.c", reduced=_red, reduced_label="q lb", rtol=1e-9,
              group="ff_polymer_worm"),
    make_spec("ff_koyama_worm", ff_koyama, ["lb", "L", "dummy", "I0"], _S_NO, source=_SF + "koyama_worm.c",
              reduced=_red, reduced_label="q lb", rtol=1e-6, c_integrates=True, group="ff_polymer_worm"),
]  # fmt: skip
