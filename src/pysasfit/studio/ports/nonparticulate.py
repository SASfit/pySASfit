"""Python ports of the SASfit plugin non_particulate (scattering laws without a particle form factor).

C deviations fixed on 2026-11-18: the Beaucage exponential/power laws gave 0/0 = NaN at q = 0 (erf(qRg k/√6)³/q
→ 0, the port and the fixed C use that limit); spinodal2 printed a, b, c on every call (sasfit_out removed); the
generalized Guinier-Porod law returned NaN when Q2² <= 0 (now a domain error, (s1-s2)/(Rg2²/(3-s2)-Rg1²/(3-s1)) > 0
or s1 = s2 required); def_g_dab passed q through param->p[MAXPAR-1] (kept: the port integrates directly).
"""

from __future__ import annotations

import numpy as np
from scipy import special

from .common import make_spec
from .quadrature import gauss_legendre

__all__ = ["SPECS"]

_S6 = np.sqrt(6.0)


def _erf3_over_q(q, r, k):
    """erf(q r k/√6)³ / q with the limit 0 at q = 0."""
    with np.errstate(divide="ignore", invalid="ignore"):
        t = special.erf(q * r * k / _S6) ** 3 / q
    return np.where(q == 0, 0.0, t)


def _pow(x, p):
    """x**p with 0**0 = 1 (as C pow)."""
    return np.power(x, p)


def beaucage_exppowlaw(q, p):
    """p = [G, B, Gs, Bs, Rg, Rsub, Rs, P, Ps]:
    G e^(-q²Rg²/3) + B e^(-q²Rsub²/3) (erf³(qRg/√6)/q)^P + Gs e^(-q²Rs²/3) + Bs (erf³(qRs/√6)/q)^Ps."""
    q = np.asarray(q, float)
    G, B, GS, BS, RG, RSUB, RS, P, PS = p[:9]
    if min(G, B, GS, BS, RG, RSUB, RS, P, PS) < 0:
        raise ValueError("all parameters >= 0 required")
    tmp = _erf3_over_q(q, RG, 1.0)
    qsub = _erf3_over_q(q, RS, 1.0)
    return (G * np.exp(-((q * RG) ** 2) / 3.0) + B * np.exp(-((q * RSUB) ** 2) / 3.0) * _pow(tmp, P)
            + GS * np.exp(-((q * RS) ** 2) / 3.0) + BS * _pow(qsub, PS))  # fmt: skip


def beaucage_exppowlaw_2(q, p):
    """p = [G, B, Rg_i, Rg_i+1, k, P]: G e^(-q²Rg_i²/3) + B e^(-q²Rg_i+1²/3) (erf³(q Rg_i k/√6)/q)^P."""
    q = np.asarray(q, float)
    G, B, RI, RI1, K, P = p[:6]
    if min(G, B, RI, RI1, K, P) < 0:
        raise ValueError("all parameters >= 0 required")
    tmp = _erf3_over_q(q, RI, K)
    return G * np.exp(-((q * RI) ** 2) / 3.0) + B * np.exp(
        -((q * RI1) ** 2) / 3.0
    ) * _pow(tmp, P)


def broad_peak(q, p):
    """p = [I0, xi, q0, m, p]: I0 / (1 + (|q - q0| xi)^m)^p."""
    q = np.asarray(q, float)
    return p[0] / np.power(1.0 + np.power(np.abs(q - p[2]) * p[1], p[3]), p[4])


def debye_anderson_brumberger(q, p):
    """p = [xi, dummy, dummy, eta]: (π η (2ξ)³)² / (1 + q²ξ²)²."""
    q = np.asarray(q, float)
    xi, eta = p[0], p[3]
    return (eta * (2.0 * xi) ** 3 * np.pi) ** 2 / ((q * xi) ** 2 + 1.0) ** 2


def _g_dab_pref(xi, h, eta):
    """((2ξ)³ Γ(H+3/2)/Γ(H) η)² π³ (Γ(H+3/2)/Γ(H) = gsl_sf_poch(H, 3/2))."""
    poch = special.gamma(h + 1.5) * special.rgamma(h)
    return ((2.0 * xi) ** 3 * poch * eta) ** 2 * np.pi**3


def g_dab(q, p):
    """p = [xi, H, dummy, eta]: ((2ξ)³ Γ(H+3/2)/Γ(H) η)² π³ / (1 + q²ξ²)^(3/2+H)."""
    q = np.asarray(q, float)
    xi, h, eta = p[0], p[1], p[3]
    if xi < 0 or h <= -1.5:
        raise ValueError("xi >= 0 and H > -1.5 required")
    return _g_dab_pref(xi, h, eta) / np.power(1.0 + (q * xi) ** 2, 1.5 + h)


def def_g_dab(q, p):
    """p = [xi, H, epsilon, eta]: gDAB orientation-averaged over an affine (uniaxial) stretch ε:
    ∫_0^1 I_gDAB(q √(1 - u² + ε² u²)) du."""
    q = np.asarray(q, float)
    xi, h, eps, eta = p[0], p[1], p[2], p[3]
    if xi < 0 or h <= -1.5 or eps <= 0:
        raise ValueError("xi >= 0, H > -1.5 and epsilon > 0 required")
    if eps == 1:
        return g_dab(q, p)
    u, w = gauss_legendre(96, 0.0, 1.0)
    qe = q[..., None] * np.sqrt(1.0 + (eps * eps - 1.0) * u * u)
    return _g_dab_pref(xi, h, eta) * np.sum(
        w / np.power(1.0 + (qe * xi) ** 2, 1.5 + h), axis=-1
    )


def extended_guinier_law(q, p):
    """p = [I0, a, Ra]: a = 0: I0 e^(-Ra²q²/3); else a π q^(-a) I0 e^(-Ra²q²/(3-a)) (a = 1 rod, 2 lamella)."""
    q = np.asarray(q, float)
    i0, a, ra = p[0], p[1], p[2]
    if a < 0 or a >= 3 or ra < 0:
        raise ValueError("0 <= a < 3 and Ra >= 0 required")
    if a == 0:
        return i0 * np.exp(-((ra * q) ** 2) / 3.0)
    with np.errstate(divide="ignore"):
        return a * np.pi * np.power(q, -a) * i0 * np.exp(-((ra * q) ** 2) / (3.0 - a))


def _ggp_q(p):
    """Q1, Q2, G1, D of the generalized Guinier-Porod law (Hammouda)."""
    g2, s2, rg2, s1, rg1, m = p[:6]
    if min(g2, s2, rg2, s1, rg1, m) < 0 or s1 >= 3 or s2 >= 3 or s2 > s1 or m < s1:
        raise ValueError("parameters >= 0, s2 <= s1 < 3 and m >= s1 required")
    den = 2.0 / (3.0 - s2) * rg2 * rg2 - 2.0 / (3.0 - s1) * rg1 * rg1
    if s1 == s2:
        q2 = 0.0
    elif den <= 0:
        raise ValueError("Rg2²/(3-s2) > Rg1²/(3-s1) required for s2 < s1")
    else:
        q2 = np.sqrt((s1 - s2) / den)
    g1 = (
        g2
        * np.exp(q2 * q2 * (rg1 * rg1 / (3.0 - s1) - rg2 * rg2 / (3.0 - s2)))
        * q2 ** (s1 - s2)
    )
    q1 = np.sqrt((m - s1) * (3.0 - s1) / 2.0) / rg1
    d = g1 * np.exp(-q1 * q1 * rg1 * rg1 / (3.0 - s1)) * q1 ** (m - s1)
    return q1, q2, g1, d


def generalized_guinier_porod_law(q, p):
    """p = [G2, s2, Rg2, s1, Rg1, m]: G2 q^-s2 e^(-q²Rg2²/(3-s2)) (q <= Q2), G1 q^-s1 e^(-q²Rg1²/(3-s1))
    (Q2 < q < Q1), D q^-m (q >= Q1); G1, D, Q1, Q2 make the law and its slope continuous."""
    q = np.asarray(q, float)
    g2, s2, rg2, s1, rg1, m = p[:6]
    q1, q2, g1, d = _ggp_q(p)
    with np.errstate(divide="ignore", invalid="ignore"):
        low = g2 / np.power(q, s2) * np.exp(-q * q * rg2 * rg2 / (3.0 - s2))
        mid = g1 / np.power(q, s1) * np.exp(-q * q * rg1 * rg1 / (3.0 - s1))
        high = d / np.power(q, m)
    return np.where(q <= q2, low, np.where(q < q1, mid, high))


def ornstein_zernike(q, p):
    """p = [I0, xi]: I0 / (1 + q²ξ²)."""
    q = np.asarray(q, float)
    return p[0] / ((q * p[1]) ** 2 + 1.0)


def spinodal(q, p):
    """p = [Imax, Qmax, gamma]: Imax x² (1 + γ/2) / (γ/2 + x^(2+γ)), x = q/Qmax."""
    q = np.asarray(q, float)
    imax, qmax, g = p[0], p[1], p[2]
    if imax < 0 or qmax < 0:
        raise ValueError("Imax >= 0 and Qmax >= 0 required")
    x = q / qmax
    c = g / 2.0 + np.power(x, 2.0 + g)
    if np.any(c == 0):
        raise ValueError("gamma/2 + (q/Qmax)^(2+gamma) == 0")
    return imax * x * x * (1.0 + g / 2.0) / c


def _spinodal2_g(phi, model):
    if model == 1:
        return 1.0 / (4.0 * np.pi * phi * (1.0 - phi))
    if model == 2:
        return 1.0 / (4.0 * np.sqrt(np.pi * phi) * (1.0 - phi))
    if model == 3:
        return 1.0 / (
            2.0 * (4.0 * np.pi / 3.0) ** (2.0 / 3.0) * phi ** (1.0 / 3.0) * (1.0 - phi)
        )
    return phi


def spinodal2(q, p):
    """p = [Imax, Qmax, phi/gamma, d, model]: Imax a x⁴/(x⁴+c) b/(b+(x²-1+d)²), x = q/Qmax,
    b = (2g(1-d)/(1-g²))², c = d/(b-d(1-d)), a = (1+c)(1+d²/b); g = gamma (model 0) or g(phi) (models 1-3)."""
    q = np.asarray(q, float)
    imax, qmax, phi, d = p[0], p[1], p[2], p[3]
    model = int(np.floor(p[4] + 0.5)) if p[4] >= 0 else -int(np.floor(-p[4] + 0.5))
    if qmax < 0 or phi <= 0:
        raise ValueError("Qmax >= 0 and phi/gamma > 0 required")
    g = _spinodal2_g(phi, model)
    if g == 1:
        return imax + 0.0 * q
    x = q / qmax
    b = (2.0 * g * (1.0 - d) / (1.0 - g * g)) ** 2
    c = d / (b - d * (1.0 - d))
    a = (1.0 + c) * (1.0 + d * d / b)
    tmp = a * x**4 / (x**4 + c) * b / (b + (x * x - 1.0 + d) ** 2)
    if np.any(tmp < 0):
        raise ValueError("I(Q) < 0, choose a value for gamma closer to 1")
    return tmp * imax


def teubner_strey(q, p):
    """p = [eta, xi, d]: 8π η²/ξ / (a² - 2 b q² + q⁴), a = k² + 1/ξ², b = k² - 1/ξ², k = 2π/d."""
    q = np.asarray(q, float)
    eta, xi, d = p[0], p[1], p[2]
    if eta < 0 or xi < 0 or d < 0:
        raise ValueError("eta, xi, d >= 0 required")
    k = 2.0 * np.pi / d
    a = k * k + 1.0 / (xi * xi)
    b = k * k - 1.0 / (xi * xi)
    c = a * a - 2.0 * b * q * q + q**4
    if np.any(c == 0):
        raise ValueError("c == 0")
    return eta * eta * 8.0 * np.pi / xi / c


def _qxi(i):
    def red(q, p):
        return np.asarray(q, float) * abs(p[i])

    return red


_NP = "plugins/non_particulate/sasfit_ff_"
_POS = (0.1, 3.0, "lin")
SPECS = [
    make_spec("ff_beaucage_exppowlaw", beaucage_exppowlaw, ["G", "B", "Gs", "Bs", "Rg", "Rsub", "Rs", "P", "Ps"],
              {0: (0.1, 10.0, "log"), 1: (1e-3, 1.0, "log"), 2: (0.1, 10.0, "log"), 3: (1e-3, 1.0, "log"),
               4: (2.0, 50.0, "log"), 5: (0.0, 5.0, "lin"), 6: (0.5, 5.0, "log"), 7: (1.0, 4.0, "lin"), 8: (1.0, 4.0, "lin")},
              source=_NP + "beaucage_exppowlaw.c", reduced=_qxi(4), reduced_label="q Rg", rtol=1e-11),
    make_spec("ff_beaucage_exppowlaw_2", beaucage_exppowlaw_2, ["G", "B", "Rg_i", "Rg_i+1", "k", "P"],
              {0: (0.1, 10.0, "log"), 1: (1e-3, 1.0, "log"), 2: (2.0, 50.0, "log"), 3: (0.0, 5.0, "lin"),
               4: (0.8, 1.2, "lin"), 5: (1.0, 4.0, "lin")},
              source=_NP + "beaucage_exppowlaw_2.c", reduced=_qxi(2), reduced_label="q Rg_i", rtol=1e-11),
    make_spec("ff_broad_peak", broad_peak, ["I0", "xi", "q0", "m", "p"],
              {0: _POS, 1: (1.0, 50.0, "log"), 2: (0.01, 1.0, "log"), 3: (1.0, 4.0, "lin"), 4: (0.5, 2.0, "lin")},
              source=_NP + "broad_peak.c", reduced=_qxi(1), reduced_label="q ξ", rtol=1e-12),
    make_spec("ff_debye_anderson_brumberger", debye_anderson_brumberger, ["xi", "dummy", "dummy", "eta"],
              {0: (1.0, 50.0, "log"), 3: _POS}, source=_NP + "debye_anderson_brumberger.c", reduced=_qxi(0),
              reduced_label="q ξ", rtol=1e-12),
    make_spec("ff_g_dab", g_dab, ["xi", "H", "dummy", "eta"], {0: (1.0, 50.0, "log"), 1: (-1.4, 2.0, "lin"), 3: _POS},
              source=_NP + "g_dab.c", reduced=_qxi(0), reduced_label="q ξ", rtol=1e-11),
    make_spec("ff_def_g_dab", def_g_dab, ["xi", "H", "epsilon", "eta"],
              {0: (1.0, 50.0, "log"), 1: (-1.4, 2.0, "lin"), 2: (0.3, 3.0, "log"), 3: _POS},
              source=_NP + "def_g_dab.c", reduced=_qxi(0), reduced_label="q ξ", rtol=1e-9, c_integrates=True),
    make_spec("ff_extended_guinier_law", extended_guinier_law, ["I0", "a", "Ra"],
              {0: _POS, 1: (0.1, 2.9, "lin"), 2: (1.0, 50.0, "log")}, source=_NP + "extended_guinier_law.c",
              reduced=_qxi(2), reduced_label="q Ra", rtol=1e-12),
    make_spec("ff_generalized_guinier_porod_law", generalized_guinier_porod_law, ["G2", "s2", "Rg2", "s1", "Rg1", "m"],
              {0: _POS, 1: (0.0, 1.0, "lin"), 2: (10.0, 50.0, "log"), 3: (1.2, 2.5, "lin"), 4: (1.0, 3.0, "lin"),
               5: (3.0, 4.5, "lin")},
              source=_NP + "generalized_guinier_porod_law.c", reduced=_qxi(2), reduced_label="q Rg2", rtol=1e-11),
    make_spec("ff_ornstein_zernike", ornstein_zernike, ["I0", "xi"], {0: _POS, 1: (1.0, 50.0, "log")},
              source=_NP + "ornstein_zernike.c", reduced=_qxi(1), reduced_label="q ξ", rtol=1e-13),
    make_spec("ff_spinodal", spinodal, ["Imax", "Qmax", "gamma"], {0: _POS, 1: (0.05, 1.0, "log"), 2: (0.5, 4.0, "lin")},
              source=_NP + "spinodal.c", rtol=1e-12),
    make_spec("ff_spinodal2", spinodal2, ["Imax", "Qmax", "phi/gamma", "d", "model to calculate gamma(phi)"],
              {0: _POS, 1: (0.05, 1.0, "log"), 2: (0.5, 0.95, "lin"), 3: (0.0, 0.3, "lin")},
              source=_NP + "spinodal2.c", rtol=1e-12, notes="compared with model 0"),
    make_spec("ff_teubner_strey", teubner_strey, ["eta", "xi", "d"], {0: _POS, 1: (1.0, 50.0, "log"), 2: (2.0, 50.0, "log")},
              source=_NP + "teubner_strey.c", reduced=_qxi(1), reduced_label="q ξ", rtol=1e-12),
]  # fmt: skip
