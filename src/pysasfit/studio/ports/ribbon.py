"""Python ports of the SASfit plugin ribbon (ff_ribbon_mono, peaks_ribbon_w_sd).

A thin rod of length C times the cross section of a ribbon (thickness A, width B, thin-ribbon approximation A << B):
    I(q) = η² C² P_rod(qC) · (A B)² j0²(qA/2) · F(qB/2),
    F(x) = 1F2(1/2; 3/2, 2; -x²) = 2∫_0^1 (1-t) J0(2xt) dt = [∫_0^{2x} J0(t) dt - J1(2x)] / x
         = [J1(2x) + 2 Σ_{k>=1} J_{2k+1}(2x)] / x          (Neumann series ∫_0^z J0 = 2 Σ_k J_{2k+1}(z)),
the in-plane orientation average of sinc²(x cos φ). The odd-order Bessel sum is evaluated by Miller's downward
recurrence (normalised with J0 + 2 Σ J_2k = 1): exact to rounding for all x. The C code used sasfit_pfq for
qB/2 <= 3 and the asymptote 2/(qB) above (errors up to 2.3 %, still 0.15 % at qB/2 = 30).

peaks_ribbon_w_sd: Schulz (gamma) distributions of A and B (mean A, B; standard deviations σ_A, σ_B), averaged
separately. <a² j0²(qa/2)> = 2(1 - Re(1 - i q σ_A²/A)^(-A²/σ_A²))/q² (closed form, as in C). For B the C code uses
3F2(1/2, 1+k/2, 3/2+k/2; 3/2, 2; -(qσ_B²/B)²) (k = B²/σ_B²) through sasfit_pfq, with the asymptote 2B/q beyond
qRg_B = 100 or on failure. The port uses
    <b² F(qb/2)> = 2 ∫_0^1 K(w) Re[k(k+1)θ² (1 - i q θ w)^-(k+2)] dw,   θ = σ_B²/B,
    K(w) = (2/π)[ln((1+√(1-w²))/w) - √(1-w²)]    (F(x) = 2∫_0^1 K(w) cos(2xw) dw),
whose integrand has at most (k+2)/4 oscillations independent of q.
"""

from __future__ import annotations

import math

import numpy as np
from scipy import special

from .common import make_spec
from .quadrature import gauss_legendre

__all__ = ["SPECS", "f1f2", "rod"]

_NGL = 12  # Gauss-Legendre points per panel (same as the C code)


def _sinc(x):
    return np.sinc(np.asarray(x, float) / np.pi)


def rod(q, length):
    """Thin rod L² (2 Si(qL)/(qL) - j0²(qL/2)) (I(0) = L²)."""
    u = np.asarray(q, float) * length
    with np.errstate(invalid="ignore", divide="ignore"):
        g = np.where(u == 0, 1.0, special.sici(u)[0] / np.where(u == 0, 1.0, u))
    return length**2 * (2.0 * g - _sinc(u / 2.0) ** 2)


def _odd_bessel_sums(z):
    """(J1(z), Σ_{k>=1} J_{2k+1}(z)) by Miller's downward recurrence, z >= 0 (array)."""
    z = np.atleast_1d(np.asarray(z, float))
    j1 = np.empty(z.shape)
    rest = np.empty(z.shape)
    for i, zi in enumerate(z):
        if zi < 1e-3:
            j1[i] = special.j1(zi)
            rest[i] = zi**3 / 48.0 * (1.0 - zi * zi / 16.0)  # J3 + J5 + ...
            continue
        n = int(zi + 10.0 * zi ** (1.0 / 3.0) + 30)
        n += n % 2  # even start
        jp, j = 0.0, 1e-280
        norm = 0.0
        sodd = 0.0
        jone = 0.0
        for m in range(n, 0, -1):  # j = J_m (unnormalised), compute J_{m-1}
            jm = 2.0 * m / zi * j - jp
            if m % 2 == 0:
                norm += 2.0 * j
            else:
                sodd += j
                if m == 1:
                    jone = j
            jp, j = j, jm
            if abs(j) > 1e250:
                jp *= 1e-250
                j *= 1e-250
                norm *= 1e-250
                sodd *= 1e-250
                jone *= 1e-250
        norm += j  # J0
        j1[i] = jone / norm
        rest[i] = (sodd - jone) / norm
    return j1, rest


def f1f2(x):
    """F(x) = 1F2(1/2; 3/2, 2; -x²) = [J1(2x) + 2 Σ_{k>=1} J_{2k+1}(2x)]/x (1 at x = 0)."""
    x = np.asarray(x, float)
    out = np.ones(x.shape)
    xs = x.ravel()
    small = np.abs(xs) < 1e-4
    res = np.empty(xs.shape)
    res[small] = 1.0 - xs[small] ** 2 / 6.0 + xs[small] ** 4 / 60.0
    if np.any(~small):
        z = 2.0 * np.abs(xs[~small])
        j1, rest = _odd_bessel_sums(z)
        res[~small] = (j1 + 2.0 * rest) / np.abs(xs[~small])
    out = res.reshape(x.shape)
    return out


def _kern(w):
    s = np.sqrt(1.0 - w * w)
    return 2.0 / np.pi * (np.log((1.0 + s) / w) - s)


def _schulz_b2_f(q, b, sig):
    """<b² F(qb/2)> for a Schulz (gamma) distribution of b with mean b and standard deviation sig."""
    q = np.atleast_1d(np.asarray(q, float))
    k = (b / sig) ** 2
    th = sig * sig / b
    out = np.empty(q.shape)
    for i, qi in enumerate(q):
        u = qi * th
        if qi * b * math.sqrt(1.0 + 2.0 / k) < 1e-3:
            out[i] = (b * b + sig * sig) * (1.0 - qi * qi * (b * b + 2 * sig * sig) * (b * b + 3 * sig * sig) / (24 * b * b))
            continue
        # b² F(qb/2) = 2b/q + oscillating terms ~ b^(1/2) cos(qb + ...), whose Schulz averages are damped by
        # (1+u²)^-(k+1/2)/2: below 1e-17 only the smooth part <2b/q> = 2B/q remains
        if k > 1.0 and -0.5 * (k - 1.0) * math.log1p(u * u) < -39.0:
            out[i] = 2.0 * b / qi
            continue
        out_edges = _panel_edges(k, u)
        t, wt = gauss_legendre(_NGL, 0.0, 1.0)
        # first panel: geometric subpanels towards w = 0 (logarithmic singularity of K)
        w1 = out_edges[1]
        g = w1 * 0.5 ** np.arange(0, 51)
        tg, wg = gauss_legendre(_NGL, g[1:], g[:-1])
        tot = np.sum(wg * _kern(tg) * _schulz_re(k, th, qi, tg))
        # last panel: w = 1 - h τ² (K ~ (1-w)^(3/2) at w = 1)
        h = 1.0 - out_edges[-2]
        w = 1.0 - h * t**2
        tot += np.sum(wt * 2.0 * h * t * _kern(w) * _schulz_re(k, th, qi, w))
        if len(out_edges) > 3:
            tm, wm = gauss_legendre(_NGL, out_edges[1:-2], out_edges[2:-1])
            tot += np.sum(wm * _kern(tm) * _schulz_re(k, th, qi, tm))
        out[i] = 2.0 * tot
    return out


def _panel_edges(k, u):
    """Panels on [0, 1] for <b² cos(qbw)> = Re[k(k+1)θ²(1-iuw)^-(k+2)], u = qθ: at most 1.5 rad of phase
    (k+2) atan(uw) per panel, and geometric (factor 2) where the amplitude (1+u²w²)^-(k+2)/2 decays."""
    phase = (k + 2.0) * math.atan(u)
    n = int(phase / 1.5) + 1
    e = [math.tan(j * phase / n / (k + 2.0)) / u for j in range(1, n)] if u > 0 else []
    w = 1.0 / u if u > 0 else 2.0
    while w < 1.0:
        e.append(w)
        w *= 2.0
    e = sorted(x for x in set(e) if 0.0 < x < 1.0)
    return np.array([0.0] + e + [1.0]) if e else np.array([0.0, 0.5, 1.0])


def _schulz_re(k, th, q, w):
    """Re[k(k+1)θ² (1 - iqθw)^-(k+2)] = <b² cos(q b w)>."""
    u = q * th * w
    return k * (k + 1.0) * th * th * np.exp(-0.5 * (k + 2.0) * np.log1p(u * u)) * np.cos((k + 2.0) * np.arctan(u))


def ribbon_mono(q, p):
    """p = [A, dummy, B, dummy, C, dummy, eta]: ribbon thickness A, width B, length C (thin rod, A << B);
    I = η² C² P_rod(qC) (AB)² j0²(qA/2) F(qB/2), I(0) = (η A B C)²."""
    q = np.asarray(q, float)
    a, b, c, eta = p[0], p[2], p[4], p[6]
    if a < 0 or b < 0 or c < 0:
        raise ValueError("A, B, C >= 0 required")
    return eta * eta * rod(q, c) * (a * b * _sinc(q * a / 2.0)) ** 2 * f1f2(q * b / 2.0)


def ribbon_mono_v(x, p, dist):
    return p[0] * p[2] * p[4] + 0.0 * np.asarray(x, float)


def ribbon_w_sd(q, p):
    """p = [A, sigma_A, B, sigma_B, C, dummy, eta]: ribbon_mono with independent Schulz distributions of A and B
    (means A, B, standard deviations σ_A, σ_B); I(0) = η² C² (A²+σ_A²)(B²+σ_B²)."""
    q = np.asarray(q, float)
    a, sa, b, sb, c, eta = p[0], p[1], p[2], p[3], p[4], p[6]
    if a <= 0 or sa <= 0 or b < 0 or sb < 0 or c < 0:
        raise ValueError("A, sigma_A > 0 and B, sigma_B, C >= 0 required")
    qs = np.atleast_1d(q).astype(float)
    # A: <a² j0²(qa/2)> = 2(1 - Re(1 - i q θ_A)^-k_A)/q²
    ka, tha = (a / sa) ** 2, sa * sa / a
    ua = qs * tha
    with np.errstate(invalid="ignore", divide="ignore"):
        # 1 - e^L cos φ = -expm1(L) cos φ + 2 sin²(φ/2)  (no cancellation for small q)
        lg = -0.5 * ka * np.log1p(ua * ua)
        ph = ka * np.arctan(ua)
        one_m_re = -np.expm1(lg) * np.cos(ph) + 2.0 * np.sin(0.5 * ph) ** 2
        resa = np.where(qs * math.sqrt(a * a + sa * sa) < 1e-6, a * a + sa * sa, 2.0 * one_m_re / (qs * qs))
    if b == 0:
        resb = np.zeros(qs.shape)
    elif sb == 0:
        resb = b * b * f1f2(qs * b / 2.0)
    else:
        resb = _schulz_b2_f(qs, b, sb)
    out = eta * eta * rod(qs, c) * resa * resb
    return out.reshape(np.shape(q))


SPECS = [
    make_spec("ff_ribbon_mono", ribbon_mono, ["A", "dummy", "B", "dummy", "C", "dummy", "eta"],
              {0: (1.0, 5.0, "log"), 2: (5.0, 50.0, "log"), 4: (50.0, 500.0, "log"), 6: (0.5, 2.0, "lin")},
              volume=ribbon_mono_v, source="plugins/ribbon/sasfit_ff_ribbon_mono.c", rtol=1e-12,
              notes="C rewritten 2026-11-18 (was sasfit_pfq and the asymptote 2/(qB) for qB/2 > 3); checked with mpmath"),
    make_spec("peaks_ribbon_w_sd", ribbon_w_sd, ["A", "sigma_a", "B", "sigma_B", "C", "dummy", "eta"],
              {0: (1.0, 5.0, "log"), 1: (0.1, 1.0, "lin"), 2: (5.0, 50.0, "log"), 3: (0.5, 10.0, "lin"),
               4: (50.0, 500.0, "log"), 6: (0.5, 2.0, "lin")},
              source="plugins/ribbon/sasfit_peaks_ribbon_w_sd.c", rtol=1e-12,
              notes="C rewritten 2026-11-18 (was sasfit_pfq for the 3F2 and the asymptote 2B/q); checked with mpmath"),
]  # fmt: skip
