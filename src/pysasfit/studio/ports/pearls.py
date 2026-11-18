"""Python ports of the SASfit plugins linear_pearls and pearl_necklace.

pearl_necklace (Schweins & Huber): N spheres (radius R, scattering weight beta_s) joined by N-1 thin rods (length l,
weight beta_r), freely jointed; I(0) = (m beta_r + N beta_s)² with m rods (SASfit convention I(0) = (Δη V)²;
the C code was normalised to P(0) = 1 until 2026-11-18). The C closed forms of the geometric sums, e.g. n/(1-γ) - γ(1-γ^n)/(1-γ)²
with γ = sin(qA)/(qA), cancel catastrophically for qA -> 0 (NaN at q = 0). The ports use the finite sums
(integer numbers of pearls) or a binomial series in 1-γ (pearl_necklace_mod, real Np); the C code was changed the
same way on 2026-11-18.
"""

from __future__ import annotations

import numpy as np
from scipy import special

from .common import make_spec, phi_sphere
from .quadrature import gauss_legendre

__all__ = ["SPECS"]


def _sinc(x):
    return np.sinc(np.asarray(x, float) / np.pi)


# --------------------------------------------------------------------------------------------- linear_pearls


def _linear_pearls_kernel(q, nn, p):
    r, sep, eta, eta_sol = p[0], p[1], p[3], p[4]
    v = 4.0 / 3.0 * np.pi * r**3
    if nn < 1 or v == 0:
        return np.zeros(q.shape)
    d = sep + 2.0 * r
    s = np.full(q.shape, float(nn))
    for k in range(1, int(nn)):
        s += 2.0 * (nn - k) * _sinc(q * d * k)
    return s * ((eta - eta_sol) * v * phi_sphere(q * r)) ** 2


def linear_pearls(q, p):
    """p = [R, edge_sep, num_pearls, eta, eta_sol]: N spheres on a straight line, centre distance edge_sep + 2R;
    S(q)(Δη V ψ(qR))² with S = N + 2 Σ_k (N-k) sin(qdk)/(qdk), so I(0) = (Δη N V)² (SASfit convention; the C code
    divided by N V, the SASview convention, until 2026-11-18); non-integer N interpolates linearly."""
    q = np.asarray(q, float)
    r, sep, n = p[0], p[1], p[2]
    if r < 0 or sep < 0 or n < 0:
        raise ValueError("R, edge_sep, num_pearls >= 0 required")
    nn = np.floor(n)
    w = n - nn
    out = (1.0 - w) * _linear_pearls_kernel(q, nn, p)
    if w > 0:
        out = out + w * _linear_pearls_kernel(q, nn + 1.0, p)
    return out


def linear_pearls_v(x, p, dist):
    return p[2] * 4.0 / 3.0 * np.pi * p[0] ** 3 + 0.0 * np.asarray(x, float)


# -------------------------------------------------------------------------------------------- pearl_necklace


def _lam(x):
    """Si(x)/x (1 at x = 0)."""
    x = np.asarray(x, float)
    with np.errstate(invalid="ignore", divide="ignore"):
        v = special.sici(x)[0] / x
    return np.where(x == 0, 1.0, v)


def _lam_ext(qr, ql):
    """(Si(qR+ql) - Si(qR))/ql = mean of sin(t)/t over [qR, qR+ql] (Gauss-Legendre for ql < 2)."""
    qr, ql = np.broadcast_arrays(np.asarray(qr, float), np.asarray(ql, float))
    t, w = gauss_legendre(16, qr, qr + ql)
    small = np.sum(w * _sinc(t), axis=-1)
    with np.errstate(invalid="ignore", divide="ignore"):
        big = (special.sici(qr + ql)[0] - special.sici(qr)[0]) / ql
        small = small / ql
    out = np.where(ql < 2.0, small, big)
    return np.where(ql == 0, _sinc(qr), out)


def _u(m, g):
    """Σ_{k=0}^{m-1} (m-k) γ^k (0 for m <= 0)."""
    out = np.zeros(np.shape(g))
    gk = np.ones(np.shape(g))
    for k in range(int(m)):
        out += (m - k) * gk
        gk = gk * g
    return out


def _check(p, np_min):
    r, l, br, bs, n = p[0], p[3], p[4], p[5], p[6]
    if r < 0 or l < 0 or br < 0 or bs < 0:
        raise ValueError("R, l, beta_r, beta_s >= 0 required")
    if n < np_min:
        raise ValueError(f"Np >= {np_min} required")


def _pn_original(q, p, n):
    r, l, br, bs = p[0], p[3], p[4], p[5]
    m = n - 1
    a = l + 2.0 * r
    f = phi_sphere(q * r)
    g = _sinc(q * a)
    j = _sinc(q * l / 2.0)
    beta = _lam_ext(q * r, q * l)
    sss = 2.0 * bs**2 * f * f * (n / 2.0 + g * _u(n - 1, g))
    srr = br**2 * (m * (2.0 * _lam(q * l) - j * j) + 2.0 * beta**2 * _u(m - 1, g))
    srs = 4.0 * bs * br * beta * f * _u(m, g)
    return sss + srr + srs


def pearl_necklace(q, p):
    """p = [R, dummy, dummy, l, beta_r, beta_s, Np, dummy]: pearl necklace (spheres R, rods l, A = l + 2R);
    non-integer Np interpolates linearly between floor(Np) and floor(Np)+1 pearls."""
    q = np.asarray(q, float)
    _check(p, 1.0)
    n = int(np.floor(p[6]))
    w = p[6] - n
    out = (1.0 - w) * _pn_original(q, p, n)
    if w != 0:
        out = out + w * _pn_original(q, p, n + 1)
    return out


def _pn_ext(q, p, f, g_int, n):
    """N pearls; between neighbours g freely jointed rods of length l. The first rod starts on the surface of
    pearl k with its axis through the centre of pearl k, the last rod ends on the surface of pearl k+1 with its
    axis through that centre: bonds R+l, l, ..., l, l+R (one bond l+2R for g = 1, the original model). Freely
    jointed bonds -> the pair terms are products of sinc(q L_bond); summed by a backward recursion, O(N g)."""
    r, l, br, bs = p[0], p[3], p[4], p[5]
    lam = _lam(q * l)
    be = _lam_ext(q * r, q * l)
    rod = 2.0 * lam - _sinc(q * l / 2.0) ** 2
    if g_int == 1:
        bonds = [(l + 2.0 * r, be, be)]
    else:
        bonds = [(r + l, be, lam)] + [(l, lam, lam)] * (g_int - 2) + [(l + r, lam, be)]
    trans = [_sinc(q * lb) for lb, _, _ in bonds]
    s = n * bs**2 * f * f + g_int * (n - 1) * br**2 * rod
    acc = np.zeros(np.shape(q))
    for k in range(n - 1, -1, -1):
        s = s + 2.0 * bs * f * acc
        acc = acc + bs * f
        if k > 0:
            for j in range(g_int - 1, -1, -1):
                _, b_in, b_out = bonds[j]
                s = s + 2.0 * br * b_out * acc
                acc = br * b_in + trans[j] * acc
    return s


def pearl_necklace_ext(q, p):
    """p = [R, dummy, dummy, l, beta_r, beta_s, Np, g]: extended pearl necklace with g freely jointed rods (length
    l each) between neighbouring pearls, the end rods radial to their pearls; g = 1 is pearl_necklace. I(0) = (g(Np-1)β_r + Np β_s)²;
    non-integer Np and g interpolate bilinearly. (Rewritten 2026-11-18: the former C formula treated the rods as
    length l+2R in the rod-rod terms and did not reduce to pearl_necklace for g = 1.)"""
    q = np.asarray(q, float)
    _check(p, 1.0)
    if p[7] < 1:
        raise ValueError("g >= 1 required")
    n = int(np.floor(p[6]))
    w = p[6] - n
    g = int(np.floor(p[7]))
    wg = p[7] - g
    f = phi_sphere(q * p[0])
    res = (1 - wg) * (1 - w) * _pn_ext(q, p, f, g, n)
    if wg != 0:
        res = res + wg * (1 - w) * _pn_ext(q, p, f, g + 1, n)
    if w != 0:
        res = res + (1 - wg) * w * _pn_ext(q, p, f, g, n + 1)
        if wg != 0:
            res = res + wg * w * _pn_ext(q, p, f, g + 1, n + 1)
    return res


def _geo_series(x, j, s):
    """j = 1: x/ε - s(1-|s|^x)/ε², j = 0: x/ε - (1-|s|^x)/ε² (ε = 1-s, real x): the binomial series
    Σ_k (-1)^k C(x+j, k+2) ε^k for small ε (where the closed form cancels), the closed form otherwise."""
    s = np.asarray(s, float)
    eps = 1.0 - s
    with np.errstate(invalid="ignore", divide="ignore"):
        closed = (
            x / eps - (s if j == 1 else 1.0) * (1.0 - np.power(np.abs(s), x)) / eps**2
        )
    use = (np.abs(eps) < 0.25) & (np.abs(x * eps) < 0.5)
    if np.any(use):
        e = np.where(use, eps, 0.0)
        y = x + j
        c = y * (y - 1.0) / 2.0  # C(y, 2)
        series = np.zeros(e.shape)
        ek = np.ones(e.shape)
        for k in range(40):
            series += c * ek
            c *= -(y - k - 2.0) / (k + 3.0)
            ek = ek * e
        closed = np.where(use, series, closed)
    return closed


def pearl_necklace_mod(q, p):
    """p = [R, dummy, dummy, l, beta_r, beta_s, Np, dummy]: pearl necklace with real Np (analytic continuation of
    the geometric sums, |γ|^Np)."""
    q = np.asarray(q, float)
    _check(p, 0.0)
    r, l, br, bs, n = p[0], p[3], p[4], p[5], p[6]
    a = l + 2.0 * r
    f = phi_sphere(q * r)
    la = _lam(q * l)
    j = _sinc(q * l / 2.0)
    s = _sinc(q * a)
    beta = _lam_ext(q * r, q * l)
    mr = n - 1.0
    sp = 2.0 * bs**2 * f * f * (_geo_series(n, 1, s) - n / 2.0)
    sr = br**2 * (mr * (2.0 * la - j * j) + 2.0 * beta**2 * _geo_series(mr, 0, s))
    sm = 4.0 * br * beta * bs * f * _geo_series(n - 1.0, 1, s)
    return sp + sr + sm


def _qa(q, p):
    return np.asarray(q, float) * (p[3] + 2.0 * p[0])


_PN = "plugins/pearl_necklace/sasfit_ff_pearl_necklace_"
_PN_SAMPLE = {
    0: (1.0, 20.0, "log"),
    3: (2.0, 50.0, "log"),
    4: (0.1, 2.0, "lin"),
    5: (0.5, 5.0, "lin"),
}
SPECS = [
    make_spec("ff_linear_pearls", linear_pearls, ["R", "edge_sep", "num_pearls", "eta", "eta_sol"],
              {0: (1.0, 20.0, "log"), 1: (0.0, 20.0, "lin"), 2: (1.0, 12.0, "lin"), 3: (0.5, 2.0, "lin"), 4: (0.0, 0.4, "lin")},
              volume=linear_pearls_v, source="plugins/linear_pearls/sasfit_ff_linear_pearls.c", rtol=1e-11),
    make_spec("ff_pearl_necklace", pearl_necklace, ["R", "dummy", "dummy", "l", "beta_r", "beta_s", "Np", "dummy"],
              {**_PN_SAMPLE, 6: (1.0, 15.0, "lin")}, source=_PN + "original.c", reduced=_qa, reduced_label="q (l+2R)",
              rtol=1e-12),
    make_spec("ff_pearl_necklace_ext", pearl_necklace_ext, ["R", "dummy", "dummy", "l", "beta_r", "beta_s", "Np", "g"],
              {**_PN_SAMPLE, 6: (1.0, 15.0, "lin"), 7: (1.0, 4.0, "lin")}, source=_PN + "extended.c", reduced=_qa,
              reduced_label="q (l+2R)", rtol=1e-12),
    make_spec("ff_pearl_necklace_mod", pearl_necklace_mod, ["R", "dummy", "dummy", "l", "beta_r", "beta_s", "Np", "dummy"],
              {**_PN_SAMPLE, 6: (1.0, 15.0, "lin")}, source=_PN + "modified.c", reduced=_qa, reduced_label="q (l+2R)",
              rtol=1e-12),
]  # fmt: skip
