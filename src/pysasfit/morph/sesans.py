"""SESANS in pySASmorph: the projected correlation function and its inversion.

Spin-echo SANS measures (Andersson et al., J. Appl. Cryst. 41 (2008) 868)

    G(delta) - G(0) = 1/(2 pi) integral I(Q) [J0(Q delta) - 1] Q dQ
                    = integral [C(sqrt(delta^2 + x^2)) - C(|x|)] dx,

with C(r) the correlation function of the scattering length density in the units of the data
(I(Q) = 4 pi integral C(r) r^2 sin(Qr)/(Qr) dr; I in 1/cm, lengths in the unit of the file:
G in [length]^-2 cm^-1, as :func:`pysasfit.core.sesans_io.load_sesans` returns it).
The real-space projection is the Abel transform

    integral C(sqrt(delta^2 + x^2)) dx = 2 integral_delta^inf C(r) r / sqrt(r^2 - delta^2) dr.

:func:`projection_matrix` evaluates it *exactly* for C piecewise linear on an r grid (the integrals
of (a + b r) r / sqrt(r^2 - delta^2) are elementary, the square-root singularity is handled
analytically). :func:`invert_sesans` solves G = A C for C on an r grid with a second-
difference (Tikhonov) penalty on nodes that follow the spacing of the data, the regularisation chosen by
the discrepancy principle (generalised cross-validation without errors); the
normalised gamma(r) = C(r)/C(0) of a SESANS curve then enters the joint fit of g(r) of
:mod:`pysasfit.morph.multiphase` like the gamma of a scattering curve, on the r range the spin-echo
lengths cover. The forward model of a SESANS curve is the projection of the level-cut covariance
C(g(r)), with one scale factor fitted.

For a periodic voxel structure :func:`voxel_correlation` gives C(r) from the autocorrelation of the
SLD map (radial average, r < L/2).
"""

from __future__ import annotations

import numpy as np

__all__ = ["projection_matrix", "project", "invert_sesans", "SESANSInversion", "voxel_correlation",
           "dab_sesans"]


def _F1(r, d):
    return np.sqrt(np.maximum(r * r - d * d, 0.0))


def _F2(r, d):
    s = np.sqrt(np.maximum(r * r - d * d, 0.0))
    with np.errstate(divide="ignore", invalid="ignore"):
        lg = np.where(d > 0, np.log(np.where(r + s > 0, r + s, 1.0)), 0.0)
    return 0.5 * (r * s + d * d * lg)


def _abel_rows(delta, r):
    """P[i, j] = integral_{r >= delta_i} phi_j(r) r / sqrt(r^2 - delta_i^2) dr for the hat functions phi_j
    on the nodes r (C = 0 beyond r[-1])."""
    delta = np.atleast_1d(np.asarray(delta, float))[:, None]
    r = np.asarray(r, float)
    a, b = r[:-1][None, :], r[1:][None, :]
    h = b - a
    lo = np.maximum(a, delta)
    act = b > lo
    lo = np.where(act, lo, b)
    I1 = _F1(b, delta) - _F1(lo, delta)
    # F2 differences: subtract the log of the lower end inside to keep precision
    I2 = _F2(b, delta) - _F2(lo, delta)
    I1 = np.where(act, I1, 0.0)
    I2 = np.where(act, I2, 0.0)
    P = np.zeros((delta.shape[0], r.size))
    P[:, :-1] += (b * I1 - I2) / h
    P[:, 1:] += (I2 - a * I1) / h
    return P


def projection_matrix(delta, r):
    """A with (A @ C)[i] = G(delta_i) - G(0) for C piecewise linear on the nodes r (C = 0 beyond r[-1])."""
    return 2.0 * (_abel_rows(delta, r) - _abel_rows([0.0], r))


def project(delta, r, C):
    """G(delta) - G(0) of the correlation function C(r)."""
    return projection_matrix(delta, r) @ np.asarray(C, float)


def dab_sesans(delta, xi, C0=1.0):
    """Exact G(delta) - G(0) of C(r) = C0 exp(-r/xi): 2 C0 [delta K1(delta/xi) - xi]."""
    from scipy.special import k1
    d = np.asarray(delta, float)
    with np.errstate(invalid="ignore"):
        out = np.where(d > 0, d * k1(np.where(d > 0, d, 1.0) / xi), xi)
    return 2.0 * C0 * (out - xi)


class SESANSInversion:
    """Result of :func:`invert_sesans`: C on the nodes, its uncertainty, the fit."""

    def __init__(self, r, C, dC, delta, G, dG, G_fit, alpha, chi2):
        self.r, self.C, self.dC = r, C, dC
        self.delta, self.G, self.dG, self.G_fit = delta, G, dG, G_fit
        self.alpha, self.chi2 = alpha, chi2

    @property
    def C0(self):
        return float(self.C[0])

    def gamma(self, r):
        """C(r)/C(0): smooth (PCHIP) interpolation of the nodes, 0 beyond the last node. A smooth curve matters:
        kinks in g(r) would show up as spurious intensity at high q."""
        from scipy.interpolate import PchipInterpolator

        r = np.asarray(r, float)
        out = PchipInterpolator(self.r, self.C, extrapolate=False)(np.minimum(r, self.r[-1]))
        return np.where(r <= self.r[-1], out, 0.0) / self.C0

    @property
    def saturated(self):
        """True if G has levelled off at the largest spin-echo lengths (structure inside the delta range)."""
        n = max(3, len(self.G) // 6)
        tail = self.G[-n:]
        return bool(np.ptp(tail) < 0.1 * np.ptp(self.G) + 2 * np.median(self.dG[-n:]))


def _nodes(d, n_nodes, r_max_factor):
    """r = 0 and n_nodes - 1 nodes that follow the density of the spin-echo lengths: geometric for log-spaced
    data, about uniform for linearly spaced data (instruments that scan delta linearly). Nodes much denser
    than the data at small r leave C there undetermined (oscillating, even C(0) < 0) under a penalty on the
    node index."""
    m = n_nodes - 1
    lo, hi = d[0] / 3.0, r_max_factor * d[-1]
    # spin-echo lengths as a function of their rank, extended to lo and hi
    t = np.concatenate([[lo], d, [hi]])
    t = np.maximum.accumulate(t)
    u = np.linspace(0.0, 1.0, t.size)
    x = np.unique(np.interp(np.linspace(0.0, 1.0, m), u, t))
    return np.concatenate([[0.0], x])


def invert_sesans(delta, G, dG=None, n_nodes=60, alphas=None, r_max_factor=1.2):
    """C(r) from G(delta) - G(0) by Tikhonov-regularised Abel inversion.

    Nodes: r = 0 and nodes from delta_min/3 to ``r_max_factor`` * delta_max that follow the spacing of the data
    (:func:`_nodes`; C = 0 beyond). Penalty: second differences of C over the nodes; alpha over ``alphas`` (default 10^-8 ...
    10^4 relative to the scale of A^T W A) by the discrepancy principle (largest alpha with chi2 <= 1) when
    errors are given, else by generalised cross-validation.
    """
    d = np.asarray(delta, float)
    G = np.asarray(G, float)
    m = np.isfinite(d) & np.isfinite(G) & (d > 0)
    s = np.ones_like(G) if dG is None else np.asarray(dG, float)
    s = np.where(np.isfinite(s) & (s > 0), s, np.nan)
    if np.all(np.isnan(s[m])):
        s = np.full_like(G, np.nanstd(G[m]) * 0.01 + 1e-30)
    s = np.where(np.isnan(s), np.nanmedian(s[m]), s)
    o = np.argsort(d[m])
    d, G, s = d[m][o], G[m][o], s[m][o]
    if d.size < 5:
        raise ValueError("at least 5 SESANS points are needed")
    r = _nodes(d, n_nodes, r_max_factor)
    A = projection_matrix(d, r)
    W = 1.0 / s
    Aw = A * W[:, None]
    yw = G * W
    D = np.diff(np.eye(r.size), 2, axis=0)
    # pin C to zero smoothly at the far end
    D = np.vstack([D, np.eye(r.size)[-1:] * 3.0])
    AtA = Aw.T @ Aw
    DtD = D.T @ D
    Aty = Aw.T @ yw
    scale = np.trace(AtA) / np.trace(DtD)
    alphas = scale * np.logspace(-8, 4, 49) if alphas is None else np.asarray(alphas, float)
    N = d.size
    sols = []
    for al in alphas:
        try:
            Minv = np.linalg.inv(AtA + al * DtD)
        except np.linalg.LinAlgError:
            continue
        c = Minv @ Aty
        res = Aw @ c - yw
        tr = np.trace(Aw @ Minv @ Aw.T)
        sols.append((float(res @ res) / max(N - tr, 1e-3) ** 2, al, c, Minv, float(res @ res)))
    # with errors: the discrepancy principle (the strongest smoothing that still fits within the errors,
    # chi2 <= 1), which does not fit the noise; without errors (or if no alpha reaches chi2 = 1): GCV
    fits = [x for x in sols if x[4] / N <= 1.0]
    best = max(fits, key=lambda x: x[1]) if (dG is not None and fits) else min(sols, key=lambda x: x[0])
    _gcv, al, c, Minv, rss = best
    cov = Minv @ AtA @ Minv
    G_fit = A @ c
    return SESANSInversion(r, c, np.sqrt(np.maximum(np.diag(cov), 0)), d, G, s, G_fit, al, rss / N)


def voxel_correlation(field, a, r_max=None):
    """Radially averaged autocorrelation C(r) = <df(0) df(r)> of a periodic n^3 field (spacing a), r <= r_max
    (default L/2). Returns (r, C) with r the bin centres (bins of width a, the first at r = 0)."""
    f = np.asarray(field, float)
    n = f.shape[0]
    df = f - f.mean()
    F = np.fft.rfftn(df)
    corr = np.fft.irfftn(np.abs(F) ** 2, s=f.shape, axes=(0, 1, 2)) / f.size
    i = np.arange(n)
    x = a * np.where(i <= n // 2, i, i - n)
    d = np.sqrt(x[:, None, None] ** 2 + x[None, :, None] ** 2 + x[None, None, :] ** 2)
    r_max = 0.5 * n * a if r_max is None else r_max
    idx = np.rint(d / a).astype(int).ravel()
    nb = int(r_max / a) + 1
    keep = idx < nb
    cnt = np.bincount(idx[keep], minlength=nb)
    sm = np.bincount(idx[keep], weights=corr.ravel()[keep], minlength=nb)
    good = cnt > 0
    return (np.arange(nb) * a)[good], sm[good] / cnt[good]
