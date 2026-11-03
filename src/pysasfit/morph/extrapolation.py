"""Extrapolation of I(q) to q = 0 and q = infinity, and the scattering invariant.

The correlation function needs the intensity over the whole q axis. As in
SAXSMorph (Ingham et al. 2011) the measured range is extended

* to q -> 0 by a Guinier, linear or constant extrapolation fitted to the first
  points (``low_q``), and
* to q -> infinity by Porod's law I = K q^-4 fitted to the last points, optionally
  together with a flat background B (I = K q^-4 + B; B is then subtracted from all
  data).

Unlike SAXSMorph the Porod tail is not cut at a finite q (q = 100 there) but
integrated analytically to infinity (:func:`pysasfit.morph.transforms.porod_tail`).

Between data points I(q) is interpolated log-log (``refine`` extra points per
interval); the transforms then treat u(q) = q I(q) as piecewise linear on these
nodes.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .transforms import moments_pl, porod_tail, sine_transform_pl

__all__ = ["ExtrapolatedCurve", "extrapolate", "fit_porod", "fit_low_q", "auto_porod_range"]

LOW_Q_METHODS = ("guinier", "linear", "constant", "none")


def _wls(A, y, w):
    """Weighted linear least squares: coefficients and their covariance."""
    Aw = A * w[:, None]
    yw = y * w
    coef, *_ = np.linalg.lstsq(Aw, yw, rcond=None)
    try:
        cov = np.linalg.inv(Aw.T @ Aw)
        res = yw - Aw @ coef
        dof = max(1, len(y) - A.shape[1])
        cov = cov * max(1.0, float(res @ res) / dof)
    except np.linalg.LinAlgError:
        cov = np.full((A.shape[1], A.shape[1]), np.nan)
    return coef, cov


def fit_porod(q, I, sigma=None, fit_background=False):
    """Fit I = K q^-4 (+ B) to the given points. Returns (K, dK, B, dB).

    Linearised as I q^4 = K + B q^4 (weights q^4 / sigma).
    """
    q = np.asarray(q, float)
    I = np.asarray(I, float)
    s = np.ones_like(I) if sigma is None else np.where(np.asarray(sigma, float) > 0, sigma, np.nan)
    if np.any(~np.isfinite(s)):
        s = np.where(np.isfinite(s), s, np.nanmedian(s) if np.any(np.isfinite(s)) else 1.0)
    y = I * q ** 4
    w = 1.0 / (s * q ** 4)
    if fit_background:
        A = np.column_stack([np.ones_like(q), q ** 4])
        (K, B), cov = _wls(A, y, w)
        return float(K), float(np.sqrt(abs(cov[0, 0]))), float(B), float(np.sqrt(abs(cov[1, 1])))
    A = np.ones((q.size, 1))
    (K,), cov = _wls(A, y, w)
    return float(K), float(np.sqrt(abs(cov[0, 0]))), 0.0, 0.0


def auto_porod_range(q, I, sigma=None, fit_background=False, min_points=6, start_fraction=0.4, tolerance=1.5,
                     chi2_ok=2.0, max_candidates=60):
    """Start of the Porod range chosen from the data.

    For every candidate start q_i (from ``start_fraction`` of the points on, at least ``min_points`` points
    left) the law I = K q^-4 (+ B) is fitted to q >= q_i, with its reduced chi^2 (relative scatter when no
    sigma is given) and the lag-1 autocorrelation rho of its residuals. A candidate is acceptable when its
    chi^2 is at most ``tolerance`` times the smallest chi^2 of all candidates, or at most ``chi2_ok`` (with
    sigma), and its residuals show no trend (rho below 3/sqrt(N), at least 0.15: a range reaching into the
    non-Porod region leaves
    residuals of one sign, even when generous error bars keep chi^2 small). The chosen start is the
    **lowest** acceptable q_i: the longest range on which the law still describes the data, which gives the
    most precise K and B. The data show no clear Porod regime (``info['clear'] = False``, with a message) if
    even the best candidate has chi^2 > 3 chi2_ok (with sigma), if the chosen range spans less than a factor
    1.5 in q (K and B cannot be separated), or if at the start of the chosen range the
    Porod term K q^-4 is smaller than the background (then the "fit" is a constant, not Porod's law).

    Returns (index of the first point, info) with the candidates (q_start, K, B, chi2) in ``info``."""
    q = np.asarray(q, float)
    I = np.asarray(I, float)
    n = q.size
    s = np.abs(I) if sigma is None else np.where(np.asarray(sigma, float) > 0, sigma, np.abs(I))
    s = np.where(s > 0, s, np.nanmedian(s[s > 0]) if np.any(s > 0) else 1.0)
    npar = 2 if fit_background else 1
    lo, hi = int(n * start_fraction), n - max(min_points, npar + 2)
    if hi <= lo:
        lo = max(0, hi)
    starts = np.unique(np.linspace(lo, max(lo, hi), min(max_candidates, max(1, hi - lo + 1))).astype(int))
    cand = []
    for i in starts:
        K, dK, B, dB = fit_porod(q[i:], I[i:], s[i:], fit_background)
        res = (I[i:] - K * q[i:] ** -4 - B) / s[i:]
        chi2 = float(res @ res) / max(1, n - i - npar)
        d = res - res.mean()
        rho = float(np.sum(d[1:] * d[:-1]) / max(np.sum(d * d), 1e-300)) if d.size > 3 else 0.0
        cand.append((float(q[i]), K, B, chi2, int(i), rho))
    ok = [c for c in cand if c[1] > 0]
    if not ok:
        return int(starts[-1]), {"mode": "auto", "clear": False, "candidates": cand, "chi2": np.nan,
                                 "message": "no candidate gives K > 0"}
    cmin = min(c[3] for c in ok)

    def no_trend(c):            # lag-1 autocorrelation within 3 sigma of white noise (sigma ~ 1/sqrt(N))
        return c[5] < max(0.15, 3.0 / np.sqrt(max(1, n - c[4])))
    good = [c for c in ok if (c[3] <= tolerance * cmin or (sigma is not None and c[3] <= chi2_ok)) and no_trend(c)]
    if not good:                # no range without a trend: the best fit
        good = [min(ok, key=lambda c: c[3])]
    chosen = min(good, key=lambda c: c[4])
    msg = ""
    if sigma is not None and cmin > 3 * chi2_ok:
        msg = f"no clear Porod regime: best chi2_red = {cmin:.3g}"
    elif q[-1] / chosen[0] < 1.5:
        msg = (f"no clear Porod regime: the law holds only on q = {chosen[0]:.4g} ... {q[-1]:.4g} (less than a "
               "factor 1.5): K and B are poorly separated")
    elif chosen[1] * chosen[0] ** -4 < abs(chosen[2]):
        msg = (f"no clear Porod regime: at the start of the range (q = {chosen[0]:.4g}) the Porod term K q^-4 is "
               f"smaller than the background B = {chosen[2]:.4g}")
    return chosen[4], {"mode": "auto", "clear": not msg, "candidates": cand, "chi2": chosen[3],
                       "chi2_min": cmin, "q_start": chosen[0], "message": msg}


def fit_low_q(q, I, sigma=None, method="guinier"):
    """Low-q extrapolation fitted to the given (first) points.

    Returns (method actually used, callable I_ext(q), info dict). Guinier falls back to
    constant if the fit gives Rg^2 <= 0 or non-positive intensities; linear falls back
    to constant if the extrapolated I(0) is negative (as in SAXSMorph).
    """
    q = np.asarray(q, float)
    I = np.asarray(I, float)
    s = np.ones_like(I) if sigma is None else np.asarray(sigma, float)
    s = np.where(s > 0, s, np.median(s[s > 0]) if np.any(s > 0) else 1.0)
    const = float(np.average(I, weights=1 / s ** 2))
    if method == "guinier" and np.all(I > 0) and q.size >= 2:
        (a, b), _ = _wls(np.column_stack([np.ones_like(q), q ** 2]), np.log(I), I / s)
        if b < 0:
            I0, Rg = float(np.exp(a)), float(np.sqrt(-3 * b))
            return "guinier", (lambda x: I0 * np.exp(-(x * Rg) ** 2 / 3)), {"I0": I0, "Rg": Rg}
        method = "constant"
    if method == "linear" and q.size >= 2:
        (a, b), _ = _wls(np.column_stack([np.ones_like(q), q]), I, 1 / s)
        if a > 0:
            return "linear", (lambda x: a + b * x), {"I0": float(a), "slope": float(b)}
        method = "constant"
    if method in ("guinier", "linear", "constant"):
        return "constant", (lambda x: np.full_like(np.asarray(x, float), const)), {"I0": const}
    return "none", None, {}


@dataclass
class ExtrapolatedCurve:
    """I(q) over the whole q axis: nodes with u(q) = q I(q) (piecewise linear) and a Porod tail."""

    q: np.ndarray                  # nodes, from 0 (or the first data point) to q_N
    I: np.ndarray                  # intensity at the nodes (background B subtracted)
    K: float                       # Porod constant of the tail I = K q^-4 for q > q_N
    dK: float = 0.0
    B: float = 0.0                 # flat background subtracted from the data
    dB: float = 0.0
    low_q: str = "none"
    low_q_info: dict = field(default_factory=dict)
    q_data: tuple = (np.nan, np.nan)   # data range actually used
    porod_range: tuple = (np.nan, np.nan)
    B_mode: str = "none"           # 'fitted', 'fixed' or 'none'
    porod_mode: str = "last 15 %"  # 'auto', 'points', 'qmin' or 'last 15 %'
    porod_info: dict = field(default_factory=dict)

    @property
    def u(self):
        return self.q * self.I

    @property
    def qN(self):
        return float(self.q[-1])

    @property
    def invariant(self):
        """Q = integral_0^inf q^2 I(q) dq (nodes + analytic Porod tail K / q_N)."""
        return moments_pl(self.q, self.u, 1) + (self.K / self.qN if self.K else 0.0)

    @property
    def porod_invariant_fraction(self):
        """Share of the invariant that comes from the Porod tail (large = data range too short)."""
        return (self.K / self.qN) / self.invariant if self.K else 0.0

    def debye_transform(self, r):
        """integral_0^inf q^2 I(q) sin(qr)/(qr) dq  (= Q gamma(r))."""
        r = np.atleast_1d(np.asarray(r, float))
        out = np.empty_like(r)
        nz = r > 0
        out[~nz] = moments_pl(self.q, self.u, 1)
        out[nz] = sine_transform_pl(self.q, self.u, r[nz]) / r[nz]
        if self.K:
            out += porod_tail(self.K, self.qN, r)
        return out

    def gamma(self, r):
        """Normalised correlation function gamma(r) = (1/Q) integral q^2 I sin(qr)/(qr) dq, gamma(0) = 1."""
        return self.debye_transform(r) / self.invariant

    def evaluate(self, q):
        """Extrapolated intensity at arbitrary q (log-log between nodes, Porod beyond q_N)."""
        q = np.asarray(q, float)
        out = np.interp(q, self.q, self.I)
        pos = (self.I > 0)
        if np.all(pos[1:]):
            m = (q > self.q[1]) & (q <= self.qN)
            out[m] = np.exp(np.interp(np.log(q[m]), np.log(self.q[1:]), np.log(self.I[1:])))
        hi = q > self.qN
        out[hi] = self.K * q[hi] ** -4
        return out


def _refine_loglog(q, I, refine):
    """Insert ``refine`` points per interval, I interpolated log-log (linear where I <= 0)."""
    if refine <= 0 or q.size < 2:
        return q, I
    t = np.arange(1, refine + 1) / (refine + 1)
    lq = np.log(q)
    qi = np.exp(lq[:-1, None] + t[None, :] * np.diff(lq)[:, None])
    pos = (I[:-1] > 0) & (I[1:] > 0)
    li = np.where(pos[:, None],
                  np.exp(np.log(np.where(I[:-1] > 0, I[:-1], 1))[:, None]
                         + t[None, :] * np.diff(np.log(np.where(I > 0, I, 1)))[:, None]),
                  I[:-1, None] + (qi - q[:-1, None]) / np.diff(q)[:, None] * np.diff(I)[:, None])
    qq = np.concatenate([np.column_stack([q[:-1], qi]).ravel(), q[-1:]])
    II = np.concatenate([np.column_stack([I[:-1], li]).ravel(), I[-1:]])
    return qq, II


def extrapolate(q, I, sigma=None, qmin=None, qmax=None, low_q="guinier", low_q_points=10,
                low_q_nodes=40, porod_points=None, porod_qmin=None, fit_background=False,
                refine=4, K=None, background=None, porod_auto=False):
    """Extend background-subtracted data to the whole q axis.

    Parameters
    ----------
    q, I, sigma : data (sigma optional, used as fit weights)
    qmin, qmax : data range used (default: all)
    low_q : 'guinier' (default), 'linear', 'constant' or 'none' (no data below q_min)
    low_q_points : number of first points the low-q extrapolation is fitted to
    low_q_nodes : nodes of the extrapolation between q = 0 and q_min
    porod_points / porod_qmin : the Porod law is fitted to the last ``porod_points``
        points or to all points with q >= ``porod_qmin``; without either: the range chosen by
        :func:`auto_porod_range` (``porod_auto``) or the last 15 %, at least 5
    fit_background : fit I = K q^-4 + B and subtract B
    background : fixed flat background B subtracted before everything (used when ``fit_background``
        is False; manual correction)
    refine : extra log-log interpolated nodes per data interval
    K : fixed Porod constant (skips the fit); 0 switches the tail off
    """
    q = np.asarray(q, float)
    I = np.asarray(I, float)
    s = None if sigma is None else np.asarray(sigma, float)
    m = np.isfinite(q) & np.isfinite(I) & (q > 0)
    if qmin is not None:
        m &= q >= qmin
    if qmax is not None:
        m &= q <= qmax
    o = np.argsort(q[m])
    q, I = q[m][o], I[m][o]
    s = None if s is None else s[m][o]
    if q.size < 8:
        raise ValueError("at least 8 data points are needed in the selected q range")
    # ---- Porod
    B_fixed = (not fit_background) and background is not None and background != 0
    if B_fixed:
        I = I - float(background)
    pinfo = {}
    if porod_qmin is not None:
        pm = q >= porod_qmin
        pmode = "qmin"
    elif porod_points:
        pm = np.zeros(q.size, bool)
        pm[-min(int(porod_points), q.size):] = True
        pmode = "points"
    elif porod_auto and K is None:
        i0, pinfo = auto_porod_range(q, I, s, fit_background and not B_fixed)
        pm = np.zeros(q.size, bool)
        pm[i0:] = True
        pmode = "auto"
    else:
        n = max(5, int(round(0.15 * q.size)))
        pm = np.zeros(q.size, bool)
        pm[-min(n, q.size):] = True
        pmode = "last 15 %"
    if pm.sum() < 3:
        raise ValueError("fewer than 3 points in the Porod range")
    if K is None:
        K, dK, B, dB = fit_porod(q[pm], I[pm], None if s is None else s[pm], fit_background and not B_fixed)
    else:
        dK, B, dB = 0.0, 0.0, 0.0
    I = I - B
    if B_fixed:
        B, dB = float(background), 0.0
    # ---- low q
    n0 = min(max(3, low_q_points), q.size)
    used, fun, info = fit_low_q(q[:n0], I[:n0], None if s is None else s[:n0], low_q)
    qq, II = _refine_loglog(q, I, refine)
    if fun is not None:
        q0 = np.linspace(0.0, q[0], low_q_nodes + 1)[:-1]
        qq = np.concatenate([q0, qq])
        II = np.concatenate([fun(q0), II])
    return ExtrapolatedCurve(q=qq, I=II, K=float(K), dK=float(dK), B=float(B), dB=float(dB), low_q=used,
                             low_q_info=info, q_data=(float(q[0]), float(q[-1])),
                             porod_range=(float(q[pm][0]), float(q[pm][-1])),
                             B_mode="fixed" if B_fixed else ("fitted" if fit_background else "none"),
                             porod_mode=pmode, porod_info=pinfo)
