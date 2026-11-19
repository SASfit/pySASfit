"""Schulz-Zimm averages of the hypergeometric building blocks as one integral (no regime switching).

Every block is a cosine transform of a kernel G on [0, T] (G even, so only t >= 0 is needed):

    F_d(x)            = 2 int_0^1 h_d(t) cos(x t) dt,   h_1 = 1/2, h_2 = (2/pi) sqrt(1-t^2), h_3 = (3/4)(1-t^2)
    F_d(a x) F_d(b x) = 2 int_0^(a+b) G_ab(t) cos(x t) dt,   G_ab = h_d,a * h_d,b  (convolution, h_a(u) = h(u/a)/a)
    P_d(x) = F_d(x)^2 = G_11 ;  line integral int_0^1 P_1(x t) dt: kernel (1/2)[ln(2/w) - (2-w)/2]

For the Schulz-Zimm distribution of x = R/X with weight x^s (alpha = z+s+1, theta = 1/(z+1)) the average of
cos(v x) is Re (1 - i v theta)^(-alpha), so with y = q X

    < f(y x) > = 2 int_0^T G(t) Re (1 - i y theta t)^(-alpha) dt.

With v = atan(a t), a = y theta, the integrand becomes (2/a) G(tan(v)/a) cos(v)^(alpha-2) cos(alpha v): the phase
is linear in v (alpha v, at most alpha pi/2 in total), the envelope cos^(alpha-2) cuts the range for large alpha.
Panels: uniform in v (<= 4 rad of phase each), geometric in t from the effective upper end down to the scale 1/a
(tan distortion near pi/2), towards t = 0 for the kernels with t^2 ln t (disc) or ln t (line integral) behaviour
and towards the square-root end of the disc amplitude; 12-point Gauss-Legendre each. All y at once (node matrix).

The constant G(0) is integrated in closed form and the rest G(t) - G(0) is written without cancellation (exact
polynomial coefficients for d = 1, 3 and the core/shell terms, U(t) + W(t) ln t from the logarithmic 2F1
connection formula for the disc). This removes the 1/y-sized terms that would otherwise cancel to the Porod tail:
<P_3> ~ y^-4 is accurate to ~1e-16 y (relative), measured 1e-10 at y = 1e4 for z+s+1 >= 10.

Exact up to rounding for every y and every alpha > 0; no switching between series, asymptotics and quadrature.
Repeated requests with the same (block, z, s) are interpolated from exact piecewise Chebyshev tables in ln y
(see "y cache" below; accuracy 1e-12 relative or the rounding level of the direct values).
"""

from __future__ import annotations

import math

import numpy as np
from scipy import special

_GX, _GW = np.polynomial.legendre.leggauss(12)
_G3X, _G3W = np.polynomial.legendre.leggauss(3)
_PHASE = 4.0  # rad of the phase alpha*v per panel (12-point Gauss-Legendre: error ~ 2^24/24! ~ 3e-17)
_CHUNK = 1000  # values of y per block of the node matrix
_NLO = 20  # default geometric levels towards a singular lower end, below t = 1/a
_NHI = 20  # geometric levels towards a singular upper end


# ------------------------------------------------------------------------------------------------- kernels
# A kernel is (g0, pieces): g(t) = g0 + gd(t) with gd(0) = 0 written without cancellation, pieces = list of
# (t_lo, t_hi, gd, grade_lo, grade_hi). The constant g0 is integrated in closed form: for large y the 1/y-sized
# terms that cancel to the Porod tail then never appear numerically.


BACKEND = "auto"  # "auto": the Numba kernel (_schulz_numba) when numba is installed, "numpy": NumPy only


class _GD:
    """gd(t) = g(t) - g0 of one piece (callable, NumPy) with kind/coef for the Numba kernel."""

    def __init__(self, kind, func, coef=(0.0,)):
        self.kind, self.func, self.coef = kind, func, np.asarray(coef, float)

    def __call__(self, t):
        return self.func(t)


def _poly(c):
    c = np.asarray(c, float)
    return _GD(0, lambda t: np.polynomial.polynomial.polyval(np.asarray(t, float), c), c)


def kernel_amp(d):
    """F_d(x) = 2 int_0^1 h_d(t) cos(xt) dt: h_1 = 1/2, h_2 = (2/pi) sqrt(1-t^2), h_3 = (3/4)(1-t^2)."""
    if d == 1:
        return 0.5, []
    if d == 2:
        def gd(t):
            return -2.0 / np.pi * t * t / (1.0 + np.sqrt(np.clip(1.0 - t * t, 0.0, None)))

        return 2.0 / np.pi, [(0.0, 1.0, _GD(1, gd), False, True)]
    return 0.75, [(0.0, 1.0, _poly([0.0, 0.0, -0.75]), False, False)]


def kernel_p(d):
    """P_d(x) = F_d(x)^2 = 2 int_0^2 G(t) cos(xt) dt, G = h_d * h_d."""
    if d == 1:
        return 0.5, [(0.0, 2.0, _poly([0.0, -0.25]), False, False)]
    if d == 2:
        g0 = 16.0 / (3.0 * np.pi**2)
        return g0, [(0.0, 2.0, _GD(2, _gd_p2), 14, False)]
    return 0.6, [(0.0, 2.0, _poly([0.0, 0.0, -0.75, 0.375, 0.0, -3.0 / 160.0]), False, False)]


def kernel_cross3(a, b):
    """F_3(ax) F_3(bx) = 2 int_0^(a+b) G_ab(t) cos(xt) dt (piecewise polynomial, exact coefficients)."""
    a, b = (a, b) if a >= b else (b, a)
    den = a**3 * b**3
    g0 = 3.0 * (5 * a * a - b * b) / (20.0 * a**3)
    c2 = [
        -3.0 * (a - b) ** 3 * (a * a + 3 * a * b + b * b) / (40.0 * den),
        9.0 * (a - b) ** 2 * (a + b) ** 2 / (32.0 * den),
        -3.0 * (a**3 + b**3) / (8.0 * den),
        3.0 * (a * a + b * b) / (16.0 * den),
        0.0,
        -3.0 / (160.0 * den),
    ]
    pieces = []
    if a > b:
        pieces.append((0.0, a - b, _poly([0.0, 0.0, -0.75 / a**3]), False, False))
    pieces.append((a - b, a + b, _poly(c2), False, False))
    return g0, pieces


def g_p2(t):
    """G_11 of the disc: (4/pi^2) r^2 A (pi/2) 2F1(-1/2, 1/2; 2; k^2), r = 1-t/2, A = 1+t/2, k = r/A."""
    t = np.asarray(t, float)
    r, A = 1.0 - 0.5 * t, 1.0 + 0.5 * t
    m = np.clip((r / A) ** 2, 0.0, 1.0)
    p = np.clip(8.0 * t / (2.0 + t) ** 2, 0.0, 1.0)  # 1 - m without cancellation
    with np.errstate(all="ignore"):
        pk = np.where(p > 0, p * special.ellipkm1(p), 0.0)  # (1-m) K(m)
        ell = 2.0 * ((1 + m) * special.ellipe(m) - pk) / (3 * m)
        hyp = 0.5 * np.pi * special.hyp2f1(-0.5, 0.5, 2.0, m)
    i = np.where(m < 0.5, hyp, ell)
    return np.where(t >= 2.0, 0.0, 4.0 / np.pi**2 * r * r * A * i)


_T1 = 0.4503  # p(t) = 8t/(2+t)^2 = 0.6


def _p2_parts(t):
    """G_11 - G_11(0) of the disc as U(t) + W(t) ln t for small t (2F1 near m = 1, logarithmic case
    c - a - b = 2): 2F1(-1/2, 1/2; 2; 1-p) = 8(1+p/4)/(3 pi) + p^2/(2 pi) sum_n c_n p^n [ln p + delta_n],
    c_n = (3/2)_n (5/2)_n/(n! (n+2)!), delta_n = psi(n+3/2) + psi(n+5/2) - psi(n+1) - psi(n+3)."""
    t = np.asarray(t, float)
    p = 8.0 * t / (2.0 + t) ** 2
    s1 = np.zeros(t.shape)
    s2 = np.zeros(t.shape)
    c, pk = 0.5, np.ones(t.shape)
    for n in range(160):
        d = special.digamma(n + 1.5) + special.digamma(n + 2.5) - special.digamma(n + 1) - special.digamma(n + 3)
        s1 += c * pk
        s2 += c * d * pk
        c *= (1.5 + n) * (2.5 + n) / ((n + 1) * (n + 3))
        pk = pk * p
    pre = 2.0 / np.pi * (1.0 - 0.5 * t) ** 2 * (1.0 + 0.5 * t)
    w = pre * p * p / (2.0 * np.pi) * s1
    # pre 8(1+p/4)/(3 pi) - G0 = G0 t^2 (t^2 + 2t - 16)/(16 + 8t)  (no cancellation)
    g0 = 16.0 / (3.0 * np.pi**2)
    u = g0 * t * t * (t * t + 2 * t - 16.0) / (16.0 + 8.0 * t) + pre * p * p / (2 * np.pi) * (
        s1 * np.log(8.0 / (2.0 + t) ** 2) + s2
    )
    return u, w


class _P2Fast:
    """Chebyshev fits: U/t^2, W/t^2 on [0, T1] (degree 18), G_11 - G_11(0) on 3 panels of [T1, 2] (degree 18)."""

    def __init__(self):
        x = np.cos(np.pi * (np.arange(19) + 0.5) / 19)
        tl = 0.5 * _T1 * (x + 1.0)
        u, w = _p2_parts(tl)  # both O(t^2): fit U/t^2, W/t^2 (relative accuracy for small t)
        self.cu = np.polynomial.chebyshev.chebfit(x, u / tl**2, 18)
        self.cw = np.polynomial.chebyshev.chebfit(x, w / tl**2, 18)
        self.edges = np.array([_T1, 0.8, 1.3, 2.0])
        self.ch = []
        for a, b in zip(self.edges[:-1], self.edges[1:]):
            th = 0.5 * (a + b) + 0.5 * (b - a) * x
            self.ch.append(np.polynomial.chebyshev.chebfit(x, g_p2(th) - 16.0 / (3.0 * np.pi**2), 18))
        self.ch_arr = np.array(self.ch)

    def __call__(self, t):
        t = np.asarray(t, float)
        out = np.empty(t.shape)
        lo = t < _T1
        tl = t[lo]
        xl = 2.0 * tl / _T1 - 1.0
        with np.errstate(divide="ignore", invalid="ignore"):
            lt = np.where(tl > 0, np.log(tl), 0.0)
        out[lo] = tl * tl * (_clenshaw(self.cu, xl) + _clenshaw(self.cw, xl) * lt)
        for k, (a, b) in enumerate(zip(self.edges[:-1], self.edges[1:])):
            m = (t >= a) & (t < b) if k < 2 else (t >= a)
            out[m] = _clenshaw(self.ch[k], (2.0 * t[m] - a - b) / (b - a))
        return out


def _clenshaw(c, x):
    """sum_j c_j T_j(x) (in place, few temporaries)."""
    x2 = 2.0 * x
    b1 = np.zeros(x.shape)
    b2 = np.zeros(x.shape)
    tmp = np.empty(x.shape)
    for cj in c[:0:-1]:
        np.multiply(x2, b1, out=tmp)
        tmp -= b2
        tmp += cj
        b1, b2, tmp = tmp, b1, b2
    np.multiply(x, b1, out=tmp)
    tmp -= b2
    tmp += c[0]
    return tmp


_P2_FAST = None


def _gd_p2(t):
    global _P2_FAST
    if _P2_FAST is None:
        _P2_FAST = _P2Fast()
    return _P2_FAST(t)


def kernel_line():
    """int_0^1 P_1(x t) dt = 2 int_0^2 K(w) cos(xw) dw, K = (1/2)[ln(2/w) - (2-w)/2]."""
    def gd(t):
        t = np.asarray(t, float)
        with np.errstate(all="ignore"):
            return np.where(t > 0, 0.5 * (np.log(2.0 / t) - 0.5 * (2.0 - t)), 0.0)
    return 0.0, [(0.0, 2.0, _GD(3, gd), 48, False)]


# ------------------------------------------------------------------------------------------------- engine
def _const_part(a, alpha, T):
    """int_0^T Re(1 - i a t)^(-alpha) dt = (1+a^2T^2)^((1-alpha)/2) sin((alpha-1) atan(aT)) / (a (alpha-1))."""
    a = np.asarray(a, float)
    at = np.arctan(a * T)
    with np.errstate(all="ignore"):
        amp = np.exp(0.5 * (1.0 - alpha) * np.log1p((a * T) ** 2))
        x = (alpha - 1.0) * at
        if alpha != 1.0:
            sx = np.where(np.abs(x) < 1e-8, at * (1.0 - x * x / 6.0), np.sin(x) / (alpha - 1.0))
        else:
            sx = at
        out = amp * sx / a
    return np.where(a > 0, out, T)


def _numba_kernel():
    if BACKEND != "auto":
        return None
    from . import _schulz_numba

    return _schulz_numba.piece_rows


def schulz_cos(y, z, s, kernel):
    """2 int_0^T g(t) Re(1 - i y theta t)^(-alpha) dt, Schulz-Zimm weight x^s (alpha = z+s+1, theta = 1/(z+1)),
    kernel = (g0, pieces) (see above). Any array shape of y."""
    y = np.abs(np.asarray(y, float))
    shape = y.shape
    y = y.ravel()
    alpha = z + s + 1.0
    if not alpha > 0:
        return np.full(shape, np.nan)
    th = 1.0 / (z + 1.0)
    g0, pieces = kernel
    T = max([p[1] for p in pieces], default=1.0)
    out = np.empty(y.shape)
    for i0 in range(0, y.size, _CHUNK):
        yy = y[i0 : i0 + _CHUNK]
        a = yy * th
        v = np.zeros(yy.shape)
        for piece in pieces:
            v += _piece(a, alpha, piece)
        if g0 != 0.0:
            v += 2.0 * g0 * _const_part(a, alpha, T)
        out[i0 : i0 + _CHUNK] = v
    return out.reshape(shape)


def _piece(a, alpha, piece):
    b_lo, b_hi, gd, grade_lo, grade_hi = piece
    if b_hi <= b_lo:
        return np.zeros(a.shape)
    zero = a <= 0.0
    a_s = np.where(zero, 1.0, a)
    vcut = math.acos(math.exp(-40.0 / (alpha - 2.0))) if alpha > 2.0 else 0.5 * math.pi
    v_lo = np.arctan(a_s * b_lo)
    v_hi = np.minimum(np.arctan(a_s * b_hi), vcut)
    live = (v_hi > v_lo) & ~zero
    out = np.zeros(a.shape)
    if live.any():
        span = np.where(live, v_hi - v_lo, 0.0)
        nu = int(np.ceil(alpha * span.max() / _PHASE)) + 1
        # geometric levels in t from the effective upper end down to the scale 1/a (tan distortion near pi/2)
        teff = np.minimum(a_s * b_hi, np.tan(v_hi))
        lev = int(np.clip(np.ceil(np.log2(max(float(np.max(np.where(live, teff, 1.0))), 1.0))) + 2, 2, 60))
        nb = _numba_kernel()
        if nb is not None and isinstance(gd, _GD):
            if gd.kind == 2:
                _gd_p2(np.zeros(1))
                cu, cw, edges, ch = _P2_FAST.cu, _P2_FAST.cw, _P2_FAST.edges, _P2_FAST.ch_arr
            else:
                cu = cw = np.zeros(1)
                edges = np.array([0.0, 1.0, 2.0, 3.0])
                ch = np.zeros((3, 1))
            nlo = 0 if not grade_lo else (_NLO if grade_lo is True else int(grade_lo))
            nhi = _NHI if grade_hi else 0
            out = nb(np.where(live, a_s, 0.0), float(alpha), float(b_lo), float(b_hi), int(gd.kind), gd.coef,
                     cu, cw, edges, ch, float(_T1), nlo, nhi, int(nu), int(lev), float(vcut), _GX, _GW)  # fmt: skip
            live = np.zeros(a.shape, bool)
    if live.any():
        et = [b_hi * 0.5 ** np.arange(1, lev + 1)]
        if grade_hi:
            et.append(b_hi - (b_hi - b_lo) * 0.5 ** np.arange(1, _NHI + 1))
        et = np.concatenate(et)
        et = et[(et > b_lo) & (et < b_hi)]
        ev_t = np.arctan(a_s[:, None] * et[None, :])
        if grade_lo:  # towards b_lo, from the effective end of the range on
            nlo = _NLO if grade_lo is True else int(grade_lo)
            top = np.minimum(b_hi, np.tan(v_hi) / a_s) - b_lo  # effective end of the range
            hh = np.maximum(top, 0.0)[:, None] * 0.5 ** np.arange(1, nlo + 1)[None, :]
            ev_t = np.concatenate([ev_t, np.arctan(a_s[:, None] * (b_lo + hh))], axis=1)
        ev_u = v_lo[:, None] + span[:, None] * (np.arange(nu + 1) / nu)[None, :]
        ev = np.concatenate([ev_u, np.minimum(np.maximum(ev_t, v_lo[:, None]), v_hi[:, None])], axis=1)
        ev = np.where(live[:, None], ev, 0.0)
        ev.sort(axis=1)
        lo, hi = ev[:, :-1], ev[:, 1:]
        c, hw = 0.5 * (hi + lo), 0.5 * (hi - lo)
        v = c[:, :, None] + hw[:, :, None] * _GX[None, None, :]
        w = hw[:, :, None] * _GW[None, None, :]
        t = np.clip(np.tan(v) / a_s[:, None, None], b_lo, b_hi)
        with np.errstate(all="ignore"):
            f = gd(t) * np.exp((alpha - 2.0) * np.log(np.cos(v))) * np.cos(alpha * v)
        f = np.where(w > 0, f, 0.0)
        out = np.where(live, 2.0 * np.sum(w * f, axis=(1, 2)) / a_s, 0.0)
    if zero.any():  # y = 0: 2 int gd
        e = np.unique(np.concatenate([[b_lo, b_hi], b_lo + (b_hi - b_lo) * 0.5 ** np.arange(1, 52),
                                      b_hi - (b_hi - b_lo) * 0.5 ** np.arange(1, 40)]))
        c, hw = 0.5 * (e[1:] + e[:-1]), 0.5 * (e[1:] - e[:-1])
        tt = (c[:, None] + hw[:, None] * _GX).ravel()
        ww = (hw[:, None] * _GW).ravel()
        out = np.where(zero, 2.0 * np.sum(ww * gd(tt)), out)
    return out


# ------------------------------------------------------------------------------------------------- y cache
# The averages are smooth functions of u = ln y. For repeated or large requests with the same (kernel, z, s) they
# are interpolated from piecewise Chebyshev tables (degree 16) built from the exact values: base panels of width
# 1/4 in u on a fixed grid, split until 4 extra exact values agree with the interpolant to CACHE_TOL (relative
# to the panel maximum). A base panel is built only after enough requests have fallen into it (CACHE_MIN), so
# single values are always computed directly. CACHE = False switches the tables off.
CACHE = True
CACHE_TOL = 1e-12
CACHE_MIN = 24
_DU = 0.25
_UMIN = math.log(1e-4)  # below: always direct (values ~ 1)
_CHEB_N = 17
_CHEB_X = np.cos(np.pi * (np.arange(_CHEB_N) + 0.5) / _CHEB_N)
_CHECK_X = np.array([-0.83, -0.31, 0.27, 0.79])
_TABLES = {}
_TABLES_MAX = 64


class _Table:
    def __init__(self, fn, z, s):
        self.fn = fn  # exact y -> value
        self.th, self.alpha = 1.0 / (z + 1.0), z + s + 1.0
        self.panels = {}  # base index -> (edges array, coefficient array)
        self.hits = {}

    def _floor(self, ymax):
        """Rounding level of the direct values (~1e-16 times the size of the integrand, 1/(a sqrt(alpha)))."""
        return 2e-15 * min(1.0, 1.0 / (ymax * self.th * math.sqrt(self.alpha)))

    def worth(self, k):
        """Tabulate base panel k only if its oscillations (period ~ pi/y in u, amplitude damped by the size
        distribution, (1+4a^2)^-(alpha-6)/2) need at most ~16 leaves."""
        ylo, yhi = math.exp(_UMIN + k * _DU), math.exp(_UMIN + (k + 1) * _DU)
        damp = math.exp(-0.5 * max(self.alpha - 6.0, 0.5) * math.log1p((2.0 * ylo * self.th) ** 2))
        return damp < 1e-14 or yhi < 600.0

    def _leaves(self, ulo, uhi, depth=0):
        x = _CHEB_X
        u = 0.5 * (ulo + uhi) + 0.5 * (uhi - ulo) * np.concatenate([x, _CHECK_X])
        v = self.fn(np.exp(u))
        c = np.polynomial.chebyshev.chebfit(x, v[:_CHEB_N], _CHEB_N - 1)
        err = np.max(np.abs(_clenshaw(c, _CHECK_X) - v[_CHEB_N:]))
        scale = np.max(np.abs(v))
        if err <= max(CACHE_TOL * scale, self._floor(math.exp(uhi))):
            return [(ulo, uhi, c)]
        if depth >= 5:
            raise ValueError("not tabulable")
        um = 0.5 * (ulo + uhi)
        return self._leaves(ulo, um, depth + 1) + self._leaves(um, uhi, depth + 1)

    def build(self, k):
        try:
            leaves = self._leaves(_UMIN + k * _DU, _UMIN + (k + 1) * _DU)
        except ValueError:
            self.panels[k] = None  # always direct
            return
        self.panels[k] = leaves
        self._flat = None

    def _flatten(self):
        leaves = sorted((lv for v in self.panels.values() if v is not None for lv in v), key=lambda lv: lv[0])
        self._lo = np.array([lv[0] for lv in leaves])
        self._hi = np.array([lv[1] for lv in leaves])
        self._c = np.array([lv[2] for lv in leaves]).reshape(len(leaves), _CHEB_N)
        self._kb = np.array(sorted(k for k, v in self.panels.items() if v is not None), dtype=np.int64)
        self._kd = np.array(sorted(k for k, v in self.panels.items() if v is None), dtype=np.int64)
        self._flat = True

    def __call__(self, y):
        out = np.empty(y.shape)
        u = np.log(np.maximum(y, 1e-300))
        k = np.floor((u - _UMIN) / _DU).astype(np.int64)
        low = u < _UMIN
        if getattr(self, "_flat", None) is None:
            self._flatten()
        tab = ~low & np.isin(k, self._kb)
        pending = ~low & ~tab & ~np.isin(k, self._kd)
        if pending.any():  # count requests, build panels that are asked for often enough
            ks, cnt = np.unique(k[pending], return_counts=True)
            changed = False
            for kk, c in zip(ks.tolist(), cnt.tolist()):
                self.hits[kk] = self.hits.get(kk, 0) + c
                if self.hits[kk] >= CACHE_MIN:
                    if self.worth(kk):
                        self.build(kk)
                    else:
                        self.panels[kk] = None
                    changed = True
            if changed:
                self._flatten()
                tab = ~low & np.isin(k, self._kb)
        if tab.any():
            us = u[tab]
            j = np.clip(np.searchsorted(self._lo, us, side="right") - 1, 0, len(self._lo) - 1)
            lo, hi = self._lo[j], self._hi[j]
            xx = np.clip((2.0 * us - lo - hi) / (hi - lo), -1.0, 1.0)
            cj = self._c[j]
            b1 = np.zeros(us.shape)
            b2 = np.zeros(us.shape)
            for m in range(_CHEB_N - 1, 0, -1):
                b1, b2 = 2.0 * xx * b1 - b2 + cj[:, m], b1
            out[tab] = xx * b1 - b2 + cj[:, 0]
        direct = ~tab
        if direct.any():
            out[direct] = self.fn(y[direct])
        return out


def _cached(key, kernel, y, z, s):
    y = np.abs(np.asarray(y, float))
    if not CACHE or y.size < 2:
        return schulz_cos(y, z, s, kernel)
    tk = (key, float(z), float(s))
    tab = _TABLES.get(tk)
    if tab is None:
        if len(_TABLES) >= _TABLES_MAX:
            _TABLES.pop(next(iter(_TABLES)))
        tab = _TABLES[tk] = _Table(lambda yy: schulz_cos(yy, z, s, kernel), z, s)
    return tab(y.ravel()).reshape(y.shape)


def clear_cache():
    _TABLES.clear()


# ------------------------------------------------------------------------------------------------- blocks
def block_value(d, y, z, s, kind="P"):
    """<P_d> (kind 'P') or <F_d> (kind 'F') for y = q X."""
    return _cached((kind, d), kernel_amp(d) if kind == "F" else kernel_p(d), y, z, s)


def cross3_value(a, b, y, z, s):
    """<F_3(a x) F_3(b x)>."""
    return _cached(("X", float(a), float(b)), kernel_cross3(a, b), y, z, s)


def line_integral_value(y, z, s):
    """int_0^1 <P_1(y x t)> dt."""
    return _cached(("L",), kernel_line(), y, z, s)
