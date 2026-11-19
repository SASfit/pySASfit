"""Numba version of pysasfit.scatter.schulz_kernel._piece (same panels, nodes and weights).

Kernel kinds: 0 polynomial gd(t) = sum coef[j] t^j, 1 disc amplitude -(2/pi) t^2/(1+sqrt(1-t^2)),
2 disc G_11 - G_11(0) (the Chebyshev fits of schulz_kernel._P2Fast), 3 line integral (1/2)[ln(2/t) - (2-t)/2].
"""
from __future__ import annotations

import math

import numba as nb
import numpy as np


def make(fastmath=False):
    """Compile the kernels (fastmath=True lets LLVM vectorise the transcendental functions with SVML if present)."""
    njit = nb.njit(cache=False, fastmath=fastmath, error_model="numpy")
    pjit = nb.njit(cache=False, fastmath=fastmath, error_model="numpy", parallel=True)

    @njit
    def cheb(c, x):
        b1 = 0.0
        b2 = 0.0
        for j in range(c.shape[0] - 1, 0, -1):
            tmp = 2.0 * x * b1 - b2 + c[j]
            b2 = b1
            b1 = tmp
        return x * b1 - b2 + c[0]

    @njit
    def gd(t, kind, coef, cu, cw, edges, ch, t1):
        if kind == 0:
            r = 0.0
            for j in range(coef.shape[0] - 1, -1, -1):
                r = r * t + coef[j]
            return r
        if kind == 1:
            u = 1.0 - t * t
            if u < 0.0:
                u = 0.0
            return -2.0 / math.pi * t * t / (1.0 + math.sqrt(u))
        if kind == 3:
            if t <= 0.0:
                return 0.0
            return 0.5 * (math.log(2.0 / t) - 0.5 * (2.0 - t))
        # kind 2: disc
        if t < t1:
            if t <= 0.0:
                return 0.0
            x = 2.0 * t / t1 - 1.0
            return t * t * (cheb(cu, x) + cheb(cw, x) * math.log(t))
        k = 0
        if t >= edges[1]:
            k = 1
        if t >= edges[2]:
            k = 2
        a = edges[k]
        b = edges[k + 1]
        x = (2.0 * t - a - b) / (b - a)
        if x > 1.0:
            x = 1.0
        return cheb(ch[k], x)

    @pjit
    def piece_rows(a, alpha, b_lo, b_hi, kind, coef, cu, cw, edges, ch, t1, nlo, nhi, nu, lev, vcut, gx, gw):
        n = a.shape[0]
        out = np.zeros(n)
        ne = (nu + 1) + lev + nhi + nlo
        ng = gx.shape[0]
        for i in nb.prange(n):
            ai = a[i]
            if ai <= 0.0:
                continue
            v_lo = math.atan(ai * b_lo)
            v_hi = min(math.atan(ai * b_hi), vcut)
            if not v_hi > v_lo:
                continue
            span = v_hi - v_lo
            ev = np.empty(ne)
            m = 0
            for j in range(nu + 1):
                ev[m] = v_lo + span * j / nu
                m += 1
            for j in range(1, lev + 1):
                tt = b_hi * 0.5 ** j
                vv = math.atan(ai * tt) if tt > b_lo else v_lo
                ev[m] = min(max(vv, v_lo), v_hi)
                m += 1
            for j in range(1, nhi + 1):
                tt = b_hi - (b_hi - b_lo) * 0.5 ** j
                ev[m] = min(max(math.atan(ai * tt), v_lo), v_hi)
                m += 1
            if nlo > 0:
                top = min(b_hi, math.tan(v_hi) / ai) - b_lo
                if top < 0.0:
                    top = 0.0
                for j in range(1, nlo + 1):
                    ev[m] = min(max(math.atan(ai * (b_lo + top * 0.5 ** j)), v_lo), v_hi)
                    m += 1
            ev.sort()
            acc = 0.0
            for p in range(ne - 1):
                lo = ev[p]
                hi = ev[p + 1]
                if hi <= lo:
                    continue
                c = 0.5 * (hi + lo)
                hw = 0.5 * (hi - lo)
                sp = 0.0
                for q in range(ng):
                    v = c + hw * gx[q]
                    t = math.tan(v) / ai
                    if t < b_lo:
                        t = b_lo
                    if t > b_hi:
                        t = b_hi
                    sp += gw[q] * gd(t, kind, coef, cu, cw, edges, ch, t1) * math.exp(
                        (alpha - 2.0) * math.log(math.cos(v))) * math.cos(alpha * v)
                acc += hw * sp
            out[i] = 2.0 * acc / ai
        return out

    return piece_rows


# ---------------------------------------------------------------------------------------------- kernels
def kernels(sk):
    """Numba descriptions of the kernels of schulz_kernel: name -> (g0, [(b_lo, b_hi, kind, coef, nlo, nhi)])."""
    sk._gd_p2(np.zeros(1))  # builds the disc fits
    out = {
        ("P", 1): (0.5, [(0.0, 2.0, 0, [0.0, -0.25], 0, 0)]),
        ("P", 2): (16.0 / (3.0 * np.pi ** 2), [(0.0, 2.0, 2, [0.0], 14, 0)]),
        ("P", 3): (0.6, [(0.0, 2.0, 0, [0.0, 0.0, -0.75, 0.375, 0.0, -3.0 / 160.0], 0, 0)]),
        ("F", 1): (0.5, []),
        ("F", 2): (2.0 / np.pi, [(0.0, 1.0, 1, [0.0], 0, sk._NHI)]),
        ("F", 3): (0.75, [(0.0, 1.0, 0, [0.0, 0.0, -0.75], 0, 0)]),
        ("L",): (0.0, [(0.0, 2.0, 3, [0.0], 48, 0)]),
    }
    return out


def cross3_kernel(a, b):
    a, b = (a, b) if a >= b else (b, a)
    den = a ** 3 * b ** 3
    g0 = 3.0 * (5 * a * a - b * b) / (20.0 * a ** 3)
    c2 = [-3.0 * (a - b) ** 3 * (a * a + 3 * a * b + b * b) / (40.0 * den), 9.0 * (a - b) ** 2 * (a + b) ** 2 / (32.0 * den),
          -3.0 * (a ** 3 + b ** 3) / (8.0 * den), 3.0 * (a * a + b * b) / (16.0 * den), 0.0, -3.0 / (160.0 * den)]
    pieces = []
    if a > b:
        pieces.append((0.0, a - b, 0, [0.0, 0.0, -0.75 / a ** 3], 0, 0))
    pieces.append((a - b, a + b, 0, c2, 0, 0))
    return g0, pieces


def value(piece_rows, sk, kernel, y, z, s):
    """Same as schulz_kernel.schulz_cos (y > 0), pieces by the numba kernel."""
    y = np.abs(np.asarray(y, float)).ravel()
    alpha = z + s + 1.0
    th = 1.0 / (z + 1.0)
    a = y * th
    g0, pieces = kernel
    fast = sk._P2_FAST
    T = max([p[1] for p in pieces], default=1.0)
    vcut = math.acos(math.exp(-40.0 / (alpha - 2.0))) if alpha > 2.0 else 0.5 * math.pi
    out = np.zeros(y.shape)
    for b_lo, b_hi, kind, coef, nlo, nhi in pieces:
        v_lo = np.arctan(a * b_lo)
        v_hi = np.minimum(np.arctan(a * b_hi), vcut)
        live = v_hi > v_lo
        if not live.any():
            continue
        span = np.where(live, v_hi - v_lo, 0.0)
        nu = int(np.ceil(alpha * span.max() / sk._PHASE)) + 1
        teff = np.minimum(a * b_hi, np.tan(v_hi))
        lev = int(np.clip(np.ceil(np.log2(max(float(np.max(np.where(live, teff, 1.0))), 1.0))) + 2, 2, 60))
        out += piece_rows(np.where(live, a, 0.0), float(alpha), float(b_lo), float(b_hi), int(kind),
                          np.asarray(coef, float), fast.cu, fast.cw, fast.edges, np.array(fast.ch), float(sk._T1),
                          int(nlo), int(nhi), int(nu), int(lev), float(vcut), sk._GX, sk._GW)
    if g0 != 0.0:
        out += 2.0 * g0 * sk._const_part(a, alpha, T)
    return out
