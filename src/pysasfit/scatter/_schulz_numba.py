"""Optional Numba kernel for schulz_kernel._piece (same panels, nodes and weights as the NumPy code).

Used when numba is installed (pysasfit.scatter._accel.HAVE_NUMBA; PYSASFIT_NO_NUMBA=1 switches it off) and
schulz_kernel.BACKEND is "auto". Parallel over the y values in blocks of rows, one work array per block (no
allocation per row). Compiled with fastmath=True: no effect without SVML (the measured results are identical), lets
LLVM vectorise tan/cos/log/exp where SVML is available. Set PYSASFIT_NUMBA_FASTMATH=0 to compile without it.

Kernel kinds: 0 polynomial gd(t) = sum coef[j] t^j, 1 disc amplitude -(2/pi) t^2/(1+sqrt(1-t^2)),
2 disc G_11 - G_11(0) (Chebyshev fits of schulz_kernel._P2Fast), 3 line integral (1/2)[ln(2/t) - (2-t)/2].
Measured on a 16-core PC (2026-11-18): 5.6x faster than NumPy in total, 1D curves 1-6 ms instead of 3-50 ms.
"""

from __future__ import annotations

import math
import os

import numpy as np

from ._accel import HAVE_NUMBA

FASTMATH = os.environ.get("PYSASFIT_NUMBA_FASTMATH", "1") != "0"

if HAVE_NUMBA:
    import numba as _nb

    _njit = _nb.njit(cache=True, fastmath=FASTMATH, error_model="numpy")
    _pjit = _nb.njit(cache=True, fastmath=FASTMATH, error_model="numpy", parallel=True)

    @_njit
    def _cheb(c, x):
        b1 = 0.0
        b2 = 0.0
        for j in range(c.shape[0] - 1, 0, -1):
            tmp = 2.0 * x * b1 - b2 + c[j]
            b2 = b1
            b1 = tmp
        return x * b1 - b2 + c[0]

    @_njit
    def _gd(t, kind, coef, cu, cw, edges, ch, t1):
        if kind == 0:
            r = 0.0
            for j in range(coef.shape[0] - 1, -1, -1):
                r = r * t + coef[j]
            return r
        if kind == 1:
            u = 1.0 - t * t
            u = max(u, 0.0)
            return -2.0 / math.pi * t * t / (1.0 + math.sqrt(u))
        if kind == 3:
            if t <= 0.0:
                return 0.0
            return 0.5 * (math.log(2.0 / t) - 0.5 * (2.0 - t))
        if t < t1:  # kind 2: disc
            if t <= 0.0:
                return 0.0
            x = 2.0 * t / t1 - 1.0
            return t * t * (_cheb(cu, x) + _cheb(cw, x) * math.log(t))
        k = 0
        if t >= edges[1]:
            k = 1
        if t >= edges[2]:
            k = 2
        a = edges[k]
        b = edges[k + 1]
        x = (2.0 * t - a - b) / (b - a)
        x = min(x, 1.0)
        return _cheb(ch[k], x)

    @_njit
    def _row(ai, ev, alpha, b_lo, b_hi, kind, coef, cu, cw, edges, ch, t1, nlo, nhi, nu, lev, vcut, gx, gw):
        v_lo = math.atan(ai * b_lo)
        v_hi = min(math.atan(ai * b_hi), vcut)
        if not v_hi > v_lo:
            return 0.0
        span = v_hi - v_lo
        m = 0
        for j in range(nu + 1):
            ev[m] = v_lo + span * j / nu
            m += 1
        for j in range(1, lev + 1):
            tt = b_hi * 0.5**j
            vv = math.atan(ai * tt) if tt > b_lo else v_lo
            ev[m] = min(max(vv, v_lo), v_hi)
            m += 1
        for j in range(1, nhi + 1):
            tt = b_hi - (b_hi - b_lo) * 0.5**j
            ev[m] = min(max(math.atan(ai * tt), v_lo), v_hi)
            m += 1
        if nlo > 0:
            top = min(b_hi, math.tan(v_hi) / ai) - b_lo
            top = max(top, 0.0)
            for j in range(1, nlo + 1):
                ev[m] = min(max(math.atan(ai * (b_lo + top * 0.5**j)), v_lo), v_hi)
                m += 1
        ev[:m].sort()
        acc = 0.0
        for p in range(m - 1):
            lo = ev[p]
            hi = ev[p + 1]
            if hi <= lo:
                continue
            c = 0.5 * (hi + lo)
            hw = 0.5 * (hi - lo)
            sp = 0.0
            for q in range(gx.shape[0]):
                v = c + hw * gx[q]
                t = math.tan(v) / ai
                t = max(t, b_lo)
                t = min(t, b_hi)
                sp += gw[q] * _gd(t, kind, coef, cu, cw, edges, ch, t1) * math.exp(
                    (alpha - 2.0) * math.log(math.cos(v))
                ) * math.cos(alpha * v)
            acc += hw * sp
        return 2.0 * acc / ai

    @_pjit
    def _piece_rows(a, alpha, b_lo, b_hi, kind, coef, cu, cw, edges, ch, t1, nlo, nhi, nu, lev, vcut, gx, gw, nblk):
        n = a.shape[0]
        out = np.zeros(n)
        ne = (nu + 1) + lev + nhi + nlo
        for b in _nb.prange(nblk):
            ev = np.empty(ne)  # one work array per block of rows
            for i in range(b, n, nblk):
                if a[i] > 0.0:
                    out[i] = _row(a[i], ev, alpha, b_lo, b_hi, kind, coef, cu, cw, edges, ch, t1, nlo, nhi, nu,
                                  lev, vcut, gx, gw)  # fmt: skip
        return out

    def piece_rows(a, *args):
        """2 int over one piece for every a (rows with a <= 0 give 0); blocks of rows over the threads."""
        nblk = max(1, min(a.shape[0], 8 * _nb.get_num_threads()))
        return _piece_rows(a, *args, nblk)

else:  # pragma: no cover
    piece_rows = None
