"""Compiled (Numba) Biggs-Andrews-accelerated fixed-point loops for the EM-family solvers.

The EM / MaxEnt-EM iterations do little arithmetic per step (one or two matrix-vector products with a
~100 x 100 kernel) but run 10^3 ... 10^5 steps, so the Python interpreter overhead of the NumPy version
(~80 us per step) dominates.  This module runs the whole accelerated loop -- the fixed-point map, the
Biggs-Andrews extrapolation of :mod:`acceleration` and the per-iterate diagnostics -- in one compiled
function.  The arithmetic is the same as in em.py / maxent.py / acceleration.py, step by step; sums are
evaluated in a different order than BLAS, so results agree to rounding (~1e-12 relative after thousands of
steps), not bitwise.

Used automatically when Numba is installed (pysasfit.core.accel.HAVE_NUMBA) and the accelerated path is
selected; ``PYSASFIT_NO_NUMBA=1`` returns to the NumPy implementation.

Step types (``mode``):
  0  EM (Lucy-Richardson), optional single / double smoothing with the tridiagonal S^(h)  (em.solve)
  1  EM + MaxEnt, constant prior m                                                          (maxent)
  2  EM + MaxEnt, adaptive prior m = Pi x                                                   (maxent)
"""
from __future__ import annotations

import numpy as np

from ...core.accel import HAVE_NUMBA

__all__ = ["HAVE_NUMBA", "accelerated_loop"]

EPS = float(np.finfo(float).eps)

if HAVE_NUMBA:
    import numba as nb

    @nb.njit(cache=True, fastmath={"reassoc", "contract"})      # SIMD dot products (reassociated sums)
    def _matvec(A, x, out):
        m, n = A.shape
        for i in range(m):
            s = 0.0
            for j in range(n):
                s += A[i, j] * x[j]
            out[i] = s

    @nb.njit(cache=True)
    def _em(x, A, AT, b, colsum, Ax, tmp, out):
        _matvec(A, x, Ax)
        for i in range(Ax.size):
            if Ax[i] == 0.0:
                Ax[i] = 2.220446049250313e-16
            tmp[i] = b[i] / Ax[i]
        _matvec(AT, tmp, out)
        for j in range(x.size):
            out[j] = x[j] * (out[j] / colsum[j])

    @nb.njit(cache=True)
    def _smooth(h, x, out):
        n = x.size
        if n == 1:
            out[0] = x[0]
            return
        out[0] = (1.0 - h) * x[0] + h * x[1]
        for i in range(1, n - 1):
            out[i] = h * x[i - 1] + (1.0 - 2.0 * h) * x[i] + h * x[i + 1]
        out[n - 1] = h * x[n - 2] + (1.0 - h) * x[n - 1]

    @nb.njit(cache=True)
    def _step(mode, x, A, AT, b, colsum, h, dbl, prior, Pi, PiT, lam, Ax, tmp, w1, w2, pw, out):
        eps = 2.220446049250313e-16
        n = x.size
        if mode == 0:
            if h >= 0.0 and dbl:
                for j in range(n):
                    w1[j] = np.log(x[j] if x[j] > 0.0 else eps)
                _smooth(h, w1, w2)
                for j in range(n):
                    w1[j] = np.exp(w2[j])
                _em(w1, A, AT, b, colsum, Ax, tmp, w2)
            else:
                _em(x, A, AT, b, colsum, Ax, tmp, w2)
            if h >= 0.0:
                _smooth(h, w2, out)
            else:
                for j in range(n):
                    out[j] = w2[j]
            return
        # MaxEnt-EM (constant prior: mode 1, adaptive prior m = Pi x: mode 2)
        if mode == 2:
            _matvec(Pi, x, w1)                       # m
        else:
            for j in range(n):
                w1[j] = prior[j]
        for j in range(n):
            if w1[j] <= 0.0:
                w1[j] = eps
        _em(x, A, AT, b, colsum, Ax, tmp, w2)        # x_em
        sx = 0.0
        sxl = 0.0
        for j in range(n):
            xs = x[j] if x[j] > 0.0 else eps
            lr = np.log(xs / w1[j])
            sx += xs
            sxl += xs * lr
        wm = sxl / sx
        if mode == 2:
            # Pi^T (x_s / m_s); out is used as scratch before the final write
            for j in range(n):
                xs = x[j] if x[j] > 0.0 else eps
                out[j] = xs / w1[j]
            _matvec(PiT, out, pw)
        for j in range(n):
            xs = x[j] if x[j] > 0.0 else eps
            lr = np.log(xs / w1[j])
            if mode == 2:
                ds = lam * xs * (wm - lr - 1.0 + pw[j])
            else:
                ds = lam * xs * (-lr + wm)
            v = x[j] + (w2[j] - x[j]) + ds
            out[j] = eps if v < 0.0 else v

    @nb.njit(cache=True)
    def _entropy(x, m):
        eps = 2.220446049250313e-16
        s = 0.0
        for j in range(x.size):
            xs = x[j] if x[j] > 0.0 else eps
            ms = m[j] if m[j] > 0.0 else eps
            s += -xs * np.log(xs / ms) + xs - ms
        return s

    @nb.njit(cache=True)
    def _record(k, mode, xn, xprev, A, b, inv_db, prior, Pi, Ax, w1, g_h, c_h, r_h, e_h):
        s = 0.0
        for j in range(xn.size):
            d = xn[j] - xprev[j]
            s += d * d
            xprev[j] = xn[j]
        g_h[k] = np.sqrt(s)
        _matvec(A, xn, Ax)
        c = 0.0
        for i in range(b.size):
            r = (Ax[i] - b[i]) * inv_db[i]
            c += r * r
        c_h[k] = c / b.size
        r = 0.0
        for j in range(xn.size - 1):
            d = xn[j + 1] - xn[j]
            r += d * d
        r_h[k] = r
        if mode == 1:
            e_h[k] = _entropy(xn, prior)
        elif mode == 2:
            _matvec(Pi, xn, w1)
            e_h[k] = _entropy(xn, w1)

    @nb.njit(cache=True)
    def _loop(mode, A, b, inv_db, x0, h, dbl, prior, Pi, lam, max_iterations, tol, kin, floor):
        eps = 2.220446049250313e-16
        M, n = A.shape
        AT = np.ascontiguousarray(A.T)
        colsum = np.empty(n)
        for j in range(n):
            s = 0.0
            for i in range(M):
                s += A[i, j]
            colsum[j] = s if s != 0.0 else eps
        Ax = np.empty(M)
        tmp = np.empty(M)
        w1 = np.empty(n)
        w2 = np.empty(n)
        cl = np.empty(n)
        pw = np.zeros(n)
        PiT = np.ascontiguousarray(Pi.T)
        nrec = max(max_iterations, 2) + 1
        g_h = np.empty(nrec)
        c_h = np.empty(nrec)
        r_h = np.empty(nrec)
        e_h = np.empty(nrec)
        xprev = x0.copy()
        xn = x0.copy()
        xp1 = np.empty(n)
        # two warm-up steps
        for j in range(n):
            cl[j] = max(xn[j], floor)
        _step(mode, cl, A, AT, b, colsum, h, dbl, prior, Pi, PiT, lam, Ax, tmp, w1, w2, pw, xp1)
        gn1 = xp1 - xn
        xn1 = xn.copy()
        xn = xp1.copy()
        _record(0, mode, xn, xprev, A, b, inv_db, prior, Pi, Ax, w1, g_h, c_h, r_h, e_h)
        for j in range(n):
            cl[j] = max(xn[j], floor)
        _step(mode, cl, A, AT, b, colsum, h, dbl, prior, Pi, PiT, lam, Ax, tmp, w1, w2, pw, xp1)
        gn2 = gn1.copy()
        gn1 = xp1 - xn
        xn2 = xn1.copy()
        xn1 = xn.copy()
        xn = xp1.copy()
        _record(1, mode, xn, xprev, A, b, inv_db, prior, Pi, Ax, w1, g_h, c_h, r_h, e_h)
        k = 2
        it = 2
        prev = np.sqrt(np.sum(xn * xn))
        rel = np.inf
        yn = np.empty(n)
        while it < max_iterations and rel >= tol:
            it += 1
            beta = np.dot(gn1, gn2)
            gamma = np.dot(gn2, gn2)
            if gamma == 0.0:
                alpha = 1.0
            else:
                if kin > 0:
                    alpha = beta / gamma
                else:
                    q = beta / gamma
                    alpha = np.sign(q) * np.sqrt(abs(q))
                bound = (it - 1.0) / (it + 2.0)
                alpha = max(min(alpha, bound), -bound)
            order = abs(kin) if alpha != 0.0 else 0
            for j in range(n):
                if order == 0:
                    yn[j] = xn[j]
                elif order == 1:
                    yn[j] = xn[j] + alpha * (xn[j] - xn1[j])
                else:
                    yn[j] = (xn[j] + alpha * (xn[j] - xn1[j])
                             + 0.5 * alpha * alpha * (xn[j] - 2.0 * xn1[j] + xn2[j]))
                cl[j] = max(yn[j], floor)
            _step(mode, cl, A, AT, b, colsum, h, dbl, prior, Pi, PiT, lam, Ax, tmp, w1, w2, pw, xp1)
            for j in range(n):
                gn2[j] = gn1[j]
                gn1[j] = xp1[j] - yn[j]
                xn2[j] = xn1[j]
                xn1[j] = xn[j]
                xn[j] = xp1[j]
            norm = np.sqrt(np.sum(xn * xn))
            rel = abs(prev - norm) / (norm + eps)
            prev = norm
            _record(k, mode, xn, xprev, A, b, inv_db, prior, Pi, Ax, w1, g_h, c_h, r_h, e_h)
            k += 1
        for j in range(n):
            xn[j] = max(xn[j], floor)
        return xn, it, it < max_iterations, g_h[:k], c_h[:k], r_h[:k], e_h[:k]


def accelerated_loop(mode, A, b, db, x0, max_iterations, tol, kin_set_maa, clip_floor,
                     smoothing_h=None, double_smoothing=False, prior=None, Pi=None, lam=0.0):
    """Run the compiled accelerated loop. Returns (x, n_iterations, converged, g_norm, chi2_r, roughness,
    entropy) with the histories as Python lists (entropy empty for mode 0)."""
    A = np.ascontiguousarray(A, float)
    n = A.shape[1]
    x, it, conv, g, c, r, e = _loop(
        int(mode), A, np.ascontiguousarray(b, float), 1.0 / np.ascontiguousarray(db, float),
        np.ascontiguousarray(x0, float), -1.0 if smoothing_h is None else float(smoothing_h),
        bool(double_smoothing), np.ascontiguousarray(prior if prior is not None else np.ones(n), float),
        np.ascontiguousarray(Pi if Pi is not None else np.eye(1), float), float(lam),
        int(max_iterations), float(tol), int(kin_set_maa), float(clip_floor))
    return x, int(it), bool(conv), g.tolist(), c.tolist(), r.tolist(), (e.tolist() if mode else [])
