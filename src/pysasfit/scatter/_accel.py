"""Optional compiled kernels (Numba) for the slowest inner loops of pySCAtter.

Numba is optional (``pip install numba``, or the MSYS2 / conda package).  When it is missing, or when
``PYSASFIT_NO_NUMBA=1`` is set, :data:`HAVE_NUMBA` is False and the callers use their NumPy code.  The kernels
compute the same sums as the NumPy code (same nodes and weights), in parallel over the q points / pixels and
without large temporary arrays.
"""
from __future__ import annotations

import os

import numpy as np

HAVE_NUMBA = False
if not os.environ.get("PYSASFIT_NO_NUMBA"):
    try:
        import numba as _nb
        HAVE_NUMBA = True
    except Exception:  # noqa: BLE001
        HAVE_NUMBA = False

__all__ = ["HAVE_NUMBA", "box_iso", "box_fixed", "superball_table", "threads"]


def threads():
    """Number of threads of the compiled kernels (1 without Numba)."""
    return int(_nb.get_num_threads()) if HAVE_NUMBA else 1


if HAVE_NUMBA:
    @_nb.njit(cache=True, inline="always")
    def _sinc(u):
        if abs(u) < 1e-8:
            return 1.0
        return np.sin(u) / u

    @_nb.njit(parallel=True, cache=True)
    def _box_iso(x, b, c, xi, w, ax, ay, az, W, power):
        out = np.empty(x.size)
        for i in _nb.prange(x.size):
            s = 0.0
            for a in range(W.size):
                acc = 0.0
                for k in range(xi.size):
                    t = x[i] * xi[k]
                    f = _sinc(t * ax[a]) * _sinc(t * b * ay[a]) * _sinc(t * c * az[a])
                    acc += w[k] * (f * f if power == 2 else f)
                s += W[a] * acc
            out[i] = s
        return out

    @_nb.njit(parallel=True, cache=True)
    def _box_fixed(X, Y, Z, xi, w, power):
        out = np.empty(X.size)
        for i in _nb.prange(X.size):
            acc = 0.0
            for k in range(xi.size):
                f = _sinc(X[i] * xi[k]) * _sinc(Y[i] * xi[k]) * _sinc(Z[i] * xi[k])
                acc += w[k] * (f * f if power == 2 else f)
            out[i] = acc
        return out

    @_nb.njit(parallel=True, cache=True)
    def _superball_amp(nx, dx, qx, qy, qz, Xf, Yf, Zf, Wf, scale):
        """A[o, k] = scale sum_j cos(x qx X_j) cos(x qy Y_j) sin(x qz Z_j)/(x qz) W_j at x = k dx; the cosines and
        sines along the uniform x grid by the Chebyshev recurrence, re-seeded every 32 steps."""
        no = qx.size
        A = np.zeros((no, nx))
        for o in _nb.prange(no):
            for j in range(Xf.size):
                tx = dx * qx[o] * Xf[j]
                ty = dx * qy[o] * Yf[j]
                tz = dx * qz[o] * Zf[j]
                c1x, c1y, c1z = np.cos(tx), np.cos(ty), np.cos(tz)
                wj = Wf[j] * scale
                k = 0
                while k < nx:
                    # seed at k: cos(k t), cos((k-1) t); sin(k tz), sin((k-1) tz)
                    cx0, cx1 = np.cos(k * tx), np.cos((k - 1) * tx)
                    cy0, cy1 = np.cos(k * ty), np.cos((k - 1) * ty)
                    sz0, sz1 = np.sin(k * tz), np.sin((k - 1) * tz)
                    kend = min(nx, k + 32)
                    while k < kend:
                        x = k * dx
                        az = x * qz[o]
                        if abs(az) < 1e-9:
                            zint = Zf[j]
                        else:
                            zint = sz0 / az
                        A[o, k] += cx0 * cy0 * zint * wj
                        cx0, cx1 = 2.0 * c1x * cx0 - cx1, cx0
                        cy0, cy1 = 2.0 * c1y * cy0 - cy1, cy0
                        sz0, sz1 = 2.0 * c1z * sz0 - sz1, sz0
                        k += 1
        return A


def superball_table(nx, dx, qx, qy, qz, W, Xf, Yf, Zf, Wf, vol):
    """(F1, F2) = orientation averages of A and A^2 on x = k dx (k < nx) for the superball table."""
    f = np.ascontiguousarray
    A = _superball_amp(int(nx), float(dx), f(qx, float), f(qy, float), f(qz, float), f(Xf, float), f(Yf, float),
                       f(Zf, float), f(Wf, float), 8.0 / float(vol))
    W = np.asarray(W, float)
    return W @ A, W @ (A * A)


def box_iso(x, b, c, xi, w, ax, ay, az, W, kind="P"):
    """sum_a W_a sum_k w_k [sinc(x xi_k ax_a) sinc(x b xi_k ay_a) sinc(x c xi_k az_a)]^p for every x = q R."""
    f = np.ascontiguousarray
    return _box_iso(f(x, float).ravel(), float(b), float(c), f(xi, float), f(w, float), f(ax, float).ravel(),
                    f(ay, float).ravel(), f(az, float).ravel(), f(W, float).ravel(), 2 if kind == "P" else 1)


def box_fixed(X, Y, Z, xi, w, kind="P"):
    """sum_k w_k [sinc(X xi_k) sinc(Y xi_k) sinc(Z xi_k)]^p (body-frame arguments q_i R e_i)."""
    f = np.ascontiguousarray
    X = np.asarray(X, float)
    return _box_fixed(f(X).ravel(), f(Y, float).ravel(), f(Z, float).ravel(), f(xi, float), f(w, float),
                      2 if kind == "P" else 1).reshape(X.shape)
