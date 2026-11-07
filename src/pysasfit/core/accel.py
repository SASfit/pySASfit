"""Shared optional accelerators for the pySASfit tools (Numba, CPU, all cores).

Numba is optional (``pip install numba``; MSYS2: pacman).  Without it, or with ``PYSASFIT_NO_NUMBA=1``,
:data:`HAVE_NUMBA` is False and every function here falls back to NumPy with identical results (to rounding).

* :func:`batched_solve` -- many small linear systems M[k] X[k] = B[k] (OZ equations per q point, ...):
  LAPACK through ``np.linalg.solve`` spends most of its time in per-matrix overhead for p <= 10; the compiled
  kernel (Gaussian elimination with partial pivoting, parallel over k) is 2-5x faster there.
* :func:`parallel_map` -- independent tasks (bootstrap replicates, parameter scans) on a thread pool; NumPy,
  SciPy and the compiled kernels release the GIL, so threads scale without pickling.
"""
from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor

import numpy as np

HAVE_NUMBA = False
if not os.environ.get("PYSASFIT_NO_NUMBA"):
    try:
        import numba as _nb
        HAVE_NUMBA = True
    except Exception:  # noqa: BLE001
        HAVE_NUMBA = False

__all__ = ["HAVE_NUMBA", "batched_solve", "parallel_map", "n_workers", "single_thread_blas_env"]


def n_workers(requested=None):
    """Worker count: ``requested`` if > 0, else the PYSASFIT_THREADS environment variable, else the CPU count."""
    if requested and int(requested) > 0:
        return int(requested)
    env = os.environ.get("PYSASFIT_THREADS")
    if env and env.isdigit() and int(env) > 0:
        return int(env)
    return max(1, os.cpu_count() or 1)


_BLAS_VARS = ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS",
              "NUMBA_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")


class single_thread_blas_env:
    """Context manager: worker processes started inside it use one BLAS / OpenMP / Numba thread each.

    Every process otherwise starts OpenBLAS with one thread per core; N worker processes then run N x N
    threads, each with its own buffers -- on Windows this ends in "OpenBLAS error: Memory allocation still
    failed after 10 retries".  The variables are read when the child imports NumPy, so they are set in the
    parent while the pool is created and restored afterwards."""

    def __enter__(self):
        self._old = {k: os.environ.get(k) for k in _BLAS_VARS}
        for k in _BLAS_VARS:
            os.environ[k] = "1"
        return self

    def __exit__(self, *exc):
        for k, v in self._old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        return False


if HAVE_NUMBA:
    @_nb.njit(parallel=True, cache=True)
    def _solve_kernel(M, B):
        n, p, _ = M.shape
        m = B.shape[2]
        X = np.empty((n, p, m))
        for k in _nb.prange(n):
            A = M[k].copy()
            Y = B[k].copy()
            for i in range(p):
                piv = i
                big = abs(A[i, i])
                for r in range(i + 1, p):
                    if abs(A[r, i]) > big:
                        big = abs(A[r, i])
                        piv = r
                if piv != i:
                    for c in range(p):
                        t = A[i, c]
                        A[i, c] = A[piv, c]
                        A[piv, c] = t
                    for c in range(m):
                        t = Y[i, c]
                        Y[i, c] = Y[piv, c]
                        Y[piv, c] = t
                inv = 1.0 / A[i, i]
                for r in range(i + 1, p):
                    f = A[r, i] * inv
                    if f != 0.0:
                        for c in range(i, p):
                            A[r, c] -= f * A[i, c]
                        for c in range(m):
                            Y[r, c] -= f * Y[i, c]
            # back substitution in the column order of LAPACK's dtrsm (divide, then update the rows above)
            for c in range(m):
                for i in range(p - 1, -1, -1):
                    if Y[i, c] != 0.0:
                        Y[i, c] /= A[i, i]
                        for r in range(i):
                            Y[r, c] -= Y[i, c] * A[r, i]
                for i in range(p):
                    X[k, i, c] = Y[i, c]
        return X


def batched_solve(M, B, max_p=12):
    """X with M[k] @ X[k] = B[k] for M (n, p, p), B (n, p, m) or (n, p).  Real float64 only.

    Compiled kernel for p <= max_p when Numba is present (faster than LAPACK there); ``np.linalg.solve``
    otherwise.  A singular matrix gives inf/NaN in the kernel (np.linalg.solve raises instead)."""
    M = np.asarray(M)
    B = np.asarray(B)
    vec = B.ndim == M.ndim - 1
    if (HAVE_NUMBA and M.ndim == 3 and M.shape[-1] <= max_p and np.isrealobj(M) and np.isrealobj(B)):
        Bm = B[..., None] if vec else B
        X = _solve_kernel(np.ascontiguousarray(M, np.float64), np.ascontiguousarray(Bm, np.float64))
        return X[..., 0] if vec else X
    if vec:
        return np.linalg.solve(M, B[..., None])[..., 0]
    return np.linalg.solve(M, B)


def parallel_map(fn, items, workers=None):
    """[fn(x) for x in items] on a thread pool (order kept). Exceptions are returned in place of the result
    so that one failing task does not discard the others."""
    items = list(items)
    w = min(n_workers(workers), max(1, len(items)))

    def safe(x):
        try:
            return fn(x)
        except Exception as e:  # noqa: BLE001
            return e
    if w <= 1:
        return [safe(x) for x in items]
    with ThreadPoolExecutor(max_workers=w) as ex:
        return list(ex.map(safe, items))
