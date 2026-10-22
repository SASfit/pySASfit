r"""B-spline representation of the density profile.

Instead of N piecewise-constant shells the represented quantity
c(r) = r^p rho(r) (``basis`` 'rho' -> p = 0, 'mass' -> p = dim-1, or a number)
is a clamped B-spline of degree k on normalised breakpoints
0 = t_0 < ... < t_m = 1 (x = r / r_max)::

    c(r) = sum_i c_i B_i(r / r_max),         rho(r) = c(r) / r^p   (r <= r_max)

Why: every shell boundary is a small step in rho and adds its own Porod-like
high-q tail; with many shells their interference can fit noise at large q.
A cubic spline is C^2 inside [0, r_max]; the only step left is the (physical)
one at r_max when the outer end is free.

The amplitude stays linear in the coefficients, F(q) = sum_i a_i(q) c_i, with

    a_i(q) = A_d int_0^r_max B_i(r/r_max) r^(d-1-p) K(qr) dr

(A_d, K as for the shells), integrated by Gauss quadrature on every knot span
(Gauss-Jacobi with weight r^(d-1-p) on the first span, so that a centre cusp is
integrated exactly). Everything else -- the quadratic model, the evidence, the
background, the resolution -- is unchanged.

End conditions (hard, by the choice of the basis):

* centre: the represented quantity is even through r = 0, c'(0) = 0, i.e. the
  first two coefficients of the clamped basis are tied (B_0 + B_1);
* outer end ``'free'``: c(r_max) free, a step to 0 at r_max (sharp surface);
  ``'zero'``: c(r_max) = 0 (last basis function dropped).

Smoothness prior: S = (h^3/4) int_0^1 c''(x)^2 dx with h = 1/N (N = number of
coefficients) -- the continuum limit of Hansen's (2016) eq. 14 on a grid of N
points, evaluated exactly from the B-splines for any knot placement.

Usage: pass ``grid=BSplines(degree=3, knots=...)`` to
:class:`~pysasfit.sldprofile.DensityProfileFit` (or the two-step functions);
``n`` is then the number of spline coefficients.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np
from scipy import special
from scipy.interpolate import BSpline

__all__ = ["BSplines", "BSplineBasis"]


@lru_cache(maxsize=256)
def _legendre(n):
    return np.polynomial.legendre.leggauss(int(n))


@lru_cache(maxsize=256)
def _jacobi(n, beta):
    return special.roots_jacobi(int(n), 0.0, float(beta))


@dataclass(frozen=True)
class BSplines:
    """Specification of a B-spline representation (the ``grid`` argument).

    degree
        2 (quadratic) or 3 (cubic, default)
    knots
        None / 'uniform' (equidistant breakpoints), ('power', gamma)
        (t_j = (j/m)^gamma; gamma > 1 refines towards the centre, gamma < 1
        towards r_max), or an array of m+1 normalised breakpoints 0 ... 1
        (e.g. from :meth:`DensityProfileFit.adaptive_grid`); its length must
        match the number of coefficients, see :meth:`n_spans`.
    """

    degree: int = 3
    knots: object = None

    @staticmethod
    def n_spans(n: int, degree: int, outer: str = "free") -> int:
        """Number of knot spans m for n free coefficients (one tie at the centre,
        one dropped function for outer='zero')."""
        return n + 1 + (outer == "zero") - degree

    def build(self, n: int, outer: str = "free") -> "BSplineBasis":
        if self.degree not in (2, 3):
            raise ValueError("B-spline degree must be 2 or 3 (the smoothness prior needs c'')")
        m = self.n_spans(n, self.degree, outer)
        if m < 1:
            raise ValueError(f"{n} coefficients are too few for degree {self.degree}; "
                             f"need at least {self.degree + (outer == 'free')}")
        k = self.knots
        if k is None or (isinstance(k, str) and k == "uniform"):
            t = np.linspace(0.0, 1.0, m + 1)
        elif isinstance(k, tuple) and k and k[0] == "power":
            t = (np.arange(m + 1) / m) ** float(k[1])
        else:
            t = np.asarray(k, float)
            if t.size != m + 1:
                raise ValueError(f"{t.size} breakpoints given, {m + 1} needed for n={n}, "
                                 f"degree {self.degree}, outer={outer!r}")
        return BSplineBasis(t, self.degree, outer, n)


class BSplineBasis:
    """A built B-spline basis (the internal ``grid`` object of the fitter)."""

    is_bspline = True

    def __init__(self, breaks, degree: int = 3, outer: str = "free", n: int | None = None):
        t = np.asarray(breaks, float)
        if abs(t[0]) > 1e-12 or abs(t[-1] - 1) > 1e-12 or np.any(np.diff(t) <= 0):
            raise ValueError("breakpoints must increase from 0 to 1")
        if outer not in ("free", "zero"):
            raise ValueError("outer must be 'free' or 'zero'")
        self.breaks = t
        self.degree = int(degree)
        self.outer = outer
        k = self.degree
        self.knots = np.r_[np.zeros(k), t, np.ones(k)]
        nf = self.knots.size - k - 1                 # = m + k
        # map T: full clamped coefficients = T @ c (centre tie, optional outer drop)
        cols = []
        for i in range(nf):
            if i == 0:
                continue
            if outer == "zero" and i == nf - 1:
                continue
            col = np.zeros(nf)
            col[i] = 1.0
            if i == 1:
                col[0] = 1.0
            cols.append(col)
        self.T = np.array(cols).T
        self.n_full = nf
        self.n = self.T.shape[1]
        if n is not None and n != self.n:
            raise ValueError(f"basis has {self.n} coefficients, expected {n}")
        self._spl = BSpline(self.knots, np.eye(nf), k, extrapolate=False)
        self._memo = {}

    # ------------------------------------------------------------------ #
    def __repr__(self):
        return f"BSplineBasis(degree={self.degree}, n={self.n}, spans={self.breaks.size - 1}, outer={self.outer!r})"

    @property
    def spec(self) -> BSplines:
        """The specification that rebuilds this basis."""
        return BSplines(self.degree, self.breaks.copy())

    def respec(self, n: int, outer: str) -> BSplines:
        """Specification of the same knot distribution for another number of
        coefficients or another outer end condition (the number of knot spans
        changes): the breakpoints are resampled along their cumulative distribution."""
        m = BSplines.n_spans(n, self.degree, outer)
        if m == self.breaks.size - 1:
            return self.spec
        u = np.linspace(0.0, 1.0, self.breaks.size)
        t = np.interp(np.linspace(0.0, 1.0, m + 1), u, self.breaks)
        t[0], t[-1] = 0.0, 1.0
        return BSplines(self.degree, t)

    def values(self, x, nu: int = 0) -> np.ndarray:
        """(len(x), n) matrix of the basis functions (or their nu-th derivative,
        w.r.t. x) at normalised positions x; 0 outside [0, 1]."""
        x = np.atleast_1d(np.asarray(x, float))
        inside = (x >= 0) & (x <= 1)
        V = np.zeros((x.size, self.n_full))
        if inside.any():
            V[inside] = np.nan_to_num(self._spl(x[inside], nu=nu))
        return V @ self.T

    def greville(self) -> np.ndarray:
        """Normalised positions of the coefficients (Greville abscissae, averaged over ties)."""
        k = self.degree
        g = np.array([self.knots[i + 1:i + k + 1].mean() for i in range(self.n_full)])
        w = self.T / self.T.sum(axis=0, keepdims=True)
        return w.T @ g

    # ------------------------------------------------------------------ #
    def _nodes(self, r_max, beta, n_per_span):
        """Quadrature nodes r and weights w (including r^beta) over [0, r_max]."""
        e = self.breaks * r_max
        rs, ws = [], []
        for j in range(e.size - 1):
            a, b = e[j], e[j + 1]
            ng = int(n_per_span[j])
            if j == 0:
                xj, wj = _jacobi(ng, beta)
                rs.append(b * (1 + xj) / 2)
                ws.append((b / 2) ** (beta + 1) * wj)
            else:
                xl, wl = _legendre(ng)
                rr = 0.5 * (b - a) * xl + 0.5 * (b + a)
                rs.append(rr)
                ws.append(0.5 * (b - a) * wl * rr**beta)
        return np.concatenate(rs), np.concatenate(ws)

    def _n_per_span(self, r_max, q_max, n_gl):
        span = np.diff(self.breaks) * r_max
        # >= n_gl, and >= 4 nodes per half period of the fastest kernel oscillation
        return np.maximum(int(n_gl), np.ceil(4 * q_max * span / np.pi).astype(int) + self.degree + 2)

    def amplitude(self, q, r_max, geometry, p, n_gl=16):
        """a (len(q) x n) with F(q) = a @ c."""
        from .geometry import DIM, _ANG, _kernel
        q = np.atleast_1d(np.asarray(q, float))
        if not np.any(q):                                  # F(0): plain moments
            key = ("a0", float(r_max), geometry, float(p), int(n_gl))
            if key not in self._memo:
                if len(self._memo) > 256:
                    self._memo.clear()
                self._memo[key] = _ANG[geometry] * self.moment(r_max, DIM[geometry] - 1 - p, n_gl)
            return np.repeat(self._memo[key][None, :], q.size, axis=0)
        beta = DIM[geometry] - 1 - p
        rn, wn = self._nodes(r_max, beta, self._n_per_span(r_max, float(np.max(np.abs(q))), n_gl))
        B = self.values(rn / r_max)                       # (nodes, n)
        K = _kernel(geometry, np.outer(q, rn))            # (nq, nodes)
        return _ANG[geometry] * (K * wn[None, :]) @ B

    def moment(self, r_max, power, n_gl=16):
        """int_0^r_max B_i(r/r_max) r^power dr  (power > -1), vector of length n."""
        rn, wn = self._nodes(r_max, power, np.full(self.breaks.size - 1, int(n_gl)))
        return wn @ self.values(rn / r_max)

    def penalty(self, n_gl=None) -> np.ndarray:
        """Exact smoothness matrix L with S = c^T L c = (h^3/4) int_0^1 c''(x)^2 dx, h = 1/n."""
        ng = self.degree + 1 if n_gl is None else n_gl
        xs, ws = [], []
        xl, wl = np.polynomial.legendre.leggauss(ng)
        for a, b in zip(self.breaks[:-1], self.breaks[1:]):
            xs.append(0.5 * (b - a) * xl + 0.5 * (b + a))
            ws.append(0.5 * (b - a) * wl)
        x, w = np.concatenate(xs), np.concatenate(ws)
        D2 = self.values(x, nu=2)
        h = 1.0 / self.n
        return (h**3 / 4.0) * (D2.T * w) @ D2
