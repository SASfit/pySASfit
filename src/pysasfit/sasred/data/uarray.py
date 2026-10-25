# SPDX-License-Identifier: BSD-3-Clause
"""
UArray: a value array that carries its exact first-order sensitivities to
named, independent uncertainty sources.

Every quantity y in pySASred is represented as

    y = value,     dy = sum_s J_s ds,     ds ~ (0, diag(var_s)), sources independent,

where s runs over named sources (the counts of a run, a monitor count, a
transmission, a dead time, ...). A source is a vector of n_s independent
elements (n_s = 1 for a scalar such as a monitor count, n_s = number of pixels
for the counts of a detector image).

For each source the Jacobian J_s (shape out x n_s, out = y.size) is stored in a
structured form that stays small through every step of a SANS reduction:

    J_s = S_s + U_s V_s^T

with S_s a scipy.sparse CSR matrix (diagonal for pixel-wise steps, banded or
block-sparse after averaging) and U_s V_s^T a low-rank part (out x k, n_s x k).
A scalar source is a low-rank term with V = [[1]]. The low-rank part arises
when a pixel-wise quantity is multiplied by a function of an aggregate of the
same counts (for example a dead-time factor that depends on the total count
rate): it is the exact rank-1 coupling, never a dense N x N matrix.

Sensitivities to the same source add linearly, so a shared term (the Cd run in
sample and empty cell, the same monitor in numerator and denominator) is never
counted twice. Variances and covariances follow as

    Cov(y) = sum_s J_s diag(var_s) J_s^T .

The propagation is exact to first order. The only approximation is the
linearisation itself, which is validated against Monte Carlo in the tests.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, Iterable, Optional

import numpy as np
import scipy.sparse as sp

__all__ = ["Source", "Term", "UArray", "constant"]


# --------------------------------------------------------------------- sources
@dataclass(frozen=True)
class Source:
    """An independent uncertainty source.

    Fields::

        key        : unique name, e.g. "counting:82130" or "monitor1:82130"
        variance   : variance of each element, shape (n,) ; n = 1 for scalars
        group      : documentation key of the doc's source table: counting,
                     monitor, transmission, flood, abs_scale, thickness, deadtime,
                     attenuator, ...
        systematic : True for fully correlated scalar factors that are reported as
                     sensitivities and not folded into Idev
    """
    key: str
    variance: np.ndarray
    group: str = "counting"
    systematic: bool = False

    @property
    def n(self) -> int:
        return int(np.asarray(self.variance).size)


# ----------------------------------------------------------------------- terms
class Term:
    """Jacobian of an output (flattened, length `out`) with respect to one
    source (length n): J = S + U V^T."""

    __slots__ = ("S", "U", "V", "out", "n")

    def __init__(self, out: int, n: int, S=None, U=None, V=None):
        self.out, self.n = int(out), int(n)
        self.S = None if S is None else sp.csr_matrix(S)
        if U is None:
            self.U = self.V = None
        else:
            U = np.asarray(U, float).reshape(out, -1)
            V = np.asarray(V, float).reshape(n, -1)
            if U.shape[1] != V.shape[1]:
                raise ValueError("U and V must have the same rank")
            self.U, self.V = U, V

    # -- constructors
    @classmethod
    def diagonal(cls, d: np.ndarray) -> "Term":
        d = np.asarray(d, float).ravel()
        return cls(d.size, d.size, S=sp.diags(d, format="csr"))

    @classmethod
    def column(cls, u: np.ndarray) -> "Term":
        """Sensitivity of every output element to a scalar source."""
        u = np.asarray(u, float).ravel()
        return cls(u.size, 1, U=u[:, None], V=np.ones((1, 1)))

    # -- algebra
    def scale_rows(self, f) -> "Term":
        """diag(f) J ; f scalar or length-out vector."""
        f = np.asarray(f, float)
        if f.size == 1:
            fs = float(f.ravel()[0])
            S = None if self.S is None else self.S * fs
            U = None if self.U is None else self.U * fs
        else:
            f = f.ravel()
            if f.size != self.out:
                raise ValueError("row scale length mismatch")
            S = None if self.S is None else sp.diags(f) @ self.S
            U = None if self.U is None else self.U * f[:, None]
        return Term(self.out, self.n, S, U, None if U is None else self.V)

    def broadcast_rows(self, out: int) -> "Term":
        """A term of a size-1 output, repeated for `out` identical rows."""
        if self.out == out:
            return self
        if self.out != 1:
            raise ValueError("only size-1 outputs broadcast")
        ones = np.ones((out, 1))
        Us, Vs = [], []
        if self.S is not None:
            Us.append(ones)
            Vs.append(self.S.toarray().reshape(self.n, 1))
        if self.U is not None:
            Us.append(ones @ self.U)          # (out, k)
            Vs.append(self.V)
        if not Us:
            return Term(out, self.n)
        return Term(out, self.n, None, np.hstack(Us), np.hstack(Vs))

    def __add__(self, other: "Term") -> "Term":
        if (self.out, self.n) != (other.out, other.n):
            raise ValueError("term shape mismatch")
        S = self.S if other.S is None else (other.S if self.S is None else self.S + other.S)
        if self.U is None:
            U, V = other.U, other.V
        elif other.U is None:
            U, V = self.U, self.V
        else:
            U, V = np.hstack([self.U, other.U]), np.hstack([self.V, other.V])
            U, V = _compress_lowrank(U, V)
        return Term(self.out, self.n, S, U, V)

    def left_matmul(self, A) -> "Term":
        """A J for a (sparse or dense) linear map A of shape (m, out)."""
        A = sp.csr_matrix(A)
        m = A.shape[0]
        S = None if self.S is None else (A @ self.S).tocsr()
        U = None if self.U is None else np.asarray(A @ self.U)
        return Term(m, self.n, S, U, None if U is None else self.V)

    def dense(self) -> np.ndarray:
        J = np.zeros((self.out, self.n))
        if self.S is not None:
            J += self.S.toarray()
        if self.U is not None:
            J += self.U @ self.V.T
        return J

    # -- second moments
    def variance(self, var_src: np.ndarray) -> np.ndarray:
        """diag(J D J^T), D = diag(var_src); never forms J densely."""
        w = np.broadcast_to(np.asarray(var_src, float).ravel(), (self.n,))
        out = np.zeros(self.out)
        if self.S is not None:
            S2 = self.S.multiply(self.S)
            out += np.asarray(S2 @ w).ravel()
        if self.U is not None:
            G = (self.V * w[:, None]).T @ self.V        # k x k
            out += np.einsum("ik,kl,il->i", self.U, G, self.U)
            if self.S is not None:
                SDV = np.asarray(self.S @ (self.V * w[:, None]))   # out x k
                out += 2.0 * np.einsum("ik,ik->i", SDV, self.U)
        return out

    def covariance(self, var_src: np.ndarray) -> np.ndarray:
        """Full J D J^T (out x out). Use only for small outputs (1D curves)."""
        if self.out > 20000:
            raise MemoryError("covariance of a large output requested")
        w = np.broadcast_to(np.asarray(var_src, float).ravel(), (self.n,))
        C = np.zeros((self.out, self.out))
        if self.S is not None:
            C += (self.S @ sp.diags(w) @ self.S.T).toarray()
        if self.U is not None:
            G = (self.V * w[:, None]).T @ self.V
            C += self.U @ G @ self.U.T
            if self.S is not None:
                SDV = np.asarray(self.S @ (self.V * w[:, None]))
                X = SDV @ self.U.T
                C += X + X.T
        return C


def _compress_lowrank(U, V, max_rank: int = 8):
    """Keep the low-rank part small. Exact: U V^T is unchanged."""
    k = U.shape[1]
    if k <= max_rank:
        return U, V
    qu, ru = np.linalg.qr(U)
    qv, rv = np.linalg.qr(V)
    m = ru @ rv.T
    a, s, bt = np.linalg.svd(m, full_matrices=False)
    keep = s > s[0] * 1e-14 if s.size else s.astype(bool)
    return (qu @ (a[:, keep] * s[keep])), qv @ bt[keep].T


# ---------------------------------------------------------------------- UArray
@dataclass
class UArray:
    value: np.ndarray
    terms: Dict[str, Term] = field(default_factory=dict)
    sources: Dict[str, Source] = field(default_factory=dict)
    unit: str = ""

    def __post_init__(self):
        self.value = np.asarray(self.value, float)
        for k, t in self.terms.items():
            if t.out != self.value.size:
                raise ValueError(f"term {k} has {t.out} rows for value size {self.value.size}")

    # ---------------------------------------------------------- constructors
    @classmethod
    def from_source(cls, value, source: Source, unit: str = "") -> "UArray":
        """The value IS the source: dy = ds (identity Jacobian)."""
        value = np.asarray(value, float)
        if source.n == 1 and value.size == 1:
            t = Term.column(np.ones(1))
        elif source.n == value.size:
            t = Term.diagonal(np.ones(value.size))
        else:
            raise ValueError("source size must match value size")
        return cls(value, {source.key: t}, {source.key: source}, unit)

    @classmethod
    def scalar(cls, value: float, sigma: float, key: str, group: str,
               systematic: bool = True, unit: str = "") -> "UArray":
        if sigma is None or sigma == 0:
            return constant(value, unit)
        return cls.from_source(value, Source(key, np.array([float(sigma) ** 2]), group, systematic), unit)

    # ---------------------------------------------------------- properties
    @property
    def shape(self):
        return self.value.shape

    @property
    def size(self):
        return self.value.size

    def variance(self, groups: Optional[Iterable[str]] = None, systematic: Optional[bool] = None,
                 keys: Optional[Iterable[str]] = None) -> np.ndarray:
        """Variance of every element, optionally restricted to a subset of the
        sources (by group, by the systematic flag, or by key)."""
        out = np.zeros(self.size)
        for k in self._select(groups, systematic, keys):
            out += self.terms[k].variance(self.sources[k].variance)
        return out.reshape(self.shape)

    def std(self, **kw) -> np.ndarray:
        return np.sqrt(self.variance(**kw))

    def covariance(self, groups=None, systematic=None, keys=None) -> np.ndarray:
        C = np.zeros((self.size, self.size))
        for k in self._select(groups, systematic, keys):
            C += self.terms[k].covariance(self.sources[k].variance)
        return C

    def budget(self) -> Dict[str, np.ndarray]:
        """Variance contribution of every source; the parts add up exactly."""
        return {k: self.terms[k].variance(self.sources[k].variance).reshape(self.shape)
                for k in self.terms}

    def _select(self, groups, systematic, keys):
        sel = []
        for k, s in self.sources.items():
            if k not in self.terms:
                continue
            if keys is not None and k not in set(keys):
                continue
            if groups is not None and s.group not in set(groups):
                continue
            if systematic is not None and s.systematic != systematic:
                continue
            sel.append(k)
        return sel

    def jacobian(self, key: str) -> np.ndarray:
        return self.terms[key].dense()

    # ---------------------------------------------------------- arithmetic
    def _lift(self, other) -> "UArray":
        return other if isinstance(other, UArray) else constant(other)

    @staticmethod
    def _combine(a: "UArray", b: "UArray", value, da, db, unit="") -> "UArray":
        """value = f(a, b) with partial derivatives da = df/da, db = df/db
        (arrays broadcastable to value)."""
        value = np.asarray(value, float)
        out = value.size
        da = np.broadcast_to(np.asarray(da, float), value.shape).ravel()
        db = np.broadcast_to(np.asarray(db, float), value.shape).ravel()
        terms: Dict[str, Term] = {}
        for x, d in ((a, da), (b, db)):
            if not np.any(d) and x.terms:
                continue
            for k, t in x.terms.items():
                t2 = t.broadcast_rows(out).scale_rows(d)
                terms[k] = terms[k] + t2 if k in terms else t2
        sources = {**a.sources, **b.sources}
        for k in sources:
            if k in a.sources and k in b.sources:
                _check_same_source(a.sources[k], b.sources[k])
        return UArray(value, terms, sources, unit)

    def __add__(self, other):
        o = self._lift(other)
        return self._combine(self, o, self.value + o.value, 1.0, 1.0, self.unit)

    __radd__ = __add__

    def __sub__(self, other):
        o = self._lift(other)
        return self._combine(self, o, self.value - o.value, 1.0, -1.0, self.unit)

    def __rsub__(self, other):
        return self._lift(other) - self

    def __neg__(self):
        return self * -1.0

    def __mul__(self, other):
        o = self._lift(other)
        return self._combine(self, o, self.value * o.value, o.value, self.value)

    __rmul__ = __mul__

    def __truediv__(self, other):
        o = self._lift(other)
        v = self.value / o.value
        return self._combine(self, o, v, 1.0 / o.value, -v / o.value)

    def __rtruediv__(self, other):
        return self._lift(other) / self

    def apply(self, f: Callable, dfdx: Callable, unit: str = "") -> "UArray":
        """Elementwise y = f(x) with derivative dfdx(x)."""
        v = f(self.value)
        return self._combine(self, constant(0.0), v, dfdx(self.value), 0.0, unit)

    def sqrt(self):
        return self.apply(np.sqrt, lambda x: 0.5 / np.sqrt(x))

    def log(self):
        return self.apply(np.log, lambda x: 1.0 / x)

    def exp(self):
        return self.apply(np.exp, np.exp)

    # ---------------------------------------------------------- linear maps
    def linear(self, A, out_shape=None) -> "UArray":
        """y = A x with a (sparse) matrix A of shape (m, x.size)."""
        A = sp.csr_matrix(A)
        if A.shape[1] != self.size:
            raise ValueError("matrix width must equal array size")
        v = np.asarray(A @ self.value.ravel()).ravel()
        if out_shape is not None:
            v = v.reshape(out_shape)
        terms = {k: t.left_matmul(A) for k, t in self.terms.items()}
        return UArray(v, terms, dict(self.sources), self.unit)

    def sum(self, mask: Optional[np.ndarray] = None) -> "UArray":
        w = np.ones(self.size) if mask is None else np.asarray(mask, float).ravel()
        return self.linear(sp.csr_matrix(w[None, :]), out_shape=())

    def reshape(self, shape) -> "UArray":
        return UArray(self.value.reshape(shape), self.terms, self.sources, self.unit)

    def __repr__(self):
        return (f"UArray(shape={self.shape}, unit='{self.unit}', sources="
                f"{sorted(self.terms)})")


def constant(value, unit: str = "") -> UArray:
    """A quantity without uncertainty."""
    return UArray(np.asarray(value, float), {}, {}, unit)


def _check_same_source(a: Source, b: Source):
    if a is b:
        return
    if a.n != b.n or not np.array_equal(np.asarray(a.variance), np.asarray(b.variance)):
        raise ValueError(f"two different sources share the key '{a.key}'")
