"""Vectorised fixed-node quadrature for the Python ports of the SASfit plugins.

SASfit evaluates its one-dimensional integrals point by point with adaptive routines (``sasfit_integrate``). A
port evaluates the integrand for all q (and ψ) at once on a fixed set of nodes instead.

:func:`tanh_sinh_01` returns double-exponential (tanh-sinh) nodes on [0, 1]. The nodes cluster doubly
exponentially at both end points, so integrands with end-point power singularities or narrow end-point layers
(as in the ring-polymer kernels exp(-mu (x(1-x))^a) for large mu) are integrated to near machine precision with
a few hundred nodes. Both x and 1-x are returned separately, accurate also next to the end points.
"""

from __future__ import annotations

import itertools
from functools import lru_cache

import numpy as np

__all__ = [
    "gauss_legendre",
    "integrate_01",
    "integrate_segments",
    "tanh_sinh_0X",
    "tanh_sinh_01",
    "tanh_sinh_ab",
]


@lru_cache(maxsize=16)
def _nodes(h, tmax):
    t = np.arange(-tmax, tmax + 0.5 * h, h)
    u = 0.5 * np.pi * np.sinh(t)
    # x = (1 + tanh u)/2 = 1/(1 + e^{-2u}),  1 - x = 1/(1 + e^{2u})
    with np.errstate(over="ignore"):
        x = 1.0 / (1.0 + np.exp(-2.0 * u))
        xc = 1.0 / (1.0 + np.exp(2.0 * u))
        w = h * 0.25 * np.pi * np.cosh(t) / np.cosh(u) ** 2
    keep = (x > 0.0) & (xc > 0.0) & (w > 0.0)
    for a in (x, xc, w):
        a.setflags(write=False)
    return x[keep], xc[keep], w[keep]


def tanh_sinh_01(h=1.0 / 32.0, tmax=3.6):
    """Nodes x, complementary nodes 1-x and weights of the tanh-sinh rule on [0, 1] (read-only arrays).

    The default (step h = 1/32, about 230 nodes) integrates smooth integrands and end-point singularities of the
    type x^a (a > -1) to about 1e-13 relative accuracy.
    """
    return _nodes(float(h), float(tmax))


def integrate_01(f, h=1.0 / 32.0, tmax=3.6):
    """∫_0^1 f(x, 1-x) dx with the tanh-sinh rule; ``f`` is called once with the node arrays.

    ``f`` may broadcast: if it returns an array of shape (..., n_nodes), the result has shape (...).
    """
    x, xc, w = tanh_sinh_01(h, tmax)
    return np.asarray(f(x, xc)) @ w


@lru_cache(maxsize=32)
def _gl(n):
    x, w = np.polynomial.legendre.leggauss(int(n))
    x.setflags(write=False)
    w.setflags(write=False)
    return x, w


def gauss_legendre(n, a, b):
    """Gauss-Legendre nodes and weights on [a, b] (a, b scalars or broadcastable arrays; nodes on the last axis)."""
    x, w = _gl(n)
    a = np.asarray(a, float)[..., None]
    b = np.asarray(b, float)[..., None]
    half = 0.5 * (b - a)
    return a + half * (x + 1.0), half * w


def tanh_sinh_0X(X, h=1.0 / 32.0, tmax=3.6):
    """tanh-sinh nodes x (shape X.shape + (n,)), 1-x and weights for ∫_0^X with X <= 1 per point.

    Used for integrands with a narrow end-point layer at 0 whose width is known: X is the point beyond which the
    integrand is negligible, so the nodes resolve the layer for every q.
    """
    t, _, w = tanh_sinh_01(h, tmax)
    X = np.asarray(X, float)[..., None]
    x = X * t
    return x, 1.0 - x, X * w


def tanh_sinh_ab(a, b, h=1.0 / 32.0, tmax=3.6):
    """tanh-sinh nodes x, 1-x and weights on [a, b] per point (a, b broadcastable arrays; nodes on the last axis).

    1-x is computed as (1-b) + (b-a)(1-t), accurate next to x = 1 when b = 1.
    """
    t, tc, w = tanh_sinh_01(h, tmax)
    a = np.asarray(a, float)[..., None]
    b = np.asarray(b, float)[..., None]
    L = b - a
    return a + L * t, (1.0 - b) + L * tc, L * w


def integrate_segments(f, edges, h=1.0 / 32.0):
    """Σ over consecutive segments [edges[k], edges[k+1]] of ∫ f(x, 1-x) dx (tanh-sinh on each segment).

    ``edges`` is a list of scalars or arrays (broadcast against each other); ``f`` returns values with the nodes on
    the last axis. Splitting at the width of an end-point layer resolves the layer for every point.
    """
    total = 0.0
    for a, b in itertools.pairwise(edges):
        x, xc, w = tanh_sinh_ab(a, b, h)
        total = total + np.sum(f(x, xc) * w, axis=-1)
    return total
