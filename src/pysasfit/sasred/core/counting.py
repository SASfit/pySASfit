# SPDX-License-Identifier: BSD-3-Clause
"""Counting error of raw data (correction step 3).

Two exact models (doc: "Counting error of raw data"):

* ``variance`` (default): sigma^2 = N, the exact variance of a Poisson count,
  the correct input for propagation and binning because variances add.
* ``interval``: the exact central 1-sigma Poisson confidence interval
  (Garwood), alpha = Phi(-1) per tail. Its half-width is for display and
  single-pixel use; it is not used for propagation.
"""
from __future__ import annotations

import numpy as np
from scipy.special import gammaincinv
from scipy.stats import norm

from ..data.uarray import Source, UArray

__all__ = ["counts_uarray", "garwood_interval", "ALPHA_1SIGMA"]

#: one-sided tail probability of a central 1-sigma interval, Phi(-1)
ALPHA_1SIGMA = float(norm.cdf(-1.0))   # 0.158655...


def _check_raw(counts: np.ndarray) -> np.ndarray:
    c = np.asarray(counts)
    if np.any(c < 0):
        raise ValueError("negative counts: the input is not raw data")
    if np.issubdtype(c.dtype, np.floating) and np.any(c != np.round(c)):
        raise ValueError("non-integer counts: the input is not raw data")
    return c.astype(float)


def counts_uarray(counts, key: str, unit: str = "counts") -> UArray:
    """Raw counts as a UArray whose own elements are the independent
    Poisson source `key` with variance N."""
    c = _check_raw(counts)
    return UArray.from_source(c, Source(key, c.ravel().copy(), "counting", False), unit)


def garwood_interval(counts, alpha: float = ALPHA_1SIGMA):
    """Exact central Poisson confidence interval for observed counts N.

    lower = P^{-1}(N, alpha)        (0 for N = 0)
    upper = P^{-1}(N + 1, 1 - alpha)
    with P the regularised lower incomplete gamma function, which is the
    chi-square quantile form of Garwood (1936).

    Returns (lower, upper, sigma) with sigma = (upper - lower) / 2.
    """
    n = _check_raw(counts)
    lower = np.where(n > 0, gammaincinv(np.maximum(n, 1e-300), alpha), 0.0)
    upper = gammaincinv(n + 1.0, 1.0 - alpha)
    return lower, upper, 0.5 * (upper - lower)
