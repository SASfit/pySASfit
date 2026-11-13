"""Scoring of fitted candidates: information criteria, Akaike weights and a runs test of the residuals.

The criteria use chi^2 as -2 ln L (Gaussian errors with the stated sigma), so they depend on correct error
bars: with underestimated errors every chi^2 difference is too large and extra parameters look justified.
``error_scale`` (see :func:`pysasfit.scout.search`) divides the residuals by a common factor to correct this.
"""

from __future__ import annotations

import math

import numpy as np

__all__ = ["aic", "aicc", "bic", "akaike_weights", "runs_test"]


def aic(chi2, k):
    return chi2 + 2.0 * k


def aicc(chi2, k, n):
    """AIC with the small-sample correction (infinite when n <= k + 1)."""
    if n - k - 1 <= 0:
        return math.inf
    return chi2 + 2.0 * k + 2.0 * k * (k + 1.0) / (n - k - 1.0)


def bic(chi2, k, n):
    return chi2 + k * math.log(n)


def akaike_weights(values):
    """Relative likelihood of each model, exp(-Delta/2) normalised to 1 (``values``: AICc or BIC)."""
    v = np.asarray(values, float)
    out = np.zeros(v.size)
    ok = np.isfinite(v)
    if ok.any():
        w = np.exp(-0.5 * (v[ok] - v[ok].min()))
        out[ok] = w / w.sum()
    return out


def runs_test(residuals):
    """Wald-Wolfowitz runs test of the signs of the residuals: ``(z, p)``.

    z < 0 means fewer runs than expected for independent residuals, i.e. systematic deviations
    (the model misses a feature of the curve); p is the two-sided probability.
    """
    sign = np.sign(np.asarray(residuals, float))
    sign = sign[sign != 0]
    n1, n2 = int(np.sum(sign > 0)), int(np.sum(sign < 0))
    n = n1 + n2
    if n1 == 0 or n2 == 0 or n < 10:
        return (-math.inf if n >= 10 else math.nan), (0.0 if n >= 10 else math.nan)
    runs = 1 + int(np.sum(sign[1:] != sign[:-1]))
    mean = 2.0 * n1 * n2 / n + 1.0
    var = 2.0 * n1 * n2 * (2.0 * n1 * n2 - n) / (n * n * (n - 1.0))
    if var <= 0:
        return math.nan, math.nan
    z = (runs - mean) / math.sqrt(var)
    return float(z), float(math.erfc(abs(z) / math.sqrt(2.0)))
