# SPDX-License-Identifier: BSD-3-Clause
"""Radial and sector averaging (correction step 14) as an explicit sparse
linear map, so that the exact covariance of the averaged curve follows from
the pixel sensitivities.

Phase 1 assigns each pixel by its centre (no pixel splitting yet).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import scipy.sparse as sp

from ..data.uarray import UArray

__all__ = ["Binning", "radial_binning", "q_edges"]


@dataclass
class Binning:
    A: sp.csr_matrix            # (nbins, npixels): mean over the pixels of each bin
    edges: np.ndarray
    q_mean: np.ndarray          # mean pixel q in each bin
    q_var_in_bin: np.ndarray    # variance of pixel q inside each bin
    n_pixels: np.ndarray
    keep: np.ndarray            # bins with at least one pixel

    def apply(self, image: UArray) -> UArray:
        return image.reshape((image.size,)).linear(self.A)


def q_edges(qmin, qmax, n, spacing="log"):
    return np.geomspace(qmin, qmax, n + 1) if spacing == "log" else np.linspace(qmin, qmax, n + 1)


def radial_binning(q: np.ndarray, valid: np.ndarray, edges: np.ndarray,
                   phi: Optional[np.ndarray] = None, sector_deg: Optional[tuple] = None,
                   drop_empty: bool = True) -> Binning:
    """Mean of the valid pixels in each q bin, optionally restricted to a
    sector (centre angle, half width, in degrees, symmetric pair included)."""
    q = np.asarray(q).ravel()
    ok = np.asarray(valid, bool).ravel().copy()
    if sector_deg is not None:
        if phi is None:
            raise ValueError("sector averaging needs the pixel azimuth phi")
        c, hw = np.deg2rad(sector_deg[0]), np.deg2rad(sector_deg[1])
        dphi = np.angle(np.exp(1j * (np.asarray(phi).ravel() - c)))
        ok &= (np.abs(dphi) <= hw) | (np.abs(np.abs(dphi) - np.pi) <= hw)
    b = np.searchsorted(edges, q, side="right") - 1
    ok &= (b >= 0) & (b < len(edges) - 1)
    nb = len(edges) - 1
    n = np.bincount(b[ok], minlength=nb).astype(float)
    keep = n > 0 if drop_empty else np.ones(nb, bool)
    row_of_bin = -np.ones(nb, int)
    row_of_bin[keep] = np.arange(keep.sum())
    rows = row_of_bin[b[ok]]
    cols = np.nonzero(ok)[0]
    vals = 1.0 / n[b[ok]]
    A = sp.csr_matrix((vals, (rows, cols)), shape=(int(keep.sum()), q.size))
    qs = np.bincount(b[ok], weights=q[ok], minlength=nb)
    qs2 = np.bincount(b[ok], weights=q[ok] ** 2, minlength=nb)
    with np.errstate(invalid="ignore", divide="ignore"):
        qm = qs / n
        qv = qs2 / n - qm ** 2
    return Binning(A, edges, qm[keep], np.maximum(qv[keep], 0.0), n[keep], keep)
