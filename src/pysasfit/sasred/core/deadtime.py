# SPDX-License-Identifier: BSD-3-Clause
"""Dead-time correction (correction step 4), non-paralysable model.

    N_true,i = N_i * f,   f = 1 / (1 - tau * R),   R = (sum of counts) / t

for a multiwire detector (one global rate). For tube detectors the rate is
taken per tube (``axis``). The correction is applied to the UArray, so the
exact sensitivity df/dN_j = f^2 tau / t of every pixel to every other pixel's
counts (a rank-1 coupling) and the sensitivity to tau are propagated.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import scipy.sparse as sp

from ..data.uarray import UArray, constant

__all__ = ["nonparalysable", "monitor_deadtime"]


def nonparalysable(counts: UArray, time_s: float, tau: UArray | float,
                   axis: Optional[int] = None, flag_limit: float = 1.2):
    """Return (corrected counts, factor, flag mask).

    Arguments::

        counts : UArray of raw counts (detector image, shape (ny, nx))
        time_s : counting time in s (taken as exact)
        tau    : dead time in s, a float or a UArray carrying its uncertainty
        axis   : None for a global rate (multiwire); 0 or 1 for a per-tube rate,
                 summing along that axis
        flag_limit : pixels whose correction factor exceeds this are flagged
    """
    tau = tau if isinstance(tau, UArray) else constant(tau)
    if axis is None:
        total = counts.sum()
    else:
        ny, nx = counts.shape
        idx = np.arange(counts.size).reshape(ny, nx)
        # one row per tube, then broadcast back to the pixels of that tube
        if axis == 0:
            rows = [idx[:, j] for j in range(nx)]
            tube_of = np.tile(np.arange(nx), ny)
        else:
            rows = [idx[i, :] for i in range(ny)]
            tube_of = np.repeat(np.arange(ny), nx)
        A = sp.lil_matrix((len(rows), counts.size))
        for r, cols in enumerate(rows):
            A[r, cols] = 1.0
        sums = counts.linear(A.tocsr())
        B = sp.csr_matrix((np.ones(counts.size), (np.arange(counts.size), tube_of)),
                          shape=(counts.size, len(rows)))
        total = sums.linear(B, out_shape=counts.shape)
    rate = total / time_s
    loss = tau * rate
    if np.any(loss.value >= 1.0):
        raise ValueError("dead-time loss >= 1: rate beyond the non-paralysable model")
    factor = 1.0 / (1.0 - loss)
    flags = np.broadcast_to(factor.value > flag_limit, counts.shape)
    return counts * factor, factor, flags


def monitor_deadtime(monitor: UArray, time_s: float, tau: UArray | float):
    """Non-paralysable dead time of a monitor: M_true = M / (1 - tau M / t)."""
    tau = tau if isinstance(tau, UArray) else constant(tau)
    loss = tau * monitor / time_s
    if np.any(loss.value >= 1.0):
        raise ValueError("monitor dead-time loss >= 1")
    return monitor / (1.0 - loss)
