"""Scattering of the reconstructed structure, for comparison with the data.

Two curves, both in the absolute scale of the data (the contrast follows from the
data invariant, Delta rho^2 = Q / (2 pi^2 phi (1-phi))):

``voxel_intensity``
    radially averaged |FFT|^2 of the binary voxel matrix -- what this particular
    realisation scatters (speckle at low q, voxel effects near the Nyquist q = pi/a);
``levelcut_intensity``
    the ensemble average of the level-cut model with the reconstructed g(r):
    4 pi integral r^2 F(g(r)) sin(qr)/(qr) dr. It reproduces the data wherever
    gamma(r) could be inverted without clipping.
"""

from __future__ import annotations

import numpy as np

from .transforms import sine_transform_pl

__all__ = ["voxel_intensity", "levelcut_intensity", "contrast_from_invariant"]


def contrast_from_invariant(Q, porosity):
    """Delta rho^2 (in the units of the data) such that Q = 2 pi^2 Delta rho^2 phi (1-phi)."""
    return Q / (2 * np.pi ** 2 * porosity * (1 - porosity))


def voxel_intensity(mat, a, drho2=1.0, nbins=None, periodic=True):
    """Radially averaged I(q) = drho2 * |FT(eta - <eta>)|^2 / V of a binary n^3 matrix.

    Returns (q, I, counts); bins are linear in |q| with width 2 pi / L up to the
    corner of the reciprocal grid (q values above pi/a are affected by the voxel size).
    """
    mat = np.asarray(mat)
    n = mat.shape[0]
    if mat.shape != (n, n, n):
        raise ValueError("cubic matrix expected")
    eta = mat.astype(np.float64)
    eta -= eta.mean()
    F = np.fft.rfftn(eta)
    P = (np.abs(F) ** 2) * a ** 3 / n ** 3
    kx = 2 * np.pi * np.fft.fftfreq(n, a)
    kz = 2 * np.pi * np.fft.rfftfreq(n, a)
    kk = np.sqrt(kx[:, None, None] ** 2 + kx[None, :, None] ** 2 + kz[None, None, :] ** 2)
    # rfft half space: planes 0 < kz < Nyquist count twice
    w = np.full(kz.shape, 2.0)
    w[0] = 1.0
    if n % 2 == 0:
        w[-1] = 1.0
    W = np.broadcast_to(w[None, None, :], kk.shape)
    dq = 2 * np.pi / (n * a)
    idx = np.rint(kk / dq).astype(int).ravel()
    nb = idx.max() + 1 if nbins is None else nbins
    keep = idx < nb
    cnt = np.bincount(idx[keep], weights=W.ravel()[keep], minlength=nb)
    s = np.bincount(idx[keep], weights=(P * W).ravel()[keep], minlength=nb)
    q = np.arange(nb) * dq
    good = (cnt > 0) & (q > 0)
    return q[good], drho2 * s[good] / cnt[good], cnt[good]


def levelcut_intensity(q, r, c, drho2=1.0):
    """drho2 * 4 pi integral r^2 c(r) sin(qr)/(qr) dr with c(r) = S2(r) - phi^2 (piecewise linear)."""
    q = np.atleast_1d(np.asarray(q, float))
    out = np.empty_like(q)
    nz = q > 0
    from .transforms import moments_pl
    out[~nz] = 4 * np.pi * moments_pl(r, r * c, 1)
    out[nz] = 4 * np.pi * sine_transform_pl(r, r * c, q[nz]) / q[nz]
    return drho2 * out
