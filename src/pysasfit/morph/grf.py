"""Gaussian random fields with a prescribed correlation function g(r), and the level cut.

Two generators:

``"fft"`` (default)
    Spectral synthesis on the periodic N^3 grid: the correlation g(|r|) is
    sampled on the grid (minimum-image distances), its discrete Fourier transform
    gives the power spectrum S(k) on exactly the grid wave vectors, and the field
    is IFFT( sqrt(S) * FFT(white noise) ). Cost O(N^3 log N); the field is
    periodic, so the box should be several correlation lengths wide.

``"cosine"``
    Berk's construction used by SAXSMorph: psi(r) = sqrt(2/N_w) sum cos(k_i.r + phi_i)
    with |k_i| drawn from P(k) = k^2 f(k)/(2 pi^2) (here by inverse CDF instead of
    rejection), isotropic directions and random phases. Not periodic, but cost
    O(N_w N^3) -- use a few thousand waves.
"""

from __future__ import annotations

import numpy as np

from .correlation import spectral_function, wavevector_distribution

__all__ = ["grid_distances", "field_fft", "field_cosine", "level_cut", "cut_level", "power_spectrum_on_grid"]


def _signed_coords(n, a):
    i = np.arange(n)
    return a * np.where(i <= n // 2, i, i - n)


def grid_distances(n, a):
    """Minimum-image distance of every voxel of a periodic n^3 grid (spacing a) to the origin."""
    x = _signed_coords(n, a)
    return np.sqrt(x[:, None, None] ** 2 + x[None, :, None] ** 2 + x[None, None, :] ** 2)


def power_spectrum_on_grid(r, g, n, a):
    """Power spectrum of the periodised g on the grid (rfftn layout) and the share of
    negative power that had to be set to zero (a diagnostic of the r range / box)."""
    gg = np.interp(grid_distances(n, a), r, g, right=0.0)
    S = np.fft.rfftn(gg).real
    neg = float(-S[S < 0].sum() / np.abs(S).sum()) if np.any(S < 0) else 0.0
    return np.clip(S, 0.0, None), neg


def field_fft(r, g, n, a, rng=None, dtype=np.float32):
    """Unit-variance periodic Gaussian field on n^3 voxels (spacing a) with correlation g(r).

    Returns (field, info) with info['negative_power'] the share of negative spectral power
    that was discarded.
    """
    rng = np.random.default_rng(rng)
    S, neg = power_spectrum_on_grid(r, g, n, a)
    w = np.fft.rfftn(rng.standard_normal((n, n, n)))
    psi = np.fft.irfftn(w * np.sqrt(S), s=(n, n, n), axes=(0, 1, 2))
    psi -= psi.mean()
    psi /= psi.std()
    return psi.astype(dtype), {"negative_power": neg}


def field_cosine(r, g, n, a, n_waves=2000, rng=None, k=None, dtype=np.float32, chunk=64, progress=None):
    """Berk's random-wave field on an n^3 grid (not periodic). Returns (field, info)."""
    rng = np.random.default_rng(rng)
    if k is None:
        k = np.linspace(0, np.pi / a * 2, 4000)[1:]
    P = wavevector_distribution(k, spectral_function(r, g, k))
    cdf = np.concatenate([[0.0], np.cumsum(0.5 * (P[1:] + P[:-1]) * np.diff(k))])
    if cdf[-1] <= 0:
        raise ValueError("spectral function has no positive weight")
    kmag = np.interp(rng.random(n_waves) * cdf[-1], cdf, np.concatenate([[k[0]], k[1:]]))
    cz = rng.uniform(-1, 1, n_waves)
    ph = rng.uniform(0, 2 * np.pi, n_waves)
    st = np.sqrt(1 - cz ** 2)
    kv = kmag[:, None] * np.column_stack([st * np.cos(ph), st * np.sin(ph), cz])
    phase = rng.uniform(0, 2 * np.pi, n_waves)
    x = a * np.arange(n)
    psi = np.zeros((n, n, n), np.complex128)
    for i0 in range(0, n_waves, chunk):
        sl = slice(i0, i0 + chunk)
        ex = np.exp(1j * (kv[sl, 0:1] * x + phase[sl, None]))    # (c, n)
        ey = np.exp(1j * kv[sl, 1:2] * x)
        ez = np.exp(1j * kv[sl, 2:3] * x)
        psi += np.einsum("ci,cj,ck->ijk", ex, ey, ez, optimize=True)
        if progress is not None:
            progress(min(1.0, (i0 + chunk) / n_waves))
    out = np.sqrt(2.0 / n_waves) * psi.real
    out -= out.mean()
    out /= out.std()
    return out.astype(dtype), {"n_waves": n_waves, "fraction_beyond_nyquist":
                               float(np.mean(kmag > np.pi / a))}


def cut_level(psi, porosity, mode="quantile", alpha=None):
    """Level of the cut: the empirical (1 - porosity) quantile of psi, or alpha."""
    if mode == "alpha":
        if alpha is None:
            from scipy.special import erfcinv
            alpha = np.sqrt(2.0) * erfcinv(2.0 * porosity)
        return float(alpha)
    return float(np.quantile(psi, 1.0 - porosity))


def level_cut(psi, porosity, mode="quantile", alpha=None):
    """Binary matrix, True = phase 2 ('pore', volume fraction ``porosity``).

    mode 'quantile' cuts at the empirical quantile, so the voxel fraction equals the
    porosity exactly; 'alpha' cuts at the theoretical level alpha (fraction fluctuates).
    """
    return psi > cut_level(psi, porosity, mode, alpha)
