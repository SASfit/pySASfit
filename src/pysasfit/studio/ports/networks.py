"""Python ports of the SASfit plugin ``deformed_polymer_network`` (WE and HSH tube models, linear and ring chains).

C sources: ``src/plugins/deformed_polymer_network/sasfit_ff_{we,hsh}__deformed_{polym,ringpolym}__netw_.c``.

Parameters (all four): p = [I0, Rg, d0, nu, lambda, dummy|gamma, psi (deg), delta (deg)].
With Δ = ψ - δ, Z_0 = q Rg cos Δ, Z_1 = q Rg sin Δ, λ_0 = λ, λ_1 = λ^{-1/2}:

    ln K(s) = Σ_i [ -a_i τ1_i (s/τ1_i - 1 + e^{-s/τ1_i}) - Z_i² τ2_i (1 - e^{-s/τ2_i}) ]

* WE:  a_i = Z_i² λ_i², τ1_i = τ2_i = ξ_i = (d0 λ_i^ν / Rg)² / (2√6)
* HSH: a_i = Z_i² λ_i² / γ, τ1_i = γ ξ_ψ, τ2_i = ξ_ψ, ξ_ψ = (d_ψ/Rg)²/(2√6), d_ψ = d0 (λ_0² cos²Δ + λ_1² sin²Δ)^ν

(the same expression as in C, written without the cancellation of -a s + a τ (1-e^{-s/τ})).
Linear chain: I = I0 ∫_0^1 2(1-x) K(x) dx; ring: s = x(1-x), and by symmetry I = I0 · 2 ∫_0^{1/2} K(x(1-x)) dx.

ln K decreases monotonically with s; the integration is split where ln K < -45 (end-point layer at large q Rg).

Differences to the C code: the WE functions check ``psi < 0`` on the scratch value p[MAXPAR-2] before it is set
(ψ of the previous call) and reject δ < 0; the port accepts any angle.
"""

from __future__ import annotations

import math

import numpy as np

from .quadrature import integrate_segments

__all__ = [
    "ff_hsh__deformed_polym__netw_",
    "ff_hsh__deformed_ringpolym__netw_",
    "ff_we__deformed_polym__netw_",
    "ff_we__deformed_ringpolym__netw_",
]

_SQRT6 = math.sqrt(6.0)
_SGRID = np.logspace(-16, 0, 129)


def _check(p, model):
    if p[1] < 0 or p[2] < 0 or p[4] < 0:
        raise ValueError("Rg, d0, lambda must be >= 0")
    if model == "we" and p[3] <= 0:
        raise ValueError("nu must be > 0")
    if model == "hsh" and (p[3] < 0 or p[5] < 1):
        raise ValueError("nu must be >= 0 and gamma >= 1")


def _terms(q, p, psi, model):
    """Coefficients (a_i, τ1_i, d_i, τ2_i), each shaped like the broadcast of q and ψ, for i = 0, 1."""
    q = np.asarray(q, float)
    if psi is None:
        psi = math.radians(p[6])
    q, psi = np.broadcast_arrays(q, np.asarray(psi, float))
    d = psi - math.radians(p[7])
    rg, d0, nu, lam = p[1], p[2], p[3], p[4]
    lmu = (lam, 1.0 / math.sqrt(lam)) if lam > 0 else (0.0, math.inf)
    z2 = ((q * rg * np.cos(d)) ** 2, (q * rg * np.sin(d)) ** 2)
    out = []
    if model == "we":
        for i in range(2):
            xi = (d0 * lmu[i] ** nu / rg) ** 2 / (2.0 * _SQRT6) if rg > 0 else 0.0
            xi = np.full(q.shape, xi)
            out.append((z2[i] * lmu[i] ** 2, xi, z2[i], xi))
    else:
        gam = p[5]
        dpsi = d0 * ((lmu[0] * np.cos(d)) ** 2 + (lmu[1] * np.sin(d)) ** 2) ** nu
        xi = (dpsi / rg) ** 2 / (2.0 * _SQRT6) if rg > 0 else np.zeros(q.shape)
        for i in range(2):
            out.append((z2[i] * lmu[i] ** 2 / gam, gam * xi, z2[i], xi))
    return out


def _lnk(s, terms):
    """ln K(s); s has the nodes on the last axis, the coefficients get a trailing axis."""
    total = 0.0
    with np.errstate(divide="ignore", invalid="ignore"):
        for a, t1, dd, t2 in terms:
            a, t1, dd, t2 = (np.asarray(v)[..., None] for v in (a, t1, dd, t2))
            u1 = s / t1
            first = np.where(
                t1 > 0, t1 * (u1 + np.expm1(-u1)), s
            )  # τ(u - 1 + e^{-u}), τ -> 0: s
            second = np.where(
                t2 > 0, -t2 * np.expm1(-s / t2), 0.0
            )  # τ(1 - e^{-s/τ}), τ -> 0: 0
            total = total - a * first - dd * second
    return total


def _cut(terms, smax):
    """Smallest s on a logarithmic grid with ln K(s) < -45 (else smax), per point."""
    lk = _lnk(_SGRID[_SGRID <= smax], terms)
    below = lk < -45.0
    grid = _SGRID[_SGRID <= smax]
    idx = np.argmax(below, axis=-1)
    return np.where(np.any(below, axis=-1), grid[idx], smax)


def _linear(q, p, psi, model):
    terms = _terms(q, p, psi, model)
    X = _cut(terms, 1.0)
    val = integrate_segments(
        lambda x, xc: 2.0 * xc * np.exp(_lnk(x, terms)), [0.0, X, 1.0]
    )
    return p[0] * val


def _ring(q, p, psi, model):
    terms = _terms(q, p, psi, model)
    sc = _cut(terms, 0.25)
    X = 0.5 * (
        1.0 - np.sqrt(np.maximum(1.0 - 4.0 * sc, 0.0))
    )  # x(1-x) = s_c on [0, 1/2]
    val = integrate_segments(
        lambda x, xc: 2.0 * np.exp(_lnk(x * xc, terms)), [0.0, X, 0.5]
    )
    return p[0] * val


def ff_we__deformed_polym__netw_(q, p, psi=None):
    """WE tube model, linear chain."""
    _check(p, "we")
    return _linear(q, p, psi, "we")


def ff_hsh__deformed_polym__netw_(q, p, psi=None):
    """HSH tube model, linear chain."""
    _check(p, "hsh")
    return _linear(q, p, psi, "hsh")


def ff_we__deformed_ringpolym__netw_(q, p, psi=None):
    """WE tube model, ring."""
    _check(p, "we")
    return _ring(q, p, psi, "we")


def ff_hsh__deformed_ringpolym__netw_(q, p, psi=None):
    """HSH tube model, ring."""
    _check(p, "hsh")
    return _ring(q, p, psi, "hsh")


def _qrg(q, p):
    return np.asarray(q, float) * p[1]


_PARS_WE = ["I0", "Rg", "d0", "nu", "lambda", "dummy", "psi", "delta"]
_PARS_HSH = ["I0", "Rg", "d0", "nu", "lambda", "gamma", "psi", "delta"]
_SAMPLE = {
    0: (0.1, 10.0, "log"),
    1: (1.0, 100.0, "log"),
    2: (1.0, 100.0, "log"),
    3: (0.3, 1.0, "lin"),
    4: (0.5, 3.0, "log"),
    6: (0.0, 180.0, "lin"),
    7: (0.0, 180.0, "lin"),
}
_SAMPLE_HSH = {**_SAMPLE, 5: (1.0, 3.0, "lin")}

SPECS = [
    {
        "name": name,
        "intensity": func,
        "parameters": pars,
        "psi_index": 6,
        "sample": sample,
        "rtol": 1e-6,
        "source": f"plugins/deformed_polymer_network/sasfit_{name}.c",
        "notes": "tanh-sinh, split at the end-point layer",
        "c_integrates": True,
        "reduced": _qrg,
    }
    for name, func, pars, sample in [
        (
            "ff_we__deformed_polym__netw_",
            ff_we__deformed_polym__netw_,
            _PARS_WE,
            _SAMPLE,
        ),
        (
            "ff_hsh__deformed_polym__netw_",
            ff_hsh__deformed_polym__netw_,
            _PARS_HSH,
            _SAMPLE_HSH,
        ),
        (
            "ff_we__deformed_ringpolym__netw_",
            ff_we__deformed_ringpolym__netw_,
            _PARS_WE,
            _SAMPLE,
        ),
        (
            "ff_hsh__deformed_ringpolym__netw_",
            ff_hsh__deformed_ringpolym__netw_,
            _PARS_HSH,
            _SAMPLE_HSH,
        ),
    ]
]
