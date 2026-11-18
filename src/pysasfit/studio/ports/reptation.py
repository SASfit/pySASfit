"""Python port of the SASfit plugin ``reptating_chain``.

C source: ``src/plugins/reptating_chain/sasfit_ff_reptating_chain.c``.

p = [I0, Rg, lambda, t/tau, theta_0 (deg), psi (deg)]. With α = (q Rg)², Δ = ψ - θ0,
L3 = sqrt(λ³-1), E = (λ + asinh(L3)/(√λ L3))/2 and β = (λ² q² cos²Δ + q² sin²Δ/λ) Rg²/E:

    I/I0 = D(α) + (α-β)/6 (1+e^{-α}) H(α,1) + (α-β)²/6 ∫_0^1 y³ (e^{β(y-1)} + e^{β(y-1)-αy}) H(α,y) dy
    H(x,y) = Σ_{n odd} 96/(πn)² / ((nπ)² + (xy)²) · exp(-(n/y)² t/τ)
    D(x) = 2(x - 1 + e^{-x})/x²   (λ = 1: I/I0 = D(α))

The y integral has a layer of width 1/β at y = 1; it is split there. D is evaluated without cancellation.

Differences to the C code:

* the C function never multiplies with I0 (p[0] is ignored);
* D(x) = 2(x-1+e^{-x})/x² loses all digits for small x (relative error ~1e-16/x², i.e. 1e-4 at q Rg = 1e-3);
* ``psi < 0`` is checked on the scratch value before it is set, θ0 < 0 and ψ < 0 are rejected;
* H stops when a term changes the sum by less than 1e-10 (the port sums n = 1 ... 99).

``c_compat=True`` reproduces the first two points (used for the comparison with the C library).
"""

from __future__ import annotations

import math

import numpy as np

from .quadrature import integrate_segments

__all__ = ["debye", "ff_reptating_chain"]

_N = np.arange(1, 100, 2, dtype=float)


def debye(x):
    """D(x) = 2(x - 1 + e^{-x})/x², exact also for small x."""
    x = np.asarray(x, float)
    out = np.empty_like(x)
    small = np.abs(x) < 0.2
    xs = x[small]
    acc = np.zeros_like(xs)
    fact = 2.0  # (k+2)!
    for k in range(14):  # D = Σ_k 2 (-x)^k / (k+2)!
        if k:
            fact *= k + 2
        acc = acc + 2.0 * (-xs) ** k / fact
    out[small] = acc
    xb = x[~small]
    out[~small] = 2.0 * (xb + np.expm1(-xb)) / xb**2
    return out


def _debye_c(x):
    x = np.asarray(x, float)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(x == 0, 1.0, 2 * (x - 1 + np.exp(-x)) / (x * x))


def _h(x, y, ttau):
    """H(x, y); x and y broadcast, n on a new last axis."""
    y0 = np.asarray(y, float)
    x = np.asarray(x, float)[..., None]
    y = y0[..., None]
    n = _N
    term = 96.0 / (math.pi * n) ** 2 / ((n * math.pi) ** 2 + (x * y) ** 2)
    if ttau != 0.0:
        with np.errstate(divide="ignore", over="ignore"):
            term = term * np.exp(-((n / y) ** 2) * ttau)
    return np.where(y0 == 0, 0.0, np.sum(term, axis=-1))


def ff_reptating_chain(q, p, psi=None, c_compat=False):
    """Reptating chain (stretched, relaxing): p = [I0, Rg, lambda, t/tau, theta_0 (deg), psi (deg)]."""
    i0, rg, lam, ttau = p[0], p[1], p[2], p[3]
    if rg < 0 or lam < 1 or ttau < 0:
        raise ValueError("Rg >= 0, lambda >= 1 and t/tau >= 0 required")
    q = np.asarray(q, float)
    if psi is None:
        psi = math.radians(p[5])
    q, psi = np.broadcast_arrays(q, np.asarray(psi, float))
    d = psi - math.radians(p[4])
    alpha = (q * rg) ** 2
    dfun = _debye_c if c_compat else debye
    scale = 1.0 if c_compat else i0
    if lam == 1.0:
        return scale * dfun(alpha)
    l3 = math.sqrt(lam**3 - 1.0)
    e = 0.5 * (lam + math.asinh(l3) / math.sqrt(lam) / l3)
    beta = ((lam * q * np.cos(d)) ** 2 + (q * np.sin(d)) ** 2 / lam) * rg * rg / e
    # ∫_0^1 y³ (e^{β(y-1)} + e^{β(y-1)-αy}) H(α,y) dy with u = 1-y: layer of width 1/β at u = 0
    a3 = alpha[..., None]
    b3 = beta[..., None]

    def kern(u, y):  # u = 1-y (accurate), y
        return y**3 * (np.exp(-b3 * u) + np.exp(-b3 * u - a3 * y)) * _h(a3, y, ttau)

    with np.errstate(divide="ignore"):
        U = np.where(beta > 45.0, 45.0 / beta, 1.0)
    integral = integrate_segments(kern, [0.0, U, 1.0])
    h1 = _h(alpha, np.ones_like(alpha), ttau)
    val = (
        dfun(alpha)
        + (alpha - beta) / 6.0 * (1.0 + np.exp(-alpha)) * h1
        + (alpha - beta) ** 2 / 6.0 * integral
    )
    return scale * val


def _qrg(q, p):
    return np.asarray(q, float) * p[1]


SPECS = [
    {
        "name": "ff_reptating_chain",
        "intensity": ff_reptating_chain,
        "parameters": ["I0", "Rg", "lambda", "t/tau", "theta_0", "psi"],
        "psi_index": 5,
        "sample": {
            0: (0.1, 10.0, "log"),
            1: (1.0, 100.0, "log"),
            2: (1.0, 3.0, "lin"),
            3: (0.0, 2.0, "lin"),
            4: (0.0, 180.0, "lin"),
            5: (0.0, 180.0, "lin"),
        },
        "rtol": 1e-6,
        "source": "plugins/reptating_chain/sasfit_ff_reptating_chain.c",
        "notes": "C ignores I0 and loses digits in D(x) for small q Rg (compared with c_compat)",
        "c_integrates": True,
        "c_compat": False,
        "reduced": _qrg,
    }
]
