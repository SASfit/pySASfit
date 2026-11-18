"""Python ports of the SASfit plugins ``deformed_polymers`` and (the parts used by them) ``polymer_rings``.

C sources: ``src/plugins/deformed_polymers/sasfit_ff_deformed_*.c`` and
``src/plugins/polymer_rings/sasfit_ff_ringpolymerB{MB,Z}.c``.

All functions are vectorised over q and, for the deformed (ψ-dependent) ones, over the direction ψ of Q on the
detector: ``intensity(q, p, psi=None)`` with ψ in radians broadcast against q. Without ``psi`` the parameter
p[5] (degrees) is used, as in SASfit without a ψ override.

Affine deformation: with Δ = ψ - θ0,

    β(q, ψ) = q² (Rg_ll² cos²Δ + Rg_⊥² sin²Δ),

and every deformed function is the isotropic one evaluated at (q Rg)² = β.

Differences to the C code (found while porting; fixed in the SASfit sources on 2026-11-17, still present in
SASfit builds made before):

* generalized Gaussian chain: the C code switches to the Guinier approximation exp(-β/3) for sqrt(β) < 0.1.
  That has a relative error of up to ~3e-6 (ν = 1/2: β²/36 at β = 0.01) and a jump at the switch. The port uses
  the exact series (u < 1) or the lower incomplete gamma form (u >= 1) everywhere.
* rings (BMB, BZ): for q Rg >~ 30 ... 400 (depending on the integration strategy) adaptive integration over
  [0, 1] misses the end-point layer of the kernel and returns 0 or wrong values. The C code now integrates
  2 ∫_0^X with the cut-off X used here as well.
* deformed rings: the C code checks ``psi < 0`` on p[MAXPAR-1] before that scratch value is set (so the check
  uses the ψ of the previous call) and rejects θ0 < 0 and ψ < 0; ``subParam`` is not initialised, so errors of
  the inner ring function are lost. The port accepts any angle.
"""

from __future__ import annotations

import math

import numpy as np
from scipy import special

from .quadrature import tanh_sinh_01

__all__ = [
    "deformed_beta",
    "ff_deformed_generalized_gaussian_chain",
    "ff_deformed_ring_polymer__bmb_",
    "ff_deformed_ring_polymer__bz_",
    "ff_ringpolymerBMB",
    "ff_ringpolymerBZ",
    "ggc_iso",
    "ring_bmb_iso",
    "ring_bz_iso",
]

_CHUNK = 4096  # q points per block in the quadrature (memory: CHUNK x ~230 nodes)


# ----------------------------------------------------------------------------------------------- isotropic
def ggc_iso(beta, nu):
    """Generalized Gaussian coil P(β) with β = (q Rg)², normalised to P(0) = 1.

    P = [u^{1/2ν} γ(1/2ν, u) - γ(1/ν, u)] / (ν u^{1/ν}),  u = (2ν+1)(2ν+2) β / 6,
    with the lower incomplete gamma function γ; for u < 1 the convergent series
    P = (1/ν) Σ_k (-u)^k / k! [1/(1/(2ν)+k) - 1/(1/ν+k)] is used (no cancellation).
    """
    beta = np.asarray(beta, float)
    nu = float(nu)
    u = (2.0 * nu + 1.0) * (2.0 * nu + 2.0) * beta / 6.0
    out = np.empty_like(u)
    small = u < 1.0
    if np.any(small):
        us = u[small]
        a, b = 0.5 / nu, 1.0 / nu
        term = np.ones_like(us)  # (-u)^k / k!
        s = np.zeros_like(us)
        for k in range(60):
            s += term * (1.0 / (a + k) - 1.0 / (b + k))
            term = term * (-us) / (k + 1)
            if np.all(np.abs(term) < 1e-18 * np.abs(s)):
                break
        out[small] = s / nu
    big = ~small
    if np.any(big):
        ub = u[big]
        a, b = 0.5 / nu, 1.0 / nu
        ga = special.gammainc(a, ub) * special.gamma(a)
        gb = special.gammainc(b, ub) * special.gamma(b)
        # u^{1/2ν} γ(a,u) / u^{1/ν} = γ(a,u) / u^{1/2ν}
        out[big] = (ga * ub ** (-a) - gb * ub ** (-b)) / nu
    return out


def _ring_quadrature(beta, nu, model):
    """P = ∫_0^1 2(1-x) exp(-μ s(x)) dx with s symmetric about x = 1/2, hence P = 2 ∫_0^{1/2} exp(-μ s(x)) dx.

    For large μ the integrand lives in a layer of width ~ μ^{-1/a} at x = 0: the upper limit is cut to
    X = min(1/2, x where μ s(x) > 45), and the tanh-sinh nodes are scaled to [0, X] for every q separately.
    """
    beta = np.asarray(beta, float)
    shape = beta.shape
    beta = beta.ravel()
    a = 2.0 * float(nu)  # 1 + ε with ε = 2ν - 1
    t, tc, w = tanh_sinh_01()
    if model == "bmb":
        eps = a - 1.0
        keps = math.exp(
            special.gammaln(2 + eps)
            + special.gammaln(3 + eps)
            - special.gammaln(5 + 2 * eps)
        )
        cut = 2.0 * 45.0  # s(x) >= (x/2)^a on [0, 1/2]
    else:  # bz: K_ε = ∫_0^1 (x(1-x))^a (1-x) / ((1-x)^a + x^a) dx
        lt, ltc = np.log(t), np.log(tc)
        keps = float(
            np.sum(
                np.exp(a * (lt + ltc) + ltc) / (np.exp(a * ltc) + np.exp(a * lt)) * w
            )
        )
        cut = 4.0 * 45.0  # s(x) >= (x/2)^a / 2 on [0, 1/2]
    mu = beta / (6.0 * keps)
    out = np.empty_like(mu)
    for i in range(0, mu.size, _CHUNK):
        m = mu[i : i + _CHUNK]
        X = np.full(m.shape, 0.5)
        pos = m > 0
        with np.errstate(over="ignore"):
            X[pos] = np.minimum(0.5, 2.0 * (0.5 * cut / m[pos]) ** (1.0 / a))
        x = np.multiply.outer(X, t)  # (n, nodes) in [0, X]
        xc = 1.0 - x
        lx, lxc = np.log(x), np.log(xc)
        s = np.exp(a * (lx + lxc))
        if model == "bz":
            s = s / (np.exp(a * lxc) + np.exp(a * lx))
        out[i : i + _CHUNK] = 2.0 * X * (np.exp(-m[:, None] * s) @ w)
    return out.reshape(shape)


def ring_bmb_iso(beta, nu):
    """Ring polymer (Bensafi-Maschke-Benoit), P(0) = 1:  P = ∫_0^1 2(1-x) exp(-μ (x(1-x))^{1+ε}) dx,
    ε = 2ν-1, μ = β/(6 K_ε), K_ε = Γ(2+ε)Γ(3+ε)/Γ(5+2ε)."""
    return _ring_quadrature(beta, nu, "bmb")


def ring_bz_iso(beta, nu):
    """Ring polymer (BZ), P(0) = 1:  P = ∫_0^1 2(1-x) exp(-μ (x(1-x))^{1+ε}/((1-x)^{1+ε}+x^{1+ε})) dx,
    μ = β/(6 K_ε), K_ε = ∫_0^1 (x(1-x))^{1+ε} (1-x)/((1-x)^{1+ε}+x^{1+ε}) dx."""
    return _ring_quadrature(beta, nu, "bz")


# ----------------------------------------------------------------------------------------------- deformed
def deformed_beta(q, p, psi=None):
    """β = q² (Rg_ll² cos²Δ + Rg_⊥² sin²Δ), Δ = ψ - θ0; θ0 = p[4] and (without ``psi``) ψ = p[5] in degrees."""
    q = np.asarray(q, float)
    if psi is None:
        psi = math.radians(p[5])
    q, psi = np.broadcast_arrays(q, np.asarray(psi, float))
    d = psi - math.radians(p[4])
    return q * q * ((p[0] * np.cos(d)) ** 2 + (p[1] * np.sin(d)) ** 2)


def _check(cond, msg):
    if cond:
        raise ValueError(msg)


def ff_deformed_generalized_gaussian_chain(q, p, psi=None):
    """p = [Rg_ll, Rg_perp, nu, I0, theta_0 (deg), psi (deg)]."""
    _check(p[0] < 0 or p[1] < 0, "Rg_ll, Rg_perp must be >= 0")
    _check(p[2] <= 0, "nu must be > 0")
    return p[3] * ggc_iso(deformed_beta(q, p, psi), p[2])


def ff_deformed_ring_polymer__bmb_(q, p, psi=None):
    """p = [Rg_ll, Rg_perp, nu, I0, theta_0 (deg), psi (deg)]."""
    _check(p[0] < 0 or p[1] < 0, "Rg_ll, Rg_perp must be >= 0")
    _check(p[2] <= 0, "nu must be > 0")
    return p[3] * ring_bmb_iso(deformed_beta(q, p, psi), p[2])


def ff_deformed_ring_polymer__bz_(q, p, psi=None):
    """p = [Rg_ll, Rg_perp, nu, I0, theta_0 (deg), psi (deg)]."""
    _check(p[0] < 0 or p[1] < 0, "Rg_ll, Rg_perp must be >= 0")
    _check(p[2] <= 0, "nu must be > 0")
    return p[3] * ring_bz_iso(deformed_beta(q, p, psi), p[2])


def ff_ringpolymerBMB(q, p, psi=None):
    """p = [Rg, dummy, nu, I0] (isotropic)."""
    _check(p[0] < 0, "Rg must be >= 0")
    _check(p[2] <= 0, "nu must be > 0")
    q = np.asarray(q, float)
    return p[3] * ring_bmb_iso((q * p[0]) ** 2, p[2])


def ff_ringpolymerBZ(q, p, psi=None):
    """p = [Rg, dummy, nu, I0] (isotropic)."""
    _check(p[0] < 0, "Rg must be >= 0")
    _check(p[2] <= 0, "nu must be > 0")
    q = np.asarray(q, float)
    return p[3] * ring_bz_iso((q * p[0]) ** 2, p[2])


def _qrg_deformed(q, p):
    return np.sqrt(deformed_beta(q, p))


def _qrg_ring(q, p):
    return np.asarray(q, float) * p[0]


# ----------------------------------------------------------------------------------------------- specs
_DEF_PARS = ["Rg_ll", "Rg_perp", "nu", "I0", "theta_0", "psi"]
_DEF_SAMPLE = {
    0: (1.0, 100.0, "log"),
    1: (1.0, 100.0, "log"),
    2: (0.3, 0.7, "lin"),
    3: (0.1, 10.0, "log"),
    4: (0.0, 180.0, "lin"),
    5: (0.0, 180.0, "lin"),
}
_RING_PARS = ["Rg", "dummy", "nu", "I0"]
_RING_SAMPLE = {0: (1.0, 100.0, "log"), 2: (0.3, 0.7, "lin"), 3: (0.1, 10.0, "log")}

SPECS = [
    {
        "name": "ff_deformed_generalized_gaussian_chain",
        "intensity": ff_deformed_generalized_gaussian_chain,
        "parameters": _DEF_PARS,
        "psi_index": 5,
        "reduced": _qrg_deformed,
        "sample": _DEF_SAMPLE,
        "rtol": 1e-5,
        "source": "plugins/deformed_polymers/sasfit_ff_deformed_generalized_gaussian_chain.c",
        "notes": "exact; SASfit builds before 2026-11-17 use Guinier for sqrt(β)<0.1 (up to ~4e-6)",
    },
    {
        "name": "ff_deformed_ring_polymer__bmb_",
        "intensity": ff_deformed_ring_polymer__bmb_,
        "parameters": _DEF_PARS,
        "psi_index": 5,
        "reduced": _qrg_deformed,
        "sample": _DEF_SAMPLE,
        "rtol": 1e-6,
        "source": "plugins/deformed_polymers/sasfit_ff_deformed_ring_polymer__bmb_.c",
        "notes": "tanh-sinh quadrature (~230 nodes) instead of adaptive sasfit_integrate",
        "c_integrates": True,
    },
    {
        "name": "ff_deformed_ring_polymer__bz_",
        "intensity": ff_deformed_ring_polymer__bz_,
        "parameters": _DEF_PARS,
        "psi_index": 5,
        "reduced": _qrg_deformed,
        "sample": _DEF_SAMPLE,
        "rtol": 1e-6,
        "source": "plugins/deformed_polymers/sasfit_ff_deformed_ring_polymer__bz_.c",
        "notes": "tanh-sinh quadrature (~230 nodes) instead of adaptive sasfit_integrate",
        "c_integrates": True,
    },
    {
        "name": "ff_ringpolymerBMB",
        "intensity": ff_ringpolymerBMB,
        "parameters": _RING_PARS,
        "psi_index": None,
        "reduced": _qrg_ring,
        "sample": _RING_SAMPLE,
        "rtol": 1e-6,
        "source": "plugins/polymer_rings/sasfit_ff_ringpolymerBMB.c",
        "notes": "tanh-sinh quadrature",
        "c_integrates": True,
    },
    {
        "name": "ff_ringpolymerBZ",
        "intensity": ff_ringpolymerBZ,
        "parameters": _RING_PARS,
        "psi_index": None,
        "reduced": _qrg_ring,
        "sample": _RING_SAMPLE,
        "rtol": 1e-6,
        "source": "plugins/polymer_rings/sasfit_ff_ringpolymerBZ.c",
        "notes": "tanh-sinh quadrature",
        "c_integrates": True,
    },
]
