"""Python ports of the SASfit plugins ``shearflow`` and ``shearflow_polymers_lin`` (polymers and rings under shear).

C sources: ``src/plugins/shearflow/sasfit_ff_{polymer,ring}_under_shearflow{,1,2}.c`` and
``src/plugins/shearflow_polymers_lin/sasfit_ff_shearflow_lin{,_g1,_g2}.c``.

All functions are vectorised over q and ψ (the direction of Q on the detector, radians, broadcast against q).
Orientation averages over the aperture sector or over Euler angles use Gauss-Legendre nodes; integrals along the
chain use tanh-sinh, split at the end-point layer of width ~1/(q Rg)².

The "two-stage" form factor of the ``*shearflow2`` and ``shearflow_lin`` models,

    P(b, c, f) = 2 ∫_0^1 (1-x) e^{-g(x)} dx,  g = b x (x < f),  g = b f + c (x - f) (x > f),

is evaluated in closed form without cancellation (the C formula 2(e^{-bf}/c + (1-e^{-bf})(1/b - 1/b²) + ...) loses
digits for small b, c and is 0/0 at q = 0).

Differences to the C code (``c_compat=True`` reproduces them for the comparison with the C library):

* polymer_under_shearflow: the sector average divides by 2Δ instead of Δ (result halved) and divides by 0 for
  Δθ0 = 0; the averaged sector is θ0 ± Δθ0/4 instead of ± Δθ0/2 (also ring_under_shearflow).
* ring_under_shearflow2: for Δθ0 > 0 the sector integral is not divided by its width; Δθ0 = 0 divides by 0.
* polymer_under_shearflow1: ψ is read from p[7], but the parameter list (header) has ψ at p[2].
* shearflow_lin_g1: Δφ = 0 gives 0/0.
* the two-stage formula as above.

* polymer/ring_under_shearflow1: no I0 (p[1] was a dummy); ring_under_shearflow1 had the weight (1-x), so P(0) = 1/2.

All fixed in the SASfit sources on 2026-11-17; c_compat=True reproduces the old behaviour.
"""

from __future__ import annotations

import math

import numpy as np
from scipy import special

from .quadrature import gauss_legendre, integrate_segments

__all__ = [
    "ff_polymer_under_shearflow",
    "ff_polymer_under_shearflow1",
    "ff_polymer_under_shearflow2",
    "ff_ring_under_shearflow",
    "ff_ring_under_shearflow1",
    "ff_ring_under_shearflow2",
    "ff_shearflow_lin",
    "ff_shearflow_lin_g1",
    "ff_shearflow_lin_g2",
    "two_stage",
]

_PI4 = math.pi**4
_NTHETA = 32


# ------------------------------------------------------------------------------------------------ helpers
def _e1(z):
    """(1 - e^{-z})/z."""
    z = np.asarray(z, float)
    out = np.empty_like(z)
    s = np.abs(z) < 0.1
    zs = z[s]
    # series Σ_k (-z)^k/(k+1)!
    acc = np.zeros_like(zs)
    fact = 1.0
    for k in range(14):
        fact *= k + 1
        acc = acc + (-zs) ** k / fact
    out[s] = acc
    zb = z[~s]
    out[~s] = -np.expm1(-zb) / zb
    return out


def _e2(z):
    """∫_0^1 t e^{-zt} dt = (1 - e^{-z}(1+z))/z²."""
    z = np.asarray(z, float)
    out = np.empty_like(z)
    s = np.abs(z) < 0.1
    zs = z[s]
    acc = np.zeros_like(zs)
    fact = 1.0
    for k in range(14):
        if k:
            fact *= k
        acc = acc + (-zs) ** k / (fact * (k + 2))
    out[s] = acc
    zb = z[~s]
    out[~s] = (-np.expm1(-zb) - zb * np.exp(-zb)) / zb**2
    return out


def two_stage(b, c, f):
    """P(b, c, f) = 2 ∫_0^1 (1-x) e^{-g(x)} dx with slope b up to x = f and c after it (exact, P(0, 0, f) = 1)."""
    b, c = np.broadcast_arrays(np.asarray(b, float), np.asarray(c, float))
    L = 1.0 - f
    i1 = 2.0 * f * (_e1(b * f) - f * _e2(b * f))
    i2 = 2.0 * np.exp(-b * f) * L * L * (_e1(c * L) - _e2(c * L))
    return i1 + i2


def _two_stage_c(b, c, f):
    """The C formula (sasfit_ff_polymer_under_shearflow2_core, sasfit_ff_shearflow_lin_core)."""
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        e = np.exp(-b * f)
        return 2 * (
            e / c
            + (1 - e) * (1 / b - 1 / (b * b))
            + e / (c * c) * (np.exp(-c * (1 - f)) - 1)
            + e * (1 / b - 1 / c) * f
        )


def _psi(q, p, psi, index):
    q = np.asarray(q, float)
    if psi is None:
        psi = (
            math.radians(p[index]) if index < len(p) else 0.0
        )  # SASfit pads p with zeros
    return np.broadcast_arrays(q, np.asarray(psi, float))


def _sector(hw, n=_NTHETA):
    """Gauss-Legendre nodes and weights for the mean over θ in [-hw, hw] (one node θ = 0 for hw = 0)."""
    if hw == 0:
        return np.zeros(1), np.ones(1)
    t, w = gauss_legendre(n, -hw, hw)
    return t, w / (2.0 * hw)


# ------------------------------------------------------------------------------------------------ Wi models
def _wi_polymer(qrg2, nu, wi, cpsi):
    """∫_0^1 2(1-x) exp(-F) dx, F = A (μ + π⁴/180 (μ Wi cos ψ)² (1 + μ - μ²/4 - 2μ³ + μ⁴)), μ = x^{2ν}."""
    a = qrg2 * (2 * nu + 1) * (2 * nu + 2) / 6.0
    with np.errstate(divide="ignore", over="ignore"):
        X = np.where(a > 0, np.minimum(1.0, (45.0 / a) ** (0.5 / nu)), 1.0)
    a3, c3 = a[..., None], cpsi[..., None]

    def f(x, xc):
        mu = x ** (2 * nu)
        poly = 1 + mu - mu**2 / 4 - 2 * mu**3 + mu**4
        return 2.0 * xc * np.exp(-a3 * (mu + _PI4 / 180.0 * (mu * wi * c3) ** 2 * poly))

    return integrate_segments(f, [0.0, X, 1.0])


def _wi_ring(qrg2, nu, wi, cpsi):
    """2 ∫_0^{1/2} exp(-g) dx, g = A (μ + π⁴/1440 (Wi cos ψ)² (μ² + 2μ³)), μ = (x(1-x))^{2ν}, A = (q Rg)²/(6 K_ε)."""
    eps = 2 * nu - 1
    keps = math.exp(
        special.gammaln(2 + eps)
        + special.gammaln(3 + eps)
        - special.gammaln(5 + 2 * eps)
    )
    a = qrg2 / (6.0 * keps)
    with np.errstate(divide="ignore", over="ignore"):
        X = np.where(a > 0, np.minimum(0.5, 2.0 * (45.0 / a) ** (0.5 / nu)), 0.5)
    a3, c3 = a[..., None], cpsi[..., None]

    def f(x, xc):
        mu = (x * xc) ** (2 * nu)
        return 2.0 * np.exp(
            -a3 * (mu + _PI4 / 1440.0 * (wi * c3) ** 2 * (mu**2 + 2 * mu**3))
        )

    return integrate_segments(f, [0.0, X, 0.5])


def _wi_model(q, p, psi, c_compat, ring):
    rg, wi, nu, i0, th0, dth = p[0], p[1], p[2], p[3], p[4], p[6]
    if rg < 0 or wi < 0 or nu <= 0:
        raise ValueError("Rg >= 0, Wi >= 0, nu > 0 required")
    q, psi = _psi(q, p, psi, 5)
    hw = math.radians(dth) / (4.0 if c_compat else 2.0)  # C: θ0 ± Δθ0/4
    t, w = _sector(hw)
    peff = psi[..., None] - math.radians(th0) - t  # (..., nθ)
    qrg2 = np.broadcast_to((q * rg)[..., None] ** 2, peff.shape)
    vals = (_wi_ring if ring else _wi_polymer)(qrg2, nu, wi, np.cos(peff))
    mean = np.sum(vals * w, axis=-1)
    if c_compat and not ring:
        mean = np.full_like(mean, np.inf) if dth == 0 else 0.5 * mean
    return i0 * mean


def ff_polymer_under_shearflow(q, p, psi=None, c_compat=False):
    """p = [Rg, Wi, nu, I0, theta_0 (deg), psi (deg), Delta_theta_0 (deg)]."""
    return _wi_model(q, p, psi, c_compat, ring=False)


def ff_ring_under_shearflow(q, p, psi=None, c_compat=False):
    """p = [Rg, Wi, nu, I0, theta_0 (deg), psi (deg), Delta_theta_0 (deg)]."""
    return _wi_model(q, p, psi, c_compat, ring=True)


# ------------------------------------------------------------------------------------------------ *1 models
def ff_polymer_under_shearflow1(q, p, psi=None, c_compat=False):
    """p = [Rg, I0, psi (deg), xi, Bx, Bz]:  I = I0 ∫_0^1 2(1-x) exp(-(q Rg)² (x + x^ξ k)) dx,
    k = sgn(Bx) Bx² cos²ψ + sgn(Bz) Bz² sin²ψ. c_compat: no I0, ψ from p[7] (old C)."""
    rg, xi, bx, bz = p[0], p[3], p[4], p[5]
    i0 = 1.0 if c_compat else p[1]
    if rg < 0 or bx < -1 or bz < -1:
        raise ValueError("Rg >= 0, Bx >= -1, Bz >= -1 required")
    q, psi = _psi(q, p, psi, 7 if c_compat else 2)
    k = np.sign(bx) * (bx * np.cos(psi)) ** 2 + np.sign(bz) * (bz * np.sin(psi)) ** 2
    qrg2 = (q * rg) ** 2
    with np.errstate(divide="ignore"):
        X = np.where(qrg2 > 50, 50.0 / qrg2, 1.0)
    a3, k3 = qrg2[..., None], k[..., None]
    return i0 * integrate_segments(
        lambda x, xc: 2.0 * xc * np.exp(-a3 * (x + x**xi * k3)), [0.0, X, 1.0]
    )


def ff_ring_under_shearflow1(q, p, psi=None, c_compat=False):
    """p = [Rg, I0, psi (deg), xi, Bx, Bz]:  I = I0 ∫_0^1 2(1-x) exp(-(q Rg)² r(x) r(1-x)) dx, P(0) = 1,
    r(μ) = μ + μ^ξ (Bx² cos²ψ - Bz² sin²ψ). c_compat: weight (1-x) and no I0 (old C)."""
    rg, xi, bx, bz = p[0], p[3], p[4], p[5]
    if rg < 0:
        raise ValueError("Rg >= 0 required")
    q, psi = _psi(q, p, psi, 2)
    k = (bx * np.cos(psi)) ** 2 - (bz * np.sin(psi)) ** 2
    qrg2 = (q * rg) ** 2
    slope = qrg2 * np.maximum(np.abs(1.0 + k), 1e-3)
    with np.errstate(divide="ignore"):
        X = np.where(slope > 100, 50.0 / slope, 0.5)
    a3, k3 = qrg2[..., None], k[..., None]

    def f(x, xc):  # symmetric exponent: ∫_0^1 2(1-x) g = 2 ∫_0^{1/2} g
        return np.exp(-a3 * (x + x**xi * k3) * (xc + xc**xi * k3))

    scale = 1.0 if c_compat else 2.0 * p[1]  # old C: ∫_0^1 (1-x) g = ∫_0^{1/2} g, no I0
    return scale * integrate_segments(f, [0.0, X, 0.5])


# ------------------------------------------------------------------------------------------------ *2 models
def _slopes(p, peff):
    al, be, ga, de = p[2], p[3], p[4], p[5]
    c, s = np.cos(peff), np.sin(peff)
    return (al * c) ** 2 + (be * s) ** 2, (ga * c) ** 2 + (de * s) ** 2


def ff_polymer_under_shearflow2(q, p, psi=None, c_compat=False):
    """p = [Rg, f, alpha, beta, gamma, delta, I0, theta_0 (deg), psi (deg), Delta_theta_0 (deg)]."""
    rg, fp, i0, th0, dth = p[0], p[1], p[6], p[7], p[9]
    if rg < 0 or not 0 <= fp <= 1:
        raise ValueError("Rg >= 0 and 0 <= f <= 1 required")
    q, psi = _psi(q, p, psi, 8)
    t, w = _sector(math.radians(dth) / 2.0)
    peff = t + math.radians(th0) - psi[..., None]
    bb, cc = _slopes(p, peff)
    qrg2 = (q * rg)[..., None] ** 2
    core = (_two_stage_c if c_compat else two_stage)(qrg2 * bb, qrg2 * cc, fp)
    return i0 * np.sum(core * w, axis=-1)


def ff_ring_under_shearflow2(q, p, psi=None, c_compat=False):
    """p = [Rg, f, alpha, beta, gamma, delta, I0, theta_0 (deg), psi (deg), Delta_theta_0 (deg)]:
    P = ∫_0^1 2(1-x) exp(-2 (q Rg)² t(x) (t(1) - t(x))) dx with the piecewise linear t(x) of the C code."""
    rg, fp, i0, th0, dth = p[0], p[1], p[6], p[7], p[9]
    if rg < 0:
        raise ValueError("Rg >= 0 required")
    q, psi = _psi(q, p, psi, 8)
    hw = math.radians(dth) / 2.0
    t, w = _sector(hw)
    peff = t + math.radians(th0) - psi[..., None]
    bb, cc = _slopes(p, peff)
    with np.errstate(divide="ignore", invalid="ignore"):
        cs = np.sqrt(bb / (cc + 2 * bb * fp - 2 * cc * fp))
    t1 = (cc - 2 * cc * fp + 2 * bb * fp) * cs
    qrg2 = np.broadcast_to((q * rg)[..., None] ** 2, peff.shape)
    slope = 2 * qrg2 * bb * cs * t1
    with np.errstate(divide="ignore", invalid="ignore"):
        X = np.where(slope > 200, 50.0 / slope, 0.25)
    b3, c3, s3, t13, a3 = (v[..., None] for v in (bb, cc, cs, t1, qrg2))

    def f(x, xc):
        tt = np.select(
            [x <= fp, x >= 1 - fp],
            [b3 * x * s3, t13 - b3 * xc * s3],
            (b3 * fp + c3 * (x - fp)) * s3,
        )
        rest = np.select(
            [x <= fp, x >= 1 - fp], [t13 - b3 * x * s3, b3 * xc * s3], t13 - tt
        )
        return 2.0 * xc * np.exp(-2.0 * a3 * tt * rest)

    # t(x) has kinks at x = f and 1 - f: they are segment edges, as is the end-point layer
    lo, hi = min(fp, 1.0 - fp), max(fp, 1.0 - fp)
    xl = np.minimum(X, lo)
    vals = integrate_segments(f, [0.0, xl, lo, hi, 1.0 - xl, 1.0])
    mean = np.sum(vals * w, axis=-1)
    if c_compat and dth != 0:
        mean = mean * 2.0 * hw  # C: the sector integral is not divided by its width
    elif c_compat:
        mean = np.full_like(mean, np.inf)
    return i0 * mean


# ------------------------------------------------------------------------------------------------ shearflow_lin
def _qfvg(q, varphi, phi, theta, psi):
    """Components of Q = q (cos φ_v, sin φ_v, 0) along the flow, velocity-gradient and vorticity axes (Euler ZYZ)."""
    s0, c0 = np.sin(phi), np.cos(phi)
    s1, c1 = np.sin(theta), np.cos(theta)
    s2, c2 = np.sin(psi), np.cos(psi)
    qx, qy = q * np.cos(varphi), q * np.sin(varphi)
    qf = qx * (c0 * c1 * c2 - s0 * s2) + qy * (s0 * c1 * c2 + c0 * s2)
    qv = qx * (-c0 * c1 * s2 - s0 * c2) + qy * (-s0 * c1 * s2 + c0 * c2)
    qg = qx * (c0 * s1) + qy * (s0 * s1)
    return qf, qv, qg


def _lin_core(p, qf, qv, qg, c_compat):
    rg, fp = p[0], abs(p[1])
    al, be, ga, a, b, c = p[2], p[3], p[4], p[5], p[6], p[7]
    bb = (rg * al * qf) ** 2 + (rg * be * qv) ** 2 + (rg * ga * qg) ** 2
    cc = (rg * a * qf) ** 2 + (rg * b * qv) ** 2 + (rg * c * qg) ** 2
    return (_two_stage_c if c_compat else two_stage)(bb, cc, fp)


def ff_shearflow_lin(q, p, psi=None, c_compat=False):
    """p = [Rg, fp, alpha, beta, gamma, A, B, C, phi, theta, psi (Euler, deg), varphi (Q on the detector, deg),
    -, -, -, I0]."""
    q, vphi = _psi(q, p, psi, 11)
    qf, qv, qg = _qfvg(
        q, vphi, math.radians(p[8]), math.radians(p[9]), math.radians(p[10])
    )
    return p[15] * _lin_core(p, qf, qv, qg, c_compat)


def ff_shearflow_lin_g1(q, p, psi=None, c_compat=False):
    """Average over φ in φ0 ± Δφ/2 (θ = ψ = 0): p[11] = varphi, p[12] = phi_0, p[13] = Delta_phi (deg)."""
    q, vphi = _psi(q, p, psi, 11)
    dphi = math.radians(p[13])
    if dphi == 0:
        if c_compat:
            return np.full(q.shape, np.nan)
        phis, w = np.array([math.radians(p[12])]), np.ones(1)
    else:
        phis, w = gauss_legendre(
            64, math.radians(p[12]) - dphi / 2, math.radians(p[12]) + dphi / 2
        )
        w = w / dphi
    qf, qv, qg = _qfvg(q[..., None], vphi[..., None], phis, 0.0, 0.0)
    return p[15] * np.sum(_lin_core(p, qf, qv, qg, c_compat) * w, axis=-1)


def _p_psi(psi, p2, p4):
    c = np.cos(psi)
    leg2 = 0.5 * (3 * c**2 - 1)
    leg4 = (35 * c**4 - 30 * c**2 + 3) / 8.0
    return 1 + 2.5 * p2 * leg2 + 4.5 * p4 * leg4


def ff_shearflow_lin_g2(q, p, psi=None, c_compat=False):
    """Average over the Euler angle ψ in [0, π] with weight 1 + 5/2 P2 P_2(cos(ψ-ψ0)) + 9/2 P4 P_4(cos(ψ-ψ0)):
    p[8] = psi_0, p[9] = P2, p[10] = P4, p[11] = varphi, p[14] = theta_0 (deg), φ = 0."""
    q, vphi = _psi(q, p, psi, 11)
    psis, w = gauss_legendre(128, 0.0, math.pi)
    qf, qv, qg = _qfvg(q[..., None], vphi[..., None], 0.0, math.radians(p[14]), psis)
    weight = _p_psi(psis - math.radians(p[8]), p[9], p[10])
    return (
        p[15]
        * np.sum(_lin_core(p, qf, qv, qg, c_compat) * weight * w, axis=-1)
        / math.pi
    )


# ------------------------------------------------------------------------------------------------ specs
def _qrg(q, p):
    return np.asarray(q, float) * p[0]


_ANG = (0.0, 180.0, "lin")
_SLOPE = (0.2, 3.0, "log")
_WI = {
    0: (1.0, 100.0, "log"),
    1: (0.0, 5.0, "lin"),
    2: (0.3, 0.7, "lin"),
    3: (0.1, 10.0, "log"),
    4: _ANG,
    5: _ANG,
    6: (1.0, 60.0, "lin"),
}
_ONE = {
    0: (1.0, 100.0, "log"),
    1: (0.1, 10.0, "log"),
    2: _ANG,
    3: (1.0, 2.0, "lin"),
    4: (-1.0, 2.0, "lin"),
    5: (-1.0, 2.0, "lin"),
}
_ONE_RING = {
    0: (1.0, 100.0, "log"),
    1: (0.1, 10.0, "log"),
    2: _ANG,
    3: (1.0, 2.0, "lin"),
    4: (0.0, 1.5, "lin"),
    5: (0.0, 0.9, "lin"),
}
_TWO = {
    0: (1.0, 100.0, "log"),
    1: (0.05, 0.95, "lin"),
    2: _SLOPE,
    3: _SLOPE,
    4: _SLOPE,
    5: _SLOPE,
    6: (0.1, 10.0, "log"),
    7: _ANG,
    8: _ANG,
    9: (1.0, 60.0, "lin"),
}
_TWO_RING = {**_TWO, 1: (0.05, 0.45, "lin")}
_LIN = {
    0: (1.0, 100.0, "log"),
    1: (0.05, 0.95, "lin"),
    2: _SLOPE,
    3: _SLOPE,
    4: _SLOPE,
    5: _SLOPE,
    6: _SLOPE,
    7: _SLOPE,
    8: _ANG,
    9: _ANG,
    10: _ANG,
    11: _ANG,
    15: (0.1, 10.0, "log"),
}
_LIN_G1 = {k: v for k, v in _LIN.items() if k not in (8, 9, 10)} | {
    12: _ANG,
    13: (1.0, 90.0, "lin"),
}
_LIN_G2 = {k: v for k, v in _LIN.items() if k not in (8, 9, 10)} | {
    8: _ANG,
    9: (-0.5, 0.5, "lin"),
    10: (-0.3, 0.3, "lin"),
    14: _ANG,
}
_PARS_WI = ["Rg", "Wi", "nu", "I0", "theta_0", "psi", "Delta_theta_0"]
_PARS_ONE = ["Rg", "I0", "psi", "xi", "Bx", "Bz"]
_PARS_TWO = [
    "Rg",
    "f",
    "alpha",
    "beta",
    "gamma",
    "delta",
    "I0",
    "theta_0",
    "psi",
    "Delta_theta_0",
]
_PARS_LIN = [
    "Rg",
    "fp",
    "alpha",
    "beta",
    "gamma",
    "A",
    "B",
    "C",
    "phi",
    "theta",
    "psi",
    "varphi",
    "-",
    "-",
    "-",
    "I0",
]
_PARS_G1 = [*_PARS_LIN[:8], "-", "-", "-", "varphi", "phi_0", "Delta_phi", "-", "I0"]
_PARS_G2 = [*_PARS_LIN[:8], "psi_0", "P2", "P4", "varphi", "-", "-", "theta_0", "I0"]


def _spec(name, func, pars, psi_index, sample, source, notes, c_integrates, rtol=1e-6):
    return {
        "name": name,
        "intensity": func,
        "parameters": pars,
        "psi_index": psi_index,
        "sample": sample,
        "rtol": rtol,
        "source": source,
        "notes": notes,
        "c_integrates": c_integrates,
        "c_compat": False,
        "reduced": _qrg,
    }


_SF = "plugins/shearflow/sasfit_"
_SL = "plugins/shearflow_polymers_lin/sasfit_"
SPECS = [
    _spec(
        "ff_polymer_under_shearflow",
        ff_polymer_under_shearflow,
        _PARS_WI,
        5,
        _WI,
        _SF + "ff_polymer_under_shearflow.c",
        "C: sector average halved, ±Δθ0/4, 1/0 for Δθ0=0 (c_compat)",
        True,
    ),
    _spec(
        "ff_polymer_under_shearflow1",
        ff_polymer_under_shearflow1,
        _PARS_ONE,
        2,
        _ONE,
        _SF + "ff_polymer_under_shearflow1.c",
        "old C: psi from p[7], no I0 (c_compat)",
        True,
    ),
    _spec(
        "ff_polymer_under_shearflow2",
        ff_polymer_under_shearflow2,
        _PARS_TWO,
        8,
        _TWO,
        _SF + "ff_polymer_under_shearflow2.c",
        "exact two-stage formula; C loses digits for small q",
        True,
    ),
    _spec(
        "ff_ring_under_shearflow",
        ff_ring_under_shearflow,
        _PARS_WI,
        5,
        _WI,
        _SF + "ff_ring_under_shearflow.c",
        "C: sector ±Δθ0/4 (c_compat)",
        True,
    ),
    _spec(
        "ff_ring_under_shearflow1",
        ff_ring_under_shearflow1,
        _PARS_ONE,
        2,
        _ONE_RING,
        _SF + "ff_ring_under_shearflow1.c",
        "old C: weight (1-x), P(0) = 1/2, no I0 (c_compat)",
        True,
    ),
    _spec(
        "ff_ring_under_shearflow2",
        ff_ring_under_shearflow2,
        _PARS_TWO,
        8,
        _TWO_RING,
        _SF + "ff_ring_under_shearflow2.c",
        "C: sector integral not divided by its width, 1/0 for Δθ0=0 (c_compat)",
        True,
    ),
    _spec(
        "ff_shearflow_lin",
        ff_shearflow_lin,
        _PARS_LIN,
        11,
        _LIN,
        _SL + "ff_shearflow_lin.c",
        "exact two-stage formula; detector angle is varphi (p[11])",
        False,
    ),
    _spec(
        "ff_shearflow_lin_g1",
        ff_shearflow_lin_g1,
        _PARS_G1,
        11,
        _LIN_G1,
        _SL + "ff_shearflow_lin_g1.c",
        "64-point Gauss-Legendre over phi",
        True,
    ),
    _spec(
        "ff_shearflow_lin_g2",
        ff_shearflow_lin_g2,
        _PARS_G2,
        11,
        _LIN_G2,
        _SL + "ff_shearflow_lin_g2.c",
        "128-point Gauss-Legendre over psi",
        True,
    ),
]
