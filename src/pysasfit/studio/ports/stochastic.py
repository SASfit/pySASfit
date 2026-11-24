"""Python ports of the SASfit plugin stochastic_models (two-phase random media from their correlation function).

I(q) = scale ∫_0^∞ 4π r² γ(r) sin(qr)/(qr) dr with the normalised correlation function γ (γ(0) = 1, γ(∞) = 0),
computed by Gauss-Legendre quadrature in r (panels of at most 2 rad of phase, range to γ < 1e-17).

Clipped (level-cut) Gaussian random fields with field correlation g(r) (gy1 … gy5 of the C code, d = period,
ξ = correlation length, rc = cut-off length), volume fraction φ of the selected phase (Berk 1987, 1991; Roberts 1997):
- one cut, phase y > α, α = √2 erfinv(1 - 2φ):
      γ = 1/(2π φ(1-φ)) ∫_0^{asin g} exp(-α²/(1+sin θ)) dθ
- two cuts, phase |y| < α, α = √2 erfinv(φ):
      γ = 1/(π φ(1-φ)) ∫_0^{asin g} [exp(-α²/(1+sin θ)) - exp(-α²/(1-sin θ))] dθ
- intersection of two independent two-cut fields (each with φ_s = √φ): p₂ = (φ_s² + X_s)²,
      γ = (2 φ_s² X_s + X_s²)/(φ(1-φ)),  X_s = φ_s(1-φ_s) γ_s.
Both normalisations were checked: ∫_0^{π/2} gives φ(1-φ) exactly (γ(0) = 1). gy4 is used normalised to 1 at r = 0
(the expression tends to a finite value ≠ 1 there, ~2.7 for d = 50, ξ = 10, rc = 2).
C before 2026-11-23: one cut subtracted φ² in the numerator (γ(∞) = -φ/(1-φ)), two cuts used the sum of the two
exponentials with 1/(2π) (γ(0) ≠ 1), the intersection C11²/(2π) - φ², abs() (integer) on asin g, gy4 not
normalised. The C code integrates with Ooura's DE rule (intdeo), accurate to ~1e-3 for long correlation lengths
(checked: the port agrees with scipy QAWO to 1e-9); hence rtol 2e-3 against C.

Boolean model of overlapping spheres with log-normal radii (median μ, width σ), matrix fraction φ:
γ = (e^{K(r)} - 1)/(1/φ - 1), K(r) = -ln φ · ⟨V k(r/R)⟩/⟨V⟩ (closed form ThetaK of the C code, checked against the
numerical average of the sphere covariogram). C before 2026-11-23: e^{K-1}. Dead-leaves model: γ = κ/(2 - κ),
κ = K(r)/K(0).
"""

from __future__ import annotations

import math

import numpy as np
from scipy import special

from .common import make_spec

__all__ = ["SPECS"]

_GX, _GW = np.polynomial.legendre.leggauss(16)


def _theta_rule(k=8, n=16):
    """Nodes v in [0, 1] and weights for ∫_0^1 dv: n-point Gauss-Legendre on panels graded geometrically towards
    v = 1 (θ = v asin g near ±π/2, where e^{-α²/(1±sin θ)} varies on a scale ~α)."""
    x, w = np.polynomial.legendre.leggauss(n)
    e = np.concatenate([1.0 - 0.5 ** np.arange(k + 1), [1.0]])
    c, h = 0.5 * (e[1:] + e[:-1]), 0.5 * (e[1:] - e[:-1])
    return (c[:, None] + h[:, None] * x).ravel(), (h[:, None] * w).ravel()


_TV, _TW = _theta_rule()


# ------------------------------------------------------------------------------------------ field correlations
def _j0(x):
    return np.sinc(np.asarray(x, float) / np.pi)


def _g(kind, r, d, xi, rc):
    r = np.asarray(r, float)
    k = 2.0 * math.pi / d
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        if kind == 1:
            x = r / xi
            return np.exp(-np.logaddexp(x, -x) + math.log(2.0)) * _j0(k * r)  # sech
        if kind == 2:
            if rc == xi:
                raise ValueError("rc == xi not allowed")
            return (np.exp(-r / xi) - rc / xi * np.exp(-r / rc)) / (1.0 - rc / xi) * _j0(k * r)
        if kind == 3:
            a = xi / rc
            if a == 1.0:
                raise ValueError("rc == xi not allowed")
            e1 = np.exp(-r / xi)
            diff = -e1 * np.expm1(-(a - 1.0) * r / xi)  # e^{-r/ξ} - e^{-a r/ξ}
            frac = np.where(r > 0, 4.0 * xi / np.where(r > 0, r, 1.0) / (a * a - 1.0) * diff, 4.0 / (a + 1.0))
            return a * (a + 1.0) / (a - 1.0) ** 2 * (e1 + np.exp(-a * r / xi) / a - frac) * _j0(k * r)
        AA, BB, CC = k, 1.0 / xi, 1.0 / rc
        a2, b2, c2 = AA * AA, BB * BB, CC * CC
        if kind == 4:
            c4 = c2 * c2
            den = (4 * a2 * b2 + (a2 - b2 + c2) ** 2) ** 2
            rs = np.where(r > 0, r, 1.0)
            val = (4 * BB * CC * (a2 + (BB + CC) ** 2) / ((BB + CC) * rs)
                   * (np.exp(-CC * r) * ((a2 - b2 + c2) / den
                                         + r / (4 * CC * ((a2 + b2) ** 2 + 2 * (a2 - b2) * c2 + c4)))
                      + np.exp(-BB * r) / (AA * BB)
                      * ((-8 * a2 * b2 + (a2 + b2) ** 2 + 2 * (a2 - b2) * c2 + c4) / (4 * den) * np.sin(AA * r)
                         - (AA * BB * (a2 - b2 + c2)) / den * np.cos(AA * r))))  # fmt: skip
            return val
        if kind == 5:
            abc = a2 + c2 - b2
            rs = np.where(r > 0, r, 1.0)
            eb = np.exp(-BB * r)
            # (e^{-CC r} - e^{-BB r} cos(AA r))/r without cancellation
            tail = eb * (np.expm1(-(CC - BB) * r) + 2.0 * np.sin(0.5 * AA * r) ** 2) / rs
            tail0 = BB - CC  # limit r -> 0 of (e^{-CC r} - e^{-BB r})/r, the cos term vanishes like r
            val = (abc * _j0(AA * r) * eb + 2.0 * BB * np.where(r > 0, tail, tail0)) / (a2 + (CC - BB) ** 2)
            return np.where(r > 0, val, 1.0)
    raise ValueError(kind)


def _g4n(r, d, xi, rc):
    """gy4 of the C code divided by its limit for r -> 0 (the C expression tends to ~5, not 1, so the C code clips
    g to 1 over a finite range); below r0 the limit itself (the expression loses digits there)."""
    r = np.asarray(r, float)
    r0 = 1e-6 * min(d, xi, rc)
    g0 = float(_g(4, np.array([r0]), d, xi, rc)[0])
    return np.where(r > r0, _g(4, np.maximum(r, r0), d, xi, rc) / g0, 1.0)


def _x_cov(g, alpha, two_cut):
    """X(g) = p₂ - φ²: 1/(2π) ∫_0^{asin g} e^{-α²/(1+s)} dθ (one cut), 1/π ∫ [e^{-α²/(1+s)} - e^{-α²/(1-s)}] (two)."""
    th = np.arcsin(np.clip(g, -1.0, 1.0))
    t = th[..., None] * _TV
    s = np.sin(t)
    a2 = alpha * alpha
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        k = np.exp(-a2 / (1.0 + s))
        k = np.where(1.0 + s > 0, k, 0.0)
        if two_cut:  # e^{-a/(1+s)} - e^{-a/(1-s)} without cancellation for small s
            inner = (1.0 + s > 0) & (1.0 - s > 0)
            k2 = np.where(1.0 - s > 0, np.exp(-a2 / (1.0 - s)), 0.0)
            k = np.where(inner, -k * np.expm1(-2.0 * a2 * s / np.where(inner, (1.0 - s) * (1.0 + s), 1.0)), k - k2)
    integral = th * np.sum(_TW * k, axis=-1)
    return integral / (math.pi if two_cut else 2.0 * math.pi)


def _gk(kind, r, d, xi, rc):
    return _g4n(r, d, xi, rc) if kind == 4 else _g(kind, r, d, xi, rc)


def _crw_gamma(kind, mode, fp, d, xi, rc):
    if not 0.0 < fp < 1.0:
        raise ValueError("0 < phi_1 < 1 required")
    if d <= 0 or xi <= 0 or (kind > 1 and rc <= 0):
        raise ValueError("d, xi (and rc) > 0 required")
    if mode == "one":
        alpha = math.sqrt(2.0) * special.erfinv(1.0 - 2.0 * fp)
        return lambda r: _x_cov(_gk(kind, r, d, xi, rc), alpha, False) / (fp * (1.0 - fp))
    if mode == "two":
        alpha = math.sqrt(2.0) * special.erfinv(fp)
        return lambda r: _x_cov(_gk(kind, r, d, xi, rc), alpha, True) / (fp * (1.0 - fp))
    fs = math.sqrt(fp)
    alpha = math.sqrt(2.0) * special.erfinv(fs)

    def gam(r):
        xs = _x_cov(_gk(kind, r, d, xi, rc), alpha, True)
        return (2.0 * fs * fs * xs + xs * xs) / (fp * (1.0 - fp))

    return gam


def _transform(q, gamma, rmax, hmax, scale):
    """scale ∫_0^rmax 4π r² γ(r) sinc(qr) dr (Gauss-Legendre, panels <= min(hmax, 2/q_max))."""
    q = np.atleast_1d(np.asarray(q, float))
    h = min(hmax, 2.0 / max(float(np.max(q)), 1e-300))
    n = max(4, math.ceil(rmax / h))
    e = np.linspace(0.0, rmax, n + 1)
    c, hw = 0.5 * (e[1:] + e[:-1]), 0.5 * (e[1:] - e[:-1])
    r = (c[:, None] + hw[:, None] * _GX).ravel()
    w = (hw[:, None] * _GW).ravel() * 4.0 * math.pi * r * r * gamma(r)
    out = np.empty(q.shape)
    for i0 in range(0, q.size, 64):
        qq = q[i0 : i0 + 64]
        out[i0 : i0 + 64] = np.sinc(qq[:, None] * r[None, :] / np.pi) @ w
    return scale * out


def _crw(kind, mode):
    def f(q, p):
        q = np.asarray(q, float)
        scale, fp, d, xi = p[0], p[1], p[2], p[3]
        rc = p[4] if kind > 1 else xi
        if scale < 0:
            raise ValueError("scale >= 0 required")
        gam = _crw_gamma(kind, mode, fp, d, xi, rc)
        lmax = max(xi, rc) if kind > 1 else xi
        lmin = min(xi, rc) if kind > 1 else xi
        return _transform(q, gam, 45.0 * lmax, min(lmin / 4.0, d / 8.0), scale)

    f.__doc__ = (f"p = [scale, phi_1, d, xi{', rc' if kind > 1 else ''}]: clipped random waves, field correlation "
                 f"gy{kind}, {'one cut' if mode == 'one' else 'two cuts' if mode == 'two' else 'intersection of two two-cut fields'}.")  # fmt: skip
    return f


# --------------------------------------------------------------------------------- Boolean / dead leaves
def theta_k(r, fp, mu, sigma):
    """K(r) = -ln φ ⟨V k(r/R)⟩/⟨V⟩ for log-normal sphere radii (median μ, width σ), k = sphere covariogram / V."""
    r = np.asarray(r, float)
    s2 = sigma * sigma
    lf = math.log(fp)
    with np.errstate(divide="ignore", invalid="ignore"):
        lm = np.log(2.0 * mu / np.where(r > 0, r, 1.0))
        sq = math.sqrt(2.0) * sigma
        val = -((r**3 * special.erfc(-lm / sq) - 12 * math.exp(2 * s2) * mu**2 * r * special.erfc(-(2 * s2 + lm) / sq)
                 + 16 * math.exp(4.5 * s2) * mu**3 * special.erfc(-(3 * s2 + lm) / sq)) * lf) / (32 * math.exp(4.5 * s2) * mu**3)  # fmt: skip
    small = r / mu < 1e-10
    return np.where(small, -lf + 3 * r * lf / (4 * math.exp(2.5 * s2) * mu), val)


def _bool_range(mu, sigma):
    return 2.0 * mu * math.exp(3.0 * sigma * sigma + 9.0 * sigma) * 1.05, mu * math.exp(-2.0 * sigma) / 8.0


def boolean_union(q, p):
    """p = [Scale, phi_1, mu, sigma]: Boolean model of overlapping spheres, log-normal radii (median mu)."""
    q = np.asarray(q, float)
    scale, fp, mu, sigma = p[0], p[1], p[2], p[3]
    if not 0 < fp < 1 or mu <= 0 or sigma < 0 or scale < 0:
        raise ValueError("0 < phi < 1, mu > 0, sigma >= 0, Scale >= 0 required")
    rmax, h = _bool_range(mu, sigma)
    return _transform(q, lambda r: np.expm1(theta_k(r, fp, mu, sigma)) / (1.0 / fp - 1.0), rmax, h, scale)


def dead_leaves(q, p):
    """p = [Scale, dummy, mu, sigma]: dead-leaves model of spheres, log-normal radii, γ = κ/(2 - κ)."""
    q = np.asarray(q, float)
    scale, mu, sigma = p[0], p[2], p[3]
    if mu <= 0 or sigma < 0 or scale < 0:
        raise ValueError("mu > 0, sigma >= 0, Scale >= 0 required")
    rmax, h = _bool_range(mu, sigma)
    k0 = float(theta_k(np.array([0.0]), 0.5, mu, sigma)[0])

    def gam(r):
        kap = theta_k(r, 0.5, mu, sigma) / k0
        return kap / (2.0 - kap)

    return _transform(q, gam, rmax, h, scale)


_SM = "plugins/stochastic_models/sasfit_ff_"
_CRW4 = ["scale", "phi_1", "d", "xi"]
_CRW5 = _CRW4 + ["rc"]
_CRWS = {0: (0.5, 2.0, "lin"), 1: (0.1, 0.9, "lin"), 2: (20.0, 100.0, "log"), 3: (5.0, 40.0, "log"), 4: (1.0, 4.0, "log")}
_BS = {0: (0.5, 2.0, "lin"), 1: (0.1, 0.9, "lin"), 2: (5.0, 30.0, "log"), 3: (0.05, 0.5, "lin")}


def _specs():
    out = []
    for mode, pre, src in (("one", "ff_clipped_random_waves_", "clipped_random_waves_"),
                           ("two", "ff_clipped_random_waves_twocut_", "clipped_random_waves_twocut_"),
                           ("inter", "ff_clipped_random_waves_twocut_intersection_", "clipped_random_waves_twocut_intersection_")):  # fmt: skip
        for kind in range(1, 6):
            out.append(make_spec(f"{pre}{kind}", _crw(kind, mode), _CRW4 if kind == 1 else _CRW5, _CRWS,
                                 source=f"{_SM}{src}{kind}.c", rtol=2e-3,
                                 notes="C fixed 2026-11-23 (see module doc); C quadrature (intdeo) ~1e-3"))  # fmt: skip
    out.append(make_spec("ff_boolean_union_model", boolean_union, ["Scale", "phi_1", "mu", "sigma"], _BS,
                         source=_SM + "boolean_union_model.c", rtol=2e-3,
                         notes="C before 2026-11-23: exp(K-1) instead of exp(K)-1; C quadrature (intdeo) ~1e-4"))  # fmt: skip
    out.append(make_spec("ff_dead_leaves_model", dead_leaves, ["Scale", "dummy", "mu", "sigma"], _BS,
                         source=_SM + "dead_leaves_model.c", rtol=2e-3,
                         notes="C quadrature (intdeo) ~1e-4"))  # fmt: skip
    return out


SPECS = _specs()
