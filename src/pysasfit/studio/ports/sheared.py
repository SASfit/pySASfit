"""Python ports of the SASfit plugin ``sheared_objects``: core-shell cylinders and spheroids with an orientation
distribution (sheared samples).

I(q, ψ) = <∫_{S²} p(û) |A(q, γ)|² dΩ>_ν

* û = (sinθ cosφ, sinθ sinφ, cosθ) is the particle axis; ∫ p dΩ = 1.
* γ is the angle between q and the particle axis. For the axially symmetric distributions (Maier-Saupe, Gauss,
  Boltzmann, Onsager, Heaviside, Legendre), q̂ = (sinψ, 0, cosψ), so cos γ = cosψ cosθ + sinψ sinθ cosφ. For
  Hayter-Penfold, q̂ = (cosψ, 0, sinψ), so cos γ = cosψ sinθ cosφ + sinψ cosθ.
* ν scales all lengths and has a normalised log-normal distribution of width σ (median 1); σ = 0 means
  monodisperse. The average uses 24-node Gauss-Hermite quadrature in ln ν.

Amplitudes (P = 2):

* cylinder: A = (η_c-η_s) πR²L Λ₁(qR sinγ) j₀(qL cosγ/2) + (η_s-η_solv) π(R+t)²L Λ₁(q(R+t) sinγ) j₀(qL cosγ/2),
  with Λ₁(x) = 2J₁(x)/x. Core and shell have the same length L (no end caps), so I(0) = (Δη V)².
* spheroid: A = (η_c-η_s) V_c Φ(x_c) + (η_s-η_solv) V_t Φ(x_t), with Φ(x) = 3j₁(x)/x and
  x = q sqrt(a² cos²γ + b² sin²γ). Semi-axes: a = R_polar (p[2]) along the axis, b = R_equatorial (p[0]); the
  shell adds t to both.

Orientation distributions (κ = p[8]; t = min(θ, π-θ)). All except Hayter-Penfold are normalised numerically, which
avoids the special functions of the C normalisation constants:

* Maier-Saupe: exp(κ cos²θ).
* Gauss: κ > 0: exp(-(κ t)²); κ < 0: exp(-(|κ| (π/2-t))²).
* Boltzmann: exp(-κ t).
* Onsager: κ > 0: cosh(κ cosθ); κ < 0: cosh(κ sinθ).
* Heaviside: κ > 0: 1 for t < 1/κ (cone around the axis); κ < 0: 1 for |θ-π/2| < 1/|κ| (equatorial band);
  |κ| < 0.1: uniform.
* Legendre: 1/2 + Σ_{l=2,4,6} (l+1/2) |S_l| P_l(cosθ), with S_2, S_4, S_6 = |p[8]|, |p[9]|, |p[10]| (not normalised
  numerically: the l > 0 terms integrate to 0).
* Hayter-Penfold: (1-cos2φ₀)(1+sin²θ cos2φ₀)^{3/2} / (4π (1-sin²θ cos2φ₀ cos 2(φ-φ₀))²), 2φ₀ = atan(8/κ);
  normalised analytically.

Integration. The θ integral runs over [0, π/2] and uses the symmetry (θ, φ) → (π-θ, φ+π), which maps cos γ to
-cos γ and keeps p. It uses Gauss-Legendre panels with breakpoints at the width of the distribution and at the
Heaviside edges. The φ integral uses the trapezoidal rule, which is spectrally accurate for these periodic
integrands. The number of nodes grows with q times the particle size.

C code. The SASfit plugin sheared_objects was corrected on 2026-11-17 to compute the same quantity. Before that:

* ``abs()`` of a double in pMaierSaupe, pGauss, pOnsager, pHeavysidePi and pLegendre truncated θ to the integer
  0 or 1.
* θ ∈ [0, π/2], φ ∈ [0, π] (a quarter of the sphere) without the factor 4 (only the cylinder Boltzmann function
  had it), and the cylinder amplitude had an extra factor 2.
* Hayter-Penfold missed the half with cos γ₋; the spheroid version never set cparam.gam (uninitialised pointer).
* pOnsager used cos(x) instead of cos(θ) for κ > 10; the Heaviside θ range was wrong for |κ| > 2/π.
* Boltzmann cylinder: the φ limits were never set. pLegendre wrote 8 values into Pl[7].
* sheared_cylinders__legendre_ was not exported and did not compile.

With the corrected C, the ports agree to ~1e-13 for σ = 0 (Linux build, tests/studio/port_reference). For σ > 0
the C function integrates ν only up to the 99.9 % point of the ν⁶-weighted log-normal
(find_LogNorm_int_range), which gives relative deviations of ~1e-5; the comparison samples σ = 0.
"""

from __future__ import annotations

import itertools
import math
import os
from functools import lru_cache

import numpy as np
from scipy import special

from . import _sheared_numba

__all__ = [
    "BACKEND",
    "DISTRIBUTIONS",
    "METHOD",
    "orientation_moments",
    "sheared_intensity",
]

#: default backend of :func:`sheared_intensity`: "auto" (Numba when installed), "numba" or "numpy";
#: the environment variable PYSASFIT_SHEARED_BACKEND overrides it, PYSASFIT_NO_NUMBA=1 disables Numba
BACKEND = os.environ.get("PYSASFIT_SHEARED_BACKEND", "auto")
#: Hayter-Penfold: largest Legendre degree of the Legendre method (above: θ, φ quadrature; memory bound)
HP_LMAX = 4000
#: default method: "auto" (Legendre expansion for the axially symmetric distributions), "legendre", "quadrature"
METHOD = os.environ.get("PYSASFIT_SHEARED_METHOD", "auto")

DISTRIBUTIONS = (
    "maier_saupe",
    "gauss",
    "boltzmann",
    "onsager",
    "heavyside",
    "legendre",
    "hayterpenfold",
)


# ------------------------------------------------------------------------------------------------ quadrature
@lru_cache(maxsize=512)
def _gl(n):
    x, w = np.polynomial.legendre.leggauss(n)
    return x, w


def _panels(edges, n_per_unit, n_min=8):
    """Gauss-Legendre nodes/weights on consecutive panels [edges[i], edges[i+1]]."""
    xs, ws = [], []
    for a, b in itertools.pairwise(edges):
        if b <= a:
            continue
        n = int(min(400, max(n_min, math.ceil(n_per_unit * (b - a)))))
        x, w = _gl(n)
        xs.append(0.5 * (b - a) * x + 0.5 * (b + a))
        ws.append(0.5 * (b - a) * w)
    return np.concatenate(xs), np.concatenate(ws)


@lru_cache(maxsize=8)
def _hermite(n=24):
    z, w = np.polynomial.hermite_e.hermegauss(n)  # weight exp(-z²/2)
    return z, w / math.sqrt(2.0 * math.pi)


def _geometric(w, top=0.5 * math.pi):
    """Breakpoints w, 2w, 4w, ... below top (refinement around a peak of width w at 0)."""
    out = []
    x = w
    while x < top and len(out) < 30:
        out.append(x)
        x *= 2.0
    return out


# ------------------------------------------------------------------------------------------------ distributions
def _width(kind, kappa):
    """Angular width of the distribution peak (None if it is broad) and where it sits (0 or π/2)."""
    k = abs(kappa)
    if kind == "maier_saupe" and k > 1:
        return 1.0 / math.sqrt(k), (0.0 if kappa > 0 else 0.5 * math.pi)
    if kind == "gauss" and k > 1:
        return 1.0 / k, (0.0 if kappa > 0 else 0.5 * math.pi)
    if kind == "boltzmann" and k > 1:
        return 1.0 / k, (0.0 if kappa > 0 else 0.5 * math.pi)
    if kind == "onsager" and k > 1:
        return 1.0 / math.sqrt(k), (0.0 if kappa > 0 else 0.5 * math.pi)
    if kind == "hayterpenfold" and kappa != 0:
        return 0.5 * math.atan(8.0 / k), 0.5 * math.pi
    return None, None


def _theta_edges(kind, kappa):
    edges = {0.0, 0.5 * math.pi}
    if kind == "heavyside" and abs(kappa) >= 0.1:
        k = 1.0 / abs(kappa)
        if k < 0.5 * math.pi:
            edges.add(k if kappa > 0 else 0.5 * math.pi - k)
    w, at = _width(kind, kappa)
    if w is not None:
        for x in _geometric(w):
            edges.add(x if at == 0.0 else 0.5 * math.pi - x)
    return sorted(edges)


def _shape(kind, theta, p):
    """Unnormalised axially symmetric distribution on θ ∈ [0, π/2]."""
    kappa = p[8]
    c, s = np.cos(theta), np.sin(theta)
    if kind == "maier_saupe":
        return np.exp(
            kappa * (c * c - (1.0 if kappa > 0 else 0.0))
        )  # scaled to avoid overflow
    if kind == "gauss":
        if kappa >= 0:
            return np.exp(-((kappa * theta) ** 2))
        return np.exp(-((abs(kappa) * (0.5 * np.pi - theta)) ** 2))
    if kind == "boltzmann":
        return np.exp(-kappa * theta)
    if kind == "onsager":
        x = np.abs(kappa) * (c if kappa >= 0 else s)
        return np.exp(np.logaddexp(x, -x) - np.log(2.0) - abs(kappa))  # cosh(x)/e^{|κ|}
    if kind == "heavyside":
        if abs(kappa) < 0.1:
            return np.ones_like(theta)
        k = 1.0 / abs(kappa)
        if kappa > 0:
            return (theta < k).astype(float)
        return (np.abs(theta - 0.5 * np.pi) < k).astype(float)
    if kind == "legendre":
        x = c
        out = 0.5 + 0.0 * x
        for l, sl in ((2, abs(p[8])), (4, abs(p[9])), (6, abs(p[10]))):
            out = out + (l + 0.5) * sl * special.eval_legendre(l, x)
        return out / (2.0 * np.pi)
    raise ValueError(kind)


def orientation_norm(kind, p):
    """∫_{S²} shape dΩ for the axially symmetric distributions (Legendre: 1)."""
    if kind == "legendre":
        return 1.0
    th, w = _panels(_theta_edges(kind, p[8]), 200.0, n_min=48)
    return 4.0 * np.pi * np.sum(w * np.sin(th) * _shape(kind, th, p))


def hayter_penfold(theta, phi, kappa):
    if kappa == 0:
        return np.full(np.broadcast(theta, phi).shape, 0.25 / np.pi)
    twophi0 = math.atan(8.0 / kappa)
    c2 = math.cos(twophi0)
    s2 = np.sin(theta) ** 2
    return (
        (1.0 - c2)
        * np.sqrt((1.0 + s2 * c2) ** 3)
        / (4.0 * np.pi * (1.0 - s2 * c2 * np.cos(2.0 * (phi - 0.5 * twophi0))) ** 2)
    )


# ------------------------------------------------------------------------------------------------ amplitudes
def _lambda1(x):
    x = np.asarray(x, float)
    small = np.abs(x) < 1e-4
    xs = np.where(small, 1.0, x)
    return np.where(small, 1.0 - x * x / 8.0, 2.0 * special.j1(xs) / xs)


def _j0(x):
    x = np.asarray(x, float)
    small = np.abs(x) < 1e-4
    xs = np.where(small, 1.0, x)
    return np.where(small, 1.0 - x * x / 6.0, np.sin(xs) / xs)


def _phi_sph(x):
    x = np.asarray(x, float)
    small = np.abs(x) < 1e-3
    xs = np.where(small, 1.0, x)
    return np.where(
        small,
        1.0 - x * x / 10.0 + x**4 / 280.0,
        3.0 * (np.sin(xs) - xs * np.cos(xs)) / xs**3,
    )


def _amp_cylinder(q, cosg, p, nu):
    r, t, length = p[0] * nu, p[1] * nu, p[2] * nu
    ec, es, ev = p[3], p[4], p[5]
    sing = np.sqrt(np.maximum(0.0, 1.0 - cosg * cosg))
    axial = _j0(0.5 * q * length * cosg)
    return axial * (
        (ec - es) * np.pi * r * r * length * _lambda1(q * r * sing)
        + (es - ev) * np.pi * (r + t) ** 2 * length * _lambda1(q * (r + t) * sing)
    )


def _amp_spheroid(q, cosg, p, nu):
    b, t, a = p[0] * nu, p[1] * nu, p[2] * nu
    ec, es, ev = p[3], p[4], p[5]
    c2 = cosg * cosg
    xc = q * np.sqrt(a * a * c2 + b * b * (1.0 - c2))
    xt = q * np.sqrt((a + t) ** 2 * c2 + (b + t) ** 2 * (1.0 - c2))
    vc = 4.0 / 3.0 * np.pi * a * b * b
    vt = 4.0 / 3.0 * np.pi * (a + t) * (b + t) ** 2
    return (ec - es) * vc * _phi_sph(xc) + (es - ev) * vt * _phi_sph(xt)


_AMP = {"cylinder": _amp_cylinder, "spheroid": _amp_spheroid}


def _size(shape, p):
    r, t, a = abs(p[0]), abs(p[1]), abs(p[2])
    if shape == "cylinder":
        return math.hypot(a, 2.0 * (r + t))
    return 2.0 * max(a + t, r + t)


# ------------------------------------------------------------------------------------------------ Legendre method
_TABLES = {}  # (lm, nt) -> (t, w, P_l(t)); bounded by _TABLE_BUDGET bytes, oldest dropped first
_TABLE_BUDGET = 256 * 2**20


def _gl_legendre(lm, nt):
    """Gauss-Legendre nodes/weights on [0, 1] and the table P_l(t) (l = 0..lm) at them (cached)."""
    key = (lm, nt)
    hit = _TABLES.get(key)
    if hit is not None:
        return hit
    t, wt = _gl(nt)
    t, wt = 0.5 * (t + 1.0), 0.5 * wt
    tab = _legendre_table(lm, t)
    tab.flags.writeable = False
    size = sum(v[2].nbytes for v in _TABLES.values())
    while _TABLES and size + tab.nbytes > _TABLE_BUDGET:
        size -= _TABLES.pop(next(iter(_TABLES)))[2].nbytes
    _TABLES[key] = (t, wt, tab)
    return t, wt, tab


def _legendre_table(lmax, x):
    """P_l(x) for l = 0..lmax (rows), x any 1D array."""
    x = np.asarray(x, float)
    out = np.empty((lmax + 1, x.size))
    out[0] = 1.0
    if lmax >= 1:
        out[1] = x
    for l in range(1, lmax):
        out[l + 1] = ((2 * l + 1) * x * out[l] - l * out[l - 1]) / (l + 1)
    return out


def _subpanels(edges, n_per_unit, max_len=0.25, n_min=16):
    """Gauss-Legendre nodes on the panels between ``edges``, each split into pieces of length <= max_len."""
    xs, ws = [], []
    for a, b in itertools.pairwise(edges):
        if b <= a:
            continue
        m = max(1, math.ceil((b - a) / max_len))
        h = (b - a) / m
        n = max(n_min, math.ceil(n_per_unit * h))
        x, w = _gl(n)
        for i in range(m):
            lo = a + i * h
            xs.append(lo + 0.5 * h * (x + 1.0))
            ws.append(0.5 * h * w)
    return np.concatenate(xs), np.concatenate(ws)


def orientation_moments(kind, p, lmax):
    """p_l = ∫ p(û) P_l(cos θ) dΩ (∫ p dΩ = 1) for l = 0..lmax; odd l vanish (p(θ) = p(π-θ))."""
    th, w = _subpanels(_theta_edges(kind, p[8]), 1.2 * lmax + 60.0)
    dens = _shape(kind, th, p) / orientation_norm(kind, p)
    pl = 4.0 * np.pi * (_legendre_table(lmax, np.cos(th)) @ (w * np.sin(th) * dens))
    pl[1::2] = 0.0
    return pl


def _assoc_legendre_m(lmax, m, x):
    """Normalised associated Legendre functions λ_l^m(x) = Y_l^m(θ, 0), l = m..lmax (rows m..lmax of an
    (lmax+1, x.size) array, rows < m zero); stable three-term recurrence in l."""
    x = np.asarray(x, float)
    out = np.zeros((lmax + 1, x.size))
    s = np.sqrt(np.maximum(0.0, 1.0 - x * x))
    pmm = np.full(x.size, math.sqrt(0.25 / math.pi))
    for k in range(1, m + 1):
        pmm = -math.sqrt((2 * k + 1) / (2.0 * k)) * s * pmm
    if m > lmax:
        return out
    out[m] = pmm
    if m + 1 <= lmax:
        out[m + 1] = x * math.sqrt(2 * m + 3) * pmm
    for l in range(m + 2, lmax + 1):
        a = math.sqrt((4.0 * l * l - 1.0) / (l * l - m * m))
        b = math.sqrt(((l - 1.0) ** 2 - m * m) / (4.0 * (l - 1.0) ** 2 - 1.0))
        out[l] = a * (x * out[l - 1] - b * out[l - 2])
    return out


def _hayter_penfold_legendre(kappa, lmax, psi, accuracy=1.0):
    """g_l(ψ) = ∫ p_HP(û) P_l(û·q̂) dΩ, q̂ = (cos ψ, 0, sin ψ), from the Fourier series of p_HP in φ,
    p = Σ_m c_m(θ) cos(m(φ-φ₀)) (m even), and the addition theorem:
    g_l = 4π/(2l+1) Σ_m ε_m λ_l^m(cos θ_q) cos(m(φ_q-φ₀)) ∫ c_m λ_l^m sin θ dθ (ε_0 = 2π, ε_m = 2π).
    Cost ~ l_max · m_max · n_θ instead of a sphere quadrature per ψ."""
    psi = np.atleast_1d(np.asarray(psi, float))
    out = np.zeros((lmax + 1, psi.size))
    if kappa == 0:
        out[0] = 1.0
        return out
    w, _ = _width("hayterpenfold", kappa)
    phi0 = 0.5 * math.atan(8.0 / kappa)
    edges = sorted({0.0, 0.5 * np.pi, *(0.5 * np.pi - x for x in _geometric(w))})
    th, wth = _subpanels(edges, accuracy * (1.2 * lmax + 60.0))
    nphi = 2 ** math.ceil(math.log2(max(64.0, accuracy * 64.0 / w)))
    phi = phi0 + np.arange(nphi) * (2.0 * np.pi / nphi)
    dens = hayter_penfold(th[:, None], phi[None, :], kappa)
    cm = (
        np.fft.rfft(dens, axis=1).real / nphi
    )  # c_m/2 for m >= 1 (cos series about φ₀), c_0 for m = 0
    cm[:, 1:] *= 2.0
    mag = np.max(np.abs(cm), axis=0)
    mmax = min(lmax, int(np.nonzero(mag > 1e-16 * mag[0])[0].max()))
    cq = np.sin(psi)  # cos θ_q
    phq = np.where(np.cos(psi) >= 0.0, 0.0, np.pi)
    wsin = (
        2.0 * wth * np.sin(th)
    )  # ∫_0^π = 2 ∫_0^{π/2} for the even (l, m) that survive
    ls = np.arange(lmax + 1)
    for m in range(0, mmax + 1, 2):
        lam = _assoc_legendre_m(lmax, m, np.cos(th))
        theta_lm = lam @ (wsin * cm[:, m])
        theta_lm[1::2] = 0.0
        lq = _assoc_legendre_m(lmax, m, cq)
        fac = (
            2.0 * np.pi * np.cos(m * (phq - phi0))
        )  # ε_m: 2π (m = 0) or 2·π (m >= 1, cos-cos average)
        out += (
            (4.0 * np.pi / (2 * ls + 1))[:, None]
            * (theta_lm[:, None] * lq)
            * fac[None, :]
        )
    return out


def orientation_legendre(kind, p, lmax, psi, accuracy=1.0, method="fourier"):
    """g_l(ψ) = ∫ p(û) P_l(û·q̂(ψ)) dΩ for l = 0..lmax (rows) and the directions ψ (columns), so that
    I(q, ψ) = Σ_l (l+1/2) a_l(q) g_l(ψ). Axially symmetric distributions: g_l = p_l P_l(cos ψ) (addition
    theorem); Hayter-Penfold (q̂ = (cos ψ, 0, sin ψ)): sphere quadrature, θ panels refined around the equator."""
    psi = np.atleast_1d(np.asarray(psi, float))
    if kind != "hayterpenfold":
        return orientation_moments(kind, p, lmax)[:, None] * _legendre_table(
            lmax, np.cos(psi)
        )
    if method == "fourier":
        return _hayter_penfold_legendre(p[8], lmax, psi, accuracy)
    kappa = p[8]
    w, _ = _width(kind, kappa)
    edges = {0.0, 0.5 * np.pi}
    if w is not None:
        edges.update(0.5 * np.pi - x for x in _geometric(w))
    edges = sorted(edges)
    th, wth = _subpanels(edges, accuracy * (1.2 * lmax + 60.0))
    th = np.concatenate([th, np.pi - th])  # p(π-θ, φ) = p(θ, φ)
    wth = np.concatenate([wth, wth])
    nphi = 2 * math.ceil(accuracy * max(1.2 * lmax + 40.0, 24.0 / w if w else 0.0))
    phi = np.arange(nphi) * (2.0 * np.pi / nphi)
    T, F = np.meshgrid(th, phi, indexing="ij")
    wts = (
        (wth * np.sin(th))[:, None] * (2.0 * np.pi / nphi) * hayter_penfold(T, F, kappa)
    )
    st, ct, cf = np.sin(T).ravel(), np.cos(T).ravel(), np.cos(F).ravel()
    wv = wts.ravel()
    out = np.empty((lmax + 1, psi.size))
    for j, ps in enumerate(
        psi
    ):  # Legendre recurrence, accumulated (no (l, node) table)
        x = math.cos(ps) * st * cf + math.sin(ps) * ct
        p0, p1 = np.ones_like(x), x
        out[0, j] = wv.sum()
        if lmax >= 1:
            out[1, j] = p1 @ wv
        for l in range(1, lmax):
            p0, p1 = p1, ((2 * l + 1) * x * p1 - l * p0) / (l + 1)
            out[l + 1, j] = p1 @ wv
    return out


def _size_nodes_q(size_dist, sigma, qd, accuracy=1.0):
    """ν nodes and weights w_k ν_k⁶ for the size average at q D = ``qd``: composite Gauss-Legendre in ν over
    the bulk of the ν⁶-weighted distribution (tails below ~1e-13 dropped), with ~3 nodes per
    oscillation of I₁(q ν) (period π/(q D) in ν) plus a floor for the shape of the distribution."""
    if size_dist == "lognormal":  # ln ν ~ N(0, σ²); ν⁶ shifts the mean of ln ν to 6σ²
        lo, hi = (
            math.exp(6.0 * sigma * sigma - 7.5 * sigma),
            math.exp(6.0 * sigma * sigma + 7.5 * sigma),
        )
    else:  # Schulz-Zimm: ν⁶ pdf ∝ ν^(z+6) e^(-(z+1)ν), a gamma distribution
        z = (1.0 - sigma * sigma) / (sigma * sigma)
        if z <= -1.0:
            raise ValueError("Schulz-Zimm needs sigma < 1")
        lo = special.gammaincinv(z + 7.0, 1e-13) / (z + 1.0)
        hi = special.gammainccinv(z + 7.0, 1e-13) / (z + 1.0)
    n = math.ceil(accuracy * (3.0 * qd * (hi - lo) / math.pi + 64.0))
    m = max(8, math.ceil(n / 48))
    x, w = _gl(math.ceil(n / m))
    h = (hi - lo) / m
    nus = (lo + h * (np.arange(m)[:, None] + 0.5 * (x[None, :] + 1.0))).ravel()
    wts = np.tile(0.5 * h * w, m)
    if size_dist == "lognormal":
        lw = (
            -0.5 * (np.log(nus) / sigma) ** 2
            - np.log(nus)
            - math.log(sigma * math.sqrt(2.0 * math.pi))
        )
    else:
        lw = (
            z * np.log(nus)
            - (z + 1.0) * nus
            - (special.gammaln(z + 1.0) - (z + 1.0) * math.log(z + 1.0))
        )
    return nus, wts * np.exp(lw + 6.0 * np.log(nus))


def _size_nodes(size_dist, sigma):
    """Nodes ν_k and weights w_k ν_k⁶ of the size distribution of ν (Σ w_k ν_k⁶ f(ν_k) ≈ ∫ pdf(ν) ν⁶ f(ν) dν).

    lognormal: median 1, width σ of ln ν (as the SASfit plugin), 24-node Gauss-Hermite in ln ν.
    schulz: Schulz-Zimm with mean 1 and relative standard deviation σ, z = (1-σ²)/σ²; generalised
    Gauss-Laguerre (32 nodes) with the weight ν^(z+6), Gauss-Legendre over ±12 standard deviations for z > 100."""
    if size_dist == "lognormal":
        z, wz = _hermite(24)
        nus = np.exp(sigma * z)
        return nus, wz * nus**6
    z = (1.0 - sigma * sigma) / (sigma * sigma)
    if z <= -1.0:
        raise ValueError("Schulz-Zimm needs sigma < 1")
    lm6 = (
        special.gammaln(z + 7.0) - special.gammaln(z + 1.0) - 6.0 * math.log(z + 1.0)
    )  # ln <ν⁶>
    if z <= 100.0:
        t, w = special.roots_genlaguerre(32, z + 6.0)
        return t / (z + 1.0), w / w.sum() * math.exp(lm6)
    sd = 1.0 / math.sqrt(z + 1.0)
    x, w = _gl(64)
    lo, hi = max(1e-6, 1.0 - 12.0 * sd), 1.0 + 12.0 * sd
    nus = lo + 0.5 * (hi - lo) * (x + 1.0)
    lw = (z + 6.0) * np.log(nus) - (z + 1.0) * nus
    ww = w * np.exp(lw - lw.max())
    return nus, ww / ww.sum() * math.exp(lm6)


def _schulz_spheroid_a2(p, sigma, tol=1e-10):
    """<A²>_ν(q, t) of the core-shell spheroid for a Schulz-Zimm distribution of ν (mean 1, width σ): analytic
    averages of the sphere block (series / closed forms of pysasfit.scatter.blocks, Wagener & Förster 2023)."""
    from pysasfit.scatter.blocks import BLOCK, cross3

    b, tt, a = p[0], p[1], p[2]
    cc = (p[3] - p[4]) * 4.0 / 3.0 * np.pi * a * b * b
    cs = (p[4] - p[5]) * 4.0 / 3.0 * np.pi * (a + tt) * (b + tt) ** 2
    z = (1.0 - sigma * sigma) / (sigma * sigma)
    m6 = math.exp(
        special.gammaln(z + 7.0) - special.gammaln(z + 1.0) - 6.0 * math.log(z + 1.0)
    )
    blk = BLOCK[3]

    def a2(q, t):
        with np.errstate(
            all="ignore"
        ):  # the series of blocks.cross3 overflow where they are not used
            return _a2(q, t)

    def _a2(q, t):
        t = np.asarray(t, float)
        c2 = t * t
        xc = q * np.sqrt(a * a * c2 + b * b * (1.0 - c2))
        xt = q * np.sqrt((a + tt) ** 2 * c2 + (b + tt) ** 2 * (1.0 - c2))
        out = cc * cc * blk.value(xc, z, 6, "P", tol) + cs * cs * blk.value(
            xt, z, 6, "P", tol
        )
        if cc != 0.0 and cs != 0.0:
            cross = np.empty(t.shape)
            for k in range(t.size):
                if xt.flat[k] == 0.0:
                    cross.flat[k] = 1.0
                else:
                    cross.flat[k] = cross3(
                        xc.flat[k] / xt.flat[k], 1.0, np.array([xt.flat[k]]), z, 6, tol
                    )[0]
            out = out + 2.0 * cc * cs * cross
        return m6 * out

    return a2


def _legendre_intensity(q, psi, p, shape, kind, accuracy, a2=None, size_scale=1.0):
    """Axially symmetric distributions: I(q, ψ) = Σ_l (2l+1)/2 p_l a_l(q) P_l(cos ψ) (Funk-Hecke), with
    a_l(q) = ∫_{-1}^{1} A²(q, t) P_l(t) dt and the orientation moments p_l. ``a2(q, t)`` replaces the
    monodisperse A² (size-averaged amplitudes); ``size_scale`` widens the angular bandwidth for it."""
    if a2 is None:
        amp = _AMP[shape]

        def a2(qi, t):
            return amp(qi, t, p, 1.0) ** 2

    d = _size(shape, p) * size_scale
    qf, pf = q.ravel(), psi.ravel()
    uq, inv = np.unique(qf, return_inverse=True)

    def lmax_of(qi):  # even, rounded up to a multiple of 16 (few distinct tables)
        return 16 * math.ceil(accuracy * (1.2 * qi * d + 40.0) / 16.0)

    lglob = lmax_of(float(uq.max()))
    upsi, pinv = np.unique(pf, return_inverse=True)
    gpsi = orientation_legendre(kind, p, lglob, upsi)  # g_l(ψ) of all distinct ψ, once
    out = np.empty(qf.size)
    order = np.argsort(inv, kind="stable")
    bounds = np.searchsorted(inv[order], np.arange(uq.size + 1))
    for i, qi in enumerate(uq):
        qi = float(qi)
        lm = lmax_of(qi)
        nt = 32 * math.ceil((accuracy * (0.7 * qi * d + 20.0) + lm // 2) / 32.0)
        t, wt, tab = _gl_legendre(lm, nt)  # [0, 1]; A² and the even P_l are even in t
        al = 2.0 * (tab @ (wt * a2(qi, t)))
        ls = np.arange(lm + 1)
        cl = (ls + 0.5) * al
        cl[1::2] = 0.0
        sel = order[bounds[i] : bounds[i + 1]]
        out[sel] = cl @ gpsi[: lm + 1, pinv[sel]]
    return out.reshape(q.shape)


# ------------------------------------------------------------------------------------------------ intensity
def sheared_intensity(
    q, p, psi=None, shape="cylinder", kind="maier_saupe", accuracy=1.0, backend=None, method=None,
    size_dist="lognormal",
):  # fmt: skip
    """I(q, ψ) of a sheared core-shell cylinder or spheroid. ``p`` as in SASfit (ψ = p[6] in degrees unless
    ``psi`` (radians) is given); ``accuracy`` scales the numbers of nodes.

    ``method``: "legendre" (Legendre expansion of A² in cos γ, addition theorem), "quadrature" (θ, φ double
    integral, as the C plugin) or None = :data:`METHOD` ("auto" = "legendre").
    ``backend`` (quadrature method): "numba" (compiled triple sum, parallel over the q points), "numpy", or
    None = :data:`BACKEND` ("auto": Numba when installed).
    ``size_dist``: "lognormal" (as the SASfit plugin) or "schulz" (Schulz-Zimm of the same mean and relative
    width σ; for spheroids with the Legendre method the size average is analytic, pysasfit.scatter.blocks)."""
    p = np.asarray(p, float)
    if p.size < 11:
        p = np.concatenate([p, np.zeros(11 - p.size)])
    q = np.asarray(q, float)
    if np.any(q < 0):
        raise ValueError("q >= 0 required")
    psi = np.radians(p[6]) if psi is None else np.asarray(psi, float)
    q, psi = np.broadcast_arrays(q, psi)
    sigma = p[7]
    if sigma < 0:
        raise ValueError("sigma >= 0 required")
    hp = kind == "hayterpenfold"
    method = (method or METHOD).lower()
    legendre = method in ("auto", "legendre")
    if size_dist not in ("lognormal", "schulz"):
        raise ValueError("size_dist must be 'lognormal' or 'schulz'")
    if sigma > 0 and size_dist == "schulz" and shape == "spheroid" and legendre:
        return _legendre_intensity(q, psi, p, shape, kind, accuracy, a2=_schulz_spheroid_a2(p, sigma),
                                   size_scale=1.0 + 3.0 * sigma)  # fmt: skip
    if sigma > 0:
        # All lengths scale with ν, so A(q, γ; ν) = ν³ A(qν, γ; 1) and the size average is a sum of monodisperse
        # intensities at the scaled q: I(q) = Σ_k w_k ν_k⁶ I₁(q ν_k). Each I₁ gets the angular nodes its own q ν_k
        # needs. The ν nodes resolve the oscillations of I₁(q ν) in ν (period ~ π/(q D)), so their number grows
        # with q (see _size_nodes_q). Quadrature method: nodes with a small weight w ν⁶ get fewer angular nodes
        # (they dominate the cost: largest q ν); the result changes by < ~2e-9 relative.
        p1 = p.copy()
        p1[7] = 0.0
        d1 = _size(shape, p)
        qf, pf = q.ravel(), psi.ravel()
        uq, inv = np.unique(qf, return_inverse=True)
        nodes = [_size_nodes_q(size_dist, sigma, float(qi) * d1, accuracy) for qi in uq]
        levels = (
            ((0.0, np.inf, 1.0),)
            if legendre
            else ((1e-3, np.inf, 1.0), (1e-6, 1e-3, 0.6), (0.0, 1e-6, 0.35))
        )
        out = np.zeros(qf.size)
        for lo, hi, factor in levels:
            qs, ps, ws, own = [], [], [], []
            for i, (nus, w6) in enumerate(nodes):
                rel = w6 / w6.max()
                sel = (rel > lo) & (rel <= hi)
                if not np.any(sel):
                    continue
                pts = np.nonzero(inv == i)[0]
                qs.append(
                    np.repeat(uq[i] * nus[sel][None, :], pts.size, axis=0).ravel()
                )
                ps.append(np.repeat(pf[pts], sel.sum()))
                ws.append(np.tile(w6[sel], pts.size))
                own.append(np.repeat(pts, sel.sum()))
            if not qs:
                continue
            i1 = sheared_intensity(np.concatenate(qs), p1, np.concatenate(ps), shape, kind, accuracy * factor,
                                   backend, "legendre" if legendre else "quadrature")  # fmt: skip
            out += np.bincount(
                np.concatenate(own), weights=i1 * np.concatenate(ws), minlength=qf.size
            )
        return out.reshape(q.shape)
    if legendre and hp:
        # beyond l ~ HP_LMAX (memory of the λ_l^m tables) the θ, φ quadrature
        big = accuracy * (1.2 * q * _size(shape, p) + 40.0) > HP_LMAX
        if np.any(big):
            out = np.empty(q.shape)
            if np.any(~big):
                out[~big] = _legendre_intensity(
                    q[~big], psi[~big], p, shape, kind, accuracy
                )
            out[big] = sheared_intensity(
                q[big], p, psi[big], shape, kind, accuracy, backend, "quadrature"
            )
            return out
    if legendre:
        return _legendre_intensity(q, psi, p, shape, kind, accuracy)
    nus, wnu = np.array([1.0]), np.array([1.0])
    numax = float(nus.max())
    d = _size(shape, p) * numax
    norm = 1.0 if hp else orientation_norm(kind, p)
    edges = _theta_edges(kind, p[8])
    w_hp, _ = _width(kind, p[8]) if hp else (None, None)

    def nodes(qi, psii):
        qd = qi * d
        # nodes: ~2 per oscillation of the kernel plus a floor; φ also resolves the Hayter-Penfold peak
        th, wth = _panels(
            edges, accuracy * (6.0 + 1.2 * qd), n_min=int(8 * accuracy) + 4
        )
        nphi = int(accuracy * (24 + 1.5 * qd * abs(math.sin(psii) if not hp else 1.0)))
        if hp and w_hp is not None:
            nphi = max(nphi, int(accuracy * 12.0 / w_hp))
        return th, wth, max(nphi, 8)

    backend = (backend or BACKEND).lower()
    if backend == "numba" and not _sheared_numba.AVAILABLE:
        raise RuntimeError("backend 'numba' requested but Numba is not installed")
    if backend in ("numba", "auto") and _sheared_numba.AVAILABLE and q.size:
        qf, pf = q.ravel(), psi.ravel()
        ths, wts, nph = [], [], np.empty(qf.size, np.int64)
        dens_cache = {}
        for i in range(qf.size):
            th, wth, nph[i] = nodes(float(qf[i]), float(pf[i]))
            w = wth * np.sin(th)
            if not hp:
                key = th.size, float(th[0]), float(th[-1])
                dn = dens_cache.get(key)
                if dn is None:
                    dn = dens_cache[key] = _shape(kind, th, p) / norm
                w = w * dn
            ths.append(th)
            wts.append(w)
        offs = np.zeros(qf.size + 1, np.int64)
        offs[1:] = np.cumsum([t.size for t in ths])
        out = _sheared_numba.triple_sum(shape, hp, qf, pf, np.concatenate(ths), np.concatenate(wts), offs, nph,
                                        nus, wnu, p, p[8])  # fmt: skip
        return out.reshape(q.shape)

    amp = _AMP[shape]
    out = np.empty(q.shape)
    for idx in np.ndindex(q.shape):
        qi, psii = float(q[idx]), float(psi[idx])
        th, wth, nphi = nodes(qi, psii)
        T = th[:, None, None]
        N = nus[None, None, :]
        if hp:
            phi = np.arange(nphi) * (np.pi / nphi)  # periodic rule on [0, π)
            wphi = np.full(nphi, np.pi / nphi)
            F = phi[None, :, None]
            base = np.sin(psii) * np.cos(T)
            side = np.cos(psii) * np.sin(T) * np.cos(F)
            a2 = amp(qi, base + side, p, N) ** 2 + amp(qi, base - side, p, N) ** 2
            dens = hayter_penfold(T, F, p[8])
            inner = np.einsum(
                "ijk,ijk,j,k->i", a2, np.broadcast_to(dens, a2.shape), wphi, wnu
            )
            val = 2.0 * np.sum(wth * np.sin(th) * inner)
        else:
            phi = np.linspace(
                0.0, np.pi, nphi + 1
            )  # trapezoid on [0, π] = periodic rule on [0, 2π)
            wphi = np.full(nphi + 1, np.pi / nphi)
            wphi[[0, -1]] *= 0.5
            F = phi[None, :, None]
            cosg = np.cos(psii) * np.cos(T) + np.sin(psii) * np.sin(T) * np.cos(F)
            a2 = amp(qi, cosg, p, N) ** 2
            inner = np.einsum("ijk,j,k->i", a2, wphi, wnu)  # ∫_0^π dφ
            dens = _shape(kind, th, p) / norm
            val = (
                2.0 * 2.0 * np.sum(wth * np.sin(th) * dens * inner)
            )  # hemispheres, φ ∈ [0, 2π)
        out[idx] = val
    return out


# ------------------------------------------------------------------------------------------------ registry
def _make(shape, kind, size_dist="lognormal"):
    def f(q, p, psi=None, c_compat=False):
        return sheared_intensity(q, p, psi, shape=shape, kind=kind, size_dist=size_dist)

    tag = "" if size_dist == "lognormal" else "_schulz_"
    f.__name__ = f"ff_sheared_{shape}s__{kind}_{tag}"
    f.__doc__ = (
        f"Sheared core-shell {shape}s, {kind.replace('_', '-')} orientation distribution"
        + (
            "."
            if size_dist == "lognormal"
            else ", Schulz-Zimm size distribution (mean 1, relative width sigma)."
        )
    )
    return f


def _qd(q, p):
    """q times the largest particle dimension (comparison report)."""
    return np.asarray(q, float) * max(abs(p[0]) + abs(p[1]), abs(p[2]))


_PARS_CYL = [
    "R",
    "t",
    "L",
    "eta_core",
    "eta_shell",
    "eta_solv",
    "psi",
    "sigma",
    "kappa",
    "dummy",
    "dummy",
]
_PARS_LEG = [
    "R",
    "t",
    "L",
    "eta_core",
    "eta_shell",
    "eta_solv",
    "psi",
    "sigma",
    "P2",
    "P4",
    "P6",
]
_PARS_SPH = [
    "R_equatorial",
    "t",
    "R_polar",
    "eta_core",
    "eta_shell",
    "eta_solv",
    "psi",
    "sigma",
    "kappa",
    "dummy",
    "dummy",
]
_KAPPA = {
    "maier_saupe": (-5.0, 10.0, "lin"),
    "gauss": (-3.0, 5.0, "lin"),
    "boltzmann": (-3.0, 5.0, "lin"),
    "onsager": (-5.0, 10.0, "lin"),
    "heavyside": (-3.0, 3.0, "lin"),
    "hayterpenfold": (0.2, 20.0, "log"),
}
_SAMPLE_BASE = {
    0: (5.0, 50.0, "log"),
    1: (1.0, 10.0, "log"),
    2: (20.0, 200.0, "log"),
    3: (1.0, 2.0, "lin"),
    4: (2.0, 3.0, "lin"),
    5: (0.0, 0.5, "lin"),
    6: (0.0, 90.0, "lin"),
}
_PRETTY = {
    "maier_saupe": "Maier-Saupe",
    "gauss": "Gauss",
    "boltzmann": "Boltzmann",
    "onsager": "Onsager",
    "heavyside": "Heavyside",
    "legendre": "Legendre",
    "hayterpenfold": "HayterPenfold",
}
_SRC = "plugins/sheared_objects/sasfit_"
_NOTE = (
    "C corrected on 2026-11-17 (older SASfit builds are wrong); compared with sigma = 0"
)


def _spec(shape, kind, size_dist="lognormal"):
    name = f"ff_sheared_{shape}s__{kind}_"
    sample = dict(_SAMPLE_BASE)
    if kind == "legendre":
        sample.update(
            {8: (0.0, 0.5, "lin"), 9: (0.0, 0.2, "lin"), 10: (0.0, 0.1, "lin")}
        )
        pars = _PARS_LEG
    else:
        sample[8] = _KAPPA[kind]
        pars = _PARS_SPH if shape == "spheroid" else _PARS_CYL
    if size_dist == "schulz":
        sample[7] = (0.0, 0.4, "lin")
        return {
            "name": name + "_schulz_",
            "intensity": _make(shape, kind, "schulz"),
            "parameters": pars,
            "psi_index": 6,
            "sample": sample,
            "rtol": 1e-6,
            "source": _SRC + name + ".c",
            "notes": "Python only: Schulz-Zimm size distribution (sigma = relative standard deviation, < 1); "
            + (
                "analytic size average (pysasfit.scatter.blocks)"
                if shape == "spheroid"
                else "size quadrature"
            ),
            "c_integrates": True,
            "c_reference": False,
            "label": f"Sheared {shape.capitalize()}s ({_PRETTY[kind]}, Schulz-Zimm)",
            "group": "ff_plugins_partly_aligned_cylsh"
            if shape == "cylinder"
            else "ff_plugins_partly_aligned_ellsh",
            "reduced": _qd,
            "reduced_label": "q D",
        }
    return {
        "name": name,
        "intensity": _make(shape, kind),
        "parameters": pars,
        "psi_index": 6,
        "sample": sample,
        "rtol": 1e-6,
        "source": _SRC + name + ".c",
        "notes": _NOTE
        if kind != "legendre"
        else "not exported by the C plugin (does not compile)",
        "c_integrates": True,
        "c_reference": True,
        "label": f"Sheared {shape.capitalize()}s ({_PRETTY[kind]})",
        "group": "ff_plugins_partly_aligned_cylsh"
        if shape == "cylinder"
        else "ff_plugins_partly_aligned_ellsh",
        "reduced": _qd,
        "reduced_label": "q D",
    }


SPECS = (
    [_spec("cylinder", k) for k in DISTRIBUTIONS]
    + [_spec("spheroid", k) for k in DISTRIBUTIONS if k != "legendre"]
    + [_spec("cylinder", k, "schulz") for k in DISTRIBUTIONS]
    + [_spec("spheroid", k, "schulz") for k in DISTRIBUTIONS if k != "legendre"]
)
