"""Python ports of the SASfit plugin helix (helical particles, orientation averaged).

The helix sums (Pringle & Schmidt) run over the layer lines n < qP/(2π) and multiply by the finite-rod factor
thinrod_helix(q, H) = H² (2 Si(qH)/(qH) - sinc²(qH/2)) (which replaces π/(qH) of the infinite helix):

* round_helix: two helical strands of circular cross-section (radii r_i, helix radii δ_i, phase α):
  Σ ε_n |β_1 J_n(δ_1 Q_n) Λ(r_1 Q_n) + β_2 J_n(δ_2 Q_n) Λ(r_2 Q_n) e^{inα}|², Q_n² = q² - (2πn/P)², Λ = 2J1(x)/x.
* beads_helix: n beads of radius R per pitch P on a helix of diameter D: (n/P V Δη Φ(qR))² Σ ε_j J_j²(qD/2 √(1-a_j²)).
* fan_helix: two blades (annular sectors aR..R of opening ω, separated by φ) wound with pitch P; the Bessel
  integrals ∫_{aR}^R r J_n(κ_n r) dr by Gauss-Legendre.
* superhelices_straight: thin helical line (radius R1, pitch P, height H) of contour length L:
  ∫_0^L 2 (L - w) sinc(q Ψ(w)) dw (Gauss-Legendre, panels of at most 2 rad).
* superhelices_coiled: thin helix (radius R1, N turns per turn) wound around a helix (radius R2, pitch P) for
  ``turns`` turns: the Debye double integral ∫∫ |r'(γ1)| |r'(γ2)| sinc(q |r(γ1) - r(γ2)|) dγ1 dγ2 on a
  Gauss-Legendre grid in γ from the explicit positions (Frenet frame of the big helix).

The C sums stop early when two successive terms change the sum by less than 1e-12 (or after 200 terms); the ports
sum all layer lines.

C fixes 2026-11-24: beads_helix and fan_helix carried an extra factor H² (I(0) = (Δη V H)² instead of (Δη V)²:
the prefactor used n H/P beads resp. the blade volume instead of the per-length values); round_helix dropped the
n = 0 term at q = 0 (I(0) = 0), fan_helix returned 0 at q = 0 and beads_helix NaN (0/0 in a_0); superhelices_coiled
used the pitch angle atan(R2/(2πP)) instead of atan(2πR2/P) (positions and |r'| inconsistent) and sin(γ1-γ2)
where the distance needs sin(N(γ1-γ2)).
"""

from __future__ import annotations

import math

import numpy as np
from scipy import special

from pysasfit.core.accel import HAVE_NUMBA

from .common import make_spec
from .quadrature import gauss_legendre

__all__ = ["SPECS"]

_GN = 16


def _sinc(x):
    return np.sinc(np.asarray(x, float) / np.pi)


def thinrod_helix(q, H):
    """H² (2 Si(qH)/(qH) - sinc²(qH/2)); H² at q = 0."""
    u = np.asarray(q, float) * H
    small = np.abs(u) < 1e-3
    us = np.where(small, 1.0, u)
    si = special.sici(us)[0]
    big = 2.0 * si / us - _sinc(0.5 * us) ** 2
    ser = 1.0 - u * u / 36.0 + u**4 / 1800.0
    return H * H * np.where(small, ser, big)


def _lam(x):
    x = np.asarray(x, float)
    small = np.abs(x) < 1e-4
    xs = np.where(small, 1.0, x)
    return np.where(small, 1.0 - x * x / 8.0, 2.0 * special.j1(xs) / xs)


def _phi(u):
    u = np.asarray(u, float)
    small = np.abs(u) < 1e-3
    us = np.where(small, 1.0, u)
    return np.where(small, 1.0 - 0.1 * u * u + u**4 / 280.0, 3.0 * (np.sin(us) - us * np.cos(us)) / us**3)


def _layer_lines(qi, P):
    """n = 0, 1, ... with 2πn/P < q (n = 0 always)."""
    nmax = math.ceil(qi * P / (2.0 * math.pi)) - 1 if qi > 0 else 0
    return np.arange(0, max(nmax, 0) + 1)


def _eps(n):
    return np.where(n == 0, 1.0, 2.0)


def _with_x(p, x, dist):
    p = np.array(p, float)
    if 0 <= dist < p.size:
        p[dist] = float(np.asarray(x, float).reshape(-1)[0]) if np.ndim(x) else float(x)
    return p


# ---------------------------------------------------------------------------------------------- round helix
def round_helix(q, p):
    """p = [r1, delta_1, r2, delta_2, alpha (deg), P, H, eta_1, eta_2, eta_solv]."""
    r1, d1, r2, d2, al, P, H, e1, e2, es = p[:10]
    if min(r1, r2, d1, d2, H, P) < 0:
        raise ValueError("round_helix: r_i, delta_i, P, H >= 0 required")
    q = np.atleast_1d(np.asarray(q, float))
    b1, b2 = math.pi * r1 * r1 * (e1 - es), math.pi * r2 * r2 * (e2 - es)
    al = al * math.pi / 180.0
    out = np.empty(q.shape)
    for i, qi in enumerate(q):
        n = _layer_lines(qi, P) if P > 0 else np.zeros(1, int)
        qp = np.sqrt(np.abs(qi * qi - (2.0 * math.pi * n / P) ** 2)) if P > 0 else np.full(1, qi)
        a1 = special.jv(n, d1 * qp) * _lam(r1 * qp) * b1
        a2 = special.jv(n, d2 * qp) * _lam(r2 * qp) * b2
        out[i] = np.sum(_eps(n) * (a1 * a1 + a2 * a2 + 2.0 * a1 * a2 * np.cos(n * al)))
    return (thinrod_helix(q, H) * out).reshape(np.shape(np.asarray(q)))


def round_helix_v(x, p, dist):
    p = _with_x(p, x, dist)
    return (p[0] ** 2 + p[2] ** 2) * math.pi * p[6] + 0.0 * np.asarray(x, float)


# ---------------------------------------------------------------------------------------------- beads helix
def beads_helix(q, p):
    """p = [R, D, n, dummy, dummy, P, H, eta_b, dummy, eta_solv]."""
    R, D, N, P, H, eb, es = p[0], p[1], p[2], p[5], p[6], p[7], p[9]
    if min(R, D, N, P, H) < 0 or P == 0:
        raise ValueError("beads_helix: R, D, n, H >= 0 and P > 0 required")
    q = np.atleast_1d(np.asarray(q, float))
    V = 4.0 / 3.0 * math.pi * R**3
    out = np.empty(q.shape)
    for i, qi in enumerate(q):
        j = _layer_lines(qi, P)
        a = np.where(j == 0, 0.0, 2.0 * math.pi * j / (P * max(qi, 1e-300)))
        ok = a * a < 1.0
        psi = np.where(ok, special.jv(j, 0.5 * qi * D * np.sqrt(np.where(ok, 1.0 - a * a, 0.0))) ** 2, 0.0)
        out[i] = np.sum(_eps(j) * psi)
    pref = (N / P * V * (eb - es) * _phi(q * R)) ** 2
    return (thinrod_helix(q, H) * pref * out).reshape(np.shape(np.asarray(q)))


def beads_helix_v(x, p, dist):
    p = _with_x(p, x, dist)
    return 4.0 / 3.0 * math.pi * p[0] ** 3 * p[2] * p[6] / p[5] + 0.0 * np.asarray(x, float)


# ---------------------------------------------------------------------------------------------- fan helix
def fan_helix(q, p):
    """p = [R, a, omega (deg), phi (deg), dummy, P, H, eta_h, dummy, eta_solv] (0 <= a < 1)."""
    R, a, om, ph, P, H, eh, es = p[0], p[1], p[2], p[3], p[5], p[6], p[7], p[9]
    if R < 0 or not 0 <= a < 1 or P <= 0 or H < 0:
        raise ValueError("fan_helix: R >= 0, 0 <= a < 1, P > 0, H >= 0 required")
    om, ph = om * math.pi / 180.0, ph * math.pi / 180.0
    q = np.atleast_1d(np.asarray(q, float))
    area = R * R * (1.0 - a * a)
    out = np.empty(q.shape)
    for i, qi in enumerate(q):
        n = _layer_lines(qi, P)
        kap = qi * np.sqrt(np.abs(1.0 - (2.0 * math.pi * n / (P * qi)) ** 2)) if qi > 0 else np.zeros(1)
        npan = max(1, math.ceil(qi * R * (1.0 - a) / 2.0))
        e = np.linspace(a * R, R, npan + 1)
        r, w = gauss_legendre(_GN, e[:-1], e[1:])
        r, w = r.ravel(), w.ravel()
        g = special.jv(n[:, None], kap[:, None] * r[None, :]) @ (w * r)
        pre = np.where(n == 0, 1.0, np.cos(0.5 * n * ph) ** 2 * _sinc(0.5 * n * om) ** 2)
        out[i] = np.sum(_eps(n) * pre * g * g) * (2.0 / area) ** 2
    return (thinrod_helix(q, H) * out * (area * om * (eh - es)) ** 2).reshape(np.shape(np.asarray(q)))


def fan_helix_v(x, p, dist):
    p = _with_x(p, x, dist)
    return p[0] ** 2 * (1.0 - p[1] ** 2) * p[2] * math.pi / 180.0 * p[6] + 0.0 * np.asarray(x, float)


# ---------------------------------------------------------------------------------------------- superhelices
def superhelices_straight(q, p):
    """p = [R1, dummy x4, P, H, dummy x3]: thin helical line, I(0) = L²."""
    R1, P, H = p[0], p[5], p[6]
    if R1 < 0 or H < 0 or P <= 0:
        raise ValueError("superhelices_straight: R1 >= 0, H >= 0, P > 0 required")
    A = P / (2.0 * math.pi)
    s = math.hypot(R1, A)
    L = H / A * s
    q = np.atleast_1d(np.asarray(q, float))
    out = np.empty(q.shape)
    for i, qi in enumerate(q):
        npan = max(4, math.ceil(qi * L / 2.0))
        e = np.linspace(0.0, L, npan + 1)
        w_, wt = gauss_legendre(_GN, e[:-1], e[1:])
        w_, wt = w_.ravel(), wt.ravel()
        th = w_ / s
        psi = np.sqrt(4.0 * R1 * R1 * np.sin(0.5 * th) ** 2 + (A * th) ** 2)
        out[i] = np.sum(wt * 2.0 * (L - w_) * _sinc(qi * psi))
    return out.reshape(np.shape(np.asarray(q)))


def superhelices_straight_v(x, p, dist):
    p = _with_x(p, x, dist)
    A = p[5] / (2.0 * math.pi)
    return p[6] / A * math.hypot(p[0], A) + 0.0 * np.asarray(x, float)


def _coiled_geometry(p):
    R1, R2, N, P, T = p[0], p[1], p[3], p[5], p[6]
    if min(R1, R2, N, T) < 0 or P < 0:
        raise ValueError("superhelices_coiled: R1, R2, N, P, turns >= 0 required")
    A = P / (2.0 * math.pi)
    s = math.hypot(R2, A)
    ca, sa = (A / s, R2 / s) if s > 0 else (1.0, 0.0)
    return R1, R2, N, A, ca, sa, 2.0 * math.pi * T


def _coiled_nodes(p, h):
    """Gauss-Legendre nodes γ (panels at most h), weights · |r'|, positions."""
    R1, R2, N, A, ca, sa, G = _coiled_geometry(p)
    npan = max(1, math.ceil(G / h))
    e = np.linspace(0.0, G, npan + 1)
    g, w = gauss_legendre(_GN, e[:-1], e[1:])
    g, w = g.ravel(), w.ravel()
    c, s = np.cos(N * g), np.sin(N * g)
    f = (R2 * R2 + A * A + R1 * R1 * (c * c + (s * ca) ** 2) + (R1 * N) ** 2 + 2.0 * R1 * R1 * N * ca
         + 2.0 * R2 * R1 * c * (1.0 + N * ca) - 2.0 * A * R1 * N * sa * c)  # fmt: skip
    speed = np.sqrt(np.abs(f))
    a_, b_ = R2 + R1 * c, R1 * ca * s
    xyz = np.stack([a_ * np.cos(g) - b_ * np.sin(g), a_ * np.sin(g) + b_ * np.cos(g), A * g - R1 * sa * s], axis=1)
    return w * speed, xyz, speed


def superhelices_coiled_v(x, p, dist):
    """Contour length ∫ |r'(γ)| dγ."""
    p = _with_x(p, x, dist)
    N = p[3]
    w, _, _ = _coiled_nodes(p, min(0.5, 0.5 / max(N, 1e-300)))
    return float(np.sum(w)) + 0.0 * np.asarray(x, float)


if HAVE_NUMBA:
    import numba

    @numba.njit(parallel=True, cache=True)
    def _debye(w, xyz, q):  # pragma: no cover - compiled
        n = w.size
        acc = np.zeros(n)
        for i in numba.prange(n):
            s = 0.0
            for j in range(i + 1, n):
                d = math.sqrt((xyz[i, 0] - xyz[j, 0]) ** 2 + (xyz[i, 1] - xyz[j, 1]) ** 2
                              + (xyz[i, 2] - xyz[j, 2]) ** 2)  # fmt: skip
                x = q * d
                s += w[j] * (math.sin(x) / x if x > 1e-8 else 1.0 - x * x / 6.0)
            acc[i] = w[i] * (w[i] + 2.0 * s)
        return acc.sum()

else:

    def _debye(w, xyz, q):
        tot = 0.0
        for i0 in range(0, w.size, 512):
            d = np.sqrt(np.sum((xyz[i0 : i0 + 512, None, :] - xyz[None, :, :]) ** 2, axis=-1))
            tot += w[i0 : i0 + 512] @ (_sinc(q * d) @ w)
        return tot


def superhelices_coiled(q, p):
    """p = [R1, R2, dummy, N, dummy, P, turns, dummy x3]: Debye double integral of the coiled thin helix."""
    q = np.atleast_1d(np.asarray(q, float))
    N = p[3]
    out = np.empty(q.shape)
    for i, qi in enumerate(q):
        _, _, speed = _coiled_nodes(p, 0.5)
        h = min(0.5, 0.5 / max(N, 1e-300), 2.0 / max(qi * float(speed.max()), 1e-300))
        w, xyz, _ = _coiled_nodes(p, h)
        out[i] = _debye(w, xyz, float(qi))
    return out.reshape(np.shape(np.asarray(q)))


# ---------------------------------------------------------------------------------------------- specs
_SRC = "plugins/helix/sasfit_ff_"
_G = "ff_plugins_helix"
_RH = ["r1", "delta_1", "r2", "delta_2", "alpha", "P", "H", "eta_1", "eta_2", "eta_solv"]
_RHS = {0: (2.0, 8.0, "lin"), 1: (5.0, 20.0, "lin"), 2: (2.0, 8.0, "lin"), 3: (5.0, 20.0, "lin"), 4: (0.0, 180.0, "lin"),
        5: (20.0, 80.0, "lin"), 6: (100.0, 500.0, "lin"), 7: (0.5, 2.0, "lin"), 8: (0.5, 2.0, "lin"),
        9: (0.0, 0.3, "lin")}  # fmt: skip
_BH = ["R", "D", "n", "dummy", "dummy", "P", "H", "eta_b", "dummy", "eta_solv"]
_BHS = {0: (2.0, 6.0, "lin"), 1: (10.0, 40.0, "lin"), 2: (5.0, 15.0, "lin"), 5: (20.0, 80.0, "lin"),
        6: (100.0, 500.0, "lin"), 7: (0.5, 2.0, "lin"), 9: (0.0, 0.3, "lin")}  # fmt: skip
_FH = ["R", "a", "omega", "phi", "dummy", "P", "H", "eta_h", "dummy", "eta_solv"]
_FHS = {0: (10.0, 40.0, "lin"), 1: (0.0, 0.6, "lin"), 2: (20.0, 120.0, "lin"), 3: (0.0, 180.0, "lin"),
        5: (20.0, 80.0, "lin"), 6: (100.0, 500.0, "lin"), 7: (0.5, 2.0, "lin"), 9: (0.0, 0.3, "lin")}  # fmt: skip
_SS = ["R1", "dummy", "dummy", "dummy", "dummy", "P", "H", "dummy", "dummy", "dummy"]
_SSS = {0: (5.0, 30.0, "lin"), 5: (20.0, 100.0, "lin"), 6: (100.0, 500.0, "lin")}
_SC = ["R1", "R2", "dummy", "N", "dummy", "P", "turns", "dummy", "dummy", "dummy"]
_SCS = {0: (2.0, 6.0, "lin"), 1: (10.0, 30.0, "lin"), 3: (2.0, 6.0, "lin"), 5: (30.0, 100.0, "lin"),
        6: (1.0, 3.0, "lin")}  # fmt: skip


def _qr(i):
    return lambda q, p: np.asarray(q, float) * abs(p[i])


SPECS = [
    make_spec("ff_round_helix", round_helix, _RH, _RHS, volume=round_helix_v, source=_SRC + "round_helix.c",
              reduced=_qr(6), reduced_label="q H", rtol=1e-9, group=_G),
    make_spec("ff_beads_helix", beads_helix, _BH, _BHS, volume=beads_helix_v, source=_SRC + "beads_helix.c",
              reduced=_qr(6), reduced_label="q H", rtol=1e-9, group=_G),
    make_spec("ff_fan_helix", fan_helix, _FH, _FHS, volume=fan_helix_v, source=_SRC + "fan_helix.c",
              reduced=_qr(6), reduced_label="q H", rtol=1e-8, c_integrates=True, group=_G),
    make_spec("ff_superhelices_straight", superhelices_straight, _SS, _SSS, volume=superhelices_straight_v,
              source=_SRC + "superhelices_straight.c", reduced=_qr(6), reduced_label="q H", rtol=1e-8,
              c_integrates=True, group=_G),
    make_spec("ff_superhelices_coiled", superhelices_coiled, _SC, _SCS, volume=superhelices_coiled_v,
              source=_SRC + "superhelices_coiled.c", reduced=_qr(1), reduced_label="q R2", rtol=1e-6,
              c_integrates=True, group=_G),
]  # fmt: skip
