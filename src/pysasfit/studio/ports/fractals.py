"""Python ports of the SASfit plugins fractal_obj (form factors), fractal_series and powerlaw.

Integrals that the C code does with adaptive quadrature (to infinity, or with power-law extrapolation beyond
q ξ = 100 / 50) are done here with Gauss-Legendre / Gauss-Jacobi rules that resolve the oscillations, for all q.

C deviations (fixed in the SASfit sources on 2026-11-18 where marked):
* two_attached_spheres_v: 4/.3 instead of 4/3 (volume 10x too large) — fixed.
* two_attached_spheres_w_distr_f/_v: the second log-normal used the median of R1 instead of R2 — fixed.
* doubleshellchain: for non-integer n the second term summed only the last row of the (n+1)² pair matrix — fixed
  (now (1-w) S(n) + w S(n+1) with the full sums).
* massfractgauss: e^{-x} 1F1((3-D)/2; 3/2; x) overflowed for large x (error branch); the port (and the fixed C) use
  Kummer's transformation 1F1(D/2; 3/2; -x).
* aggregate (exp(-x^a) / overlapping spheres cut-off): beyond q ξ = 100 (50) C extrapolates a power law from that
  point; the port integrates (c_valid excludes those q in the comparison).
* fractal_series: the closed forms (special cases fd = 1..8, series, 2F1) are replaced by the integral
  ∫ SD(x) x⁶/(1+q²x²)² dx, computed directly.
"""

from __future__ import annotations

import math
from functools import lru_cache

import numpy as np
from scipy import special

from .common import make_spec, phi_sphere

__all__ = ["SPECS"]


@lru_cache(maxsize=128)
def _gl(n):
    return np.polynomial.legendre.leggauss(n)


@lru_cache(maxsize=128)
def _gj(n, beta):
    """Gauss-Jacobi nodes/weights on [0, 1] for the weight t^beta."""
    x, w = special.roots_jacobi(n, 0.0, beta)  # weight (1-x)^0 (1+x)^beta on [-1, 1]
    return 0.5 * (x + 1.0), w / 2.0 ** (beta + 1.0)


def _sinc(x):
    return np.sinc(np.asarray(x, float) / np.pi)


def sphere_f(q, r, deta):
    return deta * 4.0 / 3.0 * np.pi * r**3 * phi_sphere(np.asarray(q, float) * r)


# ------------------------------------------------------------------------------------------- simple fractals
def fisher_burford(q, p):
    """p = [Rg, D, I0]: I0 (1 + 2/(3D) q²Rg²)^(-D/2)."""
    q = np.asarray(q, float)
    rg, d, i0 = p[0], p[1], p[2]
    if rg < 0 or d <= 0:
        raise ValueError("Rg >= 0 and D > 0 required")
    return i0 * np.power(1.0 + 2.0 / (3.0 * d) * (q * rg) ** 2, -0.5 * d)


def massfractexp(q, p):
    """p = [Rg, D, I0]: I0 sin((D-1) atan(qξ)) / ((D-1) qξ (1+q²ξ²)^((D-1)/2)), ξ² = 2Rg²/(D(D+1))."""
    q = np.asarray(q, float)
    rg, d, i0 = p[0], p[1], p[2]
    if d <= 1 or rg < 0:
        raise ValueError("D > 1 and Rg >= 0 required")
    x = q * math.sqrt(2.0 * rg * rg / (d * (d + 1.0)))
    with np.errstate(invalid="ignore", divide="ignore"):
        v = np.sin((d - 1.0) * np.arctan(x)) / (
            (d - 1.0) * x * np.power(1.0 + x * x, 0.5 * (d - 1.0))
        )
    return i0 * np.where(x == 0, 1.0, v)


def massfractgauss(q, p):
    """p = [Rg, D, I0]: I0 e^{-q²Rg²/D} 1F1((3-D)/2; 3/2; q²Rg²/D) = I0 1F1(D/2; 3/2; -q²Rg²/D) (Kummer)."""
    q = np.asarray(q, float)
    rg, d, i0 = p[0], p[1], p[2]
    if d <= 1:
        raise ValueError("D > 1 required")
    if rg == 0:
        return np.full(q.shape, i0)
    x = (q * rg) ** 2 / d
    return i0 * special.hyp1f1(0.5 * d, 1.5, -x)


def _cluster_poly(q, p, c2, c3, c4):
    q = np.asarray(q, float)
    rg, d, i0 = p[0], p[1], p[2]
    if rg < 0 or d <= 0:
        raise ValueError("Rg >= 0 and D > 0 required")
    x2 = (q * rg) ** 2
    s = 1.0 + x2 * (8.0 / 3.0 / d + x2 * (c2 + x2 * (c3 + x2 * c4)))
    return i0 * np.power(s, -d / 8.0)


def dlcaggregation(q, p):
    """p = [Rg, D, I0]: I0 (Σ C_i (qRg)^(2i))^(-D/8), C = 1, 8/(3D), 2.5, -1.52, 1.02 (DLCA)."""
    return _cluster_poly(q, p, 2.5, -1.52, 1.02)


def rlcaggregation(q, p):
    """p = [Rg, D, I0]: as DLCA with C = 1, 8/(3D), 3.13, -2.58, 0.95 (RLCA)."""
    return _cluster_poly(q, p, 3.13, -2.58, 0.95)


# ------------------------------------------------------------------------------------------- cut-off aggregates
def _oscillatory_integral(q, f, xmax, beta, hmax=np.inf, grade=False):
    """∫_0^xmax x^beta f(x) sin(qx)/(qx) dx for each q (Gauss-Jacobi on the first panel for the x^beta end point,
    Gauss-Legendre panels resolving sin(qx) beyond; panels not wider than ``hmax`` (scale of f))."""
    out = np.empty(q.shape)
    for i, qi in np.ndenumerate(q):
        n_osc = qi * xmax / np.pi
        m = max(
            1, math.ceil(n_osc / 4.0), math.ceil(xmax / hmax)
        )  # panels with <= ~4 oscillations
        h = xmax / m
        if grade:  # f has non-analytic powers at 0 (e^{-x^α}): panels h 2^-k ... h, Gauss-Jacobi below
            h0 = h * 2.0**-40
            t, w = _gj(24, beta)
            x0 = h0 * t
            s = np.sum(w * h0 ** (beta + 1.0) * f(x0) * _sinc(qi * x0))
            xg, wg = _gl(max(24, 3 * math.ceil(n_osc / m) + 24))
            lo = h * 2.0 ** -np.arange(40, 0, -1.0)
            hi = 2.0 * lo
            xx = lo[:, None] + 0.5 * (hi - lo)[:, None] * (xg[None, :] + 1.0)
            s += np.sum(
                0.5
                * (hi - lo)[:, None]
                * wg[None, :]
                * xx**beta
                * f(xx)
                * _sinc(qi * xx)
            )
        else:
            t, w = _gj(max(24, 3 * math.ceil(n_osc / m) + 24), beta)
            x0 = h * t
            s = np.sum(w * h ** (beta + 1.0) * f(x0) * _sinc(qi * x0))
        if m > 1:
            xg, wg = _gl(3 * math.ceil(n_osc / m) + 24)
            a = h * np.arange(1, m)[:, None]
            xx = a + 0.5 * h * (xg[None, :] + 1.0)
            s += np.sum(0.5 * h * wg[None, :] * xx**beta * f(xx) * _sinc(qi * xx))
        out[i] = s
    return out


def aggregate_exp_cutoff(q, p):
    """p = [xi, D, I0, alpha]: I0 ∫_0^∞ D x^(D-1) e^{-(x/ξ)^α} sin(qx)/(qx) dx / (ξ^D Γ(D/α) D/α)."""
    q = np.asarray(q, float)
    xi, d, i0, alpha = p[0], p[1], p[2], p[3]
    if xi <= 0 or d >= 3 or d <= 1 or alpha <= 0:
        raise ValueError("xi > 0, 1 < D < 3 and alpha > 0 required")
    xmax = xi * 40.0 ** (1.0 / alpha)  # e^{-(x/ξ)^α} < e^{-40}
    hmax = (
        2.0 * xi / max(1.0, alpha * 40.0 ** ((alpha - 1.0) / alpha))
    )  # e^{-(x/ξ)^α} changes by <~ e^2 per panel
    res = _oscillatory_integral(q, lambda x: d * np.exp(-((x / xi) ** alpha)), xmax, d - 1.0, hmax,
                                grade=alpha != round(alpha))  # fmt: skip
    norm = xi**d * math.exp(special.gammaln(d / alpha) + math.log(d / alpha))
    return i0 * res / norm


def aggregate_overlapsph_cutoff(q, p):
    """p = [Rg, D, I0]: overlapping-sphere cut-off, ξ² = (D+2)(D+5)/(2D(D+1)) Rg²; ∫_0^{2ξ} V(x) x^(D-1) sinc."""
    q = np.asarray(q, float)
    rg, d, i0 = p[0], p[1], p[2]
    if rg < 0 or d <= 0:
        raise ValueError("Rg >= 0 and D > 0 required")
    xi = math.sqrt((d + 2.0) * (d + 5.0) / (2.0 * d * (d + 1.0)) * rg * rg)

    def f(x):
        return (
            4.0
            / 3.0
            * np.pi
            * xi**3
            * (1.0 + x / (4.0 * xi))
            * (1.0 - x / (2.0 * xi)) ** 2
        )

    res = _oscillatory_integral(q, f, 2.0 * xi, d - 1.0)
    norm = 2.0 ** (2.0 + d) * np.pi * xi ** (3.0 + d) / (d * (3.0 + 4.0 * d + d * d))
    return i0 * res / norm


# ------------------------------------------------------------------------------------------- composite particles
def dumbbellshell(q, p):
    """p = [R, dR, L, eta_c, eta_sh]: two core-shell spheres, centre distance 2(R+dR)+L: 2A²(1 + sinc(q R12))."""
    q = np.asarray(q, float)
    r, dr, ll, ec, esh = p[:5]
    a = sphere_f(q, r + dr, esh) - sphere_f(q, r, esh - ec)
    r12 = 2.0 * (r + dr) + ll
    return a * a * 2.0 * (1.0 + _sinc(q * r12))


def two_attached_spheres(q, p):
    """p = [R1, dummy, R2, dummy, eta_0, eta_1, eta_2]: f1² + f2² + 2 f1 f2 sin(q(R1+R2))/(q(R1+R2))."""
    q = np.asarray(q, float)
    r1, r2, e0, e1, e2 = p[0], p[2], p[4], p[5], p[6]
    if r1 < 0 or r2 < 0:
        raise ValueError("R1, R2 >= 0 required")
    f1, f2 = sphere_f(q, r1, e1 - e0), sphere_f(q, r2, e2 - e0)
    return f1 * f1 + f2 * f2 + 2.0 * f1 * f2 * _sinc(q * (r1 + r2))


def two_attached_spheres_f(q, p):
    q = np.asarray(q, float)
    r1, r2, e0, e1, e2 = p[0], p[2], p[4], p[5], p[6]
    vt = (e1 - e0) * r1**3 + (e2 - e0) * r2**3
    if vt == 0:
        return np.zeros(q.shape)
    r0 = (
        -(e1 - e0) * r1**4 + (e2 - e0) * r2**4
    ) / vt  # scattering-length centre (from the contact point)
    return sphere_f(q, r1, e1 - e0) * _sinc(q * (r1 + r0)) + sphere_f(
        q, r2, e2 - e0
    ) * _sinc(q * (r2 - r0))


def two_attached_spheres_v(x, p, dist):
    return 4.0 / 3.0 * np.pi * (p[0] ** 3 + p[2] ** 3) + 0.0 * np.asarray(x, float)


def _lognorm_nodes(mu, s, qmax):
    """Nodes/weights in R of the normalised log-normal (median mu, width s), resolving oscillations q R."""
    lo, hi = mu * math.exp(-8.0 * s), mu * math.exp(8.0 * s)
    n = math.ceil(3.0 * qmax * 2.0 * (hi - lo) / np.pi + 48.0)
    m = max(6, math.ceil(n / 48))
    x, w = _gl(math.ceil(n / m))
    h = (math.log(hi) - math.log(lo)) / m
    u = (math.log(lo) + h * (np.arange(m)[:, None] + 0.5 * (x[None, :] + 1.0))).ravel()
    ww = (
        np.tile(0.5 * h * w, m)
        * np.exp(-0.5 * ((u - math.log(mu)) / s) ** 2)
        / (math.sqrt(2.0 * np.pi) * s)
    )
    return np.exp(u), ww


def _two_spheres_distr(q, p, kind):
    q = np.asarray(q, float)
    r1, s1, r2, s2 = p[0], abs(p[1]), p[2], abs(p[3])
    if r1 < 0 or r2 < 0 or s1 == 0 or s2 == 0:
        raise ValueError("R1, R2 >= 0 and sigma1, sigma2 != 0 required")
    out = np.empty(q.shape)
    qm = float(np.max(q, initial=0.0))
    x1, w1 = _lognorm_nodes(r1, s1, qm)
    x2, w2 = _lognorm_nodes(r2, s2, qm)
    fun = {"I": two_attached_spheres, "F": two_attached_spheres_f}[kind]
    for i, qi in np.ndenumerate(q):
        v = _broadcast_eval(fun, qi, p, x1, x2)
        out[i] = w1 @ v @ w2
    return out


def _broadcast_eval(fun, qi, p, x1, x2):
    e0, e1, e2 = p[4], p[5], p[6]
    r1, r2 = x1[:, None], x2[None, :]
    f1 = (e1 - e0) * 4.0 / 3.0 * np.pi * r1**3 * phi_sphere(qi * r1)
    f2 = (e2 - e0) * 4.0 / 3.0 * np.pi * r2**3 * phi_sphere(qi * r2)
    if fun is two_attached_spheres:
        return f1 * f1 + f2 * f2 + 2.0 * f1 * f2 * _sinc(qi * (r1 + r2))
    vt = (e1 - e0) * r1**3 + (e2 - e0) * r2**3
    with np.errstate(invalid="ignore", divide="ignore"):
        r0 = (-(e1 - e0) * r1**4 + (e2 - e0) * r2**4) / vt
        v = f1 * _sinc(qi * (r1 + r0)) + f2 * _sinc(qi * (r2 - r0))
    return np.where(vt == 0, 0.0, v)


def two_attached_spheres_w_distr(q, p):
    """p = [R1, sigma1, R2, sigma2, eta_0, eta_1, eta_2]: two_attached_spheres averaged over independent log-normal
    distributions of R1 and R2 (medians R1, R2)."""
    return _two_spheres_distr(q, p, "I")


def two_attached_spheres_w_distr_f(q, p):
    return _two_spheres_distr(q, p, "F")


def two_attached_spheres_w_distr_v(x, p, dist):
    r1, s1, r2, s2 = p[0], abs(p[1]), p[2], abs(p[3])
    # <R³> of a log-normal with median mu: mu³ e^{9 s²/2}
    return 4.0 / 3.0 * np.pi * (
        r1**3 * math.exp(4.5 * s1 * s1) + r2**3 * math.exp(4.5 * s2 * s2)
    ) + 0.0 * np.asarray(x, float)


def _shell_amp(q, p):
    rc, dr1, dr2 = p[0], p[1], p[2]
    ec, e1, e2, es, xs = p[4], p[5], p[6], p[7], p[8]
    d2 = (1.0 - xs) * (e2 - es)
    d1 = e1 - es
    dc = ec - es
    return (
        sphere_f(q, rc + dr1 + dr2, d2)
        + sphere_f(q, rc + dr1, d1 - d2)
        + sphere_f(q, rc, dc - d1)
    )


def _check_shell(p):
    ec, e1, e2, es, xs = p[4:9]
    if abs(xs - 0.5) > 0.5:
        raise ValueError("0 <= x_sol_sh2 <= 1 required")
    if abs(ec - es) + abs(e1 - es) + abs((1.0 - xs) * (e2 - es)) == 0:
        raise ValueError("all contrasts are 0")


def doubleshellchain(q, p):
    """p = [R_c, DR1, DR2, L, eta_c, eta_1, eta_2, eta_sol, x_sol_sh2, n]: linear chain of n double-shell
    spheres (centre distance 2(R_c+DR1+DR2)+L); non-integer n interpolates between floor(n) and floor(n)+1."""
    q = np.asarray(q, float)
    n = p[9]
    if n < 0:
        raise ValueError("n >= 0 required")
    if 2.0 * (p[0] + p[1] + p[2]) + p[3] < 0:
        raise ValueError("2(R_c+DR1+DR2)+L >= 0 required")
    _check_shell(p)
    if n == 0:
        return np.zeros(q.shape)
    a2 = _shell_amp(q, p) ** 2
    d = 2.0 * (p[0] + p[1] + p[2]) + p[3]

    def s(nn):
        out = np.full(q.shape, float(nn))
        for k in range(1, nn):
            out += 2.0 * (nn - k) * _sinc(q * k * d)
        return out

    nn = math.floor(n)
    w = n - nn
    res = (1.0 - w) * s(nn)
    if w != 0:
        res = res + w * s(nn + 1)
    return a2 * res


def tetrahedrondoubleshell(q, p):
    """p = [R_c, DR1, DR2, L, eta_c, eta_1, eta_2, eta_sol, x_sol_sh2, n]: up to 5 double-shell spheres at
    distance R12 = 2(R_c+DR1+DR2)+L; sphere 5 on the face opposite sphere 4, so the pair (4, 5) is √(8/3) R12 apart
    (two face-sharing tetrahedra; C used √(6/3) until 2026-11-18, which is not realisable in 3D)."""
    q = np.asarray(q, float)
    n = p[9]
    r12 = 2.0 * (p[0] + p[1] + p[2]) + p[3]
    if n > 5 or n <= 0:
        raise ValueError("0 < n <= 5 required")
    if r12 <= 0:
        raise ValueError("R12 > 0 required")
    _check_shell(p)
    a2 = _shell_amp(q, p) ** 2
    rr = np.full((5, 5), r12)
    np.fill_diagonal(rr, 0.0)
    rr[3, 4] = rr[4, 3] = math.sqrt(8.0 / 3.0) * r12
    nn = math.floor(n)
    w = n - nn

    def s(m):
        m = min(m, 5)
        out = np.zeros(q.shape)
        for i in range(m):
            for j in range(m):
                out += _sinc(q * rr[i, j])
        return out

    return a2 * ((1.0 - w) * s(nn) + w * s(nn + 1))


def stackdiscs(q, p):
    """p = [R, t, h, n, eta_c, eta_l, sigma, D]: stack of n discs (core t, layers h, radius R), spacing D with
    Gaussian disorder σ; orientation average over the stack axis."""
    q = np.asarray(q, float)
    r, t, h, n, ec, el, sigma, d = p[0], p[1], p[2], p[3], p[4], p[5], abs(p[6]), p[7]
    if r < 0 or t < 0 or h < 0 or n < 1 or sigma == 0 or d < 0:
        raise ValueError("R, t, h, D >= 0, n >= 1 and sigma != 0 required")
    vt, vc = np.pi * r * r * (t + 2.0 * h), np.pi * r * r * t
    nn = math.floor(n)
    wn = n - nn
    out = np.empty(q.shape)
    for i, qi in np.ndenumerate(q):
        nodes = math.ceil(1.2 * qi * max(r, t + 2 * h, (nn + 1) * d) + 48)
        x, w = _gl(nodes)
        c = 0.5 * (x + 1.0)  # cos θ on [0, 1]
        sth = np.sqrt(1.0 - c * c)
        bess = 2.0 * _jinc(qi * r * sth)
        ft = _sinc(qi * (0.5 * t + h) * c) * bess
        fc = _sinc(qi * 0.5 * t * c) * bess
        amp2 = (el * (vt * ft - vc * fc) + ec * vc * fc) ** 2
        tmp = qi * c

        def sq(m, c=c, tmp=tmp):
            s = np.ones_like(c)
            for k in range(1, m):
                s += (
                    2.0
                    / m
                    * (m - k)
                    * np.cos(k * d * tmp)
                    * np.exp(-k * (tmp * sigma) ** 2 / 2.0)
                )
            return s

        sqv = (1.0 - wn) * sq(nn) + (wn * sq(nn + 1) if wn > 0 else 0.0)
        out[i] = np.sum(0.5 * w * amp2 * sqv)
    return out


def _jinc(x):
    x = np.asarray(x, float)
    with np.errstate(invalid="ignore", divide="ignore"):
        v = special.j1(x) / x
    return np.where(np.abs(x) < 1e-8, 0.5, v)


# ------------------------------------------------------------------------------------------- fractal series
def _fractal_single(q, a, fd, xmin, xmax):
    """∫_{xmin}^{xmax} SD(x) x⁶/(1+q²x²)² dx, SD(x) = a fd x^(-1-fd)/(xmin^-fd - xmax^-fd) (integral in ln x)."""
    if a == 0 or xmin == xmax:
        return np.zeros(q.shape)
    lo, hi = math.log(xmin), math.log(xmax)
    m = max(4, math.ceil(abs(hi - lo) / 0.25))
    x, w = _gl(20)
    h = (hi - lo) / m
    u = (lo + h * (np.arange(m)[:, None] + 0.5 * (x[None, :] + 1.0))).ravel()
    wu = np.tile(0.5 * h * w, m)
    xx = np.exp(u)
    norm = a * fd / (xmin**-fd - xmax**-fd)
    f = xx ** (6.0 - fd) / (1.0 + (q[..., None] * xx) ** 2) ** 2
    return norm * (f @ wu)


def fractal_series1(q, p):
    """p = [N, Rg_min, fd1, Rg_max]: ∫ SD(x) x⁶/(1+q²x²)² dx with a power-law size distribution x^(-1-fd)."""
    q = np.asarray(q, float)
    n, xmin, fd1, xmax = p[:4]
    if n < 0 or xmin < 0 or fd1 <= 0 or xmax < 0:
        raise ValueError("N, xmin, xmax >= 0 and fd1 > 0 required")
    return _fractal_single(q, n, fd1, xmin, xmax)


def fractal_series2(q, p):
    """p = [N, xmin, fd1, x12, fd2, xmax]: two power-law pieces, continuous at x12, total amplitude N."""
    q = np.asarray(q, float)
    n, xmin, fd1, x12, fd2, xmax = p[:6]
    if min(n, xmin, fd1, x12, fd2, xmax) < 0:
        raise ValueError("parameters >= 0 required")
    a1 = (fd2 * n * xmax**fd2 * (x12**fd1 - xmin**fd1)) / (
        fd2 * x12**fd1 * xmax**fd2
        - (fd1 * x12**fd2 + (fd2 - fd1) * xmax**fd2) * xmin**fd1
    )
    a2 = (fd1 * n * (x12**fd2 - xmax**fd2) * xmin**fd1) / (
        -fd2 * x12**fd1 * xmax**fd2
        + (fd1 * x12**fd2 + (fd2 - fd1) * xmax**fd2) * xmin**fd1
    )
    return _fractal_single(q, a1, fd1, xmin, x12) + _fractal_single(
        q, a2, fd2, x12, xmax
    )


def fractal_series3(q, p):
    """p = [N, xmin, fd1, x12, fd2, x23, fd3, xmax]: three power-law pieces."""
    q = np.asarray(q, float)
    n, xmin, fd1, x12, fd2, x23, fd3, xmax = p[:8]
    if min(xmin, fd1, x12, fd2, x23, fd3, xmax) <= 0:
        raise ValueError("xmin, x12, x23, xmax, fd1..3 > 0 required")
    den = (fd2 * fd3 * x12**fd1 * x23**fd2 * xmax**fd3
           - (fd1 * fd2 * x12**fd2 * x23**fd3 + (-(fd1 * (fd2 - fd3) * x12**fd2) + (-fd1 + fd2) * fd3 * x23**fd2)
              * xmax**fd3) * xmin**fd1)  # fmt: skip
    a1 = fd2 * fd3 * n * x23**fd2 * xmax**fd3 * (x12**fd1 - xmin**fd1) / den
    a2 = -fd1 * fd3 * n * (x12**fd2 - x23**fd2) * xmax**fd3 * xmin**fd1 / den
    a3 = -fd1 * fd2 * n * x12**fd2 * (x23**fd3 - xmax**fd3) * xmin**fd1 / den
    return (_fractal_single(q, a1, fd1, xmin, x12) + _fractal_single(q, a2, fd2, x12, x23)
            + _fractal_single(q, a3, fd3, x23, xmax))  # fmt: skip


# ------------------------------------------------------------------------------------------- power-law profile
def power_law(q, p):
    """p = [R, alpha, Deta]: sphere with the profile Δη (1 - (r/R)^α):
    F = Δη [4π(sin qR - qR cos qR)/q³ - 4π R³ ∫_0^1 t^(α+2) sin(qRt)/(qRt) dt]."""
    q = np.asarray(q, float)
    r, alpha, deta = p[0], p[1], p[2]
    if r < 0 or alpha <= -3:
        raise ValueError("R >= 0 and alpha > -3 required")
    x = q * r
    nmax = math.ceil(0.6 * float(np.max(x, initial=0.0)) + 24)
    t, w = _gj(nmax, alpha + 2.0)
    integ = _sinc(x[..., None] * t) @ w
    f = 4.0 * np.pi * r**3 * (phi_sphere(x) / 3.0 - integ)
    return (deta * f) ** 2


def power_law_v(x, p, dist):
    r, alpha = p[0], p[1]
    return 4.0 * r**3 * alpha * np.pi / (9.0 + 3.0 * alpha) + 0.0 * np.asarray(x, float)


def profile_power_law(r, p):
    """p = [R, alpha, Deta]: Δη (1 - |r/R|^α) for |r| < R (profile)."""
    r = np.asarray(r, float)
    rr, alpha, deta = p[0], p[1], p[2]
    if rr < 0 or alpha <= -3:
        raise ValueError("R >= 0 and alpha > -3 required")
    if rr == 0:
        return np.where(r == 0, deta, 0.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        v = deta * (1.0 - np.power(np.abs(r / rr), alpha))
    return np.where(np.abs(r) < abs(rr), v, 0.0)


def _qr(i):
    def red(q, p):
        return np.asarray(q, float) * abs(p[i])

    return red


def _valid_xi_exp(q, p):
    return np.asarray(q, float) * p[0] <= 100.0


def _valid_xi_ov(q, p):
    d, rg = p[1], p[0]
    xi = math.sqrt((d + 2.0) * (d + 5.0) / (2.0 * d * (d + 1.0)) * rg * rg)
    return np.asarray(q, float) * xi <= 50.0


_FO = "plugins/fractal_obj/sasfit_ff_"
_FS = "plugins/fractal_series/sasfit_ff_"
_RG3 = ["Rg", "D", "I0"]
_SH = ["R_c", "DR1", "DR2", "L", "eta_c", "eta_1", "eta_2", "eta_sol", "x_sol_sh2", "n"]
_SH_S = {0: (2.0, 10.0, "log"), 1: (0.5, 3.0, "lin"), 2: (0.5, 3.0, "lin"), 3: (0.0, 5.0, "lin"), 4: (0.5, 2.0, "lin"),
         5: (1.0, 3.0, "lin"), 6: (0.5, 1.5, "lin"), 7: (0.0, 0.3, "lin"), 8: (0.0, 0.8, "lin")}  # fmt: skip
SPECS = [
    make_spec("ff_fisher_burford", fisher_burford, _RG3, {0: (1.0, 50.0, "log"), 1: (1.2, 3.0, "lin"), 2: (0.5, 2.0, "lin")},
              source=_FO + "fisher_burford.c", reduced=_qr(0), reduced_label="q Rg", rtol=1e-13),
    make_spec("ff_massfractexp", massfractexp, _RG3, {0: (1.0, 50.0, "log"), 1: (1.2, 3.0, "lin"), 2: (0.5, 2.0, "lin")},
              source=_FO + "massfractexp.c", reduced=_qr(0), reduced_label="q Rg", rtol=1e-12),
    make_spec("ff_massfractgauss", massfractgauss, _RG3, {0: (1.0, 20.0, "log"), 1: (1.2, 3.0, "lin"), 2: (0.5, 2.0, "lin")},
              source=_FO + "massfractgauss.c", reduced=_qr(0), reduced_label="q Rg", rtol=1e-9),
    make_spec("ff_dlcaggregation", dlcaggregation, _RG3, {0: (1.0, 50.0, "log"), 1: (1.2, 3.0, "lin"), 2: (0.5, 2.0, "lin")},
              source=_FO + "dlcaggregation.c", reduced=_qr(0), reduced_label="q Rg", rtol=1e-13),
    make_spec("ff_rlcaggregation", rlcaggregation, _RG3, {0: (1.0, 50.0, "log"), 1: (1.2, 3.0, "lin"), 2: (0.5, 2.0, "lin")},
              source=_FO + "rlcaggregation.c", reduced=_qr(0), reduced_label="q Rg", rtol=1e-13),
    make_spec("ff_aggregate__exp__x_a__cut_off_", aggregate_exp_cutoff, ["xi", "D", "I0", "alpha"],
              {0: (1.0, 30.0, "log"), 1: (1.2, 2.8, "lin"), 2: (0.5, 2.0, "lin"), 3: (0.7, 2.5, "lin")},
              source=_FO + "aggregate__exp__x_a__cut_off_.c", reduced=_qr(0), reduced_label="q ξ", rtol=1e-7,
              c_integrates=True, c_valid=_valid_xi_exp),
    make_spec("ff_aggregate__overlapsph_cut_off_", aggregate_overlapsph_cutoff, _RG3,
              {0: (1.0, 30.0, "log"), 1: (1.2, 2.8, "lin"), 2: (0.5, 2.0, "lin")},
              source=_FO + "aggregate__overlapsph_cut_off_.c", reduced=_qr(0), reduced_label="q Rg", rtol=1e-8,
              c_integrates=True, c_valid=_valid_xi_ov),
    make_spec("ff_dumbbellshell", dumbbellshell, ["R", "dR", "L", "eta_c", "eta_sh"],
              {0: (2.0, 20.0, "log"), 1: (0.5, 5.0, "lin"), 2: (0.0, 10.0, "lin"), 3: (0.5, 2.0, "lin"), 4: (1.0, 3.0, "lin")},
              source=_FO + "dumbbellshell.c", reduced=_qr(0), reduced_label="q R", rtol=1e-11),
    make_spec("ff_two_attached_spheres", two_attached_spheres, ["R1", "dummy", "R2", "dummy", "eta_0", "eta_1", "eta_2"],
              {0: (2.0, 20.0, "log"), 2: (2.0, 20.0, "log"), 4: (0.0, 0.5, "lin"), 5: (1.0, 2.0, "lin"), 6: (1.0, 3.0, "lin")},
              amplitude=two_attached_spheres_f, volume=two_attached_spheres_v, source=_FO + "two_attached_spheres.c",
              reduced=_qr(0), reduced_label="q R1", rtol=1e-11),
    make_spec("ff_two_attached_spheres_w_distr", two_attached_spheres_w_distr,
              ["R1", "sigma1", "R2", "sigma2", "eta_0", "eta_1", "eta_2"],
              {0: (2.0, 20.0, "log"), 1: (0.05, 0.3, "lin"), 2: (2.0, 20.0, "log"), 3: (0.05, 0.3, "lin"),
               4: (0.0, 0.5, "lin"), 5: (1.0, 2.0, "lin"), 6: (1.0, 3.0, "lin")},
              amplitude=two_attached_spheres_w_distr_f, volume=two_attached_spheres_w_distr_v,
              source=_FO + "two_attached_spheres_w_distr.c", reduced=_qr(0), reduced_label="q R1", rtol=3e-3,
              c_integrates=True, notes="C integrates the log-normals to 99.9 % only (~1e-3)"),
    make_spec("ff_doubleshellchain", doubleshellchain, _SH, {**_SH_S, 9: (1.0, 8.0, "lin")},
              source=_FO + "doubleshellchain.c", reduced=_qr(0), reduced_label="q R_c", rtol=1e-11),
    make_spec("ff_tetrahedrondoubleshell", tetrahedrondoubleshell, _SH, {**_SH_S, 9: (0.5, 5.0, "lin")},
              source=_FO + "tetrahedrondoubleshell.c", reduced=_qr(0), reduced_label="q R_c", rtol=1e-11),
    make_spec("ff_stackdiscs", stackdiscs, ["R", "t", "h", "n", "eta_c", "eta_l", "sigma", "D"],
              {0: (5.0, 30.0, "log"), 1: (1.0, 4.0, "lin"), 2: (0.5, 2.0, "lin"), 3: (1.0, 6.0, "lin"), 4: (0.5, 2.0, "lin"),
               5: (1.0, 3.0, "lin"), 6: (0.2, 2.0, "lin"), 7: (5.0, 15.0, "lin")},
              source=_FO + "stackdiscs.c", reduced=_qr(0), reduced_label="q R", rtol=1e-8, c_integrates=True),
    make_spec("ff_fractal_series1", fractal_series1, ["N", "Rg_min", "fd1", "Rg_max"],
              {0: (0.5, 2.0, "lin"), 1: (0.5, 3.0, "log"), 2: (0.5, 3.5, "lin"), 3: (20.0, 200.0, "log")},
              source=_FS + "fractal_series1.c", rtol=1e-9),
    make_spec("ff_fractal_series2", fractal_series2, ["N", "xmin", "fd1", "x12", "fd2", "xmax"],
              {0: (0.5, 2.0, "lin"), 1: (0.5, 3.0, "log"), 2: (0.5, 3.5, "lin"), 3: (5.0, 15.0, "log"), 4: (0.5, 3.5, "lin"),
               5: (20.0, 200.0, "log")},
              source=_FS + "fractal_series2.c", rtol=1e-9),
    make_spec("ff_fractal_series3", fractal_series3, ["N", "xmin", "fd1", "x12", "fd2", "x23", "fd3", "xmax"],
              {0: (0.5, 2.0, "lin"), 1: (0.5, 3.0, "log"), 2: (0.5, 3.5, "lin"), 3: (4.0, 8.0, "log"), 4: (0.5, 3.5, "lin"),
               5: (12.0, 20.0, "log"), 6: (0.5, 3.5, "lin"), 7: (40.0, 200.0, "log")},
              source=_FS + "fractal_series3.c", rtol=1e-9),
    make_spec("ff_power_law", power_law, ["R", "alpha", "Deta"], {0: (2.0, 50.0, "log"), 1: (-2.5, 4.0, "lin"), 2: (0.5, 2.0, "lin")},
              volume=power_law_v, source="plugins/powerlaw/sasfit_ff_power_law.c", reduced=_qr(0), reduced_label="q R",
              rtol=1e-8, c_reference=False, notes="C uses sasfit_pfq (not in the reference library); checked with mpmath"),
    make_spec("ff_profile__power_law", profile_power_law, ["R", "alpha", "Deta"],
              {0: (2.0, 50.0, "log"), 1: (-2.5, 4.0, "lin"), 2: (0.5, 2.0, "lin")},
              source="plugins/powerlaw/sasfit_ff_profile__power_law.c", rtol=1e-13),
]  # fmt: skip
