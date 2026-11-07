"""Uniaxially oriented axisymmetric particles: 2D patterns for orientation distribution h(delta).

h(delta) is the (unnormalised) density of particle-axis directions at polar angle delta from the
director.  The detector plane is taken perpendicular to the beam (q_z of the beam direction ~ 0),
director in the detector plane along +y... (angle phi_q measured from the director).
"""
import numpy as np
from .shapes import _gl


def h_gauss(sigma_delta):
    """Gaussian h(delta) ~ exp(-delta^2 / (2 sigma^2)) (axis distribution, delta in rad)."""
    return lambda d: np.exp(-0.5 * (d / sigma_delta) ** 2)


def h_cone(chi):
    """Uniform within a cone of half angle chi (smoothed edge-free box)."""
    return lambda d: (d <= chi).astype(float)


def h_maier_saupe(kappa):
    return lambda d: np.exp(kappa * np.cos(d) ** 2)


def pattern(shape, qx, qy, h, n_delta=None, n_psi=64, delta_max=np.pi):
    """I(qx,qy) by quadrature for axisymmetric particles whose axes follow h(delta) around the
    director (director = +x of the pattern, in the detector plane, q_z ~ 0).  Normalised: I(0)=1.

    Axis direction n = (cos d, sin d cos psi, sin d sin psi); mu = cos(angle q,n)
    = (qx cos d + qy sin d cos psi)/|q|  (Eq. 13 of the paper with chi = psi).
    """
    qx, qy = np.broadcast_arrays(np.asarray(qx, float), np.asarray(qy, float))
    q = np.hypot(qx, qy)
    n_delta = n_delta or 120
    d, wd = _gl(n_delta, 0.0, delta_max)
    wd = wd * h(d) * np.sin(d)
    wd = wd / wd.sum()
    psi = (np.arange(n_psi) + 0.5) * 2 * np.pi / n_psi
    qs = np.where(q > 0, q, 1.0)
    ux = (qx / qs)[..., None, None]
    uy = (qy / qs)[..., None, None]
    mu = ux * np.cos(d)[:, None] + uy * np.sin(d)[:, None] * np.cos(psi)[None, :]
    P = shape.P_fixed(q[..., None, None], np.clip(mu, -1, 1))
    return (P * wd[:, None]).sum(-2).mean(-1)


# ---------------------------------------------------------------------------------------
# Moment-series route (paper Eqs. 13-15, SI Sect. 7)
# ---------------------------------------------------------------------------------------
from fractions import Fraction
from math import comb, factorial
from scipy.special import gammaln
_W_CACHE = {}


def _series_coeffs(nu, power, Nmax):
    """W[N][b]: coefficient of (R^2)^N q^{2(N-b)} u^{2b} in the (power-th power of the) cylinder
    amplitude 0F1(2;-qr^2R^2/4) 0F1(3/2;-u^2 H^2/4), qr^2 = q^2-u^2, u = q.axis, H = nu R."""
    key = (float(nu), power, Nmax)
    if key in _W_CACHE:
        return _W_CACHE[key]
    nuf = Fraction(nu).limit_denominator(10 ** 9)

    def base(bb):
        out, poch = [], Fraction(1)
        for n in range(Nmax + 1):
            if n:
                poch *= (bb + n - 1)
            out.append(Fraction(-1, 4) ** n / (factorial(n) * poch))
        return out
    a1, b1 = base(Fraction(2)), base(Fraction(3, 2))
    b1 = [c * nuf ** (2 * k) for k, c in enumerate(b1)]

    def mul(x, y):
        z = [Fraction(0)] * (Nmax + 1)
        for i, xi in enumerate(x):
            if xi:
                for j in range(Nmax + 1 - i):
                    z[i + j] += xi * y[j]
        return z
    al, be = a1, b1
    if power == 2:
        al, be = mul(a1, a1), mul(b1, b1)
    W = np.zeros((Nmax + 1, Nmax + 1))
    for N in range(Nmax + 1):
        row = [Fraction(0)] * (N + 1)
        for a in range(N + 1):
            k = N - a
            c = al[a] * be[k]
            for j in range(a + 1):
                row[j + k] += c * comb(a, j) * (-1) ** j
        W[N, :N + 1] = [float(v) for v in row]
    _W_CACHE[key] = W
    return W


def _h_moments(h, Nmax, delta_max=np.pi / 2, n=400):
    d, w = _gl(n, 0.0, delta_max)
    w = w * h(d) * np.sin(d)
    w = w / w.sum()
    c2, s2 = np.cos(d) ** 2, np.sin(d) ** 2
    H = np.zeros((Nmax + 1, Nmax + 1))
    cp = [np.ones_like(d)]
    sp = [np.ones_like(d)]
    for i in range(Nmax):
        cp.append(cp[-1] * c2)
        sp.append(sp[-1] * s2)
    for i in range(Nmax + 1):
        for j in range(Nmax + 1 - i):
            H[i, j] = (w * cp[i] * sp[j]).sum()
    return H


def pattern_series(cyl, qx, qy, h, Nmax=60, kind='P', tol=1e-6, fallback=True):
    """Oriented-cylinder pattern from the *moment series* of the paper (Eqs. 13-15).

    The q-independent coefficients (series coefficients W and orientation moments H_{2l,2n-2l} of h)
    are pre-computed; each pixel then needs only polynomial sums.  The series is accepted pixelwise
    if an a-posteriori floating-point error estimate is below `tol`; other pixels are evaluated by
    quadrature (`fallback=True`) or returned as NaN.  Director along +x.  Returns I (or (I, ok)).
    """
    from .blocks import z_from_sigma
    power = 2 if kind == 'P' else 1
    sw = cyl.sP if kind == "P" else cyl.sF
    qx, qy = np.broadcast_arrays(np.asarray(qx, float), np.asarray(qy, float))
    shp = qx.shape
    qx, qy = qx.ravel(), qy.ravel()
    W = _series_coeffs(cyl.nu, power, Nmax)
    H = _h_moments(h, Nmax)
    n = qx.size
    qx2, qy2 = qx ** 2, qy ** 2
    q2 = qx2 + qy2
    # powers
    Pxy = np.empty((n, Nmax + 1)); Pyy = np.empty((n, Nmax + 1)); Pq = np.empty((n, Nmax + 1))
    Pxy[:, 0] = Pyy[:, 0] = Pq[:, 0] = 1.0
    for i in range(1, Nmax + 1):
        Pxy[:, i] = Pxy[:, i - 1] * qx2
        Pyy[:, i] = Pyy[:, i - 1] * qy2
        Pq[:, i] = Pq[:, i - 1] * q2
    # chi average of sin^{2j}
    Sj = np.array([factorial(2 * j) / (4.0 ** j * factorial(j) ** 2) for j in range(Nmax + 1)])
    U = np.zeros((n, Nmax + 1)); Uabs = np.zeros_like(U)
    for b in range(Nmax + 1):
        j = np.arange(b + 1)
        coef = np.array([comb(2 * b, 2 * jj) for jj in j], float) * H[b - j, j] * Sj[j]
        U[:, b] = (Pxy[:, b - j] * Pyy[:, j]) @ coef
    z = cyl.z
    N = np.arange(Nmax + 1)
    if np.isinf(z):
        g = np.ones(Nmax + 1)
    else:
        g = np.exp(gammaln(z + sw + 1 + 2 * N) - gammaln(z + sw + 1) - 2 * N * np.log(z + 1))
    g = g * cyl.R ** (2 * N)
    S = np.zeros(n); A = np.zeros(n); last = np.zeros(n)
    for Nn in range(Nmax + 1):
        b = np.arange(Nn + 1)
        t = (U[:, :Nn + 1] * Pq[:, Nn - b]) * (W[Nn, :Nn + 1] * g[Nn])[None, :]
        term = t.sum(1)
        S += term
        A += np.abs(t).sum(1)
        last = np.abs(term)
    ok = (last < 1e-12 * A) & (4e-16 * A < tol * np.maximum(np.abs(S), 1e-12)) & (S > 0)
    res = np.where(ok, S, np.nan)
    if fallback and (~ok).any():
        bad = ~ok
        res[bad] = pattern(cyl_fixed_wrap(cyl, kind), qx[bad], qy[bad], h, delta_max=np.pi / 2,
                           n_delta=160)
    res = res.reshape(shp)
    return res if fallback else (res, ok.reshape(shp))


class cyl_fixed_wrap:
    """adapter so that pattern() can evaluate F or P of a cylinder."""

    def __init__(self, cyl, kind):
        self.c, self.kind = cyl, kind

    def P_fixed(self, q, mu):
        return self.c.P_fixed(q, mu) if self.kind == 'P' else self.c.F_fixed(q, mu)
