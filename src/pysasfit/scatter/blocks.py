"""Hypergeometric building blocks (d = 1, 2, 3) with Schulz-Zimm size averaging.

Monodisperse amplitude   F_d(x) = 0F1((d+2)/2; -x^2/4)   (x = q R)
  d=3: 3(sin x - x cos x)/x^3     d=2: 2 J1(x)/x      d=1: sin x / x
Formfactor               P_d(x) = F_d(x)^2

Size distribution (Schulz-Zimm), mean X, z = (1-sigma^2)/sigma^2, weighted by x^s:
    <x^n> = (z+s+1)_n u^n,   u = X/(z+1)

For every block three evaluation regimes are combined (Wagener & Foerster 2023):
  I   power series in (qX)^2 with recursively computed coefficients
  II  asymptotic expansion -> averages of x^-n cos/sin(kx) (closed form)
  III (Porod) = non-oscillating leading part of II
Series and asymptotics are chosen per element by an a-posteriori error estimate;
a Gauss-Laguerre quadrature over the size distribution is the last resort.
"""
from __future__ import annotations
import numpy as np
from scipy.special import gammaln, roots_genlaguerre, spherical_jn, j1, sici

TOL = 1e-6          # default target relative precision
YCAP = 16.0         # series are not evaluated beyond q*X = YCAP
NMAX = 400


def z_from_sigma(sigma):
    """Schulz-Zimm z from the relative standard deviation sigma (0 -> inf, monodisperse)."""
    if sigma is None or sigma <= 0:
        return np.inf
    return (1.0 - sigma ** 2) / sigma ** 2


# --------------------------------------------------------------------------
# Schulz averages of x^-n * {1, cos(kx), sin(kx)},  x = qR
# --------------------------------------------------------------------------
def _gratio(n, z, s):
    """Gamma(z+s-n+1)/Gamma(z+s+1)."""
    if float(n) == int(n) and 0 <= n <= 60:
        out = 1.0
        for i in range(int(n)):
            v = z + s - i
            if v <= 0:
                return np.nan
            out /= v
        return out
    m = z + s - n + 1
    if m <= 0:
        return np.nan
    return float(np.exp(gammaln(m) - gammaln(z + s + 1)))


def avg(n, k, kind, w, z, s):
    """<(qR)^-n g(k q R)>, g in {'1','cos','sin'};  w = qX/(z+1)."""
    base = _gratio(n, z, s) * w ** (-float(n))
    if kind == '1':
        return base
    m = z + s - n + 1
    kw = k * w
    damp = np.exp(-0.5 * m * np.log1p(kw * kw))
    ph = m * np.arctan(kw)
    return base * damp * (np.cos(ph) if kind == 'cos' else np.sin(ph))


def ydamp(z, s, n_osc=3, eps=1e-9):
    """q*X above which oscillating parts of the averaged functions are < eps."""
    if np.isinf(z):
        return 400.0
    m = z + s - n_osc
    if m <= 1:
        return 20.0
    x = np.expm1(2.0 * np.log(1.0 / eps) / m)
    return float(min((z + 1) * 0.5 * np.sqrt(x), 400.0))


# --------------------------------------------------------------------------
# Hankel coefficients for J1 (d = 2 asymptotics)
# --------------------------------------------------------------------------
def _hankel(K=16):
    a = [1.0]
    for k in range(1, 2 * K + 2):
        a.append(a[-1] * (4 - (2 * k - 1) ** 2) / (8.0 * k))
    p = np.zeros(K)
    q = np.zeros(K)
    for k in range(K):
        if 2 * k < K:
            p[2 * k] = (-1) ** k * a[2 * k]
        if 2 * k + 1 < K:
            q[2 * k + 1] = (-1) ** k * a[2 * k + 1]
    pp = np.convolve(p, p)[:K]
    qq = np.convolve(q, q)[:K]
    pq = np.convolve(p, q)[:K]
    return dict(c0=pp + qq, cs=pp - qq, cc=2 * pq, fs=p + q, fc=-(p - q), K=K)


_H = _hankel(16)


# --------------------------------------------------------------------------
class Block:
    """d = 1 (sinc), 2 (2J1/x) or 3 (sphere)."""

    def __init__(self, d):
        assert d in (1, 2, 3)
        self.d = d
        self.b = (d + 2) / 2.0

    # -- monodisperse ---------------------------------------------------
    def mono(self, x, kind='P'):
        x = np.asarray(x, float)
        with np.errstate(all='ignore'):
            if self.d == 3:
                F = np.where(x < 1e-8, 1.0, 3 * spherical_jn(1, x) / np.where(x == 0, 1, x))
            elif self.d == 2:
                F = np.where(x < 1e-8, 1.0, 2 * j1(x) / np.where(x == 0, 1, x))
            else:
                F = np.sinc(x / np.pi)
        return F * F if kind == 'P' else F

    # -- Regime I ---------------------------------------------------------
    def series(self, y, z, s, kind='P', tol=TOL, extra=None):
        """Averaged power series.  Returns (value, ok).  extra(n) multiplies term n."""
        y = np.asarray(y, float)
        val = np.full(y.shape, np.nan)
        ok = np.zeros(y.shape, bool)
        cand = (y < YCAP) & (y / (z + 1) < 0.4)
        if not cand.any():
            return val, ok
        y2 = y[cand] ** 2
        t = np.ones_like(y2)
        S = np.ones_like(y2)
        A = np.ones_like(y2)
        b = self.b
        for n in range(1, NMAX + 1):
            if kind == 'P':
                r = -(b + n - 1.5) / ((2 * b + n - 2) * (b + n - 1) * n)
            else:
                r = -1.0 / (4 * (b + n - 1) * n)
            r *= (2 * n + z + s) * (2 * n + z + s - 1) / (z + 1) ** 2
            t = t * r * y2
            e = t if extra is None else t * extra(n)
            S += e
            A += np.abs(e)
            if n > 8 and np.all(np.abs(e) < 1e-17 * A):
                break
        conv = np.abs(t) < 1e-14 * A
        floor = 1e-10 if kind == 'P' else 1e-3
        good = conv & (1.5e-15 * A < tol * np.maximum(np.abs(S), floor))
        val[cand] = S
        okc = np.zeros(y.shape, bool)
        okc[cand] = good
        return val, okc

    # -- Regimes II/III -------------------------------------------------------
    def closed(self, y, z, s, kind='P'):
        """Exact (d=1,3) averaged closed forms."""
        w = np.asarray(y, float) / (z + 1)
        A = lambda n, k, g: avg(n, k, g, w, z, s)
        with np.errstate(all='ignore'):
            if self.d == 3:
                if kind == 'P':
                    return 9 * (0.5 * (A(4, 0, '1') + A(4, 2, 'cos')) - A(5, 2, 'sin')
                                + 0.5 * (A(6, 0, '1') - A(6, 2, 'cos')))
                return 3 * (A(3, 1, 'sin') - A(2, 1, 'cos'))
            if self.d == 1:
                if kind == 'P':
                    return 0.5 * (A(2, 0, '1') - A(2, 2, 'cos'))
                return A(1, 1, 'sin')
        raise ValueError

    def nonosc(self, y, z, s):
        """Non-oscillating (Porod-type) part of <P>, d=3 and d=1."""
        w = np.asarray(y, float) / (z + 1)
        with np.errstate(all='ignore'):
            if self.d == 3:
                return 4.5 * (avg(4, 0, '1', w, z, s) + avg(6, 0, '1', w, z, s))
            if self.d == 1:
                return 0.5 * avg(2, 0, '1', w, z, s)
        raise ValueError

    def asym(self, y, z, s, kind='P', tol=TOL):
        """Hankel-based asymptotics for d = 2.  Returns (value, ok)."""
        y = np.asarray(y, float)
        H = _H
        K = H['K']
        n0 = 3 if kind == 'P' else 1.5
        jmax = int(min(K - 1, np.ceil(z + s - (2 if kind == 'P' else 0.5)) - 1))
        if jmax < 1:
            return np.full(y.shape, np.nan), np.zeros(y.shape, bool)
        w = y / (z + 1)
        out = np.zeros(y.shape)
        with np.errstate(all='ignore'):
            for j in range(jmax + 1):
                n = n0 + j
                if kind == 'P':
                    out += (4 / np.pi) * (H['c0'][j] * avg(n, 0, '1', w, z, s)
                                          - H['cs'][j] * avg(n, 2, 'sin', w, z, s)
                                          - H['cc'][j] * avg(n, 2, 'cos', w, z, s))
                else:
                    out += (2 / np.sqrt(np.pi)) * (H['fs'][j] * avg(n, 1, 'sin', w, z, s)
                                                   + H['fc'][j] * avg(n, 1, 'cos', w, z, s))
            if kind == 'P':
                last = max(abs(H['c0'][jmax]), abs(H['cs'][jmax]), abs(H['cc'][jmax]))
            else:
                last = max(abs(H['fs'][jmax]), abs(H['fc'][jmax]))
            rel = last / np.where(y > 0, y, np.inf) ** jmax
        ok = np.isfinite(out) & (rel < 1e-3 * tol)
        return out, ok

    # -- last resort ------------------------------------------------------------
    def quad(self, y, z, s, kind='P', N=300, mono=None):
        """Gauss-Laguerre quadrature over the Schulz distribution."""
        y = np.asarray(y, float)
        al = z + s
        if al > 140:              # narrow distribution: Gauss-Laguerre nodes overflow -> uniform grid in x
            return self._quad_grid(y, z, s, kind, mono)
        if al <= -1:
            return np.full(y.shape, np.nan)
        try:
            t, wt = roots_genlaguerre(N, al)
        except Exception:
            return np.full(y.shape, np.nan)
        if not np.all(np.isfinite(wt)):
            return np.full(y.shape, np.nan)
        wt = wt / wt.sum()
        xi = t / (z + 1)
        f = mono if mono is not None else (lambda x: self.mono(x, kind))
        flat = y.ravel()
        res = np.empty_like(flat)
        for i in range(0, flat.size, 2000):
            res[i:i + 2000] = f(flat[i:i + 2000, None] * xi[None, :]) @ wt
        return res.reshape(y.shape)

    def _quad_grid(self, y, z, s, kind='P', mono=None, N=1601):
        """Schulz average on a uniform grid of x = R/<R> (+-9 standard deviations); for large z."""
        y = np.asarray(y, float)
        sd = 1.0 / np.sqrt(z + 1.0)
        x = np.linspace(max(1e-6, 1 - 9 * sd), 1 + 9 * sd, N)
        lw = (z + s) * np.log(x) - (z + 1) * x
        w = np.exp(lw - lw.max())
        w /= w.sum()
        f = mono if mono is not None else (lambda u: self.mono(u, kind))
        flat = y.ravel()
        res = np.empty_like(flat)
        for i in range(0, flat.size, 1000):
            res[i:i + 1000] = f(flat[i:i + 1000, None] * x[None, :]) @ w
        return res.reshape(y.shape)

    # -- hybrid ---------------------------------------------------------------------
    def value(self, y, z, s, kind='P', tol=TOL):
        """<P> or <F> (kind='P'/'F') as function of y = q X; any array shape."""
        y = np.asarray(y, float)
        if np.isinf(z):
            return self.mono(y, kind)
        shape = y.shape
        y = np.abs(y.ravel())
        val, ok = self.series(y, z, s, kind, tol)
        rest = ~ok
        if rest.any():
            yr = y[rest]
            if self.d in (1, 3):
                v = self.closed(yr, z, s, kind)
                bad = ~np.isfinite(v)
            else:
                v, good = self.asym(yr, z, s, kind, tol)
                # Hankel truncation error is underestimated near y~10-30:
                # use the quadrature there when it is available.
                mid = good & (yr < 40.0)
                if mid.any():
                    with np.errstate(all='ignore'):
                        qv = self.quad(yr[mid], z, s, kind)
                    v = np.array(v)
                    sel = np.where(mid)[0][np.isfinite(qv)]
                    v[sel] = qv[np.isfinite(qv)]
                bad = ~good
            if bad.any():
                v = np.array(v)
                v[bad] = self.quad(yr[bad], z, s, kind)
            val[rest] = v
        nf = ~np.isfinite(val)
        if nf.any():                    # last resort (all routes failed): direct average on a grid
            val[nf] = self._quad_grid(y[nf], z, s, kind)
        return val.reshape(shape)

    # -- integral of the line formfactor: int_0^1 <P_line(y t)> dt ----------------------
    def line_integral(self, y, z, s, tol=TOL):
        assert self.d == 1
        y = np.asarray(y, float)
        mono = lambda x: _line_int_mono(x)
        if np.isinf(z):
            return mono(y)
        shape = y.shape
        y = np.abs(y.ravel())
        val, ok = self.series(y, z, s, 'P', tol, extra=lambda n: 1.0 / (2 * n + 1))
        rest = ~ok
        if rest.any():
            yr = y[rest]
            K = 10
            w = yr / (z + 1)
            with np.errstate(all='ignore'):
                v = 0.5 * np.pi * avg(1, 0, '1', w, z, s) - 0.5 * avg(2, 0, '1', w, z, s)
                from math import factorial
                for k in range(1, K + 1):
                    v = v - (-1) ** k * factorial(2 * k) / 2.0 ** (2 * k + 1) * avg(2 * k + 2, 2, 'cos', w, z, s)
                for k in range(0, K + 1):
                    v = v - (-1) ** k * factorial(2 * k + 1) / 2.0 ** (2 * k + 2) * avg(2 * k + 3, 2, 'sin', w, z, s)
                trunc = factorial(2 * K + 4) / (2 * yr) ** (2 * K + 5)
            bad = ~np.isfinite(v) | ~(trunc < tol)
            if bad.any():
                v = np.array(v)
                v[bad] = self.quad(yr[bad], z, s, 'P', mono=mono)
            val[rest] = v
        return val.reshape(shape)


def _line_int_mono(x):
    x = np.asarray(x, float)
    with np.errstate(all='ignore'):
        xs = np.where(x < 1e-6, 1.0, x)
        v = sici(2 * xs)[0] / xs - np.sin(xs) ** 2 / xs ** 2
    return np.where(x < 1e-6, 1.0, v)


BLOCK = {1: Block(1), 2: Block(2), 3: Block(3)}


# --------------------------------------------------------------------------
# cross term <F3(a x) F3(b x)> for core/shell spheres (x = q R, size averaged)
# --------------------------------------------------------------------------
def cross3(a, b, y, z, s, tol=TOL):
    """Schulz-Zimm average <F_3(a q R) F_3(b q R)>, y = q X, weight R^s."""
    y = np.abs(np.asarray(y, float))
    if np.isinf(z):
        return BLOCK[3].mono(a * y, 'F') * BLOCK[3].mono(b * y, 'F')
    shape = y.shape
    y = y.ravel()
    val = np.full(y.shape, np.nan)
    # ---- series: D_N coefficients of x^{2N}
    Nm = 220
    from scipy.special import gammaln as gl
    m = np.arange(Nm)
    sgn = (-1.0) ** m
    lc = -m * np.log(4.0) - gl(m + 1) - (gl(2.5 + m) - gl(2.5))
    Am = sgn * np.exp(lc + 2 * m * np.log(max(a, 1e-300)))
    Bm = sgn * np.exp(lc + 2 * m * np.log(max(b, 1e-300)))
    D = np.convolve(Am, Bm)[:Nm]
    Dabs = np.convolve(np.abs(Am), np.abs(Bm))[:Nm]
    N = np.arange(Nm)
    lg = gl(z + s + 1 + 2 * N) - gl(z + s + 1) - 2 * N * np.log(z + 1)
    amax = max(a, b)
    cand = (y * amax < YCAP) & (y * amax / (z + 1) < 0.4)
    if cand.any():
        yc = y[cand]
        ly = np.log(np.maximum(yc, 1e-300))
        T = D[None, :] * np.exp(lg[None, :] + 2 * N[None, :] * ly[:, None])
        Ta = Dabs[None, :] * np.exp(lg[None, :] + 2 * N[None, :] * ly[:, None])
        S = T.sum(1)
        Aabs = Ta.sum(1)
        conv = np.abs(T[:, -1]) < 1e-15 * Aabs
        good = conv & (3e-15 * Aabs < tol * np.maximum(np.abs(S), 1e-10))
        tmp = np.full(yc.shape, np.nan)
        tmp[good] = S[good]
        val[cand] = tmp
    rest = ~np.isfinite(val)
    if rest.any():
        yr = y[rest]
        w = yr / (z + 1)
        sm, d = a + b, a - b
        A = lambda n, k, g: avg(n, k, g, w, z, s)
        with np.errstate(all='ignore'):
            v = 0.5 * (A(6, d, 'cos') - A(6, sm, 'cos')) \
                - 0.5 * b * (A(5, sm, 'sin') + A(5, d, 'sin')) \
                - 0.5 * a * (A(5, sm, 'sin') - A(5, d, 'sin')) \
                + 0.5 * a * b * (A(4, d, 'cos') + A(4, sm, 'cos'))
            v = 9.0 / (a ** 3 * b ** 3) * v
        bad = ~np.isfinite(v)
        if bad.any():
            v = np.array(v)
            v[bad] = BLOCK[3].quad(yr[bad], z, s, 'F',
                                   mono=lambda x: BLOCK[3].mono(a * x, 'F') * BLOCK[3].mono(b * x, 'F'))
        val[rest] = v
    return val.reshape(shape)
