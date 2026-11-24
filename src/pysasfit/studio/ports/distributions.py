"""Python ports of SASfit's distribution-type plugins: peaks (peak_trapezoidal, q_distribution,
kappa_generalized_distribution, generalized_distributions) and size distributions (metalog, t_r_y).

Every port has a complete statistics function ``_v(u, p, dist)`` with SASfit's selector ``dist`` (``enum
sasfit_dist`` in sasfit_common.h):

===  ==========================  =====================================================================
dist name                        returns
===  ==========================  =====================================================================
0    DISTRIBUTION_TYPE           1 (QUANTILE) if the distribution is defined by its quantile function,
                                 3 (PROBABILITY) if it is defined by its density
1    DISTRIBUTION_QUANTILE       Q(y), u = y in (0, 1)
2    DISTRIBUTION_QUANTILE_DENS  q(y) = dQ/dy = 1/p(Q(y)), u = y
3    DISTRIBUTION_PROBABILITY    p(x), u = x (size distributions: N p(x); peaks: the normalised shape)
4    DISTRIBUTION_CUMULATIVE     F(x), u = x (size distributions: N F(x); peaks: F(x))
5    DISTRIBUTION_MODE           argmax p
6    DISTRIBUTION_MEAN           first moment (nan where it does not exist)
7    DISTRIBUTION_VARIANCE       second central moment (nan / inf where it does not exist)
8    DISTRIBUTION_MEDIAN         Q(1/2)
9    DISTRIBUTION_SKEWNESS       Bowley (quartile) skewness (Q(3/4) + Q(1/4) - 2 Q(1/2)) / IQR
10   DISTRIBUTION_EXCESS_KURTOSIS  Moors (octile) kurtosis (Q(7/8) - Q(5/8) + Q(3/8) - Q(1/8)) / IQR
===  ==========================  =====================================================================

Skewness and kurtosis follow the C plugins that define them (metalog, t_r_y): quantile measures, which exist for
every distribution (the kurtosis is Moors' ratio, 1.2331 for a normal distribution, not an excess value). The
statistics refer to the normalised shape: amplitude N or A, background y0, the weighting x^-alpha of metalog and the
clipping to [bl, bu] are not part of them. Where the C code gives a statistic, the port computes it the same way;
the others come from closed forms, scipy.stats (GEV = genextreme, generalised hyperbolic = genhyperbolic,
generalised inverse Gaussian = geninvgauss, q-Gaussian = Student t / symmetric beta) or quadrature of the density
or of the quantile function (moments = int_0^1 Q(y)^k dy), and root finding (CDF of quantile-defined distributions,
quantile of density-defined ones). See docs/studio_manual/python_ports.md and the C fixes listed there.
"""

from __future__ import annotations

import math

import numpy as np
from scipy import integrate, optimize, special, stats

from .common import make_spec

__all__ = ["SPECS", "DIST_NAMES"]

(TYPE, QUANTILE, QUANTILE_DENS, PROBABILITY, CUMULATIVE, MODE, MEAN, VARIANCE, MEDIAN, SKEWNESS,
 EXCESS_KURTOSIS) = range(11)  # fmt: skip
DIST_NAMES = ("type", "quantile", "quantile density", "probability", "cumulative", "mode", "mean", "variance",
              "median", "skewness", "excess kurtosis")  # fmt: skip
_NRIQ = 1e-12  # sasfit_eps_get_nriq default (y = 1 is replaced by 1 - nriq in the C quantiles)


# ============================================================================== generic distribution machinery
class _Dist:
    """Normalised distribution; subclasses give either (pdf, cdf, lo, hi) or (Q, q)."""

    native = "pdf"  # or "quantile"
    lo, hi = -np.inf, np.inf

    # -- density-defined defaults
    def pdf(self, x):
        raise NotImplementedError

    def cdf(self, x):
        if self.native == "quantile":
            return self._cdf_from_quantile(x)
        x = np.atleast_1d(np.asarray(x, float))
        out = np.empty(x.shape)
        for i, xi in np.ndenumerate(x):
            if xi <= self.lo:
                out[i] = 0.0
            elif xi >= self.hi:
                out[i] = 1.0
            else:
                out[i] = min(max(self._integral(lambda t: 1.0, xi), 0.0), 1.0)
        return out

    def _integral(self, g, x=np.inf):
        """int_lo^x g(t) p(t) dt: logarithmic substitution t = lo + e^s for a finite lower end (heavy tails and
        end singularities), otherwise split at the mode."""
        kw = dict(limit=500, epsabs=0, epsrel=1e-12)
        if np.isfinite(self.lo):
            b = self.lo
            def h(s):
                e = np.exp(np.float64(s))
                pv = self._pdf1(b + e)
                return 0.0 if pv == 0 or not np.isfinite(e) else float(g(np.float64(b + e)) * pv * e)

            top = math.log(x - b) if np.isfinite(x) else np.inf
            m = self.mode()
            sm = math.log(m - b) if m > b else None
            if sm is not None and sm < top:
                return integrate.quad(h, -np.inf, sm, **kw)[0] + integrate.quad(h, sm, top, **kw)[0]
            return integrate.quad(h, -np.inf, top, **kw)[0]
        m = self.mode()
        def f(t):
            pv = self._pdf1(t)
            return 0.0 if pv == 0 else float(g(np.float64(t)) * pv)

        if x <= m:
            return integrate.quad(f, -np.inf, x, **kw)[0]
        def h(s):
            e = np.exp(np.float64(s))
            return 0.0 if not np.isfinite(e) else f(m + e) * e

        top = math.log(x - m) if np.isfinite(x) else np.inf
        return integrate.quad(f, -np.inf, m, **kw)[0] + integrate.quad(h, -np.inf, top, **kw)[0]

    def _pdf1(self, x):
        return float(np.atleast_1d(self.pdf(np.array([x])))[0])

    def _breaks(self):
        return []

    def _q_lo(self):
        return -np.inf

    def Q(self, y):
        if self.native == "quantile":
            raise NotImplementedError
        y = np.atleast_1d(np.asarray(y, float))
        out = np.empty(y.shape)
        for i, yi in np.ndenumerate(y):
            out[i] = self._ppf1(yi)
        return out

    def _bracket(self):
        """Interval that contains most of the probability (for root finding of the quantile)."""
        m = self.mode()
        s = self._scale()
        lo = self.lo if np.isfinite(self.lo) else m - s
        hi = self.hi if np.isfinite(self.hi) else m + s
        return lo, hi

    def _scale(self):
        return 1.0

    def _ppf1(self, y):
        if y <= 0:
            return self.lo
        if y >= 1:
            return self.hi
        lo, hi = self._bracket()
        def f(x):
            v = float(self.cdf(np.array([x]))[0]) - y
            return v if np.isfinite(v) else 1.0

        step = max(hi - lo, 1e-300)
        k = 0
        while f(lo) > 0 and k < 200:
            lo = max(lo - step, self.lo) if np.isfinite(self.lo) else lo - step
            step *= 2
            k += 1
        step = max(hi - lo, 1e-300)
        k = 0
        while f(hi) < 0 and k < 200:
            hi = min(hi + step, self.hi) if np.isfinite(self.hi) else hi + step
            step *= 2
            k += 1
        return optimize.brentq(f, lo, hi, xtol=1e-14, rtol=1e-15, maxiter=500)

    def q(self, y):
        y = np.atleast_1d(np.asarray(y, float))
        with np.errstate(divide="ignore", invalid="ignore"):
            return 1.0 / self.pdf(self.Q(y))

    # -- quantile-defined defaults
    def _cdf_from_quantile(self, x):
        x = np.atleast_1d(np.asarray(x, float))
        out = np.empty(x.shape)
        for i, xi in np.ndenumerate(x):
            f = lambda y: float(self.Q(np.array([y]))[0]) - xi  # noqa: E731
            a, b = 1e-300, 1.0 - 1e-16
            fa, fb = f(a), f(b)
            if not fa < 0:
                out[i] = 0.0
            elif not fb > 0:
                out[i] = 1.0
            else:
                out[i] = optimize.brentq(f, a, b, xtol=1e-300, rtol=1e-15, maxiter=500)
        return out

    def _pdf_from_quantile(self, x):
        y = self.cdf(x)
        with np.errstate(divide="ignore", invalid="ignore"):
            p = 1.0 / self.q(y)
        return np.where((y > 0) & (y < 1) & np.isfinite(p), p, 0.0)

    # -- statistics
    def median(self):
        return float(self.Q(np.array([0.5]))[0])

    def bowley(self):
        q1, q2, q3 = self.Q(np.array([0.25, 0.5, 0.75]))
        return float((q3 + q1 - 2 * q2) / (q3 - q1))

    def moors(self):
        e1, e2, e3, e5, e6, e7 = self.Q(np.array([1, 2, 3, 5, 6, 7]) / 8.0)
        return float((e7 - e5 + e3 - e1) / (e6 - e2))

    def moment(self, k):
        """E[X^k] by quadrature of Q(y)^k over (0, 1) (quantile-defined) or of x^k p(x)."""
        if k >= getattr(self, "kmax", math.inf):      # power-law tail: moment does not exist
            return np.inf
        if self.native == "quantile":
            g = lambda y: float(self.Q(np.array([y]))[0]) ** k  # noqa: E731
            parts = [(0.0, 0.5), (0.5, 1.0)]
            tot = 0.0
            for a, b in parts:
                v, err = integrate.quad(g, a, b, limit=500, epsabs=0, epsrel=1e-11)
                if not np.isfinite(v) or (abs(v) > 0 and err > 1e-6 * abs(v) + 1e-300):
                    return np.nan
                tot += v
            return tot
        v = self._integral(lambda t: t**k)
        return v if np.isfinite(v) else np.nan

    def mean(self):
        return self.moment(1)

    def var(self):
        m1 = self.mean()
        if not np.isfinite(m1):
            return np.nan
        if 2 >= getattr(self, "kmax", math.inf):
            return np.inf
        if self.native == "quantile":
            g = lambda y: (float(self.Q(np.array([y]))[0]) - m1) ** 2  # noqa: E731
            tot = 0.0
            for a, b in ((0.0, 0.5), (0.5, 1.0)):
                v, err = integrate.quad(g, a, b, limit=500, epsabs=0, epsrel=1e-11)
                if not np.isfinite(v) or err > 1e-6 * abs(v) + 1e-300:
                    return np.inf
                tot += v
            return tot
        v = self._integral(lambda t: (t - m1) ** 2)
        return v if np.isfinite(v) else np.inf

    def mode(self):
        """Maximum of the density: quantile-defined -> minimum of q(y) on (0, 1) (logit scale)."""
        if self.native == "quantile":
            f = lambda t: float(self.q(np.array([special.expit(t)]))[0])  # noqa: E731
            ts = np.linspace(-30, 30, 241)
            vals = np.array([f(t) for t in ts])
            vals = np.where(np.isfinite(vals) & (vals > 0), vals, np.inf)
            k = int(np.argmin(vals))
            if k == 0 or k == ts.size - 1:
                return float(self.Q(np.array([special.expit(ts[k])]))[0])
            r = optimize.minimize_scalar(f, bounds=(ts[k - 1], ts[k + 1]), method="bounded", options={"xatol": 1e-12})
            return float(self.Q(np.array([special.expit(r.x)]))[0])
        raise NotImplementedError


def _stat(dist_obj, u, dist, scale=1.0):
    """Value of the statistic ``dist`` (SASfit selector) of a _Dist; ``scale`` multiplies p and F (size
    distributions: N)."""
    u = np.asarray(u, float)
    d = int(dist)
    if d == TYPE:
        return np.full(u.shape, float(QUANTILE if dist_obj.native == "quantile" else PROBABILITY))
    if d == QUANTILE:
        return dist_obj.Q(u).reshape(u.shape)
    if d == QUANTILE_DENS:
        return dist_obj.q(u).reshape(u.shape)
    if d == PROBABILITY:
        return scale * np.asarray(dist_obj.pdf(u), float).reshape(u.shape)
    if d == CUMULATIVE:
        return scale * np.asarray(dist_obj.cdf(u), float).reshape(u.shape)
    fn = {MODE: dist_obj.mode, MEAN: dist_obj.mean, VARIANCE: dist_obj.var, MEDIAN: dist_obj.median,
          SKEWNESS: dist_obj.bowley, EXCESS_KURTOSIS: dist_obj.moors}.get(d)  # fmt: skip
    if fn is None:
        raise ValueError(f"dist = {d} not defined (0 ... 10, see {__name__})")
    return np.full(u.shape, float(fn()))


# ============================================================================== peak_trapezoidal
class _Trapez(_Dist):
    """General trapezoid alpha < beta < gamma < delta, rise/fall exponents n1, n3, height ratio phi."""

    def __init__(self, a, b, c, d, phi=1.0, n1=2.0, n3=2.0, closed_left=False):
        self.a, self.b, self.c, self.d = float(a), float(b), float(c), float(d)
        self.phi, self.n1, self.n3 = float(phi), float(n1), float(n3)
        self.closed_left = closed_left
        self.lo, self.hi = self.a, self.d
        a, b, c, d, phi, n1, n3 = self.a, self.b, self.c, self.d, self.phi, self.n1, self.n3
        self.u = 2 * n1 * n3 / (2 * phi * (b - a) * n3 + (phi + 1) * (c - b) * n1 * n3 + 2 * (d - c) * n1)

    def pdf(self, x):
        x = np.asarray(x, float)
        a, b, c, d, phi, n1, n3, u = self.a, self.b, self.c, self.d, self.phi, self.n1, self.n3, self.u
        with np.errstate(divide="ignore", invalid="ignore"):
            left = (a <= x) & (x < b) if self.closed_left else (a < x) & (x < b)
            r = np.where(left, phi * u * np.abs((x - a) / (b - a)) ** (n1 - 1), 0.0)
            r = np.where((b <= x) & (x < c), u * ((phi - 1) * (c - x) / (c - b) + 1), r)
            r = np.where((c <= x) & (x < d), u * np.abs((d - x) / (d - c)) ** (n3 - 1), r)
        return r

    def peak(self):
        """Maximum of the density (normalisation of the amplitude versions in C: max(u phi, u))."""
        return max(self.u * self.phi, self.u)

    def _breaks(self):
        return [self.b, self.c]

    def mode(self):
        if self.n1 < 1 or self.n3 < 1:                 # density diverges at an end: the stronger singularity
            return self.a if self.n1 <= self.n3 else self.d
        if self.phi > 1:
            return self.b
        if self.phi < 1:
            return self.c
        return 0.5 * (self.b + self.c)

    def cdf(self, x):
        x = np.asarray(x, float)
        a, b, c, d, phi, n1, n3, u = self.a, self.b, self.c, self.d, self.phi, self.n1, self.n3, self.u
        F1 = phi * u * (b - a) / n1                         # mass of the rising part
        Fm = u * (c - b) * (phi + 1) / 2                     # plateau / sloped middle
        with np.errstate(divide="ignore", invalid="ignore"):
            t1 = np.clip((x - a) / (b - a), 0, 1)
            r = F1 * t1**n1
            s = np.clip(x - b, 0, c - b)
            r = r + np.where(x > b, u * (phi * s - (phi - 1) * s * s / (2 * (c - b))), 0.0)
            t3 = np.clip((d - x) / (d - c), 0, 1)
            r = r + np.where(x > c, u * (d - c) / n3 * (1 - t3**n3), 0.0)
        del Fm
        return np.clip(r, 0.0, 1.0)

    def Q(self, y):
        y = np.atleast_1d(np.asarray(y, float))
        out = np.empty(y.shape)
        for i, yi in np.ndenumerate(y):
            if yi <= 0:
                out[i] = self.a
            elif yi >= 1:
                out[i] = self.d
            else:
                out[i] = optimize.brentq(lambda x: float(self.cdf(np.array([x]))[0]) - yi, self.a, self.d,
                                         xtol=1e-15 * max(1.0, abs(self.d)), rtol=1e-15, maxiter=500)  # fmt: skip
        return out


def _trapez_from(p, kind):
    if kind == "trapez":
        return _Trapez(p[1], p[2], p[3], p[4], 1.0, 2.0, 2.0, closed_left=True)
    if kind == "symm":
        c, wp, we = p[1], abs(p[2]), abs(p[3])
        return _Trapez(c - wp / 2 - we / 2, c - wp / 2, c + wp / 2, c + wp / 2 + we / 2, 1.0, 2.0, 2.0, True)
    if kind == "general":
        return _Trapez(p[1], p[2], p[3], p[4], p[5], p[6], p[7])
    if kind == "symm_general":
        c, wp, we = p[1], abs(p[2]), abs(p[3])
        return _Trapez(c - wp / 2 - we / 2, c - wp / 2, c + wp / 2, c + wp / 2 + we / 2, 1.0, p[6], p[6])
    raise ValueError(kind)


class _SmoothTrapez(_Dist):
    """Uniform(W1) * Uniform(W2) * Normal(sigma): symmetric about the centre."""

    def __init__(self, c, w1, w2, s):
        self.c, self.w1, self.w2, self.s = float(c), abs(float(w1)), abs(float(w2)), abs(float(s))

    def _box(self):
        return abs(abs(self.w1 - self.w2) / abs(self.w1 + self.w2) - 1) <= 1e-14

    def _sharp(self):
        return self.s / abs(self.w1 + self.w2) <= 1e-14

    def pdf(self, x):
        u = np.asarray(x, float) - self.c
        w1, w2, s = self.w1, self.w2, self.s
        if self._box():
            W = max(w1, w2)
            if self._sharp():
                return np.where(np.abs(u) <= abs(w1 + w2) / 2, 1.0 / abs(w1 + w2), 0.0)
            return 0.5 * (special.erf((W - 2 * u) / (2 * math.sqrt(2) * s))
                          + special.erf((W + 2 * u) / (2 * math.sqrt(2) * s))) / W  # fmt: skip
        if self._sharp():
            return np.clip((w1 + w2) / 2 - np.abs(u), 0.0, min(w1, w2)) / (w1 * w2)
        a, b = w1 / 2, w2 / 2

        def psi(t):  # second antiderivative of the normal density
            z = t / s
            return t * special.ndtr(z) + s * np.exp(-0.5 * z * z) / math.sqrt(2 * math.pi)

        return (psi(u + a + b) - psi(u + a - b) - psi(u - a + b) + psi(u - a - b)) / (w1 * w2)

    def cdf(self, x):
        u = np.asarray(x, float) - self.c
        w1, w2, s = self.w1, self.w2, self.s
        if self._box():
            W = max(w1, w2)
            if self._sharp():
                W = abs(w1 + w2)
                return np.clip((u + W / 2) / W, 0.0, 1.0)

            def psi(t):
                z = t / s
                return t * special.ndtr(z) + s * np.exp(-0.5 * z * z) / math.sqrt(2 * math.pi)

            return (psi(u + W / 2) - psi(u - W / 2)) / W
        a, b = w1 / 2, w2 / 2
        if self._sharp():
            def chi(t):
                return 0.5 * np.maximum(t, 0.0) ** 2
        else:
            def chi(t):  # third antiderivative of the normal density
                z = t / s
                return 0.5 * (t * t + s * s) * special.ndtr(z) + 0.5 * s * t * np.exp(-0.5 * z * z) / math.sqrt(2 * math.pi)

        return (chi(u + a + b) - chi(u + a - b) - chi(u - a + b) + chi(u - a - b)) / (w1 * w2)

    def _scale(self):
        return self.w1 + self.w2 + 6 * self.s

    def mode(self):
        return self.c

    def median(self):
        return self.c

    def mean(self):
        return self.c

    def var(self):
        if self._box():
            W = max(self.w1, self.w2)
            return W * W / 12 + self.s**2
        return (self.w1**2 + self.w2**2) / 12 + self.s**2

    def bowley(self):
        return 0.0

    def _bracket(self):
        h = (self.w1 + self.w2) / 2 + 12 * self.s
        return self.c - h, self.c + h


def _trapez_specs():
    out = []

    def mk(name, kind, ampl, pars, sample):
        def shape(p):
            return _trapez_from(p, kind)

        def inten(q, p, kind=kind, ampl=ampl):
            d = shape(p)
            c0 = p[8] if kind in ("general", "symm_general") else p[5]
            if ampl:
                return p[0] * d.pdf(q) / d.peak() + c0
            return p[0] * d.pdf(q) + c0

        def vol(u, p, dist):
            return _stat(shape(p), u, dist)

        inten.__doc__ = f"SASfit {name}: trapezoidal peak ({'height' if ampl else 'area'} A)."
        out.append(make_spec(name, inten, pars, sample, volume=vol, source=f"plugins/peak_trapezoidal/sasfit_{name}.c",
                             rtol=1e-12, label="", group="peak", reduced=None))  # fmt: skip

    base = ["A", "alpha", "beta", "gamma", "delta", "c0"]
    s_tr = {0: (0.5, 3.0, "lin"), 1: (0.0, 1.0, "lin"), 2: (1.5, 2.5, "lin"), 3: (3.0, 4.0, "lin"), 4: (5.0, 6.0, "lin"),
            5: (0.0, 0.2, "lin")}
    gen = ["A", "alpha", "beta", "gamma", "delta", "phi", "n1", "n3", "c0"]
    s_gen = {**s_tr, 5: (0.3, 3.0, "log"), 6: (0.5, 4.0, "log"), 7: (0.5, 4.0, "log"), 8: (0.0, 0.2, "lin")}
    sym = ["A", "center", "wp", "we", "dummy", "c0"]
    s_sym = {0: (0.5, 3.0, "lin"), 1: (-1.0, 1.0, "lin"), 2: (0.5, 3.0, "lin"), 3: (0.5, 3.0, "lin"), 5: (0.0, 0.2, "lin")}
    sgen = ["A", "center", "wp", "we", "dummy", "dummy", "n1", "dummy", "c0"]
    s_sgen = {**s_sym, 5: (0.0, 0.0, "lin"), 6: (0.5, 4.0, "log"), 8: (0.0, 0.2, "lin")}
    for ampl, sfx_t, sfx_a in ((False, "_area", "area"), (True, "_ampl_", "ampl_")):
        mk(f"ff_trapez_{sfx_t}", "trapez", ampl, base, s_tr)
        mk(f"ff_symm__trapez_{sfx_t}", "symm", ampl, sym, s_sym)
        mk(f"ff_general__trapez_{sfx_t}", "general", ampl, gen, s_gen)
        mk(f"ff_symm__general__trapez_{sfx_t}", "symm_general", ampl, sgen, s_sgen)
    # smoothed symmetric trapezoid
    sm = ["A", "center", "w1", "w2", "sigma", "c0"]
    s_sm = {0: (0.5, 3.0, "lin"), 1: (-1.0, 1.0, "lin"), 2: (0.5, 3.0, "lin"), 3: (0.5, 3.0, "lin"),
            4: (0.05, 1.0, "log"), 5: (0.0, 0.2, "lin")}

    def sm_shape(p):
        return _SmoothTrapez(p[1], p[2], p[3], p[4])

    def sm_area(q, p):
        """SASfit ff_smoothed_symm_trapez_area: trapezoid (box W1 * box W2) smoothed with a Gaussian sigma."""
        return p[0] * sm_shape(p).pdf(q) + p[5]

    def sm_ampl(q, p):
        """SASfit ff_smoothed_symm_trapez_ampl: height A at the centre."""
        d = sm_shape(p)
        return p[0] * d.pdf(q) / float(d.pdf(np.array([d.c]))[0]) + p[5]

    def sm_v(u, p, dist):
        return _stat(sm_shape(p), u, dist)

    out.append(make_spec("ff_smoothed_symm_trapez_area", sm_area, sm, s_sm, volume=sm_v, group="peak",
                         source="plugins/peak_trapezoidal/sasfit_ff_smoothed_symm_trapez_area.c", rtol=1e-10))
    out.append(make_spec("ff_smoothed_symm_trapez_ampl", sm_ampl, sm, s_sm, volume=sm_v, group="peak",
                         source="plugins/peak_trapezoidal/sasfit_ff_smoothed_symm_trapez_ampl.c", rtol=1e-10))
    # SASfit names: "_area" (no trailing underscore) but "_ampl_"
    for d in out:
        d["name"] = d["name"].replace("_area_", "_area") if d["name"].endswith("_area_") else d["name"]
    return out


# ============================================================================== q_distribution
def _eq(q, x):
    x = np.asarray(x, float)
    if q == 1:
        return np.exp(x)
    b = 1.0 + (1.0 - q) * x
    with np.errstate(invalid="ignore", divide="ignore", over="ignore"):
        return np.where(b > 0, np.abs(b) ** (1.0 / (1.0 - q)), 0.0)


def _lnq(q, x):
    x = np.asarray(x, float)
    with np.errstate(invalid="ignore", divide="ignore"):
        if q == 1:
            return np.log(x)
        return (x ** (1.0 - q) - 1.0) / (1.0 - q)


def _Cq(q):
    if q == 1:
        return math.sqrt(math.pi)
    if q < 1:
        return 2 * math.sqrt(math.pi) * math.gamma(1 / (1 - q)) / ((3 - q) * math.sqrt(1 - q) * math.gamma((3 - q) / (2 * (1 - q))))
    if q < 3:
        return math.sqrt(math.pi) * math.gamma((3 - q) / (2 * (q - 1))) / (math.sqrt(q - 1) * math.gamma(1 / (q - 1)))
    return math.nan


class _QExp(_Dist):
    def __init__(self, q, lam, shift):
        self.qq, self.lam, self.s = float(q), float(lam), float(shift)
        self.lo = self.s
        self.hi = self.s + 1.0 / (self.lam * (1 - self.qq)) if self.qq < 1 else np.inf
        self.native = "quantile"

    def pdf(self, x):
        z = np.asarray(x, float) - self.s
        ok = (z >= 0) & ((z < 1 / (self.lam * (1 - self.qq))) if self.qq < 1 else True)
        return np.where(ok, (2 - self.qq) * self.lam * _eq(self.qq, -self.lam * np.maximum(z, 0)), 0.0)

    def cdf(self, x):
        z = np.maximum(np.asarray(x, float) - self.s, 0.0)
        qp = 1 / (2 - self.qq)
        return 1 - _eq(qp, -self.lam * z / qp)

    def Q(self, y):
        y = np.asarray(y, float)
        qp = 1 / (2 - self.qq)
        return -qp * _lnq(qp, 1 - y) / self.lam + self.s

    def q(self, y):
        y = np.asarray(y, float)
        qp = 1 / (2 - self.qq)
        return qp / self.lam * (1 - y) ** (-qp)

    def mode(self):
        return self.s

    def mean(self):
        return self.s + 1 / (self.lam * (3 - 2 * self.qq)) if self.qq < 1.5 else np.nan

    def var(self):
        q = self.qq
        if q >= 1.5:
            return np.nan
        if q >= 4 / 3:
            return np.inf
        return (q - 2) / ((2 * q - 3) ** 2 * (3 * q - 4) * self.lam**2)


class _QGauss(_Dist):
    def __init__(self, q, beta, shift):
        self.qq, self.beta, self.s = float(q), abs(float(beta)), float(shift)
        q, b = self.qq, self.beta
        if q < 1:
            w = 1 / math.sqrt(b * (1 - q))
            self.lo, self.hi = self.s - w, self.s + w
            a = 1 / (1 - q) + 1
            self._sp = stats.beta(a, a, loc=self.s - w, scale=2 * w)
        elif q == 1:
            self._sp = stats.norm(loc=self.s, scale=1 / math.sqrt(2 * b))
        else:
            nu = (3 - q) / (q - 1)
            self._sp = stats.t(nu, loc=self.s, scale=1 / math.sqrt(b * (3 - q)))

    def pdf(self, x):
        z = np.asarray(x, float) - self.s
        q, b = self.qq, self.beta
        v = math.sqrt(b) / _Cq(q) * _eq(q, -b * z * z)
        if q < 1:
            v = np.where(np.abs(z) > 1 / math.sqrt(b * (1 - q)), 0.0, v)
        return v

    def cdf(self, x):
        return self._sp.cdf(np.asarray(x, float))

    def Q(self, y):
        return self._sp.ppf(np.asarray(y, float))

    def mode(self):
        return self.s

    def median(self):
        return self.s

    def mean(self):
        return self.s if self.qq < 2 else np.nan

    def var(self):
        q = self.qq
        if q < 5 / 3:
            return 1 / (self.beta * (5 - 3 * q))
        return np.inf if q < 2 else np.nan


class _QWeibull(_Dist):
    native = "quantile"

    def __init__(self, q, lam, kap, shift):
        self.qq, self.lam, self.kap, self.s = float(q), abs(float(lam)), abs(float(kap)), float(shift)
        self.lo = self.s
        self.hi = self.s + self.lam / (1 - self.qq) ** (1 / self.kap) if self.qq < 1 else np.inf

    def pdf(self, x):
        z = np.asarray(x, float) - self.s
        q, l, k = self.qq, self.lam, self.kap
        ok = (z > 0) & ((z < l / (1 - q) ** (1 / k)) if q < 1 else True)
        zz = np.where(ok, z, 1.0)
        return np.where(ok, (2 - q) * k / l * (zz / l) ** (k - 1) * _eq(q, -((zz / l) ** k)), 0.0)

    def cdf(self, x):
        z = np.maximum(np.asarray(x, float) - self.s, 0.0)
        qp = 1 / (2 - self.qq)
        lp = self.lam / (2 - self.qq) ** (1 / self.kap)
        return 1 - _eq(qp, -((z / lp) ** self.kap))

    def Q(self, y):
        y = np.asarray(y, float)
        qp = 1 / (2 - self.qq)
        lp = self.lam / (2 - self.qq) ** (1 / self.kap)
        return lp * np.maximum(-_lnq(qp, 1 - y), 0.0) ** (1 / self.kap) + self.s

    def q(self, y):
        with np.errstate(divide="ignore", invalid="ignore"):
            return 1.0 / self.pdf(self.Q(y))

    def mode(self):
        q, k = self.qq, self.kap
        if k <= 1:
            return self.s
        t = (k - 1) / (k + (k - 1) * (1 - q))
        return self.s + self.lam * t ** (1 / k)

    def peak(self):
        k = self.kap
        if k < 1:
            return np.inf
        if k == 1:
            return (2 - self.qq) / self.lam
        return float(self.pdf(np.array([self.mode()]))[0])


def _q_specs():
    out = []

    def add(name, cls, pars, sample, ampl):
        def shape(p, cls=cls):
            if cls is _QWeibull:
                return _QWeibull(p[1], p[2], p[3], p[4])
            return cls(p[1], p[2], p[4])

        def inten(x, p, ampl=ampl):
            d = shape(p)
            v = d.pdf(x)
            if ampl:
                pk = d.peak() if hasattr(d, "peak") else float(d.pdf(np.array([d.mode()]))[0])
                if isinstance(d, _QExp):
                    pk = (2 - d.qq) * d.lam
                return p[0] * v / pk + p[5]
            return p[0] * v + p[5]

        def vol(u, p, dist):
            return _stat(shape(p), u, dist)

        inten.__doc__ = f"SASfit {name} ({'height' if ampl else 'area'} A, background y0)."
        out.append(make_spec(name, inten, pars, sample, volume=vol, group="peak", rtol=1e-12,
                             source=f"plugins/q_distribution/sasfit_{name}.c"))  # fmt: skip

    s_e = {0: (0.5, 3.0, "lin"), 1: (0.2, 1.8, "lin"), 2: (0.3, 3.0, "log"), 4: (0.0, 1.0, "lin"), 5: (0.0, 0.2, "lin")}
    s_g = {0: (0.5, 3.0, "lin"), 1: (0.2, 2.8, "lin"), 2: (0.3, 3.0, "log"), 4: (-1.0, 1.0, "lin"), 5: (0.0, 0.2, "lin")}
    s_w = {0: (0.5, 3.0, "lin"), 1: (0.2, 1.8, "lin"), 2: (0.3, 3.0, "log"), 3: (1.0, 4.0, "lin"), 4: (0.0, 1.0, "lin"),
           5: (0.0, 0.2, "lin")}
    for ampl, sfx in ((False, "area_"), (True, "ampl_")):
        add(f"ff_q_exponential__{sfx}", _QExp, ["A", "q", "lambda", "dummy", "shift", "y0"], s_e, ampl)
        add(f"ff_q_gaussian__{sfx}", _QGauss, ["A", "q", "beta", "dummy", "shift", "y0"], s_g, ampl)
        add(f"ff_q_weibull__{sfx}", _QWeibull, ["A", "q", "lambda", "kappa", "shift", "y0"], s_w, ampl)
    return out


# ============================================================================== kappa_generalized_distribution
def _exp_kappa(k, x):
    x = np.asarray(x, float)
    if k == 0:
        return np.exp(x)
    return np.exp(np.arcsinh(k * x) / k)


class _KappaGamma(_Dist):
    def __init__(self, sigma, xi, kap, alpha, beta, nu):
        self.sig, self.xi, self.k, self.a, self.b, self.nu = map(float, (sigma, xi, kap, alpha, beta, nu))
        self.lo = self.xi
        k, nu, a, b = self.k, self.nu, self.a, self.b
        # p ~ z^(a nu - 1 - a/k) for z -> inf (kappa-exponential tail): E[z^m] exists for m < a (1/k - nu)
        self.kmax = a * (1 / k - nu) if k > 0 else math.inf
        if k == 0:
            self.norm = abs(a) * b**nu / math.gamma(nu)
        else:
            self.norm = (1 + k * nu) * (2 * k * b) ** nu * a * math.exp(
                special.gammaln(1 / (2 * k) + nu / 2) - special.gammaln(1 / (2 * k) - nu / 2) - special.gammaln(nu))

    def pdf(self, x):
        z = (np.asarray(x, float) - self.xi) / self.sig
        ok = z > 0
        zz = np.where(ok, z, 1.0)
        with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
            lz = np.log(zz)
            t = -self.b * np.exp(self.a * lz)                      # -beta z^alpha
            lek = t if self.k == 0 else np.arcsinh(self.k * t) / self.k
            v = np.exp(math.log(self.norm / self.sig) + (self.a * self.nu - 1) * lz + lek)
        return np.where(ok & np.isfinite(v), v, 0.0)

    def _scale(self):
        return self.sig * (1 + (self.nu / self.b) ** (1 / self.a))

    def mode(self):
        k, nu, a, b = self.k, self.nu, self.a, self.b
        c = nu - 1 / a
        if c <= 0:
            return self.xi
        den = 1 - (k * c) ** 2
        if den <= 0:
            return self.xi
        return self.xi + self.sig * (c / (b * math.sqrt(den))) ** (1 / a)

    def _bracket(self):
        return self.xi, self.xi + 5 * self._scale()


class _KappaLogistic(_Dist):
    def __init__(self, sigma, xi, kap, alpha, beta, lam):
        self.sig, self.xi, self.k, self.a, self.b, self.lam = map(float, (sigma, xi, kap, alpha, beta, lam))
        self.lo = self.xi
        self.kmax = self.a / self.k if self.k > 0 else math.inf      # p ~ z^(-1 - a/k) for z -> inf

    def pdf(self, x):
        z = (np.asarray(x, float) - self.xi) / self.sig
        ok = z > 0
        zz = np.where(ok, z, 1.0)
        k, a, b, l = self.k, self.a, self.b, self.lam
        e = _exp_kappa(k, -b * zz**a)
        v = l * a * b * zz ** (a - 1) / np.sqrt(1 + (k * b * zz**a) ** 2) * e / (1 + (l - 1) * e) ** 2 / self.sig
        return np.where(ok, v, 0.0)

    def cdf(self, x):
        z = (np.asarray(x, float) - self.xi) / self.sig
        e = _exp_kappa(self.k, -self.b * np.maximum(z, 0) ** self.a)
        return np.where(z > 0, (1 - e) / (1 + (self.lam - 1) * e), 0.0)

    def Q(self, y):
        y = np.asarray(y, float)
        l, k, a, b = self.lam, self.k, self.a, self.b
        with np.errstate(divide="ignore", invalid="ignore"):
            e = (1 - y) / (1 + (l - 1) * y)            # e_kappa(-b z^a)
            lnk = np.log(e) if k == 0 else np.sinh(k * np.log(e)) / k
            return self.xi + self.sig * np.maximum(-lnk / b, 0.0) ** (1 / a)

    def _scale(self):
        return self.sig * (1 / self.b) ** (1 / self.a) * 3

    def mode(self):
        f = lambda t: -float(self.pdf(np.array([self.xi + self.sig * math.exp(t)]))[0])  # noqa: E731
        ts = np.linspace(-20, 10, 301)
        v = np.array([f(t) for t in ts])
        k = int(np.argmin(v))
        if k == 0:
            return self.xi
        if k == ts.size - 1:
            return self.xi + self.sig * math.exp(ts[-1])
        r = optimize.minimize_scalar(f, bounds=(ts[k - 1], ts[k + 1]), method="bounded", options={"xatol": 1e-12})
        return self.xi + self.sig * math.exp(r.x)

    def _bracket(self):
        return self.xi, self.xi + 5 * self._scale()


def _kappa_specs():
    out = []
    pars_g = ["A", "sigma", "xi", "kappa", "alpha", "beta", "nu"]
    pars_l = ["A", "sigma", "xi", "kappa", "alpha", "beta", "lambda"]
    s_g = {0: (0.5, 3.0, "lin"), 1: (0.5, 2.0, "lin"), 2: (-0.5, 0.5, "lin"), 3: (0.0, 0.6, "lin"), 4: (1.0, 3.0, "lin"),
           5: (0.5, 2.0, "lin"), 6: (1.0, 1.6, "lin")}
    s_l = {**s_g, 6: (0.5, 3.0, "lin")}
    for name, cls, pars, smp in (("ff_kappa_generalized_gamma_distribution__", _KappaGamma, pars_g, s_g),
                                 ("ff_kappa_logistic_distribution__", _KappaLogistic, pars_l, s_l)):
        def shape(p, cls=cls):
            return cls(p[1], p[2], p[3], p[4], p[5], p[6])

        def area(x, p, shape=shape):
            return p[0] * shape(p).pdf(x)

        def ampl(x, p, shape=shape):
            d = shape(p)
            m = d.mode()
            lo = getattr(d, "lo", -math.inf)
            if m <= lo:                        # maximum at the lower end: limit x -> lo+
                m = lo + max(abs(lo), getattr(d, "sig", 1.0)) * 1e-12
            pk = float(d.pdf(np.array([m]))[0])
            return p[0] * d.pdf(x) / pk

        def vol(u, p, dist, shape=shape):
            return _stat(shape(p), u, dist)

        area.__doc__ = f"SASfit {name}area_ (area A)."
        ampl.__doc__ = f"SASfit {name}ampl_ (height A at the mode)."
        out.append(make_spec(name + "area_", area, pars, smp, volume=vol, group="peak", rtol=1e-12,
                             source=f"plugins/kappa_generalized_distribution/sasfit_{name}area_.c"))
        out.append(make_spec(name + "ampl_", ampl, pars, smp, volume=vol, group="peak", rtol=1e-12,
                             source=f"plugins/kappa_generalized_distribution/sasfit_{name}ampl_.c",
                             c_reference=cls is _KappaGamma,
                             notes="C normalises the logistic by the density at the mode of the kappa-gamma "
                                   "distribution (wrong point); the port uses the true maximum" if cls is _KappaLogistic else ""))  # fmt: skip
    return out


# ============================================================================== generalized_distributions
class _Scipy(_Dist):
    """Wrapper of a frozen scipy.stats distribution (affine transformed)."""

    def __init__(self, sp, mode=None):
        self._sp = sp
        self.lo, self.hi = sp.support()
        self._m = mode

    def pdf(self, x):
        return self._sp.pdf(np.asarray(x, float))

    def cdf(self, x):
        return self._sp.cdf(np.asarray(x, float))

    def Q(self, y):
        return self._sp.ppf(np.asarray(y, float))

    def mean(self):
        m = float(self._sp.mean())
        return m if np.isfinite(m) else np.nan

    def var(self):
        return float(self._sp.var())

    def mode(self):
        if self._m is not None:
            return float(self._m)
        lo, hi = self._sp.ppf(1e-6), self._sp.ppf(1 - 1e-6)
        xs = np.linspace(lo, hi, 2001)
        v = self._sp.pdf(xs)
        k = int(np.argmax(v))
        a, b = xs[max(k - 1, 0)], xs[min(k + 1, xs.size - 1)]
        r = optimize.minimize_scalar(lambda x: -self._sp.pdf(x), bounds=(a, b), method="bounded",
                                     options={"xatol": 1e-13 * max(1.0, abs(xs[k]))})  # fmt: skip
        return float(r.x)


class _GEV(_Scipy):
    def __init__(self, sigma, xi, mu):
        s, x, m = abs(float(sigma)), float(xi), float(mu)
        mode = m + s * ((1 + x) ** (-x) - 1) / x if x != 0 else m
        super().__init__(stats.genextreme(-x, loc=m, scale=s), mode)


class _GH(_Scipy):
    def __init__(self, lam, alpha, beta, delta, mu):
        lam, a, b, d, m = map(float, (lam, alpha, beta, delta, mu))
        super().__init__(stats.genhyperbolic(lam, a * d, b * d, loc=m, scale=d))


class _GIG(_Scipy):
    def __init__(self, theta, eta, p):
        th, et = abs(float(theta)), abs(float(eta))
        p = float(p)
        mode = et * ((p - 1) + math.sqrt((p - 1) ** 2 + th * th)) / th
        super().__init__(stats.geninvgauss(p, th, scale=et), mode)


class _GenNormal(_Dist):
    """Generalised normal, version 2 (skewed): x = xi + alpha (1 - exp(-kappa y)) / kappa, y ~ N(0, 1)."""

    def __init__(self, alpha, kappa, xi):
        self.a, self.k, self.xi = abs(float(alpha)), float(kappa), float(xi)
        if self.k > 0:
            self.hi = self.xi + self.a / self.k
        elif self.k < 0:
            self.lo = self.xi + self.a / self.k

    def _y(self, x):
        x = np.asarray(x, float)
        if self.k == 0:
            return (x - self.xi) / self.a
        with np.errstate(invalid="ignore", divide="ignore"):
            return -np.log(1 - self.k * (x - self.xi) / self.a) / self.k

    def pdf(self, x):
        x = np.asarray(x, float)
        y = self._y(x)
        with np.errstate(invalid="ignore", divide="ignore"):
            v = np.exp(-0.5 * y * y) / math.sqrt(2 * math.pi) / (self.a - self.k * (x - self.xi))
        inside = np.isfinite(y)
        return np.where(inside, v, 0.0)

    def cdf(self, x):
        y = self._y(x)
        x = np.asarray(x, float)
        v = special.ndtr(np.where(np.isfinite(y), y, 0.0))
        if self.k > 0:
            v = np.where(x >= self.hi, 1.0, v)
        elif self.k < 0:
            v = np.where(x <= self.lo, 0.0, v)
        return v

    def Q(self, p):
        y = special.ndtri(np.asarray(p, float))
        if self.k == 0:
            return self.xi + self.a * y
        return self.xi + self.a * (-np.expm1(-self.k * y)) / self.k

    def mode(self):
        k = self.k
        return self.xi + (self.a * (-math.expm1(-k * k)) / k if k != 0 else 0.0)

    def mean(self):
        k = self.k
        return self.xi - self.a / k * math.expm1(k * k / 2) if k != 0 else self.xi

    def var(self):
        k = self.k
        return self.a**2 / k**2 * math.exp(k * k) * math.expm1(k * k) if k != 0 else self.a**2

    def median(self):
        return self.xi


class _ExpGPareto(_Dist):
    """Y = log X, X ~ generalised Pareto (scale sigma, shape xi)."""

    def __init__(self, sigma, xi):
        self.sig, self.xi = abs(float(sigma)), float(xi)
        self._gp = stats.genpareto(self.xi, scale=self.sig)
        if self.xi < 0:
            self.hi = math.log(-self.sig / self.xi)

    def pdf(self, x):
        x = np.asarray(x, float)
        s, xi = self.sig, self.xi
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            if xi == 0:
                v = np.exp(x - np.exp(x / s)) / s
            else:
                v = np.exp(x) / s * (1 + xi / s * np.exp(x)) ** (-1 / xi - 1)
                big = ~np.isfinite(v)
                v = np.where(big, np.exp(-x / xi) / s * (xi / s) ** (-1 / xi - 1), v)
        if xi < 0:
            v = np.where(x > self.hi, 0.0, v)
        return np.where(np.isfinite(v), v, 0.0)

    def cdf(self, x):
        return self._gp.cdf(np.exp(np.asarray(x, float)))

    def Q(self, y):
        with np.errstate(divide="ignore"):
            return np.log(self._gp.ppf(np.asarray(y, float)))

    def q(self, y):
        y = np.asarray(y, float)
        with np.errstate(divide="ignore", invalid="ignore"):
            x = self._gp.ppf(y)
            return 1.0 / (self._gp.pdf(x) * x)

    def mode(self):
        return math.log(self.sig) if self.xi > -1 else self.hi

    def moment(self, k):
        g = lambda y: float(self.Q(np.array([y]))[0]) ** k  # noqa: E731
        v = integrate.quad(g, 0, 0.5, limit=500, epsrel=1e-12)[0] + integrate.quad(g, 0.5, 1, limit=500, epsrel=1e-12)[0]
        return v

    def var(self):
        m = self.mean()
        g = lambda y: (float(self.Q(np.array([y]))[0]) - m) ** 2  # noqa: E731
        return integrate.quad(g, 0, 0.5, limit=500, epsrel=1e-12)[0] + integrate.quad(g, 0.5, 1, limit=500, epsrel=1e-12)[0]


def _generalized_specs():
    out = []
    defs = (
        ("exponentiated_generalized_pareto", ["A", "sigma", "xi", "dummy", "dummy", "y0"],
         lambda p: _ExpGPareto(p[1], p[2]), 5,
         {0: (0.5, 3.0, "lin"), 1: (0.5, 3.0, "log"), 2: (-0.5, 1.0, "lin"), 5: (0.0, 0.2, "lin")}),
        ("generalized_extreme_value", ["A", "sigma", "xi", "mu", "dummy", "y0"], lambda p: _GEV(p[1], p[2], p[3]), 5,
         {0: (0.5, 3.0, "lin"), 1: (0.5, 3.0, "log"), 2: (-0.5, 0.5, "lin"), 3: (-1.0, 1.0, "lin"), 5: (0.0, 0.2, "lin")}),
        ("generalized_hyperbolic", ["A", "lambda", "alpha", "beta", "delta", "mu", "y0"],
         lambda p: _GH(p[1], p[2], p[3], p[4], p[5]), 6,
         {0: (0.5, 3.0, "lin"), 1: (-1.0, 2.0, "lin"), 2: (1.0, 3.0, "lin"), 3: (-0.8, 0.8, "lin"), 4: (0.5, 2.0, "lin"),
          5: (-1.0, 1.0, "lin"), 6: (0.0, 0.2, "lin")}),
        ("generalized_inverse_gaussian", ["A", "theta", "eta", "p", "dummy", "y0"], lambda p: _GIG(p[1], p[2], p[3]), 5,
         {0: (0.5, 3.0, "lin"), 1: (0.5, 3.0, "log"), 2: (0.5, 3.0, "log"), 3: (-2.0, 3.0, "lin"), 5: (0.0, 0.2, "lin")}),
        ("generalized_normal", ["A", "alpha", "kappa", "xi", "dummy", "y0"], lambda p: _GenNormal(p[1], p[2], p[3]), 5,
         {0: (0.5, 3.0, "lin"), 1: (0.5, 3.0, "log"), 2: (-0.6, 0.6, "lin"), 3: (-1.0, 1.0, "lin"), 5: (0.0, 0.2, "lin")}),
    )
    for base, pars, shape, iy0, smp in defs:
        def area(x, p, shape=shape, iy0=iy0):
            return p[0] * shape(p).pdf(x) + p[iy0]

        def ampl(x, p, shape=shape, iy0=iy0):
            d = shape(p)
            m = d.mode()
            lo = getattr(d, "lo", -math.inf)
            if m <= lo:                        # maximum at the lower end: limit x -> lo+
                m = lo + max(abs(lo), getattr(d, "sig", 1.0)) * 1e-12
            pk = float(d.pdf(np.array([m]))[0])
            return p[0] * d.pdf(x) / pk + p[iy0]

        def vol(u, p, dist, shape=shape):
            return _stat(shape(p), u, dist)

        area.__doc__ = f"SASfit ff_{base}__area_ (area A, background y0)."
        ampl.__doc__ = f"SASfit ff_{base}__ampl_ (height A at the mode, background y0); the C function is a stub (0)."
        out.append(make_spec(f"ff_{base}__area_", area, pars, smp, volume=vol, group="peak", rtol=1e-10,
                             source=f"plugins/generalized_distributions/sasfit_ff_{base}__area_.c"))
        out.append(make_spec(f"ff_{base}__ampl_", ampl, pars, smp, volume=vol, group="peak", c_reference=False,
                             source=f"plugins/generalized_distributions/sasfit_ff_{base}__ampl_.c",
                             notes="C function is a stub returning 0; port = area version normalised to height A"))
    return out


# ============================================================================== metalog
def _metalog_M(y, a):
    """M(y) and M'(y) of the 10-term metalog with coefficients a[0..9] (SASfit Mk, 1/mk)."""
    y = np.asarray(y, float)
    with np.errstate(divide="ignore", invalid="ignore"):
        L = np.log(y / (1 - y))
        c = y - 0.5
        g = y * (1 - y)
        M = a[0] + a[1] * L + a[2] * c * L + a[3] * c
        dM = a[1] / g + a[2] * (c / g + L) + a[3]
        for k in range(5, 11):
            if k % 2:
                M = M + a[k - 1] * c ** ((k - 1) // 2)
                dM = dM + a[k - 1] * (k - 1) / 2 * c ** ((k - 3) // 2)
            else:
                M = M + a[k - 1] * c ** (k // 2 - 1) * L
                dM = dM + a[k - 1] * (c ** (k // 2 - 1) / g + (k // 2 - 1) * c ** (k // 2 - 2) * L)
    return M, dM


class _Metalog(_Dist):
    native = "quantile"

    def __init__(self, a, bl, bu, kind):
        self.a, self.bl, self.bu, self.kind = np.asarray(a, float), float(bl), float(bu), kind
        if kind == "log":
            self.lo = self.bl
        elif kind == "logit":
            self.lo, self.hi = self.bl, self.bu

    def Q(self, y):
        y = np.asarray(y, float)
        M, _ = _metalog_M(np.clip(y, 1e-300, 1 - 1e-16), self.a)
        with np.errstate(over="ignore", invalid="ignore"):
            if self.kind == "plain":
                v = M
            elif self.kind == "log":
                v = self.bl + np.exp(M)
            else:
                e = np.exp(M)
                v = np.where(np.isfinite(e), (self.bl + self.bu * e) / (1 + e), self.bu)
        if self.kind == "plain":
            v = np.where(y <= 0, -np.inf, np.where(y >= 1, np.inf, v))
        elif self.kind == "log":
            v = np.where(y <= 0, self.bl, np.where(y >= 1, np.inf, v))
        else:
            v = np.where(y <= 0, self.bl, np.where(y >= 1, self.bu, v))
        return v

    def q(self, y):
        y = np.asarray(y, float)
        M, dM = _metalog_M(y, self.a)
        with np.errstate(over="ignore", invalid="ignore"):
            if self.kind == "plain":
                v = dM
            elif self.kind == "log":
                v = dM * np.exp(M)
            else:
                e = np.exp(-np.abs(M))
                v = dM * (self.bu - self.bl) * e / (1 + e) ** 2
        return np.where((y > 0) & (y < 1), v, np.inf)

    def pdf(self, x):
        return self._pdf_from_quantile(x)


def _metalog_specs():
    out = []
    pars = ["N", "bl", "bu", "alpha"] + [f"a{i}" for i in range(1, 11)]
    smp = {0: (0.5, 2.0, "lin"), 1: (0.0, 1.0, "lin"), 2: (20.0, 30.0, "lin"), 3: (0.0, 1.0, "lin"), 4: (8.0, 12.0, "lin"),
           5: (0.5, 1.5, "lin"), 6: (-0.2, 0.2, "lin"), 7: (-0.5, 0.5, "lin")}
    smp_log = {**smp, 4: (1.5, 2.5, "lin"), 5: (0.1, 0.3, "lin"), 6: (-0.02, 0.02, "lin"), 7: (-0.05, 0.05, "lin")}
    for name, kind, s in (("sd_metalog_clipped", "plain", smp), ("sd_metalog_log_clipped", "log", smp_log),
                          ("sd_metalog_logit", "logit", smp_log)):
        def shape(p, kind=kind):
            return _Metalog(p[4:14], min(p[1], p[2]), max(p[1], p[2]), kind)

        def inten(x, p, shape=shape):
            x = np.asarray(x, float)
            bl, bu = min(p[1], p[2]), max(p[1], p[2])
            d = shape(p)
            inside = (x > bl) & (x < bu)
            xx = np.where(inside, x, 0.5 * (bl + bu) if np.isfinite(bu) else bl + 1.0)
            with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
                w = xx ** (-p[3])
                v = p[0] * d.pdf(xx) * w
            return np.where(inside & np.isfinite(w), v, 0.0)

        def vol(u, p, dist, shape=shape):
            return _stat(shape(p), u, dist, scale=p[0])

        inten.__doc__ = f"SASfit {name}: N p(x) x^-alpha inside (bl, bu), metalog with 10 coefficients a1 ... a10."
        out.append(make_spec(name, inten, pars, s, volume=vol, group="size distribution", rtol=1e-9,
                             source=f"plugins/metalog/sasfit_{name}.c"))  # fmt: skip
    return out


# ============================================================================== t_r_y
class _FGLD(_Dist):
    native = "quantile"

    def __init__(self, kappa, alpha, beta, delta):
        self.k, self.al, self.be = abs(float(kappa)), float(alpha), float(beta)
        self.d = (math.atan(float(delta)) + math.pi / 2) / math.pi

    def Q(self, y):
        y = np.asarray(y, float)
        yy = np.where(y <= 0, np.finfo(float).tiny, np.where(y >= 1, 1 - _NRIQ, y))
        return self.al + self.be * ((1 - self.d) * np.log(yy) - self.d * np.log1p(-yy) + self.k * yy)

    def q(self, y):
        y = np.asarray(y, float)
        with np.errstate(divide="ignore"):
            v = self.be * ((1 - self.d) / y + self.d / (1 - y) + self.k)
        return np.where((y > 0) & (y < 1), v, np.inf)

    def pdf(self, x):
        return self._pdf_from_quantile(x)

    def mean(self):
        return self.al + self.be * (2 * self.d - 1 + self.k / 2)

    def var(self):
        d, k = self.d, self.k
        return self.be**2 * (1 - 4 * d * (1 - d) + k / 2 * (1 + k / 6) + math.pi**2 / 3 * d * (1 - d))

    def bowley(self):
        return (2 * self.d - 1) * math.log(4 / 3) / (math.log(3) + self.k / 2)

    def moors(self):
        return (math.log(21 / 5) + self.k / 2) / (math.log(3) + self.k / 2)


class _EGUWL(_Dist):
    native = "quantile"

    def __init__(self, k, alpha, beta, c, lam):
        self.k, self.a, self.b, self.c, self.l = abs(float(k)), abs(float(alpha)), abs(float(beta)), abs(float(c)), abs(float(lam))
        self.lo = 0.0

    def Q(self, y):
        y = np.asarray(y, float)
        yy = np.where(y >= 1, 1 - _NRIQ, y)
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            t = -np.log(-np.expm1(np.log1p(-yy) / self.b))          # -log(1 - (1-y)^(1/b))
            v = self.l * np.log1p((math.exp(-self.k / self.c) * t) ** (-self.c)) ** (1 / self.a)
        v = np.where(np.isnan(v), np.inf, v)
        return np.where(y <= 0, 0.0, v)

    def cdf(self, x):
        x = np.asarray(x, float)
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            z = (np.maximum(x, 0) / self.l) ** self.a
            t = np.exp(-math.exp(self.k / self.c) * np.expm1(z) ** (-1 / self.c))
            F = -np.expm1(self.b * np.log1p(-t))
        return np.where(x <= 0, 0.0, np.where(np.isnan(F), np.where(z > 1, 1.0, 0.0), F))

    def pdf(self, x):
        x = np.asarray(x, float)
        a, b, c, k, l = self.a, self.b, self.c, self.k, self.l
        xx = np.where(x > 0, x, 1.0)
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            z = (xx / l) ** a
            em = np.expm1(z)
            s = math.exp(k / c) * em ** (-1 / c)
            t = np.exp(-s)
            dtdx = t * s / c * np.exp(z) / em * a * z / xx      # dt/dx
            v = b * (1 - t) ** (b - 1) * dtdx
        return np.where((x > 0) & np.isfinite(v), v, 0.0)

    def q(self, y):
        with np.errstate(divide="ignore", invalid="ignore"):
            return 1.0 / self.pdf(self.Q(y))


class _WLE(_Dist):
    native = "quantile"

    def __init__(self, theta, alpha, beta):
        self.t, self.a, self.b = float(theta), float(alpha), float(beta)
        self.lo = 0.0

    def Q(self, y):
        y = np.asarray(y, float)
        yy = np.where(y >= 1, 1 - _NRIQ, np.where(y <= 0, 0.5, y))
        t, a, b = self.t, self.a, self.b
        arg = -(t + 1) * np.exp(-((t + 1) + (-np.log1p(-yy)) ** (1 / a) / b))
        v = -(1 + t) / t - special.lambertw(arg, -1).real / t
        return np.where(y <= 0, 0.0, v)

    def cdf(self, x):
        x = np.asarray(x, float)
        tmp = -self.b * np.log1p(-_lindley_F(np.maximum(x, 0), self.t))
        return np.where(x < 0, 0.0, -np.expm1(-(tmp**self.a)))

    def pdf(self, x):
        x = np.asarray(x, float)
        t, a, b = self.t, self.a, self.b
        xx = np.maximum(x, 0)
        tmp = -b * np.log1p(-_lindley_F(xx, t))
        with np.errstate(divide="ignore", invalid="ignore"):
            v = a * b * t * t * (1 + xx) / (1 + t + t * xx) * tmp ** (a - 1) * np.exp(-(tmp**a))
        return np.where((x < 0) | ~np.isfinite(v), 0.0, v)

    def q(self, y):
        with np.errstate(divide="ignore", invalid="ignore"):
            return 1.0 / self.pdf(self.Q(y))


def _lindley_F(x, t):
    x = np.asarray(x, float)
    return np.where(x <= 0, 0.0, 1 - (1 + t * x / (t + 1)) * np.exp(-t * x))


def _KWm1(Z, t):
    return -(1 + t) / t - special.lambertw(-np.asarray(Z, float) * (t + 1) * math.exp(-(t + 1)), -1).real / t


class _NLC(_Dist):
    native = "quantile"

    def __init__(self, theta, mu, sigma):
        self.t, self.mu, self.s = float(theta), float(mu), float(sigma)
        self.lo = 0.0

    def Q(self, y):
        y = np.asarray(y, float)
        yy = np.where(y >= 1, 1 - _NRIQ, np.where(y <= 0, 0.5, y))
        QN = self.mu + self.s * special.ndtri(yy)
        v = _KWm1(0.5 - np.arctan(QN) / math.pi, self.t)
        return np.where(y <= 0, 0.0, v)

    def _tmp(self, x):
        return (np.tan(math.pi * (_lindley_F(x, self.t) - 0.5)) - self.mu) / self.s

    def cdf(self, x):
        x = np.asarray(x, float)
        return np.where(x <= 0, 0.0, special.ndtr(self._tmp(np.maximum(x, 0))))

    def pdf(self, x):
        x = np.asarray(x, float)
        t = self.t
        xx = np.maximum(x, 0)
        F = _lindley_F(xx, t)
        with np.errstate(divide="ignore", invalid="ignore"):
            v = (math.sqrt(math.pi / 2) * t * t / (self.s * (1 + t)) * (1 + xx) * np.exp(-t * xx)
                 / np.sin(math.pi * (1 - F)) ** 2 * np.exp(-self._tmp(xx) ** 2 / 2))  # fmt: skip
        return np.where((x < 0) | ~np.isfinite(v), 0.0, v)

    def q(self, y):
        with np.errstate(divide="ignore", invalid="ignore"):
            return 1.0 / self.pdf(self.Q(y))


class _NUGL(_Dist):
    """Normal-uniform{GL}: (z^l3 - (1-z)^l4)/l2 ~ N(0,1), z = (x - a)/(b - a)."""

    native = "quantile"

    def __init__(self, a, b, l2, l3, l4):
        self.a, self.b = min(a, b), max(a, b)
        self.l2, self.l3, self.l4 = float(l2), float(l3), float(l4)
        self.lo, self.hi = self.a, self.b

    def _g(self, z):
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            return (z**self.l3 - (1 - z) ** self.l4) / self.l2

    def cdf(self, x):
        x = np.asarray(x, float)
        z = np.clip((x - self.a) / (self.b - self.a), 0, 1)
        v = special.ndtr(self._g(z))
        return np.where(x <= self.a, 0.0, np.where(x >= self.b, 1.0, v))

    def pdf(self, x):
        x = np.asarray(x, float)
        a, b, l2, l3, l4 = self.a, self.b, self.l2, self.l3, self.l4
        inside = (x > a) & (x < b)
        z = np.where(inside, (x - a) / (b - a), 0.5)
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            v = (1 / (l2 * (b - a) * math.sqrt(2 * math.pi)) * (l3 * z ** (l3 - 1) + l4 * (1 - z) ** (l4 - 1))
                 * np.exp(-0.5 * self._g(z) ** 2))  # fmt: skip
        return np.where(inside & np.isfinite(v), v, 0.0)

    def Q(self, y):
        y = np.atleast_1d(np.asarray(y, float))
        out = np.empty(y.shape)
        for i, yi in np.ndenumerate(y):
            if yi <= 0:
                out[i] = self.a
            elif yi >= 1:
                out[i] = self.b
            else:
                t = special.ndtri(yi)
                f = lambda z: float(self._g(np.float64(z))) - t  # noqa: E731
                z = optimize.brentq(f, 1e-300, 1 - 1e-16, xtol=1e-300, rtol=1e-15, maxiter=500)
                out[i] = self.a + (self.b - self.a) * z
        return out

    def q(self, y):
        with np.errstate(divide="ignore", invalid="ignore"):
            return 1.0 / self.pdf(self.Q(y))


class _UEGL(_Dist):
    """Uniform-exponential{GL}: F = (1 + (1 - e^-theta x)^l3 - e^(-theta l4 x)) / 2, x >= 0."""

    native = "quantile"

    def __init__(self, theta, l3, l4):
        self.t, self.l3, self.l4 = float(theta), float(l3), float(l4)
        self.lo = 0.0

    def cdf(self, x):
        x = np.asarray(x, float)
        xx = np.maximum(x, 0)
        e = np.exp(-self.t * xx)
        return np.where(x <= 0, 0.0, 0.5 * (1 + (-np.expm1(-self.t * xx)) ** self.l3 - e**self.l4))

    def pdf(self, x):
        x = np.asarray(x, float)
        xx = np.maximum(x, 0)
        e = np.exp(-self.t * xx)
        with np.errstate(divide="ignore", invalid="ignore"):
            v = 0.5 * self.t * e * (self.l3 * (-np.expm1(-self.t * xx)) ** (self.l3 - 1) + self.l4 * e ** (self.l4 - 1))
        return np.where((x < 0) | ~np.isfinite(v), 0.0, v)

    def Q(self, y):
        y = np.atleast_1d(np.asarray(y, float))
        out = np.empty(y.shape)
        for i, yi in np.ndenumerate(y):
            if yi <= 0:
                out[i] = 0.0
            elif yi >= 1:
                out[i] = np.inf
            else:
                f = lambda s: float(self.cdf(np.array([s]))[0]) - yi  # noqa: E731
                hi = 1.0 / self.t
                while f(hi) < 0:
                    hi *= 2
                out[i] = optimize.brentq(f, 0.0, hi, xtol=1e-300, rtol=1e-15, maxiter=500)
        return out

    def q(self, y):
        with np.errstate(divide="ignore", invalid="ignore"):
            return 1.0 / self.pdf(self.Q(y))


def _try_specs():
    out = []

    def add(name, pars, shape, smp, clip=True, doc=""):
        def inten(x, p, shape=shape, clip=clip):
            x = np.asarray(x, float)
            v = p[0] * shape(p).pdf(x)
            if clip:
                bl, bu = min(p[1], p[2]), max(p[1], p[2])
                v = np.where((x < bl) | (x > bu), 0.0, v)
            return v

        def vol(u, p, dist, shape=shape):
            return _stat(shape(p), u, dist, scale=p[0])

        inten.__doc__ = f"SASfit {name}: {doc}"
        out.append(make_spec(name, inten, pars, smp, volume=vol, group="size distribution", rtol=1e-9,
                             source=f"plugins/t_r_y/sasfit_{name}.c"))  # fmt: skip

    add("sd_fgld_", ["N", "bl", "bu", "dummy", "kappa", "alpha", "beta", "dummy", "delta"],
        lambda p: _FGLD(p[4], p[5], p[6], p[8]),
        {0: (0.5, 2.0, "lin"), 1: (0.0, 1.0, "lin"), 2: (50.0, 60.0, "lin"), 4: (0.0, 2.0, "lin"), 5: (8.0, 12.0, "lin"),
         6: (0.5, 2.0, "lin"), 8: (-2.0, 2.0, "lin")},
        doc="flattened generalised logistic distribution (quantile defined), clipped to [bl, bu].")
    add("sd_egu_w_l", ["N", "bl", "bu", "dummy", "k", "alpha", "beta", "c", "lambda"],
        lambda p: _EGUWL(p[4], p[5], p[6], p[7], p[8]),
        {0: (0.5, 2.0, "lin"), 1: (0.0, 0.0, "lin"), 2: (50.0, 60.0, "lin"), 4: (0.2, 2.0, "lin"), 5: (1.0, 3.0, "lin"),
         6: (0.5, 3.0, "lin"), 7: (0.5, 2.0, "lin"), 8: (2.0, 10.0, "lin")}, clip=False,
        doc="exponentiated generalised uniform-Weibull{log} distribution (not clipped).")
    add("sd_w_l_e_", ["N", "bl", "bu", "dummy", "theta", "alpha", "beta"],
        lambda p: _WLE(p[4], p[5], p[6]),
        {0: (0.5, 2.0, "lin"), 1: (0.0, 0.0, "lin"), 2: (50.0, 60.0, "lin"), 4: (0.2, 2.0, "lin"), 5: (0.5, 3.0, "lin"),
         6: (0.5, 3.0, "lin")}, doc="Weibull-Lindley{exponential} distribution, clipped to [bl, bu].")
    add("sd_n_l_c_", ["N", "bl", "bu", "dummy", "theta", "dummy", "dummy", "mu", "sigma"],
        lambda p: _NLC(p[4], p[7], p[8]),
        {0: (0.5, 2.0, "lin"), 1: (0.0, 0.0, "lin"), 2: (50.0, 60.0, "lin"), 4: (0.2, 2.0, "lin"), 7: (-1.0, 1.0, "lin"),
         8: (0.3, 2.0, "lin")}, doc="normal-Lindley{Cauchy} distribution, clipped to [bl, bu].")
    add("sd_n_u_gl__", ["N", "a", "b", "dummy", "lambda2", "lambda3", "lambda4"],
        lambda p: _NUGL(p[1], p[2], p[4], p[5], p[6]),
        {0: (0.5, 2.0, "lin"), 1: (0.0, 5.0, "lin"), 2: (20.0, 30.0, "lin"), 4: (-2.0, -0.3, "lin"), 5: (-2.0, -0.2, "lin"),
         6: (-2.0, -0.2, "lin")}, clip=False, doc="normal-uniform{generalised lambda} distribution on [a, b].")
    add("sd_u_e_gl__", ["N", "dummy", "dummy", "dummy", "theta", "lambda3", "lambda4"],
        lambda p: _UEGL(p[4], p[5], p[6]),
        {0: (0.5, 2.0, "lin"), 4: (0.05, 1.0, "log"), 5: (0.5, 3.0, "lin"), 6: (0.5, 3.0, "lin")}, clip=False,
        doc="uniform-exponential{generalised lambda} distribution, x >= 0.")
    return out


SPECS = _trapez_specs() + _q_specs() + _kappa_specs() + _generalized_specs() + _metalog_specs() + _try_specs()
