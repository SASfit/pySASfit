"""Excluded-volume polymer chain (SI Sect. 4.10; Foerster & Burger, Macromolecules 1998).

Regime I  : series in x = q^2 Rg^2 (float64 with error estimate, mpmath arbitrary precision otherwise)
Regime II : asymptotic q^(-1/nu) law.
Schulz-Zimm polydispersity of Rg with relative width sigma (default weighting s=0, as in the SI code).
nu=1/2, gamma=0 gives the Gaussian (Debye) chain.
"""
import numpy as np
from scipy.special import gammaln
from .blocks import z_from_sigma

try:
    import mpmath as _mp
except Exception:  # pragma: no cover
    _mp = None


class ExcludedVolumeChain:
    def __init__(self, Rg, nu=0.588, gamma=1.1619, d=3, sigma=0.0, s=0.0):
        self.Rg, self.nu, self.gam, self.d, self.s = float(Rg), nu, gamma, d, s
        self.sigma = sigma
        self.z = z_from_sigma(sigma)
        self.delta = 1.0 / (1.0 - nu)
        self.theta = (gamma - 1.0) / nu
        dl, th = self.delta, self.theta
        self.fac = (2 + d + th) / dl
        self.Ds = ((1.0 / d) * np.exp(gammaln((4 + d + th) / dl) - gammaln(self.fac))) ** (dl / 2)
        self.Bs = (dl * np.exp(gammaln(d / 2)) / (2 * np.pi ** (d / 2))) * self.Ds ** self.fac / np.exp(gammaln(self.fac)) \
            if False else (dl * np.exp(gammaln(d / 2)) / (2 * np.pi ** (d / 2))) * (self.Ds ** self.fac / np.exp(gammaln(self.fac)))
        self.As = 2 * np.pi ** (d / 2) * self.Bs / (dl * self.Ds ** self.fac)
        self.x_series_max = 400.0

    # series coefficients (log magnitude) -----------------------------------------------------
    def _logterm(self, n, z_inf=False):
        nu, d, z, s = self.nu, self.d, (np.inf if z_inf else self.z), self.s
        n = np.asarray(n, float)
        lt = gammaln((2 + d + self.theta + 2 * n) / self.delta) - gammaln(d / 2 + n)
        lt += n * (np.log(1 + 2 * nu) + np.log(2 + 2 * nu))
        if not np.isinf(z):
            lt += gammaln(z + s + 1 + 2 * n) - gammaln(z + s + 1) - 2 * n * np.log(z + 1)
        lt -= np.log(1 + 2 * nu * n) + np.log(2 + 2 * nu * n) + gammaln(n + 1)
        lt -= n * np.log(4 * d * self.Ds ** (2 / self.delta))
        return lt

    def _terms_float(self, x, nmax=400, z_inf=False):
        """returns (sum, ok) of the power series truncated at the smallest term (asymptotic if polydisperse)."""
        n = np.arange(nmax)
        lt = self._logterm(n, z_inf=z_inf)
        sgn = (-1.0) ** n
        with np.errstate(all='ignore'):
            T = sgn[None, :] * np.exp(lt[None, :] + n[None, :] * np.log(np.maximum(x[:, None], 1e-300)))
        T = np.where(np.isfinite(T), T, np.inf)
        cum = np.cumsum(T, 1)
        cabs = np.cumsum(np.abs(T), 1)
        small = (np.abs(T) < 1e-17 * cabs) & (n[None, :] >= 4)
        idx = np.argmax(small, 1)
        has = small.any(1)
        rows = np.arange(x.size)
        S = cum[rows, idx]
        A = cabs[rows, idx]
        ok = has & (4e-16 * A < 1e-7 * np.maximum(np.abs(S), 1e-300)) & (S > 0)
        return 2 * self.As * S, ok

    def _mp_coeffs(self, nmax=2600, dps=260):
        if getattr(self, "_mpc", None) is None:
            mp = _mp
            with mp.workdps(dps):
                nu, d = mp.mpf(self.nu), self.d
                th, dl = mp.mpf(self.theta), mp.mpf(self.delta)
                Ds = mp.mpf(self.Ds)
                c = []
                for n in range(nmax):
                    nn = mp.mpf(n)
                    lt = mp.loggamma((2 + d + th + 2 * nn) / dl) - mp.loggamma(d / 2 + nn)
                    lt += nn * (mp.log(1 + 2 * nu) + mp.log(2 + 2 * nu))
                    lt -= mp.log(1 + 2 * nu * nn) + mp.log(2 + 2 * nu * nn) + mp.loggamma(nn + 1)
                    lt -= nn * mp.log(4 * d * Ds ** (2 / dl))
                    c.append((-1) ** n * mp.e ** lt)
                self._mpc = c
        return self._mpc

    def _series_mp(self, x):
        """monodisperse series (z = inf) in arbitrary precision."""
        mp = _mp
        c = self._mp_coeffs()
        with mp.workdps(260):
            xx = mp.mpf(float(x))
            tot, p = mp.mpf(0), mp.mpf(1)
            for n, cn in enumerate(c):
                t = cn * p
                tot += t
                p *= xx
                if n > 10 and abs(t) < mp.mpf(10) ** (-30) * abs(tot):
                    break
            return float(2 * self.As * tot)

    def _mono_table(self):
        if getattr(self, "_tab", None) is None:
            from scipy.interpolate import CubicSpline
            xs = np.exp(np.linspace(np.log(5.0), np.log(self.x_series_max), 200))
            if not _mp:
                raise ImportError("the excluded-volume chain needs mpmath for q²Rg² > 5 "
                                  "(pip install mpmath, or pysasfit[scatter])")
            vals = np.array([self._series_mp(v) for v in xs])
            self._tab = CubicSpline(np.log(xs), np.log(vals))
        return self._tab

    def _mono(self, x):
        x = np.asarray(x, float)
        out = np.empty(x.shape)
        lo = x <= 5.0
        if lo.any():
            v, ok = self._terms_float(x[lo].ravel(), z_inf=True)
            out[lo] = v
        mid = (x > 5.0) & (x <= self.x_series_max)
        if mid.any():
            out[mid] = np.exp(self._mono_table()(np.log(x[mid])))
        hi = x > self.x_series_max
        if hi.any():
            out[hi] = self._asym(x[hi], mono=True)
        return out

    def _asym(self, x, mono=False):
        nu, d, z, dl = self.nu, self.d, self.z, self.delta
        from scipy.special import gamma as G
        pref = self.As * G(1 / (2 * nu)) * G(self.fac - 1 / (nu * dl)) / (nu * G(d / 2 - 1 / (2 * nu)))
        arg = (1 + 2 * nu) * (2 + 2 * nu) * x / (4 * d * self.Ds ** (2 / dl))
        zz = 1.0 if (mono or np.isinf(z)) else (z + 1) / z
        return pref * zz * arg ** (-1 / (2 * nu))

    def P(self, q):
        q = np.atleast_1d(np.asarray(q, float))
        x = (q * self.Rg) ** 2
        if np.isinf(self.z):
            return self._mono(x)
        out = np.full(x.shape, np.nan)
        val, ok = self._terms_float(x)          # Regime I (asymptotic power series)
        out[ok] = val[ok]
        rest = np.where(~ok)[0]
        if rest.size:
            from .shapes import SizeQuad
            sq = SizeQuad(self.sigma, self.s)    # Schulz-Zimm weights for Rg
            xr = x[rest, None] * sq.xi[None, :] ** 2
            out[rest] = (self._mono(xr) * sq.w).sum(1)
        return out
