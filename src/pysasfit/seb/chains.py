"""Linear chains given by their two-point function.

For a homogeneous linear chain of contour length L the scattering depends only on the two-point function
G(q, s) = <sin(q r)/(q r)> of two contour points a contour distance s apart. With t = s/L:

    F         = 2 int_0^1 (1 - t) G(tL) dt
    A_end     =   int_0^1 G(tL) dt                    Psi(end1, end2)   = G(L)
    A_middle  = 2 int_0^1/2 G(tL) dt                  Psi(end, middle)  = G(L/2)
    Psi(contour, end) = A_end, Psi(contour, middle) = A_middle, Psi(contour, contour) = F

(all reference points of GaussianPolymer). The integrals are done by composite Gauss-Legendre quadrature on a grid in
z = s/L that resolves the oscillations of G at large q.

Models:

* :class:`WormLikeChain`  -- Kholodenko's worm (A. L. Kholodenko, Macromolecules 26 (1993) 4179): contour length L,
  Kuhn length b; the rod limit for s << b and the Gaussian limit (Kuhn length b) for s >> b are exact; in between
  it is Kholodenko's approximation, with <R^2(s)> = (b^2/3)(z coth z - 1), z = 3 s/b (the worm-like chain has
  b s - (b^2/2)(1 - exp(-2 s/b)); Kholodenko's is up to 20 % larger around s ~ 0.7 b, < 1 % for s > 20 b).
* :class:`WormLikeChainEV` -- worm-like chain with excluded volume: the form factor of Pedersen & Schurtenberger
  (Macromolecules 29 (1996) 7602, method 3 with excluded volume; corrections of Chen et al., Langmuir 22 (2006)
  6539), the other quantities from its dependence on L (:class:`ChainFromF`); for L < 2.5 b it crosses over smoothly
  to Kholodenko's worm (PS is not valid for L < b).
* :class:`ExcludedVolumePolymer` -- excluded-volume chain with non-Gaussian distances (des Cloizeaux, J. Phys.
  (Paris) 41 (1980) 223; Foerster & Burger, Macromolecules 31 (1998) 879): distances with
  P(r) ~ r^(2+theta) exp(-(D r)^delta), delta = 1/(1 - nu), <r^2(s)> ~ s^(2 nu); parameters Rg, nu. The
  correlation-hole exponent depends on where the two points are: theta_0 = (gamma - 1)/nu = 0.275 between the two
  ends, theta_1 = 0.46 between an end and an interior point, theta_2 = 0.71 between two interior points (self-avoiding
  walk values, scaled by (nu - 1/2)/(0.588 - 1/2) for other nu, so nu = 1/2 is the Gaussian chain). F and the
  amplitude about the middle use theta_2, the amplitude about an end and Psi(end, middle) theta_1, Psi(end1, end2)
  theta_0.
"""
from __future__ import annotations

import numpy as np
from scipy import special as sp

from .subunits import TableSubUnit

__all__ = ["TwoPointChain", "ChainFromF", "WormLikeChain", "WormLikeChainEV", "ExcludedVolumePolymer", "kholodenko_g",
           "kholodenko_r2", "kholodenko_rg2",
           "des_cloizeaux_g", "wlc_r2", "wlc_rg2"]

_GX, _GW = np.polynomial.legendre.leggauss(12)
_GX = 0.5 * (_GX + 1.0)
_GW = 0.5 * _GW


def _panel_nodes(edges):
    """Gauss-Legendre nodes and weights on consecutive panels [edges[i], edges[i+1]]."""
    a, b = edges[:-1, None], edges[1:, None]
    x = (a + (b - a) * _GX[None, :]).ravel()
    w = ((b - a) * _GW[None, :]).ravel()
    return x, w


def wlc_r2(s, b):
    """<R^2(s)> of the worm-like chain, Kuhn length b."""
    s = np.asarray(s, dtype=float)
    return b * s - 0.5 * b * b * -np.expm1(-2.0 * s / b)


def wlc_rg2(L, b):
    """Rg^2 of the worm-like chain (Benoit & Doty)."""
    P = b / 2.0
    x = L / P
    return P * L / 3.0 - P * P + 2 * P**3 / L - 2 * P**4 / L**2 * (-np.expm1(-x))


def kholodenko_r2(s, b):
    """<R^2(s)> of Kholodenko's worm: (b^2/3)(z coth z - 1), z = 3 s/b."""
    z = 3.0 * np.asarray(s, dtype=float) / b
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        v = np.where(z < 1e-3, z * z / 3.0 - z**4 / 45.0, z / np.tanh(np.maximum(z, 1e-300)) - 1.0)
    return b * b / 3.0 * v


def kholodenko_rg2(L, b, n=400):
    """Rg^2 of Kholodenko's worm, int_0^1 (1 - t) <R^2(tL)> dt (numerically)."""
    x, w = np.polynomial.legendre.leggauss(n)
    t = 0.5 * (x + 1)
    return float(np.sum(0.5 * w * (1 - t) * kholodenko_r2(t * L, b)))


def kholodenko_g(q, s, b):
    """Two-point function of Kholodenko's worm, f(z), z = 3 s/b (q: (nq, 1), s: (1, ns) arrays)."""
    q = np.asarray(q, dtype=float)
    s = np.asarray(s, dtype=float)
    z = 3.0 * s / b
    k = q * b / 3.0
    out = np.empty(np.broadcast(q, s).shape)
    zz = np.broadcast_to(z, out.shape)
    kk = np.broadcast_to(k, out.shape)
    small_z = zz < 1e-8
    den = -np.expm1(-2.0 * np.maximum(zz, 1e-300))           # 1 - exp(-2z)
    lo = (kk < 1.0) & ~small_z
    with np.errstate(over="ignore", invalid="ignore"):
        E = np.sqrt(np.clip(1.0 - kk * kk, 0.0, None))
        Ez = E * zz
        # sinh(Ez)/(E sinh z) = exp((E-1) z) (1 - exp(-2Ez)) / (E (1 - exp(-2z)))
        num = np.where(E > 1e-7, -np.expm1(-2.0 * Ez) / np.where(E > 1e-7, E, 1.0), 2.0 * zz)
        v_lo = np.exp((E - 1.0) * zz) * num / den
        Fk = np.sqrt(np.clip(kk * kk - 1.0, 0.0, None))
        v_hi = np.where(Fk > 1e-7, np.sin(Fk * zz) / np.where(Fk > 1e-7, Fk, 1.0), zz) * 2.0 * np.exp(-zz) / den
    out[lo] = v_lo[lo]
    hi = ~lo & ~small_z
    out[hi] = v_hi[hi]
    out[small_z] = 1.0
    return out


# ----------------------------------------------------------------------------------- des Cloizeaux
_DC_CACHE = {}
GAMMA_SAW = 1.1619


def _dc_table(nu, theta):
    """g(y) = <sin(y x)/(y x)> over the des Cloizeaux distribution of x = r/sqrt(<r^2>), on a log grid."""
    key = (round(float(nu), 9), round(float(theta), 9))
    if key in _DC_CACHE:
        return _DC_CACHE[key]
    delta = 1.0 / (1.0 - nu)
    D = np.exp(0.5 * (sp.gammaln((5 + theta) / delta) - sp.gammaln((3 + theta) / delta)))
    # x grid: the density ~ x^(2+theta) exp(-(Dx)^delta) is negligible beyond (60)^(1/delta)/D
    xmax = 60.0 ** (1.0 / delta) / D
    edges = np.concatenate([np.linspace(0, 0.05 * xmax, 41)[:-1], np.linspace(0.05 * xmax, xmax, 2001)])
    x, w = _panel_nodes(edges)
    logp = (2 + theta) * np.log(np.maximum(x, 1e-300)) - (D * x) ** delta
    p = np.exp(logp - logp.max())
    p = p / np.sum(p * w)
    m2 = np.sum(p * w * x * x)
    x = x / np.sqrt(m2)                        # exactly <x^2> = 1 on this grid
    w = w / np.sqrt(m2)
    p = p * np.sqrt(m2)
    ys = np.geomspace(1e-2, 150.0, 500)
    g = (np.sinc(np.outer(ys, x) / np.pi) * (p * w)[None, :]).sum(1)
    # beyond the table: g ~ K y^-(3 + theta) (small-distance behaviour of the density)
    K = g[-1] * ys[-1] ** (3 + theta)
    from scipy.interpolate import CubicSpline
    tab = (np.log(ys), CubicSpline(np.log(ys), g), theta, K)
    _DC_CACHE[key] = tab
    return tab


def des_cloizeaux_g(y, nu, theta=None, gamma=GAMMA_SAW):
    """<sin(y x)/(y x)> for the des Cloizeaux distribution with <x^2> = 1; theta defaults to (gamma - 1)/nu."""
    if theta is None:
        theta = (gamma - 1.0) / nu
    ly, g, theta, K = _dc_table(nu, theta)
    y = np.asarray(y, dtype=float)
    out = g(np.clip(np.log(np.maximum(y, 1e-300)), ly[0], ly[-1]))
    small = y < np.exp(ly[0])
    out = np.where(small, 1.0 - y * y / 6.0 + y**4 / 120.0 * _dc_m4(nu, theta), out)
    big = y > np.exp(ly[-1])
    out = np.where(big, K * np.maximum(y, 1e-300) ** (-(3 + theta)), out)
    return out


def _dc_m4(nu, theta):
    """<x^4> of the des Cloizeaux distribution with <x^2> = 1."""
    delta = 1.0 / (1.0 - nu)
    a = (3 + theta) / delta
    return np.exp(sp.gammaln(a + 4 / delta) + sp.gammaln(a) - 2 * sp.gammaln(a + 2 / delta))


# ----------------------------------------------------------------------------------- the base class
class TwoPointChain(TableSubUnit):
    """Linear chain defined by its two-point function; sub-classes give ``length(p)`` and ``G(q, s, p)``."""

    specific = ("end1", "middle", "end2")
    distributed = ("contour",)
    oscillating = False          # G oscillates in s at large q (rod-like parts): finer quadrature

    def length(self, p):
        raise NotImplementedError

    def G(self, q, s, p):
        raise NotImplementedError

    def _grid(self, q, p, upper):
        """Panels in s on [0, upper]: fine where G may oscillate, geometric beyond."""
        L = self.length(p)
        qmax = float(np.max(q)) if np.size(q) else 0.0
        if self.oscillating:
            b = self.local_length(p)
            s1 = min(upper, 13.0 * b)        # beyond, the oscillating part has decayed by exp(-39)
            n1 = int(np.clip(np.ceil(qmax * s1 / np.pi) + 8, 8, 6000))
            e1 = np.linspace(0.0, s1, n1 + 1)
            if s1 < upper:
                e2 = np.geomspace(s1, upper, 40)
                edges = np.concatenate([e1, e2[1:]])
            else:
                edges = e1
        else:
            # smooth G: crowd panels at small s (where G changes on the scale 1/q at large q)
            lo = max(upper * 1e-9, 1e-12 * L)
            edges = np.concatenate([[0.0], np.geomspace(lo, upper, 60)])
        return _panel_nodes(edges)

    def _parts(self, q, p):
        q = np.atleast_1d(np.asarray(q, dtype=float))
        if self.oscillating and q.size > 48:
            # blocks of q: each with a grid fine enough for its own largest q
            order = np.argsort(q)
            out = {k: np.empty(q.size) for k in ("F", "end", "middle", "ee", "em")}
            for blk in np.array_split(order, int(np.ceil(q.size / 48))):
                part = self._parts(q[blk], p)
                for k in out:
                    out[k][blk] = part[k]
            return out
        L = self.length(p)
        s, w = self._grid(q, p, L)
        G = self.G(q[:, None], s[None, :], p)
        t = s / L
        F = 2.0 * (G * ((1.0 - t) * w)[None, :]).sum(1) / L
        Aend = (G * w[None, :]).sum(1) / L
        sm, wm = self._grid(q, p, 0.5 * L)
        Gm = self.G(q[:, None], sm[None, :], p)
        Amid = 2.0 * (Gm * wm[None, :]).sum(1) / L
        Gee = self.G(q[:, None], np.array([[L]]), p)[:, 0]
        Gem = self.G(q[:, None], np.array([[0.5 * L]]), p)[:, 0]
        return {"F": F, "end": Aend, "middle": Amid, "ee": Gee, "em": Gem}

    def _get(self, q, p, what):
        key = (np.asarray(q).tobytes(), tuple(sorted(p.items())))
        if getattr(self, "_memo_key", None) != key:
            self._memo = self._parts(q, p)
            self._memo_key = key
        return self._memo[what]

    def ff(self, q, p):
        self.check(p)
        return self._get(q, p, "F")

    def amp(self, q, p, ref):
        self.check_ref(ref)
        self.check(p)
        if ref in ("end1", "end2"):
            return self._get(q, p, "end")
        if ref == "middle":
            return self._get(q, p, "middle")
        return self._get(q, p, "F")

    def psi(self, q, p, r1, r2):
        self.check(p)
        a, b = sorted((r1, r2))
        if r1 == r2 and r1 not in self.distributed:
            return np.ones_like(np.asarray(q, dtype=float))
        if (a, b) == ("end1", "end2"):
            return self._get(q, p, "ee")
        if b == "middle" and a in ("end1", "end2"):
            return self._get(q, p, "em")
        if a == "contour" and b == "contour":
            return self._get(q, p, "F")
        other = b if a == "contour" else a
        return self.amp(q, p, other)

    def check(self, p):
        pass


class WormLikeChain(TwoPointChain):
    """Semiflexible (worm-like) chain, Kholodenko's model: contour length L, Kuhn length b (= 2 x persistence)."""
    kind = "WormLikeChain"
    size_params = ("L", "b")
    oscillating = True

    def length(self, p):
        return p["L"]

    def local_length(self, p):
        return p["b"]

    def check(self, p):
        if p["L"] <= 0 or p["b"] <= 0:
            raise ValueError("WormLikeChain: L and b must be positive")

    def G(self, q, s, p):
        return kholodenko_g(q, s, p["b"])


class ChainFromF(TwoPointChain):
    """Linear chain known only by its form factor F(q; L) as a function of the contour length L. For a homogeneous
    chain L^2 F(L) = 2 int_0^L (L - s) G(s) ds, so G(s) = (L^2 F)''/2 at L = s and

        A_end = F + (L/2) F',  A_middle = F(L/2) + (L/4) F'(L/2),
        Psi(end1, end2) = F + 2 L F' + (L^2/2) F'',  Psi(end, middle) = the same at L/2

    (derivatives in L at fixed q, by central differences)."""

    rel_step = 2e-3

    def F_of_L(self, q, L, p, ref=None):
        """F at contour length L; ref: length at which a piecewise formula chooses its branches."""
        raise NotImplementedError

    def _derivs(self, q, L, p):
        h = self.rel_step * L
        f0 = self.F_of_L(q, L, p, ref=L)
        fp = self.F_of_L(q, L + h, p, ref=L)
        fm = self.F_of_L(q, L - h, p, ref=L)
        return f0, (fp - fm) / (2 * h), (fp - 2 * f0 + fm) / (h * h)

    def _parts(self, q, p):
        q = np.atleast_1d(np.asarray(q, dtype=float))
        L = self.length(p)
        F, d1, d2 = self._derivs(q, L, p)
        Fm, m1, m2 = self._derivs(q, 0.5 * L, p)
        return {"F": F, "end": F + 0.5 * L * d1, "middle": Fm + 0.25 * L * m1,
                "ee": F + 2 * L * d1 + 0.5 * L * L * d2, "em": Fm + L * m1 + 0.125 * L * L * m2}


class WormLikeChainEV(ChainFromF):
    """Worm-like chain with excluded volume: the form factor of Pedersen & Schurtenberger (1996, method 3 with
    excluded volume, as corrected by Chen et al. 2006); amplitudes and phase factors from its L dependence
    (:class:`ChainFromF`). Contour length L, Kuhn length b."""
    kind = "WormLikeChainEV"
    size_params = ("L", "b")

    def length(self, p):
        return p["L"]

    def check(self, p):
        if p["L"] <= 0 or p["b"] <= 0:
            raise ValueError("WormLikeChainEV: L and b must be positive")

    #: below L = BLEND[0]*b the Pedersen-Schurtenberger formula is not valid (F > 1, oscillations); there the
    #: Kholodenko worm (excluded volume negligible for such stiff chains) is used, with a smooth crossover up
    #: to L = BLEND[1]*b.  The weight depends on L, so the L-derivatives stay consistent.
    BLEND = (1.0, 2.5)

    def F_of_L(self, q, L, p, ref=None):
        from .ps_worm import sk_wr
        b = p["b"]
        x = (L / b - self.BLEND[0]) / (self.BLEND[1] - self.BLEND[0])
        w = 0.0 if x <= 0 else 1.0 if x >= 1 else x * x * x * (10 - 15 * x + 6 * x * x)   # C2 smoothstep
        out = 0.0
        if w > 0:
            out = w * sk_wr(q, L, b, ref)
        if w < 1:
            out = out + (1 - w) * WormLikeChain().ff(q, {"L": L, "b": b})
        return out


class ExcludedVolumePolymer(TwoPointChain):
    """Excluded-volume chain with the des Cloizeaux distance distribution (Foerster & Burger 1998): Rg, nu."""
    kind = "ExcludedVolumePolymer"
    size_params = ("Rg", "nu")

    def length(self, p):
        return 1.0                         # contour in units of the chain: t = s

    def check(self, p):
        if not 0.34 <= p["nu"] <= 0.95:
            raise ValueError(f"ExcludedVolumePolymer: nu = {p['nu']} outside 0.34 … 0.95")

    THETA_SAW = (0.275, 0.46, 0.71)        # end-end, end-interior, interior-interior (nu = 0.588)

    def theta(self, nu, k):
        return max(0.0, self.THETA_SAW[k] * (nu - 0.5) / (0.588 - 0.5))

    def G(self, q, s, p, k=2):
        nu = p["nu"]
        R = p["Rg"] * np.sqrt((2 * nu + 1) * (2 * nu + 2))      # sqrt(<R^2>) of the whole chain
        return des_cloizeaux_g(q * R * np.power(s, nu), nu, theta=self.theta(nu, k))

    def _parts(self, q, p):
        q = np.atleast_1d(np.asarray(q, dtype=float))
        s, w = self._grid(q, p, 1.0)
        G2 = self.G(q[:, None], s[None, :], p, 2)
        G1 = self.G(q[:, None], s[None, :], p, 1)
        F = 2.0 * (G2 * ((1.0 - s) * w)[None, :]).sum(1)
        Aend = (G1 * w[None, :]).sum(1)
        sm, wm = self._grid(q, p, 0.5)
        Amid = 2.0 * (self.G(q[:, None], sm[None, :], p, 2) * wm[None, :]).sum(1)
        Gee = self.G(q[:, None], np.array([[1.0]]), p, 0)[:, 0]
        Gem = self.G(q[:, None], np.array([[0.5]]), p, 1)[:, 0]
        return {"F": F, "end": Aend, "middle": Amid, "ee": Gee, "em": Gem}
