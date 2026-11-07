"""Particle shapes: orientation-averaged and orientation-resolved Schulz-Zimm averaged
form factors.  All results are normalised so that P(q->0) = 1 and F(q->0) = 1.

Conventions
-----------
R      mean size (scale of all dimensions; aspect ratios are fixed),
sigma  relative width of the Schulz-Zimm size distribution (0 = monodisperse),
volume weighting: P ~ V^2 -> s = 6, F ~ V -> s = 3 (all bodies are 3D).
"""
from __future__ import annotations
import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.special import roots_genlaguerre, gammaln
from .blocks import BLOCK, z_from_sigma
from . import _accel


def _gl(n, a=0.0, b=1.0):
    x, w = leggauss(n)
    return 0.5 * (b - a) * x + 0.5 * (b + a), 0.5 * (b - a) * w


def volume_ratio(z):
    """<V>^2/<V^2> for V ~ R^3 under a Schulz-Zimm distribution."""
    if np.isinf(z):
        return 1.0
    return float(np.exp(2 * (gammaln(z + 4) - gammaln(z + 1)) - (gammaln(z + 7) - gammaln(z + 1))))


class SizeQuad:
    """Nodes/weights xi=R/X of the Schulz-Zimm pdf * R^s (for generic non-block averages).

    ``ymax``: largest angular frequency of the integrand in xi (for P = |F|^2 of a body with extent R e:
    about 2 q R e).  With it a Gauss-Legendre rule over the bulk of the distribution (mean - 9 sigma ... mean
    + 14 sigma) is used, with 2.5 nodes per oscillation + 24: accurate to ~1e-10 also for wide distributions
    and large q R, where the fixed Gauss-Laguerre rule (N nodes) loses accuracy.  Without ``ymax``: the former
    rules (Gauss-Laguerre, uniform grid of Nu points for narrow distributions)."""

    def __init__(self, sigma, s, N=200, Nu=801, ymax=None):
        self.z = z_from_sigma(sigma)
        if np.isinf(self.z):
            self.xi, self.w = np.array([1.0]), np.array([1.0])
            return
        al = self.z + s
        if ymax is not None:
            c = (al + 1) / (self.z + 1)                 # mean of x^al exp(-(z+1)x)
            lo, hi = max(1e-12, c * (1 - 9 * sigma)), c * (1 + 14 * sigma)
            cyc = float(ymax) * (hi - lo) / (2 * np.pi)
            n = int(np.clip(48 + 2.5 * cyc, 48, 4000))
            x, wg = _gl(n, lo, hi)
            lp = al * np.log(x) - (self.z + 1) * x
            w = wg * np.exp(lp - lp.max())
            self.xi, self.w = x, w / w.sum()
            return
        if al < 100:
            t, w = roots_genlaguerre(N, al)
            w = w / w.sum()
            self.xi, self.w = t / (self.z + 1), w
        else:  # narrow: uniform grid over mean +- 8 sigma_eff
            sg = sigma
            xi = np.linspace(max(1e-3, 1 - 8 * sg), 1 + 8 * sg, Nu)
            lp = al * np.log(xi) - (self.z + 1) * xi
            w = np.exp(lp - lp.max())
            self.xi, self.w = xi, w / w.sum()


def _check_resolution(y_max, sigma, g=1.0, nodes=200):
    if sigma and y_max * sigma * g > nodes / 8:
        import warnings
        warnings.warn("q*R*sigma is large for the generic quadrature; results may be inaccurate")


DEFAULT_WEIGHTING = "reference"


def set_default_weighting(w):
    """'reference': no volume weighting of the Schulz-Zimm distribution (s=0), as in the authors'
    code (CrystalScatter / SI Mathematica code).  'volume': P weighted by V^2 (s=2d=6), F by V
    (s=d=3), as stated in the SI text."""
    global DEFAULT_WEIGHTING
    assert w in ("reference", "volume")
    DEFAULT_WEIGHTING = w


class Shape:
    """Base class.  Subclasses define P_fixed(q, mu) (orientation-resolved) and
    P(q), F(q) (orientation averaged)."""
    name = "shape"
    nu = 1.0

    def __init__(self, R, sigma=0.0):
        self.R, self.sigma = float(R), float(sigma)
        self.z = z_from_sigma(sigma)
        self.weighting = DEFAULT_WEIGHTING     # 'reference' (s=0) or 'volume' (s=6/3); may be changed later

    @property
    def sP(self):
        return 6 if self.weighting == "volume" else 0

    @property
    def sF(self):
        return 3 if self.weighting == "volume" else 0

    # ratio <F>^2/<P> (volume weighted, absolute)  used by the decoupling approximation
    def beta(self, q):
        vr = volume_ratio(self.z) if self.weighting == 'volume' else 1.0
        return vr * self.F(q) ** 2 / self.P(q)

    def _angles(self, y):
        ymax = float(np.max(y)) * max(1.0, self.nu)
        return int(np.clip(60 + 6 * ymax, 80, 1500))


class Sphere(Shape):
    name = "sphere"

    def P(self, q):
        return BLOCK[3].value(np.asarray(q) * self.R, self.z, self.sP, 'P')

    def F(self, q):
        return BLOCK[3].value(np.asarray(q) * self.R, self.z, self.sF, 'F')

    def P_fixed(self, q, mu):
        return self.P(q) * np.ones_like(np.asarray(mu, float))


class Ellipsoid(Shape):
    """Rotational ellipsoid with semi-axes (R, R, nu R); symmetry axis along the 3rd axis."""
    name = "ellipsoid"

    def __init__(self, R, nu, sigma=0.0):
        super().__init__(R, sigma)
        self.nu = float(nu)

    def _r(self, mu):
        return np.sqrt(1 - mu ** 2 + (self.nu * mu) ** 2)

    def P_fixed(self, q, mu):
        q = np.asarray(q, float)
        mu = np.asarray(mu, float)
        return BLOCK[3].value(q * self.R * self._r(mu), self.z, self.sP, 'P')

    def F_fixed(self, q, mu):
        q = np.asarray(q, float)
        mu = np.asarray(mu, float)
        return BLOCK[3].value(q * self.R * self._r(mu), self.z, self.sF, 'F')

    def _avg(self, q, fn):
        q = np.atleast_1d(np.asarray(q, float))
        mu, w = _gl(self._angles(q * self.R))
        return (fn(q[:, None], mu[None, :]) * w).sum(-1)

    def P(self, q):
        return self._avg(q, self.P_fixed)

    def F(self, q):
        return self._avg(q, self.F_fixed)


class TriaxialEllipsoid(Shape):
    """Semi-axes (R, b R, c R), full orientational average."""
    name = "triaxial_ellipsoid"

    def __init__(self, R, b, c, sigma=0.0):
        super().__init__(R, sigma)
        self.b, self.c, self.nu = b, c, max(1.0, b, c)

    def _eval(self, q, s, kind):
        q = np.atleast_1d(np.asarray(q, float))
        n = self._angles(q * self.R)
        n = min(n, 400)
        mu, wm = _gl(n)
        ph, wp = _gl(max(16, n // 2), 0, np.pi / 2)
        MU, PH = np.meshgrid(mu, ph, indexing='ij')
        W = np.outer(wm, wp) * 4 / (np.pi / 2 * 4) * 2 / 2  # normalise below
        W = W / W.sum()
        s2 = 1 - MU ** 2
        r = np.sqrt(s2 * (np.cos(PH) ** 2 + (self.b * np.sin(PH)) ** 2) + (self.c * MU) ** 2)
        out = np.empty(q.shape)
        for i, qi in enumerate(q):
            out[i] = (BLOCK[3].value(qi * self.R * r, self.z, s, kind) * W).sum()
        return out

    def P(self, q):
        return self._eval(q, self.sP, 'P')

    def F(self, q):
        return self._eval(q, self.sF, 'F')


class _Factorised(Shape):
    _nu_nodes = 801   # nodes of the uniform size grid for narrow distributions
    _extra = 1        # extra array axis in _amp (memory budgeting)
    """Bodies whose amplitude is a product of d=1/d=2 blocks of *different* arguments:
    the Schulz average over R is carried out by quadrature (SizeQuad) at each orientation."""

    def _amp(self, q, mu, xi):  # amplitude normalised to 1, broadcast over trailing xi axis
        raise NotImplementedError

    def _ymax(self, q):
        """Largest argument of the size integrand: q R times the largest extent (cylinder: sqrt(1 + nu^2))."""
        e = max(1.0, getattr(self, "nu", 1.0), getattr(self, "b", 1.0), getattr(self, "c", 1.0))
        k = np.sqrt(3.0) if isinstance(self, Parallelepiped) else np.sqrt(2.0)    # sum of the block arguments
        return 2.0 * float(np.max(np.abs(q))) * self.R * e * k

    def _ang_avg(self, q, sq, s_kind):
        q = np.atleast_1d(np.asarray(q, float))
        _check_resolution(q.max() * self.R, self.sigma, max(1.0, self.nu))
        n = min(self._angles(q * self.R), 600)
        mu, wm = _gl(n)
        out = np.empty(q.shape)
        for i, qi in enumerate(q):
            A = self._amp(qi * self.R, mu[:, None], sq.xi[None, :])
            f = A * A if s_kind == 'P' else A
            out[i] = (wm[:, None] * f * sq.w[None, :]).sum()
        return out

    def P(self, q):
        return self._ang_avg(q, SizeQuad(self.sigma, self.sP, Nu=self._nu_nodes, ymax=self._ymax(q)), 'P')

    def F(self, q):
        return self._ang_avg(q, SizeQuad(self.sigma, self.sF, Nu=self._nu_nodes, ymax=self._ymax(q)), 'F')

    def _fixed(self, q, mu, kind):
        s = self.sP if kind == 'P' else self.sF
        q = np.asarray(q, float); mu = np.asarray(mu, float)
        sq = SizeQuad(self.sigma, s, Nu=self._nu_nodes, ymax=self._ymax(q))
        qq, mm = np.broadcast_arrays(q, mu)
        shp = qq.shape
        qf, mf = qq.ravel(), mm.ravel()
        out = np.empty(qf.size)
        step = max(1, int(2e7 // (sq.xi.size * self._extra)))
        for i in range(0, qf.size, step):
            A = self._amp(qf[i:i + step, None] * self.R, mf[i:i + step, None], sq.xi[None, :])
            out[i:i + step] = ((A * A if kind == 'P' else A) * sq.w).sum(-1)
        return out.reshape(shp)

    def P_fixed(self, q, mu):
        return self._fixed(q, mu, 'P')

    def F_fixed(self, q, mu):
        return self._fixed(q, mu, 'F')


class Cylinder(_Factorised):
    """Cylinder of radius R and length L = 2 nu R; mu = cos(angle between q and axis)."""
    name = "cylinder"

    def __init__(self, R, nu, sigma=0.0):
        super().__init__(R, sigma)
        self.nu = float(nu)

    def _amp(self, x, mu, xi):
        s = np.sqrt(np.clip(1 - mu ** 2, 0, 1))
        return BLOCK[2].mono(x * xi * s, 'F') * BLOCK[1].mono(x * xi * self.nu * mu, 'F')


class Disk(Cylinder):
    """Thin disk: cylinder with a small aspect ratio nu (default 0.05)."""
    name = "disk"

    def __init__(self, R, nu=0.05, sigma=0.0):
        super().__init__(R, nu, sigma)


class Parallelepiped(_Factorised):
    """Box with edges (2R, 2bR, 2cR).  Orientation average over the full sphere."""
    name = "parallelepiped"

    def __init__(self, R, b=1.0, c=1.0, sigma=0.0):
        super().__init__(R, sigma)
        self.b, self.c, self.nu = float(b), float(c), max(1.0, b, c)

    def _ang_avg(self, q, sq, s_kind):
        """Orientation average over an octant (mirror symmetry; for b = 1 only phi <= pi/4, x <-> y symmetry).
        The q points are taken in blocks of increasing q; each block gets its own angular grid and size
        quadrature (both grow with q R)."""
        q = np.atleast_1d(np.asarray(q, float))
        _check_resolution(q.max() * self.R, self.sigma, self.nu)
        s = self.sP if s_kind == 'P' else self.sF
        sym = abs(self.b - 1.0) < 1e-12
        out = np.empty(q.shape)
        order = np.argsort(q)
        for blk in np.array_split(order, max(1, int(np.ceil(q.size / 24)))):
            qb = q[blk]
            n = int(np.clip(30 + 3 * qb.max() * self.R * self.nu, 40, 220))
            mu, wm = _gl(n)
            ph, wp = _gl(n if not sym else max(20, (n + 1) // 2), 0, np.pi / 4 if sym else np.pi / 2)
            MU, PH = np.meshgrid(mu, ph, indexing='ij')
            W = np.outer(wm, wp); W /= W.sum()
            st = np.sqrt(1 - MU ** 2)
            ax = st * np.cos(PH); ay = st * np.sin(PH); az = MU
            sqb = SizeQuad(self.sigma, s, Nu=self._nu_nodes, ymax=self._ymax(qb))
            if _accel.HAVE_NUMBA:
                out[blk] = _accel.box_iso(qb * self.R, self.b, self.c, sqb.xi, sqb.w, ax, ay, az, W, s_kind)
                continue
            for i, qi in zip(blk, qb):
                x = qi * self.R * sqb.xi[None, None, :]
                A = (BLOCK[1].mono(x * ax[..., None], 'F') * BLOCK[1].mono(x * self.b * ay[..., None], 'F')
                     * BLOCK[1].mono(x * self.c * az[..., None], 'F'))
                f = A * A if s_kind == 'P' else A
                out[i] = ((f * sqb.w).sum(-1) * W).sum()
        return out


class Cube(Parallelepiped):
    name = "cube"

    def __init__(self, R, sigma=0.0):
        super().__init__(R, 1.0, 1.0, sigma)


# ---------------------------------------------------------------------------------------------
# body-frame (fixed orientation) evaluation P_vec(q_b) for the oriented patterns
# q_b = (qx, qy, qz) in the frame of the particle (main axis = z)
# ---------------------------------------------------------------------------------------------
def _P_vec_axisym(self, qb):
    qb = np.asarray(qb, float)
    q = np.sqrt((qb ** 2).sum(-1))
    mu = np.where(q > 0, qb[..., 2] / np.where(q > 0, q, 1.0), 1.0)
    return self.P_fixed(q, mu)


def _F_vec_axisym(self, qb):
    qb = np.asarray(qb, float)
    q = np.sqrt((qb ** 2).sum(-1))
    mu = np.where(q > 0, qb[..., 2] / np.where(q > 0, q, 1.0), 1.0)
    return self.F_fixed(q, mu)


def _tri_vec(self, qb, kind='P'):
    qb = np.asarray(qb, float)
    y = self.R * np.sqrt(qb[..., 0] ** 2 + (self.b * qb[..., 1]) ** 2 + (self.c * qb[..., 2]) ** 2)
    return BLOCK[3].value(y, self.z, self.sP if kind == 'P' else self.sF, kind)


def _box_vec(self, qb, kind='P'):
    qb = np.asarray(qb, float)
    e = np.array([1.0, self.b, self.c])
    sq = SizeQuad(self.sigma, self.sP if kind == 'P' else self.sF, Nu=self._nu_nodes,
                  ymax=2.0 * float(np.abs(qb).max()) * self.R * e.max() * np.sqrt(3.0))
    shp = qb.shape[:-1]
    flat = qb.reshape(-1, 3)
    if _accel.HAVE_NUMBA:
        X = flat * (self.R * e)[None, :]
        return _accel.box_fixed(X[:, 0], X[:, 1], X[:, 2], sq.xi, sq.w, kind).reshape(shp)
    out = np.empty(flat.shape[0])
    step = max(1, int(1e7 // sq.xi.size))
    for i in range(0, flat.shape[0], step):
        x = flat[i:i + step, None, :] * (self.R * e)[None, None, :] * sq.xi[None, :, None]
        A = BLOCK[1].mono(x[..., 0], 'F') * BLOCK[1].mono(x[..., 1], 'F') * BLOCK[1].mono(x[..., 2], 'F')
        out[i:i + step] = ((A * A if kind == 'P' else A) * sq.w).sum(-1)
    return out.reshape(shp)


Sphere.P_vec = _P_vec_axisym
Sphere.F_vec = lambda self, qb: self.F(np.sqrt((np.asarray(qb, float) ** 2).sum(-1)))
Ellipsoid.P_vec = _P_vec_axisym
Ellipsoid.F_vec = _F_vec_axisym
_Factorised.P_vec = _P_vec_axisym
_Factorised.F_vec = _F_vec_axisym
TriaxialEllipsoid.P_vec = lambda self, qb: _tri_vec(self, qb, 'P')
TriaxialEllipsoid.F_vec = lambda self, qb: _tri_vec(self, qb, 'F')
Parallelepiped.P_vec = lambda self, qb: _box_vec(self, qb, 'P')
Parallelepiped.F_vec = lambda self, qb: _box_vec(self, qb, 'F')
