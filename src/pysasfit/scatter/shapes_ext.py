"""Superball, lenses and dumbbells (SI Sect. 4.8, 4.9.2)."""
import numpy as np
import warnings
from scipy.interpolate import CubicSpline
from .blocks import BLOCK
from .shapes import Shape, _Factorised, SizeQuad, _gl


class LensDumbbell(_Factorised):
    """Rotational body with profile  r(z) = R (1 - | |z|/L - alpha |^k )^(1/k),  |z| <= L (1+alpha),
    L = nu R.  alpha < 0: lens, alpha > 0: dumbbell, alpha = 0: superellipsoid (k=2: ellipsoid).
    mu = cos(angle between q and the symmetry axis)."""
    name = "lens_dumbbell"
    _nu_nodes = 161

    def __init__(self, R, nu, alpha=0.0, k=2.0, sigma=0.0, nz=96):
        super().__init__(R, sigma)
        self.nu, self.alpha, self.k, self.nz = float(nu), float(alpha), float(k), int(nz)
        H = self.nu * (1 + self.alpha)
        t, w = _gl(self.nz, 0.0, 1.0)
        self.t = H * t
        self.wz = H * w
        r = (1 - np.abs(self.t / self.nu - self.alpha) ** self.k).clip(0, None) ** (1 / self.k)
        self.r = r
        self.vol = (self.wz * r ** 2).sum()
        self._extra = self.nz

    def _amp(self, x, mu, xi):
        s = np.sqrt(np.clip(1 - mu ** 2, 0, 1))
        xr = (x * xi * s)[..., None]
        xz = (x * xi * mu)[..., None]
        f2 = BLOCK[2].mono(xr * self.r, 'F')
        return (np.cos(xz * self.t) * self.r ** 2 * f2 * self.wz).sum(-1) / self.vol


class Lens(LensDumbbell):
    name = "lens"

    def __init__(self, R, nu, alpha=-0.5, k=2.0, sigma=0.0, **kw):
        assert alpha < 0
        super().__init__(R, nu, alpha, k, sigma, **kw)


class Dumbbell(LensDumbbell):
    name = "dumbbell"

    def __init__(self, R, nu, alpha=0.5, k=2.0, sigma=0.0, **kw):
        assert alpha > 0
        super().__init__(R, nu, alpha, k, sigma, **kw)


_TAB = {}


class Superball(Shape):
    """|x/a|^p + |y/b|^p + |z/c|^p <= 1 with a = R, b = bR, c = cR  (p = k of the SI; p = 2 ellipsoid,
    large p -> cuboid).  Orientation-averaged P(q), F(q) with Schulz-Zimm size averaging.

    The unit-size orientational averages <F>(x), <F^2>(x) (x = q R) are tabulated once with
    2D quadrature (analytic z-integration, SI Eq. S.4.8.x) up to `xmax` and then size-averaged with
    the Schulz-Zimm quadrature (q R (1+few sigma) must stay below xmax).
    """
    name = "superball"

    def __init__(self, R, p=2.0, b=1.0, c=1.0, sigma=0.0, xmax=30.0, dx=0.1, ng=64, n_mu=40, n_phi=16):
        super().__init__(R, sigma)
        self.p, self.b, self.c = float(p), float(b), float(c)
        self.nu = max(1.0, b, c)
        self.xmax, self.dx, self.ng, self.n_mu, self.n_phi = xmax, dx, ng, n_mu, n_phi

    def _fixed_orient_amp(self, x, qx, qy, qz, X, Yf, Wxy):
        """amplitude (un-normalised) at scalar x for all orientations (arrays qx,qy,qz of shape (n,))."""
        pass

    def _build(self):
        key = (self.p, self.b, self.c, self.xmax, self.dx, self.ng, self.n_mu, self.n_phi)
        if key in _TAB:
            return _TAB[key]
        p, b, c = self.p, self.b, self.c
        gx, wx = _gl(self.ng, 0.0, 1.0)
        gy, wy = _gl(self.ng, 0.0, 1.0)
        # points of the quadrant: X, Y = Ymax(X)*gy ; Z top = c (1-X^p-Y^p)^(1/p)
        Ymax = b * (1 - gx ** p) ** (1 / p)
        Xp = np.repeat(gx[:, None], self.ng, 1)
        Yp = Ymax[:, None] * gy[None, :]
        Zt = c * np.clip(1 - Xp ** p - (Yp / b) ** p, 0, None) ** (1 / p)
        Wp = wx[:, None] * (Ymax[:, None] * wy[None, :])
        vol = 8 * (Wp * Zt).sum()
        mu, wm = _gl(self.n_mu, 0.0, 1.0)
        ph, wph = _gl(self.n_phi, 0.0, np.pi / 2)
        MU, PH = np.meshgrid(mu, ph, indexing='ij')
        W = np.outer(wm, wph); W = (W / W.sum()).ravel()
        st = np.sqrt(1 - MU ** 2)
        qx, qy, qz = (st * np.cos(PH)).ravel(), (st * np.sin(PH)).ravel(), MU.ravel()
        xs = np.arange(0.0, self.xmax + self.dx / 2, self.dx)
        F1 = np.empty(xs.size); F2 = np.empty(xs.size)
        Xf, Yf, Zf, Wf = Xp.ravel(), Yp.ravel(), Zt.ravel(), Wp.ravel()
        from . import _accel
        if _accel.HAVE_NUMBA:
            F1, F2 = _accel.superball_table(xs.size, self.dx, qx, qy, qz, W, Xf, Yf, Zf, Wf, vol)
            xs_iter = []
        else:
            xs_iter = xs
        for i, x in enumerate(xs_iter):
            A = np.empty(W.size)
            for o in range(0, W.size, 64):
                sl = slice(o, o + 64)
                ax = x * qx[sl, None]; ay = x * qy[sl, None]; az = x * qz[sl, None]
                with np.errstate(all='ignore'):
                    zint = np.where(np.abs(az) < 1e-9, Zf[None, :], np.sin(az * Zf[None, :]) / np.where(az == 0, 1, az))
                A[sl] = (np.cos(ax * Xf[None, :]) * np.cos(ay * Yf[None, :]) * zint * Wf[None, :]).sum(1) * 8 / vol
            F1[i] = (A * W).sum()
            F2[i] = (A * A * W).sum()
        tab = (CubicSpline(xs, F1), CubicSpline(xs, F2), self.xmax)
        _TAB[key] = tab
        return tab

    def _avg(self, q, s, which):
        tab = self._build()
        spl = tab[0] if which == 'F' else tab[1]
        q = np.atleast_1d(np.asarray(q, float))
        sq = SizeQuad(self.sigma, s)
        x = q[:, None] * self.R * sq.xi[None, :]
        if x.max() > tab[2]:
            warnings.warn("q*R*(1+few sigma) exceeds xmax of the superball table; NaN returned there")
        v = np.where(x <= tab[2], spl(np.minimum(x, tab[2])), np.nan)
        return (v * sq.w).sum(1)

    def P(self, q):
        return self._avg(q, self.sP, 'P')

    def F(self, q):
        return self._avg(q, self.sF, 'F')


class Octahedron(Superball):
    """Regular octahedron |x|+|y|+|z| <= R  (superball with p = 1); vertices at distance R from the centre."""
    name = "octahedron"

    def __init__(self, R, sigma=0.0, **kw):
        super().__init__(R, p=1.0, b=1.0, c=1.0, sigma=sigma, **kw)


class SuperEllipsoid(Shape):
    """Superellipsoid with elliptical cross-section (SI 4.9.3): body-frame profile
    {x,y,z: (x/R)^2 + (y/(bR))^2 <= r(z)^2},  r(z) = (1 - | |z|/(nu R) - alpha |^k)^(1/k),  |z| <= nu R (1+alpha).
    k=2, alpha=0, b=1: ellipsoid; alpha != 0: lens (<0) / dumbbell (>0) with elliptical cross-section.
    Fixed-orientation amplitude by 1D quadrature in z (analytic elliptical cross-section), orientation
    average by 2D Gauss-Legendre, Schulz-Zimm average by size quadrature."""
    name = "superellipsoid"

    def __init__(self, R, b=1.0, nu=1.0, k=2.0, alpha=0.0, sigma=0.0, nz=80):
        super().__init__(R, sigma)
        self.b, self.nu, self.k, self.alpha = float(b), float(nu), float(k), float(alpha)
        H = self.nu * (1 + self.alpha)
        t, w = _gl(nz, 0.0, 1.0)
        self.t, self.wz = H * t, H * w
        self.r = (1 - np.abs(self.t / self.nu - self.alpha) ** self.k).clip(0, None) ** (1 / self.k)
        self.vol = (self.wz * self.r ** 2).sum()
        self._nu_nodes = 161

    def _amp(self, xx, xy, xz):
        """normalised amplitude, arrays xx,xy,xz (q_i * R, size-scaled) broadcast; z-quadrature axis appended."""
        xr = np.sqrt(xx ** 2 + (self.b * xy) ** 2)[..., None]
        f2 = BLOCK[2].mono(xr * self.r, 'F')
        return (np.cos(xz[..., None] * self.t) * self.r ** 2 * f2 * self.wz).sum(-1) / self.vol

    def _sizes(self, kind):
        return SizeQuad(self.sigma, self.sP if kind == 'P' else self.sF, Nu=self._nu_nodes)

    def _vec(self, qb, kind):
        qb = np.asarray(qb, float)
        sq = self._sizes(kind)
        shp = qb.shape[:-1]
        flat = qb.reshape(-1, 3)
        out = np.empty(flat.shape[0])
        step = max(1, int(1.5e7 // (sq.xi.size * self.t.size)))
        for i in range(0, flat.shape[0], step):
            s = flat[i:i + step, None, :] * self.R * sq.xi[None, :, None]
            A = self._amp(s[..., 0], s[..., 1], s[..., 2])
            out[i:i + step] = ((A * A if kind == 'P' else A) * sq.w).sum(-1)
        return out.reshape(shp)

    def P_vec(self, qb):
        return self._vec(qb, 'P')

    def F_vec(self, qb):
        return self._vec(qb, 'F')

    def _iso(self, q, kind):
        q = np.atleast_1d(np.asarray(q, float))
        n = int(np.clip(24 + 2 * q.max() * self.R * max(1, self.nu, self.b), 24, 64))
        mu, wm = _gl(n, 0.0, 1.0)
        ph, wp = _gl(n, 0.0, np.pi / 2)
        MU, PH = np.meshgrid(mu, ph, indexing='ij')
        W = np.outer(wm, wp); W /= W.sum()
        st = np.sqrt(1 - MU ** 2)
        d = np.stack([st * np.cos(PH), st * np.sin(PH), MU], -1)
        out = np.empty(q.shape)
        for i, qi in enumerate(q):
            out[i] = (self._vec(qi * d, kind) * W).sum()
        return out

    def P(self, q):
        return self._iso(q, 'P')

    def F(self, q):
        return self._iso(q, 'F')
