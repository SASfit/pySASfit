"""Ordered lattices (SI Sect. 8) in the style of CrystalScatter: conventional cells with a motif
(structure factor |sum_j exp(2 pi i h.r_j)|^2 / n) or mesophase extinction rules, unit-cell angles,
peak shapes (Gaussian, Lorentzian/Pearson-VII, pseudo-Voigt), Debye-Waller factor, oriented (zone axis,
fibre) and powder-averaged lattice factors, 1D (lamellae), 2D (rod) and 3D lattices.

Z(q) is normalised so that its average over reciprocal space is 1 (decoupling approximation
I = <P>[1 + beta (Z-1) G]).
"""
import numpy as np
from scipy.special import gamma as G_, erf, gammaln, loggamma
from .lattice import Lattice, rotation

# ------------------------------------------------------------------------------------------
# peak shapes (normalised in d dimensions, argument r = |q - G|); 'width' = std dev of the Gaussian
# with identical FWHM
# ------------------------------------------------------------------------------------------
_FW = 2 * np.sqrt(2 * np.log(2))


def _pearson_gamma(fwhm, m):
    return fwhm / (2 * np.sqrt(2 ** (1.0 / m) - 1))


def gamma_peak(x, delta, nu):
    """Generalised 1D peak shape of Foerster et al., J. Phys. Chem. B 109, 1347 (2005), eq. 27::

        L(x) = 2/(pi delta) prod_n [1 + gamma_nu^2/(n+nu/2)^2 * 4x^2/(pi^2 delta^2)]^-1
             = 2/(pi delta) |Gamma(nu/2 + i z)|^2 / Gamma(nu/2)^2,  z = gamma_nu 2x/(pi delta),
        gamma_nu = sqrt(pi) Gamma((nu+1)/2)/Gamma(nu/2)

    nu -> 0: Lorentzian (delta = FWHM); nu -> inf: Gaussian exp(-4x^2/(pi delta^2)) (sigma = sqrt(pi/8) delta).
    Normalised to 1."""
    x = np.asarray(x, float)
    a = 0.5 * nu
    g = np.sqrt(np.pi) * np.exp(gammaln(0.5 * (nu + 1)) - gammaln(a))
    z = g * 2 * x / (np.pi * delta)
    return 2 / (np.pi * delta) * np.exp(2 * (loggamma(a + 1j * z).real - gammaln(a)))


def peak(r, width, shape="gaussian", d=3, m=2.0, eta=0.5, nu=1.0):
    r = np.asarray(r, float)
    fwhm = width * _FW
    if shape == "gamma":
        if d != 1:
            raise ValueError("the generalised (gamma) peak is a 1D shape: it is used per axis (3D lattices) "
                             "or radially (powder), see Crystal")
        return gamma_peak(r, width * np.sqrt(8 / np.pi), nu)
    if shape == "gaussian":
        return np.exp(-0.5 * (r / width) ** 2) / (2 * np.pi * width ** 2) ** (d / 2)
    if shape in ("lorentzian", "pearson"):
        mm = (1.0 if d == 1 else 2.0) if shape == "lorentzian" else m
        if mm <= d / 2:
            raise ValueError("Pearson exponent must exceed d/2")
        g = _pearson_gamma(fwhm, mm)
        return G_(mm) / (np.pi ** (d / 2) * G_(mm - d / 2) * g ** d) * (1 + (r / g) ** 2) ** (-mm)
    if shape == "voigt":  # pseudo-Voigt with equal FWHM
        return eta * peak(r, width, "lorentzian", d) + (1 - eta) * peak(r, width, "gaussian", d)
    raise ValueError(shape)


def _powder_peak3(q, g, width, shape, m=2.0, eta=0.5):
    """Spherical average of a normalised 3D peak centred at |G| = g: (1/2qg) int_{|q-g|}^{q+g} f(r) r dr."""
    q = np.asarray(q, float)
    r1, r2 = np.abs(q - g), q + g
    fwhm = width * _FW
    if shape == "gaussian":
        n = (2 * np.pi * width ** 2) ** -1.5
        s2 = width ** 2
        return n * s2 / (2 * q * g) * (np.exp(-0.5 * r1 ** 2 / s2) - np.exp(-0.5 * r2 ** 2 / s2))
    if shape in ("lorentzian", "pearson"):
        mm = 2.0 if shape == "lorentzian" else m
        gm = _pearson_gamma(fwhm, mm)
        n = G_(mm) / (np.pi ** 1.5 * G_(mm - 1.5) * gm ** 3)
        F = lambda r: (1 + (r / gm) ** 2) ** (1 - mm) * gm ** 2 / (2 * (mm - 1))
        return n / (2 * q * g) * (F(r1) - F(r2))
    if shape == "voigt":
        return (eta * _powder_peak3(q, g, width, "lorentzian") + (1 - eta) * _powder_peak3(q, g, width, "gaussian"))
    raise ValueError(shape)


# ------------------------------------------------------------------------------------------
# lattice definitions: motif positions or extinction rule
# ------------------------------------------------------------------------------------------
_FCC = [(0, 0, 0), (.5, .5, 0), (0, .5, .5), (.5, 0, .5)]
_MOTIF = {
    "sc": [(0, 0, 0)], "bcc": [(0, 0, 0), (.5, .5, .5)], "fcc": _FCC,
    "hcp": [(0, 0, 0), (1 / 3, 2 / 3, .5)], "bct": [(0, 0, 0), (.5, .5, .5)],
    "fd3m": _FCC + [(.25, .25, .25), (.75, .75, .25), (.25, .75, .75), (.75, .25, .75)],
    "orthorhombic": [(0, 0, 0)], "centered_rect": [(0, 0, 0), (.5, .5, 0)],
    "hexcyl": [(0, 0, 0)], "squarecyl": [(0, 0, 0)], "lamellae": [(0, 0, 0)],
}


def domain_width(D, shape="gaussian"):
    """Peak-width parameter `width` of `Crystal` (std dev of the Gaussian of equal FWHM) for a domain size D.

    CrystalScatter / Foerster et al. (Adv. Colloid Interface Sci. 163, 53, 2011, eqs. 82, 92): peak width
    delta = 4/D.  Gaussian peak exp(-4 x^2/(pi delta^2)): sigma = sqrt(pi/8) delta, FWHM = sqrt(pi ln2) delta;
    Lorentzian: delta is the FWHM.  (The 2005 JPCB paper quotes D = 2 pi/delta instead.)"""
    delta = 4.0 / float(D)
    if shape in ("lorentzian", "pearson"):
        return delta / _FW
    return np.sqrt(np.pi / 8) * delta


def _even(x): return x % 2 == 0


def _rule_ia3d(h, k, l):
    """Ia3d (gyroid) allowed reflections as in CrystalScatter's extinction rules (hkl: h+k+l=2n,
    hhl: l=2n & 2h+l=4n, 0kl: k,l=2n, h00: h=4n)."""
    h, k, l = abs(int(h)), abs(int(k)), abs(int(l))
    nz = (h != 0) + (k != 0) + (l != 0)
    if nz == 3:
        if len({h, k, l}) == 3:
            return 1 if (h + k + l) % 2 == 0 else 0
        for (x, y, z_) in ((h, k, l), (h, l, k), (k, l, h)):     # x == y, z_ is the odd one out
            if x == y and z_ != x:
                return 1 if (z_ % 2 == 0 and (2 * x + z_) % 4 == 0) else 0
        return 1 if (h % 4 == 0) else 0                           # hhh
    if nz == 2:
        return 1 if all(v % 2 == 0 for v in (h, k, l) if v) else 0
    if nz == 1:
        return 1 if [v for v in (h, k, l) if v][0] % 4 == 0 else 0
    return 0


def _rule_pn3m(h, k, l):
    h, k, l = int(h), int(k), int(l)
    a = [h, k, l]
    nz = sum(1 for v in a if v)
    if nz == 3:
        return 1
    if nz == 2:
        return 1 if _even(sum(a)) else 0
    if nz == 1:
        return 1 if _even([v for v in a if v][0]) else 0
    return 0


def _rule_im3m(h, k, l):
    return 1 if (int(h) + int(k) + int(l)) % 2 == 0 and (h or k or l) else 0


_RULES = {"gyroid": _rule_ia3d, "ia3d": _rule_ia3d, "obdd": _rule_pn3m, "pn3m": _rule_pn3m,
          "plumbers": _rule_im3m, "im3m": _rule_im3m}
_DIM = {"lamellae": 1, "hexcyl": 2, "squarecyl": 2, "centered_rect": 2}


class Crystal(Lattice):
    """Lattice of a named type.

    Parameters
    ----------
    kind : str
        sc, bcc, fcc, hcp, bct, fd3m, orthorhombic, centered_rect, hexcyl, squarecyl, lamellae,
        gyroid (Ia3d), obdd (Pn3m), plumbers (Im3m).
    a, b, c, alpha, beta, gamma : float
        Conventional cell (lengths, angles in degrees); defaults: b = a, c = a (hcp: c = 1.633 a),
        gamma = 120 for hcp and hexcyl.
    Rot : ndarray, optional
        3x3 rotation of the lattice into the lab frame (see `zone_axis_rotation`).
    width : float
        Peak width (std dev of the Gaussian of equal FWHM).
    peak : str
        'gaussian', 'lorentzian', 'pearson', 'voigt' (or 'gamma').
    dw : float
        Debye-Waller sigma (length).
    hmax : int
        Maximum |h|, |k|, |l|.
    para : float
        Paracrystal coefficient: reflection g gets the width ``width + para |g|^2`` (0: all the same).

    For dim < 3 the non-periodic direction is the cell c-axis (rods) or the a-axis normal (lamellae).
    """

    def __init__(self, kind, a, b=None, c=None, alpha=90., beta=90., gamma=None, Rot=None, width=0.002,
                 peak="gaussian", m=2.0, eta=0.5, dw=0.0, hmax=8, nu=1.0, powder_mode="spherical", para=0.0):
        kind = kind.lower()
        self.kind = kind
        self.dim = _DIM.get(kind, 3)
        b = a if b is None else b
        c = (1.63299 * a if kind == "hcp" else a) if c is None else c
        if gamma is None:
            gamma = 120. if kind in ("hcp", "hexcyl") else 90.
        self.cellpar = (a, b, c, alpha, beta, gamma)
        al, be, ga = np.radians([alpha, beta, gamma])
        va = np.array([a, 0, 0])
        vb = np.array([b * np.cos(ga), b * np.sin(ga), 0])
        cx = c * np.cos(be)
        cy = c * (np.cos(al) - np.cos(be) * np.cos(ga)) / np.sin(ga)
        vc = np.array([cx, cy, np.sqrt(max(c * c - cx * cx - cy * cy, 0))])
        M = np.stack([va, vb, vc], axis=1)
        self.Rot = np.eye(3) if Rot is None else np.asarray(Rot, float)
        self.M = M
        self.width, self.dw = float(width), float(dw)
        # paracrystalline broadening: width of reflection g = width + para * |g|^2 (CrystalScatter's
        # 'RadioButtonPara': delta = 4/D + sigma_DW^2/dist * q^2); 0 = the same width for all reflections
        self.para = float(para)
        self._width0 = self.width
        self.peak_shape, self.m, self.eta, self.nu = peak, m, eta, nu
        self.powder_mode = powder_mode
        self.v0 = abs(np.linalg.det(M))
        B = 2 * np.pi * np.linalg.inv(self.Rot @ M).T          # reciprocal vectors (columns), lab frame
        hr = np.arange(-hmax, hmax + 1)
        hh = hr if self.dim >= 1 else [0]
        kk = hr if self.dim >= 2 else [0]
        ll = hr if self.dim >= 3 else [0]
        H = np.array(np.meshgrid(hh, kk, ll, indexing='ij')).reshape(3, -1)
        H = H[:, np.any(H != 0, axis=0)]
        motif = np.array(_MOTIF.get(kind, [(0, 0, 0)]))
        if kind in _RULES:
            w = np.array([_RULES[kind](*h) for h in H.T], float)
            n = max(w.mean(), 1e-12)          # allowed fraction: normalises <Z> over reciprocal space to 1
        else:
            F2 = np.abs(np.exp(2j * np.pi * (motif @ H)).sum(0)) ** 2
            w, n = F2, float(len(motif))
        keep = w > 1e-9
        self.hkl = H[:, keep]
        self.weights = (w / n)[keep]
        self.G = B @ self.hkl
        self.n_motif = n
        if self.dim == 2:
            self.axis = (self.Rot @ M[:, 2]) / np.linalg.norm(M[:, 2])      # rod direction (lab)
            # area of the 2D cell
            self.v0d = np.linalg.norm(np.cross(M[:, 0], M[:, 1]))
        elif self.dim == 1:
            self.axis = (self.Rot @ np.linalg.inv(M).T[:, 0])
            self.axis /= np.linalg.norm(self.axis)
            self.v0d = a
        self.hmax = hmax

    # ---- lattice factor for oriented crystallites ---------------------------------------------
    def _dist(self, qx, qy, qz, g):
        d = np.stack([qx - g[0], qy - g[1], qz - g[2]])
        if self.dim == 3:
            return np.sqrt((d ** 2).sum(0))
        ax = self.axis
        par = d[0] * ax[0] + d[1] * ax[1] + d[2] * ax[2]
        if self.dim == 1:
            return np.abs(par)
        return np.sqrt(np.maximum((d ** 2).sum(0) - par ** 2, 0.0))

    def _gamma_term(self, qx, qy, qz, g):
        """Generalised peak: product of the 1D shape along the three lab axes (3D), along the axis (1D)."""
        delta = self.width * np.sqrt(8 / np.pi)
        if self.dim == 3:
            return (gamma_peak(qx - g[0], delta, self.nu) * gamma_peak(qy - g[1], delta, self.nu)
                    * gamma_peak(qz - g[2], delta, self.nu))
        if self.dim == 1:
            return gamma_peak(self._dist(qx, qy, qz, g), delta, self.nu)
        raise ValueError("gamma peak shape is available for 1D (lamellar) and 3D lattices only")

    def Z(self, qx, qy, qz=0.0):
        qx, qy, qz = np.broadcast_arrays(np.asarray(qx, float), np.asarray(qy, float), np.asarray(qz, float))
        out = np.zeros(qx.shape)
        d = self.dim
        # paracrystal: the width at the scattering vector of the pixel (as CrystalScatter does)
        wq = self._width0 + self.para * (qx * qx + qy * qy + qz * qz) if self.para else self._width0
        for g, w in zip(self.G.T, self.weights):
            self.width = wq
            if self.peak_shape == "gamma":
                out += w * self._gamma_term(qx, qy, qz, g)
            else:
                out += w * peak(self._dist(qx, qy, qz, g), self.width, self.peak_shape, d, self.m, self.eta)
        self.width = self._width0
        vol = self.v0 if d == 3 else self.v0d
        return (2 * np.pi) ** d / vol * out

    # ---- powder average -------------------------------------------------------------------------
    def Z_powder(self, q, nth=4000):
        q = np.atleast_1d(np.asarray(q, float))
        Gm = np.linalg.norm(self.G, axis=0)
        uniq, inv = np.unique(np.round(Gm, 9), return_inverse=True)
        wsum = np.bincount(inv, weights=self.weights)
        out = np.zeros(q.shape)
        d = self.dim
        for g, w in zip(uniq, wsum):
            radial = (self.powder_mode in ("radial", "radial_g") or self.peak_shape == "gamma") and d in (2, 3)
            # paracrystal: width at q in the radial form (CrystalScatter), at the reflection otherwise
            self.width = (self._width0 + self.para * q * q) if (radial and self.para) else self._width0 + self.para * g * g
            if radial:
                # CrystalScatter/Foerster powder form: a 1D radial shape L(q-g) spread over the sphere/circle,
                # Z = (2pi)^d/V * sum w L(q-g)/(Omega_d q^(d-1)),  Omega_3 = 4 pi, Omega_2 = 2 pi
                if self.peak_shape == "gamma":
                    L1 = gamma_peak(q - g, self.width * np.sqrt(8 / np.pi), self.nu)
                else:
                    L1 = peak(q - g, self.width, self.peak_shape, 1, self.m, self.eta)
                qn = np.full_like(q, g) if self.powder_mode == "radial_g" else q    # CrystalScatter: 1/g^2 (g3)
                out += w * L1 / ((4 * np.pi * qn ** 2) if d == 3 else (2 * np.pi * qn))
            elif self.peak_shape == "gamma" and d == 1:
                # <L(q mu - g)>_mu over mu in [0,1] = [C(q-g) - C(-g)]/q with C the CDF of the 1D shape (both +-g)
                delta = self.width * np.sqrt(8 / np.pi)
                X = float(q.max() + g) + 30 * delta
                xs = np.linspace(0, X, 200001)
                cs = np.concatenate([[0.0], np.cumsum(0.5 * (gamma_peak(xs[1:], delta, self.nu)
                                                           + gamma_peak(xs[:-1], delta, self.nu)) * np.diff(xs))])
                C = lambda x: 0.5 + np.sign(x) * np.interp(np.abs(x), xs, cs)
                out += w * (C(q - g) + C(q + g) - C(-g) - C(g)) / (2 * q)
            elif self.peak_shape == "gamma":
                raise ValueError("gamma peak shape powder average is implemented for 1D, 2D and 3D lattices only")
            elif d == 3:
                out += w * _powder_peak3(q, g, self.width, self.peak_shape, self.m, self.eta)
            elif d == 1:      # <L1(q mu - g)>_mu in [0,1]
                s = self.width
                out += w / (4 * q) * (erf((q - g) / (np.sqrt(2) * s)) + erf((q + g) / (np.sqrt(2) * s)))
            else:             # 2D rods: int_0^{pi/2} sin(t) f(q sin t) dt, f = azimuthally averaged 2D Gaussian
                from scipy.special import i0e
                s2 = self.width ** 2
                x64, w64 = np.polynomial.legendre.leggauss(80)
                umin = np.full(q.shape, max(g - 9 * self.width, 0.0))
                umax = np.minimum(g + 9 * self.width, q)
                act = umax > umin
                if not act.any():
                    continue
                qa = q[act]
                t0, t1 = np.arcsin(np.clip(umin[act] / qa, 0, 1)), np.arcsin(np.clip(umax[act] / qa, 0, 1))
                t = 0.5 * (t1 - t0)[:, None] * x64[None, :] + 0.5 * (t1 + t0)[:, None]
                wt = 0.5 * (t1 - t0)[:, None] * w64[None, :]
                u = qa[:, None] * np.sin(t)
                f = np.exp(-0.5 * (u - g) ** 2 / s2) * i0e(u * g / s2) / (2 * np.pi * s2)
                part = (f * np.sin(t) * wt).sum(1)
                tmp = np.zeros(q.shape); tmp[act] = part
                out += w * tmp
        self.width = self._width0
        if d == 1:
            return (2 * np.pi) / self.v0d * out
        vol = self.v0 if d == 3 else self.v0d
        return (2 * np.pi) ** d / vol * out

    # ---- fibre texture: rotate the lattice about a lab axis ---------------------------------------
    def Z_fiber(self, qx, qy, qz=0.0, axis=(0, 0, 1), npsi=720):
        """Average of Z over rotations of the lattice about the lab `axis` (fibre texture)."""
        qx, qy, qz = np.broadcast_arrays(np.asarray(qx, float), np.asarray(qy, float), np.asarray(qz, float))
        Gs, ax0 = self.G.copy(), getattr(self, "axis", None)
        out = np.zeros(qx.shape)
        try:
            for psi in (np.arange(npsi) + 0.5) * 2 * np.pi / npsi:
                Rz = rotation(axis, psi)
                self.G = Rz @ Gs
                if ax0 is not None:
                    self.axis = Rz @ ax0
                out += self.Z(qx, qy, qz)
        finally:
            self.G = Gs
            if ax0 is not None:
                self.axis = ax0
        return out / npsi


def zone_axis_rotation(M, uvw, to=(1, 0, 0), psi=0.0):
    """Rotation that maps the real-space direction [uvw] of the cell M onto the lab direction `to`
    (e.g. beam along x) followed by a rotation psi (rad) about it."""
    v = np.asarray(M, float) @ np.asarray(uvw, float)
    v /= np.linalg.norm(v)
    t = np.asarray(to, float) / np.linalg.norm(to)
    ax = np.cross(v, t)
    s = np.linalg.norm(ax)
    if s < 1e-12:
        R0 = np.eye(3) if v @ t > 0 else rotation(np.cross(t, [0, 0, 1]) if abs(t[2]) < .9 else [1, 0, 0], np.pi)
    else:
        R0 = rotation(ax, np.arctan2(s, v @ t))
    return rotation(t, psi) @ R0


def make_lattice(kind, a, **kw):
    """Alias of ``Crystal(kind, a, **kw)``."""
    return Crystal(kind, a, **kw)
