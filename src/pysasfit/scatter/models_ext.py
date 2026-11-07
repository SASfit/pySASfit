"""Models that the hypergeometric (hypscat) part of pySCAtter does not cover, added for the pySCAtter tool:

* :class:`KratkyPorodChain` – worm-like chain, Kholodenko's approximation (Macromolecules 26, 4179 (1993)),
  Schulz-Zimm polydisperse in the contour length.
* :func:`powerlaw_shell_sphere` – core + shell with an algebraic segment-density profile φ(r) ∝ r^α
  (Förster & Burger, Macromolecules 31, 879 (1998): star polymers / block-copolymer micelles), as a stack of
  thin homogeneous shells of the exact multi-shell sphere.
* :class:`TeubnerStrey` – microemulsion / bicontinuous model (Teubner & Strey, J. Chem. Phys. 87, 3195 (1987)).
* Isotropic structure factors for the decoupling approximation I = P (1 + β (S - 1)):
  :class:`PercusYevick` (hard spheres), :class:`StickyHardSphere` (Baxter / Menon et al.),
  :class:`FractalTeixeira` (mass fractal of building blocks, Teixeira J. Appl. Cryst. 21, 781 (1988)).

All lengths in the same unit (nm in the GUI), q in the inverse unit.  ``P(0) = 1`` for the form factors.
"""
from __future__ import annotations

import numpy as np
from scipy.special import gamma as _G, gammaln

from .coreshell import CoreShellSphere

__all__ = ["KratkyPorodChain", "powerlaw_shell_sphere", "TeubnerStrey", "PercusYevick", "StickyHardSphere",
           "FractalTeixeira", "STRUCTURE_FACTORS", "make_structure_factor"]


# ----------------------------------------------------------------------------- worm-like chain
def _kholodenko_mono(q, L, b, nz=400):
    """Kholodenko's worm-like chain form factor (P(0) = 1) for contour length L and Kuhn length b."""
    q = np.atleast_1d(np.asarray(q, float))
    x = 3.0 * L / b
    # Gauss-Legendre nodes on [0, x]; the integrand is smooth (f(z) ~ 1 near 0, decays like e^-z)
    t, w = np.polynomial.legendre.leggauss(nz)
    z = 0.5 * x * (t + 1.0)
    wz = 0.5 * x * w
    out = np.empty(q.shape)
    a = q * b / 3.0
    for k, ak in enumerate(a):
        with np.errstate(over="ignore", invalid="ignore"):
            if ak < 1.0:
                E = np.sqrt(1.0 - ak * ak)
                # sinh(Ez)/(E sinh z) written with exponentials (no overflow at large z)
                f = np.where(z > 1e-8, (np.exp((E - 1) * z) - np.exp(-(E + 1) * z)) / (E * (1 - np.exp(-2 * z))),
                             1.0)
            elif ak > 1.0:
                F = np.sqrt(ak * ak - 1.0)
                f = np.where(z > 1e-8, np.sin(F * z) / (F * np.sinh(np.minimum(z, 700.0))), 1.0)
            else:
                f = np.where(z > 1e-8, z / np.sinh(np.minimum(z, 700.0)), 1.0)
        I1 = np.sum(wz * f)
        I2 = np.sum(wz * z * f)
        out[k] = 2.0 / x * (I1 - I2 / x)
    return out


class KratkyPorodChain:
    """Worm-like (Kratky-Porod) chain of contour length ``L`` and Kuhn length ``b`` (= 2 × persistence length),
    without excluded volume.  ``sigma``: relative standard deviation of L (Schulz-Zimm, number weighted by
    mass, i.e. weights ∝ L f(L))."""
    name = "kratky_porod"

    def __init__(self, L, b, sigma=0.0):
        if L <= 0 or b <= 0:
            raise ValueError("Kratky-Porod chain: L and b must be > 0")
        self.L, self.b, self.sigma = float(L), float(b), float(sigma)

    @property
    def Rg(self):
        """Radius of gyration (Benoit-Doty) of the monodisperse chain."""
        lp, L = self.b / 2.0, self.L
        return np.sqrt(lp * L / 3 - lp ** 2 + 2 * lp ** 3 / L - 2 * lp ** 4 / L ** 2 * (1 - np.exp(-L / lp)))

    def P(self, q):
        q = np.atleast_1d(np.asarray(q, float))
        if self.sigma <= 0:
            return _kholodenko_mono(q, self.L, self.b)
        z = (1 - self.sigma ** 2) / self.sigma ** 2
        t = np.linspace(max(1e-3, 1 - 6 * self.sigma), 1 + 6 * self.sigma, 41)
        lw = (z + 1) * np.log(t) - (z + 1) * t                # Schulz-Zimm × L (mass weighting)
        w = np.exp(lw - lw.max())
        w /= w.sum()
        return sum(wi * _kholodenko_mono(q, self.L * ti, self.b) for wi, ti in zip(w, t))

    def beta(self, q):
        return np.ones_like(np.atleast_1d(np.asarray(q, float)))


# ----------------------------------------------------------------------------- algebraic shell profile
def powerlaw_shell_sphere(R_core, R_out, alpha, rho_core=1.0, rho_shell=1.0, sigma=0.0, n=None,
                          rho_solvent=0.0):
    """Homogeneous core (radius ``R_core``, density ``rho_core``) and a shell up to ``R_out`` whose density
    follows ρ(r) = rho_shell (r / R_core)^alpha (alpha = -4/3 star / micelle corona in good solvent,
    -1 θ solvent, 0 homogeneous; Förster & Burger 1998).  Built from ``n`` thin homogeneous shells of the exact
    multi-shell sphere (default: enough shells for |Δρ| ≤ 3 % per step, 20 … 120)."""
    if not 0 < R_core < R_out:
        raise ValueError("power-law shell: 0 < R_core < R_out required")
    if n is None:
        span = abs(alpha) * np.log(R_out / R_core)
        n = int(np.clip(np.ceil(span / 0.03), 20, 120))
    pc = R_core / R_out
    # geometric spacing: equal steps in ln r give equal relative density steps of the power law
    edges = np.geomspace(pc, 1.0, n + 1)
    mid = np.sqrt(edges[:-1] * edges[1:])
    p = np.concatenate([[pc], edges[1:]])
    p[-1] = 1.0
    rho = np.concatenate([[rho_core], rho_shell * (mid / pc) ** alpha])
    return CoreShellSphere(R_out, p, rho, sigma, rho_solvent)


# ----------------------------------------------------------------------------- Teubner-Strey
class TeubnerStrey:
    """I(q) ∝ 1 / (a2 + c1 q² + c2 q⁴) from the repeat distance ``d`` and the correlation length ``xi``;
    normalised to P(0) = 1.  An intensity model (no particle amplitude): beta = 1."""
    name = "teubner_strey"

    def __init__(self, d, xi):
        if d <= 0 or xi <= 0:
            raise ValueError("Teubner-Strey: d and xi must be > 0")
        self.d, self.xi = float(d), float(xi)

    def P(self, q):
        q = np.atleast_1d(np.asarray(q, float))
        k, ik = 2 * np.pi / self.d, 1.0 / self.xi
        a2 = (k * k + ik * ik) ** 2
        c1 = -2.0 * (k * k - ik * ik)
        return a2 / (a2 + c1 * q * q + q ** 4)

    def beta(self, q):
        return np.ones_like(np.atleast_1d(np.asarray(q, float)))

    @property
    def amphiphilicity(self):
        """f_a = c1 / sqrt(4 a2 c2) (−1 … 1; < 0 structured microemulsion)."""
        k, ik = 2 * np.pi / self.d, 1.0 / self.xi
        return -2.0 * (k * k - ik * ik) / np.sqrt(4.0 * (k * k + ik * ik) ** 2)


# ----------------------------------------------------------------------------- structure factors
class PercusYevick:
    """Hard-sphere structure factor (Percus-Yevick closure; Wertheim/Thiele analytic solution) for hard-sphere
    radius ``R`` and volume fraction ``eta``."""
    name = "percus_yevick"

    def __init__(self, R, eta):
        if R <= 0 or not 0 <= eta < 0.74:
            raise ValueError("Percus-Yevick: R > 0 and 0 ≤ eta < 0.74")
        self.R, self.eta = float(R), float(eta)

    def S(self, q):
        q = np.atleast_1d(np.asarray(q, float))
        eta = self.eta
        a = (1 + 2 * eta) ** 2 / (1 - eta) ** 4
        b = -6 * eta * (1 + eta / 2) ** 2 / (1 - eta) ** 4
        c = eta * a / 2
        x = 2 * q * self.R
        out = np.empty_like(x)
        small = x < 0.05                      # series: the closed form cancels badly at small x
        xs = x[small]
        # series of G(x)/x at small x: integral of the direct correlation function
        out[small] = 1.0 / (1 + 24 * eta * (a / 3 + b / 4 + c / 6 - xs ** 2 * (a / 30 + b / 36 + c / 48)))
        xl = x[~small]
        s, co = np.sin(xl), np.cos(xl)
        G = (a * (s - xl * co) / xl ** 2
             + b * (2 * xl * s + (2 - xl ** 2) * co - 2) / xl ** 3
             + c * (-xl ** 4 * co + 4 * ((3 * xl ** 2 - 6) * co + (xl ** 3 - 6 * xl) * s + 6)) / xl ** 5)
        out[~small] = 1.0 / (1 + 24 * eta * G / xl)
        return out


class StickyHardSphere:
    """Baxter's adhesive hard spheres (Percus-Yevick, perturbative solution as in Menon, Manohar & Rao,
    J. Chem. Phys. 95, 9186 (1991)): hard-sphere radius ``R``, volume fraction ``eta``, stickiness ``tau``
    (large: hard spheres), well width ``delta`` (relative to the diameter, small)."""
    name = "sticky_hard_sphere"

    def __init__(self, R, eta, tau, delta=0.01):
        if R <= 0 or not 0 <= eta < 0.6 or tau <= 0 or not 0 < delta < 0.5:
            raise ValueError("sticky hard spheres: R > 0, 0 ≤ eta < 0.6, tau > 0, 0 < delta < 0.5")
        self.R, self.eta, self.tau, self.delta = float(R), float(eta), float(tau), float(delta)

    def baxter(self):
        """(eta, sigma, lambda, a, b) of Baxter's solution: eta and sigma of the sticky sphere (core + well)."""
        eta = self.eta / (1 - self.delta) ** 3            # volume fraction of the sticky spheres (core + well)
        sig = 2 * self.R / (1 - self.delta)
        A, B, C = eta / 12.0, self.tau + eta / (1 - eta), (1 + eta / 2) / (1 - eta) ** 2
        disc = B * B - 4 * A * C
        if disc < 0:
            raise ValueError("sticky hard spheres: no real solution (too sticky for this volume fraction)")
        lam = (B - np.sqrt(disc)) / (2 * A) if A > 0 else C / B          # smaller root
        mu = lam * eta * (1 - eta)
        if mu > 1 + 2 * eta:
            raise ValueError("sticky hard spheres: unphysical root")
        a = (1 + 2 * eta - mu) / (1 - eta) ** 2
        b = (mu - 3 * eta) / (2 * (1 - eta) ** 2)
        return eta, sig, lam, a, b

    def S(self, q):
        """S = 1/|Q(k)|², Q(k) = 1 − 12 η ∫₀¹ q(r) e^{ikr} dr, q(r) = a(r²−1)/2 + b(r−1) + λ/12 (Baxter 1968)."""
        q = np.atleast_1d(np.asarray(q, float))
        eta, sig, lam, a, b = self.baxter()
        t, w = np.polynomial.legendre.leggauss(64)
        r, w = 0.5 * (t + 1), 0.5 * w
        qr = a * (r * r - 1) / 2 + b * (r - 1) + lam / 12
        k = (q * sig)[:, None]
        re = 1 - 12 * eta * np.sum(w * qr * np.cos(k * r), axis=1)
        im = -12 * eta * np.sum(w * qr * np.sin(k * r), axis=1)
        return 1.0 / (re * re + im * im)


class FractalTeixeira:
    """Mass-fractal aggregate of building blocks of radius ``r0`` with fractal dimension ``D`` (1 < D < 3)
    and cut-off length ``xi`` (Teixeira 1988)."""
    name = "fractal"

    def __init__(self, r0, D, xi):
        if r0 <= 0 or xi <= 0 or not 1 < D < 3:
            raise ValueError("fractal: r0 > 0, xi > 0 and 1 < D < 3")
        self.r0, self.D, self.xi = float(r0), float(D), float(xi)

    def S(self, q):
        q = np.atleast_1d(np.asarray(q, float))
        D, xi, r0 = self.D, self.xi, self.r0
        qr = np.maximum(q * r0, 1e-300)
        num = D * _G(D - 1) * np.sin((D - 1) * np.arctan(q * xi))
        den = qr ** D * (1 + 1 / (q * xi) ** 2) ** ((D - 1) / 2)
        with np.errstate(divide="ignore", invalid="ignore"):
            out = 1 + num / den
        q0 = q * xi < 1e-6
        if np.any(q0):
            out[q0] = 1 + _G(D + 1) * (xi / r0) ** D
        return out


#: structure factors offered by the GUI: index -> (label, class, parameter keys)
STRUCTURE_FACTORS = [
    ("none", None, ()),
    ("hard spheres (Percus-Yevick)", PercusYevick, ("SFRadius", "SFEta")),
    ("sticky hard spheres (Baxter)", StickyHardSphere, ("SFRadius", "SFEta", "SFTau", "SFDelta")),
    ("mass fractal (Teixeira)", FractalTeixeira, ("SFRadius", "SFDim", "SFXi")),
]


def make_structure_factor(p):
    """Structure factor of the parameter dict (key StructureFactor), or None."""
    i = int(p.get("StructureFactor", 0) or 0)
    if i <= 0 or i >= len(STRUCTURE_FACTORS):
        return None
    _lab, cls, keys = STRUCTURE_FACTORS[i]
    return cls(*[float(p[k]) for k in keys])


# ----------------------------------------------------------------------------- cylinder / disk as in CrystalScatter
class CylinderRL:
    """Cylinder with *independent* Schulz-Zimm distributions of the radius R (relative width ``sigma_r``) and of the
    half-length L (``sigma_l``), as in CrystalScatter (where 'Length' is the half-length).  The amplitude of one
    cylinder at angle theta between q and the axis (mu = cos theta) factorises,

        F = [2 J1(q R sin theta) / (q R sin theta)] [sin(q L mu) / (q L mu)],

    so the size averages of the two factors are taken separately (hypergeometric blocks d = 2 and d = 1,
    reference weighting s = 0) and only the orientation average is done numerically:

        P(q) = int_0^1 <F_R^2>(q R sqrt(1-mu^2)) <F_L^2>(q L mu) dmu,   F(q) likewise with the amplitudes.

    A disk is the same body with a small L / R."""
    name = "cylinder_rl"

    def __init__(self, R, L, sigma_r=0.0, sigma_l=0.0, n_mu=None):
        from .blocks import z_from_sigma
        self.R, self.L = float(R), float(L)
        self.sigma, self.sigma_l = float(sigma_r), float(sigma_l)
        self.zr, self.zl = z_from_sigma(self.sigma), z_from_sigma(self.sigma_l)
        self.nu = self.L / self.R
        self.n_mu = n_mu

    def _mu(self, q):
        n = self.n_mu or int(np.clip(80 + 8 * float(np.max(q)) * max(self.R, self.L), 120, 2000))
        x, w = np.polynomial.legendre.leggauss(n)
        return 0.5 * (x + 1), 0.5 * w

    def _avg(self, q, kind):
        from .blocks import BLOCK
        q = np.atleast_1d(np.asarray(q, float))
        mu, w = self._mu(q)
        s = np.sqrt(1 - mu * mu)
        out = np.empty(q.shape)
        for i, qi in enumerate(q):
            a = BLOCK[2].value(qi * self.R * s, self.zr, 0, kind)
            b = BLOCK[1].value(qi * self.L * mu, self.zl, 0, kind)
            out[i] = np.sum(w * a * b)
        return out

    def P(self, q):
        return self._avg(q, 'P')

    def F(self, q):
        return self._avg(q, 'F')

    def beta(self, q):
        return self.F(q) ** 2 / self.P(q)

    def P_fixed(self, q, mu):
        """Fixed orientation (mu = cos of the angle between q and the axis)."""
        from .blocks import BLOCK
        q, mu = np.broadcast_arrays(np.asarray(q, float), np.asarray(mu, float))
        s = np.sqrt(np.clip(1 - mu * mu, 0, 1))
        return BLOCK[2].value(q * self.R * s, self.zr, 0, 'P') * BLOCK[1].value(q * self.L * np.abs(mu), self.zl, 0, 'P')

    def P_vec(self, qb):
        """Body frame q vector (..., 3), axis along z (for the oriented-pattern routines)."""
        qb = np.asarray(qb, float)
        qn = np.linalg.norm(qb, axis=-1)
        mu = np.where(qn > 0, qb[..., 2] / np.maximum(qn, 1e-300), 1.0)
        return self.P_fixed(qn, mu)


# ----------------------------------------------------------------------------- vesicle as in CrystalScatter
class VesicleRD:
    """Unilamellar vesicle: a spherical shell between the inner radius ``r_in`` and the outer radius ``r_out``
    (CrystalScatter: inner = 'EditRadius', outer = 'Length').  The mean radius r = (r_in + r_out)/2 and the
    half-thickness d = (r_out - r_in)/2 follow independent Schulz-Zimm distributions (``sigma_r``, ``sigma_d``).
    Exact shell amplitude, averaged numerically:

        F = [r_o^3 Phi(q r_o) - r_i^3 Phi(q r_i)] / (r_o^3 - r_i^3),   Phi(x) = 3 (sin x - x cos x) / x^3,

    P = <F^2> (number weights).  CrystalScatter evaluates the same model in a thin-shell closed form
    (polyvesicle); see the documentation for the differences."""
    name = "vesicle"

    def __init__(self, r_in, r_out, sigma_r=0.0, sigma_d=0.0, n=32):
        if not 0 < r_in < r_out:
            raise ValueError("vesicle: 0 < inner radius < outer radius")
        self.r, self.d = 0.5 * (r_in + r_out), 0.5 * (r_out - r_in)
        self.sigma, self.sigma_d = float(sigma_r), float(sigma_d)
        self._nodes = [self._schulz(self.r, self.sigma, n), self._schulz(self.d, self.sigma_d, n)]

    @staticmethod
    def _schulz(mean, s, n):
        if s <= 0:
            return np.array([mean]), np.array([1.0])
        z = (1 - s * s) / (s * s)
        t = np.linspace(max(1e-3, 1 - 7 * s), 1 + 7 * s, 4 * n + 1)
        lw = z * np.log(t) - (z + 1) * t
        w = np.exp(lw - lw.max())
        return mean * t, w / w.sum()

    def _amp(self, q):
        q = np.atleast_1d(np.asarray(q, float))[:, None, None]
        (r, wr), (d, wd) = self._nodes
        ro = (r[:, None] + d[None, :])[None]
        ri = np.maximum(r[:, None] - d[None, :], 0.0)[None]

        def phi(x):
            x = np.maximum(x, 1e-8)
            return np.where(x < 1e-3, 1 - x * x / 10, 3 * (np.sin(x) - x * np.cos(x)) / x ** 3)
        F = (ro ** 3 * phi(q * ro) - ri ** 3 * phi(q * ri)) / np.maximum(ro ** 3 - ri ** 3, 1e-300)
        return F, (wr[:, None] * wd[None, :])[None]

    def P(self, q):
        F, w = self._amp(q)
        return (F * F * w).sum((1, 2))

    def F(self, q):
        F, w = self._amp(q)
        return (F * w).sum((1, 2))

    def beta(self, q):
        return self.F(q) ** 2 / self.P(q)
