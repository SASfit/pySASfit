"""Multi-shell (core/shell) particles with hypergeometric Schulz averaging (SI Sect. 1.4).

Shells are given by relative outer radii p_1 < ... < p_N = 1 and densities rho_1..rho_N
(inside -> outside); rho_{N+1} = rho_solvent.  All dimensions scale with R (Schulz-Zimm).
Amplitude  A(q) = sum_i (rho_i - rho_{i+1}) p_i^d F_d(q p_i R),  normalised by A(0).
"""
import numpy as np
from .blocks import BLOCK, cross3, z_from_sigma
from .shapes import Shape, Ellipsoid, Cylinder, SizeQuad, _gl, volume_ratio


def _contrasts(p, rho, rho_solv, d=3):
    p = np.asarray(p, float)
    rho = np.asarray(rho, float)
    nxt = np.append(rho[1:], rho_solv)
    return (rho - nxt) * p ** d


class CoreShellSphere(Shape):
    name = "coreshell_sphere"

    def __init__(self, R, p, rho, sigma=0.0, rho_solvent=0.0):
        super().__init__(R, sigma)
        self.p = np.asarray(p, float)
        assert abs(self.p[-1] - 1) < 1e-12 and np.all(np.diff(self.p) > 0)
        self.c = _contrasts(self.p, rho, rho_solvent)
        self.c0 = self.c.sum()

    def _P(self, y):
        out = 0.0
        n = len(self.p)
        for i in range(n):
            for j in range(i + 1):
                w = self.c[i] * self.c[j] * (1 if i == j else 2)
                out = out + w * cross3(self.p[i], self.p[j], y, self.z, self.sP)
        return out / self.c0 ** 2

    def P(self, q):
        return self._P(np.asarray(q, float) * self.R)

    def F(self, q):
        y = np.asarray(q, float) * self.R
        return sum(self.c[i] * BLOCK[3].value(self.p[i] * y, self.z, self.sF, 'F') for i in range(len(self.p))) / self.c0


class CoreShellEllipsoid(Ellipsoid):
    """Ellipsoid (R, R, nu R) with proportionally scaled shells."""
    name = "coreshell_ellipsoid"

    def __init__(self, R, nu, p, rho, sigma=0.0, rho_solvent=0.0):
        super().__init__(R, nu, sigma)
        self.p = np.asarray(p, float)
        self.c = _contrasts(self.p, rho, rho_solvent)
        self.c0 = self.c.sum()

    def P_fixed(self, q, mu):
        q = np.asarray(q, float); mu = np.asarray(mu, float)
        y = q * self.R * self._r(mu)
        out = 0.0
        for i in range(len(self.p)):
            for j in range(i + 1):
                w = self.c[i] * self.c[j] * (1 if i == j else 2)
                out = out + w * cross3(self.p[i], self.p[j], y, self.z, self.sP)
        return out / self.c0 ** 2

    def F_fixed(self, q, mu):
        q = np.asarray(q, float); mu = np.asarray(mu, float)
        y = q * self.R * self._r(mu)
        return sum(self.c[i] * BLOCK[3].value(self.p[i] * y, self.z, self.sF, 'F') for i in range(len(self.p))) / self.c0


class CoreShellCylinder(Cylinder):
    """Cylinder with coaxial shells scaled proportionally in radius and length (p_i)."""
    name = "coreshell_cylinder"

    def __init__(self, R, nu, p, rho, sigma=0.0, rho_solvent=0.0):
        super().__init__(R, nu, sigma)
        self.p = np.asarray(p, float)
        self.c = _contrasts(self.p, rho, rho_solvent)  # p^3 ~ volume of proportional shells
        self.c0 = self.c.sum()

    def _amp(self, x, mu, xi):
        s = np.sqrt(np.clip(1 - mu ** 2, 0, 1))
        tot = 0.0
        for pi, ci in zip(self.p, self.c):
            tot = tot + ci * BLOCK[2].mono(pi * x * xi * s, 'F') * BLOCK[1].mono(pi * x * xi * self.nu * mu, 'F')
        return tot / self.c0


# ---------------------------------------------------------------------------------------------
# builders for radially structured spheres (inhomogeneous shells, onion / myelin-type multilayers)
# All radii scale with R (Schulz-Zimm).  The profile is discretised into thin homogeneous shells whose
# exact hypergeometric cross terms are summed (converges as 1/n^2 with the number of shells n).
# ---------------------------------------------------------------------------------------------
def gradient_shell_sphere(R, p_core, rho_core, rho_shell, sigma=0.0, n=40, rho_solvent=0.0):
    """Homogeneous core (relative radius p_core, density rho_core) + shell between p_core and 1 whose density
    follows rho_shell(t), t in [0,1] (0 at the core, 1 at the outer surface); e.g. a linear or power-law gradient."""
    edges = np.linspace(p_core, 1.0, n + 1)
    mid = 0.5 * (edges[:-1] + edges[1:])
    t = (mid - p_core) / (1 - p_core)
    p = np.concatenate([[p_core], edges[1:]])
    rho = np.concatenate([[rho_core], [rho_shell(x) for x in t]])
    return CoreShellSphere(R, p, rho, sigma, rho_solvent)


def multilayer_sphere(R, p_core, rho_core, layers, sigma=0.0, rho_solvent=0.0):
    """Onion: core (p_core, rho_core) followed by `layers` = [(thickness_fraction_of_R, rho), ...]; the
    thicknesses must add up to 1 - p_core."""
    th = np.array([l[0] for l in layers], float)
    assert abs(p_core + th.sum() - 1.0) < 1e-9, "p_core + sum(thicknesses) must equal 1"
    p = np.concatenate([[p_core], p_core + np.cumsum(th)])
    p[-1] = 1.0
    rho = np.concatenate([[rho_core], [l[1] for l in layers]])
    return CoreShellSphere(R, p, rho, sigma, rho_solvent)


def myelin_sphere(R_inner, n, l_ex, l_in, rho_ex=0.0, rho_in=1.0, sigma=0.0, rho_core=0.0, rho_solvent=0.0):
    """Concentric-sphere (onion/myelin-type) stack: n bilayers, each = inner layer (thickness l_in, density
    rho_in) and an exterior layer (thickness l_ex, density rho_ex), around a core of radius R_inner.
    Lengths are given in absolute units; R (the Schulz-Zimm scale) is the outer radius."""
    d = l_in + l_ex
    Rout = R_inner + n * d
    layers = []
    for i in range(n):
        layers += [(l_in / Rout, rho_in), (l_ex / Rout, rho_ex)]
    return multilayer_sphere(Rout, R_inner / Rout, rho_core, layers, sigma, rho_solvent)
