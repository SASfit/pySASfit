"""Sub-unit library.

Every sub-unit provides, as functions of the scattering vector q,

* ``ff``   – normalised form factor F(q) (pair-distance average of
  sin(q r)/(q r) over all pairs of scatterers),
* ``amp``  – form-factor amplitude A_ref(q) relative to a reference point,
* ``psi``  – phase factor Psi_{r1,r2}(q) between two reference points.

For a *distributed* reference point (random point on a surface/contour),
``psi(q, p, "surface", "surface")`` is the phase factor between two
independent random points of that distribution.

Expressions are the standard ones of the small-angle scattering literature
(Debye 1947; Hammouda 1992; Guinier/Fournet; Svaneborg & Pedersen,
J. Chem. Phys. 136, 104105 & 154907 (2012)).  Reference-point and parameter
names follow the SEB / pyseb conventions so structures are interchangeable.
"""
import numpy as np
from scipy import special as sp

from . import special as sf


class SubUnit:
    """Base class.  Sub-classes define ``size_params``, ``specific``, ``distributed``."""

    kind = "SubUnit"
    size_params = ()
    specific = ()
    distributed = ()

    @classmethod
    def references(cls):
        return tuple(cls.specific) + tuple(cls.distributed)

    @classmethod
    def is_distributed(cls, ref):
        return ref in cls.distributed

    @classmethod
    def check_ref(cls, ref):
        if ref not in cls.references():
            raise KeyError(f"{cls.kind} has no reference point '{ref}' "
                           f"(available: {', '.join(cls.references())})")

    # -- to be provided by sub-classes ------------------------------------
    def ff(self, q, p):
        raise NotImplementedError

    def amp(self, q, p, ref):
        raise NotImplementedError

    def psi(self, q, p, r1, r2):
        raise NotImplementedError

    # Mean-square distances (q -> 0 limits): <r^2> = 6 * (1 - X)/q^2 ...
    # are obtained generically in Structure.rg2 from the series of F/A/Psi.


class TableSubUnit(SubUnit):
    """Sub-unit given by closed-form expressions.

    ``_F(q, p)``; ``_A[ref](q, p)``; ``_P[(r1, r2)](q, p)`` with keys in
    any order.
    """
    _F = None
    _A = {}
    _P = {}

    def ff(self, q, p):
        return self._F(q, p)

    def amp(self, q, p, ref):
        self.check_ref(ref)
        return self._A[ref](q, p)

    def psi(self, q, p, r1, r2):
        if r1 == r2 and r1 not in self.distributed:
            return np.ones_like(np.asarray(q, dtype=float))
        f = self._P.get((r1, r2)) or self._P.get((r2, r1))
        if f is None:
            raise KeyError(f"{self.kind}: no phase factor between '{r1}' and '{r2}'")
        return f(q, p)


# --------------------------------------------------------------------------
class Point(TableSubUnit):
    """Massless junction point (no scattering, SEB convention)."""
    kind = "Point"
    specific = ("point",)
    _F = staticmethod(lambda q, p: np.zeros_like(q))
    _A = {"point": lambda q, p: np.zeros_like(q)}
    _P = {}


def _xg(q, p):
    return (q * p["Rg"]) ** 2


class GaussianPolymer(TableSubUnit):
    kind = "GaussianPolymer"
    size_params = ("Rg",)
    specific = ("end1", "middle", "end2")
    distributed = ("contour",)
    _F = staticmethod(lambda q, p: sf.debye(_xg(q, p)))
    _A = {
        "end1": lambda q, p: sf.one_minus_exp_over_x(_xg(q, p)),
        "end2": lambda q, p: sf.one_minus_exp_over_x(_xg(q, p)),
        # middle: two half chains (Rg^2/2 each) seen from their common end
        "middle": lambda q, p: sf.one_minus_exp_over_x(_xg(q, p) / 2),
        "contour": lambda q, p: sf.debye(_xg(q, p)),
    }
    _P = {
        ("end1", "end2"): lambda q, p: np.exp(-_xg(q, p)),
        ("end1", "middle"): lambda q, p: np.exp(-_xg(q, p) / 2),
        ("end2", "middle"): lambda q, p: np.exp(-_xg(q, p) / 2),
        ("contour", "end1"): lambda q, p: sf.one_minus_exp_over_x(_xg(q, p)),
        ("contour", "end2"): lambda q, p: sf.one_minus_exp_over_x(_xg(q, p)),
        ("contour", "middle"): lambda q, p: sf.one_minus_exp_over_x(_xg(q, p) / 2),
        ("contour", "contour"): lambda q, p: sf.debye(_xg(q, p)),
    }


class GaussianLoop(TableSubUnit):
    """Gaussian ring; Rg^2 = N b^2 / 12."""
    kind = "GaussianLoop"
    size_params = ("Rg",)
    distributed = ("contour",)
    _F = staticmethod(lambda q, p: sf.ring_debye(_xg(q, p)))
    _A = {"contour": lambda q, p: sf.ring_debye(_xg(q, p))}
    _P = {("contour", "contour"): lambda q, p: sf.ring_debye(_xg(q, p))}


def _xl(q, p):
    return q * p["L"]


class ThinRod(TableSubUnit):
    kind = "ThinRod"
    size_params = ("L",)
    specific = ("end1", "end2", "middle")
    distributed = ("contour",)
    _F = staticmethod(lambda q, p: sf.thin_rod_ff(_xl(q, p)))
    _A = {
        "end1": lambda q, p: sf.si_over_x(_xl(q, p)),
        "end2": lambda q, p: sf.si_over_x(_xl(q, p)),
        "middle": lambda q, p: sf.si_over_x(_xl(q, p) / 2),
        "contour": lambda q, p: sf.thin_rod_ff(_xl(q, p)),
    }
    _P = {
        ("end1", "end2"): lambda q, p: sf.sinc(_xl(q, p)),
        ("end1", "middle"): lambda q, p: sf.sinc(_xl(q, p) / 2),
        ("end2", "middle"): lambda q, p: sf.sinc(_xl(q, p) / 2),
        ("contour", "end1"): lambda q, p: sf.si_over_x(_xl(q, p)),
        ("contour", "end2"): lambda q, p: sf.si_over_x(_xl(q, p)),
        ("contour", "middle"): lambda q, p: sf.si_over_x(_xl(q, p) / 2),
        ("contour", "contour"): lambda q, p: sf.thin_rod_ff(_xl(q, p)),
    }


def _xr(q, p):
    return q * p["R"]


class SolidSphere(TableSubUnit):
    kind = "SolidSphere"
    size_params = ("R",)
    specific = ("center",)
    distributed = ("surface",)
    _F = staticmethod(lambda q, p: sf.sphere_amp(_xr(q, p)) ** 2)
    _A = {
        "center": lambda q, p: sf.sphere_amp(_xr(q, p)),
        "surface": lambda q, p: sf.sphere_amp(_xr(q, p)) * sf.sinc(_xr(q, p)),
    }
    _P = {
        ("center", "surface"): lambda q, p: sf.sinc(_xr(q, p)),
        ("surface", "surface"): lambda q, p: sf.sinc(_xr(q, p)) ** 2,
    }


class ThinSphericalShell(TableSubUnit):
    kind = "ThinSphericalShell"
    size_params = ("R",)
    specific = ("center",)
    distributed = ("surface",)
    _F = staticmethod(lambda q, p: sf.sinc(_xr(q, p)) ** 2)
    _A = {
        "center": lambda q, p: sf.sinc(_xr(q, p)),
        "surface": lambda q, p: sf.sinc(_xr(q, p)) ** 2,
    }
    _P = {
        ("center", "surface"): lambda q, p: sf.sinc(_xr(q, p)),
        ("surface", "surface"): lambda q, p: sf.sinc(_xr(q, p)) ** 2,
    }


class SolidSphericalShell(TableSubUnit):
    """Homogeneous shell Ri < r < Ro.  Distributed points: inner surface
    ``surfacei``, outer surface ``surfaceo`` and ``surface`` (either, area
    weighted)."""
    kind = "SolidSphericalShell"
    size_params = ("Ri", "Ro")
    specific = ("center",)
    distributed = ("surface", "surfaceo", "surfacei")

    @staticmethod
    def _ac(q, p):
        Ri, Ro = p["Ri"], p["Ro"]
        return (Ro**3 * sf.sphere_amp(q * Ro) - Ri**3 * sf.sphere_amp(q * Ri)) / (Ro**3 - Ri**3)

    @staticmethod
    def _g(q, p, ref):
        Ri, Ro = p["Ri"], p["Ro"]
        if ref == "center":
            return np.ones_like(q)
        if ref == "surfacei":
            return sf.sinc(q * Ri)
        if ref == "surfaceo":
            return sf.sinc(q * Ro)
        wo, wi = Ro**2, Ri**2
        return (wo * sf.sinc(q * Ro) + wi * sf.sinc(q * Ri)) / (wo + wi)

    def ff(self, q, p):
        return self._ac(q, p) ** 2

    def amp(self, q, p, ref):
        self.check_ref(ref)
        return self._ac(q, p) * self._g(q, p, ref)

    def psi(self, q, p, r1, r2):
        if r1 == r2 and r1 == "center":
            return np.ones_like(q)
        # centrosymmetric body: independent radial points -> product of sincs
        return self._g(q, p, r1) * self._g(q, p, r2)


class ThinCircle(TableSubUnit):
    """Infinitely thin circular ring (wire) of radius R."""
    kind = "ThinCircle"
    size_params = ("R",)
    specific = ("center",)
    distributed = ("contour",)
    _F = staticmethod(lambda q, p: sf.int_j0_over_z(2 * _xr(q, p)))
    _A = {
        "center": lambda q, p: sf.sinc(_xr(q, p)),
        "contour": lambda q, p: sf.int_j0_over_z(2 * _xr(q, p)),
    }
    _P = {
        ("center", "contour"): lambda q, p: sf.sinc(_xr(q, p)),
        ("contour", "contour"): lambda q, p: sf.int_j0_over_z(2 * _xr(q, p)),
    }


# --------------------------------------------------------------------------
class OrientedBody(SubUnit):
    """Axially symmetric body; orientational average over mu = cos(alpha)
    by Gauss-Legendre quadrature.

    Sub-classes supply ``_phi(q, mu, p)`` (orientation-dependent amplitude
    about the centre) and ``_g(ref, q, mu, p)`` = <exp(i q.r_ref)> for the
    reference point (averaged over its own distribution).  Then

        F         = <phi^2>
        A_ref     = <phi * conj(g_ref)>
        Psi_r1,r2 = <g_r1 * conj(g_r2)>.

    Closed forms may be returned from ``_closed(what, q, p, refs)``.
    """
    n_mu = 128

    def __init__(self, n_mu=None):
        self.set_n_mu(n_mu if n_mu is not None else self.n_mu)

    def set_n_mu(self, n_mu):
        """Number of Gauss-Legendre points of the orientational average."""
        self.n_mu = int(n_mu)
        x, w = np.polynomial.legendre.leggauss(self.n_mu)
        self._mu = 0.5 * (x + 1.0)
        self._w = 0.5 * w

    def _avg(self, integrand):
        return np.real(integrand) @ self._w

    def _grid(self, q):
        q = np.asarray(q, dtype=float)
        return q[:, None], self._mu[None, :]

    def _closed(self, what, q, p, refs):
        return None

    def ff(self, q, p):
        c = self._closed("ff", q, p, ())
        if c is not None:
            return c
        Q, M = self._grid(q)
        return self._avg(self._phi(Q, M, p) ** 2)

    def amp(self, q, p, ref):
        self.check_ref(ref)
        c = self._closed("amp", q, p, (ref,))
        if c is not None:
            return c
        Q, M = self._grid(q)
        return self._avg(self._phi(Q, M, p) * np.conj(self._g(ref, Q, M, p)))

    def psi(self, q, p, r1, r2):
        if r1 == r2 and r1 not in self.distributed:
            return np.ones_like(np.asarray(q, dtype=float))
        c = self._closed("psi", q, p, (r1, r2))
        if c is None:
            c = self._closed("psi", q, p, (r2, r1))
        if c is not None:
            return c
        Q, M = self._grid(q)
        return self._avg(self._g(r1, Q, M, p) * np.conj(self._g(r2, Q, M, p)))


class ThinDisk(OrientedBody):
    kind = "ThinDisk"
    size_params = ("R",)
    specific = ("center",)
    distributed = ("surface", "rim")

    def _phi(self, Q, M, p):
        return sf.jinc(Q * p["R"] * np.sqrt(1 - M**2))

    def _g(self, ref, Q, M, p):
        qp = Q * p["R"] * np.sqrt(1 - M**2)
        if ref == "center":
            return np.ones_like(qp)
        if ref == "surface":
            return sf.jinc(qp)
        return sp.j0(qp)  # rim

    def _closed(self, what, q, p, refs):
        x = q * p["R"]
        if what == "ff" or refs in (("surface",), ("surface", "surface")):
            return sf.thin_disk_ff(x)
        if refs in (("center",), ("center", "surface")):
            return sf.disk_center_amp(x)
        if refs == ("center", "rim"):
            return sf.sinc(x)
        if refs == ("rim", "rim"):
            return sf.int_j0_over_z(2 * x)
        if refs == ("rim", "surface"):
            return self.amp(q, p, "rim")
        return None


class SolidCylinder(OrientedBody):
    """Cylinder radius R, length L.  ``top``/``bottom`` = centres of the end
    caps; distributed ``ends`` (random point on either cap), ``hull``
    (mantle) and ``surface`` (whole surface, area weighted)."""
    kind = "SolidCylinder"
    size_params = ("R", "L")
    specific = ("center", "top", "bottom")
    distributed = ("hull", "ends", "surface")

    def _phi(self, Q, M, p):
        return sf.jinc(Q * p["R"] * np.sqrt(1 - M**2)) * sf.sinc(Q * p["L"] * M / 2)

    def _g(self, ref, Q, M, p):
        R, L = p["R"], p["L"]
        qp, qz = Q * R * np.sqrt(1 - M**2), Q * L * M / 2
        if ref == "center":
            return np.ones_like(qp)
        if ref == "top":
            return np.exp(1j * qz)
        if ref == "bottom":
            return np.exp(-1j * qz)
        g_ends = sf.jinc(qp) * np.cos(qz)
        g_hull = sp.j0(qp) * sf.sinc(qz)
        if ref == "ends":
            return g_ends
        if ref == "hull":
            return g_hull
        return (L * g_hull + R * g_ends) / (L + R)  # area 2 pi R L : 2 pi R^2

    def _closed(self, what, q, p, refs):
        if refs in (("center", "top"), ("center", "bottom")):
            return sf.sinc(q * p["L"] / 2)
        if refs == ("top", "bottom"):
            return sf.sinc(q * p["L"])
        return None


class Spheroid(OrientedBody):
    """Ellipsoid of revolution, equatorial semi-axis a, polar semi-axis c.

    ``north``/``south`` poles (``pole`` is an alias of ``north``), and the
    distributed ``surface``.  ``surface_measure``: "area" (uniform per
    surface area) or "parametric" (uniform in cos(theta) of the spheroidal
    parametrisation, as in pyseb 0.4).  Identical for a sphere.
    """
    kind = "Spheroid"
    size_params = ("a", "c")
    specific = ("center", "pole", "north", "south")
    distributed = ("surface",)

    def __init__(self, n_mu=None, n_theta=64, surface_measure="area"):
        super().__init__(n_mu)
        t, w = np.polynomial.legendre.leggauss(n_theta)
        self._ct = 0.5 * (t + 1.0)          # cos(theta) on [0, 1] (symmetric)
        self._wt = 0.5 * w
        self.surface_measure = surface_measure

    def _phi(self, Q, M, p):
        a, c = p["a"], p["c"]
        return sf.sphere_amp(Q * np.sqrt(a**2 * (1 - M**2) + c**2 * M**2))

    def _g(self, ref, Q, M, p):
        a, c = p["a"], p["c"]
        qz = Q * c * M
        if ref == "center":
            return np.ones_like(qz)
        if ref in ("north", "pole"):
            return np.exp(1j * qz)
        if ref == "south":
            return np.exp(-1j * qz)
        ct = self._ct[None, None, :]
        st = np.sqrt(1 - ct**2)
        w = self._wt
        if self.surface_measure == "area":
            w = w * np.sqrt(c**2 * (1 - self._ct**2) + a**2 * self._ct**2)
            w = w / w.sum()
        integrand = sp.j0(Q[..., None] * a * st * np.sqrt(1 - M[..., None] ** 2)) \
            * np.cos(qz[..., None] * ct)
        return integrand @ w

    def _closed(self, what, q, p, refs):
        y = q * p["c"]
        if refs in (("center", "north"), ("center", "south"), ("center", "pole")):
            return sf.sinc(y)
        if refs in (("north", "south"), ("pole", "south")):
            return sf.sinc(2 * y)
        if refs in (("pole", "north"),):
            return np.ones_like(np.asarray(q, dtype=float))
        return None


REGISTRY = {cls.kind: cls for cls in (
    Point, GaussianPolymer, GaussianLoop, ThinRod, SolidSphere,
    ThinSphericalShell, SolidSphericalShell, ThinCircle, ThinDisk,
    SolidCylinder, Spheroid)}
