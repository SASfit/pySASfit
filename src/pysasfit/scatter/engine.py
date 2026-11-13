"""Calculation back-end of the GUI: parameter dict (see params.py) -> hypscat objects -> 1D curve or 2D image.

Units: lengths in nm, q in 1/nm, detector distance and pixel sizes in m (only their ratio matters).
Where the original C++ GUI semantics were not fully documented, the mapping chosen here is stated in the
docstrings and in the README ("GUI" section).
"""
import functools

import numpy as np
from . import (Sphere, Ellipsoid, TriaxialEllipsoid, Cylinder, Disk, Cube, Superball, SuperEllipsoid,
                CoreShellSphere, CoreShellEllipsoid, CoreShellCylinder, ExcludedVolumeChain, Crystal,
                zone_axis_rotation)
from . import detector, orient, oriented
from .lattices import domain_width, _DIM
from .coreshell import multilayer_sphere, myelin_sphere
from .models_ext import (KratkyPorodChain, TeubnerStrey, powerlaw_shell_sphere, make_structure_factor, CylinderRL,
                         VesicleRD)
from .params import LTYPE_KIND, PARTICLES, ORDIS, PEAKS, CUBIC


#: points of the P(q) table from which the pixels of an isotropic 2D pattern are interpolated
P_TABLE_POINTS = 600


class CalcError(Exception):
    pass


def make_particle(p):
    """Particle selected by ComboBoxParticle / ComboBoxInterior.
    R = EditRadius; aspect ratio nu = Length/(2R) (cylinder, disk) or Length/R (ellipsoids); EditRadiusi:
    shell thickness (vesicle) / outer radius (core+shell) / third semi-axis (triaxial)."""
    R, sg = float(p["EditRadius"]), float(p["EditSigma"])
    L, Ri = float(p["Length"]), float(p["EditRadiusi"])
    if R <= 0:
        raise CalcError("EditRadius must be > 0")
    kind, inter = int(p["ComboBoxParticle"]), int(p["ComboBoxInterior"])
    rho_sh = float(p["EditRho"]) if p["EditRho"] else 1.0
    if kind == 9:
        return ExcludedVolumeChain(R, sigma=sg)
    if kind == 10:                       # worm-like chain: Kuhn length = EditRadius, contour length = Length
        return KratkyPorodChain(L if L > 0 else 100 * R, R, sigma=float(p["SigmaL"]))
    if kind == 11:                       # Teubner-Strey: repeat distance = Length, correlation length = EditRadius
        return TeubnerStrey(L if L > 0 else 10 * R, R)
    if inter in (3, 4):
        if kind not in (0, 3):
            raise CalcError("multi-shell / myelin interior: sphere or vesicle only")
        n = max(int(p.get("ShellN", 3)), 1)
        t = float(p.get("ShellThick", 2.0))
        if inter == 3:                   # onion: core EditRadius, n shells of thickness t, contrasts alternate
            Rout = R + n * t
            layers = [(t / Rout, rho_sh if i % 2 == 0 else float(p.get("ShellRho2", 0.0))) for i in range(n)]
            return multilayer_sphere(Rout, R / Rout, 1.0 if kind == 0 else 0.0, layers, sg)
        # myelin: core EditRadius (water), n bilayers of EditRadiusi (lipid, EditRho) + t (water)
        return myelin_sphere(R, n, t, max(Ri, 1e-6), rho_ex=0.0, rho_in=rho_sh, sigma=sg)
    if kind == 3:                         # vesicle (CrystalScatter): inner radius EditRadius, outer radius Length
        if L <= R:
            raise CalcError("vesicle: Length (outer radius) must exceed EditRadius (inner radius)")
        return VesicleRD(R, L, sg, float(p["SigmaL"]))
    if inter in (1, 2):
        if Ri <= R:
            raise CalcError("core+shell: EditRadiusi (outer radius) must exceed EditRadius (core radius)")
        Rt, pc = Ri, R / Ri
        rho = [1.0, rho_sh]
        if kind == 0:
            if inter == 2:
                if int(p.get("ShellProfile", 0)) == 1:        # algebraic profile (Förster & Burger 1998)
                    return powerlaw_shell_sphere(R, Rt, float(p.get("ProfileAlpha", -4 / 3)), 1.0, rho_sh, sg)
                from .coreshell import gradient_shell_sphere
                return gradient_shell_sphere(Rt, pc, 1.0, lambda t: rho_sh * (1 - t), sg)
            return CoreShellSphere(Rt, [pc, 1.0], rho, sg)
        nu = max(L / Rt, 1e-3)
        if kind == 5:
            return CoreShellEllipsoid(Rt, nu, [pc, 1.0], rho, sg)
        if kind in (1, 2):
            return CoreShellCylinder(Rt, max(L / Rt, 1e-3), [pc, 1.0], rho, sg)       # L = half-length
        raise CalcError("core+shell interior is available for sphere, cylinder, disk, ellipsoid, vesicle")
    if kind == 0:
        return Sphere(R, sg)
    if kind == 1:                         # CrystalScatter: Length = half-length; sigma on R, SigmaL on L
        return CylinderRL(R, max(L, 1e-6), sg, float(p["SigmaL"]))
    if kind == 2:                         # CrystalScatter: EditRadius = half-thickness, Length = disk radius
        return CylinderRL(max(L, 1e-6), R, float(p["SigmaL"]), sg)
    if kind == 4:
        return Cube(R, sg)
    if kind == 5:
        return Ellipsoid(R, max(L / R, 1e-3), sg)
    if kind == 6:
        return TriaxialEllipsoid(R, max(L / R, 1e-3), max(Ri / R, 1e-3) if Ri > 0 else 1.0, sg)
    if kind == 7:
        return SuperEllipsoid(R, max(L / R, 1e-3), 1.0, 2.0, float(p["Alpha"]), sg)
    if kind == 8:
        return Superball(R, float(p["Alpha"]) or 2.0, 1.0, 1.0, sg)
    raise CalcError("unknown particle")


def _h(ordis, width_deg):
    w = np.radians(max(width_deg, 0.5))
    base = ordis % 8 if ordis >= 8 else ordis
    if base in (0,):
        return orient.h_gauss(w)
    if base in (1, 5):
        return lambda d: np.exp(-np.abs(d) / w)
    if base == 2:
        return lambda d: np.cosh(np.cos(d) / w ** 2 * 0.5)       # Onsager-like
    if base == 3:
        return orient.h_maier_saupe(1.0 / w ** 2)
    if base == 4:
        return orient.h_cone(w)
    return orient.h_gauss(w)


def make_lattice(p, M_hint=None):
    kind = LTYPE_KIND.get(int(p["LType"]))
    if kind is None:
        return None
    pk = p["EditPeakPar"]
    # Lorentzian=Pearson VII beta=1, modified Lorentzian I: beta=2, II (intermediate): beta=3/2 (Foerster 2011, A.8);
    # Anisotropic Gaussian: handled as Gaussian here (use the API for per-axis widths)
    peak = {0: ("lorentzian", {}), 1: ("gaussian", {}), 2: ("pearson", {"m": 2.0}), 3: ("pearson", {"m": 1.5}),
            4: ("voigt", {"eta": min(max(pk / 100.0, 0.0), 1.0)}),     # CrystalScatter: eta = PeakPar / 100
            5: ("pearson", {"m": pk if pk > 0.5 else 2.0}),
            6: ("gamma", {"nu": pk if pk > 0 else 1.0}), 7: ("gaussian", {})}[int(p["ComboBoxPeak"])]
    delta, para = peak_delta(p, kind)
    k = domain_width(4.0, peak[0])                    # width parameter of Crystal per unit of Foerster's delta
    kw = dict(width=k * delta, peak=peak[0], dw=0.0, hmax=int(p["HKLmax"]),          # |h|,|k|,|l| <= HKLmax as in CrystalScatter
              para=k * para, **peak[1])
    kw = {k: v for k, v in kw.items() if k != "hmax"}
    kw.update(hmax=int(p["HKLmax"]), powder_mode="radial_g")
    # only the cell parameters that define the lattice (params.CELL_KEYS): lamellae a; hexagonal a (, c);
    # cubic a; tetragonal a, c; 2D lattices a, b, gamma; 'orthorhombic' all six (any cell)
    if kind in ("lamellae", "hexcyl") or kind in CUBIC:
        c = Crystal(kind, p["uca"], **kw)
    elif kind == "bct":
        c = Crystal(kind, p["uca"], None, p["ucc"], **kw)
    elif kind == "hcp":
        c = Crystal(kind, p["uca"], None, p["ucc"] if p["ucc"] != p["uca"] else None, 90, 90, None, **kw)
    elif _DIM.get(kind, 3) == 2:
        c = Crystal(kind, p["uca"], p["ucb"], None, 90, 90, p["ucgamma"], **kw)
    else:
        c = Crystal(kind, p["uca"], p["ucb"], p["ucc"], p["ucalpha"], p["ucbeta"], p["ucgamma"], **kw)
    return c


#: nearest-neighbour distance per lattice (CrystalScatter 'dist' for the paracrystal width): a, a/sqrt2, a sqrt3/2
_DIST = {"fcc": np.sqrt(2.0) / 2, "fd3m": np.sqrt(2.0) / 2, "bcc": np.sqrt(3.0) / 2, "gyroid": np.sqrt(3.0) / 2,
         "plumbers": np.sqrt(3.0) / 2}


def debye_waller(p, q):
    """G(q) = exp(-sigma_DW^2 q^2 / 3): mean displacement sigma_DW (EditDebyeWaller) of the particles from their lattice
    points, isotropic in 3D (CrystalScatter: dwfactor = sigma^2/3; Foerster et al. 2005, eq. 25, with the absolute
    displacement).  It damps the Bragg term and returns the intensity to the diffuse P(q) at high q."""
    s = float(p.get("EditDebyeWaller", 0.0) or 0.0)
    return np.exp(-s * s * np.asarray(q, float) ** 2 / 3.0)


def peak_delta(p, kind=None):
    """Peak width delta (Foerster et al. 2011: Gaussian exp(-4x^2/(pi delta^2)), Lorentzian FWHM delta) and the
    paracrystal coefficient c (delta(q) = delta + c q^2), following CrystalScatter (sc_gpu_generic.h):

    * 'Debye-Scherrer' (RadButDebyeScherrer): delta = 4/D, D = EditDomainSize (eq. 92 of Foerster 2011);
    * 'paracrystal' (RadioButtonPara): delta = 4/D + (sigma_DW/dist)^2 dist q^2, dist = nearest-neighbour distance;
    * neither: delta = 1 (1/length), independent of D (as in CrystalScatter);
    * anisotropic Gaussian with isotropic orientation: delta = |(4/SigX, 4/SigY, 4/SigZ)| / 3."""
    D = max(float(p.get("EditDomainSize", 0) or 0), 1e-6)
    delta, para = 1.0, 0.0
    if p.get("RadButDebyeScherrer", True):
        delta = 4.0 / D
    if p.get("RadioButtonPara", False):
        delta = 4.0 / D
        dist = float(p["uca"]) * _DIST.get(kind, 1.0)
        para = float(p["EditDebyeWaller"]) ** 2 / max(dist, 1e-12)
    if int(p.get("ComboBoxPeak", 1)) == 7 and int(p.get("Ordis", 7)) == 7:
        sig = np.array([4.0 / max(float(p.get(k, 40.0)), 1e-6) for k in ("SigX", "SigY", "SigZ")])
        delta, para = float(np.linalg.norm(sig)) / 3.0, 0.0
    return delta, para


# ---------------------------------------------------------------------------------------------------------------
# Lattice factor in the form of CrystalScatter (sc_gpu_generic.h): every reflection G contributes a radial peak
# shape in |q| - |G| times an azimuthal factor in the angle psi between q and G,
#     Z(q) = 2 pi / (n V) sum_G m f sq(|q| - g) psi_ord(psi),   psi_ord = g3 A(psi) / N(g),   g3 = 2 pi / g^2,
# with the azimuthal width phiwidth = 4 / EditAzi (EditAzi = 0: no azimuthal dependence -> powder pattern).
# N(g) normalises A over the sphere (int_0^pi A sin psi dpsi; 2 for A = 1).  The radial shapes are those of the C++
# code, including its normalisation constants c1...c4 (see the documentation, 'Peak shapes', for the deviations
# of the modified Lorentzians and Pearson VII from unit area).
_C1, _C2, _C3 = 4.0, 4.0 * (np.sqrt(2.0) - 1.0), 4.0 * (2.0 ** (2.0 / 3.0) - 1.0)


def _cpp_shape(shape, x2, w, peakpar):
    """Radial peak shape sq(q - g) of CrystalScatter; x2 = (q - g)^2 / delta^2, w = delta."""
    from scipy.special import gamma as G
    if shape == 0:                                            # Lorentzian
        return np.sqrt(_C1) / (np.pi * w * (1.0 + _C1 * x2))
    if shape == 1:                                            # Gaussian
        return np.exp(-4.0 * x2 / np.pi) / (np.pi * w / 2.0)
    if shape == 2:                                            # modified Lorentzian I
        return 2.0 * np.sqrt(_C2) / (np.pi * w * (1.0 + _C1 * x2) ** 2)
    if shape == 3:                                            # modified Lorentzian II
        return np.sqrt(_C3) / (2.0 * w * (1.0 + _C1 * x2) ** 1.5)
    if shape == 4:                                            # pseudo-Voigt, eta = PeakPar / 100
        eta = peakpar / 100.0
        return eta * np.sqrt(_C1) / (np.pi * w * (1.0 + _C1 * x2)) + (1 - eta) * np.exp(-x2) / (np.sqrt(np.pi) * w)
    if shape == 7:                                            # anisotropic Gaussian (isotropic orientation)
        return np.exp(-4.0 * x2 / np.pi) / (np.pi * w / 2.0)
    if shape == 5:                                            # Pearson VII, beta = PeakPar
        b = peakpar
        c4 = 2.0 ** (1.0 / b) - 1.0
        return G(b) * c4 * 2.0 / (G(b - 0.5) * np.pi * w * (1.0 + 4.0 * c4 * x2) ** b)
    raise ValueError(shape)


def _cpp_azimuthal(shape, psi2, q2, phiw, peakpar):
    """Azimuthal factor A(psi) of CrystalScatter (before normalisation); psi2 = psi^2."""
    if not np.isfinite(phiw):
        return np.ones_like(psi2)
    x2phi = 4.0 * q2 / phiw ** 2                              # 'b/c-factor'
    x2psi = 4.0 * q2 / (np.pi * phiw ** 2)                    # 'a-factor'
    if shape == 0:
        return 1.0 / (1.0 + x2phi * psi2)
    if shape == 1:
        return np.exp(-x2psi * psi2)
    if shape == 2:
        return 1.0 / (1.0 + x2phi * psi2) ** 2
    if shape == 3:
        return 1.0 / (1.0 + x2phi * psi2) ** 1.5
    if shape == 4:
        eta = peakpar / 100.0
        return eta / (1.0 + x2phi * psi2) + (1 - eta) * np.exp(-x2psi * psi2)
    if shape == 5:
        return 1.0 / (1.0 + x2phi * psi2) ** peakpar
    raise ValueError(shape)


_GL200 = None


@functools.lru_cache(maxsize=65536)
def _cpp_norm(shape, g, phiw, peakpar):
    """N(g) = int_0^pi A_g(psi) sin psi dpsi with the argument a = 4 g^2 / (pi phiwidth^2) of the C++ code
    (gaussnorm3, lorentznorm3, pearsonnorm3; evaluated here by quadrature instead of their series)."""
    if not np.isfinite(phiw):
        return 2.0
    global _GL200
    a = 4.0 * g * g / (np.pi * phiw ** 2)
    if _GL200 is None:
        x, wq = np.polynomial.legendre.leggauss(200)
        _GL200 = (0.5 * np.pi * (x + 1), 0.5 * np.pi * wq)
    psi, wq = _GL200
    if shape == 1:
        f = np.exp(-a * psi ** 2)
    elif shape in (0, 4):                                     # pseudo-Voigt: the C++ code uses the Lorentzian norm
        f = 1.0 / (1.0 + a * psi ** 2)
    else:
        b = {2: 2.0, 3: 1.5, 5: peakpar}[shape]
        f = (1.0 + a * psi ** 2) ** (-b)
    return float(np.sum(wq * f * np.sin(psi)))


def cpp_lattice_factor(p, c, qx, qy, qz, delta=None, para=None, powder=None):
    """Z(q) of crystal ``c`` (reflections in the lab frame of the q vectors) in the CrystalScatter form above.
    delta / para: peak width and paracrystal coefficient (default: from :func:`peak_delta`).

    Reflections of equal |G| share the radial factor and the normalisation; the angles to all reflections of a
    shell come from one matrix product.  Gaussian factors are evaluated only where they exceed ~1e-20 of their
    maximum (exact to that level); Lorentzian-type factors everywhere (long tails).

    powder: no azimuthal factor (every reflection spread uniformly over its sphere |q| = g: Debye-Scherrer
    rings, exact orientation average of this form).  Default: True for Ordis = isotropic, where EditAzi does
    not apply."""
    shape = int(p["ComboBoxPeak"])
    peakpar = float(p.get("EditPeakPar", 0.0) or 0.0)
    if delta is None:
        delta, para = peak_delta(p, c.kind)
    azi = float(p.get("EditAzi", 0.0) or 0.0)
    if powder is None:
        powder = int(p.get("Ordis", 7)) == 7
    phiw = 4.0 / azi if azi > 0 and not powder else np.inf
    qx, qy, qz = (np.asarray(a, float) for a in (qx, qy, qz))
    shp = np.broadcast(qx, qy, qz).shape
    Q = np.stack([np.broadcast_to(a, shp).ravel() for a in (qx, qy, qz)], 1)
    q2 = (Q * Q).sum(1)
    q = np.sqrt(q2) + 1e-9
    U = Q / q[:, None]
    w = delta + (para or 0.0) * q2 if para else np.full(q.shape, float(delta))
    out = np.zeros(q.shape)
    Gm = np.linalg.norm(c.G, axis=0)
    ok = Gm > 0
    G, Wt, Gm = c.G[:, ok], np.asarray(c.weights, float)[ok], Gm[ok]
    shells, inv = np.unique(np.round(Gm, 9), return_inverse=True)
    gauss_r = shape in (1, 7)
    gauss_a = shape == 1 and np.isfinite(phiw)
    order = np.argsort(q)
    qs = q[order]
    for k, g in enumerate(shells):
        members = np.flatnonzero(inv == k)
        g = float(Gm[members[0]])
        if gauss_r:                    # exp(-4 x^2 / pi) < 1e-20 beyond |x| = 6.0
            wmax = float(np.max(w))
            lo, hi = np.searchsorted(qs, [g - 6.1 * wmax, g + 6.1 * wmax])
            idx = order[lo:hi]
            if idx.size == 0:
                continue
        else:
            idx = slice(None)
        qi, wi = q[idx], w[idx]
        sq = _cpp_shape(shape, (qi - g) ** 2 / wi ** 2, wi, peakpar)
        norm = _cpp_norm(shape, g, phiw, peakpar)
        if shape == 4:                 # C++: eta N_L + (1 - eta) N_G with N_G never set (0): the result scales as 1/eta
            norm *= peakpar / 100.0
        if shape == 7:                 # anisotropic Gaussian, isotropic orientation: g3 in sq, no azimuthal factor
            norm = 1.0
        pref = (2 * np.pi / g ** 2) / norm
        if not np.isfinite(phiw) or shape == 7:
            out[idx] += pref * sq * float(Wt[members].sum())
            continue
        cosp = np.clip(U[idx] @ (G[:, members] / g), -1.0, 1.0)          # (pixels, reflections of the shell)
        q2i = q2[idx][:, None]
        if gauss_a:                    # exp(-a psi^2) < 1e-20 for a psi^2 > 46
            x2psi = 4.0 * q2i / (np.pi * phiw ** 2)
            cmin = np.cos(np.minimum(np.sqrt(46.0 / np.maximum(x2psi, 1e-300)), np.pi))
            A = np.zeros(cosp.shape)
            m = cosp >= cmin
            if m.any():
                psi2 = np.arccos(cosp[m]) ** 2
                A[m] = np.exp(-np.broadcast_to(x2psi, cosp.shape)[m] * psi2)
        else:
            A = _cpp_azimuthal(shape, np.arccos(cosp) ** 2, q2i, phiw, peakpar)
        out[idx] += pref * sq * (A @ Wt[members])
    vol = c.v0 if c.dim == 3 else c.v0d
    return (2 * np.pi / vol * out).reshape(shp)


def _uses_cpp_lattice(p):
    """Peak shapes and orientations computed in the CrystalScatter form (all but Gamma, anisotropic Gaussian, fibre)."""
    sh, od = int(p["ComboBoxPeak"]), int(p["Ordis"])
    return (sh <= 5 and od != 13) or (sh == 7 and od == 7)


def _oriented_lattice(p, c):
    """The lattice with the zone axis [ucn1 ucn2 ucn3] along the beam (x), rotated by ucpsi about it.  Lamellae
    and rod lattices too: their cell is a (layer normal), b, c (rod axis), e.g. zone axis [001] of hexagonally
    packed cylinders: rods along the beam; [010] of lamellae: layers edge-on, normal horizontal."""
    R = zone_axis_rotation(c.M, (p["ucn1"], p["ucn2"], p["ucn3"]), to=(1, 0, 0), psi=np.radians(p["ucpsi"]))
    return Crystal(kind=c.kind, a=c.cellpar[0], b=c.cellpar[1], c=c.cellpar[2], alpha=c.cellpar[3],
                   beta=c.cellpar[4], gamma=c.cellpar[5], Rot=R, width=c.width, peak=c.peak_shape,
                   m=c.m, eta=c.eta, nu=c.nu, dw=c.dw, hmax=c.hmax, para=c.para, powder_mode=c.powder_mode)


def director_lab(p):
    """Director (theta, phi) of the particle axis in the frame of the engine (x along the beam, y horizontal, z vertical).
    CrystalScatter defines it in its own frame (detector qx, qy; z along the beam):
    n_C = (sin theta cos phi, sin theta sin phi, cos theta) -> n = (n_Cz, n_Cx, n_Cy)."""
    th, ph = np.radians(p["theta"]), np.radians(p["phi"])
    return np.array([np.cos(th), np.sin(th) * np.cos(ph), np.sin(th) * np.sin(ph)])


def _frame_with_axis(n):
    """Rotation (columns = body axes in the lab) whose body z axis is ``n``."""
    n = np.asarray(n, float) / np.linalg.norm(n)
    t = np.array([1.0, 0, 0]) if abs(n[0]) < 0.9 else np.array([0.0, 1, 0])
    e1 = np.cross(t, n); e1 /= np.linalg.norm(e1)
    e2 = np.cross(n, e1)
    return np.stack([e1, e2, n], axis=1)


def _particle_P(shape, qx, qy, qz, p, ctx=None):
    """Orientation-resolved single-particle intensity <P> and beta for the chosen Ordis."""
    ordis = int(p["Ordis"])
    q = np.sqrt(qx ** 2 + qy ** 2 + qz ** 2)
    iso = ordis == 7 or isinstance(shape, (Sphere, CoreShellSphere, ExcludedVolumeChain)) or \
        not hasattr(shape, "P_fixed")
    if iso:
        qq = np.sqrt(ctx["qfull"][0] ** 2 + ctx["qfull"][1] ** 2 + ctx["qfull"][2] ** 2) if ctx else q
        if ctx is not None and "Pg" not in ctx:
            ctx["qg"] = np.linspace(0, float(qq.max()) * 1.001 + 1e-9, P_TABLE_POINTS)
            ctx["Pg"] = shape.P(np.maximum(ctx["qg"], 1e-9))
        P = np.interp(q, ctx["qg"], ctx["Pg"]) if ctx is not None else shape.P(q.ravel()).reshape(q.shape)
        return P, _LazyBeta(shape, q, ctx)
    director = director_lab(p)
    if ordis != 6 and isinstance(shape, Cube):
        raise CalcError("cube: only 'isotropic' and 'z-dir' orientations are calculated")
    if ordis == 6:               # 'z-dir': perfect orientation along the director (theta, phi)
        P = oriented.perfect(shape, qx, qy, qz, rot=_frame_with_axis(director))
    elif ordis == 13:            # fiber pattern: uniform rotation about the beam (CrystalScatter z)
        P = _axial_table(shape, qx, qy, qz, lambda d: np.ones_like(d), np.array([1.0, 0, 0]), ctx=ctx)
    else:
        P = _axial_table(shape, qx, qy, qz, _h(ordis, p["EditDbeta"]), director, ctx=ctx)
    return P, _LazyBeta(shape, q, ctx)     # decoupling approximation with the orientation-averaged beta


class _LazyBeta:
    """beta(q), computed only when a lattice or structure factor needs it (it is the expensive part for
    anisometric particles)."""

    def __init__(self, shape, q, ctx):
        self.shape, self.q, self.ctx = shape, q, ctx

    def __call__(self):
        return _beta(self.shape, self.q, self.ctx)


def _beta(shape, q, ctx):
    """beta(q) = <F>^2/<P> of the (orientation averaged) particle, tabulated once per image."""
    if not hasattr(shape, "beta"):
        return np.ones_like(q)
    if ctx is None:
        return shape.beta(q.ravel()).reshape(q.shape)
    if "bg" not in ctx:
        qx, qy, qz = ctx["qfull"]
        qg = np.linspace(0, float(np.sqrt(qx ** 2 + qy ** 2 + qz ** 2).max()) * 1.001 + 1e-9, 200)
        ctx["qb"], ctx["bg"] = qg, shape.beta(np.maximum(qg, 1e-9))     # smooth: 200 points suffice
    return np.interp(q, ctx["qb"], ctx["bg"])


def _mono_copies(shape, n=21):
    """Gauss-(generalised)Laguerre nodes of the Schulz-Zimm radius distribution -> (weights, monodisperse shapes)."""
    import copy
    from scipy.special import roots_genlaguerre
    from .shapes import z_from_sigma
    if getattr(shape, "sigma", 0) <= 0:
        return np.array([1.0]), [shape]
    from scipy.special import gammaln
    z, sg = shape.z, shape.sigma
    t = 1 + sg * np.linspace(-4.5, 4.5, n)             # R / R0 on a uniform grid; Schulz-Zimm weights
    t = t[t > 0]
    lw = (z + getattr(shape, "sP", 0)) * np.log(t) - (z + 1) * t
    w = np.exp(lw - lw.max())
    w = w / w.sum()
    t = t * (z + 1) / (z + 1)
    out = []
    for ti in t:
        c = copy.copy(shape)
        c.R = shape.R * ti
        c.sigma = 0.0
        c.z = z_from_sigma(0.0)
        out.append(c)
    return w, out


def _axial_table(shape, qx, qy, qz, h, director, n_par=56, n_perp=36, n_delta=40, n_psi=16, ctx=None):
    """Particles with axis distribution h(delta) about `director` are axially symmetric about it, so
    I depends only on (q_par, q_perp): tabulate with the quadrature pattern for each node of the size
    distribution and interpolate to the pixels."""
    from scipy.interpolate import RegularGridInterpolator
    def split(x, y, z):
        par = x * director[0] + y * director[1] + z * director[2]
        return par, np.sqrt(np.maximum(x ** 2 + y ** 2 + z ** 2 - par ** 2, 0.0))
    qpar, qper = split(qx, qy, qz)
    f = ctx.get("f") if ctx is not None else None
    if f is None:
        rng = ctx.get("qfull", (qx, qy, qz)) if ctx is not None else (qx, qy, qz)
        pa, pe = split(*rng)
        qm = max(float(np.abs(pa).max()), float(pe.max()), 1e-9) * 1.001
        ga = np.linspace(-qm, qm, n_par)
        gp = np.linspace(0, qm, n_perp)
        A, B = np.meshgrid(ga, gp, indexing="ij")
        sg = float(getattr(shape, "sigma", 0) or 0)
        if isinstance(shape, CylinderRL):                 # size averages built into P_fixed
            w, shapes = np.array([1.0]), [shape]
        else:
            w, shapes = _mono_copies(shape, n=int(np.clip(round(7 + 70 * sg), 7, 21)))   # fewer nodes when narrow
        T = sum(wi * orient.pattern(s, A, B, h, n_delta=n_delta, n_psi=n_psi) for wi, s in zip(w, shapes))
        f = RegularGridInterpolator((ga, gp), T, bounds_error=False, fill_value=None)
        if ctx is not None:
            ctx["f"] = f
    return f(np.stack([qpar, qper], -1))


def _with_sf(p, shape, q, P, beta=None):
    """P (1 + β (S − 1)) with the isotropic structure factor of the parameters (decoupling approximation);
    P unchanged without a structure factor.  Only without a lattice (LType None)."""
    sf = make_structure_factor(p)
    if sf is None:
        return P
    S = sf.S(q.ravel()).reshape(q.shape)
    if callable(beta):
        beta = beta()
    if beta is None:
        beta = shape.beta(q.ravel()).reshape(q.shape) if hasattr(shape, "beta") else 1.0
    return P * (1 + beta * (S - 1))


def intensity_1d(p, q):
    """Isotropic I(q) at the given q (1/nm): powder-averaged lattice (LType) or the form factor, with the
    structure factor (no lattice), times I0 plus Base."""
    q = np.asarray(q, float)
    shape = make_particle(p)
    qq = np.maximum(q, 1e-9)
    P = shape.P(qq)
    c = make_lattice(p)
    if c is None:
        I = _with_sf(p, shape, qq, P)
    else:
        beta = shape.beta(qq) if hasattr(shape, "beta") else 1.0
        Z = cpp_lattice_factor(p, c, qq, 0 * qq, 0 * qq, powder=True) if _uses_cpp_lattice(p) else c.Z_powder(qq)
        I = P * (1 + beta * (Z - 1) * debye_waller(p, qq))
    return p["I0"] * I + p["Base"]


#: Gauss-Hermite nodes for the resolution smearing (probabilists' weights, sum 1)
_GH_X, _GH_W = np.polynomial.hermite_e.hermegauss(15)
_GH_W = _GH_W / _GH_W.sum()


def smeared_1d(p, q, dq=None):
    """I(q) folded with a Gaussian q resolution of standard deviation ``dq`` (pySASred Qdev, 1/nm); without
    ``dq`` (or all zero) the unsmeared intensity.  The model is evaluated once on a common grid."""
    q = np.asarray(q, float)
    if dq is None or not np.any(np.asarray(dq) > 0):
        return intensity_1d(p, q)
    dq = np.where(np.isfinite(dq) & (dq > 0), dq, 0.0)
    qs = np.abs(q[:, None] + dq[:, None] * _GH_X[None, :])
    lo, hi = max(float(qs.min()), 1e-6), float(qs.max())
    grid = np.geomspace(lo, hi * 1.0001, max(200, min(4000, 4 * q.size)))
    Ig = intensity_1d(p, grid)
    I = np.interp(qs, grid, Ig)
    return np.sum(I * _GH_W[None, :], axis=1)


def calc_1d(p):
    """I(q) on a log q grid: powder-averaged lattice (LType) or the bare form factor (LType = None)."""
    nq = int(p["EditQsteps"])
    q = np.geomspace(max(p["EditQmin"], 1e-6), max(p["EditQmax"], p["EditQmin"] * 1.01), nq)
    return q, intensity_1d(p, q)


def pixel_q(p, quadrant=True):
    """q components (qx, qy, qz) per pixel from the experiment group (Eq. 1)."""
    N = int(p["GridPoints"]) if quadrant else None
    nx, ny = int(p["EditPixelNoX"]), int(p["EditPixelNoY"])
    if p.get("CenterBeam", True):
        cy, cz = p["BeamPosX"] + (nx - 1) / 2 * 0, p["BeamPosY"]
    Y, Z = detector.pixel_grid(nx, ny, 1.0, center=((nx - 1) / 2 + p["BeamPosX"], (ny - 1) / 2 + p["BeamPosY"]))
    qx, qy, qz = detector.qmap(Y * p["EditPixelX"], Z * p["EditPixelY"], p["EditDet"], p["EditWavelength"])
    return qx, qy, qz + p["Editqz"]


def q_grid_2d(p):
    """(qx, qy, qz) of the calculated image: beam along x; detector plane (qy horizontal, qz vertical).
    'Preset': a Qmax-limited q map of GridPoints²; otherwise the pixels of the experiment group (Eq. 1)."""
    if p["EditQmaxPreset"]:
        N = int(p["GridPoints"])
        qm = float(p["EditQmax"])
        ax = np.linspace(-qm, qm, N)
        qy, qz = np.meshgrid(ax, ax, indexing="ij")
        return np.zeros_like(qy), qy, qz + p["Editqz"]
    return pixel_q(p)


def calc_2d(p, progress=None, should_abort=None):
    """2D detector image.  Image size from PixelNo.; q range from the detector geometry (or from Qmax in 'Preset'
    mode, where the image is a Qmax-limited q-map of GridPoints x GridPoints pixels).  Index [i, j]: qy[i], qz[j]."""
    qx, qy, qz = q_grid_2d(p)
    return intensity_2d(p, qx, qy, qz, progress, should_abort)


def intensity_2d(p, qx, qy, qz, progress=None, should_abort=None):
    """Intensity at arbitrary q vectors (arrays of one shape; beam along x): particle orientation distribution,
    oriented / fibre / powder lattice, structure factor (no lattice), times I0 plus Base."""
    qx, qy, qz = (np.asarray(a, float) for a in (qx, qy, qz))
    if qx.ndim == 1:                                       # pixel lists (fits): one row
        qx, qy, qz = qx[None, :], qy[None, :], qz[None, :]
        return intensity_2d(p, qx, qy, qz, progress, should_abort)[0]
    key = _cache_key(p, qx, qy, qz)
    hit = _CACHE.get(key)
    if hit is not None:
        return p["I0"] * hit + p["Base"]
    shape = make_particle(p)
    c = make_lattice(p)
    out = np.zeros(qy.shape)
    rows = np.array_split(np.arange(qy.shape[0]), max(1, min(16, qy.shape[0] // 8)))
    ctx = {"qfull": (qx, qy, qz)}
    cc = None
    if c is not None:
        cc = c if int(p["Ordis"]) == 7 else _oriented_lattice(p, c)
    for i, r in enumerate(rows):
        if should_abort and should_abort():
            raise CalcError("aborted")
        sl = (r, slice(None))
        QX, QY, QZ = qx[sl], qy[sl], qz[sl]
        q = np.sqrt(QX ** 2 + QY ** 2 + QZ ** 2)
        P, beta = _particle_P(shape, QX, QY, QZ, p, ctx)
        if cc is None:
            I = _with_sf(p, shape, q, P, beta)
        else:
            beta = beta()
            if _uses_cpp_lattice(p):
                Z = cpp_lattice_factor(p, cc, QX, QY, QZ)
            elif int(p["Ordis"]) == 7:
                Z = cc.Z_powder(q.ravel()).reshape(q.shape)
            else:                          # peak shapes of the Crystal class (Gamma, anisotropic Gaussian, fibre)
                try:
                    Z = cc.Z_fiber(QX, QY, QZ, npsi=240) if int(p["Ordis"]) == 13 else cc.Z(QX, QY, QZ)
                except ValueError as e:    # e.g. a peak shape that cannot be normalised in this dimension
                    raise CalcError(f"{PEAKS[int(p['ComboBoxPeak'])]} peak, {ORDIS[int(p['Ordis'])]}: {e}") from e
            I = P * (1 + beta * (Z - 1) * debye_waller(p, q))
        out[sl] = I
        if progress:
            progress(int(100 * (i + 1) / len(rows)))
    _CACHE.put(key, out)
    return p["I0"] * out + p["Base"]


#: keys that only scale or shift the intensity (not part of the cache key)
_SCALE_KEYS = ("I0", "Base", "iso", "Comment", "NewWindow", "QuadrantFactor", "Dimension")


def _cache_key(p, qx, qy, qz):
    items = tuple(sorted((k, v) for k, v in p.items() if k not in _SCALE_KEYS and not callable(v)))
    import hashlib
    h = hashlib.sha1()
    for a in (qx, qy, qz):
        h.update(np.ascontiguousarray(a).tobytes())
    return items, qx.shape, h.hexdigest()


class _LRU:
    def __init__(self, n=4):
        self.n, self.d = n, {}

    def get(self, k):
        v = self.d.pop(k, None)
        if v is not None:
            self.d[k] = v
        return v

    def put(self, k, v):
        self.d.pop(k, None)
        self.d[k] = v
        while len(self.d) > self.n:
            self.d.pop(next(iter(self.d)))


_CACHE = _LRU()
