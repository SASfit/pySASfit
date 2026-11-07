"""GISAS in the distorted wave Born approximation for axisymmetric particles (axis along z)
sitting on a substrate (four-term DWBA with Fresnel T/R of the substrate; particle at z=0).

Particle amplitude A(q_par, q_z) for Sphere / Ellipsoid (semi-axis nu*R along z) / Cylinder
(axis along z).  Size distribution: Schulz-Zimm via quadrature over R with V^2 weighting.
"""
import numpy as np
from .shapes import Sphere, Ellipsoid, Cylinder, Disk, SizeQuad
from .blocks import BLOCK


def fresnel(alpha, delta, beta, wavelength):
    """Fresnel reflection/transmission amplitudes of a vacuum/substrate interface, grazing angle alpha.
    n = 1 - delta + i beta."""
    n2 = (1 - delta + 1j * beta) ** 2
    s = np.sin(alpha)
    kz = s
    kzs = np.sqrt(n2 - np.cos(alpha) ** 2 + 0j)
    r = (kz - kzs) / (kz + kzs)
    return r, 1 + r


def _amp(shape, qpar, qz, R):
    x = np.sqrt(qpar ** 2 * 1.0 + 0 * qz) * R
    if isinstance(shape, Sphere):
        return BLOCK[3].mono(np.sqrt(qpar ** 2 + qz ** 2) * R, 'F')
    if isinstance(shape, Ellipsoid):
        return BLOCK[3].mono(R * np.sqrt(qpar ** 2 + (shape.nu * qz) ** 2), 'F')
    if isinstance(shape, Cylinder):
        return BLOCK[2].mono(qpar * R, 'F') * BLOCK[1].mono(shape.nu * qz * R, 'F')
    raise TypeError("DWBA: unsupported shape")


def dwba(shape, qy_par, alpha_i, alpha_f, two_theta_f, wavelength, delta, beta=0.0, buried=False):
    """Returns (I_diffuse_incoherent_part_P, |<A>|^2 coherent part) per particle, both normalised
    to a particle of unit volume amplitude; arrays follow the shape of alpha_f/two_theta_f.
    Use  I = P + beta_ratio*(Z-1)*Fcoh  with Fcoh = |<A_dwba>|^2 for the lattice-coupled term."""
    k = 2 * np.pi / wavelength
    af, tf = np.broadcast_arrays(np.asarray(alpha_f, float), np.asarray(two_theta_f, float))
    ri, ti = fresnel(alpha_i, delta, beta, wavelength)
    rf, tf_ = fresnel(af, delta, beta, wavelength)
    kiz = k * np.sin(alpha_i)
    kfz = k * np.sin(af)
    if buried:   # particles inside a film of index n2 = 1 - delta (SI Sect. 9): Snell's law for the angles
        n2 = 1 - delta
        kiz = k * np.real(np.sin(np.arccos(np.cos(alpha_i) / n2 + 0j)))
        kfz = k * np.real(np.sin(np.arccos(np.cos(af) / n2 + 0j)))
    qx = k * (np.cos(af) * np.cos(tf) - np.cos(alpha_i))
    qy = k * np.cos(af) * np.sin(tf)
    qpar = np.hypot(qx, qy)
    terms = [(ti * tf_, kfz - (-kiz)), (ti * rf, -kfz + kiz), (ri * tf_, kfz - kiz), (ri * rf, -kfz - kiz)]
    # note: q_z(+) = kfz + kiz etc. (k_i,z = -kiz, so q = kf - ki)
    terms = [(ti * tf_, kfz + kiz), (ti * rf, -kfz + kiz), (ri * tf_, kfz - kiz), (ri * rf, -kfz - kiz)]
    out = {}
    for s_w, key in ((shape.sP, 'P'), (shape.sF, 'F')):
        sq = SizeQuad(shape.sigma, s_w)
        Asum = 0
        for c, qz in terms:
            Asum = Asum + c[..., None] * _amp(shape, qpar[..., None], qz[..., None], shape.R * sq.xi)
        if key == 'P':
            out['P'] = (np.abs(Asum) ** 2 * sq.w).sum(-1)
        else:
            out['Fcoh'] = np.abs((Asum * sq.w).sum(-1)) ** 2
    out['qx'], out['qy'], out['qpar'] = qx, qy, qpar
    return out


# ----------------------------------------------------------------------------------------
# Specular and incident-plane diffuse scattering (SI Eq. S.9.2) and total GISAS intensity
# ----------------------------------------------------------------------------------------
def S_diffuse(qr, qz, sigma_z, xi, h=0.5, cutoff=False, rmax_factor=40.0):
    """Incident-plane diffuse scattering.  h = 1/2 & cutoff=False: analytic expression
    S = A pi/(qr^2 + A^2 qz^4/4)^(3/2), A = 2 sigma^2/xi^(2h).  cutoff=True: numerical Hankel integral
    S = 2 pi/qz^2 exp(-qz^2 s^2) int (exp(qz^2 s^2 exp(-(R/xi)^2h)) - 1) J0(qr R) R dR."""
    qr = np.asarray(qr, float); qz = np.asarray(qz, float)
    A = 2 * sigma_z ** 2 / xi ** (2 * h)
    if not cutoff:
        if abs(h - 0.5) > 1e-12:
            from scipy.integrate import quad
            from scipy.special import j0
            qr_b, qz_b = np.broadcast_arrays(qr, qz)
            out = np.empty(qr_b.shape)
            for idx in np.ndindex(out.shape):
                f = lambda R: np.exp(-qz_b[idx] ** 2 * A * R ** (2 * h) / 2) * j0(qr_b[idx] * R) * R
                out[idx] = 2 * np.pi / qz_b[idx] ** 2 * quad(f, 0, np.inf, limit=800)[0]
            return out
        return A * np.pi / (qr ** 2 + 0.25 * A ** 2 * qz ** 4) ** 1.5
    from scipy.integrate import quad
    from scipy.special import j0
    qr_b, qz_b = np.broadcast_arrays(qr, qz)
    out = np.empty(qr_b.shape)
    for idx in np.ndindex(out.shape):
        qzz, qrr = qz_b[idx], qr_b[idx]
        f = lambda R: (np.expm1(qzz ** 2 * sigma_z ** 2 * np.exp(-(R / xi) ** (2 * h)))) * j0(qrr * R) * R
        val = quad(f, 0, rmax_factor * xi, limit=2000)[0]
        out[idx] = 2 * np.pi / qzz ** 2 * np.exp(-qzz ** 2 * sigma_z ** 2) * val
    return out


def S_specular(qr, qz, alpha_i, wavelength, delta, beta, sigma_z, width_r, width_z):
    """Specular reflection 4 pi^2 |R|^2/qz^2 exp(-qz^2 sigma^2) delta(qr) delta(qz-2 k sin ai); the
    delta functions are represented by normalised Gaussians of widths (width_r, width_z)."""
    k = 2 * np.pi / wavelength
    Ri, _ = fresnel(alpha_i, delta, beta, wavelength)
    qzr = 2 * k * np.sin(alpha_i)
    g = lambda x, w: np.exp(-0.5 * (x / w) ** 2) / (np.sqrt(2 * np.pi) * w)
    return (4 * np.pi ** 2 * np.abs(Ri) ** 2 / qzr ** 2 * np.exp(-qzr ** 2 * sigma_z ** 2)
            * g(np.asarray(qr, float), width_r) * g(np.asarray(qz, float) - qzr, width_z))


def gisas_intensity(shape, alpha_i, alpha_f, two_theta_f, wavelength, delta, beta=0.0,
                    lattice=None, sigma_z=None, xi=None, h=0.5, spec_widths=None, cutoff=False, buried=False):
    """Total GISAS intensity per particle:  I = <P>_DWBA + Fcoh (Z-1) G  (lattice optional)
    + |T_i|^2 |T_f|^2 [S_spec + S_diff]  (if sigma_z and xi given; S_spec also needs spec_widths).
    Returns dict with 'I', 'qx','qy','qz' and components."""
    k = 2 * np.pi / wavelength
    o = dwba(shape, 0.0, alpha_i, alpha_f, two_theta_f, wavelength, delta, beta, buried=buried)
    af = np.asarray(alpha_f, float)
    qz = k * (np.sin(af) + np.sin(alpha_i))
    I = o['P'].copy()
    if lattice is not None:
        Z = lattice.Z(o['qx'], o['qy'], qz)
        q = np.sqrt(o['qx'] ** 2 + o['qy'] ** 2 + qz ** 2)
        I = I + o['Fcoh'] * (Z - 1) * lattice.G_dw(q)
    res = dict(I_particles=I, **o, qz=qz)
    if sigma_z is not None and xi is not None:
        _, Ti = fresnel(alpha_i, delta, beta, wavelength)
        _, Tf = fresnel(af, delta, beta, wavelength)
        w = np.abs(Ti) ** 2 * np.abs(Tf) ** 2
        Sd = S_diffuse(o['qpar'] * np.sign(o['qy'] + 1e-300) * 0 + o['qx'], qz, sigma_z, xi, h, cutoff)
        # incident plane: q_r = q_x (|qy| ~ 0 -> in-plane component along the beam)
        res['S_diff'] = w * Sd
        I = I + res['S_diff']
        if spec_widths is not None:
            Ss = S_specular(o['qx'], qz, alpha_i, wavelength, delta, beta, sigma_z, *spec_widths)
            res['S_spec'] = w * Ss
            I = I + res['S_spec']
    res['I'] = I
    return res
