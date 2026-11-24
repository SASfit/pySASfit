"""Python ports of the spherical micelles of the plugin spherical_ellipsoidal_micelles whose chains are worm-like
(Pedersen and Schurtenberger, method 3: :func:`.worms.worm_ps3`) or generalized Gaussian coils:

* sphere+R^-a (``ff_sphere_r__a``, ``_rc``, ``_nagg``): spherical core with a corona of density ∝ r^-α between
  R_core and R_core + t, self term of the chains = worm (exvol 0 / 1) or generalized Gaussian coil (exvol >= 2, then
  "L" is Rg and "b" is ν);
* sphere+chains(SAW) (``ff_sphere_chains_saw_``, ``_rc``, ``_nagg``): Pedersen's model 39 with self-avoiding
  worm-like chains (Rg² = α²(L/b) Rg²_ideal) and a Gaussian-smeared corona.

Corona amplitude by Gauss-Legendre on panels of at most 1.5 rad phase (Gauss-Jacobi with the weight r^(2-α) on the
first panel when R_core = 0), exact to rounding.

C fixes of 2026-11-23 followed here: sphere+chains(SAW) with R_core/n_agg used the core volume V/(1 - xsolv_core)
instead of V (1 - xsolv_core) (the _rc and _nagg variants and all RW models use the latter); sphere+R^-a with
n_agg = 0 returned the normalised Φ² without volume and contrast; fRg2 of the SAW radius of gyration cancelled for
L << b. Not ported: sphere_2nd+R^-a_Rc and sphere_smooth_interface+R^-a_Nagg (the C functions are stubs returning 0).
"""

from __future__ import annotations

import math

import numpy as np
from scipy import special

from .common import make_spec, phi_sphere
from .polymers import generalized_gaussian_coil
from .quadrature import gauss_legendre
from .worms import falpha2, frg2, worm_ps3

__all__ = ["SPECS"]


def _ptwentyone(x):
    """2 (x + e^-x - 1)/x² (series below 0.5, as the fixed C)."""
    x = np.asarray(x, float)
    small = np.abs(x) < 0.5
    t = np.ones_like(x)
    s = np.zeros_like(x)
    for k in range(18):
        s = s + t
        t = t * (-x / (k + 3.0))
    xs = np.where(small, 1.0, x)
    return np.where(small, s, 2.0 * (xs + np.expm1(-xs)) / xs**2)


def _fone(q, r):
    """3 (sin x - x cos x)/x³ with the C series below x = 0.05."""
    x = np.asarray(q, float) * r
    small = np.abs(x) < 0.05
    ser = 1 - x**2 / 10.0 + x**4 / 280.0 - x**6 / 15120.0 + x**8 / 1330560.0
    xs = np.where(small, 1.0, x)
    return np.where(small, ser, 3.0 * (np.sin(xs) - xs * np.cos(xs)) / xs**3)


def _sg(q, rc, s):
    """Gaussian-smeared spherical shell amplitude of Pedersen's model (C function Sg)."""
    q = np.asarray(q, float)
    if rc == 0.0:
        return np.where(q == 0, 1.0, np.exp(-((q * s) ** 2) / 2.0))
    qs = np.where(q == 0, 1.0, q)
    if s == 0.0:
        return np.where(q == 0, 1.0, np.sin(qs * rc) / (qs * rc))
    r2p = math.sqrt(2.0 * math.pi)
    val = (1.0 / (qs * rc * (4.0 * rc * s + r2p * (rc * rc + s * s)))
           * (2.0 * rc * s * np.sin(qs * rc) + r2p * np.exp(-((qs * s) ** 2) / 2.0)
              * (qs * rc * s * s * np.cos(qs * rc) + rc * rc * np.sin(qs * rc))
              + 2.0 * math.sqrt(2.0) * special.dawsn(qs * s / math.sqrt(2.0))
              * (rc * rc * np.cos(qs * rc) - qs * rc * s * s * np.sin(qs * rc))))  # fmt: skip
    return np.where(q == 0, 1.0, val)


def _ac(q, rc, r, s):
    v0 = 2.0 * math.pi * (4.0 * rc * s * s + math.sqrt(2.0 * math.pi) * (rc * rc + s * s) * s)
    vr = 4.0 * math.pi / 3.0
    return (v0 * _sg(q, rc, s) + vr * rc**3 * _fone(q, rc) - vr * r**3 * _fone(q, r)) / (v0 + vr * rc**3 - vr * r**3)


def _p39(q, nc, rg, r, rc, s, nu, rhos, rhoc, b, L):
    """Model 39 of Pedersen (absolute units, without the 1/M² of the paper)."""
    pch = worm_ps3(q, b, L, 1) / (1.0 + nu * _ptwentyone((np.asarray(q, float) * rg) ** 2))
    p0 = 1.0 / (1.0 + nu)
    a = _ac(q, rc, r, s)
    f1 = _fone(q, r)
    return rhos**2 * f1**2 + nc * rhoc**2 * pch + nc * (nc - p0) * rhoc**2 * a * a + 2.0 * nc * rhos * rhoc * f1 * a


def _saw(q, p, kind):
    q = np.asarray(q, float)
    a, bpar, vb, ec, eb, es, x, L, b = p[:9]
    if L <= 0 or b < 0:
        raise ValueError("L > 0 and lb >= 0 required")
    rg = math.sqrt(falpha2(L, b) * frg2(L, b))
    if kind == "":
        rcore, n_agg = a, bpar
        if rcore < 0:
            raise ValueError("R_core >= 0 required")
        if x == 1.0:
            raise ValueError("xsolv_core == 1")
        v = 4.0 / 3.0 * math.pi * rcore**3
        nc = n_agg * 4.0 * math.pi * rcore**2
        rhos = v * (1.0 - x) * (ec - es)
    elif kind == "rc":
        rcore, vcore = a, bpar
        if rcore < 0 or vcore < 0:
            raise ValueError("R_core >= 0 and V_core >= 0 required")
        v = 4.0 / 3.0 * math.pi * rcore**3
        nc = v * (1.0 - x) / vcore
        rhos = nc * vcore * (ec - es)
    else:
        nagg, vcore = a, bpar
        if nagg < 0:
            raise ValueError("Nagg >= 0 required")
        if x >= 1.0:
            raise ValueError("xsolv_core < 1 required")
        v = nagg * vcore / (1.0 - x)
        rcore = (v * 3.0 / 4.0 / math.pi) ** (1.0 / 3.0)
        nc = nagg
        rhos = nc * vcore * (ec - es)
    if nc == 0.0:
        return np.zeros(q.shape)
    rhoc = vb * (eb - es)
    sigma = nc * rg**2 / (4.0 * (rcore + rg) ** 2)
    nu = 1.4 * sigma**1.04
    return _p39(q, nc, rg, rcore, rcore, rg, nu, rhos, rhoc, b, L)


def sphere_chains_saw(q, p):
    """p = [R_core, n_agg, V_brush, eta_core, eta_brush, eta_solv, xsolv_core, L, lb]: n_agg chains per core surface."""
    return _saw(q, p, "")


def sphere_chains_saw_rc(q, p):
    """p = [R_core, V_core, V_brush, eta_core, eta_brush, eta_solv, xsolv_core, L, lb]: Nagg = V (1 - x)/V_core."""
    return _saw(q, p, "rc")


def sphere_chains_saw_nagg(q, p):
    """p = [Nagg, V_core, V_brush, eta_core, eta_brush, eta_solv, xsolv_core, L, lb]."""
    return _saw(q, p, "nagg")


# ------------------------------------------------------------------------------------------ r^-alpha corona
_GX, _GW = np.polynomial.legendre.leggauss(16)


def corona_amplitude(q, rc, t, alpha):
    """∫_Rc^{Rc+t} r^-α r² sinc(qr) dr / ∫_Rc^{Rc+t} r^-α r² dr (C: sasfit_integrate of the profile, analytic
    denominator). Gauss-Legendre on panels of at most 1.5 rad phase; for R_core = 0 the first panel by
    Gauss-Jacobi with the weight r^(2-α)."""
    q = np.atleast_1d(np.asarray(q, float))
    rm = rc + t
    beta = 2.0 - alpha
    nsub = max(1, math.ceil(float(np.max(q, initial=0.0)) * t / 1.5))
    edges = np.linspace(rc, rm, nsub + 1)
    rs, ws = [], []
    for k, (lo, hi) in enumerate(zip(edges[:-1], edges[1:])):
        if lo == 0.0:
            x, w = special.roots_jacobi(24, 0.0, beta)
            h = 0.5 * hi
            rs.append(h * (1.0 + x))
            ws.append(w * h ** (beta + 1.0))
        else:
            c, h = 0.5 * (hi + lo), 0.5 * (hi - lo)
            r = c + h * _GX
            rs.append(r)
            ws.append(h * _GW * r**beta)
    r, w = np.concatenate(rs), np.concatenate(ws)
    if alpha == 3.0:
        den = math.log(rm / rc)
    else:
        den = (rm ** (beta + 1.0) - rc ** (beta + 1.0)) / (beta + 1.0)
    out = np.empty(q.shape)
    for i0 in range(0, q.size, 256):
        qq = q[i0 : i0 + 256]
        out[i0 : i0 + 256] = np.sinc(qq[:, None] * r[None, :] / np.pi) @ w
    return out / den


def _rma(q, p, kind):
    q = np.asarray(q, float)
    a, bpar, vsh, rho_c, rho_sh, rho_s, alpha, t, L, b, x, exvol = p[:12]
    if alpha < 0 or t < 0 or L <= 0 or b < 0:
        raise ValueError("alpha >= 0, t >= 0, L > 0, b >= 0 required")
    if kind == "":
        rc, n_agg = a, bpar
        if rc < 0 or n_agg < 0 or vsh < 0:
            raise ValueError("R_core, n_agg, V_brush >= 0 required")
        if rc == 0.0:
            return np.zeros(q.shape)
        V = 4.0 / 3.0 * math.pi * rc**3
        nagg = n_agg * 4.0 * math.pi * rc * rc
        fc = phi_sphere(q * rc)
        if nagg == 0.0:
            return (V * (1.0 - x) * (rho_c - rho_s)) ** 2 * fc * fc
        vc = V / nagg * (1.0 - x)
    elif kind == "rc":
        rc, vc = a, bpar
        if rc < 0 or vc <= 0:
            raise ValueError("R_core >= 0 and V_core > 0 required")
        if rc == 0.0:
            return np.zeros(q.shape)
        V = 4.0 / 3.0 * math.pi * rc**3 * (1.0 - x)
        nagg = V / vc
        fc = phi_sphere(q * rc)
    else:
        nagg = a
        if nagg < 0 or bpar < 0 or vsh < 0:
            raise ValueError("Nagg, V_core, V_brush >= 0 required")
        if x >= 1.0:
            raise ValueError("xsolv_core < 1 required")
        vc = abs(bpar)
        V = nagg * vc / (1.0 - x)
        rc = (V * 3.0 / 4.0 / math.pi) ** (1.0 / 3.0)
        fc = phi_sphere(q * rc)
    b_c = vc * (rho_c - rho_s)
    b_sh = abs(vsh) * (rho_sh - rho_s)
    if t == 0.0 or b_sh == 0.0:
        fsh = np.zeros(q.shape)
    else:
        if alpha >= 3.0 and rc == 0.0:
            raise ValueError("alpha >= 3 needs R_core > 0")
        fsh = corona_amplitude(q, rc, t, alpha)
    ex = int(round(abs(exvol)))
    if ex == 0:
        plocal = worm_ps3(q, b, L, 0)
    elif ex == 1:
        plocal = worm_ps3(q, b, L, 1)
    else:
        plocal = generalized_gaussian_coil(q, L, b)
    if vsh < 0:
        return nagg * b_sh * b_sh * plocal
    step = 0.0 if nagg < 1 else 1.0
    return (nagg * nagg * b_c * b_c * fc * fc + nagg * (nagg - 1.0) * b_sh * b_sh * fsh * fsh * step
            + 2.0 * nagg * nagg * b_c * b_sh * fc * fsh + nagg * b_sh * b_sh * plocal)  # fmt: skip


def sphere_r_a(q, p):
    """p = [R_core, n_agg, V_brush, eta_core, eta_brush, eta_solv, alpha, t, L/Rg, b/nu, xsolv_core, exvol]."""
    return _rma(q, p, "")


def sphere_r_a_rc(q, p):
    """p = [R_core, V_core, V_brush, eta_core, eta_brush, eta_solv, alpha, t, L/Rg, b/nu, xsolv_core, exvol]
    (V_brush < 0: only the self term of the chains)."""
    return _rma(q, p, "rc")


def sphere_r_a_nagg(q, p):
    """p = [Nagg, V_core, V_brush, eta_core, eta_brush, eta_solv, alpha, t, L (RG), b (nu), xsolv_core, exvol]."""
    return _rma(q, p, "nagg")


def _qr(q, p):
    return np.asarray(q, float) * abs(p[0])


_SE = "plugins/spherical_ellipsoidal_micelles/sasfit_ff_"
_SAW = ["V_brush", "eta_core", "eta_brush", "eta_solv", "xsolv_core", "L", "lb"]
_SAW_S = {2: (100.0, 2000.0, "log"), 3: (0.5, 6.0, "lin"), 4: (0.5, 2.0, "lin"), 5: (0.0, 1.0, "lin"),
          6: (0.0, 0.5, "lin"), 7: (5.0, 200.0, "log"), 8: (0.5, 20.0, "log")}  # fmt: skip
_RMA = ["V_brush", "eta_core", "eta_brush", "eta_solv", "alpha", "t", "L/Rg", "b/nu", "xsolv_core", "exvol"]
_RMA_S = {2: (100.0, 2000.0, "log"), 3: (0.5, 6.0, "lin"), 4: (0.5, 2.0, "lin"), 5: (0.0, 1.0, "lin"),
          6: (0.0, 2.9, "lin"), 7: (1.0, 20.0, "log"), 8: (5.0, 200.0, "log"), 9: (0.5, 20.0, "log"),
          10: (0.0, 0.5, "lin"), 11: (0.0, 1.0, "lin")}  # fmt: skip

SPECS = [
    make_spec("ff_sphere_chains_saw_", sphere_chains_saw, ["R_core", "n_agg", *_SAW],
              {**_SAW_S, 0: (2.0, 30.0, "log"), 1: (0.001, 0.05, "log")}, source=_SE + "sphere_chains_saw_.c",
              reduced=_qr, reduced_label="q R_core", rtol=1e-9, group="ff_spherical_micelles"),
    make_spec("ff_sphere_chains_saw__rc", sphere_chains_saw_rc, ["R_core", "V_core", *_SAW],
              {**_SAW_S, 0: (2.0, 30.0, "log"), 1: (0.5, 20.0, "log")}, source=_SE + "sphere_chains_saw__rc.c",
              reduced=_qr, reduced_label="q R_core", rtol=1e-9, group="ff_spherical_micelles"),
    make_spec("ff_sphere_chains_saw__nagg", sphere_chains_saw_nagg, ["Nagg", "V_core", *_SAW],
              {**_SAW_S, 0: (5.0, 200.0, "log"), 1: (0.5, 20.0, "log")}, source=_SE + "sphere_chains_saw__nagg.c",
              reduced=None, rtol=1e-9, group="ff_spherical_micelles"),
    make_spec("ff_sphere_r__a", sphere_r_a, ["R_core", "n_agg", *_RMA],
              {**_RMA_S, 0: (2.0, 30.0, "log"), 1: (0.001, 0.05, "log")}, source=_SE + "sphere_r__a.c",
              reduced=_qr, reduced_label="q R_core", rtol=1e-9, c_integrates=True, group="ff_spherical_micelles"),
    make_spec("ff_sphere_r__a_rc", sphere_r_a_rc, ["R_core", "V_core", *_RMA],
              {**_RMA_S, 0: (2.0, 30.0, "log"), 1: (0.5, 20.0, "log")}, source=_SE + "sphere_r__a_rc.c",
              reduced=_qr, reduced_label="q R_core", rtol=1e-9, c_integrates=True, group="ff_spherical_micelles"),
    make_spec("ff_sphere_r__a_nagg", sphere_r_a_nagg, ["Nagg", "V_core", "V_brush", "eta_core", "eta_brush",
              "eta_solv", "alpha", "t", "L (RG)", "b (nu)", "xsolv_core", "exvol"],
              {**_RMA_S, 0: (5.0, 200.0, "log"), 1: (0.5, 20.0, "log")}, source=_SE + "sphere_r__a_nagg.c",
              reduced=None, rtol=1e-9, c_integrates=True, group="ff_spherical_micelles"),
]  # fmt: skip
