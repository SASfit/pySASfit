"""Python ports of the SASfit plugin spherical_ellipsoidal_micelles (Gaussian-chain brushes, Pedersen).

Ported: sphere+chains(RW) and ellipsoid+chains(RW), each in the variants R_core/n_agg, R_core/V_core (_rc) and
Nagg/V_core (_nagg). The r^-alpha profile and SAW-chain variants call the worm form factor of Pedersen and
Schurtenberger (sasfit_sq_p__q___worm_ps3_) and follow with its port.

C deviations fixed on 2026-11-18: the ellipsoid surface used for n_agg (oblate: factor epsilon instead of epsilon²,
prolate: ae/tan(ae) instead of ae/sin(ae)); ell_chains_rw__nagg computed R_core with 4/3 = 1 (integer division);
sphere_chains_rw printed every call (sasfit_out); series of the sphere amplitude and of the Debye function only
below 1e-4 (cancellation).
"""

from __future__ import annotations

import numpy as np

from .common import debye, debye_amplitude, make_spec, phi_sphere
from .quadrature import gauss_legendre

__all__ = ["SPECS"]


def _sinc(x):
    return np.sinc(np.asarray(x, float) / np.pi)


def _sphere_gauss_chains(q, r, rg, d, nagg, rc, rs):
    """N² rs² Φ² + N rc² P_Debye + N(N-1) rc² (w j0)² + 2 N² rs rc Φ w j0, j0 = sin(q(R+d Rg))/(q(R+d Rg))."""
    phi = phi_sphere(q * r)
    u = (q * rg) ** 2
    w = debye_amplitude(u)
    sx = _sinc(q * (r + d * rg))
    step = 0.0 if nagg < 1 else 1.0
    return (nagg * nagg * rs * rs * phi * phi + nagg * rc * rc * debye(u)
            + nagg * (nagg - 1.0) * step * rc * rc * (w * sx) ** 2 + 2.0 * nagg * nagg * rs * rc * phi * w * sx)  # fmt: skip


def _check_brush(vbrush, xsolv, rg, d):
    if vbrush < 0 or rg < 0 or d < 0:
        raise ValueError("V_brush, Rg, d >= 0 required")
    if xsolv < 0 or xsolv >= 1:
        raise ValueError("0 <= xsolv_core < 1 required")


def sphere_chains_rw(q, p):
    """p = [R_core, n_agg, V_brush, eta_core, eta_brush, eta_solv, xsolv_core, Rg, d, dummy]:
    sphere with Gaussian chains, n_agg chains per surface area (Nagg = 4π R_core² n_agg)."""
    q = np.asarray(q, float)
    rcore, n_agg, vb, ec, eb, es, x, rg, d = p[:9]
    if rcore < 0 or n_agg <= 0:
        raise ValueError("R_core >= 0 and n_agg > 0 required")
    _check_brush(vb, x, rg, d)
    if rcore == 0:
        return np.zeros(q.shape)
    v = 4.0 / 3.0 * np.pi * rcore**3
    nagg = n_agg * 4.0 * np.pi * rcore**2
    rs = v / nagg * (ec - es) * (1.0 - x)
    return _sphere_gauss_chains(q, rcore, rg, d, nagg, vb * (eb - es), rs)


def sphere_chains_rw_rc(q, p):
    """p = [R_core, V_core, V_brush, eta_core, eta_brush, eta_solv, xsolv_core, Rg, d, dummy]:
    Nagg = 4/3 π R_core³ (1 - xsolv_core)/V_core."""
    q = np.asarray(q, float)
    rcore, vcore, vb, ec, eb, es, x, rg, d = p[:9]
    if rcore < 0 or vcore <= 0:
        raise ValueError("R_core >= 0 and V_core > 0 required")
    if d <= 0:
        raise ValueError("d > 0 required")
    _check_brush(vb, x, rg, d)
    nagg = 4.0 / 3.0 * np.pi * rcore**3 * (1.0 - x) / vcore
    return _sphere_gauss_chains(
        q, rcore, rg, d, nagg, vb * (eb - es), vcore * (ec - es)
    )


def sphere_chains_rw_nagg(q, p):
    """p = [Nagg, V_core, V_brush, eta_core, eta_brush, eta_solv, xsolv_core, Rg, d, dummy]:
    R_core = (3 Nagg V_core/(4π (1 - xsolv_core)))^(1/3)."""
    q = np.asarray(q, float)
    nagg, vcore, vb, ec, eb, es, x, rg, d, dummy = p[:10]
    if nagg < 0 or vcore < 0 or dummy < 0:
        raise ValueError("Nagg, V_core, dummy >= 0 required")
    _check_brush(vb, x, rg, d)
    r = (abs(nagg * vcore / (1.0 - x)) * 3.0 / 4.0 / np.pi) ** (1.0 / 3.0)
    return _sphere_gauss_chains(q, r, rg, d, nagg, vb * (eb - es), vcore * (ec - es))


def spheroid_surface(r, eps):
    """Surface of a spheroid with equatorial radius r and polar semi-axis eps r."""
    if eps < 1.0:
        e = np.sqrt(1.0 - eps * eps)
        return 2.0 * np.pi * r * r * (1.0 + eps * eps * np.arctanh(e) / e)
    if eps > 1.0:
        ae = np.arccos(1.0 / eps)
        return 2.0 * np.pi * r * r * (1.0 + eps * ae / np.sin(ae))
    return 4.0 * np.pi * r * r


def _ellip_body(q, r, eps, rg, d, nagg, r_core, r_chain):
    """Orientation average (μ = cos α) of the spheroid + Gaussian-chain brush terms."""
    q = np.asarray(q, float)
    span = np.max(q, initial=0.0) * (r * abs(eps - 1.0) + r)
    n = int(np.clip(64 + 1.5 * span, 64, 6000))
    mu, w = gauss_legendre(n, 0.0, 1.0)
    re = r * np.sqrt(1.0 - mu * mu + (eps * mu) ** 2)
    qq = q[..., None]
    f1 = phi_sphere(qq * re)
    u = (q * rg) ** 2
    wb = debye_amplitude(u)[..., None]
    j = _sinc(qq * (re + d * rg))
    fs = np.sum(w * f1 * f1, axis=-1)
    scc = np.sum(w * (wb * j) ** 2, axis=-1)
    ssc = np.sum(w * wb * f1 * j, axis=-1)
    step = 0.0 if nagg < 1 else 1.0
    return (r_core**2 * fs + nagg * r_chain**2 * debye(u) + nagg * (nagg - 1.0) * r_chain**2 * scc * step
            + 2.0 * nagg * r_core * r_chain * ssc)  # fmt: skip


def _check_ellip(vb, x, rg, d, eps):
    _check_brush(vb, x, rg, d)
    if eps <= 0:
        raise ValueError("epsilon > 0 required")


def ell_chains_rw(q, p):
    """p = [R_core, n_agg, V_brush, eta_core, eta_brush, eta_solv, xsolv_core, Rg, d, epsilon]:
    spheroid (R_core, epsilon R_core) with Gaussian chains, n_agg chains per surface area."""
    rc, n_agg, vb, ec, eb, es, x, rg, d, eps = p[:10]
    if rc < 0 or n_agg <= 0:
        raise ValueError("R_core >= 0 and n_agg > 0 required")
    _check_ellip(vb, x, rg, d, eps)
    vol = 4.0 / 3.0 * np.pi * rc**3 * eps
    nagg = n_agg * spheroid_surface(rc, eps)
    r_core = vol * (1.0 - x) * (ec - es)
    return _ellip_body(q, rc, eps, rg, d, nagg, r_core, vb * (eb - es))


def ell_chains_rw_rc(q, p):
    """p = [R_core, V_core, V_brush, eta_core, eta_brush, eta_solv, xsolv_core, Rg, d, epsilon]:
    Nagg = 4/3 π R_core³ epsilon (1 - xsolv_core)/V_core."""
    rc, vcore, vb, ec, eb, es, x, rg, d, eps = p[:10]
    if rc < 0 or vcore <= 0:
        raise ValueError("R_core >= 0 and V_core > 0 required")
    _check_ellip(vb, x, rg, d, eps)
    nagg = 4.0 / 3.0 * np.pi * rc**3 * eps * (1.0 - x) / vcore
    return _ellip_body(
        q, rc, eps, rg, d, nagg, nagg * vcore * (ec - es), vb * (eb - es)
    )


def ell_chains_rw_nagg(q, p):
    """p = [Nagg, V_core, V_brush, eta_core, eta_brush, eta_solv, xsolv_core, Rg, d, epsilon]:
    R_core = (Nagg V_core/((1 - xsolv_core) 4/3 π epsilon))^(1/3)."""
    nagg, vcore, vb, ec, eb, es, x, rg, d, eps = p[:10]
    if nagg < 0 or vcore <= 0:
        raise ValueError("Nagg >= 0 and V_core > 0 required")
    _check_ellip(vb, x, rg, d, eps)
    r = abs(nagg * vcore / (1.0 - x) / (4.0 / 3.0 * np.pi * eps)) ** (1.0 / 3.0)
    return _ellip_body(q, r, eps, rg, d, nagg, nagg * vcore * (ec - es), vb * (eb - es))


def _qr(q, p):
    return np.asarray(q, float) * abs(p[0])


_SE = "plugins/spherical_ellipsoidal_micelles/sasfit_ff_"
_RW_SAMPLE = {2: (100.0, 2000.0, "log"), 3: (0.5, 6.0, "lin"), 4: (0.5, 2.0, "lin"), 5: (0.0, 1.0, "lin"),
              6: (0.0, 0.5, "lin"), 7: (1.0, 5.0, "log"), 8: (0.5, 2.0, "lin")}  # fmt: skip
_RW = ["V_brush", "eta_core", "eta_brush", "eta_solv", "xsolv_core", "Rg", "d"]
SPECS = [
    make_spec("ff_sphere_chains_rw_", sphere_chains_rw, ["R_core", "n_agg", *_RW, "dummy"],
              {0: (3.0, 20.0, "log"), 1: (0.05, 1.0, "log"), **_RW_SAMPLE}, source=_SE + "sphere_chains_rw_.c",
              reduced=_qr, reduced_label="q R_core", rtol=1e-11),
    make_spec("ff_sphere_chains_rw__rc", sphere_chains_rw_rc, ["R_core", "V_core", *_RW, "dummy"],
              {0: (3.0, 20.0, "log"), 1: (50.0, 500.0, "log"), **_RW_SAMPLE}, source=_SE + "sphere_chains_rw__rc.c",
              reduced=_qr, reduced_label="q R_core", rtol=1e-11),
    make_spec("ff_sphere_chains_rw__nagg", sphere_chains_rw_nagg, ["Nagg", "V_core", *_RW, "dummy"],
              {0: (10.0, 300.0, "log"), 1: (50.0, 500.0, "log"), **_RW_SAMPLE}, source=_SE + "sphere_chains_rw__nagg.c",
              rtol=1e-11),
    make_spec("ff_ell_chains_rw_", ell_chains_rw, ["R_core", "n_agg", *_RW, "epsilon"],
              {0: (3.0, 20.0, "log"), 1: (0.05, 1.0, "log"), **_RW_SAMPLE, 9: (0.3, 3.0, "log")},
              source=_SE + "ell_chains_rw_.c", reduced=_qr, reduced_label="q R_core", rtol=1e-9, c_integrates=True),
    make_spec("ff_ell_chains_rw__rc", ell_chains_rw_rc, ["R_core", "V_core", *_RW, "epsilon"],
              {0: (3.0, 20.0, "log"), 1: (50.0, 500.0, "log"), **_RW_SAMPLE, 9: (0.3, 3.0, "log")},
              source=_SE + "ell_chains_rw__rc.c", reduced=_qr, reduced_label="q R_core", rtol=1e-9, c_integrates=True),
    make_spec("ff_ell_chains_rw__nagg", ell_chains_rw_nagg, ["Nagg", "V_core", *_RW, "epsilon"],
              {0: (10.0, 300.0, "log"), 1: (50.0, 500.0, "log"), **_RW_SAMPLE, 9: (0.3, 3.0, "log")},
              source=_SE + "ell_chains_rw__nagg.c", rtol=1e-9, c_integrates=True),
]  # fmt: skip
