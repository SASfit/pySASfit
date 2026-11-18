"""Python ports of the isotropic sphere-like SASfit plugins: sphericalshells, sphere_with_3_shells,
boucherprofile, kratky_sphere.

All amplitudes use the stable Φ(x) = 3(sin x - x cos x)/x³ of :mod:`.common` (series for x < 0.05).

C deviations, fixed in the SASfit sources on 2026-11-18 (documented in docs/studio_manual/python_ports.md):

* sphere amplitudes lost digits by cancellation for 1e-4 < q R < 0.01 (series now up to q R = 0.05);
* multilamellar_vesicle ``_v`` returned the outer radius, not a volume;
* kratky_sphere divided by Δq⊥: Δq⊥ = 0 gave 0/0, now the limit P(q)(1 + P16(q)).
"""

from __future__ import annotations

import numpy as np
from scipy import special

from .common import hyp0f1_neg, make_spec, phi_sphere, sphere_amplitude
from .quadrature import gauss_legendre

__all__ = ["SPECS"]

_SS = "plugins/sphericalshells/sasfit_"


def _r(q, p):
    return np.asarray(q, float) * max(abs(p[0]), 1e-300)


# ------------------------------------------------------------------------------------------------ sphericalshells
def sphere_f(q, p):
    return sphere_amplitude(q, p[0], p[3])


def sphere_i(q, p):
    """p = [R, dummy, dummy, eta]: (η V Φ(qR))²."""
    if p[0] < 0:
        raise ValueError("R >= 0 required")
    return sphere_f(q, p) ** 2


def sphere_v(x, p, dist):
    r = x if dist == 0 else p[0]
    return 4.0 / 3.0 * np.pi * np.asarray(r, float) ** 3 + 0.0 * np.asarray(x, float)


def shell_i_f(q, p):
    r1, r2, mu, eta = p[0], p[1], p[2], p[3]
    return sphere_amplitude(q, r1, eta) - sphere_amplitude(q, r2, eta * (1.0 - mu))


def shell_i(q, p):
    """p = [R1, R2, mu, eta]."""
    if p[0] < 0 or p[1] < 0:
        raise ValueError("R1, R2 >= 0 required")
    return shell_i_f(q, p) ** 2


def shell_i_v(x, p, dist):
    r = x if dist == 0 else p[0]
    return 4.0 / 3.0 * np.pi * np.asarray(r, float) ** 3 + 0.0 * np.asarray(x, float)


def shell_ii_f(q, p):
    r, nu, mu, eta = p[0], p[1], p[2], p[3]
    return sphere_amplitude(q, r, eta) - sphere_amplitude(q, nu * r, eta * (1.0 - mu))


def shell_ii(q, p):
    """p = [R, nu, mu, eta], 0 <= nu < 1."""
    if p[0] < 0 or not (0 <= p[1] < 1):
        raise ValueError("R >= 0 and 0 <= nu < 1 required")
    return shell_ii_f(q, p) ** 2


def shell_iii_f(q, p):
    r, dr, eta1, eta2 = p[0], p[1], p[2], p[3]
    return sphere_amplitude(q, r + dr, eta2) - sphere_amplitude(q, r, eta2 - eta1)


def shell_iii(q, p):
    """p = [R, dR, eta1, eta2]."""
    if p[0] < 0 or p[1] < 0:
        raise ValueError("R, dR >= 0 required")
    return shell_iii_f(q, p) ** 2


def shell_iii_v(x, p, dist):
    x = np.asarray(x, float)
    r = x + p[1] if dist == 0 else (p[0] + x if dist == 1 else p[0] + p[1] + 0.0 * x)
    return 4.0 / 3.0 * np.pi * r**3


def hollow_f(q, p):
    return shell_iii_f(q, [p[0], p[1], 0.0, p[3]])


def hollow(q, p):
    """p = [R, dR, dummy, eta]: shell of thickness dR around an empty core of radius R."""
    if p[0] < 0 or p[1] < 0:
        raise ValueError("R, dR >= 0 required")
    return hollow_f(q, p) ** 2


def _mlv_sums(q, p):
    rc, tsh, tsol, esh, esol, n = p[0], p[1], p[2], p[3], p[4], p[5]
    nl = int(n)
    w = n - nl
    d = esh - esol
    s = np.zeros(np.shape(q))
    for i in range(nl):
        s = (
            s
            + sphere_amplitude(q, rc + i * tsh + i * tsol, -d)
            + sphere_amplitude(q, rc + (i + 1) * tsh + i * tsol, d)
        )
    extra = sphere_amplitude(q, rc + nl * tsh + nl * tsol, -d) + sphere_amplitude(
        q, rc + (nl + 1) * tsh + nl * tsol, d
    )
    return s, extra, w


def mlv(q, p):
    """p = [R_c, t_sh, t_sol, eta_shell, eta_sol, n]: n shells (fractional n mixes n and n+1 shells)."""
    if min(p[0], p[1], p[2], p[3], p[4]) < 0 or p[5] < 1:
        raise ValueError(
            "R_c, t_sh, t_sol, eta_shell, eta_sol >= 0 and n >= 1 required"
        )
    s, extra, w = _mlv_sums(q, p)
    return s * s * (1.0 - w) + (s + w * extra) ** 2 * w


def mlv_f(q, p):
    s, extra, w = _mlv_sums(q, p)
    return s + w * extra


def mlv_v(x, p, dist):
    """Volume of the outer sphere (C: returns its radius)."""
    x = np.asarray(x, float)
    rc, tsh, tsol, n = p[0], p[1], p[2], p[5]
    r = {
        0: x + (n + 1) * tsh + n * tsol,
        1: rc + (n + 1) * x + n * tsol,
        2: rc + (n + 1) * tsh + n * x,
    }.get(dist, rc + (n + 1) * tsh + n * tsol + 0.0 * x)
    return 4.0 / 3.0 * np.pi * r**3


# ------------------------------------------------------------------------------------------------ 3 shells
def _sf(q, r):
    return 4.0 / 3.0 * np.pi * r**3 * phi_sphere(np.asarray(q, float) * r)


def three_shells_f(q, p):
    rc, t1, t2, t3, ec, e1, e2, e3, es = p[:9]
    r1, r2, r3 = rc + t1, rc + t1 + t2, rc + t1 + t2 + t3
    return ((ec - es) * _sf(q, rc) + (e1 - es) * (_sf(q, r1) - _sf(q, rc)) + (e2 - es) * (_sf(q, r2) - _sf(q, r1))
            + (e3 - es) * (_sf(q, r3) - _sf(q, r2)))  # fmt: skip


def three_shells(q, p):
    """p = [R_core, t_sh1, t_sh2, t_sh3, eta_core, eta_sh1, eta_sh2, eta_sh3, eta_sol]."""
    if min(p[:4]) < 0:
        raise ValueError("radii and thicknesses >= 0 required")
    return three_shells_f(q, p) ** 2


def three_shells_v(x, p, dist):
    return 4.0 / 3.0 * np.pi * (p[0] + p[1] + p[2] + p[3]) ** 3 + 0.0 * np.asarray(
        x, float
    )


# ------------------------------------------------------------------------------------------------ Boucher
def _boucher_beta(p):
    r, a, de = p[0], p[1], p[2]
    return (
        de
        * (r * np.sqrt(np.pi)) ** 3
        * np.exp(special.gammaln(a / 2.0 - 1.0) - special.gammaln(a / 2.0 + 0.5))
    )


def boucher_sphere_f(q, p):
    return _boucher_beta(p) * hyp0f1_neg(
        0.5 * (p[1] + 1.0), np.asarray(q, float) * p[0]
    )


def boucher_sphere(q, p):
    """p = [R, alpha, Delta_eta]: radial profile Δη (1 - r²/R²)^(α/2-2), α > 2."""
    if p[1] <= 2 or p[0] < 0:
        raise ValueError("alpha > 2 and R >= 0 required")
    return boucher_sphere_f(q, p) ** 2


def boucher_sphere2_f(q, p):
    return (
        p[2]
        * 4.0
        / 3.0
        * np.pi
        * p[0] ** 3
        * hyp0f1_neg(0.5 * (p[1] + 1.0), np.asarray(q, float) * p[0])
    )


def boucher_sphere2(q, p):
    """p = [R, alpha, Delta_eta]: as Boucher Sphere, forward amplitude Δη 4π/3 R³."""
    if p[0] < 0:
        raise ValueError("R >= 0 required")
    return boucher_sphere2_f(q, p) ** 2


def boucher_v(x, p, dist):
    return 4.0 / 3.0 * np.pi * p[0] ** 3 + 0.0 * np.asarray(x, float)


def _profile(r, p):
    r = np.asarray(r, float)
    inside = np.abs(r) <= abs(p[0])
    with np.errstate(divide="ignore", invalid="ignore"):
        v = p[2] * np.power(1.0 - (r / p[0]) ** 2, p[1] / 2.0 - 2.0)
    return np.where(inside, v, 0.0)


def boucher_profiles(r, p):
    """p = [R, alpha, Delta_eta]; abscissa r: Δη (1 - r²/R²)^(α/2-2) inside R, 0 outside."""
    return _profile(r, p)


def boucher_profiles2(r, p):
    """p = [R, alpha, Delta_eta]; profile normalised to the forward amplitude of Boucher Sphere2."""
    if p[1] <= 2:
        raise ValueError("alpha > 2 required")
    norm = (
        4.0
        / 3.0
        * np.pi
        / (
            np.pi**1.5
            * np.exp(
                special.gammaln(p[1] / 2.0 - 1.0) - special.gammaln(p[1] / 2.0 + 0.5)
            )
        )
    )
    return _profile(r, p) * norm


# ------------------------------------------------------------------------------------------------ Kratky sphere
def _kratky_kernel(q, p):
    r, eta, r0, xi, d, scale = p[0], p[2], p[3], p[4], p[5], p[6]
    P = (eta * 4.0 * np.pi / 3.0 * r**3 * phi_sphere(q * r)) ** 2
    if xi == 0 or r0 == 0:
        return P
    with np.errstate(divide="ignore", invalid="ignore"):
        p16 = (d * special.gamma(d - 1.0) / np.power(q * r0, d) / np.power(1.0 + 1.0 / (q * q * xi * xi), (d - 1.0) / 2.0)
               * np.sin((d - 1.0) * np.arctan(q * xi)) * scale)  # fmt: skip
    return P * (1.0 + p16)


def kratky_sphere(q, p):
    """p = [R, Delta_q_perp, eta, R0, xi, D, scale]: sphere (times the fractal structure factor P16) averaged
    over q⊥ ∈ [0, Δq⊥] (slit smearing): 1/Δq⊥ ∫ I(√(q² + q⊥²)) dq⊥."""
    q = np.asarray(q, float)
    r, dq, d = p[0], p[1], p[5]
    if r < 0 or dq < 0 or p[2] < 0 or d <= 1:
        raise ValueError("R, Delta_q_perp, eta >= 0 and D > 1 required")
    if dq == 0:
        return _kratky_kernel(q, p)
    n = int(min(2000, 24 + 3 * dq * max(r, abs(p[4]), abs(p[3]))))
    t, weights = gauss_legendre(n, 0.0, dq)
    qq = np.sqrt(q[..., None] ** 2 + t**2)
    return np.sum(weights * _kratky_kernel(qq, p), axis=-1) / dq


# ------------------------------------------------------------------------------------------------ registry
_RAD = (1.0, 100.0, "log")
_ETA = (0.1, 3.0, "lin")
SPECS = [
    make_spec("ff_sphere", sphere_i, ["R", "dummy", "dummy", "eta"], {0: _RAD, 3: _ETA}, amplitude=sphere_f,
              volume=sphere_v, source=_SS + "ff_sphere.c", reduced=_r, rtol=1e-8),
    make_spec("ff_spherical_shell_i", shell_i, ["R1", "R2", "mu", "eta"],
              {0: (20.0, 100.0, "log"), 1: (1.0, 19.0, "log"), 2: (-1.0, 2.0, "lin"), 3: _ETA},
              amplitude=shell_i_f, volume=shell_i_v, source=_SS + "ff_spherical_shell_i.c", reduced=_r, rtol=1e-8),
    make_spec("ff_spherical_shell_ii", shell_ii, ["R", "nu", "mu", "eta"],
              {0: _RAD, 1: (0.05, 0.95, "lin"), 2: (-1.0, 2.0, "lin"), 3: _ETA},
              amplitude=shell_ii_f, volume=shell_i_v, source=_SS + "ff_spherical_shell_ii.c", reduced=_r, rtol=1e-8),
    make_spec("ff_spherical_shell_iii", shell_iii, ["R", "dR", "eta1", "eta2"],
              {0: _RAD, 1: (1.0, 30.0, "log"), 2: (-1.0, 2.0, "lin"), 3: _ETA},
              amplitude=shell_iii_f, volume=shell_iii_v, source=_SS + "ff_spherical_shell_iii.c", reduced=_r, rtol=1e-8),
    make_spec("ff_hollow_sphere", hollow, ["R", "dR", "dummy", "eta"], {0: _RAD, 1: (1.0, 30.0, "log"), 3: _ETA},
              amplitude=hollow_f, volume=shell_iii_v, source=_SS + "ff_hollow_sphere.c", reduced=_r, rtol=1e-8),
    make_spec("ff_multilamellar_vesicle", mlv, ["R_c", "t_sh", "t_sol", "eta_shell", "eta_sol", "n"],
              {0: (10.0, 100.0, "log"), 1: (2.0, 8.0, "lin"), 2: (2.0, 20.0, "lin"), 3: (1.0, 3.0, "lin"),
               4: (0.0, 0.9, "lin"), 5: (1.0, 6.0, "lin")},
              amplitude=mlv_f, volume=mlv_v, source=_SS + "ff_multilamellar_vesicle.c", reduced=_r,
              notes="C _v returns the outer radius; the port returns the volume"),
    make_spec("ff_sphere_with_3_shells", three_shells,
              ["R_core", "t_sh1", "t_sh2", "t_sh3", "eta_core", "eta_sh1", "eta_sh2", "eta_sh3", "eta_sol"],
              {0: _RAD, 1: (1.0, 10.0, "lin"), 2: (1.0, 10.0, "lin"), 3: (1.0, 10.0, "lin"), 4: _ETA, 5: _ETA,
               6: _ETA, 7: _ETA, 8: (0.0, 1.0, "lin")},
              amplitude=three_shells_f, volume=three_shells_v,
              source="plugins/sphere_with_3_shells/sasfit_ff_sphere_with_3_shells.c", reduced=_r, rtol=1e-8),
    make_spec("ff_boucher_sphere", boucher_sphere, ["R", "alpha", "Delta_eta"],
              {0: _RAD, 1: (2.2, 12.0, "lin"), 2: _ETA}, amplitude=boucher_sphere_f, volume=boucher_v,
              source="plugins/boucherprofile/sasfit_ff_boucher_sphere.c", reduced=_r, rtol=1e-8),
    make_spec("ff_boucher_sphere2", boucher_sphere2, ["R", "alpha", "Delta_eta"],
              {0: _RAD, 1: (0.5, 12.0, "lin"), 2: _ETA}, amplitude=boucher_sphere2_f, volume=boucher_v,
              source="plugins/boucherprofile/sasfit_ff_boucher_sphere2.c", reduced=_r, rtol=1e-8),
    make_spec("ff_boucher_profiles", boucher_profiles, ["R", "alpha", "Delta_eta"],
              {0: (5.0, 20.0, "lin"), 1: (2.0, 12.0, "lin"), 2: _ETA},
              source="plugins/boucherprofile/sasfit_ff_boucher_profiles.c", reduced=_r, reduced_label="r/R"),
    make_spec("ff_boucher_profiles2", boucher_profiles2, ["R", "alpha", "Delta_eta"],
              {0: (5.0, 20.0, "lin"), 1: (2.2, 12.0, "lin"), 2: _ETA},
              source="plugins/boucherprofile/sasfit_ff_boucher_profiles2.c", reduced=_r, reduced_label="r/R"),
    make_spec("ff_kratkysphere", kratky_sphere, ["R", "+-Delta_q_perp", "eta", "R0", "xi", "D", "scale"],
              {0: (2.0, 30.0, "log"), 1: (0.01, 0.5, "log"), 2: _ETA, 3: (1.0, 10.0, "log"), 4: (10.0, 200.0, "log"),
               5: (1.5, 3.0, "lin"), 6: (0.0, 1.0, "lin")},
              source="plugins/kratky_sphere/sasfit_ff_kratkysphere.c", reduced=_r, c_integrates=True, rtol=1e-6,
              notes="C: 0/0 for Delta_q_perp = 0; the port returns the unsmeared limit"),
]  # fmt: skip
