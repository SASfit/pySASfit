"""Python ports of the SASfit plugin cylinders (batch 7, 2026-11-24): randomly oriented cylinders, discs, rods,
core-shell cylinders with circular or elliptical cross-section, and the torus.

* ``ff_cylinder``: <(π R² L η Λ(qR sin α) j0(qL cos α / 2))²> over the orientation, Λ(u) = 2 J1(u)/u;
* ``ff_disc``: infinitely thin disc, (π R² η)² 2/(qR)² [1 - J1(2qR)/(qR)];
* ``ff_rod``: infinitely thin rod, η² L² [2 Si(qL)/(qL) - j0(qL/2)²];
* ``ff_longcylinder``, ``ff_flatcylinder``: Porod's approximations for long (L >> R) and flat (L << R) cylinders
  (Struve functions in the flat one; the flat approximation is valid for qL ≲ 1 and becomes negative beyond),
  ``ff_porodcylinder`` switches between them by 2R/L (linear blend for 0.75 < 2R/L < 1.25), ``ff_porodcylinder_v2``
  is the same with L = ν R;
* ``ff_longcylshell``: Pcs of a circular core-shell cross-section times P'(q) of a thin rod of length L;
* ``ff_ellcylshell1``, ``ff_ellcylshell2``: core-shell cylinder with elliptical cross-section (semi-axes R and εR,
  shell t), shell only around the mantle (1) or also on both caps (2, length L + 2t):
  A = π R² ε L (η_c - η_sh) j0(q L cos α/2) Λ(q r_in sin α) + π (R+t)(εR+t)(L + 2t·type)(η_sh - η_sol)
      j0(q (L/2 + t·type) cos α) Λ(q r_out sin α),   r(φ) = √(((R+t) sin φ)² + ((εR+t) cos φ)²),
  P = (2/π) ∫_0^{π/2} dφ ∫_0^{π/2} A² sin α dα; ``ff_cylshell1``, ``ff_cylshell2``: ε = 1;
* ``ff_torus``: ring torus (major radius R, tube radius a, shell Δa, height scaled by ν), amplitude
  K(θ; a) = ∫ 4π r γ J0(q r sin θ) j0(q γ cos θ) dr, γ = ν √(a² - (r-R)²), r >= 0;
  A = (η_sh - η_sol) K(a + Δa) + (η_c - η_sh) K(a), P = ∫_0^{π/2} A² sin θ dθ.

The orientation integrals use Gauss-Legendre on panels of at most ~2 rad of phase (exact to rounding for all q).

C fixes of 2026-11-24 (cylinders), followed here: ellcylshell1/2 with ε ≠ 1 averaged A² sin α/(4π) with
``sasfit_orient_avg``, which already weights uniformly in cos α (result ~ 1/(4π) too small and discontinuous at
ε = 1); cylshell2/ellcylshell2 volumes without the caps (L instead of L + 2t); flatcylinder _f and _v swapped (the
amplitude returned the volume); the torus core integral started at R - a < 0 for a > R (negative radii counted);
disc and its amplitude lost digits for 1e-6 < qR < 1e-2 (series only below 1e-6, 1 - cos instead of 2 sin²).
"""

from __future__ import annotations

import math

import numpy as np
from scipy import special

from .common import make_spec
from .quadrature import gauss_legendre
from .thin_objects import pcs_ellcylsh, rod_pprime

__all__ = ["SPECS"]

_GN = 16


def _lam(u):
    """2 J1(u)/u (1 at u = 0)."""
    u = np.asarray(u, float)
    small = np.abs(u) < 1e-4
    us = np.where(small, 1.0, u)
    return np.where(small, 1.0 - u * u / 8.0, 2.0 * special.j1(us) / us)


def _j0(x):
    x = np.asarray(x, float)
    small = np.abs(x) < 1e-4
    xs = np.where(small, 1.0, x)
    return np.where(small, 1.0 - x * x / 6.0, np.sin(xs) / xs)


def _panels(a, b, phase, width=2.0, nmin=1):
    """Gauss-Legendre nodes/weights on [a, b] with panels of at most ``width`` rad of ``phase`` (total phase over
    the interval)."""
    n = max(nmin, math.ceil(abs(phase) / width))
    e = np.linspace(a, b, n + 1)
    x, w = gauss_legendre(_GN, e[:-1], e[1:])
    return x.ravel(), w.ravel()


def _vec(p, n=50):
    v = np.zeros(max(n, len(p)))
    v[: len(p)] = p
    return v


# ------------------------------------------------------------------------------------------------ cylinder
def _cyl_amp(qi, R, L, eta, al):
    return math.pi * R * R * L * eta * _lam(qi * R * np.sin(al)) * _j0(0.5 * qi * L * np.cos(al))


def _cylinder(q, p, power):
    q = np.atleast_1d(np.asarray(q, float))
    R, L, eta = p[0], p[2], p[3]
    if R < 0 or L < 0:
        raise ValueError("cylinder: R >= 0 and L >= 0 required")
    out = np.empty(q.shape)
    for i, qi in enumerate(q):
        al, w = _panels(0.0, math.pi / 2, qi * max(R, 0.5 * L) * math.pi / 2, nmin=2)
        out[i] = np.sum(w * np.sin(al) * _cyl_amp(qi, R, L, eta, al) ** power)
    return out


def cylinder(q, p):
    """p = [R, dummy, L, eta]."""
    return _cylinder(q, p, 2).reshape(np.shape(q))


def cylinder_f(q, p):
    return _cylinder(q, p, 1).reshape(np.shape(q))


def _vol_cyl(x, p, dist):
    pp = _vec(p)
    R = np.where(dist == 0, x, pp[0])
    L = np.where(dist == 2, x, pp[2])
    return math.pi * R * R * L


# ------------------------------------------------------------------------------------------------ disc, rod
def disc(q, p):
    """p = [R, dummy, dummy, eta]: (π R² η)² 2/(qR)² [1 - J1(2qR)/(qR)] (series for qR < 0.25)."""
    q = np.asarray(q, float)
    R, eta = p[0], p[3]
    if R < 0:
        raise ValueError("disc: R >= 0 required")
    x = q * R
    u = x * x
    # 2/x² (1 - J1(2x)/x) = 2 Σ_{k>=1} (-1)^{k+1} x^{2k-2} / (k! (k+1)!)
    series = np.zeros_like(u)
    term = np.ones_like(u)
    for k in range(1, 12):
        series += (-1) ** (k + 1) * 2.0 * term / (math.factorial(k) * math.factorial(k + 1))
        term = term * u
    xs = np.where(x < 0.25, 1.0, x)
    exact = 2.0 / (xs * xs) * (1.0 - special.j1(2.0 * xs) / xs)
    return (eta * math.pi * R * R) ** 2 * np.where(x < 0.25, series, exact)


def disc_f(q, p):
    """η π R² 2 (1 - cos qR)/(qR)² = η π R² sinc²(qR/2)."""
    q = np.asarray(q, float)
    h = 0.5 * q * p[0]
    return p[3] * math.pi * p[0] ** 2 * _j0(h) ** 2


def rod(q, p):
    """p = [L, dummy, dummy, eta]: η² L² [2 Si(qL)/(qL) - j0(qL/2)²]."""
    q = np.asarray(q, float)
    L, eta = p[0], p[3]
    if L < 0:
        raise ValueError("rod: L >= 0 required")
    u = q * L
    us = np.where(u < 1e-3, 1.0, u)
    si, _ = special.sici(us)
    g = np.where(u < 1e-3, 1.0 - u * u / 18.0 + u ** 4 / 600.0 - u ** 6 / 35280.0, si / us)
    return eta * eta * L * L * (2.0 * g - _j0(0.5 * u) ** 2)


# Small-q series of the two Porod formulas, S(q) = Σ c_k q^{2k} (coefficients from the formulas by sympy; the
# closed forms cancel to ~1e-4 at qL = 1e-4). Long: a = L, b = 2R; flat: a = L, b = 2R.
def _long_coeffs(a, b):
    return [
        1.0,
        -1/2304*(64*a**4 + 96*a**2*b**2 + 5*b**4)/a**2,
        (1/1843200)*(1024*a**6 + 1280*a**4*b**2 + 2000*a**2*b**4 + 105*b**6)/a**2,
        -1/17340825600*(122880*a**8 + 172032*a**6*b**2 + 313600*a**4*b**4 + 274400*a**2*b**6 + 12789*b**8)/a**2,
        (1/7491236659200)*(458752*a**10 + 737280*a**8*b**2 + 1935360*a**6*b**4 + 1975680*a**4*b**6
                           + 1111320*a**2*b**8 + 45045*b**10)/a**2,
        -1/116024273377689600*(44040192*a**12 + 80740352*a**10*b**2 + 297369600*a**8*b**4 + 437133312*a**6*b**6
                               + 286868736*a**4*b**8 + 112698432*a**2*b**10 + 3978117*b**12)/a**2,
        (1/18823778112796360704000)*(33218887680*a**14 + 68702699520*a**12*b**2 + 341127987200*a**10*b**4
                                     + 703576473600*a**8*b**6 + 664879767552*a**6*b**8 + 304736560128*a**4*b**10
                                     + 88428019680*a**2*b**12 + 2737609875*b**14)/a**2,
    ]


def _flat_coeffs(a, b):
    return [
        1.0,
        -1/360*(4*a**4 + 10*a**2*b**2 + 15*b**4)/b**2,
        (1/5760)*(2*a**6 + 8*a**4*b**2 + 4*a**2*b**4 + 5*b**6)/b**2,
        -1/29030400*(152*a**8 + 900*a**6*b**2 + 1008*a**4*b**4 + 300*a**2*b**6 + 315*b**8)/b**2,
        (1/696729600)*(34*a**10 + 280*a**8*b**2 + 540*a**6*b**4 + 360*a**4*b**6 + 70*a**2*b**8 + 63*b**10)/b**2,
        -1/5579410636800*(1748*a**12 + 19110*a**10*b**2 + 56056*a**8*b**4 + 64350*a**6*b**6 + 28028*a**4*b**8
                          + 3822*a**2*b**10 + 3003*b**12)/b**2,
        (1/2678117105664000)*(3958*a**14 + 55440*a**12*b**2 + 229320*a**10*b**4 + 400400*a**8*b**6
                              + 300300*a**6*b**8 + 91728*a**4*b**10 + 9240*a**2*b**12 + 6435*b**14)/b**2,
    ]


_SERIES_X = 0.3   # series for q·max(L, 2R) below this value


def _poly(c, q):
    u = np.asarray(q, float) ** 2
    out = np.zeros_like(u)
    for ck in reversed(c):
        out = out * u + ck
    return out


# ------------------------------------------------------------------------------------------------ Porod cylinders
def _omega(sig):
    """8/σ² (3 J2(σ) + J0(σ) - 1), by its series for σ < 1 (the closed form cancels for small σ)."""
    sig = np.asarray(sig, float)
    h2 = (0.5 * sig) ** 2
    ser = np.zeros_like(sig)
    pw = np.ones_like(sig)
    for k in range(1, 16):
        ck = (-1) ** k * (1.0 / math.factorial(k) ** 2 - 3.0 / (math.factorial(k - 1) * math.factorial(k + 1)))
        ser += 2.0 * ck * pw
        pw = pw * h2
    ss = np.where(sig < 1.0, 1.0, sig)
    closed = 8.0 / ss ** 2 * (3.0 * special.jv(2, ss) + special.j0(ss) - 1.0)
    return np.where(sig < 1.0, ser, closed)


def longcylinder(q, p):
    """p = [R, dummy, L, eta]: Porod's approximation for L >> R."""
    q = np.asarray(q, float)
    R, L, eta = p[0], p[2], p[3]
    if R < 0 or L < 0:
        raise ValueError("longcylinder: R >= 0 and L >= 0 required")
    V = math.pi * R * R * L
    if R == 0 or L == 0:
        return np.zeros(q.shape)
    mu = np.where(q == 0, 1.0, L * q)
    sig = np.where(q == 0, 1.0, 2.0 * R * q)
    si, _ = special.sici(mu)
    pi_mu = si + np.cos(mu) / mu + np.sin(mu) / mu ** 2
    G1 = 2.0 / (0.5 * sig) * special.j1(0.5 * sig)
    omega = _omega(sig)
    s = 2.0 / mu * (pi_mu * G1 * G1 - omega / mu - np.sin(mu) / mu ** 2)
    small = q * max(L, 2.0 * R) < _SERIES_X
    s = np.where(small, _poly(_long_coeffs(L, 2.0 * R), q), s)
    return (eta * V) ** 2 * s


def flatcylinder(q, p):
    """p = [R, dummy, L, eta]: Porod's approximation for L << R (valid for qL ≲ 1)."""
    q = np.asarray(q, float)
    R, L, eta = p[0], p[2], p[3]
    if R < 0 or L < 0:
        raise ValueError("flatcylinder: R >= 0 and L >= 0 required")
    V = math.pi * R * R * L
    if V == 0:
        return np.zeros(q.shape)
    sig = np.where(q == 0, 1.0, L * q)
    mu = np.where(q == 0, 1.0, 2.0 * R * q)

    def I1(x):
        return (2.0 * x * special.j0(x) - 2.0 * special.j1(x)
                - math.pi * x * (special.j0(x) * special.struve(1, x) - special.j1(x) * special.struve(0, x)))

    G1s = 2.0 / sig * special.j1(sig)
    G1m = 2.0 / mu * special.j1(mu)
    I1m, I1s = I1(mu), I1(sig)
    I0s = 0.5 * (I1s + sig * G1s)
    Om = 2.0 / sig * (I0s - 2.0 * special.j1(sig))
    xi = (np.sin(0.5 * sig) / (0.5 * sig)) ** 2
    s = 8.0 / mu ** 2 * (xi + I1m * Om / mu - G1m)
    small = q * max(L, 2.0 * R) < _SERIES_X
    s = np.where(small, _poly(_flat_coeffs(L, 2.0 * R), q), s)
    return (eta * V) ** 2 * s


def _porod_blend(q, R, L, eta):
    pp = [R, 0.0, L, eta]
    x = 2.0 * R / L
    if x > 1.25:
        return flatcylinder(q, pp)
    if x < 0.75:
        return longcylinder(q, pp)
    t = (x - 0.75) * 2.0
    return t * flatcylinder(q, pp) + (1.0 - t) * longcylinder(q, pp)


def porodcylinder(q, p):
    """p = [R, dummy, L, eta]."""
    q = np.asarray(q, float)
    if p[0] < 0 or p[2] < 0:
        raise ValueError("porodcylinder: R >= 0 and L >= 0 required")
    if p[0] == 0 or p[2] == 0:
        return np.zeros(q.shape)
    return _porod_blend(q, p[0], p[2], p[3])


def porodcylinder_v2(q, p):
    """p = [R, dummy, nu, eta]: L = ν R."""
    q = np.asarray(q, float)
    if p[0] < 0 or p[2] < 0:
        raise ValueError("porodcylinder_v2: R >= 0 and nu >= 0 required")
    if p[0] == 0 or p[2] == 0:
        return np.zeros(q.shape)
    return _porod_blend(q, p[0], p[2] * p[0], p[3])


# ------------------------------------------------------------------------------------------------ core-shell cylinders
def longcylshell(q, p):
    """p = [R, t, L, eta_core, eta_shell, eta_solv]: Pcs(R, t) · P'_rod(L)."""
    q = np.asarray(q, float)
    R, t, L = p[0], p[1], p[2]
    if R < 0 or t < 0 or L < 0:
        raise ValueError("longcylshell: R, t, L >= 0 required")
    pcs = pcs_ellcylsh(q, R, 0.0, 1.0, t, 0.0, p[3], p[4], p[5])
    return (pcs * rod_pprime(q, L, 0.0)).reshape(q.shape)


def _ellcylshell(q, p, typ, power, eps=None):
    q = np.atleast_1d(np.asarray(q, float))
    R, t, L = p[0], p[1], p[3]
    eps = p[2] if eps is None else eps
    if R < 0 or t < 0 or L < 0 or eps <= 0:
        raise ValueError("ellcylshell: R, t, L >= 0 and epsilon > 0 required")
    ec, es, ev = p[4], p[5], p[6]
    b1 = math.pi * R * R * eps * L * (ec - es)
    b2 = math.pi * (R + t) * (eps * R + t) * (L + 2.0 * t * typ) * (es - ev)
    out = np.empty(q.shape)
    size = max(R, eps * R) + t
    for i, qi in enumerate(q):
        al, wa = _panels(0.0, math.pi / 2, qi * max(size, 0.5 * L + t) * math.pi / 2, nmin=2)
        sa, ca = np.sin(al), np.cos(al)
        c1 = _j0(qi * 0.5 * L * ca)
        c2 = _j0(qi * (0.5 * L + t * typ) * ca)
        if eps == 1.0:
            ph, wp = np.array([0.0]), np.ones(1)
        else:
            ph, wp = _panels(0.0, math.pi / 2, qi * size * abs(1.0 - eps) * math.pi / 2)
            wp = wp * (2.0 / math.pi)
        r_in = np.hypot(R * np.sin(ph), eps * R * np.cos(ph))[:, None]
        r_out = np.hypot((R + t) * np.sin(ph), (eps * R + t) * np.cos(ph))[:, None]
        A = b1 * c1 * _lam(qi * r_in * sa) + b2 * c2 * _lam(qi * r_out * sa)          # (φ, α)
        out[i] = wp @ (A ** power) @ (wa * sa)
    return out


def _mk_ellcyl(typ, power, eps1=False):
    def f(q, p):
        return _ellcylshell(q, p, typ, power, 1.0 if eps1 else None).reshape(np.shape(q))
    return f


ellcylshell1, ellcylshell1_f = _mk_ellcyl(0, 2), _mk_ellcyl(0, 1)
ellcylshell2, ellcylshell2_f = _mk_ellcyl(1, 2), _mk_ellcyl(1, 1)
cylshell1, cylshell1_f = _mk_ellcyl(0, 2, True), _mk_ellcyl(0, 1, True)
cylshell2, cylshell2_f = _mk_ellcyl(1, 2, True), _mk_ellcyl(1, 1, True)


def _vol_ellcyl(typ, eps1=False):
    def v(x, p, dist=0):
        pp = _vec(p)
        R = np.where(dist == 0, x, pp[0])
        t = np.where(dist == 1, x, pp[1])
        eps = 1.0 if eps1 else np.where(dist == 2, x, pp[2])
        L = np.where(dist == 3, x, pp[3])
        return math.pi * (R + t) * (R * eps + t) * (L + 2.0 * t * typ)
    return v


# ------------------------------------------------------------------------------------------------ torus
def _torus_K(qi, R, a, nu, th):
    """K(θ; a) for all θ (array), r = R + a sin β, β from max(-π/2, asin(-R/a)) (r >= 0) to π/2."""
    if a <= 0:
        return np.zeros(th.shape)
    b0 = -math.pi / 2 if a <= R else math.asin(-R / a)
    be, wb = _panels(b0, math.pi / 2, qi * (R + a) * (math.pi / 2 - b0), nmin=4)
    r = R + a * np.sin(be)
    gam = nu * a * np.cos(be)
    dr = a * np.cos(be) * wb
    st, ct = np.sin(th)[:, None], np.cos(th)[:, None]
    integ = 4.0 * math.pi * r * gam * special.j0(qi * r * st) * _j0(qi * gam * ct)
    return integ @ dr


def _torus(q, p, power):
    q = np.atleast_1d(np.asarray(q, float))
    R, a, da, nu, ec, es, ev = p[0], p[1], p[2], p[3], p[4], p[5], p[6]
    if R < 0 or a < 0 or da < 0 or nu < 0:
        raise ValueError("torus: R, a, Delta_a, nu >= 0 required")
    out = np.empty(q.shape)
    for i, qi in enumerate(q):
        th, w = _panels(0.0, math.pi / 2, qi * (R + a + da + nu * (a + da)) * math.pi / 2, nmin=4)
        A = (es - ev) * _torus_K(qi, R, a + da, nu, th) + (ec - es) * _torus_K(qi, R, a, nu, th)
        out[i] = np.sum(w * np.sin(th) * A ** power)
    return out


def torus(q, p):
    """p = [R, a, Delta_a, nu, eta_c, eta_sh, eta_sol]."""
    return _torus(q, p, 2).reshape(np.shape(q))


def torus_f(q, p):
    return _torus(q, p, 1).reshape(np.shape(q))


# ------------------------------------------------------------------------------------------------ specs
_SRC = "plugins/cylinders/sasfit_ff_"
_GROUP = "ff_plugins_cylindrical_obj"


def _red_rl(q, p):
    return np.asarray(q, float) * max(abs(p[0]), 0.5 * abs(p[2]))


_RL = ["R", "dummy", "L", "eta"]
_RLS = {0: (2.0, 30.0, "log"), 2: (2.0, 120.0, "log"), 3: (0.5, 2.0, "lin")}
_ECS = ["R", "t", "epsilon", "L", "eta_core", "eta_shell", "eta_solv"]
_ECSS = {0: (2.0, 20.0, "log"), 1: (0.5, 6.0, "log"), 2: (0.4, 2.5, "log"), 3: (5.0, 100.0, "log"),
         4: (-1.0, 3.0, "lin"), 5: (0.0, 2.0, "lin"), 6: (-0.5, 0.5, "lin")}
_ECSS1 = {k: v for k, v in _ECSS.items() if k != 2}


def _red_ecs(q, p):
    return np.asarray(q, float) * (max(abs(p[0]), abs(p[0] * p[2])) + abs(p[1]) + 0.5 * abs(p[3]))


SPECS = [
    make_spec("ff_cylinder", cylinder, _RL, _RLS, amplitude=cylinder_f, volume=lambda x, p, d: _vol_cyl(x, p, d),
              source=_SRC + "cylinder.c", reduced=_red_rl, reduced_label="q max(R, L/2)", rtol=1e-10, group=_GROUP),
    make_spec("ff_disc", disc, ["R", "dummy", "dummy", "eta"], {0: (1.0, 50.0, "log"), 3: (0.5, 2.0, "lin")},
              amplitude=disc_f, volume=lambda x, p, d: math.pi * np.asarray(x if d == 0 else p[0], float) ** 2,
              source=_SRC + "disc.c", reduced=lambda q, p: np.asarray(q) * abs(p[0]), reduced_label="q R",
              rtol=1e-11, group=_GROUP),
    make_spec("ff_rod", rod, ["L", "dummy", "dummy", "eta"], {0: (1.0, 300.0, "log"), 3: (0.5, 2.0, "lin")},
              source=_SRC + "rod.c", reduced=lambda q, p: np.asarray(q) * abs(p[0]), reduced_label="q L",
              rtol=1e-11, group=_GROUP),
    make_spec("ff_longcylinder", longcylinder, _RL, {0: (1.0, 5.0, "log"), 2: (50.0, 300.0, "log"),
                                                     3: (0.5, 2.0, "lin")},
              volume=lambda x, p, d: _vol_cyl(x, p, d), source=_SRC + "longcylinder.c", reduced=_red_rl,
              reduced_label="q max(R, L/2)", rtol=1e-9, group=_GROUP),
    make_spec("ff_flatcylinder", flatcylinder, _RL, {0: (20.0, 80.0, "log"), 2: (0.5, 4.0, "log"),
                                                     3: (0.5, 2.0, "lin")},
              volume=lambda x, p, d: _vol_cyl(x, p, d), source=_SRC + "flatcylinder.c", reduced=_red_rl,
              reduced_label="q max(R, L/2)", rtol=1e-8, group=_GROUP,
              notes="Porod's approximation for L << R, valid for qL ≲ 1 (negative beyond)"),
    make_spec("ff_porodcylinder", porodcylinder, _RL, _RLS, volume=lambda x, p, d: _vol_cyl(x, p, d),
              source=_SRC + "porodcylinder.c", reduced=_red_rl, reduced_label="q max(R, L/2)", rtol=1e-8,
              group=_GROUP),
    make_spec("ff_porodcylinder_v2", porodcylinder_v2, ["R", "dummy", "nu", "eta"],
              {0: (2.0, 30.0, "log"), 2: (0.2, 20.0, "log"), 3: (0.5, 2.0, "lin")},
              volume=lambda x, p, d: math.pi * np.asarray(x if d == 0 else p[0], float) ** 3 * (p[2] if d != 2 else x),
              source=_SRC + "porodcylinder_v2.c",
              reduced=lambda q, p: np.asarray(q) * abs(p[0]) * max(1.0, 0.5 * abs(p[2])), reduced_label="q max(R, L/2)",
              rtol=1e-8, group=_GROUP),
    make_spec("ff_longcylshell", longcylshell, ["R", "t", "L", "eta_core", "eta_shell", "eta_solv"],
              {0: (2.0, 20.0, "log"), 1: (0.5, 6.0, "log"), 2: (50.0, 500.0, "log"), 3: (-1.0, 3.0, "lin"),
               4: (0.0, 2.0, "lin"), 5: (-0.5, 0.5, "lin")},
              source=_SRC + "longcylshell.c", reduced=lambda q, p: np.asarray(q) * (abs(p[0]) + abs(p[1])),
              reduced_label="q (R + t)", rtol=1e-10, group=_GROUP),
    make_spec("ff_ellcylshell1", ellcylshell1, _ECS, _ECSS, amplitude=ellcylshell1_f, volume=_vol_ellcyl(0),
              source=_SRC + "ellcylshell1.c", reduced=_red_ecs, reduced_label="q size", rtol=1e-10, group=_GROUP),
    make_spec("ff_ellcylshell2", ellcylshell2, _ECS, _ECSS, amplitude=ellcylshell2_f, volume=_vol_ellcyl(1),
              source=_SRC + "ellcylshell2.c", reduced=_red_ecs, reduced_label="q size", rtol=1e-10, group=_GROUP),
    make_spec("ff_cylshell1", cylshell1, _ECS, _ECSS1, amplitude=cylshell1_f, volume=_vol_ellcyl(0, True),
              source=_SRC + "cylshell1.c", reduced=_red_ecs, reduced_label="q size", rtol=1e-10, group=_GROUP),
    make_spec("ff_cylshell2", cylshell2, _ECS, _ECSS1, amplitude=cylshell2_f, volume=_vol_ellcyl(1, True),
              source=_SRC + "cylshell2.c", reduced=_red_ecs, reduced_label="q size", rtol=1e-10, group=_GROUP),
    make_spec("ff_torus", torus, ["R", "a", "Delta_a", "nu", "eta_c", "eta_sh", "eta_sol"],
              {0: (5.0, 40.0, "log"), 1: (1.0, 4.0, "log"), 2: (0.0, 2.0, "lin"), 3: (0.5, 2.0, "log"),
               4: (-1.0, 3.0, "lin"), 5: (0.0, 2.0, "lin"), 6: (-0.5, 0.5, "lin")},
              amplitude=torus_f, source=_SRC + "torus.c",
              reduced=lambda q, p: np.asarray(q) * (abs(p[0]) + abs(p[1]) + abs(p[2])), reduced_label="q (R + a)",
              rtol=1e-9, group=_GROUP),
]
