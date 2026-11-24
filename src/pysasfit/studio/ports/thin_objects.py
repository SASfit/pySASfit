"""Python ports of the SASfit plugin thin_objects (batch 6, 2026-11-24): elongated objects with an elliptical
core-shell cross-section, P(q) = Pcs(q) · P'(q).

* ``ff_pcs_ellcylsh``: cross-section form factor of an elliptical core-shell cylinder, core semi-axes R and εR,
  shell thickness t, log-normal distributions of R and t (median R0, t0; widths σ):
  A = π R² ε (η_core - η_shell) Λ(q r_in) + π (R + t)(εR + t)(η_shell - η_solv) Λ(q r_out), Λ(u) = 2 J1(u)/u,
  r(α) = √((a sin α)² + (b cos α)²) for the semi-axes (a, b) of the ellipse, Pcs = <A²> over α (uniform in
  [0, π/2]) and the size distributions. Pcs(0) = <(Δη A)²>.
* ``ff_rod_sd_ellcylsh``: Pcs · P'(q) of a thin rod of length L (log-normal, median L0):
  P' = <L² [2 Si(qL)/(qL) - (sin(qL/2)/(qL/2))²]>.
* ``ff_worm_ps1_sd_ellcylsh``, ``ff_worm_ps2_sd_ellcylsh``, ``ff_worm_ps3_sd_ellcylsh``,
  ``ff_worm_kholodenko_sd_ellcylsh``: Pcs · L² S(q) with the worm-like chain S(q) of
  :mod:`.worms` (Pedersen-Schurtenberger methods 1-3, Kholodenko). I(0) = (Δη A L)².

Cross-section form factors Pcs(q) (per unit length or area; the thin_objects functions multiply them with P'(q)):

* ``ff_pcs_homogeneouscyl``: <(π R² Δη Λ(qR))²> over a log-normal R;
* ``ff_pcs_homogeneousplate``: Δη² <t² j0(qt/2)²> over a log-normal thickness (σ <= 1e-6: monodisperse, as C);
* ``ff_pcs_twoinfinitelythinlayers``: <cos²(qt/2)> (two thin layers at distance t, Pcs(0) = 1);
* ``ff_pcs_bilayergauss``: <(√(2π) σ_c b_c e^{-q²σ_c²/2} + 2 √(2π) σ_o b_o e^{-q²σ_o²/2} cos(q D ν/2))²> over
  a log-normal ν (median 1, width σ_D);
* ``ff_pcs_layeredcentrosymmetricxs``: <[(η_sh - η_sol)(L + 2S) j0(q(L + 2S)/2) - (η_sh - η_c) L j0(qL/2)]²>
  over log-normal core thickness L and shell thickness S;
* ``ff_pcs_plate_chains_rw_``: planar core layer with Gaussian chains (Pedersen), per area of a disc of radius
  100 L_core (as C).

Thin discs (P'(q) of an infinitely thin disc of radius R, log-normal R:
P' = <2 (π R/q)² (1 - 2 J1(2qR)/(2qR))>, P'(0) = <(π R²)²>):

* ``ff_disc_sd_homoxs``: Pcs of a homogeneous plate (t, σ_t, η_L, η_sol) · P'_disc(R0, σ_R0);
* ``ff_disc_sd_bilayergauss``: Pcs of the Gaussian bilayer (D = t, σ_D = σ_t) · P'_disc(R0, σ_R0).

Thin closed shells (batch 8, 2026-11-24), P = Pcs · P':

* ``ff_ellsh_sd_homoxs``, ``ff_ellsh_sd_bilayergauss``: P' of an infinitely thin spheroidal shell (semi-axes
  R, R, εR; surface S), <(S j0(q R √(sin²α + ε² cos²α)))²> over α and a log-normal R; amplitudes (``_f``) as
  products of the averaged amplitudes (as C); volume <S> t;
* ``ff_cylsh_sd_homoxs``, ``ff_cylsh_sd_bilayergauss``: P' of an infinitely thin closed cylindrical shell
  (radius R, height H): <(2π R² Λ(qR sin α) cos(qH cos α/2) + 2π R H J0(qR sin α) j0(qH cos α/2))²> over α and
  log-normal R, H (C has no amplitude and volume).

The integrals over α and the log-normal distributions are done by Gauss-Legendre on panels sized to the phase
q·(radius) (ln of the size as variable, ±10 σ around the peak of the weight), so they are exact to rounding for
all q.

C fixes of 2026-11-24 (thin_objects), followed here: Pcs of an elliptical cross-section (ε ≠ 1) integrated over
α ∈ [0, π/2] without dividing by π/2 (π/2 too large, discontinuous at ε = 1); the worm variants passed their own
parameter vector to Pcs, whose layout differs: methods 1-3 read lb as the width σ_t of the thickness distribution,
the Kholodenko variant read the contrasts one place too early (η_core = dummy, η_shell = η_core,
η_solv = η_shell). Method 3 passes no excluded volume (header: dummy), as in C. P'(q) of the thin disc was
0/0 = NaN at q = 0 and lost digits for small qR (1 - 2J1(2u)/(2u) cancels; series below u = 0.1 now);
ff_disc_sd_homoxs used half of its R0 parameter as the disc radius (named D in the code, R0 in the header and in
ff_disc_sd_bilayergauss); Pcs:BilayerGauss used an uninitialised parameter structure for the log-normal.
Batch 8: Pcs:BilayerGauss had no amplitude (``_f`` returned 0, so ellsh_sd_bilayergauss_f was 0); the volume of
ellsh_sd_bilayergauss was (4π R0/3)³ e^{9σ²/2} (not a volume), that of ellsh_sd_homoxs 0 — both are <S> t now.
"""

from __future__ import annotations

import math

import numpy as np
from scipy import special

from .common import make_spec
from .quadrature import gauss_legendre
from .worms import kholodenko, worm_ps1, worm_ps2, worm_ps3

__all__ = ["SPECS", "pcs_ellcylsh", "rod_pprime", "disc_pprime", "lognorm_average", "thin_ellipsoidal_shell_pprime",
           "thin_hollow_cylinder_pprime"]

_GN = 16


def _lam(u):
    """2 J1(u)/u (1 at u = 0)."""
    u = np.asarray(u, float)
    small = np.abs(u) < 1e-4
    us = np.where(small, 1.0, u)
    return np.where(small, 1.0 - u * u / 8.0, 2.0 * special.j1(us) / us)


def _lognorm_axis(sigma, phase, shift):
    """Nodes ν and weights of ∫ LN(ν; median 1, σ) (...) dν as 16-point Gauss-Legendre in t = ln ν over
    [shift - 8σ, shift + 8σ] (shift: peak of the weight ν^k LN, i.e. k σ²), panels at most 2σ wide in t and at
    most 5 rad of the phase ``phase``·ν. σ = 0: the single node ν = 1."""
    if sigma == 0:
        return np.ones(1), np.ones(1)
    lo, hi = shift - 8.0 * sigma, shift + 8.0 * sigma
    n_t = 8
    n_ph = min(4000, math.ceil(phase * (math.exp(hi) - math.exp(lo)) / 5.0))
    edges = np.linspace(lo, hi, n_t + 1)
    if n_ph > n_t:
        edges = np.unique(np.concatenate([edges, np.log(np.linspace(math.exp(lo), math.exp(hi), n_ph + 1))]))
    t, w = gauss_legendre(_GN, edges[:-1], edges[1:])
    t, w = t.ravel(), w.ravel()
    w = w * np.exp(-0.5 * (t / sigma) ** 2) / (math.sqrt(2.0 * math.pi) * sigma)
    return np.exp(t), w


def pcs_ellcylsh(q, R0, sig_R, eps, t0, sig_t, eta_core, eta_shell, eta_solv):
    """Pcs(q) of the elliptical core-shell cross-section (see the module docstring)."""
    q = np.atleast_1d(np.asarray(q, float))
    sig_R, sig_t = abs(sig_R), abs(sig_t)
    out = np.empty(q.shape)
    dc, ds = eta_core - eta_shell, eta_shell - eta_solv
    for i, qi in enumerate(q):
        size = (R0 * max(eps, 1.0) + t0)
        n1, w1 = _lognorm_axis(sig_R, qi * R0 * max(eps, 1.0), 4.0 * sig_R ** 2)
        n2, w2 = _lognorm_axis(sig_t, qi * t0, 2.0 * sig_t ** 2)
        if eps != 1.0:
            nmax = max(n1.max(), 1.0)
            pan = max(1, math.ceil(qi * size * nmax * abs(1.0 - eps) / 5.0))
            ed = np.linspace(0.0, math.pi / 2, pan + 1)
            al, wa = gauss_legendre(_GN, ed[:-1], ed[1:])
            al, wa = al.ravel(), wa.ravel() * (2.0 / math.pi)
        else:
            al, wa = np.array([math.pi / 2]), np.ones(1)
        sa, ca = np.sin(al)[:, None], np.cos(al)[:, None]
        R = R0 * n1[None, :]
        r_in = np.sqrt((R * sa) ** 2 + (R * eps * ca) ** 2)
        a1 = math.pi * R * R * eps * dc * _lam(qi * r_in)                # (α, R)
        acc = 0.0
        for k in range(n2.size):                                          # thickness nodes: (α, R) slabs
            T = t0 * n2[k]
            r_out = np.sqrt(((R + T) * sa) ** 2 + ((R * eps + T) * ca) ** 2)
            A = a1 + math.pi * (R + T) * (R * eps + T) * ds * _lam(qi * r_out)
            acc += w2[k] * (wa @ (A * A) @ w1)
        out[i] = acc
    return out


def rod_pprime(q, L0, sig_L):
    """P'(q) of a thin rod: <L² [2 Si(qL)/(qL) - sinc²(qL/2)]> over a log-normal L (median L0)."""
    q = np.atleast_1d(np.asarray(q, float))
    sig_L = abs(sig_L)

    def ip(u):
        us = np.where(u == 0, 1.0, u)
        si, _ = special.sici(us)
        h = 0.5 * us
        return np.where(u == 0, 1.0, 2.0 * si / us - (np.sin(h) / h) ** 2)

    out = np.empty(q.shape)
    for i, qi in enumerate(q):
        nu, w = _lognorm_axis(sig_L, qi * L0, 2.0 * sig_L ** 2)
        x = L0 * nu
        out[i] = np.sum(w * x * x * ip(qi * x))
    return out


def lognorm_average(q, f, x0, sigma, power):
    """<f(q, x)> over a log-normal x (median x0, width σ; σ = 0: f(q, x0)); ``power``: growth of the weight with x
    (sets the centre of the t = ln(x/x0) range). f is vectorised in x."""
    q = np.atleast_1d(np.asarray(q, float))
    out = np.empty(q.shape)
    for i, qi in enumerate(q):
        nu, w = _lognorm_axis(abs(sigma), qi * abs(x0), power * sigma ** 2)
        out[i] = np.sum(w * f(qi, x0 * nu))
    return out


def _j0(x):
    x = np.asarray(x, float)
    small = np.abs(x) < 1e-4
    xs = np.where(small, 1.0, x)
    return np.where(small, 1.0 - x * x / 6.0, np.sin(xs) / xs)


def ff_pcs_homogeneouscyl(q, p):
    """p = [R, sigma_R, dummy x5, eta_core, dummy, eta_sol]: <(π R² Δη 2 J1(qR)/(qR))²>."""
    if p[0] < 0 or p[1] < 0:
        raise ValueError("pcs_homogeneouscyl: R >= 0 and sigma_R >= 0 required")
    d = p[7] - p[9]
    f = lambda qi, x: (math.pi * x * x * d * _lam(qi * x)) ** 2  # noqa: E731
    return lognorm_average(q, f, p[0], p[1], 4.0).reshape(np.shape(q))


def ff_pcs_homogeneousplate(q, p):
    """p = [t, sigma_t, dummy, dummy, eta_L, eta_solv]: Δη² <t² j0(qt/2)²>."""
    if p[0] < 0:
        raise ValueError("pcs_homogeneousplate: t >= 0 required")
    sig = abs(p[1]) if abs(p[1]) > 1e-6 else 0.0
    f = lambda qi, x: (x * _j0(0.5 * qi * x)) ** 2  # noqa: E731
    return ((p[4] - p[5]) ** 2 * lognorm_average(q, f, p[0], sig, 2.0)).reshape(np.shape(q))


def ff_pcs_twoinfinitelythinlayers(q, p):
    """p = [t, sigma_t]: <cos²(qt/2)>."""
    if p[0] < 0:
        raise ValueError("pcs_twoinfinitelythinlayers: t >= 0 required")
    f = lambda qi, x: np.cos(0.5 * qi * x) ** 2  # noqa: E731
    return lognorm_average(q, f, p[0], abs(p[1]), 0.0).reshape(np.shape(q))


def ff_pcs_bilayergauss(q, p):
    """p = [sigma_core, b_core, sigma_out, b_out, D, sigma_D]."""
    sc, bc, so, bo, D, sD = abs(p[0]), p[1], abs(p[2]), p[3], p[4], abs(p[5])
    qa = np.atleast_1d(np.asarray(q, float))
    out = np.empty(qa.shape)
    for i, qi in enumerate(qa):
        fc = math.sqrt(2.0 * math.pi) * sc * bc * math.exp(-0.5 * (qi * sc) ** 2)
        fo = math.sqrt(2.0 * math.pi) * so * bo * math.exp(-0.5 * (qi * so) ** 2)
        nu, w = _lognorm_axis(sD, 0.5 * qi * abs(D), 0.0)
        out[i] = np.sum(w * (fc + 2.0 * fo * np.cos(0.5 * qi * D * nu)) ** 2)
    return out.reshape(np.shape(q))


def ff_pcs_layeredcentrosymmetricxs(q, p):
    """p = [Lc, sigma_Lc, Lsh, sigma_Lsh, eta_Lc, eta_Lsh, eta_solv]."""
    if p[0] < 0 or p[2] < 0:
        raise ValueError("pcs_layeredcentrosymmetricxs: Lc >= 0 and Lsh >= 0 required")
    Lc, sL, Ls, sS, ec, es, ev = p[0], abs(p[1]), p[2], abs(p[3]), p[4], p[5], p[6]
    qa = np.atleast_1d(np.asarray(q, float))
    out = np.empty(qa.shape)
    for i, qi in enumerate(qa):
        n1, w1 = _lognorm_axis(sL, 0.5 * qi * Lc, 2.0 * sL ** 2)
        n2, w2 = _lognorm_axis(sS, qi * Ls, 2.0 * sS ** 2)
        L = Lc * n1[:, None]
        S = Ls * n2[None, :]
        F = (es - ev) * (L + 2.0 * S) * _j0(0.5 * qi * (L + 2.0 * S)) - (es - ec) * L * _j0(0.5 * qi * L)
        out[i] = w1 @ (F * F) @ w2
    return out.reshape(np.shape(q))


def _gauss_fc(q, rg):
    """Debye function 2 (e^{-u} - 1 + u)/u², u = (q Rg)² (series below 0.5)."""
    u = (np.asarray(q, float) * rg) ** 2
    small = u < 0.5
    us = np.where(small, 1.0, u)
    ser = np.zeros_like(u)
    t = np.ones_like(u)
    for k in range(18):
        ser = ser + t
        t = -t * np.where(small, u, 0.0) / (k + 3.0)
    return np.where(small, ser, 2.0 * (np.expm1(-us) + us) / (us * us))


def _rwbrush_w(q, rg):
    """(1 - e^{-u})/u, u = (q Rg)²."""
    u = (np.asarray(q, float) * rg) ** 2
    us = np.where(u < 1e-8, 1.0, u)
    return np.where(u < 1e-8, 1.0 - u / 2.0, -np.expm1(-us) / us)


def ff_pcs_plate_chains_rw_(q, p):
    """p = [L_core, n_agg, V_brush, eta_core, eta_brush, eta_solv, xsolv_core, Rg, d]: planar layer of thickness
    L_core with n_agg Gaussian chains per area (Pedersen's model), normalised to the area of a disc of radius
    100 max(L_core, 1) as in C."""
    Lc, nagg, vb, ec, eb, ev, xs, rg, d = (float(v) for v in p[:9])
    if Lc < 0 or nagg <= 0 or vb < 0 or not 0 <= xs < 1 or rg <= 0:
        raise ValueError("pcs_plate_chains_rw_: L_core >= 0, n_agg > 0, V_brush >= 0, 0 <= xsolv_core < 1, Rg > 0")
    q = np.asarray(q, float)
    surf = math.pi * (100.0 * (1.0 if Lc < 1.0 else Lc)) ** 2
    vcore = Lc * (1.0 - xs) / nagg
    N = surf * nagg
    rs = vcore * (ec - ev)
    rc = vb * (eb - ev)
    Fc = _gauss_fc(q, rg)
    w = _rwbrush_w(q, rg)
    Fs = _j0(0.5 * q * Lc)
    c = np.cos(q * (0.5 * Lc + d * rg))
    pairs = N * (N - 1.0) * (0.0 if N < 1.0 else 1.0)
    inten = (N * N * rs * rs * Fs * Fs + N * rc * rc * Fc + pairs * rc * rc * (w * c) ** 2
             + 2.0 * N * N * rs * rc * w * Fs * c)
    return inten / surf ** 2


def _disc_g(u):
    """2 (1 - 2 J1(2u)/(2u))/u² (1 at u = 0; series below 0.1)."""
    u = np.asarray(u, float)
    small = u < 0.1
    us = np.where(small, 1.0, u)
    u2 = np.where(small, u * u, 0.0)
    ser = np.zeros_like(u)
    t = np.ones_like(u)                       # 2 (-1)^(k+1) u^(2k-2)/(k!(k+1)!), k = 1: 1
    for k in range(1, 12):
        ser = ser + t
        t = -t * u2 / ((k + 1) * (k + 2))
    direct = 2.0 * (1.0 - _lam(2.0 * us)) / (us * us)
    return np.where(small, ser, direct)


def disc_pprime(q, R, sigma):
    """P'(q) of an infinitely thin disc, <(π R²)² g(qR)> over a log-normal R (median R)."""
    f = lambda qi, x: (math.pi * x * x) ** 2 * _disc_g(qi * x)  # noqa: E731
    return lognorm_average(q, f, R, abs(sigma), 4.0)


def ff_disc_sd_homoxs(q, p):
    """p = [t, sigma_t, R0, sigma_R0, dummy, dummy, eta_l, eta_sol]."""
    if p[0] < 0 or p[2] < 0:
        raise ValueError("disc_sd_homoxs: t >= 0 and R0 >= 0 required")
    pcs = ff_pcs_homogeneousplate(q, np.array([p[0], p[1], 0.0, 0.0, p[6], p[7]]))
    return (np.asarray(pcs).reshape(np.shape(q)) * disc_pprime(q, p[2], p[3]).reshape(np.shape(q)))


def ff_disc_sd_bilayergauss(q, p):
    """p = [t, sigma_t, R0, sigma_R0, dummy, dummy, sigma_out, b_out, sigma_core, b_core]: the Gaussian bilayer
    (head groups at ± t/2) times P' of the disc."""
    if p[0] < 0 or p[2] < 0:
        raise ValueError("disc_sd_bilayergauss: t >= 0 and R0 >= 0 required")
    pcs = ff_pcs_bilayergauss(q, np.array([p[8], p[9], p[6], p[7], p[0], p[1]]))
    return (np.asarray(pcs).reshape(np.shape(q)) * disc_pprime(q, p[2], p[3]).reshape(np.shape(q)))


# ------------------------------------------------------------------------------------------------ form factors
def _check(p, name, lb_index=None):
    if p[0] < 0 or p[2] <= 0 or p[3] < 0:
        raise ValueError(f"{name}: R >= 0, epsilon > 0 and t >= 0 required")
    if lb_index is not None and (p[lb_index] <= 0 or p[lb_index + 1] <= 0):
        raise ValueError(f"{name}: lb > 0 and L > 0 required")


def ff_pcs_ellcylsh(q, p):
    """p = [R0, sigma_R0, epsilon, t0, sigma_t0, dummy, dummy, eta_core, eta_shell, eta_solv]."""
    _check(p, "pcs_ellcylsh")
    return pcs_ellcylsh(q, p[0], p[1], p[2], p[3], p[4], p[7], p[8], p[9]).reshape(np.shape(q))


def ff_rod_sd_ellcylsh(q, p):
    """p = [R0, sigma_R0, epsilon, t0, sigma_t0, L0, sigma_L0, dummy, eta_core, eta_shell, eta_solv]."""
    _check(p, "rod_sd_ellcylsh")
    if p[5] < 0:
        raise ValueError("rod_sd_ellcylsh: L >= 0 required")
    pcs = pcs_ellcylsh(q, p[0], p[1], p[2], p[3], p[4], p[8], p[9], p[10])
    return (pcs * rod_pprime(q, p[5], p[6])).reshape(np.shape(q))


def _worm_sd(S, exvol_index=None, kholodenko_layout=False):
    def intensity(q, p):
        q = np.asarray(q, float)
        if kholodenko_layout:  # [R, sigma_R, eps, t0, sigma_t0, lb, L, dummy, eta_core, eta_shell, eta_solv]
            _check(p, "worm_kholodenko_sd_ellcylsh", 5)
            pcs = pcs_ellcylsh(q, p[0], p[1], p[2], p[3], p[4], p[8], p[9], p[10])
            lb, L = p[5], p[6]
            s = S(q, lb, L)
        else:  # [R, sigma_R, eps, t, lb, L, exvol | dummy, eta_core, eta_shell, eta_solv]
            _check(p, "worm_sd_ellcylsh", 4)
            pcs = pcs_ellcylsh(q, p[0], p[1], p[2], p[3], 0.0, p[7], p[8], p[9])
            lb, L = p[4], p[5]
            s = S(q, lb, L, p[exvol_index]) if exvol_index is not None else S(q, lb, L)
        s = np.where(q == 0, 1.0, np.asarray(s, float).reshape(q.shape))
        return pcs.reshape(q.shape) * L * L * s

    return intensity


ff_worm_ps1_sd_ellcylsh = _worm_sd(worm_ps1, exvol_index=6)
ff_worm_ps2_sd_ellcylsh = _worm_sd(worm_ps2)
ff_worm_ps3_sd_ellcylsh = _worm_sd(lambda q, b, L: worm_ps3(q, b, L, 0.0))  # no excluded volume, as in C
ff_worm_kholodenko_sd_ellcylsh = _worm_sd(kholodenko, kholodenko_layout=True)


# ------------------------------------------------------------------------------------------------ batch 8: thin shells
def _ell_surface(R, eps):
    """Surface of a spheroid with semi-axes (R, R, εR)."""
    R = np.asarray(R, float)
    if eps < 1.0:
        ae = math.acos(eps)
        return 2.0 * math.pi * R * R * (1.0 + eps * math.atanh(math.sin(ae)) / math.sin(ae))
    if eps > 1.0:
        ae = math.acos(1.0 / eps)
        return 2.0 * math.pi * R * R * (1.0 + eps * ae / math.tan(ae))
    return 4.0 * math.pi * R * R


def thin_ellipsoidal_shell_pprime(q, R, eps, sigma, power=2):
    """P'(q) of an infinitely thin spheroidal shell (semi-axes R, R, εR), log-normal R (median R, width σ):
    <(S j0(q R √(sin²α + ε² cos²α)))^power> over α (weight sin α on [0, π/2]) and R. P'(0) = <S²>."""
    q = np.atleast_1d(np.asarray(q, float))
    sigma = abs(sigma)
    S0 = float(_ell_surface(1.0, eps))
    out = np.empty(q.shape)
    for i, qi in enumerate(q):
        nu, wn = _lognorm_axis(sigma, qi * R * max(1.0, eps), 2.0 * power * sigma ** 2)
        if eps != 1.0:
            nmax = float(nu.max())
            pan = max(2, math.ceil(qi * R * nmax * abs(eps - 1.0) / 2.0))
            ed = np.linspace(0.0, math.pi / 2.0, pan + 1)
            al, wa = gauss_legendre(_GN, ed[:-1], ed[1:])
            al, wa = al.ravel(), wa.ravel() * np.sin(al.ravel())
        else:
            al, wa = np.array([math.pi / 2.0]), np.ones(1)
        g = np.sqrt(np.sin(al) ** 2 + (eps * np.cos(al)) ** 2)[:, None]
        r = R * nu[None, :]
        F = S0 * r * r * _j0(qi * r * g)
        out[i] = wa @ (F ** power) @ wn
    return out


def thin_hollow_cylinder_pprime(q, R, H, sig_R, sig_H):
    """P'(q) of an infinitely thin closed cylindrical shell (radius R, height H, two end discs), log-normal R and H:
    <(2π R² Λ(qR sin α) cos(qH cos α/2) + 2π R H J0(qR sin α) j0(qH cos α/2))²> over α (weight sin α) and R, H."""
    q = np.atleast_1d(np.asarray(q, float))
    sig_R, sig_H = abs(sig_R), abs(sig_H)
    out = np.empty(q.shape)
    for i, qi in enumerate(q):
        n1, w1 = _lognorm_axis(sig_R, qi * R, 4.0 * sig_R ** 2)
        n2, w2 = _lognorm_axis(sig_H, 0.5 * qi * H, 2.0 * sig_H ** 2)
        size = max(R * float(n1.max()), 0.5 * H * float(n2.max()))
        pan = max(2, math.ceil(qi * size / 2.0))
        ed = np.linspace(0.0, math.pi / 2.0, pan + 1)
        al, wa = gauss_legendre(_GN, ed[:-1], ed[1:])
        al, wa = al.ravel(), wa.ravel() * np.sin(al.ravel())
        sa, ca = np.sin(al)[:, None], np.cos(al)[:, None]
        r = R * n1[None, :]
        qr = qi * r * sa
        lam, j0r = _lam(qr), special.j0(qr)
        acc = 0.0
        for k in range(n2.size):
            h = H * n2[k]
            qh = 0.5 * qi * h * ca
            F = 2.0 * math.pi * r * r * lam * np.cos(qh) + 2.0 * math.pi * r * h * j0r * _j0(qh)
            acc += w2[k] * (wa @ (F * F) @ w1)
        out[i] = acc
    return out


def ff_pcs_homogeneousplate_f(q, p):
    """Amplitude Δη <t j0(qt/2)>."""
    sig = abs(p[1]) if abs(p[1]) > 1e-6 else 0.0
    f = lambda qi, x: x * _j0(0.5 * qi * x)  # noqa: E731
    return ((p[4] - p[5]) * lognorm_average(q, f, p[0], sig, 1.0)).reshape(np.shape(q))


def ff_pcs_bilayergauss_f(q, p):
    """Amplitude <√(2π) σ_c b_c e^{-q²σ_c²/2} + 2 √(2π) σ_o b_o e^{-q²σ_o²/2} cos(q D ν/2)> (C stub until 2026-11-24)."""
    sc, bc, so, bo, D, sD = abs(p[0]), p[1], abs(p[2]), p[3], p[4], abs(p[5])
    qa = np.atleast_1d(np.asarray(q, float))
    out = np.empty(qa.shape)
    for i, qi in enumerate(qa):
        fc = math.sqrt(2.0 * math.pi) * sc * bc * math.exp(-0.5 * (qi * sc) ** 2)
        fo = math.sqrt(2.0 * math.pi) * so * bo * math.exp(-0.5 * (qi * so) ** 2)
        nu, w = _lognorm_axis(sD, 0.5 * qi * abs(D), 0.0)
        out[i] = np.sum(w * (fc + 2.0 * fo * np.cos(0.5 * qi * D * nu)))
    return out.reshape(np.shape(q))


def _ellsh_check(p, name):
    if p[0] < 0 or p[2] < 0 or p[4] <= 0:
        raise ValueError(f"{name}: t >= 0, R0 >= 0 and epsilon > 0 required")


def ff_ellsh_sd_homoxs(q, p):
    """p = [t, sigma_t, R0, sigma_R0, epsilon, dummy, eta_l, eta_sol]: Pcs(plate) · P'(thin spheroidal shell)."""
    _ellsh_check(p, "ellsh_sd_homoxs")
    pcs = ff_pcs_homogeneousplate(q, np.array([p[0], p[1], 0.0, 0.0, p[6], p[7]]))
    return np.asarray(pcs).reshape(np.shape(q)) * thin_ellipsoidal_shell_pprime(q, p[2], p[4], p[3]).reshape(np.shape(q))


def ff_ellsh_sd_homoxs_f(q, p):
    pcs = ff_pcs_homogeneousplate_f(q, np.array([p[0], p[1], 0.0, 0.0, p[6], p[7]]))
    return np.asarray(pcs).reshape(np.shape(q)) * thin_ellipsoidal_shell_pprime(q, p[2], p[4], p[3], 1).reshape(np.shape(q))


def _ellsh_bl_pcs(p, f):
    return np.array([p[8], p[9], p[6], p[7], p[0], p[1]])


def ff_ellsh_sd_bilayergauss(q, p):
    """p = [t, sigma_t, R0, sigma_R0, epsilon, dummy, sigma_out, b_out, sigma_core, b_core]."""
    _ellsh_check(p, "ellsh_sd_bilayergauss")
    pcs = ff_pcs_bilayergauss(q, _ellsh_bl_pcs(p, None))
    return np.asarray(pcs).reshape(np.shape(q)) * thin_ellipsoidal_shell_pprime(q, p[2], p[4], p[3]).reshape(np.shape(q))


def ff_ellsh_sd_bilayergauss_f(q, p):
    pcs = ff_pcs_bilayergauss_f(q, _ellsh_bl_pcs(p, None))
    return np.asarray(pcs).reshape(np.shape(q)) * thin_ellipsoidal_shell_pprime(q, p[2], p[4], p[3], 1).reshape(np.shape(q))


def ellsh_volume(x, p, dist=0):
    """Mean shell volume <S(R)> t = S(R0, ε) exp(2 σ_R0²) t (dist replaces p[dist] by x)."""
    x = np.asarray(x, float)
    v = [np.where(dist == k, x, p[k]) for k in range(5)]
    return float(_ell_surface(1.0, float(np.asarray(v[4]).ravel()[0]))) * v[2] ** 2 * np.exp(2.0 * v[3] ** 2) * np.abs(v[0])


def ff_cylsh_sd_homoxs(q, p):
    """p = [t, sigma_t, R, sigma_R, H, sigma_H, eta_l, eta_sol]: Pcs(plate) · P'(thin closed cylindrical shell)."""
    if p[0] < 0 or p[2] < 0 or p[4] < 0:
        raise ValueError("cylsh_sd_homoxs: t, R, H >= 0 required")
    pcs = ff_pcs_homogeneousplate(q, np.array([p[0], p[1], 0.0, 0.0, p[6], p[7]]))
    return np.asarray(pcs).reshape(np.shape(q)) * thin_hollow_cylinder_pprime(q, p[2], p[4], p[3], p[5]).reshape(np.shape(q))


def ff_cylsh_sd_bilayergauss(q, p):
    """p = [t, sigma_t, R0, sigma_R0, H0, sigma_H0, sigma_out, b_out, sigma_core, b_core]."""
    if p[0] < 0 or p[2] < 0 or p[4] < 0:
        raise ValueError("cylsh_sd_bilayergauss: t, R0, H0 >= 0 required")
    pcs = ff_pcs_bilayergauss(q, np.array([p[8], p[9], p[6], p[7], p[0], p[1]]))
    return np.asarray(pcs).reshape(np.shape(q)) * thin_hollow_cylinder_pprime(q, p[2], p[4], p[3], p[5]).reshape(np.shape(q))


def _red(q, p):
    return np.asarray(q, float) * (abs(p[0]) * max(abs(p[2]), 1.0) + abs(p[3]))


_SRC = "plugins/thin_objects/sasfit_ff_"
_GROUP = "ff_thin_objects"
# sizes sampled without distribution widths for the C comparison: the C code integrates log-normal distributions
# only over SASfit's find_LogNorm_int_range (3.7 σ above the peak of x^(2 dim) LN), which drops up to ~1e-4 of the
# weight; distributions are tested separately against scipy and loosely against C
_CS = {0: (2.0, 20.0, "log"), 2: (0.4, 2.5, "log"), 3: (0.5, 5.0, "log")}
_ETA3 = {7: (-1.0, 3.0, "lin"), 8: (-1.0, 3.0, "lin"), 9: (-1.0, 1.0, "lin")}
_ETA3B = {8: (-1.0, 3.0, "lin"), 9: (-1.0, 3.0, "lin"), 10: (-1.0, 1.0, "lin")}
_PCS = ["R0", "sigma_R0", "epsilon", "t0", "sigma_t0", "dummy", "dummy", "eta_core", "eta_shell", "eta_solv"]
_ROD = ["R0", "sigma_R0", "epsilon", "t0", "sigma_t0", "L0", "sigma_L0", "dummy", "eta_core", "eta_shell",
        "eta_solv"]  # fmt: skip
_W13 = ["R", "sigma_R", "epsilon", "t", "lb", "L", "exvol", "eta_core", "eta_shell", "eta_solv"]
_W2 = ["R", "sigma_R", "epsilon", "t", "lb", "L", "dummy", "eta_core", "eta_shell", "eta_solv"]
_WK = ["R", "sigma_R", "epsilon", "t0", "sigma_t0", "lb", "L", "dummy", "eta_core", "eta_shell", "eta_solv"]
_SW = {0: (2.0, 20.0, "log"), 2: (0.4, 2.5, "log"), 3: (0.5, 5.0, "log"),
       4: (5.0, 60.0, "log"), 5: (50.0, 2000.0, "log"), **_ETA3}  # fmt: skip

SPECS = [
    make_spec("ff_pcs_ellcylsh", ff_pcs_ellcylsh, _PCS, {**_CS, **_ETA3}, source=_SRC + "pcs_ellcylsh.c",
              reduced=_red, reduced_label="q (R+t)", rtol=1e-8, c_integrates=True, group=_GROUP),
    make_spec("ff_rod_sd_ellcylsh", ff_rod_sd_ellcylsh, _ROD,
              {**_CS, 5: (20.0, 500.0, "log"), **_ETA3B}, source=_SRC + "rod_sd_ellcylsh.c",
              reduced=_red, reduced_label="q (R+t)", rtol=1e-8, c_integrates=True, group=_GROUP),
    make_spec("ff_worm_ps1_sd_ellcylsh", ff_worm_ps1_sd_ellcylsh, _W13, {**_SW, 6: (0.0, 1.0, "int")},
              source=_SRC + "worm_ps1_sd_ellcylsh.c", reduced=_red, reduced_label="q (R+t)", rtol=1e-8,
              c_integrates=True, group=_GROUP),
    make_spec("ff_worm_ps2_sd_ellcylsh", ff_worm_ps2_sd_ellcylsh, _W2, _SW, source=_SRC + "worm_ps2_sd_ellcylsh.c",
              reduced=_red, reduced_label="q (R+t)", rtol=1e-8, c_integrates=True, group=_GROUP),
    make_spec("ff_worm_ps3_sd_ellcylsh", ff_worm_ps3_sd_ellcylsh, _W2, _SW, source=_SRC + "worm_ps3_sd_ellcylsh.c",
              reduced=_red, reduced_label="q (R+t)", rtol=1e-8, c_integrates=True, group=_GROUP),
    make_spec("ff_worm_kholodenko_sd_ellcylsh", ff_worm_kholodenko_sd_ellcylsh, _WK,
              {**_CS, 5: (5.0, 60.0, "log"), 6: (50.0, 2000.0, "log"), **_ETA3B},
              source=_SRC + "worm_kholodenko_sd_ellcylsh.c", reduced=_red, reduced_label="q (R+t)", rtol=1e-8,
              c_integrates=True, group=_GROUP),
    make_spec("ff_pcs_homogeneouscyl", ff_pcs_homogeneouscyl,
              ["R", "sigma_R", "dummy", "dummy", "dummy", "dummy", "dummy", "eta_core", "dummy", "eta_sol"],
              {0: (1.0, 30.0, "log"), 7: (-1.0, 3.0, "lin"), 9: (-1.0, 1.0, "lin")},
              source=_SRC + "pcs_homogeneouscyl.c", reduced=lambda q, p: np.asarray(q, float) * abs(p[0]),
              reduced_label="q R", rtol=1e-10, group=_GROUP),
    make_spec("ff_pcs_homogeneousplate", ff_pcs_homogeneousplate,
              ["t", "sigma_t", "dummy", "dummy", "eta_L", "eta_solv"],
              {0: (1.0, 30.0, "log"), 4: (-1.0, 3.0, "lin"), 5: (-1.0, 1.0, "lin")},
              amplitude=ff_pcs_homogeneousplate_f, source=_SRC + "pcs_homogeneousplate.c", reduced=lambda q, p: np.asarray(q, float) * abs(p[0]),
              reduced_label="q t", rtol=1e-10, group=_GROUP),
    make_spec("ff_pcs_twoinfinitelythinlayers", ff_pcs_twoinfinitelythinlayers, ["t", "sigma_t"],
              {0: (1.0, 30.0, "log")}, source=_SRC + "pcs_twoinfinitelythinlayers.c",
              reduced=lambda q, p: np.asarray(q, float) * abs(p[0]), reduced_label="q t", rtol=1e-10, group=_GROUP),
    make_spec("ff_pcs_bilayergauss", ff_pcs_bilayergauss,
              ["sigma_core", "b_core", "sigma_out", "b_out", "D", "sigma_D"],
              {0: (0.2, 2.0, "log"), 1: (-1.0, 1.0, "lin"), 2: (0.2, 2.0, "log"), 3: (-1.0, 1.0, "lin"),
               4: (2.0, 8.0, "log")}, amplitude=ff_pcs_bilayergauss_f, source=_SRC + "pcs_bilayergaus.c",
              reduced=lambda q, p: np.asarray(q, float) * abs(p[4]), reduced_label="q D", rtol=1e-10, group=_GROUP),
    make_spec("ff_pcs_layeredcentrosymmetricxs", ff_pcs_layeredcentrosymmetricxs,
              ["Lc", "sigma_Lc", "Lsh", "sigma_Lsh", "eta_Lc", "eta_Lsh", "eta_solv"],
              {0: (1.0, 20.0, "log"), 2: (0.5, 10.0, "log"), 4: (-1.0, 3.0, "lin"), 5: (-1.0, 3.0, "lin"),
               6: (-1.0, 1.0, "lin")}, source=_SRC + "pcs_layeredcentrosymmetricxs.c",
              reduced=lambda q, p: np.asarray(q, float) * (abs(p[0]) + 2 * abs(p[2])), reduced_label="q (Lc+2Lsh)",
              rtol=1e-10, c_integrates=True, group=_GROUP),
    make_spec("ff_pcs_plate_chains_rw_", ff_pcs_plate_chains_rw_,
              ["L_core", "n_agg", "V_brush", "eta_core", "eta_brush", "eta_solv", "xsolv_core", "Rg", "d"],
              {0: (1.0, 20.0, "log"), 1: (0.01, 0.5, "log"), 2: (10.0, 500.0, "log"), 3: (-1.0, 3.0, "lin"),
               4: (-1.0, 3.0, "lin"), 5: (-1.0, 1.0, "lin"), 6: (0.0, 0.5, "lin"), 7: (1.0, 10.0, "log"),
               8: (0.0, 2.0, "lin")}, source=_SRC + "pcs_plate_chains_rw_.c",
              reduced=lambda q, p: np.asarray(q, float) * abs(p[0]), reduced_label="q L_core", rtol=1e-10,
              group=_GROUP),
    make_spec("ff_disc_sd_homoxs", ff_disc_sd_homoxs,
              ["t", "sigma_t", "R0", "sigma_R0", "dummy", "dummy", "eta_l", "eta_sol"],
              {0: (1.0, 10.0, "log"), 2: (5.0, 200.0, "log"), 6: (-1.0, 3.0, "lin"), 7: (-1.0, 1.0, "lin")},
              source=_SRC + "disc_sd_homoxs.c", reduced=lambda q, p: np.asarray(q, float) * abs(p[2]),
              reduced_label="q R0", rtol=1e-10, group=_GROUP),
    make_spec("ff_disc_sd_bilayergauss", ff_disc_sd_bilayergauss,
              ["t", "sigma_t", "R0", "sigma_R0", "dummy", "dummy", "sigma_out", "b_out", "sigma_core", "b_core"],
              {0: (2.0, 8.0, "log"), 2: (5.0, 200.0, "log"), 6: (0.2, 2.0, "log"), 7: (-1.0, 1.0, "lin"),
               8: (0.2, 2.0, "log"), 9: (-1.0, 1.0, "lin")},
              source=_SRC + "disc_sd_bilayergauss.c", reduced=lambda q, p: np.asarray(q, float) * abs(p[2]),
              reduced_label="q R0", rtol=1e-10, group=_GROUP),

    # batch 8 (2026-11-24): thin spheroidal / closed cylindrical shells
    make_spec("ff_ellsh_sd_homoxs", ff_ellsh_sd_homoxs,
              ["t", "sigma_t", "R0", "sigma_R0", "epsilon", "dummy", "eta_l", "eta_sol"],
              {0: (1.0, 8.0, "log"), 2: (10.0, 200.0, "log"), 4: (0.3, 3.0, "log"), 6: (-1.0, 3.0, "lin"),
               7: (-1.0, 1.0, "lin")}, amplitude=ff_ellsh_sd_homoxs_f, volume=ellsh_volume,
              source=_SRC + "ellsh_sd_homoxs.c", reduced=lambda q, p: np.asarray(q, float) * abs(p[2]),
              reduced_label="q R0", rtol=1e-9, c_integrates=True, group=_GROUP),
    make_spec("ff_ellsh_sd_bilayergauss", ff_ellsh_sd_bilayergauss,
              ["t", "sigma_t", "R0", "sigma_R0", "epsilon", "dummy", "sigma_out", "b_out", "sigma_core", "b_core"],
              {0: (2.0, 8.0, "log"), 2: (10.0, 200.0, "log"), 4: (0.3, 3.0, "log"), 6: (0.2, 2.0, "log"),
               7: (-1.0, 1.0, "lin"), 8: (0.2, 2.0, "log"), 9: (-1.0, 1.0, "lin")},
              amplitude=ff_ellsh_sd_bilayergauss_f, volume=ellsh_volume, source=_SRC + "ellsh_sd_bilayergauss.c",
              reduced=lambda q, p: np.asarray(q, float) * abs(p[2]), reduced_label="q R0", rtol=1e-9,
              c_integrates=True, group=_GROUP),
    make_spec("ff_cylsh_sd_homoxs", ff_cylsh_sd_homoxs,
              ["t", "sigma_t", "R", "sigma_R", "H", "sigma_H", "eta_l", "eta_sol"],
              {0: (1.0, 8.0, "log"), 2: (5.0, 100.0, "log"), 4: (10.0, 300.0, "log"), 6: (-1.0, 3.0, "lin"),
               7: (-1.0, 1.0, "lin")}, source=_SRC + "cylsh_sd_homoxs.c",
              reduced=lambda q, p: np.asarray(q, float) * max(abs(p[2]), abs(p[4])), reduced_label="q max(R, H)",
              rtol=1e-9, c_integrates=True, group=_GROUP),
    make_spec("ff_cylsh_sd_bilayergauss", ff_cylsh_sd_bilayergauss,
              ["t", "sigma_t", "R0", "sigma_R0", "H0", "sigma_H0", "sigma_out", "b_out", "sigma_core", "b_core"],
              {0: (2.0, 8.0, "log"), 2: (5.0, 100.0, "log"), 4: (10.0, 300.0, "log"), 6: (0.2, 2.0, "log"),
               7: (-1.0, 1.0, "lin"), 8: (0.2, 2.0, "log"), 9: (-1.0, 1.0, "lin")},
              source=_SRC + "cylsh_sd_bilayergauss.c",
              reduced=lambda q, p: np.asarray(q, float) * max(abs(p[2]), abs(p[4])), reduced_label="q max(R0, H0)",
              rtol=1e-9, c_integrates=True, group=_GROUP),
]  # fmt: skip
