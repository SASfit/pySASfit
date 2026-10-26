# SPDX-License-Identifier: BSD-3-Clause
"""Water measured at two detector distances.

A sample normalised by water measured at the same distance needs no solid-angle correction (it
cancels pixel by pixel), but it inherits the angular dependence K(λ0, 2θ) of the water scattering,
which in a flat cell is mostly multiple incoherent scattering (Strunz et al., J. Appl. Cryst. 33
(2000) 829, section 7 and fig. 4: more scattering along the cell normal).

The same pixel ij sees 2θ_a at distance L_a and 2θ_b at L_b. After the solid-angle correction the
ratio of the two water images is free of the pixel efficiency η_ij:

    I_a(ij) / I_b(ij) = C · K(2θ_a,ij) / K(2θ_b,ij)

so a fit of ln(I_a/I_b) = c + Σ_k a_k [(2θ_a)^{2k} - (2θ_b)^{2k}] gives K(2θ)/K(0) = exp(Σ a_k (2θ)^{2k})
and the constant c = ln C. In principle C = 1. A constant different from 1 means something that
does not depend on the angle differs between the two distances, e.g.

* an effective distance offset δ common to both (reference point of the distance, conversion depth
  in the detector): ΔΩ ∝ 1/(L + δ)² gives C = [(L_b + δ)/L_b]² / [(L_a + δ)/L_a]²;
* a dead-time correction that is too small at the higher count rate (shorter distance);
* a primary beam that is not the same per monitor count (collimation, attenuator, monitor drift).

The two first can be told apart by a third distance or by the same distance at another count rate
(attenuator in).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

__all__ = ["WaterComparison", "compare_water", "compare_water_runs", "distance_offset", "deadtime_for_unit_scale",
           "water_image", "DistanceFit", "fit_distance_offset", "fit_offset_images"]


@dataclass
class WaterComparison:
    L_a: float
    L_b: float
    c: float                       # ln of the angle-independent ratio
    c_sigma: float
    coeffs: np.ndarray             # a_k for (2θ)^{2k}, 2θ in rad, k = 1..n
    coeffs_sigma: np.ndarray
    chi2_ndf: float
    n_pixels: int
    two_theta_max_deg: float       # largest 2θ seen (at the shorter distance)
    notes: List[str] = field(default_factory=list)
    rate_a: Optional[float] = None  # detector count rates of the water runs [1/s]
    rate_b: Optional[float] = None
    tau_s: Optional[float] = None   # dead time used

    @property
    def deadtime_tau_s(self) -> Optional[float]:
        """Dead time (non-paralysable, global rate) that would explain C alone."""
        if None in (self.rate_a, self.rate_b, self.tau_s):
            return None
        return deadtime_for_unit_scale(self.scale, self.rate_a, self.rate_b, self.tau_s)

    @property
    def scale(self) -> float:
        """C = I_a / I_b at 2θ → 0."""
        return float(np.exp(self.c))

    def K(self, two_theta_rad) -> np.ndarray:
        """K(2θ)/K(0) of the water."""
        x = np.asarray(two_theta_rad, float)
        return np.exp(sum(a * x**(2 * (k + 1)) for k, a in enumerate(self.coeffs)))

    def K_sigma(self, two_theta_rad) -> np.ndarray:
        x = np.asarray(two_theta_rad, float)
        cov = getattr(self, "_cov", np.diag(self.coeffs_sigma**2))
        J = np.stack([x**(2 * (k + 1)) for k in range(len(self.coeffs))], axis=-1)
        return self.K(x) * np.sqrt(np.einsum("...i,ij,...j->...", J, cov, J))

    @property
    def distance_offset_m(self) -> float:
        """Common distance offset δ that would explain C alone."""
        return distance_offset(self.scale, self.L_a, self.L_b)

    def summary(self) -> str:
        deg = np.radians([2, 5, 10, 15])
        ks = ", ".join(f"{d:g}°: {k:.4f}" for d, k in zip([2, 5, 10, 15], self.K(deg)) if np.radians(d) <= np.radians(self.two_theta_max_deg) + 1e-9)
        tau = self.deadtime_tau_s
        alt = "" if tau is None else f", or a dead time of {1e6 * tau:.2f} µs instead of {1e6 * self.tau_s:.2f} µs"
        return (f"water at {self.L_a:.2f} m / {self.L_b:.2f} m: C = {self.scale:.4f} ± {self.scale * self.c_sigma:.4f} "
                f"(a common distance offset of {100 * self.distance_offset_m:+.1f} cm would explain it{alt}); "
                f"K(2θ)/K(0) = {ks}; χ²/ndf = {self.chi2_ndf:.2f} over {self.n_pixels} pixels")


def distance_offset(C: float, L_a: float, L_b: float) -> float:
    """δ with [(L_b + δ)/L_b]² / [(L_a + δ)/L_a]² = C (solved by bisection)."""
    f = lambda d: ((L_b + d) / L_b)**2 / ((L_a + d) / L_a)**2 - C
    lo, hi = -0.9 * min(L_a, L_b), 10.0
    if f(lo) * f(hi) > 0:
        return float("nan")
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if f(lo) * f(mid) <= 0:
            hi = mid
        else:
            lo = mid
    return 0.5 * (lo + hi)


def deadtime_for_unit_scale(C: float, rate_a: float, rate_b: float, tau0: float) -> float:
    """τ for which the ratio C (obtained with τ0) becomes 1: each image scales with
    (1 - R τ0)/(1 - R τ) (non-paralysable dead time on the measured global rate R)."""
    def f(t):
        return C * ((1 - rate_a * tau0) / (1 - rate_a * t)) / ((1 - rate_b * tau0) / (1 - rate_b * t)) - 1
    hi = 0.999 / max(rate_a, rate_b)
    lo = 0.0
    if f(lo) * f(hi) > 0:
        return float("nan")
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if f(lo) * f(mid) <= 0:
            hi = mid
        else:
            lo = mid
    return 0.5 * (lo + hi)


def compare_water(I_a, s_a, tt_a, I_b, s_b, tt_b, valid, L_a: float, L_b: float, order: int = 1,
                  reject_sigma: Optional[float] = 5.0) -> WaterComparison:
    """Fit ln(I_a/I_b) = c + Σ_{k≤order} a_k [(2θ_a)^{2k} - (2θ_b)^{2k}] pixel by pixel.

    ``I``, ``s``: solid-angle corrected water images (empty cell subtracted, transmission
    corrected) and their statistical σ; ``tt``: 2θ of every pixel [rad] at each distance;
    ``valid``: pixels unmasked at both distances. Pixels more than ``reject_sigma`` from the fit
    are left out once (hot pixels, beam-stop edges)."""
    I_a, I_b, s_a, s_b = (np.asarray(v, float) for v in (I_a, I_b, s_a, s_b))
    ok = np.asarray(valid, bool) & (I_a > 0) & (I_b > 0) & (s_a > 0) & (s_b > 0)
    y = np.log(I_a[ok] / I_b[ok])
    sy = np.hypot(s_a[ok] / I_a[ok], s_b[ok] / I_b[ok])
    xa, xb = np.asarray(tt_a, float)[ok], np.asarray(tt_b, float)[ok]
    A = np.column_stack([np.ones_like(y)] + [xa**(2 * k) - xb**(2 * k) for k in range(1, order + 1)])
    keep = np.ones(y.size, bool)
    notes = []
    for it in range(2):
        w = 1 / sy[keep]
        Aw = A[keep] * w[:, None]
        p, *_ = np.linalg.lstsq(Aw, y[keep] * w, rcond=None)
        res = (y - A @ p) / sy
        if reject_sigma is None or it == 1:
            break
        bad = np.abs(res) > reject_sigma
        if bad.any():
            notes.append(f"{int(bad.sum())} pixels beyond {reject_sigma:g} σ left out")
        keep = ~bad
    cov = np.linalg.inv(Aw.T @ Aw)
    ndf = max(int(keep.sum()) - len(p), 1)
    chi2 = float(np.sum(res[keep]**2) / ndf)
    cov = cov * max(chi2, 1.0)                  # scale the errors when χ²/ndf > 1
    out = WaterComparison(L_a, L_b, float(p[0]), float(np.sqrt(cov[0, 0])), p[1:].copy(),
                          np.sqrt(np.diag(cov))[1:], chi2, int(keep.sum()),
                          float(np.degrees(max(xa.max(), xb.max()))), notes)
    out._cov = cov[1:, 1:]
    return out


def water_image(waters, empties, blocked, centre, t_water, t_empty, instrument=None, dome: bool = False,
                L2_m: Optional[float] = None):
    """Solid-angle corrected water image of summed runs: returns (I, σ, mask, geometry, L2, rate).
    ``L2_m``: distance for the geometry (default: the effective distance of the first water run)."""
    from ..reduce import _as_list, _transmission, combined_normalised_counts, load_instrument
    from .efficiency import water_intensity
    from .geometry import pixel_geometry
    from . import mask as mk
    inst = instrument or load_instrument()
    tw, te = _transmission(t_water, "transmission:water"), _transmission(t_empty, "transmission:empty_cell")
    ws, ecs, bls = _as_list(waters), _as_list(empties), _as_list(blocked)
    notes = []
    m, _ = mk.suggest_mask(np.sum([w.counts for w in ws], axis=0), centre)
    W, m = combined_normalised_counts(ws, inst, notes, m)
    E = B = None
    if ecs:
        E, m = combined_normalised_counts(ecs, inst, notes, m)
    if bls:
        B, m = combined_normalised_counts(bls, inst, notes, m)
    L2 = float(getattr(ws[0], "L2_effective_m", ws[0].L2_m)) if L2_m is None else float(L2_m)
    g = pixel_geometry(ws[0].counts.shape, ws[0].pixel_size_m, centre, L2, ws[0].wavelength_A)
    I = water_intensity(W, E, tw, te if E is not None else 1.0, g.d_omega, B)
    if dome:
        from .detector_model import window_transmission
        con = inst["detector"].get("construction", {})
        c, mem = con.get("window", {}), con.get("membrane", {})
        T = window_transmission(ws[0].wavelength_A, g.x, g.y, L2, c.get("thickness_m", 0.007),
                                mem.get("thickness_m", 0.002) if mem.get("material") == "Al" else 0.0,
                                c.get("radius_of_curvature_m") or 0.93, c.get("apex_to_detector_m"))
        I = I / T
    rate = float(np.sum([w.counts.sum() for w in ws]) / np.sum([w.counting_time_s for w in ws]))
    return I.value, I.std(systematic=False), m, g, L2, rate


def compare_water_runs(water_a, empty_a, water_b, empty_b, beam_center_px, t_water, t_empty,
                       blocked_a=None, blocked_b=None, instrument=None, order: int = 1,
                       beam_center_b_px=None, dome: bool = False) -> WaterComparison:
    """:func:`compare_water` from raw runs (lists are summed): every run corrected and normalised
    as in the reduction, the empty cell subtracted with the transmissions, divided by the exact
    pixel solid angle; masks suggested from the summed water counts. With ``dome`` each image is first
    divided by the transmission of the curved Al dome window and the flat Al membrane over the
    detector (it differs between the distances for the same pixel), so that K holds the water only."""
    from ..reduce import load_instrument
    inst = instrument or load_instrument()

    def one(ws, ecs, bls, centre):
        return water_image(ws, ecs, bls, centre, t_water, t_empty, inst, dome)
    Ia, sa, ma, ga, La, Ra = one(water_a, empty_a, blocked_a, beam_center_px)
    Ib, sb, mb, gb, Lb, Rb = one(water_b, empty_b, blocked_b, beam_center_b_px or beam_center_px)
    out = compare_water(Ia, sa, ga.two_theta, Ib, sb, gb.two_theta, (ma == 0) & (mb == 0), La, Lb, order)
    out.rate_a, out.rate_b, out.tau_s = Ra, Rb, float(inst["detector"]["deadtime"]["tau_s"])
    return out


@dataclass
class DistanceFit:
    """Offset of the detection layer from the recorded distances, from water at several distances."""
    delta_m: float
    sigma_m: float
    a: float                       # ln K = a (2θ)²
    constant: Optional[float]      # common ln factor (free_constant) or None
    chi2_ndf: float
    distances_m: List[float]
    scan: tuple                    # (δ values, χ²)

    def summary(self) -> str:
        return (f"distances {', '.join(f'{L:.2f}' for L in self.distances_m)} m: offset of the detection layer "
                f"δ = {1000 * self.delta_m:+.1f} ± {1000 * self.sigma_m:.1f} mm, K(15°)/K(0) = "
                f"{np.exp(self.a * np.radians(15) ** 2):.4f}, χ²/ndf = {self.chi2_ndf:.2f}")


def fit_distance_offset(sets, t_water, t_empty, instrument=None, dome: bool = True,
                        free_constant: bool = False, deltas=np.linspace(-0.05, 0.20, 251)) -> DistanceFit:
    """Water (with its empty cell) measured at several detector distances with the same collimation:
    the offset δ of the detection layer from the recorded distances.

    ``sets``: list of dicts with ``water``, ``empty`` (runs), optional ``blocked`` and ``centre``
    (beam centre of that geometry). The images are computed with the recorded distances; pixel by
    pixel against the first set (the efficiency cancels)

        ln(I_k/I_0) = -2 ln((L_k+δ)/L_k) + 2 ln((L_0+δ)/L_0) + a [(2θ_k)² - (2θ_0)²] [+ c]

    χ²(δ) is minimised with a (and c with ``free_constant``, which needs three or more distances)
    fitted linearly for each δ; σ(δ) from Δχ² = χ²/ndf."""
    from ..reduce import load_instrument
    inst = instrument or load_instrument()
    if len(sets) < 2 or (free_constant and len(sets) < 3):
        raise ValueError("two distances needed (three with free_constant)")
    imgs = []
    for st in sets:
        ws = st["water"] if isinstance(st["water"], (list, tuple)) else [st["water"]]
        imgs.append(water_image(ws, st.get("empty"), st.get("blocked"), st["centre"], t_water, t_empty, inst, dome,
                                L2_m=ws[0].L2_m))
    return fit_offset_images([(I, s, m, g.two_theta, L) for I, s, m, g, L, _ in imgs], free_constant, deltas)


def fit_offset_images(imgs, free_constant: bool = False, deltas=np.linspace(-0.05, 0.20, 251)) -> DistanceFit:
    """The fit of :func:`fit_distance_offset` on images: list of (I, σ, mask, 2θ, L) per distance,
    computed with the recorded distances L."""
    I0, s0, m0, t0, L0 = imgs[0]
    ys, ss, dx, Ls, ks = [], [], [], [], []
    for k, (I, s, m, tt, L) in enumerate(imgs[1:], start=1):
        ok = (m == 0) & (m0 == 0) & (I > 0) & (I0 > 0) & (s > 0) & (s0 > 0)
        ys.append(np.log(I[ok] / I0[ok]))
        ss.append(np.hypot(s[ok] / I[ok], s0[ok] / I0[ok]))
        dx.append(tt[ok] ** 2 - t0[ok] ** 2)
        Ls.append(np.full(ok.sum(), L))
        ks.append(np.full(ok.sum(), k))
    y, sy, x2, Lk = (np.concatenate(v) for v in (ys, ss, dx, Ls))
    w = 1 / sy

    def solve(d):
        h = -2 * np.log((Lk + d) / Lk) + 2 * np.log((L0 + d) / L0)
        cols = [x2] + ([np.ones_like(y)] if free_constant else [])
        A = np.column_stack(cols)
        p, *_ = np.linalg.lstsq(A * w[:, None], (y - h) * w, rcond=None)
        r = (y - h - A @ p) * w
        return float(r @ r), p
    chi = np.array([solve(d)[0] for d in deltas])
    i = int(np.argmin(chi))
    lo, hi = deltas[max(i - 1, 0)], deltas[min(i + 1, len(deltas) - 1)]
    from scipy.optimize import minimize_scalar
    res = minimize_scalar(lambda d: solve(d)[0], bounds=(lo, hi), method="bounded")
    d = float(res.x)
    c2, p = solve(d)
    ndf = max(y.size - (2 + free_constant), 1)
    red = c2 / ndf
    hstep = 1e-3
    curv = (solve(d + hstep)[0] - 2 * c2 + solve(d - hstep)[0]) / hstep**2
    sig = float(np.sqrt(2 * max(red, 1.0) / curv)) if curv > 0 else float("nan")
    return DistanceFit(d, sig, float(p[0]), float(p[1]) if free_constant else None, red,
                       [L0] + [float(Ls[k][0]) for k in range(len(Ls))], (deltas, chi))
