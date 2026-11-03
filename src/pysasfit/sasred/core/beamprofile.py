# SPDX-License-Identifier: BSD-3-Clause
"""Direct-beam profile of a geometry: 2D Gaussian fit, beam divergence, and the geometric part of
the q resolution.

The attenuated direct beam (beam stop out, no sample: the empty-beam transmission runs) shows on the
detector what a point of the sample's scattering pattern is smeared with by the collimation (source
and sample apertures, their distance L1, the distance L2) and the detector's own resolution. Its
profile is fitted with a separable Gaussian integrated over every pixel plus a flat background,

    N_ij = A [Phi((x_j + 1/2 - x0)/s_x) - Phi((x_j - 1/2 - x0)/s_x)]
             [Phi((y_i + 1/2 - y0)/s_y) - Phi((y_i - 1/2 - y0)/s_y)] + B,

with Poisson weights on a window around the beam. Because the model is integrated over the pixel,
s_x and s_y are the widths of the beam itself, without the pixel size (which the averaging accounts
for separately). The angular widths seen from the sample are

    sigma_theta_x = s_x p_x / L2,   sigma_theta_y = s_y p_y / L2   (rad, standard deviations),

and they give the geometric resolution of the scattering vector q = 2k sin(theta) (k = 2 pi/lambda):
dq/d(2 theta) = k cos(theta), so in the direction phi on the detector

    sigma_q,geo^2 = k^2 cos^2(theta) (sigma_theta_x^2 cos^2 phi + sigma_theta_y^2 sin^2 phi).

The second moments of the beam (as in the centre of gravity) are kept for comparison: a beam that
is not Gaussian (box-shaped, with tails) shows as a difference between both widths.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
from scipy.optimize import least_squares
from scipy.special import ndtr

__all__ = ["BeamProfile", "fit_beam_profile", "beam_profiles"]


@dataclass
class BeamProfile:
    """Gaussian width of the direct beam of one geometry."""
    key: tuple                       # geometry (L2, lambda, L1)
    x0: float                        # fitted centre [pixel]
    y0: float
    sigma_x_px: float                # Gaussian sigma of the beam, pixel size removed [pixel]
    sigma_y_px: float
    sigma_x_err: float = np.nan
    sigma_y_err: float = np.nan
    pixel_size_m: tuple = (np.nan, np.nan)
    L2_m: float = np.nan
    wavelength_A: float = np.nan
    runs: List[int] = field(default_factory=list)
    chi2: float = np.nan             # reduced chi^2 of the fit (median over the runs)
    moment_x_px: float = np.nan      # sqrt(second moment - 1/12) of the beam, for comparison
    moment_y_px: float = np.nan
    source: str = "direct beam"      # "empty beam" | "direct beam (sample transmission runs)" | "by hand"
    notes: List[str] = field(default_factory=list)

    @property
    def sigma_x_mm(self) -> float:
        return 1e3 * self.sigma_x_px * float(self.pixel_size_m[0])

    @property
    def sigma_y_mm(self) -> float:
        return 1e3 * self.sigma_y_px * float(self.pixel_size_m[1])

    @property
    def divergence_rad(self) -> tuple:
        """(sigma_theta_x, sigma_theta_y): angular standard deviations of the beam seen from the sample."""
        return (self.sigma_x_mm * 1e-3 / self.L2_m, self.sigma_y_mm * 1e-3 / self.L2_m)

    def describe(self) -> str:
        tx, ty = self.divergence_rad
        from ..catalogue import _key_text
        s = (f"{_key_text(self.key)}: beam σx = {self.sigma_x_mm:.3f} mm, σy = {self.sigma_y_mm:.3f} mm "
             f"(divergence σθ = {1e3 * tx:.3f} × {1e3 * ty:.3f} mrad, FWHM {2.3548e3 * tx:.3f} × "
             f"{2.3548e3 * ty:.3f} mrad), centre ({self.x0:.2f}, {self.y0:.2f}) px; {self.source}"
             + (f", runs {', '.join(map(str, self.runs[:6]))}{'…' if len(self.runs) > 6 else ''}" if self.runs else "")
             + (f"; χ²_red {self.chi2:.3g}" if np.isfinite(self.chi2) else ""))
        return s + "".join(f"; {n}" for n in self.notes)

    def to_dict(self) -> dict:
        tx, ty = self.divergence_rad
        return {"key": list(self.key), "x0_px": self.x0, "y0_px": self.y0, "sigma_x_mm": self.sigma_x_mm,
                "sigma_y_mm": self.sigma_y_mm, "divergence_x_rad": tx, "divergence_y_rad": ty,
                "runs": list(self.runs), "source": self.source, "chi2": self.chi2}


def _model(p, X, Y):
    A, x0, y0, sx, sy, B = p
    gx = ndtr((X + 0.5 - x0) / sx) - ndtr((X - 0.5 - x0) / sx)
    gy = ndtr((Y + 0.5 - y0) / sy) - ndtr((Y - 0.5 - y0) / sy)
    return A * gx * gy + B


def fit_beam_profile(counts, window_px: Optional[int] = None) -> dict:
    """Pixel-integrated 2D Gaussian + background fitted to a direct-beam image around its maximum.
    Returns a dict with x0, y0, sigma_x, sigma_y (pixel), their errors, A, B, chi2 and the second-moment
    widths."""
    from .beamcentre import masked_centre_of_gravity
    c = np.asarray(counts, float)
    ny, nx = c.shape
    cg = masked_centre_of_gravity(c, radius_px=8.0)
    mx = max(cg.second_moment_x - 1 / 12, 1e-4)
    my = max(cg.second_moment_y - 1 / 12, 1e-4)
    h = int(window_px or np.clip(np.ceil(5 * np.sqrt(max(mx, my))) + 2, 5, 40))
    ix0, iy0 = int(round(cg.x)), int(round(cg.y))
    xs = slice(max(0, ix0 - h), min(nx, ix0 + h + 1))
    ys = slice(max(0, iy0 - h), min(ny, iy0 + h + 1))
    sub = c[ys, xs]
    Y, X = np.mgrid[ys, xs].astype(float)
    w = 1.0 / np.sqrt(np.maximum(sub, 1.0))
    B0 = float(np.median(np.concatenate([sub[0], sub[-1], sub[:, 0], sub[:, -1]])))
    p0 = [max(sub.sum() - B0 * sub.size, 1.0), cg.x, cg.y, np.sqrt(mx), np.sqrt(my), B0]
    lo = [0.0, cg.x - h, cg.y - h, 0.05, 0.05, -np.inf]
    hi = [np.inf, cg.x + h, cg.y + h, 2.0 * h, 2.0 * h, np.inf]
    r = least_squares(lambda p: ((_model(p, X, Y) - sub) * w).ravel(), p0, bounds=(lo, hi), x_scale="jac")
    dof = max(1, sub.size - 6)
    chi2 = float(np.sum(r.fun ** 2) / dof)
    try:
        cov = np.linalg.inv(r.jac.T @ r.jac) * max(1.0, chi2)
        err = np.sqrt(np.abs(np.diag(cov)))
    except np.linalg.LinAlgError:
        err = np.full(6, np.nan)
    A, x0, y0, sx, sy, B = r.x
    return {"x0": float(x0), "y0": float(y0), "sigma_x": float(sx), "sigma_y": float(sy),
            "sigma_x_err": float(err[3]), "sigma_y_err": float(err[4]), "A": float(A), "B": float(B),
            "chi2": chi2, "moment_x": float(np.sqrt(mx)), "moment_y": float(np.sqrt(my)),
            "window": (xs.start, xs.stop, ys.start, ys.stop)}


def beam_profiles(raws, max_runs: int = 10) -> Dict[tuple, BeamProfile]:
    """Beam profile per geometry (L2, lambda, L1) from its direct-beam runs: the empty-beam transmission
    runs when there are any (a sample in the beam can broaden it by its own small-angle scattering),
    else all direct-beam runs (sample transmissions), noted. Every run is fitted
    (:func:`fit_beam_profile`, up to ``max_runs``); the widths are the medians, their errors the spread
    (or the fit error for one run). Geometries without a direct-beam run get no profile: their
    resolution cannot be determined (no borrowing, the beam size depends on L1, L2 and the apertures)."""
    from ..catalogue import _config_key, looks_like_direct_beam
    from ..plan import role_of
    groups: Dict[tuple, list] = {}
    for r in raws:
        if r is None or getattr(r, "empty_reason", ""):
            continue
        try:
            if not looks_like_direct_beam(r):
                continue
        except Exception:  # noqa: BLE001
            continue
        groups.setdefault(_config_key(r), []).append(r)
    out: Dict[tuple, BeamProfile] = {}
    for k, rs in groups.items():
        eb = [r for r in rs if (getattr(r, "role", "") or role_of(r.sample_name)) == "empty_beam"]
        use = (eb or rs)[:max_runs]
        fits, runs = [], []
        for r in use:
            try:
                f = fit_beam_profile(r.counts)
            except Exception:  # noqa: BLE001
                continue
            fits.append(f)
            runs.append(r.run)
        if not fits:
            continue
        sx = np.array([f["sigma_x"] for f in fits])
        sy = np.array([f["sigma_y"] for f in fits])
        if len(fits) > 1:
            ex, ey = float(np.std(sx, ddof=1)), float(np.std(sy, ddof=1))
        else:
            ex, ey = fits[0]["sigma_x_err"], fits[0]["sigma_y_err"]
        r0 = use[0]
        ps = r0.pixel_size_m
        ps = (ps, ps) if np.isscalar(ps) else tuple(ps)
        p = BeamProfile(k, float(np.median([f["x0"] for f in fits])), float(np.median([f["y0"] for f in fits])),
                        float(np.median(sx)), float(np.median(sy)), ex, ey, (float(ps[0]), float(ps[1])),
                        float(getattr(r0, "L2_effective_m", r0.L2_m)), float(r0.wavelength_A), sorted(runs),
                        float(np.median([f["chi2"] for f in fits])),
                        float(np.median([f["moment_x"] for f in fits])), float(np.median([f["moment_y"] for f in fits])),
                        "empty beam" if eb else "direct beam (sample transmission runs: no empty beam here)")
        for name, s, m in (("x", p.sigma_x_px, p.moment_x_px), ("y", p.sigma_y_px, p.moment_y_px)):
            if m > 1.25 * s or m < 0.8 * s:
                p.notes.append(f"beam not Gaussian in {name}: second-moment width {m:.2f} px vs Gaussian {s:.2f} px")
        out[k] = p
    return out
