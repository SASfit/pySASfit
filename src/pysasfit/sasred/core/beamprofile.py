# SPDX-License-Identifier: BSD-3-Clause
"""Direct-beam profile of a geometry: 2D Gaussian fit, beam divergence, and the geometric part of
the q resolution.

The attenuated direct beam (beam stop out, no sample: the empty-beam transmission runs) shows on the
detector what a point of the sample's scattering pattern is smeared with by the collimation (source
and sample apertures, their distance coll, the distance SD) and the detector's own resolution. Its
profile is fitted with a separable Gaussian integrated over every pixel plus a flat background,

    N_ij = A [Phi((x_j + 1/2 - x0)/s_x) - Phi((x_j - 1/2 - x0)/s_x)]
             [Phi((y_i + 1/2 - y0)/s_y) - Phi((y_i - 1/2 - y0)/s_y)] + B,

with Poisson weights on a window around the beam. Because the model is integrated over the pixel,
s_x and s_y are the widths of the beam itself, without the pixel size (which the averaging accounts
for separately). The angular widths seen from the sample are

    sigma_theta_x = s_x p_x / SD,   sigma_theta_y = s_y p_y / SD   (rad, standard deviations),

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

__all__ = ["BeamProfile", "fit_beam_profile", "beam_profiles", "collimation_profile", "calibrated_profile",
           "fit_apertures"]


@dataclass
class BeamProfile:
    """Gaussian width of the direct beam of one geometry."""
    key: tuple                       # geometry (SD, lambda, coll)
    x0: float                        # fitted centre [pixel]
    y0: float
    sigma_x_px: float                # Gaussian sigma of the beam, pixel size removed [pixel]
    sigma_y_px: float
    sigma_x_err: float = np.nan
    sigma_y_err: float = np.nan
    pixel_size_m: tuple = (np.nan, np.nan)
    SD_m: float = np.nan
    wavelength_A: float = np.nan
    runs: List[int] = field(default_factory=list)
    chi2: float = np.nan             # reduced chi^2 of the fit (median over the runs)
    moment_x_px: float = np.nan      # sqrt(second moment - 1/12) of the beam, for comparison
    moment_y_px: float = np.nan
    source: str = "direct beam"      # "empty beam" | "direct beam (sample transmission runs)" | "by hand"
    notes: List[str] = field(default_factory=list)

    @property
    def estimated(self) -> bool:
        """True: from the collimation geometry (no direct beam measured), not fitted."""
        return self.source.startswith("collimation")

    def origin(self) -> str:
        """Where the divergence comes from, for notes."""
        if self.estimated:
            return "estimated from the collimation" + (f" ({self.notes[0]})" if self.notes else "")
        return f"from the direct beam (runs {', '.join(map(str, self.runs[:4]))})"

    @property
    def sigma_x_mm(self) -> float:
        return 1e3 * self.sigma_x_px * float(self.pixel_size_m[0])

    @property
    def sigma_y_mm(self) -> float:
        return 1e3 * self.sigma_y_px * float(self.pixel_size_m[1])

    @property
    def divergence_rad(self) -> tuple:
        """(sigma_theta_x, sigma_theta_y): angular standard deviations of the beam seen from the sample."""
        return (self.sigma_x_mm * 1e-3 / self.SD_m, self.sigma_y_mm * 1e-3 / self.SD_m)

    def describe(self) -> str:
        tx, ty = self.divergence_rad
        from ..catalogue import _key_text
        s = (f"{_key_text(self.key)}: beam σx = {self.sigma_x_mm:.3f} mm, σy = {self.sigma_y_mm:.3f} mm "
             f"(divergence σθ = {1e3 * tx:.3f} × {1e3 * ty:.3f} mrad, FWHM {2.3548e3 * tx:.3f} × "
             f"{2.3548e3 * ty:.3f} mrad)" + (f", centre ({self.x0:.2f}, {self.y0:.2f}) px" if np.isfinite(self.x0) else "")
             + f"; {self.source}"
             + (f", runs {', '.join(map(str, self.runs[:6]))}{'…' if len(self.runs) > 6 else ''}" if self.runs else "")
             + (f"; χ²_red {self.chi2:.3g}" if np.isfinite(self.chi2) else ""))
        return s + "".join(f"; {n}" for n in self.notes)

    def to_dict(self) -> dict:
        tx, ty = self.divergence_rad
        return {"key": list(self.key), "x0_px": self.x0, "y0_px": self.y0, "sigma_x_mm": self.sigma_x_mm,
                "sigma_y_mm": self.sigma_y_mm, "divergence_x_rad": tx, "divergence_y_rad": ty,
                "runs": list(self.runs), "source": self.source, "chi2": self.chi2}


def _aperture_prefs(a1, a2):
    """Source and sample apertures (x, y) [mm] of the preferences; the instrument-file values when the
    preferences cannot be read."""
    try:
        from ...core import settings as prefs
        g = lambda k: float(prefs.get(f"sasred.resolution.{k}"))   # noqa: E731
        return ((g("source_aperture_x_mm"), g("source_aperture_y_mm")),
                (g("sample_aperture_x_mm"), g("sample_aperture_y_mm")))
    except Exception:  # noqa: BLE001
        return a1, a2


def collimation_profile(raw, instrument=None) -> Optional[BeamProfile]:
    """Beam profile of the geometry of ``raw`` estimated from the collimation when no direct beam was
    measured: source aperture a1 (the neutron guide cross-section) at the distance coll before the
    sample aperture a2, detector at SD behind it. Each aperture adds a uniform distribution to the beam
    spot on the detector, a1·SD/coll (source projected through the sample aperture) and
    a2·(coll + SD)/coll (sample aperture seen from the source), so per axis

        σ_spot² = (a1 SD/coll)²/12 + (a2 (coll + SD)/coll)²/12 ,  σθ = σ_spot / SD

    (Mildner & Carpenter 1984, rectangular apertures). The apertures come from the instrument file
    (``collimation.source_aperture_mm``, ``collimation.sample_aperture_mm``, (x, y)), replaced by the
    preferences (SANS reduction → Resolution: 50 × 50 and 8 × 14 mm² by default) for an instrument whose
    file has a collimation section (SANS-I; ``from_preferences: false`` there keeps the file values);
    coll and SD from the run. None when either is missing."""
    if instrument is None:
        from ..reduce import load_instrument
        instrument = load_instrument(raw)
    col = (instrument or {}).get("collimation") or {}
    a1, a2 = col.get("source_aperture_mm"), col.get("sample_aperture_mm")
    if a1 and a2 and col.get("from_preferences", True):
        a1, a2 = _aperture_prefs(a1, a2)          # Settings → Preferences → SANS reduction → Resolution
    try:
        coll = float(raw.coll_m)
        SD = float(getattr(raw, "SD_effective_m", raw.SD_m))
    except (TypeError, ValueError, AttributeError):
        return None
    if not a1 or not a2 or not (coll > 0) or not (SD > 0) or not np.isfinite(coll + SD):
        return None
    a1 = np.broadcast_to(np.asarray(a1, float), (2,))
    a2 = np.broadcast_to(np.asarray(a2, float), (2,))
    s_mm = np.sqrt((a1 * SD / coll) ** 2 / 12 + (a2 * (coll + SD) / coll) ** 2 / 12)
    ps = raw.pixel_size_m
    ps = np.broadcast_to(np.asarray(ps, float), (2,))
    s_px = 1e-3 * s_mm / ps
    from ..catalogue import _config_key
    text = (f"guide {a1[0]:g}×{a1[1]:g} mm², sample aperture {a2[0]:g}×{a2[1]:g} mm², coll {coll:g} m, "
            f"SD {SD:.2f} m")
    return BeamProfile(_config_key(raw), np.nan, np.nan, float(s_px[0]), float(s_px[1]), np.nan, np.nan,
                       (float(ps[0]), float(ps[1])), SD, float(raw.wavelength_A), [],
                       source="collimation (no direct beam: estimated)", notes=[text])


def _spot_terms(SD, coll):
    """Coefficients (u, v) of σ_spot² = (u a₁² + v a₂²)/12 for source / sample aperture widths a₁, a₂."""
    return (SD / coll) ** 2, ((coll + SD) / coll) ** 2


def fit_apertures(measured, coll, a1_pref, a2_pref):
    """Effective source and sample aperture widths (a₁, a₂) per axis [mm] from direct-beam profiles of one
    collimation length ``coll`` [m]: σ_spot² = (a₁² (SD/coll)² + a₂² ((coll+SD)/coll)²)/12.

    Two or more detector distances: both widths, non-negative least squares on σ². One distance: the sample
    aperture of the preferences is kept and a₁ follows from the measured width (if the width is smaller
    than the sample-aperture term alone, a₁ of the preferences is kept and a₂ fitted instead).
    Returns ((a1x, a1y), (a2x, a2y), how)."""
    from scipy.optimize import nnls
    SDs = sorted({round(float(p.SD_m), 2) for p in measured})
    two = len(SDs) >= 2 and SDs[-1] >= 1.5 * SDs[0]     # distances close together cannot separate a₁ and a₂
    if not two and len(measured) > 1:                   # one distance: the profile from the most runs
        measured = [max(measured, key=lambda p: (len(p.runs), -abs(p.SD_m - SDs[-1])))]
        SDs = [round(float(measured[0].SD_m), 2)]
    a1, a2 = [0.0, 0.0], [0.0, 0.0]
    for ax in (0, 1):
        rows, rhs = [], []
        for p in measured:
            s = p.sigma_x_mm if ax == 0 else p.sigma_y_mm
            u, v = _spot_terms(float(p.SD_m), coll)
            rows.append([u, v])
            rhs.append(12.0 * s * s)
        A, b = np.asarray(rows), np.asarray(rhs)
        if two:
            sol, _ = nnls(A, b)
            a1[ax], a2[ax] = float(np.sqrt(sol[0])), float(np.sqrt(sol[1]))
        if not two or not (a1[ax] > 0 and a2[ax] > 0):     # one distance (or an implausible two-distance fit)
            i = int(np.argmax([len(p.runs) for p in measured]))   # the profile from the most runs
            u, v, bb = A[i, 0], A[i, 1], b[i]
            r1 = (bb - v * a2_pref[ax] ** 2) / u
            if r1 > 0:
                a1[ax], a2[ax] = float(np.sqrt(r1)), float(a2_pref[ax])
            else:
                a1[ax] = float(a1_pref[ax])
                a2[ax] = float(np.sqrt(max(bb - u * a1_pref[ax] ** 2, 0.0) / v))
    how = ("a₁ and a₂ fitted (direct beams at SD " + ", ".join(f"{s:g}" for s in SDs) + " m)" if two
           else f"sample aperture of the preferences kept, a₁ fitted (direct beam at SD {SDs[0]:g} m)")
    return (a1[0], a1[1]), (a2[0], a2[1]), how


def calibrated_profile(raw, profiles, instrument=None) -> Optional[BeamProfile]:
    """Beam profile of the geometry of ``raw`` (no direct beam) from the collimation model calibrated on
    the measured direct-beam profiles of the same collimation length (``profiles``: {key: BeamProfile};
    same wavelength preferred, else all wavelengths). The divergence is not simply copied: the
    sample-aperture term grows as 1/SD, so a shorter distance has a wider angular resolution. None when no
    direct beam of that collimation was measured (then :func:`collimation_profile`)."""
    try:
        coll = float(raw.coll_m)
        SD = float(getattr(raw, "SD_effective_m", raw.SD_m))
        lam = float(raw.wavelength_A)
    except (TypeError, ValueError, AttributeError):
        return None
    if not (coll > 0 and SD > 0):
        return None
    meas = [p for p in (profiles or {}).values()
            if not p.estimated and len(p.key) > 2 and abs(float(p.key[2]) - coll) < 0.05
            and np.isfinite(p.sigma_x_mm) and np.isfinite(p.sigma_y_mm) and np.isfinite(p.SD_m)]
    same = [p for p in meas if abs(float(p.key[1]) - lam) < 0.05]
    meas = same or meas
    empty = [p for p in meas if p.source == "empty beam"]   # a sample in the beam broadens it
    meas = empty or meas
    if not meas:
        return None
    if instrument is None:
        from ..reduce import load_instrument
        instrument = load_instrument(raw)
    col = (instrument or {}).get("collimation") or {}
    a1p, a2p = col.get("source_aperture_mm") or (50.0, 50.0), col.get("sample_aperture_mm") or (8.0, 14.0)
    if col.get("from_preferences", True):
        a1p, a2p = _aperture_prefs(a1p, a2p)
    a1p = tuple(np.broadcast_to(np.asarray(a1p, float), (2,)))
    a2p = tuple(np.broadcast_to(np.asarray(a2p, float), (2,)))
    a1, a2, how = fit_apertures(meas, coll, a1p, a2p)
    u, v = _spot_terms(SD, coll)
    s_mm = np.sqrt((u * np.square(a1) + v * np.square(a2)) / 12.0)
    ps = np.broadcast_to(np.asarray(raw.pixel_size_m, float), (2,))
    s_px = 1e-3 * s_mm / ps
    runs = sorted({n for p in meas for n in p.runs})
    from ..catalogue import _config_key
    text = (f"calibrated on the direct beam{'s' if len(meas) > 1 else ''} of coll {coll:g} m, runs "
            f"{', '.join(map(str, runs[:6]))}{'…' if len(runs) > 6 else ''}: effective guide {a1[0]:.1f}×{a1[1]:.1f} mm², "
            f"sample aperture {a2[0]:.1f}×{a2[1]:.1f} mm² ({how}); SD {SD:.2f} m")
    return BeamProfile(_config_key(raw), np.nan, np.nan, float(s_px[0]), float(s_px[1]), np.nan, np.nan,
                       (float(ps[0]), float(ps[1])), SD, lam, runs,
                       source="collimation calibrated on direct beams (no direct beam here)", notes=[text])


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
    """Beam profile per geometry (SD, lambda, coll) from its direct-beam runs: the empty-beam transmission
    runs when there are any (a sample in the beam can broaden it by its own small-angle scattering),
    else all direct-beam runs (sample transmissions), noted. Every run is fitted
    (:func:`fit_beam_profile`, up to ``max_runs``); the widths are the medians, their errors the spread
    (or the fit error for one run). Geometries without a direct-beam run get no profile: their
    resolution cannot be determined (no borrowing, the beam size depends on coll, SD and the apertures)."""
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
                        float(getattr(r0, "SD_effective_m", r0.SD_m)), float(r0.wavelength_A), sorted(runs),
                        float(np.median([f["chi2"] for f in fits])),
                        float(np.median([f["moment_x"] for f in fits])), float(np.median([f["moment_y"] for f in fits])),
                        "empty beam" if eb else "direct beam (sample transmission runs: no empty beam here)")
        for name, s, m in (("x", p.sigma_x_px, p.moment_x_px), ("y", p.sigma_y_px, p.moment_y_px)):
            if m > 1.25 * s or m < 0.8 * s:
                p.notes.append(f"beam not Gaussian in {name}: second-moment width {m:.2f} px vs Gaussian {s:.2f} px")
        out[k] = p
    return out
