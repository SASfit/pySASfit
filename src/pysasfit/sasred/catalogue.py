# SPDX-License-Identifier: BSD-3-Clause
"""Header table of a folder of SANS-I or SANS-LLB raw files (``sans<year>n<run>.hdf``,
``sans-llb<year>n<run>.hdf``)."""
from __future__ import annotations

import glob
import os
import re
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np

from .io.raw import read_raw
from .io.sinq import RawRun

__all__ = ["CatalogueEntry", "scan_folder", "raw_files", "is_transmission", "beamstop_limit",
           "BeamstopGeometry", "beamstop_geometry", "GeometryCentre", "beam_centres", "config_key", "beamstop_distance", "classify", "direct_beam_fraction",
           "looks_like_direct_beam"]

_RUN = re.compile(r"n0*(\d+)\.hdf$", re.IGNORECASE)


def beamstop_limit() -> float:
    """Distance [mm] of the beam stop from the beam centre (on the detector) above which it is out
    of the direct beam (preference ``sasred.transmission.beamstop_out_mm``, factory 100 mm)."""
    try:
        from ..core import settings as prefs
        return float(prefs.effective("sasred.transmission.beamstop_out_mm"))
    except Exception:  # noqa: BLE001 -- settings not available (scripts, tests)
        return 100.0


def direct_beam_fraction(counts, radius_px: float = 6.0) -> float:
    """Share of all counts within ``radius_px`` of the brightest spot (3 x 3 smoothed): close to 1
    for a direct-beam image, small when the beam stop shadows the beam."""
    from scipy.ndimage import uniform_filter
    c = np.asarray(counts, float)
    tot = c.sum()
    if tot <= 0:
        return 0.0
    j, i = np.unravel_index(np.argmax(uniform_filter(c, 3)), c.shape)
    yy, xx = np.mgrid[0:c.shape[0], 0:c.shape[1]]
    return float(c[np.hypot(xx - i, yy - j) <= radius_px].sum() / tot)


def looks_like_direct_beam(raw, fraction: float = 0.6) -> bool:
    """The image itself shows the unshadowed direct beam (Vifor: ≥ 0.97 with the beam stop out,
    ≤ 0.38 with it in)."""
    return direct_beam_fraction(raw.counts) >= fraction


def _config_key(raw):
    """(SD to 0.1 m, λ to 0.1 Å, coll to 0.1 m): the beam centre moves with all three (gravity
    with SD, λ and the collimation)."""
    return (round(raw.SD_m, 1), round(raw.wavelength_A, 1), round(float(raw.coll_m), 1))


def _key_text(k) -> str:
    return f"coll {k[2]:g} m, SD {k[0]:g} m, λ {k[1]:g} Å" if len(k) > 2 else f"SD {k[0]:g} m, λ {k[1]:g} Å"


def _nearest(cands, k, weight=None):
    """Donor configuration for ``k``: same λ required; same collimation coll preferred; then the
    nearest SD (``weight(kk)``: further preference, e.g. more direct-beam runs)."""
    same = [kk for kk in cands if kk[1] == k[1]]
    if not same:
        return None
    return min(same, key=lambda kk: (len(kk) > 2 and len(k) > 2 and kk[2] != k[2],
                                     weight(kk) if weight else 0, abs(kk[0] - k[0])))


def mask_key(raw) -> tuple:
    """Instrument configuration a mask belongs to: collimation coll and distance SD (0.1 m),
    wavelength (0.1 Å), beam stop type and beam stop position (x, y to 1 mm)."""
    bx, by = getattr(raw, "beamstop_xy_mm", (np.nan, np.nan))
    bst = str(getattr(raw, "beamstop_type", "") or "")
    if getattr(raw, "bank", ""):                     # SANS-LLB wide-angle panel: a mask of its own
        bst = f"{bst} | {raw.bank} panel"
    pd = getattr(raw, "panel_SD_m", None)
    if pd is not None:                               # SANS-LLB: the panels' distance belongs to the configuration
        bst = f"{bst} | panels {pd:.2f} m"
    return (round(float(raw.coll_m), 1), round(float(raw.SD_m), 1), round(float(raw.wavelength_A), 1),
            bst, int(round(bx)) if np.isfinite(bx) else None,
            int(round(by)) if np.isfinite(by) else None)


def legacy_mask_key(key) -> tuple:
    """The mask key as written before the panel distance was part of it (masks saved then are still
    found for their configuration)."""
    import re as _re
    if not isinstance(key[3], str) or "| panels" not in key[3]:
        return key
    return key[:3] + (_re.sub(r" \| panels [-0-9.]+ m$", "", key[3]),) + tuple(key[4:])


def mask_key_label(key) -> str:
    coll, SD, lam, bst, bx, by = key
    return (f"coll {coll:g} m, SD {SD:g} m, λ {lam:g} Å, beam stop {bst or '?'} at ({bx}, {by}) mm")


@dataclass
class BeamstopGeometry:
    """Where the beam stop is relative to the beam centre, per configuration (SD, λ).

    The beam stop motors (bsx, bsy) [mm] are tied to detector pixels by the shadow of the beam stop
    in the scattering runs (``ref_mm`` <-> ``shadow_px``, signs ``sign`` from the instrument file);
    the beam centre comes from the direct-beam runs of the configuration (or of the same wavelength
    at another distance, or the shadow itself as the last resort). The distance of the beam stop
    from the beam centre on the detector then decides whether it is out of the beam."""
    key: tuple
    beam_centre_px: Optional[tuple]
    centre_source: str
    ref_mm: Optional[tuple]          # most frequent in-beam motor position
    shadow_px: Optional[tuple]       # centroid of its shadow
    pixel_mm: tuple
    sign: tuple = (1, 1)

    def beamstop_px(self, bs_mm):
        if self.ref_mm is None or self.shadow_px is None:
            return None
        return tuple(self.shadow_px[i] + self.sign[i] * (bs_mm[i] - self.ref_mm[i]) / self.pixel_mm[i]
                     for i in (0, 1))

    def distance_mm(self, bs_mm) -> Optional[float]:
        """Beam stop centre to beam centre on the detector [mm]."""
        p = self.beamstop_px(bs_mm)
        if p is None or self.beam_centre_px is None:
            return None
        return float(np.hypot((p[0] - self.beam_centre_px[0]) * self.pixel_mm[0],
                              (p[1] - self.beam_centre_px[1]) * self.pixel_mm[1]))

    @property
    def off_centre_mm(self) -> Optional[float]:
        """In-beam position: how far the beam stop sits from the beam centre (alignment, gravity)."""
        return None if self.ref_mm is None else self.distance_mm(self.ref_mm)

    def describe(self) -> str:
        if self.ref_mm is None or self.shadow_px is None:
            return "no beam stop shadow found" + ("" if self.ref_mm is None else
                                                  f" (beam stop at bsx, bsy = {self.ref_mm[0]:g}, {self.ref_mm[1]:g} mm)")
        bc = "?" if self.beam_centre_px is None else f"({self.beam_centre_px[0]:.2f}, {self.beam_centre_px[1]:.2f})"
        off = self.off_centre_mm
        return (f"beam stop in the beam at bsx, bsy = {self.ref_mm[0]:g}, {self.ref_mm[1]:g} mm, shadow centre "
                f"({self.shadow_px[0]:.2f}, {self.shadow_px[1]:.2f}) px; beam centre {bc} px "
                f"from {self.centre_source}" + ("" if off is None else f"; beam stop {off:.1f} mm off the beam centre"))


def _signs(raw=None):
    try:
        from .reduce import load_instrument
        b = load_instrument(raw).get("beamstop", {})
        return (int(b.get("x_sign", 1)), int(b.get("y_sign", 1)))
    except Exception:  # noqa: BLE001
        return (1, 1)


def beamstop_geometry(raws) -> dict:
    """:class:`BeamstopGeometry` per configuration key (SD, λ, coll) of a set of runs."""
    from collections import Counter

    from .core.beamcentre import masked_centre_of_gravity
    from .core.mask import suggest_beamstop
    raws = [r for r in raws if r is not None and r.beamstop_xy_mm is not None]
    direct = {}
    blocked = {}
    for r in raws:
        (direct if looks_like_direct_beam(r) else blocked).setdefault(_config_key(r), []).append(r)
    centres = {}
    for k, rs in direct.items():
        cs = [masked_centre_of_gravity(r.counts) for r in rs[:20]]
        centres[k] = (float(np.median([c.x for c in cs])), float(np.median([c.y for c in cs])))
    sign = _signs(raws[0] if raws else None)
    out = {}
    for k in set(direct) | set(blocked):
        bc, src = centres.get(k), "the direct-beam runs"
        if bc is None:
            kk = _nearest(list(centres), k, lambda kk: -len(direct[kk]))
            if kk is not None:
                bc, src = centres[kk], f"the direct beam at {_key_text(kk)}"
        rs = blocked.get(k, [])
        ref = shadow = None
        if rs:
            pos = Counter((round(float(r.beamstop_xy_mm[0]), 1), round(float(r.beamstop_xy_mm[1]), 1)) for r in rs)
            ref = pos.most_common(1)[0][0]
            guess = bc if bc is not None else (rs[0].counts.shape[1] / 2, rs[0].counts.shape[0] / 2)
            cents = []
            for r in [r for r in rs if (round(float(r.beamstop_xy_mm[0]), 1),
                                         round(float(r.beamstop_xy_mm[1]), 1)) == ref][:15]:
                m, _ = suggest_beamstop(r.counts, guess, shape="shadow")
                if m.any():
                    yy, xx = np.nonzero(m)
                    cents.append((xx.mean(), yy.mean()))
            if cents:
                shadow = (float(np.median([c[0] for c in cents])), float(np.median([c[1] for c in cents])))
        if bc is None and shadow is not None:
            bc, src = shadow, "the beam stop shadow (no direct-beam run)"
        r0 = (direct.get(k) or blocked.get(k))[0]
        pix = (r0.pixel_size_m[0] * 1e3, r0.pixel_size_m[1] * 1e3)
        out[k] = BeamstopGeometry(k, bc, src if bc is not None else "nowhere", ref, shadow, pix, sign)
    # configurations without a shadow borrow the motor <-> pixel relation of another one
    donors = [g for g in out.values() if g.shadow_px is not None]
    for g in out.values():
        if g.shadow_px is None and donors:
            d = min(donors, key=lambda d: (d.key[1] != g.key[1], len(d.key) > 2 and d.key[2] != g.key[2],
                                           abs(d.key[0] - g.key[0])))
            g.ref_mm, g.shadow_px = d.ref_mm, d.shadow_px
    return out


@dataclass
class GeometryCentre:
    """Beam centre of one geometry (SD, λ, coll)."""
    key: tuple
    x: float
    y: float
    sigma_x: float
    sigma_y: float
    runs: List[int]
    source: str                      # "direct beam" | "borrowed from SD = … m" | "by hand"

    def describe(self) -> str:
        return (f"{_key_text(self.key)}: x = {self.x:.3f} ± {self.sigma_x:.3f}, "
                f"y = {self.y:.3f} ± {self.sigma_y:.3f} ({self.source}"
                + (f", runs {self.runs[0]}…{self.runs[-1]} ({len(self.runs)})" if self.runs else "") + ")")


def config_key(raw) -> tuple:
    """Geometry key (SD to 0.1 m, λ to 0.1 Å, coll to 0.1 m) under which beam centres are kept."""
    return _config_key(raw)


def beam_centres(raws, borrow: bool = True) -> Dict[tuple, GeometryCentre]:
    """Beam centre per geometry (SD, λ, coll) from its direct-beam runs (beam stop out): median of the
    masked centres of gravity, σ from their spread (or the statistical σ for one run). The solid
    angle and q of every pixel depend on it, so each geometry needs its own centre (gravity shifts
    it with SD and λ, the collimation coll changes the beam). With ``borrow`` a geometry without
    direct-beam run gets the centre of the same wavelength, preferably the same collimation, at the
    nearest distance, marked as borrowed."""
    from .core.beamcentre import masked_centre_of_gravity
    groups: Dict[tuple, list] = {}
    keys = set()
    for r in raws:
        if r is None:
            continue
        keys.add(_config_key(r))
        if looks_like_direct_beam(r):
            groups.setdefault(_config_key(r), []).append(r)
    out: Dict[tuple, GeometryCentre] = {}
    for k, rs in groups.items():
        cs = [masked_centre_of_gravity(r.counts) for r in rs[:50]]
        xs, ys = np.array([c.x for c in cs]), np.array([c.y for c in cs])
        if len(cs) > 1:
            sx, sy = float(np.std(xs, ddof=1)), float(np.std(ys, ddof=1))
        else:
            sx, sy = cs[0].sigma_x, cs[0].sigma_y
        out[k] = GeometryCentre(k, float(np.median(xs)), float(np.median(ys)), sx, sy,
                                sorted(r.run for r in rs), "direct beam")
    if borrow:
        for k in keys - set(out):
            kk = _nearest([kk for kk in out if kk in groups], k, lambda kk: -len(groups[kk]))
            if kk is not None:
                c = out[kk]
                out[k] = GeometryCentre(k, c.x, c.y, c.sigma_x, c.sigma_y, [],
                                        f"borrowed from {_key_text(kk)}: no direct-beam run here")
    return out


def beamstop_distance(raw, geometry: dict) -> Optional[float]:
    """Distance [mm] of the beam stop from the beam centre on the detector for this run."""
    g = geometry.get(_config_key(raw))
    if g is None or raw.beamstop_xy_mm is None:
        return None
    return g.distance_mm(raw.beamstop_xy_mm)


def is_transmission(raw, geometry: Optional[dict] = None, limit: Optional[float] = None) -> bool:
    """Transmission (direct-beam) run: the beam stop is moved out of the beam, in x or y (SANS-I
    typically bsy = -200 mm or lower) or out of the detection area, i.e. its distance from the beam
    centre (:func:`beamstop_geometry`) exceeds ``limit`` (preference, 100 mm). The attenuator does
    not decide: it is usually in against saturation, but not with very long wavelengths and
    collimations. Without a geometry the image decides (:func:`looks_like_direct_beam`)."""
    if geometry:
        d = beamstop_distance(raw, geometry)
        if d is not None:
            if limit is None:
                limit = beamstop_limit()
                try:
                    from .reduce import load_instrument
                    own = load_instrument(raw).get("beamstop", {}).get("out_mm")
                    if own is not None and getattr(raw, "instrument", "sans_i") != "sans_i":
                        limit = float(own)
                except Exception:  # noqa: BLE001
                    pass
            return d > limit
    return looks_like_direct_beam(raw)


def classify(raws, limit: Optional[float] = None):
    """Kind of every run ("transmission" / "scattering") from the beam stop distance to the beam
    centre, and the runs where position and image disagree (beam stop out but no direct beam in
    the image, or beam stop in but the image shows the direct beam).
    Returns (kinds, disagreements, geometry)."""
    raws = [r for r in raws if r is not None]
    geo = beamstop_geometry(raws)
    kinds, odd = {}, []
    for r in raws:
        t = is_transmission(r, geo, limit)
        kinds[r.run] = "transmission" if t else "scattering"
        if t != looks_like_direct_beam(r):
            d = beamstop_distance(r, geo)
            where = "?" if d is None else f"{d:.0f} mm"
            odd.append((r.run, f"beam stop {where} from the beam centre, but the image "
                        + ("shows no direct beam" if t else "shows the direct beam")))
    return kinds, odd, geo


@dataclass
class CatalogueEntry:
    path: str
    run: int
    raw: Optional[RawRun] = None        # header and counts, without checksum
    error: Optional[str] = None         # file that could not be read
    kind: str = ""                      # transmission | scattering (set by scan_folder)
    remark: str = ""                    # e.g. beam stop position and image disagree
    skip: str = ""                      # why the run is skipped (time 0, 0 counts, monitor 0), else ""

    def row(self, units="nm"):
        """Values for the run table (wavelength in nm or Å)."""
        from .units import units as _units
        r = self.raw
        if r is None:
            return [self.run, f"(unreadable: {self.error})"] + [""] * (len(COLUMNS) - 3) + ["skipped: unreadable"]
        tot = int(r.counts.sum())
        lam = f"{_units(units).lam(r.wavelength_A):.{3 if units == 'nm' else 2}f}"
        kind = {"transmission": "trans.", "scattering": "scatt."}.get(self.kind, "")
        if not kind:
            kind = "trans." if is_transmission(r) else "scatt."
        if self.remark:
            kind += " ?"
        SD = f"{r.SD_m:.2f}"
        pd = getattr(r, "panel_SD_m", None)
        if pd is not None:                           # SANS-LLB: central (wide-angle panels)
            SD += f" ({pd:.2f})"
            if getattr(r, "panel_SD_spread_m", 0.0) > 0.001:
                SD += " !"
        bx, by = r.beamstop_xy_mm
        bs = f"{r.beamstop_type or '?'} ({bx:.0f}, {by:.0f})"
        dl = "" if r.dlam_over_lam_fwhm is None else f"{r.dlam_over_lam_fwhm:.3f}"
        tilt = getattr(r, "selector_tilt_deg", None)
        if tilt is not None:
            dl += f" ({round(tilt, 1) + 0.0:g}°)"
        return [r.run, r.sample_name, kind, lam, dl, f"{r.coll_m:.1f}", SD, bs,
                r.attenuator,
                f"{r.monitors.get('monitor1' if r.instrument == 'sans_i' else r.preset_monitor, 0):.4g}", f"{r.counting_time_s:.1f}", f"{tot:d}",
                "yes" if r.aborted else "", self.comment()]

    def comment(self) -> str:
        """The comment column: why the run is skipped, and remarks (beam stop / image)."""
        parts = []
        if self.skip:
            parts.append(f"skipped: {self.skip}")
        if self.remark:
            parts.append(self.remark)
        r = self.raw
        if r is not None and getattr(r, "panel_SD_spread_m", 0.0) > 0.001:
            d = ", ".join(f"{k} {p.distance_m:.3f} m" for k, p in (r.panels or {}).items())
            parts.append(f"wide-angle panels at different distances ({d})")
        return "; ".join(parts)


COLUMNS = ["run", "sample", "kind", "λ [nm]", "dλ/λ (tilt)", "coll [m]", "SD [m] (panels)", "beam stop (x, y mm)", "att.", "monitor1",
           "time [s]", "counts", "aborted", "comment"]


def raw_files(folder) -> List[str]:
    files = [p for p in glob.glob(os.path.join(folder, "*.hdf")) if _RUN.search(os.path.basename(p))]
    return sorted(files, key=lambda p: int(_RUN.search(os.path.basename(p)).group(1)))


def scan_folder(folder, progress=None) -> List[CatalogueEntry]:
    """Read the header of every raw file in ``folder`` (sorted by run number).
    ``progress(i, n)`` is called after each file. Unreadable files are kept with
    their error message instead of stopping the scan."""
    files = raw_files(folder)
    out = []
    for i, p in enumerate(files):
        run = int(_RUN.search(os.path.basename(p)).group(1))
        try:
            out.append(CatalogueEntry(p, run, read_raw(p, checksum=False)))
        except Exception as e:  # noqa: BLE001 -- listed, not fatal
            out.append(CatalogueEntry(p, run, None, f"{type(e).__name__}: {e}"))
        if progress is not None:
            progress(i + 1, len(files))
    for e in out:
        if e.raw is not None:
            e.skip = e.raw.empty_reason
    kinds, odd, _ = classify([e.raw for e in out if e.raw is not None and not e.skip])
    odd = dict(odd)
    for e in out:
        if e.raw is not None and not e.skip:
            e.kind = kinds[e.run]
            e.remark = odd.get(e.run, "")
    return out
