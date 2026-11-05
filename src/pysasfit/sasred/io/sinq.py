# SPDX-License-Identifier: BSD-3-Clause
"""SinqNexusSource: read SANS-I raw HDF5 files (correction step 1).

Missing required header fields raise an error; they are never replaced by
defaults silently. Optional fields (absent in older files, e.g. the sample
thickness before 2025) are returned as None and listed in `missing_optional`.
Units are taken from the `units` attributes and converted: wavelength to
Angstrom, distances to m.
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import h5py
import numpy as np

__all__ = ["RawRun", "read_sans_i", "MissingField"]


class MissingField(KeyError):
    pass


_LENGTH_TO_M = {"m": 1.0, "metre": 1.0, "meter": 1.0, "mm": 1e-3, "cm": 1e-2}
_WAVELENGTH_TO_A = {"nm": 10.0, "angstrom": 1.0, "a": 1.0, "å": 1.0, "ang": 1.0}


@dataclass
class RawRun:
    path: str
    run: int
    sha256: str
    counts: np.ndarray            # (ny, nx), frames summed
    n_frames: int
    counting_time_s: float
    monitors: Dict[str, float]    # monitor1 ... integrated_beam
    preset: float
    count_mode: str
    wavelength_A: float
    dlam_over_lam_fwhm: Optional[float]
    coll_m: float
    SD_m: float
    pixel_size_m: tuple
    detector_offset_m: tuple      # lateral detector translation (x, y) as stored
    detector_tilt_deg: float      # chi_position (to be confirmed as the tilt)
    attenuator: str
    beamstop_xy_mm: tuple
    sample_name: str
    sample_position_mm: Optional[float]
    sample_thickness_cm: Optional[float]
    start_time: str
    end_time: Optional[str]
    missing_optional: List[str] = field(default_factory=list)
    SD_offset_m: float = 0.0      # distance offset for the geometry (SD_m stays as recorded)
    beamstop_type: str = ""       # beam stop selector (size/type), SANS/beam_stop/out_flag
    instrument: str = "sans_i"    # instrument file (instruments/sinq/<instrument>.yaml)
    preset_monitor: str = "monitor1"   # monitor the preset refers to (monitor mode)
    bank: str = ""                # detector bank: "" = the (central) area detector; SANS-LLB: "left", "bottom"
    panels: Optional[dict] = None     # SANS-LLB: {name: Panel} of the wide-angle banks (central run only)
    panel: Optional[object] = None    # the Panel of a bank run (geometry from pixel positions)
    central_pos_mm: Optional[tuple] = None   # SANS-LLB: (posx, posy) of the central pixels in the frame of the panels
    sample_name_in_file: Optional[str] = None  # set when the sample name was corrected by hand (names.py)
    role: str = ""                # role set by hand (names.py); empty: from the sample name
    selector_tilt_deg: Optional[float] = None   # Dornier selector tilt (SANS/Dornier-VS/tilt)
    dlam_source: str = ""         # where dlam_over_lam_fwhm comes from ("selector tilt …", "file header", …)
    frames: Optional[dict] = None  # multi-frame runs: per-frame counting_time_s, monitors, rotation_speed, wavelength_A
    scan: str = ""                # a selector speed scan (λ differs between the frames): not reduced
    excluded: str = ""            # excluded by hand (names.py, kept per folder): the reason; not reduced

    @property
    def SD_recorded_m(self) -> float:
        """Sample-detector distance as stored in the file (= SD_m)."""
        return self.SD_m

    @property
    def SD_effective_m(self) -> float:
        """Distance to the detection layer used for q and the solid angle: recorded + offset."""
        return self.SD_m + self.SD_offset_m

    @property
    def empty_reason(self) -> str:
        """Why the run holds no usable data, or "": measuring time 0, no counts on the detector (all banks) or the
        normalising monitor (instrument file) at 0. Such runs are skipped by the reduction."""
        why = []
        if getattr(self, "excluded", ""):
            why.append(f"excluded by hand: {self.excluded}")
        if getattr(self, "scan", ""):
            why.append(self.scan + " – λ differs between the frames")
        if not (float(self.counting_time_s or 0.0) > 0):
            why.append("measuring time 0 s")
        total = float(np.sum(self.counts)) if self.counts is not None else 0.0
        for p in (self.panels or {}).values():
            c = getattr(p, "counts", None)
            if c is not None:
                total += float(np.sum(c))
        if not total > 0:
            why.append("0 counts")
        try:
            from ..reduce import load_instrument
            mname = load_instrument(self)["monitor"]["normaliser"]
        except Exception:  # noqa: BLE001
            mname = self.preset_monitor
        mon = (self.monitors or {}).get(mname)
        if mon is None or not float(mon) > 0:
            why.append(f"{mname} 0" if mon is not None else f"no {mname}")
        return ", ".join(why)

    @property
    def aborted(self) -> bool:
        """Run that stopped before reaching its preset: the monitor count (monitor mode) or the
        counting time (timer mode) is below 99 % of the preset."""
        if self.count_mode == "monitor":
            m = self.monitors.get(self.preset_monitor)
            return m is not None and m < 0.99 * self.preset
        if self.count_mode in ("timer", "time"):
            return self.counting_time_s < 0.99 * self.preset
        return False


def _frames(run: "RawRun", t_frames, frame_monitors, rpm) -> None:
    """Multi-frame runs: the counts, times and monitors of the frames are summed; the per-frame values
    are kept in ``run.frames``. A selector speed scan (rotation speed differing between the frames) has a
    different λ in every frame (λ ∝ 1/speed; the stored λ belongs to the last frame): it is marked in
    ``run.scan`` and not reduced, but its frames serve the monitor test (core.monitors)."""
    if run.n_frames <= 1 and not frame_monitors:
        return
    sp = None if rpm is None else np.asarray(rpm, float).ravel()
    fr = {"counting_time_s": t_frames, "monitors": frame_monitors}
    if sp is not None and sp.size == run.n_frames and np.all(sp > 0):
        fr["rotation_speed"] = sp
        fr["wavelength_A"] = run.wavelength_A * sp[-1] / sp
        if sp.max() > 1.01 * sp.min():
            run.scan = (f"selector speed scan: {run.n_frames} frames, {sp.min():.0f}–{sp.max():.0f} rpm "
                        f"(λ {fr['wavelength_A'].min():.2f}–{fr['wavelength_A'].max():.2f} Å)")
    run.frames = fr


def _dlam_from_selector(run: "RawRun", tilt, dlam_header) -> None:
    """dλ/λ from the selector tilt and the calibration table of the instrument file (it depends only on
    the tilt); the header value SANS/monochromator/wavelength_spread when the tilt or the table is missing."""
    run.dlam_source = "" if dlam_header is None else "file header (monochromator/wavelength_spread)"
    if tilt is None:
        return
    run.selector_tilt_deg = float(np.asarray(tilt).ravel()[0])
    try:
        from ..reduce import load_instrument
        table = (load_instrument(run.instrument).get("wavelength") or {}).get("selector_tilt")
    except Exception:  # noqa: BLE001
        table = None
    if not table:
        return
    from ..core.resolution import dlam_from_tilt
    dl, note = dlam_from_tilt(run.selector_tilt_deg, table)
    run.dlam_over_lam_fwhm = dl
    run.dlam_source = f"selector tilt {round(run.selector_tilt_deg, 2) + 0.0:.2f}° (calibration table)" + (f"; {note}" if note else "")


def _read(g, path, required=True, missing=None):
    try:
        d = g[path]
    except KeyError:
        if required:
            raise MissingField(f"required field {g.name}/{path} missing")
        missing.append(path)
        return None, None
    v = d[()]
    if isinstance(v, np.ndarray) and v.shape == (1,):
        v = v[0]
    if isinstance(v, bytes):
        v = v.decode(errors="replace").strip()
    unit = d.attrs.get("units")
    if isinstance(unit, (bytes, np.bytes_)):
        unit = unit.decode().strip()
    return v, unit


def _length_m(v, unit, path):
    if unit is None:
        raise MissingField(f"{path}: no units attribute")
    f = _LENGTH_TO_M.get(str(unit).lower())
    if f is None:
        raise ValueError(f"{path}: unknown length unit '{unit}'")
    return float(v) * f


def sha256_of(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def distance_offset_for(start_time: str) -> float:
    """Detector distance offset [m] for a run started at ``start_time``: the preference
    ``sasred.detector.distance_offset_mm``, applied to runs started before
    ``sasred.detector.distance_offset_before`` (YYYY-MM-DD; empty = all runs), so that older files
    can be corrected once the instrument configuration writes the corrected distance."""
    try:
        from ...core import settings as prefs
        off = float(prefs.effective("sasred.detector.distance_offset_mm")) / 1000.0
        before = str(prefs.effective("sasred.detector.distance_offset_before") or "").strip()
    except Exception:  # noqa: BLE001 -- no settings (scripts, tests)
        return 0.0
    if off == 0.0:
        return 0.0
    if before and str(start_time)[:10] >= before[:10]:
        return 0.0
    return off


def read_sans_i(path, checksum: bool = True, distance_offset_m: Optional[float] = None) -> RawRun:
    """Read a SANS-I HDF file. ``checksum=False`` skips the SHA-256 (fast header
    catalogue); a reduction always reads with the checksum.

    ``distance_offset_m`` is the offset of the detection layer from the recorded detector distance:
    ``RawRun.SD_m`` stays as recorded (grouping, selection, labels), ``SD_effective_m`` = recorded +
    offset is used for q and the solid angle; None takes it from the preferences
    (:func:`distance_offset_for`)."""
    path = str(path)
    m = re.search(r"n0*(\d+)\.hdf$", Path(path).name)
    if not m:
        raise ValueError(f"cannot get the run number from '{path}'")
    missing: List[str] = []
    with h5py.File(path, "r") as f:
        g = f["entry1"]
        counts, _ = _read(g, "SANS/detector/counts")
        counts = np.asarray(counts)
        n_frames = 1
        if counts.ndim == 3:
            n_frames = counts.shape[0]
            counts = counts.sum(axis=0)
        t, tu = _read(g, "SANS/detector/counting_time")
        lam, lu = _read(g, "SANS/Dornier-VS/lambda")
        lf = _WAVELENGTH_TO_A.get(str(lu).lower()) if lu is not None else None
        if lf is None:
            raise ValueError(f"wavelength unit '{lu}' unknown or missing")
        dlam, _ = _read(g, "SANS/monochromator/wavelength_spread", False, missing)
        tilt, _ = _read(g, "SANS/Dornier-VS/tilt", False, missing)
        SD, SDu = _read(g, "SANS/detector/distance")
        coll, collu = _read(g, "SANS/collimator/geometry/size")
        pxs, pxu = _read(g, "SANS/detector/x_pixel_size")
        pys, pyu = _read(g, "SANS/detector/y_pixel_size")
        dx, dxu = _read(g, "SANS/detector/x_null", False, missing)   # not a translation; kept raw
        dy, dyu = _read(g, "SANS/detector/y_position", False, missing)
        chi, chiu = _read(g, "SANS/detector/chi_position", False, missing)
        rpm, _ = _read(g, "SANS/Dornier-VS/rotation_speed", False, missing)
        monitors, frame_monitors = {}, {}
        for k in ["monitor1", "monitor2", "monitor3", "monitor4", "monitor_5", "monitor_6",
                  "monitor_7", "monitor_8", "integrated_beam"]:
            v, _ = _read(g, f"SANS/{k}/counts", False, missing)
            if v is not None:
                a = np.asarray(v, float).ravel()
                monitors[k] = float(a.sum())          # several frames: counted over all of them
                if a.size > 1:
                    frame_monitors[k] = a
        if "monitor1" not in monitors:
            raise MissingField("monitor1 missing")
        preset, _ = _read(g, "control/preset")
        preset = float(np.asarray(preset, float).ravel()[0])          # per frame in multi-frame runs
        t_frames = np.asarray(t, float).ravel()
        t = float(t_frames.sum())
        mode, _ = _read(g, "control/mode")
        att, _ = _read(g, "SANS/attenuator/selection")
        bsx, _ = _read(g, "SANS/beam_stop/x_position")
        bsy, _ = _read(g, "SANS/beam_stop/y_position")
        bst, _ = _read(g, "SANS/beam_stop/out_flag", False, missing)      # beam stop selector (BerSANS "Beamstop")
        name, _ = _read(g, "sample/name")
        pos, _ = _read(g, "sample/position", False, missing)
        thk, thu = _read(g, "sample/thickness", False, missing)
        start, _ = _read(g, "start_time")
        end, _ = _read(g, "end_time", False, missing)
    if chiu is not None and str(chiu).lower() not in ("degree", "degrees", "deg"):
        raise ValueError(f"chi_position unit '{chiu}' unexpected")
    if thk is not None and thu is not None and str(thu).lower() != "cm":
        thk = _length_m(thk, thu, "sample/thickness") * 100.0
    run = RawRun(
        path=path, run=int(m.group(1)), sha256=sha256_of(path) if checksum else "",
        counts=counts, n_frames=n_frames, counting_time_s=float(t), monitors=monitors,
        preset=float(preset), count_mode=str(mode),
        wavelength_A=float(lam) * lf,
        dlam_over_lam_fwhm=None if dlam is None else float(dlam),
        coll_m=_length_m(coll, collu, "collimator/geometry/size"),
        SD_m=_length_m(SD, SDu, "detector/distance"),
        pixel_size_m=(_length_m(pxs, pxu, "x_pixel_size"), _length_m(pys, pyu, "y_pixel_size")),
        detector_offset_m=(None if dx is None else float(dx),
                           None if dy is None else _length_m(dy, dyu, "detector/y_position")),
        detector_tilt_deg=0.0 if chi is None else float(chi),
        attenuator=str(att).strip(), beamstop_xy_mm=(float(bsx), float(bsy)),
        beamstop_type="" if bst is None else str(np.asarray(bst).ravel()[0] if np.ndim(bst) else bst).strip(),
        sample_name=str(name), sample_position_mm=None if pos is None else float(pos),
        sample_thickness_cm=None if thk is None else float(thk),
        start_time=str(start), end_time=end, missing_optional=missing)
    _dlam_from_selector(run, tilt, dlam)
    _frames(run, t_frames, frame_monitors, rpm)
    off = distance_offset_for(run.start_time) if distance_offset_m is None else float(distance_offset_m)
    if off:
        run.SD_offset_m = off
    return run
