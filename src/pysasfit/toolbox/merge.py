# SPDX-License-Identifier: BSD-3-Clause
"""Merge reduced 1D curves of the different geometries (detector distances, wavelengths) of a
sample, grouped by the sample name (regular expression) and any other header item (temperature,
pressure, magnetic field, shear rate, ...) with a tolerance.

Readers (:func:`read_curves`):

* pySASred NXcanSAS (``.h5``, also the files written here, several entries per file)
* pySASred ASCII (``.dat``: q, I, Idev, Qdev, σ_sys; header with ``metadata_json``)
* BerSANS 1D files (``Type=SANSDIso``, e.g. ``D0082130.080``: q, I, dI, dq in 1/nm)
* plain columns q, I[, dI[, dq]] (q in 1/nm unless the header says 1/Å)
* pySASred results in memory, without a file (:func:`curve_from_reduced`; in the GUI *From pySASred*)

All q are handled in 1/nm. Every curve carries its header items (:attr:`Curve.meta`): the named
items ``sample``, ``L2_m``, ``L1_m``, ``wavelength_A``, ``run`` and everything else the file holds
(``hdf:<path>`` for pySASred, ``Sample/Temperature`` etc. for BerSANS).

Merging (:func:`merge`) keeps the absolute scale of every geometry: outside the overlaps the points
are kept as they are; in an overlap the points of all geometries are averaged with weights 1/dI² on
the bins of the coarsest curve there. Each overlap is checked: the ratio of the two curves
(log-log interpolation) gives a mismatch factor with its uncertainty and χ²/ndf. Scaling to a
reference geometry is optional (``scale_to``). ``method="all"`` keeps every point (as BerSANS'
merge). q ranges per geometry (``cuts``) and dropped points at the ends (``drop``) as BerSANS'
merge mask.
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

__all__ = ["Curve", "Overlap", "Merged", "CurveGroup", "read_curves", "read_folder", "curve_from_reduced", "sample_key", "group_curves", "merge",
           "merge_groups", "series", "write_ascii", "write_nxcansas", "file_stem", "fields_of"]

A_PER_NM = 10.0


# ------------------------------------------------------------------------------------- curves
@dataclass
class Curve:
    q: np.ndarray                  # 1/nm
    I: np.ndarray
    dI: np.ndarray
    dq: np.ndarray                 # 1/nm (zeros when unknown)
    path: str
    fmt: str                       # "nxcansas" | "pysasred-ascii" | "bersans" | "columns"
    meta: Dict[str, object] = field(default_factory=dict)
    entry: str = ""                # NXcanSAS entry name
    I_unit: str = ""

    @property
    def sample(self) -> str:
        return str(self.meta.get("sample", "") or "")

    @property
    def geometry(self) -> str:
        """Geometry label from detector distance, wavelength and collimation."""
        m = self.meta
        L2, lam, L1 = _num(m.get("L2_m")), _num(m.get("wavelength_A")), _num(m.get("L1_m"))
        if L2 is None and lam is None:
            ext = os.path.splitext(self.path)[1]
            return f"file {ext}" if ext else os.path.basename(self.path)
        s = []
        if L2 is not None:
            s.append(f"L2 {L2:.2f} m")
        if lam is not None:
            s.append(f"λ {lam / 10:.3f} nm")
        if L1 is not None:
            s.append(f"L1 {L1:.1f} m")
        return ", ".join(s)

    @property
    def name(self) -> str:
        b = os.path.basename(self.path)
        return f"{b}:{self.entry}" if self.entry else b

    def q_range(self) -> Tuple[float, float]:
        return (float(np.nanmin(self.q)), float(np.nanmax(self.q))) if len(self.q) else (np.nan, np.nan)


def _num(v) -> Optional[float]:
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None
    return x if np.isfinite(x) else None


def fields_of(curves: Sequence[Curve]) -> List[str]:
    """Header items found in the curves: named ones first, then the others alphabetically."""
    named = ["sample", "L2_m", "wavelength_A", "L1_m", "run", "thickness_cm", "geometry", "file", "format"]
    seen = set()
    for c in curves:
        seen.update(c.meta)
    return [k for k in named if k in seen or k in ("geometry", "file", "format")] + \
        sorted(k for k in seen if k not in named)


def value_of(c: Curve, key: str):
    if key == "geometry":
        return c.geometry
    if key == "file":
        return os.path.basename(c.path)
    if key == "format":
        return c.fmt
    return c.meta.get(key)


# ------------------------------------------------------------------------------------- readers
def _q_to_nm(unit: str) -> float:
    u = (unit or "").strip().lower().replace(" ", "")
    if u in ("1/a", "1/angstrom", "a^-1", "1/å", "å^-1", "angstrom^-1"):
        return A_PER_NM
    return 1.0


def _dec(v):
    if isinstance(v, bytes):
        return v.decode(errors="replace")
    if isinstance(v, np.ndarray):
        v = v.ravel()
        if v.size == 1:
            return _dec(v[0])
        return v
    if isinstance(v, np.generic):
        return v.item()
    return v


def _read_nxcansas(path) -> List[Curve]:
    import h5py
    out = []
    with h5py.File(path, "r") as f:
        for en, e in f.items():
            if not isinstance(e, h5py.Group):
                continue
            cls = _dec(e.attrs.get("canSAS_class", e.attrs.get("NX_class", b"")))
            if str(cls) not in ("SASentry", "NXentry"):
                continue
            dname = _dec(e.attrs.get("default", b"")) or next(
                (k for k, g in e.items() if isinstance(g, h5py.Group) and
                 str(_dec(g.attrs.get("canSAS_class", b""))) == "SASdata"), None)
            if not dname or dname not in e:
                continue
            d = e[dname]
            if "Q" not in d or "I" not in d:
                continue
            qf = _q_to_nm(str(_dec(d["Q"].attrs.get("units", b"1/nm"))))
            q = np.asarray(d["Q"][()], float) * qf
            I = np.asarray(d["I"][()], float)
            unc = str(_dec(d["I"].attrs.get("uncertainties", b"Idev")))
            dI = np.asarray(d[unc][()], float) if unc in d else np.full_like(I, np.nan)
            res = str(_dec(d["Q"].attrs.get("resolutions", b"Qdev")))
            dq = np.asarray(d[res][()], float) * qf if res in d else np.zeros_like(q)
            unit = str(_dec(d["I"].attrs.get("sasred_unit", d["I"].attrs.get("units", b""))))
            meta: Dict[str, object] = {}
            if "title" in e:
                meta["title"] = str(_dec(e["title"][()]))
            if "run" in e:
                meta["run"] = _dec(e["run"][()])
            for k, g in e.items():
                if not isinstance(g, h5py.Group):
                    continue
                c = str(_dec(g.attrs.get("canSAS_class", b"")))
                if c == "SASprocess":
                    for key in ("metadata", "group_values"):
                        if key in g:
                            try:
                                meta.update(json.loads(_dec(g[key][()])))
                            except (ValueError, TypeError):
                                pass
                    if "parameters" in g:
                        try:
                            p = json.loads(_dec(g["parameters"][()]))
                            meta.setdefault("sample", p.get("sample"))
                            meta.setdefault("label", p.get("label"))
                        except (ValueError, TypeError):
                            pass
                    if "raw_file" in g:
                        meta.setdefault("raw_file", str(_dec(g["raw_file"][()])))
                elif c == "SASsample":
                    for kk, ds in g.items():
                        if isinstance(ds, h5py.Dataset) and ds.size == 1:
                            v = _dec(ds[()])
                            meta.setdefault("sample" if kk == "name" else f"sample/{kk}", v)
            if not meta.get("sample"):
                meta["sample"] = meta.get("title", "")
            out.append(Curve(q, I, dI, dq, str(path), "nxcansas", meta, en, unit))
    return out


def _read_text(path) -> List[Curve]:
    with open(path, encoding="utf-8", errors="replace") as fh:
        lines = fh.read().splitlines()
    head = lines[:40]
    if any(ln.strip() == "%File" for ln in head) and any("SANSDIso" in ln for ln in head):
        return [_read_bersans(path, lines)]
    meta: Dict[str, object] = {}
    fmt, qf, unit = "columns", 1.0, ""
    rows = []
    for ln in lines:
        s = ln.strip()
        if not s:
            continue
        if s.startswith("#"):
            h = s.lstrip("#").strip()
            if h.startswith("pySASred"):
                fmt = "pysasred-ascii"
            elif h.startswith("pyTOOLbox merge"):
                fmt = "merged"
            elif h.startswith("sample key:"):
                meta["sample"] = h.split(":", 1)[1].strip()
            elif h.startswith("group_json ="):
                try:
                    g = json.loads(h.split("=", 1)[1])
                    meta.update({k: (0.5 * (v[0] + v[1]) if isinstance(v, list) and len(v) == 2 else v)
                                 for k, v in g.items()})
                except ValueError:
                    pass
            elif h.startswith("metadata_json ="):
                try:
                    meta.update(json.loads(h.split("=", 1)[1]))
                except ValueError:
                    pass
            elif h.startswith("unit of I:"):
                unit = h.split(":", 1)[1].strip()
            elif h.startswith("q ["):
                qf = _q_to_nm(h[3:h.index("]")])
            elif h.startswith("run ") and "sample" in h:
                m = re.match(r"run\s+(\S+)\s+sample\s+(.*)", h)
                if m:
                    meta.setdefault("run", m.group(1))
                    meta.setdefault("sample", m.group(2).strip())
            elif re.search(r"1/(a|å|angstrom)\b", h.lower()):
                qf = A_PER_NM
            continue
        try:
            rows.append([float(x) for x in re.split(r"[,\s;]+", s) if x])
        except ValueError:
            continue
    if not rows:
        raise ValueError(f"{path}: no numbers")
    n = min(len(r) for r in rows)
    a = np.array([r[:n] for r in rows], float)
    q, I = a[:, 0] * qf, a[:, 1]
    dI = a[:, 2] if n > 2 else np.full_like(I, np.nan)
    dq = a[:, 3] * qf if n > 3 else np.zeros_like(q)
    if not meta.get("sample"):
        meta["sample"] = os.path.splitext(os.path.basename(path))[0]
    return [Curve(q, I, dI, dq, str(path), fmt, meta, "", unit)]


_FLOAT = re.compile(r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?")


def _read_bersans(path, lines) -> Curve:
    meta: Dict[str, object] = {}
    sec, rows = "", []
    for ln in lines:
        s = ln.strip()
        if s.startswith("%"):
            sec = s[1:]
            continue
        if sec == "Counts":
            parts = _FLOAT.findall(s)          # fixed-width numbers may touch ("1.2e-01-3.4e-02")
            if parts:
                rows.append([float(x) for x in parts])
        elif "=" in s:
            k, v = s.split("=", 1)
            v = v.lstrip("=").strip()
            x = _num(v)
            meta[f"{sec}/{k.strip()}"] = x if x is not None and not re.search(r"[A-Za-z]", v) else v
    n = min(len(r) for r in rows)
    a = np.array([r[:n] for r in rows], float)
    m = meta
    meta["sample"] = m.get("Sample/SampleName", os.path.basename(path))
    if _num(m.get("Setup/SD")) is not None:
        meta["L2_m"] = _num(m["Setup/SD"])
    if _num(m.get("Setup/Lambda")) is not None:
        meta["wavelength_A"] = 10.0 * _num(m["Setup/Lambda"])
    if _num(m.get("Setup/Collimation")) is not None:
        meta["L1_m"] = _num(m["Setup/Collimation"])
    meta["run"] = re.sub(r"\D", "", os.path.basename(path).split(".")[0]) or None
    dq = a[:, 3] if n > 3 else np.zeros(len(a))
    return Curve(a[:, 0], a[:, 1], np.abs(a[:, 2]) if n > 2 else np.full(len(a), np.nan), dq,
                 str(path), "bersans", meta, "", "1/cm/sr (BerSANS)")


def read_curves(path) -> List[Curve]:
    """All curves of a file (NXcanSAS: one per entry)."""
    path = str(path)
    with open(path, "rb") as fh:
        sig = fh.read(8)
    if sig.startswith(b"\x89HDF"):
        return _read_nxcansas(path)
    return _read_text(path)


def curve_from_reduced(reduced, key=None) -> Curve:
    """A pySASred result in memory (:class:`~pysasfit.sasred.reduce.ReducedRun`, e.g. a row of the
    reduction table) as a :class:`Curve`, without writing a file. The header items are those that
    would be written to the file (:func:`pysasfit.sasred.io.metadata.run_metadata`)."""
    from ..sasred.io.metadata import run_metadata
    r = reduced
    raw = r.raw
    meta = run_metadata(raw)
    par = dict(getattr(r, "parameters", {}) or {})
    meta["sample"] = par.get("sample") or getattr(raw, "sample_name", "")
    if par.get("runs"):
        meta["runs"] = ", ".join(str(x) for x in par["runs"])
    if par.get("label"):
        meta["label"] = par["label"]
    q = np.asarray(r.q, float) * A_PER_NM                       # 1/Å -> 1/nm
    dq = np.asarray(r.q_sigma, float) * A_PER_NM
    tag = key if key is not None else (par.get("label") or r.run)
    return Curve(q, np.asarray(r.I, float), np.asarray(r.idev(), float), dq,
                 f"pySASred:{tag}", "pySASred (memory)", meta, "", getattr(r.intensity, "unit", ""))


def read_folder(folder, patterns=("*.h5", "*.nxs", "*.dat", "*.txt", "*.0[0-9][0-9]")) -> Tuple[List[Curve], List[str]]:
    """Curves of every readable file in a folder; returns (curves, problems)."""
    import glob
    curves, bad = [], []
    files = sorted({p for pat in patterns for p in glob.glob(os.path.join(folder, pat))})
    for p in files:
        try:
            curves += read_curves(p)
        except Exception as exc:  # noqa: BLE001
            bad.append(f"{os.path.basename(p)}: {exc}")
    return curves, bad


# ------------------------------------------------------------------------------------- groups
@dataclass
class CurveGroup:
    key: str                                   # sample key from the regular expression
    values: Dict[str, object]                  # field -> value or (lo, hi)
    curves: List[Curve]
    notes: List[str] = field(default_factory=list)

    @property
    def label(self) -> str:
        s = [self.key]
        for k, v in self.values.items():
            s.append(f"{_short(k)} {_fmt_val(v)}")
        return ", ".join(s)

    def value(self, key) -> Optional[float]:
        v = self.values.get(key)
        if isinstance(v, tuple):
            return 0.5 * (v[0] + v[1])
        return _num(v)

    def geometries(self) -> List[str]:
        return list(dict.fromkeys(c.geometry for c in sorted(self.curves, key=lambda c: c.q_range()[0])))


def _short(k: str) -> str:
    return k.split("/")[-1] if k.startswith(("hdf:", "Sample/", "Setup/", "sample/")) else k


def _fmt_val(v) -> str:
    if isinstance(v, tuple):
        a, b = v
        return f"{a:.4g}" if abs(b - a) <= 1e-12 * max(1.0, abs(a)) else f"{a:.4g}–{b:.4g}"
    if isinstance(v, float):
        return f"{v:.4g}"
    return str(v)


def sample_key(name: str, regex: Optional[str]) -> Optional[str]:
    """Key of a sample name: the capture groups of the regular expression (joined by "_"), or the
    whole match; None when it does not match. No expression: the name itself."""
    if not regex:
        return name
    m = re.search(regex, name or "")
    if not m:
        return None
    g = [x for x in m.groups() if x is not None]
    return "_".join(g) if g else m.group(0)


def group_curves(curves: Sequence[Curve], sample_regex: Optional[str] = None,
                 fields: Optional[Dict[str, object]] = None) -> Tuple[List[CurveGroup], List[Curve]]:
    """Groups of curves to be merged: same sample key (:func:`sample_key`) and, for every field,
    values within its tolerance (None: equal; a number: largest spread; "1%": relative).
    Returns (groups, curves whose name does not match)."""
    from ..sasred.grouping import _split
    groups: Dict[str, List[Curve]] = {}
    rest = []
    for c in curves:
        k = sample_key(c.sample, sample_regex)
        if k is None:
            rest.append(c)
        else:
            groups.setdefault(k, []).append(c)
    out = []
    for key, cs in groups.items():
        parts = [({}, cs)]
        for f, tol in (fields or {}).items():
            new = []
            for vals, sub in parts:
                for v, ss in _split(sub, f, tol if tol not in ("", 0) else None,
                                    lambda c, f=f: _numeric_or_text(value_of(c, f), f)):
                    new.append((dict(vals, **{f: v}), ss))
            parts = new
        for vals, ss in parts:
            g = CurveGroup(key, vals, sorted(ss, key=lambda c: c.q_range()[0]))
            geo = [c.geometry for c in g.curves]
            for gname in set(geo):
                n = geo.count(gname)
                if n > 1:
                    g.notes.append(f"{n} curves of geometry {gname}: averaged where they overlap; "
                                   "add a grouping item to separate them (e.g. hdf:sample/position ± 1 "
                                   "for a sample measured at two changer positions)")
            units = {c.I_unit for c in g.curves if c.I_unit}
            if len(units) > 1:
                g.notes.append("different intensity units: " + ", ".join(sorted(units)))
            out.append(g)
    out.sort(key=lambda g: (g.key, [_sortable(g.values[k]) for k in g.values]))
    return out, rest


_TEXT_KEYS = ("sample", "file", "geometry", "format", "title")


def _numeric_or_text(v, key=""):
    if v is None:
        return None
    if key in _TEXT_KEYS or isinstance(v, str) and _num(v) is None:
        return str(v)
    x = _num(v)
    return x if x is not None else str(v)


def _sortable(v):
    if isinstance(v, tuple):
        return (0, v[0], "")
    x = _num(v)
    return (0, x, "") if x is not None else (1, 0.0, str(v))


# ------------------------------------------------------------------------------------- merging
@dataclass
class Overlap:
    low: str                       # geometry with the lower q
    high: str
    q_range: Tuple[float, float]
    n: int
    factor: float                  # I_high / I_low in the overlap (weighted mean)
    sigma: float
    chi2_ndf: float                # of the difference, with factor 1

    @property
    def z(self) -> float:
        return (self.factor - 1.0) / self.sigma if self.sigma > 0 else np.inf

    def text(self) -> str:
        return (f"{self.high} / {self.low}: {self.factor:.4f} ± {self.sigma:.4f} "
                f"({100 * (self.factor - 1):+.1f} %, z {self.z:+.1f}) over q {self.q_range[0]:.4g}–"
                f"{self.q_range[1]:.4g} 1/nm, {self.n} points, χ²/ndf {self.chi2_ndf:.2f}")


@dataclass
class Merged:
    q: np.ndarray
    I: np.ndarray
    dI: np.ndarray
    dq: np.ndarray
    source: np.ndarray             # index of the geometry, -1 = averaged in an overlap
    geometries: List[str]
    group: Optional[CurveGroup] = None
    overlaps: List[Overlap] = field(default_factory=list)
    scales: Dict[str, float] = field(default_factory=dict)
    cuts: Dict[str, Tuple[Optional[float], Optional[float]]] = field(default_factory=dict)
    notes: List[str] = field(default_factory=list)
    method: str = "weighted"
    I_unit: str = ""


def _prepared(c: Curve, cut, drop) -> Tuple[np.ndarray, ...]:
    o = np.argsort(c.q)
    q, I, dI, dq = c.q[o], c.I[o], c.dI[o], c.dq[o]
    ok = np.isfinite(q) & np.isfinite(I) & (q > 0)
    ok &= np.isfinite(dI) & (dI > 0)
    if drop:
        idx = np.flatnonzero(ok)
        a, b = drop
        ok[idx[:a]] = False
        if b:
            ok[idx[len(idx) - b:]] = False
    if cut:
        lo, hi = cut
        if lo is not None:
            ok &= q >= lo
        if hi is not None:
            ok &= q <= hi
    return q[ok], I[ok], dI[ok], np.nan_to_num(dq[ok])


def _interp(qa, qb, Ib, dIb):
    """Curve b at the q of a (log-log where positive)."""
    if np.all(Ib > 0):
        lb = np.log(Ib)
        v = np.exp(np.interp(np.log(qa), np.log(qb), lb))
    else:
        v = np.interp(qa, qb, Ib)
    s = np.interp(qa, qb, dIb)
    return v, s


def _overlap(ga, a, gb, b) -> Optional[Overlap]:
    qa, Ia, dIa = a[0], a[1], a[2]
    qb, Ib, dIb = b[0], b[1], b[2]
    lo, hi = max(qa.min(), qb.min()), min(qa.max(), qb.max())
    if hi <= lo:
        return None
    sel = (qa >= lo) & (qa <= hi)
    if sel.sum() < 2:
        return None
    vb, sb = _interp(qa[sel], qb, Ib, dIb)
    va, sa = Ia[sel], dIa[sel]
    r = vb / va
    sr = np.abs(r) * np.sqrt((sb / vb) ** 2 + (sa / va) ** 2)
    good = np.isfinite(r) & (sr > 0)
    if good.sum() < 2:
        return None
    w = 1 / sr[good] ** 2
    f = float(np.sum(w * r[good]) / np.sum(w))
    sf = float(1 / np.sqrt(np.sum(w)))
    d = (vb - va)[good] / np.sqrt(sb ** 2 + sa ** 2)[good]
    return Overlap(ga, gb, (float(lo), float(hi)), int(good.sum()), f, sf, float(np.sum(d ** 2) / max(1, good.sum() - 1)))


def merge(curves: Sequence[Curve], cuts: Optional[Dict[str, Tuple[Optional[float], Optional[float]]]] = None,
          drop: Optional[Dict[str, Tuple[int, int]]] = None, method: str = "weighted",
          scale_to: Optional[str] = None, group: Optional[CurveGroup] = None) -> Merged:
    """Merge the curves of one group (see the module text). ``cuts``/``drop`` per geometry label;
    ``scale_to``: geometry label to which the others are scaled (chained through the overlaps)."""
    cuts, drop = dict(cuts or {}), dict(drop or {})
    notes: List[str] = []
    data, geos = [], []
    for c in sorted(curves, key=lambda c: c.q_range()[0]):
        p = _prepared(c, cuts.get(c.geometry), drop.get(c.geometry))
        if len(p[0]) == 0:
            notes.append(f"{c.name}: no points left after the cuts")
            continue
        data.append(p)
        geos.append(c.geometry)
    if not data:
        raise ValueError("no points to merge")
    order = np.argsort([d[0].min() for d in data])
    data, geos = [data[i] for i in order], [geos[i] for i in order]
    # overlaps of neighbours (by q), before scaling
    overlaps = []
    for i in range(len(data) - 1):
        o = _overlap(geos[i], data[i], geos[i + 1], data[i + 1])
        if o is None:
            if data[i + 1][0].min() > data[i][0].max():
                notes.append(f"no overlap between {geos[i]} and {geos[i + 1]}: gap "
                             f"{data[i][0].max():.4g}–{data[i + 1][0].min():.4g} 1/nm")
        else:
            overlaps.append(o)
    scales = {g: 1.0 for g in geos}
    if scale_to:
        if scale_to not in geos:
            notes.append(f"reference geometry {scale_to} not in this group: not scaled")
        else:
            k = geos.index(scale_to)
            for i in range(k + 1, len(data)):                  # chain upwards
                o = _overlap(geos[i - 1], data[i - 1], geos[i], data[i])
                if o is None:
                    notes.append(f"{geos[i]}: no overlap with {geos[i - 1]}, not scaled")
                    break
                s = 1.0 / o.factor
                data[i] = (data[i][0], data[i][1] * s, data[i][2] * s, data[i][3])
                scales[geos[i]] = scales[geos[i]] * s
            for i in range(k - 1, -1, -1):                     # and downwards
                o = _overlap(geos[i], data[i], geos[i + 1], data[i + 1])
                if o is None:
                    notes.append(f"{geos[i]}: no overlap with {geos[i + 1]}, not scaled")
                    break
                s = o.factor
                data[i] = (data[i][0], data[i][1] * s, data[i][2] * s, data[i][3])
                scales[geos[i]] = scales[geos[i]] * s
            notes.append("scaled to " + scale_to + ": " +
                         ", ".join(f"{g} × {s:.4f}" for g, s in scales.items() if g != scale_to))
    q = np.concatenate([d[0] for d in data])
    I = np.concatenate([d[1] for d in data])
    dI = np.concatenate([d[2] for d in data])
    dq = np.concatenate([d[3] for d in data])
    src = np.concatenate([np.full(len(d[0]), i) for i, d in enumerate(data)])
    if method == "all" or len(data) == 1:
        o = np.argsort(q, kind="stable")
        res = (q[o], I[o], dI[o], dq[o], src[o])
    else:
        res = _weighted(q, I, dI, dq, src, data)
    m = Merged(*res, geometries=geos, group=group, overlaps=overlaps, scales=scales,
               cuts={g: cuts.get(g, (None, None)) for g in geos}, notes=notes, method=method,
               I_unit=next((c.I_unit for c in curves if c.I_unit), ""))
    return m


def _weighted(q, I, dI, dq, src, data):
    """Outside the overlaps keep the points; inside average on the bins of the coarsest curve."""
    n = len(data)
    lo = np.array([d[0].min() for d in data])
    hi = np.array([d[0].max() for d in data])
    cover = np.array([np.sum((qq >= lo) & (qq <= hi)) for qq in q])
    single = cover <= 1
    out = [(q[single], I[single], dI[single], dq[single], src[single])]
    # overlap regions: maximal intervals where at least two curves are present
    edges = np.unique(np.concatenate([lo, hi]))
    regions = []
    for a, b in zip(edges[:-1], edges[1:]):
        mid = 0.5 * (a + b)
        if np.sum((lo <= mid) & (hi >= mid)) >= 2:
            if regions and abs(regions[-1][1] - a) <= 1e-12 * max(1.0, a):
                regions[-1] = (regions[-1][0], b)
            else:
                regions.append((a, b))
    for a, b in regions:
        inside = ~single & (q >= a) & (q <= b)
        if not inside.any():
            continue
        # bins: the curve with the fewest points in this region
        counts = [np.sum((d[0] >= a) & (d[0] <= b)) for d in data]
        cand = [i for i in range(n) if counts[i] > 0]
        k = min(cand, key=lambda i: counts[i])
        qc = data[k][0][(data[k][0] >= a) & (data[k][0] <= b)]
        be = np.concatenate([[a - 1e-12], 0.5 * (qc[1:] + qc[:-1]), [b + 1e-12]])
        qi, Ii, dIi, dqi = q[inside], I[inside], dI[inside], dq[inside]
        idx = np.clip(np.searchsorted(be, qi, side="right") - 1, 0, len(be) - 2)
        for j in np.unique(idx):
            s = idx == j
            w = 1.0 / dIi[s] ** 2
            W = w.sum()
            qm = np.sum(w * qi[s]) / W
            out.append((np.array([qm]), np.array([np.sum(w * Ii[s]) / W]), np.array([1 / np.sqrt(W)]),
                        np.array([np.sqrt(np.sum(w * (dqi[s] ** 2 + (qi[s] - qm) ** 2)) / W)]),
                        np.array([-1 if len(np.unique(src[inside][s])) > 1 else src[inside][s][0]])))
    qq = np.concatenate([o[0] for o in out])
    o = np.argsort(qq, kind="stable")
    return tuple(np.concatenate([x[i] for x in out])[o] for i in range(5))


def merge_groups(groups: Sequence[CurveGroup], **kw) -> List[Merged]:
    out = []
    for g in groups:
        try:
            m = merge(g.curves, group=g, **kw)
        except ValueError as exc:
            m = Merged(*(np.array([]),) * 5, geometries=[], group=g, notes=[str(exc)])
        m.notes = list(g.notes) + m.notes
        out.append(m)
    return out


def series(merged: Sequence[Merged], key: str) -> Dict[str, List[Tuple[object, Merged]]]:
    """Merged curves per sample key, sorted by the value of ``key`` (e.g. a temperature)."""
    out: Dict[str, List[Tuple[object, Merged]]] = {}
    for m in merged:
        g = m.group
        if g is None:
            continue
        v = g.value(key)
        if v is None:
            vals = [_num(value_of(c, key)) for c in g.curves]
            vals = [x for x in vals if x is not None]
            v = float(np.mean(vals)) if vals else None
        out.setdefault(g.key, []).append((v, m))
    for k in out:
        out[k].sort(key=lambda t: (t[0] is None, t[0] if t[0] is not None else 0.0))
    return out


# ------------------------------------------------------------------------------------- output
def file_stem(m: Merged) -> str:
    g = m.group
    s = g.label if g is not None else "merged"
    s = s.replace("λ", "lambda").replace("–", "-").replace(", ", "_").replace(" ", "")
    return re.sub(r"[^A-Za-z0-9_.+-]", "_", s)


def _provenance(m: Merged) -> dict:
    g = m.group
    return dict(
        software="pyTOOLbox merge", date=datetime.now(timezone.utc).isoformat(), method=m.method,
        sample_key=g.key if g else "", group_values={k: (list(v) if isinstance(v, tuple) else v)
                                                     for k, v in (g.values.items() if g else [])},
        sources=[dict(file=c.path, entry=c.entry, geometry=c.geometry, format=c.fmt, sample=c.sample,
                      q_range=list(c.q_range())) for c in (g.curves if g else [])],
        cuts={k: list(v) for k, v in m.cuts.items()}, scales=m.scales,
        overlaps=[o.text() for o in m.overlaps], notes=m.notes, I_unit=m.I_unit)


def write_ascii(path, m: Merged, units: str = "nm") -> str:
    f = A_PER_NM if units == "A" else 1.0
    qu = "1/A" if units == "A" else "1/nm"
    p = _provenance(m)
    head = [f"pyTOOLbox merge of {len(p['sources'])} curves ({m.method})",
            f"sample key: {p['sample_key']}"]
    head += [f"{k} = {_fmt_val(tuple(v) if isinstance(v, list) else v)}" for k, v in p["group_values"].items()]
    head += [f"source: {s['file']}{(':' + s['entry']) if s['entry'] else ''}  [{s['geometry']}]" for s in p["sources"]]
    head += [f"q cut {g}: {c[0]}–{c[1]} 1/nm" for g, c in p["cuts"].items() if any(x is not None for x in c)]
    head += [f"scale {g} = {s:.6g}" for g, s in p["scales"].items() if s != 1.0]
    head += [f"overlap: {t}" for t in p["overlaps"]] + [f"note: {n}" for n in p["notes"]]
    head.append(f"unit of I: {m.I_unit}")
    head.append(f"group_json = {json.dumps(p['group_values'], default=str)}")
    head.append(f"q [{qu}]  I  dI  dq [{qu}]")
    np.savetxt(path, np.column_stack([m.q / f, m.I, m.dI, m.dq / f]), fmt="%.8e", header="\n".join(head))
    return str(path)


def _s(x):
    return np.bytes_(str(x).encode("utf-8"))


def _write_entry(f, name: str, m: Merged, units: str):
    fq = A_PER_NM if units == "A" else 1.0
    qu = "1/angstrom" if units == "A" else "1/nm"
    p = _provenance(m)
    e = f.create_group(name)
    e.attrs.update({"NX_class": _s("NXentry"), "canSAS_class": _s("SASentry"), "version": _s("1.1"),
                    "default": _s("sasdata01")})
    e["definition"] = _s("NXcanSAS")
    e["title"] = _s(m.group.label if m.group else "merged")
    d = e.create_group("sasdata01")
    d.attrs.update({"NX_class": _s("NXdata"), "canSAS_class": _s("SASdata"), "signal": _s("I"),
                    "I_axes": _s("Q"), "Q_indices": np.int32(0)})
    q = d.create_dataset("Q", data=m.q / fq)
    q.attrs.update({"units": _s(qu), "resolutions": _s("Qdev")})
    d.create_dataset("Qdev", data=m.dq / fq).attrs["units"] = _s(qu)
    i = d.create_dataset("I", data=m.I)
    i.attrs.update({"units": _s("1/cm" if "cm" in m.I_unit else "arbitrary"), "uncertainties": _s("Idev"),
                    "sasred_unit": _s(m.I_unit)})
    d.create_dataset("Idev", data=m.dI)
    d.create_dataset("source_geometry", data=m.source.astype(np.int32)).attrs["geometries"] = \
        _s(json.dumps(m.geometries))
    sm = e.create_group("sassample")
    sm.attrs.update({"NX_class": _s("NXsample"), "canSAS_class": _s("SASsample")})
    sm["name"] = _s(p["sample_key"])
    for k, v in p["group_values"].items():
        key = re.sub(r"[^A-Za-z0-9_]", "_", _short(k))
        if key in sm:
            continue
        val = 0.5 * (v[0] + v[1]) if isinstance(v, list) else v
        sm[key] = val if isinstance(val, (int, float)) else _s(val)
    pr = e.create_group("sasprocess01")
    pr.attrs.update({"NX_class": _s("NXprocess"), "canSAS_class": _s("SASprocess")})
    pr["name"] = _s("pyTOOLbox merge")
    pr["date"] = _s(p["date"])
    pr["group_values"] = _s(json.dumps(dict(sample=p["sample_key"], **{
        k: (0.5 * (v[0] + v[1]) if isinstance(v, list) else v) for k, v in p["group_values"].items()}), default=str))
    pr["provenance"] = _s(json.dumps(p, default=str))
    pr["notes"] = _s("\n".join(p["notes"] + p["overlaps"]))


def write_nxcansas(path, merged, units: str = "nm") -> str:
    """One merged curve, or a list of them (one NXentry each, e.g. a temperature series)."""
    import h5py
    ms = merged if isinstance(merged, (list, tuple)) else [merged]
    with h5py.File(path, "w") as f:
        names = []
        for k, m in enumerate(ms, 1):
            n = f"sasentry{k:02d}"
            _write_entry(f, n, m, units)
            names.append(n)
        f.attrs["default"] = names[0] if names else ""
    return str(path)
