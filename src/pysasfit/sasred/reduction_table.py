# SPDX-License-Identifier: BSD-3-Clause
"""The correction matrix and the sample table of one configuration (as in BerSANS' data-reduction
window): which runs are summed for the blocked beam (Cd), the water, the water background (empty
cell), the sample background, the direct beam and every sample, with transmissions, scaling
(thickness), attenuation factor and scattering probability.

The absolute scale (:meth:`ReductionTable.scale_method`) comes from the water of this geometry or,
where water takes too long (long distances, long wavelengths), from the **direct beam** of this
geometry (attenuated, beam stop out) together with a detector **efficiency map** (usually made from
the water of a shorter distance; :attr:`ReductionTable.efficiency`).

The table can be filled automatically from a :class:`~pysasfit.sasred.plan.Plan` (runs grouped
by metadata, outliers left out with the reason in the comment of the run) and edited or filled
completely by hand. It is saved as JSON next to the reduced data.
"""
from __future__ import annotations

import datetime as _dt
import json
import re

import numpy as np
from dataclasses import asdict, dataclass, field
from typing import Dict, Iterable, List, Optional

__all__ = ["OPTIONAL", "Missing", "transmission_from_runs", "empty_beam_runs", "CORRECTIONS", "SCALE_METHODS", "RunChoice", "RowSet", "ReductionTable", "from_plan", "parse_runs", "configurations"]

#: correction rows: role -> label (BerSANS names)
CORRECTIONS = {"blocked_beam": "Cadmium", "water": "Water", "empty_cell": "Water-BG",
               "sample_background": "Sample-BG", "direct_beam": "Direct beam"}
#: how the absolute scale is made: auto (water if there is water, else direct beam + efficiency map),
#: water, direct_beam (direct beam + efficiency map), relative (no absolute scale)
SCALE_METHODS = ("auto", "water", "direct_beam", "relative")
#: correction rows that may be left out (no runs) or switched off (ReductionTable.ignored)
OPTIONAL = ("blocked_beam", "empty_cell", "sample_background")


@dataclass
class Missing:
    """An input missing for a reduction (or to be checked) and what the reduction does instead.

    ``role``: the correction role or "sample"; ``index``: the sample row (0-based) or None; ``field``: "runs",
    "transmission", "thickness", "aperture" or "efficiency" (the efficiency map of the table); ``severity``:
    "missing" (a value is assumed) or "check" (inconsistent input)."""
    role: str
    index: Optional[int]
    label: str
    field: str
    text: str
    severity: str = "missing"

    def describe(self) -> str:
        return f"{self.label}: {self.field} – {self.text}"


@dataclass
class RunChoice:
    run: int
    use: bool = True
    comment: str = ""


@dataclass
class RowSet:
    """One row: a set of runs to be summed and its parameters."""
    role: str                                  # sample | water | empty_cell | ...
    label: str                                 # "Water", "No. 3", ...
    info: str = ""                             # sample name
    runs: List[RunChoice] = field(default_factory=list)
    trans_runs: List[RunChoice] = field(default_factory=list)
    transmission: Optional[float] = None
    transmission_sigma: Optional[float] = None
    transmission_source: str = ""
    scaling: Optional[float] = None            # thickness [cm]
    aperture_cm2: Optional[float] = None       # illuminated area (sample aperture) [cm²]; None: same for sample and water
    attenuator: float = 1.0                    # A(S) of BerSANS (≥ 1)
    probability: float = 0.0                   # p_cS
    notes: str = ""

    @property
    def used(self) -> List[int]:
        return [c.run for c in self.runs if c.use]

    @property
    def number(self) -> str:
        n, u = len(self.runs), len(self.used)
        return f"{u}" if u == n else f"{u} of {n}"

    def runs_text(self) -> str:
        from .plan import _runs
        return _runs(self.used) if self.runs else ""

    def set_runs(self, runs: Iterable[int], use: bool = True, comment: str = ""):
        have = {c.run for c in self.runs}
        for r in runs:
            if r not in have:
                self.runs.append(RunChoice(int(r), use, comment))
        self.runs.sort(key=lambda c: c.run)


@dataclass
class ReductionTable:
    config: str = ""
    folder: str = ""
    corrections: Dict[str, RowSet] = field(default_factory=dict)
    samples: List[RowSet] = field(default_factory=list)
    created: str = ""
    source: str = "manual"                     # manual | plan
    water_g: Optional[float] = None            # g(λ) of the water by hand; None: instrument file (absolute.water_g)
    scale: str = "auto"                        # one of SCALE_METHODS
    efficiency: str = ""                       # efficiency map: "file.h5", "file.h5::entry" or "geometry:<config>"
    direct_beam_radius_mm: Optional[float] = None   # direct beam summed within this radius; None: automatic
    direct_beam_attenuator: Optional[float] = None  # attenuator transmission of the direct beam by hand; None: file
    efficiency_mask_filled: bool = False       # mask the pixels the efficiency map only estimates (row × column)
    ignored: List[str] = field(default_factory=list)   # optional correction roles switched off (OPTIONAL)

    def __post_init__(self):
        for role, label in CORRECTIONS.items():
            self.corrections.setdefault(role, RowSet(role, label))
        if not self.created:
            self.created = _dt.datetime.now().isoformat(timespec="seconds")

    # -------------------------------------------------------------- editing
    def add_sample(self, info: str = "") -> RowSet:
        r = RowSet("sample", f"No. {len(self.samples) + 1}", info)
        self.samples.append(r)
        return r

    def add_groups(self, groups, transmissions=None, transmission_runs=None, check=None) -> List[RowSet]:
        """One sample row per :class:`~pysasfit.sasred.grouping.RunGroup` (from
        :func:`~pysasfit.sasred.selection.group_selected`), its runs all used. ``transmissions``: rows
        of trans.txt, matched by sample name; ``transmission_runs(row)``: the transmission runs of a
        row; ``check(group)``: an :class:`~pysasfit.sasred.grouping.Equivalence` (or None) whose
        outliers are switched off with their reason."""
        trans = {t.sample: t for t in transmissions or []}
        rows = []
        for g in groups:
            names = sorted({r.sample_name for r in g.runs})
            extra = []
            for k, v in g.values.items():
                if k == "sample":
                    continue
                if isinstance(v, tuple) and v[0] is not None:
                    m = 0.5 * (v[0] + v[1])
                    extra.append(f"{k.replace('hdf:', '')} {m:.4g}")
                elif v is not None:
                    extra.append(f"{k.replace('hdf:', '')} {v}")
            row = self.add_sample(" / ".join(names) + (" · " + ", ".join(extra) if extra else ""))
            row.runs = [RunChoice(r.run, True, "") for r in g.runs]
            t = trans.get(names[0]) if len(names) == 1 else None
            if t is not None:
                row.transmission, row.transmission_sigma = t.value, t.sigma
                row.transmission_source = f"trans.txt ({t.file})"
            thk = getattr(g.runs[0], "sample_thickness_cm", None)
            row.scaling = float(thk) if thk else None
            if transmission_runs is not None:
                row.trans_runs = [RunChoice(int(x), True, "") for x in transmission_runs(row, g)]
            if check is not None and len(g.runs) > 1:
                try:
                    e = check(g)
                except Exception as ex:  # noqa: BLE001
                    row.notes = f"not checked: {type(ex).__name__}: {ex}"
                    e = None
                if e is not None:
                    nq = len(e.q)
                    for c in row.runs:
                        k = e.runs.index(c.run)
                        why = f"χ²/ndf {e.run_chi2[k] / nq:.2f} against the others, scale {e.scale[k]:.4f}"
                        if c.run in e.outliers:
                            c.use, why = False, "outlier: " + why
                        c.comment = why
                    if e.verdict not in ("equivalent", "outlier"):
                        row.notes = f"decision needed: {e.verdict}. {e.message}"
                    else:
                        row.notes = e.message
            if len(names) > 1:
                row.notes = (row.notes + "; " if row.notes else "") + f"several sample names: {', '.join(names)}"
            rows.append(row)
        return rows

    def remove_sample(self, i: int):
        del self.samples[i]
        self._renumber()

    def clear(self):
        self.samples = []
        self.corrections = {}
        self.__post_init__()

    def _renumber(self):
        for k, r in enumerate(self.samples):
            r.label = f"No. {k + 1}"

    def copy_value(self, i: int, attr: str, direction: str = "below"):
        """Copy the value of sample row ``i`` to all rows below (or above), like BerSANS."""
        v = getattr(self.samples[i], attr)
        rows = self.samples[i + 1:] if direction == "below" else self.samples[:i]
        for r in rows:
            setattr(r, attr, v)

    def rows(self) -> List[RowSet]:
        return [self.corrections[k] for k in CORRECTIONS] + list(self.samples)

    def uses(self, role: str) -> bool:
        """The correction row is used: it has runs switched on and is not switched off (:attr:`ignored`)."""
        return bool(self.corrections[role].used) and role not in self.ignored

    def scale_method(self) -> str:
        """The absolute scale actually used: "water", "direct_beam" or "relative". With ``scale = "auto"``:
        the direct beam whenever an efficiency map is chosen (the water of this geometry is then not used),
        else water when the water row has runs, else relative."""
        s = self.scale or "auto"
        if s not in SCALE_METHODS:
            raise ValueError(f"scale = {s!r}: one of {', '.join(SCALE_METHODS)}")
        if s != "auto":
            return s
        if self.efficiency:
            return "direct_beam"
        if self.corrections["water"].used:
            return "water"
        return "relative"

    def missing_inputs(self, rows: Optional[Iterable[RowSet]] = None) -> List[Missing]:
        """Inputs missing for the reduction of ``rows`` (default: all sample rows with runs), with what the
        reduction does instead – the same cases as :func:`~pysasfit.sasred.absolute.reduce_row_2d`."""
        out: List[Missing] = []
        corr = self.corrections
        method = self.scale_method()
        rows = [r for r in (self.samples if rows is None else rows)]

        def c(role, field, text, severity="missing"):
            out.append(Missing(role, None, CORRECTIONS[role], field, text, severity))

        if "blocked_beam" in self.ignored:
            c("blocked_beam", "use", "switched off: no beam-independent background subtracted", "info")
        elif not corr["blocked_beam"].used:
            c("blocked_beam", "runs", "no Cd: no beam-independent background subtracted", "info")
        sb = corr["sample_background"]
        if "sample_background" in self.ignored:
            c("sample_background", "use", "switched off: no sample background subtracted", "info")
        elif not sb.used:
            c("sample_background", "runs", "no sample background: none subtracted", "info")
        elif sb.transmission is None:
            c("sample_background", "transmission", "no transmission: T = 1 assumed")
        if method == "water":
            w, ec = corr["water"], corr["empty_cell"]
            if w.transmission is None:
                c("water", "transmission", "no water transmission: no absolute factor, result in units of water")
            if "empty_cell" in self.ignored:
                c("empty_cell", "use", "switched off: the water is not corrected for its cell", "info")
            elif not ec.used:
                c("empty_cell", "runs", "no water background: the water is not corrected for its cell", "check")
            elif ec.transmission is None:
                c("empty_cell", "transmission", "no transmission: T = 1 assumed (τ_W = T_W)")
        elif method == "direct_beam":
            if not corr["direct_beam"].used:
                c("direct_beam", "runs", "no direct-beam runs: the reduction stops")
            if not self.efficiency:
                c("direct_beam", "efficiency", "no efficiency map: the reduction stops")
        elif (self.scale or "auto") == "auto":
            c("water", "runs", "no water (and no direct beam + efficiency map): relative scale only, not absolute")
            if corr["direct_beam"].used and not self.efficiency:
                c("direct_beam", "efficiency", "direct beam without an efficiency map: not used")
        ref = corr["water"] if method == "water" else corr["direct_beam"] if method == "direct_beam" else None
        for r in rows:
            if r not in self.samples:
                continue
            i = self.samples.index(r)

            def m(field, text, severity="missing"):
                out.append(Missing("sample", i, f"{r.label} ({r.info})" if r.info else r.label, field, text, severity))
            if r.runs and not r.used:
                m("runs", "all runs switched off: not reduced")
                continue
            if not r.used:
                continue
            if r.transmission is None:
                m("transmission", "no transmission: T = 1 assumed")
            if not r.scaling:
                m("thickness", "no thickness: d = 1 cm assumed")
            if ref is not None and bool(r.aperture_cm2) != bool(ref.aperture_cm2):
                m("aperture", f"aperture given for only one of sample and {CORRECTIONS[ref.role].lower()}: "
                              "same illuminated area assumed", "check")
                if not ref.aperture_cm2:
                    c(ref.role, "aperture", "aperture missing (the samples have one)", "check")
        # each correction item once
        seen, uniq = set(), []
        for x in out:
            k = (x.role, x.index, x.field)
            if k not in seen:
                seen.add(k)
                uniq.append(x)
        return uniq

    def problems(self) -> List[str]:
        """What is still missing before an absolute reduction."""
        out = []
        method = self.scale_method()
        if method == "water" or (method == "relative" and self.scale == "auto"):
            if not self.corrections["water"].used:
                out.append("Water: no runs")
            elif not self.uses("empty_cell") and "empty_cell" not in self.ignored:
                out.append("Water-BG: no runs")
            if method == "relative" and not self.efficiency:
                out.append("or: direct beam + efficiency map")
        elif method == "direct_beam":
            if not self.corrections["direct_beam"].used:
                out.append("Direct beam: no runs")
            if not self.efficiency:
                out.append("no efficiency map for the direct-beam scale")
        for r in self.rows():
            if r.role in self.ignored:
                continue
            if r.used and r.role not in ("blocked_beam", "direct_beam") and r.transmission is None:
                if r.role in ("water", "empty_cell") and method != "water":
                    continue
                out.append(f"{r.label} ({r.info}): no transmission")
            if r.runs and not r.used and r.role not in OPTIONAL:
                out.append(f"{r.label} ({r.info}): all runs switched off")
        if not self.samples:
            out.append("no samples")
        return out

    # -------------------------------------------------------------- files
    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "ReductionTable":
        def rowset(x):
            x = dict(x)
            x["runs"] = [RunChoice(**c) for c in x.get("runs", [])]
            x["trans_runs"] = [RunChoice(**c) for c in x.get("trans_runs", [])]
            known = RowSet.__dataclass_fields__
            return RowSet(**{k: v for k, v in x.items() if k in known})
        from .legacy import rename_config
        return cls(rename_config(d.get("config", "")), d.get("folder", ""),
                   {k: rowset(v) for k, v in d.get("corrections", {}).items()},
                   [rowset(v) for v in d.get("samples", [])], d.get("created", ""), d.get("source", "manual"),
                   d.get("water_g"), d.get("scale", "auto") or "auto", rename_config(d.get("efficiency", "") or ""),
                   d.get("direct_beam_radius_mm"), d.get("direct_beam_attenuator"),
                   bool(d.get("efficiency_mask_filled", False)), list(d.get("ignored") or []))

    def save(self, path) -> str:
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(self.to_dict(), fh, indent=1)
        return str(path)

    @classmethod
    def load(cls, path) -> "ReductionTable":
        with open(path, encoding="utf-8") as fh:
            return cls.from_dict(json.load(fh))


def parse_runs(text: str) -> List[int]:
    """'82187, 82190-82193 82294' -> [82187, 82190, 82191, 82192, 82193, 82294]."""
    out = []
    for a, b in re.findall(r"(\d+)\s*(?:[-–]\s*(\d+))?", text or ""):
        a = int(a)
        b = int(b) if b else a
        if b < a:
            a, b = b, a
        if b - a > 100000:
            raise ValueError(f"range {a}-{b} too long")
        out.extend(range(a, b + 1))
    return sorted(dict.fromkeys(out))


def configurations(plan) -> List[str]:
    """Configurations with scattering runs, in the order of the plan."""
    return list(dict.fromkeys(e.config for e in plan.entries if e.kind == "scattering"))


def _choices(e) -> List[RunChoice]:
    """Outliers are switched off; runs of a group that needs a decision (drift, shape change)
    stay on, the row notes say why."""
    notes = getattr(e, "run_notes", {}) or {}
    return [RunChoice(r, r not in e.outliers, notes.get(str(r), "")) for r in e.runs]


def _lam(config: str) -> str:
    m = re.search(r"λ ([0-9.]+)", config)
    return m.group(1) if m else ""


POSITION_TOL_MM = 1.0          # transmission runs and trans.txt values belong to a sample position within this
SLOT_TOL_MM = 5.0              # another sample this close (beam size): the same slot of the changer


def empty_beam_runs(raws: Dict[int, object]) -> List[int]:
    """Transmission runs of the empty beam among ``raws`` (role from the sample name or set by hand)."""
    from .catalogue import is_transmission
    from .plan import role_of
    out = []
    for k, r in raws.items():
        if r is None:
            continue
        role = getattr(r, "role", "") or role_of(r.sample_name)
        if role == "empty_beam":
            try:
                if is_transmission(r):
                    out.append(k)
            except Exception:  # noqa: BLE001
                pass
    return sorted(out)


def transmission_from_runs(runs, raws: Dict[int, object], eb_runs: Optional[List[int]] = None):
    """τ (beam spot) of transmission runs, each against an empty-beam transmission run at the same distance
    (the same attenuator preferred, then the nearest run number): (mean, σ, runs used, empty-beam run) or
    None. ``eb_runs``: candidate empty-beam runs (default: :func:`empty_beam_runs` of ``raws``)."""
    from .core.beamcentre import masked_centre_of_gravity
    from .core.scattering_strength import aperture_transmission_runs
    if eb_runs is None:
        eb_runs = empty_beam_runs(raws)
    vals = []
    for x in runs:
        s = raws.get(x)
        ebs = [raws[y] for y in eb_runs if y in raws and s is not None
               and abs(raws[y].SD_m - s.SD_m) <= 0.01 * s.SD_m
               and abs(raws[y].wavelength_A - s.wavelength_A) <= 0.02 * s.wavelength_A]
        if not ebs:
            continue
        # same attenuator first (the nominal attenuator factors then cancel), then the nearest run
        eb = min(ebs, key=lambda r: (str(r.attenuator) != str(s.attenuator), abs(r.run - x)))
        try:
            c = masked_centre_of_gravity(eb.counts)
            t = aperture_transmission_runs(s, eb, (c.x, c.y))
            vals.append((t.tau_ul, t.tau_ul_sigma, x, eb.run))
        except Exception:  # noqa: BLE001
            continue
    if not vals:
        return None
    v = np.array([a[0] for a in vals])
    sig = float(np.std(v, ddof=1) / np.sqrt(len(v))) if len(v) > 1 else float(vals[0][1])
    return float(v.mean()), sig, [a[2] for a in vals], vals[0][3], [float(a[0]) for a in vals]


def from_plan(plan, config: str, raws: Optional[Dict[int, object]] = None, transmissions=None,
              sample_background: Optional[str] = "empty_cell") -> ReductionTable:
    """Fill the table of one configuration from a plan.

    ``raws``: run -> raw run (thickness, sample position); ``transmissions``: rows of ``trans.txt``
    (:func:`~pysasfit.sasred.io.transmission.read_transmission_summary`), matched by sample name and,
    with ``raws``, by sample position (a value measured at another changer position is not used;
    transmission runs at another position neither). Without a usable trans.txt value the transmission
    is computed from the transmission runs at the same position against the nearest empty-beam
    transmission run (beam spot τ); a trans.txt value that disagrees with the runs is flagged.
    A sample whose position is that of another sample (within 5 mm) is flagged ("changer not moved
    or wrong sample name?"), as is a sample measured at several positions;
    ``sample_background``: the role whose runs fill *Sample-BG* when the plan has no sample
    background of its own (default: the empty cell, as when the samples sit in the same cells as the
    water; None leaves it empty)."""
    t = ReductionTable(config=config, source="plan")
    raws = raws or {}
    lam = _lam(config)
    trans_by_name = {}
    for row in transmissions or []:
        trans_by_name[row.sample] = row                    # the last one wins, as find_transmission
    tr_entries = [e for e in plan.entries if e.kind == "transmission" and _lam(e.config) == lam]

    def pos(run):
        r = raws.get(run)
        v = getattr(r, "sample_position_mm", None) if r is not None else None
        return None if v is None else float(v)

    def entry_pos(e):
        ps = [p for p in (pos(x) for x in e.runs) if p is not None]
        return float(np.median(ps)) if ps else None

    # sample positions of the scattering runs of this wavelength: a sample measured at a position that
    # belongs to another sample (changer not moved, wrong name) is reported
    from .grouping import _parse_time
    slots = {}                       # sample -> {position: [first, last] start time of its runs there}
    for e in plan.entries:
        if _lam(e.config) != lam or e.role != "sample":
            continue
        for x in e.runs:
            p1, r = pos(x), raws.get(x)
            if p1 is None:
                continue
            t1 = _parse_time(getattr(r, "start_time", "")) or 0.0
            w = slots.setdefault(e.sample, {}).setdefault(round(p1, 1), [t1, t1])
            w[0], w[1] = min(w[0], t1), max(w[1], t1)
    eb_runs = [x for e in plan.entries if e.kind == "transmission" and e.role == "empty_beam"
               and _lam(e.config) == lam for x in e.summed or e.runs]

    def computed_T(runs):
        return transmission_from_runs(runs, raws, eb_runs)

    def fill(row: RowSet, e):
        row.info = e.sample
        row.runs = _choices(e)
        notes = []
        if e.verdict not in ("single", "equivalent", "outlier", "not checked"):
            notes.append(f"decision needed: {e.verdict}. {e.message}")
        elif e.message:
            notes.append(e.message)
        p0 = entry_pos(e)
        tr = [x for x in tr_entries if x.sample == e.sample]
        trc = [c for x in tr for c in _choices(x)]
        if p0 is not None and trc:
            same = [c for c in trc if pos(c.run) is None or abs(pos(c.run) - p0) <= POSITION_TOL_MM]
            if len(same) < len(trc):
                other = sorted({round(pos(c.run), 1) for c in trc if c not in same})
                notes.append(f"transmission runs at other sample positions ({', '.join(f'{v:g}' for v in other)} mm) "
                             f"not used for position {p0:.1f} mm")
            trc = same
        row.trans_runs = trc
        hit = trans_by_name.get(e.sample)
        if hit is not None and p0 is not None and pos(hit.run) is not None \
                and abs(pos(hit.run) - p0) > POSITION_TOL_MM:
            notes.append(f"trans.txt value {hit.value:g} ({hit.file}) is for position {pos(hit.run):.1f} mm, "
                         f"not {p0:.1f} mm: not used")
            hit = None
        if hit is not None:
            row.transmission, row.transmission_sigma = hit.value, hit.sigma
            row.transmission_source = f"trans.txt ({hit.file})"
            if row.trans_runs and raws:
                ct = computed_T([c.run for c in row.trans_runs if c.use])
                if ct is not None and abs(ct[0] - hit.value) > max(0.01, 5 * (ct[1] + (hit.sigma or 0))):
                    row.transmission, row.transmission_sigma = ct[0], ct[1]
                    row.transmission_source = f"runs {ct[2]} / empty beam {ct[3]} (beam spot)"
                    notes.append(f"check: trans.txt gives {hit.value:g} ({hit.file}), the transmission runs "
                                 f"{ct[2]} give {ct[0]:.4f} ± {ct[1]:.4f} (beam spot, empty beam {ct[3]}): "
                                 "the value of the runs is used")
        elif row.trans_runs and raws:
            ct = computed_T([c.run for c in row.trans_runs if c.use])
            if ct is not None:
                row.transmission, row.transmission_sigma = ct[0], ct[1]
                row.transmission_source = f"runs {ct[2]} / empty beam {ct[3]} (beam spot)"
                if len(ct) > 4 and len(ct[4]) > 1 and np.std(ct[4]) > 0.03 * np.mean(ct[4]):
                    notes.append("decision needed: the transmission runs disagree (" +
                                 ", ".join(f"{x}: {v:.3f}" for x, v in zip(ct[2], ct[4])) +
                                 ") – other sample under the same name? Open the row and untick runs")
        if p0 is not None and e.role == "sample":
            mine = slots.get(e.sample, {}).get(round(p0, 1))
            others = sorted({(n, q) for n, ps in slots.items() if n != e.sample for q, w in ps.items()
                             if abs(q - p0) <= SLOT_TOL_MM and mine is not None
                             and w[0] <= mine[1] and mine[0] <= w[1]})      # same slot at the same time
            if others:
                notes.append("decision needed: sample position {:.1f} mm is that of {} – changer not moved or "
                             "wrong sample name?".format(p0, ", ".join(f"{n} ({q:g} mm)" for n, q in others)))
            if len(slots.get(e.sample, ())) > 1:
                notes.append("measured at several positions: " +
                             ", ".join(f"{v:g}" for v in sorted(slots[e.sample])) + " mm")
        row.notes = "; ".join(dict.fromkeys(notes))
        r0 = raws.get(e.runs[0])
        if r0 is not None and getattr(r0, "sample_thickness_cm", None):
            row.scaling = float(r0.sample_thickness_cm)

    for e in plan.entries:
        if e.config != config or e.kind != "scattering":
            continue
        if e.role in CORRECTIONS:
            row = t.corrections[e.role]
            if row.runs:                                   # a second group of the same role: keep the stronger one
                def weight(runs):
                    rs = [raws.get(x) for x in runs]
                    return sum(float(r.counting_time_s) for r in rs if r is not None and not getattr(r, "aborted", False))
                old_runs = [c.run for c in row.runs]
                if weight(e.runs) > weight(old_runs):
                    old_note = row.notes
                    fill(row, e)
                    row.notes = "; ".join(x for x in (row.notes, f"also runs {old_runs} (shorter or aborted) not taken",
                                                     old_note) if x)
                else:
                    row.notes = (row.notes + "; " if row.notes else "") + \
                        f"also {e.sample} runs {e.runs} (other position, attenuator or shorter?) not taken"
                continue
            fill(row, e)
        elif e.role == "sample":
            fill(t.add_sample(), e)
    db = t.corrections["direct_beam"]
    for e in plan.entries:
        if e.config == config and e.kind == "transmission" and e.role == "empty_beam":
            db.set_runs([c.run for c in _choices(e) if c.use])
            db.set_runs([c.run for c in _choices(e) if not c.use], use=False, comment="outlier")
            if not db.info:
                db.info = e.sample
    if db.runs:
        atts = sorted({str(getattr(raws.get(c.run), "attenuator", "?")) for c in db.runs if raws.get(c.run) is not None})
        db.notes = ("empty-beam transmission runs of this geometry" +
                    (f", attenuator {', '.join(atts)}" if atts else "") +
                    ": absolute scale with an efficiency map when there is no water")
    sb = t.corrections["sample_background"]
    if not sb.runs and sample_background:
        src = t.corrections.get(sample_background)
        if src is not None and src.runs:
            sb.info, sb.runs = src.info, [RunChoice(c.run, c.use, c.comment) for c in src.runs]
            sb.trans_runs = [RunChoice(c.run, c.use, c.comment) for c in src.trans_runs]
            sb.transmission, sb.transmission_sigma = src.transmission, src.transmission_sigma
            sb.transmission_source = src.transmission_source
            sb.notes = (f"taken from {CORRECTIONS[sample_background]}: change it when the samples sit "
                        "in other containers")
    return t
