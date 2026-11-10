# SPDX-License-Identifier: BSD-3-Clause
"""The reduction plan: which runs play which role, which were summed and which were left out.

For every instrument configuration (detector distance, wavelength, collimation) the runs are
grouped by sample name and kind (scattering or transmission = direct beam with the beam stop moved
out, usually attenuated), each group
gets a role

* ``sample``: S (any other name);
* ``sample_background``: EB / SB, container or substrate of the sample (assign explicitly);
* ``water``: H2O for the efficiency and the absolute scale, W (``H2O``, ``water``);
* ``empty_cell``: EC / WB, the empty water cell (``EC``, ``empty cell``);
* ``empty_beam``: EH, open beam, transmission reference (``EB``, ``empty beam``, ``open beam``);
* ``blocked_beam``: Cd, blocked beam / electronic background (``Cd``, ``B4C``, ``blocked``);

and its repeated runs are checked before summing: scattering runs with :func:`grouping.check_runs`
(when a beam centre and q bins are given), transmission runs by their count rate per monitor.
:meth:`Plan.markdown` gives the table to read, :meth:`Plan.save` writes it as JSON next to the
reduced data, so it can always be checked later which files were summed and which were outliers.

``EB`` means *empty beam* at SANS-I (``trans.txt`` gives it T = 1); the sample background (empty
container of the samples) must be assigned explicitly with ``roles={"EC": "sample_background"}``
when the samples sit in the same cells as the water.
"""
from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional

import numpy as np
from scipy import stats

from . import grouping as gr

__all__ = ["ROLES", "PATTERNS", "Entry", "Plan", "build_plan", "check_rates"]

ROLES = ["sample", "sample_background", "water", "empty_cell", "empty_beam", "blocked_beam"]
PATTERNS = {
    "water": r"^(h2o|water)\b",
    "empty_cell": r"^(ec|empty[ _-]?cell)\b",
    "empty_beam": r"^(eb|empty[ _-]?beam|open[ _-]?beam|eh|empty[ _-]?hole)\b",
    "blocked_beam": r"^(cd|b4c|cadmium|blocked([ _-]?beam)?)\b",
}
#: roles the absolute reduction needs for every configuration (scattering / transmission)
NEEDED = {"scattering": ["water", "empty_cell", "blocked_beam"], "transmission": ["empty_beam"]}


def role_of(name: str, roles: Optional[Dict[str, str]] = None) -> str:
    if roles and name in roles:
        return roles[name]
    for role, pat in PATTERNS.items():
        if re.search(pat, name.strip(), re.IGNORECASE):
            return role
    return "sample"


@dataclass
class Entry:
    """One group of runs: same configuration, sample and kind."""
    config: str
    role: str
    sample: str
    kind: str                     # scattering | transmission
    runs: List[int]
    summed: List[int]             # runs to be summed
    outliers: List[int]
    verdict: str                  # single | equivalent | outlier | ... | not checked
    message: str = ""
    chi2_ndf: Optional[float] = None
    start: str = ""
    end: str = ""
    run_notes: Dict[str, str] = field(default_factory=dict)   # run number (str) -> why / what

@dataclass
class Plan:
    entries: List[Entry]
    missing: List[str]            # e.g. "8.0 m, 6.0 Å: no blocked_beam (scattering)"
    match: dict = field(default_factory=dict)
    beamstop: Dict[str, str] = field(default_factory=dict)   # per configuration
    skipped: List[tuple] = field(default_factory=list)       # (run, why): time 0, 0 counts or monitor 0

    def rows(self, role: Optional[str] = None, config: Optional[str] = None) -> List[Entry]:
        return [e for e in self.entries if (role is None or e.role == role)
                and (config is None or e.config == config)]

    def markdown(self) -> str:
        out = []
        for cfg in dict.fromkeys(e.config for e in self.entries):
            out.append(f"### {cfg}\n")
            if cfg in self.beamstop:
                out.append(f"Beam stop: {self.beamstop[cfg]}.\n")
            out.append("| role | sample | kind | runs | summed | outliers | check | notes |")
            out.append("|---|---|---|---|---|---|---|---|")
            for e in sorted(self.rows(config=cfg), key=lambda e: (ROLES.index(e.role), e.sample, e.kind)):
                chk = e.verdict + (f" (χ²/ndf {e.chi2_ndf:.2f})" if e.chi2_ndf is not None else "")
                out.append(f"| {e.role} | {e.sample} | {e.kind} | {_runs(e.runs)} | {_runs(e.summed)} | "
                           f"{_runs(e.outliers)} | {chk} | {e.message} |")
            miss = [m for m in self.missing if m.startswith(cfg)]
            if miss:
                out.append("")
                out.extend(f"* missing: {m[len(cfg) + 2:]}" for m in miss)
            out.append("")
        if self.skipped:
            out.append("### Skipped runs\n")
            out.extend(f"* {r}: {why}" for r, why in self.skipped)
            out.append("")
        return "\n".join(out)

    def save(self, path) -> str:
        with open(path, "w", encoding="utf-8") as fh:
            json.dump({"match": self.match, "missing": self.missing, "beamstop": self.beamstop,
                       "skipped": [list(x) for x in self.skipped],
                       "entries": [asdict(e) for e in self.entries]}, fh, indent=1, default=str)
        return str(path)

    @classmethod
    def load(cls, path) -> "Plan":
        with open(path, encoding="utf-8") as fh:
            d = json.load(fh)
        return cls([Entry(**e) for e in d["entries"]], d["missing"], d.get("match", {}), d.get("beamstop", {}),
                   [tuple(x) for x in d.get("skipped", [])])


def _runs(rs) -> str:
    """82059, 82060, 82061, 82065 -> '82059–82061, 82065'."""
    rs = sorted(rs)
    if not rs:
        return "–"
    parts, a, b = [], rs[0], rs[0]
    for r in rs[1:]:
        if r == b + 1:
            b = r
            continue
        parts.append(f"{a}" if a == b else f"{a}–{b}")
        a = b = r
    parts.append(f"{a}" if a == b else f"{a}–{b}")
    return ", ".join(parts)


def check_rates(raws, alpha: float = 0.01, monitor: str = "monitor1", rel_floor: float = 0.0,
                details: bool = False):
    """Repeated transmission (direct-beam) runs: total counts per monitor with Poisson errors.
    Returns (verdict, outlier run numbers, χ²/ndf), with ``details`` also {run: (relative
    deviation from the mean of the kept runs, deviation in σ)}."""
    c = np.array([float(np.sum(r.counts)) for r in raws])
    from .reduce import load_instrument

    def mon(r):
        if getattr(r, "instrument", "sans_i") == "sans_i":
            return monitor
        return load_instrument(r)["monitor"]["normaliser"]
    m = np.array([float(r.monitors[mon(r)]) for r in raws])
    x, s = c / m, np.sqrt(np.maximum(c, 1)) / m
    s = np.hypot(s, rel_floor * x)
    keep = list(range(len(raws)))
    out = []
    while len(keep) > 2:
        w = 1 / s[keep]**2
        mean = np.sum(w * x[keep]) / w.sum()
        z = np.abs(x[keep] - mean) / s[keep]
        k = int(np.argmax(z))
        if stats.norm.sf(z[k]) * 2 * len(keep) >= alpha:
            break
        out.append(keep.pop(k))
    w = 1 / s[keep]**2
    mean = np.sum(w * x[keep]) / w.sum()
    chi2 = float(np.sum(w * (x[keep] - mean)**2))
    ndf = max(len(keep) - 1, 1)
    ok = stats.chi2.sf(chi2, ndf) >= alpha
    dev = {raws[k].run: (float(x[k] / mean - 1), float((x[k] - mean) / s[k])) for k in range(len(raws))}
    if len(raws) == 2 and not ok:
        res = ("differ", [], chi2 / ndf)
    elif out and ok:
        res = ("outlier", [raws[k].run for k in out], chi2 / ndf)
    else:
        res = (("equivalent" if ok else "inconsistent"), [], chi2 / ndf)
    return res + (dev,) if details else res


def _two_args(f) -> bool:
    import inspect
    try:
        return len(inspect.signature(f).parameters) >= 2
    except (TypeError, ValueError):
        return False


def _join(a: Optional[str], b: str) -> str:
    return b if not a else f"{a}; {b}"


def _config(r) -> str:
    s = f"SD {r.SD_m:.1f} m, λ {r.wavelength_A:.2f} Å, coll {r.coll_m:.1f} m"
    p = getattr(r, "panel_SD_m", None)
    return s + (f", panels {p:.2f} m" if p is not None else "")


def config_base(cfg: str) -> str:
    """A configuration text without the distance of the SANS-LLB wide-angle panels (as written before
    2026-11-10): tables and exclusions saved then still find their configuration."""
    import re as _re
    return _re.sub(r", panels [-0-9.]+ m$", "", cfg or "")


def build_plan(runs, roles: Optional[Dict[str, str]] = None, match: Optional[dict] = None,
               transmission=None, beam_center_px=None, edges=None,
               mask=None, efficiency=None, alpha: float = 0.01, rel_floor: float = 0.0,
               rate_floor: float = 0.005, fields=None) -> Plan:
    """Roles, groups and checks for a folder of runs (catalogue entries or raw runs).

    Transmission runs are found from the distance of the beam stop to the beam centre
    (:func:`catalogue.classify`); runs where that and the image disagree are noted.
    ``match``: the grouping precisions (default: sample name, SD 1 %, λ 2 %, coll 1 %, attenuator,
    sample-changer position 0.5 mm). ``transmission(raw)`` tells transmission (direct-beam) runs
    from scattering runs (default: :func:`catalogue.is_transmission`, beam stop moved out of the
    beam; the attenuator does not decide). Scattering groups are checked with
    :func:`grouping.check_runs` when ``beam_center_px`` and ``edges`` are given (``beam_center_px``
    may be a function of the first raw run of a group, to use the centre of its geometry; ``mask``
    may then take (raw, centre)); transmission
    groups by their count rate with ``rate_floor`` (0.5 %) for the beam stability (``edges`` and
    ``mask`` may be functions of the first raw run of a group, e.g. to depend on SD)."""
    skipped = [(r.run, r.empty_reason) for r in (getattr(x, "raw", x) for x in runs)
               if r is not None and getattr(r, "empty_reason", "")]
    runs = [x for x in runs if getattr(x, "raw", x) is None or not getattr(getattr(x, "raw", x), "empty_reason", "")]
    raws0 = [getattr(r, "raw", r) for r in runs]
    raws0 = [r for r in raws0 if r is not None]
    geometry_notes: Dict[str, str] = {}
    odd: Dict[int, str] = {}
    if transmission is None:
        from .catalogue import _config_key, classify
        kinds, odd_l, geo = classify(raws0)
        odd = dict(odd_l)

        def transmission(r, kinds=kinds):
            return kinds[r.run] == "transmission"
        for r in raws0:
            g = geo.get(_config_key(r))
            if g is not None:
                geometry_notes.setdefault(_config(r), g.describe())
    match = match or {"sample": None, "SD_m": "1%", "panel_SD_m": "1%", "wavelength_A": "2%", "coll_m": "1%",
                      "attenuator": None, "sample_position_mm": 0.5}
    raws = [getattr(r, "raw", r) for r in runs]
    raws = [r for r in raws if r is not None]
    entries: List[Entry] = []
    for g in gr.group_runs(raws, match, fields=fields):
        r0 = g.runs[0]
        kind = "transmission" if transmission(r0) else "scattering"
        e = Entry(_config(r0), getattr(r0, "role", "") or role_of(r0.sample_name, roles), r0.sample_name, kind,
                  g.run_numbers,
                  g.run_numbers, [], "single", start=str(r0.start_time), end=str(g.runs[-1].end_time or ""))
        bad = [f"{x}: {odd[x]}" for x in g.run_numbers if x in odd]
        for x in g.run_numbers:
            if x in odd:
                e.run_notes[str(x)] = odd[x]
        for r in g.runs:
            if getattr(r, "aborted", False):
                what = (f"{r.counting_time_s:.0f} s of a {r.preset:.0f} s preset" if r.count_mode in ("timer", "time")
                        else f"monitor {r.monitors.get(r.preset_monitor, 0):.4g} of a {r.preset:.4g} preset")
                e.run_notes[str(r.run)] = _join(e.run_notes.get(str(r.run)), f"aborted: {what}")
        ab = [r.run for r in g.runs if getattr(r, "aborted", False)]
        if ab:
            bad.append(f"aborted before the preset: {ab}")
        if bad:
            e.message = "; ".join(bad)
        pos = g.values.get("sample_position_mm")
        if isinstance(pos, tuple) and pos[0] is not None:
            e.message = f"position {pos[0]:.1f} mm" + (f"; {e.message}" if e.message else "")
        if len(g.runs) > 1:
            if kind == "transmission":
                v, outl, c, dev = check_rates(g.runs, alpha, rel_floor=rate_floor, details=True)
                e.verdict, e.outliers, e.chi2_ndf = v, outl, c
                for x, (rel, z) in dev.items():
                    why = f"rate {100 * rel:+.2f} % ({z:+.1f} σ) from the mean"
                    if x in outl:
                        why = "outlier: " + why
                    e.run_notes[str(x)] = _join(e.run_notes.get(str(x)), why)
                e.summed = [x for x in g.run_numbers if x not in outl] if v in ("equivalent", "outlier") else []
            elif beam_center_px is not None and edges is not None:
                try:
                    bc = beam_center_px(r0) if callable(beam_center_px) else beam_center_px
                    ed = edges(r0) if callable(edges) else edges
                    mk = mask(r0, bc) if callable(mask) and _two_args(mask) else (mask(r0) if callable(mask) else mask)
                    q = gr.check_runs(g, bc, ed, mask=mk, efficiency=efficiency,
                                      alpha=alpha, rel_floor=rel_floor)
                    e.verdict, e.outliers, e.summed = q.verdict, q.outliers, q.accepted
                    e.chi2_ndf, e.message = q.chi2_ndf, (e.message + "; " if e.message else "") + q.message
                    nq = len(q.q)
                    for k, x in enumerate(q.runs):
                        why = (f"χ²/ndf {q.run_chi2[k] / nq:.2f} against the others, scale {q.scale[k]:.4f}, "
                               f"t = {q.t_h[k]:.1f} h")
                        if x in q.outliers:
                            why = "outlier: " + why
                        elif x not in q.accepted:
                            why = f"not summed ({q.verdict}): " + why
                        e.run_notes[str(x)] = _join(e.run_notes.get(str(x)), why)
                except Exception as ex:  # noqa: BLE001 -- reported, not fatal
                    e.verdict, e.message = "not checked", f"{type(ex).__name__}: {ex}"
            else:
                e.verdict = "not checked"
        entries.append(e)
    # the same sample at two changer positions in one configuration: flag both groups
    seen: Dict[tuple, List[Entry]] = {}
    for e in entries:
        seen.setdefault((e.config, e.sample, e.kind), []).append(e)
    for es in seen.values():
        if len(es) > 1:
            for e in es:
                e.message = (e.message + "; " if e.message else "") + \
                    f"{len(es)} groups of this sample here (positions differ): check which is right"
    missing = []
    for cfg in dict.fromkeys(e.config for e in entries):
        here = {(e.role, e.kind) for e in entries if e.config == cfg}
        kinds = {e.kind for e in entries if e.config == cfg}
        for kind in kinds:
            for role in NEEDED[kind]:
                if (role, kind) not in here:
                    missing.append(f"{cfg}: no {role} ({kind})")
    return Plan(entries, missing, {k: v for k, v in match.items()}, geometry_notes, skipped)


def main(argv=None):  # pragma: no cover -- thin CLI
    """``python -m pysasfit.sasred.plan RAW [--centre X Y] [--json plan.json]`` prints the plan.
    With ``--centre`` the scattering groups are checked too (30 q bins, suggested masks)."""
    import argparse

    from .catalogue import scan_folder
    from .core import mask as mk
    from .core.averaging import q_edges
    ap = argparse.ArgumentParser(description="roles, summed runs and outliers of a raw-data folder")
    ap.add_argument("folder")
    ap.add_argument("--centre", nargs=2, type=float, metavar=("X", "Y"))
    ap.add_argument("--auto-centre", action="store_true",
                    help="beam centre per geometry from its direct-beam runs (fallback: --centre)")
    ap.add_argument("--json")
    ap.add_argument("--role", action="append", default=[], metavar="NAME=ROLE",
                    help="assign a role to a sample name, e.g. EC=sample_background")
    a = ap.parse_args(argv)
    roles = dict(x.split("=", 1) for x in a.role)
    kw = {}
    if a.centre or a.auto_centre:
        cat = scan_folder(a.folder)
        if a.auto_centre:
            from .catalogue import beam_centres, config_key
            cents = beam_centres([e.raw for e in cat])

            def c_of(r):
                g = cents.get(config_key(r))
                return (g.x, g.y) if g is not None else tuple(a.centre or (63.5, 63.5))
        else:
            def c_of(r):
                return tuple(a.centre)

        def edges(r):
            q_min = 4 * np.pi / r.wavelength_A * np.sin(0.5 * np.arctan(4 * r.pixel_size_m[0] / r.SD_m))
            q_max = 4 * np.pi / r.wavelength_A * np.sin(0.5 * np.arctan(60 * r.pixel_size_m[0] / r.SD_m))
            return q_edges(q_min, q_max, 30)
        kw = dict(beam_center_px=c_of, edges=edges, mask=lambda r, c: mk.suggest_mask(r.counts, c)[0])
    else:
        cat = scan_folder(a.folder)
    p = build_plan(cat, roles=roles, **kw)
    print(p.markdown())
    if a.json:
        p.save(a.json)


if __name__ == "__main__":  # pragma: no cover
    main()
