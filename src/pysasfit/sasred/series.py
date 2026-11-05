# SPDX-License-Identifier: BSD-3-Clause
"""Time series of one sample: the runs of a sample row reduced one by one (or in small bins) instead of
summed, each with its time.

A sample row of the reduction table becomes a series with ``RowSet.series = True``. The runs ticked in
the row (the first to the last run of the series, set in the row's *Time series…* dialog) are sorted by
their start time and grouped into points:

* ``series_bin`` = N: N consecutive runs summed per point (1: every run is a point);
* ``series_dt_s`` = Δt: all runs whose middle lies in the same time window of Δt seconds (instead of N).

The time of a point is the count-weighted middle of its runs, t = Σ M_i t_i / Σ M_i (M: monitor or
measuring time, t_i = start + measuring time / 2), relative to t₀: ``series_t0`` (date and time, e.g.
the mixing or the temperature jump) or, when empty, the start of the first run of the series.

Every point is reduced with the corrections of its geometry exactly like a summed row; its results carry
``time_s`` (and ``time_start``, ``time_end``, ``series``, ``series_t0``, ``series_index``), written into
the output files and usable as a series key in pyTOOLbox.

Later: runs with many frames (time scans in one file) give the points in the same way (RawRun.frames).
"""
from __future__ import annotations

import datetime as _dt
from dataclasses import dataclass, field
from typing import Dict, List, Optional

__all__ = ["SeriesPoint", "parse_time", "run_times", "series_points", "SERIES_KEYS"]

#: result parameters that describe the time of a series point (also put into the files' metadata)
SERIES_KEYS = ("time_s", "time_start", "time_end", "series", "series_t0", "series_index", "series_runs")


def parse_time(text) -> Optional[_dt.datetime]:
    """'2026-09-30 01:58:02.556', '2026-09-30T01:58:02', '30.09.2026 01:58' → datetime (None if empty)."""
    if text is None:
        return None
    if isinstance(text, _dt.datetime):
        return text
    s = str(text).strip()
    if not s:
        return None
    try:
        return _dt.datetime.fromisoformat(s.replace("T", " "))
    except ValueError:
        pass
    for fmt in ("%d.%m.%Y %H:%M:%S", "%d.%m.%Y %H:%M", "%Y-%m-%d %H:%M", "%d-%b-%Y %H:%M:%S"):
        try:
            return _dt.datetime.strptime(s, fmt)
        except ValueError:
            continue
    raise ValueError(f"time '{s}' not understood (use e.g. 2026-09-30 01:58:02)")


def run_times(raw):
    """(start, middle, end) datetimes of a run: end from the file when it is there, else start + time."""
    t0 = parse_time(getattr(raw, "start_time", None))
    if t0 is None:
        raise ValueError(f"run {getattr(raw, 'run', '?')}: no start time")
    dur = float(getattr(raw, "counting_time_s", 0.0) or 0.0)
    t1 = None
    try:
        t1 = parse_time(getattr(raw, "end_time", None))
    except ValueError:
        t1 = None
    if t1 is None or t1 < t0:
        t1 = t0 + _dt.timedelta(seconds=dur)
    mid = t0 + (t1 - t0) / 2
    return t0, mid, t1


@dataclass
class SeriesPoint:
    index: int
    runs: List[int]
    time_s: float                 # weighted middle, relative to t0 [s]
    start: _dt.datetime
    end: _dt.datetime
    t0: _dt.datetime
    weights: List[float] = field(default_factory=list)

    @property
    def label(self) -> str:
        return f"t = {self.time_s:.0f} s"

    def parameters(self, series_label: str) -> Dict[str, object]:
        iso = lambda t: t.isoformat(sep=" ", timespec="seconds")   # noqa: E731
        return {"time_s": self.time_s, "time_start": iso(self.start),
                "time_end": iso(self.end), "series": series_label,
                "series_t0": iso(self.t0), "series_index": self.index,
                "series_runs": ", ".join(map(str, self.runs))}


def series_points(row, raws: Dict[int, object], normalise_by: str = "monitor", monitor: str = "monitor1"
                  ) -> List[SeriesPoint]:
    """The points of a series row: its used runs (those in ``raws``, not excluded / empty) sorted by start
    time, grouped by ``row.series_bin`` runs or ``row.series_dt_s`` seconds."""
    rs = []
    for n in row.used:
        r = raws.get(n)
        if r is None or getattr(r, "empty_reason", ""):
            continue
        t0, mid, t1 = run_times(r)
        if normalise_by == "monitor":
            w = float((getattr(r, "monitors", {}) or {}).get(monitor, 0.0) or 0.0)
        else:
            w = float(getattr(r, "counting_time_s", 0.0) or 0.0)
        rs.append((t0, mid, t1, max(w, 1e-12), n))
    if not rs:
        return []
    rs.sort()
    ref = parse_time(getattr(row, "series_t0", "") or "") or rs[0][0]
    groups: List[list] = []
    dt = getattr(row, "series_dt_s", None)
    if dt:
        for x in rs:
            k = int(((x[1] - rs[0][0]).total_seconds()) // float(dt))
            if not groups or groups[-1][0] != k:
                groups.append([k, []])
            groups[-1][1].append(x)
        groups = [g[1] for g in groups]
    else:
        nb = max(1, int(getattr(row, "series_bin", 1) or 1))
        groups = [rs[i:i + nb] for i in range(0, len(rs), nb)]
    out = []
    for i, g in enumerate(groups):
        W = sum(x[3] for x in g)
        t = sum(x[3] * (x[1] - ref).total_seconds() for x in g) / W
        out.append(SeriesPoint(i, [x[4] for x in g], t, g[0][0], g[-1][2], ref, [x[3] for x in g]))
    return out
