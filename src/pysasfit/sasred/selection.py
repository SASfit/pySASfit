# SPDX-License-Identifier: BSD-3-Clause
"""Selecting runs and grouping them into samples, as BerSANS' *Select many samples*:

1. **Select** the runs to work on: a run range, optionally a pattern (alternately use n runs and
   skip m), and conditions on any header item: text equal or matching a regular expression
   (``EXP-.*``), a number in a range (``1.99 ≤ SD ≤ 2.01``).
2. **Group** the selected runs by header items, each with a ± tolerance (sample name exactly,
   temperature ± 1 K, field ± 0.01 T, pressure, shear rate, ...): every group becomes one sample row
   of the reduction table, its runs summed.

Header items are the named entries of :data:`grouping.FIELDS` (wavelength in Å, distances in m, ...),
the run kind (scattering / transmission) and every scalar entry of the raw files, by its path
(``hdf:sample/temperature``, values as stored, e.g. the wavelength in nm).
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from . import grouping as gr

__all__ = ["Criterion", "Metadata", "select_runs", "group_selected", "HDF"]

HDF = "hdf:"


def _num(v) -> Optional[float]:
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None
    return x if np.isfinite(x) else None


@dataclass
class Criterion:
    """One condition on a header item.

    ``kind`` "match": the value (as text) equals ``text`` — with ``regex`` the whole value must
    match the regular expression (BerSANS "is exactly equal" with regular expressions on);
    numbers also compare numerically. ``kind`` "range": ``lo ≤ value ≤ hi`` (either may be None)."""
    field: str
    kind: str = "match"
    text: str = ""
    lo: Optional[float] = None
    hi: Optional[float] = None
    regex: bool = True

    def test(self, v) -> bool:
        if v is None:
            return False
        if self.kind == "range":
            x = _num(v)
            return x is not None and (self.lo is None or x >= self.lo) and (self.hi is None or x <= self.hi)
        s = str(v).strip()
        t = self.text.strip()
        if t == "":
            return True
        if self.regex:
            try:
                if re.fullmatch(t, s):
                    return True
            except re.error:
                pass
        if s == t:
            return True
        a, b = _num(s), _num(t)
        return a is not None and b is not None and a == b


class Metadata:
    """Header items of a set of runs (catalogue entries or raw runs), read once and cached."""

    def __init__(self, items, extra: Optional[dict] = None):
        """``extra``: further items as name -> getter of a raw run (e.g. a temperature from a log)."""
        self.items = [i for i in items if getattr(i, "raw", i) is not None]
        self.extra = dict(extra or {})
        self._hdf: Dict[int, dict] = {}
        self.kinds = {getattr(i, "run"): getattr(i, "kind", "") for i in self.items}
        if not any(self.kinds.values()):
            from .catalogue import classify
            k, _, _ = classify([getattr(i, "raw", i) for i in self.items])
            self.kinds = k

    @staticmethod
    def raw(item):
        return getattr(item, "raw", item)

    def _hdf_of(self, raw) -> dict:
        d = self._hdf.get(raw.run)
        if d is None:
            d = {}
            try:
                import h5py
                with h5py.File(raw.path, "r") as f:
                    def visit(name, o):
                        if isinstance(o, h5py.Dataset) and o.size == 1:
                            v = o[()]
                            v = np.asarray(v).ravel()[0]
                            if isinstance(v, bytes):
                                v = v.decode(errors="replace").strip()
                            elif isinstance(v, np.generic):
                                v = v.item()
                            d[name.split("/", 1)[1] if name.startswith(("entry1/", "entry0/")) else name] = v
                    f.visititems(visit)
            except Exception:  # noqa: BLE001 -- missing file: no header items
                pass
            self._hdf[raw.run] = d
        return d

    def fields(self) -> List[str]:
        """Named items, ``kind``, then the scalar entries of the first readable file."""
        out = list(gr.FIELDS) + list(self.extra) + ["kind"]
        for it in self.items[:1]:
            out += [HDF + k for k in sorted(self._hdf_of(self.raw(it)))]
        return out

    def value(self, item, field: str):
        raw = self.raw(item)
        if field == "kind":
            return self.kinds.get(raw.run, "")
        if field == "run":
            return raw.run
        if field.startswith(HDF):
            return self._hdf_of(raw).get(field[len(HDF):])
        if field in self.extra:
            try:
                return self.extra[field](raw)
            except Exception:  # noqa: BLE001
                return None
        try:
            return gr.FIELDS[field](raw)
        except Exception:  # noqa: BLE001
            return None

    def getters(self, fields: Iterable[str]) -> dict:
        """Getters of raw runs for :func:`grouping.group_runs`."""
        by_run = {self.raw(i).run: i for i in self.items}
        return {f: (lambda r, f=f: self.value(by_run.get(r.run, r), f)) for f in fields}

    def is_numeric(self, field: str) -> bool:
        vals = [self.value(i, field) for i in self.items[:20]]
        vals = [v for v in vals if v is not None and v != ""]
        return bool(vals) and all(_num(v) is not None and not isinstance(v, str) for v in vals)


def select_runs(meta: Metadata, run_range: Optional[Tuple[int, int]] = None,
                pattern: Optional[Tuple[int, int]] = None, criteria: Sequence[Criterion] = ()) -> list:
    """Items in the run range (inclusive), on the pattern (starting with the first run of the range,
    alternately use ``pattern[0]`` runs and skip ``pattern[1]``), meeting every criterion. Runs without usable
    data (measuring time 0, 0 counts, monitor 0) are never selected."""
    out = []
    lo, hi = run_range if run_range else (None, None)
    start = lo if lo is not None else min((Metadata.raw(i).run for i in meta.items), default=0)
    for it in meta.items:
        r = Metadata.raw(it).run
        if lo is not None and r < lo or hi is not None and r > hi:
            continue
        if pattern:
            use, skip = pattern
            if use > 0 and (r - start) % (use + skip) >= use:
                continue
        if getattr(Metadata.raw(it), "empty_reason", ""):
            continue
        if all(c.test(meta.value(it, c.field)) for c in criteria):
            out.append(it)
    return sorted(out, key=lambda i: Metadata.raw(i).run)


def group_selected(meta: Metadata, items, tolerances: Dict[str, Optional[float]],
                   min_runs: int = 1) -> List[gr.RunGroup]:
    """Group the selected items: every field with its ± tolerance (None or 0: equal values). All runs
    of a group lie within ± tolerance of the centre of the group (the span is at most twice the
    tolerance)."""
    match = {}
    for f, tol in tolerances.items():
        match[f] = None if tol in (None, 0, "") else 2.0 * float(tol)
    fields = meta.getters(match)
    return gr.group_runs(list(items), match, fields=fields, min_runs=min_runs)
