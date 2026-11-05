# SPDX-License-Identifier: BSD-3-Clause
"""Sample names and roles corrected by hand, and runs excluded by hand, per raw-data folder.

Sample names in the raw files are sometimes wrong (two cells swapped, a remeasurement with a long
explanatory name, ...). The roles of the reduction (water, empty cell, ...) and the grouping follow the
sample names, so a wrong name puts runs into the wrong row. Here a run gets a new sample name and/or a
fixed role; the raw files stay untouched.

The corrections of a folder are kept in ``sasred_names/<folder hash>.json`` next to the preferences
file (raw-data folders are often read-only) and are applied whenever a run of that folder is read
(:func:`apply`): ``raw.sample_name`` becomes the new name, the name in the file stays in
``raw.sample_name_in_file``, and ``raw.role`` (if set) overrides the role from the name
(:func:`pysasfit.sasred.plan.role_of`).

A run can also be **excluded** (sample preparation failed, phase separation during the measurement, …):
``raw.excluded`` holds the reason, the run is skipped like an empty run (never selected, not reduced)
and stays excluded when the folder is read again.
"""
from __future__ import annotations

import hashlib
import json
import os
from typing import Dict, Iterable, Optional

__all__ = ["Corrections", "load", "save", "apply", "store_path", "ROLE_CHOICES"]

#: role -> label (empty role: from the sample name)
ROLE_CHOICES = {"": "from the sample name", "sample": "sample", "water": "water (H2O)",
                "empty_cell": "empty cell (water background)", "sample_background": "sample background",
                "empty_beam": "empty beam", "blocked_beam": "Cd / blocked beam"}


class Corrections(dict):
    """{run: {"name": new sample name or None, "role": role or None, "note": text,
    "exclude": reason or None}} of one folder."""

    def __init__(self, folder: str = "", data=None):
        super().__init__()
        self.folder = os.path.abspath(folder) if folder else ""
        for k, v in (data or {}).items():
            self[int(k)] = dict(v)

    def set(self, runs: Iterable[int], name: Optional[str] = None, role: Optional[str] = None, note: str = ""):
        """New name (None or "" keeps the name of the file) and role (None or "" from the name)."""
        for r in runs:
            old = self.get(int(r), {})
            d = {"name": name or None, "role": role or None, "note": note, "exclude": old.get("exclude")}
            self._store(int(r), d)

    def _store(self, r: int, d: dict):
        if d.get("name") is None and d.get("role") is None and not d.get("exclude"):
            self.pop(r, None)
        else:
            self[r] = d

    def exclude(self, runs: Iterable[int], reason: str = ""):
        """Exclude runs from every reduction (``reason``: why, shown in the run list and the log)."""
        for r in runs:
            d = dict(self.get(int(r), {"name": None, "role": None, "note": ""}))
            d["exclude"] = (reason or "").strip() or "no reason given"
            self._store(int(r), d)

    def include(self, runs: Iterable[int]):
        """Take back the exclusion (name and role corrections stay)."""
        for r in runs:
            if int(r) in self:
                d = dict(self[int(r)])
                d["exclude"] = None
                self._store(int(r), d)

    def excluded(self) -> Dict[int, str]:
        return {r: d["exclude"] for r, d in self.items() if d.get("exclude")}

    def reset(self, runs: Iterable[int]):
        for r in runs:
            self.pop(int(r), None)

    def describe(self) -> str:
        lines = []
        for r in sorted(self):
            d = self[r]
            parts = []
            if d.get("name"):
                parts.append(f"name '{d['name']}'")
            if d.get("role"):
                parts.append(f"role {d['role']}")
            if d.get("exclude"):
                parts.append(f"EXCLUDED: {d['exclude']}")
            lines.append(f"run {r}: " + ", ".join(parts) + (f" ({d['note']})" if d.get("note") else ""))
        return "\n".join(lines)


def store_path(folder: str) -> str:
    try:
        from ..core.settings import settings_path
        base = os.path.dirname(settings_path())
    except Exception:  # noqa: BLE001
        base = os.path.expanduser("~")
    key = hashlib.sha1(os.path.normcase(os.path.abspath(folder)).encode()).hexdigest()[:16]
    return os.path.join(base, "sasred_names", f"{key}.json")


def load(folder: str) -> Corrections:
    p = store_path(folder)
    if not os.path.isfile(p):
        return Corrections(folder)
    try:
        with open(p, encoding="utf-8") as fh:
            d = json.load(fh)
        return Corrections(folder, d.get("runs", {}))
    except (OSError, ValueError):
        return Corrections(folder)


def save(c: Corrections) -> str:
    p = store_path(c.folder)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, "w", encoding="utf-8") as fh:
        json.dump({"folder": c.folder, "runs": {str(k): v for k, v in sorted(c.items())}}, fh, indent=1,
                  ensure_ascii=False)
    return p


def apply(raw, corrections: Optional[Dict[int, dict]]):
    """Apply the corrections to a raw run (in place); returns the run."""
    if raw is None:
        return raw
    if not hasattr(raw, "sample_name_in_file") or raw.sample_name_in_file is None:
        raw.sample_name_in_file = raw.sample_name
    d = (corrections or {}).get(int(raw.run))
    raw.sample_name = (d or {}).get("name") or raw.sample_name_in_file
    raw.role = (d or {}).get("role") or ""
    raw.excluded = (d or {}).get("exclude") or ""
    return raw
