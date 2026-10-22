# -*- coding: utf-8 -*-
"""Project files and the automatic last session of the GUI tools.

A project holds everything needed to reopen a piece of work: the data (path
*and* an embedded copy of the numbers), the settings of every panel control
and the last result. The same file format serves two purposes:

* **projects** -- File → Save project… / Open project… (``*.pysasfit``),
  written only when the user asks;
* **the last session** -- ``session_<tool>.pysasfit`` next to the
  preferences file (:func:`pysasfit.core.settings.settings_path`), written
  automatically after loading data, after a fit and when the window closes,
  and reopened at the next start. Settings → Forget last session deletes it.

File format (a ZIP archive)
---------------------------

``project.json``
    ``format`` = ``"pysasfit-project"``, ``format_version`` (integer, see
    :data:`FORMAT_VERSION`), ``tool``, ``app_version``, ``saved`` (ISO time),
    ``data`` (original path, size/mtime signature, column names), ``controls``
    (``{name: {"kind": ..., "value": ...}}``, one entry per panel control) and
    ``extra`` (tool-specific plain values).
``data.npz``
    the data columns as read (q, I, sigma[, dq]) -- the project still opens
    when the original data file has moved or changed.
``result.pkl``
    the last result as a Python pickle (optional). It is only needed to show
    the result without fitting again; if it cannot be read (e.g. after an
    incompatible change of the result classes) the project opens without it.

Compatibility rules
-------------------

* Reading never relies on a key being present. Before the controls are
  restored, the window is reset to its factory state, so every control that
  a file does not mention -- e.g. a parameter added in a later version --
  keeps its **default value**.
* Controls in the file that the program does not know (from a newer version)
  are ignored, and so are values a control does not accept.
* ``format_version`` is raised whenever the meaning of an existing entry
  changes; :func:`upgrade` converts older files step by step (version 1 is
  the first version, nothing to convert yet). A file with a newer
  ``format_version`` than this program is still read, with a warning.
"""

from __future__ import annotations

import io
import json
import os
import pickle
import time
import zipfile

import numpy as np

from .settings import settings_path

__all__ = ["FORMAT", "FORMAT_VERSION", "EXTENSION", "ProjectError", "session_path", "file_signature",
           "save_project", "load_project", "forget_session", "control_state", "restore_controls",
           "write_data_file", "upgrade"]

FORMAT = "pysasfit-project"
#: Version of the project format. Raise it when the meaning of an existing entry
#: changes and add the conversion to :func:`upgrade`. New entries need no new
#: version: missing entries take their default values when read.
FORMAT_VERSION = 1
EXTENSION = ".pysasfit"
TOOLS = ("sldprofile", "inversion")


class ProjectError(Exception):
    """A file that is not a readable pySASfit project for this tool."""


def session_path(tool: str) -> str:
    return os.path.join(os.path.dirname(settings_path()), f"session_{tool}{EXTENSION}")


def file_signature(path):
    """[size, mtime] of a data file, to detect a file changed since the project was saved."""
    try:
        st = os.stat(path)
        return [int(st.st_size), int(st.st_mtime)]
    except (OSError, TypeError):
        return None


def _app_version():
    try:
        from .. import __version__
        return __version__
    except Exception:  # noqa: BLE001
        return "?"


# --------------------------------------------------------------------------- write / read
def save_project(path: str, tool: str, *, data_path=None, columns=None, controls=None, extra=None,
                 result=None) -> None:
    """Write a project file (atomically). ``columns``: dict name -> 1-D array
    (q, I, sigma, dq as read; None entries are skipped)."""
    manifest = {
        "format": FORMAT,
        "format_version": FORMAT_VERSION,
        "tool": tool,
        "app_version": _app_version(),
        "saved": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "data": {"path": os.path.abspath(data_path) if data_path else None,
                 "signature": file_signature(data_path) if data_path else None,
                 "columns": [k for k, v in (columns or {}).items() if v is not None]},
        "controls": {k: {"kind": kind, "value": _jsonable(v)} for k, (kind, v) in (controls or {}).items()},
        "extra": _jsonable(extra or {}),
    }
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = path + ".tmp"
    with zipfile.ZipFile(tmp, "w", compression=zipfile.ZIP_DEFLATED) as z:
        z.writestr("project.json", json.dumps(manifest, indent=1, ensure_ascii=False))
        if columns:
            buf = io.BytesIO()
            np.savez(buf, **{k: np.asarray(v, float) for k, v in columns.items() if v is not None})
            z.writestr("data.npz", buf.getvalue())
        if result is not None:
            try:
                z.writestr("result.pkl", pickle.dumps(result, protocol=pickle.HIGHEST_PROTOCOL))
            except Exception:  # noqa: BLE001 -- settings and data are still worth saving
                pass
    os.replace(tmp, path)


def load_project(path: str, tool: str) -> dict:
    """Read a project file. Returns a dict with the manifest entries plus
    ``columns`` (dict of arrays or None), ``result`` (object or None) and
    ``notes`` (list of messages about anything that could not be restored).
    Raises :class:`ProjectError` for files that are not projects of ``tool``."""
    notes = []
    try:
        z = zipfile.ZipFile(path)
    except (OSError, zipfile.BadZipFile) as e:
        raise ProjectError(f"{path}: not a pySASfit project ({e})") from e
    with z:
        try:
            m = json.loads(z.read("project.json").decode("utf-8"))
        except Exception as e:  # noqa: BLE001
            raise ProjectError(f"{path}: project.json missing or unreadable ({e})") from e
        if m.get("format") != FORMAT:
            raise ProjectError(f"{path}: not a pySASfit project")
        if m.get("tool") != tool:
            raise ProjectError(f"{path}: project of the tool '{m.get('tool')}', not '{tool}'")
        v = int(m.get("format_version", 0) or 0)
        if v > FORMAT_VERSION:
            notes.append(f"project format {v} is newer than this program ({FORMAT_VERSION}); "
                         f"unknown entries are ignored")
        m = upgrade(m)
        cols = None
        if "data.npz" in z.namelist():
            with np.load(io.BytesIO(z.read("data.npz"))) as d:
                cols = {k: d[k] for k in d.files}
        res = None
        if "result.pkl" in z.namelist():
            try:
                res = pickle.loads(z.read("result.pkl"))
            except Exception as e:  # noqa: BLE001
                notes.append(f"the stored result could not be restored ({type(e).__name__}); run the fit again")
    m.setdefault("data", {})
    m.setdefault("controls", {})
    m.setdefault("extra", {})
    m["columns"], m["result"], m["notes"] = cols, res, notes
    return m


def upgrade(m: dict) -> dict:
    """Convert a manifest of an older ``format_version`` to the current one.
    Version 1 is the first format; future conversions go here, e.g.::

        if m["format_version"] < 2:
            ...   # rename / reinterpret entries
            m["format_version"] = 2
    """
    m.setdefault("format_version", 1)
    return m


def forget_session(tool: str | None = None) -> None:
    for t in ([tool] if tool else TOOLS):
        try:
            os.remove(session_path(t))
        except OSError:
            pass


def write_data_file(path: str, columns: dict) -> str:
    """Write embedded data columns as an ASCII file (q, I[, sigma[, dq]])."""
    names = [k for k in ("q", "I", "sigma", "dq") if columns.get(k) is not None]
    if "dq" in names and "sigma" not in names:
        names.remove("dq")
    np.savetxt(path, np.column_stack([columns[k] for k in names]),
               header="restored from a pySASfit project: " + " ".join(names))
    return path


# --------------------------------------------------------------------------- panel controls
def _jsonable(v):
    if isinstance(v, dict):
        return {str(k): _jsonable(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_jsonable(x) for x in v]
    if isinstance(v, np.generic):
        return v.item()
    if isinstance(v, np.ndarray):
        return v.tolist()
    return v


def _widgets(obj, extra=None):
    from PySide6 import QtWidgets
    kinds = (QtWidgets.QComboBox, QtWidgets.QCheckBox, QtWidgets.QAbstractSpinBox, QtWidgets.QLineEdit)
    out = {k: w for k, w in vars(obj).items() if isinstance(w, kinds)}
    out.update(extra or {})
    return out


def control_state(obj, extra=None, exclude=()) -> dict:
    """{name: (kind, value)} of the panel controls of ``obj`` (attributes that are
    combo boxes, check boxes, spin boxes or editable line edits, plus ``extra``)."""
    from PySide6 import QtWidgets
    st = {}
    for name, w in _widgets(obj, extra).items():
        if name in exclude:
            continue
        if isinstance(w, QtWidgets.QComboBox):
            data = w.currentData()
            st[name] = ("combo", {"text": w.currentText(), "index": w.currentIndex(),
                                  "data": data if isinstance(data, (str, int, float, bool)) else None})
        elif isinstance(w, QtWidgets.QCheckBox):
            st[name] = ("check", w.isChecked())
        elif isinstance(w, (QtWidgets.QSpinBox, QtWidgets.QDoubleSpinBox)):
            st[name] = ("spin", w.value())
        elif isinstance(w, QtWidgets.QLineEdit) and not w.isReadOnly():
            st[name] = ("line", w.text())
    return st


def restore_controls(obj, controls: dict, extra=None, passes: int = 2) -> list:
    """Set the controls from ``controls`` (``{name: {"kind", "value"}}`` or
    ``{name: (kind, value)}``): combo boxes first (their handlers may change
    other fields), then check boxes, then numbers and texts, repeated ``passes``
    times so that values overwritten by handlers are set again. Unknown names
    and impossible values are skipped; returns the names that were skipped."""
    from PySide6 import QtWidgets
    ws = _widgets(obj, extra)
    order = {"combo": 0, "check": 1, "spin": 2, "line": 3}
    items, skipped = [], []
    for n, e in controls.items():
        kind, val = (e["kind"], e["value"]) if isinstance(e, dict) else e
        if n in ws:
            items.append((n, kind, val))
        else:
            skipped.append(n)
    items.sort(key=lambda t: order.get(t[1], 9))
    for _ in range(passes):
        for name, kind, val in items:
            w = ws[name]
            try:
                if kind == "combo" and isinstance(w, QtWidgets.QComboBox):
                    i = -1
                    if isinstance(val, dict):
                        if val.get("data") is not None:
                            i = w.findData(val["data"])
                        if i < 0:
                            i = w.findText(val.get("text", ""))
                        if i < 0 and 0 <= int(val.get("index", -1)) < w.count():
                            i = int(val["index"])
                    if i >= 0 and i != w.currentIndex():
                        w.setCurrentIndex(i)
                elif kind == "check" and isinstance(w, QtWidgets.QCheckBox):
                    if w.isChecked() != bool(val):
                        w.setChecked(bool(val))
                elif kind == "spin" and isinstance(w, (QtWidgets.QSpinBox, QtWidgets.QDoubleSpinBox)):
                    if w.value() != val:
                        w.setValue(val)
                elif kind == "line" and isinstance(w, QtWidgets.QLineEdit):
                    if w.text() != val:
                        w.setText(str(val))
            except Exception:  # noqa: BLE001
                skipped.append(name)
    return skipped
