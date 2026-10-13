# -*- coding: utf-8 -*-
"""User preferences: numerical settings that are not on the tools' main panels.

Every entry is declared once in :mod:`pysasfit.core.settings_schema` with its
factory default and a label; the explanation of each entry is in
:mod:`pysasfit.core.settings_help`. The user's values live in one JSON
file::

    Windows:  %APPDATA%\\pySASfit\\settings.json
    Linux/macOS:  $XDG_CONFIG_HOME/pysasfit/settings.json  (default ~/.config/...)
    override: environment variable PYSASFIT_SETTINGS=<path>

Only values that differ from the factory default are written, so a later
change of a factory default reaches every user who has not set that entry.

The GUIs read the values when a calculation starts (``get``/``values``) and
pass them explicitly to the library; library functions keep their own
documented defaults, so scripts are not affected by the preferences file.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field

__all__ = ["Setting", "Section", "register", "tools", "sections", "find", "get", "values", "set_value",
           "reset", "save", "load", "settings_path", "is_default"]


@dataclass
class Setting:
    key: str                    # unique, "tool.name"
    default: object
    label: str
    help: str = ""
    kind: str = "float"         # float | int | bool | choice | str
    lo: float | None = None
    hi: float | None = None
    choices: list = field(default_factory=list)   # [(value, label), ...] for kind="choice"
    log: bool = False           # float shown in scientific notation

    def coerce(self, v):
        """Validate and convert ``v``; raises ValueError."""
        if self.kind == "bool":
            if isinstance(v, str):
                return v.strip().lower() in ("1", "true", "yes", "on")
            return bool(v)
        if self.kind == "choice":
            vals = [c[0] for c in self.choices]
            if v not in vals:
                raise ValueError(f"{self.label}: {v!r} is not one of {vals}")
            return v
        if self.kind == "str":
            return str(v)
        x = int(round(float(v))) if self.kind == "int" else float(v)
        if self.lo is not None and x < self.lo:
            raise ValueError(f"{self.label}: {x} < {self.lo}")
        if self.hi is not None and x > self.hi:
            raise ValueError(f"{self.label}: {x} > {self.hi}")
        return x


@dataclass
class Section:
    tool: str                   # e.g. "sldprofile"
    tool_title: str             # e.g. "SLD profile"
    title: str
    settings: list


_SECTIONS: list[Section] = []
_BY_KEY: dict[str, Setting] = {}
_VALUES: dict[str, object] = {}
_LOADED = False


def register(tool, tool_title, title, settings):
    from .settings_help import HELP     # the explanation of every setting
    for s in settings:
        s.help = HELP.get(s.key) or s.help
        if s.key in _BY_KEY:
            raise ValueError(f"duplicate setting {s.key}")
        if not s.key.startswith(tool + "."):
            raise ValueError(f"setting {s.key} must start with '{tool}.'")
        s.default = s.coerce(s.default)
        _BY_KEY[s.key] = s
    _SECTIONS.append(Section(tool, tool_title, title, list(settings)))


def _ensure():
    global _LOADED
    if not _BY_KEY:
        from . import settings_schema  # noqa: F401  (registers everything)
    if not _LOADED:
        _LOADED = True
        load()


def tools():
    """[(tool, title)] in registration order."""
    _ensure()
    seen, out = set(), []
    for s in _SECTIONS:
        if s.tool not in seen:
            seen.add(s.tool)
            out.append((s.tool, s.tool_title))
    return out


def sections(tool=None):
    _ensure()
    return [s for s in _SECTIONS if tool is None or s.tool == tool]


def find(key) -> Setting:
    _ensure()
    return _BY_KEY[key]


def get(key):
    """Current value (user value, else factory default)."""
    _ensure()
    s = _BY_KEY[key]
    return _VALUES.get(key, s.default)


def values(prefix):
    """{name: value} for every key 'prefix.name...' (name = rest of the key)."""
    _ensure()
    p = prefix.rstrip(".") + "."
    return {k[len(p):]: get(k) for k in _BY_KEY if k.startswith(p)}


def is_default(key):
    _ensure()
    return key not in _VALUES


def set_value(key, v):
    """Set (validated) for this session; call :func:`save` to keep it."""
    _ensure()
    s = _BY_KEY[key]
    v = s.coerce(v)
    if v == s.default:
        _VALUES.pop(key, None)
    else:
        _VALUES[key] = v


def reset(tool=None):
    """Back to factory defaults (one tool or all); call :func:`save` to keep it."""
    _ensure()
    for k in list(_VALUES):
        if tool is None or k.startswith(tool + "."):
            del _VALUES[k]


def settings_path():
    p = os.environ.get("PYSASFIT_SETTINGS")
    if p:
        return p
    if os.name == "nt":
        base = os.environ.get("APPDATA") or os.path.expanduser("~")
        return os.path.join(base, "pySASfit", "settings.json")
    base = os.environ.get("XDG_CONFIG_HOME") or os.path.join(os.path.expanduser("~"), ".config")
    return os.path.join(base, "pysasfit", "settings.json")


def load(path=None):
    """Read the preferences file. Unknown keys are kept (a newer version may know
    them); invalid values are dropped with a warning."""
    global _LOADED
    _LOADED = True
    if not _BY_KEY:
        from . import settings_schema  # noqa: F401
    _VALUES.clear()
    path = path or settings_path()
    try:
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
    except FileNotFoundError:
        return
    except (OSError, ValueError) as exc:
        import warnings
        warnings.warn(f"pySASfit settings file {path} unreadable ({exc}); using factory defaults")
        return
    for k, v in (data.get("values") or {}).items():
        s = _BY_KEY.get(k)
        if s is None:
            _VALUES[k] = v
            continue
        try:
            set_value(k, v)
        except (ValueError, TypeError):
            import warnings
            warnings.warn(f"pySASfit setting {k}={v!r} invalid; factory default used")


def save(path=None):
    path = path or settings_path()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump({"format": "pysasfit_settings_v1", "values": dict(sorted(_VALUES.items()))}, fh,
                  indent=2)
        fh.write("\n")
    os.replace(tmp, path)
    return path
