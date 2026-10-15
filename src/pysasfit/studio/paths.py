"""Where pysasfit.studio finds the SASfit libraries, plugin headers and Python plugins.

Search order for the compiled SASfit libraries (``libsasfit`` plus the GSL,
FFTW and compiler runtime it needs, and ``plugins/libsasfit_*``):

1. the folder set with :func:`set_lib_dir` at run time, or saved by the GUI
   (explorer: Settings -> SASfit installation; stored in
   ``~/.pysasfit/studio_settings.json``);
2. the environment variable ``PYSASFIT_SASFIT_LIB``;
3. the environment variable ``SASFIT_LOCATION`` (as used by sasfit-collab);
4. ``sasfit_lib/`` inside this package (filled by ``pysasfit-studio install-binaries``).

Every entry may name the folder holding ``libsasfit`` itself or a SASfit
installation around it: ``<dir>``, ``<dir>/lib``, ``<dir>/bin`` and
``<dir>/sasfit_lib`` are tried (and the parent of a ``plugins`` folder).
The plugin libraries are looked for in ``<lib>/plugins`` and in ``<lib>``.
A folder that does not contain libsasfit is skipped, so a wrong entry falls
through to the next one. Nothing depends on the current working directory.

The menu structure of the functions comes from Doxygen group files ``*.def``, found by
:func:`def_dirs` (``<lib>/plugins/def`` and the like); see :mod:`pysasfit.studio.groups`.
"""

from __future__ import annotations

import json
import os
import platform

__all__ = [
    "PACKAGE_DIR",
    "BUNDLED_HEADERS",
    "TEMPLATES",
    "LIB_EXT",
    "SETTINGS_FILE",
    "lib_dir",
    "lib_dir_source",
    "lib_dir_sources",
    "describe_lib_dir_sources",
    "find_lib_dir",
    "count_plugin_libs",
    "set_lib_dir",
    "saved_lib_dir",
    "chosen_lib_dir",
    "get_setting",
    "set_setting",
    "plugin_lib_dirs",
    "header_dirs",
    "def_dirs",
    "python_plugin_dirs",
]

PACKAGE_DIR = os.path.dirname(os.path.abspath(__file__))
BUNDLED_HEADERS = os.path.join(PACKAGE_DIR, "headers")
TEMPLATES = os.path.join(PACKAGE_DIR, "templates")
DEFAULT_LIB = os.path.join(PACKAGE_DIR, "sasfit_lib")
SETTINGS_FILE = os.path.join(os.path.expanduser("~"), ".pysasfit", "studio_settings.json")

_system = platform.system()
LIB_EXT = "dll" if _system == "Windows" else ("dylib" if _system == "Darwin" else "so")

_override = None
_UNSET = object()


# ------------------------------------------------------------------ settings file
def _read_settings():
    try:
        with open(SETTINGS_FILE, "r", encoding="utf-8") as fh:
            data = json.load(fh)
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def _write_settings(data):
    os.makedirs(os.path.dirname(SETTINGS_FILE), exist_ok=True)
    with open(SETTINGS_FILE, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2)


def get_setting(key, default=None):
    """A value remembered in :data:`SETTINGS_FILE` (JSON), e.g. 'integration' or 'precision'."""
    return _read_settings().get(key, default)


def set_setting(key, value):
    """Remember ``value`` (JSON-serialisable; None forgets the key). Raises OSError if the file cannot be written."""
    settings = _read_settings()
    if value is None:
        settings.pop(key, None)
    else:
        settings[key] = value
    _write_settings(settings)


def _clean(path):
    path = (path or "").strip().strip('"').strip("'").strip()
    return os.path.abspath(os.path.expanduser(path)) if path else None


def saved_lib_dir():
    """The folder remembered from an earlier session (``set_lib_dir(..., save=True)``), or None."""
    value = _read_settings().get("lib_dir")
    return value if isinstance(value, str) and value else None


def chosen_lib_dir():
    """The folder chosen by the user (this session's, else the saved one), valid or not."""
    return _override or saved_lib_dir()


def set_lib_dir(path, save=False):
    """Use ``path`` for the SASfit libraries; it takes precedence over the environment variables.

    ``path`` may be the folder holding libsasfit or a SASfit installation around it. Functions
    loaded earlier keep the libraries they were loaded from. With ``save=True`` the choice is
    stored in :data:`SETTINGS_FILE` and used in later sessions; ``set_lib_dir(None, save=True)``
    forgets the stored folder again (``set_lib_dir(None)`` only drops this session's choice).
    Raises OSError if the settings file cannot be written (the session choice is set anyway).
    """
    global _override
    _override = _clean(path)
    if save:
        settings = _read_settings()
        if _override:
            settings["lib_dir"] = _override
        else:
            settings.pop("lib_dir", None)
        _write_settings(settings)


# ------------------------------------------------------------------ library folder
def find_lib_dir(path):
    """The folder at/below ``path`` that holds ``libsasfit``, or None."""
    path = _clean(path)
    if not path:
        return None
    candidates = [path] + [os.path.join(path, sub) for sub in ("lib", "bin", "sasfit_lib")]
    if os.path.basename(path).lower() == "plugins":
        candidates.insert(1, os.path.dirname(path))
    for d in candidates:
        if os.path.isfile(os.path.join(d, f"libsasfit.{LIB_EXT}")):
            return d
    return None


def count_plugin_libs(lib):
    """Number of ``libsasfit_*`` plugin libraries below the library folder ``lib``."""
    n = 0
    for d in (os.path.join(lib, "plugins"), lib):
        if os.path.isdir(d):
            n += sum(1 for f in os.listdir(d) if f.startswith("libsasfit_") and f.endswith("." + LIB_EXT))
    return n


def lib_dir_sources(chosen=_UNSET):
    """The places searched for the libraries, in priority order.

    Returns ``[(key, label, raw value or None, resolved library folder or None), ...]``;
    ``chosen`` replaces the first entry (used by the GUI to preview an edited path).
    """
    first = chosen_lib_dir() if chosen is _UNSET else (chosen or None)
    rows = [
        ("gui", "folder chosen in the GUI", first),
        ("PYSASFIT_SASFIT_LIB", "PYSASFIT_SASFIT_LIB", os.environ.get("PYSASFIT_SASFIT_LIB") or None),
        ("SASFIT_LOCATION", "SASFIT_LOCATION", os.environ.get("SASFIT_LOCATION") or None),
        ("bundled", "bundled sasfit_lib", DEFAULT_LIB if os.path.isdir(DEFAULT_LIB) else None),
    ]
    return [(key, label, raw, find_lib_dir(raw)) for key, label, raw in rows]


def describe_lib_dir_sources(rows=None):
    """Plain-text table of :func:`lib_dir_sources` (``->`` marks the one in use)."""
    rows = lib_dir_sources() if rows is None else rows
    active = next((i for i, row in enumerate(rows) if row[3]), None)
    lines = []
    for i, (_, label, raw, resolved) in enumerate(rows):
        mark = "->" if i == active else "  "
        state = "ok " if resolved else ("bad" if raw else "-- ")
        lines.append(f"{mark} {i + 1}. {label:<26} {state} {raw or '(not set)'}")
    return "\n".join(lines)


def lib_dir_source():
    """Key of the entry in use ('gui', 'PYSASFIT_SASFIT_LIB', 'SASFIT_LOCATION', 'bundled') or None."""
    for key, _, _, resolved in lib_dir_sources():
        if resolved:
            return key
    return None


def lib_dir():
    """Directory holding ``libsasfit``, or None if no libraries are installed."""
    for _, _, _, resolved in lib_dir_sources():
        if resolved:
            return resolved
    return None


def plugin_lib_dirs():
    d = lib_dir()
    if d is None:
        return []
    return [p for p in (os.path.join(d, "plugins"), d) if os.path.isdir(p)]


def header_dirs():
    """Header directories; later entries override earlier ones."""
    dirs = [BUNDLED_HEADERS]
    for d in plugin_lib_dirs():
        if any(f.startswith("sasfit_") and f.endswith(".h") for f in os.listdir(d)):
            dirs.append(d)
    return dirs


def def_dirs():
    """Folders searched for the group definitions ``*.def`` (menu tree of the functions).

    The ``def`` subfolder of each plugin library folder (``<lib>/plugins/def``, ``<lib>/def``) and the
    folder itself, then the same for the Python plugin folders, so that your own groups (see
    ``templates/groups_example.def``) come last. Existing folders only.
    """
    base = [BUNDLED_HEADERS] + plugin_lib_dirs()
    base += [d for d in os.environ.get("PYSASFIT_STUDIO_PLUGINS", "").split(os.pathsep) if d]
    base.append(os.path.join(os.path.expanduser("~"), ".pysasfit", "studio_plugins"))
    out = []
    for d in base:
        for cand in (os.path.join(d, "def"), d):
            if os.path.isdir(cand) and cand not in out:
                out.append(cand)
    return out


def python_plugin_dirs():
    """Directories searched for Python plugins ``sasstudio_<name>.py``."""
    dirs = [d for d in os.environ.get("PYSASFIT_STUDIO_PLUGINS", "").split(os.pathsep) if d]
    dirs.append(os.path.join(os.path.expanduser("~"), ".pysasfit", "studio_plugins"))
    dirs += plugin_lib_dirs()
    return [d for d in dirs if os.path.isdir(d)]
