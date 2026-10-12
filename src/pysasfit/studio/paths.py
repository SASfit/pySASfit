"""Where pysasfit.studio finds the SASfit libraries, plugin headers and Python plugins.

Search order for the compiled SASfit libraries (``libsasfit`` plus the GSL,
FFTW and compiler runtime it needs, and ``plugins/libsasfit_*``):

1. the directory set with :func:`set_lib_dir` at run time;
2. the environment variable ``PYSASFIT_SASFIT_LIB``;
3. the environment variable ``SASFIT_LOCATION`` (as used by sasfit-collab);
4. ``sasfit_lib/`` inside this package (filled by ``pysasfit-studio install-binaries``).

The plugin libraries are looked for in ``<lib>/plugins`` and in ``<lib>``.
Nothing depends on the current working directory.
"""

from __future__ import annotations

import os
import platform

__all__ = [
    "PACKAGE_DIR",
    "BUNDLED_HEADERS",
    "TEMPLATES",
    "LIB_EXT",
    "lib_dir",
    "set_lib_dir",
    "plugin_lib_dirs",
    "header_dirs",
    "python_plugin_dirs",
]

PACKAGE_DIR = os.path.dirname(os.path.abspath(__file__))
BUNDLED_HEADERS = os.path.join(PACKAGE_DIR, "headers")
TEMPLATES = os.path.join(PACKAGE_DIR, "templates")
DEFAULT_LIB = os.path.join(PACKAGE_DIR, "sasfit_lib")

_system = platform.system()
LIB_EXT = "dll" if _system == "Windows" else ("dylib" if _system == "Darwin" else "so")

_override = None


def set_lib_dir(path):
    """Use ``path`` for the SASfit libraries (before the first function is loaded)."""
    global _override
    _override = os.path.abspath(path) if path else None


def lib_dir():
    """Directory holding ``libsasfit``, or None if no libraries are installed."""
    for d in (_override, os.environ.get("PYSASFIT_SASFIT_LIB"), os.environ.get("SASFIT_LOCATION"), DEFAULT_LIB):
        if d and os.path.isfile(os.path.join(d, f"libsasfit.{LIB_EXT}")):
            return d
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


def python_plugin_dirs():
    """Directories searched for Python plugins ``sasstudio_<name>.py``."""
    dirs = [d for d in os.environ.get("PYSASFIT_STUDIO_PLUGINS", "").split(os.pathsep) if d]
    dirs.append(os.path.join(os.path.expanduser("~"), ".pysasfit", "studio_plugins"))
    dirs += plugin_lib_dirs()
    return [d for d in dirs if os.path.isdir(d)]
