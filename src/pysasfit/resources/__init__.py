"""Logos and icons of the pySASfit tools (from the logo bundle, ``docs/pySASfit-logo-bundle``).

    from pysasfit.resources import tool_icon, logo_path
    window.setWindowIcon(tool_icon("pySASstudio"))  # the coloured menu icon of the tool
    logo_path("pySASfit", "light")                  # horizontal logo (svg) for documentation, or None

Two kinds of square icons are in the bundle: the **menu icons** (``menu-icons/``, one colour per tool with a
simple glyph, readable at 16 px) are used for windows, the task bar, the launcher tabs and menus
(:func:`tool_icon`, :func:`menu_icon_path`); the navy **tiles** (``icons/``) remain available as
:func:`tile_icon` / :func:`icon_path`.

The files are looked for in ``pysasfit/resources/`` (copied there by ``tools/sync_logos.py``, so that they are part of
the installed package) and, in a source checkout, in the logo bundle itself. Without any file the icon is empty and
nothing breaks.

Tools: ``pySASfit`` (the suite, launcher), ``pySLDprofile``, ``pyINVersion``, ``pyOZsolver``, ``pySASstudio``
(model explorer), ``pySASred``, ``pyTOOLbox``, ``pySCAtter``, ``pySASmorph``, ``pySEBpolymers``, ``pySLDcalculator``,
``pyDENsity``, ``pyDENSSmulti``, ``pySASscout``, ``pyATsas`` (the last four made with ``tools/make_tool_logos.py``). The former
names pyInversion and pySASbox are accepted
(:data:`ALIASES`).

The icons of the ATSAS *programs* (PRIMUS, CRYSOL, ... on the pyATsas panel) are the original ones of the user's
ATSAS installation (:func:`external_icon`, :func:`pysasfit.atsas.core.icon_files`); pySASfit ships no copy.
"""

from __future__ import annotations

import os

__all__ = ["TOOLS", "EXTERNAL_TOOLS", "WORDMARKS", "COLOURS", "TOOL_COLOURS", "ALIASES", "tool_icon", "tile_icon", "logo_path",
           "icon_path", "menu_icon_path", "wordmark"]

TOOLS = ("pySASfit", "pySLDprofile", "pyINVersion", "pyOZsolver", "pySASstudio", "pySASred", "pyTOOLbox", "pySCAtter",
         "pySASmorph", "pySEBpolymers", "pySLDcalculator", "pyDENsity", "pyDENSSmulti", "pySASscout", "pyATsas")

#: tools whose icon comes from an external program's installation, not from the logo bundle
EXTERNAL_TOOLS = ()          # pyATsas has its own logo set since 2026-11; external_icon() gives the ATSAS icon

#: wordmark of every tool as in the logos: grey "py", bold core in the tool's colour, word ending in the grey of "py"
WORDMARKS = {
    "pySASfit": ("py", "SAS", "fit"),
    "pySASred": ("py", "SAS", "red"),
    "pyOZsolver": ("py", "OZ", "solver"),
    "pyINVersion": ("py", "INV", "ersion"),
    "pySLDprofile": ("py", "SLD", "profile"),
    "pySASstudio": ("py", "SAS", "studio"),
    "pyTOOLbox": ("py", "TOOL", "box"),
    "pySASmorph": ("py", "SAS", "morph"),
    "pySCAtter": ("py", "SCA", "tter"),
    "pySEBpolymers": ("py", "SEB", "polymers"),
    "pySLDcalculator": ("py", "SLD", "calculator"),
    "pyATsas": ("py", "AT", "sas"),
    "pyDENsity": ("py", "DEN", "sity"),
    "pyDENSSmulti": ("py", "DENSS", "multi"),
    "pySASscout": ("py", "SAS", "scout"),
}

#: wordmark colours of the logo bundle (README.txt): part -> (on light, on dark background); the core takes the
#: tool's colour (:data:`TOOL_COLOURS`), these are the common parts (and the core of the suite, pySASfit)
COLOURS = {"py": ("#5E7388", "#9FB6CC"), "core": ("#0E2740", "#FFFFFF"), "suffix": ("#5E7388", "#9FB6CC")}

#: colour of every tool (tile of its logo and menu icon; core of its wordmark): (on light, on dark background,
#: lightened by about a third for contrast)
TOOL_COLOURS = {
    "pySASfit": ("#0E2740", "#FFFFFF"),        # navy: the sky-blue tile (#CFE8FB) is too light for text
    "pySASred": ("#C8283C", "#DB7380"),
    "pyOZsolver": ("#6B46C1", "#9F87D7"),
    "pyINVersion": ("#2E8B57", "#77B492"),
    "pySLDprofile": ("#C23A93", "#D77FB9"),
    "pySASstudio": ("#2F6FED", "#78A1F3"),
    "pyTOOLbox": ("#D9730D", "#E6A462"),
    "pySASmorph": ("#0E2740", "#FFFFFF"),      # marine: tile #0E2740, navy/white wordmark core as in its logo
    "pySCAtter": ("#A0522D", "#C78A6B"),       # sienna: menu icon and tile in the logo bundle
    "pySEBpolymers": ("#6B8E23", "#9CB46C"),   # olive: core with grafted chains on the menu icon
    "pySLDcalculator": ("#8E24AA", "#BB7CCB"),  # purple: calculator with the periodic table as keypad
    "pyATsas": ("#0B6666", "#5FA8A8"),         # teal of the ATSAS icon tile
    "pyDENsity": ("#2B4C9B", "#8FA6D9"),       # deep blue (own icon: contour lines of a density map)
    "pyDENSSmulti": ("#1F7A8C", "#6FB3C0"),    # teal blue (own icon: two coloured lobes, one per component)
    "pySASscout": ("#9A6A00", "#D2A84A"),      # dark amber (own icon: a magnifier on a scattering curve)
}

#: old tool names -> current ones
ALIASES = {"pySASbox": "pyTOOLbox", "pyInversion": "pyINVersion"}

HERE = os.path.dirname(os.path.abspath(__file__))
# <repo>/src/pysasfit/resources -> <repo>/docs/pySASfit-logo-bundle (only in a source checkout)
_BUNDLE = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(HERE))), "docs", "pySASfit-logo-bundle")


def _roots():
    return [HERE, _BUNDLE]


def icon_path(tool, small=False):
    """Path of the square tile (svg) of ``tool``; ``small``: thicker strokes for 16 to 32 px. None if not found."""
    tool = ALIASES.get(tool, tool)
    name = f"{tool}-icon-small.svg" if small else f"{tool}-icon.svg"
    for root in _roots():
        p = os.path.join(root, "icons", tool, name)
        if os.path.isfile(p):
            return p
    return None


def menu_icon_path(tool, fmt="svg"):
    """Path of the menu icon of ``tool`` (``fmt`` svg or ico). None if not found."""
    tool = ALIASES.get(tool, tool)
    for root in _roots():
        p = os.path.join(root, "menu-icons", tool, f"{tool}-menu.{fmt}")
        if os.path.isfile(p):
            return p
    return None


def logo_path(tool, variant="light", fmt="svg"):
    """Path of the horizontal logo of ``tool``: ``variant`` light, dark, mono-black or mono-white. None if not found."""
    tool = ALIASES.get(tool, tool)
    for root in _roots():
        p = os.path.join(root, "logos", tool, fmt, f"{tool}-{variant}.{fmt}")
        if os.path.isfile(p):
            return p
    return None


def tool_icon(tool):
    """``QIcon`` of ``tool`` for windows, the task bar, tabs and menus: the menu icon, else the tile
    (:func:`tile_icon`); empty if the files are missing."""
    if ALIASES.get(tool, tool) in EXTERNAL_TOOLS:
        return external_icon(tool)
    path = menu_icon_path(tool)
    if path is None:
        return tile_icon(tool)
    from PySide6 import QtGui

    return QtGui.QIcon(path)          # svg: sharp at every size


_EXTERNAL_CACHE = {}


def external_icon(tool, atsas_dir=None):
    """``QIcon`` of the ATSAS installation (``tool`` "pyATsas": the ATSAS icon in all sizes it provides; empty
    without ATSAS). Not used for the pyATsas tab, which has its own logo."""
    from PySide6 import QtCore, QtGui

    key = (tool, str(atsas_dir or ""))
    if key in _EXTERNAL_CACHE:
        return _EXTERNAL_CACHE[key]
    icon = QtGui.QIcon()
    tool = ALIASES.get(tool, tool)
    # pyATsas: the original ATSAS icon of the installation (preferences, last search, else the default install
    # folders -- file tests only, no program is started); pySASfit draws no substitute
    try:
        from ..atsas.core import icon_files
        for size, path in icon_files("atsas", atsas_dir or _atsas_pref()):
            icon.addFile(str(path), QtCore.QSize(size, size))
    except Exception:  # noqa: BLE001 -- no ATSAS: no icon
        pass
    if not icon.isNull():
        _EXTERNAL_CACHE[key] = icon          # not cached when empty: ATSAS may be installed / found later
    return icon


def _atsas_pref():
    try:
        from ..core import settings as prefs
        return (prefs.get("atsas.dir") or "").strip() or None
    except Exception:  # noqa: BLE001
        return None


def tile_icon(tool):
    """``QIcon`` of the navy tile of ``tool`` (the heavy-stroke version for small sizes); empty if missing."""
    from PySide6 import QtCore, QtGui

    icon = QtGui.QIcon()
    big, small = icon_path(tool), icon_path(tool, small=True)
    for size in (16, 24, 32):
        if small or big:
            icon.addFile(small or big, QtCore.QSize(size, size))
    for size in (48, 64, 128, 256):
        if big or small:
            icon.addFile(big or small, QtCore.QSize(size, size))
    return icon


def wordmark(tool):
    """Segments ``[(text, colour on light, colour on dark, bold), ...]`` of the tool's wordmark
    (for :meth:`pysasfit.core.tabs.DetachableTabWidget.set_wordmark`); None for unknown names."""
    tool = ALIASES.get(tool, tool)
    parts = WORDMARKS.get(tool)
    if parts is None:
        return None
    core = TOOL_COLOURS.get(tool, COLOURS["core"])
    return [(parts[0], *COLOURS["py"], False), (parts[1], *core, True),
            (parts[2], *COLOURS["suffix"], False)]
