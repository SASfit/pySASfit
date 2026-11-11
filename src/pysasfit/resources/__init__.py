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
(model explorer), ``pySASred``, ``pyTOOLbox``, ``pySCAtter``, ``pySASmorph``. The former names pyInversion and pySASbox are accepted
(:data:`ALIASES`).

``pyATsas`` (the ATSAS panel) has no logo in the bundle: its icon is the original ATSAS icon, read at run time
from the user's ATSAS installation (``share/icons/hicolor/<n>x<n>/apps/atsas.png``, see
:func:`pysasfit.atsas.core.icon_files`); pySASfit does not ship a copy. Without ATSAS a plain placeholder is drawn.

``pyDENsity`` (DENSS): DENSS 1.8.7 has no logo (the package contains no image file). If an installed DENSS ships
one (an image named ``*logo*`` or ``*icon*`` in its package folder), it is used; otherwise pyDENsity has its own icon
(contour lines of a density map on its blue tile).
"""

from __future__ import annotations

import os

__all__ = ["TOOLS", "EXTERNAL_TOOLS", "WORDMARKS", "COLOURS", "TOOL_COLOURS", "ALIASES", "tool_icon", "tile_icon", "logo_path",
           "icon_path", "menu_icon_path", "wordmark"]

TOOLS = ("pySASfit", "pySLDprofile", "pyINVersion", "pyOZsolver", "pySASstudio", "pySASred", "pyTOOLbox", "pySCAtter",
         "pySASmorph")

#: tools whose icon comes from an external program's installation, not from the logo bundle
EXTERNAL_TOOLS = ("pyATsas", "pyDENsity")

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
    "pyATsas": ("py", "AT", "sas"),
    "pyDENsity": ("py", "DEN", "sity"),
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
    "pyATsas": ("#0B6666", "#5FA8A8"),         # teal of the ATSAS icon tile
    "pyDENsity": ("#2B4C9B", "#8FA6D9"),       # deep blue (own icon: contour lines of a density map)
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
    """``QIcon`` of a tool in :data:`EXTERNAL_TOOLS`: pyATsas -> the ATSAS icon of the installation (all sizes
    it provides), else a plain placeholder (teal square with a p(r) curve)."""
    from PySide6 import QtCore, QtGui

    key = (tool, str(atsas_dir or ""))
    if key in _EXTERNAL_CACHE:
        return _EXTERNAL_CACHE[key]
    icon = QtGui.QIcon()
    tool = ALIASES.get(tool, tool)
    if tool == "pyDENsity":
        for path in _denss_logo_files():
            icon.addFile(str(path))
        if icon.isNull():
            icon = _density_icon(TOOL_COLOURS[tool][0])
        _EXTERNAL_CACHE[key] = icon
        return icon
    try:
        from ..atsas.core import icon_files
        for size, path in icon_files("atsas", atsas_dir):
            icon.addFile(str(path), QtCore.QSize(size, size))
    except Exception:  # noqa: BLE001 -- no ATSAS: placeholder
        pass
    if icon.isNull():
        icon = _placeholder_icon(TOOL_COLOURS.get(tool, COLOURS["core"])[0])
    _EXTERNAL_CACHE[key] = icon
    return icon


def _denss_logo_files():
    """Image files named *logo* / *icon* in the installed DENSS package (none in DENSS 1.8.7). Found without
    importing DENSS."""
    try:
        import importlib.util
        spec = importlib.util.find_spec("denss")
    except (ImportError, ValueError):
        return []
    if spec is None or not spec.submodule_search_locations:
        return []
    out = []
    for root in spec.submodule_search_locations:
        for dirpath, _dirs, files in os.walk(root):
            for f in files:
                low = f.lower()
                if low.endswith((".png", ".svg", ".ico")) and ("logo" in low or "icon" in low):
                    out.append(os.path.join(dirpath, f))
    return sorted(out)


def _density_icon(colour):
    """pyDENsity's own icon: nested contour lines of a two-lobed density on the tool's tile."""
    import math
    from PySide6 import QtCore, QtGui

    icon = QtGui.QIcon()
    for s in (16, 24, 32, 48, 64, 128, 256):
        pm = QtGui.QPixmap(s, s)
        pm.fill(QtCore.Qt.GlobalColor.transparent)
        p = QtGui.QPainter(pm)
        p.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
        p.setBrush(QtGui.QColor(colour))
        p.setPen(QtCore.Qt.PenStyle.NoPen)
        p.drawRoundedRect(QtCore.QRectF(0, 0, s, s), s * 0.16, s * 0.16)
        levels = (1.0, 0.68, 0.38) if s >= 32 else (1.0, 0.55)
        for k, lev in enumerate(levels):
            path = QtGui.QPainterPath()
            n = 72
            for i in range(n + 1):
                t = 2 * math.pi * i / n
                r = lev * (0.30 + 0.07 * math.cos(2 * t) + 0.035 * math.cos(3 * t + 0.6))
                x = 0.5 + r * math.cos(t) * 1.12
                y = 0.52 + r * math.sin(t) * 0.92
                path.moveTo(x * s, y * s) if i == 0 else path.lineTo(x * s, y * s)
            path.closeSubpath()
            c = QtGui.QColor("white")
            c.setAlphaF(1.0 if k == 0 else 0.85)
            if k == len(levels) - 1:
                p.setBrush(QtGui.QColor(255, 255, 255, 235))
            else:
                p.setBrush(QtCore.Qt.BrushStyle.NoBrush)
            p.setPen(QtGui.QPen(c, max(1.1, s / 22)))
            p.drawPath(path)
        p.end()
        icon.addPixmap(pm)
    return icon


def _placeholder_icon(colour):
    from PySide6 import QtCore, QtGui

    icon = QtGui.QIcon()
    for s in (16, 24, 32, 48, 64, 128):
        pm = QtGui.QPixmap(s, s)
        pm.fill(QtCore.Qt.GlobalColor.transparent)
        p = QtGui.QPainter(pm)
        p.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
        p.setBrush(QtGui.QColor(colour))
        p.setPen(QtCore.Qt.PenStyle.NoPen)
        p.drawRoundedRect(QtCore.QRectF(0, 0, s, s), s * 0.15, s * 0.15)
        pen = QtGui.QPen(QtGui.QColor("white"), max(1.2, s / 16))
        p.setPen(pen)
        path = QtGui.QPainterPath()
        n = 24
        for i in range(n + 1):                       # p(r) of a sphere: a skewed bell
            x = i / n
            y = 12 * x * x * (1 - 1.5 * x + 0.5 * x ** 3)
            px, py = s * (0.15 + 0.7 * x), s * (0.82 - 0.62 * min(y / 1.6, 1))
            path.moveTo(px, py) if i == 0 else path.lineTo(px, py)
        p.drawPath(path)
        p.end()
        icon.addPixmap(pm)
    return icon


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
