r"""Menu tree of the SASfit model functions, from the Doxygen group definitions (``*.def``).

SASfit builds the pull-down menus for form factors, structure factors and size
distributions from two sources:

* the ``*.def`` files of the plugin directory (a folder ``def`` next to the plugin
  libraries, or the plugin folder itself, see :func:`pysasfit.studio.paths.def_dirs`)
  define the groups and how they nest::

      /**
       * \defgroup ff_plugins_fuzzy_obj fuzzy obj.      <- group name and menu label
       * \ingroup ff_plugins                            <- parent group (the menu level above)
       *
       * This group contains form factors with many different fuzzy cross-sectional profiles.
       */

  A group with ``\toplvldescr{form factor,sasfit_ff,form_fac}`` is the root of a menu
  (here: the form factors); groups above it (``model_functions``) are not shown;
* the plugin headers (:mod:`pysasfit.studio.headers`) name the group of every function
  with ``\ingroup``.

    defs = load_groups()                                  # all .def files; defs.files lists them
    top = build_menu(headers.catalogue().values(), ("sq",), defs)
    for path, p in iter_functions(top): ...               # path: labels of the groups above p
    print("\n".join(tree_lines(top)))

Groups without functions are left out. Functions without a (known) group are collected in a
last entry, so nothing disappears when a ``.def`` file is missing; without any ``.def`` file
the menu is a flat list.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from functools import lru_cache

from . import paths

__all__ = [
    "GroupDef",
    "GroupDefs",
    "MenuNode",
    "parse_def",
    "def_files",
    "load_groups",
    "build_menu",
    "iter_functions",
    "tree_lines",
]

_COMMENT = re.compile(r"/\*\*(.*?)\*/", re.S)
_DEFGROUP = re.compile(r"\\defgroup\s+(\S+)[ \t]*([^\n]*)")
_INGROUP = re.compile(r"\\ingroup\s+(\S+)")
_TOPLVL = re.compile(r"\\toplvldescr\s*\{([^,}]*),([^,}]*),([^,}]*)\}")
_MATH = re.compile(r"\\f\[.*?\\f\]|\\f\$(.*?)\\f\$", re.S)
_DIRECTIVE = re.compile(r"^\\(defgroup|ingroup|modname|toplvldescr|sa|see)\b")


@dataclass
class GroupDef:
    """One ``\\defgroup`` of a ``.def`` file."""

    name: str  # e.g. 'ff_plugins_fuzzy_obj'
    label: str  # menu text, e.g. 'fuzzy obj.'
    parent: str = ""  # name of the group above ('' = none)
    brief: str = ""  # description (tool tip)
    toplevel: tuple | None = None  # (description, prefix, key) of ``\toplvldescr`` roots, else None
    source: str = ""  # file the group was read from


class GroupDefs(dict):
    """Group name -> :class:`GroupDef` in definition order; ``files`` lists the ``.def`` files read."""

    files: list

    def __init__(self, *args, files=(), **kw):
        super().__init__(*args, **kw)
        self.files = list(files)


def _brief(block):
    lines = []
    for raw in block.splitlines():
        line = raw.strip().lstrip("*").strip()
        if _DIRECTIVE.match(line):
            continue
        if line.startswith("\\brief"):
            line = line[len("\\brief") :].strip()
        if line:
            lines.append(line)
    text = _MATH.sub(lambda m: m.group(1) or "", " ".join(lines))
    return " ".join(text.split())[:400]


def parse_def(filename):
    """The :class:`GroupDef` entries of one ``.def`` file, in file order."""
    with open(filename, "r", errors="replace") as fh:
        text = fh.read()
    out = []
    for block in _COMMENT.findall(text):
        d = _DEFGROUP.search(block)
        if not d:
            continue
        name = d.group(1)
        parent = _INGROUP.search(block)
        top = _TOPLVL.search(block)
        out.append(
            GroupDef(
                name=name,
                label=" ".join(d.group(2).split()) or name,
                parent=parent.group(1) if parent else "",
                brief=_brief(block),
                toplevel=tuple(s.strip() for s in top.groups()) if top else None,
                source=filename,
            )
        )
    return out


def def_files(dirs=None):
    """All ``*.def`` files in ``dirs`` (default: :func:`pysasfit.studio.paths.def_dirs`)."""
    dirs = paths.def_dirs() if dirs is None else list(dirs)
    out = []
    for d in dirs:
        if os.path.isdir(d):
            out += [os.path.join(d, f) for f in sorted(os.listdir(d)) if f.lower().endswith(".def")]
    return out


def _mtime(fn):
    try:
        return os.path.getmtime(fn)
    except OSError:
        return 0.0


@lru_cache(maxsize=8)
def _load(key):
    defs = GroupDefs(files=[fn for fn, _ in key])
    for fn, _ in key:
        try:
            parsed = parse_def(fn)
        except OSError:
            continue  # an unreadable file only loses its groups
        for g in parsed:
            defs[g.name] = g  # later files override earlier ones; the first position is kept
    return defs


def load_groups(dirs=None):
    """The group definitions of all ``.def`` files as :class:`GroupDefs` (empty if there are none)."""
    return _load(tuple((fn, _mtime(fn)) for fn in def_files(dirs)))


class MenuNode:
    """A menu level: sub-groups (``children``, in definition order) and functions."""

    def __init__(self, name="", label="", brief="", order=0):
        self.name, self.label, self.brief, self.order = name, label, brief, order
        self.children = []
        self.functions = []  # PluginInfo, in the order of the entries
        self._index = {}

    def child(self, name, label, brief="", order=0):
        node = self._index.get(name)
        if node is None:
            node = self._index[name] = MenuNode(name, label, brief, order)
            self.children.append(node)
        return node

    def count(self):
        """Number of functions in this node and below."""
        return len(self.functions) + sum(c.count() for c in self.children)

    def _sort(self):
        self.children.sort(key=lambda c: c.order)  # stable: definition order
        for c in self.children:
            c._sort()


def _group_path(group, defs):
    """``(path, root)``: the group names from the first level below the menu root down to ``group``, and
    the name of that menu root (None if there is none); ``(None, None)`` for an unknown group."""
    chain, seen = [], set()
    while group in defs and group not in seen:
        seen.add(group)
        chain.append(group)
        group = defs[group].parent
    if not chain:
        return None, None
    root = None
    for i, name in enumerate(chain):
        if defs[name].toplevel:  # the root of the menu and everything above it is not shown
            root, chain = name, chain[:i]
            break
    return chain[::-1], root


def _special(p):
    if getattr(p, "builtin", False):
        return "(built-in)", "built-in (libsasfit)", 1
    if p.collection == "python" or p.collection.endswith(".py"):
        return "(python)", "Python plugins", 2
    return "(ungrouped)", "(not in a group)", 3


def build_menu(entries, types, defs=None):
    """Menu tree (:class:`MenuNode`) of the functions of the given type(s).

    ``entries``: iterable of :class:`pysasfit.studio.headers.PluginInfo`; ``types``: 'ff', 'sq',
    'sd', ... or a tuple of them; ``defs``: :func:`load_groups` (default: read the ``.def`` files).
    """
    if isinstance(types, str):
        types = (types,)
    defs = load_groups() if defs is None else defs
    order = {name: i for i, name in enumerate(defs)}
    # the menu roots (``\toplvldescr``) of these types; groups below other roots belong to other menus
    mine = {n for n, g in defs.items() if g.toplevel and g.toplevel[1].removeprefix("sasfit_") in types}
    top = MenuNode()
    for p in entries:
        if p.type not in types:
            continue
        if not defs:
            top.functions.append(p)  # no .def files: flat list
            continue
        path, root = _group_path(p.group, defs)
        if path is None:
            key, label, k = _special(p)
            node = top.child(key, label, "", len(defs) + k)
        elif mine and root is not None and root not in mine:
            # e.g. a form factor whose header says \ingroup sq_...: that group is part of another menu
            node = top.child(
                "(other menu)",
                "(filed in a group of another menu)",
                "These functions name a group of another menu (e.g. structure factors) in their plugin header.",
                len(defs) + 4,
            )
        else:
            node = top
            for name in path:
                node = node.child(name, defs[name].label, defs[name].brief, order[name])
        node.functions.append(p)
    top._sort()
    return top


def iter_functions(top, _path=()):
    """Yield ``(path, function)`` for every function of the tree; ``path`` = labels of the groups above it."""
    for g in top.children:
        yield from iter_functions(g, _path + (g.label,))
    for p in top.functions:
        yield _path, p


def tree_lines(top, functions=True, _level=0):
    """Indented text lines of a tree (group labels with the number of functions in brackets)."""
    pad = "  " * _level
    out = []
    for g in top.children:
        out.append(f"{pad}{g.label}  [{g.count()}]")
        out += tree_lines(g, functions, _level + 1)
    if functions:
        out += [f"{pad}- {p.label}   ({p.name})" for p in top.functions]
    return out
