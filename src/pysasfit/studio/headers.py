"""Plugin catalogue from the Doxygen comments of the SASfit plugin headers.

Every SASfit plugin header ``sasfit_<collection>.h`` documents each model
function in a block::

    /* ################ start ff_profile__power_law ################ */
    /**
     * \\defgroup ff_profile__power_law profile: power law Sphere
     * \\ingroup ff_plugins_fuzzy_sph
     * \\brief ...
     * \\par Required parameters:
     *      <table border="0"><tr>
     *       <td>\\b R</td>
     *       <td>radius</td>
     *      </tr> ... </table>
     */

The block gives the function name (``ff_profile__power_law``, exported by the
plugin library ``libsasfit_<collection>`` as ``sasfit_ff_profile__power_law``
plus ``_f`` and ``_v``), the label shown in the SASfit GUI, the menu group,
a brief description and the ordered parameter list. Optional third and fourth
table cells are read as default value and bounds.

No library is loaded here, so the catalogue works on every platform.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from functools import lru_cache

from . import paths

__all__ = [
    "ParameterInfo",
    "PluginInfo",
    "parse_header",
    "catalogue",
    "search",
    "info",
    "PLUGIN_TYPES",
    "write_profiles",
    "core_infos",
]

PLUGIN_TYPES = {
    "ff": "form factor",
    "sq": "structure factor",
    "sd": "size distribution",
    "peak": "peak",
    "peaks": "peak",
}

_SECTION = re.compile(r"/\*\s*#+\s*start\s+(\S+)\s*#+\s*\*/")
_DEFGROUP = re.compile(r"\\defgroup\s+(\S+)[ \t]*([^\n]*)")
_INGROUP = re.compile(r"\\ingroup\s+(\S+)")
_BRIEF = re.compile(r"\\brief\s+(.*?)(?:\n\s*\*\s*\n|\n\s*\*\s*\\)", re.S)
_PARAMS = re.compile(r"\\par\s+Required parameters:(.*?)(?:</table>|\*/)", re.S)
_CELL = re.compile(r"<td[^>]*>(.*?)</td>", re.S)
_TAG = re.compile(r"<[^>]+>")


def _clean(s):
    s = re.sub(r"\n\s*\*", " ", s)
    s = s.replace("\\<", "\x00lt").replace("\\>", "\x00gt")  # escaped < > are text, not tags
    s = _TAG.sub("", s)
    s = s.replace("\x00lt", "<").replace("\x00gt", ">")
    return " ".join(s.split())


@dataclass
class ParameterInfo:
    name: str
    description: str = ""
    default: float = 0.0
    bounds: tuple = (float("-inf"), float("inf"))

    def __str__(self):
        return f"{self.name}: {self.description} [default {self.default:g}]"


@dataclass
class PluginInfo:
    """Description of one SASfit model function."""

    name: str  # e.g. 'ff_profile__power_law' (export name without 'sasfit_')
    label: str  # GUI label, e.g. 'profile: power law Sphere'
    collection: str  # header/library stem, e.g. 'sasfit_powerlaw'
    group: str = ""
    brief: str = ""
    parameters: list = field(default_factory=list)
    header: str = ""

    @property
    def type(self):
        return self.name.split("_", 1)[0]

    @property
    def type_name(self):
        return PLUGIN_TYPES.get(self.type, "unspecified plugin type")

    @property
    def short_name(self):
        """Name without the type prefix (the name SASstudio used: 'profile__power_law')."""
        return self.name.split("_", 1)[1] if "_" in self.name else self.name

    @property
    def symbol(self):
        return "sasfit_" + self.name

    @property
    def library(self):
        return self.collection if self.collection == "libsasfit" else "lib" + self.collection

    @property
    def builtin(self):
        """True for functions compiled into libsasfit itself (no plugin header)."""
        return self.collection == "libsasfit"

    @property
    def parameter_names(self):
        return [p.name for p in self.parameters]

    @property
    def defaults(self):
        return [p.default for p in self.parameters]

    def __str__(self):
        lines = [f"{self.name}  ({self.type_name}: {self.label})", f"  library: {self.library}, group: {self.group}"]
        if self.brief:
            lines.append(f"  {self.brief}")
        for i, p in enumerate(self.parameters):
            lines.append(f"  p[{i}] {p}")
        return "\n".join(lines)


def _brief(raw):
    b = _clean(raw)
    return "" if b.startswith("<some brief description") else b


def _parse_float(s, default=0.0):
    try:
        return float(s)
    except (TypeError, ValueError):
        return default


def _parse_bounds(s):
    nums = re.findall(r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?|[-+]?inf", s or "")
    if len(nums) >= 2:
        return (float(nums[0]), float(nums[1]))
    return (float("-inf"), float("inf"))


def _parse_parameters(block):
    params = []
    cur = None
    for raw in _CELL.findall(block):
        cell = raw.strip()
        m = re.match(r"\\b\s+(.*)", cell, re.S)
        if m:  # new parameter name
            cur = [_clean(m.group(1))]
            params.append(cur)
        elif cur is not None:
            cur.append(_clean(cell))
    out = []
    for c in params:
        out.append(
            ParameterInfo(
                name=c[0],
                description=c[1] if len(c) > 1 else "",
                default=_parse_float(c[2]) if len(c) > 2 else 0.0,
                bounds=_parse_bounds(c[3]) if len(c) > 3 else (float("-inf"), float("inf")),
            )
        )
    return out


def parse_header(filename):
    """Return the list of :class:`PluginInfo` documented in one header file."""
    with open(filename, "r", errors="replace") as fh:
        text = fh.read()
    collection = os.path.splitext(os.path.basename(filename))[0]
    marks = list(_SECTION.finditer(text))
    out = []
    for i, m in enumerate(marks):
        sec = text[m.end() : marks[i + 1].start() if i + 1 < len(marks) else len(text)]
        d = _DEFGROUP.search(sec)
        name = d.group(1) if d else m.group(1)
        label = _clean(d.group(2)) if d else name
        g = _INGROUP.search(sec)
        b = _BRIEF.search(sec)
        p = _PARAMS.search(sec)
        out.append(
            PluginInfo(
                name=name,
                label=label,
                collection=collection,
                group=g.group(1) if g else "",
                brief=_brief(b.group(1)) if b else "",
                parameters=_parse_parameters(p.group(1)) if p else [],
                header=filename,
            )
        )
    return out


def core_infos():
    """PluginInfo entries for the model functions built into libsasfit."""
    from .core_functions import CORE_EXPORTS, CORE_LABELS, CORE_PARAMETERS

    out = []
    for name, _, _ in CORE_EXPORTS:
        pars = [ParameterInfo(n, d) for n, d in CORE_PARAMETERS.get(name, [])]
        out.append(
            PluginInfo(name=name, label=CORE_LABELS.get(name, name.split("_", 1)[1]), collection="libsasfit",
                       group="libsasfit", parameters=pars)
        )
    return out


@lru_cache(maxsize=8)
def _catalogue(dirs):
    cat = {p.name: p for p in core_infos()}
    for d in dirs:  # later directories override earlier ones
        if not os.path.isdir(d):
            continue
        for fn in sorted(os.listdir(d)):
            if fn.startswith("sasfit_") and fn.endswith(".h"):
                for pi in parse_header(os.path.join(d, fn)):
                    cat[pi.name] = pi
    return cat


def catalogue(header_dirs=None):
    """Dict name -> :class:`PluginInfo` of all documented SASfit functions.

    ``header_dirs`` defaults to :func:`pysasfit.studio.paths.header_dirs`
    (the headers shipped with pysasfit, then those next to the plugin
    libraries, which take precedence).
    """
    dirs = tuple(header_dirs) if header_dirs is not None else tuple(paths.header_dirs())
    return dict(_catalogue(dirs))


def info(name, header_dirs=None):
    """Look up one function by its full name ('ff_sphere') or short name ('sphere')."""
    cat = catalogue(header_dirs)
    if name in cat:
        return cat[name]
    key = name.lower().replace(" ", "_")
    hits = [p for p in cat.values() if p.short_name.lower() == key or p.name.lower() == key]
    if len(hits) == 1:
        return hits[0]
    if not hits:
        hits = [p for p in cat.values() if p.label.lower() == name.lower()]
        if len(hits) == 1:
            return hits[0]
    if len(hits) > 1:
        raise KeyError(f"'{name}' is ambiguous: {', '.join(h.name for h in hits)}")
    raise KeyError(f"no SASfit function '{name}' (try pysasfit.studio.search('{name}'))")


def search(*words, type=None, header_dirs=None):
    """Functions whose name, label, group or brief contains all ``words`` (case-insensitive)."""
    words = [w.lower() for w in words]
    out = []
    for p in catalogue(header_dirs).values():
        if type is not None and p.type != type:
            continue
        hay = " ".join([p.name, p.label, p.group, p.brief]).lower()
        if all(w in hay for w in words):
            out.append(p)
    return sorted(out, key=lambda p: p.name)


def write_profiles(out_dir, header_dirs=None, source_dir=None):
    """Write one ``profile_<name>.txt`` per function (port of SASstudio's generate_plugin_profiles.py).

    Each file lists the parameters with their descriptions. If ``source_dir``
    (a SASfit ``src/plugins`` tree) is given, the ``#define X param->p[i]``
    lines of ``sasfit_<name>.c`` are prepended, as in the original script.
    """
    os.makedirs(out_dir, exist_ok=True)
    written = []
    sources = {}
    if source_dir:
        for root, _, files in os.walk(source_dir):
            for f in files:
                if f.startswith("sasfit_") and f.endswith(".c"):
                    sources[f[7:-2]] = os.path.join(root, f)
    for p in catalogue(header_dirs).values():
        text = ""
        src = sources.get(p.name)
        if src:
            with open(src, errors="replace") as fh:
                for line in fh:
                    if "#define" in line and "param->p" in line:
                        text += line.split("#define", 1)[1].lstrip()
        text += "".join(f"{q.name}: {q.description}\n" for q in p.parameters)
        fn = os.path.join(out_dir, f"profile_{p.name}.txt")
        with open(fn, "w") as fh:
            fh.write(text)
        written.append(fn)
    return written
