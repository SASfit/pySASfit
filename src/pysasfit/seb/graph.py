"""Designs: structures drawn as graphs of sub-units, patterns and components (the pySEBpolymers designer).

A :class:`Design` is a graph whose nodes are

* **sub-units** (``type="unit"``): one sub-unit of the library (GaussianPolymer, SolidSphere, ...); its ports are the
  reference points of that sub-unit;
* **patterns**, which stand for many sub-units:

  - ``Array``: ``n`` copies of an element, each attached by the element's in-port to the point the array's
    ``anchor`` port is linked to -- at independent random points of a distributed host point (``mode="independent"``,
    e.g. chains on a surface) or all at the same point (``mode="same"``);
  - ``Chain``: ``n`` copies joined out-port -> in-port; ports ``start`` and ``end``;
  - ``Generations``: a dendrimer, ``f`` elements at the ``root`` point and ``f - 1`` new elements at the out-port of
    every element of the previous generation, ``g`` generations (optionally one tag per generation);

  the element is a sub-unit type or a **component**;
* components are designs of their own with an ``in`` port (and for chains and generations an ``out`` port), each on a
  sub-unit node of the component, e.g. a bottle-brush segment (backbone block + side chain at its end).

Links join two ports. On a distributed port (``surface``, ``contour``, ``hull`` ...) every link has a label: the same
label on the same node is the same point, different labels are independent random points. The node graph must be a
tree (no loops); :meth:`Design.link` refuses links that would close one.

:meth:`Design.to_code` writes Python code that builds the :class:`~pysasfit.seb.Structure` ``s`` with loops for the
patterns and one function per component; the panel builds the structure from that code, so designs, scripts and
"Copy Python code" are always the same thing. :meth:`Design.from_structure` draws an existing structure.
"""
from __future__ import annotations

import copy
import json
import re
from collections import deque

from .subunits import REGISTRY

__all__ = ["Design", "Node", "Link", "PATTERNS", "UNIT_KINDS", "FORMAT", "short_name", "node_width"]

FORMAT = "pysebpolymers-design-1"
PATTERNS = ("Array", "Chain", "Generations")
UNIT_KINDS = tuple(k for k in REGISTRY)
PATTERN_PORTS = {"Array": ("anchor",), "Chain": ("start", "end"), "Generations": ("root",)}
NAME_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_]*$")
_SHORT = {"GaussianPolymer": "chain", "GaussianLoop": "loop", "FloryPolymer": "fchain", "FloryLoop": "floop", "ThinRod": "rod", "SolidSphere": "sphere",
          "ThinSphericalShell": "shell", "SolidSphericalShell": "shell", "ThinCircle": "ring", "ThinDisk": "disk",
          "SolidCylinder": "cyl", "Spheroid": "spheroid", "Point": "point", "Array": "array", "Chain": "chainof",
          "Generations": "dendron"}


def node_width(design, node):
    """Width of a node on the canvas (pixels at zoom 1)."""
    if node.is_unit:
        return max(150.0, 24.0 + 58.0 * len(REGISTRY[node.kind].references()))
    return 196.0


def short_name(kind):
    return _SHORT.get(kind, kind.lower())


def _lit(content):
    """Python literal of an f-string content: plain string if it has no fields."""
    if content is None:
        return "None"
    return ('f"' if "{" in content else '"') + content + '"'


class Node:
    """A node of a design. ``type`` is "unit" or a pattern; ``kind`` the sub-unit type of a unit node; ``element``
    the sub-unit type or component of a pattern (``element_is_component``); ``tag`` the parameter tag of a unit
    node or of the copies of a sub-unit element."""

    FIELDS = ("id", "type", "name", "kind", "tag", "x", "y", "element", "element_is_component", "n", "mode",
              "in_ref", "out_ref", "f", "g", "gen_tags", "poly", "thick", "vex")

    def __init__(self, id, type, name, kind="", tag="", x=0.0, y=0.0, element="", element_is_component=False, n=10,
                 mode="independent", in_ref="", out_ref="", f=3, g=3, gen_tags=True, poly=False, thick=False,
                 vex=False):
        self.id, self.type, self.name, self.kind, self.tag = int(id), type, name, kind, tag
        self.x, self.y = float(x), float(y)
        self.element, self.element_is_component = element, bool(element_is_component)
        self.n, self.mode, self.in_ref, self.out_ref = int(n), mode, in_ref, out_ref
        self.f, self.g, self.gen_tags = int(f), int(g), bool(gen_tags)
        self.poly, self.thick, self.vex = bool(poly), bool(thick), bool(vex)   # polydisperse, cross-section, EV

    @property
    def is_unit(self):
        return self.type == "unit"

    def to_dict(self):
        d = {k: getattr(self, k) for k in self.FIELDS}
        if self.is_unit:
            for k in ("element", "element_is_component", "n", "mode", "in_ref", "out_ref", "f", "g", "gen_tags", "vex"):
                d.pop(k)
        return d

    @classmethod
    def from_dict(cls, d):
        return cls(**{k: v for k, v in d.items() if k in cls.FIELDS})

    def __repr__(self):
        return f"<Node {self.id} {self.type} {self.name}>"


class Link:
    """``a = (node id, port, label)`` -- ``b`` likewise; the label matters only on distributed ports."""

    def __init__(self, a, b):
        self.a, self.b = tuple(a), tuple(b)

    def ends(self):
        return (self.a, self.b)

    def other(self, node_id):
        return self.b if self.a[0] == node_id else self.a

    def end_of(self, node_id):
        return self.a if self.a[0] == node_id else self.b

    def to_dict(self):
        return {"a": list(self.a), "b": list(self.b)}


class Design:
    """A graph of sub-units and patterns; see the module documentation."""

    def __init__(self, name="structure", components=None):
        self.name = name
        self.nodes = {}
        self.links = []
        self.ports = {}                       # component designs: {"in"/"out": (node id, ref, label)}
        self.components = {} if components is None else components   # shared by a design and its components
        self._next = 1

    # ------------------------------------------------------------------ nodes
    def unique_name(self, base):
        base = re.sub(r"[^A-Za-z0-9_]", "", base) or "u"
        if not base[0].isalpha():
            base = "u" + base
        names = {n.name for n in self.nodes.values()}
        k = 1
        while f"{base}{k}" in names:
            k += 1
        return f"{base}{k}"

    def add_unit(self, kind, name=None, tag=None, x=0.0, y=0.0):
        if kind not in REGISTRY:
            raise KeyError(f"unknown sub-unit type '{kind}'")
        name = name or self.unique_name(short_name(kind))
        self._check_name(name)
        nd = Node(self._next, "unit", name, kind=kind, tag=tag or name, x=x, y=y)
        self.nodes[nd.id] = nd
        self._next += 1
        return nd

    def add_pattern(self, ptype, element="GaussianPolymer", component=False, name=None, tag=None, x=0.0, y=0.0,
                    **props):
        if ptype not in PATTERNS:
            raise KeyError(f"unknown pattern '{ptype}'")
        name = name or self.unique_name(short_name(ptype) if component else short_name(element) + "s")
        self._check_name(name)
        nd = Node(self._next, ptype, name, tag=tag or (short_name(element) if not component else ""), x=x, y=y,
                  element=element, element_is_component=component, **props)
        if not component and element in REGISTRY:
            refs = REGISTRY[element].references()
            nd.in_ref = nd.in_ref or ("end1" if "end1" in refs else refs[0])
            nd.out_ref = nd.out_ref or ("end2" if "end2" in refs else refs[-1])
        self.nodes[nd.id] = nd
        self._next += 1
        return nd

    def _check_name(self, name, node=None):
        if not NAME_RE.match(name or ""):
            raise ValueError(f"'{name}': names start with a letter and contain only letters, digits and _")
        if any(n.name == name and n is not node for n in self.nodes.values()):
            raise ValueError(f"name '{name}' is already used")

    def rename(self, node_id, name):
        nd = self.nodes[node_id]
        self._check_name(name, nd)
        nd.name = name

    def remove_node(self, node_id):
        self.nodes.pop(node_id, None)
        self.links = [ln for ln in self.links if ln.a[0] != node_id and ln.b[0] != node_id]
        self.ports = {k: v for k, v in self.ports.items() if v[0] != node_id}

    def by_name(self, name):
        for n in self.nodes.values():
            if n.name == name:
                return n
        raise KeyError(name)

    # ------------------------------------------------------------------ ports
    def ports_of(self, node):
        if node.is_unit:
            return REGISTRY[node.kind].references()
        return PATTERN_PORTS[node.type]

    def element_refs(self, node):
        """(in, out) of a pattern's element: reference names of a sub-unit type, or 'in'/'out' of a component."""
        if node.element_is_component:
            c = self.components.get(node.element)
            if c is None:
                return None, None
            return ("in" if "in" in c.ports else None), ("out" if "out" in c.ports else None)
        return node.in_ref or None, node.out_ref or None

    def is_distributed(self, node, port):
        """Does a link to this port need a label?"""
        if node.is_unit:
            return REGISTRY[node.kind].is_distributed(port)
        return False      # pattern ports: the element's own points

    def labels_on(self, node_id, port):
        out = []
        for ln in self.links:
            for e in ln.ends():
                if e[0] == node_id and e[1] == port and e[2] not in out:
                    out.append(e[2])
        return out

    def new_label(self, node_id, port):
        used = set(self.labels_on(node_id, port))
        k = 1
        while f"p{k}" in used:
            k += 1
        return f"p{k}"

    def links_of(self, node_id):
        return [ln for ln in self.links if ln.a[0] == node_id or ln.b[0] == node_id]

    def _connected(self, a, b):
        adj = self._adjacency()
        seen, dq = {a}, deque([a])
        while dq:
            u = dq.popleft()
            if u == b:
                return True
            for v in adj.get(u, ()):
                if v not in seen:
                    seen.add(v)
                    dq.append(v)
        return False

    def _adjacency(self):
        adj = {i: [] for i in self.nodes}
        for ln in self.links:
            adj[ln.a[0]].append(ln.b[0])
            adj[ln.b[0]].append(ln.a[0])
        return adj

    def can_link(self, a_id, a_port, b_id, b_port):
        """(True, "") or (False, reason)."""
        if a_id == b_id:
            return False, "a sub-unit cannot be linked to itself"
        for nid, port in ((a_id, a_port), (b_id, b_port)):
            nd = self.nodes.get(nid)
            if nd is None or port not in self.ports_of(nd):
                return False, f"no port {port}"
            if nd.type == "Array" and self.links_of(nid):
                return False, f"array '{nd.name}' is already anchored"
        na, nb = self.nodes[a_id], self.nodes[b_id]
        if na.type == "Array" and nb.type == "Array":
            return False, "an array must be anchored at a sub-unit or a chain end"
        if self._connected(a_id, b_id):
            return False, "this link would close a loop (structures must be trees)"
        return True, ""

    def link(self, a_id, a_port, b_id, b_port, a_label=None, b_label=None):
        ok, why = self.can_link(a_id, a_port, b_id, b_port)
        if not ok:
            raise ValueError(why)
        la = (a_label or self.new_label(a_id, a_port)) if self.is_distributed(self.nodes[a_id], a_port) else ""
        lb = (b_label or self.new_label(b_id, b_port)) if self.is_distributed(self.nodes[b_id], b_port) else ""
        ln = Link((a_id, a_port, la), (b_id, b_port, lb))
        self.links.append(ln)
        return ln

    def set_label(self, link, node_id, label):
        if not NAME_RE.match(label or ""):
            raise ValueError("labels start with a letter and contain only letters, digits and _")
        i = 0 if link.a[0] == node_id else 1
        e = list(link.ends()[i])
        e[2] = label
        if i == 0:
            link.a = tuple(e)
        else:
            link.b = tuple(e)

    def unlink(self, link):
        self.links = [ln for ln in self.links if ln is not link]

    # ------------------------------------------------------------------ components
    def new_component(self, name):
        if not NAME_RE.match(name or ""):
            raise ValueError("component names start with a letter and contain only letters, digits and _")
        if name in self.components or name in REGISTRY:
            raise ValueError(f"'{name}' is already used")
        c = Design(name, components=self.components)
        self.components[name] = c
        return c

    def set_port(self, which, node_id, ref):
        """Make ``ref`` of unit node ``node_id`` the component's ``which`` ('in'/'out') port."""
        nd = self.nodes[node_id]
        if not nd.is_unit:
            raise ValueError("component ports must be on a sub-unit")
        label = "c" if REGISTRY[nd.kind].is_distributed(ref) else ""
        self.ports[which] = (node_id, ref, label)

    def component_from(self, node_ids, name):
        """New component made of copies of the nodes ``node_ids`` and the links among them."""
        c = self.new_component(name)
        ids = set(node_ids)
        for i in sorted(ids):
            nd = copy.deepcopy(self.nodes[i])
            c.nodes[nd.id] = nd
            c._next = max(c._next, nd.id + 1)
        c.links = [copy.deepcopy(ln) for ln in self.links if ln.a[0] in ids and ln.b[0] in ids]
        return c

    def _uses(self, name, seen=None):
        """Component names used by this design, recursively."""
        seen = set() if seen is None else seen
        for nd in self.nodes.values():
            if not nd.is_unit and nd.element_is_component and nd.element not in seen:
                seen.add(nd.element)
                c = self.components.get(nd.element)
                if c is not None:
                    c._uses(name, seen)
        return seen

    # ------------------------------------------------------------------ checks
    def problems(self, component=False):
        """Reasons why the design cannot be built (empty: fine)."""
        out = []
        if not self.nodes:
            return ["empty design"]
        adj = self._adjacency()
        start = next(iter(self.nodes))
        seen, dq = {start}, deque([start])
        while dq:
            u = dq.popleft()
            for v in adj[u]:
                if v not in seen:
                    seen.add(v)
                    dq.append(v)
        if len(seen) != len(self.nodes):
            loose = [self.nodes[i].name for i in self.nodes if i not in seen]
            out.append("not connected: " + ", ".join(loose[:6]) + (" …" if len(loose) > 6 else ""))
        from .structure import MASS_SCALING, THICK_KINDS
        for nd in self.nodes.values():
            kind = nd.kind if nd.is_unit else (None if nd.element_is_component else nd.element)
            if kind is not None and nd.thick and kind not in THICK_KINDS:
                out.append(f"'{nd.name}': {kind} cannot be thick")
            if kind is not None and nd.poly and kind not in MASS_SCALING:
                out.append(f"'{nd.name}': {kind} cannot be polydisperse")
            if nd.is_unit:
                continue
            if nd.element_is_component:
                c = self.components.get(nd.element)
                if c is None:
                    out.append(f"'{nd.name}': no component '{nd.element}'")
                    continue
                if nd.element in c._uses(nd.element) or nd.element == self.name and component:
                    out.append(f"component '{nd.element}' contains itself")
                if "in" not in c.ports:
                    out.append(f"component '{nd.element}' has no in-port")
                if nd.type in ("Chain", "Generations") and "out" not in c.ports:
                    out.append(f"component '{nd.element}' has no out-port (needed by '{nd.name}')")
            else:
                refs = REGISTRY[nd.element].references() if nd.element in REGISTRY else ()
                if nd.element not in REGISTRY:
                    out.append(f"'{nd.name}': unknown element '{nd.element}'")
                elif nd.in_ref not in refs or (nd.type != "Array" and nd.out_ref not in refs):
                    out.append(f"'{nd.name}': choose the in/out points of the element")
                elif nd.type != "Array" and nd.in_ref == nd.out_ref and not REGISTRY[nd.element].is_distributed(nd.in_ref):
                    out.append(f"'{nd.name}': in and out are the same point")
            if nd.type == "Array" and not self.links_of(nd.id):
                out.append(f"array '{nd.name}' is not anchored")
            if nd.n < 1 or (nd.type == "Generations" and (nd.f < 1 or nd.g < 1)):
                out.append(f"'{nd.name}': counts must be at least 1")
        if component and "in" not in self.ports:
            out.append(f"component '{self.name}': no in-port (right-click a point of a sub-unit)")
        if not component:
            for cname in self._uses(self.name):
                c = self.components.get(cname)
                if c is not None:
                    out += [p for p in c.problems(component=True) if p not in out]
        return out

    def count_units(self):
        """Number of sub-units the design expands to."""
        total = 0
        for nd in self.nodes.values():
            if nd.is_unit:
                total += 1
                continue
            per = self.components[nd.element].count_units() if nd.element_is_component and \
                nd.element in self.components else 1
            if nd.type == "Generations":
                copies = sum(nd.f * (nd.f - 1) ** k for k in range(nd.g)) if nd.f > 1 else nd.f * nd.g
            else:
                copies = nd.n
            total += copies * per
        return total

    # ------------------------------------------------------------------ code
    def to_code(self):
        """Python code that builds the Structure ``s``."""
        probs = self.problems()
        if probs:
            raise ValueError("; ".join(probs))
        used = self._uses(self.name)
        labels = any(nd.type == "Array" and nd.vex for d in [self] + [self.components[c] for c in used]
                     for nd in d.nodes.values())
        lines = [f"# design '{self.name}' (pySEBpolymers designer): {len(self.nodes)} nodes, "
                 f"{self.count_units()} sub-units"]
        if used:
            lines += ["", "def attach(s, kind, new, at, tag, **kw):",
                      "    \"\"\"add a sub-unit as new = 'name.ref[#label]', joined at the point at (None: not joined)\"\"\"",
                      "    if at is None:",
                      "        s.add(kind, new.split('.')[0], tag=tag, **kw)",
                      "    else:",
                      "        s.link(kind, new, at, tag=tag, **kw)", ""]
            for cname in _component_order(self, used):
                lines += _component_function(self.components[cname], labels) + [""]
        lines.append(f"s = Structure({self.name!r})")
        lines += _Emitter(self, prefix="", sfx="", labels=labels, cp="[]").emit(entry=None)
        return "\n".join(lines) + "\n"

    def to_structure(self):
        from .templates import build
        return build(self.to_code())

    # ------------------------------------------------------------------ persistence
    def to_dict(self, with_components=True):
        d = {"format": FORMAT, "name": self.name, "nodes": [n.to_dict() for n in self.nodes.values()],
             "links": [ln.to_dict() for ln in self.links], "ports": {k: list(v) for k, v in self.ports.items()}}
        if with_components:
            d["components"] = {k: c.to_dict(with_components=False) for k, c in self.components.items()}
        return d

    @classmethod
    def from_dict(cls, d, components=None):
        if components is None and d.get("format") not in (None, FORMAT):
            raise ValueError(f"not a pySEBpolymers design ({d.get('format')})")
        des = cls(d.get("name", "structure"), components=components)
        for nd in d.get("nodes", []):
            n = Node.from_dict(nd)
            des.nodes[n.id] = n
            des._next = max(des._next, n.id + 1)
        des.links = [Link(ln["a"], ln["b"]) for ln in d.get("links", [])]
        des.ports = {k: tuple(v) for k, v in (d.get("ports") or {}).items()}
        if components is None:
            for k, cd in (d.get("components") or {}).items():
                des.components[k] = cls.from_dict(cd, components=des.components)
        return des

    def to_json(self):
        return json.dumps(self.to_dict(), indent=1)

    @classmethod
    def from_json(cls, text):
        return cls.from_dict(json.loads(text))

    def copy(self):
        return Design.from_dict(self.to_dict())

    # ------------------------------------------------------------------ import
    @classmethod
    def from_structure(cls, s, max_units=400):
        """Draw a built Structure (one node per sub-unit)."""
        units = s.units()
        if len(units) > max_units:
            raise ValueError(f"{len(units)} sub-units: too many to draw one by one")
        des = cls(s.name)
        ids = {}
        for name, kind, tag in units:
            nd = des.add_unit(kind, name=name if NAME_RE.match(name) else None, tag=tag)
            ids[name] = nd.id
        groups = {}
        for key in s._parent:
            groups.setdefault(s._find(key), []).append(key)
        for keys in groups.values():
            if len(keys) < 2:
                continue
            keys = sorted(keys)
            n0, r0, l0 = keys[0]
            for n1, r1, l1 in keys[1:]:
                la = l0 if l0 and not l0.startswith("_auto") else ""
                lb = l1 if l1 and not l1.startswith("_auto") else ""
                ln = des.link(ids[n0], r0, ids[n1], r1, la or None, lb or None)
                if la == "" and des.is_distributed(des.nodes[ids[n0]], r0):
                    # first end of a junction: all links of this junction use the same label
                    la = ln.a[2]
                    l0 = la
        des.auto_layout()
        return des

    # ------------------------------------------------------------------ layout
    def auto_layout(self, dx=None, dy=125.0):
        """Tree layout: the node with most links at the left, depth to the right."""
        if not self.nodes:
            return
        if dx is None:
            dx = max(node_width(self, n) for n in self.nodes.values()) + 70.0
        adj = self._adjacency()
        placed = set()
        y_next = [0.0]

        def size_of(u, parent):
            kids = [v for v in adj[u] if v != parent]
            return max(1, sum(size_of(v, u) for v in kids))

        def place(u, parent, depth, y0):
            placed.add(u)
            kids = [v for v in adj[u] if v != parent and v not in placed]
            total = sum(size_of(v, u) for v in kids)
            y = y0
            for v in kids:
                h = size_of(v, u)
                place(v, u, depth + 1, y)
                y += h * dy
            nd = self.nodes[u]
            nd.x = depth * dx
            nd.y = y0 + (max(total, 1) - 1) * dy / 2
            return max(total, 1)

        for comp_start in sorted(self.nodes, key=lambda i: -len(adj[i])):
            if comp_start in placed:
                continue
            n = place(comp_start, None, 0, y_next[0])
            y_next[0] += (n + 1) * dy


# ---------------------------------------------------------------------- code generation
def _component_order(design, used):
    """Components in dependency order (used ones first)."""
    out = []

    def visit(name, stack=()):
        if name in out or name in stack:
            return
        c = design.components[name]
        for sub in sorted(c._uses(name)):
            if sub != name:
                visit(sub, stack + (name,))
        out.append(name)
    for name in sorted(used):
        visit(name)
    return out


def _component_function(comp, labels=False):
    lines = [f"def {comp.name}(s, p, at, sfx='', via='in'" + (", cp=()" if labels else "") + "):",
             f"    \"\"\"component '{comp.name}': {_port_text(comp, 'in')}"
             + (f", out {_port_text(comp, 'out')}" if "out" in comp.ports else "")
             + " (names prefixed by p, tags suffixed by sfx)\"\"\""]
    vias = ["in"] + (["out"] if "out" in comp.ports else [])
    for k, via in enumerate(vias):
        body = _Emitter(comp, prefix="{p}", sfx="{sfx}", labels=labels, cp="list(cp)").emit(entry=(comp.ports[via], "at"))
        if len(vias) == 1:
            lines += ["    " + ln if ln else "" for ln in body]
        else:
            lines.append(f"    {'if' if k == 0 else 'else:'}" + (f" via == {via!r}:" if k == 0 else ""))
            lines += ["        " + ln if ln else "" for ln in body]
    return lines


def _port_text(comp, which):
    nid, ref, _label = comp.ports[which]
    return f"in {comp.nodes[nid].name}.{ref}" if which == "in" else f"{comp.nodes[nid].name}.{ref}"


class _Emitter:
    """Writes the code of one design body (main structure or a component function)."""

    VARS = "ijkmn"

    def __init__(self, design, prefix, sfx, labels=False, cp="[]"):
        self.d, self.P, self.S = design, prefix, sfx
        self.labels, self.cp = labels, cp       # copy labels (only when an Array has excluded volume)
        self.addr = {}          # node id -> {port: address content (without label) , distributed}
        self.lines = []
        self.var = 0

    def _v(self):
        v = self.VARS[self.var % len(self.VARS)] + ("" if self.var < len(self.VARS) else str(self.var))
        self.var += 1
        return v

    def tag(self, t):
        return t + self.S

    def kw(self, nd, copy=None):
        """Keyword arguments for the options of a sub-unit (node nd) and its copy labels."""
        out = ""
        if nd.poly:
            out += ", poly=True"
        if nd.thick:
            out += ", thick=True"
        if self.labels:
            c = copy or self.cp
            if c != "[]":
                out += f", copy={c}"
        return out

    # address of a port of a placed node, as an expression (string literal or variable)
    def address(self, nid, port, label):
        base, dist = self.addr[nid][port]
        if base.startswith("$"):                # a variable holding the address
            return base[1:]
        return _lit(base + (f"#{label or 'p'}" if dist and label is not None else ""))

    def base_of(self, nid, port):
        return self.addr[nid][port]

    def emit(self, entry):
        """entry: None (main: start at a sub-unit) or ((node id, port, label), at expression)."""
        d = self.d
        adj = {}
        for ln in d.links:
            adj.setdefault(ln.a[0], []).append(ln)
            adj.setdefault(ln.b[0], []).append(ln)
        if entry is None:
            units = [n for n in d.nodes.values() if n.type in ("unit", "Chain", "Generations")]
            root = max(units, key=lambda n: (len(adj.get(n.id, ())), n.is_unit, -n.id)) if units else None
            if root is None:
                raise ValueError("the design needs a sub-unit, a chain or generations to start from")
            self.place(root, None, None, None)
        else:
            (nid, ref, label), at = entry
            self.place(d.nodes[nid], ref, label, at)
            root = d.nodes[nid]
        done, dq = {root.id}, deque([root.id])
        while dq:
            u = dq.popleft()
            for ln in adj.get(u, ()):
                ue, ve = ln.end_of(u), ln.other(u)
                if ve[0] in done:
                    continue
                v = d.nodes[ve[0]]
                host_base, host_dist = self.base_of(u, ue[1])
                self.place(v, ve[1], ve[2], self.address(u, ue[1], ue[2]),
                           host=(host_base, host_dist, ue[2]))
                done.add(v.id)
                dq.append(v.id)
        return self.lines

    # -------------------------------------------------------------- one node
    def place(self, nd, port, label, at, host=None):
        P, add = self.P, self.lines.append
        if nd.is_unit:
            cls = REGISTRY[nd.kind]
            add(f"# {nd.name}: {nd.kind}")
            kw = self.kw(nd)
            if at is None:
                add(f"s.add({nd.kind!r}, {_lit(P + nd.name)}, tag={_lit(self.tag(nd.tag))}{kw})")
            elif at == "at":            # entry of a component function: at may be None
                new = P + nd.name + "." + port + (f"#{label}" if cls.is_distributed(port) else "")
                add(f"attach(s, {nd.kind!r}, {_lit(new)}, at, {_lit(self.tag(nd.tag))}{kw})")
            else:
                new = P + nd.name + "." + port + (f"#{label}" if cls.is_distributed(port) else "")
                add(f"s.link({nd.kind!r}, {_lit(new)}, {at}, tag={_lit(self.tag(nd.tag))}{kw})")
            self.addr[nd.id] = {r: (P + nd.name + "." + r, cls.is_distributed(r)) for r in cls.references()}
            return
        getattr(self, "_" + nd.type.lower())(nd, port, at, host)

    def _element(self, nd, name_content, at, sfx_extra="", via="in", copy=None):
        """Code that adds one element copy named name_content (f-string content) at the expression at."""
        if nd.element_is_component:
            sfx = self.S + sfx_extra
            return [f"{nd.element}(s, {_lit(name_content + '_')}, {at}, sfx={_lit(sfx)}"
                    + (f", via={via!r}" if via != "in" else "")
                    + (f", cp={copy or self.cp}" if self.labels else "") + ")"]
        ref = nd.in_ref if via == "in" else nd.out_ref
        dist = REGISTRY[nd.element].is_distributed(ref)
        new = name_content + "." + ref + ("#c" if dist else "")
        tag = _lit(self.tag(nd.tag) + sfx_extra)
        kw = self.kw(nd, copy)
        if at == "None":
            return [f"s.add({nd.element!r}, {_lit(name_content)}, tag={tag}{kw})"]
        return [f"s.link({nd.element!r}, {_lit(new)}, {at}, tag={tag}{kw})"]

    def _elem_port(self, nd, name_content, which):
        """(address content, distributed) of the element's in/out point."""
        if nd.element_is_component:
            c = self.d.components[nd.element]
            nid, ref, label = c.ports[which]
            unit = c.nodes[nid]
            dist = REGISTRY[unit.kind].is_distributed(ref)
            return name_content + "_" + unit.name + "." + ref + (f"#{label or 'c'}" if dist else ""), False
        ref = nd.in_ref if which == "in" else nd.out_ref
        dist = REGISTRY[nd.element].is_distributed(ref)
        return name_content + "." + ref + ("#c" if dist else ""), False

    def _array(self, nd, port, at, host):
        if at is None:
            raise ValueError(f"array '{nd.name}' is not anchored")
        i = self._v()
        base, dist, label = host
        what = nd.element + (" (component)" if nd.element_is_component else "")
        if nd.mode == "independent" and dist and not base.startswith("$"):
            target = _lit(base + "#" + nd.name + "{" + i + "}")
            how = "each at an independent random point"
        else:
            target = at
            how = "all at the same point"
        self.lines.append(f"# {nd.name}: {nd.n} x {what}, {how}"
                          + (f"; excluded volume between the copies (vex_{nd.name})" if nd.vex else ""))
        copy = None
        if nd.vex:
            self.lines.append(f"s.interact({nd.name!r})")
            copy = f"{self.cp} + [({_lit(self.P + nd.name)}, {i}, {nd.name!r})]"
        self.lines.append(f"for {i} in range({nd.n}):")
        self.lines += ["    " + c for c in self._element(nd, self.P + nd.name + "{" + i + "}", target, copy=copy)]
        self.addr[nd.id] = {"anchor": (base, dist)}

    def _chain(self, nd, port, at, host):
        i = self._v()
        n = nd.n
        name = self.P + nd.name + "{" + i + "}"
        what = nd.element + (" (component)" if nd.element_is_component else "")
        forward = port in (None, "start")
        a_in, a_out = ("in", "out") if forward else ("out", "in")
        prev = self.P + nd.name + "{" + i + (" - 1" if forward else " + 1") + "}"
        prev_addr = _lit(self._elem_port(nd, prev, a_out)[0])
        rng = f"range({n})" if forward else f"range({n} - 1, -1, -1)"
        first = "0" if forward else str(n - 1)
        self.lines.append(f"# {nd.name}: chain of {n} x {what}" + ("" if forward else ", built from its end"))
        self.lines.append(f"for {i} in {rng}:")
        at0 = at if at is not None else "None"
        self.lines.append(f"    at = {at0} if {i} == {first} else {prev_addr}")
        if not nd.element_is_component and at is None:
            # the first element is added, not linked
            self.lines += ["    if at is None:"] + ["        " + c for c in self._element(nd, name, "None")]
            self.lines += ["    else:"] + ["        " + c for c in self._element(nd, name, "at", via=a_in)]
        else:
            self.lines += ["    " + c for c in self._element(nd, name, "at", via=a_in)]
        start = self._elem_port(nd, self.P + nd.name + "0", "in")[0]
        end = self._elem_port(nd, self.P + nd.name + str(n - 1), "out")[0]
        self.addr[nd.id] = {"start": (start, False), "end": (end, False)}

    def _generations(self, nd, port, at, host):
        k, j = self._v(), self._v()
        f, g = nd.f, nd.g
        what = nd.element + (" (component)" if nd.element_is_component else "")
        gen = "{gen}" if nd.gen_tags else ""
        self.lines.append(f"# {nd.name}: {g} generations of {what}, {f} at the root, {max(f - 1, 1)} at every end"
                          + (", tags per generation" if nd.gen_tags else ""))
        root_first = self._elem_port(nd, self.P + nd.name + "0_0", "in")[0]
        rv = f"root_{nd.name}"
        self.lines.append(f"{rv} = {at if at is not None else 'None'}")
        self.lines.append("free = []")
        self.lines.append("gen = 0")
        self.lines.append(f"for {k} in range({f}):")
        name0 = self.P + nd.name + "0_{" + k + "}"
        self.lines.append(f"    at = {rv} if ({rv} is not None or {k} == 0) else {_lit(root_first)}")
        if not nd.element_is_component and at is None:
            self.lines += ["    if at is None:"] + ["        " + c for c in self._element(nd, name0, "None", gen)]
            self.lines += ["    else:"] + ["        " + c for c in self._element(nd, name0, "at", gen)]
        else:
            self.lines += ["    " + c for c in self._element(nd, name0, "at", gen)]
        self.lines.append(f"    free.append({_lit(self._elem_port(nd, name0, 'out')[0])})")
        self.lines.append(f"for gen in range(1, {g}):")
        self.lines.append("    new = []")
        self.lines.append("    for at in free:")
        self.lines.append(f"        for {j} in range({max(f - 1, 1)}):")
        namek = self.P + nd.name + "{gen}_{len(new)}"
        self.lines += ["            " + c for c in self._element(nd, namek, "at", gen)]
        self.lines.append(f"            new.append({_lit(self._elem_port(nd, namek, 'out')[0])})")
        self.lines.append("    free = new")
        self.lines.append(f"{rv} = {rv} if {rv} is not None else {_lit(root_first)}")
        self.addr[nd.id] = {"root": ("$" + rv, False)}
