"""Structures built from linked sub-units and their scattering.

Formalism (Svaneborg & Pedersen, J. Chem. Phys. 136, 104105 (2012)):
for a tree of sub-units I with excess scattering lengths beta_I,

    S(q) = sum_I beta_I^2 F_I
         + sum_{I != J} beta_I beta_J A_{I,a} [prod_K Psi_{K,b,c}] A_{J,d}

where the product runs over the sub-units K on the path between I and J,
entering K at reference point b and leaving at c; a and d are the
reference points through which the path leaves I and enters J.
The normalised form factor is S / (sum_I beta_I)^2.

The double sum is evaluated with a tree recursion in O(N) (instead of
O(N^2) pairs), so dendrimers and brushes with thousands of sub-units are
cheap.  ``form_factor_pairs`` is the direct O(N^2) sum, kept for testing.
"""
from collections import defaultdict, deque
import itertools

import numpy as np

from .subunits import REGISTRY, SubUnit
from . import special as _sf

#: how the size parameters of a sub-unit scale with its mass m (polydisperse sub-units): {param: exponent};
#: "nu" means the value of the sub-unit's own parameter nu
MASS_SCALING = {
    "GaussianPolymer": {"Rg": 0.5}, "GaussianLoop": {"Rg": 0.5},
    "FloryPolymer": {"Rg": "nu"}, "FloryLoop": {"Rg": "nu"}, "ExcludedVolumePolymer": {"Rg": "nu"},
    "WormLikeChain": {"L": 1.0}, "WormLikeChainEV": {"L": 1.0},
    "ThinRod": {"L": 1.0}, "ThinCircle": {"R": 1.0}, "ThinDisk": {"R": 0.5}, "ThinSphericalShell": {"R": 0.5},
    "SolidSphere": {"R": 1 / 3}, "SolidSphericalShell": {"Ri": 1 / 3, "Ro": 1 / 3},
    "SolidCylinder": {"L": 1.0}, "Spheroid": {"a": 1 / 3, "c": 1 / 3},
}
#: sub-units that can be given a cross-section (thick chains)
THICK_KINDS = ("GaussianPolymer", "GaussianLoop", "FloryPolymer", "FloryLoop", "ExcludedVolumePolymer",
               "WormLikeChain", "WormLikeChainEV", "ThinRod", "ThinCircle")
N_POLY = 32


def schulz_zimm_nodes(sigma, n=N_POLY):
    """Masses m (mean 1) and weights of a Schulz-Zimm distribution with relative width sigma: generalised
    Gauss-Laguerre (exact moments) for sigma >= 0.1, Gauss-Hermite (normal approximation) below."""
    from scipy.special import roots_genlaguerre, roots_hermitenorm
    if sigma >= 0.1:
        k = 1.0 / sigma**2
        x, w = roots_genlaguerre(n, k - 1.0)
        m = x / k
    else:
        x, w = roots_hermitenorm(20)
        m = 1.0 + sigma * x
    w = w / w.sum()
    return m, w


class _Entry:
    __slots__ = ("name", "kind", "tag", "model", "poly", "thick", "copy")

    def __init__(self, name, kind, tag, model, poly=False, thick=False, copy=()):
        self.name, self.kind, self.tag, self.model = name, kind, tag, model
        self.poly, self.thick, self.copy = bool(poly), bool(thick), tuple(tuple(c) for c in (copy or ()))


class Structure:
    """A flat structure: named sub-units joined at reference points.

    Reference points are written ``"name.ref"`` or, for distributed
    references, ``"name.ref#label"``: the same label means the same point,
    different labels independent random points.  An unlabelled distributed
    reference gets a fresh label (a new random point).

    Parameters of sub-unit ``name`` with tag ``t`` (default ``t = name``) are
    ``beta_t`` and its size parameters ``Rg_t``, ``R_t``, ``L_t`` ...; sub-units
    sharing a tag share parameters.
    """

    def __init__(self, name="structure"):
        self.name = name
        self._units = {}           # name -> _Entry
        self._parent = {}          # union-find over vertex keys
        self._auto = itertools.count()
        self._interact = set()     # parameter names p of copy groups with excluded volume (vex_p)

    # -- building ---------------------------------------------------------
    def add(self, kind, name, tag=None, model=None, poly=False, thick=False, copy=()):
        """Add a sub-unit.  ``kind`` is a type name (see ``REGISTRY``) or a
        SubUnit instance (custom sub-unit or non-default quadrature).

        ``poly``: Schulz-Zimm polydisperse in mass, relative width ``sigma_<tag>`` (sizes scale as in
        ``MASS_SCALING``, beta with the mass); ``thick``: cylindrical cross-section of radius ``Rc_<tag>`` (chain-like
        sub-units, ``THICK_KINDS``); ``copy``: [(group, index, parameter), ...] -- the sub-unit belongs to copy
        ``index`` of the group (outermost first); copies of a group declared with :meth:`interact` see each other."""
        if name in self._units:
            raise ValueError(f"sub-unit '{name}' already exists")
        if "." in name or "#" in name:
            raise ValueError("sub-unit names must not contain '.' or '#'")
        if isinstance(kind, SubUnit):
            model, kind = kind, kind.kind
        if model is None:
            if kind not in REGISTRY:
                raise KeyError(f"unknown sub-unit type '{kind}'")
            model = REGISTRY[kind]()
        if thick and kind not in THICK_KINDS:
            raise ValueError(f"{kind} cannot be thick (only {', '.join(THICK_KINDS)})")
        if poly and kind not in MASS_SCALING:
            raise ValueError(f"{kind} cannot be polydisperse")
        self._units[name] = _Entry(name, kind, tag or name, model, poly, thick, copy)
        return name

    def interact(self, param):
        """Excluded volume between the copies of the groups whose copy labels name ``param`` (e.g. crowded micelle
        coronas): a correlation hole in the relative positions of two copies. Every interference term between
        sub-units I, J of different copies becomes

            beta_I beta_J [(1 + k) (A ... Psi ... A) - k F_c(q)],    k = phi / (n - 1),

        phi = vex_<param> in [0, 1) the strength of the hole (the n - 1 other copies together remove up to the
        fraction phi of one copy's own scattering), F_c the normalised scattering of one copy (all its sub-units,
        averaged over the copies). At q = 0 every term is unchanged, so the forward scattering keeps its exact value; at large q the
        interference of a group is at least -phi times its self terms, so the intensity stays positive."""
        self._interact.add(param)
        return self

    def link(self, *args, tag=None, poly=False, thick=False, copy=()):
        """``link("a.end2", "b.end1")`` joins two existing reference points;
        ``link("GaussianPolymer", "c1.end1", "core.surface#x", tag="chain")``
        first adds sub-unit ``c1`` of the given type, then links."""
        if len(args) == 3:
            kind, new_ref, old_ref = args
            self.add(kind, new_ref.split(".")[0], tag=tag, poly=poly, thick=thick, copy=copy)
        elif len(args) == 2:
            new_ref, old_ref = args
        else:
            raise TypeError("link(ref1, ref2) or link(kind, new_ref, old_ref)")
        a, b = self._vertex(new_ref), self._vertex(old_ref)
        self._union(a, b)
        return self

    def include(self, other, prefix):
        """Copy all sub-units and links of ``other`` with names prefixed by
        ``prefix`` (tags are kept, so parameters stay shared)."""
        for e in other._units.values():
            self.add(e.model, prefix + e.name, tag=e.tag, poly=e.poly, thick=e.thick,
                     copy=[(prefix + g, i, pn) for g, i, pn in e.copy])
        self._interact |= other._interact
        for key in other._parent:
            root = other._find(key)
            k2 = (prefix + key[0],) + key[1:]
            r2 = (prefix + root[0],) + root[1:]
            self._parent.setdefault(k2, k2)
            self._parent.setdefault(r2, r2)
            self._union(k2, r2)
        return self

    def ref(self, name, refname, label=None):
        """Build a reference string."""
        return f"{name}.{refname}" + (f"#{label}" if label is not None else "")

    # -- vertex bookkeeping ------------------------------------------------
    def _vertex(self, s):
        if "." not in s:
            raise ValueError(f"reference '{s}' must be 'name.ref[#label]'")
        name, rest = s.split(".", 1)
        ref, _, label = rest.partition("#")
        if name not in self._units:
            raise KeyError(f"no sub-unit '{name}'")
        model = self._units[name].model
        model.check_ref(ref)
        if model.is_distributed(ref):
            if not label:
                label = f"_auto{next(self._auto)}"
        else:
            label = ""
        key = (name, ref, label)
        self._parent.setdefault(key, key)
        return key

    def _find(self, k):
        while self._parent[k] != k:
            self._parent[k] = self._parent[self._parent[k]]
            k = self._parent[k]
        return k

    def _union(self, a, b):
        ra, rb = self._find(a), self._find(b)
        if ra != rb:
            self._parent[ra] = rb

    def _graph(self):
        """Bipartite graph sub-unit <-> junction vertex.  Returns
        unit_vertices[name] = list of (vertex_id, ref) and
        vertex_units[vid] = list of (name, ref)."""
        roots = {}
        unit_vertices = {n: [] for n in self._units}
        vertex_units = defaultdict(list)
        for key in self._parent:
            r = self._find(key)
            vid = roots.setdefault(r, len(roots))
            name, ref, _ = key
            if any(v == vid for v, _ in unit_vertices[name]):
                raise ValueError(f"sub-unit '{name}' is linked to itself (loop)")
            unit_vertices[name].append((vid, ref))
            vertex_units[vid].append((name, ref))
        # tree test: connected and edges = nodes - 1
        n_nodes = len(self._units) + len(roots)
        n_edges = sum(len(v) for v in unit_vertices.values())
        if len(self._units) > 1 and n_edges != n_nodes - 1:
            raise ValueError("structure is not a tree (loops or disconnected parts)")
        return unit_vertices, vertex_units

    # -- parameters ---------------------------------------------------------
    def parameters(self):
        """Names of all parameters, sorted."""
        out = set()
        for e in self._units.values():
            if e.kind != "Point":
                out.add(f"beta_{e.tag}")
            out.update(f"{s}_{e.tag}" for s in e.model.size_params)
            if e.poly:
                out.add(f"sigma_{e.tag}")
            if e.thick:
                out.add(f"Rc_{e.tag}")
            for _g, _i, pn in e.copy:
                if pn in self._interact:
                    out.add(f"vex_{pn}")
        return sorted(out)

    def _unit_params(self, e, params):
        try:
            p = {s: float(params[f"{s}_{e.tag}"]) for s in e.model.size_params}
            if e.poly:
                p["_sigma"] = float(params[f"sigma_{e.tag}"])
            if e.thick:
                p["_Rc"] = float(params[f"Rc_{e.tag}"])
            return p
        except KeyError as err:
            raise KeyError(f"missing parameter {err} for sub-unit '{e.name}'") from None

    def _beta(self, e, params):
        if e.kind == "Point":
            return 0.0
        return float(params.get(f"beta_{e.tag}", 1.0))

    # -- evaluation ---------------------------------------------------------
    class _Cache:
        def __init__(self, q):
            self.q, self.d = q, {}

        def get(self, e, p, what, *refs):
            """F (beta^2-weighted average <m^2 F> for polydisperse sub-units), A (<m A>) or P (<Psi>), with the
            cross-section factors of thick sub-units."""
            key = (id(e.model), e.kind, tuple(sorted(p.items())), what, refs)
            if key not in self.d:
                base = {k: v for k, v in p.items() if not k.startswith("_")}
                sigma = p.get("_sigma", 0.0)
                if sigma > 1e-6:
                    m, w = schulz_zimm_nodes(sigma)
                    scale = MASS_SCALING[e.kind]
                    v = 0.0
                    for mi, wi in zip(m, w):
                        pi = dict(base)
                        for k, ex in scale.items():
                            pi[k] = base[k] * mi ** (base["nu"] if ex == "nu" else ex)
                        f = self._one(e.model, pi, what, refs)
                        v = v + wi * f * (mi * mi if what == "F" else mi if what == "A" else 1.0)
                else:
                    v = self._one(e.model, base, what, refs)
                if "_Rc" in p and what != "P":
                    acs = _sf.jinc(self.q * p["_Rc"])
                    v = v * (acs * acs if what == "F" else acs)
                self.d[key] = np.asarray(v, dtype=float)
            return self.d[key]

        def _one(self, m, p, what, refs):
            if what == "F":
                return m.ff(self.q, p)
            if what == "A":
                return m.amp(self.q, p, refs[0])
            return m.psi(self.q, p, *refs)

        def hole_k(self, structure, group, param, params):
            """k = phi/(n - 1) of the correlation hole of copy group `group`, phi = vex_<param>, n copies."""
            phi = float(params.get(f"vex_{param}", 0.0))
            if not 0.0 <= phi < 1.0:
                raise ValueError(f"vex_{param} = {phi} must be in [0, 1)")
            n = structure._group_size(group)
            return phi / (n - 1) if n > 1 else 0.0

    def form_factor(self, q, params, normalized=True):
        """Form factor of the structure.  ``normalized=False`` returns the
        unnormalised S(q) (S(0) = (sum beta)^2)."""
        q = np.atleast_1d(np.asarray(q, dtype=float))
        S = self._tree_sum(q, params)
        if not normalized:
            return S
        return S / self._beta_sum(params) ** 2

    def _beta_sum(self, params):
        """sqrt(S(0)) without excluded volume: (sum beta)^2 + sum of the variances of polydisperse betas."""
        tot = sum(self._beta(e, params) for e in self._units.values())
        var = sum((self._beta(e, params) * float(params[f"sigma_{e.tag}"])) ** 2
                  for e in self._units.values() if e.poly)
        return np.sqrt(tot * tot + var)

    def _group_size(self, group):
        """Number of copies of a group instance."""
        sizes = getattr(self, "_sizes", None)
        if sizes is None or sizes[0] != len(self._units):
            cnt = defaultdict(set)
            for e in self._units.values():
                for g, i, _pn in e.copy:
                    cnt[g].add(i)
            sizes = self._sizes = (len(self._units), {g: len(v) for g, v in cnt.items()})
        return sizes[1].get(group, 1)

    def _group_ref(self, param):
        """The first sub-unit of copy 0 of a group with parameter `param` (its F enters the excluded volume)."""
        for e in self._units.values():
            for _g, i, pn in e.copy:
                if pn == param:
                    return e.name
        raise KeyError(param)

    def _active_group(self, e):
        """(group, param) of the copy labels of e whose parameter has excluded volume, outermost first."""
        return [(g, pn) for g, _i, pn in e.copy if pn in self._interact]

    def _tree_sum(self, q, params):
        if self._interact and any(self._active_group(e) for e in self._units.values()):
            # q = 0 appended: the correlation hole needs the beta sums of the copies
            return self._tree_sum_q(np.append(q, 0.0), params)[:-1]
        return self._tree_sum_q(q, params)

    def _tree_sum_q(self, q, params):
        uv, vu = self._graph()
        names = list(self._units)
        cache = self._Cache(q)
        P = {n: self._unit_params(self._units[n], params) for n in names}
        B = {n: self._beta(self._units[n], params) for n in names}

        # BFS over the bipartite tree from the first sub-unit that is not a copy
        order, parent = [], {}
        root = next((n for n in names if not self._units[n].copy), names[0])
        start = ("u", root)
        parent[start] = None
        dq = deque([start])
        while dq:
            node = dq.popleft()
            order.append(node)
            if node[0] == "u":
                nbrs = [("v", v) for v, _ in uv[node[1]]]
            else:
                nbrs = [("u", n) for n, _ in vu[node[1]]]
            for nb in nbrs:
                if nb not in parent:
                    parent[nb] = node
                    dq.append(nb)
        if sum(1 for n in order if n[0] == "u") != len(names):
            raise ValueError("structure is disconnected")

        T = {}    # sub-unit -> amplitude of its subtree seen from parent vertex
        W = {}    # node -> scattering of its subtree alone (self + interference terms inside it)
        Sv = {}   # vertex -> summed amplitude of everything below it
        Svg = {}  # vertex -> {group: (summed amplitude, summed W) of the copies of that group below it}
        groups = {n: self._active_group(self._units[n]) for n in names} if self._interact else {}

        def own_group(n, above):
            """The interacting group whose copies meet at this junction: the innermost label of n that the
            parent (above) does not carry."""
            ga = groups.get(above, []) if above else []
            for g, pn in groups.get(n, []):
                if (g, pn) not in ga:
                    return (g, pn)
            return None

        def hole(g, pair_amp, pair_beta, wsum):
            """Correlation-hole correction of the interference between the copies of group g."""
            k = cache.hole_k(self, g[0], g[1], params)
            Fc = wsum / wsum[-1]                 # normalised scattering of one copy (q = 0 appended)
            return k * (pair_amp - Fc * pair_beta)

        for node in reversed(order):
            if node[0] == "v":
                kidn = [n for n, _ in vu[node[1]] if parent.get(("u", n)) == node]
                kids = [T[n] for n in kidn]
                s = np.sum(kids, axis=0) if kids else np.zeros_like(q)
                local = s * s - sum(t * t for t in kids)
                Sv[node[1]] = s
                if groups:
                    up = parent.get(node)
                    gsum = {}
                    for n in kidn:
                        g = own_group(n, up[1] if up else None)
                        if g is not None:
                            a, h, w = gsum.get(g, (0.0, 0.0, 0.0))
                            gsum[g] = (a + T[n], h + T[n] * T[n], w + W[n])
                    for g, (a, h, w) in gsum.items():
                        local = local + hole(g, a * a - h, a[-1] ** 2 - h[-1], w)
                    Svg[node[1]] = {g: (a, w) for g, (a, _h, w) in gsum.items()}
                W[node] = local + sum(W[n] for n in kidn)
                continue
            name = node[1]
            e, p, b = self._units[name], P[name], B[name]
            local = b * b * cache.get(e, p, "F") if b != 0.0 else np.zeros_like(q)
            pv = parent[node]
            pref = None
            G, H = {}, {}
            below = 0.0
            for vid, ref in uv[name]:
                if pv is not None and vid == pv[1]:
                    pref = ref
                    continue
                s = Sv[vid]
                G[ref] = G.get(ref, 0.0) + s
                H[ref] = H.get(ref, 0.0) + s * s
                below = below + W[("v", vid)]

            def bA(ref):
                return b * cache.get(e, p, "A", ref) if b != 0.0 else 0.0

            refs = list(G)
            for r in refs:
                local = local + 2.0 * bA(r) * G[r]
            for i, r in enumerate(refs):
                if e.model.is_distributed(r):
                    psi_rr = cache.get(e, p, "P", r, r)
                    local = local + psi_rr * (G[r] * G[r] - H[r])
                    if groups:
                        # copies of one group at different random points of r
                        gg = {}
                        for vid, ref in uv[name]:
                            if ref != r or (pv is not None and vid == pv[1]):
                                continue
                            for g, (a, w) in Svg.get(vid, {}).items():
                                s1, s2, sw = gg.get(g, (0.0, 0.0, 0.0))
                                gg[g] = (s1 + a, s2 + a * a, sw + w)
                        for g, (s1, s2, sw) in gg.items():
                            local = local + hole(g, psi_rr * (s1 * s1 - s2), s1[-1] ** 2 - s2[-1], sw)
                for u in refs[i + 1:]:
                    local = local + 2.0 * cache.get(e, p, "P", *sorted((r, u))) * G[r] * G[u]
            W[name] = local + below
            if pref is not None:
                t = bA(pref) + np.zeros_like(q)
                for r in refs:
                    t = t + cache.get(e, p, "P", *sorted((pref, r))) * G[r]
                T[name] = t
        return W[root] + np.zeros_like(q)

    def _effective_copies(self):
        """Copy labels as the tree sum sees them: a sub-unit hanging below a copy (seen from the first sub-unit
        that is not a copy) belongs to that copy unless it carries longer labels of its own."""
        uv, vu = self._graph()
        names = list(self._units)
        root = next((n for n in names if not self._units[n].copy), names[0])
        eff = {root: self._units[root].copy}
        dq = deque([root])
        while dq:
            u = dq.popleft()
            for vid, _r in uv[u]:
                for n, _r2 in vu[vid]:
                    if n not in eff:
                        own = self._units[n].copy
                        eff[n] = own if len(own) >= len(eff[u]) else eff[u]
                        dq.append(n)
        return eff

    def _pair_group(self, I, J):
        """(group, parameter) of the interacting group in which I and J are in different copies, or None."""
        eff = getattr(self, "_eff_cache", None)
        if eff is None:
            eff = self._eff_cache = self._effective_copies()
        a, b = eff[I], eff[J]
        for (ga, ia, pa), (gb, ib, pb) in zip(a, b):
            if ga != gb:
                return None
            if ia != ib:
                return (ga, pa) if pa in self._interact else None
        return None

    def form_factor_pairs(self, q, params, normalized=True):
        """Direct O(N^2) double sum (reference implementation for tests)."""
        q = np.atleast_1d(np.asarray(q, dtype=float))
        holes = bool(self._interact)
        if holes:
            q = np.append(q, 0.0)
        self._eff_cache = None
        uv, vu = self._graph()
        cache = self._Cache(q)
        names = list(self._units)
        P = {n: self._unit_params(self._units[n], params) for n in names}
        B = {n: self._beta(self._units[n], params) for n in names}
        selfs = {n: B[n] ** 2 * cache.get(self._units[n], P[n], "F") for n in names}
        terms = {}
        for i, I in enumerate(names):
            # BFS from I: record (previous unit, entry ref) to rebuild paths
            prev = {("u", I): None}
            dq = deque([("u", I)])
            while dq:
                node = dq.popleft()
                if node[0] == "u":
                    for vid, ref in uv[node[1]]:
                        nb = ("v", vid)
                        if nb not in prev:
                            prev[nb] = (node, ref)
                            dq.append(nb)
                else:
                    for n, ref in vu[node[1]]:
                        nb = ("u", n)
                        if nb not in prev:
                            prev[nb] = (node, ref)
                            dq.append(nb)
            for J in names[i + 1:]:
                if B[I] == 0.0 or B[J] == 0.0:
                    continue
                # walk back J -> I
                node = ("u", J)
                vnode, dref = prev[node]
                term = cache.get(self._units[J], P[J], "A", dref)
                while True:
                    unode, exit_ref = prev[vnode]          # unit that leads to vnode, via exit_ref
                    if unode == ("u", I):
                        term = term * cache.get(self._units[I], P[I], "A", exit_ref)
                        break
                    vnode, entry_ref = prev[unode]
                    K = unode[1]
                    term = term * cache.get(self._units[K], P[K], "P",
                                            *sorted((entry_ref, exit_ref)))
                terms[(I, J)] = B[I] * B[J] * term
        if holes:
            # correlation holes, innermost groups first: the scattering of a copy includes its own holes
            eff = self._effective_copies()
            depth = max((len(v) for v in eff.values()), default=0)
            for lev in range(depth - 1, -1, -1):
                inst = {}
                for n in names:
                    if len(eff[n]) > lev:
                        g, idx, pn = eff[n][lev]
                        if pn in self._interact:
                            inst.setdefault((g, pn, tuple(eff[n][:lev])), {}).setdefault(idx, []).append(n)
                for (g, pn, _up), copies in inst.items():
                    k = cache.hole_k(self, g, pn, params)
                    wsum = 0.0
                    for members in copies.values():
                        ms = set(members)
                        wsum = wsum + sum(selfs[m] for m in members) + \
                            2.0 * sum(t for (a, b2), t in terms.items() if a in ms and b2 in ms)
                    Fc = wsum / wsum[-1]
                    where = {m: idx for idx, members in copies.items() for m in members}
                    for (a, b2), t in list(terms.items()):
                        if a in where and b2 in where and where[a] != where[b2]:
                            terms[(a, b2)] = (1.0 + k) * t - k * Fc * t[-1]
        total = sum(selfs.values()) + 2.0 * sum(terms.values()) if terms else sum(selfs.values())
        if holes:
            total = total[:-1]
        if not normalized:
            return total
        return total / self._beta_sum(params) ** 2

    # -- misc ---------------------------------------------------------------
    def set_quadrature(self, n_mu):
        """Gauss-Legendre points of the orientational average of all oriented sub-units
        (SolidCylinder, Spheroid, ThinDisk)."""
        from .subunits import OrientedBody
        for e in self._units.values():
            if isinstance(e.model, OrientedBody):
                e.model.set_n_mu(n_mu)
        return self

    def units(self):
        """[(name, kind, tag)] of all sub-units."""
        return [(e.name, e.kind, e.tag) for e in self._units.values()]

    def links(self):
        """Junctions: one list of 'name.ref[#label]' per point that joins two or more sub-units."""
        groups = defaultdict(list)
        for key in self._parent:
            groups[self._find(key)].append(key)
        out = []
        for keys in groups.values():
            if len(keys) > 1:
                out.append([f"{n}.{r}" + (f"#{lab}" if lab and not lab.startswith("_auto") else "")
                            for n, r, lab in sorted(keys)])
        return out

    def radius_of_gyration2(self, params, q0=None):
        """Rg^2 from the Guinier expansion F = 1 - q^2 Rg^2/3 + O(q^4),
        by Richardson extrapolation of 3(1-F)/q^2."""
        scale = max([v for k, v in params.items() if not k.startswith("beta_")] + [1.0])
        q0 = q0 or 1e-2 / scale
        qs = np.array([q0, q0 / 2])
        F = self.form_factor(qs, params)
        g = 3 * (1 - F) / qs**2
        return (4 * g[1] - g[0]) / 3

    def __repr__(self):
        return f"<Structure {self.name}: {len(self._units)} sub-units>"
