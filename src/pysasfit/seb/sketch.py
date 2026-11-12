"""A random 3D realisation of a structure, for looking at it (the "3D sketch" panel).

The scattering formalism averages over conformations, orientations and the random points of distributed references;
:func:`realise` draws ONE such realisation: Gaussian chains and loops as random walks with the right radius of
gyration, rods as straight lines in random directions, solid bodies at random orientations, random points on surfaces
and contours (one per label). Excluded volume is ignored, as in the model. :func:`draw` puts it into a matplotlib 3D
axes, one colour per tag.
"""
from __future__ import annotations

from collections import defaultdict, deque

import numpy as np

__all__ = ["realise", "draw", "Shape"]


class Shape:
    """kind: "line" (pts), "sphere" (centre, R), "disk" (centre, normal, R), "ring" (centre, normal, R),
    "cylinder" (centre, axis, R, L), "spheroid" (centre, axis, a, c), "point" (centre)."""

    def __init__(self, kind, tag, unit, **geo):
        self.kind, self.tag, self.unit, self.geo = kind, tag, unit, geo

    def translate(self, d):
        for k in ("pts", "centre"):
            if k in self.geo:
                self.geo[k] = self.geo[k] + d


def _rot(rng):
    """Random rotation matrix (uniform)."""
    qv = rng.normal(size=4)
    a, b, c, d = qv / np.linalg.norm(qv)
    return np.array([[a*a + b*b - c*c - d*d, 2*(b*c - a*d), 2*(b*d + a*c)],
                     [2*(b*c + a*d), a*a - b*b + c*c - d*d, 2*(c*d - a*b)],
                     [2*(b*d - a*c), 2*(c*d + a*b), a*a - b*b - c*c + d*d]])


def _unit(rng):
    v = rng.normal(size=3)
    return v / np.linalg.norm(v)


def _on_sphere(rng, R):
    return R * _unit(rng)


def _geometry(kind, p, rng, steps):
    """(shapes in the local frame, sampler(ref) -> local position)."""
    R3 = _rot(rng)
    ez = R3[:, 2]
    if kind == "GaussianPolymer":
        n = steps
        b = p["Rg"] * np.sqrt(6.0 / n)
        pts = np.vstack([np.zeros(3), np.cumsum(rng.normal(0, b / np.sqrt(3), (n, 3)), axis=0)])
        refs = {"end1": pts[0], "end2": pts[-1], "middle": pts[n // 2]}
        return [Shape("line", None, None, pts=pts)], lambda r: refs.get(r, pts[rng.integers(0, n + 1)])
    if kind == "GaussianLoop":
        n = steps
        b = p["Rg"] * np.sqrt(12.0 / n)
        w = np.vstack([np.zeros(3), np.cumsum(rng.normal(0, b / np.sqrt(3), (n, 3)), axis=0)])
        w = w - np.outer(np.linspace(0, 1, n + 1), w[-1])        # Brownian bridge: closed
        return [Shape("line", None, None, pts=w)], lambda r: w[rng.integers(0, n)]
    if kind == "ThinRod":
        L = p["L"]
        refs = {"end1": np.zeros(3), "end2": L * ez, "middle": 0.5 * L * ez}
        return [Shape("line", None, None, pts=np.array([np.zeros(3), L * ez]))], \
            lambda r: refs.get(r, rng.uniform(0, L) * ez)
    if kind in ("SolidSphere", "ThinSphericalShell"):
        R = p["R"]
        return [Shape("sphere", None, None, centre=np.zeros(3), R=R, hollow=kind != "SolidSphere")], \
            lambda r: np.zeros(3) if r == "center" else _on_sphere(rng, R)
    if kind == "SolidSphericalShell":
        Ri, Ro = p["Ri"], p["Ro"]

        def s(r):
            if r == "center":
                return np.zeros(3)
            if r == "surface":
                r = "surfaceo" if rng.uniform() < Ro**2 / (Ro**2 + Ri**2) else "surfacei"
            return _on_sphere(rng, Ro if r == "surfaceo" else Ri)
        return [Shape("sphere", None, None, centre=np.zeros(3), R=Ro, hollow=True),
                Shape("sphere", None, None, centre=np.zeros(3), R=Ri, hollow=True, inner=True)], s
    if kind in ("ThinCircle", "ThinDisk"):
        R = p["R"]
        e1, e2 = R3[:, 0], R3[:, 1]

        def s(r):
            if r == "center":
                return np.zeros(3)
            phi = rng.uniform(0, 2 * np.pi)
            rad = R if r in ("contour", "rim") else R * np.sqrt(rng.uniform())
            return rad * (np.cos(phi) * e1 + np.sin(phi) * e2)
        return [Shape("ring" if kind == "ThinCircle" else "disk", None, None, centre=np.zeros(3), normal=ez, R=R)], s
    if kind == "SolidCylinder":
        R, L = p["R"], p["L"]
        e1, e2 = R3[:, 0], R3[:, 1]

        def s(r):
            if r == "center":
                return np.zeros(3)
            if r == "top":
                return 0.5 * L * ez
            if r == "bottom":
                return -0.5 * L * ez
            if r == "surface":
                r = "hull" if rng.uniform() < L / (L + R) else "ends"
            phi = rng.uniform(0, 2 * np.pi)
            if r == "hull":
                return R * (np.cos(phi) * e1 + np.sin(phi) * e2) + rng.uniform(-0.5, 0.5) * L * ez
            rad = R * np.sqrt(rng.uniform())
            return rad * (np.cos(phi) * e1 + np.sin(phi) * e2) + rng.choice([-0.5, 0.5]) * L * ez
        return [Shape("cylinder", None, None, centre=np.zeros(3), axis=ez, R=R, L=L, e1=e1, e2=e2)], s
    if kind == "Spheroid":
        a, c = p["a"], p["c"]
        e1, e2 = R3[:, 0], R3[:, 1]

        def s(r):
            if r == "center":
                return np.zeros(3)
            if r in ("north", "pole"):
                return c * ez
            if r == "south":
                return -c * ez
            ct, phi = rng.uniform(-1, 1), rng.uniform(0, 2 * np.pi)
            st = np.sqrt(1 - ct**2)
            return a * st * (np.cos(phi) * e1 + np.sin(phi) * e2) + c * ct * ez
        return [Shape("spheroid", None, None, centre=np.zeros(3), axis=ez, a=a, c=c, e1=e1, e2=e2)], s
    # Point
    return [Shape("point", None, None, centre=np.zeros(3))], lambda r: np.zeros(3)


def realise(structure, params, seed=None, steps=40, max_units=3000):
    """One random realisation: list of :class:`Shape` in absolute coordinates (centred on the centre of mass of
    the drawn points). At most ``max_units`` sub-units are drawn (breadth first from the first one)."""
    rng = np.random.default_rng(seed)
    s = structure
    uv, vu = s._graph()
    units = s._units
    names = list(units)
    if not names:
        return []
    shapes, vertex_pos = [], {}
    first = names[0]
    dq = deque([(first, None, None)])
    seen = {first}
    while dq and len(seen) <= max_units + 1:
        name, entry_vid, entry_pos = dq.popleft()
        e = units[name]
        p = {k: float(params.get(f"{k}_{e.tag}", 1.0)) for k in e.model.size_params}
        geo, sampler = _geometry(e.kind, p, rng, steps)
        local = {}
        for vid, ref in uv[name]:
            local[vid] = sampler(ref)
        shift = np.zeros(3) if entry_vid is None else entry_pos - local[entry_vid]
        for sh in geo:
            sh.translate(shift)
            sh.tag, sh.unit = e.tag, name
            shapes.append(sh)
        for vid, _ref in uv[name]:
            pos = local[vid] + shift
            vertex_pos.setdefault(vid, pos)
            for other, _r in vu[vid]:
                if other not in seen and len(seen) < max_units:
                    seen.add(other)
                    dq.append((other, vid, vertex_pos[vid]))
    pts = [sh.geo["pts"] for sh in shapes if "pts" in sh.geo] + \
          [sh.geo["centre"][None, :] for sh in shapes if "centre" in sh.geo]
    if pts:
        com = np.vstack(pts).mean(axis=0)
        for sh in shapes:
            sh.translate(-com)
    return shapes


def _surface(sh, n=18):
    g = sh.geo
    u, v = np.meshgrid(np.linspace(0, 2 * np.pi, n), np.linspace(0, np.pi, n // 2 + 1))
    if sh.kind == "sphere":
        R = g["R"]
        x, y, z = R * np.cos(u) * np.sin(v), R * np.sin(u) * np.sin(v), R * np.cos(v)
        return g["centre"][0] + x, g["centre"][1] + y, g["centre"][2] + z
    if sh.kind == "spheroid":
        a, c, e1, e2, ax = g["a"], g["c"], g["e1"], g["e2"], g["axis"]
        X = a * np.cos(u) * np.sin(v), a * np.sin(u) * np.sin(v), c * np.cos(v)
    elif sh.kind == "cylinder":
        R, L, e1, e2, ax = g["R"], g["L"], g["e1"], g["e2"], g["axis"]
        t = np.linspace(-0.5, 0.5, n // 2 + 1)[:, None] * np.ones_like(u)
        X = R * np.cos(u), R * np.sin(u), L * t
    else:                                           # disk
        e1 = np.cross(g["normal"], [1.0, 0, 0])
        if np.linalg.norm(e1) < 1e-6:
            e1 = np.cross(g["normal"], [0, 1.0, 0])
        e1 = e1 / np.linalg.norm(e1)
        e2, ax = np.cross(g["normal"], e1), g["normal"]
        r = np.linspace(0, g["R"], n // 2 + 1)[:, None] * np.ones_like(u)
        X = r * np.cos(u), r * np.sin(u), 0 * r
    c0 = g["centre"]
    out = [c0[k] + X[0] * e1[k] + X[1] * e2[k] + X[2] * ax[k] for k in range(3)]
    return out[0], out[1], out[2]


def draw(ax, shapes, colours=None):
    """Draw the shapes into a matplotlib 3D axes; returns {tag: colour}."""
    import matplotlib as mpl
    tags = []
    for sh in shapes:
        if sh.tag not in tags:
            tags.append(sh.tag)
    cyc = mpl.rcParams["axes.prop_cycle"].by_key()["color"]
    colours = colours or {t: cyc[i % len(cyc)] for i, t in enumerate(tags)}
    allpts = []
    for sh in shapes:
        col = colours[sh.tag]
        g = sh.geo
        if sh.kind == "line":
            ax.plot(*g["pts"].T, color=col, lw=1.0)
            allpts.append(g["pts"])
        elif sh.kind == "point":
            ax.scatter(*g["centre"], color=col, s=20)
            allpts.append(g["centre"][None, :])
        elif sh.kind == "ring":
            e1 = np.cross(g["normal"], [1.0, 0, 0])
            if np.linalg.norm(e1) < 1e-6:
                e1 = np.cross(g["normal"], [0, 1.0, 0])
            e1 = e1 / np.linalg.norm(e1)
            e2 = np.cross(g["normal"], e1)
            t = np.linspace(0, 2 * np.pi, 80)
            pts = g["centre"] + g["R"] * (np.outer(np.cos(t), e1) + np.outer(np.sin(t), e2))
            ax.plot(*pts.T, color=col, lw=1.5)
            allpts.append(pts)
        else:
            x, y, z = _surface(sh)
            alpha = 0.18 if g.get("hollow") else 0.45
            ax.plot_surface(x, y, z, color=col, alpha=alpha, linewidth=0, shade=True)
            allpts.append(np.column_stack([x.ravel(), y.ravel(), z.ravel()]))
    if allpts:
        P = np.vstack(allpts)
        c = 0.5 * (P.max(axis=0) + P.min(axis=0))
        h = 0.5 * float(np.max(P.max(axis=0) - P.min(axis=0))) or 1.0
        ax.set_xlim(c[0] - h, c[0] + h)
        ax.set_ylim(c[1] - h, c[1] + h)
        ax.set_zlim(c[2] - h, c[2] + h)
        try:
            ax.set_box_aspect((1, 1, 1))
        except Exception:  # noqa: BLE001
            pass
    return colours


def tag_counts(shapes):
    out = defaultdict(set)
    for sh in shapes:
        out[sh.tag].add(sh.unit)
    return {k: len(v) for k, v in out.items()}
