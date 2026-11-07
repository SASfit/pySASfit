"""Geometry of the lattice sketch of pySCAtter (Qt-free): the crystal block with its particles in the laboratory
frame of the engine, the reciprocal lattice and the reflections that fall on the detector.

The frame is that of :mod:`engine`: x along the beam, y horizontal (detector q_h), z vertical (detector q_v).
Everything is taken from the engine itself (:func:`engine.make_lattice`, the zone-axis rotation of
``_oriented_lattice``, :func:`engine.director_lab`), so the sketch shows exactly the structure that is calculated.

* :func:`build_scene` -> :class:`Scene` (cell edges, particle centres and axes, particle shape, bicontinuous
  surface field, reciprocal lattice, Bragg condition);
* :func:`detector_reflections` -> reflections on the detector (oriented crystal) or powder rings, for the
  overlay on the 2D pattern;
* :func:`particle_mesh` / :func:`slab_mesh` -> triangle meshes (vertices, faces) for a 3D viewer.

Lengths in nm, q in 1/nm.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from . import engine
from .params import LTYPES, PARTICLES

__all__ = ["Scene", "build_scene", "detector_reflections", "particle_mesh", "slab_mesh", "bicontinuous_field",
           "BICONTINUOUS"]

#: nodal approximations of the bicontinuous phases (fractional coordinates u of the cubic cell, X = 2 pi u)
BICONTINUOUS = {
    "gyroid": ("gyroid G (Ia3d)",
               lambda X, Y, Z: np.sin(X) * np.cos(Y) + np.sin(Y) * np.cos(Z) + np.sin(Z) * np.cos(X)),
    "obdd": ("Schwarz D (Pn3m)",
             lambda X, Y, Z: (np.sin(X) * np.sin(Y) * np.sin(Z) + np.sin(X) * np.cos(Y) * np.cos(Z)
                              + np.cos(X) * np.sin(Y) * np.cos(Z) + np.cos(X) * np.cos(Y) * np.sin(Z))),
    "plumbers": ("Schwarz P (Im3m)", lambda X, Y, Z: np.cos(X) + np.cos(Y) + np.cos(Z)),
}


@dataclass
class Scene:
    """Everything a viewer needs; positions in nm (lab frame, block centred at 0), G in 1/nm."""
    kind: Optional[str]                    # lattice kind of Crystal (None: no lattice)
    title: str
    dim: int = 3
    cell: np.ndarray = field(default_factory=lambda: np.eye(3))     # columns a, b, c in the lab frame
    edges: np.ndarray = field(default_factory=lambda: np.zeros((0, 2, 3)))
    centres: np.ndarray = field(default_factory=lambda: np.zeros((0, 3)))
    axes: np.ndarray = field(default_factory=lambda: np.zeros((0, 3)))
    particle: Dict = field(default_factory=dict)          # {"type": ..., dimensions}
    slab: Optional[Dict] = None                           # lamellae: {"normal", "thickness", "extent", "spacing"}
    surface: Optional[Dict] = None                        # bicontinuous: {"name", "field", "origin", "step"}
    director: Optional[np.ndarray] = None
    oriented: bool = True                                 # False: powder (all crystal orientations)
    hkl: np.ndarray = field(default_factory=lambda: np.zeros((3, 0), int))
    G: np.ndarray = field(default_factory=lambda: np.zeros((3, 0)))
    weight: np.ndarray = field(default_factory=lambda: np.zeros(0))       # m |f|^2 / n per reflection
    intensity: np.ndarray = field(default_factory=lambda: np.zeros(0))    # weight * <P>(g) (relative)
    excitation: np.ndarray = field(default_factory=lambda: np.zeros(0))   # distance from the Ewald sphere (1/nm)
    on_detector: np.ndarray = field(default_factory=lambda: np.zeros(0, bool))
    det: np.ndarray = field(default_factory=lambda: np.zeros((2, 0)))       # (q_h, q_v) of the spot on the detector
    spread: np.ndarray = field(default_factory=lambda: np.zeros(0))         # azimuthal factor A at the detector
    rings: bool = False                   # reflections smeared over the whole sphere |q| = g (powder or EditAzi = 0)
    tolerance: float = 0.0
    k: float = np.inf                                     # Ewald sphere radius (inf: flat, 'preset' q map)
    qmax: float = 1.0
    recip: np.ndarray = field(default_factory=lambda: np.eye(3))    # columns a*, b*, c* (lab)
    notes: List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------------------------------------------
def _crystal(p):
    """(crystal as calculated, crystal in the zone-axis orientation, oriented?)."""
    c = engine.make_lattice(p)
    if c is None:
        return None, None, True
    ordis = int(p["Ordis"])
    oc = engine._oriented_lattice(p, c)
    return c, oc, ordis not in (7,)


def _block(oc, ncell):
    """Lattice translations of the block (fractional) and its fractional extent per axis."""
    n = [max(1, int(ncell))] * 3
    if oc.dim == 2:
        n[2] = 1
    elif oc.dim == 1:
        n = [max(2, int(ncell) + 1), 1, 1]
    rng = [np.arange(k + 1) if (d < oc.dim) else np.arange(1) for d, k in enumerate(n)]
    T = np.array(np.meshgrid(*rng, indexing="ij")).reshape(3, -1).T
    return T, np.array(n, float)


def _edges(cellL, n, dim):
    """Outline of every unit cell of the block (segments, fractional -> lab done by the caller)."""
    segs = []
    nx, ny, nz = (int(v) for v in n)
    for i in range(nx + 1):
        for j in range(ny + 1):
            segs.append(((i, j, 0), (i, j, nz)))
    for i in range(nx + 1):
        for k in range(nz + 1):
            segs.append(((i, 0, k), (i, ny, k)))
    for j in range(ny + 1):
        for k in range(nz + 1):
            segs.append(((0, j, k), (nx, j, k)))
    s = np.array(segs, float)
    return s @ cellL.T


def _particle_spec(p):
    """Shape of one particle for the sketch (body frame: symmetry axis = z)."""
    kind, inter = int(p["ComboBoxParticle"]), int(p["ComboBoxInterior"])
    R, L, Ri = float(p["EditRadius"]), float(p["Length"]), float(p["EditRadiusi"])
    name = PARTICLES[kind] if 0 <= kind < len(PARTICLES) else "particle"
    if kind == 0:
        if inter == 1 or inter == 2:
            return {"type": "sphere", "radius": R, "shell": max(Ri, R), "name": name + ", core + shell"}
        if inter in (3, 4):
            n = max(int(p.get("ShellN", 3)), 1)
            t = float(p.get("ShellThick", 2.0))
            out = R + n * t + (n * max(Ri, 0) if inter == 4 else 0)
            return {"type": "sphere", "radius": R, "shell": out, "name": name + ", " + ("multi-shell" if inter == 3
                                                                                      else "myelin")}
        return {"type": "sphere", "radius": R, "name": name}
    if kind == 1:
        shell = max(Ri, R) if inter in (1, 2) else None
        return {"type": "cylinder", "radius": R, "half_length": max(L, 1e-3), "shell": shell, "name": name}
    if kind == 2:                       # disk: EditRadius = half-thickness, Length = radius
        return {"type": "cylinder", "radius": max(L, 1e-3), "half_length": R, "name": name}
    if kind == 3:                       # vesicle: inner radius R, outer radius L
        return {"type": "sphere", "radius": R, "shell": max(L, R), "hollow": True, "name": name}
    if kind == 4:
        return {"type": "box", "half": (R, R, R), "name": name}
    if kind == 5:
        shell = max(Ri, R) if inter in (1, 2) else None
        return {"type": "ellipsoid", "semi": (R, R, max(L, 1e-3)), "shell": shell, "name": name}
    if kind == 6:
        return {"type": "ellipsoid", "semi": (R, Ri if Ri > 0 else R, max(L, 1e-3)), "name": name}
    if kind in (7, 8):
        e = float(p.get("Alpha", 2.0)) or 2.0
        semi = (R, R, max(L, 1e-3)) if kind == 7 else (R, R, R)
        return {"type": "superquadric", "semi": semi, "exponent": e, "name": name}
    return {"type": "none", "name": name}


def _ewald(p):
    if p.get("EditQmaxPreset", True):
        return np.inf
    lam = float(p.get("EditWavelength", 0.0) or 0.0)
    return 2 * np.pi / lam if lam > 0 else np.inf


def _qmax(p):
    if p.get("EditQmaxPreset", True):
        return float(p["EditQmax"])
    try:
        qx, qy, qz = engine.pixel_q(p)
        return float(max(np.abs(qy).max(), np.abs(qz).max()))
    except Exception:  # noqa: BLE001
        return float(p.get("EditQmax", 1.0))


def _excitation(G, k):
    """Distance of each reciprocal-lattice point from the Ewald sphere through the origin (beam along +x)."""
    if not np.isfinite(k):
        return G[0].copy()
    c = np.array([-k, 0.0, 0.0])[:, None]
    return np.linalg.norm(G - c, axis=0) - k


def build_scene(p, ncell=2, axes_mode="model", seed=3):
    """Scene of the parameters ``p``.

    ncell: unit cells per direction of the drawn block (in-plane for rods, layers for lamellae).
    axes_mode: 'model' (particle axes as the model assumes them: along the director for oriented distributions,
    random for 'isotropic') or 'lattice' (along the cell c axis, the rod axis, the layer normal)."""
    p = dict(p)
    kind_lt = int(p["LType"])
    lt_name = LTYPES[kind_lt] if 0 <= kind_lt < len(LTYPES) else "?"
    c, oc, oriented = _crystal(p)
    spec = _particle_spec(p)
    sc = Scene(kind=None if c is None else c.kind, title=lt_name, particle=spec, oriented=oriented)
    sc.k, sc.qmax = _ewald(p), _qmax(p)
    ordis = int(p["Ordis"])
    sc.director = engine.director_lab(p) if ordis not in (7, 13) else None
    rng = np.random.default_rng(seed)
    if c is None:                                         # single particle
        sc.title = spec.get("name", "particle")
        sc.centres = np.zeros((1, 3))
        sc.axes = np.array([sc.director if sc.director is not None else [0.0, 0.0, 1.0]])
        sc.notes.append("no lattice: one particle" + ("" if sc.director is not None else
                                                      " (isotropic: shown along the vertical)"))
        return sc
    sc.dim = oc.dim
    cellL = oc.Rot @ oc.M
    sc.recip = 2 * np.pi * np.linalg.inv(cellL).T
    T, n = _block(oc, ncell)
    clip = None
    if oc.dim == 2:                       # rods: the drawn prism is as high as the (clipped) rods
        a_len = float(np.linalg.norm(cellL[:, 0]))
        hl = spec.get("half_length", a_len) if spec["type"] == "cylinder" else spec.get("radius", a_len)
        clip = min(float(hl), max(1.0, ncell) * a_len * 0.75)
        cellL = cellL.copy()
        cellL[:, 2] = cellL[:, 2] / np.linalg.norm(cellL[:, 2]) * 2 * clip
    sc.cell = cellL
    centre = cellL @ (n / 2.0)
    sc.particle = dict(spec, clip=clip)
    sc.edges = _edges(cellL, n, oc.dim) - centre if oc.dim != 1 else np.zeros((0, 2, 3))
    if c.kind in BICONTINUOUS:
        name, f = BICONTINUOUS[c.kind]
        sc.surface = {"name": name, "n": n, "func": f}
        sc.notes.append(f"bicontinuous phase: nodal approximation of the {name} surface")
    elif oc.dim == 1:
        normal = oc.axis
        a = float(oc.cellpar[0])
        thick = 2 * (spec.get("half_length", a / 4) if spec["type"] == "cylinder" else a / 4)
        k = np.arange(int(n[0])) - (n[0] - 1) / 2
        sc.slab = {"normal": normal, "thickness": min(thick, 0.95 * a), "spacing": a,
                   "positions": k[:, None] * a * normal[None, :], "extent": 1.6 * a * max(2, ncell)}
        sc.notes.append("lamellae: layers of the disk thickness 2·EditRadius, spacing a")
    else:
        from .lattices import _MOTIF
        motif = np.array(_MOTIF.get(c.kind, [(0, 0, 0)]), float)
        frac = (T[:, None, :] + motif[None, :, :]).reshape(-1, 3)
        if oc.dim == 2:
            frac[:, 2] = 0.5                                  # rods centred in the prism
        frac = frac[np.all(frac <= n + 1e-9, axis=1)]
        frac = np.unique(np.round(frac, 9), axis=0)
        sc.centres = frac @ cellL.T - centre
        if oc.dim == 2:
            sc.notes.append("rods: the cell c axis is the rod direction")
    m = len(sc.centres)
    lattice_axis = cellL[:, 2] / np.linalg.norm(cellL[:, 2])
    if oc.dim == 2:
        lattice_axis = oc.axis
    if axes_mode == "lattice" or ordis == 13:
        ax = np.tile(lattice_axis, (m, 1))
    elif sc.director is not None:
        ax = np.tile(sc.director, (m, 1))
    else:
        ax = rng.normal(size=(m, 3))
        ax /= np.linalg.norm(ax, axis=1)[:, None]
        if spec["type"] not in ("sphere", "none"):
            sc.notes.append("isotropic orientation: particle axes drawn at random (decoupling approximation)")
    sc.axes = ax
    # ---- reciprocal lattice and Bragg condition
    sc.hkl, sc.G, sc.weight = oc.hkl, oc.G, oc.weights
    g = np.linalg.norm(sc.G, axis=0)
    try:
        delta, para = engine.peak_delta(p, c.kind)
    except Exception:  # noqa: BLE001
        delta, para = 0.02, 0.0
    width = delta + para * g * g
    sc.tolerance = float(np.median(width)) if width.size else delta
    sc.excitation = _excitation(sc.G, sc.k)
    azi = float(p.get("EditAzi", 0.0) or 0.0)
    cpp = engine._uses_cpp_lattice(p) and oriented
    if cpp:
        # CrystalScatter form: every reflection is a radial peak at |q| = g times an azimuthal factor A(psi) in the
        # angle between q and G (arc width 4/EditAzi).  On the detector (Ewald sphere) |q| = g lies on a circle;
        # the spot is where q comes closest to G: along the projection of G, at radius sqrt(g^2 - q_beam^2).
        qb = -g * g / (2 * sc.k) if np.isfinite(sc.k) else np.zeros_like(g)
        rho = np.sqrt(np.maximum(g * g - qb * qb, 0.0))
        pr = np.hypot(sc.G[1], sc.G[2])
        u = np.where(pr > 1e-12, sc.G[1:] / np.maximum(pr, 1e-300), 0.0)
        qs = np.vstack([qb, rho * u])
        cosp = np.clip((qs * sc.G).sum(0) / np.maximum(g * g, 1e-300), -1, 1)
        psi = np.arccos(cosp)
        if azi > 0:
            A = engine._cpp_azimuthal(int(p["ComboBoxPeak"]), psi ** 2, g * g, 4.0 / azi,
                                      float(p.get("EditPeakPar", 0) or 0))
        else:
            A = np.ones_like(g)
            sc.rings = True
            sc.notes.append("EditAzi = 0: no azimuthal dependence, every reflection is a ring")
        sc.det = rho * u
        sc.spread = np.asarray(A, float)
        inside = (np.abs(sc.det[0]) <= sc.qmax) & (np.abs(sc.det[1]) <= sc.qmax) & (pr > 1e-12)
        sc.on_detector = (sc.spread > 0.02) & inside
        n_off = int(np.count_nonzero(sc.on_detector & (np.abs(sc.excitation) > 1.5 * width)))
        if n_off and not sc.rings:
            sc.notes.append(f"{n_off} reflections reach the detector only through the azimuthal spread "
                            f"(EditAzi = {azi:g} nm: arc width {4 / azi:.3g} 1/nm); they appear at |q| = g")
    else:
        sc.det = sc.G[1:].copy()
        sc.spread = np.ones_like(g)
        inside = (np.abs(sc.G[1]) <= sc.qmax) & (np.abs(sc.G[2]) <= sc.qmax)
        sc.on_detector = (np.abs(sc.excitation) <= 1.5 * width) & inside
    try:
        shape = engine.make_particle(p)
        ug, inv = np.unique(np.round(g, 9), return_inverse=True)
        Pg = np.asarray(shape.P(np.maximum(ug, 1e-9)), float)[inv]
    except Exception:  # noqa: BLE001
        Pg = np.ones_like(g)
    sc.intensity = sc.weight * np.maximum(Pg, 0) * engine.debye_waller(p, g)
    if not oriented:
        sc.rings = True
        sc.notes.append("isotropic: powder of crystallites (rings); one crystallite is drawn in the zone-axis "
                        "orientation")
    elif oc.dim < 3:
        sc.notes.append("rods / lamellae are not rotated by the zone axis (as in the engine)")
    return sc


def detector_reflections(sc, max_labels=40):
    """Reflections seen on the detector: list of dicts {hkl, qh, qv, q, I} for an oriented crystal, or powder
    rings {hkl, q, I} (one per distinct length of G inside q max, strongest hkl as label) for 'isotropic'."""
    if sc.G.shape[1] == 0:
        return []
    g = np.linalg.norm(sc.G, axis=0)
    out = []
    if sc.rings:
        ug, inv = np.unique(np.round(g, 6), return_inverse=True)
        qlim = sc.qmax * np.sqrt(2)
        for i, q in enumerate(ug):
            if q > qlim:
                continue
            sel = np.flatnonzero(inv == i)
            j = sel[np.argmax(sc.intensity[sel])]
            h = tuple(int(abs(v)) for v in sc.hkl[:, j])
            out.append({"hkl": tuple(sorted(h, reverse=True)), "q": float(q),
                        "I": float(sc.intensity[sel].sum()), "m": int(len(sel))})
        return out
    for j in np.flatnonzero(sc.on_detector):
        out.append({"hkl": tuple(int(v) for v in sc.hkl[:, j]), "qh": float(sc.det[0, j]), "qv": float(sc.det[1, j]),
                    "q": float(g[j]), "I": float(sc.intensity[j] * sc.spread[j])})
    out.sort(key=lambda d: -d["I"])
    return out


# ---------------------------------------------------------------------------------------------------------------
# meshes (vertices (n, 3), faces (m, 3)); body frame: symmetry axis z
# ---------------------------------------------------------------------------------------------------------------
def _sphere_unit(rows=14, cols=24, exponent=None):
    th = np.linspace(0, np.pi, rows + 1)
    ph = np.linspace(0, 2 * np.pi, cols, endpoint=False)
    T, Pp = np.meshgrid(th, ph, indexing="ij")
    if exponent:                        # superquadric |x|^(2e)+|y|^(2e)+|z|^(2e) = 1 (e=1 sphere, e->inf cube)
        e = 1.0 / float(exponent)
        sg = lambda x: np.sign(x) * np.abs(x) ** e  # noqa: E731
        x, y, z = sg(np.sin(T)) * sg(np.cos(Pp)), sg(np.sin(T)) * sg(np.sin(Pp)), sg(np.cos(T))
    else:
        x, y, z = np.sin(T) * np.cos(Pp), np.sin(T) * np.sin(Pp), np.cos(T)
    V = np.stack([x, y, z], -1).reshape(-1, 3)
    F = []
    for i in range(rows):
        for j in range(cols):
            a, b = i * cols + j, i * cols + (j + 1) % cols
            c_, d = (i + 1) * cols + j, (i + 1) * cols + (j + 1) % cols
            F.append((a, c_, b))
            F.append((b, c_, d))
    return V, np.array(F)


def _cylinder_unit(cols=28):
    ph = np.linspace(0, 2 * np.pi, cols, endpoint=False)
    ring = np.stack([np.cos(ph), np.sin(ph), np.zeros(cols)], 1)
    V = np.vstack([ring + [0, 0, -1], ring + [0, 0, 1], [[0, 0, -1], [0, 0, 1]]])
    F = []
    for j in range(cols):
        a, b = j, (j + 1) % cols
        F += [(a, b, cols + a), (b, cols + b, cols + a), (2 * cols, b, a), (2 * cols + 1, cols + a, cols + b)]
    return V, np.array(F)


def _box_unit():
    V = np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)], float)
    F = np.array([(0, 1, 3), (0, 3, 2), (4, 6, 7), (4, 7, 5), (0, 4, 5), (0, 5, 1), (2, 3, 7), (2, 7, 6),
                  (0, 2, 6), (0, 6, 4), (1, 5, 7), (1, 7, 3)])
    return V, F


def particle_mesh(spec, layer="core"):
    """Unit mesh of the particle in its body frame (axis z); layer 'core' or 'shell' (None if absent)."""
    t = spec.get("type")
    if layer == "shell":
        s = spec.get("shell")
        if not s:
            return None
        if t == "sphere":
            V, F = _sphere_unit()
            return V * s, F
        if t == "cylinder":
            V, F = _cylinder_unit()
            return V * [s, s, spec["half_length"]], F
        if t == "ellipsoid":
            V, F = _sphere_unit()
            a, b, c = spec["semi"]
            f = s / a
            return V * [a * f, b * f, c + (s - a)], F
        return None
    if t == "sphere":
        V, F = _sphere_unit()
        return V * spec["radius"], F
    if t == "cylinder":
        V, F = _cylinder_unit()
        return V * [spec["radius"], spec["radius"], spec["half_length"]], F
    if t == "ellipsoid":
        V, F = _sphere_unit()
        return V * np.asarray(spec["semi"]), F
    if t == "superquadric":
        V, F = _sphere_unit(18, 32, exponent=spec["exponent"])
        return V * np.asarray(spec["semi"]), F
    if t == "box":
        V, F = _box_unit()
        return V * np.asarray(spec["half"]), F
    return None


def _frame(axis):
    n = np.asarray(axis, float)
    n = n / np.linalg.norm(n)
    t = np.array([1.0, 0, 0]) if abs(n[0]) < 0.9 else np.array([0.0, 1, 0])
    e1 = np.cross(t, n)
    e1 /= np.linalg.norm(e1)
    return np.stack([e1, np.cross(n, e1), n], 1)


def place(mesh, centres, axes, clip_half=None):
    """One mesh with a copy of the unit mesh at every centre, its z axis along ``axes``. ``clip_half`` caps the
    body-frame z extent (long rods in a finite block)."""
    V0, F0 = mesh
    if clip_half is not None:
        V0 = V0.copy()
        V0[:, 2] = np.clip(V0[:, 2], -clip_half, clip_half)
    Vs, Fs = [], []
    for i, (c, a) in enumerate(zip(centres, axes)):
        Vs.append(V0 @ _frame(a).T + c)
        Fs.append(F0 + i * len(V0))
    if not Vs:
        return np.zeros((0, 3)), np.zeros((0, 3), int)
    return np.vstack(Vs), np.vstack(Fs)


def slab_mesh(slab):
    """Boxes of the lamellae (one per layer)."""
    n = np.asarray(slab["normal"], float)
    R = _frame(n)
    h = slab["extent"] / 2
    V0, F0 = _box_unit()
    V0 = V0 * [h, h, slab["thickness"] / 2]
    Vs, Fs = [], []
    for i, c in enumerate(slab["positions"]):
        Vs.append(V0 @ R.T + c)
        Fs.append(F0 + i * len(V0))
    return np.vstack(Vs), np.vstack(Fs)


def bicontinuous_field(sc, pts_per_cell=20):
    """(field on a grid of the block in fractional coordinates, transform index -> lab) for the iso-surface."""
    n = np.asarray(sc.surface["n"], int)
    shape = tuple(int(k * pts_per_cell) + 1 for k in n)
    u = [np.linspace(0, k, s) for k, s in zip(n, shape)]
    U = np.meshgrid(*u, indexing="ij")
    Fv = sc.surface["func"](*(2 * np.pi * x for x in U))
    centre = sc.cell @ (n / 2.0)

    def to_lab(idx):
        frac = np.asarray(idx, float) / pts_per_cell
        return frac @ sc.cell.T - centre
    return Fv, to_lab
