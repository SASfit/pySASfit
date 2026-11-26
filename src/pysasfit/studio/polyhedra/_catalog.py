# SPDX-License-Identifier: GPL-3.0-or-later
"""Further standard polyhedra, evaluated with the face/edge method of J. Wuttke (libformfactor, see _core.py).

Families (all return objects with ``ff(q)``, ``volume``, ``radius``, ``surface_area()``; q has shape (N, 3)):

* ``archimedean(n, edge)``  the 13 Archimedean solids (n = 1..13, see ``ARCHIMEDEAN_NAMES``), edge length ``edge``
* ``catalan(n, edge)``      their 13 duals, the Catalan solids; ``edge`` is the length of the shortest edge
* ``johnson(n, edge)``      the 92 Johnson solids (n = 1..92, ``JOHNSON_NAMES``)
* ``prism_n``, ``antiprism_n``, ``bipyramid_n``, ``trapezohedron_n``  the infinite families
* ``kepler_poinsot(n, edge)``  the four regular star polyhedra (n = 1..4)

Geometry of the Archimedean and Johnson solids is not tabulated: a rough seed (_catalog_data) is refined by Newton
iteration until all edges have unit length and all faces are exactly regular and planar (to machine precision), and
the result is verified (Euler characteristic, convexity). All bodies are centred at their centre of mass.
"""

from __future__ import annotations

import functools
import math

import numpy as np

from ._catalog_data import ARCHIMEDEAN_NAMES, JOHNSON_NAMES, SEEDS
from ._core import Face, Polyhedron, Prism, Topology

__all__ = ["ARCHIMEDEAN_NAMES", "JOHNSON_NAMES", "CATALAN_NAMES", "KEPLER_POINSOT_NAMES", "archimedean", "catalan",
           "johnson", "prism_n", "antiprism_n", "bipyramid_n", "trapezohedron_n", "kepler_poinsot",
           "CATALOG_FACTORIES"]  # fmt: skip

CATALAN_NAMES = ["Triakis tetrahedron", "Rhombic dodecahedron", "Triakis octahedron", "Tetrakis hexahedron",
                 "Deltoidal icositetrahedron", "Disdyakis dodecahedron", "Pentagonal icositetrahedron",
                 "Rhombic triacontahedron", "Triakis icosahedron", "Pentakis dodecahedron",
                 "Deltoidal hexecontahedron", "Disdyakis triacontahedron", "Pentagonal hexecontahedron"]  # fmt: skip
KEPLER_POINSOT_NAMES = ["Small stellated dodecahedron", "Great dodecahedron", "Great stellated dodecahedron",
                        "Great icosahedron"]  # fmt: skip


# ----------------------------------------------------------------------------------------------- mesh helpers
def _newell(P):
    n = np.zeros(3)
    for i in range(len(P)):
        n += np.cross(P[i], P[(i + 1) % len(P)])
    return n


def _orient_convex(V, faces, inside):
    """Reverse every face chain whose normal points towards ``inside`` (convex bodies only)."""
    out = []
    for f in faces:
        P = V[list(f)]
        if _newell(P) @ (P.mean(0) - inside) < 0:
            f = tuple(reversed(f))
        out.append(tuple(f))
    return out


def _signed_volume_and_com(V, faces):
    vol, com = 0.0, np.zeros(3)
    for f in faces:
        P = V[list(f)]
        for k in range(1, len(f) - 1):
            v = np.linalg.det(np.array([P[0], P[k], P[k + 1]])) / 6
            vol += v
            com += v * (P[0] + P[k] + P[k + 1]) / 4
    return vol, com / vol


def _hull_faces(V, tol):
    from scipy.spatial import ConvexHull

    h = ConvexHull(V)
    groups = []
    for i, e in enumerate(h.equations):
        for g in groups:
            if np.abs(g[0][:3] - e[:3]).max() < 5 * tol and abs(g[0][3] - e[3]) < tol:
                g[1].append(i)
                break
        else:
            groups.append([e, [i]])
    faces = []
    for e, ids in groups:
        n = e[:3] / np.linalg.norm(e[:3])
        vs = sorted({int(v) for i in ids for v in h.simplices[i]})
        P = V[vs]
        c = P.mean(0)
        u = P[0] - c
        u -= n * (u @ n)
        u /= np.linalg.norm(u)
        w = np.cross(n, u)
        order = np.argsort(np.arctan2((P - c) @ w, (P - c) @ u))
        faces.append(tuple(vs[j] for j in order))
    return faces


def _pairs(faces):
    """Targets of the regular-polygon constraints: all chords of every face for unit edge."""
    pairs = {}
    for f in faces:
        n = len(f)
        for i in range(n):
            for k in range(1, n // 2 + 1):
                a, b = f[i], f[(i + k) % n]
                pairs[(min(a, b), max(a, b))] = math.sin(k * math.pi / n) / math.sin(math.pi / n)
    return pairs


def _polish(V0, faces):
    from scipy.optimize import least_squares

    nv = len(V0)
    pairs = _pairs(faces)
    E = sorted(pairs)
    tgt = np.array([pairs[e] for e in E])
    ea = np.array([e[0] for e in E])
    eb = np.array([e[1] for e in E])
    pa = np.array([(f[0], f[1], f[2], f[k]) for f in faces for k in range(3, len(f))], int).reshape(-1, 4)
    ne = len(E)

    def res(x):
        V = x.reshape(nv, 3)
        r = [np.linalg.norm(V[ea] - V[eb], axis=1) - tgt]
        if len(pa):
            a, b, c, d = (V[pa[:, i]] for i in range(4))
            r.append(np.einsum("ij,ij->i", np.cross(b - a, c - a), d - a))
        return np.concatenate(r)

    def jac(x):
        V = x.reshape(nv, 3)
        J = np.zeros((ne + len(pa), 3 * nv))
        d = V[ea] - V[eb]
        d /= np.linalg.norm(d, axis=1, keepdims=True)
        for k, (a, b) in enumerate(zip(ea, eb)):
            J[k, 3 * a:3 * a + 3] = d[k]
            J[k, 3 * b:3 * b + 3] = -d[k]
        for k, (ia, ib, ic, id_) in enumerate(pa):
            a, b, c, dd = V[ia], V[ib], V[ic], V[id_]
            n = np.cross(b - a, c - a)
            J[ne + k, 3 * id_:3 * id_ + 3] = n
            J[ne + k, 3 * ib:3 * ib + 3] = np.cross(c - a, dd - a)
            J[ne + k, 3 * ic:3 * ic + 3] = np.cross(dd - a, b - a)
            J[ne + k, 3 * ia:3 * ia + 3] = -(n + np.cross(c - a, dd - a) + np.cross(dd - a, b - a))
        return J

    sol = least_squares(res, V0.ravel(), jac=jac, method="trf", xtol=1e-15, ftol=1e-15, gtol=1e-15, max_nfev=60)
    return sol.x.reshape(nv, 3), float(np.abs(res(sol.x)).max())


@functools.lru_cache(maxsize=None)
def _unit_solid(key):
    """Exact unit-edge vertices and outward oriented faces of Archimedean (A1..) / Johnson (J1..) solid."""
    V0 = np.array(SEEDS[key].split(), float).reshape(-1, 3) / 1000.0
    faces = _hull_faces(V0, 6e-3)
    V, r = _polish(V0, faces)
    if r > 1e-13:
        raise RuntimeError(f"geometry of solid {key} did not converge (residual {r:.1e})")
    faces = _orient_convex(V, faces, V.mean(0))
    if len(V) - len(_pairs_edges(faces)) + len(faces) != 2:
        raise RuntimeError(f"solid {key}: Euler characteristic violated")
    return V, faces


def _pairs_edges(faces):
    E = set()
    for f in faces:
        for i in range(len(f)):
            a, b = f[i], f[(i + 1) % len(f)]
            E.add((min(a, b), max(a, b)))
    return E


# -------------------------------------------------------------------------------- symmetry-aware topology
def _prototype(V, faces, convex=True):
    """Centre at the centre of mass; detect inversion symmetry (Ci) and two-fold face symmetry (S2).

    Returns (vertices, Topology). Faces must be outward oriented chains."""
    V = np.asarray(V, float)
    if convex:
        faces = _orient_convex(V, faces, V.mean(0))
    vol, com = _signed_volume_and_com(V, faces)
    V = V - com
    scale = np.abs(V).max()
    # inversion symmetry?
    sigma = []
    for v in V:
        d = np.linalg.norm(V + v, axis=1)
        j = int(np.argmin(d))
        if d[j] > 1e-9 * scale:
            sigma = None
            break
        sigma.append(j)
    ci_faces = None
    if sigma is not None and sorted(sigma) == list(range(len(V))):
        sets = {frozenset(f): k for k, f in enumerate(faces)}
        partner, ok = {}, True
        for k, f in enumerate(faces):
            g = sets.get(frozenset(sigma[v] for v in f))
            if g is None or g == k:
                ok = False
                break
            partner[k] = g
        if ok:
            V = (V - V[sigma]) / 2  # make the antipodal vertices exact mirror images
            reps, imgs, seen = [], [], set()
            for k in range(len(faces)):
                if k not in seen:
                    reps.append(k)
                    imgs.append(partner[k])
                    seen.update((k, partner[k]))
            ci_faces = [faces[k] for k in reps] + [faces[k] for k in reversed(imgs)]

    def build(fl, ci):
        out = []
        for f in fl:
            s2 = False
            if len(f) % 2 == 0:
                try:
                    Face(V[list(f)], True)
                    s2 = True
                except ValueError:
                    s2 = False
            out.append((tuple(f), s2))
        return Topology(out, ci)

    if ci_faces is not None:
        try:
            top = build(ci_faces, True)
            Polyhedron(top, V)
            return V, top
        except ValueError:
            pass  # not exactly Ci after all
    return V, build(faces, False)


class _Proto:
    """Unit-size prototype that is rescaled for each evaluation."""

    def __init__(self, V, faces, convex=True):
        self.V, self.top = _prototype(V, faces, convex)

    def __call__(self, scale):
        return Polyhedron(self.top, self.V * scale)


@functools.lru_cache(maxsize=None)
def _proto(key):
    V, faces = _unit_solid(key)
    return _Proto(V, faces)


def _check_edge(edge, who):
    if not edge > 0:
        raise ValueError(f"{who}: invalid parameter: edge<=0")


def _index(n, count, who):
    n = int(round(n))
    if not 1 <= n <= count:
        raise ValueError(f"{who}: index must be 1..{count}, got {n}")
    return n


def archimedean(n, edge):
    n = _index(n, 13, "Archimedean")
    _check_edge(edge, "Archimedean")
    return _proto(f"A{n}")(edge)


def johnson(n, edge):
    n = _index(n, 92, "Johnson")
    _check_edge(edge, "Johnson")
    return _proto(f"J{n}")(edge)


# ------------------------------------------------------------------------------------------- Catalan solids
@functools.lru_cache(maxsize=None)
def _catalan_proto(n):
    V, faces = _unit_solid(f"A{n}")
    V = V - _signed_volume_and_com(V, faces)[1]
    poles = []
    for f in faces:
        P = V[list(f)]
        nrm = _newell(P)
        nrm /= np.linalg.norm(nrm)
        poles.append(nrm / (nrm @ P.mean(0)))
    poles = np.array(poles)
    inc = [[] for _ in V]
    for fi, f in enumerate(faces):
        for v in f:
            inc[v].append(fi)
    dfaces = []
    for v, lst in enumerate(inc):
        c = V[v] / np.linalg.norm(V[v])
        P = poles[lst]
        u = P[0] - P.mean(0)
        u -= c * (u @ c)
        u /= np.linalg.norm(u)
        w = np.cross(c, u)
        order = np.argsort(np.arctan2((P - P.mean(0)) @ w, (P - P.mean(0)) @ u))
        dfaces.append(tuple(lst[j] for j in order))
    short = min(np.linalg.norm(poles[a] - poles[b]) for a, b in _pairs_edges(dfaces))
    return _Proto(poles / short, dfaces)


def catalan(n, edge):
    n = _index(n, 13, "Catalan")
    _check_edge(edge, "Catalan")
    return _catalan_proto(n)(edge)


# --------------------------------------------------------------------------------------- infinite families
def _regular_ngon(n, radius, phase=0.0):
    a = phase + 2 * math.pi * np.arange(n) / n
    return np.column_stack([radius * np.cos(a), radius * np.sin(a), np.zeros(n)])


def _nint(n, who, nmin=3, nmax=1000):
    n = int(round(n))
    if not nmin <= n <= nmax:
        raise ValueError(f"{who}: n must be between {nmin} and {nmax}")
    return n


def prism_n(n, edge, height):
    """Right prism with a regular n-gon base of edge length ``edge``."""
    n = _nint(n, "Prism_n")
    _check_edge(edge, "Prism_n")
    if not height > 0:
        raise ValueError("Prism_n: invalid parameter: height<=0")
    R = edge / (2 * math.sin(math.pi / n))
    return Prism(n % 2 == 0, height, _regular_ngon(n, R))


@functools.lru_cache(maxsize=None)
def _antiprism_proto(n, ratio):
    R = 1 / (2 * math.sin(math.pi / n))
    h = ratio
    V = np.vstack([_regular_ngon(n, R) + [0, 0, -h / 2], _regular_ngon(n, R, math.pi / n) + [0, 0, h / 2]])
    faces = [tuple(range(n - 1, -1, -1)), tuple(range(n, 2 * n))]
    for k in range(n):
        k1 = (k + 1) % n
        faces.append((k, k1, n + k))
        faces.append((k1, n + k1, n + k))
    return _Proto(V, faces)


def antiprism_n(n, edge, height):
    """Antiprism with regular n-gon bases of edge length ``edge`` (height free; uniform for the usual value)."""
    n = _nint(n, "Antiprism_n")
    _check_edge(edge, "Antiprism_n")
    if not height > 0:
        raise ValueError("Antiprism_n: invalid parameter: height<=0")
    P = _antiprism_proto(n, 1.0)
    V = P.V.copy()
    V[:, 2] *= height / edge  # prototype height 1 (units of edge) -> requested height
    return Polyhedron(P.top, V * edge)


@functools.lru_cache(maxsize=None)
def _bipyramid_proto(n):
    R = 1 / (2 * math.sin(math.pi / n))
    V = np.vstack([_regular_ngon(n, R), [[0, 0, 1.0], [0, 0, -1.0]]])
    faces = []
    for k in range(n):
        k1 = (k + 1) % n
        faces.append((k, k1, n))
        faces.append((k1, k, n + 1))
    return V, faces


def bipyramid_n(n, edge, height):
    """Bipyramid over a regular n-gon of edge ``edge``; ``height`` is the distance of each apex from the base plane."""
    n = _nint(n, "Bipyramid_n")
    _check_edge(edge, "Bipyramid_n")
    if not height > 0:
        raise ValueError("Bipyramid_n: invalid parameter: height<=0")
    V, faces = _bipyramid_proto(n)
    V = V * [edge, edge, height]
    return Polyhedron(*_prototype(V, faces)[::-1])


def trapezohedron_n(n, radius, height):
    """n-gonal trapezohedron (2n kite faces): apexes at z = +-height, zig-zag ring of radius ``radius``."""
    n = _nint(n, "Trapezohedron_n")
    if not (radius > 0 and height > 0):
        raise ValueError("Trapezohedron_n: radius and height must be positive")
    c = math.cos(math.pi / n)
    z0 = height * (1 - c) / (1 + c)  # planarity of the kites
    top = _regular_ngon(n, radius) + [0, 0, z0]
    bot = _regular_ngon(n, radius, math.pi / n) + [0, 0, -z0]
    V = np.vstack([top, bot, [[0, 0, height], [0, 0, -height]]])
    faces = []
    for k in range(n):
        k1 = (k + 1) % n
        faces.append((2 * n, k, n + k, k1))  # kites around the apexes; orientation is fixed in _prototype
        faces.append((2 * n + 1, n + (k - 1) % n, k, n + k))
    V, top_ = _prototype(V, faces)
    return Polyhedron(top_, V)


# -------------------------------------------------------------------------------------- Kepler-Poinsot
_PHI = (1 + math.sqrt(5)) / 2


def _icosa_vertices():
    V = []
    for s1 in (1, -1):
        for s2 in (1, -1):
            V += [(0, s1, s2 * _PHI), (s1, s2 * _PHI, 0), (s2 * _PHI, 0, s1)]
    V = np.array(V, float)
    return V / np.linalg.norm(V, axis=1, keepdims=True)


def _ring(V, centre_dir, idx, star):
    """vertices idx ordered ccw about centre_dir (star: pentagram order)."""
    c = centre_dir / np.linalg.norm(centre_dir)
    P = V[idx]
    u = P[0] - c * (P[0] @ c)
    u /= np.linalg.norm(u)
    w = np.cross(c, u)
    order = np.argsort(np.arctan2(P @ w, P @ u))
    idx = [idx[j] for j in order]
    return tuple(idx[(2 * k) % 5] for k in range(5)) if star else tuple(idx)


@functools.lru_cache(maxsize=None)
def _kp_unit(n):
    """Vertices and faces of Kepler-Poinsot solid n with unit edge (edge of the face polygons)."""
    ico = _icosa_vertices()
    if n in (1, 2):  # small stellated dodecahedron, great dodecahedron: ring of the 5 neighbours of a vertex
        d = np.linalg.norm(ico[:, None] - ico[None], axis=2)
        near = np.sort(d[0])[1]
        V = ico
        faces = []
        for i in range(12):
            nb = [j for j in range(12) if abs(d[i, j] - near) < 1e-9]
            faces.append(_ring(V, V[i], nb, star=(n == 1)))
    elif n == 4:  # great icosahedron: 20 triangles of mutually second-neighbouring vertices
        d = np.linalg.norm(ico[:, None] - ico[None], axis=2)
        far = np.sort(d[0])[6]
        V = ico
        faces = []
        for a in range(12):
            for b in range(a + 1, 12):
                for c in range(b + 1, 12):
                    if all(abs(d[x, y] - far) < 1e-9 for x, y in ((a, b), (b, c), (a, c))):
                        ctr = V[[a, b, c]].mean(0)
                        t = (a, b, c) if np.cross(V[b] - V[a], V[c] - V[a]) @ ctr > 0 else (a, c, b)
                        faces.append(t)
    else:  # great stellated dodecahedron: pentagram on the second layer of the dodecahedron about each axis
        d = np.linalg.norm(ico[:, None] - ico[None], axis=2)
        near = np.sort(d[0])[1]
        tri = [(a, b, c) for a in range(12) for b in range(a + 1, 12) for c in range(b + 1, 12)
               if all(abs(d[x, y] - near) < 1e-9 for x, y in ((a, b), (b, c), (a, c)))]  # fmt: skip
        V = np.array([ico[list(t)].mean(0) for t in tri])
        V /= np.linalg.norm(V, axis=1, keepdims=True)
        faces = []
        for i in range(12):
            h = V @ ico[i]
            levels = sorted(set(np.round(h, 9)), reverse=True)
            layer = [j for j in range(20) if abs(h[j] - levels[1]) < 1e-8]
            faces.append(_ring(V, ico[i], layer, star=True))
    first = faces[0]
    edge = np.linalg.norm(V[first[0]] - V[first[1]])
    return V / edge, faces


def kepler_poinsot(n, edge):
    """Kepler-Poinsot polyhedra: 1 small stellated dodecahedron, 2 great dodecahedron, 3 great stellated
    dodecahedron, 4 great icosahedron; ``edge`` is the edge length of the (star) faces.

    The faces self-intersect, so the body is described by its winding-number density: the scattering density is
    the number of times the surface winds around a point (e.g. 2 in the centre of a pentagram), which is what the
    face/edge formula yields without any change. The volume is accordingly the density-weighted volume."""
    n = _index(n, 4, "Kepler-Poinsot")
    _check_edge(edge, "Kepler-Poinsot")
    V, faces = _kp_unit(n)
    Vc, top = _prototype(V, faces, convex=False)
    return Polyhedron(top, Vc * edge)


# ---------------------------------------------------------------------------------------------------------
CATALOG_FACTORIES = {
    "Archimedean": archimedean, "Catalan": catalan, "Johnson": johnson,
    "Prism_n": prism_n, "Antiprism_n": antiprism_n, "Bipyramid_n": bipyramid_n, "Trapezohedron_n": trapezohedron_n,
    "KeplerPoinsot": kepler_poinsot,
}  # fmt: skip
