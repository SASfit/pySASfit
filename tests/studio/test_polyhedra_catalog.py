# SPDX-License-Identifier: GPL-3.0-or-later
"""Tests of the catalogue solids of pysasfit.studio.polyhedra (Archimedean, Catalan, Johnson, Kepler-Poinsot, ...)."""

import json
import math
import pathlib

import numpy as np
import pytest

from pysasfit.studio.polyhedra import _catalog as C
from pysasfit.studio.polyhedra import specs

_REF = json.loads(pathlib.Path(__file__).with_name("polyhedra_catalog_reference.json").read_text())
_DIRS = np.array(_REF["directions"])
_QR = _REF["qR_list"]
ALL = [("archimedean", n) for n in range(1, 14)] + [("johnson", n) for n in range(1, 93)] + [("catalan", n) for n in range(1, 14)]
_SPEC = {s["name"]: s for s in specs.SPECS}


def _p(*values):
    p = np.zeros(50)
    p[: len(values)] = values
    return p


def _shape(fam, n, edge=1.3):
    return getattr(C, fam)(n, edge)


# ------------------------------------------------------------------------------------- C++ reference values
@pytest.mark.parametrize("key", sorted(_REF["shapes"]))
def test_form_factor_against_cpp(key):
    fam, n = key.split(":")
    sh = _shape(fam, int(n))
    r = _REF["shapes"][key]
    assert sh.volume == pytest.approx(r["volume"], rel=1e-11)
    q = np.array([x / sh.radius * d for x in _QR for d in _DIRS])
    ref = np.array(r["F_re"]) + 1j * np.array(r["F_im"])
    assert np.abs(sh.ff(q) - ref).max() < 1e-9 * sh.volume


# --------------------------------------------------------------------------------- geometry of the solids
@pytest.mark.parametrize("fam,n", ALL)
def test_volume_matches_convex_hull_and_body_is_convex(fam, n):
    from scipy.spatial import ConvexHull

    sh = _shape(fam, n)
    assert sh.is_convex(1e-9)
    assert sh.volume == pytest.approx(ConvexHull(sh.vertices).volume, rel=1e-11)
    assert np.abs(sh.center_of_mass()).max() < 1e-10 * sh.radius


@pytest.mark.parametrize("fam", ["archimedean", "johnson"])
def test_unit_edges_and_regular_faces(fam):
    for n in range(1, 14 if fam == "archimedean" else 93):
        sh = _shape(fam, n, 1.0)
        for f in sh.faces:
            lengths = [2 * np.linalg.norm(e.E) for e in f.edges]
            k = len(lengths) * (2 if f.sym_S2 else 1)
            assert np.allclose(lengths, 1.0, atol=1e-12)
            assert f.area == pytest.approx(k / (4 * math.tan(math.pi / k)), rel=1e-11)  # equilateral + regular area


def test_catalan_faces_are_tangent_to_one_sphere_and_shortest_edge_is_unit():
    for n in range(1, 14):
        sh = _shape("catalan", n, 1.0)
        d = [f.rperp for f in sh.faces]
        assert np.allclose(d, d[0], rtol=1e-11)
        edges = {tuple(sorted(p)) for t in sh.topology.faces for p in zip(t[0], t[0][1:] + t[0][:1])}
        lengths = [np.linalg.norm(sh.vertices[a] - sh.vertices[b]) for a, b in edges]
        assert min(lengths) == pytest.approx(1.0, rel=1e-12)


def test_known_solids():
    a = 2.0
    assert C.archimedean(1, a).volume == pytest.approx(23 * math.sqrt(2) / 12 * a**3)  # truncated tetrahedron
    assert C.archimedean(2, a).volume == pytest.approx(5 * math.sqrt(2) / 3 * a**3)  # cuboctahedron
    assert C.archimedean(3, a).volume == pytest.approx((21 + 14 * math.sqrt(2)) / 3 * a**3)  # truncated cube
    assert C.archimedean(5, a).volume == pytest.approx((12 + 10 * math.sqrt(2)) / 3 * a**3)  # rhombicuboctahedron
    assert C.johnson(1, a).volume == pytest.approx(math.sqrt(2) / 6 * a**3)  # square pyramid
    assert C.johnson(2, a).volume == pytest.approx((5 + math.sqrt(5)) / 24 * a**3)  # pentagonal pyramid


def test_rhombic_dodecahedron_volume():
    # rhombic dodecahedron of edge a: V = 16 sqrt(3) / 9 a^3
    assert C.catalan(2, 1.0).volume == pytest.approx(16 * math.sqrt(3) / 9)


def test_invalid_arguments_raise():
    with pytest.raises(ValueError):
        C.johnson(0, 1.0)
    with pytest.raises(ValueError):
        C.johnson(93, 1.0)
    with pytest.raises(ValueError):
        C.archimedean(3, -1.0)
    with pytest.raises(ValueError):
        C.prism_n(2, 1.0, 1.0)


@pytest.mark.parametrize("fam,n", ALL[::7])
def test_hermitian_and_limit(fam, n):
    sh = _shape(fam, n)
    q = np.random.default_rng(n).normal(size=(12, 3)) / sh.radius
    assert np.abs(sh.ff(-q) - np.conj(sh.ff(q))).max() < 1e-10 * sh.volume
    assert sh.ff(np.zeros((1, 3)))[0].real == pytest.approx(sh.volume, rel=1e-12)


def test_centrosymmetric_solids_have_real_amplitude():
    for fam, n in (("archimedean", 3), ("archimedean", 12), ("catalan", 6), ("johnson", 27), ("johnson", 72)):
        sh = _shape(fam, n)
        if sh.sym_ci:
            q = np.random.default_rng(1).normal(size=(10, 3)) / sh.radius
            assert np.abs(sh.ff(q).imag).max() < 1e-11 * sh.volume


# ----------------------------------------------------------------------------------------- infinite families
@pytest.mark.parametrize("n", [3, 4, 5, 6, 8])
def test_prism_antiprism_bipyramid_trapezohedron(n):
    from scipy.spatial import ConvexHull

    for sh in (C.antiprism_n(n, 1.2, 0.9), C.bipyramid_n(n, 1.2, 1.1), C.trapezohedron_n(n, 0.8, 1.3)):
        assert sh.is_convex(1e-9)
        assert sh.volume == pytest.approx(ConvexHull(sh.vertices).volume, rel=1e-11)
    pr = C.prism_n(n, 1.2, 2.0)
    assert pr.volume == pytest.approx(n / (4 * math.tan(math.pi / n)) * 1.2**2 * 2.0)
    bp = C.bipyramid_n(n, 1.2, 1.1)
    assert bp.volume == pytest.approx(2 / 3 * 1.1 * n / (4 * math.tan(math.pi / n)) * 1.2**2)


def test_uniform_antiprism_is_the_octahedron_in_volume():
    h = math.sqrt(2 / 3) * 1.2
    assert C.antiprism_n(3, 1.2, h).volume == pytest.approx(math.sqrt(2) / 3 * 1.2**3)


# ----------------------------------------------------------------------------------------- Kepler-Poinsot
@pytest.mark.parametrize("n", [1, 2, 3, 4])
def test_kepler_poinsot_closed_surface_and_unit_edges(n):
    V, faces = C._kp_unit(n)
    directed = {}
    for f in faces:
        for i in range(len(f)):
            e = (f[i], f[(i + 1) % len(f)])
            directed[e] = directed.get(e, 0) + 1
    assert all(c == 1 for c in directed.values())
    assert all((b, a) in directed for a, b in directed)  # every edge is traversed once in each direction
    assert np.allclose([np.linalg.norm(V[a] - V[b]) for a, b in directed], 1.0)
    assert len(V) - len(directed) // 2 + len(faces) == {1: -6, 2: -6, 3: 2, 4: 2}[n]  # Euler characteristic


def test_kepler_poinsot_values():
    # circumradii for edge 1
    assert C.kepler_poinsot(2, 1.0).radius == pytest.approx(math.sqrt((5 + math.sqrt(5)) / 8))
    assert C.kepler_poinsot(1, 1.0).radius == pytest.approx(math.sin(math.pi / 5))
    sh = C.kepler_poinsot(1, 1.0)
    q = np.random.default_rng(2).normal(size=(10, 3)) * 0.5
    assert np.abs(sh.ff(q).imag).max() < 1e-12  # centrosymmetric


# ------------------------------------------------------------------------------------------ registered specs
def test_catalogue_specs_registered():
    for name in ("arch_truncated_cube", "cat_rhombic_dodecahedron", "j86_sphenocorona", "kp_small_stellated_dodecahedron",
                 "prism_n", "antiprism_n", "bipyramid_n", "trapezohedron_n"):
        assert "ff_ba_" + name in _SPEC and "ff_ba_" + name + "_oriented" in _SPEC


def test_johnson_spec_average_and_oriented_amplitude():
    s = _SPEC["ff_ba_j86_sphenocorona"]
    p = _p(4.0, 1.5)  # edge, eta
    q = np.array([0.0, 0.05, 0.4])
    I = s["intensity"](q, p)
    assert I[0] == pytest.approx((1.5 * C.johnson(86, 4.0).volume) ** 2)
    so = _SPEC["ff_ba_j86_sphenocorona_oriented"]
    po = _p(4.0, 1.5, 10.0, 20.0, 30.0, 40.0)
    F = so["amplitude"](q, po)
    assert np.iscomplexobj(F) and np.allclose(so["intensity"](q, po), np.abs(F) ** 2)
