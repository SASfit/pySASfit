"""Surfaces of the voxel structure (pysasfit.morph.surface): areas, orientation, clipping, STL."""

import sys

import numpy as np
import pytest

from pysasfit.morph import surface as S

N, A, R = 48, 2.0, 15.0


def _sphere():
    x = np.arange(N) - N / 2 + 0.5
    r = np.sqrt(x[:, None, None] ** 2 + x[None, :, None] ** 2 + x[None, None, :] ** 2)
    return r < R, (R - r).astype(np.float32)


CENTRE = np.full(3, (N / 2 - 0.5) * A)
AREA = 4 * np.pi * (R * A) ** 2


@pytest.mark.parametrize("source,tol", [("field", 0.01), ("smoothed", 0.03), ("voxels", 0.12)])
def test_sphere_area_and_orientation(source, tol):
    mat, psi = _sphere()
    m = S.phase_surface(mat, A, "pore", source, psi=psi, level=0.0)
    assert m.area == pytest.approx(AREA, rel=tol)
    assert np.mean(np.einsum("ij,ij->i", m.normals, m.vertices - CENTRE)) > 0          # outward normals
    v = m.vertices[m.faces]
    fn = np.cross(v[:, 1] - v[:, 0], v[:, 2] - v[:, 0])
    assert np.mean(np.einsum("ij,ij->i", fn, v.mean(axis=1) - CENTRE)) > 0           # outward winding


def test_solid_is_closed_at_the_box():
    mat, psi = _sphere()
    m = S.phase_surface(mat, A, "solid", "field", psi=psi, level=0.0)
    box = 6 * (N * A) ** 2
    assert m.area == pytest.approx(AREA + box, rel=0.02)


def test_clipping_closes_the_cut():
    mat, _ = _sphere()
    m = S.phase_surface(mat, A, "pore", "smoothed", clip=(2, N // 2 - 1))
    assert m.area == pytest.approx(AREA / 2 + np.pi * (R * A) ** 2, rel=0.03)


def test_largest_cluster_vertices():
    mat = np.zeros((30, 30, 30), bool)
    mat[3:10, 3:10, 3:10] = True
    mat[15:28, 15:28, 15:28] = True
    big = S.largest_cluster(mat)
    assert big.sum() == 13 ** 3
    m = S.phase_surface(mat, 1.0, "pore", "voxels")
    hit = S.vertices_in(m, big)
    assert np.all(m.vertices[hit].min(axis=0) > 13) and np.all(m.vertices[~hit].max(axis=0) < 11)


def test_stl(tmp_path):
    mat, psi = _sphere()
    m = S.phase_surface(mat, A, "pore", "field", psi=psi, level=0.0)
    p = S.write_stl(str(tmp_path / "s.stl"), m)
    raw = open(p, "rb").read()
    n = int(np.frombuffer(raw[80:84], np.uint32)[0])
    assert n == len(m) and len(raw) == 84 + 50 * n


def test_pyqtgraph_fallback(monkeypatch):
    pytest.importorskip("pyqtgraph")
    monkeypatch.setitem(sys.modules, "skimage.measure", None)        # import fails
    assert S.marching_cubes_backend() == "pyqtgraph"
    mat, psi = _sphere()
    m = S.phase_surface(mat, A, "pore", "field", psi=psi, level=0.0)
    assert m.area == pytest.approx(AREA, rel=0.01)
    assert np.mean(np.einsum("ij,ij->i", m.normals, m.vertices - CENTRE)) > 0


def test_empty_phase():
    assert S.phase_surface(np.zeros((8, 8, 8), bool), 1.0, "pore", "voxels") is None


def test_label_surface_of_a_shell():
    """Three phases: core | shell | outside; the shell surface has an inner and an outer sheet."""
    x = np.arange(N) - N / 2 + 0.5
    r = np.sqrt(x[:, None, None] ** 2 + x[None, :, None] ** 2 + x[None, None, :] ** 2)
    psi = (R - r).astype(np.float32)               # decreasing outward: core is psi > 5
    levels = np.array([0.0, 5.0])                  # outside < 0 < shell < 5 < core
    labels = np.digitize(psi, levels).astype(np.int8)
    shell = S.label_surface(labels, 1, A, "field", psi=psi, levels=levels)
    outer, inner = 4 * np.pi * (R * A) ** 2, 4 * np.pi * ((R - 5) * A) ** 2
    assert shell.area == pytest.approx(outer + inner, rel=0.01)
    core = S.label_surface(labels, 2, A, "smoothed")
    assert core.area == pytest.approx(inner, rel=0.04)
