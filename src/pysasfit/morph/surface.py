"""Interface surfaces of the reconstructed structure as triangle meshes (no Qt, no VTK).

The surface of a phase is an iso-surface (marching cubes) of

* the Gaussian field psi at the cut level (``source="field"``): the exact, smooth interface of the
  level-cut model, available when the field was kept (``reconstruct(..., keep_field=True)``);
* the binary voxel matrix, smoothed with a Gaussian of ``sigma`` voxels (``"smoothed"``) or not
  (``"voxels"``, sigma = 0).

The volume is padded by one layer of the other phase, so the surface is closed at the box faces
and at a clipping plane. Voxel i has its centre at i * a (as in the VTK export).

Marching cubes from scikit-image when installed, else pyqtgraph's ``isosurface``.
:func:`write_stl` writes a binary STL file.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage

__all__ = ["Mesh", "phase_surface", "label_surface", "largest_cluster", "vertices_in", "write_stl",
           "marching_cubes_backend"]

SOURCES = ("field", "smoothed", "voxels")


@dataclass
class Mesh:
    vertices: np.ndarray        # (n, 3) float32, length units
    faces: np.ndarray           # (m, 3) int32
    normals: np.ndarray         # (n, 3) float32, unit, pointing out of the phase

    @property
    def area(self):
        v = self.vertices[self.faces].astype(float)
        return float(0.5 * np.linalg.norm(np.cross(v[:, 1] - v[:, 0], v[:, 2] - v[:, 0]), axis=1).sum())

    def __len__(self):
        return len(self.faces)


def marching_cubes_backend():
    try:
        from skimage.measure import marching_cubes  # noqa: F401
        return "scikit-image"
    except Exception:  # noqa: BLE001
        pass
    try:
        from pyqtgraph.functions import isosurface  # noqa: F401
        return "pyqtgraph"
    except Exception:  # noqa: BLE001
        return None


def _vertex_normals(v, f):
    n = np.zeros_like(v, dtype=np.float64)
    fn = np.cross(v[f[:, 1]] - v[f[:, 0]], v[f[:, 2]] - v[f[:, 0]])
    for i in range(3):
        np.add.at(n, f[:, i], fn)
    n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-30)
    return n


def _iso(vol, level, step=1):
    """Vertices in voxel index coordinates, faces, normals pointing towards vol > level."""
    try:
        from skimage.measure import marching_cubes
        v, f, n, _ = marching_cubes(vol, level=level, step_size=step, allow_degenerate=False)
        return v, f.astype(np.int32), -n           # skimage normals point down the gradient
    except ImportError:
        pass
    try:
        from pyqtgraph.functions import isosurface
    except Exception as e:  # noqa: BLE001
        raise ImportError("surfaces need scikit-image or pyqtgraph (pip install scikit-image)") from e
    v, f = isosurface(np.ascontiguousarray(vol[::step, ::step, ::step]), level)
    v = v * step
    f = f.astype(np.int32)
    n = _vertex_normals(v, f)
    # orient the normals towards increasing vol (into the phase)
    gx, gy, gz = np.gradient(vol.astype(np.float32))
    idx = np.clip(np.rint(v).astype(int), 0, np.array(vol.shape) - 1)
    grad = np.stack([gx[tuple(idx.T)], gy[tuple(idx.T)], gz[tuple(idx.T)]], axis=1)
    if np.sum(np.einsum("ij,ij->i", n, grad)) < 0:
        n = -n
    return v, f, n


def phase_surface(mat, a=1.0, phase="pore", source="smoothed", psi=None, level=None, sigma=0.8,
                  clip=None, mask=None, step=None):
    """Closed surface of one phase.

    mat     bool (n, n, n), True = pore
    phase   'pore' or 'solid'
    source  'field' (needs psi and level), 'smoothed' (Gaussian, ``sigma`` voxels) or 'voxels'
    clip    None or (axis, k): keep only the voxels with index <= k along axis
    mask    optional bool matrix restricting the phase further (e.g. the largest cluster)
    step    marching-cubes step (default 1 up to 160 voxels per edge, then 2)
    Returns a :class:`Mesh` with outward normals (pointing out of the phase), or None if empty.
    """
    mat = np.asarray(mat, bool)
    pore = phase == "pore"
    if source == "field" and psi is not None and level is not None and mask is None:
        vol = (np.asarray(psi, np.float32) - level) * (1 if pore else -1)
        iso, outside = 0.0, float(vol.min()) - 1.0
    else:
        m = mat if pore else ~mat
        if mask is not None:
            m = m & mask
        vol = m.astype(np.float32)
        if source != "voxels" and sigma > 0:
            vol = ndimage.gaussian_filter(vol, sigma, mode="nearest")
        iso, outside = 0.5, 0.0
    if clip is not None:
        ax, k = clip
        sl = [slice(None)] * 3
        sl[ax] = slice(0, int(k) + 1)
        vol = vol[tuple(sl)]
    if vol.size == 0 or vol.max() <= iso:
        return None
    vol = np.pad(vol, 1, constant_values=outside)
    if step is None:
        step = 1 if max(mat.shape) <= 160 else 2
    v, f, n = _iso(vol, iso, step)
    if len(f) == 0:
        return None
    v = (v - 1.0) * a                               # remove the padding
    n = -n                                          # outward = away from the phase
    # counter-clockwise seen from outside (STL convention): face normals along the vertex normals
    fn = np.cross(v[f[:, 1]] - v[f[:, 0]], v[f[:, 2]] - v[f[:, 0]])
    if np.sum(np.einsum("ij,ij->i", fn, n[f].sum(axis=1))) < 0:
        f = f[:, [0, 2, 1]]
    return Mesh(v.astype(np.float32), np.ascontiguousarray(f, np.int32), n.astype(np.float32))


def largest_cluster(mask):
    lab, n = ndimage.label(mask)
    if n == 0:
        return np.zeros_like(mask, bool)
    sizes = np.bincount(lab.ravel())
    sizes[0] = 0
    return lab == sizes.argmax()


def vertices_in(mesh, region, a=1.0):
    """True for vertices with a voxel of ``region`` among the 8 voxels around them."""
    p = mesh.vertices / a
    lo = np.floor(p).astype(int)
    hit = np.zeros(len(p), bool)
    shape = np.array(region.shape)
    for dx in (0, 1):
        for dy in (0, 1):
            for dz in (0, 1):
                idx = lo + (dx, dy, dz)
                ok = np.all((idx >= 0) & (idx < shape), axis=1)
                hit[ok] |= region[tuple(idx[ok].T)]
    return hit


def write_stl(path, mesh):
    """Binary STL."""
    v = mesh.vertices[mesh.faces].astype(np.float32)
    fn = np.cross(v[:, 1] - v[:, 0], v[:, 2] - v[:, 0])
    fn /= np.maximum(np.linalg.norm(fn, axis=1, keepdims=True), 1e-30)
    rec = np.zeros(len(v), dtype=[("n", "<f4", 3), ("v", "<f4", (3, 3)), ("attr", "<u2")])
    rec["n"], rec["v"] = fn, v
    with open(path, "wb") as fh:
        fh.write(b"SASmorph surface".ljust(80, b" "))
        fh.write(np.uint32(len(v)).tobytes())
        fh.write(rec.tobytes())
    return path


def label_surface(labels, i, a=1.0, source="smoothed", psi=None, levels=None, sigma=0.8, clip=None, mask=None,
                  step=None):
    """Closed surface of phase ``i`` of a multi-phase label matrix (phases ordered along psi).

    With ``source='field'`` the surface is the iso-surface of min(psi - level_{i-1}, level_i - psi) = 0,
    i.e. the exact boundaries of the slab level_{i-1} < psi < level_i of the Gaussian field.
    """
    labels = np.asarray(labels)
    if source == "field" and psi is not None and levels is not None and mask is None:
        levels = np.asarray(levels, float)
        psi = np.asarray(psi, np.float32)
        lo = -np.inf if i == 0 else levels[i - 1]
        hi = np.inf if i >= levels.size else levels[i]
        vol = np.minimum(psi - lo if np.isfinite(lo) else np.inf, hi - psi if np.isfinite(hi) else np.inf)
        vol = np.asarray(vol, np.float32)
        return _mesh(vol, 0.0, float(np.nanmin(vol)) - 1.0, a, clip, step, labels.shape)
    m = labels == i
    if mask is not None:
        m = m & mask
    vol = m.astype(np.float32)
    if source != "voxels" and sigma > 0:
        vol = ndimage.gaussian_filter(vol, sigma, mode="nearest")
    return _mesh(vol, 0.5, 0.0, a, clip, step, labels.shape)


def _mesh(vol, iso, outside, a, clip, step, shape):
    if clip is not None:
        ax, k = clip
        sl = [slice(None)] * 3
        sl[ax] = slice(0, int(k) + 1)
        vol = vol[tuple(sl)]
    if vol.size == 0 or vol.max() <= iso:
        return None
    vol = np.pad(vol, 1, constant_values=outside)
    if step is None:
        step = 1 if max(shape) <= 160 else 2
    v, f, n = _iso(vol, iso, step)
    if len(f) == 0:
        return None
    v = (v - 1.0) * a
    n = -n
    fn = np.cross(v[f[:, 1]] - v[f[:, 0]], v[f[:, 2]] - v[f[:, 0]])
    if np.sum(np.einsum("ij,ij->i", fn, n[f].sum(axis=1))) < 0:
        f = f[:, [0, 2, 1]]
    return Mesh(v.astype(np.float32), np.ascontiguousarray(f, np.int32), n.astype(np.float32))
