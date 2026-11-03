"""Morphological statistics of a binary two-phase voxel structure.

* specific surface S/V from Porod's law (data) and from the voxel structure
  (face count with the stereological factor 2/3 for isotropic interfaces on a cubic grid)
* connectivity: clusters of each phase (6-connectivity), the largest-cluster share,
  percolation along x, y, z, bicontinuity as in SAXSMorph (a cluster of *both*
  phases touches all six faces of the box) and SAXSMorph's "degree of connectivity"
  (largest pore + largest solid cluster) / all voxels.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage

__all__ = ["specific_surface_porod", "specific_surface_voxels", "phase_connectivity", "morphology_statistics"]


def specific_surface_porod(K, Q, porosity):
    """S/V = pi phi (1-phi) K / Q  (K: Porod constant, Q: invariant; units 1/length)."""
    return np.pi * porosity * (1 - porosity) * K / Q


def specific_surface_voxels(mat, a, periodic=True):
    """Interface area per volume from the voxel faces between unlike neighbours, times 2/3."""
    mat = np.asarray(mat, bool)
    faces = 0
    for ax in range(3):
        if periodic:
            faces += int(np.count_nonzero(mat != np.roll(mat, 1, axis=ax)))
        else:
            faces += int(np.count_nonzero(np.diff(mat.view(np.int8), axis=ax)))
    return (2.0 / 3.0) * faces * a * a / (mat.size * a ** 3)


def phase_connectivity(mask):
    """Cluster analysis of one phase (True voxels)."""
    lab, nlab = ndimage.label(mask)
    n = mask.size
    if nlab == 0:
        return {"clusters": 0, "largest_fraction": 0.0, "percolates": (False, False, False),
                "percolating_fraction": 0.0, "touches_all_faces": False}
    sizes = np.bincount(lab.ravel())[1:]
    perc = []
    perc_labels = set()
    all_faces = None
    for ax in range(3):
        lo = set(np.unique(np.take(lab, 0, axis=ax))) - {0}
        hi = set(np.unique(np.take(lab, -1, axis=ax))) - {0}
        both = lo & hi
        perc.append(bool(both))
        perc_labels |= both
        all_faces = both if all_faces is None else all_faces & both
    perc_vol = sum(int(sizes[l - 1]) for l in perc_labels)
    return {"clusters": int(nlab), "largest_fraction": float(sizes.max() / mask.sum()),
            "largest_voxels": int(sizes.max()), "percolates": tuple(perc),
            "percolating_fraction": float(perc_vol / mask.sum()),
            "touches_all_faces": bool(all_faces), "volume_fraction": float(mask.sum() / n)}


def morphology_statistics(mat, a, periodic=True):
    """Statistics of a binary matrix (True = phase 2, 'pore')."""
    pore = phase_connectivity(mat)
    solid = phase_connectivity(~mat)
    return {
        "porosity_voxels": float(mat.mean()),
        "S_over_V_voxels": specific_surface_voxels(mat, a, periodic),
        "pore": pore,
        "solid": solid,
        "bicontinuous": bool(pore["touches_all_faces"] and solid["touches_all_faces"]),
        "degree_of_connectivity": float((pore.get("largest_voxels", 0) + solid.get("largest_voxels", 0)) / mat.size),
    }
