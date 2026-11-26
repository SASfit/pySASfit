# pysasfit.studio.polyhedra

Hard-polyhedron form factors (14 shapes of BornAgain / libformfactor) for pySASstudio.

**License: GNU General Public License v3 or later.** This package is a Python/numpy port of
[libformfactor](https://jugit.fz-juelich.de/mlz/libformfactor) (Copyright Forschungszentrum Juelich GmbH 2022,
Scientific Computing Group at MLZ; author Joachim Wuttke) and therefore a derivative work under the GPL. The rest of
pySASfit is BSD-3-Clause; the host code only loads this package if it is present (`ports/__init__.py`), so it can
be removed or left out of a distribution without affecting the rest. Put the GPL-3.0 text next to this file as
`LICENSE-GPL-3.0.txt` (it is the `LICENSE` file of libformfactor).

Citation: J. Wuttke, *Numerically stable form factor of any polygon and polyhedron*, J. Appl. Cryst. 54, 580-587
(2021), doi:10.1107/S1600576721001710; G. Pospelov et al., J. Appl. Cryst. 53, 262-276 (2020) (BornAgain).

Contents

* `_core.py`   Edge, Face, Polyhedron, Prism, Box (vectorised over q, real or complex)
* `_shapes.py` factories of the 14 shapes (`FACTORIES`), centred at the centre of mass
* `_orient.py` oriented complex amplitude, isotropic averages over orientations
* `_catalog.py` further standard polyhedra with the same method (below); `_catalog_data.py`, `_seeds1-3.py`: seeds
* `specs.py`   registration as `ff_ba_<shape>` (random orientation) and `ff_ba_<shape>_oriented` (complex amplitude)

Validated against a C++ build of libformfactor 0.5.2: see `tests/studio/test_polyhedra.py`.

## Further standard polyhedra (`_catalog.py`)

Registered as `ff_ba_<family>` and `ff_ba_<family>_oriented`; the first parameter `n` selects the solid (the
numbering is listed in the parameter documentation and in `ARCHIMEDEAN_NAMES`, `CATALAN_NAMES`, `JOHNSON_NAMES`,
`KEPLER_POINSOT_NAMES`).

| family | n | size parameters |
|---|---|---|
| `archimedean` | 1..13 | edge `a` |
| `catalan` (duals) | 1..13 | shortest edge `a` |
| `johnson` | 1..92 | edge `a` |
| `kepler_poinsot` | 1..4 | edge `a` of the star faces |
| `prism_n`, `antiprism_n`, `bipyramid_n` | number of base edges (>= 3) | base edge `a`, height `H` |
| `trapezohedron_n` | n >= 3 | ring radius `R`, apex distance `H` |

Geometry: the exact unit-edge coordinates of the Archimedean and Johnson solids are not tabulated. A rough seed
(1e-3, rounded from the vertex lists of G. W. Hart's Virtual Polyhedra, whose author permits non-commercial use of
the mathematical descriptions) is refined by Newton iteration to machine precision with all faces regular and planar,
and checked (Euler characteristic, convexity). The Catalan solids are the polar duals of the Archimedean solids.
Inversion (Ci) and two-fold face (S2) symmetries are detected automatically and halve the work.

Kepler-Poinsot solids have self-intersecting faces. The method gives the amplitude of the winding-number density
(density 2 at the centre of a pentagram, ...); volumes are density-weighted. A uniform-density envelope is not
implemented.

Validation (`tests/studio/test_polyhedra_catalog.py`): all 118 solids agree with libformfactor (C++, topology
without symmetry flags) to better than 2e-12 V; volume vs. convex hull, unit edges, regular faces, Hermitian
symmetry.
