# SPDX-License-Identifier: GPL-3.0-or-later
"""Hard polyhedra form factors (port of libformfactor)."""
from ._core import Box, Face, Polyhedron, Prism, Topology
from ._orient import oriented_amplitude, orientational_average, rotation_zyz, sphere_rule
from ._shapes import FACTORIES
from ._catalog import (ARCHIMEDEAN_NAMES, CATALAN_NAMES, JOHNSON_NAMES, KEPLER_POINSOT_NAMES, antiprism_n, archimedean,
                       bipyramid_n, catalan, johnson, kepler_poinsot, prism_n, trapezohedron_n)
