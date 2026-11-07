"""pySCAtter: small-angle scattering patterns of particles, lattices and orientation distributions with the
hypergeometric-function algorithms of Wagener & Förster (Sci. Rep. 13, 780 (2023)), plus the models and fits added
for pySASfit (worm-like chain, power-law shells, structure factors, 1D / 2D data fits).  Formerly the stand-alone
package *hypscat* (pySCAtter folder).  Lengths in nm, q in 1/nm.  GUI: :mod:`pysasfit.scatter.gui`."""
from .blocks import Block, BLOCK, z_from_sigma
from .shapes import set_default_weighting
from .shapes import Sphere, Ellipsoid, TriaxialEllipsoid, Cylinder, Disk, Parallelepiped, Cube
from .shapes_ext import LensDumbbell, Lens, Dumbbell, Superball, Octahedron, SuperEllipsoid
from .coreshell import CoreShellSphere, CoreShellEllipsoid, CoreShellCylinder, gradient_shell_sphere, multilayer_sphere, myelin_sphere
from .chain import ExcludedVolumeChain
from .lattice import Lattice, cubic, rotation
from .lattices import Crystal, make_lattice, zone_axis_rotation, peak
from . import orient, detector, gisas, oriented
from .models_ext import (KratkyPorodChain, powerlaw_shell_sphere, TeubnerStrey, PercusYevick, StickyHardSphere,
                         FractalTeixeira)
__version__ = "0.4.1"
