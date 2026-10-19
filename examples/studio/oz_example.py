"""
(Copied unchanged from SASstudio; it will move to pysasfit.oz.)

Example: using the Ornstein-Zernike (OZ) solver classes from
SASfitGit/src/AndersonAcc4OZ/src/RPC/ directly inside SASfitStudio's own
Python environment.

Unlike the robertus_shs-style plugin models (see SASStudio_example_script.py,
which goes through ctypes + a compiled .dll/.so), this OZ tool is pure
Python/NumPy/SciPy -- no C plugin, no ctypes, no do_init()/sasfit_param
machinery needed at all. A plain import is enough.

Requirements (once, in whatever Python environment runs this):
    pip install numpy scipy sundials4py
sundials4py (https://pypi.org/project/sundials4py/) is SUNDIALS' own
official KINSOL Python bindings -- needs Python >=3.12. Only needed for
Sundials4pyKinsolOZsolver below; Picard/Anderson/scipy-based solvers
(picardOZsolver.py, andersonOZsolver.py, scipyAndersonOZsolver.py,
scipyNewtonKrylovOZsolver.py) do not need it.

Ported from the original Python 2 RPC-client/server design to Python 3,
usable directly (no RPC server/client round-trip needed) by passing
port=0 to any solver class's constructor.
"""
from os import path, getcwd
import sys
import numpy as np
import matplotlib.pyplot as pyplot

# SASfitGit and SASstudio are assumed to be sibling directories under the
# same switchdrive root -- adjust this path if that is not the case for
# your checkout.
OZ_RPC_PATH = path.join(path.dirname(path.abspath(__file__)), "..", "..", "..", "SASfitGit", "src", "AndersonAcc4OZ", "src", "RPC")
sys.path.insert(0, OZ_RPC_PATH)

from sundials4pyKinsolOZsolver import Sundials4pyKinsolOZsolver
from picardOZsolver import PicardOZsolver

# --- Example 1: hard spheres, Percus-Yevick closure, cross-checked
# against the analytic PY+HS solution this tool has built in ---------
solver = Sundials4pyKinsolOZsolver(port=0)
solver.setVolumeDensity(0.3)
solver.setPotentialByName("HardSphere")
solver.solve()
rdf_numeric = solver.getRDF().copy()

solver.calculateAnalyticalRDFsolutionForHS()
rdf_analytic = solver.getRDF().copy()

r = solver.getrArray()
hsp = solver.getHardSphereRadiusInPoints()

pyplot.figure()
pyplot.plot(r[hsp:hsp+3000], rdf_numeric[hsp:hsp+3000], label="sundials4py KINSOL")
pyplot.plot(r[hsp:hsp+3000], rdf_analytic[hsp:hsp+3000], "--", label="analytic PY+HS")
pyplot.xlabel("r"); pyplot.ylabel("g(r)")
pyplot.legend(); pyplot.title("Hard spheres, volume fraction 0.3")
pyplot.show()

# --- Example 2: any of the potentials available -- see
# oZfixpointOperator.py's setXXXPotential() methods for the full list
# and their arguments (or call solver.printPotentialSetterArguments()
# for a live-printed list including argument names) --------------------
solver2 = PicardOZsolver(port=0)
solver2.printPotentialSetterArguments()
solver2.setVolumeDensity(0.15)
solver2.setPotentialByName("StickyHardSphere", 0.2, 0.05)  # tau, delta
solver2.solve()

pyplot.figure()
pyplot.plot(r, solver2.getRDF())
pyplot.xlabel("r"); pyplot.ylabel("g(r)")
pyplot.title("Sticky hard spheres (Baxter tau=0.2)")
pyplot.show()
