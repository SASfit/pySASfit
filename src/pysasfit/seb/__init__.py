"""pySEBpolymers -- form factors of structures built from linked sub-units.

Pure-Python (numpy/scipy) implementation of the formalism of Svaneborg &
Pedersen, J. Chem. Phys. 136, 104105 and 154907 (2012), as used by the
Scattering Equation Builder (SEB; Jarrett & Svaneborg, J. Appl. Cryst. 57,
587 (2024)).  Written from the published formalism and the textbook sub-unit
expressions (no SEB code); sub-unit, reference-point and parameter names
follow SEB/pyseb, so structures carry over.

>>> from pysasfit.seb import Structure
>>> s = Structure("micelle")
>>> s.add("SolidSphere", "core")
>>> for i in range(20):
...     s.link("GaussianPolymer", f"c{i}.end1", f"core.surface#a{i}", tag="chain")
>>> F = s.form_factor(q, {"beta_core": 3, "R_core": 25, "beta_chain": 0.4, "Rg_chain": 10})
"""
from .structure import Structure
from .subunits import REGISTRY, SubUnit, TableSubUnit, OrientedBody
from . import builders

__all__ = ["Structure", "REGISTRY", "SubUnit", "TableSubUnit", "OrientedBody", "builders"]
