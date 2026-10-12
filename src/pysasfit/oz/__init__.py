"""pysasfit.oz -- Ornstein-Zernike toolkit (from pyOZgui).

Closures, iterative OZ solvers (Picard, Anderson, MDIIS, Biggs-Andrews, scipy,
SUNDIALS KINSOL), polydisperse / multicomponent structure factors and small-angle
scattering models, analytic references (MSA, RMSA, Robertus adhesive spheres,
Gazzillo), fitting, and a Qt GUI (``pysasfit-oz``, or the "Ornstein-Zernike" tab
of ``pysasfit-gui``).

The modules keep their pyOZgui names, e.g.::

    from pysasfit.oz import ozLib
    res = ozLib.solve(potential="HardSphere", phi=0.3, closure="Percus-Yevick")

Optional dependencies: ``pip install pysasfit[oz]``. The compiled helper
libraries (librmsa, liboneyukawa, liborpa, librfa) are looked for next to the
wrapper modules in this folder; their C sources are in ``extern/oz`` of the
repository.
"""

__version__ = "0.1.0"
