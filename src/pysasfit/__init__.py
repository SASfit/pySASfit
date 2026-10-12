"""pySASfit -- Python tools for small-angle scattering from the SASfit project.

Sub-packages
------------
core        shared infrastructure: data loading, instrumental resolution
            (Rician/Gaussian pinhole, slit), background models
sldprofile  direct Bayesian estimation of radial scattering-length-density
            profiles (Hansen 2016)
inversion   Fredholm inversion: size distributions and p(r) (EM, maxent,
            Tikhonov, Bayesian evidence, ...)
studio      the SASfit plugin functions (form factors, structure factors,
            size distributions) via ctypes, models and fitting (from SASstudio)
oz          Ornstein-Zernike solvers, closures, polydisperse structure
            factors and SAS models, with a Qt GUI (from pyOZgui)
gui         common Qt launcher (``pysasfit-gui``)

Sub-packages are imported lazily, so ``import pysasfit`` stays cheap and the
GUI dependencies are only needed for the GUIs.
"""

import importlib

__version__ = "0.2.0"

_SUBPACKAGES = ("core", "sldprofile", "inversion", "studio", "oz", "gui")

__all__ = list(_SUBPACKAGES)


def __getattr__(name):
    if name in _SUBPACKAGES:
        return importlib.import_module(f"{__name__}.{name}")
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
