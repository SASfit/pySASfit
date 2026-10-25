"""Old name of :mod:`pysasfit.toolbox` (pySASbox was renamed pyTOOLbox).

``import pysasfit.sasbox.hdf5`` and the like keep working: every submodule here is the
module of :mod:`pysasfit.toolbox` itself. New code should import ``pysasfit.toolbox``.
"""
import warnings

warnings.warn("pysasfit.sasbox is now pysasfit.toolbox (pyTOOLbox)", DeprecationWarning, stacklevel=2)
