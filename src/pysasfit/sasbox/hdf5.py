"""Old name of :mod:`pysasfit.toolbox.hdf5`."""
import sys

from ..toolbox import hdf5 as _module

sys.modules[__name__] = _module
