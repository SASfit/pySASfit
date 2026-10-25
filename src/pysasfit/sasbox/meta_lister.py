"""Old name of :mod:`pysasfit.toolbox.meta_lister`."""
import sys

from ..toolbox import meta_lister as _module

sys.modules[__name__] = _module
