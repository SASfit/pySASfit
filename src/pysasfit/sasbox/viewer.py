"""Old name of :mod:`pysasfit.toolbox.viewer`."""
import sys

from ..toolbox import viewer as _module

sys.modules[__name__] = _module
