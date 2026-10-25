"""Old name of :mod:`pysasfit.toolbox.gui`."""
import sys

from ..toolbox import gui as _module

sys.modules[__name__] = _module
