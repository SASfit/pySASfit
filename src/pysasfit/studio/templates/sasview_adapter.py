"""
Auto-generated adapter that makes the SASfit plugin function NAME_STR usable in SasView
(created by `pysasfit-studio sasview-adapter`). Needs pysasfit with the SASfit libraries
installed in the Python environment SasView runs in.

NB: technically, according to standard design patterns, this is a 'facade' or 'proxy',
but in the interest of user-friendly language it is called an 'adapter'.
"""

import numpy as np

from pysasfit.studio import function

name = "NAME_STR"
title = """TITLE_STR"""
description = """DESCRIPTION_STR"""
category = "CATEGORY_STR"
structure_factor = STRUCTURE_FACTOR_BOOL
single = False

#      ["name", "units", default, [lower, upper], "type", "description"],
parameters = [
    PARAMS
]

_f = None


def Iq(q, *param_values):
    global _f
    if _f is None:
        _f = function(name)
    return _f(np.asarray(q, dtype=float), *param_values)


Iq.vectorized = True
