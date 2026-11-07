import os
import sys

sys.path.insert(0, os.path.dirname(__file__))      # helper modules of these tests

import pytest
import pysasfit.scatter.shapes as _sh

@pytest.fixture(autouse=True)
def _volume_weighting(request):
    """legacy tests compare against V^2/V weighted brute force; reference tests opt out."""
    if "reference" in request.node.keywords:
        yield
        return
    old = _sh.DEFAULT_WEIGHTING
    _sh.set_default_weighting("volume")
    yield
    _sh.DEFAULT_WEIGHTING = old
