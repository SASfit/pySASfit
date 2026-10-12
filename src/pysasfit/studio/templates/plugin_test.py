"""Tests for the plugin PLUGIN_NAME (run with pytest)."""

import numpy as np

from pysasfit import studio

f = studio.function("ff_PLUGIN_NAME")


def test_forward_scattering():
    assert np.isclose(f(1e-6, I0=2.5, Rg=10.0), 2.5)


def test_decreasing():
    q = np.geomspace(1e-3, 1, 50)
    assert np.all(np.diff(f(q)) <= 0)
