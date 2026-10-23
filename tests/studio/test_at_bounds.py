"""The 'at a bound' flag of a fit result is relative to the size of the parameter (no absolute tolerance)."""

import numpy as np

try:
    from .support import Q, make_model
except ImportError:
    from support import Q, make_model

from pysasfit.studio.fitting import FitProblem


def small_amplitude_problem(true_A, start_A, scale=1e-12):
    """I = A exp(-k q) with an amplitude of the order of 1e-12, like the amplitude N of a size distribution."""
    k = 0.8
    I = true_A * scale * np.exp(-k * Q)
    rng = np.random.default_rng(5)
    s = 0.01 * np.abs(I) + 1e-3 * scale
    p = FitProblem()
    p.add_dataset((Q, I + rng.normal(0, s), s), make_model(start_A * scale, 1.0))
    p.add_parameter("0.ff.A", lower=0.0)
    p.add_parameter("0.ff.k", lower=0.0, upper=5.0)
    return p


def test_a_tiny_parameter_in_the_middle_of_its_range_is_not_at_a_bound():
    r = small_amplitude_problem(true_A=3.0, start_A=2.0).fit("scipy")
    assert abs(r.values["ff.A"] / 1e-12 - 3.0) < 0.3
    assert r.at_bounds == []


def test_a_tiny_parameter_pushed_to_zero_is_at_the_bound():
    r = small_amplitude_problem(true_A=-1.0, start_A=2.0).fit("scipy")  # the data want a negative amplitude
    assert "ff.A" in r.at_bounds
