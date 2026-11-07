"""Shared accelerators: batched small solves, parallel map."""
import numpy as np

from pysasfit.core import accel


def test_batched_solve_matches_lapack():
    rng = np.random.default_rng(0)
    for p in (1, 2, 3, 6, 15):
        M = np.eye(p) + 0.3 * rng.normal(size=(500, p, p))
        B = rng.normal(size=(500, p, p))
        assert np.allclose(accel.batched_solve(M, B), np.linalg.solve(M, B), rtol=1e-10, atol=1e-12)
        v = rng.normal(size=(500, p))
        assert np.allclose(accel.batched_solve(M, v), np.linalg.solve(M, v[..., None])[..., 0], rtol=1e-10,
                           atol=1e-12)


def test_parallel_map_keeps_order_and_exceptions():
    def f(x):
        if x == 3:
            raise ValueError("three")
        return x * x
    out = accel.parallel_map(f, range(6), workers=3)
    assert out[:3] == [0, 1, 4] and isinstance(out[3], ValueError) and out[4:] == [16, 25]
