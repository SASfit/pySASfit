"""Compiled accelerated EM / MaxEnt-EM loops (Numba) agree with the NumPy implementation."""
import numpy as np
import pytest

from pysasfit.inversion.solvers import _compiled, em, maxent

pytestmark = pytest.mark.skipif(not _compiled.HAVE_NUMBA, reason="numba not installed")


def _problem():
    q = np.linspace(0.05, 3, 120)
    r = np.linspace(0.5, 15, 70)
    x = q[:, None] * r[None, :]
    F = 3 * (np.sin(x) - x * np.cos(x)) / x ** 3
    A = F ** 2 * (r ** 6)[None, :] * (r[1] - r[0])
    xt = np.exp(-0.5 * ((r - 6) / 1.2) ** 2)
    b = A @ xt
    db = 0.02 * b + 1e-3 * b.max()
    return A, b + np.random.default_rng(3).normal(0, db), db, xt


@pytest.mark.parametrize("case", ["smooth", "double", "me_const", "me_adapt"])
def test_compiled_equals_numpy(case, monkeypatch):
    A, b, db, xt = _problem()
    run = {"smooth": lambda: em.solve(A, b, db, smoothing_h=0.1),
           "double": lambda: em.solve(A, b, db, smoothing_h=0.05, double_smoothing=True),
           "me_const": lambda: maxent.solve_constant_prior(A, b, db, prior=np.full(70, xt.mean()), lam=0.05),
           "me_adapt": lambda: maxent.solve_adaptive_prior(A, b, db, lam=0.05)}[case]
    a = run()
    monkeypatch.setattr(_compiled, "HAVE_NUMBA", False)
    c = run()
    # converging cases only: without convergence the accelerated iteration wanders, and rounding-level
    # differences grow over thousands of steps
    assert c.converged and a.n_iterations == c.n_iterations and a.converged == c.converged
    assert np.allclose(a.x, c.x, rtol=1e-9, atol=1e-12 * np.abs(c.x).max())
    assert np.allclose(a.chi2_r_history, c.chi2_r_history, rtol=1e-9)
    assert np.allclose(a.diagnostics["g_norm_history"], c.diagnostics["g_norm_history"], rtol=1e-6,
                       atol=1e-15)
    if case.startswith("me"):
        assert np.allclose(a.diagnostics["entropy_history"], c.diagnostics["entropy_history"], rtol=1e-9,
                           atol=1e-12)
