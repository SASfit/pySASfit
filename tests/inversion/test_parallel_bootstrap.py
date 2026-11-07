"""Bootstrap re-solves in worker processes give the same replicates as the serial loop; the faster EM step
gives the same iterates."""
import numpy as np

from pysasfit.inversion import uncertainty
from pysasfit.inversion.solvers import em


def _problem():
    q = np.linspace(0.05, 3, 60)
    r = np.linspace(0.5, 15, 30)
    x = q[:, None] * r[None, :]
    F = 3 * (np.sin(x) - x * np.cos(x)) / x ** 3
    A = F ** 2 * (r ** 6)[None, :] * (r[1] - r[0])
    b = A @ np.exp(-0.5 * ((r - 6) / 1.2) ** 2)
    db = 0.02 * b + 1e-3 * b.max()
    return A, b + np.random.default_rng(3).normal(0, db), db


def test_process_pool_equals_serial():
    A, b, db = _problem()
    solve = uncertainty.replicate_solver("bayesian_evidence", A, db)
    rng = np.random.default_rng(1)
    reps = [b + rng.normal(0, db) for _ in range(5)]
    one = uncertainty._One(solve)
    serial = uncertainty._run_replicates(one, solve, reps, workers=1)
    par = uncertainty._run_replicates(one, solve, reps, workers=2, min_seconds=0.0)
    assert all(np.array_equal(a, c) for a, c in zip(serial, par))
    r1 = uncertainty.bootstrap_uncertainty(solve, b, db, n_boot=4, rng=np.random.default_rng(2), workers=1)
    r2 = uncertainty.bootstrap_uncertainty(solve, b, db, n_boot=4, rng=np.random.default_rng(2), workers=2)
    assert np.array_equal(r1.x_samples, r2.x_samples)


def test_em_plain_iterates_unchanged():
    A, b, db = _problem()
    x = np.full(A.shape[1], 1e-6)
    ref = x
    for _ in range(20):
        ref = em.em_step(ref, A, b)
    res = em.solve(A, b, db, max_iterations=20, tol=0.0, accelerate=False)
    assert np.allclose(res.x, ref, rtol=1e-12)
    assert np.isclose(res.chi2_r_history[-1], float(np.mean(((A @ ref - b) / db) ** 2)), rtol=1e-12)
