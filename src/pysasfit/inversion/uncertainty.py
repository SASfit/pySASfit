"""
Parametric bootstrap for pointwise uncertainty on a recovered distribution
x(r) (or N(s)), applicable to ANY solver in solver_registry.py -- unlike
bayesian_evidence.posterior_covariance (a Laplace/Hessian approximation
specific to the Bayesian-evidence solver family), this makes no
linearity/positivity assumptions about the solver at all, so it also
covers the EM/MaxEnt/ARLS/SVD solvers, at the cost of actually re-solving
many times instead of a single cheap matrix inverse.

Method: perturb the FITTED intensity (A @ x of the single solve; the
measured b only if no fit is given) by the stated per-point uncertainty
db (assuming independent Gaussian noise
-- the same assumption chi2_r itself already makes throughout this
package), re-run the FULL solve (including its own internal lambda/h
search) on each noisy replicate, and summarize the resulting ensemble of
recovered x(r) curves by their pointwise mean and standard deviation.

Re-running the complete solve (not just the final linear step at a fixed
hyperparameter) on every replicate is deliberate: it propagates
uncertainty in the regularization-strength CHOICE itself, not just the
uncertainty of a linear solve conditioned on one particular lambda/h --
more honest, at the cost of this being exactly as slow as N full solves
(including whatever lambda search each solver does internally).

A consequence worth knowing before reading the output: the bootstrap MEAN
of a nonlinear, positivity-constrained solver (EM, MaxEnt, ARLS) is not
generally equal to a single solve on the unperturbed data -- the ensemble
average can be smoother/biased relative to the single "best" solve,
particularly wherever a replicate's noise pushes x toward (and gets
clipped at) its positivity floor. That's not a bug in this module; it's
the real behavior of the underlying nonlinear solver under their own
positivity constraint, and precisely the kind of thing a bootstrap
reveals and a single point estimate hides.
"""
from __future__ import annotations

import os

from dataclasses import dataclass
from typing import Callable

import numpy as np

from .solvers.base import SolverResult


@dataclass
class BootstrapResult:
    x_mean: np.ndarray       # (n,) pointwise mean over successful replicates
    x_std: np.ndarray        # (n,) pointwise std over successful replicates
    x_samples: np.ndarray    # (n_ok, n) every successful replicate's x
    n_boot: int               # successful replicates actually used
    n_failed: int              # replicates skipped (solver raised, or produced non-finite x)


def bootstrap_uncertainty(
    solve_fn: Callable[[np.ndarray], SolverResult],
    b: np.ndarray,
    db: np.ndarray,
    n_boot: int = 100,
    rng: np.random.Generator | None = None,
    center: np.ndarray | None = None,
    workers: int | None = None,
    progress: Callable[[int, int], None] | None = None,
    should_abort: Callable[[], bool] | None = None,
) -> BootstrapResult:
    """
    Parameters
    ----------
    solve_fn : callable(b_replicate) -> SolverResult
        Everything else (kernel matrix A, dI used for weighting/chi2, the
        chosen solver and any of its own fixed kwargs like taper_frac)
        should already be bound in via e.g. functools.partial or a small
        lambda -- see gui/app.py's usage for the exact pattern.
    b : (M,) the real, background-subtracted measured intensity. Used as
        the bootstrap center only if `center` is None (legacy behavior).
    db : (M,) per-point uncertainty on b, used both as the noise
        generating sigma for the parametric bootstrap and (inside
        solve_fn, as always) for chi2 weighting.
    progress : callback(done, total) after every finished replicate (e.g. for a GUI status line).
    should_abort : callable; when it returns True, no further replicates are started and the
        result is built from the finished ones.
    workers : threads for the re-solves (None: PYSASFIT_THREADS or the CPU count; 1: serial). The replicates
        are drawn before the solves, so the result is the same for any number of workers.
    n_boot : number of replicates to attempt. A handful can fail (solver
        non-convergence, a lambda search that can't bracket a root on a
        particular noisy draw, ...) -- those are silently skipped and
        counted in n_failed rather than aborting the whole run, since on
        a reasonably-behaved problem a small failure rate is expected and
        shouldn't block getting an answer from the rest.
    rng : optional numpy Generator for reproducibility; a fresh default
        generator is used otherwise.
    center : (M,) noise-free curve the replicates are generated around --
        pass the single solve's fitted_b. STRONGLY recommended: the
        measured b already contains one realization of noise, so b + noise
        has variance ~2*db^2. Every replicate then has a true chi2_r of ~2,
        and any solver forced to chi2_r=1 (discrepancy principle) must
        overfit it. Found 2026-10-10 on test.dat with
        bayesian_evidence_two_segment + target_chi2_r=1: centering on b
        drove lambda down by up to 1e8 on some replicates (std(p) up to
        5.9, p peak ~1); centering on fitted_b gave std(p) <= 0.014.

    Raises
    ------
    RuntimeError if every single replicate failed (nothing to summarize).
    """
    if rng is None:
        rng = np.random.default_rng()

    base = np.asarray(b if center is None else center, dtype=float)
    # All noise realisations are drawn first, in the same order as a serial loop, so the replicates (and the
    # result) do not depend on the number of workers.  The re-solves are independent: with workers != 1 they
    # run in worker processes when one solve takes longer than ~0.2 s and solve_fn can be pickled (use
    # replicate_solver(); a lambda/closure cannot) -- threads do not help, the iterative solvers are limited by
    # the Python interpreter lock.
    reps = [base + rng.normal(0.0, db) for _ in range(n_boot)]

    def one(b_rep):
        result = solve_fn(b_rep)
        x = np.asarray(result.x, dtype=float)
        if x.size == 0 or not np.all(np.isfinite(x)):
            raise ValueError("solver returned an empty or non-finite result")
        return x

    results = _run_replicates(one, solve_fn, reps, workers, progress=progress, should_abort=should_abort)
    samples: list[np.ndarray] = []
    n_failed = 0
    for x in results:
        if isinstance(x, Exception):
            n_failed += 1
        else:
            samples.append(x)

    if not samples:
        raise RuntimeError(
            f"All {n_boot} bootstrap replicates failed -- cannot estimate "
            "uncertainty this way for this solver/data combination."
        )

    X = np.vstack(samples)
    return BootstrapResult(
        x_mean=X.mean(axis=0),
        x_std=X.std(axis=0),
        x_samples=X,
        n_boot=X.shape[0],
        n_failed=n_failed,
    )


# ------------------------------------------------------------------------------------------------------------
# parallel re-solves
# ------------------------------------------------------------------------------------------------------------
class _ReplicateSolver:
    """Picklable solve_fn: run_solver(key, A, b_rep, db, spline=spline, **kwargs)."""

    def __init__(self, key, A, db, spline=None, **kwargs):
        from . import solver_registry
        self.key, self.A, self.db, self.spline, self.kwargs = key, np.asarray(A), np.asarray(db), spline, kwargs
        self.options = dict(solver_registry.OPTIONS)     # numerical options of the parent (worker processes)
        self._pid = os.getpid()

    def __call__(self, b_rep):
        from . import solver_registry
        if os.getpid() != self._pid:                      # in a worker process: the parent's options
            solver_registry.apply_options(self.options)
            self._pid = os.getpid()
        return solver_registry.run_solver(self.key, self.A, b_rep, self.db, spline=self.spline, **self.kwargs)


def replicate_solver(key, A, db, spline=None, **kwargs):
    """solve_fn for :func:`bootstrap_uncertainty` that can be sent to worker processes."""
    return _ReplicateSolver(key, A, db, spline, **kwargs)


def _call_replicate(args):
    one, b_rep = args
    try:
        return one(b_rep)
    except Exception as e:  # noqa: BLE001
        return e


class _One:
    """Picklable wrapper: solve, check x (the body of bootstrap_uncertainty's `one`)."""

    def __init__(self, solve_fn):
        self.solve_fn = solve_fn

    def __call__(self, b_rep):
        result = self.solve_fn(b_rep)
        x = np.asarray(result.x, dtype=float)
        if x.size == 0 or not np.all(np.isfinite(x)):
            raise ValueError("solver returned an empty or non-finite result")
        return x


def _run_replicates(one, solve_fn, reps, workers, min_seconds=0.2, progress=None, should_abort=None):
    import pickle
    import time
    from concurrent.futures import ProcessPoolExecutor
    from concurrent.futures import TimeoutError as _FutTimeout

    from ..core.accel import n_workers
    w = 1 if workers == 1 else min(n_workers(workers), len(reps))
    out = []
    n = len(reps)
    aborted = (lambda: bool(should_abort and should_abort()))

    def tick():
        if progress is not None:
            progress(len(out), n)
    if not reps or aborted():
        return []
    t0 = time.perf_counter()
    out.append(_call_replicate((one, reps[0])))
    tick()
    slow = time.perf_counter() - t0 > min_seconds
    if w > 1 and slow and len(reps) > 2:
        try:
            job = _One(solve_fn)
            pickle.dumps(job)
        except Exception:  # noqa: BLE001 -- closure / lambda: serial
            job = None
        if job is not None:
            try:
                import multiprocessing as mp
                from ..core.accel import single_thread_blas_env
                # 'spawn' everywhere: forking a process that runs Qt threads is unsafe (Windows spawns anyway).
                # One BLAS thread per worker (else N workers x N OpenBLAS threads exhaust memory on Windows),
                # and at most 8 workers unless PYSASFIT_THREADS / workers asks for more.
                nw = min(w if (workers or os.environ.get("PYSASFIT_THREADS")) else min(w, 8), len(reps) - 1)
                with single_thread_blas_env():
                    ex = ProcessPoolExecutor(max_workers=nw, mp_context=mp.get_context("spawn"))
                    futs = [ex.submit(_call_replicate, (job, b)) for b in reps[1:]]   # workers start here
                # Watchdog: worker processes that do not deliver (could not start, hang) must not freeze the
                # caller -- after a generous timeout the remaining replicates are solved serially.
                t_one = time.perf_counter() - t0
                deadline = time.perf_counter() + 60.0 + 4.0 * t_one * len(futs) / max(nw, 1)
                try:
                    for f in futs:
                        while True:
                            if aborted():
                                raise KeyboardInterrupt
                            left = deadline - time.perf_counter()
                            if left <= 0:
                                raise TimeoutError
                            try:
                                out.append(f.result(timeout=min(0.5, left)))
                                break
                            except (_FutTimeout, TimeoutError):
                                continue
                        tick()
                except (KeyboardInterrupt, TimeoutError, Exception):  # noqa: BLE001 -- abort, timeout, broken pool
                    pass
                ex.shutdown(wait=False, cancel_futures=True)
                for b in reps[len(out):]:
                    if aborted():
                        break
                    out.append(_call_replicate((one, b)))
                    tick()
                return out
            except Exception:  # noqa: BLE001 -- no worker processes available: serial
                del out[1:]
    for b in reps[1:]:
        if aborted():
            break
        out.append(_call_replicate((one, b)))
        tick()
    return out
