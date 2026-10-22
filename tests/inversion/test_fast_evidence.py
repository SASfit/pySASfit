"""FastEvidence (one decomposition for all lambda) reproduces log_evidence."""
import os

import numpy as np
import pytest

from pysasfit.inversion.kernel_registry import KERNEL_REGISTRY
from pysasfit.inversion.kernels import build_kernel_on_nodes, make_r_grid
from pysasfit.inversion.solver_registry import _evidence_grid, _penalty_operator, run_solver
from pysasfit.inversion.solvers import bayesian_evidence as be

DATA = os.path.join(os.path.dirname(__file__), "data", "test.dat")


def _problem(n=95):
    d = np.loadtxt(DATA, comments="#")
    q, I, dI = d[:, 0], d[:, 1] - 0.1, d[:, 2]
    r = make_r_grid(np.pi / q.min(), n)
    return build_kernel_on_nodes(q, r, KERNEL_REGISTRY["sinc_4pi"].func), I, dI


@pytest.mark.parametrize("penalty", ["hansen_eq19", "hansen_printed", "hansen_eq19_even"])
def test_fast_evidence_matches_direct(penalty):
    A, I, dI = _problem()
    L = _penalty_operator(penalty, A.shape[1])
    fe = be.FastEvidence(A, I, dI, L)
    assert fe.ok
    for lam in _evidence_grid(A, dI, L)[::6]:
        a, b = fe.at(lam), be.log_evidence(A, I, dI, L, lam)
        assert abs(a.log_evidence - b.log_evidence) < 1e-3
        assert np.max(np.abs(a.p_map - b.p_map)) < 1e-3 * np.abs(b.p_map).max()
        assert np.isclose(a.n_good_params, b.n_good_params)


def test_singular_penalty_falls_back():
    A, I, dI = _problem(40)
    L = _penalty_operator("second_derivative", 40)
    assert not be.FastEvidence(A, I, dI, L).ok
    res = run_solver("bayesian_evidence", A, I, dI, penalty="second_derivative")
    assert np.isfinite(res.chi2_r_history[-1])
