"""PDDF kernels for the three symmetries of Hansen (2016), eqs. 1-3.

The forward models of pysasfit.inversion (kernel_registry) must agree with
the density-profile code of pysasfit.sldprofile: the exact intensity of a
radial profile rho(r) must be reproduced by the kernel applied to the PDDF
p(r) = r^(dim-1) gamma(r) computed from the same profile, and an inversion
of noisy data must recover that p(r).
"""
import numpy as np
import pytest

from pysasfit.inversion.kernel_registry import KERNEL_REGISTRY
from pysasfit.inversion.kernels import build_kernel_on_nodes, make_r_grid
from pysasfit.inversion.solver_registry import run_solver
from pysasfit.sldprofile import geometry as geo
from pysasfit.sldprofile import models

R = 5.0
PROFILE = models.core_shell(3.0, 1.0, 0.4)
CASES = [("sphere", "sinc_4pi", "hansen_eq19"),
         ("cylinder", "cylinder_pddf", "hansen_eq19"),
         ("lamella", "lamella_pddf", "hansen_eq19_even")]


def _setup(geometry, key, noise=0.005):
    q = np.linspace(0.02, 2.0, 150)
    I = models.intensity_from_profile(q, PROFILE, R, geometry)
    sig = noise * I + 1e-4 * I[0]
    b = I + np.random.default_rng(1).normal(size=q.size) * sig
    r = make_r_grid(2 * R, 60)
    A = build_kernel_on_nodes(q, r, KERNEL_REGISTRY[key].func, interpolation="linear")
    e = geo.shell_edges(R, 400)
    _, p_true = geo.pddf(PROFILE(0.5 * (e[1:] + e[:-1])), R, geometry, r=r)
    return q, I, b, sig, r, A, p_true


@pytest.mark.parametrize("geometry,key,penalty", CASES)
def test_forward_model_matches_sldprofile(geometry, key, penalty):
    q, I, b, sig, r, A, p_true = _setup(geometry, key)
    # discretization error of p(r) (60 points) and of geo.pddf below the
    # noise level of the synthetic data (0.5 % + 1e-4 I(q_min))
    assert np.sqrt(np.mean(((A @ p_true - I) / sig) ** 2)) < 1.5


@pytest.mark.parametrize("geometry,key,penalty", CASES)
def test_inversion_recovers_pddf(geometry, key, penalty):
    q, I, b, sig, r, A, p_true = _setup(geometry, key)
    res = run_solver("bayesian_evidence", A, b, sig, penalty=penalty)
    x = np.asarray(res.x)
    assert 0.5 < res.chi2_r_history[-1] < 1.5
    assert np.linalg.norm(x - p_true) / np.linalg.norm(p_true) < 0.06


def test_lamella_needs_free_p0():
    """p_t(0) = int rho^2 dx > 0: pinning p(0) to 0 biases the result and is
    disfavoured by the evidence."""
    q, I, b, sig, r, A, p_true = _setup("lamella", "lamella_pddf")
    free = run_solver("bayesian_evidence", A, b, sig, penalty="hansen_eq19_even")
    pinned = run_solver("bayesian_evidence", A, b, sig, penalty="hansen_eq19")
    assert abs(free.x[0] - p_true[0]) < 0.15 * p_true[0]
    assert abs(pinned.x[0] - p_true[0]) > 0.5 * p_true[0]
    assert (free.diagnostics["log_evidence_normalized"]
            > pinned.diagnostics["log_evidence_normalized"])


@pytest.mark.parametrize("key,factor_rg", [("cylinder_pddf", np.sqrt(2)), ("lamella_pddf", np.sqrt(12))])
def test_homogeneous_size_from_rg(key, factor_rg):
    """Homogeneous rod / sheet: R = sqrt(2) R_c and T = sqrt(12) R_t from the
    analytic PDDFs, and forward value q^k I -> forward_factor * int p dr."""
    spec = KERNEL_REGISTRY[key]
    D = 2 * R
    r = np.linspace(0, D, 4001)
    if spec.geometry == "lamella":
        p = D - r                                    # gamma_t of a sheet of thickness D
    else:
        u = np.clip(r / D, 0, 1)                     # disc of diameter D: gamma_c(r) * r
        p = r * (np.arccos(u) - u * np.sqrt(1 - u**2))
    rg = np.sqrt(np.trapezoid(p * r**2, r) / (2 * np.trapezoid(p, r)))
    size = D if spec.geometry == "lamella" else R
    assert factor_rg * rg == pytest.approx(size, rel=1e-3)
    qk = 1e-4
    k = 1 if spec.geometry == "cylinder" else 2
    A = np.array([spec.func(np.array([qk]), rj)[0] for rj in r])
    I0 = np.trapezoid(A * p, r) * qk**k
    assert I0 == pytest.approx(spec.forward_factor * np.trapezoid(p, r), rel=1e-4)
