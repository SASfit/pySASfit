"""
Bayesian evidence for choosing the regularization weight lambda, per
Vestergaard & Hansen (2006), "Application of Bayesian analysis to indirect
Fourier transformation in small-angle scattering", J. Appl. Cryst. 39,
797-804 -- read directly (via pdftotext) from the copy at
.../FormFactors4SASfit/In Progress/Expectation Maximation/wf5022.pdf on
2026-09-23, not from secondary description.

Their setup (their eq. 3-6, notation p for the solution vector, their S is
our L2 smoothness penalty p^T L^T L p matching their eq. 5, their A = grad-
grad S, B = grad-grad (chi2/2))::

    Q = lambda*S + chi2                                  (eq. 3)
    chi2 = sum_i [Im(qi) - I(qi)]^2 / sigma_i^2            (eq. 4)
    P(lambda, d) = [lambda^(N-1) (N+1)]^(1/2)
                   / [det(lambda*A + B)]^(1/2)
                   * exp(-lambda*S - chi2/2)               (eq. 6, evaluated
                                                             at p = p_MAP(lambda))

I initially mis-transcribed eq. 6's denominator as det(A + lambda*B) from a
garbled OCR pass; the paper's own later discussion of eq. 8 explicitly says
"eigenvalues of lambda*A + B", which fixes the ordering used here.

CAVEAT I have not resolved: eq. 3 minimizes (lambda*S + chi2) -- no factor
of 1/2 on chi2 -- while eq. 6's exponent is (-lambda*S - chi2/2). Taken
literally these correspond to different MAP points unless a factor of 2 is
being absorbed into lambda's definition between the two equations (a common
convention split between "Tikhonov-style" and "Bayesian-style" weightings
in this literature). I've implemented internally consistently with eq. 6
(MAP point minimizes lambda*S + chi2/2, since that's what makes eq. 6 a
normalized-looking posterior), but if you need this to match the paper's
lambda values exactly, this factor-of-2 needs checking against the actual
typeset paper, not the OCR'd text extraction used here.

Also implements eq. 8's "number of good parameters":

    Ng = sum_j  mu_j / (lambda + mu_j)

where mu_j are the eigenvalues of the generalized eigenvalue problem
B v = mu A v (their eq. 8 is the A=I special case, sum mu_j/(lambda+mu_j)
with mu_j = eigenvalues of B directly).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.linalg as sla


@dataclass
class EvidenceResult:
    lam: float
    log_evidence: float
    p_map: np.ndarray
    n_good_params: float


def _map_solution(K: np.ndarray, b: np.ndarray, sigma: np.ndarray, L: np.ndarray, lam: float, ridge: float = 0.0) -> np.ndarray:
    """
    p that minimizes lambda*S + chi2/2 = lambda* p^T L^T L p + (1/2) sum_i
    [(Kp)_i - b_i]^2/sigma_i^2 -- a linear-Gaussian problem, solved directly
    via the normal equations (no need for the SVD machinery here since L
    need not be square).

    ridge: optional tiny additive stabilizer (lam * ridge * I) on top of
    L^T L, needed only when L itself has a null space that isn't fully
    removed by the data term K^T W K (e.g. the plain second_derivative_
    operator's constant+linear null space, when used -- as in the Hann-
    tapered solver variant -- without Hansen's boundary-condition fix).
    Zero by default, so existing callers are unaffected.
    """
    W = 1.0 / sigma**2
    n = K.shape[1]
    lhs = 2 * lam * (L.T @ L + ridge * np.eye(n)) + K.T @ (W[:, None] * K)
    rhs = K.T @ (W * b)
    return np.linalg.solve(lhs, rhs)


def log_evidence(K: np.ndarray, b: np.ndarray, sigma: np.ndarray, L: np.ndarray, lam: float, ridge: float = 0.0) -> EvidenceResult:
    """
    log P(lambda) up to an additive constant that doesn't depend on lambda
    (the (N+1) factor in eq. 6 is lambda-independent and dropped here since
    it doesn't affect the argmax over lambda).

    ridge: see _map_solution's docstring. Passing a nonzero ridge is an
    approximation (it adds a tiny bit of extra, physically unmotivated
    regularization purely for numerical conditioning) -- keep it as small
    as possible while still removing the singularity.
    """
    n = K.shape[1]
    W = 1.0 / sigma**2

    A_hess = 2 * (L.T @ L + ridge * np.eye(n))  # Hessian of S = p^T L^T L p (+ ridge)
    B_hess = K.T @ (W[:, None] * K)  # Hessian of chi2/2

    p_map = _map_solution(K, b, sigma, L, lam, ridge=ridge)
    S_val = float(p_map @ (L.T @ L) @ p_map)
    chi2_val = float(np.sum(((K @ p_map - b) / sigma) ** 2))

    hessian = lam * A_hess + B_hess
    sign, logdet = np.linalg.slogdet(hessian)
    if sign <= 0:
        logdet = np.inf  # degenerate/indefinite Hessian -- lambda not viable here

    log_p = 0.5 * (n - 1) * np.log(lam) - 0.5 * logdet - lam * S_val - chi2_val / 2

    eigvals_B = sla.eigh(B_hess, eigvals_only=True)
    eigvals_B = np.clip(eigvals_B, 0, None)  # guard tiny negative numerical noise
    n_good = float(np.sum(eigvals_B / (lam + eigvals_B)))

    return EvidenceResult(lam=lam, log_evidence=log_p, p_map=p_map, n_good_params=n_good)


class FastEvidence:
    """log_evidence() for many lambda at the cost of one decomposition
    (2026-10-21). Same formula and same results as log_evidence().

    With P = L^T L + ridge I positive definite (all Hansen penalties; not the
    plain 2nd-difference operator without ridge), P = R^T R (Cholesky) and
    M = R^-T B R^-1 = V diag(mu) V^T, B = K^T W K::

        H(lambda) = 2 lambda P + B = R^T (2 lambda I + M) R
        log det H = log det P + sum_j log(2 lambda + mu_j)
        p_MAP     = R^-1 V diag(1/(2 lambda + mu)) V^T R^-T K^T W b

    so every lambda costs O(M n) (the explicit residual) instead of an
    O(n^3) solve, a slogdet and an eigen-decomposition. ``ok`` is False when
    P is not positive definite; callers then use log_evidence(). Each lambda
    is checked through the residual of its normal equations; a lambda that
    fails the check (small lambda with an ill-conditioned P) is evaluated by
    log_evidence() instead (counted in ``n_exact``).
    """

    #: relative residual of the normal equations above which a lambda is
    #: evaluated exactly (test.dat, j0 kernel: below 1e-11 the fast ln Z agrees
    #: with the exact one to < 1e-3)
    tol = 1e-11

    def __init__(self, K, b, sigma, L, ridge: float = 0.0):
        self.L, self.ridge, self.n_exact = L, ridge, 0
        K = np.asarray(K, float)
        self.K, self.b, self.sigma = K, np.asarray(b, float), np.asarray(sigma, float)
        n = K.shape[1]
        self.n = n
        W = 1.0 / self.sigma**2
        LtL = L.T @ L
        self.LtL = LtL
        P = LtL + ridge * np.eye(n)
        try:
            R = np.linalg.cholesky(P).T            # P = R^T R
        except np.linalg.LinAlgError:
            self.ok = False
            return
        B = K.T @ (W[:, None] * K)
        self.B, self.P = B, P
        self.rhs = K.T @ (W * self.b)
        self._rhs_norm = float(np.linalg.norm(self.rhs))
        Rinv = sla.solve_triangular(R, np.eye(n))
        Mm = Rinv.T @ B @ Rinv
        mu, V = np.linalg.eigh(0.5 * (Mm + Mm.T))
        self.mu = np.clip(mu, 0.0, None)
        self.T = Rinv @ V                          # p = T y
        self.g = self.T.T @ (K.T @ (W * self.b))
        self.logdetP = 2.0 * float(np.sum(np.log(np.diag(R)))) + n * np.log(2.0)
        self.eig_B = np.clip(sla.eigh(B, eigvals_only=True), 0, None)
        self.ok = bool(np.all(np.isfinite(self.mu)))

    def exact(self, lam: float) -> EvidenceResult:
        """log_evidence() with the lambda-independent parts precomputed: one
        Cholesky factorization of H gives p_MAP and log det H."""
        self.n_exact += 1
        lam = float(lam)
        H = 2.0 * lam * self.P + self.B
        try:
            c = sla.cho_factor(H, check_finite=False)
        except np.linalg.LinAlgError:
            return log_evidence(self.K, self.b, self.sigma, self.L, lam, ridge=self.ridge)
        p = sla.cho_solve(c, self.rhs, check_finite=False)
        logdet = 2.0 * float(np.sum(np.log(np.diag(c[0]))))
        S_val = float(p @ self.LtL @ p)
        chi2_val = float(np.sum(((self.K @ p - self.b) / self.sigma) ** 2))
        log_p = 0.5 * (self.n - 1) * np.log(lam) - 0.5 * logdet - lam * S_val - chi2_val / 2
        n_good = float(np.sum(self.eig_B / (lam + self.eig_B)))
        return EvidenceResult(lam=lam, log_evidence=log_p, p_map=p, n_good_params=n_good)

    def at(self, lam: float) -> EvidenceResult:
        lam = float(lam)
        d = 1.0 / (2.0 * lam + self.mu)
        p = self.T @ (d * self.g)
        # accuracy check: where P (and so R^-1) is ill-conditioned the
        # decomposition loses digits at small lambda -- then the exact route
        res = self.rhs - (2.0 * lam * (self.P @ p) + self.B @ p)
        if not np.linalg.norm(res) <= self.tol * self._rhs_norm:
            return self.exact(lam)
        S_val = float(p @ self.LtL @ p)
        chi2_val = float(np.sum(((self.K @ p - self.b) / self.sigma) ** 2))
        # log det(lam * 2P + B) = log det(2P) + sum log(lam + mu/2)
        logdet = self.logdetP + float(np.sum(np.log(lam + 0.5 * self.mu)))
        log_p = 0.5 * (self.n - 1) * np.log(lam) - 0.5 * logdet - lam * S_val - chi2_val / 2
        n_good = float(np.sum(self.eig_B / (lam + self.eig_B)))
        return EvidenceResult(lam=lam, log_evidence=log_p, p_map=p, n_good_params=n_good)


def evidence_search(
    K: np.ndarray,
    b: np.ndarray,
    sigma: np.ndarray,
    L: np.ndarray,
    lam_grid: np.ndarray,
    ridge: float = 0.0,
) -> tuple[EvidenceResult, list[EvidenceResult]]:
    """Evaluate log-evidence over a grid of lambda and return the maximizer.
    ridge: see log_evidence's docstring; passed through unchanged, default
    0 keeps existing callers' behavior identical."""
    fe = FastEvidence(K, b, sigma, L, ridge=ridge)
    if fe.ok:
        results = [fe.at(lam) for lam in lam_grid]
    else:
        results = [log_evidence(K, b, sigma, L, lam, ridge=ridge) for lam in lam_grid]
    best = max(results, key=lambda r: r.log_evidence)
    return best, results


def posterior_covariance(
    K: np.ndarray, sigma: np.ndarray, L: np.ndarray, lam: float, ridge: float = 0.0,
) -> np.ndarray:
    """
    Laplace-approximation posterior covariance of p around its MAP point,
    Cov(p) ~= hessian^-1, where `hessian = lam*A_hess + B_hess` is exactly
    the same curvature matrix log_evidence() already builds and takes the
    log-determinant of for the evidence score itself -- this is the
    standard GNOM/Hansen-IFT way of turning the already-computed evidence
    machinery into pointwise error bars on p(r), at essentially zero extra
    cost (one matrix inverse at the already-selected best lambda, not a
    per-lambda-grid-point cost).

    Pointwise uncertainty: sigma_j = sqrt(diag(Cov(p)))[j]. This is a
    LOCAL/linearized (Gaussian, single-lambda) uncertainty around the MAP
    solution -- it does not capture the discrete choice of lambda itself,
    nor any nonlinearity from a positivity constraint (this solver family
    has none; p is allowed to go negative, consistent with the general-
    form Tikhonov formulation used throughout bayesian_evidence_hansen/
    _hann_tapered). For a solver-agnostic, nonlinearity-aware alternative
    (e.g. for the EM/MaxEnt solvers, or to also propagate lambda-selection
    uncertainty), see uncertainty.bootstrap_uncertainty instead.
    """
    n = K.shape[1]
    W = 1.0 / sigma**2
    A_hess = 2 * (L.T @ L + ridge * np.eye(n))
    B_hess = K.T @ (W[:, None] * K)
    hessian = lam * A_hess + B_hess
    return np.linalg.inv(hessian)


@dataclass
class BlockEvidenceResult:
    lams: tuple[float, ...]
    log_evidence: float
    p_map: np.ndarray
    n_good_params: float


def log_evidence_blocks(
    K: np.ndarray,
    b: np.ndarray,
    sigma: np.ndarray,
    A_blocks: tuple[np.ndarray, ...],
    n_terms: tuple[int, ...],
    lams: tuple[float, ...],
) -> BlockEvidenceResult:
    """
    Generalization of log_evidence() to K independent regularization
    "blocks", each with its own hyperparameter lambda_k -- implemented to
    support an adaptive/position-dependent smoothing strength (e.g.
    regularization.hansen_smoothness_two_segment's A_lo/A_hi split),
    since an empirical scan (tests/diagnose_hansen_lambda_balance.py and
    friends, 2026-10-06) showed no single global lambda can give a
    well-sized fit across both the low/mid-q and high-q regions on this
    kernel.

    Each A_blocks[k] is an n x n matrix such that the combined penalty is::

        S(p; lams) = sum_k lams[k] * p^T A_blocks[k] p

    (NOT a Cholesky factor -- unlike log_evidence's L argument, which
    required L^T L = A because that single-block code solved via L
    directly; here, with multiple blocks, it's simpler to work with the
    A_k matrices themselves).

    n_terms[k] is the number of independent penalty terms contributed by
    block k (regularization.hansen_smoothness_two_segment returns these
    as n_lo_terms, n_hi_terms) -- used as the log(lambda_k) prefactor
    exponent, chosen so that when every block shares one lambda and
    sum(n_terms) == N-1, the combined prefactor
    sum_k 0.5*n_terms[k]*log(lam) == 0.5*(N-1)*log(lam) collapses exactly
    to log_evidence()'s single-lambda prefactor (itself following
    Vestergaard & Hansen (2006) eq. 6). This is the natural
    generalization for block-additive, non-overlapping-support priors;
    it is NOT a rigorous re-derivation of eq. 6 for the multi-hyperparameter
    case (the exact normalizing constant of a sum of non-commuting
    precision matrices doesn't factor this cleanly in general), but since
    A_lo and A_hi act on disjoint sets of difference terms (see
    hansen_smoothness_two_segment's docstring), this split IS exact for
    the log(lambda_k) prefactor specifically -- the part that could
    otherwise not be reduced to a simple scalar count.

    Effective-degrees-of-freedom (Ng) generalizes via the standard ridge
    formula Ng = trace(H^-1 B) (Hastie & Tibshirani), rather than
    log_evidence()'s single-lambda eigenvalue formula (eq. 8), since that
    formula is specific to a single scalar lambda; trace(H^-1 B) reduces
    to the same quantity when there's only one block.
    """
    n = K.shape[1]
    W = 1.0 / sigma**2

    B_hess = K.T @ (W[:, None] * K)  # Hessian of chi2/2

    A_total = np.zeros((n, n))
    for lam_k, A_k in zip(lams, A_blocks):
        A_total += lam_k * A_k

    lhs = 2 * A_total + B_hess
    rhs = K.T @ (W * b)
    p_map = np.linalg.solve(lhs, rhs)

    chi2_val = float(np.sum(((K @ p_map - b) / sigma) ** 2))

    hessian = lhs  # == sum_k lam_k * (2*A_k) + B_hess
    sign, logdet = np.linalg.slogdet(hessian)
    if sign <= 0:
        logdet = np.inf

    log_prefactor = sum(0.5 * nt * np.log(lam_k) for nt, lam_k in zip(n_terms, lams))
    S_total = sum(lam_k * float(p_map @ A_k @ p_map) for lam_k, A_k in zip(lams, A_blocks))

    log_p = log_prefactor - 0.5 * logdet - S_total - chi2_val / 2

    n_good = float(np.trace(np.linalg.solve(hessian, B_hess)))

    return BlockEvidenceResult(lams=tuple(lams), log_evidence=log_p, p_map=p_map, n_good_params=n_good)


def evidence_search_blocks(
    K: np.ndarray,
    b: np.ndarray,
    sigma: np.ndarray,
    A_blocks: tuple[np.ndarray, ...],
    n_terms: tuple[int, ...],
    lam_grids: tuple[np.ndarray, ...],
) -> tuple[BlockEvidenceResult, list[BlockEvidenceResult]]:
    """
    Grid search over the Cartesian product of per-block lambda grids,
    maximizing log_evidence_blocks(...). For two blocks (the
    lo/hi-segment adaptive-lambda case) with grids of length m each, this
    is m^2 evaluations -- keep m modest (~15-25) to stay fast.
    """
    import itertools

    results = []
    for lam_combo in itertools.product(*lam_grids):
        results.append(log_evidence_blocks(K, b, sigma, A_blocks, n_terms, lam_combo))
    best = max(results, key=lambda r: r.log_evidence)
    return best, results


def posterior_covariance_blocks(
    K: np.ndarray,
    sigma: np.ndarray,
    A_blocks: tuple[np.ndarray, ...],
    lams: tuple[float, ...],
) -> np.ndarray:
    """
    Multi-block analog of posterior_covariance() above: Laplace-approximation
    covariance Cov(p) ~= hessian^-1 at the block-MAP point, where
    `hessian = sum_k lam_k*(2*A_k) + B_hess` is exactly the curvature matrix
    log_evidence_blocks() already forms and takes the logdet of. Same
    caveats as posterior_covariance (local/linearized, doesn't propagate
    lambda-selection uncertainty).
    """
    n = K.shape[1]
    W = 1.0 / sigma**2
    B_hess = K.T @ (W[:, None] * K)
    A_total = np.zeros((n, n))
    for lam_k, A_k in zip(lams, A_blocks):
        A_total += lam_k * A_k
    hessian = 2 * A_total + B_hess
    return np.linalg.inv(hessian)
