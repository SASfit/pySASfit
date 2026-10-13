r"""Two-step (indirect) estimation of the density profile, after Glatter.

Glatter's route to rho(r) of a centro-symmetric scatterer takes two steps:

1. **Indirect Fourier transformation** (Glatter 1977): I(q) -> p(r), the
   distance distribution of the sphere, of the cylinder cross-section or of the
   lamella thickness, with D_max = 2 r_max.
2. **Deconvolution** (Glatter 1981, "DECON"): p(r) = rho^T V(r) rho is the
   self-convolution of the density profile; rho(r) is found by fitting this
   quadratic form to p(r) with a smoothness constraint.

Hansen (2016) replaces both steps by the direct estimate in
:class:`~pysasfit.sldprofile.DensityProfileFit`. Both routes are offered here so
they can be compared on the same data:

* step 1 is a Bayesian IFT (Tikhonov, Hansen 2000 eq. 19 curvature penalty with
  p(0) = p(D_max) = 0, lambda by maximum evidence) from :mod:`pysasfit.inversion`,
  with the geometry-specific transform and the instrumental resolution; D_max
  (and optionally a flat background) is chosen by the evidence;
* step 2 is :class:`DeconvolutionFit`, the same Bayesian machinery as the direct
  fit (shells, bases, smoothness or max-ent, alpha by evidence), with the
  data I(q) replaced by p(r) and the forward model by the quadratic form
  p_j = rho^T V_j rho. r_max = D_max / 2 is fixed by step 1, as in Glatter's
  method. The p(r) errors are by default taken with their full posterior
  covariance from step 1 (``errors="covariance"``); ``errors="diagonal"`` uses
  only the standard deviations, as Glatter's DECON did.

Transforms (normalisation identical to :func:`pysasfit.sldprofile.geometry.pddf`):

=========  =============================================
sphere     I(q) = 4 pi int p(r) sin(qr)/(qr) dr
cylinder   I(q) = (pi/q) 2 pi int p(r) J0(qr) dr        (per unit length)
lamella    I(q) = (2 pi/q^2) 2 int p(r) cos(qr) dr      (per unit area)
=========  =============================================
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import linalg, optimize, special

from . import geometry as geo
from .density import DensityProfileFit, ProfileResult
from ..core.resolution import NoResolution

__all__ = ["ift_kernel", "pddf_matrices", "PDDFResult", "indirect_ft", "DeconvolutionFit",
           "TwoStepResult", "two_step_fit"]


# ---------------------------------------------------------------------- #
# step 1: indirect Fourier transformation
# ---------------------------------------------------------------------- #
def ift_kernel(q, r, geometry="sphere"):
    """Matrix A with I(q_i) = sum_j A_ij p(r_j) for p on the uniform grid ``r``
    (interior points of [0, D_max]; p(0) = p(D_max) = 0)."""
    geo._check(geometry)
    q = np.asarray(q, float)[:, None]
    r = np.asarray(r, float)
    dr = r[1] - r[0] if r.size > 1 else r[0]
    x = q * r[None, :]
    if geometry == "sphere":
        K = 4 * np.pi * np.sinc(x / np.pi)
    elif geometry == "cylinder":
        K = (np.pi / q) * 2 * np.pi * special.j0(x)
    else:
        K = (2 * np.pi / q**2) * 2 * np.cos(x)
    return K * dr


@dataclass
class PDDFResult:
    """Result of step 1."""
    geometry: str
    r: np.ndarray                 # interior grid of [0, D_max]
    p: np.ndarray
    p_err: np.ndarray
    cov: np.ndarray
    D_max: float
    D_max_err: float
    background: float
    q: np.ndarray
    I_fit: np.ndarray             # model incl. background, smeared
    chi2_red: float
    log_evidence: float
    D_grid: np.ndarray = field(default_factory=lambda: np.zeros(0))
    D_posterior: np.ndarray = field(default_factory=lambda: np.zeros(0))
    note: str = ""

    def summary(self) -> str:
        return (f"step 1, IFT ({self.geometry}): D_max = {self.D_max:.5g} +/- {self.D_max_err:.2g}, "
                f"background = {self.background:.5g}, chi2/M = {self.chi2_red:.4g}, "
                f"ln Z = {self.log_evidence:.5g}")


def _ift_grid(D, n_r, geometry):
    """Grid of the IFT unknowns and the map S to the reported half grid.

    sphere / cylinder: interior points of [0, D] (p(0) = p(D) = 0).
    lamella: p = gamma(r) does NOT vanish at r = 0 (gamma(0) = int rho^2 dx), so
    the even function gamma is solved on the interior of [-D, D] (2 n_r + 1
    points, r = 0 in the middle, gamma(+-D) = 0) and reported on r >= 0 as the
    symmetrised (gamma(r) + gamma(-r)) / 2.
    """
    if geometry != "lamella":
        r = D * np.arange(1, n_r + 1) / (n_r + 1)
        return r, r, np.eye(n_r), 1.0
    nf = 2 * n_r + 1
    rf = -D + 2 * D * np.arange(1, nf + 1) / (nf + 1)
    S = np.zeros((n_r + 1, nf))
    for j in range(n_r + 1):
        S[j, n_r + j] += 0.5
        S[j, n_r - j] += 0.5
    # cos is even: the half-range kernel's factor 2 becomes 1 on [-D, D]
    return rf, rf[n_r:], S, 0.5


def _penalty_factor(n, penalty):
    """L with S = |L p|^2 / 2: Hansen (2000) eq. 19 (curvature) or the
    first-difference matrix printed in its text; both with p(0) = p(D) = 0."""
    if penalty == "hansen_eq19":
        D = np.zeros((n - 2, n))
        for j in range(n - 2):
            D[j, j:j + 3] = (-0.5, 1.0, -0.5)
        C = D.T @ D
        C[0, 0] += 0.5
        C[-1, -1] += 0.5
    elif penalty == "hansen_printed":
        C = np.eye(n) - 0.5 * (np.eye(n, k=1) + np.eye(n, k=-1))
    else:
        raise ValueError("penalty must be 'hansen_eq19' or 'hansen_printed'")
    return C


class _IFTSolution:
    """Bayesian IFT at fixed D_max: x, cov, log evidence, chi2, N_good."""

    def __init__(self, x, cov, log_evidence, chi2, n_good, alpha):
        self.x, self.cov, self.log_evidence, self.chi2 = x, cov, log_evidence, chi2
        self.n_good, self.alpha = n_good, alpha


#: Numerical details of the step-1 IFT (overridable with ``ift_options``):
#: log_alpha_lo / log_alpha_hi / n_alpha -- alpha grid, ln alpha relative to
#: ln(tr B / tr C); alpha_xatol -- refinement of ln alpha; bg_below / bg_above --
#: flat-background search interval [I_min - bg_below*s, I_min + bg_above*s] with
#: s = max(|I_min|, median sigma); bg_xatol -- its precision in units of s.
IFT_DEFAULTS = dict(log_alpha_lo=-40.0, log_alpha_hi=10.0, n_alpha=101, alpha_xatol=1e-4,
                    bg_below=1.0, bg_above=0.5, bg_xatol=1e-4)


def _bayes_ift(A, b, sigma, C, opt=None):
    """Minimise chi^2/2 + alpha p^T C p / 2 with alpha by maximum evidence.

    log Z = -alpha S - chi^2/2 - 1/2 logdet(B + alpha C) + 1/2 logdet(alpha C)
    (Gull/MacKay; Hansen 2000), B = A^T W A. The alpha search is scaled by
    tr(B)/tr(C), so it does not depend on how many singular values of A are at
    the noise level (a grid scaled by the median singular value misses the
    maximum completely for a well-sampled p(r) grid).
    """
    W = A / sigma[:, None]
    y = b / sigma
    B = W.T @ W
    g = W.T @ y
    n = C.shape[0]
    _, ldC = np.linalg.slogdet(C)
    a0 = np.trace(B) / np.trace(C)

    def at(la):
        al = np.exp(la)
        H = B + al * C
        try:
            cH = linalg.cho_factor(H)
        except linalg.LinAlgError:
            return None
        x = linalg.cho_solve(cH, g)
        res = W @ x - y
        chi2 = float(res @ res)
        S = 0.5 * float(x @ C @ x)
        ldH = 2 * np.sum(np.log(np.diag(cH[0])))
        lz = -al * S - 0.5 * chi2 - 0.5 * ldH + 0.5 * (n * la + ldC)
        return lz, x, cH, chi2, al

    o = {**IFT_DEFAULTS, **(opt or {})}
    las = np.log(a0) + np.linspace(o["log_alpha_lo"], o["log_alpha_hi"], int(o["n_alpha"]))
    vals = [at(la) for la in las]
    lz = np.array([v[0] if v is not None else -np.inf for v in vals])
    i = int(np.argmax(lz))
    lo, hi = las[max(i - 1, 0)], las[min(i + 1, las.size - 1)]
    o = optimize.minimize_scalar(lambda la: -(at(la) or (-np.inf,))[0], bounds=(lo, hi), method="bounded",
                                 options={"xatol": o["alpha_xatol"]})
    best = at(o.x)
    if best is None or best[0] < lz[i]:
        best = vals[i]
    lzb, x, cH, chi2, al = best
    cov = linalg.cho_solve(cH, np.eye(n))
    n_good = float(n - al * np.trace(C @ cov))
    return _IFTSolution(x, cov, float(lzb), chi2, n_good, al)


def _ift_at(q, I, sigma, geometry, D, n_r, qf, R, bg, penalty, opt=None):
    """One IFT at fixed D_max and flat background bg. Returns (sol, r_half, A)."""
    rf, r, S, fac = _ift_grid(D, n_r, geometry)
    A = fac * ift_kernel(qf, rf, geometry)
    if R is not None:
        A = R @ A
    sol = _bayes_ift(A, I - bg, sigma, _penalty_factor(rf.size, penalty), opt)
    sol.S = S
    return sol, r, A


def _logz(sol):
    return sol.log_evidence if np.isfinite(sol.log_evidence) else -np.inf


def indirect_ft(q, I, sigma, geometry="sphere", D_max=None, D_range=None, n_D=15, n_r=60,
                resolution=None, background=False, penalty="hansen_eq19",
                progress=None, ift_options=None) -> PDDFResult:
    """Step 1: p(r) from I(q) by a Bayesian IFT.

    D_max : fixed value; with ``D_range`` (lo, hi) the evidence is scanned over
        ``n_D`` values and the posterior mean is used (default range
        (0.7, 1.5) x 2 r_max of a Guinier estimate when neither is given)
    background : False (none), a number (fixed flat background) or True
        (flat background by maximum evidence at each D_max)
    ift_options : overrides of :data:`IFT_DEFAULTS`
    """
    unknown = set(ift_options or {}) - set(IFT_DEFAULTS)
    if unknown:
        raise ValueError(f"unknown ift_options {sorted(unknown)}")
    o = {**IFT_DEFAULTS, **(ift_options or {})}
    q = np.asarray(q, float)
    I = np.asarray(I, float)
    sigma = np.asarray(sigma, float)
    resolution = NoResolution() if resolution is None else resolution
    qf, R = resolution.setup(q)

    def solve_D(D):
        if background is True:
            # evidence as a function of b: bounded 1-D search
            Imin = float(np.min(I))
            span = max(abs(Imin), float(np.median(sigma)), 1e-300)
            cache = {}

            def f(b):
                rr = _ift_at(q, I, sigma, geometry, D, n_r, qf, R, b, penalty, o)
                cache[b] = rr
                return -_logz(rr[0])

            ob = optimize.minimize_scalar(f, bounds=(Imin - o["bg_below"] * span, Imin + o["bg_above"] * span),
                                          method="bounded", options={"xatol": o["bg_xatol"] * span})
            b = float(ob.x)
            res, r, A = cache.get(ob.x) or _ift_at(q, I, sigma, geometry, D, n_r, qf, R, b, penalty, o)
        else:
            b = float(background) if background else 0.0
            res, r, A = _ift_at(q, I, sigma, geometry, D, n_r, qf, R, b, penalty, o)
        return res, r, A, b

    if D_max is not None and D_range is None:
        Ds = np.array([float(D_max)])
    else:
        if D_range is None:
            guess = DensityProfileFit(q, np.clip(I, 1e-300, None), sigma, geometry=geometry, n=4).guess_r_max()
            D_range = (0.7 * 2 * guess, 1.5 * 2 * guess)
        Ds = np.linspace(D_range[0], D_range[1], int(n_D))
    sols = []
    for D in Ds:
        s = solve_D(D)
        sols.append(s)
        if progress is not None:
            progress(f"IFT D_max={D:.5g}  ln Z={_logz(s[0]):.6g}  "
                     f"chi2/M={float(np.sum(((s[2] @ s[0].x + s[3] - I) / sigma) ** 2)) / q.size:.3f}")
    lz = np.array([_logz(s[0]) for s in sols])
    if Ds.size > 1 and np.all(np.isfinite(lz)):
        post = np.exp(lz - lz.max())
        post /= np.trapezoid(post, Ds)
        mean = float(np.trapezoid(post * Ds, Ds))
        sd = float(np.sqrt(max(np.trapezoid(post * (Ds - mean) ** 2, Ds), 0.0)))
        i = int(np.argmax(lz))
        # final solution at the posterior mean (a fine 2nd scan is not needed:
        # the deconvolution only uses one D_max)
        res, r, A, b = solve_D(mean)
        note = f"D_max by evidence over [{Ds[0]:.4g}, {Ds[-1]:.4g}] ({Ds.size} values), best {Ds[i]:.5g}"
    else:
        post = np.ones(Ds.size)
        mean, sd = float(Ds[0]), 0.0
        res, r, A, b = sols[0]
        note = "D_max fixed"
    cov = res.cov
    Ifit = A @ res.x + b
    chi2 = float(np.sum(((Ifit - I) / sigma) ** 2))
    S = res.S
    p = S @ np.asarray(res.x, float)
    cov = S @ np.asarray(cov, float) @ S.T
    return PDDFResult(geometry=geometry, r=r, p=p, p_err=np.sqrt(np.clip(np.diag(cov), 0, None)),
                      cov=cov, D_max=mean, D_max_err=sd, background=b, q=q, I_fit=Ifit,
                      chi2_red=chi2 / q.size, log_evidence=_logz(res), D_grid=Ds, D_posterior=post, note=note)


# ---------------------------------------------------------------------- #
# step 2: deconvolution p(r) -> rho(r)
# ---------------------------------------------------------------------- #
def _sub_shells(r_max, n, geometry, basis, grid, n_sub):
    """Thin uniform sub-shells (centres, width) and the linear map B from the
    shell coefficients to rho on them."""
    m = n * n_sub
    e = geo.shell_edges(r_max, m)
    rc = 0.5 * (e[1:] + e[:-1])
    B = np.empty((m, n))
    for k in range(n):
        c = np.zeros(n)
        c[k] = 1.0
        B[:, k] = geo.coef_to_rho(c, rc, r_max, geometry, basis, grid)
    return rc, e[1] - e[0], B


def pddf_matrices(r, r_max, n, geometry="sphere", basis="rho", grid=None, n_sub=6, half_width=None):
    """V with p(r_j) = rho^T V[j] rho, p averaged over [r_j - h, r_j + h].

    ``h`` defaults to half the spacing of ``r`` (the bin a p(r) value
    represents). The thin-shell construction is that of
    :func:`pysasfit.sldprofile.geometry.pddf`; the pair-distance distributions
    of two shells are integrated exactly over each bin.
    """
    geo._check(geometry)
    r = np.asarray(r, float)
    h = 0.5 * (r[1] - r[0]) if half_width is None else float(half_width)
    rc, ds, B = _sub_shells(r_max, n, geometry, basis, grid, n_sub)
    S, T = np.meshgrid(rc, rc, indexing="ij")
    if geometry == "sphere":
        Bw = (4 * np.pi * rc**2 * ds)[:, None] * B
        d2, st4 = (S - T) ** 2, 4 * S * T

        def cdf(x):  # distance CDF of two thin spherical shells
            return np.clip((x * x - d2) / st4, 0.0, 1.0)

        norm = 1.0 / (4 * np.pi)
    elif geometry == "cylinder":
        Bw = (2 * np.pi * rc * ds)[:, None] * B
        s2t2, st2 = S**2 + T**2, 2 * S * T

        def cdf(x):  # two rings: theta uniform on [0, pi]
            return np.arccos(np.clip((s2t2 - x * x) / st2, -1.0, 1.0)) / np.pi

        norm = 1.0 / (2 * np.pi)
    else:
        # full symmetric profile on [-r_max, r_max]: 2m cells of width ds
        pos = np.r_[-rc[::-1], rc]
        Bw = ds * np.vstack([B[::-1], B])
        C = pos[:, None] - pos[None, :]

        def tri_cdf(x):  # CDF of U - V, U, V uniform on cells of width ds centred at C
            u = (x - C) / ds
            return np.where(u <= -1, 0.0, np.where(u <= 0, 0.5 * (1 + u) ** 2,
                            np.where(u <= 1, 1 - 0.5 * (1 - u) ** 2, 1.0)))

        def cdf(x):  # CDF of |U - V|
            return tri_cdf(x) - tri_cdf(-x) if x > 0 else np.zeros_like(C)

        norm = 0.5
    V = np.empty((r.size, n, n))
    for j, rj in enumerate(r):
        a, b = max(rj - h, 0.0), rj + h
        K = (cdf(b) - cdf(a)) * (norm / (b - a))
        Vj = Bw.T @ K @ Bw
        V[j] = 0.5 * (Vj + Vj.T)
    return V


class DeconvolutionFit(DensityProfileFit):
    """Step 2: Bayesian estimate of rho(r) from a distance distribution p(r).

    Same options as :class:`DensityProfileFit` (geometry, n, regularization,
    prior, outer, basis, grid); there is no background and no resolution.
    ``cov`` is the covariance of ``p`` (``errors="covariance"``) -- with
    ``errors="diagonal"`` only its diagonal, or ``sigma``, is used.
    ``q_max`` (of the measured data) sets the smallest meaningful shell width
    pi/q_max for adaptive grids.
    """

    def __init__(self, r, p, sigma=None, cov=None, geometry="sphere", n=40, regularization="smooth",
                 prior=None, outer="free", basis="rho", grid=None, errors="covariance", q_max=None, n_sub=6,
                 options=None):
        r = np.asarray(r, float)
        p = np.asarray(p, float)
        if cov is None and sigma is None:
            raise ValueError("give sigma or cov")
        if cov is not None:
            cov = np.asarray(cov, float)
            sigma = np.sqrt(np.clip(np.diag(cov), 1e-300, None))
        sigma = np.asarray(sigma, float)
        self.r_data, self.p_data, self.p_sigma = r, p, sigma
        if errors == "covariance" and cov is not None:
            # whitening: residuals L^-1 (p_model - p) with cov = L L^T
            c = cov + 1e-12 * np.trace(cov) / cov.shape[0] * np.eye(cov.shape[0])
            self._Linv = linalg.solve_triangular(linalg.cholesky(c, lower=True), np.eye(c.shape[0]), lower=True)
            Iw = self._Linv @ p
            sw = np.ones_like(p)
        elif errors in ("covariance", "diagonal"):
            self._Linv = None
            Iw, sw = p, sigma
        else:
            raise ValueError("errors must be 'covariance' or 'diagonal'")
        self.errors = errors if cov is not None else "diagonal"
        self.n_sub = int(n_sub)
        self.q_max = q_max
        super().__init__(r, Iw, sw, geometry=geometry, n=n, regularization=regularization, prior=prior,
                         outer=outer, background=False, resolution=None, basis=basis, grid=grid, options=options)

    # forward model ------------------------------------------------------
    def _V(self, r_max):
        key = ("V", float(r_max))
        if key not in self._cache:
            if len(self._cache) > 16:
                self._cache.clear()
            self._cache[key] = pddf_matrices(self.r_data, r_max, self.n, self.geometry, self.basis, self.grid,
                                             n_sub=self.n_sub)
        return self._cache[key]

    def p_model(self, rho, r_max):
        """Unwhitened p(r) on the data grid."""
        return np.einsum("jkl,k,l->j", self._V(r_max), rho, rho)

    def model(self, rho, r_max, z=None, with_background=True):
        pm = self.p_model(rho, r_max)
        return pm if self._Linv is None else self._Linv @ pm

    def _model_jac(self, x, r_max):
        rho = x[: self.n]
        V = self._V(r_max)
        Vr = np.einsum("jkl,l->jk", V, rho)
        pm = np.einsum("jk,k->j", Vr, rho)
        J = 2.0 * Vr
        if self._Linv is not None:
            return self._Linv @ pm, self._Linv @ J
        return pm, J

    def guess_r_max(self) -> float:
        """Half the extent of p(r): the last r where p exceeds 1e-3 of its maximum, plus one step."""
        p = self.p_data
        k = np.nonzero(np.abs(p) > 1e-3 * np.abs(p).max())[0]
        dr = self.r_data[1] - self.r_data[0]
        return 0.5 * (self.r_data[k[-1]] + dr) if k.size else 0.5 * self.r_data[-1]

    def _min_width(self):
        return np.pi / self.q_max if self.q_max else (self.r_data[1] - self.r_data[0])

    def fit(self, r_max=None, **kw) -> ProfileResult:
        """As :meth:`DensityProfileFit.fit`; by default r_max = D_max/2 fixed (Glatter).
        The result's ``q`` / ``I_fit`` are r and the (unwhitened) fitted p(r)."""
        if r_max is None:
            r_max = self.guess_r_max()
            kw.setdefault("fix_r_max", True)
        res = super().fit(r_max=r_max, **kw)
        res.method = ("deconvolution of p(r), Glatter 1981; p(r) errors "
                      + ("with full covariance" if self._Linv is not None else "diagonal"))
        res.q = self.r_data
        res.I_fit = self.p_model(res.best.rho, res.best.r_max)
        res.chi2_red = res.best.chi2 / self.r_data.size
        return res


@dataclass
class TwoStepResult:
    pddf: PDDFResult
    profile: ProfileResult        # step 2; its q / I_fit are r and the fitted p(r)
    I_from_profile: np.ndarray    # I(q) recomputed from the deconvolved rho (incl. step-1 background)
    chi2_red_profile: float = float("nan")   # chi2/M of I_from_profile against the data

    def summary(self) -> str:
        pd = self.pddf
        lines = [
            "Two-step method (Glatter): 1. IFT  I(q) -> p(r),  2. deconvolution  p(r) -> rho(r)",
            "=" * 60,
            f"step 1  D_max      = {pd.D_max:.5g} +/- {pd.D_max_err:.2g}   (r_max = D_max/2)",
            f"        background = {pd.background:.5g}",
            f"        chi2/M     = {pd.chi2_red:.4g}    ln Z = {pd.log_evidence:.6g}",
            f"        {pd.note}",
            f"step 2  chi2/M of I(q) from rho(r) = {self.chi2_red_profile:.4g}",
            "",
            self.profile.summary(),
        ]
        return "\n".join(lines)

    def as_profile_result(self) -> ProfileResult:
        """The step-2 result in the layout of a direct fit, for display: q / I_fit /
        chi2 of the IFT fit, the D_max posterior as an r_max posterior, the
        step-1 background. ``.twostep`` points back to this object."""
        import copy

        pd = self.pddf
        v = copy.copy(self.profile)
        v.best = copy.copy(self.profile.best)
        bg = np.full(pd.q.size, pd.background) if pd.background else None
        v.best.bg_curve = bg
        v.best.I_fit = pd.I_fit
        v.q, v.I_fit, v.chi2_red, v.bg_curve = pd.q, pd.I_fit, pd.chi2_red, bg
        v.background = (pd.background, 0.0)
        v.r_max_grid = pd.D_grid / 2
        v.r_max_posterior = 2 * pd.D_posterior
        v.twostep = self
        return v

    def save(self, prefix: str) -> None:
        """As :meth:`ProfileResult.save` (the IFT fit in ``_fit.dat``, the D_max posterior as
        r_max in ``_rmax.dat``), plus ``_ift_pr.dat`` (r, p, dp, p of the deconvolved rho)
        and ``_I_from_rho.dat``."""
        self.as_profile_result().save(prefix)
        pd = self.pddf
        np.savetxt(f"{prefix}_ift_pr.dat", np.c_[pd.r, pd.p, pd.p_err, self.profile.I_fit],
                   header="r p(r)_IFT dp(r) p(r)_from_deconvolved_rho")
        np.savetxt(f"{prefix}_I_from_rho.dat", np.c_[pd.q, pd.I_fit, self.I_from_profile],
                   header="q I_IFT_fit I_from_deconvolved_rho")
        with open(f"{prefix}_params.txt", "w") as fh:
            fh.write(self.summary() + "\n")


def two_step_fit(q, I, sigma, geometry="sphere", n=40, regularization="smooth", prior=None, outer="free",
                 basis="rho", grid=None, resolution=None, background=False, r_max=None, r_max_range=None,
                 fix_r_max=False, n_D=15, n_r=60, errors="covariance", penalty="hansen_eq19",
                 n_draws=40, seed=0, progress=None, verbose=False, options=None, ift_options=None,
                 n_sub=6, max_extend=6) -> TwoStepResult:
    """Glatter's two-step method: IFT (step 1) then deconvolution (step 2).

    options : numerical options of step 2 (see DensityProfileFit.DEFAULT_OPTIONS);
    ift_options : of step 1 (see IFT_DEFAULTS); n_sub : sub-shells per shell in V.

    r_max / r_max_range / fix_r_max mean the same as for the direct fit; they
    are turned into D_max = 2 r_max for step 1. Step 2 uses r_max = D_max / 2.
    """
    def report(msg):
        if verbose and msg:
            print(msg)
        if progress is not None:
            progress(msg)

    if r_max is not None and fix_r_max:
        D_kw = dict(D_max=2 * float(r_max))
    elif r_max_range is not None:
        D_kw = dict(D_range=(2 * r_max_range[0], 2 * r_max_range[1]), n_D=n_D)
    elif r_max is not None:
        D_kw = dict(D_range=(1.4 * r_max, 3.0 * r_max), n_D=n_D)
    else:
        D_kw = dict(n_D=n_D)
    report("step 1: indirect Fourier transformation")
    pd = indirect_ft(q, I, sigma, geometry, n_r=n_r, resolution=resolution, background=background,
                     penalty=penalty, progress=report, ift_options=ift_options, **D_kw)
    report(pd.summary())
    report("step 2: deconvolution p(r) -> rho(r)")
    dec = DeconvolutionFit(pd.r, pd.p, cov=pd.cov, geometry=geometry, n=n, regularization=regularization,
                           prior=prior, outer=outer, basis=basis, grid=grid, errors=errors,
                           q_max=float(np.max(q)), n_sub=n_sub, options=options)
    dec.progress = progress
    prof = dec.fit(r_max=pd.D_max / 2, fix_r_max=True, n_draws=n_draws, seed=seed, verbose=verbose,
                   max_extend=max_extend)
    prof.r_max = (pd.D_max / 2, pd.D_max_err / 2)
    # back to I(q): the direct forward model with the deconvolved profile
    direct = DensityProfileFit(q, np.where(np.asarray(I) > 0, I, 1e-300), sigma, geometry=geometry, n=n,
                               basis=basis, grid=grid, resolution=resolution,
                               background_value=pd.background)
    I_rho = direct.model(prof.best.rho, prof.best.r_max)
    c2 = float(np.mean(((I_rho - np.asarray(I)) / np.asarray(sigma)) ** 2))
    return TwoStepResult(pddf=pd, profile=prof, I_from_profile=I_rho, chi2_red_profile=c2)
