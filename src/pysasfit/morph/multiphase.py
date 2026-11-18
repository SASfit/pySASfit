"""Multi-phase SASmorph: one Gaussian field cut at several levels, one or several contrasts.

Model
-----
A unit-variance Gaussian field psi(r) with correlation g(r) is cut at the levels
alpha_1 < ... < alpha_{N-1}; phase i occupies alpha_i < psi < alpha_{i+1}
(alpha_0 = -inf, alpha_N = +inf), so the phases are *ordered*: phase i touches only
phases i-1 and i+1, and a middle phase forms a layer (film, shell) between its
neighbours. The volume fractions fix the levels, alpha_k = Phi^-1(phi_0 + ... + phi_{k-1}).

With the scattering length density rho_i of phase i, rho(psi) = rho_0 + sum_k D_k [psi > alpha_k]
with the jumps D_k = rho_k - rho_{k-1}. By Plackett's identity for the bivariate normal
distribution the SLD correlation is

    C(r) = <drho(0) drho(r)> = sum_kl D_k D_l B(alpha_k, alpha_l; g(r)),
    B(a, b; g) = 1/(2 pi) integral_0^g exp(-(a^2 + b^2 - 2 t a b) / (2 (1 - t^2))) / sqrt(1 - t^2) dt

(Berk 1991; Roberts 1997 for the two-cut phase; Teubner 1991). One cut gives the
two-phase relation of :mod:`pysasfit.morph.correlation`. C(0) = sum phi_i rho_i^2 - (sum phi_i rho_i)^2.

Data
----
Every curve m (a contrast: SANS contrast variation, SAXS + SANS, ...) has its own SLD set and
gives gamma_m(r) = C_m(r)/C_m(0). The common g(r) minimises, at each r,

    sum_m w_m (gamma_m(r) - G_m(g))^2,   G_m(g) = C_m(g) / C_m(0),

found on a fine g grid (robust also where G_m is not monotonic for g < 0). With several
contrasts the volume fractions can be fitted as well (``fit_fractions``): the profile misfit
over r is minimised. With SLDs in absolute units (1e-6 / A^2 = 1e10 cm^-2) and data on
absolute scale, the invariants Q_m = 2 pi^2 C_m(0) are a check of fractions, SLDs and
calibration, and can enter the fraction fit.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from dataclasses import field as dc_field

import numpy as np
from scipy.integrate import cumulative_simpson
from scipy.special import ndtr, ndtri

from .correlation import spectral_function, wavevector_distribution
from .extrapolation import ExtrapolatedCurve, extrapolate
from .grf import field_cosine, field_fft
from .scattering import levelcut_intensity, voxel_intensity
from .statistics import phase_connectivity

__all__ = ["DataSet", "PhaseSystem", "MultiCut", "MultiParameters", "CurveResult", "MultiResult",
           "correlations_multi", "reconstruct_multi", "fit_fractions", "joint_g", "interface_areas",
           "multiphase_statistics", "Q_FACTOR"]

#: Q [I unit * q unit^-3] = Q_FACTOR[q unit] * 2 pi^2 C(0) [ (1e-6/A^2)^2 ], I in 1/cm
Q_FACTOR = {"A": 1e-4, "nm": 1e-1}


def _weights2d(weights, M, nr):
    """Weights per curve (M,) or per curve and r (M, nr) -> (M, nr)."""
    if weights is None:
        return np.ones((M, nr))
    w = np.asarray(weights, float)
    return np.broadcast_to(w[:, None], (M, nr)) if w.ndim == 1 else w


def _wrms(x, w):
    """rms of x over the points with w > 0."""
    m = w > 0
    return float(np.sqrt(np.mean(x[m] ** 2))) if np.any(m) else 0.0


@dataclass
class DataSet:
    """One measured curve (background subtracted) and the SLDs of all phases for it."""

    q: np.ndarray
    I: np.ndarray
    sigma: np.ndarray | None
    sld: np.ndarray                  # one value per phase
    name: str = "data"
    path: str | None = None
    weight: float = 1.0
    fit_background: bool | None = None   # None: as MultiParameters.fit_background
    background: float | None = None      # fixed flat background (when not fitted), subtracted
    kind: str = "sas"                    # "sas" (q, I, sigma) or "sesans" (q = delta, I = G - G(0), sigma = dG)
    meta: dict = dc_field(default_factory=dict)   # SESANS: wavelength, thickness, P, dP, length unit ...

    @property
    def is_sesans(self):
        return self.kind == "sesans"

@dataclass
class PhaseSystem:
    """Ordered phases (order along psi) with volume fractions."""

    names: list
    fractions: np.ndarray

    def __post_init__(self):
        self.fractions = np.asarray(self.fractions, float)
        if len(self.names) != self.fractions.size or self.fractions.size < 2:
            raise ValueError("need at least two phases, one fraction per phase")
        if np.any(self.fractions <= 0):
            raise ValueError("all volume fractions must be > 0")
        s = self.fractions.sum()
        if abs(s - 1) > 1e-6:
            raise ValueError(f"volume fractions add up to {s:.6g}, not 1")

    @property
    def levels(self):
        """alpha_1 ... alpha_{N-1}."""
        return ndtri(np.cumsum(self.fractions)[:-1])

    def __len__(self):
        return self.fractions.size


class MultiCut:
    """Tables of B(alpha_k, alpha_l; g) on g = sin(theta) and the model functions G_m(g)."""

    def __init__(self, phases: PhaseSystem, n=4001):
        self.phases = phases
        self.alpha = phases.levels
        th = np.linspace(-0.5 * np.pi, 0.5 * np.pi, n)
        t = np.sin(th)
        K = self.alpha.size
        B = np.empty((K, K, n))
        with np.errstate(divide="ignore", over="ignore", under="ignore", invalid="ignore"):
            for k in range(K):
                for l in range(k, K):
                    a, b = self.alpha[k], self.alpha[l]
                    # the integrand in theta: dt / sqrt(1 - t^2) = d theta
                    f = np.exp(-(a * a + b * b - 2 * t * a * b) / (2 * (1 - t * t)))
                    f = np.where(np.isfinite(f), f, 0.0)
                    # limits at t = +-1 (delta functions of width 0): a = b at t = 1, a = -b at t = -1
                    # limits t -> 1 (a = b) and t -> -1 (a = -b); the integrand is smooth in theta
                    f[-1] = np.exp(-a * a / 2) if np.isclose(a, b) else 0.0
                    f[0] = np.exp(-a * a / 2) if np.isclose(a, -b) else 0.0
                    F = cumulative_simpson(f, x=th, initial=0.0) / (2 * np.pi)
                    F -= np.interp(0.0, th, F)
                    B[k, l] = B[l, k] = F
        # exact end values: B(a,b;1) = Phi2 upper-orthant cov = min(Pa,Pb) - Pa Pb (P = P(psi > a))
        P = 1 - ndtr(self.alpha)
        Bp = np.minimum.outer(P, P) - np.outer(P, P)
        Bm = np.maximum(0.0, np.add.outer(P, P) - 1) - np.outer(P, P)
        self.end_error = 0.0
        # remove the (small) quadrature error at the ends by a linear blend in theta
        for k in range(K):
            for l in range(K):
                e1, e0 = Bp[k, l] - B[k, l, -1], Bm[k, l] - B[k, l, 0]
                self.end_error = max(self.end_error, abs(e0), abs(e1))
                w = (th - th[0]) / (th[-1] - th[0])
                B[k, l] += e0 * (1 - w) + e1 * w
        self._th, self.t, self.B = th, t, B

    def covariance(self, sld, g=None):
        """C(g) (on the table grid, or at the given g) for the SLD set ``sld`` of the phases."""
        D = np.diff(np.asarray(sld, float))
        C = np.einsum("k,l,klm->m", D, D, self.B)
        if g is None:
            return C
        return np.interp(np.arcsin(np.clip(g, -1, 1)), self._th, C)

    def variance(self, sld):
        phi = self.phases.fractions
        sld = np.asarray(sld, float)
        return float(phi @ sld ** 2 - (phi @ sld) ** 2)

    def G(self, sld, g=None):
        v = self.variance(sld)
        if v <= 0:
            raise ValueError("all phases have the same SLD in one contrast: no scattering")
        return self.covariance(sld, g) / v

    def g_from_gammas(self, gammas, slds, weights=None):
        """Common g(r) for several normalised correlation functions (rows of ``gammas``).

        Returns (g, misfit per curve: rms of gamma_m - G_m(g)).
        """
        gammas = np.atleast_2d(gammas)
        M, nr = gammas.shape
        w = _weights2d(weights, M, nr)
        Gt = np.array([self.G(s) for s in slds])                  # (M, n_table)
        best = np.full(nr, np.inf)
        idx = np.zeros(nr, int)
        chunk = max(1, 4_000_000 // Gt.shape[1])
        for i0 in range(0, nr, chunk):
            sl = slice(i0, i0 + chunk)
            obj = np.zeros((min(chunk, nr - i0), Gt.shape[1]))
            for m in range(M):
                obj += w[m, sl, None] * (gammas[m, sl, None] - Gt[m][None, :]) ** 2
            j = np.argmin(obj, axis=1)
            idx[sl] = j
            best[sl] = obj[np.arange(obj.shape[0]), j]
        # parabolic refinement in theta
        j = np.clip(idx, 1, Gt.shape[1] - 2)
        th = self._th

        def objective(jj):
            return sum(w[m] * (gammas[m] - Gt[m][jj]) ** 2 for m in range(M))

        f0, f1, f2 = objective(j - 1), objective(j), objective(j + 1)
        den = f0 - 2 * f1 + f2
        shift = np.where(den > 0, 0.5 * (f0 - f2) / np.where(den > 0, den, 1), 0.0)
        theta = th[j] + np.clip(shift, -1, 1) * (th[1] - th[0])
        theta = np.where((idx == 0) | (idx == Gt.shape[1] - 1), th[idx], theta)
        g = np.sin(theta)
        misfit = np.array([_wrms(gammas[m] - self.G(slds[m], g), w[m]) for m in range(M)])
        return g, misfit


@dataclass
class MultiParameters:
    """Settings of a multi-phase reconstruction (lengths in the inverse unit of q)."""

    names: list = dc_field(default_factory=lambda: ["solid", "pore"])
    fractions: list = dc_field(default_factory=lambda: [0.5, 0.5])
    box_size: float | None = None
    n: int = 128
    qmin: float | None = None
    qmax: float | None = None
    low_q: str = "guinier"
    low_q_points: int = 10
    porod_points: int | None = None
    porod_qmin: float | None = None
    porod_auto: bool = True          # neither porod_points nor porod_qmin: range chosen from the data
    fit_background: bool = False
    refine: int = 4
    method: str = "fft"
    n_waves: int = 4000
    threshold: str = "quantile"
    seed: int | None = None
    nr: int | None = None
    nk: int = 1000
    fit_fractions: bool = False       # fit the volume fractions to the contrast series
    renormalise: bool = True          # contrast series: per-curve invariant correction in the joint fit
    common_scale: bool = True         # all curves on one intensity scale (contrast series): invariant ratios count
    sld_absolute: bool = False        # SLDs in 1e-6/A^2 (= 1e10 cm^-2) and I in 1/cm: invariant check
    q_unit: str = "A"                 # A or nm (absolute invariant only)

    def to_dict(self):
        d = asdict(self)
        d["fractions"] = [float(x) for x in self.fractions]
        return d

    @classmethod
    def from_dict(cls, d):
        names = cls.__dataclass_fields__
        return cls(**{k: v for k, v in d.items() if k in names})

    @property
    def phases(self):
        return PhaseSystem(list(self.names), np.asarray(self.fractions, float))


@dataclass
class CurveResult:
    """Result for one curve. For a SESANS curve: ``curve`` is None, ``q_data`` are the spin-echo lengths,
    ``I_data``/``I_model`` are G(delta) - G(0), ``scale`` multiplies the projected SLD covariance,
    ``inversion`` is the Abel inversion (:class:`pysasfit.morph.sesans.SESANSInversion`)."""
    data: DataSet
    curve: ExtrapolatedCurve | None
    invariant: float
    gamma: np.ndarray
    G: np.ndarray                     # model gamma with the common g
    misfit: float
    variance: float                   # C(0) in SLD units^2
    scale: float                      # Q / (2 pi^2 C(0)): data units per SLD unit^2
    q_data: np.ndarray
    I_data: np.ndarray
    sigma_data: np.ndarray | None
    I_model: np.ndarray
    chi2: float = np.nan
    q_voxel: np.ndarray | None = None
    I_voxel: np.ndarray | None = None
    Q_model_abs: float = np.nan       # 2 pi^2 C(0) in data units (sld_absolute)
    norm: float = 1.0                 # lambda: gamma rescaled by it in the joint fit (invariant / lambda)
    kind: str = "sas"
    inversion: object = None          # SESANS: Abel inversion of G(delta)
    r_range: tuple = (0.0, np.inf)    # r range in which gamma enters the joint fit


@dataclass
class MultiResult:
    params: MultiParameters
    multicut: MultiCut
    r: np.ndarray
    g: np.ndarray
    k: np.ndarray
    f: np.ndarray
    Pk: np.ndarray
    curves: list
    labels: np.ndarray | None = None      # int8 (n, n, n), phase index
    psi: np.ndarray | None = None
    levels: np.ndarray | None = None      # cut levels actually used (empirical or alpha)
    statistics: dict = dc_field(default_factory=dict)
    warnings: list = dc_field(default_factory=list)
    timings: dict = dc_field(default_factory=dict)
    fit_info: dict = dc_field(default_factory=dict)
    tail_info: dict = dc_field(default_factory=dict)   # SESANS: g(r) set to 0 beyond R

    @property
    def voxel_size(self):
        return self.params.box_size / self.params.n

    @property
    def names(self):
        return list(self.params.names)

    def report(self):
        p = self.params
        ph = self.multicut.phases
        lines = ["SASmorph multi-phase reconstruction",
                 "  phases (order along psi): " + " | ".join(
                     f"{n} {f:.4g}" for n, f in zip(ph.names, ph.fractions)),
                 "  levels alpha           " + ", ".join(f"{a:.4f}" for a in ph.levels)]
        if self.fit_info:
            lines.append(f"  fractions fitted       {self.fit_info.get('message', '')}, "
                         f"misfit {self.fit_info.get('start', np.nan):.4g} -> {self.fit_info.get('end', np.nan):.4g}")
        lines.append(f"  box                    L = {p.box_size:.5g}, n = {p.n}, voxel a = {self.voxel_size:.4g}, "
                     f"method = {p.method}")
        for c in self.curves:
            if c.kind == "sesans":
                lines.append(f"  [{c.data.name}]  SESANS, SLD " + ", ".join(f"{s:.4g}" for s in c.data.sld)
                             + f"; delta = {c.q_data[0]:.4g} ... {c.q_data[-1]:.4g}, gamma used for r = "
                             f"{c.r_range[0]:.4g} ... {c.r_range[1]:.4g}; normalisation x {1 / c.norm:.3f}; "
                             f"rms(gamma - model) = {c.misfit:.3g}; chi2_red = {c.chi2:.4g}")
                lines.append(f"      scale {c.scale:.5g} (data units per SLD unit^2)"
                             + (f", absolute check: scale / expected = {c.scale / c.Q_model_abs:.4g}"
                                if p.sld_absolute and np.isfinite(c.Q_model_abs) else ""))
                continue
            lines.append(f"  [{c.data.name}]  SLD " + ", ".join(f"{s:.4g}" for s in c.data.sld)
                         + f"; Q = {c.invariant:.5g}" + (f" (x {1 / c.norm:.3f})" if c.norm != 1 else "")
                         + f", K = {c.curve.K:.4g} (Porod q = {c.curve.porod_range[0]:.4g} ... {c.curve.porod_range[1]:.4g}, "
                         f"{c.curve.porod_mode}); rms(gamma - model) = {c.misfit:.3g}; "
                         f"chi2_red = {c.chi2:.4g}")
            cv = c.curve
            if cv.B_mode == "fitted":
                lines.append(f"      background B = {cv.B:.5g} +- {cv.dB:.2g} (fitted on q = {cv.porod_range[0]:.4g} ... "
                             f"{cv.porod_range[1]:.4g})")
            elif cv.B_mode == "fixed":
                lines.append(f"      background B = {cv.B:.5g} (fixed)")
            if p.sld_absolute and np.isfinite(c.Q_model_abs):
                lines.append(f"      absolute check: Q_data / Q_model = {c.invariant / c.Q_model_abs:.4g}")
        st = self.statistics
        if st:
            for i, n in enumerate(ph.names):
                s = st["phases"][i]
                lines.append(f"  {n:<12} voxels {s['volume_fraction']:.4f}, clusters {s['clusters']}, largest "
                             f"{100 * s['largest_fraction']:.1f} %, percolates x/y/z "
                             + "".join("+" if v else "-" for v in s["percolates"]))
            for (i, j), sv in st["interfaces"].items():
                lines.append(f"  S/V {ph.names[i]}|{ph.names[j]:<12} {sv:.5g}")
        if self.timings:
            lines.append("  timings                " + ", ".join(f"{k} {v:.2f} s" for k, v in self.timings.items()))
        if self.warnings:
            lines.append("Warnings:")
            lines += [f"  - {w}" for w in self.warnings]
        return "\n".join(lines)


def _r_grid(p, q_max, r_extra=0.0):
    a = p.box_size / p.n
    rmax = max(np.sqrt(3) * 0.5 * p.box_size * 1.02, 2 * a, r_extra)
    dr = min(a / 4.0, np.pi / (16.0 * q_max))      # q_max r steps of pi/16: accurate model I(q) at q_max
    nr = p.nr or int(np.clip(np.ceil(rmax / dr), 200, 40000 if r_extra else 20000))
    return np.linspace(0.0, rmax, nr + 1)


def _suggest_box(curves, n):
    Ls = []
    for c in curves:
        m = (c.q >= c.q_data[0]) & (c.q <= c.q_data[1])
        qs = c.q[m][np.argmax(c.q[m] ** 2 * c.I[m])]
        Ls.append(min(5 * 2 * np.pi / qs, n * 2 * np.pi / c.q_data[1]))
    return float(max(Ls))


def joint_g(mc, gammas, slds, weights=None, renormalise=True, iterations=8, bounds=(0.8, 1.25), free=None):
    """Common g(r) of several curves and, for two or more curves, a normalisation factor lambda_m per
    curve (gamma_m -> lambda_m gamma_m): the invariant of a curve whose Porod regime is not reached is
    a few per cent off, and that error would bias the common g. ``free`` (bool per curve) marks curves whose
    normalisation is only roughly known (SESANS: C(0) of the Abel inversion); their lambda may lie between
    0.05 and 20. Weights per curve or per curve and r. Returns (g, misfit, lambdas)."""
    gammas = np.atleast_2d(gammas)
    M, nr = gammas.shape
    w = _weights2d(weights, M, nr)
    lam = np.ones(M)
    g, mis = mc.g_from_gammas(gammas, slds, w)
    if not renormalise or M < 2:
        return g, mis, lam
    free = np.zeros(M, bool) if free is None else np.asarray(free, bool)
    lo = np.where(free, 0.05, bounds[0])
    hi = np.where(free, 20.0, bounds[1])
    for _ in range(iterations):
        G = np.array([mc.G(s, g) for s in slds])
        den = np.sum(w * gammas * gammas, axis=1)
        new = np.where(den > 0, np.clip(np.sum(w * G * gammas, axis=1) / np.where(den > 0, den, 1.0), lo, hi),
                       1.0)                    # a curve without weight keeps its normalisation
        if np.allclose(new, lam, rtol=1e-5):
            break
        lam = new
        g, mis = mc.g_from_gammas(gammas * lam[:, None], slds, w)
    return g, mis, lam


def _profile_misfit(phases, gammas, slds, weights, renormalise=True, free=None):  # noqa: D103
    mc = MultiCut(phases, n=1201)
    _, mis, _ = joint_g(mc, gammas, slds, weights, renormalise, iterations=4, free=free)
    wc = np.asarray(weights, float)
    wc = wc.max(axis=1) if wc.ndim == 2 else wc
    return float(np.sum(wc * mis ** 2)), mc


#: weight of the invariant term in the fraction fit: a 10 % invariant mismatch counts as an rms
#: gamma misfit of 0.01
INVARIANT_WEIGHT = 1e-2


def fit_fractions(phases, gammas, slds, weights=None, invariants=None, q_unit="A", absolute=False,
                  common_scale=True, renormalise=True, free=None):
    """Volume fractions minimising the joint misfit of the gamma_m and of the invariants.

    The shapes gamma_m alone determine the fractions only weakly. The invariants Q_m = s 2 pi^2 C_m(0)
    add the information: with ``common_scale`` (all curves on one intensity scale) their ratios, with
    ``absolute`` (SLDs in 1e-6/A^2, I in 1/cm) their values (s known). Returns (PhaseSystem, info)."""
    M = len(slds)
    from scipy.optimize import minimize

    # the fit only needs the correlated range of r, and not every point of it
    gammas = np.atleast_2d(gammas)
    w2 = _weights2d(weights, M, gammas.shape[1])
    live = np.flatnonzero(np.max(np.abs(gammas), axis=0) > 5e-3)
    last = live[-1] + 1 if live.size else gammas.shape[1]
    step = max(1, last // 500)
    gammas = gammas[:, :last:step]
    w2 = np.ascontiguousarray(w2[:, :last:step])
    w = w2.max(axis=1)
    inv = None if invariants is None else np.asarray(invariants, float)
    has_q = np.ones(M, bool) if inv is None else np.isfinite(inv)

    def unpack(x):
        e = np.exp(np.concatenate([[0.0], x]))
        return e / e.sum()

    def obj(x):
        fr = unpack(x)
        ps = PhaseSystem(phases.names, fr)
        val, mc = _profile_misfit(ps, gammas, slds, w2, renormalise, free)
        k = np.flatnonzero(has_q)                         # SESANS curves have no invariant
        if inv is not None and (absolute or (common_scale and k.size > 1)):
            model = np.array([2 * np.pi ** 2 * mc.variance(slds[i]) for i in k])
            lr = np.log(inv[k] / model)
            lr = lr - (np.log(Q_FACTOR[q_unit]) if absolute else np.average(lr, weights=w[k]))
            val += INVARIANT_WEIGHT * float(np.sum(w[k] * lr ** 2))
        return val

    x0 = np.log(phases.fractions[1:] / phases.fractions[0])
    start = obj(x0)
    res = minimize(obj, x0, method="Nelder-Mead", options={"xatol": 1e-4, "fatol": 1e-10, "maxiter": 400})
    fr = unpack(res.x)
    return PhaseSystem(phases.names, fr), {"start": start, "end": float(res.fun), "message":
                                           "converged" if res.success else "not converged", "nfev": res.nfev}


def _ramp(r, a, b):
    """0 below a, 1 above b, cosine in between."""
    x = np.clip((r - a) / max(b - a, 1e-30), 0.0, 1.0)
    return 0.5 * (1 - np.cos(np.pi * x))


def _taper(r, R):
    """1 up to R, cosine to 0 at 1.25 R."""
    x = np.clip((r - R) / (0.25 * R), 0.0, 1.0)
    return 0.5 * (1 + np.cos(np.pi * x))


def _model_low_q(c, r, cov, n_match=8, nodes=80, R=None):
    """A copy of the extrapolated curve ``c`` whose part below the first data point is the level-cut model
    of the current g(r) (covariance ``cov`` on r), scaled to the first ``n_match`` data nodes: the low-q
    extrapolation then carries the large-scale structure that SESANS has measured."""
    from dataclasses import replace

    q0 = c.q_data[0]
    data = c.q >= q0
    qd, Id = c.q[data], c.I[data]
    k = min(n_match, qd.size)
    M = levelcut_intensity(qd[:k], r, cov, 1.0)
    s = float(np.sum(M * Id[:k]) / max(np.sum(M * M), 1e-300))
    q_lo = np.concatenate([[0.0], np.geomspace(min(0.5 / (R or r[-1]), 0.1 * q0), q0, nodes)[:-1]])
    I_lo = s * levelcut_intensity(q_lo, r, cov, 1.0)
    return replace(c, q=np.concatenate([q_lo, qd]), I=np.concatenate([I_lo, Id]), low_q="model (SESANS)",
                   low_q_info={"scale": s})


def _correlation_range(inv, floor=2e-3, nsig=2.0):
    """Largest r up to which the Abel-inverted gamma of a SESANS curve differs from 0 by more than its
    uncertainty (``nsig`` sigma + ``floor``): beyond, g(r) is set smoothly to 0 (no information, and the
    r^2-weighted volume of a noisy tail would dominate the model intensity at low q)."""
    gam = inv.C / inv.C0
    sig = nsig * inv.dC / abs(inv.C0) + floor
    big = np.flatnonzero(np.abs(gam) > sig)
    return float(inv.r[big[-1] + 1]) if big.size and big[-1] + 1 < inv.r.size else float(inv.r[-1])


def sesans_sign(delta, G):
    """+1 if a SESANS curve is given with the opposite sign convention (G(0) - G(delta) or -ln P, positive),
    -1 for the usual G(delta) - G(0) <= 0: the sign of the median of the larger spin-echo half."""
    d = np.asarray(delta, float)
    G = np.asarray(G, float)
    m = np.isfinite(d) & np.isfinite(G)
    if not np.any(m):
        return -1
    o = np.argsort(d[m])
    tail = G[m][o][o.size // 2:]
    return 1 if np.median(tail) > 0 else -1


def _suggest_box_sesans(inversions, n):
    """Box for SESANS-only data: it should hold twice the largest spin-echo length (the voxel SESANS curve is
    computed up to L/2) and resolve the correlation length xi (C(xi) = C(0)/e) with about 4 voxels; if both
    cannot be met with n voxels, the resolution wins."""
    dmax = max(float(np.max(i.delta)) for i in inversions)
    L = 2.2 * dmax
    xis = []
    for inv in inversions:
        if inv.C0 > 0:
            g1 = np.flatnonzero(inv.C < inv.C0 / np.e)
            if g1.size and inv.r[g1[0]] > 0:
                xis.append(float(inv.r[g1[0]]))
    if xis:
        L = max(L, 8 * max(xis))
        L = min(L, n * min(xis) / 4)
    return float(L)


def correlations_multi(datasets, params=None):
    """Steps 1-4 for one or several curves (scattering and SESANS): extrapolation or Abel inversion,
    gamma_m(r), common g(r), f(k), model I_m(q) and G_m(delta)."""
    from .sesans import invert_sesans, projection_matrix

    p = MultiParameters() if params is None else params
    t0 = time.perf_counter()
    w = []
    phases = p.phases
    for d in datasets:
        if len(d.sld) != len(phases):
            raise ValueError(f"{d.name}: {len(d.sld)} SLD values for {len(phases)} phases")
    if not datasets:
        raise ValueError("no data")
    is_ses = [d.is_sesans for d in datasets]
    curves, inversions = [], []
    for d, ses in zip(datasets, is_ses):
        if ses:
            curves.append(None)
            G = np.asarray(d.I, float)
            if sesans_sign(d.q, G) > 0:
                G = -G
                w.append(f"{d.name}: SESANS data are positive; G(delta) - G(0) <= 0 for any structure, so the "
                         "sign was reversed (file convention G(0) - G(delta) or -ln P)")
            inversions.append(invert_sesans(d.q, G, d.sigma))
        else:
            curves.append(extrapolate(d.q, d.I, d.sigma, qmin=p.qmin, qmax=p.qmax, low_q=p.low_q,
                                      low_q_points=p.low_q_points, porod_points=p.porod_points,
                                      porod_qmin=p.porod_qmin,
                                      fit_background=p.fit_background if d.fit_background is None else
                                      d.fit_background, background=d.background, refine=p.refine,
                                      porod_auto=p.porod_auto))
            inversions.append(None)
    sas_curves = [c for c in curves if c is not None]
    for d, c in zip(datasets, curves):
        if c is not None and c.porod_info.get("message"):
            w.append(f"{d.name}: Porod range: {c.porod_info['message']}")
    for d, inv in zip(datasets, inversions):
        if inv is not None and not inv.saturated:
            w.append(f"{d.name}: G(delta) has not levelled off at the largest spin-echo length: structure larger "
                     "than the delta range, C(r) beyond it is not determined")
    if p.box_size is None:
        if sas_curves:
            p.box_size = _suggest_box(sas_curves, p.n)
        else:                      # SESANS only: a few correlation lengths of the inverted C(r)
            p.box_size = _suggest_box_sesans([i for i in inversions if i is not None], p.n)
        w.append(f"box size not given: using L = {p.box_size:.4g}")
    if not (p.box_size and np.isfinite(p.box_size) and p.box_size > 0):
        raise ValueError(f"box size must be > 0 (got {p.box_size})")
    ses_delta = [d.q for d in datasets if d.is_sesans]
    if sas_curves:
        qmax = max(c.q_data[1] for c in sas_curves)
        qmin = min(c.q_data[0] for c in sas_curves)
    else:
        qmax = 2 * np.pi / max(1e-30, min(np.min(dl[dl > 0]) for dl in ses_delta))
        qmin = 2 * np.pi / max(np.max(dl) for dl in ses_delta)
    r_extra = 1.5 * max((np.max(dl) for dl in ses_delta), default=0.0)
    r = _r_grid(p, qmax, r_extra)
    M = len(datasets)
    gammas = np.zeros((M, r.size))
    W = np.zeros((M, r.size))
    r_ranges = []
    R_sas = max((np.pi / c.q_data[0] for c in sas_curves), default=0.0)
    for m, (d, c, inv) in enumerate(zip(datasets, curves, inversions)):
        if inv is not None:
            gammas[m] = inv.gamma(r)
            lo, hi = float(inv.delta[0]), float(inv.delta[-1])
            # with scattering curves, a SESANS curve enters through their low-q extension (below), not through
            # its own gamma: a pointwise compromise between curves is no transform pair of either, and the
            # intensity at high q is very sensitive to that
            W[m] = 0.0 if sas_curves else d.weight * (1.0 - _ramp(r, hi, 1.25 * hi))
        else:
            gammas[m] = c.gamma(r)
            lo, hi = 0.0, np.inf
            W[m] = d.weight
        r_ranges.append((lo, hi))
    # every r needs some weight (else g there is arbitrary): SESANS only, below delta_min and beyond 1.25
    # delta_max the inversion (extrapolated, resp. zero) is the only information
    none = W.sum(axis=0) <= 0
    if np.any(none):
        W[:, none] = np.array([d.weight if d.is_sesans else 0.0 for d in datasets])[:, None]
    slds = [np.asarray(d.sld, float) for d in datasets]
    Qs = np.array([np.nan if c is None else c.invariant for c in curves])
    free = np.array(is_ses, bool)
    fit_info = {}
    if p.fit_fractions:
        nsas = len(sas_curves)
        if M < 2 and not p.sld_absolute:
            w.append("fitting fractions needs a contrast series on a common scale or absolute data; "
                     "fractions kept")
        else:
            if nsas == 0:
                w.append("fractions from SESANS only rest on the shapes of gamma(r): weakly determined")
            phases, fit_info = fit_fractions(phases, gammas, slds, W, Qs, p.q_unit, p.sld_absolute,
                                             p.common_scale, p.renormalise, free)
            p.fractions = [float(x) for x in phases.fractions]
    mc = MultiCut(phases)
    tail_info = {}
    if ses_delta and sas_curves:
        # The scattering curves are extended below q_min by the level-cut model of the g(r) of the SESANS curves
        # alone (for the contrast of each curve, matched to its first points) instead of a Guinier/constant
        # extrapolation, which misses the large structures SESANS sees. g(r) then follows from the extended
        # scattering curves; the SESANS curves are compared with it (chi2).
        ks = [m for m in range(M) if is_ses[m]]
        Ws = np.array([np.full(r.size, datasets[m].weight) for m in ks])
        g_ses, _mis, _lam = joint_g(mc, gammas[ks], [slds[m] for m in ks], Ws, p.renormalise)
        R = max(max(_correlation_range(inv) for inv in inversions if inv is not None), R_sas)
        g_ses = g_ses * _taper(r, R)
        for m, c in enumerate(curves):
            if c is not None:
                curves[m] = _model_low_q(c, r, mc.covariance(slds[m], g_ses), R=R)
                gammas[m] = curves[m].gamma(r)
        tail_info = {"R": R, "low_q": "model of the SESANS g(r)"}
    g, misfit, lam = joint_g(mc, gammas, slds, W, p.renormalise, free=free)
    for m, inv in enumerate(inversions):     # SESANS: misfit of its gamma on the spin-echo range
        if inv is not None:
            rng = (r >= inv.delta[0]) & (r <= inv.delta[-1])
            misfit[m] = _wrms(gammas[m] * lam[m] - mc.G(slds[m], g), rng.astype(float))
    k = np.geomspace(qmin / 10, qmax * 10, p.nk)
    f = spectral_function(r, g, k)
    Pk = wavevector_distribution(k, f)
    results = []
    for m, (d, c, inv) in enumerate(zip(datasets, curves, inversions)):
        lm, mis, gam = lam[m], misfit[m], gammas[m]
        var = mc.variance(d.sld)
        cov = mc.covariance(d.sld, g)
        if inv is not None:
            A = projection_matrix(inv.delta, r)
            model = A @ cov
            sw = 1.0 / inv.dG
            scale = float(np.sum(sw ** 2 * model * inv.G) / max(np.sum((sw * model) ** 2), 1e-300))
            G_model = scale * model
            chi2 = float(np.mean(((G_model - inv.G) / inv.dG) ** 2))
            results.append(CurveResult(data=d, curve=None, invariant=np.nan, gamma=gam * lm, G=mc.G(d.sld, g),
                                       misfit=float(mis), norm=float(lm), variance=var, scale=scale,
                                       q_data=inv.delta, I_data=inv.G, sigma_data=inv.dG, I_model=G_model,
                                       chi2=chi2, kind="sesans", inversion=inv, r_range=r_ranges[m],
                                       Q_model_abs=Q_FACTOR[p.q_unit] if p.sld_absolute else np.nan))
            if mis > 0.05:
                w.append(f"{d.name}: SESANS gamma(r) differs from the model by {mis:.3f} rms")
            continue
        Q = c.invariant
        scale = Q / (lm * 2 * np.pi ** 2 * var)
        qq = np.asarray(d.q, float)
        msk = np.isfinite(qq) & np.isfinite(d.I) & (qq >= c.q_data[0]) & (qq <= c.q_data[1])
        o = np.argsort(qq[msk])
        q_data, I_data = qq[msk][o], (np.asarray(d.I, float)[msk] - c.B)[o]
        s_data = None if d.sigma is None else np.asarray(d.sigma, float)[msk][o]
        I_model = levelcut_intensity(q_data, r, cov, scale)
        chi2 = np.nan
        if s_data is not None and np.all(s_data > 0):
            chi2 = float(np.mean(((I_model - I_data) / s_data) ** 2))
        if abs(lm - 1) > 0.1:
            w.append(f"{d.name}: invariant corrected by {100 * (1 / lm - 1):+.0f} % to fit the common g(r): "
                     "check its extrapolation (Porod range) and SLDs")
        results.append(CurveResult(data=d, curve=c, invariant=Q, gamma=gam * lm, G=mc.G(d.sld, g), misfit=float(mis),
                                   norm=float(lm),
                                   variance=var, scale=scale, q_data=q_data, I_data=I_data, sigma_data=s_data,
                                   I_model=I_model, chi2=chi2, r_range=r_ranges[m],
                                   Q_model_abs=Q_FACTOR[p.q_unit] * 2 * np.pi ** 2 * var if p.sld_absolute
                                   else np.nan))
        if mis > 0.05:
            w.append(f"{d.name}: gamma(r) differs from the model by {mis:.3f} rms: fractions, SLDs or phase "
                     "order may not match")
        if c.K <= 0:
            w.append(f"{d.name}: Porod constant K <= 0 (no q^-4 regime)")
    a = p.box_size / p.n
    if np.pi / a < qmax and sas_curves:
        w.append(f"voxel size a = {a:.4g} does not resolve q_max (pi/a = {np.pi / a:.4g} < {qmax:.4g})")
    if ses_delta and max(np.max(dl) for dl in ses_delta) > 0.5 * p.box_size:
        w.append("the box is smaller than twice the largest spin-echo length: the SESANS curve of the voxel "
                 "structure is only computed up to L/2")
    res = MultiResult(params=p, multicut=mc, r=r, g=g, k=k, f=f, Pk=Pk, curves=results, warnings=w,
                      fit_info=fit_info, tail_info=tail_info)
    res.timings["correlations"] = time.perf_counter() - t0
    return res


def interface_areas(labels, a, periodic=True):
    """{(i, j): S_ij / V} from voxel faces between phases i < j (times 2/3, isotropic interfaces)."""
    out = {}
    n = int(labels.max()) + 1
    V = labels.size * a ** 3
    for ax in range(3):
        if periodic:
            x, y = labels, np.roll(labels, 1, axis=ax)
        else:
            sl0 = [slice(None)] * 3
            sl1 = [slice(None)] * 3
            sl0[ax], sl1[ax] = slice(1, None), slice(None, -1)
            x, y = labels[tuple(sl0)], labels[tuple(sl1)]
        diff = x != y
        lo = np.minimum(x[diff], y[diff]).astype(np.int64)
        hi = np.maximum(x[diff], y[diff]).astype(np.int64)
        cnt = np.bincount(lo * n + hi, minlength=n * n)
        for i in range(n):
            for j in range(i + 1, n):
                if cnt[i * n + j]:
                    out[(i, j)] = out.get((i, j), 0.0) + cnt[i * n + j]
    return {k: (2.0 / 3.0) * v * a * a / V for k, v in sorted(out.items())}


def multiphase_statistics(labels, a, nphases, periodic=True):
    return {"phases": [phase_connectivity(labels == i) for i in range(nphases)],
            "interfaces": interface_areas(labels, a, periodic)}


def reconstruct_multi(datasets, params=None, keep_field=True, statistics=True, progress=None):
    p = MultiParameters() if params is None else params

    def tell(x, t):
        if progress is not None:
            progress(x, t)

    tell(0.0, "correlation functions")
    res = correlations_multi(datasets, p)
    a = p.box_size / p.n
    tell(0.2, f"Gaussian random field ({p.method}, {p.n}^3)")
    t0 = time.perf_counter()
    if p.method == "fft":
        psi, info = field_fft(res.r, res.g, p.n, a, rng=p.seed)
        if info["negative_power"] > 0.01:
            res.warnings.append(f"{100 * info['negative_power']:.1f} % negative spectral power discarded")
    elif p.method == "cosine":
        psi, info = field_cosine(res.r, res.g, p.n, a, n_waves=p.n_waves, rng=p.seed)
    else:
        raise ValueError(f"unknown method {p.method!r}")
    res.timings["field"] = time.perf_counter() - t0
    fr = res.multicut.phases.fractions
    if p.threshold == "quantile":
        levels = np.quantile(psi, np.cumsum(fr)[:-1])
    else:
        levels = res.multicut.alpha.copy()
    res.levels = np.asarray(levels, float)
    res.labels = np.digitize(psi, res.levels).astype(np.int8)
    if keep_field:
        res.psi = psi
    periodic = p.method == "fft"
    tell(0.7, "scattering of the structure")
    t0 = time.perf_counter()
    for c in res.curves:
        sld_map = np.asarray(c.data.sld, float)[res.labels]
        if c.kind == "sesans":
            from .sesans import projection_matrix, voxel_correlation
            rv, Cv = voxel_correlation(sld_map, a)
            dl = c.q_data[c.q_data <= rv[-1]]
            c.q_voxel = dl
            c.I_voxel = c.scale * (projection_matrix(dl, rv) @ Cv) if dl.size else dl
            continue
        c.q_voxel, c.I_voxel, _ = voxel_intensity(sld_map, a, c.scale, periodic=periodic)
    res.timings["scattering"] = time.perf_counter() - t0
    if statistics:
        tell(0.85, "statistics")
        t0 = time.perf_counter()
        res.statistics = multiphase_statistics(res.labels, a, len(fr), periodic)
        res.timings["statistics"] = time.perf_counter() - t0
    tell(1.0, "done")
    return res
