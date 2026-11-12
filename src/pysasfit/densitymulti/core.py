"""pyDENSSmulti core: 3D reconstruction from several SAS contrasts with a DENSS-type iteration (own numpy code).

The iteration follows the DENSS algorithm of T. D. Grant (Nature Methods 15, 191 (2018)): Fourier transform of the
map, replacement of the shell-averaged intensity by the measured one (phases kept), back transform, real-space
constraints, shrink-wrap support. It is written here from the published description; no DENSS code is used or
imported.

Two schemes combine the contrasts:

``"components"`` (default, the component basis)
    The particle is made of M components with known scattering-length densities ρ_c(f_k) in every contrast k
    (D2O fraction f_k, partial H/D exchange, or X-rays). The unknowns are volume fractions φ_c(x) ∈ [0, 1] with
    Σ_c φ_c ≤ 1 (the rest is solvent). Contrast k sees Δρ_k(x) = Σ_c C[k, c] φ_c(x) with the contrast matrix
    C[k, c] = ρ_c(f_k) − ρ_s(f_k). After the Fourier step every contrast proposes its own Δρ'_k(x); the new φ(x)
    is the weighted, bounded least-squares solution of Δρ'_k(x) ≈ Σ_c C[k, c] φ_c(x) in every voxel. A contrast
    in which a component is matched has a zero in that column: it says nothing about that component and everything
    about the others, so match points help instead of pulling the matched component towards the solvent.

``"average"`` (after DENSS-Multiple, Sumner & Qian, BBA Advances 2, 100063 (2022); written from the paper)
    One SLD map ρ(x), the same in every contrast (no H/D exchange), and an occupancy b(x) ∈ [0, 1]: contrast k sees
    Δρ_k = b (ρ − ρ_s,k). After the Fourier step b follows from a per-voxel least-squares fit of the proposals, and
    ρ is DENSS-Multiple's real-space (weighted) average of the proposals ρ_s,k + Δρ'_k / b, which is also the
    least-squares SLD for that occupancy. The occupancy is an addition to the paper's scheme: without it the
    solvent region becomes a fictitious material with the mean solvent SLD and the particle is not localised. Kept
    for comparison: with H/D exchange or matched components its single SLD per voxel is a compromise.

Units: q in 1/nm on input and output (as everywhere in pySASfit), Å inside (grid, voxel, maps); SLD in
1e-6 Å⁻² (= 1e10 cm⁻²); intensities in the unit of the data. With ``scale="absolute"`` the intensity is
I(q) = n_p |A(q)|² in 1/cm with the particle number density n_p from concentration (mg/ml) and molar mass (kDa).
"""
from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional, Sequence

import numpy as np

try:                                      # scipy.fft is faster and can use several threads
    from scipy import fft as _fft
    _FFT_KW = {"workers": -1}
except ImportError:                       # pragma: no cover
    _fft = np.fft
    _FFT_KW = {}

__all__ = [
    "SLD_H2O", "SLD_D2O", "SLD_XRAY_WATER", "PRESETS", "Component", "Contrast", "MultiError", "Settings",
    "MultiResult", "AverageResult", "contrast_matrix", "identifiability", "SmoothCurve", "smooth_curve",
    "make_grid", "debye_profile", "match_sld", "reconstruct", "reconstruct_many", "align_and_average",
    "project_capped_simplex", "bounded_lsq", "core_shell_curves", "dumbbell_curves", "sphere_curves", "selftest",
    "save_result", "load_result", "number_density", "fit_tail_background",
]

#: neutron SLD of light and heavy water and X-ray SLD of water at 20 °C, 1e-6 Å⁻²
SLD_H2O = -0.561
SLD_D2O = 6.364
SLD_XRAY_WATER = 9.42

#: typical component SLDs (1e-6 Å⁻²): (in H2O, in D2O at full exchange of labile H, X-ray). Typical values only:
#: compute your own from the composition (e.g. a contrast calculator) and edit them.
PRESETS = {
    "protein (typical)": (1.82, 3.13, 12.3),
    "perdeuterated protein (typical)": (6.9, 8.2, 12.3),
    "lipid tails, h (typical)": (-0.39, -0.39, 7.8),
    "lipid tails, d (typical)": (6.9, 6.9, 7.8),
    "custom": (1.0, 1.0, 10.0),
}

_NA = 6.02214076e23


class MultiError(RuntimeError):
    """A problem with the input or the reconstruction (message for the user)."""


# ===================================================================================== components and contrasts
@dataclass
class Component:
    """A component of the particle.

    ``sld_h2o``: neutron SLD in 100 % H2O; ``sld_d2o``: neutron SLD in 100 % D2O when all labile H are exchanged
    (None: no exchangeable H, same as ``sld_h2o``); ``exchange``: fraction of the labile H that actually exchanges
    (buried regions exchange less); ``sld_xray``: X-ray SLD (for SAXS curves). All in 1e-6 Å⁻².
    """
    name: str
    sld_h2o: float
    sld_d2o: Optional[float] = None
    exchange: float = 1.0
    sld_xray: float = float("nan")

    def sld(self, d2o: float) -> float:
        d = self.sld_h2o if self.sld_d2o is None else self.sld_d2o
        return float(self.sld_h2o + d2o * self.exchange * (d - self.sld_h2o))

    def match_point(self, h2o=SLD_H2O, d2o=SLD_D2O) -> float:
        """D2O fraction at which the component is matched (may lie outside 0 … 1)."""
        a = self.sld(1.0) - self.sld(0.0)
        den = (d2o - h2o) - a
        return float("nan") if abs(den) < 1e-12 else float((self.sld(0.0) - h2o) / den)

    def to_dict(self):
        return {"name": self.name, "sld_h2o": self.sld_h2o, "sld_d2o": self.sld_d2o, "exchange": self.exchange,
                "sld_xray": None if not np.isfinite(self.sld_xray) else self.sld_xray}

    @classmethod
    def from_dict(cls, d):
        x = d.get("sld_xray")
        return cls(d["name"], float(d["sld_h2o"]), None if d.get("sld_d2o") is None else float(d["sld_d2o"]),
                   float(d.get("exchange", 1.0)), float("nan") if x is None else float(x))


@dataclass
class Contrast:
    """One measured curve and its contrast.

    q (1/nm), I, dI: the data (a flat background ``bg`` is subtracted before use); ``d2o``: D2O fraction of the
    solvent (0 … 1) for neutrons; ``radiation``: "neutron" or "xray"; ``solvent_sld``: override of the solvent SLD
    (None: from ``d2o`` or water for X-rays); ``sld_override``: {component name: SLD} to override single entries
    of the contrast matrix; ``conc``: relative concentration of this sample (the model intensity is multiplied by
    it; all 1 for one stock solution); ``weight``: weight of this contrast in the per-voxel combination;
    ``qmin``/``qmax`` (1/nm): range used (None: all).
    """
    name: str
    q: np.ndarray
    I: np.ndarray
    dI: np.ndarray
    d2o: float = 0.0
    radiation: str = "neutron"
    solvent_sld: Optional[float] = None
    sld_override: dict = field(default_factory=dict)
    conc: float = 1.0
    weight: float = 1.0
    bg: float = 0.0
    qmin: Optional[float] = None
    qmax: Optional[float] = None

    def solvent(self, h2o=SLD_H2O, d2o=SLD_D2O) -> float:
        if self.solvent_sld is not None:
            return float(self.solvent_sld)
        if self.radiation == "xray":
            return SLD_XRAY_WATER
        return float(h2o + self.d2o * (d2o - h2o))

    def component_sld(self, comp: Component) -> float:
        if comp.name in self.sld_override:
            return float(self.sld_override[comp.name])
        if self.radiation == "xray":
            if not np.isfinite(comp.sld_xray):
                raise MultiError(f"component {comp.name}: no X-ray SLD for the SAXS curve {self.name}")
            return float(comp.sld_xray)
        return comp.sld(self.d2o)

    def data(self):
        """(q in 1/nm, I − bg, dI) in the q range, finite points with dI > 0 only."""
        q, I, e = (np.asarray(a, float) for a in (self.q, self.I, self.dI))
        ok = np.isfinite(q) & np.isfinite(I) & (q > 0)
        if self.qmin is not None:
            ok &= q >= self.qmin
        if self.qmax is not None:
            ok &= q <= self.qmax
        e = np.where(np.isfinite(e) & (e > 0), e, np.nan)
        if np.all(np.isnan(e[ok])):
            e = 0.01 * np.abs(I) + 1e-12 * np.nanmax(np.abs(I[ok]))       # no errors given: 1 %
        else:
            e = np.where(np.isnan(e), np.nanmedian(e[ok] / np.maximum(np.abs(I[ok]), 1e-300)) * np.abs(I), e)
        ok &= np.isfinite(e) & (e > 0)
        if ok.sum() < 8:
            raise MultiError(f"{self.name}: fewer than 8 usable points")
        o = np.argsort(q[ok])
        return q[ok][o], (I[ok] - self.bg)[o], e[ok][o]


def contrast_matrix(components: Sequence[Component], contrasts: Sequence[Contrast], h2o=SLD_H2O, d2o=SLD_D2O):
    """(C [K, M], ρ_s [K], ρ_c [K, M]): contrast matrix, solvent SLDs and component SLDs (1e-6 Å⁻²)."""
    rc = np.array([[k.component_sld(c) for c in components] for k in contrasts], float).reshape(len(contrasts),
                                                                                                len(components))
    rs = np.array([k.solvent(h2o, d2o) for k in contrasts], float)
    return rc - rs[:, None], rs, rc


def identifiability(C, weights=None, names=None, tol=0.05):
    """Can the components be separated with these contrasts?

    Returns a dict: ``rank``, ``cond`` (condition number of W^½ C), ``sensitivity`` (per component: 1/√((CᵀWC)⁻¹)_cc,
    the contrast with which the component is determined, in 1e-6 Å⁻²; 0 if not determined), ``matched`` (per
    component: indices of the contrasts in which it is matched within ``tol`` × the largest contrast) and
    ``messages`` (human-readable warnings; empty if everything is fine)."""
    C = np.atleast_2d(np.asarray(C, float))
    K, M = C.shape
    w = np.ones(K) if weights is None else np.asarray(weights, float)
    names = list(names) if names is not None else [f"component {i + 1}" for i in range(M)]
    Cw = C * np.sqrt(np.maximum(w, 0))[:, None]
    sv = np.linalg.svd(Cw, compute_uv=False) if Cw.size else np.array([])
    scale = float(np.max(np.abs(C))) if C.size else 0.0
    rank = int(np.sum(sv > 1e-9 * max(scale, 1e-30))) if sv.size else 0
    cond = float(sv[0] / sv[-1]) if sv.size and sv[-1] > 1e-12 * max(sv[0], 1e-30) else float("inf")
    msgs = []
    sens = np.zeros(M)
    if rank < M:
        msgs.append(f"The contrast matrix has rank {rank} < {M} components: the components cannot all be "
                    "separated. Add contrasts (other D2O fractions or X-rays) or merge components.")
        Hinv = np.linalg.pinv(Cw.T @ Cw)
    else:
        Hinv = np.linalg.inv(Cw.T @ Cw)
    for c in range(M):
        v = Hinv[c, c]
        sens[c] = 1 / math.sqrt(v) if v > 0 and rank == M else 0.0
    matched = []
    for c in range(M):
        idx = [k for k in range(K) if abs(C[k, c]) <= tol * max(scale, 1e-30)]
        matched.append(idx)
        if K > 1 and len(idx) == K - 1:
            msgs.append(f"{names[c]} is matched in all contrasts but one: it is determined by that curve alone.")
        if len(idx) == K:
            msgs.append(f"{names[c]} is matched in every contrast: it is invisible.")
    if np.isfinite(cond) and cond > 30 and rank == M:
        weak = names[int(np.argmin(sens))]
        msgs.append(f"Condition number {cond:.3g}: the components are poorly separated ({weak} is the weakest). "
                    "Contrasts with more different D2O fractions help.")
    return {"rank": rank, "cond": cond, "sensitivity": sens, "matched": matched, "messages": msgs}


def number_density(conc_mg_ml: float, mw_kda: float) -> float:
    """Particle number density (1/cm³) from concentration (mg/ml) and molar mass (kDa)."""
    if not (conc_mg_ml > 0 and mw_kda > 0):
        raise MultiError("concentration and molar mass must be positive for the absolute scale")
    return conc_mg_ml * 1e-3 * _NA / (mw_kda * 1e3)


# ===================================================================================== data smoothing
@dataclass
class SmoothCurve:
    """Regularised p(r) fit of a curve (q in 1/Å inside): I(q) = Σ_j p_j 4π Δr sin(q r_j)/(q r_j)."""
    r: np.ndarray
    p: np.ndarray
    dr: float
    chi2: float
    lam: float

    def __call__(self, q_A):
        q = np.asarray(q_A, float)
        return _pr_kernel(q, self.r, self.dr) @ self.p

    @property
    def I0(self):
        return float(4 * np.pi * self.dr * np.sum(self.p))

    @property
    def rg(self):
        """Rg (Å) from p(r); NaN when Σ p r² / Σ p is negative (mixed-sign contrast)."""
        s0 = np.sum(self.p)
        s2 = np.sum(self.p * self.r ** 2)
        return float(np.sqrt(s2 / (2 * s0))) if s0 != 0 and s2 / s0 > 0 else float("nan")


def _pr_kernel(q, r, dr):
    qr = np.outer(q, r)
    return 4 * np.pi * dr * np.sinc(qr / np.pi)


def smooth_curve(q_A, I, dI, dmax_A, npoints=None, lam=None):
    """Regularised (second-difference Tikhonov) p(r) fit up to ``dmax_A``, without positivity (a contrast with
    components of opposite sign has a p(r) with negative parts). λ by the discrepancy principle (reduced χ² ≈ 1, or
    1.2 × the smallest reachable χ² when the errors are underestimated)."""
    q, I, e = (np.asarray(a, float) for a in (q_A, I, dI))
    ns = dmax_A * q.max() / np.pi                               # Shannon channels
    n = int(npoints or np.clip(round(3 * ns) + 10, 30, 120))
    r = np.linspace(0, dmax_A, n + 2)[1:-1]                     # p(0) = p(Dmax) = 0
    dr = dmax_A / (n + 1)
    A = _pr_kernel(q, r, dr) / e[:, None]
    b = I / e
    L = np.diff(np.eye(n + 2), 2, axis=0)[:, 1:-1]              # second differences with zero ends
    AtA, Atb, LtL = A.T @ A, A.T @ b, L.T @ L
    norm = np.trace(AtA) / np.trace(LtL)

    def solve(lmb):
        p = np.linalg.solve(AtA + lmb * norm * LtL, Atb)
        res = A @ p - b
        return p, float(res @ res) / max(len(q) - 1, 1)

    if lam is None:
        lo, hi = -10.0, 4.0
        _, cmin = solve(10 ** lo)
        target = max(1.0, 1.2 * cmin)
        if solve(10 ** hi)[1] <= target:
            lam = 10 ** hi
        else:
            for _ in range(50):
                mid = 0.5 * (lo + hi)
                if solve(10 ** mid)[1] > target:
                    hi = mid
                else:
                    lo = mid
            lam = 10 ** lo
    p, chi2 = solve(lam)
    return SmoothCurve(r, p, dr, chi2, float(lam))


def fit_tail_background(q, I, dI, fraction=0.3, exponent=4.0):
    """Flat background of the tail (last ``fraction`` of the points): weighted fit of I = A q^-B + bg with B fixed
    (default 4, Porod). Returns (bg, σ_bg)."""
    q, I, e = (np.asarray(a, float) for a in (q, I, dI))
    k = max(0, int(len(q) * (1 - fraction)))
    q, I, e = q[k:], I[k:], e[k:]
    e = np.where(np.isfinite(e) & (e > 0), e, np.nanmedian(np.abs(I)) * 0.01 + 1e-300)
    X = np.column_stack([q ** -exponent, np.ones_like(q)]) / e[:, None]
    coef, *_ = np.linalg.lstsq(X, I / e, rcond=None)
    cov = np.linalg.pinv(X.T @ X)
    return float(coef[1]), float(np.sqrt(max(cov[1, 1], 0)))


# ===================================================================================== grid
@dataclass
class Grid:
    n: int
    side: float                 # Å
    voxel: float                # Å
    qmag: np.ndarray            # |q| of every FFT voxel (1/Å)
    shell: np.ndarray           # shell index of every FFT voxel
    qshell: np.ndarray          # mean |q| of every shell
    counts: np.ndarray          # voxels per shell
    r2: np.ndarray              # squared distance from the box centre (Å²)


def make_grid(dmax_A, qmax_A, oversampling=3.0, n=None, nmax=128, shell_width=0.5):
    """Cubic grid of side oversampling × Dmax with n points (even); n from the data qmax when None
    (q_max of the grid = π / voxel ≥ qmax of the data). The q shells are ``shell_width`` × 2π/side wide: half the
    Fourier sampling of the box (default) fits the minima of the curves markedly better than full-width shells
    (sphere test: χ² 86 instead of 309 with 1 % noise; Rg +1.7 % instead of +3.2 %)."""
    side = float(oversampling) * float(dmax_A)
    if n is None:
        n = int(math.ceil(side * qmax_A / np.pi))
        n = max(32, n + (n % 2))
        n = min(n, nmax)
    n = int(n) + (int(n) % 2)
    voxel = side / n
    f = np.fft.fftfreq(n, d=voxel) * 2 * np.pi
    qx, qy, qz = np.meshgrid(f, f, f, indexing="ij")
    qmag = np.sqrt(qx ** 2 + qy ** 2 + qz ** 2)
    dq = shell_width * 2 * np.pi / side
    shell = np.floor(qmag / dq + 0.5).astype(np.int64)
    ns = int(shell.max()) + 1
    counts = np.bincount(shell.ravel(), minlength=ns).astype(float)
    qshell = np.bincount(shell.ravel(), weights=qmag.ravel(), minlength=ns) / np.maximum(counts, 1)
    x = (np.arange(n) - n / 2) * voxel
    X, Y, Z = np.meshgrid(x, x, x, indexing="ij")
    return Grid(n, side, voxel, qmag, shell, qshell, counts, X ** 2 + Y ** 2 + Z ** 2)


def _shell_mean(I3, g: Grid):
    return np.bincount(g.shell.ravel(), weights=I3.ravel(), minlength=len(g.counts)) / np.maximum(g.counts, 1)


# ===================================================================================== constraints
def project_capped_simplex(phi):
    """Euclidean projection of every column of ``phi`` (M, N) onto {φ ≥ 0, Σ φ ≤ 1}."""
    phi = np.asarray(phi, float)
    p = np.maximum(phi, 0)
    over = p.sum(axis=0) > 1
    if np.any(over):
        v = phi[:, over]
        u = -np.sort(-v, axis=0)
        css = np.cumsum(u, axis=0) - 1
        k = np.arange(1, v.shape[0] + 1)[:, None]
        cond = u - css / k > 0
        rho = v.shape[0] - 1 - np.argmax(cond[::-1], axis=0)
        theta = css[rho, np.arange(v.shape[1])] / (rho + 1)
        p[:, over] = np.maximum(v - theta, 0)
    return p


def bounded_lsq(D, C, w, iters=40, x0=None):
    """Per-voxel weighted least squares  min Σ_k w_k (D_k − Σ_c C_kc φ_c)²  with φ ≥ 0, Σ_c φ_c ≤ 1.

    D (K, N) proposed contrast maps, C (K, M), w (K). Accelerated projected gradient (FISTA), vectorised over the N
    voxels, started from the projected unconstrained solution (or ``x0``). Returns φ (M, N)."""
    C = np.asarray(C, float)
    w = np.asarray(w, float)
    H = C.T @ (w[:, None] * C)
    G = C.T @ (w[:, None] * D)
    L = float(np.max(np.linalg.eigvalsh(H))) if H.size else 1.0
    if L <= 0:
        return np.zeros((C.shape[1], D.shape[1]))
    if x0 is None:
        x0 = project_capped_simplex(np.linalg.pinv(H) @ G)
    x = x0
    y = x.copy()
    t = 1.0
    for _ in range(iters):
        xn = project_capped_simplex(y - (H @ y - G) / L)
        tn = 0.5 * (1 + math.sqrt(1 + 4 * t * t))
        y = xn + ((t - 1) / tn) * (xn - x)
        x, t = xn, tn
    return x


def _gauss(a, sigma):
    from scipy.ndimage import gaussian_filter
    return gaussian_filter(a, sigma, mode="wrap")


def _largest_blob(mask):
    from scipy.ndimage import label
    lab, n = label(mask)
    if n <= 1:
        return mask
    sizes = np.bincount(lab.ravel())
    sizes[0] = 0
    return lab == int(np.argmax(sizes))


def _centre_shift(w):
    """Integer voxel shift that moves the centre of mass of the weight map ``w`` to the box centre."""
    n = w.shape[0]
    tot = w.sum()
    if tot <= 0:
        return (0, 0, 0)
    idx = np.arange(n)
    cm = [float((w.sum(axis=tuple(j for j in range(3) if j != a)) * idx).sum() / tot) for a in range(3)]
    return tuple(int(round(n / 2 - c)) for c in cm)


# ===================================================================================== reconstruction
@dataclass
class Settings:
    """Options of one reconstruction (defaults close to DENSS' FAST mode)."""
    dmax_nm: float = 10.0
    scheme: str = "components"           # "components" or "average"
    oversampling: float = 3.0
    n: Optional[int] = None              # grid points per side (None: from qmax of the data)
    nmax: int = 96
    shell_width: float = 0.5             # q-shell width / (2π / box side)
    max_steps: int = 3000
    min_steps: int = 1000
    conv_window: int = 100
    conv_tol: float = 1e-3
    shrinkwrap_start: int = 50
    shrinkwrap_every: int = 20
    shrinkwrap_threshold: float = 0.2
    sigma_start: float = 3.0              # voxels
    sigma_end: float = 1.5
    sigma_decay: float = 0.99
    connectivity_step: int = 500
    recentre_until: int = 1000
    scale: str = "fitted"                # "fitted" or "absolute"
    conc_mg_ml: float = 1.0
    mw_kda: float = 100.0
    lsq_iters: int = 30
    beyond_factor: float = 1.0           # shells beyond the data: amplitude × this (1 = free, 0 = removed)
    occupancy_percentile: float = 99.5   # fitted scale: this percentile of Σφ in the support is set to 1
    sld_bounds: Optional[tuple] = None   # "average": (lo, hi) bounds of the SLD map, None: none
    min_occupancy: float = 0.05          # "average": the SLD is defined where the occupancy exceeds this
    seed: Optional[int] = None
    h2o: float = SLD_H2O
    d2o: float = SLD_D2O

    def to_dict(self):
        return dict(self.__dict__)


@dataclass
class MultiResult:
    """One reconstruction. Maps on an n³ grid, voxel (Å), centred in the box."""
    scheme: str
    component_names: list
    contrast_names: list
    phi: np.ndarray                      # (M, n, n, n) volume fractions ("components"); "average": (1, n, n, n) occupancy
    rho: Optional[np.ndarray]            # (n, n, n) SLD map ("average"; NaN where unoccupied), else None
    drho: np.ndarray                     # (K, n, n, n) contrast maps Δρ_k (1e-6 Å⁻²)
    support: np.ndarray
    voxel: float
    C: np.ndarray
    solvent: np.ndarray
    fits: list                           # per contrast: dict(q (1/nm), I, err, model)
    chi2: np.ndarray                     # per contrast, against the data
    rg: np.ndarray                       # per contrast, from the map (Å; NaN for mixed-sign contrast)
    volumes: np.ndarray                  # per component Σ φ_c × voxel volume (Å³)
    scale: float
    steps: int
    stats: dict                          # per step: chi2 (S, K), support_volume (S), scale (S)
    settings: dict
    seconds: float
    warnings: list = field(default_factory=list)
    seed: Optional[int] = None
    workdir: Optional[Path] = None
    match_sld: float = float("nan")      # solvent SLD at the match point of the particle, from I(0) of the contrasts

    @property
    def chi2_total(self):
        return float(np.mean(self.chi2))

    @property
    def support_volume(self):
        return float(self.support.sum() * self.voxel ** 3)

    def occupancy(self):
        """Total particle map Σ φ_c (the occupancy in the averaging scheme)."""
        return self.phi.sum(axis=0)

    def sld_map(self, k):
        """SLD map of the particle in contrast k (solvent outside)."""
        return self.drho[k] + self.solvent[k]


@dataclass
class AverageResult:
    """Aligned and averaged maps of several reconstructions (same fields as :class:`MultiResult` for the average,
    plus the spread)."""
    mean: MultiResult
    phi_sd: Optional[np.ndarray]
    rho_sd: Optional[np.ndarray]
    runs: list
    correlation: np.ndarray             # correlation of every aligned map with the reference
    improper: list                      # whether the mirror image (enantiomer) was used
    chi2_sd: np.ndarray
    rg_sd: np.ndarray
    volumes_sd: np.ndarray


def _prepare(components, contrasts, s: Settings):
    if not contrasts:
        raise MultiError("no contrasts")
    if s.scheme not in ("components", "average"):
        raise MultiError(f"unknown scheme {s.scheme!r}")
    if s.scheme == "components" and not components:
        raise MultiError("no components")
    if s.dmax_nm is None or not s.dmax_nm > 0:
        raise MultiError("Dmax must be positive")
    dmax = 10 * float(s.dmax_nm)
    data = [k.data() for k in contrasts]
    qmax = max(float(d[0].max()) for d in data) / 10
    g = make_grid(dmax, qmax, s.oversampling, s.n, s.nmax, s.shell_width)
    warn = []
    qgrid = np.pi / g.voxel
    for k, d in zip(contrasts, data):
        if d[0].max() / 10 > qgrid * 1.001:
            warn.append(f"{k.name}: data beyond q = {10 * qgrid:.3g} 1/nm (grid limit) are not used; "
                        "more grid points or a smaller oversampling use them")
    if s.scheme == "components":
        C, rs, _rc = contrast_matrix(components, contrasts, s.h2o, s.d2o)
    else:
        rs = np.array([k.solvent(s.h2o, s.d2o) for k in contrasts], float)
        C = np.zeros((len(contrasts), 0))
    return g, data, C, rs, warn


def _targets(contrasts, data, g: Grid, dmax_A):
    """Per contrast: smooth curve, target intensity on the shells (NaN = free), weights of the shells."""
    T, smooth = [], []
    for k, (q, I, e) in zip(contrasts, data):
        qa = q / 10
        sm = smooth_curve(qa, I, e, dmax_A)
        smooth.append(sm)
        t = np.full(len(g.qshell), np.nan)
        use = (g.qshell <= qa.max()) & (g.counts > 0)
        tv = sm(g.qshell[use])
        floor = 1e-6 * max(float(np.max(np.abs(tv))), 1e-300)
        t[use] = np.maximum(tv, floor)
        T.append(t)
    return np.array(T), smooth


def debye_profile(d, voxel, q_A):
    """Exact orientational average |A(q)|² of the voxel map ``d`` (Σ_voxels d e^{iqr}, voxel-volume units) at the
    given q (1/Å): Debye sum over the pair-distance histogram of the lattice (autocorrelation by a zero-padded FFT,
    binned by the integer r²), times the form factor of a sphere with the voxel volume. Used for the final fits;
    the iteration itself works on shells."""
    n = d.shape[0]
    N = 2 * n
    F = _fft.fftn(d, s=(N, N, N), **_FFT_KW)
    ac = _fft.ifftn(F.real ** 2 + F.imag ** 2, **_FFT_KW).real
    i = np.arange(N)
    i = np.where(i < N // 2, i, i - N) ** 2
    r2 = (i[:, None, None] + i[None, :, None] + i[None, None, :]).ravel()
    gam = np.bincount(r2, weights=ac.ravel())
    keep = np.nonzero(gam)[0]
    r = np.sqrt(keep) * voxel
    q = np.asarray(q_A, float)
    I = np.empty(len(q))
    for j in range(0, len(q), 256):                     # blocks: bounded memory
        I[j:j + 256] = np.sinc(np.outer(q[j:j + 256], r) / np.pi) @ gam[keep]
    x = q * (3 / (4 * np.pi)) ** (1 / 3) * voxel
    with np.errstate(all="ignore"):
        fv = np.where(x > 1e-6, 3 * (np.sin(x) - x * np.cos(x)) / np.maximum(x, 1e-6) ** 3, 1.0)
    return I * fv ** 2


def match_sld(I0, solvent, conc=None):
    """Solvent SLD at the match point of the whole particle (1e-6 Å⁻²) from the forward intensities of several
    contrasts, I0_k = b c_k (ρ̄ − ρ_s,k)² (classical contrast variation: √I0 is linear in ρ_s and vanishes at the
    match point; with H/D exchange ρ̄ changes with the D2O fraction and the result is the solvent SLD at which it is
    matched). NaN if not determined (fewer than two contrasts or no change of the solvent)."""
    I0 = np.asarray(I0, float)
    rs = np.asarray(solvent, float)
    c = np.ones_like(I0) if conc is None else np.asarray(conc, float)
    ok = np.isfinite(I0) & (c > 0)
    if ok.sum() < 2 or np.ptp(rs[ok]) <= 0:
        return float("nan")
    y = np.sqrt(np.maximum(I0[ok], 0) / c[ok])
    x = rs[ok]
    lo, hi = x.min() - 3 * np.ptp(x), x.max() + 3 * np.ptp(x)
    grid = np.linspace(lo, hi, 4001)
    best, arg = np.inf, float("nan")
    for r in grid:
        a = np.abs(r - x)
        den = float(a @ a)
        if den <= 0:
            continue
        k = float(a @ y) / den
        res = float(np.sum((y - k * a) ** 2))
        if res < best:
            best, arg = res, float(r)
    return arg


def _rg_map(d, g: Grid):
    s0 = d.sum()
    if s0 == 0:
        return float("nan")
    n = d.shape[0]
    idx = (np.arange(n) - n / 2) * g.voxel
    cm = [float((d.sum(axis=tuple(j for j in range(3) if j != a)) * idx).sum() / s0) for a in range(3)]
    X, Y, Z = np.meshgrid(idx - cm[0], idx - cm[1], idx - cm[2], indexing="ij")
    rg2 = float((d * (X ** 2 + Y ** 2 + Z ** 2)).sum() / s0)
    return math.sqrt(rg2) if rg2 > 0 else float("nan")


def reconstruct(components: Sequence[Component], contrasts: Sequence[Contrast], settings: Optional[Settings] = None,
                progress: Optional[Callable[[int, float], None]] = None,
                cancel: Optional[Callable[[], bool]] = None) -> MultiResult:
    """One reconstruction from all contrasts. ``progress(step, χ²)`` is called every 10 steps; ``cancel()`` → stop
    after the current step (the result so far is returned with a warning)."""
    s = settings or Settings()
    t0 = time.perf_counter()
    g, data, C, rs, warn = _prepare(components, contrasts, s)
    K = len(contrasts)
    M = C.shape[1]
    dmax_A = 10 * s.dmax_nm
    T, smooth = _targets(contrasts, data, g, dmax_A)
    use = np.isfinite(T)
    conc = np.array([max(float(k.conc), 1e-12) for k in contrasts])
    wts = np.array([max(float(k.weight), 0.0) for k in contrasts])
    if not np.any(wts > 0):
        raise MultiError("all contrast weights are zero")
    # solvent SLD at the match point of the whole particle, from the forward intensities (Stuhrmann)
    rho_match = match_sld([sm.I0 for sm in smooth], rs, conc) if K > 1 else float("nan")
    vv = g.voxel ** 3
    absolute = s.scale == "absolute"
    if absolute:
        unit = number_density(s.conc_mg_ml, s.mw_kda) * 1e-28 * vv ** 2   # |F|² (1e-6 Å⁻² voxel) -> 1/cm
    else:
        unit = 1.0
    rng = np.random.default_rng(s.seed)
    n = g.n
    shape = (n, n, n)
    support = np.ones(shape, bool)
    if s.scheme == "components":
        if np.linalg.matrix_rank(C) < M:
            warn.append("the contrast matrix is rank deficient: the components are not all separable")
        phi = rng.random((M,) + shape) / M
        rho = None
    else:
        # "average": occupancy b (stored as the single 'component' of phi) and one SLD ρ per voxel, the same in every
        # contrast. Δρ_k = b (ρ − ρ_s,k); solvent outside the particle (b = 0) in every contrast.
        if K < 2 or float(np.ptp(rs)) <= 0:
            raise MultiError("the averaging scheme needs at least two contrasts with different solvent SLDs")
        M = 1
        phi = rng.random((1,) + shape) * 0.5
        guess = rho_match if np.isfinite(rho_match) else float(np.mean(rs))
        if s.sld_bounds is not None:
            guess = float(np.clip(guess, *sorted(s.sld_bounds)))
        rho = np.full(shape, guess)
        sw, sr, srr = float(wts.sum()), float(wts @ rs), float(wts @ rs ** 2)
        det2 = sw * srr - sr * sr
    scale = 1.0
    sigma = s.sigma_start
    hist_chi, hist_vol, hist_scale = [], [], []
    # errors of the smoothed curve on the shells: data errors interpolated, × √(points per shell interval)
    dq = s.shell_width * 2 * np.pi / g.side
    sig = np.ones_like(T)
    for k, (q, I, e) in enumerate(data):
        qa = q / 10
        dens = len(qa) / max(qa.max() - qa.min(), 1e-12) * dq           # data points per shell interval
        sig[k] = np.interp(g.qshell, qa, e) / math.sqrt(max(dens, 1.0))

    def contrast_maps():
        if rho is None:
            return np.einsum("kc,c...->k...", C, phi) * support
        return phi[0][None] * (rho[None] - rs[:, None, None, None]) * support

    cancelled = False
    step = 0
    for step in range(1, s.max_steps + 1):
        drho = contrast_maps()
        F = _fft.fftn(drho, axes=(1, 2, 3), **_FFT_KW)
        P = (F.real ** 2 + F.imag ** 2)
        Im = np.array([_shell_mean(P[k], g) for k in range(K)])           # shell means of |F|²
        model = Im * conc[:, None] * unit
        if not absolute:                                                   # global scale, relative LS on shells
            m = np.where(use, model / np.where(use, T, 1), 0.0)
            den = float(np.sum(m * m))
            scale = float(np.sum(m)) / den if den > 0 else 1.0
        mod = scale * model
        # χ² on the shells against the smoothed data (errors interpolated), as DENSS does
        chis = np.array([float(np.mean(((T[k, use[k]] - mod[k, use[k]]) / sig[k, use[k]]) ** 2)) for k in range(K)])
        hist_chi.append(chis)
        hist_vol.append(float(support.sum()) * vv)
        hist_scale.append(scale)
        # magnitude replacement on every shell with data
        fac = np.full_like(mod, s.beyond_factor)
        ok = use & (mod > 0)
        fac[ok] = np.sqrt(T[ok] / mod[ok])
        F *= fac[np.arange(K)[:, None, None, None], g.shell[None]]
        D = _fft.ifftn(F, axes=(1, 2, 3), **_FFT_KW).real
        # combination of the contrasts
        if rho is None:
            sel = support.ravel()
            Dk = D.reshape(K, -1)[:, sel]
            x0 = phi.reshape(M, -1)[:, sel]
            if not absolute and Dk.size:
                # fitted scale: scale and amplitude are degenerate. Fix the amplitude so that the densest voxels of
                # the unconstrained solution are fully occupied (Σφ = 1); the scale follows in the next step.
                Hc = C.T @ (wts[:, None] * C)
                phi_u = np.linalg.pinv(Hc) @ (C.T @ (wts[:, None] * Dk))
                occ = np.maximum(phi_u, 0).sum(axis=0)
                ptop = float(np.percentile(occ, s.occupancy_percentile))
                if ptop > 1e-30:
                    Dk = Dk / ptop
            new = np.zeros((M, n ** 3))
            new[:, sel] = bounded_lsq(Dk, C, wts, iters=s.lsq_iters, x0=project_capped_simplex(x0))
            phi = new.reshape((M,) + shape)
        else:
            # occupancy: per-voxel least squares of D_k = a − ρ_s,k b (a = b ρ), b clipped to [0, 1]
            SD = np.tensordot(wts, D, axes=(0, 0))
            SrD = np.tensordot(wts * rs, D, axes=(0, 0))
            b = (sr * SD - sw * SrD) / det2
            if not absolute:                       # fitted scale: the densest voxels are fully occupied
                bs = b[support]
                ptop = float(np.percentile(bs, s.occupancy_percentile)) if bs.size else 0.0
                if ptop > 1e-30:
                    b, D = b / ptop, D / ptop
            b = np.clip(b, 0.0, 1.0) * support
            # SLD: DENSS-Multiple's real-space average of the contrasts' proposals ρ_s,k + Δρ'_k / b (this is also the
            # least-squares SLD for the given occupancy)
            ok = b > s.min_occupancy
            prop = rs[:, None, None, None] + D / np.where(ok, b, 1.0)[None]
            new_rho = np.tensordot(wts / sw, prop, axes=(0, 0))
            if s.sld_bounds is not None:
                new_rho = np.clip(new_rho, *sorted(s.sld_bounds))
            rho = np.where(ok, new_rho, rho)
            phi = b[None]
        # shrink-wrap
        if step >= s.shrinkwrap_start and step % s.shrinkwrap_every == 0:
            occ = phi.sum(axis=0)
            sm = _gauss(occ, sigma)
            new_sup = sm > s.shrinkwrap_threshold * sm.max() if sm.max() > 0 else support
            if step >= s.connectivity_step:
                new_sup = _largest_blob(new_sup)
            if new_sup.sum() >= 8:
                support = new_sup
            sigma = max(s.sigma_end, sigma * s.sigma_decay)
            if step <= s.recentre_until:
                sh = _centre_shift(occ * support)
                if any(sh):
                    support = np.roll(support, sh, axis=(0, 1, 2))
                    phi = np.roll(phi, sh, axis=(1, 2, 3))
                    if rho is not None:
                        rho = np.roll(rho, sh, axis=(0, 1, 2))
        phi *= support
        if progress is not None and step % 10 == 0:
            progress(step, float(np.mean(chis)))
        if cancel is not None and cancel():
            cancelled = True
            break
        if step >= max(s.min_steps, s.connectivity_step + s.conv_window):
            h = np.array([np.mean(c) for c in hist_chi[-s.conv_window:]])
            if np.std(h) <= s.conv_tol * np.mean(h):
                break
    if cancelled:
        warn.append(f"stopped by the user after {step} steps")
    elif step >= s.max_steps:
        warn.append(f"not converged within {s.max_steps} steps")
    # final maps and fits: exact Debye average at the data q
    drho = contrast_maps()
    raw = []
    for k in range(K):
        q, I, e = data[k]
        raw.append(debye_profile(drho[k], g.voxel, q / 10) * conc[k] * unit)
    if not absolute:                                   # final scale: weighted least squares against the data
        num = sum(float(np.sum(m * d[1] / d[2] ** 2)) for m, d in zip(raw, data))
        den = sum(float(np.sum((m / d[2]) ** 2)) for m, d in zip(raw, data))
        scale = num / den if den > 0 else scale
    fits, chi2, rg = [], np.empty(K), np.empty(K)
    for k in range(K):
        q, I, e = data[k]
        mk = scale * raw[k]
        fits.append({"q": q, "I": I, "err": e, "model": mk, "smooth": smooth[k](q / 10)})
        chi2[k] = float(np.mean(((I - mk) / e) ** 2))
        rg[k] = _rg_map(drho[k], g)
        if not np.all(np.isfinite(mk)):
            warn.append(f"{contrasts[k].name}: non-finite model intensity")
    vols = phi.reshape(M, -1).sum(axis=1) * vv
    stats = {"chi2": np.array(hist_chi), "support_volume": np.array(hist_vol), "scale": np.array(hist_scale)}
    if rho is not None:
        rho = np.where(phi[0] > s.min_occupancy, rho, np.nan)
    names = [c.name for c in components] if s.scheme == "components" else ["particle (occupancy)"]
    return MultiResult(s.scheme, names, [k.name for k in contrasts], phi, rho, drho, support, g.voxel, C, rs, fits,
                       chi2, rg, vols, float(scale), step, stats, s.to_dict(), time.perf_counter() - t0, warn, s.seed,
                       None, rho_match)


def reconstruct_many(components, contrasts, settings: Optional[Settings] = None, nmaps=4, seeds=None, cores=1,
                     progress=None, cancel=None):
    """``nmaps`` independent reconstructions (different seeds), in ``cores`` threads. ``progress(i_done, n)``."""
    import concurrent.futures as cf
    from dataclasses import replace
    s = settings or Settings()
    if seeds is None:
        base = s.seed if s.seed is not None else int(np.random.default_rng().integers(1, 2 ** 31 - 1))
        seeds = [base + i for i in range(nmaps)]
    out = [None] * len(seeds)
    done = 0
    with cf.ThreadPoolExecutor(max_workers=max(1, int(cores))) as ex:
        futs = {ex.submit(reconstruct, components, contrasts, replace(s, seed=sd), None, cancel): i
                for i, sd in enumerate(seeds)}
        for f in cf.as_completed(futs):
            out[futs[f]] = f.result()
            done += 1
            if progress is not None:
                progress(done, len(seeds))
    return out


# ===================================================================================== alignment and averaging
def _inertia(w, voxel):
    n = w.shape[0]
    idx = (np.arange(n) - n / 2) * voxel
    X, Y, Z = np.meshgrid(idx, idx, idx, indexing="ij")
    tot = w.sum()
    cm = np.array([(w * X).sum(), (w * Y).sum(), (w * Z).sum()]) / tot
    R = np.stack([X - cm[0], Y - cm[1], Z - cm[2]], axis=-1).reshape(-1, 3)
    Iten = (R * w.reshape(-1, 1)).T @ R / tot
    vals, vecs = np.linalg.eigh(Iten)
    return cm, vecs


def _transform(a, R, c_from, c_to, voxel, order=1):
    """Resample ``a`` so that the point c_from + x is moved to c_to + R x (centres in Å from the box centre)."""
    from scipy.ndimage import affine_transform
    n = a.shape[0]
    ctr = np.full(3, n / 2)
    # output index o -> position p = (o − ctr) voxel ; source position = c_from + Rᵀ (p − c_to)
    Rt = R.T
    offset = ctr + (c_from - Rt @ c_to) / voxel - Rt @ ctr
    return affine_transform(a, Rt, offset=offset, order=order, mode="constant", cval=0.0)


def _amount(r):
    """b ρ of an averaging-scheme result (0 where unoccupied), for transformations and averages; None otherwise."""
    return None if r.rho is None else np.nan_to_num(r.phi[0] * r.rho, nan=0.0)


def _corr(a, b):
    a = a - a.mean()
    b = b - b.mean()
    d = math.sqrt(float((a * a).sum() * (b * b).sum()))
    return float((a * b).sum() / d) if d > 0 else 0.0


def align_and_average(results: Sequence[MultiResult], reference: Optional[int] = None) -> AverageResult:
    """Align every map onto a reference (the one with the lowest χ² unless given) by its principal axes, trying the
    8 sign choices of the axes (rotations and mirror images: the enantiomer cannot be told from SAS data), keep the
    choice with the highest correlation of the total maps, then average the maps (and their spread)."""
    rs = [r for r in results if r is not None]
    if not rs:
        raise MultiError("nothing to average")
    if len({r.drho.shape for r in rs}) != 1:
        raise MultiError("the maps have different grids")
    ref = int(np.argmin([r.chi2_total for r in rs])) if reference is None else int(reference)
    R0 = rs[ref]
    vox = R0.voxel
    c0, E0 = _inertia(R0.occupancy(), vox)
    aligned, corr, improper = [], [], []
    for i, r in enumerate(rs):
        if i == ref:
            aligned.append((r.phi, _amount(r), r.drho, r.support.astype(float)))
            corr.append(1.0)
            improper.append(False)
            continue
        occ = r.occupancy()
        c1, E1 = _inertia(occ, vox)
        best = None
        for sx in (1, -1):
            for sy in (1, -1):
                for sz in (1, -1):
                    Rm = E0 @ np.diag([sx, sy, sz]) @ E1.T
                    cc = _corr(_transform(occ, Rm, c1, c0 * 0, vox), _transform(R0.occupancy(), np.eye(3), c0,
                                                                              c0 * 0, vox))
                    if best is None or cc > best[0]:
                        best = (cc, Rm)
        cc, Rm = best
        # move both into the frame of the reference (reference centre -> its own place)
        def tr(a, Rm=Rm, c1=c1):
            return _transform(a, Rm, c1, c0, vox)
        phi = np.array([tr(p) for p in r.phi])
        amt = None if r.rho is None else tr(_amount(r))
        drho = np.array([tr(d) for d in r.drho])
        aligned.append((phi, amt, drho, tr(r.support.astype(float))))
        corr.append(cc)
        improper.append(bool(np.linalg.det(Rm) < 0))
    phis = [a[0] for a in aligned]
    phi_m = np.mean(phis, axis=0)
    phi_sd = np.std(phis, axis=0)
    rho_m = rho_sd = None
    if aligned[0][1] is not None:            # SLD: occupancy-weighted mean b ρ / b, and the spread of the proposals
        bmin = float(R0.settings.get("min_occupancy", 0.05))
        amts = np.array([a[1] for a in aligned])
        bs = np.array([a[0][0] for a in aligned])
        import warnings
        with np.errstate(all="ignore"), warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            rho_m = np.where(phi_m[0] > bmin, amts.mean(axis=0) / phi_m[0], np.nan)
            each = np.where(bs > bmin, amts / bs, np.nan)
            rho_sd = np.where(phi_m[0] > bmin, np.nanstd(each, axis=0), np.nan)
    drho_m = np.mean([a[2] for a in aligned], axis=0)
    sup = np.mean([a[3] for a in aligned], axis=0) >= 0.5
    chi = np.array([r.chi2 for r in rs])
    rg = np.array([r.rg for r in rs])
    vol = np.array([r.volumes for r in rs]) if rs[0].volumes.size else np.zeros((len(rs), 0))
    from dataclasses import replace
    import warnings
    with warnings.catch_warnings():                     # Rg is NaN for mixed-sign contrasts
        warnings.simplefilter("ignore", RuntimeWarning)
        rg_m, rg_sd = np.nanmean(rg, axis=0), np.nanstd(rg, axis=0)
    mean = replace(R0, phi=phi_m, rho=rho_m, drho=drho_m, support=sup, chi2=chi.mean(axis=0), rg=rg_m,
                   volumes=vol.mean(axis=0), warnings=sorted({w for r in rs for w in r.warnings}), workdir=None)
    return AverageResult(mean, phi_sd, rho_sd, list(rs), np.array(corr), improper, chi.std(axis=0), rg_sd,
                         vol.std(axis=0))


# ===================================================================================== output
def save_result(res, folder, prefix="multi"):
    """Write the maps (MRC: one per component, per contrast, support, SLD map), the fits (one text file per
    contrast: q (1/nm), I, err, model) and a JSON summary into ``folder``. Returns the folder."""
    from ..density.core import write_mrc
    import re
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    avg = res if isinstance(res, AverageResult) else None
    r = avg.mean if avg else res

    def safe(s):
        return re.sub(r"[^A-Za-z0-9_.-]", "_", s)[:40] or "x"
    files = {}
    vox = (r.voxel,) * 3
    if r.phi is not None:
        for c, name in enumerate(r.component_names):
            p = folder / f"{prefix}_phi_{c + 1}_{safe(name)}.mrc"
            write_mrc(p, r.phi[c], vox)
            files[f"phi:{name}"] = p.name
            if avg is not None and avg.phi_sd is not None:
                p = folder / f"{prefix}_phi_sd_{c + 1}_{safe(name)}.mrc"
                write_mrc(p, avg.phi_sd[c], vox)
    if r.rho is not None:
        p = folder / f"{prefix}_sld.mrc"
        write_mrc(p, np.nan_to_num(np.where(r.support, r.rho, 0.0), nan=0.0), vox)
        files["sld"] = p.name
    for k, name in enumerate(r.contrast_names):
        p = folder / f"{prefix}_drho_{k + 1}_{safe(name)}.mrc"
        write_mrc(p, r.drho[k], vox)
        files[f"drho:{name}"] = p.name
        f = r.fits[k]
        p = folder / f"{prefix}_fit_{k + 1}_{safe(name)}.dat"
        np.savetxt(p, np.column_stack([f["q"], f["I"], f["err"], f["model"]]),
                   header=f"pyDENSSmulti fit of {name}; chi2={r.chi2[k]:.6g}\nq(1/nm) I err I_map")
        files[f"fit:{name}"] = p.name
    p = folder / f"{prefix}_support.mrc"
    write_mrc(p, r.support.astype(float), vox)
    files["support"] = p.name
    summ = {"scheme": r.scheme, "components": r.component_names, "contrasts": r.contrast_names,
            "C": r.C.tolist(), "solvent": r.solvent.tolist(), "chi2": r.chi2.tolist(),
            "rg_A": [None if not np.isfinite(x) else float(x) for x in r.rg], "volumes_A3": r.volumes.tolist(),
            "scale": r.scale, "steps": r.steps, "voxel_A": r.voxel, "grid": r.drho.shape[1],
            "support_volume_A3": r.support_volume, "seconds": r.seconds, "seed": r.seed,
            "match_sld": None if not np.isfinite(r.match_sld) else r.match_sld,
            "settings": {k: (list(v) if isinstance(v, tuple) else v) for k, v in r.settings.items()},
            "warnings": r.warnings, "files": files}
    if avg is not None:
        summ["average"] = {"nmaps": len(avg.runs), "correlation": avg.correlation.tolist(),
                           "mirror": avg.improper, "chi2_sd": avg.chi2_sd.tolist(),
                           "volumes_sd_A3": avg.volumes_sd.tolist()}
    with open(folder / f"{prefix}_summary.json", "w", encoding="utf-8") as fh:
        json.dump(summ, fh, indent=1)
    if avg is None:
        np.savetxt(folder / f"{prefix}_stats_by_step.dat",
                   np.column_stack([r.stats["chi2"], r.stats["support_volume"], r.stats["scale"]]),
                   header="chi2 per contrast ..., support volume (A^3), scale")
    if avg is not None:
        avg.mean.workdir = folder
    else:
        r.workdir = folder
    return folder


def load_result(folder, prefix="multi"):
    """Read a result written by :func:`save_result` back (maps, fits, summary) as a :class:`MultiResult`
    (an average is read as its mean map; the per-step statistics only if written)."""
    from ..density.core import read_mrc
    folder = Path(folder)
    p = folder / f"{prefix}_summary.json"
    if not p.is_file():
        raise MultiError(f"{folder}: no {p.name}")
    with open(p, encoding="utf-8") as fh:
        S = json.load(fh)
    files = S.get("files", {})

    def mrc(name):
        rho, _vox, _h = read_mrc(folder / name)
        return np.asarray(rho, float)
    comps, cons = S["components"], S["contrasts"]
    phi = np.array([mrc(files[f"phi:{c}"]) for c in comps])
    drho = np.array([mrc(files[f"drho:{k}"]) for k in cons])
    rho = None
    if "sld" in files:
        rho = mrc(files["sld"])
        bmin = float(S.get("settings", {}).get("min_occupancy", 0.05))
        rho = np.where(phi[0] > bmin, rho, np.nan)
    support = mrc(files["support"]) > 0.5
    fits = []
    for k in cons:
        a = np.loadtxt(folder / files[f"fit:{k}"], ndmin=2)
        fits.append({"q": a[:, 0], "I": a[:, 1], "err": a[:, 2], "model": a[:, 3], "smooth": a[:, 3] * np.nan})
    stats = {"chi2": np.zeros((0, len(cons))), "support_volume": np.zeros(0), "scale": np.zeros(0)}
    sp = folder / f"{prefix}_stats_by_step.dat"
    if sp.is_file():
        a = np.loadtxt(sp, ndmin=2)
        stats = {"chi2": a[:, :len(cons)], "support_volume": a[:, len(cons)], "scale": a[:, len(cons) + 1]}
    rg = np.array([np.nan if v is None else v for v in S.get("rg_A", [None] * len(cons))], float)
    st = S.get("settings", {})
    if isinstance(st.get("sld_bounds"), list):
        st["sld_bounds"] = tuple(st["sld_bounds"])
    return MultiResult(S["scheme"], comps, cons, phi, rho, drho, support, float(S["voxel_A"]), np.array(S["C"]),
                       np.array(S["solvent"]), fits, np.array(S["chi2"]), rg, np.array(S["volumes_A3"]),
                       float(S["scale"]), int(S["steps"]), stats, st, float(S.get("seconds", 0)),
                       list(S.get("warnings", [])), S.get("seed"), folder, float(S.get("match_sld") or np.nan))


# ===================================================================================== synthetic test objects
def _sphere_amp(q_A, R):
    x = np.asarray(q_A, float) * R
    with np.errstate(all="ignore"):
        f = np.where(x > 1e-6, 3 * (np.sin(x) - x * np.cos(x)) / x ** 3, 1.0 - x * x / 10)
    return 4 / 3 * np.pi * R ** 3 * f


def core_shell_curves(R_core_nm=2.0, R_shell_nm=3.5, components=None, d2o=(0.0, 0.42, 0.7, 1.0), noise=0.01,
                      n=200, conc_mg_ml=None, mw_kda=None, seed=1):
    """Synthetic curves of a core-shell sphere: core = components[0], shell = components[1] (defaults: a
    perdeuterated core in a protein shell). Returns (components, [Contrast]). Without concentration and molar mass the
    intensity is |A|² × 1e-8 (arbitrary unit), else n_p |A|² in 1/cm."""
    if components is None:
        components = [Component("core (d-protein)", 6.9, 8.2, 1.0, 12.3), Component("shell (protein)", 1.82, 3.13,
                                                                                       1.0, 12.3)]
    rng = np.random.default_rng(seed)
    R1, R2 = 10 * R_core_nm, 10 * R_shell_nm
    q = np.linspace(0.25 / R_shell_nm, 14 / R_shell_nm, n)
    qa = q / 10
    out = []
    for f in d2o:
        rs = SLD_H2O + f * (SLD_D2O - SLD_H2O)
        d1, d2 = components[0].sld(f) - rs, components[1].sld(f) - rs
        A = d1 * _sphere_amp(qa, R1) + d2 * (_sphere_amp(qa, R2) - _sphere_amp(qa, R1))
        I = A ** 2 * (number_density(conc_mg_ml, mw_kda) * 1e-28 if conc_mg_ml else 1e-8)
        e = noise * I + noise * 1e-3 * I.max()
        out.append(Contrast(f"{round(100 * f)} % D2O", q, I + rng.normal(0, e), e, d2o=f))
    return components, out


def dumbbell_curves(R1_nm=2.0, R2_nm=1.5, components=None, d2o=(0.0, 0.42, 0.7, 1.0), noise=0.01, n=200,
                    seed=1):
    """Synthetic curves of two touching spheres of different components (defaults: a perdeuterated protein R1 and a
    protein R2; the second is matched near 42 % D2O). Returns (components, [Contrast]); arbitrary intensity unit."""
    if components is None:
        components = [Component("A (d-protein)", 6.9, 8.2, 1.0, 12.3), Component("B (protein)", 1.82, 3.13, 1.0, 12.3)]
    rng = np.random.default_rng(seed)
    R1, R2 = 10 * R1_nm, 10 * R2_nm
    d = R1 + R2
    q = np.linspace(0.25 / (R1_nm + R2_nm), 14 / (R1_nm + R2_nm), n)
    qa = q / 10
    out = []
    for f in d2o:
        rs = SLD_H2O + f * (SLD_D2O - SLD_H2O)
        a1 = (components[0].sld(f) - rs) * _sphere_amp(qa, R1)
        a2 = (components[1].sld(f) - rs) * _sphere_amp(qa, R2)
        I = 1e-8 * (a1 ** 2 + a2 ** 2 + 2 * a1 * a2 * np.sinc(qa * d / np.pi))
        e = noise * I + noise * 1e-3 * I.max()
        out.append(Contrast(f"{round(100 * f)} % D2O", q, I + rng.normal(0, e), e, d2o=f))
    return components, out


def sphere_curves(R_nm=2.0, n=200, noise=0.01, seed=1):
    """One contrast of a homogeneous sphere (one component, protein in H2O). Returns (components, [Contrast])."""
    comp = [Component("particle", 1.82, 3.13, 1.0, 12.3)]
    rng = np.random.default_rng(seed)
    q = np.linspace(0.3 / R_nm, 15 / R_nm, n)
    d = comp[0].sld(0.0) - SLD_H2O
    I = 1e-8 * (d * _sphere_amp(q / 10, 10 * R_nm)) ** 2
    e = noise * I + noise * 1e-3 * I.max()
    return comp, [Contrast("sphere, H2O", q, I + rng.normal(0, e), e, d2o=0.0)]


def selftest(out=print, steps=1500) -> bool:
    """Sphere R = 2 nm (Rg 15.49 Å) with one contrast: passes when the map's Rg is within 10 %."""
    comp, con = sphere_curves()
    r = reconstruct(comp, con, Settings(dmax_nm=4.0, seed=1, max_steps=steps, min_steps=min(steps, 1000)))
    exact = math.sqrt(0.6) * 20
    dev = r.rg[0] / exact - 1
    out(f"sphere R = 2 nm: Rg {r.rg[0]:.2f} Å (exact {exact:.2f}, {100 * dev:+.1f} %), χ² {r.chi2[0]:.3g}, "
        f"{r.steps} steps, {r.seconds:.1f} s, grid {r.drho.shape[1]}³")
    return abs(dev) < 0.10
