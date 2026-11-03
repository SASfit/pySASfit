"""The SASmorph chain: I(q) -> gamma(r) -> g(r) -> f(k) -> clipped Gaussian random field.

    from pysasfit.core import load_data
    from pysasfit.morph import MorphParameters, reconstruct

    q, I, s, _ = load_data("sample.dat")
    res = reconstruct(q, I, s, MorphParameters(porosity=0.3, box_size=2000, n=128))
    res.matrix          # bool (n, n, n), True = phase 2 ("pore")
    res.statistics      # S/V, connectivity, bicontinuity
    print(res.report())

Steps (Ingham et al., J. Appl. Cryst. 44 (2011) 221; Berk, Phys. Rev. A 44 (1991) 5069;
Quintanilla, Modelling Simul. Mater. Sci. Eng. 15 (2007) S337):

1. extrapolate I(q) to q = 0 and (Porod) to infinity, invariant Q
2. gamma(r) = (1/Q) integral q^2 I(q) sin(qr)/(qr) dq
3. g(r) from phi (1-phi) gamma(r) = F_alpha(g(r))  (level-cut relation)
4. spectral function f(k) (3D Fourier transform of g)
5. Gaussian random field with correlation g on an n^3 grid, cut at alpha
6. statistics and the back-calculated scattering of the structure

The reconstruction is *a* morphology with the measured two-point statistics, not
*the* morphology: it is meaningful for isotropic, non-periodic two-phase samples
whose data reach the Porod regime.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from dataclasses import field as dc_field

import numpy as np

from .correlation import LevelCut, spectral_function, wavevector_distribution
from .extrapolation import ExtrapolatedCurve, extrapolate
from .grf import cut_level, field_cosine, field_fft
from .scattering import contrast_from_invariant, levelcut_intensity, voxel_intensity
from .statistics import morphology_statistics, specific_surface_porod

__all__ = ["MorphParameters", "MorphResult", "reconstruct", "correlations", "suggest_box",
           "porosity_from_invariant"]


@dataclass
class MorphParameters:
    """All settings of a reconstruction (lengths in the inverse unit of q)."""

    porosity: float = 0.5            # volume fraction of phase 2 ("pore")
    box_size: float | None = None    # edge L of the cube; None: suggest_box
    n: int = 128                     # voxels per edge
    qmin: float | None = None        # data range used
    qmax: float | None = None
    low_q: str = "guinier"           # guinier | linear | constant | none
    low_q_points: int = 10
    porod_points: int | None = None  # last points for the Porod fit (default 15 %)
    porod_qmin: float | None = None  # ... or all points above this q
    porod_auto: bool = True          # neither given: range chosen from the data (else the last 15 %)
    fit_background: bool = False     # Porod fit with flat background B
    refine: int = 4                  # log-log nodes per data interval
    method: str = "fft"              # fft | cosine
    n_waves: int = 4000              # cosine method only
    threshold: str = "quantile"      # quantile | alpha
    seed: int | None = None
    nr: int | None = None            # r points (default from voxel size and q_max)
    nk: int = 1000                   # k points of the spectral function

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, d):
        names = cls.__dataclass_fields__
        return cls(**{k: v for k, v in d.items() if k in names})


@dataclass
class MorphResult:
    params: MorphParameters
    curve: ExtrapolatedCurve
    invariant: float
    r: np.ndarray
    gamma: np.ndarray
    g: np.ndarray
    k: np.ndarray
    f: np.ndarray
    Pk: np.ndarray
    levelcut: LevelCut
    q_data: np.ndarray
    I_data: np.ndarray
    sigma_data: np.ndarray | None
    I_levelcut: np.ndarray            # model intensity at q_data (ensemble average)
    q_voxel: np.ndarray | None = None
    I_voxel: np.ndarray | None = None
    matrix: np.ndarray | None = None
    psi: np.ndarray | None = None      # the Gaussian field (keep_field=True)
    level: float = np.nan              # cut level: matrix = psi > level
    statistics: dict = dc_field(default_factory=dict)
    S_over_V_porod: float = np.nan
    drho2: float = np.nan
    chi2_levelcut: float = np.nan
    warnings: list = dc_field(default_factory=list)
    timings: dict = dc_field(default_factory=dict)

    @property
    def voxel_size(self):
        return self.params.box_size / self.params.n

    def report(self):
        p, c, st = self.params, self.curve, self.statistics
        lines = [
            "SASmorph reconstruction",
            f"  data range           q = {c.q_data[0]:.4g} ... {c.q_data[1]:.4g}",
            f"  low-q extrapolation  {c.low_q} {', '.join(f'{k}={v:.4g}' for k, v in c.low_q_info.items())}",
            f"  Porod fit            K = {c.K:.4g} +- {c.dK:.2g} on q = {c.porod_range[0]:.4g} ... {c.porod_range[1]:.4g}"
            f" ({c.porod_mode})"
            + (f", background B = {c.B:.4g} +- {c.dB:.2g}" if p.fit_background else ""),
            f"  invariant            Q = {self.invariant:.5g}  (Porod tail {100 * c.porod_invariant_fraction:.1f} %)",
            f"  contrast (from Q)    drho^2 = {self.drho2:.5g}",
            f"  porosity             phi = {p.porosity:.4g}, level alpha = {self.levelcut.alpha:.4f}",
            f"  S/V (Porod)          {self.S_over_V_porod:.5g}",
            f"  box                  L = {p.box_size:.5g}, n = {p.n}, voxel a = {self.voxel_size:.4g}, method = {p.method}",
            f"  model vs data        chi2_red(level-cut) = {self.chi2_levelcut:.4g}",
        ]
        if st:
            lines += [
                f"  voxel porosity       {st['porosity_voxels']:.4f}",
                f"  S/V (voxels, x 2/3)  {st['S_over_V_voxels']:.5g}",
                f"  pore  clusters {st['pore']['clusters']}, largest {100 * st['pore']['largest_fraction']:.1f} %, "
                f"percolates x/y/z {''.join('+' if v else '-' for v in st['pore']['percolates'])}",
                f"  solid clusters {st['solid']['clusters']}, largest {100 * st['solid']['largest_fraction']:.1f} %, "
                f"percolates x/y/z {''.join('+' if v else '-' for v in st['solid']['percolates'])}",
                f"  bicontinuous         {'yes' if st['bicontinuous'] else 'no'}",
                f"  degree of connectivity {st['degree_of_connectivity']:.4f}",
            ]
        if self.timings:
            lines.append("  timings              " + ", ".join(f"{k} {v:.2f} s" for k, v in self.timings.items()))
        if self.warnings:
            lines.append("Warnings:")
            lines += [f"  - {w}" for w in self.warnings]
        return "\n".join(lines)


def porosity_from_invariant(Q, delta_rho, q_unit="A"):
    """Volume fraction (the root <= 0.5) from an absolute invariant.

    Q in cm^-1 * (q unit)^-3 with q in 1/Angstrom (``q_unit='A'``) or 1/nm ('nm'),
    delta_rho the scattering length density difference in cm^-2.
    """
    conv = {"A": 1e24, "nm": 1e21}[q_unit]
    x = Q * conv / (2 * np.pi ** 2 * delta_rho ** 2)
    if not 0 < x <= 0.25:
        raise ValueError(f"phi (1-phi) = {x:.4g} is outside (0, 1/4]: check units, contrast or calibration")
    return 0.5 * (1 - np.sqrt(1 - 4 * x))


def suggest_box(curve, n=128):
    """(L, n): about 5 x 2 pi / q* (q* = maximum of the Kratky plot q^2 I), limited so that the
    voxel size is not larger than twice pi / q_max of the data."""
    qd = curve.q[(curve.q >= curve.q_data[0]) & (curve.q <= curve.q_data[1])]
    Id = curve.I[(curve.q >= curve.q_data[0]) & (curve.q <= curve.q_data[1])]
    qs = qd[np.argmax(qd ** 2 * Id)]
    L = 5 * 2 * np.pi / qs
    L = min(L, n * 2 * np.pi / curve.q_data[1])
    return float(L), n


def _r_grid(p, curve):
    a = p.box_size / p.n
    rmax = max(np.sqrt(3) * 0.5 * p.box_size * 1.02, 2 * a)
    dr = min(a / 4.0, np.pi / (16.0 * curve.q_data[1]))      # q_max r steps of pi/16: accurate model I(q) at q_max
    nr = p.nr or int(np.clip(np.ceil(rmax / dr), 200, 20000))
    return np.linspace(0.0, rmax, nr + 1)


def correlations(q, I, sigma=None, params=None):
    """Steps 1-4 only (fast): returns a MorphResult without field and matrix."""
    p = MorphParameters() if params is None else params
    w = []
    t0 = time.perf_counter()
    curve = extrapolate(q, I, sigma, qmin=p.qmin, qmax=p.qmax, low_q=p.low_q, low_q_points=p.low_q_points,
                        porod_points=p.porod_points, porod_qmin=p.porod_qmin, fit_background=p.fit_background,
                        porod_auto=p.porod_auto,
                        refine=p.refine)
    if p.box_size is None:
        p.box_size, _ = suggest_box(curve, p.n)
        w.append(f"box size not given: using L = {p.box_size:.4g} (5 x 2 pi / q* of the Kratky maximum)")
    Q = curve.invariant
    if curve.K <= 0:
        w.append("Porod constant K <= 0: the data do not reach a q^-4 regime; S/V and the finest "
                 "structure are not reliable")
    if curve.porod_info.get("message"):
        w.append(f"Porod range: {curve.porod_info['message']}")
    if curve.porod_invariant_fraction > 0.2:
        w.append(f"{100 * curve.porod_invariant_fraction:.0f} % of the invariant comes from the Porod tail: "
                 "the q range is probably too short")
    r = _r_grid(p, curve)
    gamma = curve.gamma(r)
    lc = LevelCut(p.porosity)
    g = lc.g_from_gamma(gamma)
    if lc.clipped:
        w.append(f"gamma(r) outside the range of a level-cut field at {lc.clipped} of {r.size} r points "
                 "(values clipped): the porosity may not match the data")
    if abs(gamma[-1]) > 0.05:
        w.append(f"gamma(r_max) = {gamma[-1]:.3f}: correlations extend beyond half the box diagonal, "
                 "use a larger box")
    k = np.geomspace(curve.q_data[0] / 10, curve.q_data[1] * 10, p.nk)
    f = spectral_function(r, g, k)
    Pk = wavevector_distribution(k, f)
    drho2 = contrast_from_invariant(Q, p.porosity)
    # data points actually used (without the refinement nodes)
    qq = np.asarray(q, float)
    m = np.isfinite(qq) & np.isfinite(I) & (qq > 0) & (qq >= curve.q_data[0]) & (qq <= curve.q_data[1])
    q_data, I_data = qq[m], np.asarray(I, float)[m] - curve.B
    s_data = None if sigma is None else np.asarray(sigma, float)[m]
    o = np.argsort(q_data)
    q_data, I_data = q_data[o], I_data[o]
    s_data = None if s_data is None else s_data[o]
    I_lc = levelcut_intensity(q_data, r, lc.F(g), drho2)
    chi2 = np.nan
    if s_data is not None and np.all(s_data > 0):
        chi2 = float(np.mean(((I_lc - I_data) / s_data) ** 2))
    a = p.box_size / p.n
    if np.pi / a < curve.q_data[1]:
        w.append(f"voxel size a = {a:.4g} does not resolve the data up to q_max (pi/a = {np.pi / a:.4g} < "
                 f"{curve.q_data[1]:.4g}): structure finer than a is lost")
    res = MorphResult(params=p, curve=curve, invariant=Q, r=r, gamma=gamma, g=g, k=k, f=f, Pk=Pk, levelcut=lc,
                      q_data=q_data, I_data=I_data, sigma_data=s_data, I_levelcut=I_lc,
                      S_over_V_porod=specific_surface_porod(curve.K, Q, p.porosity), drho2=drho2,
                      chi2_levelcut=chi2, warnings=w)
    res.timings["correlations"] = time.perf_counter() - t0
    return res


def reconstruct(q, I, sigma=None, params=None, keep_field=False, statistics=True, progress=None):
    """Full chain (steps 1-6). ``progress(fraction, text)`` is called between the steps."""
    p = MorphParameters() if params is None else params

    def tell(x, t):
        if progress is not None:
            progress(x, t)

    tell(0.0, "correlation functions")
    res = correlations(q, I, sigma, p)
    a = p.box_size / p.n
    tell(0.2, f"Gaussian random field ({p.method}, {p.n}^3)")
    t0 = time.perf_counter()
    if p.method == "fft":
        psi, info = field_fft(res.r, res.g, p.n, a, rng=p.seed)
        if info["negative_power"] > 0.01:
            res.warnings.append(f"{100 * info['negative_power']:.1f} % negative spectral power discarded "
                                "(box too small for the correlation length?)")
    elif p.method == "cosine":
        psi, info = field_cosine(res.r, res.g, p.n, a, n_waves=p.n_waves, rng=p.seed,
                                 progress=(lambda x: tell(0.2 + 0.5 * x, "random waves")) if progress else None)
        if info["fraction_beyond_nyquist"] > 0.05:
            res.warnings.append(f"{100 * info['fraction_beyond_nyquist']:.0f} % of the waves are finer than the "
                                "voxel grid (aliasing): increase n or decrease the box")
    else:
        raise ValueError(f"unknown method {p.method!r}")
    res.timings["field"] = time.perf_counter() - t0
    res.level = cut_level(psi, p.porosity, p.threshold, res.levelcut.alpha)
    mat = psi > res.level
    res.matrix = mat
    if keep_field:
        res.psi = psi
    periodic = p.method == "fft"
    tell(0.75, "scattering of the structure")
    t0 = time.perf_counter()
    res.q_voxel, res.I_voxel, _ = voxel_intensity(mat, a, res.drho2, periodic=periodic)
    res.timings["scattering"] = time.perf_counter() - t0
    if statistics:
        tell(0.85, "statistics")
        t0 = time.perf_counter()
        res.statistics = morphology_statistics(mat, a, periodic=periodic)
        res.timings["statistics"] = time.perf_counter() - t0
    tell(1.0, "done")
    return res
