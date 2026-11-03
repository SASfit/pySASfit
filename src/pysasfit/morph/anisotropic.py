"""SASmorph for anisotropic samples with a symmetry axis in the detector plane (texture).

A 2D pattern I(q_x, q_y) (beam along z, small angles: q_z = 0) of a sample with **uniaxial symmetry** about an
axis e in the detector plane, at the angle ``axis_deg`` from q_x towards q_y, determines the full 3D
intensity: I(q) depends only on q (the length of the scattering vector) and mu, the absolute cosine of its angle to e
(centrosymmetry folds mu to [0, 1]), and the
detector plane contains every mu. Two models:

``mode="B"`` -- **uniaxial**, any shape of the anisotropy
    I(q, mu) = sum_l I_l(q) P_l(mu) (even l <= lmax) by least squares in q bins; each I_l(q) is extended
    to q = 0 and to infinity (l = 0 as the isotropic curve, l >= 2 with I_l ~ q^2 at low q and
    I_l = K_l q^-4 beyond the data); the 3D correlation function follows from the plane-wave expansion

        C(r) = sum_l C_l(r) P_l(r^.e),   C_l(r) = (-1)^(l/2) / (2 pi^2) integral q^2 I_l(q) j_l(qr) dq.

``mode="A"`` -- **affine**: an isotropic structure stretched by lambda along the axis (both perpendicular
    directions unchanged): I(q) = lambda I_iso(sqrt(lambda^2 q_par^2 + q_perp^2)). lambda (and the axis,
    refined from the start value) is fitted by collapsing the 2D data onto one curve; that curve goes through
    the isotropic pipeline, and gamma(r) = gamma_iso(sqrt(r_par^2/lambda^2 + r_perp^2)).

Either way the level-cut relation S_2(r) - phi^2 = F(g(r)) holds for every vector r, so g(r) = F^-1(phi (1 -
phi) gamma(r)) direction by direction; the Gaussian field is synthesised by FFT from g sampled on the periodic
grid (as :func:`pysasfit.morph.grf.field_fft`, without assuming that g depends on the distance only).

Box frame: the symmetry axis is the box x axis (array axis 0), the beam the box z axis (array axis 2); the
detector plane is the box (x, y) plane with q_par = q_x(box), q_perp = q_y(box).

Checks: the level-cut model I_l(q) (ensemble average with the reconstructed g) and the voxel I_l(q) (all
3D wave vectors of the voxel structure) against the data components, and the 2D maps.
"""

from __future__ import annotations

import os
import time
from dataclasses import asdict, dataclass
from dataclasses import field as dc_field

import numpy as np
from scipy.integrate import cumulative_simpson
from scipy.special import eval_legendre, gamma as gamma_fn, spherical_jn

from .correlation import LevelCut
from .extrapolation import extrapolate, fit_porod
from .grf import cut_level
from .pipeline import MorphParameters, correlations
from .scattering import contrast_from_invariant, levelcut_intensity
from .statistics import morphology_statistics, specific_surface_porod

__all__ = ["Data2D", "load_2d", "AnisoParameters", "AnisoResult", "estimate_axis", "axis_coordinates",
           "legendre_components", "spherical_transform", "fit_affine", "correlations_aniso",
           "reconstruct_aniso", "field_from_grid_correlation", "chord_lengths", "MODES"]

MODES = ("A", "B")


# ---------------------------------------------------------------------------------------------- data
@dataclass
class Data2D:
    """A 2D pattern as flat arrays of the valid pixels (background already subtracted, or B fitted)."""

    qx: np.ndarray
    qy: np.ndarray
    I: np.ndarray
    sigma: np.ndarray | None = None
    name: str = "data"
    path: str | None = None
    shape: tuple | None = None         # (ny, nx) of the image when it came from one
    q_unit: str = "A"                  # "A" (1/Angstrom) or "nm"

    def __post_init__(self):
        qx, qy, I = (np.asarray(a, float).ravel() for a in (self.qx, self.qy, self.I))
        s = None if self.sigma is None else np.asarray(self.sigma, float).ravel()
        ok = np.isfinite(qx) & np.isfinite(qy) & np.isfinite(I) & (np.hypot(qx, qy) > 0)
        if s is not None:
            ok &= np.isfinite(s) & (s > 0)
        self.qx, self.qy, self.I = qx[ok], qy[ok], I[ok]
        self.sigma = None if s is None else s[ok]

    @property
    def q(self):
        return np.hypot(self.qx, self.qy)

    def subset(self, qmin=None, qmax=None):
        q = self.q
        m = np.ones(q.size, bool)
        if qmin is not None:
            m &= q >= qmin
        if qmax is not None:
            m &= q <= qmax
        return Data2D(self.qx[m], self.qy[m], self.I[m], None if self.sigma is None else self.sigma[m],
                      self.name, self.path, None, self.q_unit)


def load_2d(path, q_scale=1.0):
    """2D data: NXcanSAS (``.h5 .hdf5 .nxs``, e.g. pySASred's *Save 2D NXcanSAS*: I, Idev, Qx, Qy, Mask),
    ``.npz`` (qx, qy, I [, sigma, mask]) or a text file with columns q_x q_y I [dI]. Masked pixels are
    dropped. ``q_scale`` multiplies q (e.g. 0.1 for 1/nm -> 1/Angstrom)."""
    ext = os.path.splitext(str(path))[1].lower()
    name = os.path.splitext(os.path.basename(str(path)))[0]
    unit = "A"
    if ext in (".h5", ".hdf5", ".hdf", ".nxs"):
        import h5py
        with h5py.File(path, "r") as f:
            d = None

            def find(_n, o):
                nonlocal d
                if d is None and isinstance(o, h5py.Group) and all(k in o for k in ("I", "Qx", "Qy")):
                    d = o
            f.visititems(find)
            if d is None:
                raise ValueError(f"{path}: no SASdata group with I, Qx and Qy")
            I = np.asarray(d["I"][()], float)
            qx = np.asarray(d["Qx"][()], float)
            qy = np.asarray(d["Qy"][()], float)
            s = np.asarray(d["Idev"][()], float) if "Idev" in d else None
            mask = np.asarray(d["Mask"][()]) != 0 if "Mask" in d else np.zeros(I.shape, bool)
            qu = d["Qx"].attrs.get("units", b"1/angstrom")
            qu = qu.decode() if isinstance(qu, bytes) else str(qu)
            if "nm" in qu:
                unit = "nm"
        if qx.shape != I.shape:                      # 1D axes
            qx, qy = np.meshgrid(qx.ravel(), qy.ravel())
        shape = I.shape
    elif ext == ".npz":
        z = np.load(path)
        I, qx, qy = (np.asarray(z[k], float) for k in ("I", "qx", "qy"))
        s = np.asarray(z["sigma"], float) if "sigma" in z else None
        mask = np.asarray(z["mask"], bool) if "mask" in z else np.zeros(I.shape, bool)
        if qx.shape != I.shape:
            qx, qy = np.meshgrid(qx.ravel(), qy.ravel())
        shape = I.shape
    else:
        a = np.loadtxt(path, comments=("#", "%", "//"))
        if a.ndim != 2 or a.shape[1] < 3:
            raise ValueError(f"{path}: expected columns q_x q_y I [dI]")
        qx, qy, I = a[:, 0], a[:, 1], a[:, 2]
        s = a[:, 3] if a.shape[1] > 3 else None
        mask = np.zeros(I.shape, bool)
        shape = None
    keep = ~mask
    return Data2D(qx[keep] * q_scale, qy[keep] * q_scale, I[keep], None if s is None else s[keep], name,
                  str(path), shape, "A" if q_scale != 1.0 else unit)


# ---------------------------------------------------------------------------------------------- axis
def estimate_axis(data: Data2D, qmin=None, qmax=None):
    """Direction of maximum intensity from the cos 2phi moment of the pattern (each q shell normalised to
    its mean): (phi_max in degrees, anisotropy 0..1). The pattern is mirror-symmetric about phi_max and
    phi_max + 90 deg; which of them is the 3D symmetry axis is a choice: fibres (equatorial streak) have
    their axis at phi_max + 90, layers along the axis (meridional peaks) at phi_max."""
    d = data.subset(qmin, qmax)
    q, phi = d.q, np.arctan2(d.qy, d.qx)
    edges = np.geomspace(q.min(), q.max() * 1.0001, 25)
    idx = np.clip(np.digitize(q, edges) - 1, 0, edges.size - 2)
    mean = np.bincount(idx, weights=d.I, minlength=edges.size - 1) / np.maximum(np.bincount(idx, minlength=edges.size - 1), 1)
    w = d.I / np.where(mean[idx] > 0, mean[idx], np.inf)
    c, s = np.sum(w * np.cos(2 * phi)), np.sum(w * np.sin(2 * phi))
    return float(np.degrees(0.5 * np.arctan2(s, c)) % 180.0), float(np.hypot(c, s) / max(np.sum(np.abs(w)), 1e-300))


def axis_coordinates(qx, qy, axis_deg):
    """(q_par, q_perp) along and across the axis (detector plane)."""
    t = np.radians(axis_deg)
    return qx * np.cos(t) + qy * np.sin(t), -qx * np.sin(t) + qy * np.cos(t)


# ---------------------------------------------------------------------------------------------- Legendre
def _ls(lmax):
    return np.arange(0, int(lmax) + 1, 2)


def legendre_components(q, mu, I, sigma=None, edges=None, lmax=6, nbins=60, min_points=None):
    """I(q, mu) = sum_l I_l(q) P_l(mu), even l <= lmax, by weighted least squares in q bins.

    ``mu``: absolute cosine of the angle to the axis. Returns (q_bin, I_l (nq, nl), sigma_l (nq, nl), orders used per
    bin). Bins with too few points or too narrow a mu coverage use fewer orders (the missing ones set to 0)."""
    q, mu, I = (np.asarray(a, float).ravel() for a in (q, mu, I))
    s = np.ones_like(I) if sigma is None else np.asarray(sigma, float).ravel()
    ls = _ls(lmax)
    if edges is None:
        edges = np.geomspace(q[q > 0].min(), q.max() * 1.000001, nbins + 1)
    idx = np.digitize(q, edges) - 1
    nq, nl = edges.size - 1, ls.size
    # remove the steep q dependence inside a bin: every point is moved to the bin's mean q along the
    # isotropic reference I_ref(q) (azimuthal mean in finer bins, log-log interpolated); otherwise the
    # correlation of q and mu inside a bin on a square pixel grid shows up as a spurious anisotropy
    fe = np.geomspace(edges[0], edges[-1], 4 * nq + 1)
    fi = np.digitize(q, fe) - 1
    okf = (fi >= 0) & (fi < fe.size - 1)
    cnt_f = np.bincount(fi[okf], minlength=fe.size - 1)
    mean_f = np.bincount(fi[okf], weights=I[okf], minlength=fe.size - 1) / np.maximum(cnt_f, 1)
    qf = np.bincount(fi[okf], weights=q[okf], minlength=fe.size - 1) / np.maximum(cnt_f, 1)
    good = (cnt_f > 0) & (mean_f > 0)
    if good.sum() >= 3:
        lq, lI = np.log(qf[good]), np.log(mean_f[good])

        def Iref(x):
            return np.exp(np.interp(np.log(x), lq, lI))
    else:
        Iref = None
    Il = np.full((nq, nl), np.nan)
    sl = np.full((nq, nl), np.nan)
    qb = np.full(nq, np.nan)
    cnt = np.zeros(nq, int)
    used_n = np.zeros(nq, int)
    P = np.column_stack([eval_legendre(l, mu) for l in ls])
    for b in range(nq):
        m = idx == b
        k = int(m.sum())
        cnt[b] = k
        if k < 3:
            continue
        qb[b] = np.mean(q[m])
        if Iref is not None:
            f = Iref(qb[b]) / Iref(q[m])
        else:
            f = np.ones(int(m.sum()))
        # orders the mu coverage of this bin supports: occupied tenths of [0, 1] minus one, 3 points per order
        occupied = np.unique(np.minimum((mu[m] * 10).astype(int), 9)).size
        use = min(nl, max(1, occupied - 1), max(1, k // 3))
        if min_points is not None:
            use = min(use, max(1, k // min_points))
        A = P[m, :use] / (s[m] * f)[:, None]
        y = I[m] / s[m]
        coef, *_ = np.linalg.lstsq(A, y, rcond=None)
        Il[b, :] = 0.0
        Il[b, :use] = coef
        used_n[b] = use
        try:
            cov = np.linalg.inv(A.T @ A)
            res = y - A @ coef
            dof = max(1, k - use)
            if sigma is None:
                cov *= float(res @ res) / dof
            sl[b, :] = 0.0
            sl[b, :use] = np.sqrt(np.abs(np.diag(cov)))
        except np.linalg.LinAlgError:
            pass
    ok = np.isfinite(qb)
    return qb[ok], Il[ok], sl[ok], used_n[ok]


def _full_coverage(q, I, s, used, warnings):
    """Keep the q bins where all orders could be determined (drops e.g. the detector corners, where the
    azimuth is not covered)."""
    full = used == used.max()
    if not full.all():
        lo, hi = q[~full & (q < np.median(q))], q[~full & (q >= np.median(q))]
        where = ([f"{lo.size} below q = {lo.max():.4g} (beam stop)"] if lo.size else []) + \
                ([f"{hi.size} above q = {hi.min():.4g} (detector corners)"] if hi.size else [])
        warnings.append(f"q bins without full azimuthal coverage left out: {', '.join(where)}")
    return q[full], I[full], s[full]


def _J_tail(l, a):
    """integral_a^inf j_l(x) / x^2 dx for l >= 2 (closed form at a = 0, cumulative Simpson below a)."""
    a = np.atleast_1d(np.asarray(a, float))
    J0 = np.sqrt(np.pi) / 8.0 * gamma_fn((l - 1) / 2.0) / gamma_fn((l + 4) / 2.0)
    amax = float(a.max()) if a.size else 0.0
    if amax <= 0:
        return np.full(a.shape, J0)
    x = np.linspace(0.0, amax, max(2001, int(amax * 40) + 1))
    with np.errstate(divide="ignore", invalid="ignore"):
        f = np.where(x > 0, spherical_jn(l, x) / np.where(x > 0, x, 1.0) ** 2, 0.0)
    if l == 2:
        f[0] = 1.0 / 15.0
    cum = cumulative_simpson(f, x=x, initial=0.0)
    return J0 - np.interp(a, x, cum)


def spherical_transform(qn, In, l, r, K=0.0):
    """(-1)^(l/2)/(2 pi^2) [integral_0^qN q^2 I_l(q) j_l(qr) dq + K_l r integral_{qN r}^inf j_l(x)/x^2 dx]
    for l >= 2, I_l piecewise linear on the nodes ``qn`` (trapezoid on a fine uniform grid)."""
    r = np.atleast_1d(np.asarray(r, float))
    qN = float(qn[-1])
    nq = int(np.clip(16 * qN * r.max() / np.pi, 2000, 40000))
    qq = np.linspace(0.0, qN, nq)
    u = qq ** 2 * np.interp(qq, qn, In)
    out = np.empty_like(r)
    chunk = max(1, int(5e6 // nq))
    for i in range(0, r.size, chunk):
        rr = r[i:i + chunk]
        out[i:i + chunk] = np.trapezoid(u[None, :] * spherical_jn(l, np.outer(rr, qq)), qq, axis=1)
    if K:
        out += K * r * _J_tail(l, qN * r)
    return (-1) ** (l // 2) * out / (2 * np.pi ** 2)


# ---------------------------------------------------------------------------------------------- parameters
@dataclass
class AnisoParameters:
    """Settings of an anisotropic reconstruction (lengths in the inverse unit of q)."""

    mode: str = "B"                    # "A" affine, "B" uniaxial (Legendre)
    axis_deg: float | None = None      # symmetry axis in the detector plane (from q_x to q_y); None: estimate + 90
    fit_axis: bool = True              # mode A: refine the axis angle
    stretch: float | None = None       # mode A: lambda along the axis; None: fit
    lmax: int = 6                      # mode B: highest even Legendre order
    nbins: int = 60                    # q bins of the components / the collapsed curve
    porosity: float = 0.5
    box_size: float | None = None
    n: int = 128
    qmin: float | None = None
    qmax: float | None = None
    low_q: str = "guinier"
    low_q_points: int = 6
    porod_points: int | None = None
    porod_qmin: float | None = None
    porod_auto: bool = True            # neither given: Porod range chosen from the data
    fit_background: bool = True
    background: float | None = None    # fixed flat background (subtracted) when not fitted
    threshold: str = "quantile"        # quantile | alpha
    seed: int | None = None
    lmax_model: int = 12               # orders of the level-cut model check

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, d):
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})

    def iso(self, **kw):
        p = MorphParameters(porosity=self.porosity, box_size=self.box_size, n=self.n, low_q=self.low_q,
                            low_q_points=self.low_q_points, porod_points=self.porod_points,
                            porod_qmin=self.porod_qmin, porod_auto=self.porod_auto,
                            fit_background=self.fit_background, threshold=self.threshold, seed=self.seed)
        for k, v in kw.items():
            setattr(p, k, v)
        return p


@dataclass
class AnisoResult:
    params: AnisoParameters
    mode: str
    axis_deg: float
    stretch: float                     # mode A lambda (1 for mode B)
    ls: np.ndarray                     # Legendre orders
    q_l: np.ndarray                    # q of the data components
    I_l: np.ndarray                    # (nq, nl) data components (background subtracted from l = 0)
    sigma_l: np.ndarray
    r: np.ndarray
    C_l: np.ndarray                    # (nl, nr) correlation components, normalised: C_0(0) = 1 -> gamma
    mu: np.ndarray                     # mu grid of the tables
    gamma_rm: np.ndarray               # gamma(r, mu)
    g_rm: np.ndarray                   # g(r, mu)
    levelcut: LevelCut
    invariant: float                   # isotropic invariant Q_0 = integral q^2 I_0 dq
    drho2: float
    K0: float
    B: float
    model_ls: np.ndarray = None
    I_l_model: np.ndarray = None       # (nq, nl_model) level-cut model components at q_l
    chi2_levelcut: float = np.nan
    q_voxel: np.ndarray | None = None
    I_l_voxel: np.ndarray | None = None
    matrix: np.ndarray | None = None
    psi: np.ndarray | None = None
    level: float = np.nan
    statistics: dict = dc_field(default_factory=dict)
    S_over_V_porod: float = np.nan
    clipped: int = 0
    negative_power: float = 0.0
    collapse: dict = dc_field(default_factory=dict)
    data: Data2D | None = None
    warnings: list = dc_field(default_factory=list)
    timings: dict = dc_field(default_factory=dict)

    @property
    def voxel_size(self):
        return self.params.box_size / self.params.n

    def gamma(self, r_par, r_perp):
        """gamma at (r_par, r_perp) (bilinear in the (r, mu) table)."""
        return _bilinear(self.r, self.mu, self.gamma_rm, np.hypot(r_par, r_perp), np.abs(r_par))

    def g(self, r_par, r_perp):
        """g of the Gaussian field at (r_par, r_perp)."""
        return _bilinear(self.r, self.mu, self.g_rm, np.hypot(r_par, r_perp), np.abs(r_par))

    def model_intensity(self, qx, qy):
        """Level-cut model I at detector points (box frame rotated by the axis angle)."""
        qp, qs = axis_coordinates(np.asarray(qx, float), np.asarray(qy, float), self.axis_deg)
        q = np.hypot(qp, qs)
        mu = np.where(q > 0, np.abs(qp) / np.where(q > 0, q, 1), 1.0)
        return sum(np.interp(q, self.q_l, self.I_l_model[:, i]) * eval_legendre(l, mu)
                   for i, l in enumerate(self.model_ls))

    def report(self):
        p, st = self.params, self.statistics
        lines = ["SASmorph – anisotropic (uniaxial) reconstruction",
                 f"  model                {'A, affine stretch' if self.mode == 'A' else 'B, uniaxial (Legendre)'}",
                 f"  symmetry axis        {self.axis_deg:.2f} deg from q_x towards q_y (box x axis)"]
        if self.mode == "A":
            lines.append(f"  stretch lambda       {self.stretch:.4f} along the axis"
                         + (f" (collapse misfit {self.collapse.get('misfit', np.nan):.4g})" if self.collapse else ""))
        else:
            lines.append(f"  Legendre orders      l = {', '.join(str(int(x)) for x in self.ls)}")
        lines += [f"  invariant (l = 0)    Q0 = {self.invariant:.5g}, 3D invariant 4 pi Q0",
                  f"  Porod                K0 = {self.K0:.4g}" + (f", background B = {self.B:.4g}" if self.B else ""),
                  f"  contrast (from Q0)   drho^2 = {self.drho2:.5g}",
                  f"  porosity             {p.porosity:.4f}",
                  f"  S/V (Porod)          {self.S_over_V_porod:.5g}",
                  f"  clipped gamma values {self.clipped}",
                  f"  model vs data        chi2 (level-cut, 2D) = {self.chi2_levelcut:.4g}"]
        if self.matrix is not None:
            lines += [f"  box                  L = {p.box_size:.5g}, n = {p.n}, voxel a = {self.voxel_size:.4g}",
                      f"  negative power       {100 * self.negative_power:.2f} % discarded"]
        if st:
            lines += [f"  S/V (voxels)         {st.get('S_over_V_voxels', np.nan):.5g}",
                      "  interface normals    faces normal to x (axis) / y / z: "
                      + " / ".join(f"{v:.3f}" for v in st.get("face_fractions", ())) + "  (isotropic 1/3 each)"]
            ch = st.get("chords", {})
            for ph in ("pore", "solid"):
                if ph in ch:
                    lines.append(f"  mean chord {ph:5s}     along axis {ch[ph][0]:.4g}, across {ch[ph][1]:.4g} / {ch[ph][2]:.4g}")
            for ph in ("pore", "solid"):
                d = st.get(ph, {})
                if d:
                    lines.append(f"  {ph:5s} clusters       {d['clusters']}, largest {100 * d['largest_fraction']:.1f} %, "
                                 f"percolates x/y/z {''.join('+' if v else '-' for v in d['percolates'])}")
        if self.warnings:
            lines.append("Warnings:")
            lines += [f"  - {w}" for w in self.warnings]
        return "\n".join(lines)


# ---------------------------------------------------------------------------------------------- mode A
def _collapse_misfit(qp, qs, I, s, lam, nbins=60):
    qe = np.sqrt((lam * qp) ** 2 + qs ** 2)
    ok = I > 0
    qe, li = qe[ok], np.log(I[ok])
    w = np.ones_like(li) if s is None else (I[ok] / s[ok]) ** 2
    edges = np.quantile(qe, np.linspace(0, 1, nbins + 1))
    edges = np.unique(edges)
    idx = np.clip(np.digitize(qe, edges) - 1, 0, edges.size - 2)
    sw = np.bincount(idx, weights=w, minlength=edges.size - 1)
    mean = np.bincount(idx, weights=w * li, minlength=edges.size - 1) / np.where(sw > 0, sw, 1)
    return float(np.sum(w * (li - mean[idx]) ** 2) / np.sum(w))


def fit_affine(data: Data2D, axis_deg, fit_axis=True, stretch=None, nbins=60):
    """lambda (and the axis angle) collapsing log I onto one curve of q_eff = sqrt(lambda^2 q_par^2 +
    q_perp^2). Returns (lambda, axis_deg, misfit)."""
    from scipy.optimize import minimize
    s = data.sigma

    def obj(x):
        lam = np.exp(x[0]) if stretch is None else stretch
        ax = x[-1] if fit_axis else axis_deg
        qp, qs = axis_coordinates(data.qx, data.qy, ax)
        return _collapse_misfit(qp, qs, data.I, s, lam, nbins)
    x0 = []
    if stretch is None:
        grid = np.linspace(np.log(0.2), np.log(5.0), 33)
        qp, qs = axis_coordinates(data.qx, data.qy, axis_deg)
        vals = [_collapse_misfit(qp, qs, data.I, s, np.exp(t), nbins) for t in grid]
        x0.append(grid[int(np.argmin(vals))])
    if fit_axis:
        x0.append(axis_deg)
    if not x0:
        return float(stretch), float(axis_deg), obj([])
    res = minimize(obj, np.array(x0), method="Nelder-Mead", options={"xatol": 1e-4, "fatol": 1e-12, "maxiter": 400})
    lam = float(np.exp(res.x[0])) if stretch is None else float(stretch)
    ax = float(res.x[-1]) if fit_axis else float(axis_deg)
    return lam, ax % 180.0, float(res.fun)


def _collapsed_curve(data, lam, axis_deg, nbins=60):
    qp, qs = axis_coordinates(data.qx, data.qy, axis_deg)
    qe = np.sqrt((lam * qp) ** 2 + qs ** 2)
    Ii = data.I / lam
    si = None if data.sigma is None else data.sigma / lam
    edges = np.geomspace(qe.min(), qe.max() * 1.000001, nbins + 1)
    idx = np.digitize(qe, edges) - 1
    w = np.ones_like(Ii) if si is None else 1.0 / si ** 2
    q, I, s = [], [], []
    for b in range(nbins):
        m = idx == b
        if m.sum() < 2:
            continue
        ww = w[m]
        q.append(np.average(qe[m], weights=ww))
        I.append(np.average(Ii[m], weights=ww))
        s.append(np.sqrt(1.0 / ww.sum()) if si is not None else np.std(Ii[m]) / np.sqrt(m.sum()))
    return np.array(q), np.array(I), np.array(s)


# ---------------------------------------------------------------------------------------------- common
def _r_grid(L, n, qmax, extra=1.0):
    a = L / n
    rmax = max(np.sqrt(3) * 0.5 * L * 1.02 * extra, 2 * a)
    dr = min(a / 4.0, np.pi / (16.0 * qmax))
    nr = int(np.clip(np.ceil(rmax / dr), 200, 6000))
    return np.linspace(0.0, rmax, nr + 1)


def _suggest_box(q, I, n):
    qs = q[np.argmax(q ** 2 * I)]
    return float(min(5 * 2 * np.pi / qs, n * 2 * np.pi / q.max()))


def _mu_grid(m=65):
    return np.linspace(0.0, 1.0, m)


def _gauss_mu(m=64):
    """Gauss-Legendre nodes and weights on [0, 1] (projections onto P_l of even functions)."""
    x, w = np.polynomial.legendre.leggauss(m)
    return 0.5 * (x + 1), 0.5 * w


def _model_components(r, c_of_mu, q, drho2, lmax):
    """Legendre components of the level-cut model intensity; ``c_of_mu(mu)`` gives c(r, mu) = F(g(r, mu))
    on the r grid for an array of mu (Gauss quadrature in mu: the projection of a function with a cusp at
    r = 0 needs it)."""
    ls = _ls(lmax)
    mu, wmu = _gauss_mu()
    c_rm = c_of_mu(mu)
    # projection c_l(r) = (2l+1) integral_0^1 c P_l dmu (even functions)
    cl = np.array([(2 * l + 1) * np.sum(c_rm * (eval_legendre(l, mu) * wmu)[None, :], axis=1) for l in ls])
    out = np.zeros((q.size, ls.size))
    for i, l in enumerate(ls):
        if l == 0:
            out[:, i] = levelcut_intensity(q, r, cl[0], drho2)
        else:
            # I_l(q) = drho2 4 pi (-1)^(l/2) integral r^2 c_l(r) j_l(qr) dr  (trapezoid on the r grid)
            jl = spherical_jn(l, np.outer(q, r))
            out[:, i] = drho2 * 4 * np.pi * (-1) ** (l // 2) * np.trapezoid((r ** 2 * cl[i])[None, :] * jl, r, axis=1)
    return ls, out


def correlations_aniso(data: Data2D, params: AnisoParameters | None = None):
    """Everything up to g(r, mu) and the level-cut model (fast): an :class:`AnisoResult` without field."""
    p = AnisoParameters() if params is None else params
    if p.mode not in MODES:
        raise ValueError(f"mode {p.mode!r}: 'A' (affine) or 'B' (uniaxial)")
    t0 = time.perf_counter()
    w = []
    d = data.subset(p.qmin, p.qmax)
    if d.I.size < 50:
        raise ValueError("fewer than 50 valid pixels in the q range")
    if p.axis_deg is None:
        phi_max, aniso = estimate_axis(d)
        p.axis_deg = (phi_max + 90.0) % 180.0
        w.append(f"axis not given: {p.axis_deg:.1f} deg, perpendicular to the intensity maximum ({phi_max:.1f} deg; "
                 f"anisotropy {aniso:.3f}) – the fibre convention; layers along the axis need {phi_max:.1f} deg")
    B = 0.0
    if p.background and not p.fit_background:
        d = Data2D(d.qx, d.qy, d.I - float(p.background), d.sigma, d.name, d.path, None, d.q_unit)
        B = float(p.background)
    qmax_data = float(d.q.max())
    lc = LevelCut(p.porosity)
    mu = _mu_grid()
    collapse = {}
    if p.mode == "A":
        lam, ax, mis = fit_affine(d, p.axis_deg, p.fit_axis, p.stretch, p.nbins)
        p.axis_deg = ax
        collapse = {"misfit": mis}
        qc, Ic, sc = _collapsed_curve(d, lam, ax, p.nbins)
        if p.box_size is None:
            p.box_size = _suggest_box(qc, Ic, p.n) * max(1.0, lam)
            w.append(f"box size not given: L = {p.box_size:.4g}")
        # the isotropic reference structure must cover |M^-1 r| up to the box diagonal
        iso = correlations(qc, Ic, sc, p.iso(box_size=p.box_size * max(1.0, 1.0 / lam)))
        w += [f"isotropic reference: {x}" for x in iso.warnings
              if not x.startswith("box size") and not x.startswith("voxel size")]
        r = _r_grid(p.box_size, p.n, qmax_data)

        def gamma_of_mu(m, r=r, lam=lam, iso=iso):
            rr = r[:, None]
            reff = np.sqrt((rr * m[None, :] / lam) ** 2 + (rr * np.sqrt(1 - m[None, :] ** 2)) ** 2)
            return np.interp(reff, iso.r, iso.gamma)
        gamma_rm = gamma_of_mu(mu)
        # components of gamma for plots / the voxel grid
        ls = _ls(p.lmax)
        C_l = np.array([(2 * l + 1) * np.trapezoid(gamma_rm * eval_legendre(l, mu)[None, :], mu, axis=1) for l in ls])
        # data components for the comparison
        B_fit = iso.curve.B * lam
        qp, qs = axis_coordinates(d.qx, d.qy, ax)
        q_l, I_l, s_l, used = legendre_components(d.q, np.abs(qp) / d.q, d.I - B_fit, d.sigma, lmax=p.lmax,
                                                  nbins=p.nbins)
        q_l, I_l, s_l = _full_coverage(q_l, I_l, s_l, used, w)
        Q0, B = iso.invariant, B + B_fit
        pr = q_l >= np.quantile(q_l, 0.85)
        K0 = fit_porod(q_l[pr], I_l[pr, 0])[0]
        drho2 = iso.drho2
        S_V = specific_surface_porod(K0, Q0, p.porosity)
        stretch = lam
    else:
        qp, qs = axis_coordinates(d.qx, d.qy, p.axis_deg)
        q_l, I_l, s_l, used = legendre_components(d.q, np.abs(qp) / d.q, d.I, d.sigma, lmax=p.lmax, nbins=p.nbins)
        q_l, I_l, s_l = _full_coverage(q_l, I_l, s_l, used, w)
        ls = _ls(p.lmax)
        if I_l.shape[0] < 10:
            raise ValueError("too few q bins with data")
        s0 = s_l[:, 0] if data.sigma is not None else None
        curve = extrapolate(q_l, I_l[:, 0], s0, low_q=p.low_q, low_q_points=p.low_q_points,
                            porod_points=p.porod_points, porod_qmin=p.porod_qmin, porod_auto=p.porod_auto,
                            fit_background=p.fit_background, background=None, refine=4)
        if curve.porod_info.get("message"):
            w.append(f"Porod range: {curve.porod_info['message']}")
        I_l = I_l.copy()
        I_l[:, 0] -= curve.B
        B_fit = curve.B
        B += curve.B
        if p.box_size is None:
            p.box_size = _suggest_box(q_l, np.maximum(I_l[:, 0], 0), p.n)
            w.append(f"box size not given: L = {p.box_size:.4g} (5 x 2 pi / q* of the Kratky maximum of I_0)")
        r = _r_grid(p.box_size, p.n, qmax_data)
        Q0 = curve.invariant
        K0 = curve.K
        C = [curve.debye_transform(r) / (2 * np.pi ** 2)]
        pr = (q_l >= curve.porod_range[0]) & (q_l <= curve.porod_range[1])
        qn = curve.q
        q1 = q_l[0]
        for i, l in enumerate(ls[1:], start=1):
            Kl = float(np.mean(I_l[pr, i] * q_l[pr] ** 4)) if pr.any() else 0.0
            # nodes: I_l ~ q^2 below the data, linear in q within, K_l q^-4 beyond (tail analytic)
            In = np.where(qn < q1, I_l[0, i] * (qn / q1) ** 2, np.interp(qn, q_l, I_l[:, i]))
            C.append(spherical_transform(qn, In, l, r, Kl))
        C = np.array(C)
        C_l = C / C[0, 0]

        def gamma_of_mu(m, C_l=C_l, ls=ls):
            return sum(C_l[i][:, None] * eval_legendre(l, m)[None, :] for i, l in enumerate(ls))
        gamma_rm = gamma_of_mu(mu)
        drho2 = contrast_from_invariant(Q0, p.porosity)
        S_V = specific_surface_porod(K0, Q0, p.porosity)
        stretch = 1.0
        if curve.porod_invariant_fraction > 0.2:
            w.append(f"{100 * curve.porod_invariant_fraction:.0f} % of the invariant from the Porod tail: q range too short?")
    g_rm = lc.g_from_gamma(gamma_rm)
    clipped = lc.clipped
    if clipped:
        w.append(f"gamma(r, mu) outside the range of a level-cut field at {clipped} of {gamma_rm.size} points "
                 "(clipped): porosity, background or the uniaxial model may not fit")
    if np.max(np.abs(gamma_rm[-1])) > 0.05:
        w.append("gamma does not decay within half the box diagonal: use a larger box")
    a = p.box_size / p.n
    if np.pi / a < qmax_data:
        w.append(f"voxel size a = {a:.4g} does not resolve q_max = {qmax_data:.4g} (pi/a = {np.pi / a:.4g})")
    model_ls, I_model = _model_components(r, lambda m: lc.F(lc.g_from_gamma(gamma_of_mu(m))), q_l, drho2,
                                          p.lmax_model)
    lc.clipped = clipped
    # chi2 of the model against the 2D data (level-cut ensemble average)
    res = AnisoResult(p, p.mode, float(p.axis_deg), float(stretch), ls, q_l, I_l, s_l, r, C_l, mu, gamma_rm, g_rm,
                      lc, float(Q0), float(drho2), float(K0), float(B), model_ls, I_model,
                      S_over_V_porod=float(S_V), clipped=int(clipped), collapse=collapse, data=d, warnings=w)
    if d.sigma is not None:
        Im = res.model_intensity(d.qx, d.qy)
        inside = (d.q >= q_l[0]) & (d.q <= q_l[-1])
        res.chi2_levelcut = float(np.mean(((Im[inside] - (d.I[inside] - B_fit)) / d.sigma[inside]) ** 2))
    res.timings["correlations"] = time.perf_counter() - t0
    return res


# ---------------------------------------------------------------------------------------------- field
def _grid_vectors(n, a):
    i = np.arange(n)
    x = a * np.where(i <= n // 2, i, i - n)
    return x


def field_from_grid_correlation(gg, rng=None, dtype=np.float32):
    """Unit-variance periodic Gaussian field whose covariance on the grid is ``gg`` (n^3, FFT layout:
    gg[i, j, k] = g at the minimum-image vector of voxel (i, j, k)). Returns (field, negative power share)."""
    rng = np.random.default_rng(rng)
    n = gg.shape[0]
    S = np.fft.rfftn(gg).real
    neg = float(-S[S < 0].sum() / np.abs(S).sum()) if np.any(S < 0) else 0.0
    S = np.clip(S, 0.0, None)
    wn = np.fft.rfftn(rng.standard_normal((n, n, n)))
    psi = np.fft.irfftn(wn * np.sqrt(S), s=(n, n, n), axes=(0, 1, 2))
    psi -= psi.mean()
    psi /= psi.std()
    return psi.astype(dtype), neg


def _bilinear(r, mu, table, R, Rpar):
    """Bilinear interpolation of a (uniform r) x (uniform mu) table at distances R with axial components Rpar;
    0 beyond r[-1]."""
    R = np.asarray(R, float)
    with np.errstate(invalid="ignore", divide="ignore"):
        MU = np.where(R > 0, np.asarray(Rpar, float) / np.where(R > 0, R, 1), 1.0)
    ir = np.clip(R / (r[1] - r[0]), 0, r.size - 1.000001)
    im = np.clip(MU * (mu.size - 1), 0, mu.size - 1.000001)
    i0, j0 = ir.astype(int), im.astype(int)
    fr, fm = ir - i0, im - j0
    out = ((1 - fr) * (1 - fm) * table[i0, j0] + fr * (1 - fm) * table[i0 + 1, j0]
           + (1 - fr) * fm * table[i0, j0 + 1] + fr * fm * table[i0 + 1, j0 + 1])
    return np.where(R > r[-1], 0.0, out)


def _g_on_grid(res: AnisoResult, n, a):
    """g at every minimum-image voxel vector, axis = box x."""
    x = _grid_vectors(n, a)
    X = x[:, None, None]
    R = np.sqrt(X ** 2 + x[None, :, None] ** 2 + x[None, None, :] ** 2)
    return _bilinear(res.r, res.mu, res.g_rm, R, np.abs(np.broadcast_to(X, R.shape)))


def chord_lengths(mat, a):
    """Mean chord length of each phase along the box axes x, y, z (periodic): {'pore': (lx, ly, lz), 'solid': ...},
    and the share of the voxel faces normal to x, y, z."""
    mat = np.asarray(mat, bool)
    n3 = mat.size
    faces = []
    for ax in range(3):
        faces.append(int(np.count_nonzero(mat != np.roll(mat, 1, axis=ax))))
    phi = mat.mean()
    out = {}
    for name, f in (("pore", phi), ("solid", 1 - phi)):
        out[name] = tuple(float(f * n3 * a / (nf / 2.0)) if nf else np.inf for nf in faces)
    tot = sum(faces)
    return out, tuple(float(nf / tot) if tot else np.nan for nf in faces)


def _voxel_components(mat, a, q_edges, lmax):
    """Legendre components of the voxel scattering over all 3D wave vectors (axis = box x)."""
    n = mat.shape[0]
    eta = mat.astype(np.float64)
    eta -= eta.mean()
    P = np.abs(np.fft.fftn(eta)) ** 2 * a ** 3 / n ** 3
    k = 2 * np.pi * np.fft.fftfreq(n, a)
    KX, KY, KZ = np.meshgrid(k, k, k, indexing="ij")
    q = np.sqrt(KX ** 2 + KY ** 2 + KZ ** 2).ravel()
    m = (q > 0) & (q <= q_edges[-1]) & (q >= q_edges[0])
    mu = np.abs(KX.ravel()[m]) / q[m]
    return legendre_components(q[m], mu, P.ravel()[m], None, edges=q_edges, lmax=lmax, min_points=8)


def reconstruct_aniso(data: Data2D, params: AnisoParameters | None = None, keep_field=True, statistics=True,
                      progress=None):
    """Correlations, the anisotropic Gaussian field, the cut, the voxel scattering and the statistics."""
    def tell(x, t):
        if progress is not None:
            progress(x, t)
    res = correlations_aniso(data, params)
    p = res.params
    t0 = time.perf_counter()
    n, a = p.n, p.box_size / p.n
    tell(0.2, "g on the grid")
    gg = _g_on_grid(res, n, a)
    tell(0.4, "Gaussian field")
    psi, neg = field_from_grid_correlation(gg, p.seed)
    res.negative_power = neg
    if neg > 0.02:
        res.warnings.append(f"{100 * neg:.1f} % negative spectral power discarded: g is not a valid correlation "
                            "(box too small, or the anisotropic gamma cannot come from a level-cut field)")
    lev = cut_level(psi, p.porosity, p.threshold)
    mat = psi > lev
    res.matrix, res.level = mat, float(lev)
    res.psi = psi if keep_field else None
    tell(0.7, "voxel scattering")
    lo, hi = float(res.q_l[0]), float(min(res.q_l[-1], np.pi / a))
    if hi > lo:
        edges = np.geomspace(lo, hi, min(p.nbins, 40) + 1)
        qv, Iv, _sv, _ = _voxel_components(mat, a, edges, p.lmax)
        res.q_voxel, res.I_l_voxel = qv, Iv * res.drho2
    if statistics:
        tell(0.9, "statistics")
        st = morphology_statistics(mat, a)
        st["chords"], st["face_fractions"] = chord_lengths(mat, a)
        res.statistics = st
    res.timings["reconstruct"] = time.perf_counter() - t0
    tell(1.0, "done")
    return res


def folded_map(res: AnisoResult, which="data", n=101):
    """(q_par edges, q_perp edges, map) on a regular grid of the folded quadrant: 'data' (mean of the
    pixels per cell) or 'model' (level-cut model)."""
    d = res.data
    qp, qs = axis_coordinates(d.qx, d.qy, res.axis_deg)
    qp, qs = np.abs(qp), np.abs(qs)
    qm = float(max(qp.max(), qs.max()))
    e = np.linspace(0.0, qm, n + 1)
    if which == "data":
        H, _, _ = np.histogram2d(qp, qs, bins=[e, e], weights=d.I - res.B)
        N, _, _ = np.histogram2d(qp, qs, bins=[e, e])
        with np.errstate(invalid="ignore", divide="ignore"):
            M = np.where(N > 0, H / N, np.nan)
    else:
        c = 0.5 * (e[1:] + e[:-1])
        P, S = np.meshgrid(c, c, indexing="ij")
        th = np.radians(res.axis_deg)
        qx, qy = P * np.cos(th) - S * np.sin(th), P * np.sin(th) + S * np.cos(th)
        M = res.model_intensity(qx, qy)
        q = np.hypot(P, S)
        M = np.where((q >= res.q_l[0]) & (q <= res.q_l[-1]), M, np.nan)
    return e, e, M
