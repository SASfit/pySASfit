# SPDX-License-Identifier: BSD-3-Clause
"""Runs of one sample under the same conditions: grouping by metadata, then a check whether the
runs of a group are statistically equivalent before they are summed.

Grouping follows BerSANS: a run belongs to a group when every selected metadata entry agrees
within its *precision* (sample name exactly, temperature within ±0.5 K, SD within 1 %, ...).

BerSANS then sums all files of the group. :func:`check_runs` adds what is missing there: it
compares the reduced I(q) of the runs with their propagated uncertainties and tells

* **equivalent** -- the runs scatter as their counting statistics allow; summing is safe;
* **outlier** -- one or a few runs differ, the others agree (bubble, wrong position, beam trip);
* **scale drift / scale change** -- the runs differ by a q-independent factor only (fewer
  particles in the beam: sedimentation or creaming of the whole sample, a changed thickness or
  transmission, a normalisation problem); *drift* when the factor follows the measuring time;
* **shape drift / shape change** -- the curve itself changes (aggregation, size-selective
  sedimentation, a structural change); *drift* when a q band follows the measuring time.

Statistics (runs k = 1..n, q bins i, statistical uncertainties s_ki, weights w = 1/s²):

* global χ² of all runs against the weighted mean, n_q (n-1) degrees of freedom;
* leave-one-out χ²_k of run k against the mean of the others, with the covariance of both
  (the full statistical covariance of each run when it is given);
* scale model I_ki = a_k μ_i: χ² with n_q (n-1) - (n-1) degrees of freedom and the factors a_k;
* trends: weighted straight line versus the mid time of each run, for the scale factors and for
  the mean intensity in ``n_bands`` q bands (low q first), reported as relative change per hour
  and as z = slope / σ(slope).

An optional relative floor (``rel_floor``, added in quadrature to every point) accepts known
instabilities below the counting statistics, e.g. 0.3 % beam or monitor drift.
"""
from __future__ import annotations

import datetime as _dt
import re
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Union

import numpy as np
from scipy import stats

__all__ = ["FIELDS", "hdf_field", "metadata", "group_runs", "RunGroup", "Trend", "Equivalence",
           "equivalence", "check_runs", "plot_equivalence", "mid_time"]


# ---------------------------------------------------------------------------- metadata
def _parse_time(s) -> Optional[float]:
    if not s:
        return None
    s = str(s).strip().replace("T", " ")
    for fmt in ("%Y-%m-%d %H:%M:%S.%f", "%Y-%m-%d %H:%M:%S"):
        try:
            return _dt.datetime.strptime(s, fmt).timestamp()
        except ValueError:
            pass
    return None


def mid_time(raw) -> Optional[float]:
    """Middle of the measurement (seconds since the epoch): start + counting time / 2."""
    t0 = _parse_time(getattr(raw, "start_time", None))
    return None if t0 is None else t0 + 0.5 * float(raw.counting_time_s)


#: selectable metadata entries of a run (name -> getter of the raw run)
FIELDS: Dict[str, Callable] = {
    "sample": lambda r: r.sample_name,
    "SD_m": lambda r: r.SD_m,
    "panel_SD_m": lambda r: getattr(r, "panel_SD_m", None),     # SANS-LLB wide-angle panels (None: none)
    "coll_m": lambda r: r.coll_m,
    "wavelength_A": lambda r: r.wavelength_A,
    "dlam_over_lam": lambda r: r.dlam_over_lam_fwhm,
    "attenuator": lambda r: r.attenuator,
    "detector_offset_m": lambda r: r.detector_offset_m[0],
    "detector_tilt_deg": lambda r: r.detector_tilt_deg,
    "beamstop_x_mm": lambda r: r.beamstop_xy_mm[0],
    "beamstop_y_mm": lambda r: r.beamstop_xy_mm[1],
    "beamstop_type": lambda r: getattr(r, "beamstop_type", ""),
    "sample_position_mm": lambda r: r.sample_position_mm,
    "thickness_cm": lambda r: r.sample_thickness_cm,
    "counting_time_s": lambda r: r.counting_time_s,
    "start": lambda r: _parse_time(r.start_time),
}


def hdf_field(path: str, index: int = 0) -> Callable:
    """Getter for any entry of the raw file, e.g. a sample environment that the reader does not
    know yet: ``FIELDS["temperature"] = hdf_field("entry1/sample/temperature")``.
    Arrays give their mean (a temperature log) unless ``index`` is not None."""
    def get(raw):
        import h5py
        with h5py.File(raw.path, "r") as f:
            if path not in f:
                return None
            v = f[path][()]
        if isinstance(v, bytes):
            return v.decode(errors="replace").strip()
        v = np.asarray(v)
        if v.dtype.kind in "SO":
            x = v.ravel()[0]
            return x.decode(errors="replace").strip() if isinstance(x, bytes) else str(x)
        if v.size == 1:
            return float(v.ravel()[0])
        return float(np.mean(v)) if index is None else float(v.ravel()[index])
    return get


def metadata(raw, keys: Optional[Sequence[str]] = None, fields: Optional[Dict[str, Callable]] = None) -> dict:
    """Values of the selected metadata entries of a raw run."""
    f = dict(FIELDS, **(fields or {}))
    out = {}
    for k in keys or f:
        try:
            out[k] = f[k](raw)
        except Exception:  # noqa: BLE001 -- missing optional entry
            out[k] = None
    return out


# ---------------------------------------------------------------------------- grouping
Precision = Union[None, float, str]


def _tol(prec: Precision, v: float) -> float:
    if prec is None:
        return 0.0
    if isinstance(prec, str):
        p = prec.strip()
        if p.endswith("%"):
            return abs(v) * float(p[:-1]) / 100.0
        return float(p)
    return float(prec)


@dataclass
class RunGroup:
    """Runs whose selected metadata agree within the precisions (sorted by start time)."""
    values: dict                  # key -> value (strings) or (min, max) range (numbers)
    runs: list                    # raw runs

    @property
    def run_numbers(self) -> List[int]:
        return [r.run for r in self.runs]

    def label(self) -> str:
        parts = []
        for k, v in self.values.items():
            if isinstance(v, tuple):
                parts.append(f"{k}={v[0]:.4g}" if v[0] == v[1] else f"{k}={v[0]:.4g}..{v[1]:.4g}")
            else:
                parts.append(f"{k}={v}")
        return ", ".join(parts)


def _split(runs, key, prec, get):
    vals = [get(r) for r in runs]
    if all(v is None or isinstance(v, str) for v in vals) or prec in (None, 0, "0"):
        if not all(v is None or isinstance(v, str) for v in vals):
            vals = [None if v is None else round(float(v), 9) for v in vals]
        out: Dict[object, list] = {}
        for r, v in zip(runs, vals):
            out.setdefault(v, []).append(r)
        return [(v if (v is None or isinstance(v, str)) else (v, v), rs) for v, rs in out.items()]
    # numbers within a precision: sorted chain, a group spans at most the precision
    order = sorted(range(len(runs)), key=lambda i: (vals[i] is None, vals[i] if vals[i] is not None else 0))
    groups, cur, start = [], [], None
    for i in order:
        v = vals[i]
        if v is None:
            groups.append((None, [runs[i]]))
            continue
        if cur and v - start > _tol(prec, start) + 1e-12:
            groups.append(((start, float(vals[cur[-1]])), [runs[j] for j in cur]))
            cur = []
        if not cur:
            start = float(v)
        cur.append(i)
    if cur:
        groups.append(((start, float(vals[cur[-1]])), [runs[j] for j in cur]))
    return groups


def group_runs(runs, match: Dict[str, Precision], fields: Optional[Dict[str, Callable]] = None,
               min_runs: int = 1) -> List[RunGroup]:
    """Group raw runs (or catalogue entries) by metadata.

    ``match`` maps a metadata entry to its precision: ``None`` = equal values, a number = the
    largest spread within a group in the unit of the entry, ``"1%"`` = relative spread.
    Example: ``{"sample": None, "SD_m": "1%", "wavelength_A": "2%", "attenuator": None,
    "temperature": 0.5}``. Groups are sorted by their first start time."""
    f = dict(FIELDS, **(fields or {}))
    raws = [getattr(r, "raw", r) for r in runs]
    raws = [r for r in raws if r is not None]
    groups = [({}, raws)]
    for key, prec in match.items():
        if key not in f:
            raise KeyError(f"unknown metadata entry {key!r}; known: {sorted(f)} (add others with hdf_field)")
        get = f[key]

        def safe(r, get=get):
            try:
                return get(r)
            except Exception:  # noqa: BLE001
                return None
        new = []
        for vals, rs in groups:
            for v, sub in _split(rs, key, prec, safe):
                new.append((dict(vals, **{key: v}), sub))
        groups = new
    out = []
    for vals, rs in groups:
        if len(rs) < min_runs:
            continue
        rs = sorted(rs, key=lambda r: (mid_time(r) or 0.0, r.run))
        out.append(RunGroup(vals, rs))
    return sorted(out, key=lambda g: (mid_time(g.runs[0]) or 0.0, g.runs[0].run))


# ---------------------------------------------------------------------------- statistics
@dataclass
class Trend:
    """Weighted straight line y = a + b (t - t̄) of one quantity versus time."""
    name: str
    t: np.ndarray                 # hours from the first run
    y: np.ndarray                 # relative to the weighted mean
    sy: np.ndarray
    slope_per_h: float            # relative change per hour
    sigma_slope: float
    z: float
    chi2_const: float             # constant model, n-1 degrees of freedom
    chi2_line: float              # straight line, n-2 degrees of freedom

    @property
    def significant(self) -> bool:
        """``abs(z) > 3`` (with two runs any difference is a "trend"; :func:`equivalence` uses trends
        for its verdict only from three runs on)."""
        return abs(self.z) > 3.0


def _trend(name, t_h, y, sy) -> Trend:
    w = 1.0 / sy**2
    ym = np.sum(w * y) / np.sum(w)
    y, sy = y / ym, sy / ym
    w = 1.0 / sy**2
    tm = np.sum(w * t_h) / np.sum(w)
    dt = t_h - tm
    stt = np.sum(w * dt**2)
    c0 = np.sum(w * (y - np.sum(w * y) / np.sum(w))**2)
    if stt <= 0:
        return Trend(name, t_h, y, sy, 0.0, np.inf, 0.0, float(c0), float(c0))
    b = np.sum(w * dt * y) / stt
    a = np.sum(w * y) / np.sum(w)
    sb = 1.0 / np.sqrt(stt)
    c1 = np.sum(w * (y - a - b * dt)**2)
    return Trend(name, t_h, y, sy, float(b), float(sb), float(b / sb), float(c0), float(c1))


@dataclass
class Equivalence:
    """Result of :func:`equivalence`."""
    runs: List[int]
    q: np.ndarray
    I: np.ndarray                 # (n, nq)
    sigma: np.ndarray             # (n, nq), including the floor
    mean: np.ndarray              # weighted mean (nq)
    t_h: np.ndarray               # mid times, hours after the first
    chi2: float
    dof: int
    p: float
    run_chi2: np.ndarray          # leave-one-out, nq degrees of freedom each
    run_p: np.ndarray
    pair_chi2_ndf: np.ndarray     # (n, n)
    scale: np.ndarray             # a_k, mean 1
    scale_sigma: np.ndarray
    scale_chi2: float
    scale_dof: int
    scale_p: float
    scale_trend: Optional[Trend]
    band_trends: List[Trend]
    outliers: List[int]           # run numbers
    verdict: str                  # equivalent | outlier | scale drift | scale change | shape drift | shape change
    message: str
    alpha: float
    accepted: List[int] = field(default_factory=list)

    @property
    def chi2_ndf(self) -> float:
        return self.chi2 / max(self.dof, 1)

    def summary(self) -> str:
        lines = [f"runs {self.runs}: {self.verdict} -- {self.message}",
                 f"  all runs: chi2/ndf = {self.chi2_ndf:.3f} (ndf {self.dof}, p = {self.p:.3g})",
                 f"  scale model: chi2/ndf = {self.scale_chi2 / max(self.scale_dof, 1):.3f} "
                 f"(p = {self.scale_p:.3g})"]
        for k, r in enumerate(self.runs):
            lines.append(f"  run {r}: t = {self.t_h[k]:6.2f} h  chi2/ndf vs others = "
                         f"{self.run_chi2[k] / len(self.q):.3f} (p = {self.run_p[k]:.3g})  "
                         f"scale = {self.scale[k]:.4f} ± {self.scale_sigma[k]:.4f}"
                         + ("  OUTLIER" if r in self.outliers else ""))
        for tr in ([self.scale_trend] if self.scale_trend else []) + self.band_trends:
            lines.append(f"  trend {tr.name}: {100 * tr.slope_per_h:+.3f} %/h ± {100 * tr.sigma_slope:.3f} "
                         f"(z = {tr.z:+.1f})")
        return "\n".join(lines)


def _mean_and_cov(I, w, C, idx):
    W = w[idx]
    sw = W.sum(axis=0)
    m = (W * I[idx]).sum(axis=0) / sw
    if C is None:
        return m, np.diag(1.0 / sw)
    A = W / sw
    cov = sum(np.outer(a, a) * C[j] for a, j in zip(A, idx))
    return m, cov


def _chi2(r, cov):
    try:
        return float(r @ np.linalg.solve(cov, r))
    except np.linalg.LinAlgError:
        return float(np.sum(r**2 / np.diag(cov)))


def _verdict(p, ok_after_removal, outl, scale_p, scale_tr, band_tr, alpha, n):
    if p >= alpha:
        return "equivalent", "the runs agree within their counting statistics; summing is safe"
    if outl and ok_after_removal:
        return "outlier", (f"run(s) {outl} differ, the others agree; sum without them or inspect them "
                           "(bubble, sample position, beam trip)")
    if scale_p >= alpha:
        if n >= 3 and scale_tr is not None and scale_tr.significant:
            return "scale drift", (f"same shape, intensity changes by {100 * scale_tr.slope_per_h:+.2f} %/h: "
                                   "fewer scatterers in the beam (sedimentation, creaming) or a drifting "
                                   "normalisation; sum only runs of the same state or correct per run")
        two = " (two runs: a step and a trend cannot be told apart)" if n == 2 else " without a time trend"
        return "scale change", (f"same shape, different intensity{two}: sedimentation or creaming, "
                                "thickness, transmission, filling or normalisation")
    drifting = [t for t in band_tr if t.significant] if n >= 3 else []
    if drifting:
        d = ", ".join(f"{t.name} {100 * t.slope_per_h:+.2f} %/h" for t in drifting)
        return "shape drift", (f"the curve changes with time ({d}); low q changing most points to "
                               "aggregation or size-selective sedimentation; do not sum blindly")
    if n == 2:
        big = max(band_tr, key=lambda t: abs(t.z), default=None)
        where = f", largest in {big.name} ({100 * big.slope_per_h:+.2f} %/h)" if big is not None else ""
        return "shape change", (f"the two curves differ in shape{where}; aggregation, size-selective "
                                "sedimentation or a different sample position; do not sum blindly")
    return "shape change", "the curves differ in shape without a monotonic trend; inspect the runs"


def equivalence(q, I, sigma, t, runs=None, cov=None, n_bands: int = 3, alpha: float = 0.01,
                rel_floor: float = 0.0) -> Equivalence:
    """Statistical comparison of repeated measurements I (n_runs, nq) ± sigma at times t (s).

    ``cov`` (n_runs, nq, nq), optional: the statistical covariance of each run (used for the
    leave-one-out χ²; the global and scale χ² use the diagonal). Bins that are not finite or have
    sigma <= 0 in any run are left out."""
    I = np.asarray(I, float)
    sigma = np.asarray(sigma, float)
    q = np.asarray(q, float)
    n = I.shape[0]
    if n < 2:
        raise ValueError("at least two runs are needed")
    runs = list(runs) if runs is not None else list(range(1, n + 1))
    good = np.all(np.isfinite(I) & np.isfinite(sigma) & (sigma > 0), axis=0)
    if good.sum() < 3:
        raise ValueError("fewer than three usable q bins")
    q, I, s = q[good], I[:, good], sigma[:, good]
    s = np.sqrt(s**2 + (rel_floor * np.abs(I))**2)
    C = None
    if cov is not None:
        C = np.asarray(cov, float)[:, good][:, :, good]
        if rel_floor:
            C = C + np.array([np.diag((rel_floor * Ik)**2) for Ik in I])
    nq = q.size
    w = 1.0 / s**2
    mean = (w * I).sum(axis=0) / w.sum(axis=0)
    chi2 = float(np.sum(w * (I - mean)**2))
    dof = nq * (n - 1)
    p = float(stats.chi2.sf(chi2, dof))

    # leave one out
    run_chi2 = np.zeros(n)
    for k in range(n):
        others = [j for j in range(n) if j != k]
        m, cm = _mean_and_cov(I, w, C, others)
        ck = C[k] if C is not None else np.diag(s[k]**2)
        run_chi2[k] = _chi2(I[k] - m, ck + cm)
    run_p = stats.chi2.sf(run_chi2, nq)
    pair = np.zeros((n, n))
    for a in range(n):
        for b in range(a + 1, n):
            pair[a, b] = pair[b, a] = np.sum((I[a] - I[b])**2 / (s[a]**2 + s[b]**2)) / nq

    # scale model I_k = a_k mu
    a = np.ones(n)
    for _ in range(50):
        mu = (w * a[:, None] * I).sum(axis=0) / (w * a[:, None]**2).sum(axis=0)
        a_new = (w * I * mu).sum(axis=1) / (w * mu**2).sum(axis=1)
        a_new /= a_new.mean()
        if np.max(np.abs(a_new - a)) < 1e-12:
            a = a_new
            break
        a = a_new
    mu = (w * a[:, None] * I).sum(axis=0) / (w * a[:, None]**2).sum(axis=0)
    a_sig = 1.0 / np.sqrt((w * mu**2).sum(axis=1))
    scale_chi2 = float(np.sum(w * (I - a[:, None] * mu)**2))
    scale_dof = max(dof - (n - 1), 1)
    scale_p = float(stats.chi2.sf(scale_chi2, scale_dof))

    # trends with time
    t = np.asarray([np.nan if x is None else x for x in t], float)
    trends_ok = np.all(np.isfinite(t)) and np.ptp(t) > 0
    t_h = (t - np.nanmin(t)) / 3600.0 if np.all(np.isfinite(t)) else np.full(n, np.nan)
    scale_tr, band_tr = None, []
    if trends_ok:
        scale_tr = _trend("scale", t_h, a, a_sig)
        order = np.argsort(q)
        for bi, idx in enumerate(np.array_split(order, max(1, min(n_bands, nq)))):
            J = I[:, idx].mean(axis=1)
            sJ = np.sqrt((s[:, idx]**2).sum(axis=1)) / idx.size
            name = f"q {q[idx].min():.3g}-{q[idx].max():.3g}"
            band_tr.append(_trend(name, t_h, J, sJ))

    # outliers: remove the worst run while it fails the leave-one-out test (Bonferroni) against
    # the remaining runs; it is an outlier verdict when the rest then agree
    keep, bad = list(range(n)), []
    while len(keep) > 2 and len(bad) < (n - 1) // 2:
        loo = []
        for k in keep:
            m, cm = _mean_and_cov(I, w, C, [j for j in keep if j != k])
            ck = C[k] if C is not None else np.diag(s[k]**2)
            loo.append(_chi2(I[k] - m, ck + cm))
        worst = int(np.argmax(loo))
        if stats.chi2.sf(loo[worst], nq) >= alpha / len(keep):
            break
        bad.append(keep.pop(worst))
    outliers, ok_after = [], False
    if bad:
        mk = (w[keep] * I[keep]).sum(0) / w[keep].sum(0)
        r2 = float(np.sum(w[keep] * (I[keep] - mk)**2))
        ok_after = stats.chi2.sf(r2, nq * (len(keep) - 1)) >= alpha
        outliers = [runs[k] for k in sorted(bad)]
    verdict, msg = _verdict(p, ok_after, outliers, scale_p, scale_tr, band_tr, alpha, n)
    if verdict == "outlier":
        order = list(np.argsort(t)) if np.all(np.isfinite(t)) else list(range(n))
        pos = sorted(order.index(k) for k in bad)
        if pos == list(range(len(bad))):
            msg += "; they are the first runs in time: the sample may have changed before it settled"
        elif pos == list(range(n - len(bad), n)):
            msg += "; they are the last runs in time: the sample may have started to change"
        if scale_p >= alpha:
            msg += "; the curves differ by a factor only"
    if verdict == "equivalent":
        accepted = list(runs)
    elif verdict == "outlier":
        accepted = [r for r in runs if r not in outliers]
    else:
        accepted = []
    return Equivalence(runs, q, I, s, mean, t_h, chi2, dof, p, run_chi2, run_p, pair, a, a_sig,
                       scale_chi2, scale_dof, scale_p, scale_tr, band_tr, outliers, verdict, msg,
                       alpha, accepted)


def check_runs(raws, beam_center_px, edges, mask=None, instrument=None, efficiency=None,
               normalise_by: str = "monitor", n_bands: int = 3, alpha: float = 0.01,
               rel_floor: float = 0.0, full_covariance: bool = True) -> Equivalence:
    """Reduce every run of a group with the same centre, mask and q bins and compare them with
    :func:`equivalence`. ``raws``: raw runs or a :class:`RunGroup`."""
    from .reduce import reduce_run_1d
    raws = raws.runs if isinstance(raws, RunGroup) else list(raws)
    red = [reduce_run_1d(r, beam_center_px, edges, instrument=instrument, mask=mask,
                         normalise_by=normalise_by, efficiency=efficiency) for r in raws]
    nq = min(len(r.q) for r in red)
    if any(len(r.q) != nq for r in red):
        raise ValueError("the runs give different q bins (different masks or geometry?)")
    I = np.array([r.I for r in red])
    s = np.array([r.idev() for r in red])
    q = np.mean([r.q for r in red], axis=0)
    cov = np.array([r.intensity.covariance(systematic=False) for r in red]) if full_covariance else None
    return equivalence(q, I, s, [mid_time(r) for r in raws], runs=[r.run for r in raws], cov=cov,
                       n_bands=n_bands, alpha=alpha, rel_floor=rel_floor)


def plot_equivalence(e: Equivalence, fig=None):
    """Three panels: I_k / weighted mean versus q, intensity of the q bands and the scale factor
    versus time, leave-one-out χ²/ndf per run."""
    import matplotlib.pyplot as plt
    if fig is None:
        fig = plt.figure(figsize=(11, 3.6))
    ax1, ax2, ax3 = fig.subplots(1, 3)
    for k, r in enumerate(e.runs):
        ax1.errorbar(e.q, e.I[k] / e.mean, e.sigma[k] / e.mean, fmt=".", ms=3, lw=0.6,
                     label=f"{r} ({e.t_h[k]:.1f} h)")
    ax1.axhline(1, color="k", lw=0.6)
    ax1.set_xscale("log")
    ax1.set_xlabel("q")
    ax1.set_ylabel("I / weighted mean")
    ax1.legend(fontsize=7)
    for tr in ([e.scale_trend] if e.scale_trend else []) + e.band_trends:
        ax2.errorbar(tr.t, tr.y, tr.sy, fmt="o-", ms=3, lw=0.8, label=f"{tr.name} (z={tr.z:+.1f})")
    ax2.set_xlabel("time [h]")
    ax2.set_ylabel("relative intensity")
    ax2.legend(fontsize=7)
    ax3.bar([str(r) for r in e.runs], e.run_chi2 / len(e.q),
            color=["#C8283C" if r in e.outliers else "#5E7388" for r in e.runs])
    ax3.axhline(stats.chi2.isf(e.alpha / len(e.runs), len(e.q)) / len(e.q), color="k", ls="--", lw=0.8)
    ax3.set_ylabel("χ²/ndf vs. others")
    ax3.tick_params(axis="x", rotation=45)
    fig.suptitle(f"{e.verdict}: χ²/ndf = {e.chi2_ndf:.2f}, p = {e.p:.2g}", fontsize=10)
    fig.tight_layout()
    return fig


_RX = re.compile(r"^(?P<k>[^=]+)=(?P<v>.*)$")


def main(argv=None):  # pragma: no cover -- thin CLI
    """``pysasred-groups RAW_FOLDER sample SD_m=1% wavelength_A=2% attenuator`` lists the groups."""
    import argparse

    from .catalogue import scan_folder
    ap = argparse.ArgumentParser(description="group SANS-I runs by metadata with precisions")
    ap.add_argument("folder")
    ap.add_argument("match", nargs="+", help="entry or entry=precision (number or n%%)")
    ap.add_argument("--min-runs", type=int, default=2)
    a = ap.parse_args(argv)
    match = {}
    for m in a.match:
        x = _RX.match(m)
        match[x["k"] if x else m] = (x["v"] if x["v"].endswith("%") else float(x["v"])) if x else None
    for g in group_runs(scan_folder(a.folder), match, min_runs=a.min_runs):
        print(f"{g.label()}: {g.run_numbers}")


if __name__ == "__main__":  # pragma: no cover
    main()
