"""Least-squares fit of pySCAtter model parameters to measured data (pySASred output or any 1D / 2D file the
pyTOOLbox readers understand).

* 1D: I(q) with Idev; the model is folded with the q resolution Qdev (Gaussian) when the file has it.
* 2D: an NXcanSAS 2D image (Qx, Qy, I, Idev, mask); the model is evaluated at the q vector of every used pixel
  (beam along x, detector horizontal = model qy, vertical = model qz, the beam component from the Ewald sphere
  when the wavelength is known).  Pixels can be thinned (every n-th) to keep a fit fast.

Fitted parameters are the numeric parameters with a fit flag (the SasScatter FITFLAG_), bounded by their ranges
in :mod:`params`.  Lengths in nm, q in 1/nm as everywhere in pySCAtter.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

import numpy as np
from scipy.optimize import least_squares

from . import engine
from .params import NUM

__all__ = ["Data1D", "Data2D", "FitResult", "fit", "chi2_reduced", "model_for", "FitAborted"]


class FitAborted(Exception):
    pass


@dataclass
class Data1D:
    q: np.ndarray                    # 1/nm
    I: np.ndarray
    dI: Optional[np.ndarray] = None
    dq: Optional[np.ndarray] = None  # 1/nm, standard deviation of the q resolution
    name: str = ""

    @classmethod
    def from_curve(cls, c):
        """From a :class:`pysasfit.toolbox.merge.Curve` (q already in 1/nm)."""
        dq = np.asarray(c.dq, float) if c.dq is not None and np.any(np.asarray(c.dq) > 0) else None
        return cls(np.asarray(c.q, float), np.asarray(c.I, float), np.asarray(c.dI, float), dq, c.name)

    def used(self, qmin=None, qmax=None):
        m = np.isfinite(self.q) & np.isfinite(self.I) & (self.q > 0)
        if self.dI is not None:
            m &= np.isfinite(self.dI) & (self.dI > 0)
        if qmin is not None:
            m &= self.q >= qmin
        if qmax is not None:
            m &= self.q <= qmax
        return m


@dataclass
class Data2D:
    qh: np.ndarray                   # horizontal q of every pixel, 1/nm (image shape)
    qv: np.ndarray                   # vertical q
    I: np.ndarray
    dI: Optional[np.ndarray] = None
    mask: Optional[np.ndarray] = None    # True = not used
    wavelength_nm: Optional[float] = None
    name: str = ""

    @classmethod
    def from_image(cls, im):
        """From a :class:`pysasfit.toolbox.image2d.Image2D` with known q."""
        qh, qv = im.qxy()
        lam = im.wavelength_A / 10.0 if im.wavelength_A else None
        return cls(np.asarray(qh, float), np.asarray(qv, float), np.asarray(im.I, float),
                   None if im.dI is None else np.asarray(im.dI, float), np.asarray(im.mask, bool), lam, im.name)

    def q_beam(self):
        """Beam component of q on the Ewald sphere (0 without wavelength)."""
        if not self.wavelength_nm:
            return np.zeros_like(self.qh)
        k = 2 * np.pi / self.wavelength_nm
        q2 = self.qh ** 2 + self.qv ** 2
        return -q2 / (2 * k)

    def used(self, qmin=None, qmax=None, every=1):
        m = np.isfinite(self.I)
        if self.mask is not None:
            m &= ~self.mask
        if self.dI is not None:
            m &= np.isfinite(self.dI) & (self.dI > 0)
        q = np.hypot(self.qh, self.qv)
        if qmin is not None:
            m &= q >= qmin
        if qmax is not None:
            m &= q <= qmax
        if every > 1:
            thin = np.zeros_like(m)
            thin[::every, ::every] = True
            m &= thin
        return m


def model_for(p, data, used=None):
    """Model intensity at the data points (all points without ``used``)."""
    if isinstance(data, Data1D):
        q = data.q if used is None else data.q[used]
        dq = None if data.dq is None else (data.dq if used is None else data.dq[used])
        return engine.smeared_1d(p, q, dq)
    qb, qh, qv = data.q_beam(), data.qh, data.qv
    if used is not None:
        return engine.intensity_2d(p, qb[used], qh[used], qv[used])
    return engine.intensity_2d(p, qb, qh, qv)


def _sigma(data, used):
    if data.dI is None:
        return np.maximum(np.abs(data.I[used]) * 0.01, 1e-30)     # no errors: 1 % relative weights
    return data.dI[used]


def chi2_reduced(p, data, used, n_free=0):
    r = (model_for(p, data, used) - data.I[used]) / _sigma(data, used)
    return float(np.sum(r * r) / max(1, r.size - n_free))


@dataclass
class FitResult:
    params: Dict[str, float]
    values: Dict[str, float]
    errors: Dict[str, float]
    chi2r: float
    n_points: int
    nfev: int
    success: bool
    message: str
    correlations: Optional[np.ndarray] = None
    keys: List[str] = field(default_factory=list)

    def report(self) -> str:
        lines = [f"χ²ᵣ = {self.chi2r:.4g}   points {self.n_points}   evaluations {self.nfev}   {self.message}"]
        for k in self.keys:
            e = self.errors.get(k, np.nan)
            lines.append(f"  {k:16s} = {self.values[k]:.6g} ± {e:.2g}")
        if self.correlations is not None and len(self.keys) > 1:
            c = self.correlations
            strong = [(self.keys[i], self.keys[j], c[i, j]) for i in range(len(self.keys))
                      for j in range(i) if abs(c[i, j]) > 0.9]
            for a, b, v in strong:
                lines.append(f"  strongly correlated: {a} / {b}  ({v:+.3f})")
        return "\n".join(lines)


def fit(p, flags, data, qmin=None, qmax=None, every=1, max_nfev=200, callback: Optional[Callable] = None,
        should_abort: Optional[Callable[[], bool]] = None, log_weight=False) -> FitResult:
    """Fit the flagged numeric parameters of ``p`` to ``data`` (Data1D or Data2D).

    ``callback(p, chi2r)`` after every evaluation; ``should_abort()`` stops the fit (FitAborted).
    ``log_weight``: residuals of ln I (relative errors), useful over many decades without good errors."""
    keys = [k for k in NUM if flags.get(k)]
    if not keys:
        raise ValueError("no parameter marked for fitting")
    used = data.used(qmin, qmax) if isinstance(data, Data1D) else data.used(qmin, qmax, every)
    n = int(np.count_nonzero(used))
    if n <= len(keys):
        raise ValueError(f"only {n} data points for {len(keys)} parameters")
    y = data.I[used]
    sig = _sigma(data, used)
    lo = np.array([NUM[k][3] for k in keys], float)
    hi = np.array([NUM[k][4] for k in keys], float)
    x0 = np.clip(np.array([float(p[k]) for k in keys]), lo, hi)
    x0 = np.where(x0 == lo, lo + 1e-9 * np.maximum(1, np.abs(lo)), x0)
    x0 = np.where(x0 == hi, hi - 1e-9 * np.maximum(1, np.abs(hi)), x0)
    work = dict(p)
    best = {"chi2": np.inf}

    def resid(x):
        if should_abort and should_abort():
            raise FitAborted("fit aborted")
        work.update(dict(zip(keys, map(float, x))))
        try:
            m = model_for(work, data, used)
        except engine.CalcError:
            return np.full(n, 1e6)
        if log_weight:
            r = (np.log(np.maximum(m, 1e-300)) - np.log(np.maximum(y, 1e-300))) * np.abs(y) / sig
        else:
            r = (m - y) / sig
        r = np.where(np.isfinite(r), r, 1e6)
        c2 = float(np.sum(r * r) / max(1, n - len(keys)))
        if c2 < best["chi2"]:
            best["chi2"] = c2
        if callback:
            callback(work, c2)
        return r

    res = least_squares(resid, x0, bounds=(lo, hi), x_scale="jac", max_nfev=max_nfev, diff_step=1e-4)
    vals = dict(zip(keys, map(float, res.x)))
    out = dict(p)
    out.update(vals)
    dof = max(1, n - len(keys))
    chi2r = float(np.sum(res.fun ** 2) / dof)
    errs, corr = {k: np.nan for k in keys}, None
    try:
        J = res.jac
        cov = np.linalg.pinv(J.T @ J) * chi2r
        e = np.sqrt(np.clip(np.diag(cov), 0, None))
        errs = dict(zip(keys, map(float, e)))
        with np.errstate(invalid="ignore", divide="ignore"):
            corr = cov / np.outer(e, e)
    except Exception:  # noqa: BLE001 -- errors are optional
        pass
    return FitResult(out, vals, errs, chi2r, n, int(res.nfev), bool(res.success), str(res.message), corr, keys)
