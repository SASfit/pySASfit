"""Model-free features of a scattering curve and a check of the data (step 1 of the scout).

:func:`extract` gives the Guinier radius and I(0), the power laws at low and high q, form-factor minima,
an interference peak (hint of a structure factor), a background estimate and an estimate of how well the
stated uncertainties match the point-to-point scatter. The features restrict the candidates and give the
start values and bounds of the fits; ``Features.notes`` are the hints of the data check.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field

import numpy as np

__all__ = ["Features", "extract"]


@dataclass
class Features:
    n: int
    q_min: float
    q_max: float
    rg: float = math.nan  # Guinier radius
    i0: float = math.nan
    guinier_ok: bool = False
    guinier_points: int = 0
    low_q_slope: float = math.nan  # d ln I / d ln q at the lowest q (0 globular, -1 rod, -2 disc)
    high_q_slope: float = math.nan  # at the highest q (-4 Porod), background included
    background: float = 0.0  # estimate of a flat background (0 if the curve still falls at the highest q)
    n_minima: int = 0  # form-factor minima (narrow size distribution)
    q_first_min: float = math.nan
    peak_q: float = math.nan  # interference maximum
    sq_hint: bool = False  # the data suggest a structure factor
    error_scale: float = math.nan  # point-to-point scatter / stated sigma (1: consistent)
    notes: list = field(default_factory=list)

    @property
    def size_range(self):
        """(smallest, largest) length the q range can resolve: 1/q_max ... 2 pi/q_min."""
        return 1.0 / self.q_max, 2.0 * math.pi / self.q_min

    @property
    def size(self):
        """A typical radius: the sphere radius of the Guinier radius, else the geometric middle of the size range."""
        if self.guinier_ok:
            lo, hi = self.size_range
            return min(max(math.sqrt(5.0 / 3.0) * self.rg, 1.5 * lo), hi / 1.5)
        return math.sqrt(self.size_range[0] * self.size_range[1])

    def as_dict(self):
        return asdict(self)

    def text(self):
        f = lambda v, fmt=".4g": "-" if not math.isfinite(v) else format(v, fmt)  # noqa: E731
        lines = [
            f"{self.n} points, q = {self.q_min:.4g} ... {self.q_max:.4g}",
            f"Guinier: Rg = {f(self.rg)}, I(0) = {f(self.i0)} ({self.guinier_points} points, "
            f"{'ok' if self.guinier_ok else 'not reliable'})",
            f"slope d ln I / d ln q: {f(self.low_q_slope, '.2f')} at low q, {f(self.high_q_slope, '.2f')} at high q",
            f"form-factor minima: {self.n_minima}" + (f" (first at q = {self.q_first_min:.4g})" if self.n_minima else ""),
            "interference peak: " + (f"q = {self.peak_q:.4g}" if math.isfinite(self.peak_q) else "none")
            + f"; structure factor suggested: {'yes' if self.sq_hint else 'no'}",
            f"scatter / stated error: {f(self.error_scale, '.2f')}",
        ]
        return "\n".join(lines + [f"note: {n}" for n in self.notes])


def _slope(lq, ly, w):
    A = np.vstack([lq, np.ones_like(lq)]).T * w[:, None]
    return np.linalg.lstsq(A, ly * w, rcond=None)[0][0]


def _guinier(q, I, s):
    """Weighted Guinier fit on q Rg < 1.3, range found iteratively: (Rg, I0, points, ok)."""
    pos = I > 0
    if pos.sum() < 5:
        return math.nan, math.nan, 0, False
    q, I, s = q[pos], I[pos], s[pos]
    m = max(5, q.size // 10)
    rg = i0 = math.nan
    for _ in range(20):
        x, y, w = q[:m] ** 2, np.log(I[:m]), I[:m] / s[:m]
        A = np.vstack([x, np.ones_like(x)]).T * w[:, None]
        a, b = np.linalg.lstsq(A, y * w, rcond=None)[0]
        if a >= 0:  # flat within the noise (range too short) or rising: try a longer range
            if m >= q.size:
                return math.nan, math.nan, m, False
            rg, m = math.nan, min(int(1.5 * m) + 1, q.size)
            continue
        rg, i0 = math.sqrt(-3.0 * a), math.exp(b)
        new = max(5, int(np.searchsorted(q, 1.3 / rg)))
        if new == m:
            break
        m = min(new, q.size)
    if not math.isfinite(rg):
        return math.nan, math.nan, m, False
    return rg, i0, m, bool(q[0] * rg < 1.0 and m >= 5)


def _error_scale(q, I, s):
    """Scatter of every point around the line through its neighbours, in units of its stated error (robust)."""
    if q.size < 8:
        return math.nan
    w = (q[1:-1] - q[:-2]) / (q[2:] - q[:-2])
    d = I[1:-1] - ((1.0 - w) * I[:-2] + w * I[2:])
    var = s[1:-1] ** 2 + ((1.0 - w) * s[:-2]) ** 2 + (w * s[2:]) ** 2
    return float(1.4826 * np.median(np.abs(d / np.sqrt(var))))


def extract(q, I, sigma=None):
    """Features of the curve ``I(q)`` with uncertainties ``sigma`` (see :class:`Features`)."""
    from scipy.signal import find_peaks, savgol_filter

    q, I = np.asarray(q, float), np.asarray(I, float)
    notes = []
    if sigma is None:
        sigma = np.maximum(0.03 * np.abs(I), 1e-3 * np.nanmax(np.abs(I)))
        notes.append("no uncertainties given: 3 % of the intensity assumed; the model ranking depends on the error bars")
    s = np.asarray(sigma, float)
    o = np.argsort(q)
    q, I, s = q[o], I[o], s[o]
    n = q.size
    if n < 10:
        raise ValueError("at least 10 data points are needed")
    f = Features(n=n, q_min=float(q[0]), q_max=float(q[-1]), notes=notes)

    f.rg, f.i0, f.guinier_points, f.guinier_ok = _guinier(q, I, s)

    pos = I > 0
    lq, ly, w = np.log(q[pos]), np.log(I[pos]), (I[pos] / s[pos])
    k = max(5, pos.sum() // 10)
    if pos.sum() >= 2 * k:
        f.low_q_slope = float(_slope(lq[:k], ly[:k], w[:k]))
        k2 = max(5, pos.sum() // 5)
        f.high_q_slope = float(_slope(lq[-k2:], ly[-k2:], w[-k2:]))
        if abs(f.high_q_slope) < 0.3:
            t = max(5, n // 10)
            f.background = float(np.average(I[-t:], weights=1.0 / s[-t:] ** 2))
            notes.append(f"the curve is flat at the highest q: background of about {f.background:.3g} (fitted as a constant)")

    # extrema of the smoothed curve in ln I: minima of the form factor, interference maximum
    y = np.log(np.clip(I, 1e-12 * np.nanmax(I), None))
    win = max(5, (n // 25) | 1)
    ys = savgol_filter(y, win, 2) if n > win + 2 else y
    noise = float(np.median(s[pos] / I[pos])) if pos.any() else 0.1
    prom = max(0.15, 3.0 * noise)
    mins = find_peaks(-ys, prominence=prom)[0]
    maxs = find_peaks(ys, prominence=max(0.08, 3.0 * noise))[0]
    f.n_minima = int(mins.size)
    if mins.size:
        f.q_first_min = float(q[mins[0]])
    first_min = mins[0] if mins.size else n
    lead = [i for i in maxs if i < first_min]  # a maximum before the first minimum cannot come from a form factor
    if lead:
        f.peak_q, f.sq_hint = float(q[lead[0]]), True
        notes.append(f"maximum at q = {f.peak_q:.4g} before the first minimum: interference peak, structure factor tried")
    elif math.isfinite(f.low_q_slope) and f.low_q_slope > 0.05:
        f.sq_hint = True
        notes.append("the intensity falls towards low q: repulsive interaction, structure factor tried")
    if f.sq_hint and f.guinier_ok:
        notes.append("with a structure factor the Guinier radius is only an apparent one")
    if not f.guinier_ok:
        notes.append("no Guinier range (q_min Rg > 1 or too few points): the particle size is not fixed by the low q data")

    f.error_scale = _error_scale(q, I, s)
    if math.isfinite(f.error_scale):
        if f.error_scale > 2.0:
            notes.append(f"the point-to-point scatter is {f.error_scale:.1f} times the stated uncertainty: error bars may be "
                         "underestimated (or sharp features are sampled coarsely); criteria then favour too many parameters")  # fmt: skip
        elif f.error_scale < 0.4:
            notes.append(f"the point-to-point scatter is only {f.error_scale:.2f} of the stated uncertainty: error bars may be "
                         "overestimated (or the data are smoothed / oversampled)")  # fmt: skip
    if np.any(I <= 0):
        notes.append(f"{int(np.sum(I <= 0))} points with I <= 0 (over-subtracted background?)")
    return f
