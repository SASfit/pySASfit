"""Combining SASfit functions into models and fitting them (SASfit manual, ch. 4-5).

A :class:`ScatteringContribution` is one population: a form factor P(q; x),
optionally a size distribution N(x) over one form-factor parameter x and a
structure factor S(q). With a size distribution:

* ``"monodisperse"`` (manual 4.1.1):  I = S(q) ∫ N(x) P(q;x) dx
* ``"decoupling"``   (manual 4.1.2):  I = ∫N P dx + (∫N F dx)² / ∫N dx · (S(q) - 1),
  F = scattering amplitude (the '_f' function)
* ``"lma"`` local monodisperse (4.1.3): I = ∫ N(x) P(q;x) S(q; R_V(x)) dx, with
  the first structure-factor parameter set to the volume-equivalent radius
  R_V = (3V(x)/4π)^(1/3) of the '_v' function
* without a structure factor:          I = ∫ N(x) P(q;x) dx

The x integral uses Gauss-Legendre nodes on the support of N(x) (found
automatically or given as ``x_range``).

A :class:`ModelFunction` sums contributions (with scale factors) and an
optional background (a :class:`pysasfit.core.background.Background` or a
constant), and optionally applies an instrumental resolution from
:mod:`pysasfit.core.resolution` (Rician pinhole, slit, ...).

:class:`Pipeline` holds data sets and model functions and fits named
parameters with ``scipy.optimize.least_squares``::

    pipe = Pipeline()
    d = pipe.add_dataset("test.dat")
    sc = ScatteringContribution("ff_sphere", {"R": 50, "eta": 1},
                                size_distribution="sd_LogNorm", sd_params={"N": 1, "s": 0.1, "p": 1, "mu": 50},
                                distributed="R")
    m = pipe.add_model(ModelFunction([sc], background=0.0), dataset=d)
    res = pipe.fit(m, ["0.sd.N", "0.sd.mu", "0.sd.s", "background"])
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .functions import SASfitFunction, function

__all__ = ["ScatteringContribution", "ModelFunction", "Pipeline", "FitResult"]

APPROACHES = ("monodisperse", "decoupling", "lma")


def _as_function(f):
    if f is None or isinstance(f, SASfitFunction):
        return f
    return function(f)


class ScatteringContribution:
    def __init__(
        self,
        form_factor,
        ff_params=None,
        size_distribution=None,
        sd_params=None,
        structure_factor=None,
        sq_params=None,
        distributed=0,
        approach="monodisperse",
        x_range=None,
        n_x=96,
    ):
        self.ff = _as_function(form_factor)
        self.sd = _as_function(size_distribution)
        self.sq = _as_function(structure_factor)
        self.ff_p = self._vec(self.ff, ff_params)
        self.sd_p = self._vec(self.sd, sd_params) if self.sd else None
        self.sq_p = self._vec(self.sq, sq_params) if self.sq else None
        self.distributed = self._index(self.ff, distributed)
        self.set_approach(approach)
        self.x_range = x_range
        self.n_x = int(n_x)

    @staticmethod
    def _vec(f, params):
        if params is None:
            return f.parameter_vector()
        if isinstance(params, dict):
            return f.parameter_vector(**params)
        return f.parameter_vector(*params)

    @staticmethod
    def _index(f, key):
        if isinstance(key, str):
            if key in f.parameter_names:
                return f.parameter_names.index(key)
            if key.startswith("p") and key[1:].isdigit():
                return int(key[1:])
            raise KeyError(f"{f.name} has no parameter '{key}'")
        return int(key)

    def set_approach(self, approach):
        approach = {"decoupling approximation": "decoupling", "local monodisperse": "lma"}.get(approach, approach)
        if approach not in APPROACHES:
            raise ValueError(f"approach must be one of {APPROACHES}")
        self.approach = approach

    def vector(self, role):
        return {"ff": self.ff_p, "sd": self.sd_p, "sq": self.sq_p}[role]

    def function_for(self, role):
        return {"ff": self.ff, "sd": self.sd, "sq": self.sq}[role]

    # --------------------------------------------------------- integration
    def support(self):
        """Integration range of the size distribution."""
        if self.x_range is not None:
            return tuple(self.x_range)
        x0 = abs(self.ff_p[self.distributed]) or 1.0
        for p in self.sd_p[1:4]:  # typical location parameters of SASfit distributions
            if p > 0:
                x0 = max(x0, p)
        x = np.geomspace(x0 * 1e-4, x0 * 1e3, 1400)
        w = np.abs(self.sd.evaluate_p(x, self.sd_p))
        if not np.any(w > 0):
            raise ValueError("size distribution is zero everywhere; give x_range")
        keep = np.nonzero(w > 1e-12 * w.max())[0]
        lo, hi = x[max(keep[0] - 1, 0)], x[min(keep[-1] + 1, x.size - 1)]
        return lo, hi

    def nodes(self):
        a, b = self.support()
        t, w = np.polynomial.legendre.leggauss(self.n_x)
        # split into 4 panels for skewed distributions
        edges = np.linspace(a, b, 5)
        xs, ws = [], []
        for lo, hi in zip(edges[:-1], edges[1:]):
            xs.append(0.5 * (hi - lo) * t + 0.5 * (hi + lo))
            ws.append(0.5 * (hi - lo) * w)
        return np.concatenate(xs), np.concatenate(ws)

    def _ff_matrix(self, q, x, kind="intensity"):
        M = np.empty((q.size, x.size))
        p = self.ff_p.copy()
        for j, xj in enumerate(x):
            p[self.distributed] = xj
            M[:, j] = self.ff.evaluate_p(q, p, kind)
        return M

    def intensity(self, q):
        q = np.asarray(q, dtype=float)
        S = None
        if self.sq is not None and not (self.sd is not None and self.approach == "lma"):
            S = self.sq.evaluate_p(q, self.sq_p)
        if self.sd is None:
            I = self.ff.evaluate_p(q, self.ff_p)
            if S is not None:
                I = I * S  # manual 4.1.1 without distribution
            return I
        x, w = self.nodes()
        N = self.sd.evaluate_p(x, self.sd_p)
        wN = w * N
        P = self._ff_matrix(q, x)
        if self.sq is None:
            return P @ wN
        if self.approach == "monodisperse":
            return (P @ wN) * S
        if self.approach == "decoupling":
            F = self._ff_matrix(q, x, "amplitude")
            return P @ wN + (F @ wN) ** 2 / np.sum(wN) * (S - 1.0)
        # local monodisperse approximation
        V = np.array([self.ff.volume(xj, self.ff_p, dist=self.distributed) for xj in x])
        RV = np.cbrt(np.abs(V) * 3.0 / (4.0 * np.pi))
        Sx = np.empty_like(P)
        sp = self.sq_p.copy()
        for j, r in enumerate(RV):
            sp[0] = r
            Sx[:, j] = self.sq.evaluate_p(q, sp)
        return (P * Sx) @ wN

    __call__ = intensity

    def __str__(self):
        s = f"form factor: {self.ff.name}"
        if self.sd:
            s += f", size distribution: {self.sd.name} over {self.ff.parameter_names[self.distributed] if self.distributed < len(self.ff.parameter_names) else self.distributed}"
        if self.sq:
            s += f", structure factor: {self.sq.name} ({self.approach})"
        return s


class ModelFunction:
    def __init__(self, contributions=(), scales=None, background=0.0, resolution=None):
        self.contributions = list(contributions)
        self.scales = list(scales) if scales is not None else [1.0] * len(self.contributions)
        self.background = background
        self.resolution = resolution
        self._bg_z = None

    def include(self, contribution, scale=1.0):
        self.contributions.append(contribution)
        self.scales.append(float(scale))
        return len(self.contributions) - 1

    def _background(self, q):
        b = self.background
        if b is None:
            return 0.0
        if np.isscalar(b):
            return float(b)
        if callable(getattr(b, "value", None)):  # pysasfit.core.background.Background
            if self._bg_z is None:
                b.setup(q, np.ones_like(q))
                self._bg_z = b.z0()
            return b.value(q, self._bg_z)
        return b(q)

    def particles(self, q):
        q = np.asarray(q, dtype=float)
        I = np.zeros(q.shape)
        for s, c in zip(self.scales, self.contributions):
            I = I + s * c.intensity(q)
        return I

    def intensity(self, q):
        q = np.asarray(q, dtype=float)
        if self.resolution is not None:
            qe, W = self.resolution.setup(q)
            Ip = W @ self.particles(qe)
        else:
            Ip = self.particles(q)
        return Ip + self._background(q)

    __call__ = intensity


@dataclass
class FitResult:
    names: list
    values: np.ndarray
    errors: np.ndarray
    chi2: float
    chi2_red: float
    q: np.ndarray
    I_fit: np.ndarray
    success: bool
    message: str
    raw: object = field(repr=False, default=None)

    def summary(self):
        lines = [f"chi2 = {self.chi2:.4g}, chi2/(M-n) = {self.chi2_red:.4g}  ({self.message})"]
        for n, v, e in zip(self.names, self.values, self.errors):
            lines.append(f"  {n:20s} = {v:.6g} ± {e:.3g}")
        return "\n".join(lines)


class Pipeline:
    """Data sets, model functions and fitting (port of SASstudio's Pipeline)."""

    def __init__(self):
        self.datasets = []
        self.models = []
        self.assignments = {}  # model index -> dataset index

    def add_dataset(self, data, q_scale=1.0):
        """Add a file (read with :func:`pysasfit.core.load_data`) or a tuple (q, I, sigma[, dq])."""
        if isinstance(data, str):
            from ..core.io import load_data

            q, I, s, dq = load_data(data, q_scale=q_scale)
        else:
            q, I, s, *rest = data
            dq = rest[0] if rest else None
        self.datasets.append({"q": np.asarray(q), "I": np.asarray(I), "sigma": np.asarray(s), "dq": dq})
        return len(self.datasets) - 1

    def add_model(self, model, dataset=None):
        self.models.append(model)
        m = len(self.models) - 1
        if dataset is not None:
            self.assignments[m] = dataset
        return m

    def assign(self, model, dataset):
        self.assignments[model] = dataset

    # ----------------------------------------------------------- parameters
    def _address(self, model, addr):
        """'background', 'scale.<i>', or '<i>.<ff|sd|sq>.<name or p<k>>' -> (getter, setter)."""
        m = self.models[model]
        if addr == "background":
            if not np.isscalar(m.background):
                raise ValueError("only a constant background can be fitted this way")
            return (lambda: m.background, lambda v: setattr(m, "background", float(v)))
        parts = addr.split(".")
        if parts[0] == "scale":
            i = int(parts[1])
            return (lambda: m.scales[i], lambda v: m.scales.__setitem__(i, float(v)))
        i, role, key = int(parts[0]), parts[1], ".".join(parts[2:])
        c = m.contributions[i]
        f = c.function_for(role)
        if f is None:
            raise KeyError(f"contribution {i} has no {role}")
        k = ScatteringContribution._index(f, key)
        vec = c.vector(role)
        return (lambda: vec[k], lambda v: vec.__setitem__(k, float(v)))

    def get(self, addr, model=0):
        return self._address(model, addr)[0]()

    def set(self, addr, value, model=0):
        self._address(model, addr)[1](value)

    def evaluate(self, model=0, q=None):
        if q is None:
            q = self.datasets[self.assignments[model]]["q"]
        return self.models[model].intensity(q)

    def fit(self, model, parameters, bounds=None, q_range=None, scale_errors=True, **kw):
        """Least-squares fit of ``parameters`` (addresses, see :meth:`_address`) of model ``model``.

        Errors are sqrt(diag(J^T J)^-1), multiplied by sqrt(chi2/(M-n)) if
        ``scale_errors`` (SASfit reports them without that factor).
        """
        from scipy.optimize import least_squares

        d = self.datasets[self.assignments[model]]
        q, I, s = d["q"], d["I"], d["sigma"]
        if q_range is not None:
            sel = (q >= q_range[0]) & (q <= q_range[1])
            q, I, s = q[sel], I[sel], s[sel]
        acc = [self._address(model, a) for a in parameters]
        x0 = np.array([g() for g, _ in acc], dtype=float)
        lb, ub = (np.full(x0.size, -np.inf), np.full(x0.size, np.inf)) if bounds is None else map(np.asarray, bounds)

        def resid(x):
            for (_, setter), v in zip(acc, x):
                setter(v)
            return (self.models[model].intensity(q) - I) / s

        r = least_squares(resid, x0, bounds=(lb, ub), x_scale="jac", **kw)
        resid(r.x)
        chi2 = float(np.sum(r.fun**2))
        dof = max(q.size - x0.size, 1)
        try:  # column-scaled covariance: parameters may differ by many orders of magnitude
            J = np.asarray(r.jac)
            dcol = np.linalg.norm(J, axis=0)
            dcol[dcol == 0] = 1.0
            Js = J / dcol
            cov = np.linalg.pinv(Js.T @ Js) / np.outer(dcol, dcol) * (chi2 / dof if scale_errors else 1.0)
            err = np.sqrt(np.clip(np.diag(cov), 0, None))
        except np.linalg.LinAlgError:
            err = np.full(x0.size, np.nan)
        return FitResult(
            list(parameters), r.x, err, chi2, chi2 / dof, q, self.models[model].intensity(q), r.success, r.message, r
        )

    def __str__(self):
        lines = ["Pipeline:"]
        for i, m in enumerate(self.models):
            lines.append(f" model {i} (data set {self.assignments.get(i)}), background {m.background}")
            for j, (s, c) in enumerate(zip(m.scales, m.contributions)):
                lines.append(f"   [{j}] x{s:g}: {c}")
        for i, d in enumerate(self.datasets):
            lines.append(f" data set {i}: {d['q'].size} points, q = {d['q'].min():.4g} .. {d['q'].max():.4g}")
        return "\n".join(lines)
