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

The x integral is done on the support of N(x) (found automatically or given as
``x_range``) with Gauss-Legendre nodes (default) or another method of
:mod:`pysasfit.studio.integration` (``method``, ``n_x``, ``rtol``, ``atol``,
``limit``; scipy ``quad_vec``, ``simpson``, ...). The integrals inside the SASfit
functions follow :func:`pysasfit.studio.library.configure_precision`.

A :class:`ModelFunction` sums contributions (with scale factors) and an
optional background (a :class:`pysasfit.core.background.Background` or a
constant), and optionally applies an instrumental resolution from
:mod:`pysasfit.core.resolution` (Rician pinhole, slit, ...).

``transform`` (SASfit's transform menu, :mod:`pysasfit.studio.hankel`) turns the
scattering of the particles into what an instrument measures:

* ``None`` -- I(q) itself;
* a :class:`~pysasfit.studio.hankel.SESANSTransform` -- the model is evaluated at
  spin-echo lengths δ (passed instead of q) and gives G(δ) - G(0) or P/P0; the
  constant background is an offset of G(δ) - G(0), the power-law term is unused;
* a :class:`~pysasfit.studio.hankel.MSASTransform` -- multiple small-angle
  scattering of a sample of the given thickness, before resolution and background.

A transform may also be given as a dictionary or name for
:func:`~pysasfit.studio.hankel.make_transform`, e.g.
``transform={"kind": "msas", "thickness": 0.1, "wavelength": 0.6}``.

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

import math
from dataclasses import dataclass, field

import numpy as np

from .functions import SASfitFunction, function
from .integration import size_integral

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
        method="gauss-legendre",
        rtol=1e-6,
        atol=0.0,
        limit=200,
        x_power=6.0,
        tail=1e-10,
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
        self.method = method  # integration over the size distribution, see pysasfit.studio.integration
        self.rtol = float(rtol)
        self.atol = float(atol)
        self.limit = int(limit)
        self.x_power = float(x_power)  # the range search weighs N(x) with x**x_power (form factor ~ x^2 ... x^6)
        self.tail = float(tail)  # ... and cuts where N(x) x**x_power falls below tail * its maximum
        self.last_integration = None  # info of the last size-distribution integration (method, neval, ...)
        self.cache = None  # ContributionCache while a fit runs with cache=True (see FitProblem.fit)
        self.snap = 0.0  # > 0: the automatic integration range is rounded outwards to a geometric grid of this step
        self.fixed_grid = False  # True: fixed logarithmic grid for the size-distribution integral (FitProblem.fit(grid=True))
        self.grid_margin = 1.6  # the fixed grid extends by this factor beyond the support at both ends
        self._grid = None

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
        """Integration range of the size distribution (``x_range``, or found from N(x) x**x_power).

        The intensity of a particle grows like x^2 ... x^6, so a small amount of large particles can
        still dominate I(q): the cut-off is applied to N(x) x**x_power (``x_power``, default 6),
        not to N(x) alone, at the relative level ``tail``.
        """
        if self.x_range is not None:
            return tuple(self.x_range)
        x0 = abs(self.ff_p[self.distributed]) or 1.0
        for p in self.sd_p[1:4]:  # typical location parameters of SASfit distributions
            if p > 0:
                x0 = max(x0, p)
        x = np.geomspace(x0 * 1e-4, x0 * 1e3, 1400)
        sd_p = self.sd_p
        k = self.amplitude_index()
        if k is not None and sd_p[k] == 0:  # N = 0 (a fit switched the contribution off): the support is that of N = 1
            sd_p = sd_p.copy()
            sd_p[k] = 1.0
        w = np.abs(self.sd.evaluate_p(x, sd_p)) * x**self.x_power
        if not np.any(w > 0):
            raise ValueError("size distribution is zero everywhere; give x_range")
        keep = np.nonzero(w > self.tail * w.max())[0]
        lo, hi = x[max(keep[0] - 1, 0)], x[min(keep[-1] + 1, x.size - 1)]
        if self.snap > 0:  # same nodes for nearby parameters: the form factor on them can be reused
            r = math.log1p(self.snap)
            lo, hi = math.exp(math.floor(math.log(lo) / r) * r), math.exp(math.ceil(math.log(hi) / r) * r)
        return lo, hi

    def amplitude_index(self):
        """Index of the amplitude ``N`` among the size-distribution parameters (the intensity is proportional to it)."""
        if self.sd is None or "N" not in self.sd.parameter_names:
            return None
        return list(self.sd.parameter_names).index("N")

    def enable_cache(self, snap=0.0):
        """Remember intensities and form factors at the integration nodes (stop with :meth:`disable_cache`).

        ``snap`` > 0 rounds the automatic integration range outwards to a geometric grid of that step, so
        nearby parameter sets share nodes and the form factor on them can be reused.
        """
        from .integration import ContributionCache

        self.cache = ContributionCache()
        self.snap = float(snap)
        self._grid = None

    def disable_cache(self):
        self.cache, self.snap = None, 0.0

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

    def _cache_key(self, q):
        vec = lambda v: None if v is None else np.asarray(v, float).tobytes()  # noqa: E731
        return (q.tobytes(), vec(self.ff_p), vec(self.sd_p), vec(self.sq_p), self.approach, self.method, self.n_x, self.rtol,
                self.atol, self.limit, self.x_power, self.tail, None if self.x_range is None else tuple(self.x_range),
                self.distributed, self.snap)  # fmt: skip

    def intensity(self, q):
        q = np.asarray(q, dtype=float)
        cache = self.cache
        if cache is None:
            return self._intensity(q)
        key = self._cache_key(q)
        hit = cache.get(key)
        if hit is not None:
            return hit.copy()
        I = np.asarray(self._intensity(q), dtype=float)
        cache.put(key, I.copy())
        return I

    def _intensity(self, q):
        S = None
        if self.sq is not None and not (self.sd is not None and self.approach == "lma"):
            S = self.sq.evaluate_p(q, self.sq_p)
        if self.sd is None:
            I = self.ff.evaluate_p(q, self.ff_p)
            if S is not None:
                I = I * S  # manual 4.1.1 without distribution
            return I
        return size_integral(self, q, S)  # the three approaches of the manual, method chosen by self.method

    __call__ = intensity

    def __str__(self):
        s = f"form factor: {self.ff.name}"
        if self.sd:
            s += f", size distribution: {self.sd.name} over {self.ff.parameter_names[self.distributed] if self.distributed < len(self.ff.parameter_names) else self.distributed}"
        if self.sq:
            s += f", structure factor: {self.sq.name} ({self.approach})"
        return s


class ModelFunction:
    def __init__(self, contributions=(), scales=None, background=0.0, resolution=None, bg_A=0.0, bg_alpha=4.0,
                 active=None, transform=None):
        self.contributions = list(contributions)
        self.scales = list(scales) if scales is not None else [1.0] * len(self.contributions)
        # several data sets (SASfit): a contribution counts only where it is active; the others keep their place so
        # that the addresses '<i>.ff.R', 'scale.<i>' are the same in the models of all data sets
        self.active = list(active) if active is not None else [True] * len(self.contributions)
        self.background = background
        self.bg_A = float(bg_A)  # power-law term of the background: bg_A * q**(-bg_alpha) (off while bg_A = 0)
        self.bg_alpha = float(bg_alpha)
        self.resolution = resolution
        self._bg_z = None
        self.transform = transform

    @property
    def transform(self):
        """SESANS / MSAS transform (see the module docstring) or None."""
        return self._transform

    @transform.setter
    def transform(self, t):
        if t is not None and not hasattr(t, "kind"):
            from .hankel import make_transform

            t = make_transform(t) if isinstance(t, (dict, str)) else t
        self._transform = t

    @property
    def is_linear(self):
        """True if the model is linear in the scales, the amplitudes N and the background (variable projection)."""
        t = self._transform
        return t is None or bool(getattr(t, "linear", False))

    @property
    def domain(self):
        """``"q"`` or ``"delta"`` (SESANS: the model is evaluated at spin-echo lengths)."""
        return getattr(self._transform, "domain", "q")

    def include(self, contribution, scale=1.0, active=True):
        self.contributions.append(contribution)
        self.scales.append(float(scale))
        self.active.append(bool(active))
        return len(self.contributions) - 1

    def is_active(self, i):
        return i >= len(self.active) or bool(self.active[i])

    def power_term(self, q):
        """The power-law part of the background, ``bg_A * q**(-bg_alpha)`` (0 while ``bg_A`` is 0, and with SESANS)."""
        if not self.bg_A or self.domain == "delta":
            return 0.0
        return self.bg_A * np.asarray(q, dtype=float) ** (-self.bg_alpha)

    def _background(self, q):
        """The background: ``background`` (a constant, a Background object or a function of q) + ``bg_A * q**(-bg_alpha)``."""
        return self._constant_background(q) + self.power_term(q)

    def _constant_background(self, q):
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

    def _offset(self):
        """The constant background as a number (the SESANS offset)."""
        return float(self.background) if np.isscalar(self.background) else 0.0

    def components(self, q, unit_amplitude=()):
        """``([I_i(q)], background)``: every contribution without its scale factor, resolution applied.

        For the contributions in ``unit_amplitude`` the size-distribution amplitude N is set to 1 while
        evaluating (variable projection solves for it). The background is a scalar or an array like q.
        With a linear transform (SESANS G(δ) - G(0)) the parts are the transformed contributions; a non-linear
        transform (P/P0, MSAS) has no such decomposition (ValueError).
        """
        q = np.asarray(q, dtype=float)
        t = self._transform
        if t is not None and not self.is_linear:
            raise ValueError(f"{t.kind}: the model is not a sum of its contributions")
        if t is not None:  # SESANS G(δ) - G(0): each contribution on the Q grid, then transformed
            k, Wt = t.weights(q)
            raw, _bg = self._raw_components(k, unit_amplitude, None)
            return [Wt @ v for v in raw], self._offset()
        qe, W = (q, None) if self.resolution is None else self.resolution.setup(q)
        return self._raw_components(qe, unit_amplitude, W)[0], self._background(q)

    def _raw_components(self, qe, unit_amplitude, W):
        shape = qe.shape if W is None else (W.shape[0],)
        parts = []
        for i, c in enumerate(self.contributions):
            if not self.is_active(i):
                parts.append(np.zeros(shape))
                continue
            k = c.amplitude_index() if i in unit_amplitude else None
            if k is not None:
                old = c.sd_p[k]
                c.sd_p[k] = 1.0
            try:
                v = np.asarray(c.intensity(qe), dtype=float)
            finally:
                if k is not None:
                    c.sd_p[k] = old
            parts.append(v if W is None else W @ v)
        return parts, None

    def particles(self, q):
        q = np.asarray(q, dtype=float)
        I = np.zeros(q.shape)
        for i, (s, c) in enumerate(zip(self.scales, self.contributions)):
            if self.is_active(i):
                I = I + s * c.intensity(q)
        return I

    def intensity(self, q):
        q = np.asarray(q, dtype=float)
        t = self._transform
        if t is not None:
            if t.domain == "delta":  # SESANS: q are spin-echo lengths
                G = t.g_from(q, lambda k: np.asarray(self.particles(k), dtype=float)) + self._offset()
                return t.finish(q, G)
            # MSAS: multiple scattering of the particles, then resolution and background
            qe, Wr = (q, None) if self.resolution is None else self.resolution.setup(q)
            k, _r, _w = t.weights(qe)
            Ims = t.multiple(qe, np.asarray(self.particles(k), dtype=float), np.asarray(self.particles(qe), dtype=float))
            return (Ims if Wr is None else Wr @ Ims) + self._background(q)
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
        if addr in ("bg.A", "bg.alpha"):
            attr = "bg_" + addr[3:]
            return (lambda: getattr(m, attr), lambda v: setattr(m, attr, float(v)))
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
