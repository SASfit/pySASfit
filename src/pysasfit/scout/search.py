"""The model search (steps 3 to 5 of the scout): fit every candidate, score, rank and report.

Every candidate becomes a :class:`~pysasfit.studio.ModelFunction` that the Studio's
:class:`~pysasfit.studio.fitting.FitProblem` fits with multistart (Latin-hypercube start points, the first one from
the features of the data) and variable projection for the scale factors and the background. A cheap
size-distribution integral is used during the search; the result is refined with the accurate one.
Mixtures are built greedily with a small beam: the best ``mixture_beam`` candidates are each extended by one
contribution, and the best extension is kept as long as the criterion improves by ``mixture_gain``.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import re
import time
from dataclasses import asdict, dataclass, field

import numpy as np

from . import features as _features
from . import scoring
from .candidates import DISTRIBUTIONS, FORMS, STRUCTURE_FACTORS, Component, generate

__all__ = ["ScoutOptions", "Candidate", "ScoutResult", "search", "build", "fit_candidate", "candidate_from_dict",
           "result_from_dict"]


@dataclass
class ScoutOptions:
    forms: list | None = None  # keys of candidates.FORMS (None: all)
    distributions: list | None = None  # keys of candidates.DISTRIBUTIONS (None: all)
    structure_factor: str = "auto"  # "auto" (if the data suggest one), "always", "never"
    criterion: str = "aicc"  # ranking: "aicc" or "bic"
    starts: int = 8  # start points of the multistart fit of every candidate
    max_starts: int = 64  # ... raised up to this while the best minimum is not confirmed (a quarter with a size distribution)
    seed: int = 1  # of the start points; the same seed gives the same result
    max_components: int = 1  # > 1: greedy mixtures up to this many contributions
    mixture_gain: float = 6.0  # a further contribution must lower the criterion by at least this
    mixture_beam: int = 3  # this many of the best candidates are each extended by a further contribution
    mixtures_always: bool = False  # try mixtures even if the best single candidate already describes the data
    background: bool = True  # fit a constant background
    error_scale: float | str = 1.0  # factor on the stated sigma; "auto": the scatter estimate of the features if > 1.5
    n_x: int = 24  # Gauss-Legendre nodes per panel of the size-distribution integral during the search
    n_x_final: int = 96  # ... of the final refinement
    cache: bool = True  # form-factor cache of the Studio fit
    jobs: int = 1  # worker processes (candidates are fitted independently)
    time_limit: float | None = None  # seconds; candidates not started by then are skipped
    reference: bool = False  # use the pure-Python reference functions instead of the SASfit libraries
    resolution: object = None  # pysasfit.core.resolution object applied to every model
    store: str | None = None  # folder of the results store (pysasfit.scout.store); None: nothing is stored
    reuse: bool = True  # take a candidate from the store instead of fitting it again
    keep_runs: int = 20  # searches kept per curve in the store

    def as_dict(self):
        d = asdict(self)
        d["resolution"] = None if self.resolution is None else type(self.resolution).__name__
        return d


@dataclass
class Candidate:
    components: tuple
    label: str
    success: bool = False
    message: str = ""
    parameters: dict = field(default_factory=dict)  # address -> value (fitted)
    errors: dict = field(default_factory=dict)
    fixed: dict = field(default_factory=dict)  # "<i>.<role>.<name>" -> value
    approaches: list = field(default_factory=list)  # structure-factor approach per contribution
    n: int = 0
    k: int = 0
    chi2: float = math.inf
    chi2_red: float = math.inf
    aic: float = math.inf
    aicc: float = math.inf
    bic: float = math.inf
    weight: float = 0.0  # Akaike weight within the run
    delta: float = math.inf  # criterion minus the best one
    runs_z: float = math.nan
    runs_p: float = math.nan
    warnings: list = field(default_factory=list)
    nfev: int = 0
    seconds: float = 0.0
    q: np.ndarray | None = field(default=None, repr=False)
    I_fit: np.ndarray | None = field(default=None, repr=False)
    residuals: np.ndarray | None = field(default=None, repr=False)
    stored: bool = False  # taken from the results store, not fitted in this run

    def as_dict(self, curves=True):
        d = {k: v for k, v in asdict(self).items() if k not in ("q", "I_fit", "residuals", "components")}
        d["components"] = [asdict(c) for c in self.components]
        d["functions"] = [
            {"form_factor": FORMS[c.form].function,
             "size_distribution": DISTRIBUTIONS[c.sd].function if c.sd else None,
             "structure_factor": STRUCTURE_FACTORS[c.sq].function if c.sq else None}
            for c in self.components
        ]  # fmt: skip
        if curves and self.q is not None:
            d["q"], d["I_fit"], d["residuals"] = self.q.tolist(), self.I_fit.tolist(), self.residuals.tolist()
        return _jsonable(d)

    def text(self):
        if not self.success:
            return f"{self.label}: not fitted ({self.message})"
        lines = [f"{self.label}:  chi2/dof = {self.chi2_red:.4g}, AICc = {self.aicc:.5g}, BIC = {self.bic:.5g}, "
                 f"weight = {self.weight:.3f}, {self.k} parameters"]  # fmt: skip
        for a, v in self.parameters.items():
            lines.append(f"    {a:16s} = {v:.6g} ± {self.errors.get(a, math.nan):.2g}")
        for a, v in self.fixed.items():
            lines.append(f"    {a:16s} = {v:.6g}   (fixed)")
        for i, ap in enumerate(self.approaches):
            if ap:
                lines.append(f"    contribution {i}: structure factor in the '{ap}' approximation")
        lines += [f"    ! {w}" for w in self.warnings]
        return "\n".join(lines)


def candidate_from_dict(d):
    """A :class:`Candidate` from :meth:`Candidate.as_dict` (results store, protocol files)."""
    names = {f.name: f for f in Candidate.__dataclass_fields__.values()}
    kw = {}
    for k, f in names.items():
        if k in ("components", "q", "I_fit", "residuals") or k not in d:
            continue
        v = d[k]
        if v is None and f.type == "float":  # non-finite numbers are written as null
            v = math.nan if k.startswith("runs_") else math.inf
        kw[k] = v
    for k in ("parameters", "errors", "fixed"):
        kw[k] = {a: (math.nan if v is None else v) for a, v in (kw.get(k) or {}).items()}
    c = Candidate(tuple(Component(**c) for c in d["components"]), **{k: v for k, v in kw.items() if k != "label"},
                  label=d.get("label", ""))
    if d.get("q") is not None:
        c.q, c.I_fit, c.residuals = (np.asarray(d[k], float) for k in ("q", "I_fit", "residuals"))
    return c


def result_from_dict(d):
    """A :class:`ScoutResult` from :meth:`ScoutResult.as_dict` (without the data and the resolution object)."""
    fd = dict(d["features"])
    for k, f in _features.Features.__dataclass_fields__.items():
        if f.type == "float" and fd.get(k) is None:
            fd[k] = math.nan
    feat = _features.Features(**{k: v for k, v in fd.items() if k in _features.Features.__dataclass_fields__})
    od = {k: v for k, v in d["options"].items() if k in ScoutOptions.__dataclass_fields__ and k != "resolution"}
    return ScoutResult(feat, [candidate_from_dict(c) for c in d["candidates"]], ScoutOptions(**od), d["data"],
                       d.get("versions", {}), float(d.get("seconds") or 0.0), list(d.get("notes") or []))


def _jsonable(x):
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if isinstance(x, np.ndarray):
        return _jsonable(x.tolist())
    if isinstance(x, (np.floating, float)):
        return float(x) if math.isfinite(x) else None
    if isinstance(x, np.integer):
        return int(x)
    if isinstance(x, np.bool_):
        return bool(x)
    return x


# ----------------------------------------------------------------------------- building a candidate
def _has(f, kind):
    return getattr(f, {"volume": "_py_v", "amplitude": "_py_f"}[kind], None) is not None or \
        getattr(f, {"volume": "_c_v", "amplitude": "_c_f"}[kind], None) is not None  # fmt: skip


def build(components, features, options=None, start=None, n_x=None):
    """``(model, parameters, fixed, approaches)`` of a candidate.

    ``parameters`` is a list of ``(address, value, lower, upper, linear)`` for the fit; ``start`` (address -> value)
    replaces start values, e.g. with the result of an earlier fit.
    """
    from ..studio.model import ModelFunction, ScatteringContribution

    options = options or ScoutOptions()
    start = start or {}
    contribs, params, fixed, approaches = [], [], {}, []
    wide = len(components) > 1
    for i, comp in enumerate(components):
        form = FORMS[comp.form]
        ff_p = dict(form.fixed)
        fitted = []
        for p in form.fitted:
            v, lo, hi = p.resolve(features, wide)
            ff_p[p.name] = v
            fitted.append((p.name, v, lo, hi))
        kw, approach = {}, None
        if comp.sd:
            d = DISTRIBUTIONS[comp.sd]
            sd_p = dict(d.fixed)
            sd_p[d.amplitude] = 1.0
            for p in d.fitted:
                v, lo, hi = p.resolve(features, wide)
                sd_p[p.name] = v
                params.append((f"{i}.sd.{p.name}", v, lo, hi, False))
            name, v, lo, hi = next(t for t in fitted if t[0] == form.sd_param)
            fitted = [t for t in fitted if t[0] != form.sd_param]
            sd_p[d.location] = v
            params.append((f"{i}.sd.{d.location}", v, lo, hi, False))
            params.append((f"{i}.sd.{d.amplitude}", 1.0, 0.0, math.inf, True))
            fixed.update({f"{i}.sd.{k}": float(x) for k, x in d.fixed})
            kw.update(size_distribution=d.function, sd_params=sd_p, distributed=form.sd_param)
        else:
            params.append((f"scale.{i}", 1.0, 0.0, math.inf, True))
        params += [(f"{i}.ff.{name}", v, lo, hi, False) for name, v, lo, hi in fitted]
        fixed.update({f"{i}.ff.{k}": float(x) for k, x in form.fixed})
        if comp.sq:
            s = STRUCTURE_FACTORS[comp.sq]
            sq_p = dict(s.fixed)
            for p in s.fitted:
                v, lo, hi = p.resolve(features, wide)
                sq_p[p.name] = v
                params.append((f"{i}.sq.{p.name}", v, lo, hi, False))
            lo, hi = features.size_range
            r0 = 3.4 / features.peak_q if math.isfinite(features.peak_q) else features.size
            sq_p[s.radius] = min(max(r0, lo * 1.01), hi / 1.01)
            fixed.update({f"{i}.sq.{k}": float(x) for k, x in s.fixed})
            kw.update(structure_factor=s.function, sq_params=sq_p)
            approach = "monodisperse"
        c = ScatteringContribution(form.function, ff_p, n_x=n_x or options.n_x, **kw)
        if comp.sq:
            if comp.sd and _has(c.ff, "volume"):  # the interaction radius follows the size of every particle
                approach = "lma"
                c.set_approach("lma")
            else:
                params.append((f"{i}.sq.{s.radius}", sq_p[s.radius], lo, hi, False))
        approaches.append(approach)
        contribs.append(c)
    model = ModelFunction(contribs, background=float(features.background) if options.background else 0.0,
                          resolution=options.resolution)  # fmt: skip
    if options.background:
        params.append(("background", float(features.background), -math.inf, math.inf, True))
    params = [(a, min(max(float(start.get(a, v)), lo), hi), lo, hi, lin) for a, v, lo, hi, lin in params]
    return model, params, fixed, approaches


def _label(components):
    return "  +  ".join(c.label for c in components)


def fit_candidate(components, data, features, options=None, start=None, cancel=None):
    """Fit one candidate (a tuple of :class:`Component`) to ``data = (q, I, sigma)``; returns a :class:`Candidate`.

    ``cancel`` is a ``threading.Event``: when it is set the fit stops and the candidate counts as not fitted."""
    from ..studio.fitting import FitProblem, locate

    options = options or ScoutOptions()
    if options.reference:
        from . import reference

        if not reference.registered():
            reference.register()
    cand = Candidate(tuple(components), _label(components))
    q, I, s = (np.asarray(a, float) for a in data)
    t0 = time.perf_counter()
    try:
        model, params, cand.fixed, cand.approaches = build(components, features, options, start)
        prob = FitProblem()
        prob.add_dataset((q, I, s), model, name="data")
        for a, v, lo, hi, _lin in params:
            prob.add_parameter(a, value=v, lower=lo, upper=hi, label=a)
        # Monodisperse form factors oscillate: chi2 has many narrow minima, and a few starts miss the right one. The
        # number of starts is doubled (the best point so far is kept as the first start) until the best minimum is
        # reached twice and describes the data, or the limit is reached. Smooth (polydisperse) candidates get a
        # quarter of the limit: their fits are the expensive ones.
        smooth = any(c.sd is not None for c in model.contributions)
        limit = max(options.starts, options.max_starts // 4 if smooth else options.max_starts)
        starts, nfev = options.starts, 0
        while True:
            r = prob.fit("scipy-global", "multistart", vp=True, cache=options.cache, starts=starts, seed=options.seed,
                         cancel=cancel)
            nfev += r.nfev
            if r.message == "cancelled":
                raise InterruptedError("cancelled")
            if (_hits(r.message) >= 2 and r.chi2_red < 2.0) or starts >= limit:
                break
            starts = min(2 * starts, limit)
        message = r.message
        refine = [c for c in model.contributions if c.sd is not None and c.n_x != options.n_x_final]
        if refine:  # the search used a cheap size integral: finish with the accurate one
            for c in refine:
                c.n_x = options.n_x_final
            r = prob.fit("scipy", vp=True, cache=options.cache, cancel=cancel)
            nfev += r.nfev
            if r.message == "cancelled":
                raise InterruptedError("cancelled")
    except Exception as exc:  # noqa: BLE001 -- function not installed, model cannot be evaluated, no usable start
        cand.message = "skipped: cancelled" if isinstance(exc, InterruptedError) else f"{type(exc).__name__}: {exc}"
        cand.seconds = time.perf_counter() - t0
        return cand
    cand.seconds = time.perf_counter() - t0
    cand.success, cand.message, cand.nfev = True, message, nfev
    cand.parameters = {p.label: float(p.value) for p in r.parameters}
    cand.errors = {k: float(v) for k, v in r.errors.items()}
    cand.n, cand.k = int(q.size), len(r.free_labels)
    cand.chi2, cand.chi2_red = float(r.chi2), float(r.chi2_red)
    cand.aic = scoring.aic(cand.chi2, cand.k)
    cand.aicc = scoring.aicc(cand.chi2, cand.k, cand.n)
    cand.bic = scoring.bic(cand.chi2, cand.k, cand.n)
    cand.q, cand.I_fit = r.curves[0][0], r.curves[0][1]
    cand.residuals = (cand.I_fit - I) / s
    cand.runs_z, cand.runs_p = scoring.runs_test(cand.residuals)
    cand.fixed.update({a: float(locate(model, a)[0]()) for a in list(cand.fixed)})
    cand.warnings = _warnings(cand, r, message)
    return cand


def _hits(message):
    """Number of starts of the multistart fit that ended in the best minimum (0 if the message does not say)."""
    m = re.search(r"chi2/dof \(starts\): [^ ]+ \(x(\d+)\)", message)
    return int(m.group(1)) if m else 0


def _warnings(cand, r, message):
    out = []
    if cand.chi2_red > 3.0:
        out.append(f"poor fit (chi2/dof = {cand.chi2_red:.3g})")
    if math.isfinite(cand.runs_p) and cand.runs_p < 0.01 and cand.runs_z < 0:
        out.append(f"systematic residuals (runs test z = {cand.runs_z:.1f})")
    for label in r.at_bounds:
        if label.startswith("scale.") or re.search(r"\.sd\.N$", label):
            # amplitudes are tiny numbers (N ~ 1e-12): only an amplitude of exactly 0 (the bound of the linear solve) counts
            if cand.parameters.get(label, 1.0) <= 0.0:
                out.append(f"{label} is zero: this contribution is not needed")
        else:
            out.append(f"{label} is at a bound")
    for a, v in cand.parameters.items():
        e = cand.errors.get(a, math.nan)
        if a != "background" and a not in r.at_bounds and (not math.isfinite(e) or e > abs(v)):
            out.append(f"{a} is not determined by the data (error {e:.2g} for a value of {v:.3g})")
    c = r.correlation()
    if c is not None and np.all(np.isfinite(c)):
        n = len(r.free_labels)
        pairs = [(j, i) for i in range(n) for j in range(i) if abs(c[i, j]) > 0.98]
        if len(pairs) > 2:  # a whole group: one line instead of one per pair
            names = [r.free_labels[i] for i in sorted({i for p in pairs for i in p})]
            out.append(f"{', '.join(names)} are strongly correlated (|r| > 0.98): the data do not determine them separately")
        else:
            for j, i in pairs:
                out.append(f"{r.free_labels[j]} and {r.free_labels[i]} are not determined separately (r = {c[i, j]:+.3f})")
    if _hits(message) == 1 and cand.k > 2:
        out.append("the best minimum was reached from one start point only: repeat with more starts")
    return out


def _lookup(job):
    """The stored fit of this job, or None (no store, reuse off, not stored)."""
    components, _data, _features, options, start, (did, scale) = job
    if not options.store or not options.reuse:
        return None
    from .store import ResultStore

    return ResultStore(options.store).get_fit(did, ResultStore.fit_key(components, options, start, scale))


def _fit_job(job, cancel=None):
    hit = _lookup(job)
    if hit is not None:
        return hit
    components, data, features, options, start, (did, scale) = job
    cand = fit_candidate(components, data, features, options, start, cancel)
    if options.store and cand.success:
        from .store import ResultStore

        try:
            ResultStore(options.store).put_fit(did, ResultStore.fit_key(components, options, start, scale), cand)
        except OSError:  # the store is a convenience: a full or read-only disk does not stop the search
            pass
    return cand


def _run(jobs, options, deadline, progress=None, cancel=None, stage="candidates"):
    """Fit the jobs (serially or in worker processes); jobs not started before ``deadline`` or after ``cancel`` was
    set come back unfitted. ``progress(stage, done, total, label)`` is called after every candidate."""
    out = []

    def report(c):
        out.append(c)
        if progress is not None:
            progress(stage, len(out), len(jobs), c.label)

    if options.jobs > 1 and len(jobs) > 1:
        from concurrent.futures import ProcessPoolExecutor

        with ProcessPoolExecutor(max_workers=options.jobs) as pool:
            futures = [pool.submit(_fit_job, job) for job in jobs]
            for job, f in zip(jobs, futures):
                if cancel is not None and cancel.is_set() and f.cancel():  # not started yet
                    report(Candidate(tuple(job[0]), _label(job[0]), message="skipped: cancelled"))
                else:
                    report(f.result())
        return out
    for job in jobs:
        skip = "cancelled" if cancel is not None and cancel.is_set() else (
            "time limit reached" if deadline is not None and time.perf_counter() > deadline else None)
        if skip:  # a stored fit costs nothing: it is still taken
            report(_lookup(job) or Candidate(tuple(job[0]), _label(job[0]), message="skipped: " + skip))
        else:
            report(_fit_job(job, cancel))
    return out


def _cost(components):
    return sum(1 + 4 * (c.sd is not None) + (c.sq is not None) + 3 * (c.form in ("cylinder", "spheroid")) for c in components)


# ----------------------------------------------------------------------------- result
@dataclass
class ScoutResult:
    features: _features.Features
    candidates: list  # ranked: fitted candidates by the criterion, then those that could not be fitted
    options: ScoutOptions
    data: dict  # name, points, q range, sha256 of the arrays, error scale applied
    versions: dict
    seconds: float
    notes: list = field(default_factory=list)
    _data: tuple = field(default=None, repr=False)
    store: dict | None = None  # {folder, data_id, run} when the search was written to a results store

    @property
    def best(self):
        return self.candidates[0] if self.candidates and self.candidates[0].success else None

    def model(self, rank=0):
        """The :class:`~pysasfit.studio.ModelFunction` of candidate ``rank`` with the fitted values (for the Studio)."""
        from ..studio.fitting import locate

        cand = self.candidates[rank]
        model, _params, _fixed, _ap = build(cand.components, self.features, self.options, n_x=self.options.n_x_final)
        for a, v in cand.parameters.items():
            locate(model, a)[1](v)
        return model

    def table(self, top=None):
        rows = [f"{'#':>2s}  {'chi2/dof':>9s} {'AICc':>10s} {'BIC':>10s} {'weight':>7s} {'k':>2s}  candidate"]
        for i, c in enumerate(self.candidates[:top]):
            if c.success:
                flag = ("  !" if c.warnings else "") + ("  (stored)" if c.stored else "")
                rows.append(f"{i + 1:>2d}  {c.chi2_red:9.4g} {c.aicc:10.5g} {c.bic:10.5g} {c.weight:7.3f} {c.k:>2d}  {c.label}{flag}")
            else:
                rows.append(f"{i + 1:>2d}  {'-':>9s} {'-':>10s} {'-':>10s} {'-':>7s} {'-':>2s}  {c.label}  ({c.message})")
        return "\n".join(rows)

    def text(self, top=5):
        d = self.data
        lines = [f"pySASscout: {d['name']}  ({d['points']} points, error scale {d['error_scale']:.3g})", "",
                 self.features.text(), "", f"ranking by {self.options.criterion.upper()} "
                 f"({sum(c.success for c in self.candidates)} candidates fitted in {self.seconds:.1f} s):",
                 self.table(), ""]  # fmt: skip
        lines += [f"note: {n}" for n in self.notes]
        if self.notes:
            lines.append("")
        for c in self.candidates[:top]:
            if c.success:
                lines += [c.text(), ""]
        return "\n".join(lines).rstrip() + "\n"

    def as_dict(self, curves=True):
        return _jsonable({
            "program": "pySASscout", "data": self.data, "options": self.options.as_dict(), "versions": self.versions,
            "seconds": self.seconds, "features": self.features.as_dict(), "notes": self.notes,
            "candidates": [c.as_dict(curves) for c in self.candidates],
        })  # fmt: skip

    def save(self, filename, curves=True):
        """Write the result protocol (inputs, options, seed, versions, features and all candidates) as JSON."""
        with open(filename, "w", encoding="utf-8") as fh:
            json.dump(self.as_dict(curves), fh, indent=1)
        return filename


def _versions(options):
    import scipy

    from .. import __version__
    from ..studio import paths

    try:
        lib = paths.lib_dir()
    except Exception:  # noqa: BLE001
        lib = None
    return {"pysasfit": __version__, "numpy": np.__version__, "scipy": scipy.__version__,
            "python": platform.python_version(), "platform": platform.platform(),
            "sasfit_libraries": lib, "functions": "python reference" if options.reference else "sasfit"}  # fmt: skip


def _rank(cands, criterion):
    ok = sorted((c for c in cands if c.success), key=lambda c: getattr(c, criterion))
    w = scoring.akaike_weights([getattr(c, criterion) for c in ok])
    for c, wi in zip(ok, w):
        c.weight, c.delta = float(wi), float(getattr(c, criterion) - getattr(ok[0], criterion))
    return ok + [c for c in cands if not c.success]


def search(data, options=None, name=None, progress=None, cancel=None, **kw):
    """Search the candidate space for models of ``data`` and rank them; returns a :class:`ScoutResult`.

    ``data`` is a file name (read with :func:`pysasfit.core.load_data`) or ``(q, I, sigma)``; ``sigma`` may be
    None. Options are given as :class:`ScoutOptions` or as keywords (``starts=``, ``max_components=``, ...);
    ``q_range=(lo, hi)`` restricts the data. With ``store=<folder>`` every fit and the search are written to a
    results store (:mod:`pysasfit.scout.store`) and fits found there are reused. ``progress(stage, done, total, label)`` is called after every
    candidate; ``cancel`` (a ``threading.Event``) stops the search, the candidates fitted so far are ranked.
    """
    q_range = kw.pop("q_range", None)
    path = kw.pop("path", None)  # the file the data came from (noted in the results store)
    options = options or ScoutOptions(**kw)
    if options.criterion not in ("aicc", "bic"):
        raise ValueError("criterion must be 'aicc' or 'bic'")
    if isinstance(data, str):
        from ..core.io import load_data

        q, I, s, *_ = load_data(data)
        path = path or os.path.abspath(data)
        name = name or re.split(r"[\\/]", data)[-1]
    else:
        q, I, s, *_ = tuple(data) + (None,) * (3 - len(data))
    q, I = np.asarray(q, float), np.asarray(I, float)
    feat0 = None
    if s is None:  # the features assume 3 % and say so
        s = np.maximum(0.03 * np.abs(I), 1e-3 * np.nanmax(np.abs(I)))
        feat0 = "no uncertainties given: 3 % of the intensity assumed; the model ranking depends on the error bars"
    s = np.asarray(s, float)
    ok = np.isfinite(q) & np.isfinite(I) & np.isfinite(s) & (s > 0) & (q > 0)
    if q_range is not None:
        ok &= (q >= q_range[0]) & (q <= q_range[1])
    o = np.argsort(q[ok])
    q, I, s = q[ok][o], I[ok][o], s[ok][o]
    if options.reference:
        from . import reference

        reference.register()
    t0 = time.perf_counter()
    feat = _features.extract(q, I, s)
    if feat0:
        feat.notes.insert(0, feat0)
    scale = options.error_scale
    if scale == "auto":
        scale = feat.error_scale if math.isfinite(feat.error_scale) and feat.error_scale > 1.5 else 1.0
    scale = float(scale)
    data = (q, I, s * scale)
    store, did = None, None
    if options.store:
        from .store import ResultStore, data_id

        store, did = ResultStore(options.store), data_id(q, I, s)
        try:
            store.register_data(did, q, I, s, name=name, path=path)
        except OSError as exc:
            store = None
            options = ScoutOptions(**{**{f: getattr(options, f) for f in ScoutOptions.__dataclass_fields__}, "store": None})
            feat.notes.append(f"results store not usable ({exc}): nothing is stored")
    ctx = (did, scale)
    deadline = None if options.time_limit is None else t0 + float(options.time_limit)

    singles = sorted(generate(feat, options.forms, options.distributions, options.structure_factor), key=_cost)
    cands = _run([(c, data, feat, options, None, ctx) for c in singles], options, deadline, progress, cancel)
    ranked = _rank(cands, options.criterion)
    notes = []
    crit = lambda c: getattr(c, options.criterion)  # noqa: E731
    level = [c for c in ranked if c.success]  # candidates with `depth` contributions, best first
    depth = 1
    while level and depth < options.max_components and not (cancel is not None and cancel.is_set()):
        base = level[0]
        adequate = base.chi2_red < 1.3 and not (base.runs_p < 0.01 and base.runs_z < 0)
        if adequate and not options.mixtures_always:
            notes.append(f"no mixture tried: '{base.label}' already describes the data (chi2/dof = {base.chi2_red:.3g})")
            break
        # Greedy with a beam: the best `mixture_beam` candidates are each extended by one contribution. The best single
        # candidate of a bimodal sample is often a wrong shape, so extending only it would miss the right pair.
        dists = list(DISTRIBUTIONS) if options.distributions is None else list(options.distributions)
        extra = [Component(fk, sd) for fk in (options.forms or FORMS) for sd in [None] + (dists if FORMS[fk].sd_param else [])]
        jobs, seen = [], set()
        for b in level[: max(options.mixture_beam, 1)]:
            start = {a: v for a, v in b.parameters.items() if a != "background"}
            for e in extra:
                comps = b.components + (e,)
                key = tuple(sorted((c.form, c.sd or "", c.sq or "") for c in comps))
                if key not in seen:  # A + B is the same mixture as B + A
                    seen.add(key)
                    jobs.append((comps, data, feat, options, start, ctx))
        jobs.sort(key=lambda j: _cost(j[0]))
        mixes = _run(jobs, options, deadline, progress, cancel, f"mixtures of {depth + 1}")
        cands += mixes
        ranked = _rank(cands, options.criterion)
        level = sorted((m for m in mixes if m.success), key=crit)
        gain = crit(base) - crit(level[0]) if level else -math.inf
        if gain < options.mixture_gain:
            notes.append(f"a contribution number {depth + 1} is not supported by the data "
                         f"(best gain in {options.criterion.upper()}: {gain:.3g}, needed: {options.mixture_gain:g})")  # fmt: skip
            break
        depth += 1
        notes.append(f"contribution {depth} accepted: '{level[0].label}' ({options.criterion.upper()} lower by {gain:.3g}); "
                     "mixtures of similar shapes are strongly degenerate")  # fmt: skip

    best = ranked[0] if ranked and ranked[0].success else None
    if best is None:
        notes.append("no candidate could be fitted (are the SASfit libraries installed? see 'pysasfit-studio status')")
    else:
        if best.chi2_red > 5.0:
            notes.append(f"unknown case: even the best candidate does not describe the data (chi2/dof = {best.chi2_red:.3g}); "
                         "the curve is outside the candidate space, or the error bars are too small")  # fmt: skip
        close = [c for c in ranked if c.success and c.weight >= 0.05]
        if len(close) > 1:
            notes.append(f"{len(close)} candidates are comparably plausible (weights "
                         + ", ".join(f"{c.weight:.2f}" for c in close) + "): the data do not single out one model")
    if any(c.message == "skipped: time limit reached" for c in ranked):
        notes.append("time limit reached: not all candidates were fitted")
    if cancel is not None and cancel.is_set():
        notes.append("search cancelled: not all candidates were fitted")
    sha = hashlib.sha256(np.ascontiguousarray(np.vstack([q, I, s])).tobytes()).hexdigest()
    info = {"name": name or "data", "points": int(q.size), "q_min": float(q[0]), "q_max": float(q[-1]),
            "sha256": sha, "error_scale": scale}  # fmt: skip
    n_stored = sum(c.stored for c in ranked)
    if n_stored:
        notes.append(f"{n_stored} of {sum(c.success for c in ranked)} fits were taken from the results store")
    result = ScoutResult(feat, ranked, options, info, _versions(options), time.perf_counter() - t0, notes, data)
    if store is not None:
        try:
            run = store.save_run(did, result, keep=options.keep_runs)
            result.store = {"folder": store.root, "data_id": did, "run": run}
        except OSError:
            pass
    return result
