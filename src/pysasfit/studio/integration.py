r"""Integration over the size distribution of a :class:`~pysasfit.studio.model.ScatteringContribution`.

Two different things are called "integration" in pySASfit:

* inside the compiled SASfit functions (form factors with orientation averages, Hankel
  transforms, ...) libsasfit does the integrals. Its strategy, accuracy and number of points
  are chosen with :func:`pysasfit.studio.library.configure_precision`;
* the size-distribution integral  I(q) = ∫ N(x) P(q; x) dx  (and its decoupling / local
  monodisperse variants) is done here, in Python, with one of the methods in :data:`METHODS`.
  Scipy cannot replace the first kind: the compiled functions call libsasfit directly.

The integrand is vector valued (one value per q), so a method evaluates the form factor once
per x and handles all q together.
"""

from __future__ import annotations

import math
from collections import OrderedDict

import numpy as np
from scipy import integrate as _si

__all__ = ["METHODS", "DEFAULTS", "PANELS", "ContributionCache", "integrate", "size_integral", "suggest_n_x", "format_node_check",
           "load_settings", "save_settings"]  # fmt: skip

METHODS = {
    "gauss-legendre": "Gauss-Legendre, fixed nodes (n_x per panel, 4 panels)",
    "quad_vec": "scipy quad_vec, adaptive (rtol, atol, limit)",
    "simpson": "scipy simpson, equidistant grid (4·n_x points)",
    "trapezoid": "scipy trapezoid, equidistant grid (4·n_x points)",
    "romb": "scipy romb (Romberg), equidistant grid (2^k+1 points >= 4·n_x)",
    "sasfit": "SASfit sasfit_integrate_ctm, strategy 'int_strategy' of the SASfit tab (rtol, atol, limit); "
    "one q value at a time: slow",
}
DEFAULTS = {
    "method": "gauss-legendre", "n_x": 96, "rtol": 1e-6, "atol": 0.0, "limit": 200,
    "x_power": 6.0, "tail": 1e-10,
}  # fmt: skip  (x_power, tail: range search, see ScatteringContribution.support)
PANELS = 4  # the support of N(x) is split into this many panels (skewed distributions)


def _nodes(method, a, b, n_x):
    """Nodes and weights (weights None: the method integrates the samples itself) of a fixed-node method."""
    if method == "gauss-legendre":
        t, w = np.polynomial.legendre.leggauss(int(n_x))
        edges = np.linspace(a, b, PANELS + 1)
        xs, ws = [], []
        for lo, hi in zip(edges[:-1], edges[1:]):
            xs.append(0.5 * (hi - lo) * t + 0.5 * (hi + lo))
            ws.append(0.5 * (hi - lo) * w)
        return np.concatenate(xs), np.concatenate(ws)
    n = max(int(n_x) * PANELS, 5)
    if method == "simpson":
        n += 1 - n % 2  # odd number of points
    elif method == "romb":
        n = 2 ** math.ceil(math.log2(n - 1)) + 1
    return np.linspace(a, b, n), None


def _sasfit_integrate(g, a, b, rtol, atol, limit):
    """Vector valued integral with libsasfit: one ``sasfit_integrate_ctm`` call per component.

    The integrand is evaluated for all components at once and cached per x, so components whose
    integrator visits the same nodes share the evaluations.
    """
    from .library import get_library

    lib = get_library()
    cache = {}

    def vec(x):
        y = cache.get(x)
        if y is None:
            if len(cache) > 5000:
                cache.clear()
            y = cache[x] = g(x)
        return y

    n = vec(0.5 * (a + b)).size
    return np.array([lib.integrate(lambda x, k=k: vec(x)[k], a, b, limit, atol, rtol) for k in range(n)])


def integrate(f, a, b, method="gauss-legendre", n_x=96, rtol=1e-6, atol=0.0, limit=200):
    """∫_a^b f(x) dx of a vector valued ``f(x)`` (scalar x -> array); returns ``(value, info)``.

    ``info``: method, number of evaluations of ``f`` (``neval``) and, for ``quad_vec``, the error estimate.
    """
    if method not in METHODS:
        raise ValueError(f"unknown integration method '{method}' (choose from {', '.join(METHODS)})")
    count = [0]

    def g(x):
        count[0] += 1
        return np.atleast_1d(np.asarray(f(float(x)), dtype=float))

    info = {"method": method}
    if method == "sasfit":
        value = _sasfit_integrate(g, a, b, rtol, atol, limit)
    elif method == "quad_vec":
        value, err = _si.quad_vec(g, a, b, epsabs=max(float(atol), 1e-200), epsrel=float(rtol), limit=int(limit))
        info["error"] = float(err)
    else:
        xs, ws = _nodes(method, a, b, n_x)
        Y = np.column_stack([g(x) for x in xs])
        if method == "gauss-legendre":
            value = Y @ ws
        elif method == "simpson":
            value = _si.simpson(Y, x=xs, axis=1)
        elif method == "trapezoid":
            value = _si.trapezoid(Y, x=xs, axis=1)
        else:
            value = _si.romb(Y, dx=float(xs[1] - xs[0]), axis=1)
    info["neval"] = count[0]
    return np.asarray(value, dtype=float), info


class ContributionCache:
    """What a contribution remembers during a fit (see ``FitProblem.fit(cache=True)``).

    * ``result``: the finished intensity for identical parameters (a change of another contribution, of a scale or
      of the background does not recompute it);
    * ``scope``: the form factor P(q, x) at the integration nodes x for identical *other* form-factor parameters
      (a change of the size distribution does not recompute it).

    Both are small LRU memories. The cache must be discarded when the SASfit precision settings change.
    """

    def __init__(self, max_results=128, max_scopes=24):
        self.max_results, self.max_scopes = int(max_results), int(max_scopes)
        self._results, self._scopes = OrderedDict(), OrderedDict()
        self.result_hits = self.result_misses = self.node_hits = self.node_misses = 0

    def get(self, key):
        v = self._results.get(key)
        if v is None:
            self.result_misses += 1
            return None
        self._results.move_to_end(key)
        self.result_hits += 1
        return v

    def put(self, key, value):
        self._results[key] = value
        while len(self._results) > self.max_results:
            self._results.popitem(last=False)

    def scope(self, key):
        """The dictionary ``{(kind, x): P(q, x)}`` for the form-factor parameters ``key``."""
        s = self._scopes.get(key)
        if s is None:
            s = self._scopes[key] = {}
            while len(self._scopes) > self.max_scopes:
                self._scopes.popitem(last=False)
        else:
            self._scopes.move_to_end(key)
        return s

    def clear(self):
        self._results.clear()
        self._scopes.clear()

    def text(self):
        r, n = self.result_hits + self.result_misses, self.node_hits + self.node_misses
        return f"intensities reused {self.result_hits}/{r}, form-factor nodes reused {self.node_hits}/{n}"


def size_integral(c, q, S):
    """Intensity of contribution ``c`` (with a size distribution) at ``q``; ``S`` = S(q) or None (LMA).

    ``c`` provides ff, sd, sq, ff_p, sd_p, sq_p, distributed, approach, support() and the settings
    method, n_x, rtol, atol, limit (optionally cache, fixed_grid, grid_margin). The result of the
    integration is kept in ``c.last_integration``.
    """
    a, b = c.support()
    p = c.ff_p.copy()
    nq = q.size

    def weight(x):
        return float(c.sd.evaluate_p(x, c.sd_p))

    cache = getattr(c, "cache", None)
    scope = None
    if cache is not None:  # form factor at the nodes: depends on q and the form-factor parameters except x
        scope = cache.scope((q.tobytes(), np.delete(np.asarray(c.ff_p, float), c.distributed).tobytes()))

    def P(x, kind="intensity"):
        if scope is not None:
            v = scope.get((kind, x))
            if v is not None:
                cache.node_hits += 1
                return v
            cache.node_misses += 1
        p[c.distributed] = x
        v = c.ff.evaluate_p(q, p, kind)
        if scope is not None:
            v = scope[(kind, x)] = np.array(v, dtype=float)
        return v

    if getattr(c, "fixed_grid", False) and (c.sq is None or c.approach in ("monodisperse", "decoupling")) and 0 < a < b:
        # fixed logarithmic grid: P(q, x) is computed once per grid, the distribution only gives the weights
        return _grid_integral(c, q, S, _fixed_grid(c, a, b), P, cache)

    if c.sq is None or c.approach == "monodisperse":  # manual 4.1.1:  S(q) ∫ N P dx
        def f(x):
            return weight(x) * P(x)

        def post(v):
            return v if S is None else v * S

    elif c.approach == "decoupling":  # manual 4.1.2:  ∫N P + (∫N F)^2 / ∫N (S - 1)
        def f(x):
            n = weight(x)
            return np.concatenate([n * P(x), n * P(x, "amplitude"), [n]])

        def post(v):
            return v[:nq] + v[nq : 2 * nq] ** 2 / v[2 * nq] * (S - 1.0)

    else:  # local monodisperse approximation, manual 4.1.3:  ∫ N P S(q; R_V(x)) dx
        sp = c.sq_p.copy()

        def f(x):
            n = weight(x)
            sp[0] = np.cbrt(abs(c.ff.volume(x, c.ff_p, dist=c.distributed)) * 3.0 / (4.0 * np.pi))
            return n * P(x) * c.sq.evaluate_p(q, sp)

        def post(v):
            return v

    value, info = integrate(f, a, b, c.method, c.n_x, c.rtol, c.atol, c.limit)
    c.last_integration = info
    return post(value)


def _fixed_grid(c, a, b):
    """Nodes and trapezoid weights of a grid equidistant in ln x that contains the support ``[a, b]``.

    The grid is kept in ``c._grid`` while it still contains the support with at least half the resolution (nodes
    inside the support: ``n_x * PANELS / 2``); otherwise a new one is built around the support with the margin
    ``c.grid_margin`` (default 1.6) on both sides. The integrand N(x) P(q, x) x is smooth in ln x and decays at both
    ends, where the trapezoid rule converges very fast.
    """
    n_in = max(int(c.n_x) * PANELS // 2, 48)
    g = getattr(c, "_grid", None)
    if g is not None and g[0] <= a and b <= g[1]:
        inside = int(np.count_nonzero((g[2] >= a) & (g[2] <= b)))
        if inside >= n_in // 2:
            return g[2], g[3]
    m = float(getattr(c, "grid_margin", 1.6))
    lo, hi = a / m, b * m
    d = math.log(b / a) / n_in
    n = int(math.ceil(math.log(hi / lo) / d)) + 1
    xs = lo * np.exp(d * np.arange(n))
    w = xs * d
    w[0] *= 0.5
    w[-1] *= 0.5
    c._grid = (lo, float(xs[-1]), xs, w)
    return xs, w


def _grid_integral(c, q, S, grid, P, cache):
    """The size-distribution integral as a weighted sum over the fixed grid (monodisperse and decoupling)."""
    xs, w0 = grid
    before = cache.node_misses if cache is not None else 0
    n = np.asarray(c.sd.evaluate_p(xs, c.sd_p), dtype=float) * w0
    Pm = np.column_stack([P(float(x)) for x in xs])
    if c.sq is None or c.approach == "monodisperse":
        v = Pm @ n
        v = v if S is None else v * S
    else:  # decoupling approximation
        Fm = np.column_stack([P(float(x), "amplitude") for x in xs])
        v = Pm @ n + (Fm @ n) ** 2 / float(np.sum(n)) * (S - 1.0)
    evals = (cache.node_misses - before) if cache is not None else xs.size
    c.last_integration = {"method": "fixed grid", "neval": int(evals), "nodes": int(xs.size)}
    return v


NODE_CANDIDATES = (8, 12, 16, 24, 32, 48, 64, 96, 128)
_FIXED_NODE_METHODS = ("gauss-legendre", "simpson", "trapezoid", "romb")


def suggest_n_x(c, q, tol=1e-4, candidates=NODE_CANDIDATES):
    """Smallest ``n_x`` whose intensity agrees with a reference of (at least) 192 nodes per panel to ``tol``.

    The difference is the maximum over q of ``|I_n - I_ref| / max(|I_ref|, 1e-6 max|I_ref|)``. A candidate counts only if
    the next candidate agrees as well. Returns a dictionary: ``n_x`` (None: no candidate reached ``tol``),
    ``current``, ``reference``, ``table`` [(n_x, difference)], ``tol``, ``method``; methods that choose their nodes
    themselves (``quad_vec``, ``sasfit``) return ``n_x`` None and a ``note``. The settings of ``c`` are restored.
    """
    if c.sd is None:
        raise ValueError("the contribution has no size distribution")
    out = {"method": c.method, "tol": float(tol), "current": int(c.n_x), "n_x": None, "table": [], "reference": None}
    if c.method not in _FIXED_NODE_METHODS:
        out["note"] = f"'{c.method}' chooses its nodes itself: n_x does not apply"
        return out
    q = np.asarray(q, dtype=float)
    ref_n = max(2 * int(c.n_x), 192)
    out["reference"] = ref_n
    saved = (c.n_x, getattr(c, "cache", None), getattr(c, "fixed_grid", False))
    c.cache = None
    if hasattr(c, "fixed_grid"):
        c.fixed_grid = False
    try:

        def run(n):
            c.n_x = int(n)
            return np.asarray(c.intensity(q), dtype=float)

        ref = run(ref_n)
        finite = np.isfinite(ref)
        floor = max(1e-6 * float(np.max(np.abs(ref[finite]))) if finite.any() else 0.0, 1e-300)

        def difference(I):
            with np.errstate(invalid="ignore", divide="ignore"):
                e = np.abs(I - ref) / np.maximum(np.abs(ref), floor)
            return float(np.nanmax(e)) if np.any(np.isfinite(e)) else math.inf

        cand = {int(n) for n in candidates if n < ref_n}
        if int(saved[0]) < ref_n:
            cand.add(int(saved[0]))
        streak, first = 0, None
        for n in sorted(cand):
            e = difference(run(n))
            out["table"].append((n, e))
            if e <= tol:
                first = n if streak == 0 else first
                streak += 1
            else:
                streak, first = 0, None
            if streak >= 2 and n >= int(saved[0]):
                break
        out["n_x"] = first
    finally:
        c.n_x, c.cache = saved[0], saved[1]
        if hasattr(c, "fixed_grid"):
            c.fixed_grid = saved[2]
    return out


def format_node_check(results):
    """Text of :func:`suggest_n_x` results; ``results``: ``[(label, dictionary)]``."""
    lines = []
    for label, r in results:
        lines.append(f"{label}: method {r['method']}, current n_x = {r['current']}")
        if r.get("note"):
            lines += [f"  {r['note']}", ""]
            continue
        lines.append(f"  difference to n_x = {r['reference']} (max relative, all q), goal {r['tol']:.0e}:")
        for n, e in r["table"]:
            lines.append(f"    n_x = {n:4d}   {e:9.2e}{'   <- current' if n == r['current'] else ''}")
        if r["n_x"] is None:
            lines.append("  no tested n_x reaches the goal: keep the current value or loosen the goal")
        else:
            saving = r["current"] / r["n_x"]
            lines.append(f"  n_x = {r['n_x']} is enough" + (f" ({saving:.1f}x fewer form-factor calls than now)" if saving > 1.2 else ""))
        lines.append("")
    return "\n".join(lines).rstrip()


# ------------------------------------------------------------------ settings
def load_settings():
    """The integration settings saved by the explorer (``paths.get_setting``), defaults for missing ones."""
    from . import paths

    out = dict(DEFAULTS)
    saved = paths.get_setting("integration")
    if isinstance(saved, dict):
        for key, default in DEFAULTS.items():
            if key in saved:
                try:
                    out[key] = type(default)(saved[key])
                except (TypeError, ValueError):
                    pass
    if out["method"] not in METHODS:
        out["method"] = DEFAULTS["method"]
    return out


def save_settings(settings):
    """Remember ``settings`` (all keys of :data:`DEFAULTS`: method, n_x, rtol, atol, limit, x_power, tail) for later sessions."""
    from . import paths

    paths.set_setting("integration", {k: settings[k] for k in DEFAULTS})
