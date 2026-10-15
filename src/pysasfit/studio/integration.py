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

import numpy as np
from scipy import integrate as _si

__all__ = ["METHODS", "DEFAULTS", "PANELS", "integrate", "size_integral", "load_settings", "save_settings"]

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


def size_integral(c, q, S):
    """Intensity of contribution ``c`` (with a size distribution) at ``q``; ``S`` = S(q) or None (LMA).

    ``c`` provides ff, sd, sq, ff_p, sd_p, sq_p, distributed, approach, support() and the settings
    method, n_x, rtol, atol, limit. The result of the integration is kept in ``c.last_integration``.
    """
    a, b = c.support()
    p = c.ff_p.copy()
    nq = q.size

    def weight(x):
        return float(c.sd.evaluate_p(x, c.sd_p))

    def P(x, kind="intensity"):
        p[c.distributed] = x
        return c.ff.evaluate_p(q, p, kind)

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
    """Remember ``settings`` (method, n_x, rtol, atol, limit) for later sessions."""
    from . import paths

    paths.set_setting("integration", {k: settings[k] for k in DEFAULTS})
