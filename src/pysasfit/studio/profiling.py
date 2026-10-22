"""Where does the time of one model evaluation go?

``profile_model(model, q)`` evaluates a :class:`~pysasfit.studio.ModelFunction` a few times and splits the time into
the form factor, the size distribution, the structure factor and "everything else" (Python loops, array work, the
volume function of the local monodisperse approximation, resolution smearing). The numbers tell which speed-up
pays: fewer nodes and caching help when the form factor dominates, a faster call path when the time per call is
small and the overhead dominates.

    from pysasfit.studio.profiling import profile_model, format_profile
    print(format_profile(profile_model(model, q), n_free=4))
"""

from __future__ import annotations

import statistics  # noqa: F401
import time

import numpy as np

__all__ = ["profile_model", "format_profile"]

ROLES = ("ff", "sd", "sq")


class _Probe:
    def __init__(self):
        self.reset()

    def reset(self):
        self.calls, self.seconds, self.points = 0, 0.0, 0

    def snapshot(self):
        return {"calls": self.calls, "seconds": self.seconds, "points": self.points}


def _instrument(f, probe):
    """Time ``f.evaluate_p``; returns the function that undoes it (None if ``f`` cannot be instrumented)."""
    orig = getattr(f, "evaluate_p", None)
    if orig is None:
        return None
    had = "evaluate_p" in getattr(f, "__dict__", {})

    def timed(x, *args, **kwargs):
        t = time.perf_counter()
        try:
            return orig(x, *args, **kwargs)
        finally:
            probe.seconds += time.perf_counter() - t
            probe.calls += 1
            probe.points += int(np.size(x))

    try:
        f.evaluate_p = timed
    except Exception:  # noqa: BLE001 -- a function object that does not take attributes
        return None

    def undo():
        if had:
            f.evaluate_p = orig
        else:
            try:
                del f.evaluate_p
            except AttributeError:
                pass

    return undo


def _name(f):
    return getattr(f, "name", None) or type(f).__name__


def profile_model(model, q, repeat=3):
    """Time ``model.intensity(q)``; returns a dictionary (see :func:`format_profile`).

    One evaluation before the timed ones warms up the libraries. The result belongs to the run with the median
    total time. Caches of the contributions are switched off meanwhile.
    """
    q = np.asarray(q, dtype=float)
    probes, undo, roles = {}, [], []  # probes by function object; roles: [(contribution index, role, probe)]
    saved = []
    for i, c in enumerate(model.contributions):
        saved.append((c, getattr(c, "cache", None), getattr(c, "fixed_grid", False)))
        if hasattr(c, "cache"):
            c.cache = None
        if hasattr(c, "fixed_grid"):
            c.fixed_grid = False
        for role in ROLES:
            f = c.function_for(role)
            if f is None:
                continue
            if id(f) not in probes:
                probes[id(f)] = _Probe()
                u = _instrument(f, probes[id(f)])
                if u is not None:
                    undo.append(u)
            roles.append((i, role, probes[id(f)], f))
    runs = []
    try:
        model.intensity(q)  # warm-up
        for _ in range(max(int(repeat), 1)):
            for p in probes.values():
                p.reset()
            t = time.perf_counter()
            model.intensity(q)
            total = time.perf_counter() - t
            runs.append((total, {(i, role): p.snapshot() for i, role, p, _f in roles}))
    finally:
        for u in undo:
            u()
        for c, cache, fg in saved:
            if hasattr(c, "cache"):
                c.cache = cache
            if hasattr(c, "fixed_grid"):
                c.fixed_grid = fg
    runs.sort(key=lambda r: r[0])
    total, snap = runs[len(runs) // 2]
    contributions, inside = [], 0.0
    for i, c in enumerate(model.contributions):
        entry = {"index": i, "approach": getattr(c, "approach", None), "n_x": getattr(c, "n_x", None),
                 "method": getattr(c, "method", None)}  # fmt: skip
        for role in ROLES:
            f = c.function_for(role)
            if f is None or (i, role) not in snap:
                continue
            s = dict(snap[(i, role)])
            s["name"] = _name(f)
            entry[role] = s
            inside += s["seconds"]
        contributions.append(entry)
    return {
        "q_points": int(q.size), "total": total, "totals": [r[0] for r in runs], "inside": inside,
        "overhead": max(total - inside, 0.0), "contributions": contributions,
    }  # fmt: skip


def _ms(s):
    return f"{s * 1e3:.3g} ms" if s < 10 else f"{s:.3g} s"


def format_profile(res, n_free=None):
    """Text report of :func:`profile_model` with hints; ``n_free``: number of free parameters (for a time estimate)."""
    total = res["total"]
    lines = [f"one model evaluation: {_ms(total)} for {res['q_points']} q points "
             f"(median of {len(res['totals'])}; fastest {_ms(min(res['totals']))}, slowest {_ms(max(res['totals']))})"]  # fmt: skip
    hints = []
    labels = {"ff": "form factor", "sd": "size distribution", "sq": "structure factor"}
    for c in res["contributions"]:
        parts = [r for r in ROLES if r in c]
        head = f"contribution {c['index'] + 1}"
        if "sd" in c:
            head += f"  ({c['method']}, n_x = {c['n_x']}, {c['approach']})"
        lines.append(head)
        for role in parts:
            s = c[role]
            share = 100 * s["seconds"] / total if total > 0 else 0.0
            per = f", {_ms(s['seconds'] / s['calls'])} per call" if s["calls"] else ""
            lines.append(f"  {labels[role]:18s} {s['name']:24s} {s['calls']:6d} calls, {_ms(s['seconds']):>9s} "
                         f"({share:4.1f} %){per}")  # fmt: skip
            if role == "ff" and s["calls"] >= 50 and share > 70:
                hints.append(
                    f"The form factor of contribution {c['index'] + 1} takes {share:.0f} % in {s['calls']} calls: fewer "
                    "nodes ('Check nodes'), the cache or the fixed size grid, variable projection and coarse to fine "
                    "all reduce the number of calls."
                )
            if role == "sq" and share > 30:
                hints.append(
                    f"The structure factor of contribution {c['index'] + 1} takes {share:.0f} %. It is recomputed "
                    "when only form-factor parameters change (a separate S(q) cache is not implemented yet)."
                )
    share = 100 * res["overhead"] / total if total > 0 else 0.0
    lines.append(f"outside the SASfit functions (Python loops, arrays, volume function, resolution): "
                 f"{_ms(res['overhead'])} ({share:.1f} %)")  # fmt: skip
    if share > 40:
        hints.append(
            f"{share:.0f} % of the time is outside the SASfit functions: the calls are cheap and their number is "
            "large, so call overhead dominates. A vectorised call path (one C call for all q and nodes) would help "
            "more than fewer nodes."
        )
    if n_free:
        per_iter = (n_free + 1) * total
        lines.append(f"fit with {n_free} free parameters: about {n_free + 1} evaluations per iteration = {_ms(per_iter)} "
                     f"per iteration (finite-difference Jacobian), 10-30 iterations = {_ms(10 * per_iter)} - "
                     f"{_ms(30 * per_iter)}")  # fmt: skip
    if hints:
        lines += ["", "hints:"] + [f"  - {h}" for h in hints]
    return "\n".join(lines)
