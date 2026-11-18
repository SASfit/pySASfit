"""Compare the Python ports with the C plugins of the SASfit library.

    python -m pysasfit.studio.ports.check                      # all ports, report on the console
    python -m pysasfit.studio.ports.check ff_ringpolymerBZ --sets 50 --report ports_report.md

For every port, ``--sets`` random parameter vectors are drawn within the port's sampling ranges (``PortSpec.sample``;
ψ-dependent functions also get random ψ and θ0), both implementations are evaluated on a logarithmic q grid, and the
largest relative deviation |I_py - I_c| / max(|I_c|, floor) is reported together with the evaluation times.

In Python::

    from pysasfit.studio.ports import check
    result = check.compare("ff_deformed_ring_polymer__bz_")      # reference: the C plugin of the SASfit library
    print(result)
    result = check.compare(name, reference=my_function)           # reference(q, p) -> array, e.g. a test build
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass, field

import numpy as np

from . import get, names, sample_parameters

__all__ = [
    "CompareResult",
    "compare",
    "compare_all",
    "library_reference",
    "main",
    "report",
]

DEFAULT_Q = np.logspace(
    -3, 1, 120
)  # 1/nm: together with Rg = 1 ... 100 nm, q Rg spans 1e-3 ... 1e3


@dataclass
class CompareResult:
    name: str
    n_sets: int
    n_points: int
    max_rel: float
    median_rel: float
    worst: dict = field(
        default_factory=dict
    )  # parameters and q of the largest deviation
    t_c: float = 0.0  # seconds for all sets, reference
    t_py: float = 0.0  # seconds for all sets, port
    rtol: float = 0.0
    errors: list = field(default_factory=list)  # parameter sets where one side raised
    psi_consistent: bool | None = (
        None  # vectorised ψ argument == ψ parameter (anisotropic ports)
    )
    n_excluded: int = 0  # points outside PortSpec.c_valid (C known to lose digits there), not compared
    n_ref_zero: int = 0  # points where the reference is 0 but the port is not (typically a failed C integration)
    zero_at: dict = field(
        default_factory=dict
    )  # smallest q (with its parameters) where that happened
    x_label: str = (
        ""  # name of the reduced scale (e.g. 'q Rg'), if the port defines one
    )
    x_first_bad: float | None = (
        None  # smallest reduced scale where the deviation exceeds the tolerance (or C = 0)
    )
    x_last_bad: float | None = None  # largest such scale

    @property
    def passed(self):
        return (
            self.max_rel <= self.rtol
            and not self.errors
            and self.psi_consistent is not False
            and not self.n_ref_zero
        )

    @property
    def speedup(self):
        return self.t_c / self.t_py if self.t_py > 0 else float("nan")

    def __str__(self):
        s = "PASS" if self.passed else "FAIL"
        out = (
            f"{s} {self.name}: max rel. dev. {self.max_rel:.2e} (tol {self.rtol:.0e}), median {self.median_rel:.1e}, "
            f"{self.n_sets} sets x {self.n_points} q;  C {self.t_c * 1e3:.1f} ms, Python {self.t_py * 1e3:.1f} ms "
            f"(x{self.speedup:.1f})"
        )
        if self.worst:
            xs = (
                f" ({self.x_label} = {self.worst['x']:.4g})"
                if "x" in self.worst
                else ""
            )
            out += f"\n    worst at q = {self.worst['q']:.4g}{xs}, p = {np.round(self.worst['p'], 4).tolist()}"
        if self.x_first_bad is not None:
            out += (
                f"\n    deviation > {self.rtol:.0e} or C = 0 for {self.x_label} in "
                f"[{self.x_first_bad:.3g}, {self.x_last_bad:.3g}]"
            )
        if self.psi_consistent is False:
            out += "\n    vectorised psi argument differs from the psi parameter"
        if self.n_excluded:
            out += f"\n    {self.n_excluded} points where the C function is known to lose digits not compared (PortSpec.c_valid)"
        if self.n_ref_zero:
            z = self.zero_at
            out += (
                f"\n    C = 0 at {self.n_ref_zero} points where the port is not (failed integration in C?), "
                f"not included in the deviation; first at q = {z['q']:.4g} (port {z['py']:.4g}), "
                f"p = {np.round(z['p'], 4).tolist()}"
            )
        for e in self.errors[:3]:
            out += f"\n    error: {e}"
        return out


def library_reference(name):
    """reference(q, p) evaluating the C plugin ``name`` of the SASfit library (never a Python port)."""
    from ..functions import SASfitFunction

    f = SASfitFunction(name, language="c", errors="raise")

    def reference(q, p):
        return np.asarray(
            f.evaluate_p(np.asarray(q, float), np.asarray(p, float)), float
        )

    return reference


def _has_c_compat(spec):
    import inspect

    try:
        return "c_compat" in inspect.signature(spec.intensity).parameters
    except (TypeError, ValueError):
        return False


def compare(
    name,
    reference=None,
    q=None,
    n_sets=20,
    seed=0,
    floor=1e-300,
    rtol=None,
    old_sasfit=False,
):
    """Compare the port ``name`` with ``reference(q, p)`` (default: the C plugin of the SASfit library).

    ``old_sasfit``: reproduce the C deviations of SASfit builds before 2026-11-17 (ports that support
    ``c_compat``), for comparisons with an old SASfit build."""
    spec = get(name)
    if reference is None:
        reference = library_reference(name)
    q = DEFAULT_Q if q is None else np.asarray(q, float)
    worst = {}
    max_rel, rels, errors, n_zero, zero_at = 0.0, [], [], 0, {}
    n_excl = 0
    tol = spec.rtol if rtol is None else rtol
    x_bad = x_bad_hi = None
    t_c = t_py = 0.0
    sets = sample_parameters(spec, n_sets, seed=seed)
    for p in sets:
        try:
            t0 = time.perf_counter()
            ic = np.asarray(reference(q, p), float)
            t_c += time.perf_counter() - t0
            t0 = time.perf_counter()
            if spec.c_compat or (old_sasfit and _has_c_compat(spec)):
                ip = np.asarray(
                    spec.intensity(q, np.asarray(p, float), None, c_compat=True), float
                )
            else:
                ip = np.asarray(spec(q, p), float)
            t_py += time.perf_counter() - t0
        except Exception as exc:  # noqa: BLE001 -- reported
            errors.append(
                f"{type(exc).__name__}: {exc} at p = {np.round(p[: len(spec.parameters)], 4).tolist()}"
            )
            continue
        scale = np.maximum(np.abs(ic), floor)
        rel = np.abs(ip - ic) / scale
        bad = ~np.isfinite(ip) | ~np.isfinite(ic)
        rel[bad] = np.where(np.isfinite(ip[bad]) == np.isfinite(ic[bad]), 0.0, np.inf)
        if spec.c_valid is not None:
            valid = np.asarray(spec.c_valid(q, np.asarray(p, float)), bool)
            n_excl += int(np.sum(~valid))
            rel = np.where(valid, rel, 0.0)
        zero = (ic == 0) & (ip != 0)
        if np.any(zero):
            n_zero += int(np.sum(zero))
            kz = int(np.argmax(zero))
            qz = float(q[kz])
            if not zero_at or qz < zero_at["q"]:
                zero_at = {
                    "q": qz,
                    "p": p[: len(spec.parameters)].copy(),
                    "py": float(ip[kz]),
                }
            rel = np.where(zero, 0.0, rel)  # counted separately, not in the deviation
        x = np.asarray(spec.reduced(q, p), float) if spec.reduced is not None else None
        badpts = (rel > tol) | zero
        if x is not None and np.any(badpts):
            xb, xe = float(np.min(x[badpts])), float(np.max(x[badpts]))
            x_bad = xb if x_bad is None else min(x_bad, xb)
            x_bad_hi = xe if x_bad_hi is None else max(x_bad_hi, xe)
        if zero.any() and x is not None:
            zero_at.setdefault("x", float(np.min(x[zero])))
        k = int(np.argmax(rel))
        rels.append(rel)
        if rel[k] > max_rel or not worst:
            max_rel = max(max_rel, float(rel[k]))
            worst = {
                "q": float(q[k]),
                "p": p[: len(spec.parameters)].copy(),
                "c": float(ic[k]),
                "py": float(ip[k]),
            }
            if x is not None:
                worst["x"] = float(x[k])
    psi_ok = None
    if spec.anisotropic and sets:
        # vectorised ψ (radians) against the ψ parameter (degrees), on a (ψ, q) grid
        p = sets[0].copy()
        psis = np.linspace(0.0, np.pi, 7, endpoint=False)
        grid = spec(q[None, :], p, psi=psis[:, None])
        rows = []
        for ps in psis:
            p[spec.psi_index] = np.degrees(ps) if spec.psi_unit == "deg" else ps
            rows.append(spec(q, p))
        psi_ok = bool(np.allclose(grid, np.array(rows), rtol=1e-12, atol=0.0))
    allrel = np.concatenate(rels) if rels else np.array([np.nan])
    return CompareResult(
        name=name,
        n_sets=len(sets),
        n_points=q.size,
        max_rel=max_rel if rels else float("inf"),
        median_rel=float(np.median(allrel)),
        worst=worst,
        t_c=t_c,
        t_py=t_py,
        rtol=tol,
        errors=errors,
        psi_consistent=psi_ok,
        n_ref_zero=n_zero,
        n_excluded=n_excl,
        zero_at=zero_at,
        x_label=spec.reduced_label if spec.reduced is not None else "",
        x_first_bad=x_bad,
        x_last_bad=x_bad_hi,
    )


def compare_all(which=None, **kw):
    return [compare(n, **kw) for n in (which or names())]


def report(results):
    """Markdown table of comparison results."""
    lines = [
        "| function | result | max rel. dev. | median | tolerance | C = 0 at | OK | C [ms] | Python [ms] | speed-up | notes |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in results:
        spec = get(r.name)
        ok = (
            "all"
            if r.x_first_bad is None
            else f"not in {r.x_label} [{r.x_first_bad:.3g}, {r.x_last_bad:.3g}]"
        )
        lines.append(
            f"| {r.name} | {'PASS' if r.passed else 'FAIL'} | {r.max_rel:.1e} | {r.median_rel:.1e} | {r.rtol:.0e} | "
            f"{r.n_ref_zero} pts | {ok} | {r.t_c * 1e3:.1f} | {r.t_py * 1e3:.1f} | {r.speedup:.1f} | {spec.notes} |"
        )
    return "\n".join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser(
        prog="python -m pysasfit.studio.ports.check",
        description=__doc__.split("\n\n")[0],
    )
    ap.add_argument("names", nargs="*", help="ported functions (default: all)")
    ap.add_argument(
        "--sets", type=int, default=20, help="random parameter sets per function"
    )
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--report", help="write a Markdown report to this file")
    a = ap.parse_args(argv)
    results = compare_all(a.names or None, n_sets=a.sets, seed=a.seed)
    for r in results:
        print(r)
    if a.report:
        with open(a.report, "w", encoding="utf-8") as fh:
            fh.write(report(results) + "\n")
    return 0 if all(r.passed for r in results) else 1


if __name__ == "__main__":
    sys.exit(main())
