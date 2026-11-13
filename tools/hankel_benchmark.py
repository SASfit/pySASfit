"""Speed and accuracy of the Hankel methods of pySASstudio (exact on linear pieces, FFTLog, QDHT) for SESANS and MSAS.

For every method and test case it measures

* build   -- time to set up the weights (once per data set; cached afterwards),
* eval    -- time of one model evaluation with the weights cached (what every fit step costs),
             split into the I(Q) evaluations on the nodes and the transform itself,
* nodes   -- number of points where I(Q) is evaluated,
* error   -- largest deviation from a reference (closed form, or the exact method on very fine grids, checked
             against QDHT / FFTLog with many points),
* fit     -- time and number of evaluations of a least-squares fit (SESANS: R and scale of a sphere).

The model is a Python stand-in (no SASfit library needed): a sphere, and a polydisperse sphere with a Gaussian size
distribution summed over ``--nr`` radii, which makes I(Q) about as expensive as a SASfit form factor with a size
distribution. Run from the repository root::

    python tools/hankel_benchmark.py            # all cases
    python tools/hankel_benchmark.py --quick    # fewer repetitions, no fits
    python tools/hankel_benchmark.py --csv hankel_benchmark.csv
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from pysasfit.studio import hankel  # noqa: E402
from pysasfit.studio.fitting import FitProblem  # noqa: E402
from pysasfit.studio.hankel import MSASTransform, SESANSTransform, sphere_sesans  # noqa: E402
from pysasfit.studio.model import ModelFunction  # noqa: E402

METHODS = ("filon", "fftlog", "qdht")


class _F:
    parameter_names = ["R", "eta"]


class Sphere:
    """(η V F(qR))² -- the contribution interface of ModelFunction / FitProblem; counts the q points evaluated."""

    sd = None

    def __init__(self, R=100.0, eta=1.0, poly=0.0, nr=1):
        self._v = {"ff": np.array([R, eta], float)}
        self._f = _F()
        self.poly, self.nr = poly, nr
        self.npoints = 0

    def function_for(self, role):
        return self._f if role == "ff" else None

    def vector(self, role):
        return self._v.get(role)

    def amplitude_index(self):
        return None

    def intensity(self, q):
        q = np.asarray(q, float)
        self.npoints += q.size
        R0, eta = self._v["ff"]
        if self.poly > 0 and self.nr > 1:
            Rs = R0 * (1 + self.poly * np.linspace(-3, 3, self.nr))
            w = np.exp(-0.5 * np.linspace(-3, 3, self.nr) ** 2)
            w /= w.sum()
        else:
            Rs, w = np.array([R0]), np.array([1.0])
        x = q[:, None] * Rs[None, :]
        with np.errstate(all="ignore"):
            f = np.where(x > 1e-6, 3 * (np.sin(x) - x * np.cos(x)) / x**3, 1.0)
        return ((eta * 4 / 3 * np.pi * Rs**3 * f) ** 2 * w).sum(axis=1)


def _time(fn, repeat):
    best = np.inf
    for _ in range(repeat):
        t = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - t)
    return best


def bench_case(name, kind, x, make_contrib, scale, reference, repeat, **tkw):
    """One test case for all methods: rows of dicts."""
    rows = []
    ref = reference
    for m in METHODS:
        c = make_contrib()

        def make():
            if kind == "sesans":
                return SESANSTransform("G", method=m, **tkw)
            return MSASTransform(method=m, **tkw)

        tr = make()
        model = ModelFunction([c], scales=[scale], transform=tr)
        t0 = time.perf_counter()
        try:
            y = model(x)  # first call: weights built, I evaluated
        except ValueError as exc:
            rows.append({"case": name, "method": m, "note": str(exc)})
            continue
        t_first = time.perf_counter() - t0
        c.npoints = 0
        t_eval = _time(lambda: model(x), repeat)
        nodes = c.npoints // repeat
        # the two parts of an evaluation: I on the nodes, and the transform (matrix products) alone
        if kind == "sesans":
            k, W = tr.weights(x)
            Ik = scale * c.intensity(k)
            t_I = _time(lambda: c.intensity(k), repeat)
            t_T = _time(lambda: tr.finish(x, W @ Ik), repeat)
        else:
            k, _r, _w = tr.weights(x)
            Ik, Iq = scale * c.intensity(k), scale * c.intensity(x)
            t_I = _time(lambda: (c.intensity(k), c.intensity(x)), repeat)
            t_T = _time(lambda: tr.multiple(x, Ik, Iq), repeat)
        err = float(np.max(np.abs(y - ref)) / np.max(np.abs(ref))) if ref is not None else float("nan")
        with np.errstate(all="ignore"):
            sig = np.abs(ref) > 1e-6 * np.max(np.abs(ref)) if ref is not None else None  # not the deep minima
            rel = float(np.max(np.abs(y / ref - 1)[sig])) if ref is not None and kind == "msas" else float("nan")
        rows.append({"case": name, "method": m, "nodes": nodes, "build_ms": 1e3 * (t_first - t_eval),
                     "eval_ms": 1e3 * t_eval, "I_ms": 1e3 * t_I, "transform_ms": 1e3 * t_T, "error": err, "max_rel": rel})  # fmt: skip
    return rows


def bench_fit(name, x, y, s, make_contrib, start, repeat=1, kind="sesans", **tkw):
    rows = []
    for m in METHODS:
        best = None
        for _ in range(repeat):
            c = make_contrib(start[0])
            tr = SESANSTransform("G", method=m, **tkw) if kind == "sesans" else MSASTransform(method=m, **tkw)
            p = FitProblem()
            p.add_dataset((x, y, s), ModelFunction([c], scales=[start[1]], transform=tr))
            p.add_parameter("0.ff.R", lower=start[0] / 3, upper=start[0] * 3)
            p.add_parameter("scale.0", lower=0)
            if kind == "msas":
                p.add_parameter("transform.thickness", lower=1e-4)
            t = time.perf_counter()
            r = p.fit("scipy", vp=kind == "sesans")
            dt = time.perf_counter() - t
            if best is None or dt < best[0]:
                best = (dt, r, {q.address: q.value for q in p.parameters})
        dt, r, vals = best
        rows.append({"case": name, "method": m, "fit_s": dt, "nfev": r.nfev, "chi2_red": r.chi2_red,
                     "R": vals["0.ff.R"], "thickness": vals.get("transform.thickness")})  # fmt: skip
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--quick", action="store_true", help="fewer repetitions, no fits")
    ap.add_argument("--nr", type=int, default=41, help="radii of the polydisperse stand-in (cost of I(Q))")
    ap.add_argument("--csv", help="also write the tables to this CSV file")
    a = ap.parse_args(argv)
    hankel.set_defaults(reset=True)  # built-in numerical defaults, not the saved ones
    repeat = 3 if a.quick else 10
    rows = []

    # ---------------------------------------------------------------- SESANS
    R = 10000.0  # Å, as the polystyrene example
    d = np.geomspace(400, 47000, 200)
    G0 = (4 / 3 * np.pi * R**3) ** 2 * 9 / (4 * R**2) / (2 * np.pi)
    sc = 0.21 / G0  # G(∞) - G(0) = -0.21 Å⁻²cm⁻¹ as the polystyrene data
    ref = sc * G0 * (sphere_sesans(d, R) - 1)  # closed form
    rows += bench_case("SESANS sphere 1 µm (200 δ)", "sesans", d, lambda: Sphere(R), sc, ref, repeat)
    ref_p = ModelFunction([Sphere(R, poly=0.1, nr=a.nr)], scales=[sc],
                          transform=SESANSTransform("G", per_decade=512))(d)  # reference: exact method, fine grid
    chk = ModelFunction([Sphere(R, poly=0.1, nr=a.nr)], scales=[sc],
                        transform=SESANSTransform("G", method="qdht", n_points=8192))(d)  # fmt: skip
    print(f"SESANS polydisperse: reference (exact method, 512/decade) vs QDHT 8192: "
          f"{np.max(np.abs(chk - ref_p)) / np.max(np.abs(ref_p)):.1e}")
    rows += bench_case(f"SESANS polydisperse ({a.nr} radii)", "sesans", d, lambda: Sphere(R, poly=0.1, nr=a.nr), sc,
                       ref_p, repeat)  # fmt: skip
    rows += bench_case("SESANS sphere, θ_max = 0.0168", "sesans", d, lambda: Sphere(R), sc, None, repeat,
                       theta_max=0.0168, wavelength=2.11)  # fmt: skip

    # ---------------------------------------------------------------- MSAS
    q = np.geomspace(0.01, 2.0, 150)  # 1/nm
    def poly():
        return Sphere(40.0, poly=0.15, nr=a.nr)
    probe = MSASTransform(1.0, 0.6)
    ModelFunction([poly()], scales=[1e-12], transform=probe)(q[:5])
    tau_unit = probe.last_tau / 1e-12  # Σt for scale 1, t = 1 (Σt is proportional to the scale)
    msas_scale = {}
    for target in (0.3, 2.0):
        scm = target / tau_unit  # t = 1 cm, λ = 0.6 nm
        msas_scale[target] = scm
        fine = dict(per_decade=128, r_per_decade=1024, q_min_factor=1e-3, q_max_factor=100, r_min_factor=0.1,
                    r_max_factor=100)  # reference: the exact method on fine, wide grids (memory: ~0.5 GB)
        ref_m = ModelFunction([poly()], scales=[scm], transform=MSASTransform(1.0, 0.6, **fine))(q)
        chk = ModelFunction([poly()], scales=[scm], transform=MSASTransform(1.0, 0.6, method="fftlog", n_points=4096))(q)
        print(f"MSAS Σt = {target:g}: reference (fine exact grids) vs FFTLog 4096: {np.max(np.abs(chk / ref_m - 1)):.1e}")
        rows += bench_case(f"MSAS polydisperse, Σt = {target:g}", "msas", q, poly, scm, ref_m, repeat,
                           thickness=1.0, wavelength=0.6)  # fmt: skip

    cols = ("case", "method", "nodes", "build_ms", "eval_ms", "I_ms", "transform_ms", "error", "max_rel")
    print("\nTimes in ms (best of %d); build = set-up of the weights (once per data set), eval = one model evaluation "
          "with cached weights = I on the nodes + transform; error relative to the largest value, max_rel (MSAS) the largest "
          "relative error of a single point (where I > 1e-6 of its maximum)." % repeat)
    print_table(rows, cols)

    fits = []
    if not a.quick:
        rng = np.random.default_rng(1)
        s = np.full(d.shape, 0.004)  # data from the reference (QDHT, 8192 points): no method is favoured
        fits += bench_fit("SESANS fit R, scale (vp)", d, ref_p + rng.normal(0, s), s,
                          lambda R0: Sphere(R0, poly=0.1, nr=a.nr), (8000.0, 2 * sc))  # fmt: skip
        scm = msas_scale[2.0]
        Im = ref_m  # Σt = 2, the reference
        sm = 0.01 * Im
        fits += bench_fit("MSAS fit R, scale, t (Σt = 2)", q, Im + rng.normal(0, sm), sm,
                          lambda R0: Sphere(R0, poly=0.15, nr=a.nr), (35.0, 0.7 * scm), kind="msas",
                          thickness=0.7, wavelength=0.6)  # fmt: skip
        print("\nFits (scipy least squares; best of 1):")
        print_table(fits, ("case", "method", "fit_s", "nfev", "chi2_red", "R", "thickness"))
    if a.csv:
        import csv

        with open(a.csv, "w", newline="") as fh:
            keys = sorted({k for r in rows + fits for k in r}, key=lambda k: (k not in cols, k))
            w = csv.DictWriter(fh, keys)
            w.writeheader()
            w.writerows(rows + fits)
        print(f"\nwritten {a.csv}")


def print_table(rows, cols):
    def fmt(v):
        if isinstance(v, float):
            return f"{v:.3g}" if v == v else "-"
        return "-" if v is None else str(v)

    width = {c: max(len(c), *(len(fmt(r.get(c))) for r in rows)) for c in cols}
    print("  ".join(c.ljust(width[c]) for c in cols))
    for r in rows:
        if "note" in r:
            print(f"{r['case'].ljust(width['case'])}  {r['method'].ljust(width['method'])}  {r['note']}")
            continue
        print("  ".join(fmt(r.get(c)).ljust(width[c]) for c in cols))


if __name__ == "__main__":
    main()
