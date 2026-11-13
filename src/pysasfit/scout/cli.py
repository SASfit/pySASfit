"""``pysasfit-scout`` command line.

    pysasfit-scout sample.dat                      # ranking of the candidates, details of the best five
    pysasfit-scout sample.dat --json out.json      # also write the protocol of the run
    pysasfit-scout sample.dat --components 2       # allow a mixture of two contributions
    pysasfit-scout sample.dat --sq always --forms sphere core-shell
    pysasfit-scout --list                          # the candidate space
    pysasfit-scout sample.dat --features           # only the features and the data check
    pysasfit-scout sample.dat --store              # keep the fits in the results store and reuse them
    pysasfit-scout --store-info                    # where the results store is and what it holds

Without the compiled SASfit libraries the pure-Python reference functions are used (also with ``--reference``).
"""

from __future__ import annotations

import argparse
import os
import sys

__all__ = ["main"]


def _list():
    from .candidates import DISTRIBUTIONS, FORMS, STRUCTURE_FACTORS

    print("form factors:")
    for k, f in FORMS.items():
        extra = ", ".join(x for x in (f"size distribution over {f.sd_param}" if f.sd_param else "",
                                      "structure factor" if f.sq_ok else "") if x)  # fmt: skip
        print(f"  {k:14s} {f.function:26s} fitted: {', '.join(p.name for p in f.fitted)}" + (f"  ({extra})" if extra else ""))
    print("size distributions:")
    for k, d in DISTRIBUTIONS.items():
        print(f"  {k:14s} {d.function:26s} fitted: {', '.join([d.location] + [p.name for p in d.fitted])}")
    print("structure factors:")
    for k, s in STRUCTURE_FACTORS.items():
        print(f"  {k:14s} {s.function:26s} fitted: {', '.join([s.radius] + [p.name for p in s.fitted])}")


def main(argv=None):
    ap = argparse.ArgumentParser(prog="pysasfit-scout", description="Model search for small-angle scattering data.")
    ap.add_argument("data", nargs="?", help="text file with the columns q, I, sigma")
    ap.add_argument("--list", action="store_true", help="show the candidate space and exit")
    ap.add_argument("--features", action="store_true", help="only the features of the data and the data check")
    ap.add_argument("--forms", nargs="+", metavar="KEY", help="form factors to try (default: all, see --list)")
    ap.add_argument("--no-distribution", action="store_true", help="monodisperse candidates only")
    ap.add_argument("--sq", choices=["auto", "always", "never"], default="auto", help="structure factor (default: if the data suggest one)")
    ap.add_argument("--components", type=int, default=1, help="largest number of contributions of a mixture (default 1)")
    ap.add_argument("--criterion", choices=["aicc", "bic"], default="aicc")
    ap.add_argument("--starts", type=int, default=8, help="start points per candidate (default 8)")
    ap.add_argument("--max-starts", type=int, default=64, help="limit when the best minimum is not confirmed (default 64)")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--q-min", type=float)
    ap.add_argument("--q-max", type=float)
    ap.add_argument("--q-scale", type=float, default=1.0, help="factor on q (0.1: 1/nm -> 1/Angstrom)")
    ap.add_argument("--no-background", action="store_true", help="do not fit a constant background")
    ap.add_argument("--error-scale", default="1", help="factor on the stated errors, or 'auto' (from the scatter of the data)")
    ap.add_argument("--jobs", type=int, default=1, help="worker processes")
    ap.add_argument("--time-limit", type=float, help="seconds")
    ap.add_argument("--top", type=int, default=5, help="candidates shown in detail (default 5)")
    ap.add_argument("--json", metavar="FILE", help="write the protocol of the run")
    ap.add_argument("--store", nargs="?", const="", metavar="DIR",
                    help="write the fits and the search to a results store and reuse fits found there "
                         "(without DIR: scout_store next to the pySASfit preferences file)")
    ap.add_argument("--refit", action="store_true", help="with --store: fit again, replace the stored fits")
    ap.add_argument("--store-info", action="store_true", help="show the folder and the content of the results store and exit")
    ap.add_argument("--reference", action="store_true", help="use the Python reference functions, not the SASfit libraries")
    a = ap.parse_args(argv)

    if a.list:
        _list()
        return 0
    if a.store_info:
        from .store import ResultStore, _size

        st = ResultStore(a.store or None)
        print(f"results store: {st.root}\n{st.summary_text()}")
        for d in st.datasets():
            print(f"  {d['id']}  {d.get('last_used', ''):25s} {d['fits']:4d} fits {d['runs']:3d} searches "
                  f"{_size(d['bytes']):>9s}  {d.get('name', '')}")
        return 0
    if not a.data:
        ap.print_help()
        return 1
    from ..core.io import load_data
    from ..studio import available
    from . import ScoutOptions, extract, search
    from .store import default_dir

    q, I, s, _dq = load_data(a.data, q_scale=a.q_scale, qmin=a.q_min, qmax=a.q_max)
    if a.features:
        print(extract(q, I, s).text())
        return 0
    reference = a.reference or not available()
    if reference and not a.reference:
        print("SASfit libraries not installed: using the Python reference functions (see 'pysasfit-studio status')\n")
    options = ScoutOptions(
        forms=a.forms, distributions=[] if a.no_distribution else None, structure_factor=a.sq, criterion=a.criterion,
        starts=a.starts, max_starts=a.max_starts, seed=a.seed, max_components=a.components, background=not a.no_background,
        error_scale="auto" if a.error_scale == "auto" else float(a.error_scale), jobs=a.jobs, time_limit=a.time_limit,
        reference=reference, store=None if a.store is None else (a.store or default_dir()), reuse=not a.refit,
    )  # fmt: skip
    result = search((q, I, s), options, name=os.path.basename(a.data), path=os.path.abspath(a.data))
    print(result.text(a.top))
    if result.store:
        print("results store:", result.store["folder"])
    if a.json:
        print("protocol written to", result.save(a.json))
    return 0 if result.best is not None else 2


if __name__ == "__main__":
    sys.exit(main())
