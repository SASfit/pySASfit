"""Compare the Python ports of SASfit plugins with the C plugins of your SASfit library.

Thin wrapper around :mod:`pysasfit.studio.ports.check` that runs from the repository root without installing
pySASfit and lets you choose the SASfit library folder. Run from the repository root::

    python tools/port_check.py                                  # all ports, library from the studio settings
    python tools/port_check.py --lib-dir C:/Users/me/SASstudio  # another SASfit folder (libsasfit + plugins)
    python tools/port_check.py ff_ringpolymerBZ --sets 50 --report ports_report.md
    python tools/port_check.py --list                           # ported functions
    python tools/port_check.py --aniso 1e-9 --strategy GSL_QAGS # other SASfit integration settings (this run only)
    python tools/port_check.py --rtol 1e-5                      # one tolerance for all functions

The C side uses the SASfit integration settings of the studio (Settings -> Integration...). Plugins integrate with
sasfit_integrate, whose relative tolerance is SASfit's eps "aniso" (default 1e-5); for those functions the
tolerance of the comparison is max(port tolerance, 2 max(eps aniso, eps nriq)) unless --rtol is given (functions
using sasfit_cubature are controlled by eps nriq). The exit code is 0
when every port is within its tolerance.
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(
    0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src")
)


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Python ports of SASfit plugins against the C library"
    )
    ap.add_argument("names", nargs="*", help="ported functions (default: all)")
    ap.add_argument(
        "--lib-dir",
        help="SASfit folder with libsasfit and the plugins (default: studio settings)",
    )
    ap.add_argument(
        "--plugin-dir",
        help="SASfit plugin source tree with freshly built plugins, e.g. SASfitGit/src/plugins: "
        "<dir>/<plugin>/lib/libsasfit_<plugin>.dll replaces the installed plugin (default: the saved "
        "setting, see 'pysasfit-studio plugin-build-dir')",
    )
    ap.add_argument(
        "--sets",
        type=int,
        default=20,
        help="random parameter sets per function (default 20)",
    )
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--report", help="write a Markdown table to this file")
    ap.add_argument(
        "--rtol",
        type=float,
        help="tolerance for all functions (default: the tolerance of each port)",
    )
    ap.add_argument(
        "--aniso",
        type=float,
        help="set SASfit's precision of sasfit_integrate (eps 'aniso', the relative tolerance of 1D integrals in the "
        "plugins) for this run",
    )
    ap.add_argument(
        "--nriq",
        type=float,
        help="set SASfit's eps 'nriq' for this run (not used by sasfit_integrate)",
    )
    ap.add_argument(
        "--old-sasfit",
        action="store_true",
        help="the SASfit build predates the C fixes of 2026-11-17: reproduce the old C deviations",
    )
    ap.add_argument(
        "--strategy",
        help="set SASfit's integration strategy for this run, e.g. GSL_QAGS, TanhSinh",
    )
    ap.add_argument(
        "--list", action="store_true", help="list the ported functions and exit"
    )
    a = ap.parse_args(argv)

    from pysasfit.studio import paths, ports
    from pysasfit.studio.ports import check

    if a.list:
        for s in ports.specs():
            kind = "ψ-dependent" if s.anisotropic else "isotropic"
            print(f"{s.name:42s} {kind:12s} tol {s.rtol:.0e}  {s.source}")
        return 0
    if a.lib_dir:
        paths.set_lib_dir(a.lib_dir)
    if a.plugin_dir:
        paths.set_plugin_build_dir(a.plugin_dir)
        if paths.plugin_build_dir() is None:
            print(f"plugin folder not found: {a.plugin_dir}")
            return 2
    try:
        from pysasfit.studio.library import get_library

        lib = get_library()
        print(f"SASfit library: {getattr(lib, 'path', None) or paths.lib_dir()}")
        if paths.plugin_build_dir():
            print(
                f"plugins from the build tree {paths.plugin_build_dir()} ({len(lib.build_dirs)} plugin libraries)"
            )
        if a.nriq:
            lib.set_precision(nriq=a.nriq)
        if a.aniso:
            lib.set_precision(aniso=a.aniso)
        if a.strategy:
            lib.set_precision(int_strategy=a.strategy)
        prec = lib.get_precision()
        from pysasfit.studio.library import QUADRATURE

        strategy = {v: k for k, v in QUADRATURE.items()}.get(
            prec.get("int_strategy"), prec.get("int_strategy")
        )
        aniso, nriq = prec.get("aniso"), prec.get("nriq")
        print(
            f"SASfit integration: strategy {strategy}, eps aniso {aniso} (rel. tolerance of sasfit_integrate), "
            f"eps nriq {prec.get('nriq')}"
        )
    except Exception as exc:  # noqa: BLE001 -- explain and stop
        print(f"cannot load the SASfit library: {type(exc).__name__}: {exc}")
        print(
            "give the SASfit folder with --lib-dir (the folder containing libsasfit and plugins/)"
        )
        return 2
    results = []
    skipped = []
    for name in a.names or ports.names():
        spec = ports.get(name)
        if not spec.c_reference and not a.names:
            skipped.append(f"{name}: {spec.notes}")
            continue
        tol = a.rtol
        if tol is None and spec.c_integrates and (aniso or nriq):
            # C is only as accurate as SASfit's requested precision (sasfit_integrate: eps aniso; sasfit_cubature:
            # eps nriq): compare at max(port tolerance, 2 max(eps aniso, eps nriq))
            tol = max(spec.rtol, 2.0 * max(aniso or 0.0, nriq or 0.0))
        results.append(
            check.compare(
                name, n_sets=a.sets, seed=a.seed, rtol=tol, old_sasfit=a.old_sasfit
            )
        )
    for r in results:
        print(r)
    for line in skipped:
        print(f"SKIP {line}")
    table = check.report(results)
    print()
    print(table)
    if a.report:
        with open(a.report, "w", encoding="utf-8") as fh:
            fh.write(table + "\n")
        print(f"\nreport written to {a.report}")
    return 0 if all(r.passed for r in results) else 1


if __name__ == "__main__":
    sys.exit(main())
