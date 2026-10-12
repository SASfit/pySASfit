"""``pysasfit-studio`` command line (port of SASstudio_utilities_cli.py).

    pysasfit-studio list [--type ff|sq|sd|peak]
    pysasfit-studio search sphere shell
    pysasfit-studio info ff_sphere
    pysasfit-studio eval ff_sphere --q 0.01 0.1 0.5 --set R=50 eta=1
    pysasfit-studio new-plugin my_gauss [--dir DIR]
    pysasfit-studio sasview-adapter ff_sphere [--dir DIR]
    pysasfit-studio profiles OUTDIR [--source SASfitGit/src/plugins]
    pysasfit-studio install-binaries SOURCE [--dest DIR] [--frolov]
    pysasfit-studio status
    pysasfit-studio menu            # the interactive menu of the original tool
"""

from __future__ import annotations

import argparse
import os
import sys

from . import headers, paths

__all__ = ["main", "new_plugin", "sasview_adapter"]


def _template(name):
    with open(os.path.join(paths.TEMPLATES, name)) as fh:
        return fh.read()


def new_plugin(name, directory=None):
    """Create ``sasstudio_<name>.py`` and a test file from the templates; returns both paths."""
    directory = directory or os.path.join(os.path.expanduser("~"), ".pysasfit", "studio_plugins")
    os.makedirs(os.path.join(directory, "tests"), exist_ok=True)
    plugin = os.path.join(directory, f"sasstudio_{name}.py")
    test = os.path.join(directory, "tests", f"sasstudio_{name}_test.py")
    for fn, tpl in ((plugin, "python_plugin.py"), (test, "plugin_test.py")):
        if os.path.exists(fn):
            raise FileExistsError(fn)
        with open(fn, "w") as fh:
            fh.write(_template(tpl).replace("PLUGIN_NAME", name))
    return plugin, test


def sasview_adapter(name, directory="."):
    """Write a SasView plugin model file wrapping SASfit function ``name``."""
    p = headers.info(name)
    params = ",\n".join(
        f'    ["{q.name}", "", {q.default!r}, [{q.bounds[0]!r}, {q.bounds[1]!r}], "", {q.description!r}]'
        for q in p.parameters
    ).replace("inf", 'float("inf")')
    text = (
        _template("sasview_adapter.py")
        .replace("NAME_STR", p.name)
        .replace("TITLE_STR", p.label)
        .replace("DESCRIPTION_STR", p.brief or p.label)
        .replace("CATEGORY_STR", "structure-factor" if p.type == "sq" else "shape:sasfit")
        .replace("STRUCTURE_FACTOR_BOOL", str(p.type == "sq"))
        .replace("    PARAMS", params)
    )
    os.makedirs(directory, exist_ok=True)
    fn = os.path.join(directory, f"sasview_adapter_{p.name}.py")
    if os.path.exists(fn):
        raise FileExistsError(f"{fn} exists; delete it to recreate")
    with open(fn, "w") as fh:
        fh.write(text)
    return fn


def _list(type_=None):
    for p in sorted(headers.catalogue().values(), key=lambda p: p.name):
        if type_ is None or p.type == type_:
            print(f"{p.name:45s} {p.type_name:18s} {p.label}")


def _menu():
    while True:
        print("\nSASstudio utilities (pysasfit.studio):\n")
        print("1. Create new Python plugin [and its test]")
        print("2. Create SasView adapter for an existing plugin")
        print("6. Search all plugin names for key word(s)")
        print("7. List names of all available plugins")
        print("8. Get parameter info for a plugin")
        print("0. Quit")
        mode = input("\nOption: ").strip()
        try:
            if mode == "0":
                return
            elif mode == "1":
                print("created", *new_plugin(input("name (underscores instead of spaces): ").strip()))
            elif mode == "2":
                print("created", sasview_adapter(input("SASfit function name: ").strip()))
            elif mode == "6":
                for p in headers.search(*input("key word(s): ").split()):
                    print(f"{p.name:45s} {p.label}")
            elif mode == "7":
                _list()
            elif mode == "8":
                print(headers.info(input("function name: ").strip()))
        except Exception as e:  # noqa: BLE001 -- keep the menu running
            print("error:", e)


def main(argv=None):
    ap = argparse.ArgumentParser(prog="pysasfit-studio", description="Access to the SASfit plugin functions.")
    sub = ap.add_subparsers(dest="cmd")
    s = sub.add_parser("list")
    s.add_argument("--type", choices=["ff", "sq", "sd", "peak"])
    s = sub.add_parser("search")
    s.add_argument("words", nargs="+")
    s.add_argument("--type", choices=["ff", "sq", "sd", "peak"])
    s = sub.add_parser("info")
    s.add_argument("name")
    s = sub.add_parser("eval")
    s.add_argument("name")
    s.add_argument("--q", nargs="+", type=float, required=True)
    s.add_argument("--set", nargs="*", default=[], help="NAME=VALUE")
    s = sub.add_parser("new-plugin")
    s.add_argument("name")
    s.add_argument("--dir")
    s = sub.add_parser("sasview-adapter")
    s.add_argument("name")
    s.add_argument("--dir", default=".")
    s = sub.add_parser("profiles")
    s.add_argument("outdir")
    s.add_argument("--source")
    s = sub.add_parser("install-binaries")
    s.add_argument("source")
    s.add_argument("--dest")
    s.add_argument("--frolov", action="store_true", help="also copy the FrolovPoints tables (~100 MB)")
    sub.add_parser("status")
    sub.add_parser("menu")
    a = ap.parse_args(argv)

    if a.cmd == "list":
        _list(a.type)
    elif a.cmd == "search":
        for p in headers.search(*a.words, type=a.type):
            print(f"{p.name:45s} {p.type_name:18s} {p.label}")
    elif a.cmd == "info":
        print(headers.info(a.name))
    elif a.cmd == "eval":
        from .functions import function

        named = {k: float(v) for k, v in (x.split("=", 1) for x in a.set)}
        f = function(a.name)
        for q, y in zip(a.q, f(a.q, **named)):
            print(f"{q:.6g}\t{y:.8g}")
    elif a.cmd == "new-plugin":
        print("created", *new_plugin(a.name, a.dir))
    elif a.cmd == "sasview-adapter":
        print("created", sasview_adapter(a.name, a.dir))
    elif a.cmd == "profiles":
        print(len(headers.write_profiles(a.outdir, source_dir=a.source)), "profiles written to", a.outdir)
    elif a.cmd == "install-binaries":
        from .install import install_binaries

        install_binaries(a.source, a.dest, frolov=a.frolov)
    elif a.cmd == "status":
        cat = headers.catalogue()
        print(f"header folders: {paths.header_dirs()}")
        print(f"functions documented: {len(cat)}")
        print(f"SASfit libraries: {paths.lib_dir() or 'not installed (pysasfit-studio install-binaries SOURCE)'}")
        print(f"Python plugin folders: {paths.python_plugin_dirs()}")
    elif a.cmd == "menu":
        _menu()
    else:
        ap.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
