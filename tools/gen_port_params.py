"""Generate src/pysasfit/studio/ports/param_table.py: parameter bounds and kinds of every Python port.

    python tools/gen_port_params.py [--src DIR]

The parameter names are those of the ports (= the plugin headers). The bounds come from the domain checks of
the C sources (``SASFIT_CHECK_COND1((R < 0.0), param, "R(%lg) < 0", R)``): a check on one parameter against a
number becomes a bound, every other check (several parameters, ``||``/``&&``) a rule. The kind of a parameter
is guessed from its name (:func:`pysasfit.studio.ports.params.guess_kind`). Defaults are not generated: they are
the kind defaults or the hand-written values of ``ports/param_defaults.py``.

``--src``: folder with the plugin sources (default: tests/studio/port_reference/sasfit_src, i.e. the copies used
as C reference; give SASfitGit/src/plugins for the originals).
"""

from __future__ import annotations

import argparse
import os
import pprint
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

_DEFINE = re.compile(
    r"#define\s+(\w+)\s+(fabs\s*\()?\s*param->p\[(\d+)\]\s*\)?\s*(?://.*)?$", re.M
)
_CHECK = re.compile(
    r"SASFIT_CHECK_COND[12]?\s*\(\s*\((.*?)\)\s*,\s*param\s*,\s*\"([^\"]*)\"\s*(?:,\s*([^;]*?))?\)\s*;",
    re.S,
)
_NUM = r"\(?\s*[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?(?:\s*[-+*/]\s*(?:\d+\.?\d*|\.\d+))*\s*\)?"
_SIMPLE = re.compile(rf"^\s*\(?\s*(\w+)\s*(<=|<|>=|>)\s*({_NUM})\s*\)?\s*$")


def _strip_comments(s):
    s = re.sub(r"/\*.*?\*/", "", s, flags=re.S)
    return re.sub(r"//[^\n]*", "", s)


def _num(text):
    return float(eval(text.replace(" ", ""), {"__builtins__": {}}, {}))  # noqa: S307 -- numeric literal only


def parse_source(path):
    """``(macros, checks)``: macro name -> (index, absolute value?), and the (condition, message, args) of the
    domain checks."""
    with open(path, encoding="latin-1") as fh:
        s = _strip_comments(fh.read())
    macros = {
        m.group(1): (int(m.group(3)), bool(m.group(2))) for m in _DEFINE.finditer(s)
    }
    checks = [
        (m.group(1).strip(), m.group(2), (m.group(3) or "").strip())
        for m in _CHECK.finditer(s)
    ]
    return macros, checks


def _pyexpr(cond, macros):
    e = cond.replace("&&", " and ").replace("||", " or ").replace("!=", " != ")
    e = re.sub(r"(?<![=<>!])!(?!=)", " not ", e)
    e = re.sub(r"\bfabs\b", "abs", e)

    def sub(m):
        name = m.group(0)
        if name in macros:
            i, absval = macros[name]
            return f"abs(p[{i}])" if absval else f"p[{i}]"
        return name

    return re.sub(r"\b[A-Za-z_]\w*\b", sub, e)


def bounds_and_rules(names, macros, checks):
    """Per parameter index: [lower, lower_open, upper, upper_open]; and rules [(bad-expression, message, arg)]."""
    b = {i: [float("-inf"), False, float("inf"), False] for i in range(len(names))}
    rules = []
    for cond, msg, arg in checks:
        m = _SIMPLE.match(cond)
        if m and m.group(1) in ("q", "x", "r"):
            continue  # the abscissa
        if m and m.group(1) in macros:
            i, absval = macros[m.group(1)]
            if absval or i not in b:
                continue
            op, v = m.group(2), _num(m.group(3))
            lo, lo_open, hi, hi_open = b[i]
            # the condition is the *error*: R < 0 -> R >= 0 (closed), R <= 0 -> R > 0 (open)
            if op in ("<", "<="):
                if v > lo or (v == lo and op == "<=" and not lo_open):
                    lo, lo_open = v, op == "<="
            else:
                if v < hi or (v == hi and op == ">=" and not hi_open):
                    hi, hi_open = v, op == ">="
            b[i] = [lo, lo_open, hi, hi_open]
            continue
        if re.search(r"\bq\b", cond) and not any(k in cond for k in macros):
            continue
        expr = _pyexpr(cond, macros)
        if set(re.findall(r"[A-Za-z_]\w*", expr)) - {
            "p",
            "abs",
            "math",
            "fabs",
            "sqrt",
            "pow",
            "e",
            "and",
            "or",
            "not",
        }:
            continue  # refers to a local variable of the C function (checked by the port itself)
        rules.append((expr, msg, _pyexpr(arg, macros) if arg else ""))
    return b, rules


def build(src_root):
    from pysasfit.studio import ports
    from pysasfit.studio.ports.params import guess_kind

    table = {}
    for spec in sorted(ports.specs(), key=lambda s: s.name):
        rel = spec.source.replace("plugins/", "", 1) if spec.source else ""
        path = os.path.join(src_root, rel)
        macros, checks = (
            parse_source(path) if rel and os.path.exists(path) else ({}, [])
        )
        b, rules = bounds_and_rules(spec.parameters, macros, checks)
        params = []
        for i, name in enumerate(spec.parameters):
            lo, lo_open, hi, hi_open = b[i]
            params.append((name, lo, lo_open, hi, hi_open, guess_kind(name)))
        table[spec.name] = {
            "params": params,
            "rules": rules,
            "source_found": bool(macros or checks),
        }
    return table


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument(
        "--src",
        default=os.path.join(ROOT, "tests", "studio", "port_reference", "sasfit_src"),
    )
    ap.add_argument(
        "--out",
        default=os.path.join(
            ROOT, "src", "pysasfit", "studio", "ports", "param_table.py"
        ),
    )
    a = ap.parse_args(argv)
    table = build(a.src)
    text = pprint.pformat(table, width=118, sort_dicts=True).replace("inf", "INF")
    with open(a.out, "w", newline="\n", encoding="utf-8") as fh:
        fh.write('"""Generated by tools/gen_port_params.py - do not edit (corrections go to param_defaults.py).\n\n'
                 "Per port: params = [(name, lower, lower_open, upper, upper_open, kind)] with the bounds parsed from the\n"
                 "C domain checks; rules = [(python expression that is True for an invalid p, C message, argument)].\n"
                 '"""\n\n# ruff: noqa\nINF = float("inf")\n\nTABLE = ' + text + "\n")  # fmt: skip
    missing = [n for n, v in table.items() if not v["source_found"]]
    print(
        f"{len(table)} ports written to {a.out}; without C source: {', '.join(missing) or 'none'}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
