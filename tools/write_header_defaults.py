"""Write the default values and domains of the Python ports into the SASfit plugin headers.

    python tools/write_header_defaults.py <SASfitGit/src/plugins> [--out DIR] [--check]

For every ported function the Doxygen block of its ``\\defgroup`` gets (or has replaced) a paragraph after the
parameter table::

     * \\par Default values and domains:
     *      R = 10 [0, inf); dummy = 0; dummy = 0; eta = 1

one entry per parameter, in table order (``name = default`` and the domain, open bounds with round brackets).
The SASfit Tcl GUI ignores the paragraph (it reads only the ``\\b`` lines of the table); Doxygen shows it; the
pySASfit header parser (:mod:`pysasfit.studio.headers`) reads it and uses the values as defaults and bounds.

Line endings and encoding of the headers are kept. ``--out``: write the changed headers below DIR (same
relative paths) instead of in place. ``--check``: only report.
"""

from __future__ import annotations

import argparse
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

TITLE = "\\par Default values and domains:"


def defaults_line(params):
    parts = []
    for p in params:
        if p.kind == "dummy":
            parts.append(f"{p.name} = {p.default:g}")
        else:
            dom = p.bounds_text()
            parts.append(f"{p.name} = {p.default:g}" + ("" if dom == "(-inf, inf)" else f" {dom}"))
    return "; ".join(parts)


def update_header(text, name, line, nl):
    """Insert/replace the paragraph in the block of ``\\defgroup name``; returns (new text, status)."""
    m = re.search(rf"\\defgroup\s+{re.escape(name)}\b", text)
    if not m:
        return text, "no \\defgroup"
    end = text.find("*/", m.end())
    if end < 0:
        return text, "no end of the comment block"
    block = text[m.end() : end]
    tab = block.find("</table>")
    if tab < 0:
        return text, "no parameter table"
    new_par = f" * {TITLE}{nl} *      {line}{nl}"
    old = re.search(r"[ \t]*\*[ \t]*\\par Default values and domains:.*?(?=[ \t]*\*[ \t]*(?:\\|$)|[ \t]*\*/|\Z)", block,
                    re.S | re.M)  # fmt: skip
    if old:
        start, stop = m.end() + old.start(), m.end() + old.end()
        # the paragraph runs to the next line that does not continue it
        lines = text[start:end].split(nl)
        k = 1
        while k < len(lines) and lines[k].strip().startswith("*") and not lines[k].strip().startswith(("* \\", "*/")) \
                and lines[k].strip() not in ("*",):  # fmt: skip
            k += 1
        stop = start + len(nl.join(lines[:k])) + len(nl)
        new = text[:start] + new_par + text[stop:]
        return new, ("unchanged" if new == text else "replaced")
    pos = m.end() + tab
    eol = text.find(nl, pos)
    pos = eol + len(nl)
    return text[:pos] + f" *{nl}" + new_par + text[pos:], "added"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("plugins", help="SASfitGit/src/plugins")
    ap.add_argument("--out", help="write changed headers below this folder instead of in place")
    ap.add_argument("--check", action="store_true", help="only report")
    a = ap.parse_args(argv)
    from pysasfit.studio import ports

    by_header = {}
    for spec in ports.specs():
        if not spec.source.startswith("plugins/"):
            continue
        plugin = spec.source.split("/")[1]
        by_header.setdefault(os.path.join(plugin, "include", f"sasfit_{plugin}.h"), []).append(spec)
    for rel, specs in sorted(by_header.items()):
        path = os.path.join(a.plugins, rel)
        if not os.path.exists(path):
            print(f"{rel}: missing")
            continue
        raw = open(path, "rb").read()
        text = raw.decode("latin-1")
        nl = "\r\n" if "\r\n" in text else "\n"
        report = []
        for spec in sorted(specs, key=lambda s: s.name):
            text, status = update_header(text, spec.name, defaults_line(spec.params), nl)
            report.append(f"{spec.name}: {status}")
        changed = text.encode("latin-1") != raw
        print(f"{rel}: {'changed' if changed else 'unchanged'} ({'; '.join(report)})")
        if changed and not a.check:
            dest = os.path.join(a.out, rel) if a.out else path
            os.makedirs(os.path.dirname(dest), exist_ok=True)
            with open(dest, "wb") as fh:
                fh.write(text.encode("latin-1"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
