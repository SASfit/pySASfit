"""Write the default values and domains of the Python ports into the parameter tables of the SASfit plugin headers.

    python tools/write_header_defaults.py <SASfitGit/src/plugins> [--out DIR] [--check] [--force]

For every ported function each row of the parameter table of its ``\\defgroup`` block gets two cells after the
description (syntax agreed with J. Kohlbrecher, 2026-11-24)::

     *       <td>\\b R_c</td>
     *       <td>core radius containing solvent</td>
     *       <td>(default=10)</td>
     *       <td>[0;+inf)</td>

The domain uses ``;`` as separator, ``+inf`` / ``-inf``, square brackets for closed and round brackets for open
bounds. A row without description gets an empty description cell, so the order is always name, description,
default, domain. The older paragraph ``\\par Default values and domains:`` is removed.

Values already in the header are kept when they differ from the port (they were set by hand; the difference is
reported, so that the port can be updated) unless ``--force``. Line endings and encoding of the headers are kept.
``--out``: write the changed headers below DIR (same relative paths) instead of in place. ``--check``: only report.
"""

from __future__ import annotations

import argparse
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))



def _num(v):
    """Short exact-enough text of a number (12 significant digits); +inf / -inf."""
    if v == float("inf"):
        return "+inf"
    if v == float("-inf"):
        return "-inf"
    a = f"{v:.12g}"
    if v == 0 or 1e-4 <= abs(v) < 1e6:
        return a
    m, _, e = f"{v:.12e}".partition("e")
    m = m.rstrip("0").rstrip(".")
    b = f"{m}e{int(e)}"            # 1e10, -5e9, 2.5e-7
    return b if len(b) < len(a) else a


def _domain(p):
    lo = "(" if (p.lower_open or p.lower == float("-inf")) else "["
    hi = ")" if (p.upper_open or p.upper == float("inf")) else "]"
    return f"{lo}{_num(p.lower)};{_num(p.upper)}{hi}"


_CELL = re.compile(r"<td[^>]*>(.*?)</td>", re.S)
_NUMRE = r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?|[-+]?inf"
_IS_DEFAULT = re.compile(rf"^\(\s*default\s*=\s*({_NUMRE})\s*\)$", re.I)
_IS_DOMAIN = re.compile(rf"^([\[(])\s*({_NUMRE}|[A-Za-z_][\w.]*)\s*[;,]\s*({_NUMRE}|[A-Za-z_][\w.]*)\s*([\])])$", re.I)
_ROW = re.compile(r"(?P<indent>[ \t]*\*[ \t]*)(?P<row><td>\s*\\b.*?)(?=\r?\n[ \t]*\*[ \t]*</tr>)", re.S)


def _drop_paragraph(block):
    """Remove the older paragraph "\\par Default values and domains:" (and the empty comment line before it)."""
    lines = block.splitlines(keepends=True)
    for i, ln in enumerate(lines):
        if "\\par Default values and domains:" in ln:
            j = i + 1
            while j < len(lines):
                t = lines[j].strip()
                if not t.startswith("*") or t in ("*", "*/") or t.lstrip("* ").startswith("\\") or t.startswith("*/"):
                    break
                j += 1
            k = i - 1 if i > 0 and lines[i - 1].strip() == "*" else i
            return "".join(lines[:k] + lines[j:])
    return block


def _table_rows(text, name):
    """Number of parameter rows in the table of ``\\defgroup name`` (None: no table)."""
    m = re.search(rf"\\defgroup\s+{re.escape(name)}\b", text)
    if not m:
        return None
    end = text.find("*/", m.end())
    block = text[m.end() : end]
    a, b = block.find("<table"), block.find("</table>")
    if a < 0 or b < 0:
        return None
    return len(list(_ROW.finditer(block, a, b)))


def _same(a, b):
    try:
        x, y = float(a), float(b)
    except ValueError:
        return a == b
    return x == y or abs(x - y) <= 1e-9 * max(abs(x), abs(y))


def update_header(text, name, params, nl, force=False):
    """Default/domain cells in the table of ``\\defgroup name`` (one row per entry of ``params``, in order);
    returns (new text, status, notes)."""
    m = re.search(rf"\\defgroup\s+{re.escape(name)}\b", text)
    if not m:
        return text, "no \\defgroup", []
    end = text.find("*/", m.end())
    if end < 0:
        return text, "no end of the comment block", []
    block = text[m.end() : end]
    tab_start = block.find("<table")
    tab_end = block.find("</table>")
    if tab_start < 0 or tab_end < 0:
        return text, "no parameter table", []
    rows = list(_ROW.finditer(block, tab_start, tab_end))
    notes = []
    if len(rows) != len(params):
        return text, f"table has {len(rows)} rows, the port {len(params)} parameters – not written", []
    pieces, pos = [], 0
    for r, p in zip(rows, params):
        cells = [c.strip() for c in _CELL.findall(r.group("row"))]
        name_cell = cells[0]
        desc, old_def, old_dom, extra = None, None, None, []
        for c in cells[1:]:
            md, mb = _IS_DEFAULT.match(c), _IS_DOMAIN.match(c)
            if md:
                old_def = md.group(1)
            elif mb:
                old_dom = c
            elif desc is None:
                desc = c
            else:
                extra.append(c)
        new_def, new_dom = _num(p.default), _domain(p)
        if old_def is not None and _same(old_def, new_def):
            new_def = old_def          # same value: keep how it was written ("1e10")
        if not force and old_def is not None and not _same(old_def, new_def):
            notes.append(f"{p.name}: header default {old_def} kept (port {new_def})")
            new_def = old_def
        if not force and old_dom is not None:
            mo = _IS_DOMAIN.match(old_dom)
            mn = _IS_DOMAIN.match(new_dom)
            if not (mo.group(1) == mn.group(1) and mo.group(4) == mn.group(4) and _same(mo.group(2), mn.group(2))
                    and _same(mo.group(3), mn.group(3))):
                notes.append(f"{p.name}: header domain {old_dom} kept (port {new_dom})")
                new_dom = old_dom
            else:
                new_dom = old_dom
        ind = r.group("indent")
        out_cells = [name_cell, desc if desc is not None else "", *extra, f"(default={new_def})", new_dom]
        row = (nl + ind).join(f"<td>{c}</td>" for c in out_cells)
        pieces.append(block[pos : r.start("row")])
        pieces.append(row)
        pos = r.end("row")
    pieces.append(block[pos:])
    new_block = _drop_paragraph("".join(pieces))
    new = text[: m.end()] + new_block + text[end:]
    return new, ("unchanged" if new == text else "written"), notes


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("plugins", help="SASfitGit/src/plugins")
    ap.add_argument("--out", help="write changed headers below this folder instead of in place")
    ap.add_argument("--check", action="store_true", help="only report")
    ap.add_argument("--force", action="store_true", help="overwrite defaults/domains that differ in the header")
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
            prm = list(spec.params)
            n = _table_rows(text, spec.name)
            if n is not None and n < len(prm) and all(p.kind == "dummy" or p.name in ("dummy", "-") for p in prm[n:]):
                prm = prm[:n]      # trailing unused parameters that the header table does not list
            elif n is not None and n != len(prm):
                report.append(f"{spec.name}: header table has {n} parameters, the port {len(prm)} – not written")
                continue
            text, status, notes = update_header(text, spec.name, prm, nl, a.force)
            report.append(f"{spec.name}: {status}" + (f" [{'; '.join(notes)}]" if notes else ""))
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
