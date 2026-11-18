"""Copy SASfit plugin sources into sasfit_src/ for the C reference of the Python ports.

    python tests/studio/port_reference/import_plugin.py <SASfitGit/src/plugins> <plugin> [<plugin> ...]

For every plugin the .c files (without interface.c) are copied (LF line endings) and include/private.h is
replaced by a shim version: the local definitions of the original private.h after the plugin header, the shim
header and prototypes of the functions the sources define or call (``sasfit_ff_*``, ``sasfit_sq_*``, ...), so
that cross-file calls are typed correctly without the SASfit headers. The plugin-import macros of the original
(``#ifdef MAKE_SASFIT_PLUGIN``) are dropped: imported functions are resolved by name in the combined library.
"""

from __future__ import annotations

import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))

_DEF = re.compile(r"^(?:static\s+)?(scalar|double|int|void|float)\s+\**\s*(\w+)\s*\(([^;{)]*)\)\s*\{", re.M)
_CALL = re.compile(r"\b(sasfit_(?:ff|sq|sd|peaks)_\w+)\s*\(")


def _read(path):
    with open(path, encoding="latin-1") as fh:
        return fh.read().replace("\r\n", "\n").replace("\r", "\n")


def _strip_comments(s):
    s = re.sub(r"/\*.*?\*/", "", s, flags=re.S)
    return re.sub(r"//[^\n]*", "", s)


def _local_part(private):
    """The definitions of the original private.h that follow the plugin header include."""
    s = private
    m = re.search(r'#include\s+"sasfit_\w+\.h"\s*\n', s)
    if m:
        s = s[m.end():]
    s = re.sub(r"#ifdef\s+MAKE_SASFIT_PLUGIN.*?#endif", "", s, flags=re.S)
    s = re.sub(r"#include\s*<sasfit_(common|plugin)\.h>\s*\n", "", s)
    s = re.sub(r"#endif[^\n]*\s*$", "", s.rstrip())  # the include guard
    return s.strip()


def import_plugin(src_root, name, dest_root=None):
    src = os.path.join(src_root, name)
    dest = os.path.join(dest_root or os.path.join(HERE, "sasfit_src"), name)
    os.makedirs(os.path.join(dest, "include"), exist_ok=True)
    defined, called, sources = {}, set(), []
    for fn in sorted(os.listdir(src)):
        if not fn.endswith(".c") or fn == "interface.c":
            continue
        text = _read(os.path.join(src, fn))
        sources.append((fn, text))
        body = _strip_comments(text)
        for ret, fname, args in _DEF.findall(body):
            if "static" not in body[: body.find(fname)].rsplit("\n", 1)[-1]:
                defined[fname] = f"{ret} {fname}({' '.join(args.split())});"
        called.update(_CALL.findall(body))
    protos = list(defined.values())
    for fname in sorted(called - set(defined)):
        args = "scalar, sasfit_param *, int" if fname.endswith("_v") else "scalar, sasfit_param *"
        protos.append(f"scalar {fname}({args});")
    private = os.path.join(src, "include", "private.h")
    local = _local_part(_read(private)) if os.path.exists(private) else ""
    guard = f"SHIM_{name.upper()}_PRIVATE_H"
    with open(os.path.join(dest, "include", "private.h"), "w", newline="\n") as fh:
        fh.write(f"/* shim: replaces the SASfit private.h of plugin {name} (tests/studio/port_reference/import_plugin.py) */\n")
        fh.write(f"#ifndef {guard}\n#define {guard}\n#include <sasfit_shim.h>\n\n")
        fh.write("\n".join(protos) + "\n\n")
        if local:
            fh.write(local + "\n")
        fh.write(f"#endif\n")
    for fn, text in sources:
        with open(os.path.join(dest, fn), "w", newline="\n", encoding="latin-1") as fh:
            fh.write(text)
    return [fn for fn, _ in sources]


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(1)
    for plugin in sys.argv[2:]:
        print(plugin, import_plugin(sys.argv[1], plugin))
