"""Copy the logos the package needs from the logo bundle into ``src/pysasfit/resources/``.

    python tools/sync_logos.py

The bundle (``docs/pySASfit-logo-bundle``) is the master. The package gets only what the programs use: the square
icons (svg, small svg, ico), the menu icons (svg, ico) and the light and dark horizontal logos (svg) of every tool, so that icons also work in an
installed (not editable) package. ``release.py`` runs this before the package is built.
"""

import os
import shutil
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BUNDLE = os.path.join(ROOT, "docs", "pySASfit-logo-bundle")
DEST = os.path.join(ROOT, "src", "pysasfit", "resources")


def sync():
    """Copy the files; returns the number of files that were new or changed."""
    if not os.path.isdir(BUNDLE):
        print(f"no logo bundle at {BUNDLE}")
        return 0
    n = 0
    for kind in ("icons", "menu-icons", "logos"):
        src_kind = os.path.join(BUNDLE, kind)
        for tool in sorted(os.listdir(src_kind)):
            for dirpath, dirs, files in os.walk(os.path.join(src_kind, tool)):
                rel = os.path.relpath(dirpath, src_kind)
                if rel.split(os.sep)[-1] in ("png", "glyph"):  # the programs use the svg files (and the ico)
                    dirs[:] = []
                    continue
                for f in files:
                    if kind == "logos" and not f.endswith(("-light.svg", "-dark.svg")):
                        continue
                    if not f.endswith((".svg", ".ico")):
                        continue
                    target = os.path.join(DEST, kind, rel, f)
                    source = os.path.join(dirpath, f)
                    if os.path.isfile(target) and os.path.getmtime(target) >= os.path.getmtime(source):
                        continue
                    os.makedirs(os.path.dirname(target), exist_ok=True)
                    shutil.copy2(source, target)
                    n += 1
    print(f"{n} files copied to {DEST}")
    return n


if __name__ == "__main__":
    sync()
    sys.exit(0)
