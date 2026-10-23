"""Release checks and build for pySASfit.

    python tools/release.py                      # tests, screenshots, HTML documentation, package
    python tools/release.py --skip-tests         # skip a step (see --help)
    python tools/release.py --only screenshots   # run one step

Steps, in this order (the first one that fails stops the release):

    tests        pytest tests/studio          (the other tool folders: --all-tests)
    screenshots  tools/studio_screenshots.py  renews docs/_static/studio/*.png from the current GUI
    docs         sphinx -b html docs docs/_build/html   (warnings are shown; --strict makes them errors)
    package      python -m build              sdist and wheel in dist/ (needs the 'build' package)

The screenshots need PySide6, matplotlib and the SASfit libraries. They are renewed *before* the documentation is
built, so the HTML shows the current GUI. Nothing is tagged, committed or uploaded: check ``git status`` for the
changed pictures, commit them, then tag and publish yourself.
"""

import argparse
import os
import re
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY = sys.executable
STEPS = ("tests", "screenshots", "docs", "package")


def version():
    with open(os.path.join(ROOT, "pyproject.toml"), encoding="utf-8") as fh:
        m = re.search(r'^version\s*=\s*"([^"]+)"', fh.read(), re.M)
    return m.group(1) if m else "?"


def run(title, cmd, env=None):
    print(f"\n=== {title}\n$ {' '.join(cmd)}", flush=True)
    t0 = time.time()
    r = subprocess.run(cmd, cwd=ROOT, env={**os.environ, **(env or {})})
    print(f"--- {title}: {'ok' if r.returncode == 0 else 'FAILED (exit %d)' % r.returncode}, {time.time() - t0:.0f} s")
    return r.returncode == 0


def step_tests(a):
    return run("tests", [PY, "-m", "pytest", "tests" if a.all_tests else "tests/studio", "-q"])


def step_screenshots(a):
    return run("screenshots", [PY, os.path.join("tools", "studio_screenshots.py")])


def step_docs(a):
    cmd = [PY, "-m", "sphinx", "-b", "html", "docs", os.path.join("docs", "_build", "html")]
    if a.strict:
        cmd.insert(-2, "-W")
    return run("documentation", cmd)


def step_package(a):
    return run("package", [PY, "-m", "build"])


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for s in STEPS:
        ap.add_argument(f"--skip-{s}", action="store_true", help=f"do not run the '{s}' step")
    ap.add_argument("--only", choices=STEPS, help="run this step only")
    ap.add_argument("--all-tests", action="store_true", help="the tests step runs every test folder (slow)")
    ap.add_argument("--strict", action="store_true", help="documentation warnings are errors")
    a = ap.parse_args(argv)

    print(f"pySASfit {version()}  ({ROOT})")
    todo = [a.only] if a.only else [s for s in STEPS if not getattr(a, f"skip_{s}")]
    for s in todo:
        if not globals()[f"step_{s}"](a):
            print(f"\nrelease stopped: step '{s}' failed")
            return 1
    print("\nall steps done. Next: git status (changed screenshots), commit, tag, publish.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
