"""The example scripts and the manual that includes them.

* always: every script is valid Python, every ``literalinclude`` of the manual finds its file and its markers, and the
  screenshots named in the manual are the ones ``tools/studio_screenshots.py`` makes;
* with the SASfit libraries installed: the scripts are run (without opening windows).
"""

import ast
import os
import re
import runpy
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
EXAMPLES = ROOT / "examples" / "studio"
MANUAL = ROOT / "docs" / "studio_manual"
SCRIPTS = sorted(EXAMPLES.glob("0*.py"))

pytestmark = pytest.mark.skipif(not EXAMPLES.is_dir() or not MANUAL.is_dir(), reason="examples or manual not in this checkout")


def test_there_are_examples():
    assert len(SCRIPTS) >= 5


@pytest.mark.parametrize("script", SCRIPTS, ids=lambda p: p.name)
def test_example_is_valid_python(script):
    ast.parse(script.read_text(encoding="utf-8"), filename=str(script))


def _includes():
    """(page, file, start marker, end marker) of every literalinclude in the manual."""
    pat = re.compile(r"```\{literalinclude\}\s+(\S+)\n((?::[^\n]*\n)*)", re.M)
    for page in sorted(MANUAL.glob("*.md")):
        for m in pat.finditer(page.read_text(encoding="utf-8")):
            opts = dict(re.findall(r':(\S+):\s*"?([^"\n]*)"?', m.group(2)))
            yield page, m.group(1), opts.get("start-after"), opts.get("end-before")


@pytest.mark.parametrize("page, path, start, end", list(_includes()), ids=lambda v: str(v)[-40:])
def test_manual_includes_resolve(page, path, start, end):
    target = (page.parent / path).resolve()
    assert target.is_file(), f"{page.name}: {path} does not exist"
    text = target.read_text(encoding="utf-8")
    for marker in (start, end):
        if marker:
            assert marker in text, f"{page.name}: marker {marker!r} not found in {target.name}"
    if start and end:
        assert text.index(start) < text.index(end)


def test_the_manual_has_includes():
    assert len(list(_includes())) >= 8


def test_screenshots_in_the_manual_are_made_by_the_script():
    tree = ast.parse((ROOT / "tools" / "studio_screenshots.py").read_text(encoding="utf-8"))
    names = next(
        ast.literal_eval(node.value)
        for node in tree.body
        if isinstance(node, ast.Assign) and any(getattr(t, "id", "") == "NAMES" for t in node.targets)
    )
    used = set()
    for page in MANUAL.glob("*.md"):
        used |= set(re.findall(r"_static/studio/(\w+)\.png", page.read_text(encoding="utf-8")))
    assert used, "no screenshots referenced"
    assert used <= set(names), f"made by no script: {sorted(used - set(names))}"


def test_every_toctree_page_exists():
    index = (MANUAL / "index.md").read_text(encoding="utf-8")
    block = re.search(r"```\{toctree\}\n(?::[^\n]*\n)*\n(.*?)```", index, re.S).group(1)
    for line in block.splitlines():
        line = line.strip()
        if not line:
            continue
        name = re.search(r"<(.+)>", line)
        name = name.group(1) if name else line
        assert (MANUAL / (name + ".md")).is_file(), f"toctree entry {line!r} has no page"


# ---- running the scripts (needs the SASfit libraries) ------------------------------------------------------------------


def _have_library():
    try:
        from pysasfit.studio import paths

        return paths.lib_dir() is not None
    except Exception:  # noqa: BLE001
        return False


needs_library = pytest.mark.skipif(not _have_library(), reason="no SASfit libraries")


@pytest.fixture()
def no_windows(monkeypatch):
    mpl = pytest.importorskip("matplotlib")
    mpl.use("Agg", force=True)
    monkeypatch.setenv("MPLBACKEND", "Agg")


@needs_library
@pytest.mark.parametrize("name", ["01_evaluate_functions.py", "02_mixture_model.py"])
def test_function_and_model_examples_run(name, no_windows):
    runpy.run_path(str(EXAMPLES / name), run_name="__main__")


@needs_library
def test_the_fit_example_runs_and_fits(no_windows):
    ns = runpy.run_path(str(EXAMPLES / "03_fit_with_engine.py"), run_name="example03")
    res = ns["res"]
    assert res.chi2_red < 5  # SASfit's own fit of this data set has chi2/dof of about 1


@needs_library
def test_the_export_example_writes_the_files(no_windows):
    pytest.importorskip("openpyxl")
    runpy.run_path(str(EXAMPLES / "04_export_results.py"), run_name="example04")
    out = EXAMPLES / "output"
    # save_csv writes several files next to the name it is given: fit_parameters.csv, fit_curve.csv, ...
    for name in ("fit.xlsx", "fit_parameters.csv", "fit_curve.csv", "fit_distributions.csv", "fit.pdf"):
        assert (out / name).stat().st_size > 0
