"""Logos and icons (pysasfit.resources): found in the package or in the logo bundle; the sync script is consistent."""

import os

import pytest

from pysasfit import resources

BUNDLE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "docs", "pySASfit-logo-bundle")

needs_bundle = pytest.mark.skipif(not os.path.isdir(BUNDLE) and resources.icon_path("pySASfit") is None, reason="no logos")


@needs_bundle
@pytest.mark.parametrize("tool", resources.TOOLS)
def test_every_tool_has_icons_and_logos(tool):
    assert resources.icon_path(tool) and resources.icon_path(tool, small=True)
    for variant in ("light", "dark"):
        assert resources.logo_path(tool, variant)


def test_unknown_tool_gives_none():
    assert resources.icon_path("nothing") is None and resources.logo_path("nothing") is None


@needs_bundle
def test_tool_icon_is_a_qicon():
    pytest.importorskip("PySide6")
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6 import QtWidgets

    QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    assert not resources.tool_icon("pySASstudio").isNull()
    assert resources.tool_icon("nothing").isNull()


@needs_bundle
def test_docs_logo_paths_exist():
    """Every logo the documentation refers to is in the bundle."""
    import re

    docs = os.path.join(os.path.dirname(BUNDLE))
    for root, _dirs, files in os.walk(docs):
        if "_build" in root or "pySASfit-logo-bundle" in root:
            continue
        for f in files:
            if f.endswith((".md", ".rst")):
                path = os.path.join(root, f)
                text = open(path, encoding="utf-8", errors="replace").read()
                for ref in re.findall(r"pySASfit-logo-bundle/[\w./-]+\.(?:svg|png|ico)", text):
                    assert os.path.isfile(os.path.join(docs, ref)), f"{path}: {ref}"


@needs_bundle
@pytest.mark.parametrize("tool", resources.TOOLS)
def test_every_tool_has_a_menu_icon(tool):
    assert resources.menu_icon_path(tool) and resources.menu_icon_path(tool, "ico")


def test_old_tool_name_is_accepted():
    """pySASbox was renamed pyTOOLbox."""
    assert resources.wordmark("pySASbox") == resources.wordmark("pyTOOLbox")
    assert [seg[0] for seg in resources.wordmark("pyTOOLbox")] == ["py", "TOOL", "box"]
    assert resources.wordmark("pyTOOLbox")[1][1] == "#D9730D"        # orange core as in the logo
    assert resources.wordmark("pySASred")[1][1] == "#C8283C"         # each core in the tool's colour
    assert resources.wordmark("pySASfit")[1][1] == "#0E2740"         # the suite: navy
    assert {seg[2][1] for seg in map(resources.wordmark, resources.TOOLS)} == {"#5E7388"}   # ending grey as "py"
    assert [seg[0] for seg in resources.wordmark("pyINVersion")] == ["py", "INV", "ersion"]
    assert resources.wordmark("pyInversion") == resources.wordmark("pyINVersion")     # old name
    if resources.menu_icon_path("pyTOOLbox"):
        assert resources.menu_icon_path("pySASbox") == resources.menu_icon_path("pyTOOLbox")
