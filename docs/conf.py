# -*- coding: utf-8 -*-
"""Sphinx configuration for the whole pySASfit documentation.

Build (from the repository root):

    pip install -e ".[docs]"
    python -m sphinx -b html docs docs/_build/html

or ``make html`` / ``make.bat html`` in docs/. Open docs/_build/html/index.html.

One project for all tools. The pages come from the former separate projects:
the OZ pages (Markdown, from pyOZgui), the inversion pages (reStructuredText,
from pySASfitInversion) and the SLD-profile pages (Markdown, from pySLDprofile).
MyST reads Markdown, so both formats live side by side.

NOT part of this build (they have their own builds):
  * oz/latex/            -- the OZ report (build_report.sh) and the manuscript
                            (oz/latex/manuscript/build.sh), LaTeX + BibTeX
  * inversion/report_build/ -- the inversion PDF report (Sphinx LaTeX builder)
  * oz/notes/            -- internal working notes from the pyOZgui development
"""
import os
import sys

# Build from a checkout without installing: the package is in ../src.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

project = "pySASfit"
author = "Joachim Kohlbrecher"
copyright = "2026, Paul Scherrer Institut"
try:
    from pysasfit import __version__ as release
except Exception:                      # docs must build even if the import fails
    release = "0.2.0"
version = release

extensions = [
    "sphinx.ext.autodoc",       # API reference generated from the docstrings
    "sphinx.ext.napoleon",      # NumPy/Google-style sections
    "sphinx.ext.viewcode",      # [source] links
    "sphinx.ext.intersphinx",   # numpy/scipy names become links
    "sphinx.ext.mathjax",
    "myst_parser",              # Markdown pages (OZ, SLD profile, preferences, studio)
]
# sphinx-design: the cards on the landing page (optional; without it the
# cards render as plain text)
try:
    import sphinx_design  # noqa: F401
    extensions.append("sphinx_design")
except ImportError:
    pass

source_suffix = {".rst": "restructuredtext", ".md": "markdown"}
myst_enable_extensions = ["deflist", "colon_fence", "dollarmath"]
myst_heading_anchors = 3

exclude_patterns = [
    "_build",
    "oz/latex",                 # LaTeX report and manuscript (own build)
    "oz/notes",                 # internal development notes
    "inversion/report_build",   # inversion PDF report (own Sphinx project)
    "Thumbs.db", ".DS_Store",
]

autodoc_member_order = "bysource"
autodoc_typehints = "description"
autodoc_default_options = {
    "members": True,
    "undoc-members": False,
    "show-inheritance": True,
}
# Optional back ends and GUI toolkits are mocked when missing: autodoc imports
# every module, and the docs must build where SUNDIALS, PySide6 etc. are not
# installed.
#
# Only modules that are NOT installed are mocked. Mocking an installed one
# breaks things: PySide6's import hook (shiboken) inspects every module
# imported after it, and a mock object sends it into an endless unwrap loop
# ("wrapper loop when unwrapping sundials4py"), so ozLib failed to import.
import importlib.util as _ilu

autodoc_mock_imports = [m for m in (
    "PySide6", "matplotlib", "sundials4py", "bumps", "nlopt", "mixscatter",
    "jscatter", "openpyxl", "mpmath")
    if _ilu.find_spec(m) is None]

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "scipy": ("https://docs.scipy.org/doc/scipy", None),
}

# Layout of NumPy/SciPy/pandas: pydata-sphinx-theme -- top bar with the main
# sections, section menu on the left, "On this page" on the right, light/dark
# switch. Falls back to furo / Read the Docs / alabaster when not installed.
_themes = ("pydata_sphinx_theme", "furo", "sphinx_rtd_theme")
for _theme in _themes:
    try:
        __import__(_theme)
        html_theme = _theme
        break
    except ImportError:
        continue
else:
    html_theme = "alabaster"
html_static_path = ["_static"]
html_title = f"pySASfit {release}"
html_short_title = "pySASfit"

if html_theme == "pydata_sphinx_theme":
    html_theme_options = {
        "logo": {"text": "pySASfit"},
        "icon_links": [
            {"name": "GitHub", "url": "https://github.com/SASfit/pySASfit",
             "icon": "fa-brands fa-github", "type": "fontawesome"},
            {"name": "SASfit", "url": "https://github.com/SASfit/SASfit",
             "icon": "fa-solid fa-flask", "type": "fontawesome"},
        ],
        "header_links_before_dropdown": 7,
        "navbar_end": ["theme-switcher", "navbar-icon-links"],
        "navbar_persistent": ["search-button"],
        "navbar_align": "left",
        "show_toc_level": 2,
        "navigation_depth": 3,
        "show_prev_next": True,
        "footer_start": ["copyright"],
        "footer_end": ["sphinx-version", "theme-version"],
        "secondary_sidebar_items": ["page-toc", "sourcelink"],
    }
    html_sidebars = {"index": []}      # landing page: cards, no left menu
