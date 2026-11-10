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
import re
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
# sphinx-design: the cards on the landing page (pip install sphinx-design).
# Without it, setup() below registers plain stand-ins for grid / grid-item-card:
# the cards become a list of linked titles with their text, and the build does
# not warn.
try:
    import sphinx_design  # noqa: F401
    extensions.append("sphinx_design")
    _HAVE_SPHINX_DESIGN = True
except ImportError:
    _HAVE_SPHINX_DESIGN = False

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
    "PySide6", "matplotlib", "bumps", "nlopt", "mixscatter",
    "jscatter", "openpyxl", "mpmath")
    if _ilu.find_spec(m) is None]

# The same loop happens with autodoc's mock of a MISSING module once the real
# PySide6 is imported (by the GUI pages): the mock answers every attribute,
# __wrapped__ included. sundials4py (only used inside the OZ solvers) is
# therefore replaced by a plain stub module that has no dunder attributes.
if _ilu.find_spec("sundials4py") is None:
    import importlib.abc as _iabc
    import importlib.machinery as _imach
    import types as _types

    class _Stub:
        """Stand-in for any object of the missing package."""
        def __init__(self, *a, **k):
            pass

        def __getattr__(self, name):
            if name.startswith("__") and name.endswith("__"):
                raise AttributeError(name)
            return _Stub()

        def __call__(self, *a, **k):
            return _Stub()

    class _StubModule(_types.ModuleType):
        def __getattr__(self, name):
            if name.startswith("__") and name.endswith("__"):
                raise AttributeError(name)
            return _Stub()

    class _StubFinder(_iabc.MetaPathFinder, _iabc.Loader):
        roots = ("sundials4py",)

        def find_spec(self, fullname, path=None, target=None):
            if fullname.split(".")[0] in self.roots:
                return _imach.ModuleSpec(fullname, self, is_package=True)
            return None

        def create_module(self, spec):
            mod = _StubModule(spec.name)
            mod.__path__ = []
            return mod

        def exec_module(self, module):
            pass

    sys.meta_path.insert(0, _StubFinder())

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "scipy": ("https://docs.scipy.org/doc/scipy", None),
}

# Layout of NumPy/SciPy/pandas: pydata-sphinx-theme -- top bar with the main
# sections, section menu on the left, "On this page" on the right, light/dark
# switch. Falls back to furo / Read the Docs / alabaster when not installed.
_themes = ( "sphinx_rtd_theme", "sphinxawesome_theme", "shibuya", "sphinx_material", "furo", "pydata_sphinx_theme")
for _theme in _themes:
    try:
        __import__(_theme)
        html_theme = _theme
        break
    except ImportError:
        continue
else:
    html_theme = "alabaster"
# Logos from the logo bundle (docs/pySASfit-logo-bundle, see its README.txt): the horizontal pySASfit logo in the
# header (light and dark variant with pydata-sphinx-theme, the light one with other themes), the tile as favicon.
# The tools' own logos are placed on their pages (docs/*/index.*) with the classes only-light / only-dark.
_BUNDLE = "pySASfit-logo-bundle"
_LOGO_DIR = _BUNDLE + "/logos/pySASfit/svg"
html_static_path = ["_static", _LOGO_DIR]
html_css_files = ["logos.css"]
html_logo = _LOGO_DIR + "/pySASfit-light.svg"
html_favicon = _BUNDLE + "/icons/pySASfit/pySASfit.ico"
html_title = f"pySASfit {release}"
html_scaled_image_link = False      # images with :height: are not links (e.g. the card icons)
html_short_title = "pySASfit"

if html_theme == "sphinx_rtd_theme":
    # The Read the Docs theme puts the logo on a coloured block above the menu (default: medium blue). The grey
    # "py" and "fit" of the light logo have too little contrast there: use the navy of the logo bundle for the block
    # and the logo variant made for dark backgrounds (light grey "py"/"fit", white "SAS").
    html_logo = _LOGO_DIR + "/pySASfit-dark.svg"
    html_theme_options = {"style_nav_header_background": "#0E2740", "logo_only": False}

if html_theme == "pydata_sphinx_theme":
    html_theme_options = {
        "logo": {"image_light": "pySASfit-light.svg", "image_dark": "pySASfit-dark.svg"},
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


# ---------------------------------------------------------------------------
# Stand-ins for the sphinx-design directives used on the landing page
# (only when sphinx-design is not installed).
from docutils import nodes as _nodes
from docutils.parsers.rst import Directive as _Directive, directives as _dirs


class _GridFallback(_Directive):
    has_content = True
    optional_arguments = 4
    final_argument_whitespace = True
    option_spec = {"gutter": _dirs.unchanged, "class-container": _dirs.unchanged,
                   "class-row": _dirs.unchanged, "margin": _dirs.unchanged,
                   "padding": _dirs.unchanged}

    def run(self):
        box = _nodes.container(classes=["sd-fallback-grid"])
        self.state.nested_parse(self.content, self.content_offset, box)
        return [box]


class _CardFallback(_Directive):
    has_content = True
    optional_arguments = 1
    final_argument_whitespace = True
    option_spec = {"link": _dirs.unchanged, "link-type": _dirs.unchanged,
                   "class-card": _dirs.unchanged, "shadow": _dirs.unchanged,
                   "text-align": _dirs.unchanged, "img-top": _dirs.unchanged}

    def run(self):
        from sphinx import addnodes
        card = _nodes.container(classes=["sd-fallback-card"])
        title = self.arguments[0] if self.arguments else ""
        if title:
            para = _nodes.paragraph()
            text = _nodes.strong(text=title)
            link = self.options.get("link")
            if link and self.options.get("link-type", "url") == "doc":
                env = self.state.document.settings.env
                ref = addnodes.pending_xref("", refdomain="std", reftype="doc",
                                            reftarget=link, refexplicit=True,
                                            refdoc=env.docname, refwarn=True)
                ref += text
                para += ref
            elif link:
                ref = _nodes.reference("", "", refuri=link)
                ref += text
                para += ref
            else:
                para += text
            card += para
        self.state.nested_parse(self.content, self.content_offset, card)
        return [card]


# ---------------------------------------------------------------------------
# Formulas in plain text. Many docstrings write |q|^2, |G| or |x|+|y| as ordinary text; reStructuredText reads
# |name| as a substitution reference and warns ("Undefined substitution referenced"). Every | is therefore
# escaped when autodoc inserts a docstring -- except in literal blocks (after "::"), directive bodies, inline
# literals and roles in backticks, and at the start of line blocks and table rows ("| ", "+-").
_INLINE_LITERAL = re.compile(r"(``.+?``|`[^`\n]+`)")


def _escape_pipes(lines):
    out, literal_indent = [], None
    for line in lines:
        stripped = line.strip()
        indent = len(line) - len(line.lstrip())
        if literal_indent is not None:
            if not stripped or indent > literal_indent:
                out.append(line)
                continue
            literal_indent = None
        if stripped.startswith("..") and stripped.endswith("::"):      # directive: its body stays as it is
            literal_indent = indent
            out.append(line)
            continue
        if stripped.endswith("::"):
            literal_indent = indent
        if "|" not in line or stripped.startswith(("| ", "+-", "+=")) or stripped == "|":
            out.append(line)
            continue
        parts = _INLINE_LITERAL.split(line)
        out.append("".join(p if i % 2 else re.sub(r"(?<!\\)\|", r"\\|", p) for i, p in enumerate(parts)))
    return out


def _process_docstring(app, what, name, obj, options, lines):
    lines[:] = _escape_pipes(lines)


def setup(app):
    app.connect("autodoc-process-docstring", _process_docstring)
    if not _HAVE_SPHINX_DESIGN:
        app.add_directive("grid", _GridFallback)
        app.add_directive("grid-item-card", _CardFallback)
        app.add_directive("grid-item", _GridFallback)
        app.add_directive("card", _CardFallback)
