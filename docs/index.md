---
html_theme.sidebar_secondary.remove: true
---

# pySASfit

**Python tools for small-angle scattering (SANS/SAXS) from the SASfit project**, in one installable package
`pysasfit`. Every tool can be used from a script and from a Qt GUI; `pysasfit-gui` opens all of them as tabs of
one window.

```bash
pip install -e ".[gui]"      # from a checkout of github.com/SASfit/pySASfit
pysasfit-gui
```

::::{grid} 1 2 2 2
:gutter: 3

:::{grid-item-card} SLD profile
:link: sldprofile/index
:link-type: doc

Radial scattering-length-density profiles ρ(r) of spheres, cylinders and lamellae: direct Bayesian estimation
(Hansen 2016) or Glatter's two-step route.
:::

:::{grid-item-card} Size distribution / p(r)
:link: inversion/index
:link-type: doc

Fredholm inversion: size distributions and pair-distance distributions with EM, MaxEnt, Tikhonov, SVD, ARLS,
Bayesian-evidence IFT and splines.
:::

:::{grid-item-card} Ornstein–Zernike
:link: oz/index
:link-type: doc

OZ solvers and 24 closures, polydisperse and multicomponent structure factors, SAS models, analytic references,
fitting.
:::

:::{grid-item-card} studio
:link: studio
:link-type: doc

SASfit's plugin and libsasfit model functions via ctypes, combined into models and fitted.
:::

:::{grid-item-card} Preferences
:link: preferences
:link-type: doc

Every numerical setting that is not on the panels, with its factory value and an explanation.
:::

:::{grid-item-card} API reference
:link: api/index
:link-type: doc

Signatures and docstrings of all modules, generated from the source.
:::
::::

## Documents built separately

| document | source | build |
|---|---|---|
| OZ report (ozLib/ozGUI documentation and multicomponent supplement) | `docs/oz/latex/ozGUI_ozLib_report.tex` | `sh build_report.sh` in `docs/oz/latex` |
| OZ manuscript | `docs/oz/latex/manuscript/manuscript.tex` | `sh build.sh` in `docs/oz/latex/manuscript` |
| inversion report (PDF of the inversion pages) | `docs/inversion/report_build/` | `make latexpdf` there |

`docs/oz/latex/REVIEW.md` lists what in the report and manuscript no longer matches the code. Internal working
notes from the pyOZgui development are in `docs/oz/notes/`.

```{toctree}
:hidden:
:maxdepth: 2

Installation <installation>
SLD profile <sldprofile/index>
Size distribution / p(r) <inversion/index>
Ornstein–Zernike <oz/index>
studio <studio>
Preferences <preferences>
API <api/index>
```
