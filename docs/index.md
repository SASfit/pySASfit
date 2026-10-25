---
html_theme.sidebar_secondary.remove: true
---

# pySASfit

```{image} pySASfit-logo-bundle/logos/pySASfit/svg/pySASfit-light.svg
:alt: pySASfit
:height: 90px
:class: only-light
```

```{image} pySASfit-logo-bundle/logos/pySASfit/svg/pySASfit-dark.svg
:alt: pySASfit
:height: 90px
:class: only-dark
```

**Python tools for small-angle scattering (SANS/SAXS) from the SASfit project**, in one installable package
`pysasfit`: from the raw detector data (reduction with exact error propagation) to models, fits and inversions.
Every tool can be used from a script and from a Qt GUI; `pysasfit-gui` opens all of them as tabs of one window
({doc}`launcher`).

```bash
pip install -e ".[gui,sasred,toolbox]"      # from a checkout of github.com/SASfit/pySASfit
pysasfit-gui
```

::::{grid} 1 2 2 2
:gutter: 3

:::{grid-item-card} pySASred: SANS reduction
:link: sasred/index
:link-type: doc

```{image} pySASfit-logo-bundle/menu-icons/pySASred/pySASred-menu.svg
:alt: pySASred
:height: 40px
:class: card-icon
```

SINQ SANS-I raw data to I(q) with exact first-order propagation of every uncertainty source: run table, beam
centre, suggested and drawn masks, live radial average, NXcanSAS output with covariance and error budget.
:::

:::{grid-item-card} pySASstudio: models and fits
:link: studio_manual/index
:link-type: doc

```{image} pySASfit-logo-bundle/menu-icons/pySASstudio/pySASstudio-menu.svg
:alt: pySASstudio
:height: 40px
:class: card-icon
```

SASfit's plugin and libsasfit model functions via ctypes, combined into mixtures, plotted and fitted in an
interactive explorer (`pysasfit-explorer`) with several optimisers and saved projects. Manual with screenshots
and Python examples, plus the reference.
:::

:::{grid-item-card} pySLDprofile: SLD profiles
:link: sldprofile/index
:link-type: doc

```{image} pySASfit-logo-bundle/menu-icons/pySLDprofile/pySLDprofile-menu.svg
:alt: pySLDprofile
:height: 40px
:class: card-icon
```

Radial scattering-length-density profiles ρ(r) of spheres, cylinders and lamellae: direct Bayesian estimation
(Hansen 2016) or Glatter's two-step route.
:::

:::{grid-item-card} pyINVersion: size distribution / p(r)
:link: inversion/index
:link-type: doc

```{image} pySASfit-logo-bundle/menu-icons/pyINVersion/pyINVersion-menu.svg
:alt: pyINVersion
:height: 40px
:class: card-icon
```

Fredholm inversion: size distributions and pair-distance distributions with EM, MaxEnt, Tikhonov, SVD, ARLS,
Bayesian-evidence IFT and splines.
:::

:::{grid-item-card} pyOZsolver: Ornstein–Zernike
:link: oz/index
:link-type: doc

```{image} pySASfit-logo-bundle/menu-icons/pyOZsolver/pyOZsolver-menu.svg
:alt: pyOZsolver
:height: 40px
:class: card-icon
```

OZ solvers and 24 closures, polydisperse and multicomponent structure factors, SAS models, analytic references,
fitting.
:::

:::{grid-item-card} pyTOOLbox: small helpful tools
:link: toolbox/index
:link-type: doc

```{image} pySASfit-logo-bundle/menu-icons/pyTOOLbox/pyTOOLbox-menu.svg
:alt: pyTOOLbox
:height: 40px
:class: card-icon
```

Formerly pySASbox. A local HDF5 / NeXus viewer in the spirit of myHDF5 (tree, search, line plots, heatmaps with true pixel aspect,
NeXus default plots) and the metadata lister for run ranges.
:::

:::{grid-item-card} The launcher
:link: launcher
:link-type: doc

```{image} pySASfit-logo-bundle/menu-icons/pySASfit/pySASfit-menu.svg
:alt: pySASfit
:height: 40px
:class: card-icon
```

All tools as tabs of one window; every tab can be detached into its own window and attached again.
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
Launcher <launcher>
pySASred <sasred/index>
pySASstudio <studio_manual/index>
pySLDprofile <sldprofile/index>
pyINVersion <inversion/index>
pyOZsolver <oz/index>
pyTOOLbox <toolbox/index>
Preferences <preferences>
API <api/index>
```
