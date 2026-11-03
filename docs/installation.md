# Installation

```bash
cd pySASfit
pip install -e ".[gui,dev]"     # library + PySide6/matplotlib GUIs + pytest
pip install -e ".[gui,oz]"      # + optional OZ dependencies (sundials4py, mpmath, bumps, mixscatter, openpyxl)
pip install -e ".[gui,fit,export]"  # + studio: optional optimisers (nlopt, bumps, iminuit, cma, DFO-LS), Excel export
pip install -e ".[gui,sasred]"  # + pySASred: SANS reduction (h5py, pyyaml)
pip install -e ".[gui,toolbox]"  # + pyTOOLbox: HDF5 viewer and metadata lister (h5py, openpyxl)
pip install -e .                # library only (numpy, scipy)
pip install -e ".[docs]"        # Sphinx, MyST and the theme, to build this documentation
```

Python ≥ 3.10, numpy ≥ 2.0, scipy ≥ 1.10.

**Windows with MSYS2:** use the **UCRT64** environment; MINGW64 no longer has PySide6. PyPI has no PySide6 or VTK
wheels for MSYS2's Python, so install Qt from pacman, then the rest with pip:

```bash
pacman -S mingw-w64-ucrt-x86_64-pyside6 mingw-w64-ucrt-x86_64-qt6-svg
python -m pip install -e ".[gui,sasred,toolbox]"   # with the same python that runs pySASfit
```

Without `qt6-svg` the tools stop with "DLL load failed while importing QtSvg". PyVista (the optional separate 3D
window of pySASmorph) needs VTK, which MSYS2's Python cannot get from PyPI; the 3D panel in the window works
without it.

`pysasfit.studio` also needs the compiled SASfit libraries, which are not kept in git. Copy them once:

```bash
pysasfit-studio install-binaries <SASstudio folder or unpacked SASfit release>
```

## Programs

| command | opens |
|---|---|
| `pysasfit-gui [datafile \| project.pysasfit]` | all tools as tabs of one window ({doc}`launcher`) |
| `pysasfit-sasred [raw-data folder]` | pySASred, SANS data reduction |
| `pysasfit-toolbox [file.hdf …]` | pyTOOLbox, HDF5 viewer and metadata lister |
| `pysasfit-sldprofile [datafile \| project.pysasfit]` | SLD profile |
| `pysasfit-inversion [datafile \| project.pysasfit]` | size distribution / p(r) |
| `pysasfit-oz` | Ornstein–Zernike tabs |
| `pysasfit-explorer [ff_sphere]` | studio model explorer with fit panel (also `pysasfit-studio explore`) |
| `pysasfit-studio` | studio command line (`list`, `search`, `info`, `eval`, `tree`, `status`, `new-plugin`, ...) |
| `python run_gui.py [datafile]` | the launcher without installing (finds `./src` itself) |

Without a file argument the tools reopen their last session (SLD profile and size distribution: data, panel
settings, last result and, for the SLD profile, the window layout; pySASred: folder, beam centre, mask, bins,
units and layout; pyTOOLbox: the open files; the launcher: which tools are detached); **Settings → Forget last
session** starts empty.
**File → Save project… / Open project…** keep a piece of work in a versioned `*.pysasfit` file (see the GUI pages
of the two tools and `pysasfit.core.project` in {doc}`core_api`).

Numerical settings that are not on the main panels are under **Settings → Preferences…**; see
{doc}`preferences`.

## Building this documentation

```bash
pip install -e ".[docs]"
python -m sphinx -b html docs docs/_build/html      # or: cd docs; make html  (make.bat html on Windows)
```

Open `docs/_build/html/index.html`. The API pages are generated from the docstrings at build time; optional
dependencies (PySide6, SUNDIALS, bumps, …) are mocked, so they need not be installed.
