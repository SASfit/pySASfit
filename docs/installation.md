# Installation

```bash
cd pySASfit
pip install -e ".[gui,dev]"     # library + PySide6/matplotlib GUIs + pytest
pip install -e ".[gui,oz]"      # + optional OZ dependencies (sundials4py, mpmath, bumps, mixscatter, openpyxl)
pip install -e .                # library only (numpy, scipy)
pip install -e ".[docs]"        # Sphinx, MyST and the theme, to build this documentation
```

Python ≥ 3.10, numpy ≥ 2.0, scipy ≥ 1.10.

`pysasfit.studio` also needs the compiled SASfit libraries, which are not kept in git. Copy them once:

```bash
pysasfit-studio install-binaries <SASstudio folder or unpacked SASfit release>
```

## Programs

| command | opens |
|---|---|
| `pysasfit-gui [datafile]` | all tools as tabs of one window |
| `pysasfit-sldprofile [datafile]` | SLD profile |
| `pysasfit-inversion` | size distribution / p(r) |
| `pysasfit-oz` | Ornstein–Zernike tabs |
| `pysasfit-studio` | studio command line |
| `python run_gui.py [datafile]` | the launcher without installing (finds `./src` itself) |

Numerical settings that are not on the main panels are under **Settings → Preferences…**; see
{doc}`preferences`.

## Building this documentation

```bash
pip install -e ".[docs]"
python -m sphinx -b html docs docs/_build/html      # or: cd docs; make html  (make.bat html on Windows)
```

Open `docs/_build/html/index.html`. The API pages are generated from the docstrings at build time; optional
dependencies (PySide6, SUNDIALS, bumps, …) are mocked, so they need not be installed.
