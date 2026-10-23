# pySASfit

Python tools for small-angle scattering (SANS/SAXS) from the SASfit project, in one installable package `pysasfit`.

| sub-package | content | GUI |
|---|---|---|
| `pysasfit.core` | shared infrastructure: data loading, instrumental resolution (Rician/Gaussian pinhole as in pyOZgui, slit, user matrix), background models (constant, power law, Ornstein–Zernike, user functions; linear or nonlinear) | – |
| `pysasfit.sldprofile` | Bayesian estimation of radial scattering-length-density profiles ρ(r) of spheres, cylinders and lamellae: direct (Hansen, J. Appl. Cryst. 49 (2016) 856) or Glatter's two-step route (IFT, then deconvolution) | `pysasfit-sldprofile` |
| `pysasfit.inversion` | Fredholm inversion: size distributions and p(r) (EM, signed EM, maxent, Tikhonov, SVD, ARLS, Bayesian-evidence IFT, splines) | `pysasfit-inversion` |
| `pysasfit.studio` | the SASfit plugin and libsasfit model functions (form factors, structure factors, size distributions, peaks) via ctypes, combined into models (polydispersity, monodisperse/decoupling/LMA structure factors, resolution) and fitted; from SASstudio, see `docs/studio.md` | `pysasfit-explorer` (model explorer and fit panel), `pysasfit-studio` (command line) |
| `pysasfit.oz` | Ornstein–Zernike solvers and closures, polydisperse/multicomponent structure factors, SAS models, analytic references (MSA, RMSA, Robertus, Gazzillo), fitting; from pyOZgui, GUI ported from Tk to Qt (see `docs/oz/README.md`) | `pysasfit-oz` |
| `pysasfit.gui` | one launcher window with every tool as a tab | `pysasfit-gui` |

## Installation

```bash
cd pySASfit
pip install -e ".[gui,dev]"     # library + PySide6/matplotlib GUIs + pytest
pip install -e ".[gui,oz]"      # + optional OZ dependencies (sundials4py, mpmath, bumps, mixscatter, openpyxl)
pip install -e ".[gui,fit,export]"  # + optional optimisers (nlopt, bumps, iminuit, cma, DFO-LS) and Excel export
pip install -e .                # library only (numpy, scipy)
```

Python ≥ 3.10, numpy ≥ 2.0, scipy ≥ 1.10.

`pysasfit.studio` also needs the compiled SASfit libraries, which are not kept in git. Copy them once:

```bash
pysasfit-studio install-binaries C:/Users/<you>/switchdrive/SASstudio   # or an unpacked SASfit release
```

## GUIs

```bash
pysasfit-gui [datafile]          # all tools as tabs; a data file is opened in the SLD-profile tab
pysasfit-sldprofile [datafile]   # single tools
pysasfit-inversion
pysasfit-oz
pysasfit-explorer [ff_sphere]   # studio: plot, adjust and fit SASfit models (docs/studio.md)
python run_gui.py [datafile]     # without installing (finds ./src itself)
```

**Settings → Preferences…** (launcher and every tool window) holds all numerical settings that are not on the
main panels -- α/λ ranges, iteration limits, tolerances, grids -- and saves them as your defaults for later starts.
See `docs/preferences.md` for the list, the factory values and the values that remain fixed.

## Documentation

```bash
pip install -e ".[docs]"
python -m sphinx -b html docs docs/_build/html     # open docs/_build/html/index.html
```

The studio manual (`docs/studio_manual/`, with screenshots from `python tools/studio_screenshots.py` and the
examples of `examples/studio/`) is part of it.

One HTML documentation for all tools (SLD profile, size distribution / p(r), Ornstein–Zernike, studio,
preferences, API reference generated from the docstrings). The OZ report and manuscript (LaTeX) and the inversion
PDF report have their own builds; see `docs/index.md`.

## Library use

```python
from pysasfit.core import load_data, Resolution, Background, Constant, PowerLaw
from pysasfit.sldprofile import DensityProfileFit

q, I, sig, dq = load_data("micelle.dat")                 # q, I, sigma [, dq]
fit = DensityProfileFit(q, I, sig, geometry="sphere", basis="rho",
                        background=Background([Constant(), PowerLaw(n=4.0, fit_n=True)]),
                        resolution=Resolution(dq))        # Rician kernel, dq = sigma
res = fit.fit(verbose=True)
print(res.summary())
```

```python
import numpy as np
from pysasfit.inversion.io_utils import load_sas_data
from pysasfit.inversion.kernel_registry import KERNEL_REGISTRY
from pysasfit.inversion.kernels import build_size_distribution_kernel
from pysasfit.inversion.solver_registry import SOLVER_REGISTRY

data = load_sas_data("tests/inversion/data/test.dat")
r = np.linspace(1.0, np.pi / data.q.min(), 150)
A = build_size_distribution_kernel(data.q, r, KERNEL_REGISTRY["sinc_4pi"].func, alpha=0.0)
result = SOLVER_REGISTRY["bayesian_evidence_hansen"].run(A, data.I, data.dI)
```

```python
from pysasfit import studio
print(studio.info("ff_sphere"))
I = studio.function("ff_sphere")(q, R=50, eta=1)
S = studio.function("sq_HardSphere")(q, RHS=50, fp=0.3)
```

## Tests

```bash
pytest                      # all tools (about 15 min)
pytest tests/sldprofile     # one tool
pytest tests/studio         # studio: needs PySide6; GUI tests run offscreen, those needing libsasfit skip without it
```

The inversion checks that were written as scripts with a `main()` still run on their own
(`python tests/inversion/test_em_basic.py`); pytest runs them through a `test_main` wrapper.

## Layout

```
pyproject.toml          package metadata (setup.py is only a stub)
run_gui.py              launcher without installation
src/pysasfit/
    core/               io.py, resolution.py, background.py
    sldprofile/         geometry, density, regularization, models, gui
    inversion/          kernels, solvers/, regularization, background, io_utils, ... ; gui/app.py
    studio/             headers.py (catalogue), library.py (loader), functions.py, model.py, integration.py,
                        groups.py + paths.py (menu tree, library search), cli.py, project.py, profiling.py,
                        fitting.py (engine, optimisers, bumps), export.py (csv/xlsx/figure),
                        explorer.py + param_row.py + fit_panel.py + contribution_panel.py (Qt GUI),
                        headers/ (93 plugin headers), templates/, sasfit_lib/ (installed binaries, not in git)
    oz/                 pyOZgui library modules (ozLib, solvers, closures, polydisperse SAS, fitting, wrappers,
                        librmsa.dll); oz/gui/ Qt GUI (main_window, oz_solver_tab, the polydisperse tabs, qtkit)
    gui/launcher.py
tests/                  sldprofile/, inversion/ (+ data/), studio/ (+ mock SASfit C sources), oz/, oz_gui/, gui/
extern/oz/              C sources of the OZ helper libraries (build_windows.bat, build_linux.sh)
examples/sldprofile/    demo_hansen2016.py and test data
examples/studio/        example_script.py, oz_example.py, data/ (SASstudio / FitBenchmarking data sets)
docs/                   one Sphinx project (conf.py, index.md): sldprofile/, inversion/, oz/, studio.md,
                        preferences.md, core_api.md
docs/oz/latex/          OZ report (build_report.sh) and manuscript (manuscript/build.sh), LaTeX; REVIEW.md lists
                        what no longer matches the code
docs/inversion/report_build/  PDF report of the inversion pages (Sphinx LaTeX builder)
```

The older folders of this repository (`io_tools/`, `dev/`, `tools/`, `notebooks/`, `notetabs/`, `pyozGUI/`, `data/`)
are not part of the installed package yet. `io_tools/SASformats.py` (BerSANS, HDF5 readers) is a candidate for
`pysasfit.core.io`; `pyozGUI/` is an older copy of pyOZgui, superseded by `pysasfit.oz`.

## Origin of the code

- `sldprofile`, `core`: from `SASfitGit/src/pySLDprofile` (package `pysldprofile`); `core` is its former
  `io_utils`, `resolution` and `background` modules.
- `oz`: from `SASfitGit/src/pyOZgui` (package `pyozgui`); library with package-relative imports, Tk GUI ported to Qt.
- `studio`: from `SASstudio` (github.com/ISISNeutronMuon/SASstudio), copied and reworked (see `docs/studio.md`).
- `inversion`: from `SASfitGit/src/pySASfitInversion` (package `sasfit_inversion` and `gui`), unchanged except for the
  import paths (`sasfit_inversion.X` → `pysasfit.inversion.X`, `gui.app` → `pysasfit.inversion.gui.app`). It still
  uses its own `load_sas_data` and background-subtraction module; moving these onto `pysasfit.core` is a later step.

## To do

- `pysasfit.studio`: Hankel transform / SESANS G(z) via `sasfit_hankel` (and port the sasfit-collab benchmark);
  parameter names of all built-in libsasfit functions; a studio tab in `pysasfit-gui`.
- Inversion IO/background on `pysasfit.core`; BerSANS/HDF5 readers from `io_tools/` into `pysasfit.core.io`.

## Licence

BSD 3-Clause, see `LICENSE`.
