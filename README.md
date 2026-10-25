# pySASfit

Python tools for small-angle scattering (SANS/SAXS) from the SASfit project, in one installable package `pysasfit`:
from the raw detector data (reduction with exact error propagation) to models, fits and inversions.

| sub-package | content | GUI |
|---|---|---|
| `pysasfit.sasred` | **pySASred**: SANS data reduction for SINQ SANS-I with exact first-order propagation of every uncertainty source (counts, monitor, dead time, attenuator, …): run table, beam centre, suggested and drawn masks, live radial average, cartesian / polar detector view, NXcanSAS output with covariance, systematic terms and error budget; nm or Å units; see `docs/sasred/` | `pysasfit-sasred` |
| `pysasfit.toolbox` | **pyTOOLbox** (formerly pySASbox), small helpful tools: local HDF5 / NeXus viewer in the spirit of myHDF5 (tree, search, line plots, heatmaps of N-d data with true pixel aspect, NeXus default plots, attributes) and the metadata lister for run ranges (formerly `tools/hdf_meta_lister.py`); see `docs/toolbox/` | `pysasfit-toolbox` |
| `pysasfit.core` | shared infrastructure: data loading, instrumental resolution (Rician/Gaussian pinhole as in pyOZgui, slit, user matrix), background models (constant, power law, Ornstein–Zernike, user functions; linear or nonlinear) | – |
| `pysasfit.sldprofile` | Bayesian estimation of radial scattering-length-density profiles ρ(r) of spheres, cylinders and lamellae: direct (Hansen, J. Appl. Cryst. 49 (2016) 856) or Glatter's two-step route (IFT, then deconvolution) | `pysasfit-sldprofile` |
| `pysasfit.inversion` | Fredholm inversion: size distributions and p(r) (EM, signed EM, maxent, Tikhonov, SVD, ARLS, Bayesian-evidence IFT, splines) | `pysasfit-inversion` |
| `pysasfit.studio` | the SASfit plugin and libsasfit model functions (form factors, structure factors, size distributions, peaks) via ctypes, combined into models (polydispersity, monodisperse/decoupling/LMA structure factors, resolution) and fitted; from SASstudio, see `docs/studio.md` | `pysasfit-explorer` (model explorer and fit panel), `pysasfit-studio` (command line) |
| `pysasfit.oz` | Ornstein–Zernike solvers and closures, polydisperse/multicomponent structure factors, SAS models, analytic references (MSA, RMSA, Robertus, Gazzillo), fitting; from pyOZgui, GUI ported from Tk to Qt (see `docs/oz/README.md`) | `pysasfit-oz` |
| `pysasfit.gui` | one launcher window with every tool as a tab (pySASred, pySASstudio, pySLDprofile, pyINVersion, pyOZsolver, pyTOOLbox); every tab can be detached into its own window and attached again | `pysasfit-gui` |

## Installation

```bash
cd pySASfit
pip install -e ".[gui,dev]"     # library + PySide6/matplotlib GUIs + pytest
pip install -e ".[gui,oz]"      # + optional OZ dependencies (sundials4py, mpmath, bumps, mixscatter, openpyxl)
pip install -e ".[gui,fit,export]"  # + optional optimisers (nlopt, bumps, iminuit, cma, DFO-LS) and Excel export
pip install -e ".[gui,sasred,toolbox]"  # + SANS reduction and HDF5 tools (h5py, pyyaml, openpyxl)
pip install -e .                # library only (numpy, scipy)
```

Python ≥ 3.10, numpy ≥ 2.0, scipy ≥ 1.10.

`pysasfit.studio` also needs the compiled SASfit libraries, which are not kept in git. Copy them once:

```bash
pysasfit-studio install-binaries C:/Users/<you>/switchdrive/SASstudio   # or an unpacked SASfit release
```

## GUIs

```bash
pysasfit-gui [datafile]          # all tools as tabs; a data file is opened in the pySLDprofile tab
pysasfit-sasred [raw folder]     # single tools: SANS reduction
pysasfit-toolbox [file.hdf ...]   # HDF5 viewer and metadata lister
pysasfit-sldprofile [datafile]
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

One HTML documentation for all tools (launcher, pySASred, studio, SLD profile, size distribution / p(r),
Ornstein–Zernike, pyTOOLbox, preferences, API reference generated from the docstrings). The pySASred and pyTOOLbox
screenshots come from `python tools/sasred_screenshots.py <SANS-I raw-data folder>`. The OZ report and manuscript (LaTeX) and the inversion
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
from pysasfit.sasred.io.sinq import read_sans_i
from pysasfit.sasred.core.beamcentre import masked_centre_of_gravity
from pysasfit.sasred.core.averaging import q_edges
from pysasfit.sasred.core.mask import suggest_mask
from pysasfit.sasred.reduce import reduce_run_1d
from pysasfit.sasred.io.nxcansas import write_nxcansas_1d

bc = masked_centre_of_gravity(read_sans_i("sans2026n082173.hdf").counts)   # direct beam, pixels from 0
raw = read_sans_i("sans2026n082130.hdf")
mask, report = suggest_mask(raw.counts, (bc.x, bc.y))      # border + beamstop shadow + hot pixels
r = reduce_run_1d(raw, (bc.x, bc.y), q_edges(0.006, 0.09, 60), mask=mask)   # q in 1/Å internally
print(r.q, r.I, r.idev())                                  # I in counts/monitor/sr, statistical Idev
write_nxcansas_1d("A_1_8m.h5", r)                          # Q in 1/nm, covariance and error budget included
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
pytest tests/sasred tests/toolbox   # tests on real SANS-I files run when PYSASRED_TEST_RAW points to the raw data
```

The inversion checks that were written as scripts with a `main()` still run on their own
(`python tests/inversion/test_em_basic.py`); pytest runs them through a `test_main` wrapper.

## Layout

```
pyproject.toml          package metadata (setup.py is only a stub)
run_gui.py              launcher without installation
src/pysasfit/
    core/               io.py, resolution.py, background.py, settings*.py, project*.py, docking.py (panels),
                        tabs.py (detachable tabs), colorscale.py, flowlayout.py
    sasred/             data/uarray.py (uncertainty engine), core/ (counting, dead time, normalisation, geometry,
                        beam centre, averaging, resolution, mask, polar), io/ (sinq reader, nxcansas, ascii),
                        instruments/sinq/sans_i.yaml, reduce.py, catalogue.py, units.py, gui.py, mask_editor.py
    toolbox/            hdf5.py (viewer helpers), viewer.py, meta_lister.py, gui.py
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
    resources/          tool icons and logos (copied from docs/pySASfit-logo-bundle by tools/sync_logos.py)
    sasbox/             old name of toolbox (forwards to it)
tests/                  sldprofile/, inversion/ (+ data/), studio/ (+ mock SASfit C sources), oz/, oz_gui/, gui/,
                        core/, sasred/, toolbox/
extern/oz/              C sources of the OZ helper libraries (build_windows.bat, build_linux.sh)
examples/sldprofile/    demo_hansen2016.py and test data
examples/studio/        01_evaluate_functions.py ... 05_open_explorer.py (included in the studio manual),
                        example_script.py, oz_example.py, data/ (SASstudio / FitBenchmarking data sets)
tools/                  release.py (tests, screenshots, HTML documentation, package), studio_screenshots.py,
                        sasred_screenshots.py, and older SANS helper scripts (not part of the package;
                        hdf_meta_lister.py is now pyTOOLbox's metadata lister)
docs/                   one Sphinx project (conf.py, index.md): launcher.md, sasred/, toolbox/, sldprofile/,
                        inversion/, oz/, studio_manual/, studio.md (reference), studio_api.md, installation.md,
                        preferences.md, core_api.md, _static/studio/, _static/sasred/, _static/toolbox/ (screenshots)
docs/oz/latex/          OZ report (build_report.sh) and manuscript (manuscript/build.sh), LaTeX; REVIEW.md lists
                        what no longer matches the code
docs/inversion/report_build/  PDF report of the inversion pages (Sphinx LaTeX builder)
```

The older folders of this repository (`io_tools/`, `dev/`, `notebooks/`, `notetabs/`, `pyozGUI/`, `data/`)
are not part of the installed package yet. `io_tools/SASformats.py` (BerSANS, HDF5 readers) is a candidate for
`pysasfit.core.io`; `pyozGUI/` is an older copy of pyOZgui, superseded by `pysasfit.oz`.

## Origin of the code

- `sldprofile`, `core`: from `SASfitGit/src/pySLDprofile` (package `pysldprofile`); `core` is its former
  `io_utils`, `resolution` and `background` modules.
- `oz`: from `SASfitGit/src/pyOZgui` (package `pyozgui`); library with package-relative imports, Tk GUI ported to Qt.
- `studio`: from `SASstudio` (github.com/ISISNeutronMuon/SASstudio), copied and reworked (see `docs/studio.md`).
- `sasred`: pySASred, written for PSI SINQ (phase 1 of its package definition); the plan is to bring in the
  MoDaCor 1.11 pipeline (process steps, runner, YAML configuration) on top of the same uncertainty engine.
- `toolbox` (pyTOOLbox, formerly pySASbox / `sasbox`): the HDF5 viewer is new; the metadata lister is
  `tools/hdf_meta_lister.py` ported from PyQt6 to PySide6. `pysasfit.sasbox`, `pysasfit-sasbox` and the extra
  `sasbox` still work.
- `inversion`: from `SASfitGit/src/pySASfitInversion` (package `sasfit_inversion` and `gui`), unchanged except for the
  import paths (`sasfit_inversion.X` → `pysasfit.inversion.X`, `gui.app` → `pysasfit.inversion.gui.app`). It still
  uses its own `load_sas_data` and background-subtraction module; moving these onto `pysasfit.core` is a later step.

## To do

- `pysasfit.sasred` phase 3: transmission, empty cell and Cd subtraction of the sample, detector efficiency in the
  GUI (compare with the SANS-I efficiency maps in `tools/*.mat`), thickness, absolute scale, collimation term of Qdev, pixel splitting,
  automatic grouping of runs, batch scripts, live reduction, SANS-LLB reader; confirm the dead times, the
  attenuator factors and whether `chi_position` is the detector tilt.

- `pysasfit.studio`: Hankel transform / SESANS G(z) via `sasfit_hankel` (and port the sasfit-collab benchmark);
  parameter names of all built-in libsasfit functions.
- Inversion IO/background on `pysasfit.core`; BerSANS/HDF5 readers from `io_tools/` into `pysasfit.core.io`.

## Licence

BSD 3-Clause, see `LICENSE`.
