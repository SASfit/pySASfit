# pysasfit.studio — SASfit plugin functions in Python

`pysasfit.studio` gives Python access to the compiled SASfit model functions:

- the **plugin** functions: 449 form factors, 138 structure factors, 42 size distributions and 75 peaks in 93 plugin
  libraries;
- the **built-in** functions of `libsasfit`: 136 functions, among them sphere, LogNorm, Schulz–Zimm, Percus–Yevick
  hard spheres and the Hayter–Penfold MSA.

It combines them into models (size distribution, structure factor, resolution, background) and fits them. The code
started as a copy of SASstudio (github.com/ISISNeutronMuon/SASstudio) and was reworked; see "Changes" below.

## Installation of the SASfit libraries

The Python code and the plugin headers are part of pysasfit. The compiled libraries are platform specific and
large, so they are not in git. Copy them once from a SASstudio checkout or an unpacked SASfit release:

```bash
pysasfit-studio install-binaries C:/Users/<you>/switchdrive/SASstudio      # -> src/pysasfit/studio/sasfit_lib
pysasfit-studio status
```

Alternatively, point `PYSASFIT_SASFIT_LIB` (or `SASFIT_LOCATION`, as used by sasfit-collab) to a folder containing
`libsasfit.<dll|so|dylib>` and `plugins/libsasfit_*`. On Linux and macOS this needs a SASfit build for that platform.

The catalogue (names, parameters, search) works without the libraries.

## Use

```python
import numpy as np
from pysasfit import studio

studio.search("sphere", "shell", type="ff")       # list of PluginInfo
print(studio.info("ff_sphere"))                     # parameters p[0], p[1], ... with descriptions
q = np.geomspace(1e-3, 0.5, 200)

sphere = studio.function("ff_sphere")
I = sphere(q, R=50, eta=1)                          # named, positional, or p3=1
F = sphere.amplitude(q, R=50, eta=1)                # the _f function
V = sphere.volume(50, R=50, dist=0)                 # the _v function

lognorm = studio.function("sd_LogNorm")             # size distribution N(x)
N = lognorm(np.linspace(1, 100, 50), N=1, s=0.2, p=1, mu=30)

hs = studio.function("sq_HardSphere")               # structure factor S(q)
S = hs(q, RHS=50, fp=0.3)
```

A SASfit error (`param->errStatus`) raises `studio.SASfitError` with SASfit's message. Use
`function(name, errors="warn")` or `errors="ignore"` to change this.

### Models and fitting

```python
from pysasfit.studio import ScatteringContribution, ModelFunction, Pipeline
from pysasfit.core import Resolution

sc = ScatteringContribution("ff_sphere", {"R": 1, "eta": 1},
                            size_distribution="sd_LogNorm", sd_params={"N": 1e-11, "s": 0.2, "p": 1, "mu": 70},
                            distributed="R",                       # parameter smeared by N(x)
                            structure_factor="sq_HardSphere", sq_params={"RHS": 70, "fp": 0.2},
                            approach="lma")                        # 'monodisperse' | 'decoupling' | 'lma'
model = ModelFunction([sc], background=0.1, resolution=Resolution(dq))   # Rician pinhole from pysasfit.core
pipe = Pipeline()
d = pipe.add_dataset("data.dat")
m = pipe.add_model(model, dataset=d)
res = pipe.fit(m, ["0.sd.N", "0.sd.mu", "0.sd.s", "0.sq.fp", "background"])
print(res.summary())
```

Parameter addresses are `"<contribution>.<ff|sd|sq>.<name or p<k>>"`, `"scale.<i>"` and `"background"`.

The structure-factor approaches follow the SASfit manual (ch. 4.1):

- **monodisperse:** S(q) ∫N P dx;
- **decoupling:** ∫N P dx + (∫N F dx)²/∫N dx · (S − 1);
- **local monodisperse (lma):** ∫N P S(q; R_V(x)) dx, where R_V is the volume-equivalent radius from the `_v` function.

The x integral uses Gauss–Legendre nodes on the support of N(x), or on `x_range` if you give one. With
`scale_errors=False`, the parameter errors are computed as SASfit computes them, without the √χ²_red factor.

`examples/studio/example_script.py` fits the bimodal-sphere test data and reproduces SASfit's parameters from
`data/SASfitTestData/SASfit_test.sas` to six digits.

### Python plugins

Python plugins override C plugins of the same name. There are two ways to provide one:

- **As a file:** `sasstudio_<name>.py`, written from `templates/python_plugin.py`, in a folder listed in
  `PYSASFIT_STUDIO_PLUGINS` or in `~/.pysasfit/studio_plugins`.
- **At run time:** `studio.register_python_function("ff_name", intensity, amplitude, volume, parameters=[...])`.

`pysasfit-studio new-plugin <name>` creates the plugin file and a test.

### Command line

```
pysasfit-studio list [--type ff|sq|sd|peak]      pysasfit-studio search hard sphere
pysasfit-studio info ff_sphere                    pysasfit-studio eval ff_sphere --q 0.01 0.1 --set R=50 eta=1
pysasfit-studio new-plugin NAME                   pysasfit-studio sasview-adapter ff_sphere
pysasfit-studio profiles OUTDIR                   pysasfit-studio install-binaries SOURCE
pysasfit-studio menu                              (the interactive menu of SASstudio_utilities_cli.py)
```

## How plugins are loaded

SASfit plugins are not linked against libsasfit. They reach the common SASfit functions (integration, special
functions, error reporting) through a function table that `do_init()` hands them. The loader therefore does what
`sasfit_core/sasfit_plugin_backend.c` does:

1. Load libsasfit and fetch the table with `sasfit_common_stubs_ptr()`.
2. Call `do_init(&exports, stubs, search)` for each plugin library. `search` is a Python callback that resolves the
   functions a plugin imports from other plugins.
3. Register the exported `sasfit_plugin_func_t` entries (function, `_f`, `_v`) by name.

A plugin whose imports are not loaded yet returns `SASFIT_PLUGIN_DEP_ERR`. The loader then loads the other plugin
libraries and initialises the plugin again.

The precision settings of libsasfit (`sasfit_eps`) are zero until set. The SASfit GUI defaults are applied on loading:

- `nriq` = 1e-10, `aniso` = `res` = 1e-5;
- `int_strategy` = P_CUBATURE, `sphavg_strategy` = Fibonacci (1014 points), `hankel_strategy` = Ooura DEO;
- `rootalg` = Brent, ...

Change them with `studio.get_library().set_precision(nriq=1e-8, int_strategy="GSL_QAGS")`.

## Changes with respect to SASstudio

- **Plugin initialisation:** plugins are initialised with `do_init()` and libsasfit's common-function table. Before,
  any plugin that uses a SASfit common function dereferenced a NULL pointer.
- **`sasfit_param`:** the ctypes copy now matches `sasfit_function.h`: `char errStr[256]`, `bool` = `int`,
  1104 bytes. Before, `errStr` was a `wchar_t*`, which shifted `more_p`, `moreparam`, `xarr` and `yarr`.
- **Precision:** libsasfit precision settings are initialised to the SASfit defaults.
- **Built-in functions:** the libsasfit functions (sphere, LogNorm, HardSphere, ...) are available, not only plugins.
- **Paths:** nothing depends on the working directory; `pefile`/`psutil` are replaced by `os.add_dll_directory`;
  the macOS extension is `dylib`.
- **Vectorised calls:** functions evaluate whole q arrays and accept named parameters; SASfit errors are reported.
- **Models:** size-distribution integration, the decoupling approximation and the LMA are implemented. Before,
  `run_integrator` returned 0 and the code used undefined names.
- **Fitting:** `Pipeline.fit` uses least squares with SASfit-compatible errors. Data are read with
  `pysasfit.core.load_data`, and the resolution comes from `pysasfit.core`.
- **Tools:**
  - the header parser tolerates malformed tables, such as the `</tr></tr>` in `ff_boucher_cyl`;
  - the templates produce working plugins and SasView adapters;
  - `generate_plugin_profiles.py` became `pysasfit-studio profiles`.
- **Tests:** they compile a mock libsasfit with two plugins, one of them importing from the other, so the loader is
  tested on Linux.

## To do

- **Hankel / SESANS:** wrap `sasfit_hankel` and `sasfit_integrate_ctm` with correctly typed strategy setters
  (`N_Ogata` is an `int`), compute G(z) for any form factor, and port the Hankel-strategy benchmark from
  sasfit-collab.
- **Parameter names:** take the names of the remaining built-in libsasfit functions from the SASfit Tcl GUI
  (`sasfit.vfs`). Today only `ff_sphere`, `sd_LogNorm` and `sq_HardSphere` have names; the others are addressed as
  `p0`, `p1`, ...
- **GUI:** add a studio tab to `pysasfit-gui`.
- **Testing:** test against a real Linux build of SASfit (`libsasfit.so`).
