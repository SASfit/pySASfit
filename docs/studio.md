# studio reference

```{note}
This page is the complete reference. A guided tour with screenshots and examples is the {doc}`studio manual <studio_manual/index>`.
```

`pysasfit.studio` gives Python access to the compiled SASfit model functions:

- the **plugin** functions: 449 form factors, 138 structure factors, 42 size distributions and 75 peaks in 93 plugin
  libraries;
- the **built-in** functions of `libsasfit`: 136 functions, among them sphere, LogNorm, Schulz–Zimm, Percus–Yevick
  hard spheres and the Hayter–Penfold MSA.

It combines them into models (size distribution, structure factor, resolution, background) and fits them, from a
script or in an interactive GUI, the **model explorer** (`pysasfit-explorer`). The code started as a copy of
SASstudio (github.com/ISISNeutronMuon/SASstudio) and was reworked; see "Changes" below.

| you want to                                        | use                                                                                       |
| -------------------------------------------------- | ----------------------------------------------------------------------------------------- |
| plot and fit models by hand, with data             | [model explorer](#model-explorer)                                                         |
| fit from a script                                  | [`Pipeline`](#models-and-fitting) or the [fitting engine](#fitting-engine-and-optimisers) |
| compare optimisers, constraints, several data sets | [fitting engine](#fitting-engine-and-optimisers)                                          |
| keep a session and continue later                  | [projects](#projects)                                                                     |

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

## Model explorer

```
pip install -e ".[gui]"
pysasfit-explorer                                              # a sphere
pysasfit-explorer ff_sphere --sd sd_LogNorm --sq sq_hardsphere
pysasfit-explorer ff_sphere --add ff_cylinder,sd_LogNorm       # second contribution of a mixture
pysasfit-explorer --project my.sasstudio                       # a saved session
```

The explorer plots I(q) of a model and lets you change its parameters and fit it to data.

**Model.** The model is a mixture, I(q) = Σ scale<sub>i</sub> · I<sub>i</sub>(q) + background. Every contribution
is a tab with a form factor, an optional size distribution and an optional structure factor, picked from cascading
menus that follow SASfit's groups (the `.def` files of the installation). *Add*, *Copy* and *Remove* above the tabs
change the number of contributions.

- With a size distribution, the *distr.* radio button selects the form-factor parameter that is integrated over
  the distribution. That parameter is disabled and cannot be fitted. The scale is fixed to 1 and hidden, because the
  amplitude N of the distribution is the scale. Without a size distribution, the contribution is monodisperse and
  only multiplied by its scale.
- With a size distribution and a structure factor, the approach buttons select *monodisperse*, *decoupling* or
  *lma* (see above).
- Parameters named `dummy` are not used by the function; their fields are disabled.
- *show* chooses P·S, P or S. "show contributions" draws every scale<sub>i</sub> · I<sub>i</sub> as a dotted curve.

**Parameters.** Values are typed into the fields. *View → Sliders* adds a slider to every field. The check box
next to a parameter selects it for fitting; when it is ticked, two fields below it take the lower and upper bound.
The selection and the bounds belong to the row, so they are reset when the function of that column changes.

**Integration.** *Settings → Integration…* sets how the size-distribution integral is computed. It either uses
SASfit's own integrators (the "sasfit" method) or scipy / Gauss–Legendre. It also sets the Hankel strategies of the
library and the search range with an x<sup>power</sup> tail. *Settings → SASfit installation…* selects the SASfit
folder. Both settings can be remembered between sessions.

**Data and fit.** The *Fit* panel (*View → Fit*) loads a data file (`pysasfit.core.load_data`; q × factor for
unit conversion), takes an optional fit q range and shows the parameters ticked for fitting. The residuals
Δ/σ appear below the plot. *Fit* runs the optimiser in a thread (*Stop* cancels it), *Undo fit* restores the values
before it. The *Constraints* tab takes inequalities and equalities (`c1_ff_R + c1_ff_L < 100`, `c1_scale + c2_scale == 1`;
names are the parameter labels with `.` replaced by `_`, so `ff_R` with a single contribution; the Fit panel's
tooltip lists them), the *Priors* tab priors (`label form args`: `normal`, `lognormal`, `halfnormal`, `loguniform`, see
"Priors for positive parameters and ranges") and the *Result* tab shows values, errors, χ²/dof, the constraints and the
correlation matrix. Optimiser, algorithm, and for bumps a line of options (see "bumps: optimisers and options")
are chosen above the parameter table; *Search global minimum* algorithms are described below. *Copy Python*
copies code that reproduces the model. **Export result…** saves xlsx/csv/pdf/png/svg and **Copy summary image**
puts the summary figure on the clipboard (see "Saving a fit result").

**Background.** Besides the constant `background` the rows `bg.A` and `bg.alpha` describe a power law A·q<sup>−α</sup>
(see "Power-law background").

**Windows.** Model, Fit and Size distributions are docks; each can be detached into its own window from the View
menu (see "Windows, background curve, size distributions"). The Size-distributions dock plots N(x)·x<sup>p</sup>; its
exponent p is also used by the export.

There is no "fit as priors" button on purpose (see "Priors").

The explorer fits one data set. The engine below can fit several at once.

## Fitting engine and optimisers

`pysasfit.studio.fitting.FitProblem` is the engine behind the Fit panel. It can also be used on its own:

**Several data sets.** `add_dataset(data, model)` per data set; models may share contribution objects (the values of a
data set are set right before it is evaluated). `ModelFunction(contributions, active=[True, False, …])` evaluates only
the contributions of that data set while keeping the addresses `<i>.ff.R`, `scale.<i>` the same in all models.
`add_common` / `add_local` make one parameter for all data sets or one per data set. **Global parameters**:
`add_global("P1", value, lower, upper)` and `link("P1", "1.ff.R", datasets=[1], factor=2.0)` (model value = factor ×
P1), or any expression of global parameters with `link_expression("0.ff.R", "P1 + P2")` (also `atan(P1)`,
`sqrt(P1**2 + P2**2)` …; the result's `derived` lists their values); a model parameter can be driven by one parameter
or link only. Global parameters are never eliminated by
variable projection; constraints and priors use them by name.

```python
from pysasfit.studio import fitting

prob = fitting.FitProblem()
prob.add_dataset("a.dat", model_a, name="a")                # a file or (q, I, sigma); repeat for more data sets
prob.add_common("0.ff.R", lower=1, upper=200)              # one value for all data sets
prob.add_local("background", lower=0)                      # one value per data set
prob.add_local("scale.0", lower=0)
prob.add_constraint("ff_R < 100")                          # inequality, see the Constraints tab
res = prob.fit("scipy")                                    # or "nlopt", "cma", ...
print(res.text())
```

Parameters can be common to all data sets or local to each, fixed or bounded. Constraints and priors are
supported (also priors of other forms, see below). Errors come from the numerical Jacobian at the minimum and are scaled by √χ²<sub>red</sub> unless
`scale_errors=False`. The result (`FitSummary`) holds values, errors, covariance, χ², the fitted curves and the state
of the constraints.

**Optimisers.** A backend is selected by name, then an algorithm. The Fit panel lists the backends whose package
is installed.

| backend        | package   | algorithms                                                                   | notes                                                                                 |
| -------------- | --------- | ---------------------------------------------------------------------------- | ------------------------------------------------------------------------------------- |
| `scipy`        | scipy     | `trf`, `dogbox`, `lm`                                                        | least squares; the default. Constraints by penalty.                                   |
| `scipy-global` | scipy     | `differential_evolution`, `multistart`, `dual_annealing`, `shgo`, `basinhopping` | global; finite bounds needed (basinhopping excepted).                             |
| `nlopt`        | `nlopt`   | local and global NLopt algorithms (`LN_BOBYQA`, `LN_SBPLX`, `LD_SLSQP`, ...) | derivative-free and gradient-based; handles constraints natively.                     |
| `iminuit`      | `iminuit` | `migrad`, `simplex`, `migrad+minos`                                          | MINOS gives asymmetric errors, printed as `(−a / +b)`.                                |
| `bumps`        | `bumps`   | `lm`, `amoeba`, `de`, `dream`, `newton`, `pt`                                | DREAM returns errors from the posterior samples. `de`, `dream`, `pt` need finite bounds; options: see "bumps: optimisers and options". |
| `cma`          | `cma`     | CMA-ES                                                                       | derivative-free, robust against noise.                                                |
| `dfols`        | `DFO-LS`  | default, `noisy`                                                             | derivative-free least squares with bounds; see "Faster fits".                         |

Optional packages: `pip install nlopt bumps iminuit cma DFO-LS` (or `pip install "pysasfit[fit]"`).

Options common to all backends: `maxeval` (maximum number of model evaluations) and `polish=True`. With `polish`,
the best point is refined by a least-squares fit, which gives the exact minimum and the errors; the Fit panel's
"refine" check box sets it for all optimisers except scipy least_squares. Another optimiser can be added with
`fitting.register_backend(name, runner)`.

**Noise.** The SASfit integrals are adaptive, so the model has a relative noise of about 1e-6 to 1e-5. Numerical
derivatives therefore use large steps (scipy: `diff_step=1e-4`; NLopt: 1e-3), and the iminuit errors are taken from
the least-squares Jacobian, because MINUIT's HESSE is too sensitive to the noise. Gradient-free optimisers
(CMA-ES, `LN_BOBYQA`, differential evolution) are less affected. `LN_COBYLA`, `LN_PRAXIS` and `LN_SBPLX` sometimes
stall when started far from the minimum; use "refine".

### Choosing an optimiser

The optimisers differ in what they need (a good start, finite bounds), what they cost (number of model
evaluations) and what they return (a point, or also an error estimate). A model evaluation with a size
distribution and a structure factor takes some 10 ms to 1 s, so the number of evaluations decides how long a fit
takes. The numbers below come from a small test (4 parameters, 60 points, 1e-5 integration noise,
`tests/studio/test_fitting.py`) and show the order of magnitude only; they are not a benchmark of real SASfit models.

| optimiser                                    | evaluations (4 parameters) | start              | bounds           | errors                   | use it for                                                 |
| -------------------------------------------- | -------------------------- | ------------------ | ---------------- | ------------------------ | ---------------------------------------------------------- |
| scipy `trf`                                  | 30 to 100                  | must be reasonable | yes              | covariance               | the default: fast local fit with good errors               |
| scipy `dogbox`                               | 30 to 100                  | must be reasonable | yes              | covariance               | many parameters at their bounds                            |
| scipy `lm`                                   | 30 to 100                  | must be reasonable | ignored          | covariance               | unbounded problems, smooth model                           |
| `differential_evolution`                     | 1 000 to 3 000             | any                | finite, required | after refine             | finding the right region when the start is poor            |
| `dual_annealing`                             | 2 000 and more             | any                | finite, required | after refine             | rugged landscapes; needs many evaluations to converge      |
| `shgo`                                       | 1 000 to 4 000             | any                | finite, required | after refine             | few parameters (up to about 5), smooth model               |
| `basinhopping`                               | 10 000 and more            | any                | optional         | after refine             | a few separate local minima; most expensive                |
| NLopt `LN_BOBYQA`                            | 100 to 400                 | reasonable         | yes              | after refine             | smooth model, no derivatives, noise tolerated              |
| NLopt `LN_SBPLX`, `LN_NELDERMEAD`            | 200 to 1 000               | reasonable         | yes              | after refine             | noisy model                                                |
| NLopt `LN_COBYLA`                            | 100 to 500                 | reasonable         | yes              | after refine             | native inequality constraints                              |
| NLopt `LD_SLSQP`                             | 50 to 200                  | reasonable         | yes              | after refine             | equality and inequality constraints, smooth model          |
| NLopt `GN_*` (`GN_CRS2_LM`, `GN_ISRES`, ...) | 2 000 and more             | any                | finite, required | after refine             | global search (`GN_ISRES` also with constraints)           |
| iminuit `migrad`                             | 200 to 1 500               | reasonable         | yes              | covariance               | strongly correlated parameters                             |
| iminuit `simplex`                            | 300 to 1 500               | poor is acceptable | yes              | covariance               | rough first step before `migrad`                           |
| iminuit `migrad+minos`                       | 1 000 to 3 000             | reasonable         | yes              | **asymmetric** (profile) | skewed or non-Gaussian errors                              |
| bumps `lm`                                   | 50 to 150                  | reasonable         | yes              | covariance               | like scipy `trf`                                           |
| bumps `amoeba`                               | 200 to 600                 | reasonable         | yes              | covariance               | derivative-free local search                               |
| bumps `de`                                   | 5 000 to 20 000            | any                | finite, required | covariance               | global search; slower than scipy's                         |
| bumps `dream`                                | 4 000 and more             | any                | finite, required | **posterior**            | full uncertainty, correlations, several minima             |
| CMA-ES                                       | 1 000 to 5 000             | any                | optional         | after refine             | noisy or multimodal model, up to some dozens of parameters |

"After refine" means: the optimiser returns a point only; with the "refine" box ticked (`polish=True`) a
least-squares fit starts at that point and supplies the errors.

**scipy least squares (`trf`, `dogbox`, `lm`).** They use the residuals Δ/σ directly (a Jacobian by finite
differences), so they need far fewer evaluations than any other method and give the covariance matrix. They are
local: from a poor start they stop in the nearest minimum, which for size distributions often means a wrong width or
mean. `trf` is the all-rounder; `dogbox` is better when several parameters sit at a bound; `lm` ignores bounds and
only suits well-posed, unbounded problems. Constraints are enforced by a penalty, so they may be violated by a
small amount (the Result tab says whether they hold).

**scipy global (`differential_evolution`, `dual_annealing`, `shgo`, `basinhopping`).** They need no good start.
The price is evaluations, and all but `basinhopping` need a lower and an upper bound for every fitted parameter;
without them the fit stops with a message. `differential_evolution` is the most dependable: it works with noisy
models and many parameters. `dual_annealing` converged badly within a small evaluation budget in the test, so give it
more evaluations. `shgo` is efficient for few, smooth parameters and does not scale to many. `basinhopping` starts
from your values and hops, so it is a local method with an escape route, not a global search. Combine any of them
with "refine".

**NLopt.** The package offers dozens of algorithms; the Fit panel lists those of your installation. `LN_` are
derivative-free local, `LD_` gradient-based local (the gradient comes from central differences with a step that is
large compared with the integration noise), `GN_` global. `LN_BOBYQA` is the best derivative-free choice for a
smooth model. `LN_SBPLX` and `LN_NELDERMEAD` tolerate noise. `LN_COBYLA` and `LD_SLSQP` handle constraints natively
(`LD_SLSQP` also equalities, for example fractions that add up to 1), without a penalty. In the test,
`LN_COBYLA`, `LN_PRAXIS` and `LN_SBPLX` sometimes stopped at chi²/dof of 2 to 3.5 when started far from the
minimum, so use "refine" with them.

**iminuit (MINUIT).** `migrad` is a variable-metric method that copes well with strongly correlated parameters,
such as the mean and the width of a size distribution or R and the volume fraction. `simplex` needs no gradient and
is a robust first step for a poor start (the panel runs `migrad` after it). `migrad+minos` also computes asymmetric
errors by profiling χ²: use it when the error of a parameter is clearly not symmetric (a width close to zero, a
parameter at a bound). The errors in the table stay those of the least-squares Jacobian, because MINUIT's own
HESSE estimate reacts to the noise of adaptive integrals. With a noisy model MIGRAD often ends with "stopped, EDM ..."
instead of "converged"; the point is still usually good (check χ²/dof and the Result tab). MINOS needs a valid
minimum and the panel restarts MIGRAD up to three times for it; if it still fails, the message says so.

**bumps.** `lm` and `amoeba` are local methods like the scipy and NLopt ones; `de` is a global search. `dream` is
different in kind: it samples the posterior distribution of the parameters (MCMC). The Result tab shows the best point
found, the standard deviation of the samples and the 68 % interval for each parameter. Use it when you want the
whole uncertainty, the correlations between parameters, or when you suspect several solutions. It needs many
evaluations (thousands per parameter) and finite bounds, and its intervals are only reliable if the chain has
converged: check that a longer run (*max evaluations*) gives the same intervals. For routine fitting it is too slow.

**CMA-ES (`pip install cma`).** An evolution strategy that learns the shape of the χ² valley. It needs no
gradient, so integration noise hardly matters, and it finds the right basin from a poor start. It uses more
evaluations than a local method but scales well to a few dozen parameters, which makes it a good global method for
mixtures. With "refine" the result is the exact least-squares minimum with errors.

**Recipes**

1. *Normal fit, good start:* scipy `trf`.
2. *Poor start, or the result depends on the start:* CMA-ES or `differential_evolution` with bounds and "refine",
   then check with `trf` from the result. For a mixture, give every size and fraction a sensible range.
3. *Constraints (sum of fractions, one radius smaller than another):* `LD_SLSQP` or `LN_COBYLA` with "refine", or
   any optimiser with the constraint as a penalty and a check of the Result tab.
4. *Errors that are not symmetric:* iminuit `migrad+minos`; for correlations and multiple solutions, bumps `dream`.
5. *Noisy model (coarse integration precision):* CMA-ES, `LN_SBPLX` or `differential_evolution`; tighten the
   integration settings if the gradient-based methods stall.
6. *Slow model:* keep the number of free parameters small, use a fit q range, start with `trf`, and use a global
   method only when needed. *Max evaluations* limits the work of every optimiser (for `differential_evolution` it sets
   the number of generations; the evaluations are rounded to whole generations).

The Result tab ends with the correlation matrix of the free parameters. Pairs with |r| >= 0.9 are listed with the
usual remedies: fix one of them at an independently known value, give one or both a Gaussian prior from independent
knowledge (not from the same data, which would count it twice), or do nothing if only their combination matters.
In Python: `result.correlation()`, `result.free_labels`, `result.correlation_text(threshold)`.

Degenerate parameters limit every optimiser: the scale and the amplitude N of a size distribution, or the volume
fraction and the scale, cannot be fitted together. Fix one of them, or give a Gaussian prior (Priors tab). A very
large error on a parameter, or a parameter stuck at a bound, points to this and not to the optimiser.

So far the choice rests on the test model. Benchmark the optimisers on your own SASfit problems, for example with
FitBenchmarking, before you rely on one for a long series of fits.

## Faster fits: DFO-LS, variable projection, cache, coarse to fine, fixed grid

Five options for slow models, all off by default and all giving the same result as the plain fit:

- **DFO-LS** (optimiser `DFO-LS`, `pip install DFO-LS`): derivative-free Gauss-Newton on the residuals with bounds.
  It needs far fewer evaluations than a finite-difference Jacobian when the model is slow. The variant `noisy`
  is more robust against the noise of adaptive integrals. Constraints are handled as penalty rows, like for scipy.

- **Solve linear parameters (variable projection)**: the intensity is linear in the scale factors, the background and
  the amplitude `N` of a size distribution. These are not searched by the optimiser. For every trial of the other
  parameters they are obtained exactly by linear least squares (their bounds are respected). The optimiser then
  has fewer dimensions and no strong correlations between scale and shape. Parameters that appear in a constraint
  or a prior stay with the optimiser. Errors and chi² are those of the full problem. Scale and `N` of the same
  contribution are redundant; `N` is the one that is solved.

- **Cache form factors**: during the fit the intensity of every contribution is remembered, and so is the form
  factor on the integration nodes of a size distribution. A change of the distribution parameters, of a scale or of
  another contribution then does not recompute the form factor. To make the nodes identical for nearby
  parameters, the automatically found integration range is rounded outwards to a 5 % geometric grid while
  fitting (the result differs from an unrounded range by the integration accuracy only). The Result tab reports how many
  evaluations were reused. Without cache the contributions are unchanged.

- **Coarse to fine**: the first fit uses a cheap size-distribution integral (`n_x` divided by 4, `rtol` at least
  1e-3); the second fit starts from that result with the real integration settings. A global optimiser (scipy
  global, CMA-ES, bumps `de`/`dream`) does its search in stage 1 and is followed by a least-squares `trf` in stage 2.
  The evaluations of both stages are added up. The SASfit-internal precision settings are not changed, and a
  contribution without size distribution is not affected. The gain is large only when the size-distribution integral
  dominates the time; check that the coarse stage lands near the minimum (the Result tab gives chi²/dof of stage 1).

- **Fixed size grid**: the size-distribution integral is evaluated on a grid that is equidistant in ln x (trapezoid
  rule) instead of Gauss-Legendre nodes that move with the integration range. P(q, x) is computed once on the grid
  (implies the cache); a trial of the distribution parameters (mean, width, shape, N) then only changes the weights
  N(x), so it costs a matrix-vector product and no form-factor call at all. The grid has about `n_x * 2` nodes inside
  the support of the distribution and 1.6 times the support on each side. It is rebuilt when the distribution leaves
  the grid or becomes much narrower than the grid spacing. Works for the monodisperse and the decoupling approach and
  without structure factor; the local monodisperse approximation (S depends on x) is integrated normally. A
  form-factor parameter other than the distributed one (length, shell thickness, ...) invalidates the grid values.
  The result differs from Gauss-Legendre by the integration accuracy (checked to 2e-4 relative in the tests); with a
  very narrow distribution raise `n_x`.

Two tools show what pays: **Profile model** (Fit panel) splits one model evaluation into form factor, size
distribution, structure factor and overhead and estimates the time of a fit; **Check nodes** compares the intensity
for increasing `n_x` with a reference of at least 192 nodes per panel and proposes the smallest `n_x` that agrees to
1e-4 (the default is 96, which is often more than needed; the node count is the number of form-factor calls per
model evaluation divided by 4 panels). In Python: `profiling.profile_model`, `integration.suggest_n_x`.

In Python: `prob.fit("dfols", vp=True, cache=True, coarse=True, grid=True)` (`coarse_factor` sets the reduction of `n_x`); `snap` sets the grid step (default 0.05). The best speed-up
comes from variable projection together with the cache: a fit of a size distribution with a scale and a background
then needs the form factor only when the form-factor parameters change.

## Projects

*File → Save project* (Ctrl+S) writes the session to a `.sasstudio` file; *Open project…* (Ctrl+O) or
`pysasfit-explorer --project FILE` restores it. A project is a JSON file with:

- the contributions (functions, approach, distributed parameter) and all parameter values, scales and the background,
  with their fit check boxes and bounds;
- the data (embedded in the file);
- the plot settings (q range, number of points, log axes, display mode);
- the fit settings: optimiser, algorithm, max evaluations, "refine", the speed-up options, fit
  q range, constraints, priors and the last result;
- the integration settings of the size-distribution integral.

The SASfit installation is not stored. A function that is missing in the installation used for loading is
reported in the status line.

The same functions are available from Python: `pysasfit.studio.project.save(explorer, path)` and
`project.load(explorer, path)`.

## Python plugins

Python plugins override C plugins of the same name. There are two ways to provide one:

- **As a file:** `sasstudio_<name>.py`, written from `templates/python_plugin.py`, in a folder listed in
  `PYSASFIT_STUDIO_PLUGINS` or in `~/.pysasfit/studio_plugins`.
- **At run time:** `studio.register_python_function("ff_name", intensity, amplitude, volume, parameters=[...])`.

`pysasfit-studio new-plugin <name>` creates the plugin file and a test.

## Command line

```
pysasfit-studio list [--type ff|sq|sd|peak]      pysasfit-studio search hard sphere
pysasfit-studio info ff_sphere                    pysasfit-studio eval ff_sphere --q 0.01 0.1 --set R=50 eta=1
pysasfit-studio tree [ff|sq|sd]                   (menu tree from the .def group files)
pysasfit-studio status                            (header folders, libraries, plugin and .def folders)
pysasfit-studio explore [ff_sphere]               (same as pysasfit-explorer)
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

## Searching for the global minimum

Fits of SAS models have many local minima (size, width and oscillation phase interact). Choose the algorithm in the Fit panel.

| Algorithm                                                          | Kind           | Notes                                                                       |
| ------------------------------------------------------------------ | -------------- | --------------------------------------------------------------------------- |
| `multistart`                                                       | global         | Latin-hypercube starts, each refined locally. First start = current values. |
| `differential_evolution`, `dual_annealing`, `shgo`, `basinhopping` | global (scipy) | Work in parameters rescaled to the bounds.                                  |
| `cma`                                                              | global         | CMA-ES, rescaled to bounds.                                                 |
| NLopt global (`GN_*`)                                              | global         | Optional.                                                                   |

**Rescaling.** `_Context.rescale_to_bounds()` maps every free parameter with finite bounds to `(x-lo)/(hi-lo)`. Global optimisers therefore see a unit box and parameters of very different magnitude are treated alike.

**Multistart.** `k` starting points are drawn from a Latin hypercube (`scipy.stats.qmc`). Parameters with a wide positive range (`lo > 0`, `hi > 100*lo`) are sampled log-uniformly. Per-start budget: `max(maxeval // k, 50)` evaluations. Results are grouped: minima that agree within 1e-3 (relative) count as the same minimum, so the report shows how often the best one was found. A best minimum found only once is a warning sign.

Needs finite bounds on all free parameters (the Fit panel tells you which one is missing).

## Priors for positive parameters and ranges

Bounds are a *uniform prior* between lower and upper. Additional priors, entered one per line in the Priors box as `label form args`:

| Form        | Syntax (example)          | Penalty added to chi²                         |
| ----------- | ------------------------- | --------------------------------------------- |
| Gaussian    | `ff.R normal 50 5`        | `((x-mean)/sigma)²`                           |
| log-normal  | `ff.R lognormal 50 1.3`   | `(ln(x/median)/ln(factor))²`, large if `x<=0` |
| half-normal | `ff.sigma halfnormal 0.2` | `(x/sigma)²` (x >= 0)                         |
| log-uniform | `ff.R loguniform`         | `2 ln x` (needs lower bound > 0)              |

Gaussian, log-normal and half-normal are appended as extra residual rows, so every least-squares algorithm handles them. Log-uniform is a scalar term and is refused by `scipy` (least_squares) and `dfols`, and by `multistart`. It works with NLopt, iminuit, bumps, CMA-ES and scipy-global. A parameter with a prior is kept nonlinear in variable projection. After a log-uniform fit the NLopt polish step is skipped; the coarse-to-fine mode keeps the global optimiser for stage 2.

**Which prior when?** A scale parameter (radius, length) that may span decades: log-uniform or log-normal. A relative width (`sigma`, polydispersity) that is probably small: half-normal. A known value with uncertainty: Gaussian.

**Prior from an earlier fit.** Taking the sigmas of a fit as priors is sensible only for a *new, independent* data set (sequential update). For the same data set it counts the information twice and makes the error bars too small. For that reason there is no "fit as prior" button; if you add one later, warn when the data-set name is identical.

**Bayesian view.** posterior = likelihood × prior. The best fit with priors is the MAP estimate. The bumps backend (DREAM) samples the posterior, including the priors above.

## Power-law background

Background model: `background + A * q^(-alpha)`, alpha starts at 4 (Porod). Rows in the explorer: `background`, `bg.A` (bounds 0…), `bg.alpha` (bounds 0…8). Addresses in fit tables, priors and projects: `background`, `bg.A`, `bg.alpha`.

`ModelFunction(…, bg_A=0.0, bg_alpha=4.0)`; `power_term(q)` returns `bg_A * q**-bg_alpha`.

**Variable projection.** I(q) = background·1 + A·q^-alpha + s·F(q; θ). For fixed θ and alpha the linear parameters `background`, `A`, scale and N are solved together by bounded linear least squares (`bg.A` column: `q^-alpha`). `alpha` stays nonlinear. A parameter with a prior or constraint stays nonlinear, too.

In the plot the background (constant + power law) is drawn as a dashed grey line; it is hidden when the background is zero and in the S(q) view.

## Windows, background curve, size distributions

- **Docks:** Model (left), Fit (bottom/right), Size distributions (right). The plot is the central widget.
- **Detach:** View menu → "detach '<title>' as a separate window". A detached dock has no owner window and does *not* stay on top (the usual float button does). Un-tick to dock again at its old area. Closing the main window closes detached docks.
- **Size distributions:** N(x)·x^p with free exponent p (spin box −6…12, default 3; buttons p=0/3/6, e.g. volume weight p=3, intensity weight about p=6). Optional log x-axis. Range from the distribution's `support()`, extended to `[max(lo/2, hi·1e-4), hi·1.5]`.

## bumps: optimisers and options

Backend `bumps` offers `lm`, `amoeba`, `de`, `dream`, `newton` (BFGS) and `pt` (parallel tempering, experimental; the bumps manual says it "does not yet perform well"). `de`, `dream` and `pt` need finite bounds for all free parameters. In the Fit panel a line **bumps options** accepts `name=value` pairs, in Python `problem.fit("bumps", "dream", bumps={...})`. Unknown names for the chosen method raise an error that lists the valid ones.

| Method         | Options                                                                                                                |
| -------------- | ---------------------------------------------------------------------------------------------------------------------- |
| `lm`, `newton` | steps, starts, ftol, xtol, jump                                                                                        |
| `amoeba`       | steps, starts, radius, xtol, ftol, jump                                                                                |
| `de`           | steps, pop, CR, F, ftol, xtol                                                                                          |
| `dream`        | samples, burn, pop, init (`eps`, `lhs`, `cov`, `random`), thin, alpha, outliers (`none`, `iqr`, `grubbs`, `mahal`), trim, steps |
| `pt`           | steps, nT, Tmin, Tmax, CR, burn                                                                                        |

Notes from the bumps manual: for 2 digits in the 68 % interval DREAM needs about 1e5 samples, for the 95 % interval about 1e6 (`samples=100000`). `init=lhs` spreads the start population over the bounds (multi-modal problems), `init=eps` stays near the start. If the chains look stuck use `thin`. A DREAM run has converged when the lowest chi² and the parameter traces are flat; check with more `burn` and `samples`. DREAM and `pt` give the posterior covariance and 68 % intervals shown in the result tables. UltraNest is not used: bumps DREAM covers the posterior sampling.

## Saving a fit result

Fit panel → Result page: **Export result…** and **Copy summary image**.

| Format                 | Content                                                                                                                                                                                                                       |
| ---------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `.xlsx`                | Sheets: Summary, Parameters (value, error, bounds, at-bound flag), Correlation (colour scale), Covariance, Curve (q, data, error, fit, residual, with chart), Distributions (size distributions, with chart), Plot (summary picture), Report. Needs `openpyxl` (+ `pillow` for the picture). |
| `.csv`                 | Parameters, correlation, covariance, curves and (if the model has a size distribution) `_distributions.csv`.                                                                                                                  |
| `.pdf`, `.png`, `.svg` | Summary figure: data + fit + residuals + size distributions N(x) and N(x)·x^p + parameter table + correlation. Needs `matplotlib`.                                                                                            |
| clipboard              | The same summary figure as an image.                                                                                                                                                                                          |

**Size distributions in the export.** For every fitted size distribution the export contains N(x) (amplitude included) and N(x)·x^p on 400 logarithmic x points (integration range of the distribution widened to `[max(lo/2, hi·1e-4), hi·1.5]`): sheet **Distributions** with chart (xlsx), `<name>_distributions.csv`, and two panels in the summary figure (N(x) left, N(x)·x^p right). The exponent **p is read from the Size-distribution dock** of the explorer (field "p", default 3) at the moment of the export and is written into the sheet header, the CSV header, the figure title and the Summary sheet, so the file documents which weight was used. Python: `export.save_xlsx(path, result, problem, dist_power=6)`.

A parameter is flagged *at bound* if it lies within `max(1e-4·s, 1e-6·(hi-lo))` of a bound, where s is the larger of the fitted value and the start value; the second term applies only to a finite range. There is no absolute tolerance, so a parameter of the order of 1e-12 (the amplitude N of a size distribution) is not flagged only because it is small, while one that was driven from its start value towards 0 is.

## Tests

```
pip install pytest openpyxl pillow matplotlib PySide6 bumps
set QT_QPA_PLATFORM=offscreen      (Windows cmd; Linux: export ...)
set MPLBACKEND=Agg
python -m pytest tests/studio -q
```

| File                            | Covers                                                                          |
| ------------------------------- | ------------------------------------------------------------------------------- |
| `conftest.py`, `mock_sasfit/`   | Fixtures: mock libsasfit (compiled by `build.py`), isolated settings file, Qt shutdown |
| `support.py`                    | Toy models, data generation, problem factories (no libsasfit needed)            |
| `test_catalogue.py`             | header catalogue: names, parameters, search                                     |
| `test_loader.py`                | plugin loading with `do_init`, precision defaults, errors, `install-binaries` (mock libraries) |
| `test_model.py`                 | `ScatteringContribution`, `ModelFunction`, `Pipeline`                           |
| `test_integration.py`           | integration methods of the size-distribution integral                           |
| `test_fitting.py`               | `FitProblem`: backends, constraints, several data sets, errors                  |
| `test_constraint_names.py`      | identifiers of parameters in constraints (`1.sd.mu` -> `c1_sd_mu`)              |
| `test_speedups.py`              | variable projection, cache, coarse to fine, fixed grid, profiling, node check   |
| `test_project.py`               | saving and loading `.sasstudio` projects                                        |
| `test_real_sasfit.py`           | checks against the real SASfit libraries (skipped without them)                 |
| `test_background.py`            | power-law background, VP with `bg.A`, alpha free/fixed                          |
| `test_priors.py`                | all prior forms, backend restrictions, MAP behaviour                            |
| `test_global.py`                | rescaling, multistart, global algorithms                                        |
| `test_export.py`                | xlsx/csv/pdf/png content, at-bound flags                                        |
| `test_export_distributions.py`  | size distributions N(x), N(x)·x^p in xlsx/csv/figure                            |
| `test_bumps_options.py`         | bumps methods and option passing (needs `bumps`)                                |
| `test_fit_panel_gui.py`         | Fit panel with a stub explorer (offscreen Qt)                                   |
| `test_param_row.py`             | parameter row: slider range, editing, bounds, fit box, dummy parameters (Qt)    |
| `test_groups.py`                | `.def` parsing, menu tree, special groups, cache invalidation                   |
| `test_paths.py`                 | library search order, settings file, header/def/plugin folders                  |
| `test_cli.py`                   | new-plugin, SasView adapter (also with the word 'inf' in descriptions), sub-commands |
| `test_examples.py`              | example scripts (valid; run when libsasfit is present), manual includes and screenshot names |
| `test_at_bounds.py`             | the 'at a bound' flag is relative to the size of the parameter                  |
| `test_explorer_gui.py`          | Explorer window, docks, detach, distributions; needs libsasfit, skipped otherwise |

Optional backends (NLopt, iminuit, bumps, cma, dfols) are skipped when not installed.

**GUI tests.** `test_fit_panel_gui.py` uses a stub explorer and needs no SASfit library. `test_explorer_gui.py` creates a real `ModelExplorer` with `ff_sphere` and `sd_LogNorm` and is skipped when the library or PySide6 is missing. Its fixture sets a usable size distribution (N=1, s=0.2, p=1, mu=50), the contrast `eta=1` and switches "log I" off: with the default values the intensity is zero, and `_redraw` stops before it draws the background line and the size-distribution plot.

**Qt shutdown.** `tests/studio/conftest.py` has a session fixture (`_qt_clean_shutdown`) that closes all remaining Qt windows, runs the deferred deletes and collects the Python wrappers when the test session ends. Without it, PySide6 with matplotlib canvases can end the run with a segmentation fault while the interpreter exits, although all tests have passed.


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
- **Model explorer and fitting engine:** mixtures, a compact SASfit-style parameter editor, a fit panel,
  constraints and priors, several optimiser packages and saved projects (see above). SASstudio has none of these.
- **Tools:**
  - the header parser tolerates malformed tables, such as the `</tr></tr>` in `ff_boucher_cyl`;
  - the templates produce working plugins and SasView adapters;
  - `generate_plugin_profiles.py` became `pysasfit-studio profiles`.
- **Export and priors (this fork):** results can be saved as xlsx/csv/figure with the size distributions N(x) and
  N(x)·x<sup>p</sup>; priors of four forms; a power-law background; bumps options and the methods `newton` and `pt`.
- **Bug fix:** `sasview-adapter` replaced every `inf` in the generated parameter list, also inside descriptions
  ("influence" became `float("inf")luence`, which could break the generated file); now only infinite numbers are
  written as `float("inf")`.
- **Tests:** they compile a mock libsasfit with two plugins, one of them importing from the other, so the loader is
  tested on Linux.

## To do

- **Hankel / SESANS:** wrap `sasfit_hankel` and `sasfit_integrate_ctm` with correctly typed strategy setters
  (`N_Ogata` is an `int`), compute G(z) for any form factor, and port the Hankel-strategy benchmark from
  sasfit-collab.
- **Parameter names:** take the names of the remaining built-in libsasfit functions from the SASfit Tcl GUI
  (`sasfit.vfs`). Today only `ff_sphere`, `sd_LogNorm` and `sq_HardSphere` have names; the others are addressed as
  `p0`, `p1`, ...
- **GUI:**
  - ~~add a studio tab to `pysasfit-gui`~~ (done: the *pySASstudio* tab, see {doc}`launcher`);
  - fit several data sets in the explorer (the engine can do it already);
  - use the resolution (dq) in the explorer's fit.
- **Testing:**
  - test against a real Linux build of SASfit (`libsasfit.so`);
  - benchmark the optimisers on real SASfit models (for example with FitBenchmarking) to choose the defaults.
