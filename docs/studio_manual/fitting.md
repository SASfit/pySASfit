# Fitting

The explorer's Fit dock and the script interface use the same engine, `pysasfit.studio.fitting.FitProblem`.

## Ingredients

**Parameters** are the ticked rows. Each has a start value and optional bounds; bounds are a uniform prior.
Parameters can be common to several data sets or local to each (script interface; the explorer fits one data set).

**Constraints** are inequalities between parameters, for example that the first population is larger than the
second: `sd_mu > c1_sd_mu`. In the Constraints tab the names are the parameter labels with `.` replaced by `_`.

**Priors** add knowledge to the data, one per line as `label form args`:

| form | example | meaning |
|---|---|---|
| Gaussian | `ff.R normal 50 5` | a known value with uncertainty |
| log-normal | `ff.R lognormal 50 1.3` | a positive scale known to a factor |
| half-normal | `ff.sigma halfnormal 0.2` | a relative width that is probably small |
| log-uniform | `ff.R loguniform` | a scale that may span decades |

See [Priors](../studio.md#priors-for-positive-parameters-and-ranges) in the reference for the penalty terms and
which optimisers accept which prior.

## The engine in a script

```{literalinclude} ../../examples/studio/03_fit_with_engine.py
:language: python
:start-after: "# --8<-- [start:problem]"
:end-before: "# --8<-- [end:problem]"
```

```{literalinclude} ../../examples/studio/03_fit_with_engine.py
:language: python
:start-after: "# --8<-- [start:fit]"
:end-before: "# --8<-- [end:fit]"
```

## Choosing an optimiser

```{figure} ../_static/studio/fit_panel.png
:alt: The fit dock
:width: 60%

The Fit dock: parameters ticked for fitting, optimiser and algorithm, options, and the speed-up boxes.
```

| situation | use |
|---|---|
| good start, normal fit | scipy `trf` (the default) |
| poor start, or the result depends on the start | `multistart`, CMA-ES or `differential_evolution` with bounds and *refine* |
| constraints | NLopt `LD_SLSQP` or `LN_COBYLA` with *refine* |
| asymmetric errors | iminuit `migrad+minos` |
| full uncertainty, correlations, several solutions | bumps `dream` |
| noisy or slow model | CMA-ES, DFO-LS, or the speed-ups below |

*Refine* (`polish=True`) follows a derivative-free or global optimiser by a least-squares fit, which gives the exact
minimum and the errors. The comparison of optimisers with evaluation counts is in the
[reference](../studio.md#choosing-an-optimiser).

```{literalinclude} ../../examples/studio/03_fit_with_engine.py
:language: python
:start-after: "# --8<-- [start:compare]"
:end-before: "# --8<-- [end:compare]"
:dedent: 4
```

### bumps and DREAM

The `bumps` backend has `lm`, `amoeba`, `de`, `dream`, `newton` and `pt`. The line *bumps options* in the Fit dock
takes `name=value` pairs, for example `samples=100000 burn=500 init=lhs`; in Python
`problem.fit("bumps", "dream", bumps={"samples": 100000})`. DREAM needs finite bounds and returns the posterior:
standard deviations and 68 % intervals. See [bumps options](../studio.md#bumps-optimisers-and-options).

## Searching for the global minimum

SAS fits have local minima. Fit with `multistart` (Latin-hypercube starts, each refined; the report says how often the
best minimum was found) or a global optimiser, then check from the result with `trf`. All of them need finite
bounds on the free parameters. See [Searching for the global minimum](../studio.md#searching-for-the-global-minimum).

## Faster fits

Variable projection (the linear parameters scale, N and background are solved exactly), the form-factor cache, the
coarse-to-fine fit and the fixed size grid are tick boxes in the Fit dock; they do not change the result beyond the
integration accuracy. See the [reference](../studio.md#faster-fits-dfo-ls-variable-projection-cache-coarse-to-fine-fixed-grid).

## When the fit does not work

- A parameter at a bound, or a huge error: two parameters are degenerate (scale and N, volume fraction and scale).
  Fix one or add a prior from independent knowledge.
- χ²/dof far from 1 with a good-looking curve: check the error bars of the data.
- Stops at a local minimum: better start values, or a global search.
- Slow: fewer free parameters, a fit q range, *Check nodes*, the speed-ups.
