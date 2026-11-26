# Scripting

The examples are in `examples/studio/` and run as they are (`python examples/studio/01_evaluate_functions.py`).
They need the SASfit libraries; the data are in `examples/studio/data/`.

## Evaluate functions

```{literalinclude} ../../examples/studio/01_evaluate_functions.py
:language: python
:start-after: "# --8<-- [start:evaluate]"
:end-before: "# --8<-- [end:evaluate]"
```

A SASfit error raises `studio.SASfitError` with SASfit's message; `function(name, errors="warn")` changes that.

## Build a model

See {doc}`models` and `02_mixture_model.py`. The older, shorter interface `Pipeline` fits in one call:

```python
from pysasfit.studio import ModelFunction, Pipeline
pipe = Pipeline()
d = pipe.add_dataset("data.dat")
m = pipe.add_model(model, dataset=d)
res = pipe.fit(m, ["0.sd.N", "0.sd.mu", "0.sd.s", "background"])
print(res.summary())
```

Parameter addresses are `"<contribution>.<ff|sd|sq>.<name>"`, `"scale.<i>"`, `"background"`, `"bg.A"` and `"bg.alpha"`.

## Fit with the engine

`03_fit_with_engine.py` ({doc}`fitting`) fits the bimodal sphere data of SASfit with bounds, a constraint and a prior
and compares optimisers. `FitSummary` holds values, errors, covariance, χ², the fitted curves and the state of the
constraints; `res.text()` prints them.

## Export

`04_export_results.py`, see {doc}`results`.

## Open the explorer

```{literalinclude} ../../examples/studio/05_open_explorer.py
:language: python
:start-after: "# --8<-- [start:explore]"
:end-before: "# --8<-- [end:explore]"
```

## Reproducing SASfit

The fit of `data/SASfitTestData/test.dat` reproduces the parameters of SASfit's own project file
(`SASfit_test.sas`) to six digits; `examples/studio/example_script.py` does it with the `Pipeline` interface.
