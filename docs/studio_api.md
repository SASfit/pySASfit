# studio: `pysasfit.studio`

The narrative description is in {doc}`studio`. Importing the GUI modules needs PySide6, so only the modules that
work without it are listed here.

## Functions of the SASfit libraries

```{eval-rst}
.. automodule:: pysasfit.studio.functions
   :members:
```

## Models

```{eval-rst}
.. automodule:: pysasfit.studio.model
   :members:
```

## Integration of size distributions

```{eval-rst}
.. automodule:: pysasfit.studio.integration
   :members:
```

## Fitting engine and optimisers

`FitProblem` is the engine behind the Fit panel. The optimiser backends are listed in `BACKENDS`;
`available_backends()` returns those whose package is installed, and `register_backend()` adds another one.

```{eval-rst}
.. automodule:: pysasfit.studio.fitting
   :members: FitProblem, FitParameter, FitDataset, FitSummary, FitCancelled, BACKENDS, SCIPY_ALGORITHMS,
      nlopt_algorithms, available_backends, register_backend, locate
```

## Projects

```{eval-rst}
.. automodule:: pysasfit.studio.project
   :members: save, load, to_dict, load_dict
```
