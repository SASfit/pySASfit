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

## SESANS and multiple scattering (Hankel transforms)

```{eval-rst}
.. automodule:: pysasfit.studio.hankel
   :members: SESANSTransform, MSASTransform, make_transform, hankel_weights, log_grid, sphere_sesans
```

```{eval-rst}
.. automodule:: pysasfit.core.sesans_io
   :members: load_sesans, SESANSData, is_sesans_file
```

## SasView models (sasmodels)

```{eval-rst}
.. automodule:: pysasfit.studio.sasmodels_bridge
   :members: available, register_all, set_length_unit, length_unit, model_names, DISTRIBUTIONS, sasview_distribution,
      register_distributions
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

## Saving a fit result

CSV, Excel (with charts) and the summary figure; the size distributions N(x) and N(x)·x^p are included, `dist_power`
is the exponent p (the explorer's size-distribution dock sets it).

```{eval-rst}
.. automodule:: pysasfit.studio.export
   :members: DISTRIBUTION_POWER, distribution_curves, distribution_rows, save_csv, save_xlsx, summary_figure,
      figure_bytes, save_figure
```

## Where the libraries, headers and plugins are

```{eval-rst}
.. automodule:: pysasfit.studio.paths
   :members:
```

## Menu tree of the functions

```{eval-rst}
.. automodule:: pysasfit.studio.groups
   :members:
```

## Command line

```{eval-rst}
.. automodule:: pysasfit.studio.cli
   :members: main, new_plugin, sasview_adapter
```

## Projects

```{eval-rst}
.. automodule:: pysasfit.studio.project
   :members: save, load, to_dict, load_dict
```
