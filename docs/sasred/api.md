# pySASred: API reference

Generated from the docstrings.

## Reduction of one run

```{eval-rst}
.. automodule:: pysasfit.sasred.reduce
   :members: reduce_run_1d, ReducedRun, efficiency_from_runs, normalised_counts, combined_normalised_counts,
             load_instrument, uncertain_parameter
```

## Uncertainty engine

```{eval-rst}
.. automodule:: pysasfit.sasred.data.uarray
   :members: UArray, Source, constant
```

## Correction steps

```{eval-rst}
.. automodule:: pysasfit.sasred.core.counting
   :members:

.. automodule:: pysasfit.sasred.core.deadtime
   :members:

.. automodule:: pysasfit.sasred.core.normalisation
   :members:

.. automodule:: pysasfit.sasred.core.geometry
   :members:

.. automodule:: pysasfit.sasred.core.beamcentre
   :members:

.. automodule:: pysasfit.sasred.core.averaging
   :members:

.. automodule:: pysasfit.sasred.core.resolution
   :members:

.. automodule:: pysasfit.sasred.core.polar
   :members:

.. automodule:: pysasfit.sasred.core.efficiency
   :members:

.. automodule:: pysasfit.sasred.efficiency_log
   :members: archive_path, archive_efficiency, list_maps, load_map, compare, history, plot_history
```

## Masks

```{eval-rst}
.. automodule:: pysasfit.sasred.core.mask
   :members:
```

## Files and units

```{eval-rst}
.. automodule:: pysasfit.sasred.io.sinq
   :members: read_sans_i, RawRun, MissingField

.. automodule:: pysasfit.sasred.io.nxcansas
   :members:

.. automodule:: pysasfit.sasred.io.ascii
   :members:

.. automodule:: pysasfit.sasred.io.transmission
   :members:

.. automodule:: pysasfit.sasred.catalogue
   :members:

.. automodule:: pysasfit.sasred.units
   :members:
```

## GUI

```{eval-rst}
.. automodule:: pysasfit.sasred.gui
   :members: MainWindow, main

.. automodule:: pysasfit.sasred.mask_editor
   :members: MaskEditor
```
