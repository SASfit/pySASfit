# pySASred: API reference

Generated from the docstrings.

## Reduction of one run

```{eval-rst}
.. automodule:: pysasfit.sasred.reduce
   :members: reduce_run_1d, ReducedRun, efficiency_from_runs, normalised_counts, combined_normalised_counts,
             load_instrument, uncertain_parameter
```

## Raw files (SANS-I, SANS-LLB)

```{eval-rst}
.. automodule:: pysasfit.sasred.io.raw
   :members: read_raw, instrument_of

.. automodule:: pysasfit.sasred.io.sans_llb
   :members: Panel, read_sans_llb, bank_run, panel_geometry, panel_offsets, beam_position_mm, panel_mask
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

.. automodule:: pysasfit.sasred.core.beamprofile
   :members:

.. automodule:: pysasfit.sasred.core.beamstop_transmission
   :members:

.. automodule:: pysasfit.sasred.legacy
   :members:

.. automodule:: pysasfit.sasred.core.monitors
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

## Reduction of table rows (2D, then average)

```{eval-rst}
.. automodule:: pysasfit.sasred.absolute
   :members:
```

## Selecting and grouping runs

```{eval-rst}
.. automodule:: pysasfit.sasred.selection
   :members:
```

## Reduction table

```{eval-rst}
.. automodule:: pysasfit.sasred.series
   :members:

.. automodule:: pysasfit.sasred.reduction_table
   :members:
```

## Reduction plan

```{eval-rst}
.. automodule:: pysasfit.sasred.plan
   :members: build_plan, Plan, Entry, check_rates, role_of, ROLES, PATTERNS
```

## Detector model

```{eval-rst}
.. automodule:: pysasfit.sasred.core.detector_model
   :members:
```

## Water at two distances

```{eval-rst}
.. automodule:: pysasfit.sasred.core.water
   :members:
```

## Scattering strength

```{eval-rst}
.. automodule:: pysasfit.sasred.core.scattering_strength
   :members:
```

## Grouping and equivalence of runs

```{eval-rst}
.. automodule:: pysasfit.sasred.grouping
   :members: FIELDS, hdf_field, metadata, group_runs, RunGroup, equivalence, Equivalence, Trend, check_runs, plot_equivalence, mid_time
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

.. automodule:: pysasfit.sasred.efficiency_gui
   :members: EfficiencyMixin

.. automodule:: pysasfit.sasred.table_gui
   :members: ReductionTableWidget, GeometryPage, RunSetDialog
```
