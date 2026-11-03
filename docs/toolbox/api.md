# pyTOOLbox: API reference

Generated from the docstrings.

## HDF5 helpers of the viewer

```{eval-rst}
.. automodule:: pysasfit.toolbox.hdf5
   :members:
```

## Metadata lister

```{eval-rst}
.. automodule:: pysasfit.toolbox.meta_lister
   :members: collect_entries, write_values, short_names, to_native, MetaLister
```

## Merging reduced curves

```{eval-rst}
.. automodule:: pysasfit.toolbox.merge
   :members: Curve, CurveGroup, Overlap, Merged, read_curves, read_folder, curve_from_reduced, group_curves, sample_key,
             merge, merge_groups, series, write_ascii, write_nxcansas, file_stem, fields_of

.. automodule:: pysasfit.toolbox.merge_gui
   :members: MergeTool, send_reduced
```

## 2D data

```{eval-rst}
.. automodule:: pysasfit.toolbox.image2d
   :members: Image2D, read_images, image_from_reduced

.. automodule:: pysasfit.toolbox.image_gui
   :members: ImageTool

.. automodule:: pysasfit.toolbox.analysis2d
   :members: radial_average, sector_mask, azimuthal_average, box_profile, sin2_decomposition, sum_images,
             region_integral, rocking_curve, pixels_in_shape, beam_centre_of_gravity, q_edges
```

## 1D data

```{eval-rst}
.. automodule:: pysasfit.toolbox.data1d
   :members: Data1D, from_curve, read_data1d, write_ascii, interpolate, combine

.. automodule:: pysasfit.toolbox.data1d_gui
   :members: Data1DTool
```

## GUI

```{eval-rst}
.. automodule:: pysasfit.toolbox.viewer
   :members: HDF5Viewer, image_colorbar

.. automodule:: pysasfit.toolbox.gui
   :members: MainWindow, main
```
