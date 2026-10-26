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
```

## GUI

```{eval-rst}
.. automodule:: pysasfit.toolbox.viewer
   :members: HDF5Viewer, image_colorbar

.. automodule:: pysasfit.toolbox.gui
   :members: MainWindow, main
```
