# pyTOOLbox: small helpful tools

```{image} ../pySASfit-logo-bundle/logos/pyTOOLbox/svg/pyTOOLbox-light.svg
:alt: pyTOOLbox
:height: 56px
:class: only-light
```

```{image} ../pySASfit-logo-bundle/logos/pyTOOLbox/svg/pyTOOLbox-dark.svg
:alt: pyTOOLbox
:height: 56px
:class: only-dark
```

`pysasfit.toolbox` collects small helpful tools, at present around the HDF5 / NeXus raw data of the SANS
instruments and the reduced data. It has four tabs:

* the **HDF5 viewer**, a local viewer in the spirit of [myHDF5](https://myhdf5.hdfgroup.org/) / H5Web that runs
  in the Python environment, without uploading files;
* the **metadata lister**, formerly `tools/hdf_meta_lister.py`: the metadata of a range of runs as a table, to
  copy, export and correct;
* **merge geometries**: the reduced curves of the detector distances (and wavelengths) of a sample merged
  into one curve, grouped by sample name (regular expression) and any header item (temperature, pressure,
  magnetic field, …), with series plots ({doc}`merge`);
* **2D data**: the reduced 2D images (pySASred in memory or files, BerSANS) in qx–qy, pixels or polar
  coordinates; analysis (sectors, azimuthal plots, A + B sin²ψ, box projections, sums) to follow ({doc}`image2d`).

The GUI is `pysasfit-toolbox [file.hdf …]`, or the **pyTOOLbox** tab of `pysasfit-gui` (the last tab). The two tabs
can be detached into separate windows to use them side by side. A double-click on a file name in the listing
opens that file in the viewer.

pyTOOLbox was called **pySASbox** before. The old names keep working: `import pysasfit.sasbox` (with a
deprecation warning), the command `pysasfit-sasbox`, the extra `pysasfit[sasbox]`; preferences saved under
`sasbox.…`, the last session and the lister's file choices are taken over.

```{toctree}
:maxdepth: 2

usage
merge
image2d
data1d
```

The API reference is {doc}`api` (also listed under {doc}`../api/index`).
