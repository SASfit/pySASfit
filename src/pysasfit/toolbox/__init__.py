"""pyTOOLbox: tools around the HDF5 / NeXus raw data of the SANS instruments.

viewer        local HDF5 viewer in the spirit of myHDF5 / H5Web: tree, search, line plots,
              heatmaps of N-dimensional data, matrix view, NeXus NXdata default plots,
              attributes
meta_lister   metadata of a run range as a table (copy, Excel / CSV, edit and write back
              with old_<name> copies of the original values)
hdf5          the non-GUI helpers of the viewer
gui           Qt window with both as tabs (``pysasfit-toolbox`` and the "pyTOOLbox" launcher tab)

Needs ``h5py`` (``pip install pysasfit[toolbox]``; Excel export also needs ``openpyxl``).
"""
