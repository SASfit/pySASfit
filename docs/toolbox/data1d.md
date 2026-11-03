# 1D data

The tab **1D data** of pyTOOLbox collects one-dimensional data sets — I(q), azimuthal profiles I(ψ), box
profiles, rocking curves — and does simple arithmetic on them with uncertainty propagation (module
{mod}`pysasfit.toolbox.data1d`).

## Sources

* *Add files…*: NXcanSAS, pySASred ASCII, BerSANS `SANSDIso`, merged curves and files written by this tab
  (`# pyTOOLbox 1D` header, keeps the x label and units);
* *From pySASred*: the reduced curves in memory, without writing files;
* *From merge*: the merged curves of *Merge geometries*;
* the results of *Compute → 1D data* and *Rocking curve* of the tab **2D data**.

Names can be edited in the list; *Save selected…* writes ASCII files (x, y, dy, dx) with the metadata as
json in the header.

## Operations

| operation | result |
|---|---|
| × number, ÷ number, + number | y·a (σ·a), y/a, y + c for every selected data set |
| Cut x range | only x min ≤ x ≤ x max |
| a + b, a − b, a × b, a ÷ b, a − f·b | b is interpolated at the x of a (log–log where both are positive, else linear); only the overlapping x range is kept |

Uncertainties are propagated to first order for independent data sets (σ of a ± b: √(σa² + σb²); ratio and
product: relative errors in quadrature). a and b must have the same x quantity (q with q, ψ with ψ).

## Plot

The selected data sets, with error bars; x and y scales lin / log (log is chosen automatically for I(q)),
optional vertical offset factor between the curves.
