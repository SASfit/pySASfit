# pySASred: SANS data reduction

```{image} ../pySASfit-logo-bundle/logos/pySASred/svg/pySASred-light.svg
:alt: pySASred
:height: 56px
:class: only-light
```

```{image} ../pySASfit-logo-bundle/logos/pySASred/svg/pySASred-dark.svg
:alt: pySASred
:height: 56px
:class: only-dark
```

`pysasfit.sasred` reduces the raw data of the SINQ SANS instruments (SANS-I now, SANS-LLB later) from the HDF
files to I(q). Every step propagates the uncertainty of every independent source **exactly to first order**: the
counting statistics of each pixel, the monitor counts, and (once known) the dead time, attenuator transmissions
and other calibration constants. The result carries its statistical standard deviation (Idev), the full
statistical covariance between the q points, the systematic scale uncertainties as separate sensitivities, and
a variance budget per source.

The GUI is `pysasfit-sasred`, or the **pySASred** tab of `pysasfit-gui` (the first tab).

```{image} ../_static/sasred/window.png
:alt: pySASred window with the run table, reduction panel, detector image with mask, and I(q)
```

*Run table and Reduction panel on the left, the detector image of the selected run with its mask in the middle,
I(q) on the right: a reduced run (points) and the live radial average of the selected run (dashed).*

## Status

Version **0.1.0.dev0, phase 1** of the package definition: one run is reduced to a monitor-normalised intensity
per steradian, I(q) in counts / monitor / sr.

| implemented | still to come (phase 3 of the definition) |
|---|---|
| reading SANS-I HDF files (units from the attributes, strict required fields, SHA-256 of the raw file) | SANS-LLB reader |
| counting error, dead time (non-paralysable, global or per tube), attenuator, monitor or time normalisation | |
| reduction of the table rows in 2D (transmission, Cd, sample background with (1 − p_cS), water and empty cell with g(λ), thickness, absolute scale), then the radial average; missing corrections allowed and noted | water anisotropy K(2θ) in the reduction; 2D output files |
| geometry with detector tilt, exact solid angle | |
| detector efficiency from water (empty cell, transmissions, solid angle; masked pixels from the x and y wires), transmission tables, dated archive of the maps in one HDF5 file with comparison over time | efficiency and its archive in the GUI |
| beam centre from the direct beam | grouping and equivalence check in the GUI file selection, batch scripts |
| scattering strength after Strunz et al. (2000): transmission over growing apertures (T_Si, τ_UL, lower limit of p_cS), p_cS from the curve, regime and advice, self-consistent p_cS | guided strong-scatterer step in the GUI; general formula with T_EB, (1 − p_cS), K0 |
| BerSANS-like grouping of runs by metadata with precisions; equivalence check of repeated runs (outliers, scale and shape drift, e.g. sedimentation) before summing; reduction plan with roles, summed runs, outliers and missing runs per configuration | *Runs* panel in the GUI |
| standard, suggested and drawn masks | pixel splitting in the averaging |
| radial and sector average, Qdev (wavelength + bin width) | collimation term of Qdev |
| NXcanSAS and ASCII output with the full uncertainty information | live reduction during the measurement |

Values whose uncertainty is not yet known (the dead time τ, the attenuator factors) are used but not
propagated; every output says so in its notes ("uncertainty unknown, not propagated").

```{toctree}
:maxdepth: 2

usage
masks
reduction
formula
grouping
scripting
```

The API reference is {doc}`api` (also listed under {doc}`../api/index`).
