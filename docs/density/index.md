# pyDENsity: electron-density maps with DENSS

pyDENsity reconstructs low-resolution **electron-density maps** from SAXS/SANS curves with **DENSS** (DENsity from
Solution Scattering; T. D. Grant, *Nature Methods* **15**, 191 (2018)). DENSS works on a grid in real space
(iterative structure-factor retrieval with a shrink-wrap support), so unlike bead models the map can have internal
contrast, and unlike spherical-harmonics envelopes it is not limited to star-shaped bodies.

DENSS is **not** part of pySASfit. It is a separate GPL-3.0 package, pure Python (numpy, scipy):

```bash
python -m pip install denss          # into the Python that runs pySASfit (or set Preferences → pyDENsity)
```

pyDENsity runs DENSS as a separate process (`python -m denss.scripts...`) and reads the files it writes; no DENSS code
is imported into pySASfit. Please cite Grant (2018) for maps made with it.

* launcher tab **pyDENsity**, or stand-alone: `pysasfit-density [data files]`
* scripts: `from pysasfit import density`

## NumPy 2

DENSS 1.8.7 (the current release) does not run unchanged with current NumPy: it uses `np.in1d` and `np.trapz`, which
NumPy 2.4 removed, and its MRC writer passes a one-element array to `struct.pack`, which NumPy 2 rejects (the single
map fails at the end, `denss-all` when writing the reference map). pyDENsity therefore starts DENSS through a small
shim that adds `np.in1d` (as `np.isin`) and `np.trapz` (as `np.trapezoid`) and hands the MRC writer a plain float box
size. The DENSS installation itself is not changed; with an older NumPy the shim does nothing. The same patches
are written as `sitecustomize.py` into a `.pydensity` folder of every run and put first on `PYTHONPATH`: `denss-all`
computes its maps in a `multiprocessing` pool, whose workers are fresh Python processes on Windows and macOS (spawn)
and would otherwise fail with `module 'numpy' has no attribute 'in1d'`. A `sitecustomize` of your environment still
runs after it.

## Panels

| panel | content |
|---|---|
| Data | curves (text, pySASred ASCII, NXcanSAS; q in 1/nm) with their point range, or a GNOM `.out` file (DENSS then uses its regularised curve and Dmax; give the q unit of the file); **flat background** of the high-q tail and its subtraction (see below) |
| DENSS | the installation found (Python, version), *Search again*, *Self test*. **Map**: mode FAST (simple shapes) / SLOW (complex molecules) / MEMBRANE (negative contrast allowed), Dmax (empty: from the GNOM file or estimated by DENSS, see [Dmax](#dmax)), oversampling, grid points or voxel size, steps, rotational symmetry (n, axis, cyclic / dihedral / icosahedral), seed, GPU (CuPy), further options; *Run one map*. **Averaged map**: number of maps (20 by default), cores, enantiomer selection; *Run denss-all* (maps, enantiomer selection, alignment, average, FSC resolution) |
| Density (3D) | isosurface of the selected map; *enclosed volume* sets the level so that the surface encloses the given percentage of the DENSS support volume; optionally the support (translucent); optional scale bar and size box (x y z or principal axes: length, width, thickness). Same 3D panel as pyATsas (rotate, zoom, image; see *Size guides* in the pyATsas page) |
| I(q) | the data used by DENSS, the scattering of the map (single map, or the mean of the maps of an average) and the residuals |
| Convergence | χ², Rg and support volume per step of a single map |
| Slices | three central slices through the map |
| Results | one row per run: Rg, χ², support volume, resolution (average), steps or number of maps, map file; *Log*: command line, run folder, DENSS output |

A single FAST map of a protein-sized particle takes seconds to a minute; *denss-all* with 20 maps takes minutes to
hours (use several cores). One map is one solution of a non-unique problem; for publication use the average, whose
FSC resolution says how well the maps agree. Every run gets its own folder (*Preferences → pyDENsity → folder for the
DENSS runs*) with the `.mrc` maps, fits and log; ChimeraX, PyMOL or Chimera open the maps (*File → Open a density map*
also reads them back).

Units: the panels use q in 1/nm (pySASfit), DENSS works in 1/Å and Å; pyDENsity converts. Dmax is entered in nm;
maps, Rg and volumes are in Å as DENSS reports them.

## Dmax

Give **Dmax** whenever you know it (from a P(r): pyATsas GNOM/DATGNOM, pyINVersion), or load a GNOM `.out` file. Without
it DENSS estimates Dmax itself (`estimate_dmax`: unregularised P(r) fits of the first half of the data over a range of
Dmax values). For large particles, whose Guinier range is only a few points, that estimate is unreliable:

* it can stop with `ValueError: cannot convert float NaN to integer` (the Rg of one trial P(r) is not real). pyDENsity
  then runs DENSS again with Dmax = 3.5 Rg, the rule DENSS uses itself, Rg from a Guinier fit over q·Rg ≤ 1.3
  (`density.guinier_rg`), and says so in the log and the run table;
* when it does not stop it may be far too large: for a sphere of R = 250 nm (Dmax 500 nm) measured from
  q = 0.004 1/nm it gave 4 to 29 µm depending on the noise, which makes the voxels and the run time huge. pyDENsity
  warns when DENSS's Dmax exceeds 3 × 3.5 Rg.

The voxel size of a FAST map is about oversampling × Dmax / grid points (3 × Dmax / 32); for fine detail of a large
particle use SLOW (64 points) or set the grid points.

## Flat background

DENSS fits **no constant**: its smoothing of raw data (`Sasrec`, a p(r) expansion up to Dmax) has no constant term,
and in every iteration and for the final χ² the map is compared with the data with a scale factor only
(`calc_chi2(..., scale=True, offset=False)`; the offset option of that function is used only by `denss-pdb2mrc`). A flat
background — incoherent scattering of hydrogen in SANS, a buffer mismatch — is therefore taken as particle scattering:
the map gets fine detail or density at the edge of the support, Rg and volume are distorted. Subtract it first.

The group **Flat background** in the Data panel fits I = A q⁻ᴮ + bg to the high-q tail by weighted least squares
(`pysasfit.atsas.core.fit_power_background`; numpy/scipy only, no ATSAS needed):

* *tail q* (start: the last 30 % of the points of the range, button *30 %*), the exponent B free or fixed (4: Porod,
  sharp interfaces; 2: Gaussian chains); *Fit background* shows bg ± σ, B, χ²/n and draws the fit and the bg line in
  the I(q) plot. A bg below 2σ is reported as not significant; χ²/n > 5 means the tail is not described by a power
  law plus a constant alone (form-factor oscillations still in the range, or a wrong exponent).
* The tail must lie beyond the last form-factor features (for a compact particle q·Rg ≳ 10); B and bg are strongly
  correlated when it is short.
* **Subtract → new curve** adds `… minus bg <value>` (errors unchanged) and selects it; the original stays; the value
  can be edited first. Subtracted curves come back with the session; *Copy Python* contains the fit and the
  subtraction.
* Before every run on a curve whose background was not subtracted, pyDENsity fits the tail (last 30 %) itself; if it
  finds a significant constant (> 3σ and > 10 % of the last intensities), it says so and asks whether to run anyway.

For GNOM `.out` input the background must have been subtracted before GNOM (pyATsas has the same tail fit, plus a
DATPOROD cross-check).

## Checked

* With the real DENSS 1.8.7 (through the shim, NumPy 2.5): sphere R = 2 nm: Rg 16.1 Å (exact 15.49 Å, +3.8 %); the
  lysozyme example data of ATSAS (`lyzexp.dat`): one map Rg 15.1 Å (GNOM 15.4 Å), χ² 0.19; *denss-all* with 4 maps:
  Rg 15.14 ± 0.14 Å, resolution 12.8 ± 2.4 Å. Single maps vary (one run of the GNOM `.out` gave 18.7 Å) — the reason
  for averaging.
* The tests use a small fake DENSS written in Python (also on Windows) and, when DENSS is installed, the real one
  (`pytest -m denss`).

## Scripts

```python
from pysasfit import density
from pysasfit.toolbox.merge import read_curves

c = read_curves("lysozyme.dat")[0]                                  # q in 1/nm
r = density.run_denss((c.q, c.I, c.dI), dmax_nm=5.0, mode="FAST", seed=1)
print(r.rg, r.chi2, r.support_volume, r.workdir)                    # Å, -, Å³; r.rho (grid), r.voxel (Å)
lev = density.level_for_volume(r.rho, r.voxel, r.support_volume)
verts, faces = density.isosurface(r.rho, lev, r.voxel)              # surface in Å
a = density.run_denss_all((c.q, c.I, c.dI), dmax_nm=5.0, nmaps=20, cores=8)
print(a.resolution, a.rg_mean, a.map_file)
```

Errors of DENSS raise `DenssError` with the end of its error output; doubtful results carry `warnings`.

## Icon

DENSS has no logo of its own (the DENSS 1.8.7 package contains no image file). pyDENsity uses one when an installed
DENSS ships an image named `*logo*` or `*icon*`; otherwise it shows its own icon: contour lines of a density map on
its blue tile.

## API

Functions and result classes: {doc}`api`.
