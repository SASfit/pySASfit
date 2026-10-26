# Using the pySASred window

Start it with `pysasfit-sasred [raw-data folder]` or open the **pySASred** tab of `pysasfit-gui`. Every panel
(Runs, Reduction, Detector, I(q), Error budget, Log) can be moved, stacked as tabs or detached into its own
window by its title bar; **View** shows hidden panels again and **View → Reset layout** restores the start
arrangement.

## A first reduction

1. **Choose the raw-data folder**: *Browse…* in the Runs panel (or *File → Open raw-data folder…*). The table
   lists every `sans<year>n<run>.hdf` with sample name, wavelength, collimation L1, L2, beam stop (type and
   x, y position in mm), attenuator, monitor 1, counting time,
   total counts and whether the run was aborted (monitor preset not reached). Aborted and unreadable runs are
   shown in red. The filter field below the folder selects runs by sample name or run number.
2. **Beam centre**: every geometry (collimation L1, detector distance L2 and wavelength) has its own beam
   centre, because q and the solid angle of each pixel are computed from it, gravity moves the beam with L2 and
   λ and the collimation changes the beam. When the folder
   is read, the centre of each geometry is taken from its direct-beam runs (beam stop out of the beam; median of
   their masked centres of gravity); a geometry without a direct-beam run borrows the centre of the same
   wavelength, preferably the same collimation, at the nearest distance, and says so. Selecting a run shows the centre of its geometry in the x / y
   fields. **From selected run (direct beam)** determines it again from the selected run and keeps it for that
   geometry; values typed into x / y are kept for the geometry of the run shown. The centre is the masked centre
   of gravity, re-centred until it moves by less than 0.1 pixel; the label shows its statistical uncertainty
   (about 0.004 pixel for the Vifor runs) and where it comes from. Runs of different geometries are reduced
   separately (one geometry per *Reduce selected*), each with the centre of its geometry. Pixel indices count from 0, as in Python and GRASP: x is axis 1 (horizontal),
   y is axis 0 (vertical) of the stored image.
3. **Mask**: a new image starts with the standard mask (two outer rows and columns) and a first-guess square
   beamstop around the beam centre. Adjust it with the mask tools, or click **Suggest** in the detector panel:
   it replaces the guess by the rectangle around the beamstop shadow and adds hot pixels found in the image
   shown. See {doc}`masks` for editing, loading and saving.
4. **Look at I(q) on the fly**: with *live I(q) of the selected run* switched on (the default), the radial
   average of the selected run is drawn as a dashed curve in the I(q) panel and follows every change of the beam
   centre, mask, q range, binning, normalisation or sector within a fraction of a second. It is not kept.
5. **Reduce selected** reduces every selected run with the current settings (the file is read again, with its
   checksum) and keeps the result: it appears as points with error bars (Idev) in I(q) and in the list
   *Reduced runs*. *Clear curves* removes them.
6. **Save**: *NXcanSAS…* or *ASCII…* for the run chosen under *Reduced runs*, *All…* for every reduced run
   (both formats, file names `<run>_<sample>.h5` / `.dat`). See {doc}`scripting` for the content of the files.

## The reduction table

The *Reduction table* panel (a tab next to *I(q)*, *Error budget* and *Log*; menu *Reduction → Reduction table*
brings it to the front, *Reduction → Add many samples…* opens the two-step selection directly) has one tab per
geometry (detector distance, wavelength, collimation). Each tab holds the correction data (Cadmium, Water,
Water-BG = empty cell of the water, Sample-BG) and one row per sample, with transmission, thickness, attenuation
factor A and scattering probability p_cS, as in the BerSANS data-reduction window.

When a folder is opened the panel is **filled automatically** (preference *fill the reduction table when a folder
is opened*; **Fill from runs** does it again): the runs are grouped by metadata, the roles come from the sample
names, repeated runs are compared (*check repeated runs*, about 3 s for the Vifor folder) and outliers are
switched off; transmissions come from `trans.txt` in the folder when present. The tab title gives the number of
scattering runs; a geometry with few runs or without water / empty cell is marked with **?** — often a geometry
measured for tests. **Remove geometry** (or the × of its tab) takes it out of the reduction completely, also
when the table is filled again; the raw files are not touched and **Restore removed** brings it back. With the
Vifor data: L2 8 m (91 runs), 2 m (58 runs) and 3.9 m (1 run, marked ?).

**New geometry (by hand)** adds an empty tab for a table filled completely by hand; every cell of Information,
Transmission, Thickness, Attenuator and Probability can be typed, *Copy to all samples below / above* copies
the value of the selected cell. *Add many samples…* in a tab starts with the search conditions of that geometry
(scattering runs, L2, λ and L1 within 1–2 %). **Save…** / **Load…** keep all tabs and the removed geometries in
one JSON file next to the data.

The button in the first column of a row opens its runs as a spreadsheet: *use* (summed or not), run, sample,
kind (scattering / transmission), L2, start time, counting time, monitor, counts per monitor and a comment that
says why a run is left out (outlier with its χ²/ndf against the others and its scale factor, aborted with the
time or monitor reached, beam stop position not matching the image). Outliers are red, aborted runs orange.
Runs are switched on or off with the check box, added by number (`82187, 82190-82195`), and *show other runs of
this sample* lists the runs of the same sample from other groups (e.g. another changer position). Rows whose
runs need a decision (drift, shape change) say so in red in *Notes*; the line below the tables lists what is
still missing (e.g. no Cadmium). **Save…** / **Load…** keep the table as JSON next to the data.

### Add many samples

**Add many samples…** works in two steps, as *Select many samples* in BerSANS
({mod}`pysasfit.sasred.selection`):

1. **Select runs**: a run range (*All available* resets it), optionally a pattern (starting with the first
   run, alternately use n runs and skip m), and conditions on header items checked in the list: text items
   are equal to the value or, with *Regular expressions*, match it as a whole (`EXP-.*`); numbers lie in
   `from ≤ value ≤ to`. The list holds the named items (sample, L2_m, L1_m, wavelength_A, attenuator,
   sample_position_mm, thickness, …, kind = scattering / transmission) and every scalar entry of the raw
   files as *file: <path>*, with the value as stored (e.g. *file: SANS/Dornier-VS/lambda* in nm), so any
   sample environment written to the file (temperature, field, pressure, shear rate) can be used.
   *Search and accept* lists the selected runs on the second page.
2. **Group**: check the items that define a sample and give each number a ± tolerance (empty or 0: equal
   values; text items always equal). All runs of a group lie within ± tolerance of the centre of the group.
   The table below shows the groups as they will be added; *check the repeated runs of every group*
   compares the runs of each group and switches outliers off with their reason. *Add n samples* appends
   one sample row per group, with the transmission from `trans.txt` and the transmission runs of the same
   sample and wavelength.

With the Vifor data, *sample* `EXP-.*`, λ 0.59–0.61 nm, attenuator 0, collimation 8 m and L2 1.99–2.01 m select
the same 52 runs as BerSANS; grouped by sample they give 30 samples, 31 with *sample_position_mm* ± 0.25
(A_2 was measured at two changer positions).

## The Reduction panel

| field | meaning |
|---|---|
| x (axis 1), y (axis 0) | beam centre in pixels, from 0 |
| beamstop guess | shape (square or disc) and half width / radius of the first guess of the beamstop in the mask of a new image; it follows the beam centre until the mask is edited; *Into mask* puts it into an edited mask |
| number of bins, spacing | q bins between q min and q max, logarithmic (constant Δq/q) or linear; empty bins are dropped |
| units | **nm, 1/nm** (default) or **Å, 1/Å** for wavelength and q, in the table, the fields, the plots and the saved files. Switching converts q min / q max |
| q min, q max | limits of the binning; *auto* takes the q range of the unmasked pixels |
| normalise by | monitor 1 (its Poisson error is propagated) or counting time |
| sector average | only pixels within ± half width of the sector centre φ (and of φ + 180°) are averaged. φ = 0 is the horizontal direction (+x), counter-clockwise. The sector boundaries are drawn in the detector image |
| live I(q) of the selected run | the on-the-fly radial average described above |

The start values of these fields (except the beam centre) are in **Settings → Preferences → SANS reduction**;
the window itself remembers the last values, the folder, the units, the detector-panel choices, the mask and
the panel layout (the last session). *Settings → Forget last session* in the launcher clears it.

## The detector panel

```{image} ../_static/sasred/colour_scales.png
:alt: the same detector image with the four colour scales
```

*Colour scales of the detector image: linear, log(1+I), symlog and asinh (run 82140).*

* **colour scale**: *linear*; *log(1+I)*, where zero-count pixels stay finite and small counts are spread out
  (the default); *symlog*, linear below 1 count and logarithmic above, which also shows negative values (after a
  subtraction); *asinh*, a smooth version of symlog. The colour bar is labelled 0, 1, 10, 10², … for the
  non-linear scales. The colour range comes from the **unmasked pixels only**.
* **colours**: the colour map.
* **view**: *cartesian (x, y)*, the detector as measured, drawn with the true aspect ratio of the pixels
  (from `x_pixel_size` / `y_pixel_size` of the file) and flush left in the panel; or *polar (φ, q)*: azimuth
  φ horizontally (0° = +x, counter-clockwise, as for the sector), q vertically. In the polar view isotropic
  rings become horizontal bands and anisotropy shows as waves along φ; each (φ, q) cell takes the value of the
  pixel under its centre (no interpolation), cells outside the detector are light grey.

```{image} ../_static/sasred/mask_polar.png
:alt: polar view of the detector image with the mask colours
```

*Polar view of run 82140: the beamstop shadow (red) at the bottom, the drawn rectangle (magenta) as a narrow
spot near φ = 315°, the leaking pixels at the beamstop edge (cyan) near φ = 180°, the detector corners as the
four peaks at 45°, 135°, 225° and 315°.*

The rows of controls above the plot wrap when the panel is narrow, so the panel can be made small and the
I(q) panel large.

## I(q), error budget and log

```{image} ../_static/sasred/iq_live.png
:alt: I(q) with a reduced run and the live curve
:width: 60%
```

*I(q) of the reduced run 82131 (points with Idev) and the live curve of the selected run 82140 (dashed). The
strong rise of 82140 at the lowest q comes from the beam leaking past the beamstop edge (see {doc}`masks`).*

**Error budget** shows, for the run chosen under *Reduced runs*, the relative standard deviation that every
source contributes at each q, and the totals of the statistical (Idev) and the systematic sources. For a sample
run the counting statistics dominate; the monitor is a systematic scale common to all points (a horizontal line).

```{image} ../_static/sasred/budget.png
:alt: error budget of a reduced run
:width: 60%
```

**Log** lists the beam-centre determinations, every reduction with its q range and dead-time factor, the notes
of each reduction (for example "deadtime:detector: uncertainty unknown, not propagated" or "Qdev: wavelength and
in-bin pixel spread only; collimation term not yet included"), the mask suggestions and the saved files.

## In the launcher

In `pysasfit-gui` the window is the first tab. Its own menus are hidden there; their entries appear in the
launcher's **File** menu (*Open raw-data folder…*, *Save NXcanSAS…*, *Save ASCII…*, *Save all reduced runs…*) and
**View** menu. The tab can be detached into its own window, which shows its own menus again (see
{doc}`../launcher`).
