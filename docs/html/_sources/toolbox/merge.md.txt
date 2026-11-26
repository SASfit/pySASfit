# Merge geometries

The page has three tabs on the left, one per step, so that each table gets the full height: **1 · Curves**
(the curves read, with a filter field: a regular expression on file, sample or geometry), **2 · Group and
merge** (grouping and merge options, button *Group and merge*) and **3 · Merged groups** (the result,
selected for the plot, and saving; opened after merging). The line below the tabs counts curves,
geometries and merged groups. The step numbers below refer to these tabs.

The tab **Merge geometries** of pyTOOLbox (module {mod}`pysasfit.toolbox.merge`) merges the reduced curves of
the different detector distances (and wavelengths) of a sample into one curve. Curves are grouped by the sample
name, through a regular expression, and by any other header item with a tolerance, so that a sample measured at
several temperatures, pressures, magnetic fields or shear rates gives one merged curve per condition. These
can then be plotted as a series.

## 1. Reduced curves

**Without files:** *From pySASred* takes the reduced results of every open pySASred window (single runs and
rows of the reduction table) directly from memory; in pySASred *Reduction → Send results to pyTOOLbox merge
(no files)* does the same from the other side (the pyTOOLbox tab of the launcher, or a new pyTOOLbox
window). The header items are the same as in the files ({func}`pysasfit.toolbox.merge.curve_from_reduced`).
Taking them again after a new reduction replaces the curves of the same run or table row.

*Add files…* / *Add folder…* read:

| format | recognised by | q | header items |
|---|---|---|---|
| pySASred NXcanSAS (`.h5`) | HDF5, SASentry | from the `units` attribute (1/nm or 1/Å) | `SASprocess/metadata`: named items (`sample`, `SD_m`, `wavelength_A`, `coll_m`, `thickness_cm`, …) and every scalar entry of the raw file as `hdf:<path>` (a 1-D log gives its mean) |
| pySASred ASCII (`.dat`) | `# pySASred` | header `q [1/nm]` or `q [1/A]` | `# metadata_json = {…}` |
| BerSANS 1D (`D0082130.080`, …) | `%File` … `Type=SANSDIso` | 1/nm | every `key=value` of `%Sample`, `%Setup`, `%History` as `Sample/Temperature`, `Setup/SD`, …; `SD`, `Lambda`, `Collimation` give `SD_m`, `wavelength_A`, `coll_m` |
| merged files written here | `# pyTOOLbox merge` / SASprocess `group_values` | | the group values |
| plain columns q, I[, dI[, dq]] | | 1/nm unless the header mentions 1/Å | file name as sample |

pySASred writes the header items into its NXcanSAS and ASCII files from now on (`pysasfit.sasred.io.metadata`).
Files reduced earlier have the sample name, run and parameters only. The geometry of a curve, e.g.
`SD 8.00 m, λ 0.600 nm, coll 8.0 m`, comes from the recorded distance, wavelength and collimation.

## 2. Group

* **sample name (regex)**: a regular expression searched in the sample name. The capture groups `(…)` form the
  key (joined by `_`), without groups the whole match. Names that do not match are left out, and the log
  lists them. Examples for `EXP-005370-A_1`:

  | expression | key | effect |
  |---|---|---|
  | (empty) | `EXP-005370-A_1` | one group per sample name |
  | `EXP-005370-(\w)_(\d)` | `A_1` | same, shorter key |
  | `EXP-005370-(\w)_` | `A` | all positions/repeats of sample A together (split them by further items) |
  | `EXP-005370-[AB]_1` | `EXP-005370-A_1` | only samples A and B, position 1 |

* **further header items** with *± tolerance*: text must be equal, numbers lie within the tolerance (empty:
  equal values; a number: largest spread within a group; `1%`: relative), as in pySASred's grouping. Choose,
  for example, `hdf:sample/temperature` ± 0.5 K, `Sample/Temperature` for BerSANS files, a magnetic field ± 0.01 T,
  a pressure, a shear rate, or `sample` to keep different names apart while the key joins them.

Curves of the same geometry that fall into one group (repeats that are not separated by an item) are all
kept (averaged with *weighted mean*); the group notes say so.

## 3. Merge

The **absolute scale of every geometry is kept**. *overlaps* chooses what happens where geometries overlap:

* **keep all points** (default): every point of every geometry stays as it is, with its own dI and dq
  (resolution), sorted by q, as BerSANS' merge does. The plot shows the geometries with different markers
  (no line through the interleaved points); the files carry the geometry of every point (ASCII: 5th column,
  index explained in the header; NXcanSAS: `source_geometry`).
* **weighted mean**: in an overlap the points of all geometries are averaged with weights 1/dI², on the bins of
  the coarsest curve there (q = weighted mean, dq including the spread of the q values); index −1 marks
  these points.

Every overlap is checked. The curve with the higher q is interpolated (log-log) at the q of the other, and the
weighted mean of their ratio gives a **mismatch factor** with its statistical uncertainty, z = (f − 1)/σ_f
and χ²/ndf of the difference. The column *overlap mismatch* shows the worst overlap of a group: orange above 2 %,
red above 5 %. The tooltip lists all overlaps. A mismatch points to a problem in the reduction (distance offset,
transmission, background, beam centre, the edges of a detector); the statistical σ_f does not include the
systematic uncertainties of the absolute scale.

*scale to* (optional) scales the other geometries to the chosen one. Scaling is a pure multiplication: I and dI
of a geometry are multiplied by one number; q, dq and the points are not changed.

* **one factor per geometry for all groups** (default): the factor of every geometry is the weighted mean
  (1/σ²) of the overlap factors of all groups (chained from neighbour to neighbour to the reference) and is
  then applied to all merged curves alike — a calibration of that geometry, not a fit per sample. The log
  and the notes give f ± σ and the number of groups it comes from.
* unchecked: every group is scaled with the factors of its own overlaps.

The factors are written into the output.

The table of geometries holds a **q range** and the **points dropped** at the low and high q end of every
geometry, for all groups, like BerSANS' merge mask (`.smi`, `#` = point not used).

## Underlying data of a group

The table of merged groups shows the sample name(s) of every group, and the legend of the plot starts with
them. Below it, *Curves of the selected group* lists the curves that were merged: geometry, sample, source (the
pySASred reduction-table row or run, or the file and entry), the runs summed, q range and number of points.
*Show in pySASred* (or a double-click) selects that row or run in the open pySASred window and brings it to the
front, where its runs, transmissions and notes are; *Show 2D* opens the 2D images of the selected curves. There
is no need to go back to pySASred by hand.

## 4. Plot and save

The plot shows the selected groups: the curves of each geometry (open symbols, scaled when *scale to* is
used), the merged curve (line with error bars in I and, where the curves carry a resolution dq, in q; *I error bars* / *q resolution bars* switch them) and the overlap regions (shaded). Several groups can be selected;
*offset ×* separates them. **series of the selected sample** shows all merged curves with the same key as the
selected group, ordered and coloured by the header item chosen under *by* (e.g. the temperature), with a
colour bar.

*Save selected… / Save all…* write into a folder, with q in 1/nm or 1/Å:

* **ASCII per curve**: q, I, dI, dq; the header lists the sample key, the group values, the source files and
  geometries, q cuts, scale factors, overlap checks and notes, and `group_json`;
* **NXcanSAS per curve**: SASdata (Q, I, Idev, Qdev, `source_geometry` per point: index of the geometry, −1 =
  averaged), SASsample (key and group values), SASprocess (`provenance` as JSON);
* **one HDF5 with all**: every merged curve as its own NXentry in `merged_all.h5`, e.g. a whole temperature
  series in one file.

File names come from the group label, e.g. `A_Temperature20-20.2.dat`. All three outputs can be read again
by the tool, for example to group merged curves of several beam times.

## Scripting

```python
from pysasfit.toolbox import merge as mg
curves, problems = mg.read_folder("analysed")              # NXcanSAS, ASCII, BerSANS
groups, rest = mg.group_curves(curves, r"EXP-005370-(\w)_(\d)", {"hdf:sample/temperature": 0.5})
merged = mg.merge_groups(groups, cuts={"SD 2.00 m, λ 0.600 nm, coll 8.0 m": (0.32, None)})
for m in merged:
    print(m.group.label, [o.text() for o in m.overlaps])
    mg.write_ascii(mg.file_stem(m) + ".dat", m)
mg.write_nxcansas("series.h5", merged)
by_T = mg.series(merged, "hdf:sample/temperature")       # {key: [(T, merged), ...]}
```

With the Vifor data of September 2026 (pySASred, water calibration, +53 mm distance offset) the 2 m and 8 m
curves of the samples A_1 … B_6 agree within −0.6 … −1.1 % in the overlap (0.30–0.63 1/nm). BerSANS data
of July 2024 (no offset) show +2.3 % for EXP-001763_1.
