# Grouping and summing repeated runs

BerSANS selects the files of a sample measured under the same conditions with a metadata
filter: a metadata entry is chosen and given a precision (temperature ± 0.5 K, detector distance
± 1 %, ...), and all selected files are summed. pySASred does the same and then asks the question
BerSANS leaves open: **are the runs statistically equivalent, or did something happen during the
measurement** (sedimentation, creaming, aggregation, a bubble, a wrong sample position)?

## Grouping by metadata

{func}`~pysasfit.sasred.grouping.group_runs` puts runs into one group when every selected entry
agrees within its precision:

| precision | meaning |
|---|---|
| `None` | equal values (sample name, attenuator) |
| `0.5` | the values of a group span at most 0.5 in the unit of the entry |
| `"1%"` | the values span at most 1 % |

A group spans at most the precision: 20.0, 20.3, 20.6, 20.9 °C with precision 0.5 give two
groups, not one chain. The entries of {data}`~pysasfit.sasred.grouping.FIELDS` are known
(sample, SD, coll, wavelength, attenuator, sample-changer position, thickness, detector offset and
tilt, start time, ...); any other entry of the raw file, e.g. a sample environment, is added with
{func}`~pysasfit.sasred.grouping.hdf_field`:

```python
from pysasfit.sasred.catalogue import scan_folder
from pysasfit.sasred import grouping as gr

runs = scan_folder(raw_folder)
fields = {"temperature": gr.hdf_field("entry1/sample/temperature")}   # mean of a log
groups = gr.group_runs(runs, {"sample": None, "SD_m": "1%", "wavelength_A": "2%",
                              "attenuator": None, "temperature": 0.5},
                       fields=fields, min_runs=2)
for g in groups:
    print(g.label(), g.run_numbers)
```

The command line does the same: `python -m pysasfit.sasred.grouping RAW sample SD_m=1% attenuator`.

```{tip}
Add `sample_position_mm` with a precision of about 0.5 mm. In the Vifor data of September 2026 it
separates EXP-005370-A_2: the first night measured changer position 3.0 mm, the repeat 29.3 mm.
```

## Are the runs equivalent?

{func}`~pysasfit.sasred.grouping.check_runs` reduces every run of a group with the same beam
centre, mask, efficiency and q bins and compares the curves I_k(q) with their statistical
uncertainties (systematic sources such as a transmission are common to all runs and cancel):

| test | what it tells |
|---|---|
| χ² of all runs against the weighted mean, n_q (n−1) degrees of freedom | equivalent or not |
| leave-one-out χ² of each run against the mean of the others (full statistical covariance) | which run differs; the worst is removed while it fails (Bonferroni), then the rest is tested again |
| scale model I_k(q) = a_k μ(q) | the runs differ by a factor only (shape unchanged) |
| straight line of a_k and of the mean intensity in low, middle and high q bands versus the mid time of the runs, z = slope / σ | a drift, in % per hour |

The verdict:

| verdict | meaning | summing |
|---|---|---|
| equivalent | the runs agree within the counting statistics | all runs |
| outlier | a few runs differ, the rest agree; noted when they are the first or last runs in time (settling, starting change) | without the outliers |
| scale drift | same shape, the intensity follows the time: fewer scatterers in the beam (sedimentation, creaming) or a drifting normalisation | not without a decision |
| scale change | same shape, different intensity without a trend (with two runs a step and a trend cannot be told apart): thickness, transmission, filling | not without a decision |
| shape drift | a q band follows the time; low q changing most points to aggregation or size-selective sedimentation | no |
| shape change | different shapes without a monotonic trend | no |

`Equivalence.accepted` lists the runs that can be summed (empty when a decision is needed);
`summary()` prints the table per run, {func}`~pysasfit.sasred.grouping.plot_equivalence` shows
I_k / mean versus q, the bands and the scale versus time, and the χ² per run.

```python
from pysasfit.sasred.reduce import combined_normalised_counts, load_instrument

e = gr.check_runs(groups[0], beam_centre, edges, mask=mask, efficiency=eff)
print(e.summary())
use = [r for r in groups[0].runs if r.run in e.accepted]
image, m = combined_normalised_counts(use, load_instrument(), notes=[])
```

### A floor for known instabilities

With 10⁷ counts per run the statistics resolve changes of 0.1 %; beam and monitor are rarely that
stable. `rel_floor=0.003` adds 0.3 % of each point in quadrature, so only changes beyond it count.
The default is 0 (pure counting statistics).

### Few counts

The χ² assumes Gaussian errors: use q bins with at least about 20 counts (fewer, wider bins at
high q), otherwise the χ² is too large for equivalent runs.

## What the Vifor data show

The repeated runs of September 2026 (about 11 h apart) give, with 30 q bins:

* most samples at 2 m and many at 8 m: *equivalent* (χ²/ndf 0.7–1.7);
* EXP-005370-A_2: *shape change*, χ²/ndf ≈ 2·10⁴, the two runs sat at different changer positions;
* EXP-005370-E_3…E_6 at 8 m: *shape change*, the lowest q band lost 0.8–1.9 % per hour (≈ 9–20 %
  in 11 h) while high q stayed: aggregation or size-selective sedimentation;
* D_5, D_6 at 8 m: the first run is the *outlier*, the two later runs agree;
* water at 8 m (11 runs): three runs differ by a factor only (scale model p = 0.39), the rest agree.

## Which runs were used: the reduction plan

{func}`~pysasfit.sasred.plan.build_plan` puts it all in one table per configuration (SD, λ, coll):
the role of every group (sample, sample background, water, empty cell, empty beam, blocked beam
(Cd)), its runs, the runs that are summed, the outliers and the check, plus the roles that are
missing. Roles come from the sample names (`H2O`, `EC`, `EB`, `Cd`/`B4C`, ...; change them with
`roles={"EC": "sample_background"}`). Transmission runs are the runs with the beam stop moved out
of the direct beam, in y (on SANS-I typically bsy = −200 mm or lower) or in x, or out of the
detection area: its distance from the beam centre on the detector exceeds the preference
`sasred.transmission.beamstop_out_mm` (100 mm), with or without attenuator. The beam stop motor
positions are tied to detector pixels by the beam stop shadow in the scattering runs, the beam
centre comes from the direct-beam runs of the same distance and wavelength
({func}`~pysasfit.sasred.catalogue.beamstop_geometry`). The run table shows *trans.* or *scatt.*
in the column *kind*, with "?" where the beam stop position and the image disagree (moved out but
no direct beam in the image, or in place but the direct beam is seen); the plan lists the beam
stop geometry per configuration, including how far the beam stop sits off the beam centre.
They are checked by their count rate (0.5 %
allowed for beam stability), scattering runs with {func}`~pysasfit.sasred.grouping.check_runs`.

```python
from pysasfit.sasred import plan

p = plan.build_plan(scan_folder(raw_folder), beam_center_px=centre, edges=edges, mask=mask)
print(p.markdown())             # the table
p.save("reduction_plan.json")   # kept with the reduced data; plan.Plan.load(...) reads it back
```

or on the command line: `python -m pysasfit.sasred.plan RAW --centre X Y --json plan.json`.
Groups of the same sample at different changer positions are marked ("check which is right"),
and a missing Cd run, empty cell or water at a configuration is listed under the table. In the
pySASred window this table is planned as the *Runs* panel.
