# Reduction steps and uncertainties

## Exact first-order error propagation

Every quantity in pySASred is a `UArray` ({mod}`pysasfit.sasred.data.uarray`): a value array together with its
sensitivity (Jacobian) to each **named, independent** uncertainty source,

$$
y = \text{value},\qquad \mathrm{d}y = \sum_s J_s\,\mathrm{d}s_s,\qquad
\operatorname{Cov}(y) = \sum_s J_s\,\operatorname{diag}(\operatorname{var}_s)\,J_s^{\mathsf T}.
$$

A source is a vector of independent elements: the counts of every pixel of a run (`counting:82130`), a monitor
count (`monitor1:82130`), the detector dead time, an attenuator factor. Each Jacobian is stored as a sparse part
plus a low-rank part, $J_s = S_s + U_s V_s^{\mathsf T}$: pixel-wise steps keep $S$ diagonal, the average makes it
block-sparse, and a correction that depends on an aggregate of the same counts (the dead-time factor depends on
the total count rate) adds the exact rank-1 coupling instead of a dense N × N matrix. Sensitivities to the same
source add, so a source that enters twice (the same monitor in numerator and denominator, later the Cd run in
sample and empty cell) is never counted twice.

The propagation is exact to first order; the only approximation is the linearisation. A test resamples the real
counts and monitor of run 82130 3000 times (Poisson) and compares the sample covariance of I(q) with the
propagated one, including the correlations between q points caused by the dead time and the monitor.

Each source is **statistical** or **systematic**:

* statistical sources (the counts) give **Idev** and the statistical covariance between the q points;
* systematic sources (monitor, dead time, attenuator, later transmission and calibration) are common scale
  factors of a whole curve; they are kept as separate sensitivities, not added to Idev.

A value whose uncertainty is not known yet (`uncertainty: null` in the instrument file) is used but not
propagated, and the reduction notes say so.

## The steps of one run

{func}`pysasfit.sasred.reduce.reduce_run_1d` applies the steps below (numbers as in the package definition).
Instrument constants come from `pysasfit/sasred/instruments/sinq/sans_i.yaml`.

| step | what | uncertainty |
|---|---|---|
| 1 read | SANS-I HDF file; units from the `units` attributes (λ in nm → Å, lengths in mm → m); frames summed; a missing required field is an error; SHA-256 of the file | – |
| 3 counting | σ² = N for every pixel (exact Poisson variance, the right input because variances add). The exact Garwood interval is available for display ({func}`~pysasfit.sasred.core.counting.garwood_interval`) | source `counting:<run>` |
| 2 mask | the mask image ({doc}`masks`); pixels with a dead-time factor above the flag limit get bit 8 | – |
| 4 dead time | non-paralysable, N·f with f = 1/(1 − τR), R = total counts / time (global for the multiwire detector, per tube for tube detectors); τ = 6.6·10⁻⁷ s | rank-1 coupling of all pixels; τ when its uncertainty is known |
| 5 attenuator | divided by the transmission of the selected attenuator | when known |
| 6 normalisation | divided by monitor 1 (Poisson σ² = M) or by the counting time | `monitor1:<run>`, systematic |
| 7, 11 geometry, solid angle | lab-frame position of every pixel with the detector tilt α about the vertical axis through the beam spot; 2θ = atan2(√(x²+y²), z); ΔΩ = A·L2·cos α / r³ (exact to first order in the pixel area, checked against brute-force integration) | – |
| 12 efficiency | optional: divided by the detector efficiency from water (below) | water, empty-cell and transmission sources |
| 14 average | radial or sector average as a sparse linear map over the valid pixels whose centre falls in the bin (no pixel splitting yet) | linear, exact |
| resolution | Qdev² = (q σ_λ/λ)² + spread of the pixel q in the bin; triangular wavelength distribution, σ = FWHM/√6 | – |

The result is I(q) in counts / monitor / sr (counts / s / sr when normalised by time) on the bin-averaged q.

The detector tilt is read from the file (`chi_position`) but not yet applied, until it is confirmed that this
field is the tilt; the notes say so. The dead time of the monitor is 0 for now (a factor 1.0252 seen in a BerSANS
copy suggests about 3.3 μs, to be confirmed).

## Detector efficiency from water

The efficiency of every pixel comes from water, which scatters almost isotropically
({func}`pysasfit.sasred.reduce.efficiency_from_runs`, {mod}`pysasfit.sasred.core.efficiency`). Water, empty cell
and (optionally) blocked beam are each corrected as in the steps above (dead time, attenuator, monitor) and
combined as

$$
I_\text{water} = \frac{(W - B) - \dfrac{T_w}{T_{ec}}\,(EC - B)}{T_w\,\Delta\Omega},
\qquad \varepsilon = \frac{I_\text{water}}{\langle I_\text{water}\rangle_\text{valid}} .
$$

Several runs of the same sample (the repeated water and empty-cell runs of a beam time) are summed: corrected
counts over summed monitor. The transmissions come from the transmission table of the beam time
({func}`~pysasfit.sasred.io.transmission.read_transmission_summary`, `trans.txt`).

**Masked pixels.** Behind the beamstop (and at hot pixels) the water gives no efficiency. The SANS-I detector is a
multiwire detector read out in coincidence: a pixel is the crossing of an x wire and a y wire, and its efficiency
is close to the product of a row and a column factor. A masked pixel (i, j) therefore gets

$$
\varepsilon_{ij} = \frac{R_i\,C_j}{m},
$$

R_i the mean efficiency of the valid pixels of its row (y wire), C_j that of its column (x wire), m the mean of all
valid pixels. A pixel whose whole row or column is masked (the standard border) has no estimate: its efficiency is
1 and it gets mask bit 32 (*efficiency*). On the Vifor water runs the row × column model reproduces the measured
pixels to the counting noise at 8 m; at 2 m (40 million counts) a pixel-level part of about 3 % remains that the
model cannot describe, so filled pixels carry that much more uncertainty than their propagated error says.

**Beamstop edge.** Pixels in the half shadow of the beamstop are not always in the mask of the water run, and their
low apparent efficiency would be applied to the sample. Valid pixels within 3 pixels of the masked area (not the
border) that differ from the row × column estimate by more than 5 statistical standard deviations are replaced by
the estimate too (*rejected*; `reject_sigma`, None switches it off). Elsewhere a deviating pixel is kept, since it
may be a real local defect. On the Vifor data this replaces 5 to 21 pixels at the corners of the beamstop.

**Uncertainty.** The efficiency is a `UArray`: it keeps the sensitivities to the counts and monitors of every water
and empty-cell run and to both transmissions. Dividing the sample image by it
(`reduce_run_1d(..., efficiency=eff)`) propagates them exactly, including the correlations between q points from
the normalisation and from the filled pixels (each depends on a whole row and column). A filled pixel has a much
smaller statistical error than a measured one, because it averages a row and a column. The water noise matters:
for the sample run 82140, one 1000 s water run at 8 m (≈ 7.6 % per pixel) raises Idev by about 20 % (up to a
factor 2.7 in the lowest q bin); the seven summed 8 m water runs (≈ 2.7 % per pixel) by about 1 % (up to 24 % in the
lowest bin).

**Files.** {meth}`~pysasfit.sasred.core.efficiency.Efficiency.save` writes an HDF5 file (`/entry/efficiency` as
NXdata, readable in pyTOOLbox, with the filled, undefined and rejected pixels and the provenance). Loaded again,
the map is one independent source per pixel (`flood:<file name>`): the correlations between pixels are dropped.

```python
from pysasfit.sasred.io.sinq import read_sans_i
from pysasfit.sasred.io.transmission import read_transmission_summary, find_transmission
from pysasfit.sasred.reduce import efficiency_from_runs, reduce_run_1d

raw_dir = "/data/sans-i/2026/Vifor"
run = lambda n: read_sans_i(f"{raw_dir}/sans2026n{n:06d}.hdf")
tt = read_transmission_summary(f"{raw_dir}/trans.txt")
tw, te = find_transmission(tt, "H2O"), find_transmission(tt, "EC")
eff = efficiency_from_runs([run(n) for n in range(82280, 82293, 2)],      # 7 water runs, 8 m
                           [run(n) for n in range(82281, 82294, 2)],      # 7 empty-cell runs
                           centre, (tw.value, tw.sigma), (te.value, te.sigma))
eff.save("efficiency_8m.h5")
r = reduce_run_1d(run(82140), centre, edges, mask=mask, efficiency=eff)
```

### Archive: the detector over time

To see whether something happens with the detector, every efficiency map can be logged in **one HDF5 archive
file** ({mod}`pysasfit.sasred.efficiency_log`). Its location is *Preferences → SANS reduction → efficiency archive
file*; left empty it is `efficiency_archive.h5` next to the preferences file, and the dialog shows that full path.

Each map is one NeXus entry named after the date of its water measurement, e.g.
`/2026-10-01_L2-8.0m_6.00A_run82280`, with

* `efficiency`: the map, its errors, the filled, undefined and rejected pixels, and the row and column profiles
  (mean efficiency of every y and x wire);
* `instrument`: the configuration of the water measurement: detector distance, **tilt** and offset, pixel size,
  beam centre, **wavelength** and its spread, collimation length, attenuator (NeXus fields with units);
* `process`: water, empty-cell and blocked-beam runs, transmissions, comment, all metadata;
* attributes with the summary (date, distance, wavelength, tilt, number of filled and rejected pixels, rms), so the
  archive is listed without reading the maps.

The file attribute `default` points to the newest map, so pyTOOLbox shows it first. Adding a map never changes
the maps already in the archive.

```python
from pysasfit.sasred import efficiency_log as log
log.archive_efficiency(eff, comment="Vifor, 7 water runs")   # → '2026-10-01_L2-8.0m_6.00A_run82280'
log.list_maps()                                              # summaries, oldest first
eff_old = log.load_map("2026-09-30_L2-8.0m_6.00A_run82187")  # default: the newest map
log.history()                         # every map against the oldest of its distance, wavelength and tilt
log.plot_history().savefig("efficiency_history.png")
```

{func}`~pysasfit.sasred.efficiency_log.compare` gives, on the pixels measured in both maps, the rms and mean
change, the change **expected from the errors alone** and χ² (≈ 1 when only counting noise differs), the pixels
changed by more than 4 σ, and the **wires** (rows = y wires, columns = x wires) whose profile changed by more than
4 standard errors, which is what a failing wire or amplifier channel shows. By default every map is compared with the oldest map of the
same configuration (distance, wavelength, tilt): the Vifor maps at 2 m and 8 m differ in the outer columns by more
than the noise, from the geometry corrections and beam centre, not from the detector. From the command line:

```bash
python -m pysasfit.sasred.efficiency_log [archive.h5] --plot history.png
```

## Checks

* I(q) recomputed independently with plain numpy agrees to 10⁻¹² (relative).
* Against pyFAI `integrate1d` on the same corrected image without pixel splitting: 2·10⁻⁷; with split pixels the
  differences are confined to the lowest q bins next to the beamstop.
* The beam centre of the direct-beam run 82173 agrees with the BerSANS script of the beam time to 0.1 pixel.
* Propagated covariance against 3000 Poisson resamplings of the real data (variances within 10 %, correlations
  within 0.07, and the median correlation caused by dead time and monitor reproduced).
