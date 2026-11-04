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
| 7, 11 geometry, solid angle | lab-frame position of every pixel relative to the beam centre **of the geometry of the run** (SD, λ; from its direct-beam runs), with the detector tilt α about the vertical axis through the beam spot; 2θ = atan2(√(x²+y²), z); ΔΩ = A·SD·cos α / r³ (exact to first order in the pixel area, checked against brute-force integration). SD is the effective distance (recorded + offset, if set) | – |
| 12 efficiency | optional: divided by the detector efficiency from water (below) | water, empty-cell and transmission sources |
| 14 average | radial or sector average as a sparse linear map over the valid pixels whose centre falls in the bin (no pixel splitting yet) | linear, exact |
| resolution | Qdev² = (q σ_λ/λ)² + beam divergence from the direct beam of the geometry + spread of the pixel q in the bin (below); without a direct-beam measurement the divergence is estimated from the collimation | – |

The solid angle is computed only once the beam centre of the run's geometry is known: q, 2θ and ΔΩ of every pixel
are measured from it. The order relative to steps 3–6 does not matter (all are factors per pixel); it is applied
before the efficiency (step 12, whose water was itself divided by its own ΔΩ with the centre of the water
geometry) and before the average. Each geometry has its own centre: with the Vifor data the 2 m geometry has no
direct-beam run and borrows the 8 m centre; moving it by ±0.5 pixel changes the 8 m / 2 m water ratio by
0.05 % and K(15°) by 0.1 %.

The result is I(q) in counts / monitor / sr (counts / s / sr when normalised by time) on the bin-averaged q.

## Resolution: direct-beam profile, divergence and Qdev

The direct-beam measurements of a geometry are used for three things:
1. the absolute scale (with an efficiency map);
2. the beam centre: the median of the masked centres of gravity, the default centre of the geometry;
3. the **beam profile**, which gives the divergence and with it the resolution.

**Beam profile** ({func}`~pysasfit.sasred.core.beamprofile.beam_profiles`). Every direct-beam run of the geometry
gets a 2D Gaussian with a flat background, integrated over each pixel and fitted with Poisson weights in a window
around the beam (up to 10 runs; the medians are kept):

$$
N_{ij} = A\,\bigl[\Phi\bigl(\tfrac{x_j+\frac12-x_0}{s_x}\bigr)-\Phi\bigl(\tfrac{x_j-\frac12-x_0}{s_x}\bigr)\bigr]
         \bigl[\Phi\bigl(\tfrac{y_i+\frac12-y_0}{s_y}\bigr)-\Phi\bigl(\tfrac{y_i-\frac12-y_0}{s_y}\bigr)\bigr] + B .
$$

- **Which runs:** the empty-beam runs when there are any. A sample in the beam broadens it by its own small-angle
  scattering, so sample transmission runs are used only when there is no empty beam, and the log says so.
- **Pixel size:** because the model is integrated over the pixel, $s_x, s_y$ are the widths of the beam itself,
  without the pixel size.
- **Divergence:** the angular standard deviations seen from the sample are
  $\sigma_{\theta,x} = s_x p_x / \mathrm{SD}$ and $\sigma_{\theta,y} = s_y p_y / \mathrm{SD}$.
- **Non-Gaussian beams:** the second moments of the beam are compared. A difference above 25 % (box-shaped beam,
  tails) is noted.
- **No borrowing:** a geometry without a direct-beam run does not take the profile of another geometry, because
  the beam size depends on coll, SD and the apertures. Its divergence is **estimated from the collimation**
  instead (below).

**Divergence from the collimation (no direct beam yet).** The source aperture a₁ (on SANS-I the neutron-guide
cross-section, 50 × 50 mm²) lies the collimation length coll before the sample aperture a₂ (typically 8 × 14 mm²,
x × y), the detector SD behind it. Each aperture adds a uniform distribution to the beam spot on the detector
(Mildner & Carpenter 1984), so per detector axis

$$
\sigma_\text{spot}^2 = \frac{(a_1\,\mathrm{SD}/\mathrm{coll})^2}{12} + \frac{\bigl(a_2\,(\mathrm{coll}+\mathrm{SD})/\mathrm{coll}\bigr)^2}{12},
\qquad \sigma_\theta = \sigma_\text{spot}/\mathrm{SD}
$$

({func}`~pysasfit.sasred.core.beamprofile.collimation_profile`). coll and SD come from the raw file, the apertures
from **Settings → Preferences → SANS reduction → Resolution** (source 50 × 50 mm², sample 8 × 14 mm², x × y, by
default; they replace the values of the instrument file): the sample aperture actually used is not recorded. The estimate is used like a measured profile; the log, the notes ("estimated from
the collimation (guide 50×50 mm², sample aperture 8×14 mm², …)") and *Missing input* (orange) say so. For Vifor at
coll = SD = 8 m it gives σ = 15.3 × 16.6 mm against 13.6 × 13.6 mm measured: about 15–20 % too large, on the safe
side.

**Calibrated on the direct beams of the same collimation.** When another detector distance with the same
collimation length (and preferably the same wavelength) has a direct beam, the model is calibrated on it first
({func}`~pysasfit.sasred.core.beamprofile.calibrated_profile`) and then evaluated at the distance without a
direct beam. The divergence is not copied: the sample-aperture term grows as 1/SD, so a shorter distance has a
wider angular resolution.
- direct beams at two or more distances (differing by at least a factor 1.5): both effective widths a₁, a₂ are
  fitted per axis (non-negative least squares on σ²);
- one distance: the sample aperture of the preferences is kept and the effective guide width a₁ follows from the
  measured beam;
- empty-beam runs are preferred: a sample in the beam broadens it.

For Vifor (8 m: σ = 13.6 mm) this gives an effective guide of 44 × 37.5 mm² and, at 2 m, σθ = 2.1 × 2.8 mrad
instead of 2.3 × 3.1 mrad from the nominal apertures. The notes and the log say which direct beams were used. Even
so, a short attenuated direct-beam run at every distance is better: it also gives the beam centre and the flux
for the absolute scale.

Without the collimation data (or without apertures in the instrument file, as for SANS-LLB) there is no
resolution.

**1D: Qdev per point** ({func}`~pysasfit.sasred.core.resolution.sigma_q_1d`). With $q = 2k\sin\theta$,
$k = 2\pi/\lambda$, $\mathrm dq/\mathrm d(2\theta) = k\cos\theta$, and $\phi$ the azimuth of a pixel:

$$
\mathrm{Qdev}^2 = (q\,\sigma_\lambda/\lambda)^2
  + \bigl\langle k^2\cos^2\theta\,(\sigma_{\theta,x}^2\cos^2\phi + \sigma_{\theta,y}^2\sin^2\phi)\bigr\rangle_\text{bin}
  + \operatorname{var}(q_\text{pixel})_\text{bin} .
$$

**dλ/λ from the selector tilt (SANS-I).** dλ/λ of the Dornier selector depends only on its tilt angle
(`SANS/Dornier-VS/tilt`, 0 … −10°). The reader takes it from the calibration table in the instrument file
(`wavelength.selector_tilt` in `instruments/sinq/sans_i.yaml`), interpolated monotonically (PCHIP), and not from
`monochromator/wavelength_spread`, which is a fixed 0.1 in the files:

| tilt [°] | 0 | −2 | −4 | −6 | −8 | −10 |
|---|---|---|---|---|---|---|
| dλ/λ (FWHM) | 0.0983 | 0.1081 | 0.1267 | 0.1579 | 0.2062 | 0.2718 |

A tilt outside the table is clipped to its end, with a note. Files without the tilt keep the header value; files
with neither use dλ/λ = 0.1 (FWHM) from **Settings → Preferences → SANS reduction → Resolution** (0: no
resolution), and the notes say so. The
Qdev note and the parameters of every reduced curve give the tilt and where dλ/λ came from; the run list shows
dλ/λ and the tilt in the column *dλ/λ (tilt)*.

**Flux and tilt.** At the same wavelength the flux grows roughly in proportion to dλ/λ (double dλ/λ, about
double the flux; no exact calibration). Runs of one reduction (sample, water, empty cell, direct beam, …) at
different tilts (dλ/λ differing by more than 2 %, i.e. about 0.6° near 0°) are therefore checked:
- with the **monitor normalisation** the flux cancels only if the normalising monitor is **behind the
  selector**: an *info* line in *Missing input* and the log. A monitor in front of the selector sees the white
  beam and does not follow the selector setting: a *check* line (orange), as when the position is unknown;
- with the **time normalisation** the flux is not corrected: a *check* line – use runs of the same tilt. No
  approximate factor is applied.

**Before or after the selector?** A monitor in front of the selector keeps its ratio to the proton beam
(`integrated_beam`) at every wavelength and tilt; one behind it does not. When a folder is read, pySASred
compares monitor / proton beam between the settings (λ, tilt) of its runs
({func}`~pysasfit.sasred.core.monitors.selector_test`): constant within max(3 × scatter, 5 %) → in front,
changing by more than max(5 × scatter, 15 %) → behind; the log reports the normalising monitor. It needs runs at
two settings (the Vifor folder, 6 Å only, cannot decide); every frame of a selector speed scan counts as a
setting of its own. The instrument file states it (`monitor.normaliser_after_selector`; null = take it from the
data).

**SANS-I monitors (AgBe 2025, runs 1964–1980: 8.3, 11.7, 16.4, 50 and 57.5 Å, and the speed scans 1976 and
1980, 18 frames of 600 s at 11000–28000 rpm):**

| counter | ratio to the proton beam | position |
|---|---|---|
| monitor1 (= monitor4) | changes by a factor 3–15 with λ | **behind the selector** – the normaliser |
| monitor2 | 0.0086, constant within 0.7 % | in front of the selector |
| monitor_7 | 0.0655, constant within 0.6 % | in front of the selector |
| monitor_5 | identical to `integrated_beam` | proton beam |

So the normalisation by monitor1 follows any change of the selector (λ, tilt, speed); monitor2 – used by
BerSANS in 2024 and by DarePy – does not, it only follows the proton current.

**Multi-frame runs.** Counts, measuring times and monitors of the frames are summed (the per-frame values are
kept in `RawRun.frames`). A run whose frames have different selector speeds is a speed scan with a different λ
in every frame (λ ∝ 1/speed; the stored λ is that of the last frame): it is listed as skipped
(*selector speed scan … – λ differs between the frames*) and not reduced.
- Sample runs of one row at different tilts are summed; dλ/λ of the first run is used (*check*).

The beam centre, the beam profile (divergence) and the efficiency map do not depend on the tilt and are shared
by all tilts of a geometry (coll, SD, λ).

The triangular wavelength distribution has $\sigma_\lambda/\lambda = \mathrm{FWHM}/\sqrt6$. The divergence term
is averaged over the pixels of the bin, i.e. in the radial direction of each pixel; the last term contains the
pixel size.

**2D: a Gaussian per pixel** ({func}`~pysasfit.sasred.core.resolution.resolution_2d`). In the detector axes the
covariance of $(q_x, q_y)$ is

$$
\Sigma = k^2\cos^2\theta\,\operatorname{diag}\bigl(\sigma_{\theta,x}^2 + \tfrac{(p_x/\mathrm{SD})^2}{12},\;
          \sigma_{\theta,y}^2 + \tfrac{(p_y/\mathrm{SD})^2}{12}\bigr)
       + (q\,\sigma_\lambda/\lambda)^2\,\hat q\hat q^{\mathsf T} .
$$

The divergence and the pixel width act along the detector axes; the wavelength term acts along $\hat q$.

**Files:**
- **2D NXcanSAS:** Qx and Qy carry `resolutions` = `Qxdev` / `Qydev`. Non-standard extras: `Qdev_parallel`,
  `Qdev_perpendicular` (along and across q) and `Qxy_correlation`.
- **1D NXcanSAS:** `Qdev`.
- **Text files:** the fourth column.

**Without a direct-beam measurement and without the collimation estimate** (or in a script without a profile)
the reduction runs without resolution:
- the notes, the log and the panel *Missing input* (row Direct beam, "resolution") say so;
- NXcanSAS files have no Qdev (and Q no `resolutions`);
- text files have Qdev = 0, with a header line saying the resolution was not determined, because readers take the
  fourth column as the resolution.

The same happens without dλ/λ in the raw file.

**Vifor, 8 m, λ = 0.6 nm, tilt 0°, dλ/λ = 9.8 % FWHM.** The profile comes from the empty beam 82173, 82221, 82269:
- the beam is σ = 13.6 mm in x and y, i.e. a divergence σθ = 1.69 mrad;
- the second moments give 16.8 mm (the beam is somewhat box-shaped).

Qdev/q is 27 % at q = 0.0066 Å⁻¹, where the divergence dominates, and 4.7 % at q = 0.084 Å⁻¹, where the wavelength
spread (4.1 %) dominates. The 2 m geometry has no direct-beam run: its divergence is estimated from the
collimation calibrated on the 8 m direct beam (σθ = 2.1 × 2.8 mrad).

The formulas use the small-angle form of the divergence term along the detector axes. For the SANS-LLB wide-angle
panels the divergence measured on the central detector is applied as an angle.

The detector tilt is read from the file (`chi_position`) but not yet applied, until it is confirmed that this
field is the tilt; the notes say so. The dead time of the monitor is 0 for now (a factor 1.0252 seen in a BerSANS
copy suggests about 3.3 μs, to be confirmed).

## Reduction of the table: 2D first, then the average

The rows of the reduction table are reduced pixel by pixel first ({func}`pysasfit.sasred.absolute.reduce_row_2d`),
the radial or sector average comes afterwards ({func}`pysasfit.sasred.absolute.average`). Every run is
corrected as above (counting error, dead time, attenuator, monitor), the runs of a row are summed, then

```{math}
N_{ij} = A\,\frac{S_{ij}-Cd_{ij}}{T_S} - (1-p_{cS})\,\frac{SB_{ij}-Cd_{ij}}{T_{SB}}, \qquad
W_{ij} = \frac{W_{ij}-Cd_{ij}}{T_W} - \frac{WB_{ij}-Cd_{ij}}{T_{WB}}
```

```{math}
\frac{d\Sigma}{d\Omega}_{ij} = \frac{N_{ij}}{W_{ij}}\;g(\lambda)\,\frac{1-\tau_W}{4\pi\,D\,d_S}\,\frac{A_W}{A_S}
\quad [\mathrm{cm^{-1}\,sr^{-1}}], \qquad \tau_W = T_W / T_{WB}, \qquad D = T_W \ \text{(BerSANS, SANS-I)}\ \text{or}\ \tau_W\ \text{(Strunz)}
```

(Strunz et al. 2000, eq. 8, with the water at the same geometry, so that the solid angle and the pixel
efficiency cancel). T_W is the transmission of water + cell, T_WB that of the empty cell, both against the
empty beam. Every transmission and count is an uncertainty source of the result.

**Different apertures for sample and water.** A_S and A_W are the illuminated areas (sample apertures) in cm²,
column *Aperture [cm²]* of the reduction table (Sample rows and the Water row; the raw files do not record the
sample aperture). With the same aperture leave both empty: the ratio is 1. If only one is given, it is ignored
and the notes say so.

**Relation to BerSANS.** BerSANS writes the factor as ScalingFactor(W)/ScalingFactor(S) with

| | same aperture | different apertures |
|---|---|---|
| ScalingFactor(S) | d_S | d_S · A_S |
| ScalingFactor(W) | g(λ) (1 − τ_W)/(4π T_W) | g(λ) · A_W (1 − τ_W)/(4π T_W) |

BerSANS' *Calculate* button gives ScalingFactor(W) without g(λ) (the user multiplies it in): T_W 0.5,
T_EC 0.95, A_W 1 cm² → 7.5389·10⁻². The notes of every reduced row give both scaling factors, so they can be
compared with a BerSANS command file ({func}`pysasfit.sasred.absolute.water_scaling_factor`).

BerSANS divides by T_W (water + cell), Strunz et al. eq. 8 by τ_W = T_W/T_EC (water alone); the two differ by
the factor 1/T_EC (≈ 5 % for T_EC = 0.95). The instrument file chooses (`water.denominator`):
`water_and_cell` (BerSANS, the SANS-I setting, consistent with its g(λ)) or `tau` (Strunz). Since the water counts are divided by T_W (water + cell),
the water scatters (1 − τ_W) of the beam into 4π, attenuated by the cell on the way out, the consistent
denominator is τ_W; a g(λ) calibrated with BerSANS, however, contains the factor T_EC of its calibration
cells, and then `water_and_cell` reproduces that calibration.

K0 is the forward excess g(λ) of the water measured at SANS-I (instrument file `water.g_lambda`, λ in nm):

| λ [nm] | 0.47 | 0.60 | 0.80 | 1.00 | 1.20 |
|---|---|---|---|---|---|
| measured | 1.338 | 1.284 | 1.217 | 1.150 | 1.141 |
| g = 1.36992 + 0.31389 λ − 1.07706 λ² + 0.54662 λ³ (in use) | 1.336 | 1.289 | 1.212 | 1.153 | 1.140 |
| g = 1.642 − 0.7825 λ + 0.3159 λ² (previous) | 1.344 | 1.286 | 1.218 | 1.175 | 1.158 |

The cubic is the least-squares polynomial through the five points (residuals ≤ 0.5 %, rms 0.3 %); it is applied
({func}`pysasfit.sasred.absolute.water_g`) with a relative uncertainty of 0.6 % (the fit uncertainty inside
0.47–1.2 nm) as a systematic source `water_g`. With four coefficients for five points it has one degree of
freedom: outside 0.47–1.2 nm it is unreliable (it turns up above 1.2 nm, g(1.3 nm) = 1.159 ± 0.021), the
notes say "extrapolated". The previous quadratic did not pass the points at 1.0 and 1.2 nm (+2.2 % and +1.5 %).
`reduce_row_2d(..., k0=…)` uses another value.
Missing measurements do not stop the reduction, the notes say what was left out. Cd, water background and
sample background are optional: they may have no runs or be switched off in the table (column *Use*,
`ReductionTable.ignored`).

| missing (or switched off) | then |
|---|---|
| Cd (blocked beam) | nothing subtracted for the beam-independent background |
| sample background | no background subtracted |
| water background (empty cell) | the water is not corrected for its cell, τ_W = T_W |
| water | relative scale: N / (ΔΩ · d) [· efficiency], in counts/monitor/sr/cm |
| transmission | T = 1 |
| thickness | d = 1 cm |
| direct beam or efficiency map (direct-beam scale) | the row is not reduced (error) |

{meth}`~pysasfit.sasred.reduction_table.ReductionTable.missing_inputs` lists these cases per row (stored in the
result as `parameters["missing"]`); the window colours the cells and lists them in the panel *Missing input*.

**Which absolute scale** ({meth}`~pysasfit.sasred.reduction_table.ReductionTable.scale_method`): the table's
`scale` is *auto* (default), *water*, *direct_beam* or *relative*. *auto* means the direct beam as soon as an
efficiency map is chosen for the geometry – its own water is then not used, even when measured –, else water
when the Water row has runs, else relative.

**Runs without data** – measuring time 0, 0 counts (all detector banks) or the normalising monitor at 0
(`RawRun.empty_reason`) – are skipped everywhere: marked in the run list (column *comment*), left out of the
plan (`Plan.skipped`), and a reduction given one leaves it out with a note ("run … skipped: …").

In the window: **Reduce all samples (2D → I(q))** or **Reduce selected samples** in a geometry tab of the
reduction table (or the *Reduction* menu). The mask is the edited mask when the geometry is the one shown in
the detector panel, otherwise the suggested mask of the first sample run; the beam centre is that of the
geometry; q bins, sector and normalisation come from the Reduction panel. The results appear in I(q) as
*"SD 8.0 m No. 3"* etc., *view → reduced 2D* in the detector panel shows the 2D image of the result chosen,
and they are saved like single runs. With the Vifor data (water and empty cell, no Cd) A_3 and D_3 at 8 m and
2 m agree within 1 % in the q range both cover, and water reduced as a sample gives the flat
g(λ) (1 − τ_W)/(4π T_W d) = 1.289 × 1.208 = 1.557 cm⁻¹ sr⁻¹ for d = 0.2 cm at 0.6 nm (T_W = 0.2399,
T_EC = 0.8835; with the Strunz denominator τ_W: 1.375 cm⁻¹ sr⁻¹, lower by T_EC).

The reduced curves of the geometries are merged in pyTOOLbox (*Merge geometries*, {doc}`../toolbox/merge`):
*Reduction → Send results to pyTOOLbox merge (no files)* hands them over without saving.

## Absolute scale from the direct beam and the attenuator

Instead of water, the absolute scale can come from the empty (direct) beam, measured with an attenuator
of known transmission T_att: Φ₀ = Σ(EB − bkg)/T_att, and dΣ/dΩ = N/(Φ₀ ΔΩ η d_S). The nominal
transmissions in the instrument file (`attenuator`) are calculated from the hole diameters of the sieves;
they hold only for a homogeneously illuminated sieve.

Test with the Vifor data at 0.6 nm (water at 8 m and 2 m against the attenuated empty beam 82173, 82221,
82269, attenuator 1, efficiency at the direct-beam spot from the water map, +53 mm distance offset):

| | T_att of attenuator 1 |
|---|---|
| nominal (sieve holes) | 2.06·10⁻³ = 1/485 |
| from water, BerSANS denominator (in use) | 1.64·10⁻³ ± 2 % ≈ 1/610 |
| from water, Strunz denominator | 1.45·10⁻³ ≈ 1/690 |

8 m and 2 m water agree within 1.5 % (without the distance offset by 5.6 % only). **An absolute scale
from the direct beam with the nominal attenuator transmission is therefore about 26 % lower than the
one from water.** The water calibration is trusted for now (water measured against a standard); the
attenuator is to be reconfirmed by tests (e.g. the inhomogeneous illumination of the sieve). The value is
kept in the instrument file as `attenuator."1".water_check`; the notes of every run with that attenuator
mention the difference. Transmissions are not affected (the attenuator is in both runs).

### Direct beam + efficiency map instead of water

At long distances and long wavelengths water takes too long. Such a geometry is then put on the absolute scale
with its **direct beam** (beam stop out, usually attenuated: the empty-beam transmission runs of the same
geometry, filled into the row *Direct beam* of the reduction table) and a **detector efficiency map** ε (mean 1),
usually made from the water of a shorter distance:

    Φ = Σ_disc (DB − Cd)_kl / ε_kl                          counts/monitor, attenuator corrected
    dΣ/dΩ_ij = N_ij / (ε_ij ΔΩ_ij d_S Φ) · a_DB / a_S         1/cm/sr

N is the sample minus its background as for water. The absolute efficiency of the detector and its
λ dependence cancel between N and Φ; only the relative pixel efficiency comes from the map.

* **Disc**: centred on the centre of gravity of the direct beam; its radius by default holds 99.5 % of the counts
  within the largest disc on the detector, plus two pixels (the notes give it); it can be set in mm.
* **Attenuator**: its transmission is the water calibration of the instrument file (`water_check`) when that was
  made within 5 % of the wavelength (its relative uncertainty is propagated as the systematic source
  `attenuator_db:<n>`), else the nominal value (uncertainty usually unknown – then the absolute scale is only as
  good as that factor), or a value typed in the table.
* **Map**: pixels masked or rejected in the water of the map (its beam stop) are estimated as row × column; the
  table can mask them in the sample instead. The map's uncertainty per pixel is propagated (`flood:…`).
* A map measured at another wavelength is used as it is (note in the result); a map of another shape is refused.
* SANS-LLB wide-angle panels: not yet (they would need the efficiency of the panels relative to the central
  detector).

Test with the Vifor data at 0.6 nm, the water at 8 m reduced as a sample: with the 8 m water, and with the 8 m
direct beam (82173, 82221, 82269, attenuator 1 with its water calibration) and the efficiency map of the 2 m
water. Both agree within 1–3 % over q (median +1.4 %); only the first q bin, next to the beam stop where the
2 m map is estimated, is 8 % low (masking the estimated pixels removes that bin). With the 8 m map the two agree
to 1.1 % at all q – within the 2 % of the attenuator calibration. With the nominal attenuator factor the direct
beam would give 27 % more.

## The SANS-I detector

Construction (instrument file, `detector.construction`; {mod}`pysasfit.sasred.core.detector_model`): Al dome
window 7 mm thick, Ø 1300 mm, radius of curvature 930 mm (cap depth 26.5 cm); pure ⁴He cover gas at 2.7 bar in
the dome; a flat 2 mm Al membrane; the detection gas at the same total pressure of 2.7 bar, ³He with CF₄ as
stopping gas. The CF₄ partial pressure is not known: a conventional mixture of about 25 % CF₄ (0.7 bar, ³He
2.0 bar) is assumed. Still to confirm: active depth, ⁴He path between window and membrane, the CF₄ fraction and
the plane the recorded SD refers to. The dome apex lies
26.5 cm in front of the detection plane (confirmed), i.e. the rim of the dome is in the detection plane.

| λ [Å] | Σ(³He) [cm⁻¹] | 1/Σ [cm] | T Al 7 mm (+ 2 mm) | efficiency, depth 2 / 3 / 5 cm | ⟨z⟩, depth 3 cm [cm] |
|---|---|---|---|---|---|
| 4.7 | 0.69 | 1.45 | 0.970 (0.962) | 0.72 / 0.84 / 0.93 | 1.02 |
| 6 | 0.88 | 1.14 | 0.964 (0.953) | 0.79 / 0.89 / 0.94 | 0.91 |
| 8 | 1.17 | 0.85 | 0.953 (0.940) | 0.85 / 0.91 / 0.94 | 0.76 |
| 12 | 1.76 | 0.57 | 0.933 (0.914) | 0.89 / 0.91 / 0.91 | 0.55 |

(³He at 2.0 bar, the assumed partial pressure; with pure ³He at 2.7 bar the efficiency at a depth of 3 cm would be
0.90 / 0.93 / 0.93 / 0.91 and ⟨z⟩ 0.88 / 0.75 / 0.61 / 0.42 cm. Al above the Bragg cut-off 4.68 Å: absorption,
incoherent and about 0.1 b inelastic; the membrane taken as Al.)

The reduction does not depend on the CF₄ fraction: only the Al dome and membrane transmission enter it (water
comparison at two distances), and the ³He efficiency at one wavelength is removed by the water / efficiency map.
It matters only for the numbers below (λ dependence of the efficiency, conversion depth ⟨z⟩, parallax).

Consequences:

* **Dome:** the dome is convex towards the sample (overpressure inside), its centre of curvature behind the
  window, so a neutron crosses it at γ = α + 2θ, α the inclination of the dome normal where it crosses
  (sin α = h/R). On the axis γ = 0; at 0.24 / 0.48 m from the axis γ = 20° / 42° at 2 m and 16° / 34° at 8 m,
  and the Al transmission at 6 Å is lower by 0.24 % / 1.3 % at 2 m and 0.15 % / 0.8 % at 8 m. Water at the
  same geometry cancels it pixel by pixel (absolute scale, efficiency map at the same distance); an efficiency
  map used at another distance does not (up to about 0.5 %).
* **Water at two distances:** the same pixel sees the dome at different γ at 8 m and 2 m.
  `compare_water_runs(..., dome=True)` divides it (and the flat membrane, −0.03 % at the 2 m edge) out first: for Vifor (with the +53 mm offset) the water
  anisotropy becomes K(10°) = 0.993 and K(15°) = 0.984 instead of 0.988 and 0.974, C = 1.0016 (offset 5.5 cm
  without the offset applied); about a third of the drop attributed to the water was the dome.
* **³He at an angle:** the ³He absorbs more along the longer path (+0.9 % at 17°, depth 3 cm; +0.4 % for pure ³He at 2.7 bar); the flat Al membrane
  loses 0.06 % there.
* **Parallax:** neutrons at ψ = 15° are detected about 0.25 cm farther out than their entrance point (0.3 pixel).
* **Distance:** the neutrons are detected on average ⟨z⟩ ≈ 0.9 cm (6 Å; 0.75 cm for pure ³He) behind the start of the active volume.
  This explains less than 1 cm of the +5.3 cm distance offset found with water at 8 m and 2 m; the rest has to
  come from where SD is measured (e.g. to the dome window, followed by the ⁴He layer and the membrane). ⟨z⟩
  changes with λ (1.0 cm at 4.7 Å, 0.55 cm at 12 Å), so water at two distances at another wavelength tells
  the constant part from the conversion depth.
* **Wavelength:** the efficiency falls from about 0.93 at 6 Å by the window absorption at long λ and rises with
  the ³He absorption: with water at the same λ the λ dependence cancels for the absolute scale.
* **⁴He:** the pure He cover gas at 2.7 bar scatters about 0.09 % per 10 cm of path, almost isotropically (a flat background).

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

In the window the map is made from the reduction-table rows of a geometry (Water with T_W, Water-BG with T_EC,
Cd; rows switched off with *Use* are left out), with the mask and beam centre of that geometry
({class}`~pysasfit.sasred.efficiency_gui.EfficiencyMixin`): *efficiency map → Make from the water of* in the
detector panel, or *efficiency map → water of …* in the *Absolute scale* box of the geometry that uses it. The
log states the formula and the runs and transmissions that went in.

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
`/2026-10-01_SD-8.0m_6.00A_run82280`, with

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
log.archive_efficiency(eff, comment="Vifor, 7 water runs")   # → '2026-10-01_SD-8.0m_6.00A_run82280'
log.list_maps()                                              # summaries, oldest first
eff_old = log.load_map("2026-09-30_SD-8.0m_6.00A_run82187")  # default: the newest map
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
