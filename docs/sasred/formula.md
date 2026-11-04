# The cross-section formula and its simplifications

pySASred follows the general formula of Strunz, Šaroun, Keiderling, Wiedenmann & Przenioslo,
*General formula for determination of cross-section from measured SANS intensities*,
J. Appl. Cryst. **33** (2000) 829–833. BerSANS (Keiderling, HMI Berlin) is based on the same
derivation but makes simplifications. pySASred keeps the general form and records every
simplification it applies in the notes of the result.

## The general formula

For pixel ij, with all intensities already corrected for dead time and normalised to the monitor
(eq. 8 of the paper):

```{math}
\left[\frac{d\Sigma}{d\Omega}(Q_{ij})\right]_S =
\frac{\dfrac{I_{S}-I_{Cd}}{T_{EB}\,T_{Si}} - \dfrac{I_{EB}-I_{Cd}}{T_{EB}}\,(1-p_{cS})}
     {\dfrac{I_{H2O}-I_{Cd,H2O}}{T_{EC}\,\tau_{H2O}} - \dfrac{I_{EC}-I_{Cd,H2O}}{T_{EC}}}
\; K_0\,\frac{t_{H2O}-\tau_{H2O}}{4\pi\,\tau_{H2O}\,d_S}\,
\frac{\Delta\Omega_{ij,H2O}}{\Delta\Omega_{ij}}\, r
\;-\; \frac{1}{1-p_{iS}}\left[\frac{d\Sigma}{d\Omega}\right]_{INC}
```

| symbol | meaning |
|---|---|
| t_S | transmission for absorption and large-angle coherent (Bragg) scattering |
| T_Si = t_S (1 − p_iS) | absorption and incoherent scattering: the **wide-angle** transmission (plateau of T(α)) |
| τ_S = T_Si (1 − p_cS) | total removal transmission: the transmission through the **beam spot** only |
| p_cS = d_S ∫ dΣ/dΩ dΩ | probability of at least one coherent small-angle scattering event |
| T_EB, T_EC | transmission of the substrate / container / sample environment, of the empty water cell |
| K0 | forward excess of water in a flat cell (≈ 1.4 at 6 Å for 1 mm, mostly multiple scattering) |
| r = I_prim,H2O / I_prim | primary beam with the water collimation / with the sample collimation |
| ΔΩ_ij,H2O / ΔΩ_ij | pixel solid angles at the water and the sample distance |

Two transmissions with different meaning appear: the sample is divided by T_Si, the background
is multiplied by (1 − p_cS). Only for p_cS ≈ 0 is T_Si = τ_S and the familiar formula recovered;
with τ_S in the sample term (eq. 9, aqueous samples) the whole curve is shifted by (1 − p_cS).
The formula is an integral equation for the cross-section, because p_cS is its integral.

## BerSANS

BerSANS (manual: program SANSAniDo, eq. 2) computes

```{math}
\frac{d\sigma}{d\Omega}_{ij} =
\frac{\dfrac{S_{ij}-Cd_{ij}}{T(S)}\,A(S) - \dfrac{SB_{ij}-Cd_{ij}}{T(SB)}\,(1-p(S))}
     {\dfrac{W_{ij}-Cd_{ij}}{T(W)} - \dfrac{WB_{ij}-Cd_{ij}}{T(WB)}}
\;\frac{\text{ScalingFactor}(W)}{\text{ScalingFactor}(S)}
```

with the transmissions T = Σ(S − Cd)/Σ(EH − Cd) over a selectable detector rectangle, an attenuation
factor A(S) and a scattering probability p(S) given by the user (defaults 1 and 0), and scaling
factors that hold the illuminated volumes (thickness × aperture area) and, in the tutorial, a
geometry factor col² · sd² for water measured at another distance. The water scaling factor of its
*Calculate* button is A_W (1 − T_W/T_WB)/(4π T_W) (times g(λ) by hand): the denominator is the
transmission of water + cell T_W, where Strunz et al. have τ_H2O = T_W/T_EC (see {doc}`reduction`,
instrument file `water.denominator`).

## Simplifications, one by one

| item | Strunz et al. | BerSANS | pySASred (✓ done, ○ planned) |
|---|---|---|---|
| meaning of the sample transmission | T_Si (wide angle) in eq. 8, τ_S in eq. 9 | one T(S), whose meaning depends on the detector area chosen | ✓ both from the same run ({func}`~pysasfit.sasred.core.scattering_strength.aperture_transmission`): T(α), plateau T_Si, beam spot τ_UL; ○ the mode stored with the value |
| (1 − p_cS) on the background | required | p(S), default 0 | ✓ p_cS from the transmission (lower limit), from the curve (with extrapolation) or self-consistent, regime ({func}`~pysasfit.sasred.core.scattering_strength.assess`); ○ used in the reduction |
| container / substrate | T_EB in both terms | in T(S) when it is measured against the empty beam, not against the empty cell | ○ T_EB kept separate |
| Cd background | separate Cd for the water geometry | one Cadmium | ○ separate when the water was measured at another collimation or distance |
| water standard | K0 (t − τ)/(4π τ d) gives the absolute scale | ScalingFactor(W) = g A_W (1 − τ)/(4π T_W): denominator water + cell; volume ratio in the scaling factors | ✓ K0 = g(λ) with its uncertainty as a systematic source; denominator T_W (BerSANS, SANS-I default) or τ_W (`water.denominator`); illuminated areas A_W/A_S (✓ efficiency from water) |
| water at another distance | ΔΩ_ij,H2O/ΔΩ_ij per pixel and r | col² · sd² (on-axis, fixed apertures) | ✓ exact pixel solid angles; ○ r from the collimation, with an uncertainty |
| K(λ0, 2θ) | constant for 2θ ≤ 15° | — | ○ same assumption, stated in the notes |
| λ dependence of the detector efficiency for inelastic water scattering | in K0 | — | ○ in K0 (not separated) |
| incoherent term of the sample | not subtracted, fitted as a constant later | same | same |
| dead time | before everything | global: τ Σ I / t over the detector | ✓ global or per tube |
| uncertainties | — | partial derivatives of the intensities only; transmissions, scaling and attenuation factors exact | ✓ exact first-order propagation of every source, systematic ones flagged |
| summing runs | — | all selected files | ✓ grouping with precisions plus an equivalence check ({doc}`grouping`) |

## Strong scatterers and multiple scattering

The weaker the sample, the closer the simplified formula comes to the general one. pySASred
therefore asks, per sample, how strongly it scatters and guides the reduction accordingly
({mod}`pysasfit.sasred.core.scattering_strength`):

1. **Transmission over growing apertures.** From the direct-beam runs with and without the sample,
   T(α) is summed over discs of growing radius. The beam spot gives τ_UL (an upper limit of τ_S:
   neutrons scattered into the spot are always counted), the plateau gives T_Si, and
   1 − τ_UL/T_Si ≤ p_cS. When T still rises at the detector edge, T_Si is a lower limit. The
   beam-independent background (Cd run, or a flat level from the outer half of the empty beam) must
   be subtracted first, otherwise the large discs drift towards T = 1. For strongly incoherent
   samples, incoherent scattering into the detector (≈ (1 − T_Si)/T_Si · α/4π, more for water-like
   samples in flat cells) sets the smallest p_cS that can be seen. The radial average of both runs
   per ring shows where the scattered neutrons land (sample − τ_UL · empty).
2. **p_cS from the cross-section.** After a first reduction, p_cS = d ∫ dΣ/dΩ dΩ with a Guinier
   extrapolation to q = 0 and a Porod tail; the extrapolated share is reported. For single
   scattering p_cS ∝ d λ².
3. **Regime.**

   | p_cS | regime | formula and actions |
   |---|---|---|
   | < 0.05 | weak | eq. 9 with p_cS ≈ 0 (the BerSANS default); beam-spot transmission adequate |
   | 0.05 – 0.3 | moderate | eq. 8 with p_cS on the background; wide-angle transmission; iterate reduce → p_cS → reduce |
   | > 0.3 | strong | the curve is a multiple-scattering curve (Schelten & Schmatz 1980): I(0), Rg and the invariant are not those of single scattering; τ_S cannot be measured; p_cS from the model; thickness or wavelength series; thinner sample |

   The suggested thickness or wavelength for p_cS = 0.1 is given. When the background is a large
   part of the signal near the beam stop (substrates, windows, measurements without beam stop), a
   wrong (1 − p_cS) changes the shape, not only the scale (Strunz et al., fig. 2).
4. **Multiple-scattering check.** Runs of the same sample at different thickness or wavelength are
   grouped ({doc}`grouping`) and compared after division by the thickness: a shape change that
   grows with d λ² is multiple scattering.
5. **Self-consistent p_cS.** {func}`~pysasfit.sasred.core.scattering_strength.self_consistent_probability`
   solves the integral equation by iteration.

Planned for the pySASred window: a *Scattering strength* step per sample group: the T(α)
plot with τ_UL, T_Si and the p_cS limit; p_cS from the curve with its extrapolations; the regime
with the formula used and the suggested actions; the series check when a thickness or wavelength
series exists. Every choice (transmission mode, p_cS and its source, regime) goes into the notes of
the output file.

## Water at two distances

A sample divided by water measured at the same distance needs no solid-angle correction: ΔΩ_ij
cancels pixel by pixel, together with the pixel efficiency η_ij. What remains is the angular
dependence K(2θ) of the water itself (mostly multiple incoherent scattering in the flat cell,
Strunz et al. section 7). The effects multiply: the water intensity per pixel and monitor is

```{math}
W_{ij} \propto \eta_{ij}\,\Delta\Omega_{ij}\,K(2\theta_{ij})
```

and the same pixel sees 2θ_a at L_a and 2θ_b at L_b. After dividing each image by its exact
ΔΩ_ij, the ratio is free of η_ij:

```{math}
\frac{W_{ij,a}/\Delta\Omega_{ij,a}}{W_{ij,b}/\Delta\Omega_{ij,b}} = C\,\frac{K(2\theta_{ij,a})}{K(2\theta_{ij,b})},
\qquad \ln K(2\theta)/K(0) = \sum_k a_k (2\theta)^{2k}
```

The logarithm turns the products into the sum that is fitted over all pixels
({func}`~pysasfit.sasred.core.water.compare_water_runs`). In principle C = 1; a constant C ≠ 1
is something that does not depend on the angle and differs between the distances: a common
distance offset δ (ΔΩ ∝ 1/(L+δ)²), a dead time too small at the higher rate, or a different primary
beam per monitor count. K(2θ) can be divided out of the water before the efficiency map is made
(`efficiency_from_runs(..., water_k=comparison.K)`).

Vifor, water 8 m (82280–82292, empty cell 82281–82291) against 2 m (82203, 82251; empty cell
82204, 82252), 15 129 pixels, χ²/ndf = 1.2:

* K(2θ)/K(0) = exp(−0.293 (2θ)²): 0.9978 at 5°, 0.9911 at 10°, 0.980 at 15° (2θ up to 17° at
  2 m). At 8 m (2θ ≤ 4.5°) the water is flat within 0.2 %; at 2 m a sample normalised by the
  2 m water is about 1–2 % too high at the outer pixels unless K is divided out.
* C = 1.040 ± 0.001: the 8 m water is 4 % higher per steradian than the 2 m water. A common
  distance offset of +5.3 cm would explain it, or a dead time of 1.7 µs instead of 0.66 µs. The
  samples measured at both distances do not show the rate dependence such a dead time would
  cause (their overlap ratio would rise by about 3 % from 12 to 39 kHz at 2 m; it falls), so the
  dead time is the less likely cause. Water at a third distance, or at 2 m with an attenuator,
  would decide. A distance offset would also explain mismatches between geometries on the absolute
  scale seen before.
* The offset is applied by default: preference *offset added to the recorded distance*
  (`sasred.detector.distance_offset_mm`, factory **53 mm**), optionally only for runs started before
  a date (`sasred.detector.distance_offset_before`) once the instrument configuration writes the
  corrected distance. The recorded distance stays `RawRun.SD_m` (run table, grouping, selection,
  labels); the distance to the detection layer `RawRun.SD_effective_m` = recorded + offset is used for
  q and the solid angle, and every reduction notes it. With +53 mm the 8 m and 2 m water agree per
  steradian (C = 1.0001 ± 0.0005) and the water anisotropy becomes K(2θ)/K(0) = 0.997 at 5°, 0.988 at
  10°, 0.974 at 15° (0.993 and 0.984 with the dome divided out).
* **Measuring the offset again**: water (with its empty cell) at several detector distances with the
  same collimation, then {func}`~pysasfit.sasred.core.water.fit_distance_offset`: the images are
  computed with the recorded distances and compared pixel by pixel with the first one (the pixel
  efficiency cancels); δ is fitted together with the water anisotropy. With two distances the primary
  beam per monitor count must be the same; with three or more `free_constant=True` also fits a
  common factor (e.g. another filling of the water cell). Vifor (8 m, 2 m): δ = +53.3 ± 0.8 mm
  (+55.4 mm with the dome divided out). A distance measured by hand to the detection layer (≈ 0.8 cm
  behind the start of the ³He volume at 6 Å) checks it independently; at another wavelength the
  conversion depth part changes (0.88 cm at 4.7 Å, 0.42 cm at 12 Å).
* Runs 82279 (22 s) and 82294 (84 s) are timer runs stopped long before their 1200 s preset; they
  are now marked as aborted.

## What the Vifor transmission runs show

Direct beam with the beam stop moved down (bsy = −300 mm, 300 mm from the beam centre) and
attenuator 1 at 8 m, empty beam 82173, no Cd run:

* In the scattering geometry (bsx, bsy = −27, 1 mm) the beam stop shadow is centred at pixel
  (62.9, 64.1), the direct beam at (61.7, 64.0): the beam stop sits 8.7 mm beside the beam centre
  at 8 m (5.8 mm at 2 m), well inside its shadow.

* The runs are corrected like scattering runs (dead time, attenuator, monitor). The empty beam has
  the highest rate: dead-time factor 1.0030 against 1.0006 for water, so T comes out 0.24 % lower
  than from the raw counts. Without the attenuator the empty beam would give about 2·10⁶ counts/s,
  far beyond what τ = 0.66 µs allows.

* The radially averaged runs show the beam spot within about 6 px; outside it both runs fall to a
  flat, beam-independent background of about 1.5·10⁻⁷ per pixel and monitor (≈ 0.6 % of the
  empty-beam counts over the whole detector).
* Summed over the whole detector without subtracting this background, T_Si comes out up to 1.4 %
  higher than τ_UL for the samples with T ≈ 0.24 (water and the *_1 samples), which would mimic a
  coherent scattering probability. With the flat background subtracted (the default when no Cd run
  is given) T_Si = τ_UL within 0.2 % for every sample: all samples are in the weak regime, and
  `trans.txt` (a central area) is the right transmission for them.
* EXP-005370-A_2: `trans.txt` gives 0.2966, run 82114 at changer position 29.3 mm gives 0.418
  (B_2, C_2, D_2, E_2: 0.419–0.422). The first-night runs of A_2 (position 3.0 mm, next to A_1 at
  0.3 mm) have the transmission of A_1.
* The A_1 transmission run 82113 was recorded at SD = 7.63 m instead of 8 m.
