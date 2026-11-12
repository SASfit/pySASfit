# pyDENSSmulti: 3D maps from several contrasts

pyDENSSmulti reconstructs a low-resolution 3D map of a particle from **several SAS curves of the same particle in
different contrasts**: a SANS contrast-variation series (H2O/D2O mixtures, deuterated components) and/or SAXS. One
curve leaves the internal structure of a DENSS map largely undetermined (Konarev & Svergun, *Nature Methods* **18**,
244 (2021); Grant's reply, *ibid.* 246); several contrasts add exactly that information.

* launcher tab **pyDENSSmulti**, or stand-alone: `pysasfit-densitymulti [data files]`
* scripts: `from pysasfit import densitymulti as dm`
* no external program: the iteration is pySASfit's own numpy/scipy code (DENSS need not be installed)

## Method

The iteration follows DENSS (T. D. Grant, *Nature Methods* **15**, 191 (2018)), re-implemented from the paper:
Fourier transform of the map, replacement of the shell-averaged intensity by the measured one (phases kept), back
transform, real-space constraints, shrink-wrap support (Gaussian smoothing from 3 to 1.5 voxels, threshold 0.2 of
the maximum, largest connected region from step 500, re-centring up to step 1000). The data enter through a
regularised p(r) fit up to Dmax (no positivity: a contrast whose components have opposite signs has a p(r) with
negative parts), evaluated on the q shells of the grid.

Two schemes combine the contrasts.

### Component basis (default)

The particle consists of M **components** with known SLDs ρ_c in every contrast. The unknowns are volume fractions
φ_c(x) ∈ [0, 1] with Σ_c φ_c ≤ 1 (the rest is solvent). Contrast k sees

Δρ_k(x) = Σ_c C[k, c] φ_c(x),  C[k, c] = ρ_c(f_k) − ρ_s(f_k)

with the **contrast matrix** C (f_k: D2O fraction of contrast k). In every step each contrast makes its own Fourier
step and proposes a contrast map Δρ'_k(x); the new φ(x) is, in every voxel, the **weighted bounded least-squares**
solution of Δρ'_k ≈ Σ_c C[k, c] φ_c (weights per contrast; accelerated projected gradient onto φ ≥ 0, Σφ ≤ 1).

* **Match points help.** A contrast in which component c is matched has a zero in column c: it says nothing about c
  (so it cannot pull c towards the solvent, as an equal-weight average of the contrasts would) and everything about
  the others.
* **H/D exchange** is part of ρ_c(f): ρ_c(f) = ρ_c,H2O + f · x · (ρ_c,D2O − ρ_c,H2O), with ρ_c,D2O the SLD in D2O
  at full exchange of the labile H and x the exchanged fraction.
* **SAXS** curves use the X-ray SLD of every component and of water.
* **Volumes**: Σ φ_c × voxel volume is the volume of every component.

The Contrasts panel shows C, its rank and condition number, the sensitivity of every component (1/√((CᵀWC)⁻¹)_cc)
and warnings: rank deficiency, a component matched in every contrast, or determined by one curve only, poor
conditioning.

### Averaging (DENSS-Multiple type)

After J. Sumner & S. Qian, *BBA Advances* **2**, 100063 (2022) (whose code is not public): one SLD map ρ(x), the
same in every contrast. Each contrast makes its Fourier step on its own contrast map; the SLD is the weighted
real-space **average** of the contrasts' proposals ρ_s,k + Δρ'_k / b. pySASfit adds an occupancy b(x) ∈ [0, 1]
(Δρ_k = b (ρ − ρ_s,k)), fitted per voxel: without it the solvent region turns into a fictitious material with the
mean solvent SLD and the particle is not localised. With the occupancy, the average is also the least-squares SLD.
It assumes **no H/D exchange and no matched components**; it is kept for comparison. Optional bounds of the SLD
(Reconstruction panel).

### Scale

* **fitted** (default): one factor for all curves (relative concentrations per contrast in the table). The
  amplitude is fixed by taking the densest voxels as fully occupied (99.5th percentile of Σφ = 1).
* **absolute**: data in 1/cm; I(q) = n_p |A(q)|² with the particle number density n_p from the concentration
  (mg/ml) and the molar mass (kDa). Then the volume fractions and volumes are absolute.

A **flat background** (incoherent scattering) is fitted by no part of the iteration: subtract it (column bg, or
"Fit tail backgrounds": I = A q⁻⁴ + bg over the last 30 % of every curve).

### Several maps

With maps > 1, independent maps (consecutive seeds) are computed in threads, aligned onto the map with the lowest χ²
by their principal axes (8 sign choices, mirror images included: SAS cannot tell enantiomers apart), and averaged;
the spread of every φ_c is kept. Each map is also saved in `map_xx`.

## Panels

| panel | content |
|---|---|
| Data | loaded curves; *Add selected as contrasts* (D2O fraction guessed from the name: "42% D2O", "D2O_100", "H2O") |
| Components | name, SLD in H2O, SLD in D2O (full exchange), exchanged fraction, X-ray SLD, match point; presets (typical values only); solvent SLDs of H2O and D2O |
| Contrasts | per curve: radiation, D2O %, solvent SLD (override), relative concentration, weight, bg, q range; contrast matrix and identifiability |
| Reconstruction | scheme, Dmax (with a 3.5 Rg estimate), grid points (default: from the largest q, at least 32), oversampling (3), steps, shrink-wrap threshold, seed, scale, SLD range (averaging), maps and threads; Run / Stop / Self test |
| Density (3D) | isosurface of every φ_c at a level (0.5), in the colours of the contrast matrix; the larger components see-through |
| I(q) | every contrast with its map curve (exact orientational average of the voxel map, Debye sum over the lattice), residuals |
| Convergence | χ² per contrast (on the shells) and support volume per step |
| Slices | three central slices of a φ_c, of its spread (averages), of the SLD map or of a contrast map Δρ_k |

**File → Example** loads synthetic curves (core-shell sphere or two touching spheres, a perdeuterated and a
protonated protein, 0/42/70/100 % D2O, 1 % noise) with their components, to try the tool.

Every run writes into its own folder (Preferences → pyDENSSmulti): `multi_phi_<c>_<name>.mrc` (φ_c),
`multi_drho_<k>_<curve>.mrc` (Δρ_k), `multi_support.mrc`, `multi_sld.mrc` (averaging), `multi_fit_<k>.dat`
(q, I, err, I_map), `multi_stats_by_step.dat`, `multi_summary.json`. *File → Open a result folder* reads it back;
the last session reopens the runs.

## Checks (synthetic data, 1 % noise, 32³ grid)

* **One contrast, sphere R = 2 nm** (self test): Rg 15.76 Å (exact 15.49 Å, +1.7 %). DENSS 1.8.7 on the same curve:
  16.10 Å (+3.9 %). χ² of the map (exact Debye average at the data points) 86, DENSS map 85. Half-width q shells
  are used (full-width shells: χ² 309).
* **Core-shell sphere** (perdeuterated protein core R = 2 nm, protein shell to 3.5 nm; 0, 42, 70, 100 % D2O; the
  shell is matched at 42 %): absolute scale: volumes 33 900 and 146 200 Å³ (exact 33 510 and 146 080 Å³); φ_core
  ≈ 1 inside 15 Å, φ_shell ≈ 1 between 20 and 30 Å. Fitted scale: 30 200 and 124 700 Å³.
* **Two touching spheres** of the two components: centres 32.9 Å apart (exact 35 Å), 3.6 % overlap of the two maps.
* **Averaging scheme** without exchange (SLD 4 core, 1 shell): SLD ≈ 4 in the core, ≈ 1 in the shell, total
  volume 179 900 Å³ (exact 179 600 Å³). With exchange (the core-shell example) its single SLD is a compromise
  (≈ 7 in the core, ≈ 2 in the shell) and the χ² are higher – as expected.

## In a script

```python
import numpy as np
from pysasfit import densitymulti as dm
from pysasfit.toolbox.merge import read_curves

components = [dm.Component("protein", 1.82, 3.13, exchange=0.9, sld_xray=12.3),
              dm.Component("d-protein", 6.9, 8.2, exchange=0.9, sld_xray=12.3)]
contrasts = []
for fn, f in (("cv_0.dat", 0.0), ("cv_42.dat", 0.42), ("cv_70.dat", 0.70), ("cv_100.dat", 1.0)):
    c = read_curves(fn)[0]                                          # q in 1/nm
    contrasts.append(dm.Contrast(fn, c.q, c.I, c.dI, d2o=f))
C, rs, _ = dm.contrast_matrix(components, contrasts)
print(dm.identifiability(C, names=[c.name for c in components])["messages"])
r = dm.reconstruct(components, contrasts, dm.Settings(dmax_nm=12.0, seed=1))
print(r.chi2, r.volumes)                                            # per contrast; per component (Å³)
maps = dm.reconstruct_many(components, contrasts, dm.Settings(dmax_nm=12.0), nmaps=8, cores=4)
avg = dm.align_and_average(maps)
dm.save_result(avg, "my_run")                                       # MRC maps, fits, summary
```

## Limits

* Low resolution, as DENSS: about Dmax / (grid points / oversampling). Larger grids (64³) take about 8× longer.
* The component SLDs must be right: a component that is "matched" with a slightly wrong SLD still scatters a little
  in that contrast. The exchanged fraction can be varied to test this.
* The alignment by principal axes fails for nearly spherical or highly symmetric particles (several orientations
  with the same inertia); then compare the single maps.

## References

* T. D. Grant, Ab initio electron density determination directly from solution scattering data, *Nature Methods*
  **15**, 191 (2018).
* J. Sumner, S. Qian, DENSS-Multiple, *BBA Advances* **2**, 100063 (2022).
* P. V. Konarev, D. I. Svergun, *Nature Methods* **18**, 244 (2021); T. D. Grant, *ibid.* 246 (2021).
* D. I. Svergun, Restoring low resolution structure of biological macromolecules from solution scattering using
  simulated annealing (DAMMIN / MONSA), *Biophys. J.* **76**, 2879 (1999) – the bead-model route to several contrasts.

## Icon

Two overlapping lobes in two colours (two components) on a teal-blue tile.

## API

Functions and result classes: {doc}`api`.
