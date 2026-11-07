# pySCAtter: validation against CrystalScatter

pySCAtter was compared pixel by pixel with the CrystalScatter C++ code (SasScatter2, console build
`sas_scatter2Cons`). Both programs got the same parameter file, and pySCAtter evaluated its model at the q vectors
of the C++ image.

## Method

* **C++ build.** The C++ code was built with qmake6 (Qt 6). Two small patches were needed:
  * `#include <chrono>` and `QLatin1String` in `XMLCOMP`, to compile;
  * the console's `generateSingleImage` writes q_x, q_y, q_z and I of every pixel to a text file and skips the
    parameter check, which otherwise aborts some runs.
* **Images.** 128 × 128 pixels in preset mode (q_max = 2 nm⁻¹, `GridPoints` = 64). The C++ (q_x, q_y, q_z) maps to
  the engine's (q_y, q_z, q_x).
* **Statistics.** For every case:
  * the median ratio C++/pySCAtter, which is the scale;
  * the median, 95th percentile and maximum of |ratio/scale − 1|;
  * the maximum restricted to pixels with I > 10⁻⁴ I_max.

## Results

| case | scale | median | 95 % | max (I > 10⁻⁴ I_max) | remark |
|---|---|---|---|---|---|
| sphere, σ = 0.1 | 1.0000 | 2 × 10⁻⁵ | 2.5 × 10⁻³ | 0.6 % | maxima at the form-factor minima |
| sphere, σ = 0.02 | 0.9999 | 1.3 × 10⁻³ | 1.6 % | 5.7 % | C++ `pq3` typo (see below) |
| ellipsoid, isotropic | 1.0000 | 3.5 × 10⁻⁴ | 2.7 % | 3.0 % | |
| triaxial ellipsoid | 0.9999 | 2.3 × 10⁻⁴ | 6 × 10⁻⁴ | 2.0 % | |
| cube, isotropic | 0.9998 | 8 × 10⁻⁴ | 5.2 % | 40 % | C++ switches to the Porod limit at high q |
| core + shell sphere | 0.9999 | 2.7 × 10⁻⁴ | 3 × 10⁻³ | 0.3 % | after correcting the C++ `ac9` term |
| vesicle | 1.021 | 2 × 10⁻³ | 4.7 × 10⁻³ | 2.0 % | exact shell vs. C++ thin-shell form |
| ellipsoid, z-dir (oriented) | 0.998 | 4.8 × 10⁻⁴ | 8 × 10⁻³ | 1.3 % | confirms the director frame |
| sc / bcc / fcc spheres, Gaussian, σ_DW = 1 | 0.9999 | 2.6 × 10⁻³ | 3.5 % | 5.6 % | |
| fcc spheres, Lorentzian | 0.9999 | 2.7 × 10⁻³ | 3.7 % | 5.7 % | |
| cylinder, isotropic | 0.962 | 1.7 % | 15 % | 17 % | C++ factorised P_rod·P_cs (see below) |
| disk, isotropic | 0.81 | 16 % | 45 % | 24 % | C++ disk approximation |
| cylinder, oriented | — | — | — | — | C++ evaluates P_cs at \|q\|, see below |
| gyroid spheres | 0.953 | 4 % | 23 % | 30 % | not traced |
| hexagonal cylinders, lamellae | — | 13–16 % | — | — | not matched, see below |

### Lattice factor alone

The lattice factor Z was also compared on its own: P = 1, 1D mode, with the C++ peak-width switches.

| peak shape / width mode | max. relative difference |
|---|---|
| Gaussian, sc and fcc, with Debye–Waller | 1 × 10⁻⁵ |
| Gaussian, paracrystal | 2 × 10⁻⁶ |
| modified Lorentzian I / II, Pearson VII | 2 × 10⁻⁵ |
| pseudo-Voigt | 1 × 10⁻³ |
| Lorentzian; fixed width (neither switch) | 2–4 × 10⁻³ |
| anisotropic Gaussian, isotropic orientation | median 0.1 %, isolated outliers |

The tests in `tests/scatter/test_cpp_lattice.py` check the width modes:

* the Gaussian FWHM is √(π ln 2)·4/D, independent of the lattice constant;
* the Lorentzian FWHM is 4/D;
* the paracrystal width grows as 4/D + c g²;
* the Debye–Waller factor is exp(−σ²q²/3).

## Changes made in pySCAtter to match the C++ code

* **Peak width.** δ = 4/D (Debye–Scherrer), the paracrystal width 4/D + (σ/d_nn)² d_nn q², or the fixed width 1. The
  two radio buttons became the *peak width* combo of the Model panel.
* **Lattice factor.** The lattice factor now has the C++ form: g₃ = 2π/g² at the reflection, the azimuthal
  normalisation N(g), and the peak shapes with the C++ constants.
* **Pseudo-Voigt.** η = PeakPar/100, as in the C++ code; it was PeakPar before.
* **Reflections.** Only reflections with |h|, |k|, |l| ≤ `HKLmax` are used. pySCAtter used HKLmax + 3 before, which
  changed the diffuse level between the peaks by up to 15 %.
* **Debye–Waller factor.** G(q) = exp(−σ_DW² q²/3) from `EditDebyeWaller`. `EditBFactor` is ignored, as in the C++
  code.
* **Lengths.** The cylinder `Length` is the half-length; the disk takes `EditRadius` as the half-thickness and
  `Length` as the radius; the vesicle takes `EditRadius` as the inner and `Length` as the outer radius. σ and σ_L
  are independent distributions.
* **Director.** The director (θ, φ) is defined in the C++ frame: θ = 90°, φ = 90° is vertical on the detector.
* **Narrow distributions.** A Schulz average on a uniform grid replaces the Gauss–Laguerre rule for narrow
  distributions (z > 140). The Laguerre rule gave NaN there.

## Defects and approximations found in the C++ code

These are left as they are in the C++ code. pySCAtter follows the C++ code where it is a convention, and the
physics where it is an approximation or a defect.

1. **Sphere `pq3` term.** A typo in the asymptotic sphere expression causes up to 6 % error for narrow
   distributions (σ = 0.02), near the minima.
2. **Core/shell `ac9` term.** A wrong coefficient in the core/shell cross term. pySCAtter uses the correct form.
3. **Cylinder factorisation.** For L/R ≥ 2 the C++ code computes P ≈ P_rod·P_cs instead of the orientation
   average of the product. The isotropic curve differs by 1–2 % (up to 17 % near minima). For oriented cylinders
   the cross-section factor is evaluated at |q| instead of q_⊥, which strongly suppresses the streak
   perpendicular to the axis. pySCAtter keeps the exact average.
4. **Disk.** The C++ disk formula gives wrong values for `Length` < `EditRadius`. For proper disks it differs from
   the exact average by 1–4 % at mid q and by about 15 % at high q.
5. **Modified Lorentzians.** The denominators use c₁ = 4 while the prefactors use c₂ or c₃. The peak areas are
   0.64 and 0.77 instead of 1. pySCAtter reproduces this.
6. **Pseudo-Voigt.** The Gaussian normalisation `peaknorm2` is never set, so the intensity scales as 1/η.
   pySCAtter reproduces this.
7. **Gamma peak.** It is not implemented (`#ifdef undef`), which gives s_q = 1.
8. **No check of D.** Domain sizes smaller than the unit cell are accepted without a warning.
9. **Cube.** The C++ code switches to the Porod asymptote at high q, which gives tails up to 40 % different.
10. **Spheres on a lattice.** A residual of up to 5 % appears between the peaks, in a band of q starting where
    qR ≈ 5. It is probably in the C++ amplitude ⟨F⟩ but has not been traced.

## Not yet compared

* **Core + shell ellipsoid.** The C++ run ends without output.
* **Excluded-volume chain.** The C++ output is empty.
* **Inhomogeneous shell.** The C++ definition differs (probably a power-law profile); the median difference is 60 %.
* **Core + shell cylinder.** It differs by 44 %, because of the length convention in the shell branch.
* **Superball.** The C++ run takes more than 10 min per image.
* **Oriented lattices.** The zone-axis and ψ rotation conventions are not compared yet.
* **Hexagonal cylinders and lamellae on a lattice.** The diffuse part differs by a smooth factor of 0.2–0.6.
  The C++ code probably uses a different particle term in the 2D-lattice branch; this is not resolved.
* **Kratky–Porod chain, Teubner–Strey and the structure factors.** These exist only in pySCAtter. They are
  checked against their analytic limits (see the main page).
