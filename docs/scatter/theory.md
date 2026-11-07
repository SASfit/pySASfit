# pySCAtter: theory and formulas

This page gives the formulas behind pySCAtter, with the conventions of the CrystalScatter (SasScatter2) C++ code
that the program follows. All lengths are in nm and q is in 1/nm. Equation numbers in brackets refer to the papers
in the [references](#references).

## Intensity

Without a lattice, pySCAtter computes the intensity as

$$
I(\mathbf q) = I_0 \left[ \langle P(\mathbf q)\rangle + \beta(q)\,\langle P(\mathbf q)\rangle\,\bigl(S(q) - 1\bigr)\right] + B,
\qquad \beta(q) = \frac{\langle F(q)\rangle^2}{\langle P(q)\rangle},
$$

where:

* P = |F|² is the particle form factor, normalised to P(0) = 1;
* F is the amplitude;
* ⟨…⟩ is the average over the size and orientation distributions;
* S(q) is an optional structure factor (Percus–Yevick, sticky hard spheres or mass fractal), used in the decoupling
  approximation (Kotlarchyk & Chen 1983);
* I₀ is the scale (`I0`) and B the flat background (`Base`).

With a lattice, the structure factor is replaced by the lattice factor Z(q) of the CrystalScatter form:

$$
I(\mathbf q) = I_0 \left[ S_\mathrm{lat}(\mathbf q)\,\langle P(\mathbf q)\rangle \right] + B,
\qquad
S_\mathrm{lat} = 1 + \beta(q)\,\bigl(Z(\mathbf q) - 1\bigr)\,G(q),
\qquad
G(q) = \exp\!\left(-\tfrac{1}{3}\sigma_\mathrm{DW}^2 q^2\right).
$$

G(q) is the Debye–Waller factor. σ_DW (`EditDebyeWaller`, nm) is the root-mean-square displacement of a particle
from its lattice point (Förster 2005, eq. 25; C++ `dwfactor = σ²/3`). At high q it damps the Bragg peaks and
returns the intensity to the diffuse ⟨P⟩. `EditBFactor` in `.ini` files is not used by the C++ code, and pySCAtter
ignores it too.

## Size distribution

All sizes follow a Schulz–Zimm distribution of the reduced size x = R/R̄:

$$
h(x) = \frac{(z+1)^{z+1}}{\Gamma(z+1)}\, x^{z}\, e^{-(z+1)x},
\qquad \sigma = \frac{\sigma_R}{\bar R} = \frac{1}{\sqrt{z+1}} .
$$

The relative width σ is the parameter `EditSigma` (or `SigmaL` for the second length). The averages ⟨Rⁿ⟩ have the
closed form ⟨xⁿ⟩ = Γ(z+1+n)/(Γ(z+1)(z+1)ⁿ). This makes the size-averaged form factors expandable as hypergeometric
series in q (Wagener & Förster 2023).

### Hypergeometric algorithm

Wagener & Förster (2023) write each form factor as a product of generalised hypergeometric functions. The amplitude of a sphere is
F = ₀F₁(5/2; −q²R²/4). With the Schulz average, every pixel is evaluated by one of three routes, chosen by its own
error estimate:

1. a power series in (qR)² with the Schulz moments as coefficients, used for small qR;
2. closed-form asymptotics, which are the large-q expansions of the averaged Bessel products and are exact for
   σ → 0;
3. Gauss–Laguerre quadrature over x, for the region in between.

For very narrow distributions (z + s > 140) the Laguerre nodes overflow. A uniform grid of x over ±9 standard
deviations is then used instead, and also as a last resort whenever the other routes return a non-finite value.

## Particles

The form factors below are for monodisperse particles. pySCAtter averages them over the Schulz distributions
above and over the orientation distribution.

| particle | form factor / amplitude | parameters (C++ meaning) |
|---|---|---|
| sphere | F = 3[sin(qR) − qR cos(qR)]/(qR)³ | R = `EditRadius` |
| ellipsoid of revolution | F = F_sphere(q r(μ)), r² = R²(1−μ²) + L²μ² | R equatorial, L = `Length` polar semi-axis |
| triaxial ellipsoid | F_sphere(q r), r² = a²μ_a² + b²μ_b² + c²μ_c² | R, L, R₂ |
| cylinder | F = [2J₁(q_⊥R)/(q_⊥R)] · [sin(q_∥L)/(q_∥L)] | R radius, **L = half-length** |
| disk | same as the cylinder | **`EditRadius` = half-thickness, `Length` = disk radius** |
| vesicle (thin shell) | F = [V_o F_s(qR_o) − V_i F_s(qR_i)]/(V_o − V_i) | **`EditRadius` = inner radius, `Length` = outer radius** |
| cube / parallelepiped | F = Π_k sin(q_k a_k)/(q_k a_k) | half-edges a, b, c |
| superball | \|x\|^{2p} + \|y\|^{2p} + \|z\|^{2p} ≤ R^{2p}, tabulated | R, p |
| excluded-volume chain | Regime I series (Förster & Burger 1998), Regime II ∝ q^{−1/ν} | R = R_g |
| Kratky–Porod chain | Kholodenko (1993) interpolation between rod and Gaussian coil | Kuhn length b, contour length L |
| Teubner–Strey | P ∝ 1/(a₂ + c₁q² + c₂q⁴) with peak at q² = k² − ξ⁻², k = 2π/d | ξ, d |

Here q_∥ = q·n and q_⊥ = |q − q_∥ n| for the particle axis n, and μ is the cosine of the angle between q and n.

### Cylinders and disks (C++ conventions)

* **Length.** The C++ code takes `Length` as the half-length. This holds for cylinders and for the
  `Length` of disks, which is their radius.
* **Two distributions.** σ (`EditSigma`) on the radius and σ_L (`SigmaL`) on the length are independent Schulz
  distributions. The orientation-averaged form factor is
  $$
  \langle P\rangle = \int_0^1 d\mu\;\Bigl\langle \bigl[2J_1(qR\sqrt{1-\mu^2})/(qR\sqrt{1-\mu^2})\bigr]^2\Bigr\rangle_R
  \Bigl\langle \bigl[\sin(qL\mu)/(qL\mu)\bigr]^2\Bigr\rangle_L .
  $$
  pySCAtter evaluates this exactly (`CylinderRL`), with each factor computed by the hypergeometric blocks.
* **C++ factorisation.** For L/R ≥ 2 the C++ code uses P ≈ P_rod(q)·P_cs(q), and evaluates the cross-section factor
  at |q| even for oriented cylinders. The isotropic curves differ from the exact average by 1–2 %. In oriented
  patterns, the streak perpendicular to the axis is much weaker in the C++ code than it physically is. pySCAtter
  keeps the exact form; see the validation page.
* **Disks.** A disk is the same cylinder with the roles exchanged: radius = `Length`, half-thickness =
  `EditRadius`. The C++ disk formula fails (gives wrong values) for `Length` < `EditRadius`, which is a rod, not a
  disk.

### Vesicles

The shell amplitude is

$$
F(q) = \frac{R_o^3 F_s(qR_o) - R_i^3 F_s(qR_i)}{R_o^3 - R_i^3},
\qquad F_s(x) = 3\frac{\sin x - x\cos x}{x^3},
$$

with inner radius R_i (`EditRadius`) and outer radius R_o (`Length`). The mean radius and the half-thickness carry
independent Schulz distributions (σ, σ_L), as in the C++ `polyvesicle` routine. pySCAtter averages the exact shell
amplitude (`VesicleRD`), while the C++ code uses a thin-shell closed form. The two differ by an overall factor 1.02
and by less than 2 % in shape.

### Interiors

* **Core + shell.** The amplitude is
  $$
  F = \frac{(\rho_c - \rho_s) V_c F_c(qR_c) + \rho_s V F(qR)}{(\rho_c - \rho_s) V_c + \rho_s V},
  $$
  with the core contrast set to 1 and the shell contrast ρ_s (`EditRho`). This form holds for spheres, cylinders,
  disks and ellipsoids. For core–shell cylinders the shell ratio is ν = L/R_t.
* **Power-law shell.** The shell density is ρ(r) ∝ (r/R_c)^α, built from 20–120 thin homogeneous sub-shells.
  α = −4/3 describes star polymers in a good solvent (Förster & Burger 1998).
* **Multi-shell / myelin.** These use the exact multilayer sum Σ_k Δρ_k V_k F_s(qR_k).

## Orientation distributions

The particle axis n makes the angle δ with the director n₀. The distributions h(δ) are:

| `Ordis` | h(δ) |
|---|---|
| Gaussian | exp(−δ²/(2w²)) |
| exponential, Laguerre | exp(−\|δ\|/w) |
| Onsager | cosh(cos δ /(2w²)) |
| Maier–Saupe | exp(cos²δ / w²) |
| cut-off | 1 for δ < w (smoothed edge) |
| z-dir | δ = 0 (perfect alignment) |
| isotropic | 1 (powder average) |
| fiber pattern | uniform rotation about the fibre axis |

The width w is `EditDbeta` in degrees.

The oriented pattern is
$$
\langle P(\mathbf q)\rangle = \frac{\int h(\delta)\,P(\mathbf q, \mathbf n)\, d\Omega_n}{\int h(\delta)\, d\Omega_n}.
$$
### Quadrature of the orientation averages

No general spherical grid (Lebedev, Fibonacci, spherical t-design) is used. Every average uses the symmetry of
the problem to reduce the dimension, then applies product Gauss–Legendre rules.

* **Isotropic, axisymmetric particles** (ellipsoid, cylinder, disk, core–shell, lens, dumbbell): a 1D
  Gauss–Legendre rule in μ = cos∠(q, n) on [0, 1]. The number of nodes is n = clip(60 + 6 qR_max ν, 80, 1500),
  at most 400–600 per particle, so the count grows with the number of oscillations in μ. `CylinderRL` uses
  n = clip(80 + 8 q max(R, L), 120, 2000). The size averages are done inside the integrand by the hypergeometric
  blocks.
* **Isotropic, non-axisymmetric particles** (triaxial ellipsoid, parallelepiped / cube, superball, super-ellipsoid):
  a product Gauss–Legendre rule in μ ∈ [0, 1] × φ ∈ [0, π/2]. One octant suffices because of the mirror symmetry.
* **Orientation distributions h(δ) about a director**: the averaged particle is axially symmetric about the
  director, so ⟨P⟩ depends on (q_∥, q_⊥) only. pySCAtter tabulates it on a 56 × 36 grid and interpolates bilinearly
  to the pixels. At each grid node the average uses:
  * Gauss–Legendre in δ ∈ [0, π] (40 nodes, weight h(δ) sin δ);
  * the midpoint rule in the azimuth ψ (16 points; spectrally accurate for the periodic integrand);
  * 7–21 nodes of the size distribution.

  The fibre pattern is the same table with h = 1 about the beam axis. Perfect orientation (z-dir) needs no average.
* **Lattices**: the powder average is analytic in the radial form above, with the azimuthal normalisation N(g)
  computed by a 200-node Gauss–Legendre rule. The fibre texture of a lattice averages 240 uniform rotations about
  the fibre axis. An oriented single crystal needs no average.

The moment series of Wagener & Förster (2023, eqs. 13–15) for oriented cylinders is in
`pysasfit.scatter.orient.pattern_series`. The engine does not use it at present.

### Frames

In the C++ frame, q_x and q_y lie in the detector plane (horizontal and vertical) and z is along the beam. The
director is

$$
\mathbf n_C = (\sin\theta\cos\varphi,\ \sin\theta\sin\varphi,\ \cos\theta).
$$

The pySCAtter engine uses x along the beam, y horizontal and z vertical, so n = (n_{C,z}, n_{C,x}, n_{C,y}):

* θ = 0: the axis is along the beam;
* θ = 90°, φ = 0: the axis is horizontal in the detector plane;
* θ = 90°, φ = 90°: the axis is vertical. The streak of long cylinders is then horizontal.

The q map of a 2D image is built in one of two ways:

* **Preset mode** (Qmax): q_x = q_max·i/N and q_y = q_max·j/N.
* **Detector mode**: q is computed from the detector distance, the pixel size and the wavelength. The beam
  component q_beam = −q²/(2k) comes from the Ewald sphere.

## Lattice factor

For every reflection **G** (Miller indices hkl, multiplicity m, structure factor f), the lattice factor adds a
radial peak in |q| − g, with g = |G|, multiplied by an azimuthal factor in the angle ψ between **q** and **G**
(Förster 2005, 2011; C++ `sc_gpu_generic.h`):

$$
Z(\mathbf q) = \frac{2\pi}{n V} \sum_{\mathbf G} m\, |f_{hkl}|^2\; s_q\bigl(|\mathbf q| - g\bigr)\;
\frac{g_3\, A(\psi)}{N(g)},
\qquad g_3 = \frac{2\pi}{g^2}.
$$

* V is the unit-cell volume (the cell area in 2D) and n the number of particles per cell.
* A(ψ) is a Gaussian (or the matching peak shape) in the arc length qψ with width φ_w = 4/`EditAzi` (1/nm), for
  example A = exp(−4q²ψ²/(π φ_w²)). `EditAzi` is an azimuthal domain size in nm, the analogue of D for the radial
  width.
* N(g) = ∫₀^π A(ψ) sin ψ dψ normalises A over the sphere. It is computed with Gauss–Legendre quadrature.
* With `EditAzi` = 0 there is no azimuthal dependence, A = 1 and N = 2, which gives the powder pattern.

The factor g₃ = 2π/g² is evaluated at the reflection, not at q (the 1/q² of the textbook powder average). Near a
peak the two agree. In the tails, the C++ form keeps the peak symmetric in q.

### Peak shapes

With x = |q| − g, δ the width and the C++ constants c₁ = 4, c₂ = 4(√2 − 1), c₃ = 4(2^{2/3} − 1),
c₄ = 2^{1/β} − 1, the radial peak shapes are:

| `ComboBoxPeak` | s_q(x) | area | FWHM |
|---|---|---|---|
| 0 Lorentzian | √c₁ / (πδ [1 + c₁x²/δ²]) | 1 | δ |
| 1 Gaussian | exp(−4x²/(πδ²)) / (πδ/2) | 1 | √(π ln 2) δ = 1.476 δ |
| 2 modified Lorentzian I | 2√c₂ / (πδ [1 + c₁x²/δ²]²) | 0.64 | 0.64 δ |
| 3 modified Lorentzian II | √c₃ / (2δ [1 + c₁x²/δ²]^{3/2}) | 0.77 | 0.77 δ |
| 4 pseudo-Voigt | η·Lorentzian + (1 − η) exp(−x²/δ²)/(√π δ), η = PeakPar/100 | see below | — |
| 5 Pearson VII | Γ(β) 2c₄ / (Γ(β−½) π δ [1 + 4c₄x²/δ²]^β), β = PeakPar | √(c₄/π) | δ |
| 6 Gamma | Förster 2005 generalised shape | — | — |
| 7 anisotropic Gaussian | Gaussian with δ = \|(4/σ_x, 4/σ_y, 4/σ_z)\|/3 | 1 | 1.476 δ |

These are the shapes as the C++ code implements them, constants included:

* **Modified Lorentzians.** The denominator uses c₁ but the prefactor uses c₂ or c₃. The peak areas are therefore
  0.64 and 0.77 instead of 1, and the widths shrink by the same factors.
* **Pseudo-Voigt.** The C++ normalisation `peaknorm2` of the Gaussian part is never set. The summed term is
  therefore divided by η·N_L only, and the intensity scales as 1/η. pySCAtter reproduces this, so that fits give
  the same I₀.
* **Gamma peak.** It is switched off in the C++ code (`#ifdef undef`), which gives s_q = 1. pySCAtter evaluates the
  Förster (2005) shape instead.

### Peak width

The width δ is set by two radio buttons, as in the C++ code:

| mode | δ |
|---|---|
| Debye–Scherrer (`RadButDebyeScherrer`, default) | δ = 4/D |
| paracrystal (`RadioButtonPara`) | δ(q) = 4/D + (σ_DW/d_nn)²·d_nn·q² |
| neither | δ = 1 nm⁻¹ (fixed) |

D (`EditDomainSize`) is the domain size in nm, an absolute length and not a multiple of the unit cell. The relation
D = N a = 4/δ is Förster 2011, eqs. 91–92: a domain of N cells has peaks of width ∝ 1/(N a), whatever the
lattice. δ is therefore the same for every reflection and every lattice constant. For a Gaussian the FWHM is
√(π ln 2)·4/D = 5.90/D. The C++ code does not check D against the lattice constant. A domain smaller than about
three unit cells has no physical meaning, and the peaks then merge into a broad hump.

In the paracrystal mode, d_nn is the nearest-neighbour distance:

* a for sc, lamellae, cylinder packings, hcp and bct;
* a√2/2 for fcc and Fd3m;
* a√3/2 for bcc, Ia3d and Im3m.

The width grows as q², so higher orders broaden (Hosemann's paracrystal, cumulative disorder). σ_DW
(`EditDebyeWaller`) sets both this width and the Debye–Waller factor.

### Lattices

pySCAtter generates the reflections from the lattice type, the cell parameters (a, b, c, α, β, γ) and hkl up to
`HKLmax`. It applies the structure factor of the motif and the extinction rules of CrystalScatter:

* sc, bcc, fcc, hcp, bct, Fd3m and orthorhombic;
* lamellae and the 2D cylinder packings (hexagonal, square, centred rectangular);
* gyroid (Ia3d), OBDD (Pn3m) and Plumber's Nightmare (Im3m), with the reflection conditions of the bicontinuous
  phases;
* the GISAXS lattices.

The cell is oriented by a zone axis [uvw] and a rotation ψ about it (Förster 2011, section 3).

## Fitting

The fit minimises

$$
\chi^2 = \sum_i \left(\frac{I_\mathrm{model}(q_i) - I_i}{\sigma_i}\right)^2
$$

by trust-region-reflective least squares, within the parameter ranges of the SasScatter GUI.

* In 1D the model is folded with the Gaussian q resolution σ_q (Qdev) by 15-point Gauss–Hermite quadrature.
* In 2D the model is evaluated at the q vector of every used pixel.
* The standard errors are √diag((JᵀJ)⁻¹ χ²ᵣ).
* The optional relative residuals use ln I·|I|/σ in place of I.

## References

1. M. Wagener, S. Förster, *Fast calculation of scattering patterns using hypergeometric function algorithms*,
   Sci. Rep. **13**, 780 (2023).
2. S. Förster, A. Timmann, M. Konrad, C. Schellbach, A. Meyer, S. S. Funari, P. Mulvaney, R. Knott, *Scattering
   curves of ordered mesoscopic materials*, J. Phys. Chem. B **109**, 1347 (2005).
3. S. Förster, S. Fischer, K. Zielske, C. Schellbach, M. Sztucki, P. Lindner, J. Perlich, *Calculation of
   scattering-patterns of ordered nano- and mesoscale materials*, Adv. Colloid Interface Sci. **163**, 53 (2011).
4. S. Förster, L. Apostol, W. Bras, *Scatter: software for the analysis of nano- and mesoscale small-angle
   scattering*, J. Appl. Cryst. **43**, 639 (2010).
5. S. Förster, C. Burger, *Scattering functions of polymeric core-shell structures and excluded volume chains*,
   Macromolecules **31**, 879 (1998).
6. M. Kotlarchyk, S.-H. Chen, *Analysis of small angle neutron scattering spectra from polydisperse interacting
   colloids*, J. Chem. Phys. **79**, 2461 (1983).
7. G. V. Schulz, Z. Phys. Chem. B **43**, 25 (1939); B. H. Zimm, J. Chem. Phys. **16**, 1099 (1948).
8. R. Hosemann, S. N. Bagchi, *Direct Analysis of Diffraction by Matter*, North-Holland (1962).
9. J. K. Percus, G. J. Yevick, Phys. Rev. **110**, 1 (1958); R. J. Baxter, J. Chem. Phys. **49**, 2770 (1968).
10. J. Teixeira, J. Appl. Cryst. **21**, 781 (1988); M. Teubner, R. Strey, J. Chem. Phys. **87**, 3195 (1987);
    A. L. Kholodenko, Macromolecules **26**, 4179 (1993).
11. K. G. Yager, Y. Zhang, F. Lu, O. Gang, J. Appl. Cryst. **47**, 118 (2014).
