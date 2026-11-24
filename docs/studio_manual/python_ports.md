# Python ports of SASfit plugins

`pysasfit.studio.ports` holds vectorised numpy versions of SASfit C plugin functions. They take the same
parameter vector as the C function, evaluate all q at once and, for ψ-dependent (anisotropic) functions, also a
grid of directions ψ of Q on the detector:

```python
import numpy as np
from pysasfit.studio import ports

spec = ports.get("ff_deformed_ring_polymer__bz_")
p = np.zeros(50); p[:6] = [30.0, 8.0, 0.55, 2.0, 20.0, 75.0]   # Rg_ll, Rg_perp, nu, I0, theta_0 (deg), psi (deg)
q = np.logspace(-2, 0.5, 200)
I = spec(q, p)                                             # psi from p[5]
psi = np.linspace(0, np.pi, 32, endpoint=False)
I2d = spec(q[None, :], p, psi=psi[:, None])                # (32, 200): psi in radians, broadcast against q
```

The C plugins remain the default and the reference. `ports.enable()` (or `ports.enable(["ff_..."])`) makes
`studio.function(name)` and the explorer use the ports instead; `ports.disable()` switches back.


## Using the ports

Every port is registered under its own name `ff_py_<name without ff_>` next to the C function, in the collection
"py plugin ports"; the model explorer shows them in the form-factor menu tree **py plugin ports** (a copy of the
SASfit plugin tree). From a script:

```python
from pysasfit.studio import function
f_py = function("ff_py_sheared_cylinders__maier_saupe_")   # Python port
f_c = function("ff_sheared_cylinders__maier_saupe_")       # C plugin
```

`ports.enable()` instead replaces the C plugins under their original names (scripts, comparisons).

## Comparison with the C plugins

```
python -m pysasfit.studio.ports.check                       # all ports
python -m pysasfit.studio.ports.check ff_ringpolymerBZ --sets 50 --report ports_report.md
```

or, from the repository root without installing and with a chosen SASfit folder:

```
python tools/port_check.py --lib-dir C:/Users/me/SASstudio --sets 30 --report ports_report.md
python tools/port_check.py --list
python tools/port_check.py --aniso 1e-9          # tighter SASfit precision for this run
```

Freshly built plugins can be taken directly from the SASfit source tree, without copying the DLLs:
`python tools/port_check.py --plugin-dir /c/Users/<you>/switchdrive/SASfitGit/src/plugins` (session only), or once
for all pySASfit sessions `pysasfit-studio plugin-build-dir /c/Users/<you>/switchdrive/SASfitGit/src/plugins`
(saved; `--clear` forgets it). Every `<dir>/<plugin>/lib/libsasfit_<plugin>.dll` then replaces the installed plugin of
the same name and `<dir>/<plugin>/include/sasfit_<plugin>.h` the bundled header; libsasfit itself still comes from
the SASfit installation.

SASfit plugins integrate with `sasfit_integrate`, whose relative tolerance is SASfit's eps *aniso* (studio default
1e-5), not eps *nriq*. For such functions `tools/port_check.py` compares at max(port tolerance, 2·eps aniso), so a
deviation of ~1e-6 with eps aniso = 1e-5 is SASfit's requested precision, not an error.

For every port, random parameter sets within `PortSpec.sample` are evaluated with the port and with the C
plugin of the SASfit library (`SASfitFunction(name, language="c")`) on q = 10⁻³ … 10 nm⁻¹ (with Rg = 1 … 100 nm:
q Rg = 10⁻³ … 10³). The report gives the largest relative deviation, the times of both implementations, a check
that the vectorised ψ argument equals the ψ parameter, and the number of points where the C function returned 0
but the port did not (a failed integration in C). The C side uses your SASfit integration settings
(Settings → Integration…), so integration-based functions are compared at SASfit's own precision.

The test suite (`tests/studio/test_ports.py`) compiles the unchanged C sources of the ported functions
(`tests/studio/port_reference/sasfit_src`) with small shim headers and GSL, with a reference-quality
`sasfit_integrate`, and checks the ports against them, against closed forms (Debye function, Dawson integral of the
Gaussian ring) and against the large-q asymptotics.

## Ported functions

| function | C source | accuracy | notes |
|---|---|---|---|
| ff_deformed_generalized_gaussian_chain | deformed_polymers | exact (~1e-15) | SASfit builds before 2026-11-17 use Guinier exp(-β/3) for √β < 0.1 (error up to ~4e-6) |
| ff_deformed_ring_polymer__bmb_ | deformed_polymers + polymer_rings | ~1e-14 | tanh-sinh quadrature, scaled to the end-point layer |
| ff_deformed_ring_polymer__bz_ | deformed_polymers + polymer_rings | ~1e-14 | as above |
| ff_ringpolymerBMB | polymer_rings | ~1e-14 | isotropic |
| ff_ringpolymerBZ | polymer_rings | ~1e-14 | isotropic |
| ff_we/hsh__deformed_polym__netw_ | deformed_polymer_network | ~1e-12 | tube models, split at the end-point layer |
| ff_we/hsh__deformed_ringpolym__netw_ | deformed_polymer_network | ~1e-12 | rings, symmetric half interval |
| ff_reptating_chain | reptating_chain | ~1e-15 | C ignores I0 and loses digits in D(x) for q Rg < ~0.01 |
| ff_polymer/ring_under_shearflow | shearflow | ~1e-15 | sector average with 32 Gauss-Legendre nodes |
| ff_polymer/ring_under_shearflow1 | shearflow | ~1e-15 | |
| ff_polymer/ring_under_shearflow2 | shearflow | ~1e-11 | two-stage form factor in closed form (polymer), kinks as segment edges (ring) |
| ff_shearflow_lin, _g1, _g2 | shearflow_polymers_lin | ~1e-15 | two-stage form factor in closed form; detector angle is varphi (p[11]) |
| ff_sheared_cylinders__{maier_saupe, gauss, boltzmann, onsager, heavyside, hayterpenfold, legendre}_ | sheared_objects | not compared | see below |
| ff_sheared_spheroids__{maier_saupe, gauss, boltzmann, onsager, heavyside, hayterpenfold}_ | sheared_objects | not compared | see below |
| ff_sphere, ff_spherical_shell_i/ii/iii, ff_hollow_sphere, ff_multilamellar_vesicle | sphericalshells | ~1e-11 | stable Φ(x) = 3(sin x - x cos x)/x³; C fixed 2026-11-18 (digits lost for 1e-4 < q R < 0.01, mlv `_v` returned a radius) |
| ff_sphere_with_3_shells | sphere_with_3_shells | ~1e-11 | |
| ff_boucher_sphere, _sphere2, _profiles, _profiles2 | boucherprofile | ~1e-13 | 0F1 via J_{b-1} |
| ff_kratkysphere | kratky_sphere | ~1e-7 | Gauss-Legendre over q⊥; Δq⊥ = 0 gives the unsmeared limit (C fixed 2026-11-18, was 0/0) |
| ff_gauss, _gauss2, _gauss3, ff_gauspoly | polymer_chains | ~1e-10 | C fixed 2026-11-18 (digits lost for small q Rg; ``_f`` was (1-e^{+u})/u) |
| ff_generalized_gaussian_coil, _2, _3 | polymer_chains | ~1e-10 | exact (series of u^-a γ(a,u)); C fixed 2026-11-18 (was Guinier for q Rg < 0.1; c_compat) |
| ff_benoitstar, ff_broken_rods_star, ff_dozierstar, _dozierstar2, ff_gauss_exvol_star, ff_polydispersestar | polymer_stars | ~1e-10 | |
| ff_blob__beaucage_, __dendrimer_, __star_ | blob | ~1e-15 | |
| ff_fract_branched_polym | hammouda_branched | ~1e-10 | C fixed 2026-11-18 (lost all digits for small q Rg: Γ - Γ_inc cancellation) |
| ff_polynom, ff_cv2, ff_cv | polynom, cv | exact | |
| ff_generalized_form_factor, ff_lesser_generalized_form_factor, _2 | generalized_formfactor | ~1e-12 (sigma_a = 0) | gFF with sigma_a ≠ 0: C fixed 2026-11-18 (was uninitialised); log-normal average by Gauss-Hermite (C: 99.9 % range, ~3e-4) |
| ff_beaucage_exppowlaw, _2, ff_broad_peak, ff_debye_anderson_brumberger, ff_g_dab, ff_def_g_dab, ff_extended_guinier_law, ff_generalized_guinier_porod_law, ff_ornstein_zernike, ff_spinodal, ff_spinodal2, ff_teubner_strey | non_particulate | ~1e-13 | closed forms; def_g_dab: 96-point Gauss-Legendre over cos θ. C fixed 2026-11-18 (Beaucage laws NaN at q = 0, spinodal2 printed every call, GGP law NaN for Rg2²/(3-s2) <= Rg1²/(3-s1), now a domain error) |
| ff_linear_pearls | linear_pearls | ~1e-13 | uses sasfit_sphere_fc (core, fixed 2026-11-18: series up to qR = 0.1). Normalisation changed 2026-11-18 (C and Python): I(0) = (Δη N V)², the SASfit convention (was divided by N V, the SASview convention) |
| ff_pearl_necklace, _ext, _mod | pearl_necklace | ~1e-15 | finite geometric sums (integer Np) or binomial series in 1-γ (_mod, real Np); I(0) = (m β_r + Np β_s)² with m rods (SASfit convention; C and Python were normalised to P(0) = 1 until 2026-11-18). C fixed 2026-11-18 (0/0 at q = 0, cancellation for small q(l+2R), Psi series coefficient 1/240 → 1/280). _ext rewritten 2026-11-18 (C and Python): g freely jointed rods of length l between neighbouring pearls, the end rods radial to their pearls (bonds R+l, l, …, l+R); g = 1 now equals pearl_necklace exactly (the old formula used rods of length l+2R in the rod-rod terms, 35 % off at q(l+2R) ≈ 50); checked against the exact Rg² and a Monte Carlo Debye sum |
| ff_fisher_burford, ff_massfractexp, ff_massfractgauss, ff_dlcaggregation, ff_rlcaggregation | fractal_obj | ~1e-13 | massfractgauss via Kummer 1F1(D/2; 3/2; -x) (C fixed 2026-11-18: overflow / asymptotic branch off by 4e-4) |
| ff_aggregate__exp__x_a__cut_off_, ff_aggregate__overlapsph_cut_off_ | fractal_obj | ~1e-12 | oscillatory integrals with Gauss-Jacobi/Legendre panels (graded at 0 for non-integer α), all q; C extrapolates a power law beyond q ξ = 100 / 50 (excluded from the comparison) |
| ff_dumbbellshell, ff_two_attached_spheres, ff_doubleshellchain, ff_tetrahedrondoubleshell, ff_stackdiscs | fractal_obj | ~1e-10 | C fixed 2026-11-18: two_attached_spheres_v used 4/.3; doubleshellchain summed only the last row for non-integer n. tetrahedrondoubleshell: the 5th sphere sits at √(8/3) R12 from the 4th (two face-sharing tetrahedra; C fixed 2026-11-18, the old √(6/3) distances are not realisable in 3D) |
| ff_two_attached_spheres_w_distr | fractal_obj | ~1e-5 (C cuts the log-normals at 99.9 %) | 2D log-normal average by Gauss-Legendre in ln R1, ln R2; C fixed: _f/_v used the median of R1 for both radii |
| ff_fractal_series1, 2, 3 | fractal_series | ~1e-15 | ∫ SD(x) x⁶/(1+q²x²)² dx computed directly; C fixed 2026-11-18 (closed forms with a three-term series for q xmax < 0.32: errors up to 7e-3) |
| ff_power_law, ff_profile__power_law | powerlaw | ~1e-11 (mpmath) | sphere with the profile Δη(1-(r/R)^α): Gauss-Jacobi for ∫ t^(α+2) sinc; C uses sasfit_pfq (not in the reference library) |
| ff_guwelop_sphere, _cyl_Pcs, _planar_Pcs, _profile | guwelop | ~1e-11 (mpmath) | Fourier integrals of N(1-F_GUWELOP(x)) with panels at quantiles of F; compared with C for q λ < 15 (C loses relative accuracy beyond, up to 1e-4) |
| ff_ribbon_mono, peaks_ribbon_w_sd | ribbon | ~1e-14 (mpmath), C ~1e-12 | thin rod × ribbon cross section (A << B): F(x) = 1F2(1/2;3/2,2;-x²) = [J1(2x)+2Σ_{k≥1}J_{2k+1}(2x)]/x by Miller recurrence; Schulz average of the width as 2∫₀¹K(w)Re[k(k+1)θ²(1-iqθw)^-(k+2)]dw (≤ (k+2)/4 oscillations for any q). C rewritten 2026-11-18 (was sasfit_pfq plus the asymptotes 2/(qB) for qB/2 > 3, errors up to 2.3 %, and 2B/q); _v of ribbon_mono now returns ABC |
| ff_fuzzysphere, ff_linshell, ff_linshell2, ff_expshell, ff_coreshellmicrogel (+ radial profiles) | fuzzysphere | ~1e-11 (mpmath) | core φ(qR) stable; linear shells closed form with χ(x) = ∫₀¹t³sinc(xt)dt (series below x = 1), exponential and parabolic zones by Gauss-Legendre (panels ≤ 2 rad, split at the kinks). C fixed 2026-11-23: the closed forms divided by q³…q⁵ (LinShell: 100 % errors at q R = 1e-3, fuzzysphere 1e-5 at q R = 1e-6), LinShell2 macro R1 = RTOT-DR unparenthesised (q*R1 = q*RTOT - DR), LinShell(2)_v returned 0; now same series/quadrature for small q |
| ff_gen__exp__sphere, ff_gen__gauss__sphere (+ profiles) | gen_exp_sphere | ~1e-14 (mpmath) | generalised exponential / Gaussian profiles r^α e^(-…): (θ/u) sinc form, Kummer 1F1(β/2; 3/2; -x). C fixed 2026-11-23: the gen_gauss fallback (GSL error) read an uninitialised value and kept only the Gaussian term; now the full asymptotic expansion |
| ff_clipped_random_waves_1…5, _twocut_1…5, _twocut_intersection_1…5 | stochastic_models | ~1e-9 (scipy QAWO); C ~1e-3 (intdeo) | level-cut Gaussian fields: γ = X(g)/(φ(1-φ)), X = 1/2π∫₀^asin g e^(-α²/(1+sin θ)) (one cut), 1/π∫[e^(-α²/(1+s)) - e^(-α²/(1-s))] (two cuts), 2φX_s + X_s² (intersection), γ(0) = 1; θ rule graded towards ±π/2. C fixed 2026-11-23: one cut subtracted φ² (γ(∞) ≠ 0), two cuts used the sum with 1/2π, intersection C11²/2π - φ², integer abs() on asin g (γ = 0 for \|asin g\| < 1), gy4 not normalised at r = 0 |
| ff_boolean_union_model, ff_dead_leaves_model | stochastic_models | ~1e-10; C ~1e-4 (intdeo) | log-normal spheres, ThetaK closed form checked against the covariogram average. C fixed 2026-11-23: Boolean γ used e^(K-1) instead of e^K - 1; dead leaves leaked its workspace. ff_boolean_intersection_model is a stub in C (returns 0), not ported |
| ff_sphere_chains_rw_, __rc, __nagg | spherical_ellipsoidal_micelles | ~1e-14 | Pedersen sphere + Gaussian chains |
| ff_ell_chains_rw_, __rc, __nagg | spherical_ellipsoidal_micelles | ~1e-13 | spheroid + Gaussian chains, Gauss-Legendre over cos α. C fixed 2026-11-18 (spheroid surface: oblate ε → ε², prolate ae/tan → ae/sin; _nagg: R_core with 4/3 = 1) |
| ff_worms_ps1_, ff_worms_ps2_, ff_worms_ps3_ | polymer_semi_flexible + thin_objects | ~1e-11 vs C | Pedersen–Schurtenberger methods 1–3 (with / without excluded volume); method 3 continuous in value and slope at q0. C fixed 2026-11-23: fRg2 cancelled for L << b (series), IDebye was 1 - u/3 below u = 0.01 (8e-6), Sharp–Bloomfield term 7/(15u)(1 - e^-u) cancelled (expm1), dS/dq at q0 with the absolute step 1e-8 (now 1e-3 q0) |
| ff_kholodenko_worm | polymer_semi_flexible + thin_objects | ~1e-13 vs C, 1e-9 vs scipy | contour integral by Gauss-Legendre (geometric panels, or 1.5 rad panels in the oscillating regime q lb > 3) |
| ff_freely_joined_chain_of_rods | polymer_semi_flexible + thin_objects | ~1e-14 vs C, 1e-12 vs direct sum | C fixed 2026-11-23: (1-γ)⁻¹ - n⁻¹(1-γⁿ)(1-γ)⁻² cancelled for small qb (100 % at qb = 1e-5; the series branch below 1e-6 gave S(0) = 1 - 2/(3n)); γⁿ NaN for γ = j0(qb) < 0 and non-integer n (now \|γ\|ⁿ cos πn) |
| ff_sphere_chains_saw_, __rc, __nagg | spherical_ellipsoidal_micelles | ~1e-13 | Pedersen model 39 with self-avoiding worm chains. C fixed 2026-11-23: R_core/n_agg variant used the core volume V/(1 - xsolv) instead of V(1 - xsolv); _nagg: xsolv_core >= 1 checked |
| ff_sphere_r__a, _rc, _nagg | spherical_ellipsoidal_micelles | ~1e-13 | r^-α corona (Gauss-Legendre, Gauss-Jacobi from r = 0), chains: worm (exvol 0, 1) or generalized Gaussian coil (exvol 2: L = Rg, b = ν). C fixed 2026-11-23: n_agg = 0 returned Φ² without volume and contrast; _nagg: xsolv_core >= 1 checked. Not ported: sphere_2nd+R^-a_Rc, sphere_smooth_interface+R^-a_Nagg (stubs returning 0) |
| ff_koyama_worm | polymer_semi_flexible + thin_objects | ~1e-12 vs C, 1e-10 vs scipy | Koyama (1973): pair distribution from <R²> and <R⁴> of the worm-like chain, contour integral by Gauss-Legendre on panels of the phase and the Gaussian exponent; <R⁴> by series below x = 1 (the closed form cancels; C uses its series only below 1e-4, ~1e-9 at L << lb), rod and Gaussian limits checked |
| ff_pcs_ellcylsh | thin_objects | ~1e-13 (σ = 0) vs C, 1e-9 vs scipy | elliptical core-shell cross-section, log-normal R and t, average over the orientation α of the ellipse. C fixed 2026-11-24: for ε ≠ 1 the α integral was not divided by π/2 (π/2 too large, discontinuous at ε = 1) |
| ff_rod_sd_ellcylsh | thin_objects | ~1e-13 (σ = 0) | Pcs:ellCylSh · P'(q) of a thin rod with log-normal length |
| ff_worm_ps1_sd_ellcylsh, ff_worm_ps2_sd_ellcylsh, ff_worm_ps3_sd_ellcylsh, ff_worm_kholodenko_sd_ellcylsh | thin_objects | ~1e-11 (σ = 0) | Pcs:ellCylSh · L² S(q) of the worm. C fixed 2026-11-24: the parameter vector was passed to Pcs unchanged, so methods 1–3 used lb as σ_t of the thickness distribution and the Kholodenko variant read the contrasts one place too early. Method 3 without excluded volume (header: dummy), as in C |
| ff_pcs_homogeneouscyl, ff_pcs_homogeneousplate, ff_pcs_twoinfinitelythinlayers, ff_pcs_bilayergauss, ff_pcs_layeredcentrosymmetricxs, ff_pcs_plate_chains_rw_ | thin_objects | ~1e-14 (σ = 0) vs C, 1e-10 vs scipy | cross-section form factors with log-normal widths. C fixed 2026-11-24: Pcs:BilayerGauss used an uninitialised parameter structure for its log-normal |
| ff_disc_sd_homoxs, ff_disc_sd_bilayergauss | thin_objects | ~1e-14 (σ = 0) vs C, 1e-13 vs mpmath | P'(q) of a thin disc times a plate or bilayer cross-section. C fixed 2026-11-24: P'(q) of the thin disc was NaN at q = 0 and cancelled for small qR (series); ff_disc_sd_homoxs used half of R0 as the radius |
| ff_cylinder, ff_disc, ff_rod | cylinders | ~1e-14 vs C, 1e-11 vs scipy / 1e-13 vs mpmath | randomly oriented cylinder (orientation integral by Gauss-Legendre on phase panels), infinitely thin disc and rod. C fixed 2026-11-24: ff_cylinder_f did not set q (amplitude of a previous q); the disc and its amplitude lost digits for 1e-6 < qR < 1e-2 (series only below 1e-6, 1 - cos instead of 2 sin²) |
| ff_longcylinder, ff_flatcylinder, ff_porodcylinder, ff_porodcylinder_v2 | cylinders | ~1e-12 vs C, 1e-10 vs mpmath | Porod's approximations for long and flat cylinders (Struve functions), blended by 2R/L; the flat approximation holds for qL ≲ 1 and becomes negative beyond. C fixed 2026-11-24: both closed forms cancelled for small q (long: 2e-4 at qL = 1e-4; series below q·max(L, 2R) = 0.3, 8/σ²(3J₂ + J₀ - 1) by its series), flatcylinder _f and _v swapped, porodcylinder_v2 used an uninitialised parameter structure |
| ff_longcylshell | cylinders | ~1e-13 vs C | Pcs of a circular core-shell cross-section · P'(q) of a thin rod |
| ff_ellcylshell1, ff_ellcylshell2, ff_cylshell1, ff_cylshell2 | cylinders | ~1e-14 vs C, 1e-9 vs scipy | core-shell cylinder with elliptical (circular) cross-section, shell around the mantle (1) or also on the caps (2). C fixed 2026-11-24: for ε ≠ 1 the orientation average weighted A² with sin α/(4π) although sasfit_orient_avg is already uniform in cos α (result ~1/(4π) too small, discontinuous at ε = 1); the volumes of the type-2 shells lacked the caps (L or L + t instead of L + 2t) |
| ff_torus | cylinders | ~1e-15 vs C | ring torus with shell and height ratio ν (tube integral in r = R + a sin β). C fixed 2026-11-24: the core integral started at R - a < 0 for a > R |
| ff_ellipsoidal_shell_0, _0t, _rp, _rp_t, _re, _re_t, _2, _2t | ellipsoidal_shell | ~1e-14 (σ = 0) vs C, 1e-9 vs scipy | core-shell spheroids with solvent fractions, log-normal scale of t, R_p, R_e or combinations. C fixed 2026-11-24: _re and _re_t scaled the polar instead of the equatorial radius; _re_v, _2t_v (0) and _0_v (core only) volumes |
| ff_spheroid_R, ff_spheroid_V, ff_spheroid_nu | ellipsoidal_shell | ~1e-14 vs C | homogeneous spheroid given by (R_p, R_e), (V, R_e) or (ν = R_p/R_e, R_e). C fixed 2026-11-24: spheroid_V put R_e into the polar axis (volume ≠ V, I(0) ≠ (Δη V)²) |

**Batch 2 (2026-11-18), not ported:** local_cyl_micelle and local_planar_micelles are stubs in the C sources (all
functions return 0). The sphere+R^-a and sphere+chains(SAW) functions of spherical_ellipsoidal_micelles call the
worm form factor sasfit_sq_p__q___worm_ps3_ (Pedersen-Schurtenberger); ported in batch 5 (2026-11-23). The core
functions sasfit_sphere_fc and sasfit_gauss_fc (sasfit_common/sasfit_utils.c) now use their series up to
qR = 0.1 and (qRg)² = 0.5 (before 1e-4: up to ~1e-8 relative error from cancellation); rebuild of the SASfit core
needed.

**Batch 6 (2026-11-24):** all functions with a log-normal size distribution agree with the C code only to
~1e-4 (up to 1e-3 at large q): SASfit's find_LogNorm_int_range ends the integral 3.7 σ above the peak of
x^(2 dim) LN(x) (sasfit_common, not changed), the ports integrate over ±8 σ. Not ported yet in thin_objects:
ff_ellsh_sd_homoxs, ff_ellsh_sd_bilayergauss, ff_cylsh_sd_homoxs, ff_cylsh_sd_bilayergauss (P'(q) of thin
ellipsoidal shells and closed hollow cylinders) and the P'(q) structure-factor-slot functions (worm_ps3 PRISM/RPA,
thin spherical shell, gGC).

**Batch 7 (2026-11-24):** plugins cylinders and ellipsoidal_shell. Not ported: ff_alignedcylshell (oriented,
ψ-dependent). The ellipsoidal shells with σ > 0 agree with C to ~1e-6 (C's truncated log-normal range); the
spheroids and ellipsoidal_shell_0 do not depend on their p[4] (dummy).

**Batch 3 (2026-11-18), not ported yet:** fuzzysphere, stochastic_models, gen_exp_sphere
(large tabulated data), the structure factors of fractal_obj (sq_*), the worm form factors of thin_objects
(sasfit_sq_p__q___worm_ps3_, needed for the sphere+R^-a / SAW micelles).

Deviations of the C code before 2026-11-17, all fixed in the SASfit sources on 2026-11-17 (plugins
deformed_polymer_network, reptating_chain, shearflow, shearflow_polymers_lin; rebuild needed). The ports compute
the corrected result; `intensity(..., c_compat=True)` reproduces the old C behaviour for comparisons with an old
SASfit build: `tools/port_check.py --old-sasfit` (or `check.compare(..., old_sasfit=True)`) uses it.

* **polymer_under_shearflow:** the sector average is divided by 2Δ instead of Δ (result halved), Δθ0 = 0 divides by
  0, and the averaged sector is θ0 ± Δθ0/4 instead of ± Δθ0/2 (also **ring_under_shearflow**).
* **ring_under_shearflow2:** for Δθ0 > 0 the sector integral is not divided by its width, Δθ0 = 0 divides by 0.
* **polymer_under_shearflow1:** ψ is read from p[7]; the parameter list has ψ at p[2].
* **reptating_chain:** I0 is never applied.
* **shearflow_lin_g1:** Δφ = 0 gives 0/0.
* **two-stage formula** (polymer_under_shearflow2, shearflow_lin*): 2(e^{-bf}/c + (1-e^{-bf})(1/b-1/b²) + ...) loses
  digits for small q (relative error ~1e-5 at q Rg ≈ 0.005) and is 0/0 at q = 0; the port uses the closed form of
  2∫(1-x)e^{-g(x)}dx with stable (1-e^{-z})/z terms. Likewise D(x) = 2(x-1+e^{-x})/x² in the reptating chain.
* **WE networks, reptating chain:** a ψ < 0 check on a scratch value before it is set (removed); the reptating chain
  now uses the series of D(x) for |x| < 0.2 and returns I0·D(α) for λ = 1.
* **polymer/ring_under_shearflow1:** no I0 (p[1] was a dummy parameter, now I0); ring_under_shearflow1 had the
  weight (1-x) instead of 2(1-x), so P(0) = 1/2 (now P(0) = 1).

**Sheared objects** (`ports/sheared.py`): core-shell cylinders and spheroids with an orientation distribution
p(θ, φ) and a log-normal size distribution. I(q, ψ) = <∫ p |A(q, γ)|² dΩ>_ν with ∫ p dΩ = 1, so I(0) = (Δη V)².
Checked against independent references (isotropic average for κ = 0, full-sphere double integrals, the aligned
limit, the log-normal average) and against the corrected C sources (~1e-13 for σ = 0;
`tests/studio/test_sheared_ports.py`). For σ > 0 the C function cuts the ν integral at the 99.9 % point of the
ν⁶-weighted log-normal (deviations ~1e-5), so the comparison samples σ = 0.

*Methods and speed (2026-11-18).*

* **Legendre method** (default): A²(q, γ) is expanded in Legendre polynomials of cos γ,
  a_l(q) = ∫ A² P_l dt, and I(q, ψ) = Σ_l (l+½) a_l(q) g_l(ψ) with g_l(ψ) = ∫ p(û) P_l(û·q̂) dΩ. For the axially
  symmetric distributions g_l = p_l P_l(cos ψ) (addition theorem) with the moments p_l of p(θ); for Hayter-Penfold
  from the Fourier series of p in φ (even m) and normalised associated Legendre functions (cost ~ l·m·n_θ;
  the θ, φ quadrature only above l = `HP_LMAX`). Per q only a 1D
  integral over cos γ is needed (l up to ~1.2 q D + 40), and every ψ costs a Legendre sum: a 64 × 36 map takes
  0.01 s (σ = 0) instead of ~1 s. Agrees with the quadrature method to ~1e-12.
* **Size average**: A(q, γ; ν) = ν³ A(qν, γ; 1), so I(q) = Σ_k w_k ν_k⁶ I₁(q ν_k). The ν nodes (composite
  Gauss-Legendre in ν) resolve the oscillations of I₁(q ν) in ν, ~3 per period π/(q D). **Correction:** the
  24-node Gauss-Hermite rule used until 2026-11-18 did not; for σ > 0 the ports were off by 4e-4 at q D ≈ 13 and
  up to tens of percent at q D ~ 100 (the comparison with C only sampled σ = 0).
* **Schulz-Zimm** (`size_dist="schulz"`, ports `ff_sheared_{cylinders,spheroids}__<kind>__schulz_`, Python only):
  mean 1, relative standard deviation σ < 1. For spheroids the size average of A² is analytic (series and closed
  forms of the sphere block, `pysasfit.scatter.blocks`, Wagener & Förster 2023), for cylinders by ν nodes.
* **Quadrature method** (`method="quadrature"`, as the C plugin): θ, φ double integral; with Numba as a compiled
  kernel in parallel over the q points (`ports/_sheared_numba.py`, J₁ from Cephes as in SciPy). Choose with
  `sheared_intensity(..., method=..., backend=...)`, `sheared.METHOD` / `PYSASFIT_SHEARED_METHOD`,
  `sheared.BACKEND` / `PYSASFIT_SHEARED_BACKEND`.

Errors of the C plugin, **fixed in the SASfit sources on 2026-11-17** (plugin sheared_objects; rebuild needed —
older builds give wrong results):

* `abs()` of a double in pMaierSaupe, pGauss, pOnsager, pHeavysidePi and pLegendre truncated θ to the integer 0 or 1
  → `fabs()`.
* θ ∈ [0, π/2], φ ∈ [0, π] is a quarter of the sphere; only the cylinder Boltzmann function multiplied by 4, and the
  cylinder amplitude had an extra factor 2 (`2*gsl_sf_bessel_j0`) → factor 4 everywhere, no factor 2.
* Hayter-Penfold missed the half with γ₋ → `cubature_param.gam2 = gamHPminus` is added in the integrand, factor 2;
  the spheroid version never set `cparam.gam` → set.
* pOnsager used cos(x) instead of cos(θ) for κ > 10 → fixed. The Heaviside θ range was wrong for |κ| > 2/π (band:
  [1/|κ|, π/2] instead of [π/2-1/|κ|, π/2]) → fixed.
* Boltzmann cylinder: the φ limits were never set (strategy switch) → same sasfit_cubature call as the others.
* pLegendre wrote 8 values into Pl[7] (unused) → removed; sheared_cylinders__legendre_ did not compile (`ierr`) and
  was not exported → fixed, added to interface.c, CMakeLists.txt and the header (parameters P2, P4, P6).

Observations on the C code made while porting, and the fixes made in the SASfit sources on 2026-11-17
(`src/plugins/polymer_rings/sasfit_ff_ringpolymerB{MB,Z}.c`, `src/plugins/deformed_polymers/*.c`; they take
effect after rebuilding the SASfit plugins):

* **Rings (BMB, BZ):** the kernels exp(-μ (x(1-x))^a) are concentrated in a layer of width μ^{-1/a} at the end
  points for large q Rg. Adaptive integration over [0, 1] misses it. SASfit (default settings) gave wrong values
  from q Rg ≈ 350 on and 0 beyond. Plain GSL QAGS failed from q Rg ≈ 30, CQUAD was off by ~1e-3 at q Rg ≈ 45.
  *Fix:* the exponent is symmetric under x → 1-x, so P = 2 ∫_0^{1/2} exp(-μ s(x)) dx, and the upper limit is cut
  where μ s(x) > 45. Now plain QAGS gives 1e-9 up to q Rg = 10³, and the C functions are 3–10× faster.
* **Deformed generalized Gaussian chain:** the Guinier approximation exp(-β/3) for √β < 0.1 (error up to ~4e-6)
  and the cancellation just above it were replaced by the exact series for u < 1. C and port now agree to 1e-14.
* **Deformed rings:**
  - the `psi < 0` check read the scratch value p[MAXPAR-1] before it was set; it and the `theta_0 < 0` check were
    removed (any angle is allowed);
  - the parameter structure passed to the ring function is now initialised with `sasfit_init_param`, and its
    errors are passed on.

## Parameter domains and default values

Every port has a description of its parameters, ``spec.params`` (:mod:`pysasfit.studio.ports.params`): for each
parameter its domain (lower/upper bound, open or closed), a default value and a kind (length, sld, intensity,
angle, exponent, count, fraction, dimensionless, dummy; lengths in the unit reciprocal to q, the numbers chosen
for q in nm⁻¹). It comes from

1. ``ports/param_table.py``, generated by ``python tools/gen_port_params.py``: the names of the plugin headers and
   the bounds parsed from the ``SASFIT_CHECK_COND*`` domain checks of the C sources (a check of one parameter
   against a number becomes a bound, other checks become rules), kinds guessed from the names;
2. default values by kind;
3. ``ports/param_defaults.py``: hand-written defaults and corrections (per parameter name and per port).

Derived from it:

* ``spec.check(p)`` / ``spec.validate(p)``: the domain check with the C messages (``R(-3) < 0``); the Python
  functions call it once per evaluation, so an invalid value gives the same error as the C plugin;
* ``spec.defaults()``: the default vector; the explorer starts new contributions with it (``ff_py_*``);
* ``spec.suggest(q, I)``: values suggested for measured data: all lengths scaled together so that the largest
  is 2/q_mid (q_mid: geometric mean of the q range), then I0 (or the contrasts) matched to the data at low q.
  The explorer button *Suggest from data* does the same for any function (also C plugins: kinds guessed from the
  names) and sets the scale (or the amplitude N of the size distribution).

``python tools/write_header_defaults.py <SASfitGit/src/plugins>`` writes the defaults and domains into the
plugin headers, as a paragraph after the parameter table::

     * \par Default values and domains:
     *      R = 10 [0, inf); dummy = 0; dummy = 0; eta = 1

The SASfit GUI ignores the paragraph (it reads only the ``\b`` lines of the table); Doxygen shows it, and the
pySASfit header parser uses the values as defaults and bounds, so the C plugins in the explorer get the same
defaults as the ports.

## Magnetic ports (2026-11-23)

`ff_magnetic_shell_psi` and `ff_superparamagnetic_ff_psi` (libsasfit core) are ported with an additional
`PortSpec.terms(q, p, psi)` that returns the magnetic basis terms (NN, NM, MH, MSF, CH). The channel of a data set
is then a setting of the data set, not a separate function ({doc}`polarisation`). Without a setting the ports give
the SASfit result with their `pol` parameter; the comparison with the C functions runs as for every port.
C fix: `ff_magnetic_shell_cross` wrote ψ = π/2 into the caller's ψ parameter and read it as degrees
(sin²ψ = 7.5e-4 instead of 1); it now uses ψ = 90°.

## Writing a port

A port module defines functions `f(q, p, psi=None)` and a list `SPECS` of dictionaries with the fields of
`PortSpec` (name, intensity, parameters, psi_index, sample ranges, tolerance, C source, notes); it is registered
in `ports._load()`. Quadratures should be vectorised over q (e.g. `ports.quadrature.tanh_sinh_01`).
