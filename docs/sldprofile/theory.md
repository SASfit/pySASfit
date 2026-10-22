# Theory and implementation notes

Reference: S. Hansen, J. Appl. Cryst. 49 (2016) 856–865 (equation numbers below refer to that paper).

## Forward model

ρ(r), 0 ≤ r ≤ r_max, is represented by N piecewise-constant shells, ρ_k on [k, k+1]·r_max/N.
For a centro-symmetric scatterer the orientationally averaged intensity is the square of a real amplitude that is
linear in ρ:

| geometry | amplitude a_k(q) (exact shell integral) | I(q) |
|---|---|---|
| sphere   | 4π ∫ r² sin(qr)/(qr) dr | F² |
| cylinder (∞ long, per length) | 2π ∫ r J₀(qr) dr | (π/q) F² |
| lamella (∞, per area, symmetric) | 2 ∫ cos(qx) dx | (2π/q²) F² |

with F(q) = Σ_k a_k(q) ρ_k. In matrix form this is Hansen's eq. (11), I_i = ρᵀ C_i ρ. He builds C_i from the
p(r) kernel A_ij and Glatter's overlap integrals V_jkl (eqs. 7–10). For these symmetries C_i equals
c(q_i) a(q_i) a(q_i)ᵀ exactly, so the package uses the analytic shell integrals. That gives the same model without
the p(r) discretisation error (the paper uses N_P = 160). With smearing,
C_i = Σ_m R_im c(q'_m) a(q'_m) a(q'_m)ᵀ, where the model is evaluated on a fine q′ grid and smeared with a matrix R:

* `Resolution(dq, kernel="rician"|"gaussian", fwhm=False)`: pinhole resolution. It is the same operator as pyOZgui's
  `polydisperse_fit.Resolution`; the weight matrices agree exactly. The Rician kernel
  P(Q|Q₀) = (Q/σ²)·exp(−(Q²+Q₀²)/2σ²)·I₀(QQ₀/σ²) is the default. It describes 2-D isotropic broadening in the detector
  plane followed by radial averaging, and is evaluated with the scaled Bessel function `i0e`. The Gaussian kernel is
  its large-Q/σ limit and puts weight at Q < 0 when σ/Q is large. Other details:
  * dQ is σ (`fwhm=True` divides by 2.3548);
  * the extended grid is linspace(max(min(Q−3σ), 10⁻³Q_min), max(Q+3σ), 3M) plus the measured Q, so dQ → 0 is exact;
  * trapezoid weights, rows normalised to 1;
  * the background is not smeared.

  `RicianResolution` and `GaussianResolution` are shortcuts for the two kernels.
* `SlitResolution(c)`: I_m(q) = 2∫₀^∞ P(t) I(√(q²+t²)) dt with P(t) = exp(−t²/c²)/(c√π), as in sect. 3.1.
* `MatrixResolution(q_fine, R)`: any user-supplied smearing.

A flat background b can be fitted as an extra, unregularised parameter. This plays the role of the paper's
"free p(0)" term, because p(0) at r = 0 adds a q-independent constant.

p(r) = r^(dim−1) γ(r) (eqs. 4–5) is computed from ρ for output only. For the sphere it is exact, via the shell-pair
identity sinc(qs)·sinc(qt) = (1/2st) ∫_{|s−t|}^{s+t} x·sinc(qx) dx. For the lamella and cylinder it is computed
numerically (1-D correlation, or 2-D FFT autocorrelation).

## Representation: ρ or radial mass (`basis`)

`basis="rho"` (the paper's choice) stores constant ρ_k in each shell. A smooth, compact ρ then always gives an
amplitude F(q) with real zeros, so I = F² has sharp minima. That works for particles with a well-defined surface,
but it fails for data that decay as a power law without minima (χ²/M ≈ 50–90 in tests). The solver then adds edges
and ringing to ρ(r).

Such data need not imply polydispersity. For a Gaussian coil, inverting √(Debye) gives a positive, monotonically
decreasing ρ(r) ≈ (√2/Rg)/(2π²r²)·f(r), with f → 1 for r ≪ Rg and f → 0 at about 3–4 Rg. Its F² is the Debye
function exactly, and F has no zeros. The q⁻² tail comes from the 1/r² cusp at the centre.

`basis="mass"` stores the radial mass g = r^(dim−1)ρ. g is constant within a shell, so ρ = g/r^(dim−1) there, and the
shell amplitudes are exact:

* sphere: 4π [Si(qr)/q]
* cylinder: 2π [∫₀^{qr} J₀/q]
* lamella: same as `rho`

The regularisation acts on g, which is smooth (flat at the centre) for coil-, star- and micelle-corona-like profiles.
On simulated Debye data (Rg = 217 Å, 2 % noise) this gives χ²/M = 0.68, Rg ≈ 221–229 Å and a monotone g(r) that
matches the exact inversion. The `rho` basis on the same data gives χ²/M = 92.

Information limit: structure beyond about π/q_min is barely determined. For coils the evidence for r_max is
therefore flat beyond about 3 Rg, and Rg computed from the profile becomes sensitive to the weakly determined outer
tail at large r_max. Using `outer="zero"` and a fixed or limited r_max gives stabler Rg values. If the evidence
maximum lies at the upper end of the coarse r_max grid, `fit` extends the grid automatically (`max_extend`).

### General exponent p and its Bayesian selection

The representation is generalised to c_k = r^p ρ (any p < dim), so ρ = c_k/r^p within a shell. For p = 0 and
p = dim−1 the shell integrals are analytic. For other p they are computed by Gauss quadrature: Gauss–Jacobi with
weight r^(dim−1−p) on the first shell, where the integrand can be singular, and Gauss–Legendre on the others. The
quadrature agrees with the analytic forms to 10⁻⁷ or better.

p decides which quantity the smoothness prior favours. Different p are therefore different models, and the Bayesian
evidence can compare them (`DensityProfileFit.scan_exponent(p_values, r_max)`; GUI: **Scan p**).

On simulated star data with ρ ∝ r^(−4/3)·exp(−(r/250)⁴) and background 0.05, a scan with the background *fitted*
selects p = 4/3 by ≥ 30 in ln Ev and recovers the background as 0.0500. A slightly wrong *fixed* background (0.0513)
shifts the preferred p to the upper end of the range. So fit the background when scanning p, or fix it only from an
independent measurement.

Rg for ρ = c/r^p uses the analytic shell moments ∫r^(d+1−p)dr / ∫r^(d−1−p)dr.

### Background models (linear and nonlinear)

I_model = (smeared) c(q)F² + B(q; θ), with B a sum of terms (`pysasfit.core.background`):

| term | B(q) | parameters |
|---|---|---|
| `Constant` | c | c (fixed or fitted) |
| `PowerLaw` | A q⁻ⁿ | A > 0, n > 0 (n fixed or fitted) |
| `OrnsteinZernike` | I_L/(1+q²ξ²) | I_L > 0, ξ > 0 (ξ fixed or fitted) |
| `FunctionTerm` | any f(q, θ) | user-defined, numerical Jacobian |

The parameters do *not* have to be linear. The model is already nonlinear in ρ and is minimised by Gauss–Newton, so
each free background parameter is just another unregularised column ∂I/∂θ of the Jacobian. Positive parameters are
fitted as ln θ. In the evidence they carry a flat prior and are integrated out in the same Laplace approximation as
ρ: the background block enters ln det(αA + B), not ln det(αA). Their errors come from the same posterior covariance.
The background is evaluated at the measured q and is not smeared.

The evidence also compares background models. On coil data with an added 2·10⁻⁶ q⁻⁴ upturn, a constant-only fit
gives χ²/M = 54. With a power law and n free the fit recovers n = 4.04 ± 0.08 and Rg = 219–224 Å (true 217 Å).
Fixing n = 4 is preferred by the evidence (ln Ev −89.5 vs −93.3).

The limit is **identifiability**, not linearity. A background term with the same q-dependence as the particle
scattering cannot be separated from it. Examples are A q⁻² next to a coil profile, or q⁻⁴ next to a sharp particle
surface. The fit then puts the intensity in either one, and the evidence shows a flat direction. The q range should
contain the regime where the background dominates (e.g. the low-q upturn).

`background=True` (fitted constant) and `background_value=b₀` (fixed constant) remain as shortcuts.

The sign of ρ is undetermined by I = F². The convention used is F(0) = ∫ρ dV > 0.

## Shell grid: uniform, power law, adaptive (`grid`)

The shells can sit on any grid of normalised edges 0 = t₀ < … < t_N = 1 (r_k = r_max·t_k). The grid scales with
r_max, so a single grid serves the whole r_max scan.

* `grid=None`: uniform, as in the paper.
* `grid=("power", γ)`: t_k = (k/N)^γ, so γ > 1 is finer at the centre and γ < 1 finer towards r_max.
* `grid=array`: e.g. from `DensityProfileFit.adaptive_grid(r_max)`.

On non-uniform grids the smoothness prior is written on the actual node positions. Each row is the second
derivative on the uneven grid, scaled by −h̄²/2 (h̄ = r_max/N) and weighted by √(Δr_j/h̄). S then approximates
(h̄⁴/4)∫c″(r)²dr, i.e. smoothness *in r*, whatever the node distribution. For a uniform grid the rows are exactly
Hansen's eq. (14).

### Adaptive grid (equidistribution)

A first fit on the uniform grid, deliberately over-smoothed (α raised by e²), gives c(r) for the represented
quantity r^p ρ. The nodes are then placed so that each shell carries the same integral of

  M(r) = 1 + β |c′(r)| / ⟨|c′|⟩

For a piecewise-constant representation the error per shell is about Δr|c′|, so this equalises it. β defaults
to 3. Two limits apply:

* M is capped at 8 × its minimum.
* Shells are never narrower than min(π/q_max, r_max/N). Structure finer than π/q_max is not determined by I(q),
  and in tests refining below it lowered the evidence (star, N = 80: ln Ev −99 without the floor vs −88.5 with it).

Results (fixed r_max, ln evidence):

| data | N | uniform | adaptive |
|---|---|---|---|
| star, r^(4/3)ρ | 30 | −86.6 | **−76.9** |
| star, r^(4/3)ρ | 80 | −88.6 | −88.5 |
| coil, r²ρ | 30 | −71.1 | **−70.0** |
| coil, r²ρ | 80 | −72.5 | −75.1 |
| sphere with dip, ρ | 30 / 80 | −44.9 / −44.0 | −44.9 / −44.0 |

Adaptive grids pay off when N is small, i.e. when the uniform spacing is coarser than the data resolution. Adaptive
N = 30 then matches or beats uniform N = 80 at a fraction of the cost, with unchanged Rg. Once the uniform spacing
is at or below π/q_max, the uniform grid is already optimal.

Caveat: the adaptive grid is derived from the data, so the prior on that grid partly depends on the data.
Evidence comparisons between uniform and adaptive grids are therefore indicative, not strict.

## B-spline representation (`grid=BSplines(...)`)

Instead of N steps, the represented quantity c(r) = r^p ρ(r) can be a clamped B-spline of degree k = 3 (cubic) or
2 (quadratic) on normalised breakpoints 0 = t₀ < … < t_m = 1:

  c(r) = Σ_i c_i B_i(r/r_max),  ρ(r) = c(r)/r^p  (r ≤ r_max), N = number of coefficients c_i.

**Motivation.** Every shell boundary is a small step in ρ, and each step adds its own Porod-like tail
(I ∝ q⁻⁴ for the sphere). When q_max·r_max is large, the interference of these tails is a cheap way to follow the
noise at large q. A cubic spline is C² inside [0, r_max]. The only step left is the physical one at r_max when the
outer end is free.

**Forward model.** The amplitude stays linear in the coefficients, F(q) = Σ a_i(q) c_i, with

  a_i(q) = A_d ∫₀^r_max B_i(r/r_max) r^(d−1−p) K(qr) dr

(A_d, K as for the shells). It is integrated by Gauss quadrature on every knot span, with at least 16 nodes and at
least 4 per half period of K at q_max. The first span uses Gauss–Jacobi with the weight r^(d−1−p), so a centre cusp
(p > 0) is integrated exactly. The quadratic model, the Jacobian, the evidence, the backgrounds, the resolution and
the two-step route are unchanged. A constant profile and the mass-basis cusp agree with the closed forms to 10⁻¹³.

**End conditions** are built into the basis. They are hard constraints, unlike the soft rows of the shell
operator:

* centre: c′(0) = 0, i.e. the represented quantity is even through r = 0 (the first two clamped coefficients are
  tied);
* outer end `"free"`: c(r_max) is free, giving one step to zero at r_max (a sharp surface);
* outer end `"zero"`: c(r_max) = 0 (the last basis function is dropped).

The number of knot spans is m = N + 1 + [outer = zero] − k.

**Smoothness prior.** S = (h³/4) ∫₀¹ c″(x)² dx with h = 1/N. This is the continuum limit of Hansen's eq. (14) for N
points. It is computed exactly from the B-splines (Gauss–Legendre on every span), for any knot placement. The same
ridge as for the shells removes the null space (constants). With **max-ent**, c_i > 0 implies ρ > 0, because
B-splines are non-negative. The prior shape is evaluated at the Greville abscissae.

**Knots.** `BSplines(3)` gives uniform knots. `BSplines(3, ("power", γ))` sets t_j = (j/m)^γ. `adaptive_grid` on a
spline fitter returns `BSplines(3, breakpoints)`. Its breakpoint density is

  M(x) = 1 + β √|c″(x)| / ⟨√|c″|⟩

(curvature instead of the slope used for the shells), with the same cap (8×) and the same minimum span
min(π/q_max, r_max/N).

**Comparison** at fixed r_max (best α; χ²/M of the lower/middle/upper third of the q points; N_g = number of good
parameters):

| data | representation | χ² thirds | ln Ev | N_g |
|---|---|---|---|---|
| star r^(4/3)ρ + bg, q_max·r_max ≈ 180 | 40 shells | 0.63 / 0.63 / **0.35** | −80.8 | 19.0 |
| | 20 shells | 1.37 / 0.87 / 25.3 | −467 | 16.4 |
| | 12 cubic | 0.63 / 0.65 / 0.61 | **−79.4** | 10.9 |
| | 20 cubic | 0.63 / 0.63 / 0.55 | −81.1 | 14.8 |
| | 20 quadratic, adaptive | 0.63 / 0.63 / 0.51 | **−78.7** | 13.8 |
| coil r²ρ + bg, q_max·r_max ≈ 315 | 40 shells | 0.48 / 0.63 / 1.19 | −68.6 | 9.6 |
| | 20 cubic | 0.48 / 0.67 / 1.17 | −68.5 | 8.6 |
| | 20 quadratic, adaptive | 0.48 / 0.68 / 1.17 | **−67.7** | 8.5 |
| sphere with dip, ρ, q_max·r_max ≈ 8 | 40 shells | 0.84 / 1.07 / 0.97 | **−44.4** | 3.8 |
| | 20 cubic | 0.85 / 1.07 / 0.97 | −46.0 | 3.8 |

* Where q_max·r_max is large (star), 40 shells follow the noise in the upper third of q (χ² 0.35 at N_g = 19).
  12–20 spline coefficients fit with 11–15 effective parameters, an upper-third χ² of 0.5–0.6, and the same or
  higher evidence. 20 shells are too coarse there.
* For small q_max·r_max (Hansen's examples) all representations give the same fit and Rg. The shells then have
  slightly higher evidence (by 1–2 in ln Ev).
* In synthetic tests with the true r_max (core–shell 1/0.5 and a Gaussian profile, R = 50, q ≤ 0.45, 1 % noise,
  6 seeds) the ratio χ²(fit)/χ²(true model) in the upper third was 0.99 for both shells and splines. There the
  smoothness prior alone already suppresses the steps. The spline profile was closer to the truth for the
  Gaussian profile (relative error 0.018 vs 0.026).

Recommendation: use the B-spline representation, with 12–25 coefficients, when q_max·r_max is large or the high-q
residuals look too good. Compare it with the shells by the evidence.

## Regularisation

Φ(ρ) = α S(ρ) + χ²/2 (eq. 12; the χ²/2 convention of Hansen 2000 is used, so that the evidence formula below holds).

* **Smoothness** (eq. 14): S = Σ (f_j − (f_{j−1}+f_{j+1})/2)². At r = 0 the profile is continued symmetrically
  (f_{−1} = f_1). At r_max the end point is either free (`outer="free"`, the paper's default for ρ) or pulled to
  zero (`outer="zero"`, the paper's "ρ(r_max) = 0"). With a free end, constant profiles cost nothing, so a ridge of
  10⁻⁶·tr(L)/N keeps A non-singular for the evidence.
* **Maximum entropy** (eq. 15): S = Σ f ln(f/m) − f + m, with prior m. The prior shape is a function of r/r_max or an
  array, and its amplitude is fitted to the data. Iterations run in z = ln ρ, which keeps ρ > 0 and uses the
  entropy metric.

## Minimisation

The model is quadratic in ρ, so Φ is minimised with a Levenberg–Marquardt iteration (Gauss–Newton Hessian
JᵀJ + α∇∇S, Jacobi-scaled). Hansen points out that the quadratic problem has **local minima** (sect. 4), and two
measures address this:

1. **α-continuation.** For every r_max, α starts about e⁶ above the natural scale tr(B)/tr(A), where the solution is
   essentially the homogeneous profile or the prior. It is then lowered in steps of Δln α = 0.5, and each solution
   starts from the previous one.
2. **Extra starting points.** Besides the homogeneous profile on [0, r_max], a homogeneous step profile ρ = c for
   r < R ≤ r_max is used, with R and c set by least squares. Without it, a too large r_max converges to a wrong
   branch (χ²/M ≈ 70 in tests). Whenever χ²/M > 2 during the path, both starts are retried.

## Hyper-parameters (Bayesian)

For fixed (α, r_max) the Gaussian approximation of the evidence is

  ln P(α, r_max) = −αS − χ²/2 − ½ ln det(αA + B) + ½ ln det(αA),

with A = ∇∇S and B = JᵀJ. When a background is fitted, b gets a flat prior and is integrated out. The determinants
are evaluated after Jacobi scaling, which matters for max-ent, where A = diag(1/f).

* α: the evidence maximum along the continuation path, refined with a bounded 1-D search.
* r_max: the posterior P(r_max) ∝ ∫ P(α, r_max) d ln α on a coarse grid (default 0.7–1.5 × the Guinier estimate),
  followed by a fine grid around the maximum. The result is reported as mean ± std, plus the best value.
* Error bars on ρ(r), Rg and b: the evidence-weighted average over all solutions on the fine grid. Each solution
  adds its posterior covariance (αA + B)⁻¹ through Monte-Carlo draws. This follows Hansen's IFT error estimate.

**Run time.** Most of the time of a scan used to go into r_max values that cannot fit the data (χ²/M of hundreds):
their α path never meets the stop rule (χ² < 1.5 M), runs to the smallest α with restarts, and the
Levenberg–Marquardt fits there converge slowly. Three measures (2026-10-21) remove that work without changing the
result:

* an α path also ends when χ² stalls (relative change < 2 % over three steps) while ln Ev has dropped by the stop
  value below its maximum on the path (`abandon_plateau`);
* no restarts at an r_max whose ln Ev is 50 below the best already found (`abandon_margin`); the r_max are processed
  outwards from the expected maximum so that this reference is known early;
* the fine r_max scan stays within the coarse points that carry posterior weight, plus one step (`fine_support`).

| example | N | before | after | r_max, Rg |
|---|---|---|---|---|
| star, r^(4/3)ρ + bg | 80 | 58 s | 8 s | 431 ± 53 → 423 ± 52; Rg 177.2 → 177.3 |
| star | 40 | 2.8 s | 1.9 s | 389 ± 20 → 391 ± 22; Rg 177.0 |
| coil, r²ρ + bg | 80 | 170 s | 63 s | identical |
| coil | 40 | 38 s | 15 s | identical |
| sphere with dip | 80 / 40 | 6 / 4 s | 6 / 4 s | identical |

(The small r_max shift for the star comes from the different fine-grid points, well within the posterior width.)
A cubic B-spline with 20 coefficients fits the coil in 8 s. `n_workers` > 1 runs the r_max scan in several
processes; it only pays off with several free cores and slow fits (on a 2-core test machine it was slower).

Initial r_max comes from a Guinier fit: sphere √(5/3)·Rg; cylinder √2·R_c, from ln(qI) vs q²; lamella √3·R_t, from
ln(q²I) vs q².

## Two-step method (Glatter): IFT, then deconvolution

`pysasfit.sldprofile.twostep` (GUI: *method → two-step*) offers Glatter's original route for comparison with the
direct estimate:

1. **IFT** I(q) → p(r) on n_r points of [0, D_max] (sphere: I = 4π∫p sin(qr)/(qr) dr; cylinder cross-section:
   I = (π/q)·2π∫p J₀(qr) dr; lamella thickness: I = (2π/q²)·2∫p cos(qr) dr; the same normalisation as `pddf`),
   with the instrumental resolution. Tikhonov with Hansen (2000) eq. 19 (curvature, p(0) = p(D_max) = 0), α by
   maximum evidence; D_max over the r_max scan range (× 2) and, optionally, a flat background by maximum evidence.
   For the lamella p = γ(r) does not vanish at r = 0 (γ(0) = ∫ρ² dx), so γ is solved on the interior of
   [−D_max, D_max] and reported symmetrised on r ≥ 0.
2. **Deconvolution** (Glatter 1981): p(r_j) = ρᵀV_jρ with V_j built from the same thin sub-shells as `pddf`, the
   pair-distance distribution of two shells integrated exactly over each p(r) bin. `DeconvolutionFit` is
   `DensityProfileFit` with this forward model, so shells, bases, grids, smoothness / max-ent and α by evidence are
   identical; r_max = D_max/2 is fixed by step 1. The p(r) errors enter with their standard deviations
   (`errors="diagonal"`, default since 2026-10-21, as in DECON) or with the full posterior covariance of the IFT
   (`"covariance"`, not recommended, see below).

The result shows the IFT fit of I(q), the I(q) recomputed from the deconvolved ρ(r) (with χ²/M against the data),
the IFT p(r) with errors and the p(r) of the deconvolved ρ (overlay *p(r)*), and the D_max posterior as an r_max
posterior. On the simulated step profiles of the tests (ρ = 1 to r = 3, 0.4 to r = 5, 1 % noise) both routes give
Rg within 0.3 % of each other and profile errors of 10–14 %; the two-step route was 4–12× faster there, because the
r_max scan is a scan of linear IFTs.

**Which p(r) errors (2026-10-21).** The posterior covariance of the regularised IFT is nearly singular (condition
number ≈ 4·10⁷ for 60 points on q ∈ [0.004, 0.385], D_max ≈ 630). In its small-variance directions p(r) is fixed by
the smoothness prior, not by the data, and the IFT estimate there is biased by that prior. Weighting with the full
covariance makes these directions dominate: even the *exact* p(r) of the true particle differs from the IFT p(r)
by χ²/M = 6.3 (Mahalanobis), against 0.2 with the standard deviations. Two-step fits of simulated data with the
same q range and errors (2 % + 0.002, D_max from the evidence):

| particle | errors | χ²/M step 2 | χ²/M of I(q) from ρ |
|---|---|---|---|
| homogeneous sphere R = 300 | covariance / diagonal | 14.5 / **1.4** | 1057 / 20 |
| core–shell sphere 200/300 | covariance / diagonal | 170 / **1.2** | 4721 / 77 |
| bimodal lognormal spheres R = 60 and 180, σ = 0.2 | covariance / diagonal | 206 / 8.1 | 237 / 224 |

With diagonal errors a single centro-symmetric particle is deconvolved consistently (χ²/M ≈ 1), and a polydisperse
ensemble is not (χ²/M ≈ 8): the summary then marks step 2 (threshold 3). The reason is physical: the intensities
of different particles add, I = Σ N_i F_i², which in general is not the square of *one* real amplitude, i.e. the
p(r) of a polydisperse ensemble is not the self-convolution of one radial profile. The χ²/M of I(q) recomputed from
the deconvolved ρ(r) is large even for single particles: p(r) on 60 regularised points keeps little of the
information at high q (q_max·R ≈ 115 here), which is the main argument of Hansen (2016) for the direct route.

`examples/studio/data/test.dat` (the bimodal sphere benchmark of Kohlbrecher & Breßler 2022) behaves like the
polydisperse simulation: IFT χ²/M = 0.57 with D_max ≈ 640, but step 2 χ²/M ≈ 11 (diagonal; 6–70 with the
covariance), and the deconvolution alone improves steadily for r_max well above D_max/2 (χ²/M = 11 at r_max = 321 = D_max/2, 3.1 at 360, 1.3 at 480), i.e. it needs a "particle" larger than the largest distance in p(r) -- impossible for one particle. A density profile is not a
meaningful description of these data; the size distribution (inversion tool, sphere kernel) is.

**Grids in the two-step route (2026-10-21).** Non-uniform shell or knot grids in step 2 (power law towards the
centre or the edge, adaptive) did not improve the deconvolution in simulations on the q range of the benchmark
(core–shell sphere, a density dip, a Gaussian profile; R = 300, q ∈ [0.004, 0.385]): the ln Ev of step 2 changed by
≤ 1 and the profile error by ≤ 0.01, because the resolution of p(r) is uniform in r. What matters is the number of
IFT points: with 60 points p(r) is under-sampled (Shannon number q_max·D_max/π ≈ 78), and 100 points lowered the
profile error from 0.105 to 0.086 (core–shell), 0.099 to 0.084 (dip) and 0.32 to 0.03 (Gaussian; with 60 points the
shell fit fell into a wrong solution there, a cubic B-spline did not: 0.036).

**Number of IFT points: the Shannon number (2026-10-21).** By default `n_r` is the Shannon number
N_s = ⌈q_max·D_max/π⌉ of the largest D_max of the scan (at least 15; `shannon_points`). The IFT kernel integrates
p(r) as a piecewise-linear function (hat functions, `quadrature="hat"`, on a sub-grid with ≥ 4·q_max·Δr/π steps
per interval) instead of the rectangle rule, so wide intervals are integrated exactly. Step 1 then needs no more
than N_s points: in simulations (R = 300, q ∈ [0.004, 0.385], N_s = 74, four profiles, three noise realisations)
the IFT fit was the same from 0.5 to 3 × N_s (χ²/M unchanged, ln Z falling slowly with more points). The usual
advice of 2–3 × N_s comes from the rectangle rule, which needs a fine grid to integrate the kernel.

**Weight of p(r) in step 2.** With diagonal errors, step 2 counts every p(r) value as an independent measurement,
but the IFT p(r) holds only about N_s independent values whatever the number of points. More IFT points, or p(r)
resampled on a finer grid, therefore weighted p(r) too strongly: step 2 chose too little smoothing and the profile
became worse (profile error, mean of three noise realisations, IFT points 0.5 / 1 / 2 / 3 × N_s: homogeneous
sphere 0.05 / 0.18 / 0.36 / 0.37, density dip 0.20 / 0.22 / 0.34 / 0.32; with p(r) resampled 4× even 0.28–0.53).
Step 2 now multiplies the p(r) errors by √(M/N_s) when it gets M > N_s points (`scale_errors=True`; the summary
lists the factor). The result then hardly depends on the number of IFT points (homogeneous 0.05 / 0.17 / 0.21 /
0.18, dip 0.20 / 0.22 / 0.23 / 0.22, core–shell 0.14 throughout, Gaussian 0.04 throughout); at the default
(n_r = N_s, no resampling: `dec_oversample=1`) the factor is 1. `resample_pddf` / `dec_oversample` remain
available but gained nothing once the errors were scaled.

The weight is not always optimal: for a small particle measured to high q·R (R = 5, q ≤ 2.5, q_max·D_max/π = 8,
15 IFT points) step 2 at the default smooths more than the data like (I(q) from ρ: χ²/M ≈ 5; unscaled p(r)
resampled 4×: 1.2), with a similar profile error (0.135 vs 0.094). The weighting of a regularised p(r) in a
second fit has no exact answer (the full covariance is the exact one, but nearly singular, see above). Compare
with the direct method, which has no such step. A cubic B-spline in step 2 remains the recommended
representation.

**Glatter's own choices.** Glatter (1981) represents ρ by N equidistant steps of width ΔR and states that ΔR
"should correspond to the resolution of the p(r) function", i.e. equal the knot distance of the IFT (D_max /
number of splines); the number of steps then follows from R_max. His examples use the fixed ITP setting of 20
splines: sphere R = 75 Å, h ≤ 0.1 Å⁻¹, D_max = 200 Å, 10 steps of 10 Å (about 3–4 × the Shannon number for the
IFT, ΔR ≈ 0.3 π/h_max); Glatter & Hainisch (1984): two-shell sphere R = 105 Å, h ≤ 0.202 Å⁻¹, D_max = 240 Å, 10
steps of 12 Å (≈ 1.3 × N_s, ΔR ≈ 0.8 π/h_max). Neither paper relates the grid to the Shannon number, and the
"10 steps vs 2 steps" fits of 1984 were equally good within the resolution. For the weights, Glatter (1981,
eqs. 13–16) states that the p(r) errors are correlated and the full covariance of the IFT is the proper weight,
with the standard deviations as an approximation; no correction for the number of points is made. Instabilities
of the step heights grow with N (most for spheres) and are removed by a smoothness term λK chosen by the
point-of-inflexion method -- the counterpart of the smoothness prior with α by evidence used here.

Tested (2026-10-21, three noise realisations each): Glatter's grid -- n_r/2 equidistant steps, i.e. step width =
IFT point spacing -- gives the same result as 30 shells or 30 cubic-spline coefficients (profile errors within
0.00–0.03; R = 5 sphere 0.128 in both cases, I(q) from ρ χ²/M 6.1 vs 5.1). Doubling the IFT points (n_r = 2 N_s,
N_s steps) helps the R = 5 sphere a little (0.125, χ²/M 2.9; without the error scaling 0.108, 2.1) but makes the
R = 300 homogeneous sphere and the density dip worse (0.18 → 0.34–0.39, 0.22 → 0.33–0.34). The step-2 grid is
therefore not what limits the two-step route; how strongly step 2 trusts p(r) is.

**Smoothing of step 2 from the measured I(q) (default since 2026-10-21, `alpha_from="intensity"`).** Instead of the
evidence of the p(r) fit, step 2 takes the smoothest ρ(r) on its α path whose I(q), computed with the direct
forward model (with resolution and the step-1 background), fits the *measured* data within one standard deviation
of χ²:

    chi2_I(alpha) <= chi2_ref + tol * sqrt(2 M),    tol = 1 (intensity_tolerance), M = number of data points

with χ²_ref the first minimum of χ²_I along the path from large to small α (a second, deeper minimum can appear at
much smaller α, where ρ oscillates while p(r) and I(q) hardly change; the minimum alone is therefore no criterion).
The crossing is refined by bisection in ln α. The profile error band is the posterior of that single α. The
summary lists the chosen ln α next to the one the p(r) evidence would give; `TwoStepResult.alpha_scan` holds
χ²_I/M and ln Ev along the path, the threshold and the α of both criteria (computed with either `alpha_from`; shown
in the optional GUI panel *Two-step: χ²(α) of I(q)*). `alpha_from="evidence"` restores the previous behaviour.

Profile error and χ²/M of I(q) from ρ (three noise realisations, 30 shells; first block with D_max from the
evidence as in practice, second with r_max fixed at the true value):

| case | D_max scanned: evidence | D_max scanned: I(q) | r_max fixed: evidence | r_max fixed: I(q) |
|---|---|---|---|---|
| R = 5 sphere, q ≤ 2.5 | 0.132 / 5.6 | **0.103 / 1.7** | 0.128 / 5.1 | **0.097 / 1.7** |
| R = 5 cylinder | 0.152 / 1.31 | 0.153 / 1.24 | 0.132 / 1.25 | 0.131 / 1.25 |
| R = 5 lamella | 0.134 / 0.98 | 0.135 / 0.97 | 0.131 / 0.97 | 0.132 / 0.97 |
| R = 300 core–shell | 0.164 | **0.151** | 0.139 | **0.132** |
| R = 300 homogeneous | 0.249 (±0.14) | **0.156 (±0.02)** | 0.177 (±0.21) | **0.036** |
| R = 300 density dip | 0.363 (±0.15) | **0.262 (±0.01)** | 0.221 | 0.204 |
| R = 300 Gaussian | 0.067 | **0.049** | **0.045** | 0.070 |

(R = 300: χ²/M of I(q) ≈ 0.94–1.04 in all cases.) With D_max from the evidence the I(q) criterion was better or
equal in every case and much less scattered; with r_max fixed it over-smoothed the smooth Gaussian profile. Cubic
B-splines in step 2 gave the same picture. The two-step route now uses the measured data in both steps; p(r)
remains the intermediate that fixes D_max and makes step 2 a quadratic problem.

Not available in the two-step route: power-law and Ornstein–Zernike background terms (step 1 takes a flat
background only) and the adaptive grid.

## Validation (tests and examples)

* Homogeneous sphere, R = 7.5 nm, 5 % noise, slit smearing (c = 0.1 nm⁻¹), q ∈ [0.12, 1] nm⁻¹: best r_max = 7.51 nm,
  Rg = 5.82 ± 0.01 nm (true value 5.81). With an added background b = 0.01 I(q₁), b is recovered to 0.1 %.
* Inhomogeneous profiles (density dip at 3 nm, R = 5 nm, 1 % + 10⁻⁴ noise) for all three geometries: r_max is
  within 1–4 % of 5 nm and χ²/M ≈ 0.95. As in the paper, the smoothness constraint fills in the minimum partly;
  max-ent with a uniform prior pulls the ends towards the prior.
* B-spline representation (`tests/sldprofile/test_bspline.py`): a constant profile, the radial-mass cusp and Rg are
  exact to 10⁻¹³; fits of homogeneous spheres, core–shell cylinders and lamellae, max-ent, adaptive and power-law
  knots and the two-step route with both end conditions; the star example fits with fewer effective parameters than
  40 shells.
* Two-step route (`tests/sldprofile/test_twostep.py`): IFT kernels against the direct model for all geometries; a
  single core–shell sphere deconvolves with χ²/M < 2, bimodal polydisperse spheres are flagged.
* Run-time shortcuts (`tests/sldprofile/test_scan_speed.py`): same r_max, Rg and posterior with and without them;
  the parallel r_max scan gives the serial result.

## Differences from the paper

* The exact rank-one amplitude form replaces the V_jkl overlap-integral route. The model is identical, with no
  p(r) discretisation.
* Hansen's iterative algorithm (after Glatter 1981) is not described in detail, so it is replaced by
  Levenberg–Marquardt with α-continuation.
* The r_max posterior shows a tail towards large r_max when the outer end is free: the profile can then go to zero
  inside r_max. The paper reports narrower posteriors (e.g. 7.508 ± 0.015 nm). Using `outer="zero"` or a narrower
  `r_max_range` gives a sharper estimate.
* The width c of the slit profile in the paper's Fig. 1 is garbled in the PDF text. The examples use c = 0.1 nm⁻¹.
