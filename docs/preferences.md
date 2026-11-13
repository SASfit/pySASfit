# Preferences (Settings → Preferences…)

All tools share one preferences dialog: **Settings → Preferences…** in the launcher (`pysasfit-gui`, it opens on
the page of the tool in front) and in each stand-alone window (`pysasfit-sldprofile`, `pysasfit-inversion`,
`pysasfit-oz`, `pysasfit-sasred`, `pysasfit-toolbox`, `pysasfit-atsas`, `pysasfit-density`). It holds the numerical settings that are not on the tools' main panels.

* **Save** writes them to the preferences file, and they are the defaults at every later start:
  `%APPDATA%\pySASfit\settings.json` (Windows), `~/.config/pysasfit/settings.json` (Linux/macOS), or the file named by the
  environment variable `PYSASFIT_SETTINGS`. Only values that differ from the factory value are written.
* Values are read when a calculation starts (Run / Solve / Compute / Fit), so a change applies to the next one.
* **Bold** labels mark changed values; hovering shows the explanation; *Factory values* restores one tab or all.
* Library functions keep their documented defaults; scripts are not affected by the preferences file.
* Session files still take precedence where they store a value (e.g. the numerical settings of the polydisperse
  OZ tab).

Factory values are the values the code used before the preferences existed, with **one deliberate change**:
the λ grid of the Bayesian-evidence solvers of the size-distribution tool is now centred on tr(B)/tr(2LᵀL)
instead of the squared median singular value of the kernel. With the old rule the whole grid lies many decades
below the evidence maximum whenever most singular values are at the noise floor (any finely sampled p(r)), and an
unregularised, oscillating p(r) was returned. On `tests/inversion/data/test.dat` both rules give the same λ; the
old rule can be selected ("median").

## Adjustable settings

One page per tool, and **Shared numerics** for methods used by several tools or tabs (deflation of
OZ roots, the analytic MSA/RMSA solution, the global optimisers NLopt and bumps, the start values of the
background models). Each entry gives the label shown in the dialog, the key in `settings.json`, the
factory value, the allowed range, and what the value does and when to change it. The same explanation
appears in the dialog: as a tooltip, and in the panel below the tabs when you point at a setting or
click into its field. The texts are kept in `src/pysasfit/core/settings_help.py`.

### SLD profile

#### Regularisation parameter α (direct fit and deconvolution)

- **ln α start (above ln α₀)** — `sldprofile.opt.log_alpha_hi` · factory: **6** · range: -50 … 100  
  Start of the α path, as ln α − ln α₀ with α₀ = tr(B)/tr(A) at the starting profile (B: data Hessian, A: Hessian of the regulariser). The path runs from strong smoothing (this end) to weak smoothing. Raise it if the evidence curve (ln evidence vs ln α) is still rising at its left end.

- **ln α end (relative to ln α₀)** — `sldprofile.opt.log_alpha_lo` · factory: **-16** · range: -200 … 50  
  End of the α path, relative to ln α₀. The path normally stops earlier, once the evidence has dropped by the stop criterion. Lower it only if the evidence curve is still rising at its right end when the path ends.

- **ln α step** — `sldprofile.opt.alpha_step` · factory: **0.5** · range: 0.01 … 5  
  Step in ln α along the path. Each step is one regularised fit, warm-started from the previous one. It is also the integration step of the marginalisation over α (posterior of r_max, error bands). Smaller steps: smoother evidence curve, more fits.

- **stop when ln evidence has dropped by** — `sldprofile.opt.stop_drop` · factory: **25** · range: 1 … 1000  
  The α path stops once ln evidence has fallen this far below its running maximum (and the data are fitted, see the next entry). 25 means a factor e^25 below the best α -- such solutions carry no weight in the averages. Larger values only cost time.

- **stop only when χ² < factor · M** — `sldprofile.opt.stop_chi2` · factory: **1.5** · range: 0.1 … 100  
  The early stop is allowed only when χ² < factor × M (M = number of points), i.e. once the data are actually fitted. Prevents stopping on the falling side of a first, spurious evidence maximum.

- **restart from other start profiles when χ² > factor · M** — `sldprofile.opt.retry_chi2` · factory: **2** · range: 1 … 1e+06  
  Guard against a wrong branch: while χ² > factor × M, the fit at the current α is repeated from the other starting profiles (homogeneous, truncated step, flat coefficients) and the better one is kept. Lower it if fits with a too large r_max get stuck.

- **… every n-th α step** — `sldprofile.opt.retry_every` · factory: **4** · range: 1 … 1000  
  How often the restart check above is made: every n-th α step. 1 = at every step (slowest, safest).

- **precision of the best ln α** — `sldprofile.opt.alpha_xatol` · factory: **0.001** · range: 1e-08 … 1  
  After the path, ln α is refined between the two neighbours of the best path point by a bounded one-dimensional maximisation of the evidence; this is its precision in ln α.

#### Minimiser (Levenberg–Marquardt)

- **maximum iterations** — `sldprofile.opt.max_iter` · factory: **300** · range: 1 … 1e+06  
  Maximum Levenberg–Marquardt iterations per regularised fit (fixed α and r_max). Fits are warm-started along the α path, so they normally converge in a few iterations.

- **relative tolerance** — `sldprofile.opt.tol` · factory: **1e-10** · range: 1e-16 … 0.01  
  Convergence test of each regularised fit: stop when the objective αS + χ²/2 decreases by less than tol × max(|objective|, 1) in one iteration.

- **initial damping** — `sldprofile.opt.lm_lambda0` · factory: **0.001** · range: 1e-12 … 1e+06  
  Initial Levenberg–Marquardt damping λ (relative to the scaled Hessian). It is reduced ÷3 after a successful step and increased ×4 after a rejected one, so the start value matters little.

- **smoothness ridge (× tr/N)** — `sldprofile.opt.ridge` · factory: **1e-06** · range: 0 … 1  
  Small multiple of the identity (× tr(A)/N) added to the smoothness matrix A. With a free outer end A is singular (a constant has zero curvature); the ridge makes it invertible. It enters log det A and therefore the evidence -- keep it small.

- **shortest step profile tried as start (fraction of N)** — `sldprofile.opt.truncated_min` · factory: **0.25** · range: 0.01 … 1  
  Starting profiles: besides the homogeneous one, homogeneous steps ρ = c for r < R are tried with R on the shell grid, from this fraction of N shells up to N. They rescue fits with a too large r_max.

#### Result and error bands

- **posterior draws per solution** — `sldprofile.fit.n_draws` · factory: **40** · range: 1 … 10000  
  Number of random draws from the posterior covariance per kept solution, used for the error bands of ρ(r), the radial mass, Rg and the background. More draws: smoother bands, more time.

- **random seed of the draws** — `sldprofile.fit.seed` · factory: **0** · range: 0 … 2.14748e+09  
  Seed of the random draws above. A fixed seed makes repeated runs give identical error bands.

- **drop solutions with relative evidence weight below** — `sldprofile.opt.keep_weight` · factory: **1e-06** · range: 0 … 0.5  
  Solutions whose evidence weight is below this fraction of the best one are dropped from the averages (they contribute nothing and only cost time).

- **points of the output r grid** — `sldprofile.opt.n_out` · factory: **300** · range: 10 … 100000  
  Number of cells of the r grid on which ρ(r), the radial mass and their errors are reported (display and _rho.dat). Does not change the fit.

- **output r range (× largest r_max)** — `sldprofile.opt.r_out_factor` · factory: **1.05** · range: 1 … 5  
  The output r grid extends to factor × the largest r_max among the kept solutions, so the outer edge of the profile is visible.

- **times the r_max scan may be extended upwards** — `sldprofile.fit.max_extend` · factory: **6** · range: 0 … 100  
  If the evidence maximum lies at the upper end of the r_max scan, the scan is extended upwards; this is how often. Needed for coil-like profiles that reach far beyond the Guinier estimate.

- **Guinier range q·Rg <** — `sldprofile.opt.guinier_qrg` · factory: **1.3** · range: 0.3 … 3  
  Upper limit q·Rg of the Guinier fit that gives the start value of r_max (sphere √(5/3) Rg, cylinder √2 R_c, lamella √3 R_t). 1.3 is the usual validity limit for compact particles.

- **points added per upward extension of the r_max scan (× coarse)** — `sldprofile.opt.extend_fraction` · factory: **0.5** · range: 0.05 … 5  
  Number of r_max points added per upward extension of the scan, as a fraction of the coarse points (at least 3).

- **fine r_max grid half width: posterior s.d. ×** — `sldprofile.opt.fine_sd` · factory: **4** · range: 0.5 … 20  
  After the coarse r_max scan a fine scan is made around the maximum; its half width is max(fine_sd × posterior s.d., fine_steps × coarse step). This is the first term.

- **fine r_max grid half width: at least coarse steps ×** — `sldprofile.opt.fine_steps` · factory: **1.5** · range: 0.1 … 20  
  Second term of the fine-scan half width (see above), in coarse grid steps; it keeps the fine scan from collapsing when the coarse posterior is very narrow.

- **fine r_max grid: only where the coarse posterior >** — `sldprofile.opt.fine_support` · factory: **1e-06** · range: 0 … 0.1  
  The fine r_max scan is limited to the coarse r_max whose posterior exceeds this fraction of the maximum, plus one coarse step on each side (no limit on a side where the weight reaches the end of the coarse grid). Without it, a broad coarse posterior sends the fine scan to r_max values that cannot fit the data and only cost time (star example, N = 80: 58 s → 8 s). 0 switches the limit off.

- **end an α path when χ² stalls (relative change over 3 steps) <** — `sldprofile.opt.abandon_plateau` · factory: **0.02** · range: 0 … 0.5  
  An α path also ends when the data are not fitted (χ² above the stop value) but χ² has changed by less than this fraction over the last three α steps while ln evidence has already dropped by the stop value below its maximum on the path. Such an r_max cannot fit the data; its evidence maximum is already on the path, and the remaining small-α fits are the slowest of the scan. 0 = never.

- **no restarts at r_max with ln evidence below the best by** — `sldprofile.opt.abandon_margin` · factory: **50** · range: 0 … 1e+06  
  At an r_max whose best ln evidence is this far below the best of the r_max already done, no restarts from the other starting profiles are tried (they only help near the evidence maximum). 0 = always.

- **parallel processes for the r_max scan** — `sldprofile.opt.n_workers` · factory: **0** · range: 0 … 256  
- **r_max scan: start each α path from the neighbouring r_max** — `sldprofile.opt.warm_start` · factory: **off** · range: on/off  
  Number of processes for the r_max scan (0 or 1: serial). The r_max nearest to the centre is done first, the others in parallel. Only worthwhile with several free cores and slow fits (large N): process start-up, copying the fitter and the loss of the running best evidence cost time; on a 2-core test machine the parallel scan was slower than the serial one. Falls back to serial when the fit cannot be copied to other processes (e.g. a max-ent prior or background given as a lambda).

#### Grids and scans

- **adaptive grid: pilot over-smoothing (e-folds in α)** — `sldprofile.adaptive.smoothing` · factory: **2** · range: 0 … 20  
  Adaptive grid: the first pass uses the best uniform-grid solution with α raised by this many e-folds, so the grid follows the real structure of the profile and not its noise.

- **adaptive grid: max. ratio coarsest / finest shell** — `sldprofile.adaptive.max_ratio` · factory: **8** · range: 1 … 1000  
  Adaptive grid: the finest shell is at most this many times thinner than the coarsest one. Limits how strongly the grid concentrates at steep parts of the profile.

- **adaptive grid: min. shell width (0 = min(π/q_max, r_max/N))** — `sldprofile.adaptive.min_width` · factory: **0** · range: 0 … 1e+06  
  Adaptive grid: smallest allowed shell width in r units. 0 = automatic, min(π/q_max, r_max/N): structure finer than π/q_max is not determined by the data.

- **exponent scan step** — `sldprofile.gui.p_scan_step` · factory: **0.2** · range: 0.01 … 1  
  Step of the exponent scan ('Scan p'), which compares representations r^p·ρ(r) for p = 0 … d−0.2 by their evidence (4/3 is always included for spheres).

- **max-ent 'previous result' prior floor (× max)** — `sldprofile.gui.prior_floor` · factory: **0.001** · range: 1e-12 … 0.5  
  Max-ent with the 'previous result' prior: the prior is the last profile, clipped below at this fraction of its maximum (max-ent needs a strictly positive prior).

- **max-ent 'hollow shell' prior inner level** — `sldprofile.gui.hollow_level` · factory: **0.05** · range: 1e-06 … 1  
  Max-ent with the 'hollow shell' prior: prior level inside r_in, relative to 1 in the shell.

#### Two-step method: IFT (step 1) and deconvolution (step 2)

- **IFT: ln α from (relative to ln tr B/tr C)** — `sldprofile.ift.log_alpha_lo` · factory: **-40** · range: -200 … 0  
  Two-step method, step 1 (IFT): lower end of the α grid, ln α relative to ln(tr B / tr C) (B: data Hessian, C: smoothness matrix). The evidence maximum is searched on this grid, then refined.

- **IFT: ln α to** — `sldprofile.ift.log_alpha_hi` · factory: **10** · range: -50 … 100  
  Two-step method, step 1 (IFT): upper end of the α grid, ln α relative to ln(tr B / tr C). This end is the strongly smoothed side. Raise it if the IFT evidence still rises at the upper end of the grid (very noisy or very few data).

- **IFT: α grid points** — `sldprofile.ift.n_alpha` · factory: **101** · range: 5 … 10000  
  Two-step method, step 1 (IFT): number of points of the α grid between the lower and upper end. The grid only brackets the evidence maximum, which is then refined, so more points rarely change the result; use more if the evidence curve has several maxima.

- **IFT: precision of ln α** — `sldprofile.ift.alpha_xatol` · factory: **0.0001** · range: 1e-10 … 1  
  Two-step method, step 1 (IFT): the evidence maximum found on the α grid is refined by a bounded 1-D search until ln α is known to this precision. The default is far below the width of the evidence peak; there is no reason to change it.

- **IFT background search: below I_min (× scale)** — `sldprofile.ift.bg_below` · factory: **1** · range: 0 … 100  
  IFT with 'fit' background: the flat background is chosen by maximum evidence within [I_min − below·s, I_min + above·s], s = max(|I_min|, median σ). This is 'below'.

- **IFT background search: above I_min (× scale)** — `sldprofile.ift.bg_above` · factory: **0.5** · range: 0 … 100  
  Two-step method, IFT with 'fit' background: the flat background is chosen by maximum evidence within [I_min − below·s, I_min + above·s], s = max(|I_min|, median σ). This is 'above', in units of s. Increase it if the chosen background sits at the upper end of the interval.

- **IFT background precision (× scale)** — `sldprofile.ift.bg_xatol` · factory: **0.0001** · range: 1e-12 … 0.1  
  Two-step method, IFT with 'fit' background: precision of the evidence search for the flat background, in units of s = max(|I_min|, median σ). Far below the statistical uncertainty of the background; no reason to change.

- **IFT smoothness penalty** — `sldprofile.twostep.penalty` · factory: **Hansen (2000) eq. 19, curvature** · range: "hansen_eq19" (Hansen (2000) eq. 19, curvature), "hansen_printed" (Hansen (2000) as printed, 1st difference)  
  Smoothness prior of the IFT. Hansen (2000) eq. 19: curvature, Σ[p_j − (p_{j−1}+p_{j+1})/2]² with p(0) = p(D_max) = 0. 'As printed': the matrix given in the text of that paper, which is a first-difference penalty. The normalised evidence prefers eq. 19 on our test data.

- **deconvolution: sub-shells per shell in V(r)** — `sldprofile.twostep.n_sub` · factory: **6** · range: 1 … 64  
  Deconvolution (step 2): each shell is split into this many thin sub-shells to build the matrices V(r) of p(r) = ρᵀV(r)ρ. 6 reproduces the direct I(q) to ~1e-4.

#### Quadrature

- **Gauss nodes per shell (custom exponent p)** — `sldprofile.geo.n_gl` · factory: **16** · range: 2 … 256  
  For a custom exponent p (r^p·ρ represented, p ≠ 0, d−1) the shell amplitudes are integrated numerically: Gauss–Jacobi in the first shell (weight r^(d−1−p)), Gauss–Legendre in the others, with this many nodes each.

- **output p(r): sub-shells per shell** — `sldprofile.geo.pddf_n_sub` · factory: **8** · range: 1 … 128  
  Sub-shells per shell for the p(r) shown as overlay and written to _pr.dat (not used in the fit).

- **output p(r) of cylinders: FFT raster** — `sldprofile.geo.pddf_raster` · factory: **256** · range: 32 … 4096  
  Cylinders: the p(r) of the cross-section is computed by a 2-D FFT autocorrelation on a square raster of this size (display and _pr.dat only).

#### Instrumental resolution

- **pinhole: kernel width (± n·σ_q)** — `sldprofile.res.n_sigma` · factory: **3** · range: 1 … 10  
  Pinhole resolution: the model is evaluated on an extended q grid reaching ± n·σ_q beyond the measured range, and the kernel is truncated there. 3 covers 99.7 % of a Gaussian.

- **pinhole: extended q grid oversampling** — `sldprofile.res.oversample` · factory: **3** · range: 1 … 50  
  Pinhole resolution: density of the extended q grid, as a multiple of the number of data points.

- **slit: Gauss–Hermite nodes (half range)** — `sldprofile.res.slit_nodes` · factory: **16** · range: 2 … 200  
  Slit smearing: number of Gauss–Hermite nodes on the positive half of the slit profile. The default integrates a Gaussian slit to better than 1e-6; increase it only for very long slits combined with sharp features in the model.

### Size distribution / p(r)

#### Bayesian evidence: regularisation parameter λ

- **centre of the λ grid** — `inversion.opt.evidence_lambda_scale` · factory: **tr(B) / tr(2 LᵀL)  (default)** · range: "trace" (tr(B) / tr(2 LᵀL)  (default)), "median" (median singular value² (old))  
  Centre of the λ grid of the Bayesian-evidence solvers. 'trace': tr(B)/tr(2LᵀL), the scale at which data and prior curvature are equal -- independent of how the r grid is sampled. 'median': median singular value of the kernel, squared; the rule before 2026-10-11. With many r points most singular values are at the noise floor (~1e-15) and the 'median' grid lies far below the evidence maximum, giving an unregularised p(r).

- **λ grid: decades below the centre** — `inversion.opt.evidence_decades_below` · factory: **12** · range: 0 … 60  
  λ grid: decades below the centre (trace rule). The evidence maximum of a well-determined p(r) usually lies a few decades below.

- **λ grid: decades above the centre** — `inversion.opt.evidence_decades_above` · factory: **4** · range: 0 … 60  
  Bayesian evidence λ selection (trace rule): the λ grid extends this many decades above the centre tr(B)/tr(2LᵀL). Above the centre the solution is over-smoothed, so a few decades suffice; increase it if the selected λ lies at the upper grid end.

- **λ grid points** — `inversion.opt.evidence_points` · factory: **97** · range: 5 … 10000  
  Bayesian evidence λ selection (trace rule): number of λ grid points over the whole range (below + above). The maximum is refined after the grid search, so the default is ample; more points cost time linearly.

- **refine λ between grid points** — `inversion.opt.evidence_refine` · factory: **on** · range: on/off  
  Refine λ between the two neighbours of the best grid point by a bounded 1-D maximisation of the evidence (otherwise the grid point itself is used).

- **χ² target may lower λ at most by factor** — `inversion.opt.max_lambda_reduction` · factory: **0.001** · range: 1e-12 … 1  
  With 'χ²ᵣ = 1' ticked, λ is adjusted from the evidence value to reach the target. It may be raised freely but lowered at most by this factor: reaching χ²ᵣ = 1 only by (almost) removing the regularisation means the target is wrong for the data.

- **estimated error scale s: lower limit** — `inversion.opt.noise_scale_lo` · factory: **0.01** · range: 1e-06 … 1  
  Error bars 'overall scale estimated': σ_i = s·ΔI_i with s chosen by maximum evidence within [lower, upper]. Lower limit.

- **estimated error scale s: upper limit** — `inversion.opt.noise_scale_hi` · factory: **100** · range: 1 … 1e+06  
  Error bars 'overall scale estimated': σ_i = s·ΔI_i with s chosen by maximum evidence within [lower, upper]. This is the upper limit of s. Raise it if s ends at this limit (the given errors are much too small).

- **χ²ᵣ target (when ticked)** — `inversion.gui.target_chi2` · factory: **1** · range: 0.01 … 100  
  Target of the 'χ²ᵣ = 1' option of the Bayesian-evidence solver (discrepancy principle).

#### EM, MaxEnt, GCV

- **EM: maximum iterations per trial** — `inversion.opt.em_max_iterations` · factory: **10000** · range: 10 … 1e+07  
  EM + smoothing and signed-kernel EM: maximum iterations for one trial smoothing value h (each is Biggs–Andrews accelerated).

- **EM / MaxEnt: tolerance** — `inversion.opt.em_tol` · factory: **1e-08** · range: 1e-16 … 0.01  
  EM and EM + MaxEnt: one solve stops when the relative change of the distribution between iterations falls below this value. Smaller values give a converged result at more iterations; larger values speed up the h or λ search at the price of a slightly less converged distribution.

- **EM: start of the smoothing search h** — `inversion.opt.em_h_start` · factory: **0.3** · range: 1e-06 … 1  
  EM + smoothing: the smoothing parameter h (0 … 1) is searched by the discrepancy principle, starting here and decreasing.

- **EM: smallest h** — `inversion.opt.em_h_floor` · factory: **1e-09** · range: 0 … 0.1  
  EM + smoothing: smallest h tried. Below it the fit is treated as unable to reach χ²ᵣ = 1 and the fallback (L-curve or least smoothing) is used.

- **signed EM: smallest h** — `inversion.opt.em_general_h_floor` · factory: **1e-07** · range: 0 … 0.1  
  Signed-kernel EM: smallest smoothing value h tried. Below it the fit is treated as unable to reach χ²ᵣ = 1 and the fallback (L-curve or least smoothing) is used. Lower it only if the data are very precise and the fallback is triggered too early.

- **L-curve: points** — `inversion.opt.lcurve_points` · factory: **25** · range: 3 … 1000  
  L-curve fallback of the EM methods (used when χ²ᵣ = 1 cannot be reached): number of smoothing values h sampled to locate the corner of the L-curve. More points locate the corner more precisely but cost one EM solve each.

- **L-curve: h reduction per point** — `inversion.opt.lcurve_reduction` · factory: **1.2** · range: 1.01 … 100  
  L-curve: factor between successive h values (h, h/f, h/f², …). With 25 points and 1.2 the curve spans about two decades.

- **EM+MaxEnt: smallest λ** — `inversion.opt.maxent_lambda_min` · factory: **1e-09** · range: 1e-30 … 1  
  EM + MaxEnt: smallest entropy weight λ of the grid; the λ giving χ²ᵣ closest to 1 is chosen.

- **EM+MaxEnt: largest λ** — `inversion.opt.maxent_lambda_max` · factory: **1** · range: 1e-12 … 1e+06  
  EM + MaxEnt: largest entropy weight λ of the λ grid; the λ giving χ²ᵣ closest to 1 is chosen. Raise it if the chosen λ lies at this end (the data are fitted with χ²ᵣ < 1 even at the strongest entropy weight).

- **EM+MaxEnt: λ grid points** — `inversion.opt.maxent_points` · factory: **24** · range: 2 … 1000  
  EM + MaxEnt: number of λ values on the geometric grid between smallest and largest λ. Each point is one complete MaxEnt solve; more points select λ more finely but cost proportionally more time.

- **EM+MaxEnt: maximum iterations** — `inversion.opt.maxent_max_iterations` · factory: **50000** · range: 10 … 1e+07  
  EM + MaxEnt: maximum iterations of one solve at fixed λ. If solves end at this limit (see the log) the selected λ may be off; increase it, or relax the EM / MaxEnt tolerance.

- **EM+MaxEnt adaptive prior: smoothing width σ²** — `inversion.opt.maxent_sigma2` · factory: **1** · range: 1e-06 … 1e+06  
  EM + MaxEnt, adaptive prior: the prior is the current solution smoothed with a Gaussian of this variance (in grid points²).

- **signed EM MaxEnt: λ start** — `inversion.opt.signed_maxent_lambda_start` · factory: **1e-05** · range: 1e-30 … 1  
  Signed-kernel EM with MaxEnt: start of the bidirectional discrepancy search of the entropy weight.

- **signed EM MaxEnt: smallest λ** — `inversion.opt.signed_maxent_lambda_floor` · factory: **1e-12** · range: 0 … 1  
  Signed-kernel EM with MaxEnt: smallest entropy weight the bidirectional discrepancy search may reach. If the search ends here, even almost no entropy cannot fit the data to χ²ᵣ = 1 (errors too small or model inadequate).

- **signed EM MaxEnt: largest λ** — `inversion.opt.signed_maxent_lambda_ceil` · factory: **1** · range: 1e-12 … 1e+06  
  Signed-kernel EM with MaxEnt: largest entropy weight the bidirectional discrepancy search may reach. If the search ends here, the data are over-fitted even at the strongest entropy weight (errors probably too large).

- **Hansen MaxEnt: χ²ᵣ target** — `inversion.opt.hansen_maxent_target_chi2` · factory: **1** · range: 0.01 … 100  
  Hansen's MaxEnt: λ² is chosen by the discrepancy principle so that the reduced χ² equals this target. 1 trusts the error bars; values > 1 give smoother distributions when the errors are known to be underestimated.

- **Hansen MaxEnt: λ² search start** — `inversion.opt.hansen_maxent_start` · factory: **1** · range: 1e-12 … 1e+12  
  Hansen's MaxEnt: start value of the search for λ². The search first brackets the discrepancy root by multiplying or dividing by the step factor, then refines it, so the start affects speed, not the result.

- **Hansen MaxEnt: λ² search step factor** — `inversion.opt.hansen_maxent_factor` · factory: **4** · range: 1.01 … 1000  
  Hansen's MaxEnt: factor between successive λ² values while bracketing the discrepancy root. Larger factors bracket faster; smaller ones are safer when χ²ᵣ(λ²) is not monotonic.

- **Hansen MaxEnt L-BFGS-B: ftol** — `inversion.opt.lbfgs_ftol` · factory: **1e-16** · range: 0 … 0.01  
  Hansen's MaxEnt is minimised with bound-constrained L-BFGS-B: relative function tolerance.

- **Hansen MaxEnt L-BFGS-B: gtol** — `inversion.opt.lbfgs_gtol` · factory: **1e-12** · range: 0 … 0.01  
  Hansen's MaxEnt is minimised with bound-constrained L-BFGS-B: the minimisation stops when the largest projected-gradient component is below this value. Smaller values give a better-converged distribution at more iterations.

- **Hansen MaxEnt L-BFGS-B: history (maxcor)** — `inversion.opt.lbfgs_maxcor` · factory: **30** · range: 3 … 1000  
  Hansen's MaxEnt, L-BFGS-B: number of stored correction pairs, i.e. the memory of the Hessian estimate. More pairs usually mean fewer iterations for many bins, at a small extra cost per iteration.

- **EM / MaxEnt: start vector level** — `inversion.opt.em_x0` · factory: **1e-06** · range: 1e-30 … 1e+06  
  EM and MaxEnt start from a flat vector at this level. EM converges to the same fixed point from any positive start; the level matters for the iteration count only.

- **EM+MaxEnt constant prior: floor** — `inversion.opt.maxent_prior_floor` · factory: **1e-06** · range: 1e-30 … 1  
  EM + MaxEnt with constant prior: the prior level is max(mean(I)/n, floor), n = number of bins. The floor only matters for data with very small or negative mean intensity, where it keeps the entropy defined.

- **discrepancy root search: absolute tolerance** — `inversion.opt.root_xtol` · factory: **1e-06** · range: 0 … 1  
  Discrepancy-principle searches (EM h, MaxEnt λ) end with a Brent root search, stopping when |x − x*| ≤ xtol + rtol·|x*|. The absolute 1e-6 is the historical value; for parameters far below 1e-6 it resolves the root only loosely -- set xtol = 0 and rtol = 1e-6.

- **discrepancy root search: relative tolerance** — `inversion.opt.root_rtol` · factory: **8.88e-16** · range: 8.88178e-16 … 0.1  
  Discrepancy-principle searches (EM h, MaxEnt λ) end with a Brent root search, stopping when |x − x*| ≤ xtol + rtol·|x*|. This is rtol; the default is the machine-precision limit of Brent's method. Set rtol = 1e-6 together with xtol = 0 for parameters far below 1e-6.

- **background fit: max. evaluations** — `inversion.opt.background_maxfev` · factory: **10000** · range: 10 … 1e+07  
  Maximum function evaluations of the least-squares fit of the background backgr + c·q^(−α) (scipy curve_fit). Increase it if the background fit reports that it did not converge.

- **GCV: decades below s_min** — `inversion.opt.gcv_decades_below` · factory: **3** · range: 0 … 30  
  Tikhonov / general Tikhonov with GCV: the λ grid runs from s_min·10^(−below) to s_max·10^(above) (s: singular values of the weighted kernel).

- **GCV: decades above s_max** — `inversion.opt.gcv_decades_above` · factory: **1** · range: 0 … 30  
  Tikhonov / general Tikhonov with GCV: the λ grid runs from s_min·10^(−below) to s_max·10^(above) (s: singular values of the weighted kernel). This is 'above'; raise it if GCV selects the largest λ.

- **GCV: grid points** — `inversion.opt.gcv_points` · factory: **100** · range: 5 … 10000  
  Tikhonov / general Tikhonov with GCV: number of λ values on the grid between its lower and upper end. GCV is evaluated cheaply from the SVD, so more points cost little.

- **TSVD: largest truncation rank** — `inversion.opt.tsvd_k_max` · factory: **40** · range: 2 … 100000  
  Truncated SVD: largest truncation rank k tried by GCV (capped by the number of bins and data). Raise it if GCV selects this largest k, i.e. the data support more singular components.

#### Instrumental resolution (4th data column σ_Q)

- **kernel width (± n·σ_Q)** — `inversion.res.n_sigma` · factory: **3** · range: 1 … 10  
  Resolution from the 4th column σ_Q: the kernel is evaluated on an extended q grid ± n·σ_Q beyond the data and the smearing kernel is truncated there.

- **extended q grid oversampling** — `inversion.res.oversample` · factory: **3** · range: 1 … 50  
  Resolution from the 4th column σ_Q: density of the extended q grid on which the kernel is evaluated before smearing, as a multiple of the number of data points. Increase it for sharp features (narrow distributions, large particles) where the default grid under-samples the kernel.

- **4th column is a FWHM (not σ_Q)** — `inversion.res.fwhm` · factory: **off** · range: on/off  
  Tick if the 4th column of your files is the full width at half maximum instead of σ_Q (FWHM = 2.3548 σ).

#### Grid, errors, bootstrap

- **integrate the kernel on a fine grid when Δr > factor · π/q_max** — `inversion.gui.fine_grid_factor` · factory: **1.1** · range: 0.1 … 100  
  The kernel is integrated on a fine grid (p linear or cubic between the r points) when the r spacing exceeds factor × π/q_max, the period of the fastest kernel oscillation; otherwise it is evaluated at the points.

- **suggested r points = factor × Shannon number** — `inversion.gui.r_points_factor` · factory: **1** · range: 0.1 … 20  
  After loading, the number of r points is suggested as factor × the Shannon number (q_max − q_min)·r_max/π. 1 = the number of independent parameters the data contain.

- **error model 'poly': polynomial degree** — `inversion.gui.poly_degree` · factory: **3** · range: 1 … 6  
  Error model 'local cubic fits': degree of the polynomial in ln I vs ln q fitted through each point and its neighbours.

- **error model 'poly': half width (points)** — `inversion.gui.poly_half_width` · factory: **2** · range: 1 … 50  
  Error model 'local cubic fits': number of neighbours on each side of a point used in the polynomial fit of ln I vs ln q. The scatter about the fit gives the error estimate; wider windows give smoother but possibly biased estimates where I(q) bends sharply.

- **bootstrap random seed (−1 = different every run)** — `inversion.gui.bootstrap_seed` · factory: **-1** · range: -1 … 2.14748e+09  
- **bootstrap: parallel worker processes (1 = serial, 0 = automatic)** — `inversion.gui.bootstrap_workers` · factory: **1** · range: 0 … 64  
  Seed of the Monte-Carlo (bootstrap) error estimate. −1 = different every run; a fixed value makes the error bands reproducible.

### Ornstein–Zernike

#### OZ solvers (all tabs)

- **default max. iterations (accelerated solvers)** — `oz.solvers.max_iterations` · factory: **600** · range: 10 … 1e+07  
  Default iteration limit of the accelerated OZ solvers (Anderson, MDIIS, KINSOL, Biggs–Andrews), used wherever a tab does not set its own limit.

- **default max. iterations (Picard; Mann: ÷ a)** — `oz.solvers.max_iterations_picard` · factory: **5000** · range: 10 … 1e+07  
- **Picard / Mann: stop unphysical or stalled runs early** — `oz.solvers.picard_early_stop` · factory: **on** · range: on/off  
- **Picard / Mann: unphysical when min g(r) <** — `oz.solvers.picard_gmin` · factory: **-0.5** · range: -1e+06 … 0  
  Iteration limit N of the Picard / Mann solver for a = 1 (plain Picard), used wherever a tab does not set its own limit, in particular always in the OZ solver tab. With damping a < 1 the solver runs up to ceil(N/a) iterations, because each damped step covers only a fraction a of the way (5000 at a = 0.2 gives 25000). Picard converges linearly and slowly near the spinodal, hence the higher limit than for the accelerated solvers. Example: hard spheres PY at φ = 0.53 need about 4200/a iterations (a ≤ 0.6), so N ≈ 4500 or more is needed there; the factory value 5000 (raised from 3000) covers this, also for the damped Picard / Mann verifier.

- **convergence criterion (relative progress)** — `oz.solvers.convergence` · factory: **1e-12** · range: 1e-16 … 0.001  
  Convergence criterion of the fixed-point solvers: relative change of Γ(r) between iterations.

- **SUNDIALS KIN_FP: Anderson depth** — `oz.solvers.kinsol_maa` · factory: **5** · range: 0 … 100  
  SUNDIALS KIN_FP: depth of the Anderson acceleration (number of previous iterates combined). SASfit's C solver uses 5.

- **Biggs-Andrews: extrapolation order (sign = step formula)** — `oz.solvers.biggs_maa` · factory: **2** · range: -10 … 10  
  Biggs–Andrews vector extrapolation: order |n|; the sign selects the step-size formula.

- **MDIIS: history length** — `oz.solvers.mdiis_history` · factory: **5** · range: 1 … 100  
  MDIIS: number of previous residuals used in the DIIS extrapolation. Longer histories accelerate more but can become ill-conditioned; reduce it if MDIIS oscillates, increase it for slow convergence of dense states.

#### OZ solver tab

- **verification: max. relative |ΔS|** — `oz.solve.verify_tolerance` · factory: **0.001** · range: 1e-08 … 1  
  OZ solver tab, 'verify': the result is cross-checked by an independent solver from a cold start; they must agree on S(q) to this relative accuracy, otherwise multiple roots are reported.

- **Hankel transform type** — `oz.solve.transform_type` · factory: **auto (4 for one component)** · range: "auto" (auto (4 for one component)), "1" (1 (DST-I)), "4" (4 (DST-IV))  
  Discrete sine transform of the Hankel transform. Type 4 (DST-IV) puts the grid at half-integer points and is second order for one component; type 1 is first order but works for mixtures. auto = 4 for one component, 1 for mixtures. Force one for comparisons.

- **independent residual check: max. ||T(x)-x||/||x||** — `oz.solve.residual_tolerance` · factory: **1e-06** · range: 1e-15 … 1  
  After every OZ solve the closure's fixed-point map T is applied once more to the returned solution x, and the result counts as converged only if ||T(x) - x|| / ||x|| is at most this value -- whatever the solver itself reported. This catches solvers that claim success on a non-solution (e.g. KIN_FP returning NaN for polydisperse hard-core Yukawa under HNC, or a stopping test fooled by a stalled iteration). Converged solves give 1e-16 to 1e-11, so 1e-6 rejects only real failures. Lower it for stricter checking; raise it only if well-converged results are being rejected. A failed check marks the run [NOT CONVERGED].

- **verification: damping a of the Picard / Mann verifier** — `oz.solve.verify_mann_alpha` · factory: **0.5** · range: 1e-06 … 1  
  Damping a of Picard / Mann when it serves as the second verifier (after Biggs–Andrews). Undamped Picard diverges above φ ≈ 0.42 for hard spheres, which is also where Biggs–Andrews stops converging, so dense states had no verifier at all. Damped, Picard / Mann converges there (hard spheres PY, φ = 0.53: a ≤ 0.6). Its iteration limit is the Picard limit / a, so a smaller a costs time. Must satisfy 0 < a ≤ 1. The verifier is skipped when the primary solver is Picard / Mann itself (same algorithm family).

- **Picard / Mann: start value of the damping a** — `oz.solve.mann_alpha` · factory: **1** · range: 1e-06 … 1  
  Value that the 'damping a' field of the OZ solver tab shows at start. The Picard / Mann solver iterates x <- (1 - a) x + a T(x): a = 1 is plain Picard, a < 1 is Mann's damped iteration, slower (roughly 1/a times more iterations) but convergent where undamped Picard oscillates or diverges, e.g. dense hard spheres above φ ≈ 0.42 or strongly attractive states. Must satisfy 0 < a ≤ 1. Typical values 0.2 to 0.8; at a fixed state there is an upper bound above which the iteration oscillates (hard spheres PY at φ = 0.53: a ≤ 0.6). The iteration limit is the Picard limit divided by a. Ignored by all other solvers.

- **consistent closure parameter: scan steps** — `oz.consistency.scan_steps` · factory: **28** · range: 4 … 10000  
  'Find consistent parameter' (RY, HMSA, MHNC, BPGG, CJVM, BB): number of scan points of the closure parameter before the root of the compressibility mismatch is refined.

- **consistent closure parameter: tolerance** — `oz.consistency.tolerance` · factory: **1e-05** · range: 1e-12 … 0.1  
  'Find consistent parameter' (RY, HMSA, MHNC, BPGG, CJVM, BB): tolerance of the root refinement of the compressibility mismatch in the closure parameter. Smaller values pin the parameter more precisely at more OZ solves.

#### Polydisperse (any potential) tab

- **grid exponent k (N = 2^k − 1, or 2^k for type 4)** — `oz.generic.grid_exponent` · factory: **12** · range: 4 … 20  
  Polydisperse tab: radial grid N = 2^k − 1 points (type 1) or 2^k (type 4). Start value of the tab's Numerical settings; a session file overrides it.

- **points per mean diameter** — `oz.generic.points_per_sigma` · factory: **100** · range: 4 … 100000  
  Polydisperse tab: grid points per mean diameter. With the grid size it fixes the radial range (N / points per σ diameters) and so the lowest q.

- **max. iterations** — `oz.generic.max_iterations` · factory: **8000** · range: 10 … 1e+07  
  Polydisperse tab: iteration limit of each OZ solve (including the solves during a fit and the α search). Raise it for dense or strongly attractive states that converge slowly.

- **transform type** — `oz.generic.transform_type` · factory: **1 (DST-I)** · range: "1" (1 (DST-I)), "4" (4 (DST-IV))  
  Polydisperse tab: transform type of the OZ solver. 1 is required for most mixtures (the components are transformed as a matrix); 4 is a single-component variant and is only valid for one component.

- **fit: max. model evaluations (per start)** — `oz.generic.fit_max_nfev` · factory: **200** · range: 5 … 1e+06  
  Polydisperse tab, Fit: maximum model evaluations of least_squares (each start of the multi-start fit gets this budget). Every evaluation is a full OZ solve.

- **fit: starts of 'least_squares (n starts)'** — `oz.generic.n_starts` · factory: **10** · range: 2 … 1000  
  Polydisperse tab, 'least_squares (n starts)': number of local fits from scattered start points, to find out whether the minimum is unique.

- **fit: relative finite-difference step** — `oz.generic.diff_step` · factory: **0.001** · range: 1e-08 … 0.1  
  Polydisperse tab, Fit: relative finite-difference step of the Jacobian. Too small and the OZ solver's own convergence noise dominates the derivative.

- **consistent α search: lower bracket** — `oz.generic.alpha_lo` · factory: **1e-06** · range: 1e-12 … 10  
  Polydisperse tab, 'find consistent α': lower end of the bracket of the closure parameter.

- **consistent α search: upper bracket** — `oz.generic.alpha_hi` · factory: **100** · range: 1e-06 … 1e+06  
  Polydisperse tab, 'find consistent α': upper end of the bracket of the closure parameter. If the search ends at this end the closure cannot reach thermodynamic consistency inside the bracket.

- **consistent α search: tolerance** — `oz.generic.alpha_tol` · factory: **0.0002** · range: 1e-10 … 0.1  
  Polydisperse tab, 'find consistent α': tolerance of the root search of the compressibility mismatch in α. Smaller values cost more OZ solves.

- **α accepted as consistent when |residual| < value · χ⁻¹** — `oz.generic.alpha_accept` · factory: **0.001** · range: 1e-10 … 1  
  α is reported as consistent when |compressibility mismatch| < value × χ⁻¹ (relative test; χ⁻¹ is of order 10 for a dense fluid). Otherwise the closest bracket end is used and flagged.

- **resolution: kernel width (± n·σ_Q)** — `oz.generic.res_n_sigma` · factory: **3** · range: 1 … 10  
  Polydisperse tab, resolution from σ_Q: the smearing kernel is evaluated on an extended Q grid ± n·σ_Q beyond the data and truncated there. 3 covers 99.7 % of a Gaussian; raise it for very wide resolution near steep features.

- **resolution: extended Q grid oversampling** — `oz.generic.res_oversample` · factory: **3** · range: 1 … 50  
  Polydisperse tab, resolution from σ_Q: density of the extended Q grid on which the model is evaluated before smearing, as a multiple of the number of data points. Increase it for sharp structure-factor peaks.

#### RY Polydisperse Yukawa tab

- **α search: lower bracket** — `oz.ry.alpha_lo` · factory: **0.02** · range: 1e-08 … 10  
  RY Polydisperse Yukawa tab: lower end of the bracket of the Rogers–Young mixing parameter α searched for thermodynamic consistency.

- **α search: upper bracket** — `oz.ry.alpha_hi` · factory: **3** · range: 0.001 … 10000  
  RY Polydisperse Yukawa tab: upper end of the bracket of the Rogers–Young mixing parameter α searched for thermodynamic consistency. If the search ends here, consistency is not reached in the bracket.

- **α search: tolerance** — `oz.ry.alpha_tol` · factory: **0.002** · range: 1e-10 … 0.1  
  RY Polydisperse Yukawa tab: tolerance of the root search for the consistent α. Smaller values cost more OZ solves.

- **α accepted as consistent when |residual| <** — `oz.ry.alpha_accept` · factory: **0.001** · range: 1e-10 … 1  
  α is reported as consistent when the absolute pressure-route mismatch is below this value.

- **size classes: range ± n s.d. (log-normal, Schulz-Zimm)** — `oz.ry.nsig` · factory: **5** · range: 1 … 12  
  RY tab: the size classes of a log-normal distribution span ± n standard deviations (in ln R). Schulz-Zimm/Gamma and Beta use their exact Gauss rules and ignore it.

#### Mixture validation tab

- **reject solves with residual >** — `oz.mixture.accept_residual` · factory: **0.001** · range: 1e-12 … 1  
  Mixture validation tab: a numerical OZ solve is rejected when max|G(Γ) − Γ| exceeds this value.

### SANS reduction

#### Beam centre (masked centre of gravity of the direct-beam run)

- **disc radius [px]** — `sasred.beamcentre.radius_px` · factory: **8** · range: 1 … 64  
  Radius of the disc around the current estimate whose counts enter the centre of gravity of the direct-beam run. The disc is re-centred until the shift is below the tolerance. Larger than the beam spot (about 2-3 FWHM), smaller than the distance to other bright features.

- **tolerance [px]** — `sasred.beamcentre.tol_px` · factory: **0.1** · range: 0.0001 … 5  
  The iteration of the masked centre of gravity stops when the disc moves by less than this. The statistical uncertainty of the centre is usually far smaller (0.004 px on the Vifor runs).

- **iterations** — `sasred.beamcentre.max_iter` · factory: **50** · range: 1 … 10000  
  Largest number of re-centring steps of the beam-centre search; reached only if the beam spot is at the edge of the disc or of the detector.

#### Default reduction parameters (starting values on the Reduction panel)

- **beamstop first guess: half width (square) or radius [px]** — `sasred.mask.beamstop_radius_px` · factory: **6** · range: 0 … 64  
  First guess of the beamstop shadow: a new detector image starts with the standard border plus a square of this half width (or a disc of this radius, see the shape) around the beam centre, as beamstop pixels of the mask. It is only a start: adjust it with the mask tools or replace it by Suggest. Until the mask is edited, the guess follows the beam centre and the size.

- **beamstop first guess: shape** — `sasred.mask.beamstop_shape` · factory: **square** · square, disc  
  Shape of the first-guess beamstop: square (the SANS-I beamstops are square) or disc.

- **number of q bins** — `sasred.qbins.n` · factory: **60** · range: 2 … 2000  
  Start value of the number of q bins of the radial / sector average. Empty bins are dropped, so the curve can have fewer points.

- **q-bin spacing** — `sasred.qbins.spacing` · factory: **logarithmic** · range: "log" (logarithmic), "lin" (linear)  
  Start value of the q-bin spacing: logarithmic (constant Δq/q, the usual choice for SANS) or linear (constant Δq, e.g. for peaks at large q).

- **normalise by** — `sasred.normalise_by` · factory: **monitor 1 (Poisson error propagated)** · range: "monitor" (monitor 1 (Poisson error propagated)), "time" (counting time)  
  Start value of the normalisation: monitor 1 (its Poisson error is propagated as a systematic scale shared by all points) or counting time (no monitor error).

- **units of wavelength and q** — `sasred.units` · factory: **nm and 1/nm** · range: "nm" (nm and 1/nm), "A" (Å and 1/Å)  
  Units of wavelength and q on the panels, in the plots and in the saved files (NXcanSAS Q units attribute, ASCII header). The reduction itself works in Å and 1/Å, so results do not depend on this choice. Also switchable on the Reduction panel.

- **colour scale of the detector image** — `sasred.detector.scale` · factory: **log(1+I)** · range: "linear" (linear), "log1p" (log(1+I)), "symlog" (symlog (linear below 1 count)), "asinh" (asinh)  
  Start value of the colour scale of the detector image: linear; log(1+I) (zero counts stay finite); symlog (linear below 1 count, logarithmic above); asinh (smooth version of symlog).

#### Mask suggestion (Suggest in the detector panel)

- **standard mask: outer rows and columns** — `sasred.mask.border_px` · factory: **2** · range: 0 … 20  
  The standard mask: this many outer rows and columns on every side of the detector (bit 'border'). It is the mask of a new detector image and the base of Suggest and Reset.

- **beamstop: drop below this fraction of the rim level** — `sasred.mask.drop` · factory: **0.3** · range: 0.01 … 0.95  
  Beamstop suggestion: the shadow is the connected region below drop x rim level that reaches the beam centre; the rim level is the 90th percentile of the median-filtered counts within the search radius. Lower it if the suggestion covers too much, raise it for a beamstop that transmits.

- **beamstop: search radius around the beam centre [px]** — `sasred.mask.search_radius_px` · factory: **20** · range: 3 … 200  
  Beamstop suggestion: radius around the beam centre in which the rim level is determined; it should include the bright scattering around the beamstop.

- **beamstop: grow the shadow by [px] (penumbra)** — `sasred.mask.grow_px` · factory: **1** · range: 0 … 10  
  Beamstop suggestion: the shadow is grown by this many pixels to cover the penumbra at its edge, where the counts are already up but not yet trustworthy.

- **beamstop: suggested shape** — `sasred.mask.suggest_shape` · factory: **rectangle around the shadow** · rectangle, shadow  
  Beamstop suggestion: rectangle masks the whole bounding rectangle of the shadow found in the data (square beamstops, no ragged edge); shadow masks only the pixels below the threshold.

- **masked pixels on the detector image** — `sasred.mask.display` · factory: **mask colours** · mask colours, own grey scale, hidden  
  How the masked pixels are drawn on the detector image: in the mask colours (with the opacity below), on their own grey scale with its own colour bar (to see what is behind the mask), or hidden (only the outline). The colour scale of the image always uses the unmasked pixels only.

- **opacity of the mask colours [%]** — `sasred.mask.opacity` · factory: **80** · range: 10 … 100  
  Opacity of the mask colours over the detector image. The black-and-white outline of the masked area is always drawn, so the mask stays visible at low opacity.

- **also suggest hot pixels** — `sasred.mask.hot` · factory: **on** · range: on/off  
  Suggest also isolated hot pixels: more than 5 x the median of their 5 x 5 neighbourhood and 10 Poisson standard deviations above it. Next to the beamstop they usually mean beam leaking past its edge.

#### Reduction table

- **fill the reduction table when a folder is opened** — `sasred.table.autofill` · factory: **on** · range: on/off  
  When a raw-data folder is opened, the reduction table is filled automatically: one tab per geometry (detector distance, wavelength, collimation), the runs grouped by sample, the roles taken from the sample names and, with 'check repeated runs', the repeated runs compared and outliers switched off. Geometries with few runs or without water / empty cell are marked with '?' (measured for tests?) and can be removed with the × of their tab. Off: the table stays as it is until 'Fill from runs'.

#### Detector distance

- **offset added to the recorded distance [mm]** — `sasred.detector.distance_offset_mm` · factory: **53** · range: -500 … 500  
  Effective sample-detector distance (to the detection layer) = recorded distance + this offset, used for q and the solid angle; the recorded distance stays in the run table, the grouping and the selection. Factory value 53 mm: water measured at 8 m and 2 m with the same collimation (Vifor, Sept. 2026) was 4 % apart per steradian, which a common offset of +53 mm explains (the mean conversion depth in the 3He is only about 0.75 cm at 6 Å, so most of it is where the distance is measured). To be measured again, by hand or with water at several distances and the same collimation (pysasfit.sasred.core.water.fit_distance_offset). Once the instrument configuration writes the corrected distance, limit the offset with the date below or set it to 0. The offset is noted in every reduction; its uncertainty is not propagated.

- **apply the offset to runs started before** — `sasred.detector.distance_offset_before` · factory: **(empty)** · text  
  Date (YYYY-MM-DD) from which the files already contain the corrected distance: the offset above is applied only to runs started before it. Empty: the offset is applied to all runs.

#### Transmission runs

- **beam stop out of the beam beyond [mm]** — `sasred.transmission.beamstop_out_mm` · factory: **100** · range: 1 … 2000  
  Transmission runs are measured with the beam stop moved out of the direct beam, in y (on SANS-I typically bsy = -200 mm or lower) or in x, or taken out of the detection area. A run whose beam stop is farther than this distance from the beam centre (on the detector) counts as a transmission run, whatever the attenuator: usually an attenuator is needed against saturation, but with very long wavelengths and collimations the direct beam can be measured without one. The beam stop motors are tied to detector pixels by its shadow in the scattering runs, the beam centre comes from the direct-beam runs, per detector distance and wavelength (gravity may need a realignment at long distances and wavelengths). Runs where the position and the image disagree are marked with '?' in the run table.

- **warn above this dead-time factor** — `sasred.transmission.deadtime_warn` · factory: **1.05** · range: 1 … 2  
  The direct beam concentrates the count rate: above this non-paralysable dead-time factor (1 / (1 - rate τ), τ = 0.66 µs on SANS-I) a transmission run is reported as close to saturation and a (stronger) attenuator is recommended. The factor is corrected in any case.

#### Detector efficiency

- **efficiency archive file** — `sasred.efficiency.archive` · factory: **efficiency_archive.h5 next to the preferences file** (the dialog shows the full path) · file, with a browse button  
  HDF5 file in which the detector-efficiency maps are archived, one entry per map named after the date of its water measurement, to follow the detector over time. Empty: efficiency_archive.h5 next to the pySASfit preferences file.

#### SANS-LLB detector geometry

- **central detector: offset of the detection layer [mm]** — `sasred.llb.central_dz_mm` · factory: **0** · range: -500 … 500  
  SANS-LLB: offset of the detection layer of the central detector from the recorded distance (central_detector/distance), used for q and the solid angle. The SANS-I offset (Detector distance) does not apply to SANS-LLB.

- **left panel: offset dx along posx [mm]** — `sasred.llb.left.dx_mm` · factory: **0** · range: -1000 … 1000  
  SANS-LLB left wide-angle panel: correction of the pixel positions written by NICOS (left_data/posx, posy; the same nominal geometry as GRASP's SINQ_SANS_LLB.ini). Pixel position = (posx + dx, posy + dy) at the distance left_detector/distance + dz; the beam position in that frame comes from the beam centre on the central detector. posx runs across the 16 tubes (away from the beam), posy along the tubes. Determine dx, dy, dz with a standard that puts rings on the panels (AgBe in the 4 m setting: 2.15, 3.23, 4.30 1/nm). Maren's data 2026-08 suggest that along the tubes the beam lies near posy ≈ +84 mm (dy ≈ -85 mm), to confirm.

- **left panel: offset dy along posy (along the tubes) [mm]** — `sasred.llb.left.dy_mm` · factory: **0** · range: -1000 … 1000  
  SANS-LLB left panel: offset along posy (along the tubes) [mm]; see dx.

- **left panel: offset dz along the beam [mm]** — `sasred.llb.left.dz_mm` · factory: **0** · range: -1000 … 1000  
  SANS-LLB left panel: offset of the detection plane along the beam from left_detector/distance [mm]; see dx.

- **bottom panel: offset dx along posx (along the tubes) [mm]** — `sasred.llb.bottom.dx_mm` · factory: **0** · range: -1000 … 1000  
  SANS-LLB bottom wide-angle panel: offset along posx (along the tubes) [mm]; pixel position = (posx + dx, posy + dy) at bottom_detector/distance + dz. See the left panel.

- **bottom panel: offset dy along posy [mm]** — `sasred.llb.bottom.dy_mm` · factory: **0** · range: -1000 … 1000  
  SANS-LLB bottom panel: offset along posy (across the tubes, away from the beam) [mm].

- **bottom panel: offset dz along the beam [mm]** — `sasred.llb.bottom.dz_mm` · factory: **0** · range: -1000 … 1000  
  SANS-LLB bottom panel: offset of the detection plane along the beam [mm].

- **pixels masked at both ends of every tube** — `sasred.llb.end_px` · factory: **10** · range: 0 … 60  
  SANS-LLB wide-angle panels: pixels masked at both ends of every tube (the position response of the tubes falls off there); tubes without counts are masked as well.

#### Resolution

- **dλ/λ (FWHM) when the raw file has none** — `sasred.resolution.dlam_default` · factory: **0.1** · range: 0 … 1  
  Relative wavelength spread (FWHM) used for the resolution Qdev when the raw file gives none: on SANS-I dλ/λ comes from the selector tilt and its calibration table, else from the header (monochromator/wavelength_spread); only when both are missing is this value used, and the notes say so. 0 switches the resolution off for such runs.

- **source aperture (guide cross-section) x [mm]** — `sasred.resolution.source_aperture_x_mm` · factory: **50** · range: 0.1 … 500  
  When a geometry has no direct-beam run, the beam divergence is estimated from the collimation: the source aperture (on SANS-I the neutron-guide cross-section, 50 × 50 mm²) at the collimation length coll before the sample aperture, detector SD behind it: σ_spot² = (a₁ SD/coll)²/12 + (a₂ (coll+SD)/coll)²/12 per axis. These preferences override the instrument file.

- **source aperture (guide cross-section) y [mm]** — `sasred.resolution.source_aperture_y_mm` · factory: **50** · range: 0.1 … 500  
  Vertical size of the source aperture (neutron-guide cross-section, 50 mm on SANS-I) for the divergence estimate from the collimation when a geometry has no direct-beam run.

- **sample aperture x [mm]** — `sasred.resolution.sample_aperture_x_mm` · factory: **8** · range: 0.1 … 200  
  Sample aperture used for the divergence estimate (typical SANS-I value 8 × 14 mm², x × y); the aperture actually used is not recorded in the raw file.

- **sample aperture y [mm]** — `sasred.resolution.sample_aperture_y_mm` · factory: **14** · range: 0.1 … 200  
  Vertical size of the sample aperture (typically 14 mm on SANS-I) for the divergence estimate from the collimation when a geometry has no direct-beam run.

### pyTOOLbox

Formerly pySASbox: values saved under `sasbox.…` are read as `toolbox.…`.

#### HDF5 viewer

- **largest number of elements read for a plot** — `toolbox.viewer.max_points` · factory: **4000000** · range: 10000 … 5e+08  
  Largest number of elements the HDF5 viewer reads for a plot or matrix view. Larger slices are read with a stride (every n-th element along each shown dimension); the axis labels say so.

#### Metadata lister

- **largest dataset listed as metadata (elements)** — `toolbox.lister.max_elements` · factory: **16** · range: 1 … 100000  
  Datasets with at most this many elements count as metadata and are offered as entries of the metadata lister; larger ones (images, histograms) are not.

- **digits after the decimal point of floating-point values** — `toolbox.lister.float_digits` · factory: **3** · range: 0 … 15  
  Floating-point values in the metadata listing are rounded to this many digits after the decimal point (as shown, copied and exported).

### pySCAtter

#### Calculation and fit (pySCAtter)

- **I(q) follows every parameter change** — `scatter.calc.auto_1d` · factory: **on** · on/off  
  Start value of the 'I(q) follows' box of pySCAtter: the 1D curve (and the model at the loaded data) is recalculated 0.35 s after every parameter change. The 2D pattern follows only when '2D follows' is ticked, because oriented anisometric particles take seconds.

- **fit: largest number of model evaluations** — `scatter.fit.max_nfev` · factory: **200** · range: 5 … 100000  
  Largest number of model evaluations of a pySCAtter fit (Trust Region Reflective least squares; each fitted parameter costs one evaluation per iteration for the Jacobian).

- **2D fit: use every n-th pixel (start value)** — `scatter.fit.every_px` · factory: **2** · range: 1 … 32  
  2D fits use every n-th pixel in both directions: n² times fewer model evaluations. 1 uses all pixels; the χ² and the errors refer to the pixels used.

### pySEBpolymers

#### Calculation and fit (pySEBpolymers)

- **I(q) follows every parameter change** — `seb.auto` · factory: **on** · on/off  
  The I(q) panel of pySEBpolymers is recalculated 0.15 s after every change of a parameter, the model or the q grid. Off: only with Calculate, Build or after a fit (for very large structures).

- **orientational average: Gauss-Legendre points** — `seb.n_mu` · factory: **128** · range: 8 … 4096  
  Number of Gauss-Legendre points in cos(alpha) for the orientational average of the oriented sub-units (SolidCylinder, Spheroid, ThinDisk rim). 128 is accurate to about 1e-10 up to qL of a few hundred; long cylinders at large q need more. The other sub-units have closed forms.

- **fit: largest number of model evaluations** — `seb.fit.max_nfev` · factory: **200** · range: 5 … 100000  
  Largest number of model evaluations of a pySEBpolymers fit (Trust Region Reflective least squares; each fitted parameter costs one evaluation per iteration for the Jacobian).

### pyATsas

#### ATSAS installation and runs

- **ATSAS installation folder** — `atsas.dir` · factory: **found automatically** (the dialog shows the folder found) · text  
  Folder of the ATSAS installation (the one with bin/ and atsas.lic), e.g. C:\Program Files\ATSAS-4.1.4 or ~/ATSAS-4.1.4. Empty: pyATsas looks for it itself -- environment variables PYATSAS_DIR and ATSAS (on Windows also the ATSAS value the installer writes to the registry), autorg on the PATH, then the newest ATSAS-x.y.z folder in the usual places (Program Files, the home folder, /opt, /Applications). Set it to use another of several installed versions.

- **folder for the ATSAS runs** — `atsas.workdir` · factory: **atsas_runs next to the preferences file** (the dialog shows the full path) · text  
  Every ATSAS run of the pyATsas panel gets its own sub-folder here (date, time, program, curve), with the input file and everything the program writes (.out, .fit, bead models, logs), so results can be opened later in PRIMUS or a molecular viewer. Empty: atsas_runs next to the pySASfit preferences file.

- **time limit of one program run (s)** — `atsas.timeout` · factory: **600** · range: 10 … 86400  
  A program run that takes longer is stopped and reported as failed. DAMMIF in slow mode and large CRYSOL fits may need more than the factory value.

- **DAMMIF: independent runs (start value)** — `atsas.dammif.nruns` · factory: **10** · range: 1 … 200  
  Start value of the number of independent DAMMIF runs. Several runs (10 or more) are needed to judge the uniqueness of a bead model (DAMAVER / DAMCLUST); they run in parallel on all but one core.

### pyDENsity

#### DENSS installation and runs

- **Python with DENSS** — `density.python` · factory: **found automatically** (the dialog shows the interpreter found) · text  
  Python interpreter in which DENSS is installed (pip install denss). Empty: the Python running pySASfit, then python / python3 on the PATH, then the interpreter of a 'denss' script on the PATH. Set it when DENSS lives in another environment (e.g. a python.org Python next to the MSYS2 one).

- **folder for the DENSS runs** — `density.workdir` · factory: **denss_runs next to the preferences file** (the dialog shows the full path) · text  
  Every DENSS run of the pyDENsity panel gets its own sub-folder here (date, time, kind, curve) with the input file and everything DENSS writes (maps .mrc, fits, log; denss-all: the aligned maps, the average and the FSC curve), so maps can be opened later in ChimeraX or PyMOL. Empty: denss_runs next to the pySASfit preferences file.

- **time limit of one map (s)** — `density.timeout` · factory: **7200** · range: 10 … 604800  
  A single DENSS map that takes longer is stopped and reported as failed; denss-all gets four times this limit. SLOW mode with fine grids can take more than an hour per map.

- **denss-all: cores (start value)** — `density.cores` · factory: **4** · range: 1 … 256  
  Start value of the number of maps that denss-all computes in parallel (its -j option). Each map needs one core and its own memory (grids of 64³ and more need several hundred MB).

### pyDENSSmulti

#### Runs

- **folder for the runs** — `densitymulti.workdir` · factory: **denssmulti_runs next to the preferences file** (the dialog shows the full path) · text  
  Every run of the pyDENSSmulti panel gets its own sub-folder here (date, time, scheme, number of contrasts) with the maps (one MRC file per component, per contrast, the support; the SLD map of the averaging scheme), the fits of every curve and a summary (multi_summary.json); several maps: the average and every single map in map_xx. File → Open a result folder reads them back. Empty: denssmulti_runs next to the pySASfit preferences file.

- **maps computed at the same time (start value)** — `densitymulti.threads` · factory: **4** · range: 1 … 256  
  Start value of the number of maps computed at the same time when several maps are averaged. Each map needs its own memory (a 64³ grid with four contrasts: about 100 MB).

### pySASscout

#### Results store

- **folder of the results store** — `scout.store` · factory: **scout_store next to the preferences file** (the dialog shows the full path) · text  
  Folder of the results store of pySASscout. Every fitted candidate and every search is written there (one small file per fit, grouped by curve), and fits found there are reused: repeating a search, continuing one that was stopped, or adding models only fits what is new. The folder has relative paths only, so it can be copied or moved (also to a network drive or a cluster) and then named here; two stores are merged by copying one into the other. Empty: scout_store next to the pySASfit preferences file.

- **reuse stored fits (start value)** — `scout.reuse` · factory: **on** · switch  
  Take a candidate from the results store instead of fitting it again when the curve, the candidate, the fit options (starts, seed, background, error scale) and the SASfit build are the same. Off: every candidate is fitted again and the stored fit is replaced. Start value of the check box on the pySASscout panel.

- **searches kept per curve** — `scout.keep_runs` · factory: **20** · range: 1 … 10000  
  Number of searches kept per curve in the results store (the newest ones; the fits themselves are kept independently of this). A stored search is the complete ranking with all curves, a few 10 kB.

### Appearance

#### Title bars of the panels (label … hide cross)

- **coloured title bars** — `appearance.panel_title.mode` · factory: **pale** · pale, all, active, off  
  Title bars of the docked panels coloured like window title bars under Windows. 'active panel in the colour, others pale': the panel last clicked has the full colour and white text, the others a light tint of it. Takes effect when the preferences are saved.

- **colour (#rrggbb or name)** — `appearance.panel_title.colour` · factory: **#0078d7** · text  
  Colour of the panel title bars: #rrggbb (e.g. #0078d7 Windows 10/11 blue, #2b579a Word blue, #1e3a5f dark blue) or a colour name (steelblue). The text turns white on dark colours, dark on light ones. An invalid value falls back to #0078d7.

### pySASfit window

#### Side bar (the column of tool icons)

- **background of the tool in front** — `launcher.sidebar.selected` · factory: **#FFF8C5** · colour #RRGGBB  
  Background of the entry of the tool in front in the side bar (light system theme), as #RRGGBB; the button next to the field opens the colour chooser. The tool names are written in the colours of the logos (grey py and ending, the core in the tool's colour), so the background has to be light: the factory value, a pale yellow, keeps every name at a contrast of at least 3 : 1 (the grey parts 4.5 : 1), within a few per cent of what white gives. Pale green (#EAF7EA) and pale orange are slightly lower (orange pyTOOLbox about 2.9 : 1); medium and saturated colours such as the blue selection colour of Windows make the names unreadable. Applied when the preferences are saved.

- **… on a dark system theme** — `launcher.sidebar.selected_dark` · factory: **#2B2B2B** · colour #RRGGBB  
  The same for a dark system theme, where the names use the lighter colour set: the background has to be dark. With the factory value every name has a contrast of at least 4.5 : 1. Applied when the preferences are saved.

### Shared numerics

#### Deflation of unphysical OZ roots (OZ solver, polydisperse, RY and mixture tabs)

- **power p in 1/‖x − r‖^p** — `numerics.deflation.p` · factory: **2** · range: 0.5 … 8  
  Deflation (Farrell, Birkisson & Funke 2015): after an unphysical root r is found, the solver's step is multiplied by η = Π(1/‖x − r‖^p + α), which repels it from r. p sets how sharply.

- **shift α** — `numerics.deflation.shift` · factory: **1** · range: 0 … 1e+06  
  Shift α of the deflation factor. It keeps η finite far from the known roots, so the solver cannot 'converge' by running away. Results are sensitive to it (the paper sweeps it); at our LJ test state 0.1 finds a second root, 1 does not.

- **largest deflation factor** — `numerics.deflation.cap` · factory: **1e+06** · range: 1 … 1e+300  
  Upper bound of η. Without it the deflated residual overflows close to a known root.

#### Analytic MSA / RMSA of polydisperse Yukawa (MSA tab, RY tab, approximations)

- **MSA amplitude iteration: tolerance** — `numerics.msa.tol` · factory: **1e-13** · range: 1e-16 … 0.0001  
  Analytic MSA of polydisperse hard-core Yukawa: convergence tolerance of the self-consistent amplitude iteration.

- **MSA amplitude iteration: iterations** — `numerics.msa.maxiter` · factory: **300** · range: 10 … 100000  
  Analytic MSA of polydisperse hard-core Yukawa: maximum iterations of the self-consistent amplitude iteration. If it ends at this limit the MSA is reported as not converged; raise it for strong coupling.

- **MSA amplitude iteration: fsolve tolerance** — `numerics.msa.fsolve_xtol` · factory: **1e-13** · range: 1e-16 … 0.0001  
  Analytic MSA: tolerance of the inner nonlinear solve (scipy fsolve) in each amplitude iteration. Keep it at or below the outer tolerance, otherwise the outer iteration cannot converge.

- **Γ root scan: ± span × z** — `numerics.msa.gamma_span` · factory: **6** · range: 1 … 1000  
  Γ route of the MSA: the roots of the Γ equation are bracketed on a grid over ± span × z (z: inverse screening length), geometric in |Γ| so that roots many decades below z are found.

- **Γ root scan: decades below the span** — `numerics.msa.gamma_decades` · factory: **10** · range: 1 … 40  
  Γ route of the MSA: number of decades of |Γ| below span × z covered by the root-bracketing grid. More decades find roots at very weak screening; the grid size is decades × points per decade.

- **Γ root scan: points per decade** — `numerics.msa.gamma_per_decade` · factory: **400** · range: 10 … 100000  
  Γ route of the MSA: grid points per decade of |Γ| for bracketing the roots. Increase it if two close roots are missed (they then lie in one grid interval).

- **Γ root refinement: tolerance** — `numerics.msa.gamma_xtol` · factory: **1e-15** · range: 1e-18 … 1e-06  
  Γ route of the MSA: tolerance of the Brent refinement of each bracketed Γ root. The default is near machine precision; no reason to change.

- **RMSA: largest rescaling factor λ** — `numerics.rmsa.lam_max` · factory: **6** · range: 1.01 … 100  
  Rescaled MSA (Hansen–Hayter): the common rescaling factor λ ≥ 1 that makes all contact values g_ij(σ_ij) ≥ 0 is searched up to this value.

- **RMSA: λ scan points** — `numerics.rmsa.nscan` · factory: **120** · range: 10 … 100000  
  Rescaled MSA: number of λ values between 1 and the largest λ scanned for the smallest λ that makes all contact values non-negative, before that λ is refined. More points avoid missing a narrow λ window.

- **RMSA: λ tolerance** — `numerics.rmsa.tol` · factory: **1e-12** · range: 1e-16 … 0.001  
  Rescaled MSA: tolerance of the refinement of the rescaling factor λ. The default is near machine precision; no reason to change.

#### Global optimisers for fits: NLopt and bumps

- **NLopt: max. evaluations** — `numerics.nlopt.max_nfev` · factory: **400** · range: 10 … 1e+07  
  Constrained fits with NLopt (e.g. 'shell inside the smallest core'): maximum evaluations.

- **NLopt: relative tolerance** — `numerics.nlopt.ftol` · factory: **1e-10** · range: 1e-16 … 0.01  
  Constrained fits with NLopt: the fit stops when the relative change of χ² falls below this value. Larger values stop earlier; if the fit stops before reaching the minimum, raise the maximum evaluations instead.

- **bumps: steps (0 = from samples)** — `numerics.bumps.steps` · factory: **0** · range: 0 … 1e+07  
  bumps optimisers: number of generations. 0 = derived from 'samples' (steps = samples / population).

- **bumps: population (× parameters)** — `numerics.bumps.pop` · factory: **10** · range: 1 … 1000  
  bumps optimisers (DE, DREAM, …): population size as a multiple of the number of fitted parameters. Larger populations explore the parameter space better and need more evaluations per generation.

- **bumps DREAM: burn-in steps** — `numerics.bumps.burn` · factory: **100** · range: 0 … 1e+07  
  bumps DREAM: burn-in generations discarded before the posterior is sampled. Increase it if the parameter traces still drift at the start of the sampling.

- **bumps: samples** — `numerics.bumps.samples` · factory: **10000** · range: 10 … 1e+08  
  bumps: number of samples drawn; when 'steps' is 0 it sets the number of generations (samples / population). For DREAM more samples give smoother posterior histograms and uncertainties.

#### Background models: start values of fitted terms (SLD profile)

- **constant: × median of the high-q tail** — `numerics.background.constant_level` · factory: **0.5** · range: 0 … 10  
  Start value of a fitted constant background: level × median of the highest-q points. The background fit is local, so the start decides between background and particle signal when both could explain the high-q intensity.

- **constant: high-q tail (fraction of points, ≥ 3)** — `numerics.background.constant_tail` · factory: **0.05** · range: 0.001 … 1  
  Start value of a fitted constant background: fraction of the points, from the highest q, whose median is multiplied by the level (at least 3 points). Larger fractions are more robust to noise but include more particle signal.

- **power law A q^-n: A = level × I(q_min) q_min^n** — `numerics.background.powerlaw_level` · factory: **0.05** · range: 1e-06 … 10  
  Start value of a power law A·q^(−n): A = level × I(q_min)·q_min^n, i.e. the power law starts at this fraction of the intensity at the lowest q.

- **Ornstein–Zernike: I_L = level × max I** — `numerics.background.oz_level` · factory: **0.05** · range: 1e-06 … 10  
  Start value of an Ornstein–Zernike term I_L/(1 + q²ξ²): I_L = level × max I; ξ starts at 1/median q unless given.

## Built-in values that remain fixed

The audit of all three tools found about 300 numerical constants. Those that are not adjustable fall into two
groups:

**Numerical safety guards** (no effect on a converged result): positivity and log floors (1e-300, 1e-12·max),
overflow clips (exp argument ±700), Cholesky jitters (1e-12…1e-15·tr/n), small-x series switches in the shell
amplitudes, cache sizes, edge tolerances of grids (1e-12).

**Fixed by the method** (changing them changes the method, not a setting): the smoothness stencils
(−½, 1, −½) and their end terms, the Guinier factors √(5/3), √2, √3, the FWHM→σ constant, the transform
normalisations, Hansen's evidence formula, the Levenberg–Marquardt damping schedule (×4 on reject, ÷3 on accept),
the B-spline degree 3, the Gauss quadrature rules.

All values that the audit listed as "not yet adjustable" are now in the dialog, the shared ones on the
**Shared numerics** page. The remaining fixed values are the two groups above.

## Problems found during the audit

Fixed now:

* Size distribution: the Bayesian-evidence λ grid (see above).
* Polydisperse OZ tab: **Fit** and the **consistent-α search** ignored the tab's numerical settings (grid, points
  per diameter, iterations, transform type) and always used 4095 / 100 / type 1 / solver default, so a fitted
  curve did not reproduce exactly on Compute. Both now use the same settings as Compute.
* RY Yukawa tab: the "no consistent α" message named a fixed bracket [0.02, 3]; it now shows the bracket used.

Fixed in the second round (2026-10-12):

* Polydisperse OZ tab, "Deflate if S(Q) < 0": the retry built a new solver without density, potential or closure
  and never solved it, and the result of the screen was ignored -- with deflation switched on an unphysical root
  was returned **silently**. Every attempt is now a complete solve with the deflated solver; the attempts and the
  min S(Q) of each root are listed in the error message. At Lennard-Jones ε = 0.8, φ = 0.3 (HNC) Newton-Krylov
  lands on a branch with min S = −18.3; with shift 0.1 deflation finds a second, different unphysical root
  (−20.2), and the fixed-point solvers find the physical root (min S = +0.21) directly.
* RY wrapper, automatic solver choice (`solverClass=None`): the fallback to the next solver was implemented but
  unreachable -- the code replaced `None` by the first solver before testing for it -- and would have dropped the
  closure, its second parameter and the deflation count. Both fixed; the solvers that failed are recorded in
  `solverFallback`. The RY tab itself always passes the solver chosen in its list, so it never used this path:
  an explicit choice is never replaced by another solver.
* OZ solver tab: the transform type can be chosen (Preferences, *Hankel transform type*): auto (type 4 for one
  component, the default), 1 or 4, for comparisons. `ozLib.solve(transformType=...)`, applied to the verifier too.
* Size distribution: data files with a 4th column σ_Q are read, and the kernel can be smeared with it
  (checkbox *Resolution*, Rician or Gaussian kernel; on by default when the column is present).

Reported, not changed:

* Size distribution: the background subtraction keeps only the constant term of a power-law fit (to do: move to
  the `pysasfit.core.background` models used by the SLD-profile tool).
* `core.resolution.GaussianResolution(n_nodes=…)` accepts the argument and ignores it.
* SLD two-step method: the "points fine" box has no effect (step 1 uses the coarse count for D_max).
