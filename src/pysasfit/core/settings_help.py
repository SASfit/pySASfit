# -*- coding: utf-8 -*-
"""Explanations of every preference (shown in the Preferences dialog and in
docs/preferences.md). One entry per key of settings_schema; each says what the
value controls, what raising or lowering it does, and when to change it.
A test checks that every registered setting has an entry here."""

HELP = {
    # =================================================================== SLD profile
    # --- alpha path
    "sldprofile.opt.log_alpha_hi":
        "Start of the α path, as ln α − ln α₀ with α₀ = tr(B)/tr(A) at the starting profile (B: data "
        "Hessian, A: Hessian of the regulariser). The path runs from strong smoothing (this end) to weak "
        "smoothing. Raise it if the evidence curve (ln evidence vs ln α) is still rising at its left end.",
    "sldprofile.opt.log_alpha_lo":
        "End of the α path, relative to ln α₀. The path normally stops earlier, once the evidence has "
        "dropped by the stop criterion. Lower it only if the evidence curve is still rising at its right end "
        "when the path ends.",
    "sldprofile.opt.alpha_step":
        "Step in ln α along the path. Each step is one regularised fit, warm-started from the previous "
        "one. It is also the integration step of the marginalisation over α (posterior of r_max, error "
        "bands). Smaller steps: smoother evidence curve, more fits.",
    "sldprofile.opt.stop_drop":
        "The α path stops once ln evidence has fallen this far below its running maximum (and the data "
        "are fitted, see the next entry). 25 means a factor e^25 below the best α -- such solutions carry "
        "no weight in the averages. Larger values only cost time.",
    "sldprofile.opt.stop_chi2":
        "The early stop is allowed only when χ² < factor × M (M = number of points), i.e. once the data "
        "are actually fitted. Prevents stopping on the falling side of a first, spurious evidence maximum.",
    "sldprofile.opt.retry_chi2":
        "Guard against a wrong branch: while χ² > factor × M, the fit at the current α is repeated from "
        "the other starting profiles (homogeneous, truncated step, flat coefficients) and the better one is "
        "kept. Lower it if fits with a too large r_max get stuck.",
    "sldprofile.opt.retry_every":
        "How often the restart check above is made: every n-th α step. 1 = at every step (slowest, safest).",
    "sldprofile.opt.alpha_xatol":
        "After the path, ln α is refined between the two neighbours of the best path point by a bounded "
        "one-dimensional maximisation of the evidence; this is its precision in ln α.",
    # --- minimiser
    "sldprofile.opt.max_iter":
        "Maximum Levenberg–Marquardt iterations per regularised fit (fixed α and r_max). Fits are "
        "warm-started along the α path, so they normally converge in a few iterations.",
    "sldprofile.opt.tol":
        "Convergence test of each regularised fit: stop when the objective αS + χ²/2 decreases by less than "
        "tol × max(|objective|, 1) in one iteration.",
    "sldprofile.opt.lm_lambda0":
        "Initial Levenberg–Marquardt damping λ (relative to the scaled Hessian). It is reduced ÷3 after a "
        "successful step and increased ×4 after a rejected one, so the start value matters little.",
    "sldprofile.opt.ridge":
        "Small multiple of the identity (× tr(A)/N) added to the smoothness matrix A. With a free outer end "
        "A is singular (a constant has zero curvature); the ridge makes it invertible. It enters log det A "
        "and therefore the evidence -- keep it small.",
    "sldprofile.opt.truncated_min":
        "Starting profiles: besides the homogeneous one, homogeneous steps ρ = c for r < R are tried with R "
        "on the shell grid, from this fraction of N shells up to N. They rescue fits with a too large r_max.",
    # --- result
    "sldprofile.fit.n_draws":
        "Number of random draws from the posterior covariance per kept solution, used for the error bands "
        "of ρ(r), the radial mass, Rg and the background. More draws: smoother bands, more time.",
    "sldprofile.fit.seed":
        "Seed of the random draws above. A fixed seed makes repeated runs give identical error bands.",
    "sldprofile.opt.keep_weight":
        "Solutions whose evidence weight is below this fraction of the best one are dropped from the "
        "averages (they contribute nothing and only cost time).",
    "sldprofile.opt.n_out":
        "Number of cells of the r grid on which ρ(r), the radial mass and their errors are reported "
        "(display and _rho.dat). Does not change the fit.",
    "sldprofile.opt.r_out_factor":
        "The output r grid extends to factor × the largest r_max among the kept solutions, so the outer "
        "edge of the profile is visible.",
    "sldprofile.fit.max_extend":
        "If the evidence maximum lies at the upper end of the r_max scan, the scan is extended upwards; "
        "this is how often. Needed for coil-like profiles that reach far beyond the Guinier estimate.",
    "sldprofile.opt.guinier_qrg":
        "Upper limit q·Rg of the Guinier fit that gives the start value of r_max (sphere √(5/3) Rg, "
        "cylinder √2 R_c, lamella √3 R_t). 1.3 is the usual validity limit for compact particles.",
    "sldprofile.opt.extend_fraction":
        "Number of r_max points added per upward extension of the scan, as a fraction of the coarse "
        "points (at least 3).",
    "sldprofile.opt.fine_sd":
        "After the coarse r_max scan a fine scan is made around the maximum; its half width is "
        "max(fine_sd × posterior s.d., fine_steps × coarse step). This is the first term.",
    "sldprofile.opt.fine_steps":
        "Second term of the fine-scan half width (see above), in coarse grid steps; it keeps the fine scan "
        "from collapsing when the coarse posterior is very narrow.",
    # --- grids and scans
    "sldprofile.adaptive.smoothing":
        "Adaptive grid: the first pass uses the best uniform-grid solution with α raised by this many "
        "e-folds, so the grid follows the real structure of the profile and not its noise.",
    "sldprofile.adaptive.max_ratio":
        "Adaptive grid: the finest shell is at most this many times thinner than the coarsest one. Limits "
        "how strongly the grid concentrates at steep parts of the profile.",
    "sldprofile.adaptive.min_width":
        "Adaptive grid: smallest allowed shell width in r units. 0 = automatic, min(π/q_max, r_max/N): "
        "structure finer than π/q_max is not determined by the data.",
    "sldprofile.gui.p_scan_step":
        "Step of the exponent scan ('Scan p'), which compares representations r^p·ρ(r) for p = 0 … d−0.2 by "
        "their evidence (4/3 is always included for spheres).",
    "sldprofile.gui.prior_floor":
        "Max-ent with the 'previous result' prior: the prior is the last profile, clipped below at this "
        "fraction of its maximum (max-ent needs a strictly positive prior).",
    "sldprofile.gui.hollow_level":
        "Max-ent with the 'hollow shell' prior: prior level inside r_in, relative to 1 in the shell.",
    # --- two-step
    "sldprofile.ift.log_alpha_lo":
        "Two-step method, step 1 (IFT): lower end of the α grid, ln α relative to ln(tr B / tr C) "
        "(B: data Hessian, C: smoothness matrix). The evidence maximum is searched on this grid, then "
        "refined.",
    "sldprofile.ift.log_alpha_hi":
        'Two-step method, step 1 (IFT): upper end of the α grid, ln α relative to ln(tr B / tr C). This end '
        'is the strongly smoothed side. Raise it if the IFT evidence still rises at the upper end of the grid'
        ' (very noisy or very few data).',
    "sldprofile.ift.n_alpha":
        'Two-step method, step 1 (IFT): number of points of the α grid between the lower and upper end. The '
        'grid only brackets the evidence maximum, which is then refined, so more points rarely change the '
        'result; use more if the evidence curve has several maxima.',
    "sldprofile.ift.alpha_xatol":
        'Two-step method, step 1 (IFT): the evidence maximum found on the α grid is refined by a bounded 1-D '
        'search until ln α is known to this precision. The default is far below the width of the evidence '
        'peak; there is no reason to change it.',
    "sldprofile.ift.bg_below":
        "IFT with 'fit' background: the flat background is chosen by maximum evidence within "
        "[I_min − below·s, I_min + above·s], s = max(|I_min|, median σ). This is 'below'.",
    "sldprofile.ift.bg_above":
        "Two-step method, IFT with 'fit' background: the flat background is chosen by maximum evidence within"
        " [I_min − below·s, I_min + above·s], s = max(|I_min|, median σ). This is 'above', in units of s. "
        'Increase it if the chosen background sits at the upper end of the interval.',
    "sldprofile.ift.bg_xatol":
        "Two-step method, IFT with 'fit' background: precision of the evidence search for the flat "
        'background, in units of s = max(|I_min|, median σ). Far below the statistical uncertainty of the '
        'background; no reason to change.',
    "sldprofile.twostep.penalty":
        "Smoothness prior of the IFT. Hansen (2000) eq. 19: curvature, Σ[p_j − (p_{j−1}+p_{j+1})/2]² with "
        "p(0) = p(D_max) = 0. 'As printed': the matrix given in the text of that paper, which is a "
        "first-difference penalty. The normalised evidence prefers eq. 19 on our test data.",
    "sldprofile.twostep.n_sub":
        "Deconvolution (step 2): each shell is split into this many thin sub-shells to build the matrices "
        "V(r) of p(r) = ρᵀV(r)ρ. 6 reproduces the direct I(q) to ~1e-4.",
    # --- quadrature
    "sldprofile.geo.n_gl":
        "For a custom exponent p (r^p·ρ represented, p ≠ 0, d−1) the shell amplitudes are integrated "
        "numerically: Gauss–Jacobi in the first shell (weight r^(d−1−p)), Gauss–Legendre in the others, "
        "with this many nodes each.",
    "sldprofile.geo.pddf_n_sub":
        "Sub-shells per shell for the p(r) shown as overlay and written to _pr.dat (not used in the fit).",
    "sldprofile.geo.pddf_raster":
        "Cylinders: the p(r) of the cross-section is computed by a 2-D FFT autocorrelation on a square "
        "raster of this size (display and _pr.dat only).",
    # --- resolution
    "sldprofile.res.n_sigma":
        "Pinhole resolution: the model is evaluated on an extended q grid reaching ± n·σ_q beyond the "
        "measured range, and the kernel is truncated there. 3 covers 99.7 % of a Gaussian.",
    "sldprofile.res.oversample":
        "Pinhole resolution: density of the extended q grid, as a multiple of the number of data points.",
    "sldprofile.res.slit_nodes":
        'Slit smearing: number of Gauss–Hermite nodes on the positive half of the slit profile. The default '
        'integrates a Gaussian slit to better than 1e-6; increase it only for very long slits combined with '
        'sharp features in the model.',
    # =================================================================== Size distribution / p(r)
    "inversion.opt.evidence_lambda_scale":
        "Centre of the λ grid of the Bayesian-evidence solvers. 'trace': tr(B)/tr(2LᵀL), the scale at which "
        "data and prior curvature are equal -- independent of how the r grid is sampled. 'median': "
        "median singular value of the kernel, squared; the rule before 2026-10-11. With many r points most "
        "singular values are at the noise floor (~1e-15) and the 'median' grid lies far below the evidence "
        "maximum, giving an unregularised p(r).",
    "inversion.opt.evidence_decades_below":
        "λ grid: decades below the centre (trace rule). The evidence maximum of a well-determined p(r) "
        "usually lies a few decades below.",
    "inversion.opt.evidence_decades_above":
        'Bayesian evidence λ selection (trace rule): the λ grid extends this many decades above the centre '
        'tr(B)/tr(2LᵀL). Above the centre the solution is over-smoothed, so a few decades suffice; increase '
        'it if the selected λ lies at the upper grid end.',
    "inversion.opt.evidence_points":
        'Bayesian evidence λ selection (trace rule): number of λ grid points over the whole range (below + '
        'above). The maximum is refined after the grid search, so the default is ample; more points cost time'
        ' linearly.',
    "inversion.opt.evidence_refine":
        "Refine λ between the two neighbours of the best grid point by a bounded 1-D maximisation of the "
        "evidence (otherwise the grid point itself is used).",
    "inversion.opt.max_lambda_reduction":
        "With 'χ²ᵣ = 1' ticked, λ is adjusted from the evidence value to reach the target. It may be raised "
        "freely but lowered at most by this factor: reaching χ²ᵣ = 1 only by (almost) removing the "
        "regularisation means the target is wrong for the data.",
    "inversion.opt.noise_scale_lo":
        "Error bars 'overall scale estimated': σ_i = s·ΔI_i with s chosen by maximum evidence within "
        "[lower, upper]. Lower limit.",
    "inversion.opt.noise_scale_hi":
        "Error bars 'overall scale estimated': σ_i = s·ΔI_i with s chosen by maximum evidence within [lower, "
        'upper]. This is the upper limit of s. Raise it if s ends at this limit (the given errors are much '
        'too small).',
    "inversion.gui.target_chi2":
        "Target of the 'χ²ᵣ = 1' option of the Bayesian-evidence solver (discrepancy principle).",
    "inversion.opt.em_max_iterations":
        "EM + smoothing and signed-kernel EM: maximum iterations for one trial smoothing value h (each is "
        "Biggs–Andrews accelerated).",
    "inversion.opt.em_tol":
        'EM and EM + MaxEnt: one solve stops when the relative change of the distribution between iterations '
        'falls below this value. Smaller values give a converged result at more iterations; larger values '
        'speed up the h or λ search at the price of a slightly less converged distribution.',
    "inversion.opt.em_h_start":
        "EM + smoothing: the smoothing parameter h (0 … 1) is searched by the discrepancy principle, "
        "starting here and decreasing.",
    "inversion.opt.em_h_floor":
        "EM + smoothing: smallest h tried. Below it the fit is treated as unable to reach χ²ᵣ = 1 and the "
        "fallback (L-curve or least smoothing) is used.",
    "inversion.opt.em_general_h_floor":
        'Signed-kernel EM: smallest smoothing value h tried. Below it the fit is treated as unable to reach '
        'χ²ᵣ = 1 and the fallback (L-curve or least smoothing) is used. Lower it only if the data are very '
        'precise and the fallback is triggered too early.',
    "inversion.opt.lcurve_points":
        'L-curve fallback of the EM methods (used when χ²ᵣ = 1 cannot be reached): number of smoothing values'
        ' h sampled to locate the corner of the L-curve. More points locate the corner more precisely but '
        'cost one EM solve each.',
    "inversion.opt.lcurve_reduction":
        "L-curve: factor between successive h values (h, h/f, h/f², …). With 25 points and 1.2 the curve "
        "spans about two decades.",
    "inversion.opt.maxent_lambda_min":
        "EM + MaxEnt: smallest entropy weight λ of the grid; the λ giving χ²ᵣ closest to 1 is chosen.",
    "inversion.opt.maxent_lambda_max":
        'EM + MaxEnt: largest entropy weight λ of the λ grid; the λ giving χ²ᵣ closest to 1 is chosen. Raise '
        'it if the chosen λ lies at this end (the data are fitted with χ²ᵣ < 1 even at the strongest entropy '
        'weight).',
    "inversion.opt.maxent_points":
        'EM + MaxEnt: number of λ values on the geometric grid between smallest and largest λ. Each point is '
        'one complete MaxEnt solve; more points select λ more finely but cost proportionally more time.',
    "inversion.opt.maxent_max_iterations":
        'EM + MaxEnt: maximum iterations of one solve at fixed λ. If solves end at this limit (see the log) '
        'the selected λ may be off; increase it, or relax the EM / MaxEnt tolerance.',
    "inversion.opt.maxent_sigma2":
        "EM + MaxEnt, adaptive prior: the prior is the current solution smoothed with a Gaussian of this "
        "variance (in grid points²).",
    "inversion.opt.signed_maxent_lambda_start":
        "Signed-kernel EM with MaxEnt: start of the bidirectional discrepancy search of the entropy weight.",
    "inversion.opt.signed_maxent_lambda_floor":
        'Signed-kernel EM with MaxEnt: smallest entropy weight the bidirectional discrepancy search may '
        'reach. If the search ends here, even almost no entropy cannot fit the data to χ²ᵣ = 1 (errors too '
        'small or model inadequate).',
    "inversion.opt.signed_maxent_lambda_ceil":
        'Signed-kernel EM with MaxEnt: largest entropy weight the bidirectional discrepancy search may reach.'
        ' If the search ends here, the data are over-fitted even at the strongest entropy weight (errors '
        'probably too large).',
    "inversion.opt.hansen_maxent_target_chi2":
        "Hansen's MaxEnt: λ² is chosen by the discrepancy principle so that the reduced χ² equals this "
        'target. 1 trusts the error bars; values > 1 give smoother distributions when the errors are known to'
        ' be underestimated.',
    "inversion.opt.hansen_maxent_start":
        "Hansen's MaxEnt: start value of the search for λ². The search first brackets the discrepancy root by"
        ' multiplying or dividing by the step factor, then refines it, so the start affects speed, not the '
        'result.',
    "inversion.opt.hansen_maxent_factor":
        "Hansen's MaxEnt: factor between successive λ² values while bracketing the discrepancy root. Larger "
        'factors bracket faster; smaller ones are safer when χ²ᵣ(λ²) is not monotonic.',
    "inversion.opt.lbfgs_ftol":
        "Hansen's MaxEnt is minimised with bound-constrained L-BFGS-B: relative function tolerance.",
    "inversion.opt.lbfgs_gtol":
        "Hansen's MaxEnt is minimised with bound-constrained L-BFGS-B: the minimisation stops when the "
        'largest projected-gradient component is below this value. Smaller values give a better-converged '
        'distribution at more iterations.',
    "inversion.opt.lbfgs_maxcor":
        "Hansen's MaxEnt, L-BFGS-B: number of stored correction pairs, i.e. the memory of the Hessian "
        'estimate. More pairs usually mean fewer iterations for many bins, at a small extra cost per '
        'iteration.',
    "inversion.opt.em_x0":
        "EM and MaxEnt start from a flat vector at this level. EM converges to the same fixed point from any "
        "positive start; the level matters for the iteration count only.",
    "inversion.opt.maxent_prior_floor":
        'EM + MaxEnt with constant prior: the prior level is max(mean(I)/n, floor), n = number of bins. The '
        'floor only matters for data with very small or negative mean intensity, where it keeps the entropy '
        'defined.',
    "inversion.opt.root_xtol":
        "Discrepancy-principle searches (EM h, MaxEnt λ) end with a Brent root search, stopping when "
        "|x − x*| ≤ xtol + rtol·|x*|. The absolute 1e-6 is the historical value; for parameters far below "
        "1e-6 it resolves the root only loosely -- set xtol = 0 and rtol = 1e-6.",
    "inversion.opt.root_rtol":
        'Discrepancy-principle searches (EM h, MaxEnt λ) end with a Brent root search, stopping when |x − x*|'
        " ≤ xtol + rtol·|x*|. This is rtol; the default is the machine-precision limit of Brent's method. Set"
        ' rtol = 1e-6 together with xtol = 0 for parameters far below 1e-6.',
    "inversion.opt.background_maxfev":
        'Maximum function evaluations of the least-squares fit of the background backgr + c·q^(−α) (scipy '
        'curve_fit). Increase it if the background fit reports that it did not converge.',
    "inversion.opt.gcv_decades_below":
        "Tikhonov / general Tikhonov with GCV: the λ grid runs from s_min·10^(−below) to s_max·10^(above) "
        "(s: singular values of the weighted kernel).",
    "inversion.opt.gcv_decades_above":
        'Tikhonov / general Tikhonov with GCV: the λ grid runs from s_min·10^(−below) to s_max·10^(above) (s:'
        " singular values of the weighted kernel). This is 'above'; raise it if GCV selects the largest λ.",
    "inversion.opt.gcv_points":
        'Tikhonov / general Tikhonov with GCV: number of λ values on the grid between its lower and upper '
        'end. GCV is evaluated cheaply from the SVD, so more points cost little.',
    "inversion.opt.tsvd_k_max":
        'Truncated SVD: largest truncation rank k tried by GCV (capped by the number of bins and data). Raise'
        ' it if GCV selects this largest k, i.e. the data support more singular components.',
    "inversion.res.n_sigma":
        "Resolution from the 4th column σ_Q: the kernel is evaluated on an extended q grid ± n·σ_Q beyond "
        "the data and the smearing kernel is truncated there.",
    "inversion.res.oversample":
        'Resolution from the 4th column σ_Q: density of the extended q grid on which the kernel is evaluated '
        'before smearing, as a multiple of the number of data points. Increase it for sharp features (narrow '
        'distributions, large particles) where the default grid under-samples the kernel.',
    "inversion.res.fwhm":
        "Tick if the 4th column of your files is the full width at half maximum instead of σ_Q "
        "(FWHM = 2.3548 σ).",
    "inversion.gui.fine_grid_factor":
        "The kernel is integrated on a fine grid (p linear or cubic between the r points) when the r "
        "spacing exceeds factor × π/q_max, the period of the fastest kernel oscillation; otherwise it is "
        "evaluated at the points.",
    "inversion.gui.r_points_factor":
        "After loading, the number of r points is suggested as factor × the Shannon number "
        "(q_max − q_min)·r_max/π. 1 = the number of independent parameters the data contain.",
    "inversion.gui.poly_degree":
        "Error model 'local cubic fits': degree of the polynomial in ln I vs ln q fitted through each point "
        "and its neighbours.",
    "inversion.gui.poly_half_width":
        "Error model 'local cubic fits': number of neighbours on each side of a point used in the polynomial "
        'fit of ln I vs ln q. The scatter about the fit gives the error estimate; wider windows give smoother'
        ' but possibly biased estimates where I(q) bends sharply.',
    "inversion.gui.bootstrap_seed":
        "Seed of the Monte-Carlo (bootstrap) error estimate. −1 = different every run; a fixed value makes "
        "the error bands reproducible.",

    # =================================================================== Ornstein-Zernike
    "oz.solvers.max_iterations":
        "Default iteration limit of the accelerated OZ solvers (Anderson, MDIIS, KINSOL, Biggs–Andrews), "
        "used wherever a tab does not set its own limit.",
    "oz.solvers.max_iterations_picard":
        "Iteration limit N of the Picard / Mann solver for a = 1 (plain Picard), used wherever a tab does "
        "not set its own limit, in particular always in the OZ solver tab. With damping a < 1 the solver "
        "runs up to ceil(N/a) iterations, because each damped step covers only a fraction a of the way "
        "(5000 at a = 0.2 gives 25000). Picard converges linearly and slowly near the spinodal, hence the "
        "higher limit than for the accelerated solvers. Example: hard spheres PY at φ = 0.53 need about "
        "4200/a iterations (a ≤ 0.6), so N ≈ 4500 or more is needed there; the factory value 5000 (raised "
        "from 3000) covers this, also for the damped Picard / Mann verifier.",
    "oz.solvers.convergence":
        "Convergence criterion of the fixed-point solvers: relative change of Γ(r) between iterations.",
    "oz.solvers.kinsol_maa":
        "SUNDIALS KIN_FP: depth of the Anderson acceleration (number of previous iterates combined). "
        "SASfit's C solver uses 5.",
    "oz.solvers.biggs_maa":
        "Biggs–Andrews vector extrapolation: order |n|; the sign selects the step-size formula.",
    "oz.solvers.mdiis_history":
        'MDIIS: number of previous residuals used in the DIIS extrapolation. Longer histories accelerate more'
        ' but can become ill-conditioned; reduce it if MDIIS oscillates, increase it for slow convergence of '
        'dense states.',
    "oz.solve.verify_tolerance":
        "OZ solver tab, 'verify': the result is cross-checked by an independent solver from a cold start; "
        "they must agree on S(q) to this relative accuracy, otherwise multiple roots are reported.",
    "oz.solve.transform_type":
        "Discrete sine transform of the Hankel transform. Type 4 (DST-IV) puts the grid at half-integer "
        "points and is second order for one component; type 1 is first order but works for mixtures. "
        "auto = 4 for one component, 1 for mixtures. Force one for comparisons.",
    "oz.solve.residual_tolerance":
        "After every OZ solve the closure's fixed-point map T is applied once more to the returned solution "
        "x, and the result counts as converged only if ||T(x) - x|| / ||x|| is at most this value -- whatever "
        "the solver itself reported. This catches solvers that claim success on a non-solution (e.g. KIN_FP "
        "returning NaN for polydisperse hard-core Yukawa under HNC, or a stopping test fooled by a stalled "
        "iteration). Converged solves give 1e-16 to 1e-11, so 1e-6 rejects only real failures. Lower it for "
        "stricter checking; raise it only if well-converged results are being rejected. A failed check marks "
        "the run [NOT CONVERGED].",
    "oz.solve.verify_mann_alpha":
        "Damping a of Picard / Mann when it serves as the second verifier (after Biggs–Andrews). "
        "Undamped Picard diverges above φ ≈ 0.42 for hard spheres, which is also where Biggs–Andrews stops "
        "converging, so dense states had no verifier at all. Damped, Picard / Mann converges there (hard "
        "spheres PY, φ = 0.53: a ≤ 0.6). Its iteration limit is the Picard limit / a, so a smaller a costs "
        "time. Must satisfy 0 < a ≤ 1. The verifier is skipped when the primary solver is Picard / Mann "
        "itself (same algorithm family).",
    "oz.solve.mann_alpha":
        "Value that the 'damping a' field of the OZ solver tab shows at start. The Picard / Mann solver "
        "iterates x <- (1 - a) x + a T(x): a = 1 is plain Picard, a < 1 is Mann's damped iteration, "
        "slower (roughly 1/a times more iterations) but convergent where undamped Picard oscillates or "
        "diverges, e.g. dense hard spheres above φ ≈ 0.42 or strongly attractive states. Must satisfy "
        "0 < a ≤ 1. Typical values 0.2 to 0.8; at a fixed state there is an upper bound above which the "
        "iteration oscillates (hard spheres PY at φ = 0.53: a ≤ 0.6). The iteration limit is the Picard "
        "limit divided by a. Ignored by all other solvers.",
    "oz.consistency.scan_steps":
        "'Find consistent parameter' (RY, HMSA, MHNC, BPGG, CJVM, BB): number of scan points of the "
        "closure parameter before the root of the compressibility mismatch is refined.",
    "oz.consistency.tolerance":
        "'Find consistent parameter' (RY, HMSA, MHNC, BPGG, CJVM, BB): tolerance of the root refinement of "
        'the compressibility mismatch in the closure parameter. Smaller values pin the parameter more '
        'precisely at more OZ solves.',
    "oz.generic.grid_exponent":
        "Polydisperse tab: radial grid N = 2^k − 1 points (type 1) or 2^k (type 4). Start value of the "
        "tab's Numerical settings; a session file overrides it.",
    "oz.generic.points_per_sigma":
        "Polydisperse tab: grid points per mean diameter. With the grid size it fixes the radial range "
        "(N / points per σ diameters) and so the lowest q.",
    "oz.generic.max_iterations":
        'Polydisperse tab: iteration limit of each OZ solve (including the solves during a fit and the α '
        'search). Raise it for dense or strongly attractive states that converge slowly.',
    "oz.generic.transform_type":
        'Polydisperse tab: transform type of the OZ solver. 1 is required for most mixtures (the components '
        'are transformed as a matrix); 4 is a single-component variant and is only valid for one component.',
    "oz.generic.fit_max_nfev":
        "Polydisperse tab, Fit: maximum model evaluations of least_squares (each start of the multi-start "
        "fit gets this budget). Every evaluation is a full OZ solve.",
    "oz.generic.n_starts":
        "Polydisperse tab, 'least_squares (n starts)': number of local fits from scattered start points, to "
        "find out whether the minimum is unique.",
    "oz.generic.diff_step":
        "Polydisperse tab, Fit: relative finite-difference step of the Jacobian. Too small and the OZ "
        "solver's own convergence noise dominates the derivative.",
    "oz.generic.alpha_lo":
        "Polydisperse tab, 'find consistent α': lower end of the bracket of the closure parameter.",
    "oz.generic.alpha_hi":
        "Polydisperse tab, 'find consistent α': upper end of the bracket of the closure parameter. If the "
        'search ends at this end the closure cannot reach thermodynamic consistency inside the bracket.',
    "oz.generic.alpha_tol":
        "Polydisperse tab, 'find consistent α': tolerance of the root search of the compressibility mismatch "
        'in α. Smaller values cost more OZ solves.',
    "oz.generic.alpha_accept":
        "α is reported as consistent when |compressibility mismatch| < value × χ⁻¹ (relative test; χ⁻¹ is "
        "of order 10 for a dense fluid). Otherwise the closest bracket end is used and flagged.",
    "oz.generic.res_n_sigma":
        'Polydisperse tab, resolution from σ_Q: the smearing kernel is evaluated on an extended Q grid ± '
        'n·σ_Q beyond the data and truncated there. 3 covers 99.7 % of a Gaussian; raise it for very wide '
        'resolution near steep features.',
    "oz.generic.res_oversample":
        'Polydisperse tab, resolution from σ_Q: density of the extended Q grid on which the model is '
        'evaluated before smearing, as a multiple of the number of data points. Increase it for sharp '
        'structure-factor peaks.',
    "oz.ry.alpha_lo":
        "RY Polydisperse Yukawa tab: lower end of the bracket of the Rogers–Young mixing parameter α "
        "searched for thermodynamic consistency.",
    "oz.ry.alpha_hi":
        'RY Polydisperse Yukawa tab: upper end of the bracket of the Rogers–Young mixing parameter α searched'
        ' for thermodynamic consistency. If the search ends here, consistency is not reached in the bracket.',
    "oz.ry.alpha_tol":
        'RY Polydisperse Yukawa tab: tolerance of the root search for the consistent α. Smaller values cost '
        'more OZ solves.',
    "oz.ry.alpha_accept":
        "α is reported as consistent when the absolute pressure-route mismatch is below this value.",
    "oz.ry.nsig":
        "RY tab: the size classes of a log-normal distribution span ± n standard deviations (in ln R). "
        "Schulz-Zimm/Gamma and Beta use their exact Gauss rules and ignore it.",
    "oz.mixture.accept_residual":
        "Mixture validation tab: a numerical OZ solve is rejected when max|G(Γ) − Γ| exceeds this value.",

    # =================================================================== Shared numerics
    "numerics.deflation.p":
        "Deflation (Farrell, Birkisson & Funke 2015): after an unphysical root r is found, the solver's "
        "step is multiplied by η = Π(1/‖x − r‖^p + α), which repels it from r. p sets how sharply.",
    "numerics.deflation.shift":
        "Shift α of the deflation factor. It keeps η finite far from the known roots, so the solver cannot "
        "'converge' by running away. Results are sensitive to it (the paper sweeps it); at our LJ test "
        "state 0.1 finds a second root, 1 does not.",
    "numerics.deflation.cap":
        "Upper bound of η. Without it the deflated residual overflows close to a known root.",
    "numerics.msa.tol":
        "Analytic MSA of polydisperse hard-core Yukawa: convergence tolerance of the self-consistent "
        "amplitude iteration.",
    "numerics.msa.maxiter":
        'Analytic MSA of polydisperse hard-core Yukawa: maximum iterations of the self-consistent amplitude '
        'iteration. If it ends at this limit the MSA is reported as not converged; raise it for strong '
        'coupling.',
    "numerics.msa.fsolve_xtol":
        'Analytic MSA: tolerance of the inner nonlinear solve (scipy fsolve) in each amplitude iteration. '
        'Keep it at or below the outer tolerance, otherwise the outer iteration cannot converge.',
    "numerics.msa.gamma_span":
        "Γ route of the MSA: the roots of the Γ equation are bracketed on a grid over ± span × z (z: inverse "
        "screening length), geometric in |Γ| so that roots many decades below z are found.",
    "numerics.msa.gamma_decades":
        'Γ route of the MSA: number of decades of |Γ| below span × z covered by the root-bracketing grid. '
        'More decades find roots at very weak screening; the grid size is decades × points per decade.',
    "numerics.msa.gamma_per_decade":
        'Γ route of the MSA: grid points per decade of |Γ| for bracketing the roots. Increase it if two close'
        ' roots are missed (they then lie in one grid interval).',
    "numerics.msa.gamma_xtol":
        'Γ route of the MSA: tolerance of the Brent refinement of each bracketed Γ root. The default is near '
        'machine precision; no reason to change.',
    "numerics.rmsa.lam_max":
        "Rescaled MSA (Hansen–Hayter): the common rescaling factor λ ≥ 1 that makes all contact values "
        "g_ij(σ_ij) ≥ 0 is searched up to this value.",
    "numerics.rmsa.nscan":
        'Rescaled MSA: number of λ values between 1 and the largest λ scanned for the smallest λ that makes '
        'all contact values non-negative, before that λ is refined. More points avoid missing a narrow λ '
        'window.',
    "numerics.rmsa.tol":
        'Rescaled MSA: tolerance of the refinement of the rescaling factor λ. The default is near machine '
        'precision; no reason to change.',
    "numerics.nlopt.max_nfev":
        "Constrained fits with NLopt (e.g. 'shell inside the smallest core'): maximum evaluations.",
    "numerics.nlopt.ftol":
        'Constrained fits with NLopt: the fit stops when the relative change of χ² falls below this value. '
        'Larger values stop earlier; if the fit stops before reaching the minimum, raise the maximum '
        'evaluations instead.',
    "numerics.bumps.steps":
        "bumps optimisers: number of generations. 0 = derived from 'samples' (steps = samples / population).",
    "numerics.bumps.pop":
        'bumps optimisers (DE, DREAM, …): population size as a multiple of the number of fitted parameters. '
        'Larger populations explore the parameter space better and need more evaluations per generation.',
    "numerics.bumps.burn":
        'bumps DREAM: burn-in generations discarded before the posterior is sampled. Increase it if the '
        'parameter traces still drift at the start of the sampling.',
    "numerics.bumps.samples":
        "bumps: number of samples drawn; when 'steps' is 0 it sets the number of generations (samples / "
        'population). For DREAM more samples give smoother posterior histograms and uncertainties.',
    "numerics.background.constant_level":
        "Start value of a fitted constant background: level × median of the highest-q points. The "
        "background fit is local, so the start decides between background and particle signal when "
        "both could explain the high-q intensity.",
    "numerics.background.constant_tail":
        'Start value of a fitted constant background: fraction of the points, from the highest q, whose '
        'median is multiplied by the level (at least 3 points). Larger fractions are more robust to noise but'
        ' include more particle signal.',
    "numerics.background.powerlaw_level":
        "Start value of a power law A·q^(−n): A = level × I(q_min)·q_min^n, i.e. the power law starts at "
        "this fraction of the intensity at the lowest q.",
    "numerics.background.oz_level":
        "Start value of an Ornstein–Zernike term I_L/(1 + q²ξ²): I_L = level × max I; ξ starts at "
        "1/median q unless given.",
}
