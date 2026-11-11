# -*- coding: utf-8 -*-
"""Declaration of every user-adjustable numerical setting (Settings → Preferences).

Factory defaults are the values the code used before the preferences existed,
so an empty preferences file reproduces earlier results exactly -- with one
deliberate exception, marked below: the lambda grid of the Bayesian-evidence
solvers of the size-distribution tool.

Key prefixes are consumed as dictionaries by the GUIs:

* ``sldprofile.opt.*``  -> ``DensityProfileFit(options=...)``
* ``sldprofile.ift.*``  -> ``two_step_fit(ift_options=...)``
* ``inversion.opt.*``   -> ``pysasfit.inversion.solver_registry.OPTIONS``
* ``oz.solvers.*``      -> class defaults of the OZ solver classes
"""

from .settings import Setting as S, register


def _next_to_preferences(name):
    """A file in the folder of the preferences file (resolved when it is needed)."""
    import os
    from .settings import settings_path
    return os.path.join(os.path.dirname(settings_path()), name)


# ---------------------------------------------------------------------------
# SLD profile
# ---------------------------------------------------------------------------
T, TT = "sldprofile", "SLD profile"
register(T, TT, "Regularisation parameter α (direct fit and deconvolution)", [
    S("sldprofile.opt.log_alpha_hi", 6.0, "ln α start (above ln α₀)",
      "The α path starts at ln α₀ + this value, α₀ = tr(B)/tr(A) at the starting profile "
      "(large α = strong smoothing).", lo=-50, hi=100),
    S("sldprofile.opt.log_alpha_lo", -16.0, "ln α end (relative to ln α₀)",
      "The α path runs down to ln α₀ + this value unless the evidence has dropped by the stop "
      "criterion first.", lo=-200, hi=50),
    S("sldprofile.opt.alpha_step", 0.5, "ln α step",
      "Step of the α path; also the step of the α marginalisation.", lo=0.01, hi=5),
    S("sldprofile.opt.stop_drop", 25.0, "stop when ln evidence has dropped by",
      "The path stops once ln Ev is this far below its maximum (and χ² < stop χ² factor · M).",
      lo=1, hi=1000),
    S("sldprofile.opt.stop_chi2", 1.5, "stop only when χ² < factor · M",
      "Early stop is allowed only once the data are fitted.", lo=0.1, hi=100),
    S("sldprofile.opt.retry_chi2", 2.0, "restart from other start profiles when χ² > factor · M",
      "Guards against a wrong branch on the α path.", lo=1, hi=1e6),
    S("sldprofile.opt.retry_every", 4, "… every n-th α step", "", kind="int", lo=1, hi=1000),
    S("sldprofile.opt.alpha_xatol", 1e-3, "precision of the best ln α",
      "Bounded refinement of ln α around the best path point.", lo=1e-8, hi=1, log=True),
])
register(T, TT, "Minimiser (Levenberg–Marquardt)", [
    S("sldprofile.opt.max_iter", 300, "maximum iterations", "", kind="int", lo=1, hi=1_000_000),
    S("sldprofile.opt.tol", 1e-10, "relative tolerance", "Stop when the objective decreases by less.",
      lo=1e-16, hi=1e-2, log=True),
    S("sldprofile.opt.lm_lambda0", 1e-3, "initial damping", "", lo=1e-12, hi=1e6, log=True),
    S("sldprofile.opt.ridge", 1e-6, "smoothness ridge (× tr/N)",
      "Makes the smoothness matrix non-singular for a free outer end; enters log det A and so the "
      "evidence.", lo=0, hi=1, log=True),
    S("sldprofile.opt.truncated_min", 0.25, "shortest step profile tried as start (fraction of N)",
      "", lo=0.01, hi=1),
])
register(T, TT, "Result and error bands", [
    S("sldprofile.fit.n_draws", 40, "posterior draws per solution", "Monte-Carlo draws for the error bands.",
      kind="int", lo=1, hi=10000),
    S("sldprofile.fit.seed", 0, "random seed of the draws", "", kind="int", lo=0, hi=2**31 - 1),
    S("sldprofile.opt.keep_weight", 1e-6, "drop solutions with relative evidence weight below", "",
      lo=0, hi=0.5, log=True),
    S("sldprofile.opt.n_out", 300, "points of the output r grid", "", kind="int", lo=10, hi=100000),
    S("sldprofile.opt.r_out_factor", 1.05, "output r range (× largest r_max)", "", lo=1.0, hi=5.0),
    S("sldprofile.fit.max_extend", 6, "times the r_max scan may be extended upwards",
      "When the evidence maximum is at the upper end of the scan.", kind="int", lo=0, hi=100),
    S("sldprofile.opt.guinier_qrg", 1.3, "Guinier range q·Rg <", "For the start value of r_max.",
      lo=0.3, hi=3.0),
    S("sldprofile.opt.extend_fraction", 0.5, "points added per upward extension of the r_max scan (× coarse)",
      "At least 3.", lo=0.05, hi=5),
    S("sldprofile.opt.fine_sd", 4.0, "fine r_max grid half width: posterior s.d. ×", "", lo=0.5, hi=20),
    S("sldprofile.opt.fine_steps", 1.5, "fine r_max grid half width: at least coarse steps ×", "", lo=0.1, hi=20),
    S("sldprofile.opt.fine_support", 1e-6, "fine r_max grid: only where the coarse posterior >",
      "relative to its maximum (plus one coarse step); 0 = no limit", lo=0.0, hi=0.1),
    S("sldprofile.opt.abandon_plateau", 0.02, "end an α path when χ² stalls (relative change over 3 steps) <",
      "and ln evidence has dropped by the stop value; 0 = never", lo=0.0, hi=0.5),
    S("sldprofile.opt.abandon_margin", 50.0, "no restarts at r_max with ln evidence below the best by",
      "0 = always restart", lo=0.0, hi=1e6),
    S("sldprofile.opt.n_workers", 0, "parallel processes for the r_max scan",
      "0 or 1 = serial", kind="int", lo=0, hi=256),
    S("sldprofile.opt.warm_start", False, "r_max scan: start each α path from the neighbouring r_max",
      "", kind="bool"),
])
register(T, TT, "Grids and scans", [
    S("sldprofile.adaptive.smoothing", 2.0, "adaptive grid: pilot over-smoothing (e-folds in α)", "",
      lo=0, hi=20),
    S("sldprofile.adaptive.max_ratio", 8.0, "adaptive grid: max. ratio coarsest / finest shell", "",
      lo=1.0, hi=1000),
    S("sldprofile.adaptive.min_width", 0.0, "adaptive grid: min. shell width (0 = min(π/q_max, r_max/N))",
      "", lo=0, hi=1e6),
    S("sldprofile.gui.p_scan_step", 0.2, "exponent scan step", "Step of 'Scan p' (0 … d−0.2).",
      lo=0.01, hi=1.0),
    S("sldprofile.gui.prior_floor", 1e-3, "max-ent 'previous result' prior floor (× max)", "",
      lo=1e-12, hi=0.5, log=True),
    S("sldprofile.gui.hollow_level", 0.05, "max-ent 'hollow shell' prior inner level", "",
      lo=1e-6, hi=1.0),
])
register(T, TT, "Two-step method: IFT (step 1) and deconvolution (step 2)", [
    S("sldprofile.ift.log_alpha_lo", -40.0, "IFT: ln α from (relative to ln tr B/tr C)", "", lo=-200, hi=0),
    S("sldprofile.ift.log_alpha_hi", 10.0, "IFT: ln α to", "", lo=-50, hi=100),
    S("sldprofile.ift.n_alpha", 101, "IFT: α grid points", "", kind="int", lo=5, hi=10000),
    S("sldprofile.ift.alpha_xatol", 1e-4, "IFT: precision of ln α", "", lo=1e-10, hi=1, log=True),
    S("sldprofile.ift.bg_below", 1.0, "IFT background search: below I_min (× scale)",
      "Scale = max(|I_min|, median σ).", lo=0, hi=100),
    S("sldprofile.ift.bg_above", 0.5, "IFT background search: above I_min (× scale)", "", lo=0, hi=100),
    S("sldprofile.ift.bg_xatol", 1e-4, "IFT background precision (× scale)", "", lo=1e-12, hi=0.1, log=True),
    S("sldprofile.twostep.penalty", "hansen_eq19", "IFT smoothness penalty", "",
      kind="choice", choices=[("hansen_eq19", "Hansen (2000) eq. 19, curvature"),
                              ("hansen_printed", "Hansen (2000) as printed, 1st difference")]),
    S("sldprofile.twostep.n_sub", 6, "deconvolution: sub-shells per shell in V(r)", "",
      kind="int", lo=1, hi=64),
])
register(T, TT, "Quadrature", [
    S("sldprofile.geo.n_gl", 16, "Gauss nodes per shell (custom exponent p)",
      "Shell amplitudes for r^p·ρ with p ≠ 0, d−1 are integrated numerically.", kind="int", lo=2, hi=256),
    S("sldprofile.geo.pddf_n_sub", 8, "output p(r): sub-shells per shell", "Display and _pr.dat only.",
      kind="int", lo=1, hi=128),
    S("sldprofile.geo.pddf_raster", 256, "output p(r) of cylinders: FFT raster", "Display and _pr.dat only.",
      kind="int", lo=32, hi=4096),
])
register(T, TT, "Instrumental resolution", [
    S("sldprofile.res.n_sigma", 3.0, "pinhole: kernel width (± n·σ_q)", "", lo=1, hi=10),
    S("sldprofile.res.oversample", 3, "pinhole: extended q grid oversampling", "", kind="int", lo=1, hi=50),
    S("sldprofile.res.slit_nodes", 16, "slit: Gauss–Hermite nodes (half range)", "", kind="int", lo=2, hi=200),
])

# ---------------------------------------------------------------------------
# Size distribution / p(r)
# ---------------------------------------------------------------------------
T, TT = "inversion", "Size distribution / p(r)"
register(T, TT, "Bayesian evidence: regularisation parameter λ", [
    S("inversion.opt.evidence_lambda_scale", "trace", "centre of the λ grid",
      "trace: tr(B)/tr(2LᵀL), the natural scale of the posterior. median: median singular value of "
      "A squared -- the rule before 2026-10-11, which misses the evidence maximum completely when "
      "most singular values are at the noise floor (finely sampled p(r)). CHANGED DEFAULT.",
      kind="choice", choices=[("trace", "tr(B) / tr(2 LᵀL)  (default)"),
                              ("median", "median singular value² (old)")]),
    S("inversion.opt.evidence_decades_below", 12.0, "λ grid: decades below the centre",
      "(median rule: always 6)", lo=0, hi=60),
    S("inversion.opt.evidence_decades_above", 4.0, "λ grid: decades above the centre",
      "(median rule: always 6)", lo=0, hi=60),
    S("inversion.opt.evidence_points", 97, "λ grid points", "(median rule: always 60)", kind="int",
      lo=5, hi=10000),
    S("inversion.opt.evidence_refine", True, "refine λ between grid points", "", kind="bool"),
    S("inversion.opt.max_lambda_reduction", 1e-3, "χ² target may lower λ at most by factor", "",
      lo=1e-12, hi=1, log=True),
    S("inversion.opt.noise_scale_lo", 1e-2, "estimated error scale s: lower limit", "", lo=1e-6, hi=1, log=True),
    S("inversion.opt.noise_scale_hi", 1e2, "estimated error scale s: upper limit", "", lo=1, hi=1e6, log=True),
    S("inversion.gui.target_chi2", 1.0, "χ²ᵣ target (when ticked)", "", lo=0.01, hi=100),
])
register(T, TT, "EM, MaxEnt, GCV", [
    S("inversion.opt.em_max_iterations", 10000, "EM: maximum iterations per trial", "", kind="int",
      lo=10, hi=10_000_000),
    S("inversion.opt.em_tol", 1e-8, "EM / MaxEnt: tolerance", "", lo=1e-16, hi=1e-2, log=True),
    S("inversion.opt.em_h_start", 0.3, "EM: start of the smoothing search h", "", lo=1e-6, hi=1),
    S("inversion.opt.em_h_floor", 1e-9, "EM: smallest h", "", lo=0, hi=0.1, log=True),
    S("inversion.opt.em_general_h_floor", 1e-7, "signed EM: smallest h", "", lo=0, hi=0.1, log=True),
    S("inversion.opt.lcurve_points", 25, "L-curve: points", "", kind="int", lo=3, hi=1000),
    S("inversion.opt.lcurve_reduction", 1.2, "L-curve: h reduction per point", "", lo=1.01, hi=100),
    S("inversion.opt.maxent_lambda_min", 1e-9, "EM+MaxEnt: smallest λ", "", lo=1e-30, hi=1, log=True),
    S("inversion.opt.maxent_lambda_max", 1.0, "EM+MaxEnt: largest λ", "", lo=1e-12, hi=1e6, log=True),
    S("inversion.opt.maxent_points", 24, "EM+MaxEnt: λ grid points", "", kind="int", lo=2, hi=1000),
    S("inversion.opt.maxent_max_iterations", 50000, "EM+MaxEnt: maximum iterations", "", kind="int",
      lo=10, hi=10_000_000),
    S("inversion.opt.maxent_sigma2", 1.0, "EM+MaxEnt adaptive prior: smoothing width σ²", "", lo=1e-6, hi=1e6),
    S("inversion.opt.signed_maxent_lambda_start", 1e-5, "signed EM MaxEnt: λ start", "", lo=1e-30, hi=1, log=True),
    S("inversion.opt.signed_maxent_lambda_floor", 1e-12, "signed EM MaxEnt: smallest λ", "", lo=0, hi=1, log=True),
    S("inversion.opt.signed_maxent_lambda_ceil", 1.0, "signed EM MaxEnt: largest λ", "", lo=1e-12, hi=1e6, log=True),
    S("inversion.opt.hansen_maxent_target_chi2", 1.0, "Hansen MaxEnt: χ²ᵣ target", "", lo=0.01, hi=100),
    S("inversion.opt.hansen_maxent_start", 1.0, "Hansen MaxEnt: λ² search start", "", lo=1e-12, hi=1e12, log=True),
    S("inversion.opt.hansen_maxent_factor", 4.0, "Hansen MaxEnt: λ² search step factor", "", lo=1.01, hi=1000),
    S("inversion.opt.lbfgs_ftol", 1e-16, "Hansen MaxEnt L-BFGS-B: ftol", "", lo=0, hi=1e-2, log=True),
    S("inversion.opt.lbfgs_gtol", 1e-12, "Hansen MaxEnt L-BFGS-B: gtol", "", lo=0, hi=1e-2, log=True),
    S("inversion.opt.lbfgs_maxcor", 30, "Hansen MaxEnt L-BFGS-B: history (maxcor)", "", kind="int", lo=3, hi=1000),
    S("inversion.opt.em_x0", 1e-6, "EM / MaxEnt: start vector level", "", lo=1e-30, hi=1e6, log=True),
    S("inversion.opt.maxent_prior_floor", 1e-6, "EM+MaxEnt constant prior: floor", "", lo=1e-30, hi=1, log=True),
    S("inversion.opt.root_xtol", 1e-6, "discrepancy root search: absolute tolerance",
      "brentq stops when |x − x*| ≤ xtol + rtol·|x*|. For parameters far below 1e-6 set xtol = 0 and "
      "rtol = 1e-6.", lo=0, hi=1, log=True),
    S("inversion.opt.root_rtol", 8.881784197001252e-16, "discrepancy root search: relative tolerance", "",
      lo=8.881784197001252e-16, hi=0.1, log=True),
    S("inversion.opt.background_maxfev", 10000, "background fit: max. evaluations", "", kind="int",
      lo=10, hi=10_000_000),
    S("inversion.opt.gcv_decades_below", 3.0, "GCV: decades below s_min", "", lo=0, hi=30),
    S("inversion.opt.gcv_decades_above", 1.0, "GCV: decades above s_max", "", lo=0, hi=30),
    S("inversion.opt.gcv_points", 100, "GCV: grid points", "", kind="int", lo=5, hi=10000),
    S("inversion.opt.tsvd_k_max", 40, "TSVD: largest truncation rank", "", kind="int", lo=2, hi=100000),
])
register(T, TT, "Instrumental resolution (4th data column σ_Q)", [
    S("inversion.res.n_sigma", 3.0, "kernel width (± n·σ_Q)", "", lo=1, hi=10),
    S("inversion.res.oversample", 3, "extended q grid oversampling", "", kind="int", lo=1, hi=50),
    S("inversion.res.fwhm", False, "4th column is a FWHM (not σ_Q)", "", kind="bool"),
])
register(T, TT, "Grid, errors, bootstrap", [
    S("inversion.gui.fine_grid_factor", 1.1, "integrate the kernel on a fine grid when Δr > factor · π/q_max",
      "", lo=0.1, hi=100),
    S("inversion.gui.r_points_factor", 1.0, "suggested r points = factor × Shannon number", "", lo=0.1, hi=20),
    S("inversion.gui.poly_degree", 3, "error model 'poly': polynomial degree", "", kind="int", lo=1, hi=6),
    S("inversion.gui.poly_half_width", 2, "error model 'poly': half width (points)", "", kind="int", lo=1, hi=50),
    S("inversion.gui.bootstrap_seed", -1, "bootstrap random seed (−1 = different every run)", "",
      kind="int", lo=-1, hi=2**31 - 1),
    S("inversion.gui.bootstrap_workers", 1, "bootstrap: parallel worker processes (1 = serial, 0 = automatic)",
      "", kind="int", lo=0, hi=64),
])

# ---------------------------------------------------------------------------
# Ornstein-Zernike
# ---------------------------------------------------------------------------
T, TT = "oz", "Ornstein–Zernike"
register(T, TT, "OZ solvers (all tabs)", [
    S("oz.solvers.max_iterations", 600, "default max. iterations (accelerated solvers)",
      "Used where a tab does not set its own limit.", kind="int", lo=10, hi=10_000_000),
    S("oz.solvers.max_iterations_picard", 5000, "default max. iterations (Picard; Mann: ÷ a)", "", kind="int",
      lo=10, hi=10_000_000),
    S("oz.solvers.picard_early_stop", True, "Picard / Mann: stop unphysical or stalled runs early", "",
      kind="bool"),
    S("oz.solvers.picard_gmin", -0.5, "Picard / Mann: unphysical when min g(r) <", "", lo=-1e6, hi=0.0),
    S("oz.solvers.convergence", 1e-12, "convergence criterion (relative progress)", "",
      lo=1e-16, hi=1e-3, log=True),
    S("oz.solvers.kinsol_maa", 5, "SUNDIALS KIN_FP: Anderson depth", "", kind="int", lo=0, hi=100),
    S("oz.solvers.biggs_maa", 2, "Biggs-Andrews: extrapolation order (sign = step formula)", "",
      kind="int", lo=-10, hi=10),
    S("oz.solvers.mdiis_history", 5, "MDIIS: history length", "", kind="int", lo=1, hi=100),
])
register(T, TT, "OZ solver tab", [
    S("oz.solve.verify_tolerance", 1e-3, "verification: max. relative |ΔS|", "", lo=1e-8, hi=1, log=True),
    S("oz.solve.transform_type", "auto", "Hankel transform type",
      "auto = type 4 (DST-IV, second order) for one component, type 1 for mixtures. Force 1 or 4 to "
      "compare.", kind="choice", choices=[("auto", "auto (4 for one component)"), ("1", "1 (DST-I)"),
                                          ("4", "4 (DST-IV)")]),
    S("oz.solve.residual_tolerance", 1e-6, "independent residual check: max. ||T(x)-x||/||x||", "",
      lo=1e-15, hi=1.0, log=True),
    S("oz.solve.verify_mann_alpha", 0.5, "verification: damping a of the Picard / Mann verifier", "",
      lo=1e-6, hi=1.0),
    S("oz.solve.mann_alpha", 1.0, "Picard / Mann: start value of the damping a", "", lo=1e-6, hi=1.0),
    S("oz.consistency.scan_steps", 28, "consistent closure parameter: scan steps",
      "'find consistent parameter' (RY, HMSA, MHNC, BPGG, CJVM, BB).", kind="int", lo=4, hi=10000),
    S("oz.consistency.tolerance", 1e-5, "consistent closure parameter: tolerance", "", lo=1e-12, hi=0.1, log=True),
])
register(T, TT, "Polydisperse (any potential) tab", [
    S("oz.generic.grid_exponent", 12, "grid exponent k (N = 2^k − 1, or 2^k for type 4)",
      "Start value of the tab's Numerical settings; a session file overrides it.", kind="int", lo=4, hi=20),
    S("oz.generic.points_per_sigma", 100, "points per mean diameter", "", kind="int", lo=4, hi=100000),
    S("oz.generic.max_iterations", 8000, "max. iterations", "", kind="int", lo=10, hi=10_000_000),
    S("oz.generic.transform_type", 1, "transform type", "", kind="choice",
      choices=[(1, "1 (DST-I)"), (4, "4 (DST-IV)")]),
    S("oz.generic.fit_max_nfev", 200, "fit: max. model evaluations (per start)", "", kind="int",
      lo=5, hi=1_000_000),
    S("oz.generic.n_starts", 10, "fit: starts of 'least_squares (n starts)'", "", kind="int", lo=2, hi=1000),
    S("oz.generic.diff_step", 1e-3, "fit: relative finite-difference step", "", lo=1e-8, hi=0.1, log=True),
    S("oz.generic.alpha_lo", 1e-6, "consistent α search: lower bracket", "", lo=1e-12, hi=10, log=True),
    S("oz.generic.alpha_hi", 100.0, "consistent α search: upper bracket", "", lo=1e-6, hi=1e6, log=True),
    S("oz.generic.alpha_tol", 2e-4, "consistent α search: tolerance", "", lo=1e-10, hi=0.1, log=True),
    S("oz.generic.alpha_accept", 1e-3, "α accepted as consistent when |residual| < value · χ⁻¹", "",
      lo=1e-10, hi=1, log=True),
    S("oz.generic.res_n_sigma", 3.0, "resolution: kernel width (± n·σ_Q)", "", lo=1, hi=10),
    S("oz.generic.res_oversample", 3, "resolution: extended Q grid oversampling", "", kind="int", lo=1, hi=50),
])
register(T, TT, "RY Polydisperse Yukawa tab", [
    S("oz.ry.alpha_lo", 0.02, "α search: lower bracket", "", lo=1e-8, hi=10, log=True),
    S("oz.ry.alpha_hi", 3.0, "α search: upper bracket", "", lo=1e-3, hi=1e4, log=True),
    S("oz.ry.alpha_tol", 2e-3, "α search: tolerance", "", lo=1e-10, hi=0.1, log=True),
    S("oz.ry.alpha_accept", 1e-3, "α accepted as consistent when |residual| <", "", lo=1e-10, hi=1, log=True),
    S("oz.ry.nsig", 5.0, "size classes: range ± n s.d. (log-normal, Schulz-Zimm)", "", lo=1, hi=12),
])
register(T, TT, "Mixture validation tab", [
    S("oz.mixture.accept_residual", 1e-3, "reject solves with residual >", "",
      lo=1e-12, hi=1, log=True),
])

# ---------------------------------------------------------------------------
# SANS reduction (pySASred)
# ---------------------------------------------------------------------------
T, TT = "sasred", "SANS reduction"
register(T, TT, "Beam centre (masked centre of gravity of the direct-beam run)", [
    S("sasred.beamcentre.radius_px", 8.0, "disc radius [px]",
      "Only pixels within this radius of the current estimate are used; the disc is re-centred "
      "until the shift is below the tolerance.", lo=1, hi=64),
    S("sasred.beamcentre.tol_px", 0.1, "tolerance [px]", "", lo=1e-4, hi=5, log=True),
    S("sasred.beamcentre.max_iter", 50, "iterations", "", kind="int", lo=1, hi=10000),
])
register(T, TT, "Default reduction parameters (starting values on the Reduction panel)", [
    S("sasred.mask.beamstop_radius_px", 6.0, "beamstop first guess: half width (square) or radius [px]",
      "Put into the mask of a new image around the beam centre; adjust it with the mask tools.", lo=0, hi=64),
    S("sasred.mask.beamstop_shape", "square", "beamstop first guess: shape", "", kind="choice",
      choices=[("square", "square"), ("disc", "disc")]),
    S("sasred.qbins.n", 60, "number of q bins", "", kind="int", lo=2, hi=2000),
    S("sasred.qbins.spacing", "log", "q-bin spacing", "", kind="choice",
      choices=[("log", "logarithmic"), ("lin", "linear")]),
    S("sasred.normalise_by", "monitor", "normalise by", "", kind="choice",
      choices=[("monitor", "monitor 1 (Poisson error propagated)"), ("time", "counting time")]),
    S("sasred.units", "nm", "units of wavelength and q", "Display, plots and output files; the "
      "reduction itself works in Å.", kind="choice", choices=[("nm", "nm and 1/nm"), ("A", "Å and 1/Å")]),
    S("sasred.detector.scale", "log1p", "colour scale of the detector image", "", kind="choice",
      choices=[("linear", "linear"), ("log1p", "log(1+I)"), ("symlog", "symlog (linear below 1 count)"),
               ("asinh", "asinh")]),
])
register(T, TT, "Mask suggestion (Suggest in the detector panel)", [
    S("sasred.mask.border_px", 2, "standard mask: outer rows and columns", "", kind="int", lo=0, hi=20),
    S("sasred.mask.drop", 0.3, "beamstop: drop below this fraction of the rim level",
      "The rim level is the 90th percentile of the (median-filtered) counts around the beam centre; "
      "the shadow is the connected region below drop x rim level that reaches the beam centre.",
      lo=0.01, hi=0.95),
    S("sasred.mask.search_radius_px", 20.0, "beamstop: search radius around the beam centre [px]", "",
      lo=3, hi=200),
    S("sasred.mask.grow_px", 1, "beamstop: grow the shadow by [px] (penumbra)", "", kind="int", lo=0, hi=10),
    S("sasred.mask.suggest_shape", "rectangle", "beamstop: suggested shape", "", kind="choice",
      choices=[("rectangle", "rectangle around the shadow"), ("shadow", "the shadow as found")]),
    S("sasred.mask.display", "colours", "masked pixels on the detector image", "", kind="choice",
      choices=[("colours", "mask colours"), ("scale", "own grey scale"), ("hidden", "hidden (outline only)")]),
    S("sasred.mask.opacity", 80, "opacity of the mask colours [%]", "The outline of the masked area is "
      "always drawn (white on black).", kind="int", lo=10, hi=100),
    S("sasred.mask.hot", True, "also suggest hot pixels", "Pixels > 5 x the 5 x 5 median and 10 sigma above it.",
      kind="bool"),
])
register(T, TT, "Reduction table", [
    S("sasred.table.autofill", True, "fill the reduction table when a folder is opened", "", kind="bool"),
])
register(T, TT, "Detector distance", [
    S("sasred.detector.distance_offset_mm", 53.0, "offset added to the recorded distance [mm]",
      "Effective sample-detector distance = recorded distance + offset.", lo=-500, hi=500),
    S("sasred.detector.distance_offset_before", "", "apply the offset to runs started before",
      "Date YYYY-MM-DD; empty: all runs.", kind="str"),
])
register(T, TT, "SANS-LLB detector geometry", [
    S("sasred.llb.central_dz_mm", 3.0, "central detector: offset of the detection layer [mm]",
      "Added to the recorded distance of the central detector (q and solid angle). Default: "
      "AgBe calibration 2026-11-10 (runs 9546/9568, see SANS-LLB_AgBe_offsets_2026-11-10). Only (L + dz)·λ-scale is measured: 1.0013 × nominal at 2 m.", lo=-500, hi=500),
    S("sasred.llb.left.dx_mm", -10.0, "left panel: offset dx along posx [mm]", "AgBe calibration 2026-11-10 (runs 9546/9568, see SANS-LLB_AgBe_offsets_2026-11-10). ±2 mm", lo=-1000, hi=1000),
    S("sasred.llb.left.dy_mm", -15.4, "left panel: offset dy along posy (along the tubes) [mm]", "AgBe calibration 2026-11-10 (runs 9546/9568, see SANS-LLB_AgBe_offsets_2026-11-10). ±2 mm", lo=-1000, hi=1000),
    S("sasred.llb.left.dz_mm", 45.0, "left panel: offset dz along the beam [mm]", "AgBe calibration 2026-11-10 (runs 9546/9568, see SANS-LLB_AgBe_offsets_2026-11-10). ±10 mm, common to both panels", lo=-1000, hi=1000),
    S("sasred.llb.bottom.dx_mm", -5.8, "bottom panel: offset dx along posx (along the tubes) [mm]", "AgBe calibration 2026-11-10 (runs 9546/9568, see SANS-LLB_AgBe_offsets_2026-11-10). ±2 mm", lo=-1000, hi=1000),
    S("sasred.llb.bottom.dy_mm", -8.3, "bottom panel: offset dy along posy [mm]", "AgBe calibration 2026-11-10 (runs 9546/9568, see SANS-LLB_AgBe_offsets_2026-11-10). ±2 mm", lo=-1000, hi=1000),
    S("sasred.llb.bottom.dz_mm", 45.0, "bottom panel: offset dz along the beam [mm]", "AgBe calibration 2026-11-10 (runs 9546/9568, see SANS-LLB_AgBe_offsets_2026-11-10). ±10 mm, common to both panels", lo=-1000, hi=1000),
    S("sasred.llb.end_px", 10, "pixels masked at both ends of every tube", "", kind="int", lo=0, hi=60),
])
register(T, TT, "Resolution", [
    S("sasred.resolution.dlam_default", 0.1, "dλ/λ (FWHM) when the raw file has none",
      "Used for Qdev when neither the selector tilt (SANS-I calibration table) nor the file gives dλ/λ; "
      "0: no resolution then.", lo=0.0, hi=1.0),
    S("sasred.resolution.source_aperture_x_mm", 50.0, "source aperture (guide cross-section) x [mm]",
      "SANS-I: divergence estimated from the collimation when a geometry has no direct beam.", lo=0.1, hi=500),
    S("sasred.resolution.source_aperture_y_mm", 50.0, "source aperture (guide cross-section) y [mm]", "",
      lo=0.1, hi=500),
    S("sasred.resolution.sample_aperture_x_mm", 8.0, "sample aperture x [mm]",
      "Typical sample aperture; the one used is not recorded in the file.", lo=0.1, hi=200),
    S("sasred.resolution.sample_aperture_y_mm", 14.0, "sample aperture y [mm]", "", lo=0.1, hi=200),
])
register(T, TT, "Transmission runs", [
    S("sasred.transmission.beamstop_out_mm", 100.0, "beam stop out of the beam beyond [mm]",
      "Distance of the beam stop from the beam centre on the detector above which a run is a transmission run.",
      lo=1, hi=2000),
    S("sasred.transmission.deadtime_warn", 1.05, "warn above this dead-time factor", "",
      lo=1.0, hi=2.0),
])
register(T, TT, "Detector efficiency", [
    S("sasred.efficiency.archive", "", "efficiency archive file",
      "Empty: efficiency_archive.h5 next to the preferences file.", kind="file",
      placeholder=lambda: _next_to_preferences("efficiency_archive.h5")),
])

# ---------------------------------------------------------------------------
# pyTOOLbox (HDF5 viewer, metadata lister)
# ---------------------------------------------------------------------------
T, TT = "toolbox", "pyTOOLbox"
register(T, TT, "HDF5 viewer", [
    S("toolbox.viewer.max_points", 4_000_000, "largest number of elements read for a plot",
      "Larger slices are read with a stride (every n-th element); the matrix view pages "
      "through the full data.", kind="int", lo=10_000, hi=500_000_000),
])
register(T, TT, "Metadata lister", [
    S("toolbox.lister.max_elements", 16, "largest dataset listed as metadata (elements)",
      "Datasets with more elements (images, histograms) are not offered as entries.",
      kind="int", lo=1, hi=100000),
    S("toolbox.lister.float_digits", 3, "digits after the decimal point of floating-point values",
      "", kind="int", lo=0, hi=15),
])

# ---------------------------------------------------------------------------
# pySCAtter
# ---------------------------------------------------------------------------
T, TT = "scatter", "pySCAtter"
register(T, TT, "Calculation and fit (pySCAtter)", [
    S("scatter.calc.auto_1d", True, "I(q) follows every parameter change", "", kind="bool"),
    S("scatter.fit.max_nfev", 200, "fit: largest number of model evaluations", "", kind="int", lo=5, hi=100000),
    S("scatter.fit.every_px", 2, "2D fit: use every n-th pixel (start value)", "", kind="int", lo=1, hi=32),
])

# ---------------------------------------------------------------------------
# pyATsas (ATSAS programs)
# ---------------------------------------------------------------------------
def _atsas_found():
    try:
        from ..atsas.core import atsas_root
        return f"{atsas_root()} (found automatically)"
    except Exception:  # noqa: BLE001
        return "not found automatically"


T, TT = "atsas", "pyATsas"
register(T, TT, "ATSAS installation and runs", [
    S("atsas.dir", "", "ATSAS installation folder",
      "Empty: found automatically.", kind="str", placeholder=_atsas_found),
    S("atsas.workdir", "", "folder for the ATSAS runs",
      "Empty: atsas_runs next to the preferences file.", kind="str",
      placeholder=lambda: _next_to_preferences("atsas_runs")),
    S("atsas.timeout", 600, "time limit of one program run (s)", "", kind="int", lo=10, hi=86400),
    S("atsas.dammif.nruns", 10, "DAMMIF: independent runs (start value)", "", kind="int", lo=1, hi=200),
])

# ---------------------------------------------------------------------------
# pyDENsity (DENSS)
# ---------------------------------------------------------------------------
def _denss_found():
    try:
        from ..density.core import find_denss
        i = find_denss()
        return f"{i.python} (DENSS {i.version})"
    except Exception:  # noqa: BLE001
        return "DENSS not found for this Python"


T, TT = "density", "pyDENsity"
register(T, TT, "DENSS installation and runs", [
    S("density.python", "", "Python with DENSS", "Empty: found automatically.", kind="str", placeholder=_denss_found),
    S("density.workdir", "", "folder for the DENSS runs", "Empty: denss_runs next to the preferences file.",
      kind="str", placeholder=lambda: _next_to_preferences("denss_runs")),
    S("density.timeout", 7200, "time limit of one map (s)", "", kind="int", lo=10, hi=7 * 86400),
    S("density.cores", 4, "denss-all: cores (start value)", "", kind="int", lo=1, hi=256),
])

# ---------------------------------------------------------------------------
# Appearance of the windows (all tools)
# ---------------------------------------------------------------------------
T, TT = "appearance", "Appearance"
register(T, TT, "Title bars of the panels (label … hide cross)", [
    S("appearance.panel_title.mode", "pale", "coloured title bars", "", kind="choice",
      choices=[("pale", "active panel in the colour, others pale"), ("all", "every panel in the colour"),
               ("active", "only the active panel"), ("off", "system look (not coloured)")]),
    S("appearance.panel_title.colour", "#0078d7", "colour (#rrggbb or name)", "", kind="str"),
])

# ---------------------------------------------------------------------------
# Shared numerics: methods used by several tools or tabs
# ---------------------------------------------------------------------------
T, TT = "numerics", "Shared numerics"
register(T, TT, "Deflation of unphysical OZ roots (OZ solver, polydisperse, RY and mixture tabs)", [
    S("numerics.deflation.p", 2.0, "power p in 1/‖x − r‖^p", "Farrell, Birkisson & Funke (2015), eq. 2.8.",
      lo=0.5, hi=8),
    S("numerics.deflation.shift", 1.0, "shift α", "The paper's alpha; results are sensitive to it.",
      lo=0, hi=1e6, log=True),
    S("numerics.deflation.cap", 1e6, "largest deflation factor", "Keeps the residual finite near a known root.",
      lo=1, hi=1e300, log=True),
])
register(T, TT, "Analytic MSA / RMSA of polydisperse Yukawa (MSA tab, RY tab, approximations)", [
    S("numerics.msa.tol", 1e-13, "MSA amplitude iteration: tolerance", "", lo=1e-16, hi=1e-4, log=True),
    S("numerics.msa.maxiter", 300, "MSA amplitude iteration: iterations", "", kind="int", lo=10, hi=100000),
    S("numerics.msa.fsolve_xtol", 1e-13, "MSA amplitude iteration: fsolve tolerance", "", lo=1e-16, hi=1e-4, log=True),
    S("numerics.msa.gamma_span", 6.0, "Γ root scan: ± span × z", "", lo=1, hi=1000),
    S("numerics.msa.gamma_decades", 10, "Γ root scan: decades below the span", "", kind="int", lo=1, hi=40),
    S("numerics.msa.gamma_per_decade", 400, "Γ root scan: points per decade", "", kind="int", lo=10, hi=100000),
    S("numerics.msa.gamma_xtol", 1e-15, "Γ root refinement: tolerance", "", lo=1e-18, hi=1e-6, log=True),
    S("numerics.rmsa.lam_max", 6.0, "RMSA: largest rescaling factor λ", "", lo=1.01, hi=100),
    S("numerics.rmsa.nscan", 120, "RMSA: λ scan points", "", kind="int", lo=10, hi=100000),
    S("numerics.rmsa.tol", 1e-12, "RMSA: λ tolerance", "", lo=1e-16, hi=1e-3, log=True),
])
register(T, TT, "Global optimisers for fits: NLopt and bumps", [
    S("numerics.nlopt.max_nfev", 400, "NLopt: max. evaluations", "", kind="int", lo=10, hi=10_000_000),
    S("numerics.nlopt.ftol", 1e-10, "NLopt: relative tolerance", "", lo=1e-16, hi=1e-2, log=True),
    S("numerics.bumps.steps", 0, "bumps: steps (0 = from samples)", "", kind="int", lo=0, hi=10_000_000),
    S("numerics.bumps.pop", 10, "bumps: population (× parameters)", "", kind="int", lo=1, hi=1000),
    S("numerics.bumps.burn", 100, "bumps DREAM: burn-in steps", "", kind="int", lo=0, hi=10_000_000),
    S("numerics.bumps.samples", 10000, "bumps: samples", "", kind="int", lo=10, hi=100_000_000),
])
register(T, TT, "Background models: start values of fitted terms (SLD profile)", [
    S("numerics.background.constant_level", 0.5, "constant: × median of the high-q tail", "", lo=0, hi=10),
    S("numerics.background.constant_tail", 0.05, "constant: high-q tail (fraction of points, ≥ 3)", "",
      lo=0.001, hi=1),
    S("numerics.background.powerlaw_level", 0.05, "power law A q^-n: A = level × I(q_min) q_min^n", "",
      lo=1e-6, hi=10, log=True),
    S("numerics.background.oz_level", 0.05, "Ornstein–Zernike: I_L = level × max I", "", lo=1e-6, hi=10, log=True),
])
