# Next session -- open items only

What remains to be done. Completed work, corrected mistakes and the reasoning
behind decisions have been moved to `SESSION_NOTES.md`; consult that before
re-investigating anything here, because several things that look open were
closed by measurement and several plausible-sounding hypotheses were tested
and disproved.

**One rule, learned expensively.** Of the entries in the previous version of
this file that were written from recollection rather than from a measurement,
roughly two thirds turned out to be wrong -- describing defects that did not
exist, or work already deployed. Everything backed by a number held up.
**Treat any claim here without a measurement attached as unverified, and
spend the two minutes checking before spending hours fixing.**

---

## 1. Manuscript, before submission

* **A FIT TO REAL MEASURED DATA -- THE DATA NOW EXIST.**
  `STJ1_39_2__2_07e-6.dat` supplied by Joachim: spherical SHELL particles at
  a nominal volume fraction of 0.4, with a constant incoherent background.
  Characterised, and it is well suited:

  | property | value |
  |---|---|
  | points | 171, Q = 0.018 to 1.183 1/nm |
  | statistics | median dI/I = 2.1 % |
  | 4th column dQ | PRESENT -- dQ/Q from 0.55 at the lowest Q to 0.049 at high Q |
  | structure peak | Q* = 0.090 1/nm, I = 96.8 |
  | low-Q suppression | I falls to 13.2 at Qmin, a factor 7.4 |
  | background | flat tail at I ~ 0.6 from Q ~ 1.0 |

  **UNITS: Q is in 1/nm, so all lengths are in nm** (Joachim). Every size
  below follows from that, and an earlier version of this note had them all
  ten times too small by assuming 1/A.

  The resolution column matters more here than usual: dQ/Q = 0.55 at the
  first point, so without smearing the low-Q points are badly misfitted and
  the polydispersity comes out biased HIGH (15 % on synthetic data, and
  silently -- chi^2 stays acceptable). Read with
  `loadCurve(path, withResolution=True)`; the column is the Gaussian SIGMA,
  not the FWHM.

  Peak position gives a nearest-neighbour spacing 2*pi/Q* = 70 nm. For hard
  spheres at phi = 0.39 the peak sits near Q*sigma ~ 7, so sigma ~ 78 nm and
  the radius ~ 39 nm.

  CAUTION ON READING THE FILENAME. The `39` is the VOLUME FRACTION (0.39),
  not a radius -- confirmed by Joachim. That the radius estimate is also ~39
  is pure coincidence, and an earlier version of this note treated the
  agreement as confirmation. `2_07e-6` is presumably the SLD; its units are
  NOT confirmed (2.07e-6 is a common value in 1/A^2, which would be
  2.07e-8 nm^-2, so the SLD may follow a different convention from the
  lengths -- check before fixing it in a fit).

  **This makes the test much stronger than a free-parameter fit.** phi is
  KNOWN to be 0.39, so the fitted value is a genuine check rather than
  something the model can absorb. Given the phi-R correlation of 0.75, a
  fitted phi that lands near 0.39 with a physically sensible radius is real
  evidence; one that drifts to 0.25 or 0.55 while still fitting the curve
  would be evidence of the opposite, and worth reporting either way.

  Suggested sequence: fit with phi FREE first and see where it goes -- that
  is the honest test. Then fit with phi FIXED at 0.39 and compare chi^2; if
  the two agree the curve genuinely determines phi, and if fixing it costs
  little in chi^2 but changes the radius substantially, the correlation is
  doing the work rather than the data.

  **CORE-SHELL SUPPORT IS NOT USABLE YET. Three defects, found while
  preparing this fit, in increasing order of seriousness.**

  1. *Wrong call signature, FIXED.* Both call sites in
     `generic_polydisperse_tab.py` read
     `CoreShell(p["shell"], p["rhoCore"], p["rhoShell"])` against a signature
     `CoreShell(rho_core, rho_shell, rho_solvent, thickness, ratio)`, so the
     thickness went in as rho_core and `thickness` was left None -- which the
     constructor rejects outright. Now passed by keyword.

  2. *The form factor never reaches the fitter.* `PolydisperseFit.__init__`
     has NO formfactor argument, and the GUI builds `ff` then does not pass
     it. So the FIT path always uses `Sphere()` whatever the user selects;
     only the CALCULATE path honours the choice. Fixing this needs a
     `formfactor` parameter on `PolydisperseFit`, threaded through to
     `GenericPolydisperseSAS`.

  3. *The thickness cannot vary during a fit.* `ff` is constructed once,
     outside the optimiser loop, so even once passed it is frozen. To fit
     Delta R the form factor must be REBUILT each iteration from the current
     parameter vector. `shell` also needs adding to `FIT_BOUNDS`.

  **AND THE PARAMETRISATION IS WRONG FOR THIS SAMPLE.** `CoreShell` is
  polydisperse in the OUTER radius (`R_c = R - t`, and `outer_radius(R)`
  returns R). For this sample **the polydispersity is on the CORE**
  (Joachim), so what is needed is `R_outer = R_core + Delta R` with the size
  distribution applied to `R_core`.

  That is not bookkeeping. With a constant shell on a polydisperse core the
  outer-radius distribution is the core distribution SHIFTED, not scaled, so
  its relative width is smaller than the core's; and the hard-core diameter
  entering the structure factor is `sigma = 2(R_core + Delta R)`. Using the
  existing class would bias both the fitted polydispersity and the effective
  volume fraction.

  So add a mode -- or a separate class -- with the distribution on the core,
  `outer_radius(R) = R + Delta R`, and `Delta R` fittable. Joachim's stated
  preference is to parametrise by the shell thickness Delta R rather than by
  a core/outer ratio, which is also the unambiguous choice: the `ratio` mode
  makes the shell scale with particle size, a different physical model.

  Recommended order: (2) thread the form factor through, then the core-
  polydisperse parametrisation, then (3) per-iteration rebuild. Test each
  against the CALCULATE path before fitting anything -- a form factor that
  silently defaults to `Sphere()` produced no error at all, only a wrong fit.

  **The shell thickness is to be FITTED**, with an expected value around
  2.7 nm (Joachim). On a particle of radius ~39 nm that is about 7 % of the
  radius -- a thin but real shell, so the core-shell contrast should be
  visible in the mid-Q region rather than the form factor degenerating to a
  solid sphere. Expect core radius and shell thickness to be correlated, on
  top of the phi-R correlation; report both.

  So the free parameters are: RADIUS (Joachim determines 32 nm for this data
  set -- use as the starting value, not as a fixed one), shell thickness
  (~2.7 nm), polydispersity, SCALE, background, phi, and the SHELL SLD.
  Only the CORE SLD is fixed. The shell SLD starts with the opposite sign to
  the core; the SLD convention is 1/cm^2.

  **A discrepancy worth resolving during the fit.** The peak-position
  estimate above gives ~39 nm, against Joachim's measured 32 nm. With
  R = 32 nm the hard-sphere diameter is 64 nm, so Q*sigma = 0.090 x 64 = 5.8,
  whereas the Percus-Yevick structure peak at phi = 0.39 sits near
  Q*sigma ~ 7. Either the effective interaction diameter exceeds 2R -- a
  shell or a soft repulsion holding particles apart -- or the maximum at
  Q = 0.090 has form-factor contribution mixed into it. That bears directly
  on whether HardSphere is the right potential here, so check it rather than
  assuming: compare the fitted effective diameter against 2(R + shell), and
  if it is substantially larger, try a Yukawa or a square shoulder.

  Note also that with the scale, the shell SLD and phi all free, there are
  now three quantities that can trade against each other in the forward
  amplitude. The scale and the contrast are nearly degenerate at fixed shape;
  what breaks the degeneracy is the SHAPE of the form factor, particularly
  the position and depth of the minimum produced by the sign change. Expect
  the fit to be well determined in shape and poorly determined in overall
  amplitude, and report the full correlation matrix rather than parameter
  errors alone.

  **The contrasts have OPPOSITE SIGNS for core and shell** (Joachim). Only
  the core SLD is fixed; the shell SLD is FITTED, starting with the opposite
  sign.

  **The sign change makes this an unusually good test of the paper's central
  claim, and it should be exploited rather than merely noted.** With opposite
  contrasts the form-factor amplitude F(Q) passes through zero. The
  decoupling approximation rests on beta(Q) = <F>^2/<F^2>, and near a sign
  change <F> can nearly vanish while <F^2> does not, so beta dips sharply.
  Decoupling should therefore fail QUALITATIVELY here, not by a few per cent
  -- exactly the regime where the exact multicomponent treatment earns its
  cost.

  The figure to produce is therefore not simply "exact fits the data" but the
  exact result and the six approximations overlaid on the same MEASURED
  curve, with residuals. If decoupling misfits visibly near the form-factor
  minimum while the exact treatment does not, that is the paper's argument
  demonstrated on real data instead of synthetic, and is worth more than the
  fitted parameters.

  Caution: a sign-changing F also puts a deep minimum in the intensity, where
  the fit is most sensitive to polydispersity and resolution TOGETHER.
  Smearing fills in such minima, so the dQ column matters doubly here -- an
  unsmeared fit will compensate with excess polydispersity, which is exactly
  the 15 % bias measured on synthetic data.

  **SCALE AND BACKGROUND SHOULD GET GUI ENTRIES (Joachim).** The model is
  `I(Q) = scale * I_model(Q) + background`. At present both are solved
  EXACTLY at every iteration by weighted linear least squares inside
  `polydisperse_fit`, and never appear in the interface. That default is
  right and should stay -- they enter linearly, so a linear solve is exact
  and free, whereas handing them to the nonlinear optimiser would add the two
  most strongly correlated parameters (scale against volume fraction,
  background against everything at high Q) for no gain.

  What is missing is CONTROL and VISIBILITY:

  * **show the solved values** in the interface after every calculate and
    fit. They are already returned (`payload['scale']`), so this is display
    only, and it matters: a scale far from unity, or a negative background,
    is a diagnostic that something is wrong with the contrast or the
    subtraction;
  * **allow either to be FIXED.** A measured incoherent background is often
    known independently -- for the present data set it is visible directly as
    the flat tail at I ~ 0.6 above Q ~ 1 -- and fixing it removes a degree of
    freedom that otherwise absorbs high-Q misfit. Likewise an absolute-scale
    measurement fixes the scale, turning it from a free amplitude into a
    check.

  Implementation. In `polydisperse_fit` the linear solve currently fits both;
  it needs to fit only the ones not held fixed, subtracting a fixed
  background from the data and dividing out a fixed scale before the solve.
  Both cases reduce to a smaller least-squares problem rather than a special
  case. Then two entries plus two "fix" checkboxes in the tab, following the
  pattern already used for the shape parameters.

  Test it the way the form-factor defects should have been caught: set a
  known scale and background, generate data, and check the fit recovers them
  when free and respects them when fixed. A fixed parameter that is silently
  refitted looks exactly like a good fit.

* **Administrative.** Author list, acknowledgements, funding, data
  availability. The Software Availability section has placeholders for
  repository URL, version tag or DOI, and licence.

* **AI-assistance disclosure is DRAFTED** (see the section at the end of
  `manuscript.tex`). Check it against IUCr's own policy, which may prescribe
  wording or placement, and confirm the description of the division of labour
  matches your own view of it.

* **Transfer to the IUCr document class.** Note IUCr normally uses
  author-year citations; the manuscript is currently numbered
  (`\usepackage[numbers,sort&compress]{natbib}` with `unsrtnat`). Reverting
  is two lines, documented in the preamble.

---

## 2. Report

* **Multiply-defined labels**, `sec:solvers` and `sec:closures` -- both
  fragments define them. NOT bibliography-related. This is the one problem
  here that fails SILENTLY: a `\ref` resolves to whichever Part LaTeX saw
  last, so a cross-reference can point at the wrong section with nothing in
  the log. Prefix per part (`sec:doc:`, `sec:supp:`) and grep both fragments
  for other collisions while there.

* **Confirm the build** after the bibliography merge: `sh build_report.sh`
  should report 0 undefined citations and far fewer than the 44 overfull
  boxes that prompted the `hyphenat` fix.

* **Content still predating recent work.** The report does not yet describe
  the type-4 transform, the quantile-transform quadrature,
  `gazzillo_msa.py` / `gazzillo_shs.py` / `mixscatter_bridge.py`,
  `deflation.py`, or `examples/educational_oz.py`.

---

## 3. Numerical

* **GRID SIZE IS WRONG FOR TYPE 4, and the benchmark is unfair to it.**
  The two transforms want different lengths:

  | N | DST-I (type 1) | DST-IV (type 4) |
  |---|---|---|
  | 16383 = 2^14 - 1 | **0.263 ms** | 0.308 ms |
  | 16384 = 2^14 | 0.599 ms | **0.099 ms** |
  | 16385 | 2.497 ms | 0.303 ms |
  | 16000 | 2.433 ms | 0.105 ms |

  DST-I is computed through an FFT of length 2(N+1), so N = 2^k - 1 makes
  that a power of two -- 9.5x faster than a bad size. DST-IV uses length N
  itself, so it wants N = 2^k, and 16383 = 3 x 43 x 127 is a bad size for
  it: 3.1x slower than 16384.

  All our grids are 2^k - 1 (4095, 8191, 16383), chosen for type 1 and
  inherited by type 4. **So every type-4 timing was taken on a grid that
  penalises it by about a factor of three.** That explains something recorded
  earlier as unexplained: type 4 measured SLOWER than type 1 at 400 and 800
  points per diameter, which made no sense for a transform doing the same
  work. It was the grid, not the transform.

  To fix: choose N from the transform type -- 2^k - 1 for type 1, 2^k for
  type 4 -- rather than passing one number for both. Then re-run
  `tools/transform_test.py` and `tools/benchmark.py`, and correct the
  manuscript's cost table, which currently understates type 4.

* **Type 4 for mixtures -- three routes, none taken.** Type 4 is second order
  and about 140x more accurate at the default grid, but needs every pair core
  between grid points, which quadrature nodes cannot satisfy. It is deployed,
  correct and available for ONE COMPONENT. For mixtures:

  (a) keep type 1 and accept first order (current default);
  (b) equally spaced size classes, which makes alignment achievable --
      MEASURED as costing 2-9 % on `<sigma^6>` against quadrature's machine
      precision, but that test used a fixed +-2.5 sigma window, so the error
      is window truncation rather than spacing as such and does not fall with
      p. **Rerun with the window widening as p grows before writing route (b)
      off.** If `<sigma^6>` then converges toward 1e-4 by p = 10 the trade is
      worth recomputing; if it stalls near 1 %, route (b) is dead;
  (c) a formulation not requiring every core on one grid.

* **`r <= sigma` for the five potentials that use `r < sigma`.** Measured as
  six times more accurate for ONE component, and VACUOUS for mixtures (the
  conventions differ only over a grid point at exactly sigma, and with
  irrational sigma_ij there is none -- identical to every digit across four
  resolutions). So this is a consistency fix only: `HardSphere` and
  `SquareWell(epsilon=0)` still differ for one component and should not.
  Small, no effect on mixtures, not urgent.

* **BPGG at phi = 0.4, alpha = 0.5: is there a second root?** Open. Anderson
  gives 5.252805 with |F| = 3.7e-12, a genuine fixed point. Newton-Krylov
  cannot converge there at all, so deflation through it can say nothing.
  Needs a driver that converges on that state.

---

## 4. Features not built

* **PUT EACH FIT CHECKBOX NEXT TO ITS OWN ENTRY** (Joachim), and widen the
  entry fields by 3-5 characters.

  At present the entries are in one panel and the "vary" checkboxes in
  another, so the user matches parameter names across two lists. Inline is
  plainly better: one row per parameter, value and checkbox together.

  Not a cosmetic change, but simpler than it first looks. Joachim's
  clarification settles the awkward part: a checkbox for a parameter that is
  not currently applicable is GREYED OUT AND INACTIVE rather than removed.
  Nothing is created or destroyed on a toggle, so the layout never reflows
  and there is no rebuild that can leave a stale checkbox behind -- which was
  the failure mode worth worrying about, since a surviving checkbox would
  fit a parameter the model is not using.

  So the work is: create each checkbox beside its entry once, and have the
  existing `_rebuildFitFlags` callers set `state` instead of destroying
  `fitVarFrame`. The shared `entry()` helper needs to know the parameter
  name so it can make the matching checkbox.

  Conditionally applicable, i.e. greyed unless the condition holds:
  `shell` and `rhoRatio` (core-shell form factor), `linkC` (delta link on),
  `porodA` and `porodD` (power law enabled), `closureParam` (closure takes
  one), `pot0..potN` (potential's own arguments, and the linked one stays
  greyed because it is computed from dR).

  Run `gui_path_test.py` before and after: it exercises both form factors
  and both power-law settings, so it would catch a checkbox left active when
  it should be disabled.

* **COUPLED PARAMETER CONSTRAINTS need NLopt.** Box bounds are handled by
  `least_squares` and `A >= 0` is enforced exactly in the linear solve (the
  bound is either inactive, or active and the coefficient pinned at zero with
  the rest re-solved -- exact for a convex quadratic, not a clip). What
  cannot be expressed today is any constraint RELATING two parameters.

  The concrete one: **0 < dR < R**. The lower half is already a box bound,
  and deliberately strict -- at dR = 0 the core-shell form factor degenerates
  to a sphere and the two SLDs stop being separately identifiable, so the
  optimiser can park there and report convergence on a model with fewer
  effective parameters than it believes.

  The upper half is the new part, and it is subtler than it looks. The fitted
  `meanRadius` is the MEAN core radius, and the distribution has width
  `srel`, so the smallest class has radius about R(1 - n*srel) with n the
  quadrature window (2.5). At srel = 0.116 the smallest core is ~0.71 R, so
  the shell must fit inside THAT, not inside the mean:

      dR < R * (1 - n * srel)

  A fit driving dR toward R would produce negative inner radii in the small
  classes long before the mean was reached. Still a simple function of the
  parameter vector, so NLopt handles it the same way -- it just binds sooner
  and for a better reason.

  **Why NLopt specifically.** `opt.add_inequality_constraint(f, tol)` takes
  an arbitrary f(x) <= 0, so `x[shell] - x[radius]*(1 - n*x[srel])` is direct
  to write. Only some algorithms accept constraints: AUGLAG (which WRAPS any
  other optimiser, so an evaluation-frugal local method can be kept -- the
  right choice when one evaluation is a 2.5 s OZ solve), SLSQP, COBYLA, and
  the global ISRES. `pip install nlopt` provides the official binding.

  Not worth adding NLopt for its algorithm portfolio -- bumps, multi-start
  and deflation already answer the global-search question. Add it for the
  constraints, which nothing else here provides.

  **Watch for over-constraining.** delta = c*dR ties the interaction range to
  the shell, so bounding dR also bounds delta. Decide which constraints are
  independent before adding them.

* ~~Distribution selector in the generic polydisperse tab~~ -- DONE. A
  combobox offers Schulz, Gaussian, LogNormal and Weibull, threaded through
  the fit path, the calculate path and the thermodynamic-consistency search
  (which solves the same model and would otherwise have tuned alpha for a
  Schulz system and applied it to a Weibull one). Measured: at the same mean
  and relative width the four differ by up to 78 % in I(Q), because the
  quadrature is moment-matched for each but the TAIL differs and
  <sigma^6> governs I(Q -> 0). That makes the distribution a real fitting
  choice rather than a convention.

* **bumps `de`/`dream` never run to completion**, so the uncertainty output
  is untested plumbing. NOT verified either way -- bumps was not installed
  where this was checked. `de` and `dream` share a code path and differ only
  in the fitter, so running one exercises most of the other.

---

## 5. Validation not yet done

* **The charge-coupled polydisperse route is the least-tested path.**
  `gazzillo_msa.py` provides an exact analytic reference for polydisperse
  CHARGED hard spheres, but the two have not been compared because they are
  different models: Gazzillo uses the primitive model with explicit
  microions, ours follows D'Aguanno and Klein in eliminating them into an
  effective screened-Coulomb macroion potential. A disagreement would not
  automatically indicate a defect. **Designing a comparison where the two
  models should agree is the actual task.**

* **sasmodels beta(q)** as an external check on `I_decoupling`. Deliberately
  last: the sasmodels developer has reported in person that their beta(q)
  implementation has had problems, so a disagreement would be ambiguous.

* **Blum & Hoye, J. Stat. Phys. 19, 317 (1978)** -- finite-range
  multicomponent multi-Yukawa MSA. Extending `gazzillo_shs.py` from the
  sticky limit back to finite screening would give an exact reference for the
  double-Yukawa data, reusing dyadic structure that already works. Papers in
  the `4Claude` folder.

* **Gazzillo, Giacometti, Fantoni & Sollich, Phys. Rev. E 74, 051407 (2006)**
  (in `4Claude`, read). Derives how stickiness should depend on particle size
  by mapping four realistic potentials onto an equivalent SHS through the
  second virial coefficient, finding that almost all give
  `t_ij ~ C1 M(X) (s_i s_j/s_ij^2) + C2 M(X) (s_i s_j/s_ij^2)^2 + ...`.
  Two consequences: `gazzillo_shs.py` uses a FACTORISABLE `K_ij = Y_i Y_j`,
  which that polynomial is not, so our Models I and II are special cases;
  and it gives a citable criterion for when the reduced-tail rule
  `u_ij(r) = u(r/sigma_ij)` is defensible, replacing a hand-wave in the
  manuscript. Read the derivation before changing anything.

* **Duh & Henderson (1996)**, J. Chem. Phys. 104, 6742 -- improvement on
  Duh-Haymet for Lennard-Jones mixtures, explicitly studying deviations from
  Lorentz-Berthelot. Relevant to the reduced-tail mixing rule; not read.

* **SCOZA: cite, do not implement.** Its consistency condition is a PDE over
  the (rho, beta) plane, so it is a different solver rather than a new
  closure, and it needs a mainly attractive perturbation, which excludes
  charged repulsive colloids. But it is the right citation for the
  near-critical region where our solvers fail.

---

## 6. The habit worth keeping

Every defect found in this project came from comparing against something
built by someone else, or from exercising a path that had never run. None
came from internal consistency checks, and several sat inside results already
reported as validated.

Three rules earned the hard way:

1. **Check the residual, not the flag.** `||T(x) - x||` costs one operator
   evaluation. A solver reporting convergence returned a point with
   |F| = 19.4; nothing else in the pipeline would have caught it.
2. **Run the control first.** An experiment without one measures the harness,
   not the problem. This happened three times in a single session --
   a deflation test that fenced off a root the solver was not going to, a
   hand-rolled solver that failed undeflated too, and a convergence study
   read from adjacent ratios instead of endpoints.
3. **Prefer library code to a loop written in one sitting**, and prefer a
   measurement to a recollection.
