# SLD profile: usage

Direct Bayesian estimation of the **radial scattering-length-density (contrast) profile ρ(r)**
of symmetric scatterers from small-angle scattering data. This is a Python implementation of

> S. Hansen, *Estimation of the density distribution from small-angle scattering data*,
> J. Appl. Cryst. **49** (2016) 856–865. doi:10.1107/S1600576716004969

ρ(r) is fitted to I(q) in one step, without first computing p(r) as Glatter's two-step method does. It works for
spheres, infinitely long cylinders and infinite lamellae. The regularisation is either Glatter's smoothness
constraint (eq. 14) or maximum entropy relative to a prior (eq. 15). The noise level (α) and the maximum radius r_max
are estimated from their Bayesian posterior.

## Graphical user interface

```bash
pysasfit-sldprofile [datafile | project.pysasfit]   # this tool alone
pysasfit-gui [datafile | project.pysasfit]          # all tools as tabs; a data file opens in the SLD-profile tab
python run_gui.py [datafile]          # without installing (finds ./src itself)
```

Requires PySide6 and matplotlib (`pip install PySide6 matplotlib`). The window lets you (method: direct, or Glatter's two-step route):

- load a q, I, σ [, σ_q] file, convert q units (1/nm and 1/Å) and choose the q range;
- pick the geometry, N, the regularisation (smoothness, or max-ent with a uniform, hollow-shell or previous-result
  prior), the outer end condition, a fitted background and the resolution (pinhole σ_q from column 4 or constant,
  with a Rician (default) or Gaussian kernel, σ or FWHM; or slit);
- choose the representation: shells (step profile, Hansen 2016) or a cubic or quadratic B-spline with N
  coefficients (smooth inside, one step at r_max; recommended when q_max·r_max is large, see {doc}`theory`);
- choose the r grid (shell edges or spline knots): uniform, power law (γ) or adaptive (two-pass; the log reports
  the evidence of both grids), and optionally mark the shell boundaries or knots on the profile plot;
- start from the Guinier r_max or your own value, and either fix r_max or scan it.
- two-step route: the number of IFT points follows the Shannon number N_s = q_max·D_max/π of the largest D_max
  of the scan (at least 15) and is updated with the q range, r_max and the scan range, until you change it by
  hand. A value more than 25 % below or above N_s is shown in bold with a comment below the field (below:
  p(r) under-sampled; above: more points than the data determine, no gain and slower -- step 2 weights
  p(r) as N_s independent values, see {doc}`theory`). The log lists N_s with every run.
- two-step route, *step-2 smoothing from*: **fit of the measured I(q)** (default; the smoothest ρ(r) whose I(q)
  fits the data within one standard deviation of χ²) or **evidence of the p(r) fit** (Glatter's p(r)-only route);
  the summary shows the α the other criterion would choose. See {doc}`theory`.
- optional panel **Two-step: χ²(α) of I(q)** (hidden by default; **View → Two-step: χ²(α) of I(q)**, shown as a tab
  next to *Evidence*): χ²/M of I(q) computed from ρ(r) along the step-2 α path, the threshold of the I(q) criterion,
  the α chosen from the fit of I(q) (red) and by the maximum evidence of the p(r) fit (dashed), the one used drawn
  bold and circled, and the step-2 evidence on the right axis. **View → Reset layout** hides it again.

Every panel -- the input controls (scrollable), *I(q) and residuals* (one panel), *Density profile*, *Posterior of r_max*, *Evidence*, the optional *Two-step: χ²(α) of I(q)* and *Result / log* -- can be rearranged and detached. Drag its title bar to move it to another side of the window or onto another panel (tabs). The detach button in its title bar (or a double-click on the title) moves it into an ordinary window of its own: it has its own task-bar entry, can be minimised and maximised, and does not stay on top of the main window. Closing that window docks the panel again. The **View** menu hides and shows panels (in `pysasfit-gui`: **View** of the launcher, for the tab in front); **View → Dock all panels** brings detached panels back and **View → Reset layout** restores the original arrangement. Every panel has a border, docked or detached; **View → Panel border** chooses *sunken* (default), *groove*, *raised*, *line* or *none*. The arrangement, including detached windows, their positions and the border, is kept in the last session; detached windows close with the main window. Plot titles are broken into several lines when a panel is narrow, and the fonts of a plot shrink with the panel (down to 60 %), so that axis labels and tick labels stay inside the panel; the panels have no large minimum size, because a panel that does not fit would be cut off at its edges (checked for 1280 × 720 to 1920 × 1080 and screen scaling 100–150 %). The first line of the log tells which copy of the program runs (folder and date of `gui.py`).

Fits run in the background, and the **Stop** button aborts them. The plots show:

- I(q) data with the model and normalised residuals;
- the profile with its error band. Above the plots you choose ρ(r) or the radial mass r^(d−1)ρ(r) for the left
  axis, an optional overlay on a right axis (p(r), ρ(r) or r^(d−1)ρ(r)), log r, and the y-scale of each axis: linear,
  log, symlog (linear for |y| < threshold) or asinh (smooth linear-to-log transition). The threshold is in y units;
  leave it empty for automatic (10⁻³ × max|y|). On log–log axes a centre cusp ρ ∝ r⁻ᵖ appears as a straight line;
- the posterior P(r_max);
- the evidence versus ln α. Local maxima are marked and labelled when there is more than one.

**Manual ln α (add-on to the automatic choice).** Click on the *Evidence* plot (or on the *Two-step: χ²(α) of I(q)*
plot) with no pan/zoom mode active, or tick **manual ln α** above the Evidence plot and type a value. The solution at
that α -- at the best r_max; in the two-step route step 2 at r_max = D_max/2 -- is drawn in addition, dash-dotted in
pink, in the I(q) plot (with its residuals as ×), in the profile plot and as a vertical line in the evidence and
χ²(α) plots. The line below the controls gives Δln Ev relative to the maximum, χ²/M of I(q) (two-step: also of p(r)),
Rg and N_g, and the log keeps every value tried. The automatic result, the summary and the saved files are not changed;
**automatic** removes the manual curves, and a new fit switches them off. Typical uses: an evidence curve with two
maxima, or checking how sensitive the profile is to α. A result restored from a project or the last session has no
fitter attached; run the fit again to use this. In Python: `ProfileResult.solve_at(ln_alpha)` and
`TwoStepResult.solve_at(ln_alpha)`.

**File → Save project…** writes a project file (`*.pysasfit`): the data (file name *and* an embedded copy of
the numbers), the setting of every control and the last result (incl. a p-scan); **File → Open project…**
restores it, also when the data file has moved (the embedded copy is then written next to the project and loaded;
the log says so). The last session is saved automatically in the same format (after loading data, after a fit and
when the window closes) as `session_sldprofile.pysasfit` next to the preferences file, and reopened at the next
start; **Settings → Forget last session** deletes it. `pysasfit-gui` and `pysasfit-sldprofile` also accept a project
file on the command line.

Projects carry a format version (`format_version` in `project.json` inside the ZIP file). When a project is opened,
all controls are first reset to their defaults, so every setting the file does not contain -- e.g. a parameter
added in a later version -- keeps its **default value**; entries the program does not know are ignored and
listed in the log. Details: `pysasfit.core.project`.

**Save results…** writes the data files listed below plus one PNG per plot (`_iq`, `_profile`, `_rmax`, `_evidence`). A simulated test file is in
`examples/sldprofile/data/inhomogeneous_sphere_r5nm.dat` (q in 1/nm). Two more are in q in 1/Å:
`examples/sldprofile/data/gaussian_coil_Rg217A.dat`, a Debye-coil test case for the radial-mass representation, and
`examples/sldprofile/data/star_p4over3_bg0.05.dat`, a star profile ρ ∝ r^(−4/3) with background 0.05 for the p-scan.
`examples/sldprofile/data/coil_with_porod_upturn.dat` (also q in 1/Å) is the same coil plus a low-q Porod upturn and a constant,
a test case for the background models.

## Resolution: the same operator as the OZ tools

The pinhole resolution (`pysasfit.core.resolution`) is the same linear operator as `pysasfit.oz.polydisperse_fit.Resolution`, and the weight matrices
are identical:

- **Kernel:** Rician by default, because Q is broadened in the 2-D detector plane and then radially averaged:
  P(Q|Q₀) = (Q/σ²)·exp(−(Q²+Q₀²)/2σ²)·I₀(QQ₀/σ²). A Gaussian is available as an option.
- **dQ:** the fourth data column is read as σ; pass `fwhm=True` for a FWHM.
- **Model grid:** extended, including the measured Q, with trapezoid weights and normalised rows.
- **Background:** not smeared.

The data loader (`pysasfit.core.io.load_data`) follows the OZ tools' `loadCurve` conventions: comment lines `#`, `%` and `//` are skipped, q ≤ 0 is
dropped, and the data are sorted by q.

```python
from pysasfit.sldprofile import Resolution
fit = DensityProfileFit(q, I, sig, resolution=Resolution(dq))                  # Rician, dq = sigma
fit = DensityProfileFit(q, I, sig, resolution=Resolution(dq, kernel="gaussian", fwhm=True))
```

## Quick start

```python
import numpy as np
from pysasfit.sldprofile import DensityProfileFit, SlitResolution, load_data

q, I, sig, dq = load_data("micelle.dat")              # columns q, I, sigma [, dq]
fit = DensityProfileFit(q, I, sig,
                        geometry="sphere",            # 'sphere' | 'cylinder' | 'lamella'
                        n=80,                          # number of density points (paper: 80)
                        regularization="smooth",      # 'smooth' (eq. 14) | 'maxent' (eq. 15)
                        outer="free",                 # free end point at r_max, or 'zero'
                        background=True,              # flat background (Hansen's free p(0))
                        resolution=None)              # Resolution(dq) [Rician], SlitResolution(c), MatrixResolution
res = fit.fit(verbose=True)                           # alpha and r_max by maximum evidence
print(res.summary())
res.save("micelle")                                   # *_rho.dat, *_fit.dat, *_pr.dat, *_rmax.dat, *_params.txt
res.plot(data=(q, I, sig))
```

Coil-, star- or corona-like profiles (power-law I(q) without minima): represent and smooth the radial mass
r^(dim−1)ρ instead of ρ. This allows the 1/r² centre cusp that produces I ~ q⁻² without form-factor zeros:

```python
fit = DensityProfileFit(q, I, sig, basis="mass", outer="zero", background=True)
fit = DensityProfileFit(q, I, sig, basis=4/3, background=True)          # r^p rho with any p < dim
p, lnEv, sols = fit.scan_exponent(np.arange(0, 2.9, 0.2), r_max=450)   # choose p by evidence
fit = DensityProfileFit(q, I, sig, background_value=0.05)               # known, fixed background
```

Fit the background while scanning p: a slightly wrong fixed background biases the selected exponent.

Non-constant and nonlinear backgrounds (fitted together with ρ; nonlinear parameters are fine):

```python
from pysasfit.sldprofile import Background, Constant, PowerLaw, OrnsteinZernike, FunctionTerm
bg = Background([Constant(), PowerLaw(n=4.0, fit_n=True)])           # c + A q^-n (low-q upturn)
bg = Background([Constant(), OrnsteinZernike(xi=20.0, fit_xi=True)])  # c + I_L/(1+q^2 xi^2)
bg = Background([FunctionTerm(lambda q, a, b: a * np.exp(-b * q), ["a", "b"], [1.0, 10.0], positive=[True, True])])
fit = DensityProfileFit(q, I, sig, basis="mass", background=bg)
```

Shell grid (see {doc}`theory`): uniform (default), power law, or adaptive (equidistribution of the profile slope,
never finer than the data resolution π/q_max). The adaptive grid pays off for small N:

```python
f0 = DensityProfileFit(q, I, sig, n=30, basis=4/3, background=True)
g = f0.adaptive_grid(r_max=450)                        # from an over-smoothed first pass
fit = DensityProfileFit(q, I, sig, n=30, basis=4/3, background=True, grid=g)
fit = DensityProfileFit(q, I, sig, grid=("power", 1.5))  # finer towards the centre
```

B-spline representation instead of shells (n = number of coefficients; knots uniform, power law or adaptive):

```python
from pysasfit.sldprofile import BSplines
fit = DensityProfileFit(q, I, sig, n=20, basis=4/3, background=True, grid=BSplines(3))       # cubic
fit = DensityProfileFit(q, I, sig, n=20, grid=BSplines(2, ("power", 0.7)))                  # quadratic, finer at r_max
g = fit.adaptive_grid(r_max=450)                       # -> BSplines(3, breakpoints)
fit2 = DensityProfileFit(q, I, sig, n=20, basis=4/3, background=True, grid=g)
```

Maximum entropy with a prior shape (a function of x = r/r_max, scaled to the data automatically):

```python
shell = lambda x: np.where(x < 0.3, 0.05, 1.0)          # e.g. hollow shell, r_in/r_out = 0.3
fit = DensityProfileFit(q, I, sig, regularization="maxent", prior=shell)
res = fit.fit()
# Hansen: iterate, using the previous result as the new prior
fit2 = DensityProfileFit(q, I, sig, regularization="maxent", prior=res.best.rho)
```

Lower-level access:

```python
sol, path = fit.best_alpha(r_max=5.0)       # max-evidence alpha at fixed r_max
sol = fit.solve(alpha, r_max)               # one regularised solution
r_grid, post, *_ = fit.scan_r_max(np.linspace(4, 6, 21))   # P(r_max), cf. paper Fig. 1(c)
r, p = sol.pddf()                           # distance distribution from rho (eqs. 4-5)
```

`examples/sldprofile/demo_hansen2016.py` reproduces the simulated examples of the paper (homogeneous sphere
with slit smearing, inhomogeneous sphere with smoothness and max-ent constraint).

## Package layout

| module | content |
|---|---|
| `pysasfit.sldprofile.geometry` | exact shell amplitudes for sphere/cylinder/lamella, p(r) from ρ(r), Rg |
| `pysasfit.sldprofile.bspline` | B-spline representation: amplitudes by Gauss quadrature per knot span, exact ∫c″² prior, end conditions |
| `pysasfit.sldprofile.regularization` | smoothness (eq. 14) and maximum-entropy (eq. 15) functionals |
| `pysasfit.sldprofile.density` | `DensityProfileFit`: Gauss-Newton/Levenberg-Marquardt solver, evidence, α-continuation, r_max posterior, error bars |
| `pysasfit.sldprofile.twostep` | Glatter's two-step route: indirect Fourier transformation, then deconvolution of p(r) |
| `pysasfit.sldprofile.models` | test profiles and simulated data |
| `pysasfit.sldprofile.gui` | PySide6/matplotlib GUI (`pysasfit-sldprofile`) |
| `pysasfit.core.resolution` | pinhole resolution with a Rician (default) or Gaussian kernel, slit, user matrix |
| `pysasfit.core.project`, `pysasfit.core.project_gui` | project files and the last session |
| `pysasfit.core.docking` | detachable panels (View menu) |
| `pysasfit.core.io` | reading q, I, σ [, dq] files |
| `pysasfit.core.background` | background models: constant, power law, Ornstein–Zernike, user functions (linear or nonlinear) |

See {doc}`theory` for the equations and for where this implementation deviates from the paper, and {doc}`api`
for the signatures. Settings not on the panels (α range, quadrature, Monte-Carlo draws, …) are in
**Settings → Preferences** ({doc}`../preferences`).
