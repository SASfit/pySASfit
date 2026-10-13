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
pysasfit-sldprofile [datafile]        # this tool alone
pysasfit-gui [datafile]               # all tools as tabs; the file opens in the SLD-profile tab
python run_gui.py [datafile]          # without installing (finds ./src itself)
```

Requires PySide6 and matplotlib (`pip install PySide6 matplotlib`). The window lets you (method: direct, or Glatter's two-step route):

- load a q, I, σ [, σ_q] file, convert q units (1/nm and 1/Å) and choose the q range;
- pick the geometry, N, the regularisation (smoothness, or max-ent with a uniform, hollow-shell or previous-result
  prior), the outer end condition, a fitted background and the resolution (pinhole σ_q from column 4 or constant,
  with a Rician (default) or Gaussian kernel, σ or FWHM; or slit);
- choose the r grid: uniform, power law (γ) or adaptive (two-pass; the log reports the evidence of both grids),
  and optionally mark the shell boundaries on the profile plot;
- start from the Guinier r_max or your own value, and either fix r_max or scan it.

Fits run in the background, and the **Stop** button aborts them. The plots show:

- I(q) data with the model and normalised residuals;
- the profile with its error band. Above the plots you choose ρ(r) or the radial mass r^(d−1)ρ(r) for the left
  axis, an optional overlay on a right axis (p(r), ρ(r) or r^(d−1)ρ(r)), log r, and the y-scale of each axis: linear,
  log, symlog (linear for |y| < threshold) or asinh (smooth linear-to-log transition). The threshold is in y units;
  leave it empty for automatic (10⁻³ × max|y|). On log–log axes a centre cusp ρ ∝ r⁻ᵖ appears as a straight line;
- the posterior P(r_max);
- the evidence versus ln α.

**Save results…** writes the data files listed below plus a PNG of the plots. A simulated test file is in
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
| `pysasfit.sldprofile.regularization` | smoothness (eq. 14) and maximum-entropy (eq. 15) functionals |
| `pysasfit.sldprofile.density` | `DensityProfileFit`: Gauss-Newton/Levenberg-Marquardt solver, evidence, α-continuation, r_max posterior, error bars |
| `pysasfit.sldprofile.twostep` | Glatter's two-step route: indirect Fourier transformation, then deconvolution of p(r) |
| `pysasfit.sldprofile.models` | test profiles and simulated data |
| `pysasfit.sldprofile.gui` | PySide6/matplotlib GUI (`pysasfit-sldprofile`) |
| `pysasfit.core.resolution` | pinhole resolution with a Rician (default) or Gaussian kernel, slit, user matrix |
| `pysasfit.core.io` | reading q, I, σ [, dq] files |
| `pysasfit.core.background` | background models: constant, power law, Ornstein–Zernike, user functions (linear or nonlinear) |

See {doc}`theory` for the equations and for where this implementation deviates from the paper, and {doc}`api`
for the signatures. Settings not on the panels (α range, quadrature, Monte-Carlo draws, …) are in
**Settings → Preferences** ({doc}`../preferences`).
