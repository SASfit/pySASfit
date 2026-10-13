# SLD profile

`pysasfit.sldprofile` estimates the radial scattering-length-density profile ρ(r) of spheres, cylinders and
lamellae from small-angle scattering data. Two routes are available:

- **direct** Bayesian estimation of ρ(r) in one step (Hansen, *J. Appl. Cryst.* **49** (2016) 856), with a
  smoothness or maximum-entropy constraint and the noise level α and r_max from their posterior;
- **two-step** (Glatter): indirect Fourier transformation I(q) → p(r), then deconvolution p(r) → ρ(r).

The GUI is `pysasfit-sldprofile`, or the *SLD profile* tab of `pysasfit-gui`.

```{toctree}
:maxdepth: 2

usage
theory
```

The API reference is in the {doc}`API section <../api/index>`.
