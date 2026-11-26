# SLD profile

```{image} ../pySASfit-logo-bundle/logos/pySLDprofile/svg/pySLDprofile-light.svg
:alt: pySLDprofile
:height: 56px
:class: only-light
```

```{image} ../pySASfit-logo-bundle/logos/pySLDprofile/svg/pySLDprofile-dark.svg
:alt: pySLDprofile
:height: 56px
:class: only-dark
```

`pysasfit.sldprofile` estimates the radial scattering-length-density profile ρ(r) of spheres, cylinders and
lamellae from small-angle scattering data. Two routes are available:

- **direct** Bayesian estimation of ρ(r) in one step (Hansen, *J. Appl. Cryst.* **49** (2016) 856), with a
  smoothness or maximum-entropy constraint and the noise level α and r_max from their posterior;
- **two-step** (Glatter): indirect Fourier transformation I(q) → p(r), then deconvolution p(r) → ρ(r).

The profile is represented by shells (step profile) or by a quadratic or cubic B-spline, on uniform, power-law
or adaptive grids. Projects and the last session are saved in versioned files; every panel can be detached into
its own window.

The distance distribution p(r) alone (sphere, cylinder cross-section or lamella thickness, same normalisation) can
also be obtained with every solver of the inversion tool, kernels `sinc_4pi`, `cylinder_pddf` and `lamella_pddf`
({doc}`../inversion/theory_forward_model`).

The GUI is `pysasfit-sldprofile`, or the *pySLDprofile* tab of `pysasfit-gui`.

```{toctree}
:maxdepth: 2

usage
theory
```

The API reference is in the {doc}`API section <../api/index>`.
