# pySASmorph: anisotropic data (texture)

pySASmorph also builds 3D structures from a **2D pattern of an oriented sample**, when the sample is
rotationally symmetric about an axis that lies in the detector plane (fibre or uniaxial texture). Typical
examples:

- stretched elastomers and drawn polymers;
- extruded, flow-aligned or field-aligned samples;
- compressed foams.

One 2D pattern then fixes the full 3D intensity, because the detector plane contains every angle to the axis.

*File → Anisotropic 2D data (texture)…* in pySASmorph opens the window. In Python the module is
{mod}`pysasfit.morph.anisotropic`. The theory is in {doc}`theory`, section 12.

## Data

Load one 2D pattern with the background subtracted, or fit the flat background B in the window:
- **NXcanSAS 2D**, e.g. pySASred's *File → Save 2D NXcanSAS*. The file has I, Idev, Qx, Qy and Mask; masked
  pixels are dropped.
- **`.npz`** with `qx, qy, I` and optionally `sigma, mask`.
- **Text file** with the columns q_x q_y I [dI].

Lengths are in the inverse unit of q.

## Two models

| | A – affine stretch | B – uniaxial (Legendre) |
|---|---|---|
| structure | an isotropic structure stretched by λ along the axis | any structure with uniaxial symmetry |
| fitted from the 2D data | λ and the axis angle, by collapsing log I onto one curve of q_eff = (λ²q∥² + q⊥²)^½ | the Legendre components I_l(q), l = 0, 2, …, l_max |
| then | the collapsed curve goes through the isotropic pipeline; γ(r) = γ_iso((r∥²/λ² + r⊥²)^½) | γ(r) from the components by the plane-wave expansion |
| use when | the iso-intensity lines are ellipses | streaks, meridional peaks, any shape |

Both then use the same steps:
1. The level-cut inversion $g = F^{-1}(\varphi(1-\varphi)\gamma)$, direction by direction.
2. FFT synthesis of the Gaussian field from $g(\mathbf r)$ on the periodic grid.
3. The cut, the voxel scattering, and the statistics.

## The symmetry axis

The 2D pattern is mirror-symmetric about two perpendicular lines. **Which of them is the 3D symmetry axis is a
choice:** the 3D structure is rotated about it.
- **estimate** sets the axis perpendicular to the direction of maximum intensity (the fibre convention). Fibres
  and stretched networks scatter an equatorial streak perpendicular to the axis.
- **+90°** takes the other line, for layers along the axis (meridional peaks).
- Model A refines the angle (*refine the axis*).

In the result the axis is the box **x** axis and the beam the box **z** axis.

## Settings

- **l_max** (B): the highest even Legendre order. Strong anisotropy needs more orders, noisy data fewer; 6–8 is
  typical.
- **Coverage:** q bins without full azimuthal coverage are left out and listed in the warnings. These are the
  bins at the beam stop and in the detector corners.
- **volume fraction φ, box size L, voxels n, seed:** as in the isotropic case. L defaults to 5·2π/q* of the
  Kratky maximum of I_0; in model A it is enlarged by λ.
- **q range, low-q extrapolation, Porod range, background:** as in the isotropic case. The Porod range is
  automatic by default (or the last N points, q ≥ q_min, the last 15 %). *fit flat (incoherent) background B* is
  on by default; untick it to use the fixed B (empty = 0). They apply to I_0 (model B) or to the collapsed curve
  (model A).

## Results

| tab | shows |
|---|---|
| 2D data / model | the pattern with the axis (dashed); the data folded into the quadrant (q∥, q⊥); the level-cut model on the same scale |
| Components I_l(q) | I_0(q) and I_l/I_0 for l ≥ 2: data (points), level-cut model (lines), voxel structure (×) |
| Correlations γ, g | γ(r∥, r⊥) and g(r∥, r⊥) of the Gaussian field |
| Structure | slices: a plane containing the axis (x, y) and one perpendicular to it (y, z) |
| 3D | the interface (as in the isotropic case) |
| Results | the report |

The report lists:
- the model;
- the axis and λ;
- Q_0, K_0, the contrast and S/V (Porod);
- clipped γ values;
- χ² of the level-cut model against all pixels;
- S/V of the voxels;
- the **share of interface faces normal to x / y / z**, each 1/3 when isotropic;
- the **mean chord lengths** of both phases along and across the axis;
- clusters and percolation per direction.

For a structure stretched by λ, the chords along the axis are λ times those across it.

*File → Export components and maps…* writes four files: the data and model components (text), γ and g on the
(r, μ) table (`.npz`), and the report. *Export voxel matrix…* writes VTK, TIFF or npy.

```python
from pysasfit.morph.anisotropic import AnisoParameters, load_2d, reconstruct_aniso

d = load_2d("stretched_2d.h5")
res = reconstruct_aniso(d, AnisoParameters(mode="B", axis_deg=None, porosity=0.3, n=128, lmax=8))
print(res.report())           # axis None: estimated (fibre convention), noted in the warnings
```

## Limits

- **Uniaxial only.** A general texture without a symmetry axis needs several sample orientations, as in SAXS
  tensor tomography (to come).
- **Not for locally varying orientations.** A sample made of differently oriented domains, like a polycrystal,
  is not one stationary Gaussian field and is outside the model.
- **Axis in the detector plane.** An axis tilted out of the plane leaves part of the μ range unmeasured.
- **Realisability.** As in the isotropic case, γ values a level-cut field cannot reach are clipped and counted.
  Negative spectral power of the anisotropic g is reported.

## Validation (tests/morph/test_anisotropic.py)

The tests use exact synthetic data: a level-cut structure with a Gaussian g, stretched by λ = 2 along 30°, with
1 % noise. They check:

- **Model A:** it recovers λ to 0.2 % and the axis to 0.2°.
- **γ(r∥, r⊥):** models A and B both reproduce it within 0.025 at six points along, across and between the
  axes.
- **Fit quality:** the contrast is right within 3 % and χ² stays below 20 (it is about 4–12).
- **Isotropic data:** they give no higher components, I_l/I_0 < 2 %.
- **Reconstruction:** the voxel porosity is exact; the chords along the axis are λ times those across; fewer
  interface faces are normal to the axis. Voxel and Porod S/V agree, and less than 2 % negative power is
  discarded.
- **Constants:** the tail integrals ∫j_l/x² dx have their closed forms.
- **Files and window:** loading NXcanSAS, npz and text files; the window, opened from pySASmorph's File menu.
