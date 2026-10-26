# 2D data

The tab **2D data** of pyTOOLbox shows reduced 2D scattering images next to the radially averaged curves
(module {mod}`pysasfit.toolbox.image2d`). The analysis of the images (below) will be added to this page.

## Sources

| source | how | geometry (q, ψ) |
|---|---|---|
| pySASred in memory | *From pySASred*; in *Merge geometries* select curves and press *Show 2D*; pySASred *Reduction → Send results to pyTOOLbox merge* also hands over the images | complete |
| pySASred 2D NXcanSAS | *Add files…*; pySASred writes `<name>_2d.h5` with *File → Save 2D NXcanSAS…* and next to every 1D file with *Save all* | from SASdetector (beam centre, pixel size, distance, wavelength) |
| other NXcanSAS 2D with Qx, Qy | *Add files…* | the q of the file |
| BerSANS 2D (`Type=SANSDAni`, e.g. `D0007613.008`) | *Add files…* | distance and wavelength from the file; **the beam centre is not in the file**: enter it under *Geometry of the image shown* and press *Apply geometry* (pixel 7.5 mm of SANS-I preset). Until then pixels only |

*Show 2D* in *Merge geometries* finds the image of a curve in memory (pySASred) or as `<name>_2d.h5` next
to its file.

The pySASred 2D NXcanSAS file contains I (ny, nx), Idev (statistical), Qx, Qy (`I_axes` = "Qy,Qx"), Mask
(1 = not valid) and the pySASred reason bits (`mask_bits`), the detector geometry and, as in 1D, the
parameters, notes and the header items of the raw run (`SASprocess/metadata`).

## Display

* **qx – qy** (equal axes, 1/nm), **pixels** (beam centre marked) or **polar (ψ, q)** (nearest-pixel
  resampling as in pySASred's polar view);
* colour scale linear, log(1+I), symlog or asinh, colour map, masked pixels grey, hidden or shown; the colour
  range follows the valid pixels;
* q rings (list of q in 1/nm) drawn as circles (qx – qy) or lines (polar);
* the cursor read-out gives the pixel, q, ψ and I ± dI, and whether the pixel is masked.

ψ = 0 is the horizontal direction (+x), increasing counter-clockwise, as in pySASred.

## Planned operations

| operation | content |
|---|---|
| sector average | I(q) in a sector ψ₀ ± Δψ (and the opposite one), several sectors at once |
| azimuthal plot | ring q₁ … q₂ → I(ψ), as GRASP |
| A(q) + B(q) sin²ψ | fit of every q ring to A + B sin²(ψ − ψ₀) (magnetic SANS: nuclear + magnetic part), as GRASP, with the uncertainties of A and B |
| box projection | a box (length, width) through the beam centre, tilted by an angle, projected on its long axis |
| sums | sum (or monitor-weighted mean) of several reduced images, e.g. rocking curves: intensity in a region versus a header item (angle) |

All of them will work on the image with its uncertainties and mask, and give curves that can be plotted,
saved and handed to *Merge geometries*.
