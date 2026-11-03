# 2D data

The tab **2D data** of pyTOOLbox shows reduced 2D scattering images next to the radially averaged curves
(module {mod}`pysasfit.toolbox.image2d`), with masking, averages, sums and rocking curves (module
{mod}`pysasfit.toolbox.analysis2d`). The resulting curves go to the tab **1D data** ({doc}`data1d`).

## Sources

| source | how | geometry (q, ψ) |
|---|---|---|
| pySASred in memory | *From pySASred*; in *Merge geometries* select curves and press *Show 2D*; pySASred *Reduction → Send results to pyTOOLbox merge* also hands over the images | complete |
| pySASred 2D NXcanSAS | *Add files…*; pySASred writes `<name>_2d.h5` with *File → Save 2D NXcanSAS…* and next to every 1D file with *Save all* | from SASdetector (beam centre, pixel size, distance, wavelength) |
| other NXcanSAS 2D with Qx, Qy | *Add files…* | the q of the file |
| HDF5 viewer | *→ 2D data* in the viewer hands the 2D data set shown over (SANS-I raw files with their geometry, otherwise pixels with the pixel size entered there; beam centre from the centre of gravity) | as the source |
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

## Beam centre

* **click** in the image (pixels or q view, draw mode *navigate*) sets the beam centre there; or enter x, y [pixel] under
  *Geometry of the image shown* and press *Apply geometry*;
* **Centre of gravity** refines the centre iteratively from the intensity around the current one.

q and ψ of every pixel follow the new centre, so the averages below use it at once.

## Mask

Choose *mask* or *unmask* and the shape to draw — **rectangle**, **ellipse** or **polygon** — and draw it in any
view (pixels, qx – qy or polar: a rectangle in the polar view is a sector of a ring); the pixels inside are
masked or unmasked at once. *navigate* switches drawing off (zoom, pan, click for the beam centre).
*mask below* / *mask above* masks pixels by value.
*Invert*, *Clear*, *Reset* (back to the mask of the file), *Undo*, *Suggest* (pySASred's automatic suggestion:
beam stop, edges, dead and hot pixels) and *Save* / *Load* (HDF5, compatible with pySASred masks) act on the
image shown.

## Averages → 1D data

| kind | parameters | result |
|---|---|---|
| radial I(q) | bins, log / linear spacing, q min / max (auto) | I(q) |
| sector I(q) | ψ0 ± Δψ, *both sides* adds the opposite sector | I(q) |
| azimuthal I(ψ) | ring q1 … q2, number of ψ bins | I(ψ) |
| box projection | angle, half length, width (1/nm), through the beam centre | I along the box axis |
| A(q) + B(q) sin²(ψ − ψ0) | bins as radial, ψ0 | two curves A(q), B(q) (magnetic SANS: nuclear and magnetic part) |

Every bin is the mean of its valid pixels, σ = √(Σσᵢ²)/n. A + B sin² is a weighted linear least-squares fit
of the ψ dependence in every q bin, with the uncertainties of A and B from the covariance. *show region*
draws the sector, ring or box in the image. *Compute → 1D data* puts the curves in the tab **1D data**.

## Sum / rocking

* **Sum** or **Mean of the selected images** (same detector; a pixel masked in any image stays masked)
  gives a new image in the list;
* **Rocking curve**: the mean intensity in a region (whole detector, ring q1 – q2, or ring in the sector) of
  every selected image versus a header item (e.g. `hdf:sample/omega`) → 1D data.
