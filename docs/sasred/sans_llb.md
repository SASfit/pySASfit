# SANS-LLB

pySASred reads SANS-LLB raw files (`sans-llb<year>n<run>.hdf`, NeXus `entry0`) next to SANS-I files; the reader
is chosen from the file ({func}`pysasfit.sasred.io.raw.read_raw`). Instrument file:
`instruments/sinq/sans_llb.yaml`.

## Detector banks

| bank | detector | pixels | distance |
|---|---|---|---|
| central | monoblock 128 × 128 | 5 × 5 mm | `central_detector/distance` |
| left | 16 vertical ³He tubes, 136 pixels along each | 13.4 mm across, 5 mm along | `left_detector/distance` |
| bottom | 16 horizontal ³He tubes, 136 pixels along each | 5 mm along, 13.4 mm across | `bottom_detector/distance` |

**Configuration.** The left and bottom panels sit on their own carriage, at the same distance from the sample by
design. Their distance is part of the instrument configuration: the run table shows *SD [m] (panels)* as
"4.00 (1.02)", the configurations of the run plan and the reduction table end in "panels 1.02 m", runs are grouped
only with runs at the same panel distance (1 %), and the masks are kept per panel distance (the panels can shadow
the edge of the central detector). Panels at different recorded distances are marked with "!" in the run table
and named in its comment. Beam centre and beam profile do not depend on the panels: they stay per (SD, λ, coll).
Reduction tables and masks saved before the panel distance was part of the configuration are still found for it
when only one panel distance was measured at that setting.

The run in the run table is the **central detector**: beam centre, masks, transmissions, grouping and the
reduction table work on it as for SANS-I. The wide-angle panels travel along with the run. In the Reduction
panel, **detector** chooses what *Reduce selected* and the reduction table reduce: the central
detector, one panel, or all three (results `5262`, `5262 left`, `5262 bottom`; table rows `… No. 1 left`).

A panel is reduced exactly like the central detector: counting error, dead time (τ = 3.5 µs, one rate per
bank), attenuator, monitor (`monitor2`, the preset monitor), transmission, background, and with water the
absolute scale per pixel (water measured on the same panel, so efficiency and solid angle cancel). Its mask is
the `sasred.llb.end_px` pixels at both tube ends and tubes without counts.

**Masks.** Every bank has its own mask per instrument configuration (coll, SD, λ, beam stop; the panel name is part
of the key, e.g. "70 mm | left panel"). The **detector** selector in the detector panel chooses the bank that is
shown and edited: central, left panel or bottom panel. All mask tools work on the panels as on the central
detector (draw, mask/unmask, undo, *Suggest*, save/load); a new panel mask starts with the tube ends and dead
tubes, *Suggest* adds hot pixels and keeps what was drawn by hand. The panels are shown in the orientation of the
central detector (x against posx, y with posy; on the left panel row 0 is at the top). *Save masks…* keeps the
masks of all banks and configurations in one file. The live I(q) follows the bank shown. The reduction uses the
edited mask of each bank (else the default panel mask) – also in the reduction table.

**Efficiency maps.** Every bank has its own map, made from the water measured on it (*efficiency map → Make
from the water of …* makes the maps of all three banks at once; the detector selector shows the map of its
bank). The central map has mean 1; a panel map is relative to the central detector: its own row × column
shape times the ratio of water / ΔΩ on the panel to that on the central detector (`relative_to_central` in its
metadata), so that the different efficiency of the 3He tubes is kept. Tube ends (masked) stay undefined.
*Save the shown map…* writes the maps of all banks into one file (`/entry` central, `/entry_left`,
`/entry_bottom`); chosen as the map of a geometry in the reduction table, every bank takes its own entry, and a
file with the map of one bank only is refused for another bank. *Archive* adds one entry per bank (name ending
in `_left`, `_bottom`); the history compares maps only with maps of the same bank.

**Semi-transparent beam stop.** Some SANS-LLB beam stops let a small, fixed fraction of the direct beam through.
The beam behind it can then measure the transmission of the sample during the scattering run itself. The mask
editor has a second layer for it, the **transmission region** (see [Masks](masks.md#transmission-region)): pixels
that are used for the transmitted intensity and not left out of anything. *Reduction → Transmission through the
semi-transparent beam stop…* compares it with the transmissions of the reduction table, so that you can find out
whether it is reliable before relying on it. Nothing in the reduction uses it yet. The files do not say whether
a beam stop is semi-transparent (`beam_stop` holds only its size, inclination and x, y), so a region is drawn only
for the configurations where it is.

The detector view **SANS-LLB: all detectors** shows the three banks together (counts per steradian), with the
current beam centre and panel offsets, every pixel at its scattering angle 2θ·(cos φ, sin φ) or, with
**axes → q_x, q_y**, at its scattering vector |q|·(cos φ, sin φ): rings are circles at their 2θ or q on all
three detectors, which is a direct check of the offsets. With **own scale for the panels** (default) the central
detector and the two panels have separate colour ranges, each with its colour bar; **panel colours** gives the
panels their own colour map. In this view the live I(q) shows the radial (or sector) average of
every bank as its own curve (central black, left red, bottom blue), each with its own mask and q range; in the
other views only the bank chosen under **detector**.

## Geometry and its corrections

NICOS writes for every bank the pixel positions `posx`, `posy` [mm] in one frame perpendicular to the beam
(`<bank>_data/posx|posy`); the central detector has its zero at pixel (63.5, 63.5). GRASP
(`SINQ_SANS_LLB.ini`) uses the same nominal geometry. The beam position in that frame follows from the beam
centre on the central detector, so the panels need no beam centre of their own. A panel pixel is at

    x = -(posx + dx - xb),   y = posy + dy - yb,   z = L + dz,      ΔΩ = A z / |r|³

(x and y as in the central detector's geometry; A the pixel area; panel plane perpendicular to the beam).

The offsets are set under *Settings → Preferences → pySASred → SANS-LLB detector geometry*:

| setting | meaning |
|---|---|
| `sasred.llb.left.dx_mm`, `dy_mm`, `dz_mm` | left panel: across the tubes (posx), along the tubes (posy), along the beam |
| `sasred.llb.bottom.dx_mm`, `dy_mm`, `dz_mm` | bottom panel: along the tubes (posx), across (posy), along the beam |
| `sasred.llb.central_dz_mm` | central detector: offset of the detection layer from the recorded distance |
| `sasred.llb.end_px` | pixels masked at both tube ends (default 10) |

Every reduction notes the offsets used. Determine them with a standard that puts rings on the panels (silver
behenate in the 4 m setting: 2.15, 3.23 and 4.30 nm⁻¹ on both panels).

The factory defaults are the silver behenate calibration of 2026-11-10 (runs 9546 at 8 m and 9568 at 17.5 m,
beam centres from the direct-beam runs 9535 and 9557): left dx −10.0, dy −15.4 mm; bottom dx −5.8, dy −8.3 mm
(±2 mm, they follow the beam centre 1:1); dz +45 ± 10 mm for both panels; central dz +3 mm (only (L + dz) times
the wavelength scale is measured, 1.0013 × nominal). The rings on the central detector are elliptical
(x/y = 1.032 ± 0.002 in all runs): its effective pixel pitch is about 5.08 mm in x and 4.92 mm in y, which is
not corrected yet.

### What the commissioning data (Maren, 2026-08, λ = 0.5 nm) show

* Beam centre on the central detector (half-maximum edges of the box-shaped direct beam): (62.67, 63.20) at
  4 m (attenuator 6), (62.88, 61.90) at 17 m. Attenuator 3 at 4 m is distorted by dead time.
* Along the tubes the solid-angle profile of D2O (4 m against 17 m as flat field) has its apex at posy ≈ +84 ±
  13 mm on the left panel (beam at ≈ −1.5 mm) and at posx ≈ −35 ± 28 mm on the bottom panel; the radial
  positions and the distances cannot be told from these data.
* Attenuators from the panels (sample at attenuator 0 against the same sample attenuated): 6 → 0.0049–0.0060
  (file: 0.008), 3 → 0.048–0.058 (file: 0.050); without a Cd run, indications only.
* The beam stop at y = −7 mm let part of the direct beam pass at 4 m (and below it at 17 m): mask those pixels.
* The runs named "H2O" are D2O; the real H2O runs are 5278–5280.
* Run 5279 holds no data (measuring time 0, 0 counts, monitor 0): it is skipped and marked in the run list.
* The absolute scale from the direct beam + efficiency map: the panel maps from water are relative to the central
  detector (see *Efficiency maps* above), but the angular dependence of the water (inelastic scattering,
  self-absorption at 2θ up to ≈ 30°) is in them; water as the absolute standard of the panels avoids it.
