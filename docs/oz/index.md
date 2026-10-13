# Ornstein–Zernike

`pysasfit.oz` solves the multicomponent Ornstein–Zernike equations with a choice of closures and solvers, with
polydisperse particle form factors, analytic references (MSA, RMSA, Robertus, Gazzillo), fitting, a Qt
interface (`pysasfit-oz`, or the Ornstein–Zernike tab of `pysasfit-gui`) and a programmatic API. It comes from
pyOZgui; the interface was ported from Tk to Qt (see {doc}`README`).

The GUI's *calculate* button calls {func}`pysasfit.oz.ozLib.solve`, so anything the interface can compute is
reachable from a script through the same function — a figure made in the GUI can be reproduced in a batch job
without reimplementing anything.

```{toctree}
:maxdepth: 2
:caption: Contents

manual
gui
generic_polydisperse_tab
Changes since pyOZgui <README>
```

## Where to start

- **{doc}`manual`** — the quick reference, worked examples, and the traps that have actually caught people
  here: sign conventions, the real-space range, branch selection, grid sizing, verification. Most of what will
  save you time is on that page rather than in the signatures.
- **{doc}`gui`** — the tabs, session files, how to read the parameter panel, and which solver and fitter to
  choose (including Picard / Mann damping and the iteration limits).
- **{doc}`generic_polydisperse_tab`** — the polydisperse tab in detail: size classes, the six approximation
  schemes, and what each costs.
- **{doc}`api`** (in the API section) — signatures and docstrings, generated from the source so they cannot drift from the code.
- **{doc}`README`** — what changed in the port from pyOZgui: behaviour changes and bug fixes.

Numerical settings not on the panels (iteration limits, tolerances, deflation, verification) are in
**Settings → Preferences** ({doc}`../preferences`).

## A note on the two halves

The API reference is generated; the manual is written. Where they disagree, the API reference is right and the
manual needs correcting.

## Report and manuscript

The LaTeX report (ozLib/ozGUI documentation with the multicomponent supplement) and the manuscript are in
`docs/oz/latex/` and are built separately (`build_report.sh`, `manuscript/build.sh`). They were written for
pyOZgui and have not yet been brought in line with the changes made since the port; see
`docs/oz/latex/REVIEW.md` for the list of places that no longer match the code.

## Citing

Results obtained with the two-Yukawa reference module require citation of Liu, Chen & Chen,
*J. Chem. Phys.* **122**, 044507 (2005). That is the condition under which its authors granted
redistribution, and it is not optional.
