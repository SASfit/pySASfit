# pySEBpolymers: structures of linked sub-units

```{image} ../pySASfit-logo-bundle/logos/pySEBpolymers/svg/pySEBpolymers-light.svg
:alt: pySEBpolymers
:height: 70px
:class: only-light
```

```{image} ../pySASfit-logo-bundle/logos/pySEBpolymers/svg/pySEBpolymers-dark.svg
:alt: pySEBpolymers
:height: 70px
:class: only-dark
```

`pysasfit.seb` computes the form factor of any tree-like structure built from **sub-units**
(Gaussian chains and loops, rods, spheres, shells, disks, rings, cylinders, spheroids) that are
joined at **reference points** (chain ends, centres, random points on a surface or contour).
Micelles with grafted chains, star and dendrimer polymers, bottle brushes, rod–coil block
copolymers or vesicles with loops are all a few lines of Python.

It is a pure-Python (numpy/scipy) implementation of the formalism of Svaneborg & Pedersen [1, 2] that the Scattering Equation Builder (SEB) is built on [3]. It is written from the published formalism and the textbook sub-unit
expressions, not from SEB's code, and needs no symbolic algebra; names of sub-units, reference
points and parameters are those of SEB/pyseb, so structures carry over unchanged.

## The pySEBpolymers panel

`pysasfit-seb`, or the pySEBpolymers tab of `pysasfit-gui`. Panels (View menu; each can be moved, stacked or
detached):

| panel | content |
|---|---|
| Designer | the structure drawn as a graph (see below); above it the template and its options |
| Code | the Python code of the structure, written by the designer; typed code is built with **Build** (Ctrl+B) and drawn in the designer |
| Parameters | `beta_<tag>` and the sizes of the structure, plus `scale` and `background`: value, fit, lower and upper bound, error after a fit; model (normalised F or unnormalised S), q grid; **Fit**, **Calculate**, **Copy Python** |
| I(q) | model, data with error bars and normalised residuals (I − model)/σ, χ²_red |
| 3D sketch | one random realisation of the structure (random-walk chains, rods, bodies in random orientations, random surface points), one colour per tag; **New realisation** |
| Components | sub-units, kinds, tags and the junctions |
| Log | builds and fits |

### The designer

The designer draws a structure instead of writing code. Every change that leaves a complete design rebuilds the
structure at once (the Code panel shows the code it writes; the parameter table follows).

* **Palette** (left): the 11 sub-units, the patterns and the components of the design. Drag an entry onto the
  canvas, or double-click it.
* **Canvas**: a node shows its reference points as ports along its edge — round for a point (`end1`, `center`,
  `top` …), a diamond for a distributed point (`surface`, `contour`, `hull`, `rim` …). **Drag from a port to a
  port** to join the two points. A link that would close a loop is refused (the formalism needs a tree). Wheel:
  zoom; middle button: pan; left drag on the empty canvas: select; F: fit (Shift+F: all, however small).
* **Inspector** (right): name, type and tag of a sub-unit (sub-units with the same tag share their parameters);
  pattern, element, points, counts of a pattern; for a link at a distributed point its **label** — the same label
  on the same node is the same point, different labels are independent random points (e.g. chains grafted at
  independent points of a surface, or several chains at one point).
* **Patterns** stand for many sub-units:
  * *Array*: n copies of an element, each attached by its in-point to the point the array's `anchor` is linked to —
    at independent random points of a distributed host point, or all at the same point;
  * *Chain*: n copies joined out-point → in-point, ports `start` and `end`;
  * *Generations*: a dendrimer — f elements at the `root`, f − 1 new elements at the out-point of every element of
    the previous generation, g generations, optionally with own parameters per generation (tags `<tag>0`, `<tag>1` …).
* **Components** (Editing: …, *+ Component*, *Component from selection*): designs of their own with an in-port and
  an out-port (right-click a port of a sub-unit), used as the element of a pattern — a bottle-brush segment
  (backbone block with a side chain), an A–B diblock, a rod with loops on its contour … Components may contain
  patterns and other components.
* Del deletes, Ctrl+C / Ctrl+V / Ctrl+D copy, paste, duplicate, Ctrl+Z / Ctrl+Y undo and redo, *Layout* arranges the
  nodes as a tree. File → *Open design / Save design* reads and writes `*.sebdesign` (JSON).

Code typed in the Code panel is drawn one node per sub-unit (up to 400); its loops are not turned back into patterns.

Templates start a design: micelle (sphere, shell, spheroid, cylinder or disk core with an Array of Gaussian chains,
rods or loops, at independent random surface points or all at one point), star polymer (Array on a Point), linear or
alternating multiblock chain (Chain of a block or of an A–B component), dendrimer (Generations, one set of
parameters per generation), bottle brush (Chain of a segment component), rod–coil block copolymer, vesicle with
loops, and *Custom*. The model is

$$ I(q) = \mathrm{scale}\cdot F(q) + \mathrm{background} \quad\text{or}\quad
   I(q) = \mathrm{scale}\cdot S(q) + \mathrm{background}, $$

with lengths in the inverse unit of q (Å for q in 1/Å). Data files have the columns q, I, σ (File → *Load data*,
Ctrl+L). The fit is a bounded least-squares fit (Trust Region Reflective, weights 1/σ²) of the ticked parameters;
errors from the Jacobian, scaled by χ²_red. File → *Open / Save structure code* reads and writes the code,
*Export model curve* the curve, *Copy Python code* (Ctrl+Shift+C) a script that repeats the calculation.
Preferences (Settings → Preferences → pySEBpolymers): automatic recalculation, points of the orientational
average, largest number of model evaluations of a fit. The last session (template, design, code, parameters, data,
panel layout) is restored at the next start.

Designs from a script:

```python
from pysasfit.seb.graph import Design

d = Design("micelle")
core = d.add_unit("SolidSphere", "core")
arr = d.add_pattern("Array", "GaussianPolymer", name="chains", tag="chain", n=50, in_ref="end1")
d.link(core.id, "surface", arr.id, "anchor")
s = d.to_structure()            # or d.to_code() for the Python code
```

## Example

```python
import numpy as np
from pysasfit.seb import Structure, builders

s = Structure("micelle")
s.add("SolidSphere", "core")
for i in range(20):                       # 20 chains at independent random surface points
    s.link("GaussianPolymer", f"c{i}.end1", f"core.surface#a{i}", tag="chain")

q = np.logspace(-3, 0, 500)
p = {"beta_core": 3.0, "R_core": 25.0, "beta_chain": 0.4, "Rg_chain": 10.0}
F = s.form_factor(q, p)                   # normalised, F(0) = 1
S = s.form_factor(q, p, normalized=False) # S(0) = (sum of beta)^2
print(s.parameters(), s.radius_of_gyration2(p))

d = builders.dendrimer(5, functionality=3) # tags g0 ... g4: one Rg/beta per generation
```

* `add(kind, name, tag=None)` adds a sub-unit; `link(kind, "new.ref", "old.ref#label", tag=...)`
  adds one and joins it; `link("a.ref", "b.ref")` joins two existing points; `include(other, prefix)`
  copies a whole structure.
* A distributed reference point (`surface`, `contour`, `hull`, `rim` ...) takes a label:
  the same label is the same point, different labels are independent random points.
* Parameters are `beta_<tag>` (excess scattering length; default 1) and the size parameters
  `Rg_<tag>`, `R_<tag>`, `L_<tag>`, `Ri_<tag>`/`Ro_<tag>`, `a_<tag>`/`c_<tag>`; sub-units with the
  same tag share them.
* Builders: `linear_chain`, `star`, `micelle`, `dendrimer`, `bottlebrush`.

## Formalism

For sub-units $I$ with excess scattering lengths $\beta_I$, form factors $F_I$, form-factor
amplitudes $A_{I,\alpha}$ relative to reference point $\alpha$ and phase factors $\Psi_{K,\alpha\omega}$
between two reference points of $K$,

$$
S(q) = \sum_I \beta_I^2 F_I + \sum_{I\neq J} \beta_I\beta_J\, A_{I,\alpha}
\Big[\prod_{K\in\mathrm{path}(I,J)} \Psi_{K,\alpha_K\omega_K}\Big] A_{J,\omega},
\qquad F = S/(\textstyle\sum_I\beta_I)^2 .
$$

The double sum is evaluated by a recursion over the tree of sub-units and junctions, in a time
linear in the number of sub-units (a 1533-unit dendrimer: 0.3 s for 2000 q values).
`form_factor_pairs` is the direct pair sum, kept for tests. Loops are rejected.

## Sub-units

| sub-unit | parameters | specific points | distributed points |
|---|---|---|---|
| Point | – | point | – |
| GaussianPolymer | Rg | end1, middle, end2 | contour |
| GaussianLoop | Rg | – | contour |
| ThinRod | L | end1, middle, end2 | contour |
| SolidSphere | R | center | surface |
| ThinSphericalShell | R | center | surface |
| SolidSphericalShell | Ri, Ro | center | surface, surfacei, surfaceo |
| ThinCircle | R | center | contour |
| ThinDisk | R | center | surface, rim |
| SolidCylinder | R, L | center, top, bottom | hull, ends, surface |
| Spheroid | a, c | center, north, south, pole (= north) | surface |

Closed forms are used where they exist; the cylinder, the spheroid and the disk rim are averaged
over orientations by Gauss–Legendre quadrature, with $F=\langle\Phi^2\rangle$,
$A_r=\langle\Phi g_r^*\rangle$, $\Psi_{rs}=\langle g_r g_s^*\rangle$. `Point` is a massless junction
(F = A = 0) as in SEB. The spheroid surface point is uniform per area by default;
`Spheroid(surface_measure="parametric")` reproduces pyseb (uniform in cos θ).

## Validation against pyseb 0.4.0

* Every F, A and Ψ that pyseb evaluates agrees to ≤ 1e-8 (most to 1e-12–1e-15); twelve composite
  structures (micelles, shells, vesicle with loops, rod stars, cylinder and disk with chains,
  spheroid, dendrimer, bottle brush) to ≤ 8e-8, at 10–100× the speed.
* The remaining ~1e-8 differences are pyseb's cancellation in the Debye function near q = 1e-3
  (checked against mpmath).
* pyseb 0.4.0 returns the surface *amplitude* for the Spheroid phase factor Ψ(center, surface);
  a Monte Carlo average of sin(qr)/(qr) confirms the value here.

Tests: `tests/seb` (`test_seb_pyseb_reference.py` runs only where pyseb is installed, i.e.
Python ≤ 3.12). Full comparison tables: `python tools/seb/compare_pyseb_subunits.py` and
`compare_pyseb_structures.py` from the repository root.

## References

1. C. Svaneborg, J. S. Pedersen, J. Chem. Phys. 136, 104105 (2012).
2. C. Svaneborg, J. S. Pedersen, J. Chem. Phys. 136, 154907 (2012).
3. T. Jarrett, C. Svaneborg, J. Appl. Cryst. 57, 587 (2024), doi:10.1107/S1600576724001729;
   code github.com/Tobionecenobi/SEB (GPL-3.0), Python bindings pyseb (BSD-3-Clause).

```{toctree}
:hidden:

API <api>
```
