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
| FloryPolymer | Rg, nu | end1, middle, end2 | contour |
| FloryLoop | Rg, nu | – | contour |
| ExcludedVolumePolymer | Rg, nu | end1, middle, end2 | contour |
| WormLikeChain | L, b | end1, middle, end2 | contour |
| WormLikeChainEV | L, b | end1, middle, end2 | contour |
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

## Chains with a Flory exponent

`FloryPolymer` and `FloryLoop` are Hammouda's generalised Gaussian chain (B. Hammouda, Macromol. Theory Simul. 2
(1993) 1; Benoit 1957): Gaussian statistics with ⟨r²ᵢⱼ⟩ = b²|i − j|^{2ν}, so ν = 1/2 is the ideal chain
(`GaussianPolymer`, `GaussianLoop` exactly), ν ≈ 0.588 a chain in good solvent and ν = 1/3 a collapsed one. With
U = q²⟨R²⟩/6 = q²Rg²(2ν + 1)(2ν + 2)/6 and a = 1/(2ν) (γ the lower incomplete gamma function):

| quantity | expression |
|---|---|
| form factor | F = [U^{−a} γ(a, U) − U^{−2a} γ(2a, U)] / ν |
| amplitude about an end | A_end = ∫₀¹ e^{−U t^{2ν}} dt = a U^{−a} γ(a, U) |
| amplitude about the middle | A_end with U → U 2^{−2ν} |
| phase end – end, end – middle | e^{−U}, e^{−U 2^{−2ν}} |
| phase contour – end / middle / contour | A_end, A_middle, F |
| ring | F = ∫₀¹ e^{−W [s(1−s)]^{2ν}} ds, W = q²Rg² / (3 B(2ν+1, 2ν+1)) (Gauss–Legendre) |

Two approximations: the excluded volume acts only **within** a sub-unit (chains joined in a structure do not see
each other, nor the core — four joined FloryPolymer blocks are not one long swollen chain), and distances keep a
Gaussian distribution (the end-to-end phase factor of a real self-avoiding walk differs). `tools/seb/check_flory_mc.py`
compares with Monte Carlo: chains with exactly the model's statistics (fractional Brownian motion, 128 bonds)
agree within the discreteness of the walk (F within 0.003, end amplitude 0.005 for qRg < 6); self-avoiding walks on
the cubic lattice (pivot algorithm, 128 bonds) differ from the ν = 0.588 model by up to 0.05 in F around qRg ≈ 2–6
(lattice chains of this length are still in the crossover) and up to 0.03 in the end-to-end phase factor.
ν is fitted between 0.3 and 1 by default.

## Further chain models

All chains below are described by their two-point function G(q, s) (the average of sin(qr)/(qr) over two
contour points a distance s apart, contour length L): F = 2∫₀¹(1 − t) G(tL) dt, A_end = ∫₀¹ G(tL) dt,
A_middle = 2∫₀^{1/2} G(tL) dt, Ψ_end–end = G(L), Ψ_end–middle = G(L/2); contour points use A and F as for the
Gaussian chain. The integrals use a grid adapted to the oscillations of G (fine panels up to 13 b, geometric
beyond).

**WormLikeChain** (L, b; b = Kuhn length = 2 × persistence length) — Kholodenko's worm (A. L. Kholodenko,
Macromolecules 26 (1993) 4179): G = sinh(E z)/(E sinh z) for qb < 3 and sin(F z)/(F sinh z) above, z = 3s/b,
E = √(1 − (qb/3)²), F = √((qb/3)² − 1). It goes from the rod (L ≪ b) to the Gaussian chain (L ≫ b, Rg² = Lb/6).
Note: its ⟨R²(s)⟩ = (b²/3)(z coth z − 1) is Kholodenko's, not the exact worm-like chain value
b s − (b²/2)(1 − e^{−2s/b}): it is up to 20 % larger around s ≈ 0.7 b and within 1 % beyond s ≈ 20 b, so Rg of short
worms comes out up to ~9 % larger (L ≈ b; 2 % at L = 10 b) than for the exact worm-like chain.

**WormLikeChainEV** (L, b) — worm-like chain with excluded volume: the form factor of Pedersen & Schurtenberger
(Macromolecules 29 (1996) 7602, method 3 with excluded volume) with the corrections of Chen, Butler & Magid
(J. Chem. Phys. 124 (2006) 144903), a NumPy port of `wrc_cyl.c` of sasmodels (public domain; `pysasfit.seb.ps_worm`,
equal to sasmodels' flexible_cylinder with radius → 0 to 1e-9). Only F(q; L) is known, not G. Since
F = 2∫(1 − t)G(tL)dt, differentiating in L gives the other quantities exactly:

| quantity | from F(L) |
|---|---|
| A_end | F + (L/2) F′ |
| A_middle | F(L/2) + (L/4) F′(L/2) |
| Ψ_end–end | F + 2L F′ + (L²/2) F″ |
| Ψ_end–middle | the same at L/2 |

(central differences, step 0.2 % of L; the branches of the piecewise PS formula are chosen at the reference
length, so L ± h do not jump between branches). Checked on Kholodenko's worm, where the relations reproduce the
direct G to 1e-5. The PS formula is not valid for L < b (F > 1); between L = b and 2.5 b it is blended smoothly
(C² in L) into Kholodenko's worm, which is used alone below b (excluded volume is negligible for such stiff
pieces). The PS formula itself has small kinks where its branches meet (relative steps of up to ~4e-4 in F),
visible in very smooth data only.

**ExcludedVolumePolymer** (Rg, nu) — self-avoiding chain with the des Cloizeaux distance distribution
(des Cloizeaux 1980; Förster & Burger, Macromolecules 31 (1998) 879): P(r) ∝ r^{2+θ} exp(−(Dr)^δ),
δ = 1/(1 − ν), with D fixed by ⟨r²⟩ ∝ s^{2ν}. Unlike FloryPolymer, the distance distribution is not Gaussian
(correlation hole at short r, compressed tail). The exponent θ depends on where the two points are:
θ₀ = 0.275 (end – end), θ₁ = 0.46 (end – interior), θ₂ = 0.71 (interior – interior), scaled by
(ν − ½)/(0.588 − ½), so ν = ½ is the Gaussian chain exactly; F and A_middle use θ₂, A_end and Ψ_end–middle θ₁,
Ψ_end–end θ₀. Valid for 0.34 ≤ ν ≤ 0.95. Against self-avoiding walks (pivot algorithm, 128 bonds,
`tools/seb/check_flory_mc.py`): F within 0.011, A within 0.017, Ψ within 0.038 — four times closer in F than
FloryPolymer with the same ν (0.046).

### Templates for single chains

*Freely jointed chain*: n ThinRods of length b joined end to end (a structure, not a sub-unit — the
segments are rigid, the joints free). *Worm-like chain*: one WormLikeChain or WormLikeChainEV. The chain
choices in the micelle, star, linear, bottle-brush and dendrimer templates include all chain types.

## Polydisperse and thick sub-units, crowded copies

Three options per element in the designer (inspector), or as arguments of `Structure.add` / `link`:

**Polydisperse** (`poly=True`, parameter `sigma_<tag>`): a Schulz–Zimm distribution of the mass m of each
such sub-unit (mean 1, relative width σ). Sizes scale with m (`MASS_SCALING`: chain length ∝ m,
Rg ∝ m^ν, sphere R ∝ m^{1/3}, …), the excess scattering length with m. Each sub-unit's F, A and Ψ are averaged
independently, ⟨m²F⟩, ⟨mA⟩, ⟨Ψ⟩ — exact for independent polydispersity of the sub-units, because every term of
the SEB sum contains each sub-unit at most once. The structure is normalised by (Σβ)² + Σβ²σ², so F(0) = 1.
Quadrature: generalised Gauss–Laguerre (σ ≥ 0.1) or Gauss–Hermite (σ < 0.1), 32 nodes.

**Thick** (`thick=True`, parameter `Rc_<tag>`): a cylindrical cross-section of radius Rc for chains, rods and
rings, in the usual decoupling approximation: F × [2J₁(qRc)/(qRc)]², A × 2J₁(qRc)/(qRc), Ψ unchanged.

**Copies see each other** (Array pattern, parameter `vex_<array>`, 0 ≤ φ < 1): the copies of an Array
(e.g. the chains of a micelle) exclude each other's volume. The interference between two different copies I, J
becomes

β_I β_J [(1 + k) A_I Ψ…Ψ A_J − k F_c(q)],  k = φ/(n − 1),

with F_c the normalised scattering of one copy, n the number of copies. The added −k F_c is a correlation hole:
at q = 0 the term is unchanged (k cancels), so F(0) = 1 and the forward scattering stays that of the mass; at
higher q it lowers the cross terms by up to φ of the self-scattering, as excluded volume does. With φ < 1 the
intensity stays positive. This is a form in the spirit of Pedersen & Gerstenberg (Macromolecules 29 (1996)
1363) for block copolymer micelles, applied to any Array; it is not the RPA form used there, and the tree
algorithm and the pair sum agree to rounding (tests). Arrays nested in arrays work (innermost groups first).
In code: `s.interact('vex_name')` and `copy=[(group, index, 'vex_name'), ...]` on each sub-unit of a copy; the
designer writes this.

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
