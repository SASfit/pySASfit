# Building models

## Contributions

A model is a **mixture**: I(q) = Σ<sub>i</sub> scale<sub>i</sub> · I<sub>i</sub>(q) + background (+ A·q<sup>−α</sup>).
Each contribution I<sub>i</sub> is built from up to three functions.

```{literalinclude} ../../examples/studio/02_mixture_model.py
:language: python
:start-after: "# --8<-- [start:model]"
:end-before: "# --8<-- [end:model]"
```

| part | function | role |
|---|---|---|
| form factor | `ff_…` | the shape; one of its parameters can be *distributed* |
| size distribution | `sd_…` | weight N(x) of that parameter; its amplitude N is the scale |
| structure factor | `sq_…` | interaction between the particles |

Without a size distribution the contribution is monodisperse and multiplied by its scale. With a size
distribution the scale is fixed to 1 because N is the scale.

## Structure-factor approaches

With a size distribution and a structure factor the *approach* decides how they are combined (SASfit manual,
chapter 4.1):

- **monodisperse:** S(q) · ∫N P dx;
- **decoupling:** ∫N P dx + (∫N F dx)²/∫N dx · (S − 1);
- **local monodisperse (lma):** ∫N P S(q; R<sub>V</sub>(x)) dx.

```{literalinclude} ../../examples/studio/02_mixture_model.py
:language: python
:start-after: "# --8<-- [start:variants]"
:end-before: "# --8<-- [end:variants]"
```

## Background

`background` is a constant. The rows `bg.A` and `bg.alpha` add a power law A·q<sup>−α</sup> (α starts at 4, the Porod
value); with A = 0 it is off. In the plot the background is the dashed grey line.

## Integration of the size distribution

The x integral uses Gauss–Legendre nodes on the support of N(x), or SASfit's integrators. *Settings →
Integration…* sets the number of nodes and tolerances. **Check nodes** in the Fit dock compares the result for
increasing node numbers and proposes the smallest sufficient one; **Profile model** shows where the time of one
evaluation goes. See the [reference](../studio.md#faster-fits-dfo-ls-variable-projection-cache-coarse-to-fine-fixed-grid).

## Own functions

Python functions can be registered as plugins and then appear in the menus like SASfit's own; see {doc}`extending`.
