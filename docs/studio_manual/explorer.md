# The model explorer

```bash
pysasfit-explorer                                              # a sphere
pysasfit-explorer ff_sphere --sd sd_LogNorm --sq sq_hardsphere
pysasfit-explorer ff_sphere --add ff_cylinder,sd_LogNorm       # second contribution of a mixture
pysasfit-explorer --project my.sasstudio                       # a saved session
```

From a script: `explore(...)`, see {doc}`scripting`.

## Windows

The plot is the central widget. Around it are three docks that you can move, hide and detach:

| dock | content | open |
|---|---|---|
| **Model** | the contributions as tabs, background, parameters | always |
| **Fit** | data, fit selection, constraints, priors, optimiser, result | *View → Fit* |
| **Size distributions** | N(x)·x<sup>p</sup> of the distributions of the model | *View → Size distributions* |

To put a dock into a window of its own, use *View → detach '…' as a separate window*. A detached window does not
stay on top of the main window, and closing the main window closes it.

## The model dock

```{figure} ../_static/studio/model_dock.png
:alt: The model dock with two contributions
:width: 60%

Two contributions, each with a form factor, a size distribution and an optional structure factor.
```

Each tab is one *contribution*: a form factor, optionally a size distribution and a structure factor, chosen from
cascading menus that follow SASfit's groups. *Add*, *Copy* and *Remove* above the tabs change the number of
contributions; the model is the sum of the contributions (each with its scale) plus the background.
{doc}`models` describes the options.

## Parameter rows

```{figure} ../_static/studio/parameter_rows.png
:alt: Parameter rows
:width: 60%

A parameter row: name, value, slider (optional), *distr.* radio button and fit box with its bounds.
```

- **Value:** type it and press Enter (a decimal comma is accepted). The slider range is re-centred on a typed value.
- **Sliders:** *View → Sliders* adds a slider to each row. The slider covers 0 to twice the value, limited by the
  bounds.
- **Fit box:** ticks the parameter for fitting. Two fields below it take the lower and the upper bound; empty means no
  bound.
- **Distr. radio button:** the form-factor parameter that is integrated over the size distribution.
- Parameters named `dummy` are not used by the function; their fields are disabled.

## Data and the plot

The *Fit* dock loads a data file (`q, I, ΔI [, Δq]`, with a factor for unit conversion) and takes an optional fit
range. The plot shows data, model, the dotted contributions (*show contributions*), the background, and the
normalised residuals Δ/σ below. *show* switches between I = P·S, P and S.

## Size distributions

```{figure} ../_static/studio/size_distribution.png
:alt: Size-distribution dock
:width: 60%

The size distribution N(x)·x<sup>p</sup>. The exponent p (0, 3 or 6 with the buttons) weights the distribution:
p = 3 gives the volume weight.
```

The exponent p set here is also the one used when you save a result (see {doc}`results`).

## Settings

- *Settings → Integration…*: how the size-distribution integral is computed (SASfit's integrators or scipy
  Gauss–Legendre, number of nodes, range, tolerances).
- *Settings → SASfit installation…*: the folder with the libraries; remembered between sessions.

## Saving a session

*File → Save project* (Ctrl+S) writes everything to a `.sasstudio` file: contributions, values, bounds, data, plot and
fit settings, constraints, priors and the last result. *Open project…* (Ctrl+O) or `--project FILE` restores it.
`examples/studio/data/project.sasstudio` is a ready-made example.
