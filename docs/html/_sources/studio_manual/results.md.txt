# Results and export

## The Result tab

After a fit the Result tab lists the optimiser, the number of evaluations, χ² and χ²/dof, every parameter with its
error (and, for iminuit MINOS, its asymmetric errors; for DREAM the 68 % interval), parameters sitting at a bound,
the state of the constraints, and the correlation matrix of the free parameters. Pairs with |r| ≥ 0.9 are listed
with the usual remedies. *Undo fit* restores the values from before the fit. The *Copy Python* button below the plot copies code that
reproduces the model.

## Saving a result

**Export result…** in the Result tab writes one of

| format | content |
|---|---|
| `.xlsx` | Summary, Parameters, Correlation, Covariance, Curve (with chart), Distributions (with chart), Plot, Report (needs `openpyxl`) |
| `.csv` | parameters, correlation, covariance, curves and `<name>_distributions.csv` |
| `.pdf`, `.png`, `.svg` | the summary figure (needs `matplotlib`) |

**Copy summary image** puts the same figure on the clipboard, ready to paste into a report or an e-mail.

```{figure} ../_static/studio/summary_figure.png
:alt: Summary figure
:width: 100%

The summary figure: data, fit and residuals, the size distributions N(x) and N(x)·x<sup>p</sup>, the parameters with
errors and the correlation matrix.
```

## Size distributions in the export

For every fitted size distribution the export contains N(x) and N(x)·x<sup>p</sup> on 400 logarithmic x points.
The exponent **p is read from the Size-distributions dock** at the moment of the export and written into the file
(sheet header, CSV header, figure title), so the file documents which weight was used.

## From a script

```{literalinclude} ../../examples/studio/04_export_results.py
:language: python
:start-after: "# --8<-- [start:export]"
:end-before: "# --8<-- [end:export]"
```
