# studio manual

The **studio** tools give you the model functions of SASfit (form factors, structure factors, size distributions,
peaks) in Python: you can plot them, combine them into models, fit them to data and save the result. You can work
interactively in the **model explorer** (`pysasfit-explorer`) or from a script. This manual is a guided tour; the
complete description of every option is in the {doc}`reference <../studio>` and the signatures are in the
{doc}`API pages <../studio_api>`.

- {doc}`getting_started`: install, check the libraries, a first plot and a first fit.
- {doc}`explorer`: windows and docks, parameter rows, sliders, data, saving a session.
- {doc}`models`: mixtures, size distributions, structure factors, background, integration.
- {doc}`fitting`: bounds, constraints, priors, optimisers, global search, speed-ups.
- {doc}`results`: Result tab, correlations, Excel, CSV, summary figure, size distributions.
- {doc}`scripting`: Python examples for functions, models, the fitting engine and the export.
- {doc}`extending`: own functions, SasView adapter, where things are found.

```{toctree}
:maxdepth: 2

getting_started
explorer
models
fitting
results
scripting
extending
Reference <../studio>
```

## About the pictures and examples

The screenshots in this manual are made by `tools/studio_screenshots.py` from the test data set of SASfit
(`examples/studio/data/SASfitTestData`), so they can be renewed after a change of the GUI:

```bash
python tools/studio_screenshots.py
```

`python tools/release.py` does this as one step of a release (tests, screenshots, HTML documentation, package), so
the published pictures always show the current GUI.

The code examples are the scripts in `examples/studio/` (`01_evaluate_functions.py` … `05_open_explorer.py`). They are
included into the pages from those files, so what you read is what runs; `tests/studio/test_examples.py` checks
that they are valid and, when the SASfit libraries are installed, runs them.
