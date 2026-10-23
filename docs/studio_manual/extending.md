# Extending studio

## Python plugins

A Python function appears in the menus like a SASfit plugin and overrides a C plugin of the same name. Create the
files from the template:

```bash
pysasfit-studio new-plugin my_gauss          # -> ~/.pysasfit/studio_plugins/sasstudio_my_gauss.py and a test
```

Edit the function and its `PARAMETERS`; use it as `studio.function("ff_my_gauss")` or pick it in the explorer. The
plugin folder is `~/.pysasfit/studio_plugins` or a folder in `PYSASFIT_STUDIO_PLUGINS`. At run time:
`studio.register_python_function("ff_name", intensity, amplitude, volume, parameters=[...])`.

A `*.def` file in the plugin folder defines your own menu groups (see `templates/groups_example.def`); name the
group in the plugin's `GROUP`.

## SasView

```bash
pysasfit-studio sasview-adapter ff_sphere --dir .
```

writes a SasView model file that calls the SASfit function (SasView must run in an environment with pysasfit and the
libraries).

## Where things are found

`pysasfit-studio status` shows the folders. The libraries come from, in this order: the folder chosen in the GUI,
`PYSASFIT_SASFIT_LIB`, `SASFIT_LOCATION`, and `src/pysasfit/studio/sasfit_lib` (filled by
`pysasfit-studio install-binaries`).

## Tests

```bash
python -m pytest tests/studio -q
```

Most tests need no SASfit library; the tests that need it are skipped without it. See the
[reference](../studio.md#tests) for the list of test files.
