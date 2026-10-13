# Shared infrastructure: `pysasfit.core`

Used by several tools: data loading, instrumental resolution, background models and the preferences.

## Data loading

```{eval-rst}
.. automodule:: pysasfit.core.io
   :members:
```

## Instrumental resolution

The pinhole operator is the same as in the OZ tools (Rician kernel by default, Gaussian as an option).

```{eval-rst}
.. automodule:: pysasfit.core.resolution
   :members:
```

## Background models

```{eval-rst}
.. automodule:: pysasfit.core.background
   :members:
```

## Preferences

The settings themselves are listed, with explanations, in {doc}`preferences`.

```{eval-rst}
.. automodule:: pysasfit.core.settings
   :members: get, values, set_value, reset, save, load, settings_path, tools, sections
```
