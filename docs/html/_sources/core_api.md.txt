# Shared infrastructure: `pysasfit.core`

Used by several tools: data loading, instrumental resolution, background models, the preferences and project files.

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

## Projects and last session

```{eval-rst}
.. automodule:: pysasfit.core.project
   :members: save_project, load_project, upgrade, session_path, forget_session, control_state, restore_controls, FORMAT_VERSION, ProjectError

.. automodule:: pysasfit.core.project_gui
   :members: ProjectMixin
```

## Detachable panels

```{eval-rst}
.. automodule:: pysasfit.core.docking
   :members: DockMixin, PanelWindow, plot_panel, small_toolbar
```

## Detachable tabs

Used by the launcher and by pyTOOLbox (see {doc}`launcher`).

```{eval-rst}
.. automodule:: pysasfit.core.tabs
   :members: DetachableTabWidget, DragOutTabBar
```

## Colour scales of detector images

```{eval-rst}
.. automodule:: pysasfit.core.colorscale
   :members:
```

## Wrapping control bars

```{eval-rst}
.. automodule:: pysasfit.core.flowlayout
   :members:
```
