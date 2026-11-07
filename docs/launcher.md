# The launcher: all tools in one window

`pysasfit-gui` (or `python run_gui.py` in a checkout) opens every tool as a tab of one window:

| tab | tool | stand-alone |
|---|---|---|
| **py**SAS**red** | SANS data reduction ({doc}`sasred/index`) | `pysasfit-sasred` |
| **py**SAS**studio** | SASfit model explorer and fitting ({doc}`studio_manual/index`) | `pysasfit-explorer` |
| **py**SLD**profile** | radial SLD profiles ({doc}`sldprofile/index`) | `pysasfit-sldprofile` |
| **py**INV**ersion** | size distributions and p(r) ({doc}`inversion/index`) | `pysasfit-inversion` |
| **py**OZ**solver** | Ornstein–Zernike structure factors ({doc}`oz/index`) | `pysasfit-oz` |
| **py**SCA**tter** | scattering patterns of particles and ordered structures, fits to 1D / 2D data ({doc}`scatter/index`) | `pysasfit-scatter` |
| **py**TOOL**box** | small helpful tools: HDF5 viewer and metadata lister ({doc}`toolbox/index`) | `pysasfit-toolbox` |

Each tab shows the tool's menu icon (one colour per tool, also used for the tool's window and task-bar entry) and
its name as in the logos: grey *py*, the bold core in the tool's colour (red pySASred, blue pySASstudio, magenta
pySLDprofile, green pyINVersion, violet pyOZsolver, sienna pySCAtter, orange pyTOOLbox) and the word ending in the grey of *py*
(lighter colours on a dark system theme). Hovering a tab tells what the tool does. A tool that cannot be loaded (for
example a missing optional dependency) gets a tab with the error message instead of stopping the launcher.

A data file given on the command line is opened in pySLDprofile, a `*.pysasfit` project file in its tool. The
other tools reopen their last session.

## Menus

The tools' own menu bars are hidden in the launcher; the launcher's menus act on the tool in front:

* **File**: *Open project… / Save project…* of the tool in front, followed by the tool's own file entries the
  launcher does not provide itself (for example *Save project as…* of pySASstudio, *Open raw-data folder…* and
  *Save NXcanSAS…* of pySASred, *Open HDF5 files…* of pyTOOLbox). Their keyboard shortcuts work as in the
  stand-alone window.
* **View**: the tool's own View entries (*Sliders* of pySASstudio), the panels of the tool in front (show / hide,
  panel border, *Arrange panels*, *Dock all panels*, *Reset layout*), the detach / attach entries described below, and under **Tools** one
  entry per tool with its icon: it brings the tool to the front, also when it is detached (**Alt+1** … **Alt+6**
  in the order of the tabs).
* **Settings**: *Preferences…* (opens on the page of the tool in front), *Forget last session (all tools)*, and
  the tool's own settings entries (for example *SASfit installation…* and *Integration…* of pySASstudio).

## Panels: active frame, borders, arrangement

The panel last clicked (or holding the keyboard focus) is framed in magenta, as the active window in Origin; the
other panels have a raised border (*View → Panel border*: raised is the default; sunken, groove, line, none).
*View → Arrange panels* puts all docked panels **side by side** (tile horizontally), **one above the other**
(tile vertically), in a **grid**, or **all as tabs** (cascade); the detached panel windows can be cascaded or
tiled over the screen. *View → Reset layout* returns to the start arrangement. The selected tab of every tab bar
has a light grey background, a blue edge and bold text.

## Detaching tabs

Any tab can be turned into an independent window and back:

* **detach**: drag the tab out of the tab bar, double-click it, right-click it (*Detach … into a window*), or
  *View → Detach '…' into a window*;
* **attach**: close that window, click **Attach** at the right end of its menu bar, or use the right-click menu /
  *View → Attach '…'* / *Attach all tools*.

A detached tool is an ordinary window with its own task-bar entry; it is not kept above the launcher, and it
shows its own menu bar again, as when started on its own. Attaching puts the tab back at its place (also after
the tabs were reordered by dragging them along the tab bar).

Which tools are detached, where their windows are, the tab in front and the launcher's own size and position
are part of the last session; *Settings → Forget last session (all tools)* clears them too. Closing the launcher
closes the detached windows as well.

The same detachable tabs are used inside tools: the two tabs of pyTOOLbox (viewer and metadata lister) can be
separate windows. For developers: replace `QtWidgets.QTabWidget()` by
{class}`pysasfit.core.tabs.DetachableTabWidget`; nothing else changes for the pages
(`set_wordmark` gives a tab the coloured logo label).

Within each tool, the panels (plots, controls, text output) can be detached as well, by the button in their
title bar ({class}`pysasfit.core.docking.DockMixin`).
