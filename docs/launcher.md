# The launcher: all tools in one window

`pysasfit-gui` (or `python run_gui.py` in a checkout) hosts every tool in one window. The tools are chosen in a
**side bar**: a column of icons at the left edge of the window (a row of tabs no longer fits the screen).

* **Hovering** an icon shows the tool's name, what it does and its shortcut (**Alt+1** … **Alt+9**).
* A **click** brings the tool to the front (and loads it the first time). The tool in front has a bar in its
  colour at the left of its icon.
* **»** at the top of the column shows the names next to the icons (**«** hides them again; also
  *View → Tool names next to the icons*). The choice is kept for the next start.
* The tool in front has a **pale yellow background** (light theme; contrast of at least 3 : 1 for every name in its logo colours, the Windows selection blue would be about 1 : 1). Another colour: *Settings → Preferences → pySASfit window*.
* A filled **dot** at an icon marks a tool that is loaded, a **ring** one that is detached into its own window.
* **Double-click** an icon, **drag** it out of the column, or use the **right mouse button** to detach the tool
  (see below).

| icon | tool | stand-alone |
|---|---|---|
| **py**SAS**red** | SANS data reduction ({doc}`sasred/index`) | `pysasfit-sasred` |
| **py**SAS**studio** | SASfit model explorer and fitting ({doc}`studio_manual/index`) | `pysasfit-explorer` |
| **py**SLD**profile** | radial SLD profiles ({doc}`sldprofile/index`) | `pysasfit-sldprofile` |
| **py**INV**ersion** | size distributions and p(r) ({doc}`inversion/index`) | `pysasfit-inversion` |
| **py**OZ**solver** | Ornstein–Zernike structure factors ({doc}`oz/index`) | `pysasfit-oz` |
| **py**SCA**tter** | scattering patterns of particles and ordered structures, fits to 1D / 2D data ({doc}`scatter/index`) | `pysasfit-scatter` |
| **py**SEB**polymers** | form factors of structures of linked sub-units (micelles, stars, dendrimers, bottle brushes …), fits to 1D data ({doc}`seb/index`) | `pysasfit-seb` |
| **py**AT**sas** | the ATSAS programs (installed separately): Guinier, p(r), CRYSOL, DAMMIF; starts PRIMUS, CHROMIXS, PEAK … ({doc}`atsas/index`) | `pysasfit-atsas` |
| **py**DEN**sity** | electron-density maps with DENSS (installed separately): single and averaged maps, isosurfaces, slices ({doc}`density/index`) | `pysasfit-density` |
| **py**DENSS**multi** | 3D maps from several contrasts (SANS contrast variation, SANS + SAXS) ({doc}`densitymulti/index`) | `pysasfit-densitymulti` |
| **py**TOOL**box** | small helpful tools: HDF5 viewer and metadata lister ({doc}`toolbox/index`) | `pysasfit-toolbox` |

Each tool has its menu icon (one colour per tool, also used for the tool's window and task-bar entry) and
its name as in the logos: grey *py*, the bold core in the tool's colour (red pySASred, blue pySASstudio, magenta
pySLDprofile, green pyINVersion, violet pyOZsolver, sienna pySCAtter, olive pySEBpolymers, teal pyATsas, blue pyDENsity, orange pyTOOLbox) and the word ending in the grey of *py*
(lighter colours on a dark system theme). A tool that cannot be loaded (for
example a missing optional dependency) shows the error message in its place instead of stopping the launcher.

A data file given on the command line is opened in pySLDprofile, a `*.pysasfit` project file in its tool. The
other tools reopen their last session.

## Fast start: nothing is calculated before you ask

The launcher window opens at once: it only creates the side bar. A tool is loaded the first time it is needed (its icon
is clicked, it is detached, a project of it is opened, **Alt+n**); its page says so for the moment it takes. Its last
session is restored then, **without recalculating or rescanning anything** — the inputs, data files and settings
come back, and the tool calculates when its button is pressed:

| tool | at the start | calculates / reads on |
|---|---|---|
| pySASred | the last raw-data folder is named, not read | **Rescan** (the reduction tables of the last session are restored then) |
| pySASstudio | model, parameters, data sets and fit settings of the last session; no curve, the button shows *Calculate \** | **Calculate** (or a change with *auto* on) |
| pySCAtter | parameters and data, no model curve | **Calculate 2D** (or a parameter change with *I(q) follows*) |
| pySEBpolymers | structure and data, no curve, no 3D sketch | **Calculate** |
| pySASmorph | curves and parameters | **Correlations** |
| pyATsas | the ATSAS folder of the preferences or of the last search | **Search again** (a run searches if none was found yet) |
| pyDENsity | the DENSS Python of the last search, no Python is started | **Search again** (a run searches if none was found yet) |

The installations found by *Search again* are kept in `installations.json` next to the preferences file
({mod}`pysasfit.core.installcache`); the Preferences dialog shows them without searching either.

## Menus

The tools' own menu bars are hidden in the launcher; the launcher's menus act on the tool in front:

* **File**: *Open project… / Save project…* of the tool in front, followed by the tool's own file entries the
  launcher does not provide itself (for example *Save project as…* of pySASstudio, *Open raw-data folder…* and
  *Save NXcanSAS…* of pySASred, *Open HDF5 files…* of pyTOOLbox), and *Copy Python code* (**Ctrl+Shift+C**): a
  script that repeats the calculation of the tool in front ([Copy Python code](python_code.md)). Their keyboard
  shortcuts work as in the stand-alone window.
* **View**: the tool's own View entries (*Sliders* of pySASstudio), the panels of the tool in front (show / hide,
  panel border, *Arrange panels*, *Dock all panels*, *Reset layout*), the detach / attach entries described below, and under **Tools** one
  entry per tool with its icon: it brings the tool to the front, also when it is detached (**Alt+1** … **Alt+6**
  in the order of the side bar).
* **Settings**: *Preferences…* (opens on the page of the tool in front), *Forget last session (all tools)*, and
  the tool's own settings entries (for example *SASfit installation…* and *Integration…* of pySASstudio).

## Panels: active frame, borders, arrangement

The panel last clicked (or holding the keyboard focus) is framed in magenta, as the active window in Origin; the
other panels have a raised border (*View → Panel border*: raised is the default; sunken, groove, line, none).
*View → Arrange panels* puts all docked panels **side by side** (tile horizontally), **one above the other**
(tile vertically), in a **grid**, or **all as tabs** (cascade); the detached panel windows can be cascaded or
tiled over the screen. *View → Reset layout* returns to the start arrangement. The selected tab of every tab bar
has a light grey background, a blue edge and bold text.

## Window size

The window can be made small: the panels of every tool get scroll bars when there is less room than their
controls need (before, the widest and the tallest tool fixed the smallest size of the whole window, about
1230 × 660 pixels). For developers: panels added with `DockMixin._add_dock` scroll by themselves; the pages of
other tab widgets are passed to {func}`pysasfit.core.scrolling.scroll_inside`. Do not call `setMinimumWidth` /
`setMinimumSize` on a panel; give it its start size with `resizeDocks` or `QSplitter.setSizes`.

## Detaching tools

Any tool can be turned into an independent window and back:

* **detach**: drag the tool's icon out of the side bar, double-click it, right-click it (*Detach … into a
  window*), or *View → Detach '…' into a window*;
* **attach**: close that window, click **Attach** at the right end of its menu bar, or use the right-click menu /
  *View → Attach '…'* / *Attach all tools*.

A detached tool is an ordinary window with its own task-bar entry; it is not kept above the launcher, and it
shows its own menu bar again, as when started on its own. Attaching puts the tool back
into the launcher window.

Which tools are detached, where their windows are, the tool in front, whether the side bar shows the names, and the launcher's own size and position
are part of the last session; *Settings → Forget last session (all tools)* clears them too. Closing the launcher
closes the detached windows as well.

The same detachable tabs are used inside tools: the two tabs of pyTOOLbox (viewer and metadata lister) can be
separate windows. For developers: replace `QtWidgets.QTabWidget()` by
{class}`pysasfit.core.tabs.DetachableTabWidget`; nothing else changes for the pages
(`set_wordmark` gives a tab the coloured logo label).

Within each tool, the panels (plots, controls, text output) can be detached as well, by the button in their
title bar ({class}`pysasfit.core.docking.DockMixin`).
