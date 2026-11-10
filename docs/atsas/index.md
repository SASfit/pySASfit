# pyATsas: the ATSAS programs in pySASfit

pyATsas runs the programs of **ATSAS** (EMBL Hamburg / BIOSAXS GmbH) on the curves of pySASfit and shows their
results next to the other tools: Guinier analysis (AUTORG), distance distributions (DATGNOM, GNOM), model curves of
atomic structures (CRYSOL, CRYSON) and ab-initio bead models (DAMMIF). It also starts the interactive ATSAS programs
(PRIMUS, CHROMIXS, PEAK, …) with the curve or p(r) selected in pySASfit.

ATSAS is **not** part of pySASfit. It has to be installed separately, with its own licence
(<https://www.embl-hamburg.de/biosaxs/software.html>). pyATsas only calls its programs and reads the files they
write. The icon of the pyATsas tab is the original ATSAS icon, read at run time from that installation
(`share/icons/hicolor/<n>×<n>/apps/atsas.png`); pySASfit does not ship a copy. Without ATSAS the tab shows a plain
placeholder, and the run buttons are disabled.

* launcher tab **pyATsas**, or stand-alone: `pysasfit-atsas [data files]`
* scripts: `from pysasfit import atsas` (no Qt needed)
* check the installation: `python -m pysasfit.atsas --selftest`

## Finding ATSAS

pyATsas works on Windows (native or MSYS2 Python), Linux and macOS. The installation is looked for in this order:

1. **Settings → Preferences → pyATsas → ATSAS installation folder** (or `atsas_dir=` in scripts), the root folder
   or its `bin/`;
2. the environment variables `PYATSAS_DIR` and `ATSAS`; on Windows also the `ATSAS` value the installer writes to
   the registry, which a Python started from MSYS2 or an IDE may not have in its environment;
3. `autorg` on the `PATH` (Linux packages);
4. the newest `ATSAS-x.y.z` folder (compared as version numbers, 4.10 after 4.9) in the default locations:
   `C:\Program Files` and `%LOCALAPPDATA%\Programs` (Windows), the home folder (default of ATSAS 4 on Linux and
   macOS), `/opt`, `/usr/local`, `/Applications`. A Python started from the macOS desktop does not read
   `~/.zprofile`, so this step matters there even when the installer set up the shell.

The **ATSAS** panel shows the folder found and whether the licence file `atsas.lic` is in it. *Search again* looks
again after an installation or a change of the preferences; *Self test* runs AUTORG, DATGNOM and GNOM on a synthetic
sphere (R = 5 nm: Rg = 3.873 nm, Dmax = 10 nm) and reports the deviations in the Log panel.

Every program gets `ATSAS` (the installation root; ATSAS 4 finds its licence through it) and its `bin/` first on the
`PATH`. Input files are copied into the run folder under short ASCII names, because some ATSAS programs fail on
spaces, umlauts or long paths (OneDrive folders, for example).

## Panels

| panel | content |
|---|---|
| Data | curves: text (q I dI [dq]), pySASred ASCII, NXcanSAS (every entry), as in pyTOOLbox; q in 1/nm. *points*: the range given to every program (1-based, as in ATSAS) |
| Analysis | AUTORG (min. interval, q_max·Rg, minimum Rg); DATGNOM (Rg from AUTORG, which is run first if missing); GNOM (Dmax, systems 0–6, zero conditions, r points, α); CRYSOL / CRYSON (model curve, or fit to the selected curve, further options as text); DAMMIF (runs in parallel, mode, symmetry, unit; uses the GNOM p(r), else DATGNOM). *Rg + p(r) of all*: AUTORG and DATGNOM for every loaded curve |
| I(q) | data, points outside the range (grey ×), Guinier range and line, DATGNOM / GNOM fits, model curve; log–log, log–lin, Guinier (ln I vs q², low-q part), Kratky (q² I), linear |
| p(r) | DATGNOM and GNOM distributions with their errors, optionally normalised |
| Results | one row per curve: Rg ± error and I(0) (Guinier), its points and quality, Dmax, real-space Rg and total estimate (DATGNOM, GNOM), χ² of the model fit, number of bead models; *File → Export results table* writes it as CSV |
| Log | every command line, its folder and the last lines of the programs' output |
| ATSAS | installation, licence, self test, and buttons with the original icons for the interactive programs: ATSAS launcher, PRIMUS, CHROMIXS, PEAK, SASPLOT, GNOMPLOT, SVDPLOT, SASDOC. With *open with the selected curve / p(r)*, PRIMUS and SASPLOT get the selected curve (its point range), GNOMPLOT its last `.out` file |

Programs run in the background; the window stays usable. Every run gets its own folder
(`<date>-<time>_<program>_<curve>` in **Preferences → folder for the ATSAS runs**, by default `atsas_runs` next to the
preferences file) with the input file and everything the program writes, so `.out`, `.fit` and bead-model files can be
opened later (*File → Open the run folder*).

Units: the ATSAS programs work in the reciprocal unit of the data. pySASfit curves are in 1/nm, so Rg, Dmax and r
are in nm. CRYSOL's *q max* is in the unit CRYSOL assumes for the model curve (1/Å unless changed with `--units`).

*Copy Python* (Ctrl+Shift+C) puts a script on the clipboard that repeats AUTORG, DATGNOM and GNOM of the selected
curve with the panel's settings.

## Scripts

```python
from pysasfit import atsas
from pysasfit.toolbox.merge import read_curves

c = read_curves("lysozyme.dat")[0]
g = atsas.autorg((c.q, c.I, c.dI))                 # GuinierResult: rg, rg_err, i0, first, last, quality
p = atsas.datgnom((c.q, c.I, c.dI), rg=g.rg)       # GnomResult: r, pr, pr_err, q, i_reg, dmax, rg_real, ...
p2 = atsas.gnom("lysozyme.dat", dmax=6.0, system=0)
m = atsas.crysol("6lyz.pdb", "lysozyme.dat")       # CrysolResult: q, i_model, chi2 (cryson: program="cryson")
d = atsas.dammif(p.out_file, nruns=10)             # DammifResult: models, fits, workdir
atsas.launch("primusqt", ["lysozyme.dat"])         # interactive programs
```

Data are a file name or a `(q, I, dI)` tuple. Each result keeps the command, its folder and the output of the run
(`.run`). Errors of ATSAS (no Guinier region, missing licence, …) raise `AtsasError` with the program's message.

## Limits

* GNOM output is parsed by its header lines; the format changed between ATSAS versions. Tested against the formats
  of ATSAS 3 and 4 as documented; please report a `.out` file that is not read.
* GNOM system 2 and slit smearing need GNOM's interactive input and are not offered.
* DAMAVER / DAMCLUST of the bead models are not yet called from the panel (the models are in the run folder).

## API

Functions and result classes: {doc}`api`.
