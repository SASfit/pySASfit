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
| Analysis | AUTORG (min. interval, q_max·Rg, minimum Rg); flat background of the high-q tail (see below); DATGNOM (Rg from AUTORG, which is run first if missing; fit up to s·Rg ≈ 8 or over all points of the range); GNOM (Dmax, systems 0–6, zero conditions, r points, α); CRYSOL / CRYSON (model curve, or fit to the selected curve, further options as text); DAMMIF (runs in parallel, mode, symmetry, unit; uses the GNOM p(r), else DATGNOM). *Rg + p(r) of all*: AUTORG and DATGNOM for every loaded curve |
| I(q) | data, points outside the range (grey ×), Guinier range and line, DATGNOM / GNOM fits and where the DATGNOM fit ends (red dotted line), the two background fits of the tail and the background to subtract (dashed), model curve; log–log, log–lin, Guinier (ln I vs q², low-q part), Kratky (q² I), linear |
| p(r) | DATGNOM and GNOM distributions with their errors, optionally normalised |
| Results | one row per curve: Rg ± error and I(0) (Guinier), its points and quality, flat background (or the one subtracted), Dmax, real-space Rg and total estimate (DATGNOM, GNOM), χ² of the model fit, number of bead models; *File → Export results table* writes it as CSV |
| Log | every command line, its folder and the last lines of the programs' output |
| 3D model | bead models and spherical-harmonics envelopes, several at once (tick to show), bead size, envelope opacity; *Load…* reads `.pdb`, `.cif` and `.flm` files (see below) |
| ATSAS | installation, licence, self test, and buttons with the original icons for the interactive programs: ATSAS launcher, PRIMUS, CHROMIXS, PEAK, SASPLOT, GNOMPLOT, SVDPLOT, SASDOC, MASSHA (legacy). With *open with the selected curve / p(r)*, PRIMUS and SASPLOT get the selected curve (its point range), GNOMPLOT its last `.out` file |

Programs run in the background; the window stays usable. Every run gets its own folder
(`<date>-<time>_<program>_<curve>` in **Preferences → folder for the ATSAS runs**, by default `atsas_runs` next to the
preferences file) with the input file and everything the program writes, so `.out`, `.fit` and bead-model files can be
opened later (*File → Open the run folder*).

Units: the ATSAS programs work in the reciprocal unit of the data. pySASfit curves are in 1/nm, so Rg, Dmax and r
are in nm. CRYSOL's *q max* is in the unit CRYSOL assumes for the model curve (1/Å unless changed with `--units`).

*Copy Python* (Ctrl+Shift+C) puts a script on the clipboard that repeats AUTORG, DATGNOM and GNOM of the selected
curve with the panel's settings.

## Flat background (incoherent scattering)

GNOM has no option to fit a constant, and DATGNOM stops at s·Rg ≈ 8 unless `--last` is given. Over a wider range a
flat background (incoherent scattering of hydrogen in SANS, an imperfect buffer subtraction) is fitted as if it were
particle scattering: p(r) gets a spike or oscillations at small r. The page **Flat background (DATPOROD)** estimates the
constant from the high-q tail, so that it can be subtracted before GNOM:

* **DATPOROD `--exponent --constant`** (ATSAS) fits I(s) = A s⁻ᴮ − C to the points of the tail q range. The manual
  does not say on which side C is subtracted; pyATsas evaluates both readings at the fitted points and takes the one
  that describes the data (the result line says which). `bg` is always the value to subtract from I.
* **Independent fit** (pySASfit, no ATSAS needed) of I = A q⁻ᴮ + bg with weighted least squares, with B free or
  fixed (4 = Porod for sharp interfaces, 2 for Gaussian chains). A difference of more than 3σ between the two
  estimates is flagged: then the tail does not follow one power law plus a constant (choose another q range, fix the
  exponent, or the data end before the Porod region).
* B and bg are strongly correlated when the tail is short or the power law is still far above the background; the
  errors show it. Use the widest tail that is clearly beyond the last form-factor features.
* **Subtract → new curve** adds the curve I − bg (errors unchanged) to the data list, named `… minus bg <value>`; the
  original stays. The value can be edited before subtracting. Subtracted curves are restored with the session.

Then run DATGNOM with *all points of the range* (or GNOM) on the new curve. *Copy Python* includes the background
fits and the subtraction.

## Shapes: bead models and spherical-harmonics envelopes

ATSAS has two kinds of ab-initio shape:

* **Bead models** (dummy atoms on a grid): DAMMIF and DAMMIN, MONSA (several phases, e.g. SAXS with SANS contrast
  variation), DAMMIX (mixtures), GASBOR (dummy residues), then DAMAVER / DAMCLUST over repeated runs. The panel runs
  DAMMIF; models of the others can be loaded in the 3D panel.
* **Envelopes** r(ω) = Σ F_lm Y_lm(ω) from spherical harmonics: **SASHA**. It exists only in the `Legacy` folder of the
  Windows installation (`sasha20.exe`; `sashaqw.exe` is the same with its own window), not in the main ATSAS 4
  programs. DAMMIN/DAMMIF, GASBOR and CRYSOL also expand amplitudes in spherical harmonics, but only to compute the
  scattering; their result is beads or atoms.

Viewing: ATSAS 4 has no 3D viewer of its own; the legacy **MASSHA** (`Legacy\massha.exe`) shows envelopes, bead
models and atomic models (from PDB files it shows only CA and P atoms, which is how bead models are written).
Otherwise PyMOL, ChimeraX or VMD open the model files (beads as atoms; show them as spheres of the bead radius).

**Bead models (DAMMIF page).** *model file* PDB (default here, MASSHA reads it) or mmCIF (ATSAS 4 default; converted
to PDB for MASSHA). After the run the models are listed; *Show in 3D* draws the selected one (Shift: all), *Open in
MASSHA* starts MASSHA with it. The I(q) plot shows the fit of model 1 (`.fir`). Bead models are read from PDB
(`ATOM`/`HETATM`) or mmCIF (`_atom_site`), coordinates in Å; the bead radius from the header ("DAM packing radius"
...), else half the median nearest-neighbour distance. Checked against the DAMMIN example of ATSAS 4.1.4 (608 beads,
radius 1.8 Å, Rg 15.40 Å and Dmax 49.1 Å as in its header).

**Envelopes (page "Envelope (SASHA, spherical harmonics)").** SASHA is an interactive dialog program from the ATSAS 2
era. *Run SASHA* writes the selected curve (point range, up to *q max*) as `mon.dat` and answers the documented
dialog through stdin: working mode User, data `mon.dat`, units 2 (4π sinθ/λ in 1/nm), no dimer data, output `sas`,
s_max, default start shape, symmetry (P1, P2, P222, Pmmm, P3), harmonic order L, start anisometry, molecular weight
(kDa) and Rg (Å) — *estimate* derives both from the Guinier Rg for a compact protein. The transcript is kept in
`sasha_dialog.log` in the run folder and checked: if the prompts did not appear in the documented order, a warning
says so. If SASHA writes no `sas.flm`, the run fails with its output. *SASHA interactively…* prepares the run folder
and opens SASHA there in a console to answer the dialog yourself; *Load .flm…* then reads the result.

The layout of the `.flm` file is not documented. pyATsas reconstructs the envelope from the coefficients in several
possible readings (indexed rows or sequences, complex or real harmonics) and accepts one only if its volume and Rg
agree within 3 % with the values SASHA writes into the same file. Only then is the envelope drawn in the 3D panel
(translucent, over a bead model if one is shown); otherwise the reason is shown and MASSHA is the way to look at it.
This check has been tested with synthetic files only: the first real SASHA file will show whether one of the readings
fits — if not, please send it.

The 3D panel needs `pyqtgraph` and `PyOpenGL` (`pip install -e ".[atsas]"`, the same packages as the pySASmorph 3D
panel). It also works when detached into its own window (the OpenGL context is rebuilt).

## When a program returns no result

Every call checks what the program left behind, not only its exit code (ATSAS programs sometimes exit with 0 and
write nothing). The call raises `AtsasError`, with the last lines of the program's output, when:

* the program cannot be started (not executable, blocked) or exceeds the time limit;
* it exits with an error (no Guinier region, missing licence, …);
* the expected output file is missing or empty (DATGNOM/GNOM `.out`, CRYSOL `.fit` when a fit was asked for,
  `.int`/`.abs` otherwise);
* the output has no usable numbers: no p(r) table or NaN in it, no valid Rg from AUTORG, no valid DATPOROD fit, NaN
  in a model curve;
* DAMMIF produced no bead model in any run. If only some runs fail, the models of the others are kept and the failed
  runs are listed in `errors` and in a warning.

Results that exist but are doubtful carry `warnings` (a list of texts): AUTORG quality below 0.5 or aggregation
flagged; GNOM total estimate below 0.5, p(r) negative by more than 5 % of its maximum, no real-space Rg or fitted
I(q) found; no χ² in the CRYSOL output; fewer DAMMIF models than runs.

In the panel a failure turns the status bar red, marks the curve with ✖ in the Results table (the tooltip gives
the message) and brings the Log panel forward; warnings mark it with ⚠ (orange). *Rg + p(r) of all* goes on with
the other curves when one fails, and keeps the AUTORG result when only DATGNOM fails.

## Scripts

```python
from pysasfit import atsas
from pysasfit.toolbox.merge import read_curves

c = read_curves("lysozyme.dat")[0]
g = atsas.autorg((c.q, c.I, c.dI))                 # GuinierResult: rg, rg_err, i0, first, last, quality
p = atsas.datgnom((c.q, c.I, c.dI), rg=g.rg)       # GnomResult: r, pr, pr_err, q, i_reg, dmax, rg_real, ...
p2 = atsas.gnom("lysozyme.dat", dmax=6.0, system=0)
b = atsas.datporod_exponent((c.q, c.I, c.dI), first=300, last=len(c.q))   # PorodResult: a, b, bg, dbg, chi2
f = atsas.fit_power_background(c.q[299:], c.I[299:], c.dI[299:], exponent=4) # the same without ATSAS
p3 = atsas.datgnom((c.q, c.I - b.bg, c.dI), rg=g.rg, last=len(c.q))       # all points, background subtracted
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
