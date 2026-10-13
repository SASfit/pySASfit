# Report and manuscript: places that no longer match the code

The LaTeX report (`ozGUI_ozLib_report.tex` with its two parts) and the manuscript (`manuscript/`) were written for
pyOZgui. They were moved here unchanged on 2026-10-12. Since then the code has moved on: package `pysasfit.oz`,
Qt GUI, sign convention A (positive = attractive), Picard / Mann iteration with damping, verification by
independent algorithm families, default transform type 4 for one component, and so on.

This file lists every statement found to be out of date, checked against the code, with line numbers (as of the
move), the current behaviour and a suggested wording. **Nothing in the .tex files has been changed**: what to
change in the manuscript is the authors' decision. Measured numbers (benchmarks, figures) were not re-run; they
are flagged only where the text describes behaviour the code no longer has.

Classes: **must fix** = factually wrong now · **should update** = outdated name, path or terminology ·
**optional**.

## Open points beyond the text

- **Scripts that produce figures and numbers** are cited but were not moved: `tools/benchmark.py`,
  `fig_realdata_fit.py`, `fig1_discretisation.py`, `fig3_oscillation.py`, `fig_solver_speed.py`,
  `section53_survey.py`, `rmsa_vs_sasmodels.py`, `two_yukawa_*.py`, `numerics_test.py`, `residual_check.py`,
  `gui_screenshots.py`, ... They are in `SASfitGit/src/pyOZgui/tools/`. Moving them needs their imports changed to
  `pysasfit.oz` and, for the surveys, the potential signs flipped (convention A); otherwise a re-run computes
  different systems.
- **Two code behaviours the texts describe more strongly than the code does** (decide: change the code or the
  text):
  - ~~`ozLib.solve()` checks min S(Q) ≥ 0 only when `deflateOnUnphysical > 0`~~ **Resolved in the code
    (2026-10-12):** the screen now runs on every solve (`OZResult.minSq`, `physical`, warning; deflation only
    decides whether to retry). The texts' "applied unconditionally" is now correct; findings M6 and the matching
    report entries reduce to describing the retry as optional.
  - ~~`residualTolerance` unused~~ **Resolved in the code (2026-10-12):** every `ozLib.solve()` now recomputes
    the residual of the returned solution and marks the result not converged above `residualTolerance`
    (preference `oz.solve.residual_tolerance`). The texts' "every solve recomputes the residual" (finding M9 and
    the matching report entries) is now correct for `ozLib.solve()`; the polydisperse route
    (`GenericPolydisperseSAS`) keeps its own check for mixtures.
- Summary counts: manuscript 30 findings (13 must fix, 11 should update, 6 optional); report 47 findings
  (14 must fix, 24 should update, 9 optional).

# Part A: manuscript


Scope: statements in the three LaTeX files checked against `src/pysasfit/oz` (package `pysasfit` 0.2.0). Measured numbers (tables, figures) were not re-run. They are flagged only where the text describes behaviour the code no longer has. Paths below are relative to `src/pysasfit/oz/` unless stated otherwise.

Classes: **must fix** = factually wrong now · **should update** = outdated name/path/terminology · **optional**.

---

### manuscript.tex

#### M1. L98–104 (Synopsis) and L838–840 (Conclusions): closure and potential counts for the polydisperse route — **must fix**
> "exact partial structure factors for polydisperse interacting particles under any of twenty-four closure relations and any of nineteen pair potentials"

- `ozLib.CLOSURE_SETTERS` does have 24 entries. The polydisperse route only accepts `ozLib.multicomponentCapableClosures()`, which is **20**. It excludes Reference HNC, Modified HNC, Rescaled MSA and ZSEP (`MULTICOMPONENT_INCAPABLE_CLOSURES`). `GenericPolydisperseSAS` and the polydisperse tab enforce this.
- There are 21 potential setters. The polydisperse tab (`gui/generic_polydisperse_tab.py::_availablePotentials`) drops the charge-coupled DLVO, DLVOHydra and IonicMicrogel. That leaves 16 one-component potentials made polydisperse through Eq. (reducedtail), plus `PolydisperseHardCoreYukawa`. "Any of nineteen" therefore does not hold for polydisperse systems.
- Suggested wording: "…under any of twenty closure relations (twenty-four for one component) and any of sixteen pair potentials, plus a dedicated charged hard-core Yukawa construction". The Conclusions sentence "arbitrary pair potentials and twenty-four closure relations" needs the same change.

#### M2. L364: GUI toolkit — **must fix**
> "The package is written in Python (NumPy/SciPy) with a Tk interface"

- The Tk `oZgui.py` is gone. The GUI is now Qt (PySide6): `gui/main_window.py`, with tabs "0: OZ solver", "1: Polydisperse (any potential)", "2: RY Polydisperse Yukawa" and "3: Mixture validation". It starts with the command `pysasfit-oz` and is also a tab of `pysasfit-gui`.
- Suggested wording: "…written in Python (NumPy/SciPy) with a Qt (PySide6) interface, distributed as the `pysasfit.oz` sub-package of pySASfit…"

#### M3. L365–369: number of polydisperse closures and which ones are excluded — **must fix**
> "Nineteen closure relations are available for polydisperse systems; four (reference HNC, modified HNC, rescaled MSA and the Euler--Rahman closure) … are excluded"

- Now **twenty** are available.
- The four excluded ones are Reference HNC, Modified HNC, Rescaled MSA and **ZSEP** (`ozLib.MULTICOMPONENT_INCAPABLE_CLOSURES`; ZSEP's conditions are hard-sphere specific).
- Euler–Rahman is not offered anywhere, not even for one component (comment in `CLOSURE_SETTERS`: "DELIBERATELY NOT offered").
- Suggested wording: "Twenty of the twenty-four closures are available for polydisperse systems; four (reference HNC, modified HNC, rescaled MSA and ZSEP) … are excluded. The Euler–Rahman closure is not offered at all, because it does not converge reliably."

#### M4. L371–378: solver fallback chain — **must fix**
> "Fixed-point acceleration follows the same strategy in both implementations: Anderson acceleration (SUNDIALS KINSOL KIN_FP) is tried first, with Newton--Krylov … and Picard iteration as successive fallbacks."

- The Python code has no such cascade.
- `ozLib.solve()` and `GenericPolydisperseSAS._makeSolver` run exactly one solver: the one chosen, or by default the first key of `ozLib.SOLVER_CLASSES`. That default is `"sundials4py: Fixed-Point (Anderson)"` when sundials4py is installed and `"scipy Anderson"` otherwise.
- Substituting another solver is explicitly avoided (`ozLib._deflateUntilPhysical` docstring: "silently returns a result from a solver the caller did not choose").
- The only automatic cascade is in `rypolydisperseWrapper._solverClasses()`: scipy Anderson → MDIIS → Picard. It has no SUNDIALS and no Newton–Krylov.
- Suggested wording: "The default solver is Anderson-accelerated fixed-point iteration (SUNDIALS KINSOL `KIN_FP`, or SciPy's Anderson mixing when SUNDIALS is not installed). Newton–Krylov, Biggs–Andrews, MDIIS and Picard / Mann iteration are available as alternatives; a failed solve is reported, not silently retried with another solver." Drop "the same strategy in both implementations" unless the C plugin has been re-checked.

#### M5. L380–386: "The Picard fallback" — **should update**
> "The Picard fallback is implemented in the averaged (Mann, or Krasnoselskii--Mann) form"

- The solver is now a selectable method named **"Picard / Mann iteration"** (`ozLib._BASE_SOLVER_CLASSES`; "Picard iteration" is still accepted as an alias). It is not a fallback.
- The damping `mannAlpha` is set in `ozLib.solve(mannAlpha=…)` or in the OZ solver tab (default 1, preference `oz.solve.mann_alpha`).
- The iteration limit is ceil(N/α), with N = 5000 (`PicardOZsolver.DEFAULT_MAX_ITERATIONS`, preference `oz.solvers.max_iterations_picard`).
- Convergence is tested on the residual ‖T(x)−x‖/‖x‖, not on the damped increment (`picardOZsolver.py`).
- Suggested wording: "The Picard / Mann solver implements the averaged form … with α = 1 (undamped Picard) as the default; the iteration limit scales as 1/α, and convergence is tested on the residual ‖T(Γ)−Γ‖ rather than on the damped step."

#### M6. L479–484: physicality screen "applied unconditionally" — **must fix**
> "removes every one of the unphysical solutions above. We therefore apply it unconditionally rather than as a warning."

- The screen is unconditional only on the polydisperse route: `GenericPolydisperseSAS._verifyPhysical` raises when min S(Q) < −1e-6.
- `ozLib.solve()` screens only when `deflateOnUnphysical > 0`. Its default is 0, and the OZ solver tab's "deflate if S(Q)<0" box also defaults to "0" (`gui/oz_solver_tab.py`). With the defaults, a one-component solve returns a root with min S(Q) < 0 without complaint.
- Suggested action: either make the screen unconditional in `ozLib.solve`, or reword to "On the polydisperse route the screen is applied unconditionally; for one-component solves it is applied when deflation is enabled (`deflateOnUnphysical`)."

#### M7. L485–487: verification "by default, every solve" — **must fix**
> "the library verifies by default: every solve is repeated with an independent solver from a cold start and the result returned only if the two agree."

- This is true only for `ozLib.solve()` (`verify=True` by default).
- `GenericPolydisperseSAS`, which is the polydisperse route the paper is about, and `polydisperse_fit` do **no** cross-solver verification. They only run the residual and physicality checks.
- Even in `ozLib.solve`, a result can now come back without agreement having been shown. If no verifier converges, `VerificationUnavailable` is raised and carries the unverified result as `.result`, and the GUI shows the run tagged "[UNVERIFIED]". A non-converged primary solve gives a `RuntimeWarning` and "[NOT CONVERGED]".
- Suggested wording: "`ozLib.solve` verifies by default: the solve is repeated from a cold start with an independent solver, and a disagreement raises an error; if no verifier converges the result is returned explicitly marked as unverified. The polydisperse route relies on the residual and physicality checks instead." Alternatively, add verification to the polydisperse route.

#### M8. L489–497: choice of verifier — **must fix**
> "We therefore verify with a second fixed-point method --- a different implementation rather than a different algorithm class"

- The code now requires a different **algorithm family** (`ozLib.SOLVER_FAMILIES`).
- KIN_FP, scipy Anderson and MDIIS are one family, "Anderson/DIIS", and never verify one another. With the default primary solver, a second Anderson implementation would therefore be skipped.
- The default `verifyWith = ("Biggs-Andrews", "Picard / Mann iteration")`. Only a verifier that has **converged** counts, and the Picard / Mann verifier is damped with `verifyMannAlpha = 0.5`.
- Newton–Krylov is still excluded, as the text says.
- Suggested wording: "We therefore verify with a fixed-point method from a different algorithm family — Biggs–Andrews extrapolation first, then damped (α = 0.5) Picard / Mann iteration — never with a Newton–Krylov method, and never with another Anderson/DIIS variant, which shares the primary's failure modes."

#### M9. L516–524: "Every solve therefore recomputes the residual" — **must fix**
> "Every solve therefore recomputes the residual before accepting the result"

- `ozLib.solve()` does not recompute the residual. Its `residualTolerance` argument is accepted but never used.
- `Sundials4pyKinsolFPOZsolver` sets `converged = True` whenever KINSOL returns success, which is exactly the case the paragraph warns about.
- The residual is recomputed only in `GenericPolydisperseSAS._verify`, and only for p > 1. The one-component reference solve checks finiteness only, as stated in the code comment there.
- Suggested action: implement the check in `ozLib.solve` (the unused `residualTolerance` suggests this was intended), or reword to "the polydisperse route recomputes ‖T(Γ)−Γ‖ before accepting a multicomponent result; one-component solves rely on cross-solver verification."

#### M10. L519–522 and L533–535: defective Picard stopping test — **optional**
> "a damped Picard iteration whose convergence test compared successive norms of Γ … reported a converged solution"

- This is historically correct, but the defect has since been fixed. `PicardOZsolver` now tests ‖T(x)−x‖/‖x‖ (`picardOZsolver.py`, comments "CONVERGENCE ON THE INCREMENT" and "ON THE RESIDUAL").
- Suggested action: add "(since corrected)" so a reader does not think the shipped solver still has the defect.

#### M11. L575–581 (Table tab:timing): solver labels — **should update**
> "Anderson acceleration 0.0100" … "Picard iteration 0.0542"

- "Anderson acceleration" (`andersonOZsolver`) is deregistered and no longer offered (comment in `ozLib._BASE_SOLVER_CLASSES`).
- MDIIS, which replaced it, is not in the table.
- "Picard iteration" is now named "Picard / Mann iteration".
- Suggested action: keep the measurements but relabel "Picard / Mann iteration (α = 1)". Either drop the "Anderson acceleration" row or footnote that it is no longer offered. Re-measuring with MDIIS is optional.

#### M12. L558–561 and L928–931: benchmark script — **should update**
> "using the benchmark script distributed with the package" / "Measured with tools/benchmark.py"

- No `tools/benchmark.py` exists anywhere in `pySASfit_build`, and none is shipped with `pysasfit`.
- Suggested action: add the script to the repository (for example `docs/oz/latex/manuscript/tools/`) and cite its real path, or remove the claim.

#### M13. L626–630: code example — **must fix**
> `result = ozLib.solve("HardSphere", phi=0.3, … solver="SUNDIALS KIN_FP")`

- `"SUNDIALS KIN_FP"` is not a valid solver name. Checked: it raises `ValueError: unknown solver 'SUNDIALS KIN_FP'`. The label is `"sundials4py: Fixed-Point (Anderson)"`.
- The module is now imported as `from pysasfit.oz import ozLib`.
- Suggested replacement:
  `from pysasfit.oz import ozLib`
  `result = ozLib.solve("HardSphere", phi=0.3, closure="Rogers-Young", findConsistentParameter=True, solver="sundials4py: Fixed-Point (Anderson)")`
  This is also the default, so the `solver=` argument can be dropped.

#### M14. L774: figure generator — **should update**
> "Drawn by tools/fig_realdata_fit.py directly from the saved session"

- No `fig_realdata_fit.py` exists in `pySASfit_build`.
- Suggested action: add the script or remove the claim.

#### M15. L889–902 (Open item 3): transform-type decision — **must fix**
> "Type 4 is nonetheless NOT the default … Decide whether to default the one-component path to type 4"

- This has been decided and implemented. `ozLib.solve(transformType=None)` is automatic: `oZfixpointOperator.resolvedTransformType()` returns 4 (DST-IV) for one component and 1 for a mixture. Checked: a default `ozLib.solve("HardSphere", 0.3)` uses type 4.
- The setting is the preference `oz.solve.transform_type` ("auto"). The polydisperse tab defaults to type 1 (`FACTORY_SETTINGS`).
- Suggested action: close the item. In the main text, state "DST-IV by default for one component, DST-I for mixtures (selectable)".

#### M16. L955–958 (Fig. fig:survey caption): "square-well fluid" — **should update**
> "for hard spheres (top) and a square-well fluid (bottom)"

- Section 5.3 itself says this system is a repulsive **shoulder**. Under convention A (`sign_convention.py`) a shoulder is ε < 0, so a reader of the current code will read "square-well" as attractive. See S1.
- Suggested wording: "…and a square-shoulder fluid (ε = −1 k_BT in the current sign convention)".

#### M17. L1019–1022 (Software availability) — **should update**
> "The software is distributed with SASfit."

- The Python code is now the `pysasfit.oz` sub-package of the `pysasfit` distribution (pyproject `name = "pysasfit"`), with the GUI commands `pysasfit-oz` and `pysasfit-gui`.
- Suggested wording: "The Python implementation is distributed as `pysasfit.oz` within the pySASfit package (GUI: `pysasfit-oz`); the C solver is part of SASfit. [URL, version, DOI, licence]".

#### M18. L91 and L364–365: C plugin vs Python package sign convention — **should update**
> "distributed both as a Python package … and as a C plugin for SASfit"

- The two implementations no longer share parameter signs. The Python code uses convention A, where a positive strength is attractive (`sign_convention.py`). SASfit's C OZ solver still uses β u = +ε for the square-well and piecewise-constant potentials (docstring of `setSquareWellPotential`).
- Session files from pyOZgui are converted on load, but a user moving parameters between the C plugin and Python by hand is not protected.
- Suggested action: add one sentence stating the sign convention used in the paper, and that the SASfit C solver's square-well and piecewise-constant ε has the opposite sign.

#### M19. L1001–1006 (Fig. fig:controls) and L1011–1015 (Fig. fig:guierror): screenshots — **optional**
- `gui_controls.png` (and presumably `gui_error.png`) were captured from the Tk pyOZgui. The Qt tab "1: Polydisperse (any potential)" has the same controls: "Classes (S)", "Classes (form f.)" and an "Approx. error" plot tab (`gui/generic_polydisperse_tab.py`). The captions remain true.
- Suggested action: retake the screenshots from `pysasfit-oz` before submission.

#### M20. L223–235: number of distributions vs "All four" — **optional**
> "All five distributions implemented (Schulz, Gaussian, log-normal, Weibull, gamma) … All four are moment-exact"

- `polydisperse_nodes.DISTRIBUTIONS` has five entries. Gamma uses the same Gauss–Laguerre path as Schulz, so all five rules are moment-exact.
- Suggested wording: "All five are moment-exact…".

---

### validation_section.tex

#### V1. L457–459 and L490–496: type-1 transform "retained as the default" — **must fix**
> "We nevertheless retain type 1 as the default, because the type-4 condition cannot be met for a mixture."

- Type 4 is now the default for one component and type 1 for mixtures (`oZfixpointOperator.resolvedTransformType`, `ozLib.solve(transformType=None)`, preference `oz.solve.transform_type = "auto"`).
- The opening claim at L457–459 ("Every agreement quoted in this section was obtained with the type-1 discrete sine transform") already conflicts with L180–183, which call the one-component PY row a DST-IV result. Under the new default, one-component results are type 4 unless type 1 is forced.
- Suggested wording for L490ff: "Type 4 is therefore the default for one component and type 1 for mixtures, where the type-4 condition cannot be met…; either can be forced (`transformType`)." State per row which transform was used.

#### V2. L161–167, L238 and L164–165: tool scripts referenced but not shipped — **should update**
> "fig1_discretisation.pdf now has tools/fig1_discretisation.py" / "tools/rmsa_vs_sasmodels.py reproduces sasmodels"

- Neither `fig1_discretisation.py` nor `rmsa_vs_sasmodels.py` exists in `pySASfit_build`. Code comments also cite `tools/residual_check.py`, which is likewise missing.
- `section53_survey.py` does exist, at `docs/oz/latex/manuscript/tools/`. It still imports with the pre-package form `from generic_polydisperse_sas import GenericPolydisperseSAS`, so it will not run against `pysasfit.oz` as it stands. See also S4.
- Suggested action: add the scripts, update them to `from pysasfit.oz.… import …`, and cite their real paths. Otherwise drop the reproducibility claims.

#### V3. L297–305: shelved Ginoza module — **should update**
> "A partial implementation of the Ginoza route exists in ginoza_yukawa_msa.py and is deliberately shelved"

- There is no `ginoza_yukawa_msa.py` in `pysasfit.oz` or anywhere else in the build.
- Suggested action: say that it was not carried over into pysasfit, or restore the file and give its path.

#### V4. L147–159: sign of K — **optional**
> "Liu2005 write the tail as −K exp[−Z(r−1)]/r, so positive K is ATTRACTIVE. This package's solver follows that convention"

- This is still correct (`HardSphereDoubleYukawa` was already convention A). Convention A is now project-wide, though. The single Yukawa and the square-well / piecewise-constant potentials were flipped relative to pyOZgui (`sign_convention.FLIPPED_ARGS`).
- Suggested action: one sentence saying that the whole package uses positive = attractive. That also gives S1 and S2 something to refer to.

#### V5. L169–172: module name — **optional**
> "polydisperse_nodes offers two discretisations --- sizeClasses … and quantileClasses"

- The functions exist under these names. The full module path is now `pysasfit.oz.polydisperse_nodes`.
- Suggested action: use the full path if module names are kept in the text.

---

### section_5_3_schemes.tex

#### S1. L20–22 and L34–45: square-shoulder sign note — **must fix**
> "The implementation writes the pair potential as u(r) = ε … so a POSITIVE ε … is REPULSIVE --- a square shoulder. Attraction requires ε < 0."

- The sign is now reversed. `setSquareWellPotential` sets β u = −ε in the well: ε > 0 is an attractive well and ε < 0 a shoulder (`oZfixpointOperator.py`, `sign_convention.py`, "convention A").
- The survey ran under the old convention, so the system it computed corresponds to **ε = −1 k_BT** today.
- `tools/section53_survey.py` still passes `SquareWell (1.0, 0.1)`. Re-running it with the current code would compute an attractive well, a different system from the one tabulated.
- Suggested action:
  - L21: write "$\varepsilon = -1\,k_BT$ (repulsive shoulder)".
  - Rewrite the note: "In the sign convention used throughout (positive strength = attractive; β u = −ε inside the well) the survey's system has ε = −1 k_BT, a repulsive shoulder. The survey was computed with the earlier pyOZgui convention (β u = +ε)."
  - Change `SYSTEMS` in the survey script to `(-1.0, 0.1)`.

#### S2. L23–24, L65–67, L74–90 (Table tab:yukawa) and L93–94: Yukawa sign — **must fix**
> "hard-core Yukawa fluid (screening length 0.5σ, contact strength 2 k_BT)" … "the longest-ranged and the strongest attraction"

- The survey script uses `("Yukawa", (0.5, 2.0, 1.0))`. In pyOZgui the Yukawa strength was β u itself (K > 0 repulsive; `sign_convention.FLIPPED_ARGS["Yukawa"]`), so the tabulated Yukawa systems were **repulsive**. That fits the note at L43–44, which says the adhesive row is the only attractive system.
- In convention A the same systems are K = −2 and K = −5 (`setYukawaPotential`: K > 0 attractive). L93's "strongest attraction" is wrong for the K = 5 case unless those two failed runs used a different setup.
- Suggested action:
  - State the sign explicitly: "repulsive hard-core Yukawa fluid, K = −2 k_BT in the present convention".
  - Change the K column of Table tab:yukawa to −2.0 and −5.0, or say "|K|, repulsive".
  - Replace "strongest attraction" with "strongest repulsion" after checking the run.
  - Update the survey script.

#### S3. L157 and L185: "square well" labels — **should update**
> `\multicolumn{2}{c}{square well}` … `square well & 14.8 & 43.2`

- The note at L39–41 says every "square well" label "has been corrected", but these two remain. Under the current sign convention the label is also misleading about the sign (see S1).
- Suggested action: rename to "square shoulder".

#### S4. L4–8: regeneration commands — **should update**
> "python tools/section53_survey.py --summarise survey.jsonl"

- The script is at `docs/oz/latex/manuscript/tools/section53_survey.py`. It imports `from generic_polydisperse_sas import …` (pre-package layout) and uses old-convention potential signs (S1, S2).
- Suggested action: switch the import to `from pysasfit.oz.generic_polydisperse_sas import GenericPolydisperseSAS` and convert the `SYSTEMS` signs, or note that `survey.jsonl` was produced with pyOZgui and `--summarise`/`--plot` only re-read the stored data.

#### S5. L245–254: residual recomputation and physicality screen — **optional**
> "solutions are therefore screened for physicality in addition to the residual test … every solve consequently recomputes its own residual"

- For the polydisperse route used in this section this is correct: `GenericPolydisperseSAS._verifyPhysical` always runs, and `_verify` recomputes ‖G(γ)−γ‖ for p > 1. The p = 1 reference solve only checks finiteness, though, and one-component `ozLib.solve` does neither by default (see M6, M9).
- Suggested action: scope the sentence to "every multicomponent solve" so it stays true when read alongside §3.

---

### Summary

| Class | Count | Items |
|---|---|---|
| must fix | 13 | M1, M2, M3, M4, M6, M7, M8, M9, M13, M15, V1, S1, S2 |
| should update | 11 | M5, M11, M12, M14, M16, M17, M18, V2, V3, S3, S4 |
| optional | 6 | M10, M19, M20, V4, V5, S5 |

Total: 30 findings (manuscript.tex 20, validation_section.tex 5, section_5_3_schemes.tex 5).


# Part B: report


Reviewed against `src/pysasfit/oz` (and `src/pysasfit/oz/gui`, `src/pysasfit/core/settings_schema.py`).
Source paths below are relative to `src/pysasfit/oz/` unless stated otherwise. Benchmark numbers were not re-run.

Classes: **must fix** = factually wrong now; **should update** = outdated name/path/terminology/count; **optional**.

---

### ozGUI_ozLib_report.tex (master)

1. **L118, L137, L145–148 — optional.** "ozGUI / ozLib: Ornstein--Zernike Solver Toolkit" (pdftitle, header, title).
   The code is now the `pysasfit.oz` subpackage of pySASfit (GUI command `pysasfit-oz`, also a tab of `pysasfit-gui`).
   *Action:* consider retitling, e.g. "pysasfit.oz (formerly ozGUI/ozLib): Ornstein–Zernike Solver Toolkit". The same title/header strings appear in both fragments (doc L29, L48, L56; suppl. L29, L48, L57, L61).

2. **L16, L34, L86–94 (comments only) — optional.** "Figures live in docs/figures/", "docs/manuscript/manuscript.tex", "docs/ozgui.bib".
   These files are now at `docs/oz/latex/figures/`, `docs/oz/latex/manuscript/manuscript.tex`, `docs/oz/latex/ozgui.bib`. The build still works because the paths are relative. *Action:* fix the comment paths. The same "docs/figures/" comment is in doc L5 and suppl. L5.

---

### ozGUI_ozLib_documentation.tex (Part I)

#### Must fix

3. **L155–160 and L183–201 — must fix.** "$r_i = i\,\Delta r$ …" / "implemented here as a type-I discrete sine transform (DST) … $\Delta q=\pi/((N+1)\Delta r)$".
   For a one-component solve the default is now **DST-IV**: nodes at $r_i=(i+\tfrac12)\Delta r$, $\Delta q=\pi/(N\Delta r)$, $q_k=(k+\tfrac12)\Delta q$. DST-I is used only for mixtures or when `transformType=1` is set. See `oZfixpointOperator.hankelTransform()` / `resolvedTransformType()` and `ozLib.solve(transformType=None)`, which is automatic: 4 for one component, 1 for a mixture. The preference is `oz.solve.transform_type`.
   *Action:* describe both transforms, say that type 4 is the one-component default, give the type-4 grid and normalisation, and say why mixtures keep type 1 (pair cores cannot all fall between nodes, `checkTransformAlignment()`). Mention that the default yields a 2nd-order scheme where type 1 is 1st order, as quantified in suppl. Table `tab:S0accuracy`.

4. **L168–172 — must fix.** "the historical defaults are $N=4096$, $n_\sigma=100$".
   The default is now $N=2^{12}-1=4095$ (`oZfixpointOperator.__init__`, `numberOfRadialSamplingPoints … else 2**12 - 1`). $n_\sigma$=`DEFAULT_POINTS_PER_SIGMA`=100 is unchanged. When `transformType` is passed explicitly, the grid is snapped with `bestGridSize(k, type)`: $2^k-1$ for type 1, $2^k$ for type 4.
   *Action:* replace with "defaults $N=4095$ ($=2^{12}-1$), $n_\sigma=100$".

5. **L269–277 — must fix (sign convention).** "$U(r)/k_BT = K\,\frac{\sigma}{r}\,e^{-r/\lambda}$".
   `setYukawaPotential` now uses convention A: $\beta u = -K\,e^{-(r-\sigma)/\lambda}/(r/\sigma)$ with a hard core, or $-K e^{-r/\lambda}/(r/\sigma)$ without one. **$K>0$ is attractive.** pyOZgui used $K>0$ repulsive. See `sign_convention.py` (`FLIPPED_ARGS["Yukawa"]`); old sessions are converted on load.
   *Action:* put a minus sign in the formula, state "$K>0$ attractive, $K<0$ repulsive (convention A; opposite to pyOZgui)", and give the shifted form explicitly.

6. **L412–425 — must fix (sign convention).** "$U/k_BT = \varepsilon_1$ … each $\varepsilon_j$ may be positive (shoulder) or negative (well)".
   `setPiecewiseConstantHSPotential` now uses $\beta u = -\varepsilon_j$, so $\varepsilon_j>0$ is a **well** and $\varepsilon_j<0$ a shoulder. This is opposite to pyOZgui and to SASfit's C `sasfit_oz_potential_piecewise_constant_HS.c`.
   *Action:* use $-\varepsilon_j$ in the equation and swap "shoulder"/"well".

7. **L442–449 — must fix (sign convention).** "$\varepsilon$ for $\sigma\le r\le\sigma+\delta$ … $\varepsilon<0$ for an attractive well".
   `setSquareWellPotential` now uses $\beta u=-\varepsilon$: $\varepsilon$ is the **depth**, $\varepsilon>0$ attractive and $\varepsilon<0$ a shoulder. This matches the SASfit SQ plugins and is opposite to the C OZ solver. In addition, δ is an absolute length, not a fraction of σ (docs/oz/manual.md).
   *Action:* fix the equation and the sign sentence, and add a note on the convention change and session conversion.

8. **L1310–1314 — must fix.** "Picard iteration — Plain, undamped fixed-point iteration … until the relative change in $\|\Gamma_n\|$ falls below a convergence criterion."
   The solver is now "**Picard / Mann iteration**" (alias "Picard iteration" still accepted, `ozLib.SOLVER_ALIASES`). It uses $x\leftarrow(1-a)x+aT(x)$ with $0<a\le1$ (`mannAlpha`, default 1). Convergence is tested on the **residual** $\|T(x)-x\|/\|x\|$, not on successive norms. The iteration limit is $\lceil N/a\rceil$ with $N$=`PicardOZsolver.DEFAULT_MAX_ITERATIONS`=5000 (preference `oz.solvers.max_iterations_picard`). The solver sets `converged`, `finalRelativeProgress`, `iterations` and `iterationLimit` (`picardOZsolver.picardIteration`).
   *Action:* rewrite the subsection: rename it, give the Mann update, the residual test, the $\lceil N/a\rceil$ limit, and when to damp (dense or strongly attractive states).

9. **L1316–1324 — must fix.** "Anderson acceleration (hand-written) — A direct Python implementation of Anderson mixing …".
   This solver is **no longer offered**: it was deregistered as a duplicate of scipy Anderson and moved to `_obsolete/` (`ozLib._BASE_SOLVER_CLASSES` comments). **MDIIS** (`mdiisOZsolver.MDIISOZsolver`) was added in its place and has no subsection.
   *Action:* replace the subsection with one on MDIIS (Anderson/DIIS family, regularised, history depth `oz.solvers.mdiis_history`; a SUNDIALS-free alternative when scipy Anderson stalls). If desired, keep one historical sentence on the removed solver.

10. **L66–84, L108–117, L1409–1413, L1490–1531, L1533–1548, L1550–1551, L1675 — must fix (GUI technology / entry point).** "an interactive Tkinter GUI (`oZgui.py`)", "`oZgui.py` is a Tkinter application, run directly with `python oZgui.py`", "a Tkinter equivalent of that Tcl/Tk/BLT window".
    There is no Tk any more. The GUI is a Qt (PySide6) port: `pysasfit.oz.gui.main_window` (command `pysasfit-oz`, or `python -m pysasfit.oz.gui.main_window`), also embedded as a tab of `pysasfit-gui`. The one-component solver is the tab "0: OZ solver" (`gui/oz_solver_tab.OZSolverTab`), alongside "1: Polydisperse (any potential)", "2: RY Polydisperse Yukawa" and "3: Mixture validation". The matplotlib toolbar is `NavigationToolbar2QT`.
    *Action:* replace Tkinter/oZgui.py throughout with Qt / `pysasfit-oz` / "OZ solver tab", and mention the four top-level tabs.

11. **L1504–1514 — must fix (GUI controls).** "grid size controls (`Grid points (N)`/`Points per σ`, both defaulting to "auto" …), max. iterations, calculate/interrupt/…".
    In `oz_solver_tab._buildLeftPanel` the **"Max iterations" field no longer exists**. Limits come from Settings → Preferences (`oz.solvers.max_iterations`=600, `oz.solvers.max_iterations_picard`=5000), and a grey line shows the limit that will apply. The grid control is now **"Gridsize (2^n - 1), n:"**, an exponent defaulting to auto = n=12 → 4095, plus "Points per σ".
    New controls are not mentioned: "damping a" (Picard/Mann only, greyed otherwise), the checkbox "verify against a second solver" (on by default), "deflate if S(Q)<0" (0–3), a "Symlog/asinh threshold" entry, and a run "Label". Run labels get "[NOT CONVERGED]" / "[UNVERIFIED]" suffixes.
    *Action:* update the control list accordingly.

#### Should update

12. **L1306 — should update.** "Eight solver algorithms are available".
    `ozLib.SOLVER_CLASSES` has 9 entries, 7 algorithms: Picard/Mann, scipy Anderson, MDIIS, scipy Newton–Krylov, Biggs–Andrews, KINSOL Newton–Krylov (GMRES/FGMRES/TFQMR), and KINSOL KIN_FP. The SUNDIALS ones appear only if sundials4py is installed.
    *Action:* update the count and list. Optionally mention `ozLib.SOLVER_FAMILIES`: Anderson/DIIS, Newton–Krylov, vector extrapolation, Picard/Mann.

13. **L946–951, L1385–1403 (table column "Anderson" + caption), L1866–1879, L1894–1901, L1641 — should update (removed hand-written Anderson).**
    These are: "The hand-written **Anderson acceleration** solver … should not be trusted uncross-checked", "Anderson … wrong answer, uncaught", "followed by the hand-written Anderson solver", "`Anderson acceleration` / `scipy Anderson` — split into two".
    That solver is not registered any more (see #9). The measurements may stay as history, but the text should not recommend for or against a solver users cannot select.
    *Action:* mark the column/rows as "(removed solver, historical measurement)" or drop them. In the Tcl table (L1641), map Tcl "Anderson mixing" → `scipy Anderson` / `MDIIS` / `sundials4py: Fixed-Point (Anderson)`.

14. **L662, L670, L1310, L1453, L1461–1462, L1635, L1651, L1844, L1868 — should update (solver name).** "Picard iteration" / `solver="Picard iteration"`.
    The canonical name is now `"Picard / Mann iteration"`; the old name is only an alias (`ozLib.resolveSolverName`). In the Tcl naming table (L1635–1638, caption L1650–1651), the "identical (the one exact match)" claim no longer holds. Tcl's `Mann`/`Krasnoselskij` variants are now covered by Picard/Mann with damping `a`.
    *Action:* rename, and update the table rows and caption.

15. **L1415–1417 (and other module-path mentions: L200, L469, L507, L1220, L1409, L1457, L1497–1498, L1520, L1588, L1710) — should update (module paths).** "`from ozLib import solve`", "`ozLib.py`", "`oZfixpointOperator.py`", "`rmsaWrapper.py`".
    The modules now live in the package `pysasfit.oz` (`src/pysasfit/oz/…`).
    *Action:* change the example to `from pysasfit.oz.ozLib import solve` (or `from pysasfit.oz import ozLib`). Say once that every module named in the report is `pysasfit.oz.<module>`.

16. **L1436–1444 — should update.** The listed `solve(...)` signature stops at `onSolverCreated=None`.
    The current signature (`ozLib.solve`) adds `closureParam2`, `verify=True`, `verifyWith=("Biggs-Andrews","Picard / Mann iteration")`, `verifyTolerance=1e-3`, `residualTolerance=1e-6`, `deflateOnUnphysical=0`, `transformType=None`, `mannAlpha=None` and `verifyMannAlpha=0.5`. `OZResult` now also carries `converged`, `finalRelativeProgress`, `iterations`, `iterationLimit`, `verifiedWith`, `verificationDifference` and `mannAlpha`. A non-converged solve emits a `RuntimeWarning`, and `VerificationUnavailable` is raised when no verifier converges.
    *Action:* update the listing and add a short paragraph on verification, deflation and convergence reporting. It can reference Part II, but Part I never mentions that verification is on by default.

17. **L1463–1471 — should update.** "(Rogers–Young, HMSA, Modified HNC, BPGG, CJVM, BB — see `ozLib.CONSISTENT_PARAMETER_CLOSURES`)".
    The dict now also contains Khanpour, Modified Verlet, Carbajal-Tinoco and Extended Rogers-Young (`ozLib.CONSISTENT_PARAMETER_CLOSURES`).
    *Action:* extend the list.

18. **L1021–1029 — should update.** "19 of the 24 closures work multicomponent unchanged … The five that do not — RHNC, MHNC, Rescaled MSA, EuRah, ZSEP".
    `ozLib.multicomponentCapableClosures()` returns **20 of the 24** offered closures. Only Reference HNC, Modified HNC, Rescaled MSA and ZSEP are hidden. EuRah is not among the 24 offered at all.
    *Action:* "20 of the 24 … The four that do not (RHNC, MHNC, RMSA, ZSEP) …; EuRah is not offered in any case".

19. **L1205–1212 and L1278–1283 — should update (sign in examples).** "at $\varepsilon=-0.02$ … to … $\varepsilon=-0.5$" and "($\varphi=0.15$, $\varepsilon=-0.5k_BT$, $\delta=0.3\sigma$)".
    With convention A an attractive well has $\varepsilon>0$. `oZsolver.calculateAnalyticalStructureFactorForSquareWell` (ε = depth), `orpaWrapper.orpa_compute` (`eps_kT>0` attractive) and `rfaWrapper.rfa1_compute` (depths) all now take the positive number.
    *Action:* write $\varepsilon=+0.02$ / $+0.5\,k_BT$ (well depth) and state that ε is the depth.

20. **L1571–1574 (Tcl potential-name table) — should update.** "`PiecewiseConstant`, `SoftSphere`, `Fermi`, … `DLVO Hydra`, `GGCM-n`, `HS 3Yukawa` — same".
    The Python names (`getAvailablePotentialNames()`) are `PiecewiseConstantHS`, `FermiDistribution`, `PSM`, `DLVOHydra`, `GGCMn` and `HS3Yukawa`, so several are not "same". More importantly, the **ε sign of SquareWell and PiecewiseConstant(HS) is opposite** between the Tcl/C OZ solver and pysasfit.oz (`sign_convention.py`), and the table should say so.
    *Action:* correct the names and add a "sign flipped (convention A)" note for SquareWell, PiecewiseConstantHS and Yukawa.

21. **L1617–1620 (Tcl closure table) — should update.** Only `Kovalenko-Hirata` and `ZSEP` are listed as "Python only".
    `Khanpour`, `Carbajal-Tinoco`, `Modified Verlet` and `Extended Rogers-Young` are also Python-only (`ozLib.CLOSURE_SETTERS`).
    *Action:* add the four rows.

22. **L1660–1679 (KINSOL table + caption) — should update.**
    - `KINSetNumMaxIters` → `"Max iterations:"` → `maxIterations=`: there is no GUI field any more. Use Preferences `oz.solvers.max_iterations` / `max_iterations_picard`, or `maxIterations=` in scripts.
    - `KINSetMAA` "not directly exposed": now Preferences `oz.solvers.kinsol_maa` (`preferences.apply_solver_settings`).
    - `KINSetFuncNormTol` "`self.convergenceCriterion` instead": the Python KINSOL driver uses a fixed 1e-10 (`sundials4pyKinsolOZsolver`). The general criterion is Preferences `oz.solvers.convergence`.
    - Caption: "plus the two simpler knobs above" refers to controls that are gone.

    *Action:* point the rows at Settings → Preferences.

23. **L1866–1871 — should update.** "Dense hard spheres ($\varphi\gtrsim0.45$): plain Picard iteration fails to converge outright".
    True only for undamped Picard. Picard/Mann with $a\le0.6$ converges, e.g. HS/PY at φ=0.53 needs about $4170/a$ iterations within the $\lceil 5000/a\rceil$ limit (`picardOZsolver` comments). The damped Picard/Mann is also the second default verifier with $a=0.5$.
    *Action:* add the damped alternative.

24. **L1910–1919 — should update.** "this project's own historical default, $N=4096$ … is not actually the fastest … $N=4095$ … would be measurably faster".
    The default is now 4095 (see #4). For the one-component default (DST-IV), $N=2^k$ is the fast size, and `bestGridSize(n, transformType)` returns the right one. The text also calls the transform "the type-I DST used here".
    *Action:* say this recommendation has been adopted, and add the DST-IV size rule.

25. **L1217–1228, L1263, L1277, L1290 — should update (paths).** "`rmsa_c_source/`", "`oneyukawa_c_source/`", "`orpa_c_source/`", "`rfa_c_source/`", "`tools/rmsa_vs_sasmodels.py`".
    The C sources are now under `extern/oz/<name>_c_source/`, and the compiled libraries ship as `src/pysasfit/oz/lib*.so`. `tools/rmsa_vs_sasmodels.py` does not exist in the pySASfit tree.
    *Action:* update the paths and remove or redirect the script reference (see Part II #42).

#### Optional

26. **L1346 — optional.** "extrapolates using order |`KINSetMAA`|". For Biggs–Andrews this order is the preference `oz.solvers.biggs_maa` (default 2). *Action:* mention the preference.

27. **L668–676 — optional.** "neither … converges within 2000 plain-Picard iterations". This is a historical measurement; the current undamped limit is 5000 and damping is available. *Action:* add "(undamped, historical limit)".

---

### ozGUI_ozLib_multicomponent_supplement.tex (Part II)

#### Must fix

28. **L154–161 and L2076–2079 — must fix.** "do those reading `repulsivePartOfP2Ppotential` (HMSA, VM, CJVM, BB, DH, CG, SMSA), whose split arrays are still 1-D" and the Known limitation "…are not yet wired for the multicomponent potential; those arrays remain 1-D. PY, HNC, RY, Verlet and BPGG work."
    These closures are now in `ozLib.multicomponentCapableClosures()`. Only RHNC, MHNC, RMSA and ZSEP are excluded (`MULTICOMPONENT_INCAPABLE_CLOSURES`). Checked by running: `ozLib.solve('Polydisperse', …, closure='HMSA'|'Vompe-Martynov'|'Duh-Haymet')` converges with 3 classes.
    *Action:* reduce the remark and limitation to the reference-solve closures (RHNC, MHNC, RMSA, EuRah, ZSEP), and delete limitation item 2.

29. **L801–806 — must fix.** "Each tab writes its own file … holding the computed runs, every input field, …".
    Computed runs are **no longer saved**: `gui/tab_controls._onSaveAll` writes `"runs": []`, and the generic tab's autosave likewise writes no runs. This also contradicts the supplement's own L1838–1846.
    *Action:* replace "the computed runs" with "the settings"; they are regenerated by Compute.

30. **L820–846 — must fix (modelling-tab description).** "Both modelling tabs follow the same pattern: … plot tabs for $I(Q)$, the approximation error, $S_{ij}(Q)$, $S(Q)$ and the size distribution; … (Tk is not thread-safe)" and "Because both new SAS classes derive from the same `PolydisperseSASBase`, all six … schemes … The approximation code is byte-identical between the tabs".
    This pattern applies only to the generic tab (`generic_polydisperse_tab.PLOT_TABS`). The RY Polydisperse Yukawa tab computes S only: its plot tabs are "S(Q)", "S_ij(Q)", "Size distribution" plus Summary (`ry_polydisperse_yukawa_tab.PLOT_TABS`). `APPROX_SCHEMES` is defined there but unused, and the form factor is built and discarded. The "two new SAS classes" sharing `PolydisperseSASBase` refer to the old MSA-Yukawa and Robertus tabs, which are no longer registered (`main_window.EXTRA_TABS`). The GUI is Qt, not Tk (worker thread plus queue, via `qtkit`).
    *Action:* describe the generic tab and the RY tab separately. For the RY tab, add that it uses the selected distribution's own quadrature (`polydisperse_yukawa_sas.discretize`), takes $\sigma=2R$, and treats the log-normal "Mean radius" as the mean, not the median (`_makeDistribution`). Change "Tk" to "Qt".

31. **L1435–1439 and L1628–1632 — must fix.** "The physicality test … It is applied unconditionally" and "Every route through this package screens $\min S(Q)$ and, where asked, deflates".
    `ozLib.solve` applies the $\min S(Q)\ge0$ screen **only when `deflateOnUnphysical>0`**. With the default 0, an unphysical root is returned with no error and no warning (`ozLib.solve` → `_deflateUntilPhysical` is called only `if deflateOnUnphysical`). The generic route always screens (`GenericPolydisperseSAS._verifyPhysical` raises), and so does the RY wrapper (`RYPolydisperseYukawa(checkPhysical=True)`).
    *Action:* state per route what happens. Either say that `ozLib.solve` needs `deflateOnUnphysical≥1` to screen, or flag this as a code inconsistency to fix.

32. **L1441–1452 — must fix (verification).** "the result returned only if the two agree … Plain Picard iteration … [is the verifier]; Biggs–Andrews … is the fallback when Picard fails."
    The default is now `verifyWith=("Biggs-Andrews", "Picard / Mann iteration")`, so **Biggs–Andrews comes first** (`ozLib.solve` comment "BIGGS-ANDREWS FIRST, PICARD AS FALLBACK"). Verifiers of the same algorithm family as the primary are skipped (`SOLVER_FAMILIES`). Only a *converged* verifier counts, and the Picard/Mann verifier is damped with `verifyMannAlpha=0.5`. If no verifier converges, `VerificationUnavailable` is raised; it carries `.result`, and the GUI shows the run as "[UNVERIFIED]". The result records `verifiedWith` and `verificationDifference`.
    *Action:* rewrite the paragraph accordingly.

#### Should update

33. **L97–99 — should update.** "Picard, the two Anderson variants, the two `scipy` solvers, Biggs–Andrews and both SUNDIALS KINSOL drivers".
    The current set is Picard/Mann, scipy Anderson, MDIIS, scipy Newton–Krylov, Biggs–Andrews, and the KINSOL NK and KIN_FP drivers. The hand-written Anderson is gone. *Action:* update the list.

34. **L757–770 — should update.** "`oZgui.py` gained a top-level notebook and an `_addExtraTabs()` hook … Each tab is a self-contained `ttk.Frame`".
    Now `gui/main_window.OZWidget` (a `QTabWidget`) builds tabs from `EXTRA_TABS`. Each tab is a `QWidget`; a failed import shows an error widget. The registry listing itself is still correct. *Action:* rename to main_window / OZWidget / QWidget.

35. **L1154–1158 — should update.** "the GUI now offers 19 of 23".
    `multicomponentCapableClosures()` returns **20 of 24**. RHNC, MHNC, RMSA and ZSEP are hidden. EuRah is not one of the 24. *Action:* change the numbers and the list. See also #28: CJVM and VM etc. are included.

36. **L887–892 and L957–961 — should update.** "selectable among Schulz, Gaussian, log-normal and Weibull" / "All four supported distributions".
    `polydisperse_nodes.DISTRIBUTIONS` and the generic tab dropdown now have five: Schulz, Gaussian, LogNormal, Weibull and **Gamma**. The supplement's own L1911 already says "All five". *Action:* add Gamma, and say which quadrature rule it uses (generalised Gauss–Laguerre).

37. **L944–948 — should update.** "Charge-coupled potentials are refused. DLVO, DLVO-Hydra, IonicMicrogel and the dedicated polydisperse Yukawa …".
    The generic builder (`setPolydispersePotential`) still refuses them. However, the generic tab now offers **`PolydisperseHardCoreYukawa`**, and `GenericPolydisperseSAS._buildAndSolve` dispatches it to its own setter, which handles $Z_i\propto\sigma_i^n$ and the global κ (`generic_polydisperse_tab._availablePotentials`). Only DLVO, DLVOHydra and IonicMicrogel stay out. *Action:* qualify the sentence.

38. **L1565–1571 — should update.** "It now tests the increment, and exposes a `converged` attribute".
    It tests the residual of the undamped map, $\|T(x)-x\|/\|x\|$; with damping this differs from the increment by the factor $a$. The limit is $\lceil N/a\rceil$ with N=5000 (`picardOZsolver.picardIteration`). `ozLib.solve` copies `converged` to `OZResult`, warns with `RuntimeWarning`, and the GUI marks "[NOT CONVERGED]". *Action:* update the wording.

39. **L1450, L1477, L1575, L1598 — should update (solver name).** "Plain Picard iteration", "`Picard iteration`". Canonical name: `Picard / Mann iteration`. *Action:* rename, as in Part I #14.

40. **L1475–1487 (table "Solver behaviour, measured") — should update.** The row "Anderson acceleration | 5/5 | correct, fastest" and "Anderson acceleration does not require Picard pre-conditioning".
    That solver is no longer offered. MDIIS replaced it and is not in the table. *Action:* label the row as a removed solver (historical), and optionally note MDIIS.

41. **L624 — should update.** "after `KINInit`, as pyozGUI does".
    Now `pysasfit.oz.sundials4pyKinsolFPOZsolver`, which still calls `KINSetMAA` after `KINInit`, so the statement is otherwise still true. *Action:* rename.

42. **L797, L1583, L1876–1877, L1897 (+ Part I L1228) — should update (missing scripts).** "`tools/gui_screenshots.py`", "`tools/residual_check.py`", "`tools/two_yukawa_reference.py`", "`tools/two_yukawa_test.py`", "`tools/numerics_test.py` … 27 checks", "`tools/rmsa_vs_sasmodels.py`".
    None of these exist in the pySASfit tree. `docs/oz/latex/tools/` holds only `bibitem_to_bibtex.py`. The closest equivalents are under `tests/oz/`: `test_numerics_options.py` with 19 tests, plus `test_multicomponent.py`, `test_sas_validation.py` and `sq_validation.py`.
    *Action:* re-point each reference to the pytest file that now covers it, or mark it as "pyOZgui repository" and keep the counts as historical.

#### Optional

43. **L76 (abstract) — optional.** "two new GUI tabs". There are now three extra tabs: "1: Polydisperse (any potential)", "2: RY Polydisperse Yukawa", "3: Mixture validation". *Action:* update the count.

44. **L603–608 — optional.** "KIN_FP is the best of the Python solvers on every count".
    On the Python RY route, `rypolydisperseWrapper._solverClasses()` (the wrapper's own fallback cascade) is scipy Anderson → MDIIS → Picard, with **no KIN_FP**. The RY tab's α search uses that wrapper default (scipy Anderson). KIN_FP is reached only through the tab's Solver dropdown, where it is the first entry. *Action:* add one sentence, or mark it as a known inconsistency.

45. **L1242–1248 (defect item 5) — optional.** "That Python tolerates it suggests the setting may simply be ignored there … worth checking". This was already checked and found unfounded (the supplement's own L622–629). *Action:* add "(checked: MAA is active, see §Solver strategy)".

46. **L1364–1367 (defect "Tk touched from a worker thread") — optional.** Historical. The GUI is now Qt, and the same rule is applied: controls are read on the main thread. *Action:* add "(pyOZgui Tk version)".

47. **Figure `gui_mixture_validation.png` (L781) — optional.** The screenshot shows a "Max iterations: 60000" field that `gui/mixture_validation_tab.py` no longer has (it shows Grid exponent k and Refinement factor). The caption also cites the missing `tools/gui_screenshots.py`. *Action:* regenerate the screenshot.

---

### Summary

| Class | Part I (documentation) | Part II (supplement) | Master | Total |
|---|---|---|---|---|
| must fix | 9 (#3–#11) | 5 (#28–#32) | 0 | **14** |
| should update | 14 (#12–#25) | 10 (#33–#42) | 0 | **24** |
| optional | 2 (#26–#27) | 5 (#43–#47) | 2 (#1–#2) | **9** |

Most important:
- Sign convention A: Yukawa, SquareWell and PiecewiseConstantHS formulas and examples (#5–#7, #19, #20).
- The transform default (DST-IV for one component) and the grid default $N=4095$ (#3, #4, #24).
- The Picard solver description and the removed hand-written Anderson / missing MDIIS (#8, #9, #12–#14).
- The Tk → Qt GUI and the GUI controls: no Max-iterations field; damping, verify and deflate controls (#10, #11, #22).
- Verification order and family skipping, the physicality screen not being unconditional in `ozLib.solve`, the multicomponent closure availability (20/24, split-reading closures now work), and sessions no longer storing runs (#28–#32, #35).
