# ozLib — manual and quick reference

`ozLib` is the programmatic entry point to the Ornstein–Zernike solver. The
GUI's Calculate button calls `ozLib.solve()`; anything the GUI can compute is
reachable from a script through the same function, which means a figure made
in the GUI can be reproduced in a batch job without reimplementing anything.

---

## Quick reference

```python
from pysasfit.oz import ozLib

res = ozLib.solve("HardSphere", phi=0.3)          # simplest possible call
q, S = res.q, res.Sq                              # both plain numpy arrays
```

| what | how |
|---|---|
| one solve | `ozLib.solve(potential, phi, ...)` → `OZResult` |
| agreement of several solvers | `ozLib.solveWithConsensus(...)` |
| which closures work for mixtures | `ozLib.multicomponentCapableClosures()` |
| is this potential a mixture one | `ozLib.isMulticomponentPotential(name)` |
| does this closure take a 2nd parameter | `ozLib.secondClosureParam(name)` |
| names of everything | `ozLib.CLOSURE_SETTERS`, `SOLVER_CLASSES`, `CURVE_NAMES` |

### `OZResult` attributes

Curves are plain arrays on the solver's own grid, **not** functions:

| real space (vs `res.r`) | reciprocal (vs `res.q`) |
|---|---|
| `gr` g(r), `cr` c(r), `hr` h(r) | `Sq` S(Q) |
| `gamma` γ(r), `Br` B(r), `yr` y(r) | |
| `Ur` βu(r), `fr` Mayer f(r) | |

Plus `potential`, `potentialArgs`, `phi`, `closure`, `closureParam`,
`solverName`, and `solverInstance` for the underlying solver object.

**Beware the three spellings.** The same two quantities are named differently
at each layer, and writing a call from memory is how an afternoon gets lost:

| layer | grid size | resolution | q and S(Q) |
|---|---|---|---|
| `ozLib.solve` | `numberOfRadialSamplingPoints` | `hardSphereDiameterInPoints` | returns `OZResult` |
| `OZResult` | — | — | `.q`, `.Sq` (arrays) |
| solver object | — | — | `getqArray()`, `getSq()` (methods) |
| `GenericPolydisperseSAS` | `gridN` | `pointsPerSigma` | `I_exact(Q)` |

---

## `solve()` in full

```python
ozLib.solve(potential, phi,
            potentialArgs=(),
            closure="Percus-Yevick",
            closureParam=None, closureParam2=None,
            findConsistentParameter=False,
            solver="scipy Anderson",
            maxIterations=None,
            numberOfRadialSamplingPoints=None,
            hardSphereDiameterInPoints=None,
            onSolverCreated=None,
            verify=True,
            verifyWith=("Biggs-Andrews", "Picard / Mann iteration"),
            verifyTolerance=1e-3,
            residualTolerance=1e-6,
            deflateOnUnphysical=0,
            transformType=None,
            mannAlpha=None,
            verifyMannAlpha=0.5)
```

`verifyWith` defaults to Biggs–Andrews, then Picard / Mann. The verifiers are tried in order. The solver being
verified is skipped, and so is any verifier of the **same algorithm family** (`ozLib.SOLVER_FAMILIES`). For
example, KIN_FP, scipy Anderson and MDIIS are all Anderson/DIIS acceleration and never verify each other. The
first verifier that *converges* is used, each with its own iteration limit from the preferences
(`maxIterations` applies to the primary only). The Picard / Mann verifier is damped with `verifyMannAlpha`
(default 0.5), because undamped Picard and Biggs–Andrews both fail for dense states (hard spheres,
φ ≥ 0.5).

If no verifier converges, `ozLib.VerificationUnavailable` (a `RuntimeError`) is raised. It carries the
unverified result as `.result` and what each verifier did as `.attempts`. A disagreement between converged
solvers raises `ValueError` as before. A verified result has `verifiedWith` and `verificationDifference`.

`mannAlpha` (0 < a ≤ 1, otherwise `ValueError`) is the damping of the Picard / Mann solver,
x ← (1−a)x + a·T(x). `None` or 1 means plain Picard. **The iteration limit of Picard / Mann is ⌈N/a⌉.** Here N
is `maxIterations`, or, when that is `None`, `PicardOZsolver.DEFAULT_MAX_ITERATIONS` (preference
`oz.solvers.max_iterations_picard`, factory 5000). N is therefore the limit for a = 1. A damped step covers
only the fraction a of the way, so the iterations needed grow as 1/a, and the limit grows with them.
`PicardOZsolver.mannIterationLimit(N, a)` returns the limit. `OZResult.iterationLimit` gives the limit that
applied, and `OZResult.iterations` the iterations used (Picard / Mann). `mannAlpha` applies to the primary solve and a deflation retry. It does not apply to the
verifier, which uses `verifyMannAlpha`. All other solvers ignore it. The value used is stored as `OZResult.mannAlpha`
(`None` for the other solvers). The former solver name `"Picard iteration"` is still accepted
(`ozLib.SOLVER_ALIASES`).

**`OZResult.converged`** tells whether the solve met its convergence criterion, and
`OZResult.finalRelativeProgress` gives the last relative change. A solve that stops at the iteration limit
still returns its last iterate, but it now also issues a `RuntimeWarning` ("… did not converge; the result is
the last iterate, not a solution"). Check the flag in scripts. For hard spheres (PY, φ = 0.53), Picard / Mann
with a fixed limit of 1000 iterations returned g_max between 4.35 and 6.77 depending on a (the limit is now
N/a), while every converged run gives
5.556220 (see [the GUI guide](gui.md)). RMSA, ZSEP and the consistent-parameter search end with an internal
Picard solve; the flag now reports that final solve. Before, it was always False for them.

```python
```

**`maxIterations=None` means the solver's own limit, not "unlimited".** Each
solver carries a `DEFAULT_MAX_ITERATIONS`: **600** for the accelerated
family, **5000** for Picard (3000 before Picard / Mann became the damped
verifier; see below). The difference is justified by measurement — a
solve that converges takes 17–60 iterations under SUNDIALS KIN_FP, 18–179
under scipy Anderson and 30–97 under MDIIS, against 336 to several thousand
for Picard.

Passing a flat number to every solver, which this used to do, gives the
accelerated ones ten to a hundred times what they need. That costs nothing
on a solve that converges — it stops at its tolerance long before the ceiling
— but a great deal on one that does not: at an unphysical or near-spinodal
state point the solver grinds through thousands of iterations before
admitting defeat, and during a global search those are exactly the
evaluations that dominate. Pass a number only if you mean to override the
solver's judgement.

**`potentialArgs` is positional and its meaning is per-potential.** There is
no keyword form. `(tau, delta)` for StickyHardSphere, `(K1, z1, K2, z2)` for
HardSphereDoubleYukawa — see the table below, or call
`getAvailablePotentialNames()` on a solver instance for the live list.

**`verify=True` is the default and should stay that way.** It re-solves with
two other solvers and raises if they land on different fixed points. That is
not paranoia: the closure equations genuinely admit several solutions, and
a solver reporting success is not saying it found the *physical* one. At a
Lennard-Jones state tested during development, every Newton–Krylov variant
converged to residuals below 1e-11 on branches with min S(Q) ≈ −38 — real
roots, entirely unphysical. Turn verification off for speed only once you
know the state point is well-behaved.

**`findConsistentParameter=True`** solves the closure's own α by
thermodynamic consistency (compressibility route against virial) instead of
taking `closureParam`. It costs three OZ solves per trial value, so a single
Calculate becomes dozens of solves.

---

## Potentials (21)

All lengths are in units of the hard-core diameter σ; all energies in k_BT.

| name | arguments |
|---|---|
| `HardSphere` | — |
| `SquareWell` | `(epsilon, delta)` |
| `StickyHardSphere` | `(tau, delta)` |
| `Yukawa` | `(shieldingLength, strength, doAddHardSphere)` |
| `HardSphereDoubleYukawa` | `(K1, z1, K2, z2)` |
| `HS3Yukawa` | `(K1, l1, K2, l2, K3, l3)` |
| `LennardJones` | `(epsilon)` |
| `SoftSphere` | `(epsilon, n)` |
| `DLVO` | `(kappa, Z, LB, A)` |
| `DLVOHydra` | `(kappa, Z, LB, A, GHY, DH)` |
| `Depletion` | `(sigmaRatio, phi2)` |
| `FermiDistribution` | `(epsilon, xi)` |
| `GGCMn` | `(epsilon, n, alpha)` |
| `IonicMicrogel` | `(Z, ED, KPi, EPSILON)` |
| `PSM` | `(epsilon)` |
| `ParabolicSphere` | `(epsilon)` |
| `PiecewiseConstantHS` | `(eps1, d1, eps2, d2, eps3, d3)` |
| `StarPolymerHighF` | `(NumberOfArms)` |
| `StarPolymerLowF` | `(NumberOfArms)` |
| `Polydisperse` | `(potentialName, potentialArgs, srel, nComponents, ...)` |
| `PolydisperseHardCoreYukawa` | `(srel, nComponents, valence, bjerrum, ...)` |

**Sign convention (project-wide, "convention A"): positive = attractive.**
Every well depth and Yukawa amplitude in pySASfit is positive for attraction:
`SquareWell` ε and `PiecewiseConstantHS` ε_i are depths (βu = −ε, ε > 0 a well,
ε < 0 a shoulder), and the Yukawa amplitudes of `Yukawa`, `HardSphereDoubleYukawa`,
`HS3Yukawa` and the polydisperse MSA/RY Yukawa code are K > 0 attractive, K < 0
repulsive. This matches the SASfit structure-factor plugins and the literature.
pyOZgui, and SASfit's C OZ solver (`sasfit_oz_potential_square_well.c`,
`..._piecewise_constant_HS.c`), use the opposite sign for `SquareWell`,
`PiecewiseConstantHS` and pyOZgui's single `Yukawa`; session files written
before the change are converted on load.

**`SquareWell` and `StickyHardSphere` take δ as an absolute length**, not as
a fraction of σ, so it is directly comparable with a shell thickness.

---

## Closures (24)

`ozLib.CLOSURE_SETTERS` maps each name to its setter and whether it takes a
parameter. `ozLib.multicomponentCapableClosures()` filters to those valid for
mixtures.

**Parameter-free**: Percus-Yevick, Hypernetted-Chain, MSA, Modified MSA,
Symmetric MSA, Rescaled MSA, Verlet, Martynov-Sarkisov, Duh-Haymet,
Kovalenko-Hirata, Choudhury-Ghosh, Vompe-Martynov, Reference HNC, ZSEP.

**Take an α, settable directly or by consistency** (`ozLib.CONSISTENT_
PARAMETER_CLOSURES`): Rogers-Young, Extended Rogers-Young, HMSA, Modified
HNC, Modified Verlet, BPGG, CJVM, BB, Khanpour, Carbajal-Tinoco.

**Takes a second parameter**: Extended Rogers-Young, `(a, 0.0, "quadratic
coefficient; a = 0 gives plain RY")` — see `ozLib.secondClosureParam`.

---

## Solvers

Nine with SUNDIALS installed, five without. Timings are from
`tools/benchmark.py` on a 4095-point grid; the ranking is what
`ozLib.SOLVER_CLASSES` lists, fastest first.

| name | family | seconds | note |
|---|---|---|---|
| `sundials4py: Fixed-Point (Anderson)` | fixed point | **0.0049** | the default when built |
| `scipy Anderson` | fixed point | 0.0077 | the default without SUNDIALS |
| `MDIIS` | fixed point | 0.0096 | see below |
| `sundials4py: Newton-Krylov (FGMRES)` | Newton–Krylov | 0.0100 | |
| `sundials4py: Newton-Krylov (GMRES)` | Newton–Krylov | 0.0108 | |
| `sundials4py: Newton-Krylov (TFQMR)` | Newton–Krylov | 0.0187 | |
| `scipy Newton-Krylov` | Newton–Krylov | 0.0232 | **see the warning below** |
| `Biggs-Andrews` | fixed point | 0.0402 | |
| `Picard / Mann iteration` | fixed point | 0.0493 | plainest; undamped (a = 1) **diverges above φ ≈ 0.42** |

**Picard's limit is 0.42, not the 0.45 this table used to say** — measured on
a 1023-point grid, where the iteration count grows geometrically (41, 49, 60,
… 946 in steps of Δφ = 0.02) and then fails outright with a residual of 1.1.
And **a density ramp does not rescue it**: the contraction factor exceeds 1
there, so the basin has not moved but ceased to exist, and no starting point
inside it helps. Continuation methods fail for the same reason.

**`Anderson acceleration` was removed.** It duplicated `scipy Anderson` —
the same algorithm, hand-written rather than from scipy — and two entries for
one method cost a choice nobody has a basis to make. The module remains
importable; it is simply not registered.

**`MDIIS` is a fallback, not a recommendation.** Third of nine here, and it
fails at φ = 0.58 where KIN_FP and scipy Anderson both converge. Its use is
on installations without SUNDIALS, where scipy Anderson's iteration count
swings erratically with density (28, 116, 40, 179) while MDIIS rises
smoothly — at φ = 0.52 it is the better of the two. Try it when scipy
Anderson stalls.

**Prefer a fixed-point method.** Near a fold the Jacobian is singular, and
the Newton–Krylov family is drawn onto negative-compressibility branches
exactly where the fold lies. Measured at one Lennard-Jones state: the
fixed-point family found g_max = 2.16 with min S(Q) = +0.21; the
Newton–Krylov family found 1.13 and 1.22 with min S(Q) ≈ −38. All had
residuals below 1e-11. They are all correct answers to the question asked;
only one is physical.

**Picard damping (Mann iteration).** The Picard / Mann solver iterates
x ← (1−a)x + a·T(x). a = 1 is plain Picard. Below 1 it is Mann's iteration,
which is slower (about 1/a times more iterations) but converges where undamped
Picard diverges. For hard spheres (PY, 1023 points), a = 0.7 converges at
φ = 0.45 in 2311 iterations, a = 0.5 at φ = 0.50 in 5776, and a = 0.3 at
φ = 0.55 in 17917. Undamped Picard fails at φ = 0.45 even after 20000
iterations. It also rescued a Lennard-Jones case at a = 0.5. Pass it as
`solve(..., mannAlpha=0.5)`; setting `sol.mannAlpha` in `onSolverCreated`
still works. The iteration limit is then ⌈N/a⌉ (see `mannAlpha` above). Convergence is tested on the residual ‖T(x) − x‖/‖x‖, not on the
damped step, so a small a does not stop early. The full table is in
[the GUI guide](gui.md).

**Every solver exposes `converged`**, declared in `OZsolver.__init__` and
defaulting to `False` — so a solver that forgets to set it reports failure
rather than a success it never established. Until recently only Picard set
it, and the rest returned `None`, which a caller cannot tell apart from
"ran and did not converge".

**But it is not a correctness check**, and the distinction matters more here
than in most numerical work. On one Lennard-Jones state point all four
Newton–Krylov solvers report `converged = True`, are genuine fixed points,
and reach residuals of 10⁻¹³ — at **min S(Q) ≈ −38**. Every question of the
form "did it converge?" answers yes; only the `min S(Q) ≥ 0` screen
separates them from the correct g_max = 2.1636.

So `ozLib.solve()` checks both, on every solve: min S(Q) ≥ 0 (below) and the residual.

### The independent residual check (`residualTolerance`)

A solve looks for x with T(x) = x, where T is the closure's fixed-point map (x = Γ = h − c for one component,
the packed pair matrix for a mixture). After the solver has finished, T is applied **once more** to the
returned x, and

  r = ‖T(x) − x‖ / ‖x‖

is stored as `OZResult.residual`. If r > `residualTolerance` (default 10⁻⁶; preference
`oz.solve.residual_tolerance`), the result is marked **not converged**: `OZResult.converged = False`, a
`RuntimeWarning`, and "[NOT CONVERGED]" in the OZ solver tab. This applies whatever the solver reported, so a
solver's flag can be overruled but never upgraded. `residualTolerance=None` switches the check off.

Why the check matters: the solvers decide convergence by their own internal tests, and those can be fooled.
Examples:

- The old Picard test compared successive norms and accepted a "solution" with residual 19.4.
- A damped step is shorter than the residual.
- SUNDIALS KIN_FP reports success for polydisperse hard-core Yukawa (HNC, φ = 0.1) and returns S(Q) = NaN; its
  residual is infinite.

Converged solves give r = 10⁻¹⁶ … 10⁻¹¹ (all 24 closures checked on hard spheres, Lennard-Jones and a square
well), so 10⁻⁶ rejects only real failures.

The residual cannot tell a physical root from an unphysical one. The Newton–Krylov roots above have
r ≈ 10⁻¹³ and min S(Q) ≈ −38. The three checks answer different questions:

| check | question |
|---|---|
| residual | is the returned x a fixed point? |
| min S(Q) ≥ 0 | is this fixed point a fluid? |
| verification | does an independent solver find the same one? |

### The physicality screen: min S(Q) ≥ 0 on every solve

S(Q) is a variance and cannot be negative, but no convergence test sees that. `ozLib.solve()` therefore checks
min S(Q) on **every** solve, whatever `deflateOnUnphysical` is:

- `OZResult.minSq` holds the minimum of S(Q), and `OZResult.physical` is `minSq >= 0`.
- With `deflateOnUnphysical = 0` (the default), an unphysical root is returned with a `RuntimeWarning`. The OZ
  solver tab marks the run "[S(Q)<0]" and shows a warning.
- With `deflateOnUnphysical > 0` the root is rejected and the same solver is retried with it deflated away (next
  section). If no physical root is found, a `ValueError` is raised.

Example: Lennard-Jones ε = 0.8, φ = 0.30, HNC, 1023 points. SUNDIALS KIN_FP finds the physical root
(min S(Q) = 0.217). scipy Newton–Krylov (min S = −9.3), SUNDIALS Newton–Krylov (−10.8) and scipy Anderson
(−13.7) all report convergence on unphysical roots. scipy Anderson is the default solver when SUNDIALS is not
installed. Before this change, the screen ran only when deflation was on, so these roots came back without notice.

### Deflation: what to do when the root is unphysical

The screen above tells you the solver landed on a branch that is not a fluid.
`deflateOnUnphysical` does something about it:

```python
r = ozLib.solve("LennardJones", potentialArgs=(0.8,), phi=0.30,
                closure="Hypernetted-Chain",
                solver="scipy Newton-Krylov",
                deflateOnUnphysical=2)      # 0 = off, the default
```

Each attempt divides the residual by a factor singular at every root already
found, so the solver **cannot** return there and must find something else or
fail. The same solver is kept throughout.

**Why not simply switch solver?** That would often work — the fixed-point and
Newton–Krylov families empirically favour different branches, and at the
state point above a fixed-point solver finds the correct root directly. Two
reasons against. It silently hands back a result from a solver you did not
choose. And it leans on a bias that happens to hold where it has been tested
rather than on anything guaranteed: nothing stops both families landing on
the same wrong root elsewhere.

**Failure is informative, not a defect.** Past a spinodal there may be no
physical solution at the state point, and deflation will go on finding
unphysical ones because that is all there is. The exception therefore lists
every root found with its min S(Q) rather than reporting a bare
non-convergence — those numbers say something about the state point. At
ε = 0.8, φ = 0.30 under HNC with Newton–Krylov you get roots at −18.73 and
−20.69 and then a refusal, which is the honest answer.

**In the GUI** every tab has a *Deflate if S(Q)<0* control, defaulting to 0.
On the polydisperse tab it applies to **Compute only, never to Fit**: the
optimiser calls the solver hundreds of times and wanders into unphysical
territory routinely, and rescuing some of those points and not others would
make chi-squared jump between neighbouring parameter values. Least squares
assumes a smooth residual. The refusal is also a signal to the optimiser to
back off, and hiding it helps nobody.

The workflow it is actually for runs the other way: fit, get a suspicious
answer, then recompute at those parameters with deflation on to ask whether a
second branch exists there.

**A caution on reading too much into a second root.** If deflation finds one
while S(Q) ≥ 0 — which it will do if you ask it to search rather than only to
rescue — resist the temptation to pick between them on thermodynamic
consistency. For a parameter-free closure the compressibility/virial mismatch
is a property of the closure, not of the branch, and PY hard spheres are
inconsistent by a large and known amount at every density. For the
consistency-parameter closures it is circular: α is tuned precisely to make
the two routes agree, so two roots give two values of α each achieving it.
Continuity from a known-good state point, or agreement with an analytic limit
where one exists, discriminates where consistency cannot.

---

## Grids and transforms

**N and the resolution are not independent.** The real-space range is

    r_max = N · σ / pointsPerSigma

so raising the resolution at fixed N *shrinks the box*. Refining that way
makes the result better near contact and worse at low Q, and a convergence
test done that way measures nothing. Scale both together. This mistake has
been made twice in this project and caught both times only by a plot.

**Every precision claim in this documentation was checked against that rule.**
The transform timings and accuracies, the first-order convergence of type 1,
the stability of the low-q extrapolation: all compare grids at
r_max = 40.95–40.96 σ, varying only the density. The one comparison that does
vary the range — 41 σ to 655 σ at fixed resolution, which changed S(0) in the
sixth decimal — is described as such, and it is what established that the
type-1 error is a grid-alignment effect rather than truncation.

When writing a new comparison, pair `bestGridSize(N, transformType)` with a
proportionally scaled `pointsPerSigma` and the range stays fixed without
anyone having to remember it. Print `N/pointsPerSigma` if in doubt: it takes
a second and it is the check that would have caught both earlier mistakes.

**Grid size depends on the transform type**, and the difference is large:

| N | DST-I (type 1) | DST-IV (type 4) |
|---|---|---|
| 16383 = 2¹⁴−1 | **0.263 ms** | 0.308 ms |
| 16384 = 2¹⁴ | 0.599 ms | **0.099 ms** |
| 16385 | 2.497 ms | 0.303 ms |

A DST-I goes through an FFT of length 2(N+1), so it wants N = 2^k − 1; a
DST-IV is a length-N transform and wants N = 2^k. Use
`oZfixpointOperator.bestGridSize(n, transformType)` rather than choosing by
hand. Sizing both for type 1 made type 4 benchmark *slower* than type 1 for
some time, which is how the rule was found.

**Type 4 is second order** — measured 139× to 1092× more accurate than type 1
at the same grid, and faster on a power-of-two grid. It is **correct only for
one size class**: it requires every pair core to fall between grid points,
which cannot be arranged for a mixture because the σ_ij are irrational.

**How much that costs, measured absolutely.** Percus–Yevick hard spheres have
a closed form for the compressibility, S(0) = (1−φ)⁴/(1+2φ)², which is the
only exact answer available anywhere in this package. Against it:

| φ | type 1, pps=100 | type 1, pps=400 | type 4, pps=100 |
|---|---|---|---|
| 0.30 | 3.67% | 0.92% | **0.015%** |
| 0.40 | 5.27% | 1.33% | **0.036%** |

The cause is where the discontinuity lands. A DST-I places nodes at (n+1)Δr,
so the hard core at r = σ falls **on** a node and is carried by a point that
is neither inside nor outside; the excluded volume is then wrong at first
order. A DST-IV places them at (n+½)Δr, the core falls **between** points,
and the step is resolved to second order.

Two things make this worth knowing rather than filing under "type 4 is
better". The error is **independent of the real-space range** — extending
r_max from 41σ to 655σ changed S(0) in the sixth decimal — so the usual
instinct of giving the solver more room does nothing. And refining the
resolution *does* reduce it, by a clean factor of 3.96 per fourfold step, so
a convergence study sees orderly first-order behaviour and concludes the
method is working. It is, slowly, towards an answer whose leading error is
set by grid alignment.

S(0) is the isothermal compressibility, so this is a systematic bias in
exactly the quantity the volume fraction controls: **a fit absorbs it by
moving φ**, and the denser the sample the more it moves. For one component,
use type 4. For a mixture there is no such option, and a dense polydisperse
fit should use the finest affordable resolution with its fitted volume
fraction read accordingly.

`tools/numerics_test.py` asserts all of this, and
`tools/validation_table_test.py` reports it per transform and per resolution.

---

## `solveWithConsensus()`

```python
ozLib.solveWithConsensus(potential, potentialArgs=(),
                         closure="doPYclosure", closureParam=None,
                         volumeDensity=0.3,
                         solvers=("scipy Anderson", "Biggs-Andrews",
                                  "Picard / Mann iteration"),
                         tolerance=1e-3,
                         gridN=4095, pointsPerSigma=100,
                         maxIterations=None, transformType=1,
                         quantity="Sq")
```

Runs several solvers and reports whether they agree. Use it when a state
point is suspect, or when a result matters enough to want more than one
route to it. Note it spells its arguments `gridN`/`pointsPerSigma`, unlike
`solve()`.

---

## Worked examples

### A single curve

```python
from pysasfit.oz import ozLib
res = ozLib.solve("StickyHardSphere", phi=0.3, potentialArgs=(0.2, 0.02))
q, S = res.q, res.Sq
```

### An attractive square well, at a chosen resolution

```python
from oZfixpointOperator import bestGridSize
res = ozLib.solve("SquareWell", phi=0.3,
                  potentialArgs=(1.0, 0.1),       # POSITIVE = attractive (convention A)
                  closure="Hypernetted-Chain",
                  numberOfRadialSamplingPoints=bestGridSize(16383, 1),
                  hardSphereDiameterInPoints=400)
```

### Solving α by thermodynamic consistency

```python
res = ozLib.solve("HardSphere", phi=0.4, closure="Rogers-Young",
                  findConsistentParameter=True)
print(res.closureParam)          # the α that made the two routes agree
```

### Checking convergence properly

```python
from oZfixpointOperator import bestGridSize
for pps, k in ((100, 12), (400, 14)):        # r_max held CONSTANT
    res = ozLib.solve("HardSphere", phi=0.3,
                      numberOfRadialSamplingPoints=bestGridSize(2**k - 1, 1),
                      hardSphereDiameterInPoints=pps)
```

A fourfold refinement should reduce a first-order error fourfold. A ratio
near 1 means the discrepancy is not discretisation, and something else is
wrong — that is how a defect giving every pair in a mixture the same hard
core was eventually found.

### Damped Picard for a hard state point

```python
res = ozLib.solve("HardSphere", phi=0.5, solver="Picard / Mann iteration",
                  mannAlpha=0.5, maxIterations=5000)     # limit 5000/0.5 = 10000
print(res.mannAlpha, res.converged, res.iterationLimit)   # 0.5 True 10000
```

---

## Things that will bite

| symptom | cause |
|---|---|
| a converged fit with min S(Q) < 0 | a real root on an unphysical branch — screen on S(Q) ≥ 0 |
| `TypeError` on unexpected keyword | `gridN` vs `numberOfRadialSamplingPoints`; the layers differ |
| a "square well" that repels | negative ε is a shoulder; a pyOZgui-era script that used ε < 0 for a well needs the sign flipped |
| error grows under refinement | N held fixed while resolution rose; the box shrank |
| type 4 slower than type 1 | grid sized 2^k−1 instead of 2^k |
| solver reports success, answer is wrong | KINSOL can report success after diverging; keep `verify=True` |

---

## Where to look next

- `tools/two_yukawa_test.py` — a worked validation against a published
  reference, showing the convergence-ratio pattern
- `tools/gui_path_test.py` — exercises the GUI's own compute and fit paths
- `docs/ozGUI_ozLib_report.tex` — the full report, including the derivations
- `oZfixpointOperator.py` — the transform, the grid rules, `bestGridSize`
