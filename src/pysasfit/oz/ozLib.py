# -*- coding: utf-8 -*-
'''
Library API of the Ornstein-Zernike solvers: the same calculation the OZ
solver tab's "calculate" button runs (the tab calls solve() from here), so a
result from the GUI can be reproduced from a script::

    from pysasfit.oz import ozLib
    result = ozLib.solve("HardSphere", phi=0.3, closure="Percus-Yevick")
    print(result.converged, result.physical, result.residual)
    q, S = result.q, result.Sq

SOLVER_CLASSES and CLOSURE_SETTERS below hold the valid ``solver=`` and
``closure=`` names (``print(list(ozLib.SOLVER_CLASSES))``); SOLVER_ALIASES maps
former names, SOLVER_FAMILIES the algorithm families used for verification.

Every solve is checked three ways (see solve()): the residual of the returned
solution (``residualTolerance``), min S(Q) >= 0, and -- unless verify=False --
an independent solver of another algorithm family.

The radial grid (numberOfRadialSamplingPoints, hardSphereDiameterInPoints) is
passed straight through to the solver class; see oZfixpointOperator for why
long-range potentials may need a larger grid.
'''
import warnings

import numpy as np

from .oZsolver import OZsolver
from .picardOZsolver import PicardOZsolver
#andersonOZsolver is no longer imported: "Anderson acceleration" was
#deregistered below as a duplicate of "scipy Anderson", and the module has
#been moved to _obsolete/. Restoring it means restoring the file, this
#import, and the SOLVER_CLASSES entry together.
from .scipyAndersonOZsolver import ScipyAndersonOZsolver
from .scipyNewtonKrylovOZsolver import ScipyNewtonKrylovOZsolver
from .biggsAndrewsOZsolver import BiggsAndrewsOZsolver
from .mdiisOZsolver import MDIISOZsolver
try:
    from .sundials4pyKinsolOZsolver import Sundials4pyKinsolOZsolver, AVAILABLE_LINEAR_SOLVERS
    from .sundials4pyKinsolFPOZsolver import Sundials4pyKinsolFPOZsolver
    _HAVE_SUNDIALS4PY = True
except ImportError:
    _HAVE_SUNDIALS4PY = False


# Each value is (solverClass, linearSolverNameOrNone) -- the second
# element, when not None, is assigned to solverInstance.linearSolver
# right after construction (see Sundials4pyKinsolOZsolver.py's own
# AVAILABLE_LINEAR_SOLVERS for why this is a post-construction
# attribute, not a constructor argument).
_BASE_SOLVER_CLASSES = {
    #PICARD / MANN: x <- (1-a) x + a T(x). a = 1 (default) is plain Picard;
    #a < 1 is Mann's damped iteration. a is the solve(mannAlpha=...) argument.
    #Called "Picard iteration" until this name was introduced; the old name
    #is still accepted (SOLVER_ALIASES), so old scripts and saved runs work.
    "Picard / Mann iteration": (PicardOZsolver, None),
    #"Anderson acceleration" (andersonOZsolver.AndersonOZsolver) is
    #DELIBERATELY NOT REGISTERED. The module is still present and importable;
    #it is simply not offered, because it duplicates "scipy Anderson" in
    #substance -- the same algorithm, one hand-written here and one from
    #scipy -- and two entries for one method cost the user a choice they
    #have no basis to make.
    #
    #IN FAIRNESS TO THE ONE REMOVED: measured at N = 1023 it was the FASTER
    #of the two at high density (0.009 s against scipy's 0.013 s at
    #phi = 0.52, 0.006 against 0.008 at 0.45), so this is a maintenance
    #decision rather than a performance one. Single runs, so the margin may
    #be noise; re-measure before reversing it.
    "scipy Anderson": (ScipyAndersonOZsolver, None),
    #MDIIS is offered in its place. It is not a new class of method --
    #Anderson mixing and Pulay's DIIS are algebraically the same thing for
    #this problem -- but it adds explicit regularisation and history-depth
    #control, and is the standard accelerator in the 3D-RISM literature.
    #
    #NOT A SPEED RECOMMENDATION. Benchmarked on the real grid (N = 4095,
    #SUNDIALS present) it is third of nine at 0.0096 s, behind KIN_FP at
    #0.0049 and scipy Anderson at 0.0077, and it FAILS at phi = 0.58 where
    #both of those converge. An earlier measurement in a sandbox at N = 1023
    #with SUNDIALS absent put it ahead of scipy Anderson; that was wrong, and
    #wrong in the flattering direction, because the small grid hid the cost
    #of its Python-level (m+1)x(m+1) solve per step and the fastest solver
    #was not in the comparison at all.
    #
    #Its place is the SUNDIALS-FREE fallback, which is the configuration this
    #module drops into when _HAVE_SUNDIALS4PY is false -- anyone installing
    #from PyPI without building SUNDIALS. There the choice is between this
    #and scipy Anderson, whose iteration count swings erratically with
    #density (28, 116, 40, 179 at phi = 0.45, 0.52, 0.55, 0.58) where MDIIS
    #rises smoothly; at phi = 0.52 it is the better of the two. Offer it as
    #the alternative to try when scipy Anderson stalls, and nothing more.
    "MDIIS": (MDIISOZsolver, None),
    "scipy Newton-Krylov": (ScipyNewtonKrylovOZsolver, None),
    "Biggs-Andrews": (BiggsAndrewsOZsolver, None),
}
if _HAVE_SUNDIALS4PY:
    _BASE_SOLVER_CLASSES["sundials4py: Newton-Krylov (GMRES)"] = (Sundials4pyKinsolOZsolver, "SUNLinSol_SPGMR")
    _BASE_SOLVER_CLASSES["sundials4py: Newton-Krylov (FGMRES)"] = (Sundials4pyKinsolOZsolver, "SUNLinSol_SPFGMR")
    _BASE_SOLVER_CLASSES["sundials4py: Newton-Krylov (TFQMR)"] = (Sundials4pyKinsolOZsolver, "SUNLinSol_SPTFQMR")
    _BASE_SOLVER_CLASSES["sundials4py: Fixed-Point (Anderson)"] = (Sundials4pyKinsolFPOZsolver, None)

_defaultSolverName = ("sundials4py: Fixed-Point (Anderson)" if _HAVE_SUNDIALS4PY
                      else "scipy Anderson")
# Reordered so the FIRST key is this project's recommended DEFAULT
# solver -- oZgui.py's own dropdown uses next(iter(SOLVER_CLASSES)) as
# its initial pre-selected value. SUNDIALS' own Anderson-accelerated
# fixed-point strategy (KIN_FP) when sundials4py is installed --
# SUNDIALS' production-grade implementation of the same acceleration
# idea as this project's own hand-written "Anderson acceleration"/
# "scipy Anderson" entries, generally more robust and faster to
# converge for the harder closures (HMSA/RY/BPGG/CJVM/BB) than plain
# unaccelerated Picard iteration, which is why it replaces Picard as
# the default here. Falls back to scipy's own Anderson acceleration
# (scipy.optimize.anderson-based, no extra dependency) when
# sundials4py isn't installed, rather than falling all the way back to
# plain Picard -- Picard is still fully present in the dict, just no
# longer pre-selected. Every other key keeps its original relative
# order from _BASE_SOLVER_CLASSES above; only the chosen default moves
# to the front. (solve()'s own solver= keyword argument default below
# is set to this same _defaultSolverName, not left hardcoded to
# "Picard / Mann iteration" -- so a script calling ozLib.solve() directly,
# without going through the GUI at all, gets the same improved default
# rather than silently falling back to unaccelerated Picard.)
SOLVER_CLASSES = {_defaultSolverName: _BASE_SOLVER_CLASSES.pop(_defaultSolverName)}
SOLVER_CLASSES.update(_BASE_SOLVER_CLASSES)

# Former solver names, still accepted everywhere a solver name is (solve(),
# solveWithConsensus(), verifyWith, the GUI). Not in SOLVER_CLASSES itself so
# that the dropdowns do not show the same solver twice.
SOLVER_ALIASES = {"Picard iteration": "Picard / Mann iteration"}


def resolveSolverName(name):
    """Current solver name for ``name`` (maps the SOLVER_ALIASES)."""
    return SOLVER_ALIASES.get(name, name)


# ALGORITHM FAMILIES, for the independence of verification: a solver never
# verifies one of its own family. Anderson mixing (1965) and Pulay's DIIS
# (1980) are algebraically equivalent, so SUNDIALS KIN_FP, scipy Anderson
# and MDIIS form one family. Unlisted names are their own family.
SOLVER_FAMILIES = {
    "sundials4py: Fixed-Point (Anderson)": "Anderson/DIIS",
    "scipy Anderson": "Anderson/DIIS",
    "MDIIS": "Anderson/DIIS",
    "Biggs-Andrews": "vector extrapolation",
    "Picard / Mann iteration": "Picard/Mann",
    "scipy Newton-Krylov": "Newton-Krylov",
    "sundials4py: Newton-Krylov (GMRES)": "Newton-Krylov",
    "sundials4py: Newton-Krylov (FGMRES)": "Newton-Krylov",
    "sundials4py: Newton-Krylov (TFQMR)": "Newton-Krylov",
}


def solverFamily(name):
    """Algorithm family of a solver name (see SOLVER_FAMILIES)."""
    name = resolveSolverName(name)
    return SOLVER_FAMILIES.get(name, name)


class VerificationUnavailable(RuntimeError):
    """No independent verifier converged. Not a disagreement: ``.result`` is
    the (unverified) result of the primary solve, ``.attempts`` lists what
    each verifier did."""

    def __init__(self, message, result=None, attempts=()):
        super().__init__(message)
        self.result = result
        self.attempts = list(attempts)


def isMannSolver(name):
    """True when ``name`` is the Picard / Mann solver (the only one that uses
    a damping value a)."""
    entry = SOLVER_CLASSES.get(resolveSolverName(name))
    return entry is not None and issubclass(entry[0], PicardOZsolver)

# Each value is (closure-setter method name, needsExtraScalarParam).
# Closures that take one extra scalar parameter share self.alpha's
# reused role across RY/HMSA/BPGG/CJVM/BB, and eta for MHNC -- see
# oZfixpointOperator.py's doXXXclosure() setters. Three entries are
# "special" (marked below): each needs its own orchestration method on
# OZsolver (solveRHNC()/solveRMSA()/solveEuRah()) instead of a plain
# doXXXclosure() + solve() call, since each needs a reference-system
# sub-solve first -- see oZsolver.py for exactly what each does.
CLOSURE_SETTERS = {
    "Percus-Yevick":        ("doPYclosure",    False),
    "Hypernetted-Chain":    ("doHNCclosure",   False),
    "Reference HNC":        ("doRHNCclosure",  False),   # special: solveRHNC()
    "Modified HNC":         ("doMHNCclosure",  True),
    "MSA":                  ("doMSAclosure",   False),
    "Rescaled MSA":         ("doRMSAclosure",  False),   # special: solveRMSA()
    "Modified MSA":         ("doMMSAclosure",  False),
    "Symmetric MSA":        ("doSMSAclosure",  False),
    "Rogers-Young":         ("doRYclosure",    True),
    "HMSA":                 ("doHMSAclosure",  True),
    "Verlet":                ("doVerletClosure", False),
    "Martynov-Sarkisov":    ("doMSclosure",    False),
    "Vompe-Martynov":       ("doVMclosure",    False),
    "BPGG":                  ("doBPGGclosure",  True),
    "CJVM":                  ("doCJVMclosure",  True),
    "BB":                    ("doBBclosure",    True),
    # Euler-Rahman ("Euler-Rahman"/EuRah, ported from sasfit_oz_solver.c's
    # own CEURAH machinery) is DELIBERATELY NOT offered here. Extensive
    # testing (see the conversation this decision was made in) found a
    # genuine exponential positive-feedback instability intrinsic to the
    # closure's own mathematical structure (c(r) inside the hard core
    # feeds into an exp(Gamma) term that dominates the very next c(r)
    # estimate), which defeated every numerical strategy tried --
    # Newton-Krylov, damped Picard, density continuation, dense-Jacobian
    # Newton, Levenberg-Marquardt, Broyden, df-sane, and SUNDIALS' own
    # KINSOL with this project's best-tuned settings. The underlying
    # solveEuRah()/doEuRahClosure() methods still exist on OZsolver for
    # anyone who wants to experiment with it directly, but it is
    # excluded from this dict (and therefore from both the GUI dropdown
    # and ozLib.solve()'s own closure= validation) since it does not
    # reliably converge for ordinary use.
    # Three closures found missing when checking SASfit's own
    # src/sasfit_oz/include/sasfit_oz.h enum against this list directly
    # (rather than assuming this project's existing 17 were complete) --
    # see oZfixpointOperator.py's own doKHclosure()/doDHclosure()/
    # doCGclosure() and their update_c() branches for the full story on
    # each, including a documented quirk (KH's own c(r) formula in
    # SASfit's C code is identical to HNC's, and a known, deliberate gap
    # (DH's Lennard-Jones-specific special case is not replicated).
    "Kovalenko-Hirata":      ("doKHclosure",    False),
    "Duh-Haymet":            ("doDHclosure",    False),
    "Choudhury-Ghosh":       ("doCGclosure",    False),
    #Added after cross-validating against OrnsteinZernike.jl (Closures.jl).
    #All three are standalone bridge functions of Gamma, so they inherit the
    #multicomponent path, every solver and this dropdown automatically.
    #"Khanpour" is spelled out because "KH" above is Kovalenko-Hirata --
    #different closure, same natural abbreviation.
    "Khanpour":              ("doKhanpourClosure", True),
    #Implicit bridge function (solved by fixed-point iteration inside
    #update_c); lambda sets the amplitude and is r-dependent for lambda <= 0.
    "Carbajal-Tinoco":       ("doCarbajalTinokoClosure", True),
    "Modified Verlet":       ("doModifiedVerletClosure", True),
    #Extended RY takes TWO parameters (alpha and the quadratic coefficient a);
    #the second is set on the solver directly, so only alpha comes through the
    #single-parameter GUI field. a = 0 reduces exactly to Rogers-Young.
    "Extended Rogers-Young": ("doExtendedRYclosure", True),
    # ZSEP (Lee 1995, hard spheres only) is a "special" entry, like
    # Reference HNC/Rescaled MSA above: it needs its own orchestration
    # method (fitZSEPparameters()) rather than a plain doXXXclosure()+
    # solve() call, since it self-fits all three of its own parameters
    # (zeta, phi, alpha) against Lee's exact zero-separation-theorem
    # conditions instead of taking any of them from the caller. See
    # solve() below for exactly how it's dispatched, and
    # oZsolver.py's own fitZSEPparameters() docstring for the physics
    # and its documented ill-conditioning caveat.
    "ZSEP":                  ("fitZSEPparameters", False),
}

# Closures with a single free parameter (needsExtraScalarParam==True
# above) that ALSO have a published thermodynamic-consistency
# condition determining that parameter automatically -- see
# OZsolver.findThermodynamicallyConsistentParameter() (already fully
# generic across potentials, not limited to any specific one; ported
# directly from sasfit_oz_solver.c's own root_finding(), NOT the
# same as the older, abandoned, LJ/Yukawa-only optimizeRYalpha.py
# prototype elsewhere in this project). Maps this module's own
# closure name to that method's own closureName argument. Used by
# solve()'s new findConsistentParameter= option below -- Modified HNC
# is included here too (it already appears above with
# needsExtraScalarParam=True for its own eta parameter, which this
# same consistency condition also determines).
# Closures needing a SECOND scalar beyond closureParam, as
#   dropdown label -> (argument name, default, short description)
#
# Only Extended Rogers-Young needs this so far: its bridge function is the
# ordinary RY construction plus one extra quadratic term,
#     phi = (exp(f*Gamma)-1)/f,  B = -Gamma + log1p(phi + a*phi^2)
# so it carries the RY switching rate alpha AND the coefficient a. a = 0
# reduces exactly to RY, which is why that is the default.
#
# Kept as its own table rather than widening CLOSURE_SETTERS to 3-tuples,
# so every existing consumer of CLOSURE_SETTERS keeps working unchanged.
SECOND_CLOSURE_PARAM = {
    "Extended Rogers-Young": ("a", 0.0,
                              "quadratic coefficient; a = 0 gives plain RY"),
}


def secondClosureParam(closureName):
    """(name, default, description) or None if the closure needs only one."""
    return SECOND_CLOSURE_PARAM.get(closureName)


# Which closures can be used with a MULTICOMPONENT (polydisperse) potential.
#
# The distinction is structural, not a matter of taste. Every closure whose
# bridge function is a plain elementwise expression in Gamma (and optionally
# the repulsive/attractive potential split) operates on the (p,p,N) pair
# matrices unchanged, so it is multicomponent for free. The excluded ones each
# need something that only exists for a single component:
#
#   Reference HNC   needs a one-component hard-sphere reference solve (g0/G0)
#   Modified HNC    needs an analytic one-component PY hard-sphere bridge
#                   function at a single packing fraction; the mixture
#                   equivalent is the Lado-type variational problem, i.e. a
#                   research question rather than a port
#   Rescaled MSA    is a one-component diameter-rescaling procedure; the
#                   polydisperse counterpart is the separate analytic module
#                   polydisperse_rmsa.py
#   EuRah           uses a precomputed one-component HS/PY array (CEURAH)
#   ZSEP            fixes its three parameters from the Carnahan-Starling
#                   zero-separation theorems, which are hard-sphere specific
#                   (fitZSEPparameters() already refuses anything else)
#
# Martynov-Sarkisov, Vompe-Martynov and CJVM are NOT excluded here: they are
# structurally fine and do run, but all three build a square-root bridge
# function and exp(G + B) overflows for strongly coupled charged systems --
# a closure-domain limitation that applies equally in one component, so it is
# reported as a failed solve rather than hidden as an unavailable option.
MULTICOMPONENT_INCAPABLE_CLOSURES = {
    "Reference HNC", "Modified HNC", "Rescaled MSA", "EuRah", "ZSEP",
}


def multicomponentCapableClosures():
    """Dropdown labels usable with a polydisperse potential."""
    return [name for name in CLOSURE_SETTERS
            if name not in MULTICOMPONENT_INCAPABLE_CLOSURES]


def isMulticomponentPotential(potentialName):
    """True for potentials that set up more than one component.

    A name test rather than a probe of the solver, so the GUI can filter the
    closure list the moment a potential is chosen, before anything is built.
    """
    return potentialName.startswith("Polydisperse")


CONSISTENT_PARAMETER_CLOSURES = {
    "Rogers-Young": "RY", "HMSA": "HMSA", "Modified HNC": "MHNC",
    "BPGG": "BPGG", "CJVM": "CJVM", "BB": "BB",
    #The three closures added from OrnsteinZernike.jl each carry a single
    #free parameter, so the same compressibility-vs-virial search applies.
    "Khanpour": "Khanpour", "Modified Verlet": "ModifiedVerlet",
    "Carbajal-Tinoco": "CarbajalTinoko",
    "Extended Rogers-Young": "ExtendedRY",
}

# The 9 curves every solve derives, same set oZgui.py's own plot tabs
# show (S(Q), g(r), c(r), Gamma(r), h(r), U(r)/kT, B(r), y(r), f(r)).
CURVE_NAMES = ["Sq", "gr", "cr", "gamma", "hr", "Ur", "Br", "yr", "fr"]


class OZResult:
    '''
    Everything one solve() call produced: the r/q grids, every derived
    curve, and the parameters that produced them. Curves are available
    both via .curves['gr'] and the shorthand properties below
    (.gr, .cr, .Sq, .gamma, .hr, .Ur, .Br, .yr, .fr).
    '''
    def __init__(self, r, q, potential, potentialArgs, closure, closureParam,
                 phi, solverName, curves, solverInstance):
        self.r = r
        self.q = q
        self.potential = potential
        self.potentialArgs = potentialArgs
        self.closure = closure
        self.closureParam = closureParam
        self.phi = phi
        self.solverName = solverName
        self.curves = curves
        #kept for anyone who wants lower-level access (e.g.
        #solverInstance.getHardSphereRadius(), .boltzmannOfP2Ppotential,
        #or to re-solve at a nearby state using this one as a warm start)
        #-- not required for normal use, everything routine is already
        #on this object directly.
        self.solverInstance = solverInstance

    @property
    def gr(self): return self.curves["gr"]
    @property
    def cr(self): return self.curves["cr"]
    @property
    def Sq(self): return self.curves["Sq"]
    @property
    def gamma(self): return self.curves["gamma"]
    @property
    def hr(self): return self.curves["hr"]
    @property
    def Ur(self): return self.curves["Ur"]
    @property
    def Br(self): return self.curves["Br"]
    @property
    def yr(self): return self.curves["yr"]
    @property
    def fr(self): return self.curves["fr"]

    def __repr__(self):
        return (f"OZResult(potential={self.potential!r}, potentialArgs={self.potentialArgs!r}, "
                f"closure={self.closure!r}, phi={self.phi!r}, solver={self.solverName!r})")


def _minSq(curves):
    """min S(Q) from a curve dict, or +inf when there is nothing to screen."""
    try:
        y = np.real(np.asarray(curves["Sq"], float))
    except Exception:
        return float("inf")
    if not np.all(np.isfinite(y)):
        return float("-inf")
    return float(np.min(y))


def _fixpointResidual(inst, curves):
    """Relative residual ||T(x) - x|| / ||x|| of the returned solution, with T
    the closure's fixed-point map, evaluated ONCE MORE after the solve.

    This is independent of the solver's own stopping test and of its success
    flag: it asks the equations, not the solver. x is the fixpoint variable
    the solver iterated on -- Gamma = h - c (one component, from the derived
    curves), the packed pair matrix (mixtures) -- and T is applied WITHOUT
    any deflation factor (a deflated solver's map is scaled near known
    roots; its fixed points are those of T, which is what is checked).
    Returns None when the residual cannot be formed (the [h, c] variant of
    the operator, or no fixpoint stored); then nothing is checked.
    """
    try:
        if inst.isMulticomponent():
            G = getattr(inst, "gammaMatrixMulticomponent", None)
            if G is None:
                return None
            x = np.asarray(inst.packPairs(G), float)
            tx = np.asarray(inst.fixPointOperatorForGammaMulticomponent(x)[0], float)
        elif getattr(inst, "doUseGammaFixPointOperator", True):
            x = np.asarray(curves["gamma"], float)
            tx = np.asarray(inst.fixPointOperatorForGamma(x)[0], float)
        else:
            return None
    except Exception:
        return None
    if not (np.all(np.isfinite(x)) and np.all(np.isfinite(tx))):
        return float("inf")
    return float(np.linalg.norm(tx - x) / max(np.linalg.norm(x), 1e-300))


def _deflateUntilPhysical(solverInstance, curves, buildAndSolve, maxAttempts,
                          deriveCurves):
    """Reject a root with min S(Q) < 0 and re-solve with it deflated away.

    Returns (solverInstance, curves) for the first physical root found, and
    raises ValueError when there is none.

    WHY SCREEN AT ALL. S(Q) is a variance and cannot be negative, but nothing
    about the iteration can detect a negative one. tools/residual_check.py
    finds every Newton-Krylov variant converging at one Lennard-Jones state
    point to a residual of 1e-13, with `converged` true and the returned
    point a genuine fixed point -- at min S(Q) = -38.96. It is a real
    solution of the closure equations on a branch that does not describe a
    fluid, and only this test separates it from the correct root at the same
    state point.

    WHY DEFLATION RATHER THAN A DIFFERENT SOLVER. Retrying with the other
    family would often work, since fixed-point and Newton-Krylov methods
    empirically favour different branches. Two reasons against. It silently
    returns a result from a solver the caller did not choose, which is the
    kind of substitution this package works to eliminate. And it leans on a
    bias that happens to hold at the state points tested rather than on
    anything guaranteed -- nothing stops both families landing on the same
    unphysical root elsewhere.

    Deflation divides the residual by a factor singular at each root already
    found, so the SAME solver cannot return there: it must find something
    else or fail honestly. The caller's choice of solver is respected, and
    since makeDeflatedClass wraps any base class it does not require SUNDIALS
    to be installed.

    WHEN IT FAILS, THAT IS INFORMATION. Past a spinodal there may be no
    physical solution at the state point, and deflation will go on finding
    unphysical ones because that is what exists. The exception therefore
    names every root found with its min S(Q) rather than reporting a bare
    non-convergence: those numbers say something about the state point, and
    discarding them would throw the evidence away.
    """
    worst = _minSq(curves)
    if worst >= 0.0:
        return solverInstance, curves

    try:
        from .deflation import makeDeflatedClass
    except Exception as exc:
        raise ValueError(
            f"min S(Q) = {worst:.4g} < 0: the solver converged to a root that "
            f"is not a fluid, and deflation is unavailable "
            f"({type(exc).__name__}: {exc}). Pass deflateOnUnphysical=0 to "
            f"accept the unphysical root.") from exc

    roots = []
    history = [("initial", worst)]
    for attempt in range(max(1, int(maxAttempts))):
        #The fixpoint variable is gamma; that is what the deflation factor
        #measures distance from.
        try:
            roots.append(np.asarray(curves["gamma"], float).copy())
        except Exception:
            break
        try:
            solverInstance = buildAndSolve(
                makeDeflatedClass(type(solverInstance).__mro__[1]
                                  if type(solverInstance).__name__.startswith(
                                      "Deflated") else type(solverInstance),
                                  list(roots)))
            curves = deriveCurves(solverInstance)
        except Exception as exc:
            #Summarise the exception rather than interpolating it: scipy's
            #NoConvergence carries the whole residual vector, so str(exc) is
            #a thousand numbers and the useful part -- that the deflated
            #solve could not converge -- is lost in them.
            detail = str(exc).split("\n")[0][:120]
            raise ValueError(
                f"min S(Q) = {worst:.4g} < 0, and the re-solve with that root "
                f"deflated away did not converge ({type(exc).__name__}). "
                f"Roots found: {_rootSummary(history)}. That the solver "
                f"cannot reach another root once this one is excluded is "
                f"itself evidence: there may be no physical solution at this "
                f"state point. Detail: {detail}") from exc
        worst = _minSq(curves)
        history.append((f"deflation {attempt + 1}", worst))
        if worst >= 0.0:
            return solverInstance, curves

    raise ValueError(
        f"no physical solution found after {len(roots)} deflation(s): every "
        f"root located has min S(Q) < 0. These are genuine solutions of the "
        f"closure equations on branches with negative compressibility, not "
        f"failed calculations, which is what a state point at or past a "
        f"spinodal looks like. Roots found: {_rootSummary(history)}. Try a "
        f"different solver, a different closure, or a less extreme state; "
        f"pass deflateOnUnphysical=0 to accept the unphysical root instead.")


def _rootSummary(history):
    return ", ".join(f"{name}: min S(Q) = {v:.4g}" for name, v in history)


def solve(potential, phi, potentialArgs=(), closure="Percus-Yevick", closureParam=None,
          closureParam2=None,
          findConsistentParameter=False,
          solver=_defaultSolverName, maxIterations=None,
          numberOfRadialSamplingPoints=None, hardSphereDiameterInPoints=None,
          onSolverCreated=None, verify=True,
          verifyWith=("Biggs-Andrews", "Picard / Mann iteration"),
          verifyTolerance=1e-3, residualTolerance=1e-6,
          deflateOnUnphysical=0, transformType=None, mannAlpha=None,
          verifyMannAlpha=0.5):
    '''
    Run one full OZ solve and return an OZResult with every derived
    curve. This is the exact same workflow oZgui.py's own "calculate"
    button runs (that button calls this function too) -- see
    SOLVER_CLASSES/CLOSURE_SETTERS in this module for the exact list
    of valid `solver=`/`closure=` names.

    Parameters
    ----------
    potential : str
        Potential name, e.g. 'HardSphere', 'StickyHardSphere', 'DLVO'.
        See oZfixpointOperator.py's own setXXXPotential() methods, or call
        getAvailablePotentialNames() on any solver instance, for the full
        current list.
    phi : float
        Volume fraction.
    potentialArgs : tuple
        Positional arguments for that potential's own setXXXPotential(),
        e.g. ``(tau, delta)`` for StickyHardSphere. There is no keyword
        form, and the meaning is per-potential.
    closure : str
        Closure name, see CLOSURE_SETTERS.
    closureParam : float, optional
        The closure's own alpha/eta value, required for closures where
        ``CLOSURE_SETTERS[closure][1]`` is True (Modified HNC,
        Rogers-Young, HMSA, BPGG, CJVM, BB) UNLESS findConsistentParameter
        is given instead. Not used for ZSEP, which always self-fits all
        three of its own parameters; see fitZSEPparameters() on OZsolver
        to fix one rather than fit all three.
    closureParam2 : float, optional
        Second parameter, used only by Extended Rogers-Young. See
        SECOND_CLOSURE_PARAM.
    findConsistentParameter : bool, optional
        If True, search for the closureParam that makes the
        compressibility-route and virial-route isothermal compressibility
        agree, rather than taking one from the caller. See
        OZsolver.findThermodynamicallyConsistentParameter() for the physics
        and its caveats: a located root can occasionally be a numerical
        artefact, and that method screens for it and falls back to a
        closest-approach value with a printed warning when it cannot find a
        clean one. Valid only for CONSISTENT_PARAMETER_CLOSURES; raises
        ValueError otherwise. When True, closureParam is ignored.
    solver : str, optional
        Solver name, see SOLVER_CLASSES. Defaults to SUNDIALS'
        Anderson-accelerated fixed point when sundials4py is installed,
        otherwise "scipy Anderson". Picard remains available as
        ``solver="Picard / Mann iteration"`` (the former name
        "Picard iteration" is still accepted), but is no longer the default.

        Prefer a FIXED-POINT method. Near a fold the Jacobian is singular
        and the Newton-Krylov family is drawn onto
        negative-compressibility branches precisely where the fold lies:
        measured at one Lennard-Jones state, the fixed-point family found
        g_max = 2.16 with min S(Q) = +0.21 while Newton-Krylov found 1.13
        and 1.22 with min S(Q) near -38, all at residuals below 1e-11.
        Every one is a correct answer; one is physical.
    maxIterations : int, optional
        Upper bound on iterations or function evaluations. None (default)
        uses each solver's own limit (Preferences: oz.solvers.max_iterations
        and oz.solvers.max_iterations_picard). For Picard / Mann the value
        is the limit at a = 1 and is multiplied by 1/a (see mannAlpha).
    numberOfRadialSamplingPoints : int, optional
        Radial grid size. None keeps this project's original 4095 points.
        Note the real-space range is ``N*sigma/pointsPerSigma``, so raising
        the resolution at fixed N SHRINKS the box -- scale both together or
        a convergence test measures nothing. Use
        oZfixpointOperator.bestGridSize() rather than choosing by hand: a
        DST-I wants 2^k - 1 and a DST-IV wants 2^k.
    hardSphereDiameterInPoints : int, optional
        Radial resolution. None keeps 100 points per hard-sphere diameter.
    onSolverCreated : callable, optional
        Called with the just-constructed solver instance after the
        potential and closure are applied but BEFORE solve() runs. This is
        how a caller reaches the live instance while a long solve is in
        progress -- solve() itself returns only once the computation has
        finished -- and how the GUI's Interrupt button works. It is also
        Not needed for normal scripted use.
    verify : bool, optional
        Re-solve with other solvers and raise if they land on different
        fixed points. True by default and worth leaving so: the closure
        equations genuinely admit several solutions, and a solver reporting
        success is not claiming to have found the physical one.
    verifyWith : tuple of str, optional
        Verifiers, tried in order. Skipped: the solver being verified and
        any verifier of the SAME ALGORITHM FAMILY (SOLVER_FAMILIES; e.g.
        MDIIS, scipy Anderson and SUNDIALS KIN_FP are all Anderson/DIIS
        acceleration and never verify each other). The first verifier that
        CONVERGES is used; one that stops at its iteration limit is no
        reference and the next is tried. Each verifier uses its own
        iteration limit from the preferences, not maxIterations. If none
        converges, VerificationUnavailable is raised (it carries the
        unverified result as ``.result``). Newton-Krylov methods are
        deliberately absent (see below).
    verifyMannAlpha : float in (0, 1], optional
        Damping a of the Picard / Mann iteration when it serves as a
        verifier (default 0.5; GUI: preference oz.solve.verify_mann_alpha).
        Undamped Picard diverges above phi ~ 0.42, where Biggs-Andrews also
        stops converging; damped it remains a verifier for dense states.
        Its limit is the Picard limit / a.
    verifyTolerance : float, optional
        How far the verifying solutions may differ before raising.
    residualTolerance : float or None, optional
        Independent convergence check (default 1e-6; GUI: preference
        oz.solve.residual_tolerance). After the solve the closure's
        fixed-point map T is applied once more to the returned solution x;
        if ||T(x) - x|| / ||x|| exceeds this, the result is marked NOT
        converged (RuntimeWarning, ``OZResult.converged = False``) whatever
        the solver reported. ``OZResult.residual`` holds the value. None
        switches the check off. Converged solves give 1e-16 to 1e-11.
    mannAlpha : float in (0, 1], optional
        Damping a of the Picard / Mann iteration,
        x_{n+1} = (1 - a) x_n + a T(x_n). None or 1 is plain Picard; a < 1 is
        Mann's iteration (Mann 1953): slower, but it converges where undamped
        Picard oscillates or diverges (dense or strongly attractive states).
        Each damped step moves only a fraction a of the way, so the
        iteration limit is ceil(N/a), N being the limit for a = 1
        (maxIterations, or the Picard default from the preferences,
        oz.solvers.max_iterations_picard). It applies to every Picard / Mann solve of this call -- the primary
        solve, a deflation retry and the Picard fallback verifier -- and is
        ignored by the other solvers, which control their own step. The value
        used is stored as ``OZResult.mannAlpha``.

    Returns
    -------
    OZResult
        Every derived curve as a plain array: ``Sq`` against ``q``, and
        ``gr``, ``cr``, ``hr``, ``gamma``, ``Ur``, ``Br``, ``yr``, ``fr``
        against ``r``. Also carries the inputs and ``solverInstance``.

    Raises
    ------
    ValueError
        For an unrecognised closure or solver name; a closure needing
        closureParam when none was given and findConsistentParameter was
        not requested; ZSEP for any potential other than HardSphere; or a
        verification disagreement.

    Examples
    --------
    >>> import ozLib
    >>> res = ozLib.solve("HardSphere", phi=0.3)
    >>> q, S = res.q, res.Sq

    A square well. Sign convention A (project-wide): eps is the DEPTH,
    beta u = -eps in the well, so a POSITIVE eps is attractive and a negative
    one a shoulder. (pyOZgui used the opposite sign.)

    >>> res = ozLib.solve("SquareWell", phi=0.3, potentialArgs=(1.0, 0.1),
    ...                   closure="Hypernetted-Chain")

    Notes
    -----
    The ValueErrors above are all checked HERE, before the solver is
    touched, so a mistake is reported immediately rather than surfacing
    later as a confusing internal AttributeError.
    '''
    if closure not in CLOSURE_SETTERS:
        raise ValueError(f"unknown closure {closure!r}, must be one of {list(CLOSURE_SETTERS)}")
    solver = resolveSolverName(solver)
    if solver not in SOLVER_CLASSES:
        raise ValueError(f"unknown solver {solver!r}, must be one of {list(SOLVER_CLASSES)}")
    verifyMannAlpha = float(verifyMannAlpha)
    if not (0.0 < verifyMannAlpha <= 1.0):
        raise ValueError(f"verifyMannAlpha must be in (0, 1], got {verifyMannAlpha}")
    if mannAlpha is not None:
        mannAlpha = float(mannAlpha)
        if not (0.0 < mannAlpha <= 1.0):
            raise ValueError(f"mannAlpha must be in (0, 1], got {mannAlpha}")
    if closure == "ZSEP" and potential != "HardSphere":
        # OZsolver.fitZSEPparameters() ALSO checks this itself (see that
        # method's own docstring) -- but only via a print()ed warning
        # and an early `return None`, not an exception. Without this
        # check here too, that would leave solverInstance completely
        # unsolved (radialDistributionFunction etc. still at their
        # __init__ all-zeros defaults) and this function would still
        # happily package that up into a normal-looking OZResult full of
        # zeros/S(Q)=1 everywhere, with no exception and nothing wrong
        # visible except a Log-tab message a GUI user could easily miss.
        # ZSEP's three fitting conditions (Lee 1995) are derived from the
        # exact Carnahan-Starling hard-sphere equation of state
        # specifically -- there is no meaningful way to "fit" it against
        # any other potential's own equation of state.
        raise ValueError("closure='ZSEP' is only meaningful for potential='HardSphere' "
                          "(Lee (1995)'s zero-separation conditions are derived from the exact "
                          f"Carnahan-Starling hard-sphere equation of state), not {potential!r}")

    setterName, needsParam = CLOSURE_SETTERS[closure]
    if findConsistentParameter and closure not in CONSISTENT_PARAMETER_CLOSURES:
        raise ValueError(f"findConsistentParameter=True is only supported for "
                          f"{list(CONSISTENT_PARAMETER_CLOSURES)}, not {closure!r}")
    if needsParam and closureParam is None and not findConsistentParameter:
        raise ValueError(f"closure {closure!r} needs a closureParam value "
                          f"(or findConsistentParameter=True)")

    solverClass, solverLinearSolver = SOLVER_CLASSES[solver]

    gridKwargs = {}
    if numberOfRadialSamplingPoints is not None:
        gridKwargs["numberOfRadialSamplingPoints"] = numberOfRadialSamplingPoints
    if hardSphereDiameterInPoints is not None:
        gridKwargs["hardSphereDiameterInPoints"] = hardSphereDiameterInPoints

    #CONSTRUCTION EXTRACTED INTO A CLOSURE so it can be repeated.
    #
    #This is the same sequence as before, unchanged, but callable: the
    #deflation retry below needs to build and run the SAME solver again with
    #its rootOperator wrapped, and inlining it twice would guarantee the two
    #copies drift apart. `cls` defaults to the class the caller asked for;
    #the retry passes a deflated subclass of it.
    def _buildAndSolve(cls=None):
        inst = (cls or solverClass)(port=0, **gridKwargs)
        #transformType: None = automatic (type 4 for one component, 1 for a
        #mixture, see resolvedTransformType); 1 or 4 forces it, e.g. to
        #compare the two. Set BEFORE the potential: the radial grid depends
        #on it.
        if transformType is not None:
            inst.transformType = int(transformType)
        if solverLinearSolver is not None:
            inst.linearSolver = solverLinearSolver
        #Mann damping: only the Picard / Mann solver reads it.
        if mannAlpha is not None and isinstance(inst, PicardOZsolver):
            inst.mannAlpha = mannAlpha
        inst.setNumberOfIterations(maxIterations)
        inst.setVolumeDensity(phi)
        inst.setPotentialByName(potential, *potentialArgs)

        if onSolverCreated is not None:
            onSolverCreated(inst)

        if setterName == "doRHNCclosure":
            inst.solveRHNC()
        elif setterName == "doRMSAclosure":
            inst.solveRMSA()
        elif setterName == "fitZSEPparameters":
            inst.fitZSEPparameters()
        elif findConsistentParameter:
            inst.findThermodynamicallyConsistentParameter(CONSISTENT_PARAMETER_CLOSURES[closure])
        else:
            if needsParam:
                #Closures listed in SECOND_CLOSURE_PARAM take a second
                #scalar; pass it through when the caller supplied one,
                #otherwise let the setter apply its own default (a = 0 for
                #Extended RY, which reduces it exactly to Rogers-Young).
                extra = SECOND_CLOSURE_PARAM.get(closure)
                if extra is not None and closureParam2 is not None:
                    getattr(inst, setterName)(closureParam, closureParam2)
                else:
                    getattr(inst, setterName)(closureParam)
            else:
                getattr(inst, setterName)()
            inst.solve()
        return inst

    solverInstance = _buildAndSolve()

    #CURVE DERIVATION EXTRACTED TOO, for the same reason as the construction
    #above: the deflation retry produces a new solver instance and needs the
    #same curves derived from it, and two copies of this would drift.
    def _deriveCurves(inst):
        curves = {}
        curves["gr"] = inst.getRDF()
        curves["cr"] = inst.getDirectCorrelationFunction()
        curves["Sq"] = np.real(inst.getSq())
        g = curves["gr"]; c = curves["cr"]
        curves["gamma"] = g - c - 1.0
        curves["hr"] = g - 1.0
        U = -np.log(np.clip(inst.boltzmannOfP2Ppotential, 1e-300, None))
        curves["Ur"] = U
        EN = inst.boltzmannOfP2Ppotential
        curves["fr"] = EN - 1.0
        # cavity y(r)=g/EN outside the hard core, and bridge
        # B(r)=log(y(r))-Gamma(r) wherever y(r)>0 -- matches the original
        # Tcl GUI's own B(r)/y(r) tabs, derived the same way (from
        # whatever g/c/Gamma the chosen closure and solver actually
        # converged to, not re-solved).
        valid = EN != 0.0
        y = np.where(valid, g / np.clip(EN, 1e-300, None), np.nan)
        curves["yr"] = y
        with np.errstate(invalid="ignore", divide="ignore"):
            curves["Br"] = np.where(y > 0, np.log(y) - curves["gamma"], np.nan)
        return curves

    curves = _deriveCurves(solverInstance)

    #PHYSICALITY SCREEN AND DEFLATION RETRY, before verification: there is no
    #point cross-checking a root that is about to be rejected. See
    #_deflateUntilPhysical for why deflation rather than switching solver.
    #The screen itself (min S(Q) >= 0) runs on EVERY solve, see below;
    #deflateOnUnphysical only decides whether an unphysical root is retried.
    if deflateOnUnphysical:
        solverInstance, curves = _deflateUntilPhysical(
            solverInstance, curves, _buildAndSolve,
            deflateOnUnphysical, _deriveCurves)

    result = OZResult(solverInstance.getrArray(), solverInstance.getqArray(),
                     potential, tuple(potentialArgs), closure, closureParam,
                     phi, solver, curves, solverInstance)
    #PHYSICALITY SCREEN, ALWAYS. S(Q) is a variance and cannot be negative,
    #but no convergence test can see that: Newton-Krylov converges for
    #Lennard-Jones eps = 0.8, phi = 0.3 (HNC) to a genuine root with
    #min S(Q) = -35 and converged = True. Before, the screen ran only when
    #deflateOnUnphysical > 0 (default 0), so such a root came back without
    #any notice. Now every result carries minSq / physical, and an
    #unphysical one raises a RuntimeWarning (with deflation on, an
    #unphysical root that deflation cannot replace has already raised).
    result.minSq = _minSq(curves)
    result.physical = bool(result.minSq >= 0.0)
    if not result.physical:
        msg = (f"min S(Q) = {result.minSq:.4g} < 0: {solver} converged to a root that is not a "
               f"fluid (negative compressibility branch, or past a spinodal). Use "
               f"deflateOnUnphysical > 0 to search for a physical root, or a fixed-point solver.")
        print("WARNING:", msg)
        warnings.warn(msg, RuntimeWarning, stacklevel=2)
    result.convergenceMessage = getattr(result, "convergenceMessage", None)
    result.verifiedWith = None             # set below when verification ran
    result.verificationDifference = None
    #the damping actually used (None for the solvers that have none)
    result.mannAlpha = ((mannAlpha if mannAlpha is not None else 1.0)
                        if isMannSolver(solver) else None)
    #NOT CONVERGED IS REPORTED, not hidden. A solve that stops at the
    #iteration limit still returns its last iterate, which looks like a
    #solution: for hard spheres (PY, phi = 0.53) Picard/Mann after 1000
    #iterations gave g_max between 4.35 and 6.77 depending on a, while the
    #converged root is 5.556220 for every a <= 0.6. The flag lets callers
    #(and the GUI) tell such an iterate from a solution.
    result.converged = bool(getattr(solverInstance, "converged", False))
    result.finalRelativeProgress = getattr(solverInstance, "finalRelativeProgress", None)
    #the iteration limit that applied (Picard / Mann: ceil(N/a)) and, where
    #the solver counts them, the iterations used
    result.iterationLimit = getattr(solverInstance, "iterationLimit",
                                    getattr(solverInstance, "numberOfIterations", None))
    result.iterations = getattr(solverInstance, "iterations", None)
    #INDEPENDENT RESIDUAL CHECK. The solver's own flag is a claim; the
    #residual of the returned solution is evidence. T is applied once more to
    #the returned fixpoint and the result is NOT converged when
    #||T(x) - x|| / ||x|| > residualTolerance, whatever the solver reported.
    #Example: polydisperse hard-core Yukawa (HNC, phi = 0.1) under KIN_FP
    #reports success and returns S(Q) = NaN everywhere; the residual is
    #infinite. Converged solves give 1e-16 ... 1e-11, so 1e-6 rejects only
    #real failures.
    result.residual = _fixpointResidual(solverInstance, curves)
    result.residualTolerance = residualTolerance
    residualFailed = (residualTolerance is not None and result.residual is not None
                      and not result.residual <= residualTolerance)
    #only a CLAIMED convergence can be overruled; an iteration that stopped at
    #its limit is reported as such (its residual is then expected to be large)
    residualFailed = residualFailed and result.converged
    if residualFailed:
        result.converged = False
    if not result.converged:
        prog = result.finalRelativeProgress
        if residualFailed:
            msg = (f"{solver} reported convergence, but the residual of the returned solution is "
                   f"||T(x)-x||/||x|| = {result.residual:.3g} > residualTolerance = "
                   f"{residualTolerance:g}; the result is not a solution.")
        else:
            msg = (f"{solver} did not converge"
                   + (f" within {result.iterationLimit} iterations" if result.iterationLimit else "")
                   + (f" (relative progress {prog:.2e})" if prog is not None and np.isfinite(prog) else "")
                   + (f", residual ||T(x)-x||/||x|| = {result.residual:.3g}"
                      if result.residual is not None else "")
                   + "; the result is the last iterate, not a solution.")
        result.convergenceMessage = msg
        print("WARNING:", msg)
        warnings.warn(msg, RuntimeWarning, stacklevel=2)

    #VERIFICATION IS ON BY DEFAULT. A converged residual is not evidence that
    #the answer is right: the closure equations admit multiple fixed points,
    #and this package has produced cases where a solver reported convergence
    #and returned a root 34 % away from the correct one, finite and
    #plausible. Reliability matters more here than speed, so the result is
    #cross-checked against an INDEPENDENT solver from a COLD START before it
    #is returned.
    #
    #Pass verify=False for an interactive sweep where the cost of a second
    #solve matters and the states are already known to be well behaved. Note
    #that the second solve must NOT be warm-started from the first: a wrong
    #root is still a root, so a warm-started check agrees with anything it is
    #given. See solveWithConsensus() for the measurement.
    if verify and verifyWith:
        #verifyWith may be a single name or a sequence tried in order.
        #
        #BIGGS-ANDREWS FIRST, PICARD AS FALLBACK. Picard was first here on
        #independence grounds -- no acceleration, no history vectors, so it
        #cannot share an acceleration failure mode. That reasoning is sound
        #but was outweighed by two measurements:
        #
        #  * it is the SLOWEST solver available, 0.0886 s against 0.0075 s
        #    for the fastest, so verification cost a factor of ten;
        #  * it was the one solver that returned a NON-SOLUTION while
        #    reporting success (BPGG alpha=0.5, phi=0.4: g_max = 3.0039 with
        #    a residual of 19.4). That specific defect is now fixed -- the
        #    convergence test compares the increment rather than successive
        #    norms -- but a verifier should be the sturdiest solver
        #    available, not the one with the most recent repair.
        #
        #Biggs-Andrews is also a fixed-point method, so it still does not
        #share a Newton-family failure mode, and it converged to the correct
        #root on every case tested. Note it converges to ~5e-11 on hard
        #spheres where accelerated solvers reach 1e-13; a verifier's own
        #tolerance sets the floor on the disagreement it can detect.
        #
        #DO NOT use a Newton-Krylov method as the verifier. Measured at the
        #Lennard-Jones state, all four Newton-Krylov variants converge to
        #negative-compressibility branches (min S(Q) ~ -38) while every
        #fixed-point method finds the physical root. Such a verifier would
        #report disagreement everywhere near a fold, and be wrong itself.
        names = ([verifyWith] if isinstance(verifyWith, str)
                 else list(verifyWith))
        names = [resolveSolverName(n) for n in names]
        #INDEPENDENCE: a verifier from the same algorithm family as the
        #primary is no check (SUNDIALS KIN_FP, scipy Anderson and MDIIS are
        #all Anderson/DIIS acceleration -- Pulay's DIIS and Anderson mixing
        #are algebraically equivalent -- so they can share a failure mode and
        #agree on a wrong answer). Such verifiers are skipped.
        fam = solverFamily(solver)
        skipped = [n for n in names if n in SOLVER_CLASSES and n != solver
                   and solverFamily(n) == fam]
        names = [n for n in names if n != solver and n in SOLVER_CLASSES
                 and solverFamily(n) != fam]
        if not names:
            raise ValueError(
                f"no independent verifier for solver {solver!r}: tried "
                f"{verifyWith!r}"
                + (f" ({', '.join(skipped)} skipped: same algorithm family)" if skipped else "")
                + ". Pass verify=False to accept an unverified result.")
        check, attempts = None, []
        for nm in names:
            try:
                #Announce the verifier before it runs. Its own solve() prints
                #"<solver> converged after N steps" exactly as the primary
                #solve does, so the log showed two converged lines with
                #nothing to say which was which -- it reads as though the
                #solver had been switched, which is precisely what does NOT
                #happen: the returned result is always from the solver the
                #caller chose, and this second solve only cross-checks it.
                print(f"verifying with {nm} (independent cross-check; the "
                      f"result returned is still {solver}'s)")
                #Each verifier runs with ITS OWN iteration limit from the
                #preferences (maxIterations was chosen for the primary). The
                #Picard / Mann verifier is DAMPED with verifyMannAlpha:
                #undamped Picard diverges above phi ~ 0.42, exactly where
                #Biggs-Andrews also stops converging, which left no verifier
                #for dense states. Its limit is scaled by 1/a as usual.
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", RuntimeWarning)
                    c = solve(potential, phi, potentialArgs, closure,
                              closureParam, closureParam2,
                              findConsistentParameter, solver=nm,
                              numberOfRadialSamplingPoints=numberOfRadialSamplingPoints,
                              hardSphereDiameterInPoints=hardSphereDiameterInPoints,
                              verify=False, transformType=transformType,
                              mannAlpha=verifyMannAlpha if isMannSolver(nm) else None,
                              residualTolerance=residualTolerance)
                b = np.real(np.asarray(c.curves["Sq"], float))
                if not np.all(np.isfinite(b)):
                    attempts.append(f"{nm}: non-finite S(Q)")
                elif not c.converged:
                    #A verifier that stopped at its iteration limit is no
                    #reference: comparing with it would report an unfinished
                    #iteration as a disagreement. Try the next one.
                    attempts.append(f"{nm}: not converged within {c.iterationLimit} iterations"
                                    + (f" (a = {c.mannAlpha:g})" if c.mannAlpha is not None else ""))
                else:
                    check, verifyWith = c, nm
                    break
            except Exception as exc:
                attempts.append(f"{nm}: {type(exc).__name__}: {exc}")
        if check is None:
            #NO INDEPENDENT VERIFIER CONVERGED. This is not a disagreement, and
            #must not pass silently as agreement either: the result is
            #returned on the exception so a caller (the GUI) can still show
            #it, clearly marked as unverified.
            raise VerificationUnavailable(
                f"could not verify the result of {solver!r}: no independent "
                f"verifier converged ({'; '.join(attempts)}"
                + (f"; skipped as same algorithm family: {', '.join(skipped)}" if skipped else "")
                + "). Raise the iteration limits in the preferences, or pass "
                  "verify=False to accept the unverified result.",
                result=result, attempts=attempts)
        result.verifiedWith = verifyWith
        a = np.real(np.asarray(curves["Sq"], float))
        b = np.real(np.asarray(check.curves["Sq"], float))
        scale = max(float(np.max(np.abs(a))), 1e-30)
        d = float(np.max(np.abs(a - b)))/scale
        result.verificationDifference = d
        if d > verifyTolerance:
            raise ValueError(
                f"solver {solver!r} and {verifyWith!r} disagree by "
                f"{d:.3g} (tolerance {verifyTolerance:g}) on S(Q) for "
                f"{potential} at phi={phi}, closure {closure}. "
                + (f"Both converged. This is the signature of multiple "
                   f"fixed points -- do not use either result without an "
                   f"independent reference. "
                   if result.converged else
                   f"{solver} did NOT converge, so this is an unfinished "
                   f"iteration, not a second root: raise the iteration limit"
                   + (" or lower the damping a" if isMannSolver(solver) else "") + ". ")
                + "Pass verify=False to override.")

    return result


# ----------------------------------------------------------------------
def solveWithConsensus(potential, potentialArgs=(), closure="doPYclosure",
                       closureParam=None, volumeDensity=0.3,
                       #"Anderson acceleration" WAS the first default here
                       #and no longer exists -- removing it from
                       #_BASE_SOLVER_CLASSES broke every call to this
                       #function, since the name is looked up there. Found by
                       #checking the manual's worked example against the
                       #signature rather than by anything failing, because
                       #nothing in the package calls this.
                       #
                       #The three chosen are deliberately DIFFERENT
                       #implementations -- scipy's Anderson, Biggs-Andrews
                       #and plain Picard -- since the point is agreement
                       #between independent routes. Two copies of one
                       #algorithm agreeing would prove nothing.
                       solvers=("scipy Anderson", "Biggs-Andrews",
                                "Picard / Mann iteration"),
                       tolerance=1e-3, gridN=4095, pointsPerSigma=100,
                       maxIterations=None, transformType=1, quantity="Sq"):
    """Solve with several solvers and return a result only if they AGREE.

    A converged residual is not evidence that the answer is right. The
    closure equations admit multiple fixed points, and this project has
    produced three separate cases where a solver reported convergence and
    returned a wrong answer:

      * two accelerated solvers converging to residuals ~1e-12 and returning
        S(Q) = 17.0 and 9.7, both with min S(Q) < 0;
      * a polydisperse hard-sphere mixture solved as though every particle
        had the same diameter (the identical-cores defect), which looked
        entirely normal;
      * Lennard-Jones at epsilon = 0.8, phi = 0.3 under HNC, where
        `scipy Anderson` returns g_max = 1.43 against the correct 2.16 --
        and does so for every value of its alpha tried (1.22, 1.43, 1.60),
        so it is not a misconfiguration.

    The last is reachable from a DEFAULT configuration with a standard
    potential, not an exotic state point. Its wrong root IS caught by the
    min S(Q) >= 0 screen (min S(Q) = -46.5 there), but only because S(Q) is
    the quantity screened: g_max alone is 1.43 and entirely plausible. This
    function does not rely on any single screen.

    Returns (solverInstance, report). Raises ValueError when no two solvers
    agree, reporting every difference rather than picking a majority
    silently -- three different answers is something the caller needs to see,
    not a coin toss.

    COST. Two to three solves. With Anderson at well under a second that is
    cheap beside being silently wrong by a third. Reduce `solvers` to two for
    speed, but keep them from different families: in testing
    `Anderson acceleration` and `Biggs-Andrews` agreed on every case tried,
    while both scipy wrappers proved unreliable on soft potentials.

    DO NOT WARM-START THE LATER SOLVERS from the first one's answer. It is
    the obvious optimisation, it IS faster -- 0.005 s against 0.017 s when
    tried -- and it destroys the check entirely. Measured on the
    Lennard-Jones case above: started from scipy Anderson's wrong root
    (g_max = 1.430723), `Anderson acceleration` STAYS THERE and returns
    1.430723, where from a cold start the same solver finds the correct
    2.163595. The two would then "agree" and this function would accept the
    wrong answer.

    The reason is structural. A wrong root is still a ROOT: any solver
    started on it sees a vanishing residual and has no reason to move. Warm
    starting asks "is this a fixed point?", which the first solver's own
    residual already answered. The question worth asking is "do independent
    searches from different starting points land in the same basin?", and
    that requires the starting points to be independent. The cost of the
    cold starts is not overhead to be optimised away -- it is the entire
    content of the check.
    """
    results, failures = {}, {}
    for name in solvers:
        name = resolveSolverName(name)
        if name not in SOLVER_CLASSES:
            failures[name] = "not installed"
            continue
        cls, _linear = SOLVER_CLASSES[name]
        try:
            s = cls(port=0, numberOfRadialSamplingPoints=gridN,
                    hardSphereDiameterInPoints=pointsPerSigma)
            s.transformType = transformType
            s.setNumberOfIterations(maxIterations)
            s.setVolumeDensity(volumeDensity)
            s.setPotentialByName(potential, *potentialArgs)
            if closureParam is not None:
                getattr(s, closure)(closureParam)
            else:
                getattr(s, closure)()
            s.solve()
            y = np.real(np.asarray(getattr(s, "get" + quantity)(), float))
            if not np.all(np.isfinite(y)):
                failures[name] = "non-finite result"
                continue
            if quantity == "Sq" and np.min(y) < 0.0:
                failures[name] = f"unphysical: min S(Q) = {np.min(y):.3g} < 0"
                continue
            results[name] = (s, y)
        except Exception as exc:
            failures[name] = f"{type(exc).__name__}: {exc}"

    if len(results) < 2:
        raise ValueError(
            f"fewer than two solvers produced a usable result, so nothing can "
            f"be verified. Failures: {failures}")

    names = list(results)
    ref = results[names[0]][1]
    scale = max(float(np.max(np.abs(ref))), 1e-30)
    diffs = {n: float(np.max(np.abs(results[n][1] - ref)))/scale
             for n in names}
    agreeing = [n for n in names if diffs[n] <= tolerance]
    if len(agreeing) < 2:
        #The first solver may itself be the outlier, so try each as the
        #reference before concluding that nothing agrees.
        for cand in names:
            r = results[cand][1]
            sc = max(float(np.max(np.abs(r))), 1e-30)
            grp = [n for n in names
                   if float(np.max(np.abs(results[n][1] - r)))/sc <= tolerance]
            if len(grp) >= 2:
                agreeing, ref = grp, r
                diffs = {n: float(np.max(np.abs(results[n][1] - r)))/sc
                         for n in names}
                break
    if len(agreeing) < 2:
        raise ValueError(
            f"no two solvers agree to {tolerance:g}. This is the signature of "
            f"multiple fixed points: every one of them 'converged'. Relative "
            f"differences {diffs}; failures {failures}. Do NOT use any of "
            f"these results without an independent reference.")

    report = dict(agreeing=agreeing, differences=diffs, failures=failures,
                  disagreeing=[n for n in names if n not in agreeing])
    return results[agreeing[0]][0], report
