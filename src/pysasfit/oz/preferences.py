# -*- coding: utf-8 -*-
"""Apply the OZ entries and the shared numerics of the pySASfit preferences
(Settings -> Preferences) to the OZ library: solver class defaults, deflation
constants, consistent-parameter search, analytic MSA/RMSA. Called by the OZ
GUI at start-up and after the preferences were saved; scripts are unaffected
unless they call it."""

from ..core import settings as prefs


def apply_solver_settings():
    """Set the module and class defaults of the OZ library from the preferences."""
    v = prefs.values("oz.solvers")
    from .oZsolver import OZsolver
    from .picardOZsolver import PicardOZsolver
    OZsolver.DEFAULT_MAX_ITERATIONS = int(v["max_iterations"])
    PicardOZsolver.DEFAULT_MAX_ITERATIONS = int(v["max_iterations_picard"])
    OZsolver.CONVERGENCE_CRITERION = float(v["convergence"])
    for mod, cls, attr, key in (("sundials4pyKinsolFPOZsolver", "Sundials4pyKinsolFPOZsolver", "MAA", "kinsol_maa"),
                                ("biggsAndrewsOZsolver", "BiggsAndrewsOZsolver", "MAA", "biggs_maa"),
                                ("mdiisOZsolver", "MDIISOZsolver", "HISTORY", "mdiis_history")):
        try:
            m = __import__(f"{__package__}.{mod}", fromlist=[cls])
        except Exception:          # optional dependency (sundials4py) missing
            continue
        setattr(getattr(m, cls), attr, int(v[key]))
    c = prefs.values("oz.consistency")
    OZsolver.CONSISTENCY_SCAN_STEPS = int(c["scan_steps"])
    OZsolver.CONSISTENCY_TOLERANCE = float(c["tolerance"])
    from . import deflation, polydisperse_rmsa, polydisperse_yukawa_msa
    d = prefs.values("numerics.deflation")
    deflation.DEFAULTS.update(p=d["p"], alpha=d["shift"], cap=d["cap"])
    polydisperse_yukawa_msa.MSA_DEFAULTS.update(prefs.values("numerics.msa"))
    polydisperse_rmsa.RMSA_DEFAULTS.update(prefs.values("numerics.rmsa"))


apply_settings = apply_solver_settings
