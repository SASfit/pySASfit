# -*- coding: utf-8 -*-
"""Project-wide sign convention for interaction strengths ("convention A").

In pySASfit every well depth and every Yukawa amplitude is POSITIVE FOR
ATTRACTION:

    square well, piecewise-constant steps:  beta u = -epsilon   (epsilon > 0 a well)
    Yukawa tails:                            beta u = -K exp(...)/r   (K > 0 attractive)

This is the convention of the SASfit structure-factor plugins (Sharma-Sharma
and Zepeda-Lopez square well, nonlinear ORPA, TwoYukawa, OneYukawa) and of the
literature (Blum/Hoye and Liu/Chen Yukawa, Sharma square well).

pyOZgui, and SASfit's C OZ solver for the square-well and piecewise-constant
potentials, used the opposite sign for three potentials -- the parameter was
beta u itself, positive = repulsive. :data:`FLIPPED_ARGS` lists them, and the
helpers here convert values and saved sessions written in that convention.
"""

CONVENTION = "A"

#potential name -> positions in potentialArgs whose sign changed relative to
#pyOZgui. Every other potential kept its meaning (HardSphereDoubleYukawa and
#HS3Yukawa were already convention A; LJ epsilon is a depth by definition;
#soft/PSM/Hertzian/charges/tau are not signed interaction strengths).
FLIPPED_ARGS = {
    "SquareWell": (0,),                    # (epsilon, delta)
    "PiecewiseConstantHS": (0, 2, 4),      # (eps1, d1, eps2, d2, eps3, d3)
    "Yukawa": (1,),                        # (shieldingLength, strength, doAddHardSphere)
}


def _negate(value):
    """-value, keeping a string a string (GUI entries store text)."""
    if isinstance(value, str):
        try:
            v = float(value)
        except ValueError:
            return value
        return repr(-v) if v != 0 else value
    try:
        return -value if value != 0 else value
    except TypeError:
        return value


def from_pyozgui(potential, args):
    """potentialArgs written for pyOZgui (convention B) -> convention A."""
    idx = FLIPPED_ARGS.get(potential, ())
    return tuple(_negate(a) if i in idx else a for i, a in enumerate(args))


def convert_generic_session(state):
    """Convert a generic-polydisperse-tab session state in place.

    Returns a short note when something was converted, else "". A state that
    already carries ``signConvention == "A"`` is left alone; one without the
    key was written before the change and uses pyOZgui's signs.
    """
    if not isinstance(state, dict) or state.get("signConvention") == CONVENTION:
        return ""
    state["signConvention"] = CONVENTION
    potential = (state.get("entries") or {}).get("potentialVar")
    idx = FLIPPED_ARGS.get(potential, ())
    if not idx:
        return ""
    vals = state.get("potParamVars")
    if isinstance(vals, list):
        state["potParamVars"] = [_negate(v) if i in idx else v for i, v in enumerate(vals)]
    bounds = state.get("fitBounds") or {}
    for i in idx:
        b = bounds.get(f"pot{i}")
        if isinstance(b, list) and len(b) == 2:
            #negate AND swap: [lo, hi] -> [-hi, -lo]
            bounds[f"pot{i}"] = [_negate(b[1]), _negate(b[0])]
    fit = state.get("fit")
    if isinstance(fit, dict) and isinstance(fit.get("parameters"), dict):
        for i in idx:
            if f"pot{i}" in fit["parameters"]:
                fit["parameters"][f"pot{i}"] = _negate(fit["parameters"][f"pot{i}"])
    return (f"{potential} strength sign converted to convention A "
            f"(positive = attractive)")


#Every SIGNED interaction strength the GUIs show, for a tooltip that states
#the convention next to the number (potential -> argument positions).
SIGNED_STRENGTH_ARGS = {
    "SquareWell": (0,),
    "PiecewiseConstantHS": (0, 2, 4),
    "Yukawa": (1,),
    "HardSphereDoubleYukawa": (0, 2),
    "HS3Yukawa": (0, 2, 4),
    "LennardJones": (0,),
}

HINT = "> 0 attractive (well), < 0 repulsive (shoulder)"


def strength_hint(potential, index):
    """Tooltip text for argument `index` of `potential`, or None."""
    return HINT if index in SIGNED_STRENGTH_ARGS.get(potential, ()) else None
