"""The candidate space of the scout (step 2): form factor x size distribution x structure factor.

Every entry names a SASfit function of the Studio catalogue and says which of its parameters are fitted
(with start value and bounds derived from the :class:`~pysasfit.scout.features.Features` of the data), which
are fixed, over which parameter a size distribution may run and whether a structure factor makes sense.
The parameter names are those of the SASfit plugin headers. More entries are added with
:func:`register_form`, :func:`register_distribution` and :func:`register_structure_factor`.

A fitted parameter is a :class:`Par`:

* ``Par("R", "size", 1.0)``: a length; start value = factor x typical size of the data
  (``features.size``), bounds = the lengths the q range resolves (times ``lo``, ``hi``);
* ``Par("s", "abs", 0.15, 0.02, 0.8)``: start value and bounds as given.

The contrast of a homogeneous particle is fixed at 1: the intensity scale is a linear parameter (the scale
factor of the contribution, or N of the size distribution) that variable projection solves exactly.
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = ["Par", "FormSpec", "DistSpec", "SqSpec", "Component", "FORMS", "DISTRIBUTIONS", "STRUCTURE_FACTORS",
           "register_form", "register_distribution", "register_structure_factor", "generate"]  # fmt: skip


SPAN = 4.0  # with a Guinier range, lengths are searched within start / SPAN ... start * SPAN


@dataclass(frozen=True)
class Par:
    name: str
    kind: str = "size"  # "size": value is a factor of features.size, lo/hi multiply the size range; "abs": as given
    value: float = 1.0
    lo: float = 1.0
    hi: float = 1.0

    def resolve(self, features, wide=False):
        """(start, lower, upper) for the data with these features; ``wide``: the whole resolvable range of lengths
        (mixtures: the Guinier radius belongs to the largest population, the others may be much smaller)."""
        if self.kind == "abs":
            return self.value, self.lo, self.hi
        lo, hi = features.size_range
        lo, hi = lo * self.lo, hi * self.hi
        v = min(max(self.value * features.size, lo * 1.01), hi / 1.01)
        if features.guinier_ok and not wide:  # the Guinier radius fixes the overall size: search within a factor SPAN of the start
            lo, hi = max(lo, v / SPAN), min(hi, v * SPAN)
        return v, lo, hi


@dataclass(frozen=True)
class FormSpec:
    key: str  # short name used in labels and on the command line
    function: str  # SASfit function
    fitted: tuple  # of Par
    fixed: tuple = ()  # ((name, value), ...)
    sd_param: str | None = None  # parameter a size distribution may run over
    sq_ok: bool = False  # a structure factor of spheres is meaningful
    label: str = ""


@dataclass(frozen=True)
class DistSpec:
    key: str
    function: str
    amplitude: str  # linear parameter N
    location: str  # takes the place of the distributed form-factor parameter
    fitted: tuple = ()
    fixed: tuple = ()
    label: str = ""


@dataclass(frozen=True)
class SqSpec:
    key: str
    function: str
    radius: str  # first parameter: interaction radius (set from the particle volume in the local monodisperse approx.)
    fitted: tuple = ()
    fixed: tuple = ()
    label: str = ""


@dataclass(frozen=True)
class Component:
    """One population of a candidate: keys of the form factor, the size distribution and the structure factor."""

    form: str
    sd: str | None = None
    sq: str | None = None

    @property
    def label(self):
        s = FORMS[self.form].label or self.form
        if self.sd:
            s += " + " + (DISTRIBUTIONS[self.sd].label or self.sd)
        if self.sq:
            s += " x " + (STRUCTURE_FACTORS[self.sq].label or self.sq)
        return s


FORMS, DISTRIBUTIONS, STRUCTURE_FACTORS = {}, {}, {}


def register_form(spec):
    FORMS[spec.key] = spec
    return spec


def register_distribution(spec):
    DISTRIBUTIONS[spec.key] = spec
    return spec


def register_structure_factor(spec):
    STRUCTURE_FACTORS[spec.key] = spec
    return spec


# ----------------------------------------------------------------------------- the first candidate list
register_form(FormSpec("sphere", "ff_sphere", (Par("R"),), (("eta", 1.0),), sd_param="R", sq_ok=True, label="sphere"))
register_form(FormSpec(
    "core-shell", "ff_spherical_shell_iii",
    (Par("R", "size", 0.7), Par("dR", "size", 0.3, 0.5, 0.5), Par("eta1", "abs", 0.4, -3.0, 3.0)),
    (("eta2", 1.0),), sd_param="R", sq_ok=True, label="core-shell sphere"))  # fmt: skip
register_form(FormSpec(
    "hollow-sphere", "ff_hollow_sphere", (Par("R", "size", 0.7), Par("dR", "size", 0.3, 0.5, 0.5)),
    (("eta", 1.0),), sd_param="R", sq_ok=True, label="hollow sphere"))  # fmt: skip
register_form(FormSpec(
    "spheroid", "ff_spheroid_R", (Par("R_polar", "size", 1.6), Par("R_equatorial", "size", 0.8)),
    (("eta_core", 1.0), ("eta_solv", 0.0)), label="spheroid"))  # fmt: skip
register_form(FormSpec(
    "cylinder", "ff_cylinder", (Par("R", "size", 0.5), Par("L", "size", 3.0, 1.0, 5.0)), (("eta", 1.0),),
    label="cylinder"))  # fmt: skip
register_form(FormSpec("rod", "ff_rod", (Par("L", "size", 3.0, 1.0, 10.0),), (("eta", 1.0),), label="thin rod"))
register_form(FormSpec("disc", "ff_disc", (Par("R", "size", 1.2, 1.0, 5.0),), (("eta", 1.0),), label="thin disc"))

register_distribution(DistSpec(
    "lognormal", "sd_LogNorm", amplitude="N", location="mu", fitted=(Par("s", "abs", 0.15, 0.02, 0.8),),
    fixed=(("p", 1.0),), label="LogNorm"))  # fmt: skip

register_structure_factor(SqSpec(
    "hard-sphere", "sq_HardSphere", radius="RHS", fitted=(Par("fp", "abs", 0.15, 0.005, 0.55),),
    label="hard sphere (PY)"))  # fmt: skip


def generate(features, forms=None, distributions=None, structure_factor="auto"):
    """All single-component candidates (tuples with one :class:`Component`) for data with these features.

    ``forms`` and ``distributions`` are lists of keys (default: all registered). ``structure_factor`` is
    ``"auto"`` (only if the features suggest one), ``"always"`` or ``"never"``.
    """
    forms = list(FORMS) if forms is None else list(forms)
    dists = list(DISTRIBUTIONS) if distributions is None else list(distributions)
    for key, table, what in [(k, FORMS, "form factor") for k in forms] + [(k, DISTRIBUTIONS, "size distribution") for k in dists]:
        if key not in table:
            raise KeyError(f"unknown {what} '{key}' (known: {', '.join(table)})")
    with_sq = structure_factor == "always" or (structure_factor == "auto" and features.sq_hint)
    if structure_factor not in ("auto", "always", "never"):
        raise ValueError("structure_factor must be 'auto', 'always' or 'never'")
    out = []
    for fk in forms:
        spec = FORMS[fk]
        sds = [None] + (dists if spec.sd_param else [])
        sqs = [None] + (list(STRUCTURE_FACTORS) if with_sq and spec.sq_ok else [])
        for sd in sds:
            for sq in sqs:
                out.append((Component(fk, sd, sq),))
    return out
