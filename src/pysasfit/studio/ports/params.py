"""Parameter descriptions of the Python ports: domain, default value and kind of every parameter.

Every port has a list of :class:`Param` (``PortSpec.params``) built from three sources, later ones winning:

1. the generated table :mod:`.param_table` (``tools/gen_port_params.py``): names from the plugin headers, bounds
   parsed from the ``SASFIT_CHECK_COND*`` checks of the C sources, kinds guessed from the names;
2. kind defaults (:data:`KIND_DEFAULTS`) for parameters without an explicit default;
3. hand-written defaults and corrections in :mod:`.param_defaults`.

Two things are derived from it:

* :func:`check` - the domain check, with messages as in C (``R(-3) < 0``), once per parameter set;
* :func:`default_vector` / :func:`suggest` - default values, and values suggested from measured data.

Lengths carry ``kind="length"``: their defaults are numbers in the unit reciprocal to q (chosen for q in nm⁻¹);
the explorer can rescale them for other q units. Only the few functions with absolute units say so explicitly
(``unit``).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace

import numpy as np

__all__ = [
    "KIND_DEFAULTS",
    "Param",
    "Rule",
    "check",
    "default_vector",
    "guess_kind",
    "params_from_info",
    "resolve",
    "scale_lengths",
    "suggest",
]

INF = math.inf


@dataclass(frozen=True)
class Param:
    """One parameter of a port."""

    name: str
    default: float | None = None  # None: from the kind (KIND_DEFAULTS) or the bounds
    lower: float = -INF
    upper: float = INF
    lower_open: bool = False  # True: lower bound excluded (x > lower)
    upper_open: bool = False  # True: upper bound excluded (x < upper)
    kind: str = "dimensionless"  # length, sld, intensity, angle, exponent, count, fraction, dimensionless, dummy
    unit: str = (
        ""  # only for parameters with absolute units (e.g. "deg", "g/mol", "g/cm^3")
    )
    doc: str = ""

    @property
    def bounds(self):
        return (self.lower, self.upper)

    def violation(self, value):
        """C-like message if ``value`` is outside the domain, else ''."""
        if self.kind == "dummy":
            return ""
        if not math.isfinite(value) and not (
            math.isinf(value) and (self.upper == INF or self.lower == -INF)
        ):
            return f"{self.name}({value:g}) is not a number"
        if self.lower_open and value <= self.lower:
            return f"{self.name}({value:g}) <= {self.lower:g}"
        if not self.lower_open and value < self.lower:
            return f"{self.name}({value:g}) < {self.lower:g}"
        if self.upper_open and value >= self.upper:
            return f"{self.name}({value:g}) >= {self.upper:g}"
        if not self.upper_open and value > self.upper:
            return f"{self.name}({value:g}) > {self.upper:g}"
        return ""

    def bounds_text(self):
        lo = "(" if self.lower_open or self.lower == -INF else "["
        hi = ")" if self.upper_open or self.upper == INF else "]"
        f = lambda v: "inf" if v == INF else ("-inf" if v == -INF else f"{v:g}")
        return f"{lo}{f(self.lower)}, {f(self.upper)}{hi}"


@dataclass(frozen=True)
class Rule:
    """A condition between parameters: ``test(p)`` is True when the parameter vector is valid."""

    test: object
    message: str


#: default values by kind (lengths in the unit reciprocal to q, chosen for q in nm⁻¹)
KIND_DEFAULTS = {
    "length": 10.0,
    "inverse_length": 0.1,
    "sld": 1.0,
    "intensity": 1.0,
    "angle": 0.0,
    "exponent": 2.0,
    "count": 1.0,
    "fraction": 0.5,
    "dimensionless": 1.0,
    "dummy": 0.0,
}

_LENGTH = (
    "r",
    "rg",
    "l",
    "lb",
    "t",
    "d",
    "a",
    "b",
    "c",
    "xi",
    "dr",
    "h",
    "w",
    "sigma_r",
    "zeta",
    "delta",
)


def guess_kind(name):
    """Kind of a parameter from its name (as used in the SASfit headers)."""
    n = name.strip().lower().lstrip("+-")
    if n.startswith("dummy") or n in ("", "not used"):
        return "dummy"
    if n.startswith(
        ("eta", "delta_eta", "rho", "b_p", "sld", "drho", "beta_")
    ) or n in ("contrast",):
        return "sld"
    if n in ("i0", "i_0", "scale", "c_nu", "c_p", "n", "a_n") or n.startswith(
        ("i0", "scale", "i_")
    ):
        return "intensity" if n not in ("n",) else "count"
    if n.startswith(
        ("psi", "theta", "phi", "varphi", "alpha_deg", "delta_theta", "delta_phi")
    ) or n.endswith("_deg"):
        return "angle"
    if n in (
        "nu",
        "mu",
        "alpha",
        "beta",
        "gamma",
        "d_f",
        "df",
        "dim",
        "s",
        "p",
        "q",
        "m",
        "exponent",
        "vm",
        "xi_exp",
    ):
        return "exponent"
    if n in ("f", "np", "g", "nlayers", "k"):
        return "count"
    if n in ("phi", "fp", "eta_vol", "fraction", "x"):
        return "fraction"
    if (
        n in _LENGTH
        or n.startswith(
            (
                "r_",
                "r1",
                "r2",
                "r3",
                "rg",
                "l_",
                "t_",
                "d_",
                "radius",
                "length",
                "thick",
                "sigma_r",
            )
        )
        or (n.startswith("r") and n[1:].isdigit())
    ):
        return "length"
    return "dimensionless"


def _from_bounds(p):
    """A default inside the bounds when the kind default is outside."""
    d = KIND_DEFAULTS.get(p.kind, 1.0)
    if p.violation(d) == "":
        return d
    lo, hi = p.lower, p.upper
    if math.isfinite(lo) and math.isfinite(hi):
        return 0.5 * (lo + hi)
    if math.isfinite(lo):
        return (
            lo
            + (abs(lo) if lo != 0 else 1.0) * (1.0 if p.lower_open else 0.0)
            + (1.0 if lo == 0 else 0.0)
        )
    if math.isfinite(hi):
        return hi - 1.0
    return d


def resolve(params):
    """Params with every default filled in (explicit, else kind default, else from the bounds)."""
    out = []
    for p in params:
        d = p.default
        if d is None:
            d = _from_bounds(p)
        out.append(replace(p, default=float(d)))
    return out


def default_vector(params, size=50):
    v = np.zeros(size)
    for i, p in enumerate(resolve(params)):
        v[i] = p.default
    return v


def check(params, rules, p):
    """Messages for all violated bounds and rules of the parameter vector ``p`` (empty list: valid)."""
    msgs = []
    for i, prm in enumerate(params):
        if i < len(p):
            m = prm.violation(float(p[i]))
            if m:
                msgs.append(m)
    for r in rules or ():
        try:
            ok = bool(r.test(p))
        except Exception:  # noqa: BLE001 -- an invalid vector for the rule itself
            ok = False
        if not ok:
            msgs.append(r.message_for(p) if hasattr(r, "message_for") else r.message)
    return msgs


def params_from_info(info):
    """Params of any SASfit function from its header info (kinds guessed from the names; header defaults)."""
    out = []
    for prm in info.parameters:
        lo, hi = prm.bounds
        out.append(Param(prm.name, prm.default, lo, hi, kind=guess_kind(prm.name)))
    return out


def scale_lengths(params, p, q):
    """Scale all length parameters of ``p`` together so that the largest is 2/q_mid (in place; returns p)."""
    q = np.asarray(q, float)
    q = q[np.isfinite(q) & (q > 0)]
    if q.size == 0:
        return p
    qmid = math.sqrt(q.min() * q.max())
    lengths = [
        i
        for i, prm in enumerate(params)
        if i < len(p) and prm.kind == "length" and p[i] > 0
    ]
    if not lengths:
        return p
    s = (2.0 / qmid) / max(p[i] for i in lengths)
    for i, prm in enumerate(params):
        if i < len(p) and prm.kind in ("length", "inverse_length") and p[i] != 0:
            v = p[i] * s if prm.kind == "length" else p[i] / s
            if prm.violation(v) == "":
                p[i] = v
    return p


def suggest(spec, q, intensity=None, base=None):
    """Parameter vector suggested for data at ``q`` (and, if given, ``intensity``).

    Lengths are scaled together so that the largest one is about 2/q_mid (q_mid: geometric mean of the q range);
    then, if intensities are given, one intensity-like parameter (I0/scale, else the first sld, which enters
    squared) is scaled so that the model matches the median of the lowest tenth of the data.
    """
    params = resolve(spec.params)
    p = default_vector(params) if base is None else np.array(base, float)
    q = np.asarray(q, float)
    if intensity is not None and np.shape(intensity) == q.shape:
        keep = np.isfinite(q) & (q > 0)
        q, intensity = q[keep], np.asarray(intensity, float)[keep]
    else:
        q = q[np.isfinite(q) & (q > 0)]
    if q.size == 0:
        return p
    scale_lengths(params, p, q)
    if intensity is not None:
        y = np.asarray(intensity, float)
        qq = np.asarray(q, float)
        if y.shape == qq.shape:
            low = qq <= np.quantile(qq, 0.1)
            target = np.median(y[low & np.isfinite(y)]) if np.any(low) else np.nan
            try:
                model = np.median(np.asarray(spec(qq[low], p), float))
            except Exception:  # noqa: BLE001 -- keep the unscaled vector
                model = np.nan
            if np.isfinite(target) and np.isfinite(model) and model != 0 and target > 0:
                ratio = target / model
                amp = [i for i, prm in enumerate(params) if prm.kind == "intensity"]
                sld = [i for i, prm in enumerate(params) if prm.kind == "sld"]
                if amp:
                    p[amp[0]] *= ratio
                elif sld and ratio > 0:
                    for i in sld:
                        p[i] *= math.sqrt(ratio)
    return p


@dataclass
class ParamSet:
    """Params and rules of one port (as stored in the generated table)."""

    params: list = field(default_factory=list)
    rules: list = field(default_factory=list)
