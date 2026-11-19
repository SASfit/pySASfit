"""Python ports of SASfit C plugin functions.

A port is a vectorised numpy implementation of a SASfit plugin function with the same parameter vector::

    from pysasfit.studio import ports
    spec = ports.get("ff_deformed_generalized_gaussian_chain")
    I = spec.intensity(q, p)              # p: parameter vector as in SASfit (p[0], p[1], ...)
    I2 = spec.intensity(q, p, psi=psi)    # ψ-dependent functions: ψ (rad) broadcast against q

Every port is also available under its own name, ``ff_py_<name without ff_>`` (e.g.
``ff_py_sheared_cylinders__maier_saupe_``), next to the C function: :func:`register_aliases` adds them to the
function catalogue in the collection ``"py plugin ports"``. The model explorer shows them in the form-factor menu
under "py plugin ports", a copy of the group tree of the SASfit plugins::

    from pysasfit.studio import function
    f_py = function("ff_py_reptating_chain")   # Python port
    f_c = function("ff_reptating_chain")       # C plugin

Alternatively the ports can replace the C plugins under the original names::

    ports.enable()                         # all ports (or ports.enable(["ff_..."]))
    ports.disable()

The C plugins stay the reference: :mod:`pysasfit.studio.ports.check` compares every port with the C function
of the SASfit library (``python -m pysasfit.studio.ports.check``).
"""

from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass, field

import numpy as np

__all__ = [
    "PORTS_COLLECTION",
    "PortSpec",
    "alias",
    "disable",
    "enable",
    "enabled",
    "get",
    "names",
    "original",
    "register_aliases",
    "specs",
]

PORTS_COLLECTION = (
    "py plugin ports"  # collection (and menu tree) of the ff_py_* functions
)
_PY = "py_"


@dataclass
class PortSpec:
    """Description of one ported function."""

    name: str  # SASfit function name, e.g. 'ff_deformed_generalized_gaussian_chain'
    intensity: object  # intensity(q, p, psi=None) -> array like q
    parameters: list  # parameter names (p[0], p[1], ...)
    psi_index: int | None = (
        None  # index of the ψ parameter (Q direction on the detector, deg) or None
    )
    psi_unit: str = "deg"
    amplitude: object = None  # amplitude(q, p, psi=None) or None
    volume: object = None  # volume(x, p, dist) or None
    sample: dict = field(
        default_factory=dict
    )  # index -> (low, high, 'lin'|'log') for the comparison with C
    rtol: float = 1e-8  # accepted relative deviation from the C function
    source: str = ""  # C source file (relative to the SASfit src folder)
    notes: str = ""
    reentrant: bool = True  # pure function of (q, p, psi): may be evaluated in parallel
    reduced: object = None  # reduced(q, p) -> dimensionless scale such as q Rg, used in the comparison report
    reduced_label: str = "q Rg"
    c_integrates: bool = False  # the C function uses sasfit_integrate (rel. precision = SASfit's eps "aniso")
    c_compat: bool = False  # intensity(..., c_compat=True) reproduces known deviations of the C code (comparison)
    label: str = ""  # menu label for a port without an installed header
    group: str = ""  # menu group (Doxygen group of the plugin header) for a port without an installed header
    c_valid: object = None  # c_valid(q, p) -> bool mask: where the C function is accurate (comparison only there)
    c_reference: bool = True  # False: the C plugin is known to be wrong, the port is not compared with it

    @property
    def params(self):
        """:class:`.params.Param` of every parameter (generated table + kind defaults + param_defaults.py)."""
        cache = self.__dict__.get("_params")
        if cache is None:
            cache = self.__dict__["_params"] = _build_params(self)
        return cache[0]

    @property
    def rules(self):
        self.params  # noqa: B018 -- builds both
        return self.__dict__["_params"][1]

    def check(self, p):
        """Messages for the parameters of ``p`` outside their domain (empty: valid)."""
        from .params import check

        return check(self.params, self.rules, np.asarray(p, float))

    def validate(self, p):
        """Raise ValueError (C-like messages) if ``p`` is outside the domain."""
        msgs = self.check(p)
        if msgs:
            raise ValueError(f"{self.name}: " + "; ".join(msgs))

    def defaults(self, size=50):
        """Default parameter vector."""
        from .params import default_vector

        return default_vector(self.params, size)

    def suggest(self, q, intensity=None, base=None):
        """Parameter vector suggested for data at ``q`` (and ``intensity``), see :func:`.params.suggest`."""
        from .params import suggest

        return suggest(self, q, intensity, base)

    @property
    def anisotropic(self):
        return self.psi_index is not None

    def __call__(self, q, p, psi=None):
        return self.intensity(q, np.asarray(p, float), psi)


def _build_params(spec):
    from . import param_defaults
    from .params import Param, guess_kind, resolve

    try:
        from .param_table import TABLE
    except ImportError:  # not generated yet
        TABLE = {}
    entry = TABLE.get(spec.name)
    if entry and len(entry["params"]) == len(spec.parameters):
        params = [
            Param(n, None, lo, hi, lo_open, hi_open, kind)
            for n, lo, lo_open, hi, hi_open, kind in entry["params"]
        ]
    else:
        params = [Param(n, kind=guess_kind(n)) for n in spec.parameters]
    over = param_defaults.OVERRIDES.get(spec.name, {})
    out = []
    for i, prm in enumerate(params):
        fields = {}
        if prm.name in param_defaults.BY_NAME:
            fields.update(param_defaults.BY_NAME[prm.name])
        fields.update(over.get(prm.name, {}))
        fields.update(over.get(i, {}))
        out.append(dataclasses.replace(prm, **fields) if fields else prm)
    if spec.psi_index is not None and spec.psi_index < len(out):
        out[spec.psi_index] = dataclasses.replace(
            out[spec.psi_index], kind="angle", unit=spec.psi_unit
        )
    rules = []
    for expr, msg, arg in (entry or {}).get("rules", []):
        try:
            bad = eval(
                f"lambda p: {expr}", {"__builtins__": {"abs": abs}, "math": math}
            )
            value = (
                eval(f"lambda p: ({arg})", {"__builtins__": {"abs": abs}, "math": math})
                if arg
                else None
            )
        except SyntaxError:
            continue

        def ok(p, bad=bad):
            return not bad(p)

        text = msg
        if value is not None and "%" in msg:
            text = (
                msg.replace("%lg", "{:g}").replace("%lf", "{:g}").replace("%g", "{:g}")
            )
        rules.append(_MsgRule(ok, text, value))
    rules += list(param_defaults.RULES.get(spec.name, []))
    return resolve(out), rules


class _MsgRule:
    """A rule from the C checks; the message gets the value of the checked expression (as the C format)."""

    def __init__(self, test, text, value):
        self._test, self._text, self._value = test, text, value

    def test(self, p):
        try:
            return self._test(p)
        except (IndexError, NameError, TypeError):
            return True  # the C expression refers to something the port does not have

    @property
    def message(self):
        return self._text.replace("{:g}", "?")

    def message_for(self, p):
        if self._value is None or "{:g}" not in self._text:
            return self._text
        try:
            return self._text.format(float(self._value(p)))
        except Exception:  # noqa: BLE001
            return self.message


_REGISTRY: dict[str, PortSpec] = {}
_ENABLED: set[str] = set()


def _register_module(mod):
    for d in mod.SPECS:
        s = PortSpec(**d)
        _REGISTRY[s.name] = s


def _load():
    if _REGISTRY:
        return
    from . import (
        deformed_polymers,
        fractals,
        guwelop,
        micelles,
        misc,
        networks,
        nonparticulate,
        pearls,
        polymers,
        reptation,
        ribbon,
        sheared,
        shearflow,
        spheres,
    )

    for mod in (
        deformed_polymers,
        networks,
        reptation,
        shearflow,
        sheared,
        spheres,
        polymers,
        misc,
        nonparticulate,
        pearls,
        micelles,
        fractals,
        guwelop,
        ribbon,
    ):
        _register_module(mod)


def specs():
    _load()
    return list(_REGISTRY.values())


def names():
    _load()
    return sorted(_REGISTRY)


def alias(name):
    """Name of the Python port of the SASfit function ``name``: ``ff_x`` -> ``ff_py_x``."""
    kind, rest = name.split("_", 1)
    return f"{kind}_{_PY}{rest}"


def original(name):
    """The SASfit name of an alias ``ff_py_x`` (``ff_x``); other names unchanged."""
    kind, _, rest = name.partition("_")
    return f"{kind}_{rest[len(_PY) :]}" if rest.startswith(_PY) else name


def get(name):
    _load()
    try:
        return _REGISTRY[original(name)]
    except KeyError:
        raise KeyError(
            f"no Python port of '{name}' (ports: {', '.join(names())})"
        ) from None


def _wrap(spec):
    def intensity(x, p):
        p = np.asarray(p, float)
        spec.validate(p)
        return spec.intensity(np.asarray(x, float), p)

    intensity.vectorized = True
    intensity.port = spec
    return intensity


def _wrap_amplitude(spec):
    if spec.amplitude is None:
        return None

    def amplitude(x, p):
        return spec.amplitude(np.asarray(x, float), np.asarray(p, float))

    amplitude.vectorized = True
    amplitude.port = spec
    return amplitude


def _info(spec, name, collection, label_suffix=""):
    from .. import headers

    try:
        base = headers.info(spec.name)
    except KeyError:
        base = headers.PluginInfo(
            name=spec.name,
            label=spec.label
            or spec.name.split("_", 1)[1].replace("__", ": ").replace("_", " ").strip(),
            collection=collection,
            group=spec.group,
            parameters=[headers.ParameterInfo(n, "", 0.0) for n in spec.parameters],
        )
    # defaults and bounds from the parameter description of the port, texts from the header
    pars = list(base.parameters)
    for i, prm in enumerate(spec.params):
        desc = pars[i].description if i < len(pars) else prm.doc
        info = headers.ParameterInfo(prm.name if i >= len(pars) else pars[i].name, desc or prm.doc, prm.default,
                                     (prm.lower, prm.upper))  # fmt: skip
        if i < len(pars):
            pars[i] = info
        else:
            pars.append(info)
    return dataclasses.replace(
        base,
        name=name,
        label=base.label + label_suffix,
        collection=collection,
        group=base.group or spec.group,
        parameters=pars,
        brief=(f"Python port of {spec.name}. " + (base.brief or "")).strip()
        if name != spec.name
        else base.brief,
    )


def register_aliases():
    """Register every port as ``ff_py_<name>`` (collection :data:`PORTS_COLLECTION`); returns the names."""
    from .. import functions

    out = []
    for spec in specs():
        a = alias(spec.name)
        if a not in functions._PYTHON_FUNCTIONS:
            functions._PYTHON_FUNCTIONS[a] = (
                _info(spec, a, PORTS_COLLECTION, " (py)"),
                _wrap(spec),
                _wrap_amplitude(spec),
                spec.volume,
            )
        out.append(a)
    return out


def enable(which=None):
    """Use the Python ports instead of the C plugins (all ports, or the names in ``which``)."""
    from .. import functions

    for name in names() if which is None else which:
        spec = get(name)
        info = _info(spec, name, "python port")
        functions._PYTHON_FUNCTIONS[name] = (
            info,
            _wrap(spec),
            _wrap_amplitude(spec),
            spec.volume,
        )
        _ENABLED.add(name)


def disable(which=None):
    """Back to the C plugins (all ports, or the names in ``which``)."""
    from .. import functions

    for name in list(_ENABLED) if which is None else which:
        entry = functions._PYTHON_FUNCTIONS.get(name)
        if entry is not None and getattr(entry[1], "port", None) is not None:
            del functions._PYTHON_FUNCTIONS[name]
        _ENABLED.discard(name)


def enabled():
    return sorted(_ENABLED)


def sample_parameters(spec, n, seed=0, base=None):
    """``n`` random parameter vectors (length 50) within ``spec.sample``; other entries from ``base``."""
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        p = np.zeros(50) if base is None else np.array(base, float)
        for i, (lo, hi, kind) in spec.sample.items():
            if kind == "log":
                p[i] = math.exp(rng.uniform(math.log(lo), math.log(hi)))
            else:
                p[i] = rng.uniform(lo, hi)
        out.append(p)
    return out
