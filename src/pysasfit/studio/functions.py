"""Calling SASfit model functions from Python.

    from pysasfit import studio
    sphere = studio.function("ff_sphere")
    I = sphere(q, R=50.0, eta=1.0)            # vectorised over q, named or positional parameters
    F = sphere.amplitude(q, R=50.0, eta=1.0)  # the '_f' function (non-squared amplitude)
    V = sphere.volume(50.0, R=50.0, dist=0)   # the '_v' function

A function is either a compiled SASfit plugin (described by the plugin
headers, see :mod:`pysasfit.studio.headers`) or a Python plugin:

* a file ``sasstudio_<name>.py`` in one of :func:`pysasfit.studio.paths.python_plugin_dirs`
  (format: ``templates/python_plugin.py``), or
* functions registered at run time with :func:`register_python_function`.

Python plugins take precedence, so a C plugin can be overridden for testing.
"""

from __future__ import annotations

import importlib.util
import os
import warnings

import numpy as np

from . import headers, paths
from ._ctypes import FUNC_ONE, FUNC_VOL, MAXPAR, SasfitParam

__all__ = [
    "SASfitError",
    "SASfitFunction",
    "function",
    "register_python_function",
    "unregister_python_function",
]


class SASfitError(RuntimeError):
    """Raised when a SASfit function sets ``param->errStatus``."""


_PYTHON_FUNCTIONS = {}


def register_python_function(name, intensity, amplitude=None, volume=None, parameters=(), label=None, defaults=None):
    """Make a Python function available as SASfit function ``name``.

    ``name`` carries the type prefix: 'ff_...', 'sq_...' or 'sd_...'. The
    callables are ``intensity(x, p)``, ``amplitude(x, p)`` and
    ``volume(x, p, dist)`` with ``p`` the parameter vector (length MAXPAR).
    """
    defaults = list(defaults) if defaults is not None else [0.0] * len(parameters)
    pinfo = [headers.ParameterInfo(n, "", d) for n, d in zip(parameters, defaults)]
    info = headers.PluginInfo(name=name, label=label or name, collection="python", parameters=pinfo)
    _PYTHON_FUNCTIONS[name] = (info, intensity, amplitude, volume)
    return info


def unregister_python_function(name):
    _PYTHON_FUNCTIONS.pop(name, None)


def _load_python_plugin(name, short):
    for d in paths.python_plugin_dirs():
        for stem in (name, short):
            fn = os.path.join(d, f"sasstudio_{stem}.py")
            if os.path.isfile(fn):
                spec = importlib.util.spec_from_file_location(f"pysasfit_studio_plugin_{stem}", fn)
                mod = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(mod)

                def get(kind):
                    return getattr(mod, f"{kind}_{stem}", None) or getattr(mod, kind, None)

                pars = getattr(mod, "PARAMETERS", [])
                names = [p[0] if isinstance(p, (tuple, list)) else p for p in pars]
                defs = [float(p[2]) if isinstance(p, (tuple, list)) and len(p) > 2 else 0.0 for p in pars]
                full = name if "_" in name and name.split("_", 1)[0] in headers.PLUGIN_TYPES else f"ff_{name}"
                info = headers.PluginInfo(
                    name=full,
                    label=getattr(mod, "LABEL", full),
                    collection=os.path.basename(fn),
                    parameters=[headers.ParameterInfo(n, "", v) for n, v in zip(names, defs)],
                    header=fn,
                )
                return (
                    info,
                    get("scattering_intensity") or get("intensity"),
                    get("scattering_amplitude") or get("amplitude"),
                    get("volume"),
                )
    return None


class SASfitFunction:
    """One SASfit model function (form factor, structure factor, size distribution or peak)."""

    def __init__(self, name, library=None, errors="raise"):
        if name in _PYTHON_FUNCTIONS:
            impl = _PYTHON_FUNCTIONS[name]
        else:
            try:
                cinfo = headers.info(name)
            except KeyError:
                cinfo = None
            short = cinfo.short_name if cinfo else name.split("_", 1)[-1]
            impl = _load_python_plugin(cinfo.name if cinfo else name, short)
            if impl is None and cinfo is None:
                raise KeyError(f"no SASfit function '{name}' (try pysasfit.studio.search('{name}'))")
        self.errors = errors
        if impl is not None:
            self.info, self._py_i, self._py_f, self._py_v = impl
            self.language = "python"
        else:
            self.info = cinfo
            self.language = "c"
            from .library import get_library

            lib = library or get_library()
            fi, ff, fv = lib.pointers(cinfo.name, cinfo.collection)
            self._c_i = FUNC_ONE(fi) if fi else None
            self._c_f = FUNC_ONE(ff) if ff else None
            self._c_v = FUNC_VOL(fv) if fv else None

    # -------------------------------------------------------------- params
    @property
    def name(self):
        return self.info.name

    @property
    def parameter_names(self):
        return self.info.parameter_names

    def parameter_vector(self, *values, **named):
        """Full parameter vector p[0..MAXPAR-1]: defaults, then positional values, then named ones."""
        p = np.zeros(MAXPAR)
        d = self.info.defaults
        p[: len(d)] = d
        if len(values) == 1 and np.ndim(values[0]) == 1:
            values = tuple(values[0])
        if len(values) > MAXPAR:
            raise ValueError(f"at most {MAXPAR} parameters")
        p[: len(values)] = values
        names = self.parameter_names
        for k, v in named.items():
            if k in names:
                p[names.index(k)] = v
            elif k.startswith("p") and k[1:].isdigit():
                p[int(k[1:])] = v
            else:
                raise KeyError(f"{self.name} has no parameter '{k}' (parameters: {', '.join(names)})")
        return p

    def _check(self, s):
        err = s.error
        if err:
            s.clear_error()
            if self.errors == "raise":
                raise SASfitError(f"{self.name}: {err}")
            if self.errors == "warn":
                warnings.warn(f"{self.name}: {err}", RuntimeWarning, stacklevel=3)

    def _eval(self, cfunc, pyfunc, x, p, kind):
        x = np.asarray(x, dtype=float)
        out = np.empty(x.shape)
        if self.language == "python":
            if pyfunc is None:
                raise NotImplementedError(f"{self.name} has no {kind} function")
            for i, xi in np.ndenumerate(x):
                out[i] = pyfunc(float(xi), p)
            return out if out.ndim else float(out)
        if cfunc is None:
            raise NotImplementedError(f"{self.name} has no {kind} function")
        s = SasfitParam.from_values(p)
        for i, xi in np.ndenumerate(x):
            out[i] = cfunc(float(xi), s)
            if s.errStatus:
                self._check(s)
        return out if out.ndim else float(out)

    # ------------------------------------------------------------- evaluate
    def __call__(self, x, *values, **named):
        """Intensity I(q) (form factor, structure factor, peak) or weight N(x) (size distribution)."""
        p = self.parameter_vector(*values, **named)
        return self._eval(getattr(self, "_c_i", None), getattr(self, "_py_i", None), x, p, "intensity")

    intensity = __call__

    def amplitude(self, q, *values, **named):
        p = self.parameter_vector(*values, **named)
        return self._eval(getattr(self, "_c_f", None), getattr(self, "_py_f", None), q, p, "amplitude")

    def volume(self, x, *values, dist=0, **named):
        """Volume with parameter ``dist`` replaced by ``x`` (SASfit ``_v`` convention)."""
        p = self.parameter_vector(*values, **named)
        x = np.asarray(x, dtype=float)
        out = np.empty(x.shape)
        if self.language == "python":
            if self._py_v is None:
                raise NotImplementedError(f"{self.name} has no volume function")
            for i, xi in np.ndenumerate(x):
                out[i] = self._py_v(float(xi), p, int(dist))
        else:
            if self._c_v is None:
                raise NotImplementedError(f"{self.name} has no volume function")
            s = SasfitParam.from_values(p)
            for i, xi in np.ndenumerate(x):
                out[i] = self._c_v(float(xi), s, int(dist))
                if s.errStatus:
                    self._check(s)
        return out if out.ndim else float(out)

    def evaluate_p(self, x, p, kind="intensity"):
        """Evaluate with a ready parameter vector ``p`` (used by the model classes)."""
        if kind == "intensity":
            return self._eval(getattr(self, "_c_i", None), getattr(self, "_py_i", None), x, p, kind)
        if kind == "amplitude":
            return self._eval(getattr(self, "_c_f", None), getattr(self, "_py_f", None), x, p, kind)
        raise ValueError(kind)

    def __repr__(self):
        return f"<SASfitFunction {self.name} ({self.language}): {', '.join(self.parameter_names)}>"

    def __str__(self):
        return str(self.info)


def function(name, errors="raise"):
    """Shorthand for :class:`SASfitFunction` (``errors``: 'raise', 'warn' or 'ignore')."""
    return SASfitFunction(name, errors=errors)
