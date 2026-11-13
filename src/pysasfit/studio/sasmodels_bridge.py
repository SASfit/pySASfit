"""The models of SasView (the ``sasmodels`` package, ``pip install sasmodels``) as pySASstudio functions.

Every sasmodels model becomes a function next to the SASfit plugins: ``ff_sv_<name>`` for form factors and
shape-independent models, ``sq_sv_<name>`` for structure factors (menu groups "SasView: <category>"). They take part
in everything the studio does with SASfit functions: mixtures, size distributions (the studio integrates over a
parameter itself; sasmodels' polydispersity is not used), structure factors with the decoupling approximation or the
local monodisperse approximation, fits, SESANS / MSAS transforms, projects and "Copy Python".

Convention (SASfit, chosen 2026-11-13): a form factor is the intensity of one particle,

    P(q) = (Δρ V F(q))²  = ⟨F²⟩ of sasmodels  = V · I_sasview(q; scale = 1, background = 0),

not SasView's per-volume intensity I_sasview = scale/V · (Δρ V F)². So with a size distribution N(x) the integral
∫ N(x) P(q; x) dx weights the sizes as for the SASfit plugins (V², not V), and N is a number density. The
amplitude is ⟨F⟩ (decoupling approximation), the volume the form volume of sasmodels (LMA). Models without an
amplitude in sasmodels (e.g. the shape-independent ones) have V = 1 there, so P = I_sasview.

Units: sasmodels works in Å (q in 1/Å, lengths in Å, SLDs in 10⁻⁶/Å², I in 1/cm). The studio's functions carry no
units, so :func:`set_length_unit` says in which length unit q and the length parameters are given ("nm" for q in
1/nm, as the explorer's q unit); the bridge converts q, every parameter with a length unit (Å, Å², Å³) and the
volume. SLD parameters stay in 10⁻⁶/Å² (the unit SasView uses). Then P(q) = 10⁻⁴ (Δρ V F)² in cm⁻¹ · [length]³:
a number density N in 1/[length]³ gives I = N P in 1/cm.

Parameters: sasmodels' ``scale`` and ``background`` are fixed at 1 and 0 (the studio has its own). Vector parameters
(``sld[n]``, ``thickness[n]`` of core_multi_shell) become numbered parameters ``sld1`` … ``sld10``; orientation angles
(2D only) and magnetic parameters are left out. Models with more parameters than SASfit's MAXPAR (50) are skipped.
SasView's size distributions (gaussian, lognormal, schulz, rectangle, uniform, boltzmann of sasmodels.weights) are
``sd_sv_<name>`` with the parameters N, center, width (σ/center) and nsigmas (see :data:`DISTRIBUTIONS`).
The C kernel of a model is compiled on first use (sasmodels needs a C compiler; on Windows the ``tinycc`` package or
gcc of MSYS2).
"""

from __future__ import annotations

import threading
from collections import OrderedDict

import numpy as np

__all__ = ["available", "register_all", "set_length_unit", "length_unit", "model_names", "PREFIX_FF", "PREFIX_SQ",
           "PREFIX_SD", "COLLECTION", "DISTRIBUTIONS", "register_distributions", "sasview_distribution"]  # fmt: skip

PREFIX_FF, PREFIX_SQ, PREFIX_SD = "ff_sv_", "sq_sv_", "sd_sv_"
COLLECTION = "sasmodels"
_ANGSTROM_PER = {"A": 1.0, "Å": 1.0, "nm": 10.0, "m": 1e10, "um": 1e4}  # Å per unit
_unit = "nm"
_lock = threading.RLock()
_registered = {}  # function name -> _Model
_kernels = OrderedDict()  # (model name, q bytes) -> kernel


def available():
    """True if the ``sasmodels`` package can be imported."""
    try:
        import sasmodels.core  # noqa: F401
    except Exception:  # noqa: BLE001 -- not installed or broken
        return False
    return True


def set_length_unit(unit):
    """Length unit of q (q in 1/unit) and of the length parameters of the SasView models: "nm", "A" (Å), "m"."""
    global _unit
    unit = {"Å": "A", "1/nm": "nm", "1/Å": "A", "1/m": "m"}.get(unit, unit)
    if unit not in _ANGSTROM_PER:
        raise ValueError(f"unknown length unit '{unit}' (nm, A, m)")
    if unit != _unit:
        _unit = unit
        with _lock:
            _kernels.clear()


def length_unit():
    return _unit


def _c():
    """Å per user length unit."""
    return _ANGSTROM_PER[_unit]


def _length_power(units):
    """Power of the length in a sasmodels unit string ('Ang' 1, 'Ang^2' 2, '1/Ang' -1 ...); 0 otherwise.
    SLD units (1e-6/Ang^2) are kept as they are."""
    u = (units or "").replace(" ", "")
    if "1e-6/Ang^2" in u or "1e-6/Ang" in u or "/Ang^2" in u and u.startswith("1e"):
        return 0
    table = {"Ang": 1, "Ang^2": 2, "Ang^3": 3, "1/Ang": -1, "1/Ang^2": -2, "Ang^-1": -1}
    return table.get(u, 0)


class _Model:
    """One sasmodels model: parameter mapping, lazily compiled kernel, P(q), amplitude, volume."""

    def __init__(self, info):
        self.info = info
        self.name = info.id
        self.is_sq = bool(info.structure_factor) or info.category == "structure-factor"
        self.params = []  # (studio name, sasmodels name, index or None, units, default, limits, description)
        for p in info.parameters.kernel_parameters:
            if p.type == "orientation" or p.type == "magnetic":
                continue
            base = p.name.split("[")[0]
            if p.length > 1:
                for i in range(p.length):
                    self.params.append((f"{base}{i + 1}", base, i, p.units, p.default, p.limits, p.description))
            else:
                self.params.append((p.name, p.name, None, p.units, p.default, p.limits, p.description))
        self._model = None
        self._build_lock = threading.Lock()

    # ------------------------------------------------------------------ build
    def model(self):
        if self._model is None:
            with self._build_lock:
                if self._model is None:
                    from sasmodels import core

                    self._model = core.build_model(self.info)
        return self._model

    def kernel(self, q_angstrom):
        key = (self.name, q_angstrom.tobytes())
        with _lock:
            k = _kernels.get(key)
            if k is not None:
                _kernels.move_to_end(key)
                return k
        k = self.model().make_kernel([q_angstrom])
        with _lock:
            _kernels[key] = k
            while len(_kernels) > 64:
                old = _kernels.popitem(last=False)[1]
                try:
                    old.release()
                except Exception:  # noqa: BLE001
                    pass
        return k

    def pars(self, p):
        """sasmodels parameter dict from the studio's parameter vector p (lengths converted to Å)."""
        c = _c()
        out = {}
        for j, (_n, base, idx, units, _d, _lim, _desc) in enumerate(self.params):
            v = float(p[j]) * c ** _length_power(units)
            if idx is None:
                out[base] = v
            else:
                out[f"{base}{idx + 1}"] = v  # sasmodels names vector entries sld1, sld2, ...
        return out

    # --------------------------------------------------------------- evaluate
    def _fq(self, q, p):
        from sasmodels.direct_model import call_Fq

        q = np.atleast_1d(np.asarray(q, float))
        qa = q / _c()  # 1/unit -> 1/Å
        k = self.kernel(qa)
        return call_Fq(k, self.pars(p))  # F1, F2, R_eff, V_shell, V_ratio (Å units)

    def intensity(self, q, p):
        q = np.asarray(q, float)
        if self.is_sq:
            from sasmodels.direct_model import call_kernel

            qa = np.atleast_1d(q) / _c()
            val = call_kernel(self.kernel(qa), {**self.pars(p), "scale": 1.0, "background": 0.0})
        else:
            _F1, F2, _r, _v, _vr = self._fq(q, p)
            val = np.asarray(F2, float) / _c() ** 3  # (Δρ V F)²: V² in the user length unit … per V in Å³ → unit³
        return val.reshape(q.shape) if q.ndim else float(np.asarray(val).ravel()[0])

    def amplitude(self, q, p):
        q = np.asarray(q, float)
        F1, _F2, _r, _v, _vr = self._fq(q, p)
        if F1 is None:
            raise NotImplementedError(f"sasmodels model {self.name} has no amplitude F(q)")
        val = np.asarray(F1, float) / _c() ** 1.5
        return val.reshape(q.shape) if q.ndim else float(np.asarray(val).ravel()[0])

    def volume(self, x, p, dist):
        """Form volume (user length unit³) with parameter ``dist`` replaced by x (SASfit's '_v' convention)."""
        x = np.atleast_1d(np.asarray(x, float))
        out = np.empty(x.shape)
        pp = np.array(p, float)
        for i, xi in enumerate(x):
            if 0 <= dist < len(self.params):
                pp[dist] = xi
            _F1, _F2, _r, V, _vr = self._fq(np.array([1e-3 * _c()]), pp)
            out[i] = float(V) / _c() ** 3
        return out


def _make_info(m):
    from . import headers

    unit = {"A": "Å"}.get(_unit, _unit)
    pinfo = []
    for name, _base, _idx, units, default, limits, desc in m.params:
        pw = _length_power(units)
        d = float(default) / _c() ** pw if pw else float(default)
        lo, hi = (float(limits[0]), float(limits[1])) if limits else (-np.inf, np.inf)
        if pw:
            lo, hi = lo / _c() ** pw if np.isfinite(lo) else lo, hi / _c() ** pw if np.isfinite(hi) else hi
        shown = units.replace("Ang", unit) if pw else units
        pinfo.append(headers.ParameterInfo(name, f"{desc} [{shown}]" if shown else desc, d, (lo, hi)))
    prefix = PREFIX_SQ if m.is_sq else PREFIX_FF
    title = " ".join((m.info.title or "").split())
    return headers.PluginInfo(name=prefix + m.name, label=f"SasView: {m.name}", collection=COLLECTION,
                              group=f"sasview:{m.info.category or 'other'}", brief=title, parameters=pinfo)  # fmt: skip


def model_names():
    """The model names of the installed sasmodels package ([] if it is not installed)."""
    if not available():
        return []
    from sasmodels import core

    return list(core.list_models())


def register_all(unit=None):
    """Register every sasmodels model as a studio function (again after a change of the length unit, so the
    defaults and bounds are shown in it). Returns the names registered; [] without sasmodels."""
    from . import functions

    if unit is not None:
        set_length_unit(unit)
    if not available():
        return register_distributions()
    from sasmodels import core

    names = []
    for mname in core.list_models():
        try:
            info = core.load_model_info(mname)
            m = _Model(info)
        except Exception:  # noqa: BLE001 -- a model that cannot be loaded is left out
            continue
        if len(m.params) > 50:
            continue
        pi = _make_info(m)
        vec = lambda f: setattr(f, "vectorized", True) or f  # noqa: E731
        amp = None if m.is_sq else vec(lambda q, p, m=m: m.amplitude(q, p))
        volf = None if m.is_sq else vec(lambda x, p, dist, m=m: m.volume(x, p, dist))
        functions._PYTHON_FUNCTIONS[pi.name] = (pi, vec(lambda q, p, m=m: m.intensity(q, p)), amp, volf)
        _registered[pi.name] = m
        names.append(pi.name)
    return names + register_distributions()


def is_sasview_name(name):
    return str(name).startswith((PREFIX_FF, PREFIX_SQ, PREFIX_SD))


# --------------------------------------------------------------------------- size distributions (sasmodels.weights)
# SasView's polydispersity distributions as studio size distributions sd_sv_<name>. Parameters as in SasView:
# center (the value of the distributed parameter), width = σ / center (the relative "PD" of SasView), nsigmas (where
# SasView cuts the distribution: its defaults), plus the amplitude N (number density, as SASfit's distributions).
# The formulas are those of sasmodels.weights; like SasView, the distribution is cut at center ± nsigmas·σ and at
# x > 0, and normalised on what is left (∫ N(x) dx = N). The studio integrates over it with its own integration
# settings (SasView sums over npts points instead). "array" (an empirical table) is not offered.
DISTRIBUTIONS = {  # name: (label, default nsigmas, description)
    "gaussian": ("SasView: gaussian", 3.0, "Gaussian, σ = width · center, cut at ± nsigmas σ"),
    "lognormal": ("SasView: lognormal", 8.0, "log-normal with median center and σ_ln = width (σ/center), cut at ± nsigmas σ"),
    "schulz": ("SasView: schulz", 8.0, "Schulz (gamma) with mean center and z = 1/width², cut at ± nsigmas σ"),
    "rectangle": ("SasView: rectangle", 3**0.5, "uniform over center ± √3 σ (σ = width · center)"),
    "uniform": ("SasView: uniform", 1.0, "uniform over center ± σ (σ = width · center)"),
    "boltzmann": ("SasView: boltzmann", 3.0, "exp(-|x - center|/σ), cut at ± nsigmas σ"),
}


def _sd_shape(name, x, c, sig):
    """The unnormalised density of sasmodels.weights (same formulas)."""
    x = np.asarray(x, float)
    if name == "gaussian":
        return np.exp((x - c) ** 2 / (-2.0 * sig * sig))
    if name == "lognormal":
        s = abs(sig / c)
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(x > 0, np.exp(-0.5 * ((np.log(np.where(x > 0, x, 1.0)) - np.log(c)) / s) ** 2) / (np.where(x > 0, x, 1.0) * s), 0.0)
    if name == "schulz":
        from scipy.special import gammaln

        z = (c / sig) ** 2
        with np.errstate(divide="ignore", invalid="ignore"):
            R = np.where(x > 0, x / c, 1.0)
            arg = z * np.log(z) + (z - 1) * np.log(R) - R * z - np.log(c) - gammaln(z)
            return np.where(x > 0, np.exp(arg), 0.0)
    if name == "boltzmann":
        return np.exp(-np.abs(x - c) / abs(sig))
    if name in ("rectangle", "uniform"):
        return np.ones_like(x)
    raise KeyError(name)


def _sd_support(name, c, sig, nsig):
    half = {"rectangle": 3**0.5 * abs(sig), "uniform": abs(sig)}.get(name, nsig * abs(sig))
    lo, hi = c - half, c + half
    floor = 1e-8 if name in ("lognormal", "schulz") else 0.0
    return max(lo, floor), max(hi, floor)


def _sd_norm(name, c, sig, lo, hi):
    """∫_lo^hi of the shape (closed forms)."""
    from scipy import special

    if hi <= lo:
        return 0.0
    if name == "gaussian":
        return sig * np.sqrt(2 * np.pi) * (special.ndtr((hi - c) / sig) - special.ndtr((lo - c) / sig))
    if name == "lognormal":
        s = abs(sig / c)
        return np.sqrt(2 * np.pi) * (special.ndtr(np.log(hi / c) / s) - special.ndtr(np.log(lo / c) / s))
    if name == "schulz":
        z = (c / sig) ** 2
        return special.gammainc(z, z * hi / c) - special.gammainc(z, z * lo / c)
    if name == "boltzmann":
        s = abs(sig)

        def cdf(t):  # ∫_-inf^t exp(-|x-c|/s) dx
            return s * np.exp((t - c) / s) if t < c else 2 * s - s * np.exp(-(t - c) / s)

        return cdf(hi) - cdf(lo)
    return hi - lo  # rectangle, uniform


def sasview_distribution(name, x, N=1.0, center=1.0, width=0.1, nsigmas=None):
    """N(x) of SasView distribution ``name``: N times the density cut at center ± nsigmas σ (x > 0), normalised."""
    if name not in DISTRIBUTIONS:
        raise KeyError(f"unknown SasView distribution '{name}' ({', '.join(DISTRIBUTIONS)})")
    x = np.asarray(x, float)
    nsig = DISTRIBUTIONS[name][1] if nsigmas is None or nsigmas <= 0 else float(nsigmas)
    sig = abs(width) * abs(center)
    if sig == 0 or center == 0:
        return np.zeros_like(x)  # a delta distribution: give the size directly instead (no distribution)
    lo, hi = _sd_support(name, center, sig, nsig)
    Z = _sd_norm(name, center, sig, lo, hi)
    if not Z > 0:
        return np.zeros_like(x)
    inside = (x >= lo) & (x <= hi)
    return np.where(inside, N * _sd_shape(name, x, center, sig) / Z, 0.0)


def register_distributions():
    """Register the SasView size distributions as ``sd_sv_<name>`` (no sasmodels package needed: the formulas are
    here). Returns the names."""
    from . import functions, headers

    names = []
    for dname, (label, nsig, desc) in DISTRIBUTIONS.items():
        pinfo = [
            headers.ParameterInfo("N", "number density (amplitude: ∫ N(x) dx = N)", 1.0, (0.0, np.inf)),
            headers.ParameterInfo("center", "center of the distribution (SasView: the parameter value)", 10.0, (0.0, np.inf)),
            headers.ParameterInfo("width", "relative width σ/center (SasView: PD)", 0.1, (0.0, np.inf)),
            headers.ParameterInfo("nsigmas", f"cut at ± nsigmas σ (SasView default {nsig:g}; 0 = default)", nsig, (0.0, np.inf)),
        ]
        info = headers.PluginInfo(name=PREFIX_SD + dname, label=label, collection=COLLECTION,
                                  group="sasview:polydispersity", brief=desc, parameters=pinfo)  # fmt: skip

        def dens(x, p, dname=dname):
            return sasview_distribution(dname, x, p[0], p[1], p[2], p[3])

        def support(p, dname=dname):  # the exact cut: no integration panel straddles it
            sig = abs(p[2]) * abs(p[1])
            nsig = DISTRIBUTIONS[dname][1] if p[3] <= 0 else float(p[3])
            return _sd_support(dname, float(p[1]), sig, nsig)

        dens.vectorized = True
        dens.support = support
        functions._PYTHON_FUNCTIONS[info.name] = (info, dens, None, None)
        names.append(info.name)
    return names
