# SPDX-License-Identifier: BSD-3-Clause
"""SANSPOL: correction of half-polarised data (polariser and flipper before the sample, no spin
analysis).

Measured states, one reduced data set each (2D image or 1D curve, fully corrected: background, cell,
transmission, efficiency and absolute scale applied to every state as for unpolarised data):

    polariser out                       ->  I_u   = (I+ + I-)/2
    polariser in,  flipper off          ->  I_off = n+_off I+ + n-_off I-
    polariser in,  flipper on           ->  I_on  = n+_on  I+ + n-_on  I-

I+ and I- are the cross sections for a fully polarised incident beam with the neutron spin parallel
(+) and antiparallel (-) to the field at the sample, summed over the final spin states:
I± = I±± + I±∓. With P the beam polarisation, f the flipped fraction of the flipper (flipper
"efficiency" F = 2f - 1) and r the depolarisation factor of the sample:

    Pe     = P r,      r = (1 - P_f) / (-ln P_f)          (Quan et al., IUCrJ 8 (2021) 455; r = 1 for P_f = 1)
    n+_off = (1 + Pe)/2,  n-_off = (1 - Pe)/2
    n-_on  = f n+_off + (1 - f) n-_off,  n+_on = 1 - n-_on       (exact)
    n-_on  = f n+_off,                   n+_on = 1 - n-_on       ("bersans": Keiderling et al.,
                                                                  Meas. Sci. Technol. 19 (2008) 034009, eq. 3b)

``flipper_off_is_plus = False`` exchanges the roles of + and -. P_f is the polarisation measured after
the sample in transmission (needs an analyser); without it, P_f = 1.

Inversion (two of the three states):

    I+ - I- = (I_off - I_on) / (f Pe)                     (exact form)
    I_u     = I_off - (I_off - I_on) / (2 f)              (independent of P)
    I±      = I_u ± (I+ - I-)/2

With all three states the two unknowns follow per pixel from weighted least squares (weights from
the statistical variance), and chi² per pixel (one degree of freedom) checks the consistency of P, f
and the scale of the polariser-out data. Without an analyser **P and the magnetic contrast are
degenerate**: only P (I+ - I-) is measured, so P must come from the metadata, from the instrument
table (SANS-I: Aswal et al., NIM A 586 (2008) 86) or from a separate measurement. f is the only
efficiency that can be determined from the data (:func:`flipped_fraction_from_data`, needs all three
states on the same absolute scale).

The correction is linear and is applied to the fully corrected data, never to raw counts
(Keiderling et al. 2008). Uncertainties: the input UArrays keep all their sources (sources shared by
the states, e.g. the same background or water, are counted once); P, f and P_f enter as systematic
scalar sources (group ``polarisation``) when their uncertainty is known, otherwise they are not
propagated and a note says so.

Labels: + / - mean the neutron **spin** parallel / antiparallel to the field (Moon, Riste & Koehler, Phys. Rev.
181, 920 (1969)); then I+ - I- = +4 N·M for nuclear and magnetic amplitudes of the same sign. Michels, SASfit and
part of the magnetic-SANS literature label by the neutron moment (antiparallel to the spin): there "++" is our "--"
(Quan et al., J. Phys. Conf. Ser. 1316 (2019) 012010).
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field, replace
from typing import Dict, List, Optional, Sequence, Union

import numpy as np

from .data.uarray import UArray, constant

__all__ = ["PolarisationSetup", "depolarisation_factor", "state_weights", "sanspol_forward",
           "SanspolResult", "sanspol_correct", "flipped_fraction_from_data", "beam_polarisation_table",
           "setup_from_metadata", "correct_reduced", "correct_files", "STATES"]

#: measured states: polariser out; polariser in with flipper off; polariser in with flipper on
STATES = ("unpolarised", "off", "on")

Number = Union[float, UArray]


# ------------------------------------------------------------------------- helpers
def _v(x):
    """Value of a float, array or UArray."""
    return x.value if isinstance(x, UArray) else np.asarray(x, float)


def _param(value, sigma, key, notes: Optional[List[str]]):
    if isinstance(value, UArray):
        return value
    if sigma is None or sigma == 0:
        if notes is not None and sigma is None:
            notes.append(f"{key}: uncertainty unknown, not propagated")
        return float(value)
    return UArray.scalar(float(value), float(sigma), key, "polarisation", systematic=True)


def _mul(a, b):
    """a * b with a UArray on the left when there is one (numpy arrays would otherwise broadcast
    the UArray as an object)."""
    if isinstance(b, UArray) and not isinstance(a, UArray):
        return b * a
    return a * b


def _div(a, b):
    if isinstance(b, UArray) and not isinstance(a, UArray):
        return constant(a) / b
    return a / b


def _sum(it):
    out = None
    for x in it:
        out = x if out is None else (x + out if isinstance(x, UArray) and not isinstance(out, UArray) else out + x)
    return out


def depolarisation_factor(P_f: Number) -> Number:
    """r = (1 - P_f) / (-ln P_f): depth average of the beam polarisation in a sample with
    exponential depolarisation P(x) = P exp(-D x), P_f = exp(-D l). r = 1 for P_f = 1; series
    1 - u/2 - u²/12 for u = 1 - P_f < 1e-4."""
    def r(x):
        x = np.asarray(x, float)
        u = 1.0 - x
        with np.errstate(divide="ignore", invalid="ignore"):
            exact = u / (-np.log(np.where(x > 0, x, np.nan)))
        return np.where(np.abs(u) < 1e-4, 1.0 - u / 2.0 - u * u / 12.0, exact)

    def dr(x):
        x = np.asarray(x, float)
        u = 1.0 - x
        with np.errstate(divide="ignore", invalid="ignore"):
            L = np.log(np.where(x > 0, x, np.nan))
            exact = (L + u / x) / (L * L)
        return np.where(np.abs(u) < 1e-4, 0.5 + u / 6.0, exact)

    if isinstance(P_f, UArray):
        return P_f.apply(r, dr)
    if not 0.0 < float(P_f) <= 1.0 + 1e-12:
        raise ValueError(f"final polarisation P_f must be in (0, 1], got {P_f}")
    return float(r(P_f))


@dataclass
class PolarisationSetup:
    """Efficiencies of a SANSPOL measurement (one setting of wavelength and collimation).

    ``P``: beam polarisation (2p - 1); ``f``: flipped fraction of the flipper (F = 2f - 1);
    ``P_f``: polarisation after the sample (depolarisation; 1 = none); ``sigma_*``: standard
    uncertainties (None: unknown, not propagated); ``form``: "exact" or "bersans";
    ``flipper_off_is_plus``: flipper off gives spin parallel to the field (+); ``source``: where
    every value comes from (shown in notes and outputs)."""
    P: float = 0.98
    f: float = 0.995
    P_f: float = 1.0
    sigma_P: Optional[float] = None
    sigma_f: Optional[float] = None
    sigma_P_f: Optional[float] = 0.0
    form: str = "exact"
    flipper_off_is_plus: bool = True
    source: Dict[str, str] = field(default_factory=dict)

    def __post_init__(self):
        if self.form not in ("exact", "bersans"):
            raise ValueError("form must be 'exact' or 'bersans'")
        if not 0.0 < abs(float(self.P)) <= 1.0:
            raise ValueError(f"beam polarisation P must be in (0, 1], got {self.P}")
        if not 0.5 < float(self.f) <= 1.0:
            raise ValueError(f"flipped fraction f must be in (0.5, 1], got {self.f} (F = 2f - 1)")

    @property
    def F(self) -> float:
        return 2.0 * float(self.f) - 1.0

    @property
    def r(self) -> float:
        return float(depolarisation_factor(float(self.P_f)))

    def parameters(self, notes: Optional[List[str]] = None):
        """(P, f, r) as floats or scalar UArrays (systematic sources ``polarisation:P`` ...)."""
        P = _param(self.P, self.sigma_P, "polarisation:P", notes)
        f = _param(self.f, self.sigma_f, "polarisation:f", notes)
        Pf = _param(self.P_f, self.sigma_P_f, "polarisation:P_f", notes)
        return P, f, depolarisation_factor(Pf)

    def describe(self) -> str:
        src = lambda k: f" ({self.source[k]})" if k in self.source else ""   # noqa: E731
        sig = lambda s: "" if s is None else f" ± {s:.2g}"                    # noqa: E731
        return (f"SANSPOL: P = {float(self.P):.4f}{sig(self.sigma_P)}{src('P')}, f = {float(self.f):.4f}"
                f"{sig(self.sigma_f)} (F = {self.F:.4f}){src('f')}, P_f = {float(self.P_f):.4f}{src('P_f')} "
                f"(r = {self.r:.4f}), {self.form} flipper form, flipper off = "
                f"{'+' if self.flipper_off_is_plus else '-'} (spin relative to the field)")

    def as_dict(self) -> dict:
        return {"P": float(self.P), "f": float(self.f), "P_f": float(self.P_f), "sigma_P": self.sigma_P,
                "sigma_f": self.sigma_f, "sigma_P_f": self.sigma_P_f, "form": self.form,
                "flipper_off_is_plus": self.flipper_off_is_plus, "source": dict(self.source)}


def state_weights(setup: PolarisationSetup, state: str, notes: Optional[List[str]] = None, params=None):
    """(n+, n-): fractions of the incident beam with spin + and - in a measured state
    ("unpolarised", "off", "on"); floats or scalar UArrays."""
    if state not in STATES:
        raise ValueError(f"state must be one of {STATES}, got {state!r}")
    if state == "unpolarised":
        return 0.5, 0.5
    P, f, r = params if params is not None else setup.parameters(notes)
    Pe = P * r
    a_off = (1.0 + Pe) * 0.5           # majority state with flipper off
    b_off = (1.0 - Pe) * 0.5
    if state == "off":
        a, b = a_off, b_off
    else:
        b = f * a_off + (1.0 - f) * b_off if setup.form == "exact" else f * a_off
        a = 1.0 - b
    return (a, b) if setup.flipper_off_is_plus else (b, a)


def sanspol_forward(I_plus, I_minus, setup: PolarisationSetup, state: str):
    """Measured intensity of a state from I+ and I- (the forward model used for fitting)."""
    a, b = state_weights(setup, state, params=setup.parameters(None))
    return _sum([_mul(a, I_plus), _mul(b, I_minus)])


# ------------------------------------------------------------------------- correction
@dataclass
class SanspolResult:
    """I+ and I- (spin parallel / antiparallel to the field, summed over final spins), the
    unpolarised cross section (I+ + I-)/2 and the difference I- - I+ (nuclear-magnetic
    interference), as UArrays when the input was; ``chi2``: per point, only with three states."""
    plus: object
    minus: object
    unpolarised: object
    difference: object
    chi2: Optional[np.ndarray]
    states: List[str]
    setup: PolarisationSetup
    notes: List[str] = field(default_factory=list)

    def items(self):
        return {"plus": self.plus, "minus": self.minus, "unpolarised": self.unpolarised,
                "difference": self.difference}.items()


def _with_unit(x, unit):
    if isinstance(x, UArray):
        x.unit = unit
    return x


def sanspol_correct(measured: Dict[str, object], setup: PolarisationSetup,
                    weights: Optional[Dict[str, np.ndarray]] = None) -> SanspolResult:
    """I+ and I- from two or three measured states.

    ``measured``: {"off": I_off, "on": I_on, "unpolarised": I_u} (at least two; arrays or UArrays of
    the same shape, all on the same scale); ``weights`` (three states only): per point 1/variance of
    each state, default from the statistical variance of the UArrays (equal weights for plain
    arrays)."""
    notes: List[str] = []
    states = [s for s in STATES if measured.get(s) is not None]
    unknown = set(measured) - set(STATES)
    if unknown:
        raise ValueError(f"unknown states {sorted(unknown)}; use {STATES}")
    if len(states) < 2:
        raise ValueError("SANSPOL correction needs two of the states 'off', 'on', 'unpolarised'")
    shapes = {np.shape(_v(measured[s])) for s in states}
    if len(shapes) != 1:
        raise ValueError(f"the states have different shapes {shapes}")
    unit = next((getattr(measured[s], "unit", "") for s in states if getattr(measured[s], "unit", "")), "")
    params = setup.parameters(notes)
    A = {s: state_weights(setup, s, params=params) for s in states}
    y = {s: measured[s] for s in states}
    chi2 = None
    if len(states) == 2:
        s1, s2 = states
        (a1, b1), (a2, b2) = A[s1], A[s2]
        det = a1 * b2 - a2 * b1
        if abs(float(_v(det))) < 1e-6:
            raise ValueError(f"states {s1} and {s2} do not separate I+ and I- (determinant {float(_v(det)):.2g})")
        plus = _div(_mul(b2, y[s1]) - _mul(b1, y[s2]), det)
        minus = _div(_mul(a1, y[s2]) - _mul(a2, y[s1]), det)
    else:
        if weights is None:
            weights = {}
            for s in states:
                if isinstance(y[s], UArray):
                    var = np.asarray(y[s].variance(systematic=False), float)
                    pos = var[np.isfinite(var) & (var > 0)]
                    floor = pos.min() if pos.size else 1.0
                    weights[s] = 1.0 / np.where(np.isfinite(var) & (var > 0), var, floor)
                else:
                    weights[s] = np.ones(np.shape(y[s]))
            if not all(isinstance(y[s], UArray) for s in states):
                notes.append("three states without uncertainties: equal weights")
        w = {s: np.asarray(weights[s], float) for s in states}
        M11 = _sum(_mul(_mul(A[s][0], A[s][0]), w[s]) for s in states)
        M12 = _sum(_mul(_mul(A[s][0], A[s][1]), w[s]) for s in states)
        M22 = _sum(_mul(_mul(A[s][1], A[s][1]), w[s]) for s in states)
        r1 = _sum(_mul(_mul(A[s][0], w[s]), y[s]) for s in states)
        r2 = _sum(_mul(_mul(A[s][1], w[s]), y[s]) for s in states)
        det = _mul(M11, M22) - _mul(M12, M12)
        plus = _div(_mul(M22, r1) - _mul(M12, r2), det)
        minus = _div(_mul(M11, r2) - _mul(M12, r1), det)
        res = {s: _v(y[s]) - (float(_v(A[s][0])) * _v(plus) + float(_v(A[s][1])) * _v(minus)) for s in states}
        chi2 = sum(w[s] * res[s] ** 2 for s in states)
        ok = np.isfinite(chi2)
        if ok.any():
            notes.append(f"three states: weighted least squares per point, median chi² = "
                         f"{float(np.median(chi2[ok])):.3g} (1 degree of freedom; >> 1: P, f or the scale of "
                         f"the polariser-out data inconsistent)")
    plus, minus = _with_unit(plus, unit), _with_unit(minus, unit)
    unpol = _with_unit(_mul(_sum([plus, minus]), 0.5), unit)
    diff = _with_unit(_sum([minus, _mul(plus, -1.0)]), unit)
    notes.insert(0, setup.describe() + f"; states used: {', '.join(states)}")
    if setup.form == "bersans":
        notes.append("flipper form 'bersans' (n-_on = p f, Keiderling et al. 2008 eq. 3b): the (1-p)(1-f) term "
                     "is neglected")
    return SanspolResult(plus, minus, unpol, diff, chi2, states, setup, notes)


def flipped_fraction_from_data(I_off, I_on, I_unpolarised, mask=None) -> float:
    """Flipped fraction f from the three states (exact form, independent of P and P_f):
    f = Σ (I_off - I_on) / (2 Σ (I_off - I_u)), summed over the valid points (``mask``: True =
    use). Needs a clear magnetic contrast (I_off ≠ I_on) and the polariser-out data on the same
    absolute scale as the polarised data."""
    a, b, u = (np.asarray(_v(x), float).ravel() for x in (I_off, I_on, I_unpolarised))
    m = np.isfinite(a) & np.isfinite(b) & np.isfinite(u)
    if mask is not None:
        m &= np.asarray(mask, bool).ravel()
    den = 2.0 * np.sum(a[m] - u[m])
    if den == 0:
        raise ValueError("no magnetic contrast: I_off = I_unpolarised")
    return float(np.sum(a[m] - b[m]) / den)


# ------------------------------------------------------------------------- instrument defaults
def beam_polarisation_table(wavelength_nm: float, collimation_m: float, inst: Optional[dict] = None):
    """Default beam polarisation of the instrument (SANS-I: Aswal et al. 2008, Fig. 4) for a
    wavelength and collimation length: linear in λ and in log(collimation). Returns
    (P, sigma, text); outside the measured range the nearest value is used and the text says
    "extrapolated". Raises KeyError when the instrument has no polarisation option."""
    if inst is None:
        from .reduce import load_instrument
        inst = load_instrument("sans_i")
    pol = (inst or {}).get("polarisation")
    if not pol:
        raise KeyError(f"{inst.get('instrument', 'this instrument')} has no polarisation option")
    tab = pol["beam_polarisation"]
    lam = np.asarray(tab["wavelength_nm"], float)
    col = np.log(np.asarray(tab["collimation_m"], float))
    Ptab = np.array([[np.nan if v is None else float(v) for v in row] for row in tab["P"]])
    notes = []
    cmax = pol.get("max_collimation_m")
    if cmax is not None and collimation_m > float(cmax) + 1e-6:
        notes.append(f"collimation {collimation_m:g} m > {cmax} m: the polariser cannot be in (metadata inconsistent)")
    lc = math.log(max(float(collimation_m), 1e-3))
    rows, outside = [], []
    for row in Ptab:
        ok = np.isfinite(row)
        outside.append(lc < col[ok][0] - 1e-9 or lc > col[ok][-1] + 1e-9)
        rows.append(np.interp(lc, col[ok], row[ok]))
    rows = np.asarray(rows)
    # rows that enter the interpolation in λ
    j = int(np.clip(np.searchsorted(lam, wavelength_nm), 1, lam.size - 1))
    t = (wavelength_nm - lam[j - 1]) / (lam[j] - lam[j - 1])
    used = [i for i, wgt in ((j - 1, 1 - t), (j, t)) if wgt > 1e-9] if 0 <= t <= 1 else \
        [0 if wavelength_nm < lam[0] else lam.size - 1]
    if any(outside[i] for i in used):
        notes.append("extrapolated in collimation")
    if wavelength_nm < lam[0] or wavelength_nm > lam[-1]:
        notes.append("extrapolated in wavelength")
    P = float(np.interp(wavelength_nm, lam, rows))
    text = f"{tab.get('source', 'instrument table')}: λ {wavelength_nm:.3g} nm, collimation {collimation_m:g} m"
    if notes:
        text += " (" + "; ".join(dict.fromkeys(notes)) + ")"
    return P, tab.get("uncertainty"), text


def _lookup(meta: dict, keys: Sequence[str]):
    for k in keys or ():
        for kk in (k, f"hdf:{k}"):
            if kk in meta and meta[kk] is not None:
                return meta[kk], kk
    return None, None


def _truthy(v) -> Optional[bool]:
    if isinstance(v, str):
        s = v.strip().lower()
        if s in ("in", "on", "1", "true", "yes", "up"):
            return True
        if s in ("out", "off", "0", "false", "no", "down"):
            return False
        return None
    try:
        return bool(float(v))
    except (TypeError, ValueError):
        return None


def setup_from_metadata(meta: dict, inst: Optional[dict] = None, **override):
    """State and efficiencies of a reduced data set from its metadata (SASprocess ``metadata``:
    the header items of the raw run), else the instrument defaults (SANS-I: P from
    :func:`beam_polarisation_table`, f from the instrument file). The metadata keys are listed in the
    instrument file under ``polarisation: metadata_keys``. Returns (state or None, PolarisationSetup);
    ``override`` sets fields of the setup (values given by the user win)."""
    if inst is None:
        from .reduce import load_instrument
        inst = load_instrument("sans_i")
    pol = (inst or {}).get("polarisation") or {}
    keys = pol.get("metadata_keys") or {}
    source: Dict[str, str] = {}
    vals: Dict[str, object] = {}
    polariser, _ = _lookup(meta, keys.get("polariser", []))
    flipper, _ = _lookup(meta, keys.get("flipper", []))
    state = None
    pin, fon = _truthy(polariser), _truthy(flipper)
    if pin is False:
        state = "unpolarised"
    elif pin is True and fon is not None:
        state = "on" if fon else "off"
        if meta.get("flipper_source"):
            source["state"] = f"flipper {flipper}: {meta['flipper_source']}"
    elif pin is True:
        source["state"] = "polariser in, flipper state unknown (not stored in the raw file)"
    for name, sig in (("P", "sigma_P"), ("f", "sigma_f"), ("P_f", "sigma_P_f")):
        v, k = _lookup(meta, keys.get(name, []))
        if v is not None:
            vals[name] = float(v)
            source[name] = f"from file ({k})"
    if "P" not in vals and pol.get("beam_polarisation") and meta.get("wavelength_A") and meta.get("coll_m"):
        P, s, text = beam_polarisation_table(float(meta["wavelength_A"]) / 10.0, float(meta["coll_m"]), inst)
        vals["P"], vals["sigma_P"], source["P"] = P, s, text
    if "f" not in vals and pol.get("flipped_fraction"):
        ff = pol["flipped_fraction"]
        vals["f"], vals["sigma_f"] = float(ff["f"]), ff.get("uncertainty")
        source["f"] = ff.get("source", "instrument file")
    if "P_f" not in vals:
        source["P_f"] = "default: no depolarisation"
    if "flipper_off_is_plus" in pol:
        vals["flipper_off_is_plus"] = bool(pol["flipper_off_is_plus"])
    for k, v in override.items():
        vals[k] = v
        if k in ("P", "f", "P_f"):
            source[k] = "entered"
    return state, PolarisationSetup(source=source, **vals)


# ------------------------------------------------------------------------- reduced data
def _union_mask(objs):
    m = None
    for o in objs:
        mk = getattr(o, "mask", None)
        if mk is None:
            mk = getattr(o, "flags", None)
        if mk is None:
            continue
        mk = np.asarray(mk, np.uint32)
        m = mk.copy() if m is None else (m | mk)
    return m


def correct_reduced(setup: PolarisationSetup, off=None, on=None, unpolarised=None) -> Dict[str, object]:
    """SANSPOL correction of reduced results (:class:`~pysasfit.sasred.absolute.Reduced2D` or
    :class:`~pysasfit.sasred.reduce.ReducedRun`) of the same geometry and binning. Returns
    {"plus", "minus", "unpolarised", "difference"}: copies of the first given result with the
    corrected image (and intensity for 1D), the union of the masks, the setup in ``parameters
    ['polarisation']`` and the notes; plus "result": the :class:`SanspolResult` of the main array."""
    given = {s: r for s, r in (("off", off), ("on", on), ("unpolarised", unpolarised)) if r is not None}
    if len(given) < 2:
        raise ValueError("give two or three of off, on, unpolarised")
    first = next(iter(given.values()))
    for s, r in given.items():
        if hasattr(first, "q") and hasattr(r, "q") and not np.allclose(first.q, r.q, rtol=1e-6, atol=0):
            raise ValueError(f"state {s}: q bins differ from the first state (reduce with the same binning)")
        if np.shape(_v(r.image)) != np.shape(_v(first.image)):
            raise ValueError(f"state {s}: image shape differs")
    out: Dict[str, object] = {}
    results = {}
    for attr in ("image", "intensity"):
        if all(isinstance(getattr(r, attr, None), (UArray, np.ndarray)) for r in given.values()):
            results[attr] = sanspol_correct({s: getattr(r, attr) for s, r in given.items()}, setup)
    main = results.get("intensity") or results["image"]
    mask = _union_mask(given.values())
    for name in ("plus", "minus", "unpolarised", "difference"):
        kw = {attr: getattr(res, name) for attr, res in results.items()}
        if mask is not None:
            kw["mask" if hasattr(first, "mask") else "flags"] = mask
        par = dict(getattr(first, "parameters", {}) or {})
        par["polarisation"] = {**setup.as_dict(), "output": name, "states": main.states,
                               "runs": {s: getattr(r, "runs", None) or getattr(r, "run", None)
                                        for s, r in given.items()}}
        kw["parameters"] = par
        kw["notes"] = list(getattr(first, "notes", []) or []) + main.notes + [f"SANSPOL output: {name}"]
        if hasattr(first, "label"):
            kw["label"] = f"{first.label} [{name}]"
        out[name] = replace(first, **kw)
    out["result"] = main
    return out


# ------------------------------------------------------------------------- files
_NAMES = {"plus": "I+ (spin parallel to the field)", "minus": "I- (spin antiparallel to the field)",
          "unpolarised": "(I+ + I-)/2", "difference": "I- - I+ (nuclear-magnetic interference)"}


def _read(path):
    import h5py
    with h5py.File(path, "r") as f:
        e = f[f.attrs.get("default", "sasentry01") if not isinstance(f.attrs.get("default"), bytes)
              else f.attrs["default"].decode()]
        d = e[e.attrs.get("default", "sasdata01") if not isinstance(e.attrs.get("default"), bytes)
              else e.attrs["default"].decode()]
        I = np.asarray(d["I"][()], float)
        dI = np.asarray(d["Idev"][()], float) if "Idev" in d else None
        mask = np.asarray(d["mask_bits"][()], np.uint32) if "mask_bits" in d else (
            np.asarray(d["Mask"][()], np.uint32) if "Mask" in d else None)
        q = np.asarray(d["Q"][()], float) if "Q" in d else None
        meta = {}
        for p in e.values():
            if hasattr(p, "keys") and "metadata" in p:
                try:
                    meta = json.loads(p["metadata"][()].decode() if isinstance(p["metadata"][()], bytes)
                                      else str(p["metadata"][()]))
                except Exception:  # noqa: BLE001
                    meta = {}
        unit = d["I"].attrs.get("sasred_unit", b"")
        unit = unit.decode() if isinstance(unit, bytes) else str(unit)
    return I, dI, mask, q, meta, unit


def correct_files(paths: Dict[str, str], out_prefix: str, setup: Optional[PolarisationSetup] = None,
                  inst: Optional[dict] = None, **override) -> Dict[str, str]:
    """SANSPOL correction of reduced NXcanSAS files (1D or 2D, as written by pySASred).

    ``paths``: {"off": file, "on": file, "unpolarised": file} (two or three); the state of each file
    is taken as given. ``setup``: None = from the metadata of the first polarised file and the
    instrument defaults (:func:`setup_from_metadata`, ``override`` wins). Writes
    ``<out_prefix>_plus.h5``, ``_minus.h5``, ``_unpolarised.h5``, ``_difference.h5``: copies of the
    first polarised file with I, Idev (statistical, from the Idev of the inputs) and the union of the
    masks; the covariance, systematic sensitivities and budget of the input are removed (they belong
    to the uncorrected intensity), the setup and notes are added to SASprocess. Returns the paths."""
    import shutil
    import h5py
    from .data.uarray import Source
    paths = {s: p for s, p in paths.items() if p}
    if len(paths) < 2:
        raise ValueError("give two or three files")
    data = {s: _read(p) for s, p in paths.items()}
    ref = next((s for s in ("off", "on", "unpolarised") if s in data))
    if setup is None:
        _, setup = setup_from_metadata(data[ref][4], inst, **override)
    q0 = data[ref][3]
    meas = {}
    for s, (I, dI, mask, q, meta, unit) in data.items():
        if I.shape != data[ref][0].shape:
            raise ValueError(f"{paths[s]}: shape {I.shape} differs from {data[ref][0].shape}")
        if q is not None and q0 is not None and not np.allclose(q, q0, rtol=1e-6):
            raise ValueError(f"{paths[s]}: q bins differ")
        if dI is None:
            meas[s] = I
        else:
            meas[s] = UArray.from_source(I, Source(f"counting:{s}", np.nan_to_num(dI.ravel() ** 2), "counting",
                                                   False), unit)
    res = sanspol_correct(meas, setup)
    mask = None
    for s in data:
        mk = data[s][2]
        if mk is not None:
            mask = mk.copy() if mask is None else (mask | mk)
    written = {}
    for name, arr in res.items():
        out = f"{out_prefix}_{name}.h5"
        shutil.copyfile(paths[ref], out)
        val = np.asarray(_v(arr), float).reshape(data[ref][0].shape)
        dev = (np.asarray(arr.std(systematic=False), float).reshape(val.shape) if isinstance(arr, UArray)
               else None)
        with h5py.File(out, "r+") as f:
            e = f["sasentry01"]
            d = e["sasdata01"]
            attrs = dict(d["I"].attrs)
            del d["I"]
            i = d.create_dataset("I", data=val)
            i.attrs.update(attrs)
            if "Idev" in d:
                da = dict(d["Idev"].attrs)
                del d["Idev"]
                if dev is not None:
                    d.create_dataset("Idev", data=dev).attrs.update(da)
            for g in ("I_covariance_statistical", "I_systematic", "I_budget"):
                if g in d:
                    del d[g]
            if mask is not None:
                for k, v in (("mask_bits", mask), ("Mask", (mask != 0).astype(np.int8))):
                    if k in d:
                        del d[k]
                        d.create_dataset(k, data=v)
            t = e["title"][()]
            t = t.decode() if isinstance(t, bytes) else str(t)
            del e["title"]
            e["title"] = np.bytes_(f"{t} [{_NAMES[name]}]".encode())
            p = e["sasprocess01"] if "sasprocess01" in e else e.create_group("sasprocess01")
            notes = p["notes"][()].decode() if "notes" in p and isinstance(p["notes"][()], bytes) else ""
            if "notes" in p:
                del p["notes"]
            p["notes"] = np.bytes_(("\n".join([notes] + res.notes + [f"SANSPOL output: {_NAMES[name]}"]
                                              + [f"input {s}: {pth}" for s, pth in paths.items()])).encode())
            if "polarisation" in p:
                del p["polarisation"]
            p["polarisation"] = np.bytes_(json.dumps({**setup.as_dict(), "output": name,
                                                      "inputs": paths}).encode())
        written[name] = out
    return written
