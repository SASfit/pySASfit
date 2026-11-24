"""Polarised neutrons: what a data set measures, as a setting of the data set instead of separate functions.

SASfit has one function per polarisation channel (``_pp``, ``_mm``, ``_pm``, ``_POLARIS``, ``_cross``, ...) and the
efficiencies as parameters of every model. Here a magnetic model returns its **basis terms** once,

    NN  = <|N|²>                      nuclear
    NM  = <Re N M⊥h*>                 nuclear-magnetic interference (h: field axis)
    MH  = <|M⊥h|²>                    magnetic, non-spin-flip
    MSF = <|M⊥|² - |M⊥h|²>            magnetic, spin-flip
    CH  = <Im(M⊥a M⊥b*)>              chiral (0 unless the model is chiral)

(per (q, ψ), ψ = angle between q and the field on the detector; M = magnetisation, i.e. moment density), and the
data set says what was measured (:class:`Polarisation`).

**Convention (Moon, Riste & Koehler, Phys. Rev. 181, 920 (1969)):** + and - are the neutron **spin** parallel /
antiparallel to the field (σ_z|±> = ±|±>, z along the field). The neutron moment is opposite to its spin, so the
amplitudes are U±± = b ± p M⊥z: spin parallel to the magnetisation adds the magnetic amplitude. The ideal cross
sections are

    I++ = NN + MH + 2NM,  I-- = NN + MH - 2NM,  I+- = MSF - 2CH,  I-+ = MSF + 2CH,
    SANSPOL  I± = I±± + I±∓ = NN + MH + MSF ± 2NM ∓ 2CH,    I- - I+ = -4NM + 4CH.

Michels and the SASfit C functions (``_pp``, ``pol`` = +1) label the states by the neutron **moment**: their "++"
is "--" here, their ``pol`` is -(n+ - n-) here.

and, for data that are not corrected for the instrument, the mixing of I+ and I- with the beam polarisation P,
the flipped fraction f of the flipper and the depolarisation of the sample (P_f), exactly as in
:mod:`pysasfit.sasred.polarisation`:

=================  ==========================================  ==================
quantity           data set                                    needs
=================  ==========================================  ==================
``unpolarised``    polariser out, or (I+ + I-)/2               --
``off``            polariser in, flipper off (not corrected)   P (, P_f)
``on``             polariser in, flipper on (not corrected)    P, f (, P_f)
``plus``/``minus`` I+, I- corrected (pySASred SANSPOL output)   --
``difference``     I- - I+ (corrected)                         --
``pp`` ... ``mp``  four channels with analyser, corrected      --
=================  ==========================================  ==================

``flipper_off_is_plus`` says which flipper state is + (spin parallel to the field). Non-magnetic contributions enter with the NN weight (1, or 0 in the spin-flip channels and the
difference). A model gets the setting as ``ModelFunction.polarisation``; while it is evaluated the setting is
active (:func:`active`) and magnetic functions read it (:func:`current`). Without a setting a magnetic port behaves
like its SASfit original (its ``pol`` parameter in SASfit's moment labels: I = (1+pol)/2 I- + (1-pol)/2 I+ here);
C functions with a ``pol`` parameter get pol = n- - n+ of the setting (SANSPOL quantities only).

Parameter addresses for fits: ``pol.P``, ``pol.f``, ``pol.P_f``, ``pol.psi_H``.
"""

from __future__ import annotations

import contextvars
import json
from contextlib import contextmanager, nullcontext
from dataclasses import asdict, dataclass, field

import numpy as np

__all__ = ["TERMS", "QUANTITIES", "Polarisation", "basis_terms", "combine", "active", "current", "cache_key",
           "is_magnetic", "pol_parameter_index", "from_metadata", "from_file", "angular_moments"]  # fmt: skip

TERMS = ("NN", "NM", "MH", "MSF", "CH")

# spin convention (Moon, Riste & Koehler 1969): + = neutron spin parallel to the field
_W_PLUS = np.array([1.0, 2.0, 1.0, 1.0, -2.0])
_W_MINUS = np.array([1.0, -2.0, 1.0, 1.0, 2.0])
_FIXED = {
    "unpolarised": np.array([1.0, 0.0, 1.0, 1.0, 0.0]),
    "plus": _W_PLUS,
    "minus": _W_MINUS,
    "difference": _W_MINUS - _W_PLUS,
    "pp": np.array([1.0, 2.0, 1.0, 0.0, 0.0]),
    "mm": np.array([1.0, -2.0, 1.0, 0.0, 0.0]),
    "pm": np.array([0.0, 0.0, 0.0, 1.0, -2.0]),
    "mp": np.array([0.0, 0.0, 0.0, 1.0, 2.0]),
}

#: quantity -> (label, instrument parameters it needs)
QUANTITIES = {
    "unpolarised": ("unpolarised (polariser out, or (I+ + I−)/2)", ()),
    "off": ("polariser in, flipper off – not corrected", ("P", "P_f")),
    "on": ("polariser in, flipper on – not corrected", ("P", "f", "P_f")),
    "plus": ("I+ (SANSPOL, corrected)", ()),
    "minus": ("I− (SANSPOL, corrected)", ()),
    "difference": ("I− − I+ (corrected)", ()),
    "pp": ("I++ (analyser, corrected)", ()),
    "mm": ("I−− (analyser, corrected)", ()),
    "pm": ("I+− spin flip (analyser, corrected)", ()),
    "mp": ("I−+ spin flip (analyser, corrected)", ()),
}

_SANSPOL = ("unpolarised", "off", "on", "plus", "minus")


@dataclass
class Polarisation:
    """What a data set measures (``quantity``, see the module docstring) and the instrument values it needs.

    ``P``: beam polarisation, ``f``: flipped fraction of the flipper (F = 2f - 1), ``P_f``: polarisation after
    the sample (depolarisation, 1 = none), ``form``: "exact" or "bersans" flipper mixing, ``flipper_off_is_plus``:
    flipper off = spin parallel to the field, ``psi_H``: field direction on the detector (deg from +q_x,
    counter-clockwise), ``average``: azimuthal average over ψ (sin²ψ -> 1/2, sin⁴ψ -> 3/8; 1D radial averages)
    instead of the ψ parameter of the functions, ``source``: where each value comes from (file, instrument
    table, entered)."""

    quantity: str = "unpolarised"
    P: float = 0.98
    f: float = 0.995
    P_f: float = 1.0
    form: str = "exact"
    flipper_off_is_plus: bool = True
    psi_H: float = 0.0
    average: bool = False
    source: dict = field(default_factory=dict)

    def __post_init__(self):
        if self.quantity not in QUANTITIES:
            raise ValueError(f"quantity must be one of {tuple(QUANTITIES)}, got {self.quantity!r}")
        if self.form not in ("exact", "bersans"):
            raise ValueError("form must be 'exact' or 'bersans'")

    # ------------------------------------------------------------------ weights
    def needs(self):
        """Instrument parameters the quantity depends on (P, f, P_f)."""
        return QUANTITIES[self.quantity][1]

    @property
    def sanspol(self):
        """True if the quantity is a mixture of I+ and I- only (no analyser channel, no difference)."""
        return self.quantity in _SANSPOL

    def mixing(self):
        """(n+, n-): fractions of I+ and I- in a SANSPOL quantity."""
        q = self.quantity
        if q == "plus":
            return 1.0, 0.0
        if q == "minus":
            return 0.0, 1.0
        if q == "unpolarised":
            return 0.5, 0.5
        if q not in ("off", "on"):
            raise ValueError(f"{q}: not a mixture of I+ and I-")
        from ..sasred.polarisation import PolarisationSetup, state_weights

        s = PolarisationSetup(P=float(self.P), f=float(self.f), P_f=float(self.P_f), form=self.form,
                              flipper_off_is_plus=self.flipper_off_is_plus, sigma_P_f=0.0)  # fmt: skip
        a, b = state_weights(s, q, params=(float(self.P), float(self.f), s.r))
        return float(a), float(b)

    def effective_pol(self):
        """SASfit's ``pol`` parameter for this quantity (SANSPOL quantities). SASfit labels by the neutron moment, so
        its pol = +1 is spin antiparallel to the field: pol = n- - n+."""
        a, b = self.mixing()
        return b - a

    @staticmethod
    def sasfit_pol_weights(pol_sasfit):
        """Weights of SASfit's ``pol`` parameter: (1+pol)/2 I(moment +) + (1-pol)/2 I(moment -) =
        (1+pol)/2 I- + (1-pol)/2 I+ in the spin labels used here."""
        P = float(pol_sasfit)
        return 0.5 * (1.0 + P) * _W_MINUS + 0.5 * (1.0 - P) * _W_PLUS

    def weights(self):
        """Weights of (NN, NM, MH, MSF, CH)."""
        if self.quantity in _FIXED:
            return _FIXED[self.quantity].copy()
        a, b = self.mixing()
        return a * _W_PLUS + b * _W_MINUS

    def key(self):
        """Hashable description for caches."""
        return (tuple(np.round(self.weights(), 15)), float(self.psi_H), bool(self.average))

    # ------------------------------------------------------------------ text, files
    def describe(self):
        lab = QUANTITIES[self.quantity][0]
        parts = []
        for k in self.needs():
            v = getattr(self, k)
            src = self.source.get(k)
            parts.append(f"{k} = {v:.4g}" + (f" ({src})" if src else ""))
        if "f" in self.needs():
            parts.append(f"F = 2f − 1 = {2 * self.f - 1:.4g}")
        geo = "azimuthal average" if self.average else f"field at ψ_H = {self.psi_H:g}°"
        return f"{lab}" + (": " + ", ".join(parts) if parts else "") + f"; {geo}"

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, d):
        d = dict(d or {})
        known = {k: d[k] for k in cls.__dataclass_fields__ if k in d}
        return cls(**known)


# --------------------------------------------------------------------------- context
_CURRENT = contextvars.ContextVar("pysasfit_polarisation", default=None)


@contextmanager
def active(pol):
    """Make ``pol`` (a :class:`Polarisation` or None) the setting seen by magnetic functions."""
    if pol is None:
        with nullcontext():
            yield
        return
    token = _CURRENT.set(pol)
    try:
        yield
    finally:
        _CURRENT.reset(token)


def current():
    """The active :class:`Polarisation` or None."""
    return _CURRENT.get()


def cache_key():
    p = _CURRENT.get()
    return None if p is None else p.key()


# --------------------------------------------------------------------------- terms
def angular_moments(psi_deg, average=False):
    """(sin²ψ, sin⁴ψ) of the angle between q and the field (or the azimuthal averages 1/2, 3/8)."""
    if average:
        return 0.5, 0.375
    s2 = np.sin(np.radians(np.asarray(psi_deg, float))) ** 2
    return s2, s2 * s2


def basis_terms(kn, km, S1, S2, s2, s4):
    """Basis terms of particles with nuclear amplitude ``kn`` and magnetic amplitude ``km`` (same units, SASfit
    I(0) = (Δη V)² normalisation) whose moments have the orientation moments S1 = <cos θ> and S2 = <P2(cos θ)>
    with respect to the field (saturated: 1, 1; random: 0, 0; Langevin: L(ξ), 1 - 3L(ξ)/ξ), for the
    perpendicular scattering geometry (field ⟂ beam). ``s2``, ``s4``: sin²ψ, sin⁴ψ. Returns an array (5, ...)."""
    kn, km = np.asarray(kn, float), np.asarray(km, float)
    m2 = km * km
    NN = kn * kn
    NM = kn * km * S1 * s2
    MH = m2 * (S2 * s4 + (1.0 - S2) / 3.0 * s2)
    MSF = m2 * (2.0 * (1.0 - S2) / 3.0 - S2 * s4 + (4.0 * S2 - 1.0) / 3.0 * s2)
    NN, NM, MH, MSF = np.broadcast_arrays(NN, NM, MH, MSF)
    return np.stack([NN, NM, MH, MSF, np.zeros_like(NN)])


def combine(terms, weights):
    """Σ w_k T_k."""
    w = np.asarray(weights, float).reshape((5,) + (1,) * (np.ndim(terms) - 1))
    return np.sum(w * np.asarray(terms, float), axis=0)


# --------------------------------------------------------------------------- functions
def is_magnetic(f):
    """True if the function ``f`` (SASfitFunction) returns basis terms (a magnetic Python port)."""
    if f is None:
        return False
    from . import ports

    try:
        return ports.get(f.name).terms is not None
    except (KeyError, AttributeError):
        return False


def pol_parameter_index(f):
    """Index of SASfit's ``pol`` parameter of a (C) function, or None."""
    if f is None:
        return None
    names = [str(n).lower() for n in getattr(f, "parameter_names", ())]
    for cand in ("pol", "p_pol"):
        if cand in names:
            return names.index(cand)
    return None


# --------------------------------------------------------------------------- data files
_OUTPUT_TO_QUANTITY = {"plus": "plus", "minus": "minus", "unpolarised": "unpolarised", "difference": "difference"}


def from_metadata(polarisation_json=None, metadata=None, instrument=None):
    """The :class:`Polarisation` of a reduced pySASred file, or None if it says nothing about polarisation.

    ``polarisation_json``: SASprocess ``polarisation`` of a SANSPOL output (:func:`pysasfit.sasred.polarisation.
    correct_files`): the quantity is that output, P and f are kept for the record. ``metadata``: SASprocess
    ``metadata`` (header items of the raw run): the state (polariser out / flipper off / on) from the keys of the
    instrument file, P and f from the file or the instrument defaults (:func:`pysasfit.sasred.polarisation.
    setup_from_metadata`)."""
    if polarisation_json:
        d = json.loads(polarisation_json) if isinstance(polarisation_json, (str, bytes)) else dict(polarisation_json)
        q = _OUTPUT_TO_QUANTITY.get(d.get("output"))
        if q is not None:
            src = {k: "SANSPOL correction in pySASred" for k in ("P", "f", "P_f")}
            return Polarisation(quantity=q, P=float(d.get("P", 0.98)), f=float(d.get("f", 0.995)),
                                P_f=float(d.get("P_f", 1.0)), form=d.get("form", "exact"),
                                flipper_off_is_plus=bool(d.get("flipper_off_is_plus", True)), source=src)  # fmt: skip
    if metadata:
        from ..sasred.polarisation import setup_from_metadata

        meta = json.loads(metadata) if isinstance(metadata, (str, bytes)) else dict(metadata)
        try:
            state, s = setup_from_metadata(meta, instrument)
        except Exception:  # noqa: BLE001 - other instruments
            return None
        if state is None:
            return None
        return Polarisation(quantity=state, P=float(s.P), f=float(s.f), P_f=float(s.P_f), form=s.form,
                            flipper_off_is_plus=s.flipper_off_is_plus, source=dict(s.source))  # fmt: skip
    return None


def from_file(path):
    """:func:`from_metadata` of an NXcanSAS file (None for other files or without polarisation information)."""
    try:
        import h5py
    except ImportError:
        return None
    try:
        with h5py.File(path, "r") as fh:
            pj = meta = None
            for e in fh.values():
                if not hasattr(e, "values"):
                    continue
                for g in e.values():
                    if hasattr(g, "keys"):
                        if "polarisation" in g and pj is None:
                            pj = g["polarisation"][()]
                        if "metadata" in g and meta is None:
                            meta = g["metadata"][()]
            dec = lambda v: v.decode() if isinstance(v, bytes) else (None if v is None else str(v))  # noqa: E731
            return from_metadata(dec(pj), dec(meta))
    except (OSError, ValueError, KeyError):
        return None
