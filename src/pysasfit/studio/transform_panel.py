"""The "Transform" box of the model explorer: SASfit's transform menu (I(Q), SESANS, MSAS) and its settings.

The settings are a plain dictionary (stored per data set and in project files)::

    {"kind": "none" | "sesans" | "sesans_p" | "msas", "thickness": 0.1 (cm), "wavelength": 0.6 (unit of 1/Q),
     "lambda_from_file": True, "norm": "incident" | "transmission", "per_decade": 0 (0 = default),
     "accept": "none" | "q" | "theta", "q_accept": None, "theta_max": None,
     "method": "" (Settings → Integration…) | "filon" | "fftlog" | "qdht", "n_points": 0 (0 = from the settings)}

SESANS resolution: the Q integrals end at the acceptance Q_max of the instrument -- a fixed ``q_accept`` or
Q_max = 2π sin(θ_max)/λ per spin-echo length from the acceptance angle ``theta_max`` (rad).

:func:`build_transform` turns them into a :mod:`pysasfit.studio.hankel` transform.
"""

from __future__ import annotations

import math

import numpy as np
from PySide6 import QtCore, QtWidgets

from .hankel import METHOD_LABELS, METHODS, TRANSFORM_KINDS, make_transform, transform_label

__all__ = ["TransformBox", "default_settings", "build_transform", "domain_of", "LENGTH_OF_Q_UNIT"]

LENGTH_OF_Q_UNIT = {"1/nm": "nm", "1/Å": "Å", "1/m": "m"}
_FILE_UNIT = {"nm": "nm", "Å": "A", "m": "m"}  # explorer unit -> pysasfit.core.sesans_io unit


def default_settings(kind="none"):
    return {"kind": kind, "thickness": 0.1, "wavelength": 0.6, "lambda_from_file": True, "norm": "incident",
            "per_decade": 0, "accept": "none", "q_accept": None, "theta_max": None, "method": "",
            "n_points": 0}  # fmt: skip


def domain_of(settings):
    """``"delta"`` for SESANS (spin-echo lengths), else ``"q"``."""
    return "delta" if (settings or {}).get("kind") in ("sesans", "sesans_p") else "q"


def build_transform(settings, wavelengths=None, deltas=None):
    """The transform of ``settings`` (None for I(Q)); ``wavelengths``/``deltas``: λ per spin-echo length of a file."""
    s = dict(settings or {})
    kind = s.get("kind", "none")
    if kind == "none":
        return None
    kw = {"thickness": float(s.get("thickness", 0.1))}
    if kind in ("sesans", "sesans_p"):
        if s.get("lambda_from_file", True) and wavelengths is not None and deltas is not None and len(wavelengths):
            lam = np.asarray(wavelengths, float)
            kw["wavelength"] = float(lam[0]) if np.allclose(lam, lam[0]) else lam
            if np.ndim(kw["wavelength"]):
                kw["wavelength_delta"] = np.asarray(deltas, float)
        else:
            kw["wavelength"] = float(s.get("wavelength", 1.0))
        mode = _accept_mode(s)
        if mode == "q" and s.get("q_accept"):
            kw["q_accept"] = float(s["q_accept"])
        elif mode == "theta" and s.get("theta_max"):
            kw["theta_max"] = float(s["theta_max"])  # Q_max = 2π sin θ_max / λ (per point with λ from the file)
    else:
        kw["wavelength"] = float(s.get("wavelength", 0.6))
        kw["norm"] = s.get("norm", "incident")
    if s.get("method"):  # "": the default of Settings → Integration… → Hankel transform
        kw["method"] = s["method"]
    if s.get("n_points"):
        kw["n_points"] = int(s["n_points"])
    if s.get("per_decade"):
        kw["per_decade"] = int(s["per_decade"])
    return make_transform(kind, **kw)


def _accept_mode(s):
    """``"none"``, ``"q"`` or ``"theta"`` (settings without "accept": a q_accept means "q")."""
    mode = s.get("accept")
    if mode in ("none", "q", "theta"):
        return mode
    return "q" if s.get("q_accept") else "none"


def _float(text, name, positive=True, allow_empty=False):
    text = text.strip().replace(",", ".")
    if not text:
        if allow_empty:
            return None
        raise ValueError(f"{name}: a number is needed")
    try:
        v = float(text)
    except ValueError:
        raise ValueError(f"{name}: '{text}' is not a number") from None
    if positive and not (v > 0 and math.isfinite(v)):
        raise ValueError(f"{name} must be > 0")
    return v


class TransformBox(QtWidgets.QGroupBox):
    """Kind (SASfit's ``transform`` menu), sample thickness, wavelength and numerical settings."""

    changed = QtCore.Signal()

    def __init__(self, parent=None):
        super().__init__("Transform  (as SASfit's 'transform': I(Q), SESANS, multiple scattering)", parent)
        self._block = False
        self._file_lambda = None  # (min, max) of the wavelengths of a SESANS file
        g = QtWidgets.QGridLayout(self)
        g.setContentsMargins(6, 4, 6, 4)
        g.setVerticalSpacing(2)
        self.kind = QtWidgets.QComboBox()
        for k in TRANSFORM_KINDS:
            self.kind.addItem(transform_label(k), k)
        self.kind.setToolTip(
            "1: the intensity I(Q).\n"
            "H{…}/2π: SESANS, G(δ)−G(0) = 1/(2π)∫ I(Q)[J0(Qδ)−1] Q dQ = ln(P/P0)/(λ²t) (x axis: spin-echo length δ).\n"
            "Exp(λ²t·H{…}/2π): SESANS, the polarisation P/P0 (λ = t = 1 gives SASfit's Exp(H{…}/2π)).\n"
            "MSAS: the intensity with multiple small-angle scattering of a slab of thickness t.\n"
            "Units: I(Q) in 1/cm, t in cm, λ in the reciprocal unit of Q (nm for Q in 1/nm)."
        )
        g.addWidget(QtWidgets.QLabel("transform"), 0, 0)
        g.addWidget(self.kind, 0, 1, 1, 3)
        self.thickness = QtWidgets.QLineEdit("0.1")
        self.thickness.setToolTip("sample thickness t in cm (with I(Q) in absolute units 1/cm)")
        self.wavelength = QtWidgets.QLineEdit("0.6")
        self.wavelength.setToolTip("neutron wavelength λ in the reciprocal unit of Q (nm for Q in 1/nm, Å for Q in 1/Å)")
        self.lam_unit = QtWidgets.QLabel("nm")
        self.from_file = QtWidgets.QCheckBox("λ from file")
        self.from_file.setToolTip("use the wavelength of every spin-echo length given in the SESANS file "
                                  "(time-of-flight SESANS: λ varies with δ)")
        self.t_label = QtWidgets.QLabel("thickness t [cm]")
        self.l_label = QtWidgets.QLabel("wavelength λ")
        g.addWidget(self.t_label, 1, 0)
        g.addWidget(self.thickness, 1, 1)
        g.addWidget(self.l_label, 1, 2)
        lam = QtWidgets.QHBoxLayout()
        lam.addWidget(self.wavelength, 1)
        lam.addWidget(self.lam_unit)
        lam.addWidget(self.from_file)
        g.addLayout(lam, 1, 3)
        self.norm = QtWidgets.QComboBox()
        self.norm.addItem("incident beam: × exp(−Σt)  (SASfit)", "incident")
        self.norm.addItem("measured transmission (contains exp(−Σt))", "transmission")
        self.norm.setToolTip(
            "how the data were normalised: to the incident beam / a transmission without the loss by small-angle "
            "scattering (the model is multiplied by exp(−Σt), as SASfit's MSAS) or to the measured transmission, which "
            "already contains exp(−Σt)"
        )
        self.norm_label = QtWidgets.QLabel("data normalised to")
        g.addWidget(self.norm_label, 2, 0)
        g.addWidget(self.norm, 2, 1, 1, 3)
        self.per_decade = QtWidgets.QSpinBox()
        self.per_decade.setRange(0, 1024)
        self.per_decade.setSpecialValueText("auto")
        self.per_decade.setToolTip("points per decade of the Q grid on which I(Q) is evaluated for the Hankel transform "
                                   "(auto: Settings → Integration… → Hankel transform; built in 64 for SESANS, 48 for "
                                   "MSAS); more for sharp features of I(Q)")
        self.method = QtWidgets.QComboBox()
        self.method.addItem("as in Settings → Integration…", "")
        for m in METHODS:
            self.method.addItem(METHOD_LABELS[m], m)
        self.method.setToolTip(
            "numerical Hankel transform:\n"
            "exact on linear pieces: I(Q)·Q linear between the points of a log grid, each piece integrated exactly "
            "against J0; few points of I(Q) (~300), exact resolution cut (default).\n"
            "FFTLog: Hamilton's fast Hankel transform on a log grid (scipy.fft.fht), as SASfit's HANKEL_FFTLOG / "
            "MSAS_FFTLOG; N points of I(Q).\n"
            "QDHT: quasi-discrete Hankel transform on the zeros of J0, as SASfit's HANKEL_QDHT / MSAS_QDHT; spectrally "
            "accurate for smooth I(Q), needs G(r) to vanish inside the support; N points of I(Q).")
        self.n_points = QtWidgets.QSpinBox()
        self.n_points.setRange(0, 65536)
        self.n_points.setSingleStep(256)
        self.n_points.setSpecialValueText("auto")
        self.n_points.setToolTip("number of grid points of FFTLog / QDHT (= evaluations of I(Q)); auto: Settings → "
                                 "Integration… → Hankel transform (built in 2048, as SASfit)")
        self.accept = QtWidgets.QComboBox()
        self.accept.addItem("none", "none")
        self.accept.addItem("Q_max", "q")
        self.accept.addItem("θ_max [rad]", "theta")
        self.accept.setToolTip(
            "SESANS resolution: only neutrons scattered up to the acceptance of the instrument are measured, so both Q "
            "integrals end at Q_max (as SASfit with its resolution column).\n"
            "Q_max: a fixed value in the unit of Q.\n"
            "θ_max: the acceptance angle (Theta_zmax of a .ses file); Q_max = 2π sin(θ_max)/λ for every spin-echo "
            "length, so it changes with λ in time-of-flight SESANS."
        )
        self.q_accept = QtWidgets.QLineEdit()
        self.q_accept.setPlaceholderText("value")
        self.q_accept.setToolTip("Q_max in the unit of Q, or θ_max in rad")
        self.grid_label = QtWidgets.QLabel("Q grid / decade")
        self.acc_label = QtWidgets.QLabel("resolution")
        g.addWidget(self.grid_label, 3, 0)
        g.addWidget(self.per_decade, 3, 1)
        g.addWidget(self.acc_label, 3, 2)
        acc = QtWidgets.QHBoxLayout()
        acc.addWidget(self.accept)
        acc.addWidget(self.q_accept, 1)
        g.addLayout(acc, 3, 3)
        self.method_label = QtWidgets.QLabel("Hankel method")
        self.n_label = QtWidgets.QLabel("N points")
        g.addWidget(self.method_label, 4, 0)
        g.addWidget(self.method, 4, 1)
        g.addWidget(self.n_label, 4, 2)
        g.addWidget(self.n_points, 4, 3)
        self.info = QtWidgets.QLabel("")
        self.info.setWordWrap(True)
        g.addWidget(self.info, 5, 0, 1, 4)
        g.setColumnStretch(1, 1)
        g.setColumnStretch(3, 1)
        self.kind.currentIndexChanged.connect(self._edited)
        self.method.currentIndexChanged.connect(self._edited)
        self.n_points.editingFinished.connect(self._edited)
        self.accept.currentIndexChanged.connect(self._accept_changed)
        self.norm.currentIndexChanged.connect(self._edited)
        self.from_file.toggled.connect(self._edited)
        self.per_decade.editingFinished.connect(self._edited)
        for e in (self.thickness, self.wavelength, self.q_accept):
            e.editingFinished.connect(self._edited)
        self._update_visible()

    # ------------------------------------------------------------------ state
    def set_length_unit(self, unit):
        self.lam_unit.setText(unit)

    def set_settings(self, s, file_lambda=None):
        """Show the settings ``s``; ``file_lambda``: (min, max) of the wavelengths of a SESANS file, or None."""
        s = {**default_settings(), **(s or {})}
        self._block = True
        try:
            self._file_lambda = file_lambda
            self.kind.setCurrentIndex(max(self.kind.findData(s["kind"]), 0))
            self.thickness.setText(f"{float(s['thickness']):g}")
            self.wavelength.setText(f"{float(s['wavelength']):g}")
            self.from_file.setChecked(bool(s.get("lambda_from_file", True)))
            self.norm.setCurrentIndex(max(self.norm.findData(s.get("norm", "incident")), 0))
            self.per_decade.setValue(int(s.get("per_decade") or 0))
            self.method.setCurrentIndex(max(self.method.findData(s.get("method") or ""), 0))
            self.n_points.setValue(int(s.get("n_points") or 0))
            mode = _accept_mode(s)
            self.accept.setCurrentIndex(max(self.accept.findData(mode), 0))
            self._acc_values = {"q": s.get("q_accept"), "theta": s.get("theta_max")}
            self._acc_mode = mode
            v = self._acc_values.get(mode)
            self.q_accept.setText("" if not v else f"{float(v):g}")
        finally:
            self._block = False
        self._update_visible()

    def settings(self):
        """The settings shown (ValueError for a field that is not a valid number)."""
        return {
            "kind": self.kind.currentData(),
            "thickness": _float(self.thickness.text(), "thickness"),
            "wavelength": _float(self.wavelength.text(), "wavelength"),
            "lambda_from_file": self.from_file.isChecked(),
            "norm": self.norm.currentData(),
            "per_decade": int(self.per_decade.value()),
            "method": self.method.currentData(),
            "n_points": int(self.n_points.value()),
            **self._accept_settings(),
        }

    def _accept_settings(self):
        mode = self.accept.currentData()
        vals = dict(getattr(self, "_acc_values", {}))
        if mode in ("q", "theta"):
            vals[mode] = _float(self.q_accept.text(), "Q_max" if mode == "q" else "θ_max")
        self._acc_values, self._acc_mode = vals, mode
        return {"accept": mode, "q_accept": vals.get("q"), "theta_max": vals.get("theta")}

    def _accept_changed(self, *_):
        """Another kind of limit: show its value (each kind keeps its own)."""
        if self._block:
            return
        vals = dict(getattr(self, "_acc_values", {}))
        old = getattr(self, "_acc_mode", "none")
        if old in ("q", "theta"):  # keep what was typed for the previous kind
            try:
                vals[old] = _float(self.q_accept.text(), old, allow_empty=True)
            except ValueError:
                pass
        mode = self.accept.currentData()
        self._acc_values, self._acc_mode = vals, mode
        v = vals.get(mode)
        self._block = True
        self.q_accept.setText("" if not v else f"{float(v):g}")
        self._block = False
        self._update_visible()
        if mode == "none" or v:
            self.changed.emit()

    def set_kind(self, kind):
        self.kind.setCurrentIndex(max(self.kind.findData(kind), 0))

    def show_info(self, text):
        self.info.setText(text)

    # --------------------------------------------------------------- private
    def _update_visible(self):
        kind = self.kind.currentData()
        ses, msas = kind in ("sesans", "sesans_p"), kind == "msas"
        need_tl = kind in ("sesans_p", "msas") or (ses and self._file_lambda is not None)
        for w in (self.t_label, self.thickness, self.l_label, self.wavelength, self.lam_unit):
            w.setVisible(need_tl or ses)
        self.from_file.setVisible(ses and self._file_lambda is not None)
        self.wavelength.setEnabled(not (ses and self._file_lambda is not None and self.from_file.isChecked()))
        for w in (self.norm_label, self.norm):
            w.setVisible(msas)
        from .hankel import defaults

        filon = (self.method.currentData() or defaults()["method"]) == "filon"
        for w in (self.grid_label, self.per_decade):
            w.setVisible(kind != "none" and filon)
        for w in (self.method_label, self.method):
            w.setVisible(kind != "none")
        for w in (self.n_label, self.n_points):
            w.setVisible(kind != "none" and not filon)
        for w in (self.acc_label, self.accept):
            w.setVisible(ses)
        self.q_accept.setVisible(ses and self.accept.currentData() != "none")
        if kind == "sesans":
            tip = "t and λ are used for P/P0 only (G(δ)−G(0) does not depend on them)"
        elif ses and self._file_lambda is not None and self.from_file.isChecked():
            lo, hi = self._file_lambda
            tip = f"λ from the file: {lo:g}" + (f" … {hi:g}" if hi != lo else "")
        else:
            tip = ""
        if kind != "none":
            from .hankel import fast_lengths, is_fast_length

            n = int(self.n_points.value())
            meth = self.method.currentData() or defaults()["method"]
            if meth == "fftlog" and n >= 2 and not is_fast_length(n):
                prev, nxt = fast_lengths(n)
                tip = (tip + "  " if tip else "") + f"N = {n} is a slow FFT length; faster: {prev} or {nxt}."
        if kind == "none":
            self.info.setText("")
        elif tip:
            self.info.setText(tip)

    def _edited(self, *_):
        self._update_visible()
        if not self._block:
            self.changed.emit()
