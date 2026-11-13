"""Qt dialog for the integration settings of the explorer.

Three tabs:

* "SASfit library": strategy, accuracy and number of points of the integrals inside the SASfit functions
  (global settings of libsasfit, see :func:`pysasfit.studio.library.configure_precision`);
* "Size distribution (scipy)": the method for the integral over the size distribution
  (:mod:`pysasfit.studio.integration`);
* "Hankel transform (SESANS, MSAS)": method and grids of the transforms of :mod:`pysasfit.studio.hankel`
  (:data:`~pysasfit.studio.hankel.DEFAULTS`).

    dlg = IntegrationDialog(sasfit_values, scipy_values, hankel=hankel_values)   # dicts; missing entries: defaults
    if dlg.exec():
        dlg.sasfit_result(), dlg.scipy_result(), dlg.hankel_result(), dlg.remember.isChecked()
"""

from __future__ import annotations

import math

from PySide6 import QtCore, QtWidgets

from . import hankel as hk
from . import integration
from .library import DEFAULT_PRECISION, HANKEL, QUADRATURE, ROOTALG, SPHAVG

__all__ = ["IntegrationDialog", "SASFIT_CHOICES", "SASFIT_FLOATS", "SASFIT_INTS", "HANKEL_FIELDS"]

HANKEL_FIELDS = (  # (key, label, tooltip) of the numbers in the Hankel tab (the method is a combo box)
    ("sesans_per_decade", "SESANS: Q points / decade", "exact method: points per decade of the Q grid of SESANS"),
    ("sesans_q_min_factor", "SESANS: Q_min · δ_max", "the Q grid starts at this value / δ_max"),
    ("sesans_q_max_factor", "SESANS: Q_max · δ_min", "the Q grid ends at this value / δ_min (unless a resolution Q_max is set)"),
    ("msas_per_decade", "MSAS: Q points / decade", "exact method: points per decade of the Q grid of MSAS (evaluations of I(Q))"),
    ("msas_r_per_decade", "MSAS: r points / decade", "exact method: points per decade of the r grid (cheap; the back "
                                                     "transform needs a fine grid for a large dynamic range of I(Q))"),
    ("msas_q_min_factor", "MSAS: Q grid from Q_min ×", "the Q grid starts at this factor × the smallest q of the data"),
    ("msas_q_max_factor", "MSAS: Q grid to Q_max ×", "the Q grid ends at this factor × the largest q of the data"),
    ("msas_r_min_factor", "MSAS: r grid from … / k_max", "the r grid starts at this value / the end of the Q grid"),
    ("msas_r_max_factor", "MSAS: r grid to … / k_min", "the r grid ends at this value / the start of the Q grid"),
    ("tail_power", "Porod tail exponent (0: none)", "I(Q) ∝ Q^-p above the Q grid completes G(0) (exact method; 4 = Porod)"),
    ("n_points", "FFTLog / QDHT: N points", "grid points of FFTLog and QDHT = evaluations of I(Q) (SASfit: 2048)"),
    ("fft_low", "FFTLog: Q grid from … / δ_max (× Q_min)", "start of the FFTLog grid: this value / δ_max (SESANS) or × Q_min (MSAS)"),
    ("sesans_fft_high", "FFTLog SESANS: to … / δ_min", "end of the FFTLog grid of SESANS: this value / δ_min"),
    ("msas_fft_high", "FFTLog MSAS: to Q_max ×", "end of the FFTLog grid of MSAS: this factor × Q_max"),
    ("qdht_margin", "QDHT SESANS: support × δ_max", "QDHT assumes G(r) = 0 beyond this factor × δ_max"),
    ("qdht_r_factor", "QDHT MSAS: support … / Q_min", "QDHT assumes G(r) = 0 beyond this value / Q_min of the data"),
)

SASFIT_CHOICES = (  # (setting, label, {name: value}) of the libsasfit settings that choose a strategy
    ("int_strategy", "integration strategy", QUADRATURE),
    ("sphavg_strategy", "orientation average", SPHAVG),
    ("hankel_strategy", "Hankel transform", HANKEL),
    ("rootalg", "root finder", ROOTALG),
)
SASFIT_FLOATS = ("nriq", "aniso", "res", "h", "N_Ogata", "h_Ogata")
SASFIT_INTS = (
    "jmax_nriq", "jmax_aniso", "jmax_res", "gausslegendre", "fibonacci", "lebedev", "spherical_t_design",
    "iter_4_mc", "sg_level", "sincos_quad_strategy",
)  # fmt: skip


def _name_of(table, value):
    """Name of a strategy given as name or number (None if the number is not in the table)."""
    if isinstance(value, str):
        return value
    return next((k for k, v in table.items() if v == value), None)


class IntegrationDialog(QtWidgets.QDialog):
    """Integration settings: libsasfit (strategies, accuracy, points) and the scipy method for the size distribution."""

    def __init__(self, sasfit, scipy_settings, parent=None, hankel=None):
        super().__init__(parent)
        self.setWindowTitle("Integration")
        self.setMinimumWidth(620)
        self._sasfit, self._scipy, self._hankel = {}, {}, {}
        lay = QtWidgets.QVBoxLayout(self)
        tabs = QtWidgets.QTabWidget()
        lay.addWidget(tabs, 1)

        # ---- libsasfit
        page = QtWidgets.QWidget()
        pl = QtWidgets.QVBoxLayout(page)
        note = QtWidgets.QLabel(
            "Global settings of libsasfit: they apply to every SASfit function that uses SASfit's common "
            "integration routines (orientation averages, Hankel transforms, general integrals, ...)."
        )
        note.setWordWrap(True)
        pl.addWidget(note)
        form = QtWidgets.QFormLayout()
        self.choice, self.floats, self.ints = {}, {}, {}
        for key, label, table in SASFIT_CHOICES:
            cb = QtWidgets.QComboBox()
            for name in table:
                cb.addItem(name + ("   (stochastic)" if "_MC_" in name else ""), name)
            cb.setToolTip(f"SASfit setting '{key}'")
            self.choice[key] = cb
            form.addRow(label, cb)
        for key in SASFIT_FLOATS:
            edit = QtWidgets.QLineEdit()
            edit.setToolTip(f"SASfit setting '{key}' (default {DEFAULT_PRECISION[key]!r})")
            self.floats[key] = edit
            form.addRow(key, edit)
        for key in SASFIT_INTS:
            spin = QtWidgets.QSpinBox()
            spin.setRange(0, 2_000_000_000)
            spin.setToolTip(f"SASfit setting '{key}' (default {DEFAULT_PRECISION[key]!r})")
            self.ints[key] = spin
            form.addRow(key, spin)
        pl.addLayout(form)
        pl.addStretch(1)
        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(page)
        tabs.addTab(scroll, "SASfit library")

        # ---- scipy / size distribution
        page2 = QtWidgets.QWidget()
        p2 = QtWidgets.QVBoxLayout(page2)
        note2 = QtWidgets.QLabel(
            "Integral over the size distribution (form factor integrated over the distributed parameter), "
            "done in Python. Scipy cannot replace the integrals inside the SASfit functions (other tab). "
            "Method 'sasfit' uses libsasfit's own integrator with the 'integration strategy' of the other tab."
        )
        note2.setWordWrap(True)
        p2.addWidget(note2)
        form2 = QtWidgets.QFormLayout()
        self.method = QtWidgets.QComboBox()
        for i, (name, text) in enumerate(integration.METHODS.items()):
            self.method.addItem(name, name)
            self.method.setItemData(i, text, QtCore.Qt.ItemDataRole.ToolTipRole)
        self.method_note = QtWidgets.QLabel()
        self.method_note.setWordWrap(True)
        self.n_x = QtWidgets.QSpinBox()
        self.n_x.setRange(2, 5000)
        self.n_x.setToolTip("Gauss-Legendre: nodes per panel (4 panels); equidistant methods: 4 x n_x points")
        self.rtol = QtWidgets.QLineEdit()
        self.atol = QtWidgets.QLineEdit()
        self.limit = QtWidgets.QSpinBox()
        self.limit.setRange(1, 100000)
        self.limit.setToolTip("quad_vec, sasfit: maximum number of subdivisions")
        form2.addRow("method", self.method)
        form2.addRow("", self.method_note)
        form2.addRow("n_x", self.n_x)
        form2.addRow("rtol", self.rtol)
        form2.addRow("atol", self.atol)
        form2.addRow("limit", self.limit)
        self.x_power = QtWidgets.QDoubleSpinBox()
        self.x_power.setRange(0.0, 12.0)
        self.x_power.setDecimals(1)
        self.x_power.setSingleStep(1.0)
        self.x_power.setToolTip(
            "Range search: the intensity grows like x^2 ... x^6 with the size x, so the tail cut-off uses "
            "N(x) * x^power (6: large particles are not clipped; 0: N(x) alone)"
        )
        self.tail = QtWidgets.QLineEdit()
        self.tail.setToolTip("Range search: cut where N(x) * x^power falls below this fraction of its maximum")
        form2.addRow("range: x power", self.x_power)
        form2.addRow("range: tail cut-off", self.tail)
        p2.addLayout(form2)
        p2.addStretch(1)
        tabs.addTab(page2, "Size distribution (scipy)")
        self.method.currentIndexChanged.connect(self._method_changed)

        # ---- Hankel transforms of SESANS and MSAS (Python)
        page3 = QtWidgets.QWidget()
        p3 = QtWidgets.QVBoxLayout(page3)
        note3 = QtWidgets.QLabel(
            "Hankel transforms of the SESANS and multiple-scattering (MSAS) transforms of the model explorer (Transform "
            "box), done in Python. These are the defaults; a data set can choose another method and number of points in "
            "its Transform box. The 'Hankel transform' strategy of the SASfit library tab applies only inside the SASfit "
            "functions."
        )
        note3.setWordWrap(True)
        p3.addWidget(note3)
        form3 = QtWidgets.QFormLayout()
        self.h_method = QtWidgets.QComboBox()
        for m in hk.METHODS:
            self.h_method.addItem(hk.METHOD_LABELS[m], m)
        self.h_method.setToolTip(
            "exact on linear pieces: I(Q)·Q linear between the points of a log grid, every piece integrated exactly "
            "against J0 (~300 evaluations of I(Q), exact resolution cut).\nFFTLog: Hamilton's fast Hankel transform "
            "(scipy.fft.fht), as SASfit's HANKEL_FFTLOG.\nQDHT: quasi-discrete Hankel transform on the zeros of J0, as "
            "SASfit's HANKEL_QDHT."
        )
        form3.addRow("method", self.h_method)
        self.h_fields = {}
        for key, label, tip in HANKEL_FIELDS:
            edit = QtWidgets.QLineEdit()
            edit.setToolTip(f"{tip} (built in {hk.DEFAULTS[key]!r})")
            self.h_fields[key] = edit
            form3.addRow(label, edit)
            if key == "n_points":  # FFTLog: a suggestion only, the value entered is used
                self.h_nhint = QtWidgets.QLabel("")
                self.h_nhint.setWordWrap(True)
                self.h_nhint.setStyleSheet("color: #8a5a00;")
                form3.addRow("", self.h_nhint)
                edit.textChanged.connect(self._n_hint)
        p3.addLayout(form3)
        p3.addStretch(1)
        scroll3 = QtWidgets.QScrollArea()
        scroll3.setWidgetResizable(True)
        scroll3.setWidget(page3)
        tabs.addTab(scroll3, "Hankel transform (SESANS, MSAS)")
        self.tabs = tabs
        self._hankel_in = dict(hankel) if hankel is not None else hk.defaults()

        self.error = QtWidgets.QLabel()
        self.error.setStyleSheet("color: #b00020;")
        self.error.setWordWrap(True)
        lay.addWidget(self.error)
        self.remember = QtWidgets.QCheckBox("Remember these settings (next sessions and scripts use them too)")
        self.remember.setChecked(True)
        lay.addWidget(self.remember)
        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Ok
            | QtWidgets.QDialogButtonBox.StandardButton.Cancel
            | QtWidgets.QDialogButtonBox.StandardButton.RestoreDefaults
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        defaults = buttons.button(QtWidgets.QDialogButtonBox.StandardButton.RestoreDefaults)
        defaults.setToolTip("SASfit, scipy and Hankel-transform defaults")
        defaults.clicked.connect(lambda: self._load(DEFAULT_PRECISION, integration.DEFAULTS, hk.DEFAULTS))
        lay.addWidget(buttons)
        self._load(sasfit, scipy_settings, self._hankel_in)

    # ------------------------------------------------------------------ values
    def _load(self, sasfit, scipy_settings, hankel=None):
        """Show the settings (dicts; missing entries: defaults)."""
        for key, _label, table in SASFIT_CHOICES:
            value = sasfit.get(key, DEFAULT_PRECISION[key])
            name = _name_of(table, value)
            cb = self.choice[key]
            idx = cb.findData(name) if name is not None else -1
            if idx < 0:  # a number the table does not know: keep it selectable
                cb.addItem(f"{value} (not in the list)", value)
                idx = cb.count() - 1
            cb.setCurrentIndex(idx)
        for key, edit in self.floats.items():
            edit.setText(repr(float(sasfit.get(key, DEFAULT_PRECISION[key]))))
        for key, spin in self.ints.items():
            spin.setValue(int(sasfit.get(key, DEFAULT_PRECISION[key])))
        s = {**integration.DEFAULTS, **scipy_settings}
        self.method.setCurrentIndex(max(self.method.findData(s["method"]), 0))
        self.n_x.setValue(int(s["n_x"]))
        self.rtol.setText(repr(float(s["rtol"])))
        self.atol.setText(repr(float(s["atol"])))
        self.limit.setValue(int(s["limit"]))
        self.x_power.setValue(float(s["x_power"]))
        self.tail.setText(repr(float(s["tail"])))
        h = {**hk.DEFAULTS, **(hankel or {})}
        self.h_method.setCurrentIndex(max(self.h_method.findData(h["method"]), 0))
        for key, edit in self.h_fields.items():
            v = h[key]
            edit.setText(str(int(v)) if isinstance(hk.DEFAULTS[key], int) else f"{float(v):g}")
        self.error.clear()
        self._method_changed()

    def _method_changed(self, *_):
        name = self.method.currentData()
        self.method_note.setText(integration.METHODS.get(name, ""))
        adaptive = name in ("quad_vec", "sasfit")  # these two use rtol, atol and limit
        self.n_x.setEnabled(not adaptive)
        for w in (self.rtol, self.atol, self.limit):
            w.setEnabled(adaptive)

    @staticmethod
    def _number(edit, name, positive=False):
        try:
            v = float(edit.text().replace(",", "."))
        except ValueError:
            raise ValueError(f"{name}: not a number") from None
        if not math.isfinite(v) or v < 0 or (positive and v == 0):
            raise ValueError(f"{name}: must be {'> 0' if positive else '>= 0'}")
        return v

    def accept(self):
        try:
            sasfit = {k: cb.currentData() for k, cb in self.choice.items()}
            sasfit.update({k: self._number(e, k) for k, e in self.floats.items()})
            sasfit.update({k: sp.value() for k, sp in self.ints.items()})
            scipy_settings = {
                "method": self.method.currentData(),
                "n_x": self.n_x.value(),
                "rtol": self._number(self.rtol, "rtol", positive=True),
                "atol": self._number(self.atol, "atol"),
                "limit": self.limit.value(),
                "x_power": self.x_power.value(),
                "tail": self._number(self.tail, "tail cut-off", positive=True),
            }
            hankel = {"method": self.h_method.currentData()}
            for key, edit in self.h_fields.items():
                v = self._number(edit, key, positive=key != "tail_power")
                if isinstance(hk.DEFAULTS[key], int):
                    if v != int(v) or v < 2:
                        raise ValueError(f"{key}: a whole number >= 2")
                    v = int(v)
                hankel[key] = v
            if 0 < hankel["tail_power"] <= 2:
                raise ValueError("tail_power: must be 0 (no tail) or > 2")
        except ValueError as exc:
            self.error.setText(f"✘ {exc}")
            return  # stay open
        self._sasfit, self._scipy, self._hankel = sasfit, scipy_settings, hankel
        super().accept()

    def _n_hint(self, *_):
        """Suggest the nearest fast FFT lengths when N is a slow one for FFTLog (the value itself is kept)."""
        try:
            n = int(float(self.h_fields["n_points"].text()))
        except ValueError:
            self.h_nhint.setText("")
            return
        if n < 2 or hk.is_fast_length(n):
            self.h_nhint.setText("")
            return
        prev, nxt = hk.fast_lengths(n)
        self.h_nhint.setText(f"{n} is a slow FFT length for FFTLog (factors other than 2, 3, 5); faster: {prev} or "
                             f"{nxt}. Used as entered. (QDHT does not use an FFT: any N.)")

    def hankel_result(self):
        """Hankel-transform defaults (keys of :data:`pysasfit.studio.hankel.DEFAULTS`)."""
        return dict(self._hankel)

    def sasfit_result(self):
        """libsasfit settings: strategy names (or numbers), floats and ints, ready for ``configure_precision``."""
        return dict(self._sasfit)

    def scipy_result(self):
        """Size-distribution integration settings: method, n_x, rtol, atol, limit, x_power, tail (the keys of ``integration.DEFAULTS``)."""
        return dict(self._scipy)
