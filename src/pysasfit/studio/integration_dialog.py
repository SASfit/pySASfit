"""Qt dialog for the integration settings of the explorer.

Two tabs:

* "SASfit library": strategy, accuracy and number of points of the integrals inside the SASfit functions
  (global settings of libsasfit, see :func:`pysasfit.studio.library.configure_precision`);
* "Size distribution (scipy)": the method for the integral over the size distribution
  (:mod:`pysasfit.studio.integration`).

    dlg = IntegrationDialog(sasfit_values, scipy_values)      # dicts; missing entries: defaults
    if dlg.exec():
        dlg.sasfit_result(), dlg.scipy_result(), dlg.remember.isChecked()
"""

from __future__ import annotations

import math

from PySide6 import QtCore, QtWidgets

from . import integration
from .library import DEFAULT_PRECISION, HANKEL, QUADRATURE, ROOTALG, SPHAVG

__all__ = ["IntegrationDialog", "SASFIT_CHOICES", "SASFIT_FLOATS", "SASFIT_INTS"]

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

    def __init__(self, sasfit, scipy_settings, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Integration")
        self.setMinimumWidth(620)
        self._sasfit, self._scipy = {}, {}
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
        defaults.setToolTip("SASfit and scipy defaults")
        defaults.clicked.connect(lambda: self._load(DEFAULT_PRECISION, integration.DEFAULTS))
        lay.addWidget(buttons)
        self._load(sasfit, scipy_settings)

    # ------------------------------------------------------------------ values
    def _load(self, sasfit, scipy_settings):
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
        except ValueError as exc:
            self.error.setText(f"✘ {exc}")
            return  # stay open
        self._sasfit, self._scipy = sasfit, scipy_settings
        super().accept()

    def sasfit_result(self):
        """libsasfit settings: strategy names (or numbers), floats and ints, ready for ``configure_precision``."""
        return dict(self._sasfit)

    def scipy_result(self):
        """Size-distribution integration settings: method, n_x, rtol, atol, limit."""
        return dict(self._scipy)
