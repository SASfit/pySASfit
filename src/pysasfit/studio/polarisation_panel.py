"""The "Polarisation" box of the model explorer: what a data set measured with polarised neutrons.

The settings are a plain dictionary (stored per data set and in project files), the fields of
:class:`pysasfit.studio.polarisation.Polarisation`, or None for "not polarised" (the magnetic functions then use
their own ``pol`` parameter, as in SASfit)::

    {"quantity": "off", "P": 0.98, "f": 0.995, "P_f": 1.0, "form": "exact", "flipper_off_is_plus": True,
     "psi_H": 0.0, "average": False, "source": {"P": "SANS-I table (Aswal et al. 2008): ..."}}

Only the instrument values the quantity needs are shown (P for flipper off, P and f for flipper on; nothing for
corrected data). Every value shows where it comes from (file, instrument table, entered).
"""

from __future__ import annotations

from PySide6 import QtCore, QtWidgets

from .polarisation import QUANTITIES, Polarisation

__all__ = ["PolarisationBox", "build_polarisation"]


def build_polarisation(settings):
    """:class:`Polarisation` of ``settings`` (None: not polarised)."""
    if not settings or settings.get("quantity") in (None, "none"):
        return None
    return Polarisation.from_dict(settings)


class PolarisationBox(QtWidgets.QGroupBox):
    """Quantity of the data set, instrument values (P, f, P_f) with their source, field direction."""

    changed = QtCore.Signal()

    def __init__(self, parent=None):
        super().__init__("Polarisation  (polarised neutrons: what the data set measures)", parent)
        self._source = {}
        self._last = {}
        g = QtWidgets.QGridLayout(self)
        g.setContentsMargins(6, 4, 6, 4)
        g.setVerticalSpacing(2)
        self.quantity = QtWidgets.QComboBox()
        self.quantity.addItem("— not polarised (magnetic functions use their 'pol' parameter)", "none")
        for k, (label, _needs) in QUANTITIES.items():
            self.quantity.addItem(label, k)
        self.quantity.setToolTip(
            "What the data set contains.\n"
            "Not corrected (flipper off / on): the model mixes I+ and I− with the beam polarisation P, the flipped\n"
            "fraction f of the flipper and the depolarisation of the sample (P_f) – the same mixing as pySASred.\n"
            "Corrected (I+, I−, I− − I+, the four channels): no instrument values are needed.\n"
            "+ / −: neutron spin parallel / antiparallel to the field (Moon, Riste & Koehler 1969; SASfit and Michels\n"
            "label by the neutron moment, their ++ is −− here). Non-magnetic contributions count with the\n"
            "nuclear weight (0 in the spin-flip channels and in I− − I+)."
        )
        g.addWidget(QtWidgets.QLabel("data set"), 0, 0)
        g.addWidget(self.quantity, 0, 1, 1, 5)
        self.P = QtWidgets.QLineEdit("0.98")
        self.P.setToolTip("beam polarisation P = 2p − 1 (SANS-I default: Aswal et al. 2008, P(λ, collimation))")
        self.f = QtWidgets.QLineEdit("0.995")
        self.f.setToolTip("flipped fraction f of the flipper; flipper efficiency F = 2f − 1 (F = 0.99 → f = 0.995)")
        self.P_f = QtWidgets.QLineEdit("1")
        self.P_f.setToolTip("polarisation after the sample (depolarisation, measured in transmission); 1 = none.\n"
                            "The incident polarisation is replaced by P·r, r = (1 − P_f)/(−ln P_f) (Quan et al. 2021)")
        self.lab_P, self.lab_f, self.lab_Pf = (QtWidgets.QLabel("P"), QtWidgets.QLabel("f"),
                                               QtWidgets.QLabel("P_f"))  # fmt: skip
        for col, (lab, w) in enumerate(((self.lab_P, self.P), (self.lab_f, self.f), (self.lab_Pf, self.P_f))):
            g.addWidget(lab, 1, 2 * col)
            g.addWidget(w, 1, 2 * col + 1)
        self.src = QtWidgets.QLabel("")
        self.src.setWordWrap(True)
        self.src.setStyleSheet("color: gray")
        g.addWidget(self.src, 2, 0, 1, 6)
        self.form = QtWidgets.QComboBox()
        self.form.addItem("exact flipper mixing", "exact")
        self.form.addItem("BerSANS approximation (n−_on = p f)", "bersans")
        self.form.setToolTip("how the flipper mixes the states; BerSANS (Keiderling et al. 2008, eq. 3b) leaves out "
                             "the (1 − p)(1 − f) term")
        self.plus = QtWidgets.QCheckBox("flipper off = +")
        self.plus.setChecked(True)
        self.plus.setToolTip("flipper off gives neutrons with spin parallel to the field at the sample")
        g.addWidget(self.form, 3, 0, 1, 3)
        g.addWidget(self.plus, 3, 3, 1, 3)
        self.psi_H = QtWidgets.QLineEdit("0")
        self.psi_H.setToolTip("field direction on the detector: angle from +q_x, counter-clockwise (pySASfit frame); "
                              "the ψ parameter of magnetic functions is measured from here")
        self.average = QtWidgets.QCheckBox("azimuthal average (1D radial average: sin²ψ → 1/2, sin⁴ψ → 3/8)")
        g.addWidget(QtWidgets.QLabel("field ψ_H [°]"), 4, 0)
        g.addWidget(self.psi_H, 4, 1)
        g.addWidget(self.average, 4, 2, 1, 4)
        self._geo = (self.psi_H, self.average, self.form, self.plus)
        self.quantity.currentIndexChanged.connect(self._quantity_changed)
        for w in (self.P, self.f, self.P_f, self.psi_H):
            w.editingFinished.connect(self._value_edited)
        for w in (self.form, self.plus, self.average):
            (w.currentIndexChanged if isinstance(w, QtWidgets.QComboBox) else w.toggled).connect(self._emit)
        self._update_visible()

    # ------------------------------------------------------------------ settings
    def settings(self):
        """The settings dictionary (None: not polarised); ValueError for a bad number."""
        q = self.quantity.currentData()
        if q == "none":
            return None

        def num(w, name, lo, hi):
            try:
                v = float(w.text().replace(",", "."))
            except ValueError:
                raise ValueError(f"polarisation: {name} must be a number") from None
            if not lo <= v <= hi:
                raise ValueError(f"polarisation: {name} must be in [{lo}, {hi}]")
            return v

        return {"quantity": q, "P": num(self.P, "P", 1e-6, 1.0), "f": num(self.f, "f", 0.5, 1.0),
                "P_f": num(self.P_f, "P_f", 1e-6, 1.0), "form": self.form.currentData(),
                "flipper_off_is_plus": self.plus.isChecked(), "psi_H": num(self.psi_H, "ψ_H", -360.0, 360.0),
                "average": self.average.isChecked(), "source": dict(self._source)}  # fmt: skip

    def set_settings(self, s):
        widgets = (self.quantity, self.P, self.f, self.P_f, self.form, self.plus, self.psi_H, self.average)
        for w in widgets:
            w.blockSignals(True)
        try:
            s = dict(s or {})
            q = s.get("quantity") or "none"
            i = self.quantity.findData(q)
            self.quantity.setCurrentIndex(max(i, 0))
            p = Polarisation.from_dict({k: v for k, v in s.items() if k != "quantity"})
            self.P.setText(f"{p.P:g}")
            self.f.setText(f"{p.f:g}")
            self.P_f.setText(f"{p.P_f:g}")
            self.form.setCurrentIndex(max(self.form.findData(p.form), 0))
            self.plus.setChecked(bool(p.flipper_off_is_plus))
            self.psi_H.setText(f"{p.psi_H:g}")
            self.average.setChecked(bool(p.average))
            self._source = dict(s.get("source") or {})
            self._last = {"P": self.P.text(), "f": self.f.text(), "P_f": self.P_f.text()}
        finally:
            for w in widgets:
                w.blockSignals(False)
        self._update_visible()

    # ------------------------------------------------------------------ internals
    def _update_visible(self):
        q = self.quantity.currentData()
        needs = QUANTITIES.get(q, ("", ()))[1]
        for name, lab, w in (("P", self.lab_P, self.P), ("f", self.lab_f, self.f), ("P_f", self.lab_Pf, self.P_f)):
            lab.setVisible(name in needs)
            w.setVisible(name in needs)
        on = q != "none"
        for w in self._geo:
            w.setVisible(on)
        self.form.setVisible(q == "on")
        self.plus.setVisible(q in ("off", "on"))
        txt = [f"{k}: {self._source[k]}" for k in needs if self._source.get(k)]
        missing = [k for k in needs if k not in self._source]
        if missing and q in ("off", "on"):
            txt.append(f"{', '.join(missing)}: no value from the file – check the entry")
        self.src.setText("; ".join(txt))
        self.src.setVisible(bool(txt))

    def _quantity_changed(self, *_):
        self._update_visible()
        self._emit()

    def _value_edited(self):
        w = self.sender()
        for name, widget in (("P", self.P), ("f", self.f), ("P_f", self.P_f)):
            if w is widget and widget.text() != self._last.get(name):
                self._source[name] = "entered"
                self._last[name] = widget.text()
        self._update_visible()
        self._emit()

    def _emit(self, *_):
        self.changed.emit()
