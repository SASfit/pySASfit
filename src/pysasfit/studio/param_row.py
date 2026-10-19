"""Compact parameter row of the model explorer: label | value field | [slider] | fit checkbox.

Like the SASfit parameter lists the value is typed in (no arrows); the optional slider is switched on for
the whole window with View -> Sliders. The checkbox marks the parameter for fitting; ``add_radio`` adds the
radio button that selects the form-factor parameter which is integrated over the size distribution.
"""

from __future__ import annotations

import math

from PySide6 import QtCore, QtWidgets

__all__ = ["ParameterRow", "slider_range", "clear_layout"]


def slider_range(value, bounds=(-math.inf, math.inf)):
    """Slider range around ``value``: 0..2v (or 2v..0), clipped to the parameter bounds."""
    lo, hi = bounds
    if value > 0:
        a, b = 0.0, 2.0 * value
    elif value < 0:
        a, b = 2.0 * value, 0.0
    else:
        a, b = 0.0, 1.0
    if math.isfinite(lo):
        a = max(a, lo)
    if math.isfinite(hi):
        b = min(b, hi)
    a, b = min(a, value), max(b, value)  # always contain the current value
    if not b > a:
        a, b = value - 1.0, value + 1.0
    return a, b


class ParameterRow(QtWidgets.QWidget):
    """label | value field | (slider) | (radio) | fit checkbox for one model parameter."""

    valueChanged = QtCore.Signal(float)
    fitToggled = QtCore.Signal(bool)
    boundsEdited = QtCore.Signal()
    STEPS = 1000

    def __init__(self, name, value, description="", bounds=(-math.inf, math.inf), sliders=False,
                 fittable=True, dummy=False, parent=None):  # fmt: skip
        super().__init__(parent)
        self.name = name
        self.dummy = bool(dummy)  # SASfit "dummy" parameters are not used by the function
        self.bounds = bounds
        self.value = float(value)
        self.radio = None
        self._silent = False
        self._tip = description or name
        outer = QtWidgets.QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)
        lay = QtWidgets.QHBoxLayout()
        lay.setSpacing(4)
        outer.addLayout(lay)
        self.label = QtWidgets.QLabel(name)
        self.label.setMinimumWidth(64)
        self.label.setToolTip(self._tip)
        self.edit = QtWidgets.QLineEdit()
        self.edit.setFixedWidth(88)
        self.edit.setToolTip(self._tip)
        self.slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.slider.setRange(0, self.STEPS)
        self.slider.setToolTip(self._tip)
        self.slider.setMaximumWidth(110)
        self.slider.setVisible(sliders)
        self.fit = QtWidgets.QCheckBox()
        self.fit.setToolTip("fit this parameter (bounds, constraints: Fit panel)")
        self.fit.setVisible(fittable)
        self._layout = lay
        # bounds of the fit, shown beneath the fit box while it is ticked (empty = unbounded)
        self.bounds_box = QtWidgets.QWidget()
        bl = QtWidgets.QHBoxLayout(self.bounds_box)
        bl.setContentsMargins(0, 0, 0, 2)
        bl.setSpacing(3)
        lo, hi = bounds
        self.lower = QtWidgets.QLineEdit(f"{lo:g}" if math.isfinite(lo) else "")
        self.upper = QtWidgets.QLineEdit(f"{hi:g}" if math.isfinite(hi) else "")
        for e, ph in ((self.lower, "-∞"), (self.upper, "∞")):
            e.setFixedWidth(56)
            e.setPlaceholderText(ph)
            e.setToolTip("bound of the fit (empty: none)")
            e.editingFinished.connect(self._on_bounds)
            e._last = e.text()
        bl.addStretch(1)
        for w in (QtWidgets.QLabel("min"), self.lower, QtWidgets.QLabel("max"), self.upper):
            bl.addWidget(w)
        self.bounds_box.setVisible(False)
        outer.addWidget(self.bounds_box)
        lay.addWidget(self.label)
        lay.addWidget(self.edit)
        lay.addWidget(self.slider)
        lay.addStretch(0)
        lay.addWidget(self.fit)
        self.slider.valueChanged.connect(self._on_slider)
        self.edit.editingFinished.connect(self._on_edit)
        self.fit.toggled.connect(self._on_fit)
        self.a, self.b = slider_range(self.value, bounds)
        self._show(self.value)
        if self.dummy:
            self.set_editable(False)

    # ----------------------------------------------------------------- value
    def _show(self, v):
        self._silent = True
        self.edit.setText(f"{v:.6g}")
        pos = round((v - self.a) / (self.b - self.a) * self.STEPS)
        self.slider.setValue(min(max(pos, 0), self.STEPS))
        self._silent = False

    def set_value(self, v, emit=True, recenter=True):
        self.value = float(v)
        if recenter:
            self.a, self.b = slider_range(self.value, self.bounds)
        self._show(self.value)
        if emit:
            self.valueChanged.emit(self.value)

    def _on_slider(self, pos):
        if self._silent:
            return
        self.value = self.a + (self.b - self.a) * pos / self.STEPS
        self._silent = True
        self.edit.setText(f"{self.value:.6g}")
        self._silent = False
        self.valueChanged.emit(self.value)

    def _on_edit(self):
        try:
            v = float(self.edit.text().replace(",", "."))
        except ValueError:
            v = math.nan
        if not math.isfinite(v):
            self._show(self.value)  # revert
            return
        if v != self.value:
            self.set_value(v)  # typed values re-centre the slider range

    # ------------------------------------------------------------ fit, bounds
    def _on_fit(self, on):
        self.bounds_box.setVisible(bool(on))
        self.fitToggled.emit(bool(on))

    def _on_bounds(self):
        for e in (self.lower, self.upper):
            text = e.text().strip().replace(",", ".")
            try:
                if text:
                    float(text)
            except ValueError:
                e.setText(e._last)  # revert
                continue
            e.setText(text)
            e._last = text
        self.boundsEdited.emit()

    def bounds_text(self):
        """(lower, upper) as typed; '' = no bound."""
        return self.lower.text().strip(), self.upper.text().strip()

    # ------------------------------------------------------------ appearance
    def set_sliders(self, show):
        self.slider.setVisible(bool(show))

    def is_fit(self):
        return self.fit.isChecked()

    def set_fit(self, on, emit=False):
        """Tick / untick the fit box (silent unless ``emit``)."""
        if self.fit.isChecked() == bool(on):
            return
        self.fit.blockSignals(not emit)
        self.fit.setChecked(bool(on))
        self.fit.blockSignals(False)

    def add_radio(self, group):
        """Add the 'distributed' radio button in front of the fit box; returns it."""
        self.radio = QtWidgets.QRadioButton()
        self.radio.setToolTip("integrate over the size distribution: this parameter takes the values x of the distribution")
        group.addButton(self.radio)
        self._layout.insertWidget(self._layout.indexOf(self.fit), self.radio)
        return self.radio

    def set_editable(self, editable, why=""):
        """Disable value, slider and fit box (parameter set by something else); the radio stays usable."""
        if self.dummy:
            editable, why = False, "not used by this function (dummy parameter)"
        for w in (self.label, self.edit, self.slider, self.fit, self.bounds_box):
            w.setEnabled(editable)
        tip = self._tip if editable else (why or self._tip)
        for w in (self.label, self.edit, self.slider):
            w.setToolTip(tip)


def clear_layout(layout):
    while layout.count():
        item = layout.takeAt(0)
        if item.widget() is not None:
            item.widget().deleteLater()
