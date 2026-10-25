"""One contribution of the mixture shown in the model explorer.

Laid out like the SASfit parameter window: one tab "size distribution and form factor" (two columns, the
radio button marks the form-factor parameter that is integrated over the distribution) and one tab
"structure factor" (with the approach as radio buttons). Parameters are typed in; every row has a fit
checkbox. The model is a :class:`~pysasfit.studio.ScatteringContribution`; the explorer has one panel per
tab of the mixture.
"""

from __future__ import annotations

import math

from PySide6 import QtCore, QtWidgets

from .model import ScatteringContribution
from .param_row import ParameterRow, clear_layout
from .picker import FunctionPicker

__all__ = ["ContributionPanel", "ROLES", "ROLE_TITLES", "APPROACHES"]

ROLES = ("ff", "sd", "sq")


def is_dummy(name):
    """SASfit marks parameters that a function does not use with the name 'dummy...'."""
    return str(name).lower().startswith("dummy")


ROLE_TITLES = {"ff": "Form factor", "sd": "Size distribution", "sq": "Structure factor"}
APPROACHES = [
    ("monodisperse approximation\nI(Q) = <F²(Q,R)> S(Q)", "monodisperse"),
    ("decoupling approximation\nI(Q) = <F²(Q,R)> + <F(Q,R)>² [S(Q) - 1]", "decoupling"),
    ("local monodisperse approximation (LMA)\nI(Q) = ∫ N(R) F²(Q,R) S(Q,R) dR", "lma"),
]
NO_SD_NOTE = "not used with a size distribution: its amplitude N is the scale"


def _column(title):
    """An empty group box with a vertical layout; the caller adds the picker, the description and the parameter rows."""
    box = QtWidgets.QGroupBox(title)
    lay = QtWidgets.QVBoxLayout(box)
    lay.setContentsMargins(6, 6, 6, 6)
    lay.setSpacing(3)
    return box, lay


class ContributionPanel(QtWidgets.QWidget):
    changed = QtCore.Signal()  # a parameter, the scale or the structure changed: redraw
    structure_changed = QtCore.Signal()  # other functions selected: titles, fit table
    fit_toggled = QtCore.Signal(str, str, bool)  # role ('scale' for the scale), parameter name, checked

    def __init__(self, explorer, ff="ff_sphere", sd=None, sq=None, parent=None):
        super().__init__(parent)
        self.explorer = explorer
        self.funcs = {r: None for r in ROLES}
        self.names = {r: None for r in ROLES}
        self.contrib = None
        self.param_rows = {r: [] for r in ROLES}  # the ParameterRow widgets of each role
        self._dist = 0  # index of the form-factor parameter integrated over the size distribution
        self._radios = QtWidgets.QButtonGroup(self)
        self._approach = QtWidgets.QButtonGroup(self)
        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(2, 2, 2, 2)

        self.scale_row = ParameterRow(
            "scale", 1.0, "scale factor of this contribution in the mixture", (0.0, math.inf), sliders=explorer.show_sliders
        )
        self.scale_row.valueChanged.connect(lambda _v: self.changed.emit())
        self.scale_row.fitToggled.connect(lambda on: self.fit_toggled.emit("scale", "scale", on))
        self.scale_row.boundsEdited.connect(lambda: self.fit_toggled.emit("scale", "scale", self.scale_row.is_fit()))
        lay.addWidget(self.scale_row)

        self.picker = {
            "ff": FunctionPicker(placeholder="(select a form factor)"),
            "sd": FunctionPicker(none_label="(none)"),
            "sq": FunctionPicker(none_label="(none)"),
        }
        for role, pk in self.picker.items():
            pk.set_menu(explorer._menu_tree(role))
            pk.changed.connect(lambda _name, r=role: self._apply_roles((r,)))

        self.inner = QtWidgets.QTabWidget()
        lay.addWidget(self.inner, 1)

        # ---- tab 1: size distribution | form factor
        page = QtWidgets.QWidget()
        cols = QtWidgets.QHBoxLayout(page)
        cols.setContentsMargins(2, 2, 2, 2)
        self.group, self.group_layout, self.brief = {}, {}, {}
        for role in ("sd", "ff"):
            box, blay = _column(ROLE_TITLES[role])
            blay.addWidget(self.picker[role])
            self.brief[role] = QtWidgets.QLabel()
            self.brief[role].setWordWrap(True)
            self.brief[role].setStyleSheet("color: gray;")
            blay.addWidget(self.brief[role])
            head = QtWidgets.QLabel("distr.     fit" if role == "ff" else "fit")
            head.setAlignment(QtCore.Qt.AlignmentFlag.AlignRight)
            blay.addWidget(head)
            self.group_layout[role] = QtWidgets.QVBoxLayout()
            self.group_layout[role].setSpacing(2)
            blay.addLayout(self.group_layout[role])
            blay.addStretch(1)
            self.group[role] = box
            scroll = QtWidgets.QScrollArea()
            scroll.setWidgetResizable(True)
            scroll.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
            scroll.setWidget(box)
            cols.addWidget(scroll, 1)
        self.inner.addTab(page, "size distrib. and form factor")

        # ---- tab 2: structure factor | approach
        page = QtWidgets.QWidget()
        cols = QtWidgets.QHBoxLayout(page)
        cols.setContentsMargins(2, 2, 2, 2)
        box, blay = _column(ROLE_TITLES["sq"])
        blay.addWidget(self.picker["sq"])
        self.brief["sq"] = QtWidgets.QLabel()
        self.brief["sq"].setWordWrap(True)
        self.brief["sq"].setStyleSheet("color: gray;")
        blay.addWidget(self.brief["sq"])
        self.group_layout["sq"] = QtWidgets.QVBoxLayout()
        self.group_layout["sq"].setSpacing(2)
        blay.addLayout(self.group_layout["sq"])
        blay.addStretch(1)
        self.group["sq"] = box
        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        scroll.setWidget(box)
        cols.addWidget(scroll, 1)
        abox, alay = _column("Approach (with a size distribution)")
        self.approach_buttons = {}
        for text, data in APPROACHES:
            b = QtWidgets.QRadioButton(text)
            self._approach.addButton(b)
            self.approach_buttons[data] = b
            alay.addWidget(b)
        self.approach_buttons["monodisperse"].setChecked(True)
        alay.addStretch(1)
        cols.addWidget(abox, 1)
        self.approach_box = abox
        self.inner.addTab(page, "structure factor")
        self._approach.buttonToggled.connect(lambda _b, on: on and self._on_structure_option())
        self._radios.buttonToggled.connect(self._on_radio)
        self._initial(ff, sd, sq)

    # --------------------------------------------------------------- selection
    def approach(self):
        return next((k for k, b in self.approach_buttons.items() if b.isChecked()), "monodisperse")

    def set_approach(self, name):
        self.approach_buttons.get(name, self.approach_buttons["monodisperse"]).setChecked(True)

    def _set_combo(self, role, name):
        """Select ``name`` in the picker of ``role`` without updating the model; False if it is not in the menu."""
        return self.picker[role].set_current(name)

    def _initial(self, ff, sd, sq):
        ex = self.explorer
        if "ff_sphere" in ex._entries:
            self._set_combo("ff", "ff_sphere")
        for role, name in (("ff", ff), ("sd", sd), ("sq", sq)):
            if not name:
                continue
            try:
                full = ex._resolve(name)
                if not self._set_combo(role, full):
                    raise KeyError(f"{full} is not a {ROLE_TITLES[role].lower()} function")
            except KeyError as exc:
                ex._status(str(exc.args[0]), error=True)
        self._apply_roles(ROLES)

    def refresh_catalogue(self):
        """The function lists changed (other SASfit folder): new menus, same selection where possible."""
        ex = self.explorer
        current = {r: self.picker[r].current() for r in ROLES}
        for role in ROLES:
            pk = self.picker[role]
            pk.set_menu(ex._menu_tree(role))
            name = current[role] if current[role] in ex._entries else None
            if role == "ff" and name is None and "ff_sphere" in ex._entries:
                name = "ff_sphere"
            pk.set_current(name)
        self.funcs = {r: None for r in ROLES}
        self.names = {r: None for r in ROLES}
        self.contrib = None
        self._apply_roles(ROLES)

    def _apply_roles(self, changed):
        ex = self.explorer
        names = {r: self.picker[r].current() for r in ROLES}
        funcs = dict(self.funcs)
        try:
            for r in changed:
                funcs[r] = ex._function(names[r]) if names[r] else None
            if funcs["ff"] is None:
                raise ValueError("no form factor selected")
        except Exception as exc:  # noqa: BLE001 -- show it, keep the previous model
            ex._error(exc)
            for r in changed:
                if self.names[r] is not None or r != "ff":
                    self._set_combo(r, self.names[r])
            return
        old_ff = self.funcs["ff"]
        self.funcs, self.names = funcs, names
        if "ff" in changed:
            self._choose_distributed(old_ff)
        self._make_contribution()
        for r in changed:
            self._fill_group(r)
        self._update_enabled()
        self.structure_changed.emit()
        self.changed.emit()

    def _choose_distributed(self, old_ff):
        """Which form-factor parameter is integrated over the distribution: keep the name, else R…, else the first."""
        f = self.funcs["ff"]
        names = f.parameter_names if f else []
        prev = old_ff.parameter_names[self._dist] if old_ff is not None and self._dist < len(old_ff.parameter_names) else None
        if prev in names:
            self._dist = names.index(prev)
        else:
            self._dist = next((i for i, n in enumerate(names) if n.upper().startswith("R") and not is_dummy(n)), 0)

    def _on_radio(self, button, on):
        if not on or self._filling:
            return
        for i, row in enumerate(self.param_rows["ff"]):
            if row.radio is button:
                self._dist = i
        self._on_structure_option()

    _filling = False

    def _update_enabled(self):
        sd, sq = self.funcs["sd"], self.funcs["sq"]
        self.approach_box.setEnabled(sd is not None and sq is not None)
        # with a size distribution its amplitude N is the scale: the contribution scale is fixed to 1
        if sd is not None:
            self.scale_row.set_value(1.0, emit=False)
        self.scale_row.setVisible(sd is None)
        self._mark_distributed()

    def _mark_distributed(self):
        """The form-factor parameter that is integrated over the size distribution is not a free parameter."""
        sd = self.funcs["sd"]
        for i, row in enumerate(self.param_rows["ff"]):
            used = sd is not None and i == self._dist
            row.set_editable(not used, "integrated over the size distribution: set by the distribution")
            if used:
                row.set_fit(False)
            if row.radio is not None:
                row.radio.setEnabled(sd is not None and not row.dummy)

    def _make_contribution(self):
        ff, sd, sq = (self.funcs[r] for r in ROLES)
        if ff is None:
            self.contrib = None
            return
        old = self.contrib

        def keep(role):  # keep the user's values of functions that did not change
            f = self.funcs[role]
            of = old.function_for(role) if old is not None else None
            return old.vector(role).copy() if f is not None and of is not None and of.name == f.name else None

        self.contrib = ScatteringContribution(
            ff, keep("ff"), sd, keep("sd"), sq, keep("sq"),
            distributed=max(self._dist, 0),
            approach=self.approach(),
            **self.explorer.integration,
        )  # fmt: skip

    def _on_structure_option(self):
        if self.contrib is not None:
            self._make_contribution()
            self._mark_distributed()
            self.structure_changed.emit()
            self.changed.emit()

    def apply_integration(self, settings):
        """Use new settings of the size-distribution integral."""
        if self.contrib is not None:
            for k, v in settings.items():
                setattr(self.contrib, k, v)

    # --------------------------------------------------------------- parameters
    def _fill_group(self, role):
        lay = self.group_layout[role]
        clear_layout(lay)
        self.param_rows[role] = []
        f, c = self.funcs[role], self.contrib
        # the column holds the picker: it stays visible, also while "(none)" is selected
        self.group[role].setVisible(True)
        self.brief[role].setText("")
        if f is None or c is None:
            return
        self.brief[role].setText(getattr(f.info, "brief", "") or "")
        vec = c.vector(role)
        if not f.info.parameters:
            lay.addWidget(QtWidgets.QLabel("(no documented parameters)"))
        self._filling = True
        try:
            for i, p in enumerate(f.info.parameters):
                if i >= len(vec):
                    break
                row = ParameterRow(
                    p.name, float(vec[i]), p.description, p.bounds, sliders=self.explorer.show_sliders, dummy=is_dummy(p.name)
                )
                row.valueChanged.connect(lambda v, r=role, k=i: self._set_param(r, k, v))
                row.fitToggled.connect(lambda on, r=role, n=p.name: self.fit_toggled.emit(r, n, on))
                row.boundsEdited.connect(lambda r=role, n=p.name, w=row: self.fit_toggled.emit(r, n, w.is_fit()))
                if role == "ff":
                    row.add_radio(self._radios)
                    row.radio.setChecked(i == self._dist)
                lay.addWidget(row)
                self.param_rows[role].append(row)
        finally:
            self._filling = False

    def _set_param(self, role, k, value):
        if self.contrib is not None:
            self.contrib.vector(role)[k] = value
            self.changed.emit()

    def sync_rows(self):
        """Show the values of the parameter vectors in the fields (after a fit or a copy)."""
        c = self.contrib
        if c is None:
            return
        for role in ROLES:
            vec = c.vector(role)
            for k, row in enumerate(self.param_rows[role]):
                if vec is not None and k < len(vec):
                    row.set_value(vec[k], emit=False)

    def set_sliders(self, show):
        self.scale_row.set_sliders(show)
        for rows in self.param_rows.values():
            for row in rows:
                row.set_sliders(show)

    def reset(self):
        """Default parameters of the selected functions, scale 1."""
        self.contrib = None  # forget the values: defaults again
        self._make_contribution()
        for r in ROLES:
            self._fill_group(r)
        self._update_enabled()
        self.scale_row.set_value(1.0, emit=False)
        self.structure_changed.emit()
        self.changed.emit()

    def copy_from(self, other):
        """Take the functions' parameter values, options and scale of another panel (same functions selected)."""
        self._dist = other._dist
        self.set_approach(other.approach())
        self._make_contribution()
        if self.contrib is not None and other.contrib is not None:
            for role in ROLES:
                a, b = self.contrib.vector(role), other.contrib.vector(role)
                if a is not None and b is not None and self.funcs[role].name == other.funcs[role].name:
                    a[:] = b
        self._fill_group("ff")
        self._update_enabled()
        self.scale_row.set_value(other.scale_row.value, emit=False)
        self.sync_rows()
        self.changed.emit()

    # ------------------------------------------------------------------- info
    @property
    def scale(self):
        """Scale of the contribution in the mixture: fixed to 1 with a size distribution (its amplitude N is the scale)."""
        return 1.0 if self.funcs["sd"] is not None else self.scale_row.value

    def distributed_index(self):
        """Index of the form-factor parameter integrated over the size distribution, or -1 without a distribution."""
        return self._dist if self.funcs["sd"] is not None else -1

    def label(self):
        f = self.funcs["ff"]
        return f.info.label if f is not None else "(none)"

    def param_dict(self, role):
        f, vec = self.funcs[role], self.contrib.vector(role)
        return {p.name: float(vec[i]) for i, p in enumerate(f.info.parameters) if i < len(vec)}
