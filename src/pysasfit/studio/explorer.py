"""Interactive model explorer for pysasfit.studio (in the spirit of the SasModels explorer app).

Plots a SASfit model -- a form factor, optionally integrated over a size
distribution and multiplied by a structure factor -- on a log-log plot and lets
you change every parameter with a slider or by typing a value. Start with

    pysasfit-studio explore [ff_sphere] [--sq sq_hardsphere] [--sd sd_LogNorm]
    pysasfit-explorer ff_sphere --sq sq_hardsphere
    python -m pysasfit.studio.explorer

or from Python / IPython (``%gui qt``)::

    from pysasfit.studio.explorer import explore
    explore("ff_sphere", sq="sq_hardsphere")

The combination of the three functions is a
:class:`pysasfit.studio.ScatteringContribution` (manual ch. 4), so the curve
shown is exactly what :class:`~pysasfit.studio.ModelFunction` evaluates; the
"Copy Python" button puts the code to reproduce it on the clipboard.

The functions are chosen from cascading menus structured like those of SASfit: the group
files ``*.def`` in the ``def`` subfolder of the plugin directory define the tree, the plugin
headers say where each function belongs (:mod:`pysasfit.studio.groups`); "Find…" searches
all functions by label, name or group.

SASfit functions carry no units: use consistent ones (radius in nm <-> q in
1/nm). The unit selector only changes the axis label.

Requires PySide6 and matplotlib (``pip install pysasfit[gui]``).
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import time
import warnings

import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets

import matplotlib

matplotlib.use("QtAgg")
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402

from . import groups, headers, integration, paths  # noqa: E402
from .functions import _PYTHON_FUNCTIONS, function  # noqa: E402
from .integration_dialog import IntegrationDialog  # noqa: E402
from .library import (  # noqa: E402
    DEFAULT_PRECISION,
    SASfitLibraryNotFound,
    configure_precision,
    reset_library,
    user_precision,
)
from .model import ScatteringContribution  # noqa: E402
from .picker import FunctionPicker  # noqa: E402

__all__ = ["ModelExplorer", "ParameterRow", "explore", "main"]

ROLES = ("ff", "sd", "sq")
ROLE_TITLES = {"ff": "Form factor", "sd": "Size distribution", "sq": "Structure factor"}
FF_TYPES = ("ff", "peak", "peaks")
APPROACHES = [
    ("monodisperse  I = S ∫N P", "monodisperse"),
    ("decoupling approximation", "decoupling"),
    ("local monodisperse (LMA)", "lma"),
]
MODES = [("P*S", "PS"), ("P  (no structure factor)", "P"), ("S  (structure factor only)", "S")]
Q_UNITS = ["1/nm", "1/Å", "1/m"]


# --------------------------------------------------------------------------- helpers
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
    """label | slider | value field for one model parameter."""

    valueChanged = QtCore.Signal(float)
    STEPS = 1000

    def __init__(self, name, value, description="", bounds=(-math.inf, math.inf), parent=None):
        super().__init__(parent)
        self.name = name
        self.bounds = bounds
        self.value = float(value)
        self._silent = False
        lay = QtWidgets.QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        self.label = QtWidgets.QLabel(name)
        self.label.setMinimumWidth(90)
        self.label.setToolTip(description or name)
        self.slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.slider.setRange(0, self.STEPS)
        self.slider.setToolTip(description or name)
        self.edit = QtWidgets.QLineEdit()
        self.edit.setFixedWidth(85)
        lay.addWidget(self.label)
        lay.addWidget(self.slider, 1)
        lay.addWidget(self.edit)
        self.slider.valueChanged.connect(self._on_slider)
        self.edit.editingFinished.connect(self._on_edit)
        self.a, self.b = slider_range(self.value, bounds)
        self._show(self.value)

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


def _clear(layout):
    while layout.count():
        item = layout.takeAt(0)
        if item.widget() is not None:
            item.widget().deleteLater()


def _entries():
    cat = dict(headers.catalogue())
    for name, impl in _PYTHON_FUNCTIONS.items():  # functions registered at run time
        cat[name] = impl[0]
    return cat


# --------------------------------------------------------------------------- SASfit installation
class InstallationDialog(QtWidgets.QDialog):
    """Choose the folder of a local SASfit installation (libsasfit and the plugin libraries)."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("SASfit installation")
        self.setMinimumWidth(680)
        ext = paths.LIB_EXT
        lay = QtWidgets.QVBoxLayout(self)
        intro = QtWidgets.QLabel(
            f"Folder of a local SASfit installation containing libsasfit.{ext} and the plugin libraries "
            f"(plugins/libsasfit_*.{ext}). A folder chosen here takes precedence over the "
            "environment variables listed below."
        )
        intro.setWordWrap(True)
        lay.addWidget(intro)

        row = QtWidgets.QHBoxLayout()
        self.edit = QtWidgets.QLineEdit(paths.chosen_lib_dir() or "")
        self.edit.setPlaceholderText("empty: use the environment variables / bundled libraries")
        browse = QtWidgets.QPushButton("Browse…")
        browse.clicked.connect(self._browse)
        row.addWidget(self.edit, 1)
        row.addWidget(browse)
        lay.addLayout(row)

        self.check = QtWidgets.QLabel()
        self.check.setWordWrap(True)
        lay.addWidget(self.check)
        self.remember = QtWidgets.QCheckBox("Remember this folder (next sessions and scripts use it too)")
        self.remember.setChecked(True)
        lay.addWidget(self.remember)

        gb = QtWidgets.QGroupBox("Search order  (-> = in use)")
        gl = QtWidgets.QVBoxLayout(gb)
        self.sources = QtWidgets.QLabel()
        self.sources.setFont(QtGui.QFontDatabase.systemFont(QtGui.QFontDatabase.SystemFont.FixedFont))
        self.sources.setTextInteractionFlags(QtCore.Qt.TextInteractionFlag.TextSelectableByMouse)
        gl.addWidget(self.sources)
        lay.addWidget(gb)

        self.buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Ok | QtWidgets.QDialogButtonBox.StandardButton.Cancel
        )
        env = self.buttons.addButton("Use environment variable", QtWidgets.QDialogButtonBox.ButtonRole.ActionRole)
        env.setToolTip("forget the chosen folder; use PYSASFIT_SASFIT_LIB / SASFIT_LOCATION / bundled libraries")
        env.clicked.connect(self._use_environment)
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)
        lay.addWidget(self.buttons)

        self.edit.textChanged.connect(self._validate)
        self._validate()

    def path(self):
        """The folder entered ('' = use the environment / bundled libraries)."""
        return self.edit.text().strip().strip('"')

    def _browse(self):
        start = self.path() or os.path.expanduser("~")
        d = QtWidgets.QFileDialog.getExistingDirectory(self, "SASfit installation folder", start)
        if d:
            self.edit.setText(os.path.normpath(d))

    def _use_environment(self):
        self.edit.clear()
        self.accept()

    def _validate(self):
        text = self.path()
        ok = self.buttons.button(QtWidgets.QDialogButtonBox.StandardButton.Ok)
        if not text:
            self.check.setStyleSheet("")
            self.check.setText("No folder chosen: the environment variables / bundled libraries are used.")
            ok.setEnabled(True)
        else:
            found = paths.find_lib_dir(text)
            if found:
                n = paths.count_plugin_libs(found)
                self.check.setStyleSheet("color: #1b7a2b;")
                self.check.setText(f"✔ libsasfit.{paths.LIB_EXT} found in {found}  ({n} plugin libraries)")
                ok.setEnabled(True)
            else:
                self.check.setStyleSheet("color: #b00020;")
                self.check.setText(
                    f"✘ no libsasfit.{paths.LIB_EXT} in this folder (also looked in lib, bin and sasfit_lib)."
                )
                ok.setEnabled(False)
        self.sources.setText(paths.describe_lib_dir_sources(paths.lib_dir_sources(chosen=text or None)))


# --------------------------------------------------------------------------- main window
class ModelExplorer(QtWidgets.QMainWindow):
    def __init__(self, ff="ff_sphere", sd=None, sq=None, qmin=0.01, qmax=1.0, parent=None):
        super().__init__(parent)
        self.setWindowTitle("pySASfit model explorer")
        self.resize(1180, 700)
        self._entries = _entries()
        self._defs = groups.load_groups()
        self.integration = integration.load_settings()  # size-distribution integral (scipy / Gauss-Legendre)
        self._funcs = {r: None for r in ROLES}
        self._names = {r: None for r in ROLES}
        self._cache = {}
        self.contrib = None
        self._last = None
        self._held = []
        self._timer = QtCore.QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.setInterval(40)  # debounce slider drags
        self._timer.timeout.connect(self._redraw)
        self._build_ui(qmin, qmax)
        self._initial(ff, sd, sq)

    # ------------------------------------------------------------------ UI
    def _menu_tree(self, role):
        """Menu tree of the functions of a role: groups from the .def files, functions from the headers."""
        types = FF_TYPES if role == "ff" else (role,)
        return groups.build_menu(self._entries.values(), types, self._defs)

    def _build_ui(self, qmin, qmax):
        menu = self.menuBar().addMenu("&Settings")
        act = menu.addAction("SASfit &installation…")
        act.setShortcut("Ctrl+L")
        act.triggered.connect(self._choose_installation)
        act = menu.addAction("&Integration…")
        act.setShortcut("Ctrl+I")
        act.setToolTip("integration strategy of the SASfit library and of the size-distribution integral (scipy)")
        act.triggered.connect(self._choose_integration)

        split = QtWidgets.QSplitter()
        self.setCentralWidget(split)

        # ---- left: model selection, scale/background, parameters
        left = QtWidgets.QWidget()
        left.setMinimumWidth(470)
        lv = QtWidgets.QVBoxLayout(left)

        inst = QtWidgets.QHBoxLayout()
        self.lib_label = QtWidgets.QLabel()
        self.lib_label.setSizePolicy(QtWidgets.QSizePolicy.Policy.Ignored, QtWidgets.QSizePolicy.Policy.Preferred)
        self.lib_label.setMinimumWidth(50)
        change = QtWidgets.QPushButton("SASfit installation…")
        change.setToolTip("choose the folder of a local SASfit installation (libsasfit + plugin libraries)")
        change.clicked.connect(self._choose_installation)
        inst.addWidget(self.lib_label, 1)
        inst.addWidget(change)
        lv.addLayout(inst)

        box = QtWidgets.QGroupBox("Model")
        form = QtWidgets.QFormLayout(box)
        self.picker = {
            "ff": FunctionPicker(placeholder="(select a form factor)"),
            "sd": FunctionPicker(none_label="(none)"),
            "sq": FunctionPicker(none_label="(none)"),
        }
        for role, pk in self.picker.items():
            pk.set_menu(self._menu_tree(role))
            pk.changed.connect(lambda _name, r=role: self._on_combo(r))
        self.dist_combo = QtWidgets.QComboBox()
        self.mode_combo = QtWidgets.QComboBox()
        for text, data in MODES:
            self.mode_combo.addItem(text, data)
        self.approach_combo = QtWidgets.QComboBox()
        for text, data in APPROACHES:
            self.approach_combo.addItem(text, data)
        form.addRow("Form factor", self.picker["ff"])
        form.addRow("Size distribution", self.picker["sd"])
        form.addRow("   distributed parameter", self.dist_combo)
        form.addRow("Structure factor", self.picker["sq"])
        form.addRow("   show", self.mode_combo)
        form.addRow("   approach (with distribution)", self.approach_combo)
        lv.addWidget(box)
        self.dist_combo.currentIndexChanged.connect(lambda _i: self._on_structure_option())
        self.approach_combo.currentIndexChanged.connect(lambda _i: self._on_structure_option())
        self.mode_combo.currentIndexChanged.connect(self._schedule)

        gbox = QtWidgets.QGroupBox("Intensity")
        gl = QtWidgets.QVBoxLayout(gbox)
        gl.setSpacing(2)
        self.scale_row = ParameterRow("scale", 1.0, "overall scale factor of the model")
        self.bg_row = ParameterRow("background", 0.0, "constant background added to I(q)")
        for row in (self.scale_row, self.bg_row):
            row.valueChanged.connect(self._schedule)
            gl.addWidget(row)
        lv.addWidget(gbox)

        container = QtWidgets.QWidget()
        cl = QtWidgets.QVBoxLayout(container)
        self._group, self._group_layout = {}, {}
        for role in ROLES:
            gb = QtWidgets.QGroupBox(ROLE_TITLES[role])
            lay = QtWidgets.QVBoxLayout(gb)
            lay.setSpacing(2)
            self._group[role], self._group_layout[role] = gb, lay
            cl.addWidget(gb)
        cl.addStretch(1)
        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(container)
        lv.addWidget(scroll, 1)
        split.addWidget(left)

        # ---- right: plot and controls
        right = QtWidgets.QWidget()
        rv = QtWidgets.QVBoxLayout(right)
        self.fig = Figure(figsize=(6, 4), layout="constrained")
        self.ax = self.fig.add_subplot(111)
        self.canvas = FigureCanvasQTAgg(self.fig)
        rv.addWidget(NavigationToolbar2QT(self.canvas, self))
        rv.addWidget(self.canvas, 1)
        (self.line,) = self.ax.plot([qmin, qmax], [1.0, 1.0], lw=1.8)  # placeholder: a log axis needs data to draw
        self.ax.grid(True, which="both", alpha=0.25)

        bar = QtWidgets.QHBoxLayout()
        self.qmin = QtWidgets.QLineEdit(f"{qmin:g}")
        self.qmax = QtWidgets.QLineEdit(f"{qmax:g}")
        for e in (self.qmin, self.qmax):
            e.setFixedWidth(80)
            e.editingFinished.connect(self._schedule)
        self.units = QtWidgets.QComboBox()
        self.units.addItems(Q_UNITS)
        self.units.setToolTip("Label only: SASfit functions are unit-agnostic, use consistent units.")
        self.units.currentIndexChanged.connect(self._update_labels)
        self.npoints = QtWidgets.QSpinBox()
        self.npoints.setRange(10, 5000)
        self.npoints.setValue(200)
        self.npoints.setToolTip("number of q points")
        self.npoints.editingFinished.connect(self._schedule)
        self.logx = QtWidgets.QCheckBox("log q")
        self.logy = QtWidgets.QCheckBox("log I")
        self.autoscale = QtWidgets.QCheckBox("autoscale")
        for c in (self.logx, self.logy, self.autoscale):
            c.setChecked(True)
        self.logx.toggled.connect(self._on_scales)
        self.logy.toggled.connect(self._on_scales)
        self.autoscale.toggled.connect(self._schedule)
        for w in (
            QtWidgets.QLabel("Q min:"), self.qmin, self.units, QtWidgets.QLabel("Q max:"), self.qmax,
            QtWidgets.QLabel("points:"), self.npoints, self.logx, self.logy, self.autoscale,
        ):  # fmt: skip
            bar.addWidget(w)
        bar.addStretch(1)
        rv.addLayout(bar)

        buttons = QtWidgets.QHBoxLayout()
        for text, slot, tip in (
            ("Hold", self._hold, "keep the current curve as a grey reference"),
            ("Clear held", self._clear_held, "remove the reference curves"),
            ("Reset parameters", self._reset, "back to the default parameters of the selected functions"),
            ("Export…", self._export, "save q, I(q) and the parameters as a text file"),
            ("Copy Python", self._copy_python, "copy code that reproduces the curve"),
        ):
            b = QtWidgets.QPushButton(text)
            b.setToolTip(tip)
            b.clicked.connect(slot)
            buttons.addWidget(b)
        buttons.addStretch(1)
        rv.addLayout(buttons)
        self.status = QtWidgets.QLabel("")
        self.status.setWordWrap(True)
        rv.addWidget(self.status)
        split.addWidget(right)
        split.setStretchFactor(0, 0)
        split.setStretchFactor(1, 1)
        split.setSizes([500, 680])
        self._on_scales()

    # --------------------------------------------------------------- selection
    def _status(self, text, error=False):
        self.status.setStyleSheet("color: #b00020;" if error else "")
        self.status.setText(text)

    def _error(self, exc):
        if isinstance(exc, SASfitLibraryNotFound):
            self._status(
                "SASfit libraries not found. Choose the folder of a local SASfit installation with the "
                "'SASfit installation…' button (Ctrl+L), or set PYSASFIT_SASFIT_LIB.",
                error=True,
            )
            return
        self._status(f"{type(exc).__name__}: {exc}", error=True)

    # ------------------------------------------------------- SASfit installation
    def _update_installation_label(self):
        d = paths.lib_dir()
        if d:
            key = paths.lib_dir_source()
            how = {"gui": "chosen folder", "bundled": "bundled"}.get(key, f"environment variable {key}")
            self.lib_label.setText(f"SASfit: {d}  ({how})")
        else:
            self.lib_label.setText("SASfit: not found")
        self.lib_label.setStyleSheet("" if d else "color: #b00020;")
        self.lib_label.setToolTip(paths.describe_lib_dir_sources())

    def _choose_installation(self):
        dlg = InstallationDialog(self)
        if dlg.exec() == QtWidgets.QDialog.DialogCode.Accepted:
            self.set_installation(dlg.path(), save=dlg.remember.isChecked())

    def set_installation(self, path, save=False):
        """Use the SASfit installation in ``path`` ('' = environment variable / bundled) and reload the models."""
        try:
            paths.set_lib_dir(path or None, save=save or not path)
        except OSError as exc:
            QtWidgets.QMessageBox.warning(
                self, "SASfit installation", f"The folder could not be saved ({exc}).\nIt is used for this session only."
            )
        reset_library()
        self._cache.clear()
        self._refresh_catalogue()

    # ------------------------------------------------------- integration
    def _choose_integration(self):
        dlg = IntegrationDialog({**DEFAULT_PRECISION, **user_precision()}, self.integration, self)
        if dlg.exec() != QtWidgets.QDialog.DialogCode.Accepted:
            return
        self._apply_integration(dlg.sasfit_result(), dlg.scipy_result(), save=dlg.remember.isChecked())

    def _apply_integration(self, sasfit, scipy_settings, save=False):
        """Use these libsasfit settings and size-distribution integration settings and redraw."""
        try:
            configure_precision(save=save, **sasfit)
            if save:
                integration.save_settings(scipy_settings)
        except OSError as exc:
            QtWidgets.QMessageBox.warning(
                self, "Integration", f"The settings could not be saved ({exc}).\nThey are used for this session only."
            )
        except (ValueError, KeyError, TypeError) as exc:
            self._error(exc)
            return
        self.integration = {**integration.DEFAULTS, **scipy_settings}
        if self.contrib is not None:
            for k, v in self.integration.items():
                setattr(self.contrib, k, v)
        self._schedule()

    def _refresh_catalogue(self):
        """Re-read the function lists after the library folder changed and rebuild the current model."""
        self._entries = _entries()
        self._defs = groups.load_groups()  # the .def files live next to the plugins and may have changed
        current = {r: self.picker[r].current() for r in ROLES}
        for role in ROLES:
            pk = self.picker[role]
            pk.set_menu(self._menu_tree(role))
            name = current[role] if current[role] in self._entries else None
            if role == "ff" and name is None and "ff_sphere" in self._entries:
                name = "ff_sphere"
            pk.set_current(name)
        self._funcs = {r: None for r in ROLES}
        self._names = {r: None for r in ROLES}
        self.contrib = None
        self._last = None
        self._clear_held()
        self._apply_roles(ROLES)
        self._update_installation_label()

    def _resolve(self, name):
        return name if name in self._entries else headers.info(name).name

    def _set_combo(self, role, name):
        """Select ``name`` in the picker of ``role`` without updating the model; False if it is not in the menu."""
        return self.picker[role].set_current(name)

    def _initial(self, ff, sd, sq):
        self._update_installation_label()
        if "ff_sphere" in self._entries:
            self._set_combo("ff", "ff_sphere")
        for role, name in (("ff", ff), ("sd", sd), ("sq", sq)):
            if not name:
                continue
            try:
                full = self._resolve(name)
                if not self._set_combo(role, full):
                    raise KeyError(f"{full} is not a {ROLE_TITLES[role].lower()} function")
            except KeyError as exc:
                self._status(str(exc.args[0]), error=True)
        self._apply_roles(ROLES)
        if not self._entries:
            self._status("No SASfit functions found. Run 'pysasfit-studio status' to check the installation.", True)

    def _function(self, name):
        if name not in self._cache:
            self._cache[name] = function(name, errors="warn")
        return self._cache[name]

    def _on_combo(self, role):
        self._apply_roles((role,))

    def _apply_roles(self, changed):
        names = {r: self.picker[r].current() for r in ROLES}
        funcs = dict(self._funcs)
        try:
            for r in changed:
                funcs[r] = self._function(names[r]) if names[r] else None
            if funcs["ff"] is None:
                raise ValueError("no form factor selected")
        except Exception as exc:  # noqa: BLE001 -- show it, keep the previous model
            self._error(exc)
            for r in changed:
                if self._names[r] is not None or r != "ff":
                    self._set_combo(r, self._names[r])
            return
        self._funcs, self._names = funcs, names
        if "ff" in changed:
            self._fill_dist_combo()
        self._make_contribution()
        for r in changed:
            self._fill_group(r)
        self._update_enabled()
        self._update_labels()
        self._schedule()

    def _fill_dist_combo(self):
        f = self._funcs["ff"]
        names = f.parameter_names if f else []
        prev = self.dist_combo.currentText()
        self.dist_combo.blockSignals(True)
        self.dist_combo.clear()
        self.dist_combo.addItems(names)
        if names:
            idx = names.index(prev) if prev in names else next(
                (i for i, n in enumerate(names) if n.upper().startswith("R")), 0
            )
            self.dist_combo.setCurrentIndex(idx)
        self.dist_combo.blockSignals(False)

    def _update_enabled(self):
        sd, sq = self._funcs["sd"], self._funcs["sq"]
        self.dist_combo.setEnabled(sd is not None)
        self.mode_combo.setEnabled(sq is not None)
        self.approach_combo.setEnabled(sd is not None and sq is not None)

    def _make_contribution(self):
        ff, sd, sq = (self._funcs[r] for r in ROLES)
        if ff is None:
            self.contrib = None
            return
        old = self.contrib

        def keep(role):  # keep the user's values of functions that did not change
            f = self._funcs[role]
            of = old.function_for(role) if old is not None else None
            return old.vector(role).copy() if f is not None and of is not None and of.name == f.name else None

        self.contrib = ScatteringContribution(
            ff, keep("ff"), sd, keep("sd"), sq, keep("sq"),
            distributed=max(self.dist_combo.currentIndex(), 0),
            approach=self.approach_combo.currentData(),
            **self.integration,
        )  # fmt: skip

    def _on_structure_option(self):
        if self.contrib is not None:
            self._make_contribution()
            self._schedule()

    def _fill_group(self, role):
        lay = self._group_layout[role]
        _clear(lay)
        f, c = self._funcs[role], self.contrib
        self._group[role].setVisible(f is not None)
        if f is None or c is None:
            return
        self._group[role].setTitle(f"{ROLE_TITLES[role]}: {f.info.label}")
        vec = c.vector(role)
        if not f.info.parameters:
            lay.addWidget(QtWidgets.QLabel("(no documented parameters)"))
        for i, p in enumerate(f.info.parameters):
            if i >= len(vec):
                break
            row = ParameterRow(p.name, float(vec[i]), p.description, p.bounds)
            row.valueChanged.connect(lambda v, r=role, k=i: self._set_param(r, k, v))
            lay.addWidget(row)

    def _set_param(self, role, k, value):
        if self.contrib is not None:
            self.contrib.vector(role)[k] = value
            self._schedule()

    def _reset(self):
        self.contrib = None  # forget the values: defaults again
        self._make_contribution()
        for r in ROLES:
            self._fill_group(r)
        self.scale_row.set_value(1.0, emit=False)
        self.bg_row.set_value(0.0, emit=False)
        self._schedule()

    # ------------------------------------------------------------------ plot
    def _on_scales(self, *_):
        self.ax.set_xscale("log" if self.logx.isChecked() else "linear")
        if self.logy.isChecked():
            self.ax.set_yscale("log", nonpositive="mask")
        else:
            self.ax.set_yscale("linear")
        self._update_labels()
        self._schedule()

    def _update_labels(self, *_):
        self.ax.set_xlabel(f"q ({self.units.currentText()})")
        self.ax.set_ylabel("I(q)")
        parts = [self._funcs[r].info.label for r in ("sd", "ff", "sq") if self._funcs[r] is not None]
        if self._funcs["ff"] is not None:
            title = " · ".join(parts)
            self.setWindowTitle(f"pySASfit model explorer - {title}")
            self.ax.set_title(title, fontsize=10)
        self.canvas.draw_idle()

    def _schedule(self, *_):
        self._timer.start()

    def _q(self):
        try:
            lo, hi = float(self.qmin.text()), float(self.qmax.text())
        except ValueError:
            raise ValueError("Q min and Q max must be numbers") from None
        if not hi > lo:
            raise ValueError("Q max must be larger than Q min")
        n = self.npoints.value()
        if self.logx.isChecked():
            if lo <= 0:
                raise ValueError("Q min must be > 0 for a logarithmic q axis")
            return np.geomspace(lo, hi, n)
        return np.linspace(lo, hi, n)

    def _compute(self, q):
        c = self.contrib
        mode = self.mode_combo.currentData() if c.sq is not None else "P"
        if mode == "S":  # structure factor alone: no scale, no background
            return np.asarray(c.sq.evaluate_p(q, c.sq_p), dtype=float)
        if mode == "P" and c.sq is not None:
            sq, c.sq = c.sq, None
            try:
                particles = c.intensity(q)
            finally:
                c.sq = sq
        else:
            particles = c.intensity(q)
        return self.scale_row.value * np.asarray(particles, dtype=float) + self.bg_row.value

    def _redraw(self):
        if self.contrib is None:
            return
        t0 = time.perf_counter()
        try:
            q = self._q()
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                I = self._compute(q)
        except Exception as exc:  # noqa: BLE001 -- keep the last good curve on screen
            self._status(f"{type(exc).__name__}: {exc}", error=True)
            return
        if self.logy.isChecked() and not np.any(I > 0):  # matplotlib cannot draw a log axis without positive values
            self._status("no positive values to show on the logarithmic I axis (switch 'log I' off)", error=True)
            return
        self._last = (q, I)
        self.line.set_data(q, I)
        if self.autoscale.isChecked():
            self.ax.relim()
            self.ax.autoscale_view()
        self.canvas.draw_idle()
        msg = f"{q.size} points in {(time.perf_counter() - t0) * 1e3:.0f} ms"
        info = getattr(self.contrib, "last_integration", None)
        if info and self.contrib.sd is not None:
            msg += f" ({info['method']}, {info['neval']} evaluations of the form factor)"
        bad = int(np.count_nonzero(~np.isfinite(I)))
        if bad:
            msg += f" - {bad} non-finite values"
        if caught:
            msg += f" - warning: {caught[0].message}"
        self._status(msg, error=bool(bad))

    def _hold(self):
        if self._last is not None:
            (ln,) = self.ax.plot(*self._last, color="0.6", lw=1.2, ls="--")
            self._held.append(ln)
            self.canvas.draw_idle()

    def _clear_held(self):
        for ln in self._held:
            ln.remove()
        self._held.clear()
        self.canvas.draw_idle()

    # ----------------------------------------------------------- export, code
    def _param_dict(self, role):
        f, vec = self._funcs[role], self.contrib.vector(role)
        return {p.name: float(vec[i]) for i, p in enumerate(f.info.parameters) if i < len(vec)}

    def _python_snippet(self):
        c = self.contrib
        ff, sd, sq = (self._funcs[r] for r in ROLES)
        dist = ff.parameter_names[c.distributed] if c.distributed < len(ff.parameter_names) else c.distributed
        lines = ["from pysasfit.studio import ModelFunction, ScatteringContribution"]
        extra = user_precision()
        if extra:
            lines += ["from pysasfit.studio.library import configure_precision", "", f"configure_precision(**{extra!r})"]
        lines += [
            "",
            "c = ScatteringContribution(",
            f'    "{ff.name}", {self._param_dict("ff")!r},',
        ]
        if sd is not None:
            lines.append(f'    size_distribution="{sd.name}", sd_params={self._param_dict("sd")!r}, distributed={dist!r},')
        if sq is not None:
            lines.append(f'    structure_factor="{sq.name}", sq_params={self._param_dict("sq")!r},')
            if sd is not None:
                lines.append(f'    approach="{c.approach}",')
        if sd is not None:
            lines.append("    " + ", ".join(f"{k}={v!r}" for k, v in self.integration.items()) + ",")
        lines += [")", f"model = ModelFunction([c], scales=[{self.scale_row.value!r}], background={self.bg_row.value!r})"]
        lines.append("I = model(q)")
        return "\n".join(lines)

    def _copy_python(self):
        if self.contrib is not None:
            QtWidgets.QApplication.clipboard().setText(self._python_snippet())
            self._status("Python code copied to the clipboard")

    def _export(self):
        if self._last is None or self.contrib is None:
            return
        name = self._funcs["ff"].name
        fn, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Export curve", f"{name}.dat", "Data files (*.dat *.txt);;All files (*)"
        )
        if not fn:
            return
        q, I = self._last
        head = [self._python_snippet(), "", f"q ({self.units.currentText()})    I(q)"]
        try:
            np.savetxt(fn, np.column_stack([q, I]), header="\n".join(head), fmt="%.8g")
            self._status(f"saved {fn}")
        except OSError as exc:
            self._error(exc)


# --------------------------------------------------------------------------- entry points
def explore(ff="ff_sphere", sd=None, sq=None, qmin=0.01, qmax=1.0):
    """Open the explorer; blocks until it is closed unless a Qt event loop is already running."""
    app = QtWidgets.QApplication.instance()
    own = app is None
    if own:
        app = QtWidgets.QApplication(sys.argv[:1])
    win = ModelExplorer(ff, sd=sd, sq=sq, qmin=qmin, qmax=qmax)
    win.show()
    if own:
        app.exec()
    return win


def main(argv=None):
    ap = argparse.ArgumentParser(prog="pysasfit-explorer", description="Interactive plot of SASfit model functions.")
    ap.add_argument("model", nargs="?", default="ff_sphere", help="form factor, e.g. ff_sphere (see 'pysasfit-studio list')")
    ap.add_argument("--sq", help="structure factor, e.g. sq_hardsphere")
    ap.add_argument("--sd", help="size distribution, e.g. sd_LogNorm")
    ap.add_argument("--qmin", type=float, default=0.01)
    ap.add_argument("--qmax", type=float, default=1.0)
    ap.add_argument("--lib", help="folder of a local SASfit installation (this session only; see Settings in the GUI)")
    a = ap.parse_args(argv)
    if a.lib:
        paths.set_lib_dir(a.lib)
    explore(a.model, sd=a.sd, sq=a.sq, qmin=a.qmin, qmax=a.qmax)
    return 0


if __name__ == "__main__":
    sys.exit(main())
