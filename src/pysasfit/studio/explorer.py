"""Interactive model explorer for pysasfit.studio (in the spirit of the SasModels explorer app).

Plots a SASfit model on a log-log plot and lets you change every parameter with a slider or by typing
a value. The model is a **mixture**: a sum of contributions, each with its own scale factor, plus a
background

    I(q) = sum_i  scale_i * I_i(q)  +  background  +  A * q**(-alpha)

(the power-law term is off while A = 0; with a size distribution the amplitude N of the distribution is the scale of
the contribution).

and each contribution is a form factor, optionally integrated over a size distribution and multiplied by
a structure factor (:class:`pysasfit.studio.ScatteringContribution`, manual ch. 4). Contributions are
tabs: "Add" makes a new one, "Copy" duplicates the current one (for mixtures of the same shape in
different sizes), "Remove" drops it. Start with

    pysasfit-studio explore [ff_sphere] [--sq sq_hardsphere] [--sd sd_LogNorm] [--add ff_cylinder,sd_LogNorm]
    pysasfit-explorer ff_sphere --sq sq_hardsphere
    python -m pysasfit.studio.explorer

or from Python / IPython (``%gui qt``)::

    from pysasfit.studio.explorer import explore
    explore("ff_sphere", sq="sq_hardsphere", extra=[("ff_cylinder", "sd_LogNorm", None)])

The curve shown is exactly what :class:`~pysasfit.studio.ModelFunction` evaluates; the "Copy Python" button
puts the code to reproduce it on the clipboard. The "Fit" panel (View menu) fits the mixture to a data set.

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

from . import groups, headers, integration, paths, project  # noqa: E402
from .contribution_panel import APPROACHES, ROLE_TITLES, ROLES, ContributionPanel, is_dummy  # noqa: E402,F401
from .fit_panel import FitPanel  # noqa: E402
from .functions import _PYTHON_FUNCTIONS, function  # noqa: E402
from .integration_dialog import IntegrationDialog  # noqa: E402
from .library import (  # noqa: E402
    DEFAULT_PRECISION,
    SASfitLibraryNotFound,
    configure_precision,
    reset_library,
    user_precision,
)
from .param_row import ParameterRow, clear_layout, slider_range  # noqa: E402,F401
from .model import ModelFunction  # noqa: E402
from . import multidata  # noqa: E402
from ..resources import tool_icon  # noqa: E402
from ..core.docking import DockMixin  # noqa: E402

__all__ = ["ModelExplorer", "ParameterRow", "explore", "main"]

FF_TYPES = ("ff", "peak", "peaks")
MODES = [("P*S", "PS"), ("P  (no structure factor)", "P"), ("S  (structure factor only)", "S")]
Q_UNITS = ["1/nm", "1/Å", "1/m"]


def _entries():
    cat = dict(headers.catalogue())
    for name, impl in _PYTHON_FUNCTIONS.items():  # functions registered at run time
        cat[name] = impl[0]
    return cat


def _scrolled(widget):
    """``widget`` in a frameless scroll area: its panel then has no large minimum size, so the window
    fits a laptop screen; a panel made narrower than its content scrolls."""
    sc = QtWidgets.QScrollArea()
    sc.setWidget(widget)
    sc.setWidgetResizable(True)
    sc.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
    return sc


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
class ModelExplorer(DockMixin, QtWidgets.QMainWindow):
    """Panels (2026-11-07, as pySASred / pySASmorph): Model, I(q), Fit and Size distributions; each can be
    moved, stacked as tabs, detached into its own window (button in its title bar) and hidden (View menu)."""

    def __init__(self, ff="ff_sphere", sd=None, sq=None, qmin=0.01, qmax=1.0, extra=(), parent=None):
        super().__init__(parent)
        self.setWindowTitle("pySASfit model explorer")
        self.setWindowIcon(tool_icon("pySASstudio"))
        self.resize(1180, 920)
        self._entries = _entries()
        self._defs = groups.load_groups()
        self.integration = integration.load_settings()  # size-distribution integral (scipy / Gauss-Legendre)
        self.project_path = None  # file of the project last opened / saved
        self.datasets = []  # data sets shown with the model and fitted: dicts q, I, sigma, name, bg, artists
        self.current_ds = 0  # the data set whose background the rows show (and whose contributions are drawn dotted)
        self.overrides = {}  # data set -> {address: value}: values of parameters fitted separately per data set
        self._linked_rows = []  # parameter rows set by a link to a global parameter (not editable)
        self.show_sliders = False
        self._busy = False  # a fit is running: the parameters must not change
        self._res_visible = False
        self._part_lines = []  # one dotted line per contribution
        self._cache = {}
        self._last = None
        self._held = []
        self._timer = QtCore.QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.setInterval(40)  # debounce slider drags
        self._timer.timeout.connect(self._redraw)
        self._build_ui(qmin, qmax)
        self._initial(ff, sd, sq, extra)

    # ------------------------------------------------------------------ UI
    def _menu_tree(self, role):
        """Menu tree of the functions of a role: groups from the .def files, functions from the headers."""
        types = FF_TYPES if role == "ff" else (role,)
        return groups.build_menu(self._entries.values(), types, self._defs)

    def _build_ui(self, qmin, qmax):
        menu = self.menuBar().addMenu("&File")
        for text, key, slot in (
            ("&Open project…", "Ctrl+O", self._open_project),
            ("&Save project", "Ctrl+S", self._save_project),
            ("Save project &as…", "Ctrl+Shift+S", self._save_project_as),
        ):
            act = menu.addAction(text)
            act.setShortcut(key)
            act.triggered.connect(slot)
        from ..core import pycode
        pycode.install(self, self.python_code, status=lambda m: self._status(m))
        menu = self.menuBar().addMenu("&Settings")
        act = menu.addAction("SASfit &installation…")
        act.setShortcut("Ctrl+L")
        act.triggered.connect(self._choose_installation)
        act = menu.addAction("&Integration…")
        act.setShortcut("Ctrl+I")
        act.setToolTip("integration strategy of the SASfit library and of the size-distribution integral (scipy)")
        act.triggered.connect(self._choose_integration)

        # ---- left (a dock that can be detached): background, contributions (tabs)
        left = QtWidgets.QWidget()
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

        gbox = QtWidgets.QGroupBox("Mixture:  I(q) = Σ scale_i · I_i(q) + background + A · q^-alpha")
        gl = QtWidgets.QVBoxLayout(gbox)
        gl.setSpacing(2)
        self.bg_row = ParameterRow("background", 0.0, "constant background added to I(q)")
        self.bg_row.valueChanged.connect(self._schedule)
        self.bg_row.fitToggled.connect(lambda on: self.fit_panel.refresh_table())
        self.bg_row.boundsEdited.connect(lambda: self.fit_panel.refresh_table())
        gl.addWidget(self.bg_row)
        self.bgA_row = ParameterRow("bg.A", 0.0, "amplitude A of a power-law background A * q^(-alpha); 0 = off (e.g. Porod-like tail, alpha = 4)")
        self.bgalpha_row = ParameterRow("bg.alpha", 4.0, "exponent alpha of the power-law background A * q^(-alpha)")
        for r in (self.bgA_row, self.bgalpha_row):
            r.valueChanged.connect(self._schedule)
            r.fitToggled.connect(lambda on: self.fit_panel.refresh_table())
            r.boundsEdited.connect(lambda: self.fit_panel.refresh_table())
            gl.addWidget(r)
        show = QtWidgets.QHBoxLayout()
        self.mode_combo = QtWidgets.QComboBox()
        for text, data in MODES:
            self.mode_combo.addItem(text, data)
        self.mode_combo.setToolTip("P*S: the model; P: without the structure factors; S: structure factor of the current tab")
        self.mode_combo.currentIndexChanged.connect(self._schedule)
        self.show_parts = QtWidgets.QCheckBox("show contributions")
        self.show_parts.setChecked(True)
        self.show_parts.setToolTip("dotted curves: scale_i · I_i(q) of the individual contributions")
        self.show_parts.toggled.connect(self._schedule)
        show.addWidget(QtWidgets.QLabel("show"))
        show.addWidget(self.mode_combo)
        show.addWidget(self.show_parts)
        show.addStretch(1)
        gl.addLayout(show)
        dsrow = QtWidgets.QHBoxLayout()
        self.ds_label = QtWidgets.QLabel("background above is that of data set")
        self.ds_combo = QtWidgets.QComboBox()
        self.ds_combo.setToolTip(
            "several data sets: every data set has its own background (and its own values of parameters fitted "
            "separately per data set); the rows above show those of this data set, its contributions are drawn dotted"
        )
        self.ds_combo.currentIndexChanged.connect(self._ds_changed)
        dsrow.addWidget(self.ds_label)
        dsrow.addWidget(self.ds_combo, 1)
        self.ds_widgets = (self.ds_label, self.ds_combo)
        for w in self.ds_widgets:
            w.setVisible(False)
        gl.addLayout(dsrow)
        lv.addWidget(gbox)

        self.tabs = QtWidgets.QTabWidget()
        self.tabs.currentChanged.connect(self._schedule)
        bar = QtWidgets.QHBoxLayout()
        bar.addWidget(QtWidgets.QLabel("Contributions:"))
        for text, slot, tip in (
            ("Add", lambda: self.add_contribution(), "add a contribution to the mixture"),
            ("Copy", self._copy_contribution, "duplicate the current contribution (same functions, parameters, scale)"),
            ("Remove", self._remove_contribution, "remove the current contribution"),
        ):
            b = QtWidgets.QPushButton(text)
            b.setToolTip(tip)
            b.clicked.connect(slot)
            bar.addWidget(b)
        bar.addStretch(1)
        lv.addLayout(bar)
        lv.addWidget(self.tabs, 1)
        self._left = left                         # frozen while a fit runs
        # a small screen: the panel scrolls instead of a wide window
        self.model_dock = self._add_dock("model", "Model", left, QtCore.Qt.DockWidgetArea.LeftDockWidgetArea,
                                         scrollable=True)

        # ---- right: plot and controls
        right = QtWidgets.QWidget()
        rv = QtWidgets.QVBoxLayout(right)
        self.fig = Figure(figsize=(6, 4), layout="constrained")
        self._gs = self.fig.add_gridspec(2, 1, height_ratios=[1, 1e-3], hspace=0.05)  # residuals appear with data
        self.ax = self.fig.add_subplot(self._gs[0])
        self.ax_res = self.fig.add_subplot(self._gs[1], sharex=self.ax)
        self.canvas = FigureCanvasQTAgg(self.fig)
        rv.addWidget(NavigationToolbar2QT(self.canvas, self))
        rv.addWidget(self.canvas, 1)
        (self.line,) = self.ax.plot([qmin, qmax], [1.0, 1.0], lw=1.8)  # placeholder: a log axis needs data to draw
        (self.bg_line,) = self.ax.plot([], [], ls="--", lw=1.2, color="0.45", zorder=2, label="background")
        self.bg_line.set_visible(False)
        self.ax.grid(True, which="both", alpha=0.25)
        (self.res_line,) = self.ax_res.plot([], [], "o", ms=2.5, color="C0")
        self.ax_res.axhline(0.0, color="0.5", lw=0.8)
        self.ax_res.set_ylabel("Δ/σ")
        self.ax_res.grid(True, which="both", alpha=0.25)
        self.ax_res.set_visible(False)

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
        #AUTO-CALCULATE AND AN EXPLICIT CALCULATE BUTTON. On by default, so
        #nothing changes for models that are cheap to evaluate; turn it off
        #for one that takes seconds and the plot updates only when asked.
        #The button's label gains an asterisk while a change is outstanding,
        #so a stale curve is never passed off as current.
        self._autocalc = True
        self._pending = False
        self.autocalc = QtWidgets.QCheckBox("auto-calculate")
        self.autocalc.setChecked(True)
        self.autocalc.toggled.connect(self._on_autocalc)
        self.calc_btn = QtWidgets.QPushButton("Calculate")
        self.calc_btn.clicked.connect(self._force_calc)
        from ..core.flowlayout import flow_bar
        rv.addWidget(flow_bar(                    # wraps to two rows in a narrow window
            "Q min:", self.qmin, self.units, "Q max:", self.qmax,
            "points:", self.npoints, self.logx, self.logy, self.autoscale,
            self.autocalc, self.calc_btn,
        ))  # fmt: skip

        button_list = []
        for text, slot, tip in (
            ("Hold", self._hold, "keep the current curve as a grey reference"),
            ("Clear held", self._clear_held, "remove the reference curves"),
            ("Reset parameters", self._reset, "current contribution: default parameters of its functions, scale 1; background 0"),
            ("Export…", self._export, "save q, I(q) and the parameters as a text file"),
            ("Copy Python", self._copy_python, "copy code that reproduces the curve"),
        ):
            b = QtWidgets.QPushButton(text)
            b.setToolTip(tip)
            b.clicked.connect(slot)
            button_list.append(b)
        rv.addWidget(flow_bar(*button_list))
        self.status = QtWidgets.QLabel("")
        self.status.setWordWrap(True)
        rv.addWidget(self.status)
        rv.setContentsMargins(2, 2, 2, 2)
        self.plot_dock = self._add_dock("plot", "I(q)", right, QtCore.Qt.DockWidgetArea.RightDockWidgetArea)
        self._on_scales()

        self.fit_panel = FitPanel(self)
        self.fit_dock = self._add_dock("fit", "Fit", self.fit_panel, QtCore.Qt.DockWidgetArea.BottomDockWidgetArea,
                                       scrollable=True)
        self._build_distribution_dock()
        H, V = QtCore.Qt.Orientation.Horizontal, QtCore.Qt.Orientation.Vertical
        self.splitDockWidget(self.model_dock, self.plot_dock, H)
        self.splitDockWidget(self.plot_dock, self.fit_dock, V)      # first: the fit panel spans plot and distributions
        self.splitDockWidget(self.plot_dock, self.dist_dock, H)
        view = self.menuBar().addMenu("&View")
        act = view.addAction("&Sliders")
        self.sliders_action = act
        act.setCheckable(True)
        act.setToolTip("show a slider next to every parameter field")
        act.toggled.connect(self._set_sliders)
        view.addSeparator()
        self.extra_view_actions = [act]           # also in the View menu of the pySASfit launcher
        self._finish_docks()                      # panel entries, border, arrangement in the View menu
        self.hide_stray_tab_bars()

    # ------------------------------------------------- size distributions
    def _build_distribution_dock(self):
        """Dock with a plot of the size distribution N(x) of every contribution that has one."""
        w = QtWidgets.QWidget()
        v = QtWidgets.QVBoxLayout(w)
        v.setContentsMargins(2, 2, 2, 2)
        self.dist_fig = Figure(figsize=(4, 3), layout="constrained")
        self.dist_ax = self.dist_fig.add_subplot(111)
        self.dist_canvas = FigureCanvasQTAgg(self.dist_fig)
        v.addWidget(NavigationToolbar2QT(self.dist_canvas, w))
        v.addWidget(self.dist_canvas, 1)
        row = QtWidgets.QHBoxLayout()
        self.dist_logx = QtWidgets.QCheckBox("log x")
        self.dist_logx.toggled.connect(self._schedule)
        self.dist_xpow = QtWidgets.QDoubleSpinBox()
        self.dist_xpow.setRange(-6.0, 12.0)
        self.dist_xpow.setDecimals(2)
        self.dist_xpow.setSingleStep(1.0)
        self.dist_xpow.setValue(3.0)
        self.dist_xpow.setToolTip("plot N(x) * x^p: p = 0 number distribution, 3 volume-like, 6 what the intensity sees at low q")
        self.dist_xpow.valueChanged.connect(self._schedule)
        row.addWidget(self.dist_logx)
        row.addWidget(QtWidgets.QLabel("N(x) · x^p, p ="))
        row.addWidget(self.dist_xpow)
        for p, text in ((0, "p=0"), (3, "p=3"), (6, "p=6")):
            b = QtWidgets.QPushButton(text)
            b.setFixedWidth(46)
            b.clicked.connect(lambda _c=False, p=p: self.dist_xpow.setValue(p))
            row.addWidget(b)
        row.addStretch(1)
        v.addLayout(row)
        self.dist_dock = self._add_dock("dist", "Size distributions", w, QtCore.Qt.DockWidgetArea.RightDockWidgetArea)
        self.dist_dock.visibilityChanged.connect(lambda _v: self._schedule())

    @staticmethod
    def _sd_range(c):
        """x range for plotting the size distribution of contribution ``c``."""
        try:
            lo, hi = c.support()
        except TypeError:
            lo, hi = c.support(0.0)
        if not (np.isfinite(lo) and np.isfinite(hi) and hi > lo):
            raise ValueError("no usable size range")
        return max(lo / 2.0, hi * 1e-4), hi * 1.5

    def _update_distributions(self):
        if not self.dist_dock.isVisible():
            return
        ax = self.dist_ax
        ax.clear()
        k, lab = float(self.dist_xpow.value()), ""
        n = 0
        for i, p in enumerate(self.models()):
            c = p.contrib
            if c.sd is None:
                continue
            try:
                lo, hi = self._sd_range(c)
                x = np.geomspace(lo, hi, 400) if self.dist_logx.isChecked() else np.linspace(lo, hi, 400)
                y = np.asarray(c.sd.evaluate_p(x, c.sd_p), dtype=float) * x**k
            except Exception:  # noqa: BLE001 -- a distribution that cannot be drawn is left out
                continue
            ax.plot(x, y, color=f"C{i + 1}", label=f"{i + 1}: {c.sd.name}")
            try:
                lab = c.ff.parameter_names[c.distributed]
            except Exception:  # noqa: BLE001
                lab = lab or "x"
            n += 1
        if n:
            ax.set_xlabel(lab or "x")
            ax.set_ylabel("N(x)" if not k else f"N(x) · x^{k:g}")
            ax.grid(True, which="both", alpha=0.25)
            if self.dist_logx.isChecked():
                ax.set_xscale("log")
            ax.legend(fontsize=8)
        else:
            ax.text(0.5, 0.5, "no contribution has a size distribution", ha="center", va="center", transform=ax.transAxes)
            ax.set_axis_off()
        self.dist_canvas.draw_idle()

    def _update_background(self, q):
        """Dashed grey line: constant background + A q^-alpha (only when there is one; of the current data set)."""
        bg = self.bg_row.value + self._power_term(q)
        show = self.mode_combo.currentData() != "S" and bool(np.any(np.asarray(bg) != 0))
        if show:
            self.bg_line.set_data(q, np.broadcast_to(bg, np.shape(q)))
        self.bg_line.set_visible(show)

    # -------------------------------------------------------------- project
    def _open_project(self):
        fn, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Open project", self.project_path or "", f"pySASfit studio project (*{project.SUFFIX});;All files (*)"
        )
        if fn:
            self.open_project(fn)

    def open_project(self, path):
        """Replace the session by a saved project."""
        try:
            project.load(self, path)
        except Exception as exc:  # noqa: BLE001 -- unreadable file, functions missing in this installation ...
            self._error(exc)
            return False
        self.project_path = str(path)
        self.setWindowTitle(f"pySASfit model explorer - {os.path.basename(path)}")
        self._status(f"project {path} loaded")
        return True

    def save_project(self, path):
        try:
            self.project_path = project.save(self, path)
        except OSError as exc:
            self._error(exc)
            return False
        self.setWindowTitle(f"pySASfit model explorer - {os.path.basename(self.project_path)}")
        self._status(f"project saved to {self.project_path}")
        return True

    def _save_project(self):
        if self.project_path:
            self.save_project(self.project_path)
        else:
            self._save_project_as()

    def _save_project_as(self):
        fn, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Save project", self.project_path or f"project{project.SUFFIX}", f"pySASfit studio project (*{project.SUFFIX})"
        )
        if fn:
            self.save_project(fn)

    # --------------------------------------------------------------- status
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
        for p in self.panels():
            p.apply_integration(self.integration)
        self._schedule()

    def _refresh_catalogue(self):
        """Re-read the function lists after the library folder changed and rebuild the current model."""
        self._entries = _entries()
        self._defs = groups.load_groups()  # the .def files live next to the plugins and may have changed
        for p in self.panels():
            p.refresh_catalogue()
        self._last = None
        self._clear_held()
        self._update_installation_label()

    def _resolve(self, name):
        return name if name in self._entries else headers.info(name).name

    def _function(self, name):
        if name not in self._cache:
            self._cache[name] = function(name, errors="warn")
        return self._cache[name]

    # ------------------------------------------------------- contributions
    def panels(self):
        return [self.tabs.widget(i) for i in range(self.tabs.count())]

    def models(self):
        """The contributions that can be evaluated, in tab order (index i = contribution i of the model)."""
        return [p for p in self.panels() if p.contrib is not None]

    def current_panel(self):
        return self.tabs.currentWidget()

    def has_model(self):
        return bool(self.models())

    def _initial(self, ff, sd, sq, extra=()):
        self._update_installation_label()
        self.add_contribution(ff, sd, sq)
        for item in extra:
            self.add_contribution(*item)
        if not self._entries:
            self._status("No SASfit functions found. Run 'pysasfit-studio status' to check the installation.", True)

    def add_contribution(self, ff="ff_sphere", sd=None, sq=None, copy_from=None):
        """Add a contribution (a tab) to the mixture; ``copy_from``: another panel to take the values from."""
        panel = ContributionPanel(self, ff, sd, sq)
        panel.changed.connect(self._schedule)
        panel.structure_changed.connect(self._structure_changed)
        panel.fit_toggled.connect(lambda role, name, on, p=panel: self._fit_toggled(p, role, name, on))
        if copy_from is not None:
            panel.copy_from(copy_from)
        self.tabs.addTab(panel, "")
        self.tabs.setCurrentWidget(panel)
        self._structure_changed()
        return panel

    def _copy_contribution(self):
        p = self.current_panel()
        if p is not None and p.contrib is not None:
            names = p.names
            self.add_contribution(names["ff"], names["sd"], names["sq"], copy_from=p)

    def _remove_contribution(self):
        if self.tabs.count() <= 1:
            self._status("the mixture needs at least one contribution", error=True)
            return
        i = self.tabs.currentIndex()
        w = self.tabs.widget(i)
        self.tabs.removeTab(i)
        w.deleteLater()
        self._structure_changed()

    def _structure_changed(self):
        """Contributions or their functions changed: tab titles, plot title, fit table."""
        multi = len(self.datasets) > 1
        for i, p in enumerate(self.panels()):
            ds = p.datasets_text()
            self.tabs.setTabText(i, f"{i + 1}: {p.label()}" + (f"  [{ds}]" if multi and ds != "all" else ""))
            for w in (p.ds_label, p.ds_edit):
                w.setVisible(multi or ds != "all")
        self.mode_combo.setEnabled(any(p.contrib is not None and p.contrib.sq is not None for p in self.panels()))
        self._update_labels()
        self.fit_panel.refresh_table()
        self._schedule()

    def _reset(self):
        p = self.current_panel()
        if p is not None:
            p.reset()
        self.bg_row.set_value(0.0, emit=False)
        self.bgA_row.set_value(0.0, emit=False)
        self.bgalpha_row.set_value(4.0, emit=False)
        self._schedule()

    # ------------------------------------------------------------------ plot
    def showEvent(self, event):
        super().showEvent(event)
        if not getattr(self, "_sized", False):
            self._sized = True
            QtCore.QTimer.singleShot(0, self._default_dock_sizes)

    # ------------------------------------------------------------------ panel layout of the last session
    @staticmethod
    def _layout_path():
        from ..core import settings as prefs
        return os.path.join(os.path.dirname(prefs.settings_path()), "studio_layout.json")

    def save_session(self):
        """Keep the panel arrangement (moved, stacked, detached, hidden panels) for the next start."""
        try:
            import json
            p = self._layout_path()
            os.makedirs(os.path.dirname(p), exist_ok=True)
            with open(p, "w", encoding="utf-8") as fh:
                json.dump({"dock_state": self.dock_state()}, fh)
        except (OSError, TypeError, ValueError):
            pass

    def restore_session(self):
        """Panel arrangement of the last session (the model itself is kept in project files)."""
        try:
            import json
            with open(self._layout_path(), encoding="utf-8") as fh:
                st = json.load(fh).get("dock_state")
        except (OSError, ValueError, AttributeError):
            return False
        self._sized = True                    # no default sizes over the restored arrangement
        return self.restore_dock_state(st)

    def forget_session(self):
        try:
            os.remove(self._layout_path())
        except OSError:
            pass

    def shutdown(self):
        """Called by the pySASfit launcher when it closes."""
        self.save_session()

    def closeEvent(self, event):
        self.save_session()
        super().closeEvent(event)

    def _default_dock_sizes(self):
        """Start sizes in proportion to the window, so that the I(q) plot keeps a useful size on a laptop
        (1600 × 1000: model 560, plot ≈ 650, size distributions 390; the fit panel a third of the height)."""
        W, H = max(self.width(), 900), max(self.height(), 600)
        try:
            self.resizeDocks([self.model_dock, self.dist_dock],
                             [int(min(max(0.35 * W, 420), 640)), int(min(max(0.24 * W, 280), 460))],
                             QtCore.Qt.Orientation.Horizontal)
            self.resizeDocks([self.fit_dock], [int(min(max(0.32 * H, 220), 420))], QtCore.Qt.Orientation.Vertical)
        except Exception:  # noqa: BLE001 -- a layout nicety
            pass

    def _on_scales(self, *_):
        self.ax.set_xscale("log" if self.logx.isChecked() else "linear")
        if self.logy.isChecked():
            self.ax.set_yscale("log", nonpositive="mask")
        else:
            self.ax.set_yscale("linear")
        self._update_labels()
        self._schedule()

    def _update_labels(self, *_):
        label = f"q ({self.units.currentText()})"
        shown, hidden = (self.ax_res, self.ax) if self._res_visible else (self.ax, self.ax_res)
        shown.set_xlabel(label)
        hidden.set_xlabel("")
        self.ax.set_ylabel("I(q)")
        titles = []
        for p in self.panels():
            parts = [p.funcs[r].info.label for r in ("sd", "ff", "sq") if p.funcs[r] is not None]
            if parts:
                titles.append(" · ".join(parts))
        if titles:
            title = "  +  ".join(titles)
            self.setWindowTitle(f"pySASfit model explorer - {title}")
            self.ax.set_title(title, fontsize=10)
        self.canvas.draw_idle()

    def _schedule(self, *_):
        #AUTOMATIC RECALCULATION IS OPTIONAL. Every parameter change funnels
        #through here -- sliders, spin boxes, the Q range, the tab selection,
        #contribution panels -- and starts a 40 ms debounce. That is right
        #for a sphere and wrong for anything expensive: a model taking
        #seconds per curve makes the interface unusable, since each keystroke
        #queues another full calculation.
        #
        #With auto-calculate off the request is remembered rather than
        #dropped, so pressing Calculate afterwards redraws once with
        #everything that changed in the meantime.
        if not getattr(self, "_autocalc", True):
            self._pending = True
            if hasattr(self, "calc_btn"):
                #Say so, rather than leaving the plot silently stale. A
                #curve that no longer matches the controls is worse than a
                #slow one.
                self.calc_btn.setText("Calculate *")
            return
        if not self._busy:
            self._timer.start()

    def _on_autocalc(self, on):
        """Auto-calculate toggled: catch up immediately when switched on."""
        self._autocalc = bool(on)
        if self._autocalc:
            self._force_calc()

    def _force_calc(self, *_):
        """Calculate now, whatever the auto setting."""
        self._pending = False
        if hasattr(self, "calc_btn"):
            self.calc_btn.setText("Calculate")
        if not self._busy:
            self._redraw()

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

    def _compute(self, q, k=None):
        """``(total, parts)``: I(q) of the model of data set ``k`` (None: the whole mixture) and scale_i * I_i(q) of every
        contribution (None for a contribution not assigned to data set ``k``)."""
        if k is not None:
            model = self.model_for(k)
            with multidata.overridden(model, self.overrides_for(k)):
                return self._compute_model(q, model)
        mode = self.mode_combo.currentData()
        if mode == "S":  # structure factor of the current contribution alone: no scale, no background
            p = self.current_panel()
            c = p.contrib if p is not None else None
            if c is None or c.sq is None:
                raise ValueError("the current contribution has no structure factor")
            return np.asarray(c.sq.evaluate_p(q, c.sq_p), dtype=float), []
        parts = []
        for p in self.models():
            c = p.contrib
            if mode == "P" and c.sq is not None:
                sq, c.sq = c.sq, None
                try:
                    particles = c.intensity(q)
                finally:
                    c.sq = sq
            else:
                particles = c.intensity(q)
            parts.append(p.scale_row.value * np.asarray(particles, dtype=float))
        return sum(parts) + self.bg_row.value + self._power_term(q), parts

    def _compute_model(self, q, model):
        mode = self.mode_combo.currentData()
        if mode == "S":
            return self._compute(q)
        parts = []
        for i, (sc, c) in enumerate(zip(model.scales, model.contributions)):
            if not model.is_active(i):
                parts.append(None)
                continue
            if mode == "P" and c.sq is not None:
                sq, c.sq = c.sq, None
                try:
                    particles = c.intensity(q)
                finally:
                    c.sq = sq
            else:
                particles = c.intensity(q)
            parts.append(sc * np.asarray(particles, dtype=float))
        total = np.zeros(np.shape(q)) + model._background(q)
        for pt in parts:
            if pt is not None:
                total = total + pt
        return total, parts

    def _power_term(self, q):
        A = self.bgA_row.value
        return A * np.asarray(q, dtype=float) ** (-self.bgalpha_row.value) if A else 0.0

    def _redraw(self):
        if not self.has_model() or self._busy:
            return
        t0 = time.perf_counter()
        Id, parts = None, []  # the model at the data points (per data set); the contributions at the curve points
        self._apply_links_to_rows()
        curves = {}
        try:
            q = self._q()
            full = self._data_q() is not None
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                if not self.datasets:
                    I, parts = self._compute(q)
                else:  # per data set: one call for the curve and its data points
                    Id = []
                    for k, d in enumerate(self.datasets):
                        qq = np.concatenate([q, d["q"]]) if full else q
                        both, pk = self._compute(qq, k)
                        curves[k] = both[: q.size]
                        Id.append(both[q.size:] if full else None)
                        if k == self.current_ds:
                            parts = [None if p is None else p[: q.size] for p in pk]
                    I = curves[self.current_ds]
        except Exception as exc:  # noqa: BLE001 -- keep the last good curve on screen
            self._status(f"{type(exc).__name__}: {exc}", error=True)
            return
        if self.logy.isChecked() and not np.any(I > 0):  # matplotlib cannot draw a log axis without positive values
            self._status("no positive values to show on the logarithmic I axis (switch 'log I' off)", error=True)
            return
        self._last = (q, I)
        multi = len(self.datasets) > 1
        self.line.set_data(q, I)
        self.line.set_visible(not multi)
        for k, d in enumerate(self.datasets):
            if multi:
                if d.get("line") is None:
                    (d["line"],) = self.ax.plot([], [], lw=1.6, color=self._colour(k), zorder=3)
                d["line"].set_data(q, curves[k])
                d["line"].set_visible(True)
            elif d.get("line") is not None:
                d["line"].set_visible(False)
        if multi:
            self.ax.legend(fontsize=8, loc="best")
        elif self.ax.get_legend() is not None:
            self.ax.get_legend().remove()
        self._update_parts(q, parts)
        self._update_background(q)
        self._update_distributions()
        fit_note = self._update_residuals(Id)
        self.fit_panel.refresh_values()
        if self.autoscale.isChecked():
            self.ax.relim(visible_only=True)
            self.ax.autoscale_view()
        self.canvas.draw_idle()
        msg = f"{q.size} points in {(time.perf_counter() - t0) * 1e3:.0f} ms"
        infos = [p.contrib.last_integration for p in self.models() if p.contrib.sd is not None and p.contrib.last_integration]
        if infos:
            msg += f" ({infos[0]['method']}, {sum(i['neval'] for i in infos)} evaluations of the form factor)"
        if fit_note:
            msg += f" - {fit_note}"
        bad = int(np.count_nonzero(~np.isfinite(I)))
        if bad:
            msg += f" - {bad} non-finite values"
        if caught:
            msg += f" - warning: {caught[0].message}"
        self._status(msg, error=bool(bad))

    def _update_parts(self, q, parts):
        """Dotted curve of every contribution (only for a mixture; with several data sets those of the current one)."""
        show = self.show_parts.isChecked() and sum(p is not None for p in parts) > 1
        while len(self._part_lines) < len(parts):
            k = len(self._part_lines)
            (ln,) = self.ax.plot([], [], ls=":", lw=1.2, color=f"C{k + 1}", zorder=2)
            self._part_lines.append(ln)
        for k, ln in enumerate(self._part_lines):
            if show and k < len(parts) and parts[k] is not None:
                ln.set_data(q, parts[k])
                ln.set_visible(True)
            else:
                ln.set_visible(False)

    # ------------------------------------------------------------ data, fit
    @property
    def data(self):
        """The first data set (dict q, I, sigma, name) or None."""
        return self.datasets[0] if self.datasets else None

    def set_data(self, q, I, sigma, name="data"):
        """Show one data set with the model (replaces the data sets; see :meth:`add_dataset` for several)."""
        self.clear_data()
        return self.add_dataset(q, I, sigma, name)

    def add_dataset(self, q, I, sigma, name="data", bg=None):
        """Add a data set (shown with its model curve and fitted together with the others); returns its index."""
        q, I = np.asarray(q, float), np.asarray(I, float)
        sigma = np.full(q.shape, np.nan) if sigma is None else np.asarray(sigma, float)
        ok = np.isfinite(q) & np.isfinite(I) & (q > 0)
        if not ok.any():
            raise ValueError("no data points with q > 0")
        q, I, sigma = q[ok], I[ok], sigma[ok]
        k = len(self.datasets)
        if k == 0:
            first = {"background": self.bg_row.value, "bg_A": self.bgA_row.value, "bg_alpha": self.bgalpha_row.value}
        else:
            self._store_bg()
            first = dict(self.datasets[self.current_ds]["bg"])
        if bg is not None and k == self.current_ds:
            self.bg_row.set_value(bg["background"], emit=False)
            self.bgA_row.set_value(bg["bg_A"], emit=False)
            self.bgalpha_row.set_value(bg["bg_alpha"], emit=False)
        d = {"q": q, "I": I, "sigma": sigma, "name": os.path.basename(str(name)), "bg": dict(bg or first), "path": str(name),
             "artist": None, "line": None, "res": None}  # fmt: skip
        self.datasets.append(d)
        for j in range(len(self.datasets)):  # colours and legend labels depend on the number of data sets
            self._draw_dataset_points(j)
        all_q = np.concatenate([x["q"] for x in self.datasets])
        self.qmin.setText(f"{all_q.min() * 0.9:.3g}")  # the curve covers the data
        self.qmax.setText(f"{all_q.max() * 1.1:.3g}")
        self._refresh_ds_combo()
        self.fit_panel.refresh_datasets()
        self._structure_changed()
        return k

    def _colour(self, k):
        return "0.25" if len(self.datasets) == 1 else f"C{k % 10}"

    def _draw_dataset_points(self, k):
        d = self.datasets[k]
        if d["artist"] is not None:
            d["artist"].remove()
        err = np.where(np.isfinite(d["sigma"]) & (d["sigma"] > 0), d["sigma"], 0.0)
        d["artist"] = self.ax.errorbar(
            d["q"], d["I"], yerr=err, fmt="o", ms=3, color=self._colour(k), ecolor="0.7", elinewidth=0.8, zorder=1,
            label=d["name"] if len(self.datasets) > 1 else None,
        )
        self.line.set_zorder(3)

    def remove_dataset(self, k):
        """Remove data set ``k`` (index from 0)."""
        if not 0 <= k < len(self.datasets):
            return
        self._store_bg()
        d = self.datasets.pop(k)
        for key in ("artist", "line", "res"):
            if d.get(key) is not None:
                d[key].remove()
        self.overrides = {(j if j < k else j - 1): v for j, v in self.overrides.items() if j != k}
        for j in range(len(self.datasets)):  # colours follow the position
            self._draw_dataset_points(j)
            for key in ("line", "res"):
                if self.datasets[j].get(key) is not None:
                    self.datasets[j][key].remove()
                    self.datasets[j][key] = None
        self.current_ds = min(self.current_ds, max(len(self.datasets) - 1, 0))
        if self.datasets:
            self._load_bg(self.current_ds)
        else:
            self._show_residuals(False)
        self._refresh_ds_combo()
        self.fit_panel.refresh_datasets()
        self.canvas.draw_idle()
        self._structure_changed()

    def clear_data(self):
        while self.datasets:
            self.remove_dataset(len(self.datasets) - 1)
        self.overrides = {}
        self._show_residuals(False)
        self._refresh_ds_combo()
        self.canvas.draw_idle()
        self._schedule()

    # background of the data sets: the rows show that of the current data set
    def _store_bg(self):
        if self.datasets and 0 <= self.current_ds < len(self.datasets):
            self.datasets[self.current_ds]["bg"] = {
                "background": self.bg_row.value, "bg_A": self.bgA_row.value, "bg_alpha": self.bgalpha_row.value,
            }  # fmt: skip

    def _load_bg(self, k):
        bg = self.datasets[k]["bg"]
        self.bg_row.set_value(bg["background"], emit=False)
        self.bgA_row.set_value(bg["bg_A"], emit=False)
        self.bgalpha_row.set_value(bg["bg_alpha"], emit=False)

    def background_of(self, k=None):
        if k is None or not self.datasets or k == self.current_ds:
            return {"background": self.bg_row.value, "bg_A": self.bgA_row.value, "bg_alpha": self.bgalpha_row.value}
        return dict(self.datasets[k]["bg"])

    def _refresh_ds_combo(self):
        self.ds_combo.blockSignals(True)
        self.ds_combo.clear()
        for k, d in enumerate(self.datasets):
            self.ds_combo.addItem(f"{k + 1}: {d['name']}", k)
        self.ds_combo.setCurrentIndex(self.current_ds if self.datasets else -1)
        self.ds_combo.blockSignals(False)
        for w in self.ds_widgets:
            w.setVisible(len(self.datasets) > 1)

    def _ds_changed(self, i):
        if i < 0 or i == self.current_ds or i >= len(self.datasets):
            return
        self._store_bg()
        self.current_ds = i
        self._load_bg(i)
        self._schedule()

    # model of a data set
    def model_for(self, k=None):
        """:class:`~pysasfit.studio.ModelFunction` of data set ``k`` (contributions assigned to it, its background);
        ``None``: the whole mixture with the background rows. The contributions are the live objects."""
        ms = self.models()
        n = len(self.datasets)
        active = None if k is None else [p.applies_to(k, n) for p in ms]
        bg = self.background_of(k)
        return ModelFunction(
            [p.contrib for p in ms], scales=[p.scale_row.value for p in ms], background=bg["background"],
            bg_A=bg["bg_A"], bg_alpha=bg["bg_alpha"], active=active,
        )  # fmt: skip

    def overrides_for(self, k):
        """``{address: value}`` that differ in data set ``k`` from the rows: parameters fitted separately per data
        set and links to global parameters that hold only in some data sets."""
        out = dict(self.overrides.get(k, {})) if k is not None else {}
        if k is None:
            return out
        try:
            links, glob = self.fit_panel.links(), self.fit_panel.global_values()
        except ValueError:
            return out
        amap = self.label_map()
        for ln in links:
            if ln.datasets is not None and k in ln.datasets and ln.target in amap and ln.names <= set(glob):
                try:
                    out[amap[ln.target]] = ln.value(glob)
                except (ValueError, ZeroDivisionError, OverflowError):
                    pass
        return out

    def label_map(self):
        """parameter label -> fit address (also ``c1.ff.R`` with a single contribution, and the address itself)."""
        out = {}
        single = len(self.models()) == 1
        for address, label, _lo, _hi in self.parameter_entries():
            out[label] = address
            out[address] = address
            if single and not label.startswith(("background", "bg.")):
                out["c1." + label] = address
        return out

    def _apply_links_to_rows(self):
        """Links to global parameters that hold in all data sets: the row takes factor * P and cannot be edited."""
        released = list(self._linked_rows)
        for r in released:
            r.set_editable(True)
        self._linked_rows = []
        if released:  # rows that a panel keeps disabled for its own reasons (distributed parameter, …)
            for p in self.panels():
                if p.contrib is not None:
                    p._update_enabled()
        try:
            links, glob = self.fit_panel.links(), self.fit_panel.global_values()
        except ValueError as exc:
            self.fit_panel.link_message(str(exc), error=True)
            return
        amap = self.label_map()
        bad = []
        model = self.model_for(None)
        from .fitting import locate

        for ln in links:
            if ln.target not in amap:
                bad.append(f"unknown parameter '{ln.target}'")
                continue
            missing = ln.names - set(glob)
            if missing:
                bad.append(f"unknown global parameter {', '.join(sorted(missing))}")
                continue
            if ln.datasets is not None:
                continue
            address = amap[ln.target]
            row = self._row(address)
            try:
                v = ln.value(glob)
            except (ValueError, ZeroDivisionError, OverflowError) as exc:
                bad.append(f"{ln.text}: {exc}")
                continue
            if row is None:
                continue
            if not (address.startswith("scale.") or address in ("background", "bg.A", "bg.alpha")):
                locate(model, address)[1](v)          # the parameter vector of the contribution
            row.set_value(v, emit=False)
            row.set_editable(False, f"{ln.target} {ln.describe()} (global parameter, Fit panel)")
            self._linked_rows.append(row)
        self.fit_panel.link_message("; ".join(bad), error=bool(bad))

    def _data_q(self):
        """q values of the data if the curve shown is the full model (so residuals make sense), else None."""
        if self.data is None or not self.has_model():
            return None
        if self.mode_combo.currentData() != "PS" and any(p.contrib.sq is not None for p in self.models()):
            return None
        return self.data["q"]

    def _show_residuals(self, show):
        if show == self._res_visible:
            return
        self._res_visible = show
        self.ax_res.set_visible(show)
        self._gs.set_height_ratios([4, 1] if show else [1, 1e-3])
        self.ax.tick_params(labelbottom=not show)
        self._update_labels()

    def _update_residuals(self, Id):
        """Residuals (model - data) / sigma below the plot; ``Id``: model at the data points of every data set
        (list, None where the curve is not the full model). Returns the chi² note for the status line."""
        if not self.datasets:
            self._show_residuals(False)
            return ""
        if Id is None or all(x is None for x in Id):
            self._show_residuals(False)
            return "residuals: the curve shown is not the full model"
        try:
            rng = self.fit_panel.q_range()
        except ValueError:
            rng = None
        tot, n = 0.0, 0
        self.res_line.set_data([], [])
        for k, (d, Ik) in enumerate(zip(self.datasets, Id)):
            if Ik is None:
                continue
            with np.errstate(invalid="ignore", divide="ignore"):
                r = (Ik - d["I"]) / d["sigma"]
            r = np.where(np.isfinite(r), r, np.nan)
            if d.get("res") is None:
                (d["res"],) = self.ax_res.plot([], [], "o", ms=2.5, color=self._colour(k) if len(self.datasets) > 1 else "C0")
            d["res"].set_data(d["q"], r)
            inside = np.isfinite(r) & ((d["q"] >= rng[0]) & (d["q"] <= rng[1]) if rng else True)
            tot += float(np.sum(r[inside] ** 2))
            n += int(np.count_nonzero(inside))
        self._show_residuals(True)
        self.ax_res.relim()
        self.ax_res.autoscale_view(scalex=False)
        if not n:
            return ""
        return f"chi²/n = {tot / n:.4g} ({n} points" + (f", {len(self.datasets)} data sets)" if len(self.datasets) > 1 else ")")

    def set_busy(self, busy):
        """A fit is running (``True``): freeze the model controls; ``False``: redraw."""
        self._busy = bool(busy)
        self._left.setEnabled(not busy)
        self.menuBar().setEnabled(not busy)
        if not busy:
            self._schedule()

    def _set_sliders(self, on):
        self.show_sliders = bool(on)
        for r in (self.bg_row, self.bgA_row, self.bgalpha_row):
            r.set_sliders(on)
        for p in self.panels():
            p.set_sliders(on)

    def _fit_toggled(self, panel, role, name, on):
        """A fit box or bound next to a parameter changed: the fit panel shows the selection."""
        self.fit_panel.refresh_table()

    def _row(self, address):
        """The parameter row of a fit address (see :meth:`parameter_entries`)."""
        if address == "background":
            return self.bg_row
        if address == "bg.A":
            return self.bgA_row
        if address == "bg.alpha":
            return self.bgalpha_row
        ms = self.models()
        if address.startswith("scale."):
            return ms[int(address[6:])].scale_row
        i, role, name = address.split(".", 2)
        return next((r for r in ms[int(i)].param_rows[role] if r.name == name), None)

    def fit_selection(self):
        """``[(address, label, lower text, upper text)]`` of the parameters ticked for fitting."""
        out = []
        for address, label, _lo, _hi in self.parameter_entries():
            row = self._row(address)
            if row is not None and row.is_fit():
                out.append((address, label, *row.bounds_text()))
        return out

    # What the fit panel needs: the model, its parameters, a way back.
    def build_model(self):
        """A :class:`~pysasfit.studio.ModelFunction` of the mixture (the contributions are the live objects; with data
        sets the background of the current one)."""
        return self.model_for(None)

    def parameter_entries(self):
        """``[(address, label, default lower, default upper)]`` of all parameters of the mixture (fit addresses)."""
        ms = self.models()
        multi = len(ms) > 1
        out = [("background", "background", "", ""), ("bg.A", "bg.A", "0", ""), ("bg.alpha", "bg.alpha", "0", "8")]
        for i, p in enumerate(ms):
            pre = f"c{i + 1}." if multi else ""
            has_sd = p.funcs["sd"] is not None
            if not has_sd:  # with a size distribution its amplitude N is the scale
                out.append((f"scale.{i}", pre + "scale", "0", ""))
            for role in ROLES:
                f = p.funcs[role]
                if f is None:
                    continue
                vec = p.contrib.vector(role)
                for k, par in enumerate(f.info.parameters[: len(vec)]):
                    if (has_sd and role == "ff" and k == p.contrib.distributed) or is_dummy(par.name):
                        continue  # integrated over the distribution / not used by the function
                    lo, hi = par.bounds
                    out.append((
                        f"{i}.{role}.{par.name}", f"{pre}{role}.{par.name}",
                        f"{lo:g}" if np.isfinite(lo) else "", f"{hi:g}" if np.isfinite(hi) else "",
                    ))  # fmt: skip
        return out

    def value_of(self, address):
        """Current value of a fit address (see :meth:`parameter_entries`)."""
        if address == "background":
            return self.bg_row.value
        if address == "bg.A":
            return self.bgA_row.value
        if address == "bg.alpha":
            return self.bgalpha_row.value
        ms = self.models()
        if address.startswith("scale."):
            return ms[int(address[6:])].scale_row.value
        i, role, name = address.split(".", 2)
        p = ms[int(i)]
        return float(p.contrib.vector(role)[p.funcs[role].parameter_names.index(name)])

    def snapshot(self):
        """All parameter values, to come back to with :meth:`restore`."""
        self._store_bg()
        return {
            "background": self.bg_row.value, "bg_A": self.bgA_row.value, "bg_alpha": self.bgalpha_row.value,
            "ds_bg": [dict(d["bg"]) for d in self.datasets],
            "overrides": {k: dict(v) for k, v in self.overrides.items()},
            "globals": self.fit_panel.global_table_values(),
            "items": [
                (p, p.scale_row.value, {r: p.contrib.vector(r).copy() for r in ROLES if p.contrib.vector(r) is not None})
                for p in self.models()
            ],
        }

    def restore(self, snap):
        for p, scale, vecs in snap["items"]:
            if p.contrib is None:
                continue
            for role, v in vecs.items():
                vec = p.contrib.vector(role)
                if vec is not None and vec.shape == v.shape:
                    vec[:] = v
            p.scale_row.set_value(scale, emit=False)
        self.bg_row.set_value(snap["background"], emit=False)
        self.bgA_row.set_value(snap.get("bg_A", 0.0), emit=False)
        self.bgalpha_row.set_value(snap.get("bg_alpha", 4.0), emit=False)
        for d, bg in zip(self.datasets, snap.get("ds_bg", [])):
            d["bg"] = dict(bg)
        self.overrides = {k: dict(v) for k, v in snap.get("overrides", {}).items()}
        if "globals" in snap:
            self.fit_panel.set_global_table_values(snap["globals"])
        self.after_fit()

    def apply_model_values(self, model):
        """Take the scales and the background of a (fitted) model built by :meth:`build_model`."""
        for p, s in zip(self.models(), model.scales):
            p.scale_row.set_value(s, emit=False)
        self.bg_row.set_value(model.background, emit=False)
        self.bgA_row.set_value(model.bg_A, emit=False)
        self.bgalpha_row.set_value(model.bg_alpha, emit=False)
        self.after_fit()

    def after_fit(self):
        """The parameter vectors changed (fit, undo): bring the sliders along and redraw."""
        for p in self.panels():
            p.sync_rows()
        self.fit_panel.refresh_values()
        self._schedule()

    # --------------------------------------------------------------- curves
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
    def _python_snippet(self):
        ms = self.models()
        lines = ["from pysasfit.studio import ModelFunction, ScatteringContribution"]
        extra = user_precision()
        if extra:
            lines += ["from pysasfit.studio.library import configure_precision", "", f"configure_precision(**{extra!r})"]
        for i, p in enumerate(ms):
            c = p.contrib
            ff, sd, sq = (p.funcs[r] for r in ROLES)
            dist = ff.parameter_names[c.distributed] if c.distributed < len(ff.parameter_names) else c.distributed
            lines += ["", f"c{i + 1} = ScatteringContribution(", f'    "{ff.name}", {p.param_dict("ff")!r},']
            if sd is not None:
                lines.append(f'    size_distribution="{sd.name}", sd_params={p.param_dict("sd")!r}, distributed={dist!r},')
            if sq is not None:
                lines.append(f'    structure_factor="{sq.name}", sq_params={p.param_dict("sq")!r},')
                if sd is not None:
                    lines.append(f'    approach="{c.approach}",')
            if sd is not None:
                lines.append("    " + ", ".join(f"{k}={v!r}" for k, v in self.integration.items()) + ",")
            lines.append(")")
        names = ", ".join(f"c{i + 1}" for i in range(len(ms)))
        scales = ", ".join(repr(p.scale_row.value) for p in ms)
        bg = f"background={self.bg_row.value!r}"
        if self.bgA_row.value:
            bg += f", bg_A={self.bgA_row.value!r}, bg_alpha={self.bgalpha_row.value!r}"
        lines += ["", f"model = ModelFunction([{names}], scales=[{scales}], {bg})", "I = model(q)"]
        return "\n".join(lines)

    def _copy_python(self):
        from ..core.pycode import copy_to_clipboard

        copy_to_clipboard(self.python_code, self, self._status)

    def python_code(self):
        """Script: the contributions, the curve of the plot (q grid, P*S / P / S) and, with data sets, their files,
        the model of every data set and the fit as set up in the Fit panel ("Copy Python code")."""
        from ..core.pycode import Code, Script, call, lit, path_lit

        if not self.has_model():
            raise ValueError("No model yet: choose a form factor first.")
        ms = self.models()
        names = [f"c{i + 1}" for i in range(len(ms))]
        sc = Script("pySASstudio", "SASfit model" + (" and fit" if self.datasets else ""))
        sc.imp("import numpy as np")
        body = self._python_snippet().split("\n")
        # the imports of the snippet go to the top, the model line is rebuilt below
        while body and (body[0].startswith(("from ", "import ")) or not body[0].strip()):
            sc.imp(body.pop(0)) if body[0].strip() else body.pop(0)
        body = [ln for ln in body if not ln.startswith(("model = ModelFunction(", "I = model(q)"))]
        from .functions import _PYTHON_FUNCTIONS

        pyf = sorted({f.name for p in ms for f in p.funcs.values() if f is not None and f.name in _PYTHON_FUNCTIONS})
        if pyf:
            sc.comment(f"{', '.join(pyf)}: Python function(s) of this session; register them first with "
                       "pysasfit.studio.register_python_function (or put the plugin file in a python plugin folder)")
        sc.comment("the contributions (form factor, size distribution, structure factor)")
        sc.add(*body)
        while sc.body and not sc.body[-1].strip():
            sc.body.pop()
        # the curve of the plot
        lo, hi = float(self.qmin.text()), float(self.qmax.text())
        n = self.npoints.value()
        sc.comment(f"the curve of the plot (q in {self.units.currentText()})")
        sc.add(f"q = np.{'geomspace' if self.logx.isChecked() else 'linspace'}({lit(lo)}, {lit(hi)}, {n})")
        bg = self.background_of(None)
        mkw = dict(scales=[p.scale_row.value for p in ms], background=bg["background"])
        if bg["bg_A"]:
            mkw.update(bg_A=bg["bg_A"], bg_alpha=bg["bg_alpha"])
        sc.add(call("model = ModelFunction", Code("[" + ", ".join(names) + "]"), **mkw))
        mode = self.mode_combo.currentData()
        if mode == "S":
            p = self.current_panel()
            i = ms.index(p) if p in ms else 0
            sc.add(f"I = np.asarray({names[i]}.sq.evaluate_p(q, {names[i]}.sq_p), float)   # structure factor S(q) of {names[i]}")
            ylabel = "S(q)"
        elif mode == "P":
            sc.add("",
                   "def without_structure_factor(c, q):",
                   "    sq, c.sq = c.sq, None",
                   "    try:",
                   "        return c.intensity(q)",
                   "    finally:",
                   "        c.sq = sq",
                   "",
                   "# P(q): the contributions without their structure factors",
                   "I = sum(s * without_structure_factor(c, q) for s, c in zip(model.scales, model.contributions))",
                   f"I = I + {lit(bg['background'])}" + (f" + {lit(bg['bg_A'])} * q ** -{lit(bg['bg_alpha'])}"
                                                          if bg["bg_A"] else ""))
            ylabel = "P(q)"
        else:
            sc.add("I = model(q)")
            ylabel = "I(q)"
        plot = ["fig, ax = plt.subplots()", f"ax.plot(q, I, label={ylabel!r})"]
        if self.datasets:
            sc.imp("from pysasfit.core.io import load_data")
            sc.comment("the data sets and the model of each (its contributions and background)")
            dnames = []
            for k, d in enumerate(self.datasets):
                v = f"d{k + 1}"
                dnames.append(v)
                if d.get("path") and os.path.isfile(d["path"]):
                    qs = d.get("q_scale", 1.0)
                    sc.add(f"q{k + 1}, I{k + 1}, s{k + 1}, *_ = load_data({path_lit(d['path'])}"
                           + (f", q_scale={lit(qs)})" if qs != 1.0 else ")"))
                    sc.add(f"ok = np.isfinite(q{k + 1}) & np.isfinite(I{k + 1}) & (q{k + 1} > 0)",
                           f"q{k + 1}, I{k + 1}, s{k + 1} = q{k + 1}[ok], I{k + 1}[ok], s{k + 1}[ok]")
                else:
                    sc.add(f"# {d['name']}: no file (from a project); the values as stored",
                           f"q{k + 1} = {lit(d['q'])}", f"I{k + 1} = {lit(d['I'])}", f"s{k + 1} = {lit(d['sigma'])}")
                bgk = self.background_of(k)
                kw = dict(scales=[p.scale_row.value for p in ms], background=bgk["background"])
                if bgk["bg_A"]:
                    kw.update(bg_A=bgk["bg_A"], bg_alpha=bgk["bg_alpha"])
                active = [p.applies_to(k, len(self.datasets)) for p in ms]
                if not all(active):
                    kw["active"] = active
                sc.add(call(f"model{k + 1} = ModelFunction", Code("[" + ", ".join(names) + "]"), **kw))
                plot.append(f"ax.errorbar(q{k + 1}, I{k + 1}, s{k + 1}, fmt='o', ms=3, label={d['name']!r})")
            self.fit_panel.python_fit_code(sc, len(self.datasets))
            for k in range(len(self.datasets)):
                plot.append(f"ax.plot(q{k + 1}, model{k + 1}(q{k + 1}), '-', label='model {k + 1}')")
        if self.logx.isChecked():
            plot.append("ax.set_xscale('log')")
        if self.logy.isChecked():
            plot.append("ax.set_yscale('log')")
        plot += [f"ax.set_xlabel('q ({self.units.currentText()})')", f"ax.set_ylabel({ylabel!r})", "ax.legend()"]
        sc.add_plot(*plot)
        return sc.text()

    def _export(self):
        if self._last is None or not self.has_model():
            return
        name = "_".join(p.funcs["ff"].name for p in self.models())
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
def explore(ff="ff_sphere", sd=None, sq=None, qmin=0.01, qmax=1.0, extra=(), project=None, restore_layout=False):
    """Open the explorer; blocks until it is closed unless a Qt event loop is already running.

    ``extra``: more contributions of the mixture, each ``(form factor, size distribution, structure factor)``.
    ``project``: a ``.sasstudio`` file to open (File > Open project).
    ``restore_layout``: arrange the panels as in the last session (the stand-alone program does).
    """
    app = QtWidgets.QApplication.instance()
    own = app is None
    if own:
        app = QtWidgets.QApplication(sys.argv[:1])
    win = ModelExplorer(ff, sd=sd, sq=sq, qmin=qmin, qmax=qmax, extra=extra)
    if project:
        win.open_project(project)
    if restore_layout:
        win.restore_session()
    win.show()
    if own:
        app.exec()
    return win


def main(argv=None):
    ap = argparse.ArgumentParser(prog="pysasfit-explorer", description="Interactive plot of SASfit model functions.")
    ap.add_argument("model", nargs="?", default="ff_sphere", help="form factor, e.g. ff_sphere (see 'pysasfit-studio list')")
    ap.add_argument("--sq", help="structure factor, e.g. sq_hardsphere")
    ap.add_argument("--sd", help="size distribution, e.g. sd_LogNorm")
    ap.add_argument(
        "--add", action="append", default=[], metavar="FF[,SD[,SQ]]",
        help="another contribution of the mixture, e.g. --add ff_cylinder,sd_LogNorm (repeat for more)",
    )  # fmt: skip
    ap.add_argument("--project", metavar="FILE", help="open a project saved from the GUI (File > Save project)")
    ap.add_argument("--qmin", type=float, default=0.01)
    ap.add_argument("--qmax", type=float, default=1.0)
    ap.add_argument("--lib", help="folder of a local SASfit installation (this session only; see Settings in the GUI)")
    a = ap.parse_args(argv)
    if a.lib:
        paths.set_lib_dir(a.lib)
    extra = []
    for spec in a.add:
        parts = [s.strip() or None for s in spec.split(",")][:3]
        extra.append(tuple(parts + [None] * (3 - len(parts))))
    explore(a.model, sd=a.sd, sq=a.sq, qmin=a.qmin, qmax=a.qmax, extra=extra, project=a.project, restore_layout=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
