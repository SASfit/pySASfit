"""pyDENSSmulti window: 3D maps from several SAS contrasts (SANS contrast variation, SANS + SAXS).

    pysasfit-densitymulti [data files]
    python -m pysasfit.densitymulti.gui [data files]

Also the "pyDENSSmulti" tab of the pySASfit launcher. Panels (View menu; each can be moved, stacked or detached):

Data            loaded curves (text, pySASred ASCII, NXcanSAS); "Add as contrasts" puts the selected ones into the
                contrast table (the D2O fraction is guessed from the name, e.g. "42% D2O")
Components      the components of the particle: SLD in H2O, in D2O (full exchange of labile H), exchanged fraction,
                X-ray SLD; the match point; solvent SLDs of H2O and D2O
Contrasts       one row per curve: radiation, D2O fraction, relative concentration, weight, flat background, q range;
                the contrast matrix, its condition number and what can be separated, the match point from I(0)
Reconstruction  scheme (component basis or DENSS-Multiple-type averaging), Dmax, grid, steps, scale (fitted or
                absolute from concentration and molar mass), several maps with alignment and averaging
Density (3D)    isosurfaces of the volume fractions (one colour per component) or of the occupancy
I(q)            data and the scattering of the map for every contrast, residuals
Convergence     χ² per contrast and support volume per step
Slices          three central slices of a component, of the SLD map or of a contrast map
Results / Log   one row per run; messages

The reconstruction is pySASfit's own numpy code (no DENSS installation needed). Every run keeps its maps (MRC), fits
and a summary in its own folder (Settings -> Preferences -> pyDENSSmulti). q in 1/nm, maps in Å, SLD in 1e-6 Å⁻².
"""
from __future__ import annotations

import datetime
import json
import os
import re
import sys
import threading
import traceback
from pathlib import Path

import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402

from ..core import settings as prefs  # noqa: E402
from ..core.docking import DockMixin, plot_panel  # noqa: E402
from ..resources import tool_icon  # noqa: E402
from . import core as M  # noqa: E402

__all__ = ["MainWindow", "main"]

Dock = QtCore.Qt.DockWidgetArea
TOOL = "pyDENSSmulti"
COMP_COLS = ["name", "SLD H2O", "SLD D2O", "exchanged", "SLD X-ray", "match point"]
CON_COLS = ["curve", "radiation", "D2O %", "solvent SLD", "conc.", "weight", "bg", "q min", "q max"]
RES_COLS = ["run", "scheme", "maps", "χ² per contrast", "volumes (Å³)", "support V (Å³)", "steps", "time (s)", "folder"]
# component colours (as the 3D view's palette order is not under our control, the slices and plots use these)
PALETTE = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e", "#17becf", "#8c564b", "#e377c2"]


def _pref(key, default):
    try:
        v = prefs.get(key)
    except Exception:  # noqa: BLE001 -- key not registered (older preferences schema)
        return default
    return default if v in (None, "") else v


def _canvas(size=(5, 4)):
    return FigureCanvasQTAgg(Figure(figsize=size, layout="constrained"))


def _fmt(v, n=4):
    try:
        v = float(v)
    except (TypeError, ValueError):
        return "" if v is None else str(v)
    return "" if not np.isfinite(v) else f"{v:.{n}g}"


def _safe(name, n=40):
    return re.sub(r"[^A-Za-z0-9_.-]", "_", name)[:n] or "run"


def _spin(lo, hi, val, dec=0, special=None, suffix="", tip="", step=None):
    sp = QtWidgets.QDoubleSpinBox() if dec else QtWidgets.QSpinBox()
    sp.setRange(lo, hi)
    if dec:
        sp.setDecimals(dec)
    sp.setValue(val)
    if special:
        sp.setSpecialValueText(special)
    if suffix:
        sp.setSuffix(suffix)
    if tip:
        sp.setToolTip(tip)
    if step:
        sp.setSingleStep(step)
    sp.setKeyboardTracking(False)
    return sp


def guess_d2o(name):
    """D2O fraction (0 … 1) from a curve name such as 'lyz_42%D2O', '70 % D2O', 'D2O_100'; None if not found."""
    s = name.replace(",", ".")
    for pat in (r"(\d+(?:\.\d+)?)\s*%\s*D2O", r"D2O[\s_-]*(\d+(?:\.\d+)?)\s*%?", r"(\d+(?:\.\d+)?)\s*%?\s*[_-]?D2O"):
        m = re.search(pat, s, re.IGNORECASE)
        if m:
            v = float(m.group(1))
            if 0 <= v <= 100:
                return v / 100
    if re.search(r"\bH2O\b", s, re.IGNORECASE):
        return 0.0
    return None


class _Signals(QtCore.QObject):
    done = QtCore.Signal(object, object, object)
    progress = QtCore.Signal(str)


class MainWindow(DockMixin, QtWidgets.QMainWindow):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("pyDENSSmulti – 3D maps from several SAS contrasts")
        self.setWindowIcon(tool_icon(TOOL))
        self.curves = []               # loaded curves (toolbox.merge.Curve)
        self.runs = []                 # dicts: name, kind ("single"/"average"), result (MultiResult), avg
        self._busy = False
        self._cancel = False
        self._signals = _Signals()
        self._signals.done.connect(self._on_done)
        self._signals.progress.connect(lambda t: self.progress_label.setText(t))
        self._build_menus()
        self._build_data_panel()
        self._build_components_panel()
        self._build_contrasts_panel()
        self._build_run_panel()
        self._build_plots()
        from ..atsas.view3d import ModelView
        self.view3d = ModelView()
        self.view3d.centre.setChecked(False)          # the components share one frame
        self.density_widget = self._density_panel(self.view3d)
        self.table = QtWidgets.QTableWidget(0, len(RES_COLS))
        self.table.setHorizontalHeaderLabels(RES_COLS)
        self.table.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.ResizeMode.ResizeToContents)
        self.table.currentCellChanged.connect(lambda r, *_a: self._select_run(r))
        self.log_view = QtWidgets.QPlainTextEdit()
        self.log_view.setReadOnly(True)
        self.log_view.setFont(QtGui.QFontDatabase.systemFont(QtGui.QFontDatabase.SystemFont.FixedFont))
        self._add_dock("data", "Data", self.data_widget, Dock.LeftDockWidgetArea, scrollable=True)
        self._add_dock("components", "Components", self.comp_widget, Dock.LeftDockWidgetArea, scrollable=True)
        self._add_dock("contrasts", "Contrasts", self.con_widget, Dock.LeftDockWidgetArea, scrollable=True)
        self._add_dock("run", "Reconstruction", self.run_widget, Dock.LeftDockWidgetArea, scrollable=True)
        self._add_dock("density", "Density (3D)", self.density_widget, Dock.RightDockWidgetArea)
        self._add_dock("iq", "I(q)", self.iq_panel, Dock.RightDockWidgetArea)
        self._add_dock("conv", "Convergence", self.conv_panel, Dock.RightDockWidgetArea)
        self._add_dock("slices", "Slices", self.slice_panel, Dock.RightDockWidgetArea)
        self._add_dock("results", "Results", self.table, Dock.RightDockWidgetArea)
        self._add_dock("log", "Log", self.log_view, Dock.RightDockWidgetArea)
        self.tabifyDockWidget(self.docks["data"], self.docks["components"])
        self.tabifyDockWidget(self.docks["data"], self.docks["contrasts"])
        self.splitDockWidget(self.docks["data"], self.docks["run"], QtCore.Qt.Orientation.Vertical)
        self.splitDockWidget(self.docks["density"], self.docks["iq"], QtCore.Qt.Orientation.Horizontal)
        self.splitDockWidget(self.docks["density"], self.docks["results"], QtCore.Qt.Orientation.Vertical)
        self.tabifyDockWidget(self.docks["iq"], self.docks["conv"])
        self.tabifyDockWidget(self.docks["iq"], self.docks["slices"])
        self.tabifyDockWidget(self.docks["results"], self.docks["log"])
        self.docks["contrasts"].raise_()
        self.docks["iq"].raise_()
        self.docks["results"].raise_()
        self._finish_docks()
        self.resize(1550, 950)
        self._sized = False
        self.hide_stray_tab_bars()
        self.set_components([M.Component(n, *M.PRESETS[n][:2], 1.0, M.PRESETS[n][2])
                             for n in ("protein (typical)",)])
        self.update_matrix()

    def showEvent(self, event):
        super().showEvent(event)
        if not self._sized:
            self._sized = True
            QtCore.QTimer.singleShot(0, self._initial_sizes)

    def _initial_sizes(self):
        if getattr(self, "_session_layout", False):
            return
        w = max(self.width(), 900)
        self.resizeDocks([self.docks["data"], self.docks["density"], self.docks["iq"]],
                         [int(0.34 * w), int(0.34 * w), int(0.32 * w)], QtCore.Qt.Orientation.Horizontal)
        h = max(self.height(), 600)
        self.resizeDocks([self.docks["data"], self.docks["run"]], [int(0.55 * h), int(0.45 * h)],
                         QtCore.Qt.Orientation.Vertical)
        self._start_layout = self.saveState()

    # ------------------------------------------------------------------ construction
    def _build_menus(self):
        m = self.menuBar().addMenu("&File")
        a = m.addAction("&Load data…", self.on_load)
        a.setShortcut(QtGui.QKeySequence("Ctrl+L"))
        m.addAction("Open a result folder…", self.on_open_result)
        m.addSeparator()
        m.addAction("Example: core-shell sphere (4 contrasts)", lambda: self.load_example("core-shell"))
        m.addAction("Example: two spheres (4 contrasts)", lambda: self.load_example("dumbbell"))
        m.addSeparator()
        m.addAction("Export results table (CSV)…", self.on_export_table)
        m.addAction("Open the run folder", self.on_open_runs)
        from ..core import pycode
        pycode.install(self, self.python_code, menu=m)
        from ..core.settings_dialog import add_preferences_menu
        add_preferences_menu(self, "densitymulti")
        self.menuBar().addMenu("&View")

    def _build_data_panel(self):
        w = QtWidgets.QWidget()
        lay = QtWidgets.QVBoxLayout(w)
        lay.setContentsMargins(4, 4, 4, 4)
        row = QtWidgets.QHBoxLayout()
        for text, slot, tip in (("Load…", self.on_load, "Curves: text, pySASred ASCII, NXcanSAS; q in 1/nm"),
                                ("Remove", self.on_remove_curve, "Remove the selected curves (and their contrasts)")):
            b = QtWidgets.QPushButton(text)
            b.setToolTip(tip)
            b.clicked.connect(lambda _c=False, s=slot: s())
            row.addWidget(b)
        lay.addLayout(row)
        self.curve_list = QtWidgets.QListWidget()
        self.curve_list.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.ExtendedSelection)
        self.curve_list.currentRowChanged.connect(self._on_curve_changed)
        lay.addWidget(self.curve_list, 1)
        self.curve_info = QtWidgets.QLabel("no data")
        self.curve_info.setWordWrap(True)
        self.curve_info.setStyleSheet("color: gray;")
        lay.addWidget(self.curve_info)
        b = QtWidgets.QPushButton("Add selected as contrasts")
        b.setToolTip("One row per selected curve in the Contrasts panel; the D2O fraction is guessed from the name")
        b.clicked.connect(self.on_add_contrasts)
        lay.addWidget(b)
        self.data_widget = w

    def _build_components_panel(self):
        w = QtWidgets.QWidget()
        lay = QtWidgets.QVBoxLayout(w)
        lay.setContentsMargins(4, 4, 4, 4)
        note = QtWidgets.QLabel("SLD in 1e-6 Å⁻². 'SLD D2O': in 100 % D2O when all labile H are exchanged "
                                "(= SLD H2O for components without exchangeable H); 'exchanged': fraction of the "
                                "labile H that do exchange. Presets are typical values – compute yours from the "
                                "composition.")
        note.setWordWrap(True)
        note.setStyleSheet("color: gray;")
        lay.addWidget(note)
        self.comp_table = QtWidgets.QTableWidget(0, len(COMP_COLS))
        self.comp_table.setHorizontalHeaderLabels(COMP_COLS)
        self.comp_table.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.ResizeMode.ResizeToContents)
        self.comp_table.setMinimumHeight(120)
        lay.addWidget(self.comp_table, 1)
        row = QtWidgets.QHBoxLayout()
        self.preset = QtWidgets.QComboBox()
        self.preset.addItems(list(M.PRESETS))
        row.addWidget(self.preset, 1)
        b = QtWidgets.QPushButton("Add")
        b.clicked.connect(self.on_add_component)
        row.addWidget(b)
        b = QtWidgets.QPushButton("Remove")
        b.clicked.connect(self.on_remove_component)
        row.addWidget(b)
        lay.addLayout(row)
        g = QtWidgets.QGroupBox("Solvent (neutrons)")
        f = QtWidgets.QFormLayout(g)
        self.sld_h2o = _spin(-5, 10, M.SLD_H2O, dec=3, suffix=" e-6 Å⁻²")
        self.sld_d2o = _spin(-5, 10, M.SLD_D2O, dec=3, suffix=" e-6 Å⁻²")
        for sp in (self.sld_h2o, self.sld_d2o):
            sp.valueChanged.connect(lambda _v: self.update_matrix())
        f.addRow("H2O", self.sld_h2o)
        f.addRow("D2O", self.sld_d2o)
        lay.addWidget(g)
        self.comp_widget = w

    def _build_contrasts_panel(self):
        w = QtWidgets.QWidget()
        lay = QtWidgets.QVBoxLayout(w)
        lay.setContentsMargins(4, 4, 4, 4)
        self.con_table = QtWidgets.QTableWidget(0, len(CON_COLS))
        self.con_table.setHorizontalHeaderLabels(CON_COLS)
        self.con_table.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.ResizeMode.ResizeToContents)
        self.con_table.setMinimumHeight(140)
        tips = {"conc.": "relative concentration of this sample (model intensity × this); 1 for one stock",
                "weight": "weight of this contrast in the per-voxel combination",
                "bg": "flat background subtracted from I (DENSS-type fits use no constant)",
                "solvent SLD": "from the D2O fraction (X-rays: water); type a value to override",
                "q min": "1/nm, 0 = from the first point", "q max": "1/nm, 0 = up to the last point"}
        for j, c in enumerate(CON_COLS):
            if c in tips:
                self.con_table.horizontalHeaderItem(j).setToolTip(tips[c])
        lay.addWidget(self.con_table, 1)
        row = QtWidgets.QHBoxLayout()
        b = QtWidgets.QPushButton("Remove")
        b.clicked.connect(self.on_remove_contrast)
        row.addWidget(b)
        b = QtWidgets.QPushButton("Fit tail backgrounds")
        b.setToolTip("Flat background of every contrast from its last 30 % of points: I = A q⁻⁴ + bg")
        b.clicked.connect(self.fit_backgrounds)
        row.addWidget(b)
        row.addStretch(1)
        lay.addLayout(row)
        g = QtWidgets.QGroupBox("Contrast matrix C = ρ_component − ρ_solvent (1e-6 Å⁻²)")
        v = QtWidgets.QVBoxLayout(g)
        self.matrix_label = QtWidgets.QLabel("")
        self.matrix_label.setTextFormat(QtCore.Qt.TextFormat.RichText)
        self.matrix_label.setWordWrap(True)
        self.matrix_label.setTextInteractionFlags(QtCore.Qt.TextInteractionFlag.TextSelectableByMouse)
        v.addWidget(self.matrix_label)
        lay.addWidget(g)
        self.con_widget = w

    def _build_run_panel(self):
        w = QtWidgets.QWidget()
        lay = QtWidgets.QVBoxLayout(w)
        lay.setContentsMargins(4, 4, 4, 4)
        g = QtWidgets.QGroupBox("Map")
        f = QtWidgets.QFormLayout(g)
        self.scheme = QtWidgets.QComboBox()
        self.scheme.addItem("component basis (volume fractions)", "components")
        self.scheme.addItem("averaging (DENSS-Multiple type, one SLD)", "average")
        self.scheme.setToolTip("Component basis: one volume-fraction map per component, combined by per-voxel "
                               "least squares with the contrast matrix (match points and H/D exchange handled).\n"
                               "Averaging: one SLD map, the same in every contrast (no exchange), the contrasts' "
                               "proposals averaged in real space (after Sumner & Qian 2022) – for comparison.")
        self.scheme.currentIndexChanged.connect(lambda _i: (self._scheme_changed(), self.update_matrix()))
        f.addRow("scheme", self.scheme)
        self.dmax = _spin(0.1, 1e4, 10.0, dec=2, suffix=" nm", tip="Maximum dimension of the particle")
        b = QtWidgets.QPushButton("estimate")
        b.setToolTip("3.5 × the Guinier Rg of the first contrast (a rough start; check with p(r))")
        b.clicked.connect(self.estimate_dmax)
        r = QtWidgets.QHBoxLayout()
        r.addWidget(self.dmax, 1)
        r.addWidget(b)
        f.addRow("Dmax", r)
        self.grid_n = _spin(0, 160, 0, special="from the data", tip="Grid points per side (even); default: so that "
                            "the grid reaches the largest q of the data (at least 32)")
        f.addRow("grid points", self.grid_n)
        self.oversampling = _spin(2.0, 6.0, 3.0, dec=1, tip="Box side = oversampling × Dmax")
        f.addRow("oversampling", self.oversampling)
        self.steps = _spin(100, 100000, 3000, tip="Largest number of iterations (convergence after ≥ 1000)")
        f.addRow("max. steps", self.steps)
        self.threshold = _spin(0.05, 0.6, 0.2, dec=2, step=0.05, tip="Shrink-wrap: support where the smoothed map "
                               "exceeds this fraction of its maximum (DENSS default 0.2)")
        f.addRow("shrink-wrap", self.threshold)
        self.seed = _spin(0, 2_000_000_000, 0, special="random", tip="For a reproducible map")
        f.addRow("seed", self.seed)
        lay.addWidget(g)
        g = QtWidgets.QGroupBox("Scale")
        f = QtWidgets.QFormLayout(g)
        self.scale_mode = QtWidgets.QComboBox()
        self.scale_mode.addItem("fitted (one factor for all curves)", "fitted")
        self.scale_mode.addItem("absolute (data in 1/cm)", "absolute")
        self.scale_mode.setToolTip("Fitted: one common factor; the densest voxels are taken as fully occupied.\n"
                                   "Absolute: I = n_p |A|² with n_p from concentration and molar mass; then the "
                                   "volume fractions are absolute.")
        self.scale_mode.currentIndexChanged.connect(lambda _i: self._scheme_changed())
        f.addRow("scale", self.scale_mode)
        self.conc = _spin(0.0001, 1000, 5.0, dec=4, suffix=" mg/ml")
        self.mw = _spin(0.1, 1e6, 100.0, dec=2, suffix=" kDa")
        f.addRow("concentration", self.conc)
        f.addRow("molar mass", self.mw)
        lay.addWidget(g)
        self.avg_group = QtWidgets.QGroupBox("Averaging scheme")
        f = QtWidgets.QFormLayout(self.avg_group)
        self.sld_lo = _spin(-10, 20, 0.0, dec=2, tip="Lower bound of the particle SLD (1e-6 Å⁻²)")
        self.sld_hi = _spin(-10, 20, 0.0, dec=2, tip="Upper bound of the particle SLD; equal bounds: no bounds")
        r = QtWidgets.QHBoxLayout()
        r.addWidget(self.sld_lo)
        r.addWidget(QtWidgets.QLabel("…"))
        r.addWidget(self.sld_hi)
        f.addRow("SLD range", r)
        lay.addWidget(self.avg_group)
        g = QtWidgets.QGroupBox("Several maps")
        f = QtWidgets.QFormLayout(g)
        self.nmaps = _spin(1, 200, 1, tip="Independent maps (different seeds); > 1: aligned (principal axes, mirror "
                           "images allowed) and averaged")
        f.addRow("maps", self.nmaps)
        self.threads = _spin(1, 256, int(_pref("densitymulti.threads", max(1, min(4, (os.cpu_count() or 2) - 1)))),
                             tip="Maps computed at the same time")
        f.addRow("threads", self.threads)
        lay.addWidget(g)
        row = QtWidgets.QHBoxLayout()
        self.run_button = QtWidgets.QPushButton("Run")
        self.run_button.setToolTip("Reconstruct from all contrasts (32³ grid: seconds to a minute per map)")
        self.run_button.clicked.connect(self.run)
        row.addWidget(self.run_button)
        self.stop_button = QtWidgets.QPushButton("Stop")
        self.stop_button.setEnabled(False)
        self.stop_button.clicked.connect(self.stop)
        row.addWidget(self.stop_button)
        self.selftest_button = QtWidgets.QPushButton("Self test")
        self.selftest_button.setToolTip("One map of a sphere R = 2 nm (exact Rg 15.49 Å); passes within 10 %")
        self.selftest_button.clicked.connect(self.run_selftest)
        row.addWidget(self.selftest_button)
        lay.addLayout(row)
        self.progress = QtWidgets.QProgressBar()
        self.progress.setRange(0, 1)
        self.progress.setTextVisible(False)
        self.progress.setMaximumHeight(8)
        lay.addWidget(self.progress)
        self.progress_label = QtWidgets.QLabel("")
        self.progress_label.setStyleSheet("color: gray;")
        lay.addWidget(self.progress_label)
        row = QtWidgets.QHBoxLayout()
        from ..core import pycode
        row.addWidget(pycode.button(self.python_code, w))
        row.addStretch(1)
        lay.addLayout(row)
        note = QtWidgets.QLabel("Iteration after DENSS (T. D. Grant, Nature Methods 15, 191 (2018)); averaging after "
                                "DENSS-Multiple (J. Sumner & S. Qian, BBA Advances 2, 100063 (2022)). pySASfit's own "
                                "implementation: please cite the methods when you publish maps.")
        note.setWordWrap(True)
        note.setStyleSheet("color: gray;")
        lay.addWidget(note)
        lay.addStretch(1)
        self.run_widget = w
        self._scheme_changed()

    def _scheme_changed(self):
        avg = self.scheme.currentData() == "average"
        self.avg_group.setVisible(avg)
        absolute = self.scale_mode.currentData() == "absolute"
        self.conc.setEnabled(absolute)
        self.mw.setEnabled(absolute)

    def _density_panel(self, view):
        w = QtWidgets.QWidget()
        lay = QtWidgets.QVBoxLayout(w)
        lay.setContentsMargins(2, 2, 2, 2)
        bar = QtWidgets.QHBoxLayout()
        bar.addWidget(QtWidgets.QLabel("level φ ="))
        self.level = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.level.setRange(5, 95)
        self.level.setValue(50)
        self.level.setToolTip("Isosurface of each volume fraction (or of the occupancy) at this value")
        self.level.sliderReleased.connect(self.update_surface)
        self.level.valueChanged.connect(lambda v: self.level_label.setText(f"{v / 100:.2f}"))
        bar.addWidget(self.level, 1)
        self.level_label = QtWidgets.QLabel("0.50")
        bar.addWidget(self.level_label)
        self.see_through = QtWidgets.QCheckBox("see-through")
        self.see_through.setChecked(True)
        self.see_through.setToolTip("All components but the smallest translucent (opacity: 'envelope' slider)")
        self.see_through.toggled.connect(lambda _c: self.update_surface())
        bar.addWidget(self.see_through)
        self.show_support = QtWidgets.QCheckBox("support")
        self.show_support.setToolTip("Also draw the final support (translucent)")
        self.show_support.toggled.connect(lambda _c: self.update_surface())
        bar.addWidget(self.show_support)
        lay.addLayout(bar)
        lay.addWidget(view, 1)
        return w

    def _build_plots(self):
        from ..core.flowlayout import flow_bar
        self.canvas_iq = _canvas()
        self.iq_scale = QtWidgets.QComboBox()
        self.iq_scale.addItems(["log I vs log q", "log I vs q", "Kratky q² I"])
        self.iq_scale.currentIndexChanged.connect(lambda _i: self.draw_iq())
        self.iq_offset = QtWidgets.QCheckBox("offset curves")
        self.iq_offset.setChecked(True)
        self.iq_offset.setToolTip("Multiply curve k by 10^k for clarity (log scales)")
        self.iq_offset.toggled.connect(lambda _c: self.draw_iq())
        self.iq_panel = plot_panel(self.canvas_iq, self, bars=(flow_bar(QtWidgets.QLabel("scale"), self.iq_scale,
                                                                        self.iq_offset),))
        self.canvas_conv = _canvas()
        self.conv_panel = plot_panel(self.canvas_conv, self)
        self.canvas_slices = _canvas((6, 2.4))
        self.slice_what = QtWidgets.QComboBox()
        self.slice_what.setMinimumContentsLength(18)
        self.slice_what.currentIndexChanged.connect(lambda _i: self.draw_slices())
        self.slice_panel = plot_panel(self.canvas_slices, self, bars=(flow_bar(QtWidgets.QLabel("map"),
                                                                               self.slice_what),))
        self._slice_timer = QtCore.QTimer(self)
        self._slice_timer.setSingleShot(True)
        self._slice_timer.setInterval(250)
        self._slice_timer.timeout.connect(self.draw_slices)
        self.canvas_slices.installEventFilter(self)

    def eventFilter(self, obj, event):
        if obj is getattr(self, "canvas_slices", None) and event.type() == QtCore.QEvent.Type.Resize:
            w, h = event.size().width(), event.size().height()
            ow, oh = event.oldSize().width(), event.oldSize().height()
            if ow <= 0 or (h > 1.6 * w) != (oh > 1.6 * ow):        # layout changes: redraw (delayed)
                self._slice_timer.start()
        return super().eventFilter(obj, event)

    # ------------------------------------------------------------------ data
    def on_load(self):
        files, _ = QtWidgets.QFileDialog.getOpenFileNames(self, "Load curves", getattr(self, "_last_dir", ""),
                                                          "SAS curves (*.dat *.txt *.csv *.h5 *.nxs);;All files (*)")
        if files:
            self._last_dir = os.path.dirname(files[0])
            self.load_files(files)

    def load_files(self, files):
        from ..toolbox.merge import read_curves
        n0 = len(self.curves)
        for fn in files:
            try:
                for c in read_curves(fn):
                    ok = np.isfinite(c.q) & np.isfinite(c.I) & (c.q > 0)
                    if ok.sum() < 8:
                        continue
                    for k in ("q", "I", "dI", "dq"):
                        setattr(c, k, np.asarray(getattr(c, k))[ok])
                    self.add_curve(c)
            except Exception as e:  # noqa: BLE001
                self.log(f"cannot read {fn}: {e}")
        if len(self.curves) > n0:
            self.curve_list.setCurrentRow(n0)
        return len(self.curves) - n0

    def add_curve(self, c):
        self.curves.append(c)
        self.curve_list.addItem(self._cname(c))
        self._refresh_curve_combos()

    @staticmethod
    def _cname(c):
        return getattr(c, "name", None) or Path(getattr(c, "path", "curve")).stem

    def on_remove_curve(self):
        rows = sorted({i.row() for i in self.curve_list.selectedIndexes()}, reverse=True)
        for r in rows:
            name = self._cname(self.curves[r])
            for k in reversed(range(self.con_table.rowCount())):
                if self._con_curve_name(k) == name:
                    self.con_table.removeRow(k)
            self.curves.pop(r)
            self.curve_list.takeItem(r)
        self._refresh_curve_combos()
        self.update_matrix()

    def _on_curve_changed(self, row):
        if not 0 <= row < len(self.curves):
            self.curve_info.setText("no data")
            return
        c = self.curves[row]
        g = guess_d2o(self._cname(c))
        self.curve_info.setText(f"{len(c.q)} points, q = {c.q.min():.4g} … {c.q.max():.4g} 1/nm"
                                + (f"; D2O {100 * g:.0f} % (from the name)" if g is not None else ""))
        self.draw_iq()

    def curve_by_name(self, name):
        return next((c for c in self.curves if self._cname(c) == name), None)

    def load_example(self, kind):
        """Synthetic curves (1 % noise) with their components, to try the tool."""
        if kind == "dumbbell":
            comps, cons = M.dumbbell_curves()
            dmax = 7.0
        else:
            comps, cons = M.core_shell_curves()
            dmax = 7.0
        from ..toolbox.merge import Curve
        self.set_components(comps)
        for r in reversed(range(self.con_table.rowCount())):
            self.con_table.removeRow(r)
        for k in cons:
            name = f"{kind} {k.name}"
            c = self.curve_by_name(name)
            if c is None:
                c = Curve(k.q, k.I, k.dI, np.zeros_like(k.q), name, "columns")
                c._example_name = name
                self.add_curve(c)
            self.add_contrast(name, d2o=k.d2o)
        self.dmax.setValue(dmax)
        self.docks["contrasts"].raise_()
        self.log(f"example '{kind}': {len(cons)} synthetic contrasts (1 % noise), Dmax {dmax} nm")

    # ------------------------------------------------------------------ components
    def set_components(self, comps):
        t = self.comp_table
        t.blockSignals(True)
        t.setRowCount(0)
        for c in comps:
            self._add_component_row(c)
        t.blockSignals(False)
        self.update_matrix()

    def _add_component_row(self, c):
        t = self.comp_table
        i = t.rowCount()
        t.insertRow(i)
        t.setItem(i, 0, QtWidgets.QTableWidgetItem(c.name))
        vals = (c.sld_h2o, c.sld_h2o if c.sld_d2o is None else c.sld_d2o, c.exchange,
                c.sld_xray if np.isfinite(c.sld_xray) else 0.0)
        for j, (v, lo, hi, dec) in enumerate(zip(vals, (-10, -10, 0, -10), (20, 20, 1, 50), (3, 3, 3, 3)), start=1):
            sp = _spin(lo, hi, float(v), dec=dec, step=0.05 if j == 3 else 0.1)
            sp.valueChanged.connect(lambda _v: self.update_matrix())
            t.setCellWidget(i, j, sp)
        it = QtWidgets.QTableWidgetItem("")
        it.setFlags(it.flags() & ~QtCore.Qt.ItemFlag.ItemIsEditable)
        t.setItem(i, 5, it)
        if not getattr(self, "_comp_conn", False):
            t.itemChanged.connect(lambda _it: self.update_matrix())
            self._comp_conn = True

    def components(self):
        out = []
        t = self.comp_table
        for i in range(t.rowCount()):
            name = (t.item(i, 0).text() if t.item(i, 0) else "").strip() or f"component {i + 1}"
            h, d, x, xr = (t.cellWidget(i, j).value() for j in range(1, 5))
            out.append(M.Component(name, h, d, x, xr))
        names = [c.name for c in out]
        if len(set(names)) != len(names):
            raise M.MultiError("component names must be different")
        return out

    def on_add_component(self):
        name = self.preset.currentText()
        h, d, x = M.PRESETS[name]
        base = name.replace(" (typical)", "")
        names = {c.name for c in self._components_safe()}
        new, i = base, 2
        while new in names:
            new, i = f"{base} {i}", i + 1
        self._add_component_row(M.Component(new, h, d, 1.0, x))
        self.update_matrix()

    def on_remove_component(self):
        r = self.comp_table.currentRow()
        if r >= 0:
            self.comp_table.removeRow(r)
            self.update_matrix()

    def _components_safe(self):
        try:
            return self.components()
        except Exception:  # noqa: BLE001
            return []

    # ------------------------------------------------------------------ contrasts
    def on_add_contrasts(self):
        rows = sorted({i.row() for i in self.curve_list.selectedIndexes()})
        if not rows and self.curve_list.currentRow() >= 0:
            rows = [self.curve_list.currentRow()]
        for r in rows:
            c = self.curves[r]
            g = guess_d2o(self._cname(c))
            self.add_contrast(self._cname(c), d2o=0.0 if g is None else g)
            if g is None:
                self.log(f"{self._cname(c)}: D2O fraction not found in the name – set it in the Contrasts panel")
        self.docks["contrasts"].raise_()

    def add_contrast(self, curve_name, d2o=0.0, radiation="neutron", solvent=None, conc=1.0, weight=1.0, bg=0.0,
                     qmin=0.0, qmax=0.0):
        t = self.con_table
        i = t.rowCount()
        t.insertRow(i)
        cb = QtWidgets.QComboBox()
        cb.addItems([self._cname(c) for c in self.curves])
        cb.setCurrentText(curve_name)
        cb.currentIndexChanged.connect(lambda _i: (self.update_matrix(), self.draw_iq()))
        t.setCellWidget(i, 0, cb)
        rad = QtWidgets.QComboBox()
        rad.addItem("SANS", "neutron")
        rad.addItem("SAXS", "xray")
        rad.setCurrentIndex(1 if radiation == "xray" else 0)
        rad.currentIndexChanged.connect(lambda _i: self.update_matrix())
        t.setCellWidget(i, 1, rad)
        specs = [(2, 0, 100, 100 * d2o, 1), (4, 0, 1e6, conc, 4), (5, 0, 1e6, weight, 3), (6, -1e12, 1e12, bg, 6),
                 (7, 0, 1e4, qmin, 4), (8, 0, 1e4, qmax, 4)]
        for j, lo, hi, v, dec in specs:
            sp = _spin(lo, hi, float(v), dec=dec)
            if j in (7, 8):
                sp.setSpecialValueText("all")
            sp.valueChanged.connect(lambda _v: (self.update_matrix(), self.draw_iq()))
            t.setCellWidget(i, j, sp)
        it = QtWidgets.QTableWidgetItem("" if solvent is None else f"{solvent:.4g}")
        it.setToolTip("Empty: from the D2O fraction (SAXS: water). Type a value to override.")
        t.setItem(i, 3, it)
        if not getattr(self, "_con_conn", False):
            t.itemChanged.connect(lambda _it: self.update_matrix())
            self._con_conn = True
        self.update_matrix()
        self.draw_iq()

    def _con_curve_name(self, k):
        w = self.con_table.cellWidget(k, 0)
        return w.currentText() if w is not None else ""

    def _refresh_curve_combos(self):
        names = [self._cname(c) for c in self.curves]
        for k in range(self.con_table.rowCount()):
            cb = self.con_table.cellWidget(k, 0)
            cur = cb.currentText()
            cb.blockSignals(True)
            cb.clear()
            cb.addItems(names)
            cb.setCurrentText(cur)
            cb.blockSignals(False)

    def on_remove_contrast(self):
        r = self.con_table.currentRow()
        if r >= 0:
            self.con_table.removeRow(r)
            self.update_matrix()
            self.draw_iq()

    def contrast_rows(self):
        """[dict] of the contrast table (for sessions and the Python code)."""
        out = []
        t = self.con_table
        for k in range(t.rowCount()):
            it = t.item(k, 3)
            txt = (it.text() if it else "").strip()
            try:
                solvent = float(txt) if txt else None
            except ValueError:
                solvent = None
            out.append({"curve": self._con_curve_name(k), "radiation": t.cellWidget(k, 1).currentData(),
                        "d2o": t.cellWidget(k, 2).value() / 100, "solvent": solvent,
                        "conc": t.cellWidget(k, 4).value(), "weight": t.cellWidget(k, 5).value(),
                        "bg": t.cellWidget(k, 6).value(), "qmin": t.cellWidget(k, 7).value(),
                        "qmax": t.cellWidget(k, 8).value()})
        return out

    def contrasts(self):
        out = []
        for r in self.contrast_rows():
            c = self.curve_by_name(r["curve"])
            if c is None:
                raise M.MultiError(f"curve {r['curve']!r} not loaded")
            name = r["curve"]
            out.append(M.Contrast(name, c.q, c.I, np.asarray(c.dI, float), d2o=r["d2o"], radiation=r["radiation"],
                                  solvent_sld=r["solvent"], conc=r["conc"], weight=r["weight"], bg=r["bg"],
                                  qmin=r["qmin"] or None, qmax=r["qmax"] or None))
        names = [k.name for k in out]
        if len(set(names)) != len(names):
            raise M.MultiError("a curve is used in two contrasts")
        return out

    def fit_backgrounds(self):
        for k, r in enumerate(self.contrast_rows()):
            c = self.curve_by_name(r["curve"])
            if c is None:
                continue
            try:
                bg, dbg = M.fit_tail_background(c.q, c.I, np.asarray(c.dI, float))
            except Exception as e:  # noqa: BLE001
                self.log(f"✖ background of {r['curve']}: {e}")
                continue
            sig = abs(bg) > 2 * dbg
            self.log(f"{r['curve']}: tail background {bg:.4g} ± {dbg:.2g}" + ("" if sig else " (not significant, "
                                                                                 "set to 0)"))
            self.con_table.cellWidget(k, 6).setValue(bg if sig else 0.0)

    def update_matrix(self):
        """Contrast matrix, condition number, separability messages, match points."""
        if not hasattr(self, "matrix_label"):
            return
        h2o, d2o = self.sld_h2o.value(), self.sld_d2o.value()
        comps = self._components_safe()
        for i, c in enumerate(comps):                 # match point column
            it = self.comp_table.item(i, 5)
            if it is not None:
                mp = c.match_point(h2o, d2o)
                self.comp_table.blockSignals(True)
                it.setText(f"{100 * mp:.1f} % D2O" if np.isfinite(mp) else "")
                self.comp_table.blockSignals(False)
        try:
            cons = self.contrasts()
        except Exception as e:  # noqa: BLE001
            self.matrix_label.setText(f"<span style='color:#c00'>{e}</span>")
            return
        t = self.con_table
        for k, con in enumerate(cons):                # computed solvent SLDs as placeholders
            it = t.item(k, 3)
            if it is not None and not it.text().strip():
                it.setToolTip(f"{con.solvent(h2o, d2o):.4g} (from the D2O fraction). Type a value to override.")
        if not cons:
            self.matrix_label.setText("<i>no contrasts yet: load curves and add them as contrasts</i>")
            return
        if self.scheme.currentData() == "average":
            rs = [k.solvent(h2o, d2o) for k in cons]
            html = "<table cellspacing=4><tr><th>contrast</th><th>ρ_solvent</th></tr>"
            for k, r in zip(cons, rs):
                html += f"<tr><td>{k.name}</td><td align=right>{r:.3f}</td></tr>"
            html += "</table>"
            msg = ["Averaging: one SLD per voxel, the same in every contrast (component SLDs not used)."]
            if len(cons) < 2 or np.ptp(rs) <= 0:
                msg.append("<span style='color:#c00'>needs at least two different solvent SLDs</span>")
            self.matrix_label.setText(html + "<br>".join(msg))
            return
        if not comps:
            self.matrix_label.setText("<i>no components</i>")
            return
        try:
            C, rs, _rc = M.contrast_matrix(comps, cons, h2o, d2o)
        except Exception as e:  # noqa: BLE001
            self.matrix_label.setText(f"<span style='color:#c00'>{e}</span>")
            return
        w = [k.weight for k in cons]
        idf = M.identifiability(C, w, [c.name for c in comps])
        html = "<table cellspacing=4><tr><th>contrast</th><th>ρ_s</th>" + "".join(
            f"<th><span style='color:{PALETTE[i % len(PALETTE)]}'>{c.name}</span></th>"
            for i, c in enumerate(comps)) + "</tr>"
        for k, con in enumerate(cons):
            cells = "".join(f"<td align=right>{'<b>0 (matched)</b>' if k in idf['matched'][i] else f'{C[k, i]:.3f}'}"
                            "</td>" for i in range(len(comps)))
            html += f"<tr><td>{con.name}</td><td align=right>{rs[k]:.3f}</td>{cells}</tr>"
        html += "</table>"
        cond = idf["cond"]
        html += (f"rank {idf['rank']} of {len(comps)}, condition number "
                 f"{'∞' if not np.isfinite(cond) else f'{cond:.3g}'}; sensitivity "
                 + ", ".join(f"{c.name} {s:.3g}" for c, s in zip(comps, idf["sensitivity"])))
        if idf["messages"]:
            html += "<br>" + "<br>".join(f"<span style='color:#b36b00'>⚠ {m}</span>" for m in idf["messages"])
        else:
            html += "<br><span style='color:#2e7d32'>✔ the components can be separated with these contrasts</span>"
        self.matrix_label.setText(html)

    def estimate_dmax(self):
        try:
            k = self.contrasts()[0]
        except Exception as e:  # noqa: BLE001
            return self.log(f"✖ {e}")
        from ..density.core import guinier_rg
        q, I, e = k.data()
        try:
            rg = guinier_rg(q, I, e)
            rg = rg[0] if isinstance(rg, tuple) else rg
        except Exception as ex:  # noqa: BLE001
            return self.log(f"✖ Guinier fit of {k.name}: {ex}")
        if not np.isfinite(rg) or rg <= 0:
            return self.log(f"✖ no Guinier Rg for {k.name}")
        self.dmax.setValue(3.5 * rg)
        self.log(f"Dmax ≈ 3.5 Rg = {3.5 * rg:.3g} nm (Rg {rg:.3g} nm of {k.name}); check it with p(r)")

    # ------------------------------------------------------------------ run
    def settings(self):
        lo, hi = sorted((self.sld_lo.value(), self.sld_hi.value()))
        return M.Settings(dmax_nm=self.dmax.value(), scheme=self.scheme.currentData(),
                          oversampling=self.oversampling.value(), n=self.grid_n.value() or None,
                          max_steps=self.steps.value(), min_steps=min(1000, self.steps.value()),
                          shrinkwrap_threshold=self.threshold.value(), scale=self.scale_mode.currentData(),
                          conc_mg_ml=self.conc.value(), mw_kda=self.mw.value(),
                          sld_bounds=(lo, hi) if hi > lo else None, seed=self.seed.value() or None,
                          h2o=self.sld_h2o.value(), d2o=self.sld_d2o.value())

    def _run_dir(self, tag):
        base = str(_pref("densitymulti.workdir", "")).strip() or \
            os.path.join(os.path.dirname(prefs.settings_path()), "denssmulti_runs")
        stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S-%f")[:-3]
        d = os.path.join(base, f"{stamp}_{_safe(tag)}")
        os.makedirs(d, exist_ok=True)
        return d

    def _set_busy(self, busy):
        self._busy = busy
        self.run_button.setEnabled(not busy)
        self.selftest_button.setEnabled(not busy)
        self.stop_button.setEnabled(busy)
        self.progress.setRange(0, 0 if busy else 1)
        if not busy:
            self.progress_label.setText("")

    def _start(self, tag, fn, text):
        if self._busy:
            return
        self._cancel = False
        self._set_busy(True)
        self.log(f"--- {text}")
        self._status(text + " …", "run")

        def work():
            try:
                res, err = fn(), None
            except Exception as e:  # noqa: BLE001
                res, err = None, f"{type(e).__name__}: {e}"
                if not isinstance(e, M.MultiError):
                    err += "\n" + traceback.format_exc()
            self._signals.done.emit(tag, res, err)

        self._thread = threading.Thread(target=work, daemon=True)
        self._thread.start()

    def stop(self):
        self._cancel = True
        self.log("stop requested (after the current step)")

    def run(self):
        try:
            comps = self.components() if self.scheme.currentData() == "components" else []
            cons = self.contrasts()
            st = self.settings()
            if not cons:
                raise M.MultiError("no contrasts: load curves and add them as contrasts")
            M._prepare(comps, cons, st)                       # validates the input before the thread starts
        except Exception as e:  # noqa: BLE001
            self._status(f"✖ {e}", "error")
            return self.log(f"✖ {e}")
        n = self.nmaps.value()
        threads = self.threads.value()
        folder = self._run_dir(f"{st.scheme}_{len(cons)}contrasts")
        cancel = lambda: self._cancel  # noqa: E731

        def prog(step, chi):
            self._signals.progress.emit(f"step {step}, mean χ² (shells) {chi:.4g}")

        def prog_many(i, nn):
            self._signals.progress.emit(f"{i} of {nn} maps done")

        def fn():
            if n == 1:
                r = M.reconstruct(comps, cons, st, progress=prog, cancel=cancel)
                M.save_result(r, folder)
                return {"kind": "single", "result": r, "avg": None}
            rs = M.reconstruct_many(comps, cons, st, nmaps=n, cores=threads, progress=prog_many, cancel=cancel)
            a = M.align_and_average(rs)
            M.save_result(a, folder)
            for i, r in enumerate(rs):
                M.save_result(r, os.path.join(folder, f"map_{i + 1:02d}"))
            return {"kind": "average", "result": a.mean, "avg": a}
        self._start(("run", f"{st.scheme}"), fn, f"{'one map' if n == 1 else f'{n} maps'} from {len(cons)} contrasts "
                    f"({'component basis' if st.scheme == 'components' else 'averaging'})")

    def run_selftest(self):
        import io

        def fn():
            buf = io.StringIO()
            ok = M.selftest(out=lambda s: buf.write(s + "\n"))
            return {"text": buf.getvalue(), "ok": ok}
        self._start(("selftest", ""), fn, "pyDENSSmulti self test")

    def _on_done(self, tag, res, err):
        self._set_busy(False)
        kind = tag[0]
        if err:
            self.log(f"✖ {err}")
            self._status(f"✖ failed: {err.strip().splitlines()[0]}", "error")
            self._raise("log")
            return
        if kind == "selftest":
            self.log(res["text"])
            self._status("self test " + ("OK" if res["ok"] else "PROBLEM (see Log)"), "ok" if res["ok"] else "warning")
            return
        self.add_run(res["result"], res["kind"], res["avg"])

    def add_run(self, r, kind="single", avg=None):
        self.runs.append({"name": f"{len(self.runs) + 1}", "kind": kind, "result": r, "avg": avg})
        n = len(avg.runs) if avg is not None else 1
        self.log(f"run {len(self.runs)}: {r.scheme}, {n} map(s), grid {r.drho.shape[1]}³, voxel {r.voxel:.3g} Å, "
                 f"{r.steps} steps, {r.seconds:.1f} s\n  χ² " + ", ".join(f"{c}: {x:.3g}" for c, x in
                                                                       zip(r.contrast_names, r.chi2))
                 + "\n  volumes " + ", ".join(f"{c}: {v:.4g} Å³" for c, v in zip(r.component_names, r.volumes))
                 + (f"\n  match point of the particle (I(0)): solvent SLD {r.match_sld:.3g}"
                    if np.isfinite(r.match_sld) else "") + f"\n  files: {r.workdir}")
        if avg is not None:
            self.log("  alignment: correlation " + ", ".join(f"{c:.3f}" for c in avg.correlation)
                     + (f"; mirror images used for maps {[i + 1 for i, m in enumerate(avg.improper) if m]}"
                        if any(avg.improper) else ""))
        for w in r.warnings:
            self.log(f"⚠ {w}")
        self._fill_table()
        self.table.selectRow(len(self.runs) - 1)
        self._select_run(len(self.runs) - 1)
        self._status(f"run {len(self.runs)} done" + (f" – {r.warnings[0]}" if r.warnings else ""),
                     "warning" if r.warnings else "ok")

    def _status(self, text, level="ok"):
        colour = {"error": "#c00000", "warning": "#b36b00"}.get(level, "")
        self.statusBar().setStyleSheet(f"QStatusBar {{ color: {colour}; }}" if colour else "")
        self.statusBar().showMessage(text, 0 if (colour or level == "run") else 6000)

    def _raise(self, name):
        d = self.docks.get(name)
        if d is not None and not self.detached.get(name):
            d.raise_()

    # ------------------------------------------------------------------ output
    def log(self, text):
        self.log_view.appendPlainText(str(text))

    def _fill_table(self):
        t = self.table
        t.setRowCount(len(self.runs))
        for i, run in enumerate(self.runs):
            x = run["result"]
            a = run["avg"]
            vals = [run["name"], x.scheme, str(len(a.runs)) if a is not None else "1",
                    ", ".join(_fmt(v, 3) for v in x.chi2),
                    ", ".join(_fmt(v, 4) for v in x.volumes), _fmt(x.support_volume), str(x.steps),
                    _fmt(x.seconds, 3), str(x.workdir or "")]
            for j, v in enumerate(vals):
                it = QtWidgets.QTableWidgetItem(v)
                if j == 0 and x.warnings:
                    it.setText("⚠ " + v)
                    it.setToolTip("\n".join(x.warnings))
                if j == 3:
                    it.setToolTip("\n".join(f"{c}: χ² {v:.4g}" for c, v in zip(x.contrast_names, x.chi2)))
                if j == 4:
                    it.setToolTip("\n".join(f"{c}: {v:.4g} Å³" for c, v in zip(x.component_names, x.volumes)))
                t.setItem(i, j, it)

    def selected_run(self):
        r = self.table.currentRow()
        return self.runs[r] if 0 <= r < len(self.runs) else (self.runs[-1] if self.runs else None)

    def _select_run(self, _row):
        run = self.selected_run()
        self.slice_what.blockSignals(True)
        cur = self.slice_what.currentText()
        self.slice_what.clear()
        if run is not None:
            x = run["result"]
            for c in x.component_names:
                self.slice_what.addItem(f"φ {c}", ("phi", x.component_names.index(c)))
            if run["avg"] is not None:
                for c in x.component_names:
                    self.slice_what.addItem(f"spread of φ {c}", ("phi_sd", x.component_names.index(c)))
            if x.rho is not None:
                self.slice_what.addItem("SLD map", ("rho", 0))
            for k, c in enumerate(x.contrast_names):
                self.slice_what.addItem(f"Δρ {c}", ("drho", k))
            i = self.slice_what.findText(cur)
            self.slice_what.setCurrentIndex(max(i, 0))
        self.slice_what.blockSignals(False)
        self.update_surface()
        self.draw_iq()
        self.draw_conv()
        self.draw_slices()

    def update_surface(self):
        run = self.selected_run()
        if run is None:
            return
        x = run["result"]
        from ..density.core import isosurface
        lev = self.level.value() / 100
        self.view3d.clear()
        for c, name in enumerate(x.component_names):
            m = x.phi[c]
            if m.max() <= lev:
                self.log(f"{name}: no voxel above φ = {lev:.2f} (max {m.max():.3g})")
                continue
            try:
                v, f = isosurface(m, lev, x.voxel)
            except Exception as e:  # noqa: BLE001
                self.log(f"✖ isosurface of {name}: {e}")
                continue
            info = (f"run {run['name']}: {name}\nφ = {lev:.2f}; volume Σφ·V_voxel = {x.volumes[c]:.4g} Å³\n"
                    f"grid {m.shape[0]}³, voxel {x.voxel:.3g} Å")
            o = self.view3d.add_mesh(name, v, f, translucent=self.see_through.isChecked() and
                                     len(x.component_names) > 1 and c != int(np.argmin(x.volumes)), info=info)
            self._colour_object(o, PALETTE[c % len(PALETTE)])
        if self.show_support.isChecked() and x.support.any():
            sv, sf = isosurface(x.support.astype(float), 0.5, x.voxel)
            self.view3d.add_mesh("support", sv, sf, translucent=True, info="final support")

    def _colour_object(self, o, colour):
        """Give a 3D object the component's colour (as in the contrast matrix and the slices)."""
        if not isinstance(o, dict):
            return
        qc = QtGui.QColor(colour)
        o["colour"] = (qc.redF(), qc.greenF(), qc.blueF())
        i = self.view3d.objects.index(o)
        it = self.view3d.list.item(i)
        if it is not None:
            pm = QtGui.QPixmap(12, 12)
            pm.fill(qc)
            it.setIcon(QtGui.QIcon(pm))
        self.view3d.redraw()

    def draw_iq(self):
        fig = self.canvas_iq.figure
        fig.clear()
        gs = fig.add_gridspec(2, 1, height_ratios=[3, 1])
        ax = fig.add_subplot(gs[0])
        axr = fig.add_subplot(gs[1], sharex=ax)
        s = self.iq_scale.currentIndex()
        off = self.iq_offset.isChecked() and s in (0, 1)

        def tr(q, I):
            return (q, q ** 2 * I) if s == 2 else (q, I)
        run = self.selected_run()
        fits = {}
        if run is not None:
            x = run["result"]
            fits = {n: (f, x.chi2[k]) for k, (n, f) in enumerate(zip(x.contrast_names, x.fits))}
        rows = self.contrast_rows() if hasattr(self, "con_table") else []
        names = [r["curve"] for r in rows] or list(fits)
        for k, name in enumerate(names):
            col = f"C{k % 10}"
            fac = 10.0 ** k if off else 1.0
            if name in fits:
                f, chi = fits[name]
                ax.errorbar(*tr(f["q"], fac * f["I"]), yerr=fac * f["err"] * (f["q"] ** 2 if s == 2 else 1),
                            fmt="o", ms=2, color=col, alpha=0.45,
                            label=name + (f" ×10^{k}" if off and k else "") + f" (χ² {_fmt(chi, 3)})")
                ax.plot(*tr(f["q"], fac * f["model"]), "-", color=col, lw=1.4)
                with np.errstate(all="ignore"):
                    axr.plot(f["q"], (f["I"] - f["model"]) / f["err"], ".", ms=2, color=col)
            else:
                c = self.curve_by_name(name)
                if c is None:
                    continue
                bg = rows[k]["bg"] if k < len(rows) else 0.0
                ax.plot(*tr(c.q, fac * (c.I - bg)), "o", ms=2, color=col,
                        label=name + (f" ×10^{k}" if off and k else "") + (" − bg" if bg else ""))
        if not names and 0 <= self.curve_list.currentRow() < len(self.curves):
            c = self.curves[self.curve_list.currentRow()]
            ax.plot(*tr(c.q, c.I), "o", ms=2, color="0.5", label=self._cname(c))
        axr.axhline(0, color="0.5", lw=0.6)
        axr.set_xlabel("q (1/nm)")
        axr.set_ylabel("Δ/σ")
        ax.set_ylabel("q² I(q)" if s == 2 else "I(q)")
        if s == 0:
            ax.set_xscale("log")
            from matplotlib.ticker import NullFormatter
            for a_ in (ax, axr):
                a_.xaxis.set_minor_formatter(NullFormatter())
        if s in (0, 1):
            ax.set_yscale("log")
        ax.tick_params(labelbottom=False)
        if ax.lines:
            ax.legend(fontsize="x-small")
        self.canvas_iq.draw_idle()

    def draw_conv(self):
        fig = self.canvas_conv.figure
        fig.clear()
        run = self.selected_run()
        st = run["result"].stats if run is not None else None
        if st is None or not len(st["chi2"]):
            ax = fig.add_subplot(111)
            ax.text(0.5, 0.5, "convergence of a single map\n(averages: see the map_xx folders)", ha="center",
                    transform=ax.transAxes, color="0.5")
            ax.set_axis_off()
            self.canvas_conv.draw_idle()
            return
        axs = fig.subplots(2, 1, sharex=True)
        for k, name in enumerate(run["result"].contrast_names):
            axs[0].semilogy(st["chi2"][:, k], color=f"C{k % 10}", lw=1, label=name)
        axs[0].set_ylabel("χ² (shells)")
        axs[0].legend(fontsize="x-small")
        axs[1].semilogy(st["support_volume"], color="0.3")
        axs[1].set_ylabel("support V (Å³)")
        axs[1].set_xlabel("step")
        self.canvas_conv.draw_idle()

    def draw_slices(self):
        fig = self.canvas_slices.figure
        fig.clear()
        run = self.selected_run()
        what = self.slice_what.currentData()
        if run is None or what is None:
            self.canvas_slices.draw_idle()
            return
        x = run["result"]
        kind, i = what
        if kind == "phi":
            m, cmap, lo, hi, lab = x.phi[i], "viridis", 0.0, 1.0, "φ"
        elif kind == "phi_sd":
            m = run["avg"].phi_sd[i]
            cmap, lo, hi, lab = "magma", 0.0, max(float(np.nanmax(m)), 1e-6), "σ(φ)"
        elif kind == "rho":
            m = x.rho
            fin = m[np.isfinite(m)]
            lo, hi = (float(fin.min()), float(fin.max())) if fin.size else (0.0, 1.0)
            cmap, lab = "plasma", "SLD (1e-6 Å⁻²)"
        else:
            m = x.drho[i]
            v = float(np.max(np.abs(m))) or 1.0
            cmap, lo, hi, lab = "RdBu_r", -v, v, "Δρ (1e-6 Å⁻²)"
        n = np.array(m.shape) // 2
        w_px, h_px = self.canvas_slices.width(), self.canvas_slices.height()
        axs = fig.subplots(3, 1) if h_px > 1.6 * w_px else fig.subplots(1, 3)   # follow the shape of the panel
        half = m.shape[0] * x.voxel / 2
        dmax = 10 * float(x.settings.get("dmax_nm") or 0)
        zoom = min(half, 0.75 * dmax) if dmax > 0 else half       # the particle, not the whole (oversampled) box
        im = None
        for ax, sl, t in zip(axs, (m[n[0], :, :], m[:, n[1], :], m[:, :, n[2]]),
                             ("x = 0 (y, z)", "y = 0 (x, z)", "z = 0 (x, y)")):
            im = ax.imshow(sl.T, origin="lower", cmap=cmap, vmin=lo, vmax=hi, extent=[-half, half, -half, half])
            ax.set_xlim(-zoom, zoom)
            ax.set_ylim(-zoom, zoom)
            ax.set_title(t, fontsize="small")
            ax.set_xlabel("Å")
        fig.colorbar(im, ax=axs, shrink=0.8, label=lab)
        self.canvas_slices.draw_idle()

    # ------------------------------------------------------------------ files
    def on_open_result(self, folder=None):
        if not folder:
            folder = QtWidgets.QFileDialog.getExistingDirectory(self, "Result folder (with multi_summary.json)",
                                                                getattr(self, "_last_dir", ""))
            if not folder:
                return
        try:
            r = M.load_result(folder)
        except Exception as e:  # noqa: BLE001
            return self.log(f"✖ {e}")
        self.add_run(r, "single")

    def on_export_table(self):
        fn, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Results", "denssmulti_results.csv", "CSV (*.csv)")
        if not fn:
            return
        import csv
        with open(fn, "w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(RES_COLS)
            for i in range(self.table.rowCount()):
                w.writerow([self.table.item(i, j).text() if self.table.item(i, j) else ""
                            for j in range(len(RES_COLS))])

    def on_open_runs(self):
        run = self.selected_run()
        base = str(run["result"].workdir) if run and run["result"].workdir else (
            str(_pref("densitymulti.workdir", "")).strip()
            or os.path.join(os.path.dirname(prefs.settings_path()), "denssmulti_runs"))
        os.makedirs(base, exist_ok=True)
        QtGui.QDesktopServices.openUrl(QtCore.QUrl.fromLocalFile(base))

    def python_code(self):
        from ..core.pycode import Code, Script, call, lit, path_lit
        rows = self.contrast_rows()
        if not rows:
            raise ValueError("Add contrasts first.")
        st = self.settings()
        sc = Script(TOOL, f"3D map from {len(rows)} contrasts ({st.scheme})")
        sc.imp("import numpy as np", "from pysasfit import densitymulti as dm",
               "from pysasfit.toolbox.merge import read_curves")
        if st.scheme == "components":
            sc.comment("components: SLD in H2O, in D2O (full exchange of labile H), exchanged fraction, X-ray SLD "
                       "(1e-6 Å⁻²)")
            sc.add("components = [")
            for c in self.components():
                sc.add(f"    dm.Component({c.name!r}, {c.sld_h2o!r}, {c.sld_d2o!r}, {c.exchange!r}, {c.sld_xray!r}),")
            sc.add("]")
        else:
            sc.add("components = []   # the averaging scheme uses one SLD map")
        sc.comment("contrasts: one curve each (q in 1/nm); flat background bg subtracted")
        sc.add("contrasts = []")
        for r in rows:
            c = self.curve_by_name(r["curve"])
            path = getattr(c, "path", "")
            ex = getattr(c, "_example_name", None)
            if ex:                                   # synthetic example curve
                kind, label = ex.split(" ", 1)
                fn = "dm.dumbbell_curves()" if kind == "dumbbell" else "dm.core_shell_curves()"
                sc.add(f"c = next(k for k in {fn}[1] if k.name == {label!r})   # synthetic example")
            elif path and os.path.isfile(str(path)):
                sc.add(f"c = next(c for c in read_curves({path_lit(path)}) if c.name == {r['curve']!r})")
            else:
                sc.add(f"c = ...   # {r['curve']!r}: give q (1/nm), I, dI here")
            kw = {"d2o": r["d2o"], "radiation": r["radiation"], "solvent_sld": r["solvent"], "conc": r["conc"],
                  "weight": r["weight"], "bg": r["bg"], "qmin": r["qmin"] or None, "qmax": r["qmax"] or None}
            kw = {k: v for k, v in kw.items() if v is not None and not (k in ("conc", "weight") and v == 1.0)
                  and not (k == "bg" and v == 0.0) and not (k == "radiation" and v == "neutron")}
            sc.add(call("contrasts.append", Code(call("dm.Contrast", r["curve"], Code("c.q"), Code("c.I"),
                                                      Code("c.dI"), **kw))))
        defaults = M.Settings().to_dict()
        kw = {k: v for k, v in st.to_dict().items() if defaults.get(k) != v}
        if st.scale != "absolute":
            kw.pop("conc_mg_ml", None)
            kw.pop("mw_kda", None)
        sc.add(f"settings = dm.Settings(**{lit(kw)})")
        sc.add("print(dm.identifiability(*dm.contrast_matrix(components, contrasts, settings.h2o, "
               "settings.d2o)[:1])['messages'])" if st.scheme == "components" else "")
        if self.nmaps.value() > 1:
            sc.comment(f"{self.nmaps.value()} maps, aligned and averaged")
            sc.add(call("maps = dm.reconstruct_many", Code("components"), Code("contrasts"), Code("settings"),
                        nmaps=self.nmaps.value(), cores=self.threads.value()),
                   "avg = dm.align_and_average(maps)", "r = avg.mean")
        else:
            sc.add("r = dm.reconstruct(components, contrasts, settings)")
        sc.add("for name, chi in zip(r.contrast_names, r.chi2):", "    print(f'{name}: chi2 {chi:.3g}')",
               "for name, v in zip(r.component_names, r.volumes):", "    print(f'{name}: volume {v:.4g} A^3')",
               "# dm.save_result(r, 'my_run')   # MRC maps, fits, summary")
        sc.add_plot("fig, axs = plt.subplots(1, 2, figsize=(11, 4.5))",
                    "for k, f in enumerate(r.fits):",
                    "    axs[0].loglog(f['q'], 10.0 ** k * f['I'], 'o', ms=2, color=f'C{k}', label=r.contrast_names[k])",
                    "    axs[0].loglog(f['q'], 10.0 ** k * f['model'], '-', color=f'C{k}')",
                    "axs[0].set_xlabel('q (1/nm)'); axs[0].legend(fontsize='small')",
                    "n = r.phi.shape[1] // 2",
                    "axs[1].imshow(np.dstack([r.phi[c][:, :, n].T for c in range(min(3, len(r.phi)))]"
                    " + [np.zeros_like(r.phi[0][:, :, n].T)] * (3 - min(3, len(r.phi)))), origin='lower')",
                    "axs[1].set_title('central slice: components in R, G, B')")
        return sc.text()

    # ------------------------------------------------------------------ session
    @staticmethod
    def _session_path():
        return os.path.join(os.path.dirname(prefs.settings_path()), "densitymulti_session.json")

    def session_state(self):
        files, examples = [], []
        for c in self.curves:
            p = getattr(c, "path", "")
            if getattr(c, "_example_name", None):
                if c._example_name.split(" ")[0] not in examples:
                    examples.append(c._example_name.split(" ")[0])
            elif p and p not in files:
                files.append(str(p))
        return {"version": 1, "files": files, "examples": examples,
                "components": [c.to_dict() for c in self._components_safe()],
                "solvent": [self.sld_h2o.value(), self.sld_d2o.value()],
                "contrasts": self.contrast_rows(),
                "options": {"scheme": self.scheme.currentIndex(), "dmax": self.dmax.value(),
                            "n": self.grid_n.value(), "os": self.oversampling.value(), "steps": self.steps.value(),
                            "threshold": self.threshold.value(), "seed": self.seed.value(),
                            "scale": self.scale_mode.currentIndex(), "conc": self.conc.value(), "mw": self.mw.value(),
                            "sld": [self.sld_lo.value(), self.sld_hi.value()], "nmaps": self.nmaps.value(),
                            "threads": self.threads.value(), "level": self.level.value()},
                "runs": [str(r["result"].workdir) for r in self.runs if r["result"].workdir],
                "guides": self.view3d.guide_state(), "dock_state": self.dock_state()}

    def save_session(self):
        try:
            st = self.session_state()
            p = self._session_path()
            os.makedirs(os.path.dirname(p), exist_ok=True)
            with open(p, "w", encoding="utf-8") as fh:
                json.dump(st, fh, indent=1)
        except (OSError, TypeError, ValueError):
            pass

    def apply_state(self, st):
        for f in st.get("files") or []:
            if os.path.isfile(f):
                self.load_files([f])
        for ex in st.get("examples") or []:
            from ..toolbox.merge import Curve
            comps, cons = (M.dumbbell_curves() if ex == "dumbbell" else M.core_shell_curves())
            for k in cons:
                name = f"{ex} {k.name}"
                if self.curve_by_name(name) is None:
                    c = Curve(k.q, k.I, k.dI, np.zeros_like(k.q), name, "columns")
                    c._example_name = name
                    self.add_curve(c)
        if st.get("components") is not None:
            self.set_components([M.Component.from_dict(d) for d in st["components"]])
        sv = st.get("solvent")
        if sv:
            self.sld_h2o.setValue(sv[0])
            self.sld_d2o.setValue(sv[1])
        for r in reversed(range(self.con_table.rowCount())):
            self.con_table.removeRow(r)
        for r in st.get("contrasts") or []:
            if self.curve_by_name(r["curve"]) is not None:
                self.add_contrast(r["curve"], r.get("d2o", 0.0), r.get("radiation", "neutron"), r.get("solvent"),
                                  r.get("conc", 1.0), r.get("weight", 1.0), r.get("bg", 0.0), r.get("qmin", 0.0),
                                  r.get("qmax", 0.0))
        o = st.get("options") or {}
        self.scheme.setCurrentIndex(int(o.get("scheme", 0)))
        self.dmax.setValue(float(o.get("dmax", 10.0)))
        self.grid_n.setValue(int(o.get("n", 0)))
        self.oversampling.setValue(float(o.get("os", 3.0)))
        self.steps.setValue(int(o.get("steps", 3000)))
        self.threshold.setValue(float(o.get("threshold", 0.2)))
        self.seed.setValue(int(o.get("seed", 0)))
        self.scale_mode.setCurrentIndex(int(o.get("scale", 0)))
        self.conc.setValue(float(o.get("conc", 5.0)))
        self.mw.setValue(float(o.get("mw", 100.0)))
        lo_hi = o.get("sld", [0.0, 0.0])
        self.sld_lo.setValue(float(lo_hi[0]))
        self.sld_hi.setValue(float(lo_hi[1]))
        self.nmaps.setValue(int(o.get("nmaps", 1)))
        self.threads.setValue(int(o.get("threads", self.threads.value())))
        self.level.setValue(int(o.get("level", 50)))
        for d in st.get("runs") or []:
            if os.path.isfile(os.path.join(d, "multi_summary.json")):
                try:
                    self.add_run(M.load_result(d), "single")
                except Exception as e:  # noqa: BLE001
                    self.log(f"run folder {d}: {e}")
        self.view3d.set_guide_state(st.get("guides"))
        self.update_matrix()

    def restore_session(self):
        try:
            with open(self._session_path(), encoding="utf-8") as fh:
                st = json.load(fh)
        except (OSError, ValueError):
            return False
        try:
            self.apply_state(st)
            self._session_layout = bool(self.restore_dock_state(st.get("dock_state")))
        except Exception:  # noqa: BLE001
            pass
        return True

    def forget_session(self):
        try:
            os.remove(self._session_path())
        except OSError:
            pass

    def shutdown(self):
        self._cancel = True
        self.save_session()

    def closeEvent(self, event):
        self.shutdown()
        super().closeEvent(event)


def main(argv=None):
    argv = sys.argv if argv is None else argv
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(argv)
    from ..core.tabstyle import apply_tab_style
    apply_tab_style(app)
    app.setWindowIcon(tool_icon(TOOL))
    w = MainWindow()
    w.show()
    files = [a for a in argv[1:] if os.path.isfile(a)]
    if files:
        w.load_files(files)
    else:
        w.restore_session()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
