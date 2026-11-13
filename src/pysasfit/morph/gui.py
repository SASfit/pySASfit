"""pySASmorph window: representative 3D two- and multi-phase morphologies from SAS data.

    pysasfit-morph [datafile ...]
    python -m pysasfit.morph.gui [datafile ...]

Also the "pySASmorph" tab of the pySASfit launcher. Thin layer over
:mod:`pysasfit.morph.multiphase` (which contains the two-phase SAXSMorph case): every
number comes from the library.

Input: one or several curves of the same sample (a contrast series: SANS contrast variation,
SAXS + SANS) and the phases in their order along the Gaussian field, with volume fractions and
one SLD per phase and curve. Two phases and one curve is SAXSMorph.

Panels (View menu; each can be moved, stacked or detached): Input, I(q), Correlation
functions, Spectral function, Structure (slices through the voxel matrix), 3D and Results.
*Correlations* runs the fast steps (extrapolation, gamma_m(r), common g(r), f(k), model I_m(q));
*Reconstruct* also generates the random field, the phases, their scattering and statistics.
"""

from __future__ import annotations

import json
import os
import sys
import traceback

import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg  # noqa: E402
from matplotlib.colors import ListedColormap  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

from ..core import settings as prefs  # noqa: E402
from ..core.docking import DockMixin, plot_panel  # noqa: E402
from ..resources import tool_icon  # noqa: E402
from . import multiphase  # noqa: E402
from .export import MATRIX_FORMATS, write_curves_multi, write_matrix  # noqa: E402
from .io import load_curve  # noqa: E402
from .multiphase import DataSet, MultiParameters  # noqa: E402
from .view3d_gl import phase_colour  # noqa: E402

__all__ = ["MainWindow", "main"]

Dock = QtCore.Qt.DockWidgetArea

HELP = {
    "phases": "Phases in their ORDER along the Gaussian field ψ: phase i lies between the levels α(i−1) and α(i), "
              "so it touches only its neighbours in the list; a middle phase forms a layer (film, shell) "
              "between them. Fractions must add up to 1 (they are normalised). One SLD column per curve; "
              "only SLD differences matter, absolute values only for the invariant check.",
    "fit": "Fit the volume fractions to the curves. Needs a contrast series on one intensity scale or "
           "absolute data with absolute SLDs; the fractions are only weakly determined (± a few hundredths) – "
           "use known compositions when you have them.",
    "absolute": "SLDs in 10⁻⁶ Å⁻² (= 10¹⁰ cm⁻²) and I(q) in 1/cm: the invariants are compared with "
                "2π² ⟨Δρ²⟩ (calibration, fraction and SLD check).",
    "box": "Edge length L of the cube, in the inverse unit of q. Empty: about 5 x 2 pi / q* "
           "(q* = maximum of the Kratky plot). Rule of thumb: several correlation lengths.",
    "n": "Voxels per edge. Memory and time grow as n^3; the voxel size L/n should resolve pi/q_max and "
         "the thinnest phase (film).",
    "method": "fft: periodic spectral synthesis on the voxel grid (fast, exact grid statistics). "
              "cosine: Berk's random waves as in SAXSMorph (not periodic, slower).",
    "threshold": "quantile: the voxel fractions equal the input exactly. alpha: cut at the "
                 "theoretical levels (fractions fluctuate).",
    "low_q": "Extrapolation to q = 0 from the first points: Guinier, linear, constant or none.",
    "porod": "Range of the Porod law I = K q^-4 (+ B), extrapolated analytically to infinity.\n"
             "automatic: the longest high-q range on which the law still describes the data (chi² at most 1.5 "
             "times the best, or ≤ 2 with errors); the chosen range is shown in the I(q) and Porod plots and "
             "in the report.\nlast N points / q ≥ q_min: set by hand.\nlast 15 %: the former default.",
    "fit_bg": "Fit a flat (incoherent) background B together with the Porod law for every curve that is "
              "loaded; per curve it can be switched off (column 'fit B') or fixed (column 'B').",
}

CURVE_COLOURS = ["C0", "C3", "C2", "C4", "C1", "C5", "C6", "C8", "C9"]


def _float_or_none(text):
    text = text.strip()
    if not text:
        return None
    return float(text)


def _item(text, editable=True, align_right=False):
    it = QtWidgets.QTableWidgetItem(str(text))
    if not editable:
        it.setFlags(it.flags() & ~QtCore.Qt.ItemFlag.ItemIsEditable)
    if align_right:
        it.setTextAlignment(int(QtCore.Qt.AlignmentFlag.AlignRight | QtCore.Qt.AlignmentFlag.AlignVCenter))
    return it


class MainWindow(DockMixin, QtWidgets.QMainWindow):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("pySASmorph – 3D two- and multi-phase morphology from SAS data")
        self.setWindowIcon(tool_icon("pySASmorph"))
        self.datasets = []          # dicts: name, path, q, I, sigma, weight
        self.result = None
        self._building = False
        self._build_menus()
        self._build_input()
        self._build_plots()
        self.results_text = QtWidgets.QPlainTextEdit()
        self.results_text.setReadOnly(True)
        self.results_text.setLineWrapMode(QtWidgets.QPlainTextEdit.LineWrapMode.NoWrap)
        self.results_text.setFont(QtGui.QFontDatabase.systemFont(QtGui.QFontDatabase.SystemFont.FixedFont))
        self._add_dock("input", "Input", self.input_panel, Dock.LeftDockWidgetArea, scrollable=True)
        self._add_dock("iq", "I(q)", self.iq_panel, Dock.RightDockWidgetArea)
        self._add_dock("corr", "Correlation functions", self.corr_panel, Dock.RightDockWidgetArea)
        self._add_dock("spec", "Spectral function", self.spec_panel, Dock.RightDockWidgetArea)
        self._add_dock("struct", "Structure", self.struct_panel, Dock.RightDockWidgetArea)
        self._add_dock("results", "Results", self.results_text, Dock.RightDockWidgetArea)
        from .view3d_gl import MorphGLView
        self.view3d = MorphGLView()
        self.view3d.status.connect(lambda t: self.statusBar().showMessage(t, 8000))
        self._add_dock("view3d", "3D", self.view3d, Dock.RightDockWidgetArea)
        self.splitDockWidget(self.docks["iq"], self.docks["corr"], QtCore.Qt.Orientation.Horizontal)
        self.splitDockWidget(self.docks["iq"], self.docks["struct"], QtCore.Qt.Orientation.Vertical)
        self.splitDockWidget(self.docks["corr"], self.docks["results"], QtCore.Qt.Orientation.Vertical)
        self.tabifyDockWidget(self.docks["corr"], self.docks["spec"])
        self.tabifyDockWidget(self.docks["struct"], self.docks["view3d"])
        self.docks["struct"].raise_()
        self.docks["corr"].raise_()
        self._finish_docks()
        self.hide_stray_tab_bars()       # Qt leaves an empty tab bar after tabifying at start
        self.setDockNestingEnabled(True)
        self.resize(1450, 900)
        self.set_phases(["solid", "pore"], [0.5, 0.5])
        self._update_enabled()

    # ------------------------------------------------------------------ construction
    def _build_menus(self):
        m = self.menuBar().addMenu("&File")
        a = m.addAction("&Open data…", self.on_open)
        a.setShortcut(QtGui.QKeySequence.StandardKey.Open)
        m.addAction("&Add curve (contrast)…", self.on_add)
        m.addSeparator()
        self.act_export_curves = m.addAction("Export &curves…", self.on_export_curves)
        self.act_export_matrix = m.addAction("Export voxel &matrix…", self.on_export_matrix)
        self.act_export_stl = m.addAction("Export surface (STL)…", self.on_export_stl)
        m.addSeparator()
        m.addAction("Anisotropic 2D data (&texture)…", self.on_anisotropic)
        from ..core import pycode
        pycode.install(self, self.python_code, menu=m)
        from ..core.settings_dialog import add_preferences_menu
        add_preferences_menu(self, None)
        self.menuBar().addMenu("&View")

    def _build_input(self):
        w = QtWidgets.QWidget()
        lay = QtWidgets.QVBoxLayout(w)
        # ---- data
        g = QtWidgets.QGroupBox("Data")
        g.setToolTip("Several curves of the same sample = a contrast series (SANS contrast variation, SAXS + SANS): "
                     "one common g(r), one SLD column per curve")
        v = QtWidgets.QVBoxLayout(g)
        self.curve_table = QtWidgets.QTableWidget(0, 5)
        self.curve_table.setHorizontalHeaderLabels(["curve", "points", "weight", "fit B", "B"])
        for col, tip in ((3, "Fit a flat background B together with the Porod law (I = K/q⁴ + B) on the Porod "
                             "range; the fitted value is shown in column B"),
                         (4, "Flat background B subtracted from the curve. Fitted (fit B ticked) or set by hand: "
                             "typing a value (or using the B box above the I(q) plot) fixes it")):
            self.curve_table.horizontalHeaderItem(col).setToolTip(tip)
        hh = self.curve_table.horizontalHeader()
        hh.setSectionResizeMode(QtWidgets.QHeaderView.ResizeMode.ResizeToContents)
        hh.setSectionResizeMode(0, QtWidgets.QHeaderView.ResizeMode.Interactive)
        hh.setSectionResizeMode(4, QtWidgets.QHeaderView.ResizeMode.Stretch)
        self.curve_table.setColumnWidth(0, 64)
        self.curve_table.verticalHeader().setVisible(False)
        self.curve_table.setMaximumHeight(110)
        self.curve_table.setMinimumWidth(80)
        self.curve_table.setMinimumHeight(80)
        self.curve_table.setToolTip("Name (editable, used as SLD column header), number of points, weight in "
                                    "the joint fit of g(r), flat background B (fitted or fixed)")
        self.curve_table.currentCellChanged.connect(lambda *_a: self._sync_b_box())
        self.curve_table.itemChanged.connect(self._on_curve_item)
        v.addWidget(self.curve_table)
        row = QtWidgets.QHBoxLayout()
        for text, slot, tip in (("Open…", self.on_open, "Replace all curves by one file"),
                                ("Add…", self.on_add, "Add curves of the same sample (other contrasts)"),
                                ("Remove", self.on_remove, "Remove the selected curve")):
            b = QtWidgets.QPushButton(text)
            b.setToolTip(tip)
            b.clicked.connect(slot)
            row.addWidget(b)
        v.addLayout(row)
        f = QtWidgets.QFormLayout()
        self.qmin = QtWidgets.QLineEdit()
        self.qmax = QtWidgets.QLineEdit()
        for e in (self.qmin, self.qmax):
            e.setPlaceholderText("all")
            e.setValidator(QtGui.QDoubleValidator(0, 1e9, 10))
        f.addRow("q min", self.qmin)
        f.addRow("q max", self.qmax)
        self.unit = QtWidgets.QComboBox()
        self.unit.addItems(["1/Å", "1/nm"])
        self.unit.setToolTip("Axis labels and absolute invariant: lengths are in the inverse unit of q")
        f.addRow("q unit", self.unit)
        v.addLayout(f)
        lay.addWidget(g)
        # ---- phases
        g = QtWidgets.QGroupBox("Phases (in order along ψ) and SLDs")
        v = QtWidgets.QVBoxLayout(g)
        self.phase_table = QtWidgets.QTableWidget(0, 2)
        self.phase_table.verticalHeader().setVisible(False)
        self.phase_table.setToolTip(HELP["phases"])
        self.phase_table.setMaximumHeight(150)
        self.phase_table.setMinimumWidth(80)
        self.phase_table.setMinimumHeight(120)
        self.phase_table.itemChanged.connect(self._on_phase_item)
        v.addWidget(self.phase_table)
        row = QtWidgets.QHBoxLayout()
        for text, slot, tip in (("+", self.on_add_phase, "Add a phase"), ("−", self.on_remove_phase,
                                                                          "Remove the selected phase"),
                                ("▲", lambda: self.on_move_phase(-1), "Move up (lower ψ)"),
                                ("▼", lambda: self.on_move_phase(1), "Move down (higher ψ)")):
            b = QtWidgets.QToolButton()
            b.setText(text)
            b.setToolTip(tip)
            b.clicked.connect(slot)
            row.addWidget(b)
        self.frac_label = QtWidgets.QLabel("")
        row.addWidget(self.frac_label, 1)
        v.addLayout(row)
        self.fit_fractions = QtWidgets.QCheckBox("fit volume fractions")
        self.fit_fractions.setToolTip(HELP["fit"])
        v.addWidget(self.fit_fractions)
        self.sld_absolute = QtWidgets.QCheckBox("absolute: SLD in 10⁻⁶ Å⁻², I in 1/cm")
        self.sld_absolute.setToolTip(HELP["absolute"])
        v.addWidget(self.sld_absolute)
        lay.addWidget(g)
        # ---- extrapolation
        g = QtWidgets.QGroupBox("Extrapolation (all curves)")
        f = QtWidgets.QFormLayout(g)
        self.low_q = QtWidgets.QComboBox()
        self.low_q.addItems(["guinier", "linear", "constant", "none"])
        self.low_q.setToolTip(HELP["low_q"])
        f.addRow("low q", self.low_q)
        self.low_q_points = QtWidgets.QSpinBox()
        self.low_q_points.setRange(3, 1000)
        self.low_q_points.setValue(10)
        f.addRow("low-q points", self.low_q_points)
        prow = QtWidgets.QHBoxLayout()
        self.porod_mode = QtWidgets.QComboBox()
        for text, key in (("automatic", "auto"), ("last N points", "points"), ("q ≥ q_min", "qmin"),
                          ("last 15 %", "fixed15")):
            self.porod_mode.addItem(text, key)
        self.porod_mode.setToolTip(HELP["porod"])
        self.porod_points = QtWidgets.QSpinBox()
        self.porod_points.setRange(3, 10000)
        self.porod_points.setValue(40)
        self.porod_points.setToolTip("number of last points (Porod range 'last N points')")
        self.porod_qmin = QtWidgets.QLineEdit()
        self.porod_qmin.setPlaceholderText("q_min")
        self.porod_qmin.setToolTip("start of the Porod range (Porod range 'q ≥ q_min')")
        prow.addWidget(self.porod_mode, 1)
        prow.addWidget(self.porod_points)
        prow.addWidget(self.porod_qmin)
        f.addRow("Porod range", prow)
        self.porod_mode.currentIndexChanged.connect(self._update_porod_widgets)
        self.fit_bg_default = QtWidgets.QCheckBox("fit background B for new curves")
        self.fit_bg_default.setChecked(True)
        self.fit_bg_default.setToolTip(HELP["fit_bg"])
        f.addRow("", self.fit_bg_default)
        self._update_porod_widgets()
        lay.addWidget(g)
        # ---- morphology
        g = QtWidgets.QGroupBox("Morphology")
        f = QtWidgets.QFormLayout(g)
        row = QtWidgets.QHBoxLayout()
        self.box = QtWidgets.QLineEdit()
        self.box.setPlaceholderText("auto")
        self.box.setValidator(QtGui.QDoubleValidator(0, 1e12, 6))
        self.box.setToolTip(HELP["box"])
        row.addWidget(self.box, 1)
        b = QtWidgets.QPushButton("suggest")
        b.clicked.connect(self.on_suggest_box)
        row.addWidget(b)
        f.addRow("box size L", row)
        self.n = QtWidgets.QSpinBox()
        self.n.setRange(16, 512)
        self.n.setSingleStep(16)
        self.n.setValue(128)
        self.n.setToolTip(HELP["n"])
        f.addRow("voxels per edge", self.n)
        self.method = QtWidgets.QComboBox()
        self.method.addItems(["fft", "cosine"])
        self.method.setToolTip(HELP["method"])
        f.addRow("field", self.method)
        self.n_waves = QtWidgets.QSpinBox()
        self.n_waves.setRange(100, 100000)
        self.n_waves.setSingleStep(500)
        self.n_waves.setValue(4000)
        f.addRow("random waves", self.n_waves)
        self.threshold = QtWidgets.QComboBox()
        self.threshold.addItems(["quantile", "alpha"])
        self.threshold.setToolTip(HELP["threshold"])
        f.addRow("level cut", self.threshold)
        self.seed = QtWidgets.QSpinBox()
        self.seed.setRange(-1, 2 ** 31 - 1)
        self.seed.setValue(-1)
        self.seed.setSpecialValueText("random")
        f.addRow("seed", self.seed)
        lay.addWidget(g)
        # ---- run
        row = QtWidgets.QHBoxLayout()
        self.btn_corr = QtWidgets.QPushButton("Correlations")
        self.btn_corr.setToolTip("Extrapolation, γ(r) of every curve, common g(r), f(k) and the model I(q) (fast)")
        self.btn_corr.clicked.connect(self.on_correlations)
        self.btn_run = QtWidgets.QPushButton("Reconstruct")
        self.btn_run.setToolTip("Also the random field, the phases, their scattering and statistics")
        self.btn_run.clicked.connect(self.on_reconstruct)
        f2 = self.btn_run.font()
        f2.setBold(True)
        self.btn_run.setFont(f2)
        row.addWidget(self.btn_corr)
        row.addWidget(self.btn_run)
        from ..core import pycode
        row.addWidget(pycode.button(self.python_code, w))
        lay.addLayout(row)
        self.progress = QtWidgets.QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setTextVisible(True)
        self.progress.setFormat("")
        lay.addWidget(self.progress)
        # ---- 3D
        g = QtWidgets.QGroupBox("3D")
        v = QtWidgets.QVBoxLayout(g)
        b = QtWidgets.QPushButton("Show 3D panel")
        b.setToolTip("Interface surfaces in the '3D' panel (pyqtgraph / OpenGL); "
                     "phase, surface, largest cluster and clipping are set there")
        b.clicked.connect(self.show_3d_panel)
        v.addWidget(b)
        self.btn_3d = QtWidgets.QPushButton("Open in PyVista window")
        self.btn_3d.setToolTip("Optional: the phase selected in the 3D panel in a separate PyVista/VTK window")
        self.btn_3d.clicked.connect(self.on_view3d)
        v.addWidget(self.btn_3d)
        lay.addWidget(g)
        lay.addStretch(1)
        self.input_panel = w
        self.method.currentTextChanged.connect(lambda t: self.n_waves.setEnabled(t == "cosine"))
        self.n_waves.setEnabled(False)

    def _build_plots(self):
        self.fig_iq = Figure(figsize=(5, 4), layout="constrained")
        self.canvas_iq = FigureCanvasQTAgg(self.fig_iq)
        self.ax_iq = self.fig_iq.add_subplot(111)
        self.iq_mode = QtWidgets.QComboBox()
        self.iq_mode.addItem("I(q)", "loglog")
        self.iq_mode.addItem("Porod: I·q⁴/K vs q⁴", "porod")
        self.iq_mode.setToolTip("Porod plot: a flat background B is a straight line I q⁴/K = 1 + (B/K) q⁴; after "
                                "subtracting the right B the high-q points lie on the horizontal line 1")
        self.iq_mode.currentIndexChanged.connect(lambda _i: self._plot_iq())
        self.offset_curves = QtWidgets.QCheckBox("offset (× 10ᵐ)")
        self.offset_curves.setChecked(True)
        self.offset_curves.toggled.connect(lambda _c: self._plot_iq())
        self.b_label = QtWidgets.QLabel("B:")
        self.b_box = QtWidgets.QDoubleSpinBox()
        self.b_box.setDecimals(8)
        self.b_box.setRange(-1e12, 1e12)
        self.b_box.setStepType(QtWidgets.QAbstractSpinBox.StepType.AdaptiveDecimalStepType)
        self.b_box.setKeyboardTracking(False)
        self.b_box.setToolTip("Flat background of the curve selected in the Data table: arrows / wheel change it in "
                              "small steps, the plots follow; a value set here is fixed (fit B is switched off)")
        self.b_box.valueChanged.connect(self._on_b_box)
        self.b_fit = QtWidgets.QToolButton()
        self.b_fit.setText("fit")
        self.b_fit.setToolTip("Fit B again for the selected curve")
        self.b_fit.clicked.connect(self._on_b_fit)
        self._b_timer = QtCore.QTimer(self, singleShot=True, interval=350)
        self._b_timer.timeout.connect(self.on_correlations)
        bar = QtWidgets.QHBoxLayout()
        bar.addWidget(self.iq_mode)
        bar.addWidget(self.offset_curves)
        bar.addStretch(1)
        bar.addWidget(self.b_label)
        bar.addWidget(self.b_box)
        bar.addWidget(self.b_fit)
        self.iq_panel = plot_panel(self.canvas_iq, self, bars=(bar,))
        self.fig_corr = Figure(figsize=(5, 4), layout="constrained")
        self.canvas_corr = FigureCanvasQTAgg(self.fig_corr)
        self.ax_corr = self.fig_corr.add_subplot(111)
        self.corr_panel = plot_panel(self.canvas_corr, self)
        self.fig_spec = Figure(figsize=(5, 4), layout="constrained")
        self.canvas_spec = FigureCanvasQTAgg(self.fig_spec)
        self.ax_spec = self.fig_spec.add_subplot(111)
        self.spec_panel = plot_panel(self.canvas_spec, self)
        self.fig_struct = Figure(figsize=(4, 4), layout="constrained")
        self.canvas_struct = FigureCanvasQTAgg(self.fig_struct)
        self.ax_struct = self.fig_struct.add_subplot(111)
        bar = QtWidgets.QHBoxLayout()
        self.slice_axis = QtWidgets.QComboBox()
        self.slice_axis.addItems(["z", "y", "x"])
        self.slice_pos = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.slice_pos.setRange(0, 0)
        self.slice_label = QtWidgets.QLabel("")
        self.show_field = QtWidgets.QCheckBox("field ψ")
        self.show_field.setToolTip("Show the Gaussian random field with the cut levels instead of the phases")
        bar.addWidget(QtWidgets.QLabel("slice ⟂"))
        bar.addWidget(self.slice_axis)
        bar.addWidget(self.slice_pos, 1)
        bar.addWidget(self.slice_label)
        bar.addWidget(self.show_field)
        self.struct_panel = plot_panel(self.canvas_struct, self, bars=(bar,))
        self.slice_axis.currentIndexChanged.connect(self._plot_structure)
        self.slice_pos.valueChanged.connect(self._plot_structure)
        self.show_field.toggled.connect(self._plot_structure)

    # ------------------------------------------------------------------ curves and phases
    def _curve_names(self):
        return [d["name"] for d in self.datasets]

    def _fill_curve_table(self):
        self._building = True
        self.curve_table.setRowCount(len(self.datasets))
        for i, d in enumerate(self.datasets):
            it = _item(d["name"])
            it.setToolTip(d.get("path") or "")
            it.setForeground(QtGui.QBrush(QtGui.QColor(_mpl_colour(CURVE_COLOURS[i % len(CURVE_COLOURS)]))))
            self.curve_table.setItem(i, 0, it)
            self.curve_table.setItem(i, 1, _item(f"{d['q'].size}" + ("" if d["sigma"] is not None else ", no σ"),
                                                 editable=False, align_right=True))
            self.curve_table.setItem(i, 2, _item(f"{d['weight']:g}", align_right=True))
            chk = _item("", editable=False)
            chk.setFlags(chk.flags() | QtCore.Qt.ItemFlag.ItemIsUserCheckable)
            chk.setCheckState(QtCore.Qt.CheckState.Checked if d["fit_bg"] else QtCore.Qt.CheckState.Unchecked)
            self.curve_table.setItem(i, 3, chk)
            b = _item(f"{d['B']:.6g}", align_right=True)
            b.setToolTip(("fitted" + (f", ± {d['dB']:.2g}" if d.get("dB") else "")) if d["fit_bg"] else
                         ("fixed (set by hand)" if d["B"] else "no background"))
            if d["fit_bg"]:
                b.setForeground(QtGui.QBrush(QtGui.QColor("#2E8B57")))
            self.curve_table.setItem(i, 4, b)
        self._building = False
        self._sync_b_box()

    def _current_curve(self):
        i = self.curve_table.currentRow()
        return i if 0 <= i < len(self.datasets) else (0 if self.datasets else -1)

    def _sync_b_box(self):
        i = self._current_curve()
        self.b_box.blockSignals(True)
        self.b_box.setEnabled(i >= 0)
        self.b_fit.setEnabled(i >= 0)
        if i >= 0:
            d = self.datasets[i]
            self.b_box.setValue(d["B"])
            self.b_label.setText(f"B ({d['name']}, {'fitted' if d['fit_bg'] else 'fixed'}):")
        else:
            self.b_label.setText("B:")
        self.b_box.blockSignals(False)

    def _on_b_box(self, value):
        i = self._current_curve()
        if i < 0:
            return
        d = self.datasets[i]
        d["B"], d["fit_bg"], d["dB"] = float(value), False, 0.0
        self._fill_curve_table()
        self._b_timer.start()

    def _on_b_fit(self):
        i = self._current_curve()
        if i < 0:
            return
        self.datasets[i]["fit_bg"] = True
        self._fill_curve_table()
        self.on_correlations()

    def set_background(self, i, B=None, fit=False):
        """Background of curve i: fitted (``fit=True``) or the fixed value ``B``."""
        d = self.datasets[i]
        d["fit_bg"] = bool(fit)
        if B is not None:
            d["B"] = float(B)
        self._fill_curve_table()

    def _on_curve_item(self, it):
        if self._building:
            return
        i, c = it.row(), it.column()
        if i >= len(self.datasets):
            return
        if c == 0 and it.text().strip():
            self.datasets[i]["name"] = it.text().strip()
            self._set_sld_headers()
        elif c == 2:
            try:
                self.datasets[i]["weight"] = max(0.0, float(it.text()))
            except ValueError:
                self._fill_curve_table()
        elif c == 3:
            fit = it.checkState() == QtCore.Qt.CheckState.Checked
            if fit != self.datasets[i]["fit_bg"]:
                self.datasets[i]["fit_bg"] = fit
                self._fill_curve_table()
                self.on_correlations()
        elif c == 4:
            try:
                B = float(it.text())
            except ValueError:
                self._fill_curve_table()
                return
            d = self.datasets[i]
            if B != d["B"] or d["fit_bg"]:
                d["B"], d["fit_bg"], d["dB"] = B, False, 0.0
                self._fill_curve_table()
                self.on_correlations()

    def _set_sld_headers(self):
        names = self._curve_names() or ["SLD"]
        self.phase_table.setHorizontalHeaderLabels(["phase", "fraction"] + [f"SLD {n}" for n in names])

    def set_phases(self, names, fractions, sld_columns=None):
        """Fill the phase table: names, fractions and SLDs (``sld_columns[m][i]``: curve m, phase i)."""
        ncur = max(1, len(self.datasets))
        if sld_columns is None:
            old = self._sld_columns()
            sld_columns = [[1.0 if i == 0 else 0.0 for i in range(len(names))] for _ in range(ncur)]
            for m in range(min(ncur, len(old))):
                for i in range(min(len(names), len(old[m]))):
                    sld_columns[m][i] = old[m][i]
        self._building = True
        self.phase_table.setColumnCount(2 + ncur)
        self.phase_table.setRowCount(len(names))
        for i, (n, fr) in enumerate(zip(names, fractions)):
            it = _item(n)
            pm = QtGui.QPixmap(12, 12)
            pm.fill(QtGui.QColor.fromRgbF(*phase_colour(i)))
            it.setIcon(QtGui.QIcon(pm))
            self.phase_table.setItem(i, 0, it)
            self.phase_table.setItem(i, 1, _item(f"{fr:.4g}", align_right=True))
            for m in range(ncur):
                val = sld_columns[m][i] if m < len(sld_columns) and i < len(sld_columns[m]) else 0.0
                self.phase_table.setItem(i, 2 + m, _item(f"{val:.4g}", align_right=True))
        self._set_sld_headers()
        self.phase_table.resizeColumnsToContents()
        self._building = False
        self._update_fraction_label()

    def _table_float(self, r, c, default=0.0):
        it = self.phase_table.item(r, c)
        try:
            return float(it.text())
        except (AttributeError, ValueError):
            return default

    def _phase_names(self):
        return [(self.phase_table.item(i, 0).text().strip() if self.phase_table.item(i, 0) else f"phase {i}")
                or f"phase {i}" for i in range(self.phase_table.rowCount())]

    def _fractions(self):
        return [self._table_float(i, 1) for i in range(self.phase_table.rowCount())]

    def _sld_columns(self):
        return [[self._table_float(i, 2 + m) for i in range(self.phase_table.rowCount())]
                for m in range(max(0, self.phase_table.columnCount() - 2))]

    def _on_phase_item(self, _it):
        if not self._building:
            self._update_fraction_label()

    def _update_fraction_label(self):
        s = sum(self._fractions())
        ok = abs(s - 1) < 1e-6
        self.frac_label.setText(f"Σ fractions = {s:.4g}" + ("" if ok else "  (normalised)"))
        self.frac_label.setStyleSheet("" if ok else "color: #c8246e")

    def on_add_phase(self):
        names, fr, slds = self._phase_names(), self._fractions(), self._sld_columns()
        i = self.phase_table.currentRow()
        i = len(names) if i < 0 else i + 1
        names.insert(i, f"phase {len(names)}")
        fr.insert(i, 0.1)
        for col in slds:
            col.insert(i, 0.0)
        self.set_phases(names, fr, slds)
        self.phase_table.selectRow(i)

    def on_remove_phase(self):
        names, fr, slds = self._phase_names(), self._fractions(), self._sld_columns()
        i = self.phase_table.currentRow()
        if len(names) <= 2 or not 0 <= i < len(names):
            return
        for lst in [names, fr] + slds:
            del lst[i]
        self.set_phases(names, fr, slds)

    def on_move_phase(self, step):
        names, fr, slds = self._phase_names(), self._fractions(), self._sld_columns()
        i = self.phase_table.currentRow()
        j = i + step
        if not (0 <= i < len(names) and 0 <= j < len(names)):
            return
        for lst in [names, fr] + slds:
            lst[i], lst[j] = lst[j], lst[i]
        self.set_phases(names, fr, slds)
        self.phase_table.selectRow(j)

    def _update_porod_widgets(self, _i=None):
        m = self.porod_mode.currentData()
        self.porod_points.setVisible(m == "points")
        self.porod_qmin.setVisible(m == "qmin")

    # ------------------------------------------------------------------ parameters
    def params(self):
        fr = np.asarray(self._fractions(), float)
        if np.any(fr <= 0):
            raise ValueError("every volume fraction must be > 0")
        fr = fr / fr.sum()
        return MultiParameters(
            names=self._phase_names(), fractions=[float(x) for x in fr],
            box_size=_float_or_none(self.box.text()), n=self.n.value(),
            qmin=_float_or_none(self.qmin.text()), qmax=_float_or_none(self.qmax.text()),
            low_q=self.low_q.currentText(), low_q_points=self.low_q_points.value(),
            porod_points=self.porod_points.value() if self.porod_mode.currentData() == "points" else None,
            porod_qmin=_float_or_none(self.porod_qmin.text()) if self.porod_mode.currentData() == "qmin" else None,
            porod_auto=self.porod_mode.currentData() == "auto", fit_background=False,
            method=self.method.currentText(), n_waves=self.n_waves.value(),
            threshold=self.threshold.currentText(),
            seed=None if self.seed.value() < 0 else self.seed.value(),
            fit_fractions=self.fit_fractions.isChecked(), sld_absolute=self.sld_absolute.isChecked(),
            q_unit="A" if self.unit.currentIndex() == 0 else "nm")

    def set_params(self, p: MultiParameters):
        self.set_phases(list(p.names), list(p.fractions))
        self.box.setText("" if p.box_size is None else f"{p.box_size:.6g}")
        self.n.setValue(p.n)
        self.qmin.setText("" if p.qmin is None else f"{p.qmin:.6g}")
        self.qmax.setText("" if p.qmax is None else f"{p.qmax:.6g}")
        self.low_q.setCurrentText(p.low_q)
        self.low_q_points.setValue(p.low_q_points)
        mode = ("points" if p.porod_points else "qmin" if p.porod_qmin is not None else
                "auto" if getattr(p, "porod_auto", True) else "fixed15")
        self.porod_mode.setCurrentIndex(max(0, self.porod_mode.findData(mode)))
        if p.porod_points:
            self.porod_points.setValue(int(p.porod_points))
        self.porod_qmin.setText("" if p.porod_qmin is None else f"{p.porod_qmin:.6g}")
        if p.fit_background:                           # sessions before per-curve backgrounds
            for d in self.datasets:
                d["fit_bg"] = True
        self.method.setCurrentText(p.method)
        self.n_waves.setValue(p.n_waves)
        self.threshold.setCurrentText(p.threshold)
        self.seed.setValue(-1 if p.seed is None else int(p.seed))
        self.fit_fractions.setChecked(p.fit_fractions)
        self.sld_absolute.setChecked(p.sld_absolute)

    def dataset_objects(self):
        cols = self._sld_columns()
        return [DataSet(q=d["q"], I=d["I"], sigma=d["sigma"], sld=np.asarray(cols[m], float), name=d["name"],
                        path=d.get("path"), weight=d["weight"], fit_background=d["fit_bg"],
                        background=None if d["fit_bg"] else d["B"]) for m, d in enumerate(self.datasets)]

    def _length_unit(self):
        return "Å" if self.unit.currentIndex() == 0 else "nm"

    def _q_unit(self):
        return self.unit.currentText()

    def _update_enabled(self):
        has = bool(self.datasets)
        for b in (self.btn_corr, self.btn_run):
            b.setEnabled(has)
        hasres = self.result is not None
        self.act_export_curves.setEnabled(hasres)
        hasmat = hasres and self.result.labels is not None
        self.act_export_matrix.setEnabled(hasmat)
        self.act_export_stl.setEnabled(hasmat)
        self.btn_3d.setEnabled(hasmat)

    # ------------------------------------------------------------------ data
    @property
    def path(self):
        return self.datasets[0]["path"] if self.datasets else None

    @property
    def data(self):
        """(q, I, sigma) of the first curve (or None)."""
        if not self.datasets:
            return None
        d = self.datasets[0]
        return d["q"], d["I"], d["sigma"]

    def _ask_files(self, title):
        start = os.path.dirname(self.path) if self.path else ""
        paths, _ = QtWidgets.QFileDialog.getOpenFileNames(self, title, start,
                                                          "Data (*.dat *.txt *.csv *.pdh *.xy);;All files (*)")
        return paths

    def on_open(self):
        paths = self._ask_files("Open SAS data (q, I [, σ])")
        if paths:
            self.load_file(paths[0])
            for p in paths[1:]:
                self.load_file(p, add=True)

    def on_add(self):
        for p in self._ask_files("Add curves of the same sample (other contrasts)"):
            self.load_file(p, add=True)

    def on_remove(self):
        i = self.curve_table.currentRow()
        if not 0 <= i < len(self.datasets):
            return
        cols = self._sld_columns()
        del self.datasets[i]
        if i < len(cols):
            del cols[i]
        self._fill_curve_table()
        self.set_phases(self._phase_names(), self._fractions(), cols or None)
        self.result = None
        self._update_enabled()
        if self.datasets:
            self.on_correlations()
        else:
            self.refresh()

    def load_file(self, path, add=False, correlations=True):
        path = os.fspath(path)
        try:
            q, I, s = load_curve(path)
        except Exception as e:  # noqa: BLE001
            QtWidgets.QMessageBox.critical(self, "Open data", f"{path}:\n{e}")
            return False
        cols = self._sld_columns()
        name = os.path.splitext(os.path.basename(path))[0]
        if add:
            names = self._curve_names()
            base, k = name, 2
            while name in names:
                name, k = f"{base} ({k})", k + 1
            self.datasets.append({"name": name, "path": path, "q": q, "I": I, "sigma": s, "weight": 1.0,
                              "fit_bg": self.fit_bg_default.isChecked(), "B": 0.0, "dB": 0.0})
            cols.append(list(cols[-1]) if cols else [1.0] + [0.0] * (self.phase_table.rowCount() - 1))
        else:
            self.datasets = [{"name": name, "path": path, "q": q, "I": I, "sigma": s, "weight": 1.0,
                              "fit_bg": self.fit_bg_default.isChecked(), "B": 0.0, "dB": 0.0}]
            cols = cols[:1]
        self.result = None
        self._fill_curve_table()
        self.set_phases(self._phase_names(), self._fractions(), cols)
        self._update_enabled()
        if correlations:
            self.on_correlations()
        return True

    # ------------------------------------------------------------------ runs
    def _set_progress(self, x, text):
        self.progress.setValue(int(100 * x))
        self.progress.setFormat(text)
        QtWidgets.QApplication.processEvents()

    def _run(self, full):
        if not self.datasets:
            return
        QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.CursorShape.WaitCursor)
        try:
            p = self.params()
            sets = self.dataset_objects()
            if full:
                res = multiphase.reconstruct_multi(sets, p, keep_field=True, progress=self._set_progress)
            else:
                self._set_progress(0.1, "correlations" + (" and fractions fit" if p.fit_fractions else ""))
                res = multiphase.correlations_multi(sets, p)
                self._set_progress(1.0, "correlations done")
        except Exception as e:  # noqa: BLE001
            QtWidgets.QApplication.restoreOverrideCursor()
            self._set_progress(0, "failed")
            QtWidgets.QMessageBox.critical(self, "SASmorph", f"{e}\n\n{traceback.format_exc(limit=3)}")
            return
        QtWidgets.QApplication.restoreOverrideCursor()
        if self.box.text().strip() == "" and res.params.box_size is not None:
            self.box.setPlaceholderText(f"auto ({res.params.box_size:.4g})")
        if res.fit_info:                                  # fitted fractions back into the table
            self.fit_fractions.blockSignals(True)
            self.set_phases(self._phase_names(), list(res.params.fractions), self._sld_columns())
            self.fit_fractions.blockSignals(False)
        self.result = res
        for d, c in zip(self.datasets, res.curves):        # fitted backgrounds into the table
            if d["fit_bg"]:
                d["B"], d["dB"] = float(c.curve.B), float(c.curve.dB)
        self._fill_curve_table()
        if res.labels is not None:
            n = res.labels.shape[0]
            self.slice_pos.blockSignals(True)
            self.slice_pos.setRange(0, n - 1)
            self.slice_pos.setValue(n // 2)
            self.slice_pos.blockSignals(False)
        self._update_enabled()
        self.refresh()
        self.view3d.set_result(res if res.labels is not None else None, self._length_unit())
        self.statusBar().showMessage("; ".join(res.warnings) if res.warnings else "ok", 15000)

    def python_code(self):
        """Script: the curves with their SLDs, weights and backgrounds, the parameters of the panel and the
        reconstruction (Reconstruct, if the last run was one, else Correlations) ("Copy Python code")."""
        from ..core.pycode import Code, Script, call, lit, nondefault, path_lit

        p = self.params()
        sets = self.dataset_objects()
        full = self.result is not None and self.result.labels is not None
        sc = Script("pySASmorph", "3D morphology from SAS data" + (" (reconstruction)" if full else " (correlations)"))
        sc.imp("import numpy as np", "from pysasfit.morph.io import load_curve",
               "from pysasfit.morph.multiphase import DataSet, MultiParameters, correlations_multi, reconstruct_multi")
        sc.comment("the curves (background-subtracted or with a flat background B) and the SLD of every phase")
        sc.add("sets = []")
        if not sets:
            sc.add(f"q, I, sigma = load_curve({path_lit(None)})",
                   call("sets.append(DataSet", Code("q"), Code("I"), Code("sigma"),
                        sld=np.asarray(self._sld_columns()[0] if self._sld_columns() else [1.0, 0.0], float)) + ")")
        for d in sets:
            if d.path and os.path.isfile(d.path):
                sc.add(f"q, I, sigma = load_curve({path_lit(d.path)})")
                q, I, sg = "q", "I", "sigma"
            else:
                sc.add(f"q, I, sigma = {lit(d.q)}, {lit(d.I)}, {lit(d.sigma)}")
                q, I, sg = "q", "I", "sigma"
            kw = dict(sld=d.sld, name=d.name)
            if d.weight != 1.0:
                kw["weight"] = d.weight
            kw["fit_background"] = d.fit_background
            if d.background is not None:
                kw["background"] = d.background
            sc.add(call("sets.append(DataSet", Code(q), Code(I), Code(sg), **kw) + ")")
        pd = nondefault(p.to_dict(), MultiParameters().to_dict())
        sc.comment("parameters of the panel (lengths in the inverse unit of q)")
        sc.add(call("params = MultiParameters", **pd))
        if full:
            sc.comment("extrapolation, correlation functions, random field, phases, their scattering and statistics")
            sc.add("res = reconstruct_multi(sets, params, keep_field=True)")
        else:
            sc.comment("extrapolation, correlation functions, common g(r), f(k) and the model I(q)")
            sc.add("res = correlations_multi(sets, params)")
        sc.add("print(res.report())")
        plot = ["fig, axes = plt.subplots(1, 2 if res.labels is not None else 1, figsize=(12, 5), squeeze=False)",
                "ax = axes[0, 0]",
                "for c in res.curves:",
                "    ax.plot(c.q_data, c.I_data, 'o', ms=3, mfc='none', label=c.data.name)",
                "    ax.plot(c.q_data, c.I_model, 'k-', lw=1)",
                "    if c.I_voxel is not None:",
                "        ax.plot(c.q_voxel, c.I_voxel, '.', ms=3)",
                "ax.set_xscale('log')", "ax.set_yscale('log')",
                f"ax.set_xlabel('q ({self._q_unit()})')", "ax.set_ylabel('I(q)')", "ax.legend()",
                "if res.labels is not None:",
                "    n = res.labels.shape[0]",
                "    axes[0, 1].imshow(res.labels[:, :, n // 2].T, origin='lower', cmap='viridis',",
                "                      extent=(0, res.params.box_size, 0, res.params.box_size))",
                f"    axes[0, 1].set_title('slice z = L/2 ({self._length_unit()})')"]
        sc.add_plot(*plot, "fig.tight_layout()")
        return sc.text()

    def on_correlations(self):
        self._run(False)

    def on_reconstruct(self):
        n = self.n.value()
        if n >= 320 and QtWidgets.QMessageBox.question(
                self, "SASmorph", f"{n}³ voxels need several GB of memory and take a while. Continue?") \
                != QtWidgets.QMessageBox.StandardButton.Yes:
            return
        self._run(True)

    def on_suggest_box(self):
        if not self.datasets:
            return
        p = self.params()
        p.box_size, p.fit_fractions = None, False
        res = multiphase.correlations_multi(self.dataset_objects(), p)
        self.box.setText(f"{res.params.box_size:.4g}")

    # ------------------------------------------------------------------ exports
    def _base_name(self):
        if self.path:
            return os.path.splitext(self.path)[0]
        return os.path.join(os.getcwd(), "sasmorph")

    def on_export_curves(self):
        if self.result is None:
            return
        base, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Export curves (base name)", self._base_name(),
                                                        "Base name (*)")
        if base:
            files = write_curves_multi(self.result, os.path.splitext(base)[0])
            self.statusBar().showMessage(f"{len(files)} files written: {', '.join(map(os.path.basename, files))}",
                                         15000)

    def on_export_matrix(self):
        if self.result is None or self.result.labels is None:
            return
        path, flt = QtWidgets.QFileDialog.getSaveFileName(self, "Export voxel matrix (phase index per voxel)",
                                                          self._base_name() + ".vtk", ";;".join(MATRIX_FORMATS))
        if not path:
            return
        if not os.path.splitext(path)[1]:
            path += MATRIX_FORMATS.get(flt, ".vtk")
        try:
            write_matrix(self.result, path)
        except Exception as e:  # noqa: BLE001
            QtWidgets.QMessageBox.critical(self, "Export", str(e))
            return
        self.statusBar().showMessage(f"written: {path}", 10000)

    def on_anisotropic(self):
        """Window for anisotropic 2D data (uniaxial texture, models A and B)."""
        from .aniso_gui import AnisoWindow
        if getattr(self, "_aniso", None) is None:
            self._aniso = AnisoWindow()
        self._aniso.show()
        self._aniso.raise_()
        return self._aniso

    def show_3d_panel(self):
        self.set_panel_visible("view3d", True)
        self.docks["view3d"].raise_()

    def _selected_phases(self):
        v = self.view3d
        n = len(self.result.params.names)
        sel = v.phase.currentData() if v.view is not None else n - 1
        return list(range(n)) if sel is None or sel < 0 else [sel]

    def on_export_stl(self):
        """The surface(s) as in the 3D panel (phase, surface, clipping) as binary STL, one file per phase."""
        res = self.result
        if res is None or res.labels is None:
            return
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Export surface", self._base_name() + ".stl",
                                                        "STL (*.stl)")
        if not path:
            return
        from .surface import label_surface, write_stl
        v = self.view3d
        src, clip = (v.source.currentData(), v.clip()) if v.view is not None else ("smoothed", None)
        phases = self._selected_phases()
        written = []
        try:
            for i in phases:
                m = label_surface(res.labels, i, res.voxel_size, src, psi=res.psi, levels=res.levels, clip=clip)
                if m is None:
                    continue
                out = path if len(phases) == 1 else os.path.splitext(path)[0] + f"_{res.params.names[i]}.stl"
                written.append(write_stl(out, m))
        except Exception as e:  # noqa: BLE001
            QtWidgets.QMessageBox.critical(self, "Export surface", str(e))
            return
        self.statusBar().showMessage("written: " + ", ".join(written), 10000)

    def on_view3d(self):
        if self.result is None or self.result.labels is None:
            return
        from .view3d import pyvista_problem, show_3d
        problem = pyvista_problem()
        if problem:
            box = QtWidgets.QMessageBox(self)
            box.setIcon(QtWidgets.QMessageBox.Warning)
            box.setWindowTitle("3D view")
            box.setText("PyVista cannot be imported in the Python that runs pySASfit.\n\n"
                        "Install it for exactly this Python:\n    \"<Python below>\" -m pip install pyvista pyvistaqt\n\n"
                        "The '3D' panel of this window needs no PyVista; or export the voxel matrix (VTK)\n"
                        "and open it in ParaView.")
            box.setDetailedText(problem)
            box.setInformativeText(problem.split("\n\n")[0] + "\n" + problem.split("\n\n")[1])
            box.exec()
            return
        try:
            i = self._selected_phases()[-1]
            self._viewer = show_3d(self.result.labels == i, self.result.voxel_size, phase="pore",
                                   highlight_largest=self.view3d.view is not None and self.view3d.largest.isChecked(),
                                   color="#%02x%02x%02x" % tuple(int(255 * c) for c in phase_colour(i)[:3]),
                                   title=f"SASmorph – {self.result.params.names[i]}")
        except Exception as e:  # noqa: BLE001
            QtWidgets.QMessageBox.critical(self, "3D view", str(e))

    # ------------------------------------------------------------------ plots
    def refresh(self):
        self._plot_iq()
        self._plot_corr()
        self._plot_spec()
        self._plot_structure()
        self.results_text.setPlainText(self.result.report() if self.result else "")

    def _offsets(self, n):
        if n < 2 or not self.offset_curves.isChecked():
            return [1.0] * n
        return [10.0 ** (n - 1 - m) for m in range(n)]

    def _plot_iq(self):
        ax = self.ax_iq
        ax.clear()
        self.offset_curves.setEnabled(self.iq_mode.currentData() == "loglog")
        if self.iq_mode.currentData() == "porod":
            self._plot_porod(ax)
            self.canvas_iq.draw_idle()
            return
        res = self.result
        n = len(self.datasets)
        off = self._offsets(n)
        ymin, ymax = np.inf, 0.0
        for m, d in enumerate(self.datasets):
            col = CURVE_COLOURS[m % len(CURVE_COLOURS)]
            label = d["name"] + (f" × {off[m]:g}" if off[m] != 1 else "")
            if d["sigma"] is not None:
                ax.errorbar(d["q"], d["I"] * off[m], d["sigma"] * off[m], fmt="o", ms=2.5, mfc="none", color=col,
                            alpha=0.55, lw=0.5, label=label, zorder=1)
            else:
                ax.plot(d["q"], d["I"] * off[m], "o", ms=2.5, mfc="none", color=col, alpha=0.55, label=label)
            pos = d["I"][d["I"] > 0] * off[m]
            if pos.size:
                ymin, ymax = min(ymin, pos.min()), max(ymax, pos.max())
        if res is not None:
            for m, c in enumerate(res.curves):
                col = CURVE_COLOURS[m % len(CURVE_COLOURS)]
                cv = c.curve
                ax.plot(cv.q[cv.q > 0], cv.I[cv.q > 0] * off[m], "-", color=col, lw=0.8, alpha=0.6)
                qt = np.geomspace(cv.qN, cv.qN * 10, 50)
                ax.plot(qt, cv.K * qt ** -4 * off[m], "--", color=col, lw=0.8, alpha=0.6)
                ax.plot(c.q_data, c.I_model * off[m], "-", color="k", lw=1.0,
                        label="level-cut model" if m == 0 else None)
                if cv.B:
                    # the background: horizontal line, Porod law + B on the raw data, data after subtraction
                    ax.axhline(abs(cv.B) * off[m], color=col, lw=0.9, ls="-.", alpha=0.9)
                    ax.text(c.q_data[-1], abs(cv.B) * off[m], f" B = {cv.B:.4g}" + (" (fitted)" if cv.B_mode == "fitted"
                                                                                    else " (fixed)")
                            + (" < 0" if cv.B < 0 else ""), color=col, va="bottom", ha="right", fontsize=7)
                    qp = np.geomspace(cv.porod_range[0], cv.porod_range[1], 60)
                    ax.plot(qp, (cv.K * qp ** -4 + cv.B) * off[m], "-", color=col, lw=1.6, alpha=0.9,
                            label="Porod K/q⁴ + B" if m == 0 else None)
                    pos = c.I_data > 0
                    ax.plot(c.q_data[pos], c.I_data[pos] * off[m], "x", ms=2.5, color=col, alpha=0.5,
                            label="data − B" if m == 0 else None)
                    if np.any(pos):
                        ymin = min(ymin, c.I_data[pos].min() * off[m])
                if c.I_voxel is not None:
                    ax.plot(c.q_voxel, c.I_voxel * off[m], ".", color=col, ms=3,
                            label="voxel structure" if m == 0 else None)
            if res.labels is not None:
                qn = np.pi / res.voxel_size
                ax.axvline(qn, color="0.4", lw=0.8, ls=":")
                ax.text(qn, 0.98, " π/a", color="0.4", transform=ax.get_xaxis_transform(), va="top", fontsize=8)
            qlo = min(c.curve.q_data[0] for c in res.curves)
            qhi = max(c.curve.q_data[1] for c in res.curves)
            ax.set_xlim(qlo / 5, qhi * 3)
        if np.isfinite(ymin) and ymax > 0:
            ax.set_ylim(ymin * 0.2, ymax * 5)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(f"q ({self._q_unit()})")
        ax.set_ylabel("I(q)")
        if ax.has_data():
            ax.legend(fontsize=7)
        self.canvas_iq.draw_idle()

    def _plot_porod(self, ax):
        """I q^4 / K against q^4 over the upper part of every curve: a flat background is a straight line
        1 + (B/K) q^4 there, the data after subtracting B scatter around 1."""
        res = self.result
        qu = self._q_unit()
        if res is None:
            ax.text(0.5, 0.5, "no correlations yet", ha="center", va="center", transform=ax.transAxes, color="0.5")
            return
        for m, (d, c) in enumerate(zip(self.datasets, res.curves)):
            col = CURVE_COLOURS[m % len(CURVE_COLOURS)]
            cv = c.curve
            K = cv.K if cv.K > 0 else np.nanmax(d["I"] * d["q"] ** 4)
            q0 = 0.5 * cv.porod_range[0]
            sel = (d["q"] >= q0) & (d["q"] <= cv.q_data[1])
            x = d["q"][sel] ** 4
            ax.plot(x, d["I"][sel] * x / K, "o", ms=3, mfc="none", color=col, alpha=0.6, label=f"{d['name']}")
            if cv.B:
                ax.plot(x, (d["I"][sel] - cv.B) * x / K, "x", ms=3, color=col, alpha=0.7)
            xp = np.linspace(cv.porod_range[0] ** 4, cv.q_data[1] ** 4, 50)
            ax.plot(xp, 1 + cv.B / K * xp, "-", color=col, lw=1.3,
                    label=f"1 + (B/K) q⁴, B = {cv.B:.4g}" + (" fitted" if cv.B_mode == "fitted" else
                                                            " fixed" if cv.B_mode == "fixed" else ""))
            ax.axvspan(cv.porod_range[0] ** 4, cv.porod_range[1] ** 4, color=col, alpha=0.06)
        ax.axhline(1.0, color="k", lw=0.8)
        # y range from the Porod ranges (the low-q points rise steeply and would squeeze the plot)
        ys = []
        for d, c in zip(self.datasets, res.curves):
            cv = c.curve
            K = cv.K if cv.K > 0 else np.nanmax(d["I"] * d["q"] ** 4)
            sel = (d["q"] >= cv.porod_range[0]) & (d["q"] <= cv.q_data[1])
            ys += [d["I"][sel] * d["q"][sel] ** 4 / K, (d["I"][sel] - cv.B) * d["q"][sel] ** 4 / K]
        ys = np.concatenate(ys) if ys else np.array([])
        ys = ys[np.isfinite(ys)]
        if ys.size:
            lo, hi = min(ys.min(), 1.0), max(ys.max(), 1.0)
            pad = 0.08 * (hi - lo or 1.0)
            ax.set_ylim(lo - pad, hi + pad)
        ax.set_xlabel(f"q⁴ ({qu}⁴)")
        ax.set_ylabel("I q⁴ / K")
        ax.set_title("o raw data, x data − B; shaded: Porod fit range", fontsize=8)
        ax.legend(fontsize=7)

    def _plot_corr(self):
        ax = self.ax_corr
        ax.clear()
        res = self.result
        if res is not None:
            for m, c in enumerate(res.curves):
                col = CURVE_COLOURS[m % len(CURVE_COLOURS)]
                ax.plot(res.r, c.gamma, color=col, lw=2.2, alpha=0.4, label=f"γ {c.data.name}")
                ax.plot(res.r, c.G, color=col, lw=0.9, ls="--")
            ax.plot(res.r, res.g, color="k", lw=1.3, label="g(r)  (Gaussian field)")
            ax.axhline(0, color="0.6", lw=0.6)
            below = np.flatnonzero(res.curves[0].gamma < np.exp(-1))
            rc = res.r[below[0]] if below.size else res.r[-1]
            ax.set_xlim(0, min(res.r[-1], 10 * rc))
            ax.legend(fontsize=7, title="thick: data, dashed: model", title_fontsize=7)
        ax.set_xlabel(f"r ({self._length_unit()})")
        ax.set_ylabel("correlation")
        self.canvas_corr.draw_idle()

    def _plot_spec(self):
        ax = self.ax_spec
        ax.clear()
        res = self.result
        if res is not None:
            ax.plot(res.k, res.Pk, color="C4", label="P(k) = k² f(k) / 2π²")
            ax.set_xscale("log")
            ax.axvline(np.pi / res.voxel_size, color="C2", lw=0.8, ls=":", label="π/a (voxel)")
            ax.axvline(2 * np.pi / res.params.box_size, color="0.5", lw=0.8, ls=":", label="2π/L (box)")
            ax.legend(fontsize=8)
        ax.set_xlabel(f"k ({self._q_unit()})")
        ax.set_ylabel("distribution of wave vectors")
        self.canvas_spec.draw_idle()

    def _plot_structure(self):
        fig = self.fig_struct
        ax = self.ax_struct
        ax.clear()
        res = self.result
        if res is None or res.labels is None:
            ax.text(0.5, 0.5, "Reconstruct to see the structure", ha="center", va="center",
                    transform=ax.transAxes, color="0.5")
            ax.set_axis_off()
            self.canvas_struct.draw_idle()
            return
        ax.set_axis_on()
        axis = 2 - self.slice_axis.currentIndex()          # z, y, x
        i = min(self.slice_pos.value(), res.labels.shape[axis] - 1)
        use_field = self.show_field.isChecked() and res.psi is not None
        L = res.params.box_size
        names = list(res.params.names)
        lab = np.take(res.labels, i, axis=axis)
        if use_field:
            sl = np.take(res.psi, i, axis=axis)
            ax.imshow(sl.T, origin="lower", extent=(0, L, 0, L), cmap="RdBu_r", vmin=-3, vmax=3)
            ax.contour(np.linspace(0, L, sl.shape[0]), np.linspace(0, L, sl.shape[1]), sl.T, levels=np.sort(res.levels),
                       colors="k", linewidths=0.5)
            ax.set_title("field ψ, cut at " + ", ".join(f"{x:.3f}" for x in res.levels), fontsize=9)
        else:
            cmap = ListedColormap([phase_colour(k)[:3] for k in range(len(names))])
            ax.imshow(lab.T, origin="lower", extent=(0, L, 0, L), cmap=cmap, vmin=-0.5, vmax=len(names) - 0.5,
                      interpolation="nearest")
            ax.legend(handles=[Patch(color=phase_colour(k)[:3], label=f"{n} ({f:.3f})")
                               for k, (n, f) in enumerate(zip(names, res.params.fractions))],
                      fontsize=7, loc="upper right", framealpha=0.85)
        xyz = "xyz"
        others = [xyz[j] for j in range(3) if j != axis]
        ax.set_xlabel(f"{others[0]} ({self._length_unit()})")
        ax.set_ylabel(f"{others[1]} ({self._length_unit()})")
        self.slice_label.setText(f"{xyz[axis]} = {(i + 0.5) * res.voxel_size:.4g}")
        fig.canvas.draw_idle()

    # ------------------------------------------------------------------ session (launcher hooks)
    @staticmethod
    def _session_path():
        return os.path.join(os.path.dirname(prefs.settings_path()), "morph_session.json")

    def save_session(self):
        try:
            p = self._session_path()
            os.makedirs(os.path.dirname(p), exist_ok=True)
            try:
                params = self.params().to_dict()
            except ValueError:
                params = {}
            with open(p, "w", encoding="utf-8") as fh:
                json.dump({"version": 2, "curves": [{"path": d["path"], "name": d["name"], "weight": d["weight"],
                                                     "fit_bg": d["fit_bg"], "B": d["B"]} for d in self.datasets],
                           "sld": self._sld_columns(), "params": params,
                           "unit": self.unit.currentIndex(), "dock_state": self.dock_state(),
                           "fit_bg_default": self.fit_bg_default.isChecked(),
                           "view3d": self.view3d.state()}, fh, indent=1)
        except (OSError, TypeError, ValueError):
            pass

    def restore_session(self):
        try:
            with open(self._session_path(), encoding="utf-8") as fh:
                s = json.load(fh)
        except (OSError, ValueError):
            return False
        try:
            if s.get("params"):
                self.set_params(MultiParameters.from_dict(s["params"]))
            self.unit.setCurrentIndex(int(s.get("unit", 0)))
            self.fit_bg_default.setChecked(bool(s.get("fit_bg_default", True)))
            if s.get("dock_state"):
                self.restore_dock_state(s["dock_state"])
        except Exception:  # noqa: BLE001 -- an old or broken session must not stop the tool
            pass
        curves = s.get("curves")
        if curves is None and s.get("file"):                        # session of the two-phase version
            curves = [{"path": s["file"]}]
        loaded = False
        for c in curves or []:
            if c.get("path") and os.path.isfile(c["path"]):
                self.load_file(c["path"], add=loaded, correlations=False)
                d = self.datasets[-1]
                d["name"], d["weight"] = c.get("name", d["name"]), float(c.get("weight", 1.0))
                legacy = bool((s.get("params") or {}).get("fit_background", False))
                d["fit_bg"], d["B"] = bool(c.get("fit_bg", legacy)), float(c.get("B", 0.0))
                loaded = True
        if loaded:
            sld = s.get("sld")
            if sld and len(sld) == len(self.datasets):
                self.set_phases(self._phase_names(), self._fractions(), sld)
            self._fill_curve_table()
            self._set_sld_headers()
            # not calculated at the start: the Correlations button
            self.statusBar().showMessage("last session restored: press Correlations to calculate", 15000)
        try:
            self.view3d.set_state(s.get("view3d", {}))
        except Exception:  # noqa: BLE001
            pass
        return True

    def forget_session(self):
        try:
            os.remove(self._session_path())
        except OSError:
            pass

    def shutdown(self):
        self.save_session()

    def closeEvent(self, event):
        self.save_session()
        super().closeEvent(event)


def _mpl_colour(c):
    from matplotlib.colors import to_hex
    return to_hex(c)


def main(argv=None):
    argv = sys.argv if argv is None else argv
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(argv)
    from ..core.tabstyle import apply_tab_style
    apply_tab_style(app)
    app.setWindowIcon(tool_icon("pySASmorph"))
    w = MainWindow()
    w.show()
    files = [a for a in argv[1:] if os.path.isfile(a)]
    if files:
        for k, f in enumerate(files):
            w.load_file(f, add=k > 0, correlations=k == len(files) - 1)
    else:
        w.restore_session()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
