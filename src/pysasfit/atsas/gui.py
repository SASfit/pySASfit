"""pyATsas window: the ATSAS programs on the curves of pySASfit.

    pysasfit-atsas [data files]
    python -m pysasfit.atsas.gui [data files]

Also the "pyATsas" tab of the pySASfit launcher. Panels (View menu; each can be moved, stacked or detached):

Data         curves (text, pySASred ASCII, NXcanSAS -- all entries), point range used for every program
Analysis     AUTORG (Guinier), DATGNOM (automatic p(r)), GNOM (fixed Dmax, systems 0-6), CRYSOL / CRYSON
             (model curve, fit to the curve), DAMMIF (bead models from the last p(r))
I(q)         data, Guinier range and line, GNOM fit, model curve; log-log, Guinier, Kratky ...
p(r)         DATGNOM and GNOM distance distributions with errors
Results      one row per curve (Rg, I(0), Dmax, total estimate, χ², ...)
Log          every command line and the programs' output
ATSAS        the interactive ATSAS programs (PRIMUS, CHROMIXS, PEAK, ...) with their original icons,
             the installation found, its licence, a self test

ATSAS (EMBL Hamburg / BIOSAXS GmbH) must be installed separately with its own licence. Every run gets its own
folder (Settings -> Preferences -> pyATsas), so its files stay available. q in 1/nm, lengths in nm.
"""
from __future__ import annotations

import datetime
import json
import os
import re
import sys
import threading
import traceback

import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402

from ..core import settings as prefs  # noqa: E402
from ..core.docking import DockMixin, plot_panel  # noqa: E402
from ..resources import tool_icon  # noqa: E402
from . import core as A  # noqa: E402

__all__ = ["MainWindow", "main"]

Dock = QtCore.Qt.DockWidgetArea

GNOM_SYSTEMS = [(0, "0: monodisperse, p(r)"), (1, "1: polydisperse spheres, D_V(R)"),
                (3, "3: rods, cross-section p_c(r)"), (4, "4: lamellae, thickness p_t(r)"),
                (5, "5: cylinders of radius rad56, length distribution"),
                (6, "6: spherical shells, rad56 = relative thickness")]
IQ_SCALES = ["log I vs log q", "log I vs q", "Guinier: ln I vs q²", "Kratky: q² I vs q", "linear"]
COLUMNS = ["curve", "Rg (Guinier)", "I(0)", "points", "quality", "Dmax (DATGNOM)", "Rg (real)",
           "total est.", "Dmax (GNOM)", "Rg (GNOM)", "χ² (model)", "bead models"]


def _canvas(size=(5, 4)):
    return FigureCanvasQTAgg(Figure(figsize=size, layout="constrained"))


def _fmt(v, n=4):
    try:
        v = float(v)
    except (TypeError, ValueError):
        return "" if v is None else str(v)
    return "" if not np.isfinite(v) else f"{v:.{n}g}"


def _safe(name, n=40):
    return re.sub(r"[^A-Za-z0-9_.-]", "_", name)[:n] or "curve"


class _Signals(QtCore.QObject):
    done = QtCore.Signal(object, object, object)       # tag, result, error text


# ===================================================================================== main window
class MainWindow(DockMixin, QtWidgets.QMainWindow):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("pyATsas – the ATSAS programs on pySASfit data")
        self.setWindowIcon(tool_icon("pyATsas"))
        self.curves = []                 # pysasfit.toolbox.merge.Curve (q in 1/nm)
        self.results = {}                # curve name -> {"autorg": GuinierResult, "datgnom": ..., ...}
        self.model_path = ""
        self._busy = False
        self._signals = _Signals()
        self._signals.done.connect(self._on_done)
        self._build_menus()
        self._build_data_panel()
        self._build_analysis_panel()
        self._build_plots()
        self._build_results_panel()
        self._build_programs_panel()
        self.log_view = QtWidgets.QPlainTextEdit()
        self.log_view.setReadOnly(True)
        self.log_view.setFont(QtGui.QFontDatabase.systemFont(QtGui.QFontDatabase.SystemFont.FixedFont))
        self._add_dock("data", "Data", self.data_widget, Dock.LeftDockWidgetArea, scrollable=True)
        self._add_dock("analysis", "Analysis", self.analysis_widget, Dock.LeftDockWidgetArea, scrollable=True)
        self._add_dock("iq", "I(q)", self.iq_panel, Dock.RightDockWidgetArea)
        self._add_dock("pr", "p(r)", self.pr_panel, Dock.RightDockWidgetArea)
        self._add_dock("results", "Results", self.table, Dock.RightDockWidgetArea)
        self._add_dock("log", "Log", self.log_view, Dock.RightDockWidgetArea)
        self._add_dock("programs", "ATSAS", self.programs_widget, Dock.LeftDockWidgetArea, scrollable=True)
        self.splitDockWidget(self.docks["iq"], self.docks["pr"], QtCore.Qt.Orientation.Horizontal)
        self.splitDockWidget(self.docks["iq"], self.docks["results"], QtCore.Qt.Orientation.Vertical)
        self.tabifyDockWidget(self.docks["results"], self.docks["log"])
        self.tabifyDockWidget(self.docks["analysis"], self.docks["programs"])
        self.docks["results"].raise_()
        self.docks["analysis"].raise_()
        self._finish_docks()
        self.resize(1500, 920)
        self._sized = False
        self.hide_stray_tab_bars()
        self.refresh_installation()
        self._update_buttons()

    def showEvent(self, event):
        super().showEvent(event)
        if not self._sized:          # panel widths once the window has its real size (also inside the launcher)
            self._sized = True
            QtCore.QTimer.singleShot(0, self._initial_sizes)

    def _initial_sizes(self):
        if getattr(self, "_session_layout", False):
            return
        w = max(self.width(), 900)
        self.resizeDocks([self.docks["data"], self.docks["iq"], self.docks["pr"]],
                         [int(0.27 * w), int(0.42 * w), int(0.31 * w)], QtCore.Qt.Orientation.Horizontal)
        self.resizeDocks([self.docks["data"], self.docks["analysis"]], [int(0.3 * self.height()), int(0.7 * self.height())],
                         QtCore.Qt.Orientation.Vertical)
        self._start_layout = self.saveState()          # Reset layout: with these sizes

    # ------------------------------------------------------------------ construction
    def _build_menus(self):
        m = self.menuBar().addMenu("&File")
        a = m.addAction("&Load data…", self.on_load)
        a.setShortcut(QtGui.QKeySequence("Ctrl+L"))
        m.addAction("Open selected curve in PRIMUS", lambda: self.on_launch("primusqt", with_data=True))
        m.addSeparator()
        m.addAction("Export results table (CSV)…", self.on_export_table)
        m.addAction("Export p(r) of the selected curve…", self.on_export_pr)
        m.addAction("Open the run folder", self.on_open_runs)
        from ..core import pycode
        pycode.install(self, self.python_code, menu=m)
        from ..core.settings_dialog import add_preferences_menu
        add_preferences_menu(self, "atsas")
        self.menuBar().addMenu("&View")

    def _build_data_panel(self):
        w = QtWidgets.QWidget()
        lay = QtWidgets.QVBoxLayout(w)
        lay.setContentsMargins(4, 4, 4, 4)
        row = QtWidgets.QHBoxLayout()
        b = QtWidgets.QPushButton("Load…")
        b.setToolTip("Curves: q I [dI [dq]] text files, pySASred ASCII, NXcanSAS (every entry); q in 1/nm "
                     "(1/Å files with a unit header are converted)")
        b.clicked.connect(self.on_load)
        row.addWidget(b)
        b = QtWidgets.QPushButton("Remove")
        b.clicked.connect(self.on_remove)
        row.addWidget(b)
        row.addStretch(1)
        lay.addLayout(row)
        self.curve_list = QtWidgets.QListWidget()
        self.curve_list.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.ExtendedSelection)
        self.curve_list.currentRowChanged.connect(self._on_curve_changed)
        lay.addWidget(self.curve_list, 1)
        self.curve_info = QtWidgets.QLabel("no data")
        self.curve_info.setWordWrap(True)
        self.curve_info.setStyleSheet("color: gray;")
        lay.addWidget(self.curve_info)
        form = QtWidgets.QFormLayout()
        self.first = QtWidgets.QSpinBox()
        self.first.setRange(1, 1)
        self.first.setToolTip("First point given to the ATSAS programs (1 = first point of the curve)")
        self.last = QtWidgets.QSpinBox()
        self.last.setRange(1, 1)
        self.last.setToolTip("Last point given to the ATSAS programs")
        for sp in (self.first, self.last):
            sp.valueChanged.connect(lambda _v: self.draw_iq())
        rng = QtWidgets.QHBoxLayout()
        rng.addWidget(self.first)
        rng.addWidget(QtWidgets.QLabel("…"))
        rng.addWidget(self.last)
        b = QtWidgets.QPushButton("all")
        b.setMaximumWidth(40)
        b.clicked.connect(self._full_range)
        rng.addWidget(b)
        form.addRow("points", rng)
        self.range_info = QtWidgets.QLabel("")
        self.range_info.setStyleSheet("color: gray;")
        form.addRow("", self.range_info)
        lay.addLayout(form)
        self.data_widget = w

    def _run_button(self, text, slot, tip):
        b = QtWidgets.QPushButton(text)
        b.setToolTip(tip)
        b.clicked.connect(lambda _c=False: slot())
        self._run_buttons.append(b)
        return b

    def _build_analysis_panel(self):
        self._run_buttons = []
        w = QtWidgets.QWidget()
        lay = QtWidgets.QVBoxLayout(w)
        lay.setContentsMargins(2, 2, 2, 2)
        row = QtWidgets.QHBoxLayout()
        row.addWidget(self._run_button("Rg + p(r) of all", self.run_batch,
                                       "AUTORG and DATGNOM for every loaded curve, one after the other"))
        from ..core import pycode
        row.addWidget(pycode.button(self.python_code, w))
        lay.addLayout(row)
        self.progress = QtWidgets.QProgressBar()
        self.progress.setRange(0, 1)
        self.progress.setTextVisible(False)
        self.progress.setMaximumHeight(8)
        lay.addWidget(self.progress)
        tb = QtWidgets.QToolBox()
        lay.addWidget(tb, 1)

        # --- AUTORG
        p = QtWidgets.QWidget()
        f = QtWidgets.QFormLayout(p)
        self.ag_minint = QtWidgets.QSpinBox()
        self.ag_minint.setRange(2, 1000)
        self.ag_minint.setValue(3)
        self.ag_minint.setToolTip("--mininterval: shortest Guinier interval in points")
        f.addRow("min. interval", self.ag_minint)
        self.ag_smaxrg = QtWidgets.QDoubleSpinBox()
        self.ag_smaxrg.setRange(0.3, 3.0)
        self.ag_smaxrg.setSingleStep(0.05)
        self.ag_smaxrg.setValue(1.3)
        self.ag_smaxrg.setToolTip("--smaxrg: largest q·Rg of the Guinier range (1.3 for globular particles)")
        f.addRow("q_max·Rg ≤", self.ag_smaxrg)
        self.ag_minrg = QtWidgets.QDoubleSpinBox()
        self.ag_minrg.setRange(0.0, 1e4)
        self.ag_minrg.setValue(1.0)
        self.ag_minrg.setToolTip("--minrg: smallest acceptable Rg (nm with q in 1/nm)")
        f.addRow("Rg ≥", self.ag_minrg)
        f.addRow(self._run_button("Run AUTORG", self.run_autorg, "Guinier analysis of the selected curve"))
        self.ag_result = QtWidgets.QLabel("")
        self.ag_result.setTextInteractionFlags(QtCore.Qt.TextInteractionFlag.TextSelectableByMouse)
        f.addRow(self.ag_result)
        tb.addItem(p, "Guinier (AUTORG)")

        # --- DATGNOM / GNOM
        p = QtWidgets.QWidget()
        f = QtWidgets.QFormLayout(p)
        f.addRow(self._run_button("Run DATGNOM (automatic Dmax)", self.run_datgnom,
                                  "p(r) with automatic Dmax; Rg from AUTORG (run first if missing)"))
        self.gn_dmax = QtWidgets.QDoubleSpinBox()
        self.gn_dmax.setRange(0.01, 1e5)
        self.gn_dmax.setDecimals(3)
        self.gn_dmax.setValue(10.0)
        self.gn_dmax.setToolTip("--rmax: Dmax (system 0) or largest size of the chosen system, nm")
        b = QtWidgets.QPushButton("from DATGNOM")
        b.clicked.connect(self._dmax_from_datgnom)
        r = QtWidgets.QHBoxLayout()
        r.addWidget(self.gn_dmax, 1)
        r.addWidget(b)
        f.addRow("Dmax", r)
        self.gn_system = QtWidgets.QComboBox()
        for v, t in GNOM_SYSTEMS:
            self.gn_system.addItem(t, v)
        f.addRow("system", self.gn_system)
        self.gn_rad56 = QtWidgets.QDoubleSpinBox()
        self.gn_rad56.setRange(0.0, 1e4)
        self.gn_rad56.setDecimals(3)
        self.gn_rad56.setToolTip("--rad56: cylinder radius (system 5) or relative shell thickness 0..1 (system 6)")
        f.addRow("rad56", self.gn_rad56)
        self.gn_zero_min = QtWidgets.QCheckBox("p(rmin) = 0")
        self.gn_zero_min.setChecked(True)
        self.gn_zero_max = QtWidgets.QCheckBox("p(Dmax) = 0")
        self.gn_zero_max.setChecked(True)
        r = QtWidgets.QHBoxLayout()
        r.addWidget(self.gn_zero_min)
        r.addWidget(self.gn_zero_max)
        f.addRow("force", r)
        self.gn_nr = QtWidgets.QSpinBox()
        self.gn_nr.setRange(0, 2000)
        self.gn_nr.setSpecialValueText("automatic")
        self.gn_nr.setToolTip("--nr: points in real space")
        f.addRow("r points", self.gn_nr)
        self.gn_alpha = QtWidgets.QDoubleSpinBox()
        self.gn_alpha.setRange(0.0, 1e6)
        self.gn_alpha.setDecimals(4)
        self.gn_alpha.setSpecialValueText("automatic")
        self.gn_alpha.setToolTip("--alpha: regularisation parameter (0 = GNOM chooses by its perceptual criteria)")
        f.addRow("α", self.gn_alpha)
        f.addRow(self._run_button("Run GNOM", self.run_gnom, "p(r) (or the system's distribution) with this Dmax"))
        self.gn_result = QtWidgets.QLabel("")
        self.gn_result.setTextInteractionFlags(QtCore.Qt.TextInteractionFlag.TextSelectableByMouse)
        self.gn_result.setWordWrap(True)
        f.addRow(self.gn_result)
        tb.addItem(p, "p(r) (DATGNOM, GNOM)")

        # --- CRYSOL / CRYSON
        p = QtWidgets.QWidget()
        f = QtWidgets.QFormLayout(p)
        self.cr_prog = QtWidgets.QComboBox()
        self.cr_prog.addItems(["crysol", "cryson"])
        self.cr_prog.setToolTip("CRYSOL: X-rays; CRYSON: neutrons (extra options, e.g. --D2O=…, below)")
        f.addRow("program", self.cr_prog)
        self.cr_model = QtWidgets.QLineEdit()
        self.cr_model.setPlaceholderText("PDB or mmCIF file")
        b = QtWidgets.QPushButton("…")
        b.setMaximumWidth(30)
        b.clicked.connect(self.on_choose_model)
        r = QtWidgets.QHBoxLayout()
        r.addWidget(self.cr_model, 1)
        r.addWidget(b)
        f.addRow("model", r)
        self.cr_lm = QtWidgets.QSpinBox()
        self.cr_lm.setRange(1, 100)
        self.cr_lm.setValue(20)
        f.addRow("harmonics (lm)", self.cr_lm)
        self.cr_ns = QtWidgets.QSpinBox()
        self.cr_ns.setRange(10, 5000)
        self.cr_ns.setValue(101)
        f.addRow("points (ns)", self.cr_ns)
        self.cr_smax = QtWidgets.QDoubleSpinBox()
        self.cr_smax.setRange(0.01, 10)
        self.cr_smax.setDecimals(3)
        self.cr_smax.setValue(0.5)
        self.cr_smax.setToolTip("--smax: largest q of the model curve, in the unit CRYSOL assumes (1/Å)")
        f.addRow("q max", self.cr_smax)
        self.cr_fit = QtWidgets.QCheckBox("fit to the selected curve")
        self.cr_fit.setChecked(True)
        f.addRow(self.cr_fit)
        self.cr_extra = QtWidgets.QLineEdit()
        self.cr_extra.setPlaceholderText("further options, e.g. --D2O=1.0 --units=2")
        f.addRow("options", self.cr_extra)
        f.addRow(self._run_button("Run", self.run_crysol, "model curve, or its fit to the selected curve"))
        self.cr_result = QtWidgets.QLabel("")
        f.addRow(self.cr_result)
        tb.addItem(p, "Model curve (CRYSOL, CRYSON)")

        # --- DAMMIF
        p = QtWidgets.QWidget()
        f = QtWidgets.QFormLayout(p)
        self.dm_runs = QtWidgets.QSpinBox()
        self.dm_runs.setRange(1, 200)
        self.dm_runs.setValue(int(prefs.get("atsas.dammif.nruns") or 10))
        f.addRow("runs", self.dm_runs)
        self.dm_mode = QtWidgets.QComboBox()
        self.dm_mode.addItems(["fast", "slow"])
        f.addRow("mode", self.dm_mode)
        self.dm_sym = QtWidgets.QComboBox()
        self.dm_sym.setEditable(True)
        self.dm_sym.addItems(["P1", "P2", "P3", "P4", "P5", "P6", "P222", "P32", "P42", "P52", "P62"])
        f.addRow("symmetry", self.dm_sym)
        self.dm_unit = QtWidgets.QComboBox()
        self.dm_unit.addItem("from the .out file", "")
        self.dm_unit.addItem("nm", "NANOMETER")
        self.dm_unit.addItem("Å", "ANGSTROM")
        self.dm_unit.setToolTip("--unit of the p(r) file; pySASfit curves are in 1/nm")
        f.addRow("unit", self.dm_unit)
        f.addRow(self._run_button("Run DAMMIF", self.run_dammif,
                                  "bead models from the GNOM p(r) of the selected curve (else DATGNOM)"))
        self.dm_result = QtWidgets.QLabel("")
        self.dm_result.setWordWrap(True)
        f.addRow(self.dm_result)
        tb.addItem(p, "Bead models (DAMMIF)")
        self.analysis_widget = w

    def _build_plots(self):
        self.canvas_iq = _canvas()
        self.iq_scale = QtWidgets.QComboBox()
        self.iq_scale.addItems(IQ_SCALES)
        self.iq_scale.currentIndexChanged.connect(lambda _i: self.draw_iq())
        self.iq_all = QtWidgets.QCheckBox("all curves")
        self.iq_all.toggled.connect(lambda _c: self.draw_iq())
        from ..core.flowlayout import flow_bar
        bar = flow_bar(QtWidgets.QLabel("scale"), self.iq_scale, self.iq_all)
        self.iq_panel = plot_panel(self.canvas_iq, self, bars=(bar,))
        self.canvas_pr = _canvas()
        self.pr_norm = QtWidgets.QCheckBox("normalised to max 1")
        self.pr_norm.toggled.connect(lambda _c: self.draw_pr())
        self.pr_panel = plot_panel(self.canvas_pr, self, bars=(flow_bar(self.pr_norm),))

    def _build_results_panel(self):
        t = QtWidgets.QTableWidget(0, len(COLUMNS))
        t.setHorizontalHeaderLabels(COLUMNS)
        t.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
        t.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows)
        t.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.ResizeMode.ResizeToContents)
        t.cellClicked.connect(lambda r, _c: self.curve_list.setCurrentRow(r))
        self.table = t

    def _build_programs_panel(self):
        w = QtWidgets.QWidget()
        lay = QtWidgets.QVBoxLayout(w)
        lay.setContentsMargins(4, 4, 4, 4)
        self.inst_label = QtWidgets.QLabel("")
        self.inst_label.setWordWrap(True)
        self.inst_label.setTextInteractionFlags(QtCore.Qt.TextInteractionFlag.TextSelectableByMouse)
        lay.addWidget(self.inst_label)
        row = QtWidgets.QHBoxLayout()
        b = QtWidgets.QPushButton("Search again")
        b.setToolTip("Look for the ATSAS installation again (after installing it or changing the preferences)")
        b.clicked.connect(lambda: self.refresh_installation(force=True))
        row.addWidget(b)
        self.selftest_button = self._run_button("Self test", self.run_selftest,
                                                "AUTORG, DATGNOM and GNOM on a synthetic sphere (R = 5 nm)")
        row.addWidget(self.selftest_button)
        row.addStretch(1)
        lay.addLayout(row)
        self.with_data = QtWidgets.QCheckBox("open with the selected curve / p(r)")
        self.with_data.setChecked(True)
        self.with_data.setToolTip("PRIMUS and SASPLOT get the curve, GNOMPLOT the last .out file of the curve")
        lay.addWidget(self.with_data)
        grid = QtWidgets.QGridLayout()
        self.program_buttons = {}
        for i, (label, exe, icon, desc) in enumerate(A.GUI_PROGRAMS):
            b = QtWidgets.QToolButton()
            b.setText(f"{label}\n{desc}")
            b.setToolButtonStyle(QtCore.Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
            b.setIconSize(QtCore.QSize(40, 40))
            b.setSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Preferred)
            b.setAutoRaise(False)
            b.clicked.connect(lambda _c=False, e=exe: self.on_launch(e, with_data=self.with_data.isChecked()))
            grid.addWidget(b, i, 0)
            self.program_buttons[exe] = (b, icon)
        lay.addLayout(grid)
        note = QtWidgets.QLabel("ATSAS © EMBL Hamburg / BIOSAXS GmbH, installed separately with its own licence. "
                                "Icons: the original ones of the installation.")
        note.setWordWrap(True)
        note.setStyleSheet("color: gray;")
        lay.addWidget(note)
        lay.addStretch(1)
        self.programs_widget = w

    # ------------------------------------------------------------------ installation
    def atsas_dir(self):
        return (prefs.get("atsas.dir") or "").strip() or None

    def refresh_installation(self, force=False):
        if force:
            A._ATSAS_BIN = None
            from .. import resources
            resources._EXTERNAL_CACHE.clear()
            self.setWindowIcon(tool_icon("pyATsas"))
        d = self.atsas_dir()
        try:
            b = A.find_atsas(d)
            lic = A.license_file(d)
            txt = (f"<b>ATSAS</b>: {A.atsas_root(d)}<br>licence: "
                   + (str(lic) if lic else "<span style='color:#c00'>atsas.lic not found</span>"))
            self.installed = True
        except A.AtsasError as e:
            b, txt = None, f"<span style='color:#c00'>{e}</span>"
            self.installed = False
        self.inst_label.setText(txt)
        for exe, (btn, icon) in self.program_buttons.items():
            files = A.icon_files(icon, d) if b else []
            ic = QtGui.QIcon()
            for size, path in files:
                ic.addFile(str(path), QtCore.QSize(size, size))
            btn.setIcon(ic)
            have = b is not None and A.program_path(exe, d) is not None
            btn.setEnabled(have)
            btn.setToolTip("" if have else f"{exe} is not in this ATSAS installation")
        self._update_buttons()

    # ------------------------------------------------------------------ data
    def on_load(self):
        files, _ = QtWidgets.QFileDialog.getOpenFileNames(
            self, "Load curves", getattr(self, "_last_dir", ""),
            "SAS curves (*.dat *.txt *.csv *.h5 *.nxs *.hdf5);;All files (*)")
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
                    if ok.sum() < 5:
                        continue
                    for k in ("q", "I", "dI", "dq"):
                        setattr(c, k, np.asarray(getattr(c, k))[ok])
                    self.curves.append(c)
                    self.curve_list.addItem(c.name)
            except Exception as e:  # noqa: BLE001 -- report, go on with the other files
                self.log(f"cannot read {fn}: {e}")
        if len(self.curves) > n0:
            self.curve_list.setCurrentRow(n0)
        self._fill_table()

    def on_remove(self):
        rows = sorted({i.row() for i in self.curve_list.selectedIndexes()}, reverse=True)
        for r in rows:
            c = self.curves.pop(r)
            self.results.pop(c.name, None)
            self.curve_list.takeItem(r)
        self._fill_table()
        self.draw_all()

    def current(self):
        r = self.curve_list.currentRow()
        return self.curves[r] if 0 <= r < len(self.curves) else None

    def _on_curve_changed(self, _row):
        c = self.current()
        if c is None:
            self.curve_info.setText("no data")
        else:
            n = len(c.q)
            for sp in (self.first, self.last):
                sp.blockSignals(True)
                sp.setRange(1, n)
            st = self.results.get(c.name, {}).get("range")
            self.first.setValue(st[0] if st else 1)
            self.last.setValue(st[1] if st else n)
            for sp in (self.first, self.last):
                sp.blockSignals(False)
            self.curve_info.setText(f"{n} points, q = {c.q.min():.4g} … {c.q.max():.4g} 1/nm")
            self.curve_info.setToolTip(c.path)
        self._show_results()
        self.draw_all()

    def _full_range(self):
        c = self.current()
        if c is not None:
            self.first.setValue(1)
            self.last.setValue(len(c.q))

    def used(self, c=None):
        """(q, I, dI) of the point range of curve ``c`` (default: the selected one), and the 0-based offset."""
        c = c or self.current()
        if c is None:
            raise A.AtsasError("no curve selected")
        if c is self.current():
            i0, i1 = self.first.value() - 1, self.last.value()
            self.results.setdefault(c.name, {})["range"] = (i0 + 1, i1)
        else:
            i0, i1 = (self.results.get(c.name, {}).get("range") or (1, len(c.q)))
            i0 -= 1
        if i1 - i0 < 5:
            raise A.AtsasError("fewer than 5 points in the range")
        dI = np.asarray(c.dI, float)[i0:i1]
        return (c.q[i0:i1], c.I[i0:i1], dI), i0

    # ------------------------------------------------------------------ runs
    def _run_dir(self, program, c):
        base = (prefs.get("atsas.workdir") or "").strip()
        if not base:
            base = os.path.join(os.path.dirname(prefs.settings_path()), "atsas_runs")
        stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S-%f")[:-3]
        d = os.path.join(base, f"{stamp}_{program}_{_safe(c.name if c else 'test')}")
        os.makedirs(d, exist_ok=True)
        return d

    def _start(self, tag, fn, text):
        if self._busy:
            return
        self._busy = True
        self._update_buttons()
        self.progress.setRange(0, 0)
        self.log(f"--- {text}")
        self.statusBar().showMessage(text + " …")

        def work():
            try:
                res, err = fn(), None
            except Exception as e:  # noqa: BLE001
                res, err = None, f"{type(e).__name__}: {e}"
                if not isinstance(e, A.AtsasError):
                    err += "\n" + traceback.format_exc()
            self._signals.done.emit(tag, res, err)

        threading.Thread(target=work, daemon=True).start()

    def _on_done(self, tag, res, err):
        self._busy = False
        self.progress.setRange(0, 1)
        self._update_buttons()
        if err:
            self.log(err)
            self.statusBar().showMessage(f"{tag[0]} failed", 8000)
            self._show_results()
            return
        kind, name = tag
        if kind == "multi":
            self.results.setdefault(name, {}).update(res)
        elif kind == "batch":
            for nm, r in res.items():
                self.results.setdefault(nm, {}).update(r)
        elif kind == "selftest":
            self.log(res)
        else:
            self.results.setdefault(name, {})[kind] = res
        for r in (res.values() if isinstance(res, dict) else [res]):
            for run in self._runs_of(r):
                self._log_run(run)
        self.statusBar().showMessage(f"{kind} done", 5000)
        self._fill_table()
        self._show_results()
        self.draw_all()

    @staticmethod
    def _runs_of(r):
        out = []
        if isinstance(r, dict):
            for v in r.values():
                out += MainWindow._runs_of(v)
            return out
        run = getattr(r, "run", None)
        if run is not None:
            out.append(run)
        out += list(getattr(r, "runs", []) or [])
        return out

    def _log_run(self, run):
        self.log("$ " + " ".join(f'"{a}"' if " " in str(a) else str(a) for a in run.cmd))
        self.log(f"  (in {run.cwd})")
        for s in (run.stdout, run.stderr):
            s = (s or "").strip()
            if s:
                self.log("\n".join(s.splitlines()[-30:]))

    def _common(self):
        return dict(atsas_dir=self.atsas_dir())

    def _timeout(self):
        return float(prefs.get("atsas.timeout") or 600)

    def run_autorg(self):
        c = self.current()
        try:
            data, off = self.used(c)
        except A.AtsasError as e:
            return self.log(str(e))
        kw = dict(mininterval=self.ag_minint.value(), smaxrg=self.ag_smaxrg.value(), minrg=self.ag_minrg.value())
        cwd = self._run_dir("autorg", c)

        def fn():
            r = A.autorg(data, cwd=cwd, **kw, **self._common())
            r.offset = off
            return r
        self._start(("autorg", c.name), fn, f"AUTORG {c.name}")

    def run_datgnom(self):
        c = self.current()
        try:
            data, off = self.used(c)
        except A.AtsasError as e:
            return self.log(str(e))
        cwd = self._run_dir("datgnom", c)
        g0 = self.results.get(c.name, {}).get("autorg")
        if g0 is not None and getattr(g0, "offset", None) != off:
            g0 = None                                    # AUTORG of another point range: run again
        common = self._common()

        def fn():
            g = g0
            if g is None:
                g = A.autorg(data, cwd=cwd, **common)
                g.offset = off
            return {"autorg": g, "datgnom": A.datgnom(data, rg=g.rg, cwd=cwd, **common)}
        self._start(("multi", c.name), fn, f"DATGNOM {c.name}")

    def _dmax_from_datgnom(self):
        c = self.current()
        p = self.results.get(c.name, {}).get("datgnom") if c else None
        if p is not None and np.isfinite(p.dmax):
            self.gn_dmax.setValue(p.dmax)

    def run_gnom(self):
        c = self.current()
        try:
            data, off = self.used(c)
        except A.AtsasError as e:
            return self.log(str(e))
        kw = dict(dmax=self.gn_dmax.value(), system=self.gn_system.currentData(),
                  force_zero_rmin=self.gn_zero_min.isChecked(), force_zero_rmax=self.gn_zero_max.isChecked(),
                  nr=self.gn_nr.value() or None, alpha=self.gn_alpha.value() or None,
                  rad56=self.gn_rad56.value() if self.gn_system.currentData() in (5, 6) else None)
        cwd = self._run_dir("gnom", c)
        self._start(("gnom", c.name), lambda: A.gnom(data, cwd=cwd, **kw, **self._common()), f"GNOM {c.name}")

    def on_choose_model(self):
        fn, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Atomic model", self.cr_model.text(),
                                                      "Models (*.pdb *.cif *.ent);;All files (*)")
        if fn:
            self.cr_model.setText(fn)

    def run_crysol(self):
        model = self.cr_model.text().strip()
        if not model or not os.path.isfile(model):
            return self.log("choose a PDB or mmCIF model first")
        c = self.current()
        data = None
        if self.cr_fit.isChecked():
            try:
                data, _off = self.used(c)
            except A.AtsasError as e:
                return self.log(str(e))
        prog = self.cr_prog.currentText()
        kw = dict(lm=self.cr_lm.value(), ns=self.cr_ns.value(), smax=self.cr_smax.value(),
                  extra=self.cr_extra.text().split(), program=prog)
        cwd = self._run_dir(prog, c)
        name = c.name if c is not None else os.path.basename(model)
        self._start(("crysol", name), lambda: A.crysol(model, data, cwd=cwd, **kw, **self._common()),
                    f"{prog.upper()} {os.path.basename(model)}")

    def run_dammif(self):
        c = self.current()
        if c is None:
            return self.log("no curve selected")
        r = self.results.get(c.name, {})
        p = r.get("gnom") or r.get("datgnom")
        if p is None:
            return self.log("run GNOM or DATGNOM for this curve first")
        extra = []
        if self.dm_unit.currentData():
            extra.append(f"--unit={self.dm_unit.currentData()}")
        kw = dict(nruns=self.dm_runs.value(), mode=self.dm_mode.currentText(), symmetry=self.dm_sym.currentText(),
                  extra=extra, timeout=self._timeout())
        cwd = self._run_dir("dammif", c)
        self._start(("dammif", c.name), lambda: A.dammif(p.out_file, cwd=cwd, **kw, **self._common()),
                    f"DAMMIF {self.dm_runs.value()} × {c.name}")

    def run_batch(self):
        curves = list(self.curves)
        if not curves:
            return self.log("no curves loaded")
        jobs = []
        for c in curves:
            try:
                data, off = self.used(c)
            except A.AtsasError as e:
                self.log(f"{c.name}: {e}")
                continue
            jobs.append((c, data, off, self._run_dir("batch", c)))

        def fn():
            out = {}
            for c, data, off, cwd in jobs:
                try:
                    g = A.autorg(data, cwd=cwd, **self._common())
                    g.offset = off
                    p = A.datgnom(data, rg=g.rg, cwd=cwd, **self._common())
                    out[c.name] = {"autorg": g, "datgnom": p}
                except A.AtsasError as e:
                    out[c.name] = {"error": str(e)}
            return out
        self._start(("batch", ""), fn, f"AUTORG + DATGNOM of {len(jobs)} curves")

    def run_selftest(self):
        import contextlib
        import io

        def fn():
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                A.selftest(self.atsas_dir())
            return buf.getvalue()
        self._start(("selftest", ""), fn, "self test (synthetic sphere)")

    def _update_buttons(self):
        ok = getattr(self, "installed", False) and not self._busy
        for b in getattr(self, "_run_buttons", []):
            b.setEnabled(ok)

    # ------------------------------------------------------------------ launching ATSAS programs
    def on_launch(self, exe, with_data=False):
        args = []
        c = self.current()
        if with_data and c is not None:
            r = self.results.get(c.name, {})
            if exe == "gnomplotqt":
                p = r.get("gnom") or r.get("datgnom")
                if p is not None:
                    args = [p.out_file]
            elif exe in ("primusqt", "sasplotqt"):
                try:
                    data, _off = self.used(c)
                    d = self._run_dir(exe, c)
                    args = [A.write_dat(os.path.join(d, _safe(os.path.splitext(c.name)[0]) + ".dat"), *data,
                                        title=c.name)]
                except (A.AtsasError, OSError) as e:
                    self.log(str(e))
        try:
            A.launch(exe, args, atsas_dir=self.atsas_dir())
            self.statusBar().showMessage(f"{exe} started", 5000)
        except (A.AtsasError, OSError) as e:
            self.log(f"{exe}: {e}")

    def on_open_runs(self):
        base = (prefs.get("atsas.workdir") or "").strip() or \
            os.path.join(os.path.dirname(prefs.settings_path()), "atsas_runs")
        os.makedirs(base, exist_ok=True)
        QtGui.QDesktopServices.openUrl(QtCore.QUrl.fromLocalFile(base))

    # ------------------------------------------------------------------ output
    def log(self, text):
        self.log_view.appendPlainText(str(text))

    def _row(self, c):
        r = self.results.get(c.name, {})
        g, p, q, m, d = r.get("autorg"), r.get("datgnom"), r.get("gnom"), r.get("crysol"), r.get("dammif")
        pts = ""
        if g is not None:
            o = getattr(g, "offset", 0)
            pts = f"{g.first + o}–{g.last + o}"
        return [c.name,
                f"{_fmt(g.rg)} ± {_fmt(g.rg_err, 2)}" if g else "", _fmt(g.i0) if g else "", pts,
                _fmt(g.quality, 3) if g else "",
                _fmt(p.dmax) if p else "", _fmt(p.rg_real) if p else "", _fmt(p.total_estimate, 3) if p else "",
                _fmt(q.dmax) if q else "", _fmt(q.rg_real) if q else "",
                _fmt(m.chi2) if m else "", str(len(d.models)) if d else ""]

    def _fill_table(self):
        t = self.table
        t.setRowCount(len(self.curves))
        for i, c in enumerate(self.curves):
            for j, v in enumerate(self._row(c)):
                it = QtWidgets.QTableWidgetItem(v)
                t.setItem(i, j, it)
            err = self.results.get(c.name, {}).get("error")
            if err:
                t.item(i, 0).setToolTip(err)

    def _show_results(self):
        c = self.current()
        r = self.results.get(c.name, {}) if c else {}
        g = r.get("autorg")
        self.ag_result.setText(
            f"Rg = {_fmt(g.rg)} ± {_fmt(g.rg_err, 2)} nm\nI(0) = {_fmt(g.i0)} ± {_fmt(g.i0_err, 2)}\n"
            f"points {g.first + getattr(g, 'offset', 0)}–{g.last + getattr(g, 'offset', 0)}, "
            f"quality {_fmt(g.quality, 3)}" if g else "")
        lines = []
        for k, lab in (("datgnom", "DATGNOM"), ("gnom", "GNOM")):
            p = r.get(k)
            if p is not None:
                lines.append(f"{lab}: Dmax = {_fmt(p.dmax)} nm, Rg = {_fmt(p.rg_real)} nm, I(0) = {_fmt(p.i0_real)}, "
                             f"total estimate {_fmt(p.total_estimate, 3)}")
        self.gn_result.setText("\n".join(lines))
        m = r.get("crysol")
        self.cr_result.setText(f"χ² = {_fmt(m.chi2)}" if m is not None and np.isfinite(m.chi2) else
                               ("model curve calculated" if m is not None else ""))
        d = r.get("dammif")
        self.dm_result.setText(f"{len(d.models)} models in\n{d.workdir}" if d else "")

    def draw_all(self):
        self.draw_iq()
        self.draw_pr()

    def _transform(self, q, I):
        s = self.iq_scale.currentIndex()
        with np.errstate(all="ignore"):
            if s == 2:
                return q ** 2, np.log(I)
            if s == 3:
                return q, q ** 2 * I
        return q, I

    def draw_iq(self):
        fig = self.canvas_iq.figure
        fig.clear()
        ax = fig.add_subplot(111)
        s = self.iq_scale.currentIndex()
        c0 = self.current()
        curves = self.curves if self.iq_all.isChecked() else ([c0] if c0 else [])
        for c in curves:
            x, y = self._transform(c.q, c.I)
            sel = c is c0
            ax.plot(x, y, "o", ms=2.5 if sel else 1.5, alpha=1 if sel else 0.5, label=c.name)
        if c0 is not None:
            n0, n1 = self.first.value() - 1, self.last.value()
            if n0 > 0 or n1 < len(c0.q):
                mask = np.ones(len(c0.q), bool)
                mask[n0:n1] = False
                x, y = self._transform(c0.q[mask], c0.I[mask])
                ax.plot(x, y, "x", color="0.6", ms=3, label="not used")
            r = self.results.get(c0.name, {})
            g = r.get("autorg")
            if g is not None and np.isfinite(g.rg):
                o = getattr(g, "offset", 0)
                i0, i1 = g.first - 1 + o, g.last + o
                x, y = self._transform(c0.q[i0:i1], c0.I[i0:i1])
                ax.plot(x, y, "o", mfc="none", mec="k", ms=5, label="Guinier range")
                qq = np.linspace(0, c0.q[min(i1, len(c0.q)) - 1] * 1.3, 50)[1:]
                x, y = self._transform(qq, g.i0 * np.exp(-(qq * g.rg) ** 2 / 3))
                ax.plot(x, y, "k-", lw=1, label=f"Guinier Rg = {_fmt(g.rg)} nm")
            for k, sty in (("datgnom", "r-"), ("gnom", "m--")):
                p = r.get(k)
                if p is not None and len(p.q):
                    x, y = self._transform(p.q, p.i_reg)
                    ax.plot(x, y, sty, lw=1.2, label=k.upper() + " fit")
            m = r.get("crysol")
            if m is not None and len(m.q):
                x, y = self._transform(m.q, m.i_model)
                ax.plot(x, y, "g-", lw=1.2, label="model" + (f" χ² = {_fmt(m.chi2, 3)}" if np.isfinite(m.chi2) else ""))
        labels = {0: ("q (1/nm)", "I(q)"), 1: ("q (1/nm)", "I(q)"), 2: ("q² (1/nm²)", "ln I"),
                  3: ("q (1/nm)", "q² I(q)"), 4: ("q (1/nm)", "I(q)")}[s]
        ax.set_xlabel(labels[0])
        ax.set_ylabel(labels[1])
        if s == 0:
            ax.set_xscale("log")
            ax.set_yscale("log")
        elif s == 1:
            ax.set_yscale("log")
        if s == 2 and c0 is not None:                     # Guinier plot: the low-q part
            g = self.results.get(c0.name, {}).get("autorg")
            if g is not None and np.isfinite(g.rg):
                ax.set_xlim(0, (2.0 / g.rg) ** 2)
        if ax.lines:
            ax.legend(fontsize="small")
        self.canvas_iq.draw_idle()

    def draw_pr(self):
        fig = self.canvas_pr.figure
        fig.clear()
        ax = fig.add_subplot(111)
        c = self.current()
        r = self.results.get(c.name, {}) if c else {}
        for k, col in (("datgnom", "r"), ("gnom", "m")):
            p = r.get(k)
            if p is None or not len(p.r):
                continue
            norm = np.nanmax(p.pr) if self.pr_norm.isChecked() and np.nanmax(p.pr) > 0 else 1.0
            ax.plot(p.r, p.pr / norm, color=col, label=f"{k.upper()} Dmax = {_fmt(p.dmax)} nm")
            ax.fill_between(p.r, (p.pr - p.pr_err) / norm, (p.pr + p.pr_err) / norm, color=col, alpha=0.2)
        ax.axhline(0, color="0.6", lw=0.6)
        if len(ax.lines) == 1:
            ax.text(0.5, 0.5, "no p(r) yet:\nDATGNOM or GNOM", transform=ax.transAxes, ha="center",
                    color="0.5")
            ax.set_yticks([])
        ax.set_xlabel("r (nm)")
        ax.set_ylabel("p(r)")
        if ax.lines[1:]:
            ax.legend(fontsize="small")
        self.canvas_pr.draw_idle()

    # ------------------------------------------------------------------ export
    def on_export_table(self):
        fn, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Results table", "atsas_results.csv", "CSV (*.csv)")
        if not fn:
            return
        import csv
        with open(fn, "w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(COLUMNS)
            for c in self.curves:
                w.writerow(self._row(c))
        self.statusBar().showMessage(f"saved {fn}", 5000)

    def on_export_pr(self):
        c = self.current()
        r = self.results.get(c.name, {}) if c else {}
        p = r.get("gnom") or r.get("datgnom")
        if p is None:
            return self.log("no p(r) for this curve")
        fn, _ = QtWidgets.QFileDialog.getSaveFileName(self, "p(r)", _safe(c.name) + "_pr.dat",
                                                      "text (*.dat *.txt)")
        if fn:
            np.savetxt(fn, np.c_[p.r, p.pr, p.pr_err], header=f"{c.name}: r (nm)  p(r)  error   "
                       f"Dmax = {p.dmax}  Rg = {p.rg_real}  (from {p.out_file})")

    def python_code(self):
        """Script: AUTORG, DATGNOM (and GNOM with the panel's Dmax) of the selected curve, with plots."""
        from ..core.pycode import Code, Script, call, path_lit
        c = self.current()
        if c is None:
            raise ValueError("Load a curve first.")
        sc = Script("pyATsas", f"Guinier analysis and p(r) of {c.name} with ATSAS")
        sc.imp("import numpy as np", "from pysasfit import atsas", "from pysasfit.toolbox.merge import read_curves")
        sc.comment("the curve (q in 1/nm) and the point range used")
        sc.add(f"curve = next(c for c in read_curves({path_lit(c.path)}) if c.name == {c.name!r})",
               f"i0, i1 = {self.first.value() - 1}, {self.last.value()}",
               "data = (curve.q[i0:i1], curve.I[i0:i1], curve.dI[i0:i1])")
        d = self.atsas_dir()
        kw = {"atsas_dir": d} if d else {}
        sc.comment("Guinier analysis and automatic p(r)")
        sc.add(call("g = atsas.autorg", Code("data"), mininterval=self.ag_minint.value(),
                    smaxrg=self.ag_smaxrg.value(), minrg=self.ag_minrg.value(), **kw),
               "print(f'Rg = {g.rg:.4g} +- {g.rg_err:.2g} nm, I(0) = {g.i0:.4g}')",
               call("p = atsas.datgnom", Code("data"), rg=Code("g.rg"), **kw),
               "print(f'DATGNOM: Dmax = {p.dmax:.4g} nm, Rg = {p.rg_real:.4g} nm, total estimate {p.total_estimate:.3g}')")
        sc.comment("p(r) with the Dmax and options of the panel")
        sc.add(call("p2 = atsas.gnom", Code("data"), dmax=self.gn_dmax.value(), system=self.gn_system.currentData(),
                    force_zero_rmin=self.gn_zero_min.isChecked(), force_zero_rmax=self.gn_zero_max.isChecked(),
                    nr=self.gn_nr.value() or None, alpha=self.gn_alpha.value() or None, **kw))
        sc.add_plot("fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))",
                    "ax1.loglog(curve.q, curve.I, 'o', ms=2, label=curve.name)",
                    "ax1.loglog(p.q, p.i_reg, 'r-', label='DATGNOM fit')",
                    "ax1.set_xlabel('q (1/nm)'); ax1.set_ylabel('I(q)'); ax1.legend()",
                    "ax2.plot(p.r, p.pr, 'r-', label='DATGNOM')",
                    "ax2.plot(p2.r, p2.pr, 'm--', label='GNOM')",
                    "ax2.set_xlabel('r (nm)'); ax2.set_ylabel('p(r)'); ax2.legend()")
        return sc.text()

    # ------------------------------------------------------------------ session (launcher hooks)
    @staticmethod
    def _session_path():
        return os.path.join(os.path.dirname(prefs.settings_path()), "atsas_session.json")

    def _options(self):
        return {"ag": [self.ag_minint.value(), self.ag_smaxrg.value(), self.ag_minrg.value()],
                "gn": [self.gn_dmax.value(), self.gn_system.currentIndex(), self.gn_rad56.value(),
                       self.gn_zero_min.isChecked(), self.gn_zero_max.isChecked(), self.gn_nr.value(),
                       self.gn_alpha.value()],
                "cr": [self.cr_prog.currentIndex(), self.cr_model.text(), self.cr_lm.value(), self.cr_ns.value(),
                       self.cr_smax.value(), self.cr_fit.isChecked(), self.cr_extra.text()],
                "dm": [self.dm_runs.value(), self.dm_mode.currentIndex(), self.dm_sym.currentText(),
                       self.dm_unit.currentIndex()],
                "iq_scale": self.iq_scale.currentIndex()}

    def _set_options(self, o):
        try:
            a = o.get("ag")
            if a:
                self.ag_minint.setValue(a[0]); self.ag_smaxrg.setValue(a[1]); self.ag_minrg.setValue(a[2])  # noqa: E702
            g = o.get("gn")
            if g:
                self.gn_dmax.setValue(g[0]); self.gn_system.setCurrentIndex(g[1]); self.gn_rad56.setValue(g[2])  # noqa: E702
                self.gn_zero_min.setChecked(g[3]); self.gn_zero_max.setChecked(g[4])  # noqa: E702
                self.gn_nr.setValue(g[5]); self.gn_alpha.setValue(g[6])  # noqa: E702
            c = o.get("cr")
            if c:
                self.cr_prog.setCurrentIndex(c[0]); self.cr_model.setText(c[1]); self.cr_lm.setValue(c[2])  # noqa: E702
                self.cr_ns.setValue(c[3]); self.cr_smax.setValue(c[4]); self.cr_fit.setChecked(c[5])  # noqa: E702
                self.cr_extra.setText(c[6])
            d = o.get("dm")
            if d:
                self.dm_runs.setValue(d[0]); self.dm_mode.setCurrentIndex(d[1])  # noqa: E702
                self.dm_sym.setCurrentText(d[2]); self.dm_unit.setCurrentIndex(d[3])  # noqa: E702
            self.iq_scale.setCurrentIndex(int(o.get("iq_scale", 0)))
        except (TypeError, ValueError, IndexError):
            pass

    def save_session(self):
        try:
            p = self._session_path()
            os.makedirs(os.path.dirname(p), exist_ok=True)
            files = []
            for c in self.curves:
                if c.path not in files:
                    files.append(c.path)
            ranges = {c.name: self.results.get(c.name, {}).get("range") for c in self.curves}
            s = {"version": 1, "files": files, "ranges": ranges, "current": self.curve_list.currentRow(),
                 "options": self._options(), "dock_state": self.dock_state()}
            with open(p, "w", encoding="utf-8") as fh:
                json.dump(s, fh, indent=1)
        except (OSError, TypeError, ValueError):
            pass

    def restore_session(self):
        try:
            with open(self._session_path(), encoding="utf-8") as fh:
                s = json.load(fh)
        except (OSError, ValueError):
            return False
        try:
            self._set_options(s.get("options") or {})
            files = [f for f in s.get("files") or [] if os.path.isfile(f)]
            for name, rng in (s.get("ranges") or {}).items():
                if rng:
                    self.results.setdefault(name, {})["range"] = tuple(rng)
            if files:
                self.load_files(files)
                cur = int(s.get("current", 0))
                if 0 <= cur < len(self.curves):
                    self.curve_list.setCurrentRow(cur)
            self._session_layout = bool(self.restore_dock_state(s.get("dock_state")))
        except Exception:  # noqa: BLE001 -- an old or broken session must not stop the tool
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
        self.shutdown()
        super().closeEvent(event)


def main(argv=None):
    argv = sys.argv if argv is None else argv
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(argv)
    from ..core.tabstyle import apply_tab_style
    apply_tab_style(app)
    app.setWindowIcon(tool_icon("pyATsas"))
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
