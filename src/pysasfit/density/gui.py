"""pyDENsity window: electron-density maps from SAXS/SANS curves with DENSS (T. D. Grant, Nature Methods 2018).

    pysasfit-density [data files]
    python -m pysasfit.density.gui [data files]

Also the "pyDENsity" tab of the pySASfit launcher. Panels (View menu; each can be moved, stacked or detached):

Data         curves (text, pySASred ASCII, NXcanSAS) and their point range, or a GNOM .out file; flat background
             of the high-q tail (I = A q^-B + bg, independent fit) and its subtraction -- DENSS fits no constant
DENSS        the installation found, self test; options of a single map (mode, Dmax, grid, steps, symmetry, seed)
             and of the averaged run (number of maps, cores, enantiomer selection)
I(q)         data and the scattering of the map (single map or average), residuals
Convergence  χ², Rg and support volume per step of the single map
Density      isosurface of the map in 3D (level by enclosed volume), with the support
Slices       three central slices of the map
Results      one row per run;  Log: command lines and DENSS output

DENSS is a separate GPL-3.0 package (pip install denss); pyDENsity runs it as a separate process and reads its
files. Every run keeps its own folder (Settings -> Preferences -> pyDENsity). q in 1/nm on the panels, maps in Å.
"""
from __future__ import annotations

import datetime
import json
import os
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
from . import core as D  # noqa: E402

__all__ = ["MainWindow", "main"]

Dock = QtCore.Qt.DockWidgetArea
COLUMNS = ["run", "curve", "kind", "Rg (Å)", "χ²", "support V (Å³)", "resolution (Å)", "steps / maps", "map file"]


def _canvas(size=(5, 4)):
    return FigureCanvasQTAgg(Figure(figsize=size, layout="constrained"))


def _fmt(v, n=4):
    try:
        v = float(v)
    except (TypeError, ValueError):
        return "" if v is None else str(v)
    return "" if not np.isfinite(v) else f"{v:.{n}g}"


def _safe(name, n=40):
    import re
    return re.sub(r"[^A-Za-z0-9_.-]", "_", name)[:n] or "curve"


class _Signals(QtCore.QObject):
    done = QtCore.Signal(object, object, object)


def _spin(lo, hi, val, dec=0, special=None, suffix="", tip=""):
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
    return sp


class MainWindow(DockMixin, QtWidgets.QMainWindow):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("pyDENsity – electron density from solution scattering (DENSS)")
        self.setWindowIcon(tool_icon("pyDENsity"))
        self.curves = []
        self.sources = []              # per list entry: ("curve", Curve) or ("file", path, units)
        self.runs = []                 # dicts: name, curve, kind ("single"/"average"), result
        self.derived = {}              # name of a background-subtracted curve -> (source name, bg)
        self.bg_fits = {}              # curve name -> PorodResult of the tail fit
        self.confirm_background = True    # ask before a run when the tail looks like a flat background
        self._busy = False
        self._signals = _Signals()
        self._signals.done.connect(self._on_done)
        self._build_menus()
        self._build_data_panel()
        self._build_denss_panel()
        self._build_plots()
        from ..atsas.view3d import ModelView
        self.view3d = ModelView()
        self.density_widget = self._density_panel(self.view3d)
        self.table = QtWidgets.QTableWidget(0, len(COLUMNS))
        self.table.setHorizontalHeaderLabels(COLUMNS)
        self.table.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.ResizeMode.ResizeToContents)
        self.table.currentCellChanged.connect(lambda r, *_a: self._select_run(r))
        self.log_view = QtWidgets.QPlainTextEdit()
        self.log_view.setReadOnly(True)
        self.log_view.setFont(QtGui.QFontDatabase.systemFont(QtGui.QFontDatabase.SystemFont.FixedFont))
        self._add_dock("data", "Data", self.data_widget, Dock.LeftDockWidgetArea, scrollable=True)
        self._add_dock("denss", "DENSS", self.denss_widget, Dock.LeftDockWidgetArea, scrollable=True)
        self._add_dock("density", "Density (3D)", self.density_widget, Dock.RightDockWidgetArea)
        self._add_dock("iq", "I(q)", self.iq_panel, Dock.RightDockWidgetArea)
        self._add_dock("conv", "Convergence", self.conv_panel, Dock.RightDockWidgetArea)
        self._add_dock("slices", "Slices", self.slice_panel, Dock.RightDockWidgetArea)
        self._add_dock("results", "Results", self.table, Dock.RightDockWidgetArea)
        self._add_dock("log", "Log", self.log_view, Dock.RightDockWidgetArea)
        self.splitDockWidget(self.docks["density"], self.docks["iq"], QtCore.Qt.Orientation.Horizontal)
        self.splitDockWidget(self.docks["density"], self.docks["results"], QtCore.Qt.Orientation.Vertical)
        self.tabifyDockWidget(self.docks["iq"], self.docks["conv"])
        self.tabifyDockWidget(self.docks["iq"], self.docks["slices"])
        self.tabifyDockWidget(self.docks["results"], self.docks["log"])
        self.docks["iq"].raise_()
        self.docks["results"].raise_()
        self._finish_docks()
        self.resize(1500, 920)
        self._sized = False
        self.hide_stray_tab_bars()
        self.refresh_installation()

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
                         [int(0.26 * w), int(0.40 * w), int(0.34 * w)], QtCore.Qt.Orientation.Horizontal)
        h = max(self.height(), 600)
        self.resizeDocks([self.docks["data"], self.docks["denss"]], [int(0.52 * h), int(0.48 * h)],
                         QtCore.Qt.Orientation.Vertical)
        self._start_layout = self.saveState()

    # ------------------------------------------------------------------ construction
    def _build_menus(self):
        m = self.menuBar().addMenu("&File")
        a = m.addAction("&Load data…", self.on_load)
        a.setShortcut(QtGui.QKeySequence("Ctrl+L"))
        m.addAction("Load GNOM .out…", self.on_load_out)
        m.addAction("Open a density map (.mrc)…", self.on_open_mrc)
        m.addSeparator()
        m.addAction("Export results table (CSV)…", self.on_export_table)
        m.addAction("Open the run folder", self.on_open_runs)
        from ..core import pycode
        pycode.install(self, self.python_code, menu=m)
        from ..core.settings_dialog import add_preferences_menu
        add_preferences_menu(self, "density")
        self.menuBar().addMenu("&View")

    def _build_data_panel(self):
        w = QtWidgets.QWidget()
        lay = QtWidgets.QVBoxLayout(w)
        lay.setContentsMargins(4, 4, 4, 4)
        row = QtWidgets.QHBoxLayout()
        for text, slot, tip in (("Load…", self.on_load, "Curves: text, pySASred ASCII, NXcanSAS; q in 1/nm"),
                                ("GNOM .out…", self.on_load_out,
                                 "A GNOM/DATGNOM .out file: DENSS uses its regularised curve and Dmax"),
                                ("Remove", self.on_remove, "")):
            b = QtWidgets.QPushButton(text)
            b.setToolTip(tip)
            b.clicked.connect(lambda _c=False, s=slot: s())
            row.addWidget(b)
        lay.addLayout(row)
        self.curve_list = QtWidgets.QListWidget()
        self.curve_list.currentRowChanged.connect(self._on_curve_changed)
        lay.addWidget(self.curve_list, 1)
        self.curve_info = QtWidgets.QLabel("no data")
        self.curve_info.setWordWrap(True)
        self.curve_info.setStyleSheet("color: gray;")
        lay.addWidget(self.curve_info)
        f = QtWidgets.QFormLayout()
        self.first = _spin(1, 1, 1)
        self.last = _spin(1, 1, 1)
        for sp in (self.first, self.last):
            sp.valueChanged.connect(lambda _v: self.draw_iq())
        r = QtWidgets.QHBoxLayout()
        r.addWidget(self.first)
        r.addWidget(QtWidgets.QLabel("…"))
        r.addWidget(self.last)
        f.addRow("points", r)
        lay.addLayout(f)
        lay.addWidget(self._build_background_group())
        self.data_widget = w

    def _build_background_group(self):
        """Flat background of the high-q tail, I = A q^-B + bg (independent least-squares fit, no ATSAS). DENSS fits
        no constant: neither its smoothing of raw data (Sasrec) nor the comparison of map and data (scale only)."""
        from ..atsas.gui import _FloatEdit
        g = QtWidgets.QGroupBox("Flat background (DENSS fits none)")
        g.setToolTip("DENSS compares map and data with a scale factor only and fits no constant: a flat background "
                     "(incoherent scattering, buffer mismatch) is taken as particle scattering. Fit and subtract it.")
        f = QtWidgets.QFormLayout(g)
        self.bg_qmin = _spin(0, 1e4, 0, dec=6, suffix=" 1/nm")     # 6 decimals: the tail starts exactly at a point
        self.bg_qmax = _spin(0, 1e4, 0, dec=6, suffix=" 1/nm")
        for sp in (self.bg_qmin, self.bg_qmax):
            sp.setSingleStep(0.1)
            sp.setMinimumWidth(70)
            sp.setSizePolicy(QtWidgets.QSizePolicy.Policy.Ignored, QtWidgets.QSizePolicy.Policy.Fixed)
            sp.valueChanged.connect(lambda _v: self.draw_iq())
        b = QtWidgets.QPushButton("30 %")
        b.setToolTip("Tail = the last 30 % of the points of the range")
        b.clicked.connect(self._default_tail)
        r = QtWidgets.QHBoxLayout()
        r.addWidget(self.bg_qmin, 1)
        r.addWidget(QtWidgets.QLabel("…"))
        r.addWidget(self.bg_qmax, 1)
        r.addWidget(b)
        f.addRow("tail q", r)
        self.bg_fix = QtWidgets.QCheckBox("exponent fixed at")
        self.bg_fix.setToolTip("B = 4: Porod (sharp interfaces); 2: Gaussian chains; free: fitted")
        self.bg_exp = _spin(0.5, 6.0, 4.0, dec=2)
        r = QtWidgets.QHBoxLayout()
        r.addWidget(self.bg_fix)
        r.addWidget(self.bg_exp)
        r.addStretch(1)
        f.addRow(r)
        b = QtWidgets.QPushButton("Fit background")
        b.setToolTip("Weighted least squares of I = A q^-B + bg over the tail q range")
        b.clicked.connect(lambda _c=False: self.fit_background())
        f.addRow(b)
        self.bg_result = QtWidgets.QLabel("")
        self.bg_result.setWordWrap(True)
        self.bg_result.setTextInteractionFlags(QtCore.Qt.TextInteractionFlag.TextSelectableByMouse)
        f.addRow(self.bg_result)
        self.bg_value = _FloatEdit()
        self.bg_value.setToolTip("Background subtracted from I (same unit as I); can be edited")
        self.bg_value.valueChanged.connect(lambda _v: self.draw_iq())
        b = QtWidgets.QPushButton("Subtract → new curve")
        b.setToolTip("Adds the curve I − bg (errors unchanged) to the list and selects it; the original stays")
        b.clicked.connect(self.on_subtract_background)
        r = QtWidgets.QHBoxLayout()
        r.addWidget(self.bg_value, 1)
        r.addWidget(b)
        f.addRow("bg", r)
        return g

    def _build_denss_panel(self):
        w = QtWidgets.QWidget()
        lay = QtWidgets.QVBoxLayout(w)
        lay.setContentsMargins(4, 4, 4, 4)
        self.inst = QtWidgets.QLabel("")
        self.inst.setWordWrap(True)
        self.inst.setTextInteractionFlags(QtCore.Qt.TextInteractionFlag.TextSelectableByMouse)
        lay.addWidget(self.inst)
        row = QtWidgets.QHBoxLayout()
        b = QtWidgets.QPushButton("Search again")
        b.clicked.connect(lambda: self.refresh_installation(force=True))
        row.addWidget(b)
        self.selftest_button = QtWidgets.QPushButton("Self test")
        self.selftest_button.setToolTip("One FAST map of a sphere R = 2 nm (exact Rg 15.49 Å); passes within 10 %")
        self.selftest_button.clicked.connect(self.run_selftest)
        row.addWidget(self.selftest_button)
        row.addStretch(1)
        lay.addLayout(row)
        g = QtWidgets.QGroupBox("Map")
        f = QtWidgets.QFormLayout(g)
        self.mode = QtWidgets.QComboBox()
        for t, v in (("FAST (simple shapes)", "FAST"), ("SLOW (complex molecules)", "SLOW"),
                     ("MEMBRANE (negative contrast allowed)", "MEMBRANE")):
            self.mode.addItem(t, v)
        f.addRow("mode", self.mode)
        self.dmax = _spin(0, 1e4, 0, dec=2, special="from the data", suffix=" nm",
                          tip="Maximum dimension. 'from the data': from the GNOM file, else estimated by DENSS")
        f.addRow("Dmax", self.dmax)
        self.oversampling = _spin(0, 10, 0, dec=1, special="default (3)",
                                  tip="-os: real-space box = oversampling × Dmax")
        f.addRow("oversampling", self.oversampling)
        self.nsamples = _spin(0, 512, 0, special="default", tip="-n: grid points per dimension (sets the voxel size)")
        f.addRow("grid points", self.nsamples)
        self.voxel = _spin(0, 100, 0, dec=2, special="default", suffix=" Å", tip="-v: voxel size (overrides grid points)")
        f.addRow("voxel", self.voxel)
        self.steps = _spin(0, 1_000_000, 0, special="automatic", tip="-s: maximum number of iterations")
        f.addRow("steps", self.steps)
        self.ncs = _spin(0, 60, 0, special="none", tip="-ncs: rotational symmetry (n-fold)")
        self.ncs_axis = QtWidgets.QComboBox()
        self.ncs_axis.addItems(["1 (longest)", "2", "3 (shortest)"])
        self.ncs_type = QtWidgets.QComboBox()
        for t, v in (("cyclic C", "C"), ("dihedral D", "D"), ("icosahedral", "icosahedral")):
            self.ncs_type.addItem(t, v)
        r = QtWidgets.QHBoxLayout()
        r.addWidget(self.ncs)
        r.addWidget(self.ncs_axis)
        r.addWidget(self.ncs_type)
        f.addRow("symmetry", r)
        self.seed = _spin(0, 2_000_000_000, 0, special="random", tip="--seed: for a reproducible map")
        f.addRow("seed", self.seed)
        self.gpu = QtWidgets.QCheckBox("GPU (needs CuPy)")
        f.addRow(self.gpu)
        self.extra = QtWidgets.QLineEdit()
        self.extra.setPlaceholderText("further DENSS options, e.g. --ne 10000")
        f.addRow("options", self.extra)
        self.run_one = QtWidgets.QPushButton("Run one map")
        self.run_one.setToolTip("denss: one reconstruction (FAST: seconds to a minute)")
        self.run_one.clicked.connect(self.run_single)
        f.addRow(self.run_one)
        lay.addWidget(g)
        g = QtWidgets.QGroupBox("Averaged map (denss-all)")
        f = QtWidgets.QFormLayout(g)
        self.nmaps = _spin(2, 1000, 20, tip="-nm: number of independent maps (20 is the DENSS default)")
        f.addRow("maps", self.nmaps)
        self.cores = _spin(1, 256, int(prefs.get("density.cores") or max(1, (os.cpu_count() or 2) - 1)),
                           tip="-j: maps computed in parallel")
        f.addRow("cores", self.cores)
        self.enan = QtWidgets.QCheckBox("select enantiomers before averaging")
        self.enan.setChecked(True)
        f.addRow(self.enan)
        self.run_all = QtWidgets.QPushButton("Run denss-all")
        self.run_all.setToolTip("maps, enantiomer selection, alignment, average, FSC resolution (minutes to hours)")
        self.run_all.clicked.connect(self.run_average)
        f.addRow(self.run_all)
        lay.addWidget(g)
        self.progress = QtWidgets.QProgressBar()
        self.progress.setRange(0, 1)
        self.progress.setTextVisible(False)
        self.progress.setMaximumHeight(8)
        lay.addWidget(self.progress)
        row = QtWidgets.QHBoxLayout()
        from ..core import pycode
        row.addWidget(pycode.button(self.python_code, w))
        row.addStretch(1)
        lay.addLayout(row)
        note = QtWidgets.QLabel("DENSS: T. D. Grant, Nature Methods 15, 191 (2018); GPL-3.0, installed separately "
                                "(pip install denss). Please cite it when you publish maps.")
        note.setWordWrap(True)
        note.setStyleSheet("color: gray;")
        lay.addWidget(note)
        lay.addStretch(1)
        self.denss_widget = w

    def _density_panel(self, view):
        w = QtWidgets.QWidget()
        lay = QtWidgets.QVBoxLayout(w)
        lay.setContentsMargins(2, 2, 2, 2)
        bar = QtWidgets.QHBoxLayout()
        bar.addWidget(QtWidgets.QLabel("enclosed volume"))
        self.level = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.level.setRange(20, 300)
        self.level.setValue(100)
        self.level.setToolTip("Isosurface level chosen so that it encloses this percentage of the DENSS support "
                              "volume (100 % = the support volume)")
        self.level.sliderReleased.connect(self.update_surface)
        self.level.valueChanged.connect(lambda v: self.level_label.setText(f"{v} %"))
        bar.addWidget(self.level, 1)
        self.level_label = QtWidgets.QLabel("100 %")
        bar.addWidget(self.level_label)
        self.show_support = QtWidgets.QCheckBox("support")
        self.show_support.setToolTip("Also draw the final support (translucent)")
        self.show_support.toggled.connect(lambda _c: self.update_surface())
        bar.addWidget(self.show_support)
        lay.addLayout(bar)
        lay.addWidget(view, 1)
        return w

    def _build_plots(self):
        self.canvas_iq = _canvas()
        self.iq_scale = QtWidgets.QComboBox()
        self.iq_scale.addItems(["log I vs log q", "log I vs q", "Kratky q² I"])
        self.iq_scale.currentIndexChanged.connect(lambda _i: self.draw_iq())
        from ..core.flowlayout import flow_bar
        self.iq_panel = plot_panel(self.canvas_iq, self, bars=(flow_bar(QtWidgets.QLabel("scale"), self.iq_scale),))
        self.canvas_conv = _canvas()
        self.conv_panel = plot_panel(self.canvas_conv, self)
        self.canvas_slices = _canvas((6, 2.4))
        self.slice_panel = plot_panel(self.canvas_slices, self)

    # ------------------------------------------------------------------ installation
    def python_path(self):
        return (prefs.get("density.python") or "").strip() or None

    def refresh_installation(self, force=False):
        try:
            info = D.find_denss(self.python_path(), refresh=force)
            self.inst.setText(f"<b>DENSS {info.version}</b><br>{info.python}")
            self.installed = True
        except D.DenssError as e:
            self.inst.setText(f"<span style='color:#c00'>{str(e).splitlines()[0]}</span><br>"
                              + "<br>".join(str(e).splitlines()[1:3]))
            self.installed = False
        self._update_buttons()

    def _update_buttons(self):
        ok = getattr(self, "installed", False) and not self._busy
        for b in (self.run_one, self.run_all, self.selftest_button):
            b.setEnabled(ok)

    # ------------------------------------------------------------------ data
    def on_load(self):
        files, _ = QtWidgets.QFileDialog.getOpenFileNames(self, "Load curves", getattr(self, "_last_dir", ""),
                                                          "SAS curves (*.dat *.txt *.csv *.h5 *.nxs);;All files (*)")
        if files:
            self._last_dir = os.path.dirname(files[0])
            self.load_files(files)

    def load_files(self, files):
        from ..toolbox.merge import read_curves
        n0 = len(self.sources)
        for fn in files:
            try:
                for c in read_curves(fn):
                    ok = np.isfinite(c.q) & np.isfinite(c.I) & (c.q > 0)
                    if ok.sum() < 5:
                        continue
                    for k in ("q", "I", "dI", "dq"):
                        setattr(c, k, np.asarray(getattr(c, k))[ok])
                    self.sources.append(("curve", c))
                    self.curve_list.addItem(c.name)
            except Exception as e:  # noqa: BLE001
                self.log(f"cannot read {fn}: {e}")
        if len(self.sources) > n0:
            self.curve_list.setCurrentRow(n0)

    def on_load_out(self, path=None, units=None):
        if path is None:
            path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "GNOM .out", getattr(self, "_last_dir", ""),
                                                            "GNOM (*.out);;All files (*)")
            if not path:
                return
            units = QtWidgets.QInputDialog.getItem(self, "q unit of the .out file", "q in", ["1/nm", "1/Å"], 0,
                                                   False)[0]
        u = "a" if units in ("1/Å", "a", "A") else "nm"
        self.sources.append(("file", str(path), u))
        self.curve_list.addItem(f"{Path(path).name} (GNOM, q in {'1/Å' if u == 'a' else '1/nm'})")
        self.curve_list.setCurrentRow(len(self.sources) - 1)

    def on_remove(self):
        r = self.curve_list.currentRow()
        if 0 <= r < len(self.sources):
            self.sources.pop(r)
            self.curve_list.takeItem(r)

    def current(self):
        r = self.curve_list.currentRow()
        return self.sources[r] if 0 <= r < len(self.sources) else None

    def _name(self, src):
        return src[1].name if src[0] == "curve" else Path(src[1]).name

    def _on_curve_changed(self, _row):
        s = self.current()
        if s is None:
            self.curve_info.setText("no data")
        elif s[0] == "curve":
            c = s[1]
            n = len(c.q)
            for sp in (self.first, self.last):
                sp.blockSignals(True)
                sp.setRange(1, n)
                sp.setEnabled(True)
            self.first.setValue(1)
            self.last.setValue(n)
            for sp in (self.first, self.last):
                sp.blockSignals(False)
            self.curve_info.setText(f"{n} points, q = {c.q.min():.4g} … {c.q.max():.4g} 1/nm"
                                    + (f"\nbackground {self.derived[c.name][1]:.6g} subtracted"
                                       if c.name in self.derived else ""))
            if self.bg_qmax.value() <= self.bg_qmin.value() or self.bg_qmin.value() > c.q.max() \
                    or self.bg_qmax.value() < c.q.min():
                self._default_tail()
            fit = self.bg_fits.get(c.name)
            self.bg_value.setValue(fit.bg if fit is not None else 0.0)
            self._show_bg(c.name)
        else:
            for sp in (self.first, self.last):
                sp.setEnabled(False)
            self.curve_info.setText(f"GNOM file {s[1]}: DENSS uses its regularised curve and Dmax")
        self.draw_iq()

    def _data(self):
        s = self.current()
        if s is None:
            raise D.DenssError("no data selected")
        if s[0] == "file":
            return s[1], s[2]
        c = s[1]
        i0, i1 = self.first.value() - 1, self.last.value()
        if i1 - i0 < 10:
            raise D.DenssError("fewer than 10 points in the range")
        return (c.q[i0:i1], c.I[i0:i1], np.asarray(c.dI, float)[i0:i1]), "nm"

    # ------------------------------------------------------------------ flat background
    def _default_tail(self):
        s = self.current()
        if s is None or s[0] != "curve":
            return
        c = s[1]
        i1 = self.last.value()
        i0 = max(self.first.value() - 1, int(i1 - 0.3 * (i1 - self.first.value() + 1)))
        self.bg_qmin.setValue(float(c.q[i0]))
        self.bg_qmax.setValue(float(c.q[i1 - 1]))

    def fit_background(self):
        s = self.current()
        if s is None or s[0] != "curve":
            return self.log("select a curve (GNOM files are already regularised curves)")
        c = s[1]
        if self.bg_qmax.value() <= self.bg_qmin.value():
            self._default_tail()
        sel = (c.q >= self.bg_qmin.value() * (1 - 1e-9)) & (c.q <= self.bg_qmax.value() * (1 + 1e-9))
        if sel.sum() < 5:
            return self.log("fewer than 5 points in the tail q range")
        from ..atsas.core import fit_power_background
        try:
            fit = fit_power_background(c.q[sel], c.I[sel], np.asarray(c.dI, float)[sel],
                                       exponent=self.bg_exp.value() if self.bg_fix.isChecked() else None)
        except Exception as e:  # noqa: BLE001
            return self.log(f"✖ background fit: {e}")
        self.bg_fits[c.name] = fit
        self.bg_value.setValue(fit.bg)
        self._show_bg(c.name)
        self.log(f"{c.name}: tail {self.bg_qmin.value():.4g} … {self.bg_qmax.value():.4g} 1/nm ({sel.sum()} points): "
                 f"bg = {fit.bg:.6g} ± {fit.dbg:.2g}, B = {fit.b:.4g} ± {fit.db:.2g}, χ²/n = {fit.chi2:.3g}")
        self.draw_iq()

    def _show_bg(self, name):
        fit = self.bg_fits.get(name)
        if fit is None:
            self.bg_result.setText("")
            return
        txt = (f"bg = {_fmt(fit.bg)} ± {_fmt(fit.dbg, 2)}, B = {_fmt(fit.b, 3)}"
               + (f" ± {_fmt(fit.db, 2)}" if fit.db else " (fixed)") + f", χ²/n = {_fmt(fit.chi2, 3)}")
        if np.isfinite(fit.dbg) and abs(fit.bg) < 2 * fit.dbg:
            txt += "\nbg not significant (< 2σ): probably no background to subtract"
        if fit.chi2 > 5:
            txt += ("\n⚠ χ²/n > 5: power law + constant does not describe the tail alone (form-factor oscillations "
                    "still in the range, or a wrong exponent) — check the fit in the I(q) plot; bg may still be fine")
        self.bg_result.setText(txt)

    def subtract_background(self, c, bg):
        """Add the curve I − bg of ``c`` (the original stays) and return it."""
        import dataclasses
        entry = (c.entry + " " if c.entry else "") + f"minus bg {bg:.4g}"
        new = dataclasses.replace(c, I=np.asarray(c.I, float) - bg, meta=dict(c.meta), entry=entry)
        for i, s in enumerate(self.sources):
            if s[0] == "curve" and s[1].name == new.name:
                return s[1]
        self.sources.append(("curve", new))
        self.curve_list.addItem(new.name)
        self.derived[new.name] = (c.name, bg)
        return new

    def on_subtract_background(self):
        s = self.current()
        if s is None or s[0] != "curve":
            return
        bg = self.bg_value.value()
        if bg == 0:
            return self.log("background is 0: fit it first or enter a value")
        new = self.subtract_background(s[1], bg)
        self.curve_list.setCurrentRow(next(i for i, x in enumerate(self.sources) if x[0] == "curve" and x[1] is new))
        self.log(f"{new.name}: I − {bg:.6g}")
        self._status(f"background {bg:.4g} subtracted: {new.name}", "ok")

    # ------------------------------------------------------------------ runs
    def _run_dir(self, kind, name):
        base = (prefs.get("density.workdir") or "").strip() or \
            os.path.join(os.path.dirname(prefs.settings_path()), "denss_runs")
        stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S-%f")[:-3]
        d = os.path.join(base, f"{stamp}_{kind}_{_safe(name)}")
        os.makedirs(d, exist_ok=True)
        return d

    def _options(self):
        return dict(dmax_nm=self.dmax.value() or None, mode=self.mode.currentData(),
                    nsamples=self.nsamples.value() or None, voxel=self.voxel.value() or None,
                    oversampling=self.oversampling.value() or None, steps=self.steps.value() or None,
                    ncs=self.ncs.value(), ncs_axis=self.ncs_axis.currentIndex() + 1,
                    ncs_type=self.ncs_type.currentData(), gpu=self.gpu.isChecked(),
                    extra=self.extra.text().split())

    def _start(self, tag, fn, text):
        if self._busy:
            return
        self._busy = True
        self._update_buttons()
        self.progress.setRange(0, 0)
        self.log(f"--- {text}")
        self._status(text + " …", "run")

        def work():
            try:
                res, err = fn(), None
            except Exception as e:  # noqa: BLE001
                res, err = None, f"{type(e).__name__}: {e}"
                if not isinstance(e, D.DenssError):
                    err += "\n" + traceback.format_exc()
            self._signals.done.emit(tag, res, err)

        threading.Thread(target=work, daemon=True).start()

    def _background_ok(self):
        """Before a run: a curve whose tail looks like power law + a significant constant -> ask (or only log)."""
        s = self.current()
        if s is None or s[0] != "curve" or s[1].name in self.derived:
            return True
        c = s[1]
        i0, i1 = self.first.value() - 1, self.last.value()
        k = max(i0, int(i1 - 0.3 * (i1 - i0)))
        try:
            from ..atsas.core import fit_power_background
            fit = fit_power_background(c.q[k:i1], c.I[k:i1], np.asarray(c.dI, float)[k:i1])
        except Exception:  # noqa: BLE001 -- no verdict
            return True
        tail = float(np.median(c.I[max(k, i1 - 10):i1]))
        if not (np.isfinite(fit.bg) and fit.bg > 3 * (fit.dbg if np.isfinite(fit.dbg) else np.inf)
                and tail > 0 and fit.bg > 0.1 * tail):
            return True
        msg = (f"The tail of {c.name} looks like a power law plus a flat background of about {fit.bg:.4g} "
               f"({100 * fit.bg / tail:.0f} % of the last intensities, B = {fit.b:.3g}). DENSS fits no constant and "
               "would take it as particle scattering. Fit and subtract it in the Data panel (Flat background).")
        self.log(f"⚠ {msg}")
        if not self.confirm_background:
            return True
        ans = QtWidgets.QMessageBox.question(self, "Flat background?", msg + "\n\nRun DENSS anyway?",
                                             QtWidgets.QMessageBox.StandardButton.Yes
                                             | QtWidgets.QMessageBox.StandardButton.No,
                                             QtWidgets.QMessageBox.StandardButton.No)
        return ans == QtWidgets.QMessageBox.StandardButton.Yes

    def run_single(self):
        try:
            data, units = self._data()
        except D.DenssError as e:
            return self.log(str(e))
        if not self._background_ok():
            return
        name = self._name(self.current())
        kw = self._options()
        kw["seed"] = self.seed.value() or None
        cwd = self._run_dir("denss", name)
        py, tmo = self.python_path(), float(prefs.get("density.timeout") or 7200)
        self._start(("single", name), lambda: D.run_denss(data, units_in=units, python=py, cwd=cwd, timeout=tmo, **kw),
                    f"DENSS {kw['mode']} map of {name}")

    def run_average(self):
        try:
            data, units = self._data()
        except D.DenssError as e:
            return self.log(str(e))
        if not self._background_ok():
            return
        name = self._name(self.current())
        kw = self._options()
        cwd = self._run_dir("denss-all", name)
        py = self.python_path()
        tmo = 4 * float(prefs.get("density.timeout") or 7200)
        n, j, en = self.nmaps.value(), self.cores.value(), self.enan.isChecked()
        self._start(("average", name),
                    lambda: D.run_denss_all(data, nmaps=n, cores=j, enantiomers=en, units_in=units, python=py, cwd=cwd,
                                            timeout=tmo, **kw),
                    f"denss-all: {n} maps of {name} on {j} cores")

    def run_selftest(self):
        import io
        py = self.python_path()
        cwd = self._run_dir("selftest", "sphere")

        def fn():
            buf = io.StringIO()
            ok = D.selftest(py, cwd=cwd, out=lambda s: buf.write(s + "\n"))
            return {"text": buf.getvalue(), "ok": ok}
        self._start(("selftest", ""), fn, "DENSS self test")

    def _on_done(self, tag, res, err):
        self._busy = False
        self.progress.setRange(0, 1)
        self._update_buttons()
        kind, name = tag
        if err:
            self.log(f"✖ {kind} {name}: {err}")
            self._status(f"✖ {kind} failed: {err.strip().splitlines()[0]}", "error")
            self._raise("log")
            return
        if kind == "selftest":
            self.log(res["text"])
            self._status("self test " + ("OK" if res["ok"] else "PROBLEM (see Log)"), "ok" if res["ok"] else "warning")
            return
        run = res.run or {}
        self.log("$ " + " ".join(str(a) for a in run.get("cmd", [])) + f"\n  (in {run.get('cwd')}, "
                 f"{run.get('seconds', 0):.1f} s, {run.get('python')})")
        out = (run.get("stdout") or "").strip()
        if out:
            self.log("\n".join(out.splitlines()[-15:]))
        self.runs.append({"name": f"{len(self.runs) + 1}", "curve": name, "kind": kind, "result": res})
        for w in res.warnings:
            self.log(f"⚠ {name}: {w}")
        self._fill_table()
        self.table.selectRow(len(self.runs) - 1)
        self._select_run(len(self.runs) - 1)
        self._status(f"{kind} map of {name} done" + (f" – {res.warnings[0]}" if res.warnings else ""),
                     "warning" if res.warnings else "ok")

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
        for i, r in enumerate(self.runs):
            x = r["result"]
            single = r["kind"] == "single"
            vals = [r["name"], r["curve"], "one map" if single else "average",
                    _fmt(x.rg) if single else f"{_fmt(x.rg_mean)} ± {_fmt(x.rg_sd, 2)}", _fmt(x.chi2),
                    _fmt(x.support_volume) if single else "",
                    "" if single else f"{_fmt(x.resolution, 3)} ± {_fmt(x.resolution_err, 2)}",
                    str(x.steps) if single else str(len(x.maps)), str(x.map_file)]
            for j, v in enumerate(vals):
                it = QtWidgets.QTableWidgetItem(v)
                if j == 0 and x.warnings:
                    it.setText("⚠ " + v)
                    it.setToolTip("\n".join(x.warnings))
                t.setItem(i, j, it)

    def selected_run(self):
        r = self.table.currentRow()
        return self.runs[r] if 0 <= r < len(self.runs) else (self.runs[-1] if self.runs else None)

    def _select_run(self, _row):
        self.update_surface()
        self.draw_iq()
        self.draw_conv()
        self.draw_slices()

    def update_surface(self):
        r = self.selected_run()
        if r is None:
            return
        x = r["result"]
        vol = getattr(x, "support_volume", np.nan)
        if not np.isfinite(vol):                 # average: the volume of voxels above 10 % of the maximum
            vol = float(np.sum(x.rho > 0.1 * x.rho.max()) * np.prod(x.voxel))
        frac = self.level.value() / 100
        try:
            lev = D.level_for_volume(x.rho, x.voxel, frac * vol)
            v, f = D.isosurface(x.rho, lev, x.voxel)
        except Exception as e:  # noqa: BLE001
            return self.log(f"✖ isosurface: {e}")
        title = f"run {r['name']}: {r['curve']} ({'one map' if r['kind'] == 'single' else 'average'})"
        info = (f"{title}\nlevel {lev:.4g} encloses {frac * vol:.4g} Å³ ({self.level.value()} % of "
                f"{vol:.4g} Å³)\ngrid {x.rho.shape}, voxel {x.voxel[0]:.3g} Å\n{x.map_file}")
        self.view3d.clear()
        self.view3d.add_mesh(title, v, f, info=info)
        sup = getattr(x, "support", None)
        if self.show_support.isChecked() and sup is not None and sup.any():
            sv, sf = D.isosurface(sup.astype(float), 0.5, x.voxel)
            self.view3d.add_mesh("support", sv, sf, translucent=True, info="final DENSS support")

    def draw_iq(self):
        fig = self.canvas_iq.figure
        fig.clear()
        gs = fig.add_gridspec(2, 1, height_ratios=[3, 1])
        ax = fig.add_subplot(gs[0])
        axr = fig.add_subplot(gs[1], sharex=ax)
        s = self.iq_scale.currentIndex()

        def tr(q, I):
            return (q, q ** 2 * I) if s == 2 else (q, I)
        src = self.current()
        if src is not None and src[0] == "curve":
            c = src[1]
            ax.plot(*tr(c.q, c.I), "o", ms=2, color="0.5", label=c.name)
            fit = self.bg_fits.get(c.name)
            if fit is not None:
                qq = np.geomspace(max(self.bg_qmin.value(), 1e-6), max(self.bg_qmax.value(), 1e-5), 80)
                ax.plot(*tr(qq, fit.model(qq)), "--", color="#b36b00", lw=1.4,
                        label=f"tail fit: B = {_fmt(fit.b, 3)}, bg = {_fmt(fit.bg)}")
            if self.bg_value.value() and s in (0, 1):
                ax.axhline(self.bg_value.value(), color="#b36b00", ls=":", lw=0.9, label="bg to subtract")
        r = self.selected_run()
        if r is not None and len(r["result"].fit_q):
            x = r["result"]
            ax.errorbar(*tr(x.fit_q, x.fit_I), yerr=(x.fit_err * (x.fit_q ** 2 if s == 2 else 1)), fmt="o", ms=2,
                        color="k", alpha=0.5, label="data used by DENSS")
            ax.plot(*tr(x.fit_q, x.fit_map), "-", color="#2B4C9B", lw=1.5,
                    label=f"map, χ² = {_fmt(x.chi2, 3)}")
            with np.errstate(all="ignore"):
                axr.plot(x.fit_q, (x.fit_I - x.fit_map) / x.fit_err, ".", ms=2, color="#2B4C9B")
            axr.axhline(0, color="0.5", lw=0.6)
        axr.set_xlabel("q (1/nm)")
        axr.set_ylabel("Δ/σ")
        ax.set_ylabel("q² I(q)" if s == 2 else "I(q)")
        if s == 0:
            ax.set_xscale("log")
            from matplotlib.ticker import NullFormatter
            for a_ in (ax, axr):                           # narrow decades: only major labels, no overlap
                a_.xaxis.set_minor_formatter(NullFormatter())
        if s in (0, 1):
            ax.set_yscale("log")
        ax.tick_params(labelbottom=False)
        if ax.lines:
            ax.legend(fontsize="small")
        self.canvas_iq.draw_idle()

    def draw_conv(self):
        fig = self.canvas_conv.figure
        fig.clear()
        r = self.selected_run()
        if r is None or r["kind"] != "single" or not len(r["result"].stats[0]):
            ax = fig.add_subplot(111)
            ax.text(0.5, 0.5, "convergence of a single map\n(not written for averages)", ha="center",
                    transform=ax.transAxes, color="0.5")
            ax.set_axis_off()
            self.canvas_conv.draw_idle()
            return
        chi, rg, vol = r["result"].stats
        axs = fig.subplots(3, 1, sharex=True)
        axs[0].semilogy(chi, color="#2B4C9B")
        axs[0].set_ylabel("χ²")
        axs[1].plot(rg, color="#2B4C9B")
        axs[1].set_ylabel("Rg (Å)")
        axs[2].semilogy(vol, color="#2B4C9B")
        axs[2].set_ylabel("support V (Å³)")
        axs[2].set_xlabel("step")
        self.canvas_conv.draw_idle()

    def draw_slices(self):
        fig = self.canvas_slices.figure
        fig.clear()
        r = self.selected_run()
        if r is None:
            self.canvas_slices.draw_idle()
            return
        x = r["result"]
        rho = x.rho
        n = np.array(rho.shape) // 2
        vmax = float(np.max(np.abs(rho))) or 1.0
        axs = fig.subplots(1, 3)
        for ax, sl, lab in zip(axs, (rho[n[0], :, :], rho[:, n[1], :], rho[:, :, n[2]]),
                               ("x = 0 (y, z)", "y = 0 (x, z)", "z = 0 (x, y)")):
            ext = [-sl.shape[1] * x.voxel[0] / 2, sl.shape[1] * x.voxel[0] / 2,
                   -sl.shape[0] * x.voxel[0] / 2, sl.shape[0] * x.voxel[0] / 2]
            im = ax.imshow(sl, origin="lower", cmap="RdBu_r" if rho.min() < -0.05 * vmax else "viridis",
                           vmin=-vmax if rho.min() < -0.05 * vmax else 0, vmax=vmax, extent=ext)
            ax.set_title(lab, fontsize="small")
            ax.set_xlabel("Å")
        fig.colorbar(im, ax=axs, shrink=0.8, label="density")
        self.canvas_slices.draw_idle()

    # ------------------------------------------------------------------ files
    def on_open_mrc(self):
        fn, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Density map", "", "MRC (*.mrc *.map *.ccp4);;All (*)")
        if not fn:
            return
        try:
            rho, vox, _h = D.read_mrc(fn)
        except D.DenssError as e:
            return self.log(f"✖ {e}")
        res = D.DenssResult(rho, vox, None, np.array([]), np.array([]), np.array([]), np.array([]), np.nan, np.nan,
                            np.nan, 0, (np.array([]),) * 3, Path(fn), Path(fn).parent)
        self.runs.append({"name": f"{len(self.runs) + 1}", "curve": Path(fn).name, "kind": "single", "result": res})
        self._fill_table()
        self.table.selectRow(len(self.runs) - 1)

    def on_export_table(self):
        fn, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Results", "denss_results.csv", "CSV (*.csv)")
        if not fn:
            return
        import csv
        with open(fn, "w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(COLUMNS)
            for i in range(self.table.rowCount()):
                w.writerow([self.table.item(i, j).text() if self.table.item(i, j) else ""
                            for j in range(len(COLUMNS))])

    def on_open_runs(self):
        r = self.selected_run()
        base = str(r["result"].workdir) if r else ((prefs.get("density.workdir") or "").strip()
                                                   or os.path.join(os.path.dirname(prefs.settings_path()), "denss_runs"))
        os.makedirs(base, exist_ok=True)
        QtGui.QDesktopServices.openUrl(QtCore.QUrl.fromLocalFile(base))

    def python_code(self):
        from ..core.pycode import Code, Script, call, path_lit
        s = self.current()
        if s is None:
            raise ValueError("Load a curve first.")
        sc = Script("pyDENsity", f"DENSS electron-density map of {self._name(s)}")
        sc.imp("import numpy as np", "from pysasfit import density")
        kw = self._options()
        kw = {k: v for k, v in kw.items() if v not in (None, 0, False, [], "")}
        if self.python_path():
            kw["python"] = self.python_path()
        if s[0] == "curve":
            root, bg_total = s[1].name, 0.0
            while root in self.derived:                     # background-subtracted curve: back to the file
                root, b_ = self.derived[root]
                bg_total += b_
            sc.imp("from pysasfit.toolbox.merge import read_curves")
            sc.add(f"curve = next(c for c in read_curves({path_lit(s[1].path)}) if c.name == {root!r})")
            if bg_total or root in self.bg_fits:
                sc.imp("from pysasfit.atsas.core import fit_power_background")
                sc.comment("flat background of the high-q tail (DENSS fits no constant): I = A q^-B + bg")
                sc.add(f"tail = (curve.q >= {self.bg_qmin.value()!r}) & (curve.q <= {self.bg_qmax.value()!r})",
                       call("fit = fit_power_background", Code("curve.q[tail]"), Code("curve.I[tail]"),
                            Code("curve.dI[tail]"), exponent=self.bg_exp.value() if self.bg_fix.isChecked() else None),
                       "print(f'bg = {fit.bg:.4g} +- {fit.dbg:.2g}, B = {fit.b:.3g}')")
            sc.add(f"bg = {bg_total!r}   # subtracted on the panel (fit.bg is the fitted value)" if bg_total
                   else "bg = 0.0   # no background subtracted",
                   f"i0, i1 = {self.first.value() - 1}, {self.last.value()}",
                   "data = (curve.q[i0:i1], curve.I[i0:i1] - bg, curve.dI[i0:i1])   # q in 1/nm")
            src = Code("data")
        else:
            src = Code(path_lit(s[1]))
            kw["units_in"] = s[2]
        sc.comment("one map (DENSS runs as a separate process; the files stay in r.workdir)")
        sc.add(call("r = density.run_denss", src, seed=self.seed.value() or None, **kw),
               "print(f'Rg {r.rg:.4g} Å, chi2 {r.chi2:.3g}, support volume {r.support_volume:.4g} Å³, {r.workdir}')")
        sc.comment(f"averaged map of {self.nmaps.value()} runs (slow):")
        sc.add("# a = " + call("density.run_denss_all", src, nmaps=self.nmaps.value(), cores=self.cores.value(),
                               **{k: v for k, v in kw.items() if k != "seed"}))
        sc.add_plot("lev = density.level_for_volume(r.rho, r.voxel, r.support_volume)",
                    "fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.5))",
                    "a1.loglog(r.fit_q, r.fit_I, 'o', ms=2, label='data'); a1.loglog(r.fit_q, r.fit_map, label='map')",
                    "a1.set_xlabel('q (1/nm)'); a1.legend()",
                    "n = r.rho.shape[2] // 2",
                    "a2.imshow(r.rho[:, :, n].T, origin='lower'); a2.contour(r.rho[:, :, n].T, [lev], colors='w')",
                    "a2.set_title('central slice and isosurface level')")
        return sc.text()

    # ------------------------------------------------------------------ session
    @staticmethod
    def _session_path():
        return os.path.join(os.path.dirname(prefs.settings_path()), "density_session.json")

    def save_session(self):
        try:
            files = []
            for s in self.sources:
                if s[0] == "curve" and s[1].path not in [f for f, _u in files]:
                    files.append((s[1].path, ""))
                elif s[0] == "file":
                    files.append((s[1], s[2]))
            st = {"version": 1, "files": files, "current": self.curve_list.currentRow(),
                  "derived": [[n, src, bg] for n, (src, bg) in self.derived.items()],
                  "bg": [self.bg_qmin.value(), self.bg_qmax.value(), self.bg_fix.isChecked(), self.bg_exp.value()],
                  "options": {"mode": self.mode.currentIndex(), "dmax": self.dmax.value(), "nmaps": self.nmaps.value(),
                              "cores": self.cores.value(), "enan": self.enan.isChecked(),
                              "level": self.level.value()},
                  "maps": [str(r["result"].map_file) for r in self.runs], "dock_state": self.dock_state()}
            p = self._session_path()
            os.makedirs(os.path.dirname(p), exist_ok=True)
            with open(p, "w", encoding="utf-8") as fh:
                json.dump(st, fh, indent=1)
        except (OSError, TypeError, ValueError):
            pass

    def restore_session(self):
        try:
            with open(self._session_path(), encoding="utf-8") as fh:
                st = json.load(fh)
        except (OSError, ValueError):
            return False
        try:
            for f, u in st.get("files") or []:
                if os.path.isfile(f):
                    if u:
                        self.on_load_out(f, u)
                    else:
                        self.load_files([f])
            for _n, src, bg in st.get("derived") or []:          # background-subtracted curves
                base = next((x[1] for x in self.sources if x[0] == "curve" and x[1].name == src), None)
                if base is not None:
                    self.subtract_background(base, float(bg))
            b = st.get("bg")
            if b:
                self.bg_qmin.setValue(b[0]); self.bg_qmax.setValue(b[1])  # noqa: E702
                self.bg_fix.setChecked(b[2]); self.bg_exp.setValue(b[3])  # noqa: E702
            o = st.get("options") or {}
            self.mode.setCurrentIndex(int(o.get("mode", 0)))
            self.dmax.setValue(float(o.get("dmax", 0)))
            self.nmaps.setValue(int(o.get("nmaps", 20)))
            self.cores.setValue(int(o.get("cores", self.cores.value())))
            self.enan.setChecked(bool(o.get("enan", True)))
            self.level.setValue(int(o.get("level", 100)))
            cur = int(st.get("current", 0))
            if 0 <= cur < self.curve_list.count():
                self.curve_list.setCurrentRow(cur)
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
        self.save_session()

    def closeEvent(self, event):
        self.shutdown()
        super().closeEvent(event)


def main(argv=None):
    argv = sys.argv if argv is None else argv
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(argv)
    from ..core.tabstyle import apply_tab_style
    apply_tab_style(app)
    app.setWindowIcon(tool_icon("pyDENsity"))
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
