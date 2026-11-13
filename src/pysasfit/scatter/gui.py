"""pySCAtter window: scattering patterns of particles, ordered mesoscale structures and their orientation
distributions (hypergeometric-function algorithms after Wagener & Förster, Sci. Rep. 13, 780 (2023)), and fits
to measured 1D and 2D data.

    pysasfit-scatter [params.ini]
    python -m pysasfit.scatter.gui [params.ini]

Also the "pySCAtter" tab of the pySASfit launcher.  Panels (View menu; each can be moved, stacked or detached):
Model (parameters with fit flags), 2D pattern, I(q), Lattice view, Data & fit, Log.  Parameter files are SasScatter2 / CrystalScatter
``.ini`` files (same keys, FITFLAG_<key>); pySCAtter adds a few keys (structure factor, shells) that older files
lack.  Lengths in nm, q in 1/nm.
"""
from __future__ import annotations

import json
import os
import sys
import time
import traceback

import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg  # noqa: E402
from matplotlib.colors import LogNorm, Normalize  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402

from ..core import settings as prefs  # noqa: E402
from ..core.docking import DockMixin, plot_panel  # noqa: E402
from ..resources import tool_icon  # noqa: E402
from . import engine  # noqa: E402
from . import params as P  # noqa: E402
from .fit import Data1D, Data2D, FitAborted, fit as run_fit, model_for  # noqa: E402

__all__ = ["MainWindow", "main"]

Dock = QtCore.Qt.DockWidgetArea
GROUP_TITLES = {"Particle": "Particle", "Interior": "Interior (shells)", "Orientation": "Orientation",
                "Lattice": "Lattice", "Peak Shape": "Peak shape and disorder", "Control": "Lattice factor",
                "Structure factor": "Structure factor (no lattice)", "Pixel manipulation": "Intensity",
                "Calculation": "Calculation", "Experiment": "Detector (2D, not 'preset')"}
GROUP_ORDER = ["Particle", "Interior", "Orientation", "Lattice", "Peak Shape", "Control", "Structure factor",
               "Pixel manipulation", "Calculation", "Experiment"]
#: parameters shown on another page than their SasScatter group
PAGE_EXTRA = {"Orientation": ["EditDbeta", "EditAzi"]}
#: toolbox page opened when a combo box changes (its parameters are there)
COMBO_PAGE = {"ComboBoxParticle": "Particle", "ComboBoxInterior": "Interior", "Ordis": "Orientation",
              "LType": "Lattice", "ComboBoxPeak": "Peak Shape", "StructureFactor": "Structure factor"}
TOOLTIP_EXTRA = {
    "ComboBoxParticle": "Kratky Porod chain: R = Kuhn length, L = contour length, σ_L its spread.\n"
                        "Teubner-Strey: R = correlation length ξ, L = repeat distance d.\n"
                        "Cylinder: R = radius, L = half length. Disk: R = half thickness, L = radius.\n"
                        "Vesicle: R = inner radius, L = outer radius. Ellipsoids: L = semi-axis along the "
                        "symmetry axis (triaxial: R₂ = third semi-axis).",
    "ComboBoxInterior": "core + shell: R = core radius, R₂ = outer radius, ρ shell = shell contrast (core 1).\n"
                        "inhomogeneous shell: linear gradient or power law (Förster & Burger 1998).\n"
                        "multi-shell: core R and n shells of equal thickness, contrasts alternate ρ shell / ρ 2nd shell.\n"
                        "myelin: water core R, n bilayers of R₂ (lipid, ρ shell) + shell thickness (water).",
    "Ordis": "isotropic: random orientation (powder). z-dir: perfectly aligned along z. fiber pattern: "
             "uniform rotation about z. Others: distribution of the particle axis about the director (θ, φ) "
             "of width 'particle axis: width' (page Orientation; anisometric particles only).\n"
             "Lattices: oriented single crystal (zone axis, ψ); its reflections are spread over arcs of width "
             "4/D_ψ ('lattice: azimuthal domain', page Orientation) with the profile of the peak shape -- the "
             "kind of distribution chosen here acts on the particles only.\n"
             "Orientation distributions are calculated for cylinder, disk and ellipsoid; the cube has 'z-dir' "
             "only; triaxial ellipsoid, super ellipsoid and superball are always orientation averaged.",
    "LType": "None: single particles (structure factor below possible). Otherwise the decoupling approximation "
             "I = P (1 + β (Z − 1) e^(−B q²)) with the lattice factor Z: powder average for isotropic orientation, "
             "oriented single crystal (zone axis, ψ) otherwise, fibre average for 'fiber pattern'.",
    "StructureFactor": "Isotropic structure factor in the decoupling approximation I = P (1 + β (S − 1)); only "
                       "without a lattice.",
}


def _canvas(size=(5, 4)):
    fig = Figure(figsize=size, layout="constrained")
    return FigureCanvasQTAgg(fig)


# ===================================================================================== parameter panel
class ParamPanel(QtWidgets.QWidget):
    """Combo boxes and numeric parameters (grouped, with fit flags); parameters that the current settings do
    not use are greyed out."""
    changed = QtCore.Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.combo, self.spin, self.flag, self.label = {}, {}, {}, {}
        self._extra = {}                     # parameters not shown (unused .ini keys, bools, comment)
        self._building = True
        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(4, 4, 4, 4)
        form = QtWidgets.QFormLayout()
        form.setFieldGrowthPolicy(QtWidgets.QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)
        for key, lab in P.COMBO_LABELS.items():
            cb = QtWidgets.QComboBox()
            cb.addItems(P.COMBOS[key])
            for i, why in P.NOT_AVAILABLE.get(key, {}).items():      # listed (old .ini files), not selectable
                cb.model().item(i).setEnabled(False)
                cb.setItemData(i, why, QtCore.Qt.ItemDataRole.ToolTipRole)
            cb.setMinimumContentsLength(12)
            cb.setSizeAdjustPolicy(QtWidgets.QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
            if key in TOOLTIP_EXTRA:
                cb.setToolTip(TOOLTIP_EXTRA[key])
            cb.currentIndexChanged.connect(self._on_change)
            if key in COMBO_PAGE:
                cb.activated.connect(lambda _i, g=COMBO_PAGE[key]: self.show_page(g))
            self.combo[key] = cb
            form.addRow(lab, cb)
        self.width_mode = QtWidgets.QComboBox()
        self.width_mode.addItems(["4/D (Debye-Scherrer)", "4/D + paracrystal (σ_DW²/dist) q²", "fixed 1 (SasScatter2 default)"])
        self.width_mode.setToolTip("Peak width δ of the lattice reflections (Förster et al. 2011, eq. 92: D = 4/δ for a "
                                   "domain of size D). Paracrystal: δ grows with q, δ = 4/D + σ_DW²/dist · q² with the "
                                   "nearest-neighbour distance dist. 'fixed 1': δ = 1, D ignored -- what SasScatter2 "
                                   "does when neither option is ticked (.ini files without RadButDebyeScherrer).")
        self.width_mode.currentIndexChanged.connect(self._on_change)
        form.addRow("peak width", self.width_mode)
        self.preset = QtWidgets.QCheckBox("2D: q range from q max (else detector geometry)")
        self.preset.setToolTip("Preset: the 2D pattern is a square q map of ±q max with 'grid points' pixels. Off: "
                               "the pixels of the detector group (distance, pixel size and number, beam position)")
        self.preset.toggled.connect(self._on_change)
        form.addRow(self.preset)
        lay.addLayout(form)
        hint = QtWidgets.QLabel("☑ = fitted parameter. Grey: not used with these settings.")
        hint.setStyleSheet("color: gray;")
        hint.setWordWrap(True)
        lay.addWidget(hint)
        self.toolbox = QtWidgets.QToolBox()
        self.page_index = {}                 # group -> toolbox index
        moved = {k: g for g, ks in PAGE_EXTRA.items() for k in ks}
        for g in GROUP_ORDER:
            keys = PAGE_EXTRA.get(g, []) + [k for k in P.GROUPS[g] if k not in P.UNUSED and k not in moved]
            if not keys:
                continue
            page = QtWidgets.QWidget()
            grid = QtWidgets.QGridLayout(page)
            grid.setContentsMargins(2, 2, 2, 2)
            grid.setColumnStretch(1, 1)
            for i, k in enumerate(keys):
                _key, _lab, default, lo, hi, dec, tip = P.NUM[k]
                lab = QtWidgets.QLabel(P.LABELS.get(k, k))
                lab.setToolTip(f"{tip}\n(.ini key {k})")
                sp = QtWidgets.QDoubleSpinBox()
                sp.setRange(lo, hi)
                sp.setDecimals(max(int(dec), 0) if dec else 0)
                if dec >= 3:
                    sp.setDecimals(min(int(dec) + 2, 8))
                sp.setStepType(QtWidgets.QAbstractSpinBox.StepType.AdaptiveDecimalStepType)
                sp.setKeyboardTracking(False)
                sp.setValue(default)
                sp.setToolTip(tip)
                sp.valueChanged.connect(self._on_change)
                fl = QtWidgets.QCheckBox()
                fl.setToolTip(f"fit {P.LABELS.get(k, k)}")
                grid.addWidget(lab, i, 0)
                grid.addWidget(sp, i, 1)
                grid.addWidget(fl, i, 2)
                self.spin[k], self.flag[k], self.label[k] = sp, fl, lab
            grid.setRowStretch(len(keys), 1)
            self.page_index[g] = self.toolbox.addItem(page, GROUP_TITLES.get(g, g))
        lay.addWidget(self.toolbox, 1)
        self._building = False
        self.set_params(P.defaults(), {})

    def show_page(self, group):
        """Open the toolbox page of ``group`` (after the user picked an entry of the combo box it belongs to)."""
        if group in self.page_index:
            self.toolbox.setCurrentIndex(self.page_index[group])

    # ---------------------------------------------------------------- values
    def _on_change(self, *_a):
        if self._building:
            return
        self._mark_relevant()
        self.changed.emit()

    def params(self):
        p = dict(self._extra)
        for k, cb in self.combo.items():
            p[k] = cb.currentIndex()
        for k, sp in self.spin.items():
            p[k] = float(sp.value())
        p["EditQmaxPreset"] = self.preset.isChecked()
        m = self.width_mode.currentIndex()
        p["RadButDebyeScherrer"], p["RadioButtonPara"] = m == 0, m == 1
        return p

    def flags(self):
        return {k: True for k, f in self.flag.items() if f.isChecked() and f.isEnabled()}

    def set_params(self, p, flags=None):
        self._building = True
        try:
            full = P.defaults()
            full.update(p)
            self._extra = {k: v for k, v in full.items() if k not in self.combo and k not in self.spin}
            for k, cb in self.combo.items():
                cb.setCurrentIndex(int(np.clip(int(full.get(k, 0)), 0, cb.count() - 1)))
            for k, sp in self.spin.items():
                sp.setValue(float(full[k]))
            self.preset.setChecked(bool(full.get("EditQmaxPreset", True)))
            self.width_mode.setCurrentIndex(1 if full.get("RadioButtonPara") else
                                            0 if full.get("RadButDebyeScherrer", True) else 2)
            if flags is not None:
                for k, f in self.flag.items():
                    f.setChecked(bool(flags.get(k)))
        finally:
            self._building = False
        self._mark_relevant()
        self.changed.emit()

    def update_values(self, values):
        """Set a few numbers (after a fit) without touching anything else."""
        self._building = True
        try:
            for k, v in values.items():
                if k in self.spin:
                    self.spin[k].setValue(float(v))
        finally:
            self._building = False
        self.changed.emit()

    def _mark_relevant(self):
        used = P.relevant(self.params())
        for k in self.spin:
            on = k in used
            self.spin[k].setEnabled(on)
            self.flag[k].setEnabled(on)
            self.label[k].setEnabled(on)
        self.combo["ComboBoxPeak"].setEnabled(P.LTYPE_KIND.get(self.combo["LType"].currentIndex()) is not None)
        self.combo["StructureFactor"].setEnabled(P.LTYPE_KIND.get(self.combo["LType"].currentIndex()) is None)
        self.combo["ShellProfile"].setEnabled(self.combo["ComboBoxInterior"].currentIndex() == 2)
        self.width_mode.setEnabled(P.LTYPE_KIND.get(self.combo["LType"].currentIndex()) is not None)


# ===================================================================================== worker threads
class Worker(QtCore.QThread):
    """Runs one function in a thread: progress(int), done(object), failed(str)."""
    progress = QtCore.Signal(int)
    done = QtCore.Signal(object)
    failed = QtCore.Signal(str)

    def __init__(self, fn, parent=None):
        super().__init__(parent)
        self.fn = fn
        self.abort = False

    def run(self):
        try:
            self.done.emit(self.fn(self))
        except (engine.CalcError, FitAborted) as exc:
            self.failed.emit(str(exc))
        except Exception as exc:  # noqa: BLE001 -- shown in the log
            self.failed.emit(f"{type(exc).__name__}: {exc}\n{traceback.format_exc(limit=4)}")


# ===================================================================================== main window
class MainWindow(DockMixin, QtWidgets.QMainWindow):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("pySCAtter – scattering patterns of particles and ordered structures")
        self.setWindowIcon(tool_icon("pySCAtter"))
        self.ini_path = None
        self.result1d = None            # (q, I) of the last 1D calculation
        self.result2d = None            # (qx, qy, qz, I) of the last 2D calculation
        self.data1d = []                # Data1D
        self.data2d = []                # (Data2D, Image2D)
        self.worker = None
        self.fit_worker = None
        self._pending = False
        self._build_menus()
        self.params_panel = ParamPanel()
        self.params_panel.changed.connect(self._on_params_changed)
        self._build_model_dock()
        self._build_plots()
        self._build_data_dock()
        from .latview_gui import LatticeView
        self.lattice_view = LatticeView()
        self.lattice_view.status.connect(lambda m: self.statusBar().showMessage(m, 4000))
        self.log_view = QtWidgets.QPlainTextEdit()
        self.log_view.setReadOnly(True)
        self._add_dock("model", "Model", self.model_widget, Dock.LeftDockWidgetArea, scrollable=True)
        self._add_dock("pattern", "2D pattern", self.pattern_panel, Dock.RightDockWidgetArea)
        self._add_dock("iq", "I(q)", self.iq_panel, Dock.RightDockWidgetArea)
        self._add_dock("data", "Data and fit", self.data_widget, Dock.RightDockWidgetArea, scrollable=True)
        self._add_dock("log", "Log", self.log_view, Dock.RightDockWidgetArea)
        self._add_dock("lattice", "Lattice view", self.lattice_view, Dock.RightDockWidgetArea)
        self.splitDockWidget(self.docks["pattern"], self.docks["iq"], QtCore.Qt.Orientation.Horizontal)
        self.splitDockWidget(self.docks["pattern"], self.docks["data"], QtCore.Qt.Orientation.Vertical)
        self.tabifyDockWidget(self.docks["data"], self.docks["log"])
        self.tabifyDockWidget(self.docks["iq"], self.docks["lattice"])
        self.docks["data"].raise_()
        self.docks["iq"].raise_()
        self._finish_docks()
        self.hide_stray_tab_bars()
        self._recalc_timer = QtCore.QTimer(self, singleShot=True, interval=350)
        self._recalc_timer.timeout.connect(self._auto_calculate)
        self.resize(1500, 920)
        self._quiet = True               # nothing is calculated when the tool starts: Calculate 2D
        self._on_params_changed()
        self._quiet = False

    # ------------------------------------------------------------------ construction
    def _build_menus(self):
        m = self.menuBar().addMenu("&File")
        a = m.addAction("&Open parameters (.ini)…", self.on_open_ini)
        a.setShortcut(QtGui.QKeySequence.StandardKey.Open)
        a = m.addAction("&Save parameters…", self.on_save_ini)
        a.setShortcut(QtGui.QKeySequence.StandardKey.Save)
        m.addSeparator()
        m.addAction("Load &1D data…", self.on_load_1d)
        m.addAction("Load &2D data…", self.on_load_2d)
        m.addSeparator()
        m.addAction("Export I(q) (text)…", self.on_export_1d)
        m.addAction("Export 2D pattern (text / npy)…", self.on_export_2d)
        from ..core import pycode
        pycode.install(self, self.python_code, menu=m)
        from ..core.settings_dialog import add_preferences_menu
        add_preferences_menu(self, "scatter")
        self.menuBar().addMenu("&View")

    def _build_model_dock(self):
        w = QtWidgets.QWidget()
        lay = QtWidgets.QVBoxLayout(w)
        lay.setContentsMargins(2, 2, 2, 2)
        row = QtWidgets.QHBoxLayout()
        self.calc_button = QtWidgets.QPushButton("Calculate 2D")
        self.calc_button.setToolTip("Calculate the 2D pattern (and I(q)) with the current parameters")
        self.calc_button.clicked.connect(lambda _c=False: self.calculate(two_d=True))
        self.abort_button = QtWidgets.QPushButton("Abort")
        self.abort_button.setEnabled(False)
        self.abort_button.clicked.connect(self.on_abort)
        row.addWidget(self.calc_button)
        row.addWidget(self.abort_button)
        from ..core import pycode
        row.addWidget(pycode.button(self.python_code, w))
        lay.addLayout(row)
        row = QtWidgets.QHBoxLayout()
        self.auto_1d = QtWidgets.QCheckBox("I(q) follows")
        self.auto_1d.setChecked(bool(prefs.get("scatter.calc.auto_1d")))
        self.auto_1d.setToolTip("Recalculate I(q) at once when a parameter changes (fast)")
        self.auto_2d = QtWidgets.QCheckBox("2D follows")
        self.auto_2d.setToolTip("Recalculate the 2D pattern when a parameter changes (can take seconds for "
                                "oriented anisometric particles)")
        row.addWidget(self.auto_1d)
        row.addWidget(self.auto_2d)
        lay.addLayout(row)
        self.progress = QtWidgets.QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setTextVisible(False)
        self.progress.setMaximumHeight(8)
        lay.addWidget(self.progress)
        # only the parameters scroll: the buttons and check boxes above stay in view
        from ..core.scrolling import scrolled
        self.params_scroll = scrolled(self.params_panel)
        lay.addWidget(self.params_scroll, 1)
        self.model_widget = w

    def _build_plots(self):
        # ---- 2D
        self.canvas2d = _canvas()
        bar2 = QtWidgets.QHBoxLayout()
        self.view2d = QtWidgets.QComboBox()
        for text, data in (("model", "model"), ("data", "data"), ("data | model", "split"),
                           ("residuals (data − model)/σ", "resid")):
            self.view2d.addItem(text, data)
        self.view2d.setToolTip("What the 2D panel shows. With 2D data: the model is calculated at the data pixels "
                               "(Calculate 2D or the fit); 'data | model' shows the data left, the model right "
                               "of q_h = 0")
        self.view2d.currentIndexChanged.connect(lambda _i: self.draw_2d())
        self.log2d = QtWidgets.QCheckBox("log")
        self.log2d.setChecked(True)
        self.log2d.toggled.connect(lambda _c: self.draw_2d())
        self.cmap2d = QtWidgets.QComboBox()
        self.cmap2d.addItems(["viridis", "jet", "turbo", "magma", "gray", "cividis"])
        self.cmap2d.currentIndexChanged.connect(lambda _i: self.draw_2d())
        self.hkl2d = QtWidgets.QCheckBox("hkl")
        self.hkl2d.setToolTip("Mark the Bragg reflections of the lattice that lie on the detector (oriented crystal) "
                              "or the powder rings, with their hkl (see the Lattice view panel)")
        self.hkl2d.toggled.connect(lambda _c: self.draw_2d())
        from ..core.flowlayout import flow_bar
        bar2.addWidget(flow_bar(QtWidgets.QLabel("show"), self.view2d, self.log2d, self.cmap2d, self.hkl2d), 1)
        self.pattern_panel = plot_panel(self.canvas2d, self, bars=(bar2,))
        self.canvas2d.mpl_connect("motion_notify_event", self._on_motion_2d)
        # ---- 1D
        self.canvas1d = _canvas()
        bar1 = QtWidgets.QHBoxLayout()
        self.iq_scale = QtWidgets.QComboBox()
        for text, data in (("log-log", "loglog"), ("lin-log", "linlog"), ("Kratky I q²", "kratky"),
                           ("lin-lin", "linlin")):
            self.iq_scale.addItem(text, data)
        self.iq_scale.currentIndexChanged.connect(lambda _i: self.draw_1d())
        self.show_resid = QtWidgets.QCheckBox("residuals")
        self.show_resid.setChecked(True)
        self.show_resid.toggled.connect(lambda _c: self.draw_1d())
        self.show_unsmeared = QtWidgets.QCheckBox("unsmeared model")
        self.show_unsmeared.setToolTip("Also the model without the q resolution of the data")
        self.show_unsmeared.toggled.connect(lambda _c: self.draw_1d())
        bar1.addWidget(flow_bar(self.iq_scale, self.show_resid, self.show_unsmeared), 1)
        self.iq_panel = plot_panel(self.canvas1d, self, bars=(bar1,))

    def _build_data_dock(self):
        w = QtWidgets.QWidget()
        lay = QtWidgets.QVBoxLayout(w)
        g = QtWidgets.QGroupBox("Measured data (pySASred output, NXcanSAS, text, BerSANS)")
        v = QtWidgets.QVBoxLayout(g)
        self.data_list = QtWidgets.QListWidget()
        self.data_list.setMaximumHeight(110)
        self.data_list.setToolTip("Loaded curves and images; the selected one is plotted and fitted")
        self.data_list.currentRowChanged.connect(self._on_data_selected)
        v.addWidget(self.data_list)
        row = QtWidgets.QHBoxLayout()
        for text, slot in (("Load 1D…", self.on_load_1d), ("Load 2D…", self.on_load_2d),
                           ("Remove", self.on_remove_data)):
            b = QtWidgets.QPushButton(text)
            b.clicked.connect(lambda _c=False, s=slot: s())
            row.addWidget(b)
        v.addLayout(row)
        lay.addWidget(g)
        g = QtWidgets.QGroupBox("Fit of the ☑ parameters to the selected data")
        f = QtWidgets.QFormLayout(g)
        self.fit_qmin = QtWidgets.QLineEdit()
        self.fit_qmax = QtWidgets.QLineEdit()
        for e in (self.fit_qmin, self.fit_qmax):
            e.setPlaceholderText("all")
            e.setValidator(QtGui.QDoubleValidator(0, 1e9, 10))
        f.addRow("q min (1/nm)", self.fit_qmin)
        f.addRow("q max (1/nm)", self.fit_qmax)
        self.use_dq = QtWidgets.QCheckBox("fold with the q resolution (Qdev) of the data")
        self.use_dq.setChecked(True)
        self.use_dq.toggled.connect(lambda _c: self._update_data_model())
        f.addRow(self.use_dq)
        self.log_weight = QtWidgets.QCheckBox("relative residuals (ln I)")
        self.log_weight.setToolTip("Weight the residuals of ln I: every decade counts alike, useful when the "
                                   "statistical errors are underestimated at low q")
        f.addRow(self.log_weight)
        self.every_px = QtWidgets.QSpinBox()
        self.every_px.setRange(1, 32)
        self.every_px.setValue(int(prefs.get("scatter.fit.every_px")))
        self.every_px.setToolTip("2D fits: use every n-th pixel in both directions (n² fewer model evaluations)")
        f.addRow("2D: every n-th pixel", self.every_px)
        self.max_nfev = QtWidgets.QSpinBox()
        self.max_nfev.setRange(5, 100000)
        self.max_nfev.setValue(int(prefs.get("scatter.fit.max_nfev")))
        f.addRow("max. evaluations", self.max_nfev)
        row = QtWidgets.QHBoxLayout()
        self.fit_button = QtWidgets.QPushButton("Fit")
        self.fit_button.clicked.connect(lambda _c=False: self.on_fit())
        self.fit_abort = QtWidgets.QPushButton("Stop")
        self.fit_abort.setEnabled(False)
        self.fit_abort.clicked.connect(self.on_fit_abort)
        self.chi2_label = QtWidgets.QLabel("χ²ᵣ –")
        row.addWidget(self.fit_button)
        row.addWidget(self.fit_abort)
        row.addWidget(self.chi2_label, 1)
        f.addRow(row)
        lay.addWidget(g)
        self.fit_text = QtWidgets.QPlainTextEdit()
        self.fit_text.setReadOnly(True)
        self.fit_text.setLineWrapMode(QtWidgets.QPlainTextEdit.LineWrapMode.NoWrap)
        self.fit_text.setFont(QtGui.QFontDatabase.systemFont(QtGui.QFontDatabase.SystemFont.FixedFont))
        self.fit_text.setMinimumHeight(90)
        lay.addWidget(self.fit_text, 1)
        self.data_widget = w

    # ------------------------------------------------------------------ helpers
    def log(self, text):
        self.log_view.appendPlainText(time.strftime("%H:%M:%S ") + text)

    def params(self):
        return self.params_panel.params()

    def current_data(self):
        i = self.data_list.currentRow()
        items = [("1d", d) for d in self.data1d] + [("2d", d) for d, _im in self.data2d]
        return items[i] if 0 <= i < len(items) else (None, None)

    def _data_for_model(self, d):
        """The data object with or without its q resolution (checkbox)."""
        if isinstance(d, Data1D) and not self.use_dq.isChecked():
            return Data1D(d.q, d.I, d.dI, None, d.name)
        return d

    # ------------------------------------------------------------------ calculation
    def _on_params_changed(self):
        if hasattr(self, "lattice_view"):
            self.lattice_view.set_params(self.params())
            if self.hkl2d.isChecked():
                self.draw_2d()
        if (self.auto_1d.isChecked() or self.auto_2d.isChecked()) and not getattr(self, "_quiet", False):
            self._recalc_timer.start()

    def _auto_calculate(self):
        self.calculate(two_d=self.auto_2d.isChecked())

    def calculate(self, two_d=True):
        """1D (and 2D) calculation in a thread; a request while one runs is queued once."""
        if self.worker is not None and self.worker.isRunning():
            self._pending = True
            self._pending_2d = getattr(self, "_pending_2d", False) or two_d
            return
        p = self.params()
        kind, d = self.current_data()

        def job(wk):
            t0 = time.time()
            out = {"p": p, "q1": None, "I1": None, "two_d": two_d}
            q, I = engine.calc_1d(p)
            out["q1"], out["I1"] = q, I
            if kind is not None:
                dd = self._data_for_model(d)
                used = dd.used() if kind == "1d" else dd.used(every=1)
                out["data_model"] = (d, used, model_for(p, dd, used))
                if kind == "1d" and dd.dq is not None:
                    out["data_model_unsmeared"] = engine.intensity_1d(p, dd.q[used])
            if two_d:
                qx, qy, qz = engine.q_grid_2d(p)
                I2 = engine.intensity_2d(p, qx, qy, qz, progress=wk.progress.emit, should_abort=lambda: wk.abort)
                out["I2"] = (qx, qy, qz, I2)
            out["seconds"] = time.time() - t0
            return out

        self.worker = Worker(job, self)
        self.worker.progress.connect(self.progress.setValue)
        self.worker.done.connect(self._calc_done)
        self.worker.failed.connect(self._calc_failed)
        self.worker.finished.connect(self._calc_finished)
        self.calc_button.setEnabled(False)
        self.abort_button.setEnabled(True)
        self.progress.setValue(0)
        self.worker.start()

    def on_abort(self):
        if self.worker is not None:
            self.worker.abort = True

    def _calc_done(self, out):
        self.result1d = (out["q1"], out["I1"])
        self.data_model = out.get("data_model")
        self.data_model_unsmeared = out.get("data_model_unsmeared")
        if "I2" in out:
            self.result2d = out["I2"]
            self.result2d_params = out["p"]
            self.log(f"calculated ({out['seconds']:.1f} s)")
        self.draw_1d()
        if "I2" in out or self.view2d.currentData() != "model":
            self.draw_2d()
        self.progress.setValue(100)

    def _calc_failed(self, msg):
        self.log("calculation: " + msg)
        self.statusBar().showMessage(msg.splitlines()[0], 8000)

    def _calc_finished(self):
        self.calc_button.setEnabled(True)
        self.abort_button.setEnabled(False)
        if self._pending:
            self._pending = False
            two_d, self._pending_2d = getattr(self, "_pending_2d", False), False
            self.calculate(two_d=two_d)

    def _update_data_model(self):
        self.calculate(two_d=False)

    # ------------------------------------------------------------------ drawing
    def draw_1d(self):
        fig = self.canvas1d.figure
        fig.clear()
        resid = self.show_resid.isChecked()
        kind, d = self.current_data()
        dm = getattr(self, "data_model", None)
        has_data = kind == "1d" and dm is not None and dm[0] is d
        if resid and has_data:
            gs = fig.add_gridspec(2, 1, height_ratios=(3, 1))
            ax = fig.add_subplot(gs[0])
            axr = fig.add_subplot(gs[1], sharex=ax)
        else:
            ax, axr = fig.add_subplot(111), None
        mode = self.iq_scale.currentData()

        def y(q, I):
            return I * q * q if mode == "kratky" else I

        if self.result1d is not None:
            q, I = self.result1d
            ax.plot(q, y(q, I), "-", color="C0", lw=1.3, label="model")
        if kind == "1d":
            u = d.used()
            ax.errorbar(d.q[u], y(d.q[u], d.I[u]), yerr=None if d.dI is None else y(d.q[u], d.dI[u]),
                        fmt="o", ms=2.5, color="0.25", lw=0.6, capsize=0, label=d.name or "data", zorder=1)
            if has_data:
                _d, used, Im = dm
                ax.plot(d.q[used], y(d.q[used], Im), "-", color="C3", lw=1.3,
                        label="model at the data" + (" (smeared)" if d.dq is not None and self.use_dq.isChecked()
                                                     else ""))
                if self.show_unsmeared.isChecked() and getattr(self, "data_model_unsmeared", None) is not None:
                    ax.plot(d.q[used], y(d.q[used], self.data_model_unsmeared), "--", color="C1", lw=1,
                            label="unsmeared")
                if axr is not None:
                    sig = d.dI[used] if d.dI is not None else np.abs(d.I[used]) * 0.01
                    axr.plot(d.q[used], (d.I[used] - Im) / sig, ".", ms=3, color="C3")
                    axr.axhline(0, color="0.5", lw=0.8)
                    axr.set_ylabel("Δ/σ")
                    axr.set_xlabel("q [1/nm]")
        if mode in ("loglog", "kratky"):
            ax.set_xscale("log")
        if mode in ("loglog", "linlog"):
            ax.set_yscale("log", nonpositive="mask")
        if axr is not None and mode in ("loglog", "kratky"):
            axr.set_xscale("log")
        ax.set_ylabel("I q² [a.u.]" if mode == "kratky" else "I [a.u.]")
        if axr is None:
            ax.set_xlabel("q [1/nm]")
        if ax.get_legend_handles_labels()[0]:
            ax.legend(fontsize="small")
        self.canvas1d.draw_idle()

    def draw_2d(self):
        fig = self.canvas2d.figure
        fig.clear()
        ax = fig.add_subplot(111)
        view = self.view2d.currentData()
        kind, d = self.current_data()
        cmap = self.cmap2d.currentText()
        log = self.log2d.isChecked()
        im = None
        if view == "model" or kind != "2d":
            if self.result2d is not None:
                qx, qy, qz, I = self.result2d
                ext = (float(qy.min()), float(qy.max()), float(qz.min()), float(qz.max()))
                Z = I.T
                im = self._imshow(ax, Z, ext, cmap, log)
                ax.set_title("model", fontsize="small")
            elif kind != "2d":
                ax.set_title("Calculate 2D to see the pattern", fontsize="small")
        else:
            dm = getattr(self, "data_model", None)
            model = None
            if dm is not None and dm[0] is d:
                model = np.full(d.I.shape, np.nan)
                model[dm[1]] = dm[2]
            ext = (float(np.nanmin(d.qh)), float(np.nanmax(d.qh)), float(np.nanmin(d.qv)), float(np.nanmax(d.qv)))
            data = np.where(d.used(), d.I, np.nan)
            if view == "data" or model is None:
                im = self._imshow(ax, data, ext, cmap, log)
                ax.set_title(d.name + ("" if model is not None else "  (model: Calculate 2D)"), fontsize="small")
            elif view == "split":
                Z = np.where(d.qh < 0, data, model)
                im = self._imshow(ax, Z, ext, cmap, log)
                ax.axvline(0, color="w", lw=0.8)
                ax.set_title("data  |  model", fontsize="small")
            else:
                sig = d.dI if d.dI is not None else np.abs(d.I) * 0.01
                R = (data - model) / sig
                lim = float(np.nanpercentile(np.abs(R), 99)) if np.isfinite(R).any() else 1.0
                im = ax.imshow(R, origin="lower", extent=ext, cmap="RdBu_r", vmin=-lim, vmax=lim, aspect="equal",
                               interpolation="nearest")
                ax.set_title("(data − model)/σ", fontsize="small")
        if im is not None:
            fig.colorbar(im, ax=ax, fraction=0.05, pad=0.02)
        if self.hkl2d.isChecked():
            self._overlay_hkl(ax)
        ax.set_xlabel("q_h [1/nm]")
        ax.set_ylabel("q_v [1/nm]")
        self.canvas2d.draw_idle()

    def _overlay_hkl(self, ax):
        from . import latview as LV
        from .latview_gui import draw_reflections
        # the reflections of the parameters the shown model was calculated with (the image may be older than the
        # current parameters when '2D follows' is off)
        pp = getattr(self, "result2d_params", None)
        if pp is None or self.view2d.currentData() != "model":
            pp = self.params()
        try:
            sc = LV.build_scene(pp, 1)
        except Exception:  # noqa: BLE001
            return
        if sc.kind is None:
            return
        xl, yl = ax.get_xlim(), ax.get_ylim()
        draw_reflections(ax, sc, labels=True, colour="w")
        ax.set_xlim(xl)
        ax.set_ylim(yl)

    @staticmethod
    def _imshow(ax, Z, ext, cmap, log):
        Z = np.asarray(Z, float)
        fin = Z[np.isfinite(Z)]
        if log and fin.size and np.any(fin > 0):
            pos = fin[fin > 0]
            norm = LogNorm(vmin=max(float(pos.min()), float(pos.max()) * 1e-7), vmax=float(pos.max()))
            Z = np.where(Z > 0, Z, np.nan)
        else:
            norm = Normalize(vmin=float(fin.min()) if fin.size else 0, vmax=float(fin.max()) if fin.size else 1)
        return ax.imshow(Z, origin="lower", extent=ext, cmap=cmap, norm=norm, aspect="equal",
                         interpolation="nearest")

    def _on_motion_2d(self, ev):
        if ev.inaxes is None or ev.xdata is None:
            return
        q = np.hypot(ev.xdata, ev.ydata)
        self.statusBar().showMessage(f"q_h {ev.xdata:.4g}  q_v {ev.ydata:.4g}  |q| {q:.4g} 1/nm  "
                                     f"d = 2π/q = {2 * np.pi / q if q > 0 else np.inf:.4g} nm", 4000)

    # ------------------------------------------------------------------ data
    def on_load_1d(self, path=None):
        if path is None:
            path, _ = QtWidgets.QFileDialog.getOpenFileName(
                self, "Load 1D data", self._dir(), "1D data (*.h5 *.nxs *.hdf *.dat *.txt *.csv *.0*);;all (*)")
            if not path:
                return
        from ..toolbox.merge import read_curves
        try:
            curves = read_curves(path)
        except Exception as exc:  # noqa: BLE001
            QtWidgets.QMessageBox.warning(self, "Load 1D data", f"{path}:\n{exc}")
            return
        for c in curves:
            d = Data1D.from_curve(c)
            d.path = path
            self.data1d.append(d)
            self.log(f"1D data {d.name}: {d.q.size} points, q {np.nanmin(d.q):.4g} … {np.nanmax(d.q):.4g} 1/nm"
                     + (", with Qdev" if d.dq is not None else ""))
        self._fill_data_list(select=len(self.data1d) - 1)

    def on_load_2d(self, path=None):
        if path is None:
            path, _ = QtWidgets.QFileDialog.getOpenFileName(
                self, "Load 2D data", self._dir(), "2D data (*.h5 *.nxs *.hdf *.0*);;all (*)")
            if not path:
                return
        from ..toolbox.image2d import read_images
        try:
            ims = read_images(path)
        except Exception as exc:  # noqa: BLE001
            QtWidgets.QMessageBox.warning(self, "Load 2D data", f"{path}:\n{exc}")
            return
        n = 0
        for im in ims:
            if not im.has_q:
                self.log(f"{im.name}: geometry unknown (no q) – not loaded")
                continue
            d = Data2D.from_image(im)
            d.path = path
            self.data2d.append((d, im))
            n += 1
            self.log(f"2D data {d.name}: {d.I.shape[1]}×{d.I.shape[0]} pixels, "
                     f"|q| ≤ {float(np.nanmax(np.hypot(d.qh, d.qv))):.4g} 1/nm")
        if n:
            self._fill_data_list(select=len(self.data1d) + len(self.data2d) - 1)
            self.view2d.setCurrentIndex(self.view2d.findData("split"))

    def _dir(self):
        return os.path.dirname(self.ini_path) if self.ini_path else os.getcwd()

    def on_remove_data(self):
        i = self.data_list.currentRow()
        if i < 0:
            return
        if i < len(self.data1d):
            del self.data1d[i]
        else:
            del self.data2d[i - len(self.data1d)]
        self._fill_data_list(select=min(i, len(self.data1d) + len(self.data2d) - 1))

    def _fill_data_list(self, select=-1):
        self.data_list.blockSignals(True)
        self.data_list.clear()
        for d in self.data1d:
            self.data_list.addItem(f"1D  {d.name}")
        for d, _im in self.data2d:
            self.data_list.addItem(f"2D  {d.name}")
        self.data_list.blockSignals(False)
        self.data_list.setCurrentRow(select)
        self._on_data_selected(select)

    def _on_data_selected(self, _row):
        self.data_model = None
        if getattr(self, "_quiet", False):     # last session: show the data, calculate on request only
            self.draw_1d()
            return
        self.calculate(two_d=False)

    # ------------------------------------------------------------------ Python code
    def python_code(self):
        """Script: the parameters (those that differ from the defaults), I(q), the 2D pattern and, with data
        selected, its model and the fit of the ticked parameters ("Copy Python code")."""
        from ..core.pycode import Code, Script, call, lit, path_lit

        p = self.params()
        sc = Script("pySCAtter", "scattering pattern" + (" and fit" if self.current_data()[0] else ""))
        sc.imp("import numpy as np", "from pysasfit.scatter import engine, params as P")
        d0 = P.defaults()
        changed = {k: v for k, v in p.items() if k not in d0 or d0[k] != v}
        sc.comment("parameters (CrystalScatter names): the defaults and the values set on the panel")
        sc.add("p = P.defaults()", f"p.update({lit(changed)})" if changed else "")
        sc.comment("I(q) on the log q grid (EditQmin ... EditQmax) and the 2D pattern")
        sc.add("q, I = engine.calc_1d(p)", "qx, qy, qz = engine.q_grid_2d(p)", "I2 = engine.intensity_2d(p, qx, qy, qz)")
        kind, d = self.current_data()
        plot = ["fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))", "ax1.loglog(q, I, label='model')"]
        if kind is not None:
            sc.comment(f"data: {d.name}")
            path = getattr(d, "path", None)
            if kind == "1d":
                sc.imp("from pysasfit.scatter.fit import Data1D", "from pysasfit.toolbox.merge import read_curves")
                sc.add(f"curves = read_curves({path_lit(path)})",
                       f"data = Data1D.from_curve(next(c for c in curves if c.name == {d.name!r}))")
                if not self.use_dq.isChecked():
                    sc.add("data = Data1D(data.q, data.I, data.dI, None, data.name)   # without its q resolution")
                plot.append("ax1.errorbar(data.q, data.I, data.dI, fmt='o', ms=2, label=data.name)")
            else:
                sc.imp("from pysasfit.scatter.fit import Data2D", "from pysasfit.toolbox.image2d import read_images")
                sc.add(f"images = read_images({path_lit(path)})",
                       f"data = Data2D.from_image(next(im for im in images if Data2D.from_image(im).name == {d.name!r}))")
            flags = self.params_panel.flags()
            if flags:
                sc.imp("from pysasfit.scatter.fit import fit")
                sc.comment("fit of the ticked parameters")
                sc.add(call("res = fit", Code("p"), flags, Code("data"), self._q_limit(self.fit_qmin),
                            self._q_limit(self.fit_qmax), every=self.every_px.value(),
                            max_nfev=self.max_nfev.value(), log_weight=self.log_weight.isChecked()),
                       "print(res.report())",
                       "p.update(res.values)")
            sc.imp("from pysasfit.scatter.fit import model_for")
            sc.add("used = data.used()" if kind == "1d" else "used = data.used(every=1)",
                   "I_model = model_for(p, data, used)   # the model at the data points")
            if kind == "1d":
                plot.append("ax1.loglog(data.q[used], I_model, 'r-', label='model at the data')")
        plot += ["ax1.set_xlabel('q (1/nm)')", "ax1.set_ylabel('I(q)')", "ax1.legend()",
                 "ext = (float(qy.min()), float(qy.max()), float(qz.min()), float(qz.max()))",
                 "ax2.imshow(np.log10(np.clip(I2, I2.max() * 1e-8, None)).T, origin='lower', extent=ext)",
                 "ax2.set_xlabel('q_y (1/nm)')", "ax2.set_ylabel('q_z (1/nm)')", "fig.tight_layout()"]
        sc.add_plot(*plot)
        return sc.text()

    # ------------------------------------------------------------------ fit
    def _q_limit(self, edit):
        t = edit.text().strip()
        return float(t) if t else None

    def on_fit(self):
        kind, d = self.current_data()
        if kind is None:
            QtWidgets.QMessageBox.information(self, "Fit", "Load data first (1D or 2D).")
            return
        flags = self.params_panel.flags()
        if not flags:
            QtWidgets.QMessageBox.information(self, "Fit", "Tick the parameters to fit (☑ next to the values).")
            return
        p = self.params()
        dd = self._data_for_model(d)
        qmin, qmax = self._q_limit(self.fit_qmin), self._q_limit(self.fit_qmax)
        every, nfev, logw = self.every_px.value(), self.max_nfev.value(), self.log_weight.isChecked()
        last = {"t": 0.0}

        def job(wk):
            def cb(work, c2):
                if time.time() - last["t"] > 0.3:
                    last["t"] = time.time()
                    wk.progress.emit(int(min(c2, 1e9) * 1000))      # χ²ᵣ × 1000 to the label
            return run_fit(p, flags, dd, qmin, qmax, every=every, max_nfev=nfev, callback=cb,
                           should_abort=lambda: wk.abort, log_weight=logw)

        self.fit_worker = Worker(job, self)
        self.fit_worker.progress.connect(lambda v: self.chi2_label.setText(f"χ²ᵣ {v / 1000:.4g} …"))
        self.fit_worker.done.connect(self._fit_done)
        self.fit_worker.failed.connect(self._fit_failed)
        self.fit_worker.finished.connect(lambda: (self.fit_button.setEnabled(True), self.fit_abort.setEnabled(False)))
        self.fit_button.setEnabled(False)
        self.fit_abort.setEnabled(True)
        self.log(f"fit of {', '.join(flags)} to {d.name} …")
        self.fit_worker.start()

    def on_fit_abort(self):
        if self.fit_worker is not None:
            self.fit_worker.abort = True

    def _fit_done(self, res):
        self.fit_text.setPlainText(res.report())
        self.chi2_label.setText(f"χ²ᵣ {res.chi2r:.4g}")
        self.log("fit: " + res.report().splitlines()[0])
        self.params_panel.update_values(res.values)
        self.calculate(two_d=self.view2d.currentData() == "model" and self.result2d is not None)

    def _fit_failed(self, msg):
        self.fit_text.setPlainText(msg)
        self.chi2_label.setText("χ²ᵣ –")
        self.log("fit: " + msg.splitlines()[0])

    # ------------------------------------------------------------------ files
    def on_open_ini(self, path=None):
        if path is None:
            path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Open parameters", self._dir(),
                                                            "parameters (*.ini);;all (*)")
            if not path:
                return
        try:
            p, flags, section = P.read_ini(path)
        except OSError as exc:
            QtWidgets.QMessageBox.warning(self, "Open parameters", str(exc))
            return
        self.ini_path = path
        self.params_panel.set_params(p, flags)
        self.log(f"parameters {os.path.basename(path)}" + (f" [{section}]" if section else ""))
        self.calculate(two_d=True)

    def on_save_ini(self):
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save parameters",
                                                        self.ini_path or os.path.join(self._dir(), "pyscatter.ini"),
                                                        "parameters (*.ini)")
        if path:
            P.write_ini(path, self.params(), self.params_panel.flags(), section="pySCAtter")
            self.ini_path = path
            self.log(f"saved {path}")

    def on_export_1d(self):
        if self.result1d is None:
            return
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Export I(q)", os.path.join(self._dir(), "iq.dat"),
                                                        "text (*.dat *.txt)")
        if path:
            q, I = self.result1d
            np.savetxt(path, np.column_stack([q, I]), header="pySCAtter model\nq [1/nm]   I")
            self.log(f"saved {path}")

    def on_export_2d(self):
        if self.result2d is None:
            return
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Export 2D pattern",
                                                        os.path.join(self._dir(), "pattern.npy"),
                                                        "NumPy (*.npy);;text (*.dat *.txt)")
        if path:
            qx, qy, qz, I = self.result2d
            if path.endswith(".npy"):
                np.save(path, np.stack([qy, qz, I]))
            else:
                np.savetxt(path, np.column_stack([qy.ravel(), qz.ravel(), I.ravel()]),
                           header="pySCAtter model\nq_h [1/nm]   q_v [1/nm]   I")
            self.log(f"saved {path}")

    # ------------------------------------------------------------------ session (launcher hooks)
    @staticmethod
    def _session_path():
        return os.path.join(os.path.dirname(prefs.settings_path()), "scatter_session.json")

    def save_session(self):
        try:
            p = self._session_path()
            os.makedirs(os.path.dirname(p), exist_ok=True)
            s = {"version": 1, "params": self.params(), "flags": self.params_panel.flags(),
                 "ini": self.ini_path, "data1d": [getattr(d, "path", "") for d in self.data1d],
                 "data2d": [getattr(d, "path", "") for d, _ in self.data2d],
                 "fit_qmin": self.fit_qmin.text(), "fit_qmax": self.fit_qmax.text(),
                 "use_dq": self.use_dq.isChecked(), "log_weight": self.log_weight.isChecked(),
                 "auto_1d": self.auto_1d.isChecked(), "auto_2d": self.auto_2d.isChecked(),
                 "view2d": self.view2d.currentData(), "log2d": self.log2d.isChecked(),
                 "iq_scale": self.iq_scale.currentData(), "hkl2d": self.hkl2d.isChecked(),
                 "lattice_view": self.lattice_view.state(), "dock_state": self.dock_state()}
            with open(p, "w", encoding="utf-8") as fh:
                json.dump(s, fh, indent=1, default=float)
        except (OSError, TypeError, ValueError):
            pass

    def restore_session(self):
        try:
            with open(self._session_path(), encoding="utf-8") as fh:
                s = json.load(fh)
        except (OSError, ValueError):
            return False
        self._quiet = True                     # the last session is shown, not calculated (Calculate 2D)
        try:
            self.ini_path = s.get("ini")
            self.params_panel.set_params(s.get("params") or {}, s.get("flags") or {})
            self.fit_qmin.setText(s.get("fit_qmin", ""))
            self.fit_qmax.setText(s.get("fit_qmax", ""))
            self.use_dq.setChecked(bool(s.get("use_dq", True)))
            self.log_weight.setChecked(bool(s.get("log_weight", False)))
            self.auto_1d.setChecked(bool(s.get("auto_1d", True)))
            self.auto_2d.setChecked(bool(s.get("auto_2d", False)))
            self.log2d.setChecked(bool(s.get("log2d", True)))
            self.hkl2d.setChecked(bool(s.get("hkl2d", False)))
            self.lattice_view.set_state(s.get("lattice_view") or {})
            for cb, key in ((self.view2d, "view2d"), (self.iq_scale, "iq_scale")):
                i = cb.findData(s.get(key))
                if i >= 0:
                    cb.setCurrentIndex(i)
            if s.get("dock_state"):
                self.restore_dock_state(s["dock_state"])
            for path in dict.fromkeys(s.get("data1d") or []):
                if path and os.path.isfile(path):
                    self.on_load_1d(path)
            for path in dict.fromkeys(s.get("data2d") or []):
                if path and os.path.isfile(path):
                    self.on_load_2d(path)
        except Exception:  # noqa: BLE001 -- an old or broken session must not stop the tool
            pass
        finally:
            self._quiet = False
        self._recalc_timer.stop()
        self.draw_1d()
        self.draw_2d()
        self.log("last session restored; press Calculate 2D (or change a parameter) to calculate")
        return True

    def forget_session(self):
        try:
            os.remove(self._session_path())
        except OSError:
            pass

    def shutdown(self):
        for w in (self.worker, self.fit_worker):
            if w is not None and w.isRunning():
                w.abort = True
                w.wait(5000)
        self.save_session()

    def closeEvent(self, event):
        self.shutdown()
        super().closeEvent(event)


def main(argv=None):
    argv = sys.argv if argv is None else argv
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(argv)
    from ..core.tabstyle import apply_tab_style
    apply_tab_style(app)
    app.setWindowIcon(tool_icon("pySCAtter"))
    w = MainWindow()
    w.show()
    files = [a for a in argv[1:] if os.path.isfile(a)]
    if files and files[0].lower().endswith(".ini"):
        w.on_open_ini(files[0])
    else:
        w.restore_session()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
