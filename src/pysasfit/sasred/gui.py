# SPDX-License-Identifier: BSD-3-Clause
"""pySASred Qt tool: SANS-I raw data to I(q) with full error propagation.

    pysasfit-sasred [raw-data folder]
    python -m pysasfit.sasred.gui [raw-data folder]

Also hosted as the "pySASred" tab of the pySASfit launcher (the first tab).

Panels (every one detachable, View menu)::

    Runs           raw-data folder and the header table of its runs
    Reduction      beam centre (from a direct-beam run or by hand), first-guess beamstop,
                   q bins, units (nm or Å), normalisation, sector, live I(q);
                   reduce the selected runs; save
    Detector       2D image of the selected run (colour scale, cartesian or polar view)
                   with beam centre and mask; mask tools (suggest, draw, undo, load, save)
    I(q)           reduced curves with the statistical error bars (Idev) and the
                   live radial average of the selected run
    Error budget   relative standard deviation of each source versus q
    Log            notes of every reduction (e.g. "uncertainty unknown, not propagated")

Numbers that are not on the panels (beam-centre disc radius and tolerance,
start values of the panel fields) are in Settings → Preferences → SANS reduction.
"""
from __future__ import annotations

import json
import os
import sys
import warnings

import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets

warnings.filterwarnings("ignore", message="constrained_layout not applied")

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402

from .efficiency_gui import EfficiencyMixin  # noqa: E402
from matplotlib.ticker import LogFormatterSciNotation, LogLocator, NullFormatter  # noqa: E402

from ..core import settings as prefs  # noqa: E402
from ..core.colorscale import SCALES, decorate_colorbar, make_norm  # noqa: E402
from ..core.docking import DockMixin, plot_panel  # noqa: E402
from ..core.settings_dialog import add_preferences_menu  # noqa: E402
from ..resources import tool_icon  # noqa: E402
from . import __version__  # noqa: E402
from .units import units as get_units  # noqa: E402

__all__ = ["MainWindow", "main"]

Area = QtCore.Qt.DockWidgetArea


LAYOUT_VERSION = 3       # 2: detector and reduction table as tabs in the middle column; 3: column widths


def _session_path():
    return os.path.join(os.path.dirname(prefs.settings_path()), "sasred_session.json")


def _dspin(lo, hi, val, dec=3, step=None, suffix="", special=None):
    w = QtWidgets.QDoubleSpinBox()
    w.setRange(lo, hi)
    w.setDecimals(dec)
    w.setValue(val)
    if step:
        w.setSingleStep(step)
    if suffix:
        w.setSuffix(suffix)
    if special:
        w.setSpecialValueText(special)
    w.setKeyboardTracking(False)
    return w


def _log_q_ticks(ax):
    """Readable q axis on a log scale: below about 1.5 decades also label 2 and 5 between the
    decades (otherwise a curve over one decade shows a single tick label); above, decades only."""
    lo, hi = ax.get_xlim()
    if lo > 0 and np.log10(hi / lo) < 1.5:
        ax.xaxis.set_minor_locator(LogLocator(subs=(2.0, 5.0)))
        ax.xaxis.set_minor_formatter(LogFormatterSciNotation(labelOnlyBase=False, minor_thresholds=(10, 10)))
    else:
        ax.xaxis.set_minor_formatter(NullFormatter())


def _canvas():
    fig = Figure(figsize=(5, 4), layout="constrained")
    return FigureCanvasQTAgg(fig)


def _qerr(r, u):
    """Horizontal error bars of a reduced curve: the q resolution Qdev (1 σ) in the display unit, or None
    when it is not determined (no direct beam and no collimation estimate)."""
    qs = getattr(r, "q_sigma", None)
    if qs is None:
        return None
    qs = np.asarray(u.q_out(np.asarray(qs, float)), float)
    q = np.asarray(u.q_out(np.asarray(r.q, float)), float)
    return np.minimum(np.where(np.isfinite(qs), qs, 0.0), 0.999 * q)       # q − σ stays > 0 on the log axis


def _lam_column() -> int:
    from .catalogue import COLUMNS
    return COLUMNS.index("λ [nm]")


def load_instrument_default(raw):
    """The instrument file of a run as read (for "Copy Python code": what the GUI added to it)."""
    from .reduce import load_instrument
    return load_instrument(raw)


class MainWindow(EfficiencyMixin, DockMixin, QtWidgets.QMainWindow):
    instances = __import__("weakref").WeakSet()      # open windows (pyTOOLbox takes their results)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle(f"pySASred {__version__} – SANS data reduction (SINQ SANS-I)")
        self.setWindowIcon(tool_icon("pySASred"))
        self.folder = ""
        self.entries = []            # CatalogueEntry list of the folder
        self.name_corrections = None # sample names / roles corrected by hand (names.Corrections)
        self.results = {}            # run -> ReducedRun
        self.beam_centre = None      # BeamCentre of the last determination
        self.monitor_tests = {}      # monitor -> core.monitors.MonitorTest (before/after the selector)
        self.centres = {}            # geometry (SD, λ) -> GeometryCentre: q and solid angle use it
        self.profiles = {}           # geometry -> BeamProfile of its direct beam: the resolution uses it
        self.masks = {}              # instrument configuration (catalogue.mask_key) -> edited uint32 mask
        self.tregions = {}           # instrument configuration -> bool transmission region (semi-transparent beam stop)
        self._mask_key = None        # configuration of the mask in the editor
        self._centre_sync = False
        self.preview = None          # on-the-fly radial average of the selected run (not kept)
        self.preview_banks = []      # SANS-LLB, view "all detectors": the live averages of the other banks
        self.unit = prefs.get("sasred.units")      # "nm" or "A": display and output units
        self._preview_timer = QtCore.QTimer(self)
        self._preview_timer.setSingleShot(True)
        self._preview_timer.setInterval(150)       # debounce typing / spinning
        self._preview_timer.timeout.connect(self.update_preview)
        fmenu = self.menuBar().addMenu("&File")
        fmenu.addAction("Open raw-data &folder…", self.on_browse)
        fmenu.addAction("Save &NXcanSAS…", self.on_save_nxcansas)
        fmenu.addAction("Save &ASCII…", self.on_save_ascii)
        fmenu.addAction("Save &2D NXcanSAS…", self.on_save_2d)
        fmenu.addSeparator()
        fmenu.addAction("Save masks of all configurations…", self.on_save_masks)
        fmenu.addAction("Load masks of configurations…", self.on_load_masks)
        fmenu.addAction("Save &all reduced runs…", self.on_save_all)
        from ..core import pycode
        pycode.install(self, self.python_code, menu=fmenu)
        add_preferences_menu(self, "sasred")
        self._build_runs()
        self._build_reduction()
        self._build_plots()
        self._build_log()
        self._build_table()
        self.splitDockWidget(self.docks["runs"], self.docks["reduction"], QtCore.Qt.Orientation.Vertical)
        self.tabifyDockWidget(self.docks["iq"], self.docks["budget"])
        self.tabifyDockWidget(self.docks["iq"], self.docks["log"])
        self.tabifyDockWidget(self.docks["iq"], self.docks["missing"])
        self.docks["iq"].raise_()
        self._finish_docks()
        self._middle_tabs_done = False
        self.statusBar().showMessage("Choose the raw-data folder (File → Open raw-data folder)")
        self.resize(1400, 900)
        MainWindow.instances.add(self)
        # the session is also written every two minutes: a launcher that is killed (closed console,
        # crash, logout) does not lose the work since the start
        self._autosave = QtCore.QTimer(self)
        self._autosave.setInterval(120_000)
        self._autosave.timeout.connect(self._autosave_session)
        self._autosave.start()
        app = QtWidgets.QApplication.instance()
        if app is not None:
            app.aboutToQuit.connect(self._autosave_session)
        # preferences that set panel controls: a change in Settings → Preferences is applied at once
        self._pref_seen = {k: prefs.get(k) for k in self._PANEL_PREFS}
        from ..core.settings_dialog import on_saved
        import weakref
        ref = weakref.ref(self)
        on_saved(lambda: ref() is not None and ref().apply_preferences())

    # ------------------------------------------------------------------ panels
    def _build_runs(self):
        from .catalogue import COLUMNS
        w = QtWidgets.QWidget()
        lay = QtWidgets.QVBoxLayout(w)
        row = QtWidgets.QHBoxLayout()
        self.folder_edit = QtWidgets.QLineEdit()
        self.folder_edit.setPlaceholderText("raw-data folder (sans<year>n<run>.hdf)")
        self.folder_edit.returnPressed.connect(lambda: self.set_folder(self.folder_edit.text()))
        b = QtWidgets.QPushButton("Browse…")
        b.clicked.connect(self.on_browse)
        r = QtWidgets.QPushButton("Rescan")
        r.clicked.connect(lambda: self.set_folder(self.folder_edit.text()))
        row.addWidget(self.folder_edit, 1)
        row.addWidget(b)
        row.addWidget(r)
        lay.addLayout(row)
        self.filter_edit = QtWidgets.QLineEdit()
        self.filter_edit.setPlaceholderText("filter: part of the sample name or run number")
        self.filter_edit.textChanged.connect(self._apply_filter)
        lay.addWidget(self.filter_edit)
        t = QtWidgets.QTableWidget(0, len(COLUMNS))
        t.setHorizontalHeaderLabels(COLUMNS)
        t.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows)
        t.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.ExtendedSelection)
        t.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
        t.verticalHeader().setVisible(False)
        t.horizontalHeader().setStretchLastSection(True)
        t.itemSelectionChanged.connect(self._on_selection)
        t.setContextMenuPolicy(QtCore.Qt.ContextMenuPolicy.CustomContextMenu)
        t.customContextMenuRequested.connect(self._runs_menu)
        self.table = t
        lay.addWidget(t, 1)
        nb = QtWidgets.QPushButton("Sample names and roles…")
        nb.setToolTip("Correct the sample name and/or set the role (water, empty cell, …) of the selected runs, "
                      "e.g. when two cells were swapped or a remeasurement got a long name. The raw files stay "
                      "untouched; the corrections are kept for this folder and used by the grouping and the "
                      "reduction table. Also with the right mouse button in the run table.")
        nb.clicked.connect(self.on_names)
        xb = QtWidgets.QPushButton("Exclude…")
        xb.setToolTip("Exclude the selected runs from every reduction (sample preparation failed, phase "
                      "separation during the measurement, …) with a reason. Kept for this folder, also after "
                      "Rescan; the runs stay in the list, struck through. Right mouse button: Include runs again.")
        xb.clicked.connect(self.on_exclude)
        row = QtWidgets.QHBoxLayout()
        row.addWidget(nb, 1)
        row.addWidget(xb)
        lay.addLayout(row)
        self._add_dock("runs", "Runs", w, Area.LeftDockWidgetArea)

    def _build_reduction(self):
        v = prefs.values("sasred")
        w = QtWidgets.QWidget()
        lay = QtWidgets.QVBoxLayout(w)

        g = QtWidgets.QGroupBox("Beam centre [pixel]")
        g.setToolTip("Pixel indices count from 0, as in GRASP and Python")
        f = QtWidgets.QFormLayout(g)
        f.setRowWrapPolicy(QtWidgets.QFormLayout.RowWrapPolicy.WrapLongRows)
        self.xc = _dspin(-1000, 1000, 63.5, 3, 0.1)
        self.yc = _dspin(-1000, 1000, 63.5, 3, 0.1)
        f.addRow("x (axis 1)", self.xc)
        f.addRow("y (axis 0)", self.yc)
        b = QtWidgets.QPushButton("From selected run (direct beam)")
        b.setToolTip("Masked centre of gravity of the first selected run (attenuated direct beam)")
        b.clicked.connect(self.on_beam_centre)
        f.addRow(b)
        self.bc_label = QtWidgets.QLabel("")
        self.bc_label.setWordWrap(True)
        f.addRow(self.bc_label)
        lay.addWidget(g)

        g = QtWidgets.QGroupBox("Mask and q bins")
        f = QtWidgets.QFormLayout(g)
        f.setRowWrapPolicy(QtWidgets.QFormLayout.RowWrapPolicy.WrapLongRows)     # narrow panel: field below its label
        # first guess of the beamstop: put into the mask of a new image, then adjusted with the mask tools
        self.bs_shape = QtWidgets.QComboBox()
        self.bs_shape.addItem("square", "square")
        self.bs_shape.addItem("disc", "disc")
        self.bs_shape.setCurrentIndex(max(0, self.bs_shape.findData(v["mask.beamstop_shape"])))
        self.bs_radius = _dspin(0, 64, v["mask.beamstop_radius_px"], 1, 0.5, " px")
        self.bs_radius.setToolTip("half width of the square (radius of the disc) around the beam centre")
        bs_put = QtWidgets.QPushButton("Into mask")
        bs_put.setToolTip("replace the beamstop pixels of the mask by this first guess (undoable)")
        bs_put.clicked.connect(lambda: self.mask_editor.place_guess())
        row = QtWidgets.QHBoxLayout()
        for wdg in (self.bs_shape, self.bs_radius, bs_put):
            row.addWidget(wdg)
        row.setSpacing(3)
        bs_put.setSizePolicy(QtWidgets.QSizePolicy.Policy.Minimum, QtWidgets.QSizePolicy.Policy.Fixed)
        lab = QtWidgets.QLabel("beamstop")
        lab.setToolTip("First guess of the beamstop shadow in the mask of a new image (the SANS-I beamstops are "
                       "square). Adjust it with the mask tools or use Suggest; until the mask is edited it "
                       "follows the beam centre and the size.")
        f.addRow(lab, row)
        self.nbins = QtWidgets.QSpinBox()
        self.nbins.setRange(2, 2000)
        self.nbins.setValue(v["qbins.n"])
        f.addRow("number of bins", self.nbins)
        self.unit_combo = QtWidgets.QComboBox()
        self.unit_combo.addItem("nm, 1/nm", "nm")
        self.unit_combo.addItem("Å, 1/Å", "A")
        self.unit_combo.setCurrentIndex(max(0, self.unit_combo.findData(self.unit)))
        self.unit_combo.setToolTip("units of wavelength and q on the panels, in the plots and in the saved files")
        f.addRow("units", self.unit_combo)
        u = get_units(self.unit)
        self.qmin = _dspin(0, 100, 0.0, 5, 0.001, " " + u.q, special="auto")
        self.qmax = _dspin(0, 100, 0.0, 5, 0.001, " " + u.q, special="auto")
        f.addRow("q min", self.qmin)
        f.addRow("q max", self.qmax)
        self.spacing = QtWidgets.QComboBox()
        self.spacing.addItem("logarithmic", "log")
        self.spacing.addItem("linear", "lin")
        self.spacing.setCurrentIndex(max(0, self.spacing.findData(v["qbins.spacing"])))
        f.addRow("spacing", self.spacing)
        self.bank_combo = QtWidgets.QComboBox()
        for t, d in (("central detector", "central"), ("left wide-angle panel", "left"),
                     ("bottom wide-angle panel", "bottom"), ("central + both panels", "all")):
            self.bank_combo.addItem(t, d)
        self.bank_combo.setToolTip("SANS-LLB: which detector banks are reduced (Reduce selected and the reduction "
                                   "table). The panels use the beam centre of the central detector and their "
                                   "offsets from Settings → Preferences → pySASred → SANS-LLB detector geometry; "
                                   "their mask is the tube ends and dead tubes. SANS-I: central only.")
        f.addRow("detector", self.bank_combo)
        self.norm = QtWidgets.QComboBox()
        self.norm.addItem("monitor 1", "monitor")
        self.norm.addItem("counting time", "time")
        self.norm.setCurrentIndex(max(0, self.norm.findData(v["normalise_by"])))
        f.addRow("normalise by", self.norm)
        self.sector = QtWidgets.QCheckBox("sector average")
        self.sector_c = _dspin(-180, 180, 0.0, 1, 5, " °")
        self.sector_hw = _dspin(0.5, 90, 15.0, 1, 5, " °")
        f.addRow(self.sector)
        f.addRow("sector centre φ", self.sector_c)
        f.addRow("half width", self.sector_hw)
        self.sector.toggled.connect(self._sector_enabled)
        self._sector_enabled(False)
        self.live = QtWidgets.QCheckBox("live I(q) of the selected run")
        self.live.setChecked(True)
        self.live.setToolTip("radial (or sector) average of the run selected in the table, updated on the fly "
                             "with the current beam centre, mask and bins (dashed curve in I(q))")
        f.addRow(self.live)
        lay.addWidget(g)

        row = QtWidgets.QHBoxLayout()
        self.reduce_button = QtWidgets.QPushButton("Reduce selected")
        self.reduce_button.setDefault(True)
        self.reduce_button.clicked.connect(self.on_reduce)
        clr = QtWidgets.QPushButton("Clear curves")
        clr.clicked.connect(self.clear_results)
        row.addWidget(self.reduce_button, 1)
        row.addWidget(clr)
        from ..core import pycode
        row.addWidget(pycode.button(self.python_code, w))
        lay.addLayout(row)

        g = QtWidgets.QGroupBox("Reduced runs")
        gl = QtWidgets.QVBoxLayout(g)
        self.result_combo = QtWidgets.QComboBox()
        self.result_combo.currentIndexChanged.connect(
            lambda _i: (self._draw_budget(), self._draw_detector() if self.det_view.currentData() == "reduced2d" else None))
        gl.addWidget(self.result_combo)
        row = QtWidgets.QHBoxLayout()
        for text, slot in (("NXcanSAS…", self.on_save_nxcansas), ("ASCII…", self.on_save_ascii),
                           ("All…", self.on_save_all)):
            bb = QtWidgets.QPushButton(text)
            bb.clicked.connect(slot)
            row.addWidget(bb)
        gl.addLayout(row)
        lay.addWidget(g)
        lay.addStretch(1)
        for wdg in (self.xc, self.yc):
            wdg.valueChanged.connect(lambda _v: self._on_centre_edit())
        for wdg in (self.xc, self.yc, self.bs_radius):
            wdg.valueChanged.connect(lambda _v: (self.mask_editor.follow_guess(), self._draw_detector()))
        self.bs_shape.currentIndexChanged.connect(
            lambda _i: (self.mask_editor.follow_guess(), self._draw_detector(), self._schedule_preview()))
        for wdg in (self.xc, self.yc, self.bs_radius, self.qmin, self.qmax, self.sector_c, self.sector_hw):
            wdg.valueChanged.connect(lambda _v: self._schedule_preview())
        self.nbins.valueChanged.connect(lambda _v: self._schedule_preview())
        for c in (self.spacing, self.norm):
            c.currentIndexChanged.connect(lambda _i: self._schedule_preview())
        self.sector.toggled.connect(lambda _on: (self._schedule_preview(), self._draw_detector()))
        self.live.toggled.connect(lambda _on: self._schedule_preview())
        self.unit_combo.currentIndexChanged.connect(lambda _i: self.set_units(self.unit_combo.currentData()))
        self._add_dock("reduction", "Reduction", w, Area.LeftDockWidgetArea, scrollable=True)

    def _sector_enabled(self, on):
        self.sector_c.setEnabled(on)
        self.sector_hw.setEnabled(on)

    def _build_plots(self):
        self.det_canvas = _canvas()
        from ..core.flowlayout import FlowLayout
        bar_w = QtWidgets.QWidget()                     # wraps in a narrow panel (no minimum width)
        bar = FlowLayout(bar_w)
        sp = bar_w.sizePolicy()
        sp.setHeightForWidth(True)
        sp.setHorizontalPolicy(QtWidgets.QSizePolicy.Policy.Ignored)
        bar_w.setSizePolicy(sp)
        def labelled(text, w):
            box = QtWidgets.QWidget()
            hl = QtWidgets.QHBoxLayout(box)
            hl.setContentsMargins(0, 0, 0, 0)
            hl.setSpacing(3)
            hl.addWidget(QtWidgets.QLabel(text))
            hl.addWidget(w)
            return box
        self.det_scale = QtWidgets.QComboBox()
        for key, label in SCALES.items():
            self.det_scale.addItem(label, key)
        self.det_scale.setToolTip("linear: I;  log(1+I): zero stays finite;  symlog: linear below 1 count, "
                                  "logarithmic above;  asinh: smooth version of symlog")
        self.det_scale.setCurrentIndex(max(0, self.det_scale.findData(prefs.get("sasred.detector.scale"))))
        self.det_scale.currentIndexChanged.connect(lambda _i: self._draw_detector())
        bar.addWidget(labelled("colour scale", self.det_scale))
        self.det_cmap = QtWidgets.QComboBox()
        self.det_cmap.addItems(["viridis", "inferno", "magma", "cividis", "turbo", "jet", "gray"])
        self.det_cmap.currentIndexChanged.connect(lambda _i: self._draw_detector())
        bar.addWidget(labelled("colours", self.det_cmap))
        self.det_view = QtWidgets.QComboBox()
        self.det_view.addItem("cartesian (x, y)", "cartesian")
        self.det_view.addItem("polar (φ, q)", "polar")
        self.det_view.addItem("reduced 2D (result chosen)", "reduced2d")
        self.det_view.addItem("SANS-LLB: all detectors (2θ or q)", "llb_all")
        self.det_view.addItem("efficiency map", "efficiency")
        self.det_view.setToolTip("polar: azimuth φ horizontally (0° = horizontal direction +x, counter-clockwise), "
                                 "q vertically; rings become horizontal lines, anisotropy shows as waves")
        self.det_view.currentIndexChanged.connect(lambda _i: (self._draw_detector(), self._schedule_preview()))
        for cb in (self.det_view, self.bank_combo):        # as wide as a short item; the list shows them in full
            cb.setSizeAdjustPolicy(QtWidgets.QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
            cb.setMinimumContentsLength(14)
        bar.addWidget(labelled("view", self.det_view))
        self.det_axes = QtWidgets.QComboBox()
        self.det_axes.addItem("pixels", "pixels")
        self.det_axes.addItem("angle 2θ", "angle")
        self.det_axes.addItem("q_x, q_y", "q")
        self.det_axes.setToolTip(
            "Coordinates of the cartesian view and of 'SANS-LLB: all detectors'. angle: 2θ·(cos φ, sin φ) [°]; "
            "q: |q|·(cos φ, sin φ) — every pixel at its scattering angle or scattering vector (beam centre of the "
            "geometry, SANS-LLB panel offsets of the preferences), so rings are circles at their 2θ or q and the "
            "central detector and the panels sit at their true places. Masks are edited in pixels.")
        self.det_axes.currentIndexChanged.connect(lambda _i: self._draw_detector())
        bar.addWidget(labelled("axes", self.det_axes))
        self.det_sep = QtWidgets.QCheckBox("own scale for the panels")
        self.det_sep.setChecked(True)
        self.det_sep.setToolTip("SANS-LLB: all detectors: the central detector and the two wide-angle panels get "
                                "separate colour ranges (each with its own colour bar), so that both are readable")
        self.det_sep.toggled.connect(lambda _on: self._draw_detector())
        bar.addWidget(self.det_sep)
        self.det_cmap_panels = QtWidgets.QComboBox()
        self.det_cmap_panels.addItem("as central", "")
        self.det_cmap_panels.addItems(["inferno", "magma", "viridis", "cividis", "turbo", "jet", "gray"])
        self.det_cmap_panels.setToolTip("SANS-LLB: all detectors: colour map of the left and bottom panels")
        self.det_cmap_panels.currentIndexChanged.connect(lambda _i: self._draw_detector())
        bar.addWidget(labelled("panel colours", self.det_cmap_panels))
        self.det_bank = QtWidgets.QComboBox()
        for t, d in (("central", ""), ("left panel", "left"), ("bottom panel", "bottom")):
            self.det_bank.addItem(t, d)
        self.det_bank.setToolTip("SANS-LLB: detector shown and masked (each panel has its own mask per configuration; "
                                 "a new panel mask starts with the tube ends and dead tubes)")
        self.det_bank.currentIndexChanged.connect(
            lambda _i: (self.sync_shown_efficiency(), self._draw_detector(), self._schedule_preview()))
        bar.addWidget(labelled("detector", self.det_bank))
        self.det_eff = QtWidgets.QCheckBox("÷ efficiency")
        self.det_eff.setToolTip("Divide the raw image by the efficiency map shown (button 'efficiency map'); pixels "
                                "without an estimate are left blank")
        self.det_domega = QtWidgets.QCheckBox("÷ solid angle")
        self.det_domega.setToolTip("Divide the raw image by the solid angle of every pixel (counts per steradian; "
                                   "with the beam centre of the geometry)")
        for cb in (self.det_eff, self.det_domega):
            cb.toggled.connect(lambda _on: self._draw_detector())
            bar.addWidget(cb)
        effb = QtWidgets.QToolButton()
        effb.setText("efficiency map")
        effb.setPopupMode(QtWidgets.QToolButton.ToolButtonPopupMode.InstantPopup)
        em = QtWidgets.QMenu(effb)
        mk_menu = em.addMenu("Make from the water of")
        mk_menu.aboutToShow.connect(lambda m=mk_menu: self.fill_make_menu(m))
        em.addAction("Make from the water of the shown geometry", self.on_make_efficiency)
        em.addAction("Load…", self.on_load_efficiency)
        em.addAction("Save the shown map…", self.on_save_efficiency)
        em.addAction("Archive the shown map", self.on_archive_efficiency)
        em.addAction("Show", lambda: self.det_view.setCurrentIndex(self.det_view.findData("efficiency")))
        effb.setMenu(em)
        bar.addWidget(effb)
        self.eff_label = QtWidgets.QLabel("no efficiency map")
        bar.addWidget(self.eff_label)
        from .mask_editor import MaskEditor
        self.mask_editor = MaskEditor(self)                 # mask: suggest, draw, undo, load / save
        self.mask_editor.changed.connect(self._schedule_preview)
        tools = QtWidgets.QToolButton()
        tools.setText("mask tools")
        tools.setCheckable(True)
        tools.setChecked(True)
        tools.setToolTip("show / hide the mask tools (more room for the image in a small panel)")
        tools.toggled.connect(lambda on: (self.mask_editor.bar.setVisible(on), self.mask_cfg.setVisible(on)))
        bar.addWidget(tools)
        self.mask_tools_button = tools
        self.mask_cfg = QtWidgets.QLabel("")
        self.mask_cfg.setWordWrap(False)
        self.mask_cfg.setSizePolicy(QtWidgets.QSizePolicy.Policy.Ignored, QtWidgets.QSizePolicy.Policy.Fixed)
        self.mask_cfg.setToolTip("Each instrument configuration (coll, SD, λ, beam stop type and position) has its own "
                                 "mask: selecting a run of another configuration switches the mask; reductions use "
                                 "the mask of their configuration (or the suggested one if none was edited)")
        self._add_dock("detector", "Detector", plot_panel(self.det_canvas, self,
                                                          bars=(bar_w, self.mask_editor.bar, self.mask_cfg)),
                       Area.RightDockWidgetArea)
        self.iq_canvas = _canvas()
        iq_bar = QtWidgets.QHBoxLayout()
        self.iq_show = QtWidgets.QComboBox()
        self.iq_show.addItem("selected runs / table rows", "selected")
        self.iq_show.addItem("last reduction", "last")
        self.iq_show.addItem("all reduced", "all")
        self.iq_show.setToolTip("Which reduced curves I(q) shows. 'selected': the runs selected in the run table and "
                                "the rows selected in the reduction table (the last reduction when none of them "
                                "is reduced). The legend is shown up to 12 curves.")
        self.iq_show.currentIndexChanged.connect(lambda _i: self._draw_iq())
        self.iq_offset = QtWidgets.QDoubleSpinBox()
        self.iq_offset.setRange(1.0, 1000.0)
        self.iq_offset.setDecimals(1)
        self.iq_offset.setPrefix("offset × ")
        self.iq_offset.setToolTip("successive curves multiplied by this factor (1: none)")
        self.iq_offset.valueChanged.connect(lambda _v: self._draw_iq())
        self.iq_err_I = QtWidgets.QCheckBox("I error bars")
        self.iq_err_I.setToolTip("vertical error bars: Idev (statistical, 1 σ)")
        self.iq_err_q = QtWidgets.QCheckBox("q resolution bars")
        self.iq_err_q.setToolTip("horizontal bars: Qdev (1 σ), when the resolution is known (direct beam or "
                                 "collimation estimate)")
        for cb in (self.iq_err_I, self.iq_err_q):
            cb.setChecked(True)
            cb.toggled.connect(lambda _on: self._draw_iq())
        self.iq_xscale, self.iq_yscale = QtWidgets.QComboBox(), QtWidgets.QComboBox()
        for cb, axis in ((self.iq_xscale, "q"), (self.iq_yscale, "I")):
            cb.addItem("log", "log")
            cb.addItem("lin", "linear")
            cb.setToolTip(f"{axis} axis scale; kept when other runs or rows are shown (also a change made with the "
                          f"plot toolbar or the keys k / l is kept). Log I: points with I ≤ 0 are not drawn.")
            cb.currentIndexChanged.connect(lambda _i: self._draw_iq())
        from ..core.flowlayout import flow_bar
        iq_bar.addWidget(flow_bar(labelled("show", self.iq_show), self.iq_offset, labelled("q", self.iq_xscale),
                                  labelled("I", self.iq_yscale), self.iq_err_I, self.iq_err_q), 1)
        self._last_keys = []
        self._add_dock("iq", "I(q)", plot_panel(self.iq_canvas, self, bars=(iq_bar,)), Area.RightDockWidgetArea)
        self.budget_canvas = _canvas()
        self._add_dock("budget", "Error budget", plot_panel(self.budget_canvas, self), Area.RightDockWidgetArea)
        self.splitDockWidget(self.docks["detector"], self.docks["iq"], QtCore.Qt.Orientation.Horizontal)

    def _build_log(self):
        self.log_view = QtWidgets.QPlainTextEdit()
        self.log_view.setReadOnly(True)
        self._add_dock("log", "Log", self.log_view, Area.RightDockWidgetArea)
        self.missing_view = QtWidgets.QTreeWidget()
        self.missing_view.setHeaderLabels(["Geometry", "Row", "Input", "What the reduction did"])
        self.missing_view.setRootIsDecorated(False)
        self.missing_view.setAlternatingRowColors(True)
        self.missing_view.setToolTip("Inputs missing in the last reduction (red: missing, orange: to check, dark red: "
                                     "the row was not reduced, grey: optional correction left out). Double-click: the "
                                     "cell in the reduction table.")
        self.missing_view.itemDoubleClicked.connect(self._on_missing_clicked)
        self._add_dock("missing", "Missing input", self.missing_view, Area.RightDockWidgetArea)

    def _build_table(self):
        from .table_gui import ReductionTableWidget
        self.table_panel = ReductionTableWidget(lambda: self.entries, self.centre_for, self, log=self.log)
        self.table_panel.reduce_rows = self.reduce_table_rows
        self.table_panel.python_code_rows = self.python_code_table
        self.table_panel.show_efficiency = self.show_table_efficiency
        self.table_panel.rows_selected.connect(self._draw_iq)
        self.table_panel.changed.connect(self._schedule_check)      # a table edited: check again
        self.table_panel.tabs.currentChanged.connect(lambda _i: self._draw_iq())
        self._add_dock("table", "Reduction table", self.table_panel, Area.RightDockWidgetArea,
                       scrollable=True)                    # no large minimum size: scrolls in a small window
        m = self.menuBar().addMenu("&Reduction")
        m.addAction("Reduction &table", self.show_table)
        m.addAction("&Fill table from runs", lambda: (self.show_table(), self.table_panel.on_fill()))
        m.addAction("Add &many samples… (select runs, then group)",
                    lambda: (self.show_table(), self.table_panel.on_add_many()))
        m.addAction("&Empty table (fill by hand)", lambda: (self.show_table(), self.table_panel.on_clear()))
        m.addAction("&Check all geometries (missing input)", lambda: (self.check_all_geometries(),
                                                                       self.docks["missing"].raise_()))
        m.addSeparator()
        m.addAction("&Reduce geometry shown (2D, then I(q))", lambda: self.table_panel.page.on_reduce(all_rows=True))
        m.addAction("Reduce &selected rows", lambda: self.table_panel.page.on_reduce(all_rows=False))
        m.addSeparator()
        m.addAction("Transmission through the semi-transparent &beam stop…", self.on_beamstop_transmission)
        m.addSeparator()
        m.addAction("Send results to pyTOOLbox &merge (no files)", self.on_send_to_merge)

    def on_beamstop_transmission(self):
        """T from the counts in the transmission region (semi-transparent beam stop) of every row of the
        reduction table shown, next to the transmission from the transmission runs."""
        from .beamstop_gui import BeamstopTransmissionDialog
        self._commit_mask()
        self.show_table()
        dlg = BeamstopTransmissionDialog(self)
        self._beamstop_dialog = dlg
        dlg.show()
        return dlg

    def _runs_menu(self, pos):
        m = QtWidgets.QMenu(self)
        m.addAction("Sample names and roles…", self.on_names)
        m.addAction("Reset name and role to the file", lambda: self.set_names([e.run for e in self.selected_entries()]))
        m.addSeparator()
        m.addAction("Exclude runs… (kept for this folder)", self.on_exclude)
        m.addAction("Include runs again", lambda: self.include_runs([e.run for e in self.selected_entries()]))
        m.exec(self.table.viewport().mapToGlobal(pos))

    def on_names(self):
        sel = [e for e in self.selected_entries() if e.raw is not None]
        if not sel:
            QtWidgets.QMessageBox.information(self, "Sample names and roles", "Select runs in the run table first.")
            return
        from .names import ROLE_CHOICES
        from .plan import role_of
        d = QtWidgets.QDialog(self)
        d.setWindowTitle("Sample names and roles")
        f = QtWidgets.QFormLayout(d)
        infile = sorted({e.raw.sample_name_in_file or e.raw.sample_name for e in sel})
        lab = QtWidgets.QLabel(f"{len(sel)} run(s): {', '.join(str(e.run) for e in sel[:12])}"
                               + (" …" if len(sel) > 12 else "") + "\nname in the file: " + "; ".join(infile))
        lab.setWordWrap(True)
        f.addRow(lab)
        name = QtWidgets.QLineEdit(sel[0].raw.sample_name if len({e.raw.sample_name for e in sel}) == 1 else "")
        name.setPlaceholderText("empty: the name in the file")
        f.addRow("sample name", name)
        role = QtWidgets.QComboBox()
        for k, v in ROLE_CHOICES.items():
            role.addItem(v, k)
        cur = {getattr(e.raw, "role", "") for e in sel}
        if len(cur) == 1:
            role.setCurrentIndex(max(0, role.findData(cur.pop())))
        f.addRow("role", role)
        hint = QtWidgets.QLabel("")
        hint.setStyleSheet("color: grey")

        def upd():
            n = name.text().strip() or (infile[0] if infile else "")
            hint.setText(f"role from the name: {role_of(n)}")
        name.textChanged.connect(lambda _t: upd())
        upd()
        f.addRow("", hint)
        note = QtWidgets.QLineEdit()
        note.setPlaceholderText("why (optional), e.g. cells swapped: this is D2O")
        f.addRow("note", note)
        bb = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.StandardButton.Ok
                                        | QtWidgets.QDialogButtonBox.StandardButton.Cancel)
        bb.accepted.connect(d.accept)
        bb.rejected.connect(d.reject)
        f.addRow(bb)
        if d.exec() == QtWidgets.QDialog.DialogCode.Accepted:
            self.set_names([e.run for e in sel], name.text().strip(), role.currentData(), note.text().strip())

    def set_names(self, runs, name: str = "", role: str = "", note: str = ""):
        """Correct sample name and/or role of runs (empty: back to the file / the name), keep it for the
        folder, re-apply to the run table and refill the reduction table."""
        from . import names as nm
        if not self.folder or not runs:
            return
        if self.name_corrections is None:
            self.name_corrections = nm.load(self.folder)
        in_file = {e.run: e.raw.sample_name_in_file or e.raw.sample_name for e in self.entries if e.raw is not None}
        keep_name = [r for r in runs if name and in_file.get(r) == name]     # same as the file: no rename
        self.name_corrections.set([r for r in runs if r not in keep_name], name or None, role or None, note)
        if keep_name:
            self.name_corrections.set(keep_name, None, role or None, note)
        p = nm.save(self.name_corrections)
        self._apply_corrections()
        what = (f"name '{name}'" if name else "name from the file") + (f", role {role}" if role else ", role from the name")
        self.log(f"runs {', '.join(map(str, runs))}: {what} (kept in {p})")
        self._after_corrections("reduction table filled again with the corrected names and roles")

    def _apply_corrections(self):
        """Names, roles and exclusions of the folder onto the runs; the run list's skip reasons follow."""
        from . import names as nm
        for e in self.entries:
            if e.raw is not None:
                nm.apply(e.raw, self.name_corrections)
                e.skip = e.raw.empty_reason

    def exclude_runs(self, runs, reason: str = ""):
        """Exclude runs from every reduction (kept for the folder, also after Rescan); they are greyed and
        struck through in the run list with the reason, never selected for a table and not reduced."""
        from . import names as nm
        if not self.folder or not runs:
            return
        if self.name_corrections is None:
            self.name_corrections = nm.load(self.folder)
        self.name_corrections.exclude(runs, reason)
        p = nm.save(self.name_corrections)
        self._apply_corrections()
        self.log(f"runs {', '.join(map(str, runs))} excluded: {reason or 'no reason given'} (kept in {p})")
        self._after_corrections("reduction table filled again without the excluded runs")

    def include_runs(self, runs):
        """Take back the exclusion of runs."""
        from . import names as nm
        if not self.folder or not runs or self.name_corrections is None:
            return
        self.name_corrections.include(runs)
        p = nm.save(self.name_corrections)
        self._apply_corrections()
        self.log(f"runs {', '.join(map(str, runs))} included again (kept in {p})")
        self._after_corrections("reduction table filled again with the runs included")

    def on_exclude(self):
        sel = [e.run for e in self.selected_entries() if e.raw is not None]
        if not sel:
            QtWidgets.QMessageBox.information(self, "Exclude runs", "Select runs in the run table first.")
            return
        reason, ok = QtWidgets.QInputDialog.getText(
            self, "Exclude runs", f"Exclude run(s) {', '.join(map(str, sel[:12]))}{' …' if len(sel) > 12 else ''} "
            "from every reduction.\nReason (e.g. sample preparation failed, phase separation during the run):")
        if ok:
            self.exclude_runs(sel, reason.strip())

    def _after_corrections(self, msg):
        sel = [e.run for e in self.selected_entries()]
        self._fill_table()
        self.select_runs(sel)
        self._update_centres(log=False)
        self._find_profiles(log=False)
        if prefs.get("sasred.table.autofill"):
            self.table_panel.on_fill()
            self.log(msg)
        self.check_all_geometries(log=False)

    def _series_rows(self, table, row):
        """[(row to reduce, table row, series parameters or None)]: a summed row as it is; a time-series row as
        one row per point (its runs summed, the time in the parameters)."""
        if not getattr(row, "series", False):
            return [(row, row, None)]
        import copy
        from .reduction_table import RunChoice
        from .series import series_points
        raws = {e.run: e.raw for e in self.entries if e.raw is not None}
        try:
            mon = (self.instrument_for(raws[row.used[0]]).get("monitor") or {}).get("normaliser", "monitor1")
        except Exception:  # noqa: BLE001
            mon = "monitor1"
        pts = series_points(row, raws, self.norm.currentData(), mon)
        name = f"{table.config} {row.label} ({row.info})"
        self.log(f"{row.label} ({row.info}): time series of {len(pts)} points from {len(row.used)} runs")
        out = []
        for p in pts:
            sub = copy.copy(row)
            sub.runs = [RunChoice(n, True) for n in p.runs]
            sub.series = False
            sub.label = f"{row.label} t={p.time_s:.0f}s"
            out.append((sub, row, p.parameters(name)))
        return out

    def banks_for(self, raw):
        """Detector banks to reduce for a run: [""] (the area detector) for SANS-I; for SANS-LLB the
        choice of the Reduction panel ("" = central, "left", "bottom")."""
        if getattr(raw, "instrument", "sans_i") != "sans_llb":
            return [""]
        b = self.bank_combo.currentData()
        have = list((raw.panels or {}).keys())
        if b == "all":
            return [""] + have
        if b in ("left", "bottom"):
            return [b] if b in have else []
        return [""]

    @staticmethod
    def result_key(run, bank=""):
        return run if not bank else f"{run} {bank}"

    def mask_for(self, raw):
        """The mask for a run and where it comes from: the mask of its instrument configuration
        (coll, SD, λ, beam stop type and position; :func:`catalogue.mask_key`) – the one in the editor
        or a stored one – else the suggested mask of the run (border, beam stop shadow, hot pixels).
        A SANS-LLB wide-angle panel: tube ends and dead tubes."""
        from .catalogue import mask_key, mask_key_label
        from .core import mask as mk
        k = mask_key(raw)
        if getattr(raw, "panel", None) is not None:
            ed = self.mask_editor.mask
            if k == self._mask_key and ed is not None and ed.shape == raw.counts.shape and not self.mask_editor.pristine:
                return ed.copy(), f"mask of {mask_key_label(k)} (detector panel)"
            m = self._stored(self.masks, k)
            if m is not None and m.shape == raw.counts.shape:
                return m.copy(), f"mask of {mask_key_label(k)}"
            return mk.default_mask(raw, self.centre_for(raw))[0], \
                f"{raw.bank} panel: tube ends and dead tubes masked (no mask edited for it)"
        ed = self.mask_editor.mask
        if k == self._mask_key and ed is not None and ed.shape == raw.counts.shape:
            return self.full_mask(raw.counts.shape), f"mask of {mask_key_label(k)} (detector panel)"
        m = self._stored(self.masks, k)
        if m is not None and m.shape == raw.counts.shape:
            return m.copy(), f"mask of {mask_key_label(k)}"
        return mk.suggest_mask(raw.counts, self.centre_for(raw))[0], \
            f"suggested mask (no mask edited for {mask_key_label(k)})"

    @staticmethod
    def _stored(store, k):
        """``store[k]``; for a SANS-LLB configuration also a mask saved before the panel distance was part of
        the key (catalogue.legacy_mask_key)."""
        from .catalogue import legacy_mask_key
        v = store.get(k)
        if v is None:
            old = legacy_mask_key(k)
            if old != k:
                v = store.get(old)
        return v

    def region_for(self, raw):
        """Transmission region (bool image: pixels of the beam transmitted through a semi-transparent beam
        stop) of the instrument configuration of ``raw``, or None when none is drawn."""
        from .catalogue import mask_key
        k = mask_key(raw)
        ed = self.mask_editor
        if k == self._mask_key and ed.mask is not None and ed.mask.shape == raw.counts.shape:
            r = ed.region
        else:
            r = self._stored(self.tregions, k)
        if r is None or r.shape != raw.counts.shape or not r.any():
            return None
        return r.copy()

    def _sync_mask_config(self, raw):
        """Switch the editor to the mask of the configuration of ``raw``; the edited mask of the
        previous configuration is kept."""
        from .catalogue import mask_key, mask_key_label
        k = mask_key(raw)
        ed = self.mask_editor
        if k != self._mask_key:
            if self._mask_key is not None and ed.mask is not None and not ed.pristine:
                self.masks[self._mask_key] = ed.mask.copy()
            if self._mask_key is not None:
                self._commit_region()
            if self._mask_key is None and ed.mask is not None and not ed.pristine and k not in self.masks:
                self.masks[k] = ed.mask.copy()               # mask restored from an older session
            m = self._stored(self.masks, k)
            r = self._stored(self.tregions, k)
            ed._undo.clear()
            ed._redo.clear()
            ed.region = None
            if m is not None and m.shape == raw.counts.shape:
                ed.mask = m.copy()
                ed.pristine = False
            else:
                ed.mask = None
                ed.ensure(raw.counts.shape)
            ed.region = r.copy() if r is not None and r.shape == raw.counts.shape else None
            ed._update_stats()
            self._mask_key = k
        n = len(set(self.masks) | ({k} if not ed.pristine else set()))
        coll, SD, lam, bst, bx, by = k
        self.mask_cfg.setText(f"mask: coll {coll:g}, SD {SD:g} m, {lam:g} Å, BS {bst or '?'} ({bx}, {by})"
                              + (" · new" if ed.pristine else "") + (f" · {n} edited" if n else "")
                              + (" · transmission region" if ed.region is not None else ""))
        self.mask_cfg.setToolTip(f"mask for {mask_key_label(k)}" + (" (not edited yet)" if ed.pristine else "")
                                 + f"\nmasks edited for {n} configuration{'s' if n != 1 else ''}")

    def _commit_mask(self):
        if self._mask_key is not None and self.mask_editor.mask is not None and not self.mask_editor.pristine:
            self.masks[self._mask_key] = self.mask_editor.mask.copy()
        self._commit_region()

    def _commit_region(self):
        """The transmission region in the editor into the store of its configuration (none: removed)."""
        if self._mask_key is None:
            return
        r = self.mask_editor.region
        if r is not None and r.any():
            self.tregions[self._mask_key] = r.copy()
        else:
            self.tregions.pop(self._mask_key, None)

    def save_masks(self, path):
        """All configuration masks into one HDF5 file (one group per configuration)."""
        import h5py
        from .catalogue import mask_key_label
        self._commit_mask()
        with h5py.File(path, "w") as f:
            f.attrs["description"] = ("pySASred masks per instrument configuration (coll, SD, λ, beam stop); "
                                      "transmission_region: pixels of the beam through a semi-transparent beam stop")
            keys = sorted(set(self.masks) | set(self.tregions), key=str)
            for i, k in enumerate(keys):
                g = f.create_group(f"mask{i:03d}")
                if k in self.masks:
                    g.create_dataset("pixel_mask", data=np.asarray(self.masks[k], np.uint32), compression="gzip")
                if k in self.tregions:
                    g.create_dataset("transmission_region", data=np.asarray(self.tregions[k], np.uint8),
                                     compression="gzip")
                g.attrs["key"] = json.dumps(list(k))
                g.attrs["label"] = mask_key_label(k)
        return path

    def load_masks(self, path, replace=False):
        import h5py
        new, regions = {}, {}
        with h5py.File(path, "r") as f:
            for g in f.values():
                k = json.loads(g.attrs["key"])
                k[0:3] = [float(x) for x in k[0:3]]
                if "pixel_mask" in g:
                    new[tuple(k)] = np.asarray(g["pixel_mask"][()], np.uint32)
                if "transmission_region" in g:
                    regions[tuple(k)] = np.asarray(g["transmission_region"][()]).astype(bool)
        if replace:
            self.masks = {}
            self.tregions = {}
        self.masks.update(new)
        self.tregions.update(regions)
        if self._mask_key in new:
            self.mask_editor.set_mask(new[self._mask_key], message="mask of the shown configuration loaded")
        if self._mask_key in regions:
            self.mask_editor.set_region(regions[self._mask_key], record=False)
        return len(set(new) | set(regions))

    def on_save_masks(self):
        self._commit_mask()
        if not self.masks and not self.tregions:
            QtWidgets.QMessageBox.information(self, "Masks", "No edited masks yet.")
            return
        p, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save masks of all configurations",
                                                     os.path.join(self.folder or os.getcwd(), "masks.h5"),
                                                     "HDF5 (*.h5)")
        if p:
            self.save_masks(p)
            self.log(f"masks of {len(self.masks)} configurations saved to {p}")

    def on_load_masks(self):
        p, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Load masks of configurations", self.folder or "",
                                                     "HDF5 (*.h5)")
        if p:
            n = self.load_masks(p)
            self.log(f"masks of {n} configurations loaded from {p}")
            self._draw_detector()

    def reduce_table_rows(self, table, rows):
        """Reduce sample rows of the reduction table: 2D per pixel, then the average with the q bins,
        sector and normalisation of the Reduction panel. Results appear in I(q) and as 2D image."""
        from .absolute import reduce_row_2d, average
        from .io.raw import read_raw
        paths = {e.run: e.path for e in self.entries}

        cache = {}

        def read_bank(bank):
            def read(n):
                if n not in paths:
                    raise KeyError(f"run {n} not in the folder")
                if n not in cache:
                    from . import names as nm
                    cache[n] = nm.apply(read_raw(paths[n]), self.name_corrections)
                r = cache[n]
                if bank:
                    from .io.sans_llb import bank_run
                    return bank_run(r, bank)
                return r
            return read
        sector = (self.sector_c.value(), self.sector_hw.value()) if self.sector.isChecked() else None
        done = None
        self._last_keys = []
        missing_seen = self.check_all_geometries(log=False)    # every geometry, then what the reduction finds
        n_before = self.missing_view.topLevelItemCount()
        jobs = []
        for row in rows:
            banks = [""]
            if row.used and row.used[0] in paths:
                try:
                    banks = self.banks_for(read_bank("")(row.used[0])) or [""]
                except Exception:  # noqa: BLE001
                    pass
            for sub, orig, spar in self._series_rows(table, row):
                jobs += [(sub, b, orig, spar) for b in banks]
        QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.CursorShape.WaitCursor)
        try:
            for row, bank, orig, spar in jobs:
                read = read_bank(bank)
                tag = f"{table.config.split(',')[0]} {row.label}" + (f" {bank}" if bank else "")
                try:
                    raw0 = read(row.used[0]) if row.used else None
                    if raw0 is None:
                        raise ValueError("no runs switched on")
                    mask, how = self.mask_for(raw0)
                    eff = None
                    if table.scale_method() != "water" and table.efficiency:
                        eff = self.efficiency_for(table, bank)
                    prof = self.profile_for(raw0)
                    r2 = reduce_row_2d(table, row, read, self.centre_for(raw0), mask=mask,
                                       instrument=self.instrument_for(raw0),
                                       normalise_by=self.norm.currentData(), k0=table.water_g, efficiency=eff,
                                       beam_profile=prof)
                    if prof is None or prof.estimated:
                        txt = ("no direct-beam measurement of this geometry: no resolution (Qdev) – reduced without it"
                               if prof is None else
                               "no direct-beam measurement of this geometry: beam divergence estimated from the "
                               f"collimation ({prof.notes[0] if prof.notes else ''})")
                        self._add_missing(table.config, dict(role="direct_beam", index=None, label="Direct beam",
                                                             field="resolution", text=txt, severity="check"),
                                          missing_seen)
                    r2.notes.insert(0, how)
                    r1 = average(r2, self._edges(raw0, r2.mask), sector_deg=sector)
                except Exception as exc:  # noqa: BLE001
                    self.log(f"{tag} ({row.info}): FAILED – {type(exc).__name__}: {exc}")
                    for x in table.missing_inputs([orig]):
                        self._add_missing(table.config, x.__dict__, missing_seen)
                    self._add_missing(table.config, dict(role="sample", index=table.samples.index(orig)
                                                         if orig in table.samples else None,
                                                         label=f"{row.label} ({row.info})", field="reduction",
                                                         text=f"not reduced: {exc}", severity="error"), missing_seen)
                    continue
                for x in r2.parameters.get("missing") or []:
                    self._add_missing(table.config, x, missing_seen)
                if spar:                                   # a point of a time series: its time travels along
                    r1.parameters.update(spar)
                    r2.parameters.update(spar)
                    r1.notes.insert(0, f"time series {spar['series']}: point {spar['series_index'] + 1}, "
                                       f"t = {spar['time_s']:.1f} s after {spar['series_t0']} (runs {spar['series_runs']})")
                self.results[tag] = r1
                self._last_keys.append(tag)
                done = tag
                self.log(f"{tag} ({row.info}): runs {row.runs_text()}, {len(r1.q)} bins "
                         f"({getattr(self, '_last_bins', '')}), {r1.intensity.unit}")
                for n in dict.fromkeys(r1.notes):
                    self.log(f"    {n}")
                QtWidgets.QApplication.processEvents()
        finally:
            QtWidgets.QApplication.restoreOverrideCursor()
        n = self._finish_missing()
        mine = [self.missing_view.topLevelItem(k) for k in range(self.missing_view.topLevelItemCount())
                if self.missing_view.topLevelItem(k).data(0, QtCore.Qt.ItemDataRole.UserRole)[0] == table.config
                or k >= n_before]
        if mine:                                       # the log lists what concerns the geometry reduced
            self.log(f"{table.config}: {len(mine)} inputs missing, to check or left out – see the panel "
                     "'Missing input'; the cells are coloured in the reduction table")
            for it in mine:
                self.log(f"    {'not used' if it.data(1, QtCore.Qt.ItemDataRole.UserRole) == 'info' else 'missing'}: "
                         f"{it.text(1)} · {it.text(2)}: {it.text(3)}")
        if n:
            self.statusBar().showMessage(f"{n} inputs missing or to check: panel 'Missing input'", 15000)
        self._refresh_results(select=done)
        if done is not None:
            self.docks["iq"].raise_()
        return done

    # ------------------------------------------------------------------ complete check of all geometries
    def check_all_geometries(self, log: bool = True) -> set:
        """Fill the panel 'Missing input' for every geometry of the reduction table without reducing:
        the inputs of each table (runs, transmissions, thickness, absolute scale), the resolution (direct
        beam or estimate), dλ/λ, the beam centre, the efficiency map chosen, and runs of one reduction at
        different selector tilts. Runs automatically when the folder is read and when a table changes;
        Reduction → Check all geometries. Returns the set of items shown (keys)."""
        self.missing_view.clear()
        seen: set = set()
        raws = {e.run: e.raw for e in self.entries if e.raw is not None}
        counts = {}
        for page in self.table_panel.pages():
            t = page.table
            n0 = len(seen)
            try:
                for x in t.missing_inputs():
                    self._add_missing(t.config, x.__dict__, seen)
                for x in self._geometry_checks(t, page, raws):
                    self._add_missing(t.config, x, seen)
            except Exception as exc:  # noqa: BLE001 -- a broken table must not stop the check
                self._add_missing(t.config, dict(role="sample", index=None, label="table", field="check",
                                                 text=f"check failed: {type(exc).__name__}: {exc}",
                                                 severity="error"), seen)
            counts[t.config] = self._missing_count(t.config)
        self._mark_tabs(counts)
        n = self._finish_missing()
        if log:
            bad = {c: k for c, k in counts.items() if k[0]}
            self.log(f"check of {len(counts)} geometr{'y' if len(counts) == 1 else 'ies'}: "
                     + ("nothing missing" if not bad else
                        "; ".join(f"{c}: {k[0]} missing / to check" for c, k in bad.items()))
                     + (" – see the panel 'Missing input'" if n else ""))
        return seen

    def _geometry_checks(self, t, page, raws) -> list:
        """Checks of one geometry that need the runs (not only the table)."""
        from .absolute import _tilt_checks
        from .catalogue import config_key
        out = []
        used = [n for r in t.samples for n in r.used]
        raw = next((raws[n] for n in used if n in raws), None) or page.ref_raw
        if raw is None:
            return out

        def item(role, label, field, text, severity="check"):
            out.append(dict(role=role, index=None, label=label, field=field, text=text, severity=severity))
        prof = self.profile_for(raw)
        if prof is None:
            item("direct_beam", "Direct beam", "resolution",
                 "no direct beam and no collimation data: no resolution (Qdev)")
        elif prof.estimated:
            item("direct_beam", "Direct beam", "resolution",
                 "no direct-beam measurement of this geometry: beam divergence " + prof.origin())
        if raw.dlam_over_lam_fwhm is None:
            from .core.resolution import default_dlam
            d = default_dlam()
            item("direct_beam", "Direct beam", "dλ/λ",
                 f"dλ/λ not in the raw file: {d:g} (FWHM) of the preferences used" if d else
                 "dλ/λ not in the raw file and no default: no resolution (Qdev)", "info" if d else "check")
        c = self.centres.get(config_key(raw))
        if c is None:
            item("sample", "Beam centre", "beam centre", "no beam centre for this geometry: the x / y fields are used")
        elif c.source.startswith("borrowed"):
            item("sample", "Beam centre", "beam centre", f"no direct beam: centre {c.source}")
        spec = (t.efficiency or "").strip()
        if spec and t.scale_method() != "water":
            from .efficiency_gui import GEOMETRY_PREFIX
            if spec.startswith(GEOMETRY_PREFIX):
                src = self._table_of(spec[len(GEOMETRY_PREFIX):])
                if src is None or not src.corrections["water"].used:
                    item("direct_beam", "Direct beam", "efficiency",
                         f"efficiency map from {spec[len(GEOMETRY_PREFIX):]}: that geometry has no water runs",
                         "missing")
            else:
                path = spec.partition("::")[0].strip()
                if not os.path.isabs(path) and self.folder:
                    path = os.path.join(self.folder, path)
                if not os.path.isfile(path):
                    item("direct_beam", "Direct beam", "efficiency", f"efficiency map file {path} not found", "missing")
        tilts = {n: float(r.selector_tilt_deg) for n, r in raws.items()
                 if getattr(r, "selector_tilt_deg", None) is not None}
        if tilts:
            inst = self.instrument_for(raw)
            for row in t.samples:
                if row.used:
                    out += _tilt_checks(t, row, tilts, self.norm.currentData(), [], inst)
        return out

    def _missing_count(self, config) -> tuple:
        n = n_info = 0
        for k in range(self.missing_view.topLevelItemCount()):
            it = self.missing_view.topLevelItem(k)
            if it.data(0, QtCore.Qt.ItemDataRole.UserRole)[0] != config:
                continue
            if it.data(1, QtCore.Qt.ItemDataRole.UserRole) == "info":
                n_info += 1
            else:
                n += 1
        return n, n_info

    def _mark_tabs(self, counts):
        """Geometry tabs with something missing: dark red text and the count in the tool tip."""
        tabs = self.table_panel.tabs
        for i in range(tabs.count()):
            page = tabs.widget(i)
            cfg = getattr(getattr(page, "table", None), "config", None)
            n, n_info = counts.get(cfg, (0, 0))
            tabs.tabBar().setTabTextColor(i, QtGui.QColor("#b00020") if n else QtGui.QColor())
            tip = (tabs.tabToolTip(i) or "").split("\n⚠")[0]
            if n or n_info:
                tip += f"\n⚠ {n} inputs missing or to check, {n_info} left out (panel 'Missing input')"
            tabs.setTabToolTip(i, tip)

    def _finish_missing(self) -> int:
        n_all = self.missing_view.topLevelItemCount()
        n = sum(self.missing_view.topLevelItem(k).data(1, QtCore.Qt.ItemDataRole.UserRole) != "info"
                for k in range(n_all))
        self.docks["missing"].setWindowTitle(f"Missing input ({n})" if n else "Missing input")
        return n

    def _schedule_check(self):
        if not hasattr(self, "_check_timer"):
            self._check_timer = QtCore.QTimer(self)
            self._check_timer.setSingleShot(True)
            self._check_timer.setInterval(400)
            self._check_timer.timeout.connect(lambda: self.check_all_geometries(log=False))
        self._check_timer.start()

    _MISSING_COLOURS = {"missing": QtGui.QColor(255, 196, 196), "check": QtGui.QColor(255, 226, 180),
                        "error": QtGui.QColor(230, 120, 120), "info": QtGui.QColor(232, 232, 232)}

    def _add_missing(self, config, x: dict, seen: set):
        """One line of the panel 'Missing input' (a correction item only once per geometry)."""
        key = (config, x.get("role"), x.get("index"), x.get("field"))
        if key in seen:
            return
        seen.add(key)
        short = config.replace(", λ", " · λ").replace(", coll", " · coll")
        it = QtWidgets.QTreeWidgetItem([short, x.get("label", ""), x.get("field", ""), x.get("text", "")])
        col = self._MISSING_COLOURS.get(x.get("severity", "missing"))
        for j in range(4):
            it.setBackground(j, QtGui.QBrush(col))
            it.setToolTip(j, f"{config}\n{x.get('text', '')}")
        it.setData(0, QtCore.Qt.ItemDataRole.UserRole, (config, x.get("role"), x.get("index"), x.get("field")))
        it.setData(1, QtCore.Qt.ItemDataRole.UserRole, x.get("severity", "missing"))
        self.missing_view.addTopLevelItem(it)
        for j in range(3):
            self.missing_view.resizeColumnToContents(j)

    def _on_missing_clicked(self, it, _col=0):
        config, role, index, field = it.data(0, QtCore.Qt.ItemDataRole.UserRole)
        self.show_table()
        if field != "reduction":
            self.table_panel.show_missing(config, role, index, field)
        elif role == "sample" and index is not None:
            self.table_panel.show_missing(config, role, index, "runs")

    def show_table(self):
        """Show the reduction table panel (a tab next to the detector by default)."""
        d = self.docks["table"]
        if d.isHidden():
            self.set_panel_visible("table", True)
        d.raise_()

    def log(self, text):
        self.log_view.appendPlainText(text)

    # ------------------------------------------------------------------ runs
    def on_browse(self):
        d = QtWidgets.QFileDialog.getExistingDirectory(self, "Raw-data folder", self.folder or os.getcwd())
        if d:
            self.set_folder(d)

    def set_folder(self, folder):
        from .catalogue import scan_folder
        folder = str(folder).strip()
        if not folder or not os.path.isdir(folder):
            self.statusBar().showMessage(f"not a folder: {folder}", 8000)
            return
        self.folder = folder
        self.folder_edit.setText(folder)

        def progress(i, n):
            self.statusBar().showMessage(f"reading headers {i}/{n}")
            QtWidgets.QApplication.processEvents()

        QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.CursorShape.WaitCursor)
        try:
            self.entries = scan_folder(folder, progress)
        finally:
            QtWidgets.QApplication.restoreOverrideCursor()
        from . import names as nm
        self.name_corrections = nm.load(folder)
        self._apply_corrections()
        if self.name_corrections:
            self.log(f"sample names / roles corrected and runs excluded by hand ({nm.store_path(folder)}):")
            for ln in self.name_corrections.describe().splitlines():
                self.log("    " + ln)
        self._fill_table()
        self._update_centres(log=True)
        self._find_profiles()
        self._test_monitors()
        if prefs.get("sasred.table.autofill"):
            self.table_panel.on_fill()
        self.check_all_geometries()
        if getattr(self, "_unread", None):
            self._restore_session_tables()
        bad = sum(e.raw is None for e in self.entries)
        self.statusBar().showMessage(f"{len(self.entries)} runs in {folder}" + (f", {bad} unreadable" if bad else ""),
                                     10000)

    def _test_monitors(self):
        """Which monitors are in front of / behind the selector (ratio to the proton beam at different
        λ or tilts, core.monitors); the normaliser's position decides whether a change of the selector
        tilt cancels in the monitor normalisation."""
        from .core.monitors import selector_test
        from .reduce import load_instrument
        raws = [e.raw for e in self.entries if e.raw is not None]
        self.monitor_tests = {}
        if not raws:
            return
        try:
            self.monitor_tests = selector_test(raws)
        except Exception as exc:  # noqa: BLE001
            self.log(f"monitor test failed: {exc}")
            return
        mon = load_instrument(raws[0]).get("monitor") or {}
        name = mon.get("normaliser", "monitor1")
        t = self.monitor_tests.get(name)
        if t is None:
            return
        given = mon.get("normaliser_after_selector")
        if given is not None:
            self.log(f"normalising monitor {name}: {'behind' if given else 'in front of'} the selector "
                     "(instrument file)" + (f"; data: {t.position}" if t.after_selector is not None
                                            and t.after_selector != given else ""))
        else:
            self.log("normalising monitor " + t.describe())
        for m, x in sorted(self.monitor_tests.items()):
            if m != name and x.after_selector is not None:
                self.log("    monitor " + x.describe())

    def instrument_for(self, raw) -> dict:
        """Instrument file of a run, with the selector position of the normalising monitor from the data
        when the file does not give it."""
        from .reduce import load_instrument
        inst = load_instrument(raw)
        mon = inst.setdefault("monitor", {})
        if mon.get("normaliser_after_selector") is None:
            t = (getattr(self, "monitor_tests", None) or {}).get(mon.get("normaliser", "monitor1"))
            if t is not None and t.after_selector is not None:
                mon["normaliser_after_selector"] = t.after_selector
        return inst

    @staticmethod
    def _user_centre(c) -> bool:
        """A centre set by hand or fitted to a chosen run (not the automatic direct-beam one)."""
        return c.source != "direct beam" and not c.source.startswith("borrowed")

    def _update_centres(self, log=True):
        """Beam centres of the geometries from their direct-beam runs; centres set by hand or fitted to a
        run before (also from the last session) are kept – a folder read again or names corrected do
        not forget them."""
        from .catalogue import beam_centres
        kept = {k: c for k, c in (self.centres or {}).items() if self._user_centre(c)}
        self.centres = beam_centres([e.raw for e in self.entries])
        self.centres.update(kept)
        if log:
            for c in sorted(self.centres.values(), key=lambda c: c.key):
                self.log("beam centre " + c.describe() + ("  (kept: set before)" if c.key in kept else ""))
        raw = self.displayed_raw() if hasattr(self, "displayed_raw") else None
        if raw is not None:
            from .catalogue import config_key
            c = self.centres.get(config_key(raw))
            if c is not None:
                self._centre_sync = True
                try:
                    self.xc.setValue(c.x)
                    self.yc.setValue(c.y)
                finally:
                    self._centre_sync = False
                self.bc_label.setText(c.describe())

    def _fill_table(self):
        from .catalogue import COLUMNS as COLUMNS_RT
        t = self.table
        t.setSortingEnabled(False)
        t.setRowCount(len(self.entries))
        for i, e in enumerate(self.entries):
            for j, val in enumerate(e.row(self.unit)):
                it = QtWidgets.QTableWidgetItem()
                if j == 0:
                    it.setData(QtCore.Qt.ItemDataRole.DisplayRole, int(val))
                    it.setData(QtCore.Qt.ItemDataRole.UserRole, i)
                else:
                    it.setText(str(val))
                if e.raw is None or getattr(e, "skip", ""):
                    it.setForeground(QtGui.QBrush(QtGui.QColor("gray")))
                    f = it.font()
                    f.setStrikeOut(j > 0 and j < len(COLUMNS_RT) - 1)
                    it.setFont(f)
                    it.setToolTip(f"skipped: {e.skip or 'unreadable'} – not used in any reduction")
                elif e.raw.aborted:
                    it.setForeground(QtGui.QBrush(QtGui.QColor("firebrick")))
                if j == 1 and e.raw is not None and (e.raw.sample_name != (e.raw.sample_name_in_file or e.raw.sample_name)
                                                     or getattr(e.raw, "role", "")):
                    f = it.font()
                    f.setItalic(True)
                    it.setFont(f)
                    it.setForeground(QtGui.QBrush(QtGui.QColor("#1f5fa8")))
                    it.setToolTip(f"name in the file: {e.raw.sample_name_in_file}"
                                  + (f"\nrole set by hand: {e.raw.role}" if e.raw.role else ""))
                t.setItem(i, j, it)
        t.setHorizontalHeaderItem(_lam_column(), QtWidgets.QTableWidgetItem(f"λ [{self.units.length}]"))
        t.setSortingEnabled(True)
        t.sortItems(0, QtCore.Qt.SortOrder.AscendingOrder)
        t.resizeColumnsToContents()
        self._apply_filter(self.filter_edit.text())

    def _apply_filter(self, text):
        text = text.strip().lower()
        for i in range(self.table.rowCount()):
            hit = not text or any(text in (self.table.item(i, j).text().lower())
                                  for j in (0, 1) if self.table.item(i, j) is not None)
            self.table.setRowHidden(i, not hit)

    def selected_entries(self):
        rows = sorted({ix.row() for ix in self.table.selectionModel().selectedRows()})
        out = []
        for r in rows:
            it = self.table.item(r, 0)
            if it is not None:
                out.append(self.entries[it.data(QtCore.Qt.ItemDataRole.UserRole)])
        return out

    def select_runs(self, runs):
        """Select the rows of the given run numbers (scripting and tests)."""
        runs = set(int(r) for r in runs)
        self.table.clearSelection()
        mode = self.table.selectionMode()
        self.table.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.MultiSelection)
        for r in range(self.table.rowCount()):
            if int(self.table.item(r, 0).data(QtCore.Qt.ItemDataRole.DisplayRole)) in runs:
                self.table.selectRow(r)
        self.table.setSelectionMode(mode)

    def showEvent(self, event):
        super().showEvent(event)
        if not self._middle_tabs_done:
            self._middle_tabs_done = True
            QtCore.QTimer.singleShot(0, self._arrange_middle_tabs)

    def _arrange_middle_tabs(self):
        """The detector and the reduction table share the wide middle column as tabs (each gets the full
        width there); I(q), error budget and log stay visible next to them. Done once the window is
        shown: forming a second tab group before that leaves a stray tab bar in Qt."""
        det, tab = self.docks["detector"], self.docks["table"]
        if tab in self.tabifiedDockWidgets(det) or det.isFloating() or tab.isFloating():
            return                                         # arranged already (last session)
        if self.dockWidgetArea(det) != self.dockWidgetArea(tab):
            return
        self.tabifyDockWidget(det, tab)
        det.raise_()
        self.hide_stray_tab_bars()
        self._default_column_widths()

    def _default_column_widths(self):
        """Start widths of the three columns, in proportion to the window (a laptop of 1600 × 1000 gives
        about 340 | 790 | 470 px): enough for the run list and the Reduction panel without scrolling
        sideways, the rest for the detector image."""
        W = max(self.width(), 900)
        left = int(min(max(0.21 * W, 300), 420))
        right = int(min(max(0.29 * W, 340), 620))
        try:
            self.resizeDocks([self.docks["runs"], self.docks["detector"], self.docks["iq"]],
                             [left, max(W - left - right, 300), right], QtCore.Qt.Orientation.Horizontal)
            H = max(self.height(), 600)
            self.resizeDocks([self.docks["runs"], self.docks["reduction"]], [int(0.42 * H), int(0.58 * H)],
                             QtCore.Qt.Orientation.Vertical)
        except Exception:  # noqa: BLE001 -- a layout nicety
            pass

    def reset_layout(self):
        """View → Reset layout: the start arrangement, with the detector and the reduction table as tabs."""
        super().reset_layout()
        self._arrange_middle_tabs()

    def _on_selection(self):
        if self.selected_entries() and not self.docks["detector"].isFloating():
            self.docks["detector"].raise_()             # a run chosen: its detector image (tab next to the table)
        self._show_centre_of(self.displayed_raw())
        self._draw_detector()
        self._schedule_preview()          # also redraws I(q) (selected curves)

    # ------------------------------------------------------------------ beam centre per geometry
    def _find_profiles(self, log=True):
        """Direct-beam profile (2D Gaussian fit) of every geometry: beam divergence for the resolution."""
        from .catalogue import _key_text, config_key
        from .core.beamprofile import beam_profiles
        try:
            self.profiles = beam_profiles([e.raw for e in self.entries if e.raw is not None])
        except Exception as exc:  # noqa: BLE001 -- a bad run must not stop opening the folder
            self.profiles = {}
            self.log(f"beam profiles: {type(exc).__name__}: {exc}")
        if log:
            for p in sorted(self.profiles.values(), key=lambda p: p.key):
                self.log("beam profile " + p.describe())
        keys = sorted({config_key(e.raw) for e in self.entries if e.raw is not None and not e.skip})
        missing = [k for k in keys if k not in self.profiles]
        from .core.beamprofile import calibrated_profile, collimation_profile
        measured = dict(self.profiles)
        first = {}
        for e in self.entries:
            if e.raw is not None and not e.skip:
                first.setdefault(config_key(e.raw), e.raw)
        for k in missing:
            try:
                inst = self.instrument_for(first[k])
                est = calibrated_profile(first[k], measured, inst) or collimation_profile(first[k], inst)
            except Exception:  # noqa: BLE001
                est = None
            if est is not None:
                self.profiles[k] = est
                if log:
                    self.log(f"WARNING {_key_text(k)}: no direct-beam measurement – beam divergence estimated from "
                             f"the collimation{' (calibrated on direct beams)' if 'calibrated' in est.source else ''}: "
                             + est.describe())
            elif log:
                self.log(f"WARNING {_key_text(k)}: no direct-beam measurement and no collimation data – no "
                         "resolution (Qdev) for this geometry")

    def profile_for(self, raw):
        """Direct-beam profile of the geometry of ``raw`` (None: none measured)."""
        from .catalogue import config_key
        return self.profiles.get(config_key(raw)) if raw is not None else None

    def centre_for(self, raw) -> tuple:
        """Beam centre of the geometry of ``raw`` (direct beam of that geometry, a borrowed one,
        or set by hand); the spin boxes when there is none."""
        from .catalogue import config_key
        c = self.centres.get(config_key(raw)) if raw is not None else None
        if c is None:
            return (self.xc.value(), self.yc.value())
        return (round(float(c.x), self.xc.decimals()), round(float(c.y), self.yc.decimals()))   # as shown

    def _show_centre_of(self, raw):
        from .catalogue import config_key
        if raw is None:
            return
        c = self.centres.get(config_key(raw))
        if c is None:
            self.bc_label.setText(f"no beam centre for SD {raw.SD_m:.1f} m, λ {raw.wavelength_A:.2f} Å yet: "
                                  "determine it from a direct-beam run of this geometry or enter it")
            return
        self._centre_sync = True
        try:
            self.xc.setValue(c.x)
            self.yc.setValue(c.y)
        finally:
            self._centre_sync = False
        self.bc_label.setText(c.describe())

    def _on_centre_edit(self):
        """A centre typed in the spin boxes belongs to the geometry of the displayed run."""
        from .catalogue import GeometryCentre, config_key
        raw = self.displayed_raw()
        if self._centre_sync or raw is None:
            return
        k = config_key(raw)
        self.centres[k] = GeometryCentre(k, self.xc.value(), self.yc.value(), 0.0, 0.0, [], "by hand")
        self.bc_label.setText(self.centres[k].describe())

    # ------------------------------------------------------------------ units
    @property
    def units(self):
        return get_units(self.unit)

    def set_units(self, key):
        """Switch the display/output units ("nm" or "A"); q fields, table and plots follow."""
        new = get_units(key)
        old = self.units
        i = self.unit_combo.findData(key)
        if self.unit_combo.currentIndex() != i:
            self.unit_combo.blockSignals(True)
            self.unit_combo.setCurrentIndex(i)
            self.unit_combo.blockSignals(False)
        if key == self.unit:
            return
        for w in (self.qmin, self.qmax):
            w.blockSignals(True)
            w.setValue(new.q_out(old.q_in(w.value())))
            w.setSuffix(" " + new.q)
            w.blockSignals(False)
        self.unit = key
        self._draw_detector()
        self._fill_table()
        self._draw_iq()
        self._draw_budget()

    # ------------------------------------------------------------------ live preview
    def _schedule_preview(self):
        self._preview_timer.start()

    def on_send_to_merge(self):
        """Hand the reduced results to pyTOOLbox *Merge geometries* without writing files."""
        if not self.results:
            QtWidgets.QMessageBox.information(self, "Merge geometries", "Nothing reduced yet.")
            return
        from ..toolbox.merge_gui import send_reduced
        send_reduced(self.results, "pySASred")
        self.log(f"{len(self.results)} reduced curves sent to pyTOOLbox → Merge geometries")

    _PANEL_PREFS = ("sasred.qbins.n", "sasred.qbins.spacing", "sasred.normalise_by",
                    "sasred.mask.beamstop_shape", "sasred.mask.beamstop_radius_px")

    def apply_preferences(self):
        """Set the panel controls whose preference was changed (the others keep what was set in
        the panel or restored from the last session)."""
        try:
            changed = [k for k in self._PANEL_PREFS if prefs.get(k) != self._pref_seen.get(k)]
        except RuntimeError:          # window already deleted
            return
        for k in changed:
            v = prefs.get(k)
            if k == "sasred.qbins.n":
                self.nbins.setValue(int(v))
            elif k == "sasred.mask.beamstop_radius_px":
                self.bs_radius.setValue(float(v))
            else:
                combo = {"sasred.qbins.spacing": self.spacing, "sasred.normalise_by": self.norm,
                         "sasred.mask.beamstop_shape": self.bs_shape}[k]
                i = combo.findData(v)
                if i >= 0:
                    combo.setCurrentIndex(i)
            self._pref_seen[k] = v
        if changed:
            self.log("preferences applied to the Reduction panel: " + ", ".join(k.split(".", 1)[1] for k in changed))

    def update_preview(self):
        """Radial / sector average of the first selected run with the current settings, shown in
        I(q) without being kept (uses the header already read, no checksum)."""
        from .reduce import reduce_run_1d
        self.preview = None
        self.preview_banks = []
        sel = [e for e in self.selected_entries() if e.raw is not None] if self.live.isChecked() else []
        if sel:
            raw = self.displayed_raw()
            sector = (self.sector_c.value(), self.sector_hw.value()) if self.sector.isChecked() else None
            try:
                self._sync_mask_config(raw)
                mask = self._mask(raw.counts.shape)
                self.preview = reduce_run_1d(raw, self.centre_for(raw), self._edges(raw, mask),
                                             mask=mask, normalise_by=self.norm.currentData(), sector_deg=sector,
                                             beam_profile=self.profile_for(raw))
                if self.preview is not None and getattr(raw, "bank", ""):
                    self.preview.parameters["bank"] = raw.bank
            except Exception as exc:  # noqa: BLE001 -- e.g. beam centre outside, empty q range
                self.statusBar().showMessage(f"live I(q) of run {raw.run}: {exc}", 8000)
            # view "SANS-LLB: all detectors": the other banks too, each with its own mask and q range
            full = sel[0].raw
            if (self.det_view.currentData() == "llb_all" and getattr(full, "instrument", "") == "sans_llb"
                    and getattr(full, "panels", None)):
                from .io.sans_llb import bank_run
                shown = getattr(raw, "bank", "") or ""
                for b in ["", *full.panels]:
                    if b == shown:
                        continue
                    r = bank_run(full, b) if b else full
                    try:
                        m = self.mask_for(r)[0]
                        p = reduce_run_1d(r, self.centre_for(r), self._edges(r, m), mask=m,
                                          normalise_by=self.norm.currentData(), sector_deg=sector,
                                          beam_profile=self.profile_for(r))
                        p.parameters["bank"] = b or "central"
                        self.preview_banks.append(p)
                    except Exception as exc:  # noqa: BLE001
                        self.statusBar().showMessage(f"live I(q) of run {full.run} {b or 'central'}: {exc}", 8000)
        self._draw_iq()

    # ------------------------------------------------------------------ beam centre
    def on_beam_centre(self):
        from .core.beamcentre import masked_centre_of_gravity
        sel = [e for e in self.selected_entries() if e.raw is not None]
        if not sel:
            QtWidgets.QMessageBox.information(self, "Beam centre", "Select the direct-beam run in the run table.")
            return
        e = sel[0]
        v = prefs.values("sasred")
        try:
            bc = masked_centre_of_gravity(e.raw.counts, radius_px=v["beamcentre.radius_px"],
                                          tol_px=v["beamcentre.tol_px"], max_iter=v["beamcentre.max_iter"])
        except Exception as exc:  # noqa: BLE001
            QtWidgets.QMessageBox.critical(self, "Beam centre", f"run {e.run}: {exc}")
            return
        from .catalogue import GeometryCentre, config_key
        self.beam_centre = bc
        k = config_key(e.raw)
        self.centres[k] = GeometryCentre(k, bc.x, bc.y, bc.sigma_x, bc.sigma_y, [e.run], f"run {e.run}")
        self._centre_sync = True
        try:
            self.xc.setValue(bc.x)
            self.yc.setValue(bc.y)
        finally:
            self._centre_sync = False
        self.bc_label.setText(f"run {e.run}: x = {bc.x:.3f} ± {bc.sigma_x:.3f}, y = {bc.y:.3f} ± {bc.sigma_y:.3f}"
                              f", {bc.iterations} iterations"
                              + (f", {100 * bc.edge_fraction:.0f} % of the disc off the detector"
                                 if bc.edge_fraction > 0 else "")
                              + f"; kept for coll {k[2]:g} m, SD {k[0]:g} m, λ {k[1]:g} Å")
        self.log(f"beam centre from run {e.run}: x = {bc.x:.4f} ± {bc.sigma_x:.4f}, y = {bc.y:.4f} ± {bc.sigma_y:.4f}")

    # ------------------------------------------------------------------ reduction
    def beamstop_guess(self, shape):
        """First-guess beamstop of the Reduction panel (bool image), or None for size 0."""
        from .core.mask import beamstop_guess
        r = self.bs_radius.value()
        if r <= 0:
            return None
        return beamstop_guess(shape, (self.xc.value(), self.yc.value()), r, self.bs_shape.currentData())

    def full_mask(self, shape):
        """The mask used everywhere: the edited mask (border, beamstop, hot, user pixels). The
        first-guess beamstop is part of it (see :meth:`beamstop_guess`)."""
        return self.mask_editor.ensure(shape).copy()

    def _mask(self, shape):
        return self.full_mask(shape)

    def displayed_raw(self):
        """The run shown in the detector panel (first selected), or None."""
        sel = [e for e in self.selected_entries() if e.raw is not None]
        if not sel:
            return None
        raw = sel[0].raw
        b = self.det_bank.currentData() if hasattr(self, "det_bank") else ""
        if b and getattr(raw, "instrument", "") == "sans_llb" and raw.panels and b in raw.panels:
            from .io.sans_llb import bank_run
            return bank_run(raw, b)
        return raw

    def _edges(self, raw, mask):
        from .core.averaging import q_edges
        qmin, qmax = self._q_limits(raw, mask)
        return q_edges(qmin, qmax, self.nbins.value(), self.spacing.currentData())

    def _q_limits(self, raw, mask):
        """(q min, q max) of the bins in 1/Å: the panel values, the data range of the unmasked pixels where
        a value is 0 (and for a wide-angle panel)."""
        from .core.geometry import run_geometry
        u = self.units
        qmin, qmax = u.q_in(self.qmin.value()), u.q_in(self.qmax.value())      # internal: 1/Å
        if getattr(raw, "panel", None) is not None:
            qmin = qmax = 0.0            # a wide-angle panel: its own q range
        if qmin <= 0 or qmax <= 0:
            g = run_geometry(raw, self.centre_for(raw))
            q = g.q[mask == 0]
            q = q[q > 0]
            qmin = qmin if qmin > 0 else float(q.min())
            qmax = qmax if qmax > 0 else float(q.max()) * (1 + 1e-9)
        if qmax <= qmin:
            raise ValueError(f"q max ({u.q_out(qmax):g}) must be larger than q min ({u.q_out(qmin):g}) {u.q}")
        self._last_bins = (f"{self.nbins.value()} {self.spacing.currentText()} q bins, "
                           f"{u.q_out(qmin):.4g}–{u.q_out(qmax):.4g} {u.q}")
        return qmin, qmax

    def python_code_table(self, table, rows):
        """Script that repeats the reduction of sample rows of the reduction table (2D with the correction data,
        then the average with the bins, sector and normalisation of the Reduction panel) ("Copy Python code")."""
        import base64
        import copy
        import dataclasses
        import zlib

        from ..core.pycode import Code, Script, call, lit, path_lit
        from .efficiency_gui import GEOMETRY_PREFIX
        from .reduction_table import RunChoice  # noqa: F401
        paths = {e.run: e.path for e in self.entries}
        u = self.units
        sc = Script("pySASred", f"reduction table {table.config}: " + ", ".join(r.label for r in rows[:6])
                    + (" ..." if len(rows) > 6 else ""))
        sc.imp("import base64", "import copy", "import zlib", "import numpy as np",
               "from pysasfit.sasred import names", "from pysasfit.sasred.absolute import average, reduce_row_2d",
               "from pysasfit.sasred.core.averaging import q_edges", "from pysasfit.sasred.io.raw import read_raw",
               "from pysasfit.sasred.reduce import load_instrument",
               "from pysasfit.sasred.reduction_table import ReductionTable, RunChoice")
        used = sorted({n for r in [*rows, *table.corrections.values()] for n in getattr(r, "used", []) or []
                       if n in paths})
        corr = {n: dict(self.name_corrections[n]) for n in used
                if self.name_corrections is not None and n in self.name_corrections}
        sc.comment("the reduction table of this geometry (correction data and samples)")
        sc.add(f"table = ReductionTable.from_dict({lit(table.to_dict())})")
        sc.comment("raw files of the runs the table uses; names and roles corrected by hand")
        sc.add(f"paths = {lit({n: os.path.abspath(paths[n]) for n in used})}",
               f"corrections = {lit(corr)}",
               "_cache = {}",
               "",
               "def reader(bank=''):",
               "    def read(n):",
               "        if n not in _cache:",
               "            _cache[n] = names.apply(read_raw(paths[n]), corrections)",
               "        if bank:",
               "            from pysasfit.sasred.io.sans_llb import bank_run",
               "            return bank_run(_cache[n], bank)",
               "        return _cache[n]",
               "    return read",
               "",
               "def unpack_mask(text, shape):",
               "    return np.frombuffer(zlib.decompress(base64.b64decode(text)), np.uint32).reshape(shape).copy()",
               "",
               "results = {}")
        eff_spec = (getattr(table, "efficiency", "") or "").strip()
        if table.scale_method() != "water" and eff_spec:
            if eff_spec.startswith(GEOMETRY_PREFIX):
                sc.comment(f"efficiency map from the table of {eff_spec[len(GEOMETRY_PREFIX):]}: reduce that table first "
                           "and pass its map here (not part of this script)")
                sc.add("efficiency = None")
            else:
                spec = eff_spec
                path = spec.partition("::")[0].strip()
                if not os.path.isabs(path) and getattr(self, "folder", ""):
                    spec = os.path.join(self.folder, path) + ("::" + spec.partition("::")[2] if "::" in spec else "")
                sc.imp("from pysasfit.sasred.absolute import load_efficiency")
                sc.add(f"efficiency = load_efficiency({spec!r})")
        else:
            sc.add("efficiency = None")
        sector = (self.sector_c.value(), self.sector_hw.value()) if self.sector.isChecked() else None
        masks = {}
        read0 = {}
        for row in rows:
            banks = [""]
            if row.used and row.used[0] in paths:
                try:
                    from .io.raw import read_raw
                    from . import names as nm
                    if row.used[0] not in read0:
                        read0[row.used[0]] = nm.apply(read_raw(paths[row.used[0]]), self.name_corrections)
                    banks = self.banks_for(read0[row.used[0]]) or [""]
                except Exception:  # noqa: BLE001
                    pass
            for sub, orig, spar in self._series_rows(table, row):
                for bank in banks:
                    raw0 = read0.get(sub.used[0])
                    if raw0 is None:
                        from .io.raw import read_raw
                        from . import names as nm
                        raw0 = read0[sub.used[0]] = nm.apply(read_raw(paths[sub.used[0]]), self.name_corrections)
                    if bank:
                        from .io.sans_llb import bank_run
                        raw0 = bank_run(raw0, bank)
                    tag = f"{table.config.split(',')[0]} {sub.label}" + (f" {bank}" if bank else "")
                    mask, how = self.mask_for(raw0)
                    m = np.ascontiguousarray(mask, np.uint32)
                    blob = base64.b64encode(zlib.compress(m.tobytes(), 9)).decode()
                    sc.comment(f"{tag} ({orig.info})")
                    if (m.shape, blob) not in masks:
                        masks[(m.shape, blob)] = f"mask{len(masks) + 1}"
                        sc.add(f"# {how}", f"{masks[(m.shape, blob)]} = unpack_mask(")
                        for i in range(0, len(blob), 100):
                            sc.add(f"    {blob[i:i + 100]!r}")
                        sc.add(f"    , {m.shape})")
                    k = table.samples.index(orig)
                    if sub is orig:
                        sc.add(f"row = table.samples[{k}]")
                    else:
                        sc.add(f"row = copy.copy(table.samples[{k}])   # one point of the time series",
                               f"row.runs = [RunChoice(n, True) for n in {lit([c.run for c in sub.runs])}]",
                               "row.series = False", f"row.label = {sub.label!r}")
                    inst = self.instrument_for(raw0)
                    sc.add(f"read = reader({bank!r})", "raw0 = read(row.used[0])", "instrument = load_instrument(raw0)")
                    nas = (inst.get("monitor") or {}).get("normaliser_after_selector")
                    if nas is not None and load_instrument_default(raw0).get("monitor", {}).get("normaliser_after_selector") is None:
                        sc.add(f"instrument.setdefault('monitor', {{}})['normaliser_after_selector'] = {lit(nas)}"
                               "   # from the monitor test of the folder")
                    prof = self.profile_for(raw0)
                    if prof is not None:
                        sc.imp("from pysasfit.sasred.core.beamprofile import BeamProfile")
                        sc.add(call("profile = BeamProfile", **dataclasses.asdict(prof)))
                    else:
                        sc.add("profile = None")
                    sc.add(call("r2 = reduce_row_2d", Code("table"), Code("row"), Code("read"),
                                tuple(float(v) for v in self.centre_for(raw0)), mask=Code(masks[(m.shape, blob)]),
                                instrument=Code("instrument"), normalise_by=self.norm.currentData(),
                                k0=Code("table.water_g"),
                                efficiency=Code("efficiency" if table.scale_method() != "water" else "None"),
                                beam_profile=Code("profile")))
                    qmin, qmax = self._q_limits(raw0, mask)
                    sc.add(call("r1 = average", Code("r2"), Code(f"q_edges({lit(qmin)}, {lit(qmax)}, "
                                                                 f"{self.nbins.value()}, {self.spacing.currentData()!r})"),
                                sector_deg=sector))
                    if spar:
                        sc.add(f"r1.parameters.update({lit(spar)})")
                    sc.add(f"results[{tag!r}] = r1")
        sc.add("for key, r in results.items():", "    print(key, len(r.q), 'bins', r.intensity.unit)")
        sc.add_plot("fig, ax = plt.subplots()", f"to_q = {lit(u.q_out(1.0))}   # 1/Å -> {u.q}",
                    "for key, r in results.items():",
                    "    ax.errorbar(r.q * to_q, r.I, r.idev(), fmt='.-', ms=3, label=key)",
                    "ax.set_xscale('log')", "ax.set_yscale('log')", f"ax.set_xlabel('q ({u.q})')",
                    "ax.set_ylabel('I')", "ax.legend(fontsize=7)")
        return sc.text()

    def python_code(self):
        """Script that repeats "Reduce selected" for the selected runs: raw file, name corrections, mask of the
        configuration (embedded, compressed), beam centre, beam profile (Qdev), q bins, normalisation, sector
        ("Copy Python code")."""
        import base64
        import dataclasses
        import zlib

        from ..core.pycode import Code, Script, call, lit, path_lit
        from .io.raw import read_raw
        from . import names as nm

        fw = QtWidgets.QApplication.focusWidget()
        tp = getattr(self, "table_panel", None)
        if tp is not None and fw is not None and tp.isAncestorOf(fw) and tp.page is not None:
            return tp.page.python_code()              # the reduction table is in use: its rows
        sel = [e for e in self.selected_entries() if e.raw is not None]
        if not sel:
            raise ValueError("Select the runs to reduce in the run table first.")
        u = self.units
        sc = Script("pySASred", f"reduction of run{'s' if len(sel) > 1 else ''} "
                    + ", ".join(str(e.run) for e in sel[:8]) + (" ..." if len(sel) > 8 else ""))
        sc.imp("import base64", "import zlib", "import numpy as np",
               "from pysasfit.sasred import names", "from pysasfit.sasred.core.averaging import q_edges",
               "from pysasfit.sasred.io.raw import read_raw", "from pysasfit.sasred.reduce import reduce_run_1d")
        sc.add("",
               "def unpack_mask(text, shape):",
               "    \"\"\"The pixel mask (uint32 reason bits) as copied from the GUI: zlib + base64.\"\"\"",
               "    return np.frombuffer(zlib.decompress(base64.b64decode(text)), np.uint32).reshape(shape).copy()",
               "",
               "results = {}")
        sector = (self.sector_c.value(), self.sector_hw.value()) if self.sector.isChecked() else None
        masks_written = {}
        for e in sel:
            for bank in self.banks_for(e.raw):
                raw = nm.apply(read_raw(e.path, bank=bank), self.name_corrections)
                mask, where = self.mask_for(raw)
                qmin, qmax = self._q_limits(raw, mask)
                key = self.result_key(raw.run, bank)
                sc.comment(f"run {key}: {raw.sample_name}")
                sc.add(f"raw = read_raw({path_lit(e.path)}" + (f", bank={bank!r})" if bank else ")"))
                corr = (self.name_corrections or {}).get(int(raw.run)) if self.name_corrections is not None else None
                if corr:
                    sc.add(f"names.apply(raw, {{{int(raw.run)}: {lit(dict(corr))}}})   # name / role corrected by hand")
                m = np.ascontiguousarray(mask, np.uint32)
                blob = base64.b64encode(zlib.compress(m.tobytes(), 9)).decode()
                mk = (m.shape, blob)
                if mk not in masks_written:
                    masks_written[mk] = f"mask{len(masks_written) + 1}"
                    sc.add(f"# {where}")
                    sc.add(f"{masks_written[mk]} = unpack_mask(")
                    for i in range(0, len(blob), 100):
                        sc.add(f"    {blob[i:i + 100]!r}")
                    sc.add(f"    , {m.shape})")
                prof = self.profile_for(raw)
                if prof is not None:
                    sc.imp("from pysasfit.sasred.core.beamprofile import BeamProfile")
                    sc.add(call("profile = BeamProfile", **dataclasses.asdict(prof)))
                else:
                    sc.add("profile = None   # no direct-beam profile for this geometry: no Qdev")
                sc.add(call("r = reduce_run_1d", Code("raw"), tuple(float(v) for v in self.centre_for(raw)),
                            Code(f"q_edges({lit(qmin)}, {lit(qmax)}, {self.nbins.value()}, "
                                 f"{self.spacing.currentData()!r})"),
                            mask=Code(masks_written[mk]), normalise_by=self.norm.currentData(), sector_deg=sector,
                            beam_profile=Code("profile")))
                if bank:
                    sc.add(f"r.parameters['bank'] = {bank!r}")
                sc.add(f"results[{key!r}] = r")
        sc.add("for key, r in results.items():",
               "    print(key, r.raw.sample_name, len(r.q), 'bins, dead-time factor', round(r.deadtime_factor, 4))")
        sc.add_plot("fig, ax = plt.subplots()",
                    f"to_q = {lit(u.q_out(1.0))}   # 1/Å -> {u.q}",
                    "for key, r in results.items():",
                    "    ax.errorbar(r.q * to_q, r.I, r.idev(), fmt='.-', ms=3, label=f'{key} {r.raw.sample_name}')",
                    "ax.set_xscale('log')", "ax.set_yscale('log')", f"ax.set_xlabel('q ({u.q})')",
                    "ax.set_ylabel('I (counts / monitor / sr)')", "ax.legend(fontsize=7)")
        return sc.text()

    def reduce_entry(self, e, bank: str = ""):
        """Reduce one catalogue entry with the panel settings (re-read with checksum); ``bank``: a
        SANS-LLB wide-angle panel ("left", "bottom") instead of the central detector."""
        from .io.raw import read_raw
        from .reduce import reduce_run_1d
        from . import names as nm
        raw = nm.apply(read_raw(e.path, bank=bank), self.name_corrections)
        mask = self.mask_for(raw)[0]
        sector = (self.sector_c.value(), self.sector_hw.value()) if self.sector.isChecked() else None
        r = reduce_run_1d(raw, self.centre_for(raw), self._edges(raw, mask), mask=mask,
                          normalise_by=self.norm.currentData(), sector_deg=sector,
                          beam_profile=self.profile_for(raw))
        if bank:
            r.parameters["bank"] = bank
        return r

    def on_reduce(self):
        sel = [e for e in self.selected_entries() if e.raw is not None]
        if not sel:
            QtWidgets.QMessageBox.information(self, "Reduce", "Select one or more runs in the run table.")
            return
        from .catalogue import config_key
        geos = sorted({config_key(e.raw) for e in sel})
        if len(geos) > 1:
            QtWidgets.QMessageBox.information(
                self, "Reduce", "The selected runs belong to " + ", ".join(f"SD {k[0]:g} m / λ {k[1]:g} Å" for k in geos)
                + ". Reduce one geometry at a time: the beam centre, the solid angle and the mask belong to "
                  "the geometry.")
            return
        QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.CursorShape.WaitCursor)
        self._last_keys = []
        try:
            for e, bank in [(e, b) for e in sel for b in self.banks_for(e.raw)]:
                try:
                    r = self.reduce_entry(e, bank)
                except Exception as exc:  # noqa: BLE001
                    self.log(f"run {e.run}{' ' + bank if bank else ''}: FAILED – {type(exc).__name__}: {exc}")
                    continue
                key = self.result_key(r.run, bank)
                self.results[key] = r
                self._last_keys.append(key)
                self.log(f"run {key} ({r.raw.sample_name}): {len(r.q)} bins ({getattr(self, '_last_bins', '')}), q = {self.units.q_out(r.q[0]):.4g} … "
                         f"{self.units.q_out(r.q[-1]):.4g} {self.units.q}, "
                         f"dead-time factor {r.deadtime_factor:.4f}")
                for n in r.notes:
                    self.log(f"    {n}")
                QtWidgets.QApplication.processEvents()
        finally:
            QtWidgets.QApplication.restoreOverrideCursor()
        self._refresh_results(select=self._last_keys[-1] if self._last_keys else None)

    def clear_results(self):
        self.results.clear()
        self._refresh_results()

    def _refresh_results(self, select=None):
        self.result_combo.blockSignals(True)
        cur = select if select is not None else self.result_combo.currentData()
        self.result_combo.clear()
        for run, r in self.results.items():
            self.result_combo.addItem(f"{run}  {r.parameters.get('sample', r.raw.sample_name)}", run)
        i = self.result_combo.findData(cur)
        self.result_combo.setCurrentIndex(i if i >= 0 else self.result_combo.count() - 1)
        self.result_combo.blockSignals(False)
        self._draw_iq()
        self._draw_budget()

    def current_result(self):
        run = self.result_combo.currentData()
        return self.results.get(run)

    # ------------------------------------------------------------------ plots
    def _draw_reduced_2d(self, fig):
        r = self.current_result()
        ax = fig.add_subplot(111)
        if r is None or r.image is None:
            ax.text(0.5, 0.5, "no reduced result chosen", ha="center", va="center", transform=ax.transAxes)
            return
        img = np.array(r.image.value, float)
        bad = (np.asarray(r.flags) != 0) | ~np.isfinite(img)
        img[bad] = np.nan
        norm = make_norm(self.det_scale.currentData(), np.nan_to_num(img, nan=0.0), valid=~bad)
        px, py = r.raw.pixel_size_m
        im = ax.imshow(img, origin="lower", norm=norm, cmap=self.det_cmap.currentText(),
                       interpolation="nearest", aspect=py / px)
        from ..toolbox.viewer import image_colorbar
        decorate_colorbar(image_colorbar(fig, ax, im, r.intensity.unit or ""), self.det_scale.currentData())
        c = r.parameters.get("beam_center_px")
        if c:
            ax.plot([c[0]], [c[1]], "+", color="red", ms=12, mew=1.5, zorder=5)
        ax.set_xlabel(f"x [pixel of {px * 1e3:g} mm]")
        ax.set_ylabel(f"y [pixel of {py * 1e3:g} mm]")
        lab = r.parameters.get("label")
        ax.set_title(f"reduced 2D: {lab + ' ' if lab else ''}{r.parameters.get('sample', r.raw.sample_name)}",
                     fontsize="medium")

    def _draw_detector(self):
        fig = self.det_canvas.figure
        fig.clear()
        if self.det_view.currentData() == "reduced2d":
            self._draw_reduced_2d(fig)
            self.mask_editor._ax = None
            self.det_canvas.draw_idle()
            return
        if self.det_view.currentData() == "efficiency":
            self._draw_efficiency(fig)
            self.mask_editor._ax = None
            self.det_canvas.draw_idle()
            return
        sel = [e for e in self.selected_entries() if e.raw is not None]
        if not sel:
            self.det_canvas.draw_idle()
            return
        raw = self.displayed_raw()
        self._sync_mask_config(raw)
        ax = fig.add_subplot(111)
        if self.det_view.currentData() == "llb_all":
            coord = self.det_axes.currentData()
            self._draw_llb_all(fig, ax, sel[0].raw, "angle" if coord == "pixels" else coord)
            self.mask_editor._ax = None
            self.det_canvas.draw_idle()
            return
        c = raw.counts.astype(float)
        c, cunit, cnote = self.display_image(raw, c)
        # colour range from the unmasked pixels only: bright pixels at the edge of a badly placed
        # beamstop must not dominate the colours
        valid = (self.full_mask(c.shape) == 0) & np.isfinite(c)
        norm = make_norm(self.det_scale.currentData(), np.nan_to_num(c), valid=valid)
        if self.det_view.currentData() == "polar" and raw.panel is None:
            self._draw_detector_polar(fig, ax, raw, c, norm, cunit)
            if cnote:
                ax.set_title(ax.get_title() + cnote, fontsize="medium")
            self.mask_editor._ax = None                     # editing only in the cartesian view
            self.mask_editor.attach()
            self.det_canvas.draw_idle()
            return
        if self.det_axes.currentData() in ("angle", "q"):
            self._draw_llb_all(fig, ax, raw, self.det_axes.currentData(), single=(c, cunit, cnote, norm))
            self.mask_editor._ax = None                     # editing only in pixels
            self.det_canvas.draw_idle()
            return
        px, py = raw.pixel_size_m
        im = ax.imshow(c, origin="lower", norm=norm, cmap=self.det_cmap.currentText(),
                       interpolation="nearest", aspect=py / px)        # true pixel aspect ratio
        from ..toolbox.viewer import image_colorbar
        decorate_colorbar(image_colorbar(fig, ax, im, cunit), self.det_scale.currentData())
        self.mask_editor.draw_overlay(ax, c.shape, image=np.nan_to_num(c), scale=self.det_scale.currentData())
        x, y = self.xc.value(), self.yc.value()
        if raw.panel is not None:                  # wide-angle panel: no beam centre on it
            # orientation as the central detector: x grows against posx, y with posy (the left panel's rows
            # run from top to bottom)
            if raw.panel.posy_mm[0] > raw.panel.posy_mm[-1]:
                ax.invert_yaxis()
            if raw.panel.posx_mm[0] < raw.panel.posx_mm[-1]:
                ax.invert_xaxis()
            ax.set_xlabel(f"x [pixel of {px * 1e3:g} mm]")
            ax.set_ylabel(f"y [pixel of {py * 1e3:g} mm]" + (" (row 0 at the top)" if raw.panel.posy_mm[0] > raw.panel.posy_mm[-1] else ""))
            ax.set_title(f"run {raw.run}: {raw.sample_name} – {raw.bank} panel{cnote}", fontsize="medium")
            self.det_canvas.draw_idle()
            return
        ax.plot([x], [y], "+", color="red", ms=12, mew=1.5, zorder=5)
        if self.sector.isChecked():
            L = max(c.shape)
            for s in (-1, 1):
                for base in (0.0, 180.0):
                    a = np.deg2rad(base + self.sector_c.value() + s * self.sector_hw.value())
                    ax.plot([x, x + L * np.cos(a)], [y, y + L * np.sin(a)], ":", color="white", lw=1)
            ax.set_xlim(-0.5, c.shape[1] - 0.5)
            ax.set_ylim(-0.5, c.shape[0] - 0.5)
        ax.set_xlabel(f"x [pixel of {px * 1e3:g} mm]")
        ax.set_ylabel(f"y [pixel of {py * 1e3:g} mm]")
        ax.set_title(f"run {raw.run}: {raw.sample_name}{cnote}", fontsize="medium")
        self.det_canvas.draw_idle()

    def detector_coordinates(self, g, coord):
        """Plot coordinates of the pixels of geometry ``g``: ``coord`` "angle": 2θ·(cos φ, sin φ) in degrees,
        "q": |q|·(cos φ, sin φ) in the q unit shown; φ the azimuth of the pixel (x of the central detector = 0°).
        Rings are circles of radius 2θ or q, whatever the detector distance."""
        phi = np.arctan2(g.y, g.x)
        if coord == "q":
            rad = self.units.q_out(np.asarray(g.q, float))
        else:
            rad = np.degrees(np.arctan2(np.hypot(g.x, g.y), g.z))
        return rad * np.cos(phi), rad * np.sin(phi)

    def _draw_llb_all(self, fig, ax, raw, coord="angle", single=None):
        """Detectors over scattering angle or scattering vector (geometry with the current beam centre and the
        SANS-LLB panel offsets): without ``single`` the SANS-LLB central detector and the wide-angle panels
        together, counts per steradian; with ``single`` = (image, unit, note, norm) only ``raw`` (one bank, any
        instrument) with the image of the cartesian view. The mask of every bank is shown as in the cartesian
        view (mask colours, own grey scale or hidden, and the outline)."""
        from matplotlib.collections import LineCollection
        from matplotlib.colors import LogNorm
        from scipy.interpolate import RegularGridInterpolator
        from .core.geometry import run_geometry
        from .mask_editor import MASKED_CMAP, outline_segments, overlay_rgba
        if single is None and (getattr(raw, "instrument", "") != "sans_llb" or not raw.panels):
            ax.text(0.5, 0.5, "not a SANS-LLB run", ha="center", va="center", transform=ax.transAxes)
            return
        centre = self.centre_for(raw)
        mode = self.mask_editor.mode
        alpha = self.mask_editor.opacity.value() / 100
        parts = []
        if single is None:
            from .io.sans_llb import bank_run
            runs = [(b or "central", bank_run(raw, b) if b else raw) for b in ["", *raw.panels]]
        else:
            runs = [(getattr(raw, "bank", "") or "", raw)]
        for b, r in runs:
            g = run_geometry(r, centre)
            ang_x, ang_y = self.detector_coordinates(g, coord)
            ny, nx = ang_x.shape
            fx = RegularGridInterpolator((np.arange(ny), np.arange(nx)), ang_x, bounds_error=False, fill_value=None)
            fy = RegularGridInterpolator((np.arange(ny), np.arange(nx)), ang_y, bounds_error=False, fill_value=None)

            def to_angles(px, py, fx=fx, fy=fy):           # pixel coordinates (x = column, y = row) -> angles
                pts = np.column_stack([np.ravel(py), np.ravel(px)])
                return fx(pts).reshape(np.shape(px)), fy(pts).reshape(np.shape(px))
            ey, ex = np.mgrid[-0.5:ny, -0.5:nx]
            Xe, Ye = to_angles(ex, ey)
            dens = r.counts.astype(float) / g.d_omega if single is None else np.asarray(single[0], float)
            m = np.asarray(self.mask_for(r)[0], np.uint32)
            parts.append((b, ang_x, ang_y, Xe, Ye, dens, m, to_angles))
        sep = single is None and self.det_sep.isChecked()

        def group(name):                                   # colour group of a bank
            return ("central" if name == "central" else "panels") if sep else "all"
        norms = {}
        if single is None:
            for gname in dict.fromkeys(group(p[0]) for p in parts):
                sub = [p for p in parts if group(p[0]) == gname]
                good = [d[(m == 0) & (d > 0)] for *_, d, m, _t in sub]
                allv = np.concatenate(good) if good else np.array([])
                if allv.size == 0:
                    allv = np.concatenate([d[d > 0] for *_, d, m, _t in sub]) if sub else np.array([])
                if allv.size == 0:
                    allv = np.array([1.0])
                norms[gname] = LogNorm(vmin=max(np.percentile(allv, 1), 1e-12),
                                       vmax=max(np.percentile(allv, 99.9), np.percentile(allv, 1) * 1.0001, 1e-12))
        else:
            norms["all"] = single[3]
        cmap_c = self.det_cmap.currentText()
        cmap_p = self.det_cmap_panels.currentText() if self.det_cmap_panels.currentData() != "" else cmap_c
        ims = {}
        im = None
        for name, ax_, ay_, Xe, Ye, d, m, to_angles in parts:
            masked = m != 0
            hide = ~np.isfinite(d) | ((d <= 0) if single is None else False)
            show = np.ma.masked_where(hide | (masked & (mode != "colours")), d)
            gname = group(name)
            im = ax.pcolormesh(Xe, Ye, show, norm=norms[gname], shading="flat",
                               cmap=cmap_p if name not in ("central", "") and single is None else cmap_c)
            ims.setdefault(gname, im)
            if masked.any():
                if mode == "colours":
                    qm = ax.pcolormesh(Xe, Ye, np.zeros(d.shape), shading="flat", zorder=3)
                    qm.set_array(None)
                    qm.set_facecolor(overlay_rgba(m, alpha).reshape(-1, 4))
                elif mode == "scale":
                    dm = np.ma.masked_where(~masked | (d <= 0), d)
                    if dm.count():
                        ax.pcolormesh(Xe, Ye, dm, cmap=MASKED_CMAP, shading="flat", zorder=3,
                                      norm=LogNorm(vmin=max(float(dm.min()), 1e-12), vmax=float(dm.max()) * 1.0001))
                else:
                    ax.pcolormesh(Xe, Ye, np.ma.masked_where(~masked, np.ones(d.shape)), cmap="Greys", vmin=0, vmax=4,
                                  shading="flat", zorder=3)
                if not masked.all():
                    segs = []
                    for (x0, y0), (x1, y1) in outline_segments(masked):
                        n = max(2, int(round(abs(x1 - x0) + abs(y1 - y0))) + 1)   # follow curved pixel rows
                        px, py = np.linspace(x0, x1, n), np.linspace(y0, y1, n)
                        X, Y = to_angles(px, py)
                        segs.append(np.column_stack([X, Y]))
                    for col, lw in (("black", 2.2), ("white", 0.9)):
                        ax.add_collection(LineCollection(segs, colors=col, linewidths=lw, zorder=4), autolim=False)
            if name:
                ax.text(float(np.median(ax_)), float(np.max(Ye)), name, ha="center", va="bottom", fontsize=8)
        ax.plot([0], [0], "+", color="red", ms=12, mew=1.5, zorder=5)
        ax.set_aspect("equal")
        ax.autoscale_view()
        if coord == "q":
            ax.set_xlabel(f"q_x = |q| cos φ [{self.units.q}]")
            ax.set_ylabel(f"q_y = |q| sin φ [{self.units.q}]")
        else:
            ax.set_xlabel("2θ cos φ [°]")
            ax.set_ylabel("2θ sin φ [°]")
        n_m = sum(int(np.count_nonzero(m)) for *_, m, _t in parts)
        if single is None:
            ax.set_title(f"run {raw.run}: {raw.sample_name} – all detectors, counts/sr ({n_m} px masked)",
                         fontsize="medium")
            for gname, im_ in reversed(list(ims.items())):     # the last bar is next to the axes
                fig.colorbar(im_, ax=ax, pad=0.02, fraction=0.05,
                             label="counts / sr" if gname == "all" else f"{gname}: counts / sr")
        else:
            bank = f" – {raw.bank} panel" if getattr(raw, "bank", "") else ""
            ax.set_title(f"run {raw.run}: {raw.sample_name}{bank}{single[2]}", fontsize="medium")
            if im is not None:
                from ..toolbox.viewer import image_colorbar
                decorate_colorbar(image_colorbar(fig, ax, im, single[1]), self.det_scale.currentData())

    def _draw_detector_polar(self, fig, ax, raw, c, norm, cunit="counts"):
        """Detector image over (φ, q): φ = 0 horizontal (+x), counter-clockwise; masked cells in the
        mask colours, outside the detector light grey."""
        import matplotlib
        from .core.polar import polar_image
        u = self.units
        xc, yc = self.xc.value(), self.yc.value()
        nonfinite = ~np.isfinite(c)
        c = np.nan_to_num(c)
        pol = polar_image(c, (xc, yc), raw.pixel_size_m, getattr(raw, "SD_effective_m", raw.SD_m), raw.wavelength_A,
                          invalid=(self._mask(c.shape) != 0) | nonfinite)
        cmap = matplotlib.colormaps[self.det_cmap.currentText()].with_extremes(bad="0.85")
        qe = u.q_out(pol.q_edges)
        im = ax.pcolormesh(pol.phi_edges_deg, qe, pol.image, norm=norm, cmap=cmap, shading="flat",
                           rasterized=True)
        from ..toolbox.viewer import image_colorbar
        decorate_colorbar(image_colorbar(fig, ax, im, cunit), self.det_scale.currentData())
        ax.set_anchor("W")
        mode = self.mask_editor.mode
        if mode == "scale":                                  # masked cells on their own grey scale
            from .mask_editor import MASKED_CMAP
            masked = self._mask(c.shape) != 0
            if masked.any():
                full = polar_image(c, (xc, yc), raw.pixel_size_m, getattr(raw, "SD_effective_m", raw.SD_m), raw.wavelength_A).image
                inv = polar_image(masked.astype(float), (xc, yc), raw.pixel_size_m, getattr(raw, "SD_effective_m", raw.SD_m),
                                  raw.wavelength_A).image
                sel = np.isfinite(inv) & (np.nan_to_num(inv) > 0)
                if sel.any():
                    im2 = ax.pcolormesh(pol.phi_edges_deg, qe, np.ma.masked_where(~sel, full), cmap=MASKED_CMAP,
                                        norm=make_norm(self.det_scale.currentData(), c, valid=masked),
                                        shading="flat", rasterized=True, zorder=3)
                    cax = ax.inset_axes([1.24, 0.0, 0.045, 1.0])
                    decorate_colorbar(fig.colorbar(im2, cax=cax, label="masked pixels"),
                                      self.det_scale.currentData())
        if mode == "colours":                                # masked cells in the mask colours
            from matplotlib.colors import ListedColormap
            from .mask_editor import OVERLAY_COLOURS
            bits = polar_image(self._mask(c.shape).astype(float), (xc, yc), raw.pixel_size_m, getattr(raw, "SD_effective_m", raw.SD_m),
                               raw.wavelength_A).image
            alpha = self.mask_editor.opacity.value() / 100
            for bit, col in OVERLAY_COLOURS:
                sel = np.isfinite(bits) & ((np.nan_to_num(bits).astype(np.uint32) & bit) != 0)
                if sel.any():
                    ax.pcolormesh(pol.phi_edges_deg, qe, np.ma.masked_where(~sel, np.ones_like(bits)),
                                  cmap=ListedColormap([col]), alpha=alpha, shading="flat", rasterized=True,
                                  zorder=3)
        if self.sector.isChecked():
            for base in (0.0, 180.0):
                for sgn in (-1, 1):
                    a = (base + self.sector_c.value() + sgn * self.sector_hw.value()) % 360.0
                    ax.axvline(a, color="white", ls=":", lw=1)
        ax.set_xlim(0, 360)
        ax.set_xticks(np.arange(0, 361, 45))
        ax.set_ylim(0, qe[-1])
        ax.set_xlabel("azimuth φ [°]  (0° = horizontal, +x)")
        ax.set_ylabel(f"q [{u.q}]")
        ax.set_title(f"run {raw.run}: {raw.sample_name}", fontsize="medium")

    def shown_results(self):
        """(key, result) pairs that I(q) shows (see the 'show' selector of the I(q) panel)."""
        mode = self.iq_show.currentData() if hasattr(self, "iq_show") else "all"
        if mode == "all":
            keys = list(self.results)
        else:
            keys = []
            if mode == "selected":
                keys = [e.run for e in self.selected_entries() if e.run in self.results]
                try:
                    page = self.table_panel.page
                    if page is not None:
                        pre = page.table.config.split(",")[0]
                        keys += [k for k in (f"{pre} {r.label}" for r in page.selected_rows()) if k in self.results]
                except Exception:  # noqa: BLE001
                    pass
            if not keys:
                keys = [k for k in self._last_keys if k in self.results]
        return [(k, self.results[k]) for k in dict.fromkeys(keys)]

    def _iq_scales(self):
        """(x, y) scale of the I(q) plot. A scale changed on the plot itself (toolbar figure options, keys k / l)
        is taken over into the selectors first, so that redrawing for other curves keeps it."""
        if not hasattr(self, "iq_xscale"):
            return "log", "log"
        axes = getattr(self, "_iq_ax", None)
        drawn = getattr(self, "_iq_drawn_scales", None)
        if axes is not None and drawn is not None and axes.figure is self.iq_canvas.figure:
            for cb, now, was in ((self.iq_xscale, axes.get_xscale(), drawn[0]),
                                 (self.iq_yscale, axes.get_yscale(), drawn[1])):
                if now != was and now in ("log", "linear"):
                    i = cb.findData(now)
                    if i >= 0 and i != cb.currentIndex():
                        cb.blockSignals(True)
                        cb.setCurrentIndex(i)
                        cb.blockSignals(False)
        return self.iq_xscale.currentData(), self.iq_yscale.currentData()

    def _draw_iq(self):
        xscale, yscale = self._iq_scales()
        fig = self.iq_canvas.figure
        fig.clear()
        ax = fig.add_subplot(111)
        u = self.units
        shown_items = self.shown_results()
        f = self.iq_offset.value() if hasattr(self, "iq_offset") else 1.0
        show_dI = self.iq_err_I.isChecked() if hasattr(self, "iq_err_I") else True
        show_dq = self.iq_err_q.isChecked() if hasattr(self, "iq_err_q") else True
        # points of one time series: coloured by their time (colour bar), the legend gives way to it
        times = [r.parameters.get("time_s") for _, r in shown_items]
        series = (len(shown_items) > 1 and all(t is not None for t in times)
                  and len({r.parameters.get("series") for _, r in shown_items}) == 1)
        if series:
            import matplotlib as mpl
            norm = mpl.colors.Normalize(min(times), max(times) if max(times) > min(times) else min(times) + 1)
            cmap = mpl.colormaps["viridis"]
        for k, (key, r) in enumerate(shown_items):
            lab = r.parameters.get("label") or key
            col = {"color": cmap(norm(r.parameters["time_s"]))} if series else {}
            ax.errorbar(u.q_out(r.q), r.I * f ** k, yerr=r.idev() * f ** k if show_dI else None,
                        xerr=_qerr(r, u) if show_dq else None, fmt="o", ms=3, **col,
                        capsize=0, lw=0.8,
                        label=f"{lab} {r.parameters.get('sample', r.raw.sample_name)}" + (f" ×{f ** k:g}" if k and f != 1 else ""))
        p = self.preview
        lives = ([p] if p is not None else []) + list(getattr(self, "preview_banks", []))
        live_cols = {"central": "0.15", "left": "tab:red", "bottom": "tab:blue"}
        for lp in lives:
            bank = lp.parameters.get("bank", "")
            tag = f" ({bank or 'central'})" if len(lives) > 1 or bank else ""
            ax.errorbar(u.q_out(lp.q), lp.I, yerr=lp.idev() if show_dI else None,
                        xerr=_qerr(lp, u) if show_dq else None, fmt="--", color=live_cols.get(bank or "central", "0.15"),
                        lw=1.0, capsize=0, elinewidth=0.6, label=f"live: {lp.run} {lp.raw.sample_name}{tag}")
        shown = [r for _, r in shown_items] + lives
        if shown:
            ax.set_xscale(xscale)
            if yscale == "log":
                ax.set_yscale("log", nonpositive="mask")       # I ≤ 0 (after subtraction) not drawn, scale kept
            if xscale == "log":
                _log_q_ticks(ax)
            if series:
                sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
                fig.colorbar(sm, ax=ax, label="t [s]", fraction=0.06, pad=0.02, aspect=30)
                ax.set_title(f"time series, {len(shown_items)} points; t₀ = "
                             f"{shown_items[0][1].parameters.get('series_t0', '')}", fontsize="small")
            elif len(shown) <= 12:
                ax.legend(fontsize="small")
            else:                                  # a legend of many curves would squeeze the axes away
                ax.set_title(f"{len(shown)} curves – select runs or table rows to see fewer (with legend)",
                             fontsize="small")
            ax.set_ylabel(f"I [{shown[0].intensity.unit}]")
        elif self.results:
            ax.set_title("select reduced runs or table rows ('show' above the plot)", fontsize="small")
        ax.set_xlabel(f"q [{u.q}]")
        self._iq_ax, self._iq_drawn_scales = ax, (ax.get_xscale(), ax.get_yscale())   # as drawn (empty: linear)
        self.iq_canvas.draw_idle()

    def _draw_budget(self):
        fig = self.budget_canvas.figure
        fig.clear()
        ax = fig.add_subplot(111)
        r = self.current_result()
        if r is not None:
            I = np.abs(r.I)
            src = r.intensity.sources
            for key, var in sorted(r.intensity.budget().items(), key=lambda kv: -np.max(kv[1])):
                kind = "systematic" if src[key].systematic else "statistical"
                ax.plot(self.units.q_out(r.q), np.sqrt(var) / I, "-", label=f"{key} ({kind})")
            ax.plot(self.units.q_out(r.q), r.idev() / I, "k--", lw=1.5, label="total statistical (Idev)")
            ax.plot(self.units.q_out(r.q), r.systematic_std() / I, "k:", lw=1.5, label="total systematic")
            ax.set_xscale("log")
            ax.set_yscale("log")
            _log_q_ticks(ax)
            ax.legend(fontsize="x-small")
            ax.set_title(f"run {r.run}", fontsize="medium")
        ax.set_xlabel(f"q [{self.units.q}]")
        ax.set_ylabel("relative standard deviation")
        self.budget_canvas.draw_idle()

    # ------------------------------------------------------------------ saving
    def _need_result(self):
        r = self.current_result()
        if r is None:
            QtWidgets.QMessageBox.information(self, "Save", "No reduced run yet.")
        return r

    def _default_name(self, r, ext):
        base = f"{r.run}_{''.join(ch if ch.isalnum() or ch in '-_' else '_' for ch in r.raw.sample_name)}"
        if getattr(r.raw, "bank", ""):
            base += f"_{r.raw.bank}"
        return os.path.join(self.folder or os.getcwd(), base + ext)

    def on_save_nxcansas(self):
        r = self._need_result()
        if r is None:
            return
        p, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save NXcanSAS", self._default_name(r, ".h5"),
                                                     "NXcanSAS (*.h5 *.nxs)")
        if p:
            from .io.nxcansas import write_nxcansas_1d
            write_nxcansas_1d(p, r, units=self.unit)
            self.log(f"saved {p}")

    def on_save_2d(self):
        r = self._need_result()
        if r is None:
            return
        p, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save 2D NXcanSAS", self._default_name(r, "_2d.h5"),
                                                     "NXcanSAS (*.h5 *.nxs)")
        if p:
            from .io.nxcansas import write_nxcansas_2d
            write_nxcansas_2d(p, r, units=self.unit)
            self.log(f"saved {p}")

    def on_save_ascii(self):
        r = self._need_result()
        if r is None:
            return
        p, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save ASCII", self._default_name(r, ".dat"),
                                                     "text (*.dat *.txt)")
        if p:
            from .io.ascii import write_ascii_1d
            write_ascii_1d(p, r, units=self.unit)
            self.log(f"saved {p}")

    def save_all(self, folder):
        """NXcanSAS and ASCII of every reduced run, and the 2D image as ``<name>_2d.h5``, into
        ``folder``; returns the paths."""
        from .io.ascii import write_ascii_1d
        from .io.nxcansas import write_nxcansas_1d, write_nxcansas_2d
        out = []
        for r in self.results.values():
            stem = os.path.splitext(os.path.basename(self._default_name(r, "")))[0]
            p = os.path.join(folder, stem + ".h5")
            write_nxcansas_1d(p, r, units=self.unit)
            out.append(p)
            out.append(write_ascii_1d(os.path.join(folder, stem + ".dat"), r, units=self.unit))
            out.append(write_nxcansas_2d(os.path.join(folder, stem + "_2d.h5"), r, units=self.unit))
        return out

    def on_save_all(self):
        if not self.results:
            QtWidgets.QMessageBox.information(self, "Save", "No reduced run yet.")
            return
        d = QtWidgets.QFileDialog.getExistingDirectory(self, "Folder for the reduced data", self.folder or os.getcwd())
        if d:
            for p in self.save_all(d):
                self.log(f"saved {p}")

    # ------------------------------------------------------------------ session (launcher hooks)
    def _session(self):
        out = {"folder": self.folder, "xc": self.xc.value(), "yc": self.yc.value(),
                "beamstop_radius_px": self.bs_radius.value(), "beamstop_shape": self.bs_shape.currentData(),
                "nbins": self.nbins.value(),
                "prefs_at_save": {k: prefs.get(k) for k in self._PANEL_PREFS},
                "qmin": self.units.q_in(self.qmin.value()), "qmax": self.units.q_in(self.qmax.value()),
                "units": self.unit, "live": self.live.isChecked(), "spacing": self.spacing.currentData(),
                "normalise_by": self.norm.currentData(), "sector": self.sector.isChecked(),
                "sector_c": self.sector_c.value(), "sector_hw": self.sector_hw.value(),
                "detector_scale": self.det_scale.currentData(), "detector_cmap": self.det_cmap.currentText(),
                "detector_view": self.det_view.currentData(), "bank": self.bank_combo.currentData(),
                "detector_bank": self.det_bank.currentData(),
                "detector_axes": self.det_axes.currentData(), "detector_panel_scale": self.det_sep.isChecked(),
                "detector_panel_cmap": self.det_cmap_panels.currentText(), "detector_efficiency": self.det_eff.isChecked(), "detector_solid_angle": self.det_domega.isChecked(),
                "mask_tools": self.mask_tools_button.isChecked(), "iq_show": self.iq_show.currentData(),
                "iq_error_bars_I": self.iq_err_I.isChecked(), "iq_error_bars_q": self.iq_err_q.isChecked(),
                "iq_xscale": self._iq_scales()[0], "iq_yscale": self._iq_scales()[1],
                "centres": [[c.key[0], c.key[1], c.key[2], c.x, c.y, c.source] for c in self.centres.values()
                            if self._user_centre(c)],
                "tables": dict(self.table_panel.to_dict(), folder=self.folder),
                "table_tab": self.table_panel.tabs.currentIndex(),
                "dock_state": self.dock_state(), "layout_version": LAYOUT_VERSION}
        u = getattr(self, "_unread", None)
        if u and not self.folder:           # the last folder was not read in this session: keep it as it was
            out.update(folder=u["folder"], tables=u["tables"], table_tab=u.get("table_tab"))
        return out

    @staticmethod
    def _session_mask_path():
        return os.path.join(os.path.dirname(prefs.settings_path()), "sasred_session_mask.h5")

    @staticmethod
    def _session_masks_path():
        return os.path.join(os.path.dirname(prefs.settings_path()), "sasred_session_masks.h5")

    def save_session(self):
        try:
            self._commit_mask()
            if self.masks or self.tregions:
                os.makedirs(os.path.dirname(self._session_masks_path()), exist_ok=True)
                self.save_masks(self._session_masks_path())
        except Exception:  # noqa: BLE001 -- the session is a convenience
            pass
        if self.mask_editor.mask is not None:
            try:
                from .core.mask import save_mask
                os.makedirs(os.path.dirname(self._session_mask_path()), exist_ok=True)
                save_mask(self._session_mask_path(), self.mask_editor.mask,
                          beam_center_px=(self.xc.value(), self.yc.value()))
            except Exception:  # noqa: BLE001 -- the session is a convenience
                pass
        try:
            p = _session_path()
            os.makedirs(os.path.dirname(p), exist_ok=True)
            with open(p, "w", encoding="utf-8") as fh:
                json.dump(self._session(), fh, indent=1)
        except OSError:
            pass

    def _autosave_session(self):
        if not getattr(self, "folder", ""):            # nothing opened yet: keep the last session
            return
        try:
            self.save_session()
        except Exception:  # noqa: BLE001 -- the session is a convenience
            pass

    def restore_session(self):
        try:
            with open(_session_path(), encoding="utf-8") as fh:
                s = json.load(fh)
        except (OSError, ValueError):
            return False
        if s.get("units") in ("nm", "A"):
            self.set_units(s["units"])
        for key, w in (("xc", self.xc), ("yc", self.yc), ("beamstop_radius_px", self.bs_radius),
                       ("sector_c", self.sector_c), ("sector_hw", self.sector_hw)):
            if isinstance(s.get(key), (int, float)):
                w.setValue(float(s[key]))
        for key, w in (("qmin", self.qmin), ("qmax", self.qmax)):          # stored in 1/Å
            if isinstance(s.get(key), (int, float)):
                w.setValue(self.units.q_out(float(s[key])))
        if "live" in s:
            self.live.setChecked(bool(s["live"]))
        # a preference changed since the session was saved wins over the session value (otherwise a
        # q-bin spacing set in Settings → Preferences would be undone at every start)
        saved = s.get("prefs_at_save") or {}

        def keep(key, pref):
            if pref not in saved:                       # older session: the preference wins when they differ
                return s.get(key) == prefs.get(pref)
            return saved[pref] == prefs.get(pref)
        if isinstance(s.get("nbins"), int) and keep("nbins", "sasred.qbins.n"):
            self.nbins.setValue(s["nbins"])
        for key, combo, pref in (("spacing", self.spacing, "sasred.qbins.spacing"),
                                 ("normalise_by", self.norm, "sasred.normalise_by"),
                                 ("beamstop_shape", self.bs_shape, "sasred.mask.beamstop_shape")):
            if not keep(key, pref):
                continue
            i = combo.findData(s.get(key))
            if i >= 0:
                combo.setCurrentIndex(i)
        self.sector.setChecked(bool(s.get("sector")))
        i = self.det_scale.findData(s.get("detector_scale"))
        if i >= 0:
            self.det_scale.setCurrentIndex(i)
        i = self.det_cmap.findText(str(s.get("detector_cmap", "")))
        if i >= 0:
            self.det_cmap.setCurrentIndex(i)
        i = self.det_view.findData(s.get("detector_view"))
        if i >= 0:
            self.det_view.setCurrentIndex(i)
        i = self.bank_combo.findData(s.get("bank"))
        if i >= 0:
            self.bank_combo.setCurrentIndex(i)
        i = self.det_bank.findData(s.get("detector_bank", ""))
        if i >= 0:
            self.det_bank.setCurrentIndex(i)
        if "mask_tools" in s:
            self.mask_tools_button.setChecked(bool(s["mask_tools"]))
        self.det_eff.setChecked(bool(s.get("detector_efficiency", False)))
        self.det_axes.setCurrentIndex(max(0, self.det_axes.findData(s.get("detector_axes", "pixels"))))
        self.det_sep.setChecked(bool(s.get("detector_panel_scale", True)))
        self.det_cmap_panels.setCurrentIndex(max(0, self.det_cmap_panels.findText(s.get("detector_panel_cmap", "as central"))))
        self.det_domega.setChecked(bool(s.get("detector_solid_angle", False)))
        for key, cb in (("iq_error_bars_I", self.iq_err_I), ("iq_error_bars_q", self.iq_err_q)):
            if key in s:
                cb.setChecked(bool(s[key]))
        for key, cb in (("iq_xscale", self.iq_xscale), ("iq_yscale", self.iq_yscale)):
            i = cb.findData(s.get(key))
            if i >= 0:
                cb.setCurrentIndex(i)
        i = self.iq_show.findData(s.get("iq_show"))
        if i >= 0:
            self.iq_show.setCurrentIndex(i)
        if s.get("layout_version") == LAYOUT_VERSION:     # an older arrangement is not restored
            self.restore_dock_state(s.get("dock_state"))
        if os.path.isfile(self._session_mask_path()):
            try:
                from .core.mask import BEAMSTOP, beamstop_guess, load_mask
                m, _info = load_mask(self._session_mask_path())
                if "beamstop_shape" not in s and self.bs_radius.value() > 0 and not (m & BEAMSTOP).any():
                    # older sessions kept the beamstop disc outside the mask: put it in
                    m[beamstop_guess(m.shape, (self.xc.value(), self.yc.value()), self.bs_radius.value(),
                                     "disc")] |= BEAMSTOP
                self.mask_editor.mask = m
                self.mask_editor.pristine = False
                self.mask_editor._update_stats()
            except Exception:  # noqa: BLE001
                pass
        if os.path.isfile(self._session_masks_path()):
            try:
                self.load_masks(self._session_masks_path())
            except Exception:  # noqa: BLE001
                pass
        from .catalogue import GeometryCentre
        for item in s.get("centres", []) or []:      # before the folder: set_folder keeps them
            try:
                if len(item) == 5:          # sessions before the centre depended on coll: not used
                    continue
                SD, lam, coll, x, y, src = item
                k = (float(SD), float(lam), float(coll))
                self.centres[k] = GeometryCentre(k, float(x), float(y), 0.0, 0.0, [], str(src))
            except (TypeError, ValueError):
                pass
        if s.get("folder"):
            # the folder is not read now (reading the headers of every run takes long): Rescan reads it, and
            # then the reduction tables of the last session are put back
            self.folder_edit.setText(s["folder"])
            self._unread = {"folder": s["folder"], "tables": s.get("tables") or {}, "table_tab": s.get("table_tab")}
            self.log(f"last raw-data folder: {s['folder']} -- press Rescan to read its runs "
                     "(the reduction tables of the last session are restored then)")
            self.statusBar().showMessage("last folder not read yet: press Rescan", 15000)
        return True

    def _restore_session_tables(self):
        """After the folder of the last session was read: its reduction tables, with every change by hand."""
        u = getattr(self, "_unread", None)
        self._unread = None
        if not u or os.path.normpath(u["folder"]) != os.path.normpath(self.folder):
            return
        tb = u["tables"]
        if tb.get("geometries") and os.path.normpath(tb.get("folder") or "") == os.path.normpath(self.folder):
            try:
                self.table_panel.from_dict(tb)
                i = u.get("table_tab")
                if isinstance(i, int) and 0 <= i < self.table_panel.tabs.count():
                    self.table_panel.tabs.setCurrentIndex(i)
                self.log("reduction tables of the last session restored")
            except Exception as exc:  # noqa: BLE001 -- the session is a convenience
                self.log(f"reduction tables of the last session not restored: {exc}")

    def forget_session(self):
        for p in (_session_path(), self._session_mask_path()):
            try:
                os.remove(p)
            except OSError:
                pass

    def shutdown(self):
        self.save_session()

    def closeEvent(self, event):
        self.shutdown()
        MainWindow.instances.discard(self)          # a closed window no longer offers its results
        super().closeEvent(event)


def main(argv=None):
    argv = sys.argv if argv is None else argv
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(argv)
    from ..core.tabstyle import apply_tab_style
    apply_tab_style(app)
    app.setWindowIcon(tool_icon("pySASred"))
    w = MainWindow()
    w.show()
    if len(argv) > 1 and os.path.isdir(argv[1]):
        w.restore_session()
        w.set_folder(argv[1])
    else:
        w.restore_session()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
