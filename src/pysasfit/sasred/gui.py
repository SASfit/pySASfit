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


def _lam_column() -> int:
    from .catalogue import COLUMNS
    return COLUMNS.index("λ [nm]")


class MainWindow(DockMixin, QtWidgets.QMainWindow):
    instances = __import__("weakref").WeakSet()      # open windows (pyTOOLbox takes their results)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle(f"pySASred {__version__} – SANS data reduction (SINQ SANS-I)")
        self.setWindowIcon(tool_icon("pySASred"))
        self.folder = ""
        self.entries = []            # CatalogueEntry list of the folder
        self.results = {}            # run -> ReducedRun
        self.beam_centre = None      # BeamCentre of the last determination
        self.centres = {}            # geometry (L2, λ) -> GeometryCentre: q and solid angle use it
        self.masks = {}              # instrument configuration (catalogue.mask_key) -> edited uint32 mask
        self._mask_key = None        # configuration of the mask in the editor
        self._centre_sync = False
        self.preview = None          # on-the-fly radial average of the selected run (not kept)
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
        add_preferences_menu(self, "sasred")
        self._build_runs()
        self._build_reduction()
        self._build_plots()
        self._build_log()
        self._build_table()
        self.splitDockWidget(self.docks["runs"], self.docks["reduction"], QtCore.Qt.Orientation.Vertical)
        self.tabifyDockWidget(self.docks["iq"], self.docks["budget"])
        self.tabifyDockWidget(self.docks["iq"], self.docks["log"])
        self.tabifyDockWidget(self.docks["iq"], self.docks["table"])
        self.docks["iq"].raise_()
        self._finish_docks()
        self.statusBar().showMessage("Choose the raw-data folder (File → Open raw-data folder)")
        self.resize(1400, 900)
        MainWindow.instances.add(self)
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
        self.table = t
        lay.addWidget(t, 1)
        self._add_dock("runs", "Runs", w, Area.LeftDockWidgetArea)

    def _build_reduction(self):
        v = prefs.values("sasred")
        w = QtWidgets.QWidget()
        lay = QtWidgets.QVBoxLayout(w)

        g = QtWidgets.QGroupBox("Beam centre (pixel indices from 0, as in GRASP and Python)")
        f = QtWidgets.QFormLayout(g)
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
        lab = QtWidgets.QLabel("beamstop guess")
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
        self.det_view.setToolTip("polar: azimuth φ horizontally (0° = horizontal direction +x, counter-clockwise), "
                                 "q vertically; rings become horizontal lines, anisotropy shows as waves")
        self.det_view.currentIndexChanged.connect(lambda _i: self._draw_detector())
        bar.addWidget(labelled("view", self.det_view))
        from .mask_editor import MaskEditor
        self.mask_editor = MaskEditor(self)                 # mask: suggest, draw, undo, load / save
        self.mask_editor.changed.connect(self._schedule_preview)
        self.mask_cfg = QtWidgets.QLabel("")
        self.mask_cfg.setWordWrap(True)
        self.mask_cfg.setToolTip("Each instrument configuration (L1, L2, λ, beam stop type and position) has its own "
                                 "mask: selecting a run of another configuration switches the mask; reductions use "
                                 "the mask of their configuration (or the suggested one if none was edited)")
        self._add_dock("detector", "Detector", plot_panel(self.det_canvas, self,
                                                          bars=(bar_w, self.mask_editor.bar, self.mask_cfg)),
                       Area.RightDockWidgetArea)
        self.iq_canvas = _canvas()
        self._add_dock("iq", "I(q)", plot_panel(self.iq_canvas, self), Area.RightDockWidgetArea)
        self.budget_canvas = _canvas()
        self._add_dock("budget", "Error budget", plot_panel(self.budget_canvas, self), Area.RightDockWidgetArea)
        self.splitDockWidget(self.docks["detector"], self.docks["iq"], QtCore.Qt.Orientation.Horizontal)

    def _build_log(self):
        self.log_view = QtWidgets.QPlainTextEdit()
        self.log_view.setReadOnly(True)
        self._add_dock("log", "Log", self.log_view, Area.RightDockWidgetArea)

    def _build_table(self):
        from .table_gui import ReductionTableWidget
        self.table_panel = ReductionTableWidget(lambda: self.entries, self.centre_for, self, log=self.log)
        self.table_panel.reduce_rows = self.reduce_table_rows
        self._add_dock("table", "Reduction table", self.table_panel, Area.RightDockWidgetArea)
        m = self.menuBar().addMenu("&Reduction")
        m.addAction("Reduction &table", self.show_table)
        m.addAction("&Fill table from runs", lambda: (self.show_table(), self.table_panel.on_fill()))
        m.addAction("Add &many samples… (select runs, then group)",
                    lambda: (self.show_table(), self.table_panel.on_add_many()))
        m.addAction("&Empty table (fill by hand)", lambda: (self.show_table(), self.table_panel.on_clear()))
        m.addSeparator()
        m.addAction("&Reduce geometry shown (2D, then I(q))", lambda: self.table_panel.page.on_reduce(all_rows=True))
        m.addAction("Reduce &selected rows", lambda: self.table_panel.page.on_reduce(all_rows=False))
        m.addSeparator()
        m.addAction("Send results to pyTOOLbox &merge (no files)", self.on_send_to_merge)

    def mask_for(self, raw):
        """The mask for a run and where it comes from: the mask of its instrument configuration
        (L1, L2, λ, beam stop type and position; :func:`catalogue.mask_key`) – the one in the editor
        or a stored one – else the suggested mask of the run (border, beam stop shadow, hot pixels)."""
        from .catalogue import mask_key, mask_key_label
        from .core import mask as mk
        k = mask_key(raw)
        ed = self.mask_editor.mask
        if k == self._mask_key and ed is not None and ed.shape == raw.counts.shape:
            return self.full_mask(raw.counts.shape), f"mask of {mask_key_label(k)} (detector panel)"
        m = self.masks.get(k)
        if m is not None and m.shape == raw.counts.shape:
            return m.copy(), f"mask of {mask_key_label(k)}"
        return mk.suggest_mask(raw.counts, self.centre_for(raw))[0], \
            f"suggested mask (no mask edited for {mask_key_label(k)})"

    def _sync_mask_config(self, raw):
        """Switch the editor to the mask of the configuration of ``raw``; the edited mask of the
        previous configuration is kept."""
        from .catalogue import mask_key, mask_key_label
        k = mask_key(raw)
        ed = self.mask_editor
        if k != self._mask_key:
            if self._mask_key is not None and ed.mask is not None and not ed.pristine:
                self.masks[self._mask_key] = ed.mask.copy()
            if self._mask_key is None and ed.mask is not None and not ed.pristine and k not in self.masks:
                self.masks[k] = ed.mask.copy()               # mask restored from an older session
            m = self.masks.get(k)
            ed._undo.clear()
            ed._redo.clear()
            if m is not None and m.shape == raw.counts.shape:
                ed.mask = m.copy()
                ed.pristine = False
            else:
                ed.mask = None
                ed.ensure(raw.counts.shape)
            ed._update_stats()
            self._mask_key = k
        n = len(set(self.masks) | ({k} if not ed.pristine else set()))
        self.mask_cfg.setText(f"mask for {mask_key_label(k)}" + ("" if not ed.pristine else " (not edited yet)")
                              + (f" – masks edited for {n} configuration{'s' if n != 1 else ''}" if n else ""))

    def _commit_mask(self):
        if self._mask_key is not None and self.mask_editor.mask is not None and not self.mask_editor.pristine:
            self.masks[self._mask_key] = self.mask_editor.mask.copy()

    def save_masks(self, path):
        """All configuration masks into one HDF5 file (one group per configuration)."""
        import h5py
        from .catalogue import mask_key_label
        self._commit_mask()
        with h5py.File(path, "w") as f:
            f.attrs["description"] = "pySASred masks per instrument configuration (L1, L2, λ, beam stop)"
            for i, (k, m) in enumerate(sorted(self.masks.items(), key=lambda kv: str(kv[0]))):
                g = f.create_group(f"mask{i:03d}")
                g.create_dataset("pixel_mask", data=np.asarray(m, np.uint32), compression="gzip")
                g.attrs["key"] = json.dumps(list(k))
                g.attrs["label"] = mask_key_label(k)
        return path

    def load_masks(self, path, replace=False):
        import h5py
        new = {}
        with h5py.File(path, "r") as f:
            for g in f.values():
                k = json.loads(g.attrs["key"])
                k[0:3] = [float(x) for x in k[0:3]]
                new[tuple(k)] = np.asarray(g["pixel_mask"][()], np.uint32)
        if replace:
            self.masks = {}
        self.masks.update(new)
        if self._mask_key in new:
            self.mask_editor.set_mask(new[self._mask_key], message="mask of the shown configuration loaded")
        return len(new)

    def on_save_masks(self):
        self._commit_mask()
        if not self.masks:
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
        from .io.sinq import read_sans_i
        paths = {e.run: e.path for e in self.entries}

        def read(n):
            if n not in paths:
                raise KeyError(f"run {n} not in the folder")
            return read_sans_i(paths[n])
        sector = (self.sector_c.value(), self.sector_hw.value()) if self.sector.isChecked() else None
        done = None
        QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.CursorShape.WaitCursor)
        try:
            for row in rows:
                tag = f"{table.config.split(',')[0]} {row.label}"
                try:
                    raw0 = read(row.used[0]) if row.used else None
                    if raw0 is None:
                        raise ValueError("no runs switched on")
                    mask, how = self.mask_for(raw0)
                    r2 = reduce_row_2d(table, row, read, self.centre_for(raw0), mask=mask,
                                       normalise_by=self.norm.currentData(), k0=table.water_g)
                    r2.notes.insert(0, how)
                    r1 = average(r2, self._edges(raw0, r2.mask), sector_deg=sector)
                except Exception as exc:  # noqa: BLE001
                    self.log(f"{tag} ({row.info}): FAILED – {type(exc).__name__}: {exc}")
                    continue
                self.results[tag] = r1
                done = tag
                self.log(f"{tag} ({row.info}): runs {row.runs_text()}, {len(r1.q)} bins, {r1.intensity.unit}")
                for n in dict.fromkeys(r1.notes):
                    self.log(f"    {n}")
                QtWidgets.QApplication.processEvents()
        finally:
            QtWidgets.QApplication.restoreOverrideCursor()
        self._refresh_results(select=done)
        if done is not None:
            self.docks["iq"].raise_()
        return done

    def show_table(self):
        """Show the reduction table panel (a tab next to I(q) by default)."""
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
        self._fill_table()
        from .catalogue import beam_centres
        self.centres = beam_centres([e.raw for e in self.entries])
        for c in sorted(self.centres.values(), key=lambda c: c.key):
            self.log("beam centre " + c.describe())
        if prefs.get("sasred.table.autofill"):
            self.table_panel.on_fill()
        bad = sum(e.raw is None for e in self.entries)
        self.statusBar().showMessage(f"{len(self.entries)} runs in {folder}" + (f", {bad} unreadable" if bad else ""),
                                     10000)

    def _fill_table(self):
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
                if e.raw is None or (e.raw is not None and e.raw.aborted):
                    it.setForeground(QtGui.QBrush(QtGui.QColor("firebrick")))
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

    def _on_selection(self):
        self._show_centre_of(self.displayed_raw())
        self._draw_detector()
        self._schedule_preview()

    # ------------------------------------------------------------------ beam centre per geometry
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
            self.bc_label.setText(f"no beam centre for L2 {raw.L2_m:.1f} m, λ {raw.wavelength_A:.2f} Å yet: "
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
        sel = [e for e in self.selected_entries() if e.raw is not None] if self.live.isChecked() else []
        if sel:
            raw = sel[0].raw
            try:
                self._sync_mask_config(raw)
                mask = self._mask(raw.counts.shape)
                sector = (self.sector_c.value(), self.sector_hw.value()) if self.sector.isChecked() else None
                self.preview = reduce_run_1d(raw, self.centre_for(raw), self._edges(raw, mask),
                                             mask=mask, normalise_by=self.norm.currentData(), sector_deg=sector)
            except Exception as exc:  # noqa: BLE001 -- e.g. beam centre outside, empty q range
                self.statusBar().showMessage(f"live I(q) of run {raw.run}: {exc}", 8000)
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
                              + f"; kept for L1 {k[2]:g} m, L2 {k[0]:g} m, λ {k[1]:g} Å")
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
        return sel[0].raw if sel else None

    def _edges(self, raw, mask):
        from .core.averaging import q_edges
        from .core.geometry import pixel_geometry
        u = self.units
        qmin, qmax = u.q_in(self.qmin.value()), u.q_in(self.qmax.value())      # internal: 1/Å
        if qmin <= 0 or qmax <= 0:
            g = pixel_geometry(raw.counts.shape, raw.pixel_size_m, self.centre_for(raw),
                               getattr(raw, "L2_effective_m", raw.L2_m), raw.wavelength_A)
            q = g.q[mask == 0]
            q = q[q > 0]
            qmin = qmin if qmin > 0 else float(q.min())
            qmax = qmax if qmax > 0 else float(q.max()) * (1 + 1e-9)
        if qmax <= qmin:
            raise ValueError(f"q max ({u.q_out(qmax):g}) must be larger than q min ({u.q_out(qmin):g}) {u.q}")
        return q_edges(qmin, qmax, self.nbins.value(), self.spacing.currentData())

    def reduce_entry(self, e):
        """Reduce one catalogue entry with the panel settings (re-read with checksum)."""
        from .io.sinq import read_sans_i
        from .reduce import reduce_run_1d
        raw = read_sans_i(e.path)
        mask = self.mask_for(raw)[0]
        sector = (self.sector_c.value(), self.sector_hw.value()) if self.sector.isChecked() else None
        r = reduce_run_1d(raw, self.centre_for(raw), self._edges(raw, mask), mask=mask,
                          normalise_by=self.norm.currentData(), sector_deg=sector)
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
                self, "Reduce", "The selected runs belong to " + ", ".join(f"L2 {k[0]:g} m / λ {k[1]:g} Å" for k in geos)
                + ". Reduce one geometry at a time: the beam centre, the solid angle and the mask belong to "
                  "the geometry.")
            return
        QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.CursorShape.WaitCursor)
        try:
            for e in sel:
                try:
                    r = self.reduce_entry(e)
                except Exception as exc:  # noqa: BLE001
                    self.log(f"run {e.run}: FAILED – {type(exc).__name__}: {exc}")
                    continue
                self.results[r.run] = r
                self.log(f"run {r.run} ({r.raw.sample_name}): {len(r.q)} bins, q = {self.units.q_out(r.q[0]):.4g} … "
                         f"{self.units.q_out(r.q[-1]):.4g} {self.units.q}, "
                         f"dead-time factor {r.deadtime_factor:.4f}")
                for n in r.notes:
                    self.log(f"    {n}")
                QtWidgets.QApplication.processEvents()
        finally:
            QtWidgets.QApplication.restoreOverrideCursor()
        self._refresh_results(select=sel[-1].run)

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
        sel = [e for e in self.selected_entries() if e.raw is not None]
        if not sel:
            self.det_canvas.draw_idle()
            return
        raw = sel[0].raw
        self._sync_mask_config(raw)
        ax = fig.add_subplot(111)
        c = raw.counts.astype(float)
        # colour range from the unmasked pixels only: bright pixels at the edge of a badly placed
        # beamstop must not dominate the colours
        valid = self.full_mask(c.shape) == 0
        norm = make_norm(self.det_scale.currentData(), c, valid=valid)
        if self.det_view.currentData() == "polar":
            self._draw_detector_polar(fig, ax, raw, c, norm)
            self.mask_editor._ax = None                     # editing only in the cartesian view
            self.mask_editor.attach()
            self.det_canvas.draw_idle()
            return
        px, py = raw.pixel_size_m
        im = ax.imshow(c, origin="lower", norm=norm, cmap=self.det_cmap.currentText(),
                       interpolation="nearest", aspect=py / px)        # true pixel aspect ratio
        from ..toolbox.viewer import image_colorbar
        decorate_colorbar(image_colorbar(fig, ax, im, "counts"), self.det_scale.currentData())
        self.mask_editor.draw_overlay(ax, c.shape, image=c, scale=self.det_scale.currentData())
        x, y = self.xc.value(), self.yc.value()
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
        ax.set_title(f"run {raw.run}: {raw.sample_name}", fontsize="medium")
        self.det_canvas.draw_idle()

    def _draw_detector_polar(self, fig, ax, raw, c, norm):
        """Detector image over (φ, q): φ = 0 horizontal (+x), counter-clockwise; masked cells in the
        mask colours, outside the detector light grey."""
        import matplotlib
        from .core.polar import polar_image
        u = self.units
        xc, yc = self.xc.value(), self.yc.value()
        pol = polar_image(c, (xc, yc), raw.pixel_size_m, getattr(raw, "L2_effective_m", raw.L2_m), raw.wavelength_A,
                          invalid=self._mask(c.shape) != 0)
        cmap = matplotlib.colormaps[self.det_cmap.currentText()].with_extremes(bad="0.85")
        qe = u.q_out(pol.q_edges)
        im = ax.pcolormesh(pol.phi_edges_deg, qe, pol.image, norm=norm, cmap=cmap, shading="flat",
                           rasterized=True)
        from ..toolbox.viewer import image_colorbar
        decorate_colorbar(image_colorbar(fig, ax, im, "counts"), self.det_scale.currentData())
        ax.set_anchor("W")
        mode = self.mask_editor.mode
        if mode == "scale":                                  # masked cells on their own grey scale
            from .mask_editor import MASKED_CMAP
            masked = self._mask(c.shape) != 0
            if masked.any():
                full = polar_image(c, (xc, yc), raw.pixel_size_m, getattr(raw, "L2_effective_m", raw.L2_m), raw.wavelength_A).image
                inv = polar_image(masked.astype(float), (xc, yc), raw.pixel_size_m, getattr(raw, "L2_effective_m", raw.L2_m),
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
            bits = polar_image(self._mask(c.shape).astype(float), (xc, yc), raw.pixel_size_m, getattr(raw, "L2_effective_m", raw.L2_m),
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

    def _draw_iq(self):
        fig = self.iq_canvas.figure
        fig.clear()
        ax = fig.add_subplot(111)
        u = self.units
        for run, r in self.results.items():
            ax.errorbar(u.q_out(r.q), r.I, yerr=r.idev(), fmt="o", ms=3, capsize=0, lw=0.8,
                        label=f"{run} {r.parameters.get('sample', r.raw.sample_name)}")
        p = self.preview
        if p is not None:
            ax.errorbar(u.q_out(p.q), p.I, yerr=p.idev(), fmt="--", color="0.15", lw=1.0, capsize=0,
                        elinewidth=0.6, label=f"live: {p.run} {p.raw.sample_name}")
        shown = list(self.results.values()) + ([p] if p is not None else [])
        if shown:
            ax.set_xscale("log")
            if all(np.all(r.I > 0) for r in shown):
                ax.set_yscale("log")
            _log_q_ticks(ax)
            ax.legend(fontsize="small")
            ax.set_ylabel(f"I [{shown[0].intensity.unit}]")
        ax.set_xlabel(f"q [{u.q}]")
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
        return {"folder": self.folder, "xc": self.xc.value(), "yc": self.yc.value(),
                "beamstop_radius_px": self.bs_radius.value(), "beamstop_shape": self.bs_shape.currentData(),
                "nbins": self.nbins.value(),
                "qmin": self.units.q_in(self.qmin.value()), "qmax": self.units.q_in(self.qmax.value()),
                "units": self.unit, "live": self.live.isChecked(), "spacing": self.spacing.currentData(),
                "normalise_by": self.norm.currentData(), "sector": self.sector.isChecked(),
                "sector_c": self.sector_c.value(), "sector_hw": self.sector_hw.value(),
                "detector_scale": self.det_scale.currentData(), "detector_cmap": self.det_cmap.currentText(),
                "detector_view": self.det_view.currentData(),
                "centres": [[c.key[0], c.key[1], c.key[2], c.x, c.y, c.source] for c in self.centres.values()
                            if c.source != "direct beam" and not c.source.startswith("borrowed")],
                "dock_state": self.dock_state()}

    @staticmethod
    def _session_mask_path():
        return os.path.join(os.path.dirname(prefs.settings_path()), "sasred_session_mask.h5")

    @staticmethod
    def _session_masks_path():
        return os.path.join(os.path.dirname(prefs.settings_path()), "sasred_session_masks.h5")

    def save_session(self):
        try:
            self._commit_mask()
            if self.masks:
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
        if isinstance(s.get("nbins"), int):
            self.nbins.setValue(s["nbins"])
        for key, combo in (("spacing", self.spacing), ("normalise_by", self.norm), ("beamstop_shape", self.bs_shape)):
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
        if s.get("folder") and os.path.isdir(s["folder"]):
            self.set_folder(s["folder"])
        from .catalogue import GeometryCentre
        for item in s.get("centres", []) or []:
            try:
                if len(item) == 5:          # sessions before the centre depended on L1: not used
                    continue
                L2, lam, L1, x, y, src = item
                k = (float(L2), float(lam), float(L1))
                self.centres[k] = GeometryCentre(k, float(x), float(y), 0.0, 0.0, [], str(src))
            except (TypeError, ValueError):
                pass
        return True

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
        super().closeEvent(event)


def main(argv=None):
    argv = sys.argv if argv is None else argv
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(argv)
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
