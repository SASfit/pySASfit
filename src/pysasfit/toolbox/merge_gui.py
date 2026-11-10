# SPDX-License-Identifier: BSD-3-Clause
"""pyTOOLbox page *Merge geometries*: read reduced curves (pySASred NXcanSAS / ASCII, BerSANS),
group them by the sample name (regular expression) and further header items with tolerances,
merge the geometries of every group (absolute, weighted mean in the overlaps), check the overlaps,
plot groups or a series (e.g. one sample at several temperatures) and save the merged curves.
"""
from __future__ import annotations

import os
import weakref
from typing import Dict, List, Optional

import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure

from ..core.docking import plot_panel
from . import merge as mg

__all__ = ["MergeTool", "send_reduced"]

_TOOLS: "weakref.WeakSet" = weakref.WeakSet()      # open merge tools (launcher tab, own windows)
_OWN_WINDOWS: list = []                              # pyTOOLbox windows opened by send_reduced


def _reveal(w):
    """Bring a widget to the front: the tabs that hold it (pyTOOLbox, launcher) and its window."""
    x = w
    while x is not None:
        p = x.parentWidget()
        if isinstance(p, QtWidgets.QStackedWidget) and isinstance(p.parentWidget(), QtWidgets.QTabWidget):
            p.parentWidget().setCurrentWidget(x)
        x = p
    win = w.window()
    win.show()
    win.raise_()
    win.activateWindow()


def send_reduced(results: Dict[object, object], source: str = "pySASred") -> "MergeTool":
    """Hand pySASred results (key -> ReducedRun) to a merge tool without writing files: the open one
    (e.g. the pyTOOLbox tab of the launcher), else a new pyTOOLbox window. Curves sent before under
    the same key are replaced."""
    tool = next(iter(_TOOLS), None)
    if tool is None:
        from .gui import MainWindow
        w = MainWindow()
        _OWN_WINDOWS.append(w)
        w.show()
        tool = w.merger
    tool.add_reduced(results, source)
    from .image_gui import _TOOLS as _IMAGE_TOOLS
    w = tool.window()
    page = next((p for p in _IMAGE_TOOLS if p.window() is w), None) or next(iter(_IMAGE_TOOLS), None)
    if page is not None:                     # the 2D images go to the 2D page (not brought to the front)
        page.add_reduced(results)
    tool.reveal()
    return tool

_ORANGE, _RED = QtGui.QColor(200, 120, 0), QtGui.QColor(200, 0, 0)
_MARKERS = "osD^v<>ph*"


def _f(x) -> str:
    return "" if x is None else f"{x:g}"


def _parse(text: str):
    t = text.strip()
    if not t:
        return None
    return float(t)



def _dq(m, sel=None):
    """q error bars (resolution dq, 1 σ) of a merged curve, or None when no point has one (dq = 0:
    unknown); kept below q for the log axis."""
    dq = np.asarray(getattr(m, "dq", None) if getattr(m, "dq", None) is not None else [], float)
    q = np.asarray(m.q, float)
    if sel is not None and len(dq):
        dq, q = dq[sel], q[sel]
    if not len(dq) or not np.any(np.isfinite(dq) & (dq > 0)):
        return None
    return np.minimum(np.where(np.isfinite(dq), dq, 0.0), 0.999 * q)

class MergeTool(QtWidgets.QWidget):
    status = QtCore.Signal(str)
    images_requested = QtCore.Signal(object)       # list of image2d.Image2D for the 2D page

    def __init__(self, parent=None):
        super().__init__(parent)
        self.curves: List[mg.Curve] = []
        self._reduced: Dict[str, object] = {}       # curve path "pySASred:<key>" -> ReducedRun (2D image)
        self.groups: List[mg.CurveGroup] = []
        self.merged: List[mg.Merged] = []
        split = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal)
        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(2, 2, 2, 2)
        lay.addWidget(split)

        # ---------------------------------------------------------------- left: files, grouping, options
        # three steps as tabs, so that each table gets the full height
        left = QtWidgets.QWidget()
        lv = QtWidgets.QVBoxLayout(left)
        lv.setContentsMargins(0, 0, 0, 0)
        self.steps = QtWidgets.QTabWidget()
        lv.addWidget(self.steps, 1)
        self.summary = QtWidgets.QLabel("")
        lv.addWidget(self.summary)
        fb = QtWidgets.QWidget()
        fl = QtWidgets.QVBoxLayout(fb)
        fl.addWidget(QtWidgets.QLabel("pySASred results in memory, pySASred NXcanSAS / ASCII files, BerSANS 1D files"))
        row = QtWidgets.QHBoxLayout()
        for text, slot in (("Add files…", self.on_add_files), ("Add folder…", self.on_add_folder),
                           ("From pySASred", self.on_from_sasred), ("Show 2D", self.on_show_2d),
                           ("Remove", self.on_remove), ("Clear", self.on_clear)):
            b = QtWidgets.QPushButton(text)
            b.clicked.connect(slot)
            row.addWidget(b)
        row.addStretch(1)
        fl.addLayout(row)
        self.files = QtWidgets.QTableWidget(0, 6)
        self.files.setHorizontalHeaderLabels(["file", "format", "sample", "geometry", "q range [1/nm]", "points"])
        self.files.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows)
        self.files.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
        self.files.horizontalHeader().setStretchLastSection(True)
        self.filter = QtWidgets.QLineEdit()
        self.filter.setPlaceholderText("filter rows: text in file, sample or geometry (regular expression)")
        self.filter.textChanged.connect(self._apply_filter)
        fl.addWidget(self.filter)
        fl.addWidget(self.files, 1)
        self.steps.addTab(fb, "1 · Curves")

        page2 = QtWidgets.QWidget()
        p2 = QtWidgets.QVBoxLayout(page2)
        gb = QtWidgets.QGroupBox("Group")
        gl = QtWidgets.QFormLayout(gb)
        self.regex = QtWidgets.QLineEdit()
        self.regex.setPlaceholderText(r"e.g. EXP-005370-(\w)_\d   (empty: the whole name)")
        self.regex.setToolTip("Regular expression on the sample name. The capture groups (…) form the key "
                              "(joined by _), without groups the whole match; names that do not match are "
                              "left out. Example: EXP-005370-(\\w)_(\\d) → A_1, A_2, …")
        gl.addRow("sample name (regex):", self.regex)
        self.items = QtWidgets.QTableWidget(0, 2)
        self.items.setHorizontalHeaderLabels(["further header item", "± tolerance"])
        self.items.horizontalHeader().setStretchLastSection(True)
        self.items.setToolTip("Groups are split by these items: text = equal values, numbers within ± "
                              "tolerance (empty: equal; '1%': relative). E.g. a temperature ± 0.5 K.")
        self.items.setMinimumHeight(90)
        self.items.setColumnWidth(0, 260)
        ib = QtWidgets.QVBoxLayout()
        add = QtWidgets.QPushButton("Add item")
        add.clicked.connect(lambda: self.add_item())
        rem = QtWidgets.QPushButton("Remove item")
        rem.clicked.connect(lambda: self.items.removeRow(self.items.currentRow()))
        ib.addWidget(add)
        ib.addWidget(rem)
        ib.addStretch(1)
        ih = QtWidgets.QHBoxLayout()
        ih.addWidget(self.items, 1)
        ih.addLayout(ib)
        gl.addRow(ih)
        p2.addWidget(gb, 1)

        ob = QtWidgets.QGroupBox("Merge")
        ol = QtWidgets.QFormLayout(ob)
        self.method = QtWidgets.QComboBox()
        self.method.addItem("keep all points (each with its dI and dq)", "all")
        self.method.addItem("weighted mean in the overlaps", "weighted")
        ol.addRow("overlaps:", self.method)
        self.scale = QtWidgets.QComboBox()
        self.scale.addItem("none – absolute scale of every geometry", None)
        self.scale.setToolTip("Optional: scale the other geometries to this one (factor from the overlaps)")
        ol.addRow("scale to:", self.scale)
        self.scale_common = QtWidgets.QCheckBox("one factor per geometry for all groups")
        self.scale_common.setChecked(True)
        self.scale_common.setToolTip("Checked: every other geometry is multiplied by one factor, the same for all "
                                     "merged curves (weighted mean of the overlap factors of all groups).\n"
                                     "Unchecked: every group is scaled with the factors of its own overlaps.")
        ol.addRow("", self.scale_common)
        self.cuts = QtWidgets.QTableWidget(0, 5)
        self.cuts.setHorizontalHeaderLabels(["geometry", "q min [1/nm]", "q max [1/nm]", "drop first", "drop last"])
        self.cuts.horizontalHeader().setStretchLastSection(True)
        self.cuts.setToolTip("q range and points dropped at the ends, per geometry (for all groups)")
        self.cuts.setMinimumHeight(90)
        ol.addRow(self.cuts)
        b = QtWidgets.QPushButton("Group and merge")
        b.setStyleSheet("font-weight: bold")
        b.clicked.connect(self.on_merge)
        p2.addWidget(ob, 1)
        p2.addWidget(b)
        from ..core import pycode
        p2.addWidget(pycode.button(self.python_code, self))
        self.steps.addTab(page2, "2 · Group and merge")

        rb = QtWidgets.QWidget()
        rl = QtWidgets.QVBoxLayout(rb)
        self.table = QtWidgets.QTableWidget(0, 6)
        self.table.setHorizontalHeaderLabels(["group", "sample", "curves", "geometries", "overlap mismatch", "notes"])
        self.table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.itemSelectionChanged.connect(self.replot)
        self.table.itemSelectionChanged.connect(self._show_sources)
        rl.addWidget(self.table, 3)
        rl.addWidget(QtWidgets.QLabel("Curves of the selected group (the underlying data):"))
        self.sources = QtWidgets.QTableWidget(0, 6)
        self.sources.setHorizontalHeaderLabels(["geometry", "sample", "source", "runs", "q range [1/nm]", "points"])
        self.sources.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows)
        self.sources.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
        self.sources.horizontalHeader().setStretchLastSection(True)
        self.sources.setToolTip("Double-click: show this curve in pySASred (reduction-table row or run) or, for a "
                                "file, its folder in the curve list")
        self.sources.itemDoubleClicked.connect(lambda _it: self.on_show_source())
        rl.addWidget(self.sources, 2)
        srow = QtWidgets.QHBoxLayout()
        bsrc = QtWidgets.QPushButton("Show in pySASred")
        bsrc.setToolTip("select the reduction-table row (or run) of the selected curve in pySASred")
        bsrc.clicked.connect(self.on_show_source)
        b2d = QtWidgets.QPushButton("Show 2D")
        b2d.clicked.connect(lambda: self._emit_images(self._source_curves(selected_only=True)))
        srow.addWidget(bsrc)
        srow.addWidget(b2d)
        srow.addStretch(1)
        rl.addLayout(srow)
        sv = QtWidgets.QHBoxLayout()
        self.out_ascii = QtWidgets.QCheckBox("ASCII per curve")
        self.out_ascii.setChecked(True)
        self.out_nx = QtWidgets.QCheckBox("NXcanSAS per curve")
        self.out_nx.setChecked(True)
        self.out_all = QtWidgets.QCheckBox("one HDF5 with all")
        self.units = QtWidgets.QComboBox()
        self.units.addItem("q in 1/nm", "nm")
        self.units.addItem("q in 1/Å", "A")
        for w in (self.out_ascii, self.out_nx, self.out_all, self.units):
            sv.addWidget(w)
        rl.addLayout(sv)
        sb = QtWidgets.QHBoxLayout()
        b1 = QtWidgets.QPushButton("Save selected…")
        b1.clicked.connect(lambda: self.on_save(selected=True))
        b2 = QtWidgets.QPushButton("Save all…")
        b2.clicked.connect(lambda: self.on_save(selected=False))
        sb.addWidget(b1)
        sb.addWidget(b2)
        sb.addStretch(1)
        rl.addLayout(sb)
        self.steps.addTab(rb, "3 · Merged groups")
        split.addWidget(left)

        # ---------------------------------------------------------------- right: plot
        self.canvas = FigureCanvasQTAgg(Figure(figsize=(6, 5), layout="constrained"))
        bar = QtWidgets.QHBoxLayout()
        self.mode = QtWidgets.QComboBox()
        self.mode.addItem("selected groups: curves and merged", "groups")
        self.mode.addItem("series of the selected sample", "series")
        self.mode.currentIndexChanged.connect(self.replot)
        self.series_key = QtWidgets.QComboBox()
        self.series_key.setToolTip("Header item that orders and colours the series (e.g. temperature)")
        self.series_key.currentIndexChanged.connect(self.replot)
        self.offset = QtWidgets.QDoubleSpinBox()
        self.offset.setRange(1.0, 1000.0)
        self.offset.setValue(1.0)
        self.offset.setDecimals(2)
        self.offset.setPrefix("offset × ")
        self.offset.setToolTip("Successive curves multiplied by this factor (1: none)")
        self.offset.valueChanged.connect(self.replot)
        self.show_src = QtWidgets.QCheckBox("geometries")
        self.show_src.setChecked(True)
        self.show_src.toggled.connect(self.replot)
        self.err_I = QtWidgets.QCheckBox("I error bars")
        self.err_q = QtWidgets.QCheckBox("q resolution bars")
        self.err_q.setToolTip("horizontal bars: dq (1 σ) of the curves that carry a resolution")
        for cb in (self.err_I, self.err_q):
            cb.setChecked(True)
            cb.toggled.connect(self.replot)
        from ..core.flowlayout import flow_bar
        bar.addWidget(flow_bar(self.mode, "by:", self.series_key, self.offset, self.show_src, self.err_I,
                               self.err_q), 1)
        split.addWidget(plot_panel(self.canvas, self, bars=[bar]))
        self.log = QtWidgets.QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumBlockCount(2000)
        right_split = split.widget(1)
        right_split.layout().addWidget(self.log)
        self.log.setMaximumHeight(140)
        split.setStretchFactor(0, 2)
        split.setStretchFactor(1, 3)

        _TOOLS.add(self)

    # ------------------------------------------------------------------ files
    def reveal(self):
        """Bring this page to the front (its tab in pyTOOLbox / the launcher, its window)."""
        w = self
        while w is not None:
            p = w.parentWidget()
            if isinstance(p, QtWidgets.QStackedWidget) and isinstance(p.parentWidget(), QtWidgets.QTabWidget):
                p.parentWidget().setCurrentWidget(w)
            w = p
        win = self.window()
        win.show()
        win.raise_()
        win.activateWindow()

    def add_reduced(self, results: Dict[object, object], source: str = "pySASred"):
        """Add pySASred results in memory (key -> ReducedRun); same key again replaces the curve."""
        new = []
        for k, r in results.items():
            try:
                new.append(mg.curve_from_reduced(r, k))
            except Exception as exc:  # noqa: BLE001
                self._log(f"{source} {k}: not taken – {exc}")
        paths = {c.path for c in new}
        self.curves = [c for c in self.curves if c.path not in paths] + new
        for k, r in results.items():
            self._reduced[f"pySASred:{k}"] = r
        self._log(f"{len(new)} curves from {source} (in memory)")
        self._refresh_files()

    def on_from_sasred(self):
        """Take the reduced results of every open pySASred window."""
        try:
            from ..sasred.gui import MainWindow as Red
        except Exception as exc:  # noqa: BLE001  (pySASred extras not installed)
            self._log(f"pySASred not available: {exc}")
            return
        wins = [w for w in list(Red.instances) if getattr(w, "results", None)]
        if not wins:
            self._log("no reduced data in pySASred (reduce runs or table rows first)")
            return
        for w in wins:
            self.add_reduced(w.results, f"pySASred ({os.path.basename(w.folder) or 'no folder'})")

    def _log(self, t):
        self.log.appendPlainText(t)
        self.status.emit(t)

    def add_paths(self, paths):
        bad = []
        for p in paths:
            try:
                self.curves += mg.read_curves(p)
            except Exception as exc:  # noqa: BLE001
                bad.append(f"{os.path.basename(p)}: {exc}")
        for b in bad:
            self._log("not read: " + b)
        self._refresh_files()

    def add_folder(self, folder):
        cs, bad = mg.read_folder(folder)
        self.curves += cs
        for b in bad:
            self._log("not read: " + b)
        self._log(f"{len(cs)} curves from {folder}")
        self._refresh_files()

    def on_add_files(self):
        ps, _ = QtWidgets.QFileDialog.getOpenFileNames(
            self, "Reduced curves", "", "reduced data (*.h5 *.nxs *.dat *.txt *.0*);;all files (*)")
        if ps:
            self.add_paths(ps)

    def on_add_folder(self):
        d = QtWidgets.QFileDialog.getExistingDirectory(self, "Folder with reduced curves")
        if d:
            self.add_folder(d)

    def images_for(self, curves) -> list:
        """2D images belonging to curves: pySASred results in memory, else ``<name>_2d.h5`` next to
        the file of the curve."""
        from . import image2d as i2
        out, missing = [], []
        for c in curves:
            r = self._reduced.get(c.path)
            try:
                if r is not None:
                    out.append(i2.image_from_reduced(r, c.path.split(":", 1)[1]))
                    continue
                cand = os.path.splitext(c.path)[0] + "_2d.h5"
                if os.path.isfile(cand):
                    out += i2.read_images(cand)
                else:
                    missing.append(c.name)
            except Exception as exc:  # noqa: BLE001
                missing.append(f"{c.name} ({exc})")
        if missing:
            self._log("no 2D image for: " + ", ".join(missing[:8]) + (" …" if len(missing) > 8 else ""))
        return out

    # ------------------------------------------------------------------ underlying data of a group
    def _source_curves(self, selected_only=False):
        sel = self.selected()
        if not sel:
            return []
        cs = list(sel[0].group.curves)
        if selected_only:
            rows = sorted({i.row() for i in self.sources.selectedIndexes()})
            if rows:
                cs = [cs[r] for r in rows if r < len(cs)]
        return cs

    def _show_sources(self):
        cs = self._source_curves()
        t = self.sources
        t.setRowCount(len(cs))
        for i, c in enumerate(cs):
            lo, hi = c.q_range()
            src = (f"pySASred: {c.path.split(':', 1)[1]}" if c.path.startswith("pySASred:")
                   else c.path + (f" : {c.entry}" if c.entry else ""))
            runs = str(c.meta.get("runs") or c.meta.get("run") or "")
            for j, v in enumerate((c.geometry, c.sample, src, runs, f"{lo:.4g} – {hi:.4g}", str(len(c.q)))):
                it = QtWidgets.QTableWidgetItem(v)
                it.setToolTip(v if j != 2 else c.path)
                t.setItem(i, j, it)
        t.resizeColumnsToContents()

    def _emit_images(self, curves):
        imgs = self.images_for(curves)
        if imgs:
            self.images_requested.emit(imgs)

    def on_show_source(self):
        """Select the reduction-table row (or run) of the selected curve in an open pySASred window."""
        cs = self._source_curves(selected_only=True)
        if not cs:
            return
        c = cs[0]
        if not c.path.startswith("pySASred:"):
            self._log(f"{c.name}: file {c.path}")
            return
        key = c.path.split(":", 1)[1]
        try:
            from ..sasred.gui import MainWindow as Red
        except Exception as exc:  # noqa: BLE001
            self._log(f"pySASred not available: {exc}")
            return
        for w in list(Red.instances):
            res = getattr(w, "results", {})
            k = int(key) if key.isdigit() else key
            if k not in res:
                continue
            if isinstance(k, int):
                w.select_runs([k])
            else:
                panel = w.table_panel
                for i, page in enumerate(panel.pages()):
                    pre = page.table.config.split(",")[0]
                    for r, row in enumerate(page.table.samples):
                        if f"{pre} {row.label}" == key:
                            panel.tabs.setCurrentIndex(i)
                            w.show_table() if hasattr(w, "show_table") else None
                            page.samp.selectRow(r)
            _reveal(w)
            self._log(f"{key} shown in pySASred")
            return
        self._log(f"{key}: not (any more) in an open pySASred window")

    def on_show_2d(self):
        rows = sorted({i.row() for i in self.files.selectedIndexes()})
        imgs = self.images_for([self.curves[r] for r in rows if r < len(self.curves)])
        if imgs:
            self.images_requested.emit(imgs)

    def on_remove(self):
        rows = sorted({i.row() for i in self.files.selectedIndexes()}, reverse=True)
        for r in rows:
            del self.curves[r]
        self._refresh_files()

    def on_clear(self):
        self.curves = []
        self._refresh_files()

    def _apply_filter(self, text=None):
        import re as _re
        text = self.filter.text().strip() if text is None else text.strip()
        try:
            rx = _re.compile(text, _re.I) if text else None
        except _re.error:
            rx = None
        for i, c in enumerate(self.curves):
            hide = rx is not None and not any(rx.search(v or "") for v in (c.name, c.sample, c.geometry))
            self.files.setRowHidden(i, hide)

    def _update_summary(self):
        geos = {c.geometry for c in self.curves}
        s = f"{len(self.curves)} curves, {len(geos)} geometr{'y' if len(geos) == 1 else 'ies'}"
        if self.merged:
            bad = sum(1 for m in self.merged if any(abs(o.factor - 1) > 0.05 for o in m.overlaps))
            s += f"; {len(self.merged)} groups merged" + (f", {bad} with overlap mismatch > 5 %" if bad else "")
        self.summary.setText(s)
        self.steps.setTabText(0, f"1 · Curves ({len(self.curves)})")
        self.steps.setTabText(2, f"3 · Merged groups ({len(self.merged)})")

    def _refresh_files(self):
        t = self.files
        t.setRowCount(len(self.curves))
        for i, c in enumerate(self.curves):
            lo, hi = c.q_range()
            for j, v in enumerate((c.name, c.fmt, c.sample, c.geometry, f"{lo:.4g} – {hi:.4g}", str(len(c.q)))):
                it = QtWidgets.QTableWidgetItem(v)
                it.setToolTip(c.path)
                t.setItem(i, j, it)
        t.resizeColumnsToContents()
        self._apply_filter()
        self._update_summary()
        fields = mg.fields_of(self.curves)
        for r in range(self.items.rowCount()):
            cb = self.items.cellWidget(r, 0)
            cur = cb.currentText()
            cb.clear()
            cb.addItems(fields)
            cb.setCurrentText(cur)
        cur = self.series_key.currentText()
        self.series_key.blockSignals(True)
        self.series_key.clear()
        self.series_key.addItems([f for f in fields if f not in ("sample", "file", "format", "geometry")])
        if cur:
            self.series_key.setCurrentText(cur)
        self.series_key.blockSignals(False)
        geos = list(dict.fromkeys(c.geometry for c in sorted(self.curves, key=lambda c: c.q_range()[0])))
        have = {self.cuts.item(r, 0).text(): r for r in range(self.cuts.rowCount())}
        for g in geos:
            if g not in have:
                r = self.cuts.rowCount()
                self.cuts.insertRow(r)
                it = QtWidgets.QTableWidgetItem(g)
                it.setFlags(QtCore.Qt.ItemFlag.ItemIsEnabled)
                self.cuts.setItem(r, 0, it)
                for j in range(1, 5):
                    self.cuts.setItem(r, j, QtWidgets.QTableWidgetItem(""))
        self.cuts.resizeColumnsToContents()
        cur = self.scale.currentData()
        self.scale.clear()
        self.scale.addItem("none – absolute scale of every geometry", None)
        for g in geos:
            self.scale.addItem(g, g)
        i = self.scale.findData(cur)
        self.scale.setCurrentIndex(max(0, i))

    # ------------------------------------------------------------------ grouping
    def add_item(self, field: str = "", tol: str = ""):
        r = self.items.rowCount()
        self.items.insertRow(r)
        cb = QtWidgets.QComboBox()
        cb.setEditable(True)
        cb.addItems(mg.fields_of(self.curves))
        cb.setCurrentText(field)
        self.items.setCellWidget(r, 0, cb)
        self.items.setItem(r, 1, QtWidgets.QTableWidgetItem(tol))

    def grouping(self) -> Dict[str, object]:
        out = {}
        for r in range(self.items.rowCount()):
            f = self.items.cellWidget(r, 0).currentText().strip()
            if not f:
                continue
            t = (self.items.item(r, 1).text() if self.items.item(r, 1) else "").strip()
            out[f] = None if not t else (t if t.endswith("%") else float(t))
        return out

    def cut_settings(self):
        cuts, drop = {}, {}
        for r in range(self.cuts.rowCount()):
            g = self.cuts.item(r, 0).text()
            vals = [(self.cuts.item(r, j).text() if self.cuts.item(r, j) else "") for j in range(1, 5)]
            lo, hi = _parse(vals[0]), _parse(vals[1])
            if lo is not None or hi is not None:
                cuts[g] = (lo, hi)
            a, b = int(_parse(vals[2]) or 0), int(_parse(vals[3]) or 0)
            if a or b:
                drop[g] = (a, b)
        return cuts, drop

    def on_merge(self):
        try:
            fields = self.grouping()
            cuts, drop = self.cut_settings()
        except ValueError as exc:
            self._log(f"not a number: {exc}")
            return
        regex = self.regex.text().strip() or None
        try:
            self.groups, rest = mg.group_curves(self.curves, regex, fields)
        except Exception as exc:  # noqa: BLE001  (bad regular expression, unknown item)
            self._log(f"grouping failed: {exc}")
            return
        if rest:
            self._log(f"{len(rest)} curves do not match {regex!r}: " + ", ".join(c.sample for c in rest[:10]))
        self.merged = mg.merge_groups(self.groups, cuts=cuts, drop=drop, method=self.method.currentData(),
                                      scale_to=self.scale.currentData(), common=self.scale_common.isChecked())
        if self.scale.currentData() and self.scale_common.isChecked() and self.merged and self.merged[0].notes:
            self._log(next((n for n in self.merged[0].notes if n.startswith("one scale factor")), ""))
        t = self.table
        t.setRowCount(len(self.merged))
        for i, m in enumerate(self.merged):
            g = m.group
            worst = max(m.overlaps, key=lambda o: abs(o.factor - 1), default=None)
            mis = "" if worst is None else f"{100 * (worst.factor - 1):+.1f} % (z {worst.z:+.1f})"
            # geometries: only the parts that differ between them (e.g. "SD 8.00 m | SD 2.00 m")
            parts = [x.split(", ") for x in m.geometries]
            keep = [i for i in range(max(map(len, parts), default=0))
                    if len({p[i] if i < len(p) else "" for p in parts}) > 1] or [0]
            geo_short = " | ".join(", ".join(p[i] for i in keep if i < len(p)) for p in parts)
            samples = sorted({c.sample for c in g.curves})
            vals = (g.label, ", ".join(samples), str(len(g.curves)), geo_short, mis, "; ".join(m.notes))
            for j, v in enumerate(vals):
                it = QtWidgets.QTableWidgetItem(v)
                if j == 4 and worst is not None:
                    it.setToolTip("\n".join(o.text() for o in m.overlaps))
                    if abs(worst.factor - 1) > 0.05:
                        it.setForeground(QtGui.QBrush(_RED))
                    elif abs(worst.factor - 1) > 0.02:
                        it.setForeground(QtGui.QBrush(_ORANGE))
                if j == 3:
                    it.setToolTip("\n".join(m.geometries))
                if j == 1:
                    it.setToolTip("\n".join(samples))
                if j == 5:
                    it.setToolTip("\n".join(m.notes))
                t.setItem(i, j, it)
        t.resizeColumnsToContents()
        t.setColumnWidth(0, min(t.columnWidth(0), 320))
        t.setColumnWidth(2, min(t.columnWidth(2), 360))
        self._log(f"{len(self.merged)} groups merged ({self.method.currentText()})")
        self._update_summary()
        self.steps.setCurrentIndex(2)
        if self.merged:
            t.selectRow(0)
        self.replot()

    def python_code(self):
        """Script: read the curve files, group them (sample expression, header items) and merge them with the cuts,
        method and scaling of the panel ("Copy Python code")."""
        from ..core.pycode import Code, Script, call, lit

        if not self.curves:
            raise ValueError("No curves yet: add reduced curves first.")
        fields = self.grouping()
        cuts, drop = self.cut_settings()
        sc = Script("pyTOOLbox", "merge of the detector distances (geometries)")
        sc.imp("import numpy as np", "from pysasfit.toolbox import merge as mg")
        files = list(dict.fromkeys(c.path for c in self.curves if os.path.isfile(c.path)))
        memory = [c for c in self.curves if not os.path.isfile(c.path)]
        sc.comment("the reduced curves (NXcanSAS, pySASred ASCII, BerSANS, columns)")
        sc.add("curves = []", "for path in " + lit(files, 4) + ":", "    curves += mg.read_curves(path)")
        if memory:
            sc.comment(f"{len(memory)} curves taken from pySASred in memory (not saved as files): their values")
            for c in memory:
                sc.add(call("curves.append(mg.Curve", c.q, c.I, c.dI, c.dq, c.path, c.fmt, meta=dict(c.meta),
                            entry=c.entry, I_unit=c.I_unit) + ")")
        sc.comment("groups: same sample (regular expression) and header items within their tolerance")
        sc.add(call("groups, rest = mg.group_curves", Code("curves"), self.regex.text().strip() or None, fields),
               "if rest:", "    print(len(rest), 'curves do not match the sample expression')")
        sc.comment("merge: q cuts and dropped points per geometry, method, scaling")
        sc.add(call("merged = mg.merge_groups", Code("groups"), cuts=cuts, drop=drop, method=self.method.currentData(),
                    scale_to=self.scale.currentData(), common=self.scale_common.isChecked()),
               "for m in merged:",
               "    worst = max(m.overlaps, key=lambda o: abs(o.factor - 1), default=None)",
               "    print(m.group.label, len(m.group.curves), 'curves',",
               "          '' if worst is None else f'overlap mismatch {100 * (worst.factor - 1):+.1f} %')",
               "# save: mg.write_nxcansas('merged.h5', merged) or mg.write_ascii(mg.file_stem(m) + '.dat', m)")
        sc.add_plot("fig, ax = plt.subplots()",
                    "for m in merged:",
                    "    if len(m.q):",
                    "        ax.errorbar(m.q, m.I, m.dI, fmt='-', lw=1, label=m.group.label)",
                    "ax.set_xscale('log')", "ax.set_yscale('log')", "ax.set_xlabel('q [1/nm]')", "ax.set_ylabel('I')",
                    "ax.legend(fontsize=7)")
        return sc.text()

    # ------------------------------------------------------------------ plot
    def selected(self) -> List[mg.Merged]:
        rows = sorted({i.row() for i in self.table.selectedIndexes()})
        return [self.merged[r] for r in rows if r < len(self.merged)]

    def _dI(self, dI):
        return dI if self.err_I.isChecked() else None

    def _dq(self, m, sel=None):
        return _dq(m, sel) if self.err_q.isChecked() else None

    def replot(self):
        fig = self.canvas.figure
        fig.clear()
        ax = fig.add_subplot(111)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("q [1/nm]")
        sel = self.selected()
        unit = next((m.I_unit for m in sel if m.I_unit), "")
        ax.set_ylabel(f"I [{unit}]" if unit else "I")
        off = self.offset.value()
        if self.mode.currentData() == "series" and sel:
            key = self.series_key.currentText()
            ser = mg.series(self.merged, key).get(sel[0].group.key, [])
            vals = [v for v, _ in ser if v is not None]
            import matplotlib as mpl
            norm = mpl.colors.Normalize(min(vals), max(vals)) if len(vals) > 1 else None
            cmap = mpl.colormaps["viridis"]
            for k, (v, m) in enumerate(ser):
                if not len(m.q):
                    continue
                col = cmap(norm(v)) if (norm and v is not None) else None
                f = off ** k
                ax.errorbar(m.q, m.I * f, self._dI(m.dI * f), self._dq(m), fmt="-", lw=1, color=col, elinewidth=0.5,
                            label=f"{mg._short(key)} = {v:.4g}" if v is not None else m.group.label)
            if norm:
                sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
                fig.colorbar(sm, ax=ax, label=mg._short(key))
            ax.set_title(f"{sel[0].group.key}: {len(ser)} curves by {mg._short(key)}", fontsize=9)
        else:
            import matplotlib as mpl
            cols = mpl.rcParams["axes.prop_cycle"].by_key()["color"]
            for k, m in enumerate(sel):
                col = cols[k % len(cols)]
                f = off ** k
                if self.show_src.isChecked():
                    geos = m.geometries
                    for c in m.group.curves:
                        gi = geos.index(c.geometry) if c.geometry in geos else 0
                        ok = np.isfinite(c.I) & (c.I > 0)
                        ax.plot(c.q[ok], c.I[ok] * f * m.scales.get(c.geometry, 1.0),
                                _MARKERS[gi % len(_MARKERS)], ms=3, mfc="none", color=col, alpha=0.5,
                                label=f"{c.geometry}" if k == 0 else None)
                lab = (f"{', '.join(sorted({c.sample for c in m.group.curves}))} [{m.group.label}]"
                       + (f" (× {f:g})" if f != 1 else ""))
                if len(m.q) and m.method == "all" and len(m.geometries) > 1:
                    # all points kept: one marker per geometry, no line through interleaved points
                    for gi in np.unique(m.source):
                        s = m.source == gi
                        ax.errorbar(m.q[s], m.I[s] * f, self._dI(m.dI[s] * f), self._dq(m, s), fmt=_MARKERS[int(gi) % len(_MARKERS)],
                                    ms=3, color=col, elinewidth=0.6, label=lab if gi == m.source.min() else None)
                elif len(m.q):
                    ax.errorbar(m.q, m.I * f, self._dI(m.dI * f), self._dq(m), fmt="-", lw=1.2, color=col, elinewidth=0.6, label=lab)
                for o in m.overlaps:
                    ax.axvspan(*o.q_range, color=col, alpha=0.06)
            if len(sel) == 1:
                def short(g):
                    return g.split(",")[0]
                ax.set_title(f"{sel[0].group.label}: " + "; ".join(
                    f"{short(o.high)} / {short(o.low)} = {o.factor:.3f} ± {o.sigma:.3f} (z {o.z:+.1f})"
                    for o in sel[0].overlaps), fontsize=8)
        if ax.has_data():
            ax.legend(fontsize=7, loc="best")
        self.canvas.draw_idle()

    # ------------------------------------------------------------------ save
    def on_save(self, selected=True, folder: Optional[str] = None, all_file: Optional[str] = None) -> List[str]:
        ms = self.selected() if selected else self.merged
        ms = [m for m in ms if len(m.q)]
        if not ms:
            self._log("nothing to save")
            return []
        if folder is None:
            folder = QtWidgets.QFileDialog.getExistingDirectory(self, "Folder for the merged curves")
            if not folder:
                return []
        u = self.units.currentData()
        out = []
        for m in ms:
            stem = os.path.join(folder, mg.file_stem(m))
            if self.out_ascii.isChecked():
                out.append(mg.write_ascii(stem + ".dat", m, u))
            if self.out_nx.isChecked():
                out.append(mg.write_nxcansas(stem + ".h5", m, u))
        if self.out_all.isChecked():
            p = all_file or os.path.join(folder, "merged_all.h5")
            out.append(mg.write_nxcansas(p, ms, u))
        self._log(f"saved {len(out)} files to {folder}")
        return out

    # ------------------------------------------------------------------ session
    def state(self) -> dict:
        return dict(regex=self.regex.text(), items=[(f, "" if t is None else str(t)) for f, t in self.grouping().items()],
                    overlap_method=self.method.currentData(), units=self.units.currentData(),
                    scale_common=self.scale_common.isChecked(),
                    outputs=[self.out_ascii.isChecked(), self.out_nx.isChecked(), self.out_all.isChecked()])

    def restore_state(self, s: dict):
        if not s:
            return
        self.regex.setText(s.get("regex", ""))
        self.items.setRowCount(0)
        for f, t in s.get("items", []):
            self.add_item(f, t)
        self.scale_common.setChecked(bool(s.get("scale_common", True)))
        i = self.method.findData(s.get("overlap_method"))
        if i >= 0:
            self.method.setCurrentIndex(i)
        i = self.units.findData(s.get("units"))
        if i >= 0:
            self.units.setCurrentIndex(i)
        o = s.get("outputs")
        if o and len(o) == 3:
            self.out_ascii.setChecked(o[0])
            self.out_nx.setChecked(o[1])
            self.out_all.setChecked(o[2])
