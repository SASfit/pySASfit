# SPDX-License-Identifier: BSD-3-Clause
"""pyTOOLbox page *1D data*: curves from files, pySASred (in memory), merged groups and the 2D
analyses (sector / azimuthal / box averages, A(q) and B(q), rocking curves); plot them and compute with
them: × a, ÷ a, + c, cut an x range, and a + b, a − b, a × b, a / b, a − f·b of two data sets (b
interpolated at the x of a). Results are new data sets that can be saved (ASCII with the labels and units,
read again here)."""
from __future__ import annotations

import os
import re
from typing import List, Optional

import numpy as np
from PySide6 import QtCore, QtWidgets
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure

from ..core.docking import plot_panel
from . import data1d as d1

__all__ = ["Data1DTool"]


def _num(text: str) -> Optional[float]:
    try:
        return float(text.replace(",", "."))
    except ValueError:
        return None


class Data1DTool(QtWidgets.QWidget):
    status = QtCore.Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.data: List[d1.Data1D] = []
        self.merge_tool = None                       # set by the pyTOOLbox window (From merge)
        split = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal)
        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(2, 2, 2, 2)
        lay.addWidget(split)

        left = QtWidgets.QWidget()
        lv = QtWidgets.QVBoxLayout(left)
        lv.setContentsMargins(0, 0, 0, 0)
        row = QtWidgets.QHBoxLayout()
        for text, slot in (("Add files…", self.on_add_files), ("From pySASred", self.on_from_sasred),
                           ("From merge", self.on_from_merge), ("Remove", self.on_remove), ("Clear", self.on_clear),
                           ("Save selected…", self.on_save)):
            b = QtWidgets.QPushButton(text)
            b.clicked.connect(slot)
            row.addWidget(b)
        from ..core import pycode
        row.addWidget(pycode.button(self.python_code, self))
        row.addStretch(1)
        lv.addLayout(row)
        self.table = QtWidgets.QTableWidget(0, 5)
        self.table.setHorizontalHeaderLabels(["#", "name", "x", "points", "source"])
        self.table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.DoubleClicked)
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.verticalHeader().setVisible(False)
        self.table.itemSelectionChanged.connect(self.replot)
        self.table.itemChanged.connect(self._on_rename)
        self.table.setToolTip("select data sets to plot them; double-click a name to rename")
        lv.addWidget(self.table, 3)

        ob = QtWidgets.QGroupBox("Operations (on the selected data sets)")
        ol = QtWidgets.QGridLayout(ob)
        self.value = QtWidgets.QLineEdit("1")
        self.value.setMaximumWidth(110)
        ol.addWidget(QtWidgets.QLabel("number:"), 0, 0)
        ol.addWidget(self.value, 0, 1)
        for j, (text, op) in enumerate((("× number", "scale"), ("÷ number", "divide"), ("+ number", "offset"),
                                        ("− number", "minus"))):
            b = QtWidgets.QPushButton(text)
            b.clicked.connect(lambda _c=False, o=op: self.on_scalar(o))
            ol.addWidget(b, 0, 2 + j)
        self.xmin, self.xmax = QtWidgets.QLineEdit(), QtWidgets.QLineEdit()
        for w, t in ((self.xmin, "x min"), (self.xmax, "x max")):
            w.setPlaceholderText(t)
            w.setMaximumWidth(110)
        bc = QtWidgets.QPushButton("Cut x range")
        bc.clicked.connect(self.on_cut)
        ol.addWidget(QtWidgets.QLabel("x range:"), 1, 0)
        ol.addWidget(self.xmin, 1, 1)
        ol.addWidget(self.xmax, 1, 2)
        ol.addWidget(bc, 1, 3)
        self.a_combo, self.b_combo = QtWidgets.QComboBox(), QtWidgets.QComboBox()
        self.op = QtWidgets.QComboBox()
        for t, o in (("a − b", "-"), ("a + b", "+"), ("a / b (ratio)", "/"), ("a × b", "*"), ("a − f · b", "a-f*b")):
            self.op.addItem(t, o)
        self.f = QtWidgets.QLineEdit("1")
        self.f.setMaximumWidth(70)
        self.f.setToolTip("factor f of a − f·b (e.g. a transmission or a fraction)")
        bb = QtWidgets.QPushButton("Compute")
        bb.clicked.connect(self.on_binary)
        ol.addWidget(QtWidgets.QLabel("a:"), 2, 0)
        ol.addWidget(self.a_combo, 2, 1, 1, 3)
        ol.addWidget(QtWidgets.QLabel("b:"), 3, 0)
        ol.addWidget(self.b_combo, 3, 1, 1, 3)
        ol.addWidget(self.op, 4, 1)
        ol.addWidget(QtWidgets.QLabel("f:"), 4, 2)
        ol.addWidget(self.f, 4, 3)
        ol.addWidget(bb, 4, 4)
        lv.addWidget(ob)
        split.addWidget(left)

        self.canvas = FigureCanvasQTAgg(Figure(figsize=(6, 5), layout="constrained"))
        bar = QtWidgets.QHBoxLayout()
        self.xscale, self.yscale = QtWidgets.QComboBox(), QtWidgets.QComboBox()
        for c in (self.xscale, self.yscale):
            c.addItem("auto", "auto")
            c.addItem("log", "log")
            c.addItem("linear", "linear")
            c.currentIndexChanged.connect(self.replot)
        self.offset = QtWidgets.QDoubleSpinBox()
        self.offset.setRange(1.0, 1000.0)
        self.offset.setDecimals(1)
        self.offset.setPrefix("offset × ")
        self.offset.valueChanged.connect(self.replot)
        bar.addWidget(QtWidgets.QLabel("x"))
        bar.addWidget(self.xscale)
        bar.addWidget(QtWidgets.QLabel("y"))
        bar.addWidget(self.yscale)
        bar.addWidget(self.offset)
        bar.addStretch(1)
        split.addWidget(plot_panel(self.canvas, self, bars=[bar]))
        split.setStretchFactor(0, 2)
        split.setStretchFactor(1, 3)

    # ------------------------------------------------------------------ data sets
    def _log(self, t):
        self.status.emit(t)

    def _recipe_of(self, d, expr, deps=()):
        """Remember how ``d`` was made (for "Copy Python code"): ``expr`` with {0}, {1} for the data sets ``deps``."""
        if not hasattr(self, "_recipes"):
            self._recipes = {}
        self._recipes[id(d)] = (expr, list(deps), d)
        return d

    def python_code(self):
        """Script that makes the selected data sets (all when none is selected) the way they were made here:
        files read, data from pySASred / merge as values, the operations in order ("Copy Python code")."""
        from ..core.pycode import Script, call, lit

        targets = self.selected() or list(self.data)
        if not targets:
            raise ValueError("No 1D data yet.")
        recipes = getattr(self, "_recipes", {})
        sc = Script("pyTOOLbox", "1D data arithmetic")
        sc.imp("import numpy as np", "from pysasfit.toolbox import data1d as d1")
        names = {}

        def emit(d):
            if id(d) in names:
                return names[id(d)]
            r = recipes.get(id(d))
            if r is not None and r[2] is d:
                args = [emit(x) for x in r[1]]
                expr = r[0].format(*args)
            else:                                   # from pySASred / merge / 2D: the values
                expr = call("d1.Data1D", d.x, d.y, d.dy, d.dx, name=d.name, xlabel=d.xlabel, xunit=d.xunit,
                            yunit=d.yunit, source=d.source)
            v = f"ds{len(names) + 1}"
            names[id(d)] = v
            sc.add(f"{v} = {expr}   # {d.name}" if "\n" not in expr else f"{v} = {expr}")
            return v

        for d in targets:
            emit(d)
        out = ", ".join(names[id(d)] for d in targets)
        sc.add(f"results = [{out}]")
        sc.add_plot("fig, ax = plt.subplots()", "for d in results:",
                    "    ax.errorbar(d.x, d.y, d.dy, fmt='.-', ms=3, label=d.name)",
                    "ax.set_xlabel(f'{results[0].xlabel} [{results[0].xunit}]')",
                    "if results[0].is_q:", "    ax.set_xscale('log')", "    ax.set_yscale('log')",
                    "ax.legend(fontsize=7)")
        return sc.text()

    def add_data(self, ds, select=True):
        ds = [d for d in ds if d is not None and len(d.x)]
        self.data += ds
        self._refresh()
        if select and ds:
            self.table.clearSelection()
            for i in range(len(self.data) - len(ds), len(self.data)):
                self.table.selectRow(i) if i == len(self.data) - len(ds) else \
                    self.table.selectionModel().select(self.table.model().index(i, 0),
                                                       QtCore.QItemSelectionModel.SelectionFlag.Select
                                                       | QtCore.QItemSelectionModel.SelectionFlag.Rows)
        return ds

    def on_add_files(self):
        ps, _ = QtWidgets.QFileDialog.getOpenFileNames(self, "1D data", "",
                                                       "1D data (*.dat *.txt *.h5 *.nxs *.0*);;all files (*)")
        new = []
        for p in ps:
            try:
                got = d1.read_data1d(p)
                for k, d in enumerate(got):
                    self._recipe_of(d, f"d1.read_data1d({os.path.abspath(p)!r})[{k}]")
                new += got
            except Exception as exc:  # noqa: BLE001
                self._log(f"not read: {os.path.basename(p)}: {exc}")
        self.add_data(new)

    def on_from_sasred(self):
        try:
            from ..sasred.gui import MainWindow as Red
            from .merge import curve_from_reduced
        except Exception as exc:  # noqa: BLE001
            self._log(f"pySASred not available: {exc}")
            return
        new = []
        for w in list(Red.instances):
            for k, r in getattr(w, "results", {}).items():
                new.append(d1.from_curve(curve_from_reduced(r, k)))
        self.add_data(new)
        self._log(f"{len(new)} curves from pySASred" if new else "no reduced data in pySASred")

    def on_from_merge(self):
        mt = self.merge_tool
        ms = [m for m in getattr(mt, "merged", []) if len(m.q)] if mt is not None else []
        if not ms:
            self._log("no merged groups (Merge geometries tab)")
            return
        sel = mt.selected() or ms
        self.add_data([d1.from_curve(m) for m in sel if len(m.q)])

    def selected(self) -> List[d1.Data1D]:
        rows = sorted({i.row() for i in self.table.selectedIndexes()})
        return [self.data[r] for r in rows if r < len(self.data)]

    def on_remove(self):
        rows = sorted({i.row() for i in self.table.selectedIndexes()}, reverse=True)
        for r in rows:
            del self.data[r]
        self._refresh()
        self.replot()

    def on_clear(self):
        self.data = []
        self._refresh()
        self.replot()

    def on_save(self, folder: Optional[str] = None):
        sel = self.selected()
        if not sel:
            self._log("select data sets to save")
            return []
        if folder is None:
            folder = QtWidgets.QFileDialog.getExistingDirectory(self, "Folder for the data sets")
            if not folder:
                return []
        out = []
        for d in sel:
            stem = re.sub(r"[^A-Za-z0-9_.+-]+", "_", d.name).strip("_")[:120] or "data"
            out.append(d1.write_ascii(os.path.join(folder, stem + ".dat"), d))
        self._log(f"saved {len(out)} data sets to {folder}")
        return out

    def _refresh(self):
        t = self.table
        t.blockSignals(True)
        t.setRowCount(len(self.data))
        for i, d in enumerate(self.data):
            vals = (str(i + 1), d.name, f"{d.xlabel} [{d.xunit}]" if d.xunit else d.xlabel, str(len(d.x)), d.source)
            for j, v in enumerate(vals):
                it = QtWidgets.QTableWidgetItem(v)
                if j != 1:
                    it.setFlags(it.flags() & ~QtCore.Qt.ItemFlag.ItemIsEditable)
                it.setToolTip(v)
                t.setItem(i, j, it)
        t.resizeColumnsToContents()
        t.setColumnWidth(1, min(t.columnWidth(1), 340))
        t.blockSignals(False)
        for c in (self.a_combo, self.b_combo):
            cur = c.currentIndex()
            c.clear()
            c.addItems([f"{i + 1}: {d.name}" for i, d in enumerate(self.data)])
            if 0 <= cur < c.count():
                c.setCurrentIndex(cur)
        if self.b_combo.count() > 1 and self.b_combo.currentIndex() == self.a_combo.currentIndex():
            self.b_combo.setCurrentIndex(1 if self.a_combo.currentIndex() == 0 else 0)

    def _on_rename(self, item):
        if item.column() == 1 and item.row() < len(self.data):
            self.data[item.row()].name = item.text()
            self._refresh()
            self.replot()

    # ------------------------------------------------------------------ operations
    def on_scalar(self, op):
        v = _num(self.value.text())
        if v is None:
            self._log(f"not a number: {self.value.text()}")
            return
        new = []
        for d in self.selected():
            try:
                r = {"scale": d.scale, "divide": d.divide, "offset": d.offset,
                     "minus": lambda x: d.offset(-x)}[op](v)
                meth, arg = ("offset", -v) if op == "minus" else (op, v)
                new.append(self._recipe_of(r, f"{{0}}.{meth}({arg!r})", [d]))
            except ZeroDivisionError as exc:
                self._log(str(exc))
        self.add_data(new)

    def on_cut(self):
        lo, hi = _num(self.xmin.text()), _num(self.xmax.text())
        self.add_data([self._recipe_of(d.cut(lo, hi), f"{{0}}.cut({lo!r}, {hi!r})", [d]) for d in self.selected()])

    def on_binary(self):
        ia, ib = self.a_combo.currentIndex(), self.b_combo.currentIndex()
        if not (0 <= ia < len(self.data) and 0 <= ib < len(self.data)):
            return
        f = _num(self.f.text()) or 1.0
        try:
            a, b = self.data[ia], self.data[ib]
            self.add_data([self._recipe_of(d1.combine(a, b, self.op.currentData(), f),
                                           f"d1.combine({{0}}, {{1}}, {self.op.currentData()!r}, {f!r})", [a, b])])
        except Exception as exc:  # noqa: BLE001
            self._log(f"not computed: {exc}")

    # ------------------------------------------------------------------ plot
    def replot(self):
        fig = self.canvas.figure
        fig.clear()
        ax = fig.add_subplot(111)
        sel = self.selected()
        f = self.offset.value()
        for k, d in enumerate(sel):
            m = f ** k
            ax.errorbar(d.x, d.y * m, yerr=None if np.all(np.isnan(d.dy)) else d.dy * m, fmt="o-", ms=3, lw=0.8,
                        elinewidth=0.6, capsize=0, label=d.name + (f" × {m:g}" if k and f != 1 else ""))
        if sel:
            d0 = sel[0]
            xs, ys = self.xscale.currentData(), self.yscale.currentData()
            allpos_x = all(np.all(d.x > 0) for d in sel)
            allpos_y = all(np.all(d.y > 0) for d in sel)
            ax.set_xscale(("log" if d0.is_q and allpos_x else "linear") if xs == "auto" else
                          ("log" if allpos_x else "linear") if xs == "log" else "linear")
            ax.set_yscale(("log" if d0.is_q and allpos_y else "linear") if ys == "auto" else
                          ("log" if allpos_y else "linear") if ys == "log" else "linear")
            ax.set_xlabel(f"{d0.xlabel} [{d0.xunit}]" if d0.xunit else d0.xlabel)
            ax.set_ylabel(f"I [{d0.yunit}]" if d0.yunit else "I")
            if len(sel) <= 12:
                ax.legend(fontsize="small")
            else:
                ax.set_title(f"{len(sel)} data sets", fontsize="small")
        self.canvas.draw_idle()
