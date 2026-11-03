"""Fit panel of the model explorer: load a data set, choose the optimiser, constraints, priors, run.

The parameters to fit are ticked in the explorer next to each parameter (fit box); the bounds appear
beneath the box while it is ticked. The panel is a widget that docks into
:class:`pysasfit.studio.explorer.ModelExplorer`. It builds a :class:`pysasfit.studio.fitting.FitProblem`
from the model on screen and its data sets (several data sets as in SASfit: every contribution belongs to some of
them, every data set has its own background; parameters are common or separate per data set; global parameters
P1, P2 … drive model parameters through links ``c2.ff.R = 2 * P1``, see :mod:`.multidata`) and runs it in a thread, so the window stays responsive and the fit can be stopped. The result
is written back into the parameter fields.

Explorer interface used here: ``has_model``, ``parameter_entries``, ``fit_selection``, ``build_model``,
``snapshot``, ``restore``, ``apply_model_values``, ``set_data``, ``clear_data``, ``set_busy`` and ``data``.
The model is a mixture of contributions; with several contributions the parameters are labelled
``c1.ff.R``, ``c2.ff.R`` ... (``c1.scale``; ``ff.R`` with a single one), the background ones ``background``,
``bg.A``, ``bg.alpha``.
"""

from __future__ import annotations

import math
import re
import threading
import time

import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets

from . import export, fitting, integration, multidata, profiling

__all__ = ["FitPanel"]


_TAB10 = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"]


def _bound(text, default):
    text = text.strip().replace(",", ".")
    if not text:
        return default
    return float(text)


class _ParameterTable(QtWidgets.QTableWidget):
    """Table of the parameters to fit; ``toPlainText`` gives it as text (label [lower, upper] per line)."""

    def toPlainText(self):
        lines = ["parameters to fit   [lower, upper]", ""]
        for i in range(self.rowCount()):
            cells = [self.item(i, j).text() if self.item(i, j) is not None else "" for j in range(3)]
            if cells[1] or cells[2]:
                local = self.item(i, 3) is not None and self.item(i, 3).checkState() == QtCore.Qt.CheckState.Checked
                lines.append(f"{cells[0]:<22s} [{cells[1]}, {cells[2]}]" + ("  per data set" if local else ""))
            else:
                lines.append(cells[0])
        return "\n".join(lines)


class _FitThread(QtCore.QThread):
    progress = QtCore.Signal(int, float)
    done = QtCore.Signal(object)  # FitSummary or an exception

    def __init__(self, problem, backend, algorithm, options, parent=None):
        super().__init__(parent)
        self.problem, self.backend, self.algorithm, self.options = problem, backend, algorithm, options
        self.cancel = threading.Event()
        self._last = 0.0

    def run(self):
        def callback(n, chi2_red):
            now = time.monotonic()
            if now - self._last > 0.1:  # the window need not hear about every evaluation
                self._last = now
                self.progress.emit(n, chi2_red)

        try:
            result = self.problem.fit(self.backend, self.algorithm, callback=callback, cancel=self.cancel, **self.options)
        except Exception as exc:  # noqa: BLE001 -- shown in the panel
            result = exc
        self.done.emit(result)


class FitPanel(QtWidgets.QWidget):
    def __init__(self, explorer, parent=None):
        super().__init__(parent)
        self.explorer = explorer
        self._selection = []  # (address, label, lower text, upper text) of the parameters ticked in the explorer
        self._thread = None
        self._before = None  # parameter values before the last fit (for Undo)
        self._summary = None  # FitSummary of the last finished fit (for the export)
        self._problem = None
        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(6, 4, 6, 4)

        # ---- data
        row = QtWidgets.QHBoxLayout()
        self.load_btn = QtWidgets.QPushButton("Add data…")
        self.load_btn.setToolTip(
            "q, I(q), sigma(q) columns (pysasfit.core.io.load_data); one or several files. Several data sets are fitted "
            "together: every contribution belongs to the data sets given in its tab ('contributes to data sets'), every "
            "data set has its own background"
        )
        self.load_btn.clicked.connect(self._load)
        self.remove_btn = QtWidgets.QPushButton("Remove")
        self.remove_btn.setToolTip("remove the data sets selected in the list")
        self.remove_btn.clicked.connect(self._remove)
        self.clear_btn = QtWidgets.QPushButton("Clear all")
        self.clear_btn.clicked.connect(self._clear)
        self.q_scale = QtWidgets.QLineEdit("1")
        self.q_scale.setFixedWidth(60)
        self.q_scale.setToolTip("factor applied to the q values of a file loaded next (e.g. 10 for 1/Å -> 1/nm)")
        self.data_label = QtWidgets.QLabel("no data")
        self.data_label.setSizePolicy(QtWidgets.QSizePolicy.Policy.Ignored, QtWidgets.QSizePolicy.Policy.Preferred)
        self.fit_qmin, self.fit_qmax = QtWidgets.QLineEdit(), QtWidgets.QLineEdit()
        for e in (self.fit_qmin, self.fit_qmax):
            e.setFixedWidth(70)
            e.setPlaceholderText("all")
            e.setToolTip("only data points inside this q range are fitted (empty: all)")
        for w in (self.load_btn, QtWidgets.QLabel("q ×"), self.q_scale, self.remove_btn, self.clear_btn, self.data_label):
            row.addWidget(w, 1 if w is self.data_label else 0)
        row.addWidget(QtWidgets.QLabel("fit q range"))
        row.addWidget(self.fit_qmin)
        row.addWidget(QtWidgets.QLabel("…"))
        row.addWidget(self.fit_qmax)
        lay.addLayout(row)
        self.data_list = QtWidgets.QListWidget()
        self.data_list.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.ExtendedSelection)
        self.data_list.setMaximumHeight(64)
        self.data_list.setToolTip("data sets (numbered from 1: these numbers are used in 'contributes to data sets' and "
                                  "in links such as  background[2] = P3)")
        self.data_list.setVisible(False)
        lay.addWidget(self.data_list)

        # ---- method
        row = QtWidgets.QHBoxLayout()
        self.backend = QtWidgets.QComboBox()
        self.available = fitting.available_backends()
        for key, (label, _algs) in self.available.items():
            self.backend.addItem(label, key)
        self.algorithm = QtWidgets.QComboBox()
        self.backend.currentIndexChanged.connect(self._fill_algorithms)
        self.max_eval = QtWidgets.QSpinBox()
        self.max_eval.setRange(0, 10_000_000)
        self.max_eval.setSpecialValueText("auto")
        self.max_eval.setToolTip("maximum number of model evaluations (auto: backend default)")
        self.polish = QtWidgets.QCheckBox("refine")
        self.polish.setChecked(True)
        self.polish.setToolTip(
            "finish with a least-squares fit started at the best point found "
            "(reaches the minimum of chi² exactly and gives the errors)"
        )
        self.fit_btn = QtWidgets.QPushButton("Fit")
        self.fit_btn.clicked.connect(self._fit_or_stop)
        self.undo_btn = QtWidgets.QPushButton("Undo fit")
        self.undo_btn.setEnabled(False)
        self.undo_btn.clicked.connect(self._undo)
        self.status = QtWidgets.QLabel("")
        self.status.setSizePolicy(QtWidgets.QSizePolicy.Policy.Ignored, QtWidgets.QSizePolicy.Policy.Preferred)
        for w in (QtWidgets.QLabel("optimiser"), self.backend, self.algorithm, QtWidgets.QLabel("max evaluations"),
                  self.max_eval, self.polish, self.fit_btn, self.undo_btn):  # fmt: skip
            row.addWidget(w)
        row.addWidget(self.status, 1)
        lay.addLayout(row)
        row = QtWidgets.QHBoxLayout()
        self.bumps_label = QtWidgets.QLabel("bumps options")
        self.bumps_opts = QtWidgets.QLineEdit()
        self.bumps_opts.setPlaceholderText("e.g.  samples=100000, burn=200, pop=20, init=lhs, thin=2   (empty: defaults)")
        self.bumps_opts.setToolTip(
            "options of the bumps method, name=value separated by commas.\n"
            "dream: steps samples burn pop thin init(eps|lhs|cov|random) alpha outliers(none|iqr|grubbs|mahal) trim\n"
            "de: steps pop CR F ftol xtol     amoeba: steps starts radius xtol ftol jump\n"
            "lm, newton: steps starts ftol xtol jump     pt: steps nT Tmin Tmax CR burn"
        )
        row.addWidget(self.bumps_label)
        row.addWidget(self.bumps_opts, 1)
        lay.addLayout(row)
        self._fill_algorithms()

        # ---- speed-ups (both optional; the result is the same)
        row = QtWidgets.QHBoxLayout()
        self.vp = QtWidgets.QCheckBox("solve linear parameters (variable projection)")
        self.vp.setToolTip(
            "scale factors, the background and the amplitude N of a size distribution are not searched by the "
            "optimiser but solved exactly by linear least squares for every trial: fewer parameters, faster, "
            "more robust. Parameters used in a constraint or prior stay with the optimiser."
        )
        self.cache = QtWidgets.QCheckBox("cache form factors")
        self.cache.setToolTip(
            "remember the intensity of every contribution and the form factor on the integration nodes during the fit: "
            "a change of the size distribution, a scale or another contribution does not recompute them. "
            "The automatic integration range is rounded outwards to a 5 % geometric grid while fitting."
        )
        self.coarse = QtWidgets.QCheckBox("coarse to fine")
        self.coarse.setToolTip(
            "first fit with a cheap size-distribution integral (a quarter of the nodes, rtol >= 1e-3), then repeat from "
            "that result with the integration settings of the Integration dialog. Needs a size distribution; a global "
            "optimiser is followed by a least-squares refinement."
        )
        self.grid = QtWidgets.QCheckBox("fixed size grid")
        self.grid.setToolTip(
            "compute the form factor once on a fixed logarithmic grid of sizes and integrate every trial of the size "
            "distribution as a weighted sum on it (implies the cache). Fast when the distribution parameters change; "
            "the grid resolves the distribution with about n_x * 2 nodes (Integration dialog) and is rebuilt when the "
            "distribution leaves it. Not used with the local monodisperse approximation."
        )
        self.profile_btn = QtWidgets.QPushButton("Profile model")
        self.profile_btn.setToolTip("time one evaluation of the model: form factor, size distribution, structure factor, overhead")
        self.profile_btn.clicked.connect(self._profile)
        self.nodes_btn = QtWidgets.QPushButton("Check nodes")
        self.nodes_btn.setToolTip(
            "find the smallest number of integration nodes n_x per panel whose result agrees with a much finer integration "
            "to 1e-4; offers to use it"
        )
        self.nodes_btn.clicked.connect(self._check_nodes)
        self._cache_before = False  # the cache box as the user set it, while the fixed grid forces it on
        self._coarse_tip = self.coarse.toolTip()
        self.grid.toggled.connect(self._grid_toggled)
        self.coarse.toggled.connect(self._coarse_note)
        row.addWidget(self.vp)
        row.addWidget(self.cache)
        row.addWidget(self.coarse)
        row.addWidget(self.grid)
        row.addStretch(1)
        row.addWidget(self.profile_btn)
        row.addWidget(self.nodes_btn)
        lay.addLayout(row)

        # ---- parameters to fit | constraints, priors, result
        split = QtWidgets.QSplitter()
        self.summary = _ParameterTable(0, 4)
        self.summary.setHorizontalHeaderLabels(["parameter to fit", "lower", "upper", "per data set"])
        self.summary.verticalHeader().setVisible(False)
        self.summary.horizontalHeader().setStretchLastSection(True)
        self.summary.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
        self.summary.setToolTip(
            "parameters ticked for fitting (fit box next to the parameter; bounds below it). 'per data set': a separate "
            "value for every data set instead of one common value (with several data sets)"
        )
        self.summary.itemChanged.connect(self._local_toggled)
        self._local = set()  # addresses fitted separately per data set
        split.addWidget(self.summary)

        tabs = QtWidgets.QTabWidget()
        self.constraints = QtWidgets.QPlainTextEdit()
        self.constraints.setPlaceholderText(
            "one inequality or equality per line, e.g.\n   c1_ff_R + c1_ff_L < 100\n   c2_scale < c1_scale\n"
            "names: the parameter labels with '.' replaced by '_' (c1.ff.R -> c1_ff_R; with one contribution ff_R); "
            "the tooltip lists them"
        )
        self.priors = QtWidgets.QPlainTextEdit()
        self.priors.setPlaceholderText(
            "priors, one per line (a term is added to chi²):\n"
            "   c1.ff.R  50  2               Gaussian: name mean sigma\n"
            "   c1.scale lognormal 1e-3 3    positive, known to a factor: name lognormal median factor\n"
            "   background halfnormal 0.1    small values preferred, >= 0: name halfnormal sigma\n"
            "   bg.A loguniform              every decade equally likely (lower bound > 0; not with scipy/DFO-LS)"
        )
        self.result = QtWidgets.QPlainTextEdit()
        self.result.setReadOnly(True)
        self.result.setFont(QtGui.QFontDatabase.systemFont(QtGui.QFontDatabase.SystemFont.FixedFont))
        tabs.addTab(self.constraints, "Constraints")
        tabs.addTab(self.priors, "Priors")
        gpage = QtWidgets.QWidget()
        gl = QtWidgets.QVBoxLayout(gpage)
        gl.setContentsMargins(2, 2, 2, 2)
        grow = QtWidgets.QHBoxLayout()
        b = QtWidgets.QPushButton("Add P")
        b.setToolTip("a new global parameter P1, P2, …")
        b.clicked.connect(lambda: self.add_global())
        grow.addWidget(b)
        b = QtWidgets.QPushButton("Remove")
        b.clicked.connect(self._remove_global)
        grow.addWidget(b)
        grow.addStretch(1)
        gl.addLayout(grow)
        self.globals = QtWidgets.QTableWidget(0, 5)
        self.globals.setHorizontalHeaderLabels(["name", "value", "fit", "lower", "upper"])
        self.globals.verticalHeader().setVisible(False)
        self.globals.setMaximumHeight(130)
        self.globals.setToolTip("global parameters: fitted like any parameter, they drive the parameters linked to them below")
        self.globals.itemChanged.connect(self._globals_edited)
        gl.addWidget(self.globals)
        self.links_edit = QtWidgets.QPlainTextEdit()
        self.links_edit.setPlaceholderText(
            "links, one per line:  parameter = factor * P\n"
            "   c1.ff.R = P1              radius of contribution 1 = P1\n"
            "   c2.ff.R = 2 * P1          … of contribution 2 twice that\n"
            "   c1.sd.mu[2] = 0.5 * P2    only in data set 2\n"
            "   background[1,3] = P3      same background in data sets 1 and 3\n"
            "   c1.ff.L = P1 + P2         any expression: + - * / **, sqrt exp log sin cos tan atan …\n"
            "the parameter names are those of the fit table (ff.R, c1.ff.R, c1.scale, background, bg.A …)"
        )
        self.links_edit.textChanged.connect(self._links_edited)
        gl.addWidget(self.links_edit, 1)
        self.link_status = QtWidgets.QLabel("")
        self.link_status.setWordWrap(True)
        gl.addWidget(self.link_status)
        tabs.addTab(gpage, "Global parameters")
        self.result_page = QtWidgets.QWidget()
        page = QtWidgets.QVBoxLayout(self.result_page)
        page.setContentsMargins(0, 0, 0, 0)
        brow = QtWidgets.QHBoxLayout()
        self.export_btn = QtWidgets.QPushButton("Export result…")
        self.export_btn.setToolTip(
            "save the last fit: Excel workbook (parameters, correlation matrix, curve, plot), PDF/PNG summary page or CSV files"
        )
        self.export_btn.clicked.connect(self._export)
        self.copy_btn = QtWidgets.QPushButton("Copy summary image")
        self.copy_btn.setToolTip("copy the summary page (data, fit, residuals, parameters, correlation matrix) as a PNG to the clipboard")
        self.copy_btn.clicked.connect(self._copy_image)
        brow.addWidget(self.export_btn)
        brow.addWidget(self.copy_btn)
        brow.addStretch(1)
        page.addLayout(brow)
        page.addWidget(self.result, 1)
        tabs.addTab(self.result_page, "Result")
        self.tabs = tabs
        split.addWidget(tabs)
        split.setSizes([300, 640])
        lay.addWidget(split, 1)
        self._update_buttons()

    # ------------------------------------------------------------------ selection
    def refresh_table(self):
        """Show the parameters ticked for fitting (called by the explorer when the model or a fit box changes)."""
        e = self.explorer
        self._selection = e.fit_selection() if e.has_model() else []
        multi = len(getattr(e, "datasets", [])) > 1
        t = self.summary
        t.blockSignals(True)
        t.setRowCount(0)
        for address, label, lo, hi in self._selection:
            i = t.rowCount()
            t.insertRow(i)
            for j, v in enumerate((label, lo or "-inf", hi or "inf")):
                it = QtWidgets.QTableWidgetItem(v)
                if j == 0:
                    it.setData(QtCore.Qt.ItemDataRole.UserRole, address)
                t.setItem(i, j, it)
            loc = QtWidgets.QTableWidgetItem("")
            flags = QtCore.Qt.ItemFlag.ItemIsUserCheckable | (QtCore.Qt.ItemFlag.ItemIsEnabled if multi else QtCore.Qt.ItemFlag.NoItemFlags)
            loc.setFlags(flags)
            loc.setCheckState(QtCore.Qt.CheckState.Checked if address in self._local else QtCore.Qt.CheckState.Unchecked)
            loc.setData(QtCore.Qt.ItemDataRole.UserRole, address)
            t.setItem(i, 3, loc)
        if not self._selection:
            t.insertRow(0)
            t.setItem(0, 0, QtWidgets.QTableWidgetItem("(none: tick the fit boxes next to the parameters)"))
        t.setColumnHidden(3, not multi)
        t.resizeColumnsToContents()
        t.blockSignals(False)
        names = [fitting._ident(label) for _a, label, *_ in e.parameter_entries()] if e.has_model() else []
        names += [g.name for g in self.global_parameters()]
        self.constraints.setToolTip("usable names: " + ", ".join(names))
        self.priors.setToolTip("usable names: " + ", ".join(names))
        self._update_buttons()

    def _local_toggled(self, item):
        if item.column() != 3:
            return
        address = item.data(QtCore.Qt.ItemDataRole.UserRole)
        if item.checkState() == QtCore.Qt.CheckState.Checked:
            self._local.add(address)
        else:
            self._local.discard(address)

    def local_addresses(self):
        return set(self._local)

    def set_local_addresses(self, addresses):
        self._local = set(addresses or ())
        self.refresh_table()

    # ------------------------------------------------------- global parameters
    def add_global(self, name=None, value=1.0, fit=True, lower="", upper=""):
        names = {g.name for g in self.global_parameters()}
        if name is None:
            k = 1
            while f"P{k}" in names:
                k += 1
            name = f"P{k}"
        t = self.globals
        t.blockSignals(True)
        i = t.rowCount()
        t.insertRow(i)
        t.setItem(i, 0, QtWidgets.QTableWidgetItem(name))
        t.setItem(i, 1, QtWidgets.QTableWidgetItem(f"{float(value):.6g}"))
        it = QtWidgets.QTableWidgetItem("")
        it.setFlags(QtCore.Qt.ItemFlag.ItemIsUserCheckable | QtCore.Qt.ItemFlag.ItemIsEnabled)
        it.setCheckState(QtCore.Qt.CheckState.Checked if fit else QtCore.Qt.CheckState.Unchecked)
        t.setItem(i, 2, it)
        t.setItem(i, 3, QtWidgets.QTableWidgetItem(str(lower)))
        t.setItem(i, 4, QtWidgets.QTableWidgetItem(str(upper)))
        t.blockSignals(False)
        self._globals_edited()
        return name

    def _remove_global(self):
        rows = sorted({ix.row() for ix in self.globals.selectedIndexes()}, reverse=True)
        if not rows and self.globals.rowCount():
            rows = [self.globals.rowCount() - 1]
        for r in rows:
            self.globals.removeRow(r)
        self._globals_edited()

    def global_parameters(self):
        """The global parameters of the table (:class:`~.multidata.GlobalParameter`); a bad number raises ValueError."""
        out = []
        t = self.globals
        for i in range(t.rowCount()):
            cell = lambda j: (t.item(i, j).text() if t.item(i, j) is not None else "").strip()  # noqa: E731
            name = cell(0)
            if not name:
                continue
            if not name.isidentifier():
                raise ValueError(f"global parameter '{name}': use a name like P1")
            try:
                value = float(cell(1).replace(",", "."))
                lo, hi = _bound(cell(3), -math.inf), _bound(cell(4), math.inf)
            except ValueError:
                raise ValueError(f"global parameter {name}: value, lower and upper must be numbers") from None
            fit = t.item(i, 2) is not None and t.item(i, 2).checkState() == QtCore.Qt.CheckState.Checked
            out.append(multidata.GlobalParameter(name, value, fit, lo, hi))
        return out

    def global_values(self):
        return {g.name: g.value for g in self.global_parameters()}

    def global_table_values(self):
        try:
            return [(g.name, g.value, g.fit, g.lower, g.upper) for g in self.global_parameters()]
        except ValueError:
            return []

    def set_global_table_values(self, rows, links=None):
        self.globals.blockSignals(True)
        self.globals.setRowCount(0)
        self.globals.blockSignals(False)
        for name, value, fit, lo, hi in rows:
            self.add_global(name, value, fit, "" if lo in (None, -math.inf) else f"{lo:g}",
                            "" if hi in (None, math.inf) else f"{hi:g}")
        if links is not None:
            self.links_edit.setPlainText(links)

    def set_global_value(self, name, value):
        t = self.globals
        for i in range(t.rowCount()):
            if t.item(i, 0) is not None and t.item(i, 0).text().strip() == name:
                t.blockSignals(True)
                t.item(i, 1).setText(f"{value:.6g}")
                t.blockSignals(False)

    def links(self):
        """Links of the text box (:class:`~.multidata.Link`); errors raise ValueError."""
        return multidata.parse_links(self.links_edit.toPlainText(), len(self._datasets()) or None)

    def link_message(self, text, error=False):
        self.link_status.setStyleSheet("color: #b00020;" if error else "color: gray;")
        self.link_status.setText(text)

    def _globals_edited(self, *_):
        try:
            self.global_parameters()
            self.link_message("")
        except ValueError as exc:
            self.link_message(str(exc), error=True)
            return
        self.refresh_table()
        if hasattr(self.explorer, "_schedule"):
            self.explorer._schedule()

    def _links_edited(self):
        try:
            links = self.links()
        except ValueError as exc:
            self.link_message(str(exc), error=True)
            return
        self.link_message(f"{len(links)} link(s)" if links else "")
        if hasattr(self.explorer, "_schedule"):
            self.explorer._schedule()

    # ------------------------------------------------------------- data sets
    def refresh_datasets(self):
        """The data sets of the explorer changed: list, label, buttons, table."""
        e = self.explorer
        self.data_list.clear()
        for k, d in enumerate(e.datasets):
            it = QtWidgets.QListWidgetItem(f"{k + 1}: {d['name']}  ({d['q'].size} points)")
            if len(e.datasets) > 1:
                it.setForeground(QtGui.QBrush(QtGui.QColor(_TAB10[k % 10])))
            self.data_list.addItem(it)
        self.data_list.setVisible(len(e.datasets) > 1)
        if not e.datasets:
            self.data_label.setText("no data")
        elif len(e.datasets) == 1:
            self.data_label.setText(f"{e.datasets[0]['name']}: {e.datasets[0]['q'].size} points")
        else:
            self.data_label.setText(f"{len(e.datasets)} data sets, {sum(d['q'].size for d in e.datasets)} points")
        self.refresh_table()

    def refresh_values(self):
        """(The values are shown next to the parameters in the explorer.)"""

    # ------------------------------------------------------------------- data
    def _load(self):
        fns, _ = QtWidgets.QFileDialog.getOpenFileNames(self, "Add data", "", "Data files (*.dat *.txt *.csv *.ses *.*);;All files (*)")
        if fns:
            self.load_files(fns)

    def load_files(self, fns):
        """Add data files (each a data set)."""
        from ..core.io import load_data

        try:
            scale = float(self.q_scale.text().replace(",", "."))
        except ValueError:
            self._message("q × must be a number", error=True)
            return
        for fn in fns:
            try:
                q, I, s, *_ = load_data(fn, q_scale=scale)
                if hasattr(self.explorer, "add_dataset"):
                    self.explorer.add_dataset(q, I, s, fn)
                else:
                    self.explorer.set_data(q, I, s, fn)
            except Exception as exc:  # noqa: BLE001 -- file format problems etc.
                self._message(f"{fn}: {type(exc).__name__}: {exc}", error=True)
                return
        self.data_label.setToolTip("\n".join(fns))
        self._message("")
        self._update_buttons()

    def _remove(self):
        rows = sorted({i.row() for i in self.data_list.selectedIndexes()}, reverse=True)
        if not hasattr(self.explorer, "remove_dataset"):
            self._clear()
            return
        if not rows and len(self._datasets()) == 1:
            rows = [0]
        for r in rows:
            self.explorer.remove_dataset(r)
        self._update_buttons()

    def _clear(self):
        self.explorer.clear_data()
        self.data_label.setText("no data")
        self._update_buttons()

    def q_range(self):
        """Fit range (q min, q max) or None."""
        lo = _bound(self.fit_qmin.text(), -math.inf)
        hi = _bound(self.fit_qmax.text(), math.inf)
        return None if (lo == -math.inf and hi == math.inf) else (lo, hi)

    # ----------------------------------------------------------------- method
    def _fill_algorithms(self, *_):
        self.algorithm.clear()
        key = self.backend.currentData()
        self.polish.setVisible(key != "scipy")
        self.bumps_label.setVisible(key == "bumps")
        self.bumps_opts.setVisible(key == "bumps")
        algs = self.available.get(key, ("", {}))[1]
        for name, text in algs.items():
            self.algorithm.addItem(name or "default", name)
            self.algorithm.setItemData(self.algorithm.count() - 1, text, QtCore.Qt.ItemDataRole.ToolTipRole)

    def _update_buttons(self):
        busy = self._thread is not None
        ready = self.explorer.has_model() and self.explorer.data is not None
        self.fit_btn.setEnabled(ready or busy)
        self.fit_btn.setText("Stop" if busy else "Fit")
        self.load_btn.setEnabled(not busy)
        self.profile_btn.setEnabled(not busy)
        self.nodes_btn.setEnabled(not busy)
        self.clear_btn.setEnabled(not busy and self.explorer.data is not None)
        self.remove_btn.setEnabled(not busy and self.explorer.data is not None)
        have = self._summary is not None and self._problem is not None
        self.export_btn.setEnabled(have and not busy)
        self.copy_btn.setEnabled(have and not busy and export.have_matplotlib())

    def _message(self, text, error=False):
        self.status.setStyleSheet("color: #b00020;" if error else "")
        self.status.setText(text)

    # ------------------------------------------------------------------- fit
    def _datasets(self):
        """The data sets of the explorer (an explorer without ``datasets`` has one, ``data``)."""
        e = self.explorer
        ds = getattr(e, "datasets", None)
        if ds is not None:
            return ds
        return [e.data] if getattr(e, "data", None) is not None else []

    def _model_for(self, k):
        e = self.explorer
        return e.model_for(k) if hasattr(e, "model_for") else e.build_model()

    def _build_problem(self):
        e = self.explorer
        for name in ("_store_bg", "_apply_links_to_rows"):
            if hasattr(e, name):
                getattr(e, name)()
        problem = fitting.FitProblem()
        sets = self._datasets()
        n = len(sets)
        names = [d["name"] for d in sets]
        if len(set(names)) < n:  # labels of separate parameters need distinct names
            names = [f"{k + 1}" for k in range(n)]
        for k, d in enumerate(sets):
            m = self._model_for(k)
            if not any(m.is_active(i) for i in range(len(m.contributions))):
                raise ValueError(f"data set {k + 1} ({d['name']}) has no contribution: see 'contributes to data sets'")
            problem.add_dataset((d["q"], d["I"], d["sigma"]), m, name=names[k], q_range=self.q_range())
        model = problem.datasets[0].model
        self.refresh_table()
        links = self.links()
        amap = e.label_map() if hasattr(e, "label_map") else {lab: a for a, lab, *_ in e.parameter_entries()}
        linked = {}  # address -> data sets (None = all) driven by links
        for ln in links:
            if ln.target not in amap:
                raise ValueError(f"link '{ln.text}': unknown parameter '{ln.target}'")
            linked.setdefault(amap[ln.target], set()).update(range(n) if ln.datasets is None else ln.datasets)
        for address, label, lo_text, hi_text in self._selection:
            try:
                lo, hi = _bound(lo_text, -math.inf), _bound(hi_text, math.inf)
            except ValueError:
                raise ValueError(f"{label}: lower/upper must be numbers") from None
            if not lo < hi:
                raise ValueError(f"{label}: lower must be smaller than upper")
            free = [k for k in range(n) if k not in linked.get(address, set())]
            if not free:
                continue  # set by a link in every data set
            if address in self._local and n > 1:
                base = label
                for k in free:
                    problem.add_parameter(address, [k], lower=lo, upper=hi, label=f"{base}[{problem.datasets[k].name}]")
            else:
                problem.add_parameter(address, free, lower=lo, upper=hi, label=label)
        # values that differ per data set (separate parameters of an earlier fit) are the start values there
        for k, ov in getattr(e, "overrides", {}).items():
            for a, v in ov.items():
                if k < n:
                    try:
                        problem._remember(k, a)
                        problem._base[(k, a)] = float(v)
                    except KeyError:
                        pass
                    for p in problem.parameters:
                        if p.address == a and p.datasets == (k,):
                            p.value = float(v)
        glob = {g.name: g for g in self.global_parameters()}
        used = set().union(*[ln.names for ln in links]) if links else set()
        for name in used:
            if name not in glob:
                raise ValueError(f"global parameter '{name}' is used in a link but not in the table")
        for g in glob.values():
            if g.name in used:
                problem.add_global(g.name, g.value, g.lower, g.upper, fixed=not g.fit)
        for ln in links:
            if ln.factor is not None:
                problem.link(ln.name, amap[ln.target], ln.datasets, ln.factor)
            else:
                problem.link_expression(amap[ln.target], ln.expression, ln.datasets, label=ln.target)
        if not any(not p.fixed for p in problem.parameters):
            raise ValueError("tick the fit boxes of the parameters to fit (or fit a global parameter)")
        for line in self.constraints.toPlainText().splitlines():
            if line.strip() and not line.strip().startswith("#"):
                problem.add_constraint(line)
        for line in self.priors.toPlainText().splitlines():
            if not line.strip() or line.strip().startswith("#"):
                continue
            problem.add_prior_text(line)
        return problem, model

    def _fit_or_stop(self):
        if self._thread is not None:
            self._thread.cancel.set()
            self._message("stopping …")
            return
        try:
            problem, model = self._build_problem()
        except (ValueError, KeyError) as exc:
            self._message(str(exc.args[0]), error=True)
            return
        e = self.explorer
        self._before = e.snapshot()
        options = {}
        if self.max_eval.value():
            options["maxeval"] = self.max_eval.value()
        if self.backend.currentData() != "scipy" and self.polish.isChecked():
            options["polish"] = True
        if self.backend.currentData() == "bumps" and self.bumps_opts.text().strip():
            try:
                options["bumps"] = fitting.parse_option_text(self.bumps_opts.text())
            except ValueError as exc:
                self._message(str(exc), error=True)
                return
        if self.vp.isChecked():
            options["vp"] = True
        if self.cache.isChecked():
            options["cache"] = True
        if self.coarse.isChecked():
            options["coarse"] = True
        if self.grid.isChecked():
            options["grid"] = True
        self._problem, self._model = problem, model
        self._thread = _FitThread(problem, self.backend.currentData(), self.algorithm.currentData(), options, self)
        self._thread.progress.connect(lambda n, c: self._message(f"fitting …  {n} evaluations, chi²/dof ≈ {c:.4g}"))
        self._thread.done.connect(self._finished)
        e.set_busy(True)
        self._update_buttons()
        self._message("fitting …")
        self._thread.start()

    def _finished(self, result):
        thread, self._thread = self._thread, None
        thread.wait()
        thread.deleteLater()
        e = self.explorer
        e.set_busy(False)
        self._update_buttons()
        if isinstance(result, Exception):
            self._message(f"{type(result).__name__}: {result}", error=True)
            self._restore()
            return
        self._write_back(self._problem)
        self._summary = result
        self.result.setPlainText(result.text())
        self.tabs.setCurrentWidget(self.result_page)
        self.undo_btn.setEnabled(True)
        self._update_buttons()
        self._message(
            f"{'converged' if result.success else 'NOT converged: ' + result.message}   chi²/dof = {result.chi2_red:.4g}",
            error=not result.success,
        )

    def _write_back(self, problem):
        """Fitted values into the explorer: common values into the rows, separate ones per data set as overrides,
        backgrounds per data set, global parameters into their table."""
        e = self.explorer
        n = len(problem.datasets)
        if not hasattr(e, "datasets"):  # an explorer with one data set and no per-data-set values
            for p in problem.parameters:
                if not p.address:
                    self.set_global_value(p.label, p.value)
            problem.apply(0)
            e.apply_model_values(problem.datasets[0].model)
            return
        overrides = {}
        for p in problem.parameters:
            if p.address and len(p.datasets) == 1 and n > 1 and p.address in self._local:
                overrides.setdefault(p.datasets[0], {})[p.address] = p.value
            if not p.address:
                self.set_global_value(p.label, p.value)
        for k in range(n):
            problem.apply(k)
            m = problem.datasets[k].model
            if k < len(e.datasets):
                e.datasets[k]["bg"] = {"background": float(m.background), "bg_A": float(m.bg_A), "bg_alpha": float(m.bg_alpha)}
        # values of the contributions (shared objects): those of the current data set; the scale rows from its model
        cur = min(e.current_ds, n - 1) if e.datasets else 0
        problem.apply(cur)
        m = problem.datasets[cur].model
        e.overrides = overrides
        for p, s_ in zip(e.models(), m.scales):
            p.scale_row.set_value(s_, emit=False)
        if e.datasets:
            e._load_bg(e.current_ds)
        else:
            e.bg_row.set_value(m.background, emit=False)
            e.bgA_row.set_value(m.bg_A, emit=False)
            e.bgalpha_row.set_value(m.bg_alpha, emit=False)
        e.after_fit()

    # ------------------------------------------------- combining the speed-ups
    def cache_requested(self):
        """The cache box as the user set it (the fixed grid ticks it without the user asking for it)."""
        return self._cache_before if self.grid.isChecked() else self.cache.isChecked()

    def _grid_toggled(self, on):
        """The fixed grid stores its form factors in the cache: tick and lock the cache box meanwhile."""
        if on:
            self._cache_before = self.cache.isChecked()
            self.cache.setChecked(True)
            self.cache.setEnabled(False)
            self.cache.setToolTip("switched on by the fixed size grid, which keeps its form factors in the cache")
        else:
            self.cache.setEnabled(True)
            self.cache.setChecked(self._cache_before)
            self.cache.setToolTip(
                "remember the intensity of every contribution and the form factor on the integration nodes during the "
                "fit: a change of the size distribution, a scale or another contribution does not recompute them. "
                "The automatic integration range is rounded outwards to a 5 % geometric grid while fitting."
            )
        self._coarse_note()

    def _coarse_note(self, *_):
        both = self.grid.isChecked() and self.coarse.isChecked()
        note = (
            "\n\nWith the fixed size grid this only helps when form-factor parameters (length, thickness, ...) are "
            "fitted too: for distribution parameters alone the grid is already cheap, and the second stage would build "
            "a second, finer grid."
        )
        self.coarse.setToolTip(self._coarse_tip + (note if self.grid.isChecked() else ""))
        if both:
            self._message("coarse to fine plus fixed grid: useful only if form-factor parameters are fitted")

    # ---------------------------------------------------------- speed analysis
    def _model_q(self):
        """q values used for timing and for the node check: the data (inside the fit range) or the curve shown."""
        d = self.explorer.data
        if d is not None:
            q = np.concatenate([np.asarray(x["q"], float) for x in self._datasets()])
            rng = self.q_range()
            if rng is not None:
                q = q[(q >= rng[0]) & (q <= rng[1])]
            if q.size:
                return q
        return self.explorer._q()

    def _show_text(self, text):
        self.result.setPlainText(text)
        self.tabs.setCurrentWidget(self.result_page)

    def _profile(self):
        if self._thread is not None or not self.explorer.has_model():
            return
        QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.CursorShape.WaitCursor)
        try:
            q = self._model_q()
            self.refresh_table()
            res = profiling.profile_model(self.explorer.build_model(), q)
            text = profiling.format_profile(res, n_free=len(self._selection) or None)
        except Exception as exc:  # noqa: BLE001 -- shown in the panel
            self._message(f"{type(exc).__name__}: {exc}", error=True)
            return
        finally:
            QtWidgets.QApplication.restoreOverrideCursor()
        self._show_text(text)
        self._message(f"one evaluation: {res['total'] * 1e3:.3g} ms")

    def _check_nodes(self):
        if self._thread is not None or not self.explorer.has_model():
            return
        e = self.explorer
        QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.CursorShape.WaitCursor)
        try:
            q = self._model_q()
            results = []
            for i, c in enumerate(e.build_model().contributions):
                if c.sd is not None:
                    label = f"contribution {i + 1} ({c.ff.name} over {c.sd.name})"
                    results.append((label, integration.suggest_n_x(c, q)))
        except Exception as exc:  # noqa: BLE001 -- shown in the panel
            self._message(f"{type(exc).__name__}: {exc}", error=True)
            return
        finally:
            QtWidgets.QApplication.restoreOverrideCursor()
        if not results:
            self._message("no contribution has a size distribution", error=True)
            return
        self._show_text(integration.format_node_check(results))
        found = [r["n_x"] for _l, r in results if r["n_x"] is not None]
        current = e.integration["n_x"]
        if not found or len(found) < sum(1 for _l, r in results if not r.get("note")):
            self._message("node check done (see Result tab)")
            return
        n = max(found)
        if n == current:
            self._message(f"n_x = {current} is already the smallest sufficient value")
            return
        answer = QtWidgets.QMessageBox.question(
            self, "Integration nodes", f"n_x = {n} is enough (now {current}). Use it in the integration settings?"
        )
        if answer == QtWidgets.QMessageBox.StandardButton.Yes:
            e.integration = {**e.integration, "n_x": int(n)}
            for p in e.panels():
                p.apply_integration(e.integration)
            e._schedule()
            self._message(f"n_x = {n} in use (Settings > Integration to change)")

    # ------------------------------------------------------------------ export
    def _dist_power(self):
        """The weight p of N(x) x^p set in the explorer's size-distribution dock (default 3 without an explorer dock)."""
        box = getattr(self.explorer, "dist_xpow", None)
        try:
            return float(box.value())
        except (AttributeError, TypeError, ValueError):
            return export.DISTRIBUTION_POWER

    def _export(self):
        if self._summary is None or self._problem is None:
            return
        filters = {}
        if export.have_xlsx():
            filters["Excel workbook (*.xlsx)"] = ".xlsx"
        if export.have_matplotlib():
            filters["PDF summary page (*.pdf)"] = ".pdf"
            filters["PNG picture (*.png)"] = ".png"
        filters["CSV files (*.csv)"] = ".csv"
        name = re.sub(r"\W+", "_", str(self._problem.datasets[0].name)).strip("_") or "fit"
        fn, chosen = QtWidgets.QFileDialog.getSaveFileName(
            self, "Export fit result", name + "_fit" + next(iter(filters.values())), ";;".join(filters)
        )
        if not fn:
            return
        suffix = filters.get(chosen, ".csv")
        ext = fn[fn.rfind("."):].lower() if "." in fn.replace("\\", "/").rsplit("/", 1)[-1] else ""
        if ext not in (".xlsx", ".pdf", ".png", ".csv", ".svg"):
            fn += suffix
            ext = suffix
        try:
            if ext == ".xlsx":
                export.save_xlsx(fn, self._summary, self._problem, dist_power=self._dist_power())
                done = fn
            elif ext == ".csv":
                done = ", ".join(p.replace("\\", "/").rsplit("/", 1)[-1] for p in export.save_csv(fn, self._summary, self._problem, dist_power=self._dist_power()))
            else:
                export.save_figure(fn, self._summary, self._problem, dist_power=self._dist_power())
                done = fn
        except Exception as exc:  # noqa: BLE001 -- missing package, file in use, ...
            self._message(f"{type(exc).__name__}: {exc}", error=True)
            return
        self._message(f"saved {done}")

    def _copy_image(self):
        if self._summary is None or self._problem is None:
            return
        try:
            png = export.figure_bytes(self._summary, self._problem, "png", dpi=150, dist_power=self._dist_power())
            image = QtGui.QImage.fromData(png, "PNG")
            QtWidgets.QApplication.clipboard().setImage(image)
        except Exception as exc:  # noqa: BLE001 -- matplotlib missing
            self._message(f"{type(exc).__name__}: {exc}", error=True)
            return
        self._message("summary image copied to the clipboard")

    def _restore(self):
        """Back to the values before the last fit."""
        if self._before is not None:
            self.explorer.restore(self._before)

    def _undo(self):
        self._restore()
        self.undo_btn.setEnabled(False)
        self._message("parameters restored")

