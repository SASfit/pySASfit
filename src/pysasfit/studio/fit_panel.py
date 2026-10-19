"""Fit panel of the model explorer: load a data set, choose the optimiser, constraints, priors, run.

The parameters to fit are ticked in the explorer next to each parameter (fit box); the bounds appear
beneath the box while it is ticked. The panel is a widget that docks into
:class:`pysasfit.studio.explorer.ModelExplorer`. It builds a :class:`pysasfit.studio.fitting.FitProblem`
from the model on screen (one data set for now; the engine handles several data sets with common and local
parameters) and runs it in a thread, so the window stays responsive and the fit can be stopped. The result
is written back into the parameter fields.

Explorer interface used here: ``has_model``, ``parameter_entries``, ``fit_selection``, ``build_model``,
``snapshot``, ``restore``, ``apply_model_values``, ``set_data``, ``clear_data``, ``set_busy`` and ``data``.
The model is a mixture of contributions; parameters of several contributions are labelled
``c1.ff.R``, ``c2.ff.R`` ... (``c1.scale``, ``background``).
"""

from __future__ import annotations

import math
import re
import threading
import time

from PySide6 import QtCore, QtGui, QtWidgets

from . import fitting

__all__ = ["FitPanel"]


def _bound(text, default):
    text = text.strip().replace(",", ".")
    if not text:
        return default
    return float(text)


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
        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(6, 4, 6, 4)

        # ---- data
        row = QtWidgets.QHBoxLayout()
        self.load_btn = QtWidgets.QPushButton("Load data…")
        self.load_btn.setToolTip("q, I(q), sigma(q) columns (pysasfit.core.io.load_data); shown with the model")
        self.load_btn.clicked.connect(self._load)
        self.clear_btn = QtWidgets.QPushButton("Clear data")
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
        for w in (self.load_btn, QtWidgets.QLabel("q ×"), self.q_scale, self.clear_btn, self.data_label):
            row.addWidget(w, 1 if w is self.data_label else 0)
        row.addWidget(QtWidgets.QLabel("fit q range"))
        row.addWidget(self.fit_qmin)
        row.addWidget(QtWidgets.QLabel("…"))
        row.addWidget(self.fit_qmax)
        lay.addLayout(row)

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
        self._fill_algorithms()

        # ---- parameters to fit | constraints, priors, result
        split = QtWidgets.QSplitter()
        self.summary = QtWidgets.QPlainTextEdit()
        self.summary.setReadOnly(True)
        self.summary.setFont(QtGui.QFontDatabase.systemFont(QtGui.QFontDatabase.SystemFont.FixedFont))
        self.summary.setToolTip("parameters ticked for fitting (fit box next to the parameter; bounds below it)")
        split.addWidget(self.summary)

        tabs = QtWidgets.QTabWidget()
        self.constraints = QtWidgets.QPlainTextEdit()
        self.constraints.setPlaceholderText(
            "one inequality per line, e.g.\n   c1_ff_R + c1_ff_L < 100\n   c2_scale < c1_scale\n"
            "names: parameter names with '.' replaced by '_' (see tooltip)"
        )
        self.priors = QtWidgets.QPlainTextEdit()
        self.priors.setPlaceholderText(
            "Gaussian priors, one per line: name  mean  sigma\n   c1.ff.R  50  2\nadds ((value - mean)/sigma)^2 to chi²"
        )
        self.result = QtWidgets.QPlainTextEdit()
        self.result.setReadOnly(True)
        self.result.setFont(QtGui.QFontDatabase.systemFont(QtGui.QFontDatabase.SystemFont.FixedFont))
        tabs.addTab(self.constraints, "Constraints")
        tabs.addTab(self.priors, "Priors")
        tabs.addTab(self.result, "Result")
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
        lines = ["parameters to fit   [lower, upper]", ""]
        for _a, label, lo, hi in self._selection:
            lines.append(f"{label:<22s} [{lo or '-inf'}, {hi or 'inf'}]")
        if not self._selection:
            lines.append("(none: tick the fit boxes next to the parameters)")
        self.summary.setPlainText("\n".join(lines))
        names = ", ".join(_ident(label) for _a, label, *_ in e.parameter_entries()) if e.has_model() else ""
        self.constraints.setToolTip("usable names: " + names)
        self.priors.setToolTip("usable names: " + names)
        self._update_buttons()

    def refresh_values(self):
        """(The values are shown next to the parameters in the explorer.)"""

    # ------------------------------------------------------------------- data
    def _load(self):
        fn, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Load data", "", "Data files (*.dat *.txt *.csv *.ses *.*);;All files (*)")
        if not fn:
            return
        try:
            scale = float(self.q_scale.text().replace(",", "."))
            from ..core.io import load_data

            q, I, s, *_ = load_data(fn, q_scale=scale)
            self.explorer.set_data(q, I, s, fn)
        except Exception as exc:  # noqa: BLE001 -- file format problems etc.
            self._message(f"{type(exc).__name__}: {exc}", error=True)
            return
        self.data_label.setText(f"{self.explorer.data['name']}: {self.explorer.data['q'].size} points")
        self.data_label.setToolTip(fn)
        self._message("")
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
        self.clear_btn.setEnabled(not busy and self.explorer.data is not None)

    def _message(self, text, error=False):
        self.status.setStyleSheet("color: #b00020;" if error else "")
        self.status.setText(text)

    # ------------------------------------------------------------------- fit
    def _build_problem(self):
        e = self.explorer
        d = e.data
        model = e.build_model()
        problem = fitting.FitProblem()
        problem.add_dataset((d["q"], d["I"], d["sigma"]), model, name=d["name"], q_range=self.q_range())
        self.refresh_table()
        for address, label, lo_text, hi_text in self._selection:
            try:
                lo, hi = _bound(lo_text, -math.inf), _bound(hi_text, math.inf)
            except ValueError:
                raise ValueError(f"{label}: lower/upper must be numbers") from None
            if not lo < hi:
                raise ValueError(f"{label}: lower must be smaller than upper")
            problem.add_parameter(address, lower=lo, upper=hi, label=label)
        if not problem.parameters:
            raise ValueError("tick the fit boxes of the parameters to fit")
        for line in self.constraints.toPlainText().splitlines():
            if line.strip() and not line.strip().startswith("#"):
                problem.add_constraint(line)
        for line in self.priors.toPlainText().splitlines():
            if not line.strip() or line.strip().startswith("#"):
                continue
            parts = [t for t in re.split(r"[\s,;=±]+|\+-", line.strip()) if t]
            if len(parts) != 3:
                raise ValueError(f"prior '{line.strip()}': write  name  mean  sigma")
            try:
                problem.add_prior(parts[0], float(parts[1]), float(parts[2]))
            except (KeyError, ValueError) as exc:
                raise ValueError(f"prior '{line.strip()}': {exc.args[0]}") from None
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
        e.apply_model_values(self._model)
        self.result.setPlainText(result.text())
        self.tabs.setCurrentWidget(self.result)
        self.undo_btn.setEnabled(True)
        self._message(
            f"{'converged' if result.success else 'NOT converged: ' + result.message}   chi²/dof = {result.chi2_red:.4g}",
            error=not result.success,
        )

    def _restore(self):
        """Back to the values before the last fit."""
        if self._before is not None:
            self.explorer.restore(self._before)

    def _undo(self):
        self._restore()
        self.undo_btn.setEnabled(False)
        self._message("parameters restored")


def _ident(label):
    return re.sub(r"\W+", "_", label).strip("_")
