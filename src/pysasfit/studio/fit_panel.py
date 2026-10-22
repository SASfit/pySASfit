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

import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets

from . import export, fitting, integration, profiling

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
        self._summary = None  # FitSummary of the last finished fit (for the export)
        self._problem = None
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
        have = self._summary is not None and self._problem is not None
        self.export_btn.setEnabled(have and not busy)
        self.copy_btn.setEnabled(have and not busy and export.have_matplotlib())

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
        e.apply_model_values(self._model)
        self._summary = result
        self.result.setPlainText(result.text())
        self.tabs.setCurrentWidget(self.result_page)
        self.undo_btn.setEnabled(True)
        self._update_buttons()
        self._message(
            f"{'converged' if result.success else 'NOT converged: ' + result.message}   chi²/dof = {result.chi2_red:.4g}",
            error=not result.success,
        )

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
            q = np.asarray(d["q"], float)
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


def _ident(label):
    return re.sub(r"\W+", "_", label).strip("_")
