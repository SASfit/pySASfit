"""pySASscout window: model search for a measured small-angle scattering curve.

    pysasfit-scout-gui [data file]
    python -m pysasfit.scout.gui [data file]

Also the "pySASscout" tab of the pySASfit launcher. Panels (View menu; each can be moved, stacked or detached):

Data and search  the curve and its q range; the candidate space (form factors, size distribution, structure
                 factor, mixtures) and the settings of the search; Features, Search, Stop
Ranking          one row per candidate: chi2/dof, AICc, BIC, Akaike weight, number of parameters, warnings
Fit              data and the candidate selected in the ranking, residuals below
Details          parameters with uncertainties and the warnings of the selected candidate
Features         model-free features of the curve, the data check and the notes of the search
Log              progress of the search

Results store: every fit and every search is written to a folder in the user profile (shown on the panel and
under Settings -> Preferences -> pySASscout, where it can be changed) and reused, so repeating a search, continuing
one that was stopped, or adding models only fits what is new. "Load last result" shows the stored search of the
curve without fitting anything.

Nothing is calculated when the window opens or a session is restored: Features and Search run on their buttons.
The search runs in the background and can be stopped; the candidates fitted until then are ranked.
"""
from __future__ import annotations

import json
import math
import os
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
from .candidates import DISTRIBUTIONS, FORMS, STRUCTURE_FACTORS  # noqa: E402
from .search import ScoutOptions  # noqa: E402

__all__ = ["MainWindow", "main"]

Dock = QtCore.Qt.DockWidgetArea
COLUMNS = ["#", "candidate", "χ²/dof", "AICc", "BIC", "weight", "k", "from", "warnings"]


class _Signals(QtCore.QObject):
    progress = QtCore.Signal(str, int, int, str)
    done = QtCore.Signal(object, str)


def _fmt(v, n=4):
    try:
        v = float(v)
    except (TypeError, ValueError):
        return ""
    return "" if not math.isfinite(v) else f"{v:.{n}g}"


class MainWindow(DockMixin, QtWidgets.QMainWindow):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("pySASscout – model search for small-angle scattering data")
        self.setWindowIcon(tool_icon("pySASscout"))
        self.path = None               # data file
        self.data = None               # (q, I, sigma, dq or None) as read
        self.result = None             # ScoutResult of the last search
        self.features = None           # Features of the last Features / Search
        self._busy = False
        self._cancel = threading.Event()
        self._signals = _Signals()
        self._signals.progress.connect(self._on_progress)
        self._signals.done.connect(self._on_done)
        self._build_menus()
        self._build_input_panel()
        self.table = QtWidgets.QTableWidget(0, len(COLUMNS))
        self.table.setHorizontalHeaderLabels(COLUMNS)
        self.table.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.SingleSelection)
        self.table.verticalHeader().setVisible(False)
        h = self.table.horizontalHeader()
        h.setSectionResizeMode(QtWidgets.QHeaderView.ResizeMode.ResizeToContents)
        h.setSectionResizeMode(len(COLUMNS) - 1, QtWidgets.QHeaderView.ResizeMode.Stretch)
        self.table.currentCellChanged.connect(lambda r, *_a: self._select(r))
        self.canvas = FigureCanvasQTAgg(Figure(figsize=(6, 5), layout="constrained"))
        self.ax, self.ax_res = self.canvas.figure.subplots(2, 1, sharex=True, height_ratios=[3, 1])
        self.fit_panel = plot_panel(self.canvas, self)
        fixed = QtGui.QFontDatabase.systemFont(QtGui.QFontDatabase.SystemFont.FixedFont)
        self.details, self.features_view, self.log_view = (QtWidgets.QPlainTextEdit() for _ in range(3))
        for w in (self.details, self.features_view, self.log_view):
            w.setReadOnly(True)
            w.setFont(fixed)
            w.setLineWrapMode(QtWidgets.QPlainTextEdit.LineWrapMode.NoWrap)
        self._add_dock("input", "Data and search", self.input_widget, Dock.LeftDockWidgetArea, scrollable=True)
        self._add_dock("ranking", "Ranking", self.table, Dock.RightDockWidgetArea)
        self._add_dock("fit", "Fit", self.fit_panel, Dock.RightDockWidgetArea)
        self._add_dock("details", "Details", self.details, Dock.RightDockWidgetArea)
        self._add_dock("features", "Features", self.features_view, Dock.RightDockWidgetArea)
        self._add_dock("log", "Log", self.log_view, Dock.RightDockWidgetArea)
        self.splitDockWidget(self.docks["ranking"], self.docks["fit"], QtCore.Qt.Orientation.Horizontal)
        self.splitDockWidget(self.docks["ranking"], self.docks["details"], QtCore.Qt.Orientation.Vertical)
        self.tabifyDockWidget(self.docks["details"], self.docks["features"])
        self.tabifyDockWidget(self.docks["details"], self.docks["log"])
        self.docks["details"].raise_()
        self._finish_docks()
        self.resize(1500, 920)
        self._sized = False
        self.hide_stray_tab_bars()
        self._update_buttons()
        self.draw()

    def showEvent(self, event):
        super().showEvent(event)
        if not self._sized:
            self._sized = True
            QtCore.QTimer.singleShot(0, self._initial_sizes)

    def _initial_sizes(self):
        if getattr(self, "_session_layout", False):
            return
        w = max(self.width(), 900)
        self.resizeDocks([self.docks["input"], self.docks["ranking"], self.docks["fit"]],
                         [int(0.22 * w), int(0.40 * w), int(0.38 * w)], QtCore.Qt.Orientation.Horizontal)
        self._start_layout = self.saveState()

    # ------------------------------------------------------------------ construction
    def _build_menus(self):
        m = self.menuBar().addMenu("&File")
        a = m.addAction("&Load data…", self.on_load)
        a.setShortcut(QtGui.QKeySequence("Ctrl+L"))
        m.addSeparator()
        m.addAction("Save protocol (JSON)…", self.on_save_protocol)
        m.addAction("Save ranking (text)…", self.on_save_text)
        m.addSeparator()
        m.addAction("Load the stored result of this curve", lambda: self.load_stored())
        m.addAction("Results store…", lambda: self.show_store_info())
        from ..core import pycode
        pycode.install(self, self.python_code, menu=m)
        self.menuBar().addMenu("&View")
        from ..core.settings_dialog import add_preferences_menu
        add_preferences_menu(self, "scout")

    def _build_input_panel(self):
        w = QtWidgets.QWidget()
        lay = QtWidgets.QVBoxLayout(w)

        g = QtWidgets.QGroupBox("Data")
        f = QtWidgets.QFormLayout(g)
        self.load_btn = QtWidgets.QPushButton("Load data…")
        self.load_btn.setToolTip("Text file with the columns q, I, sigma (a fourth column dq is read but not used yet)")
        self.load_btn.clicked.connect(self.on_load)
        self.file_label = QtWidgets.QLabel("no data")
        self.file_label.setWordWrap(True)
        f.addRow(self.load_btn)
        f.addRow(self.file_label)
        self.q_min, self.q_max = QtWidgets.QLineEdit(), QtWidgets.QLineEdit()
        for e, tip in ((self.q_min, "lowest q used (empty: first point)"), (self.q_max, "highest q used (empty: last point)")):
            e.setPlaceholderText("all")
            e.setToolTip(tip)
            e.editingFinished.connect(self.draw)
            e.editingFinished.connect(lambda: self.refresh_store())
        f.addRow("q min", self.q_min)
        f.addRow("q max", self.q_max)
        lay.addWidget(g)

        g = QtWidgets.QGroupBox("Candidates")
        v = QtWidgets.QVBoxLayout(g)
        self.form_boxes = {}
        for key, spec in FORMS.items():
            b = QtWidgets.QCheckBox(spec.label or key)
            b.setChecked(True)
            b.setToolTip(f"{spec.function}; fitted: {', '.join(p.name for p in spec.fitted)}"
                         + (f"; size distribution over {spec.sd_param}" if spec.sd_param else ""))
            v.addWidget(b)
            self.form_boxes[key] = b
        self.dist_boxes = {}
        for key, spec in DISTRIBUTIONS.items():
            b = QtWidgets.QCheckBox(f"with size distribution: {spec.label or key}")
            b.setChecked(True)
            b.setToolTip(f"{spec.function}: each form factor that allows it is also tried with this distribution")
            v.addWidget(b)
            self.dist_boxes[key] = b
        f = QtWidgets.QFormLayout()
        self.sq = QtWidgets.QComboBox()
        for text, key in (("if the data suggest one", "auto"), ("always", "always"), ("never", "never")):
            self.sq.addItem(text, key)
        self.sq.setToolTip("Structure factor (" + ", ".join(s.label or k for k, s in STRUCTURE_FACTORS.items())
                           + "): tried when the curve has an interference maximum or falls towards low q")
        f.addRow("structure factor", self.sq)
        self.components = QtWidgets.QSpinBox()
        self.components.setRange(1, 3)
        self.components.setToolTip("Largest number of contributions of a mixture. A further contribution is added to the "
                                   "best candidates only while the criterion improves clearly.")
        f.addRow("contributions up to", self.components)
        v.addLayout(f)
        lay.addWidget(g)

        g = QtWidgets.QGroupBox("Search")
        f = QtWidgets.QFormLayout(g)
        self.criterion = QtWidgets.QComboBox()
        self.criterion.addItem("AICc", "aicc")
        self.criterion.addItem("BIC", "bic")
        self.criterion.setToolTip("Ranking criterion; both penalise parameters, BIC more strongly")
        f.addRow("criterion", self.criterion)
        self.starts, self.max_starts, self.seed, self.jobs = (QtWidgets.QSpinBox() for _ in range(4))
        self.starts.setRange(1, 200)
        self.starts.setValue(8)
        self.starts.setToolTip("Start points of the multistart fit of every candidate")
        self.max_starts.setRange(1, 2000)
        self.max_starts.setValue(64)
        self.max_starts.setToolTip("The starts are doubled up to this number while the best minimum is not confirmed "
                                   "(a quarter of it for candidates with a size distribution)")
        self.seed.setRange(0, 999999)
        self.seed.setValue(1)
        self.seed.setToolTip("Seed of the start points: the same seed gives the same result")
        self.jobs.setRange(1, max(os.cpu_count() or 1, 1))
        self.jobs.setToolTip("Worker processes: candidates are fitted independently")
        f.addRow("starts", self.starts)
        f.addRow("starts at most", self.max_starts)
        f.addRow("seed", self.seed)
        f.addRow("processes", self.jobs)
        self.errors = QtWidgets.QComboBox()
        self.errors.addItem("as stated", "1")
        self.errors.addItem("scaled to the scatter (auto)", "auto")
        self.errors.setToolTip("The criteria need correct error bars. 'auto' multiplies the stated errors by the ratio of "
                               "the point-to-point scatter to the stated error when that ratio is above 1.5.")
        f.addRow("error bars", self.errors)
        self.background = QtWidgets.QCheckBox("fit a constant background")
        self.background.setChecked(True)
        f.addRow(self.background)
        self.reference = QtWidgets.QCheckBox("Python reference functions")
        self.reference.setToolTip("Use the pure-Python versions of the candidate functions instead of the SASfit libraries "
                                  "(slower; used automatically when the libraries are not installed)")
        f.addRow(self.reference)
        lay.addWidget(g)

        row = QtWidgets.QHBoxLayout()
        self.features_btn = QtWidgets.QPushButton("Features")
        self.features_btn.setToolTip("Guinier radius, slopes, minima, interference peak and the data check (fast)")
        self.features_btn.clicked.connect(self.run_features)
        self.search_btn = QtWidgets.QPushButton("Search")
        self.search_btn.setToolTip("Fit and rank all selected candidates")
        self.search_btn.clicked.connect(self.run_search)
        self.stop_btn = QtWidgets.QPushButton("Stop")
        self.stop_btn.setToolTip("Stop the search; the candidates fitted so far are ranked")
        self.stop_btn.clicked.connect(self.stop)
        for b in (self.features_btn, self.search_btn, self.stop_btn):
            row.addWidget(b)
        lay.addLayout(row)
        self.progress = QtWidgets.QProgressBar()
        self.progress.setRange(0, 1)
        self.progress.setTextVisible(True)
        lay.addWidget(self.progress)
        g = QtWidgets.QGroupBox("Results store")
        v = QtWidgets.QVBoxLayout(g)
        self.store_label = QtWidgets.QLabel()
        self.store_label.setWordWrap(True)
        self.store_label.setTextInteractionFlags(QtCore.Qt.TextInteractionFlag.TextSelectableByMouse)
        self.store_label.setToolTip("Folder in which every fit and every search is kept (one small file per fit). "
                                    "Change it under Settings → Preferences → pySASscout; the folder can be moved.")
        v.addWidget(self.store_label)
        self.store_summary = QtWidgets.QLabel()
        self.store_summary.setStyleSheet("color: gray;")
        v.addWidget(self.store_summary)
        self.reuse = QtWidgets.QCheckBox("reuse stored fits")
        self.reuse.setChecked(bool(prefs.get("scout.reuse")))
        self.reuse.setToolTip("A candidate already fitted to this curve with the same settings is taken from the store. "
                              "Off: every candidate is fitted again and the stored fit is replaced.")
        v.addWidget(self.reuse)
        row = QtWidgets.QHBoxLayout()
        self.stored_btn = QtWidgets.QPushButton("Load last result")
        self.stored_btn.setToolTip("Show the last stored search of this curve (nothing is fitted)")
        self.stored_btn.clicked.connect(lambda: self.load_stored())
        info = QtWidgets.QPushButton("Info…")
        info.setToolTip("Folder, size and content of the results store")
        info.clicked.connect(lambda: self.show_store_info())
        opn = QtWidgets.QPushButton("Open folder")
        opn.clicked.connect(self.open_store_folder)
        for b in (self.stored_btn, info, opn):
            row.addWidget(b)
        v.addLayout(row)
        lay.addWidget(g)
        try:                                           # another folder after Preferences → Save
            import weakref
            from ..core.settings_dialog import on_saved
            ref = weakref.ref(self)
            on_saved(lambda: ref() is not None and ref().refresh_store())
        except Exception:  # noqa: BLE001
            pass
        lay.addStretch(1)
        self.input_widget = w

    # ------------------------------------------------------------------ results store
    def store_dir(self):
        """Folder of the results store (Settings → Preferences → pySASscout; default: next to the preferences file)."""
        return os.path.abspath((prefs.effective("scout.store") or "").strip() or _default_store())

    def _store(self):
        from .store import ResultStore

        return ResultStore(self.store_dir())

    def _data_id(self):
        from .store import data_id

        sel = self._selection()
        return None if sel is None or not sel[0].size else data_id(*sel)

    def refresh_store(self, summary=False):
        """Show the folder; ``summary``: also count what it holds (reads the folder)."""
        try:
            self.store_label.setText(self.store_dir())
            if summary:
                self.store_summary.setText(self._store().summary_text())
            did = self._data_id()
            runs = self._store().runs(did) if did else []
            self.stored_btn.setEnabled(bool(runs) and not self._busy)
            self.stored_btn.setText(f"Load last result ({len(runs)})" if runs else "Load last result")
        except RuntimeError:  # window already deleted
            pass

    def load_stored(self):
        """Show the newest stored search of the curve and q range on the panel; nothing is fitted."""
        did = self._data_id()
        res = self._store().load_run(did) if did else None
        if res is None:
            self._status("✖ no stored result for this curve and q range", "error")
            return False
        self.result, self.features = res, res.features
        self._show_result(f"stored result of {res.store['run'][:15]} UTC")
        return True

    def show_store_info(self, show=True):
        """Dialog with the folder, the size and the curves of the results store (``show=False``: built, not shown)."""
        st = self._store()
        ds = st.datasets()
        lines = [f"Folder:  {st.root}", f"Content: {st.summary_text()}", ""]
        if ds:
            lines.append("Curves (last used first):")
            from .store import _size
            for d in ds[:25]:
                lines.append(f"  {d.get('last_used', '')[:16].replace('T', ' ')}  {d['fits']:4d} fits  {d['runs']:3d} searches  "
                             f"{_size(d['bytes']):>9s}  {d.get('name', '')}")
            if len(ds) > 25:
                lines.append(f"  … and {len(ds) - 25} more")
        else:
            lines.append("The store is empty: it is filled by the first search.")
        lines += ["", "The folder is set under Settings → Preferences → pySASscout. It has relative paths only: it can be",
                  "copied or moved (network drive, cluster) and two stores can be merged by copying one into the other."]
        box = QtWidgets.QMessageBox(self)
        box.setWindowTitle("pySASscout results store")
        box.setIcon(QtWidgets.QMessageBox.Icon.Information)
        box.setText("Results store of pySASscout")
        box.setInformativeText(st.root)
        box.setDetailedText("\n".join(lines))
        opn = box.addButton("Open folder", QtWidgets.QMessageBox.ButtonRole.ActionRole)
        pref = box.addButton("Preferences…", QtWidgets.QMessageBox.ButtonRole.ActionRole)
        box.addButton(QtWidgets.QMessageBox.StandardButton.Close)
        self.store_summary.setText(st.summary_text())
        if not show:
            return box
        box.exec()
        if box.clickedButton() is opn:
            self.open_store_folder()
        elif box.clickedButton() is pref:
            from ..core.settings_dialog import open_preferences
            open_preferences(self, "scout")
        return box

    def open_store_folder(self):
        d = self.store_dir()
        try:
            os.makedirs(d, exist_ok=True)
        except OSError as e:
            self._status(f"✖ {d}: {e}", "error")
            return
        QtGui.QDesktopServices.openUrl(QtCore.QUrl.fromLocalFile(d))

    # ------------------------------------------------------------------ data
    def on_load(self):
        start = os.path.dirname(self.path) if self.path else ""
        fn, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Load data", start, "Data (*.dat *.txt *.csv *.abs);;All files (*)")
        if fn:
            self.load_file(fn)

    def load_file(self, path):
        """Read a curve (columns q, I, sigma[, dq]); returns True if it could be read."""
        from ..core.io import load_data

        try:
            self.data = load_data(path)
        except Exception as e:  # noqa: BLE001
            self._status(f"✖ {os.path.basename(path)}: {e}", "error")
            return False
        self.path = os.path.abspath(path)
        self.result = self.features = None
        q = self.data[0]
        self.file_label.setText(f"{os.path.basename(path)}\n{q.size} points, q = {q[0]:.4g} … {q[-1]:.4g}")
        self.file_label.setToolTip(self.path)
        self._fill_table()
        self.details.clear()
        self.features_view.clear()
        self._update_buttons()
        self.draw()
        self.refresh_store()
        n = len(self._store().runs(self._data_id() or ""))
        self._status(f"{os.path.basename(path)} loaded" + (f" – {n} stored search{'es' if n != 1 else ''} of this curve: "
                                                           "Load last result" if n else ""))
        return True

    def _q_range(self):
        def val(e):
            try:
                return float(e.text().replace(",", "."))
            except ValueError:
                return None
        return val(self.q_min), val(self.q_max)

    def _selection(self):
        """(q, I, sigma) inside the q range, or None without data."""
        if self.data is None:
            return None
        q, I, s = self.data[:3]
        lo, hi = self._q_range()
        ok = np.ones(q.size, bool)
        if lo is not None:
            ok &= q >= lo
        if hi is not None:
            ok &= q <= hi
        return q[ok], I[ok], s[ok]

    def options(self):
        """:class:`ScoutOptions` of the panel."""
        from ..studio import available

        forms = [k for k, b in self.form_boxes.items() if b.isChecked()]
        err = self.errors.currentData()
        return ScoutOptions(
            forms=forms, distributions=[k for k, b in self.dist_boxes.items() if b.isChecked()],
            structure_factor=self.sq.currentData(), criterion=self.criterion.currentData(), starts=self.starts.value(),
            max_starts=max(self.max_starts.value(), self.starts.value()), seed=self.seed.value(),
            max_components=self.components.value(), background=self.background.isChecked(),
            error_scale="auto" if err == "auto" else float(err), jobs=self.jobs.value(),
            reference=self.reference.isChecked() or not available(),
            store=self.store_dir(), reuse=self.reuse.isChecked(), keep_runs=int(prefs.get("scout.keep_runs")),
        )  # fmt: skip

    # ------------------------------------------------------------------ actions
    def _update_buttons(self):
        have = self.data is not None
        self.features_btn.setEnabled(have and not self._busy)
        self.search_btn.setEnabled(have and not self._busy)
        self.stop_btn.setEnabled(self._busy)
        self.load_btn.setEnabled(not self._busy)
        if hasattr(self, "stored_btn"):
            self.refresh_store()

    def run_features(self):
        from .features import extract

        sel = self._selection()
        if sel is None:
            return
        try:
            self.features = extract(*sel)
        except Exception as e:  # noqa: BLE001
            self._status(f"✖ features: {e}", "error")
            return
        self.features_view.setPlainText(self.features.text())
        self._raise("features")
        self._status("features of the curve")

    def run_search(self):
        sel = self._selection()
        if sel is None or self._busy:
            return
        opts = self.options()
        if not opts.forms:
            self._status("✖ select at least one form factor", "error")
            return
        if opts.reference and not self.reference.isChecked():
            self.log("SASfit libraries not installed: using the Python reference functions")
        self._busy = True
        self._cancel.clear()
        self._update_buttons()
        self.progress.setRange(0, 0)
        name = os.path.basename(self.path or "data")
        self.log(f"--- search: {name}, {sel[0].size} points, {len(opts.forms)} form factors, seed {opts.seed}")
        self._status("searching …", "run")
        sig, cancel = self._signals, self._cancel

        def work():
            from .search import search

            try:
                res = search(sel, opts, name=name, path=self.path, cancel=cancel,
                             progress=lambda stage, i, n, label: sig.progress.emit(stage, i, n, label))
                err = ""
            except Exception as e:  # noqa: BLE001
                res, err = None, f"{type(e).__name__}: {e}\n{traceback.format_exc()}"
            sig.done.emit(res, err)

        threading.Thread(target=work, daemon=True).start()

    def stop(self):
        if self._busy:
            self._cancel.set()
            self._status("stopping …", "run")

    def _on_progress(self, stage, i, n, label):
        self.progress.setRange(0, n)
        self.progress.setValue(i)
        self.progress.setFormat(f"{stage}: %v / %m")
        self.log(f"{stage} {i}/{n}: {label}")

    def _on_done(self, res, err):
        self._busy = False
        self.progress.setRange(0, 1)
        self.progress.setValue(0)
        self.progress.setFormat("")
        self._update_buttons()
        if err:
            self.log("✖ " + err)
            self._status("✖ search failed: " + err.splitlines()[0], "error")
            self._raise("log")
            return
        self.result, self.features = res, res.features
        fitted = sum(c.success for c in res.candidates)
        self._show_result(f"{fitted} candidates in {res.seconds:.1f} s"
                          + (f", {sum(c.stored for c in res.candidates)} from the store" if any(c.stored for c in res.candidates) else ""))
        self.refresh_store(summary=True)

    def _show_result(self, what):
        res = self.result
        self.features_view.setPlainText(res.features.text() + ("\n\n" + "\n".join(f"note: {n}" for n in res.notes) if res.notes else ""))
        self._fill_table()
        for n in res.notes:
            self.log("note: " + n)
        if res.best is None:
            self._status("✖ no candidate could be fitted (see Details and Log)", "error")
            self.details.setPlainText("\n".join(f"{c.label}: {c.message}" for c in res.candidates))
        else:
            self.table.selectRow(0)
            self._select(0)
            self._status(f"{what}; best: {res.best.label}" + (f" – {res.notes[0]}" if res.notes else ""),
                         "warning" if res.notes else "ok")
        self._raise("details")

    def _status(self, text, level="ok"):
        colour = {"error": "#c00000", "warning": "#b36b00"}.get(level, "")
        self.statusBar().setStyleSheet(f"QStatusBar {{ color: {colour}; }}" if colour else "")
        self.statusBar().showMessage(text, 0 if (colour or level == "run") else 6000)

    def _raise(self, name):
        d = self.docks.get(name)
        if d is not None and not self.detached.get(name):
            d.raise_()

    def log(self, text):
        self.log_view.appendPlainText(str(text))

    # ------------------------------------------------------------------ output
    def _fill_table(self):
        cands = self.result.candidates if self.result is not None else []
        self.table.setRowCount(len(cands))
        for r, c in enumerate(cands):
            if c.success:
                cells = [str(r + 1), c.label, _fmt(c.chi2_red), _fmt(c.aicc, 6), _fmt(c.bic, 6), f"{c.weight:.3f}", str(c.k),
                         "store" if c.stored else "fit", "; ".join(c.warnings)]
            else:
                cells = [str(r + 1), c.label, "", "", "", "", "", "", "not fitted: " + c.message]
            for k, text in enumerate(cells):
                it = QtWidgets.QTableWidgetItem(text)
                if k not in (1, len(COLUMNS) - 1):
                    it.setTextAlignment(QtCore.Qt.AlignmentFlag.AlignRight | QtCore.Qt.AlignmentFlag.AlignVCenter)
                if not c.success:
                    it.setForeground(QtGui.QBrush(QtGui.QColor("gray")))
                it.setToolTip("\n".join(c.warnings) if c.success else c.message)
                self.table.setItem(r, k, it)

    def selected(self):
        r = self.table.currentRow()
        if self.result is None or not 0 <= r < len(self.result.candidates):
            return None
        return self.result.candidates[r]

    def _select(self, _row):
        c = self.selected()
        self.details.setPlainText("" if c is None else c.text())
        self.draw()

    def draw(self):
        ax, axr = self.ax, self.ax_res
        ax.clear()
        axr.clear()
        sel = self._selection()
        if self.data is not None:
            q, I, s = self.data[:3]
            ax.errorbar(q, I, s, fmt=".", ms=3, lw=0.6, color="0.75", zorder=1)       # all points, grey
            if sel is not None and sel[0].size:
                ax.errorbar(*sel, fmt=".", ms=3, lw=0.6, color="C0", label="data", zorder=2)
        c = self.selected()
        if c is not None and c.success:
            ax.plot(c.q, c.I_fit, "-", color="C3", lw=1.4, label=c.label, zorder=3)
            axr.plot(c.q, c.residuals, ".", ms=3, color="C3")
            axr.axhline(0.0, color="0.4", lw=0.8)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_ylabel("I(q)")
        axr.set_xlabel("q")
        axr.set_ylabel("(fit − data) / σ")
        if ax.get_legend_handles_labels()[0]:
            ax.legend(fontsize="small", loc="lower left")
        self.canvas.draw_idle()

    def on_save_protocol(self):
        if self.result is None:
            self._status("✖ no search result yet", "error")
            return
        base = os.path.splitext(self.path or "scout")[0] + "_scout.json"
        fn, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save protocol", base, "JSON (*.json)")
        if fn:
            self.result.save(fn)
            self._status(f"protocol written to {fn}")

    def on_save_text(self):
        if self.result is None:
            self._status("✖ no search result yet", "error")
            return
        base = os.path.splitext(self.path or "scout")[0] + "_scout.txt"
        fn, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save ranking", base, "Text (*.txt)")
        if fn:
            with open(fn, "w", encoding="utf-8") as fh:
                fh.write(self.result.text(top=len(self.result.candidates)))
            self._status(f"ranking written to {fn}")

    def python_code(self):
        from ..core.pycode import Script, call, path_lit

        if self.path is None:
            raise ValueError("Load a curve first.")
        o, d = self.options(), ScoutOptions()
        sc = Script("pySASscout", f"model search for {os.path.basename(self.path)}")
        sc.imp("from pysasfit import scout")
        kw = {}
        if set(o.forms) != set(FORMS):
            kw["forms"] = o.forms
        if set(o.distributions) != set(DISTRIBUTIONS):
            kw["distributions"] = o.distributions
        for key in ("structure_factor", "criterion", "starts", "max_starts", "seed", "max_components", "background",
                    "error_scale", "jobs"):
            if getattr(o, key) != getattr(d, key):
                kw[key] = getattr(o, key)
        if self.reference.isChecked():
            kw["reference"] = True
        kw["store"] = Code_path(path_lit(self.store_dir()))    # the results store of the panel (fits are reused)
        if not o.reuse:
            kw["reuse"] = False
        lo, hi = self._q_range()
        if lo is not None or hi is not None:
            q = self.data[0]
            kw["q_range"] = (lo if lo is not None else float(q[0]), hi if hi is not None else float(q[-1]))
        sc.add(call("result = scout.search", Code_path(path_lit(self.path)), **kw),
               "print(result.text())",
               "# result.save('scout.json')        # protocol of the run",
               "# model = result.model(0)          # studio.ModelFunction of the best candidate")
        sc.add_plot("best = result.best",
                    "q, I, s = result._data",
                    "fig, (a1, a2) = plt.subplots(2, 1, sharex=True, height_ratios=[3, 1])",
                    "a1.errorbar(q, I, s, fmt='.', ms=3, label='data'); a1.loglog(best.q, best.I_fit, label=best.label)",
                    "a1.legend(); a2.semilogx(best.q, best.residuals, '.'); a2.axhline(0, color='0.4')",
                    "a2.set_xlabel('q'); a2.set_ylabel('(fit - data) / sigma')")
        return sc.text()

    # ------------------------------------------------------------------ session
    @staticmethod
    def _session_path():
        return os.path.join(os.path.dirname(prefs.settings_path()), "scout_session.json")

    def save_session(self):
        try:
            st = {"version": 1, "file": self.path, "q_min": self.q_min.text(), "q_max": self.q_max.text(),
                  "forms": [k for k, b in self.form_boxes.items() if b.isChecked()],
                  "distributions": [k for k, b in self.dist_boxes.items() if b.isChecked()],
                  "sq": self.sq.currentData(), "components": self.components.value(),
                  "criterion": self.criterion.currentData(), "starts": self.starts.value(),
                  "max_starts": self.max_starts.value(), "seed": self.seed.value(), "jobs": self.jobs.value(),
                  "errors": self.errors.currentData(), "background": self.background.isChecked(),
                  "reference": self.reference.isChecked(), "reuse": self.reuse.isChecked(),
                  "dock_state": self.dock_state()}
            p = self._session_path()
            os.makedirs(os.path.dirname(p), exist_ok=True)
            with open(p, "w", encoding="utf-8") as fh:
                json.dump(st, fh, indent=1)
        except (OSError, TypeError, ValueError):
            pass

    def restore_session(self):
        """The inputs of the last session (the curve is read, nothing is calculated)."""
        try:
            with open(self._session_path(), encoding="utf-8") as fh:
                st = json.load(fh)
        except (OSError, ValueError):
            return False
        try:
            if st.get("file") and os.path.isfile(st["file"]):
                self.load_file(st["file"])
            self.q_min.setText(st.get("q_min", ""))
            self.q_max.setText(st.get("q_max", ""))
            if "forms" in st:
                for k, b in self.form_boxes.items():
                    b.setChecked(k in st["forms"])
            if "distributions" in st:
                for k, b in self.dist_boxes.items():
                    b.setChecked(k in st["distributions"])
            for combo, key in ((self.sq, "sq"), (self.criterion, "criterion"), (self.errors, "errors")):
                i = combo.findData(st.get(key))
                if i >= 0:
                    combo.setCurrentIndex(i)
            for spin, key in ((self.components, "components"), (self.starts, "starts"), (self.max_starts, "max_starts"),
                              (self.seed, "seed"), (self.jobs, "jobs")):
                if key in st:
                    spin.setValue(int(st[key]))
            self.background.setChecked(bool(st.get("background", True)))
            self.reference.setChecked(bool(st.get("reference", False)))
            if "reuse" in st:
                self.reuse.setChecked(bool(st["reuse"]))
            self._session_layout = bool(self.restore_dock_state(st.get("dock_state")))
            self.draw()
            self.refresh_store()
        except Exception:  # noqa: BLE001 -- the last session is a convenience
            pass
        return True

    def forget_session(self):
        try:
            os.remove(self._session_path())
        except OSError:
            pass

    def shutdown(self):
        self._cancel.set()
        self.save_session()

    def closeEvent(self, event):
        self.shutdown()
        super().closeEvent(event)


def _default_store():
    from .store import default_dir

    return default_dir()


def Code_path(text):
    from ..core.pycode import Code

    return Code(text)


def main(argv=None):
    argv = sys.argv if argv is None else argv
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(argv)
    from ..core.tabstyle import apply_tab_style
    apply_tab_style(app)
    app.setWindowIcon(tool_icon("pySASscout"))
    w = MainWindow()
    w.show()
    files = [a for a in argv[1:] if os.path.isfile(a)]
    if files:
        w.load_file(files[0])
    else:
        w.restore_session()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
