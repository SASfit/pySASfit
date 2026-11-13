"""pySEBpolymers window: form factors of structures built from linked sub-units.

    pysasfit-seb [data file]
    python -m pysasfit.seb.gui [data file]

Also the "pySEBpolymers" tab of the pySASfit launcher. Panels (View menu; each can be moved, stacked or detached):

Designer     the structure drawn as a graph: sub-units, patterns (Array, Chain, Generations) and components,
             linked at their reference points (pysasfit.seb.editor); a template (micelle, star, multiblock chain,
             dendrimer, bottle brush, rod-coil, vesicle with loops, custom) and its options start a design
Code         the Python code of the structure: written by the designer, or typed and built with "Build" (Ctrl+B)
Parameters   excess scattering lengths beta_<tag> and sizes (Rg_, R_, L_, ...) of the structure, scale and flat
             background: value, fit, lower and upper bound; q grid of the model; Fit
I(q)         model, data and residuals (normalised residuals (I - model)/sigma)
3D sketch    one random realisation of the structure (pysasfit.seb.sketch)
Components   the sub-units, their tags and the junctions
Log          builds and fits

I(q) = scale * F(q) + background with the normalised form factor F (F(0) = 1), or scale * S(q) + background with
the unnormalised S(q) = (sum beta)^2 F(q). Lengths are in the inverse unit of q (Å for q in 1/Å).
"""
from __future__ import annotations

import json
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
from . import templates as T  # noqa: E402

__all__ = ["MainWindow", "main"]

Dock = QtCore.Qt.DockWidgetArea
COLS = ["parameter", "value", "fit", "min", "max", "± error"]
EXTRA = ("scale", "background")


def _num(text, default=None):
    try:
        v = float(str(text).replace(",", "."))
    except (TypeError, ValueError):
        return default
    return v


def _fmt(v, n=6):
    if v is None or not np.isfinite(v):
        return "inf" if v == np.inf else ("-inf" if v == -np.inf else "")
    return f"{v:.{n}g}"


class _Signals(QtCore.QObject):
    fitted = QtCore.Signal(object, object)


class MainWindow(DockMixin, QtWidgets.QMainWindow):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("pySEBpolymers – form factors of linked sub-units")
        self.setWindowIcon(tool_icon("pySEBpolymers"))
        self.structure = None
        self.code_built = None
        self.data = None                  # dict q, I, s, path
        self._fit_running = False
        self._signals = _Signals()
        self._signals.fitted.connect(self._on_fitted)
        self._timer = QtCore.QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.setInterval(150)
        self._timer.timeout.connect(self.recalc)
        self._design_code = None          # code generated from the designer (None: structure from typed code)
        self._seed = 1
        self._sketch_stale = True
        self._build_menus()
        self._build_structure_panel()
        self._build_designer()
        self._build_parameter_panel()
        self._build_plot()
        self._build_sketch()
        self.components = QtWidgets.QPlainTextEdit()
        self.components.setReadOnly(True)
        mono = QtGui.QFontDatabase.systemFont(QtGui.QFontDatabase.SystemFont.FixedFont)
        self.components.setFont(mono)
        self.log_view = QtWidgets.QPlainTextEdit()
        self.log_view.setReadOnly(True)
        self.log_view.setFont(mono)
        self._add_dock("designer", "Designer", self.designer_widget, Dock.LeftDockWidgetArea)
        self._add_dock("structure", "Code", self.structure_widget, Dock.LeftDockWidgetArea)
        self._add_dock("params", "Parameters", self.param_widget, Dock.LeftDockWidgetArea)
        self._add_dock("iq", "I(q)", self.iq_panel, Dock.RightDockWidgetArea)
        self._add_dock("sketch", "3D sketch", self.sketch_panel, Dock.RightDockWidgetArea)
        self._add_dock("components", "Components", self.components, Dock.RightDockWidgetArea)
        self._add_dock("log", "Log", self.log_view, Dock.RightDockWidgetArea)
        self.splitDockWidget(self.docks["designer"], self.docks["iq"], QtCore.Qt.Orientation.Horizontal)
        self.splitDockWidget(self.docks["designer"], self.docks["params"], QtCore.Qt.Orientation.Vertical)
        self.tabifyDockWidget(self.docks["designer"], self.docks["structure"])
        self.splitDockWidget(self.docks["iq"], self.docks["components"], QtCore.Qt.Orientation.Vertical)
        self.tabifyDockWidget(self.docks["iq"], self.docks["sketch"])
        self.tabifyDockWidget(self.docks["components"], self.docks["log"])
        self.docks["designer"].raise_()
        self.docks["iq"].raise_()
        self.docks["components"].raise_()
        self.docks["sketch"].visibilityChanged.connect(
            lambda v: v and self._sketch_stale and getattr(self, "curve", (None,))[0] is not None
            and self.draw_sketch())
        self._finish_docks()
        self.resize(1450, 900)
        self._sized = False
        self.hide_stray_tab_bars()
        self.on_template(self.template.currentText(), build=False)
        self._quiet = True                   # built, not calculated, at the start: Calculate
        try:
            self.build()
        finally:
            self._quiet = False

    def showEvent(self, event):
        super().showEvent(event)
        if not self._sized:
            self._sized = True
            QtCore.QTimer.singleShot(0, self._initial_sizes)

    def _initial_sizes(self):
        if getattr(self, "_session_layout", False):
            return
        w = max(self.width(), 900)
        self.resizeDocks([self.docks["designer"], self.docks["iq"]], [int(0.55 * w), int(0.45 * w)],
                         QtCore.Qt.Orientation.Horizontal)
        h = max(self.height(), 600)
        self.resizeDocks([self.docks["designer"], self.docks["params"]], [int(0.62 * h), int(0.38 * h)],
                         QtCore.Qt.Orientation.Vertical)
        self.resizeDocks([self.docks["iq"], self.docks["components"]], [int(0.72 * h), int(0.28 * h)],
                         QtCore.Qt.Orientation.Vertical)
        self._start_layout = self.saveState()

    # ------------------------------------------------------------------ construction
    def _build_menus(self):
        m = self.menuBar().addMenu("&File")
        a = m.addAction("&Load data…", self.on_load)
        a.setShortcut(QtGui.QKeySequence("Ctrl+L"))
        m.addAction("Remove data", self.remove_data)
        m.addSeparator()
        m.addAction("Open design…", self.on_open_design)
        m.addAction("Save design…", self.on_save_design)
        m.addAction("Open structure code…", self.on_open_code)
        m.addAction("Save structure code…", self.on_save_code)
        m.addAction("Export model curve…", self.on_export)
        from ..core import pycode
        pycode.install(self, self.python_code, menu=m)
        from ..core.settings_dialog import add_preferences_menu
        add_preferences_menu(self, "seb")
        self.menuBar().addMenu("&View")

    def _build_structure_panel(self):
        # template row (shown above the designer)
        tb = QtWidgets.QWidget()
        tl = QtWidgets.QHBoxLayout(tb)
        tl.setContentsMargins(4, 2, 4, 2)
        tl.addWidget(QtWidgets.QLabel("Template"))
        self.template = QtWidgets.QComboBox()
        self.template.addItems(list(T.TEMPLATES))
        self.template.setToolTip("start from a ready-made design; its options below, everything else in the designer")
        self.template.currentTextChanged.connect(lambda n: self.on_template(n))
        self.template.setSizeAdjustPolicy(QtWidgets.QComboBox.SizeAdjustPolicy.AdjustToContents)
        tl.addWidget(self.template)
        self.opt_box = QtWidgets.QWidget()
        self.opt_form = QtWidgets.QHBoxLayout(self.opt_box)
        self.opt_form.setContentsMargins(6, 0, 0, 0)
        tl.addWidget(self.opt_box)
        tl.addStretch(1)
        self.template_bar = tb
        self.opt_widgets = {}
        # code panel
        w = QtWidgets.QWidget()
        lay = QtWidgets.QVBoxLayout(w)
        lay.setContentsMargins(4, 4, 4, 4)
        hint = QtWidgets.QLabel("Python code of the structure: written by the designer, or typed here. Build "
                                "(Ctrl+B) uses this code and draws it in the designer.")
        hint.setWordWrap(True)
        hint.setStyleSheet("color: gray;")
        lay.addWidget(hint)
        self.editor = QtWidgets.QPlainTextEdit()
        self.editor.setFont(QtGui.QFontDatabase.systemFont(QtGui.QFontDatabase.SystemFont.FixedFont))
        self.editor.setLineWrapMode(QtWidgets.QPlainTextEdit.LineWrapMode.NoWrap)
        self.editor.setToolTip("Python code that builds the Structure s (pysasfit.seb). Edit it and press Build.")
        self.editor.textChanged.connect(self._code_edited)
        lay.addWidget(self.editor, 1)
        row = QtWidgets.QHBoxLayout()
        self.build_button = QtWidgets.QPushButton("Build")
        self.build_button.setShortcut(QtGui.QKeySequence("Ctrl+B"))
        self.build_button.setToolTip("Build the structure from the code (Ctrl+B)")
        self.build_button.clicked.connect(lambda: self.build())
        row.addWidget(self.build_button)
        self.struct_status = QtWidgets.QLabel("")
        self.struct_status.setWordWrap(True)
        row.addWidget(self.struct_status, 1)
        lay.addLayout(row)
        self.structure_widget = w

    def _build_designer(self):
        from .editor import DesignerWidget
        self.designer = DesignerWidget()
        self.designer.changed.connect(self.on_design)
        w = QtWidgets.QWidget()
        lay = QtWidgets.QVBoxLayout(w)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(self.template_bar)
        lay.addWidget(self.designer, 1)
        self.designer_widget = w

    def _build_sketch(self):
        self.sketch_canvas = FigureCanvasQTAgg(Figure(figsize=(5, 5), layout="constrained"))
        row = QtWidgets.QHBoxLayout()
        b = QtWidgets.QPushButton("New realisation")
        b.setToolTip("another random conformation, orientation and choice of the random points")
        b.clicked.connect(lambda: self.draw_sketch(new=True))
        row.addWidget(b)
        self.sketch_follow = QtWidgets.QCheckBox("follow changes")
        self.sketch_follow.setChecked(True)
        row.addWidget(self.sketch_follow)
        row.addWidget(QtWidgets.QLabel("at most"))
        self.sketch_max = QtWidgets.QSpinBox()
        self.sketch_max.setRange(10, 50000)
        self.sketch_max.setValue(1500)
        self.sketch_max.setSuffix(" sub-units")
        self.sketch_max.setKeyboardTracking(False)
        self.sketch_max.valueChanged.connect(lambda _v: self.draw_sketch())
        row.addWidget(self.sketch_max)
        row.addStretch(1)
        self.sketch_panel = plot_panel(self.sketch_canvas, self, bars=[row])
        self._sketch_timer = QtCore.QTimer(self)
        self._sketch_timer.setSingleShot(True)
        self._sketch_timer.setInterval(400)
        self._sketch_timer.timeout.connect(self.draw_sketch)

    def _build_parameter_panel(self):
        w = QtWidgets.QWidget()
        lay = QtWidgets.QVBoxLayout(w)
        lay.setContentsMargins(4, 4, 4, 4)
        self.table = QtWidgets.QTableWidget(0, len(COLS))
        self.table.setHorizontalHeaderLabels(COLS)
        self.table.verticalHeader().setVisible(False)
        self.table.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.ResizeMode.ResizeToContents)
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.itemChanged.connect(self._param_changed)
        lay.addWidget(self.table, 1)
        f = QtWidgets.QFormLayout()
        self.mode = QtWidgets.QComboBox()
        self.mode.addItems(["normalised F(q)", "unnormalised S(q)"])
        self.mode.setToolTip("I = scale·F + background (F(0) = 1), or I = scale·S + background with S(0) = (Σβ)²")
        self.mode.currentIndexChanged.connect(lambda _i: self.schedule())
        f.addRow("model", self.mode)
        g = QtWidgets.QHBoxLayout()
        self.qmin = QtWidgets.QLineEdit("0.001")
        self.qmax = QtWidgets.QLineEdit("0.5")
        self.nq = QtWidgets.QSpinBox()
        self.nq.setRange(10, 100000)
        self.nq.setValue(400)
        for e in (self.qmin, self.qmax):
            e.setMaximumWidth(80)
            e.editingFinished.connect(self.schedule)
        self.nq.valueChanged.connect(lambda _v: self.schedule())
        g.addWidget(self.qmin)
        g.addWidget(QtWidgets.QLabel("…"))
        g.addWidget(self.qmax)
        g.addWidget(QtWidgets.QLabel("points"))
        g.addWidget(self.nq)
        g.addStretch(1)
        f.addRow("q grid (log)", g)
        self.data_label = QtWidgets.QLabel("no data (File → Load data, Ctrl+L)")
        self.data_label.setStyleSheet("color: gray;")
        self.data_label.setWordWrap(True)
        f.addRow("data", self.data_label)
        lay.addLayout(f)
        row = QtWidgets.QHBoxLayout()
        self.fit_button = QtWidgets.QPushButton("Fit")
        self.fit_button.setToolTip("Least-squares fit of the ticked parameters to the data (weights 1/σ²)")
        self.fit_button.clicked.connect(lambda: self.fit())
        row.addWidget(self.fit_button)
        b = QtWidgets.QPushButton("Calculate")
        b.clicked.connect(self.recalc)
        row.addWidget(b)
        from ..core import pycode
        row.addWidget(pycode.button(self.python_code, self))
        row.addStretch(1)
        self.fit_status = QtWidgets.QLabel("")
        row.addWidget(self.fit_status)
        lay.addLayout(row)
        self.param_widget = w

    def _build_plot(self):
        self.canvas = FigureCanvasQTAgg(Figure(figsize=(6, 5), layout="constrained"))
        self.iq_panel = plot_panel(self.canvas, self)

    # ------------------------------------------------------------------ template and structure
    def on_template(self, name, build=True):
        while self.opt_form.count():
            it = self.opt_form.takeAt(0)
            if it.widget() is not None:
                it.widget().deleteLater()
        self.opt_widgets = {}
        for op in T.TEMPLATES[name][1]:
            if op.kind == "int":
                wdg = QtWidgets.QSpinBox()
                wdg.setRange(op.lo, op.hi)
                wdg.setValue(op.default)
                wdg.valueChanged.connect(lambda _v: self._options_changed())
            elif op.kind == "bool":
                wdg = QtWidgets.QCheckBox()
                wdg.setChecked(op.default)
                wdg.toggled.connect(lambda _v: self._options_changed())
            else:
                wdg = QtWidgets.QComboBox()
                wdg.addItems(list(op.choices))
                wdg.setCurrentText(op.default)
                wdg.currentTextChanged.connect(lambda _v: self._options_changed())
            if op.tip:
                wdg.setToolTip(op.tip)
            self.opt_widgets[op.key] = wdg
            self.opt_form.addWidget(QtWidgets.QLabel(op.label))
            self.opt_form.addWidget(wdg)
        self.opt_box.setVisible(bool(self.opt_widgets))
        self._options_changed(build=build)

    def options(self):
        out = {}
        for k, wdg in self.opt_widgets.items():
            if isinstance(wdg, QtWidgets.QSpinBox):
                out[k] = wdg.value()
            elif isinstance(wdg, QtWidgets.QCheckBox):
                out[k] = wdg.isChecked()
            else:
                out[k] = wdg.currentText()
        return out

    def set_options(self, opts):
        for k, v in (opts or {}).items():
            wdg = self.opt_widgets.get(k)
            if wdg is None:
                continue
            wdg.blockSignals(True)
            if isinstance(wdg, QtWidgets.QSpinBox):
                wdg.setValue(int(v))
            elif isinstance(wdg, QtWidgets.QCheckBox):
                wdg.setChecked(bool(v))
            else:
                wdg.setCurrentText(str(v))
            wdg.blockSignals(False)

    def _options_changed(self, build=True):
        d = T.design_for(self.template.currentText(), self.options())
        self.designer.set_design(d)
        code = d.to_code()
        self._design_code = code
        self.editor.blockSignals(True)
        self.editor.setPlainText(code)
        self.editor.blockSignals(False)
        self._code_edited()
        if build:
            self.build(from_design=True)

    def on_design(self, design):
        """The designer changed: build the structure from its code (if the design is complete)."""
        probs = design.problems()
        if probs:
            self.struct_status.setText("<span style='color:#c8283c'>design: " + "; ".join(probs) + "</span>")
            return False
        try:
            code = design.to_code()
        except Exception as exc:  # noqa: BLE001
            self.struct_status.setText(f"<span style='color:#c8283c'>{exc}</span>")
            return False
        self._design_code = code
        self.editor.blockSignals(True)
        self.editor.setPlainText(code)
        self.editor.blockSignals(False)
        return self.build(from_design=True)

    def _code_edited(self):
        edited = self.editor.toPlainText() != self.code_built
        self.build_button.setStyleSheet("font-weight: bold;" if edited else "")

    def build(self, code=None, from_design=False):
        """Build the structure from ``code`` (default: the editor). Code that the designer did not write is drawn
        in the designer (one node per sub-unit). Returns True on success."""
        if code is not None:
            self.editor.setPlainText(code)
        code = self.editor.toPlainText()
        try:
            s = T.build(code)
        except Exception as exc:  # noqa: BLE001 -- show, don't crash
            msg = "".join(traceback.format_exception_only(type(exc), exc)).strip()
            self.struct_status.setText(f"<span style='color:#c8283c'>{msg}</span>")
            self.log(f"build failed: {msg}")
            return False
        self.structure = s
        self.code_built = code
        self._code_edited()
        if not from_design and code != self._design_code:
            from .graph import Design
            try:
                d = Design.from_structure(s)
                self._design_code = None
                self.designer.set_design(d)
                self.designer.status.setText("drawn from the code (patterns and loops of the code are shown one "
                                             "sub-unit at a time)")
            except ValueError as exc:
                self.designer.set_design(Design(s.name))
                self.designer.status.setText(f"structure from code: {exc}; edit it in the Code panel")
        s.set_quadrature(int(prefs.get("seb.n_mu")))
        self.struct_status.setText(f"{len(s.units())} sub-units, {len(s.links())} junctions, "
                                   f"{len(s.parameters())} parameters")
        self._fill_table()
        self._show_components()
        self.log(f"built '{s.name}': {len(s.units())} sub-units")
        self._sketch_stale = True
        self.recalc()
        return True

    def _show_components(self):
        s = self.structure
        lines = [f"{s.name}: {len(s.units())} sub-units", "", "sub-unit            kind                  tag"]
        units = s.units()
        for n, k, t in units[:500]:
            lines.append(f"{n:19s} {k:21s} {t}")
        if len(units) > 500:
            lines.append(f"... and {len(units) - 500} more")
        links = s.links()
        lines += ["", f"junctions ({len(links)})"]
        for j in links[:500]:
            lines.append("  " + " = ".join(j))
        if len(links) > 500:
            lines.append(f"  ... and {len(links) - 500} more")
        self.components.setPlainText("\n".join(lines))

    # ------------------------------------------------------------------ parameter table
    def _row_of(self):
        return {self.table.item(r, 0).text(): r for r in range(self.table.rowCount()) if self.table.item(r, 0)}

    def param_rows(self):
        """{name: dict(value, fit, lo, hi)} of the table."""
        out = {}
        for r in range(self.table.rowCount()):
            name = self.table.item(r, 0).text()
            out[name] = {"value": _num(self.table.item(r, 1).text(), 0.0),
                         "fit": self.table.item(r, 2).checkState() == QtCore.Qt.CheckState.Checked,
                         "lo": _num(self.table.item(r, 3).text(), -np.inf),
                         "hi": _num(self.table.item(r, 4).text(), np.inf)}
        return out

    def _fill_table(self, rows=None):
        old = self.param_rows() if rows is None else rows
        names = list(EXTRA) + self.structure.parameters()
        self.table.blockSignals(True)
        self.table.setRowCount(len(names))
        for r, n in enumerate(names):
            if n in old:
                d = old[n]
            elif n == "scale":
                d = {"value": 1.0, "fit": True, "lo": 0.0, "hi": np.inf}
            elif n == "background":
                d = {"value": 0.0, "fit": False, "lo": -np.inf, "hi": np.inf}
            elif n.startswith("nu_"):
                d = {"value": T.default_value(n), "fit": False, "lo": 0.3, "hi": 1.0}
            elif n.startswith("sigma_"):
                d = {"value": T.default_value(n), "fit": False, "lo": 0.0, "hi": 3.0}
            elif n.startswith("vex_"):
                d = {"value": T.default_value(n), "fit": False, "lo": 0.0, "hi": 0.99}
            else:
                d = {"value": T.default_value(n), "fit": False,
                     "lo": -np.inf if n.startswith("beta_") else 0.0, "hi": np.inf}
            it = QtWidgets.QTableWidgetItem(n)
            it.setFlags(it.flags() & ~QtCore.Qt.ItemFlag.ItemIsEditable)
            self.table.setItem(r, 0, it)
            self.table.setItem(r, 1, QtWidgets.QTableWidgetItem(_fmt(d["value"])))
            fit = QtWidgets.QTableWidgetItem("")
            fit.setFlags(QtCore.Qt.ItemFlag.ItemIsUserCheckable | QtCore.Qt.ItemFlag.ItemIsEnabled)
            fit.setCheckState(QtCore.Qt.CheckState.Checked if d["fit"] else QtCore.Qt.CheckState.Unchecked)
            self.table.setItem(r, 2, fit)
            self.table.setItem(r, 3, QtWidgets.QTableWidgetItem(_fmt(d["lo"])))
            self.table.setItem(r, 4, QtWidgets.QTableWidgetItem(_fmt(d["hi"])))
            err = QtWidgets.QTableWidgetItem(d.get("err", ""))
            err.setFlags(err.flags() & ~QtCore.Qt.ItemFlag.ItemIsEditable)
            self.table.setItem(r, 5, err)
        self.table.blockSignals(False)

    def set_param(self, name, value=None, fit=None, lo=None, hi=None):
        r = self._row_of()[name]
        self.table.blockSignals(True)
        if value is not None:
            self.table.item(r, 1).setText(_fmt(value))
        if fit is not None:
            self.table.item(r, 2).setCheckState(QtCore.Qt.CheckState.Checked if fit else QtCore.Qt.CheckState.Unchecked)
        if lo is not None:
            self.table.item(r, 3).setText(_fmt(lo))
        if hi is not None:
            self.table.item(r, 4).setText(_fmt(hi))
        self.table.blockSignals(False)
        self.schedule()

    def _param_changed(self, item):
        if item.column() in (1, 2):
            self.schedule()

    def values(self):
        return {k: d["value"] for k, d in self.param_rows().items()}

    # ------------------------------------------------------------------ model
    def q_model(self):
        lo, hi = _num(self.qmin.text(), 1e-3), _num(self.qmax.text(), 0.5)
        if self.data is not None:
            lo, hi = min(lo, self.data["q"].min()), max(hi, self.data["q"].max())
        lo, hi = max(lo, 1e-8), max(hi, lo * 1.01)
        return np.geomspace(lo, hi, self.nq.value())

    def model(self, q, values=None):
        v = self.values() if values is None else values
        F = self.structure.form_factor(q, v, normalized=self.mode.currentIndex() == 0)
        return v.get("scale", 1.0) * F + v.get("background", 0.0)

    def schedule(self):
        if prefs.get("seb.auto"):
            self._timer.start()

    def recalc(self):
        if self.structure is None:
            return
        if getattr(self, "_quiet", False):       # start / last session: shown, calculated on request only
            self.curve = (None, None)
            self.draw()
            return
        try:
            q = self.q_model()
            I = self.model(q)
        except Exception as exc:  # noqa: BLE001
            self.fit_status.setText(f"<span style='color:#c8283c'>{exc}</span>")
            return
        self.curve = (q, I)
        self.draw()
        self._sketch_stale = True
        if self.sketch_follow.isChecked() and self._sketch_visible():
            self._sketch_timer.start()

    def draw(self):
        fig = self.canvas.figure
        fig.clear()
        d = self.data
        if d is not None:
            ax, axr = fig.subplots(2, 1, sharex=True, height_ratios=[3, 1])
        else:
            ax, axr = fig.add_subplot(111), None
        if d is not None:
            ax.errorbar(d["q"], d["I"], d["s"], fmt="o", ms=3, color="0.35", ecolor="0.7", label=d["name"], zorder=1)
        q, I = getattr(self, "curve", (None, None))
        if q is not None:
            ax.plot(q, I, "-", color="#6B8E23", lw=1.8, label="model", zorder=2)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_ylabel("I(q)")
        if ax.get_legend_handles_labels()[0]:
            ax.legend(loc="best", fontsize="small")
        if q is None:
            ax.text(0.5, 0.5, "press Calculate to compute I(q)" if self.structure is not None else "",
                    transform=ax.transAxes, ha="center", va="center", color="0.5")
        ax.grid(True, which="both", alpha=0.25)
        if axr is not None and q is not None:
            try:
                r = (d["I"] - self.model(d["q"])) / d["s"]
                axr.plot(d["q"], r, ".", color="#6B8E23", ms=3)
                axr.axhline(0, color="0.5", lw=0.8)
                axr.set_ylabel("(I − model)/σ")
                self.chi2 = float(np.sum(r ** 2) / max(1, r.size - len(self._free())))
                ax.set_title(f"χ²_red = {self.chi2:.4g}", fontsize="small")
            except Exception:  # noqa: BLE001
                pass
            axr.set_xlabel("q")
        else:
            ax.set_xlabel("q")
        self.canvas.draw_idle()

    # ------------------------------------------------------------------ data
    def on_load(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Load data (q, I, σ)", "",
                                                        "Data (*.dat *.txt *.csv *.xy);;All files (*)")
        if path:
            self.load_file(path)

    def load_file(self, path):
        from ..core.io import load_data
        try:
            q, I, s, _dq = load_data(path)
        except Exception as exc:  # noqa: BLE001
            QtWidgets.QMessageBox.critical(self, "Load data", f"{path}:\n{exc}")
            return False
        self.data = {"q": q, "I": I, "s": s, "path": os.path.abspath(path), "name": os.path.basename(path)}
        self.data_label.setText(f"{self.data['name']}: {q.size} points, q = {q.min():.4g} … {q.max():.4g}")
        self.data_label.setStyleSheet("")
        self.log(f"data {path}")
        self.recalc()
        return True

    def remove_data(self):
        self.data = None
        self.data_label.setText("no data (File → Load data, Ctrl+L)")
        self.data_label.setStyleSheet("color: gray;")
        self.recalc()

    # ------------------------------------------------------------------ fit
    def _free(self):
        return [k for k, d in self.param_rows().items() if d["fit"]]

    def fit(self, wait=False):
        """Fit the ticked parameters; ``wait``: in this thread (scripts, tests)."""
        if self.data is None:
            QtWidgets.QMessageBox.information(self, "Fit", "Load data first (File → Load data).")
            return None
        if self._fit_running:
            return None
        free = self._free()
        if not free:
            QtWidgets.QMessageBox.information(self, "Fit", "Tick at least one parameter in the 'fit' column.")
            return None
        rows = self.param_rows()
        args = (self.structure, dict(self.values()), free, [rows[k]["lo"] for k in free], [rows[k]["hi"] for k in free],
                self.mode.currentIndex() == 0, self.data, int(prefs.get("seb.fit.max_nfev")))
        self._fit_running = True
        self.fit_button.setEnabled(False)
        self.fit_status.setText("fitting …")
        if wait:
            res, err = _run_fit(*args)
            self._on_fitted(res, err)
            return res

        def work():
            res, err = _run_fit(*args)
            self._signals.fitted.emit(res, err)
        threading.Thread(target=work, daemon=True).start()
        return None

    def _on_fitted(self, res, err):
        self._fit_running = False
        self.fit_button.setEnabled(True)
        if err:
            self.fit_status.setText("<span style='color:#c8283c'>fit failed</span>")
            self.log("fit failed:\n" + err)
            return
        self.table.blockSignals(True)
        rows = self._row_of()
        for k, v in res["values"].items():
            r = rows.get(k)
            if r is None:
                continue
            self.table.item(r, 1).setText(_fmt(v))
            e = res["errors"].get(k)
            self.table.item(r, 5).setText("" if e is None else _fmt(e, 3))
        self.table.blockSignals(False)
        self.fit_status.setText(f"χ²_red = {res['chi2']:.4g} ({res['nfev']} evaluations)")
        self.log(f"fit: χ²_red = {res['chi2']:.5g}, {res['nfev']} evaluations, {res['message']}\n" +
                 "\n".join(f"   {k} = {_fmt(res['values'][k])} ± {_fmt(res['errors'].get(k), 3)}" for k in res["free"]))
        self.last_fit = res
        self.recalc()

    # ------------------------------------------------------------------ files
    def on_open_code(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Open structure code", "", "Python (*.py);;All files (*)")
        if path:
            with open(path, encoding="utf-8") as fh:
                self.template.blockSignals(True)
                self.template.setCurrentText("Custom")
                self.template.blockSignals(False)
                self.on_template("Custom", build=False)
                self._design_code = None
                self.build(fh.read())

    def on_save_code(self):
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save structure code", "structure.py", "Python (*.py)")
        if path:
            with open(path, "w", encoding="utf-8") as fh:
                fh.write(self.editor.toPlainText())

    def on_export(self):
        if getattr(self, "curve", None) is None:
            return
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Export model curve", "model.dat", "Text (*.dat *.txt)")
        if path:
            self.export_curve(path)

    def export_curve(self, path):
        q, I = self.curve
        head = "pySEBpolymers model\n" + "\n".join(f"{k} = {_fmt(v)}" for k, v in self.values().items()) + "\nq I"
        np.savetxt(path, np.column_stack([q, I]), header=head)

    # ------------------------------------------------------------------ 3D sketch
    def _sketch_visible(self):
        d = self.docks.get("sketch") if hasattr(self, "docks") else None
        return d is not None and (d.isVisible() and not d.visibleRegion().isEmpty() or "sketch" in self.detached)

    def draw_sketch(self, new=False):
        """Draw one random realisation of the structure into the 3D sketch panel."""
        if self.structure is None:
            return
        from . import sketch
        if new:
            self._seed += 1
        fig = self.sketch_canvas.figure
        fig.clear()
        ax = fig.add_subplot(111, projection="3d")
        try:
            shapes = sketch.realise(self.structure, self.values(), seed=self._seed, max_units=self.sketch_max.value())
            colours = sketch.draw(ax, shapes)
        except Exception as exc:  # noqa: BLE001
            ax.set_title(f"no sketch: {exc}", fontsize="small")
            self.sketch_canvas.draw_idle()
            return
        counts = sketch.tag_counts(shapes)
        from matplotlib.lines import Line2D
        handles = [Line2D([], [], color=c, lw=3, label=f"{t} ({counts.get(t, 0)})") for t, c in colours.items()]
        if handles:
            ax.legend(handles=handles, loc="upper left", fontsize="small", frameon=False)
        n, N = sum(counts.values()), len(self.structure.units())
        ax.set_title("one random realisation" + (f" ({n} of {N} sub-units)" if n < N else ""), fontsize="small")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
        self._sketch_stale = False
        self.sketch_canvas.draw_idle()

    # ------------------------------------------------------------------ design files
    def on_open_design(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Open design", "",
                                                        "pySEBpolymers design (*.sebdesign *.json);;All files (*)")
        if path:
            self.open_design(path)

    def open_design(self, path):
        from .graph import Design
        try:
            with open(path, encoding="utf-8") as fh:
                d = Design.from_json(fh.read())
        except Exception as exc:  # noqa: BLE001
            QtWidgets.QMessageBox.critical(self, "Open design", f"{path}:\n{exc}")
            return False
        self.template.blockSignals(True)
        self.template.setCurrentText("Custom")
        self.template.blockSignals(False)
        self.on_template("Custom", build=False)
        self.designer.set_design(d)
        return self.on_design(d)

    def on_save_design(self):
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save design", f"{self.designer.root.name}.sebdesign",
                                                        "pySEBpolymers design (*.sebdesign)")
        if path:
            self.save_design(path)

    def save_design(self, path):
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(self.designer.root.to_json())

    # ------------------------------------------------------------------ Python code
    def python_code(self):
        from ..core import pycode
        if self.structure is None:
            raise ValueError("no structure built")
        sc = pycode.Script("pySEBpolymers", "form factor of a structure of linked sub-units")
        sc.imp("import numpy as np", "from pysasfit.seb import Structure")
        sc.comment("structure (code of the Structure panel)")
        sc.add(self.code_built.rstrip())
        n_mu = int(prefs.get("seb.n_mu"))
        if n_mu != 128:
            sc.add(f"s.set_quadrature({n_mu})   # Preferences: orientational average")
        sc.blank()
        sc.add("params = " + pycode.lit(self.values()))
        norm = self.mode.currentIndex() == 0
        if self.data is not None:
            sc.imp("from pysasfit.core.io import load_data")
            sc.add(f"q_data, I_data, s_data, _dq = load_data({pycode.path_lit(self.data['path'])})")
        q = self.q_model()
        sc.add(f"q = np.geomspace({q[0]!r}, {q[-1]!r}, {q.size})")
        sc.add(f"I = params['scale'] * s.form_factor(q, params, normalized={norm}) + params['background']")
        if self.data is not None:
            sc.add(f"model = params['scale'] * s.form_factor(q_data, params, normalized={norm}) + params['background']",
                   "chi2_red = np.sum(((I_data - model) / s_data) ** 2) / max(1, q_data.size - "
                   f"{len(self._free())})",
                   'print(f"chi2_red = {chi2_red:.5g}")')
        plot = ["fig, ax = plt.subplots()"]
        if self.data is not None:
            plot.append("ax.errorbar(q_data, I_data, s_data, fmt='o', ms=3, color='0.35', label='data')")
        plot += ["ax.loglog(q, I, '-', label='model')", "ax.set_xlabel('q')", "ax.set_ylabel('I(q)')", "ax.legend()"]
        sc.add_plot(*plot)
        return sc.text()

    # ------------------------------------------------------------------ session
    def log(self, text):
        import datetime
        self.log_view.appendPlainText(f"[{datetime.datetime.now():%H:%M:%S}] {text}")

    @staticmethod
    def _session_path():
        return os.path.join(os.path.dirname(prefs.settings_path()), "seb_session.json")

    def session_state(self):
        rows = {k: {kk: (None if not np.isfinite(vv) else vv) if isinstance(vv, float) else vv for kk, vv in d.items()}
                for k, d in self.param_rows().items()}
        return {"template": self.template.currentText(), "options": self.options(),
                "code": self.code_built or self.editor.toPlainText(), "params": rows,
                "design": self.designer.root.to_dict(),
                "source": "design" if self.code_built is not None and self.code_built == self._design_code else "code",
                "mode": self.mode.currentIndex(), "qmin": self.qmin.text(), "qmax": self.qmax.text(),
                "nq": self.nq.value(), "data": self.data["path"] if self.data else None,
                "dock_state": self.dock_state()}

    def apply_state(self, st, calculate=True):
        """Put back a saved state; ``calculate=False`` (last session at the start): no calculation."""
        quiet = getattr(self, "_quiet", False)
        self._quiet = not calculate
        try:
            self._apply_state(st)
        finally:
            self._quiet = quiet

    def _apply_state(self, st):
        self.template.blockSignals(True)
        self.template.setCurrentText(st.get("template") or "Custom")
        self.template.blockSignals(False)
        self.on_template(self.template.currentText(), build=False)
        self.set_options(st.get("options"))
        self.mode.setCurrentIndex(int(st.get("mode", 0)))
        self.qmin.setText(str(st.get("qmin", "0.001")))
        self.qmax.setText(str(st.get("qmax", "0.5")))
        self.nq.setValue(int(st.get("nq", 400)))
        rows = {k: {kk: (np.inf if kk == "hi" else -np.inf) if vv is None and kk in ("lo", "hi") else vv
                    for kk, vv in d.items()} for k, d in (st.get("params") or {}).items()}
        ok = False
        if st.get("design") and st.get("source") == "design":
            from .graph import Design
            try:
                d = Design.from_dict(st["design"])
                self.designer.set_design(d)
                ok = self.on_design(d)
            except Exception:  # noqa: BLE001 -- fall back to the code
                ok = False
        if not ok:
            self.editor.setPlainText(st.get("code") or "")
            ok = self.build()
        if ok:
            self._fill_table(rows)
        if st.get("data") and os.path.isfile(st["data"]):
            self.load_file(st["data"])
        self.recalc()

    def save_session(self):
        try:
            p = self._session_path()
            os.makedirs(os.path.dirname(p), exist_ok=True)
            with open(p, "w", encoding="utf-8") as fh:
                json.dump(self.session_state(), fh, indent=1)
        except (OSError, TypeError, ValueError):
            pass

    def restore_session(self):
        try:
            with open(self._session_path(), encoding="utf-8") as fh:
                st = json.load(fh)
        except (OSError, ValueError):
            return False
        try:
            self.apply_state(st, calculate=False)     # Calculate (button) computes the curve
            self._session_layout = bool(self.restore_dock_state(st.get("dock_state")))
        except Exception:  # noqa: BLE001 -- a broken session must not stop the tool
            self.log("last session could not be restored:\n" + traceback.format_exc())
            return False
        return True

    def forget_session(self):
        try:
            os.remove(self._session_path())
        except OSError:
            pass

    def shutdown(self):
        self.save_session()

    def closeEvent(self, event):
        self.save_session()
        super().closeEvent(event)


def _run_fit(structure, values, free, lo, hi, normalized, data, max_nfev):
    """Least-squares fit (scipy, trust region reflective). Returns (result dict, None) or (None, traceback)."""
    try:
        from scipy.optimize import least_squares
        q, I, s = data["q"], data["I"], data["s"]
        v = dict(values)

        def resid(x):
            v.update(zip(free, x))
            F = structure.form_factor(q, v, normalized=normalized)
            return (v["scale"] * F + v["background"] - I) / s

        x0 = np.array([values[k] for k in free], dtype=float)
        lo_, hi_ = np.array(lo, dtype=float), np.array(hi, dtype=float)
        x0 = np.minimum(np.maximum(x0, lo_), hi_)
        r = least_squares(resid, x0, bounds=(lo_, hi_), x_scale="jac", max_nfev=max_nfev)
        dof = max(1, q.size - len(free))
        chi2 = float(np.sum(r.fun ** 2) / dof)
        try:
            cov = np.linalg.pinv(r.jac.T @ r.jac) * chi2
            err = np.sqrt(np.clip(np.diag(cov), 0, None))
        except np.linalg.LinAlgError:
            err = np.full(len(free), np.nan)
        out = dict(values)
        out.update(zip(free, r.x))
        return {"values": out, "free": free, "errors": dict(zip(free, err)), "chi2": chi2, "nfev": int(r.nfev),
                "message": r.message, "success": bool(r.success)}, None
    except Exception:  # noqa: BLE001
        return None, traceback.format_exc()


def main(argv=None):
    argv = sys.argv if argv is None else argv
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(argv)
    app.setWindowIcon(tool_icon("pySEBpolymers"))
    w = MainWindow()
    w.show()
    if len(argv) > 1 and os.path.isfile(argv[1]):
        w.load_file(argv[1])
    else:
        w.restore_session()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
