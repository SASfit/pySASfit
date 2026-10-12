# -*- coding: utf-8 -*-
"""The "0: OZ solver" tab: one-component Ornstein-Zernike solver with closures.

Qt port of the solver panel of pyOZgui's Tk ``oZgui.py`` (itself the Python
equivalent of SASfit's Tcl ``sasfit_OZ_solver.tcl``). Layout and behaviour are
unchanged:

  - left panel:  potential + closure dropdowns, parameter entries built by
                 introspecting the ``setXXXPotential()`` methods of
                 oZfixpointOperator (a new potential appears automatically),
                 volume fraction, solver, verification, deflation, x-axis
                 scale and threshold, grid size, points per sigma, maximum
                 iterations, label, calculate/interrupt/clear/delete,
                 Save/Load (.oz JSON), export of the selected/last run
                 (ASCII, CSV, Excel, PNG, clipboard) and the run history;
  - right panel: S(Q), g(r), c(r), Gamma(r), h(r), U(r)/kT, B(r), y(r),
                 f(r) plot tabs (one per curve of the original 9 BLT graph
                 tabs) and a Log tab that shows every status message and
                 everything printed to stdout/stderr.

The calculation itself is ``ozLib.solve()`` -- identical results to scripts
that call it directly. The solve runs in a worker thread; results and log
text reach the GUI thread through a queue drained by a QTimer (Qt widgets,
like Tk, must only be touched from the GUI thread).
"""

import datetime
import inspect
import json
import os
import queue
import re
import sys
import threading
import traceback

import numpy as np
from PySide6 import QtCore, QtWidgets

from . import qtkit
from .qtkit import askdirectory, askopenfilename, asksaveasfilename, showerror, showinfo
from .. import ozLib
from ..ozLib import CLOSURE_SETTERS, SOLVER_CLASSES, isMulticomponentPotential, multicomponentCapableClosures
from ..picardOZsolver import PicardOZsolver
from ..sign_convention import strength_hint

_HAVE_SUNDIALS4PY = ozLib._HAVE_SUNDIALS4PY

# Curves shown, one tab each: (tab label, key in ozLib's OZResult.curves, ylabel)
CURVE_TABS = [
    ("S(Q*sigma)", "Sq", "$S(Q\\sigma)$"),
    ("g(r)", "gr", "g(r)"),
    ("c(r)", "cr", "c(r)"),
    ("Gamma(r)", "gamma", "Γ(r)"),
    ("h(r)", "hr", "h(r)"),
    ("U(r)/kT", "Ur", "U(r) / kT"),
    ("B(r)", "Br", "B(r)"),
    ("y(r)", "yr", "y(r)"),
    ("f(r)", "fr", "f(r)"),
]

# Grid-size exponent n (gridsize = 2**n - 1) clamp range, as in the C/Tcl OZ
# solver GUI (sasfit_oz_tclcmd.c): the floor keeps the DST-I transform
# meaningfully sized, the ceiling guards against an astronomically large grid.
MIN_GRID_EXPONENT = 4
MAX_GRID_EXPONENT = 20


class RunResult:
    """One computed curve set in the run history."""

    def __init__(self, label, r, q, potential, closure, phi):
        self.label = label
        self.r = r
        self.q = q
        self.potential = potential
        self.closure = closure
        self.phi = phi
        self.curves = {}  # name -> y array, keys matching CURVE_TABS[*][1]


class StreamRedirector:
    """File-like object that tees writes into a queue (drained on the GUI thread
    into the Log tab) and to the original stream, so the many diagnostic
    print()s inside the solver modules reach the GUI even without a console."""

    def __init__(self, targetQueue, alsoWriteTo=None):
        self._queue = targetQueue
        self._alsoWriteTo = alsoWriteTo

    def write(self, text):
        if self._alsoWriteTo is not None:
            try:
                self._alsoWriteTo.write(text)
            except Exception:
                pass
        if text:
            self._queue.put(("log", text))

    def flush(self):
        if self._alsoWriteTo is not None:
            try:
                self._alsoWriteTo.flush()
            except Exception:
                pass

    def __getattr__(self, name):  # isatty, encoding, ...
        return getattr(self._alsoWriteTo, name)


def plt_colors():
    import matplotlib.pyplot as plt

    return plt.rcParams["axes.prop_cycle"].by_key()["color"]


class OZSolverTab(QtWidgets.QWidget):
    """Left control panel + right plot notebook of the one-component OZ solver."""

    def __init__(self, parent=None, redirect_output=True):
        super().__init__(parent)
        self.runs = []  # list of RunResult, oldest first
        self.colors = plt_colors()
        self.solver = None  # currently-running OZsolver instance
        self.worker = None
        self.resultQueue = queue.Queue()
        self._poller = qtkit.QueuePoller(self, self.resultQueue, self._handleQueueItem)

        # stdout/stderr -> Log tab (tee'd to the real streams). Restored in
        # restoreOutput(), which the main window calls when it closes.
        self._origStdout, self._origStderr = sys.stdout, sys.stderr
        self._redirected = False
        if redirect_output:
            sys.stdout = StreamRedirector(self.resultQueue, alsoWriteTo=sys.stdout)
            sys.stderr = StreamRedirector(self.resultQueue, alsoWriteTo=sys.stderr)
            self._redirected = True

        lay = QtWidgets.QHBoxLayout(self)
        lay.setContentsMargins(4, 4, 4, 4)
        splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal)
        lay.addWidget(splitter)
        splitter.addWidget(self._buildLeftPanel())
        splitter.addWidget(self._buildRightPanel())
        splitter.setStretchFactor(1, 1)

    def restoreOutput(self):
        if self._redirected:
            sys.stdout, sys.stderr = self._origStdout, self._origStderr
            self._redirected = False

    # ------------------------------------------------------------------
    def _buildLeftPanel(self):
        inner, g = qtkit.grid()
        inner.setContentsMargins(6, 6, 6, 6)
        row = 0

        def put(text, widget, span=False):
            nonlocal row
            if span:
                g.addWidget(widget, row, 0, 1, 2)
            else:
                g.addWidget(QtWidgets.QLabel(text), row, 0, QtCore.Qt.AlignmentFlag.AlignRight)
                g.addWidget(widget, row, 1, QtCore.Qt.AlignmentFlag.AlignLeft)
            row += 1

        # Potential names discovered by introspection: add a setXXXPotential()
        # method to oZfixpointOperator.py and it appears here automatically.
        self.probe = PicardOZsolver(port=0)
        self.potentialNames = sorted(self.probe.getAvailablePotentialNames())
        self.potentialVar = qtkit.StringVar(value="HardSphere")
        self.potentialCombo = qtkit.combo(self.potentialVar, self.potentialNames, width=22,
                                          on_select=self._onPotentialChanged)
        put("Potential:", self.potentialCombo)

        self.potParamsHost, self.potParamsGrid = qtkit.grid()
        put("", self.potParamsHost, span=True)
        self.potParamVars = []
        self._rebuildPotentialParams()

        self.closureVar = qtkit.StringVar(value="Percus-Yevick")
        self.closureCombo = qtkit.combo(self.closureVar, list(CLOSURE_SETTERS), width=22,
                                        on_select=self._rebuildClosureParam)
        put("Closure:", self.closureCombo)
        # Shown only while a polydisperse potential is selected: how many
        # closures are hidden and why.
        self.closureNoteVar = qtkit.StringVar(value="")
        put("", qtkit.label(var=self.closureNoteVar, color="grey", wrap=250), span=True)

        self.closureParamHost = QtWidgets.QWidget()
        self.closureParamGrid = QtWidgets.QGridLayout(self.closureParamHost)
        self.closureParamGrid.setContentsMargins(0, 0, 0, 0)
        put("", self.closureParamHost, span=True)
        self.closureParamVar = None
        self._rebuildClosureParam()

        self.phiVar = qtkit.StringVar(value="0.3")
        put("Volume fraction φ:", qtkit.entry(self.phiVar, 10))

        g.addWidget(QtWidgets.QLabel("Solver:"), row, 0, 1, 2)
        row += 1
        self.solverVar = qtkit.StringVar(value=next(iter(SOLVER_CLASSES)))
        solverWidth = max(len(name) for name in SOLVER_CLASSES) + 2
        put("", qtkit.combo(self.solverVar, list(SOLVER_CLASSES), width=solverWidth), span=True)

        # VERIFICATION, on by default: a converged residual is not evidence the
        # answer is right; a second solve is the only check that catches a
        # solver landing on a different root. Costs about eight times the work.
        self.verifyVar = qtkit.BooleanVar(value=True)
        put("", qtkit.check(self.verifyVar, "verify against a second solver"), span=True)

        # DEFLATION as a count of retries: zero (accept whatever the solver
        # returns) is as meaningful as a larger number.
        self.deflateVar = qtkit.StringVar(value="0")
        put("deflate if S(Q)<0:", qtkit.combo(self.deflateVar, ["0", "1", "2", "3"], width=4))

        self.xScaleVar = qtkit.StringVar(value="symlog")
        put("X-axis scale:", qtkit.combo(self.xScaleVar, ["symlog", "linear", "log", "asinh"], width=22,
                                         on_select=self._replot))

        # "auto": linthresh/linear_width from the smallest positive x plotted;
        # a number gives a visibly distinct linear region instead.
        self.xThresholdVar = qtkit.StringVar(value="auto")
        put("Symlog/asinh threshold:", qtkit.entry(self.xThresholdVar, 10, on_return=self._replot,
                                                   on_focus_out=self._replot))

        self.gridPointsVar = qtkit.StringVar(value="auto")
        self.gridSizeLabelVar = qtkit.StringVar(value="")
        put("Gridsize (2^n - 1), n:", qtkit.hbox(qtkit.entry(self.gridPointsVar, 10),
                                                  qtkit.label(var=self.gridSizeLabelVar, color="grey")))
        self.gridPointsVar.trace_add("write", self._updateGridSizeLabel)
        self._updateGridSizeLabel()

        # Gridsize sets the total array (with points per sigma: the real-space
        # range); points per sigma sets the near-contact resolution. Increasing
        # n at fixed points per sigma extends the range -- needed for slowly
        # decaying potentials (small-kappa Yukawa/DLVO tails).
        self.pointsPerSigmaVar = qtkit.StringVar(value="auto")
        put("Points per σ:", qtkit.entry(self.pointsPerSigmaVar, 10))

        self.maxIterVar = qtkit.StringVar(value="1000")
        put("Max iterations:", qtkit.entry(self.maxIterVar, 10))

        # Empty by default so the auto label "<n>: <potential>, <closure>, phi=.."
        # applies from the first calculation on.
        self.labelVar = qtkit.StringVar(value="")
        put("Label:", qtkit.entry(self.labelVar, 22))

        self.calcBtn = qtkit.button("calculate", self._onCalculate)
        self.interruptBtn = qtkit.button("interrupt", self._onInterrupt, enabled=False)
        put("", qtkit.hbox(self.calcBtn, self.interruptBtn, qtkit.button("clear all", self._onClear),
                           qtkit.button("delete last", self._onDeleteLast)), span=True)
        put("", qtkit.hbox(qtkit.button("Save...", self._onSaveAll), qtkit.button("Load...", self._onLoadAll)),
            span=True)

        g.addWidget(QtWidgets.QLabel("Export selected/last run:"), row, 0, 1, 2)
        row += 1
        self.exportFormatVar = qtkit.StringVar(value="ASCII")
        self._exportDispatch = {
            "ASCII": self._onExportASCII,
            "CSV": self._onExportCSV,
            "Excel": self._onExportExcel,
            "PNG (all tabs)": self._onExportPNG,
            "Copy to clipboard": self._onCopyToClipboard,
        }
        put("", qtkit.hbox(qtkit.combo(self.exportFormatVar, list(self._exportDispatch), width=16),
                           qtkit.button("Export", self._onExportSelected)), span=True)

        g.addWidget(QtWidgets.QLabel("Run history:"), row, 0, 1, 2)
        row += 1
        self.historyList = QtWidgets.QListWidget()
        self.historyList.setMinimumHeight(140)
        put("", self.historyList, span=True)

        # Every statusVar.set() is time-stamped into the Log tab by this trace.
        self.statusVar = qtkit.StringVar(value="ready")
        self.statusVar.trace_add("write", self._onStatusChanged)
        put("", qtkit.label(var=self.statusVar, color="blue", wrap=300), span=True)
        g.setRowStretch(row, 1)
        g.setColumnStretch(1, 1)
        return qtkit.scroll_area(inner, min_width=inner.sizeHint().width() + 30)

    def _updateGridSizeLabel(self, *args):
        # Mirrors updateOZgridsizeLabel in sasfit_OZ_solver.tcl (same clamp
        # range, same message shape).
        text = self.gridPointsVar.get().strip().lower()
        if text == "auto":
            self.gridSizeLabelVar.set(f"= {2**12 - 1} points (auto, n=12)")
            return
        try:
            n = int(text)
        except ValueError:
            self.gridSizeLabelVar.set("(enter an integer n, e.g. 12, or 'auto')")
            return
        nClamped = min(max(n, MIN_GRID_EXPONENT), MAX_GRID_EXPONENT)
        gridsize = 2**nClamped - 1
        if nClamped != n:
            self.gridSizeLabelVar.set(f"= {gridsize} points (n will be clamped to {nClamped})")
        else:
            self.gridSizeLabelVar.set(f"= {gridsize} points")

    @staticmethod
    def _clearLayout(lay):
        while lay.count():
            item = lay.takeAt(0)
            w = item.widget()
            if w is not None:
                w.deleteLater()

    def _rebuildPotentialParams(self, probe=None):
        probe = probe or self.probe
        self._clearLayout(self.potParamsGrid)
        self.potParamVars = []
        methodName = "set" + self.potentialVar.get() + "Potential"
        spec = inspect.getfullargspec(getattr(probe, methodName))
        argNames = spec[0][1:]
        # Seed each box with the SETTER'S OWN default where it has one (e.g.
        # chargeExponent = 2.0), not a blanket 1.0 that would silently change
        # the model. A default of None (e.g. meanDiameter) falls back to 1.0,
        # which for that argument IS the hard-sphere diameter in reduced units.
        nDefaults = len(spec.defaults) if spec.defaults else 0
        firstWithDefault = len(argNames) - nDefaults
        for i, name in enumerate(argNames):
            initial = "1.0"
            if i >= firstWithDefault:
                value = spec.defaults[i - firstWithDefault]
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    initial = repr(float(value))
                elif isinstance(value, bool):
                    initial = str(value)
            var = qtkit.StringVar(value=initial)
            label, box = QtWidgets.QLabel(name + ":"), qtkit.entry(var, 10)
            hint = strength_hint(self.potentialVar.get(), i)
            if hint:
                #Sign convention A, stated where the number is typed.
                label.setText(name + " (+ = attr.):")
                label.setToolTip(hint)
                box.setToolTip(hint)
            self.potParamsGrid.addWidget(label, i, 0, QtCore.Qt.AlignmentFlag.AlignRight)
            self.potParamsGrid.addWidget(box, i, 1, QtCore.Qt.AlignmentFlag.AlignLeft)
            self.potParamVars.append(var)
        if not argNames:
            self.potParamsGrid.addWidget(QtWidgets.QLabel("(no parameters)"), 0, 0)

    def _onPotentialChanged(self, probe=None):
        """Rebuild the potential's parameter fields AND narrow the closure list:
        a polydisperse potential cannot be combined with closures that need a
        one-component reference solve (ozLib.MULTICOMPONENT_INCAPABLE_CLOSURES)."""
        self._rebuildPotentialParams(probe)
        self._refreshClosureChoices()

    def _refreshClosureChoices(self):
        allNames = list(CLOSURE_SETTERS.keys())
        if isMulticomponentPotential(self.potentialVar.get()):
            allowed = multicomponentCapableClosures()
            hidden = len(allNames) - len(allowed)
            self.closureNoteVar.set(
                f"{hidden} closures hidden: they need a one-component reference solve or are hard-sphere specific."
            )
        else:
            allowed = allNames
            self.closureNoteVar.set("")
        qtkit.set_combo_values(self.closureCombo, allowed, self.closureVar)
        # A closure no longer offered falls back to Percus-Yevick rather than
        # leaving a stale name in the box.
        if self.closureVar.get() not in allowed:
            previous = self.closureVar.get()
            self.closureVar.set("Percus-Yevick")
            self.statusVar.set(f"'{previous}' is not available for a polydisperse potential; switched to Percus-Yevick")
            self._rebuildClosureParam()

    def _rebuildClosureParam(self):
        self._clearLayout(self.closureParamGrid)
        closureName = self.closureVar.get()
        _, needsParam = CLOSURE_SETTERS[closureName]
        self.closureParamVar = None
        self.closureParamEntry = None
        self.closureParam2Var = None
        self.closureParam2Entry = None
        # Recreated every time so _onCalculate can read it unconditionally.
        self.findConsistentVar = qtkit.BooleanVar(value=False)
        g = self.closureParamGrid
        if needsParam:
            self.closureParamVar = qtkit.StringVar(value="1.0")
            self.closureParamEntry = qtkit.entry(self.closureParamVar, 10)
            g.addWidget(QtWidgets.QLabel("α / η:"), 0, 0, QtCore.Qt.AlignmentFlag.AlignRight)
            g.addWidget(self.closureParamEntry, 0, 1, QtCore.Qt.AlignmentFlag.AlignLeft)
        # A second closure scalar (Extended Rogers-Young's quadratic
        # coefficient a), declared generically in ozLib.SECOND_CLOSURE_PARAM.
        extra = ozLib.secondClosureParam(closureName)
        if extra is not None:
            name, default, description = extra
            self.closureParam2Var = qtkit.StringVar(value=repr(float(default)))
            self.closureParam2Entry = qtkit.entry(self.closureParam2Var, 10)
            g.addWidget(QtWidgets.QLabel(name + ":"), 1, 0, QtCore.Qt.AlignmentFlag.AlignRight)
            g.addWidget(self.closureParam2Entry, 1, 1, QtCore.Qt.AlignmentFlag.AlignLeft)
            g.addWidget(qtkit.label(description, color="grey", wrap=240), 2, 0, 1, 2)
        # Closures that can search for the parameter making the compressibility
        # and virial routes agree (Rogers & Young's consistency idea). The
        # manual entry is disabled (not hidden) while the search is on.
        if closureName in ozLib.CONSISTENT_PARAMETER_CLOSURES:

            def onToggleConsistent():
                if self.closureParamEntry is not None:
                    self.closureParamEntry.setEnabled(not self.findConsistentVar.get())

            g.addWidget(qtkit.check(self.findConsistentVar, "find thermodynamically consistent value",
                                    command=onToggleConsistent), 3, 0, 1, 2)

    # ------------------------------------------------------------------
    def _buildRightPanel(self):
        self.notebook = qtkit.PlotTabs()
        for tabLabel, key, ylabel in CURVE_TABS:
            ax = self.notebook.add_plot(tabLabel, key)
            ax.set_xlabel(r"$Q\sigma$" if key == "Sq" else r"$r/\sigma$")
            ax.set_ylabel(ylabel)
            ax.set_xscale("symlog", linthresh=1.0)
        self.axes = self.notebook.axes
        self.canvases = self.notebook.canvases
        self._buildLogTab()
        return self.notebook

    def _buildLogTab(self):
        page = QtWidgets.QWidget()
        lay = QtWidgets.QVBoxLayout(page)
        lay.setContentsMargins(2, 2, 2, 2)
        lay.addWidget(qtkit.hbox(qtkit.button("Clear log", self._onClearLog),
                                 qtkit.button("Save log...", self._onSaveLog)))
        self.logText = qtkit.text_view(mono=False, wrap=True)
        lay.addWidget(self.logText, 1)
        self.notebook.addTab(page, "Log")

    def _appendLog(self, text):
        # GUI thread only (status trace, or the queue poller).
        cursor = self.logText.textCursor()
        cursor.movePosition(cursor.MoveOperation.End)
        cursor.insertText(text)
        self.logText.setTextCursor(cursor)
        self.logText.ensureCursorVisible()

    def _onStatusChanged(self, *_traceArgs):
        timestamp = datetime.datetime.now().strftime("%H:%M:%S")
        self._appendLog(f"[{timestamp}] {self.statusVar.get()}\n")

    def _onClearLog(self):
        self.logText.clear()

    def _onSaveLog(self):
        content = self.logText.toPlainText()
        if not content.strip():
            showinfo("nothing to save", "the log is empty")
            return
        path = asksaveasfilename(defaultextension=".txt", filetypes=[("ASCII text", "*.txt"), ("All files", "*.*")],
                                 initialfile="oz_log.txt", title="Save log")
        if not path:
            return
        try:
            with open(path, "w") as f:
                f.write(content)
        except OSError as e:
            showerror("save failed", str(e))

    # ------------------------------------------------------------------
    def _onCalculate(self):
        if self.worker is not None and self.worker.is_alive():
            showinfo("busy", "a calculation is already running")
            return
        try:
            phi = float(self.phiVar.get())
            maxIter = int(self.maxIterVar.get())
            potArgs = [float(v.get()) for v in self.potParamVars]
        except ValueError:
            showerror("input error", "potential parameters, phi and max iterations must be numbers")
            return

        # "auto" keeps ozLib.solve()'s default grid (4095 points, 100 per
        # sigma); an integer n means gridsize = 2**n - 1 (clamped).
        gridKwargs = {}
        gridPointsText = self.gridPointsVar.get().strip().lower()
        if gridPointsText != "auto":
            try:
                parsedN = int(gridPointsText)
                nClamped = min(max(parsedN, MIN_GRID_EXPONENT), MAX_GRID_EXPONENT)
                gridKwargs["numberOfRadialSamplingPoints"] = 2**nClamped - 1
            except ValueError:
                pass
        pointsPerSigmaText = self.pointsPerSigmaVar.get().strip().lower()
        if pointsPerSigmaText != "auto":
            try:
                parsedHsp = int(pointsPerSigmaText)
                if parsedHsp > 0:
                    gridKwargs["hardSphereDiameterInPoints"] = parsedHsp
            except ValueError:
                pass

        potentialName = self.potentialVar.get()
        closureName = self.closureVar.get()
        _, needsParam = CLOSURE_SETTERS[closureName]
        closureParam = None
        closureParam2 = None
        findConsistent = self.findConsistentVar.get()
        if needsParam and not findConsistent:
            try:
                closureParam = float(self.closureParamVar.get())
            except ValueError:
                showerror("input error", "closure parameter must be a number")
                return
        # The second closure scalar is read even with the consistency search
        # on: the search determines alpha, not a.
        if self.closureParam2Var is not None:
            try:
                closureParam2 = float(self.closureParam2Var.get())
            except ValueError:
                extra = ozLib.secondClosureParam(closureName)
                showerror("input error", f"closure parameter '{extra[0]}' must be a number")
                return

        solverName = self.solverVar.get()
        # Deflation on an unphysical root (S(Q) < 0) with THE SAME solver.
        try:
            deflateAttempts = int(self.deflateVar.get())
        except ValueError:
            deflateAttempts = 0
        verifyResult = bool(self.verifyVar.get())
        autoLabel = f"{len(self.runs)+1}: {potentialName}, {closureName}, phi={phi:.3g}"
        label = self.labelVar.get() or autoLabel

        self.calcBtn.setEnabled(False)
        self.interruptBtn.setEnabled(True)
        self.statusVar.set("calculating...")

        def work():
            try:

                def captureSolverInstance(solverInstance):
                    # lets "interrupt" reach the live solver
                    self.solver = solverInstance

                result = ozLib.solve(potential=potentialName, phi=phi, potentialArgs=potArgs, closure=closureName,
                                     closureParam=closureParam, closureParam2=closureParam2,
                                     findConsistentParameter=findConsistent, solver=solverName,
                                     maxIterations=maxIter, deflateOnUnphysical=deflateAttempts, verify=verifyResult,
                                     onSolverCreated=captureSolverInstance, **gridKwargs)
                run = RunResult(label, result.r, result.q, potentialName, closureName, phi)
                run.curves = result.curves
                self.resultQueue.put(("done", run))
            except Exception:
                self.resultQueue.put(("error", traceback.format_exc()))

        self.worker = threading.Thread(target=work, daemon=True)
        self.worker.start()

    def _handleQueueItem(self, item):
        kind, payload = item
        if kind == "done":
            self._onCalculateDone(payload)
        elif kind == "log":
            self._appendLog(payload)
        else:
            self._onCalculateError(payload)

    def _onCalculateDone(self, run):
        self.runs.append(run)
        self.historyList.addItem(run.label)
        self._replot()
        self.calcBtn.setEnabled(True)
        self.interruptBtn.setEnabled(False)
        self.statusVar.set(f"done: {run.label}")
        self.labelVar.set("")

    def _onCalculateError(self, tb):
        self.calcBtn.setEnabled(True)
        self.interruptBtn.setEnabled(False)
        self.statusVar.set("error (see Log tab)")
        print(tb)
        showerror("calculation failed", tb.strip().splitlines()[-1])

    def _onInterrupt(self):
        if self.solver is not None:
            self.solver.isInterrupted = True
        self.statusVar.set("interrupt requested...")

    def _onClear(self):
        self.runs = []
        self.historyList.clear()
        self._replot()
        self.statusVar.set("cleared")

    def _onDeleteLast(self):
        if self.runs:
            self.runs.pop()
            self.historyList.takeItem(self.historyList.count() - 1)
            self._replot()

    def _onSaveAll(self):
        if not self.runs:
            showinfo("nothing to save", "run at least one calculation first")
            return
        path = asksaveasfilename(defaultextension=".oz",
                                 filetypes=[("Ornstein-Zernike GUI data", "*.oz"), ("All files", "*.*")],
                                 title="Save all plots")
        if not path:
            return
        data = {
            "format": "sasfit_oz_gui_save_v1",
            "runs": [
                {
                    "label": run.label,
                    "potential": run.potential,
                    "closure": run.closure,
                    "phi": run.phi,
                    "r": np.asarray(run.r).tolist(),
                    "q": np.asarray(run.q).tolist() if run.q is not None else None,
                    "curves": {name: np.asarray(arr).tolist() for name, arr in run.curves.items()},
                }
                for run in self.runs
            ],
        }
        try:
            # allow_nan (default) is relied on: B(r)/y(r) contain NaN inside the core.
            with open(path, "w") as f:
                json.dump(data, f)
            self.statusVar.set(f"saved {len(self.runs)} run(s) to {path}")
        except OSError as e:
            showerror("save failed", str(e))

    def _onLoadAll(self):
        path = askopenfilename(filetypes=[("Ornstein-Zernike GUI data", "*.oz"), ("All files", "*.*")],
                               title="Load plots")
        if not path:
            return
        try:
            with open(path, "r") as f:
                data = json.load(f)
            if data.get("format") != "sasfit_oz_gui_save_v1":
                showerror("load failed", "unrecognised file format")
                return
            for runData in data["runs"]:
                qArr = np.array(runData["q"]) if runData.get("q") is not None else None
                run = RunResult(runData["label"], np.array(runData["r"]), qArr, runData["potential"],
                                runData["closure"], runData["phi"])
                run.curves = {name: np.array(arr) for name, arr in runData["curves"].items()}
                self.runs.append(run)
                self.historyList.addItem(run.label)
            self._replot()
            self.statusVar.set(f"loaded {len(data['runs'])} run(s) from {path}")
        except (OSError, json.JSONDecodeError, KeyError) as e:
            showerror("load failed", str(e))

    def _getSelectedOrLastRun(self):
        items = self.historyList.selectedItems()
        if items:
            return self.runs[self.historyList.row(items[0])]
        if self.runs:
            return self.runs[-1]
        return None

    def _buildTableText(self, run, delimiter, commentedHeader):
        # One column per curve; r[i] and q[i] are the same transform grid point
        # in real and reciprocal space. q and S(Q) are named "qsigma"/"Sqsigma"
        # since both are dimensionless Q*sigma quantities.
        keys = [key for _, key, _ in CURVE_TABS]
        headerNames = ["Sqsigma" if key == "Sq" else key for key in keys]
        header = delimiter.join(["r", "qsigma"] + headerNames)
        lines = [("# " if commentedHeader else "") + header]
        for i in range(len(run.r)):
            row = [f"{run.r[i]:.10g}", f"{run.q[i]:.10g}" if run.q is not None else ""]
            for key in keys:
                y = run.curves.get(key)
                val = y[i] if y is not None else float("nan")
                row.append(f"{val:.10g}")
            lines.append(delimiter.join(row))
        return "\n".join(lines) + "\n"

    def _writeTable(self, path, run, delimiter, commentedHeader):
        with open(path, "w") as f:
            f.write(self._buildTableText(run, delimiter, commentedHeader))

    def _onExportSelected(self):
        handler = self._exportDispatch.get(self.exportFormatVar.get())
        if handler is not None:
            handler()

    def _onExportASCII(self):
        run = self._getSelectedOrLastRun()
        if run is None:
            showinfo("nothing to export", "run at least one calculation first")
            return
        path = asksaveasfilename(defaultextension=".txt", filetypes=[("ASCII text", "*.txt"), ("All files", "*.*")],
                                 initialfile=run.label + ".txt", title="Export as ASCII")
        if not path:
            return
        try:
            self._writeTable(path, run, delimiter="\t", commentedHeader=True)
            self.statusVar.set(f"exported '{run.label}' to {path} (ASCII)")
        except OSError as e:
            showerror("export failed", str(e))

    def _onCopyToClipboard(self):
        run = self._getSelectedOrLastRun()
        if run is None:
            showinfo("nothing to copy", "run at least one calculation first")
            return
        # Tab-separated without '#' header: pastes cleanly into spreadsheets.
        qtkit.clipboard_set(self._buildTableText(run, delimiter="\t", commentedHeader=False))
        self.statusVar.set(f"copied '{run.label}' to clipboard ({len(run.r)} rows, tab-separated)")

    def _onExportCSV(self):
        run = self._getSelectedOrLastRun()
        if run is None:
            showinfo("nothing to export", "run at least one calculation first")
            return
        path = asksaveasfilename(defaultextension=".csv", filetypes=[("CSV", "*.csv"), ("All files", "*.*")],
                                 initialfile=run.label + ".csv", title="Export as CSV")
        if not path:
            return
        try:
            self._writeTable(path, run, delimiter=",", commentedHeader=False)
            self.statusVar.set(f"exported '{run.label}' to {path} (CSV)")
        except OSError as e:
            showerror("export failed", str(e))

    def _onExportExcel(self):
        run = self._getSelectedOrLastRun()
        if run is None:
            showinfo("nothing to export", "run at least one calculation first")
            return
        try:
            import openpyxl
        except ImportError:
            showerror("missing dependency", "Excel export needs the 'openpyxl' package.\n"
                                            "Install it with: pip install openpyxl")
            return
        path = asksaveasfilename(defaultextension=".xlsx",
                                 filetypes=[("Excel workbook", "*.xlsx"), ("All files", "*.*")],
                                 initialfile=run.label + ".xlsx", title="Export as Excel")
        if not path:
            return
        try:
            wb = openpyxl.Workbook()
            ws = wb.active
            ws.title = re.sub(r"[\\/*?:\[\]]", "_", run.label)[:31] or "OZ data"
            keys = [key for _, key, _ in CURVE_TABS]
            headerNames = ["Sqsigma" if key == "Sq" else key for key in keys]
            ws.append(["r", "qsigma"] + headerNames)
            for i in range(len(run.r)):
                row = [float(run.r[i]), float(run.q[i]) if run.q is not None else None]
                for key in keys:
                    y = run.curves.get(key)
                    val = float(y[i]) if y is not None else None
                    # openpyxl cannot write NaN: empty cell instead
                    row.append(None if (val is not None and val != val) else val)
                ws.append(row)
            wb.save(path)
            self.statusVar.set(f"exported '{run.label}' to {path} (Excel)")
        except OSError as e:
            showerror("export failed", str(e))

    def _onExportPNG(self):
        if not self.runs:
            showinfo("nothing to export", "run at least one calculation first")
            return
        directory = askdirectory(title="Choose a folder for the PNG files (one per tab)")
        if not directory:
            return
        saved = []
        try:
            for tabLabel, key, ylabel in CURVE_TABS:
                fname = os.path.join(directory, f"oz_{key}.png")
                self.axes[key].figure.savefig(fname, dpi=150, bbox_inches="tight")
                saved.append(fname)
            self.statusVar.set(f"saved {len(saved)} PNG file(s) to {directory}")
        except OSError as e:
            showerror("export failed", str(e))

    def _replot(self):
        xScale = self.xScaleVar.get()
        manualThreshold = None
        thresholdText = self.xThresholdVar.get().strip().lower()
        if thresholdText != "auto":
            try:
                parsedThreshold = float(thresholdText)
                if parsedThreshold > 0:
                    manualThreshold = parsedThreshold
            except ValueError:
                pass

        for tabLabel, key, ylabel in CURVE_TABS:
            ax = self.axes[key]
            ax.clear()
            ax.set_xlabel(r"$Q\sigma$" if key == "Sq" else r"$r/\sigma$")
            ax.set_ylabel(ylabel)
            xMinPositive = None
            for i, run in enumerate(self.runs):
                y = run.curves.get(key)
                if y is None:
                    continue
                color = self.colors[i % len(self.colors)]
                # S(Q) lives on the q grid, everything else on the r grid.
                x = run.q if key == "Sq" and run.q is not None else run.r
                ax.plot(x, y, label=run.label, color=color, linewidth=1.2)
                positive = x[x > 0]
                if positive.size:
                    xMinPositive = positive.min() if xMinPositive is None else min(xMinPositive, positive.min())
            threshold = manualThreshold if manualThreshold is not None else (xMinPositive if xMinPositive else 1.0)
            if xScale == "symlog":
                ax.set_xscale("symlog", linthresh=threshold)
            elif xScale == "asinh":
                # smooth linear-to-log transition (SASfit's historical arcsinh axis)
                ax.set_xscale("asinh", linear_width=threshold)
            else:
                ax.set_xscale(xScale)
            if self.runs:
                ax.legend(fontsize=8)
            self.canvases[key].draw_idle()
