# -*- coding: utf-8 -*-
"""
Rogers-Young (RY) polydisperse Yukawa structure factor notebook tab.

Qt port of pyOZgui's Tk ry_polydisperse_yukawa_tab.py. This is a
SELF-CONTAINED QWidget subclass. It calculates exact structure factors
using the Rogers-Young closure approximation and compares them against various
polydispersity approximation schemes.

    from pysasfit.oz.gui.ry_polydisperse_yukawa_tab import RYPolydisperseYukawaTab
    tab = RYPolydisperseYukawaTab()
    someTabWidget.addTab(tab, "RY Polydisperse Yukawa")

It can also be run on its own for testing:

    python -m pysasfit.oz.gui.ry_polydisperse_yukawa_tab

The Rogers-Young closure is an alternative to MSA that can reach higher
concentrations and different interaction regimes.
"""
import os
import queue
import sys
import threading
import traceback

import matplotlib
import numpy as np
from PySide6 import QtCore, QtWidgets

from . import qtkit
from .qtkit import asksaveasfilename, showerror, showwarning
from .tab_controls import PolydisperseTabControls
from ...core import settings as prefs


def _solverChoices():
    """Fixpoint strategies offered in this tab's Solver dropdown.

    Only the ones that exist in this installation are listed. Order is the
    measured preference on this problem: SUNDIALS KIN_FP is fastest and, on
    hard states, fails cleanly instead of converging onto a spurious root;
    scipy Anderson converges where plain Picard diverges; Picard is the
    fallback of last resort but is the most robust at extreme dilution.
    """
    choices = {}
    try:
        from ..sundials4pyKinsolFPOZsolver import Sundials4pyKinsolFPOZsolver
        choices["SUNDIALS KIN_FP (Anderson)"] = Sundials4pyKinsolFPOZsolver
    except Exception:
        pass
    try:
        from ..scipyAndersonOZsolver import ScipyAndersonOZsolver
        choices["scipy Anderson"] = ScipyAndersonOZsolver
    except Exception:
        pass
    try:
        #MDIIS in place of the hand-written "Anderson", matching ozLib's own
        #SOLVER_CLASSES: that one duplicated "scipy Anderson" -- the same
        #algorithm, hand-written rather than from scipy -- while MDIIS adds
        #explicit regularisation and history control.
        #
        #NOT A SPEED RECOMMENDATION. Benchmarked on the real grid it is third
        #of nine at 0.0096 s, behind SUNDIALS KIN_FP (0.0049) and scipy
        #Anderson (0.0077), and it FAILS at phi = 0.58 where both of those
        #converge. An earlier note here claimed it was faster than scipy
        #Anderson at every density tried; that came from a sandbox
        #measurement at a quarter the grid size with SUNDIALS absent, and was
        #wrong in the flattering direction.
        #
        #Its place is as the alternative to try when scipy Anderson stalls on
        #an installation WITHOUT SUNDIALS -- scipy Anderson's iteration count
        #swings erratically with density (28, 116, 40, 179) where this one
        #rises smoothly, and at phi = 0.52 it is the better of the two.
        from ..mdiisOZsolver import MDIISOZsolver
        choices["MDIIS"] = MDIISOZsolver
    except Exception:
        pass
    try:
        from ..picardOZsolver import PicardOZsolver
        choices["Picard / Mann"] = PicardOZsolver
    except Exception:
        pass
    return choices


IMPORT_ERROR = None
try:
    from .. import rypolydisperseWrapper as rypw
    from ..polydisperse_yukawa_sas import Beta, CoreShell, Gamma, LogNormal, SchulzZimm, Sphere, discretize
except Exception as _exc:                                  # pragma: no cover
    IMPORT_ERROR = _exc


# Approximation schemes
APPROX_SCHEMES = [
    ("0  monodisperse",              "I_monodisperse"),
    ("1  decoupling (Kotlarchyk-Chen)", "I_decoupling"),
    ("2  local monodisperse (Pedersen)", "I_lma"),
    ("3  partial structure factors", "I_partial_sf"),
    ("4  scaling (Gazzillo)",        "I_scaling"),
    ("5  van der Waals one-fluid",   "I_vdw1"),
]

# Size distribution parameters
DIST_PARAMS = {
    "Log-normal": [
        ("Mean radius", "R0",    "50.0", "in the same length unit as 1/z"),
        ("Rel. s.d.",   "relsd", "0.20", None),
    ],
    "Schulz-Zimm": [
        ("Mean radius", "R0",    "50.0", "in the same length unit as 1/z"),
        ("Rel. s.d.",   "relsd", "0.20", None),
    ],
    "Gamma": [
        ("Shape k",  "k",     "25.0", None),
        ("Scale th", "theta", "2.0",
         "mean = k*th,  rel.s.d. = 1/sqrt(k)\n"
         "same family as Schulz-Zimm: k = z+1, th = <R>/(z+1)"),
    ],
    "Beta": [
        ("R min", "Rmin", "20.0", None),
        ("R max", "Rmax", "90.0", "bounded support: 0 < R min < R max"),
        ("a",     "a",    "2.0",  None),
        ("b",     "b",    "5.0",  "shape parameters, both > 0"),
    ],
}
DIST_NAMES = list(DIST_PARAMS)

PLOT_TABS = [
    ("S(Q)",            "sq"),
    ("S_ij(Q)",         "sij"),
    ("Size distribution", "dist"),
]

# Core radius / outer radius used for the "Core-shell" form factor. Same
# default as the MSA polydisperse Yukawa tab's "core fraction" entry. See the
# comment in _computeWorker: this tab computes S(Q) only, so the value does
# not enter any number shown here.
_CORE_SHELL_RATIO = 0.8


class _Result:
    """Everything one Compute produces."""
    def __init__(self):
        self.Q = None
        self.S = None           # (nQ, N, N)
        self.S_number = None
        self.R = None
        self.w = None
        self.sigma = None
        # Rogers-Young mixing parameter, SOLVED for (not user input) by
        # requiring the compressibility and virial pressure routes to agree.
        self.alpha = None
        self.alpha_residual = None
        self.alpha_converged = False
        self.summary = ""
        self.label = ""


class _Interrupted(Exception):
    """Raised inside the worker when the user pressed Interrupt."""


def _makeDistribution(name, pars):
    """Build the distribution object named by the dropdown."""
    if name == "Log-normal":
        s = float(np.sqrt(np.log(1.0 + pars["relsd"]**2)))
        # "Mean radius" IS THE MEAN, as for Schulz-Zimm and as in the generic
        # tab and polydisperse_nodes. LogNormal(R0, s) takes the MEDIAN
        # (<R> = R0 e^{s^2/2}); pyOZgui passed the entry straight in, so the
        # actual mean came out sqrt(1 + relsd^2) too large (2% at relsd 0.2).
        median = pars["R0"]/np.sqrt(1.0 + pars["relsd"]**2)
        return LogNormal(median, s)
    if name == "Schulz-Zimm":
        return SchulzZimm(pars["R0"], relsd=pars["relsd"])
    if name == "Gamma":
        return Gamma(pars["k"], pars["theta"])
    if name == "Beta":
        return Beta(pars["Rmin"], pars["Rmax"], pars["a"], pars["b"])
    raise ValueError(f"unknown distribution '{name}'")


def _explain(exc):
    """Turn common failures into plain language."""
    msg = str(exc)
    if "library not found" in msg.lower():
        return ("The RY Polydisperse Yukawa library could not be loaded.\n\n"
                "Make sure the plugin is properly built and installed.")
    return f"{type(exc).__name__}: {msg}"


def _viridis(n):
    """n colours from viridis without importing pyplot (Tk used plt.cm)."""
    return matplotlib.colormaps["viridis"](np.linspace(0, 1, n))


class RYPolydisperseYukawaTab(PolydisperseTabControls, QtWidgets.QWidget):
    #Session file identity -- see GenericPolydisperseTab. Index 2 matches the
    #position in the notebook: the OZ solver tab is 0, the generic
    #polydisperse tab 1, this one 2. The numbering counts TABS ON SCREEN,
    #not entries in EXTRA_TABS, because it exists for the user to read.
    SESSION_TAB_INDEX = 2
    SESSION_TAB_NAME = "RY Polydisperse Yukawa"

    def __init__(self, parent=None):
        super().__init__(parent)

        if IMPORT_ERROR is not None:
            self._buildImportErrorNotice()
            return

        self.result = None
        self.resultQueue = queue.Queue()
        self.worker = None
        # Set by the Interrupt button, read by the worker between solves
        # (see _trackedSolverClass / _computeWorker).
        self._interruptRequested = False
        self.solver = None
        # Replaces the Tk after(100, self._pollResultQueue) loop.
        self._poller = qtkit.QueuePoller(self, self.resultQueue, self._handleQueueItem)

        lay = QtWidgets.QHBoxLayout(self)
        lay.setContentsMargins(4, 4, 4, 4)
        splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal)
        lay.addWidget(splitter)
        splitter.addWidget(self._buildControls())
        splitter.addWidget(self._buildPlots())
        splitter.setStretchFactor(1, 1)
        self._syncEnabledStates()

    # ------------------------------------------------------------------
    def _buildImportErrorNotice(self):
        msg = ("The RY Polydisperse Yukawa solver modules could not be imported,\n"
               "so this tab is inactive. The Ornstein-Zernike solver tab is\n"
               "unaffected.\n\n"
               f"{type(IMPORT_ERROR).__name__}: {IMPORT_ERROR}\n\n"
               "Expected in pysasfit.oz:\n"
               "    polydisperse_yukawa_sas.py\n"
               "    polydisperse_sas_base.py\n"
               "    rypolydisperseWrapper.py")
        lay = QtWidgets.QVBoxLayout(self)
        lay.addWidget(qtkit.label(msg, color="#a00"))
        lay.addStretch(1)

    # ------------------------------------------------------------------
    def _buildControls(self):
        # Footer (buttons, status) below a scrollable parameter block, so a
        # taller block cannot push the buttons off-screen.
        outer = QtWidgets.QWidget()
        outerLay = QtWidgets.QVBoxLayout(outer)
        outerLay.setContentsMargins(6, 6, 6, 6)
        body, g = qtkit.grid()
        row = 0
        right = QtCore.Qt.AlignmentFlag.AlignRight
        leftAlign = QtCore.Qt.AlignmentFlag.AlignLeft

        def header(text, r):
            g.addWidget(qtkit.separator(), r, 0, 1, 2)
            g.addWidget(qtkit.heading(text), r + 1, 0, 1, 2)
            return r + 2

        def entry(label, var, r, width=10, tip=None):
            g.addWidget(QtWidgets.QLabel(label), r, 0, right)
            g.addWidget(qtkit.entry(var, width), r, 1, leftAlign)
            if tip:
                g.addWidget(qtkit.label(tip, color="#666", small=True), r + 1, 0, 1, 2)
                return r + 2
            return r + 1

        # ---- size distribution ----
        row = header("Size distribution", row)
        self.distVar = qtkit.StringVar(value="Log-normal")
        g.addWidget(QtWidgets.QLabel("Type:"), row, 0, right)
        self.distBox = qtkit.combo(self.distVar, DIST_NAMES, width=14,
                                   on_select=self._syncEnabledStates)
        g.addWidget(self.distBox, row, 1, leftAlign)
        row += 1

        # Host for the per-distribution entries; _rebuildDistParams replaces
        # its content (Tk destroyed and re-created the children of a Frame).
        self.distParamFrame = QtWidgets.QWidget()
        QtWidgets.QVBoxLayout(self.distParamFrame).setContentsMargins(0, 0, 0, 0)
        self._distParamInner = None
        g.addWidget(self.distParamFrame, row, 0, 1, 2)
        self.distVars = {}
        for name, params in DIST_PARAMS.items():
            for _lab, key, default, _tip in params:
                self.distVars.setdefault((name, key), qtkit.StringVar(value=default))
        row += 1

        self.nbinsVar = qtkit.StringVar(value="12")
        row = entry("Bins N:", self.nbinsVar, row)

        # ---- form factor ----
        row = header("Form factor", row)
        self.ffVar = qtkit.StringVar(value="Sphere")
        g.addWidget(QtWidgets.QLabel("Type:"), row, 0, right)
        self.ffBox = qtkit.combo(self.ffVar, ["Sphere", "Core-shell"], width=14)
        g.addWidget(self.ffBox, row, 1, leftAlign)
        row += 1

        # ---- interaction (RY specific) ----
        row = header("Yukawa interaction (RY)", row)
        self.phiVar = qtkit.StringVar(value="0.15")
        row = entry("Volume fraction:", self.phiVar, row)
        self.zVar = qtkit.StringVar(value="0.06")
        row = entry("Screening z:", self.zVar, row)
        self.KVar = qtkit.StringVar(value="1.0")
        row = entry("Contact K:", self.KVar, row)
        # No RY alpha entry: alpha is not a free parameter. Rogers-Young fixes
        # it by requiring the compressibility and virial routes to the pressure
        # to agree, so it is SOLVED for on every Compute (see _computeWorker's
        # call to rypolydisperseWrapper.solve_alpha) and reported in the
        # Summary tab. Exposing it as an input invited inconsistent results.
        self.alphaVar = None

        # ---- Q range ----
        row = header("Q range", row)
        self.QminVar = qtkit.StringVar(value="1e-3")
        row = entry("Q min:", self.QminVar, row)
        self.QmaxVar = qtkit.StringVar(value="0.3")
        row = entry("Q max:", self.QmaxVar, row)
        self.nQVar = qtkit.StringVar(value="200")
        row = entry("Points:", self.nQVar, row)

        #DEFLATION ON AN UNPHYSICAL ROOT. This tab is the most likely of the
        #four to meet one: charged, polydisperse, and with a consistency
        #search that moves alpha around. The wrapper already screens
        #min S_NN(q) and raises -- strongly attractive states at appreciable
        #phi are the usual cause, and the closure genuinely has several
        #roots there -- but until now it could only refuse. Each retry
        #excludes one more root and re-solves with the SAME solver.
        self.deflateVar = qtkit.StringVar(value="0")
        row = entry("Deflate if S(Q)<0:", self.deflateVar, row)
        g.setRowStretch(row, 1)

        self._controlBody = body
        outerLay.addWidget(qtkit.scroll_area(body), 1)
        outerLay.addWidget(qtkit.separator())

        # ---- actions ----
        # Standard control set shared with the other polydisperse tabs (see
        # tab_controls.py): Compute / Interrupt, Save / Load, the export
        # selector, the (hidden) run-history list and the status line.
        #
        # Unlike the analytic MSA/RMSA and Robertus tabs, this one drives a
        # genuine iterative Ornstein-Zernike solve, so BOTH extras apply:
        #   - supportsInterrupt=True: the fixpoint loops poll
        #     solverInstance.isInterrupted between iterations, exactly as
        #     the OZ solver tab's own interrupt button relies on.
        #   - a solver dropdown: rypolydisperseWrapper accepts solverClass,
        #     and which fixpoint strategy is used matters a great deal here.
        #     Measured on this problem, scipy Anderson converges where Picard
        #     diverges, and SUNDIALS KIN_FP is faster still AND refuses hard
        #     states rather than converging onto a spurious root -- so the
        #     choice is worth exposing rather than hard-coding.
        footer = QtWidgets.QWidget()
        self._controlFooter = footer
        outerLay.addWidget(footer)
        self.buildStandardControls(footer, supportsInterrupt=True,
                                   solverChoices=_solverChoices(),
                                   statusWraplength=240)
        self.computeButton = self.computeBtn   # name used elsewhere in this file

        self._rebuildDistParams()
        return outer

    def _rebuildDistParams(self):
        lay = self.distParamFrame.layout()
        if self._distParamInner is not None:
            lay.removeWidget(self._distParamInner)
            self._distParamInner.hide()
            self._distParamInner.deleteLater()
        inner, g = qtkit.grid()
        name = self.distVar.get()
        r = 0
        for label, key, _default, tip in DIST_PARAMS.get(name, []):
            g.addWidget(QtWidgets.QLabel(label + ":"), r, 0, QtCore.Qt.AlignmentFlag.AlignRight)
            g.addWidget(qtkit.entry(self.distVars[(name, key)], 10), r, 1, QtCore.Qt.AlignmentFlag.AlignLeft)
            r += 1
            if tip:
                g.addWidget(qtkit.label(tip, color="#666", small=True), r, 0, 1, 2)
                r += 1
        g.setColumnStretch(1, 1)
        lay.addWidget(inner)
        self._distParamInner = inner

    def _syncEnabledStates(self):
        self._rebuildDistParams()

    # ------------------------------------------------------------------
    def _buildPlots(self):
        self.notebook = qtkit.PlotTabs()
        for label, key in PLOT_TABS:
            self._decorateAxes(self.notebook.add_plot(label, key), key)
        self.axes = self.notebook.axes
        self.canvases = self.notebook.canvases
        self.summaryText = self.notebook.add_text("Summary")
        return self.notebook

    @staticmethod
    def _decorateAxes(ax, key):
        if key == "sq":
            ax.set_xlabel("Q"); ax.set_ylabel("S(Q)  (number-number)")
            ax.set_xscale("log")
        elif key == "sij":
            ax.set_xlabel("Q"); ax.set_ylabel(r"$S_{ij}(Q)$")
            ax.set_xscale("log")
        elif key == "dist":
            ax.set_xlabel("R"); ax.set_ylabel("weight")

    # ------------------------------------------------------------------
    def _readInputs(self):
        def f(var, name, lo=None, hi=None):
            try:
                v = float(var.get())
            except ValueError:
                raise ValueError(f"{name}: '{var.get()}' is not a number")
            if lo is not None and v < lo:
                raise ValueError(f"{name} must be >= {lo}")
            if hi is not None and v > hi:
                raise ValueError(f"{name} must be <= {hi}")
            return v

        p = {}
        p["dist"] = name = self.distVar.get()
        limits = {"R0": (1e-12, None), "relsd": (1e-6, 1.0),
                  "k": (1e-6, None), "theta": (1e-12, None),
                  "Rmin": (None, None), "Rmax": (None, None),
                  "a": (1e-6, None), "b": (1e-6, None)}
        p["distpars"] = {}
        for label, key, _default, _tip in DIST_PARAMS[name]:
            lo, hi = limits.get(key, (None, None))
            p["distpars"][key] = f(self.distVars[(name, key)], label, lo=lo, hi=hi)
        if name == "Beta":
            dp = p["distpars"]
            if dp["Rmin"] <= 0:
                raise ValueError("Beta: R min must be greater than 0")
            if dp["Rmax"] <= dp["Rmin"]:
                raise ValueError("Beta: R max must be greater than R min")
        p["nbins"] = int(f(self.nbinsVar, "Bins N", lo=1, hi=64))
        p["phi"] = f(self.phiVar, "Volume fraction", lo=1e-6, hi=0.62)
        p["z"] = f(self.zVar, "Screening z", lo=1e-12)
        p["K"] = f(self.KVar, "Contact K")
        # p["alpha"] deliberately absent -- solved for, not read from the GUI.
        p["Qmin"] = f(self.QminVar, "Q min", lo=1e-12)
        p["Qmax"] = f(self.QmaxVar, "Q max", lo=1e-12)
        p["nQ"] = int(f(self.nQVar, "Points", lo=2, hi=5000))
        p["deflate"] = int(f(self.deflateVar, "Deflate if S(Q)<0",
                             lo=0, hi=5))
        if p["Qmax"] <= p["Qmin"]:
            raise ValueError("Q max must exceed Q min")
        # Form factor. This was missing, so _computeWorker's own
        # `if p["ff"] == "Core-shell"` raised KeyError: 'ff' on every single
        # Compute -- self.ffVar was built in _buildControls and read in the
        # worker, but never carried across in between.
        p["ff"] = self.ffVar.get()
        return p

    def _onCompute(self):
        if self.worker is not None and self.worker.is_alive():
            self.statusVar.set("a calculation is already running")
            return
        try:
            p = self._readInputs()
        except ValueError as exc:
            self.statusVar.set(str(exc))
            showerror("Invalid input", str(exc), parent=self)
            return
        # Read the dropdown on the GUI thread, not inside the worker.
        p["solverClass"] = self.selectedSolverClass()

        self._interruptRequested = False
        self.solver = None
        self.computeButton.setEnabled(False)
        # The Tk tab never enabled Interrupt, so it could not be pressed.
        self.interruptBtn.setEnabled(True)
        self.statusVar.set("solving...")
        self.worker = threading.Thread(target=self._computeWorker, args=(p,), daemon=True)
        self.worker.start()

    def _onInterrupt(self):
        # Two levels: the flag stops the worker between OZ solves (the alpha
        # search runs many of them), and the mixin sets isInterrupted on the
        # solver currently iterating, which the fixpoint loops poll.
        if self.worker is None or not self.worker.is_alive():
            self.statusVar.set("nothing to interrupt")
            return
        self._interruptRequested = True
        if self.solver is not None:
            super()._onInterrupt()
        else:
            self.statusVar.set("interrupt requested...")

    def _trackedSolverClass(self, base):
        """Subclass of `base` that registers each instance as self.solver.

        rypolydisperseWrapper constructs its solver internally, so this is the
        only way to get hold of the instance whose isInterrupted flag the
        fixpoint loop polls. Behaviour is otherwise that of `base`.
        """
        tab = self

        class _Tracked(base):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                tab.solver = self
                if tab._interruptRequested:
                    self.isInterrupted = True

        # The wrapper reports solverClass.__name__ in its messages.
        _Tracked.__name__ = base.__name__
        _Tracked.__qualname__ = base.__qualname__
        return _Tracked

    def _checkInterrupt(self):
        if self._interruptRequested:
            raise _Interrupted("interrupted by user")

    def _computeWorker(self, p):
        def progress(msg):
            self._checkInterrupt()
            self.resultQueue.put(("status", msg))

        try:
            res = _Result()
            dist = _makeDistribution(p["dist"], p["distpars"])

            if p["ff"] == "Core-shell":
                # BUG FIX (Tk original): CoreShell() raised "give exactly one
                # of thickness, ratio" on every Compute. This tab computes
                # S(Q) only -- no I(Q) -- and CoreShell.outer_radius(R) == R,
                # so the hard core is the same as for a sphere and the form
                # factor does not enter any number shown here. It is built
                # with the MSA tab's default core fraction so the choice is at
                # least valid and ready for an I(Q) extension.
                ff = CoreShell(ratio=_CORE_SHELL_RATIO)
            else:
                ff = Sphere()
            del ff  # not used further, see above

            progress("Building RY distribution...")
            # THE SELECTED DISTRIBUTION IS USED. pyOZgui's Tk tab built the
            # classes from a mean and a relative s.d. only -- evenly spaced
            # over +-3 s.d. with log-normal weights -- whatever distribution
            # was selected, so Schulz-Zimm, Gamma and Beta all gave the
            # log-normal answer (and relsd > 1/3 gave negative sizes). The
            # classes now come from the distribution's own quadrature
            # (discretize: Gauss-Legendre in log space for log-normal,
            # generalised Gauss-Laguerre for Schulz-Zimm/Gamma, Gauss-Jacobi
            # for Beta), exactly as in the MSA tab.
            #
            # The distributions are of the RADIUS R (the entries say "Mean
            # radius", "R min", "R max"); the RY solver takes hard-sphere
            # DIAMETERS. The Tk tab used the radius values as diameters,
            # i.e. particles half the size entered. sigma = 2R.
            ry_prefs = prefs.values("oz.ry")
            R, weights = discretize(dist, p["nbins"], nsig=ry_prefs["nsig"])
            sigmas = 2.0*R

            # Number densities from volume fraction
            sigma3 = np.sum(weights * sigmas**3)
            x_array = weights * p["phi"] * 6 / (np.pi * sigma3)

            progress("Computing RY structure factor...")
            # rypolydisperseWrapper's signature deliberately mirrors
            # PolydisperseOneYukawaMSA's: (sigma, rho, z, K, alpha, delta).
            # x_array above is already NUMBER DENSITIES (weights scaled by
            # phi*6/(pi*<sigma^3>)), so the volume fraction is contained in
            # them and must NOT be passed separately -- doing so shifted every
            # argument by one and put arrays where scalars were expected,
            # which is what raised "TypeError: only 0-dimensional arrays can
            # be converted to Python scalars" on every Compute.
            #
            # alpha is NOT a user input. Rogers-Young's whole point is that the
            # mixing parameter is FIXED by thermodynamic consistency: the
            # compressibility and virial routes to the pressure are required to
            # agree. solve_alpha() does that search (three OZ solves per trial
            # value, hence the progress messages).
            progress("Solving for thermodynamically consistent alpha...")
            # As in the Tk original, the alpha search uses the wrapper's
            # default solver, not the dropdown's. It is passed explicitly
            # (the wrapper's own first choice, which is exactly what it uses
            # for solverClass=None) only so that the instance can be tracked
            # for Interrupt.
            alphaSolver = self._trackedSolverClass(rypw._solverClasses()[0][1])
            alpha, alpha_res = rypw.solve_alpha(sigmas, x_array, p["z"], p["K"],
                                                bracket=(ry_prefs["alpha_lo"], ry_prefs["alpha_hi"]),
                                                tol=ry_prefs["alpha_tol"],
                                                progress=progress,
                                                solverClass=alphaSolver)
            self._checkInterrupt()
            res.alpha = alpha
            res.alpha_residual = alpha_res
            # A large leftover residual means no alpha in the bracket makes the
            # two routes agree -- report it rather than quietly presenting the
            # nearest endpoint as if it were a solution.
            res.alpha_converged = abs(alpha_res) < ry_prefs["alpha_accept"]
            res.alpha_bracket = (ry_prefs["alpha_lo"], ry_prefs["alpha_hi"])
            progress(f"alpha = {alpha:.4f}" if res.alpha_converged else
                     f"alpha = {alpha:.4f} (NOT consistent, residual {alpha_res:+.3g})")
            #PASS THE SELECTED SOLVER. The dropdown was built from
            #_solverChoices() and the user's choice read back by
            #selectedSolverClass(), and then not used -- every solve fell
            #through to rypolydisperseWrapper's own cascade, which starts at
            #scipy Anderson. So picking "SUNDIALS KIN_FP" in the interface
            #produced "scipy Anderson converged after N steps" in the log,
            #and the control had no effect at all.
            #
            #None is still meaningful and is what the wrapper expects when
            #no dropdown is offered: it then chooses from its own ordered
            #list with a fallback cascade.
            solverClass = p.get("solverClass")
            if solverClass is not None:
                solverClass = self._trackedSolverClass(solverClass)
            ry = rypw.RYPolydisperseYukawa(sigmas, x_array, p["z"], p["K"], alpha,
                                           solverClass=solverClass,
                                           deflateOnUnphysical=p.get("deflate", 0))
            self._checkInterrupt()

            Q = np.logspace(np.log10(p["Qmin"]), np.log10(p["Qmax"]), p["nQ"])
            res.Q = Q

            progress("Calculating S matrices...")
            S_list = []
            S_num_list = []
            for q in Q:
                S = ry.S_matrix(q)
                S_list.append(S)
                n_total = np.sum(x_array)
                w = x_array / n_total
                wv = np.sqrt(w)
                S_num = float(wv @ S @ wv)
                S_num_list.append(S_num)

            res.S = np.array(S_list)
            res.S_number = np.array(S_num_list)
            res.R = R
            res.w = weights
            res.sigma = sigmas

            lines = []
            lines.append(f"distribution        : {dist!r}")
            lines.append(f"components N        : {len(sigmas)}")
            lines.append(f"R range             : {R.min():.4g} .. {R.max():.4g}")
            lines.append(f"sigma range         : {sigmas.min():.4g} .. {sigmas.max():.4g}")
            lines.append(f"z*sigma range       : {p['z']*sigmas.min():.3f} .. {p['z']*sigmas.max():.3f}")
            lines.append(f"phi                 : {p['phi']:.4g}")
            lines.append(f"z                   : {p['z']:.4g}")
            lines.append(f"K                   : {p['K']:.4g}")
            lines.append(f"alpha (RY)          : {res.alpha:.4g}"
                         + ("  [thermodynamically consistent,"
                            f" residual {res.alpha_residual:+.2e}]"
                            if res.alpha_converged else
                            f"  [NO consistent alpha found in [{res.alpha_bracket[0]:g}, {res.alpha_bracket[1]:g}];"
                            f" best residual {res.alpha_residual:+.3g}]"))

            res.summary = "\n".join(lines)
            res.label = f"RY K={p['K']:.4f} alpha={res.alpha:.4f}"
            self.resultQueue.put(("result", res))
        except Exception as exc:
            if self._interruptRequested:
                self.resultQueue.put(("interrupted", None))
            else:
                print(traceback.format_exc(), file=sys.stderr)
                self.resultQueue.put(("error", _explain(exc)))

    def _handleQueueItem(self, item):
        """GUI-thread side of the worker queue (Tk: _pollResultQueue)."""
        kind, data = item
        if kind == "status":
            self.statusVar.set(data)
            return
        if kind == "result":
            self._displayResult(data)
        elif kind == "interrupted":
            self.statusVar.set("interrupted")
        elif kind == "error":
            self.statusVar.set("error")
            showerror("Computation failed", str(data), parent=self)
        # result / interrupted / error are the worker's last message.
        self.solver = None
        self.computeButton.setEnabled(True)
        self.interruptBtn.setEnabled(False)

    def _displayResult(self, res):
        # registerRun() (from PolydisperseTabControls) appends to the run
        # history and sets self.result, so Save / Export operate on it.
        self.registerRun(res, res.label)
        self._plotSQ()
        self._plotSij()
        self._plotDist()
        self.summaryText.setPlainText(res.summary)
        self.statusVar.set(res.label)

    def _replot(self):
        """Redraw everything from self.result (needed by the shared controls
        after Load / Clear all / Delete last)."""
        self._plotSQ()
        self._plotSij()
        self._plotDist()
        self.summaryText.setPlainText(self.result.summary if self.result is not None else "")

    def _plotSQ(self):
        ax = self.axes["sq"]
        ax.clear()
        self._decorateAxes(ax, "sq")
        res = self.result
        if res is not None and getattr(res, "S_number", None) is not None:
            ax.semilogx(res.Q, res.S_number, "k-", linewidth=2)
            ax.grid(True, alpha=0.3)
        self.canvases["sq"].draw_idle()

    def _plotSij(self):
        ax = self.axes["sij"]
        ax.clear()
        self._decorateAxes(ax, "sij")
        res = self.result
        if res is not None and getattr(res, "S", None) is not None:
            S = res.S
            nij = S.shape[1]
            colors = _viridis(nij * (nij + 1) // 2)
            idx = 0
            for i in range(nij):
                for j in range(i, nij):
                    ax.semilogx(res.Q, S[:, i, j], color=colors[idx], alpha=0.7,
                                label=f"S_{i}{j}" if nij <= 3 else None)
                    idx += 1
            if nij <= 3:
                ax.legend(loc="best", fontsize=8)
            ax.grid(True, alpha=0.3)
        self.canvases["sij"].draw_idle()

    def _plotDist(self):
        ax = self.axes["dist"]
        ax.clear()
        self._decorateAxes(ax, "dist")
        res = self.result
        if res is not None and getattr(res, "sigma", None) is not None:
            sigma = np.asarray(res.sigma, float)
            # np.diff of a single bin is empty (mean -> NaN width).
            width = 0.7*np.diff(sigma).mean() if sigma.size > 1 else 0.1*sigma[0]
            ax.bar(sigma / 2.0, res.w, width=width / 2.0, alpha=0.7, edgecolor="black")
            ax.grid(True, alpha=0.3, axis="y")
        self.canvases["dist"].draw_idle()

    def _onClear(self):
        self.result = None
        for ax in self.axes.values():
            ax.clear()
        for canvas in self.canvases.values():
            canvas.draw_idle()
        self.summaryText.clear()
        self.statusVar.set("cleared")

    def _exportColumns(self):
        """Columns for the shared exporter (ASCII/CSV/clipboard): the same
        two columns the Tk tab's own _onExport wrote."""
        r = self.selectedRun()
        return ["Q", "S_number"], [r.Q, r.S_number]

    def _onExport(self):
        if self.result is None:
            showwarning("No result", "No result to export.", parent=self)
            return
        path = asksaveasfilename(defaultextension=".csv",
                                 filetypes=[("CSV", "*.csv"), ("Excel", "*.xlsx"),
                                            ("All files", "*")], parent=self)
        if not path:
            return
        try:
            res = self.result
            if path.endswith(".xlsx"):
                try:
                    import openpyxl
                    wb = openpyxl.Workbook()
                    ws = wb.active
                    ws.append(["Q", "S_number"])
                    for q, s in zip(res.Q, res.S_number):
                        ws.append([float(q), float(s)])
                    wb.save(path)
                except ImportError:
                    showerror("Export failed", "openpyxl not installed", parent=self)
                    return
            else:
                with open(path, "w") as fd:
                    fd.write("Q,S_number\n")
                    for q, s in zip(res.Q, res.S_number):
                        fd.write(f"{q},{s}\n")
            self.statusVar.set(f"exported to {os.path.basename(path)}")
        except Exception as exc:
            showerror("Export failed", str(exc), parent=self)

    # ------------------------------------------------------------------
    # Session state (used by the shared Save... / Load... buttons). Not in
    # the Tk original, whose session files therefore restored nothing.
    _SESSION_VARS = ("distVar", "nbinsVar", "ffVar", "phiVar", "zVar", "KVar",
                     "QminVar", "QmaxVar", "nQVar", "deflateVar", "solverVar")

    def sessionState(self):
        state = {"entries": {}, "distParams": {}}
        for name in self._SESSION_VARS:
            var = getattr(self, name, None)
            if var is not None:
                state["entries"][name] = var.get()
        for (dname, key), var in self.distVars.items():
            state["distParams"].setdefault(dname, {})[key] = var.get()
        return state

    def restoreSessionState(self, state):
        for name, value in (state.get("entries") or {}).items():
            var = getattr(self, name, None)
            if var is None:
                continue
            if name == "solverVar" and value not in self.solverChoices:
                continue        # solver not available in this installation
            if name == "distVar" and value not in DIST_PARAMS:
                continue
            var.set(value)
        for dname, pars in (state.get("distParams") or {}).items():
            for key, value in pars.items():
                var = self.distVars.get((dname, key))
                if var is not None:
                    var.set(value)
        self._syncEnabledStates()


def main():                                                # pragma: no cover
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
    tab = RYPolydisperseYukawaTab()
    tab.setWindowTitle("RY Polydisperse Yukawa Tab Test")
    tab.resize(1150, 750)
    tab.show()
    return app.exec()


if __name__ == "__main__":                                 # pragma: no cover
    sys.exit(main())
