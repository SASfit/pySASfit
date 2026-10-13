# -*- coding: utf-8 -*-
"""
Polydisperse hard-sphere-Yukawa (MSA / RMSA) notebook tab for oZgui.py.

Qt port of pyOZgui's Tk polydisperse_yukawa_tab.py. This is a SELF-CONTAINED
QWidget subclass. It knows nothing about the main window and touches none of
its state, so adding it to the GUI is one line in
pysasfit.oz.gui.main_window.EXTRA_TABS and removing it again is one line back.
The existing Ornstein-Zernike solver tab is untouched by design.

    from pysasfit.oz.gui.polydisperse_yukawa_tab import PolydisperseYukawaTab
    tabs.addTab(PolydisperseYukawaTab(), "Polydisperse Yukawa")

It can also be run on its own for testing:

    python -m pysasfit.oz.gui.polydisperse_yukawa_tab

What it exposes, all of it already implemented and validated elsewhere in this
package (see README_polydisperse_yukawa.md and docs/report/report.pdf):

  - a continuous size distribution (log-normal or Schulz-Zimm) discretised
    onto N components,
  - a form factor (homogeneous sphere or core-shell),
  - the analytic N-component one-Yukawa MSA closure and the exact partial
    structure factors S_ij(Q) that follow from it,
  - the rescaled closure (RMSA), with one common rescaling factor lambda
    chosen so that min_ij g_ij = 0,
  - the exact I(Q), and all six of SASfit's approximate schemes for combining
    a structure factor with a size distribution, so the error of each can be
    read straight off the plot.

SIGN CONVENTION, worth stating on screen and stating here: the MSA closure
sets c(r) = -beta U(r) outside the core, so K > 0 is an ATTRACTIVE tail and
K < 0 is a REPULSIVE one. This is the opposite of the sign convention some
SASfit plugins use for the Yukawa amplitude.

AI_USAGE.md note: the physics modules this drives were derived and verified by
the authors. The only new physics-adjacent code here is _apply_rmsa(), which
swaps a solved rescaled system into an already-constructed SAS object; it is
assembly of validated pieces rather than a new derivation, but it is flagged
here as wanting a look before it is relied on for published numbers.
"""
import os
import queue
import sys
import threading
import traceback

import numpy as np
from PySide6 import QtCore, QtWidgets

from . import qtkit
from ...core import settings as prefs

# The solver stack lives in the parent package. Import failure must not take
# the whole GUI down -- the tab shows a placeholder notice instead.
IMPORT_ERROR = None
try:
    from ..polydisperse_yukawa_sas import (LogNormal, SchulzZimm, Gamma, Beta,
                                           Sphere, CoreShell,
                                           PolydisperseYukawaSAS, moment_report)
    from ..polydisperse_rmsa import solve_rmsa
    from ..polydisperse_yukawa_sq import PolydisperseOneYukawaSq
except Exception as _exc:                                  # pragma: no cover
    IMPORT_ERROR = _exc


# Approximation schemes, in SASfit's own numbering. The label is what the user
# sees; the attribute is the method on PolydisperseSASBase.
APPROX_SCHEMES = [
    ("0  monodisperse",              "I_monodisperse"),
    ("1  decoupling (Kotlarchyk-Chen)", "I_decoupling"),
    ("2  local monodisperse (Pedersen)", "I_lma"),
    ("3  partial structure factors", "I_partial_sf"),
    ("4  scaling (Gazzillo)",        "I_scaling"),
    ("5  van der Waals one-fluid",   "I_vdw1"),
]

# Parameters shown for each distribution: (label, key, default, hint).
# The panel rebuilds itself from this when the dropdown changes, so adding a
# distribution is a matter of adding a class in polydisperse_yukawa_sas.py,
# an entry here, and a branch in _makeDistribution().
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

# The two independent solution routes of polydisperse_yukawa_msa.py. These are
# NOT iteration schemes -- the closure is solved in closed form -- so there is
# nothing here corresponding to the OZ solver tab's Picard/Anderson/KINSOL
# choice. They are two different algebraic reductions of the same equations,
# and they agree to machine precision; running both is a self-check.
ROUTE_GAMMA = "Gamma (eq.72)"
ROUTE_AMPL = "amplitudes (eq.55)"
ROUTE_BOTH = "both (cross-check)"
ROUTES = [ROUTE_GAMMA, ROUTE_AMPL, ROUTE_BOTH]

PLOT_TABS = [
    ("I(Q)",            "iq"),
    ("Approx. error",   "err"),
    ("S_ij(Q)",         "sij"),
    ("S(Q)",            "sq"),
    ("Size distribution", "dist"),
]


class _Result:
    """Everything one Compute produces, handed from the worker thread to the
    main thread through a queue (Qt widgets are not thread-safe -- same
    pattern, and for the same reason, as OZgui._pollResultQueue)."""

    def __init__(self):
        self.Q = None
        self.I_exact = None
        self.approx = {}        # label -> I(Q)
        self.S = None           # (nQ, N, N)
        self.S_number = None
        self.R = None
        self.w = None
        self.sigma = None
        self.g = None           # contact matrix
        self.summary = ""
        self.label = ""


def _apply_rmsa(sas, K, lam_max=None):
    """Swap a solved RMSA system into an already-built PolydisperseYukawaSAS.

    S_partials() and hence I_exact() route through sas.sq, so replacing
    sas.solver/sas.sol/sas.sq with the rescaled ones is enough to put every
    downstream quantity on the RMSA footing. The number densities sas.rho and
    the radii sas.R are physical and deliberately NOT rescaled -- the form
    factor and the concentration are what they are; only the closure's
    effective hard core moves.

    K is passed explicitly rather than read from sas.solver, because for the
    RMSA path the SAS object is deliberately built at K = 0 (see
    _buildSAS): the whole point of RMSA is the regime where the plain MSA has
    no root at all, and PolydisperseYukawaSAS.__init__ solves the closure
    eagerly, so constructing it at the true K would raise before we ever got
    here.

    Returns the dict from solve_rmsa (lam, min_g, phi, phi_rescaled, status).
    """
    res = solve_rmsa(sas.solver.sigma, sas.solver.rho,
                     sas.solver.z, K, lam_max=lam_max)
    if res.get("lam") is None:
        return res
    sas.solver = res["solver"]
    sas.sol = res["sol"]
    sas.sq = PolydisperseOneYukawaSq(res["solver"], res["sol"])
    sas.__dict__.pop("_mono_cache", None)
    return res


def _makeDistribution(name, pars, momentRule=False):
    """Build the distribution object named by the dropdown from the parameter
    dict the control panel produced."""
    if name == "Log-normal":
        # the panel asks for the relative s.d. of R; convert to the log-space
        # width the class takes:  relsd = sqrt(e^{s^2} - 1)
        s = float(np.sqrt(np.log(1.0 + pars["relsd"]**2)))
        # "Mean radius" IS THE MEAN, as for Schulz-Zimm and as in the generic
        # tab and polydisperse_nodes. LogNormal(R0, s) takes the MEDIAN
        # (<R> = R0 e^{s^2/2}); pyOZgui passed the entry straight in, so the
        # actual mean came out sqrt(1 + relsd^2) too large (2% at relsd 0.2).
        median = pars["R0"]/np.sqrt(1.0 + pars["relsd"]**2)
        return LogNormal(median, s,
                         rule="moment" if momentRule else "legendre")
    if name == "Schulz-Zimm":
        return SchulzZimm(pars["R0"], relsd=pars["relsd"])
    if name == "Gamma":
        return Gamma(pars["k"], pars["theta"])
    if name == "Beta":
        return Beta(pars["Rmin"], pars["Rmax"], pars["a"], pars["b"])
    raise ValueError(f"unknown distribution '{name}'")


def _explain(exc):
    """Turn the common, physically-meaningful failures into plain language.
    These are properties of the closure, not bugs: the MSA simply has no
    physical solution everywhere in the (phi, z, K) space."""
    msg = str(exc)
    if "no root for Gamma" in msg:
        return ("The MSA closure has no physical solution at this state point.\n"
                "\n"
                "This is a property of the mean-spherical approximation, not a\n"
                "numerical failure. It happens for strong coupling -- typically\n"
                "near the spinodal for attraction (K > 0), or at high volume\n"
                "fraction. Things to try:\n"
                "  - reduce |K| (weaker tail),\n"
                "  - reduce the volume fraction,\n"
                "  - increase the screening z (shorter-ranged tail),\n"
                "  - for strong REPULSION (K < 0), switch the closure to RMSA,\n"
                "    which rescales the core until the solution is physical.\n"
                "    RMSA does not rescue strong ATTRACTION.")
    if msg.startswith("RMSA:"):
        return ("RMSA could not find a rescaling factor.\n\n" + msg + "\n\n"
                "The rescaling inflates the hard cores until every g_ij is\n"
                "non-negative; if no factor up to the search limit achieves\n"
                "that, the state point is out of reach of this closure.")
    if "mpmath" in msg:
        return ("The moment-exact log-normal nodes need the mpmath package:\n\n"
                "    pip install mpmath\n\n"
                "Untick 'moment-exact nodes' to use the default rule instead.")
    if "undefined" in msg:
        return ("An approximate scheme has no solution in this state.\n\n" + msg)
    return f"{type(exc).__name__}: {msg}"


def _hint(text):
    """Small grey help text (the Tk code's 7-pt '#666' labels)."""
    return qtkit.label(text, color="#666", small=True)


def _unbindOnDestroy(var, widget):
    """qtkit.check() does not unregister its variable binding when the widget
    is destroyed (qtkit.entry() does). The distribution panel is rebuilt on
    every dropdown change, so without this a later var.set() would reach a
    deleted QCheckBox."""
    def drop(*_a):
        var._widget_setters[:] = [s for s in var._widget_setters
                                  if getattr(s, "_source", None) is not widget]
    widget.destroyed.connect(drop)


class PolydisperseYukawaTab(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)

        if IMPORT_ERROR is not None:
            self._buildImportErrorNotice()
            return

        self.result = None
        self.resultQueue = queue.Queue()
        self.worker = None
        # QueuePoller replaces the Tk after(100, _pollResultQueue) loop.
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
        msg = ("The polydisperse Yukawa solver modules could not be imported,\n"
               "so this tab is inactive. The Ornstein-Zernike solver tab is\n"
               "unaffected.\n\n"
               f"{type(IMPORT_ERROR).__name__}: {IMPORT_ERROR}\n\n"
               "Expected in pysasfit.oz:\n"
               "    polydisperse_yukawa_msa.py\n"
               "    polydisperse_yukawa_sq.py\n"
               "    polydisperse_yukawa_sas.py\n"
               "    polydisperse_sas_base.py\n"
               "    polydisperse_rmsa.py")
        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(20, 20, 20, 20)
        lay.addWidget(qtkit.label(msg, color="#a00"))
        lay.addStretch(1)

    # ------------------------------------------------------------------
    def _buildControls(self):
        # The control panel is taller than the window on a default-sized
        # display, so a fixed layout can push "Compute" off the bottom edge.
        # Two structural fixes rather than a bigger default window:
        #   * the action buttons and the status line live in a FOOTER below
        #     the parameter block, so they are visible at any window height;
        #   * everything above them scrolls.
        outer = QtWidgets.QWidget()
        outerLay = QtWidgets.QVBoxLayout(outer)
        outerLay.setContentsMargins(6, 6, 6, 6)

        body, g = qtkit.grid()
        self._controlBody = body
        row = 0
        right = QtCore.Qt.AlignmentFlag.AlignRight
        leftAl = QtCore.Qt.AlignmentFlag.AlignLeft

        def header(text, r):
            g.addWidget(qtkit.separator(), r, 0, 1, 2)
            g.addWidget(qtkit.heading(text), r + 1, 0, 1, 2)
            return r + 2

        def entry(label, var, r, width=10, tip=None):
            g.addWidget(QtWidgets.QLabel(label), r, 0, right)
            e = qtkit.entry(var, width)
            g.addWidget(e, r, 1, leftAl)
            if tip:
                g.addWidget(_hint(tip), r + 1, 0, 1, 2)
                return r + 2
            return r + 1

        # ---- size distribution ----
        row = header("Size distribution", row)
        self.distVar = qtkit.StringVar(value="Log-normal")
        g.addWidget(QtWidgets.QLabel("Type:"), row, 0, right)
        distBox = qtkit.combo(self.distVar, DIST_NAMES, width=14,
                              on_select=self._syncEnabledStates)
        g.addWidget(distBox, row, 1, leftAl)
        row += 1

        # Distribution-specific parameters live in their own frame, rebuilt
        # when the dropdown changes -- the same pattern oZgui.py already uses
        # for its potential parameters. Values entered for one distribution
        # are remembered if the user switches away and back.
        self.distParamFrame, self._distParamGrid = qtkit.grid()
        g.addWidget(self.distParamFrame, row, 0, 1, 2)
        self.momentRuleVar = qtkit.BooleanVar(value=False)
        self.distVars = {}
        for name, params in DIST_PARAMS.items():
            for _lab, key, default, _tip in params:
                self.distVars.setdefault((name, key), qtkit.StringVar(value=default))
        row += 1

        self.nbinsVar = qtkit.StringVar(value="12")
        row = entry("Bins N:", self.nbinsVar, row)
        self.nsigVar = qtkit.StringVar(value="5.0")
        self.nsigRow = row
        row = entry("Truncate at:", self.nsigVar, row,
                    tip="+/- this many s.d. (unused for Beta: bounded support)")

        # ---- form factor ----
        row = header("Form factor", row)
        self.ffVar = qtkit.StringVar(value="Sphere")
        g.addWidget(QtWidgets.QLabel("Type:"), row, 0, right)
        ffBox = qtkit.combo(self.ffVar, ["Sphere", "Core-shell"], width=14,
                            on_select=self._syncEnabledStates)
        g.addWidget(ffBox, row, 1, leftAl)
        row += 1
        self.coreFracVar = qtkit.StringVar(value="0.6")
        g.addWidget(QtWidgets.QLabel("Core/outer ratio:"), row, 0, right)
        self.coreFracEntry = qtkit.entry(self.coreFracVar, 10)
        g.addWidget(self.coreFracEntry, row, 1, leftAl)
        row += 1
        self.rhoCoreVar = qtkit.StringVar(value="2.0")
        g.addWidget(QtWidgets.QLabel("SLD core / shell:"), row, 0, right)
        self.rhoCoreEntry = qtkit.entry(self.rhoCoreVar, 5)
        self.rhoShellVar = qtkit.StringVar(value="1.0")
        self.rhoShellEntry = qtkit.entry(self.rhoShellVar, 5)
        g.addWidget(qtkit.hbox(self.rhoCoreEntry, self.rhoShellEntry, spacing=3), row, 1, leftAl)
        row += 1

        # ---- interaction ----
        row = header("Hard-sphere-Yukawa interaction", row)
        self.phiVar = qtkit.StringVar(value="0.15")
        row = entry("Volume fraction:", self.phiVar, row)
        self.zVar = qtkit.StringVar(value="0.06")
        row = entry("Screening z:", self.zVar, row, tip="inverse length; z*sigma is the usual Z")
        self.KVar = qtkit.StringVar(value="1.0")
        row = entry("Contact K:", self.KVar, row,
                    tip="MSA sets c = -beta U:  K > 0 attractive,  K < 0 repulsive")

        self.closureVar = qtkit.StringVar(value="MSA")
        g.addWidget(QtWidgets.QLabel("Closure:"), row, 0, right)
        g.addWidget(qtkit.combo(self.closureVar, ["MSA", "RMSA (rescaled)"], width=14), row, 1, leftAl)
        row += 1

        # There is no fixed-point iteration here to accelerate -- see the
        # module docstring. What there IS is two independent closed-form
        # routes to the same solution, which is worth exposing because
        # running both is a live self-check.
        self.routeVar = qtkit.StringVar(value=ROUTES[0])
        g.addWidget(QtWidgets.QLabel("Solve via:"), row, 0, right)
        g.addWidget(qtkit.combo(self.routeVar, ROUTES, width=14), row, 1, leftAl)
        row += 1
        g.addWidget(_hint("two closed-form routes to the same\n"
                          "solution; 'both' cross-checks them"), row, 0, 1, 2)
        row += 1

        # ---- Q range ----
        row = header("Q range", row)
        self.QminVar = qtkit.StringVar(value="1e-3")
        row = entry("Q min:", self.QminVar, row)
        self.QmaxVar = qtkit.StringVar(value="0.3")
        row = entry("Q max:", self.QmaxVar, row)
        self.nQVar = qtkit.StringVar(value="200")
        row = entry("Points:", self.nQVar, row)

        # ---- approximations to overlay ----
        row = header("Compare against", row)
        self.approxVars = {}
        for label, attr in APPROX_SCHEMES:
            v = qtkit.BooleanVar(value=(attr == "I_decoupling"))
            g.addWidget(qtkit.check(v, label), row, 0, 1, 2)
            self.approxVars[label] = (v, attr)
            row += 1
        g.addWidget(_hint("schemes 2-5 each solve O(N^2) auxiliary\n"
                          "monodisperse systems -- ~10 s apiece at N=12"), row, 0, 1, 2)
        row += 1
        g.setRowStretch(row, 1)

        outerLay.addWidget(qtkit.scroll_area(body, min_width=270), 1)
        outerLay.addWidget(qtkit.separator())

        # ---- actions (in the always-visible footer) ----
        footer = QtWidgets.QWidget()
        self._controlFooter = footer
        fLay = QtWidgets.QVBoxLayout(footer)
        fLay.setContentsMargins(0, 4, 0, 4)
        self.computeButton = qtkit.button("Compute", self._onCompute)
        fLay.addWidget(qtkit.hbox(self.computeButton,
                                  qtkit.button("Clear", self._onClear),
                                  qtkit.button("Export...", self._onExport)))
        self.statusVar = qtkit.StringVar(value="ready")
        fLay.addWidget(qtkit.label(var=self.statusVar, color="blue", wrap=240))
        outerLay.addWidget(footer)

        self._distBox, self._ffBox = distBox, ffBox
        self._rebuildDistParams()
        return outer

    def _rebuildDistParams(self):
        """Show only the parameters the selected distribution actually takes."""
        g = self._distParamGrid
        while g.count():
            item = g.takeAt(0)
            w = item.widget()
            if w is not None:
                w.hide()
                w.setParent(None)
                w.deleteLater()
        name = self.distVar.get()
        right = QtCore.Qt.AlignmentFlag.AlignRight
        leftAl = QtCore.Qt.AlignmentFlag.AlignLeft
        r = 0
        for label, key, _default, tip in DIST_PARAMS.get(name, []):
            g.addWidget(QtWidgets.QLabel(label + ":"), r, 0, right)
            g.addWidget(qtkit.entry(self.distVars[(name, key)], 10), r, 1, leftAl)
            r += 1
            if tip:
                g.addWidget(_hint(tip), r, 0, 1, 2)
                r += 1
        if name == "Log-normal":
            # The log-normal is the one distribution with no closed-form
            # moment-exact Gaussian rule, so its default nodes reach only
            # ~1e-5 on <R^6> where Gamma and Beta reach 1e-16. This switches
            # to a Golub-Welsch rule built from the analytic moments.
            cb = qtkit.check(self.momentRuleVar, "moment-exact nodes")
            _unbindOnDestroy(self.momentRuleVar, cb)
            g.addWidget(cb, r, 0, 1, 2)
            r += 1
            g.addWidget(_hint("exact <R^n> to 1e-16 instead of ~1e-5\n"
                              "(needs mpmath; ignores 'Truncate at')"), r, 0, 1, 2)
            r += 1
        g.setColumnStretch(2, 1)

    def _syncEnabledStates(self):
        self._rebuildDistParams()
        enabled = self.ffVar.get() == "Core-shell"
        for w in (self.coreFracEntry, self.rhoCoreEntry, self.rhoShellEntry):
            w.setEnabled(enabled)

    # ------------------------------------------------------------------
    def _buildPlots(self):
        self.notebook = qtkit.PlotTabs(figsize=(6, 5), dpi=100)
        for label, key in PLOT_TABS:
            self._decorateAxes(self.notebook.add_plot(label, key), key)
        self.axes = self.notebook.axes
        self.canvases = self.notebook.canvases

        # a text tab for the contact matrix and the run summary
        self.summaryText = self.notebook.add_text("Summary")
        return self.notebook

    @staticmethod
    def _decorateAxes(ax, key):
        if key == "iq":
            ax.set_xlabel("Q"); ax.set_ylabel("I(Q)")
            ax.set_xscale("log"); ax.set_yscale("log")
        elif key == "err":
            ax.set_xlabel("Q"); ax.set_ylabel(r"$|I_{approx}/I_{exact}-1|$")
            ax.set_xscale("log"); ax.set_yscale("log")
        elif key == "sij":
            ax.set_xlabel("Q"); ax.set_ylabel(r"$S_{ij}^{AL}(Q)$")
            ax.set_xscale("log")
        elif key == "sq":
            ax.set_xlabel("Q"); ax.set_ylabel("S(Q)  (number-number)")
            ax.set_xscale("log")
        elif key == "dist":
            ax.set_xlabel("R"); ax.set_ylabel("weight")

    # ------------------------------------------------------------------
    def _readInputs(self):
        """Parse and sanity-check the control panel. Raises ValueError with a
        message meant to be shown to the user."""
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
        # distribution parameters, with per-parameter bounds
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
            # The class enforces this too, but catching it here turns it into
            # an inline message on the control panel rather than a traceback
            # arriving from the worker thread.
            if dp["Rmax"] <= dp["Rmin"]:
                raise ValueError("Beta: R max must be greater than R min")
        p["momentRule"] = bool(self.momentRuleVar.get())
        p["nbins"] = int(f(self.nbinsVar, "Bins N", lo=1, hi=64))
        p["nsig"] = f(self.nsigVar, "Truncate at", lo=0.5, hi=10.0)
        p["phi"] = f(self.phiVar, "Volume fraction", lo=1e-6, hi=0.62)
        p["z"] = f(self.zVar, "Screening z", lo=1e-12)
        p["K"] = f(self.KVar, "Contact K")
        p["Qmin"] = f(self.QminVar, "Q min", lo=1e-12)
        p["Qmax"] = f(self.QmaxVar, "Q max", lo=1e-12)
        p["nQ"] = int(f(self.nQVar, "Points", lo=2, hi=5000))
        if p["Qmax"] <= p["Qmin"]:
            raise ValueError("Q max must exceed Q min")
        p["ff"] = self.ffVar.get()
        p["coreFrac"] = f(self.coreFracVar, "Core/outer ratio", lo=1e-3, hi=0.999)
        p["rhoCore"] = f(self.rhoCoreVar, "SLD core")
        p["rhoShell"] = f(self.rhoShellVar, "SLD shell")
        p["closure"] = self.closureVar.get()
        p["route"] = self.routeVar.get()
        p["approx"] = [(lab, attr) for lab, (v, attr) in self.approxVars.items()
                       if v.get()]
        return p

    def _onCompute(self):
        if self.worker is not None and self.worker.is_alive():
            self.statusVar.set("a calculation is already running")
            return
        try:
            p = self._readInputs()
        except ValueError as exc:
            self.statusVar.set(str(exc))
            qtkit.showerror("Invalid input", str(exc), parent=self)
            return

        self.computeButton.setEnabled(False)
        self.statusVar.set("solving...")
        self.worker = threading.Thread(target=self._computeWorker, args=(p,),
                                       daemon=True)
        self.worker.start()

    # ------------------------------------------------------------------
    def _computeWorker(self, p):
        """Runs OFF the main thread. Must not touch a single Qt widget --
        everything goes back through self.resultQueue."""
        def progress(msg):
            self.resultQueue.put(("status", msg))

        try:
            res = _Result()
            dist = _makeDistribution(p["dist"], p["distpars"],
                                     momentRule=p.get("momentRule", False))

            if p["ff"] == "Core-shell":
                ff = CoreShell(rho_core=p["rhoCore"], rho_shell=p["rhoShell"],
                               rho_solvent=0.0, ratio=p["coreFrac"])
            else:
                ff = Sphere()

            routeDiff = None
            rmsa = p["closure"].startswith("RMSA")
            # For RMSA, construct at K = 0 -- always solvable -- and put the
            # true K back afterwards, so that a state with no MSA root (which
            # is precisely when RMSA is wanted) can still be reached.
            sas = PolydisperseYukawaSAS(dist, p["phi"], p["z"],
                                        0.0 if rmsa else p["K"],
                                        formfactor=ff, nbins=p["nbins"],
                                        nsig=p["nsig"])
            if rmsa:
                sas.K = p["K"]
                sas.__dict__.pop("_mono_cache", None)

            route = p.get("route", ROUTE_GAMMA)
            if not rmsa and route != ROUTE_GAMMA:
                # solve() is singular at exactly K = 0; solve_gamma() takes
                # that branch analytically, so fall back rather than fail.
                if p["K"] == 0.0:
                    progress("K = 0: amplitude route is singular, using Gamma")
                else:
                    progress(f"re-solving via the {route} route...")
                    solAmp = sas.solver.solve()
                    if route == ROUTE_AMPL:
                        sas.sol = solAmp
                        sas.sq = PolydisperseOneYukawaSq(sas.solver, solAmp)
                    else:
                        gA = sas.solver.contact_value(solAmp)
                        gG = sas.solver.contact_value(sas.sol)
                        scale = max(float(np.max(np.abs(gG))), 1e-300)
                        routeDiff = float(np.max(np.abs(gA - gG)))/scale

            lines, notes = [], []
            if rmsa:
                rm = _apply_rmsa(sas, p["K"], lam_max=None)
                if rm.get("lam") is None:
                    raise RuntimeError("RMSA: " + rm.get("status", "no solution"))
                lines.append(f"RMSA: lambda = {rm['lam']:.6f}   "
                             f"phi {rm['phi']:.4f} -> {rm['phi_rescaled']:.4f}")
                lines.append(f"      min g_ij = {rm['min_g']:.3e}   ({rm['status']})")
                res.label = f"RMSA lambda={rm['lam']:.4f}"
            else:
                res.label = "MSA"

            Q = np.logspace(np.log10(p["Qmin"]), np.log10(p["Qmax"]), p["nQ"])
            res.Q = Q
            progress("exact I(Q) and S_ij(Q)...")
            res.I_exact = sas.I_exact(Q)
            res.S = sas.S_partials(Q)
            res.S_number = sas.S_number(Q)
            res.R, res.w, res.sigma = sas.R, sas.w, sas.sigma
            res.g = sas.solver.contact_value(sas.sol)

            for n, (label, attr) in enumerate(p["approx"], 1):
                progress(f"approximation {n}/{len(p['approx'])}: {label.strip()}...")
                try:
                    res.approx[label] = getattr(sas, attr)(Q)
                except Exception as exc:
                    notes.append(f"{label}:\n    unavailable -- {exc}")

            # summary text
            lines.append("")
            if routeDiff is not None:
                lines.append(f"route cross-check   : {routeDiff:.2e}"
                             "   (eq.55 vs eq.72, relative)")
            lines.append(f"distribution        : {dist!r}")
            lines.append(f"components N        : {len(res.R)}")
            lines.append(f"sigma range         : {res.sigma.min():.4g} .. {res.sigma.max():.4g}")
            lines.append(f"z*sigma range       : {p['z']*res.sigma.min():.3f} .. "
                         f"{p['z']*res.sigma.max():.3f}")
            if isinstance(sas.sol, dict) and "gamma" in sas.sol:
                lines.append(f"Gamma               : {float(sas.sol['gamma']):.8g}")
            gmin, gmax = float(np.min(res.g)), float(np.max(res.g))
            # The RMSA rescaling condition drives min g_ij to zero, so it lands
            # at +/- a few 1e-15. Printing that as "-0.000000" and calling it
            # unphysical would be wrong -- tolerance it against the scale of g.
            gtol = 1e-9*max(1.0, abs(gmax))
            gminShown = 0.0 if abs(gmin) < gtol else gmin
            lines.append(f"contact g_ij        : {gminShown:.6f} .. {gmax:.6f}")
            asym = float(np.max(np.abs(res.g - res.g.T)))
            lines.append(f"|g - g^T|           : {asym:.2e}   (symmetry check)")
            if gmin < -gtol:
                lines.append("")
                lines.append("WARNING: g_ij(sigma+) < 0 -- this MSA solution is")
                lines.append("unphysical. Switch the closure to RMSA.")
            lines.append("")
            try:
                mr = moment_report(dist, res.R, res.w)
                lines.append("moment    discretised        analytic      rel.err")
                for n in sorted(mr):
                    num, ex, rel = mr[n]
                    lines.append(f"  <R^{n}>  {num:16.8g} {ex:16.8g} {rel:10.2e}")
            except Exception as exc:
                lines.append(f"(moment report unavailable: {exc})")
            if notes:
                lines.append("")
                lines.append("Approximations that have no solution in this state")
                lines.append("(a property of the scheme, not of the solver):")
                lines.extend(notes)
            res.summary = "\n".join(lines)

            self.resultQueue.put(("ok", res))
        except Exception as exc:
            self.resultQueue.put(("err", _explain(exc) + "\n\n"
                                  + "-"*60 + "\n" + traceback.format_exc()))

    def _pollResultQueue(self):
        """Drain the result queue now (the QueuePoller also does this every
        100 ms on the GUI thread)."""
        self._poller.poll()

    def _handleQueueItem(self, item):
        kind, payload = item
        if kind == "status":
            self.statusVar.set(payload)
            return
        if kind == "ok":
            self.result = payload
            self._replot()
            self.statusVar.set(f"done  ({payload.label})")
        else:
            self.statusVar.set("failed -- see Summary tab")
            self.summaryText.setPlainText(payload)
            self.notebook.setCurrentIndex(len(PLOT_TABS))
        self.computeButton.setEnabled(True)

    # ------------------------------------------------------------------
    def _onClear(self):
        self.result = None
        for key, ax in self.axes.items():
            ax.clear()
            self._decorateAxes(ax, key)
            self.canvases[key].draw_idle()
        self.summaryText.clear()
        self.statusVar.set("cleared")

    def _replot(self):
        r = self.result
        if r is None:
            return
        for key, ax in self.axes.items():
            ax.clear()
            self._decorateAxes(ax, key)

        ax = self.axes["iq"]
        ax.plot(r.Q, r.I_exact, "k-", lw=2, label=f"exact ({r.label})")
        for label, I in r.approx.items():
            ax.plot(r.Q, np.abs(I), lw=1, label=label)
        ax.legend(fontsize=7)

        ax = self.axes["err"]
        pos = r.I_exact > 0
        for label, I in r.approx.items():
            with np.errstate(divide="ignore", invalid="ignore"):
                e = np.abs(I/np.where(pos, r.I_exact, np.nan) - 1.0)
            ax.plot(r.Q, e, lw=1, label=label)
        if r.approx:
            ax.legend(fontsize=7)
        else:
            ax.text(0.5, 0.5, "no schemes selected", ha="center",
                    transform=ax.transAxes, color="#888")

        ax = self.axes["sij"]
        N = r.S.shape[1]
        # N can be 12+, so N^2 curves would be unreadable -- show the diagonal
        # plus the extreme off-diagonal, which is what actually differs.
        # (Port fix: the Tk code iterated (0, N//2, N-1) directly, which for
        # N <= 2 drew the same diagonal curve twice or three times.)
        for i in sorted({0, N//2, N - 1}):
            ax.plot(r.Q, r.S[:, i, i], lw=1.2, label=f"S_{i}{i}")
        if N > 1:
            ax.plot(r.Q, r.S[:, 0, N - 1], "--", lw=1.2, label=f"S_0{N-1}")
        ax.axhline(1.0, color="#bbb", lw=0.6)
        ax.legend(fontsize=7)

        ax = self.axes["sq"]
        ax.plot(r.Q, r.S_number, "k-", lw=1.5)
        ax.axhline(1.0, color="#bbb", lw=0.6)

        ax = self.axes["dist"]
        # (Port fix: np.gradient needs >= 2 points; for N = 1 the Tk code
        # raised here, after the result had already been accepted.)
        width = np.gradient(r.R)*0.8 if len(r.R) > 1 else 0.1*r.R
        ax.bar(r.R, r.w, width=width, color="#4477aa")
        ax.set_title(f"{len(r.R)} components", fontsize=9)

        for key in self.axes:
            self.canvases[key].draw_idle()

        self.summaryText.setPlainText(r.summary)

    # ------------------------------------------------------------------
    def _onExport(self):
        if self.result is None:
            self.statusVar.set("nothing to export -- press Compute first")
            return
        path = qtkit.asksaveasfilename(
            parent=self, defaultextension=".dat",
            filetypes=[("ASCII table", "*.dat *.txt"), ("All files", "*.*")],
            title="Export I(Q) and S(Q)")
        if not path:
            return
        r = self.result
        cols = [("Q", r.Q), ("I_exact", r.I_exact), ("S_number", r.S_number)]
        cols += [(lab.split()[0] + "_" + lab.split()[1] if len(lab.split()) > 1
                  else lab, I) for lab, I in r.approx.items()]
        try:
            with open(path, "w", encoding="utf-8") as fh:
                fh.write(f"# polydisperse hard-sphere-Yukawa, {r.label}\n")
                for line in r.summary.splitlines():
                    fh.write(f"# {line}\n")
                fh.write("# " + "\t".join(name for name, _ in cols) + "\n")
                for k in range(len(r.Q)):
                    fh.write("\t".join(f"{col[k]:.8e}" for _, col in cols) + "\n")
        except OSError as exc:
            qtkit.showerror("Export failed", str(exc), parent=self)
            return
        self.statusVar.set(f"exported {os.path.basename(path)}")


def main():                                                # pragma: no cover
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
    tab = PolydisperseYukawaTab()
    tab.setWindowTitle("Polydisperse hard-sphere-Yukawa (MSA / RMSA)")
    tab.resize(1200, 750)
    tab.show()
    return app.exec()


if __name__ == "__main__":                                 # pragma: no cover
    sys.exit(main())
