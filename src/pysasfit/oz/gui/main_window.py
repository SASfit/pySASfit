# -*- coding: utf-8 -*-
"""Ornstein-Zernike GUI main window (Qt port of pyOZgui's Tk ``oZgui.py``).

    pysasfit-oz                     # or: python -m pysasfit.oz.gui.main_window

Top-level tabs: "0: OZ solver" (one-component solver, closures, all curves)
followed by the tools in :data:`EXTRA_TABS`. Each extra tab is a
self-contained QWidget in its own module; a module that fails to import or
construct costs only its own tab (the error is printed and shown in its place).
"""

import importlib
import sys
import traceback

from PySide6 import QtCore, QtWidgets

from .oz_solver_tab import OZSolverTab

# (module in pysasfit.oz.gui, class, tab label)
EXTRA_TABS = [
    # The generic tab combines ANY potential with ANY multicomponent-capable
    # closure and ANY form factor. It replaced the two analytic special-case
    # tabs "Polydisperse Yukawa" (polydisperse_yukawa_tab: Yukawa + MSA/RMSA)
    # and "Robertus SHS" (robertus_shs_tab: exact multicomponent PY adhesive
    # spheres). Both modules are still present; add a line here to bring
    # either back. What they offer that the generic tab does not: RMSA (a
    # one-component rescaling, not multicomponent-capable) and the exact
    # Baxter Q-matrix solution for adhesive spheres.
    ("generic_polydisperse_tab", "GenericPolydisperseTab", "Polydisperse (any potential)"),
    # KEPT DELIBERATELY: the generic builder refuses charge-coupled potentials
    # (amplitude scales with size, kappa depends on the whole distribution), so
    # this tab is the only route to a polydisperse charged Yukawa system.
    ("ry_polydisperse_yukawa_tab", "RYPolydisperseYukawaTab", "RY Polydisperse Yukawa"),
    # Validation tool: the numerical multicomponent route beside independent
    # ANALYTIC mixture solutions (mixscatter's Vrij PY, Gazzillo's charged MSA
    # and adhesive models), with the coarse/fine grid ratio as diagnostic.
    ("mixture_validation_tab", "MixtureValidationTab", "Mixture validation"),
]


def _error_widget(label, text):
    w = QtWidgets.QPlainTextEdit()
    w.setReadOnly(True)
    w.setPlainText(f"The tab '{label}' could not be loaded:\n\n{text}")
    return w


class OZWidget(QtWidgets.QTabWidget):
    """All OZ tools as tabs -- embeddable (the pysasfit launcher hosts this)."""

    def __init__(self, parent=None, extra_tabs=None, redirect_output=True):
        super().__init__(parent)
        self.setDocumentMode(True)
        # The OZ solver is index 0 because it is FIRST; the extra tabs carry
        # their own SESSION_TAB_INDEX (used in their session file extension
        # .oz<i>) and are numbered from 1 for the same reason.
        self.ozTab = OZSolverTab(redirect_output=redirect_output)
        self.addTab(self.ozTab, "0: OZ solver")
        self.tools = {}
        for moduleName, className, tabLabel in (EXTRA_TABS if extra_tabs is None else extra_tabs):
            try:
                mod = importlib.import_module(f"{__package__}.{moduleName}")
                widget = getattr(mod, className)()
                # Label number from the widget's own SESSION_TAB_INDEX, so it
                # matches what its session FILE says even if a tab is missing.
                index = getattr(widget, "SESSION_TAB_INDEX", None)
                text = tabLabel if index is None else f"{int(index)}: {tabLabel}"
                self.addTab(widget, text)
                self.tools[tabLabel] = widget
            except Exception:
                tb = traceback.format_exc()
                print(f"[oz gui] optional tab '{tabLabel}' unavailable:\n{tb}", file=sys.stderr)
                self.addTab(_error_widget(tabLabel, tb), f"{tabLabel} (error)")

    def shutdown(self):
        """Restore stdout/stderr (redirected into the Log tab)."""
        self.ozTab.restoreOutput()


class OZMainWindow(QtWidgets.QMainWindow):
    def __init__(self, extra_tabs=None):
        super().__init__()
        self.setWindowTitle("Ornstein-Zernike solver (pySASfit)")
        self.oz = OZWidget(extra_tabs=extra_tabs)
        self.setCentralWidget(self.oz)
        self.resize(1200, 776)

    def closeEvent(self, event):
        self.oz.shutdown()
        super().closeEvent(event)


def main(argv=None):
    argv = sys.argv if argv is None else argv
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(argv)
    w = OZMainWindow()
    # Maximised on start: the parameter panels have outgrown 1200x776, which
    # would hide the lower controls from a new user.
    w.showMaximized()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
