"""
pysasfit.inversion.gui
======================

Thin Qt (PySide6) layer over the ``pysasfit.inversion`` library. No numerics
live here -- every computation (loading, background fitting/subtraction,
kernel construction, solving) calls straight into the library, so the
library stays usable headless/scriptable/testable on its own. See
``pysasfit.inversion.gui.app.MainWindow``.
"""
