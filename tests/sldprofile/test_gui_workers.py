"""'parallel processes' spin box of the fit panel = preference sldprofile.opt.n_workers, both ways."""
import os

import pytest

pytest.importorskip("PySide6")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6 import QtWidgets  # noqa: E402

from pysasfit.core import settings as prefs  # noqa: E402
from pysasfit.sldprofile.gui import MainWindow  # noqa: E402

DATA = os.path.join(os.path.dirname(__file__), "..", "..", "examples", "sldprofile", "data",
                    "inhomogeneous_sphere_r5nm.dat")


def test_workers_spin_box():
    QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    old = prefs.get("sldprofile.opt.n_workers")
    try:
        w = MainWindow()
        w.load_file(DATA)
        assert w.sp_workers.value() == int(old or 0)
        w.sp_workers.setValue(3)
        assert prefs.get("sldprofile.opt.n_workers") == 3              # spin box -> preference
        assert w._fitter_factory()(None).opt["n_workers"] == 3          # and into the fit
        prefs.set_value("sldprofile.opt.n_workers", 1)                  # preference dialog -> spin box
        assert w._fitter_factory()(None).opt["n_workers"] == 1 and w.sp_workers.value() == 1
        w.close()
    finally:
        prefs.set_value("sldprofile.opt.n_workers", old)
