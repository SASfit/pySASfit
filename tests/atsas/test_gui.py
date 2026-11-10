"""pyATsas panel: builds without ATSAS, loads curves, writes runnable Python code, runs a (fake) ATSAS program
in the background and shows its result; the tab icon is the ATSAS icon of the installation or a placeholder."""
import os
import sys

import numpy as np
import pytest

pytest.importorskip("PySide6")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6 import QtCore, QtWidgets  # noqa: E402

from pysasfit.atsas import core as A  # noqa: E402

GNOM_OUT = """\
   Real space range                 :  0.0000 to  10.000
   Real space Rg     : 0.3872E+01 +- 0.1000E-01
   Total Estimate    : 0.92
      S          J EXP       ERROR       J REG       I REG
  0.1000E+00  0.9900E+03  0.1000E+01  0.9890E+03  0.9890E+03
  0.2000E+00  0.9500E+03  0.1000E+01  0.9560E+03  0.9560E+03

           R          P(R)      ERROR
  0.0000E+00  0.0000E+00  0.0000E+00
  0.5000E+01  0.1000E+01  0.1000E-01
  0.1000E+02  0.0000E+00  0.0000E+00
"""


def _fake_install(root, version="4.1.4"):
    b = root / f"ATSAS-{version}" / "bin"
    b.mkdir(parents=True)
    for exe in ("autorg", "datgnom", "gnom", "primusqt"):
        (b / (exe + A.EXE_SUFFIX)).write_text("")
    return b.parent


@pytest.fixture
def app():
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


@pytest.fixture
def sphere_file(tmp_path):
    q, I, e = A.sphere_curve()
    f = tmp_path / "sphere.dat"
    np.savetxt(f, np.c_[q, I, e])
    return f


def _wait(win, timeout=10000):
    loop = QtCore.QEventLoop()
    win._signals.done.connect(lambda *_a: QtCore.QTimer.singleShot(0, loop.quit))
    QtCore.QTimer.singleShot(timeout, loop.quit)
    if win._busy:
        loop.exec()


def test_window_builds_and_loads(app, sphere_file):
    from pysasfit.atsas.gui import MainWindow
    w = MainWindow()
    assert set(w.docks) == {"data", "analysis", "iq", "pr", "results", "log", "programs"}
    w.load_files([str(sphere_file)])
    assert len(w.curves) == 1 and w.last.value() == 400
    w.first.setValue(3)
    (q, _I, _e), off = w.used()
    assert off == 2 and len(q) == 398
    code = w.python_code()
    compile(code, "<pyATsas>", "exec")
    assert "atsas.autorg" in code and "atsas.datgnom" in code
    w.iq_scale.setCurrentIndex(2)
    w.draw_all()
    w.close()


def test_tab_icon(app, tmp_path, monkeypatch):
    from pysasfit import resources
    resources._EXTERNAL_CACHE.clear()
    monkeypatch.setattr(A, "find_atsas", lambda *_a, **_k: (_ for _ in ()).throw(A.AtsasError("none")))
    assert not resources.tool_icon("pyATsas").isNull()                    # placeholder without ATSAS
    assert [s[0] for s in resources.wordmark("pyATsas")] == ["py", "AT", "sas"]
    resources._EXTERNAL_CACHE.clear()


@pytest.mark.skipif(sys.platform.startswith("win"), reason="fake programs are shell scripts")
def test_run_in_background(app, tmp_path, sphere_file, monkeypatch):
    root = _fake_install(tmp_path / "inst")
    b = root / "bin"
    (b / "autorg").write_text("#!/bin/sh\necho \"3.870 0.010 1000.0 2.0 5 40 0.95 0.0 $1\"\n")
    (tmp_path / "x.out").write_text(GNOM_OUT)
    (b / "datgnom").write_text(f'#!/bin/sh\ncp "{tmp_path / "x.out"}" "$3"\n')
    for f in b.iterdir():
        f.chmod(0o755)
    from pysasfit.core import settings as prefs
    monkeypatch.setattr(A, "_ATSAS_BIN", None)
    prefs.set_value("atsas.dir", str(root))
    prefs.set_value("atsas.workdir", str(tmp_path / "runs"))
    try:
        from pysasfit.atsas.gui import MainWindow
        w = MainWindow()
        assert w.installed and w.program_buttons["primusqt"][0].isEnabled()
        assert not w.program_buttons["chromixs"][0].isEnabled()
        w.load_files([str(sphere_file)])
        w.run_datgnom()                       # runs AUTORG first
        _wait(w)
        r = w.results["sphere.dat"]
        assert r["autorg"].rg == pytest.approx(3.87) and r["datgnom"].dmax == 10
        assert w.table.item(0, 5).text() == "10"
        assert len(os.listdir(tmp_path / "runs")) == 1
        w.close()
    finally:
        prefs.reset()
        A._ATSAS_BIN = None
