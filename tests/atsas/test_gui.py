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
    assert set(w.docks) == {"data", "analysis", "iq", "pr", "results", "log", "programs", "model3d"}
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
        assert w.table.item(0, 6).text() == "10"
        assert len(os.listdir(tmp_path / "runs")) == 1
        w.close()
    finally:
        prefs.reset()
        A._ATSAS_BIN = None


def test_background_fit_and_subtraction(app, tmp_path):
    """Independent fit (works without ATSAS): sphere + flat background 0.5; subtracting gives a new curve."""
    from pysasfit.atsas.gui import MainWindow
    q, I, e = A.sphere_curve(R=5.0, qmax=6.0, noise=0.0)
    I = I + 0.5
    f = tmp_path / "sphere_bg.dat"
    np.savetxt(f, np.c_[q, I, 0.01 * I])
    w = MainWindow()
    w.installed = False                     # only the independent fit
    w.load_files([str(f)])
    w.bg_qmin.setValue(3.0)
    w.bg_qmax.setValue(6.0)
    w.bg_fix.setChecked(True)               # Porod: B = 4
    w.run_background()
    _wait(w)
    fit = w.results["sphere_bg.dat"]["background"]["fit"]
    assert isinstance(fit, A.PorodResult) and fit.bg == pytest.approx(0.5, rel=0.05)
    assert w.bg_value.value() == pytest.approx(fit.bg)
    n = len(w.curves)
    w.on_subtract_background()
    assert len(w.curves) == n + 1 and w.current() is w.curves[-1]
    new = w.curves[-1]
    assert np.allclose(new.I, I - fit.bg, rtol=1e-12, atol=1e-6) and "minus bg" in new.name
    code = w.python_code()
    compile(code, "<pyATsas>", "exec")
    assert "fit_power_background" in code and "bg = " in code
    w.save_session()
    w.close()
    w2 = MainWindow()
    w2.restore_session()                    # the subtracted curve is rebuilt from the file
    assert [c.name for c in w2.curves] == [c.name for c in w.curves]
    w2.close()


@pytest.mark.skipif(sys.platform.startswith("win"), reason="fake programs are shell scripts")
def test_failure_is_shown(app, tmp_path, sphere_file, monkeypatch):
    """A program that returns nothing: red status bar, ✖ in the Results table, message in the Log;
    a later success clears the mark."""
    root = _fake_install(tmp_path / "inst")
    b = root / "bin"
    (b / "autorg").write_text("#!/bin/sh\necho 3.870 0.010 1000.0 2.0 5 40 0.30 0.0 $1\n")    # poor quality
    (b / "datgnom").write_text("#!/bin/sh\ntrue\n")                                          # writes nothing
    for f in b.iterdir():
        f.chmod(0o755)
    from pysasfit.core import settings as prefs
    monkeypatch.setattr(A, "_ATSAS_BIN", None)
    prefs.set_value("atsas.dir", str(root))
    prefs.set_value("atsas.workdir", str(tmp_path / "runs"))
    try:
        from pysasfit.atsas.gui import MainWindow
        w = MainWindow()
        w.load_files([str(sphere_file)])
        w.run_datgnom()
        _wait(w)
        r = w.results["sphere.dat"]
        assert "wrote no input.out" in r["error"] and "datgnom" not in r
        assert w.table.item(0, 0).text().startswith("✖")
        assert "failed" in w.statusBar().currentMessage()
        assert "wrote no input.out" in w.log_view.toPlainText()
        w.run_autorg()                                   # works, with a warning
        _wait(w)
        r = w.results["sphere.dat"]
        assert "error" not in r and any("quality" in x for x in r["warnings"])
        assert w.table.item(0, 0).text().startswith("⚠")
        w.close()
    finally:
        prefs.reset()
        A._ATSAS_BIN = None


DAM_PDB = """\
 Number of particle atoms ............................... : 4
 DAM packing radius ..................................... : 1.800
 Radius of gyration ..................................... : 2.55
ATOM      1  CA  ASP    1       -1.800   0.000   0.000  1.00 20.00 0 2 201
ATOM      2  CA  ASP    1        1.800   0.000   0.000  1.00 20.00 0 2 201
ATOM      3  CA  ASP    1        0.000   3.600   0.000  1.00 20.00 0 2 201
ATOM      4  CA  ASP    1        0.000   0.000   3.600  1.00 20.00 0 2 201
"""


def test_3d_panel_beads_and_envelope(app, tmp_path):
    from pysasfit.atsas import models as M
    from pysasfit.atsas.gui import MainWindow
    f = tmp_path / "dam-1.pdb"
    f.write_text(DAM_PDB)
    w = MainWindow()
    w.show_beads_3d(paths=[f])
    v = w.model_view
    assert len(v.objects) == 1 and v.objects[0]["data"].radius == 1.8
    a, b, c = 30.0, 20.0, 15.0
    coef = M.flm_of_shape(lambda T, P: 1 / np.sqrt((np.sin(T) * np.cos(P) / a) ** 2
                                                   + (np.sin(T) * np.sin(P) / b) ** 2 + (np.cos(T) / c) ** 2), 10)
    flm = M.write_flm_reference(tmp_path / "sas.flm", coef, np.sqrt((a * a + b * b + c * c) / 5),
                                4 / 3 * np.pi * a * b * c)
    q, I, e = A.sphere_curve()
    np.savetxt(tmp_path / "s.dat", np.c_[q, I, e])
    w.load_files([str(tmp_path / "s.dat")])
    w.load_flm(flm)
    w.show_envelope_3d()
    assert len(v.objects) == 2 and v.objects[1]["kind"] == "envelope"
    meshes = v.meshes()
    assert len(meshes) == 2 and meshes[1][3]                   # the envelope is translucent
    bad = M.write_flm_reference(tmp_path / "bad.flm", coef, 99.0, 1.0)
    w.load_flm(bad)
    w.show_envelope_3d()                                       # not verified: not drawn
    assert len(v.objects) == 2 and "not drawn" in w.statusBar().currentMessage()
    w.close()
