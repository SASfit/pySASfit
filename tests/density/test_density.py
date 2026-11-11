"""pyDENsity: MRC reading/writing, isosurfaces, running DENSS (a fake DENSS in Python, so this runs everywhere;
the real DENSS when it is installed), NumPy-2 shim, and the window."""
import os

import numpy as np
import pytest

from pysasfit.density import core as D

FAKE_CORE = '''
import struct, numpy as np
def write_mrc(rho, side, filename="map.mrc"):
    side = np.atleast_1d(side)
    a, b, c = (side, side, side) if len(side) == 1 else side
    with open(filename, "wb") as f:
        xs, ys, zs = rho.shape
        f.write(struct.pack("<iiii", xs, ys, zs, 2)); f.write(struct.pack("<iii", 0, 0, 0))
        f.write(struct.pack("<iii", xs, ys, zs)); f.write(struct.pack("<fff", a, b, c))
        f.write(b"\\0" * (1024 - 52)); f.write(np.asarray(rho, np.float32).flatten("F").tobytes())
'''

FAKE_ABINITIO = '''
import sys, argparse, numpy as np
import denss.core as dc
p = argparse.ArgumentParser(); p.add_argument("-f"); p.add_argument("-u"); p.add_argument("-o"); p.add_argument("-m")
p.add_argument("-d", type=float); p.add_argument("--seed"); p.add_argument("--plot_off", action="store_true")
p.add_argument("-q", action="store_true")
a, _rest = p.parse_known_args()
np.in1d(np.arange(3), [1]); np.trapz([1, 2, 3])          # removed in NumPy 2.4: the shim must provide them
d = np.loadtxt(a.f, comments="#")
n, vox = 24, 2.5
x = (np.arange(n) - n / 2 + 0.5) * vox
X, Y, Z = np.meshgrid(x, x, x, indexing="ij")
rho = (X ** 2 + Y ** 2 + Z ** 2 <= 20.0 ** 2).astype(float)
dc.write_mrc(rho, n * vox, a.o + ".mrc")                 # one-element side: the shim makes it a float triple
dc.write_mrc(rho, n * vox, a.o + "_support.mrc")
np.savetxt(a.o + "_map.fit", np.c_[d[:, 0], d[:, 1], d[:, 2], d[:, 1] * 1.01], header="q,I,err,I(map); chi2=1.234")
np.savetxt(a.o + "_stats_by_step.dat", np.c_[[5.0, 2.0, 1.234], [20.0, 16.0, 15.6], [4e4, 3.4e4, 3.35e4]],
           header="Chi2 Rg SupportVolume")
open(a.o + ".log", "w").write("Maximum dimension: 40.000\\nNumber of steps: 3\\nFinal Chi2: 1.234e+00\\n"
                               "Final Rg: 15.6\\nFinal Support Volume: 33510.3\\n")
print(a.o)
'''


@pytest.fixture
def fake_denss(tmp_path, monkeypatch):
    root = tmp_path / "site"
    (root / "denss" / "scripts").mkdir(parents=True)
    (root / "denss" / "__init__.py").write_text('__version__ = "0.0-fake"\n')
    (root / "denss" / "core.py").write_text(FAKE_CORE)
    (root / "denss" / "scripts" / "__init__.py").write_text("")
    (root / "denss" / "scripts" / "denss_abinitio.py").write_text(
        FAKE_ABINITIO.replace("\nimport sys,", "\nimport sys,") + '\n')
    monkeypatch.setenv("PYTHONPATH", str(root) + os.pathsep + os.environ.get("PYTHONPATH", ""))
    D._INFO.clear()
    yield root
    D._INFO.clear()


def test_mrc_roundtrip(tmp_path):
    rho = np.random.default_rng(0).random((5, 6, 7))
    D.write_mrc(tmp_path / "m.mrc", rho, 2.5)
    r, vox, h = D.read_mrc(tmp_path / "m.mrc")
    assert r.shape == (5, 6, 7) and np.allclose(r, rho, atol=1e-6) and np.allclose(vox, 2.5)
    (tmp_path / "bad.mrc").write_bytes(b"x" * 100)
    with pytest.raises(D.DenssError):
        D.read_mrc(tmp_path / "bad.mrc")


def test_isosurface_and_level():
    n, vox, R = 40, 1.0, 12.0
    x = np.arange(n) - n / 2 + 0.5
    X, Y, Z = np.meshgrid(x, x, x, indexing="ij")
    rho = np.exp(-(X ** 2 + Y ** 2 + Z ** 2) / (2 * 6.0 ** 2))
    V = 4 / 3 * np.pi * R ** 3
    lev = D.level_for_volume(rho, vox, V)
    v, f = D.isosurface(rho, lev, vox)
    r = np.linalg.norm(v, axis=1)
    assert len(f) > 100 and r.mean() == pytest.approx(R, rel=0.05)


def test_run_with_fake_denss(tmp_path, fake_denss):
    info = D.find_denss()
    assert info.version == "0.0-fake"
    q, I, e = D.sphere_curve()
    r = D.run_denss((q, I, e), dmax_nm=4.0, seed=3, cwd=tmp_path / "run")
    assert r.rho.shape == (24, 24, 24) and r.voxel == pytest.approx([2.5] * 3)
    assert r.rg == 15.6 and r.chi2 == 1.234 and r.steps == 3 and r.support.any()
    assert r.fit_q == pytest.approx(q, rel=1e-6)                       # back in 1/nm
    data = np.loadtxt(tmp_path / "run" / "data.dat")
    assert data[:, 0] == pytest.approx(q / 10, rel=1e-6)                # DENSS got 1/Å
    cmd = r.run["cmd"]
    assert "-d" in cmd and cmd[cmd.index("-d") + 1] == "40" and "--seed" in cmd


FAKE_ALL = '''
import argparse, os, numpy as np
import denss
p = argparse.ArgumentParser(); p.add_argument("-f"); p.add_argument("-o"); p.add_argument("-nm", type=int)
a, _rest = p.parse_known_args()
out = a.o; os.mkdir(out)
rho = np.zeros((16, 16, 16)); rho[4:12, 4:12, 4:12] = 1.0
denss.write_mrc(rho, 16 * 3.0, out + "/" + a.o + "_avg.mrc")       # the package-level name, as denss-all does
for i in range(a.nm):
    denss.write_mrc(rho, 16 * 3.0, out + "/%s_%d_aligned.mrc" % (a.o, i))
d = np.loadtxt(a.f, comments="#")
np.savetxt(out + "/" + a.o + "_map.fit", np.c_[d[:, :3], d[:, 1], d[:, 1] * 1.02], header="q I error I_fit_0 I_fit_1")
np.savetxt(out + "/" + a.o + "_fsc.dat", [[0, 1], [0.05, 0.6], [0.1, 0.3]], header="1/resolution, FSC; Resolution=12.8 +- 2.4 A")
open(out + "/" + a.o + "_final.log", "w").write("Average |Rg|.............: 15.146 +- 0.102\\n"
                                               "Average Chi2.............: 1.973e-01 +- 1.6e-02\\n")
'''


def test_run_all_with_fake_denss(tmp_path, fake_denss):
    (fake_denss / "denss" / "__init__.py").write_text('__version__ = "0.0-fake"\nfrom .core import write_mrc\n')
    (fake_denss / "denss" / "scripts" / "denss_all.py").write_text(FAKE_ALL)
    q, I, e = D.sphere_curve()
    r = D.run_denss_all((q, I, e), nmaps=3, cores=2, cwd=tmp_path / "all")
    assert r.rho.shape == (16, 16, 16) and len(r.maps) == 3 and not r.warnings
    assert (r.resolution, r.resolution_err) == (12.8, 2.4) and r.rg_mean == 15.146 and r.chi2 == 0.1973
    assert r.fit_map == pytest.approx(I * 1.01, rel=1e-4)              # mean of the per-map fits
    assert "-nm" in r.run["cmd"] and "-j" in r.run["cmd"]


def test_failures_are_reported(tmp_path, fake_denss):
    (fake_denss / "denss" / "scripts" / "denss_abinitio.py").write_text("print('nothing written')\n")
    with pytest.raises(D.DenssError, match="wrote no denss.mrc"):
        D.run_denss(D.sphere_curve(), cwd=tmp_path / "r1")
    (fake_denss / "denss" / "scripts" / "denss_abinitio.py").write_text("raise SystemExit('bad input')\n")
    with pytest.raises(D.DenssError, match="bad input"):
        D.run_denss(D.sphere_curve(), cwd=tmp_path / "r2")


def test_not_installed(monkeypatch, tmp_path):
    D._INFO.clear()
    with pytest.raises(D.DenssError, match="pip install denss"):
        D.find_denss(python=str(tmp_path / "no-python-here"))


def _real_denss():
    try:
        D._INFO.clear()
        return D.find_denss()
    except D.DenssError:
        return None
    finally:
        D._INFO.clear()


@pytest.mark.denss
@pytest.mark.skipif(_real_denss() is None, reason="DENSS is not installed")
def test_real_denss_sphere(tmp_path):
    """The real DENSS (through the NumPy-2 shim): sphere R = 2 nm, Rg within 10 % of 15.49 Å."""
    assert D.selftest(cwd=tmp_path, out=lambda s: None)


def test_window(tmp_path, fake_denss):
    pytest.importorskip("PySide6")
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6 import QtCore, QtWidgets
    QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    from pysasfit.core import settings as prefs
    prefs.set_value("density.workdir", str(tmp_path / "runs"))
    try:
        from pysasfit.density.gui import MainWindow
        w = MainWindow()
        w.confirm_background = False                    # no question dialog in tests
        assert w.installed and set(w.docks) >= {"data", "denss", "density", "iq", "conv", "slices"}
        q, I, e = D.sphere_curve()
        np.savetxt(tmp_path / "s.dat", np.c_[q, I, e])
        w.load_files([str(tmp_path / "s.dat")])
        compile(w.python_code(), "<pyDENsity>", "exec")
        loop = QtCore.QEventLoop()
        w._signals.done.connect(lambda *_a: QtCore.QTimer.singleShot(0, loop.quit))
        QtCore.QTimer.singleShot(30000, loop.quit)
        w.run_single()
        loop.exec()
        assert len(w.runs) == 1, w.log_view.toPlainText()
        assert w.table.item(0, 3).text() == "15.6"
        assert len(w.view3d.objects) == 1, w.log_view.toPlainText()
        assert w.view3d.objects[0]["kind"] == "mesh"
        w.level.setValue(50)
        w.update_surface()
        w.close()
    finally:
        prefs.reset()


def test_flat_background(tmp_path, fake_denss):
    """Tail fit I = A q^-B + bg (no ATSAS), subtraction as a new curve, warning before a run, session restore."""
    pytest.importorskip("PySide6")
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6 import QtWidgets
    QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    from pysasfit.core import settings as prefs
    prefs.set_value("density.workdir", str(tmp_path / "runs"))
    try:
        from pysasfit.density.gui import MainWindow
        q = np.linspace(0.06, 6.0, 400)                          # sphere R = 5 nm + flat background 0.5;
        x = q * 5.0                                              # tail q = 3..6 (qR = 15..30) is in the Porod range
        I = 1e3 * (3 * (np.sin(x) - x * np.cos(x)) / x ** 3) ** 2 + 0.5
        np.savetxt(tmp_path / "bg.dat", np.c_[q, I, 0.01 * I])
        w = MainWindow()
        w.confirm_background = False
        w.load_files([str(tmp_path / "bg.dat")])
        assert w.bg_qmin.value() == pytest.approx(q[280], rel=1e-6) and w.bg_qmax.value() == pytest.approx(q[-1])
        w._background_ok()
        assert "flat background of about" in w.log_view.toPlainText()     # the pre-run check sees it
        w.bg_qmin.setValue(3.0)
        w.bg_fix.setChecked(True)                                # B = 4
        w.fit_background()
        fit = w.bg_fits["bg.dat"]
        assert fit.bg == pytest.approx(0.5, rel=0.05) and w.bg_value.value() == fit.bg
        w.on_subtract_background()
        new = w.current()[1]
        assert "minus bg" in new.name and np.allclose(new.I, I - fit.bg) and new.name in w.derived
        n = w.log_view.toPlainText().count("flat background of about")
        w._background_ok()                                       # subtracted curve: no warning
        assert w.log_view.toPlainText().count("flat background of about") == n
        code = w.python_code()
        compile(code, "<pyDENsity>", "exec")
        assert "fit_power_background" in code and "curve.I[i0:i1] - bg" in code
        w.save_session()
        w.close()
        w2 = MainWindow()
        w2.restore_session()
        assert [w2._name(s) for s in w2.sources] == ["bg.dat", new.name]
        w2.close()
    finally:
        prefs.reset()
