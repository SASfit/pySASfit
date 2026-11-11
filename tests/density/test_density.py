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


# ------------------------------------------------- Dmax when DENSS's own estimate fails (large particles)
FAKE_DMAX_CRASH = '''
import sys
if "-d" not in sys.argv:          # DENSS 1.8.7: estimate_dmax -> Sasrec -> shannon_channels(NaN)
    sys.stderr.write('  File "core.py", line 942, in estimate_dmax\\nValueError: cannot convert float NaN to integer\\n')
    raise SystemExit(1)
'''


def large_particle_curve(R_nm=250.0, seed=None):
    """Sphere of radius R (Rg = 194 nm) measured from q = 0.004 1/nm on: only a few Guinier points.  With noise
    (``seed`` 0) DENSS 1.8.7's own Dmax estimate stops with 'cannot convert float NaN to integer'."""
    q = np.geomspace(0.004, 0.385, 91)
    x = q * R_nm
    I = 3500.0 * (3 * (np.sin(x) - x * np.cos(x)) / x ** 3) ** 2
    e = 0.025 * I + 0.01
    if seed is not None:
        I = I + np.random.default_rng(seed).normal(0, e)
    return q, I, e


def test_guinier_rg_uses_the_guinier_range():
    q = np.linspace(0.01, 1.0, 200)
    rg = 3.0
    I = 100 * np.exp(-(q * rg) ** 2 / 3)
    r, I0, n = D.guinier_rg(q, I, 0.01 * I)
    assert r == pytest.approx(rg, rel=1e-6) and I0 == pytest.approx(100, rel=1e-6)
    assert q[n - 1] * r <= 1.3 < q[n] * r
    r, _I0, n = D.guinier_rg(*large_particle_curve())          # sphere: Rg = sqrt(3/5) R = 193.6 nm
    assert r == pytest.approx(193.6, rel=0.05) and n >= 5
    with pytest.raises(ValueError, match="Guinier"):
        D.guinier_rg(q, np.exp(q), None)                          # rising: no Guinier law


def test_dmax_fallback_when_denss_cannot_estimate_it(tmp_path, fake_denss):
    script = fake_denss / "denss" / "scripts" / "denss_abinitio.py"
    script.write_text(FAKE_DMAX_CRASH + script.read_text())
    q, I, e = large_particle_curve()
    r = D.run_denss((q, I, e), cwd=tmp_path / "run")
    rg = D.guinier_rg(q, I, e)[0]
    assert r.dmax_nm == pytest.approx(3.5 * rg)
    cmd = r.run["cmd"]
    assert float(cmd[cmd.index("-d") + 1]) == pytest.approx(35 * rg, rel=1e-5)   # Å
    assert len(r.warnings) == 1 and "could not estimate Dmax" in r.warnings[0]
    # a given Dmax is never replaced: the error stays
    script.write_text(FAKE_DMAX_CRASH.replace('"-d" not in sys.argv', "True"))
    with pytest.raises(D.DenssError, match="estimate_dmax"):
        D.run_denss((q, I, e), dmax_nm=400, cwd=tmp_path / "run2")
    # a file as input (q in 1/nm) works the same way
    script.write_text(FAKE_DMAX_CRASH + FAKE_ABINITIO)
    f = tmp_path / "curve.dat"
    np.savetxt(f, np.c_[q, I, e])
    r = D.run_denss(f, cwd=tmp_path / "run3")
    assert r.dmax_nm == pytest.approx(3.5 * rg)


def test_no_guinier_law_gives_a_clear_message(tmp_path, fake_denss):
    script = fake_denss / "denss" / "scripts" / "denss_abinitio.py"
    script.write_text(FAKE_DMAX_CRASH + script.read_text())
    q = np.linspace(0.01, 1.0, 50)
    with pytest.raises(D.DenssError, match="enter Dmax"):
        D.run_denss((q, np.exp(q), 0.01 * np.exp(q)), cwd=tmp_path / "run")


def test_a_far_too_large_dmax_of_denss_is_flagged(tmp_path, fake_denss):
    script = fake_denss / "denss" / "scripts" / "denss_abinitio.py"
    script.write_text(script.read_text().replace("Maximum dimension: 40.000", "Maximum dimension: 30000.0"))
    r = D.run_denss(large_particle_curve(), cwd=tmp_path / "run")      # Rg 194 nm, DENSS: Dmax 3000 nm
    assert r.dmax_nm is None and len(r.warnings) == 1 and "far too large" in r.warnings[0]
    r = D.run_denss(large_particle_curve(), dmax_nm=3000, cwd=tmp_path / "run2")   # the user's choice: no warning
    assert not r.warnings


@pytest.mark.denss
@pytest.mark.skipif(_real_denss() is None, reason="DENSS is not installed")
def test_real_denss_large_particle(tmp_path):
    """DENSS 1.8.7 crashes in its Dmax estimate on this curve; pyDENsity runs it with 3.5 Guinier Rg."""
    r = D.run_denss(large_particle_curve(seed=0), seed=1, cwd=tmp_path / "run")
    assert r.dmax_nm == pytest.approx(3.5 * 193.6, rel=0.1)
    assert "could not estimate Dmax" in r.warnings[0] and np.all(np.isfinite(r.rho))


# ------------------------------------- denss-all workers are fresh processes (spawn: Windows, macOS; forkserver)
FAKE_WORKER = '''

def make_map(args):
    """One map in a worker process (like denss.core.multi_denss): NumPy-2 names and write_mrc must be patched."""
    import numpy as np
    import denss
    i, out, o = args
    np.in1d(np.arange(3), [1]); np.trapz([1, 2, 3])
    rho = np.zeros((16, 16, 16)); rho[4:12, 4:12, 4:12] = 1.0
    denss.write_mrc(rho, np.array([48.0]), out + "/%s_%d_aligned.mrc" % (o, i))
    return i
'''


def test_denss_all_workers_get_the_numpy2_patches(tmp_path, fake_denss):
    (fake_denss / "denss" / "__init__.py").write_text('__version__ = "0.0-fake"\nfrom .core import write_mrc\n')
    core = fake_denss / "denss" / "core.py"
    core.write_text(core.read_text() + FAKE_WORKER)
    body = FAKE_ALL.replace(
        'for i in range(a.nm):\n    denss.write_mrc(rho, 16 * 3.0, out + "/%s_%d_aligned.mrc" % (a.o, i))',
        'import multiprocessing as mp, denss.core\n'
        'with mp.get_context("spawn").Pool(2) as pool:\n'
        '    pool.map(denss.core.make_map, [(i, out, a.o) for i in range(a.nm)])')
    assert "spawn" in body
    # spawned workers import the script again as __mp_main__: guarded like the real denss_all.py
    body = "def main():\n" + "".join("    " + ln + "\n" for ln in body.strip().splitlines()) + \
        'if __name__ == "__main__":\n    main()\n'
    (fake_denss / "denss" / "scripts" / "denss_all.py").write_text(body)
    r = D.run_denss_all(D.sphere_curve(), nmaps=3, cores=2, cwd=tmp_path / "all")
    assert len(r.maps) == 3 and not r.warnings


def test_the_run_sitecustomize_keeps_a_shadowed_one(tmp_path, fake_denss, monkeypatch):
    other = tmp_path / "other"
    other.mkdir()
    (other / "sitecustomize.py").write_text("import os\nos.environ['PYDENSITY_TEST_OTHER'] = 'ran'\n")
    env = D._shim_env({"PYTHONPATH": str(other)}, tmp_path / "run")
    import subprocess
    import sys
    code = "import os, numpy as np; print(hasattr(np, 'in1d'), os.environ.get('PYDENSITY_TEST_OTHER'))"
    out = subprocess.run([sys.executable, "-c", code], env={**os.environ, **env},
                         capture_output=True, text=True, check=True).stdout.split()
    assert out == ["True", "ran"]
