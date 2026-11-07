"""pySCAtter window: calculation, .ini files, 1D and 2D data, fits, session."""
import os
import time

import numpy as np
import pytest

pytest.importorskip("PySide6")
h5py = pytest.importorskip("h5py")


@pytest.fixture
def app():
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6 import QtWidgets
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


@pytest.fixture
def win(app, tmp_path, monkeypatch):
    monkeypatch.setenv("PYSASFIT_SETTINGS", str(tmp_path / "settings.json"))
    from pysasfit.scatter import gui as g
    monkeypatch.setattr(g.MainWindow, "_session_path", staticmethod(lambda: str(tmp_path / "scatter_session.json")))
    w = g.MainWindow()
    yield w
    w.shutdown()
    w.close()


def wait(app, w, attr="worker", timeout=120):
    t = time.time()
    while time.time() - t < timeout:
        app.processEvents()
        wk = getattr(w, attr)
        if (wk is None or not wk.isRunning()) and not getattr(w, "_pending", False):
            break
        time.sleep(0.02)
    for _ in range(10):
        app.processEvents()


def write_2d(path, qh, qv, I, dI, lam_A=6.0):
    with h5py.File(path, "w") as f:
        e = f.create_group("sasentry01")
        e.attrs["canSAS_class"] = "SASentry"
        d = e.create_group("sasdata01")
        d.attrs["canSAS_class"] = "SASdata"
        d["I"] = I
        d["I"].attrs["uncertainties"] = "Idev"
        d["Idev"] = dI
        d["Qx"] = qh
        d["Qx"].attrs["units"] = "1/nm"
        d["Qy"] = qv
        d["Qy"].attrs["units"] = "1/nm"
        e.create_group("sasprocess").attrs["canSAS_class"] = "SASprocess"
        import json
        e["sasprocess"]["metadata"] = json.dumps({"wavelength_A": lam_A, "sample": "synthetic"})
    return path


def test_calculate_and_ini_roundtrip(app, win, tmp_path):
    p = win.params()
    p.update(ComboBoxParticle=1, EditRadius=4, Length=40, Ordis=0, EditDbeta=15, theta=90, phi=90, GridPoints=24)
    win.params_panel.set_params(p, {"EditDbeta": True})
    win.calculate(two_d=True)
    wait(app, win)
    qx, qy, qz, I = win.result2d
    assert I.shape == (24, 24) and np.all(np.isfinite(I))
    # C++ frame: theta=phi=90 -> director along detector qy (vertical); index [i, j] = qy[i], qz[j]
    assert I[18, 12] > 3 * I[12, 18]           # streak along the horizontal (perpendicular to the director)
    ini = tmp_path / "a.ini"
    from pysasfit.scatter import params as P
    P.write_ini(ini, win.params(), win.params_panel.flags())
    win.params_panel.set_params(P.defaults(), {})
    win.on_open_ini(str(ini))
    wait(app, win)
    assert win.params()["Length"] == 40 and win.params_panel.flags() == {"EditDbeta": True}


def test_fit_1d_with_resolution(app, win, tmp_path):
    from pysasfit.scatter import engine
    p = win.params()
    p.update(EditRadius=8, EditSigma=0.08, I0=100, Base=0.05)
    q = np.geomspace(0.05, 2, 120)
    dq = 0.03 * q + 0.004
    I = engine.smeared_1d(p, q, dq)
    rng = np.random.default_rng(3)
    dI = 0.02 * I
    path = tmp_path / "c.dat"
    np.savetxt(path, np.column_stack([q, I + rng.normal(0, dI), dI, dq]), header="q I Idev Qdev")
    win.on_load_1d(str(path))
    wait(app, win)
    assert len(win.data1d) == 1 and win.data1d[0].dq is not None
    p0 = dict(p, EditRadius=7.0, EditSigma=0.12, I0=50.0, Base=0.0)
    win.params_panel.set_params(p0, {"EditRadius": True, "EditSigma": True, "I0": True, "Base": True})
    win.on_fit()
    wait(app, win, "fit_worker")
    wait(app, win)
    v = win.params()
    assert v["EditRadius"] == pytest.approx(8, rel=0.01) and v["EditSigma"] == pytest.approx(0.08, rel=0.05)
    assert "χ²ᵣ" in win.fit_text.toPlainText()
    win.draw_1d()
    assert len(win.canvas1d.figure.axes) == 2                     # with residuals


def test_fit_2d_orientation_width(app, win, tmp_path):
    from pysasfit.scatter import engine
    p = win.params()
    p.update(ComboBoxParticle=1, EditRadius=4, Length=40, EditSigma=0.05, Ordis=0, EditDbeta=15, theta=90, phi=90,
             I0=10)
    ax = np.linspace(-1, 1, 36)
    qh, qv = np.meshgrid(ax, ax)
    k = 2 * np.pi / 0.6
    I = engine.intensity_2d(p, -(qh ** 2 + qv ** 2) / (2 * k), qh, qv)
    path = write_2d(tmp_path / "x_2d.h5", qh, qv, I, 0.02 * I)
    win.on_load_2d(str(path))
    wait(app, win)
    assert len(win.data2d) == 1
    win.params_panel.set_params(dict(p, EditDbeta=25.0, I0=7.0), {"EditDbeta": True, "I0": True})
    win.every_px.setValue(3)
    win.on_fit()
    wait(app, win, "fit_worker", timeout=240)
    wait(app, win)
    assert win.params()["EditDbeta"] == pytest.approx(15, abs=0.3)
    for view in ("data", "split", "resid"):
        win.view2d.setCurrentIndex(win.view2d.findData(view))
        win.draw_2d()
        assert win.canvas2d.figure.axes


def test_session_roundtrip(app, win, tmp_path):
    p = win.params()
    p.update(StructureFactor=1, SFEta=0.33)
    win.params_panel.set_params(p, {"SFEta": True})
    win.fit_qmax.setText("1.5")
    win.save_session()
    win.params_panel.set_params({}, {})
    win.fit_qmax.setText("")
    assert win.restore_session()
    wait(app, win)
    assert win.params()["SFEta"] == pytest.approx(0.33) and win.params_panel.flags() == {"SFEta": True}
    assert win.fit_qmax.text() == "1.5"
