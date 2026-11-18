"""SESANS in pySASmorph: exact Abel projection, inversion, joint use with scattering curves, GUI."""

import os

import numpy as np
import pytest

from pysasfit.morph.multiphase import (DataSet, MultiCut, MultiParameters, PhaseSystem, correlations_multi,
                                       reconstruct_multi)
from pysasfit.morph.scattering import levelcut_intensity
from pysasfit.morph.sesans import dab_sesans, invert_sesans, project, projection_matrix, voxel_correlation


def gfun(r, xi=600.0, d=1500.0):
    kk = 2 * np.pi / d
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.exp(-(r / xi) ** 2) * np.where(r > 0, np.sin(kk * r) / (kk * r), 1.0)


PS = PhaseSystem(["solid", "pore"], [0.7, 0.3])
SLD = np.array([1.0, 0.0])


def synthetic(noise=0.01, q=(0.01, 0.3), delta=(50.0, 8000.0), seed=0):
    """Scattering curve and SESANS curve (I in 1/cm, G in A^-2 cm^-1 for SLD in 1e-6/A^2: scale 1e-4) of a
    level-cut structure with large domains (1500 A): the scattering curve alone starts too late."""
    mc = MultiCut(PS)
    rr = np.linspace(0, 20000, 80001)
    cov = mc.covariance(SLD, gfun(rr))
    rng = np.random.default_rng(seed)
    qq = np.geomspace(*q, 200)
    I = levelcut_intensity(qq, rr, cov, 1e-4)
    d = np.geomspace(*delta, 40)
    G = 1e-4 * projection_matrix(d, rr) @ cov
    dG = noise * np.abs(G).max() * np.ones_like(G)
    sas = DataSet(qq, I * (1 + noise * rng.normal(size=qq.size)), noise * I, SLD, name="sans")
    ses = DataSet(d, G + dG * rng.normal(size=d.size), dG, SLD, name="sesans", kind="sesans")
    return sas, ses


# ------------------------------------------------------------------ projection and inversion
def test_projection_is_exact_for_dab():
    xi = 50.0
    r = np.linspace(0, 3000, 6001)
    d = np.geomspace(5, 1500, 40)
    assert np.max(np.abs(project(d, r, np.exp(-r / xi)) - dab_sesans(d, xi))) < 2e-5 * 2 * xi
    assert dab_sesans([1e9], xi)[0] == pytest.approx(-2 * xi, rel=1e-6)       # -G(0) = -2 int C dr


def test_projection_agrees_with_pysasstudio_hankel():
    hankel = pytest.importorskip("pysasfit.studio.hankel")
    xi = 50.0
    d = np.geomspace(5, 1500, 30)
    G = hankel.SESANSTransform(observable="G").g_from(d, lambda q: 8 * np.pi * xi ** 3 / (1 + (q * xi) ** 2) ** 2)
    r = np.linspace(0, 3000, 6001)
    assert np.max(np.abs(G - project(d, r, np.exp(-r / xi)))) < 1e-3 * 2 * xi


def test_inversion_recovers_the_correlation_function():
    xi = 50.0
    r = np.linspace(0, 3000, 6001)
    d = np.geomspace(5, 1500, 40)
    G = project(d, r, np.exp(-r / xi))
    dG = 0.005 * np.abs(G).max() * np.ones_like(G)
    inv = invert_sesans(d, G + np.random.default_rng(1).normal(size=d.size) * dG, dG)
    assert inv.C0 == pytest.approx(1.0, abs=0.05)
    rr = np.array([20.0, 50.0, 100.0, 200.0])
    assert np.allclose(inv.gamma(rr), np.exp(-rr / xi), atol=0.04)      # the cusp of exp(-r/xi) at 0 is the hardest
    assert 0.3 < inv.chi2 < 2 and inv.saturated


def test_voxel_correlation_of_random_field():
    rng = np.random.default_rng(0)
    f = rng.normal(size=(32, 32, 32))
    r, C = voxel_correlation(f, 2.0)
    assert r[0] == 0 and C[0] == pytest.approx(f.var(), rel=1e-6) and np.all(np.abs(C[1:]) < 0.05)


# ------------------------------------------------------------------ joint use
def test_sesans_alone():
    _sas, ses = synthetic()
    res = correlations_multi([ses], MultiParameters(names=PS.names, fractions=[0.7, 0.3], box_size=6000, n=48))
    m = res.r < 3000
    assert np.max(np.abs(res.g[m] - gfun(res.r[m]))) < 0.03
    c = res.curves[0]
    assert c.kind == "sesans" and c.chi2 < 2
    assert c.scale == pytest.approx(1e-4, rel=0.05)


def test_sesans_extends_scattering_curve():
    sas, ses = synthetic()
    p = dict(names=PS.names, fractions=[0.7, 0.3], box_size=6000, n=48)
    alone = correlations_multi([sas], MultiParameters(**p))
    both = correlations_multi([sas, ses], MultiParameters(**p))
    m = both.r < 3000
    ma = alone.r < 3000
    err_alone = np.max(np.abs(alone.g[ma] - gfun(alone.r[ma])))
    err_both = np.max(np.abs(both.g[m] - gfun(both.r[m])))
    assert err_alone > 0.15 and err_both < 0.06                     # q_min = 0.01 misses the 1500 A domains
    c_sas, c_ses = both.curves
    assert c_sas.curve.low_q == "model (SESANS)"
    assert c_sas.chi2 < 2 and c_ses.chi2 < 3
    assert "SESANS" in both.report()


def test_reconstruct_with_sesans_and_export(tmp_path):
    from pysasfit.morph.export import write_curves_multi

    sas, ses = synthetic()
    res = reconstruct_multi([sas, ses], MultiParameters(names=PS.names, fractions=[0.7, 0.3], box_size=6000,
                                                        n=48, seed=1))
    c = res.curves[1]
    assert c.I_voxel is not None and c.q_voxel.max() <= 3000
    ratio = c.I_voxel / np.interp(c.q_voxel, c.q_data, c.I_data)
    assert 0.75 < np.median(ratio) < 1.25
    files = write_curves_multi(res, str(tmp_path / "x"))
    assert any(f.endswith("sesans_sesans.txt") for f in files) and all(os.path.isfile(f) for f in files)


# ------------------------------------------------------------------ GUI
def _write_ses(path, ses, wavelength=2.0, thickness=0.2):
    lam = np.full(ses.q.size, wavelength)
    P = np.exp(lam ** 2 * thickness * ses.I)
    with open(path, "w") as fh:
        fh.write("FileFormatVersion 1.0\nDataFileTitle synthetic\nSample synthetic\n"
                 f"Thickness {thickness}\nThickness_unit cm\nTheta_zmax 0.02\nTheta_zmax_unit radians\n"
                 "Orientation Z\nSpinEchoLength_unit A\nDepolarisation_unit A-2 cm-1\nWavelength_unit A\n\n"
                 "BEGIN_DATA\nSpinEchoLength Depolarisation Depolarisation_error SpinEchoLength_error Wavelength "
                 "Wavelength_error Polarisation Polarisation_error\n")
        for row in zip(ses.q, ses.I, ses.sigma, 0.01 * ses.q, lam, 0.01 * lam, P, 0.001 * P):
            fh.write(" ".join(f"{v:.8g}" for v in row) + "\n")
    return path


def test_gui_with_sesans(tmp_path):
    pytest.importorskip("PySide6")
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6 import QtWidgets

    from pysasfit.morph.gui import MainWindow

    QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    sas, ses = synthetic()
    sas_path = tmp_path / "sans.dat"
    np.savetxt(sas_path, np.column_stack([sas.q, sas.I, sas.sigma]))
    ses_path = _write_ses(tmp_path / "sample.ses", ses)
    w = MainWindow()
    w.load_file(sas_path, correlations=False)
    assert w.load_sesans_file(ses_path, add=True, correlations=False)
    assert w.datasets[1]["kind"] == "sesans" and "SESANS" in w.curve_table.item(1, 1).text()
    assert np.allclose(w.datasets[1]["I"], ses.I, rtol=1e-5)
    w.set_phases(["solid", "pore"], [0.7, 0.3], [[1, 0], [1, 0]])
    w.box.setText("6000")
    w.n.setValue(32)
    w.seed.setValue(2)
    w.on_reconstruct()
    res = w.result
    assert [c.kind for c in res.curves] == ["sas", "sesans"]
    assert w.ax_ses.get_ylabel().startswith("G(δ)")
    w.ses_mode.setCurrentIndex(1)                                   # polarisation
    assert w.ax_ses.get_ylabel() == "P/P0"
    code = w.python_code()
    assert "load_sesans(" in code and ("kind='sesans'" in code or 'kind="sesans"' in code)
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
    from pycode_util import run_script
    ns = run_script(code)
    assert np.allclose(ns["res"].g, res.g, atol=1e-6) and ns["res"].curves[1].kind == "sesans"
    w.unit.setCurrentIndex(1)                                       # nm: SESANS read again in nm
    assert w.datasets[1]["q"][0] == pytest.approx(ses.q[0] / 10, rel=1e-6)
    w.unit.setCurrentIndex(0)
    w.save_session()
    w2 = MainWindow()
    assert w2.restore_session()
    assert [d.get("kind", "sas") for d in w2.datasets] == ["sas", "sesans"]
    w2.on_correlations()
    assert w2.result.curves[1].kind == "sesans"
    w.close()
    w2.close()


def test_open_and_add_accept_ses_files(tmp_path, monkeypatch):
    """Open…/Add… take SAS and SESANS files in one selection; SESANS files go through the unit dialog."""
    pytest.importorskip("PySide6")
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6 import QtWidgets

    from pysasfit.morph import gui
    from pysasfit.morph.gui import MainWindow

    QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    sas, ses = synthetic()
    sas_path = tmp_path / "sans.dat"
    np.savetxt(sas_path, np.column_stack([sas.q, sas.I, sas.sigma]))
    ses_path = _write_ses(tmp_path / "sample.ses", ses)
    seen = {}

    def fake_files(parent, title, start, filt):
        seen["filter"] = filt
        return [str(sas_path), str(ses_path)], ""

    shown = []
    monkeypatch.setattr(QtWidgets.QFileDialog, "getOpenFileNames", staticmethod(fake_files))
    monkeypatch.setattr(gui.SESANSLoadDialog, "exec",
                        lambda self: shown.append(self.path) or QtWidgets.QDialog.DialogCode.Accepted)
    w = MainWindow()
    calls = []
    monkeypatch.setattr(w, "on_correlations", lambda: calls.append(1))
    w.on_open()
    assert "*.ses" in seen["filter"]
    assert [d.get("kind", "sas") for d in w.datasets] == ["sas", "sesans"]
    assert shown == [str(ses_path)] and calls == [1]                 # dialog for the .ses only, one calculation
    assert np.allclose(w.datasets[1]["I"], ses.I, rtol=1e-5)
    # Add… keeps what is loaded; a cancelled unit dialog skips the file
    monkeypatch.setattr(gui.SESANSLoadDialog, "exec", lambda self: QtWidgets.QDialog.DialogCode.Rejected)
    w.on_add()
    assert [d.get("kind", "sas") for d in w.datasets] == ["sas", "sesans", "sas"]
    # Open… with only a SESANS file replaces everything
    monkeypatch.setattr(QtWidgets.QFileDialog, "getOpenFileNames",
                        staticmethod(lambda *a: ([str(ses_path)], "")))
    monkeypatch.setattr(gui.SESANSLoadDialog, "exec", lambda self: QtWidgets.QDialog.DialogCode.Accepted)
    w.on_open()
    assert [d.get("kind", "sas") for d in w.datasets] == ["sesans"]
    w.close()


def test_positive_sesans_convention_is_accepted():
    """Files that store G(0) - G(delta) (positive) gave C(0) < 0, a box of 0 and a ZeroDivisionError."""
    from pysasfit.morph.multiphase import sesans_sign
    _sas, ses = synthetic()
    flipped = DataSet(ses.q, -ses.I, ses.sigma, SLD, name="pos", kind="sesans")
    assert sesans_sign(ses.q, ses.I) == -1 and sesans_sign(flipped.q, flipped.I) == 1
    p = dict(names=PS.names, fractions=PS.fractions, n=32)
    ref = correlations_multi([ses], MultiParameters(**p))
    res = correlations_multi([flipped], MultiParameters(**p))
    assert res.params.box_size > 0 and res.params.box_size == pytest.approx(ref.params.box_size)
    assert np.allclose(res.g, ref.g) and res.curves[0].chi2 == pytest.approx(ref.curves[0].chi2)
    assert any("sign was reversed" in x for x in res.warnings)
    assert res.params.box_size >= 2 * ses.q.max() or "the box is smaller" in " ".join(res.warnings)


def test_inversion_of_linearly_spaced_sphere_data():
    """Linearly scanned spin-echo lengths (as se008731_01.ses, 2 um polystyrene): a geometric node grid left
    C at small r undetermined and gave C(0) < 0 (box 0, ZeroDivisionError in the GUI)."""
    from pysasfit.studio.hankel import sphere_sesans
    R = 10600.0
    d = np.linspace(390.0, 47000.0, 200)
    G0 = -0.21
    rng = np.random.default_rng(3)
    dG = np.full(d.size, 0.0037)
    G = G0 * (1 - sphere_sesans(d, R)) + dG * rng.normal(size=d.size)
    inv = invert_sesans(d, G, dG)
    assert inv.C0 > 0 and inv.chi2 < 1.3
    # sphere: C(0) = -G(inf) / (2 * integral gamma dr), integral gamma_sphere dr = 3R/4 ... gamma(2R) = 0
    assert inv.C0 == pytest.approx(-G0 / (2 * 0.75 * R), rel=0.15)
    assert abs(inv.gamma(np.array([2.3 * R]))[0]) < 0.05 and inv.gamma(np.array([R]))[0] > 0.3
