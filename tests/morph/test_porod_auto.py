"""Automatic Porod range and background: chosen from the data, or set by hand, or switched off."""
import os

import numpy as np
import pytest

from pysasfit.morph.extrapolation import auto_porod_range, extrapolate


def _dab(noise=0.01, B=None, seed=2):
    xi = 50.0
    q = np.geomspace(2e-3, 0.5, 300)
    I = 1e4 / (1 + (xi * q) ** 2) ** 2          # Debye-Anderson-Brumberger: Porod K = 1e4 / xi^4 at q xi >> 1
    B = 0.5 * I[-1] if B is None else B
    s = noise * I + 0.01 * B
    rng = np.random.default_rng(seed)
    return q, I + B + rng.normal(0, s), s, B, 1e4 / xi ** 4


def test_auto_range_finds_background():
    q, I, s, B, K = _dab()
    i0, info = auto_porod_range(q, I, s, fit_background=True)
    assert info["clear"] and q[i0] > 3 / 50.0                      # starts in the Porod regime (q xi > 3)
    c = extrapolate(q, I, s, fit_background=True, porod_auto=True)
    assert c.porod_mode == "auto" and c.porod_range[0] == pytest.approx(q[i0])
    # the q^-6 term of the DAB law is partly absorbed by B: within 10 %
    assert c.B == pytest.approx(B, rel=0.10)
    assert c.K == pytest.approx(K, rel=0.05)
    # longer than the fixed last 15 %: a more precise B
    f = extrapolate(q, I, s, fit_background=True)
    assert f.porod_mode == "last 15 %" and c.dB < f.dB


def test_manual_range_and_fixed_background():
    q, I, s, B, K = _dab()
    c = extrapolate(q, I, s, fit_background=True, porod_qmin=0.2, porod_auto=True)       # by hand wins
    assert c.porod_mode == "qmin" and c.porod_range[0] >= 0.2
    c = extrapolate(q, I, s, fit_background=True, porod_points=30, porod_auto=True)
    assert c.porod_mode == "points" and np.sum(q >= c.porod_range[0]) == 30
    c = extrapolate(q, I, s, background=B, porod_auto=True)                               # B fixed
    assert c.B_mode == "fixed" and c.B == B and c.K == pytest.approx(K, rel=0.05)
    c = extrapolate(q, I, s, porod_auto=True)                                             # no background at all
    assert c.B_mode == "none" and c.B == 0.0


def test_no_porod_regime_is_reported():
    q = np.geomspace(1e-3, 0.05, 200)
    I = 1e4 / (1 + (50 * q) ** 2)                                      # q^-2: no Porod regime
    s = 0.01 * I
    _i0, info = auto_porod_range(q, I, s, fit_background=True)
    assert not info["clear"] and "no clear Porod regime" in info["message"]


def test_gui_defaults_and_manual_settings(tmp_path):
    pytest.importorskip("PySide6")
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6 import QtWidgets
    QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    from pysasfit.morph.aniso_gui import AnisoWindow
    from pysasfit.morph.gui import MainWindow
    q, I, s, B, K = _dab()
    p = tmp_path / "dab.dat"
    np.savetxt(p, np.column_stack([q, I, s]))
    w = MainWindow()
    assert w.fit_bg_default.isChecked() and w.porod_mode.currentData() == "auto"
    w.load_file(str(p))
    assert w.datasets[0]["fit_bg"]
    cv = w.result.curves[0].curve
    assert cv.porod_mode == "auto" and cv.B_mode == "fitted" and cv.B == pytest.approx(B, rel=0.1)
    assert "auto" in w.result.report()
    w.porod_mode.setCurrentIndex(w.porod_mode.findData("qmin"))
    w.porod_qmin.setText("0.2")
    pp = w.params()
    assert pp.porod_qmin == 0.2 and pp.porod_points is None and not pp.porod_auto
    w.porod_mode.setCurrentIndex(w.porod_mode.findData("points"))
    w.porod_points.setValue(25)
    assert w.params().porod_points == 25
    w.fit_bg_default.setChecked(False)
    w.load_file(str(p), add=True)
    assert not w.datasets[1]["fit_bg"]
    w.close()
    a = AnisoWindow()
    assert a.fit_b.isChecked() and a.porod_mode.currentData() == "auto"
    a.porod_mode.setCurrentIndex(a.porod_mode.findData("fixed15"))
    assert not a.params().porod_auto
    a.close()
