import json

import numpy as np
import pytest


def _tab(nbins="3", nQ="25"):
    from pysasfit.oz.gui.ry_polydisperse_yukawa_tab import IMPORT_ERROR, RYPolydisperseYukawaTab

    assert IMPORT_ERROR is None
    t = RYPolydisperseYukawaTab()
    t.resize(1200, 800)
    t.nbinsVar.set(nbins)
    t.nQVar.set(nQ)
    return t


def _run(t, wait, timeout=60):
    n = len(t.runs)
    t._onCompute()
    assert not t.computeBtn.isEnabled() and t.interruptBtn.isEnabled()
    wait(lambda: not t.worker.is_alive() and t.computeBtn.isEnabled(), timeout)
    assert len(t.runs) == n + 1, t.statusVar.get()
    assert not t.interruptBtn.isEnabled()
    return t.result


def _direct(R0, relsd, nbins, phi, z, K, Qmin, Qmax, nQ, solverClass):
    """The library calls the tab makes: classes from the log-normal
    distribution's own quadrature, diameters sigma = 2R."""
    import pysasfit.oz.rypolydisperseWrapper as rypw
    from pysasfit.oz.polydisperse_yukawa_sas import LogNormal, discretize

    R, w = discretize(LogNormal(R0/np.sqrt(1.0 + relsd**2), np.sqrt(np.log(1.0 + relsd**2))), nbins)
    sigmas = 2.0*R
    x = w * phi * 6 / (np.pi * np.sum(w * sigmas**3))
    alpha, _res = rypw.solve_alpha(sigmas, x, z, K, bracket=(0.02, 3.0))
    ry = rypw.RYPolydisperseYukawa(sigmas, x, z, K, alpha, solverClass=solverClass)
    Q = np.logspace(np.log10(Qmin), np.log10(Qmax), nQ)
    wv = np.sqrt(x / x.sum())
    return alpha, Q, np.array([float(wv @ ry.S_matrix(q) @ wv) for q in Q])


def test_construct_defaults_and_solver_choices(qapp):
    from pysasfit.oz.gui.ry_polydisperse_yukawa_tab import DIST_NAMES, _solverChoices

    t = _tab()
    choices = _solverChoices()
    assert "scipy Anderson" in choices and "Picard / Mann" in choices
    assert list(t.solverChoices) == list(choices)
    assert t.solverVar.get() == next(iter(choices))
    for name, cls in choices.items():
        t.solverVar.set(name)
        assert t.selectedSolverClass() is cls
    # defaults as in the Tk tab
    assert (t.distVar.get(), t.ffVar.get(), t.phiVar.get(), t.zVar.get(), t.KVar.get()) == \
        ("Log-normal", "Sphere", "0.15", "0.06", "1.0")
    assert (t.QminVar.get(), t.QmaxVar.get(), t.deflateVar.get()) == ("1e-3", "0.3", "0")
    assert t.SESSION_TAB_INDEX == 2 and t.SESSION_TAB_NAME == "RY Polydisperse Yukawa"
    assert not t.interruptBtn.isEnabled()
    # the per-distribution entries are rebuilt on a distribution change
    from PySide6 import QtWidgets
    for name in DIST_NAMES:
        t.distVar.set(name)
        t._syncEnabledStates()
        qapp.processEvents()
        n = len(t._distParamInner.findChildren(QtWidgets.QLineEdit))
        assert n == {"Log-normal": 2, "Schulz-Zimm": 2, "Gamma": 2, "Beta": 4}[name]


def test_compute_matches_library(qapp, dialogs, wait):
    from pysasfit.oz.gui.ry_polydisperse_yukawa_tab import _solverChoices

    t = _tab()
    t.solverVar.set("scipy Anderson")
    r = _run(t, wait)
    assert dialogs.kinds() == []
    assert r.Q.size == 25 and r.S.shape == (25, 3, 3) and np.all(np.isfinite(r.S_number))
    assert np.allclose(r.S, np.transpose(r.S, (0, 2, 1)))
    assert t.statusVar.get() == r.label and r.label.startswith("RY K=1.0000 alpha=")
    summary = t.summaryText.toPlainText()
    assert "alpha (RY)" in summary and "components N        : 3" in summary
    assert len(t.axes["sq"].lines) == 1 and len(t.axes["sij"].lines) == 6
    assert len(t.axes["dist"].patches) == 3
    np.testing.assert_allclose(r.sigma, 2 * r.R)

    alpha, Q, S = _direct(50.0, 0.2, 3, 0.15, 0.06, 1.0, 1e-3, 0.3, 25, _solverChoices()["scipy Anderson"])
    assert r.alpha == pytest.approx(alpha, rel=1e-12)
    np.testing.assert_allclose(r.Q, Q)
    np.testing.assert_allclose(r.S_number, S, rtol=1e-10)


@pytest.mark.parametrize("solver", ["Picard / Mann", "SUNDIALS KIN_FP (Anderson)", "MDIIS"])
def test_compute_other_solvers(qapp, dialogs, wait, solver):
    from pysasfit.oz.gui.ry_polydisperse_yukawa_tab import _solverChoices

    if solver not in _solverChoices():
        pytest.skip(f"{solver} not available")
    t = _tab(nbins="2", nQ="15")
    t.solverVar.set(solver)
    r = _run(t, wait)
    assert dialogs.kinds() == [], dialogs.calls
    assert np.all(np.isfinite(r.S_number))
    # the same physical answer as the default solver
    _alpha, _Q, S = _direct(50.0, 0.2, 2, 0.15, 0.06, 1.0, 1e-3, 0.3, 15, _solverChoices()["scipy Anderson"])
    np.testing.assert_allclose(r.S_number, S, rtol=1e-3)


@pytest.mark.parametrize("dist", ["Gamma", "Beta", "Schulz-Zimm"])
def test_other_distributions_and_core_shell(qapp, dialogs, wait, dist):
    # Gamma/Beta raised KeyError 'D0' and Core-shell raised ValueError in the Tk original.
    t = _tab(nbins="3", nQ="15")
    t.distVar.set(dist)
    t._syncEnabledStates()
    if dist == "Beta":
        # narrow, symmetric Beta (rel. s.d. ~0.1); the default (a=2, b=5, rel. s.d. 0.28) spans
        # sigma 6..74 and scipy Anderson finds no fixpoint at alpha=3 -- a solver limit, not the GUI
        t.distVars[("Beta", "a")].set("20")
        t.distVars[("Beta", "b")].set("20")
    t.ffVar.set("Core-shell")
    r = _run(t, wait)
    assert dialogs.kinds() == [], dialogs.calls
    assert np.all(np.isfinite(r.S_number))
    assert dist.split("-")[0] in t.summaryText.toPlainText()


def test_interrupt(qapp, dialogs, wait):
    t = _tab(nbins="12", nQ="50")
    t._onInterrupt()
    assert t.statusVar.get() == "nothing to interrupt"
    t._onCompute()
    wait(lambda: t.solver is not None, 30)
    t._onInterrupt()
    assert t._interruptRequested
    wait(lambda: not t.worker.is_alive() and t.computeBtn.isEnabled(), 60)
    assert t.statusVar.get() == "interrupted"
    assert t.result is None and t.runs == []
    assert "error" not in dialogs.kinds()
    assert not t.interruptBtn.isEnabled()


def test_session_export_and_invalid_input(qapp, dialogs, wait, tmp_path):
    t = _tab()
    r = _run(t, wait)

    # ASCII export through the shared controls
    out = tmp_path / "ry.txt"
    dialogs.answers["save"] = str(out)
    t.exportFormatVar.set("ASCII")
    t._onExportSelected()
    lines = out.read_text().splitlines()
    assert lines[0] == "# Q\tS_number" and len(lines) == 26
    np.testing.assert_allclose(np.loadtxt(out), np.column_stack([r.Q, r.S_number]), rtol=1e-9)

    # the Tk-style CSV export
    csv = tmp_path / "ry.csv"
    dialogs.answers["save"] = str(csv)
    t._onExport()
    assert csv.read_text().splitlines()[0] == "Q,S_number"

    # session state round trip
    t.distVar.set("Beta")
    t.distVars[("Beta", "a")].set("3.5")
    t.KVar.set("-2.0")
    t.solverVar.set("Picard / Mann")
    state = t.sessionState()
    json.dumps(state)
    t2 = _tab()
    t2.restoreSessionState(state)
    assert t2.distVar.get() == "Beta" and t2.distVars[("Beta", "a")].get() == "3.5"
    assert t2.KVar.get() == "-2.0" and t2.solverVar.get() == "Picard / Mann"
    assert t2.nbinsVar.get() == "3" and t2.nQVar.get() == "25"

    # save / load through the dialog hook
    sess = tmp_path / "ry.oz2"
    dialogs.answers["save"] = str(sess)
    t._onSaveAll()
    payload = json.loads(sess.read_text())
    assert payload["tab"] == "RY Polydisperse Yukawa" and payload["tabIndex"] == 2
    t3 = _tab(nbins="7")
    dialogs.answers["open"] = str(sess)
    t3._onLoadAll()
    assert "session restored" in t3.statusVar.get()
    assert t3.nbinsVar.get() == "3" and t3.distVar.get() == "Beta" and t3.KVar.get() == "-2.0"

    # invalid input -> error dialog, no worker started
    worker = t.worker
    t.QmaxVar.set("abc")
    t._onCompute()
    assert dialogs.kinds()[-1] == "error"
    assert "Q max" in dialogs.calls[-1][1]["message"]
    assert t.worker is worker and t.computeBtn.isEnabled()
    t.QmaxVar.set("1e-4")
    t._onCompute()
    assert "Q max must exceed Q min" in dialogs.calls[-1][1]["message"]


def test_selected_distribution_is_used(qapp, dialogs, wait):
    """The classes are the selected distribution's own nodes and weights (the
    Tk tab used log-normal weights for every choice), in RADIUS, sigma = 2R."""
    from pysasfit.oz.gui.ry_polydisperse_yukawa_tab import _makeDistribution
    from pysasfit.oz.polydisperse_yukawa_sas import discretize

    t = _tab(nbins="3", nQ="10")
    t.solverVar.set("scipy Anderson")
    out = {}
    for name in ("Log-normal", "Schulz-Zimm"):           # same mean radius and rel. s.d.
        t.distVar.set(name)
        t._syncEnabledStates()
        r = _run(t, wait)
        R, w = discretize(_makeDistribution(name, {"R0": 50.0, "relsd": 0.2}), 3)
        np.testing.assert_allclose(r.R, R, rtol=1e-12)
        np.testing.assert_allclose(r.w, w, rtol=1e-12)
        np.testing.assert_allclose(r.sigma, 2 * R, rtol=1e-12)
        out[name] = r
    assert not np.allclose(out["Log-normal"].R, out["Schulz-Zimm"].R)
    assert not np.allclose(out["Log-normal"].S_number, out["Schulz-Zimm"].S_number, rtol=1e-6)
    # a width the old +-3 s.d. grid could not do (negative sizes above 1/3)
    t.distVar.set("Log-normal")
    t.distVars[("Log-normal", "relsd")].set("0.4")
    r = _run(t, wait)
    assert np.all(r.R > 0) and np.all(np.isfinite(r.S_number))


@pytest.mark.parametrize("module", ["ry_polydisperse_yukawa_tab", "polydisperse_yukawa_tab"])
@pytest.mark.parametrize("name", ["Log-normal", "Schulz-Zimm"])
def test_mean_radius_is_the_mean(module, name):
    """'Mean radius' and 'Rel. s.d.' are the number-weighted mean and relative
    s.d. of R for both distributions (log-normal used to take it as median)."""
    import importlib
    mk = importlib.import_module(f"pysasfit.oz.gui.{module}")._makeDistribution
    d = mk(name, {"R0": 50.0, "relsd": 0.3})
    m1, m2 = d.moment(1), d.moment(2)
    assert m1 == pytest.approx(50.0, rel=1e-12)
    assert np.sqrt(m2 - m1**2) / m1 == pytest.approx(0.3, rel=1e-12)
