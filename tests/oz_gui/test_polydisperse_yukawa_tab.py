"""Offscreen tests of the Qt port of the polydisperse hard-sphere-Yukawa tab."""

import numpy as np
import pytest

NQ = 30


def _tab(qapp, **settings):
    from pysasfit.oz.gui.polydisperse_yukawa_tab import IMPORT_ERROR, PolydisperseYukawaTab

    assert IMPORT_ERROR is None
    t = PolydisperseYukawaTab()
    t.resize(1200, 800)
    t.nQVar.set(str(NQ))
    t.nbinsVar.set("4")
    for name, value in settings.items():
        getattr(t, name).set(value)
    t._syncEnabledStates()
    return t


def _run(t, wait):
    t._onCompute()
    assert not t.computeButton.isEnabled()
    wait(lambda: not t.worker.is_alive() and t.resultQueue.empty() and t.computeButton.isEnabled(), 60)


def _direct(dist, pars, phi=0.15, z=0.06, K=1.0, nbins=4, ff=None, rmsa=False, momentRule=False):
    """The library called directly, exactly as the tab's worker does."""
    from pysasfit.oz.gui.polydisperse_yukawa_tab import _apply_rmsa, _makeDistribution
    from pysasfit.oz.polydisperse_yukawa_sas import PolydisperseYukawaSAS, Sphere

    d = _makeDistribution(dist, pars, momentRule=momentRule)
    sas = PolydisperseYukawaSAS(d, phi, z, 0.0 if rmsa else K, formfactor=ff or Sphere(), nbins=nbins, nsig=5.0)
    if rmsa:
        sas.K = K
        assert _apply_rmsa(sas, K)["lam"] is not None
    Q = np.logspace(-3, np.log10(0.3), NQ)
    return sas, Q


def test_construction_defaults_and_panel(qapp):
    t = _tab(qapp)
    assert t.statusVar.get() == "ready"
    assert t.notebook.count() == 6 and t.notebook.tabText(5) == "Summary"
    assert t.axes["iq"].get_xscale() == "log" and t.axes["iq"].get_yscale() == "log"
    # only decoupling is ticked by default
    assert [lab for lab, (v, _) in t.approxVars.items() if v.get()] == ["1  decoupling (Kotlarchyk-Chen)"]
    # Sphere: core-shell entries disabled; Core-shell enables them
    assert not t.coreFracEntry.isEnabled() and not t.rhoShellEntry.isEnabled()
    t.ffVar.set("Core-shell")
    t._syncEnabledStates()
    assert t.coreFracEntry.isEnabled() and t.rhoCoreEntry.isEnabled() and t.rhoShellEntry.isEnabled()

    from PySide6 import QtWidgets

    def panel():
        qapp.processEvents()
        return t.distParamFrame.findChildren(QtWidgets.QLineEdit), t.distParamFrame.findChildren(QtWidgets.QCheckBox)

    edits, checks = panel()
    assert len(edits) == 2 and len(checks) == 1          # Log-normal: R0, relsd + moment-exact
    t.distVar.set("Beta")
    t._syncEnabledStates()
    edits, checks = panel()
    assert len(edits) == 4 and not checks
    # values survive switching away and back
    t.distVars[("Beta", "a")].set("3.5")
    t.distVar.set("Log-normal")
    t._syncEnabledStates()
    t.distVar.set("Beta")
    t._syncEnabledStates()
    edits, _ = panel()
    assert "3.5" in [e.text() for e in edits]
    # rebuilding destroyed the old checkbox; setting the variable must not touch it
    t.distVar.set("Log-normal")
    t._syncEnabledStates()
    edits, checks = panel()
    t.momentRuleVar.set(True)
    assert checks[0].isChecked()


def test_msa_lognormal_sphere_matches_library_then_export_and_clear(qapp, dialogs, wait, tmp_path):
    t = _tab(qapp)
    _run(t, wait)
    assert t.statusVar.get() == "done  (MSA)"
    r = t.result
    sas, Q = _direct("Log-normal", {"R0": 50.0, "relsd": 0.2})
    np.testing.assert_allclose(r.Q, Q)
    np.testing.assert_allclose(r.I_exact, sas.I_exact(Q), rtol=1e-12)
    np.testing.assert_allclose(r.S_number, sas.S_number(Q), rtol=1e-12)
    np.testing.assert_allclose(r.approx["1  decoupling (Kotlarchyk-Chen)"], sas.I_decoupling(Q), rtol=1e-12)
    assert r.S.shape == (NQ, 4, 4) and r.g.shape == (4, 4)

    text = t.summaryText.toPlainText()
    for key in ("components N        : 4", "contact g_ij", "symmetry check", "<R^6>", "Gamma"):
        assert key in text
    assert len(t.axes["iq"].lines) == 2 and len(t.axes["err"].lines) == 1
    assert len(t.axes["sij"].lines) == 5                  # 3 diagonals + 1 off-diagonal + S=1 line
    assert len(t.axes["dist"].patches) == 4

    out = tmp_path / "yuk.dat"
    dialogs.answers["save"] = str(out)
    t._onExport()
    assert dialogs.kinds() == ["save"]
    assert t.statusVar.get() == "exported yuk.dat"
    lines = out.read_text().splitlines()
    assert lines[0] == "# polydisperse hard-sphere-Yukawa, MSA"
    header = [ln for ln in lines if ln.startswith("# Q\t")]
    assert header == ["# Q\tI_exact\tS_number\t1_decoupling"]
    data = np.loadtxt(out)
    assert data.shape == (NQ, 4)
    np.testing.assert_allclose(data[:, 1], r.I_exact, rtol=1e-7)

    # cancelled dialog -> nothing happens
    dialogs.answers["save"] = ""
    t._onExport()
    assert t.statusVar.get() == "exported yuk.dat"
    # unwritable path -> error dialog
    dialogs.answers["save"] = str(tmp_path / "no" / "such" / "dir.dat")
    t._onExport()
    assert dialogs.kinds()[-1] == "error"

    t._onClear()
    assert t.result is None and t.statusVar.get() == "cleared"
    assert t.summaryText.toPlainText() == "" and not t.axes["iq"].lines
    assert t.axes["iq"].get_xscale() == "log"
    t._onExport()
    assert t.statusVar.get().startswith("nothing to export")


@pytest.mark.parametrize("dist, ff", [
    ("Log-normal", "Core-shell"),
    ("Schulz-Zimm", "Sphere"),
    ("Schulz-Zimm", "Core-shell"),
    ("Gamma", "Sphere"),
    ("Gamma", "Core-shell"),
    ("Beta", "Sphere"),
    ("Beta", "Core-shell"),
])
def test_distributions_and_form_factors(qapp, wait, dist, ff):
    from pysasfit.oz.gui.polydisperse_yukawa_tab import DIST_PARAMS
    from pysasfit.oz.polydisperse_yukawa_sas import CoreShell

    t = _tab(qapp, distVar=dist, ffVar=ff, coreFracVar="0.5")
    _run(t, wait)
    assert t.statusVar.get() == "done  (MSA)", t.summaryText.toPlainText()
    pars = {key: float(default) for _l, key, default, _t in DIST_PARAMS[dist]}
    cs = CoreShell(rho_core=2.0, rho_shell=1.0, rho_solvent=0.0, ratio=0.5) if ff == "Core-shell" else None
    sas, Q = _direct(dist, pars, ff=cs)
    np.testing.assert_allclose(t.result.I_exact, sas.I_exact(Q), rtol=1e-12)
    assert np.all(np.isfinite(t.result.S))


def test_lognormal_moment_exact_nodes(qapp, wait):
    t = _tab(qapp, momentRuleVar=True)
    _run(t, wait)
    assert t.statusVar.get() == "done  (MSA)", t.summaryText.toPlainText()
    sas, Q = _direct("Log-normal", {"R0": 50.0, "relsd": 0.2}, momentRule=True)
    np.testing.assert_allclose(t.result.I_exact, sas.I_exact(Q), rtol=1e-12)


def test_rmsa_matches_library(qapp, wait):
    t = _tab(qapp, closureVar="RMSA (rescaled)", KVar="-20")
    _run(t, wait)
    assert t.statusVar.get().startswith("done  (RMSA lambda=")
    sas, Q = _direct("Log-normal", {"R0": 50.0, "relsd": 0.2}, K=-20.0, rmsa=True)
    np.testing.assert_allclose(t.result.I_exact, sas.I_exact(Q), rtol=1e-10)
    text = t.summaryText.toPlainText()
    assert text.startswith("RMSA: lambda = ") and "(rescaled)" in text
    assert "contact g_ij        : 0.000000" in text and "WARNING" not in text


def test_routes(qapp, wait):
    from pysasfit.oz.gui.polydisperse_yukawa_tab import ROUTE_AMPL, ROUTE_BOTH

    t = _tab(qapp, routeVar=ROUTE_BOTH)
    _run(t, wait)
    line = [ln for ln in t.summaryText.toPlainText().splitlines() if ln.startswith("route cross-check")]
    assert line and float(line[0].split(":")[1].split()[0]) < 1e-8
    ref = t.result.I_exact
    t.routeVar.set(ROUTE_AMPL)
    _run(t, wait)
    np.testing.assert_allclose(t.result.I_exact, ref, rtol=1e-8)
    # K = 0: amplitude route is singular, falls back to Gamma
    t.KVar.set("0")
    _run(t, wait)
    assert t.statusVar.get() == "done  (MSA)"


def test_all_schemes_and_single_component(qapp, wait):
    t = _tab(qapp, nbinsVar="3")
    for v, _attr in t.approxVars.values():
        v.set(True)
    _run(t, wait)
    r = t.result
    assert set(r.approx) | set(_notes(t)) == set(t.approxVars)
    assert len(t.axes["iq"].lines) == 1 + len(r.approx)
    # N = 1 (accepted by the input check) used to crash in np.gradient in _replot
    t.nbinsVar.set("1")
    for v, _attr in t.approxVars.values():
        v.set(False)
    _run(t, wait)
    assert t.statusVar.get() == "done  (MSA)" and t.result.S.shape == (NQ, 1, 1)
    assert len(t.axes["sij"].lines) == 2
    assert len(t.axes["err"].texts) == 1                  # "no schemes selected"


def _notes(t):
    text = t.summaryText.toPlainText()
    return [lab for lab in t.approxVars if lab + ":\n    unavailable" in text]


def test_invalid_input_and_solver_failures(qapp, dialogs, wait):
    t = _tab(qapp)
    t.QmaxVar.set("abc")
    t._onCompute()
    assert dialogs.calls[-1][0] == "error" and dialogs.calls[-1][1]["title"] == "Invalid input"
    assert t.worker is None and t.statusVar.get() == "Q max: 'abc' is not a number"
    t.QmaxVar.set("1e-4")
    t._onCompute()
    assert t.statusVar.get() == "Q max must exceed Q min"
    t.QmaxVar.set("0.3")
    t.distVar.set("Beta")
    t.distVars[("Beta", "Rmax")].set("10")
    t._onCompute()
    assert t.statusVar.get() == "Beta: R max must be greater than R min"
    assert dialogs.kinds() == ["error"] * 3
    t.distVar.set("Log-normal")

    # MSA without a physical root -> explanation in the Summary tab
    t.KVar.set("20")
    _run(t, wait)
    assert t.statusVar.get() == "failed -- see Summary tab"
    assert t.summaryText.toPlainText().startswith("The MSA closure has no physical solution")
    assert t.notebook.currentIndex() == 5

    # RMSA without a rescaling factor
    t.notebook.setCurrentIndex(0)
    t.KVar.set("-50")
    t.closureVar.set("RMSA (rescaled)")
    _run(t, wait)
    assert t.summaryText.toPlainText().startswith("RMSA could not find a rescaling factor")
    assert t.notebook.currentIndex() == 5 and t.result is None
