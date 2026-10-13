"""Offscreen tests of the Qt port of the "Polydisperse (any potential)" tab."""
import json
import shutil
import time
from pathlib import Path

import numpy as np
import pytest

DATA = Path(__file__).resolve().parents[1] / "oz" / "STJ1_lowres.dat"

# A coarse grid keeps each OZ solve in the millisecond range.
FAST_SETTINGS = {"gridN": 1023, "pointsPerSigma": 50, "maxIterations": 4000, "transformType": 1}


@pytest.fixture
def tab(qapp):
    from pysasfit.oz.gui.generic_polydisperse_tab import IMPORT_ERROR, GenericPolydisperseTab

    assert IMPORT_ERROR is None
    t = GenericPolydisperseTab()
    t.resize(1400, 900)
    yield t
    if t.worker is not None and t.worker.is_alive():
        t._abortFit = True
        t.worker.join(30)
    t.deleteLater()


def _fast(t, nQ="40"):
    t._settings = dict(FAST_SETTINGS)
    t.nQVar.set(nQ)
    t.nFFVar.set("12")
    t.nbinsVar.set("2")


def _compute(t, wait):
    t._onCompute()
    wait(lambda: not t.worker.is_alive(), 60)
    wait(lambda: t.computeBtn.isEnabled(), 10)


def _load(t, dialogs, tmp_path):
    # copied, so the fit's autosave (restore.oz1 beside the data) lands in tmp_path
    path = tmp_path / DATA.name
    shutil.copy(DATA, path)
    dialogs.answers["open"] = str(path)
    t._onLoadData()
    return path


def _hover(widget, qapp):
    from PySide6 import QtCore, QtWidgets

    QtWidgets.QApplication.sendEvent(widget, QtCore.QEvent(QtCore.QEvent.Type.Enter))


def _leave(widget):
    from PySide6 import QtCore, QtWidgets

    QtWidgets.QApplication.sendEvent(widget, QtCore.QEvent(QtCore.QEvent.Type.Leave))


# ---------------------------------------------------------------- controls
def test_construct_and_dependent_controls(qapp, tab):
    from pysasfit.oz import ozLib

    t = tab
    pots = t._availablePotentials()
    assert "HardSphere" in pots and "PolydisperseHardCoreYukawa" in pots
    assert not {"DLVO", "DLVOHydra", "IonicMicrogel"} & set(pots)
    assert list(t.solverChoices)[0] == "SUNDIALS KIN_FP"
    # hard spheres: no potential parameters, PY: no closure parameter
    assert t.potParamVars == [] and t.closureParamVar is None
    assert not t.findAlphaCheck.isEnabled()
    assert t.fitBtn.isEnabled() is False and t.smearCheck.isEnabled() is False
    # default fit flags (ticked = free)
    assert {n for n, v in t.fitFlags.items() if v.get()} == {"meanRadius", "srel", "phi", "scale", "background",
                                                               "porodA"}
    # Sphere: core-shell-only fields greyed, across value, vary box and bounds
    assert not t._fitChecks["shell"].isEnabled() and not t._fitEntries["shell"].isEnabled()
    assert not any(e.isEnabled() for e in t._fitBoundEntries["shell"])
    assert not any(e.isEnabled() for e in t._ffEntries)

    # potential -> parameter entries and pot0..N fit boxes
    t.potentialVar.set("SquareWell")
    t._onPotentialChanged()
    assert len(t.potParamVars) >= 2
    assert {"pot0", "pot1"} <= set(t.fitFlags)
    # link delta = c x dR: second argument greyed and no longer fittable, c becomes live
    t.linkDeltaVar.set(True)
    t._syncLinkDelta()
    assert not t._potEntries[1].isEnabled() and t._potEntries[0].isEnabled()
    assert "pot1" not in t.fitFlags and "pot0" in t.fitFlags
    assert t._fitChecks["linkC"].isEnabled()
    t.linkDeltaVar.set(False)
    t._syncLinkDelta()
    assert "pot1" in t.fitFlags and not t._fitChecks["linkC"].isEnabled()
    # charged Yukawa: its own three arguments and defaults
    t.potentialVar.set("PolydisperseHardCoreYukawa")
    t._onPotentialChanged()
    assert [v.get() for v in t.potParamVars] == ["200.0", "0.02804", "2.0"]

    # closure with a parameter: alpha entry, "solve alpha" possible, closureParam fittable
    t.closureVar.set("Rogers-Young")
    t._onClosureChanged()
    assert t.closureParamVar is not None and t.findAlphaCheck.isEnabled()
    assert "closureParam" in t.fitFlags
    t.findAlphaCheck.setChecked(True)
    assert t.findAlphaVar.get() and not t._alphaEntry.isEnabled()
    t.findAlphaCheck.setChecked(False)
    assert t._alphaEntry.isEnabled()
    # second closure parameter
    t.closureVar.set("Extended Rogers-Young")
    t._onClosureChanged()
    assert t.closureParam2Var is not None
    name, default, _ = ozLib.secondClosureParam("Extended Rogers-Young")
    assert float(t.closureParam2Var.get()) == float(default)
    # back to PY: everything closure-specific gone and "solve alpha" reset
    t.findAlphaVar.set(True)
    t.closureVar.set("Percus-Yevick")
    t._onClosureChanged()
    assert t.closureParamVar is None and "closureParam" not in t.fitFlags
    assert not t.findAlphaCheck.isEnabled() and t.findAlphaVar.get() is False

    # form factor
    t.ffVar.set("Core-shell")
    t._syncFormFactor()
    assert t._fitChecks["shell"].isEnabled() and t._fitChecks["rhoRatio"].isEnabled()
    assert all(e.isEnabled() for e in t._ffEntries)
    assert t.scaleVar is not None and t._fitEntries["scale"].isEnabled()
    t.ffVar.set("Sphere")
    t._syncFormFactor()
    assert not t._fitChecks["shell"].isEnabled() and not any(e.isEnabled() for e in t._ffEntries)

    # power-law background
    assert not t._fitChecks["porodD"].isEnabled()
    t.porodVar.set(True)
    t._syncPorod()
    assert t._fitChecks["porodD"].isEnabled() and t._fitChecks["porodA"].isEnabled()

    # fitter that ignores bounds greys them (only where they apply) and says so
    t.fitterVar.set("bumps lm")
    assert t.fitterNoteVar.get() == "this fitter ignores bounds"
    assert not any(e.isEnabled() for e in t._fitBoundEntries["meanRadius"])
    t.fitterVar.set("scipy least_squares")
    assert t.fitterNoteVar.get() == ""
    assert all(e.isEnabled() for e in t._fitBoundEntries["meanRadius"])
    assert not any(e.isEnabled() for e in t._fitBoundEntries["shell"])

    # Mann damping only for Picard
    mann = t._keyedEntries["mannAlpha"]
    assert not mann.isEnabled()
    t.solverVar.set("Picard / Mann")
    assert mann.isEnabled()
    t.solverVar.set("MDIIS")
    assert not mann.isEnabled()

    # SLD core / shell / ratio stay in step both ways
    t.rhoCoreVar.set("2.0")
    t.rhoShellVar.set("1.0")
    assert t.rhoRatioVar.get() == "0.5"
    t.rhoRatioVar.set("-1.5")
    assert t.rhoShellVar.get() == "-3"


def test_hover_help(qapp, tab):
    t = tab
    lab_or_entry = t._keyedEntries["meanRadius"]
    _hover(lab_or_entry, qapp)
    assert t.helpVar.get() == t.HELP["meanRadius"]
    assert t.helpLabel.text() == t.HELP["meanRadius"]
    _leave(lab_or_entry)
    assert t.helpVar.get() == ""
    _hover(t.findAlphaCheck, qapp)
    assert t.helpVar.get().startswith("Solve the closure's mixing parameter")
    # the box is sized for the longest text, so it never resizes under the cursor
    assert t.helpLabel.height() == t.helpLabel.maximumHeight() > 100


def test_settings_dialog(qapp, dialogs, tab):
    t = tab
    dlg = t._onSettings()
    assert dlg.isModal() and dlg.isVisible()
    assert dlg.kVar.get() == "12" and dlg.nLabelVar.get().startswith("N = 4095")
    dlg.kVar.set("10")
    assert dlg.nLabelVar.get().startswith("N = 1023")
    dlg.settingVars["transformType"].set("4")
    assert dlg.nLabelVar.get().startswith("N = 1024")
    dlg.settingVars["transformType"].set("2")
    assert dlg.apply() is False and dialogs.kinds()[-1] == "error"
    assert not hasattr(t, "_settings")
    dlg.settingVars["transformType"].set("1")
    dlg.settingVars["pointsPerSigma"].set("60")
    assert dlg.apply() is True
    assert t._settings == {"gridN": 1023, "pointsPerSigma": 60, "maxIterations": 8000, "transformType": 1}
    # reopen: current values shown; Defaults restores them
    dlg = t._onSettings()
    assert dlg.kVar.get() == "10" and dlg.settingVars["pointsPerSigma"].get() == "60"
    dlg.defaults()
    assert dlg.kVar.get() == "12" and dlg.settingVars["pointsPerSigma"].get() == "100"
    dlg.reject()
    assert t._settings["pointsPerSigma"] == 60          # Cancel keeps the old settings


# ---------------------------------------------------------------- compute
def test_compute_matches_library_and_exports(qapp, dialogs, wait, tmp_path, tab):
    from pysasfit.oz.generic_polydisperse_sas import GenericPolydisperseSAS
    from pysasfit.oz.gui.generic_polydisperse_tab import APPROX_SCHEMES
    from pysasfit.oz.polydisperse_yukawa_sas import CoreShellFixedShell

    t = tab
    _fast(t)
    t.ffVar.set("Core-shell")
    t._syncFormFactor()
    t.scaleVar.set("2.0")
    t.backgroundVar.set("0.5")
    t0 = time.time()
    _compute(t, wait)
    assert time.time() - t0 < 60
    assert t.statusVar.get().startswith("done: HardSphere, Percus-Yevick, Schulz")
    r = t.result
    assert r is t.runs[-1] and r.Q.size == 40 and np.all(np.isfinite(r.I_exact))
    assert r.S.shape == (40, 12, 12) and r.sigma.size == 12
    assert all(r.approx[lab] is not None for lab, _ in APPROX_SCHEMES)
    summary = t.summaryText.toPlainText()
    assert "sigma_eff" in summary and "form factor    : Core-shell" in summary
    # plots: exact + six schemes on I(Q), six on the error plot, bars on the distribution
    assert len(t.axes["iq"].lines) == 7 and len(t.axes["err"].lines) == 6
    assert len(t.axes["dist"].patches) == 12 and len(t.axes["sq"].lines) == 1
    assert t.interruptBtn.isEnabled() is False and t.computeBtn.isEnabled()

    # Same numbers as calling the library directly, the way the worker does
    sas = GenericPolydisperseSAS(
        "HardSphere", (), 0.2, 0.2, nbins=2, nFF=12, closure="Percus-Yevick", meanRadius=50.0,
        closureParam=None, closureParam2=None, distribution="Schulz",
        formfactor=CoreShellFixedShell(rho_core=2.0, rho_shell=1.0, thickness=2.7),
        solverClass=t.solverChoices["SUNDIALS KIN_FP"], deflateOnUnphysical=0, mannAlpha=1.0,
        **{k: FAST_SETTINGS[k] for k in ("gridN", "pointsPerSigma", "maxIterations", "transformType")})
    Q = np.logspace(-4, np.log10(0.3), 40)
    np.testing.assert_allclose(r.I_exact, 2.0*sas.I_exact(Q) + 0.5, rtol=1e-12)
    np.testing.assert_allclose(r.approx["3  partial structure factors"], 2.0*sas.I_partial_sf(Q) + 0.5, rtol=1e-12)
    np.testing.assert_allclose(r.S_number, sas.S_number(Q), rtol=1e-12)

    # ASCII export through the shared controls
    out = tmp_path / "gp.txt"
    dialogs.answers["save"] = str(out)
    t.exportFormatVar.set("ASCII")
    t._onExportSelected()
    lines = out.read_text().splitlines()
    assert lines[0] == "# Q\tI_exact\tS_number\tscheme0\tscheme1\tscheme2\tscheme3\tscheme4\tscheme5"
    assert len(lines) == 41
    np.testing.assert_allclose(np.loadtxt(out)[:, 1], r.I_exact, rtol=1e-9)

    # bad input -> error dialog, no worker started
    t.QmaxVar.set("abc")
    before = t.worker
    t._onCompute()
    assert dialogs.kinds()[-1] == "error" and t.worker is before


def test_compute_failure_is_reported(qapp, dialogs, wait, tab):
    t = tab
    _fast(t)
    t.potentialVar.set("SquareWell")
    t._onPotentialChanged()
    t.potParamVars[0].set("1.0")
    t.potParamVars[1].set("5.4")
    t.solverVar.set("Picard / Mann")
    t._settings["maxIterations"] = 200
    t.phiVar.set("0.15")
    _compute(t, wait)
    assert t.statusVar.get() == "failed (see Summary tab)"
    assert "did not converge" in t.summaryText.toPlainText()
    assert t.result is None and t.computeBtn.isEnabled()


# ---------------------------------------------------------------- data + fit
def test_load_data_compute_chi2_and_fit(qapp, dialogs, wait, tmp_path, tab):
    t = tab
    path = _load(t, dialogs, tmp_path)
    assert t.dataLabelVar.get().startswith("49 points, Q 0.018..0.3679, with dI, dQ/Q")
    assert t.smearCheck.isEnabled() and t.smearVar.get() and t.fitBtn.isEnabled()
    assert len(t.axes["iq"].containers) == 1                 # error bars drawn at once
    assert t.dataPath == str(path)

    _fast(t)
    t.meanRadiusVar.set("30")
    t.scaleVar.set("1e-3")
    _compute(t, wait)
    # smeared onto the data's own Q points, with a chi^2 against them
    assert t.result.smeared and t.result.Q.size == 49 and t.result.chi2 > 0
    assert any("chi^2_{red}" in tx.get_text() for tx in t.axes["iq"].get_legend().get_texts())

    # A short fit: three nonlinear parameters, scale and background solved exactly.
    t.FIT_MAX_NFEV = 6
    t._onFit()
    assert not t.fitBtn.isEnabled() and not t.computeBtn.isEnabled() and t.interruptBtn.isEnabled()
    wait(lambda: not t.worker.is_alive(), 120)
    wait(lambda: t.fitBtn.isEnabled(), 10)
    fr = t.fitResult
    assert not fr.get("provisional") and set(fr["parameters"]) == {"meanRadius", "srel", "phi"}
    assert t.statusVar.get().startswith(("fit done: chi2_red = ", "fit interrupted"))
    assert t.summaryText.toPlainText().startswith("FIT")
    # fitted values written back into their entries
    assert float(t.meanRadiusVar.get()) == pytest.approx(fr["parameters"]["meanRadius"], rel=1e-5)
    assert float(t.scaleVar.get()) == pytest.approx(fr["scale"], rel=1e-5)
    assert any("fit:" in tx.get_text() for tx in t.axes["iq"].get_legend().get_texts())
    assert t.computeBtn.isEnabled() and not t.interruptBtn.isEnabled()


def test_fit_interrupt_and_autosave(qapp, dialogs, wait, tmp_path, tab):
    t = tab
    _load(t, dialogs, tmp_path)
    _fast(t)
    t.meanRadiusVar.set("30")
    t.nbinsVar.set("3")
    t.FIT_MAX_NFEV = 400
    t._onFit()
    # let it report progress at least once (throttled to 1/s), then interrupt
    wait(lambda: t.statusVar.get().startswith("fitting...") or not t.worker.is_alive(), 60)
    t._onInterrupt()
    assert t._abortFit and t.statusVar.get().startswith("interrupt requested")
    wait(lambda: not t.worker.is_alive(), 60)
    wait(lambda: t.fitBtn.isEnabled(), 10)
    assert t.fitResult.get("aborted") or t.statusVar.get().startswith("fit done")
    if t.fitResult.get("aborted"):
        assert t.statusVar.get().startswith("fit interrupted; best chi2_red = ")
        assert t.summaryText.toPlainText().startswith("FIT (INTERRUPTED -- best point so far)")
    # the progress handler autosaved a normal session beside the data
    saved = json.loads((tmp_path / "restore.oz1").read_text())
    assert saved["tab"] == "Polydisperse (any potential)" and saved["session"]["data"]["Q"]


def test_fit_refuses_bad_bounds(qapp, dialogs, tmp_path, tab):
    t = tab
    _load(t, dialogs, tmp_path)
    t._fitBoundVars["phi"][1].set("abc")                  # crashed the Tk original
    t._onFit()
    assert dialogs.calls[-1][1]["title"] == "bad bound" and t.worker is None
    t._fitBoundVars["phi"][1].set("1.5")                  # wider than the physical domain
    t._onFit()
    assert dialogs.calls[-1][1]["title"] == "bounds outside the physical domain"
    t._fitBoundVars["phi"][1].set("0.1")                  # value outside its own bounds
    t._onFit()
    assert dialogs.calls[-1][1]["title"] == "value outside its bounds"
    t._fitBoundVars["phi"][1].set("1")
    for v in t.fitFlags.values():
        v.set(False)
    t._onFit()
    assert dialogs.calls[-1][1]["title"] == "nothing to fit"
    # holding background at zero asks first; declining starts nothing
    t.fitFlags["meanRadius"].set(True)
    t._onFit()
    assert dialogs.kinds()[-1] == "okcancel" and t.worker is None


# ---------------------------------------------------------------- sessions
def _configure(t, dialogs, tmp_path):
    _load(t, dialogs, tmp_path)
    t._settings = dict(FAST_SETTINGS)
    t.potentialVar.set("SquareWell")
    t._onPotentialChanged()
    t.potParamVars[0].set("0.3")
    t.potParamVars[1].set("4.0")
    t.closureVar.set("Rogers-Young")
    t._onClosureChanged()
    t.closureParamVar.set("0.7")
    t.ffVar.set("Core-shell")
    t._syncFormFactor()
    t.porodVar.set(True)
    t._syncPorod()
    t.porodDVar.set("0.4")
    t.distVar.set("Weibull")
    t.meanRadiusVar.set("31.5")
    t.rhoRatioVar.set("-0.25")
    t.fitFlags["pot0"].set(True)
    t.fitFlags["closureParam"].set(True)
    t.fitFlags["srel"].set(False)
    t._fitBoundVars["phi"][1].set("0.4")
    t.fitterVar.set("bumps amoeba")
    t.kernelVar.set("gaussian")
    t.smearVar.set(False)
    t.fitResult = {"Q": np.array([0.1, 0.2]), "fit": np.array([3.0, 2.0]), "chi2_reduced": 1.5,
                   "scale": 2.0, "background": -0.1, "parameters": {"phi": 0.2}}


def test_session_state_roundtrip(qapp, dialogs, tmp_path, tab):
    from pysasfit.oz.gui.generic_polydisperse_tab import GenericPolydisperseTab

    t = tab
    _configure(t, dialogs, tmp_path)
    state = json.loads(json.dumps(t.sessionState()))
    assert state["entries"]["potentialVar"] == "SquareWell" and state["potParamVars"][:2] == ["0.3", "4.0"]
    assert state["settings"] == FAST_SETTINGS and state["fitBounds"]["phi"] == ["1e-09", "0.4"]

    t2 = GenericPolydisperseTab()
    try:
        t2.restoreSessionState(state)
        assert json.loads(json.dumps(t2.sessionState())) == state
        assert t2.closureParamVar.get() == "0.7" and t2._alphaEntry is not None
        assert t2._fitChecks["shell"].isEnabled() and t2._fitChecks["porodD"].isEnabled()
        assert t2.fitBtn.isEnabled() and t2.smearCheck.isEnabled() and not t2.smearVar.get()
        assert t2.dataLabelVar.get().endswith("restored from session")
        assert t2.rhoShellVar.get() == "-0.5"
        t2._replot()                                       # restored fit result is drawable
        assert t2.fitResult["background"] == -0.1
        assert any("NEGATIVE" in tx.get_text() for tx in t2.axes["iq"].get_legend().get_texts())
    finally:
        t2.deleteLater()


def test_session_file_save_and_load(qapp, dialogs, wait, tmp_path, tab):
    from pysasfit.oz.gui.generic_polydisperse_tab import GenericPolydisperseTab

    t = tab
    dialogs.answers["save"] = str(tmp_path / "s.oz1")
    t._onSaveAll()                                         # nothing computed yet
    assert dialogs.kinds()[-1] == "info" and not (tmp_path / "s.oz1").exists()

    _configure(t, dialogs, tmp_path)
    t.potentialVar.set("HardSphere")
    t._onPotentialChanged()
    t.closureVar.set("Percus-Yevick")
    t._onClosureChanged()
    t.porodVar.set(False)
    t._syncPorod()
    _fast(t)
    _compute(t, wait)
    assert t.statusVar.get().startswith("done")
    t._onSaveAll()
    payload = json.loads((tmp_path / "s.oz1").read_text())
    assert payload["tab"] == "Polydisperse (any potential)" and payload["tabIndex"] == 1
    assert payload["runs"] == [] and payload["session"]["entries"]["distVar"] == "Weibull"
    assert dialogs.calls[-1][1]["defaultextension"] == ".oz1"

    t2 = GenericPolydisperseTab()
    try:
        dialogs.answers["open"] = str(tmp_path / "s.oz1")
        t2._onLoadAll()
        assert t2.statusVar.get() == "loaded 0 run(s), session restored"
        assert t2.sessionState() == t.sessionState()
        assert t2._settings == FAST_SETTINGS
        # a file from another tab is refused
        other = dict(payload, tab="RY Polydisperse Yukawa")
        (tmp_path / "x.oz2").write_text(json.dumps(other))
        dialogs.answers["open"] = str(tmp_path / "x.oz2")
        t2._onLoadAll()
        assert dialogs.calls[-1][0] == "error" and dialogs.calls[-1][1]["title"] == "wrong tab"
    finally:
        t2.deleteLater()


def test_main_window_registers_tab(qapp):
    from pysasfit.oz.gui import main_window

    w = main_window.OZWidget(extra_tabs=[e for e in main_window.EXTRA_TABS
                                         if e[0] == "generic_polydisperse_tab"], redirect_output=False)
    try:
        assert w.tabText(1) == "1: Polydisperse (any potential)"
        assert type(w.widget(1)).__name__ == "GenericPolydisperseTab"
    finally:
        w.shutdown()
        w.deleteLater()


def test_session_keeps_solver_and_deflate(qapp, dialogs, wait, tmp_path, tab):
    """The OZ solver and the deflation count are part of the calculation and
    must survive a save/load (pyOZgui's Tk tab dropped both)."""
    from pysasfit.oz.gui.generic_polydisperse_tab import GenericPolydisperseTab

    t = tab
    if getattr(t, "solverVar", None) is None or len(t.solverChoices) < 2:
        pytest.skip("needs at least two OZ solvers")
    other = list(t.solverChoices)[1]
    t.solverVar.set(other)
    t.deflateVar.set("2")
    state = json.loads(json.dumps(t.sessionState()))
    assert state["entries"]["solverVar"] == other and state["entries"]["deflateVar"] == "2"

    t2 = GenericPolydisperseTab()
    try:
        t2.restoreSessionState(state)
        assert t2.solverVar.get() == other and t2.deflateVar.get() == "2"
        assert t2._restoreNote == ""
        assert t2._readInputs()["deflate"] == 2 if hasattr(t2, "_readInputs") else True

        # a solver this installation does not offer falls back, with a note
        bad = json.loads(json.dumps(state))
        bad["entries"]["solverVar"] = "no such solver"
        bad["entries"]["deflateVar"] = "7"
        t2.restoreSessionState(bad)
        first = next(iter(t2.solverChoices))
        assert t2.solverVar.get() == first and t2.deflateVar.get() == "0"
        assert "no such solver" in t2._restoreNote

        # ... and the note reaches the status line after loading a file
        (tmp_path / "bad.oz1").write_text(json.dumps(
            {"format": "sasfit_polydisperse_tab_save_v2", "tab": "Polydisperse (any potential)",
             "tabIndex": 1, "runs": [], "session": bad}))
        dialogs.answers["open"] = str(tmp_path / "bad.oz1")
        t2._onLoadAll()
        assert "not available here" in t2.statusVar.get()

        # older files without the keys keep the current choice
        old = json.loads(json.dumps(state))
        del old["entries"]["solverVar"], old["entries"]["deflateVar"]
        t2.solverVar.set(other)
        t2.deflateVar.set("1")
        t2.restoreSessionState(old)
        assert t2.solverVar.get() == other and t2.deflateVar.get() == "1"
    finally:
        t2.deleteLater()


def test_pyozgui_session_sign_is_converted(qapp, dialogs, tmp_path, tab):
    """A session without 'signConvention' was written with pyOZgui's square-well
    sign (positive = repulsive); it is converted to convention A on load."""
    from pysasfit.oz.gui.generic_polydisperse_tab import GenericPolydisperseTab

    t = tab
    t.potentialVar.set("SquareWell")
    t._onPotentialChanged()
    t.potParamVars[0].set("0.3")
    state = json.loads(json.dumps(t.sessionState()))
    assert state["signConvention"] == "A"

    old = json.loads(json.dumps(state))
    del old["signConvention"]
    old["potParamVars"][0] = "-0.3"                          # pyOZgui: a well was negative
    t2 = GenericPolydisperseTab()
    try:
        t2.restoreSessionState(old)
        assert float(t2.potParamVars[0].get()) == 0.3
        assert "convention A" in t2._restoreNote
        assert "signConvention" not in old                  # the caller's dict is untouched
        t2.restoreSessionState(state)                       # a new file is not converted again
        assert float(t2.potParamVars[0].get()) == 0.3 and t2._restoreNote == ""
    finally:
        t2.deleteLater()
