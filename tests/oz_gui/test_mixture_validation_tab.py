"""Offscreen tests of the Qt mixture validation tab (port of pyOZgui's Tk tab)."""

import json

import numpy as np
import pytest
from PySide6 import QtCore, QtWidgets

# Small, fast settings: 3 classes, N = 1023 / 2047, 40 Q points.
QUICK = dict(nbinsVar="3", gridKVar="10", ppsVar="40", refineVar="2", nQVar="40", phiVar="0.30")


def _tab(qapp):
    from pysasfit.oz.gui.mixture_validation_tab import IMPORT_ERROR, MixtureValidationTab

    assert IMPORT_ERROR is None
    t = MixtureValidationTab()
    t.resize(1200, 800)
    t.show()
    qapp.processEvents()
    return t


def _set(t, **values):
    for name, value in values.items():
        getattr(t, name).set(value)


def _run(t, wait):
    t._onCompare()
    assert not t.computeBtn.isEnabled()
    wait(lambda: not t.worker.is_alive() and t.computeBtn.isEnabled(), 60)
    assert t.statusVar.get() == "done", t.summary.toPlainText()


def _plotted(ax):
    return {ln.get_label(): ln.get_ydata() for ln in ax.get_lines()}


def _reference_curves(t, p, Q):
    """The analytic references and S_M(Q) computed directly from the libraries, as the Tk code does."""
    import mixscatter as ms

    from pysasfit.oz import gazzillo_msa as GZ
    from pysasfit.oz.polydisperse_nodes import sizeClasses

    sigma, x = sizeClasses("Schulz", p["srel"], p["nbins"], 2.0*p["meanRadius"])
    rhoT = p["phi"]/((np.pi/6.0)*float(np.sum(x*sigma**3)))
    gz = GZ.measurableStructureFactor(Q, sigma, np.zeros_like(sigma), rhoT*x, None, 0.0)
    mix = ms.Mixture(radius=sigma/2.0, number_fraction=x)
    S = np.asarray(ms.PercusYevick(Q, mix, volume_fraction_total=p["phi"]).number_weighted_partial_structure_factor)
    SAL = S/np.sqrt(np.outer(x, x))[:, :, None]
    F = t._sphereF(Q, sigma)
    w = F/np.sqrt(np.sum(x[None, :]*F**2, axis=1))[:, None]
    sx = np.sqrt(x)
    vrij = np.einsum("a,b,qa,qb,abq->q", sx, sx, w, w, SAL)
    return sigma, x, gz, vrij


def test_construct_defaults_and_controls(qapp):
    t = _tab(qapp)
    assert t.SESSION_TAB_INDEX == 3 and t.SESSION_TAB_NAME == "Mixture validation"
    assert t.meanRadiusVar.get() == "50.0" and t.phiVar.get() == "0.20" and t.nQVar.get() == "200"
    assert t.closureVar.get() == "Percus-Yevick"
    assert t.refineVar.get() == "4" and [t._refineBox.itemText(i) for i in range(4)] == ["2", "4", "8", "16"]
    assert t._nLabelVar.get() == "N = 4095 / 16383"
    t.refineVar.set("8")
    assert t._nLabelVar.get() == "N = 4095 / 32767"
    t.gridKVar.set("abc")
    assert t._nLabelVar.get() == "?"
    t.gridKVar.set("inf")          # used to raise OverflowError in the trace callback
    assert t._nLabelVar.get() == "?"
    t.gridKVar.set("12")
    # charge entries greyed
    for lab in ("Macroion z:", "Bjerrum length:", "Microion diam.:"):
        assert not t._entryWidgets[lab].isEnabled()
    # damping only for Picard / Mann; default solver is the most robust one
    assert t.solverVar.get() == next(iter(t.solverChoices))
    assert "Picard" not in t.solverVar.get()
    assert not t._entryWidgets["   damping a:"].isEnabled()
    t.solverVar.set("Picard / Mann")
    assert t._entryWidgets["   damping a:"].isEnabled()
    # hover help box: starts on "refine", follows the hovered widget and stays (no leave)
    assert t.helpText.toPlainText() == t.HELP["refine"]
    e = t._entryWidgets["Deflate if S(Q)<0:"]
    QtWidgets.QApplication.sendEvent(e, QtCore.QEvent(QtCore.QEvent.Type.Enter))
    assert t.helpText.toPlainText() == t.HELP["deflate"]
    QtWidgets.QApplication.sendEvent(e, QtCore.QEvent(QtCore.QEvent.Type.Leave))
    assert t.helpText.toPlainText() == t.HELP["deflate"]
    QtWidgets.QApplication.sendEvent(t._solverBox, QtCore.QEvent(QtCore.QEvent.Type.Enter))
    assert t.helpVar.get() == t.HELP["solver"]
    # forced neutral even if z is set from code / an old file
    t.zVar.set("12")
    assert t._readInputs()["z"] == 0.0


def test_compare_mixscatter_and_gazzillo_against_library(qapp, wait):
    pytest.importorskip("mixscatter")
    from pysasfit.oz.mixscatter_bridge import OZLiquidStructure

    t = _tab(qapp)
    _set(t, **QUICK)
    p = t._readInputs()
    _run(t, wait)

    c = t.curves
    assert set(c["lines"]) == {"Gazzillo MSA (neutral = PY)", "mixscatter (Vrij PY)",
                               "numerical (coarse)", "numerical (fine)"}
    Q = np.logspace(-3, np.log10(0.3), 40)
    assert np.allclose(c["Q"], Q)
    sigma, x, gz, vrij = _reference_curves(t, p, Q)
    assert np.allclose(c["sigma"], sigma) and np.allclose(c["x"], x)
    assert np.allclose(c["lines"]["Gazzillo MSA (neutral = PY)"], gz, rtol=1e-12)
    assert np.allclose(c["lines"]["mixscatter (Vrij PY)"], vrij, rtol=1e-12)
    # the two analytic PY references agree to machine precision
    assert np.max(np.abs(gz/vrij - 1)) < 1e-9

    # numerical coarse solve, rebuilt directly through the bridge
    class _Mix:
        pass
    m = _Mix()
    m.radius, m.number_fraction = sigma/2.0, x
    oz = OZLiquidStructure(Q, m, volume_fraction_total=0.30, closure="Percus-Yevick", pointsPerSigma=40,
                           gridN=1023, solverClass=p["solverClass"], mannAlpha=1.0, maxIterations=60000,
                           deflateOnUnphysical=0)
    SAL = oz.number_weighted_partial_structure_factor/np.sqrt(np.outer(x, x))[:, :, None]
    F = t._sphereF(Q, sigma)
    w = F/np.sqrt(np.sum(x[None, :]*F**2, axis=1))[:, None]
    coarse = np.einsum("a,b,qa,qb,abq->q", np.sqrt(x), np.sqrt(x), w, w, SAL)
    assert np.allclose(c["lines"]["numerical (coarse)"], coarse, rtol=1e-6, atol=1e-8)

    # numerical close to the reference, and the fine grid closer than the coarse one
    eCo = np.max(np.abs(c["lines"]["numerical (coarse)"] - vrij))/np.max(np.abs(vrij))
    eFi = np.max(np.abs(c["lines"]["numerical (fine)"] - vrij))/np.max(np.abs(vrij))
    assert eFi < eCo < 0.05

    s = t.summary.toPlainText()
    assert s.startswith("3 size classes")
    assert "reference: mixscatter (Vrij PY)" in s
    assert f"numerical (coarse)     {eCo:.5f}" in s and f"numerical (fine)       {eFi:.5f}" in s
    assert "solver residual max|G(gamma)-gamma|:" in s
    assert f"coarse/fine ratio: {eCo/eFi:.2f}" in s
    assert "at 2x refinement: expect 2 if first order, 4 if second" in s

    # upper panel: all four curves; error panel: everything vs the first analytic curve
    up = _plotted(t.ax)
    assert set(up) == set(c["lines"])
    assert np.allclose(up["numerical (fine)"], c["lines"]["numerical (fine)"])
    err = _plotted(t.axErr)
    assert set(err) == {f"{n} vs Gazzillo MSA (neutral = PY)" for n in
                        ("mixscatter (Vrij PY)", "numerical (coarse)", "numerical (fine)")}
    assert np.allclose(err["numerical (coarse) vs Gazzillo MSA (neutral = PY)"],
                       np.abs(c["lines"]["numerical (coarse)"]/gz - 1))
    assert t.ax.get_xscale() == "log" and t.axErr.get_yscale() == "log"

    t._onClear()
    assert t.curves == {} and t.summary.toPlainText() == "" and not t.ax.get_lines()


def test_compare_gazzillo_only_without_mixscatter(qapp, wait, monkeypatch):
    """Without mixscatter the Gazzillo neutral (= PY) curve is the reference."""
    import pysasfit.oz.gui.mixture_validation_tab as mod

    monkeypatch.setattr(mod, "_ms", None)
    t = _tab(qapp)
    assert t.statusVar.get() == "ready  (mixscatter not installed)"
    # also exercise the damped Picard / Mann solver at a low density it handles
    _set(t, **dict(QUICK, phiVar="0.20"))
    t.solverVar.set("Picard / Mann")
    t.mannAlphaVar.set("0.8")
    p = t._readInputs()
    assert p["solverClass"] is t.solverChoices["Picard / Mann"] and p["mannAlpha"] == 0.8
    _run(t, wait)
    lines = t.curves["lines"]
    assert set(lines) == {"Gazzillo MSA (neutral = PY)", "numerical (coarse)", "numerical (fine)"}
    s = t.summary.toPlainText()
    assert "reference: Gazzillo MSA (neutral = PY)" in s
    assert "coarse/fine ratio" in s
    assert set(_plotted(t.ax)) == set(lines)
    assert len(t.axErr.get_lines()) == 2


def test_charged_gazzillo_branch(qapp):
    """The charged path is unreachable from the GUI (z is forced to 0) but its code is kept: run the worker on it."""
    t = _tab(qapp)
    _set(t, **QUICK)
    p = dict(t._readInputs(), z=10.0, phi=0.05)
    t._worker(p)
    kind, out = t.resultQueue.get(timeout=30)
    assert kind == "done"
    assert not any(k.startswith(("numerical", "mixscatter")) for k in out["lines"])
    assert any(n.startswith("numerical curve omitted") for n in out["notes"])
    if "Gazzillo MSA (charged)" in out["lines"]:
        S = out["lines"]["Gazzillo MSA (charged)"]
        assert np.all(np.isfinite(S)) and S.min() > -1e-6
    else:
        assert any(n.startswith("CHARGED SOLVE FAILED") for n in out["notes"])
    t._handleQueueItem((kind, out))
    assert t.statusVar.get() == "done"


def test_save_load_roundtrip(qapp, dialogs, wait, tmp_path):
    pytest.importorskip("mixscatter")
    t = _tab(qapp)
    _set(t, **QUICK)
    t.solverVar.set("scipy Anderson")
    _run(t, wait)
    saved_lines = {k: v.copy() for k, v in t.curves["lines"].items()}
    saved_summary = t.summary.toPlainText()

    path = tmp_path / "mix.oz3"
    dialogs.answers["save"] = str(path)
    t._onSave()
    assert dialogs.calls[-1][0] == "save" and dialogs.calls[-1][1]["defaultextension"] == ".oz3"
    payload = json.loads(path.read_text())
    assert payload["format"] == "sasfit_mixture_validation_save_v1"
    assert payload["tab"] == "Mixture validation" and payload["tabIndex"] == 3
    assert payload["entries"]["nbinsVar"] == "3" and payload["entries"]["gridKVar"] == "10"
    assert payload["entries"]["solverVar"] == "scipy Anderson" and payload["entries"]["refineVar"] == "2"
    assert t.statusVar.get() == f"saved to {path}"

    # fresh tab: load restores settings, curves, plot and summary
    t2 = _tab(qapp)
    t2.summary.setPlainText("stale summary from an earlier run")
    dialogs.answers["open"] = str(path)
    t2._onLoad()
    assert t2.statusVar.get() == f"loaded {path}"
    for name, value in QUICK.items():
        assert getattr(t2, name).get() == value
    assert t2.solverVar.get() == "scipy Anderson" and t2._solverBox.currentText() == "scipy Anderson"
    assert t2._nLabelVar.get() == "N = 1023 / 2047"
    for k, v in saved_lines.items():
        assert np.allclose(t2.curves["lines"][k], v)
    assert set(_plotted(t2.ax)) == set(saved_lines)
    assert t2.summary.toPlainText() == saved_summary
    assert "error" not in dialogs.kinds()

    # unavailable solver in the file -> reset to the default and reported
    payload["entries"]["solverVar"] = "no such solver"
    bad = tmp_path / "bad_solver.oz3"
    bad.write_text(json.dumps(payload))
    dialogs.answers["open"] = str(bad)
    t2._onLoad()
    assert t2.solverVar.get() == next(iter(t2.solverChoices))
    assert t2._solverBox.currentText() == t2.solverVar.get()
    assert "error" not in dialogs.kinds()

    # wrong tab / wrong format / unreadable -> error dialogs, state untouched
    for name, content in (("other.oz1", dict(payload, tab="Polydisperse")),
                          ("fmt.oz3", {"format": "something else"}),
                          ("junk.oz3", None)):
        f = tmp_path / name
        f.write_text("{not json" if content is None else json.dumps(content))
        dialogs.answers["open"] = str(f)
        n = len(dialogs.calls)
        t2._onLoad()
        assert dialogs.kinds()[n:] == ["open", "error"]
    assert t2.nbinsVar.get() == "3"

    # cancelled dialogs do nothing
    dialogs.answers.clear()
    t2._onSave()
    t2._onLoad()
    assert dialogs.kinds()[-2:] == ["save", "open"]


def test_invalid_input_shows_error(qapp, dialogs):
    t = _tab(qapp)
    for name, value, text in (("QmaxVar", "abc", "Q max: 'abc' is not a number"),
                              ("phiVar", "0.7", "Volume fraction must be <= 0.6"),
                              ("QminVar", "0.5", "Q max must exceed Q min"),
                              ("gridKVar", "4", "Grid exponent k must be >= 8")):
        t2 = _tab(qapp) if name != "QmaxVar" else t
        getattr(t2, name).set(value)
        n = len(dialogs.calls)
        t2._onCompare()
        assert dialogs.kinds()[n:] == ["error"]
        assert dialogs.calls[-1][1]["title"] == "input error"
        assert dialogs.calls[-1][1]["message"] == text
        assert t2.worker is None and t2.computeBtn.isEnabled()


def test_busy_while_running(qapp, dialogs, wait):
    t = _tab(qapp)
    _set(t, **QUICK)
    t._onCompare()
    if t.worker.is_alive():
        t._onCompare()
        assert dialogs.kinds()[-1] == "info"
    wait(lambda: not t.worker.is_alive() and t.computeBtn.isEnabled(), 60)
