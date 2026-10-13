"""Deflation in the polydisperse model, the RY solver fallback, the transform
type of ozLib.solve, and the shared numerics of the preferences."""
import numpy as np
import pytest

from pysasfit.oz import deflation, ozLib


LJ = dict(potential="LennardJones", potentialArgs=(0.8,), phi=0.3, srel=0.05, nbins=3,
          closure="Hypernetted-Chain", gridN=1023, pointsPerSigma=50, maxIterations=300)


def _nk():
    return ozLib.SOLVER_CLASSES["scipy Newton-Krylov"][0]


def test_polydisperse_deflation_actually_runs():
    """Newton-Krylov lands on an unphysical branch (min S ~ -18) at this
    Lennard-Jones state. Before the fix, deflation discarded its own result
    and returned the unphysical root SILENTLY; now every attempt is a full
    solve and an unphysical outcome is still reported."""
    from pysasfit.oz.generic_polydisperse_sas import GenericPolydisperseSAS

    with pytest.raises(RuntimeError, match="UNPHYSICAL") as e0:
        GenericPolydisperseSAS(**LJ, solverClass=_nk(), deflateOnUnphysical=0)
    assert "Deflation tried" not in str(e0.value)
    old = dict(deflation.DEFAULTS)
    try:
        deflation.DEFAULTS.update(alpha=0.1)        # shift at which a second root is found
        with pytest.raises(RuntimeError, match="UNPHYSICAL") as e1:
            GenericPolydisperseSAS(**LJ, solverClass=_nk(), deflateOnUnphysical=2)
    finally:
        deflation.DEFAULTS.update(old)
    msg = str(e1.value)
    assert "Deflation tried: initial: min S = -18.31, 1: min S = -20.24" in msg   # a DISTINCT root


def test_polydisperse_fixed_point_solver_needs_no_deflation():
    from pysasfit.oz.generic_polydisperse_sas import GenericPolydisperseSAS

    cls = ozLib.SOLVER_CLASSES["MDIIS"][0]
    s = GenericPolydisperseSAS(**LJ, solverClass=cls, deflateOnUnphysical=2)
    assert float(np.min(np.real(s.solver.getSq()))) > 0.2
    assert not hasattr(s, "deflationHistory")


def test_ry_auto_solver_fallback(monkeypatch):
    """solverClass=None ('auto') falls back to the next solver when one fails;
    the closure and deflation settings are kept."""
    import pysasfit.oz.rypolydisperseWrapper as rypw

    real = rypw._solverClasses()
    good = dict(real)["Picard / Mann"] if "Picard / Mann" in dict(real) else real[-1][1]

    class Failing(good):
        def solve(self):
            raise RuntimeError("boom")

    monkeypatch.setattr(rypw, "_solverClasses", lambda: [("failing", Failing), ("good", good)])
    sig = np.array([0.9, 1.0, 1.1])
    rho = np.array([0.05, 0.1, 0.05])
    ry = rypw.RYPolydisperseYukawa(sig, rho, 2.0, -0.5, alpha=0.5, solverClass=None,
                                   gridN=1023, pointsPerSigma=50, closure="Rogers-Young")
    assert ry.solverFallback == ["Failing"] and ry.solverName == good.__name__ and ry.autoSolver
    assert ry.closure == "Rogers-Young"
    with pytest.raises(RuntimeError):                # an explicit choice is never replaced
        rypw.RYPolydisperseYukawa(sig, rho, 2.0, -0.5, alpha=0.5, solverClass=Failing,
                                  gridN=1023, pointsPerSigma=50)


def test_ozlib_transform_type_choice():
    kw = dict(phi=0.3, closure="Percus-Yevick", verify=False, numberOfRadialSamplingPoints=2047,
              hardSphereDiameterInPoints=50)
    auto = ozLib.solve("HardSphere", **kw)
    t1 = ozLib.solve("HardSphere", transformType=1, **kw)
    t4 = ozLib.solve("HardSphere", transformType=4, **kw)
    assert auto.solverInstance.resolvedTransformType() == 4 == t4.solverInstance.resolvedTransformType()
    assert t1.solverInstance.resolvedTransformType() == 1
    np.testing.assert_allclose(auto.Sq, t4.Sq)
    # the two transforms sample q differently; compare S(q) on a common grid
    q = np.linspace(1.0, 15.0, 50)
    s1 = np.interp(q, t1.q, np.real(t1.Sq))
    s4 = np.interp(q, t4.q, np.real(t4.Sq))
    assert 1e-6 < np.max(np.abs(s1 - s4)) < 0.05


def test_shared_numerics_reach_the_library(tmp_path, monkeypatch):
    from pysasfit.core import settings as prefs
    from pysasfit.oz import polydisperse_rmsa, polydisperse_yukawa_msa
    from pysasfit.oz.oZsolver import OZsolver
    from pysasfit.oz.preferences import apply_solver_settings

    monkeypatch.setenv("PYSASFIT_SETTINGS", str(tmp_path / "s.json"))
    prefs.load()
    f = {s.key: s.default for sec in prefs.sections() for s in sec.settings}
    # factory values = the library's own
    assert deflation.DEFAULTS == dict(p=f["numerics.deflation.p"], alpha=f["numerics.deflation.shift"],
                                      cap=f["numerics.deflation.cap"])
    assert polydisperse_yukawa_msa.MSA_DEFAULTS == prefs.values("numerics.msa")
    assert polydisperse_rmsa.RMSA_DEFAULTS == prefs.values("numerics.rmsa")
    assert OZsolver.CONSISTENCY_SCAN_STEPS == f["oz.consistency.scan_steps"]
    try:
        prefs.set_value("numerics.deflation.shift", 0.1)
        prefs.set_value("numerics.msa.gamma_per_decade", 100)
        prefs.set_value("oz.consistency.tolerance", 1e-7)
        apply_solver_settings()
        assert deflation.DEFAULTS["alpha"] == 0.1
        assert polydisperse_yukawa_msa.MSA_DEFAULTS["gamma_per_decade"] == 100
        assert OZsolver.CONSISTENCY_TOLERANCE == 1e-7
    finally:
        prefs.reset()
        apply_solver_settings()
    assert deflation.DEFAULTS["alpha"] == 1.0 and OZsolver.CONSISTENCY_TOLERANCE == 1e-5


# ----------------------------------------------------------------------
# Picard / Mann iteration
def test_picard_mann_name_and_alias():
    from pysasfit.oz import ozLib
    assert "Picard / Mann iteration" in ozLib.SOLVER_CLASSES
    assert "Picard iteration" not in ozLib.SOLVER_CLASSES      # no duplicate in the dropdowns
    assert ozLib.resolveSolverName("Picard iteration") == "Picard / Mann iteration"
    assert ozLib.isMannSolver("Picard iteration") and not ozLib.isMannSolver("Biggs-Andrews")
    r = ozLib.solve("HardSphere", phi=0.3, solver="Picard iteration", verify=False,
                    numberOfRadialSamplingPoints=1023)
    assert r.solverName == "Picard / Mann iteration" and r.mannAlpha == 1.0


def test_mann_damping_converges_where_picard_does_not():
    import numpy as np
    from pysasfit.oz import ozLib
    kw = dict(phi=0.45, verify=False, numberOfRadialSamplingPoints=1023)
    ref = ozLib.solve("HardSphere", solver="scipy Anderson", **kw)
    plain = ozLib.solve("HardSphere", solver="Picard / Mann iteration", maxIterations=3000, **kw)
    assert not plain.solverInstance.converged
    damped = ozLib.solve("HardSphere", solver="Picard / Mann iteration", mannAlpha=0.5,
                         maxIterations=20000, **kw)
    assert damped.solverInstance.converged and damped.mannAlpha == 0.5
    assert abs(np.nanmax(damped.gr) - np.nanmax(ref.gr)) < 1e-6


def test_mann_damping_same_solution_and_residual_test():
    """a < 1 reaches the same root, and convergence is judged on the residual
    ||T(x)-x||, not on the damped step (which would stop 1/a too early)."""
    import numpy as np
    from pysasfit.oz import ozLib
    kw = dict(phi=0.3, verify=False, numberOfRadialSamplingPoints=1023, solver="Picard / Mann iteration")
    a1 = ozLib.solve("HardSphere", **kw)
    a3 = ozLib.solve("HardSphere", mannAlpha=0.3, **kw)
    assert np.max(np.abs(a1.Sq - a3.Sq)) < 1e-8
    s = a3.solverInstance
    assert s.converged


def test_mann_alpha_validated_and_ignored_by_other_solvers():
    import pytest
    from pysasfit.oz import ozLib
    with pytest.raises(ValueError):
        ozLib.solve("HardSphere", phi=0.3, solver="Picard / Mann iteration", mannAlpha=0.0, verify=False)
    r = ozLib.solve("HardSphere", phi=0.3, solver="Biggs-Andrews", mannAlpha=0.5, verify=False,
                    numberOfRadialSamplingPoints=1023)
    assert r.mannAlpha is None and not hasattr(r.solverInstance, "mannAlpha")


def test_oz_tab_damping_field(tmp_path, monkeypatch):
    import os
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6 import QtWidgets
    QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    from pysasfit.oz.gui.oz_solver_tab import OZSolverTab
    tab = OZSolverTab(redirect_output=False)
    tab.solverVar.set("Biggs-Andrews")
    assert not tab.mannAlphaEntry.isEnabled()
    tab.solverVar.set("Picard / Mann iteration")
    assert tab.mannAlphaEntry.isEnabled()
    assert float(tab.mannAlphaVar.get()) == 1.0
    assert not hasattr(tab, "maxIterVar")            # limit comes from the preferences
    from pysasfit.oz.picardOZsolver import PicardOZsolver
    n = PicardOZsolver.DEFAULT_MAX_ITERATIONS
    assert f"iteration limit {n} " in tab.iterLimitVar.get()
    tab.mannAlphaVar.set("0.2")
    assert f"iteration limit {5 * n} = {n}/a" in tab.iterLimitVar.get()
    for bad in ("0", "-0.1", "1.5", "x"):
        tab.mannAlphaVar.set(bad)
        assert "0 < a ≤ 1" in tab.iterLimitVar.get()
        assert "ffd0d0" in tab.mannAlphaEntry.styleSheet()


def test_mann_unconverged_is_flagged_not_a_solution():
    """HS/PY at phi = 0.53: an iteration stopped at the limit is flagged
    (converged False, RuntimeWarning); converged runs agree for every a."""
    import numpy as np
    import pytest
    from pysasfit.oz import ozLib
    kw = dict(phi=0.53, verify=False, numberOfRadialSamplingPoints=1023)
    ref = ozLib.solve("HardSphere", solver="scipy Anderson", **kw)
    assert ref.converged
    with pytest.warns(RuntimeWarning, match="did not converge"):
        r = ozLib.solve("HardSphere", solver="Picard / Mann iteration", mannAlpha=0.5, maxIterations=1000, **kw)
    assert not r.converged
    for a in (0.6, 0.3):
        r = ozLib.solve("HardSphere", solver="Picard / Mann iteration", mannAlpha=a, maxIterations=30000, **kw)
        assert r.converged
        assert np.max(np.abs(r.Sq - ref.Sq)) < 1e-6
    with pytest.warns(RuntimeWarning):
        r = ozLib.solve("HardSphere", solver="Picard / Mann iteration", mannAlpha=0.8, maxIterations=5000, **kw)
    assert not r.converged          # oscillates: a too large at this density


def test_special_closures_report_convergence():
    from pysasfit.oz import ozLib
    for cl, kw in (("Rescaled MSA", {}), ("ZSEP", {}), ("Rogers-Young", dict(findConsistentParameter=True))):
        r = ozLib.solve("HardSphere", phi=0.3, closure=cl, solver="Picard / Mann iteration", verify=False,
                        numberOfRadialSamplingPoints=1023, **kw)
        assert r.converged, cl


def test_oz_tab_warns_when_not_converged(monkeypatch):
    import os
    import numpy as np
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6 import QtWidgets
    QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    from pysasfit.oz.gui import oz_solver_tab as m
    shown = []
    monkeypatch.setattr(m, "showwarning", lambda t, msg, **k: shown.append(msg))
    tab = m.OZSolverTab(redirect_output=False)
    r = np.linspace(0.01, 5, 50)
    run = m.RunResult("1: x [NOT CONVERGED]", r, r, "HardSphere", "Percus-Yevick", 0.53,
                      solver="Picard / Mann iteration", mannAlpha=0.5, converged=False)
    run.curves = {"gr": r, "Sq": r}
    tab._onCalculateDone(run)
    assert shown and "last iterate" in shown[0] and "damping a" in shown[0]


def test_mann_iteration_limit_scales_with_one_over_a():
    import pytest
    from pysasfit.oz import ozLib
    from pysasfit.oz.picardOZsolver import PicardOZsolver as P
    assert P.mannIterationLimit(3000, 1.0) == 3000
    assert P.mannIterationLimit(3000, 0.2) == 15000
    assert P.mannIterationLimit(1000, 0.3) == 3334
    for bad in (0.0, -0.5, 1.01):
        with pytest.raises(ValueError):
            P.mannIterationLimit(3000, bad)
    kw = dict(phi=0.3, verify=False, numberOfRadialSamplingPoints=1023, solver="Picard / Mann iteration")
    r = ozLib.solve("HardSphere", mannAlpha=0.25, maxIterations=400, **kw)
    assert r.iterationLimit == 1600 and r.converged and r.iterations > 400   # needs ~1400
    r = ozLib.solve("HardSphere", mannAlpha=0.25, **kw)
    assert r.iterationLimit == 4 * P.DEFAULT_MAX_ITERATIONS


def test_verifier_must_converge():
    """HS/PY phi = 0.5: Biggs-Andrews does not converge; the damped Picard /
    Mann verifier (a = 0.5, limit 5000/a) does and verifies KIN_FP."""
    from pysasfit.oz import ozLib
    r = ozLib.solve("HardSphere", phi=0.5, numberOfRadialSamplingPoints=1023)
    assert r.converged and r.verifiedWith == "Picard / Mann iteration"
    assert r.verificationDifference < 1e-6


def test_oz_tab_calculate_end_to_end(monkeypatch):
    """Press 'calculate' for real (worker thread + result queue), for a solver
    that converges and one that stops at its limit. Guards against errors in
    the worker such as the UnboundLocalError on `label`."""
    import os
    import time
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6 import QtWidgets
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    from pysasfit.oz.gui import oz_solver_tab as m
    errors, warns = [], []
    monkeypatch.setattr(m, "showerror", lambda t, msg="", **k: errors.append(msg))
    monkeypatch.setattr(m, "showwarning", lambda t, msg="", **k: warns.append(msg))
    tab = m.OZSolverTab(redirect_output=False)
    tab.gridPointsVar.set("10")
    tab.verifyVar.set(False)

    def run(solver, phi):
        tab.solverVar.set(solver)
        tab.phiVar.set(str(phi))
        n = len(tab.runs)
        tab._onCalculate()
        t0 = time.time()
        while len(tab.runs) == n and not errors and time.time() - t0 < 120:
            while not tab.resultQueue.empty():
                tab._handleQueueItem(tab.resultQueue.get())
            app.processEvents()
            time.sleep(0.02)
        assert not errors, errors
        return tab.runs[-1]

    r = run("Biggs-Andrews", 0.3)
    assert r.converged and "NOT CONVERGED" not in r.label and not warns
    r = run("Biggs-Andrews", 0.53)          # stops at its limit of 600
    assert not r.converged and r.label.endswith("[NOT CONVERGED]") and warns
    warns.clear()
    tab.verifyVar.set(True)
    tab.mannAlphaVar.set("0.5")
    r = run("Picard / Mann iteration", 0.53)   # converges; no independent verifier converges
    assert r.converged and r.label.endswith("[UNVERIFIED]") and any("not verified" in w or "unverified" in w
                                                                    for w in warns)


def test_verifier_independence_families():
    import pytest
    from pysasfit.oz import ozLib
    fam = ozLib.solverFamily
    assert fam("MDIIS") == fam("scipy Anderson") == fam("sundials4py: Fixed-Point (Anderson)")
    assert fam("Biggs-Andrews") != fam("MDIIS") and fam("Picard iteration") == "Picard/Mann"
    # MDIIS never verifies scipy Anderson: only same-family verifiers given -> no verifier
    with pytest.raises(ValueError, match="same algorithm family"):
        ozLib.solve("HardSphere", phi=0.3, solver="scipy Anderson", verifyWith=("MDIIS",),
                    numberOfRadialSamplingPoints=1023)
    r = ozLib.solve("HardSphere", phi=0.3, solver="MDIIS", numberOfRadialSamplingPoints=1023)
    assert r.verifiedWith == "Biggs-Andrews"


def test_verification_unavailable_carries_result():
    """Primary Picard / Mann at phi = 0.53: its own family is skipped and
    Biggs-Andrews does not converge -> VerificationUnavailable with .result."""
    import pytest
    from pysasfit.oz import ozLib
    with pytest.raises(ozLib.VerificationUnavailable) as ei:
        ozLib.solve("HardSphere", phi=0.53, solver="Picard / Mann iteration", mannAlpha=0.5,
                    numberOfRadialSamplingPoints=1023)
    e = ei.value
    assert e.result is not None and e.result.converged
    assert any("Biggs-Andrews" in a for a in e.attempts)
    with pytest.raises(ValueError):
        ozLib.solve("HardSphere", phi=0.3, verifyMannAlpha=1.5)


def test_oz_tab_points_per_sigma_label():
    import os
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6 import QtWidgets
    QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    from pysasfit.oz.gui.oz_solver_tab import OZSolverTab
    t = OZSolverTab(redirect_output=False)
    assert t.pointsPerSigmaLabelVar.get() == "= 100 (auto), r up to 40.95 σ"
    t.pointsPerSigmaVar.set("200")
    assert t.pointsPerSigmaLabelVar.get() == "r up to 20.48 σ"
    t.gridPointsVar.set("14")
    assert t.pointsPerSigmaLabelVar.get() == "r up to 81.92 σ"


def test_physicality_screen_always_on():
    """min S(Q) >= 0 is checked on every solve, not only with deflation:
    Newton-Krylov finds an unphysical root for LJ eps=0.8, phi=0.3, HNC."""
    import pytest
    from pysasfit.oz import ozLib
    kw = dict(phi=0.3, potentialArgs=(0.8,), closure="Hypernetted-Chain", verify=False,
              numberOfRadialSamplingPoints=1023)
    with pytest.warns(RuntimeWarning, match="min S\\(Q\\)"):
        r = ozLib.solve("LennardJones", solver="scipy Newton-Krylov", **kw)
    assert r.converged and not r.physical and r.minSq < 0
    # scipy Anderson lands on an unphysical root here too (min S = -13.7), KIN_FP on the physical one
    with pytest.warns(RuntimeWarning, match="min S\\(Q\\)"):
        r = ozLib.solve("LennardJones", solver="scipy Anderson", **kw)
    assert not r.physical
    kin = "sundials4py: Fixed-Point (Anderson)"
    if kin in ozLib.SOLVER_CLASSES:
        r = ozLib.solve("LennardJones", solver=kin, **kw)
        assert r.physical and abs(r.minSq - 0.2168) < 1e-3


def test_residual_check_overrides_solver_claim():
    """KIN_FP reports success for polydisperse hard-core Yukawa (HNC, phi=0.1)
    but returns NaN; the independent residual check marks it not converged."""
    import pytest
    from pysasfit.oz import ozLib
    kin = "sundials4py: Fixed-Point (Anderson)"
    if kin not in ozLib.SOLVER_CLASSES:
        pytest.skip("SUNDIALS not installed")
    kw = dict(phi=0.1, potentialArgs=(0.2, 3, 200.0, 7.01 / 250.0), closure="Hypernetted-Chain",
              solver=kin, verify=False)
    with pytest.warns(RuntimeWarning, match="residual"):
        r = ozLib.solve("PolydisperseHardCoreYukawa", **kw)
    assert not r.converged and r.solverInstance.converged      # solver claimed success
    assert "residual" in r.convergenceMessage
    with pytest.warns(RuntimeWarning):                          # still unphysical (NaN S(Q))
        r = ozLib.solve("PolydisperseHardCoreYukawa", residualTolerance=None, **kw)
    assert r.converged                                          # check switched off


def test_residual_small_for_converged_solves():
    from pysasfit.oz import ozLib
    for cl in ("Percus-Yevick", "Hypernetted-Chain", "Rescaled MSA"):
        r = ozLib.solve("HardSphere", phi=0.3, closure=cl, verify=False, numberOfRadialSamplingPoints=1023)
        assert r.converged and r.residual is not None and r.residual < 1e-10, (cl, r.residual)
