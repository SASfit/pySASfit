"""Tests of pyDENSSmulti (pysasfit.densitymulti): contrasts, components, the constrained combination, the
reconstruction on synthetic multi-contrast objects, averaging, files, and the window (offscreen)."""
import math
import os

import numpy as np
import pytest

from pysasfit.densitymulti import core as M


# ------------------------------------------------------------------ contrasts and components
def test_component_sld_and_match_point():
    p = M.Component("protein", 1.82, 3.13, 1.0)
    assert p.sld(0.0) == pytest.approx(1.82)
    assert p.sld(1.0) == pytest.approx(3.13)
    assert 100 * p.match_point() == pytest.approx(42.4, abs=0.2)
    half = M.Component("half exchanged", 1.82, 3.13, 0.5)
    assert half.sld(1.0) == pytest.approx(1.82 + 0.5 * 1.31)
    lip = M.Component("tails", -0.39)                     # no exchangeable H
    assert lip.sld(1.0) == pytest.approx(-0.39)
    d = M.Component.from_dict(p.to_dict())
    assert d.sld(0.7) == pytest.approx(p.sld(0.7))


def test_contrast_matrix_and_identifiability():
    comps, cons = M.core_shell_curves()
    C, rs, rc = M.contrast_matrix(comps, cons)
    assert C.shape == (4, 2)
    assert rs[0] == pytest.approx(M.SLD_H2O) and rs[-1] == pytest.approx(M.SLD_D2O)
    idf = M.identifiability(C, names=[c.name for c in comps])
    assert idf["rank"] == 2 and idf["cond"] < 5
    assert idf["matched"][1] == [1]                       # protein shell matched at 42 % D2O
    assert not idf["messages"]
    # two components with the same SLDs cannot be separated
    same = [M.Component("a", 1.82, 3.13), M.Component("b", 1.82, 3.13)]
    C2, _, _ = M.contrast_matrix(same, cons)
    idf2 = M.identifiability(C2, names=["a", "b"])
    assert idf2["rank"] == 1 and any("rank" in m for m in idf2["messages"])
    # one contrast only: a component matched in all but one
    C3, _, _ = M.contrast_matrix(comps, cons[:2])
    assert any("all contrasts but one" in m for m in M.identifiability(C3, names=["core", "shell"])["messages"])


def test_xray_contrast_needs_xray_sld():
    comps = [M.Component("a", 1.0)]                       # no X-ray SLD
    k = M.Contrast("saxs", np.linspace(0.1, 2, 50), np.ones(50), np.ones(50) * 0.01, radiation="xray")
    with pytest.raises(M.MultiError):
        M.contrast_matrix(comps, [k])
    comps = [M.Component("a", 1.0, sld_xray=12.0)]
    C, rs, _ = M.contrast_matrix(comps, [k])
    assert rs[0] == pytest.approx(M.SLD_XRAY_WATER) and C[0, 0] == pytest.approx(12.0 - M.SLD_XRAY_WATER)


# ------------------------------------------------------------------ constraints
def test_capped_simplex_projection():
    rng = np.random.default_rng(0)
    x = rng.normal(0.4, 0.8, (3, 2000))
    p = M.project_capped_simplex(x)
    assert p.min() >= -1e-12 and p.sum(axis=0).max() <= 1 + 1e-9
    inside = (x >= 0).all(axis=0) & (x.sum(axis=0) <= 1)
    assert np.allclose(p[:, inside], x[:, inside])
    # optimality against random feasible points
    for j in range(20):
        y = M.project_capped_simplex(rng.random((3, 50)) * 2)
        assert np.all(np.sum((x[:, j:j + 1] - p[:, j:j + 1]) ** 2, axis=0)
                      <= np.sum((x[:, j:j + 1] - y) ** 2, axis=0) + 1e-9)


def test_bounded_lsq_matches_reference():
    from scipy.optimize import minimize
    rng = np.random.default_rng(1)
    C = np.array([[7.5, 2.4], [5.1, 0.0], [3.5, -1.5], [1.8, -3.2]])
    w = np.array([1.0, 2.0, 1.0, 0.5])
    D = rng.normal(2, 4, (4, 30))
    phi = M.bounded_lsq(D, C, w, iters=400)
    for j in range(30):
        f = lambda x: float(np.sum(w * (D[:, j] - C @ x) ** 2))  # noqa: E731
        ref = minimize(f, [0.3, 0.3], method="SLSQP", bounds=[(0, 1), (0, 1)],
                       constraints=[{"type": "ineq", "fun": lambda x: 1 - x.sum()}])
        assert f(phi[:, j]) <= ref.fun * (1 + 1e-4) + 1e-8


# ------------------------------------------------------------------ data handling
def test_smooth_curve_and_debye():
    comp, cons = M.sphere_curves(R_nm=2.0)
    q, I, e = cons[0].data()
    sm = M.smooth_curve(q / 10, I, e, 40.0)
    assert sm.rg == pytest.approx(math.sqrt(0.6) * 20, rel=0.02)
    assert 0.5 < sm.chi2 < 2
    # Debye average of a supersampled voxel sphere against the analytic form factor
    n, vox = 48, 2.0
    x = ((np.arange(4 * n) + 0.5) / 4 - n / 2) * vox
    X, Y, Z = np.meshgrid(x, x, x, indexing="ij", sparse=True)
    m = ((X ** 2 + Y ** 2 + Z ** 2) <= 400).astype(float).reshape(n, 4, n, 4, n, 4).mean(axis=(1, 3, 5))
    qa = np.linspace(0.01, 0.25, 40)
    I_map = M.debye_profile(m, vox, qa) * vox ** 6
    I_ex = M._sphere_amp(qa, 20.0) ** 2
    assert np.allclose(I_map[:10], I_ex[:10], rtol=0.01)


def test_match_point_from_forward_intensities():
    comps, cons = M.core_shell_curves()
    data = [k.data() for k in cons]
    I0 = [M.smooth_curve(q / 10, I, e, 70.0).I0 for q, I, e in data]
    rs = [k.solvent() for k in cons]
    # exact: volume-weighted SLDs (core 33510 Å³ d-protein, shell 146084 Å³ protein) matched at 59.3 % D2O
    assert M.match_sld(I0, rs) == pytest.approx(3.54, abs=0.06)


def test_tail_background():
    rng = np.random.default_rng(3)
    q = np.linspace(0.5, 5.0, 300)
    I0 = 50 * q ** -4 + 0.02
    e = 0.01 * I0
    bg, dbg = M.fit_tail_background(q, I0 + rng.normal(0, e), e)
    assert bg == pytest.approx(0.02, abs=max(4 * dbg, 1e-3))
    assert dbg < 0.002


# ------------------------------------------------------------------ reconstructions
def _radial(m, vox, edges, weights=None):
    n = m.shape[0]
    w = np.clip(np.nan_to_num(m if weights is None else weights), 0, None)
    idx = (np.arange(n) - n / 2) * vox
    X, Y, Z = np.meshgrid(idx, idx, idx, indexing="ij")
    t = w.sum()
    c = [(w * X).sum() / t, (w * Y).sum() / t, (w * Z).sum() / t]
    R = np.sqrt((X - c[0]) ** 2 + (Y - c[1]) ** 2 + (Z - c[2]) ** 2)
    return np.array([np.nanmean(np.where((R >= a) & (R < b), m, np.nan)) for a, b in zip(edges[:-1], edges[1:])])


def test_selftest_sphere():
    lines = []
    assert M.selftest(out=lines.append, steps=1200)
    assert "Rg" in lines[0]


def test_core_shell_components_absolute():
    comps, cons = M.core_shell_curves(conc_mg_ml=5, mw_kda=150)
    r = M.reconstruct(comps, cons, M.Settings(dmax_nm=7.0, seed=1, scale="absolute", conc_mg_ml=5, mw_kda=150,
                                              max_steps=1500))
    v_core, v_shell = 4 / 3 * np.pi * 20 ** 3, 4 / 3 * np.pi * (35 ** 3 - 20 ** 3)
    assert r.volumes[0] == pytest.approx(v_core, rel=0.15)
    assert r.volumes[1] == pytest.approx(v_shell, rel=0.15)
    edges = np.array([0, 14, 22, 32, 45.0])
    core, shell = (_radial(p, r.voxel, edges) for p in r.phi)
    assert core[0] > 0.8 and shell[0] < 0.2                    # centre: core
    assert shell[2] > 0.7 and core[2] < 0.2                    # 22 … 32 Å: shell
    assert np.all(r.chi2 < 1000)


def test_dumbbell_components_separate():
    comps, cons = M.dumbbell_curves()
    r = M.reconstruct(comps, cons, M.Settings(dmax_nm=7.0, seed=3, max_steps=1500))
    n = r.phi.shape[1]
    idx = (np.arange(n) - n / 2) * r.voxel
    X, Y, Z = np.meshgrid(idx, idx, idx, indexing="ij")
    cs = [np.array([(p * X).sum(), (p * Y).sum(), (p * Z).sum()]) / p.sum() for p in r.phi]
    assert np.linalg.norm(cs[0] - cs[1]) == pytest.approx(35.0, rel=0.25)
    assert np.minimum(r.phi[0], r.phi[1]).sum() / r.phi.sum() < 0.1


def test_average_scheme_without_exchange():
    comps = [M.Component("core", 4.0), M.Component("shell", 1.0)]
    comps, cons = M.core_shell_curves(components=comps, conc_mg_ml=5, mw_kda=150)
    r = M.reconstruct([], cons, M.Settings(dmax_nm=7.0, seed=1, scheme="average", scale="absolute", conc_mg_ml=5,
                                           mw_kda=150, max_steps=1500))
    assert r.rho is not None and r.phi.shape[0] == 1
    assert r.volumes[0] == pytest.approx(4 / 3 * np.pi * 35 ** 3, rel=0.15)
    prof = _radial(r.rho, r.voxel, np.array([0, 16, 26, 34.0]), weights=r.phi[0])
    assert prof[0] > prof[2] + 1.0                              # SLD higher in the core
    with pytest.raises(M.MultiError):
        M.reconstruct([], cons[:1], M.Settings(dmax_nm=7.0, scheme="average"))


def test_several_maps_save_load_cancel(tmp_path):
    comps, cons = M.dumbbell_curves()
    st = M.Settings(dmax_nm=7.0, seed=5, max_steps=1100)
    rs = M.reconstruct_many(comps, cons, st, nmaps=2, cores=2)
    a = M.align_and_average(rs)
    assert a.correlation[0] == 1.0 or a.correlation[1] == 1.0
    assert np.all(a.correlation > 0.7)
    assert a.mean.phi.shape == rs[0].phi.shape and a.phi_sd.shape == rs[0].phi.shape
    d = M.save_result(a, tmp_path / "avg")
    assert (d / "multi_summary.json").is_file()
    back = M.load_result(d)
    assert np.allclose(back.phi, a.mean.phi, atol=1e-5)
    d1 = M.save_result(rs[0], tmp_path / "one")
    one = M.load_result(d1)
    assert one.stats["chi2"].shape == rs[0].stats["chi2"].shape
    # cancel after the first step
    r = M.reconstruct(comps, cons, st, cancel=lambda: True)
    assert r.steps == 1 and any("stopped" in w for w in r.warnings)


def test_input_errors():
    comps, cons = M.core_shell_curves()
    with pytest.raises(M.MultiError):
        M.reconstruct([], cons, M.Settings(dmax_nm=7.0))
    with pytest.raises(M.MultiError):
        M.reconstruct(comps, [], M.Settings(dmax_nm=7.0))
    with pytest.raises(M.MultiError):
        M.reconstruct(comps, cons, M.Settings(dmax_nm=0.0))


# ------------------------------------------------------------------ window
@pytest.fixture
def window(tmp_path, monkeypatch):
    monkeypatch.setenv("PYSASFIT_SETTINGS", str(tmp_path / "settings.json"))
    monkeypatch.setenv("QT_QPA_PLATFORM", os.environ.get("QT_QPA_PLATFORM", "offscreen"))
    pytest.importorskip("PySide6")
    from PySide6 import QtWidgets
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    from pysasfit.densitymulti import gui as G
    w = G.MainWindow()
    yield w, app
    w._cancel = True
    w.close()


def test_guess_d2o():
    from pysasfit.densitymulti.gui import guess_d2o
    assert guess_d2o("lyz_42%D2O.dat") == pytest.approx(0.42)
    assert guess_d2o("sample 70 % D2O") == pytest.approx(0.70)
    assert guess_d2o("D2O_100") == pytest.approx(1.0)
    assert guess_d2o("buffer H2O") == 0.0
    assert guess_d2o("sample.dat") is None


def test_window_run_session_and_code(window, tmp_path):
    import time
    w, app = window
    w.load_example("core-shell")
    assert w.con_table.rowCount() == 4
    assert "separated" in w.matrix_label.text()
    w.steps.setValue(300)
    w.seed.setValue(2)
    w.run()
    t0 = time.time()
    while w._busy and time.time() - t0 < 120:
        app.processEvents()
        time.sleep(0.02)
    app.processEvents()
    assert len(w.runs) == 1 and w.table.rowCount() == 1
    r = w.runs[0]["result"]
    assert r.workdir is not None and (r.workdir / "multi_summary.json").is_file()
    assert w.view3d.list.count() >= 1
    # session: components, contrasts and the run come back
    st = w.session_state()
    w2 = type(w)()
    w2.apply_state(st)
    assert w2.con_table.rowCount() == 4 and len(w2.components()) == 2 and len(w2.runs) == 1
    w2.close()
    # the generated script runs
    code = w.python_code().replace("PLOT = True", "PLOT = False")
    ns = {}
    exec(compile(code, "generated", "exec"), ns)
    assert len(ns["r"].chi2) == 4
    # the averaging scheme hides the component matrix
    w.scheme.setCurrentIndex(1)
    assert "one SLD" in w.matrix_label.text()


def test_launcher_has_tab(tmp_path, monkeypatch):
    monkeypatch.setenv("PYSASFIT_SETTINGS", str(tmp_path / "settings.json"))
    pytest.importorskip("PySide6")
    from pysasfit.gui import launcher
    assert any(t[0] == "pyDENSSmulti" and t[1] == "pysasfit.densitymulti.gui" for t in launcher.TOOLS)
    from pysasfit.resources import tool_icon, wordmark
    from PySide6 import QtWidgets
    QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    assert not tool_icon("pyDENSSmulti").isNull()
    assert wordmark("pyDENSSmulti")[1][0] == "DENSS"
