"""pySASscout: features, scoring, candidate space and the model search on simulated data.

Runs without the SASfit libraries: the pure-Python reference functions of ``pysasfit.scout.reference`` are
registered under the SASfit names (a Python function takes precedence over the compiled one).
"""

import json
import math

import numpy as np
import pytest

from pysasfit import scout
from pysasfit.scout import Component, reference, scoring
from pysasfit.studio import function
from pysasfit.studio.model import ModelFunction, ScatteringContribution

Q = np.geomspace(0.005, 0.4, 70)


@pytest.fixture(autouse=True)
def _reference_functions():
    reference.register()
    yield
    reference.unregister()


def noisy(model, q=Q, rel=0.03, seed=0, understate=1.0):
    """Simulated measurement of ``model``: 3 % noise plus a floor; ``understate`` divides the reported sigma."""
    rng = np.random.default_rng(seed)
    I = model(q)
    s = rel * I + 0.0015 * I[-1]
    return q, I + rng.normal(0, s), s / understate


def sphere_lognorm(R=60.0, s=0.2, N=1e-12, bg=0.002, **kw):
    c = ScatteringContribution("ff_sphere", {"R": R, "eta": 1}, size_distribution="sd_LogNorm",
                               sd_params={"N": N, "s": s, "p": 1, "mu": R}, distributed="R", **kw)  # fmt: skip
    return ModelFunction([c], background=bg)


def core_shell(bg=0.002):
    c = ScatteringContribution("ff_spherical_shell_iii", {"R": 50, "dR": 15, "eta1": -0.5, "eta2": 1})
    return ModelFunction([c], scales=[1e-10], background=bg)


# ----------------------------------------------------------------------------- reference functions
def test_reference_functions_limits():
    q = np.array([1e-5, 0.01, 0.05, 0.2])
    V = 4 / 3 * math.pi * 50.0**3
    assert function("ff_sphere")(q, R=50, eta=2)[0] == pytest.approx((2 * V) ** 2, rel=1e-6)
    # a shell with the contrast of the core is a sphere of the outer radius; a hollow sphere is a shell with an empty core
    a = function("ff_spherical_shell_iii")(q, R=30, dR=20, eta1=1, eta2=1)
    assert np.allclose(a, function("ff_sphere")(q, R=50, eta=1), rtol=1e-9)
    b = function("ff_spherical_shell_iii")(q, R=30, dR=20, eta1=0, eta2=1)
    assert np.allclose(b, function("ff_hollow_sphere")(q, R=30, dR=20, eta=1), rtol=1e-9)
    # a spheroid with equal axes is a sphere
    c = function("ff_spheroid_R")(q, R_polar=50, R_equatorial=50, eta_core=1, eta_solv=0)
    assert np.allclose(c, function("ff_sphere")(q, R=50, eta=1), rtol=1e-6)
    # forward scattering of the cylinder, Guinier range of rod and disc
    assert function("ff_cylinder")(1e-6, R=20, L=100, eta=1) == pytest.approx((math.pi * 400 * 100) ** 2, rel=1e-6)
    assert function("ff_rod")(1e-3, L=100, eta=1) == pytest.approx(100.0**2 * math.exp(-(1e-3**2) * 100**2 / 36), rel=1e-6)
    assert function("ff_disc")(1e-3, R=50, eta=1) == pytest.approx((math.pi * 2500) ** 2 * math.exp(-(1e-3**2) * 2500 / 6), rel=1e-5)


def test_reference_lognorm_and_hard_sphere():
    x = np.linspace(1, 400, 40001)
    N = function("sd_LogNorm")(x, N=3.0, s=0.2, p=1, mu=60)
    assert np.sum(0.5 * (N[1:] + N[:-1]) * np.diff(x)) == pytest.approx(3.0, rel=1e-5)
    S = function("sq_HardSphere")(np.array([1e-5, 0.05, 2.0]), RHS=50, fp=0.3)
    assert S[0] == pytest.approx((1 - 0.3) ** 4 / (1 + 0.6) ** 2, rel=1e-3)
    assert S[2] == pytest.approx(1.0, abs=0.02)


# ----------------------------------------------------------------------------- features
def test_guinier_radius_of_spheres():
    q, I, s = noisy(sphere_lognorm(R=60, s=0.05))
    f = scout.extract(q, I, s)
    assert f.guinier_ok and f.rg == pytest.approx(math.sqrt(3 / 5) * 60, rel=0.1)
    assert not f.sq_hint and 0.5 < f.error_scale < 1.6
    lo, hi = f.size_range
    assert lo < f.size < hi and f.size == pytest.approx(60, rel=0.15)


def test_minima_of_a_monodisperse_form_factor_are_counted():
    f = scout.extract(*noisy(core_shell()))
    assert f.n_minima >= 3 and f.q_first_min < 0.1 and not f.sq_hint


def test_interference_peak_suggests_a_structure_factor():
    m = sphere_lognorm(s=0.15, structure_factor="sq_HardSphere", sq_params={"RHS": 60, "fp": 0.3}, approach="lma")
    f = scout.extract(*noisy(m))
    assert f.sq_hint and 0.02 < f.peak_q < 0.06
    assert any("structure factor" in n for n in f.notes)


def test_understated_errors_are_noticed():
    q, I, s = noisy(sphere_lognorm(), understate=4.0)
    f = scout.extract(q, I, s)
    assert f.error_scale > 2.5 and any("underestimated" in n for n in f.notes)


def test_missing_uncertainties_and_too_few_points():
    q, I, _s = noisy(sphere_lognorm())
    assert any("no uncertainties" in n for n in scout.extract(q, I).notes)
    with pytest.raises(ValueError):
        scout.extract(q[:5], I[:5])


# ----------------------------------------------------------------------------- scoring
def test_information_criteria():
    assert scoring.aic(10.0, 3) == 16.0
    assert scoring.aicc(10.0, 3, 20) == pytest.approx(16.0 + 24.0 / 16.0)
    assert scoring.aicc(10.0, 3, 4) == math.inf
    assert scoring.bic(10.0, 3, 100) == pytest.approx(10 + 3 * math.log(100))
    w = scoring.akaike_weights([100.0, 102.0, math.inf, 120.0])
    assert w.sum() == pytest.approx(1.0) and w[2] == 0 and w[0] / w[1] == pytest.approx(math.e)


def test_runs_test_separates_noise_from_structure():
    rng = np.random.default_rng(3)
    z, p = scoring.runs_test(rng.normal(size=200))
    assert abs(z) < 2.5 and p > 0.01
    z, p = scoring.runs_test(np.sin(np.linspace(0, 6, 200)))  # two long runs
    assert z < -10 and p < 1e-6


# ----------------------------------------------------------------------------- candidate space
def test_candidates_follow_the_features():
    f = scout.extract(*noisy(sphere_lognorm()))
    plain = scout.candidates.generate(f)
    assert (Component("sphere"),) in plain and (Component("sphere", "lognormal"),) in plain
    assert all(c[0].sq is None for c in plain)
    assert not any(c[0].form == "cylinder" and c[0].sd for c in plain)  # no size distribution declared for it
    with_sq = scout.candidates.generate(f, structure_factor="always")
    assert (Component("sphere", "lognormal", "hard-sphere"),) in with_sq and len(with_sq) > len(plain)
    assert not any(c[0].form == "rod" and c[0].sq for c in with_sq)
    assert len(scout.candidates.generate(f, forms=["sphere"], distributions=[])) == 1
    with pytest.raises(KeyError):
        scout.candidates.generate(f, forms=["dodecahedron"])


def test_bounds_follow_the_guinier_radius_except_in_mixtures():
    f = scout.extract(*noisy(sphere_lognorm()))
    _m, params, fixed, _a = scout.build((Component("sphere"),), f)
    a, v, lo, hi, lin = next(p for p in params if p[0] == "0.ff.R")
    assert lo == pytest.approx(v / 4) and hi == pytest.approx(v * 4) and not lin and fixed["0.ff.eta"] == 1.0
    assert next(p for p in params if p[0] == "scale.0")[4]  # linear
    _m, params, _f, _a = scout.build((Component("sphere"), Component("sphere")), f)
    a, v, lo, hi, lin = next(p for p in params if p[0] == "1.ff.R")
    assert lo == pytest.approx(1 / f.q_max) and hi == pytest.approx(2 * math.pi / f.q_min)


def test_structure_factor_radius_follows_the_particle_size_with_a_distribution():
    f = scout.extract(*noisy(sphere_lognorm()))
    model, params, _f, approaches = scout.build((Component("sphere", "lognormal", "hard-sphere"),), f)
    assert approaches == ["lma"] and model.contributions[0].approach == "lma"
    assert "0.sq.RHS" not in [p[0] for p in params] and "0.sq.fp" in [p[0] for p in params]
    _m, params, _f, approaches = scout.build((Component("sphere", None, "hard-sphere"),), f)
    assert approaches == ["monodisperse"] and "0.sq.RHS" in [p[0] for p in params]


# ----------------------------------------------------------------------------- the search
FAST = dict(reference=True, starts=4, max_starts=32, n_x=12, n_x_final=48)


@pytest.fixture(scope="module")
def polydisperse_result():
    reference.register()
    return scout.search(noisy(sphere_lognorm()), forms=["sphere", "core-shell", "rod"], name="sim", **FAST)


def test_polydisperse_spheres_are_found(polydisperse_result):
    r = polydisperse_result
    best = r.best
    assert best.components == (Component("sphere", "lognormal"),)
    assert best.parameters["0.sd.mu"] == pytest.approx(60, rel=0.1) and best.parameters["0.sd.s"] == pytest.approx(0.2, abs=0.05)
    assert best.parameters["background"] == pytest.approx(0.002, rel=0.2)
    assert 0.5 < best.chi2_red < 1.5 and best.k == 4 and best.n == Q.size
    assert sum(c.weight for c in r.candidates) == pytest.approx(1.0)
    crit = [c.aicc for c in r.candidates if c.success]
    assert crit == sorted(crit) and r.candidates[0].delta == 0
    rod = next(c for c in r.candidates if c.components == (Component("rod"),))
    assert rod.chi2_red > 20 and any("poor fit" in w for w in rod.warnings) and rod.weight < 1e-6


def test_text_and_protocol(polydisperse_result, tmp_path):
    r = polydisperse_result
    text = r.text(top=2)
    assert "sphere + LogNorm" in text and "ranking by AICC" in text and "Guinier" in text
    fn = r.save(str(tmp_path / "scout.json"))
    d = json.load(open(fn))
    assert d["options"]["seed"] == 1 and d["data"]["points"] == Q.size and len(d["data"]["sha256"]) == 64
    assert d["versions"]["functions"] == "python reference" and "numpy" in d["versions"]
    best = d["candidates"][0]
    assert best["functions"][0] == {"form_factor": "ff_sphere", "size_distribution": "sd_LogNorm", "structure_factor": None}
    assert len(best["I_fit"]) == Q.size and best["parameters"]["0.sd.mu"] == r.best.parameters["0.sd.mu"]
    assert "I_fit" not in r.as_dict(curves=False)["candidates"][0]


def test_model_of_a_candidate_reproduces_its_curve(polydisperse_result):
    r = polydisperse_result
    model = r.model(0)
    assert np.allclose(model(r.best.q), r.best.I_fit, rtol=1e-6)


def test_same_seed_same_result(polydisperse_result):
    again = scout.search(noisy(sphere_lognorm()), forms=["sphere", "core-shell", "rod"], name="sim", **FAST)
    for a, b in zip(polydisperse_result.candidates, again.candidates):
        assert a.label == b.label and a.chi2 == pytest.approx(b.chi2, rel=1e-9)
        assert a.parameters == pytest.approx(b.parameters, rel=1e-9)


def test_monodisperse_core_shell_needs_and_gets_more_starts():
    r = scout.search(noisy(core_shell()), forms=["sphere", "core-shell"], distributions=[], reference=True, starts=4, max_starts=64)
    best = r.best
    assert best.components == (Component("core-shell"),) and best.chi2_red < 1.5
    assert best.parameters["0.ff.R"] == pytest.approx(50, rel=0.03) and best.parameters["0.ff.dR"] == pytest.approx(15, rel=0.06)
    assert best.parameters["0.ff.eta1"] == pytest.approx(-0.5, abs=0.06) and best.weight > 0.99


def test_structure_factor_is_tried_only_when_the_data_suggest_it():
    m = sphere_lognorm(s=0.15, structure_factor="sq_HardSphere", sq_params={"RHS": 60, "fp": 0.25}, approach="lma")
    r = scout.search(noisy(m), forms=["sphere"], **FAST)
    assert r.best.components == (Component("sphere", "lognormal", "hard-sphere"),)
    assert r.best.parameters["0.sq.fp"] == pytest.approx(0.25, abs=0.03) and r.best.approaches == ["lma"]
    assert "'lma' approximation" in r.best.text()
    dilute = scout.search(noisy(sphere_lognorm()), forms=["sphere"], **FAST)
    assert all(c.components[0].sq is None for c in dilute.candidates)


def test_mixture_of_two_sizes():
    m = ModelFunction([ScatteringContribution("ff_sphere", {"R": 30, "eta": 1}), ScatteringContribution("ff_sphere", {"R": 150, "eta": 1})],
                      scales=[1e-9, 2e-12], background=0.002)  # fmt: skip
    r = scout.search(noisy(m), forms=["sphere"], distributions=[], max_components=3, reference=True, starts=4, max_starts=64)
    best = r.best
    assert best.components == (Component("sphere"), Component("sphere")) and best.chi2_red < 1.5
    assert sorted([best.parameters["0.ff.R"], best.parameters["1.ff.R"]]) == pytest.approx([30, 150], rel=0.03)
    assert any("contribution 2 accepted" in n for n in r.notes)
    assert any("no mixture tried" in n or "not supported" in n for n in r.notes)  # no third contribution


def test_no_mixture_when_one_contribution_is_enough(polydisperse_result):
    r = scout.search(noisy(sphere_lognorm()), forms=["sphere"], max_components=2, **FAST)
    assert len(r.best.components) == 1 and any("no mixture tried" in n for n in r.notes)
    assert all(len(c.components) == 1 for c in r.candidates)


def test_zero_amplitude_of_a_distribution_does_not_break_the_model():
    c = ScatteringContribution("ff_sphere", {"R": 50, "eta": 1}, size_distribution="sd_LogNorm",
                               sd_params={"N": 0.0, "s": 0.2, "p": 1, "mu": 50}, distributed="R")  # fmt: skip
    assert np.all(c.intensity(Q) == 0)


def test_unknown_case_is_reported():
    """A curve outside the candidate space: a warning instead of a confident recommendation."""
    q = Q
    I = 5.0 / (1 + (q * 80) ** 2) + 3.0 * np.exp(-0.5 * ((q - 0.15) / 0.01) ** 2) + 0.01
    s = 0.02 * I
    r = scout.search((q, I + np.random.default_rng(1).normal(0, s), s), forms=["sphere", "rod"], structure_factor="never", **FAST)
    assert any("unknown case" in n for n in r.notes) and r.best.warnings


def test_candidate_with_a_missing_function_is_listed_as_not_fitted():
    from pysasfit.scout import FormSpec, Par, candidates

    scout.register_form(FormSpec("ghost", "ff_no_such_function", (Par("R"),), label="ghost"))
    try:
        r = scout.search(noisy(sphere_lognorm()), forms=["sphere", "ghost"], distributions=[], **FAST)
    finally:
        del candidates.FORMS["ghost"]
    ghost = r.candidates[-1]
    assert ghost.label == "ghost" and not ghost.success and "ff_no_such_function" in ghost.message
    assert r.best.label == "sphere" and "ghost" in r.table()


def test_time_limit_and_options():
    r = scout.search(noisy(sphere_lognorm()), forms=["sphere"], time_limit=0.0, **FAST)
    assert r.best is None and all("time limit" in c.message for c in r.candidates)
    assert any("time limit" in n for n in r.notes)
    with pytest.raises(ValueError):
        scout.search(noisy(sphere_lognorm()), criterion="likes")


def test_error_scale_auto_and_bic():
    data = noisy(sphere_lognorm(), understate=4.0)
    raw = scout.search(data, forms=["sphere"], **FAST)
    auto = scout.search(data, forms=["sphere"], error_scale="auto", criterion="bic", **FAST)
    assert raw.best.chi2_red > 8 and auto.data["error_scale"] > 2.5
    assert auto.best.chi2_red < raw.best.chi2_red / 6
    assert [c.bic for c in auto.candidates] == sorted(c.bic for c in auto.candidates)


def test_worker_processes_give_the_same_ranking(polydisperse_result):
    r = scout.search(noisy(sphere_lognorm()), forms=["sphere", "core-shell", "rod"], name="sim", jobs=2, **FAST)
    assert [c.label for c in r.candidates] == [c.label for c in polydisperse_result.candidates]
    assert r.best.chi2 == pytest.approx(polydisperse_result.best.chi2, rel=1e-9)


# ----------------------------------------------------------------------------- command line
def test_cli(tmp_path, capsys):
    from pysasfit.scout.cli import main

    q, I, s = noisy(sphere_lognorm())
    fn = tmp_path / "sim.dat"
    np.savetxt(fn, np.column_stack([q, I, s]), header="q I sigma")
    out = tmp_path / "sim.json"
    assert main([str(fn), "--reference", "--forms", "sphere", "--starts", "4", "--json", str(out), "--top", "1"]) == 0
    text = capsys.readouterr().out
    assert "sphere + LogNorm" in text and "protocol written" in text and json.load(open(out))["program"] == "pySASscout"
    assert main([str(fn), "--features"]) == 0 and "Guinier" in capsys.readouterr().out
    assert main(["--list"]) == 0
    listing = capsys.readouterr().out
    assert "ff_spherical_shell_iii" in listing and "sd_LogNorm" in listing and "sq_HardSphere" in listing
    assert main([]) == 1
