"""Global search: scaling of the search box, multistart, differential evolution and the other global backends."""

import threading

import numpy as np
import pytest

from pysasfit.studio import fitting
from pysasfit.studio.fitting import FitProblem, _Context

try:
    from .support import Q, installed, osc_problem, problem, make_data
except ImportError:
    from support import Q, installed, osc_problem, problem, make_data


def W(r):
    return r.values["ff.w"]


# ------------------------------------------------------------------ scaling of the box
def test_rescale_gives_unit_boxes_and_keeps_the_start_point():
    p = osc_problem()
    p.parameters[2].value = 1e-6  # the background starts almost at its lower bound
    ctx = _Context(p, None, None, 1e3)
    before = ctx.vals(ctx.u0).copy()
    ctx.rescale_to_bounds()
    assert np.allclose(ctx.ub - ctx.lb, 1.0)
    assert np.allclose(ctx.vals(ctx.u0), before)


def test_default_scale_is_the_start_value_and_rescale_uses_the_range():
    p = osc_problem()
    ctx = _Context(p, None, None, 1e3)
    assert np.allclose(ctx.scale, [1.0, 3.0, 0.4])  # start values
    ctx.rescale_to_bounds()
    assert np.allclose(ctx.scale, [9.9, 11.0, 2.0])  # upper - lower


def test_rescale_leaves_unbounded_variables_alone():
    p = FitProblem()
    p.add_dataset(make_data(), problem().datasets[0].model)
    p.add_parameter("0.ff.A", lower=0, upper=20)
    p.add_parameter("0.ff.k", lower=0)  # no upper bound
    p.add_parameter("background")  # no bounds
    ctx = _Context(p, None, None, 1e3)
    s0 = ctx.scale.copy()
    ctx.rescale_to_bounds()
    assert ctx.scale[0] == 20.0 and np.allclose(ctx.scale[1:], s0[1:])
    assert np.isfinite(ctx.ub[0]) and not np.isfinite(ctx.ub[1])


def test_rescale_is_idempotent():
    ctx = _Context(osc_problem(), None, None, 1e3)
    ctx.rescale_to_bounds()
    a = (ctx.scale.copy(), ctx.u0.copy(), ctx.lb.copy(), ctx.ub.copy())
    ctx.rescale_to_bounds()
    for x, y in zip(a, (ctx.scale, ctx.u0, ctx.lb, ctx.ub)):
        assert np.allclose(x, y)


def test_variables_map_back_to_the_original_units_after_rescaling():
    ctx = _Context(osc_problem(), None, None, 1e3)
    ctx.rescale_to_bounds()
    v = ctx.vals(ctx.lb)
    assert np.allclose(v, [0.1, 1.0, 0.0])
    assert np.allclose(ctx.vals(ctx.ub), [10.0, 12.0, 2.0])


def test_errors_after_a_global_fit_agree_with_a_local_fit():
    local = osc_problem(w_start=7.0).fit("scipy")
    glob = osc_problem(w_start=7.0).fit("scipy-global", "differential_evolution", polish=True)
    for k in ("ff.A", "ff.w", "background"):
        assert np.isclose(glob.errors[k], local.errors[k], rtol=0.1)


# ------------------------------------------------------------------ bounds are required
@pytest.mark.parametrize("alg", ["differential_evolution", "dual_annealing", "shgo", "multistart"])
def test_global_methods_need_finite_bounds(alg):
    p = problem(upper=False)
    with pytest.raises(ValueError, match="lower and an upper bound"):
        p.fit("scipy-global", alg)


def test_basinhopping_works_without_bounds():
    p = problem(upper=False)
    r = p.fit("scipy-global", "basinhopping", niter=3)
    assert np.isfinite(r.chi2)


def test_unknown_global_method():
    with pytest.raises(ValueError, match="unknown global method"):
        problem().fit("scipy-global", "annealing_of_dreams")


def test_multistart_is_listed_with_the_other_global_algorithms():
    algs = fitting.available_backends()["scipy-global"][1]
    assert "multistart" in algs and "differential_evolution" in algs


# ------------------------------------------------------------------ multistart
def test_multistart_finds_the_global_minimum_where_a_local_fit_does_not():
    local = osc_problem().fit("scipy")
    r = osc_problem().fit("scipy-global", "multistart", starts=25)
    assert abs(W(r) - 7.0) < 0.05
    assert r.chi2_red < local.chi2_red
    assert "distinct minima" in r.message and "multistart" in r.message


def test_multistart_is_reproducible_with_a_seed_and_robust_to_the_seed():
    a = osc_problem().fit("scipy-global", "multistart", starts=25, seed=7)
    b = osc_problem().fit("scipy-global", "multistart", starts=25, seed=7)
    assert W(a) == W(b)
    c = osc_problem().fit("scipy-global", "multistart", starts=40, seed=123)
    assert abs(W(c) - 7.0) < 0.05


def test_multistart_with_one_start_is_a_local_fit_from_the_current_values():
    one = osc_problem().fit("scipy-global", "multistart", starts=1)
    local = osc_problem().fit("scipy")
    assert np.isclose(W(one), W(local), atol=1e-3)
    assert "1 of 1" in one.message


def test_multistart_with_a_good_start_reports_every_start_in_the_best_minimum_or_others():
    r = osc_problem(w_start=7.0).fit("scipy-global", "multistart", starts=10)
    assert abs(W(r) - 7.0) < 0.05
    assert "of 10" in r.message


def test_multistart_counts_evaluations_and_respects_a_budget():
    small = osc_problem().fit("scipy-global", "multistart", starts=4)
    big = osc_problem().fit("scipy-global", "multistart", starts=16)
    assert big.nfev > small.nfev
    capped = osc_problem().fit("scipy-global", "multistart", starts=10, maxeval=500)
    assert capped.nfev < 10 * 600  # at most max(maxeval // starts, 50) evaluations per start (plus the Jacobian steps)


def test_multistart_returns_cancelled_when_stopped():
    cancel = threading.Event()
    cancel.set()
    r = osc_problem().fit("scipy-global", "multistart", starts=10, cancel=cancel)
    assert r.message == "cancelled" and not r.success


def test_multistart_start_points_stay_in_the_bounds_and_are_spread_out(monkeypatch):
    seen = []
    real = fitting._run_scipy

    def spy(ctx, algorithm, options):
        seen.append((np.array(options["start"]) * ctx.scale, ctx.lb * ctx.scale, ctx.ub * ctx.scale))
        return real(ctx, algorithm, {**options, "max_nfev": 2})

    monkeypatch.setattr(fitting, "_run_scipy", spy)
    osc_problem().fit("scipy-global", "multistart", starts=30, seed=4)
    pts = np.array([s for s, _lo, _hi in seen])
    lo, hi = seen[0][1], seen[0][2]
    assert len(seen) == 30
    assert np.all(pts >= lo - 1e-12) and np.all(pts <= hi + 1e-12)
    assert np.allclose(pts[0], [1.0, 3.0, 0.4])  # the first start is the current parameter set
    # a Latin hypercube covers every tenth of the range of w
    w = np.sort(pts[1:, 1])
    bins = np.histogram(w, bins=10, range=(1, 12))[0]
    assert np.all(bins >= 1)


def test_multistart_samples_wide_positive_ranges_log_uniformly(monkeypatch):
    p = FitProblem()
    p.add_dataset(make_data(), problem().datasets[0].model)
    p.add_parameter("0.ff.A", lower=1e-3, upper=1e3)  # six decades
    p.add_parameter("0.ff.k", lower=0, upper=5)
    seen = []
    real = fitting._run_scipy

    def spy(ctx, algorithm, options):
        seen.append(np.array(options["start"]) * ctx.scale)
        return real(ctx, algorithm, {**options, "max_nfev": 2})

    monkeypatch.setattr(fitting, "_run_scipy", spy)
    p.fit("scipy-global", "multistart", starts=41, seed=1)
    a = np.array([s[0] for s in seen[1:]])
    assert np.exp(np.mean(np.log(a))) < 10 and np.exp(np.mean(np.log(a))) > 0.1  # geometric centre near 1
    assert np.sum(a < 1.0) >= 10 and np.sum(a > 1.0) >= 10  # not all in the top decade as a uniform sample would be


def test_multistart_groups_equal_minima_in_the_message(monkeypatch):
    results = iter([1.0, 1.0004, 5.0, 5.0, 5.0, 9.0])  # chi2/dof of the six starts

    def fake(ctx, algorithm, options):
        return np.array(options["start"], float), "ok", True

    monkeypatch.setattr(fitting, "_run_scipy", fake)
    monkeypatch.setattr(fitting, "_penalised", lambda ctx: (lambda u: next(results)))
    ctx = _Context(osc_problem(), None, None, 1e3)
    u, text, ok = fitting._run_multistart(ctx, {"starts": 6})
    assert ok and "3 distinct minima" in text and "1 (x2)" in text and "5 (x3)" in text and "9 (x1)" in text


def test_multistart_survives_starts_that_fail(monkeypatch):
    calls = {"n": 0}
    real = fitting._run_scipy

    def flaky(ctx, algorithm, options):
        calls["n"] += 1
        if calls["n"] % 2 == 0:
            raise RuntimeError("model cannot be evaluated here")
        return real(ctx, algorithm, options)

    monkeypatch.setattr(fitting, "_run_scipy", flaky)
    r = osc_problem().fit("scipy-global", "multistart", starts=8)
    assert "4 of 8" in r.message


def test_multistart_raises_when_no_start_works(monkeypatch):
    def broken(ctx, algorithm, options):
        raise RuntimeError("nothing works")

    monkeypatch.setattr(fitting, "_run_scipy", broken)
    with pytest.raises(ValueError, match="no start point"):
        osc_problem().fit("scipy-global", "multistart", starts=3)


def test_multistart_respects_constraints():
    p = osc_problem()
    p.add_constraint("ff_w < 5")
    r = p.fit("scipy-global", "multistart", starts=15)
    assert r.values["ff.w"] <= 5.0 + 1e-3
    assert all(ok for _text, _g, ok in r.constraints)


def test_multistart_with_variable_projection():
    r = osc_problem().fit("scipy-global", "multistart", starts=20, vp=True)
    assert abs(W(r) - 7.0) < 0.05
    assert "variable projection" in r.message


# ------------------------------------------------------------------ the other global methods
def test_differential_evolution_finds_the_global_minimum():
    r = osc_problem().fit("scipy-global", "differential_evolution", polish=True)
    assert abs(W(r) - 7.0) < 0.05


def test_differential_evolution_is_reproducible_and_respects_a_budget():
    a = osc_problem().fit("scipy-global", "differential_evolution", seed=3, maxeval=2000)
    b = osc_problem().fit("scipy-global", "differential_evolution", seed=3, maxeval=2000)
    assert W(a) == W(b)
    assert a.nfev < 2000 * 2


def test_differential_evolution_with_variable_projection_searches_fewer_parameters():
    plain = osc_problem().fit("scipy-global", "differential_evolution", seed=1)
    vp = osc_problem().fit("scipy-global", "differential_evolution", seed=1, vp=True)
    assert abs(W(vp) - 7.0) < 0.05
    assert vp.nfev < plain.nfev


def test_differential_evolution_with_a_constraint():
    p = osc_problem()
    p.add_constraint("ff_w < 5")
    r = p.fit("scipy-global", "differential_evolution")
    assert r.values["ff.w"] < 5.0 + 1e-6


def test_global_fit_does_not_end_worse_than_it_started():
    start = osc_problem(w_start=3.0)
    chi_start = float(np.sum(start._residuals(np.array([x.value for x in start.parameters])) ** 2))
    for alg in ("differential_evolution", "dual_annealing", "multistart"):
        opts = {"starts": 5} if alg == "multistart" else {"seed": 1}
        r = osc_problem(w_start=3.0).fit("scipy-global", alg, **opts)
        assert r.chi2 <= chi_start + 1e-9, alg


def test_dual_annealing_and_shgo_run_and_return_finite_results():
    for alg, opts in (("dual_annealing", {"maxeval": 3000}), ("shgo", {"n": 16, "maxeval": 800})):
        r = osc_problem().fit("scipy-global", alg, **opts)
        assert np.isfinite(r.chi2) and r.nfev > 0, alg


def test_polish_after_a_global_search_gives_errors_and_a_converged_result():
    r = osc_problem().fit("scipy-global", "differential_evolution", polish=True)
    assert "refined with least_squares" in r.message
    assert all(np.isfinite(e) and e > 0 for e in r.errors.values())


@pytest.mark.skipif(not installed("cma"), reason="cma is not installed")
def test_cma_finds_the_global_minimum_in_a_unit_box():
    r = osc_problem().fit("cma", seed=2, polish=True)
    assert abs(W(r) - 7.0) < 0.2


@pytest.mark.skipif(not installed("nlopt"), reason="nlopt is not installed")
def test_nlopt_global_algorithm_needs_bounds_and_finds_the_minimum():
    with pytest.raises(ValueError, match="global"):
        problem(upper=False).fit("nlopt", "GN_DIRECT_L")
    r = osc_problem().fit("nlopt", "GN_DIRECT_L", maxeval=4000, polish=True)
    assert abs(W(r) - 7.0) < 0.1


def test_coarse_to_fine_with_a_global_optimiser_is_a_no_op_without_size_distribution():
    r = osc_problem().fit("scipy-global", "multistart", starts=15, coarse=True)
    assert abs(W(r) - 7.0) < 0.05
