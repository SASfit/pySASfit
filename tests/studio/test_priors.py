"""Priors: Gaussian, log-normal, half-normal, log-uniform (parsing, residual rows, scalar term, backends)."""

import math

import numpy as np
import pytest

from pysasfit.studio.fitting import BIG, FitProblem

try:
    from .support import Q, installed, make_data, make_model, osc_problem, problem
except ImportError:
    from support import Q, installed, make_data, make_model, osc_problem, problem


def vals_of(p):
    return np.array([x.value for x in p.parameters], float)


# ------------------------------------------------------------------ parsing
def test_gaussian_prior_text_variants():
    for line in ("ff.w 7 0.5", "ff.w, 7, 0.5", "ff.w = 7 ± 0.5", "ff.w 7 +- 0.5", "ff.w;7;0.5", "  ff.w   normal  7  0.5 "):
        p = osc_problem()
        p.add_prior_text(line)
        assert p.priors == [(1, 7.0, 0.5)], line


def test_prior_kinds_are_case_insensitive():
    p = osc_problem()
    p.add_prior_text("ff.A LogNormal 2 1.5")
    p.add_prior_text("background HALFNORMAL 0.5")
    assert len(p.prior_extra) == 2


def test_all_four_kinds_are_stored_separately():
    p = osc_problem()
    p.add_prior_text("ff.w 7 0.5")
    p.add_prior_text("ff.A lognormal 2 1.5")
    p.add_prior_text("background halfnormal 0.5")
    p.add_prior_text("ff.A loguniform")
    assert len(p.priors) == 1 and len(p.prior_extra) == 2 and p.log_uniform == [0]
    assert p.n_prior_rows() == 3  # the log-uniform prior is no residual


@pytest.mark.parametrize(
    "bad",
    [
        "ff.w 7",  # sigma missing
        "ff.w 7 0.5 1",  # too many numbers
        "ff.w abc def",  # not numbers
        "ff.w 7 0",  # sigma = 0
        "ff.w 7 -1",  # sigma < 0
        "ff.A lognormal 2",  # factor missing
        "ff.A lognormal 2 1",  # factor must be > 1
        "ff.A lognormal 2 0.5",
        "ff.A lognormal -2 1.5",  # median must be > 0
        "ff.A lognormal 0 1.5",
        "background halfnormal",  # sigma missing
        "background halfnormal 0",
        "background halfnormal -1",
        "background halfnormal 1 2",
        "background loguniform",  # lower bound is 0
        "ff.A loguniform 3",  # no numbers allowed
        "nothing 1 2",  # unknown parameter
        "nothing lognormal 1 2",
        "",
    ],
)
def test_bad_prior_lines_raise_value_error(bad):
    with pytest.raises(ValueError):
        osc_problem().add_prior_text(bad)


def test_error_messages_name_the_line_and_the_usage():
    p = osc_problem()
    with pytest.raises(ValueError, match="name  lognormal  median  factor"):
        p.add_prior_text("ff.A lognormal 2")
    with pytest.raises(ValueError, match="unknown parameter"):
        p.add_prior_text("zzz 1 2")
    with pytest.raises(ValueError, match="ff.w abc def"):
        p.add_prior_text("ff.w abc def")


def test_add_prior_api_and_unknown_kind():
    p = osc_problem()
    p.add_prior("ff.w", 7.0, 0.5)
    p.add_prior("ff.A", 2.0, 1.5, kind="lognormal")
    p.add_prior("background", 0.5, kind="halfnormal")
    p.add_prior("ff.A", kind="loguniform")
    assert p.n_prior_rows() == 3
    with pytest.raises(ValueError, match="unknown prior"):
        p.add_prior("ff.w", 1, 2, kind="cauchy")
    with pytest.raises(ValueError):
        p.add_prior("ff.w", 1)  # wrong number of arguments


def test_a_failed_prior_leaves_the_problem_unchanged():
    p = osc_problem()
    for bad in ("ff.A lognormal 2 0.5", "background loguniform", "ff.w 7 0"):
        with pytest.raises(ValueError):
            p.add_prior_text(bad)
    assert p.priors == [] and p.prior_extra == [] and p.log_uniform == []


# ------------------------------------------------------------------ residual rows
def test_gaussian_row():
    p = osc_problem()
    p.add_prior_text("ff.w 7 0.5")
    rows = np.concatenate(p._prior_rows(np.array([1.0, 8.0, 0.2])))
    assert np.allclose(rows, [(8.0 - 7.0) / 0.5])


def test_lognormal_and_halfnormal_rows():
    p = osc_problem()
    p.add_prior_text("ff.A lognormal 2 1.5")
    p.add_prior_text("background halfnormal 0.5")
    rows = np.concatenate(p._prior_rows(np.array([4.0, 7.0, 0.25])))
    assert np.allclose(rows, [math.log(4.0 / 2.0) / math.log(1.5), 0.25 / 0.5])


def test_lognormal_is_symmetric_in_log_space_and_zero_at_the_median():
    p = osc_problem()
    p.add_prior_text("ff.A lognormal 2 3")
    up, down, mid = (np.concatenate(p._prior_rows(np.array([a, 7.0, 0.2])))[0] for a in (6.0, 2.0 / 3.0, 2.0))
    assert np.isclose(up, 1.0) and np.isclose(down, -1.0) and abs(mid) < 1e-12  # one factor = one sigma


def test_lognormal_refuses_non_positive_values_with_a_big_residual():
    p = osc_problem()
    p.add_prior_text("ff.A lognormal 2 1.5")
    for x in (0.0, -1.0):
        assert np.concatenate(p._prior_rows(np.array([x, 7.0, 0.2])))[0] == BIG


def test_halfnormal_is_quadratic_from_zero():
    p = osc_problem()
    p.add_prior_text("background halfnormal 0.5")
    for x in (0.0, 0.5, 1.0):
        assert np.isclose(np.concatenate(p._prior_rows(np.array([1.0, 7.0, x])))[0], x / 0.5)


def test_prior_rows_are_appended_to_the_residuals_in_order():
    p = osc_problem()
    p.add_prior_text("ff.w 7 0.5")
    p.add_prior_text("ff.A lognormal 2 1.5")
    p.add_prior_text("background halfnormal 0.5")
    v = np.array([4.0, 8.0, 0.25])
    r = p._residuals(v)
    assert r.size == Q.size + 3
    assert np.allclose(r[-3:], [2.0, math.log(2.0) / math.log(1.5), 0.5])


def test_prior_penalty_of_the_loguniform_prior():
    p = osc_problem()
    assert p.prior_penalty(np.array([1.0, 7.0, 0.2])) == 0.0
    p.add_prior_text("ff.A loguniform")
    assert np.isclose(p.prior_penalty(np.array([math.e, 7.0, 0.2])), 2.0)
    assert np.isclose(p.prior_penalty(np.array([1.0, 7.0, 0.2])), 0.0)
    assert p.prior_penalty(np.array([-1.0, 7.0, 0.2])) == BIG
    assert p._residuals(np.array([math.e, 7.0, 0.2])).size == Q.size  # no extra residual row


def test_loguniform_penalty_prefers_small_values_per_decade_equally():
    p = osc_problem()
    p.add_prior_text("ff.A loguniform")
    # equal ratios cost equal amounts: ln(10x) - ln(x) is the same for every x
    d1 = p.prior_penalty(np.array([1.0, 7.0, 0.2])) - p.prior_penalty(np.array([0.1, 7.0, 0.2]))
    d2 = p.prior_penalty(np.array([10.0, 7.0, 0.2])) - p.prior_penalty(np.array([1.0, 7.0, 0.2]))
    assert np.isclose(d1, d2)


# ------------------------------------------------------------------ fits
def test_gaussian_prior_pulls_a_poorly_determined_parameter():
    q = np.linspace(0.1, 3, 8)
    data = make_data(q=q)
    data = (q, data[1], data[2] * 30)  # very noisy: the data barely constrain k
    free = problem(data)
    free.fit("scipy")
    tight = problem(data)
    tight.add_prior_text("ff.k 0.5 0.01")
    r = tight.fit("scipy")
    assert abs(r.values["ff.k"] - 0.5) < 0.02


def test_a_prior_shifts_a_parameter_toward_the_prior_mean():
    base = osc_problem(w_start=7.0).fit("scipy")
    p = osc_problem(w_start=7.0)
    p.add_prior_text("ff.A 1.0 0.5")  # data say about 2
    r = p.fit("scipy")
    assert 1.0 < r.values["ff.A"] < base.values["ff.A"]


def test_a_tight_prior_wins_against_the_data():
    p = osc_problem(w_start=7.0)
    p.add_prior_text("ff.A 1.0 0.001")
    a = p.fit("scipy").values["ff.A"]
    assert abs(a - 1.0) < 0.2 < abs(a - 2.0)  # much closer to the prior mean than to the value the data prefer (about 2)


def test_prior_rows_count_as_points_in_the_degrees_of_freedom():
    base = osc_problem(w_start=7.0).fit("scipy")
    p = osc_problem(w_start=7.0)
    p.add_prior_text("ff.A lognormal 2 3")
    p.add_prior_text("background halfnormal 5")
    assert p.fit("scipy").dof == base.dof + 2


def test_lognormal_prior_keeps_a_parameter_positive_and_near_the_median():
    p = osc_problem(w_start=7.0)
    p.add_prior_text("ff.A lognormal 1 1.05")
    r = p.fit("scipy")
    assert r.values["ff.A"] > 0
    assert r.values["ff.A"] < osc_problem(w_start=7.0).fit("scipy").values["ff.A"]


def test_halfnormal_prior_pulls_toward_zero():
    base = osc_problem(w_start=7.0).fit("scipy")
    p = osc_problem(w_start=7.0)
    p.add_prior_text("background halfnormal 0.05")
    r = p.fit("scipy")
    assert 0 <= r.values["background"] < base.values["background"]


def test_variable_projection_keeps_prior_parameters_with_the_optimiser():
    p = problem()
    p.add_prior_text("background halfnormal 0.5")
    p.add_prior_text("ff.A lognormal 5 2")
    assert p.linear_parameters() == []  # A is no linear parameter; background has a prior
    r = p.fit("scipy", vp=True)
    assert r.success and np.isfinite(r.chi2)


def test_vp_and_plain_fit_agree_with_priors():
    def build():
        p = problem(make_data())
        p.add_prior_text("background halfnormal 1")
        return p

    a, b = build().fit("scipy", vp=False), build().fit("scipy", vp=True)
    assert abs(a.chi2 - b.chi2) < 0.05 * max(a.chi2, 1.0)


def test_loguniform_is_refused_by_the_least_squares_backends():
    p = osc_problem(w_start=7.0)
    p.add_prior_text("ff.A loguniform")
    with pytest.raises(ValueError, match="log-uniform"):
        p.fit("scipy")
    with pytest.raises(ValueError, match="log-uniform"):
        p.fit("scipy-global", "multistart")


@pytest.mark.skipif(not installed("dfols"), reason="DFO-LS is not installed")
def test_loguniform_is_refused_by_dfols():
    p = osc_problem(w_start=7.0)
    p.add_prior_text("ff.A loguniform")
    with pytest.raises(ValueError, match="log-uniform"):
        p.fit("dfols")


def test_loguniform_with_differential_evolution_finds_the_data_minimum():
    p = osc_problem(w_start=7.0)
    p.add_prior_text("ff.A loguniform")
    r = p.fit("scipy-global", "differential_evolution", polish=True)  # polish is skipped silently
    assert abs(r.values["ff.w"] - 7.0) < 0.05 and abs(r.values["ff.A"] - 2.0) < 0.1
    assert "refined with least_squares" not in r.message


@pytest.mark.skipif(not installed("nlopt"), reason="nlopt is not installed")
def test_loguniform_with_nlopt():
    p = osc_problem(w_start=7.0)
    p.add_prior_text("ff.A loguniform")
    r = p.fit("nlopt", "LN_BOBYQA", polish=True)
    assert abs(r.values["ff.w"] - 7.0) < 0.1


@pytest.mark.skipif(not installed("cma"), reason="cma is not installed")
def test_loguniform_with_cma():
    p = osc_problem(w_start=7.0)
    p.add_prior_text("ff.A loguniform")
    r = p.fit("cma", maxeval=3000, seed=2)
    assert abs(r.values["ff.w"] - 7.0) < 0.2


def test_loguniform_term_changes_the_choice_among_equal_chi2_points():
    # with a flat likelihood only the prior counts: ln x is cheaper for small x
    ctx_problem = osc_problem(w_start=7.0)
    ctx_problem.add_prior_text("ff.A loguniform")
    small, large = np.array([0.2, 7.0, 0.5]), np.array([5.0, 7.0, 0.5])
    assert ctx_problem.prior_penalty(small) < ctx_problem.prior_penalty(large)


def test_coarse_to_fine_keeps_a_global_optimiser_with_loguniform():
    # without a size distribution coarse to fine is a no-op; the call must still accept a log-uniform prior
    p = osc_problem(w_start=7.0)
    p.add_prior_text("ff.A loguniform")
    r = p.fit("scipy-global", "differential_evolution", coarse=True)
    assert abs(r.values["ff.w"] - 7.0) < 0.05


def test_priors_do_not_leak_between_problems():
    a, b = osc_problem(), osc_problem()
    a.add_prior_text("ff.w 7 0.5")
    a.add_prior_text("ff.A loguniform")
    assert b.priors == [] and b.log_uniform == [] and b.prior_extra == []


def test_summary_text_still_works_with_every_prior_kind():
    p = osc_problem(w_start=7.0)
    p.add_prior_text("ff.w 7 0.5")
    p.add_prior_text("ff.A lognormal 2 3")
    p.add_prior_text("background halfnormal 5")
    assert "chi" in p.fit("scipy").text().lower()


def test_two_data_sets_with_priors():
    try:
        from .support import two_dataset_problem
    except ImportError:
        from support import two_dataset_problem
    free = two_dataset_problem().fit("scipy")
    p = two_dataset_problem()
    p.add_prior_text("ff.k 0.8 0.02")
    r = p.fit("scipy")
    assert abs(r.values["ff.k"] - 0.8) < abs(free.values["ff.k"] - 0.8)
    assert r.dof == sum(d.q.size for d in p.datasets) + 1 - len(r.free_labels)
