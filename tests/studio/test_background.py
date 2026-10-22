"""Power-law background  I(q) = sum + background + A * q^-alpha  (model, addresses, fit, variable projection)."""

import numpy as np
import pytest

from pysasfit.studio.fitting import FitProblem
from pysasfit.studio.model import ModelFunction, Pipeline

try:
    from .support import Q, Toy, installed, make_data, make_model, power_problem
except ImportError:  # run from inside tests/studio
    from support import Q, Toy, installed, make_data, make_model, power_problem


# ------------------------------------------------------------------ the model
def test_power_law_is_off_by_default():
    m = make_model(bg=0.5)
    assert m.bg_A == 0.0 and m.bg_alpha == 4.0
    assert m.power_term(Q) == 0.0
    assert np.allclose(m(Q), ModelFunction([Toy(2.0, 1.0)], scales=[1.0], background=0.5)(Q))


def test_power_law_is_added_to_the_model():
    m = make_model(bg=0.5, bg_A=2.0, bg_alpha=3.0)
    base = make_model(bg=0.5)
    assert np.allclose(m(Q) - base(Q), 2.0 * Q**-3.0)


@pytest.mark.parametrize("alpha", [0.0, 1.0, 2.5, 4.0, 6.0])
def test_power_term_values(alpha):
    m = make_model(bg_A=0.3, bg_alpha=alpha)
    assert np.allclose(m.power_term(Q), 0.3 * Q ** (-alpha))


def test_power_term_accepts_a_scalar_and_a_list():
    m = make_model(bg_A=2.0, bg_alpha=2.0)
    assert np.isclose(m.power_term(2.0), 0.5)
    assert np.allclose(m.power_term([1.0, 2.0]), [2.0, 0.5])


def test_power_law_adds_to_a_background_function_and_to_none():
    m = ModelFunction([Toy()], scales=[1.0], background=lambda q: 0.1 * np.ones_like(q), bg_A=1.0, bg_alpha=2.0)
    base = ModelFunction([Toy()], scales=[1.0], background=lambda q: 0.1 * np.ones_like(q))
    assert np.allclose(m(Q) - base(Q), Q**-2.0)
    m = ModelFunction([Toy()], scales=[1.0], background=None, bg_A=1.0, bg_alpha=2.0)
    assert np.allclose(m(Q) - ModelFunction([Toy()], scales=[1.0], background=None)(Q), Q**-2.0)


def test_components_return_the_whole_background():
    m = make_model(bg=0.5, bg_A=2.0, bg_alpha=3.0)
    parts, bg = m.components(Q)
    assert np.allclose(bg, 0.5 + 2.0 * Q**-3.0)
    assert np.allclose(m(Q), m.scales[0] * parts[0] + bg)


def test_zero_q_gives_a_non_finite_value_not_an_exception():
    m = make_model(bg_A=1.0, bg_alpha=2.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        assert not np.all(np.isfinite(m(np.array([0.0, 1.0]))))


def test_bg_attributes_can_be_set_after_construction():
    m = make_model()
    m.bg_A, m.bg_alpha = 1.0, 2.0
    assert np.allclose(m.power_term(Q), Q**-2.0)


# ------------------------------------------------------------------ addresses
def test_fit_addresses_read_and_write_the_model():
    p = power_problem(alpha_free=True)
    names = [x.address for x in p.parameters]
    assert "bg.A" in names and "bg.alpha" in names
    model = p.datasets[0].model
    p.apply(0, np.array([2.0, 1.0, 0.1, 0.7, 5.0]))
    assert model.bg_A == 0.7 and model.bg_alpha == 5.0


def test_pipeline_addresses():
    pipe = Pipeline()
    m = make_model(bg_A=0.5, bg_alpha=3.0)
    pipe.add_model(m)
    assert pipe.get("bg.A") == 0.5 and pipe.get("bg.alpha") == 3.0
    pipe.set("bg.A", 2.0)
    pipe.set("bg.alpha", 1.5)
    assert m.bg_A == 2.0 and m.bg_alpha == 1.5


def test_unknown_background_address_is_refused():
    p = FitProblem()
    p.add_dataset(make_data(), make_model())
    with pytest.raises((KeyError, ValueError, IndexError)):
        p.add_parameter("bg.beta")


# ------------------------------------------------------------------ fitting
@pytest.mark.parametrize("vp", [False, True])
def test_power_law_background_is_fitted(vp):
    r = power_problem().fit("scipy", vp=vp)
    assert r.success
    assert abs(r.values["bg.A"] - 0.02) < 0.004
    assert abs(r.values["background"] - 0.3) < 0.05
    assert abs(r.values["ff.A"] - 5.0) < 0.3 and abs(r.values["ff.k"] - 0.8) < 0.05


def test_variable_projection_gives_the_same_minimum():
    # no jitter: the finite-difference Jacobian is exact, so both reach the same minimum
    a = power_problem(jitter=0.0).fit("scipy", vp=False)
    b = power_problem(jitter=0.0).fit("scipy", vp=True)
    assert abs(a.chi2 - b.chi2) < 1e-4 * max(a.chi2, 1.0)
    for key in ("ff.A", "ff.k", "background", "bg.A"):
        assert abs(a.values[key] - b.values[key]) < 1e-2 * max(abs(a.values[key]), 0.1)


def test_exponent_can_be_fitted_with_and_without_variable_projection():
    for vp in (False, True):
        r = power_problem(alpha_free=True).fit("scipy", vp=vp)
        assert abs(r.values["bg.alpha"] - 4.0) < 0.6


def test_variable_projection_treats_bg_A_as_linear_and_alpha_as_nonlinear():
    p = power_problem(alpha_free=True)
    lin = p.linear_parameters()
    kinds = {p.parameters[i].address: p._lin_kinds[i][0] for i in lin}
    assert kinds.get("bg.A") == "bgA" and kinds.get("background") == "bg"
    assert "bg.alpha" not in kinds and "0.ff.k" not in kinds


def test_a_prior_keeps_bg_A_with_the_optimiser():
    p = power_problem()
    p.add_prior_text("bg.A 0.02 0.01")
    kinds = {p.parameters[i].address for i in p.linear_parameters()}
    assert "bg.A" not in kinds


def test_fixed_bg_A_is_an_offset_in_variable_projection():
    p = power_problem(bg_A=0.02)
    p.parameter("bg.A").fixed = True
    r = p.fit("scipy", vp=True)
    assert abs(r.values["background"] - 0.3) < 0.05 and abs(r.values["bg.A"] - 0.02) < 1e-12
    assert "bg.A" not in r.errors


def test_lower_bound_keeps_bg_A_non_negative_when_there_is_no_power_law_in_the_data():
    p = FitProblem()
    p.add_dataset(make_data(5.0, 0.8, 0.3), make_model(2.0, 1.0, bg=0.1, bg_A=0.01))
    for a, lo, hi in (("0.ff.A", 0, 20), ("0.ff.k", 0, 5), ("background", -1, 2)):
        p.add_parameter(a, lower=lo, upper=hi)
    p.add_parameter("bg.A", lower=0)
    r = p.fit("scipy", vp=True)
    assert r.values["bg.A"] >= 0.0 and r.values["bg.A"] < 0.01
    assert abs(r.values["background"] - 0.3) < 0.1


def test_the_error_of_bg_A_is_reported_and_positive():
    r = power_problem().fit("scipy")
    assert r.errors["bg.A"] > 0 and np.isfinite(r.errors["bg.A"])
    assert "bg.A" in r.free_labels


def test_two_data_sets_with_a_common_exponent_and_a_local_amplitude():
    qa = Q
    rng = np.random.default_rng(5)

    def data(A, bgA, seed):
        I = A * np.exp(-0.8 * qa) + 0.2 + bgA * qa**-3.0
        s = 0.01 * (I + 0.05)
        return qa, I + np.random.default_rng(seed).normal(0, s), s

    model = make_model(bg=0.1, bg_A=0.05, bg_alpha=3.0)
    p = FitProblem()
    p.add_dataset(data(5, 0.02, 1), model, name="a")
    p.add_dataset(data(3, 0.05, 2), model, name="b")
    p.add_common("0.ff.k", lower=0)
    p.add_common("background", lower=-1, upper=2)
    p.add_common("bg.alpha", lower=1, upper=6)
    p.add_local("0.ff.A", lower=0)
    p.add_local("bg.A", lower=0)
    r = p.fit("scipy")
    assert abs(r.values["bg.alpha"] - 3.0) < 0.5
    assert abs(r.values["background"] - 0.2) < 0.05
    del rng


def test_export_lists_the_background_parameters(tmp_path):
    from pysasfit.studio import export

    p = power_problem(alpha_free=True)
    r = p.fit("scipy")
    labels = [row[0] for row in export.parameter_rows(r)]
    assert "bg.A" in labels and "bg.alpha" in labels


@pytest.mark.skipif(not installed("nlopt"), reason="nlopt is not installed")
def test_power_law_with_nlopt():
    r = power_problem().fit("nlopt", "LN_BOBYQA", polish=True, vp=True)
    assert abs(r.values["bg.A"] - 0.02) < 0.006
