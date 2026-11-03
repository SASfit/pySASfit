"""Several data sets: contributions per data set, global parameters P1, P2 … with factors, constraints on them."""
import math

import numpy as np
import pytest

from pysasfit.studio import ModelFunction, ScatteringContribution, register_python_function, unregister_python_function
from pysasfit.studio.fitting import FitProblem


def _sphere(q, p):
    R, eta = p[0], p[1]
    x = q * R
    with np.errstate(invalid="ignore", divide="ignore"):
        f = np.where(x > 1e-6, 3 * (np.sin(x) - x * np.cos(x)) / x**3, 1.0)
    V = 4 / 3 * np.pi * R**3
    return (eta * V * f) ** 2


@pytest.fixture(autouse=True)
def plugin():
    register_python_function("ff_pysphere", _sphere, parameters=("R", "eta"), defaults=(5.0, 1.0))
    yield
    unregister_python_function("ff_pysphere")


def _data(model, q, rel=0.01, seed=0):
    rng = np.random.default_rng(seed)
    I = model(q)
    s = rel * I
    return q, I + rng.normal(0, s), s


def test_contribution_active_per_dataset():
    c1 = ScatteringContribution("ff_pysphere", {"R": 5.0, "eta": 1.0})
    c2 = ScatteringContribution("ff_pysphere", {"R": 2.0, "eta": 1.0})
    q = np.geomspace(0.05, 2, 50)
    m_a = ModelFunction([c1, c2], scales=[1e-6, 1e-4], active=[True, False])
    m_b = ModelFunction([c1, c2], scales=[1e-6, 1e-4], active=[False, True])
    assert np.allclose(m_a(q), 1e-6 * c1(q)) and np.allclose(m_b(q), 1e-4 * c2(q))
    parts, _bg = m_a.components(q)
    assert np.all(parts[1] == 0)


def test_global_parameter_with_factors_two_datasets():
    """Data set a: spheres of radius P1; data set b: spheres of radius 2 P1 (core and twice-larger shell, say)."""
    q = np.geomspace(0.05, 1.5, 80)
    c1 = ScatteringContribution("ff_pysphere", {"R": 6.0, "eta": 1.0})
    c2 = ScatteringContribution("ff_pysphere", {"R": 12.0, "eta": 1.0})
    truth_a = ModelFunction([c1, c2], scales=[1e-6, 1e-7], active=[True, False], background=0.01)
    truth_b = ModelFunction([c1, c2], scales=[1e-6, 1e-7], active=[False, True], background=0.02)
    da, db = _data(truth_a, q, seed=1), _data(truth_b, q, seed=2)
    c1.ff_p[0], c2.ff_p[0] = 5.0, 9.0                        # start values off
    ma = ModelFunction([c1, c2], scales=[1e-6, 1e-7], active=[True, False], background=0.0)
    mb = ModelFunction([c1, c2], scales=[1e-6, 1e-7], active=[False, True], background=0.0)
    prob = FitProblem()
    prob.add_dataset(da, ma, name="a")
    prob.add_dataset(db, mb, name="b")
    prob.add_global("P1", 5.5, lower=1, upper=20)
    prob.link("P1", "0.ff.R", datasets=[0])
    prob.link("P1", "1.ff.R", datasets=[1], factor=2.0)
    prob.add_local("background")
    res = prob.fit("scipy")
    assert res.success
    assert res.values["P1"] == pytest.approx(6.0, rel=0.01)
    assert res.values["background[a]"] == pytest.approx(0.01, rel=0.2)
    prob.apply(1)
    assert c2.ff_p[0] == pytest.approx(2 * res.values["P1"])
    assert "P1" in res.text()
    # a model parameter can only be driven once
    with pytest.raises(ValueError):
        prob.link("P1", "0.ff.R", datasets=[0])


def test_constraint_on_global_parameters():
    q = np.geomspace(0.05, 1.5, 60)
    c1 = ScatteringContribution("ff_pysphere", {"R": 6.0, "eta": 1.0})
    m = ModelFunction([c1], scales=[1e-6])
    d = _data(m, q)
    prob = FitProblem()
    prob.add_dataset(d, ModelFunction([c1], scales=[1e-6]), name="a")
    prob.add_global("P1", 4.0, lower=1, upper=20)
    prob.add_global("P2", 2.0, lower=0.1, upper=20)
    prob.link("P1", "0.ff.R")
    prob.link("P2", "0.ff.eta", factor=0.5)
    prob.add_constraint("P1 < 5")
    res = prob.fit("scipy")
    assert res.values["P1"] <= 5.0 + 1e-2
    assert all(ok for _t, _g, ok in res.constraints)


def test_global_not_eliminated_by_variable_projection():
    q = np.geomspace(0.05, 1.5, 60)
    c1 = ScatteringContribution("ff_pysphere", {"R": 6.0, "eta": 1.0})
    d = _data(ModelFunction([c1], scales=[2e-6], background=0.01), q)
    prob = FitProblem()
    prob.add_dataset(d, ModelFunction([c1], scales=[1e-6], background=0.0), name="a")
    prob.add_global("P1", 1e-6, lower=0)
    prob.link("P1", "scale.0")
    prob.add_common("background")
    assert 0 not in prob.linear_parameters()
    res = prob.fit("scipy", vp=True)
    assert res.values["P1"] == pytest.approx(2e-6, rel=0.02)


def test_expression_links():
    """A model parameter as any expression of global parameters: P1 + P2, atan(P1), sqrt(...)."""
    q = np.geomspace(0.05, 1.5, 60)
    c1 = ScatteringContribution("ff_pysphere", {"R": 6.0, "eta": 1.0})
    c2 = ScatteringContribution("ff_pysphere", {"R": 3.0, "eta": 1.0})
    truth = ModelFunction([c1, c2], scales=[1e-6, 1e-5])
    d = _data(truth, q, seed=4)
    c1.ff_p[0], c2.ff_p[0] = 5.0, 2.0
    prob = FitProblem()
    prob.add_dataset(d, ModelFunction([c1, c2], scales=[1e-6, 1e-5]), name="a")
    prob.add_global("P1", 2.8, lower=0.5, upper=10)
    prob.add_global("P2", 2.8, lower=0.5, upper=10)
    prob.link_expression("0.ff.R", "P1 + P2")           # R1 = P1 + P2
    prob.link("P1", "1.ff.R")                          # R2 = P1
    res = prob.fit("scipy")
    assert res.values["P1"] == pytest.approx(3.0, rel=0.01) and res.values["P2"] == pytest.approx(3.0, rel=0.02)
    assert res.derived and res.derived[0][2] == pytest.approx(6.0, rel=0.01) and "P1 + P2" in res.text()
    prob.link_expression("0.ff.eta", "1 + 0*atan(P1)")
    prob.apply(0)
    assert c1.ff_p[1] == pytest.approx(1.0)
    with pytest.raises(ValueError):
        prob.link_expression("1.ff.eta", "P9 * 2")        # unknown global
    with pytest.raises(ValueError):
        prob.link_expression("1.ff.eta", "__import__('os')")


def test_link_parser_expressions():
    from pysasfit.studio.multidata import parse_link
    ln = parse_link("c1.ff.R = P1+P1")
    assert ln.factor is None and ln.names == {"P1"} and ln.value({"P1": 2.0}) == 4.0
    ln = parse_link("c1.ff.R = atan(P1)")
    assert ln.factor is None and ln.value({"P1": 1.0}) == pytest.approx(math.pi / 4)
    ln = parse_link("c1.ff.R[2] = 3 * P2")
    assert ln.factor == 3.0 and ln.name == "P2" and ln.datasets == [1]
    with pytest.raises(ValueError):
        parse_link("c1.ff.R = 5")                         # no global parameter
