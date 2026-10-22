"""bumps backend: the optimisers and options of the bumps manual (lm, amoeba, de, dream, newton, pt)."""

import pytest

pytest.importorskip("bumps")
try:
    from .support import problem
except ImportError:
    from support import problem

from pysasfit.studio import fitting  # noqa: E402


def test_parse_option_text():
    d = fitting.parse_option_text("steps=2000, pop=20; init=lhs\nCR=0.5, trim=True")
    assert d == {"steps": 2000, "pop": 20, "init": "lhs", "CR": 0.5, "trim": True}
    assert fitting.parse_option_text("") == {} and fitting.parse_option_text(None) == {}


@pytest.mark.parametrize("bad", ["steps", "=3", "a=1, b"])
def test_parse_option_text_needs_name_value(bad):
    with pytest.raises(ValueError, match="name=value"):
        fitting.parse_option_text(bad)


def test_the_algorithms_of_the_manual_are_offered():
    assert {"lm", "amoeba", "de", "dream", "newton", "pt"} <= set(fitting._BUMPS_ALGORITHMS)


@pytest.mark.parametrize("name", ["lm", "amoeba", "newton"])
def test_local_bumps_methods_find_the_minimum(name):
    p = problem()
    r = p.fit("bumps", name, bumps={"steps": 500})
    assert r.chi2_red < 3.0
    assert abs(r.values["ff.k"] - 0.8) < 0.1


def test_differential_evolution_with_options():
    r = problem().fit("bumps", "de", bumps={"steps": 60, "pop": 6, "CR": 0.8})
    assert r.chi2_red < 3.0


def test_dream_options_reach_bumps_and_give_posterior_errors():
    r = problem().fit("bumps", "dream", bumps={"samples": 1500, "burn": 30, "pop": 6, "init": "lhs", "thin": 1})
    assert "posterior samples" in r.message
    assert r.asymmetric  # 68 % intervals from the chain


def test_parallel_tempering_runs_and_needs_bounds():
    r = problem().fit("bumps", "pt", bumps={"steps": 40, "nT": 4, "burn": 10})
    assert r.nfev > 0 and r.chi2_red == r.chi2_red  # experimental in bumps: it must run, not be accurate
    with pytest.raises(ValueError, match="bound"):
        problem(upper=False).fit("bumps", "pt")


def test_unknown_option_is_refused_with_the_known_ones():
    with pytest.raises(ValueError, match="unknown option.*bogus.*steps"):
        problem().fit("bumps", "lm", bumps={"bogus": 1})
    with pytest.raises(ValueError, match="unknown option.*nT"):  # nT belongs to pt, not to dream
        problem().fit("bumps", "dream", bumps={"nT": 3})
