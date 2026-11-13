"""pySASscout against the real SASfit libraries; skipped unless they are installed.

Checks that every function of the default candidate list exists with the parameter names the scout uses, and
that the pure-Python reference functions agree with the compiled ones.
"""

import numpy as np
import pytest

from pysasfit import scout, studio
from pysasfit.scout import reference

pytestmark = pytest.mark.skipif(not studio.available(), reason="SASfit libraries not installed")

Q = np.geomspace(0.004, 0.3, 25)
VALUES = {
    "ff_sphere": {"R": 50, "eta": 1},
    "ff_spherical_shell_iii": {"R": 40, "dR": 12, "eta1": -0.5, "eta2": 1},
    "ff_hollow_sphere": {"R": 40, "dR": 12, "eta": 1},
    "ff_cylinder": {"R": 20, "L": 150, "eta": 1},
    "ff_spheroid_R": {"R_polar": 80, "R_equatorial": 30, "eta_core": 1, "eta_solv": 0},
    "ff_rod": {"L": 300, "eta": 1},
    "ff_disc": {"R": 100, "eta": 1},
    "sq_HardSphere": {"RHS": 50, "fp": 0.25},
}


@pytest.fixture(autouse=True)
def _compiled_functions():
    reference.unregister()
    yield
    reference.unregister()


def test_candidate_functions_exist_with_the_expected_parameters():
    specs = list(scout.FORMS.values()) + list(scout.DISTRIBUTIONS.values()) + list(scout.STRUCTURE_FACTORS.values())
    for spec in specs:
        f = studio.function(spec.function)
        assert f.language == "c", spec.function
        names = [p.name for p in spec.fitted] + [k for k, _v in spec.fixed]
        names += [getattr(spec, a) for a in ("sd_param", "amplitude", "location", "radius") if getattr(spec, a, None)]
        for n in names:
            assert n in f.parameter_names, f"{spec.function} has no parameter {n}: {f.parameter_names}"


@pytest.mark.parametrize("name", sorted(VALUES))
def test_reference_agrees_with_sasfit(name):
    compiled = studio.function(name)(Q, **VALUES[name])
    reference.register()
    python = studio.function(name)(Q, **VALUES[name])
    assert np.allclose(python, compiled, rtol=2e-3, atol=1e-6 * np.max(np.abs(compiled))), name


def test_reference_lognorm_agrees_with_sasfit():
    x = np.linspace(10, 200, 50)
    compiled = studio.function("sd_LogNorm")(x, N=2, s=0.2, p=1, mu=60)
    reference.register()
    assert np.allclose(studio.function("sd_LogNorm")(x, N=2, s=0.2, p=1, mu=60), compiled, rtol=1e-8)


def test_search_with_the_compiled_functions():
    from pysasfit.studio.model import ModelFunction, ScatteringContribution

    c = ScatteringContribution("ff_sphere", {"R": 60, "eta": 1}, size_distribution="sd_LogNorm",
                               sd_params={"N": 1e-12, "s": 0.2, "p": 1, "mu": 60}, distributed="R")  # fmt: skip
    q = np.geomspace(0.005, 0.4, 70)
    I = ModelFunction([c], background=0.002)(q)
    s = 0.03 * I + 0.0015 * I[-1]
    r = scout.search((q, I + np.random.default_rng(0).normal(0, s), s), forms=["sphere", "core-shell", "cylinder"], starts=4)
    assert r.versions["functions"] == "sasfit"
    assert r.best.components == (scout.Component("sphere", "lognormal"),)
    assert r.best.parameters["0.sd.mu"] == pytest.approx(60, rel=0.1)
