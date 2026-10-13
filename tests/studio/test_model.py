"""Size-distribution integration, structure-factor approaches and fitting."""

import numpy as np
from scipy.integrate import quad

from pysasfit import studio
from pysasfit.studio import ModelFunction, Pipeline, ScatteringContribution


def lognorm(x, s, mu):
    return np.exp(-0.5 * (np.log(x / mu) / s) ** 2) / (np.sqrt(2 * np.pi) * s * x)


def sphere_I(q, R):
    return (4 * np.pi * (np.sin(q * R) - q * R * np.cos(q * R)) / q**3) ** 2


def test_monodisperse_without_distribution(studio_lib):
    q = np.linspace(0.01, 0.5, 30)
    c = ScatteringContribution("ff_sphere", {"R": 20, "eta": 1}, structure_factor="sq_HardSphere",
                               sq_params={"RHS": 20, "fp": 0.2})
    S = studio.function("sq_HardSphere")(q, RHS=20, fp=0.2)
    assert np.allclose(c.intensity(q), sphere_I(q, 20) * S)
    assert S[0] < 0.5 and abs(S[-1] - 1) < 0.1  # PY: compressed at low q


def test_lognormal_polydispersity(studio_lib):
    q = np.array([0.02, 0.1, 0.3])
    c = ScatteringContribution("ff_sphere", {"R": 1, "eta": 1}, size_distribution="sd_LogNorm",
                               sd_params={"N": 1, "s": 0.2, "p": 1, "mu": 30}, distributed="R")
    ref = [quad(lambda R: lognorm(R, 0.2, 30) * sphere_I(qq, R), 1, 200, limit=400)[0] for qq in q]
    assert np.allclose(c.intensity(q), ref, rtol=1e-6)


def test_decoupling_and_lma(studio_lib):
    q = np.array([0.02, 0.06, 0.15])
    kw = dict(ff_params={"R": 1, "eta": 1}, size_distribution="sd_LogNorm",
              sd_params={"N": 1, "s": 0.1, "p": 1, "mu": 30}, distributed="R",
              structure_factor="sq_HardSphere", sq_params={"RHS": 30, "fp": 0.3})
    mono = ScatteringContribution("ff_sphere", approach="monodisperse", **kw).intensity(q)
    dec = ScatteringContribution("ff_sphere", approach="decoupling", **kw).intensity(q)
    lma = ScatteringContribution("ff_sphere", approach="lma", **kw).intensity(q)
    # reference decoupling: <P> + <F>^2 (S-1)
    S = studio.function("sq_HardSphere")(q, RHS=30, fp=0.3)
    F = lambda qq, R: 4 * np.pi * (np.sin(qq * R) - qq * R * np.cos(qq * R)) / qq**3  # noqa: E731
    P = np.array([quad(lambda R: lognorm(R, 0.1, 30) * F(qq, R) ** 2, 5, 100)[0] for qq in q])
    Fm = np.array([quad(lambda R: lognorm(R, 0.1, 30) * F(qq, R), 5, 100)[0] for qq in q])
    assert np.allclose(dec, P + Fm**2 * (S - 1), rtol=1e-6)
    assert np.allclose(mono, P * S, rtol=1e-6)
    # LMA: S evaluated at each R
    Sx = studio.function("sq_HardSphere")
    L = np.array([quad(lambda R: lognorm(R, 0.1, 30) * F(qq, R) ** 2 * Sx(qq, RHS=R, fp=0.3), 5, 100)[0] for qq in q])
    assert np.allclose(lma, L, rtol=1e-5)
    assert not np.allclose(lma, mono)


def test_python_plugin_and_fit(studio_lib):
    studio.register_python_function(
        "ff_test_guinier",
        lambda q, p: p[0] * np.exp(-(q * p[1]) ** 2 / 3),
        parameters=["I0", "Rg"],
        defaults=[1.0, 10.0],
    )
    try:
        q = np.geomspace(0.005, 0.3, 60)
        rng = np.random.default_rng(1)
        true = 50 * np.exp(-(q * 25) ** 2 / 3) + 0.05
        s = 0.02 * true
        I = true + rng.normal(size=q.size) * s
        pipe = Pipeline()
        d = pipe.add_dataset((q, I, s))
        m = pipe.add_model(ModelFunction([ScatteringContribution("ff_test_guinier", {"I0": 10, "Rg": 15})],
                                         background=0.0), dataset=d)
        r = pipe.fit(m, ["0.ff.I0", "0.ff.Rg", "background"])
        assert r.success and 0.5 < r.chi2_red < 1.6
        assert abs(r.values[1] - 25) < 3 * r.errors[1] + 0.2
        assert abs(pipe.get("0.ff.Rg") - r.values[1]) < 1e-12
        assert "chi2" in r.summary() and "ff_test_guinier" in str(pipe)
    finally:
        studio.unregister_python_function("ff_test_guinier")


def test_python_plugin_file(tmp_path, monkeypatch):
    from pysasfit.studio import cli

    plugin, _ = cli.new_plugin("tguin", str(tmp_path))
    monkeypatch.setenv("PYSASFIT_STUDIO_PLUGINS", str(tmp_path))
    f = studio.function("ff_tguin")
    assert f.language == "python" and f.parameter_names == ["I0", "Rg"]
    assert np.isclose(f(1e-8, I0=2.5), 2.5)
    assert f.volume(3.0, dist=1) > 0


def test_resolution_in_model(studio_lib):
    from pysasfit.core import Resolution

    q = np.linspace(0.02, 0.3, 40)
    c = ScatteringContribution("ff_sphere", {"R": 30, "eta": 1})
    sharp = ModelFunction([c]).intensity(q)
    smeared = ModelFunction([c], resolution=Resolution(0.1 * q)).intensity(q)
    i = np.argmin(sharp[5:]) + 5  # first minimum gets filled in
    assert smeared[i] > 5 * sharp[i]


def test_reproduces_sasfit_project_result(studio_lib):
    """SASfit test data: bimodal LogNorm spheres; parameters and errors from data/SASfitTestData/SASfit_test.sas.

    The mock libsasfit implements ff_sphere and sd_LogNorm with the formulas of SASfit's C sources.
    """
    import os

    fn = os.path.join(os.path.dirname(__file__), "..", "..", "examples", "studio", "data", "SASfitTestData", "test.dat")
    pipe = Pipeline()
    d = pipe.add_dataset(fn)
    sd = lambda N, s, mu: dict(size_distribution="sd_LogNorm", sd_params={"N": N, "s": s, "p": 1, "mu": mu},  # noqa: E731
                               distributed="R")
    m = pipe.add_model(ModelFunction([ScatteringContribution("ff_sphere", {"R": 1, "eta": 1}, **sd(3.42718e-12, 0.303892, 155.173)),
                                      ScatteringContribution("ff_sphere", {"R": 1, "eta": 1}, **sd(7.39795e-11, 0.200369, 69.43))],
                                     background=0.0999523), dataset=d)
    r = pipe.fit(m, ["0.sd.mu"], scale_errors=False, max_nfev=3)
    assert 1.0 < r.chi2 / (r.q.size - 7) < 1.3
    assert abs(r.values[0] - 155.173) < 0.05
    # SASfit's error of mu (a4,err) with all 7 parameters free is 5.44 -- here only mu is free, so smaller
    assert 0 < r.errors[0] < 5.44
