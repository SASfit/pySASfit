"""Preferences: registry, persistence, dialog, and that the tools use them."""
import json
import os

import numpy as np
import pytest

from pysasfit.core import settings as prefs


@pytest.fixture
def prefs_file(tmp_path, monkeypatch):
    path = tmp_path / "settings.json"
    monkeypatch.setenv("PYSASFIT_SETTINGS", str(path))
    prefs.load()
    yield path
    prefs.reset()
    prefs.load(str(tmp_path / "none.json"))


def test_defaults_roundtrip_and_validation(prefs_file):
    keys = [s.key for sec in prefs.sections() for s in sec.settings]
    assert len(keys) == len(set(keys)) > 80
    assert prefs.get("sldprofile.opt.alpha_step") == 0.5
    prefs.set_value("sldprofile.opt.alpha_step", "0.25")
    prefs.set_value("inversion.opt.evidence_lambda_scale", "median")
    prefs.set_value("sldprofile.opt.max_iter", 300)            # = factory: not stored
    with pytest.raises(ValueError):
        prefs.set_value("sldprofile.opt.alpha_step", 0.0)      # below its lower limit
    with pytest.raises(ValueError):
        prefs.set_value("inversion.opt.evidence_lambda_scale", "nonsense")
    prefs.save()
    data = json.loads(prefs_file.read_text())
    assert data["values"] == {"inversion.opt.evidence_lambda_scale": "median",
                              "sldprofile.opt.alpha_step": 0.25}
    prefs.reset()
    assert prefs.get("sldprofile.opt.alpha_step") == 0.5
    prefs.load()                                               # back from the file
    assert prefs.get("sldprofile.opt.alpha_step") == 0.25 and not prefs.is_default("sldprofile.opt.alpha_step")
    assert prefs.values("sldprofile.ift")["n_alpha"] == 101
    prefs.reset("sldprofile")
    assert prefs.get("sldprofile.opt.alpha_step") == 0.5
    assert prefs.get("inversion.opt.evidence_lambda_scale") == "median"


def test_bad_file_entries_fall_back(prefs_file):
    prefs_file.write_text(json.dumps({"values": {"sldprofile.opt.alpha_step": -3,
                                                 "future.unknown": 1}}))
    with pytest.warns(UserWarning):
        prefs.load()
    assert prefs.get("sldprofile.opt.alpha_step") == 0.5
    prefs.save()
    assert json.loads(prefs_file.read_text())["values"] == {"future.unknown": 1}   # kept for newer versions


def test_factory_values_match_the_code():
    """An empty preferences file must reproduce the library defaults."""
    from pysasfit.sldprofile.density import DensityProfileFit
    from pysasfit.sldprofile.twostep import IFT_DEFAULTS
    from pysasfit.inversion.solver_registry import OPTION_DEFAULTS
    from pysasfit.oz.oZsolver import OZsolver
    from pysasfit.oz.picardOZsolver import PicardOZsolver

    f = {s.key: s.default for sec in prefs.sections() for s in sec.settings}
    for k, v in DensityProfileFit.DEFAULT_OPTIONS.items():
        assert f[f"sldprofile.opt.{k}"] == v, k
    for k, v in IFT_DEFAULTS.items():
        assert f[f"sldprofile.ift.{k}"] == v, k
    for k, v in OPTION_DEFAULTS.items():
        assert f[f"inversion.opt.{k}"] == v, k
    assert f["oz.solvers.max_iterations"] == 600 and f["oz.solvers.max_iterations_picard"] == 5000
    from pysasfit.sldprofile import geometry
    from pysasfit.core import background
    assert geometry.DEFAULTS == prefs.values("sldprofile.geo")
    assert background.START_DEFAULTS == prefs.values("numerics.background")
    import inspect
    from pysasfit.oz.polydisperse_fit import fitWithBumps, fitWithNLopt
    nl, bp = inspect.signature(fitWithNLopt).parameters, inspect.signature(fitWithBumps).parameters
    assert (nl["maxNfev"].default, nl["ftol"].default) == (f["numerics.nlopt.max_nfev"], f["numerics.nlopt.ftol"])
    assert (bp["pop"].default, bp["burn"].default, bp["samples"].default) == (
        f["numerics.bumps.pop"], f["numerics.bumps.burn"], f["numerics.bumps.samples"])
    assert OZsolver.CONVERGENCE_CRITERION == f["oz.solvers.convergence"]
    assert PicardOZsolver.DEFAULT_MAX_ITERATIONS == 5000


def test_sld_options_reach_the_fit(prefs_file):
    from pysasfit.sldprofile import DensityProfileFit, models

    f = lambda r: np.where(r < 4, 1.0, 0.0)  # noqa: E731
    q, I, s = models.simulate(np.linspace(0.1, 2, 60), f, 4.0, seed=1)[:3]
    a = DensityProfileFit(q, I, s, n=20)
    b = DensityProfileFit(q, I, s, n=20, options={"alpha_step": 1.0, "log_alpha_lo": -10.0})
    pa, pb = a.alpha_path(4.0), b.alpha_path(4.0)
    assert len(pb) < len(pa)
    assert np.allclose(np.diff(np.log([s_.alpha for s_ in pb])), -1.0)
    with pytest.raises(ValueError):
        DensityProfileFit(q, I, s, options={"no_such": 1})


def test_inversion_lambda_grid_scale():
    """The trace-scaled grid finds the evidence maximum where the median rule fails."""
    from pysasfit.inversion import solver_registry as sr
    from pysasfit.sldprofile import models
    from pysasfit.sldprofile.twostep import ift_kernel

    f = lambda r: np.where(r < 3, 1.0, np.where(r < 5, 0.4, 0.0))  # noqa: E731
    q, I, s = models.simulate(np.linspace(0.1, 2.5, 120), f, 5.0, seed=2)[:3]
    A = ift_kernel(q, 10 * np.arange(1, 61) / 61, "sphere")
    out = {}
    try:
        for sc in ("median", "trace"):
            sr.OPTIONS["evidence_lambda_scale"] = sc
            res = sr.run_solver("bayesian_evidence", A, I, s, penalty="hansen_eq19")
            out[sc] = res
    finally:
        sr.OPTIONS.update(sr.OPTION_DEFAULTS)
    lz = {k: v.diagnostics["log_evidence_normalized"] for k, v in out.items()}
    assert lz["trace"] > lz["median"] + 100
    assert np.abs(out["trace"].x).max() < 0.1 < np.abs(out["median"].x).max()


def test_oz_solver_class_settings(prefs_file):
    from pysasfit.oz.oZsolver import OZsolver
    from pysasfit.oz.picardOZsolver import PicardOZsolver
    from pysasfit.oz.preferences import apply_solver_settings

    try:
        prefs.set_value("oz.solvers.max_iterations_picard", 1234)
        prefs.set_value("oz.solvers.convergence", 1e-9)
        apply_solver_settings()
        sol = PicardOZsolver(port=0, numberOfRadialSamplingPoints=255, hardSphereDiameterInPoints=20)
        assert sol.numberOfIterations == 1234 and sol.convergenceCriterion == 1e-9
    finally:
        prefs.reset()
        apply_solver_settings()
    assert PicardOZsolver.DEFAULT_MAX_ITERATIONS == 5000 and OZsolver.CONVERGENCE_CRITERION == 1e-12


def test_polydisperse_fit_uses_compute_grid(monkeypatch):
    """The fit builds its models with the tab's numerical settings (it used the defaults)."""
    from pysasfit.oz import polydisperse_fit as pf

    seen = {}

    class Fake:
        def __init__(self, *a, **kw):
            seen.update(kw)
            raise RuntimeError("stop")

    monkeypatch.setattr(pf, "GenericPolydisperseSAS", Fake, raising=False)
    import pysasfit.oz.generic_polydisperse_sas as g
    monkeypatch.setattr(g, "GenericPolydisperseSAS", Fake)
    Q = np.linspace(0.01, 0.2, 20)
    fit = pf.PolydisperseFit(Q, np.ones_like(Q), parameters={"phi": (0.1, 0.01, 0.4)},
                             fixed={"srel": 0.1, "meanRadius": 50.0},
                             sasOptions={"gridN": 2047, "pointsPerSigma": 50, "transformType": 1})
    fit._modelShape(np.array([0.1]))
    assert seen.get("gridN") == 2047 and seen.get("pointsPerSigma") == 50


def test_every_setting_is_explained():
    """Every registered setting has an explanation (settings_help.HELP), and
    the dialog/docs use it."""
    from pysasfit.core import settings as p
    from pysasfit.core.settings_help import HELP
    keys = [s.key for sec in p.sections() for s in sec.settings]
    assert set(keys) == set(HELP), (set(keys) ^ set(HELP))
    for k in keys:
        assert len(HELP[k].strip()) >= 40, k
        assert p.find(k).help == HELP[k]


def test_docs_list_every_setting():
    import pathlib
    from pysasfit.core import settings as p
    doc = (pathlib.Path(__file__).parents[2] / "docs" / "preferences.md").read_text(encoding="utf-8")
    for sec in p.sections():
        for s in sec.settings:
            assert f"`{s.key}`" in doc, s.key
