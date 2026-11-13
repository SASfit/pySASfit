"""SasView models (sasmodels) as pySASstudio functions: SASfit convention, units, amplitude/volume, vector parameters,
structure factors, size distributions, fit, explorer. Skipped without the sasmodels package."""

import os

import numpy as np
import pytest

pytest.importorskip("sasmodels")

from pysasfit.studio import function, sasmodels_bridge as sb  # noqa: E402
from pysasfit.studio.model import ModelFunction, ScatteringContribution  # noqa: E402


@pytest.fixture(autouse=True)
def _angstrom():
    sb.set_length_unit("A")
    sb.register_all()
    yield
    sb.set_length_unit("nm")
    sb.register_all()


def _sphere(q, R, drho):
    x = q * R
    V = 4 / 3 * np.pi * R**3
    return 1e-4 * (drho * V * 3 * (np.sin(x) - x * np.cos(x)) / x**3) ** 2


def test_models_are_registered_with_their_parameters():
    names = sb.register_all()
    assert len(names) > 60 and "ff_sv_sphere" in names and "sq_sv_hardsphere" in names
    f = function("ff_sv_cylinder")
    assert f.parameter_names == ["sld", "sld_solvent", "radius", "length"]  # no theta, phi (2D only)
    assert "Å" in f.info.parameters[2].description and f.info.label == "SasView: cylinder"


def test_sasfit_convention_and_units():
    q = np.geomspace(1e-3, 0.5, 40)
    f = function("ff_sv_sphere")
    P = f(q, sld=4, sld_solvent=1, radius=50)
    assert np.allclose(P, _sphere(q, 50, 3), rtol=2e-3)  # (Δρ V F)², not SasView's per-volume intensity
    # the same physics in nm: q × 10, lengths / 10, P per nm³ (V in nm³)
    sb.set_length_unit("nm")
    sb.register_all()
    f = function("ff_sv_sphere")
    assert f.info.defaults[2] == pytest.approx(5.0)
    assert np.allclose(f(q * 10, sld=4, sld_solvent=1, radius=5), P * 1e-3, rtol=1e-9)


def test_amplitude_and_volume():
    q = np.geomspace(1e-3, 0.2, 20)
    f = function("ff_sv_sphere")
    A = f.amplitude(q, sld=4, sld_solvent=1, radius=50)
    assert np.allclose(A**2, f(q, sld=4, sld_solvent=1, radius=50), rtol=1e-6)
    assert f.volume(30.0, sld=4, sld_solvent=1, radius=50, dist=2) == pytest.approx(4 / 3 * np.pi * 30**3)


def test_vector_parameters_become_numbered_ones():
    f = function("ff_sv_core_multi_shell")
    assert {"sld1", "thickness1", "sld10", "thickness10"} <= set(f.parameter_names)
    q = np.geomspace(1e-3, 0.3, 20)
    shell = f(q, sld_core=4, radius=40, sld_solvent=1, n=1, sld1=4, thickness1=10)  # a shell like the core
    assert np.allclose(shell, function("ff_sv_sphere")(q, sld=4, sld_solvent=1, radius=50), rtol=1e-6)


def test_structure_factor():
    S = function("sq_sv_hardsphere")
    phi = 0.2
    assert S(np.array([1e-5]), radius_effective=50, volfraction=phi)[0] == pytest.approx((1 - phi) ** 4 / (1 + 2 * phi) ** 2,
                                                                                         rel=1e-3)


def test_size_distribution_and_decoupling():
    from pysasfit.studio import register_python_function, unregister_python_function

    def gauss(x, p):
        N, s, mu = p[0], p[1], p[2]
        return N * np.exp(-0.5 * ((x - mu) / s) ** 2) / (np.sqrt(2 * np.pi) * s)

    register_python_function("sd_pygauss", gauss, parameters=("N", "s", "mu"), defaults=(1.0, 5.0, 50.0))
    try:
        q = np.geomspace(1e-3, 0.3, 30)
        c = ScatteringContribution("ff_sv_sphere", {"sld": 4, "sld_solvent": 1, "radius": 50}, size_distribution="sd_pygauss",
                                   sd_params={"N": 1e-6, "s": 5, "mu": 50}, distributed="radius", x_range=(20, 80))
        I = ModelFunction([c])(q)
        R = np.linspace(20, 80, 2001)
        w = gauss(R, [1e-6, 5, 50])
        ref = np.trapezoid(w[None, :] * _sphere(q[:, None], R[None, :], 3), R, axis=1)
        assert np.allclose(I, ref, rtol=1e-3)
        c2 = ScatteringContribution("ff_sv_sphere", {"sld": 4, "sld_solvent": 1, "radius": 50}, size_distribution="sd_pygauss",
                                    sd_params={"N": 1e-6, "s": 5, "mu": 50}, distributed="radius", x_range=(20, 80),
                                    structure_factor="sq_sv_hardsphere", sq_params={"radius_effective": 55, "volfraction": 0.2},
                                    approach="decoupling")  # fmt: skip
        Is = ModelFunction([c2])(q)
        assert Is[0] < I[0] and np.allclose(Is[-1], I[-1], rtol=0.05)  # S(0) < 1, S → 1 at large q
    finally:
        unregister_python_function("sd_pygauss")


def test_fit_a_sasview_model():
    from pysasfit.studio.fitting import FitProblem

    q = np.geomspace(5e-3, 0.3, 60)
    I = 1e-9 * _sphere(q, 60, 3) + 0.01
    s = 0.01 * I
    y = I + np.random.default_rng(0).normal(0, s)
    c = ScatteringContribution("ff_sv_sphere", {"sld": 4, "sld_solvent": 1, "radius": 45})
    p = FitProblem()
    p.add_dataset((q, y, s), ModelFunction([c], scales=[2e-9]))
    p.add_parameter("0.ff.radius", lower=10, upper=200)
    p.add_parameter("scale.0", lower=0)
    p.add_parameter("background")
    p.fit("scipy", vp=True)
    vals = {x.address: x.value for x in p.parameters}
    assert vals["0.ff.radius"] == pytest.approx(60, rel=0.01) and vals["scale.0"] == pytest.approx(1e-9, rel=0.03)


def test_explorer_with_a_sasview_model(tmp_path, monkeypatch):
    pytest.importorskip("PySide6")
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6 import QtWidgets

    monkeypatch.setenv("PYSASFIT_SETTINGS", str(tmp_path / "settings.json"))
    _app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    from pysasfit.studio.explorer import ModelExplorer

    e = ModelExplorer(ff="ff_sv_sphere")
    try:
        assert sb.length_unit() == "nm" and "ff_sv_sphere" in e._entries
        p = e.models()[0]
        assert p.funcs["ff"].name == "ff_sv_sphere"
        e._force_calc()
        assert e._last is not None and np.all(np.isfinite(e._last[1]))
        e.units.setCurrentIndex(e.units.findText("1/Å"))
        assert sb.length_unit() == "A" and "not converted" in e.status.text()
        code = e.python_code()
        assert "sasmodels_bridge.set_length_unit('A')" in code and "ff_sv_sphere" in code
        compile(code, "generated", "exec")
        e.sasview_action.setChecked(False)
        assert "ff_sv_sphere" not in e._entries
        e.sasview_action.setChecked(True)
        assert "ff_sv_sphere" in e._entries
    finally:
        e.close()


# --------------------------------------------------------------------------- SasView size distributions
DISTS = ["gaussian", "lognormal", "schulz", "rectangle", "uniform", "boltzmann"]


@pytest.mark.parametrize("name", DISTS)
def test_sasview_distributions_match_sasmodels_and_are_normalised(name):
    import sasmodels.weights as w

    f = function("sd_sv_" + name)
    assert f.parameter_names == ["N", "center", "width", "nsigmas"]
    d = w.MODELS[name](npts=201, width=0.15, nsigmas=sb.DISTRIBUTIONS[name][1])
    x, px = d.get_weights(50.0, 0.0, np.inf, relative=True)
    ours = f(x, N=1.0, center=50.0, width=0.15, nsigmas=0)
    keep = px > 1e-12 * px.max()
    ratio = ours[keep] / px[keep]
    assert np.allclose(ratio, ratio[0], rtol=1e-9)  # the same shape as sasmodels.weights
    lo, hi = sb._sd_support(name, 50.0, 7.5, sb.DISTRIBUTIONS[name][1])
    xx = np.linspace(lo, hi, 200001)
    assert np.trapezoid(f(xx, N=2.0, center=50.0, width=0.15, nsigmas=0), xx) == pytest.approx(2.0, rel=1e-6)


def test_sasview_polydispersity_in_the_studio_matches_sasview():
    from sasmodels.core import load_model
    from sasmodels.data import empty_data1D
    from sasmodels.direct_model import DirectModel

    q = np.geomspace(1e-3, 0.3, 40)
    c = ScatteringContribution("ff_sv_sphere", {"sld": 4, "sld_solvent": 1, "radius": 50}, size_distribution="sd_sv_gaussian",
                               sd_params={"N": 1, "center": 50, "width": 0.15, "nsigmas": 0}, distributed="radius")
    assert c.support() == pytest.approx((27.5, 72.5))  # the exact cut of the distribution
    I = ModelFunction([c])(q)
    Isv = DirectModel(empty_data1D(q), load_model("sphere"))(radius=50, sld=4, sld_solvent=1, scale=1, background=0,
                                                            radius_pd=0.15, radius_pd_type="gaussian", radius_pd_n=1001,
                                                            radius_pd_nsigma=3)  # fmt: skip
    r = I / Isv  # SASfit convention: N ⟨F²⟩ against SasView's ⟨F²⟩ / ⟨V⟩ -> a constant, the mean volume
    assert np.ptp(r) / r.mean() < 3e-4
    R = np.linspace(27.5, 72.5, 20001)
    wgt = np.exp(-((R - 50) ** 2) / (2 * 7.5**2))
    assert r.mean() == pytest.approx(np.sum(wgt * 4 / 3 * np.pi * R**3) / wgt.sum(), rel=3e-4)
