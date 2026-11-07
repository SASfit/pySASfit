"""pySCAtter models added to hypscat: worm-like chain, power-law shell, Teubner-Strey, structure factors."""
import numpy as np
import pytest
from scipy.special import sici

from pysasfit.scatter import params, engine
from pysasfit.scatter.models_ext import (KratkyPorodChain, PercusYevick, StickyHardSphere, FractalTeixeira,
                                         TeubnerStrey, powerlaw_shell_sphere)
from pysasfit.scatter.coreshell import CoreShellSphere

Q = np.array([1e-4, 0.01, 0.05, 0.1, 0.3])


def test_kratky_porod_limits():
    x = Q * 100
    rod = 2 * sici(x)[0] / x - (np.sin(x / 2) / (x / 2)) ** 2
    assert np.allclose(KratkyPorodChain(100, 2000).P(Q), rod, rtol=2e-3, atol=1e-4)     # stiff: rod
    u = Q ** 2 * 10000 * 2 / 6
    debye = 2 * (np.exp(-u) + u - 1) / u ** 2
    assert np.allclose(KratkyPorodChain(10000, 2).P(Q), debye, rtol=2e-3, atol=1e-4)    # flexible: Gaussian coil
    assert KratkyPorodChain(10000, 2).Rg ** 2 == pytest.approx(10000 * 2 / 6, rel=2e-3)


def test_kratky_porod_polydisperse_normalised():
    assert KratkyPorodChain(200, 10, sigma=0.3).P(np.array([1e-5]))[0] == pytest.approx(1, abs=1e-4)


@pytest.mark.parametrize("eta", [0.05, 0.2, 0.4])
def test_percus_yevick_compressibility_and_sticky_limit(eta):
    py = PercusYevick(5, eta)
    assert py.S(np.array([1e-6]))[0] == pytest.approx((1 - eta) ** 4 / (1 + 2 * eta) ** 2, rel=1e-6)
    q = np.linspace(1e-3, 3, 50)
    hs = StickyHardSphere(5 * (1 - 1e-7), eta, 1e9, 1e-7).S(q)           # tau -> inf: hard spheres
    assert np.allclose(hs, py.S(q), rtol=1e-4)


def test_sticky_spheres_raise_compressibility():
    hs = PercusYevick(5, 0.2).S(np.array([1e-4]))[0]
    assert StickyHardSphere(5, 0.2, 0.3).S(np.array([1e-4]))[0] > hs


def test_fractal_limits():
    f = FractalTeixeira(2.0, 2.0, 50.0)
    assert f.S(np.array([1e-9]))[0] == pytest.approx(1 + 2 * (50 / 2) ** 2, rel=1e-3)   # Γ(D+1) (ξ/r0)^D
    assert f.S(np.array([50.0]))[0] == pytest.approx(1, abs=1e-3)


def test_teubner_strey_peak():
    ts = TeubnerStrey(20, 30)
    q = np.linspace(0.01, 1, 2000)
    qmax = q[np.argmax(ts.P(q))]
    k, ik = 2 * np.pi / 20, 1 / 30
    assert qmax == pytest.approx(np.sqrt(k * k - ik * ik), rel=2e-3)
    assert ts.P(np.array([0.0]))[0] == pytest.approx(1)


def test_powerlaw_shell_homogeneous_limit():
    q = np.linspace(1e-3, 1, 40)
    a = powerlaw_shell_sphere(5, 15, 0.0, 1.0, 0.5).P(q)
    b = CoreShellSphere(15, [5 / 15, 1.0], [1.0, 0.5]).P(q)
    assert np.allclose(a, b, rtol=1e-8, atol=1e-12)


@pytest.mark.parametrize("extra", [
    dict(ComboBoxParticle=10, EditRadius=2, Length=200),
    dict(ComboBoxParticle=11, EditRadius=10, Length=20),
    dict(ComboBoxInterior=3, ShellN=4, ShellThick=1.5),
    dict(ComboBoxInterior=4, ShellN=3, ShellThick=2, EditRadiusi=4),
    dict(StructureFactor=1, SFRadius=5.5, SFEta=0.3),
    dict(StructureFactor=2, SFRadius=5.5, SFEta=0.2, SFTau=0.3),
    dict(StructureFactor=3, SFRadius=5, SFDim=2.1, SFXi=80),
])
def test_engine_new_options(extra):
    p = params.defaults()
    p.update(EditQmin=0.01, EditQmax=2, EditQsteps=40, GridPoints=24, EditRadius=5, Length=40, EditRho=0.5)
    p.update(extra)
    q, I = engine.calc_1d(p)
    im = engine.calc_2d(p)
    assert np.all(np.isfinite(I)) and np.all(np.isfinite(im)) and np.all(I > 0)
    assert np.allclose(im, im[::-1, ::-1], rtol=1e-6)


def test_ini_roundtrip_new_keys(tmp_path):
    p = params.defaults()
    p.update(StructureFactor=2, SFTau=0.7, ShellProfile=1, ProfileAlpha=-1.0, ShellN=5)
    f = tmp_path / "x.ini"
    params.write_ini(f, p, {"SFTau": True})
    p2, fit, _ = params.read_ini(f)
    assert (p2["StructureFactor"], p2["SFTau"], p2["ShellProfile"], p2["ShellN"]) == (2, 0.7, 1, 5)
    assert fit["SFTau"] and not fit.get("ShellN")
