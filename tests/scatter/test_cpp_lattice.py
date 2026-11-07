"""Lattice factor in the CrystalScatter form: peak-width modes, Debye-Waller factor, peak FWHM."""
import numpy as np
import pytest

from pysasfit.scatter import engine, params


def _p(**kw):
    p = params.defaults()
    p.update(LType=7, ComboBoxParticle=0, EditRadius=3, ComboBoxPeak=1, Ordis=7, uca=20.0, EditAzi=0.0,
             EditDebyeWaller=0.0, EditDomainSize=200.0, RadButDebyeScherrer=True, RadioButtonPara=False)
    p.update(kw)
    p.update(ucb=p["uca"], ucc=p["uca"])
    return p


def _fwhm(p, a):
    """FWHM of the first sc reflection (g = 2 pi / a) of the powder lattice factor."""
    c = engine.make_lattice(p)
    g = 2 * np.pi / a
    q = np.linspace(g * 0.85, g * 1.15, 40001)                # next sc reflection at sqrt(2) g
    Z = engine.cpp_lattice_factor(p, c, q, 0 * q, 0 * q)
    m = Z >= Z.max() / 2
    return q[m][-1] - q[m][0]


def test_width_modes():
    assert engine.peak_delta(_p(EditDomainSize=100))[0] == pytest.approx(0.04)
    d, c = engine.peak_delta(_p(RadioButtonPara=True, EditDebyeWaller=2.0, EditDomainSize=100), "fcc")
    assert d == pytest.approx(0.04) and c == pytest.approx(4.0 / (20 * np.sqrt(2) / 2))
    assert engine.peak_delta(_p(RadButDebyeScherrer=False, EditDomainSize=100))[0] == 1.0     # fixed width
    d, _ = engine.peak_delta(_p(ComboBoxPeak=7, SigX=40, SigY=40, SigZ=40))
    assert d == pytest.approx(np.sqrt(3) * 0.1 / 3)


def test_debye_waller():
    assert engine.debye_waller(_p(EditDebyeWaller=3.0), 1.0) == pytest.approx(np.exp(-3.0))
    assert engine.debye_waller(_p(), 5.0) == 1.0


@pytest.mark.parametrize("a", [10.0, 25.0])
@pytest.mark.parametrize("D", [150.0, 400.0])
def test_gaussian_fwhm_is_4_over_D_and_lattice_independent(a, D):
    p = _p(uca=a, EditDomainSize=D, ComboBoxPeak=1)            # Gaussian: FWHM = sqrt(pi ln 2) * 4/D
    assert _fwhm(p, a) == pytest.approx(np.sqrt(np.pi * np.log(2)) * 4 / D, rel=2e-3)


def test_lorentzian_fwhm():
    p = _p(EditDomainSize=200.0, ComboBoxPeak=0)
    assert _fwhm(p, 20.0) == pytest.approx(4 / 200.0, rel=1e-2)


def test_paracrystal_broadens_with_q():
    p = _p(RadioButtonPara=True, EditDebyeWaller=0.5, EditDomainSize=1000.0, ComboBoxPeak=1)
    d, c = engine.peak_delta(p, "sc")
    for h in (1, 3):
        g = 2 * np.pi * h / 20.0
        q = np.linspace(g - 0.02, g + 0.02, 20001)
        Z = engine.cpp_lattice_factor(p, engine.make_lattice(p), q, 0 * q, 0 * q)
        qq, zz = q, Z
        m = zz >= zz.max() / 2
        assert qq[m][-1] - qq[m][0] == pytest.approx(np.sqrt(np.pi * np.log(2)) * (d + c * g * g), rel=3e-2)
