"""Isotropic orientation (Ordis = 7): Debye-Scherrer rings in 2D, whatever EditAzi is, and a 1D curve equal to the
ring intensity (EditAzi applies to oriented lattices only)."""
import numpy as np
import pytest

from pysasfit.scatter import engine, params


def _p(**kw):
    p = params.defaults()
    p.update(LType=5, ComboBoxParticle=0, EditRadius=5.0, Length=20.0, ComboBoxPeak=1, Ordis=7, uca=30.0)
    p.update(kw)
    p.update(ucb=p["uca"], ucc=p["uca"])
    return p


_PHI = np.linspace(0, 2 * np.pi, 181)[:-1]
_G111 = 2 * np.pi / 30.0 * np.sqrt(3.0)


def _ring(p, g):
    return engine.intensity_2d(p, np.zeros_like(_PHI), g * np.cos(_PHI), g * np.sin(_PHI))


@pytest.mark.parametrize("peak", [0, 1, 2, 3, 4, 5])
@pytest.mark.parametrize("particle", [0, 1])
def test_isotropic_rings_independent_of_EditAzi(peak, particle):
    kw = dict(ComboBoxPeak=peak, ComboBoxParticle=particle, EditPeakPar=50.0 if peak == 4 else 2.0)
    ref = None
    for azi in (0.0, 20.0, 180.0):
        p = _p(EditAzi=azi, **kw)
        for g in (_G111, _G111 * 2 / np.sqrt(3), _G111 * np.sqrt(8.0 / 3)):          # 111, 200, 220
            I = _ring(p, g)
            assert I.max() == pytest.approx(I.min(), rel=1e-9)                         # ring
            assert engine.intensity_1d(p, np.array([g]))[0] == pytest.approx(I[0], rel=2e-3)      # 2D: tabulated P
        I111 = _ring(p, _G111)[0]
        ref = I111 if ref is None else ref
        assert I111 == pytest.approx(ref, rel=1e-12)


def test_1d_is_powder_for_oriented_settings():
    """The I(q) panel shows the powder average also when the 2D pattern is oriented."""
    q = np.array([_G111])
    assert engine.intensity_1d(_p(Ordis=0, EditAzi=180.0), q)[0] == \
        pytest.approx(engine.intensity_1d(_p(Ordis=7, EditAzi=0.0), q)[0], rel=1e-12)


def test_oriented_lattice_still_shows_spots():
    I = _ring(_p(Ordis=0, EditAzi=180.0), _G111 * np.sqrt(8.0 / 3))
    assert I.max() > 50 * I.min()
