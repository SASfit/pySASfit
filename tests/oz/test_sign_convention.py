"""Project-wide sign convention A: well depths and Yukawa amplitudes are
POSITIVE FOR ATTRACTION, in every potential and every independent route."""
import numpy as np
import pytest

from pysasfit.oz import ozLib
from pysasfit.oz import sign_convention as sc
from pysasfit.oz.picardOZsolver import PicardOZsolver


def _solver():
    return PicardOZsolver(port=0, numberOfRadialSamplingPoints=4095, hardSphereDiameterInPoints=100)


def _outside(sol):
    r = sol.getrArray()
    return r > sol.getHardSphereDiameter() * 1.0001


def test_square_well_positive_epsilon_is_a_well():
    sol = _solver()
    sol.setSquareWellPotential(1.0, 0.2)
    r, EN = sol.getrArray(), sol.boltzmannOfP2Ppotential
    inWell = (r > 1.0001) & (r < 1.19)
    assert np.allclose(EN[inWell], np.e)                    # exp(-beta u) = e^{+1}: attraction
    sol.setSquareWellPotential(-1.0, 0.2)
    assert np.allclose(sol.boltzmannOfP2Ppotential[inWell], np.exp(-1.0))


def test_piecewise_constant_steps_are_depths():
    sol = _solver()
    sol.setPiecewiseConstantHSPotential(-1.0, 0.1, 2.0, 0.1, 0.0, 0.1)   # shoulder, then well
    r, EN = sol.getrArray(), sol.boltzmannOfP2Ppotential
    assert np.allclose(EN[(r > 1.001) & (r < 1.099)], np.exp(-1.0))
    assert np.allclose(EN[(r > 1.101) & (r < 1.199)], np.exp(2.0))


def test_single_yukawa_matches_double_yukawa_sign():
    """Yukawa(lambda, K, HS) and HardSphereDoubleYukawa(K, 1/lambda, 0, 1) are
    the same potential -- the two used opposite signs in pyOZgui."""
    for K in (1.5, -1.5):
        a, b = _solver(), _solver()
        a.setYukawaPotential(0.5, K, True)
        b.setHardSphereDoubleYukawaPotential(K, 2.0, 0.0, 1.0)
        m = _outside(a)
        np.testing.assert_allclose(a.boltzmannOfP2Ppotential[m], b.boltzmannOfP2Ppotential[m],
                                   rtol=1e-12)
        #K > 0: exp(-beta u) > 1 outside the core
        assert (a.boltzmannOfP2Ppotential[m][0] > 1.0) == (K > 0)
        #the repulsive/attractive split follows the sign
        assert np.all(a.attractivePartOfP2Ppotential <= 0) and np.all(a.repulsivePartOfP2Ppotential >= 0)
        if K > 0:
            assert not np.any(a.repulsivePartOfP2Ppotential[m])
        else:
            assert not np.any(a.attractivePartOfP2Ppotential[m])


def test_attraction_raises_low_q_structure_factor():
    S = {}
    for eps in (0.5, 0.0, -0.5):
        res = ozLib.solve("SquareWell", phi=0.15, potentialArgs=(eps, 0.3), closure="Percus-Yevick",
                          verify=False)
        S[eps] = float(np.asarray(res.Sq)[0])
    assert S[0.5] > S[0.0] > S[-0.5]


def test_rfa_and_orpa_use_the_same_sign():
    """The C step-potential solvers take the same depth as setSquareWellPotential."""
    from pysasfit.oz import orpaWrapper, rfaWrapper

    rfaWrapper._load_lib(); orpaWrapper._load_lib()
    if rfaWrapper._lib is None or orpaWrapper._lib is None:
        pytest.skip("RFA/ORPA libraries not built")
    phi, eps, d = 0.15, 0.5, 0.3
    res = ozLib.solve("SquareWell", phi=phi, potentialArgs=(eps, d), closure="Percus-Yevick", verify=False)
    q = np.linspace(0.5, 15, 60)
    Sn = np.interp(q, np.asarray(res.q), np.asarray(res.Sq))
    Sr, ok = rfaWrapper.rfa1_compute(phi, [1 + d], [eps], q)
    assert ok and np.max(abs(Sr - Sn)) < 0.05
    So = orpaWrapper.orpa_compute(phi, 1 + d, eps, q)
    So = np.asarray(So[0] if isinstance(So, tuple) else So)
    assert np.max(abs(So - Sn)) < 0.1


def test_old_generic_session_is_converted():
    state = {"entries": {"potentialVar": "SquareWell"}, "potParamVars": ["-0.7", "4.0"],
             "fitBounds": {"pot0": ["-2", "0.5"], "phi": ["0", "0.5"]},
             "fit": {"parameters": {"pot0": -0.7, "phi": 0.2}}}
    note = sc.convert_generic_session(state)
    assert "SquareWell" in note and state["signConvention"] == "A"
    assert state["potParamVars"] == ["0.7", "4.0"]
    assert state["fitBounds"]["pot0"] == ["-0.5", "2.0"] and state["fitBounds"]["phi"] == ["0", "0.5"]
    assert state["fit"]["parameters"] == {"pot0": 0.7, "phi": 0.2}
    assert sc.convert_generic_session(state) == ""           # idempotent
    other = {"entries": {"potentialVar": "HardSphereDoubleYukawa"}, "potParamVars": ["1", "2", "3", "4"]}
    assert sc.convert_generic_session(other) == "" and other["potParamVars"] == ["1", "2", "3", "4"]
    assert sc.from_pyozgui("Yukawa", (0.5, 2.0, True)) == (0.5, -2.0, True)
