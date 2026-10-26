"""SANS-I detector model: ³He absorption, conversion depth, Al windows, angular efficiency."""
import numpy as np
import pytest

from pysasfit.sasred.core import detector_model as dm
from pysasfit.sasred.reduce import load_instrument


def test_construction_in_instrument_file():
    c = load_instrument()["detector"]["construction"]
    assert c["window"]["thickness_m"] == 0.007 and c["window"]["diameter_m"] == 1.3
    assert c["fill_gas"]["gas"] == "4He" and c["detection_gas"]["stopping_gas"] == "CF4"
    assert c["membrane"]["thickness_m"] == 0.002 and c["detection_gas"]["active_depth_m"] is None


def test_he3_numbers():
    # n = p / kT; Σ(1.798 Å) = n · 5333 b
    assert dm.number_density(2.7) == pytest.approx(6.675e25, rel=1e-3)
    assert dm.sigma_3he(1.798) * 2.7 == pytest.approx(6.675e25 * 5333e-28, rel=1e-3)
    assert dm.sigma_3he(12.0) / dm.sigma_3he(6.0) == pytest.approx(2.0)
    assert dm.mean_conversion_depth(6.0) == pytest.approx(1 / (dm.sigma_3he(6.0) * 2.7))
    assert dm.mean_conversion_depth(6.0, 0.03) < dm.mean_conversion_depth(6.0)
    assert dm.absorption_probability(6.0, 1.0) == pytest.approx(1.0)


def test_angular_efficiency_is_small():
    """Across the 2 m detector (ψ ≤ 17°) the efficiency changes by < 0.3 %: the 2 % drop of the
    water at 15° is the water itself (multiple incoherent scattering), not the detector."""
    for psi in (5, 10, 17):
        r = dm.efficiency(6.0, np.radians(psi)) / dm.efficiency(6.0, 0.0)
        assert 1.0 < r < 1.003
    with pytest.raises(ValueError):
        dm.al_transmission(4.0, 0.007)
    assert 0.95 < dm.al_transmission(6.0, 0.007) < 0.98


def test_dome_geometry():
    assert dm.dome_sag() == pytest.approx(0.93 - np.sqrt(0.93**2 - 0.65**2))
    assert dm.dome_crossing_angle(0.0, 0.0, 8.0) == pytest.approx(0.0, abs=1e-6)
    g2 = dm.dome_crossing_angle(0.48, 0.0, 2.0)
    assert g2 == pytest.approx(dm.dome_crossing_angle(0.0, -0.48, 2.0))          # axial symmetry
    assert g2 > dm.dome_crossing_angle(0.48, 0.0, 8.0) > 0                        # convex towards the sample
    # analytic: γ = α + 2θ with sin α = h / R at the crossing point
    x, L2 = 0.3, 2.0
    tt = np.arctan(x / L2)
    a = dm.dome_sag()
    zc = L2 - a + 0.93
    t = zc * np.cos(tt) - np.sqrt((zc * np.cos(tt))**2 - zc**2 + 0.93**2)
    alpha = np.arcsin(t * np.sin(tt) / 0.93)
    assert dm.dome_crossing_angle(x, 0.0, L2) == pytest.approx(alpha + tt, rel=1e-9)
    T = dm.dome_transmission(6.0, np.array([0.0, 0.24, 0.48]), 0.0, 2.0)
    assert T[0] == pytest.approx(1.0) and np.all(np.diff(T) < 0) and 0.98 < T[2] < 0.995


def test_construction_dome_radius():
    c = load_instrument()["detector"]["construction"]["window"]
    assert c["radius_of_curvature_m"] == 0.93 and c["apex_to_detector_m"] == 0.265


def test_membrane_flat_al():
    c = load_instrument()["detector"]["construction"]["membrane"]
    assert c["material"] == "Al" and c["shape"] == "flat"
    T = dm.window_transmission(6.0, 0.48, 0.0, 2.0)
    D = dm.dome_transmission(6.0, 0.48, 0.0, 2.0)
    assert 0.999 < T / D < 1.0                       # the flat membrane adds a little at the edge
