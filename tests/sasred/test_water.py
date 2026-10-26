"""Water at two detector distances: K(2θ) of the water and the angle-independent ratio."""
import os
from types import SimpleNamespace

import numpy as np
import pytest

from pysasfit.sasred.core import water as wt
from pysasfit.sasred.core.geometry import pixel_geometry

RAW = os.environ.get("PYSASRED_TEST_RAW", "/mnt/user-data/uploads/Sept2026/raw")
need = pytest.mark.skipif(not os.path.isdir(RAW), reason="Vifor raw data not available")


def test_ratio_gives_k_and_scale_independent_of_efficiency():
    rng = np.random.default_rng(1)
    shape, c = (128, 128), (61.7, 64.0)
    ga = pixel_geometry(shape, (0.0075, 0.0075), c, 8.0, 6.0)
    gb = pixel_geometry(shape, (0.0075, 0.0075), c, 2.0, 6.0)
    eta = np.outer(1 + 0.1 * rng.standard_normal(128), 1 + 0.1 * rng.standard_normal(128))
    a, C = -0.3, 1.04
    K = lambda x: np.exp(a * x**2)
    Ia = 100 * eta * K(ga.two_theta) * C
    Ib = 100 * eta * K(gb.two_theta)
    sa, sb = 0.01 * Ia, 0.01 * Ib
    Ia = Ia * (1 + 0.01 * rng.standard_normal(shape))
    Ib = Ib * (1 + 0.01 * rng.standard_normal(shape))
    valid = np.ones(shape, bool)
    valid[58:70, 56:68] = False
    r = wt.compare_water(Ia, sa, ga.two_theta, Ib, sb, gb.two_theta, valid, 8.0, 2.0)
    assert r.scale == pytest.approx(C, rel=1e-3) and r.coeffs[0] == pytest.approx(a, abs=0.02)
    assert 0.8 < r.chi2_ndf < 1.2 and r.K(np.radians(10)) == pytest.approx(K(np.radians(10)), abs=1e-3)
    assert r.K_sigma(np.radians(10)) < 1e-3
    assert "distance offset" in r.summary()


def test_distance_offset_and_deadtime_inversions():
    d = 0.05
    C = ((2 + d) / 2)**2 / ((8 + d) / 8)**2
    assert wt.distance_offset(C, 8.0, 2.0) == pytest.approx(d, abs=1e-6)
    tau0, tau, Ra, Rb = 0.66e-6, 1.5e-6, 3000.0, 38000.0
    C = ((1 - Rb * tau0) / (1 - Rb * tau)) / ((1 - Ra * tau0) / (1 - Ra * tau))
    assert wt.deadtime_for_unit_scale(C, Ra, Rb, tau0) == pytest.approx(tau, rel=1e-6)


def test_aborted_timer_run():
    from pysasfit.sasred.io.sinq import RawRun
    r = SimpleNamespace(count_mode="timer", preset=1200.0, counting_time_s=22.0, monitors={"monitor1": 1.0})
    assert RawRun.aborted.fget(r)
    r.counting_time_s = 1200.0
    assert not RawRun.aborted.fget(r)


@need
def test_vifor_water_8m_2m():
    from pysasfit.sasred.io.sinq import read_sans_i

    def run(n):                                       # as recorded, without the distance offset
        return read_sans_i(os.path.join(RAW, f"sans2026n{n:06d}.hdf"), checksum=False, distance_offset_m=0.0)
    assert run(82279).aborted and run(82294).aborted and not run(82280).aborted
    r = wt.compare_water_runs([run(n) for n in (82280, 82282, 82284)], [run(n) for n in (82281, 82283)],
                              [run(82203), run(82251)], [run(82204), run(82252)], (61.73, 63.96),
                              (0.2399, 0.0008), (0.8835, 0.0019))
    assert 1.03 < r.scale < 1.05 and 0.98 < r.K(np.radians(10)) < 0.997
    assert 0.03 < r.distance_offset_m < 0.08 and 1.3e-6 < r.deadtime_tau_s < 2.2e-6
    assert r.chi2_ndf < 1.5 and r.n_pixels > 14000


def test_distance_offset_preference(monkeypatch):
    from pysasfit.core import settings as prefs
    from pysasfit.sasred.io import sinq
    vals = {"sasred.detector.distance_offset_mm": 53.0, "sasred.detector.distance_offset_before": "2027-01-01"}
    monkeypatch.setattr(prefs, "effective", lambda k: vals[k])
    assert sinq.distance_offset_for("2026-09-30 20:14:32") == pytest.approx(0.053)
    assert sinq.distance_offset_for("2027-03-01 08:00:00") == 0.0
    vals["sasred.detector.distance_offset_before"] = ""
    assert sinq.distance_offset_for("2027-03-01 08:00:00") == pytest.approx(0.053)


@need
def test_vifor_water_with_distance_offset():
    """With +53 mm on both distances the 8 m and 2 m water agree per steradian."""
    from pysasfit.sasred.io.sinq import read_sans_i
    from pysasfit.sasred.reduce import reduce_run_1d
    from pysasfit.sasred.core.averaging import q_edges

    def run(n, off=0.053):
        return read_sans_i(os.path.join(RAW, f"sans2026n{n:06d}.hdf"), checksum=False, distance_offset_m=off)
    w2 = run(82203)
    assert w2.L2_effective_m == pytest.approx(w2.L2_m + 0.053) and w2.L2_offset_m == 0.053
    r = wt.compare_water_runs([run(n) for n in (82280, 82282, 82284)], [run(n) for n in (82281, 82283)],
                              [w2, run(82251)], [run(82204), run(82252)], (61.73, 63.96),
                              (0.2399, 0.0008), (0.8835, 0.0019))
    assert abs(r.scale - 1) < 0.005 and abs(r.distance_offset_m) < 0.01
    red = reduce_run_1d(w2, (61.73, 63.96), q_edges(0.03, 0.2, 10))
    assert any("offset" in n for n in red.notes)


def _images(Ls, delta, a, c=0.0, seed=3):
    rng = np.random.default_rng(seed)
    shape, cen = (128, 128), (61.7, 64.0)
    eta = np.outer(1 + 0.1 * rng.standard_normal(128), 1 + 0.1 * rng.standard_normal(128))
    out = []
    for k, L in enumerate(Ls):
        g = pixel_geometry(shape, (0.0075, 0.0075), cen, L, 6.0)                      # recorded L
        I = 100 * eta * np.exp(a * g.two_theta**2) * ((L / (L + delta))**2) * (np.exp(c) if k else 1.0)
        I = I * (1 + 0.005 * rng.standard_normal(shape))
        m = np.zeros(shape, np.uint32)
        m[58:70, 56:68] = 2
        out.append((I, 0.005 * I, m, g.two_theta, L))
    return out


def test_fit_offset_two_and_three_distances():
    f = wt.fit_offset_images(_images([8.0, 2.0], 0.053, -0.3))
    assert f.delta_m == pytest.approx(0.053, abs=0.002) and f.a == pytest.approx(-0.3, abs=0.02)
    assert 0.8 < f.chi2_ndf < 1.2 and f.sigma_m < 0.002
    # three distances: δ even with a free common constant (e.g. another water filling)
    f3 = wt.fit_offset_images(_images([8.0, 4.5, 2.0], 0.04, -0.2, c=0.01), free_constant=True)
    assert f3.delta_m == pytest.approx(0.04, abs=0.004) and f3.constant == pytest.approx(0.01, abs=0.003)
    assert "δ =" in f3.summary()
    with pytest.raises(ValueError):
        wt.fit_distance_offset([{"water": None}] * 2, 0.24, 0.88, free_constant=True)


@need
def test_vifor_offset_from_water():
    from pysasfit.sasred.io.sinq import read_sans_i

    def run(n):
        return read_sans_i(os.path.join(RAW, f"sans2026n{n:06d}.hdf"), checksum=False, distance_offset_m=0.0)
    c = (61.728, 63.931)
    sets = [dict(water=[run(n) for n in (82280, 82282, 82284)], empty=[run(n) for n in (82281, 82283)], centre=c),
            dict(water=[run(82203), run(82251)], empty=[run(82204), run(82252)], centre=c)]
    f = wt.fit_distance_offset(sets, (0.2399, 0.0008), (0.8835, 0.0019), dome=False)
    assert 0.045 < f.delta_m < 0.062
