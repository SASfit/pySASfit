"""Shortcuts of the r_max scan (abandoned alpha paths, limited fine grid,
parallel processes) must not change the result."""
import numpy as np

from pysasfit.sldprofile import DensityProfileFit, models


def _data():
    q = np.linspace(0.12, 1.0, 50)
    return models.simulate(q, models.homogeneous, 7.5, rel_noise=0.05, abs_noise=0.0, seed=1)[:3]


def test_shortcuts_keep_result():
    q, I, s = _data()
    kw = dict(r_max=7.5, r_max_range=(4.0, 11.0), n_coarse=8, n_fine=7)
    off = dict(abandon_plateau=0.0, abandon_margin=0.0, fine_support=0.0)
    a = DensityProfileFit(q, I, s, n=30, options=off).fit(**kw)
    b = DensityProfileFit(q, I, s, n=30).fit(**kw)
    assert abs(a.best.r_max - b.best.r_max) < 0.1
    assert abs(a.Rg[0] - b.Rg[0]) < 0.02
    assert abs(a.r_max[0] - b.r_max[0]) < 0.5 * max(a.r_max[1], 0.05)


def test_hopeless_r_max_path_is_short():
    """Star data at r_max = 200 (best ~400): chi2 stalls at ~250 M; the path ends
    early with the same evidence maximum."""
    import os

    from pysasfit.sldprofile import load_data

    q, I, s, _ = load_data(os.path.join(os.path.dirname(__file__), "..", "..", "examples", "sldprofile", "data",
                                        "star_p4over3_bg0.05.dat"))
    kw = dict(n=20, basis=4 / 3, background=True)
    short = DensityProfileFit(q, I, s, **kw).alpha_path(200.0)
    full = DensityProfileFit(q, I, s, **kw, options=dict(abandon_plateau=0.0)).alpha_path(200.0)
    assert len(short) < 0.7 * len(full)
    assert max(t.log_evidence for t in short) == max(t.log_evidence for t in full)


def test_parallel_scan_equals_serial():
    q, I, s = _data()
    rm = np.linspace(6.5, 8.5, 5)
    a = DensityProfileFit(q, I, s, n=20).scan_r_max(rm)
    b = DensityProfileFit(q, I, s, n=20, options=dict(n_workers=2)).scan_r_max(rm)
    assert np.allclose(a[1], b[1], rtol=1e-6)
    assert [x.log_evidence for x in a[2]] == [x.log_evidence for x in b[2]]
