"""Warm start of the r_max scan (option warm_start) gives the same result as the cold scan."""
import numpy as np

from pysasfit.sldprofile import DensityProfileFit, models


def test_warm_start_same_result():
    q = np.linspace(0.12, 1.0, 50)
    q, I, s = models.simulate(q, models.homogeneous, 7.5, rel_noise=0.05, abs_noise=0.0, seed=1)[:3]
    kw = dict(r_max=7.5, r_max_range=(5.0, 10.0), n_coarse=6, n_fine=5)
    a = DensityProfileFit(q, I, s, n=20).fit(**kw)
    b = DensityProfileFit(q, I, s, n=20, options=dict(warm_start=True)).fit(**kw)
    assert abs(a.best.r_max - b.best.r_max) < 1e-9
    assert abs(a.Rg[0] - b.Rg[0]) < 1e-3 * a.Rg[0]
    assert abs(a.best.log_evidence - b.best.log_evidence) < 0.1
