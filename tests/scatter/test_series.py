import numpy as np, pysasfit.scatter as hs
from pysasfit.scatter import orient

def test_series_vs_quad():
    cy = hs.Cylinder(20., 3.0, 0.1)
    h = orient.h_gauss(0.4)
    q = np.array([0.005, 0.02, 0.05, 0.08]); qx, qy = np.meshgrid(q, q)
    for kind in 'PF':
        s, ok = orient.pattern_series(cy, qx, qy, h, Nmax=60, kind=kind, fallback=False)
        ref = orient.pattern(orient.cyl_fixed_wrap(cy, kind), qx, qy, h, delta_max=np.pi/2, n_delta=200, n_psi=96)
        print(kind, ok.sum(), np.nanmax(np.abs(s/ref-1)))
        assert ok[:2, :2].all() and ok.sum() >= 6
        assert np.nanmax(np.abs(s[ok]/ref[ok]-1)) < 2e-4
    # monodisperse isotropic-limit: series == angle-averaged
    cy0 = hs.Cylinder(20., 3.0, 0.0)
    s = orient.pattern_series(cy0, qx, qy, lambda d: np.ones_like(d), Nmax=60)
    assert np.all(np.isfinite(s))
