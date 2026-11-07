import numpy as np, pysasfit.scatter as hs
from pysasfit.scatter.blocks import BLOCK

def test_lens_equals_ellipsoid():
    el = hs.Ellipsoid(20., 2.0, 0.1)
    ld = hs.LensDumbbell(20., 2.0, 0.0, 2.0, 0.1)
    q = np.array([0.01, 0.05, 0.12])
    assert np.allclose(ld.P(q), el.P(q), rtol=3e-4)
    assert np.allclose(ld.F(q), el.F(q), rtol=3e-4)

def test_lens_dumbbell_run():
    q = np.array([0.02, 0.1])
    for o in (hs.Lens(20., 1.5, -0.4, 2.0, 0.1), hs.Dumbbell(20., 1.0, 0.5, 2.0, 0.1)):
        v = o.P(q)
        assert np.all((v > 0) & (v <= 1.0001)), v

def test_superball_p2_is_sphere():
    sb = hs.Superball(20., 2.0, 1.0, 1.0, 0.1, xmax=12.0, dx=0.1)
    sp = hs.Sphere(20., 0.1)
    q = np.array([0.01, 0.1, 0.2, 0.3])
    assert np.allclose(sb.P(q), sp.P(q), rtol=2e-3), (sb.P(q), sp.P(q))
    assert np.allclose(sb.F(q), sp.F(q), rtol=2e-3)
