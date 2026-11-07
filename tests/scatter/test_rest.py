import numpy as np
import pysasfit.scatter as h

def test_orient_limits():
    sh = h.Ellipsoid(20., 3.0, 0.1)
    q = np.array([0.02, 0.08]); qx, qy = np.meshgrid(q, q)
    # isotropic distribution -> orientation average
    I = h.orient.pattern(sh, qx, qy, lambda d: np.ones_like(d))
    qq = np.hypot(qx, qy)
    assert np.allclose(I, sh.P(qq.ravel()).reshape(qq.shape), rtol=3e-3)

def test_lattice_and_dwba():
    sp = h.Sphere(20., 0.1)
    L = h.Lattice(h.cubic(60., 'sc'), width=0.004)
    g = 2*np.pi/60
    assert L.Z(g, 0, 0) > 50 and L.Z(0.5*g, 0, 0) < 1
    assert np.isfinite(L.I(sp, np.array([g]), np.array([0.]))).all()
    o = h.gisas.dwba(sp, 0, np.radians(0.4), np.radians(np.array([0.2, 0.5])),
                     np.radians(np.array([0.1, 0.1])), 1.0, 4e-6, 1e-8)
    assert np.all(o['P'] > 0)
    qx, qy, qz = h.detector.qmap(0.05, 0.1, 2.0, 1.0)
    assert abs(qx) < 1e-9 + abs(qy)
