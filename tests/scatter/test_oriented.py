import numpy as np, pysasfit.scatter as hs
from pysasfit.scatter import oriented, orient

def test_box_perfect_vs_formula():
    bx = hs.Parallelepiped(20., 1.5, 0.7, 0.0)     # monodisperse: P = (sinc sinc sinc)^2
    qx, qy = np.meshgrid([0.02, 0.07], [0.03, 0.1])
    p = oriented.perfect(bx, qx, qy, 0.0)
    s = lambda x: np.sinc(x/np.pi)
    ref = (s(qx*20)*s(qy*30)*s(0.0*14))**2
    assert np.allclose(p, ref, rtol=1e-9)

def test_triaxial_axial_isotropic_equals_P():
    t = hs.TriaxialEllipsoid(20., 1.3, 2.0, 0.1)
    qx, qy = np.meshgrid([0.02, 0.06], [0.03, 0.09])
    q = np.hypot(qx, qy)
    I = oriented.axial(t, qx, qy, lambda d: np.ones_like(d), delta_max=np.pi, n_delta=48, n_psi=24, n_spin=16)
    assert np.allclose(I, t.P(q.ravel()).reshape(q.shape), rtol=5e-3)

def test_cylinder_axial_matches_orient_pattern():
    cy = hs.Cylinder(20., 3.0, 0.1)
    h = orient.h_gauss(0.4)
    qx, qy = np.meshgrid([0.01, 0.04, 0.08], [0.01, 0.05])
    a = oriented.axial(cy, qx, qy, h, director=(1, 0, 0), delta_max=np.pi/2, n_delta=96, n_psi=48)
    # orient.pattern puts the director along x with the third axis out of plane (same geometry)
    b = orient.pattern(cy, qx, qy, h, delta_max=np.pi/2, n_delta=96, n_psi=48)
    assert np.allclose(a, b, rtol=2e-3)

def test_octahedron_porod():
    # |x|+|y|+|z| <= R: V = 4/3 R^3, S = 4 sqrt(3) R^2 ; monodisperse mean q^4 P -> 2 pi S / V^2
    oc = hs.Octahedron(10., 0.02, xmax=24.0)
    x = np.linspace(14, 19, 300); q = x/10.
    mean = np.mean(oc.P(q)*q**4)
    por = 2*np.pi*(4*np.sqrt(3)*100)/((4/3*1000)**2)
    assert abs(mean/por - 1) < 0.06, (mean, por)


def test_superellipsoid_reduces_to_ellipsoid():
    se = hs.SuperEllipsoid(20., 1.0, 2.0, 2.0, 0.0, 0.1)
    el = hs.Ellipsoid(20., 2.0, 0.1)
    q = np.array([0.02, 0.08, 0.2])
    assert np.allclose(se.P(q), el.P(q), rtol=1e-3)
    # triaxial limit: b != 1
    se2 = hs.SuperEllipsoid(20., 1.4, 2.0, 2.0, 0.0, 0.1)
    tr = hs.TriaxialEllipsoid(20., 1.4, 2.0, 0.1)
    assert np.allclose(se2.P(q), tr.P(q), rtol=2e-3)

def test_buried_dwba():
    sp = hs.Sphere(10., 0.08)
    a = hs.gisas.dwba(sp, 0, np.radians(0.3), np.radians([0.2, 0.5]), np.radians([0.1, 0.1]), 0.1, 4e-6, 1e-8)
    b = hs.gisas.dwba(sp, 0, np.radians(0.3), np.radians([0.2, 0.5]), np.radians([0.1, 0.1]), 0.1, 4e-6, 1e-8, buried=True)
    assert np.all(np.isfinite(b['P'])) and not np.allclose(a['P'], b['P'], rtol=1e-6)
