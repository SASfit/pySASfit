import numpy as np, pysasfit.scatter as hs
from pysasfit.scatter import gisas

def test_powder_Z_vs_numeric_average():
    L = hs.Lattice(hs.cubic(60., 'fcc'), width=0.004)
    q = np.array([0.08, 0.11, 0.13, 0.2])
    # numerical spherical average of the 3D lattice factor
    mu, w = np.polynomial.legendre.leggauss(80); ph = np.linspace(0, 2*np.pi, 160, endpoint=False)
    ref = []
    for qq in q:
        M, P = np.meshgrid(mu, ph, indexing='ij')
        st = np.sqrt(1-M**2)
        Z = L.Z(qq*st*np.cos(P), qq*st*np.sin(P), qq*M)
        ref.append((Z*w[:, None]).sum()/2/ph.size)
    a = L.Z_powder(q)
    print(a, ref)
    assert np.allclose(a, ref, rtol=0.03)
    assert np.all(np.isfinite(L.I_powder(hs.Sphere(20., 0.1), q)))

def test_diffuse_analytic_vs_numeric():
    from scipy.integrate import quad
    from scipy.special import j0
    sig, xi, qz = 700., 7., 0.01
    qr = np.array([0.5, 3.0, 10.0])
    a = gisas.S_diffuse(qr, qz, sig, xi, 0.5)
    A = 2*sig**2/xi
    n = [2*np.pi/qz**2*quad(lambda R: np.exp(-qz**2*A*R/2)*j0(x*R)*R, 0, 40, limit=2000)[0] for x in qr]
    assert np.allclose(a, n, rtol=2e-3), (a, n)
    # general h path (h=0.5 via numeric integral) agrees as well
    b = gisas.S_diffuse(qr[:1], qz, sig, xi, 0.5 + 1e-9)
    assert np.allclose(b, a[:1], rtol=1e-2)

def test_total():
    sp = hs.Sphere(10., 0.08)
    L = hs.Lattice(hs.cubic(22., 'bcc'), width=0.01)
    r = gisas.gisas_intensity(sp, np.radians(0.15), np.radians(np.linspace(0.05, 0.6, 5)),
                              np.radians(np.linspace(-1, 1, 5)), 0.1, 1e-6, 1e-8, lattice=L,
                              sigma_z=1., xi=1e5, spec_widths=(1e-3, 1e-3))
    assert np.all(np.isfinite(r['I']))
