import numpy as np
from scipy.special import gammaln
from pysasfit.scatter.shapes import *
from pysasfit.scatter.blocks import BLOCK

def gam_w(sigma, s, n=1500):
    z = (1-sigma**2)/sigma**2
    r = np.linspace(max(1e-3, 1-7*sigma), 1+7*sigma, n)
    w = np.exp((z+s)*np.log(r)-(z+1)*r); return r, w/np.trapezoid(w, r)

def brute(amp, q, R, sigma, s, kind, n_mu=1200):
    r, w = gam_w(sigma, s)
    mu = (np.arange(n_mu)+.5)/n_mu
    out = []
    for qq in q:
        A = amp(qq*R*r[None, :], mu[:, None], r[None, :]*0+1)
        A = A*A if kind == 'P' else A
        out.append(np.trapezoid(A*w, r, axis=1).mean())
    return np.array(out)

def test_ellipsoid_cyl_box():
    q = np.array([0.02, 0.1, 0.3, 0.6])/1.0
    R, sg = 20., 0.15
    # ellipsoid
    el = Ellipsoid(R, 2.5, sg)
    amp = lambda x, mu, _: BLOCK[3].mono(x*np.sqrt(1-mu**2+(2.5*mu)**2), 'F')
    for kind in 'PF':
        a = getattr(el, kind)(q); b = brute(amp, q, R, sg, 6 if kind=='P' else 3, kind)
        assert np.allclose(a, b, rtol=3e-4, atol=1e-9), (kind, a, b)
    cy = Cylinder(R, 3.0, sg)
    amp = lambda x, mu, _: BLOCK[2].mono(x*np.sqrt(1-mu**2), 'F')*BLOCK[1].mono(x*3.0*mu, 'F')
    for kind in 'PF':
        a = getattr(cy, kind)(q); b = brute(amp, q, R, sg, 6 if kind=='P' else 3, kind)
        assert np.allclose(a, b, rtol=3e-3, atol=1e-9), (kind, a, b)
    sp = Sphere(R, sg)
    assert np.allclose(sp.P(q), brute(lambda x, mu, _: BLOCK[3].mono(x, 'F')+0*mu, q, R, sg, 6, 'P', 2), rtol=3e-4)
    # monodisperse cube: P(0)=1, small-q Guinier consistent
    cu = Cube(R, 0.0)
    assert abs(cu.P(np.array([1e-6]))[0]-1) < 1e-6
    # triaxial with b=1 equals biaxial
    t = TriaxialEllipsoid(R, 1.0, 2.5, sg); assert np.allclose(t.P(q), el.P(q), rtol=1e-3)
