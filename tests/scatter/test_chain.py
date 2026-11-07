import numpy as np
from pysasfit.scatter.chain import ExcludedVolumeChain

def debye(x): return 2*(np.exp(-x)+x-1)/x**2

def test_gauss_mono():
    c = ExcludedVolumeChain(1.0, nu=0.5, gamma=0.0, sigma=0.0)
    q = np.array([0.1, 1, 3, 8, 20])
    assert np.allclose(c.P(q), debye(q**2), rtol=1e-6)

def test_gauss_schulz_vs_brute():
    from scipy.special import gammaln
    sg = 0.2; z = (1-sg**2)/sg**2
    c = ExcludedVolumeChain(1.5, nu=0.5, gamma=0.0, sigma=sg)
    r = np.linspace(1e-3, 6, 200001)
    w = np.exp(z*np.log(r)-(z+1)*r); w /= np.trapezoid(w, r)
    q = np.array([0.05, 0.5, 2.0, 6.0])
    ref = [np.trapezoid(debye((qq*1.5*r)**2)*w, r) for qq in q]
    assert np.allclose(c.P(q), ref, rtol=1e-5)

def test_ev_continuity_and_asymptote():
    c = ExcludedVolumeChain(1.2, sigma=0.1)
    q = np.array([1e-4])
    assert abs(c.P(q)[0]-1) < 1e-4
    x = np.array([300., 450.])
    a = c.P(np.sqrt(x)/1.2)
    print(a, c._asym(x))
    assert abs(a[1]/c._asym(x)[1]-1) < 0.05 and a[0] > 0
