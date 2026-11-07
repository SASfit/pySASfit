import numpy as np
from pysasfit.scatter.coreshell import *
from pysasfit.scatter.blocks import BLOCK
from scatter_testutil import gam_w

def test_cs_sphere():
    R, sg = 25., 0.12
    p, rho = [0.6, 1.0], [2.0, 1.0]
    cs = CoreShellSphere(R, p, rho, sg)
    q = np.array([1e-4, 0.01, 0.05, 0.15, 0.4, 1.0])
    r, w = gam_w(sg, 6)
    c = np.array([(2-1)*0.6**3, 1.0*1.0])  # (rho1-rho2) p1^3, (rho2-0) p2^3
    ref = []
    for qq in q:
        A = (c[0]*BLOCK[3].mono(qq*R*r*0.6, 'F') + c[1]*BLOCK[3].mono(qq*R*r, 'F'))/c.sum()
        ref.append(np.trapezoid(A*A*w, r))
    a = cs.P(q)
    assert np.allclose(a, ref, rtol=1e-5, atol=1e-12), (a, ref)
    # series / closed form agree across regime boundary
    qq = np.linspace(0.001, 1.5, 400)
    assert np.all(np.isfinite(cs.P(qq)))
    # ellipsoid shell reduces to sphere at nu=1
    ce = CoreShellEllipsoid(R, 1.0, p, rho, sg)
    assert np.allclose(ce.P(q), a, rtol=1e-6)


def test_builders():
    import pysasfit.scatter as hs
    from test_shapes import gam_w
    q = np.array([0.003, 0.02, 0.06, 0.15])
    R, sg = 60., 0.1
    # constant gradient == homogeneous shell
    g = hs.gradient_shell_sphere(R, 0.5, 2.0, lambda t: 1.0, sg, n=6)
    ref = hs.CoreShellSphere(R, [0.5, 1.0], [2.0, 1.0], sg)
    assert np.allclose(g.P(q), ref.P(q), rtol=1e-8)
    # linear gradient converges with n
    a = hs.gradient_shell_sphere(R, 0.5, 2.0, lambda t: 1.0 - 0.8*t, sg, n=30).P(q)
    b = hs.gradient_shell_sphere(R, 0.5, 2.0, lambda t: 1.0 - 0.8*t, sg, n=120).P(q)
    assert np.allclose(a, b, rtol=3e-3, atol=1e-7)
    # myelin stack against brute-force size average
    m = hs.myelin_sphere(20., 3, 4., 3., 0.0, 1.0, sg); m.weighting = "reference"   # brute force below uses s=0
    Rout = 20 + 3*7.
    r, w = gam_w(sg, 0)
    from pysasfit.scatter.blocks import BLOCK
    # shells (outer radius, density): core 20 (0), then [in 3 (1), ex 4 (0)] x3
    edges = [20 + 7*i + 3 for i in range(3)]      # outer radius of the dense layers
    inner = [20 + 7*i for i in range(3)]
    def amp(qq):
        s = 0
        for lo, hi in zip(inner, edges):
            f = lambda x: 4*np.pi/3*x**3*BLOCK[3].mono(qq*x*r, "F")
            s = s + (f(hi) - f(lo))
        return s
    ref = np.array([np.trapezoid((amp(qq)/amp(1e-9)[...] if False else amp(qq))**2*w, r) for qq in q])
    ref = ref/ (4*np.pi/3*sum(h**3 - l**3 for l, h in zip(inner, edges)))**2
    assert np.allclose(m.P(q), ref, rtol=2e-6), (m.P(q), ref)
