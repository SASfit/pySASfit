import numpy as np, pytest
import pysasfit.scatter as hs

def peak_ratios(kind, a=50., n=6, **kw):
    L = hs.Crystal(kind, a, **kw)
    g = np.linalg.norm(L.G, axis=0)
    u = np.unique(np.round(g/ (2*np.pi/a), 6))
    return u**2

def test_peak_sequences():
    # squared reflection orders (in units of (2pi/a)^2)
    assert np.allclose(peak_ratios('bcc')[:4], [2, 4, 6, 8])
    assert np.allclose(peak_ratios('fcc')[:4], [3, 4, 8, 11])
    assert np.allclose(peak_ratios('sc')[:4], [1, 2, 3, 4])
    assert np.allclose(peak_ratios('gyroid')[:6], [6, 8, 14, 16, 20, 22])
    assert np.allclose(peak_ratios('obdd')[:6], [2, 3, 4, 6, 8, 9])
    assert np.allclose(peak_ratios('plumbers')[:4], [2, 4, 6, 8])
    assert np.allclose(peak_ratios('fd3m')[:3], [3, 8, 11])

@pytest.mark.parametrize("kind,kw", [("fcc", {}), ("bcc", {}), ("hcp", {}), ("fd3m", {}), ("gyroid", {}),
                                      ("hexcyl", {}), ("squarecyl", {}), ("lamellae", {})])
def test_powder_mean_is_one(kind, kw):
    a = 50.
    L = hs.Crystal(kind, a, width=0.0015, hmax=(26 if kind == "hcp" else 14), **kw)
    q = np.linspace(1.0, 1.6, 4000)       # many peaks, high order
    Z = L.Z_powder(q)
    assert abs(Z.mean() - 1) < 0.08, (kind, Z.mean())

def test_powder_equals_numeric_average():
    L = hs.Crystal("hcp", 40., width=0.006)
    q = np.array([0.18, 0.21, 0.3])
    mu, w = np.polynomial.legendre.leggauss(160); ph = np.linspace(0, 2*np.pi, 320, endpoint=False)
    M, P = np.meshgrid(mu, ph, indexing='ij'); st = np.sqrt(1-M**2)
    ref = [((L.Z(qq*st*np.cos(P), qq*st*np.sin(P), qq*M)*w[:, None]).sum()/2/ph.size) for qq in q]
    assert np.allclose(L.Z_powder(q), ref, rtol=0.05)

def test_shapes_and_fiber():
    for sh in ("gaussian", "lorentzian", "pearson", "voigt"):
        L = hs.Crystal("fcc", 40., width=0.003, peak=sh)
        assert np.all(np.isfinite(L.Z_powder(np.array([0.2, 0.3]))))
    # normalisation of each 3D peak
    x = np.concatenate([[0.0], np.logspace(-7, 3, 2000001)])
    for sh in ("gaussian", "lorentzian", "pearson"):
        v = hs.peak(x, 0.004, sh, 3)
        assert abs(np.trapezoid(4*np.pi*x**2*v, x) - 1) < 5e-3, sh
    L = hs.Crystal("sc", 40., width=0.004)
    g = 2*np.pi/40
    assert L.Z_fiber(g, 0.0, 0.0, axis=(0, 0, 1), npsi=360) > 5     # (100) ring survives fibre average
    Rz = hs.zone_axis_rotation(L.M, (0, 0, 1), to=(1, 0, 0))
    assert np.allclose(Rz @ L.M[:, 2]/np.linalg.norm(L.M[:, 2]), [1, 0, 0])

def test_decoupling_with_crystal():
    sp = hs.Sphere(10., 0.08)
    L = hs.Crystal("fcc", 28., width=0.01)
    q = np.array([0.2, 0.3]); assert np.all(np.isfinite(L.I_powder(sp, q)))
