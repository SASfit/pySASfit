"""Cross-check against Python ports of formulas in CrystalScatter (JCNS-1, Foerster/Wagener), the authors'
C++ implementation: sphere series + closed-form asymptotics, core/shell sphere closed form, cube Porod level.
Uses the default 'reference' weighting (s=0), which is what the C++ code and the SI Mathematica code do."""
import numpy as np, pytest
import pysasfit.scatter as hs
from pysasfit.scatter.blocks import BLOCK, z_from_sigma
import scatter_ref_cpp_sphere as rs
import scatter_ref_cpp_coreshell as rc
from scatter_testutil import gam_w

pytestmark = pytest.mark.reference


def test_sphere_series_and_asymptotics():
    R, sg = 50., 0.1
    q = np.array([0.001, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5])
    cpp, lim = rs.ref_P(q, sg, R)
    fixed, _ = rs.ref_P(q, sg, R, buggy=False)
    mine = hs.Sphere(R, sg).P(q)
    assert np.allclose(mine[:5], cpp[:5], rtol=2e-5)           # series region of the C++ code
    assert np.allclose(mine, fixed, rtol=1e-5)                 # with the pq3 typo corrected
    assert abs(cpp[5] / mine[5] - 1) > 5e-4                    # documented typo in the C++ asymptotics (/2.0)


def test_coreshell_sphere():
    sg, Rout, p1, rho = 0.1, 40., 0.6, 0.5
    q = np.array([0.05, 0.1, 0.2, 0.4, 0.8])
    cpp = rc.cpp_cs(q, sg, Rout, p1, rho)
    cs = hs.CoreShellSphere(Rout, [p1, 1.0], [1.0, rho], sg)
    assert cs.weighting == "reference"
    mine = cs.P(q)
    r, w = gam_w(sg, 0)
    c = np.array([(1 - rho) * p1 ** 3, rho])
    br = np.array([np.trapezoid(((c[0] * BLOCK[3].mono(qq * Rout * r * p1, 'F') + c[1] * BLOCK[3].mono(qq * Rout * r, 'F')) / c.sum()) ** 2 * w, r) for qq in q])
    assert np.allclose(mine, br, rtol=1e-6)
    assert np.allclose(cpp, mine, rtol=1e-2) and not np.allclose(cpp, mine, rtol=1e-3)   # C++ deviates up to 0.7 %
    fixed = rc.cpp_cs(q, sg, Rout, p1, rho, fix_ac9=True)      # ... because of the ac9 term (zr-3 instead of zr-5)
    assert np.allclose(fixed, mine, rtol=1e-10)


def test_cube_porod_level():
    R, sg = 10., 0.1
    z = z_from_sigma(sg)
    por = 2 * np.pi * (z + 1) ** 4 * 24 * R ** 2 / (z * (z - 1) * (z - 2) * (z - 3) * (8 * R ** 3) ** 2)
    q = np.linspace(1.5, 2.5, 120)
    assert abs(np.mean(hs.Cube(R, sg).P(q) * q ** 4) / por - 1) < 2e-4


def test_vitess_sphere_image():
    """CrystalScatter's documentation example (256x256 image: sphere R=16.7, sigma=0.0639, D=10, lambda=0.154,
    pixel 0.0025): end-to-end check of P(q) (s=0) and of the detector q-map (Eq. 1) on the central row,
    column and diagonal."""
    import os
    d = np.loadtxt(os.path.join(os.path.dirname(__file__), "data", "vitess_sphere_slices.csv"), delimiter=",")
    ix = d[:, 0]
    sp = hs.Sphere(16.7, 0.0639)
    assert sp.weighting == "reference"
    for col, (fx, fy) in zip((1, 2, 3), ((1, 0), (0, 1), (1, 1))):
        qx, qy, qz = hs.detector.qmap(ix * 0.0025 * fx, ix * 0.0025 * fy, 10.0, 0.154)
        q = np.sqrt(qx ** 2 + qy ** 2 + qz ** 2)
        P = sp.P(q)
        rel = d[:, col] / P - 1
        assert np.abs(rel).max() < 4.5e-3 and np.abs(rel).std() < 1e-3
    sp.weighting = "volume"       # the SI-text weighting would be off by tens of percent
    q = np.sqrt(sum(c ** 2 for c in hs.detector.qmap(ix * 0.0025, 0 * ix, 10.0, 0.154)))
    assert np.abs(d[:, 1] / sp.P(q) - 1).max() > 0.05
