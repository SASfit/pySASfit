"""Fast paths of the 2D calculation give the same numbers as the reference code."""
import numpy as np
import pytest

from pysasfit.scatter import _accel, engine, params, shapes_ext
from pysasfit.scatter.shapes import Cube, Parallelepiped, SizeQuad


def test_sizequad_gauss_legendre_matches_dense():
    f = lambda xi, y: np.sinc(y * xi / np.pi) ** 2 * np.sinc(0.7 * y * xi / np.pi) ** 2   # noqa: E731
    for sig in (0.07, 0.3, 0.5):
        for y in (10.0, 45.0):
            a = SizeQuad(sig, 0, ymax=3.4 * y)
            b = SizeQuad(sig, 0, ymax=5 * 3.4 * y)
            assert (f(a.xi, y) * a.w).sum() == pytest.approx((f(b.xi, y) * b.w).sum(), rel=1e-9)


def test_lattice_factor_shell_grouping_against_direct_sum():
    p = params.defaults()
    p.update(LType=17, uca=30, ucb=30, ucc=42, ucalpha=94, ucbeta=94, HKLmax=4, Ordis=6, EditDomainSize=140,
             EditDebyeWaller=1, ComboBoxPeak=0, EditAzi=60)
    cc = engine._oriented_lattice(p, engine.make_lattice(p))
    rng = np.random.default_rng(1)
    Q = rng.uniform(-1.5, 1.5, (3, 400))
    Z = engine.cpp_lattice_factor(p, cc, *Q)
    delta, para = engine.peak_delta(p, cc.kind)
    q2 = (Q * Q).sum(0)
    q = np.sqrt(q2) + 1e-9
    ref = np.zeros(400)
    for G, wt in zip(cc.G.T, cc.weights):                       # the former per-reflection loop
        g = np.linalg.norm(G)
        sq = engine._cpp_shape(0, (q - g) ** 2 / delta ** 2, delta, 0)
        cosp = np.clip(G @ Q / (q * g), -1, 1)
        A = engine._cpp_azimuthal(0, np.arccos(cosp) ** 2, q2, 4 / 60, 0)
        ref += wt * sq * (2 * np.pi / g ** 2) * A / engine._cpp_norm(0, float(g), 4 / 60, 0.0)
    ref *= 2 * np.pi / cc.v0
    assert np.allclose(Z, ref, rtol=1e-10, atol=0)


@pytest.mark.skipif(not _accel.HAVE_NUMBA, reason="numba not installed")
def test_numba_kernels_equal_numpy(monkeypatch):
    q = np.linspace(1e-3, 2.5, 30)
    for mk in (lambda: Cube(8, 0.07), lambda: Parallelepiped(5, 1.5, 2.5, 0.2)):
        a = mk().P(q)
        monkeypatch.setattr(_accel, "HAVE_NUMBA", False)
        b = mk().P(q)
        monkeypatch.setattr(_accel, "HAVE_NUMBA", True)
        assert np.allclose(a, b, rtol=1e-12)
    from pysasfit.scatter.shapes_ext import Superball
    shapes_ext._TAB.clear()
    sb = Superball(1, 4.0, xmax=8.0)
    a = sb._build()
    shapes_ext._TAB.clear()
    monkeypatch.setattr(_accel, "HAVE_NUMBA", False)
    b = Superball(1, 4.0, xmax=8.0)._build()
    x = np.linspace(0, 7.9, 50)
    assert np.allclose(a[1](x), b[1](x), rtol=1e-11)
