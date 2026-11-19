"""Single-representation Schulz averages of the hypergeometric blocks (schulz_kernel, METHOD = "integral")."""
import numpy as np
import pytest
from scipy import special, stats

from pysasfit.scatter import blocks
from pysasfit.scatter import schulz_kernel as sk


def _direct(mono, y, z, s):
    """Schulz average of mono(y x) (weight x^(z+s) e^-(z+1)x) by composite Gauss-Legendre in x (positive or
    bounded integrands; reference)."""
    al, sc = z + s + 1, 1.0 / (z + 1)
    lo, hi = stats.gamma.ppf(1e-18, al, scale=sc), stats.gamma.isf(1e-18, al, scale=sc)
    gx, gw = np.polynomial.legendre.leggauss(20)
    out = []
    for yi in np.atleast_1d(y):
        e = np.linspace(lo, hi, int(max(50, (hi - lo) * yi / np.pi * 2)) + 1)
        e = np.unique(np.concatenate([e[1] * 0.5 ** np.arange(60, 0, -1), e]))
        c, h = 0.5 * (e[1:] + e[:-1]), 0.5 * (e[1:] - e[:-1])
        x, w = (c[:, None] + h[:, None] * gx).ravel(), (h[:, None] * gw).ravel()
        pdf = np.exp((al - 1) * np.log(x) - x / sc - special.gammaln(al) - al * np.log(sc))
        out.append(np.sum(w * pdf * mono(yi * x)))
    return np.array(out)


def _f(d, x):
    x = np.asarray(x, float)
    xs = np.where(x == 0, 1.0, x)
    if d == 1:
        return np.sinc(x / np.pi)
    if d == 2:
        return np.where(x < 1e-8, 1.0, 2 * special.j1(xs) / xs)
    return np.where(x < 1e-4, 1 - x * x / 10, 3 * (np.sin(xs) - xs * np.cos(xs)) / xs**3)


Y = np.array([0.01, 0.5, 3.0, 10.0, 30.0, 100.0, 300.0])


@pytest.mark.parametrize("z,s", [(3, 6), (24, 6), (99, 6), (3, 0), (0.5, 3)])
@pytest.mark.parametrize("d", [1, 2, 3])
def test_form_factor_average_against_direct(d, z, s):
    ref = _direct(lambda x: _f(d, x) ** 2, Y, z, s)
    np.testing.assert_allclose(sk.block_value(d, Y, z, s, "P"), ref, rtol=1e-9)


@pytest.mark.parametrize("z,s", [(3, 3), (24, 3), (0.5, 1)])
@pytest.mark.parametrize("d", [1, 2, 3])
def test_amplitude_average_against_direct(d, z, s):
    ref = _direct(lambda x: _f(d, x), Y, z, s)
    np.testing.assert_allclose(sk.block_value(d, Y, z, s, "F"), ref, rtol=1e-9, atol=1e-14)


@pytest.mark.parametrize("a,b", [(1.0, 0.7), (0.4, 1.0), (1.0, 1.0)])
def test_cross3_against_direct(a, b):
    for z, s in [(3, 6), (24, 6), (0.5, 3)]:
        ref = _direct(lambda x: _f(3, a * x) * _f(3, b * x), Y, z, s)
        np.testing.assert_allclose(sk.cross3_value(a, b, Y, z, s), ref, rtol=1e-9, atol=1e-15)


def test_line_integral_against_direct():
    mono = blocks._line_int_mono
    for z, s in [(3, 2), (24, 2)]:
        np.testing.assert_allclose(sk.line_integral_value(Y, z, s), _direct(mono, Y, z, s), rtol=1e-10)


def test_disc_kernel_fast_against_elliptic():
    t = np.concatenate([np.linspace(1e-6, 2, 2001), [0.4503, 0.8, 1.3]])
    np.testing.assert_allclose(sk._gd_p2(t), sk.g_p2(t) - 16 / (3 * np.pi**2), rtol=1e-12, atol=3e-15)


def test_y_zero_and_large_y():
    for d in (1, 2, 3):
        assert sk.block_value(d, np.array([0.0]), 3, 6, "P")[0] == pytest.approx(1.0, rel=1e-14)
        assert sk.block_value(d, np.array([0.0]), 3, 6, "F")[0] == pytest.approx(1.0, rel=1e-14)
    # Porod regime: <P_3> -> 4.5 <x^-4>/y^4 ... compare with the closed forms (exact for d = 3)
    y = np.array([1e3, 3e3])
    np.testing.assert_allclose(sk.block_value(3, y, 24, 6, "P"), blocks.BLOCK[3].closed(y, 24, 6, "P"), rtol=1e-9)


def test_broad_distributions_where_the_hybrid_fails():
    """z+s+1 small: the hybrid's closed forms are undefined (Gamma poles), its fallbacks were off by up to 100 %."""
    y = np.array([30.0, 100.0, 300.0])
    for d in (2, 3):
        ref = _direct(lambda x, d=d: _f(d, x) ** 2, y, 0.5, 3)
        np.testing.assert_allclose(blocks.BLOCK[d].value(y, 0.5, 3, "P"), ref, rtol=1e-8)


def test_y_cache_matches_direct():
    rng = np.random.default_rng(3)
    for z, s, d, kind in [(7, 0, 2, "P"), (24, 6, 3, "P"), (3, 3, 3, "F"), (399, 6, 2, "P"), (9999, 6, 1, "P")]:
        y = np.exp(rng.uniform(np.log(1e-3), np.log(2e3), 3000))
        sk.clear_cache()
        old = sk.CACHE
        try:
            sk.CACHE = False
            ref = sk.block_value(d, y, z, s, kind)
            sk.CACHE = True
            a = sk.block_value(d, y, z, s, kind)  # builds the tables
            b = sk.block_value(d, y, z, s, kind)  # uses them
        finally:
            sk.CACHE = old
        tol = 1e-11 * np.abs(ref) + 1e-13
        assert np.all(np.abs(a - ref) <= tol) and np.all(np.abs(b - ref) <= tol), (z, s, d, kind)


def test_numba_kernel_equals_numpy():
    from pysasfit.scatter import _accel

    if not _accel.HAVE_NUMBA:
        pytest.skip("numba not installed")
    y = np.concatenate([[0.0], np.logspace(-3, 4, 300)])
    old, oldc = sk.BACKEND, sk.CACHE
    sk.CACHE = False
    try:
        for z, s in [(3, 6), (99, 6), (0.5, 3)]:
            for kind, d in [("P", 1), ("P", 2), ("P", 3), ("F", 2), ("F", 3)]:
                sk.BACKEND = "numpy"
                a = sk.block_value(d, y, z, s, kind)
                sk.BACKEND = "auto"
                b = sk.block_value(d, y, z, s, kind)
                np.testing.assert_allclose(b, a, rtol=1e-8, atol=1e-14)
            for fn in (lambda z=z, s=s: sk.line_integral_value(y, z, s), lambda z=z, s=s: sk.cross3_value(1.0, 0.6, y, z, s)):
                sk.BACKEND = "numpy"
                a = fn()
                sk.BACKEND = "auto"
                np.testing.assert_allclose(fn(), a, rtol=1e-8, atol=1e-14)
    finally:
        sk.BACKEND, sk.CACHE = old, oldc
