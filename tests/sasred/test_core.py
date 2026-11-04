import numpy as np
import pytest

from pysasfit.sasred.core.averaging import q_edges, radial_binning
from pysasfit.sasred.core.beamcentre import masked_centre_of_gravity
from pysasfit.sasred.core.counting import counts_uarray, garwood_interval
from pysasfit.sasred.core.deadtime import nonparalysable
from pysasfit.sasred.core.geometry import pixel_geometry
from pysasfit.sasred.data.uarray import UArray


def test_garwood_reference_values():
    # values stated in the package definition
    _, _, s = garwood_interval(np.array([0, 1, 10, 1000]))
    np.testing.assert_allclose(s, [0.921, 1.563, 3.688, 32.125], atol=5e-4)


def test_garwood_coverage_is_exact():
    # the interval [lo, hi] must have P(N <= n | hi) = alpha and P(N >= n | lo) = alpha
    from scipy.stats import poisson
    from pysasfit.sasred.core.counting import ALPHA_1SIGMA
    n = np.arange(1, 50)
    lo, hi, _ = garwood_interval(n)
    np.testing.assert_allclose(poisson.cdf(n, hi), ALPHA_1SIGMA, rtol=1e-9)
    np.testing.assert_allclose(poisson.sf(n - 1, lo), ALPHA_1SIGMA, rtol=1e-9)


def test_negative_counts_rejected():
    with pytest.raises(ValueError):
        counts_uarray(np.array([[1, -1]]), "x")
    with pytest.raises(ValueError):
        counts_uarray(np.array([[1.5, 2.0]]), "x")


def test_deadtime_factor_and_tube_mode():
    N = np.full((4, 5), 100.0)
    c = counts_uarray(N, "c")
    y, f, flags = nonparalysable(c, 2.0, 1e-4)
    ft = 1.0 / (1 - 1e-4 * N.sum() / 2.0)
    np.testing.assert_allclose(y.value, N * ft)
    y0, f0, _ = nonparalysable(c, 2.0, 1e-4, axis=0)     # per column
    np.testing.assert_allclose(f0.value, 1.0 / (1 - 1e-4 * 400 / 2.0))
    assert not flags.any()


def test_solid_angle_untilted_and_tilted():
    shape, px, SD, lam = (64, 64), 0.0075, 2.0, 6.0
    g = pixel_geometry(shape, px, (31.3, 30.7), SD, lam)
    np.testing.assert_allclose(g.d_omega, px * px * np.cos(g.two_theta) ** 3 / SD ** 2, rtol=1e-12)
    np.testing.assert_allclose(g.q, 4 * np.pi / lam * np.sin(g.two_theta / 2), rtol=1e-12)
    np.testing.assert_allclose(np.hypot(np.hypot(g.qx, g.qy), g.qz), g.q, rtol=1e-12)
    # tilted: compare with brute-force integration of one pixel's solid angle
    a = 20.0
    gt = pixel_geometry(shape, px, (31.3, 30.7), SD, lam, alpha_deg=a)
    ar = np.deg2rad(a)
    for iy, ix in [(0, 0), (40, 63), (31, 31)]:
        s = np.linspace(-0.5, 0.5, 401)[:-1] + 1 / 800
        uu, vv = np.meshgrid((ix + s - 31.3) * px, (iy + s - 30.7) * px)
        r = np.stack([uu * np.cos(ar), vv, SD + uu * np.sin(ar)])
        d = np.linalg.norm(r, axis=0)
        n = np.array([-np.sin(ar), 0, np.cos(ar)])
        dom = ((np.tensordot(n, r, 1) / d) / d ** 2).mean() * px * px
        np.testing.assert_allclose(gt.d_omega[iy, ix], dom, rtol=2e-5)


def test_beam_centre_recovers_true_centre():
    rng = np.random.default_rng(7)
    yy, xx = np.mgrid[:128, :128]
    for x0, y0 in [(61.73, 63.93), (20.2, 100.6)]:
        lam = 5e4 * np.exp(-((xx - x0) ** 2 + (yy - y0) ** 2) / (2 * 1.3 ** 2))
        bc = masked_centre_of_gravity(rng.poisson(lam + 0.2))
        assert abs(bc.x - x0) < 0.1 and abs(bc.y - y0) < 0.1
        assert bc.sigma_x < 0.01


def test_radial_binning_is_a_mean():
    q = np.array([0.1, 0.15, 0.3, 0.35, 0.9])
    b = radial_binning(q, np.array([1, 1, 1, 0, 1], bool), np.array([0.0, 0.2, 0.4, 1.0]))
    np.testing.assert_allclose(b.A.toarray(), [[.5, .5, 0, 0, 0], [0, 0, 1, 0, 0], [0, 0, 0, 0, 1]])
    np.testing.assert_allclose(b.n_pixels, [2, 1, 1])


def test_polar_image_orientation_and_pixels():
    from pysasfit.sasred.core.polar import polar_image
    ny, nx = 40, 60
    yy, xx = np.mgrid[:ny, :nx].astype(float)
    xc, yc = 30.0, 20.0
    img = xx * 1000 + yy                           # encodes the source pixel
    pol = polar_image(img, (xc, yc), (0.005, 0.008), 2.0, 6.0, n_phi=360)
    ix, iy = np.divmod(pol.image, 1000)[0], np.divmod(pol.image, 1000)[1]
    r = 0.5 * (pol.r_edges_m[1:] + pol.r_edges_m[:-1])
    k = np.searchsorted(r, 0.05)                   # 5 cm from the centre
    # phi = 0 (first column, centre 0.5 deg): to the right (+x), same row
    assert ix[k, 0] == round(xc + r[k] / 0.005) and iy[k, 0] == yc
    # phi = 90 deg: up (+y), same column
    assert ix[k, 90] == xc and iy[k, 90] == round(yc + r[k] * np.sin(np.deg2rad(90.5)) / 0.008)
    # phi = 180 deg: to the left
    assert ix[k, 180] < xc and iy[k, 180] == yc
    assert np.isnan(pol.image[-1]).any()          # outside the detector
    q = 4 * np.pi / 6.0 * np.sin(0.5 * np.arctan(pol.r_edges_m / 2.0))
    np.testing.assert_allclose(pol.q_edges, q)
    inv = np.zeros((ny, nx), bool)
    inv[18:23, 28:33] = True
    pol2 = polar_image(img, (xc, yc), (0.005, 0.008), 2.0, 6.0, invalid=inv)
    assert np.all(np.isnan(pol2.image[0]))        # beamstop region blank
