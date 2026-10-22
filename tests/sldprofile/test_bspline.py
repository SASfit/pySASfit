"""B-spline representation of the density profile (pysasfit.sldprofile.bspline)."""
import os

import numpy as np
import pytest
from scipy.special import sici

from pysasfit.sldprofile import DensityProfileFit, geometry as geo, load_data, models
from pysasfit.sldprofile.bspline import BSplineBasis, BSplines

STAR = os.path.join(os.path.dirname(__file__), "..", "..", "examples", "sldprofile", "data",
                    "star_p4over3_bg0.05.dat")


@pytest.mark.parametrize("geometry", ["sphere", "cylinder", "lamella"])
@pytest.mark.parametrize("degree", [2, 3])
def test_constant_profile_is_exact(geometry, degree):
    """B-splines are a partition of unity: all coefficients 1 is the homogeneous
    particle, amplitude and Rg as for one shell."""
    b = BSplines(degree).build(12)
    q = np.linspace(1e-3, 3.0, 120)
    a = geo.amplitude_matrix(q, 5.0, 12, geometry, "rho", b) @ np.ones(12)
    ex = geo.amplitude_matrix(q, 5.0, 1, geometry)[:, 0]
    assert np.allclose(a, ex, rtol=1e-10, atol=1e-10 * np.abs(ex).max())
    assert np.isclose(geo.radius_of_gyration(np.ones(12), 5.0, geometry, "rho", b),
                      geo.radius_of_gyration(np.ones(1), 5.0, geometry))


def test_mass_basis_cusp_is_exact():
    """r^2 rho = 1 (rho = 1/r^2): F = 4 pi Si(qR)/q, through the Gauss-Jacobi first span."""
    b = BSplines(3).build(10)
    q = np.linspace(0.01, 1, 30)
    a = geo.amplitude_matrix(q, 50.0, 10, "sphere", "mass", b).sum(1)
    assert np.allclose(a, 4 * np.pi * sici(q * 50.0)[0] / q, rtol=1e-9)


def test_end_conditions_and_penalty():
    b = BSplines(3).build(15)
    x = np.linspace(0, 1, 401)
    # even at the centre: c'(0) = 0 for every basis function
    assert np.allclose(b.values(np.array([0.0]), nu=1), 0.0, atol=1e-9)
    # c = x^2 is in the space; S = (h^3/4) int (c'')^2 = (h^3/4) * 4
    c = np.linalg.lstsq(b.values(x), x**2, rcond=None)[0]
    assert np.allclose(b.values(x) @ c, x**2, atol=1e-10)
    assert np.isclose(c @ b.penalty() @ c, (1 / 15) ** 3)
    # outer='zero': c(1) = 0 for every basis function
    bz = BSplines(3).build(15, outer="zero")
    assert bz.n == 15 and np.allclose(bz.values(np.array([1.0])), 0.0)
    with pytest.raises(ValueError):
        BSplines(3, knots=np.linspace(0, 1, 5)).build(15)      # wrong number of breakpoints


def test_fit_homogeneous_sphere():
    q = np.linspace(0.12, 1.0, 50)
    q, I, s, It, sc = models.simulate(q, models.homogeneous, 7.5, rel_noise=0.05, abs_noise=0.0, seed=1)
    f = DensityProfileFit(q, I, s, n=15, grid=BSplines(3))
    sol, _ = f.best_alpha(7.5)
    assert sol.converged and 0.5 < sol.chi2 / q.size < 1.5
    assert abs(sol.Rg() - np.sqrt(0.6) * 7.5) < 0.1
    assert np.isclose(np.median(sol.rho_on(np.linspace(0.5, 7, 20))), np.sqrt(sc), rtol=0.1)
    r, p = sol.pddf()
    assert np.all(np.isfinite(p))


@pytest.mark.parametrize("geometry", ["cylinder", "lamella"])
def test_other_geometries(geometry):
    prof = models.core_shell(2.5, 1.0, 0.5)
    q = np.linspace(0.15, 1.5, 50)
    q, I, s, It, sc = models.simulate(q, prof, 5.0, geometry=geometry, seed=3)
    f = DensityProfileFit(q, I, s, n=15, geometry=geometry, grid=BSplines(3))
    sol, _ = f.best_alpha(5.0)
    assert 0.5 < sol.chi2 / q.size < 1.5
    rho = sol.rho_on(np.array([0.5, 1.5, 3.5, 4.5]))
    assert rho[:2].mean() > rho[2:].mean()


def test_maxent_spline_is_positive():
    q = np.linspace(0.12, 1.0, 50)
    q, I, s, It, sc = models.simulate(q, models.homogeneous, 7.5, rel_noise=0.05, abs_noise=0.0, seed=1)
    f = DensityProfileFit(q, I, s, n=15, grid=BSplines(3), regularization="maxent")
    sol, _ = f.best_alpha(7.5)
    assert np.all(sol.rho > 0) and np.all(sol.rho_on(np.linspace(0, 7.5, 50)) > 0)
    assert 0.5 < sol.chi2 / q.size < 1.5


def test_star_less_overfitting_than_shells():
    """Star-like rho ~ r^-4/3 with q_max r_max ~ 180: 40 shells follow the noise at
    large q (chi2 of the upper third ~0.35, N_g ~ 19); 20 cubic-spline coefficients
    fit with fewer effective parameters and no loss of evidence."""
    q, I, s, _ = load_data(STAR)
    up = slice(2 * q.size // 3, None)
    sh, _ = DensityProfileFit(q, I, s, n=40, basis=4 / 3, background=True).best_alpha(450.0)
    sp, _ = DensityProfileFit(q, I, s, n=20, basis=4 / 3, background=True, grid=BSplines(3)).best_alpha(450.0)
    hi = lambda sol: np.mean(((sol.I_fit - I)[up] / s[up]) ** 2)  # noqa: E731
    assert sp.n_good < sh.n_good - 3
    assert hi(sp) > 1.3 * hi(sh)
    assert sp.log_evidence > sh.log_evidence - 1.0
    assert abs(sp.Rg() - sh.Rg()) < 2.0


def test_adaptive_knots_and_power_knots():
    q, I, s, _ = load_data(STAR)
    f = DensityProfileFit(q, I, s, n=15, basis=4 / 3, background=True, grid=BSplines(3))
    g = f.adaptive_grid(450.0)
    assert isinstance(g, BSplines) and g.degree == 3
    m = BSplines.n_spans(15, 3)
    assert g.knots.size == m + 1
    assert np.diff(g.knots).min() * 450.0 >= min(np.pi / q.max(), 450.0 / 15) * 0.999
    fa = DensityProfileFit(q, I, s, n=15, basis=4 / 3, background=True, grid=g)
    sa, _ = fa.best_alpha(450.0)
    assert sa.chi2 / q.size < 1.5
    assert isinstance(sa.grid, BSplineBasis)
    fp = DensityProfileFit(q, I, s, n=15, basis=4 / 3, background=True, grid=BSplines(3, ("power", 1.5)))
    assert np.isclose(fp.grid.breaks[1], (1 / m) ** 1.5)


def test_two_step_with_splines():
    from pysasfit.sldprofile import two_step_fit

    q = np.linspace(0.12, 1.0, 50)
    q, I, s, It, sc = models.simulate(q, models.homogeneous, 7.5, rel_noise=0.02, abs_noise=0.0, seed=2)
    res = two_step_fit(q, I, s, n=12, grid=BSplines(3), r_max=7.5, fix_r_max=True)
    prof = res.as_profile_result()
    assert abs(prof.Rg[0] - np.sqrt(0.6) * 7.5) < 0.3


@pytest.mark.parametrize("outer", ["free", "zero"])
@pytest.mark.parametrize("degree", [2, 3])
def test_two_step_splines_with_end_conditions(outer, degree):
    """Regression (2026-10-21): two-step with B-splines and outer='zero' failed
    ('41 breakpoints given, 40 needed'): the I(q)-from-rho model used outer='free'."""
    from pysasfit.sldprofile import two_step_fit

    q = np.linspace(0.12, 1.0, 50)
    q, I, s, It, sc = models.simulate(q, models.homogeneous, 7.5, rel_noise=0.02, abs_noise=0.0, seed=2)
    res = two_step_fit(q, I, s, n=15, outer=outer, grid=BSplines(degree, ("power", 1.5)),
                       r_max=7.5, fix_r_max=True)
    assert np.isfinite(res.chi2_red_profile)
    assert res.profile.best.grid.outer == outer


def test_basis_respec_keeps_knot_distribution():
    b = BSplines(3, ("power", 1.5)).build(20, outer="zero")
    f = DensityProfileFit(np.linspace(0.1, 1, 30), np.ones(30), np.ones(30), n=20, outer="free", grid=b)
    assert f.grid.outer == "free" and f.grid.n == 20
    assert f.grid.breaks.size == BSplines.n_spans(20, 3, "free") + 1
    assert f.grid.breaks[1] < 1.0 / (f.grid.breaks.size - 1)      # still finer at the centre
