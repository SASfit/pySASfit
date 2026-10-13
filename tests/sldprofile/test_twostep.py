"""Glatter's two-step route (IFT, then deconvolution) against the direct model."""
import numpy as np
import pytest

from pysasfit.sldprofile import DensityProfileFit, models
from pysasfit.sldprofile.twostep import DeconvolutionFit, ift_kernel, indirect_ft, pddf_matrices, two_step_fit

GEOMS = ["sphere", "cylinder", "lamella"]


def _profile():
    n, rm = 20, 5.0
    rho = 1 + 0.5 * np.cos(np.linspace(0, 3, n))
    rho[-3:] *= 0.3
    return n, rm, rho


@pytest.mark.parametrize("geometry", GEOMS)
def test_pddf_quadratic_form_reproduces_direct_intensity(geometry):
    """p = rho^T V rho, transformed with the IFT kernel, gives the direct model I(q)."""
    n, rm, rho = _profile()
    q = np.geomspace(0.05, 3, 40)
    I_dir = DensityProfileFit(q, np.ones_like(q), np.ones_like(q), geometry=geometry, n=n).model(
        rho, rm, with_background=False)
    if geometry == "lamella":                               # gamma(0) != 0: full symmetric grid
        nr = 200
        rf = -2 * rm + 4 * rm * np.arange(1, 2 * nr + 2) / (2 * nr + 2)
        V = pddf_matrices(np.abs(rf), rm, n, geometry, n_sub=6, half_width=0.5 * (rf[1] - rf[0]))
        I = 0.5 * ift_kernel(q, rf, geometry) @ np.einsum("jkl,k,l->j", V, rho, rho)
        tol = 2e-3
    else:
        r = 2 * rm * np.arange(1, 201) / 201
        V = pddf_matrices(r, rm, n, geometry, n_sub=6)
        I = ift_kernel(q, r, geometry) @ np.einsum("jkl,k,l->j", V, rho, rho)
        tol = 2e-4
    assert np.max(np.abs(I - I_dir)) / I_dir.max() < tol


@pytest.mark.parametrize("geometry", GEOMS)
def test_two_step_recovers_profile(geometry):
    f = lambda r: np.where(r < 3, 1.0, np.where(r < 5, 0.4, 0.0))  # noqa: E731
    q, I, s = models.simulate(np.linspace(0.1, 2.5, 120), f, 5.0, geometry=geometry,
                              rel_noise=0.01, abs_noise=1e-4, seed=2)[:3]
    T = two_step_fit(q, I, s, geometry=geometry, n=30)
    assert abs(T.pddf.D_max - 10.0) < 3 * max(T.pddf.D_max_err, 0.1) + 0.3
    assert 0.7 < T.pddf.chi2_red < 1.3
    assert T.chi2_red_profile < 1.5                         # I(q) of the deconvolved rho fits the data
    P = T.profile
    r = P.r
    m = r < 5.5
    c = P.rho[r < 2].mean()
    assert np.sqrt(np.mean((P.rho[m] - f(r[m]) * c) ** 2)) / c < 0.2
    v = T.as_profile_result()
    assert v.twostep is T and v.q is T.pddf.q and np.allclose(v.r_max_grid, T.pddf.D_grid / 2)


def test_lamella_pddf_is_finite_at_zero():
    f = lambda r: np.where(r < 3, 1.0, 0.0)  # noqa: E731
    q, I, s = models.simulate(np.linspace(0.1, 2.5, 100), f, 3.0, geometry="lamella", seed=1)[:3]
    pd = indirect_ft(q, I, s, "lamella", D_max=6.0)
    assert pd.r[0] == 0.0 and pd.p[0] > 0.5 * pd.p.max()  # gamma(0) = int rho^2 dx is the maximum


def test_diagonal_errors_and_fixed_r_max():
    f = lambda r: np.where(r < 4, 1.0, 0.0)  # noqa: E731
    q, I, s = models.simulate(np.linspace(0.1, 2.0, 100), f, 4.0, seed=3)[:3]
    T = two_step_fit(q, I, s, n=20, r_max=4.0, fix_r_max=True, errors="diagonal")
    assert T.pddf.D_max == 8.0 and T.profile.r_max[0] == 4.0
    assert "diagonal" in T.profile.method
    dec = DeconvolutionFit(T.pddf.r, T.pddf.p, sigma=T.pddf.p_err, n=20)
    assert dec.errors == "diagonal" and abs(dec.guess_r_max() - 4.0) < 0.3
