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
    # I(q) of the deconvolved rho fits the data (alpha of step 2 from I(q); with the
    # p(r) evidence the sphere gave chi2/M ~ 5)
    assert T.chi2_red_profile < (2.0 if geometry == "sphere" else 1.5)
    assert T.sigma_scale == 1.0 or geometry == "lamella"     # default: n_r = N_s, no scaling
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


def _spheres(q, modes):
    """Polydisperse homogeneous spheres: modes = [(R0, sigma_lognormal, volume fraction)]."""
    out = np.zeros_like(q)
    for R0, sg, w in modes:
        R = np.linspace(R0 * np.exp(-4 * sg), R0 * np.exp(4 * sg), 300)
        n = np.exp(-np.log(R / R0) ** 2 / (2 * sg**2)) / R
        V = 4 / 3 * np.pi * R**3
        n = w * n / np.trapezoid(n * V, R)
        x = np.outer(q, R)
        F = 3 * (np.sin(x) - x * np.cos(x)) / x**3
        out += np.trapezoid((V**2 * F**2) * n[None, :], R, axis=1)
    return out


def test_single_particle_consistent_polydisperse_flagged():
    """2026-10-21: with the default diagonal p(r) errors a single core-shell sphere
    is deconvolved with chi2/M ~ 1 (full covariance gave ~170), while bimodal
    polydisperse spheres (cf. examples/studio/data/test.dat) are flagged."""
    q = np.geomspace(0.004, 0.385, 91)
    rng = np.random.default_rng(1)
    It = models.intensity_from_profile(q, models.core_shell(200, 1.0, 0.4), 300)
    It = It / It[0] * 3500
    s = 0.02 * It + 0.002
    T = two_step_fit(q, It + 0.1 + rng.normal(size=q.size) * s, s, n=40, background=True, r_max=300,
                     r_max_range=(150, 450), n_D=10)
    assert T.profile.chi2_red < 2.0
    assert "not the self-convolution" not in T.summary()
    It = _spheres(q, [(60, 0.2, 1.9), (180, 0.2, 1.0)])
    It = It / It[0] * 3500
    s = 0.02 * It + 0.002
    T = two_step_fit(q, It + 0.1 + rng.normal(size=q.size) * s, s, n=40, background=True, r_max=300,
                     r_max_range=(150, 450), n_D=10)
    assert T.profile.chi2_red > 3.0
    assert "not the self-convolution" in T.summary()


def test_step2_errors_scaled_to_shannon_number():
    """More IFT points than N_s carry no new information: step 2 multiplies the p(r)
    errors by sqrt(M / N_s), so the profile hardly depends on the number of points."""
    from pysasfit.sldprofile.twostep import shannon_points
    f = lambda r: np.where(r < 3, 1.0, np.where(r < 5, 0.4, 0.0))  # noqa: E731
    q, I, s = models.simulate(np.linspace(0.1, 2.5, 120), f, 5.0, rel_noise=0.01, abs_noise=1e-4, seed=2)[:3]
    T = two_step_fit(q, I, s, n=30, r_max=5.0, fix_r_max=True, n_r=60)
    ns = shannon_points(q.max(), 10.0)
    assert np.isclose(T.sigma_scale, np.sqrt(60 / ns))
    r2, p2, s2 = T.step2_data
    assert np.allclose(s2, T.pddf.p_err * T.sigma_scale)
    assert "p(r) errors x" in T.summary()
    T1 = two_step_fit(q, I, s, n=30, r_max=5.0, fix_r_max=True, n_r=60, scale_errors=False)
    assert T1.sigma_scale == 1.0


def test_step2_alpha_from_intensity():
    """Default: step-2 alpha from the fit of the measured I(q) (smoothest rho within
    one standard deviation of chi2). The small sphere (q_max D_max / pi = 8) was
    smoothed too much by the p(r) evidence: I(q) from rho chi2/M ~ 5 -> < 2."""
    f = lambda r: np.where(r < 3, 1.0, np.where(r < 5, 0.4, 0.0))  # noqa: E731
    q, I, s = models.simulate(np.linspace(0.1, 2.5, 120), f, 5.0, rel_noise=0.01, abs_noise=1e-4, seed=2)[:3]
    T = two_step_fit(q, I, s, n=30, r_max=5.0, fix_r_max=True)
    Te = two_step_fit(q, I, s, n=30, r_max=5.0, fix_r_max=True, alpha_from="evidence")
    assert T.alpha_from == "intensity" and Te.alpha_from == "evidence"
    a, ae = T.alpha_scan, Te.alpha_scan
    assert a["alpha_from"] == "intensity" and np.isclose(a["chosen"], a["intensity_alpha"])
    assert ae["alpha_from"] == "evidence" and np.isclose(ae["chosen"], Te.profile.best.alpha)
    assert ae["alpha"].size == ae["chi2_I"].size == ae["log_evidence"].size > 10
    assert T.chi2_red_profile <= a["threshold"] + 1e-9 and T.chi2_red_profile < 2.0 < Te.chi2_red_profile
    assert a["chosen"] < a["evidence_alpha"]
    assert "alpha of step 2 from I(q)" in T.summary() and "fit of I(q)" in T.profile.method
    P = T.profile
    m = P.r < 5.5
    c = P.rho[P.r < 2].mean()
    assert np.sqrt(np.mean((P.rho[m] - f(P.r[m]) * c) ** 2)) / c < 0.12
    with pytest.raises(ValueError):
        two_step_fit(q, I, s, n=10, r_max=5.0, fix_r_max=True, alpha_from="nonsense")
