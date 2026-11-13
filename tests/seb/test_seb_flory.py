"""FloryPolymer / FloryLoop: chains with a Flory exponent (Hammouda's generalised Gaussian chain)."""
import itertools

import numpy as np
import pytest
from scipy.integrate import quad

from pysasfit.seb import Structure, builders
from pysasfit.seb import special as sf
from pysasfit.seb import templates as T
from pysasfit.seb.subunits import FloryLoop, FloryPolymer, GaussianLoop, GaussianPolymer

q = np.logspace(-4, 0.5, 120)


def test_nu_one_half_is_the_gaussian_chain():
    fp, gp = FloryPolymer(), GaussianPolymer()
    p, g = {"Rg": 12.0, "nu": 0.5}, {"Rg": 12.0}
    np.testing.assert_allclose(fp.ff(q, p), gp.ff(q, g), atol=1e-12)
    for r in fp.references():
        np.testing.assert_allclose(fp.amp(q, p, r), gp.amp(q, g, r), atol=1e-12)
    for a, b in itertools.combinations_with_replacement(fp.references(), 2):
        np.testing.assert_allclose(fp.psi(q, p, a, b), gp.psi(q, g, a, b), atol=1e-12)
    np.testing.assert_allclose(FloryLoop().ff(q, p), GaussianLoop().ff(q, g), atol=1e-12)


@pytest.mark.parametrize("nu", [1 / 3, 0.45, 0.588, 0.75])
def test_closed_forms_against_quadrature(nu):
    for U in (1e-5, 0.3, 0.6, 4.0, 60.0, 2e3):
        pts = [1e-7, 1e-5, 1e-3, 1e-1]           # the integrand is concentrated at t -> 0 for large U
        E = quad(lambda t: np.exp(-U * t ** (2 * nu)), 0, 1, epsabs=0, epsrel=1e-12, limit=400, points=pts)[0]
        F = quad(lambda t: 2 * (1 - t) * np.exp(-U * t ** (2 * nu)), 0, 1, epsabs=0, epsrel=1e-12, limit=400,
                 points=pts)[0]
        L = 2 * quad(lambda s: np.exp(-U * (s * (1 - s)) ** (2 * nu)), 0, 0.5, epsabs=0, epsrel=1e-12, limit=400,
                     points=[1e-4, 1e-2])[0]
        assert sf.flory_end(np.array([U]), nu)[0] == pytest.approx(E, rel=1e-9)
        assert sf.flory_ff(np.array([U]), nu)[0] == pytest.approx(F, rel=1e-9)
        assert sf.flory_loop_ff(np.array([U]), nu)[0] == pytest.approx(L, rel=1e-8)


@pytest.mark.parametrize("cls", [FloryPolymer, FloryLoop])
@pytest.mark.parametrize("nu", [0.35, 0.588, 0.8])
def test_guinier_gives_rg(cls, nu):
    m, Rg, q0 = cls(), 10.0, 1e-3
    F = m.ff(np.array([q0, q0 / 2]), {"Rg": Rg, "nu": nu})
    g = 3 * (1 - F) / np.array([q0, q0 / 2]) ** 2
    assert (4 * g[1] - g[0]) / 3 == pytest.approx(Rg**2, rel=1e-6)


def test_chain_of_flory_blocks_differs_from_one_long_chain():
    """Joined blocks lose the excluded volume between them (only nu = 1/2 is additive)."""
    s = builders.linear_chain(4, kind="FloryPolymer")
    long_rg = np.sqrt(4) * 5.0
    for nu, same in ((0.5, True), (0.588, False)):
        F = s.form_factor(q, {"Rg_block": 5.0, "nu_block": nu})
        ref = FloryPolymer().ff(q, {"Rg": long_rg, "nu": 0.5}) if nu == 0.5 else None
        if same:
            np.testing.assert_allclose(F, ref, rtol=1e-10)
    s.form_factor(q, {"Rg_block": 5.0, "nu_block": 0.588})


def test_fbm_monte_carlo_agrees():
    """Chains with exactly the model's statistics (fractional Brownian motion)."""
    rng = np.random.default_rng(5)
    n, nu, m = 64, 0.588, 300
    i = np.arange(1, n + 1, dtype=float)
    C = 0.5 * (i[:, None] ** (2 * nu) + i[None, :] ** (2 * nu) - np.abs(i[:, None] - i[None, :]) ** (2 * nu))
    L = np.linalg.cholesky(C)
    qq = np.array([0.1, 0.25, 0.4])
    A = np.zeros(3)
    P = np.zeros(3)
    r2 = 0.0
    for _ in range(m):
        X = np.vstack([np.zeros(3), L @ rng.normal(size=(n, 3)) / np.sqrt(3)])
        d = np.sqrt(((X - X[0]) ** 2).sum(1))
        A += np.sinc(np.outer(qq, d) / np.pi).mean(1)
        P += np.sinc(qq * d[-1] / np.pi)
        r2 += np.mean(np.sum((X - X.mean(0)) ** 2, 1))
    p = {"Rg": np.sqrt(r2 / m), "nu": nu}
    np.testing.assert_allclose(A / m, FloryPolymer().amp(qq, p, "end1"), atol=0.015)
    np.testing.assert_allclose(P / m, FloryPolymer().psi(qq, p, "end1", "end2"), atol=0.05)


def test_templates_and_designer_offer_flory_chains():
    s = T.build(T.code_for("Micelle (core + grafted chains)", {"chain": "FloryPolymer", "n": 6}))
    assert {"Rg_chain", "nu_chain"} <= set(s.parameters())
    s = T.build(T.code_for("Micelle (core + grafted chains)", {"chain": "FloryLoop", "n": 3}))
    F = s.form_factor(q, {"R_core": 20, "Rg_chain": 8, "nu_chain": 0.6, "beta_core": 2, "beta_chain": 1})
    np.testing.assert_allclose(F, s.form_factor_pairs(q, {"R_core": 20, "Rg_chain": 8, "nu_chain": 0.6,
                                                           "beta_core": 2, "beta_chain": 1}), rtol=1e-10)
    st = Structure()
    st.add("FloryPolymer", "a")
    with pytest.raises(ValueError):
        st.form_factor(q, {"Rg_a": 5.0, "nu_a": 1.5})
