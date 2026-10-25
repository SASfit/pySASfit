import numpy as np
import pytest

from pysasfit.sasred.data.uarray import Source, Term, UArray, constant


def make(rng, shape=(5, 6), mean=40):
    N = rng.poisson(mean, shape).astype(float)
    return N, UArray.from_source(N, Source("cnt", N.ravel(), "counting"))


def dense_check(y, key, f, x0, var, eps=1e-6):
    """Compare the stored Jacobian with central finite differences of f."""
    x0 = np.asarray(x0, float).ravel()
    J = np.zeros((np.size(f(x0)), x0.size))
    for j in range(x0.size):
        d = np.zeros_like(x0)
        d[j] = eps * max(1.0, abs(x0[j]))
        J[:, j] = (np.ravel(f(x0 + d)) - np.ravel(f(x0 - d))) / (2 * d[j])
    np.testing.assert_allclose(y.jacobian(key), J, rtol=1e-6, atol=1e-12)
    np.testing.assert_allclose(y.variance(keys=[key]).ravel(), (J ** 2 * var).sum(1), rtol=1e-8)


def test_elementwise_chain_matches_finite_differences():
    rng = np.random.default_rng(0)
    N, c = make(rng)
    y = (c * 2.0 + 3.0).sqrt() / (c + 1.0) - c.log()
    f = lambda x: np.sqrt(2 * x + 3) / (x + 1) - np.log(x)
    dense_check(y, "cnt", f, N, N.ravel())


def test_rank_one_coupling_from_aggregate():
    rng = np.random.default_rng(1)
    N, c = make(rng)
    tau, t = 2e-4, 3.0
    y = c / (1.0 - tau * c.sum() / t)
    f = lambda x: x / (1 - tau * x.sum() / t)
    dense_check(y, "cnt", f, N, N.ravel())
    assert y.terms["cnt"].U is not None          # low-rank part present
    np.testing.assert_allclose(np.diag(y.covariance()), y.variance().ravel())


def test_same_source_adds_linearly():
    rng = np.random.default_rng(2)
    _, c = make(rng)
    m = UArray.from_source(1e5, Source("mon", np.array([1e5]), "monitor", True))
    z = c / m - c / m
    assert np.allclose(z.variance(), 0.0)
    r = (c / m) / (c / m)
    assert np.allclose(r.variance(), 0.0, atol=1e-25)


def test_linear_map_and_covariance():
    rng = np.random.default_rng(3)
    N, c = make(rng, (4, 4))
    m = UArray.from_source(2e4, Source("mon", np.array([2e4]), "monitor", True))
    img = c / m
    A = rng.random((3, 16))
    y = img.reshape((16,)).linear(A)
    J = A / 2e4
    C = J @ np.diag(N.ravel()) @ J.T + np.outer(y.value, y.value) / 2e4
    np.testing.assert_allclose(y.covariance(), C, rtol=1e-10)
    np.testing.assert_allclose(y.covariance(systematic=False), J @ np.diag(N.ravel()) @ J.T, rtol=1e-10)


def test_budget_adds_up():
    rng = np.random.default_rng(4)
    _, c = make(rng)
    a = UArray.scalar(0.1, 0.002, "att", "attenuator")
    m = UArray.from_source(1e4, Source("mon", np.array([1e4]), "monitor", True))
    y = c / a / m
    np.testing.assert_allclose(sum(y.budget().values()), y.variance(), rtol=1e-12)


def test_conflicting_sources_rejected():
    a = UArray.from_source(1.0, Source("k", np.array([1.0])))
    b = UArray.from_source(1.0, Source("k", np.array([2.0])))
    with pytest.raises(ValueError):
        a + b


def test_lowrank_compression_is_exact():
    rng = np.random.default_rng(5)
    U = rng.random((7, 12))
    V = rng.random((9, 12))
    t = Term(7, 9, None, U[:, :6], V[:, :6]) + Term(7, 9, None, U[:, 6:], V[:, 6:])
    np.testing.assert_allclose(t.dense(), U @ V.T, rtol=1e-10)
