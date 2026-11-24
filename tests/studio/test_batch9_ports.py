"""Batch 9 of the ports (2026-11-24): peaks and distributions with complete statistics in _v
(ports/distributions.py): peak_trapezoidal, q_distribution, kappa_generalized_distribution,
generalized_distributions, metalog and t_r_y.

The _v selector follows enum sasfit_dist (sasfit_common.h): 0 TYPE, 1 QUANTILE(y), 2 QUANTILE_DENS(y),
3 PROBABILITY(x), 4 CUMULATIVE(x), 5 MODE, 6 MEAN, 7 VARIANCE, 8 MEDIAN, 9 SKEWNESS (Bowley), 10 EXCESS_KURTOSIS
(Moors ratio, as in the C code).

C comparison against the (fixed) C code, then independent checks of every statistic: normalisation, F against the
integral of p, Q(F(x)) = x, q(y) p(Q(y)) = 1, mean and variance against quadrature, the mode as maximum, the quantile
measures from Q, and closed forms against scipy.stats."""

from __future__ import annotations

import ctypes
import importlib.util
import math
import os

import numpy as np
import pytest
from scipy import integrate, stats

from pysasfit.studio.ports import distributions as D
from pysasfit.studio.ports import get, sample_parameters

HERE = os.path.dirname(os.path.abspath(__file__))
NAMES = [s["name"] for s in D.SPECS]
C_NAMES = [s["name"] for s in D.SPECS if s.get("c_reference", True)]
SD_NAMES = [n for n in NAMES if n.startswith("sd_")]


class _Param(ctypes.Structure):
    _fields_ = [
        ("p", ctypes.c_double * 50),
        ("errStatus", ctypes.c_int),
        ("errStr", ctypes.c_char * 256),
        ("kernelSelector", ctypes.c_int),
    ]


@pytest.fixture(scope="module")
def portref(tmp_path_factory):
    spec = importlib.util.spec_from_file_location("port_reference_build_b9", os.path.join(HERE, "port_reference", "build.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    lib = mod.build(str(tmp_path_factory.mktemp("portref_b9")))
    if lib is None:
        pytest.skip("no C compiler or GSL for the C reference of the ports")
    return ctypes.CDLL(lib)


def _ref(lib, name, kind=""):
    f = getattr(lib, "sasfit_" + name + kind)
    f.restype = ctypes.c_double
    f.argtypes = [ctypes.c_double, ctypes.POINTER(_Param)] + ([ctypes.c_int] if kind == "_v" else [])

    def call(x, p, *a):
        s = _Param()
        s.p[: len(p)] = [float(v) for v in p[:50]]
        r = f(float(x), ctypes.byref(s), *a)
        return math.nan if s.errStatus else r

    return call


def _v(spec, u, p, dist):
    return np.asarray(spec.volume(np.atleast_1d(np.asarray(u, float)), p, dist), float)


def _v1(spec, u, p, dist):
    return float(_v(spec, u, p, dist)[0])


def _norm(spec, p):
    """PROBABILITY and CUMULATIVE of the size distributions are N p and N F."""
    return p[0] if spec.name.startswith("sd_") else 1.0


def _xgrid(spec, p, n=37):
    lo, hi = _v(spec, [0.01, 0.99], p, D.QUANTILE)
    return np.linspace(lo - 0.2 * (hi - lo), hi + 0.2 * (hi - lo), n)


# ---------------------------------------------------------------------------------------------------- C reference
@pytest.mark.parametrize("name", C_NAMES)
def test_intensity_matches_c(portref, name):
    spec = get(name)
    f = _ref(portref, name)
    for p in sample_parameters(spec, 5, seed=3):
        xs = _xgrid(spec, p)
        py = np.asarray(spec.intensity(xs, p), float)
        c = np.array([f(x, p) for x in xs])
        ok = np.isfinite(c)
        assert ok.sum() > 20
        np.testing.assert_allclose(py[ok], c[ok], rtol=1e-9, atol=1e-11 * np.max(np.abs(py)))


@pytest.mark.parametrize("name", SD_NAMES)
def test_size_distribution_statistics_match_c(portref, name):
    spec = get(name)
    fv = _ref(portref, name, "_v")
    dists = [D.QUANTILE, D.QUANTILE_DENS, D.PROBABILITY, D.CUMULATIVE, D.MEDIAN, D.SKEWNESS, D.EXCESS_KURTOSIS]
    if name == "sd_fgld_":
        dists += [D.MEAN, D.VARIANCE]
    for p in sample_parameters(spec, 4, seed=5):
        xs = _xgrid(spec, p)[5:32:6]
        for dist in dists:
            us = [0.1, 0.3, 0.7, 0.9] if dist in (D.QUANTILE, D.QUANTILE_DENS) else (xs if dist in (D.PROBABILITY, D.CUMULATIVE) else [0.5])
            py = _v(spec, us, p, dist)
            c = np.array([fv(u, p, dist) for u in us])
            ok = np.isfinite(c) & (c != 0)        # 0: selector not implemented in C
            np.testing.assert_allclose(py[ok], c[ok], rtol=1e-8, atol=1e-12 * max(1.0, np.max(np.abs(py))),
                                       err_msg=f"{name} dist={dist} p={p[:9]}")


@pytest.mark.parametrize("name", ["ff_q_exponential__area_", "ff_q_weibull__area_"])
def test_peak_quantile_matches_c(portref, name):
    spec = get(name)
    fv = _ref(portref, name, "_v")
    for p in sample_parameters(spec, 4, seed=7):
        for y in (0.1, 0.5, 0.9):
            c = fv(y, p, D.QUANTILE)
            if math.isfinite(c) and c != 0:
                assert _v1(spec, y, p, D.QUANTILE) == pytest.approx(c, rel=1e-9)


# ---------------------------------------------------------------------------------------- independent statistics
def _support(spec, p, eps=1e-9):
    return _v(spec, [eps, 0.5, 1 - eps], p, D.QUANTILE)


@pytest.mark.parametrize("name", NAMES)
def test_type_selector(name):
    spec = get(name)
    p = spec.defaults()
    assert _v1(spec, 0.5, p, D.TYPE) in (1.0, 3.0)


@pytest.mark.parametrize("name", NAMES)
def test_normalisation_and_cdf(name):
    spec = get(name)
    for p in sample_parameters(spec, 3, seed=11):
        n = _norm(spec, p)
        lo, med, hi = _support(spec, p)
        pdf = lambda x: _v1(spec, x, p, D.PROBABILITY) / n  # noqa: E731
        tot = integrate.quad(pdf, lo, med, limit=400)[0] + integrate.quad(pdf, med, hi, limit=400)[0]
        assert tot == pytest.approx(1.0, abs=2e-6), p
        for y in (0.05, 0.25, 0.75, 0.95):
            x = _v1(spec, y, p, D.QUANTILE)
            assert _v1(spec, x, p, D.CUMULATIVE) / n == pytest.approx(y, abs=1e-9), (y, p)
            assert integrate.quad(pdf, lo, x, limit=400)[0] == pytest.approx(y, abs=3e-6), (y, p)


@pytest.mark.parametrize("name", NAMES)
def test_quantile_density_is_inverse_pdf(name):
    spec = get(name)
    for p in sample_parameters(spec, 3, seed=13):
        n = _norm(spec, p)
        for y in (0.1, 0.4, 0.6, 0.9):
            x = _v1(spec, y, p, D.QUANTILE)
            qd = _v1(spec, y, p, D.QUANTILE_DENS)
            assert qd * _v1(spec, x, p, D.PROBABILITY) / n == pytest.approx(1.0, rel=1e-6), (y, p)


@pytest.mark.parametrize("name", NAMES)
def test_median_and_quantile_measures(name):
    spec = get(name)
    for p in sample_parameters(spec, 3, seed=17):
        q = lambda y: _v1(spec, y, p, D.QUANTILE)  # noqa: E731
        assert _v1(spec, 0.5, p, D.MEDIAN) == pytest.approx(q(0.5), rel=1e-9, abs=1e-12)
        iqr = q(0.75) - q(0.25)
        bowley = (q(0.75) + q(0.25) - 2 * q(0.5)) / iqr
        moors = (q(7 / 8) - q(5 / 8) + q(3 / 8) - q(1 / 8)) / iqr
        assert _v1(spec, 0.5, p, D.SKEWNESS) == pytest.approx(bowley, abs=1e-8)
        assert _v1(spec, 0.5, p, D.EXCESS_KURTOSIS) == pytest.approx(moors, rel=1e-8)


@pytest.mark.parametrize("name", NAMES)
def test_mean_variance_against_quadrature(name):
    spec = get(name)
    for p in sample_parameters(spec, 3, seed=19):
        mean, var = _v1(spec, 0.5, p, D.MEAN), _v1(spec, 0.5, p, D.VARIANCE)
        if not (math.isfinite(mean) and math.isfinite(var)):
            continue          # heavy tails: moments do not exist
        n = _norm(spec, p)
        lo, med, hi = _support(spec, p, 1e-13)
        pdf = lambda x: _v1(spec, x, p, D.PROBABILITY) / n  # noqa: E731
        m1 = sum(integrate.quad(lambda x: x * pdf(x), a, b, limit=400)[0] for a, b in ((lo, med), (med, hi)))
        m2 = sum(integrate.quad(lambda x: (x - m1) ** 2 * pdf(x), a, b, limit=400)[0] for a, b in ((lo, med), (med, hi)))
        sd = math.sqrt(var)
        assert mean == pytest.approx(m1, abs=1e-4 * sd + 1e-10), p
        assert var == pytest.approx(m2, rel=1e-3), p


@pytest.mark.parametrize("name", NAMES)
def test_mode_is_maximum(name):
    spec = get(name)
    for p in sample_parameters(spec, 3, seed=23):
        m = _v1(spec, 0.5, p, D.MODE)
        xs = np.linspace(*_support(spec, p, 1e-6)[[0, 2]], 2001)
        px = _v(spec, xs, p, D.PROBABILITY)
        pm = _v1(spec, m, p, D.PROBABILITY)
        if pm == 0:   # maximum at an end of the support (density may diverge there): limit from the inside
            h = 1e-13 * max(1.0, abs(m))
            pm = max(_v1(spec, m + h, p, D.PROBABILITY), _v1(spec, m - h, p, D.PROBABILITY))
        assert np.all(px <= pm * (1 + 1e-6) + 1e-300), p


# ------------------------------------------------------------------------------------------- closed forms, scipy
def test_trapez_against_scipy():
    spec = get("ff_trapez__area")
    p = [1.0, 0.1, 0.3, 0.6, 1.0, 0.0]
    ref = stats.trapezoid((0.3 - 0.1) / 0.9, (0.6 - 0.1) / 0.9, loc=0.1, scale=0.9)
    xs = np.linspace(0, 1.1, 23)
    np.testing.assert_allclose(_v(spec, xs, p, D.PROBABILITY), ref.pdf(xs), rtol=1e-12, atol=1e-14)
    np.testing.assert_allclose(_v(spec, xs, p, D.CUMULATIVE), ref.cdf(xs), rtol=1e-12, atol=1e-14)
    assert _v1(spec, 0.5, p, D.MEAN) == pytest.approx(ref.mean(), rel=1e-12)
    assert _v1(spec, 0.5, p, D.VARIANCE) == pytest.approx(ref.var(), rel=1e-12)


def test_smoothed_trapez_variance_and_limit():
    spec = get("ff_smoothed_symm_trapez_area")
    c, w1, w2, s = 0.5, 0.4, 0.2, 0.05
    p = [1.0, c, w1, w2, s, 0.0]
    assert _v1(spec, 0.5, p, D.VARIANCE) == pytest.approx((w1**2 + w2**2) / 12 + s**2, rel=1e-12)
    assert _v1(spec, 0.5, p, D.MEAN) == pytest.approx(c, abs=1e-14)
    # convolution of two boxes (sigma -> 0) = trapezoid with plateau |w1 - w2| and edges w2
    p0 = [1.0, c, w1, w2, 1e-9, 0.0]
    ref = stats.trapezoid(w2 / (w1 + w2), w1 / (w1 + w2), loc=c - (w1 + w2) / 2, scale=w1 + w2)
    xs = np.linspace(0.1, 0.9, 17)
    np.testing.assert_allclose(_v(spec, xs, p0, D.CUMULATIVE), ref.cdf(xs), atol=1e-8)


def test_q_gaussian_is_student_t():
    spec = get("ff_q_gaussian__area_")
    q, beta = 1.5, 2.0
    nu = (3 - q) / (q - 1)
    ref = stats.t(nu, loc=0.3, scale=1 / math.sqrt(beta * (3 - q)))
    p = [1.0, q, beta, 0.0, 0.3, 0.0]
    xs = np.linspace(-2, 2, 21)
    np.testing.assert_allclose(_v(spec, xs, p, D.PROBABILITY), ref.pdf(xs), rtol=1e-10)
    assert _v1(spec, 0.5, p, D.VARIANCE) == pytest.approx(ref.var(), rel=1e-10)


def test_q_exponential_moments():
    spec = get("ff_q_exponential__area_")
    q, lam = 1.2, 2.0
    p = [1.0, q, lam, 0.0, 0.0, 0.0]
    assert _v1(spec, 0.5, p, D.MEAN) == pytest.approx(1 / (lam * (3 - 2 * q)), rel=1e-12)
    # q -> 1: exponential distribution
    p1 = [1.0, 1.0, lam, 0.0, 0.0, 0.0]
    assert _v1(spec, 0.3, p1, D.QUANTILE) == pytest.approx(stats.expon(scale=1 / lam).ppf(0.3), rel=1e-12)


def test_generalized_against_scipy():
    gev = get("ff_generalized_extreme_value__area_")
    p = [1.0, 1.5, 0.2, 0.5, 0.0, 0.0]
    ref = stats.genextreme(-0.2, loc=0.5, scale=1.5)
    assert _v1(gev, 0.5, p, D.MEAN) == pytest.approx(ref.mean(), rel=1e-10)
    assert _v1(gev, 0.5, p, D.VARIANCE) == pytest.approx(ref.var(), rel=1e-9)
    assert _v1(gev, 0.5, p, D.MODE) == pytest.approx(0.5 + 1.5 * ((1 + 0.2) ** -0.2 - 1) / 0.2, rel=1e-12)
    gn = get("ff_generalized_normal__area_")
    p = [1.0, 1.0, 0.0, 0.5, 0.0, 0.0]        # kappa = 0: normal distribution
    assert _v1(gn, 0.8, p, D.QUANTILE) == pytest.approx(stats.norm(0.5, 1.0).ppf(0.8), rel=1e-12)
    assert _v1(gn, 0.5, p, D.EXCESS_KURTOSIS) == pytest.approx(1.2330951, rel=1e-6)   # Moors ratio of N(0,1)


def test_metalog_two_terms_is_logistic():
    spec = get("sd_metalog_clipped")
    p = [1.0, -50.0, 50.0, 0.0, 0.5, 0.2] + [0.0] * 8
    ref = stats.logistic(loc=0.5, scale=0.2)
    for y in (0.1, 0.5, 0.9):
        assert _v1(spec, y, p, D.QUANTILE) == pytest.approx(ref.ppf(y), rel=1e-12)
        x = ref.ppf(y)
        assert _v1(spec, x, p, D.PROBABILITY) == pytest.approx(ref.pdf(x), rel=1e-9)
