"""Tests of the size-distribution integration methods (no SASfit libraries needed)."""

from types import SimpleNamespace

import numpy as np
import pytest

from pysasfit.studio import integration as I


def _f(x):
    return np.array([x**2, np.sin(x), np.exp(-x)])


EXACT = np.array([1 / 3, 1 - np.cos(1), 1 - np.exp(-1)])


SCIPY_METHODS = [m for m in I.METHODS if m != "sasfit"]


class _FakeLibrary:
    """Stands in for libsasfit: Simpson rule behind the same ``integrate`` signature."""

    def integrate(self, f, a, b, limit=10000, epsabs=0.0, epsrel=None):
        x = np.linspace(a, b, 2001)
        y = np.array([f(float(v)) for v in x])
        return float((b - a) / 2000 / 3 * (y[0] + y[-1] + 4 * y[1:-1:2].sum() + 2 * y[2:-1:2].sum()))


def test_sasfit_method_with_fake_library(monkeypatch):
    from pysasfit.studio import library

    monkeypatch.setattr(library, "get_library", lambda *a, **k: _FakeLibrary())
    value, info = I.integrate(_f, 0.0, 1.0, method="sasfit")
    np.testing.assert_allclose(value, EXACT, rtol=1e-6)
    assert info["method"] == "sasfit"


@pytest.mark.parametrize("method", SCIPY_METHODS)
def test_integrate_known_integrals(method):
    value, info = I.integrate(_f, 0.0, 1.0, method=method)
    np.testing.assert_allclose(value, EXACT, rtol=1e-3)
    assert info["method"] == method and info["neval"] > 0


def test_unknown_method():
    with pytest.raises(ValueError, match="unknown integration method"):
        I.integrate(_f, 0, 1, method="nope")


class _Fn:
    def __init__(self, fn):
        self.fn = fn

    def evaluate_p(self, x, p, kind="intensity"):
        out = self.fn(np.asarray(x, float), p, kind)
        return float(out) if np.ndim(out) == 0 else out

    def volume(self, x, p, dist=0):
        return 4 / 3 * np.pi * x**3


def _sphere(q, p, kind):
    qr = q * p[0]
    amp = 3 * (np.sin(qr) - qr * np.cos(qr)) / qr**3 * 4 / 3 * np.pi * p[0] ** 3
    return amp if kind == "amplitude" else amp**2


def _gauss(x, p, kind):
    return np.exp(-0.5 * ((x - p[1]) / p[0]) ** 2) / (p[0] * np.sqrt(2 * np.pi))


def _sq(q, p, kind):
    return 1 - 0.8 * np.exp(-((q * p[0]) ** 2) / 4)


def _contribution(approach, **kw):
    c = SimpleNamespace(
        ff=_Fn(_sphere), sd=_Fn(_gauss), sq=_Fn(_sq),
        ff_p=np.array([50.0, 0, 0, 0, 0, 0]), sd_p=np.array([5.0, 50.0]), sq_p=np.array([50.0, 0.2]),
        distributed=0, approach=approach, **{**I.DEFAULTS, **kw},
    )  # fmt: skip
    c.support = lambda: (20.0, 80.0)
    return c


@pytest.mark.parametrize("approach", ["monodisperse", "decoupling", "lma"])
def test_methods_agree(approach):
    q = np.geomspace(0.005, 0.3, 30)
    ref_c = _contribution(approach, n_x=400)
    S = None if approach == "lma" else ref_c.sq.evaluate_p(q, ref_c.sq_p)
    ref = I.size_integral(ref_c, q, S)
    for method, kw in (("gauss-legendre", {}), ("quad_vec", {"rtol": 1e-8}), ("simpson", {}), ("romb", {})):
        c = _contribution(approach, method=method, **kw)
        np.testing.assert_allclose(I.size_integral(c, q, S), ref, rtol=2e-3, err_msg=method)
        assert c.last_integration["method"] == method


def test_settings_round_trip(monkeypatch):
    store = {}
    from pysasfit.studio import paths

    monkeypatch.setattr(paths, "get_setting", lambda key, default=None: store.get(key, default))
    monkeypatch.setattr(paths, "set_setting", lambda key, value: store.__setitem__(key, value))
    I.save_settings({**I.DEFAULTS, "method": "quad_vec", "n_x": 50, "rtol": 1e-5, "atol": 0.0, "limit": 300})
    assert I.load_settings()["method"] == "quad_vec"
    store["integration"] = {"method": "bogus", "n_x": "abc", "rtol": "1e-4"}
    s = I.load_settings()
    assert s["method"] == I.DEFAULTS["method"] and s["n_x"] == I.DEFAULTS["n_x"] and s["rtol"] == 1e-4
