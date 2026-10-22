"""Shared helpers of the new test files: small stand-in models, data and problems (no SASfit library needed)."""

from importlib.util import find_spec

import numpy as np

from pysasfit.studio.fitting import FitProblem
from pysasfit.studio.model import ModelFunction

Q = np.linspace(0.1, 3, 60)


def installed(module):
    return find_spec(module) is not None


class _Function:
    parameter_names = ["A", "k"]


class Toy:
    """A * exp(-k q) with a tiny numerical jitter (like an adaptive integral); the contribution interface of the fit."""

    def __init__(self, A=2.0, k=1.0, jitter=1e-5):
        self._v = {"ff": np.array([A, k], float)}
        self._f = _Function()
        self.jitter = jitter

    def function_for(self, role):
        return self._f if role == "ff" else None

    def vector(self, role):
        return self._v.get(role)

    def intensity(self, q):
        A, k = self._v["ff"]
        h = np.sin(1e7 * (A * 1.234 + k * 2.345) + np.arange(q.size)) * self.jitter
        return A * np.exp(-k * q) * (1 + h)


class _OscFunction:
    parameter_names = ["A", "w"]


class Osc:
    """A (1 + cos(w q)): many local minima in w."""

    def __init__(self, A=1.0, w=3.0):
        self._v = {"ff": np.array([A, w], float)}
        self._f = _OscFunction()

    def function_for(self, role):
        return self._f if role == "ff" else None

    def vector(self, role):
        return self._v.get(role)

    def intensity(self, q):
        A, w = self._v["ff"]
        return A * (1 + np.cos(w * q))


def make_data(A=5.0, k=0.8, bg=0.3, seed=1, q=Q):
    rng = np.random.default_rng(seed)
    I = A * np.exp(-k * q) + bg
    s = 0.01 * (I + 0.05)
    return q, I + rng.normal(0, s), s


def make_model(A=2.0, k=1.0, scale=1.0, bg=0.0, **kw):
    return ModelFunction([Toy(A, k)], scales=[scale], background=bg, **kw)


def problem(data=None, upper=True, start=(2.0, 1.0)):
    """Fit of A, k and the background of a decaying exponential."""
    p = FitProblem()
    p.add_dataset(data or make_data(), make_model(*start))
    p.add_parameter("0.ff.A", lower=0, upper=20 if upper else np.inf)
    p.add_parameter("0.ff.k", lower=0, upper=5 if upper else np.inf)
    p.add_parameter("background", lower=-1, upper=2 if upper else np.inf)
    return p


def two_dataset_problem():
    """Two data sets of the same model: k common, A local."""
    model = make_model()
    p = FitProblem()
    p.add_dataset(make_data(5, 0.8, 0.3, seed=1), model, name="a")
    p.add_dataset(make_data(2, 0.8, 0.1, seed=2), model, name="b")
    p.add_common("0.ff.k", lower=0)
    p.add_local("0.ff.A", lower=0)
    return p


def osc_problem(w_start=3.0, seed=2):
    """Fit of A, w and the background of 2 (1 + cos(7 q)) + 0.5: the fit in w has many local minima."""
    I = 2.0 * (1 + np.cos(7.0 * Q)) + 0.5
    s = np.full(Q.size, 0.02)
    rng = np.random.default_rng(seed)
    p = FitProblem()
    p.add_dataset((Q, I + rng.normal(0, s), s), ModelFunction([Osc(1.0, w_start)], scales=[1.0], background=0.4))
    p.add_parameter("0.ff.A", lower=0.1, upper=10)
    p.add_parameter("0.ff.w", lower=1, upper=12)
    p.add_parameter("background", lower=0, upper=2)
    return p


def power_problem(alpha_free=False, bg_A=0.05, seed=3, jitter=1e-5):
    """Exponential plus background 0.3 plus 0.02 q^-4."""
    rng = np.random.default_rng(seed)
    I = 5.0 * np.exp(-0.8 * Q) + 0.3 + 0.02 * Q**-4.0
    s = 0.01 * (I + 0.05)
    p = FitProblem()
    p.add_dataset((Q, I + rng.normal(0, s), s), ModelFunction([Toy(2.0, 1.0, jitter)], scales=[1.0], background=0.1, bg_A=bg_A, bg_alpha=4.0))
    p.add_parameter("0.ff.A", lower=0, upper=20)
    p.add_parameter("0.ff.k", lower=0, upper=5)
    p.add_parameter("background", lower=-1, upper=2)
    p.add_parameter("bg.A", lower=0)
    if alpha_free:
        p.add_parameter("bg.alpha", lower=2, upper=6)
    return p
