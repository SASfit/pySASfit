"""Pure-Python reference implementations of the SASfit functions the scout uses by default.

They follow the conventions of the SASfit sources (parameter order, absolute scale) and serve two purposes:

* the scout and its tests run without the compiled SASfit libraries (``register()``; the command line does this
  automatically when the libraries are not installed, or on request with ``--reference``);
* an independent cross-check of the compiled functions (``tests/scout/test_real_sasfit_scout.py``).

A registered Python function takes precedence over the compiled one of the same name
(:func:`pysasfit.studio.register_python_function`), so ``unregister()`` restores the compiled functions.
"""

from __future__ import annotations

import math

import numpy as np

__all__ = ["register", "unregister", "registered", "NAMES"]

_T, _W = np.polynomial.legendre.leggauss(48)
_X01, _W01 = 0.5 * (_T + 1.0), 0.5 * _W  # nodes and weights on [0, 1] (orientation average)


def _K(q, R, eta):
    """Amplitude of a homogeneous sphere: eta * 4 pi (sin qR - qR cos qR) / q^3."""
    x = q * R
    if x < 1e-4:
        return eta * 4.0 / 3.0 * math.pi * R**3 * (1.0 - x * x / 10.0 + x**4 / 280.0)
    return eta * 4.0 * math.pi * (math.sin(x) - x * math.cos(x)) / q**3


def _vsphere(x, p, dist):
    return 4.0 / 3.0 * math.pi * (x if dist == 0 else p[0]) ** 3


# ----------------------------------------------------------------------------- form factors
def _sphere_f(q, p):  # R, -, -, eta
    return _K(q, p[0], p[3])


def _shell3_f(q, p):  # R (inner radius), dR, eta1 (core contrast), eta2 (shell contrast)
    return _K(q, p[0], p[2] - p[3]) + _K(q, p[0] + p[1], p[3])


def _shell3_v(x, p, dist):
    R, dR = (x, p[1]) if dist == 0 else ((p[0], x) if dist == 1 else (p[0], p[1]))
    return 4.0 / 3.0 * math.pi * (R + dR) ** 3


def _hollow_f(q, p):  # R (core of solvent), dR, -, eta
    return _K(q, p[0] + p[1], p[3]) - _K(q, p[0], p[3])


def _jinc(x):
    """2 J1(x) / x."""
    from scipy.special import j1

    x = np.asarray(x, float)
    small = np.abs(x) < 1e-8
    return np.where(small, 1.0, 2.0 * j1(np.where(small, 1.0, x)) / np.where(small, 1.0, x))


def _cylinder(q, p):  # R, -, L, eta: orientation average over x = cos(theta)
    R, L, eta = p[0], p[2], p[3]
    a = _jinc(q * R * np.sqrt(1.0 - _X01**2)) * np.sinc(q * L * _X01 / (2.0 * math.pi))
    return float((math.pi * R * R * L * eta) ** 2 * np.sum(_W01 * a * a))


def _cylinder_v(x, p, dist):
    R, L = (x, p[2]) if dist == 0 else ((p[0], x) if dist == 2 else (p[0], p[2]))
    return math.pi * R * R * L


def _spheroid(q, p):  # R_polar, R_equatorial, -, -, -, eta_core, -, eta_solv
    a, b, eta = p[0], p[1], p[5] - p[7]
    r = np.sqrt(b * b * (1.0 - _X01**2) + a * a * _X01**2)
    x = q * r
    f = np.where(x < 1e-4, 1.0 - x * x / 10.0, 3.0 * (np.sin(x) - x * np.cos(x)) / np.where(x < 1e-4, 1.0, x) ** 3)
    return float((eta * 4.0 / 3.0 * math.pi * a * b * b) ** 2 * np.sum(_W01 * f * f))


def _spheroid_v(x, p, dist):
    a, b = (x, p[1]) if dist == 0 else ((p[0], x) if dist == 1 else (p[0], p[1]))
    return 4.0 / 3.0 * math.pi * a * b * b


def _rod(q, p):  # L, -, -, eta: infinitely thin rod
    from scipy.special import sici

    x = q * p[0]
    if x < 1e-4:
        return (p[3] * p[0]) ** 2 * (1.0 - x * x / 36.0)
    return (p[3] * p[0]) ** 2 * (2.0 * sici(x)[0] / x - (math.sin(x / 2.0) / (x / 2.0)) ** 2)


def _disc(q, p):  # R, -, -, eta: infinitely thin disc
    from scipy.special import j1

    x = q * p[0]
    amp = (p[3] * math.pi * p[0] ** 2) ** 2
    if x < 1e-3:
        return amp * (1.0 - x * x / 6.0)
    return amp * 2.0 / (x * x) * (1.0 - j1(2.0 * x) / x)


# ----------------------------------------------------------------------------- size distribution, structure factor
def _lognorm(x, p):  # N, s, p, mu
    N, s, pw, mu = p[0], p[1], p[2], p[3]
    if x <= 0 or s <= 0 or mu <= 0:
        return 0.0
    c = math.sqrt(2.0 * math.pi) * s * mu ** (1.0 - pw) * math.exp((1.0 - pw) ** 2 * s * s / 2.0)
    return N / c * x ** (-pw) * math.exp(-0.5 * (math.log(x / mu) / s) ** 2)


def _hard_sphere(q, p):  # RHS, fp: Percus-Yevick
    R, f = p[0], p[1]
    if f <= 0 or R <= 0:
        return 1.0
    a = (1.0 + 2.0 * f) ** 2 / (1.0 - f) ** 4
    b = -6.0 * f * (1.0 + f / 2.0) ** 2 / (1.0 - f) ** 4
    g = f * a / 2.0
    A = 2.0 * q * R
    if A < 0.1:  # the closed form cancels badly at small A; S is even in A: quadratic between S(0) = 1/a and S(0.1)
        s1 = _hard_sphere(0.05 / R, p)
        return 1.0 / a + (s1 - 1.0 / a) * (A / 0.1) ** 2
    s, c = math.sin(A), math.cos(A)
    G = (
        a * (s - A * c) / A**2
        + b * (2.0 * A * s + (2.0 - A * A) * c - 2.0) / A**3
        + g * (-(A**4) * c + 4.0 * ((3.0 * A * A - 6.0) * c + (A**3 - 6.0 * A) * s + 6.0)) / A**5
    )
    return 1.0 / (1.0 + 24.0 * f * G / A)


_FUNCTIONS = {
    # name: (intensity, amplitude, volume, parameter names, defaults)
    "ff_sphere": (lambda q, p: _sphere_f(q, p) ** 2, _sphere_f, _vsphere, ("R", "dummy", "dummy", "eta"), (10, 0, 0, 1)),
    "ff_spherical_shell_iii": (lambda q, p: _shell3_f(q, p) ** 2, _shell3_f, _shell3_v, ("R", "dR", "eta1", "eta2"), (10, 2, 0, 1)),
    "ff_hollow_sphere": (lambda q, p: _hollow_f(q, p) ** 2, _hollow_f, _shell3_v, ("R", "dR", "dummy", "eta"), (10, 2, 0, 1)),
    "ff_cylinder": (_cylinder, None, _cylinder_v, ("R", "dummy", "L", "eta"), (10, 0, 50, 1)),
    "ff_spheroid_R": (_spheroid, None, _spheroid_v,
                      ("R_polar", "R_equatorial", "dummy", "dummy", "dummy", "eta_core", "dummy", "eta_solv"),
                      (20, 10, 0, 0, 0, 1, 0, 0)),
    "ff_rod": (_rod, None, None, ("L", "dummy", "dummy", "eta"), (100, 0, 0, 1)),
    "ff_disc": (_disc, None, None, ("R", "dummy", "dummy", "eta"), (50, 0, 0, 1)),
    "sd_LogNorm": (_lognorm, None, None, ("N", "s", "p", "mu"), (1, 0.1, 1, 10)),
    "sq_HardSphere": (_hard_sphere, None, None, ("RHS", "fp"), (10, 0.1)),
}  # fmt: skip
NAMES = tuple(_FUNCTIONS)
_registered = False


def register():
    """Register the reference functions under their SASfit names (they then replace the compiled ones)."""
    global _registered
    from ..studio import register_python_function

    for name, (intensity, amplitude, volume, names, defaults) in _FUNCTIONS.items():
        register_python_function(name, intensity, amplitude, volume, parameters=names, defaults=defaults,
                                 label=name.split("_", 1)[1] + " (scout reference)")  # fmt: skip
    _registered = True


def unregister():
    global _registered
    from ..studio import unregister_python_function

    for name in _FUNCTIONS:
        unregister_python_function(name)
    _registered = False


def registered():
    return _registered
