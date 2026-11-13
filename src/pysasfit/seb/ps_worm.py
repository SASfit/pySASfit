"""Form factor of the worm-like chain with excluded volume: Pedersen & Schurtenberger, Macromolecules 29 (1996)
7602 ("method 3" with excluded volume), with the corrections of Chen, Butler & Magid, Langmuir 22 (2006) 6539.

NumPy port (vectorised in q) of the C implementation in sasmodels (models/lib/wrc_cyl.c, public domain), same
constants and branches. L contour length, b Kuhn length.
"""
from __future__ import annotations

import numpy as np

__all__ = ["sk_wr", "rg2_ev"]


def rg2_ev(L, b):
    """Rg^2 with the expansion factor alpha^2 = [1 + (x/3.12)^2 + (x/8.67)^3]^(0.176/3), x = L/b (PS eq 15)."""
    x = L / b
    alphasq = (1.0 + x * x * (1.027284681130835e-01 + 1.534414548417740e-03 * x)) ** 5.866666666666667e-02
    return alphasq * L * b / 6.0


def _rg2_short(L, b):
    r = b / L
    return rg2_ev(L, b) * (1.0 + r * (-1.5 + r * (1.5 + r * 0.75 * np.expm1(-2.0 / r))))


def _w(x):
    return 0.5 + 0.5 * np.tanh((x - 1.523) / 0.1477)


def _sdebye(x):
    x = np.asarray(x, dtype=float)
    out = np.empty_like(x)
    lo = x < 0.25
    xl = x[lo]
    out[lo] = (((((1. / 75600.) * xl + 0.) * xl + 1. / 60) * xl + 1. / 15.) * xl + 1.) / \
        (((((1. / 5040.) * xl + 1. / 180.) * xl + 1. / 15.) * xl + 2. / 5.) * xl + 1.)
    xh = x[~lo]
    out[~lo] = 2. * (np.expm1(-xh) + xh) / (xh * xh)
    return out


def _sexv(q, L, b):
    C1, C2, C3, miu = 1.22, 0.4288, -1.651, 0.585
    qr = q * np.sqrt(rg2_ev(L, b))
    qr_miu = qr ** (-1.0 / miu)
    w = _w(qr)
    return _sdebye(qr * qr) * (1.0 - w) + w * ((C3 * qr_miu + C2) * qr_miu + C1) * qr_miu


def _sexv_new(q, L, b, ref=None):
    ref = L if ref is None else ref
    qr = q * np.sqrt(rg2_ev(L, b))
    qr2 = qr * qr
    C = 3.06 * (L / b) ** -0.44 if ref / b > 10.0 else 1.0
    t9 = C * b / L * (4.0 - np.exp(-qr2) * (11.0 + 7.0 / qr2) + 7.0 / qr2) / 15.0
    s0 = _sexv(q, L, b)
    sr = _sexv(q, ref, b)
    qdel = (_sexv(q * 1.05, ref, b) - sr) / (q * 0.05)     # branch chosen at the reference length
    w = _w(qr)
    return np.where(qdel < 0, t9 + s0, t9 + _sdebye(qr2) * (1.0 - w))


def _a_long(q, L, b, ref=None):
    ref = L if ref is None else ref
    p1, p2, q0 = 4.12, 4.42, 3.1
    C1, C2, C3, C4, C5, miu = 1.22, 0.4288, -1.651, 1.523, 0.1477, 0.585
    C = 3.06 * (L / b) ** -0.44 if ref / b > 10.0 else 1.0
    r2 = rg2_ev(L, b)
    r = np.sqrt(r2)
    qr_b = q0 * r / b
    qr_b_sq = qr_b * qr_b
    qr_b_4 = qr_b_sq * qr_b_sq
    qr_b_miu = qr_b ** (-1.0 / miu)
    em1 = np.expm1(-qr_b_sq)
    sech2 = 1.0 / np.cosh((qr_b - C4) / C5) ** 2
    w = _w(qr_b)
    t1 = q0 ** (1.0 + p1 + p2) / (b * (p1 - p2))
    t2 = C / (15.0 * L) * (14.0 * b * b * em1 / (q0 * qr_b_sq) + 2.0 * q0 * r2 * np.exp(-qr_b_sq) * (11.0 + 7.0 / qr_b_sq))
    t11 = ((C3 * qr_b_miu + C2) * qr_b_miu + C1) * qr_b_miu
    t3 = r * sech2 / (2. * C5) * t11
    t4 = r * (em1 + qr_b_sq) * sech2 / (C5 * qr_b_4)
    t5 = -4.0 * r * qr_b * em1 / qr_b_4 * (1.0 - w)
    t10 = 2.0 * (em1 + qr_b_sq) / qr_b_4 * (1.0 - w)
    t6 = 4.0 * b / q0 * t10
    t7 = r * ((-3.0 * C3 * qr_b_miu - 2.0 * C2) * qr_b_miu - 1.0 * C1) * qr_b_miu / (miu * qr_b)
    t9 = C * b / L * (4.0 - np.exp(-qr_b_sq) * (11.0 + 7.0 / qr_b_sq) + 7.0 / qr_b_sq) / 15.0
    t12 = b * b * np.pi / (L * q0 * q0) + t2 + t3 - t4 + t5 - t6 + t7 * w
    t13 = -b * np.pi / (L * q0) + t9 + t10 + t11 * w
    a1 = q0 ** p1 * t13 - t1 * q0 ** (-p2) * (t12 + b * p1 / q0 * t13)
    a2 = t1 * q0 ** (-p1) * (t12 + b * p1 / q0 * t13)
    return a1 * (q * b) ** (-p1) + a2 * (q * b) ** (-p2) + np.pi / (q * L)


def _short(r2, exp_qr_b, L, b, p1s, p2s, q0):
    qr2 = q0 * q0 * r2
    b3 = b * b * b
    q0p = q0 ** (-4.0 + p1s)
    return 1.0 / (L * r2 * r2) * b / exp_qr_b * q0p * (
        8.0 * b3 * L - 8.0 * b3 * exp_qr_b * L + 2.0 * b3 * exp_qr_b * L * p2s - 2.0 * b * exp_qr_b * L * p2s * qr2
        + 4.0 * b * exp_qr_b * L * qr2 - 2.0 * b3 * L * p2s + 4.0 * b * L * qr2 - np.pi * exp_qr_b * qr2 * q0 * r2
        + np.pi * exp_qr_b * p2s * qr2 * q0 * r2)


def _a_short(q, L, b, q0):
    p1s, p2s = 5.36, 5.62
    r2 = _rg2_short(L, b)
    e = np.exp(r2 * (q0 / b) ** 2)
    a1 = _short(r2, e, L, b, p1s, p2s, q0) / (p1s - p2s)
    a2 = -_short(r2, e, L, b, p2s, p1s, q0) / (p1s - p2s)
    return a1 * (q * b) ** (-p1s) + a2 * (q * b) ** (-p2s) + np.pi / (q * L)


def sk_wr(q, L, b, ref=None):
    """Normalised form factor (F(0) = 1) of the worm-like chain with excluded volume (q array, L, b scalars).

    ``ref``: contour length at which the branches of the piecewise formula are chosen (default L) -- for
    derivatives in L, so that L +- h stay on the branches of L."""
    ref = L if ref is None else ref
    q = np.atleast_1d(np.asarray(q, dtype=float))
    q = np.maximum(q, 1e-12)
    out = np.empty_like(q)
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        if ref > 4 * b:
            lo = q * b <= 3.1
            if lo.any():
                out[lo] = _sexv_new(q[lo], L, b, ref)
            if (~lo).any():
                out[~lo] = _a_long(q[~lo], L, b, ref)
        else:
            q0s = max(1.9 / np.sqrt(_rg2_short(ref, b)), 3.0)
            lo = q * b <= q0s
            if lo.any():
                out[lo] = _sdebye((q[lo] ** 2) * _rg2_short(L, b))
            if (~lo).any():
                out[~lo] = _a_short(q[~lo], L, b, q0s)
    return out
