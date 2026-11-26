# SPDX-License-Identifier: GPL-3.0-or-later
"""Orientation handling for the polyhedra form factors: oriented complex amplitudes and isotropic averages.

F(q) of a body is a function of q in the body frame. For a particle with orientation matrix R (body -> lab) and a
scattering vector q_lab, F = F_body(R^T q_lab). Because the density is real, F(-q) = conj F(q): |F|^2 and Re F are
even in q, Im F is odd. The isotropic (powder) averages are therefore

    <I>(q) = <|F|^2>  and  <F>(q) = <Re F>      (both real; <Im F> vanishes),

and the integral over the unit sphere is done over the upper hemisphere only.
"""

from __future__ import annotations

import math

import numpy as np

__all__ = ["rotation_zyz", "oriented_amplitude", "orientational_average", "sphere_rule"]


def rotation_zyz(alpha=0.0, beta=0.0, gamma=0.0):
    """Rotation matrix R = Rz(alpha) Ry(beta) Rz(gamma) (body -> lab), angles in radians."""
    ca, sa, cb, sb, cg, sg = math.cos(alpha), math.sin(alpha), math.cos(beta), math.sin(beta), math.cos(gamma), math.sin(gamma)
    rz = lambda c, s: np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    ry = np.array([[cb, 0.0, sb], [0.0, 1.0, 0.0], [-sb, 0.0, cb]])
    return rz(ca, sa) @ ry @ rz(cg, sg)


def oriented_amplitude(shape, q_lab, rotation=None):
    """Complex amplitude F(q) of a particle with orientation ``rotation`` (3x3, body -> lab; None: body frame) for
    scattering vectors ``q_lab`` of shape (..., 3). Real or complex q."""
    q = np.asarray(q_lab)
    flat = q.reshape(-1, 3)
    if rotation is not None:
        flat = flat @ np.asarray(rotation, float)  # rows: (R^T q)^T = q^T R
    return shape.ff(flat).reshape(q.shape[:-1])


def sphere_rule(n_mu, n_phi):
    """Directions (n, 3) and weights (summing to 1) for the average over the unit sphere, using its upper
    hemisphere: Gauss-Legendre in cos(theta) in (0, 1] times the trapezoidal rule in phi."""
    x, w = np.polynomial.legendre.leggauss(int(n_mu))
    mu = 0.5 * (x + 1.0)
    wm = 0.5 * w
    phi = 2 * math.pi * (np.arange(int(n_phi)) + 0.5) / n_phi
    st = np.sqrt(1.0 - mu**2)
    d = np.empty((len(mu), len(phi), 3))
    d[..., 0] = st[:, None] * np.cos(phi)[None, :]
    d[..., 1] = st[:, None] * np.sin(phi)[None, :]
    d[..., 2] = mu[:, None]
    wt = np.repeat(wm[:, None] / n_phi, n_phi, axis=1)
    return d.reshape(-1, 3), wt.ravel()


def _nodes_for(qr, safety):
    n_mu = int(math.ceil(1.2 * safety * qr)) + 16
    n_phi = int(math.ceil(2.4 * safety * qr)) + 16
    return n_mu, n_phi


def orientational_average(shape, q, safety=1.0, chunk=250_000):
    """Isotropic averages (<|F|^2>, <F>) of ``shape`` for scattering magnitudes ``q`` (1d array); both real.

    The number of nodes grows with q*R (R: radius of the enclosing sphere); ``safety`` scales it (the default
    reproduces the result to about 1e-12 relative accuracy; see the tests)."""
    q = np.atleast_1d(np.asarray(q, float))
    I = np.empty(q.shape)
    A = np.empty(q.shape)
    keys = [_nodes_for(abs(qi) * shape.radius, safety) for qi in q]
    for key in sorted(set(keys)):
        idx = [i for i, k in enumerate(keys) if k == key]
        dirs, w = sphere_rule(*key)
        per = max(1, chunk // len(w))
        for s in range(0, len(idx), per):
            sub = idx[s : s + per]
            Q = (q[sub][:, None, None] * dirs[None, :, :]).reshape(-1, 3)
            F = shape.ff(Q).reshape(len(sub), len(w))
            I[sub] = (np.abs(F) ** 2) @ w
            A[sub] = F.real @ w
    return I, A
