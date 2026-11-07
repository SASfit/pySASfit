"""Lattice factor Z(q) for (para)crystalline arrangements and the decoupling approximation
I(q) = rho N <P> [ 1 + beta (Z-1) G ],   beta = <F>^2/<P>   (Eq. in the paper)."""
import numpy as np


def rotation(axis, angle):
    a = np.asarray(axis, float); a = a / np.linalg.norm(a)
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * K @ K


def cubic(a, kind='sc'):
    """Primitive cell matrix (columns = lattice vectors) for sc / bcc / fcc."""
    if kind == 'sc':
        return a * np.eye(3)
    if kind == 'bcc':
        return a * 0.5 * np.array([[-1, 1, 1], [1, -1, 1], [1, 1, -1]], float).T
    if kind == 'fcc':
        return a * 0.5 * np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]], float).T
    raise ValueError(kind)


class Lattice:
    """M: 3x3 matrix whose columns are the real-space cell vectors; Rot: orientation of the lattice;
    width: Gaussian peak width (1/length, std dev of each peak); dw: Debye-Waller sigma_d (length)."""

    def __init__(self, M, Rot=None, width=0.002, dw=0.0, hmax=6):
        self.M = np.asarray(M, float)
        self.Rot = np.eye(3) if Rot is None else np.asarray(Rot, float)
        self.width, self.dw = float(width), float(dw)
        self.v0 = abs(np.linalg.det(self.M))
        B = 2 * np.pi * np.linalg.inv(self.Rot @ self.M).T  # columns: reciprocal vectors
        h = np.arange(-hmax, hmax + 1)
        H = np.array(np.meshgrid(h, h, h, indexing='ij')).reshape(3, -1)
        H = H[:, np.any(H != 0, axis=0)]
        self.hkl = H
        self.G = B @ H

    def Z(self, qx, qy, qz=0.0, qmax=None):
        qx, qy, qz = np.broadcast_arrays(np.asarray(qx, float), np.asarray(qy, float), np.asarray(qz, float))
        out = np.zeros(qx.shape)
        s2 = self.width ** 2
        norm = (2 * np.pi) ** 3 / self.v0 / ((2 * np.pi * s2) ** 1.5)
        for g in self.G.T:
            d2 = (qx - g[0]) ** 2 + (qy - g[1]) ** 2 + (qz - g[2]) ** 2
            out += np.exp(-0.5 * d2 / s2)
        Z = norm * out
        if self.dw:
            Z = Z  # peaks keep unit weight; DW enters through G(q)
        return Z

    def G_dw(self, q):
        return np.exp(-(np.asarray(q) * self.dw) ** 2)

    def I(self, shape, qx, qy, qz=0.0):
        """Decoupling approximation (orientation-averaged particles), normalised per particle."""
        q = np.sqrt(np.asarray(qx) ** 2 + np.asarray(qy) ** 2 + np.asarray(qz) ** 2)
        P = shape.P(q.ravel()).reshape(q.shape)
        beta = shape.beta(q.ravel()).reshape(q.shape)
        return P * (1 + beta * (self.Z(qx, qy, qz) - 1) * self.G_dw(q))


def _powder_Z(self, q):
    """Powder (isotropically oriented crystallites) lattice factor: the spherical average of the
    3D Gaussian peaks, <exp(-|q-G|^2/2s^2)>_Omega = exp(-(q-G)^2/2s^2) (1-exp(-2a))/(2a), a = qG/s^2,
    summed over all reciprocal vectors G; normalised so that <Z> -> 1 (G=0 excluded)."""
    q = np.atleast_1d(np.asarray(q, float))
    s2 = self.width ** 2
    norm = (2 * np.pi) ** 3 / self.v0 / ((2 * np.pi * s2) ** 1.5)
    Gm = np.linalg.norm(self.G, axis=0)
    # merge equivalent |G| (rounded) to save work
    uniq, inv = np.unique(np.round(Gm, 10), return_inverse=True)
    mult = np.bincount(inv)
    out = np.zeros(q.shape)
    for g, m in zip(uniq, mult):
        a = q * g / s2
        with np.errstate(all='ignore'):
            f = np.exp(-0.5 * (q - g) ** 2 / s2) * np.where(a > 1e-8, -np.expm1(-2 * a) / (2 * a), 1.0)
        out += m * f
    return norm * out


def _powder_I(self, shape, q):
    q = np.atleast_1d(np.asarray(q, float))
    P, beta = shape.P(q), shape.beta(q)
    return P * (1 + beta * (self.Z_powder(q) - 1) * self.G_dw(q))


Lattice.Z_powder = _powder_Z
Lattice.I_powder = _powder_I
