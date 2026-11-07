"""Perfectly oriented and axially distributed particles of any shape (SI Sects. 6 and 7).

Lab frame: the detector plane is (x, y); q = (qx, qy, qz) with qz ~ 0 for small angles.
Particle frame: main axis = z_b.  `rot` maps particle-frame vectors to the lab frame
(q_b = rot^T q_lab).  Size distribution (Schulz-Zimm) is averaged inside shape.P_vec().
"""
import numpy as np
from .shapes import _gl


def euler_rotation(theta, phi, chi=0.0):
    """Particle->lab rotation: main axis at polar angle theta, azimuth phi from the lab z axis, then spin chi."""
    ct, st, cp, sp, cc, sc = np.cos(theta), np.sin(theta), np.cos(phi), np.sin(phi), np.cos(chi), np.sin(chi)
    Rz1 = np.array([[cc, -sc, 0], [sc, cc, 0], [0, 0, 1]])
    Ry = np.array([[ct, 0, st], [0, 1, 0], [-st, 0, ct]])
    Rz2 = np.array([[cp, -sp, 0], [sp, cp, 0], [0, 0, 1]])
    return Rz2 @ Ry @ Rz1


def _qstack(qx, qy, qz):
    qx, qy, qz = np.broadcast_arrays(np.asarray(qx, float), np.asarray(qy, float), np.asarray(qz, float))
    return np.stack([qx, qy, qz], -1)


def perfect(shape, qx, qy, qz=0.0, rot=None, kind='P'):
    """Pattern of perfectly aligned particles (rot = particle->lab rotation; None = body axes = lab axes)."""
    q = _qstack(qx, qy, qz)
    if rot is not None:
        q = q @ np.asarray(rot)            # (R^T q) as row vectors: q_b = q_lab @ R
    return shape.P_vec(q) if kind == 'P' else shape.F_vec(q)


def _frame(director):
    d = np.asarray(director, float); d = d / np.linalg.norm(d)
    t = np.array([1.0, 0, 0]) if abs(d[0]) < 0.9 else np.array([0.0, 1, 0])
    e1 = np.cross(d, t); e1 /= np.linalg.norm(e1)
    e2 = np.cross(d, e1)
    return d, e1, e2


def axial(shape, qx, qy, h, qz=0.0, director=(1, 0, 0), n_delta=64, n_psi=32, spin="uniform", n_spin=16,
          delta_max=np.pi / 2, kind='P'):
    """Pattern for particles whose main axis has the distribution h(delta) about `director` (uniform in the
    azimuth psi about the director).  spin: 'uniform' (free rotation about the main axis, n_spin angles),
    or a float (fixed spin angle in rad, body x-axis in the plane of axis and director) -- axisymmetric
    shapes need no spin average."""
    q = _qstack(qx, qy, qz)
    shp = q.shape[:-1]
    qf = q.reshape(-1, 3)
    d, e1, e2 = _frame(director)
    dl, wd = _gl(n_delta, 0.0, delta_max)
    wd = wd * h(dl) * np.sin(dl)
    wd /= wd.sum()
    psis = (np.arange(n_psi) + 0.5) * 2 * np.pi / n_psi
    axisym = getattr(shape, "_axisym", None)
    if axisym is None:
        axisym = hasattr(shape, "P_fixed") and not type(shape).__name__ in ("TriaxialEllipsoid", "Parallelepiped", "Cube")
    chis = np.array([0.0]) if (axisym) else ((np.arange(n_spin) + 0.5) * 2 * np.pi / n_spin if spin == "uniform" else np.array([float(spin)]))
    out = np.zeros(qf.shape[0])
    fn = shape.P_vec if kind == 'P' else shape.F_vec
    for a, wa in zip(dl, wd):
        for ps in psis:
            n = np.cos(a) * d + np.sin(a) * (np.cos(ps) * e1 + np.sin(ps) * e2)
            u = np.cross(n, d)
            if np.linalg.norm(u) < 1e-9:
                u = e1
            u = u / np.linalg.norm(u)
            v = np.cross(n, u)
            for ch in chis:
                xb = np.cos(ch) * u + np.sin(ch) * v
                yb = np.cross(n, xb)
                qb = np.stack([qf @ xb, qf @ yb, qf @ n], -1)
                out += wa / psis.size / chis.size * fn(qb)
    return out.reshape(shp)
