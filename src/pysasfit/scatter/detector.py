"""Detector pixel -> scattering vector (Eq. 1 of the paper, generalised for grazing incidence).

Laboratory frame: x along the (projected) beam, y lateral (horizontal), z vertical.
k_i = k (cos ai, 0, -sin ai)   (ai = incidence angle; ai = 0 for transmission SAS)
Detector plane at x = D, pixel at (y, z) relative to the direct beam position.
"""
import numpy as np


def qmap(y, z, D, wavelength, alpha_i=0.0):
    """Return (qx, qy, qz) in 1/length for detector coordinates y, z (same length unit as D)."""
    y, z = np.broadcast_arrays(np.asarray(y, float), np.asarray(z, float))
    k = 2 * np.pi / wavelength
    # reference (direct beam) at z=0; with alpha_i the beam hits the detector at -D tan(ai) -> shift
    z = z + D * np.tan(alpha_i) * 0.0
    rho = np.sqrt(D * D + y * y + z * z)
    kf = k * np.stack([D / rho, y / rho, z / rho])
    ki = k * np.array([np.cos(alpha_i), 0.0, -np.sin(alpha_i)]).reshape(3, *([1] * y.ndim))
    q = kf - ki
    return q[0], q[1], q[2]


def exit_angles(y, z, D):
    """In-plane (2theta_f) and exit (alpha_f) angles of detector pixels."""
    y, z = np.broadcast_arrays(np.asarray(y, float), np.asarray(z, float))
    return np.arctan2(y, D), np.arctan2(z, np.hypot(D, y))


def pixel_grid(n_y, n_z, pixel, center=None):
    """Pixel coordinates (y, z) for an n_y x n_z detector, beam/center at `center` (pixels)."""
    cy, cz = center if center is not None else ((n_y - 1) / 2, (n_z - 1) / 2)
    Y, Z = np.meshgrid((np.arange(n_y) - cy) * pixel, (np.arange(n_z) - cz) * pixel, indexing='ij')
    return Y, Z
