"""2D detector maps I(q_x, q_y) of a model with ψ-dependent (anisotropic) functions.

ψ is the direction of q on the detector, measured from the horizontal axis q_x towards q_y, in degrees (the
SASfit ψ parameter of the anisotropic plugins and of their Python ports). I(q) = I(-q), so ψ ∈ [0°, 180°) is
enough. Frame of pySASfit (docs/conventions.md): the map is shown as seen looking towards the sample, q_x to the
right, q_y up, ψ counter-clockwise (positive rotation about the beam).

The ψ parameter of a function is found from its Python port (``PortSpec.psi_index``; e.g. varphi p[11] for
shearflow_lin) or, for other functions, from a parameter named ``psi``. :func:`evaluate_polar` writes ψ into
that parameter of every contribution (form factor and structure factor) and evaluates the model once per ψ;
:func:`detector_map` then interpolates the polar grid onto a Cartesian (q_x, q_y) grid.

Example::

    from pysasfit.studio import ModelFunction, ScatteringContribution, map2d, ports
    ports.enable()
    c = ScatteringContribution("ff_sheared_cylinders__maier_saupe_", [20, 5, 200, 1, 2, 0, 0, 0, 5])
    model = ModelFunction([c])
    qx, qy, I = map2d.detector_map(model, qmax=0.1)
    map2d.plot(qx, qy, I)
"""

from __future__ import annotations

import math
import time

import numpy as np

__all__ = [
    "anisotropic_parameters",
    "detector_map",
    "evaluate_polar",
    "plot",
    "psi_index",
]

_PSI_NAMES = ("psi", "psi_deg")


def psi_index(f):
    """Index of the ψ (detector-angle) parameter of the function ``f`` (a SASfitFunction), or None."""
    if f is None:
        return None
    from . import ports

    try:
        return ports.get(f.name).psi_index
    except KeyError:
        pass
    names = [str(n).lower() for n in f.parameter_names]
    for cand in _PSI_NAMES:
        if cand in names:
            return names.index(cand)
    return None


def anisotropic_parameters(model):
    """``[(vector, index)]``: the ψ entries of the parameter vectors of all contributions of ``model``."""
    out = []
    for c in getattr(model, "contributions", ()):
        for role in ("ff", "sq"):
            f = c.function_for(role)
            k = psi_index(f)
            if k is None:
                continue
            if role == "ff" and c.sd is not None and k == c.distributed:
                continue
            vec = c.vector(role)
            if vec is not None and k < len(vec):
                out.append((vec, k))
    return out


def evaluate_polar(model, q, psi_deg, intensity=None):
    """I[ψ, q] of ``model`` (``intensity(q)``: what to evaluate, default ``model.intensity``)."""
    q = np.asarray(q, float)
    psi_deg = np.atleast_1d(np.asarray(psi_deg, float))
    fn = intensity or model.intensity
    targets = anisotropic_parameters(model)
    if not targets:
        row = np.asarray(fn(q), float)
        return np.tile(row, (psi_deg.size, 1))
    saved = [vec[k] for vec, k in targets]
    out = np.empty((psi_deg.size, q.size))
    try:
        for i, ps in enumerate(psi_deg):
            for vec, k in targets:
                vec[k] = ps
            out[i] = np.asarray(fn(q), float)
    finally:
        for (vec, k), v in zip(targets, saved):
            vec[k] = v
    return out


def detector_map(model, qmax, n=128, n_psi=36, n_q=None, intensity=None):
    """``(qx, qy, I)`` on an n × n grid over [-qmax, qmax]², ``I[iy, ix]``.

    The model is evaluated on a polar grid (``n_psi`` angles in [0°, 180°), ``n_q`` radii up to √2 qmax, default
    n) and interpolated bilinearly; isotropic models need one evaluation only."""
    from scipy.interpolate import RegularGridInterpolator

    qmax = float(qmax)
    if not qmax > 0:
        raise ValueError("qmax must be > 0")
    n = int(n)
    n_q = int(n_q or n)
    qx = np.linspace(-qmax, qmax, n)
    qy = qx.copy()
    QX, QY = np.meshgrid(qx, qy)
    Q = np.hypot(QX, QY)
    qr = np.linspace(0.0, math.sqrt(2.0) * qmax, n_q)
    qr[0] = qr[1] * 1e-3  # q = 0 is a removable singularity of many form factors
    Qc = np.maximum(Q, qr[0])
    if not anisotropic_parameters(model):
        row = np.asarray((intensity or model.intensity)(qr), float)
        return qx, qy, np.interp(Qc, qr, row)
    psi = np.arange(int(n_psi)) * (180.0 / int(n_psi))
    polar = evaluate_polar(model, qr, psi, intensity)
    psi_ext = np.append(psi, 180.0)
    polar_ext = np.vstack([polar, polar[:1]])
    PSI = np.mod(np.degrees(np.arctan2(QY, QX)), 180.0)
    f = RegularGridInterpolator(
        (psi_ext, qr), polar_ext, method="linear", bounds_error=False, fill_value=None
    )
    return qx, qy, f(np.stack([PSI, Qc], axis=-1))


def plot(qx, qy, intensity, ax=None, log=True, cmap="viridis", unit=""):
    """Show a detector map with matplotlib; returns the image."""
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm

    if ax is None:
        _, ax = plt.subplots(figsize=(5, 4.4), layout="constrained")
    data = np.asarray(intensity, float)
    norm = None
    if log:
        pos = data[data > 0]
        if pos.size:
            norm = LogNorm(vmin=max(pos.min(), pos.max() * 1e-8), vmax=pos.max())
            data = np.where(data > 0, data, np.nan)
    im = ax.imshow(data, origin="lower", extent=(qx[0], qx[-1], qy[0], qy[-1]), cmap=cmap, norm=norm,
                   interpolation="nearest", aspect="equal")  # fmt: skip
    ax.set_xlabel(f"q_x {unit}".strip())
    ax.set_ylabel(f"q_y {unit}".strip())
    ax.figure.colorbar(im, ax=ax, label="I(q)")
    return im


def timed_map(*args, **kw):
    """:func:`detector_map` and the time it took (s)."""
    t0 = time.perf_counter()
    out = detector_map(*args, **kw)
    return out, time.perf_counter() - t0
