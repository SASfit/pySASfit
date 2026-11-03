"""Optional 3D view of the reconstructed structure with PyVista (``pip install pyvista``).

    from pysasfit.morph.view3d import show_3d
    show_3d(res.matrix, res.voxel_size, phase="pore")

The interface is drawn as a smoothed iso-surface (marching cubes) of the binary
matrix; optionally the largest pore cluster in a second colour (as SAXSMorph).
PyVista opens its own window (VTK render window); without PyVista an
ImportError with installation hint is raised. :func:`surface_mesh` returns the
mesh for export (``mesh.save("structure.stl")``).
"""

from __future__ import annotations

import numpy as np

__all__ = ["have_pyvista", "pyvista_problem", "surface_mesh", "show_3d"]


def pyvista_problem():
    """None if PyVista (and VTK) import in this Python, else a text with the real error and the Python used.

    PyVista can be installed and still fail to import, e.g. when VTK is missing or its DLLs do not load, or
    when it was installed into another Python than the one running pySASfit."""
    import sys
    import traceback

    try:
        import pyvista  # noqa: F401
        return None
    except Exception as e:  # noqa: BLE001 -- also a broken VTK install
        tb = "".join(traceback.format_exception(type(e), e, e.__traceback__)[-6:])
        return (f"import pyvista failed: {type(e).__name__}: {e}\n\n"
                f"Python: {sys.executable}\n(version {sys.version.split()[0]}, {sys.platform})\n\n{tb}")


def have_pyvista():
    return pyvista_problem() is None


def _pv():
    try:
        import pyvista as pv
    except Exception as e:  # noqa: BLE001
        raise ImportError("the 3D view needs PyVista: pip install pyvista") from e
    return pv


def surface_mesh(mask, a=1.0, smooth=20):
    """Iso-surface (level 0.5) of a binary matrix as a PyVista PolyData (padded so the box faces close)."""
    pv = _pv()
    m = np.pad(np.asarray(mask, np.float32), 1)
    grid = pv.ImageData(dimensions=m.shape, spacing=(a, a, a), origin=(-a, -a, -a))
    grid.point_data["phase"] = m.ravel(order="F")
    surf = grid.contour([0.5], scalars="phase")
    if smooth and surf.n_points:
        surf = surf.smooth_taubin(n_iter=smooth, pass_band=0.1)
    return surf


def largest_cluster(mask):
    from scipy import ndimage

    lab, n = ndimage.label(mask)
    if n == 0:
        return np.zeros_like(mask, bool)
    sizes = np.bincount(lab.ravel())
    sizes[0] = 0
    return lab == sizes.argmax()


def show_3d(mat, a=1.0, phase="pore", highlight_largest=False, color="#2E8B8B", color2="#C8246E",
            smooth=20, title="SASmorph", block=False):
    """Open a PyVista window with the iso-surface of ``phase`` ('pore' = True voxels, 'solid')."""
    pv = _pv()
    mask = np.asarray(mat, bool)
    if phase == "solid":
        mask = ~mask
    pl = None
    if not block:
        try:            # inside a Qt application: a non-blocking window (pip install pyvistaqt)
            from pyvistaqt import BackgroundPlotter
            pl = BackgroundPlotter(title=title)
        except Exception:  # noqa: BLE001
            pl = None
    background = pl is not None
    if pl is None:
        pl = pv.Plotter(title=title)
    if highlight_largest:
        big = largest_cluster(mask)
        pl.add_mesh(surface_mesh(mask & ~big, a, smooth), color=color, smooth_shading=True)
        pl.add_mesh(surface_mesh(big, a, smooth), color=color2, smooth_shading=True)
    else:
        pl.add_mesh(surface_mesh(mask, a, smooth), color=color, smooth_shading=True)
    n = mask.shape[0]
    pl.add_mesh(pv.Box(bounds=(0, n * a, 0, n * a, 0, n * a)).outline(), color="grey")
    pl.show_axes()
    if not background:
        pl.show()          # own VTK window; returns when it is closed
    return pl
