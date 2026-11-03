"""Output of a reconstruction.

Curves (tab-separated text, as SAXSMorph): ``<base>_newiq.txt`` (extrapolated I(q)),
``<base>_gamma.txt``, ``<base>_gr.txt``, ``<base>_spec.txt`` (f(k) and P(k)),
``<base>_iq_model.txt`` (data and level-cut model), ``<base>_iq_voxel.txt`` and
``<base>_log.txt`` (report and parameters).

Voxel matrix: VTK legacy structured points (ParaView, PyVista; no dependency),
HDF5 (h5py; matrix plus all curves and parameters), TIFF stack (tifffile or
Pillow; Fiji/ImageJ, Avizo, Dragonfly), NumPy ``.npy``.
"""

from __future__ import annotations

import json
import os

import numpy as np

__all__ = ["write_curves", "write_vtk", "write_hdf5", "write_tiff_stack", "write_npy", "write_matrix",
           "MATRIX_FORMATS", "write_curves_multi", "write_hdf5_multi", "write_labels"]

#: file filter -> extension for the GUI
MATRIX_FORMATS = {
    "VTK structured points (*.vtk)": ".vtk",
    "HDF5 with curves and parameters (*.h5)": ".h5",
    "TIFF stack (*.tif)": ".tif",
    "NumPy array (*.npy)": ".npy",
}


def _save(path, cols, header):
    np.savetxt(path, np.column_stack(cols), delimiter="\t", header="\t".join(header), fmt="%.8g")
    return path


def write_curves(res, base):
    """Write all curves of ``res`` next to ``base`` (path without extension). Returns the paths."""
    c = res.curve
    out = [
        _save(base + "_newiq.txt", [c.q, c.I], ["q", "I_extrapolated"]),
        _save(base + "_gamma.txt", [res.r, res.gamma], ["r", "gamma"]),
        _save(base + "_gr.txt", [res.r, res.g], ["r", "g"]),
        _save(base + "_spec.txt", [res.k, res.f, res.Pk], ["k", "f(k)", "P(k)"]),
    ]
    cols = [res.q_data, res.I_data, res.I_levelcut]
    head = ["q", "I_data", "I_levelcut"]
    if res.sigma_data is not None:
        cols.insert(2, res.sigma_data)
        head.insert(2, "sigma")
    out.append(_save(base + "_iq_model.txt", cols, head))
    if res.I_voxel is not None:
        out.append(_save(base + "_iq_voxel.txt", [res.q_voxel, res.I_voxel], ["q", "I_voxel"]))
    with open(base + "_log.txt", "w", encoding="utf-8") as fh:
        fh.write(res.report() + "\n\nparameters:\n" + json.dumps(res.params.to_dict(), indent=1) + "\n")
    out.append(base + "_log.txt")
    return out


def write_vtk(mat, path, a=1.0):
    """Binary legacy VTK, STRUCTURED_POINTS, unsigned char 'phase' (1 = pore) on the voxel centres."""
    mat = np.asarray(mat)
    nx, ny, nz = mat.shape
    with open(path, "wb") as fh:
        fh.write(b"# vtk DataFile Version 3.0\nSASmorph two-phase structure (1 = pore)\nBINARY\n")
        fh.write(f"DATASET STRUCTURED_POINTS\nDIMENSIONS {nx} {ny} {nz}\nORIGIN 0 0 0\n"
                 f"SPACING {a:.8g} {a:.8g} {a:.8g}\nPOINT_DATA {mat.size}\n"
                 "SCALARS phase unsigned_char 1\nLOOKUP_TABLE default\n".encode())
        # VTK: x varies fastest -> Fortran order of (x, y, z)
        fh.write(np.asarray(mat, np.uint8).ravel(order="F").tobytes())
        fh.write(b"\n")
    return path


def write_hdf5(res, path, include_field=False):
    import h5py

    with h5py.File(path, "w") as h:
        h.attrs["creator"] = "pysasfit.morph (SASmorph)"
        h.attrs["parameters"] = json.dumps(res.params.to_dict())
        h.attrs["report"] = res.report()
        if res.matrix is not None:
            d = h.create_dataset("matrix", data=np.asarray(res.matrix, np.uint8), compression="gzip", chunks=True)
            d.attrs["voxel_size"] = res.voxel_size
            d.attrs["meaning"] = "1 = phase 2 (pore), 0 = solid"
        if include_field and res.psi is not None:
            h.create_dataset("field", data=res.psi, compression="gzip", chunks=True)
        g = h.create_group("curves")
        for name, arr in (("q_extrapolated", res.curve.q), ("I_extrapolated", res.curve.I), ("r", res.r),
                          ("gamma", res.gamma), ("g", res.g), ("k", res.k), ("f", res.f), ("Pk", res.Pk),
                          ("q_data", res.q_data), ("I_data", res.I_data), ("I_levelcut", res.I_levelcut)):
            g.create_dataset(name, data=np.asarray(arr))
        if res.I_voxel is not None:
            g.create_dataset("q_voxel", data=res.q_voxel)
            g.create_dataset("I_voxel", data=res.I_voxel)
        s = h.create_group("statistics")
        for k, v in _flat(res.statistics).items():
            s.attrs[k] = v
        s.attrs["S_over_V_porod"] = res.S_over_V_porod
        s.attrs["invariant"] = res.invariant
        s.attrs["porod_K"] = res.curve.K
    return path


def _flat(d, prefix=""):
    out = {}
    for k, v in d.items():
        if isinstance(v, dict):
            out.update(_flat(v, prefix + k + "."))
        elif isinstance(v, (tuple, list)):
            out[prefix + k] = np.asarray(v)
        else:
            out[prefix + k] = v
    return out


def write_tiff_stack(mat, path, a=None):
    """One 8-bit page per z slice (0 / 255)."""
    data = np.ascontiguousarray(np.moveaxis(np.asarray(mat, np.uint8) * 255, 2, 0))
    try:
        import tifffile

        kw = {}
        if a:
            kw = {"imagej": True, "resolution": (1.0 / a, 1.0 / a), "metadata": {"spacing": a, "axes": "ZYX"}}
        tifffile.imwrite(path, data, **kw)
        return path
    except ImportError:
        pass
    try:
        from PIL import Image
    except ImportError as e:
        raise ImportError("TIFF export needs tifffile or Pillow (pip install tifffile)") from e
    pages = [Image.fromarray(s) for s in data]
    pages[0].save(path, save_all=True, append_images=pages[1:], compression="tiff_deflate")
    return path


def write_npy(mat, path):
    np.save(path, np.asarray(mat, bool))
    return path


def write_matrix(res, path):
    """Write the voxel matrix in the format given by the extension (.vtk, .h5/.hdf5, .tif/.tiff, .npy)."""
    if hasattr(res, "labels"):                       # multi-phase result
        return write_labels(res, path)
    ext = os.path.splitext(path)[1].lower()
    if res.matrix is None:
        raise ValueError("no voxel matrix: run the reconstruction first")
    if ext == ".vtk":
        return write_vtk(res.matrix, path, res.voxel_size)
    if ext in (".h5", ".hdf5", ".hdf", ".nxs"):
        return write_hdf5(res, path)
    if ext in (".tif", ".tiff"):
        return write_tiff_stack(res.matrix, path, res.voxel_size)
    if ext == ".npy":
        return write_npy(res.matrix, path)
    raise ValueError(f"unknown matrix format {ext!r}")


# ---------------------------------------------------------------------- multi-phase results
def write_curves_multi(res, base):
    """Curves of a :class:`pysasfit.morph.multiphase.MultiResult`: common g(r) and f(k), and per curve
    ``<base>_<name>_newiq/_gamma/_iq_model/_iq_voxel.txt``; ``<base>_log.txt``."""
    import re

    out = [_save(base + "_gr.txt", [res.r, res.g], ["r", "g"]),
           _save(base + "_spec.txt", [res.k, res.f, res.Pk], ["k", "f(k)", "P(k)"])]
    for c in res.curves:
        tag = base + "_" + re.sub(r"[^\w.-]+", "_", c.data.name)
        out.append(_save(tag + "_newiq.txt", [c.curve.q, c.curve.I], ["q", "I_extrapolated"]))
        out.append(_save(tag + "_gamma.txt", [res.r, c.gamma, c.G], ["r", "gamma_data", "gamma_model"]))
        cols, head = [c.q_data, c.I_data, c.I_model], ["q", "I_data", "I_model"]
        if c.sigma_data is not None:
            cols.insert(2, c.sigma_data)
            head.insert(2, "sigma")
        out.append(_save(tag + "_iq_model.txt", cols, head))
        if c.I_voxel is not None:
            out.append(_save(tag + "_iq_voxel.txt", [c.q_voxel, c.I_voxel], ["q", "I_voxel"]))
    with open(base + "_log.txt", "w", encoding="utf-8") as fh:
        fh.write(res.report() + "\n\nparameters:\n" + json.dumps(res.params.to_dict(), indent=1) + "\n")
        fh.write("\ncurves:\n" + json.dumps([{"name": c.data.name, "path": c.data.path, "weight": c.data.weight,
                                              "sld": [float(x) for x in c.data.sld]} for c in res.curves],
                                            indent=1) + "\n")
    out.append(base + "_log.txt")
    return out


def write_hdf5_multi(res, path, include_field=False):
    import h5py

    with h5py.File(path, "w") as h:
        h.attrs["creator"] = "pysasfit.morph (SASmorph, multi-phase)"
        h.attrs["parameters"] = json.dumps(res.params.to_dict())
        h.attrs["report"] = res.report()
        if res.labels is not None:
            d = h.create_dataset("labels", data=np.asarray(res.labels, np.uint8), compression="gzip", chunks=True)
            d.attrs["voxel_size"] = res.voxel_size
            d.attrs["phases"] = json.dumps(list(res.params.names))
            d.attrs["levels"] = np.asarray(res.levels)
        if include_field and res.psi is not None:
            h.create_dataset("field", data=res.psi, compression="gzip", chunks=True)
        h.create_dataset("r", data=res.r)
        h.create_dataset("g", data=res.g)
        h.create_dataset("k", data=res.k)
        h.create_dataset("f", data=res.f)
        for c in res.curves:
            g = h.create_group("curves/" + c.data.name.replace("/", "_"))
            g.attrs["sld"] = np.asarray(c.data.sld, float)
            g.attrs["invariant"] = c.invariant
            g.attrs["norm"] = c.norm
            for name, arr in (("q_data", c.q_data), ("I_data", c.I_data), ("I_model", c.I_model),
                              ("gamma", c.gamma), ("gamma_model", c.G)):
                g.create_dataset(name, data=np.asarray(arr))
            if c.I_voxel is not None:
                g.create_dataset("q_voxel", data=c.q_voxel)
                g.create_dataset("I_voxel", data=c.I_voxel)
        st = h.create_group("statistics")
        for i, s in enumerate(res.statistics.get("phases", [])):
            for k, v in _flat(s).items():
                st.attrs[f"{res.params.names[i]}.{k}"] = v
        for (i, j), v in res.statistics.get("interfaces", {}).items():
            st.attrs[f"S/V {res.params.names[i]}|{res.params.names[j]}"] = v
    return path


def write_labels(res, path):
    """Label matrix of a multi-phase result: .vtk (phase index), .h5, .tif (grey levels), .npy."""
    ext = os.path.splitext(path)[1].lower()
    lab = res.labels
    if lab is None:
        raise ValueError("no voxel matrix: run the reconstruction first")
    if ext == ".vtk":
        return write_vtk(lab, path, res.voxel_size)
    if ext in (".h5", ".hdf5", ".hdf", ".nxs"):
        return write_hdf5_multi(res, path)
    if ext in (".tif", ".tiff"):
        n = max(1, int(lab.max()))
        step = 255 // n
        data = np.ascontiguousarray(np.moveaxis(np.asarray(lab, np.uint8) * step, 2, 0))
        try:
            import tifffile

            tifffile.imwrite(path, data, imagej=True, resolution=(1 / res.voxel_size, 1 / res.voxel_size),
                             metadata={"spacing": res.voxel_size, "axes": "ZYX"})
        except ImportError:
            from PIL import Image
            pages = [Image.fromarray(s) for s in data]
            pages[0].save(path, save_all=True, append_images=pages[1:], compression="tiff_deflate")
        return path
    if ext == ".npy":
        np.save(path, np.asarray(lab, np.int8))
        return path
    raise ValueError(f"unknown matrix format {ext!r}")
