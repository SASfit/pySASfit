"""HDF5 helpers of the pyTOOLbox viewer (no GUI): children of a group, description of an
object, slices of N-dimensional datasets, NeXus NXdata default plots.

Files are opened only for the duration of each call, so that other tools (the metadata
lister writing values, the instrument software) are not blocked by an open file.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import h5py
import numpy as np

__all__ = ["Child", "children", "describe", "kind_of", "read_slice", "nxdata", "find", "to_text",
           "is_numeric", "pixel_size", "pixel_aspect"]


@dataclass
class Child:
    name: str
    path: str
    kind: str            # group | dataset | softlink | externallink | broken
    info: str = ""       # shape and type, link target, NX_class
    is_group: bool = False   # a group, or a link to one (has members)


def _s(v):
    if isinstance(v, bytes):
        return v.decode("utf-8", errors="replace")
    if isinstance(v, np.ndarray) and v.dtype.kind in "SO" and v.size == 1:
        return _s(v.reshape(-1)[0])
    return v


def to_text(v, max_items=64):
    """Readable text of an attribute or a small value."""
    v = _s(v)
    if isinstance(v, np.ndarray):
        if v.size > max_items:
            flat = v.reshape(-1)[:max_items]
            return f"{v.shape} {v.dtype}: [" + ", ".join(str(_s(x)) for x in flat) + ", …]"
        if v.dtype.kind in "SO":
            return "[" + ", ".join(str(_s(x)) for x in v.reshape(-1)) + "]"
        return np.array2string(v, threshold=max_items, max_line_width=200)
    if isinstance(v, np.generic):
        return str(v.item())
    return str(v)


def is_numeric(dtype) -> bool:
    return dtype.kind in "biufc"


def kind_of(obj) -> str:
    if isinstance(obj, h5py.Group):
        return "group"
    if isinstance(obj, h5py.Dataset):
        return "dataset"
    return "other"


def _info(obj) -> str:
    if isinstance(obj, h5py.Dataset):
        sh = "scalar" if obj.shape == () else "×".join(str(n) for n in obj.shape)
        return f"{sh}  {obj.dtype}"
    nx = _s(obj.attrs.get("NX_class", ""))
    return str(nx) if nx else ""


def children(path, h5path="/") -> List[Child]:
    """Members of the group ``h5path`` in file ``path`` (in storage order)."""
    out = []
    with h5py.File(path, "r") as f:
        g = f[h5path]
        for name in g:
            p = (h5path.rstrip("/") + "/" + name) if h5path != "/" else "/" + name
            link = g.get(name, getlink=True)
            if isinstance(link, h5py.SoftLink):
                try:
                    tgt = g[name]
                    out.append(Child(name, p, "softlink", f"→ {link.path}  {_info(tgt)}",
                                     isinstance(tgt, h5py.Group)))
                except KeyError:
                    out.append(Child(name, p, "broken", f"→ {link.path} (missing)"))
                continue
            if isinstance(link, h5py.ExternalLink):
                try:
                    tgt = g[name]
                    out.append(Child(name, p, "externallink", f"→ {link.filename}:{link.path}",
                                     isinstance(tgt, h5py.Group)))
                except (KeyError, OSError):
                    out.append(Child(name, p, "broken", f"→ {link.filename}:{link.path} (not found)"))
                continue
            try:
                obj = g[name]
            except (KeyError, OSError) as e:
                out.append(Child(name, p, "broken", str(e)))
                continue
            out.append(Child(name, p, kind_of(obj), _info(obj), isinstance(obj, h5py.Group)))
    return out


def _filters(ds) -> str:
    parts = []
    if ds.compression:
        parts.append(f"{ds.compression}" + (f" ({ds.compression_opts})" if ds.compression_opts is not None else ""))
    if ds.shuffle:
        parts.append("shuffle")
    if ds.fletcher32:
        parts.append("fletcher32")
    if ds.scaleoffset is not None:
        parts.append(f"scaleoffset {ds.scaleoffset}")
    return ", ".join(parts) or "none"


def describe(path, h5path) -> Tuple[List[Tuple[str, str]], List[Tuple[str, str, str]]]:
    """(metadata rows [(key, value)], attributes [(name, value, type)]) of an object."""
    with h5py.File(path, "r") as f:
        obj = f[h5path]
        meta = [("File", str(path)), ("Path", obj.name), ("Name", obj.name.rsplit("/", 1)[-1] or "/")]
        if isinstance(obj, h5py.Dataset):
            meta += [("Kind", "dataset"), ("Type", str(obj.dtype)),
                     ("Shape", "scalar" if obj.shape == () else str(obj.shape)),
                     ("Elements", str(obj.size)), ("Storage", f"{obj.id.get_storage_size()} bytes"),
                     ("Chunks", str(obj.chunks) if obj.chunks else "contiguous"),
                     ("Filters", _filters(obj))]
            if obj.maxshape != obj.shape:
                meta.append(("Max. shape", str(obj.maxshape)))
            if obj.fillvalue is not None and obj.dtype.kind in "biuf":
                meta.append(("Fill value", to_text(obj.fillvalue)))
        else:
            meta += [("Kind", "group" if obj.name != "/" else "file (root group)"), ("Members", str(len(obj)))]
        attrs = []
        for k in obj.attrs:
            try:
                v = obj.attrs[k]
                t = str(obj.attrs.get_id(k).dtype)
                attrs.append((k, to_text(v), t))
            except Exception as e:  # noqa: BLE001
                attrs.append((k, f"(unreadable: {e})", ""))
    return meta, attrs


def _stride(n, limit):
    return max(1, int(np.ceil(n / limit))) if limit else 1


def read_slice(path, h5path, x_dim: Optional[int] = None, y_dim: Optional[int] = None, index=(),
               max_points: int = 2_000_000):
    """Read a 1D or 2D view of an N-dimensional dataset.

    ``x_dim`` (and ``y_dim`` for 2D) are the dimensions kept; every other dimension is
    fixed at ``index[d]``. Very large views are read with a stride so that at most
    ``max_points`` elements are loaded. Returns (array, strides) where array has
    shape (len_y, len_x) for 2D and (len_x,) for 1D."""
    with h5py.File(path, "r") as f:
        ds = f[h5path]
        if ds.shape == ():
            return ds[()], ()
        nd = len(ds.shape)
        index = list(index) + [0] * (nd - len(index))
        keep = [d for d in (y_dim, x_dim) if d is not None]
        per = int(max_points ** (1.0 / max(len(keep), 1)))
        sel, strides = [], []
        for d in range(nd):
            if d in keep:
                s = _stride(ds.shape[d], per)
                sel.append(slice(None, None, s))
                strides.append(s)
            else:
                sel.append(int(np.clip(index[d], 0, ds.shape[d] - 1)))
        a = ds[tuple(sel)]
    if y_dim is not None and x_dim is not None and y_dim > x_dim:
        a = a.T                                      # rows = y
    return np.asarray(a), tuple(strides)


@dataclass
class NXdataPlot:
    signal: str
    axes: List[Optional[str]] = field(default_factory=list)
    errors: Optional[str] = None
    title: str = ""


def nxdata(path, h5path) -> Optional[NXdataPlot]:
    """Default plot of a NeXus/NXcanSAS NXdata group (@signal, @axes, <signal>_errors or
    @uncertainties); follows @default from a file or NXentry. None if there is none."""
    with h5py.File(path, "r") as f:
        g = f[h5path]
        for _ in range(4):                           # follow @default chains
            if not isinstance(g, h5py.Group):
                return None
            d = _s(g.attrs.get("default", ""))
            if _s(g.attrs.get("NX_class", "")) in ("NXdata", "SASdata") or "signal" in g.attrs:
                break
            if d and d in g:
                g = g[d]
                continue
            return None
        sig = _s(g.attrs.get("signal", ""))
        if not sig or sig not in g:
            return None
        ds = g[sig]
        ax = g.attrs.get("axes", None)
        if ax is None:                                     # NXcanSAS: I_axes = "Q"
            ax = g.attrs.get(f"{sig}_axes", None)
        if ax is None:
            axes = []
        else:
            ax = [_s(a) for a in (np.atleast_1d(ax) if not isinstance(ax, str) else [ax])]
            axes = [a for part in ax for a in str(part).split(":")]
        axes = [(g.name + "/" + a) if a and a != "." and a in g else None for a in axes]
        err = None
        unc = _s(ds.attrs.get("uncertainties", ""))
        if unc and unc in g:
            err = g.name + "/" + unc
        elif sig + "_errors" in g:
            err = g.name + "/" + sig + "_errors"
        elif "errors" in g:
            err = g.name + "/errors"
        title = _s(g.attrs.get("title", "")) or g.name
        return NXdataPlot(g.name + "/" + sig, axes, err, str(title))


def find(path, text, limit=500) -> List[str]:
    """Paths of all objects whose path contains ``text`` (case-insensitive)."""
    t = text.lower()
    out = []
    with h5py.File(path, "r") as f:
        def visit(name):
            if t in name.lower():
                out.append("/" + name)
                if len(out) >= limit:
                    return True
            return None
        f.visit(visit)
    return out


_TO_M = {"m": 1.0, "metre": 1.0, "meter": 1.0, "cm": 1e-2, "mm": 1e-3, "um": 1e-6, "µm": 1e-6,
         "micron": 1e-6, "nm": 1e-9}


def _length(ds):
    """Value of a length dataset in metres (scalar or first element); None if not a length."""
    try:
        v = float(np.asarray(ds[()]).reshape(-1)[0])
    except (TypeError, ValueError, IndexError):
        return None
    u = _s(ds.attrs.get("units", "m"))
    f = _TO_M.get(str(u).strip().lower())
    return None if f is None or not np.isfinite(v) or v <= 0 else v * f


def _sizes_in(g):
    if "x_pixel_size" in g and "y_pixel_size" in g:
        px, py = _length(g["x_pixel_size"]), _length(g["y_pixel_size"])
        if px and py:
            return px, py
    return None


def pixel_size(path, h5path):
    """(x, y) pixel size in metres of a detector image, from the NeXus fields
    ``x_pixel_size`` / ``y_pixel_size`` (with their units) of the dataset's group, or of
    the NXdetector group that holds the same dataset (hard or soft link, e.g. NXdata).
    x belongs to the last dimension, y to the one before. None if not found."""
    with h5py.File(path, "r") as f:
        obj = f[h5path]
        if isinstance(obj, h5py.Dataset):
            s = _sizes_in(obj.parent)
            if s:
                return s
            target = obj
        else:
            return _sizes_in(obj)
        found = []

        def visit(name, o):
            if isinstance(o, h5py.Group) and _s(o.attrs.get("NX_class", "")) == "NXdetector":
                for k in o:
                    try:
                        if o[k] == target:
                            s = _sizes_in(o)
                            if s:
                                found.append(s)
                                return True
                    except (KeyError, OSError):
                        pass
            return None
        f.visititems(visit)
        return found[0] if found else None


def pixel_aspect(sizes, ndim, x_dim, y_dim):
    """matplotlib ``aspect`` (height / width of one data unit) for showing dimensions
    ``x_dim`` (horizontal) and ``y_dim`` (vertical) of an ``ndim`` array with the pixel
    sizes ``sizes`` = (x, y) of :func:`pixel_size`; None if the shown dimensions are not
    the two detector dimensions."""
    if not sizes or ndim < 2:
        return None
    size = {ndim - 1: sizes[0], ndim - 2: sizes[1]}
    if x_dim not in size or y_dim not in size or x_dim == y_dim:
        return None
    return size[y_dim] / size[x_dim]
