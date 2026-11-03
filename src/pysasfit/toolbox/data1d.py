# SPDX-License-Identifier: BSD-3-Clause
"""One-dimensional data sets of pyTOOLbox: I(q) curves, azimuthal profiles I(ψ), box profiles, rocking
curves, … with uncertainties, and the arithmetic of the *1D data* page.

:class:`Data1D` holds x, y, dy (statistical standard deviation; NaN when unknown) and dx (resolution or
bin spread; zeros when unknown) with labels and units. Operations propagate the uncertainties to first
order (independent data sets):

* scalar: ``scale(a)`` (y·a), ``offset(c)`` (y + c), ``divide(a)`` (y / a)
* two data sets: ``combine(a, b, op)`` with op "+", "-", "*", "/" or "a-f*b"; b is interpolated at the x
  of a (log-log where both are positive, else linear; only the overlapping x range is kept)
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field, replace
from typing import Dict, List, Optional

import numpy as np

__all__ = ["Data1D", "from_curve", "read_data1d", "write_ascii", "combine", "interpolate"]


@dataclass
class Data1D:
    x: np.ndarray
    y: np.ndarray
    dy: np.ndarray
    dx: np.ndarray = None
    name: str = ""
    xlabel: str = "q"
    xunit: str = "1/nm"
    yunit: str = ""
    meta: Dict[str, object] = field(default_factory=dict)
    source: str = ""                  # file, "pySASred:…", "2D: radial average of …", "A − B", …

    def __post_init__(self):
        self.x = np.asarray(self.x, float)
        self.y = np.asarray(self.y, float)
        self.dy = np.full_like(self.y, np.nan) if self.dy is None else np.asarray(self.dy, float)
        self.dx = np.zeros_like(self.x) if self.dx is None else np.asarray(self.dx, float)

    @property
    def is_q(self) -> bool:
        return self.xlabel == "q"

    def copy(self, **kw) -> "Data1D":
        d = replace(self, x=self.x.copy(), y=self.y.copy(), dy=self.dy.copy(), dx=self.dx.copy(),
                    meta=dict(self.meta))
        for k, v in kw.items():
            setattr(d, k, v)
        return d

    # ---------------------------------------------------------------- scalar operations
    def scale(self, a: float, name: Optional[str] = None) -> "Data1D":
        return self.copy(y=self.y * a, dy=self.dy * abs(a), name=name or f"{self.name} × {a:g}",
                         source=f"{self.name} × {a:g}")

    def divide(self, a: float, name: Optional[str] = None) -> "Data1D":
        if a == 0:
            raise ZeroDivisionError("division by zero")
        return self.copy(y=self.y / a, dy=self.dy / abs(a), name=name or f"{self.name} / {a:g}",
                         source=f"{self.name} / {a:g}")

    def offset(self, c: float, name: Optional[str] = None) -> "Data1D":
        return self.copy(y=self.y + c, name=name or f"{self.name} {'+' if c >= 0 else '−'} {abs(c):g}",
                         source=f"{self.name} + {c:g}")

    def cut(self, xmin: Optional[float] = None, xmax: Optional[float] = None) -> "Data1D":
        ok = np.ones(len(self.x), bool)
        if xmin is not None:
            ok &= self.x >= xmin
        if xmax is not None:
            ok &= self.x <= xmax
        return self.copy(x=self.x[ok], y=self.y[ok], dy=self.dy[ok], dx=self.dx[ok],
                         name=f"{self.name} [{xmin if xmin is not None else ''}…{xmax if xmax is not None else ''}]")


def from_curve(c) -> Data1D:
    """A :class:`pysasfit.toolbox.merge.Curve` (or ``Merged``) as Data1D."""
    if hasattr(c, "group") and not hasattr(c, "path"):          # Merged
        g = c.group
        name = f"{', '.join(sorted({x.sample for x in g.curves}))} [{g.label}] merged" if g else "merged"
        return Data1D(c.q, c.I, c.dI, c.dq, name=name, yunit=c.I_unit, source="merged",
                      meta=dict(g.values) if g else {})
    name = c.sample or c.name
    lab = c.meta.get("label")
    return Data1D(c.q, c.I, c.dI, c.dq, name=f"{name} ({lab})" if lab else name, yunit=c.I_unit,
                  meta=dict(c.meta), source=c.path + (f":{c.entry}" if c.entry else ""))


def read_data1d(path) -> List[Data1D]:
    """Files written by :func:`write_ascii` keep their x label and units; everything else is read as
    I(q) with :func:`pysasfit.toolbox.merge.read_curves`."""
    with open(path, "rb") as fh:
        head = fh.read(4000)
    if b"# pyTOOLbox 1D" in head:
        meta = {}
        for ln in head.decode("utf-8", "replace").splitlines():
            if ln.startswith("# json = "):
                meta = json.loads(ln[len("# json = "):])
                break
        a = np.loadtxt(path, ndmin=2)
        return [Data1D(a[:, 0], a[:, 1], a[:, 2] if a.shape[1] > 2 else None, a[:, 3] if a.shape[1] > 3 else None,
                       name=meta.get("name", os.path.basename(path)), xlabel=meta.get("xlabel", "q"),
                       xunit=meta.get("xunit", "1/nm"), yunit=meta.get("yunit", ""), meta=meta.get("meta", {}),
                       source=str(path))]
    from .merge import read_curves
    return [from_curve(c) for c in read_curves(path)]


def write_ascii(path, d: Data1D) -> str:
    info = dict(name=d.name, xlabel=d.xlabel, xunit=d.xunit, yunit=d.yunit, source=d.source,
                meta={k: v for k, v in d.meta.items() if isinstance(v, (str, int, float, bool)) or v is None})
    head = [f"pyTOOLbox 1D: {d.name}", f"source: {d.source}", f"json = {json.dumps(info, default=str)}",
            f"{d.xlabel} [{d.xunit}]  I [{d.yunit}]  dI  d{d.xlabel} [{d.xunit}]"]
    np.savetxt(path, np.column_stack([d.x, d.y, d.dy, d.dx]), fmt="%.8e", header="\n".join(head))
    return str(path)


def interpolate(b: Data1D, x: np.ndarray):
    """y and dy of ``b`` at ``x`` (log-log where x and y are positive, else linear)."""
    o = np.argsort(b.x)
    bx, by, bdy = b.x[o], b.y[o], b.dy[o]
    if np.all(bx > 0) and np.all(x > 0) and np.all(by > 0):
        y = np.exp(np.interp(np.log(x), np.log(bx), np.log(by)))
    else:
        y = np.interp(x, bx, by)
    dy = np.interp(x, bx, np.nan_to_num(bdy, nan=0.0))
    if np.all(np.isnan(bdy)):
        dy = np.full_like(y, np.nan)
    return y, dy


def combine(a: Data1D, b: Data1D, op: str = "-", f: float = 1.0) -> Data1D:
    """a op b on the x of a within the x range of b; op "+", "-", "*", "/" or "a-f*b"."""
    if a.xlabel != b.xlabel:
        raise ValueError(f"different x: {a.xlabel} and {b.xlabel}")
    lo, hi = np.nanmin(b.x), np.nanmax(b.x)
    ok = (a.x >= lo) & (a.x <= hi)
    if ok.sum() == 0:
        raise ValueError("the x ranges do not overlap")
    x, ya, sa = a.x[ok], a.y[ok], a.dy[ok]
    yb, sb = interpolate(b, x)
    if op == "+":
        y, s, sym = ya + yb, np.hypot(sa, sb), "+"
    elif op == "-":
        y, s, sym = ya - yb, np.hypot(sa, sb), "−"
    elif op == "a-f*b":
        y, s, sym = ya - f * yb, np.hypot(sa, f * sb), f"− {f:g} ×"
    elif op == "*":
        y = ya * yb
        s, sym = np.abs(y) * np.hypot(sa / ya, sb / yb), "×"
    elif op == "/":
        with np.errstate(divide="ignore", invalid="ignore"):
            y = ya / yb
            s = np.abs(y) * np.hypot(sa / ya, sb / yb)
        sym = "/"
    else:
        raise ValueError(f"unknown operation {op!r}")
    yunit = a.yunit if op in ("+", "-", "a-f*b") else ("" if op == "/" else f"{a.yunit}·{b.yunit}")
    name = f"({a.name}) {sym} ({b.name})"
    return Data1D(x, y, s, a.dx[ok], name=name, xlabel=a.xlabel, xunit=a.xunit, yunit=yunit,
                  meta={}, source=name)
