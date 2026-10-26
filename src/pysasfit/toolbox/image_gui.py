# SPDX-License-Identifier: BSD-3-Clause
"""pyTOOLbox page *2D data*: display reduced 2D images (pySASred in memory or 2D NXcanSAS files,
BerSANS 2D files) in pixels, in qx–qy or in polar coordinates (ψ, q), with the mask, q rings and a
cursor read-out of q, ψ and I ± dI. The analysis of the images (sector and azimuthal averages,
A(q) + B(q) sin²ψ, box projections, sums) will be added here."""
from __future__ import annotations

import os
from typing import List, Optional

import numpy as np
from PySide6 import QtCore, QtWidgets
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure

from ..core.colorscale import SCALES, decorate_colorbar, make_norm
from ..core.docking import plot_panel
from . import image2d as i2

__all__ = ["ImageTool"]

import weakref  # noqa: E402

_TOOLS: "weakref.WeakSet" = weakref.WeakSet()      # open 2D pages

_CMAPS = ["viridis", "magma", "inferno", "jet", "turbo", "gray", "cividis"]


class ImageTool(QtWidgets.QWidget):
    status = QtCore.Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.images: List[i2.Image2D] = []
        split = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal)
        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(2, 2, 2, 2)
        lay.addWidget(split)

        left = QtWidgets.QWidget()
        lv = QtWidgets.QVBoxLayout(left)
        lv.setContentsMargins(0, 0, 0, 0)
        row = QtWidgets.QHBoxLayout()
        for text, slot in (("Add files…", self.on_add_files), ("From pySASred", self.on_from_sasred),
                           ("Remove", self.on_remove), ("Clear", self.on_clear)):
            b = QtWidgets.QPushButton(text)
            b.clicked.connect(slot)
            row.addWidget(b)
        row.addStretch(1)
        lv.addLayout(row)
        self.list = QtWidgets.QTableWidget(0, 4)
        self.list.setHorizontalHeaderLabels(["image", "sample", "geometry", "unit"])
        self.list.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows)
        self.list.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.SingleSelection)
        self.list.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
        self.list.horizontalHeader().setStretchLastSection(True)
        self.list.itemSelectionChanged.connect(self._on_select)
        lv.addWidget(self.list, 1)

        gb = QtWidgets.QGroupBox("Geometry of the image shown")
        gl = QtWidgets.QFormLayout(gb)

        def spin(lo, hi, dec, step, suffix=""):
            s = QtWidgets.QDoubleSpinBox()
            s.setRange(lo, hi)
            s.setDecimals(dec)
            s.setSingleStep(step)
            s.setSuffix(suffix)
            return s
        self.cx, self.cy = spin(-1000, 2000, 3, 0.1), spin(-1000, 2000, 3, 0.1)
        self.pix = spin(0.01, 100, 3, 0.1, " mm")
        self.L2 = spin(0.0, 100, 4, 0.1, " m")
        self.lam = spin(0.0, 10, 4, 0.01, " nm")
        c = QtWidgets.QHBoxLayout()
        c.addWidget(self.cx)
        c.addWidget(self.cy)
        gl.addRow("beam centre x, y [px]:", c)
        gl.addRow("pixel size:", self.pix)
        gl.addRow("detector distance:", self.L2)
        gl.addRow("wavelength:", self.lam)
        ap = QtWidgets.QPushButton("Apply geometry")
        ap.setToolTip("Needed for BerSANS 2D files (no beam centre in the file); q and ψ follow from it")
        ap.clicked.connect(self.on_apply_geometry)
        gl.addRow(ap)
        lv.addWidget(gb)
        split.addWidget(left)

        self.canvas = FigureCanvasQTAgg(Figure(figsize=(6, 5), layout="constrained"))
        bar = QtWidgets.QHBoxLayout()
        self.view = QtWidgets.QComboBox()
        self.view.addItem("qx – qy", "q")
        self.view.addItem("pixels", "pixels")
        self.view.addItem("polar (ψ, q)", "polar")
        self.scale = QtWidgets.QComboBox()
        for k, t in SCALES.items():
            self.scale.addItem(t, k)
        self.scale.setCurrentIndex(self.scale.findData("asinh"))
        self.cmap = QtWidgets.QComboBox()
        self.cmap.addItems(_CMAPS)
        self.masked = QtWidgets.QComboBox()
        self.masked.addItem("masked grey", "grey")
        self.masked.addItem("masked hidden", "hidden")
        self.masked.addItem("masked shown", "shown")
        self.rings = QtWidgets.QLineEdit()
        self.rings.setPlaceholderText("q rings, e.g. 0.1 0.3 [1/nm]")
        self.rings.setMaximumWidth(180)
        for w in (self.view, self.scale, self.cmap, self.masked):
            w.currentIndexChanged.connect(self.replot)
            bar.addWidget(w)
        self.rings.editingFinished.connect(self.replot)
        bar.addWidget(self.rings)
        bar.addStretch(1)
        self.readout = QtWidgets.QLabel(" ")
        right = plot_panel(self.canvas, self, bars=[bar, self.readout])
        split.addWidget(right)
        split.setStretchFactor(0, 1)
        split.setStretchFactor(1, 3)
        self.canvas.mpl_connect("motion_notify_event", self._on_move)
        self._shown: Optional[i2.Image2D] = None
        self._ax = None
        _TOOLS.add(self)

    # ------------------------------------------------------------------ list
    def _log(self, t):
        self.status.emit(t)

    def add_images(self, imgs: List[i2.Image2D], select_last=True):
        """Add images; an image with the same source (file and entry, or pySASred key) is replaced."""
        paths = {im.path + im.entry for im in imgs}
        self.images = [im for im in self.images if im.path + im.entry not in paths] + list(imgs)
        self._refresh()
        if imgs and select_last:
            self.list.selectRow(len(self.images) - 1)

    def add_paths(self, paths):
        new = []
        for p in paths:
            try:
                new += i2.read_images(p)
            except Exception as exc:  # noqa: BLE001
                self._log(f"not read: {os.path.basename(p)}: {exc}")
        self.add_images(new)
        return new

    def add_reduced(self, results: dict):
        new = []
        for k, r in results.items():
            try:
                new.append(i2.image_from_reduced(r, k))
            except Exception as exc:  # noqa: BLE001
                self._log(f"pySASred {k}: no 2D image – {exc}")
        self.add_images(new)
        return new

    def on_add_files(self):
        ps, _ = QtWidgets.QFileDialog.getOpenFileNames(
            self, "2D data", "", "2D data (*.h5 *.nxs *.0*);;all files (*)")
        if ps:
            self.add_paths(ps)

    def on_from_sasred(self):
        try:
            from ..sasred.gui import MainWindow as Red
        except Exception as exc:  # noqa: BLE001
            self._log(f"pySASred not available: {exc}")
            return
        n = 0
        for w in list(Red.instances):
            if getattr(w, "results", None):
                n += len(self.add_reduced(w.results))
        self._log(f"{n} 2D images from pySASred" if n else "no reduced data in pySASred")

    def on_remove(self):
        r = self.list.currentRow()
        if 0 <= r < len(self.images):
            del self.images[r]
            self._refresh()
            self.replot()

    def on_clear(self):
        self.images = []
        self._refresh()
        self.replot()

    def _refresh(self):
        t = self.list
        t.blockSignals(True)
        t.setRowCount(len(self.images))
        for i, im in enumerate(self.images):
            for j, v in enumerate((im.name, im.sample, im.geometry_label, im.unit)):
                it = QtWidgets.QTableWidgetItem(v)
                it.setToolTip(im.path)
                t.setItem(i, j, it)
        t.resizeColumnsToContents()
        t.blockSignals(False)

    def current(self) -> Optional[i2.Image2D]:
        r = self.list.currentRow()
        return self.images[r] if 0 <= r < len(self.images) else None

    def _on_select(self):
        im = self.current()
        if im is not None:
            for w, v in ((self.cx, im.centre_px[0] if im.centre_px else 0.0),
                         (self.cy, im.centre_px[1] if im.centre_px else 0.0),
                         (self.pix, 1000 * im.pixel_m[0] if im.pixel_m else 7.5),
                         (self.L2, im.L2_m or 0.0), (self.lam, (im.wavelength_A or 0.0) / 10)):
                w.blockSignals(True)
                w.setValue(float(v) if np.isfinite(v) else 0.0)
                w.blockSignals(False)
        self.replot()

    def on_apply_geometry(self):
        im = self.current()
        if im is None:
            return
        if self.L2.value() <= 0 or self.lam.value() <= 0:
            self._log("distance and wavelength must be > 0")
            return
        im.set_geometry((self.cx.value(), self.cy.value()), self.pix.value() / 1000, self.L2.value(),
                        10 * self.lam.value())
        self._refresh()
        self.replot()

    # ------------------------------------------------------------------ plot
    def replot(self):
        fig = self.canvas.figure
        fig.clear()
        self._ax, self._shown, self._polar = None, None, None
        im = self.current()
        if im is None:
            self.canvas.draw_idle()
            return
        ax = fig.add_subplot(111)
        view = self.view.currentData()
        if view in ("q", "polar") and not im.has_q:
            view = "pixels"
            self.readout.setText("geometry unknown (BerSANS 2D): pixels shown – set the beam centre and apply")
        data = np.array(im.I, float)
        valid = im.valid()
        how = self.masked.currentData()
        if how == "hidden":
            data = np.where(valid, data, np.nan)
        norm = make_norm(self.scale.currentData(), data, valid=valid)
        cmap = __import__("matplotlib").colormaps[self.cmap.currentText()].copy()
        cmap.set_bad("lightgrey")
        if how == "grey":
            data = np.ma.masked_where(~valid, data)
        if view == "polar":
            from ..sasred.core.polar import polar_image
            pol = polar_image(np.asarray(np.ma.filled(data, np.nan), float), im.centre_px, im.pixel_m, im.L2_m,
                              im.wavelength_A, invalid=(~valid if how != "shown" else None))
            qe = pol.q_edges * 10
            art = ax.imshow(pol.image, origin="lower", aspect="auto", norm=norm, cmap=cmap,
                            extent=(0, 360, qe[0], qe[-1]), interpolation="nearest")
            ax.set_xlabel("ψ [°]")
            ax.set_ylabel("q [1/nm]")
            ax.set_xticks(range(0, 361, 45))
            self._polar = pol
        elif view == "q":
            ext = im.q_extent()
            art = ax.imshow(data, origin="lower", extent=ext, norm=norm, cmap=cmap, interpolation="nearest")
            ax.set_xlabel("qx [1/nm]")
            ax.set_ylabel("qy [1/nm]")
            ax.set_aspect("equal")
        else:
            art = ax.imshow(data, origin="lower", norm=norm, cmap=cmap, interpolation="nearest")
            ax.set_xlabel("x [pixel]")
            ax.set_ylabel("y [pixel]")
            if im.centre_px and all(np.isfinite(im.centre_px)):
                ax.plot(*im.centre_px, "+", color="w", ms=10)
        for qr in self._rings():
            if view == "q":
                t = np.linspace(0, 2 * np.pi, 361)
                ax.plot(qr * np.cos(t), qr * np.sin(t), "w--", lw=0.7)
            elif view == "polar":
                ax.axhline(qr, color="w", ls="--", lw=0.7)
        from .viewer import image_colorbar
        cb = image_colorbar(fig, ax, art, im.unit)
        decorate_colorbar(cb, self.scale.currentData())
        ax.set_title(f"{im.sample}  [{im.geometry_label}]", fontsize=9)
        self._ax, self._shown, self._view = ax, im, view
        self.canvas.draw_idle()

    def _rings(self):
        out = []
        for x in self.rings.text().replace(",", " ").split():
            try:
                out.append(float(x))
            except ValueError:
                pass
        return out

    def _on_move(self, ev):
        im, ax = self._shown, self._ax
        if im is None or ev.inaxes is not ax or ev.xdata is None:
            return
        ny, nx = im.I.shape
        if self._view == "pixels":
            ix, iy = int(round(ev.xdata)), int(round(ev.ydata))
        elif self._view == "q":
            qx, qy = im.qxy()
            d = (qx - ev.xdata) ** 2 + (qy - ev.ydata) ** 2
            iy, ix = np.unravel_index(np.argmin(d), d.shape)
        else:
            psi, q = np.radians(ev.xdata), ev.ydata
            qx, qy = im.qxy()
            d = (qx - q * np.cos(psi)) ** 2 + (qy - q * np.sin(psi)) ** 2
            iy, ix = np.unravel_index(np.argmin(d), d.shape)
        if not (0 <= ix < nx and 0 <= iy < ny):
            return
        s = f"pixel ({ix}, {iy})"
        if im.has_q:
            s += f"   q = {im.q()[iy, ix]:.4g} 1/nm   ψ = {im.psi_deg()[iy, ix]:.1f}°"
        v = im.I[iy, ix]
        e = im.dI[iy, ix] if im.dI is not None else np.nan
        s += f"   I = {v:.4g}" + (f" ± {e:.2g}" if np.isfinite(e) else "") + ("   (masked)" if im.mask[iy, ix] else "")
        self.readout.setText(s)
