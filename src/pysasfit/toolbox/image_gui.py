# SPDX-License-Identifier: BSD-3-Clause
"""pyTOOLbox page *2D data*: reduced 2D images (pySASred in memory or 2D NXcanSAS files, BerSANS 2D
files, detector images sent from the HDF5 viewer) in pixels, in qx–qy or in polar coordinates (ψ, q),
with the mask, q rings and a cursor read-out of q, ψ and I ± dI.

Tools (tabs below the image list):

* **Mask** – mask or unmask pixels in rectangles, ellipses or polygons drawn in any view, below / above a
  value; invert, clear, reset, undo; save / load.
* **Averages** – radial, sector, azimuthal (ring), box (tilted) and A(q) + B(q) sin²(ψ − ψ0); the region
  can be shown on the image. Results go to the *1D data* page.
* **Sum / rocking** – sum or mean of the selected images (a new image); the mean intensity in a q range
  (and sector) of every selected image versus a header item (rocking curve).

The beam centre can be typed, set by a click on the image or taken from the centre of gravity."""
from __future__ import annotations

import os
from typing import List, Optional


import numpy as np
from PySide6 import QtCore, QtWidgets
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure

from ..core.colorscale import SCALES, decorate_colorbar, make_norm
from ..core.docking import plot_panel
from . import analysis2d as an
from . import image2d as i2

__all__ = ["ImageTool"]

import weakref  # noqa: E402

_TOOLS: "weakref.WeakSet" = weakref.WeakSet()      # open 2D pages

_CMAPS = ["viridis", "magma", "inferno", "jet", "turbo", "gray", "cividis"]


class ImageTool(QtWidgets.QWidget):
    status = QtCore.Signal(str)
    data_produced = QtCore.Signal(object)          # list of data1d.Data1D (to the 1D data page)

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
        self.list.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.ExtendedSelection)
        self.list.setToolTip("the current row is shown; several rows: for sums and rocking curves")
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
        self.SD = spin(0.0, 100, 4, 0.1, " m")
        self.lam = spin(0.0, 10, 4, 0.01, " nm")
        c = QtWidgets.QHBoxLayout()
        c.addWidget(self.cx)
        c.addWidget(self.cy)
        gl.addRow("beam centre x, y [px]:", c)
        gl.addRow("pixel size:", self.pix)
        gl.addRow("detector distance:", self.SD)
        gl.addRow("wavelength:", self.lam)
        ap = QtWidgets.QPushButton("Apply geometry")
        ap.setToolTip("Needed for BerSANS 2D files (no beam centre in the file); q and ψ follow from it")
        ap.clicked.connect(self.on_apply_geometry)
        self.click_centre = QtWidgets.QCheckBox("click on the image sets the beam centre")
        cog = QtWidgets.QPushButton("Centre of gravity")
        cog.setToolTip("beam centre from the centre of gravity of the brightest spot (a direct-beam image)")
        cog.clicked.connect(self.on_centre_of_gravity)
        rowc = QtWidgets.QHBoxLayout()
        rowc.addWidget(ap)
        rowc.addWidget(cog)
        gl.addRow(rowc)
        gl.addRow(self.click_centre)
        lv.addWidget(gb)
        self.ops = QtWidgets.QTabWidget()
        self.ops.addTab(self._build_mask_tab(), "Mask")
        self.ops.addTab(self._build_avg_tab(), "Averages")
        self.ops.addTab(self._build_sum_tab(), "Sum / rocking")
        lv.addWidget(self.ops)
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
        self.canvas.mpl_connect("button_press_event", self._on_click)
        self._selector = None
        self._shown: Optional[i2.Image2D] = None
        self._ax = None
        self._view = None
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
        self._update_keys()

    def current(self) -> Optional[i2.Image2D]:
        r = self.list.currentRow()
        return self.images[r] if 0 <= r < len(self.images) else None

    def _on_select(self):
        im = self.current()
        if im is not None:
            for w, v in ((self.cx, im.centre_px[0] if im.centre_px else 0.0),
                         (self.cy, im.centre_px[1] if im.centre_px else 0.0),
                         (self.pix, 1000 * im.pixel_m[0] if im.pixel_m else 7.5),
                         (self.SD, im.SD_m or 0.0), (self.lam, (im.wavelength_A or 0.0) / 10)):
                w.blockSignals(True)
                w.setValue(float(v) if np.isfinite(v) else 0.0)
                w.blockSignals(False)
        self.replot()

    def on_apply_geometry(self):
        im = self.current()
        if im is None:
            return
        if self.SD.value() <= 0 or self.lam.value() <= 0:
            self._log("distance and wavelength must be > 0")
            return
        im.set_geometry((self.cx.value(), self.cy.value()), self.pix.value() / 1000, self.SD.value(),
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
            pol = polar_image(np.asarray(np.ma.filled(data, np.nan), float), im.centre_px, im.pixel_m, im.SD_m,
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
        if self.show_region.isChecked() and view in ("q", "polar"):
            self._draw_region(ax, view)
        self._attach_selector(ax)
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

    # ------------------------------------------------------------------ beam centre
    def _set_centre(self, x, y):
        im = self.current()
        if im is None:
            return
        self.cx.setValue(float(x))
        self.cy.setValue(float(y))
        if im.pixel_m is not None and im.SD_m and im.wavelength_A:
            im.set_geometry((x, y))
        else:
            im.centre_px = (float(x), float(y))
        self._log(f"beam centre ({x:.2f}, {y:.2f}) px")
        self._refresh()
        self.replot()

    def on_centre_of_gravity(self):
        im = self.current()
        if im is not None:
            self._set_centre(*an.beam_centre_of_gravity(np.where(im.mask, 0, im.I)))

    def _on_click(self, ev):
        im, ax = self._shown, self._ax
        if (im is None or ev.inaxes is not ax or ev.xdata is None or not self.click_centre.isChecked()
                or self.mask_tool.currentData() != "navigate"):
            return
        if self._view == "pixels":
            self._set_centre(ev.xdata, ev.ydata)
        elif self._view == "q" and im.has_q:
            # the click is the new q = 0: shift the centre by the clicked q in pixels
            k = 4 * np.pi / (im.wavelength_A / 10.0)
            r = np.hypot(ev.xdata, ev.ydata)
            tth = 2 * np.arcsin(min(1.0, r / k))
            d = im.SD_m * np.tan(tth)
            dx = d * ev.xdata / r if r else 0.0
            dy = d * ev.ydata / r if r else 0.0
            self._set_centre(im.centre_px[0] + dx / im.pixel_m[0], im.centre_px[1] + dy / im.pixel_m[1])

    # ------------------------------------------------------------------ mask
    def _build_mask_tab(self):
        w = QtWidgets.QWidget()
        g = QtWidgets.QGridLayout(w)
        self.mask_tool = QtWidgets.QComboBox()
        for t, k in (("navigate (zoom / pan / click)", "navigate"), ("rectangle", "rectangle"),
                     ("ellipse", "ellipse"), ("polygon", "polygon")):
            self.mask_tool.addItem(t, k)
        self.mask_tool.currentIndexChanged.connect(self.replot)
        self.mask_mode = QtWidgets.QComboBox()
        self.mask_mode.addItem("mask", True)
        self.mask_mode.addItem("unmask", False)
        g.addWidget(QtWidgets.QLabel("draw:"), 0, 0)
        g.addWidget(self.mask_tool, 0, 1, 1, 2)
        g.addWidget(self.mask_mode, 0, 3)
        self.mask_value = QtWidgets.QLineEdit("0")
        self.mask_value.setMaximumWidth(90)
        g.addWidget(QtWidgets.QLabel("value:"), 1, 0)
        g.addWidget(self.mask_value, 1, 1)
        for j, (t, k) in enumerate((("mask below", "below"), ("mask above", "above"))):
            b = QtWidgets.QPushButton(t)
            b.clicked.connect(lambda _c=False, kk=k: self.on_mask_value(kk))
            g.addWidget(b, 1, 2 + j)
        for j, (t, f) in enumerate((("Invert", self.on_mask_invert), ("Clear", self.on_mask_clear),
                                    ("Reset", self.on_mask_reset), ("Undo", self.on_mask_undo))):
            b = QtWidgets.QPushButton(t)
            b.clicked.connect(f)
            g.addWidget(b, 2, j)
        for j, (t, f) in enumerate((("Suggest", self.on_mask_suggest), ("Save mask…", self.on_mask_save),
                                    ("Load mask…", self.on_mask_load))):
            b = QtWidgets.QPushButton(t)
            b.clicked.connect(f)
            g.addWidget(b, 3, j)
        self.mask_info = QtWidgets.QLabel("")
        g.addWidget(self.mask_info, 3, 3)
        return w

    def _attach_selector(self, ax):
        from matplotlib.widgets import EllipseSelector, PolygonSelector, RectangleSelector
        self._selector = None
        tool = self.mask_tool.currentData()
        if tool == "navigate" or self._view is None:
            return
        props = dict(edgecolor="red", facecolor="none", linewidth=1)
        if tool == "rectangle":
            self._selector = RectangleSelector(ax, lambda e0, e1: self._apply_shape(
                "rectangle", (e0.xdata, e1.xdata, e0.ydata, e1.ydata)), useblit=False, props=props, interactive=False)
        elif tool == "ellipse":
            self._selector = EllipseSelector(ax, lambda e0, e1: self._apply_shape(
                "ellipse", (e0.xdata, e1.xdata, e0.ydata, e1.ydata)), useblit=False, props=props, interactive=False)
        else:
            self._selector = PolygonSelector(ax, lambda verts: self._apply_shape("polygon", verts),
                                             props=dict(color="red", linewidth=1))

    def _push_mask(self, im):
        if not hasattr(im, "_mask0"):
            im._mask0 = im.mask.copy()
            im._undo = []
        im._undo.append(im.mask.copy())
        del im._undo[:-30]

    def _apply_sel(self, sel, what):
        im = self.current()
        if im is None:
            return
        self._push_mask(im)
        if self.mask_mode.currentData():
            im.mask = im.mask | sel
        else:
            im.mask = im.mask & ~sel
        self.mask_info.setText(f"{int(im.mask.sum())} px masked")
        self._log(f"{what}: {int(sel.sum())} px {'masked' if self.mask_mode.currentData() else 'unmasked'}")
        self.replot()

    def _apply_shape(self, kind, pts):
        im = self.current()
        if im is None or self._view is None:
            return
        if kind != "polygon" and None in pts:
            return
        self._apply_sel(an.pixels_in_shape(im, self._view, kind, pts), kind)

    def on_mask_value(self, kind):
        im = self.current()
        try:
            v = float(self.mask_value.text().replace(",", "."))
        except ValueError:
            self._log(f"not a number: {self.mask_value.text()}")
            return
        if im is not None:
            self._apply_sel(an.pixels_in_shape(im, self._view or "pixels", kind, v), f"{kind} {v:g}")

    def on_mask_suggest(self):
        """Suggested mask as in pySASred: detector border, beam stop shadow around the beam centre, hot
        pixels (added to the mask)."""
        im = self.current()
        if im is None:
            return
        from ..sasred.core.mask import suggest_mask
        c = im.centre_px if im.centre_px and all(np.isfinite(im.centre_px)) else \
            ((im.I.shape[1] - 1) / 2, (im.I.shape[0] - 1) / 2)
        try:
            m, _ = suggest_mask(np.nan_to_num(im.I), c)
        except Exception as exc:  # noqa: BLE001
            self._log(f"no suggestion: {exc}")
            return
        mode = self.mask_mode.currentIndex()
        self.mask_mode.setCurrentIndex(0)
        self._apply_sel(m != 0, "suggested mask")
        self.mask_mode.setCurrentIndex(mode)

    def on_mask_invert(self):
        im = self.current()
        if im is not None:
            self._push_mask(im)
            im.mask = ~im.mask
            self.replot()

    def on_mask_clear(self):
        im = self.current()
        if im is not None:
            self._push_mask(im)
            im.mask = np.zeros_like(im.mask)
            self.replot()

    def on_mask_reset(self):
        im = self.current()
        if im is not None and hasattr(im, "_mask0"):
            self._push_mask(im)
            im.mask = im._mask0.copy()
            self.replot()

    def on_mask_undo(self):
        im = self.current()
        if im is not None and getattr(im, "_undo", None):
            im.mask = im._undo.pop()
            self.replot()

    def on_mask_save(self):
        im = self.current()
        if im is None:
            return
        p, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save mask", "mask.h5", "mask (*.h5 *.npy *.txt)")
        if p:
            from ..sasred.core.mask import USER, save_mask
            save_mask(p, im.mask.astype(np.uint32) * USER, beam_center_px=im.centre_px)
            self._log(f"mask saved to {p}")

    def on_mask_load(self):
        im = self.current()
        if im is None:
            return
        p, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Load mask", "", "mask (*.h5 *.npy *.txt *.msk)")
        if p:
            from ..sasred.core.mask import load_mask
            try:
                m, _info = load_mask(p, im.I.shape)
            except Exception as exc:  # noqa: BLE001
                self._log(f"mask not loaded: {exc}")
                return
            self._push_mask(im)
            im.mask = m != 0
            self.replot()

    # ------------------------------------------------------------------ averages
    def _build_avg_tab(self):
        w = QtWidgets.QWidget()
        g = QtWidgets.QGridLayout(w)

        def dspin(lo, hi, v, dec=3, step=0.01, suffix=""):
            s = QtWidgets.QDoubleSpinBox()
            s.setRange(lo, hi)
            s.setDecimals(dec)
            s.setSingleStep(step)
            s.setValue(v)
            s.setSuffix(suffix)
            s.valueChanged.connect(lambda _v: self.show_region.isChecked() and self.replot())
            return s
        self.avg_kind = QtWidgets.QComboBox()
        for t, k in (("radial I(q)", "radial"), ("sector I(q)", "sector"), ("azimuthal I(ψ) in a ring", "azimuthal"),
                     ("box projection", "box"), ("A(q) + B(q) sin²(ψ − ψ0)", "sin2")):
            self.avg_kind.addItem(t, k)
        self.avg_kind.currentIndexChanged.connect(lambda _i: self.show_region.isChecked() and self.replot())
        self.nbins = QtWidgets.QSpinBox()
        self.nbins.setRange(2, 2000)
        self.nbins.setValue(60)
        self.spacing = QtWidgets.QComboBox()
        self.spacing.addItem("log", "log")
        self.spacing.addItem("linear", "lin")
        self.qmin, self.qmax = dspin(0, 100, 0), dspin(0, 100, 0)
        for sp in (self.qmin, self.qmax):
            sp.setSpecialValueText("auto")
        self.psi0, self.dpsi = dspin(-360, 360, 0, 1, 1, "°"), dspin(0, 180, 15, 1, 1, "°")
        self.both = QtWidgets.QCheckBox("both sides")
        self.both.setChecked(True)
        self.q1, self.q2 = dspin(0, 100, 0.1), dspin(0, 100, 0.3)
        self.npsi = QtWidgets.QSpinBox()
        self.npsi.setRange(4, 720)
        self.npsi.setValue(36)
        self.box_angle, self.box_len, self.box_w = dspin(-360, 360, 0, 1, 1, "°"), dspin(0, 100, 1.0), dspin(0, 100, 0.05)
        self.show_region = QtWidgets.QCheckBox("show region")
        self.show_region.toggled.connect(lambda _on: self.replot())
        rows = [("kind", [self.avg_kind]), ("bins, spacing", [self.nbins, self.spacing]),
                ("q min, q max [1/nm]", [self.qmin, self.qmax]),
                ("sector ψ0, ± Δψ (also ψ0 of sin²)", [self.psi0, self.dpsi, self.both]),
                ("ring q1, q2 [1/nm], ψ bins", [self.q1, self.q2, self.npsi]),
                ("box angle, half length, width", [self.box_angle, self.box_len, self.box_w])]
        for r, (lab, ws) in enumerate(rows):
            g.addWidget(QtWidgets.QLabel(lab), r, 0)
            for j, x in enumerate(ws):
                g.addWidget(x, r, 1 + j)
        b = QtWidgets.QPushButton("Compute → 1D data")
        b.setStyleSheet("font-weight: bold")
        b.clicked.connect(self.on_average)
        g.addWidget(self.show_region, len(rows), 0)
        g.addWidget(b, len(rows), 1, 1, 3)
        from ..core import pycode
        g.addWidget(pycode.button(self.python_code, w), len(rows), 4)
        return w

    def python_code(self):
        """Script: read the selected images (the one shown), their geometry and mask as set here, and the
        analysis of the Averages tab ("Copy Python code")."""
        import base64
        import zlib

        from ..core.pycode import Code, Script, call, lit, path_lit

        sel = self.selected_images() or ([self.current()] if self.current() else [])
        if not sel:
            raise ValueError("No image: add 2D data first.")
        k = self.avg_kind.currentData()
        sc = Script("pyTOOLbox", f"2D data: {self.avg_kind.currentText()}")
        sc.imp("import base64", "import zlib", "import numpy as np", "from pysasfit.toolbox import analysis2d as an",
               "from pysasfit.toolbox import image2d as i2")
        sc.add("",
               "def unpack_mask(text, shape):",
               "    \"\"\"The mask as set in the GUI (True = not valid): packed bits, zlib, base64.\"\"\"",
               "    bits = np.unpackbits(np.frombuffer(zlib.decompress(base64.b64decode(text)), np.uint8))",
               "    return bits[:int(np.prod(shape))].reshape(shape).astype(bool)",
               "",
               "results = []")
        for im in sel:
            sc.comment(im.name)
            if not os.path.isfile(im.path):
                sc.comment(f"{im.name} is not a file (pySASred result in memory): save it as 2D NXcanSAS first; "
                           "left out")
                continue
            sc.add(f"im = next(x for x in i2.read_images({path_lit(im.path)}) if x.name == {im.name!r})")
            if None not in (im.centre_px, im.pixel_m, im.SD_m, im.wavelength_A):
                sc.add(call("im.set_geometry", tuple(im.centre_px), tuple(im.pixel_m), im.SD_m, im.wavelength_A))
            blob = base64.b64encode(zlib.compress(np.packbits(np.asarray(im.mask, bool).ravel()).tobytes(), 9)).decode()
            sc.add("im.mask = unpack_mask(")
            for i in range(0, len(blob), 100):
                sc.add(f"    {blob[i:i + 100]!r}")
            sc.add(f"    , {tuple(im.mask.shape)})")
            if k in ("radial", "sector", "sin2"):
                lo, hi = self.qmin.value(), self.qmax.value()
                if lo <= 0 or hi <= 0:
                    sc.add("q = im.q()[im.valid() & (im.q() > 0)]")
                lo_c = lit(lo) if lo > 0 else "float(q.min())"
                hi_c = lit(hi) if hi > 0 else "float(q.max()) * (1 + 1e-9)"
                sc.add(f"edges = an.q_edges({lo_c}, {hi_c}, {self.nbins.value()}, {self.spacing.currentData()!r})")
            if k == "radial":
                sc.add("results.append(an.radial_average(im, edges))")
            elif k == "sector":
                sc.add(call("results.append(an.radial_average", Code("im"), Code("edges"),
                            sector=(self.psi0.value(), self.dpsi.value(), self.both.isChecked())) + ")")
            elif k == "azimuthal":
                sc.add(call("results.append(an.azimuthal_average", Code("im"), self.q1.value(), self.q2.value(),
                            self.npsi.value()) + ")")
            elif k == "box":
                sc.add(call("results.append(an.box_profile", Code("im"), self.box_angle.value(), self.box_len.value(),
                            self.box_w.value(), self.nbins.value()) + ")")
            else:
                sc.add(f"A, B, chi = an.sin2_decomposition(im, edges, psi0={lit(self.psi0.value())})",
                       "results += [A, B]")
        sc.add("for d in results:", "    print(d.name, len(d.x), 'points')")
        sc.add_plot("fig, ax = plt.subplots()", "for d in results:",
                    "    ax.errorbar(d.x, d.y, d.dy, fmt='.-', ms=3, label=d.name)",
                    "ax.set_xlabel(f'{results[0].xlabel} [{results[0].xunit}]' if results else 'x')",
                    "ax.legend(fontsize=7)")
        if k in ("radial", "sector", "sin2"):
            sc.add_plot("ax.set_xscale('log')", "ax.set_yscale('log')")
        return sc.text()

    def _edges(self, im):
        lo, hi = self.qmin.value(), self.qmax.value()
        if lo <= 0 or hi <= 0:
            q = im.q()[im.valid() & (im.q() > 0)]
            lo = lo if lo > 0 else float(q.min())
            hi = hi if hi > 0 else float(q.max()) * (1 + 1e-9)
        return an.q_edges(lo, hi, self.nbins.value(), self.spacing.currentData())

    def average(self, im=None):
        """The analysis chosen in the Averages tab for an image (default: the one shown): list of Data1D."""
        im = im or self.current()
        if im is None:
            return []
        if not im.has_q:
            raise ValueError("geometry unknown: set the beam centre, distance and wavelength first")
        k = self.avg_kind.currentData()
        if k == "radial":
            return [an.radial_average(im, self._edges(im))]
        if k == "sector":
            return [an.radial_average(im, self._edges(im), sector=(self.psi0.value(), self.dpsi.value(),
                                                                    self.both.isChecked()))]
        if k == "azimuthal":
            return [an.azimuthal_average(im, self.q1.value(), self.q2.value(), self.npsi.value())]
        if k == "box":
            return [an.box_profile(im, self.box_angle.value(), self.box_len.value(), self.box_w.value(),
                                   self.nbins.value())]
        a, b, chi = an.sin2_decomposition(im, self._edges(im), psi0=self.psi0.value())
        return [a, b]

    def on_average(self):
        sel = self.selected_images() or ([self.current()] if self.current() else [])
        out = []
        for im in sel:
            try:
                out += self.average(im)
            except Exception as exc:  # noqa: BLE001
                self._log(f"{im.name}: {exc}")
        if out:
            self.data_produced.emit(out)
            self._log(f"{len(out)} data sets → 1D data")
        return out

    def _draw_region(self, ax, view):
        k = self.avg_kind.currentData()
        kw = dict(color="w", lw=1.0, ls="--")
        t = np.linspace(0, 2 * np.pi, 361)
        if k in ("sector", "sin2") and view == "q":
            for base in ((0.0, 180.0) if self.both.isChecked() or k == "sin2" else (0.0,)):
                for sgn in ((-1, 1) if k == "sector" else (0,)):
                    a = np.radians(self.psi0.value() + base + sgn * self.dpsi.value())
                    R = max(abs(np.array(ax.get_xlim() + ax.get_ylim())))
                    ax.plot([0, R * np.cos(a)], [0, R * np.sin(a)], **kw)
        elif k in ("sector", "sin2") and view == "polar":
            for base in ((0.0, 180.0) if self.both.isChecked() or k == "sin2" else (0.0,)):
                for sgn in ((-1, 1) if k == "sector" else (0,)):
                    ax.axvline((self.psi0.value() + base + sgn * self.dpsi.value()) % 360, **kw)
        elif k == "azimuthal":
            for q in (self.q1.value(), self.q2.value()):
                if view == "q":
                    ax.plot(q * np.cos(t), q * np.sin(t), **kw)
                else:
                    ax.axhline(q, **kw)
        elif k == "box" and view == "q":
            a = np.radians(self.box_angle.value())
            L, W = self.box_len.value(), self.box_w.value() / 2
            u = np.array([-L, L, L, -L, -L])
            v = np.array([-W, -W, W, W, -W])
            ax.plot(u * np.cos(a) - v * np.sin(a), u * np.sin(a) + v * np.cos(a), **kw)
        lo, hi = self.qmin.value(), self.qmax.value()
        if k in ("radial", "sector", "sin2"):
            for q in (lo, hi):
                if q > 0:
                    if view == "q":
                        ax.plot(q * np.cos(t), q * np.sin(t), color="w", lw=0.8, ls=":")
                    else:
                        ax.axhline(q, color="w", lw=0.8, ls=":")

    # ------------------------------------------------------------------ sums and rocking curves
    def _build_sum_tab(self):
        w = QtWidgets.QWidget()
        g = QtWidgets.QGridLayout(w)
        b1 = QtWidgets.QPushButton("Sum of the selected images")
        b1.clicked.connect(lambda: self.on_sum(False))
        b2 = QtWidgets.QPushButton("Mean of the selected images")
        b2.clicked.connect(lambda: self.on_sum(True))
        g.addWidget(b1, 0, 0, 1, 2)
        g.addWidget(b2, 0, 2, 1, 2)
        self.rock_key = QtWidgets.QComboBox()
        self.rock_key.setEditable(True)
        self.rock_key.setToolTip("header item of the x axis, e.g. hdf:sample/omega (rocking angle)")
        self.rock_region = QtWidgets.QComboBox()
        self.rock_region.addItem("whole detector", "all")
        self.rock_region.addItem("ring q1–q2 (Averages)", "ring")
        self.rock_region.addItem("ring q1–q2 in the sector ψ0 ± Δψ", "sector")
        g.addWidget(QtWidgets.QLabel("rocking: x ="), 1, 0)
        g.addWidget(self.rock_key, 1, 1, 1, 3)
        g.addWidget(QtWidgets.QLabel("region:"), 2, 0)
        g.addWidget(self.rock_region, 2, 1, 1, 2)
        b3 = QtWidgets.QPushButton("Rocking curve → 1D data")
        b3.clicked.connect(self.on_rocking)
        g.addWidget(b3, 3, 0, 1, 4)
        return w

    def selected_images(self) -> List[i2.Image2D]:
        rows = sorted({i.row() for i in self.list.selectedIndexes()})
        return [self.images[r] for r in rows if r < len(self.images)]

    def on_sum(self, mean=False):
        sel = self.selected_images()
        if len(sel) < 2:
            self._log("select at least two images")
            return None
        try:
            im = an.sum_images(sel, mean)
        except ValueError as exc:
            self._log(str(exc))
            return None
        self.add_images([im])
        return im

    def on_rocking(self):
        sel = self.selected_images()
        key = self.rock_key.currentText().strip()
        if not sel or not key:
            self._log("select images and a header item")
            return None
        reg = self.rock_region.currentData()
        kw = {}
        if reg in ("ring", "sector"):
            kw = dict(q1=self.q1.value(), q2=self.q2.value())
        if reg == "sector":
            kw["sector"] = (self.psi0.value(), self.dpsi.value(), self.both.isChecked())
        try:
            d = an.rocking_curve(sel, key, **kw)
        except ValueError as exc:
            self._log(str(exc))
            return None
        self.data_produced.emit([d])
        return d

    def _update_keys(self):
        keys = set()
        for im in self.images:
            keys.update(k for k, v in im.meta.items() if isinstance(v, (int, float)) and not isinstance(v, bool))
        cur = self.rock_key.currentText()
        self.rock_key.blockSignals(True)
        self.rock_key.clear()
        self.rock_key.addItems(sorted(keys))
        self.rock_key.setCurrentText(cur)
        self.rock_key.blockSignals(False)
