# SPDX-License-Identifier: BSD-3-Clause
"""Interactive mask editing in the detector panel of pySASred.

Tool bar below the colour-scale row of the detector plot:

* **Suggest** – the mask for the image shown: standard border (2 outer rows and columns),
  beamstop shadow (sudden drop around the beam centre) and hot pixels
  (:func:`pysasfit.sasred.core.mask.suggest_mask`). Pixels you drew yourself are kept.
* **Tool** – navigate (zoom / pan with the plot tool bar), or add (+) / remove (−) pixels
  with a rectangle, an ellipse, a polygon (click the corners, close it on the first one)
  or a brush (drag; size in pixels). Remove clears every reason of a pixel.
* **Undo / Redo** (Ctrl+Z / Ctrl+Y), **Reset** (border only), **Load… / Save…**
  (``.h5`` with the reason bits, ``.npy``, or a 0/1 ``.txt`` matrix), **show** (overlay).

Editing works in the cartesian view; the polar view shows the masked pixels in grey.
The mask is used at once by the live I(q), the reduction and the polar view, and is
kept with the last session.
"""
from __future__ import annotations

import os

import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets

from ..core import settings as prefs
from ..core.flowlayout import FlowLayout
from .core import mask as mk

__all__ = ["MaskEditor", "OVERLAY_COLOURS", "MASKED_CMAP"]

#: overlay colour (RGB) of each reason, in the order they are painted (last on top); saturated
#: colours that none of the colour maps uses, plus a black-and-white outline (see draw_overlay)
OVERLAY_COLOURS = [(mk.BORDER, (0.12, 0.12, 0.12)), (mk.BEAMSTOP, (1.0, 0.0, 0.0)),
                   (mk.USER, (1.0, 0.0, 1.0)), (mk.HOT, (0.0, 1.0, 1.0)), (mk.DEADTIME, (1.0, 1.0, 1.0)),
                   (mk.EFFICIENCY, (1.0, 0.55, 0.0))]


def outline_segments(masked):
    """Pixel-exact boundary of a bool image as merged straight line segments (pixel centres at
    integers). The edge of the image itself is not outlined."""
    m = np.pad(np.asarray(masked, bool), 1, mode="edge")
    segs = []
    dy = m[1:, 1:-1] != m[:-1, 1:-1]                 # edge between rows y-1 and y
    for y in np.nonzero(dy.any(axis=1))[0]:
        xs = np.nonzero(dy[y])[0]
        for run in np.split(xs, np.nonzero(np.diff(xs) > 1)[0] + 1):
            segs.append(((run[0] - 0.5, y - 0.5), (run[-1] + 0.5, y - 0.5)))
    dx = m[1:-1, 1:] != m[1:-1, :-1]                 # edge between columns x-1 and x
    for x in np.nonzero(dx.any(axis=0))[0]:
        ys = np.nonzero(dx[:, x])[0]
        for run in np.split(ys, np.nonzero(np.diff(ys) > 1)[0] + 1):
            segs.append(((x - 0.5, run[0] - 0.5), (x - 0.5, run[-1] + 0.5)))
    return segs

TOOLS = [("navigate", "navigate (zoom / pan)"), ("+rect", "+ rectangle"), ("-rect", "− rectangle"),
         ("+ellipse", "+ ellipse"), ("-ellipse", "− ellipse"), ("+polygon", "+ polygon"),
         ("-polygon", "− polygon"), ("+brush", "+ brush"), ("-brush", "− brush")]
UNDO_DEPTH = 100
FILE_FILTER = "pySASred mask (*.h5);;NumPy array (*.npy);;0/1 text matrix (*.txt *.msk)"


#: colour map of the masked pixels drawn on their own scale (neutral, unlike the image colour maps)
MASKED_CMAP = "gray"


def _flat(colour):
    from matplotlib.colors import ListedColormap
    return ListedColormap([colour])


def overlay_rgba(mask, alpha=0.8):
    m = np.asarray(mask, np.uint32)
    out = np.zeros(m.shape + (4,))
    for bit, col in OVERLAY_COLOURS:
        out[(m & bit) != 0] = (*col, alpha)
    return out


class MaskEditor(QtCore.QObject):
    """Mask state, its tool bar and the mouse interaction on the detector plot of ``win``
    (the pySASred MainWindow)."""

    changed = QtCore.Signal()

    def __init__(self, win):
        super().__init__(win)
        self.win = win
        self.mask = None                 # uint32, detector shape (None until an image is shown)
        self._undo, self._redo = [], []
        self._selector = None
        self._cids = []
        self._painting = False
        self._overlay = None
        self.pristine = False            # True while the mask is the untouched start (border + first guess)
        self.bar = self._build_bar()

    # ------------------------------------------------------------------ tool bar
    def _build_bar(self):
        w = QtWidgets.QWidget()
        lay = FlowLayout(w)                 # wraps to more rows in a narrow panel
        sp = w.sizePolicy()
        sp.setHeightForWidth(True)
        sp.setHorizontalPolicy(QtWidgets.QSizePolicy.Policy.Ignored)
        w.setSizePolicy(sp)
        lay.addWidget(QtWidgets.QLabel("mask"))
        b = QtWidgets.QPushButton("Suggest")
        b.setToolTip("border + beamstop shadow (sudden drop around the beam centre) + hot pixels of the image "
                     "shown; your own (user) pixels are kept")
        b.clicked.connect(self.suggest)
        lay.addWidget(b)
        self.tool = QtWidgets.QComboBox()
        for key, label in TOOLS:
            self.tool.addItem(label, key)
        self.tool.setToolTip("add (+) or remove (−) masked pixels on the detector image (cartesian view)")
        self.tool.currentIndexChanged.connect(lambda _i: self.attach())
        lay.addWidget(self.tool)
        self.brush = QtWidgets.QSpinBox()
        self.brush.setRange(0, 20)
        self.brush.setValue(1)
        self.brush.setPrefix("r ")
        self.brush.setSuffix(" px")
        self.brush.setToolTip("brush radius (0 = single pixel)")
        lay.addWidget(self.brush)
        st = w.style()
        for text, icon, tip, slot in (
                ("", QtWidgets.QStyle.StandardPixmap.SP_ArrowBack, "Undo (Ctrl+Z)", self.undo),
                ("", QtWidgets.QStyle.StandardPixmap.SP_ArrowForward, "Redo (Ctrl+Y)", self.redo),
                ("Reset", None, "standard mask (outer rows and columns) and the first-guess beamstop", self.reset),
                ("Load…", None, "load a mask file (.h5, .npy, 0/1 .txt)", self.on_load),
                ("Save…", None, "save the mask (.h5 with the reasons, .npy, 0/1 .txt)", self.on_save)):
            bb = QtWidgets.QToolButton()
            if icon is not None:
                bb.setIcon(st.standardIcon(icon))
            else:
                bb.setText(text)
            bb.setToolTip(tip)
            bb.clicked.connect(slot)
            lay.addWidget(bb)
        self.display = QtWidgets.QComboBox()
        for key, label in (("colours", "colours"), ("scale", "own scale"), ("hidden", "hidden")):
            self.display.addItem(label, key)
        self.display.setCurrentIndex(max(0, self.display.findData(prefs.get("sasred.mask.display"))))
        self.display.setToolTip("masked pixels: in the mask colours; on their own grey scale with its own "
                                "colour bar (to see what is behind the mask); or hidden (outline only). "
                                "The colour scale of the image uses the unmasked pixels only.")
        self.display.currentIndexChanged.connect(lambda _i: self.win._draw_detector())
        lay.addWidget(self.display)
        self.opacity = QtWidgets.QSpinBox()
        self.opacity.setRange(10, 100)
        self.opacity.setSingleStep(10)
        self.opacity.setValue(prefs.get("sasred.mask.opacity"))
        self.opacity.setSuffix(" %")
        self.opacity.setToolTip("opacity of the mask colours (the outline is always drawn)")
        self.opacity.valueChanged.connect(lambda _v: self.win._draw_detector())
        lay.addWidget(self.opacity)
        self.stats = QtWidgets.QLabel("")
        lay.addWidget(self.stats)
        for seq, slot in ((QtGui.QKeySequence.StandardKey.Undo, self.undo),
                          (QtGui.QKeySequence.StandardKey.Redo, self.redo)):
            sc = QtGui.QShortcut(seq, w)
            sc.setContext(QtCore.Qt.ShortcutContext.WidgetWithChildrenShortcut)
            sc.activated.connect(slot)
        return w

    # ------------------------------------------------------------------ state
    @property
    def mode(self):
        """How masked pixels are drawn: colours, scale (own grey scale) or hidden."""
        return self.display.currentData()

    def _panel_raw(self, shape=None):
        raw = self.win.displayed_raw()
        if raw is not None and getattr(raw, "panel", None) is not None and (shape is None or raw.counts.shape == tuple(shape)):
            return raw
        return None

    def start_mask(self, shape):
        """Standard border + the first-guess beamstop of the Reduction panel; a SANS-LLB wide-angle panel:
        tube ends and dead tubes."""
        pr = self._panel_raw(shape)
        if pr is not None:
            return mk.default_mask(pr, (self.win.xc.value(), self.win.yc.value()))[0]
        m = mk.border_mask(shape, prefs.get("sasred.mask.border_px"))
        guess = self.win.beamstop_guess(shape)
        if guess is not None:
            m[guess] |= mk.BEAMSTOP
        return m

    def ensure(self, shape):
        """The mask for a detector of ``shape`` (the start mask if there is none yet)."""
        shape = tuple(shape)
        if self.mask is None or self.mask.shape != shape:
            self.mask = self.start_mask(shape)
            self.pristine = True
            self._undo.clear()
            self._redo.clear()
            self._update_stats()
        return self.mask

    def follow_guess(self):
        """While the mask is still the untouched start mask, move the first guess with the beam
        centre and the size (no undo step)."""
        if self.mask is not None and self.pristine:
            self.mask = self.start_mask(self.mask.shape)
            self._update_stats()
            self.changed.emit()

    def place_guess(self):
        """Replace the beamstop pixels of the mask by the first guess (undoable); other reasons are kept."""
        if self.mask is None:
            return
        m = self.mask & ~np.uint32(mk.BEAMSTOP)
        guess = self.win.beamstop_guess(m.shape)
        if guess is not None:
            m[guess] |= mk.BEAMSTOP
        n = int(np.count_nonzero(m & mk.BEAMSTOP))
        self.set_mask(m, message=f"beamstop first guess put into the mask ({n} px)")

    def set_mask(self, m, record=True, message=None, pristine=False):
        m = np.asarray(m, np.uint32).copy()
        if record and self.mask is not None and self.mask.shape == m.shape:
            self._undo.append(self.mask.copy())
            del self._undo[:-UNDO_DEPTH]
            self._redo.clear()
        self.mask = m
        self.pristine = pristine
        self._update_stats()
        if message:
            self.win.log(message)
        self.refresh()
        self.changed.emit()

    def _update_stats(self):
        if self.mask is None:
            self.stats.setText("")
            return
        s = mk.summary(self.mask)
        parts = [f"{k} {v}" for k, v in s.items() if k != "total" and v]
        self.stats.setText(f"{s['total']} px masked")
        self.stats.setToolTip(", ".join(parts) if parts else "nothing masked")

    def undo(self):
        self.pristine = False
        if self._undo:
            self._redo.append(self.mask.copy())
            self.mask = self._undo.pop()
            self._after_history("undo")

    def redo(self):
        if self._redo:
            self._undo.append(self.mask.copy())
            self.mask = self._redo.pop()
            self._after_history("redo")

    def _after_history(self, what):
        self._update_stats()
        self.refresh()
        self.changed.emit()
        self.win.statusBar().showMessage(f"mask: {what}", 3000)

    def reset(self):
        if self.mask is not None:
            self.set_mask(self.start_mask(self.mask.shape), pristine=True,
                          message="mask reset to the standard border and the first-guess beamstop")

    # ------------------------------------------------------------------ suggestion
    def suggest(self):
        raw = self.win.displayed_raw()
        if raw is None:
            QtWidgets.QMessageBox.information(self.win, "Suggest mask", "Select a run in the run table first.")
            return
        v = prefs.values("sasred.mask")
        bc = (self.win.xc.value(), self.win.yc.value())
        if getattr(raw, "panel", None) is not None:
            m = mk.default_mask(raw, bc)[0]
            n_hot = 0
            if v["hot"]:
                hot = mk.suggest_hot_pixels(raw.counts, valid=m == 0)
                hot = hot[0] if isinstance(hot, tuple) else hot
                m[np.asarray(hot, bool)] |= mk.HOT
                n_hot = int(np.count_nonzero(np.asarray(hot, bool)))
            old = self.ensure(raw.counts.shape)
            m |= old & mk.USER
            self.set_mask(m, message=f"mask suggested for the {raw.bank} panel of run {raw.run}: tube ends "
                                     f"({int(np.count_nonzero(m & mk.BORDER))} px), dead tubes, hot pixels {n_hot}")
            return
        m, rep = mk.suggest_mask(raw.counts, bc, border_width=v["border_px"], search_radius_px=v["search_radius_px"],
                                 drop=v["drop"], grow_px=v["grow_px"], hot=v["hot"], beamstop_shape=v["suggest_shape"])
        old = self.ensure(raw.counts.shape)
        m |= old & mk.USER                                          # keep what the user drew
        b = rep["beamstop"]
        lines = [f"mask suggested from run {raw.run} (beam centre {bc[0]:.2f}, {bc[1]:.2f}):",
                 f"    border: {rep['border']} px ({v['border_px']} outer rows and columns)"]
        if b.get("n_pixels"):
            x0, x1, y0, y1 = b["bbox"]
            what = "rectangle around the shadow" if b.get("shape") == "rectangle" else "shadow"
            lines.append(f"    beamstop ({what}): {b['n_pixels']} px, x {x0}…{x1}, y {y0}…{y1} (counts below "
                         f"{b['threshold']:.4g} = {v['drop']:g} × rim level {b['rim_level']:.4g}; median in the "
                         f"shadow {b['shadow_median']:.4g}); it replaces the first guess")
        else:
            lines.append(f"    beamstop: none – {b.get('reason', '')}")
        if b.get("warning"):
            lines.append(f"    WARNING: {b['warning']}")
        if v["hot"]:
            lines.append(f"    hot pixels: {rep['hot']}" + (f" ({rep['leak']} at the beamstop edge: beam leaking past it)"
                                                          if rep.get("leak") else ""))
        self.set_mask(m, message="\n".join(lines))
        self.win.statusBar().showMessage(lines[0] + " " + self.stats.text(), 10000)

    # ------------------------------------------------------------------ files
    def on_save(self):
        if self.mask is None:
            QtWidgets.QMessageBox.information(self.win, "Save mask", "No mask yet (select a run).")
            return
        start = os.path.join(self.win.folder or os.getcwd(), "mask.h5")
        p, flt = QtWidgets.QFileDialog.getSaveFileName(self.win, "Save mask", start, FILE_FILTER)
        if not p:
            return
        if not os.path.splitext(p)[1]:
            p += ".npy" if "npy" in flt else ".txt" if "txt" in flt else ".h5"
        raw = self.win.displayed_raw()
        mk.save_mask(p, self.mask, beam_center_px=(self.win.xc.value(), self.win.yc.value()),
                     run=raw.run if raw is not None else None)
        self.win.log(f"mask saved: {p}")

    def on_load(self):
        p, _ = QtWidgets.QFileDialog.getOpenFileName(self.win, "Load mask", self.win.folder or os.getcwd(), FILE_FILTER)
        if p:
            self.load(p)

    def load(self, path):
        raw = self.win.displayed_raw()
        try:
            m, info = mk.load_mask(path, shape=raw.counts.shape if raw is not None else None)
        except Exception as e:  # noqa: BLE001
            QtWidgets.QMessageBox.critical(self.win, "Load mask", f"{path}:\n{e}")
            return False
        msg = f"mask loaded: {path}"
        if "beam_center_px" in info:
            msg += f" (saved with beam centre {info['beam_center_px'][0]:.2f}, {info['beam_center_px'][1]:.2f})"
        self.set_mask(m, record=self.mask is not None and self.mask.shape == m.shape, message=msg)
        return True

    # ------------------------------------------------------------------ drawing and interaction
    def draw_overlay(self, ax, shape, image=None, scale="linear"):
        """Masked pixels on the cartesian detector image (see :attr:`mode`) and the outline of the
        masked area; called by the window after each redraw. ``image``: the counts (needed for the
        own grey scale of the masked pixels)."""
        self.ensure(shape)
        self._overlay = None
        self._outline = []
        self._ax = ax
        masked = self.win.full_mask(shape) != 0
        if self.mode == "colours":
            self._overlay = ax.imshow(overlay_rgba(self.win.full_mask(shape), self.opacity.value() / 100),
                                      origin="lower", interpolation="nearest", aspect=ax.get_aspect(), zorder=3)
        elif self.mode == "scale" and image is not None and masked.any():
            from ..core.colorscale import decorate_colorbar, make_norm
            img = np.ma.masked_where(~masked, np.asarray(image, float))
            im = ax.imshow(img, origin="lower", interpolation="nearest", aspect=ax.get_aspect(), zorder=3,
                           cmap=MASKED_CMAP, norm=make_norm(scale, image, valid=masked))
            cax = ax.inset_axes([1.24, 0.0, 0.045, 1.0])
            decorate_colorbar(ax.figure.colorbar(im, cax=cax, label="masked pixels"), scale)
        elif self.mode == "hidden":
            ax.imshow(np.ma.masked_where(~masked, np.ones(masked.shape)), origin="lower", interpolation="nearest",
                      aspect=ax.get_aspect(), zorder=3, cmap=_flat("0.85"), vmin=0, vmax=1)
        self._draw_outline()
        self.attach()

    def _draw_outline(self):
        """Boundary of the masked area: white line on a black one, visible on any colour."""
        for c in self._outline:
            try:
                c.remove()
            except (ValueError, AttributeError):
                pass
        self._outline = []
        from matplotlib.collections import LineCollection
        m = self.win.full_mask(self.mask.shape) != 0
        if not m.any() or m.all():
            return
        segs = outline_segments(m)
        for col, lw in (("black", 2.6), ("white", 1.1)):
            lc = LineCollection(segs, colors=col, linewidths=lw, zorder=4, capstyle="projecting",
                                joinstyle="miter")
            self._outline.append(self._ax.add_collection(lc, autolim=False))

    def refresh(self):
        """Update the overlay (or redraw) after the mask changed."""
        if self._overlay is not None and self.mask is not None:
            self._overlay.set_data(overlay_rgba(self.win.full_mask(self.mask.shape), self.opacity.value() / 100))
            self._draw_outline()
            self.win.det_canvas.draw_idle()
        else:
            self.win._draw_detector()

    def _detach(self):
        canvas = self.win.det_canvas
        for cid in self._cids:
            canvas.mpl_disconnect(cid)
        self._cids = []
        if self._selector is not None:
            try:
                self._selector.set_active(False)
                self._selector.disconnect_events()
            except Exception:  # noqa: BLE001
                pass
        self._selector = None

    def _navigating(self):
        tb = getattr(self.win.det_canvas, "toolbar", None)
        return bool(tb is not None and getattr(tb, "mode", ""))

    def attach(self):
        """Activate the chosen tool on the current detector axes."""
        from matplotlib.widgets import EllipseSelector, PolygonSelector, RectangleSelector
        self._detach()
        ax = getattr(self, "_ax", None)
        key = self.tool.currentData()
        if key != "navigate":
            # drawing works on the cartesian image only: switch to it instead of ignoring the mouse silently
            v = self.win.det_view
            if v.currentData() != "cartesian":
                i = v.findData("cartesian")
                self.win.statusBar().showMessage("mask tools draw on the cartesian detector image: view switched "
                                                 "to cartesian", 6000)
                if i >= 0 and v.currentIndex() != i:
                    v.setCurrentIndex(i)               # redraws the detector and calls attach() again
                    return
            # pan / zoom of the toolbar would swallow the mouse: switch it off
            tb = getattr(self.win.det_canvas, "toolbar", None)
            mode = str(getattr(tb, "mode", "") or "") if tb is not None else ""
            if mode:
                try:
                    (tb.pan if "pan" in mode else tb.zoom)()
                    self.win.statusBar().showMessage("pan / zoom switched off for drawing the mask", 6000)
                except Exception:  # noqa: BLE001
                    pass
        if ax is None or ax.figure is not self.win.det_canvas.figure or key == "navigate" \
                or self.win.det_view.currentData() != "cartesian":
            return
        add = key.startswith("+")
        kind = key[1:]
        props = dict(edgecolor="orange" if add else "cyan", facecolor="none", linewidth=1.5)
        if kind in ("rect", "ellipse"):
            cls = RectangleSelector if kind == "rect" else EllipseSelector

            def done(e0, e1, add=add, kind=kind):
                if self._navigating():
                    self.win.statusBar().showMessage("pan / zoom is on: switch it off to draw the mask", 6000)
                    return
                if None in (e0.xdata, e1.xdata):
                    return
                f = mk.pixels_in_rectangle if kind == "rect" else mk.pixels_in_ellipse
                self.apply(f(self.mask.shape, e0.xdata, e1.xdata, e0.ydata, e1.ydata), add)
            # no minimum span: a drag along one row (or a click on one pixel) must also count
            self._selector = cls(ax, done, useblit=True, button=[1], minspanx=-1, minspany=-1,
                                 spancoords="data", interactive=False, props=props)
        elif kind == "polygon":
            def done_poly(verts, add=add):
                if len(verts) >= 3:
                    self.apply(mk.pixels_in_polygon(self.mask.shape, verts), add)
                QtCore.QTimer.singleShot(0, self.attach)            # fresh selector for the next polygon
            self._selector = PolygonSelector(ax, done_poly, useblit=True,
                                             props=dict(color=props["edgecolor"], linewidth=1.5))
        elif kind == "brush":
            c = self.win.det_canvas
            self._cids = [c.mpl_connect("button_press_event", lambda e, a=add: self._paint(e, a, start=True)),
                          c.mpl_connect("motion_notify_event", lambda e, a=add: self._paint(e, a)),
                          c.mpl_connect("button_release_event", self._paint_end)]

    def _paint(self, e, add, start=False):
        if e.inaxes is not self._ax or e.xdata is None or self._navigating():
            return
        if start:
            if e.button != 1:
                return
            self._painting = True
            self.pristine = False
            self._undo.append(self.mask.copy())
            del self._undo[:-UNDO_DEPTH]
            self._redo.clear()
        if not self._painting:
            return
        sel = mk.pixels_in_disc(self.mask.shape, e.xdata, e.ydata, self.brush.value() + 0.5)
        self._set_pixels(sel, add)
        self._update_stats()
        self.refresh()

    def _paint_end(self, _e):
        if self._painting:
            self._painting = False
            self.changed.emit()

    def _set_pixels(self, sel, add):
        if add:
            self.mask[sel] |= mk.USER
        else:
            self.mask[sel] = 0

    def apply(self, sel, add):
        """Add (user bit) or remove (all bits) the pixels ``sel``; undoable."""
        if not np.any(sel):
            return
        m = self.mask.copy()
        if add:
            m[sel] |= mk.USER
        else:
            m[sel] = 0
        self.set_mask(m)
