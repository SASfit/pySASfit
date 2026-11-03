"""3D panel of pySASmorph: interface surfaces drawn with pyqtgraph.opengl, inside the Qt window.

Needs ``pyqtgraph`` and ``PyOpenGL`` (pure Python, ``pip install pyqtgraph PyOpenGL``); marching
cubes from scikit-image when installed (faster), else from pyqtgraph. Without pyqtgraph the panel
shows the import error and the installation hint instead.

Controls: phase (each phase of a two- or multi-phase result, or all), surface (exact cut of the field, smoothed voxels, voxels),
largest cluster in a second colour, clipping plane (axis and position), box outline, view reset,
image export. The meshes are computed only while the panel is visible.
"""

from __future__ import annotations

import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets

from .surface import label_surface, largest_cluster, marching_cubes_backend, vertices_in

__all__ = ["MorphGLView", "gl_problem", "phase_data", "phase_colour"]

PORE_COLOUR = (0.06, 0.49, 0.50, 1.0)      # teal, the pySASmorph colour
SOLID_COLOUR = (0.78, 0.76, 0.72, 1.0)     # light stone
#: colour of phase i (phase 0 stone, phase 1 teal, then orange, violet, blue, olive, ...)
PHASE_COLOURS = [SOLID_COLOUR, PORE_COLOUR, (0.85, 0.45, 0.05, 1.0), (0.42, 0.27, 0.76, 1.0),
                 (0.18, 0.44, 0.93, 1.0), (0.55, 0.60, 0.15, 1.0), (0.60, 0.30, 0.20, 1.0)]


def phase_colour(i):
    return PHASE_COLOURS[i % len(PHASE_COLOURS)]


def phase_data(res):
    """(labels, levels, names, psi) of a two-phase MorphResult or a multi-phase MultiResult; labels None
    before a reconstruction."""
    if res is None:
        return None, None, [], None
    if hasattr(res, "labels"):
        return res.labels, res.levels, list(res.params.names), res.psi
    if getattr(res, "matrix", None) is None:
        return None, None, ["solid", "pore"], None
    lev = getattr(res, "level", np.nan)
    return (res.matrix.astype(np.int8), (np.array([lev]) if np.isfinite(lev) else None), ["solid", "pore"],
            res.psi)
LARGEST_COLOUR = (0.78, 0.14, 0.43, 1.0)   # magenta, as SAXSMorph's highlighted pore


def headlight_shader():
    """Name of a two-sided 'headlight' variant of pyqtgraph's 'shaded' shader (light along the view
    direction, |n . l|, so every surface facing the camera is lit, cut faces included); falls back to
    'shaded' if the pyqtgraph shader source has an unexpected form."""
    from pyqtgraph.opengl import shaders

    if "sasmorph" in shaders.ShaderProgram.names:
        return "sasmorph"
    try:
        base = shaders.ShaderProgram.names["shaded"]
        vert, frag = base.shaders[0], base.shaders[1]
        code = frag.code
        light = "normalize(vec3(1.0, -1.0, -1.0))"
        clamp = "p = p < 0. ? 0. : p * 0.8;"
        if light not in code or clamp not in code:
            return "shaded"
        code = code.replace(light, "normalize(vec3(0.35, 0.45, 1.0))").replace(clamp, "p = abs(p) * 0.72;")
        code = code.replace("(0.2 + p)", "(0.28 + p)")
        shaders.ShaderProgram("sasmorph", [shaders.VertexShader(vert.code), shaders.FragmentShader(code)])
        return "sasmorph"
    except Exception:  # noqa: BLE001
        return "shaded"


def gl_problem():
    """None if pyqtgraph.opengl imports, else the error text."""
    try:
        import pyqtgraph.opengl  # noqa: F401
        return None
    except Exception as e:  # noqa: BLE001
        import sys
        return (f"import pyqtgraph.opengl failed: {type(e).__name__}: {e}\n"
                f"Python: {sys.executable}\n\nInstall for this Python:\n"
                f"    \"{sys.executable}\" -m pip install pyqtgraph PyOpenGL scikit-image")


class MorphGLView(QtWidgets.QWidget):
    status = QtCore.Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.result = None
        self.length_unit = "Å"
        self._dirty = False
        self._items = []
        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(2, 2, 2, 2)
        problem = gl_problem()
        self.view = None
        if problem:
            msg = QtWidgets.QPlainTextEdit(problem + "\n\nAlternatively: File → Export voxel matrix (VTK) "
                                           "and open it in ParaView.")
            msg.setReadOnly(True)
            lay.addWidget(msg)
            return
        import pyqtgraph.opengl as gl

        self._gl = gl
        bar = QtWidgets.QHBoxLayout()
        self.phase = QtWidgets.QComboBox()
        self._names = []
        self._set_phase_items(["solid", "pore"])
        self.source = QtWidgets.QComboBox()
        self.source.addItem("exact cut of ψ", "field")
        self.source.addItem("voxels, smoothed", "smoothed")
        self.source.addItem("voxels", "voxels")
        self.source.setToolTip("exact cut: iso-surface of the Gaussian field at the cut level (smooth);\n"
                               "voxels: the binary matrix as it is exported")
        self.largest = QtWidgets.QCheckBox("largest cluster")
        self.largest.setToolTip("Largest connected cluster of the shown phase in magenta")
        self.clip_axis = QtWidgets.QComboBox()
        self.clip_axis.addItems(["no clip", "clip x", "clip y", "clip z"])
        self.clip_pos = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.clip_pos.setRange(0, 0)
        self.clip_pos.setEnabled(False)
        self.box = QtWidgets.QCheckBox("box")
        self.box.setChecked(True)
        reset = QtWidgets.QToolButton()
        reset.setText("reset view")
        reset.clicked.connect(self.reset_camera)
        save = QtWidgets.QToolButton()
        save.setText("save image…")
        save.clicked.connect(self.on_save_image)
        for w in (self.phase, self.source, self.largest, self.clip_axis):
            bar.addWidget(w)
        bar.addWidget(self.clip_pos, 1)
        for w in (self.box, reset, save):
            bar.addWidget(w)
        lay.addLayout(bar)
        self.view = gl.GLViewWidget()
        self.view.setBackgroundColor("w")
        self.view.setMinimumSize(150, 150)
        lay.addWidget(self.view, 1)
        self.info = QtWidgets.QLabel("Reconstruct to see the structure")
        self.info.setStyleSheet("color: grey")
        lay.addWidget(self.info)
        self.phase.currentIndexChanged.connect(self.redraw)
        self._prefer_field = True
        self.source.currentIndexChanged.connect(lambda i: setattr(self, "_prefer_field", i == 0))
        self.source.currentIndexChanged.connect(self.redraw)
        self.largest.toggled.connect(self.redraw)
        self.box.toggled.connect(self.redraw)
        self.clip_axis.currentIndexChanged.connect(self._on_clip_axis)
        self._clip_timer = QtCore.QTimer(self, singleShot=True, interval=150)
        self._clip_timer.timeout.connect(self.redraw)
        self.clip_pos.valueChanged.connect(lambda _v: self._clip_timer.start())

    def _set_phase_items(self, names):
        if names == self._names:
            return
        cur = self.phase.currentText()
        self.phase.blockSignals(True)
        self.phase.clear()
        for i, n in enumerate(names):
            pm = QtGui.QPixmap(12, 12)
            pm.fill(QtGui.QColor.fromRgbF(*phase_colour(i)))
            self.phase.addItem(QtGui.QIcon(pm), n, i)
        self.phase.addItem("all phases", -1)
        j = self.phase.findText(cur)
        self.phase.setCurrentIndex(j if j >= 0 else (1 if len(names) == 2 else len(names)))
        self.phase.blockSignals(False)
        self._names = list(names)

    # ------------------------------------------------------------------ state
    def state(self):
        if self.view is None:
            return {}
        return {"phase": self.phase.currentText(), "source": self.source.currentIndex(),
                "largest": self.largest.isChecked(), "box": self.box.isChecked()}

    def set_state(self, s):
        if self.view is None or not s:
            return
        if isinstance(s.get("phase"), str) and self.phase.findText(s["phase"]) >= 0:
            self.phase.setCurrentIndex(self.phase.findText(s["phase"]))
        if isinstance(s.get("source"), int) and 0 <= s["source"] < self.source.count():
            self.source.setCurrentIndex(s["source"])                 # source 0 also sets _prefer_field
        self.largest.setChecked(bool(s.get("largest", False)))
        self.box.setChecked(bool(s.get("box", True)))

    def set_result(self, res, length_unit="Å"):
        self.result = res
        self.length_unit = length_unit
        if self.view is None:
            return
        labels, levels, names, psi = phase_data(res)
        if names:
            self._set_phase_items(names)
        n = labels.shape[0] if labels is not None else 0
        self.clip_pos.blockSignals(True)
        self.clip_pos.setRange(0, max(0, n - 1))
        self.clip_pos.setValue(max(0, n // 2))
        self.clip_pos.blockSignals(False)
        if labels is not None:
            has_field = psi is not None and levels is not None
            self.source.model().item(0).setEnabled(bool(has_field))
            self.source.blockSignals(True)
            if not has_field and self.source.currentIndex() == 0:
                self.source.setCurrentIndex(1)
            elif has_field and self._prefer_field:
                self.source.setCurrentIndex(0)
            self.source.blockSignals(False)
        self._dirty = True
        if self.isVisible():
            self.redraw()
            self.reset_camera()

    def showEvent(self, event):
        super().showEvent(event)
        if self._dirty and self.view is not None:
            self.redraw()
            self.reset_camera()

    def _on_clip_axis(self):
        self.clip_pos.setEnabled(self.clip_axis.currentIndex() > 0)
        self.redraw()

    def clip(self):
        i = self.clip_axis.currentIndex()
        return None if i == 0 else (i - 1, self.clip_pos.value())

    # ------------------------------------------------------------------ drawing
    def meshes(self):
        """[(mesh, colours or colour)] for the current settings (no drawing; used by the tests and STL export)."""
        res = self.result
        labels, levels, names, psi = phase_data(res)
        if labels is None:
            return []
        out = []
        sel = self.phase.currentData()
        phases = list(range(len(names))) if sel is None or sel < 0 else [sel]
        src = self.source.currentData()
        for ph in phases:
            m = label_surface(labels, ph, res.voxel_size, src, psi=psi, levels=levels, clip=self.clip())
            if m is None:
                continue
            colour = phase_colour(ph)
            if self.largest.isChecked():
                big = largest_cluster(labels == ph)
                hit = vertices_in(m, big, res.voxel_size)
                cols = np.tile(np.asarray(colour, np.float32), (len(m.vertices), 1))
                cols[hit] = LARGEST_COLOUR
                out.append((m, cols))
            else:
                out.append((m, colour))
        return out

    def redraw(self):
        if self.view is None:
            return
        if not self.isVisible():
            self._dirty = True
            return
        self._dirty = False
        gl = self._gl
        for it in self._items:
            self.view.removeItem(it)
        self._items = []
        res = self.result
        if phase_data(res)[0] is None:
            self.info.setText("Reconstruct to see the structure")
            return
        QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.CursorShape.WaitCursor)
        try:
            meshes = self.meshes()
        except Exception as e:  # noqa: BLE001
            QtWidgets.QApplication.restoreOverrideCursor()
            self.info.setText(f"surface failed: {e}")
            return
        QtWidgets.QApplication.restoreOverrideCursor()
        ntri = 0
        shader = headlight_shader()
        for m, col in meshes:
            md = gl.MeshData(vertexes=m.vertices, faces=m.faces)
            if isinstance(col, np.ndarray):
                md.setVertexColors(col)
                item = gl.GLMeshItem(meshdata=md, smooth=True, shader=shader, glOptions="opaque")
            else:
                item = gl.GLMeshItem(meshdata=md, smooth=True, color=col, shader=shader, glOptions="opaque")
            self.view.addItem(item)
            self._items.append(item)
            ntri += len(m)
        if self.box.isChecked():
            self._add_box()
        L = res.params.box_size
        self.info.setText(f"{ntri:,} triangles · box {L:.4g} {self.length_unit} · marching cubes: "
                          f"{marching_cubes_backend()} · drag: rotate, right/wheel: zoom, middle: pan")
        self.status.emit(f"3D: {ntri:,} triangles")

    def _add_box(self):
        res = self.result
        a = res.voxel_size
        lo, hi = -0.5 * a, (phase_data(res)[0].shape[0] - 0.5) * a
        c = np.array([[x, y, z] for x in (lo, hi) for y in (lo, hi) for z in (lo, hi)], np.float32)
        edges = [(0, 1), (2, 3), (4, 5), (6, 7), (0, 2), (1, 3), (4, 6), (5, 7), (0, 4), (1, 5), (2, 6), (3, 7)]
        pts = np.array([c[i] for e in edges for i in e], np.float32)
        item = self._gl.GLLinePlotItem(pos=pts, mode="lines", color=(0.35, 0.35, 0.35, 1.0), width=1.0,
                                       antialias=True)
        self.view.addItem(item)
        self._items.append(item)

    def reset_camera(self):
        labels = phase_data(self.result)[0]
        if self.view is None or labels is None:
            return
        n, a = labels.shape[0], self.result.voxel_size
        c = (n - 1) * a / 2
        self.view.opts["center"] = QtGui.QVector3D(c, c, c)
        self.view.setCameraPosition(distance=3.3 * n * a, elevation=22, azimuth=35)
        self.view.update()

    def grab_image(self):
        return self.view.grabFramebuffer() if self.view is not None else None

    def on_save_image(self):
        if self.view is None:
            return
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save 3D image", "sasmorph_3d.png",
                                                        "PNG (*.png);;JPEG (*.jpg);;TIFF (*.tif)")
        if path:
            self.grab_image().save(path)
            self.status.emit(f"written: {path}")
