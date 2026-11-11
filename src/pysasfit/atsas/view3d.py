"""3D panel of pyATsas: bead models (DAMMIF, DAMMIN, MONSA, GASBOR, DAMAVER ...) and spherical-harmonics envelopes
(SASHA) drawn with pyqtgraph.opengl inside the Qt window, several objects at once (e.g. the envelope over the
beads).  Needs ``pyqtgraph`` and ``PyOpenGL`` (as the 3D panel of pySASmorph); without them the panel shows the
installation hint.  Lengths in Å, as in the model files.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets

from . import models as M

__all__ = ["ModelView", "COLOURS"]

#: colours of the objects (RGBA 0..1): teal (ATSAS), orange, violet, blue, magenta, olive
COLOURS = [(0.10, 0.62, 0.60, 1.0), (0.85, 0.45, 0.05, 1.0), (0.42, 0.27, 0.76, 1.0), (0.18, 0.44, 0.93, 1.0),
           (0.78, 0.14, 0.43, 1.0), (0.55, 0.60, 0.15, 1.0)]


def headlight_shader():
    """Two-sided 'headlight' variant of pyqtgraph's 'shaded' shader (light along the view direction, so every
    surface facing the camera is lit); 'shaded' if the shader source has an unexpected form."""
    from pyqtgraph.opengl import shaders

    if "pyatsas" in shaders.ShaderProgram.names:
        return "pyatsas"
    try:
        base = shaders.ShaderProgram.names["shaded"]
        vert, frag = base.shaders[0], base.shaders[1]
        code = frag.code
        light = "normalize(vec3(1.0, -1.0, -1.0))"
        clamp = "p = p < 0. ? 0. : p * 0.8;"
        if light not in code or clamp not in code:
            return "shaded"
        code = code.replace(light, "normalize(vec3(0.35, 0.45, 1.0))").replace(clamp, "p = abs(p) * 0.72;")
        code = code.replace("(0.2 + p)", "(0.32 + p)")
        shaders.ShaderProgram("pyatsas", [shaders.VertexShader(vert.code), shaders.FragmentShader(code)])
        return "pyatsas"
    except Exception:  # noqa: BLE001
        return "shaded"


def _view_class():
    """GLViewWidget that survives a new OpenGL context: Qt creates one when the widget moves to another window
    (a detached panel); pyqtgraph keeps compiled shader programs in a global cache that would then be invalid
    ("GLError 1281 in glGetAttribLocation").  On every new context the cache is cleared and ``contextReset``
    asks the owner to rebuild its items."""
    import pyqtgraph.opengl as gl
    from pyqtgraph.opengl import shaders

    class _View(gl.GLViewWidget):
        contextReset = QtCore.Signal()

        def __init__(self, *a, **k):
            self._n_init = 0
            super().__init__(*a, **k)

        def initializeGL(self):
            for sp in shaders.ShaderProgram.names.values():
                sp.prog = None
            super().initializeGL()
            self._n_init += 1
            if self._n_init > 1:
                QtCore.QTimer.singleShot(0, self.contextReset.emit)

    return _View


def _gl_problem():
    try:
        from ..morph.view3d_gl import gl_problem
        return gl_problem()
    except Exception:  # noqa: BLE001
        try:
            import importlib
            importlib.import_module("pyqtgraph.opengl")
            return None
        except Exception as e:  # noqa: BLE001
            return f"pyqtgraph.opengl is not available: {e}\n\npip install pyqtgraph PyOpenGL"


class ModelView(QtWidgets.QWidget):
    """List of objects (tick to show) + OpenGL view.  ``add_beads(model, name)``, ``add_envelope(env, name)``."""
    status = QtCore.Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.objects = []          # dicts: name, kind, data, colour, item
        lay = QtWidgets.QHBoxLayout(self)
        lay.setContentsMargins(2, 2, 2, 2)
        side_w = QtWidgets.QWidget()
        side_w.setMaximumWidth(250)
        side_w.setSizePolicy(QtWidgets.QSizePolicy.Policy.Maximum, QtWidgets.QSizePolicy.Policy.Preferred)
        side = QtWidgets.QVBoxLayout(side_w)
        side.setContentsMargins(0, 0, 0, 0)
        self.list = QtWidgets.QListWidget()
        self.list.itemChanged.connect(lambda _it: self.redraw())
        self.list.currentRowChanged.connect(lambda _r: self._show_info())
        side.addWidget(self.list, 1)
        self.info = QtWidgets.QLabel("")
        self.info.setWordWrap(True)
        self.info.setTextInteractionFlags(QtCore.Qt.TextInteractionFlag.TextSelectableByMouse)
        side.addWidget(self.info)
        f = QtWidgets.QFormLayout()
        self.bead_scale = QtWidgets.QDoubleSpinBox()
        self.bead_scale.setRange(0.1, 2.0)
        self.bead_scale.setSingleStep(0.1)
        self.bead_scale.setValue(1.0)
        self.bead_scale.setToolTip("Drawn bead radius / bead radius of the model (smaller shows the inside)")
        self.bead_scale.valueChanged.connect(lambda _v: self.redraw())
        f.addRow("bead size", self.bead_scale)
        self.opacity = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.opacity.setRange(10, 100)
        self.opacity.setValue(45)
        self.opacity.setToolTip("Opacity of envelopes (beads inside stay visible)")
        self.opacity.valueChanged.connect(lambda _v: self.redraw())
        f.addRow("envelope", self.opacity)
        self.centre = QtWidgets.QCheckBox("centre each object")
        self.centre.setChecked(True)
        self.centre.setToolTip("Move every object to its centre of mass (bead models and SASHA envelopes are "
                               "not in a common frame; superposition needs SUPCOMB / CIFSUP or MASSHA)")
        self.centre.toggled.connect(lambda _c: self.redraw())
        f.addRow(self.centre)
        side.addLayout(f)
        grid = QtWidgets.QGridLayout()
        for i, (text, slot) in enumerate((("Load…", self.on_load), ("Remove", self.on_remove),
                                          ("Reset view", self.reset_camera), ("Image…", self.on_save_image))):
            b = QtWidgets.QPushButton(text)
            b.clicked.connect(lambda _c=False, s=slot: s())
            grid.addWidget(b, i // 2, i % 2)
        side.addLayout(grid)
        lay.addWidget(side_w)
        self.view = None
        prob = _gl_problem()
        if prob:
            msg = QtWidgets.QPlainTextEdit(prob)
            msg.setReadOnly(True)
            lay.addWidget(msg, 1)
            self.gl_error = prob
        else:
            self.gl_error = None
            self.view = _view_class()()
            self.view.contextReset.connect(self.redraw)
            self.view.setBackgroundColor("w")
            self.view.setMinimumSize(200, 200)
            self.view.setSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Expanding)
            lay.addWidget(self.view, 1)
        self._items = []

    # ------------------------------------------------------------------ objects
    def _add(self, name, kind, data):
        for o in self.objects:                      # same name: replace
            if o["name"] == name:
                o["data"], o["kind"] = data, kind
                self.redraw()
                return o
        o = {"name": name, "kind": kind, "data": data, "colour": COLOURS[len(self.objects) % len(COLOURS)]}
        self.objects.append(o)
        it = QtWidgets.QListWidgetItem(name)
        it.setFlags(it.flags() | QtCore.Qt.ItemFlag.ItemIsUserCheckable)
        it.setCheckState(QtCore.Qt.CheckState.Checked)
        pm = QtGui.QPixmap(12, 12)
        pm.fill(QtGui.QColor.fromRgbF(*o["colour"]))
        it.setIcon(QtGui.QIcon(pm))
        self.list.blockSignals(True)
        self.list.addItem(it)
        self.list.blockSignals(False)
        self.list.setCurrentRow(self.list.count() - 1)
        self.redraw()
        self.reset_camera()
        return o

    def add_beads(self, model: M.BeadModel, name=None):
        return self._add(name or model.path.name, "beads", model)

    def add_envelope(self, env: M.FlmEnvelope, name=None):
        return self._add(name or (env.path.name if env.path else "envelope"), "envelope", env)

    def add_mesh(self, name, vertices, faces, translucent=False, info=""):
        """Any surface (e.g. an isosurface of a density map), coordinates in Å."""
        return self._add(name, "mesh", {"v": np.asarray(vertices, float), "f": np.asarray(faces, np.int32),
                                        "translucent": translucent, "info": info})

    def load_file(self, path):
        path = Path(path)
        if path.suffix.lower() == ".flm":
            env = M.read_flm(path)
            if not env.verified:
                raise M.AtsasError(f"{path.name}: {env.check}. Not drawn; open it in MASSHA.")
            return self.add_envelope(env)
        return self.add_beads(M.read_bead_model(path))

    def on_load(self):
        files, _ = QtWidgets.QFileDialog.getOpenFileNames(self, "Bead models or envelopes", "",
                                                          "models (*.pdb *.cif *.ent *.flm);;All files (*)")
        for f in files:
            try:
                self.load_file(f)
            except Exception as e:  # noqa: BLE001
                QtWidgets.QMessageBox.warning(self, "3D model", str(e))

    def on_remove(self):
        r = self.list.currentRow()
        if 0 <= r < len(self.objects):
            self.objects.pop(r)
            self.list.takeItem(r)
            self.redraw()

    def clear(self):
        self.objects.clear()
        self.list.clear()
        self.redraw()

    def _show_info(self):
        r = self.list.currentRow()
        if not 0 <= r < len(self.objects):
            self.info.setText("")
            return
        o = self.objects[r]
        d = o["data"]
        if o["kind"] == "beads":
            self.info.setText(f"{d.n} beads, radius {d.radius:.3g} Å ({d.radius_from})\nRg {d.rg():.4g} Å, "
                              f"Dmax {d.dmax():.4g} Å\n{d.path}")
        elif o["kind"] == "mesh":
            self.info.setText(d.get("info", ""))
        else:
            self.info.setText(f"envelope, L = {d.lmax}\nRg {d.rg:.4g} Å, V {d.volume:.4g} Å³\n{d.check}")

    # ------------------------------------------------------------------ drawing
    def meshes(self):
        """[(vertices, faces, colour, translucent)] of the ticked objects (no OpenGL needed)."""
        out = []
        for i, o in enumerate(self.objects):
            it = self.list.item(i)
            if it is None or it.checkState() != QtCore.Qt.CheckState.Checked:
                continue
            d = o["data"]
            if o["kind"] == "beads":
                v, f = M.beads_mesh(d, self.bead_scale.value(), rows=5 if d.n > 2000 else 6,
                                    cols=8 if d.n > 2000 else 10)
                c = d.xyz.mean(axis=0)
                translucent = False
            elif o["kind"] == "mesh":
                v, f = d["v"], d["f"]
                c = v.mean(axis=0) if len(v) else np.zeros(3)
                translucent = bool(d.get("translucent"))
            else:
                v, f = M.envelope_mesh(d)
                c = v.mean(axis=0)
                translucent = True
            if self.centre.isChecked():
                v = v - c
            out.append((v, f, o["colour"], translucent))
        return out

    def redraw(self):
        self._show_info()
        if self.view is None:
            return
        import pyqtgraph.opengl as gl
        for it in self._items:
            self.view.removeItem(it)
        self._items = []
        shader = headlight_shader()
        alpha = self.opacity.value() / 100
        for v, f, col, translucent in sorted(self.meshes(), key=lambda m: m[3]):   # opaque first
            if not len(v) or not len(f):
                continue
            md = gl.MeshData(vertexes=v.astype(np.float32), faces=f)
            colour = (col[0], col[1], col[2], alpha if translucent else 1.0)
            item = gl.GLMeshItem(meshdata=md, smooth=not translucent, shader=shader, color=colour,
                                 glOptions="translucent" if translucent else "opaque")
            self.view.addItem(item)
            self._items.append(item)

    def reset_camera(self):
        if self.view is None:
            return
        ext = 50.0
        for v, _f, _c, _t in self.meshes():
            if len(v):
                ext = max(ext, float(np.abs(v).max()))
        self.view.setCameraPosition(distance=1.9 * ext, elevation=20, azimuth=35)
        self.view.opts["center"] = QtGui.QVector3D(0, 0, 0)
        self.view.update()

    def on_save_image(self):
        if self.view is None:
            return
        fn, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Image", "model3d.png", "PNG (*.png)")
        if fn:
            self.view.grabFramebuffer().save(fn)
