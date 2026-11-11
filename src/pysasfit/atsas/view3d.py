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

__all__ = ["COLOURS", "ModelView", "camera_direction", "nice_length", "scale_guides"]

#: colours of the objects (RGBA 0..1): teal (ATSAS), orange, violet, blue, magenta, olive
COLOURS = [(0.10, 0.62, 0.60, 1.0), (0.85, 0.45, 0.05, 1.0), (0.42, 0.27, 0.76, 1.0), (0.18, 0.44, 0.93, 1.0),
           (0.78, 0.14, 0.43, 1.0), (0.55, 0.60, 0.15, 1.0)]


# ------------------------------------------------------------------ size guides (no OpenGL needed)
def nice_length(x: float) -> float:
    """Largest 1, 2 or 5 × 10ⁿ not above ``x`` (> 0)."""
    if not np.isfinite(x) or x <= 0:
        return 0.0
    e = 10.0 ** np.floor(np.log10(x))
    for m in (5.0, 2.0, 1.0):
        if m * e <= x * (1 + 1e-9):
            return float(m * e)
    return float(e)


def _unit(unit: str, extent_A: float):
    """('Å' or 'nm', Å per unit); 'auto': nm from an object size of 100 Å (10 nm) on."""
    if unit == "auto":
        unit = "nm" if extent_A >= 100.0 else "Å"
    return (unit, 10.0) if unit == "nm" else ("Å", 1.0)


def _fmt(value_A, unit, per):
    v = value_A / per
    return f"{v:.3g} {unit}" if abs(v) < 1000 else f"{v:.0f} {unit}"


def camera_direction(azimuth=35.0, elevation=20.0):
    """Unit vector from the scene centre towards a pyqtgraph camera (angles in degrees)."""
    a, e = np.radians(azimuth), np.radians(elevation)
    return np.array([np.cos(e) * np.cos(a), np.cos(e) * np.sin(a), np.sin(e)])


def scale_guides(vertices, bar=True, box=False, box_axes="xyz", unit="auto", view_dir=None):
    """Lines and labels that show the size of the objects (all vertices given, in Å).

    ``bar``: a scale bar of a round length (1, 2 or 5 × 10ⁿ in the shown unit, about 40 % of the largest extent)
    below the objects (z is up in the view), along the horizontal of the screen so that it is not foreshortened,
    with end ticks.  ``box``: the bounding box, along the coordinate axes (``"xyz"``) or along the principal axes of
    the vertices (``"principal"``: length, width and thickness of the particle, independent of its orientation),
    with its three edge lengths written at the edges that meet in the lower corner nearest the camera (so that the
    labels are not drawn over the objects).  ``view_dir``: unit vector from the scene towards the camera (default: the
    starting view of the panel).
    -> (segments, labels, info): segments an (n, 3) array of line end points in pairs (GL 'lines' mode), labels
    [(position, text)], info a short text (bar length, box edges)."""
    v = np.asarray(vertices, float).reshape(-1, 3)
    v = v[np.all(np.isfinite(v), axis=1)]
    if not len(v):
        return np.zeros((0, 3)), [], ""
    d = camera_direction() if view_dir is None else np.asarray(view_dir, float)
    d = d / (np.linalg.norm(d) or 1.0)
    centre = 0.5 * (v.min(axis=0) + v.max(axis=0))
    axes = np.eye(3)
    if box and box_axes == "principal" and len(v) >= 4:
        c = v.mean(axis=0)
        w, vec = np.linalg.eigh(np.cov((v - c).T))
        axes = vec[:, np.argsort(w)[::-1]].T          # rows: longest, middle, shortest axis
        if np.linalg.det(axes) < 0:
            axes[2] *= -1
        centre = c
    p = (v - centre) @ axes.T                          # coordinates in the box frame
    lo, hi = p.min(axis=0), p.max(axis=0)
    ext = hi - lo
    big = float(ext.max()) if ext.max() > 0 else 1.0
    name, per = _unit(unit, big)
    segs, labels, info = [], [], []

    def world(x):
        return centre + np.asarray(x, float) @ axes

    if box:
        corners = np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])])
        for i in range(8):
            for j in range(i + 1, 8):
                if np.sum(corners[i] != corners[j]) == 1:   # neighbours: one coordinate differs
                    segs += [world(corners[i]), world(corners[j])]
        cw = corners @ axes                            # corners relative to the centre, in the scene frame
        low = cw[:, 2] <= np.median(cw[:, 2])          # the lower four (labels below and in front of the objects)
        score = cw @ d - np.where(low, 0.0, np.inf)
        near = corners[int(np.argmax(score))]          # the lower corner nearest the camera
        mid = 0.5 * (lo + hi)
        pad = 0.05 * big
        for k, lab in enumerate("xyz" if box_axes != "principal" else ("length", "width", "thickness")):
            at = near.astype(float).copy()
            at[k] = mid[k]
            for o in range(3):                        # a little outside the box
                if o != k:
                    at[o] += pad * np.sign(near[o] - mid[o])
            labels.append((world(at), _fmt(ext[k], name, per)))
            info.append(f"{lab} {_fmt(ext[k], name, per)}")
    if bar:
        L = nice_length(0.4 * big / per) * per          # Å, round in the shown unit
        h = np.cross([0.0, 0.0, 1.0], d)               # the horizontal of the screen (to the right)
        if np.linalg.norm(h) < 1e-6:                   # looking straight down: along x
            h = np.array([1.0, 0.0, 0.0])
        h /= np.linalg.norm(h)
        if L > 0:
            bottom = v[:, 2].min()                     # below the objects and the box, and its labels
            if box:
                bottom = min(bottom, float((cw[:, 2] + centre[2]).min()))
            mid = np.array([centre[0], centre[1], bottom - 0.3 * big])
            a, b = mid - 0.5 * L * h, mid + 0.5 * L * h
            t = np.array([0.0, 0.0, 0.03 * big])
            segs += [a, b, a - t, a + t, b - t, b + t]
            labels.append((mid - np.array([0.0, 0.0, 0.1 * big]), _fmt(L, name, per)))
            info.insert(0, f"scale bar {_fmt(L, name, per)}")
    return (np.array(segs) if segs else np.zeros((0, 3))), labels, ", ".join(info)


def _overlay_bar_class():
    """GL item that paints a scale bar flat in the lower left corner of the view (like a micrograph).  Its length is
    a round value (1, 2 or 5 × 10ⁿ) of about 20 % of the view width, measured across the line of sight at the
    depth of ``anchor`` (the centre of the objects), and is recomputed at every repaint, so it stays right while the
    view is rotated or zoomed.  None for pyqtgraph without ``mvpMatrix`` (< 0.13): the 3D bar is used then."""
    from pyqtgraph.opengl.GLGraphicsItem import GLGraphicsItem

    if not hasattr(GLGraphicsItem, "mvpMatrix"):
        return None

    class _OverlayBar(GLGraphicsItem):
        def __init__(self, anchor, unit="auto", extent=100.0):
            super().__init__()
            self.anchor = np.asarray(anchor, float)
            self.unit, self.extent = unit, float(extent)
            self.length_text = ""                       # the label of the last paint (tests, info)

        def bar(self, width_px, px_per_A):
            """(length in Å, label, length in pixels) for a view ``width_px`` wide."""
            name, per = _unit(self.unit, self.extent)
            L = nice_length(0.2 * width_px / px_per_A / per) * per
            return L, _fmt(L, name, per), L * px_per_A

        def paint(self):
            view = self.view()
            rect = QtCore.QRectF(view.rect())
            ndc = QtGui.QMatrix4x4()
            ndc.viewport(rect.left(), rect.bottom(), rect.width(), -rect.height())
            P = ndc * self.mvpMatrix()
            c = view.cameraPosition() - view.opts["center"]
            d = np.array([c.x(), c.y(), c.z()], float)
            h = np.cross([0.0, 0.0, 1.0], d)
            if np.linalg.norm(h) < 1e-9:
                h = np.array([1.0, 0.0, 0.0])
            h *= 1.0 / np.linalg.norm(h)
            unit_len = max(self.extent, 1e-9) * 0.1
            p0 = P.map(QtGui.QVector3D(*self.anchor)).toPointF()
            p1 = P.map(QtGui.QVector3D(*(self.anchor + unit_len * h))).toPointF()
            px_per_A = np.hypot(p1.x() - p0.x(), p1.y() - p0.y()) / unit_len
            if not np.isfinite(px_per_A) or px_per_A <= 0:
                return
            _L, text, px = self.bar(rect.width(), px_per_A)
            self.length_text = text
            x0, y0 = rect.left() + 24, rect.bottom() - 22
            painter = QtGui.QPainter(view)
            painter.setRenderHints(QtGui.QPainter.RenderHint.Antialiasing | QtGui.QPainter.RenderHint.TextAntialiasing)
            pen = QtGui.QPen(QtGui.QColor(20, 20, 20))
            pen.setWidthF(2.0)
            painter.setPen(pen)
            painter.drawLine(QtCore.QPointF(x0, y0), QtCore.QPointF(x0 + px, y0))
            for x in (x0, x0 + px):
                painter.drawLine(QtCore.QPointF(x, y0 - 5), QtCore.QPointF(x, y0 + 5))
            font = QtGui.QFont()
            font.setPointSize(11)
            painter.setFont(font)
            fm = QtGui.QFontMetricsF(font)
            painter.drawText(QtCore.QPointF(x0 + 0.5 * px - 0.5 * fm.horizontalAdvance(text), y0 - 9), text)
            painter.end()

    return _OverlayBar


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
        cameraMoved = QtCore.Signal()          # after a rotation / zoom with the mouse (labels follow the camera)

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

        def mouseReleaseEvent(self, ev):
            super().mouseReleaseEvent(ev)
            self.cameraMoved.emit()

        def wheelEvent(self, ev):
            super().wheelEvent(ev)
            self.cameraMoved.emit()

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
        guide = QtWidgets.QHBoxLayout()
        self.show_bar = QtWidgets.QCheckBox("scale bar")
        self.show_bar.setToolTip("a bar of a round length (1, 2 or 5 × 10ⁿ) below the objects")
        self.show_box = QtWidgets.QCheckBox("box")
        self.show_box.setToolTip("bounding box of the shown objects with its edge lengths")
        self.box_axes = QtWidgets.QComboBox()
        self.box_axes.addItem("x y z", "xyz")
        self.box_axes.addItem("principal axes", "principal")
        self.box_axes.setToolTip("box along the coordinate axes, or along the principal axes of the objects "
                                 "(length, width, thickness: independent of the orientation)")
        self.scale_unit = QtWidgets.QComboBox()
        for u in ("auto", "nm", "Å"):
            self.scale_unit.addItem(u)
        self.scale_unit.setToolTip("unit of the labels (auto: nm from an object size of 10 nm on)")
        for w in (self.show_bar, self.show_box):
            w.toggled.connect(lambda c: (self.redraw(), self.reset_camera() if c else None))
        for w in (self.box_axes, self.scale_unit):
            w.currentIndexChanged.connect(lambda _i: self.redraw())
        guide.addWidget(self.show_bar)
        guide.addWidget(self.show_box)
        f.addRow(guide)
        guide2 = QtWidgets.QHBoxLayout()
        guide2.addWidget(self.box_axes, 1)
        guide2.addWidget(self.scale_unit)
        f.addRow(guide2)
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
            self.view.cameraMoved.connect(self._redraw_guides)
            self.view.setBackgroundColor("w")
            self.view.setMinimumSize(200, 200)
            self.view.setSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Expanding)
            lay.addWidget(self.view, 1)
        self._items = []
        self._guide_items = []
        self.guide_info = ""
        self.overlay_bar = None

    def guide_state(self):
        """Scale bar / box settings (for a session file)."""
        return {"bar": self.show_bar.isChecked(), "box": self.show_box.isChecked(),
                "box_axes": self.box_axes.currentData(), "unit": self.scale_unit.currentText()}

    def set_guide_state(self, st):
        st = st or {}
        for w in (self.show_bar, self.show_box, self.box_axes, self.scale_unit):
            w.blockSignals(True)
        self.show_bar.setChecked(bool(st.get("bar", False)))
        self.show_box.setChecked(bool(st.get("box", False)))
        i = self.box_axes.findData(st.get("box_axes", "xyz"))
        self.box_axes.setCurrentIndex(max(i, 0))
        i = self.scale_unit.findText(st.get("unit", "auto"))
        self.scale_unit.setCurrentIndex(max(i, 0))
        for w in (self.show_bar, self.show_box, self.box_axes, self.scale_unit):
            w.blockSignals(False)
        self.redraw()

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
        self._draw_guides(gl)

    def _view_dir(self):
        if self.view is None:
            return None
        try:
            c = self.view.cameraPosition() - self.view.opts["center"]
            d = np.array([c.x(), c.y(), c.z()], float)
            return d if np.linalg.norm(d) > 0 else None
        except Exception:  # noqa: BLE001 -- older pyqtgraph
            return None

    def guides(self):
        """Scale bar and box of the ticked objects: (segments, labels, info) of :func:`scale_guides`."""
        if not (self.show_bar.isChecked() or self.show_box.isChecked()):
            return np.zeros((0, 3)), [], ""
        vs = [v for v, _f, _c, _t in self.meshes() if len(v)]
        if not vs:
            return np.zeros((0, 3)), [], ""
        bar3d = self.show_bar.isChecked() and self._overlay_cls() is None
        return scale_guides(np.vstack(vs), bar=bar3d, box=self.show_box.isChecked(),
                            box_axes=self.box_axes.currentData(), unit=self.scale_unit.currentText(),
                            view_dir=self._view_dir())

    def _overlay_cls(self):
        if self.view is None:
            return None
        if not hasattr(self, "_overlay_class"):
            try:
                self._overlay_class = _overlay_bar_class()
            except Exception:  # noqa: BLE001
                self._overlay_class = None
        return self._overlay_class

    def _redraw_guides(self):
        if self.view is not None and (self.show_bar.isChecked() or self.show_box.isChecked()):
            import pyqtgraph.opengl as gl
            self._draw_guides(gl)

    def _draw_guides(self, gl):
        for it in self._guide_items:
            self.view.removeItem(it)
        self._guide_items = []
        segs, labels, info = self.guides()
        self.overlay_bar = None
        cls = self._overlay_cls()
        vs = [v for v, _f, _c, _t in self.meshes() if len(v)] if self.show_bar.isChecked() and cls else []
        if vs:
            allv = np.vstack(vs)
            lo, hi = allv.min(axis=0), allv.max(axis=0)
            self.overlay_bar = cls(0.5 * (lo + hi), self.scale_unit.currentText(), float((hi - lo).max()))
            self.view.addItem(self.overlay_bar)
            self._guide_items.append(self.overlay_bar)
            info = ", ".join(x for x in ("scale bar: lower left", info) if x)
        if info != self.guide_info or info:
            self._show_info()
            if info:
                self.info.setText((self.info.text() + "\n" if self.info.text() else "") + info)
        self.guide_info = info
        if not len(segs):
            return
        line = gl.GLLinePlotItem(pos=segs.astype(np.float32), mode="lines", color=(0.1, 0.1, 0.1, 1.0), width=2,
                                 antialias=True, glOptions="translucent")
        self.view.addItem(line)
        self._guide_items.append(line)
        if not hasattr(gl, "GLTextItem"):                # pyqtgraph < 0.13: the lengths are in the info text
            return
        font = QtGui.QFont()
        font.setPointSize(11)
        for pos, text in labels:
            t = gl.GLTextItem(pos=np.asarray(pos, float), text=text, color=(20, 20, 20, 255), font=font)
            self.view.addItem(t)
            self._guide_items.append(t)

    def reset_camera(self):
        if self.view is None:
            return
        ext = 50.0
        for v, _f, _c, _t in self.meshes():
            if len(v):
                ext = max(ext, float(np.abs(v).max()))
        segs, labels, _info = self.guides()               # the scale bar and the box must fit as well
        for p in [*segs, *(pos for pos, _t in labels)]:
            ext = max(ext, float(np.linalg.norm(p)))
        self.view.setCameraPosition(distance=(2.4 if len(segs) else 1.9) * ext, elevation=20, azimuth=35)
        self.view.opts["center"] = QtGui.QVector3D(0, 0, 0)
        self._redraw_guides()
        self.view.update()

    def on_save_image(self):
        if self.view is None:
            return
        fn, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Image", "model3d.png", "PNG (*.png)")
        if fn:
            self.view.grabFramebuffer().save(fn)
