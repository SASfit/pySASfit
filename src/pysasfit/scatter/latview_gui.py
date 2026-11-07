"""'Lattice view' panel of pySCAtter: the crystal block with its particles and its orientation to the beam and the
detector (pyqtgraph.opengl), the view along the beam, the reciprocal lattice with the Bragg condition, and the
reflections on the detector.  Geometry from :mod:`latview` (Qt-free).

Needs ``pyqtgraph`` and ``PyOpenGL`` for the 3D views (the 'morph' extra); the detector view (matplotlib) works
without them.
"""
from __future__ import annotations

import numpy as np
from PySide6 import QtCore, QtWidgets

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure

from . import latview as LV

__all__ = ["LatticeView"]

PARTICLE = (0.63, 0.32, 0.18, 1.0)        # sienna, the pySCAtter colour
SHELL = (0.85, 0.65, 0.35, 0.28)
SLAB = (0.63, 0.32, 0.18, 0.55)
SURFACE = (0.18, 0.45, 0.75, 1.0)
CELL = (0.15, 0.15, 0.15, 1.0)
BEAM = (0.85, 0.10, 0.10, 1.0)
DIRECTOR = (0.95, 0.55, 0.0, 1.0)
AXES_COL = {"a": (0.80, 0.10, 0.10, 1), "b": (0.10, 0.55, 0.10, 1), "c": (0.10, 0.25, 0.85, 1)}
ON = (0.85, 0.10, 0.10, 1.0)
OFF = (0.55, 0.55, 0.55, 0.55)

VIEWS = [("real space 3D", "real"), ("real space along the beam", "beam"), ("reciprocal lattice", "recip"),
         ("detector: Bragg reflections", "det")]


def gl_problem():
    try:
        import pyqtgraph.opengl  # noqa: F401
        return None
    except Exception as e:  # noqa: BLE001
        import sys
        return (f"3D view needs pyqtgraph and PyOpenGL: {type(e).__name__}: {e}\n"
                f"    \"{sys.executable}\" -m pip install pyqtgraph PyOpenGL")


class LatticeView(QtWidgets.QWidget):
    """Panel; call :meth:`set_params` when the parameters change (redrawn only while visible)."""
    status = QtCore.Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.p = None
        self.scene = None
        self._dirty = False
        self._items = []
        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(2, 2, 2, 2)
        bar = QtWidgets.QHBoxLayout()
        self.view_mode = QtWidgets.QComboBox()
        for t, d in VIEWS:
            self.view_mode.addItem(t, d)
        self.view_mode.setToolTip("real space: unit cells and particles, beam (red) and detector plane;\n"
                                  "along the beam: parallel projection as seen from the detector (q_h →, q_v ↑);\n"
                                  "reciprocal lattice: reflections (red: on the Ewald sphere / detector);\n"
                                  "detector: positions of the Bragg reflections (powder: rings) with hkl")
        self.ncell = QtWidgets.QSpinBox()
        self.ncell.setRange(1, 5)
        self.ncell.setValue(2)
        self.ncell.setPrefix("cells ")
        self.ncell.setToolTip("Unit cells per direction of the drawn block")
        self.axes_mode = QtWidgets.QComboBox()
        self.axes_mode.addItem("particle axes: as in the model", "model")
        self.axes_mode.addItem("particle axes: along the lattice", "lattice")
        self.axes_mode.setToolTip("as in the model: along the director (θ, φ) for oriented distributions, random "
                                  "for 'isotropic';\nalong the lattice: along the cell c axis / rod axis")
        self.labels = QtWidgets.QCheckBox("hkl")
        self.labels.setChecked(True)
        self.labels.setToolTip("Label the reflections")
        reset = QtWidgets.QToolButton()
        reset.setText("reset view")
        reset.clicked.connect(self.reset_camera)
        save = QtWidgets.QToolButton()
        save.setText("save image…")
        save.clicked.connect(self.on_save_image)
        from ..core.flowlayout import flow_bar
        bar.addWidget(flow_bar(self.view_mode, self.ncell, self.axes_mode, self.labels, reset, save), 1)
        lay.addLayout(bar)
        self.stack = QtWidgets.QStackedWidget()
        lay.addWidget(self.stack, 1)
        self.view = None
        problem = gl_problem()
        if problem:
            msg = QtWidgets.QPlainTextEdit(problem)
            msg.setReadOnly(True)
            self.stack.addWidget(msg)
        else:
            import pyqtgraph.opengl as gl
            self._gl = gl
            self.view = gl.GLViewWidget()
            self.view.setBackgroundColor("w")
            self.view.setMinimumSize(150, 150)
            self.stack.addWidget(self.view)
        self.fig = Figure(figsize=(4, 4), layout="constrained")
        self.canvas = FigureCanvasQTAgg(self.fig)
        self.stack.addWidget(self.canvas)
        self.info = QtWidgets.QLabel("")
        self.info.setWordWrap(True)
        self.info.setStyleSheet("color: grey")
        lay.addWidget(self.info)
        self.view_mode.currentIndexChanged.connect(lambda _i: self._on_view_changed())
        self.ncell.valueChanged.connect(lambda _v: self.rebuild())
        self.axes_mode.currentIndexChanged.connect(lambda _i: self.rebuild())
        self.labels.toggled.connect(lambda _c: self.redraw())

    # ------------------------------------------------------------------ state
    def state(self):
        return {"view": self.view_mode.currentData(), "ncell": self.ncell.value(),
                "axes": self.axes_mode.currentData(), "labels": self.labels.isChecked()}

    def set_state(self, s):
        if not s:
            return
        for cb, key in ((self.view_mode, "view"), (self.axes_mode, "axes")):
            i = cb.findData(s.get(key))
            if i >= 0:
                cb.setCurrentIndex(i)
        if isinstance(s.get("ncell"), int):
            self.ncell.setValue(s["ncell"])
        self.labels.setChecked(bool(s.get("labels", True)))

    # ------------------------------------------------------------------ updates
    def set_params(self, p):
        self.p = dict(p)
        self._dirty = True
        if self.isVisible():
            self.rebuild()

    def showEvent(self, ev):
        super().showEvent(ev)
        if self._dirty:
            QtCore.QTimer.singleShot(0, self.rebuild)

    def rebuild(self):
        if self.p is None:
            return
        self._dirty = False
        try:
            self.scene = LV.build_scene(self.p, self.ncell.value(), self.axes_mode.currentData())
        except Exception as e:  # noqa: BLE001
            self.scene = None
            self.info.setText(f"no sketch: {type(e).__name__}: {e}")
            return
        self.redraw(reset=True)

    def _on_view_changed(self):
        self.redraw(reset=True)

    def redraw(self, reset=False):
        sc = self.scene
        if sc is None:
            return
        mode = self.view_mode.currentData()
        notes = list(sc.notes)
        title = sc.title + (f" — {sc.particle.get('name', '')}" if sc.kind else "")
        if mode == "det" or self.view is None:
            self.stack.setCurrentWidget(self.canvas)
            self._draw_detector(sc)
            if mode != "det":
                notes.insert(0, "3D views need pyqtgraph + PyOpenGL")
        else:
            self.stack.setCurrentWidget(self.view)
            self._clear()
            if mode == "recip":
                self._draw_recip(sc)
            else:
                self._draw_real(sc, frame=(mode == "real"))
            if reset:
                self.reset_camera()
        n_on = int(np.count_nonzero(sc.on_detector)) if not sc.rings else 0
        extra = ""
        if sc.kind:
            extra = (f"  ·  {n_on} reflections on the detector" if not sc.rings else "  ·  rings")
            extra += "  ·  " + ("flat Ewald sphere (preset q map)" if not np.isfinite(sc.k)
                               else f"Ewald sphere k = {sc.k:.4g} 1/nm")
        self.info.setText(title + extra + ("\n" + "; ".join(notes) if notes else ""))

    # ------------------------------------------------------------------ GL helpers
    def _clear(self):
        for it in self._items:
            try:
                self.view.removeItem(it)
            except Exception:  # noqa: BLE001
                pass
        self._items = []

    def _add(self, item):
        self.view.addItem(item)
        self._items.append(item)
        return item

    def _mesh(self, V, F, colour, translucent=False):
        if len(V) == 0:
            return None
        gl = self._gl
        md = gl.MeshData(vertexes=np.asarray(V, np.float32), faces=np.asarray(F, np.uint32))
        it = gl.GLMeshItem(meshdata=md, color=colour, shader=self._shader(), smooth=True,
                           glOptions="translucent" if translucent else "opaque")
        return self._add(it)

    def _shader(self):
        if not hasattr(self, "_shader_name"):
            try:
                from ..morph.view3d_gl import headlight_shader
                self._shader_name = headlight_shader()
            except Exception:  # noqa: BLE001
                self._shader_name = "shaded"
        return self._shader_name

    def _lines(self, segs, colour, width=1.5):
        if len(segs) == 0:
            return
        self._add(self._gl.GLLinePlotItem(pos=np.asarray(segs, np.float32).reshape(-1, 3), mode="lines",
                                          color=colour, width=width, antialias=True))

    def _arrow(self, start, vec, colour, width=3, head=0.08):
        start, vec = np.asarray(start, float), np.asarray(vec, float)
        end = start + vec
        n = np.linalg.norm(vec)
        if n == 0:
            return
        u = vec / n
        t = np.cross(u, [0, 0, 1.0]) if abs(u[2]) < 0.9 else np.cross(u, [1.0, 0, 0])
        t /= np.linalg.norm(t)
        h = head * n
        segs = [[start, end], [end, end - h * u + 0.5 * h * t], [end, end - h * u - 0.5 * h * t]]
        self._lines(segs, colour, width)

    def _text(self, pos, text, colour=(0, 0, 0, 1)):
        try:
            from PySide6 import QtGui
            it = self._gl.GLTextItem(pos=np.asarray(pos, float), text=text,
                                     color=QtGui.QColor.fromRgbF(*colour), font=QtGui.QFont("Sans", 9))
            self._add(it)
        except Exception:  # noqa: BLE001
            pass

    def _extent(self, sc):
        pts = [np.zeros((1, 3))]
        if len(sc.edges):
            pts.append(sc.edges.reshape(-1, 3))
        if len(sc.centres):
            pts.append(sc.centres)
        if sc.slab:
            pts.append(sc.slab["positions"] + sc.slab["extent"] / 2)
        r = float(np.abs(np.vstack(pts)).max())
        spec = sc.particle
        size = max(spec.get("radius", 0) or 0, spec.get("shell", 0) or 0, spec.get("half_length", 0) or 0,
                   max(spec.get("semi", (0,))), max(spec.get("half", (0,))))
        if spec.get("clip"):
            size = min(size, spec["clip"])
        return max(r + size, 1.0)

    # ------------------------------------------------------------------ real space
    def _draw_real(self, sc, frame=True):
        R = self._extent(sc)
        spec = sc.particle
        if sc.surface is not None:
            import pyqtgraph as pg
            Fv, to_lab = LV.bicontinuous_field(sc)
            V, F = pg.isosurface(Fv, 0.0)
            if len(V):
                self._mesh(to_lab(V), F, SURFACE)
        elif sc.slab is not None:
            V, F = LV.slab_mesh(sc.slab)
            self._mesh(V, F, SLAB, translucent=True)
        else:
            core = LV.particle_mesh(spec, "core")
            if core is not None and len(sc.centres):
                V, F = LV.place(core, sc.centres, sc.axes, spec.get("clip"))
                self._mesh(V, F, PARTICLE)
            shell = LV.particle_mesh(spec, "shell")
            if shell is not None and len(sc.centres):
                V, F = LV.place(shell, sc.centres, sc.axes, spec.get("clip"))
                self._mesh(V, F, SHELL, translucent=True)
        self._lines(sc.edges, CELL, 1.2)
        if sc.kind and len(sc.edges):
            corner = sc.edges[0, 0]
            for name, v in zip("abc", sc.cell.T):
                if sc.dim == 2 and name == "c":
                    continue
                self._arrow(corner, v, AXES_COL[name], 3, 0.1)
                self._text(corner + 1.05 * v, name, AXES_COL[name])
        if not frame:                      # view along the beam: only the detector axes
            self._arrow([0, 0, 0], [0, 1.05 * R, 0], (0.2, 0.2, 0.5, 1), 1.5, 0.05)
            self._arrow([0, 0, 0], [0, 0, 1.05 * R], (0.2, 0.2, 0.5, 1), 1.5, 0.05)
            self._text([0, 1.08 * R, 0], "q_h", (0.2, 0.2, 0.5, 1))
            self._text([0, 0, 1.08 * R], "q_v", (0.2, 0.2, 0.5, 1))
            if sc.director is not None and spec.get("type") not in ("sphere", "none"):
                self._arrow(-0.8 * R * sc.director, 1.6 * R * sc.director, DIRECTOR, 2, 0.06)
            return
        # beam and detector plane (x = beam, y = q_h, z = q_v)
        self._arrow([-1.6 * R, 0, 0], [3.0 * R, 0, 0], BEAM, 2.5, 0.05)
        self._text([-1.6 * R, 0, 0.08 * R], "beam", BEAM)
        xd = 1.6 * R
        quad = np.array([[xd, -R, -R], [xd, R, -R], [xd, R, R], [xd, -R, R]])
        self._mesh(quad, np.array([[0, 1, 2], [0, 2, 3]]), (0.35, 0.45, 0.65, 0.18), translucent=True)
        self._lines([[quad[i], quad[(i + 1) % 4]] for i in range(4)], (0.35, 0.45, 0.65, 1), 1.0)
        self._arrow([xd, 0, 0], [0, 0.9 * R, 0], (0.2, 0.2, 0.5, 1), 2, 0.1)
        self._arrow([xd, 0, 0], [0, 0, 0.9 * R], (0.2, 0.2, 0.5, 1), 2, 0.1)
        self._text([xd, 0.95 * R, 0], "q_h", (0.2, 0.2, 0.5, 1))
        self._text([xd, 0, 0.95 * R], "q_v", (0.2, 0.2, 0.5, 1))
        self._text([xd, -R, -R * 1.08], "detector", (0.2, 0.2, 0.5, 1))
        if sc.director is not None and spec.get("type") not in ("sphere", "none"):
            self._arrow(-0.8 * R * sc.director, 1.6 * R * sc.director, DIRECTOR, 3, 0.06)
            self._text(0.85 * R * sc.director, "director", DIRECTOR)

    # ------------------------------------------------------------------ reciprocal space
    def _draw_recip(self, sc):
        gl = self._gl
        if sc.G.shape[1] == 0:
            self._text([0, 0, 0], "no lattice", (0, 0, 0, 1))
            return
        G = sc.G.T
        g = np.linalg.norm(G, axis=1)
        Q = float(max(sc.qmax, g.max()) * 1.05)
        Inorm = sc.intensity / max(float(sc.intensity.max()), 1e-300)
        rad = Q * (0.006 + 0.022 * np.sqrt(np.clip(Inorm, 0, 1)))          # sphere radius ~ sqrt(I)
        unit = LV._sphere_unit(8, 12)
        for sel, colour in ((sc.on_detector, ON), (~sc.on_detector, OFF)):
            idx = np.flatnonzero(sel)
            if idx.size:
                V0, F0 = unit
                V = np.vstack([V0 * rad[j] + G[j] for j in idx])
                F = np.vstack([F0 + i * len(V0) for i in range(idx.size)])
                self._mesh(V, F, colour, translucent=colour[3] < 1)
        V0, F0 = unit
        self._mesh(V0 * Q * 0.015, F0, (0, 0, 0, 1))
        for name, v in zip("abc", sc.recip.T):
            if sc.dim == 2 and name == "c" or sc.dim == 1 and name != "a":
                continue
            self._arrow([0, 0, 0], v, AXES_COL[name], 3, 0.12)
            self._text(1.08 * v, name + "*", AXES_COL[name])
        # detector plane = Ewald sphere near the origin (flat for the preset map), beam
        qm = sc.qmax
        if np.isfinite(sc.k):
            u = np.linspace(-qm, qm, 21)
            Y, Z = np.meshgrid(u, u, indexing="ij")
            X = -(Y ** 2 + Z ** 2) / (2 * sc.k)
            pts = np.stack([X, Y, Z], -1)
            segs = [[pts[i, j], pts[i + 1, j]] for i in range(20) for j in (0, 10, 20)] + \
                   [[pts[i, j], pts[i, j + 1]] for j in range(20) for i in (0, 10, 20)]
            self._lines(segs, (0.35, 0.45, 0.65, 1), 1.0)
            quad = np.array([pts[0, 0], pts[20, 0], pts[20, 20], pts[0, 20]])
        else:
            quad = np.array([[0, -qm, -qm], [0, qm, -qm], [0, qm, qm], [0, -qm, qm]])
            self._lines([[quad[i], quad[(i + 1) % 4]] for i in range(4)], (0.35, 0.45, 0.65, 1), 1.0)
        self._mesh(quad, np.array([[0, 1, 2], [0, 2, 3]]), (0.35, 0.45, 0.65, 0.15), translucent=True)
        self._text(quad[2], "detector (Ewald sphere)", (0.2, 0.2, 0.5, 1))
        self._arrow([-1.3 * Q, 0, 0], [0.8 * Q, 0, 0], BEAM, 2.5, 0.08)
        self._text([-1.3 * Q, 0, 0.06 * Q], "beam", BEAM)
        if self.labels.isChecked():
            idx = np.flatnonzero(sc.on_detector)
            idx = idx[np.argsort(-sc.intensity[idx])][:40]
            for j in idx:
                self._text(G[j] * 1.04, "({} {} {})".format(*(int(v) for v in sc.hkl[:, j])), ON)
        if not sc.oriented:
            self._text([0, 0, -Q], "isotropic: every crystallite orientation — rings at |G|", (0.3, 0.3, 0.3, 1))

    # ------------------------------------------------------------------ detector (matplotlib)
    def _draw_detector(self, sc):
        fig = self.fig
        fig.clear()
        ax = fig.add_subplot(111)
        draw_reflections(ax, sc, labels=self.labels.isChecked(), standalone=True)
        self.canvas.draw_idle()

    # ------------------------------------------------------------------ camera, export
    def reset_camera(self):
        if self.view is None or self.scene is None:
            return
        mode = self.view_mode.currentData()
        if mode == "recip":
            G = self.scene.G
            R = float(max(self.scene.qmax, np.linalg.norm(G, axis=0).max() if G.size else 1.0))
        else:
            R = self._extent(self.scene)
        if mode == "beam":                     # from the detector side: q_h to the right, q_v up
            self.view.opts["fov"] = 1.0
            self.view.setCameraPosition(pos=QtGui_vec(0, 0, 0), distance=R * 2.6 / np.tan(np.radians(0.5)),
                                        elevation=0, azimuth=0)
        else:
            self.view.opts["fov"] = 45.0
            self.view.setCameraPosition(pos=QtGui_vec(0, 0, 0), distance=(4.6 if mode == "recip" else 6.2) * R,
                                        elevation=22, azimuth=-58)

    def on_save_image(self):
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save image", "lattice.png", "PNG (*.png)")
        if not path:
            return
        if self.stack.currentWidget() is self.canvas:
            self.fig.savefig(path, dpi=200)
        elif self.view is not None:
            self.view.grabFramebuffer().save(path)
        self.status.emit(f"saved {path}")


def QtGui_vec(x, y, z):
    from PySide6 import QtGui
    return QtGui.QVector3D(float(x), float(y), float(z))


def hkl_text(hkl):
    return "".join(f"{abs(int(v))}̅" if int(v) < 0 else str(int(v)) for v in hkl)


def draw_reflections(ax, sc, labels=True, standalone=False, colour="r", max_labels=30):
    """Bragg reflections of the scene on the detector (q_h, q_v): spots of an oriented crystal, rings of a powder.
    Used by the panel (``standalone``) and as an overlay on the 2D pattern."""
    import matplotlib.patches as mpatches
    refl = LV.detector_reflections(sc)
    qm = sc.qmax
    if standalone:
        ax.set_xlim(-qm, qm)
        ax.set_ylim(-qm, qm)
        ax.set_aspect("equal")
        ax.set_xlabel("q_h [1/nm]")
        ax.set_ylabel("q_v [1/nm]")
        ax.axhline(0, color="0.85", lw=0.6)
        ax.axvline(0, color="0.85", lw=0.6)
        ax.set_title(sc.title + ("  (rings)" if sc.rings else ""), fontsize="small")
    if not refl:
        if standalone:
            ax.text(0, 0, "no reflection on the detector" if sc.kind else "no lattice", ha="center",
                    va="center", color="0.4")
        return []
    Imax = max(r["I"] for r in refl) or 1.0
    if sc.rings:
        for i, r in enumerate(refl):
            a = 0.25 + 0.75 * np.sqrt(r["I"] / Imax)
            ax.add_patch(mpatches.Circle((0, 0), r["q"], fill=False, color=colour, lw=0.8, alpha=a))
            if labels and i < max_labels:
                ang = np.radians(35 + 9 * (i % 6))
                ax.text(r["q"] * np.cos(ang), r["q"] * np.sin(ang), hkl_text(r["hkl"]), color=colour,
                        fontsize=7, ha="left", va="bottom")
        return refl
    x = np.array([r["qh"] for r in refl])
    y = np.array([r["qv"] for r in refl])
    s = 12 + 80 * np.sqrt(np.array([r["I"] for r in refl]) / Imax)
    if standalone:
        ax.scatter(x, y, s=s, c=colour, alpha=0.8, edgecolors="k", linewidths=0.4)
    else:
        ax.scatter(x, y, s=s, facecolors="none", edgecolors=colour, linewidths=0.8)
    if labels:
        for r in refl[:max_labels]:
            ax.annotate(hkl_text(r["hkl"]), (r["qh"], r["qv"]), xytext=(3, 3), textcoords="offset points",
                        fontsize=7, color=colour)
    return refl
