"""pySASmorph window for anisotropic 2D data (texture): uniaxial samples with the axis in the detector plane.

    python -m pysasfit.morph.aniso_gui [2D data file]

Opened from pySASmorph with *File → Anisotropic 2D data (texture)…*. Thin layer over
:mod:`pysasfit.morph.anisotropic`: model A (affine stretch) or B (uniaxial, Legendre components), axis angle
(estimated or typed), then *Correlations* (fast) or *Reconstruct* (field, cut, voxel scattering, statistics).
Tabs: 2D data and model, Legendre components I_l(q), correlation maps gamma / g over (r_par, r_perp),
structure slices, 3D, results.
"""

from __future__ import annotations

import os
import sys
import traceback

import numpy as np
from PySide6 import QtCore, QtWidgets

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg  # noqa: E402
from matplotlib.colors import LogNorm  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402

from .anisotropic import AnisoParameters, estimate_axis, folded_map, load_2d, reconstruct_aniso  # noqa: E402
from .anisotropic import correlations_aniso  # noqa: E402

__all__ = ["AnisoWindow", "main"]

MATRIX_FORMATS = {"VTK structured points (*.vtk)": ".vtk", "TIFF stack (*.tif)": ".tif", "NumPy (*.npy)": ".npy"}


def _fig():
    f = Figure(figsize=(6, 4), layout="constrained")
    return f, FigureCanvasQTAgg(f)


def _opt_float(text):
    t = text.strip()
    return None if not t else float(t)


class AnisoWindow(QtWidgets.QMainWindow):
    """Anisotropic SASmorph: input form on the left, result tabs on the right."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("pySASmorph – anisotropic 2D data (texture)")
        self.data = None
        self.result = None
        split = QtWidgets.QSplitter()
        self.setCentralWidget(split)
        form_w = QtWidgets.QWidget()
        self._build_form(form_w)
        sc = QtWidgets.QScrollArea()
        sc.setWidget(form_w)
        sc.setWidgetResizable(True)
        sc.setMinimumWidth(330)
        split.addWidget(sc)
        self.tabs = QtWidgets.QTabWidget()
        split.addWidget(self.tabs)
        split.setStretchFactor(1, 1)
        self.fig_2d, self.can_2d = _fig()
        self.fig_l, self.can_l = _fig()
        self.fig_c, self.can_c = _fig()
        self.fig_s, self.can_s = _fig()
        for name, can in (("2D data / model", self.can_2d), ("Components I_l(q)", self.can_l),
                          ("Correlations γ, g", self.can_c), ("Structure", self.can_s)):
            self.tabs.addTab(can, name)
        from .view3d_gl import MorphGLView
        self.view3d = MorphGLView(self)
        self.view3d.status.connect(lambda t: self.statusBar().showMessage(t, 8000))
        self.tabs.addTab(self.view3d, "3D")
        self.results_text = QtWidgets.QPlainTextEdit()
        self.results_text.setReadOnly(True)
        f = self.results_text.font()
        f.setFamily("monospace")
        self.results_text.setFont(f)
        self.tabs.addTab(self.results_text, "Results")
        m = self.menuBar().addMenu("&File")
        m.addAction("&Open 2D data…", self.on_open)
        m.addSeparator()
        m.addAction("Export &components and maps…", self.on_export_curves)
        m.addAction("Export voxel &matrix…", self.on_export_matrix)
        self.resize(1400, 850)

    # ------------------------------------------------------------------ form
    def _build_form(self, w):
        lay = QtWidgets.QVBoxLayout(w)
        fb = QtWidgets.QHBoxLayout()
        b = QtWidgets.QPushButton("Open 2D data…")
        b.setToolTip("NXcanSAS 2D (pySASred: File → Save 2D NXcanSAS), .npz (qx, qy, I [, sigma, mask]) or columns "
                     "q_x q_y I [dI]; the background must be subtracted (or fitted below)")
        b.clicked.connect(self.on_open)
        fb.addWidget(b)
        self.file_label = QtWidgets.QLabel("no data")
        self.file_label.setWordWrap(True)
        fb.addWidget(self.file_label, 1)
        lay.addLayout(fb)
        box = QtWidgets.QGroupBox("Model")
        fl = QtWidgets.QFormLayout(box)
        self.mode = QtWidgets.QComboBox()
        self.mode.addItem("B – uniaxial (Legendre components)", "B")
        self.mode.addItem("A – affine stretch of an isotropic structure", "A")
        self.mode.setToolTip("B: any uniaxial anisotropy, I(q, μ) = Σ I_l(q) P_l(μ).\nA: an isotropic structure "
                             "stretched by λ along the axis (both perpendicular directions unchanged); λ is fitted "
                             "by collapsing the 2D data onto one curve.")
        self.mode.currentIndexChanged.connect(self._update_enabled)
        fl.addRow("model:", self.mode)
        ax = QtWidgets.QHBoxLayout()
        self.axis = QtWidgets.QDoubleSpinBox()
        self.axis.setRange(0.0, 180.0)
        self.axis.setDecimals(1)
        self.axis.setSuffix(" °")
        self.axis.setToolTip("Symmetry axis in the detector plane, from q_x towards q_y. The 3D structure is "
                             "rotationally symmetric about it.")
        ax.addWidget(self.axis)
        be = QtWidgets.QPushButton("estimate")
        be.setToolTip("Axis perpendicular to the direction of maximum intensity (fibres, equatorial streak); "
                      "use '+90°' for layers along the axis (meridional peaks)")
        be.clicked.connect(self.on_estimate)
        ax.addWidget(be)
        b90 = QtWidgets.QPushButton("+90°")
        b90.setToolTip("The other mirror line of the pattern as the symmetry axis")
        b90.clicked.connect(lambda: self.axis.setValue((self.axis.value() + 90.0) % 180.0))
        ax.addWidget(b90)
        fl.addRow("axis:", ax)
        self.fit_axis = QtWidgets.QCheckBox("refine the axis (model A)")
        self.fit_axis.setChecked(True)
        fl.addRow("", self.fit_axis)
        self.stretch = QtWidgets.QLineEdit()
        self.stretch.setPlaceholderText("fit")
        self.stretch.setToolTip("λ: stretch along the axis relative to the perpendicular directions; empty: fitted")
        fl.addRow("stretch λ (A):", self.stretch)
        self.lmax = QtWidgets.QSpinBox()
        self.lmax.setRange(2, 12)
        self.lmax.setSingleStep(2)
        self.lmax.setValue(6)
        self.lmax.setToolTip("Highest even Legendre order of I(q, μ); strong anisotropy needs more, noisy data fewer")
        fl.addRow("l max (B):", self.lmax)
        lay.addWidget(box)
        box2 = QtWidgets.QGroupBox("Structure")
        f2 = QtWidgets.QFormLayout(box2)
        self.porosity = QtWidgets.QDoubleSpinBox()
        self.porosity.setRange(0.01, 0.99)
        self.porosity.setDecimals(3)
        self.porosity.setSingleStep(0.01)
        self.porosity.setValue(0.3)
        f2.addRow("volume fraction φ:", self.porosity)
        self.box_size = QtWidgets.QLineEdit()
        self.box_size.setPlaceholderText("automatic")
        f2.addRow("box size L:", self.box_size)
        self.n = QtWidgets.QSpinBox()
        self.n.setRange(32, 320)
        self.n.setSingleStep(16)
        self.n.setValue(128)
        f2.addRow("voxels n:", self.n)
        self.seed = QtWidgets.QLineEdit()
        self.seed.setPlaceholderText("random")
        f2.addRow("seed:", self.seed)
        lay.addWidget(box2)
        box3 = QtWidgets.QGroupBox("Data")
        f3 = QtWidgets.QFormLayout(box3)
        self.qmin = QtWidgets.QLineEdit()
        self.qmax = QtWidgets.QLineEdit()
        for e in (self.qmin, self.qmax):
            e.setPlaceholderText("all")
        f3.addRow("q min:", self.qmin)
        f3.addRow("q max:", self.qmax)
        self.low_q = QtWidgets.QComboBox()
        self.low_q.addItems(["guinier", "linear", "constant"])
        f3.addRow("low-q extrapolation:", self.low_q)
        prow = QtWidgets.QHBoxLayout()
        self.porod_mode = QtWidgets.QComboBox()
        for text, key in (("automatic", "auto"), ("last N points", "points"), ("q ≥ q_min", "qmin"),
                          ("last 15 %", "fixed15")):
            self.porod_mode.addItem(text, key)
        self.porod_mode.setToolTip("Range of the Porod law (+ B) on I_0 (model B) or the collapsed curve (model A).\n"
                                   "automatic: the longest high-q range the law still describes; or set by hand")
        self.porod_points = QtWidgets.QSpinBox()
        self.porod_points.setRange(3, 10000)
        self.porod_points.setValue(15)
        self.porod_qmin = QtWidgets.QLineEdit()
        self.porod_qmin.setPlaceholderText("q_min")
        prow.addWidget(self.porod_mode, 1)
        prow.addWidget(self.porod_points)
        prow.addWidget(self.porod_qmin)
        self.porod_mode.currentIndexChanged.connect(self._update_porod_widgets)
        f3.addRow("Porod range:", prow)
        self.fit_b = QtWidgets.QCheckBox("fit flat (incoherent) background B")
        self.fit_b.setChecked(True)
        self.fit_b.setToolTip("Fitted with the Porod law; untick to use the fixed B below (empty = 0)")
        f3.addRow("", self.fit_b)
        self.background = QtWidgets.QLineEdit()
        self.background.setPlaceholderText("0")
        f3.addRow("fixed B:", self.background)
        lay.addWidget(box3)
        bb = QtWidgets.QHBoxLayout()
        self.b_corr = QtWidgets.QPushButton("Correlations")
        self.b_corr.setToolTip("Components, γ(r), g(r) and the level-cut model (fast)")
        self.b_corr.clicked.connect(lambda: self.run(False))
        self.b_rec = QtWidgets.QPushButton("Reconstruct")
        self.b_rec.setToolTip("Also the anisotropic Gaussian field, the cut, the voxel scattering and statistics")
        self.b_rec.clicked.connect(lambda: self.run(True))
        bb.addWidget(self.b_corr)
        bb.addWidget(self.b_rec)
        lay.addLayout(bb)
        self.progress = QtWidgets.QLabel("")
        lay.addWidget(self.progress)
        lay.addStretch(1)
        self._update_porod_widgets()
        self._update_enabled()

    def _update_porod_widgets(self, _i=None):
        m = self.porod_mode.currentData()
        self.porod_points.setVisible(m == "points")
        self.porod_qmin.setVisible(m == "qmin")

    def _update_enabled(self):
        a = self.mode.currentData() == "A"
        self.fit_axis.setEnabled(a)
        self.stretch.setEnabled(a)
        self.lmax.setEnabled(True)
        has = self.data is not None
        self.b_corr.setEnabled(has)
        self.b_rec.setEnabled(has)

    def params(self) -> AnisoParameters:
        pm = self.porod_mode.currentData()
        seed = self.seed.text().strip()
        return AnisoParameters(mode=self.mode.currentData(), axis_deg=self.axis.value(),
                               fit_axis=self.fit_axis.isChecked(), stretch=_opt_float(self.stretch.text()),
                               lmax=self.lmax.value(), porosity=self.porosity.value(),
                               box_size=_opt_float(self.box_size.text()), n=self.n.value(),
                               qmin=_opt_float(self.qmin.text()), qmax=_opt_float(self.qmax.text()),
                               low_q=self.low_q.currentText(),
                               porod_points=self.porod_points.value() if pm == "points" else None,
                               porod_qmin=_opt_float(self.porod_qmin.text()) if pm == "qmin" else None,
                               porod_auto=pm == "auto",
                               fit_background=self.fit_b.isChecked(),
                               background=_opt_float(self.background.text()), seed=int(seed) if seed else None)

    # ------------------------------------------------------------------ actions
    def on_open(self):
        p, _ = QtWidgets.QFileDialog.getOpenFileName(self, "2D data", "",
                                                     "2D data (*.h5 *.hdf5 *.nxs *.npz *.dat *.txt);;All files (*)")
        if p:
            self.load_file(p)

    def load_file(self, path):
        try:
            self.data = load_2d(path)
        except Exception as e:  # noqa: BLE001
            QtWidgets.QMessageBox.critical(self, "Open 2D data", f"{type(e).__name__}: {e}")
            return
        d = self.data
        unit = "1/nm" if d.q_unit == "nm" else "1/Å"
        self.file_label.setText(f"{os.path.basename(path)}: {d.I.size} pixels, q = {d.q.min():.4g} … {d.q.max():.4g} {unit}")
        self.result = None
        self.on_estimate()
        self._update_enabled()
        self._plot_2d()

    def on_estimate(self):
        if self.data is None:
            return
        phi_max, an = estimate_axis(self.data, _opt_float(self.qmin.text()), _opt_float(self.qmax.text()))
        self.axis.setValue((phi_max + 90.0) % 180.0)
        self.statusBar().showMessage(f"intensity maximum at {phi_max:.1f}°, anisotropy {an:.3f}: axis set to "
                                     f"{(phi_max + 90) % 180:.1f}° (fibre convention; '+90°' for the other)", 12000)

    def _set_progress(self, x, text):
        self.progress.setText(f"{text} ({100 * x:.0f} %)")
        QtWidgets.QApplication.processEvents()

    def run(self, full):
        if self.data is None:
            return None
        p = self.params()
        QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.CursorShape.WaitCursor)
        try:
            res = (reconstruct_aniso(self.data, p, progress=self._set_progress) if full
                   else correlations_aniso(self.data, p))
        except Exception as e:  # noqa: BLE001
            QtWidgets.QApplication.restoreOverrideCursor()
            QtWidgets.QMessageBox.critical(self, "pySASmorph", f"{type(e).__name__}: {e}\n\n{traceback.format_exc(limit=3)}")
            return None
        QtWidgets.QApplication.restoreOverrideCursor()
        self.result = res
        self.axis.blockSignals(True)
        self.axis.setValue(res.axis_deg)
        self.axis.blockSignals(False)
        if res.mode == "A" and not self.stretch.text().strip():
            self.stretch.setPlaceholderText(f"fit (last: {res.stretch:.4g})")
        if not self.box_size.text().strip():
            self.box_size.setPlaceholderText(f"automatic (last: {res.params.box_size:.4g})")
        self.progress.setText("done" if full else "correlations done")
        self.refresh()
        if full:
            self.view3d.set_result(res, "nm" if self.data.q_unit == "nm" else "Å")
        return res

    def on_export_curves(self):
        res = self.result
        if res is None:
            return
        base, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Export components and maps (base name)",
                                                        (self.data.name if self.data else "aniso"))
        if not base:
            return
        base = os.path.splitext(base)[0]
        hdr = "q " + " ".join(f"I_{int(l)}" for l in res.ls)
        np.savetxt(base + "_components.txt", np.column_stack([res.q_l, res.I_l]), header=hdr)
        np.savetxt(base + "_model_components.txt", np.column_stack([res.q_l, res.I_l_model]),
                   header="q " + " ".join(f"I_{int(l)}" for l in res.model_ls))
        np.savez(base + "_correlations.npz", r=res.r, mu=res.mu, gamma=res.gamma_rm, g=res.g_rm, C_l=res.C_l,
                 ls=res.ls, axis_deg=res.axis_deg, stretch=res.stretch)
        with open(base + "_log.txt", "w", encoding="utf-8") as fh:
            fh.write(res.report() + "\n")
        self.statusBar().showMessage(f"written: {base}_components.txt, _model_components.txt, _correlations.npz, _log.txt",
                                     10000)

    def on_export_matrix(self):
        res = self.result
        if res is None or res.matrix is None:
            return
        path, flt = QtWidgets.QFileDialog.getSaveFileName(self, "Export voxel matrix (1 = phase 2, axis = x)",
                                                          (self.data.name if self.data else "aniso") + ".vtk",
                                                          ";;".join(MATRIX_FORMATS))
        if not path:
            return
        if not os.path.splitext(path)[1]:
            path += MATRIX_FORMATS.get(flt, ".vtk")
        from .export import write_npy, write_tiff_stack, write_vtk
        ext = os.path.splitext(path)[1].lower()
        try:
            if ext == ".vtk":
                write_vtk(res.matrix, path, res.voxel_size)
            elif ext in (".tif", ".tiff"):
                write_tiff_stack(res.matrix, path, res.voxel_size)
            else:
                write_npy(res.matrix, path)
        except Exception as e:  # noqa: BLE001
            QtWidgets.QMessageBox.critical(self, "Export", str(e))
            return
        self.statusBar().showMessage(f"written: {path}", 10000)

    # ------------------------------------------------------------------ plots
    def refresh(self):
        self._plot_2d()
        self._plot_components()
        self._plot_corr()
        self._plot_structure()
        self.results_text.setPlainText(self.result.report() if self.result else "")

    def _plot_2d(self):
        f = self.fig_2d
        f.clear()
        d = self.data
        if d is None:
            self.can_2d.draw_idle()
            return
        res = self.result
        axs = f.subplots(1, 3 if res is not None else 1, squeeze=False)[0]
        pos = d.I > 0
        vmin, vmax = np.percentile(d.I[pos], [2, 99.8]) if pos.any() else (1e-3, 1)
        norm = LogNorm(vmin=max(vmin, vmax * 1e-6), vmax=vmax)
        ax = axs[0]
        ax.scatter(d.qx[pos], d.qy[pos], c=d.I[pos], s=2, marker="s", norm=norm, cmap="viridis", linewidths=0)
        t = np.radians(self.axis.value() if res is None else res.axis_deg)
        qm = float(d.q.max())
        ax.plot([-qm * np.cos(t), qm * np.cos(t)], [-qm * np.sin(t), qm * np.sin(t)], "--", color="magenta", lw=1)
        ax.set_aspect("equal")
        ax.set_xlabel("q_x")
        ax.set_ylabel("q_y")
        ax.set_title("data, symmetry axis dashed", fontsize="small")
        if res is not None:
            e1, e2, Md = folded_map(res, "data")
            _, _, Mm = folded_map(res, "model")
            for a2, M, title in ((axs[1], Md, "data (folded)"), (axs[2], Mm, "level-cut model")):
                im = a2.pcolormesh(e1, e2, np.where(M > 0, M, np.nan).T, norm=norm, cmap="viridis", shading="flat")
                a2.set_aspect("equal")
                a2.set_xlabel("q_∥ (along the axis)")
                a2.set_ylabel("q_⊥")
                a2.set_title(title, fontsize="small")
            f.colorbar(im, ax=list(axs), shrink=0.8, label="I")
        self.can_2d.draw_idle()

    def _plot_components(self):
        f = self.fig_l
        f.clear()
        res = self.result
        if res is None:
            self.can_l.draw_idle()
            return
        a0, a1 = f.subplots(2, 1, sharex=True)
        a0.loglog(res.q_l, np.abs(res.I_l[:, 0]), "o", ms=3, label="data I_0")
        a0.loglog(res.q_l, np.abs(res.I_l_model[:, 0]), "-", label="level-cut model")
        if res.q_voxel is not None:
            a0.loglog(res.q_voxel, np.abs(res.I_l_voxel[:, 0]), "x", ms=4, label="voxels")
        a0.set_ylabel("I_0(q)")
        a0.legend(fontsize="small")
        I0 = np.where(np.abs(res.I_l[:, 0]) > 0, res.I_l[:, 0], np.nan)
        I0m = np.where(np.abs(res.I_l_model[:, 0]) > 0, res.I_l_model[:, 0], np.nan)
        for i, l in enumerate(res.ls[1:], start=1):
            line, = a1.semilogx(res.q_l, res.I_l[:, i] / I0, "o", ms=3, label=f"l = {int(l)}")
            j = list(res.model_ls).index(l) if l in res.model_ls else None
            if j is not None:
                a1.semilogx(res.q_l, res.I_l_model[:, j] / I0m, "-", color=line.get_color())
            if res.q_voxel is not None and i < res.I_l_voxel.shape[1]:
                v0 = np.where(np.abs(res.I_l_voxel[:, 0]) > 0, res.I_l_voxel[:, 0], np.nan)
                a1.semilogx(res.q_voxel, res.I_l_voxel[:, i] / v0, "x", ms=4, color=line.get_color())
        a1.axhline(0, color="0.6", lw=0.8)
        a1.set_ylabel("I_l / I_0")
        a1.set_xlabel("q")
        a1.legend(fontsize="small", ncol=2)
        a1.set_title("points: data, lines: level-cut model, ×: voxel structure", fontsize="small")
        self.can_l.draw_idle()

    def _plot_corr(self):
        f = self.fig_c
        f.clear()
        res = self.result
        if res is None:
            self.can_c.draw_idle()
            return
        # range: where |gamma| has fallen below 0.02 in every direction, with margin, at most half the box
        live = np.flatnonzero(np.max(np.abs(res.gamma_rm), axis=1) > 0.02)
        R = float(min(res.r[-1], 0.5 * res.params.box_size,
                      1.5 * res.r[live[-1]] if live.size else res.r[-1]))
        x = np.linspace(0, R, 161)
        P, S = np.meshgrid(x, x, indexing="ij")
        G = res.gamma(P, S)
        axs = f.subplots(1, 2)
        im = axs[0].pcolormesh(x, x, G.T, cmap="RdBu_r", vmin=-1, vmax=1, shading="auto")
        axs[0].contour(x, x, G.T, levels=[0.1, 0.5], colors="k", linewidths=0.7)
        axs[0].set_title("γ(r_∥, r_⊥); contours 0.1 and 0.5", fontsize="small")
        gg = res.g(P, S)
        axs[1].pcolormesh(x, x, gg.T, cmap="RdBu_r", vmin=-1, vmax=1, shading="auto")
        axs[1].set_title("g(r_∥, r_⊥) of the Gaussian field", fontsize="small")
        for a in axs:
            a.set_aspect("equal")
            a.set_xlabel("r_∥ (along the axis)")
            a.set_ylabel("r_⊥")
        f.colorbar(im, ax=list(axs), shrink=0.8)
        self.can_c.draw_idle()

    def _plot_structure(self):
        f = self.fig_s
        f.clear()
        res = self.result
        if res is None or res.matrix is None:
            ax = f.add_subplot(111)
            ax.text(0.5, 0.5, "Reconstruct to see the structure", ha="center", va="center", transform=ax.transAxes)
            ax.set_axis_off()
            self.can_s.draw_idle()
            return
        m = res.matrix
        n = m.shape[0]
        L = res.params.box_size
        axs = f.subplots(1, 2)
        axs[0].imshow(m[:, :, n // 2].T, origin="lower", cmap="gray_r", extent=(0, L, 0, L), interpolation="nearest")
        axs[0].set_title("plane containing the axis (x, y)", fontsize="small")
        axs[0].set_xlabel("x (axis)")
        axs[0].set_ylabel("y")
        axs[1].imshow(m[n // 2, :, :].T, origin="lower", cmap="gray_r", extent=(0, L, 0, L), interpolation="nearest")
        axs[1].set_title("plane perpendicular to the axis (y, z)", fontsize="small")
        axs[1].set_xlabel("y")
        axs[1].set_ylabel("z (beam)")
        self.can_s.draw_idle()


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv[:1])
    w = AnisoWindow()
    w.show()
    if argv:
        w.load_file(argv[0])
    return app.exec()


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
