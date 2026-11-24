"""Panel "2D map" of the model explorer: I(q_x, q_y) of the current model (see :mod:`pysasfit.studio.map2d`)."""

from __future__ import annotations

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib.colors import LogNorm
from matplotlib.figure import Figure
from PySide6 import QtCore, QtWidgets

from . import map2d

AUTO_LIMIT_S = 3.0  # a map slower than this switches automatic updates off


class MapPanel(QtWidgets.QWidget):
    """2D detector map of the explorer's model. ``explorer`` provides map_model_copy(), queue_map(), wait_calc(),
    has_model(), qmax, units."""

    def __init__(self, explorer, parent=None):
        super().__init__(parent)
        self.ex = explorer
        self.stale = True
        v = QtWidgets.QVBoxLayout(self)
        v.setContentsMargins(2, 2, 2, 2)
        self.fig = Figure(figsize=(5, 4.4), layout="constrained")
        self.ax = self.fig.add_subplot(111)
        self.canvas = FigureCanvasQTAgg(self.fig)
        self.canvas.setMinimumSize(320, 280)
        self._im = None
        self._cbar = None
        v.addWidget(NavigationToolbar2QT(self.canvas, self))
        v.addWidget(self.canvas, 1)

        row = QtWidgets.QHBoxLayout()
        row.addWidget(QtWidgets.QLabel("q max"))
        self.qmax = QtWidgets.QLineEdit("")
        self.qmax.setPlaceholderText("Q max of I(q)")
        self.qmax.setToolTip(
            "half width of the map: q_x, q_y in [-q max, q max] (empty: Q max of the I(q) panel)"
        )
        self.qmax.setMaximumWidth(80)
        self.qmax.editingFinished.connect(self.request)
        row.addWidget(self.qmax)
        row.addWidget(QtWidgets.QLabel("pixels"))
        self.npix = QtWidgets.QSpinBox()
        self.npix.setRange(16, 1024)
        self.npix.setValue(128)
        self.npix.setToolTip("pixels per axis")
        self.npix.valueChanged.connect(self.request)
        row.addWidget(self.npix)
        row.addWidget(QtWidgets.QLabel("ψ steps"))
        self.npsi = QtWidgets.QSpinBox()
        self.npsi.setRange(4, 720)
        self.npsi.setValue(36)
        self.npsi.setToolTip("number of directions ψ in [0°, 180°) at which the model is evaluated (one model "
                             "evaluation each); the map interpolates between them")  # fmt: skip
        self.npsi.valueChanged.connect(self.request)
        row.addWidget(self.npsi)
        self.log = QtWidgets.QCheckBox("log I")
        self.log.setChecked(True)
        self.log.toggled.connect(self._restyle)
        row.addWidget(self.log)
        self.cmap = QtWidgets.QComboBox()
        self.cmap.addItems(["viridis", "magma", "jet", "gray", "cividis"])
        self.cmap.currentTextChanged.connect(self._restyle)
        row.addWidget(self.cmap)
        self.auto = QtWidgets.QCheckBox("auto")
        self.auto.setChecked(True)
        self.auto.setToolTip(f"update with every change of the model (switched off when a map takes more than "
                             f"{AUTO_LIMIT_S:g} s)")  # fmt: skip
        row.addWidget(self.auto)
        b = QtWidgets.QPushButton("Update")
        b.clicked.connect(lambda: self.request(force=True))
        row.addWidget(b)
        row.addStretch(1)
        v.addLayout(row)
        self.note = QtWidgets.QLabel("ψ is the direction of q on the detector, measured from q_x (horizontal) "
                                     "towards q_y (counter-clockwise; view towards the sample).")  # fmt: skip
        self.note.setWordWrap(True)
        v.addWidget(self.note)
        self._data = None

    # ------------------------------------------------------------------
    def request(self, *_, force=False):
        """The model changed: update in the background if visible and automatic (``force``: Update button), else
        when shown. The explorer runs the map after the curve, one calculation at a time."""
        self.stale = True
        if force or (self.isVisible() and self.auto.isChecked()):
            if hasattr(self.ex, "queue_map"):
                self.ex.queue_map()
            else:
                self.update_map()

    def showEvent(self, ev):  # Qt
        super().showEvent(ev)
        if self.stale and self.auto.isChecked():
            QtCore.QTimer.singleShot(0, self.request)

    def _qmax(self):
        txt = self.qmax.text().strip() or self.ex.qmax.text()
        return float(txt)

    def make_job(self):
        """``{"run": ..., "finish": ...}`` for a background calculation, or None (nothing to do; note set).
        ``run`` uses a copy of the model and no Qt object; ``finish(result)`` draws (window's thread)."""
        ex = self.ex
        if not ex.has_model():
            return None
        if ex._shown_domain != "q":
            self.note.setText(
                "The 2D map shows I(q); switch the transform off (the plot shows a real-space function)."
            )
            return None
        qmax = self._qmax()
        n, n_psi = self.npix.value(), self.npsi.value()
        model, intensity = ex.map_model_copy()
        aniso = bool(map2d.anisotropic_parameters(model))

        def run():
            return map2d.timed_map(model, qmax, n=n, n_psi=n_psi, intensity=intensity)

        def finish(res):
            if isinstance(res, Exception):
                self.note.setText(f"{type(res).__name__}: {res}")
                return
            (qx, qy, img), dt = res
            self._show(qx, qy, img, dt, aniso)

        return {"run": run, "finish": finish}

    def update_map(self):
        """Calculate and draw now, in the window's thread."""
        ex = self.ex
        if hasattr(ex, "wait_calc"):
            ex.wait_calc()
        try:
            job = self.make_job()
            if job is None:
                return
            QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.CursorShape.WaitCursor)
            try:
                res = job["run"]()
            finally:
                QtWidgets.QApplication.restoreOverrideCursor()
        except Exception as exc:  # noqa: BLE001 -- shown in the panel
            self.note.setText(f"{type(exc).__name__}: {exc}")
            return
        job["finish"](res)

    def _show(self, qx, qy, img, dt, aniso):
        ex = self.ex
        self.stale = False
        self._data = (qx, qy, img)
        self._draw()
        unit = ex.units.currentText() if hasattr(ex, "units") else ""
        msg = (f"{'anisotropic' if aniso else 'isotropic'} model, {dt:.2f} s. ψ is the direction of q on the "
               f"detector, measured from q_x (horizontal) towards q_y (counter-clockwise, view towards the sample); "
               f"q in {unit}.")  # fmt: skip
        if not aniso:
            msg += " No function of the model depends on ψ."
        if dt > AUTO_LIMIT_S and self.auto.isChecked():
            self.auto.setChecked(False)
            msg += (
                f" Automatic update switched off (> {AUTO_LIMIT_S:g} s): press Update."
            )
        self.note.setText(msg)

    def _restyle(self, *_):
        if self._data is not None:
            self._draw()

    def _draw(self):
        qx, qy, img = self._data
        data = np.asarray(img, float)
        norm = None
        if self.log.isChecked():
            pos = data[np.isfinite(data) & (data > 0)]
            if pos.size:
                norm = LogNorm(vmin=max(pos.min(), pos.max() * 1e-8), vmax=pos.max())
                data = np.where(data > 0, data, np.nan)
        self.fig.clear()
        self.ax = self.fig.add_subplot(111)
        im = self.ax.imshow(data, origin="lower", extent=(qx[0], qx[-1], qy[0], qy[-1]), cmap=self.cmap.currentText(),
                            norm=norm, interpolation="nearest", aspect="equal")  # fmt: skip
        unit = self.ex.units.currentText() if hasattr(self.ex, "units") else ""
        self.ax.set_xlabel(f"q_x [{unit}]" if unit else "q_x")
        self.ax.set_ylabel(f"q_y [{unit}]" if unit else "q_y")
        self.fig.colorbar(im, ax=self.ax, label="I(q)", shrink=0.9, fraction=0.06)
        self.canvas.draw_idle()
