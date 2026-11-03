"""HDF5 viewer of the pyTOOLbox (local, in the spirit of myHDF5 / H5Web).

Left: the files that are open, as a tree (groups, datasets, soft and external links,
loaded when a group is expanded) and a search over all paths. Right::

    Display   scalars and strings as text; 1D data as a line plot; 2D and higher as a heatmap
              of two chosen dimensions with the others fixed by index (spin boxes); any
              dataset as a matrix (table). NeXus / NXcanSAS groups with @signal (also via
              @default) are plotted with their axes and error bars.
    Inspect   path, type, shape, chunks, filters and all attributes.

Files are opened only while they are read, so the metadata lister or the instrument
software can write to them while they are shown.
"""
from __future__ import annotations

import os

import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.colors import LogNorm
from matplotlib.figure import Figure

from ..core import settings as prefs
from ..core.docking import plot_panel
from . import hdf5

__all__ = ["HDF5Viewer", "image_colorbar"]


def image_colorbar(fig, ax, im, label=""):
    """Image flush left in its panel (a fixed pixel aspect leaves free space on the right)
    with the colour bar attached directly to its right edge."""
    ax.set_anchor("W")
    cax = ax.inset_axes([1.03, 0.0, 0.045, 1.0])
    return fig.colorbar(im, cax=cax, label=label)

ROLE = QtCore.Qt.ItemDataRole.UserRole
_ICON = {"file": QtWidgets.QStyle.StandardPixmap.SP_DriveHDIcon,
         "group": QtWidgets.QStyle.StandardPixmap.SP_DirIcon,
         "dataset": QtWidgets.QStyle.StandardPixmap.SP_FileIcon,
         "softlink": QtWidgets.QStyle.StandardPixmap.SP_FileLinkIcon,
         "externallink": QtWidgets.QStyle.StandardPixmap.SP_FileLinkIcon,
         "broken": QtWidgets.QStyle.StandardPixmap.SP_MessageBoxWarning}
HDF_FILTER = "HDF5 / NeXus (*.hdf *.h5 *.hdf5 *.nxs *.nx *.he5 *.cxi);;all files (*)"


class ArrayModel(QtCore.QAbstractTableModel):
    """Table view of a 0D, 1D or 2D numpy array (formatted on demand)."""

    def __init__(self, a=None, row_step=1, col_step=1):
        super().__init__()
        self.set_array(a, row_step, col_step)

    def set_array(self, a, row_step=1, col_step=1):
        self.beginResetModel()
        a = np.asarray(a) if a is not None else np.zeros((0, 0))
        if a.ndim == 0:
            a = a.reshape(1, 1)
        elif a.ndim == 1:
            a = a.reshape(-1, 1)
        self.a, self.rs, self.cs = a, row_step, col_step
        self.endResetModel()

    def rowCount(self, parent=QtCore.QModelIndex()):
        return 0 if parent.isValid() else self.a.shape[0]

    def columnCount(self, parent=QtCore.QModelIndex()):
        return 0 if parent.isValid() else self.a.shape[1]

    def data(self, index, role=QtCore.Qt.ItemDataRole.DisplayRole):
        if role != QtCore.Qt.ItemDataRole.DisplayRole or not index.isValid():
            return None
        v = self.a[index.row(), index.column()]
        if isinstance(v, (bytes, np.bytes_)):
            return v.decode("utf-8", errors="replace")
        if isinstance(v, (float, np.floating)):
            return f"{v:.6g}"
        if isinstance(v, np.void):
            return str(tuple(v.tolist()))
        return str(v)

    def headerData(self, section, orientation, role=QtCore.Qt.ItemDataRole.DisplayRole):
        if role != QtCore.Qt.ItemDataRole.DisplayRole:
            return None
        step = self.rs if orientation == QtCore.Qt.Orientation.Vertical else self.cs
        return str(section * step)


class HDF5Viewer(QtWidgets.QWidget):
    status = QtCore.Signal(str)
    to_2d = QtCore.Signal(object)       # image2d.Image2D of the shown detector image (2D data page)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.files = []            # open file paths (in tree order)
        self.current = None        # (file, h5path) shown
        self._shape = ()
        self._dtype = None
        self._nx = None
        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(2, 2, 2, 2)
        split = QtWidgets.QSplitter()
        lay.addWidget(split, 1)
        split.addWidget(self._build_left())
        split.addWidget(self._build_right())
        split.setStretchFactor(0, 0)
        split.setStretchFactor(1, 1)
        split.setSizes([360, 900])
        self.setAcceptDrops(True)

    # ------------------------------------------------------------------ left: files, tree, search
    def _build_left(self):
        w = QtWidgets.QWidget()
        v = QtWidgets.QVBoxLayout(w)
        v.setContentsMargins(0, 0, 0, 0)
        row = QtWidgets.QHBoxLayout()
        b = QtWidgets.QPushButton("Open files…")
        b.clicked.connect(self.on_open)
        c = QtWidgets.QPushButton("Close file")
        c.setToolTip("remove the file of the selected item from the tree")
        c.clicked.connect(self.close_current_file)
        row.addWidget(b)
        row.addWidget(c)
        v.addLayout(row)
        self.tree = QtWidgets.QTreeWidget()
        self.tree.setHeaderLabels(["name", "info"])
        self.tree.setColumnWidth(0, 200)
        self.tree.itemExpanded.connect(self._expand)
        self.tree.currentItemChanged.connect(lambda cur, _prev: self._select(cur))
        v.addWidget(self.tree, 3)
        self.search = QtWidgets.QLineEdit()
        self.search.setPlaceholderText("search paths in the open files (Enter)")
        self.search.setClearButtonEnabled(True)
        self.search.returnPressed.connect(self._search)
        v.addWidget(self.search)
        self.results = QtWidgets.QListWidget()
        self.results.itemActivated.connect(lambda it: self.select_path(*it.data(ROLE)))
        self.results.itemClicked.connect(lambda it: self.select_path(*it.data(ROLE)))
        v.addWidget(self.results, 1)
        return w

    def dragEnterEvent(self, e):
        if e.mimeData().hasUrls():
            e.acceptProposedAction()

    def dropEvent(self, e):
        for u in e.mimeData().urls():
            if u.isLocalFile():
                self.open_file(u.toLocalFile())

    def on_open(self):
        start = os.path.dirname(self.files[-1]) if self.files else os.getcwd()
        paths, _ = QtWidgets.QFileDialog.getOpenFileNames(self, "Open HDF5 files", start, HDF_FILTER)
        for p in paths:
            self.open_file(p)

    def open_file(self, path, select=True):
        """Add a file to the tree (or select it if it is already open)."""
        import h5py
        path = os.path.abspath(str(path))
        for i in range(self.tree.topLevelItemCount()):
            it = self.tree.topLevelItem(i)
            if it.data(0, ROLE)[0] == path:
                if select:
                    self.tree.setCurrentItem(it)
                return it
        try:
            if not h5py.is_hdf5(path):
                raise OSError("not an HDF5 file")
        except OSError as e:
            QtWidgets.QMessageBox.critical(self, "Open", f"{path}:\n{e}")
            return None
        it = QtWidgets.QTreeWidgetItem([os.path.basename(path), os.path.dirname(path)])
        it.setData(0, ROLE, (path, "/", "file"))
        it.setIcon(0, self.style().standardIcon(_ICON["file"]))
        it.setToolTip(0, path)
        it.addChild(QtWidgets.QTreeWidgetItem(["…"]))
        self.tree.addTopLevelItem(it)
        self.files.append(path)
        if select:
            self.tree.setCurrentItem(it)
            it.setExpanded(True)
        return it

    def close_current_file(self):
        it = self.tree.currentItem()
        while it is not None and it.parent() is not None:
            it = it.parent()
        if it is None:
            return
        path = it.data(0, ROLE)[0]
        self.tree.takeTopLevelItem(self.tree.indexOfTopLevelItem(it))
        self.files = [f for f in self.files if f != path]
        if self.current and self.current[0] == path:
            self.current = None
            self._clear_display()

    def _expand(self, item):
        if item.childCount() != 1 or item.child(0).data(0, ROLE) is not None:
            return                                          # already loaded
        item.takeChild(0)
        path, h5path, kind = item.data(0, ROLE)
        try:
            kids = hdf5.children(path, h5path)
        except Exception as e:  # noqa: BLE001
            self.status.emit(f"{os.path.basename(path)}{h5path}: {e}")
            return
        st = self.style()
        for c in kids:
            ch = QtWidgets.QTreeWidgetItem([c.name, c.info])
            ch.setData(0, ROLE, (path, c.path, c.kind))
            ch.setIcon(0, st.standardIcon(_ICON.get(c.kind, _ICON["dataset"])))
            ch.setToolTip(0, c.path)
            if c.is_group:
                ch.addChild(QtWidgets.QTreeWidgetItem(["…"]))
            item.addChild(ch)
        self.tree.resizeColumnToContents(1)

    def select_path(self, path, h5path):
        """Expand the tree down to ``h5path`` of file ``path`` and show it."""
        top = self.open_file(path, select=False)
        if top is None:
            return
        item = top
        parts = [p for p in h5path.split("/") if p]
        for p in parts:
            item.setExpanded(True)
            self._expand(item)
            nxt = None
            for i in range(item.childCount()):
                if item.child(i).text(0) == p:
                    nxt = item.child(i)
                    break
            if nxt is None:
                break
            item = nxt
        self.tree.setCurrentItem(item)
        self.tree.scrollToItem(item)

    def _search(self):
        self.results.clear()
        text = self.search.text().strip()
        if not text:
            return
        n = 0
        for f in self.files:
            try:
                hits = hdf5.find(f, text)
            except Exception as e:  # noqa: BLE001
                self.status.emit(f"{f}: {e}")
                continue
            for h in hits:
                it = QtWidgets.QListWidgetItem(f"{os.path.basename(f)}  {h}")
                it.setData(ROLE, (f, h))
                self.results.addItem(it)
                n += 1
        self.status.emit(f"{n} matches for '{text}'")

    # ------------------------------------------------------------------ right: display, inspect
    def _build_right(self):
        self.right = QtWidgets.QTabWidget()
        # display
        d = QtWidgets.QWidget()
        dv = QtWidgets.QVBoxLayout(d)
        dv.setContentsMargins(2, 2, 2, 2)
        self.title = QtWidgets.QLabel("")
        self.title.setTextInteractionFlags(QtCore.Qt.TextInteractionFlag.TextSelectableByMouse)
        dv.addWidget(self.title)
        bar = QtWidgets.QHBoxLayout()
        self.mode = QtWidgets.QComboBox()
        for m in ("Auto", "Line", "Heatmap", "Matrix", "Text"):
            self.mode.addItem(m)
        self.mode.currentIndexChanged.connect(lambda _i: self._rebuild_dims())
        bar.addWidget(QtWidgets.QLabel("view"))
        bar.addWidget(self.mode)
        self.logx = QtWidgets.QCheckBox("log x")
        self.logy = QtWidgets.QCheckBox("log y")
        self.logc = QtWidgets.QCheckBox("log colour")
        self.logc.setChecked(True)
        self.errbars = QtWidgets.QCheckBox("error bars")
        self.errbars.setChecked(True)
        self.keep_aspect = QtWidgets.QCheckBox("pixel aspect")
        self.keep_aspect.setChecked(True)
        self.keep_aspect.setToolTip("heatmaps of detector images keep the aspect ratio of the pixels "
                                    "(x_pixel_size, y_pixel_size of the NXdetector); off: fill the panel")
        self.cmap = QtWidgets.QComboBox()
        self.cmap.addItems(["viridis", "inferno", "magma", "cividis", "gray", "jet", "turbo", "coolwarm"])
        for w in (self.logx, self.logy, self.logc, self.errbars, self.keep_aspect):
            w.toggled.connect(lambda _on: self._draw())
            bar.addWidget(w)
        self.cmap.currentIndexChanged.connect(lambda _i: self._draw())
        bar.addWidget(self.cmap)
        self.to2d_button = QtWidgets.QPushButton("→ 2D data")
        self.to2d_button.setToolTip("open the shown detector image on the 2D data page: beam centre (click, "
                                    "type or centre of gravity), radial / sector / azimuthal averages, polar view")
        self.to2d_button.clicked.connect(self.on_to_2d)
        bar.addWidget(self.to2d_button)
        bar.addStretch(1)
        dv.addLayout(bar)
        self.dims_box = QtWidgets.QWidget()
        self.dims_lay = QtWidgets.QHBoxLayout(self.dims_box)
        self.dims_lay.setContentsMargins(0, 0, 0, 0)
        dv.addWidget(self.dims_box)
        self.stack = QtWidgets.QStackedWidget()
        self.canvas = FigureCanvasQTAgg(Figure(figsize=(6, 4.5), layout="constrained"))
        self.stack.addWidget(plot_panel(self.canvas, self))           # 0
        self.matrix = QtWidgets.QTableView()
        self.matrix_model = ArrayModel()
        self.matrix.setModel(self.matrix_model)
        self.stack.addWidget(self.matrix)                              # 1
        self.text = QtWidgets.QPlainTextEdit()
        self.text.setReadOnly(True)
        self.text.setFont(QtGui.QFontDatabase.systemFont(QtGui.QFontDatabase.SystemFont.FixedFont))
        self.stack.addWidget(self.text)                                # 2
        dv.addWidget(self.stack, 1)
        self.right.addTab(d, "Display")
        # inspect
        ins = QtWidgets.QWidget()
        iv = QtWidgets.QVBoxLayout(ins)
        self.meta_table = QtWidgets.QTableWidget(0, 2)
        self.meta_table.setHorizontalHeaderLabels(["", "value"])
        self.attr_table = QtWidgets.QTableWidget(0, 3)
        self.attr_table.setHorizontalHeaderLabels(["attribute", "value", "type"])
        for t in (self.meta_table, self.attr_table):
            t.verticalHeader().setVisible(False)
            t.horizontalHeader().setStretchLastSection(True)
            t.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
            t.setWordWrap(True)
        iv.addWidget(QtWidgets.QLabel("Object"))
        iv.addWidget(self.meta_table, 1)
        iv.addWidget(QtWidgets.QLabel("Attributes"))
        iv.addWidget(self.attr_table, 1)
        self.right.addTab(ins, "Inspect")
        return self.right

    def _clear_display(self):
        self.title.setText("")
        self.canvas.figure.clear()
        self.canvas.draw_idle()
        self.matrix_model.set_array(None)
        self.text.setPlainText("")
        self.meta_table.setRowCount(0)
        self.attr_table.setRowCount(0)

    def _select(self, item):
        if item is None or item.data(0, ROLE) is None:
            return
        path, h5path, kind = item.data(0, ROLE)
        self.show_object(path, h5path)

    def show_object(self, path, h5path):
        """Show object ``h5path`` of ``path`` (Display and Inspect)."""
        import h5py
        self.current = (path, h5path)
        self._nx = None
        self._pix = None
        try:
            meta, attrs = hdf5.describe(path, h5path)
            with h5py.File(path, "r") as f:
                obj = f[h5path]
                if isinstance(obj, h5py.Dataset):
                    self._shape, self._dtype = obj.shape, obj.dtype
                else:
                    self._shape, self._dtype = None, None
            if self._shape is None:
                self._nx = hdf5.nxdata(path, h5path)
            target = self._nx.signal if self._nx is not None else h5path
            if self._nx is not None or len(self._shape) >= 2:
                try:
                    self._pix = hdf5.pixel_size(path, target)
                except Exception:  # noqa: BLE001 -- only cosmetic
                    self._pix = None
            self.keep_aspect.setEnabled(self._pix is not None)
        except Exception as e:  # noqa: BLE001
            self._clear_display()
            self.title.setText(f"{os.path.basename(path)}  {h5path}: {e}")
            return
        self.meta_table.setRowCount(len(meta))
        for r, (k, v) in enumerate(meta):
            self.meta_table.setItem(r, 0, QtWidgets.QTableWidgetItem(k))
            self.meta_table.setItem(r, 1, QtWidgets.QTableWidgetItem(v))
        self.attr_table.setRowCount(len(attrs))
        for r, row in enumerate(attrs):
            for c, v in enumerate(row):
                self.attr_table.setItem(r, c, QtWidgets.QTableWidgetItem(v))
        self.meta_table.resizeColumnToContents(0)
        self.attr_table.resizeColumnToContents(0)
        info = "" if self._shape is None else ("  scalar" if self._shape == () else f"  {self._shape}") + \
            f"  {self._dtype}"
        self.title.setText(f"<b>{os.path.basename(path)}</b>  {h5path}{info}")
        self._rebuild_dims()

    # ------------------------------------------------------------------ dimensions and modes
    def _effective_mode(self):
        m = self.mode.currentText()
        if self._shape is None:                                  # group
            if self._nx is None:
                return "Text"
            return "NX" if m in ("Auto", "Line", "Heatmap") else ("Text" if m == "Text" else "NX")
        numeric = hdf5.is_numeric(self._dtype)
        nd = len(self._shape)
        size = int(np.prod(self._shape)) if nd else 1
        if m == "Auto":
            if nd == 0 or size == 1:
                return "Text"
            if not numeric:
                return "Matrix"
            return "Line" if nd == 1 else "Heatmap"
        if m in ("Line", "Heatmap") and (not numeric or nd == 0):
            return "Matrix" if nd else "Text"
        if m == "Heatmap" and nd < 2:
            return "Line"
        return m

    def _rebuild_dims(self):
        while self.dims_lay.count():
            it = self.dims_lay.takeAt(0)
            if it.widget():
                it.widget().deleteLater()
        self.xdim = self.ydim = None
        self.index_spins = {}
        mode = self._effective_mode()
        shape = self._shape or ()
        nd = len(shape)
        if self.current is not None and nd >= 2 and mode in ("Line", "Heatmap", "Matrix"):
            self.dims_lay.addWidget(QtWidgets.QLabel("x"))
            self.xdim = QtWidgets.QComboBox()
            for d in range(nd):
                self.xdim.addItem(f"dim {d} ({shape[d]})", d)
            self.xdim.setCurrentIndex(nd - 1)
            self.dims_lay.addWidget(self.xdim)
            if mode in ("Heatmap", "Matrix"):
                self.dims_lay.addWidget(QtWidgets.QLabel("y"))
                self.ydim = QtWidgets.QComboBox()
                for d in range(nd):
                    self.ydim.addItem(f"dim {d} ({shape[d]})", d)
                self.ydim.setCurrentIndex(nd - 2)
                self.dims_lay.addWidget(self.ydim)
                self.ydim.currentIndexChanged.connect(lambda _i: self._update_spins())
            self.xdim.currentIndexChanged.connect(lambda _i: self._update_spins())
            for d in range(nd):
                lab = QtWidgets.QLabel(f"dim {d}")
                sp = QtWidgets.QSpinBox()
                sp.setRange(0, shape[d] - 1)
                sp.setKeyboardTracking(False)
                sp.valueChanged.connect(lambda _v: self._draw())
                self.dims_lay.addWidget(lab)
                self.dims_lay.addWidget(sp)
                self.index_spins[d] = (lab, sp)
            self._update_spins(draw=False)
        self.dims_lay.addStretch(1)
        self._draw()

    def _update_spins(self, draw=True):
        keep = {c.currentData() for c in (self.xdim, self.ydim) if c is not None}
        for d, (lab, sp) in self.index_spins.items():
            lab.setVisible(d not in keep)
            sp.setVisible(d not in keep)
        if draw:
            self._draw()

    def _index(self):
        return [sp.value() for _d, (_l, sp) in sorted(self.index_spins.items())]

    # ------------------------------------------------------------------ drawing
    def _draw(self):
        if self.current is None:
            return
        path, h5path = self.current
        mode = self._effective_mode()
        maxp = prefs.get("toolbox.viewer.max_points")
        try:
            if mode == "NX":
                self.stack.setCurrentIndex(0)
                self._draw_nx(path, maxp)
            elif mode == "Text":
                self.stack.setCurrentIndex(2)
                self._draw_text(path, h5path)
            elif mode == "Matrix":
                self.stack.setCurrentIndex(1)
                self._draw_matrix(path, h5path, maxp)
            elif mode == "Line":
                self.stack.setCurrentIndex(0)
                self._draw_line(path, h5path, maxp)
            else:
                self.stack.setCurrentIndex(0)
                self._draw_heatmap(path, h5path, maxp)
        except Exception as e:  # noqa: BLE001
            self.stack.setCurrentIndex(2)
            self.text.setPlainText(f"cannot display {h5path}:\n{type(e).__name__}: {e}")

    def _dims(self):
        nd = len(self._shape)
        x = self.xdim.currentData() if self.xdim is not None else nd - 1
        y = self.ydim.currentData() if self.ydim is not None else None
        return x, y

    def _draw_text(self, path, h5path):
        import h5py
        with h5py.File(path, "r") as f:
            obj = f[h5path]
            if isinstance(obj, h5py.Dataset):
                v = obj[()] if obj.size <= 10000 else obj[tuple(slice(0, 10) for _ in obj.shape)]
                self.text.setPlainText(hdf5.to_text(v, max_items=10000))
            else:
                lines = [f"group with {len(obj)} members", ""]
                for c in hdf5.children(path, h5path):
                    lines.append(f"{c.name:30s} {c.kind:12s} {c.info}")
                self.text.setPlainText("\n".join(lines))

    def _draw_matrix(self, path, h5path, maxp):
        nd = len(self._shape)
        if nd == 0:
            a, st = hdf5.read_slice(path, h5path)
            self.matrix_model.set_array(np.asarray(a))
            return
        x, y = self._dims()
        if nd == 1:
            a, st = hdf5.read_slice(path, h5path, x_dim=0, max_points=maxp)
            self.matrix_model.set_array(a, st[0], 1)
            return
        if y is None or y == x:
            y = next(d for d in range(nd) if d != x)
        a, st = hdf5.read_slice(path, h5path, x_dim=x, y_dim=y, index=self._index(), max_points=maxp)
        sx = st[sorted((x, y)).index(x)]
        sy = st[sorted((x, y)).index(y)]
        self.matrix_model.set_array(a, sy, sx)

    def _draw_line(self, path, h5path, maxp):
        fig = self.canvas.figure
        fig.clear()
        ax = fig.add_subplot(111)
        x, _ = self._dims()
        a, st = hdf5.read_slice(path, h5path, x_dim=x, index=self._index(), max_points=maxp)
        if np.iscomplexobj(a):
            a = np.abs(a)
        idx = np.arange(a.size) * (st[0] if st else 1)
        ax.plot(idx, a, "-", lw=1)
        ax.set_xlabel(f"index (dim {x})" + (f", every {st[0]}." if st and st[0] > 1 else ""))
        ax.set_ylabel(h5path.rsplit("/", 1)[-1])
        self._scales(ax, idx, a)
        self.canvas.draw_idle()

    def _draw_heatmap(self, path, h5path, maxp):
        fig = self.canvas.figure
        fig.clear()
        ax = fig.add_subplot(111)
        x, y = self._dims()
        if y == x:
            ax.text(0.5, 0.5, "choose two different dimensions", ha="center", transform=ax.transAxes)
            self.canvas.draw_idle()
            return
        a, st = hdf5.read_slice(path, h5path, x_dim=x, y_dim=y, index=self._index(), max_points=maxp)
        a = np.abs(a) if np.iscomplexobj(a) else a.astype(float)
        sx = st[sorted((x, y)).index(x)]
        sy = st[sorted((x, y)).index(y)]
        norm = None
        if self.logc.isChecked():
            pos = a[np.isfinite(a) & (a > 0)]
            if pos.size:
                norm = LogNorm(vmin=pos.min(), vmax=max(pos.max(), pos.min() * 1.0001))
                a = np.where(a > 0, a, np.nan)
        ext = (-0.5 * sx, (a.shape[1] - 0.5) * sx, -0.5 * sy, (a.shape[0] - 0.5) * sy)
        im = ax.imshow(a, origin="lower", aspect=self._aspect(len(self._shape), x, y), interpolation="nearest",
                       norm=norm, cmap=self.cmap.currentText(), extent=ext)
        image_colorbar(fig, ax, im)
        ax.set_xlabel(f"dim {x}" + self._pix_label(len(self._shape), x) + (f" (every {sx}.)" if sx > 1 else ""))
        ax.set_ylabel(f"dim {y}" + self._pix_label(len(self._shape), y) + (f" (every {sy}.)" if sy > 1 else ""))
        ax.set_title(h5path.rsplit("/", 1)[-1], fontsize="medium")
        self.canvas.draw_idle()

    def _aspect(self, ndim, x, y):
        """Pixel aspect ratio of a detector image (x_pixel_size / y_pixel_size), else fill the panel."""
        if not self.keep_aspect.isChecked():
            return "auto"
        r = hdf5.pixel_aspect(self._pix, ndim, x, y)
        return "auto" if r is None else r

    def _pix_label(self, ndim, d):
        if not self._pix or ndim < 2:
            return ""
        s = {ndim - 1: ("x", self._pix[0]), ndim - 2: ("y", self._pix[1])}.get(d)
        return f", {s[0]}: {s[1] * 1e3:g} mm pixels" if s else ""

    def _scales(self, ax, x, y):
        x, y = np.asarray(x, float), np.asarray(y, float)
        if self.logx.isChecked() and np.any(x > 0):
            ax.set_xscale("log")
        if self.logy.isChecked() and np.any(y > 0):
            ax.set_yscale("log")

    def _draw_nx(self, path, maxp):
        import h5py
        nx = self._nx
        fig = self.canvas.figure
        fig.clear()
        ax = fig.add_subplot(111)
        with h5py.File(path, "r") as f:
            sig = f[nx.signal]
            unit = hdf5.to_text(sig.attrs.get("units", ""))
            if sig.ndim == 1:
                y = sig[()]
                xp = nx.axes[0] if nx.axes else None
                xv = f[xp][()] if xp else np.arange(y.size)
                xun = hdf5.to_text(f[xp].attrs.get("units", "")) if xp else ""
                e = f[nx.errors][()] if nx.errors and self.errbars.isChecked() else None
                if e is not None and e.shape == y.shape:
                    ax.errorbar(xv, y, yerr=e, fmt="o", ms=3, lw=0.8, capsize=0)
                else:
                    ax.plot(xv, y, "o-", ms=3, lw=0.8)
                ax.set_xlabel((xp.rsplit("/", 1)[-1] if xp else "index") + (f" [{xun}]" if xun else ""))
                ax.set_ylabel(nx.signal.rsplit("/", 1)[-1] + (f" [{unit}]" if unit else ""))
                self._scales(ax, xv, y)
            else:
                nd = sig.ndim
                a = sig[tuple([0] * (nd - 2) + [slice(None), slice(None)])].astype(float)   # last two dims
                norm = None
                if self.logc.isChecked():
                    pos = a[np.isfinite(a) & (a > 0)]
                    if pos.size:
                        norm = LogNorm(vmin=pos.min(), vmax=max(pos.max(), pos.min() * 1.0001))
                        a = np.where(a > 0, a, np.nan)
                im = ax.imshow(a, origin="lower", aspect=self._aspect(nd, nd - 1, nd - 2), interpolation="nearest",
                               norm=norm, cmap=self.cmap.currentText())
                image_colorbar(fig, ax, im, unit)
                ax.set_xlabel("x [pixel]" + self._pix_label(nd, nd - 1))
                ax.set_ylabel("y [pixel]" + self._pix_label(nd, nd - 2))
        ax.set_title(f"NXdata {nx.title}", fontsize="medium")
        self.canvas.draw_idle()

    # ------------------------------------------------------------------ to the 2D data page
    def image_of_current(self):
        """The shown dataset (2D, or summed over the leading dimensions) as an Image2D with the geometry
        found in the file (SANS-I raw files: everything; NeXus: the pixel size)."""
        from . import image2d as i2
        if self.current is None:
            raise ValueError("no dataset shown")
        path, h5path = self.current
        import h5py
        with h5py.File(path, "r") as f:
            obj = f[h5path]
            if not isinstance(obj, h5py.Dataset) or obj.ndim < 2:
                raise ValueError("not a detector image (a dataset with at least 2 dimensions)")
            a = np.asarray(obj[()])
            integer = a.dtype.kind in "iu"
        a = a.astype(float)
        while a.ndim > 2:
            a = a.sum(axis=0)
        dI = np.sqrt(np.clip(a, 0, None)) if integer else None
        meta = {"sample": os.path.basename(path)}
        img = i2.Image2D(a, dI, np.zeros(a.shape, bool), path, "hdf5", meta, entry=h5path,
                         unit="counts" if integer else "")
        try:                                           # SANS-I raw file: full geometry and header items
            from ..sasred.io.metadata import run_metadata
            from ..sasred.io.raw import read_raw
            raw = read_raw(path, checksum=False)
            if raw.counts.shape == a.shape:
                img.meta.update(run_metadata(raw))
                img.meta["sample"] = raw.sample_name
                ps = raw.pixel_size_m
                img.pixel_m = (ps, ps) if np.isscalar(ps) else tuple(ps)
                img.L2_m = float(raw.L2_effective_m)
                img.wavelength_A = float(raw.wavelength_A)
        except Exception:  # noqa: BLE001  (not a SANS-I raw file)
            sz = hdf5.pixel_size(path, h5path)
            if sz:
                img.pixel_m = tuple(sz)
        from .analysis2d import beam_centre_of_gravity
        ny, nx = a.shape
        img.centre_px = beam_centre_of_gravity(a) if a.max() > 50 * max(np.median(a), 1) else ((nx - 1) / 2, (ny - 1) / 2)
        return img

    def on_to_2d(self):
        try:
            img = self.image_of_current()
        except Exception as exc:  # noqa: BLE001
            self.status.emit(f"→ 2D data: {exc}")
            return
        self.to_2d.emit(img)

