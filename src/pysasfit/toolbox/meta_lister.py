"""
HDF Metadata Lister (pyTOOLbox tab; from ``tools/hdf_meta_lister.py``)
----------------------------------------------------------------------
Reads metadata (attributes and small datasets) from HDF5 files named
<instr><year>n<number>.hdf (e.g. sans2026n052752.hdf), lists the selected
entries in a table, copies it to the clipboard (tab separated, ready for a
spreadsheet) and saves it as Excel, CSV or plain text.

Values in the table can be edited (double-click or F2). Edited cells are
highlighted and are only written into the HDF files when you press
"Write changed values to HDF files" (optionally with a .bak backup).

Requirements:  pip install pysasfit[gui,toolbox]  (PySide6, h5py; openpyxl only for Excel export)

All settings (folder, instrument, year, number range, the list of known
entries and which of them are selected) are stored in ``toolbox_meta_lister.json``
next to the pySASfit preferences file and restored at the next start (the file
``~/.hdf_meta_lister.json`` of the stand-alone script is read once if it exists).
The largest dataset still counted as metadata and the number of digits of
floating-point values are in Settings -> Preferences -> pyTOOLbox.
"""
import csv
import datetime
import json
import os
import re
import shutil

import h5py
import numpy as np
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor, QKeySequence, QShortcut
from PySide6.QtWidgets import (
    QAbstractItemView, QApplication, QCheckBox, QFileDialog, QFrame,
    QGroupBox, QHBoxLayout, QHeaderView, QLabel, QLineEdit,
    QMessageBox, QPushButton, QScrollArea, QSplitter,
    QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget,
)

from ..core import settings as prefs

__all__ = ["MetaLister", "collect_entries", "write_values", "short_names", "to_native", "column_group"]

OLD_CONFIG_FILE = os.path.join(os.path.expanduser("~"), ".hdf_meta_lister.json")


def config_file():
    return os.path.join(os.path.dirname(prefs.settings_path()), "toolbox_meta_lister.json")


def _legacy_config_file():
    """Settings file of the lister when the tool was called pySASbox (read once if present)."""
    return os.path.join(os.path.dirname(prefs.settings_path()), "sasbox_meta_lister.json")


def max_dataset_elements():
    """Larger datasets (images, histograms) are not metadata."""
    return prefs.get("toolbox.lister.max_elements")


def float_digits():
    return prefs.get("toolbox.lister.float_digits")

OLD_PREFIX = "old_"          # prefix of the entry that keeps the original value
CHANGED_COLOR = QColor("#ffd966")   # edited, not yet written (black text on yellow)


# ----------------------------------------------------------------- helpers
class ArrayText(str):
    """Text form of a multi-element array value; shown, but not editable."""


def to_native(v):
    """Convert h5py/numpy values into plain python values (int/float/str)."""
    if isinstance(v, bytes):
        return v.decode("utf-8", errors="replace").strip("\x00").strip()
    if isinstance(v, np.ndarray):
        if v.size == 1:
            return to_native(v.reshape(-1)[0])
        return ArrayText("[" + ", ".join(str(to_native(x)) for x in v.reshape(-1)) + "]")
    if isinstance(v, np.generic):
        return to_native(v.item())
    if isinstance(v, (list, tuple)):
        return ArrayText("[" + ", ".join(str(to_native(x)) for x in v) + "]")
    if isinstance(v, float):
        return round(v, float_digits())   # all floating point numbers: n digits after "."
    return v


def short_names(keys):
    """Column names: last path component only (e.g. '/entry/sample/temp' -> 'temp').
    If two selected entries share the same last name, parent folders are
    prepended until the names are unique."""
    parts = [k.strip("/").split("/") for k in keys]
    depth = [1] * len(keys)

    def name(i):
        return "/".join(parts[i][-depth[i]:])

    while True:
        names = [name(i) for i in range(len(keys))]
        dup = False
        for n in set(names):
            idx = [i for i, x in enumerate(names) if x == n]
            if len(idx) > 1:
                for i in idx:
                    if depth[i] < len(parts[i]):
                        depth[i] += 1
                        dup = True
        if not dup:
            return names


def column_group(key):
    """Group a listed entry belongs to: the HDF group holding it, e.g. '/entry1/SANS/attenuator/selection'
    -> 'attenuator', '/entry1/sample/name' -> 'sample', '/entry1/SANS/detector@x' -> 'detector'."""
    path = key.split("@", 1)[0].strip("/").split("/")
    if "@" in key:
        return path[-1] if path and path[-1] else ""
    return path[-2] if len(path) >= 2 else ""


def collect_entries(path):
    """Return {entry_key: value} for all attributes and small datasets.
    Keys: '/group/dataset' for datasets, '/group@attribute' for attributes."""
    out = {}
    nmax = max_dataset_elements()
    with h5py.File(path, "r") as f:
        def add_attrs(name, obj):
            for a, v in obj.attrs.items():
                try:
                    out[f"{name}@{a}"] = to_native(v)
                except Exception:
                    pass

        def visit(name, obj):
            full = "/" + name
            add_attrs(full, obj)
            if isinstance(obj, h5py.Dataset):
                try:
                    if obj.size <= nmax and obj.dtype.kind != "V":
                        out[full] = to_native(obj[()])
                except Exception:
                    pass

        add_attrs("/", f)
        f.visititems(visit)
    return out


def replace_string_dataset(f, ds, raw):
    """A fixed-length string dataset cannot hold a longer text in place. Create a
    new dataset with a long enough fixed length (same shape, same attributes),
    then replace the old one under the same name. The new dataset is built under
    a temporary name first, so the old one is only removed when the new one exists."""
    parent_path, _, base = ds.name.rpartition("/")
    parent = f[parent_path or "/"]
    tmp = base + "__new_tmp"
    dt = np.dtype(f"S{len(raw)}")
    data = np.asarray(raw, dtype=dt) if ds.shape == () else np.full(ds.shape, raw, dtype=dt)
    try:
        new = parent.create_dataset(tmp, data=data)
        for k in ds.attrs.keys():
            new.attrs.create(k, ds.attrs[k], dtype=ds.attrs.get_id(k).dtype)
        del parent[base]
        parent.move(tmp, base)
    except Exception:
        if tmp in parent:
            del parent[tmp]
        raise


def write_values(path, changes):
    """Write new values into an HDF5 file (opened read/write).
    changes: {entry_key: python value (int, float or str)}.
    Before an entry is changed, its original value is stored as a new entry
    'old_<name>' (same group / same object, same type). An already existing
    'old_<name>' is left untouched, so it keeps the very first original value.
    The storage type of numeric entries is kept. Returns {entry_key: error text}
    for the entries that could not be written (empty dict = all written)."""
    errors = {}
    with h5py.File(path, "r+") as f:
        for key, val in changes.items():
            try:
                if "@" in key:                                  # attribute
                    target, attr = key.rsplit("@", 1)
                    obj = f[target or "/"]
                    old_name = OLD_PREFIX + attr
                    if old_name not in obj.attrs:               # keep the original value
                        obj.attrs.create(old_name, obj.attrs[attr],
                                         dtype=obj.attrs.get_id(attr).dtype)
                    if isinstance(val, str):
                        obj.attrs[attr] = val
                    else:
                        old_dtype = np.asarray(obj.attrs[attr]).dtype
                        obj.attrs.modify(attr, np.asarray(val, dtype=old_dtype))
                else:                                           # dataset
                    ds = f[key]
                    parent, _, base = ds.name.rpartition("/")
                    old_name = OLD_PREFIX + base
                    if old_name not in f[parent or "/"]:        # keep the original value
                        f[parent or "/"].copy(ds, f[parent or "/"], name=old_name)
                    if isinstance(val, str):
                        if ds.dtype.kind == "S":                # fixed-length string
                            raw = val.encode("utf-8")
                            if len(raw) > ds.dtype.itemsize:
                                # fixed length too small: store as a new, longer
                                # fixed-length string dataset (same name, shape, attributes)
                                replace_string_dataset(f, ds, raw)
                            else:
                                ds[...] = np.asarray(raw, dtype=ds.dtype)
                        else:                                   # variable-length string
                            try:
                                ds[...] = val
                            except Exception:
                                ds[...] = np.asarray([val], dtype=object)
                    else:
                        ds[...] = np.asarray(val, dtype=ds.dtype)
            except Exception as e:
                errors[key] = str(e)
    return errors


def load_config():
    for p in (config_file(), _legacy_config_file(), OLD_CONFIG_FILE):
        try:
            with open(p, "r", encoding="utf-8") as fh:
                return json.load(fh)
        except Exception:
            continue
    return {}


def save_config(cfg):
    try:
        p = config_file()
        os.makedirs(os.path.dirname(p), exist_ok=True)
        with open(p, "w", encoding="utf-8") as fh:
            json.dump(cfg, fh, indent=2)
    except Exception as e:
        print("Could not save config:", e)


# --------------------------------------------------------------------- GUI
class MetaLister(QWidget):
    """The lister as a widget (a tab of the pyTOOLbox window)."""

    #: double-click on a file name: (absolute path)
    open_file_requested = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("HDF Metadata Lister")
        cfg = load_config()

        # entry key -> selected (True = list, False = skip); keeps insertion order
        self.entry_state = {k: bool(v) for k, v in cfg.get("entries", {}).items()}
        self.entry_widgets = {}          # key -> (row widget, check box: listed or not)
        self.columns = []                # single-line names (used for copy / save)
        self.units = []                  # unit of each column ("" if none), from the @units attribute
        self.groups = []                 # group of each column (attenuator, beam_stop, detector, ...)
        self.rows = []                   # current values (edits included)
        self.keys = []                   # entry keys of the listed columns (after "file")
        self.paths = []                  # HDF file of each row
        self.missing = set()             # (row, col) cells whose entry is absent in that file
        self.pending = {}                # (row, col) -> new value, not yet written to the file
        self.original = {}               # (row, col) -> value as read from the file
        self._invalid = False            # set by on_item_changed when a typed value is rejected

        # ---- file selection
        files_box = QGroupBox("Files")
        self.folder_edit = QLineEdit(cfg.get("folder", ""))
        browse = QPushButton("Browse...")
        browse.clicked.connect(self.browse)
        self.instr_edit = QLineEdit(cfg.get("instr", "sans"))
        self.instr_edit.setMaximumWidth(110)
        self.year_edit = QLineEdit(cfg.get("year") or str(datetime.date.today().year))
        self.year_edit.setMaximumWidth(70)
        self.from_edit = QLineEdit(str(cfg.get("num_from", 1)))
        self.from_edit.setMaximumWidth(90)
        self.to_edit = QLineEdit(str(cfg.get("num_to", 999999)))
        self.to_edit.setMaximumWidth(90)

        r1 = QHBoxLayout()
        r1.addWidget(QLabel("Folder:"))
        r1.addWidget(self.folder_edit, 1)
        r1.addWidget(browse)
        r2 = QHBoxLayout()
        for text, w in (("Instrument:", self.instr_edit), ("Year:", self.year_edit),
                        ("Number from:", self.from_edit), ("to:", self.to_edit)):
            r2.addWidget(QLabel(text))
            r2.addWidget(w)
        r2.addStretch(1)
        fl = QVBoxLayout(files_box)
        fl.addLayout(r1)
        fl.addLayout(r2)

        # ---- left: entry selection (a check box per entry)
        left_box = QGroupBox("Entries to list")
        scan_btn = QPushButton("Scan files for entries")
        scan_btn.clicked.connect(self.scan)
        all_btn = QPushButton("All")
        all_btn.clicked.connect(lambda: self.set_all(True))
        none_btn = QPushButton("None")
        none_btn.clicked.connect(lambda: self.set_all(False))
        bar = QHBoxLayout()
        bar.addWidget(scan_btn)
        bar.addWidget(all_btn)
        bar.addWidget(none_btn)
        self.filter_edit = QLineEdit()
        self.filter_edit.setPlaceholderText("filter entries ...")
        self.filter_edit.setClearButtonEnabled(True)
        self.filter_edit.textChanged.connect(self.apply_filter)
        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.scroll.setFrameShape(QFrame.Shape.StyledPanel)
        ll = QVBoxLayout(left_box)
        ll.addLayout(bar)
        ll.addWidget(self.filter_edit)
        ll.addWidget(self.scroll, 1)

        # ---- right: listing table (values editable)
        right_box = QGroupBox("Listing  (double-click / F2 edits a value; double-click a file name opens it "
                              "in the HDF5 viewer)")
        self.table = QTableWidget()
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.DoubleClicked |
                                   QAbstractItemView.EditTrigger.EditKeyPressed)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.table.setAlternatingRowColors(True)
        # high contrast independent of the system (dark/light) theme:
        # black text on white / very light grey rows, light-blue selection
        self.table.setStyleSheet(
            "QTableWidget { background-color: #ffffff; alternate-background-color: #eef2f7;"
            " color: #000000; gridline-color: #c8ced6;"
            " selection-background-color: #b3d7ff; selection-color: #000000; }")
        hh = self.table.horizontalHeader()
        hh.setDefaultAlignment(Qt.AlignmentFlag.AlignCenter)
        hh.setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        self.table.itemChanged.connect(self.on_item_changed)
        self.table.cellDoubleClicked.connect(self._on_double_click)
        sc = QShortcut(QKeySequence.StandardKey.Copy, self.table)
        sc.setContext(Qt.ShortcutContext.WidgetShortcut)
        sc.activated.connect(lambda: self.copy_clipboard(selection_only=True))
        sp = QShortcut(QKeySequence.StandardKey.Paste, self.table)
        sp.setContext(Qt.ShortcutContext.WidgetShortcut)
        sp.activated.connect(self.paste_clipboard)
        rl = QVBoxLayout(right_box)
        rl.addWidget(self.table)

        splitter = QSplitter()
        splitter.addWidget(left_box)
        splitter.addWidget(right_box)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([430, 820])

        # ---- bottom buttons
        gen = QPushButton("Generate listing")
        gen.clicked.connect(self.generate)
        cp = QPushButton("Copy to clipboard")
        cp.clicked.connect(lambda: self.copy_clipboard(False))
        sv = QPushButton("Save as Excel / CSV / Text...")
        sv.clicked.connect(self.save_file)
        pb = QPushButton("Paste from clipboard")
        pb.setToolTip("Pastes the clipboard (one value per line, tab = next column) "
                      "into the table, starting at the current cell (Ctrl+V)")
        pb.clicked.connect(self.paste_clipboard)
        self.write_btn = QPushButton("Write changed values to HDF files")
        self.write_btn.setToolTip("Writes all highlighted (edited) values into the HDF files")
        self.write_btn.setEnabled(False)
        self.write_btn.clicked.connect(self.write_changes_to_files)
        self.status = QLabel("Ready.")
        bottom = QHBoxLayout()
        bottom.addWidget(gen)
        from ..core import pycode
        bottom.addWidget(pycode.button(self.python_code, self))
        bottom.addWidget(cp)
        bottom.addWidget(sv)
        bottom.addWidget(pb)
        bottom.addWidget(self.write_btn)
        bottom.addSpacing(12)
        bottom.addWidget(self.status, 1)

        lay = QVBoxLayout(self)
        lay.addWidget(files_box)
        lay.addWidget(splitter, 1)
        lay.addLayout(bottom)

        self.rebuild_entry_list()

    # ---- helpers
    def _on_double_click(self, r, c):
        if c == 0 and 0 <= r < len(self.paths):
            self.open_file_requested.emit(self.paths[r])

    def set_status(self, text):
        self.status.setText(text)
        QApplication.processEvents()

    def warn(self, title, text):
        QMessageBox.warning(self, title, text)

    # ---- entry list
    def rebuild_entry_list(self):
        container = QWidget()
        vl = QVBoxLayout(container)
        vl.setContentsMargins(4, 4, 4, 4)
        vl.setSpacing(2)
        self.entry_widgets = {}
        for key, sel in self.entry_state.items():
            if key.endswith("@units"):          # units go into the column headers, not into columns
                continue
            row = QWidget()
            hl = QHBoxLayout(row)
            hl.setContentsMargins(0, 0, 0, 0)
            cb = QCheckBox()                     # checked: listed; no text, little space
            cb.setChecked(bool(sel))
            cb.setToolTip("list this entry")
            cb.toggled.connect(lambda on, k=key: self.entry_state.__setitem__(k, on))
            label = QLabel(key)
            label.setToolTip(key)
            hl.addWidget(cb)
            hl.addWidget(label, 1)
            vl.addWidget(row)
            self.entry_widgets[key] = (row, cb)
        vl.addStretch(1)
        self.scroll.setWidget(container)
        self.apply_filter()

    def apply_filter(self):
        needle = self.filter_edit.text().strip().lower()
        for key, (row, _) in self.entry_widgets.items():
            row.setVisible(needle in key.lower())

    def set_all(self, value):
        needle = self.filter_edit.text().strip().lower()
        for key, (_, cb) in self.entry_widgets.items():
            if needle in key.lower():
                cb.setChecked(bool(value))

    # ---- file handling
    def browse(self):
        d = QFileDialog.getExistingDirectory(
            self, "Folder containing the HDF files", self.folder_edit.text())
        if d:
            self.folder_edit.setText(d)

    def matching_files(self):
        folder = self.folder_edit.text().strip()
        if not os.path.isdir(folder):
            raise ValueError("Please select an existing folder.")
        try:
            lo, hi = int(self.from_edit.text()), int(self.to_edit.text())
        except ValueError:
            raise ValueError("Number range must be integers.")
        pat = re.compile(
            rf"^{re.escape(self.instr_edit.text().strip())}"
            rf"{re.escape(self.year_edit.text().strip())}n(\d+)\.hdf$", re.IGNORECASE)
        found = []
        for name in os.listdir(folder):
            m = pat.match(name)
            if m and lo <= int(m.group(1)) <= hi:
                found.append((int(m.group(1)), name))
        found.sort()
        return [(os.path.join(folder, name), name) for _, name in found]

    def scan(self):
        try:
            files = self.matching_files()
        except ValueError as e:
            self.warn("Error", str(e))
            return
        if not files:
            self.warn("No files", "No matching files found.")
            return
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        new = 0
        try:
            for k, (path, name) in enumerate(files, 1):
                self.set_status(f"Scanning {name} ({k}/{len(files)}) ...")
                try:
                    for key in collect_entries(path):
                        if key not in self.entry_state:
                            self.entry_state[key] = False
                            new += 1
                except Exception as e:
                    print(f"{name}: {e}")
        finally:
            QApplication.restoreOverrideCursor()
        self.rebuild_entry_list()
        self.set_status(f"Scan done: {len(self.entry_state)} entries known, {new} new.")
        self.persist()

    def python_code(self):
        """Script: the files of the number range and the ticked entries, listed as a table (pandas when
        installed) ("Copy Python code")."""
        from ..core.pycode import Script, lit, path_lit

        keys = [k for k, v in self.entry_state.items() if v and not k.endswith("@units")]
        if not keys:
            raise ValueError("Tick the entries to list first.")
        sc = Script("pyTOOLbox", "HDF metadata listing")
        sc.imp("import os", "import re", "from pysasfit.toolbox.meta_lister import collect_entries")
        folder = self.folder_edit.text().strip()
        sc.add(f"folder = {path_lit(folder) if folder else path_lit(None)}",
               f"instrument, year = {self.instr_edit.text().strip()!r}, {self.year_edit.text().strip()!r}",
               f"first, last = {self.from_edit.text().strip() or 0}, {self.to_edit.text().strip() or 0}",
               f"keys = {lit(keys, 0)}",
               "pat = re.compile(rf'^{re.escape(instrument)}{re.escape(year)}n(\\d+)\\.hdf$', re.IGNORECASE)",
               "files = sorted((int(m.group(1)), name) for name in os.listdir(folder)",
               "               if (m := pat.match(name)) and first <= int(m.group(1)) <= last)",
               "rows = []",
               "for _, name in files:",
               "    data = collect_entries(os.path.join(folder, name))",
               "    rows.append([name] + [data.get(k, '') for k in keys])",
               "try:",
               "    import pandas as pd",
               "    table = pd.DataFrame(rows, columns=['file'] + keys)",
               "    print(table.to_string())",
               "except ImportError:",
               "    for row in rows:",
               "        print('\\t'.join(str(v) for v in row))")
        return sc.text()

    # ---- listing
    def generate(self):
        keys = [k for k, v in self.entry_state.items() if v and not k.endswith("@units")]
        if not keys:
            self.warn("Nothing selected", "Select at least one entry (tick it in the list).")
            return
        if self.pending and QMessageBox.question(
                self, "Unsaved changes",
                f"{len(self.pending)} edited value(s) have not been written to the HDF "
                "files and will be lost. Continue?") != QMessageBox.StandardButton.Yes:
            return
        try:
            files = self.matching_files()
        except ValueError as e:
            self.warn("Error", str(e))
            return
        if not files:
            self.warn("No files", "No matching files found.")
            return
        self.keys = keys
        self.columns = ["file"] + short_names(keys)
        self.units = [""] * len(self.columns)
        self.groups = [""] + [column_group(k) for k in keys]
        self.paths = [p for p, _ in files]
        self.rows, self.missing = [], set()
        self.pending, self.original = {}, {}
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            for r, (path, name) in enumerate(files):
                self.set_status(f"Reading {name} ({r + 1}/{len(files)}) ...")
                try:
                    data = collect_entries(path)
                    for c, key in enumerate(keys, 1):
                        u = data.get(f"{key}@units")
                        if u and not self.units[c]:
                            self.units[c] = str(u)
                    row = [name]
                    for c, key in enumerate(keys, 1):
                        if key in data:
                            row.append(data[key])
                        else:
                            row.append("")
                            self.missing.add((r, c))
                except Exception as e:
                    row = [name, f"ERROR: {e}"] + [""] * (len(keys) - 1)
                    self.missing.update((r, c) for c in range(1, len(keys) + 1))
                self.rows.append(row)
            self.fill_table()
        finally:
            QApplication.restoreOverrideCursor()
        self.update_write_button()
        self.set_status(f"{len(self.rows)} files listed.")
        self.persist()

    def fill_table(self):
        # Headers: the group (attenuator, beam_stop, detector, sample, monitor1, ...) over the name,
        # the unit in brackets below. Copied / saved headers are one line: "name [unit]".
        any_unit = any(self.units)
        heads = []
        for c, (col, grp) in enumerate(zip(self.columns, self.groups)):
            last = col.rpartition("/")[2]
            unit = f"[{self.units[c]}]" if self.units[c] else ""
            heads.append(f"{grp}\n{last}" + (f"\n{unit}" if any_unit else ""))
        self.table.blockSignals(True)
        self.table.clear()
        self.table.setColumnCount(len(self.columns))
        self.table.setRowCount(len(self.rows))
        self.table.setHorizontalHeaderLabels(heads)
        for r, row in enumerate(self.rows):
            for c, val in enumerate(row):
                item = QTableWidgetItem(str(val))
                editable = (c >= 1 and (r, c) not in self.missing
                            and not isinstance(val, (ArrayText, bool)))
                if not editable:
                    item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
                self.table.setItem(r, c, item)
        self.table.blockSignals(False)
        self.table.resizeColumnsToContents()   # narrowest width that shows everything
        self.table.horizontalHeader().setMinimumHeight(
            self.table.horizontalHeader().sizeHint().height())

    # ---- editing values
    def on_item_changed(self, item):
        r, c = item.row(), item.column()
        if c < 1 or r >= len(self.rows):
            return
        old = self.original.get((r, c), self.rows[r][c])
        text = item.text().strip()
        try:
            if isinstance(old, int):
                new = int(text)
            elif isinstance(old, float):
                new = float(text.replace(",", "."))
            else:
                new = item.text()
        except ValueError:
            kind = "an integer" if isinstance(old, int) else "a number"
            self._invalid = True
            self.table.blockSignals(True)
            item.setText(str(self.rows[r][c]))
            self.table.blockSignals(False)
            self.set_status(f"'{text}' is not {kind} - value not changed.")
            return
        self.table.blockSignals(True)
        if new == old:
            self.pending.pop((r, c), None)
            self.original.pop((r, c), None)
            item.setData(Qt.ItemDataRole.BackgroundRole, None)
            self.rows[r][c] = old
            item.setText(str(old))
        else:
            self.original.setdefault((r, c), old)
            self.pending[(r, c)] = new
            self.rows[r][c] = new
            item.setBackground(CHANGED_COLOR)
            item.setText(str(new))
        self.table.blockSignals(False)
        self.table.resizeColumnToContents(c)
        self.update_write_button()

    def paste_clipboard(self):
        """Paste clipboard text into the table like a spreadsheet: one line per row
        (file), tab-separated values go to the following columns, starting at the
        current cell. Every value is checked like a typed value; empty values,
        non-editable cells and cells outside the table are skipped. Nothing is
        written to the HDF files until 'Write changed values' is used."""
        if not self.rows:
            return
        text = QApplication.clipboard().text()
        grid = [line.split("\t") for line in text.splitlines()]
        if not grid or not any(v.strip() for row in grid for v in row):
            self.set_status("Clipboard is empty.")
            return
        r0, c0 = self.table.currentRow(), self.table.currentColumn()
        if r0 < 0 or c0 < 0:
            self.set_status("Click the first target cell, then paste.")
            return
        changed = invalid = skipped = beyond = 0
        for i, row in enumerate(grid):
            r = r0 + i
            if r >= self.table.rowCount():
                beyond += 1
                continue
            for j, val in enumerate(row):
                c = c0 + j
                val = val.strip("\r")
                if not val.strip():
                    continue                              # empty -> leave value unchanged
                item = self.table.item(r, c) if c < self.table.columnCount() else None
                if item is None or not (item.flags() & Qt.ItemFlag.ItemIsEditable):
                    skipped += 1
                    continue
                before = item.text()
                self._invalid = False
                item.setText(val)
                if self._invalid:
                    invalid += 1
                elif item.text() != before:
                    changed += 1
        msg = f"Pasted: {changed} value(s) changed"
        if invalid:
            msg += f", {invalid} invalid (kept)"
        if skipped:
            msg += f", {skipped} skipped (not editable)"
        if beyond:
            msg += f", {beyond} line(s) beyond the last file"
        self.set_status(msg + ". Not yet written to the HDF files.")

    def update_write_button(self):
        n = len(self.pending)
        self.write_btn.setEnabled(n > 0)
        self.write_btn.setText(f"Write changed values to HDF files ({n})" if n
                               else "Write changed values to HDF files")
        if n:
            self.set_status(f"{n} edited value(s) - not yet written to the HDF files.")

    def write_changes_to_files(self):
        if not self.pending:
            return
        files = sorted({r for r, _ in self.pending})
        box = QMessageBox(self)
        box.setWindowTitle("Write changed values to HDF files")
        box.setIcon(QMessageBox.Icon.Question)
        box.setText(f"Overwrite {len(self.pending)} value(s) in {len(files)} HDF file(s)?\n\n"
                    "The files are modified in place. The storage type of numbers is kept.\n"
                    "Each original value is first saved as a new entry old_<name> "
                    "(unless old_<name> already exists - it is then left untouched).")
        backup = QCheckBox("Keep a backup copy of each changed file (<name>.hdf.bak)")
        backup.setChecked(True)
        box.setCheckBox(backup)
        box.setStandardButtons(QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
        box.setDefaultButton(QMessageBox.StandardButton.No)
        if box.exec() != QMessageBox.StandardButton.Yes:
            return
        self.apply_pending(backup.isChecked())

    def apply_pending(self, make_backup=True):
        failed = []                                  # messages
        written = 0
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            by_row = {}
            for (r, c), val in self.pending.items():
                by_row.setdefault(r, {})[c] = val
            for n, (r, cells) in enumerate(sorted(by_row.items()), 1):
                path = self.paths[r]
                self.set_status(f"Writing {os.path.basename(path)} ({n}/{len(by_row)}) ...")
                try:
                    if make_backup and not os.path.exists(path + ".bak"):
                        shutil.copy2(path, path + ".bak")   # never overwrites an older backup
                    errors = write_values(path, {self.keys[c - 1]: v for c, v in cells.items()})
                except Exception as e:                       # file locked, read-only, ...
                    errors = {self.keys[c - 1]: str(e) for c in cells}
                for c in cells:
                    key = self.keys[c - 1]
                    if key in errors:
                        failed.append(f"{os.path.basename(path)}  {key}: {errors[key]}")
                        continue
                    written += 1
                    self.pending.pop((r, c))
                    self.original.pop((r, c), None)
                    item = self.table.item(r, c)
                    self.table.blockSignals(True)
                    item.setData(Qt.ItemDataRole.BackgroundRole, None)
                    self.table.blockSignals(False)
        finally:
            QApplication.restoreOverrideCursor()
        self.update_write_button()
        self.set_status(f"{written} value(s) written to the HDF files."
                        + (f" {len(failed)} failed (still highlighted)." if failed else ""))
        if failed:
            self.warn("Some values could not be written",
                      "\n".join(failed[:15]) + ("\n..." if len(failed) > 15 else ""))
        return written, failed

    # ---- export
    @staticmethod
    def clean(x):
        return str(x).replace("\t", " ").replace("\r", " ").replace("\n", " ")

    def column_titles(self):
        """One-line column titles for copy / save: "name [unit]"."""
        units = self.units or [""] * len(self.columns)
        return [c + (f" [{u}]" if u else "") for c, u in zip(self.columns, units)]

    def as_text(self, rows):
        lines = ["\t".join(self.clean(c) for c in self.column_titles())]
        lines += ["\t".join(self.clean(x) for x in r) for r in rows]
        return "\n".join(lines) + "\n"

    def copy_clipboard(self, selection_only=False):
        if not self.rows:
            return
        rows = self.rows
        if selection_only:
            idx = sorted({i.row() for i in self.table.selectionModel().selectedRows()})
            if idx:
                rows = [self.rows[i] for i in idx]
        QApplication.clipboard().setText(self.as_text(rows))
        self.set_status(f"{len(rows)} rows copied (tab separated) - paste into your spreadsheet.")

    def save_file(self):
        if not self.rows:
            QMessageBox.information(self, "Empty", "Generate a listing first.")
            return
        path, flt = QFileDialog.getSaveFileName(
            self, "Save listing", "",
            "Excel (*.xlsx);;CSV (*.csv);;Plain text, tab separated (*.txt)")
        if not path:
            return
        ext = os.path.splitext(path)[1].lower()
        if ext not in (".xlsx", ".csv", ".txt"):
            ext = ".csv" if "csv" in flt.lower() else ".txt" if "txt" in flt.lower() else ".xlsx"
            path += ext
        try:
            if ext == ".xlsx":
                from openpyxl import Workbook
                wb = Workbook()
                ws = wb.active
                ws.title = "Metadata"
                ws.append(self.column_titles())
                for r in self.rows:
                    ws.append([x if isinstance(x, (int, float)) else str(x) for x in r])
                wb.save(path)
            elif ext == ".csv":
                with open(path, "w", newline="", encoding="utf-8-sig") as fh:
                    w = csv.writer(fh)
                    w.writerow(self.column_titles())
                    w.writerows(self.rows)
            else:
                with open(path, "w", encoding="utf-8") as fh:
                    fh.write(self.as_text(self.rows))
            self.set_status(f"Saved {path}")
        except ImportError:
            self.warn("openpyxl missing", "Excel export needs:  pip install openpyxl")
        except Exception as e:
            self.warn("Save failed", str(e))

    # ---- persistence
    def persist(self):
        try:
            lo, hi = int(self.from_edit.text()), int(self.to_edit.text())
        except ValueError:
            lo, hi = 1, 999999
        save_config({
            "folder": self.folder_edit.text(),
            "instr": self.instr_edit.text(),
            "year": self.year_edit.text(),
            "num_from": lo,
            "num_to": hi,
            "entries": dict(self.entry_state),
        })

    def may_close(self):
        """False when the user wants to keep edited values that are not yet written."""
        if self.pending and QMessageBox.question(
                self, "Unsaved changes",
                f"{len(self.pending)} edited value(s) have not been written to the HDF "
                "files. Quit anyway?") != QMessageBox.StandardButton.Yes:
            return False
        self.persist()
        return True
