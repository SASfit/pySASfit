# SPDX-License-Identifier: BSD-3-Clause
"""*Add many samples*: step 1 selects runs (run range, pattern, conditions on header items),
step 2 groups them by header items with ± tolerances; every group becomes a sample row."""
from __future__ import annotations

from typing import Callable, Dict, List, Optional

from PySide6 import QtCore, QtWidgets

from .selection import HDF, Criterion, Metadata, group_selected, select_runs

__all__ = ["FieldPicker", "AddManySamplesDialog"]


def _float(text: str) -> Optional[float]:
    text = text.strip()
    if not text:
        return None
    return float(text)


class FieldPicker(QtWidgets.QWidget):
    """Check list of header items (left) and one editor row per checked item (right).
    ``mode`` "search": text → equal / regular expression, number → lo ≤ value ≤ hi;
    ``mode`` "group": ± tolerance (empty: equal values)."""

    changed = QtCore.Signal()

    def __init__(self, meta: Metadata, mode: str, checked=(), parent=None):
        super().__init__(parent)
        self.meta, self.mode = meta, mode
        lay = QtWidgets.QHBoxLayout(self)
        left = QtWidgets.QVBoxLayout()
        self.filter = QtWidgets.QLineEdit()
        self.filter.setPlaceholderText("filter items")
        self.filter.textChanged.connect(self._filter)
        self.list = QtWidgets.QListWidget()
        for f in meta.fields():
            it = QtWidgets.QListWidgetItem(f.replace(HDF, "file: "))
            it.setData(QtCore.Qt.ItemDataRole.UserRole, f)
            it.setFlags(it.flags() | QtCore.Qt.ItemFlag.ItemIsUserCheckable)
            it.setCheckState(QtCore.Qt.CheckState.Checked if f in checked else QtCore.Qt.CheckState.Unchecked)
            self.list.addItem(it)
        self.list.itemChanged.connect(lambda _it: self._rebuild())
        left.addWidget(self.filter)
        left.addWidget(self.list, 1)
        lay.addLayout(left, 2)
        self.form_box = QtWidgets.QWidget()
        self.form = QtWidgets.QGridLayout(self.form_box)
        self.form.setAlignment(QtCore.Qt.AlignmentFlag.AlignTop)
        sa = QtWidgets.QScrollArea()
        sa.setWidgetResizable(True)
        sa.setWidget(self.form_box)
        lay.addWidget(sa, 3)
        self.editors: Dict[str, tuple] = {}
        self._rebuild()

    def _filter(self, text):
        text = text.lower()
        for i in range(self.list.count()):
            it = self.list.item(i)
            self.list.setRowHidden(i, bool(text) and text not in it.text().lower())

    def checked(self) -> List[str]:
        return [self.list.item(i).data(QtCore.Qt.ItemDataRole.UserRole) for i in range(self.list.count())
                if self.list.item(i).checkState() == QtCore.Qt.CheckState.Checked]

    def check(self, field: str, on: bool = True):
        for i in range(self.list.count()):
            if self.list.item(i).data(QtCore.Qt.ItemDataRole.UserRole) == field:
                self.list.item(i).setCheckState(QtCore.Qt.CheckState.Checked if on else QtCore.Qt.CheckState.Unchecked)

    def _rebuild(self):
        old = {f: tuple(w.text() for w in ws) for f, (ws, _) in self.editors.items()}
        while self.form.count():
            w = self.form.takeAt(0).widget()
            if w is not None:
                w.hide()
                w.setParent(None)
                w.deleteLater()
        self.editors = {}
        for row, f in enumerate(self.checked()):
            num = self.meta.is_numeric(f)
            self.form.addWidget(QtWidgets.QLabel(f.replace(HDF, "") + ":"), row, 0)
            if self.mode == "search" and num:
                a, b = QtWidgets.QLineEdit(), QtWidgets.QLineEdit()
                self.form.addWidget(a, row, 1)
                self.form.addWidget(QtWidgets.QLabel("≤ value ≤"), row, 2)
                self.form.addWidget(b, row, 3)
                ws = (a, b)
            elif self.mode == "search":
                a = QtWidgets.QLineEdit()
                self.form.addWidget(a, row, 1, 1, 3)
                self.form.addWidget(QtWidgets.QLabel("is equal / matches"), row, 4)
                ws = (a,)
            elif num:
                a = QtWidgets.QLineEdit()
                a.setPlaceholderText("0: equal")
                self.form.addWidget(a, row, 1)
                self.form.addWidget(QtWidgets.QLabel("± tolerance"), row, 2)
                ws = (a,)
            else:
                lab = QtWidgets.QLabel("equal")
                self.form.addWidget(lab, row, 1)
                ws = ()
            for w, t in zip(ws, old.get(f, ())):
                w.setText(t)
            for w in ws:
                w.textChanged.connect(lambda _t: self.changed.emit())
            self.editors[f] = (ws, num)
        self.changed.emit()

    def set_values(self, field: str, *texts):
        self.check(field)
        for w, t in zip(self.editors[field][0], texts):
            w.setText(str(t))

    def clear_values(self):
        for ws, _ in self.editors.values():
            for w in ws:
                w.clear()

    def remove_all(self):
        for i in range(self.list.count()):
            self.list.item(i).setCheckState(QtCore.Qt.CheckState.Unchecked)

    def criteria(self, regex: bool) -> List[Criterion]:
        out = []
        for f, (ws, num) in self.editors.items():
            if num and len(ws) == 2:
                out.append(Criterion(f, "range", lo=_float(ws[0].text()), hi=_float(ws[1].text())))
            elif ws:
                out.append(Criterion(f, "match", text=ws[0].text(), regex=regex))
        return out

    def tolerances(self) -> Dict[str, Optional[float]]:
        out = {}
        for f, (ws, num) in self.editors.items():
            out[f] = _float(ws[0].text()) if ws else None
        return out


class AddManySamplesDialog(QtWidgets.QDialog):
    """Step 1: select runs; step 2: group them; the groups are returned by :meth:`groups`."""

    def __init__(self, entries, parent=None, check: Optional[Callable] = None, presets=None):
        """``presets``: [(item, values), ...] filled into the search page (e.g. the geometry of a tab)."""
        super().__init__(parent)
        self.setWindowTitle("Add many samples")
        self.meta = Metadata(entries)
        self.selected = []
        self._groups = []
        self.check_fn = check
        runs = [Metadata.raw(i).run for i in self.meta.items]
        lo, hi = (min(runs), max(runs)) if runs else (0, 0)
        lay = QtWidgets.QVBoxLayout(self)
        self.tabs = QtWidgets.QTabWidget()
        lay.addWidget(self.tabs, 1)

        # ---- step 1
        p1 = QtWidgets.QWidget()
        l1 = QtWidgets.QVBoxLayout(p1)
        info = QtWidgets.QLabel(f"{len(runs)} runs can be searched (available {lo} → {hi}).")
        l1.addWidget(info)
        rr = QtWidgets.QHBoxLayout()
        self.run_lo, self.run_hi = QtWidgets.QSpinBox(), QtWidgets.QSpinBox()
        for s, v in ((self.run_lo, lo), (self.run_hi, hi)):
            s.setRange(lo, max(hi, lo))
            s.setValue(v)
        allb = QtWidgets.QPushButton("All available")
        allb.clicked.connect(lambda: (self.run_lo.setValue(lo), self.run_hi.setValue(hi)))
        rr.addWidget(QtWidgets.QLabel("Runs"))
        rr.addWidget(self.run_lo)
        rr.addWidget(QtWidgets.QLabel("→"))
        rr.addWidget(self.run_hi)
        rr.addWidget(allb)
        rr.addStretch(1)
        l1.addLayout(rr)
        pr = QtWidgets.QHBoxLayout()
        self.pattern = QtWidgets.QCheckBox("Run pattern: starting with the first run, alternately use")
        self.use, self.skip = QtWidgets.QSpinBox(), QtWidgets.QSpinBox()
        self.use.setRange(1, 1000)
        self.skip.setRange(0, 1000)
        self.skip.setValue(1)
        pr.addWidget(self.pattern)
        pr.addWidget(self.use)
        pr.addWidget(QtWidgets.QLabel("runs and skip"))
        pr.addWidget(self.skip)
        pr.addWidget(QtWidgets.QLabel("runs"))
        pr.addStretch(1)
        l1.addLayout(pr)
        rx = QtWidgets.QHBoxLayout()
        rx.addWidget(QtWidgets.QLabel("Search the selected runs for these items:"))
        rx.addStretch(1)
        self.regex = QtWidgets.QCheckBox("Regular expressions")
        self.regex.setChecked(True)
        rx.addWidget(self.regex)
        l1.addLayout(rx)
        self.search = FieldPicker(self.meta, "search", checked=("sample", "kind"))
        self.search.set_values("kind", "scattering")
        for f, vals in presets or []:
            if f in self.meta.fields():
                self.search.set_values(f, *vals)
        l1.addWidget(self.search, 1)
        b1 = QtWidgets.QHBoxLayout()
        for text, fn in (("Remove all", self.search.remove_all), ("Clear all values", self.search.clear_values)):
            b = QtWidgets.QPushButton(text)
            b.clicked.connect(fn)
            b1.addWidget(b)
        b1.addStretch(1)
        go = QtWidgets.QPushButton("Search and accept →")
        go.clicked.connect(self.on_search)
        b1.addWidget(go)
        l1.addLayout(b1)
        self.tabs.addTab(p1, "1  Select runs")

        # ---- step 2
        p2 = QtWidgets.QWidget()
        l2 = QtWidgets.QHBoxLayout(p2)
        left = QtWidgets.QVBoxLayout()
        self.sel_list = QtWidgets.QListWidget()
        self.sel_count = QtWidgets.QLabel("0 runs")
        left.addWidget(QtWidgets.QLabel("Selected runs"))
        left.addWidget(self.sel_list, 1)
        left.addWidget(self.sel_count)
        l2.addLayout(left, 1)
        right = QtWidgets.QVBoxLayout()
        right.addWidget(QtWidgets.QLabel("Group by these items (± tolerance; text items: equal)"))
        self.group = FieldPicker(self.meta, "group", checked=("sample",))
        self.group.changed.connect(self.on_preview)
        right.addWidget(self.group, 2)
        self.preview = QtWidgets.QTableWidget(0, 3)
        self.preview.setHorizontalHeaderLabels(["group", "runs", "run numbers"])
        self.preview.horizontalHeader().setStretchLastSection(True)
        self.preview.verticalHeader().setVisible(False)
        right.addWidget(self.preview, 2)
        self.do_check = QtWidgets.QCheckBox("check the repeated runs of every group (outliers off)")
        self.do_check.setChecked(check is not None)
        self.do_check.setEnabled(check is not None)
        right.addWidget(self.do_check)
        l2.addLayout(right, 3)
        self.tabs.addTab(p2, "2  Group")

        bb = QtWidgets.QDialogButtonBox()
        self.ok = bb.addButton("Add as samples", QtWidgets.QDialogButtonBox.ButtonRole.AcceptRole)
        bb.addButton(QtWidgets.QDialogButtonBox.StandardButton.Cancel)
        bb.accepted.connect(self.accept)
        bb.rejected.connect(self.reject)
        lay.addWidget(bb)
        self.resize(950, 650)

    # ------------------------------------------------------------------ step 1
    def on_search(self):
        try:
            crit = self.search.criteria(self.regex.isChecked())
        except ValueError as e:
            QtWidgets.QMessageBox.warning(self, "Search", f"not a number: {e}")
            return
        pattern = (self.use.value(), self.skip.value()) if self.pattern.isChecked() else None
        self.selected = select_runs(self.meta, (self.run_lo.value(), self.run_hi.value()), pattern, crit)
        self.sel_list.clear()
        for it in self.selected:
            r = Metadata.raw(it)
            self.sel_list.addItem(f"{r.run}  {r.sample_name}")
        self.sel_count.setText(f"{len(self.selected)} runs")
        self.tabs.setCurrentIndex(1)
        self.on_preview()

    # ------------------------------------------------------------------ step 2
    def on_preview(self):
        if not hasattr(self, "preview"):
            return
        try:
            self._groups = group_selected(self.meta, self.selected, self.group.tolerances()) if self.selected else []
        except (ValueError, KeyError) as e:
            self._groups = []
            self.sel_count.setText(str(e))
        from .plan import _runs
        self.preview.setRowCount(len(self._groups))
        for i, g in enumerate(self._groups):
            for j, v in enumerate((g.label(), str(len(g.runs)), _runs(g.run_numbers))):
                self.preview.setItem(i, j, QtWidgets.QTableWidgetItem(v))
        self.preview.resizeColumnsToContents()
        self.ok.setText(f"Add {len(self._groups)} samples")

    def groups(self):
        return list(self._groups)

    def wants_check(self) -> bool:
        return self.do_check.isChecked()
