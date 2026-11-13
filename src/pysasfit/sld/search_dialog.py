"""Search dialog of pySLDcalculator: materials by name from local tables and online databases
(:mod:`pysasfit.sld.lookup`). The search runs in a thread; a double click (or *Add*) puts a hit into the
list of materials."""
from __future__ import annotations

from PySide6 import QtCore, QtGui, QtWidgets

from . import lookup

__all__ = ["SearchDialog"]

SOURCE_COLOURS = {"builtin": "#555555", "xraydb": "#1f77b4", "orso": "#2ca02c", "wikipedia": "#9467bd",
                  "pubchem": "#d62728"}


class _Worker(QtCore.QThread):
    done = QtCore.Signal(int, str, list, dict)           # ticket, source, hits, errors

    def __init__(self, ticket, query, sources, limit, parent=None):
        super().__init__(parent)
        self.ticket, self.query, self.sources, self.limit = ticket, query, sources, limit

    def run(self):
        for src in self.sources:                         # one source after the other: results appear as they come
            if self.isInterruptionRequested():
                return
            hits, errors = lookup.search(self.query, (src,), self.limit)
            self.done.emit(self.ticket, src, hits, errors)


class SearchDialog(QtWidgets.QDialog):
    """``material_chosen(dict)`` is emitted with the entry of the hit to add."""
    material_chosen = QtCore.Signal(dict)

    def __init__(self, parent=None, sources=None):
        super().__init__(parent)
        self.setWindowTitle("Search material")
        self.resize(980, 520)
        self.hits = []
        self._ticket = 0
        self._workers = []
        lay = QtWidgets.QVBoxLayout(self)
        h = QtWidgets.QHBoxLayout()
        self.query = QtWidgets.QLineEdit()
        self.query.setPlaceholderText("name of a material or compound, e.g. toluene, PMMA, magnetite, sapphire")
        self.query.returnPressed.connect(self.start)
        self.button = QtWidgets.QPushButton("Search")
        self.button.clicked.connect(self.start)
        h.addWidget(self.query, 1)
        h.addWidget(self.button)
        lay.addLayout(h)
        h = QtWidgets.QHBoxLayout()
        h.addWidget(QtWidgets.QLabel("in"))
        self.boxes = {}
        for key, label in lookup.SOURCES.items():
            b = QtWidgets.QCheckBox(label + (" (online)" if key in lookup.ONLINE else ""))
            b.setChecked(sources is None or key in sources)
            self.boxes[key] = b
            h.addWidget(b)
        h.addStretch(1)
        lay.addLayout(h)
        self.boxes["orso"].setToolTip("slddb.reflectometry.org: materials collected for reflectometry and SAS, with "
                                      "reference and validation flag")
        self.boxes["wikipedia"].setToolTip("Formula and density of the infobox of the English Wikipedia articles "
                                           "found for the name")
        self.boxes["pubchem"].setToolTip("Formula of a named compound and its experimental densities (handbook "
                                         "values with temperature); exact compound names work best")
        self.table = QtWidgets.QTableWidget(0, 5)
        self.table.setHorizontalHeaderLabels(["name", "formula", "density (g/cm³)", "source", "note"])
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.verticalHeader().setVisible(False)
        self.table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.SingleSelection)
        self.table.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.doubleClicked.connect(lambda *_: self.add_selected())
        lay.addWidget(self.table, 1)
        self.status = QtWidgets.QLabel("Densities from Wikipedia and PubChem are read from free text (see the note): "
                                       "check them. A hit without density is added with density 0.")
        self.status.setWordWrap(True)
        lay.addWidget(self.status)
        h = QtWidgets.QHBoxLayout()
        self.add = QtWidgets.QPushButton("Add to materials")
        self.add.clicked.connect(self.add_selected)
        self.open = QtWidgets.QPushButton("Open web page")
        self.open.clicked.connect(self.open_page)
        close = QtWidgets.QPushButton("Close")
        close.clicked.connect(self.close)
        h.addWidget(self.add)
        h.addWidget(self.open)
        h.addStretch(1)
        h.addWidget(close)
        lay.addLayout(h)

    def sources(self):
        return [k for k, b in self.boxes.items() if b.isChecked()]

    def start(self):
        q = self.query.text().strip()
        srcs = self.sources()
        if not q or not srcs:
            return
        self._ticket += 1
        self.hits = []
        self._errors = {}
        self._pending = set(srcs)
        self.table.setRowCount(0)
        self._show_status()
        w = _Worker(self._ticket, q, srcs, 12, self)
        w.done.connect(self._on_done)
        w.finished.connect(lambda w=w: self._workers.remove(w) if w in self._workers else None)
        self._workers.append(w)
        w.start()

    def _on_done(self, ticket, source, hits, errors):
        if ticket != self._ticket:                       # answer of an older search
            return
        self._pending.discard(source)
        self._errors.update(errors)
        for hit in hits:
            self._add_row(hit)
        self._show_status()

    def _add_row(self, hit):
        self.hits.append(hit)
        r = self.table.rowCount()
        self.table.insertRow(r)
        vals = [hit.name, hit.formula, f"{hit.density:.5g}" if hit.density else
                (f"V = {hit.volume:.5g} Å³" if hit.volume else "–"), lookup.SOURCES.get(hit.source, hit.source),
                hit.note]
        for j, s in enumerate(vals):
            it = QtWidgets.QTableWidgetItem(s)
            if j == 3:
                it.setForeground(QtGui.QBrush(QtGui.QColor(SOURCE_COLOURS.get(hit.source, "#000000"))))
            if j == 4:
                it.setToolTip(hit.note + ("\n" + hit.url if hit.url else ""))
            self.table.setItem(r, j, it)
        if r == 0:
            self.table.selectRow(0)
        self.table.resizeColumnsToContents()
        self.table.setColumnWidth(0, min(self.table.columnWidth(0), 260))
        self.table.setColumnWidth(1, min(self.table.columnWidth(1), 220))

    def _show_status(self):
        parts = [f"{len(self.hits)} hit{'s' if len(self.hits) != 1 else ''}"]
        if self._pending:
            parts.append("searching " + ", ".join(lookup.SOURCES[s] for s in lookup.SOURCES if s in self._pending)
                         + " …")
        for s, msg in self._errors.items():
            parts.append(f"{lookup.SOURCES.get(s, s)}: {msg}")
        self.status.setText(";  ".join(parts))

    def selected(self):
        r = self.table.currentRow()
        return self.hits[r] if 0 <= r < len(self.hits) else None

    def add_selected(self):
        hit = self.selected()
        if hit is not None and hit.formula:
            self.material_chosen.emit(lookup.hit_entry(hit))

    def open_page(self):
        hit = self.selected()
        if hit is not None and hit.url:
            QtGui.QDesktopServices.openUrl(QtCore.QUrl(hit.url))

    def wait(self):
        """Wait for the running searches (tests)."""
        for w in list(self._workers):
            w.wait()
        QtWidgets.QApplication.processEvents()

    def closeEvent(self, event):
        self._ticket += 1
        for w in list(self._workers):
            w.requestInterruption()
            w.wait(15000)
        super().closeEvent(event)
