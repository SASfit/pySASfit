"""pySLDcalculator window: scattering length densities for neutrons and X-rays.

    pysasfit-sld
    python -m pysasfit.sld

Also the "pySLDcalculator" tab of the pySASfit launcher.

Search…  (Ctrl+F) looks a material up by name in the built-in library, the xraydb table and, online, the ORSO
         SLD database, Wikipedia and PubChem (:mod:`pysasfit.sld.lookup`)
Left     the list of materials (formula or protein / DNA / RNA sequence, with mass density or volume of the
         formula unit), the editor of the selected one, the radiation (neutron wavelength; X-ray energy,
         wavelength or K-alpha source; Chantler or Henke table) and the matrix: vacuum, an H2O/D2O mixture
         or one of the materials
Right    Properties      every quantity of every material side by side, and the contrast to the matrix:
                         Δρ, (Δρ)², excess scattering length, I(0)/c, match point
         Elements        the atoms of the selected material: b_c, cross sections, f1, f2
         D2O contrast    neutron SLD of the materials and of the solvent against the D2O fraction, match points
         X-ray energy    SLD (real, imaginary) or f1, f2 of the selected material across absorption edges
         Residues        the residue table used for sequences

The values follow the inputs at once; nothing has to be started.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg  # noqa: E402
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402

from ..core import settings as prefs  # noqa: E402
from ..resources import tool_icon  # noqa: E402
from . import entries as E  # noqa: E402
from .core import HC_KEV_A, XRAY_SOURCES, Exchange, Probe, d2o_scan, energy_scan, have_xraydb  # noqa: E402
from .core import match_point, water, xray_source_energy  # noqa: E402
from .library import LIBRARY  # noqa: E402

__all__ = ["MainWindow", "main"]

KINDS = [("formula", "formula"), ("aa", "protein sequence"), ("dna", "DNA sequence"), ("rna", "RNA sequence")]
SIZE_MODES = [("density", "mass density (g/cm³)"), ("volume", "volume of the formula unit (Å³)"),
              ("residues", "sum of the residue volumes")]
COLOURS = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e", "#8c564b", "#17becf", "#7f7f7f"]

FORMULA_HELP = (
    "Chemical formula (grammar of periodictable / the NIST calculator):\n"
    "  SiO2, C6H5CH3, Fe3O4, Pd1.3Si, CaCO3 + 6H2O\n"
    "  isotopes:  D2O, C7D8, Ca[44]CO3, O[18]\n"
    "  labile (exchangeable) hydrogen:  H[1], e.g. C2H5OH[1] or C3H4H[1]NO\n"
    "  density in the formula:  SiO2@2.2\n"
    "  mixtures:  3%wt NaCl // H2O   (give the density of the mixture)\n"
    "Sequences: one-letter codes; chains separated by '/' or new lines; FASTA headers are ignored.")


def _session_path():
    return os.path.join(os.path.dirname(prefs.settings_path()), "sld_session.json")


def _num(text, default=None):
    try:
        return float(str(text).replace(",", "."))
    except (TypeError, ValueError):
        return default


def _spin(lo, hi, value, step=0.05, decimals=3):
    s = QtWidgets.QDoubleSpinBox()
    s.setRange(lo, hi)
    s.setDecimals(decimals)
    s.setSingleStep(step)
    s.setValue(value)
    s.setKeyboardTracking(False)
    return s


class _Plot(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.fig = Figure(constrained_layout=True)
        self.canvas = FigureCanvasQTAgg(self.fig)
        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        self.top = QtWidgets.QHBoxLayout()
        lay.addLayout(self.top)
        lay.addWidget(self.canvas, 1)
        lay.addWidget(NavigationToolbar2QT(self.canvas, self))


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("pySLDcalculator – scattering length densities for neutrons and X-rays")
        self.setWindowIcon(tool_icon("pySLDcalculator"))
        self.entries = [dict(e) for e in E.DEFAULT_ENTRIES]
        self._loading = False
        self._rows = []
        self._columns = []
        self._timer = QtCore.QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.setInterval(250)
        self._timer.timeout.connect(self._editor_changed)

        split = QtWidgets.QSplitter()
        # both halves scroll: their controls must not fix the smallest size of the window (and, in the
        # launcher, of the whole pySASfit window)
        from ..core.scrolling import scrolled
        split.addWidget(scrolled(self._build_left()))
        split.addWidget(scrolled(self._build_right()))
        split.setStretchFactor(0, 0)
        split.setStretchFactor(1, 1)
        split.setSizes([430, 970])
        self.split = split
        self.setCentralWidget(split)

        m = self.menuBar().addMenu("&File")
        m.addAction("&New material", self.add_material, QtGui.QKeySequence.StandardKey.New)
        m.addAction("Search &material…", self.search_material, QtGui.QKeySequence.StandardKey.Find)
        m.addAction("&Open materials…", self.on_open)
        m.addAction("&Save materials…", self.on_save)
        m.addSeparator()
        m.addAction("Copy &table", self.copy_table)
        m.addAction("&Export table (CSV)…", self.export_table)
        from ..core import pycode
        pycode.install(self, self.python_code, menu=m)
        from ..core.settings_dialog import add_preferences_menu
        add_preferences_menu(self, None)
        self.resize(1400, 900)
        self._fill_list()
        self.list.setCurrentCell(0, 0)
        self.recalc()

    # ------------------------------------------------------------------ left side
    def _build_left(self):
        w = QtWidgets.QWidget()
        lay = QtWidgets.QVBoxLayout(w)

        # materials
        box = QtWidgets.QGroupBox("Materials")
        v = QtWidgets.QVBoxLayout(box)
        self.list = QtWidgets.QTableWidget(0, 3)
        self.list.setHorizontalHeaderLabels(["name", "formula / sequence", "ρ or V"])
        self.list.horizontalHeader().setStretchLastSection(True)
        self.list.horizontalHeader().setSectionResizeMode(1, QtWidgets.QHeaderView.ResizeMode.Stretch)
        self.list.verticalHeader().setVisible(False)
        self.list.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows)
        self.list.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.SingleSelection)
        self.list.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
        self.list.currentCellChanged.connect(lambda *a: self._show_entry())
        self.list.setMinimumHeight(130)
        v.addWidget(self.list)
        h = QtWidgets.QHBoxLayout()
        for text, fn in (("New", self.add_material), ("Duplicate", self.duplicate_material),
                         ("Remove", self.remove_material)):
            b = QtWidgets.QPushButton(text)
            b.clicked.connect(fn)
            h.addWidget(b)
        lib = QtWidgets.QPushButton("Library")
        lib.setToolTip("Add a common material (typical densities near 20 °C; check the density of your sample)")
        menu = QtWidgets.QMenu(lib)
        for group, items in LIBRARY.items():
            sub = menu.addMenu(group)
            for name, formula, rho in items:
                sub.addAction(f"{name}   {formula}   {rho:g} g/cm³",
                              lambda n=name, f=formula, r=rho: self.add_material(E.new_entry(n, f, size=r)))
        lib.setMenu(menu)
        h.addWidget(lib)
        find = QtWidgets.QPushButton("Search…")
        find.setToolTip("Search a material by name: built-in library, xraydb, ORSO SLD database, Wikipedia, PubChem "
                        "(Ctrl+F)")
        find.clicked.connect(self.search_material)
        h.addWidget(find)
        v.addLayout(h)
        lay.addWidget(box, 1)

        # editor
        box = QtWidgets.QGroupBox("Selected material")
        f = QtWidgets.QFormLayout(box)
        self.name = QtWidgets.QLineEdit()
        self.kind = QtWidgets.QComboBox()
        for key, text in KINDS:
            self.kind.addItem(text, key)
        self.text = QtWidgets.QPlainTextEdit()
        self.text.setToolTip(FORMULA_HELP)
        self.text.setFixedHeight(84)
        self.text.setFont(QtGui.QFontDatabase.systemFont(QtGui.QFontDatabase.SystemFont.FixedFont))
        self.size_mode = QtWidgets.QComboBox()
        for key, text in SIZE_MODES:
            self.size_mode.addItem(text, key)
        self.size = QtWidgets.QLineEdit()
        self.size.setToolTip("Mass density of the formula as written (labile H as H, before deuteration), "
                             "or the volume of one formula unit.\nFor a sequence with a known partial specific "
                             "volume v̄ use the density 1/v̄.")
        self.deut = _spin(0, 1, 0)
        self.deut.setToolTip("Fraction of the non-exchangeable hydrogen (written H) that is deuterium, e.g. a "
                             "protein expressed in D₂O medium.\nThe volume of the formula unit is kept.")
        self.moment = QtWidgets.QLineEdit("0")
        self.moment.setToolTip("Magnetic moment per formula unit in Bohr magnetons (for the magnetic SLD; 0: none)")
        hs = QtWidgets.QHBoxLayout()
        hs.addWidget(self.size_mode, 1)
        hs.addWidget(self.size)
        self.natural = QtWidgets.QCheckBox("density is that of the protonated material")
        self.natural.setToolTip("The density belongs to the compound with natural isotopes, e.g. the density of "
                                "C7H8 (0.867) for the formula C7D8.\nThe volume of the formula unit is then that of "
                                "the protonated compound; the density of the deuterated one follows from its mass "
                                "(shown in the table).\nWithout it a deuterated formula with a protonated density "
                                "gives an SLD that is too low.")
        f.addRow("name", self.name)
        f.addRow("type", self.kind)
        f.addRow("formula", self.text)
        f.addRow("size", hs)
        f.addRow("", self.natural)
        f.addRow("deuteration", self.deut)
        f.addRow("moment (μ_B)", self.moment)
        self.error = QtWidgets.QLabel()
        self.error.setWordWrap(True)
        self.error.setStyleSheet("color: #c0392b;")
        f.addRow(self.error)
        self.name.editingFinished.connect(self._editor_changed)
        self.size.editingFinished.connect(self._editor_changed)
        self.moment.editingFinished.connect(self._editor_changed)
        self.text.textChanged.connect(self._timer.start)
        self.kind.currentIndexChanged.connect(self._kind_changed)
        self.size_mode.currentIndexChanged.connect(self._editor_changed)
        self.deut.valueChanged.connect(self._editor_changed)
        self.natural.toggled.connect(self._editor_changed)
        lay.addWidget(box)

        # radiation
        box = QtWidgets.QGroupBox("Radiation")
        f = QtWidgets.QFormLayout(box)
        self.n_wavelength = _spin(0.1, 100, 6.0, 0.5, 3)
        self.n_wavelength.setSuffix(" Å")
        f.addRow("neutron wavelength", self.n_wavelength)
        self.x_unit = QtWidgets.QComboBox()
        self.x_unit.addItems(["energy (keV)", "wavelength (Å)"])
        self.x_value = QtWidgets.QLineEdit()
        self.x_source = QtWidgets.QComboBox()
        self.x_source.addItem("source …")
        for s in XRAY_SOURCES:
            self.x_source.addItem(f"{s} Kα", s)
        hx = QtWidgets.QHBoxLayout()
        hx.addWidget(self.x_unit)
        hx.addWidget(self.x_value, 1)
        hx.addWidget(self.x_source)
        f.addRow("X-rays", hx)
        self.x_table = QtWidgets.QComboBox()
        self.x_table.addItem("Chantler (xraydb)", "chantler")
        self.x_table.addItem("Henke / CXRO (30 eV – 30 keV)", "henke")
        self.x_table.setToolTip("Tables of f′, f″ of free atoms. Chantler: fine energy grid at the edges, Z ≤ 98 "
                                "(needs the package xraydb).\nClose to an edge the true values depend on the "
                                "chemical state.")
        f.addRow("f′, f″ table", self.x_table)
        self._energy = xray_source_energy("Cu")
        self._show_energy()
        self.n_wavelength.valueChanged.connect(self.recalc)
        self.x_value.editingFinished.connect(self._energy_edited)
        self.x_unit.currentIndexChanged.connect(self._show_energy)
        self.x_source.activated.connect(self._source_chosen)
        self.x_table.currentIndexChanged.connect(self.recalc)
        lay.addWidget(box)

        # matrix
        box = QtWidgets.QGroupBox("Matrix / solvent and hydrogen exchange")
        f = QtWidgets.QFormLayout(box)
        self.matrix = QtWidgets.QComboBox()
        self.matrix.setToolTip("Reference of the contrast and of the excess scattering length Δb = b − ρ₀V")
        f.addRow("matrix", self.matrix)
        self.d2o = _spin(0, 1, 0, 0.05, 3)
        self.d2o.setToolTip("D₂O fraction of the water: SLD of the H₂O/D₂O mixture and exchange of the labile "
                            "hydrogen (H[1], and the N-H, O-H, S-H of sequences)")
        f.addRow("D₂O fraction", self.d2o)
        self.ex_backbone = _spin(0, 1, 0.9)
        self.ex_backbone.setToolTip("Fraction of the backbone amide N-H of proteins that exchanges (0.9: Perkins "
                                    "1986, CRYSON)")
        self.ex_side = _spin(0, 1, 1.0)
        self.ex_side.setToolTip("Fraction of all other labile hydrogen that exchanges (side chains, chain ends, "
                                "nucleic acids, H[1] of formulas)")
        he = QtWidgets.QHBoxLayout()
        he.addWidget(QtWidgets.QLabel("backbone N-H"))
        he.addWidget(self.ex_backbone)
        he.addWidget(QtWidgets.QLabel("other"))
        he.addWidget(self.ex_side)
        f.addRow("exchange", he)
        self.thickness = _spin(0.0001, 100, 0.1, 0.05, 4)
        self.thickness.setSuffix(" cm")
        f.addRow("thickness", self.thickness)
        for s in (self.d2o, self.ex_backbone, self.ex_side, self.thickness):
            s.valueChanged.connect(self.recalc)
        self.matrix.currentIndexChanged.connect(self.recalc)
        lay.addWidget(box)
        return w

    # ------------------------------------------------------------------ right side
    def _build_right(self):
        self.tabs = QtWidgets.QTabWidget()
        mono = QtGui.QFontDatabase.systemFont(QtGui.QFontDatabase.SystemFont.FixedFont)

        self.table = QtWidgets.QTableWidget()
        self.table.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.verticalHeader().setVisible(False)
        self.table.setAlternatingRowColors(True)
        QtGui.QShortcut(QtGui.QKeySequence.StandardKey.Copy, self.table, self.copy_selection,
                        context=QtCore.Qt.ShortcutContext.WidgetShortcut)
        self.tabs.addTab(self.table, "Properties")

        self.elements = QtWidgets.QTableWidget()
        self.elements.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
        self.elements.verticalHeader().setVisible(False)
        self.tabs.addTab(self.elements, "Elements")

        self.d2o_plot = _Plot()
        self.tabs.addTab(self.d2o_plot, "D₂O contrast")

        self.e_plot = _Plot()
        self.e_min = QtWidgets.QLineEdit("5")
        self.e_max = QtWidgets.QLineEdit("25")
        self.e_n = QtWidgets.QSpinBox()
        self.e_n.setRange(20, 20000)
        self.e_n.setValue(1500)
        self.e_what = QtWidgets.QComboBox()
        self.e_what.addItems(["SLD", "f₁, f₂ per formula unit", "contrast (Δρ)² to the matrix"])
        t = self.e_plot.top
        for lab, wid in (("E from", self.e_min), ("to", self.e_max), ("keV, points", self.e_n), ("show", self.e_what)):
            t.addWidget(QtWidgets.QLabel(lab))
            t.addWidget(wid)
        t.addStretch(1)
        self.e_min.setMaximumWidth(70)
        self.e_max.setMaximumWidth(70)
        self.e_min.editingFinished.connect(self._plot_energy)
        self.e_max.editingFinished.connect(self._plot_energy)
        self.e_n.editingFinished.connect(self._plot_energy)
        self.e_what.currentIndexChanged.connect(self._plot_energy)
        self.tabs.addTab(self.e_plot, "X-ray energy")

        w = QtWidgets.QWidget()
        v = QtWidgets.QVBoxLayout(w)
        h = QtWidgets.QHBoxLayout()
        self.res_kind = QtWidgets.QComboBox()
        for key, text in (("aa", "amino acids"), ("dna", "DNA nucleotides"), ("rna", "RNA nucleotides")):
            self.res_kind.addItem(text, key)
        self.res_kind.currentIndexChanged.connect(self._fill_residues)
        h.addWidget(self.res_kind)
        lab = QtWidgets.QLabel("residue in a chain; SLD with all labile H exchanged; volumes and formulas of "
                               "periodictable (Perkins 1988, Jacrot 1976; nucleotides Buckin 1989)")
        lab.setWordWrap(True)
        h.addWidget(lab, 1)
        v.addLayout(h)
        self.residues = QtWidgets.QTableWidget()
        self.residues.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
        self.residues.verticalHeader().setVisible(False)
        self.residues.setFont(mono)
        v.addWidget(self.residues)
        self.tabs.addTab(w, "Residues")
        self.tabs.currentChanged.connect(self._tab_changed)
        return self.tabs

    # ------------------------------------------------------------------ settings
    def probe(self):
        return Probe(self.n_wavelength.value(), self._energy, self.x_table.currentData())

    def exchange(self):
        return Exchange(self.ex_backbone.value(), self.ex_side.value())

    def matrix_key(self):
        d = self.matrix.currentData()
        return None if d == "vacuum" else d

    def _show_energy(self):
        v = self._energy if self.x_unit.currentIndex() == 0 else HC_KEV_A / self._energy
        self.x_value.setText(f"{v:.6g}")

    def _energy_edited(self):
        v = _num(self.x_value.text())
        if v is None or v <= 0:
            self._show_energy()
            return
        e = v if self.x_unit.currentIndex() == 0 else HC_KEV_A / v
        if abs(e - self._energy) > 1e-12:
            self._energy = e
            self.x_source.setCurrentIndex(0)
            self.recalc()

    def _source_chosen(self, i):
        s = self.x_source.itemData(i)
        if s:
            self._energy = xray_source_energy(s)
            self._show_energy()
            self.recalc()

    # ------------------------------------------------------------------ materials
    def current(self):
        r = self.list.currentRow()
        return r if 0 <= r < len(self.entries) else None

    def _fill_list(self):
        self._loading = True
        row = self.list.currentRow()
        self.list.setRowCount(len(self.entries))
        for i, e in enumerate(self.entries):
            text = " ".join(str(e.get("text", "")).split())
            mode = e.get("size_mode")
            size = "residues" if mode == "residues" else \
                f"{_num(e.get('size'), 0):g} {'g/cm³' if mode == 'density' else 'Å³'}" \
                + (" (protonated)" if mode == "density" and e.get("natural_density") else "")
            for j, s in enumerate((e.get("name", ""), text, size)):
                it = QtWidgets.QTableWidgetItem(s)
                it.setForeground(QtGui.QBrush(QtGui.QColor(COLOURS[i % len(COLOURS)])) if j == 0
                                 else it.foreground())
                self.list.setItem(i, j, it)
        self.list.setCurrentCell(min(max(row, 0), len(self.entries) - 1), 0)
        # matrix choices
        cur = self.matrix.currentData() if self.matrix.count() else "water"
        self.matrix.blockSignals(True)
        self.matrix.clear()
        self.matrix.addItem("H₂O / D₂O mixture", "water")
        self.matrix.addItem("vacuum", "vacuum")
        for i, e in enumerate(self.entries):
            self.matrix.addItem(e.get("name") or f"material {i + 1}", i)
        k = self.matrix.findData(cur)
        self.matrix.setCurrentIndex(k if k >= 0 else 0)
        self.matrix.blockSignals(False)
        self._loading = False

    def _show_entry(self):
        if self._loading:
            return
        i = self.current()
        if i is None:
            return
        e = self.entries[i]
        self._loading = True
        self.name.setText(e.get("name", ""))
        self.kind.setCurrentIndex(max(self.kind.findData(e.get("kind", "formula")), 0))
        self.text.setPlainText(str(e.get("text", "")))
        self.size_mode.setCurrentIndex(max(self.size_mode.findData(e.get("size_mode", "density")), 0))
        self.size.setText(f"{_num(e.get('size'), 0):g}")
        self.deut.setValue(_num(e.get("deuteration"), 0.0))
        self.moment.setText(f"{_num(e.get('moment'), 0):g}")
        self.natural.setChecked(bool(e.get("natural_density")))
        self._loading = False
        self._enable_editor()
        self._fill_elements()
        self._replot()

    def _enable_editor(self):
        seq = self.kind.currentData() != "formula"
        self.size.setEnabled(self.size_mode.currentData() != "residues")
        item = self.size_mode.model().item(self.size_mode.findData("residues"))
        item.setEnabled(seq)
        self.moment.setEnabled(not seq)
        self.natural.setEnabled(not seq and self.size_mode.currentData() == "density")

    def _kind_changed(self):
        if self._loading:
            return
        seq = self.kind.currentData() != "formula"
        self._loading = True
        if seq:
            self.size_mode.setCurrentIndex(self.size_mode.findData("residues"))
        elif self.size_mode.currentData() == "residues":
            self.size_mode.setCurrentIndex(self.size_mode.findData("density"))
        self._loading = False
        self._editor_changed()

    def _editor_changed(self):
        if self._loading:
            return
        i = self.current()
        if i is None:
            return
        e = self.entries[i]
        new = {"name": self.name.text().strip() or f"material {i + 1}", "kind": self.kind.currentData(),
               "text": self.text.toPlainText().strip(), "size_mode": self.size_mode.currentData(),
               "size": _num(self.size.text(), 0.0), "deuteration": self.deut.value(),
               "moment": _num(self.moment.text(), 0.0)}
        self._enable_editor()
        new["natural_density"] = self.natural.isChecked() and self.natural.isEnabled()
        if new == e:
            return
        self.entries[i] = new
        self._fill_list()
        self.recalc()

    def add_material(self, entry=None):
        if not isinstance(entry, dict):
            entry = E.new_entry(f"material {len(self.entries) + 1}", "H2O", size=1.0)
        self.entries.append(dict(entry))
        self._fill_list()
        self.list.setCurrentCell(len(self.entries) - 1, 0)
        self.recalc()

    def search_material(self):
        """Search dialog (not modal: several hits can be added one after the other)."""
        from .search_dialog import SearchDialog
        dlg = getattr(self, "_search", None)
        if dlg is None:
            dlg = self._search = SearchDialog(self, getattr(self, "_search_sources", None))
            dlg.material_chosen.connect(self.add_material)
        dlg.show()
        dlg.raise_()
        dlg.activateWindow()
        dlg.query.setFocus()
        dlg.query.selectAll()
        return dlg

    def duplicate_material(self):
        i = self.current()
        if i is not None:
            e = dict(self.entries[i])
            e["name"] = e.get("name", "") + " (copy)"
            self.add_material(e)

    def remove_material(self):
        i = self.current()
        if i is None or len(self.entries) <= 1:
            return
        m = self.matrix.currentData()
        del self.entries[i]
        if isinstance(m, int):                      # keep the matrix on the same material
            self.matrix.blockSignals(True)
            self.matrix.clear()
            self.matrix.blockSignals(False)
            self._fill_list()
            k = self.matrix.findData(m - 1 if m > i else m) if m != i else 0
            self.matrix.setCurrentIndex(max(k, 0))
        else:
            self._fill_list()
        self._show_entry()
        self.recalc()

    # ------------------------------------------------------------------ calculation
    def recalc(self, *args):
        if self._loading:
            return
        try:
            rows, columns, errors = E.result_table(self.entries, self.matrix_key(), self.probe(), self.d2o.value(),
                                                   self.exchange(), self.thickness.value())
        except ImportError as exc:
            self.error.setText(str(exc))
            return
        self._rows, self._columns, self._errors = rows, columns, errors
        i = self.current()
        msg = errors.get(i) or errors.get("matrix", "")
        if not msg and self.x_table.currentData() == "chantler" and not have_xraydb():
            msg = "xraydb is not installed: the Henke table is used (pip install xraydb)"
        self.error.setText(msg)
        self._fill_table()
        self._fill_elements()
        self._replot()
        bad = [self._columns[k] for k in errors if isinstance(k, int) and k < len(self._columns)]
        self.statusBar().showMessage("not calculated: " + ", ".join(bad) if bad else "", 15000)

    def _fill_table(self):
        t = self.table
        rows, columns = self._rows, self._columns
        t.clear()
        t.setColumnCount(len(columns) + 2)
        t.setHorizontalHeaderLabels(["quantity", *columns, "unit"])
        n = len(rows) + len({r[0] for r in rows})
        t.setRowCount(n)
        bold = t.font()
        bold.setBold(True)
        group = None
        k = 0
        head = self.palette().color(QtGui.QPalette.ColorRole.Midlight)
        for g, q, u, vals in rows:
            if g != group:
                group = g
                it = QtWidgets.QTableWidgetItem(g)
                it.setFont(bold)
                it.setBackground(head)
                t.setItem(k, 0, it)
                t.setSpan(k, 0, 1, len(columns) + 2)
                k += 1
            t.setItem(k, 0, QtWidgets.QTableWidgetItem(q))
            for j, v in enumerate(vals):
                it = QtWidgets.QTableWidgetItem(E.fmt(v))
                it.setTextAlignment(QtCore.Qt.AlignmentFlag.AlignRight | QtCore.Qt.AlignmentFlag.AlignVCenter)
                t.setItem(k, j + 1, it)
            t.setItem(k, len(columns) + 1, QtWidgets.QTableWidgetItem(u))
            k += 1
        t.resizeColumnsToContents()
        for j in range(1, len(columns) + 1):
            t.setColumnWidth(j, max(t.columnWidth(j), 110))

    def _material(self, i=None):
        i = self.current() if i is None else i
        if i is None:
            return None
        try:
            return E.material(self.entries[i])
        except ValueError:
            return None

    def _fill_elements(self):
        t = self.elements
        heads = ["atom", "Z", "n", "mass fraction", "b_c (fm)", "b_c imag (fm)", "σ_inc (barn)",
                 "σ_abs (barn) at λ", "f₁ = Z + f′", "f₂ = f″"]
        t.clear()
        t.setColumnCount(len(heads))
        t.setHorizontalHeaderLabels(heads)
        m = self._material()
        if m is None:
            t.setRowCount(0)
            return
        try:
            p = m.properties(self.probe(), self.d2o.value(), self.exchange(), self.thickness.value())
        except Exception:  # noqa: BLE001
            t.setRowCount(0)
            return
        t.setRowCount(len(p.elements))
        for r, el in enumerate(p.elements):
            vals = [el["atom"], f"{el['Z']}", E.fmt(el["n"]), E.fmt(el["mass_fraction"], 5), E.fmt(el["b_c"]),
                    E.fmt(el["b_c_imag"]), E.fmt(el["sigma_inc"]), E.fmt(el["sigma_abs"]), E.fmt(el["f1"]),
                    E.fmt(el["f2"])]
            for j, s in enumerate(vals):
                it = QtWidgets.QTableWidgetItem(s)
                if j:
                    it.setTextAlignment(QtCore.Qt.AlignmentFlag.AlignRight | QtCore.Qt.AlignmentFlag.AlignVCenter)
                t.setItem(r, j, it)
        t.resizeColumnsToContents()

    def _fill_residues(self):
        from .bio import residue_table
        t = self.residues
        heads = ["code", "name", "formula", "V (Å³)", "labile H", "SLD in H₂O", "SLD in D₂O", "match (% D₂O)"]
        try:
            rows = residue_table(self.res_kind.currentData(), self.probe())
        except Exception as exc:  # noqa: BLE001
            self.statusBar().showMessage(str(exc), 10000)
            return
        t.clear()
        t.setColumnCount(len(heads))
        t.setHorizontalHeaderLabels(heads)
        t.setRowCount(len(rows))
        for r, d in enumerate(rows):
            vals = [d["code"], d["name"], d["formula"], f"{d['volume']:.1f}", f"{d['n_labile']:g}",
                    f"{d['sld_h2o']:.3f}", f"{d['sld_d2o']:.3f}", f"{100 * d['match']:.1f}"]
            for j, s in enumerate(vals):
                it = QtWidgets.QTableWidgetItem(s)
                if j > 2:
                    it.setTextAlignment(QtCore.Qt.AlignmentFlag.AlignRight | QtCore.Qt.AlignmentFlag.AlignVCenter)
                t.setItem(r, j, it)
        t.resizeColumnsToContents()

    # ------------------------------------------------------------------ plots
    def _tab_changed(self, i):
        if self.tabs.widget(i) is not self.table and self.tabs.tabText(i) == "Residues" \
                and not self.residues.rowCount():
            self._fill_residues()
        self._replot()

    def _replot(self):
        if self._loading:
            return
        w = self.tabs.currentWidget()
        if w is self.d2o_plot:
            self._plot_d2o()
        elif w is self.e_plot:
            self._plot_energy()

    def _plot_d2o(self):
        fig = self.d2o_plot.fig
        fig.clear()
        ax = fig.add_subplot(111)
        pr, ex = self.probe(), self.exchange()
        x = np.linspace(0, 1, 101)
        try:
            s0, s1 = water(0).properties(pr).n_sld, water(1).properties(pr).n_sld
        except Exception:  # noqa: BLE001
            self.d2o_plot.canvas.draw_idle()
            return
        ax.plot(100 * x, s0 + x * (s1 - s0), color="k", lw=1.5, label="H₂O / D₂O")
        for i in range(len(self.entries)):
            m = self._material(i)
            if m is None:
                continue
            try:
                _x, y, _s = d2o_scan(m, pr, ex, 101)
                xm = match_point(m, pr, ex)
            except Exception:  # noqa: BLE001
                continue
            c = COLOURS[i % len(COLOURS)]
            lab = m.name + (f"  (match {100 * xm:.1f} %)" if 0 <= xm <= 1 else "")
            ax.plot(100 * x, y, color=c, lw=2.2 if i == self.current() else 1.2, label=lab)
            if 0 <= xm <= 1:
                ax.plot([100 * xm], [s0 + xm * (s1 - s0)], "o", color=c, ms=6)
        ax.axvline(100 * self.d2o.value(), color="0.6", ls=":", lw=1)
        ax.set_xlabel("D₂O fraction of the solvent (%)")
        ax.set_ylabel("neutron SLD (10⁻⁶ Å⁻²)")
        ax.set_xlim(0, 100)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8, loc="best")
        self.d2o_plot.canvas.draw_idle()

    def _plot_energy(self):
        fig = self.e_plot.fig
        fig.clear()
        m = self._material()
        lo, hi = _num(self.e_min.text()), _num(self.e_max.text())
        if m is None or lo is None or hi is None or not 0 < lo < hi:
            self.e_plot.canvas.draw_idle()
            return
        en = np.linspace(lo, hi, self.e_n.value())
        table = self.x_table.currentData()
        try:
            s = energy_scan(m, en, table, self.d2o.value(), self.exchange())
            mat = E.matrix_material(self.matrix_key(), self.entries, self.d2o.value())
            sm = energy_scan(mat, en, table, self.d2o.value(), self.exchange()) if mat is not None else None
        except Exception as exc:  # noqa: BLE001
            self.statusBar().showMessage(str(exc), 10000)
            self.e_plot.canvas.draw_idle()
            return
        what = self.e_what.currentIndex()
        c = COLOURS[(self.current() or 0) % len(COLOURS)]
        if what == 2:
            ax = fig.add_subplot(111)
            ref = sm["sld"] if sm is not None else 0.0
            refi = sm["sld_imag"] if sm is not None else 0.0
            ax.plot(en, (s["sld"] - ref) ** 2 + (s["sld_imag"] - refi) ** 2, color=c)
            ax.set_ylabel(f"|Δρ|² to {'vacuum' if mat is None else mat.name} (10²⁰ cm⁻⁴)")
            axes = [ax]
        else:
            ax = fig.add_subplot(211)
            ax2 = fig.add_subplot(212, sharex=ax)
            k1, k2 = ("sld", "sld_imag") if what == 0 else ("f1", "f2")
            ax.plot(en, s[k1], color=c, label=m.name)
            ax2.plot(en, s[k2], color=c, label=m.name)
            if sm is not None and what == 0:
                ax.plot(en, sm["sld"], color="k", lw=1, label=mat.name)
                ax2.plot(en, sm["sld_imag"], color="k", lw=1, label=mat.name)
            ax.set_ylabel("SLD real (10⁻⁶ Å⁻²)" if what == 0 else "f₁ = Σ(Z + f′)  (e)")
            ax2.set_ylabel("SLD imaginary (10⁻⁶ Å⁻²)" if what == 0 else "f₂ = Σ f″  (e)")
            ax.legend(fontsize=8)
            axes = [ax, ax2]
        for a in axes:
            a.grid(alpha=0.3)
            a.axvline(self._energy, color="0.6", ls=":", lw=1)
        top = axes[0]
        for sym, name, e in s["edges"]:
            for a in axes:
                a.axvline(e, color="#c0392b", lw=0.6, alpha=0.6)
            top.annotate(f"{sym} {name}", (e, 1.0), xycoords=("data", "axes fraction"), fontsize=7, rotation=90,
                         va="top", ha="right", color="#c0392b")
        axes[-1].set_xlabel(f"X-ray energy (keV)   [{s['table']}]")
        self.e_plot.canvas.draw_idle()

    # ------------------------------------------------------------------ table export
    def copy_table(self):
        QtWidgets.QApplication.clipboard().setText(E.table_text(self._rows, self._columns))
        self.statusBar().showMessage("table copied", 5000)

    def copy_selection(self):
        rng = self.table.selectedRanges()
        if not rng:
            return self.copy_table()
        r = rng[0]
        lines = []
        for i in range(r.topRow(), r.bottomRow() + 1):
            cells = [self.table.item(i, j) for j in range(r.leftColumn(), r.rightColumn() + 1)]
            lines.append("\t".join(c.text() if c else "" for c in cells))
        QtWidgets.QApplication.clipboard().setText("\n".join(lines))

    def export_table(self):
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Export table", "sld.csv", "CSV (*.csv);;Text (*.txt)")
        if path:
            sep = "\t" if path.lower().endswith(".txt") else ","
            with open(path, "w", encoding="utf-8-sig", newline="") as fh:
                fh.write(E.table_text(self._rows, self._columns, sep))

    # ------------------------------------------------------------------ Python code
    def python_code(self):
        from ..core.pycode import Script, lit
        sc = Script("pySLDcalculator", "scattering length densities, contrast and excess scattering length")
        sc.imp("from pysasfit.sld import Exchange, Probe", "from pysasfit.sld.entries import result_table, table_text")
        sc.add(f"entries = {lit(self.entries)}")
        sc.add(f"probe = Probe(neutron_wavelength={lit(self.n_wavelength.value())}, "
               f"xray_energy={lit(self._energy)}, xray_table={lit(self.x_table.currentData())})")
        sc.add(f"exchange = Exchange(backbone={lit(self.ex_backbone.value())}, side={lit(self.ex_side.value())})")
        sc.add(f"rows, columns, errors = result_table(entries, matrix={lit(self.matrix_key())}, probe=probe, "
               f"d2o={lit(self.d2o.value())},",
               f"                                     exchange=exchange, thickness={lit(self.thickness.value())})")
        sc.add("print(table_text(rows, columns))", "print(errors)")
        return sc.text()

    # ------------------------------------------------------------------ files and session
    def state(self):
        return {"entries": self.entries, "selected": self.current(), "matrix": self.matrix.currentData(),
                "neutron_wavelength": self.n_wavelength.value(), "xray_energy": self._energy,
                "xray_unit": self.x_unit.currentIndex(), "xray_table": self.x_table.currentData(),
                "d2o": self.d2o.value(), "exchange_backbone": self.ex_backbone.value(),
                "exchange_side": self.ex_side.value(), "thickness": self.thickness.value(),
                "e_min": self.e_min.text(), "e_max": self.e_max.text(), "e_n": self.e_n.value(),
                "e_what": self.e_what.currentIndex(), "tab": self.tabs.currentIndex(),
                "splitter": self.split.sizes(),
                "search_sources": self._search.sources() if getattr(self, "_search", None) is not None
                else getattr(self, "_search_sources", None)}

    def apply_state(self, st):
        ents = [dict(E.new_entry(), **e) for e in st.get("entries") or [] if isinstance(e, dict)]
        if not ents:
            return False
        self._loading = True
        try:
            self.entries = ents
            self.n_wavelength.setValue(float(st.get("neutron_wavelength", 6.0)))
            self._energy = float(st.get("xray_energy", self._energy))
            self.x_unit.setCurrentIndex(int(st.get("xray_unit", 0)))
            self.x_table.setCurrentIndex(max(self.x_table.findData(st.get("xray_table", "chantler")), 0))
            self.d2o.setValue(float(st.get("d2o", 0.0)))
            self.ex_backbone.setValue(float(st.get("exchange_backbone", 0.9)))
            self.ex_side.setValue(float(st.get("exchange_side", 1.0)))
            self.thickness.setValue(float(st.get("thickness", 0.1)))
            self.e_min.setText(str(st.get("e_min", "5")))
            self.e_max.setText(str(st.get("e_max", "25")))
            self.e_n.setValue(int(st.get("e_n", 1500)))
            self.e_what.setCurrentIndex(int(st.get("e_what", 0)))
            if isinstance(st.get("search_sources"), list) and getattr(self, "_search", None) is None:
                self._search_sources = st["search_sources"]
        finally:
            self._loading = False
        self._show_energy()
        self._fill_list()
        k = self.matrix.findData(st.get("matrix", "water"))
        self.matrix.blockSignals(True)
        self.matrix.setCurrentIndex(k if k >= 0 else 0)
        self.matrix.blockSignals(False)
        sel = st.get("selected")
        self.list.setCurrentCell(sel if isinstance(sel, int) and 0 <= sel < len(ents) else 0, 0)
        if isinstance(st.get("splitter"), list) and len(st["splitter"]) == 2:
            self.split.setSizes([int(v) for v in st["splitter"]])
        i = st.get("tab")
        if isinstance(i, int) and 0 <= i < self.tabs.count():
            self.tabs.setCurrentIndex(i)
        self._show_entry()
        self.recalc()
        return True

    def on_save(self):
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save materials", "materials.sld.json",
                                                        "pySLDcalculator (*.sld.json *.json)")
        if path:
            with open(path, "w", encoding="utf-8") as fh:
                json.dump(self.state(), fh, indent=1)

    def on_open(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Open materials", "",
                                                        "pySLDcalculator (*.sld.json *.json);;All files (*)")
        if path:
            self.open_file(path)

    def open_file(self, path):
        try:
            with open(path, encoding="utf-8") as fh:
                st = json.load(fh)
            if not self.apply_state(st):
                raise ValueError("no materials in the file")
        except (OSError, ValueError, TypeError) as exc:
            QtWidgets.QMessageBox.warning(self, "Open materials", f"{path}\n\n{exc}")

    def save_session(self):
        try:
            p = _session_path()
            os.makedirs(os.path.dirname(p), exist_ok=True)
            with open(p, "w", encoding="utf-8") as fh:
                json.dump(self.state(), fh, indent=1)
        except (OSError, TypeError, ValueError):
            pass

    def restore_session(self):
        try:
            with open(_session_path(), encoding="utf-8") as fh:
                st = json.load(fh)
            return bool(self.apply_state(st))
        except Exception:  # noqa: BLE001 -- a broken session must not stop the tool
            return False

    def forget_session(self):
        try:
            os.remove(_session_path())
        except OSError:
            pass

    def shutdown(self):
        self.save_session()

    def closeEvent(self, event):
        self.save_session()
        super().closeEvent(event)


def main(argv=None):
    argv = sys.argv if argv is None else argv
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(argv)
    try:
        from ..core.tabstyle import apply_tab_style
        apply_tab_style(app)
    except Exception:  # noqa: BLE001
        pass
    app.setWindowIcon(tool_icon("pySLDcalculator"))
    w = MainWindow()
    w.show()
    if len(argv) > 1 and os.path.isfile(argv[1]):
        w.open_file(argv[1])
    else:
        w.restore_session()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
