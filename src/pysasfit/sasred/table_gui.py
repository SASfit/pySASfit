# SPDX-License-Identifier: BSD-3-Clause
"""Reduction table panel: the correction matrix (Cadmium, Water, Water-BG, Sample-BG, Direct beam) and the
sample table of one configuration, filled from the run plan or by hand. The button in the first
column of each row opens the run list of that row: which runs are summed, which are left out and
why (outliers, aborted runs, beam stop position), with an editable comment."""
from __future__ import annotations

from typing import Callable, Dict, List, Optional

from PySide6 import QtCore, QtGui, QtWidgets

from .reduction_table import CORRECTIONS, OPTIONAL, ReductionTable, RowSet, RunChoice, configurations, from_plan, parse_runs

__all__ = ["ReductionTableWidget", "GeometryPage", "RunSetDialog"]

_RED = QtGui.QColor("firebrick")
_ORANGE = QtGui.QColor("darkorange")
_GREY = QtGui.QColor("gray")
MISSING_BG = QtGui.QColor(255, 196, 196)        # cell of a missing input
CHECK_BG = QtGui.QColor(255, 226, 180)          # cell of an input to check
#: table column of a missing field (Missing.field)
FIELD_COLUMN = {"use": "Use", "runs": "Summed runs", "transmission": "Transmission", "thickness": "Thickness [cm]",
                "aperture": "Aperture [cm²]"}

# correction rows (Cd, water, water background, sample background) need transmissions but no
# thickness (the water's thickness is in g(λ), the backgrounds are subtracted per thickness of the sample)
COLS_CORR = ["Runs", "Use", "Information", "Number", "Summed runs", "Transmission", "Aperture [cm²]", "Notes"]
COLS_SAMPLE = ["Runs", "Information", "Number", "Summed runs", "Transmission", "Thickness [cm]", "Aperture [cm²]",
               "Attenuator A", "Probability p_cS", "Notes"]
_EDIT = {"Information": "info", "Transmission": "transmission", "Thickness [cm]": "scaling",
         "Aperture [cm²]": "aperture_cm2",
         "Attenuator A": "attenuator", "Probability p_cS": "probability"}


def _fmt(v, f="{:.4g}"):
    return "" if v is None else f.format(v)


class RunSetDialog(QtWidgets.QDialog):
    """Spreadsheet of the runs of one row: use (summed) or not, with the reason."""

    COLS = ["use", "run", "sample", "kind", "L1 [m]", "L2 [m]", "start", "time [s]", "monitor", "counts/monitor", "comment"]

    OTHER = "other run of this geometry"

    def __init__(self, row: RowSet, raws: Dict[int, object], parent=None, candidates: Optional[List[int]] = None,
                 geometry_runs: Optional[tuple] = None):
        super().__init__(parent)
        self.setWindowTitle(f"{row.label}: {row.info or 'runs'}")
        self.row = row
        self.raws = raws
        self.candidates = candidates or []
        self.geometry_runs = geometry_runs or ([], [])     # (scattering runs, transmission runs) of the geometry
        self.new_transmission = None
        lay = QtWidgets.QVBoxLayout(self)
        if row.notes:
            note = QtWidgets.QLabel(row.notes)
            note.setWordWrap(True)
            lay.addWidget(note)
        self.tabs = QtWidgets.QTabWidget()
        self.scatt = self._table(row.runs)
        self.trans = self._table(row.trans_runs)
        self.tabs.addTab(self.scatt, "Scattering runs")
        self.tabs.addTab(self.trans, "Transmission runs")
        lay.addWidget(self.tabs, 1)
        add = QtWidgets.QHBoxLayout()
        self.add_edit = QtWidgets.QLineEdit()
        self.add_edit.setPlaceholderText("add runs, e.g. 82187, 82190-82195")
        self.add_edit.returnPressed.connect(self.on_add)
        b = QtWidgets.QPushButton("Add")
        b.clicked.connect(self.on_add)
        self.cand = QtWidgets.QCheckBox("show other runs of this sample")
        self.cand.toggled.connect(self.on_candidates)
        self.cand.setEnabled(bool(self.candidates))
        self.allg = QtWidgets.QCheckBox("list all runs of this geometry")
        self.allg.setToolTip("Lists every scattering run of this geometry (tab Scattering runs) and every transmission "
                             "run of this wavelength (tab Transmission runs), not ticked: tick the ones that belong "
                             "to this row, untick the ones that do not.")
        self.allg.toggled.connect(self.on_all_geometry)
        self.allg.setEnabled(bool(self.geometry_runs[0] or self.geometry_runs[1]))
        self.filt = QtWidgets.QLineEdit()
        self.filt.setPlaceholderText("filter: sample name or run")
        self.filt.textChanged.connect(self._filter)
        add.addWidget(self.add_edit, 1)
        add.addWidget(b)
        add.addWidget(self.cand)
        add.addWidget(self.allg)
        add.addWidget(self.filt)
        lay.addLayout(add)
        trow = QtWidgets.QHBoxLayout()
        tb = QtWidgets.QPushButton("Transmission from the ticked transmission runs")
        tb.setToolTip("τ (beam spot) of every ticked transmission run against the empty-beam transmission run of the "
                      "same distance (same attenuator preferred); taken over with OK")
        tb.clicked.connect(self.on_transmission)
        self.t_label = QtWidgets.QLabel(f"now: {row.transmission:.4f} ({row.transmission_source})"
                                        if row.transmission is not None else "now: none")
        trow.addWidget(tb)
        trow.addWidget(self.t_label, 1)
        lay.addLayout(trow)
        bb = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.StandardButton.Ok
                                        | QtWidgets.QDialogButtonBox.StandardButton.Cancel)
        bb.accepted.connect(self.accept)
        bb.rejected.connect(self.reject)
        lay.addWidget(bb)
        self.resize(1000, 420)

    def _table(self, choices: List[RunChoice]) -> QtWidgets.QTableWidget:
        t = QtWidgets.QTableWidget(0, len(self.COLS))
        t.setHorizontalHeaderLabels(self.COLS)
        t.verticalHeader().setVisible(False)
        t.horizontalHeader().setStretchLastSection(True)
        for c in choices:
            self._add_row(t, c)
        t.resizeColumnsToContents()
        return t

    def _add_row(self, t, c: RunChoice):
        i = t.rowCount()
        t.insertRow(i)
        use = QtWidgets.QTableWidgetItem()
        use.setFlags(QtCore.Qt.ItemFlag.ItemIsUserCheckable | QtCore.Qt.ItemFlag.ItemIsEnabled)
        use.setCheckState(QtCore.Qt.CheckState.Checked if c.use else QtCore.Qt.CheckState.Unchecked)
        t.setItem(i, 0, use)
        r = self.raws.get(c.run)
        vals = [str(c.run)] + [""] * (len(self.COLS) - 3)
        if r is not None:
            from .catalogue import is_transmission
            m = float(r.monitors.get("monitor1" if r.instrument == "sans_i" else r.preset_monitor, 0))
            vals = [str(c.run), r.sample_name, "trans." if is_transmission(r) else "scatt.", f"{r.L1_m:.1f}",
                    f"{r.L2_m:.2f}",
                    str(r.start_time)[:19], f"{r.counting_time_s:.0f}", f"{m:.4g}",
                    f"{float(r.counts.sum()) / m:.5g}" if m else ""]
        for j, v in enumerate(vals, start=1):
            it = QtWidgets.QTableWidgetItem(v)
            it.setFlags(QtCore.Qt.ItemFlag.ItemIsEnabled | QtCore.Qt.ItemFlag.ItemIsSelectable)
            t.setItem(i, j, it)
        why = getattr(r, "empty_reason", "") if r is not None else ""
        shown = c.comment
        if why and not c.comment.startswith("skipped"):
            shown = f"skipped: {why}" + (f"; {c.comment}" if c.comment else "")
        if why:
            t.item(i, 0).setCheckState(QtCore.Qt.CheckState.Unchecked)
            t.item(i, 0).setFlags(QtCore.Qt.ItemFlag.ItemIsEnabled)          # cannot be ticked
            t.item(i, 0).setToolTip(f"no usable data: {why}")
        com = QtWidgets.QTableWidgetItem(shown)
        t.setItem(i, len(self.COLS) - 1, com)
        colour = None
        low = shown.lower()
        if why:
            colour = _GREY
        elif r is None:
            com.setToolTip("run not in the folder")
            colour = _GREY
        elif "outlier" in low or "not summed" in low:
            colour = _RED
        elif "aborted" in low or "beam stop" in low:
            colour = _ORANGE
        if colour is not None:
            for j in range(1, len(self.COLS)):
                t.item(i, j).setForeground(QtGui.QBrush(colour))

    def _current(self):
        return self.scatt if self.tabs.currentIndex() == 0 else self.trans

    def _runs_in(self, t) -> List[int]:
        return [int(t.item(i, 1).text()) for i in range(t.rowCount())]

    def on_add(self):
        try:
            runs = parse_runs(self.add_edit.text())
        except ValueError as e:
            QtWidgets.QMessageBox.warning(self, "Runs", str(e))
            return
        t = self._current()
        have = set(self._runs_in(t))
        for r in runs:
            if r not in have:
                self._add_row(t, RunChoice(r, True, "added by hand"))
        self.add_edit.clear()

    def on_candidates(self, on):
        t = self.scatt
        have = set(self._runs_in(t))
        if on:
            for r in self.candidates:
                if r not in have:
                    self._add_row(t, RunChoice(r, False, "same sample, other group"))
        else:
            for i in reversed(range(t.rowCount())):
                it = t.item(i, len(self.COLS) - 1)
                if it.text() == "same sample, other group" and t.item(i, 0).checkState() != QtCore.Qt.CheckState.Checked:
                    t.removeRow(i)

    def on_all_geometry(self, on):
        for t, runs in ((self.scatt, self.geometry_runs[0]), (self.trans, self.geometry_runs[1])):
            have = set(self._runs_in(t))
            if on:
                for r in runs:
                    if r not in have:
                        self._add_row(t, RunChoice(r, False, self.OTHER))
                t.sortItems(1)
            else:
                for i in reversed(range(t.rowCount())):
                    if t.item(i, len(self.COLS) - 1).text() == self.OTHER and \
                            t.item(i, 0).checkState() != QtCore.Qt.CheckState.Checked:
                        t.removeRow(i)
        self._filter(self.filt.text())

    def _filter(self, text):
        text = text.strip().lower()
        for t in (self.scatt, self.trans):
            for i in range(t.rowCount()):
                hay = (t.item(i, 1).text() + " " + (t.item(i, 2).text() if t.item(i, 2) else "")).lower()
                t.setRowHidden(i, bool(text) and text not in hay)

    def on_transmission(self):
        from .reduction_table import transmission_from_runs
        runs = [c.run for c in self._read(self.trans) if c.use]
        if not runs:
            self.t_label.setText("no transmission run ticked")
            return
        ct = transmission_from_runs(runs, self.raws)
        if ct is None:
            self.t_label.setText("no empty-beam transmission run at this distance and wavelength")
            return
        self.new_transmission = ct
        each = ", ".join(f"{x}: {v:.4f}" for x, v in zip(ct[2], ct[4]))
        self.t_label.setText(f"τ = {ct[0]:.4f} ± {ct[1]:.4f} (runs {each}; empty beam {ct[3]}) – taken over with OK")

    @staticmethod
    def _read(t) -> List[RunChoice]:
        out = []
        for i in range(t.rowCount()):
            use = t.item(i, 0).checkState() == QtCore.Qt.CheckState.Checked
            com = t.item(i, len(RunSetDialog.COLS) - 1).text()
            base = com.split("; ", 1)[1] if com.startswith("skipped:") and "; " in com else com
            if base in ("same sample, other group", RunSetDialog.OTHER) and not use:
                continue
            if com in ("same sample, other group", RunSetDialog.OTHER) and not use:
                continue
            if com == RunSetDialog.OTHER:
                com = "added by hand"
            out.append(RunChoice(int(t.item(i, 1).text()), use, com))
        return sorted(out, key=lambda c: c.run)

    def apply(self):
        self.row.runs = self._read(self.scatt)
        self.row.trans_runs = self._read(self.trans)
        if self.new_transmission is not None:
            ct = self.new_transmission
            self.row.transmission, self.row.transmission_sigma = ct[0], ct[1]
            self.row.transmission_source = f"runs {ct[2]} / empty beam {ct[3]} (beam spot)"


_TABLE_SETTINGS = ("water_g", "scale", "efficiency", "direct_beam_radius_mm", "direct_beam_attenuator",
                   "efficiency_mask_filled", "ignored")
_ROW_VALUES = ("scaling", "aperture_cm2", "attenuator", "probability", "notes")


def carry_over(new, old) -> str:
    """Refilling a geometry (folder read again, names corrected) keeps what was set by hand in its table
    before: the absolute-scale settings (method, efficiency map, direct beam, g(λ), switched-off rows),
    and per row (corrections by role, samples by name) the runs switched off and their comments, a
    transmission typed by hand, and thickness, aperture, attenuator, probability and notes that differ
    from the values the files give. New runs are added as usual. Returns a short summary ("" = nothing)."""
    if old is None:
        return ""
    what = []
    for k in _TABLE_SETTINGS:
        v = getattr(old, k)
        if v != getattr(new, k):
            setattr(new, k, list(v) if isinstance(v, list) else v)
            what.append(k.replace("_", " "))
    pairs = [(new.corrections[r], old.corrections.get(r)) for r in new.corrections]
    olds = {}
    for r in old.samples:
        olds.setdefault(r.info, r)
    pairs += [(r, olds.get(r.info)) for r in new.samples]
    n_off = n_val = 0
    for nr, orow in pairs:
        if orow is None:
            continue
        for lst, olst in ((nr.runs, orow.runs), (nr.trans_runs, orow.trans_runs)):
            o = {c.run: c for c in olst}
            for c in lst:
                if c.run in o:
                    if c.use != o[c.run].use:
                        n_off += 1
                    c.use, c.comment = o[c.run].use, o[c.run].comment or c.comment
        if orow.transmission_source == "by hand":
            nr.transmission, nr.transmission_sigma = orow.transmission, orow.transmission_sigma
            nr.transmission_source = "by hand"
            n_val += 1
        for k in _ROW_VALUES:
            v = getattr(orow, k)
            if v != getattr(nr, k) and v not in (None, ""):
                setattr(nr, k, v)
                n_val += 1
    if n_off:
        what.append(f"{n_off} run(s) switched on/off")
    if n_val:
        what.append(f"{n_val} value(s) set by hand")
    return ", ".join(what)


class _PopupCombo(QtWidgets.QComboBox):
    """Combo box that rebuilds its entries just before it opens."""

    def __init__(self, before_popup, parent=None):
        super().__init__(parent)
        self._before = before_popup

    def showPopup(self):
        self._before()
        super().showPopup()


class GeometryPage(QtWidgets.QWidget):
    """Correction matrix + sample table of one geometry (one tab of the reduction table panel)."""

    changed = QtCore.Signal()

    def __init__(self, panel: "ReductionTableWidget", table: ReductionTable, ref_raw=None):
        super().__init__(panel)
        self.panel = panel
        self.table = table
        self.ref_raw = ref_raw                     # a run of this geometry (for presets)
        lay = QtWidgets.QVBoxLayout(self)
        box = QtWidgets.QGroupBox("Correction data")
        bl = QtWidgets.QVBoxLayout(box)
        self.corr = self._make_table(COLS_CORR, len(CORRECTIONS))
        self.corr.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        bl.addWidget(self.corr)
        top = QtWidgets.QHBoxLayout()
        top.addWidget(box, 1)
        wbox = QtWidgets.QGroupBox("Absolute scale")
        wl = QtWidgets.QFormLayout(wbox)
        self.scale_combo = QtWidgets.QComboBox()
        for text, key in (("automatic", "auto"), ("water", "water"), ("direct beam + efficiency map", "direct_beam"),
                          ("relative (per sr and cm)", "relative")):
            self.scale_combo.addItem(text, key)
        self.scale_combo.setToolTip("automatic: the direct beam (row Direct beam) as soon as an efficiency map is "
                                    "chosen below – the water of this geometry is then not used –, else water when the "
                                    "Water row has runs, else relative.\nDirect beam: dΣ/dΩ = N/(ε ΔΩ d Φ), "
                                    "Φ = Σ (DB − Cd)/ε around the beam, corrected for its attenuator")
        self.scale_combo.currentIndexChanged.connect(self._on_scale)
        self.scale_used = QtWidgets.QLabel()
        self.eff_combo = _PopupCombo(self._update_scale_box)
        self.eff_combo.setToolTip("Detector efficiency map for the direct-beam scale: made from the water rows of a "
                                  "geometry (usually a shorter distance) or read from a file (a saved map or an "
                                  "efficiency archive)")
        self.eff_combo.activated.connect(self._on_eff)
        eff_row = QtWidgets.QHBoxLayout()
        eff_row.addWidget(self.eff_combo, 1)
        sb = QtWidgets.QPushButton("Show")
        sb.setToolTip("Show the efficiency map in the detector panel")
        sb.clicked.connect(self._on_show_eff)
        eff_row.addWidget(sb)
        self.eff_fill = QtWidgets.QCheckBox("mask the pixels the map only estimates")
        self.eff_fill.setToolTip("Pixels masked or rejected in the water of the map (e.g. behind its beam stop) are "
                                 "estimated as row × column; ticked: they are masked in the sample too")
        self.eff_fill.toggled.connect(self._on_eff_fill)
        self.db_radius = QtWidgets.QLineEdit()
        self.db_radius.setPlaceholderText("automatic")
        self.db_radius.setToolTip("Radius [mm] around the direct beam within which it is summed; empty: the radius "
                                  "holding 99.5 % of its counts, + 2 pixels")
        self.db_radius.editingFinished.connect(lambda: self._on_number(self.db_radius, "direct_beam_radius_mm"))
        self.db_att = QtWidgets.QLineEdit()
        self.db_att.setToolTip("Transmission of the attenuator of the direct-beam runs; empty: the water calibration "
                               "of the instrument file at this wavelength if there is one, else its nominal value")
        self.db_att.editingFinished.connect(lambda: self._on_number(self.db_att, "direct_beam_attenuator"))
        wl.addRow("method:", self.scale_combo)
        wl.addRow("used:", self.scale_used)
        wl.addRow("efficiency map:", eff_row)
        wl.addRow("", self.eff_fill)
        wl.addRow("direct beam radius [mm]:", self.db_radius)
        wl.addRow("attenuator transmission:", self.db_att)
        self.g_auto = QtWidgets.QLabel()
        self.g_edit = QtWidgets.QLineEdit()
        self.g_edit.setPlaceholderText("from instrument file")
        self.g_edit.setToolTip("g(λ) (forward excess K0 of the water) by hand; empty: the polynomial of the "
                               "instrument file (water.g_lambda) at the wavelength of this geometry")
        self.g_edit.editingFinished.connect(self._on_g_edit)
        self.w_den = QtWidgets.QLabel()
        self.w_scale = QtWidgets.QLabel()
        self.w_scale.setToolTip("BerSANS ScalingFactor(W) = g A_W (1 - T_W/T_EC)/(4π D), incl. g(λ); "
                                "the sample factor is its thickness (× aperture when both apertures are given)")
        wl.addRow("g(λ), instrument:", self.g_auto)
        wl.addRow("g(λ) used:", self.g_edit)
        wl.addRow("denominator D:", self.w_den)
        wl.addRow("water scaling:", self.w_scale)
        top.addWidget(wbox)
        lay.addLayout(top)
        box2 = QtWidgets.QGroupBox("Sample data")
        sl = QtWidgets.QVBoxLayout(box2)
        self.samp = self._make_table(COLS_SAMPLE, 0)
        self.samp.itemSelectionChanged.connect(lambda: self.panel.rows_selected.emit())
        sl.addWidget(self.samp, 1)
        row = QtWidgets.QHBoxLayout()
        red = QtWidgets.QHBoxLayout()
        b = QtWidgets.QPushButton("Reduce all samples (2D → I(q))")
        b.setToolTip("Every sample row of this geometry: per pixel with the correction data, then the radial average")
        b.clicked.connect(lambda: self.on_reduce(all_rows=True))
        b2 = QtWidgets.QPushButton("Reduce selected samples")
        b2.clicked.connect(lambda: self.on_reduce(all_rows=False))
        red.addWidget(b)
        red.addWidget(b2)
        red.addStretch(1)
        for text, fn in (("Add a sample", self.on_add_sample), ("Add many samples…", self.on_add_many),
                         ("Remove sample", self.on_remove_sample),
                         ("Copy to all samples below", lambda: self.on_copy("below")),
                         ("Copy to all samples above", lambda: self.on_copy("above"))):
            b = QtWidgets.QPushButton(text)
            b.clicked.connect(fn)
            row.addWidget(b)
        row.addStretch(1)
        sl.addLayout(row)
        lay.addWidget(box2, 1)
        self.problems = QtWidgets.QLabel()
        self.problems.setWordWrap(True)
        lay.addWidget(self.problems)
        lay.addLayout(red)
        self._filling = False
        self.refresh()

    def selected_rows(self) -> List[RowSet]:
        rows = sorted({ix.row() for ix in self.samp.selectedIndexes()})
        return [self.table.samples[i] for i in rows if 0 <= i < len(self.table.samples)]

    def on_reduce(self, all_rows: bool = True):
        rows = list(self.table.samples) if all_rows else self.selected_rows()
        rows = [r for r in rows if r.used]
        if not rows:
            self._log("no sample rows with runs to reduce" + ("" if all_rows else " (select cells of the rows)"))
            return None
        fn = getattr(self.panel, "reduce_rows", None)
        if fn is None:
            self._log("reduction not available here")
            return None
        return fn(self.table, rows)

    def _log(self, text):
        self.panel._log(text)

    # ------------------------------------------------------------------ tables
    def _make_table(self, cols, n):
        t = QtWidgets.QTableWidget(n, len(cols))
        t.setHorizontalHeaderLabels(cols)
        t.verticalHeader().setVisible(False)
        t.horizontalHeader().setStretchLastSection(True)
        t.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectionBehavior.SelectItems)
        t.itemChanged.connect(lambda it, t=t: self._on_edit(t, it))
        return t

    def _row_of(self, t, i) -> RowSet:
        return self.table.corrections[list(CORRECTIONS)[i]] if t is self.corr else self.table.samples[i]
    def _missing_of(self, t, i, r: RowSet):
        """{column: Missing} of a row."""
        out = {}
        for x in getattr(self, "_missing", []):
            if (t is self.corr and x.index is None and x.role == r.role) or \
                    (t is not self.corr and x.role == "sample" and x.index == i):
                col = FIELD_COLUMN.get(x.field)
                if col:
                    out.setdefault(col, x)
        return out

    def _fill_row(self, t, i, r: RowSet, cols):
        miss = self._missing_of(t, i, r)
        b = QtWidgets.QPushButton(r.label)
        b.setToolTip("Runs of this row: summed, left out, comments")
        if "Summed runs" in miss and miss["Summed runs"].severity == "info":
            b.setToolTip(miss["Summed runs"].text + "\n(click: the runs of this row)")
        elif "Summed runs" in miss:
            b.setStyleSheet("QPushButton { color: firebrick; font-weight: bold; }")
            b.setToolTip(miss["Summed runs"].text + "\n(click: the runs of this row)")
        b.clicked.connect(lambda _=False, t=t, i=i: self.open_runs(t, i))
        t.setCellWidget(i, 0, b)
        vals = {"Information": r.info, "Number": r.number if r.runs else "",
                "Summed runs": r.runs_text(), "Transmission": _fmt(r.transmission),
                "Thickness [cm]": _fmt(r.scaling), "Aperture [cm²]": _fmt(r.aperture_cm2), "Attenuator A": _fmt(r.attenuator),
                "Probability p_cS": _fmt(r.probability), "Notes": r.notes}
        for j, c in enumerate(cols[1:], start=1):
            it = QtWidgets.QTableWidgetItem(vals.get(c, ""))
            if c not in _EDIT:
                it.setFlags(QtCore.Qt.ItemFlag.ItemIsEnabled | QtCore.Qt.ItemFlag.ItemIsSelectable)
            if r.role in ("water", "empty_cell") and t is self.corr and self._method != "water" and r.runs:
                it.setForeground(QtGui.QBrush(_GREY))
                it.setToolTip("not used: " + ("absolute scale from the direct beam and the efficiency map"
                                              if self._method == "direct_beam" else "relative scale"))
            if c == "Use":
                if r.role in OPTIONAL:
                    it.setFlags(QtCore.Qt.ItemFlag.ItemIsUserCheckable | QtCore.Qt.ItemFlag.ItemIsEnabled)
                    off = r.role in self.table.ignored
                    it.setCheckState(QtCore.Qt.CheckState.Unchecked if off else QtCore.Qt.CheckState.Checked)
                    it.setToolTip("Untick to reduce without this correction (it is optional); the runs stay in the row")
                else:
                    it.setFlags(QtCore.Qt.ItemFlag.ItemIsEnabled)
            if c in ("Transmission",) and r.role == "direct_beam":
                it.setText("—")
                it.setFlags(QtCore.Qt.ItemFlag.ItemIsEnabled)
                it.setToolTip("the direct beam needs no transmission (its attenuator is corrected)")
            if c == "Transmission" and r.transmission_source:
                it.setToolTip(r.transmission_source + (f", σ = {r.transmission_sigma:g}" if r.transmission_sigma else ""))
            if c == "Notes":
                it.setToolTip(r.notes)
                if r.notes.startswith("decision needed"):
                    it.setForeground(QtGui.QBrush(_RED))
            if c == "Number" and r.runs and len(r.used) < len(r.runs):
                it.setForeground(QtGui.QBrush(_ORANGE))
            x = miss.get(c)
            if x is not None and x.severity == "info":
                it.setToolTip(x.text + (f"\n{it.toolTip()}" if it.toolTip() else ""))
            elif x is not None:
                it.setBackground(QtGui.QBrush(MISSING_BG if x.severity == "missing" else CHECK_BG))
                it.setToolTip(x.text + (f"\n{it.toolTip()}" if it.toolTip() else ""))
            t.setItem(i, j, it)

    def refresh(self):
        self._filling = True
        try:
            try:
                self._missing = self.table.missing_inputs()
                self._method = self.table.scale_method()
            except Exception:  # noqa: BLE001 -- e.g. an unknown scale method in a loaded table
                self._missing, self._method = [], ""
            for i, role in enumerate(CORRECTIONS):
                self._fill_row(self.corr, i, self.table.corrections[role], COLS_CORR)
            self.samp.setRowCount(len(self.table.samples))
            for i, r in enumerate(self.table.samples):
                self._fill_row(self.samp, i, r, COLS_SAMPLE)
            for t in (self.corr, self.samp):
                t.resizeColumnsToContents()
                t.resizeRowsToContents()
                for j in range(1, t.columnCount() - 1):
                    t.setColumnWidth(j, min(t.columnWidth(j), 260))
            h = self.corr.horizontalHeader().height() + sum(self.corr.rowHeight(i) for i in range(self.corr.rowCount()))
            self.corr.setFixedHeight(h + 2 * self.corr.frameWidth() + self.corr.horizontalScrollBar().sizeHint().height())
            self._update_water_box()
            self._update_scale_box()
            p = self.table.problems()
            self.problems.setText("" if not p else "Still missing: " + "; ".join(p))
        finally:
            self._filling = False

    def _update_water_box(self):
        from .absolute import water_denominator, water_g, water_scaling_factor
        from .reduce import load_instrument
        r = self.ref_raw
        inst = load_instrument(r)
        if r is None:
            self.g_auto.setText("– (no run of this geometry)")
            gv = None
        else:
            gv, rel, note = water_g(r.wavelength_A, inst)
            ext = " – extrapolated!" if "extrapolated" in note else ""
            self.g_auto.setText(f"{gv:.4f} ± {100 * (rel or 0):.1f} % at λ = {r.wavelength_A / 10:.3f} nm{ext}")
        if self.table.water_g is not None:
            gv = self.table.water_g
        if self.g_edit.text().strip() != ("" if self.table.water_g is None else f"{self.table.water_g:g}"):
            self.g_edit.setText("" if self.table.water_g is None else f"{self.table.water_g:g}")
        den = water_denominator(inst)
        self.w_den.setText("T_W (water + cell, BerSANS)" if den == "water_and_cell" else "τ_W = T_W/T_EC (Strunz)")
        w, ec = self.table.corrections["water"], self.table.corrections["empty_cell"]
        if gv is None or not w.transmission:
            self.w_scale.setText("– (water transmission missing)")
        else:
            f = water_scaling_factor(w.transmission, ec.transmission, w.aperture_cm2, gv, den)
            self.w_scale.setText(f"{f:.5g}" + ("" if ec.transmission else "  (no T_EC: τ_W = T_W)"))

    def _update_scale_box(self):
        t = self.table
        self.scale_combo.blockSignals(True)
        self.scale_combo.setCurrentIndex(max(0, self.scale_combo.findData(t.scale or "auto")))
        self.scale_combo.blockSignals(False)
        try:
            used = t.scale_method()
        except ValueError as exc:
            used = str(exc)
        self.scale_used.setText({"water": "water", "direct_beam": "direct beam + efficiency map",
                                 "relative": "relative – no absolute scale"}.get(used, used))
        self.eff_combo.blockSignals(True)
        self.eff_combo.clear()
        self.eff_combo.addItem("none", "")
        for other in self.panel.tables():
            if other.corrections["water"].used:
                self.eff_combo.addItem(f"water of {other.config}" + (" (this geometry)" if other is t else ""),
                                       f"geometry:{other.config}")
        if t.efficiency and self.eff_combo.findData(t.efficiency) < 0:
            self.eff_combo.addItem(("file: " if not t.efficiency.startswith("geometry:") else "") + t.efficiency,
                                   t.efficiency)
        self.eff_combo.addItem("file…", "__file__")
        self.eff_combo.setCurrentIndex(max(0, self.eff_combo.findData(t.efficiency or "")))
        self.eff_combo.blockSignals(False)
        em = next((x for x in getattr(self, "_missing", []) if x.field == "efficiency"), None)
        self.eff_combo.setStyleSheet("QComboBox { background: rgb(255, 196, 196); }" if em else "")
        self.eff_combo.setToolTip(em.text if em else "Detector efficiency map for the direct-beam scale: made from "
                                  "the water rows of a geometry (usually a shorter distance) or read from a file (a "
                                  "saved map or an efficiency archive)")
        self.eff_fill.blockSignals(True)
        self.eff_fill.setChecked(bool(t.efficiency_mask_filled))
        self.eff_fill.blockSignals(False)
        for w, v in ((self.db_radius, t.direct_beam_radius_mm), (self.db_att, t.direct_beam_attenuator)):
            if not w.hasFocus():
                w.setText("" if v is None else f"{v:g}")
        ph = "from the instrument file"
        r = self.ref_raw
        dbr = t.corrections["direct_beam"]
        raws = self.panel.raws() if dbr.used else {}
        atts = sorted({str(raws[n].attenuator) for n in dbr.used if raws.get(n) is not None})
        if r is not None and atts:
            try:
                from .absolute import direct_beam_attenuator
                from .reduce import load_instrument
                inst = load_instrument(r)
                ph = "; ".join(f"{direct_beam_attenuator(a, r.wavelength_A, inst)[0]:.5g} (att. {a})" for a in atts)
            except Exception:  # noqa: BLE001
                pass
        self.db_att.setPlaceholderText(ph)

    def show_missing(self, role: str, index, field: str):
        """Select the cell of a missing input (or the efficiency map box)."""
        if field == "efficiency":
            self.eff_combo.setFocus()
            return
        col = FIELD_COLUMN.get(field)
        if role == "sample":
            t, i, cols = self.samp, index, COLS_SAMPLE
        else:
            t, i, cols = self.corr, list(CORRECTIONS).index(role), COLS_CORR
        if i is None or not (0 <= i < t.rowCount()):
            return
        j = cols.index(col) if col in cols else 0
        t.setCurrentCell(i, j)
        t.scrollToItem(t.item(i, j) or t.item(i, 1))
        t.setFocus()

    def _changed(self):
        self.refresh()
        self.changed.emit()
        self.panel.changed.emit()

    def _on_scale(self, _i):
        k = self.scale_combo.currentData()
        if k != self.table.scale:
            self.table.scale = k
            self._changed()

    def _on_eff(self, _i):
        k = self.eff_combo.currentData()
        if k == "__file__":
            p, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Efficiency map", self.table.folder or "",
                                                         "HDF5 (*.h5 *.hdf5 *.nxs)")
            if not p:
                self._update_scale_box()
                return
            k = p
        if k != self.table.efficiency:
            self.table.efficiency = k
            # a map means: absolute scale from the direct beam, the water of this geometry is not used
            if k and self.table.scale == "water":
                self.table.scale = "auto"
            if not k and self.table.scale == "direct_beam":
                self.table.scale = "auto"
            self._changed()
            if k:
                self._on_show_eff()          # made (or read) now and shown: errors appear at once

    def _on_show_eff(self):
        fn = getattr(self.panel, "show_efficiency", None)
        if fn is None:
            self._log("the efficiency map can be shown in pySASred's detector panel")
            return
        fn(self.table)

    def _on_eff_fill(self, on):
        self.table.efficiency_mask_filled = bool(on)
        self._changed()

    def _on_number(self, w, attr):
        text = w.text().strip()
        try:
            v = None if text == "" else float(text)
        except ValueError:
            self._log(f"not a number: {text}")
            return
        if v is not None and v <= 0:
            self._log(f"{text}: must be positive")
            return
        if v != getattr(self.table, attr):
            setattr(self.table, attr, v)
            self._changed()

    def _on_g_edit(self):
        text = self.g_edit.text().strip()
        try:
            v = None if text == "" else float(text)
        except ValueError:
            self._log(f"not a number: {text}")
            return
        if v != self.table.water_g:
            self.table.water_g = v
            self.refresh()
            self.changed.emit()
            self.panel.changed.emit()

    def _on_edit(self, t, it):
        if self._filling:
            return
        cols = COLS_CORR if t is self.corr else COLS_SAMPLE
        name = cols[it.column()]
        if name == "Use" and t is self.corr:
            r = self._row_of(t, it.row())
            off = it.checkState() == QtCore.Qt.CheckState.Unchecked
            if r.role in OPTIONAL and off != (r.role in self.table.ignored):
                if off:
                    self.table.ignored.append(r.role)
                else:
                    self.table.ignored.remove(r.role)
                self.refresh()
                self.changed.emit()
                self.panel.changed.emit()
            return
        attr = _EDIT.get(name)
        if attr is None:
            return
        r = self._row_of(t, it.row())
        text = it.text().strip()
        try:
            if attr == "info":
                r.info = text
            else:
                v = None if text == "" else float(text)
                if attr == "transmission":
                    r.transmission, r.transmission_source = v, "by hand"
                elif attr in ("attenuator", "probability"):
                    setattr(r, attr, 1.0 if (v is None and attr == "attenuator") else (v or 0.0))
                else:
                    setattr(r, attr, v)
        except ValueError:
            self._log(f"not a number: {text}")
        self.refresh()
        self.changed.emit()
        self.panel.changed.emit()

    # ------------------------------------------------------------------ actions
    def raws(self) -> Dict[int, object]:
        return self.panel.raws()

    def _geometry_runs(self):
        """(scattering runs of this geometry, transmission runs of its wavelength) of the folder."""
        from .catalogue import _config_key
        ref = self.ref_raw
        if ref is None:
            return [], []
        k = _config_key(ref)
        sc, tr = [], []
        for e in sorted(self.panel._entries(), key=lambda e: e.run):
            n, x = e.run, getattr(e, "raw", None)
            if x is None:
                continue
            t = e.kind == "transmission"
            if t and abs(x.wavelength_A - ref.wavelength_A) <= 0.02 * ref.wavelength_A:
                tr.append(n)
            elif not t and _config_key(x) == k:
                sc.append(n)
        return sc, tr

    def open_runs(self, t, i):
        r = self._row_of(t, i)
        raws = self.raws()
        cands = []
        if r.info:
            cands = sorted(k for k, x in raws.items() if x.sample_name == r.info and k not in {c.run for c in r.runs})
        d = RunSetDialog(r, raws, self, cands, self._geometry_runs())
        if d.exec() == QtWidgets.QDialog.DialogCode.Accepted:
            d.apply()
            if not r.info and r.runs:
                x = raws.get(r.runs[0].run)
                r.info = x.sample_name if x is not None else r.info
            self.refresh()
            self.changed.emit()
        self.panel.changed.emit()

    def on_add_many(self):
        from .selection_gui import AddManySamplesDialog
        entries = [e for e in self.panel._entries() if getattr(e, "raw", None) is not None]
        if not entries:
            self._log("no runs: open a raw-data folder first")
            return
        d = AddManySamplesDialog(entries, self, check=self.panel.check_group, presets=self.presets())
        if d.exec() != QtWidgets.QDialog.DialogCode.Accepted:
            return
        self.add_groups(d.groups(), d.wants_check())

    def add_groups(self, groups, check: bool = False):
        """Sample rows from run groups (transmissions from trans.txt, transmission runs of the same
        sample and wavelength)."""
        es = [e for e in self.panel._entries() if getattr(e, "raw", None) is not None]

        def trans_runs(row, g):
            names = {r.sample_name for r in g.runs}
            lam = g.runs[0].wavelength_A
            return [e.run for e in es if e.kind == "transmission" and e.raw.sample_name in names
                    and abs(e.raw.wavelength_A - lam) < 0.02 * lam]
        QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.CursorShape.WaitCursor)
        try:
            rows = self.table.add_groups(groups, self.panel._trans_table(), trans_runs,
                                         self.panel.check_group if check else None)
        finally:
            QtWidgets.QApplication.restoreOverrideCursor()
        self._log(f"{len(rows)} samples added")
        self.refresh()
        self.changed.emit()
        self.panel.changed.emit()

    def on_add_sample(self):
        self.table.add_sample()
        self.refresh()
        self.open_runs(self.samp, len(self.table.samples) - 1)

    def on_remove_sample(self):
        i = self.samp.currentRow()
        if 0 <= i < len(self.table.samples):
            self.table.remove_sample(i)
            self.refresh()
            self.changed.emit()
        self.panel.changed.emit()

    def on_copy(self, direction):
        i, j = self.samp.currentRow(), self.samp.currentColumn()
        if not (0 <= i < len(self.table.samples)) or j < 0:
            return
        attr = _EDIT.get(COLS_SAMPLE[j])
        if attr in (None, "info"):
            self._log("select a cell of Transmission, Thickness, Aperture, Attenuator or Probability")
            return
        self.table.copy_value(i, attr, direction)
        self.refresh()
        self.changed.emit()
        self.panel.changed.emit()

    def presets(self):
        """Search conditions of this geometry for *Add many samples* (scattering runs, L2, λ, L1)."""
        r = self.ref_raw
        if r is None:
            return [("kind", ("scattering",))]
        return [("kind", ("scattering",)),
                ("L2_m", (round(0.99 * r.L2_m, 3), round(1.01 * r.L2_m, 3))),
                ("wavelength_A", (round(0.98 * r.wavelength_A, 3), round(1.02 * r.wavelength_A, 3))),
                ("L1_m", (round(0.99 * r.L1_m, 3), round(1.01 * r.L1_m, 3)))]



class ReductionTableWidget(QtWidgets.QWidget):
    """Reduction table panel: one tab per geometry (detector distance, wavelength, collimation),
    filled from the runs of the folder or by hand. Geometries measured only for tests can be removed
    from the reduction completely (the raw files stay untouched)."""

    changed = QtCore.Signal()
    rows_selected = QtCore.Signal()          # sample rows selected (or another geometry tab shown)

    def __init__(self, entries: Callable[[], list], centre: Callable[[object], tuple], parent=None, log=None):
        """``centre(raw)``: the beam centre of the geometry of a raw run."""
        super().__init__(parent)
        self._entries = entries
        self._centre = centre
        self._log = log or (lambda s: None)
        self.plan = None
        self.reduce_rows = None                    # set by the main window: (table, rows) -> reduce
        self.show_efficiency = None                # set by the main window: (table) -> show its efficiency map
        self.excluded: List[str] = []              # geometries removed from the reduction
        lay = QtWidgets.QVBoxLayout(self)
        top = QtWidgets.QHBoxLayout()
        self.check = QtWidgets.QCheckBox("check repeated runs")
        self.check.setChecked(True)
        self.check.setToolTip("Compare repeated runs (equivalence check, outliers) when filling")
        fill = QtWidgets.QPushButton("Fill from runs")
        fill.setToolTip("Group the runs of the folder by metadata, one tab per geometry, roles from the sample names")
        fill.clicked.connect(self.on_fill)
        new = QtWidgets.QPushButton("New geometry (by hand)")
        new.setToolTip("An empty table, filled completely by hand")
        new.clicked.connect(self.on_clear)
        rem = QtWidgets.QPushButton("Remove geometry")
        rem.setToolTip("Remove the geometry shown from the reduction (e.g. runs taken for tests); the files stay")
        rem.clicked.connect(lambda: self.remove_geometry(self.tabs.currentIndex()))
        self.restore = QtWidgets.QPushButton("Restore removed")
        self.restore.setToolTip("Put the removed geometries back (Fill from runs again)")
        self.restore.clicked.connect(self.on_restore)
        load = QtWidgets.QPushButton("Load…")
        load.clicked.connect(self.on_load)
        save = QtWidgets.QPushButton("Save…")
        save.clicked.connect(self.on_save)
        for w in (self.check, fill, new, rem, self.restore, load, save):
            top.addWidget(w)
        top.addStretch(1)
        lay.addLayout(top)
        self.tabs = QtWidgets.QTabWidget()
        self.tabs.setTabsClosable(True)
        self.tabs.tabCloseRequested.connect(lambda i: self.remove_geometry(i))
        lay.addWidget(self.tabs, 1)
        self.info = QtWidgets.QLabel()
        self.info.setWordWrap(True)
        lay.addWidget(self.info)
        self._add_page(ReductionTable(config="by hand"))
        self._update_info()

    # ------------------------------------------------------------------ pages
    def pages(self) -> List[GeometryPage]:
        return [self.tabs.widget(i) for i in range(self.tabs.count())]

    def show_missing(self, config: str, role: str, index, field: str) -> bool:
        """Bring the geometry tab of ``config`` to the front and select the cell of a missing input."""
        for p in self.pages():
            if p.table.config == config:
                self.tabs.setCurrentWidget(p)
                p.show_missing(role, index, field)
                return True
        return False

    @property
    def page(self) -> Optional[GeometryPage]:
        return self.tabs.currentWidget()

    # the shown geometry (scripts, tests)
    @property
    def table(self) -> ReductionTable:
        return self.page.table

    @property
    def corr(self):
        return self.page.corr

    @property
    def samp(self):
        return self.page.samp

    def tables(self) -> List[ReductionTable]:
        return [p.table for p in self.pages()]

    def _add_page(self, table: ReductionTable, ref_raw=None, title=None, tip="") -> GeometryPage:
        p = GeometryPage(self, table, ref_raw)
        i = self.tabs.addTab(p, title or table.config or "geometry")
        self.tabs.setTabToolTip(i, tip or table.config)
        for q in self.pages():              # the efficiency-map lists offer the water of every geometry
            q._update_scale_box()
        return p

    def _clear_pages(self):
        while self.tabs.count():
            w = self.tabs.widget(0)
            self.tabs.removeTab(0)
            w.deleteLater()

    def _update_info(self):
        n = self.tabs.count()
        txt = f"{n} geometr{'y' if n == 1 else 'ies'}"
        if self.excluded:
            txt += "; removed: " + "; ".join(self.excluded)
        self.info.setText(txt)
        self.restore.setEnabled(bool(self.excluded))

    def raws(self) -> Dict[int, object]:
        return {e.run: e.raw for e in self._entries() if getattr(e, "raw", None) is not None}

    # ------------------------------------------------------------------ geometry actions
    def remove_geometry(self, i: int, confirm: bool = True):
        if not (0 <= i < self.tabs.count()):
            return
        p = self.tabs.widget(i)
        cfg = p.table.config
        if confirm:
            n = sum(len(r.runs) for r in p.table.rows())
            ans = QtWidgets.QMessageBox.question(
                self, "Remove geometry", f"Remove {cfg} ({n} runs in the table) from the reduction?\n"
                "The raw files are not touched; 'Restore removed' brings it back.")
            if ans != QtWidgets.QMessageBox.StandardButton.Yes:
                return
        self.tabs.removeTab(i)
        p.deleteLater()
        if cfg and cfg != "by hand" and cfg not in self.excluded:
            self.excluded.append(cfg)
        self._log(f"geometry removed from the reduction: {cfg}")
        self._update_info()
        self.changed.emit()

    def on_restore(self):
        self.excluded = []
        if self.plan is not None:
            self._fill_tabs()
        self._update_info()

    # ------------------------------------------------------------------ filling
    def _edges(self, r):
        import numpy as np

        from .core.averaging import q_edges
        lo = 4 * np.pi / r.wavelength_A * np.sin(0.5 * np.arctan(4 * r.pixel_size_m[0] / r.L2_m))
        hi = 4 * np.pi / r.wavelength_A * np.sin(0.5 * np.arctan(60 * r.pixel_size_m[0] / r.L2_m))
        return q_edges(lo, hi, 30)

    def _trans_table(self):
        import os

        from .io.transmission import read_transmission_summary
        es = self._entries()
        if not es:
            return None
        p = os.path.join(os.path.dirname(es[0].path), "trans.txt")
        return read_transmission_summary(p) if os.path.isfile(p) else None

    def check_group(self, g):
        """Equivalence check of the runs of a group with the centre of its geometry."""
        from . import grouping as gr
        from .core import mask as mk
        r0 = g.runs[0]
        c = self._centre(r0)
        return gr.check_runs(g, c, self._edges(r0), mask=mk.suggest_mask(r0.counts, c)[0])

    def on_fill(self):
        from .plan import build_plan
        entries = self._entries()
        if not entries:
            self._log("no runs: open a raw-data folder first")
            return
        kw = {}
        if self.check.isChecked():
            from .core import mask as mk
            kw = dict(beam_center_px=self._centre, edges=self._edges,
                      mask=lambda r, c: mk.suggest_mask(r.counts, c)[0])
        QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.CursorShape.WaitCursor)
        try:
            self.plan = build_plan(entries, **kw)
        finally:
            QtWidgets.QApplication.restoreOverrideCursor()
        self._fill_tabs()
        for m in self.plan.missing:
            self._log("missing: " + m)

    def _fill_tabs(self):
        import os
        raws = self.raws()
        trans = self._trans_table()
        es = self._entries()
        folder = os.path.dirname(es[0].path) if es else ""
        prev = {t.config: t for t in self.tables() if t.source == "plan" and t.folder == folder}
        self._clear_pages()
        for cfg in configurations(self.plan):
            if cfg in self.excluded:
                continue
            t = from_plan(self.plan, cfg, raws, trans)
            t.folder = folder
            kept = carry_over(t, prev.get(cfg))
            if kept:
                self._log(f"{cfg}: kept from the table before – {kept}")
            ents = [e for e in self.plan.entries if e.config == cfg and e.kind == "scattering"]
            n = sum(len(e.runs) for e in ents)
            ref = raws.get(ents[0].runs[0]) if ents else None
            short = cfg.replace(", L1", " · L1").replace(", λ", " · λ")
            have = {e.role for e in ents}
            doubt = n < 4 or not {"water", "empty_cell"} <= have
            title = f"{short} ({n} run{'' if n == 1 else 's'})" + ("  ?" if doubt else "")
            tip = cfg + f": {n} scattering runs, {len(t.samples)} samples"
            if doubt:
                tip += ("\nFew runs or no water / empty cell: measured for tests? "
                        "Remove the geometry with the × of the tab.")
            self._add_page(t, ref, title, tip)
        if not self.tabs.count():
            self._add_page(ReductionTable(config="by hand"))
        self._update_info()
        self.changed.emit()

    def on_clear(self):
        """A new tab for a geometry filled completely by hand."""
        n = sum(1 for p in self.pages() if p.table.source == "manual") + 1
        p = self._add_page(ReductionTable(config=f"by hand {n}"))
        self.tabs.setCurrentWidget(p)
        self._update_info()
        self.changed.emit()

    # the shown geometry
    def on_add_many(self):
        self.page.on_add_many()

    def add_groups(self, groups, check: bool = False):
        self.page.add_groups(groups, check)

    def on_copy(self, direction):
        self.page.on_copy(direction)

    # ------------------------------------------------------------------ files
    def to_dict(self) -> dict:
        return {"geometries": [t.to_dict() for t in self.tables()], "excluded": list(self.excluded)}

    def from_dict(self, d: dict):
        self._clear_pages()
        if "geometries" not in d:                      # one table (first version)
            d = {"geometries": [d], "excluded": []}
        self.excluded = list(d.get("excluded", []))
        raws = self.raws()
        for td in d["geometries"]:
            t = ReductionTable.from_dict(td)
            ref = next((raws[c.run] for r in t.rows() for c in r.runs if c.run in raws), None)
            self._add_page(t, ref)
        if not self.tabs.count():
            self._add_page(ReductionTable(config="by hand"))
        self._update_info()
        self.changed.emit()

    def save(self, path) -> str:
        import json
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(self.to_dict(), fh, indent=1)
        return str(path)

    def load(self, path):
        import json
        with open(path, encoding="utf-8") as fh:
            self.from_dict(json.load(fh))

    def on_save(self):
        p, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save reduction table", "reduction_table.json",
                                                     "JSON (*.json)")
        if p:
            self.save(p)
            self._log(f"reduction table saved to {p}")

    def on_load(self):
        p, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Load reduction table", "", "JSON (*.json)")
        if p:
            self.load(p)
