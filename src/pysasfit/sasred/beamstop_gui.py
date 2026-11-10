# SPDX-License-Identifier: BSD-3-Clause
"""pySASred: check the transmission measured through a semi-transparent beam stop (Reduction menu).

For every row of the reduction table shown (geometry tab), the counts of its scattering runs in the
transmission region of the configuration (second layer of the mask editor) give I_bs; divided by I_bs of a
reference row (empty beam, or empty cell) they give T_bs, listed next to the transmission of the table
(from the transmission runs, beam stop out) rescaled to the same reference. The ratio and z = difference /
σ show whether T_bs can stand in for the transmission runs. Nothing in the table is changed.
"""
from __future__ import annotations

import csv
import os
from typing import List, Optional

import numpy as np
from PySide6 import QtCore, QtWidgets

from .core.beamstop_transmission import beamstop_transmission, compare, region_intensity

__all__ = ["BeamstopTransmissionDialog", "check_rows"]

HEADERS = ["row", "sample", "runs", "region px", "counts", "I_bs", "T_bs", "σ(T_bs)", "T (transmission runs)",
           "σ(T)", "T_bs / T", "σ", "z", "note"]


def check_rows(win, table, ref_index: int, normalise_by: str = "monitor") -> List[dict]:
    """One result per row of ``table`` with scattering runs: I_bs, T_bs against the reference row
    ``ref_index`` (index into ``table.rows()``), the table's transmission rescaled to that reference."""
    raws = {e.run: e.raw for e in win.entries if e.raw is not None}
    rows = [r for r in table.rows() if r.used]
    all_rows = table.rows()
    ref = all_rows[ref_index]
    ref_raws = [raws[n] for n in ref.used if n in raws]
    if not ref_raws:
        raise ValueError(f"reference row {ref.label}: no scattering runs in the folder")
    from .catalogue import mask_key, mask_key_label
    region = win.region_for(ref_raws[0])
    if region is None:
        raise ValueError(f"no transmission region for {mask_key_label(mask_key(ref_raws[0]))}: draw it in the "
                         "detector panel (mask tools, edit: transmission region)")
    k_ref = mask_key(ref_raws[0])
    inst = win.instrument_for(ref_raws[0])
    t_ref = ref.transmission if ref.transmission is not None else None
    s_ref = ref.transmission_sigma or 0.0
    out = []
    for r in rows:
        res = {"row": r.label, "sample": r.info, "runs": r.runs_text(), "note": ""}
        rr = [raws[n] for n in r.used if n in raws]
        try:
            if not rr:
                raise ValueError("runs not in the folder")
            other = sorted({mask_key_label(mask_key(x)) for x in rr if mask_key(x) != k_ref})
            if other:
                res["note"] = "other beam stop / configuration than the reference: " + "; ".join(other)
            bt = beamstop_transmission(rr, ref_raws, region, inst, normalise_by)
            res.update(n_px=bt.sample.n_pixels, counts=bt.sample.counts, I=bt.sample.I, I_unit=bt.sample.unit,
                       T=bt.value, sT=bt.sigma, sT_total=bt.sigma_total)
            if r.transmission is not None:
                t, s = float(r.transmission), float(r.transmission_sigma or 0.0)
                if r is not ref and t_ref:                 # rescaled to the reference row (e.g. the empty cell)
                    s = (t / t_ref) * np.hypot(s / t if t else 0.0, s_ref / t_ref)
                    t = t / t_ref
                elif r is ref:
                    t, s = 1.0, 0.0
                res.update(Tr=t, sTr=s)
                ratio, s_ratio, z = compare(bt.value, bt.sigma, t, s)
                res.update(ratio=ratio, s_ratio=s_ratio, z=z)
        except Exception as exc:  # noqa: BLE001 -- shown in the row
            res["note"] = f"{type(exc).__name__}: {exc}"
        out.append(res)
    return out


def _f(v, fmt):
    return "" if v is None or (isinstance(v, float) and not np.isfinite(v)) else format(v, fmt)


class BeamstopTransmissionDialog(QtWidgets.QDialog):
    """Table of T_bs for the rows of the reduction table shown, against a chosen reference row."""

    def __init__(self, win, parent=None):
        super().__init__(parent or win)
        self.win = win
        self.setWindowTitle("Transmission through the semi-transparent beam stop")
        self.resize(1100, 520)
        lay = QtWidgets.QVBoxLayout(self)
        self.info = QtWidgets.QLabel()
        self.info.setWordWrap(True)
        lay.addWidget(self.info)
        row = QtWidgets.QHBoxLayout()
        row.addWidget(QtWidgets.QLabel("reference (T = 1):"))
        self.ref = QtWidgets.QComboBox()
        self.ref.setToolTip("Row whose runs give I_bs of the unattenuated beam: the empty beam (beam stop in) "
                            "or the empty cell (T_bs is then relative to the cell; the transmission of the table is "
                            "rescaled to the same reference)")
        row.addWidget(self.ref, 1)
        b = QtWidgets.QPushButton("Calculate")
        b.setDefault(True)
        b.clicked.connect(self.calculate)
        row.addWidget(b)
        lay.addLayout(row)
        self.view = QtWidgets.QTableWidget(0, len(HEADERS))
        self.view.setHorizontalHeaderLabels(HEADERS)
        self.view.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
        self.view.horizontalHeader().setStretchLastSection(True)
        lay.addWidget(self.view, 1)
        bb = QtWidgets.QHBoxLayout()
        for text, slot in (("Copy", self.copy), ("Save CSV…", self.save), ("Close", self.close)):
            x = QtWidgets.QPushButton(text)
            x.clicked.connect(slot)
            bb.addWidget(x)
        bb.insertStretch(0, 1)
        lay.addLayout(bb)
        self.results = []
        self.fill_references()

    @property
    def table(self):
        page = self.win.table_panel.page
        return None if page is None else page.table

    def fill_references(self):
        from .plan import role_of
        self.ref.clear()
        t = self.table
        if t is None:
            self.info.setText("No reduction table: fill it first (Reduction → Fill table from runs).")
            return
        default = None
        for i, r in enumerate(t.rows()):
            if not r.used:
                continue
            self.ref.addItem(f"{r.label}: {r.info} (runs {r.runs_text()})", i)
            if default is None and (r.role == "empty_beam" or role_of(r.info or "") == "empty_beam"):
                default = self.ref.count() - 1
        if default is None:
            default = next((j for j in range(self.ref.count())
                            if t.rows()[self.ref.itemData(j)].role == "empty_cell"), 0)
        self.ref.setCurrentIndex(default)
        self.info.setText(f"Geometry {t.config}. I_bs: counts in the transmission region of the configuration "
                          f"(green in the detector panel), dead time and attenuator corrected, per "
                          f"{'monitor' if self.win.norm.currentData() == 'monitor' else 'second'}; "
                          "T_bs = I_bs / I_bs(reference). T: the transmission of the table (transmission runs), "
                          "rescaled to the reference. z = (T_bs − T) / σ.")

    def calculate(self):
        t = self.table
        if t is None or self.ref.currentIndex() < 0:
            return
        QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.CursorShape.WaitCursor)
        try:
            self.results = check_rows(self.win, t, self.ref.currentData(), self.win.norm.currentData())
        except Exception as exc:  # noqa: BLE001
            QtWidgets.QApplication.restoreOverrideCursor()
            QtWidgets.QMessageBox.information(self, "Beam-stop transmission", str(exc))
            return
        QtWidgets.QApplication.restoreOverrideCursor()
        self.view.setRowCount(len(self.results))
        for i, r in enumerate(self.results):
            vals = self.values(r)
            for j, v in enumerate(vals):
                it = QtWidgets.QTableWidgetItem(v)
                if j == 12 and r.get("z") is not None and abs(r["z"]) > 3:
                    it.setForeground(QtCore.Qt.GlobalColor.red)
                self.view.setItem(i, j, it)
        self.view.resizeColumnsToContents()
        self.win.log(f"beam-stop transmission, {t.config}, reference {self.ref.currentText()}:")
        for r in self.results:
            self.win.log("    " + " | ".join(f"{h} {v}" for h, v in zip(HEADERS, self.values(r)) if v))

    @staticmethod
    def values(r) -> List[str]:
        return [r["row"], r.get("sample", ""), r.get("runs", ""), _f(r.get("n_px"), "d"), _f(r.get("counts"), ".0f"),
                _f(r.get("I"), ".5g"), _f(r.get("T"), ".4f"), _f(r.get("sT"), ".4f"), _f(r.get("Tr"), ".4f"),
                _f(r.get("sTr"), ".4f"), _f(r.get("ratio"), ".4f"), _f(r.get("s_ratio"), ".4f"),
                _f(r.get("z"), "+.1f"), r.get("note", "")]

    def text(self) -> str:
        return "\n".join(["\t".join(HEADERS)] + ["\t".join(self.values(r)) for r in self.results])

    def copy(self):
        QtWidgets.QApplication.clipboard().setText(self.text())

    def save(self, path: Optional[str] = None):
        if path is None:
            path, _ = QtWidgets.QFileDialog.getSaveFileName(
                self, "Save beam-stop transmissions", os.path.join(self.win.folder or os.getcwd(),
                                                                   "beamstop_transmission.csv"), "CSV (*.csv)")
            if not path:
                return None
        with open(path, "w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(HEADERS)
            for r in self.results:
                w.writerow(self.values(r))
        return path
