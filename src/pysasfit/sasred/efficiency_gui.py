# SPDX-License-Identifier: BSD-3-Clause
"""Detector efficiency maps in the pySASred window (mixed into :class:`~pysasfit.sasred.gui.MainWindow`).

* The map of a geometry's reduction table (``ReductionTable.efficiency``): ``"geometry:<config>"`` – made
  from the water rows (water, water background, Cd) of that geometry's table, kept for the session – or
  an HDF5 file (``file.h5`` or ``file.h5::entry`` of an efficiency archive).
* The **shown map** of the detector panel: the view "efficiency map" draws it (estimated pixels outlined),
  "÷ efficiency" divides the raw image by it; "÷ solid angle" gives counts per steradian.
"""
from __future__ import annotations

import os
from typing import Dict, Optional

import numpy as np
from PySide6 import QtCore, QtWidgets

__all__ = ["EfficiencyMixin", "GEOMETRY_PREFIX"]

GEOMETRY_PREFIX = "geometry:"


class EfficiencyMixin:
    """Needs ``entries``, ``name_corrections``, ``centre_for``, ``mask_for``, ``table_panel``, ``log``,
    ``det_view``, ``_draw_detector`` and ``norm`` of the main window."""

    _eff_cache: Dict[tuple, object]
    shown_efficiency = None                 # the map of the detector panel
    shown_efficiency_name = ""

    # ------------------------------------------------------------------ reading runs
    def run_reader(self, bank: str = "", cache: Optional[dict] = None):
        """``read(run)`` of the runs of the folder (names corrected by hand; a SANS-LLB bank when given)."""
        from . import names as nm
        from .io.raw import read_raw
        paths = {e.run: e.path for e in self.entries}
        cache = {} if cache is None else cache

        def read(n):
            if n not in paths:
                raise KeyError(f"run {n} not in the folder")
            if n not in cache:
                cache[n] = nm.apply(read_raw(paths[n]), self.name_corrections)
            r = cache[n]
            if bank:
                from .io.sans_llb import bank_run
                return bank_run(r, bank)
            return r
        return read

    # ------------------------------------------------------------------ maps
    def _table_of(self, config: str):
        for t in self.table_panel.tables():
            if t.config == config:
                return t
        return None

    def efficiency_from_table(self, table, bank: str = "", read=None):
        """Efficiency map from the water rows of a reduction table (cached for the same runs, transmissions,
        beam centre and mask)."""
        from .core import mask as mk
        from .reduce import efficiency_from_runs
        corr = table.corrections
        w, ec, cd = corr["water"], corr["empty_cell"], corr["blocked_beam"]
        uses = getattr(table, "uses", lambda role: bool(corr[role].used))
        ec_runs = list(ec.used) if uses("empty_cell") else []          # switched off ('Use') or no runs: left out
        cd_runs = list(cd.used) if uses("blocked_beam") else []
        if not w.used:
            raise ValueError(f"{table.config}: no water runs for an efficiency map")
        if w.transmission is None:
            raise ValueError(f"{table.config}: the water has no transmission")
        read = read or self.run_reader(bank)
        raw0 = read(w.used[0])
        centre = self.centre_for(raw0)
        mask, how = self.mask_for(raw0)
        key = (table.config, bank, tuple(w.used), tuple(ec_runs), tuple(cd_runs), w.transmission,
               ec.transmission, tuple(map(float, centre)), mask.tobytes().__hash__())
        if not hasattr(self, "_eff_cache"):
            self._eff_cache = {}
        if key in self._eff_cache:
            return self._eff_cache[key]
        if ec_runs and ec.transmission is None:
            raise ValueError(f"{table.config}: the water background has no transmission")
        # the efficiency needs the mask of the detector defects and the beam stop, not the dead-time flags
        mask = np.asarray(mask, np.uint32) & ~np.uint32(8)
        eff = efficiency_from_runs([read(n) for n in w.used], [read(n) for n in ec_runs] or None, centre,
                                   (w.transmission, w.transmission_sigma),
                                   (ec.transmission, ec.transmission_sigma) if ec_runs else None,
                                   mask=mask, blocked=[read(n) for n in cd_runs] or None,
                                   normalise_by=self.norm.currentData())
        eff.meta["file"] = f"water of {table.config}" + (f" ({bank})" if bank else "")
        eff.meta["mask"] = how
        eff.meta["corrections"] = (
            "[(W − Cd)/T_W" + (" − (EC − Cd)/T_EC" if ec_runs else "") + "] / ΔΩ, normalised to mean 1; "
            f"water runs {w.runs_text()} (T_W = {w.transmission:.4f}), "
            + (f"water background runs {ec.runs_text()} (T_EC = {ec.transmission:.4f})" if ec_runs else
               "no water background" + (" (switched off)" if ec.used else ""))
            + ", " + (f"Cd runs {cd.runs_text()} subtracted from both" if cd_runs else
                      "no Cd" + (" (switched off)" if cd.used else "")) + "; each run dead-time, attenuator and "
            "monitor corrected")
        self._eff_cache[key] = eff
        return eff

    def efficiency_for(self, table, bank: str = "", read=None):
        """The efficiency map chosen for a reduction table (None: none chosen)."""
        spec = (getattr(table, "efficiency", "") or "").strip()
        if not spec:
            return None
        if spec.startswith(GEOMETRY_PREFIX):
            src = self._table_of(spec[len(GEOMETRY_PREFIX):])
            if src is None:
                raise ValueError(f"efficiency map: no reduction table of {spec[len(GEOMETRY_PREFIX):]}")
            return self.efficiency_from_table(src, bank)
        from .absolute import load_efficiency
        path = spec.partition("::")[0].strip()
        if not os.path.isabs(path) and getattr(self, "folder", ""):
            path = os.path.join(self.folder, path)
            spec = path + ("::" + spec.partition("::")[2] if "::" in spec else "")
        if not os.path.isfile(path):
            raise FileNotFoundError(f"efficiency map file {path} not found")
        key = ("file", spec, os.path.getmtime(path))
        if not hasattr(self, "_eff_cache"):
            self._eff_cache = {}
        if key not in self._eff_cache:
            self._eff_cache[key] = load_efficiency(spec)
        return self._eff_cache[key]

    def show_efficiency(self, eff, name: str = ""):
        """Put a map into the detector panel and switch to the view "efficiency map"."""
        self.shown_efficiency = eff
        self.shown_efficiency_name = name or str((eff.meta or {}).get("file", ""))
        self._update_eff_label()
        i = self.det_view.findData("efficiency")
        if i >= 0 and self.det_view.currentIndex() != i:
            self.det_view.setCurrentIndex(i)          # redraws
        else:
            self._draw_detector()
        d = getattr(self, "docks", {}).get("detector")
        if d is not None:
            d.raise_()

    def show_table_efficiency(self, table):
        """Button "Show" of the reduction table: the map chosen there in the detector panel."""
        QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.CursorShape.WaitCursor)
        try:
            eff = self.efficiency_for(table)
        except Exception as exc:  # noqa: BLE001
            QtWidgets.QApplication.restoreOverrideCursor()
            self._eff_problem(f"map for {table.config} ({table.efficiency}): {type(exc).__name__}: {exc}")
            return None
        QtWidgets.QApplication.restoreOverrideCursor()
        if eff is None:
            self._eff_problem(f"{table.config}: no efficiency map chosen – choose 'water of …' or 'file…' in "
                              "Absolute scale → efficiency map")
            return None
        self.show_efficiency(eff, table.efficiency)
        return eff

    def _update_eff_label(self):
        lab = getattr(self, "eff_label", None)
        if lab is None:
            return
        e = self.shown_efficiency
        if e is None:
            lab.setText("no efficiency map")
            lab.setToolTip("")
        else:
            m = e.meta or {}
            lab.setText(f"map: {self.shown_efficiency_name or 'loaded'}")
            lab.setToolTip(f"{self.shown_efficiency_name}\nshape {e.shape}, water runs {m.get('water_runs', '?')}, "
                           f"SD {m.get('SD_m', '?')} m, λ {m.get('wavelength_A', '?')} Å")

    def on_load_efficiency(self):
        p, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Efficiency map", getattr(self, "folder", "") or "",
                                                     "HDF5 (*.h5 *.hdf5 *.nxs)")
        if not p:
            return
        from .absolute import load_efficiency
        try:
            eff = load_efficiency(p)
        except Exception as exc:  # noqa: BLE001
            QtWidgets.QMessageBox.critical(self, "Efficiency map", f"{type(exc).__name__}: {exc}")
            return
        self.show_efficiency(eff, p)

    def _eff_problem(self, text):
        """A failed efficiency action: log and a message box (not only the log)."""
        self.log("efficiency map: " + text)
        if not getattr(self, "_quiet", False):
            QtWidgets.QMessageBox.warning(self, "Efficiency map", text)

    def geometries_with_water(self):
        """Reduction tables whose Water row has runs."""
        return [t for t in self.table_panel.tables() if t.corrections["water"].used]

    def fill_make_menu(self, menu):
        """Submenu 'Make from the water of': one entry per geometry with water (the bank of the detector panel
        for SANS-LLB)."""
        menu.clear()
        tabs = self.geometries_with_water()
        if not tabs:
            a = menu.addAction("no geometry with water runs (open a folder, fill the reduction table)")
            a.setEnabled(False)
            return
        for t in tabs:
            w = t.corrections["water"]
            menu.addAction(f"{t.config}  (water runs {w.runs_text()})",
                           lambda t=t: self.make_efficiency(t))

    def make_efficiency(self, table, bank: Optional[str] = None):
        """Map from the water rows of ``table``, shown in the detector panel; errors in a message box."""
        if bank is None:
            bank = self.det_bank.currentData() if hasattr(self, "det_bank") else ""
            w = table.corrections["water"]
            if bank and w.used:
                try:
                    r = self.run_reader("")(w.used[0])
                    if getattr(r, "instrument", "") != "sans_llb" or bank not in (r.panels or {}):
                        bank = ""
                except Exception:  # noqa: BLE001
                    bank = ""
        if not table.corrections["water"].used:
            self._eff_problem(f"{table.config} has no water runs. Choose a geometry with water (usually a "
                              "shorter distance) in 'efficiency map → Make from the water of'.")
            return None
        if table.corrections["water"].transmission is None:
            self._eff_problem(f"{table.config}: the water has no transmission – enter it in the Water row of the "
                              "reduction table (or let it be computed from its transmission runs).")
            return None
        QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.CursorShape.WaitCursor)
        try:
            eff = self.efficiency_from_table(table, bank or "")
        except Exception as exc:  # noqa: BLE001
            self._eff_problem(f"{table.config}: {type(exc).__name__}: {exc}")
            return None
        finally:
            QtWidgets.QApplication.restoreOverrideCursor()
        m = eff.meta
        self.log(f"efficiency map from the water of {table.config}{' (' + bank + ')' if bank else ''}: water runs "
                 f"{m.get('water_runs')}, {m.get('n_filled', 0)} pixels estimated, {m.get('n_rejected', 0)} rejected "
                 f"at the beam stop edge, {m.get('n_undefined', 0)} undefined")
        self.log("    " + str(m.get("corrections", "")))
        self.show_efficiency(eff, f"{GEOMETRY_PREFIX}{table.config}")
        self.statusBar().showMessage(f"efficiency map made from the water of {table.config}: shown in the detector "
                                     "panel; choose it for a geometry in the reduction table (Absolute scale)", 15000)
        return eff

    def on_make_efficiency(self):
        """Map from the water of the geometry of the shown run (no run shown: of the reduction-table tab shown);
        when that geometry has no water: the only geometry with water, else a message naming them."""
        tables = self.table_panel.tables()
        if not tables:
            self._eff_problem("no reduction table yet: open a raw-data folder (the table is filled from its runs)")
            return None
        raw = self.displayed_raw()
        t = None
        if raw is not None:
            from .plan import _config
            t = next((x for x in tables if x.config == _config(raw)), None)
        if raw is None:                       # no run shown: the reduction-table tab shown
            page = self.table_panel.page
            if page is not None and page.table.corrections["water"].used:
                t = page.table
        if t is None or not t.corrections["water"].used:
            wt = self.geometries_with_water()
            if len(wt) == 1:
                t = wt[0]
            elif wt:
                self._eff_problem("the shown geometry has no water. Choose the geometry whose water makes the map "
                                  "in 'efficiency map → Make from the water of': " + "; ".join(x.config for x in wt))
                return None
            else:
                self._eff_problem("no geometry has water runs: an efficiency map needs a water measurement")
                return None
        return self.make_efficiency(t)

    def on_save_efficiency(self):
        e = self.shown_efficiency
        if e is None:
            self.log("no efficiency map shown")
            return
        p, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save efficiency map", getattr(self, "folder", "") or "",
                                                     "HDF5 (*.h5)")
        if not p:
            return
        e.save(p, description=self.shown_efficiency_name)
        self.log(f"efficiency map saved to {p}")

    def on_archive_efficiency(self):
        e = self.shown_efficiency
        if e is None:
            self.log("no efficiency map shown")
            return
        from .efficiency_log import archive_efficiency, archive_path
        try:
            name = archive_efficiency(e)
        except Exception as exc:  # noqa: BLE001
            self.log(f"archive: {type(exc).__name__}: {exc}")
            return
        self.log(f"efficiency map archived as {name} in {archive_path()}")

    # ------------------------------------------------------------------ display
    def display_image(self, raw, c):
        """The raw image of the detector panel with the chosen corrections: (image, unit, note)."""
        unit, note = "counts", ""
        if getattr(self, "det_eff", None) is not None and self.det_eff.isChecked():
            e = self.shown_efficiency
            if e is None:
                note = " – no efficiency map shown (load or make one)"
            elif e.shape != c.shape:
                note = f" – efficiency map {e.shape} does not fit {c.shape}"
            else:
                v = np.asarray(e.value.value, float)
                c = np.where(e.undefined | (v <= 0), np.nan, c / np.where(v > 0, v, 1.0))
                unit = "counts / efficiency"
        if getattr(self, "det_domega", None) is not None and self.det_domega.isChecked():
            from .core.geometry import run_geometry
            g = run_geometry(raw, self.centre_for(raw))
            c = c / g.d_omega
            unit = unit + " / sr"
        return c, unit, note

    def _draw_efficiency(self, fig):
        """The shown efficiency map: value per pixel, estimated pixels outlined, undefined grey."""
        import matplotlib
        from matplotlib.colors import TwoSlopeNorm
        ax = fig.add_subplot(111)
        e = self.shown_efficiency
        if e is None:
            ax.text(0.5, 0.5, "no efficiency map: 'efficiency map' → Load… or Make from water",
                    ha="center", va="center", transform=ax.transAxes)
            return
        v = np.array(e.value.value, float)
        v[e.undefined] = np.nan
        ok = np.isfinite(v)
        lo, hi = (np.nanpercentile(v[ok], [1, 99]) if ok.any() else (0.9, 1.1))
        lo, hi = min(lo, 0.999), max(hi, 1.001)
        cmap = matplotlib.colormaps["RdBu_r"].with_extremes(bad="0.8")
        im = ax.imshow(v, origin="lower", cmap=cmap, norm=TwoSlopeNorm(1.0, lo, hi), interpolation="nearest")
        from ..toolbox.viewer import image_colorbar
        image_colorbar(fig, ax, im, "relative efficiency")
        if e.filled.any():
            ax.contour(e.filled.astype(float), levels=[0.5], colors="k", linewidths=0.8)
        m = e.meta or {}
        ax.set_xlabel("x [pixel]")
        ax.set_ylabel("y [pixel]")
        std = np.nanmedian(np.where(e.measured, e.std(), np.nan)) if e.measured.any() else float("nan")
        ax.set_title(f"efficiency: {self.shown_efficiency_name}\nwater runs {m.get('water_runs', '?')}, "
                     f"SD {m.get('SD_m', float('nan')):.2f} m, λ {m.get('wavelength_A', float('nan')):.2f} Å; "
                     f"rms {np.nanstd(v[e.measured]) if e.measured.any() else float('nan'):.3f}, median σ {std:.3f}; "
                     "outlined: estimated (row × column)", fontsize="small")
