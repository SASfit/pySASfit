# SPDX-License-Identifier: BSD-3-Clause
"""SANS-LLB efficiency maps: one map per detector bank, determined from the water on that bank (panels relative
to the central detector), and kept per bank in files, in the archive and in the detector panel."""
import os
import sys

import h5py
import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(__file__))
from test_sans_llb import _nominal_llb_geometry, write_llb  # noqa: E402,F401 -- autouse fixture: offsets 0


def _water_folder(tmp_path, rng):
    """Water (run 100) and a sample (run 101); the panels get tube gains × a profile along the tubes."""
    d = tmp_path / "raw"
    d.mkdir()
    gains = {}
    for run, name in ((100, "H2O"), (101, "Protein")):
        p = write_llb(d / f"sans-llb2026n{run:06d}.hdf", name, seed=run)
        with h5py.File(p, "r+") as f:
            for b in ("left", "bottom"):
                g = f[f"entry0/SANS-LLB/{b}_detector"]
                a = g["data"][()].astype(float)
                if b not in gains:
                    tubes = 1 + 0.15 * rng.standard_normal(16)
                    along = 1 + 0.1 * np.sin(np.linspace(0, 3, 136))
                    gains[b] = tubes[None, :] * along[:, None] if b == "left" else tubes[:, None] * along[None, :]
                del g["data"]
                g.create_dataset("data", data=np.round(a * gains[b]).astype(np.uint32))
    return str(d), gains


def test_panel_map_recovers_tube_gains(tmp_path):
    from pysasfit.sasred.core import mask as mk
    from pysasfit.sasred.io.raw import read_raw
    from pysasfit.sasred.io.sans_llb import bank_run
    from pysasfit.sasred.reduce import efficiency_from_runs
    d, gains = _water_folder(tmp_path, np.random.default_rng(3))
    raw = read_raw(os.path.join(d, "sans-llb2026n000100.hdf"))
    for b in ("left", "bottom"):
        r = bank_run(raw, b)
        m, _ = mk.default_mask(r, (63.5, 63.5))
        e = efficiency_from_runs([r], None, (63.5, 63.5), (0.5, 0.005), mask=m)
        assert e.shape == r.counts.shape and e.bank == b and e.meta["detector_distance_m"] == pytest.approx(1.02)
        ok = e.measured
        ratio = np.asarray(e.value.value)[ok] / (gains[b][ok] / gains[b][ok].mean())
        assert abs(ratio.mean() - 1) < 2e-3 and ratio.std() < 0.035     # Poisson of ~1600 counts: 0.025


def test_maps_of_all_banks_made_saved_loaded_archived(tmp_path, monkeypatch):
    pytest.importorskip("PySide6")
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    monkeypatch.setenv("PYSASFIT_SETTINGS", str(tmp_path / "settings.json"))
    from PySide6 import QtWidgets
    from pysasfit.sasred import efficiency_log as el
    from pysasfit.sasred.absolute import load_efficiency
    from pysasfit.sasred.core.efficiency import Efficiency
    from pysasfit.sasred.gui import MainWindow
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    d, _ = _water_folder(tmp_path, np.random.default_rng(5))
    w = MainWindow()
    w._quiet = True
    try:
        w.set_folder(d)
        app.processEvents()
        t = w.table_panel.tables()[0]
        wr = t.corrections["water"]
        assert wr.used == [100]
        wr.transmission, wr.transmission_sigma = 0.5, 0.005
        w.det_bank.setCurrentIndex(w.det_bank.findData("left"))
        w.make_efficiency(t)
        maps = w.shown_efficiencies
        assert set(maps) == {"", "left", "bottom"} and w.shown_efficiency.bank == "left"
        assert maps["left"].shape == (136, 16) and maps[""].shape == (128, 128)
        for b in ("left", "bottom"):                 # relative to the central detector: water / ΔΩ ratio
            rel = maps[b].meta["relative_to_central"]
            assert rel == pytest.approx(maps[b].meta["water_mean"] / maps[""].meta["water_mean"])
            assert np.mean(np.asarray(maps[b].value.value)[maps[b].measured]) == pytest.approx(rel, rel=1e-6)
        w.det_bank.setCurrentIndex(w.det_bank.findData("bottom"))
        assert w.shown_efficiency.bank == "bottom"
        # save: all banks in one file; the reduction takes the map of its bank from it
        path = str(tmp_path / "eff.h5")
        monkeypatch.setattr(QtWidgets.QFileDialog, "getSaveFileName", staticmethod(lambda *a, **k: (path, "")))
        w.on_save_efficiency()
        assert set(Efficiency.load_banks(path)) == {"", "left", "bottom"}
        t.efficiency = path
        for b in ("", "left", "bottom"):
            e = w.efficiency_for(t, b)
            assert e.bank == b and np.allclose(e.value.value, maps[b].value.value)
        # a file with the map of one bank is refused for another bank
        one = str(tmp_path / "left.h5")
        maps["left"].save(one)
        assert load_efficiency(one, "left").shape == (136, 16)
        with pytest.raises(ValueError, match="left detector"):
            load_efficiency(one, "bottom")
        # load: all banks back into the detector panel
        monkeypatch.setattr(QtWidgets.QFileDialog, "getOpenFileName", staticmethod(lambda *a, **k: (path, "")))
        w.shown_efficiencies = {}
        w.on_load_efficiency()
        assert set(w.shown_efficiencies) == {"", "left", "bottom"} and w.shown_efficiency.bank == "bottom"
        # archive: one entry per bank, compared only with maps of the same bank
        arch = str(tmp_path / "archive.h5")
        for e in w.shown_efficiencies.values():
            el.archive_efficiency(e, arch)
        rows = el.list_maps(arch)
        assert sorted(r["bank"] for r in rows) == ["", "bottom", "left"]
        assert all(r["instrument"] == "SANS-LLB" for r in rows)
        hist = {r["entry"]: r["reference"] for r in el.history(arch)}
        assert all(k == v for k, v in hist.items())
    finally:
        w.close()
