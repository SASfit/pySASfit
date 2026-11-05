"""Time series: the runs of a sample row reduced one by one (or binned), each with its time."""
import datetime as dt
import os
from types import SimpleNamespace as NS

import h5py
import numpy as np
import pytest

from pysasfit.sasred.reduction_table import RowSet, RunChoice
from pysasfit.sasred.series import parse_time, run_times, series_points

RAW = os.environ.get("PYSASRED_TEST_RAW", "/mnt/user-data/uploads/Sept2026/raw")


def _raw(n, start, dur=60.0, mon=1000.0):
    return NS(run=n, start_time=start.isoformat(sep=" "), end_time=None, counting_time_s=dur,
              monitors={"monitor1": mon}, empty_reason="")


def test_points_times_and_bins():
    t = dt.datetime(2026, 11, 5, 10, 0, 0)
    raws = {100 + i: _raw(100 + i, t + dt.timedelta(seconds=70 * i), mon=1000.0 * (1 + (i == 1)))
            for i in range(6)}
    row = RowSet("sample", "No. 1", "gel", [RunChoice(n) for n in raws], series=True)
    p = series_points(row, raws)
    assert [x.runs for x in p] == [[n] for n in sorted(raws)]
    assert p[0].time_s == pytest.approx(30) and p[3].time_s == pytest.approx(210 + 30)
    row.series_bin = 2                                     # two runs per point, monitor-weighted time
    p = series_points(row, raws)
    assert [x.runs for x in p] == [[100, 101], [102, 103], [104, 105]]
    assert p[0].time_s == pytest.approx((1000 * 30 + 2000 * 100) / 3000)
    row.series_bin, row.series_dt_s = 1, 150.0             # windows of 150 s
    assert [x.runs for x in series_points(row, raws)] == [[100, 101], [102, 103], [104, 105]]
    row.series_dt_s, row.series_t0 = None, "2026-11-05 09:59:00"     # own time zero
    assert series_points(row, raws)[0].time_s == pytest.approx(90)
    row.runs[2].use = False                                # not ticked: not in the series
    assert 102 not in [n for x in series_points(row, raws) for n in x.runs]
    assert "series: 5 of 6" in row.number
    assert parse_time("05.11.2026 10:00") == dt.datetime(2026, 11, 5, 10, 0)
    with pytest.raises(ValueError):
        parse_time("tomorrow")
    a, m, b = run_times(_raw(1, t, 100.0))
    assert (m - a).total_seconds() == pytest.approx(50)


def test_series_row_reduced_point_by_point(tmp_path, monkeypatch):
    if not os.path.isdir(RAW):
        pytest.skip("Vifor raw data not available")
    pytest.importorskip("PySide6")
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    monkeypatch.setenv("PYSASFIT_SETTINGS", str(tmp_path / "settings.json"))
    from PySide6 import QtWidgets
    QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    from pysasfit.sasred import gui as g
    monkeypatch.setattr(g, "_session_path", lambda: str(tmp_path / "s.json"))
    from pysasfit.sasred.io.nxcansas import write_nxcansas_1d
    w = g.MainWindow()
    w.set_folder(RAW)
    page = next(p for p in w.table_panel.pages() if p.table.config.startswith("SD 8.0"))
    t = page.table
    row = next(r for r in t.samples if len(r.runs) >= 2)
    runs = sorted(c.run for c in row.runs)
    page.set_series(row, True, runs[0], runs[-1])
    assert row.series and all(c.use for c in row.runs)
    tag = w.reduce_table_rows(t, [row])
    keys = [k for k in w._last_keys]
    assert len(keys) == len(runs) and all(" t=" in k for k in keys)
    res = [w.results[k] for k in keys]
    ts = [r.parameters["time_s"] for r in res]
    assert ts[0] > 0 and all(b > a for a, b in zip(ts, ts[1:]))
    assert all(r.parameters["series"] == res[0].parameters["series"] for r in res)
    assert any("time series" in n for n in res[0].notes)
    p = tmp_path / "pt.h5"
    write_nxcansas_1d(p, res[1])
    import json
    with h5py.File(p) as f:
        meta = json.loads(f["sasentry01/sasprocess01/metadata"][()])
    assert meta["time_s"] == pytest.approx(ts[1]) and meta["series_index"] == 1
    from pysasfit.toolbox.merge import curve_from_reduced
    assert curve_from_reduced(res[0]).meta["time_s"] == pytest.approx(ts[0])
    page.set_series(row, False)                            # summed again
    tag = w.reduce_table_rows(t, [row])
    assert "time_s" not in w.results[tag].parameters
    w.close()


def test_time_series_button_opens_the_dialog(tmp_path, monkeypatch):
    """The button (clicked passes a 'checked' flag) opens the dialog for the selected row; OK makes it a series."""
    if not os.path.isdir(RAW):
        pytest.skip("Vifor raw data not available")
    pytest.importorskip("PySide6")
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    monkeypatch.setenv("PYSASFIT_SETTINGS", str(tmp_path / "settings.json"))
    from PySide6 import QtWidgets
    QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    from pysasfit.sasred import gui as g
    monkeypatch.setattr(g, "_session_path", lambda: str(tmp_path / "s.json"))
    w = g.MainWindow()
    w.set_folder(RAW)
    page = next(p for p in w.table_panel.pages() if p.table.config.startswith("SD 8.0"))
    i = next(k for k, r in enumerate(page.table.samples) if len(r.runs) >= 2)
    page.samp.setCurrentCell(i, 1)
    shown = {}

    def fake_exec(dlg):
        shown["title"] = dlg.windowTitle()
        dlg.findChild(QtWidgets.QCheckBox).setChecked(True)
        return QtWidgets.QDialog.DialogCode.Accepted
    monkeypatch.setattr(QtWidgets.QDialog, "exec", fake_exec)
    btn = next(b for b in page.findChildren(QtWidgets.QPushButton) if b.text() == "Time series…")
    btn.click()
    assert shown["title"].startswith("Time series: No.")
    assert page.table.samples[i].series and page.table.samples[i].number.startswith("series:")
    w.close()
