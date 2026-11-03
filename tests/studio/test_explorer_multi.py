"""Model explorer: several data sets, contributions per data set, global parameters with links, project file."""
import os
import time

import numpy as np
import pytest

pytest.importorskip("PySide6")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


def _sph(q, p):
    R, eta = p[0], p[1]
    x = q * R
    with np.errstate(all="ignore"):
        f = np.where(x > 1e-6, 3 * (np.sin(x) - x * np.cos(x)) / x**3, 1.0)
    return (eta * 4 / 3 * np.pi * R**3 * f) ** 2


@pytest.fixture
def explorer(tmp_path, monkeypatch):
    from PySide6 import QtWidgets
    from pysasfit.studio import register_python_function, unregister_python_function
    monkeypatch.setenv("PYSASFIT_SETTINGS", str(tmp_path / "settings.json"))
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    register_python_function("ff_pysphere", _sph, parameters=("R", "eta"), defaults=(5.0, 1.0))
    from pysasfit.studio.explorer import ModelExplorer
    e = ModelExplorer(ff="ff_pysphere")
    yield e, app
    e.close()
    unregister_python_function("ff_pysphere")


def _data(R, sc, bg, seed):
    q = np.geomspace(0.05, 1.5, 70)
    rng = np.random.default_rng(seed)
    I = sc * _sph(q, [R, 1.0]) + bg
    s = 0.01 * I
    return q, I + rng.normal(0, s), s


def _wait(fp, app):
    t0 = time.time()
    while fp._thread is not None and time.time() - t0 < 120:
        app.processEvents()
        time.sleep(0.02)


def test_two_datasets_global_radius(explorer, tmp_path):
    from pysasfit.studio import project
    e, app = explorer
    e.add_contribution("ff_pysphere")
    p1, p2 = e.panels()
    p1.set_datasets_text("1")
    p2.set_datasets_text("2")
    e.add_dataset(*_data(6.0, 1e-6, 0.01, 1), name="a.dat")
    e.add_dataset(*_data(12.0, 1e-7, 0.02, 2), name="b.dat")
    fp = e.fit_panel
    assert fp.data_list.count() == 2
    fp.add_global(value=5.5, lower="1", upper="20")
    fp.links_edit.setPlainText("c1.ff.R = P1\nc2.ff.R = 2 * P1")
    for r in (e.bg_row, p1.scale_row, p2.scale_row):
        r.set_fit(True, emit=True)
    p1.scale_row.set_value(1e-6)
    p2.scale_row.set_value(1e-7)
    fp._local.add("background")
    fp.refresh_table()
    e._redraw()
    R2 = next(r for r in p2.param_rows["ff"] if r.name == "R")
    assert R2.value == pytest.approx(11.0) and not R2.edit.isEnabled()
    fp._fit_or_stop()
    _wait(fp, app)
    s = fp._summary
    assert s is not None and s.success
    assert s.values["P1"] == pytest.approx(6.0, rel=0.01)
    assert [d["bg"]["background"] for d in e.datasets] == pytest.approx([0.01, 0.02], rel=0.05)
    assert fp.global_values()["P1"] == pytest.approx(6.0, rel=0.01)
    # project file keeps everything
    e.ds_combo.setCurrentIndex(1)
    path = project.save(e, tmp_path / "multi.sasstudio")
    from pysasfit.studio.explorer import ModelExplorer
    e2 = ModelExplorer(ff="ff_pysphere")
    try:
        project.load(e2, path)
        assert len(e2.datasets) == 2 and e2.current_ds == 1
        assert e2.bg_row.value == pytest.approx(e.datasets[1]["bg"]["background"])
        assert [p.datasets_text() for p in e2.panels()] == ["1", "2"]
        assert "c2.ff.R = 2 * P1" in e2.fit_panel.links_edit.toPlainText()
        assert e2.fit_panel.local_addresses() == {"background"}
    finally:
        e2.close()
    # undo brings back the values before the fit
    fp._undo()
    assert fp.global_values()["P1"] == pytest.approx(5.5)


def test_single_dataset_as_before(explorer):
    e, app = explorer
    e.set_data(*_data(6.0, 1e-6, 0.01, 3), name="one.dat")
    fp = e.fit_panel
    R = next(r for r in e.panels()[0].param_rows["ff"] if r.name == "R")
    R.set_value(5.0)
    e.panels()[0].scale_row.set_value(1e-6)
    for r in (R, e.bg_row, e.panels()[0].scale_row):
        r.set_fit(True, emit=True)
    fp._fit_or_stop()
    _wait(fp, app)
    assert fp._summary.values["ff.R"] == pytest.approx(6.0, rel=0.01)
    assert e.bg_row.value == pytest.approx(0.01, rel=0.05)


def test_link_parsing():
    from pysasfit.studio.multidata import parse_datasets, parse_link
    ln = parse_link("c1.sd.mu[2, 3] = 0.5 * P2", 3)
    assert (ln.target, ln.datasets, ln.factor, ln.name) == ("c1.sd.mu", [1, 2], 0.5, "P2")
    assert parse_link("background = P3 * 2").factor == 2.0
    assert parse_link("ff.R = P1 / 4").factor == 0.25
    assert parse_datasets("2-4") == [1, 2, 3] and parse_datasets("all") is None
    with pytest.raises(ValueError):
        parse_datasets("5", 2)
    with pytest.raises(ValueError):
        parse_link("ff.R == P1")


def test_expression_link_in_the_explorer(explorer):
    e, app = explorer
    e.add_contribution("ff_pysphere")
    fp = e.fit_panel
    fp.add_global(value=3.0)
    fp.links_edit.setPlainText("c2.ff.R = P1 + P1\nc1.ff.R = atan(P1) * 10")
    e._redraw()
    p1, p2 = e.panels()
    R1 = next(r for r in p1.param_rows["ff"] if r.name == "R")
    R2 = next(r for r in p2.param_rows["ff"] if r.name == "R")
    assert R2.value == pytest.approx(6.0) and not R2.edit.isEnabled()
    assert R1.value == pytest.approx(10 * np.arctan(3.0))
    assert p2.contrib.ff_p[0] == pytest.approx(6.0)
