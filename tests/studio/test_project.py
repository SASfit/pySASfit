"""Save and reload of an explorer session (needs PySide6 and the mock SASfit libraries)."""

import json
import os

import numpy as np
import pytest

pytest.importorskip("PySide6")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


@pytest.fixture
def explorer(studio_lib):
    from PySide6 import QtWidgets

    from pysasfit.studio import explorer as E

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    w = E.ModelExplorer("ff_sphere")
    yield w
    w.close()
    w.deleteLater()


def _session(w):
    """A mixture of two contributions with fit marks, bounds, constraints and data."""
    w.add_contribution("ff_sphere", "sd_LogNorm", None)
    p1, p2 = w.panels()
    p1.scale_row.set_value(1.7)
    p1.scale_row.set_fit(True, emit=True)
    rows = {r.name: r for r in p2.param_rows["sd"]}
    for name, value in ("N", 1.0), ("s", 0.2), ("p", 1.0), ("mu", 40.0):  # explicit: defaults differ per installation
        rows[name].set_value(value)
    rows["mu"].lower.setText("20")
    rows["mu"]._on_bounds()
    rows["mu"].set_fit(True, emit=True)
    w.bg_row.set_value(0.01)
    q = np.geomspace(0.02, 0.5, 40)
    sigma = np.full_like(q, 0.1)
    sigma[3] = np.nan  # missing errors must survive the round trip
    w.set_data(q, 1.0 / q**2, sigma, "x.dat")
    w.fit_panel.constraints.setPlainText("c1_scale < 5")
    w.fit_panel.priors.setPlainText("c2_sd_mu 40 5")
    w.logy.setChecked(False)
    return q


def test_round_trip(explorer, tmp_path):
    from pysasfit.studio import explorer as E

    q = _session(explorer)
    before = explorer.fit_selection()
    mu_before = explorer.panels()[1].contrib.vector("sd").copy()
    path = tmp_path / "session"
    assert explorer.save_project(path)
    assert explorer.project_path.endswith(".sasstudio")  # the suffix is added

    other = E.ModelExplorer("ff_sphere")
    try:
        assert other.open_project(explorer.project_path), other.status.text()
        assert len(other.panels()) == 2
        assert [p.names["sd"] for p in other.panels()] == [None, "sd_LogNorm"]
        assert other.fit_selection() == before  # fit ticks and bounds
        assert np.allclose(other.panels()[1].contrib.vector("sd"), mu_before)
        assert other.panels()[0].scale == 1.7
        assert other.bg_row.value == 0.01
        assert other.data["name"] == "x.dat" and other.data["q"].size == 40
        assert np.allclose(other.data["q"], q)
        assert np.isnan(other.data["sigma"][3])
        assert other.fit_panel.constraints.toPlainText() == "c1_scale < 5"
        assert other.fit_panel.priors.toPlainText() == "c2_sd_mu 40 5"
        assert not other.logy.isChecked()
    finally:
        other.close()


def test_loaded_model_gives_the_same_curve(explorer, tmp_path):
    from pysasfit.studio import explorer as E

    _session(explorer)
    q = np.geomspace(0.02, 0.4, 25)
    expected = explorer.build_model()(q)
    explorer.save_project(tmp_path / "p.sasstudio")
    other = E.ModelExplorer("ff_sphere")
    try:
        other.open_project(str(tmp_path / "p.sasstudio"))
        assert np.allclose(other.build_model()(q), expected, rtol=1e-9)
    finally:
        other.close()


def test_the_file_is_plain_json_without_the_installation(explorer, tmp_path):
    _session(explorer)
    explorer.save_project(tmp_path / "p.sasstudio")
    d = json.loads((tmp_path / "p.sasstudio").read_text(encoding="utf-8"))
    assert d["format"] == "pysasfit-studio-project" and d["version"] == 1
    assert len(d["contributions"]) == 2 and d["data"]["name"] == "x.dat"
    assert None in d["data"]["sigma"]  # NaN is stored as null: valid JSON
    assert "installation" not in d  # the SASfit folder belongs to the computer, not to the project


def test_a_wrong_file_is_reported_not_raised(explorer, tmp_path):
    bad = tmp_path / "bad.sasstudio"
    bad.write_text('{"format": "something else"}', encoding="utf-8")
    assert explorer.open_project(str(bad)) is False
    assert "not a pySASfit studio project" in explorer.status.text()
    assert len(explorer.panels()) == 1  # the session is untouched

    bad.write_text("not json", encoding="utf-8")
    assert explorer.open_project(str(bad)) is False


def test_a_newer_format_is_refused(explorer, tmp_path):
    from pysasfit.studio import project

    d = project.to_dict(explorer)
    d["version"] = project.VERSION + 1
    with pytest.raises(ValueError, match="newer"):
        project.load_dict(explorer, d)


def test_unknown_optimiser_keeps_the_default(explorer, tmp_path):
    from pysasfit.studio import project

    d = project.to_dict(explorer)
    d["fit"]["backend"] = "not-installed-here"
    project.load_dict(explorer, d)
    assert explorer.fit_panel.backend.currentData() in explorer.fit_panel.available
