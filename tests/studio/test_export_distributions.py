"""Size distributions N(x) x^3 in the export: xlsx sheet, csv file, summary picture."""

import csv
import io

import numpy as np
import pytest

from pysasfit.studio import export

try:
    from .support import problem
except ImportError:
    from support import problem


class _Dist:
    name = "sd_Fake"
    parameter_names = ["N", "mu", "s"]

    def evaluate_p(self, x, p):
        N, mu, s = p
        x = np.asarray(x, float)
        return N * np.exp(-0.5 * (np.log(x / mu) / s) ** 2) / (x * s * np.sqrt(2 * np.pi))


def with_distribution(p, support=(20.0, 120.0)):
    """Give the first contribution of the problem's model a size distribution (like a real SASfit contribution)."""
    c = p.datasets[0].model.contributions[0]
    c.sd, c.sd_p = _Dist(), np.array([2.0, 50.0, 0.3])
    c.support = lambda: support
    return p


@pytest.fixture(scope="module")
def fitted():
    p = with_distribution(problem())
    return p, p.fit("scipy")


def test_no_distribution_no_curves():
    assert export.distribution_curves(problem()) == []
    assert export.distribution_rows(problem()) == []


def test_curves_have_the_weighted_values_and_range(fitted):
    p, _ = fitted
    (c,) = export.distribution_curves(p)
    assert c["p"] == 3.0 and c["name"] == "sd_Fake"
    assert c["x"][0] == pytest.approx(max(10.0, 120 * 1e-4)) and c["x"][-1] == pytest.approx(180.0)
    assert np.allclose(c["Nxp"], c["N"] * c["x"] ** 3)
    assert np.argmax(c["Nxp"]) > np.argmax(c["N"])  # the weight moves the maximum to larger x
    assert np.all(np.diff(c["x"]) > 0)


def test_other_exponent_and_support_with_argument(fitted):
    p, _ = fitted
    p.datasets[0].model.contributions[0].support = lambda *a: (_ for _ in ()).throw(TypeError()) if not a else (20.0, 120.0)
    (c,) = export.distribution_curves(p, p=6)
    assert c["p"] == 6 and np.allclose(c["Nxp"], c["N"] * c["x"] ** 6)
    p.datasets[0].model.contributions[0].support = lambda: (20.0, 120.0)


def test_shared_model_is_listed_once():
    p = with_distribution(problem())
    d = p.datasets[0]
    p.add_dataset((d.q, d.I, d.sigma), d.model, name="second")  # same model object: still one distribution
    assert len(export.distribution_curves(p)) == 1


def test_rows(fitted):
    p, _ = fitted
    rows = export.distribution_rows(p)
    assert len(rows) == 400 and len(rows[0]) == 4 and "sd_Fake" in rows[0][0]


def test_csv_has_a_distribution_file(fitted, tmp_path):
    p, r = fitted
    files = export.save_csv(tmp_path / "fit.csv", r, p, png=False)
    (name,) = [f for f in files if f.endswith("_distributions.csv")]
    rows = list(csv.reader(open(name, encoding="utf-8")))
    assert rows[0] == ["distribution", "x", "N(x)", "N(x) x^3"] and len(rows) == 401
    x, n, nxp = (float(v) for v in rows[1][1:])
    assert nxp == pytest.approx(n * x**3)


def test_csv_without_distribution_has_no_such_file(tmp_path):
    p = problem()
    files = export.save_csv(tmp_path / "fit.csv", p.fit("scipy"), p, png=False)
    assert not any("distributions" in f for f in files)


def test_xlsx_sheet_and_chart(fitted, tmp_path):
    openpyxl = pytest.importorskip("openpyxl")
    p, r = fitted
    path = export.save_xlsx(tmp_path / "fit.xlsx", r, p, png=False)
    wb = openpyxl.load_workbook(path)
    assert "Distributions" in wb.sheetnames
    ws = wb["Distributions"]
    assert ws.max_row == 401 and "x^3" in ws.cell(1, 3).value
    assert ws.cell(2, 3).value == pytest.approx(ws.cell(2, 2).value * ws.cell(2, 1).value ** 3)
    assert len(ws._charts) == 1


def test_xlsx_without_distribution_has_no_sheet(tmp_path):
    openpyxl = pytest.importorskip("openpyxl")
    p = problem()
    path = export.save_xlsx(tmp_path / "f.xlsx", p.fit("scipy"), p, png=False)
    assert "Distributions" not in openpyxl.load_workbook(path).sheetnames


def test_summary_figure_has_a_distribution_panel(fitted):
    pytest.importorskip("matplotlib")
    p, r = fitted
    fig = export.summary_figure(r, p)
    labels = [a.get_ylabel() for a in fig.axes]
    assert "N(x) x^3" in labels
    ax = fig.axes[labels.index("N(x) x^3")]
    assert ax.get_xscale() == "log" and len(ax.get_lines()) == 1


def test_summary_figure_shows_N_and_the_weighted_N(fitted):
    pytest.importorskip("matplotlib")
    p, r = fitted
    fig = export.summary_figure(r, p, dist_power=6)
    labels = [a.get_ylabel() for a in fig.axes]
    assert "N(x)" in labels and "N(x) x^6" in labels
    plain, weighted = fig.axes[labels.index("N(x)")], fig.axes[labels.index("N(x) x^6")]
    (lp,), (lw,) = plain.get_lines(), weighted.get_lines()
    assert np.allclose(lw.get_ydata(), lp.get_ydata() * lp.get_xdata() ** 6)


def test_the_weight_is_documented_in_the_summary_sheet(fitted, tmp_path):
    openpyxl = pytest.importorskip("openpyxl")
    p, r = fitted
    ws = openpyxl.load_workbook(export.save_xlsx(tmp_path / "f.xlsx", r, p, png=False, dist_power=4))["Summary"]
    texts = [str(c.value) for row in ws.iter_rows() for c in row if c.value]
    assert any("size distribution" in t for t in texts) and any("p = 4" in t for t in texts)


def test_xlsx_and_csv_have_N_and_the_weighted_N_for_any_p(fitted, tmp_path):
    openpyxl = pytest.importorskip("openpyxl")
    p, r = fitted
    ws = openpyxl.load_workbook(export.save_xlsx(tmp_path / "f.xlsx", r, p, png=False, dist_power=6))["Distributions"]
    assert [ws.cell(1, k).value for k in (2, 3)] == ["N(x)", "N(x) x^6"]
    x, n, nxp = (ws.cell(5, k).value for k in (1, 2, 3))
    assert nxp == pytest.approx(n * x**6)
    files = export.save_csv(tmp_path / "f.csv", r, p, png=False, dist_power=6)
    assert open([f for f in files if "distributions" in f][0], encoding="utf-8").readline().strip().endswith("N(x) x^6")


def test_summary_figure_without_distribution_is_unchanged():
    pytest.importorskip("matplotlib")
    p = problem()
    fig = export.summary_figure(p.fit("scipy"), p)
    assert not any("N(x)" in a.get_ylabel() for a in fig.axes)


@pytest.mark.parametrize("fmt", ["png", "pdf", "svg"])
def test_picture_bytes_with_distribution(fitted, fmt):
    pytest.importorskip("matplotlib")
    p, r = fitted
    data = export.figure_bytes(r, p, fmt)
    assert len(data) > 2000
    assert data[:4] == {"png": b"\x89PNG", "pdf": b"%PDF", "svg": b"<?xm"}[fmt]
