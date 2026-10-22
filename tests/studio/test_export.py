"""Export of a fit result: CSV files, Excel workbook, one-page summary picture (PDF / PNG / SVG)."""

import csv
import io
import sys

import numpy as np
import pytest

from pysasfit.studio import export
from pysasfit.studio.fitting import FitProblem

try:
    from .support import Q, installed, make_data, make_model, problem, two_dataset_problem
except ImportError:
    from support import Q, installed, make_data, make_model, problem, two_dataset_problem

needs_xlsx = pytest.mark.skipif(not installed("openpyxl"), reason="openpyxl is not installed")
needs_mpl = pytest.mark.skipif(not installed("matplotlib"), reason="matplotlib is not installed")
needs_pil = pytest.mark.skipif(not installed("PIL"), reason="Pillow is not installed")
PNG, PDF = b"\x89PNG", b"%PDF"


@pytest.fixture(scope="module")
def fitted():
    p = problem()
    return p, p.fit("scipy")


def tiny_png():
    from PIL import Image

    buf = io.BytesIO()
    Image.new("RGB", (40, 30), "white").save(buf, format="PNG")
    return buf.getvalue()


def read_csv(path, delimiter=","):
    with open(path, newline="", encoding="utf-8") as fh:
        return list(csv.reader(fh, delimiter=delimiter))


# ------------------------------------------------------------------ rows
def test_package_checks_are_booleans():
    assert export.have_xlsx() == installed("openpyxl")
    assert export.have_matplotlib() == installed("matplotlib")


def test_parameter_rows(fitted):
    p, r = fitted
    rows = export.parameter_rows(r)
    assert [row[0] for row in rows] == ["ff.A", "ff.k", "background"]
    for row, par in zip(rows, r.parameters):
        assert row[1] == pytest.approx(par.value)
        assert row[2] == pytest.approx(r.errors[par.label]) and row[2] > 0
        assert row[5] == "" and len(row) == len(export.PARAMETER_HEADER)
    assert rows[0][3] == 0 and rows[0][4] == 20  # bounds of ff.A


def test_parameter_rows_for_a_fixed_parameter_and_infinite_bounds():
    p = problem(upper=False)
    p.parameter("ff.k").fixed = True
    r = p.fit("scipy")
    rows = {row[0]: row for row in export.parameter_rows(r)}
    assert rows["ff.k"][5] == "yes" and rows["ff.k"][2] is None  # fixed: no error
    assert rows["ff.A"][4] is None  # upper = inf: empty cell, not "inf"


def test_parameter_rows_flag_parameters_at_a_bound():
    p = problem()
    p.parameter("background").lower = 0.5  # the data want 0.3: the fit sits on the bound
    p.parameter("background").upper = 2
    r = p.fit("scipy")
    rows = {row[0]: row for row in export.parameter_rows(r)}
    assert rows["background"][6] == "yes" and rows["ff.A"][6] == ""


def test_parameter_rows_with_asymmetric_intervals(fitted):
    p, r = fitted
    r2 = type(r)(**{**vars(r), "asymmetric": {"ff.A": (-0.1, 0.2)}})
    rows = {row[0]: row for row in export.parameter_rows(r2)}
    assert rows["ff.A"][7] == -0.1 and rows["ff.A"][8] == 0.2 and rows["ff.k"][7] is None


def test_curve_rows(fitted):
    p, r = fitted
    rows = export.curve_rows(r, p, 0)
    d = p.datasets[0]
    assert len(rows) == Q.size and len(rows[0]) == len(export.CURVE_HEADER)
    q, I, s, fit, res = (np.array([row[i] for row in rows], float) for i in range(5))
    assert np.allclose(q, d.q) and np.allclose(I, d.I) and np.allclose(s, d.sigma)
    assert np.allclose(res, (fit - d.I) / d.sigma)


def test_curve_rows_use_empty_cells_for_non_finite_values(fitted):
    p, r = fitted
    d = p.datasets[0]
    old = d.sigma.copy()
    d.sigma[3] = 0.0  # residual = inf
    try:
        rows = export.curve_rows(r, p, 0)
        assert rows[3][4] is None
    finally:
        d.sigma[:] = old


# ------------------------------------------------------------------ CSV
def test_csv_export_writes_all_files(fitted, tmp_path):
    p, r = fitted
    files = export.save_csv(tmp_path / "fit.csv", r, p, png=b"png-bytes", delimiter=";")
    names = sorted(f.replace("\\", "/").split("fit_")[-1] for f in files)
    assert names == ["correlation.csv", "covariance.csv", "curve.csv", "parameters.csv", "plot.png"]
    assert (tmp_path / "fit_plot.png").read_bytes() == b"png-bytes"


def test_csv_correlation_is_symmetric_with_unit_diagonal(fitted, tmp_path):
    p, r = fitted
    export.save_csv(tmp_path / "fit.csv", r, p, png=False)
    rows = read_csv(tmp_path / "fit_correlation.csv")
    assert rows[0] == ["", "ff.A", "ff.k", "background"] and len(rows) == 4
    m = np.array([[float(v) for v in row[1:]] for row in rows[1:]])
    assert np.allclose(np.diag(m), 1.0) and np.allclose(m, m.T) and np.all(np.abs(m) <= 1 + 1e-9)
    assert [row[0] for row in rows[1:]] == ["ff.A", "ff.k", "background"]


def test_csv_covariance_matches_the_summary(fitted, tmp_path):
    p, r = fitted
    export.save_csv(tmp_path / "fit.csv", r, p, png=False)
    rows = read_csv(tmp_path / "fit_covariance.csv")
    m = np.array([[float(v) for v in row[1:]] for row in rows[1:]])
    assert np.allclose(m, r.covariance, rtol=1e-9)


def test_csv_default_delimiter_is_a_comma_and_semicolon_is_possible(fitted, tmp_path):
    p, r = fitted
    export.save_csv(tmp_path / "a.csv", r, p, png=False)
    export.save_csv(tmp_path / "b.csv", r, p, png=False, delimiter=";")
    assert read_csv(tmp_path / "a_parameters.csv")[0][:3] == ["parameter", "value", "error"]
    assert (tmp_path / "b_parameters.csv").read_text(encoding="utf-8").splitlines()[0].startswith("parameter;value;error")


def test_csv_curve_has_a_row_per_point_with_the_data_set_name(fitted, tmp_path):
    p, r = fitted
    export.save_csv(tmp_path / "fit.csv", r, p, png=False)
    rows = read_csv(tmp_path / "fit_curve.csv")
    assert rows[0][:3] == ["data set", "q", "I data"] and len(rows) == Q.size + 1
    assert {row[0] for row in rows[1:]} == {p.datasets[0].name}


def test_csv_of_two_data_sets_stacks_the_curves(tmp_path):
    p = two_dataset_problem()
    r = p.fit("scipy")
    export.save_csv(tmp_path / "two.csv", r, p, png=False)
    rows = read_csv(tmp_path / "two_curve.csv")
    assert len(rows) == 2 * Q.size + 1
    assert [row[0] for row in rows[1 : Q.size + 1]] == ["a"] * Q.size and rows[-1][0] == "b"


def test_csv_without_a_covariance_has_no_matrix_files(tmp_path):
    p = problem()
    r = p.fit("scipy")
    r.covariance = None
    files = export.save_csv(tmp_path / "x.csv", r, p, png=False)
    names = {f.replace("\\", "/").rsplit("_", 1)[-1] for f in files}
    assert names == {"parameters.csv", "curve.csv"}


def test_csv_png_false_writes_no_picture_and_bytes_are_written_as_given(fitted, tmp_path):
    p, r = fitted
    files = export.save_csv(tmp_path / "x.csv", r, p, png=False)
    assert not any(f.endswith("_plot.png") for f in files)
    files = export.save_csv(tmp_path / "y.csv", r, p, png=b"abc")
    assert (tmp_path / "y_plot.png").read_bytes() == b"abc" and files[-1].endswith("_plot.png")


@needs_mpl
def test_csv_makes_the_summary_picture_by_default(fitted, tmp_path):
    p, r = fitted
    export.save_csv(tmp_path / "z.csv", r, p)
    assert (tmp_path / "z_plot.png").read_bytes()[:4] == PNG


def test_csv_name_without_suffix_and_with_dots(fitted, tmp_path):
    p, r = fitted
    export.save_csv(tmp_path / "my.sample.v2", r, p, png=False)
    assert (tmp_path / "my.sample_parameters.csv").exists() or (tmp_path / "my.sample.v2_parameters.csv").exists()


def test_csv_handles_unicode_paths(fitted, tmp_path):
    p, r = fitted
    d = tmp_path / "Größe_µm"
    d.mkdir()
    export.save_csv(d / "fit.csv", r, p, png=False)
    assert (d / "fit_parameters.csv").exists()


# ------------------------------------------------------------------ Excel
@needs_xlsx
def test_xlsx_has_the_sheets_in_order(fitted, tmp_path):
    from openpyxl import load_workbook

    p, r = fitted
    wb = load_workbook(export.save_xlsx(tmp_path / "fit.xlsx", r, p, png=False))
    assert wb.sheetnames == ["Summary", "Parameters", "Correlation", "Covariance", "Curve", "Report"]


@needs_xlsx
def test_xlsx_summary_sheet_content(fitted, tmp_path):
    from openpyxl import load_workbook

    p, r = fitted
    ws = load_workbook(export.save_xlsx(tmp_path / "fit.xlsx", r, p, png=False))["Summary"]
    cells = {ws.cell(i, 1).value: ws.cell(i, 2).value for i in range(1, ws.max_row + 1)}
    assert cells["optimiser"] == "scipy" and cells["converged"] == "yes"
    assert cells["chi2"] == pytest.approx(r.chi2) and cells["chi2 / dof"] == pytest.approx(r.chi2_red)
    assert cells["degrees of freedom"] == r.dof and cells["evaluations"] == r.nfev
    assert any(str(k).startswith("data set") for k in cells)


@needs_xlsx
def test_xlsx_summary_lists_constraints(tmp_path):
    from openpyxl import load_workbook

    p = problem()
    p.add_constraint("ff_k < 5")
    r = p.fit("scipy")
    ws = load_workbook(export.save_xlsx(tmp_path / "c.xlsx", r, p, png=False))["Summary"]
    keys = [str(ws.cell(i, 1).value) for i in range(1, ws.max_row + 1)]
    assert any(k.startswith("constraint ff_k < 5") for k in keys)


@needs_xlsx
def test_xlsx_parameters_sheet(fitted, tmp_path):
    from openpyxl import load_workbook

    p, r = fitted
    ws = load_workbook(export.save_xlsx(tmp_path / "fit.xlsx", r, p, png=False))["Parameters"]
    assert [c.value for c in ws[1]] == export.PARAMETER_HEADER
    for i, par in enumerate(r.parameters, start=2):
        assert ws.cell(i, 1).value == par.label
        assert ws.cell(i, 2).value == pytest.approx(par.value)
        assert ws.cell(i, 3).value == pytest.approx(r.errors[par.label])


@needs_xlsx
def test_xlsx_correlation_and_covariance_sheets(fitted, tmp_path):
    from openpyxl import load_workbook

    p, r = fitted
    wb = load_workbook(export.save_xlsx(tmp_path / "fit.xlsx", r, p, png=False))
    cor, cov = wb["Correlation"], wb["Covariance"]
    n = len(r.free_labels)
    m = np.array([[cor.cell(i + 2, j + 2).value for j in range(n)] for i in range(n)], float)
    assert np.allclose(np.diag(m), 1.0) and np.allclose(m, m.T)
    c = np.array([[cov.cell(i + 2, j + 2).value for j in range(n)] for i in range(n)], float)
    assert np.allclose(c, r.covariance)
    assert [cor.cell(1, j + 2).value for j in range(n)] == r.free_labels
    assert cor.conditional_formatting  # the colour scale from -1 to 1


@needs_xlsx
def test_xlsx_curve_sheet_and_chart(fitted, tmp_path):
    from openpyxl import load_workbook

    p, r = fitted
    ws = load_workbook(export.save_xlsx(tmp_path / "fit.xlsx", r, p, png=False))["Curve"]
    assert [c.value for c in ws[1]] == export.CURVE_HEADER and ws.max_row == Q.size + 1
    assert len(ws._charts) == 1


@needs_xlsx
def test_xlsx_report_sheet_holds_the_result_text(fitted, tmp_path):
    from openpyxl import load_workbook

    p, r = fitted
    ws = load_workbook(export.save_xlsx(tmp_path / "fit.xlsx", r, p, png=False))["Report"]
    text = "\n".join(str(c.value) for c in ws["A"] if c.value is not None)
    assert "correlation matrix" in text
    keep = lambda t: [ln for ln in t.splitlines() if ln.strip()]  # noqa: E731 -- empty lines are empty cells
    assert keep(text) == keep(r.text())


@needs_xlsx
@needs_pil
def test_xlsx_with_a_given_picture_has_a_plot_sheet(fitted, tmp_path):
    from openpyxl import load_workbook

    p, r = fitted
    wb = load_workbook(export.save_xlsx(tmp_path / "fit.xlsx", r, p, png=tiny_png()))
    assert "Plot" in wb.sheetnames and len(wb["Plot"]._images) == 1


@needs_xlsx
@needs_pil
@needs_mpl
def test_xlsx_makes_the_picture_by_default(fitted, tmp_path):
    from openpyxl import load_workbook

    p, r = fitted
    wb = load_workbook(export.save_xlsx(tmp_path / "fit.xlsx", r, p))
    assert len(wb["Plot"]._images) == 1


@needs_xlsx
def test_xlsx_png_false_leaves_the_plot_sheet_out(fitted, tmp_path):
    from openpyxl import load_workbook

    p, r = fitted
    assert "Plot" not in load_workbook(export.save_xlsx(tmp_path / "fit.xlsx", r, p, png=False)).sheetnames


@needs_xlsx
def test_xlsx_of_two_data_sets_has_a_curve_sheet_each(tmp_path):
    from openpyxl import load_workbook

    p = two_dataset_problem()
    r = p.fit("scipy")
    wb = load_workbook(export.save_xlsx(tmp_path / "two.xlsx", r, p, png=False))
    assert "Curve a" in wb.sheetnames and "Curve b" in wb.sheetnames and "Curve" not in wb.sheetnames
    assert wb["Curve a"].max_row == Q.size + 1


@needs_xlsx
def test_xlsx_sheet_names_are_valid_and_unique(tmp_path):
    from openpyxl import load_workbook

    model = make_model()
    p = FitProblem()
    p.add_dataset(make_data(5, 0.8, 0.3, seed=1), model, name="sample/1: [a]*?")
    p.add_dataset(make_data(2, 0.8, 0.1, seed=2), model, name="sample/2: [b]*?")
    p.add_dataset(make_data(3, 0.8, 0.2, seed=3), model, name="x" * 60)
    p.add_common("0.ff.k", lower=0)
    p.add_local("0.ff.A", lower=0)
    r = p.fit("scipy")
    wb = load_workbook(export.save_xlsx(tmp_path / "n.xlsx", r, p, png=False))
    names = wb.sheetnames
    assert len(set(names)) == len(names)
    assert all(len(n) <= 31 and not any(ch in n for ch in "[]:*?/\\") for n in names)
    assert sum(n.startswith("Curve") for n in names) == 3


@needs_xlsx
def test_xlsx_without_covariance_has_no_matrix_sheets(tmp_path):
    from openpyxl import load_workbook

    p = problem()
    r = p.fit("scipy")
    r.covariance = None
    wb = load_workbook(export.save_xlsx(tmp_path / "x.xlsx", r, p, png=False))
    assert "Correlation" not in wb.sheetnames and "Covariance" not in wb.sheetnames
    assert "Parameters" in wb.sheetnames


@needs_xlsx
def test_xlsx_nan_and_inf_become_empty_cells(tmp_path):
    from openpyxl import load_workbook

    p = problem(upper=False)
    r = p.fit("scipy")
    ws = load_workbook(export.save_xlsx(tmp_path / "x.xlsx", r, p, png=False))["Parameters"]
    assert ws.cell(2, 5).value is None  # upper bound inf


@needs_xlsx
def test_xlsx_without_openpyxl_gives_a_helpful_error(fitted, tmp_path, monkeypatch):
    p, r = fitted
    for name in [m for m in sys.modules if m == "openpyxl" or m.startswith("openpyxl.")]:
        monkeypatch.delitem(sys.modules, name)
    monkeypatch.setitem(sys.modules, "openpyxl", None)
    with pytest.raises(ImportError, match="openpyxl"):
        export.save_xlsx(tmp_path / "x.xlsx", r, p, png=False)


@needs_xlsx
def test_xlsx_can_be_overwritten(fitted, tmp_path):
    p, r = fitted
    export.save_xlsx(tmp_path / "a.xlsx", r, p, png=False)
    export.save_xlsx(tmp_path / "a.xlsx", r, p, png=False)
    assert (tmp_path / "a.xlsx").stat().st_size > 0


# ------------------------------------------------------------------ the one-page picture
@needs_mpl
@pytest.mark.parametrize("fmt, magic", [("png", PNG), ("pdf", PDF), ("svg", b"<?xml")])
def test_figure_bytes_formats(fitted, fmt, magic):
    p, r = fitted
    assert export.figure_bytes(r, p, fmt, dpi=60)[:5].startswith(magic)


@needs_mpl
def test_save_figure_follows_the_suffix(fitted, tmp_path):
    p, r = fitted
    for suffix, magic in ((".pdf", PDF), (".png", PNG), (".PNG", PNG), (".svg", b"<?xml")):
        path = export.save_figure(tmp_path / f"s{suffix}", r, p, dpi=60)
        assert open(path, "rb").read(5).startswith(magic)
    path = export.save_figure(tmp_path / "noext", r, p, dpi=60)
    assert open(path, "rb").read(4) == PNG  # no suffix: png


@needs_mpl
def test_save_figure_refuses_unknown_suffixes(fitted, tmp_path):
    p, r = fitted
    with pytest.raises(ValueError, match="unknown picture format"):
        export.save_figure(tmp_path / "s.xyz", r, p)


@needs_mpl
def test_figure_dpi_changes_the_size(fitted):
    p, r = fitted
    assert len(export.figure_bytes(r, p, "png", dpi=40)) < len(export.figure_bytes(r, p, "png", dpi=150))


@needs_mpl
def test_figure_layout_axes_and_scales(fitted):
    p, r = fitted
    fig = export.summary_figure(r, p)
    ax_main = fig.axes[0]
    assert ax_main.get_xscale() == "log" and ax_main.get_yscale() == "log"
    assert len(fig.axes) == 5  # data, residuals, correlation, parameter table, notes
    assert "chi" in fig._suptitle.get_text()


@needs_mpl
def test_figure_title_can_be_set(fitted):
    p, r = fitted
    assert export.summary_figure(r, p, title="My sample").\
        _suptitle.get_text() == "My sample"


@needs_mpl
def test_figure_uses_a_linear_axis_for_non_positive_data():
    q = np.linspace(0.1, 3, 40)
    q_, I, s = make_data(q=q)
    I = I - 4.0  # negative values: a log axis would drop them
    p = FitProblem()
    p.add_dataset((q_, I, s), make_model(bg=0.0))
    p.add_parameter("0.ff.A", lower=0, upper=20)
    p.add_parameter("background", lower=-10, upper=10)
    r = p.fit("scipy")
    assert export.summary_figure(r, p).axes[0].get_yscale() == "linear"


@needs_mpl
def test_figure_with_two_data_sets(tmp_path):
    p = two_dataset_problem()
    r = p.fit("scipy")
    fig = export.summary_figure(r, p)
    labels = [t.get_text() for t in fig.axes[0].get_legend().get_texts()]
    assert "a" in labels and "b" in labels and "a fit" in labels and "b fit" in labels
    assert export.save_figure(tmp_path / "two.pdf", r, p, dpi=60)


@needs_mpl
def test_figure_without_covariance_still_renders(tmp_path):
    p = problem()
    r = p.fit("scipy")
    r.covariance = None
    assert export.figure_bytes(r, p, "png", dpi=50)[:4] == PNG


@needs_mpl
def test_figure_warns_about_strongly_correlated_parameters(fitted):
    p, r = fitted
    r2 = type(r)(**{**vars(r)})
    n = len(r2.free_labels)
    cov = np.eye(n)
    cov[0, 1] = cov[1, 0] = 0.99
    r2.covariance = cov
    fig = export.summary_figure(r2, p)
    texts = " ".join(t.get_text() for ax in fig.axes for t in ax.texts)
    assert "strongly correlated" in texts


@needs_mpl
def test_figure_for_many_parameters_uses_numbers_for_the_matrix_ticks(fitted):
    p, r = fitted
    r2 = type(r)(**{**vars(r)})
    n = 14
    r2.free_labels = [f"p{i}" for i in range(n)]
    r2.covariance = np.eye(n)
    fig = export.summary_figure(r2, p)
    ticks = [t.get_text() for t in fig.axes[2].get_xticklabels()]
    assert ticks[:3] == ["1", "2", "3"]
    assert not fig.axes[2].texts  # no numbers inside the cells above 10 parameters


@needs_mpl
def test_figure_notes_mention_non_convergence_and_violated_constraints(fitted):
    p, r = fitted
    r2 = type(r)(**{**vars(r)})
    r2.success, r2.message = False, "max evaluations reached"
    r2.constraints = [("ff_k < 0.1", 0.3, False)]
    texts = " ".join(t.get_text() for ax in export.summary_figure(r2, p).axes for t in ax.texts)
    assert "NOT converged" in texts and "VIOLATED" in texts


@needs_mpl
def test_figure_marks_fixed_parameters_and_bounds(tmp_path):
    p = problem()
    p.parameter("ff.k").fixed = True
    r = p.fit("scipy")
    texts = " ".join(t.get_text() for ax in export.summary_figure(r, p).axes for t in ax.texts)
    assert "(fixed)" in texts


def test_at_bounds_detects_a_parameter_that_stopped_just_inside_its_bound():
    p = problem()
    p.parameter("background").lower = 0.5
    r = p.fit("scipy")
    assert "background" in r.at_bounds and "ff.A" not in r.at_bounds
    p = problem()
    r = p.fit("scipy")
    assert r.at_bounds == []
