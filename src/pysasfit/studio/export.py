"""Save a fit result: an Excel workbook (``.xlsx``) or CSV files.

The workbook has the sheets

* **Summary**: optimiser, convergence, chi², chi²/dof, degrees of freedom, evaluations, time, data sets;
* **Parameters**: value, error, bounds, fixed, at a bound, MINOS / posterior intervals;
* **Correlation** (colour scale from -1 to 1) and **Covariance** of the free parameters;
* **Curve** (one sheet per data set): q, I(q), sigma, fitted I(q), residual (fit - data) / sigma, with an Excel
  chart (log-log) of data and fit;
* **Distributions** (only if the model has a size distribution): x, N(x) and N(x) x^p of every fitted distribution (p: the weight set in the explorer, default 3), with a chart;
* **Plot**: the one-page summary picture (data, fit, residuals, size distribution N(x) x^3, parameters, correlation matrix; needs matplotlib and
  Pillow);
* **Report**: the text of the Result tab.

CSV files (``<name>_parameters.csv``, ``_correlation.csv``, ``_covariance.csv``, ``_curve.csv``, ``_plot.png``) can be
read by every program. Use ``delimiter=";"`` for Excel with a German or Swiss locale. ``save_figure`` writes the
one-page summary as PDF, PNG or SVG; ``figure_bytes`` gives it as bytes (for the clipboard).

    from pysasfit.studio import export
    result = problem.fit("scipy")
    export.save_xlsx("fit.xlsx", result, problem)      # png=None: the summary picture is made if matplotlib is there
    export.save_csv("fit.csv", result, problem)
    export.save_figure("fit.pdf", result, problem)
"""

from __future__ import annotations

import csv
import datetime
import io
import math
import os
import re
import textwrap
from importlib.util import find_spec

import numpy as np

__all__ = ["have_xlsx", "have_matplotlib", "save_xlsx", "save_csv", "save_figure", "figure_bytes", "summary_figure",
           "parameter_rows", "curve_rows", "distribution_curves", "distribution_rows", "DISTRIBUTION_POWER"]  # fmt: skip

PARAMETER_HEADER = ["parameter", "value", "error", "lower", "upper", "fixed", "at a bound", "interval minus", "interval plus"]
CURVE_HEADER = ["q", "I data", "sigma", "I fit", "(fit - data) / sigma"]


DISTRIBUTION_POWER = 3.0  # the size distributions are shown as N(x) * x**3 (volume weighted)
DISTRIBUTION_HEADER = ["distribution", "x", "N(x)", "N(x) x^p"]


def distribution_curves(problem, p=DISTRIBUTION_POWER, n=400):
    """The fitted size distributions: ``[{"label", "name", "p", "x", "N", "Nxp"}, ...]``, one per contribution with a
    size distribution (a model shared by several data sets is listed once).

    ``N`` is N(x) as used in the fit (amplitude included), ``Nxp`` = N(x) * x**p. The x range is the integration range
    of the distribution (``support()``) widened to [max(lo/2, hi*1e-4), hi*1.5], on ``n`` logarithmic points."""
    out, seen = [], set()
    multi = len(problem.datasets) > 1
    for d in problem.datasets:
        model = d.model
        if id(model) in seen:
            continue
        seen.add(id(model))
        for i, c in enumerate(getattr(model, "contributions", [])):
            sd = getattr(c, "sd", None)
            if sd is None:
                continue
            try:
                lo, hi = c.support()
            except TypeError:
                lo, hi = c.support(0.0)
            lo, hi = max(lo / 2.0, hi * 1e-4), hi * 1.5
            x = np.geomspace(lo, hi, n) if lo > 0 else np.linspace(lo, hi, n)
            N = np.asarray(sd.evaluate_p(x, c.sd_p), float)
            label = f"{sd.name} (contribution {i + 1}{', ' + d.name if multi else ''})"
            out.append({"label": label, "name": sd.name, "p": float(p), "x": x, "N": N, "Nxp": N * x**p})
    return out


def distribution_rows(problem, p=DISTRIBUTION_POWER):
    """Rows ``[distribution, x, N(x), N(x) x^p]`` of all fitted size distributions."""
    rows = []
    for c in distribution_curves(problem, p):
        rows += [[c["label"], _num(x), _num(a), _num(b)] for x, a, b in zip(c["x"], c["N"], c["Nxp"])]
    return rows


def have_xlsx():
    """True if openpyxl (the Excel writer) is installed."""
    return find_spec("openpyxl") is not None


def have_matplotlib():
    """True if matplotlib (the summary picture, PDF and PNG) is installed."""
    return find_spec("matplotlib") is not None


def _num(x):
    """A float for a cell, None (empty cell) for NaN and infinity."""
    if x is None:
        return None
    x = float(x)
    return x if math.isfinite(x) else None


def parameter_rows(summary):
    """``[[parameter, value, error, lower, upper, fixed, at a bound, interval minus, interval plus], ...]``."""
    rows = []
    for p in summary.parameters:
        asym = summary.asymmetric.get(p.label)
        rows.append([
            p.label, _num(p.value), None if p.fixed else _num(summary.errors.get(p.label)), _num(p.lower), _num(p.upper),
            "yes" if p.fixed else "", "yes" if p.label in summary.at_bounds else "",
            _num(asym[0]) if asym else None, _num(asym[1]) if asym else None,
        ])  # fmt: skip
    return rows


def curve_rows(summary, problem, k):
    """Rows of data set ``k``: q, I data, sigma, I fit, residual."""
    d = problem.datasets[k]
    q_fit, I_fit = summary.curves[k]
    I_fit = np.asarray(I_fit, float)
    with np.errstate(invalid="ignore", divide="ignore"):
        res = (I_fit - d.I) / d.sigma
    return [[_num(a), _num(b), _num(c), _num(e), _num(f)] for a, b, c, e, f in zip(d.q, d.I, d.sigma, I_fit, res)]


def _summary_pairs(summary, problem, dist_power=DISTRIBUTION_POWER):
    pairs = [
        ("created", datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")),
        ("optimiser", f"{summary.backend}{'/' + summary.algorithm if summary.algorithm else ''}"),
        ("converged", "yes" if summary.success else "NO"),
        ("message", summary.message),
        ("evaluations", summary.nfev),
        ("seconds", round(summary.seconds, 3)),
        ("chi2", _num(summary.chi2)),
        ("chi2 / dof", _num(summary.chi2_red)),
        ("degrees of freedom", summary.dof),
    ]
    for name, n, c2, c2r in summary.datasets:
        pairs.append((f"data set {name}", f"{n} points, chi2 = {c2:.5g}, chi2/n = {c2r:.4g}"))
    for text, g, ok in summary.constraints:
        pairs.append((f"constraint {text}", f"{'ok' if ok else 'VIOLATED'} (lhs - rhs = {g:.4g})"))
    for c in distribution_curves(problem, dist_power):
        pairs.append((f"size distribution {c['label']}", f"N(x) and N(x) x^p with p = {dist_power:g} (the weight set in the explorer)"))
    return pairs


def _matrix(summary, which):
    m = summary.correlation() if which == "correlation" else summary.covariance
    return None if m is None else np.asarray(m, float)


def _matrix_rows(summary, which):
    m = _matrix(summary, which)
    if m is None:
        return None
    return [[_num(v) for v in row] for row in m]


# --------------------------------------------------------------------------- CSV
def _csv_names(path):
    base, ext = os.path.splitext(str(path))
    return {k: f"{base}_{k}{e}" for k, e in (("parameters", ".csv"), ("correlation", ".csv"), ("covariance", ".csv"),
                                              ("curve", ".csv"), ("distributions", ".csv"), ("plot", ".png"))}  # fmt: skip


def save_csv(path, summary, problem, png=None, delimiter=",", dist_power=DISTRIBUTION_POWER):
    """Write ``<name>_parameters.csv``, ``_correlation.csv``, ``_covariance.csv``, ``_curve.csv``, ``_distributions.csv`` (only if the model has a size distribution; N(x) and N(x) x^``dist_power``) (and ``_plot.png`` if
    ``png`` is given or made) next to ``path``; returns the list of files written.

    ``png``: bytes of the picture, ``None`` (default) to make the summary picture if matplotlib is installed,
    ``False`` to write none."""
    names = _csv_names(path)
    if png is None and have_matplotlib():
        png = figure_bytes(summary, problem, "png", dpi=150, dist_power=dist_power)
    written = []

    def write(key, header, rows):
        with open(names[key], "w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh, delimiter=delimiter)
            w.writerow(header)
            w.writerows([["" if v is None else v for v in row] for row in rows])
        written.append(names[key])

    write("parameters", PARAMETER_HEADER, parameter_rows(summary))
    for which in ("correlation", "covariance"):
        rows = _matrix_rows(summary, which)
        if rows is not None:
            labels = summary.free_labels
            write(which, [""] + labels, [[lab] + row for lab, row in zip(labels, rows)])
    curve = []
    for k, d in enumerate(problem.datasets):
        for row in curve_rows(summary, problem, k):
            curve.append([d.name] + row)
    write("curve", ["data set"] + CURVE_HEADER, curve)
    drows = distribution_rows(problem, dist_power)
    if drows:
        write("distributions", ["distribution", "x", "N(x)", f"N(x) x^{dist_power:g}"], drows)
    if png:
        with open(names["plot"], "wb") as fh:
            fh.write(png)
        written.append(names["plot"])
    return written


# --------------------------------------------------------------------------- Excel
def _sheet_name(text, used):
    name = re.sub(r"[\[\]:*?/\\]", "_", str(text))[:31] or "sheet"
    base, i = name, 1
    while name in used:
        i += 1
        name = f"{base[: 31 - len(str(i)) - 1]}_{i}"
    used.add(name)
    return name


def save_xlsx(path, summary, problem, png=None, dist_power=DISTRIBUTION_POWER):
    """Write the workbook described in the module docstring.

    ``png``: bytes of the picture for the Plot sheet, ``None`` (default) to make the summary picture if matplotlib is
    installed, ``False`` to leave the sheet out."""
    if png is None and have_matplotlib():
        png = figure_bytes(summary, problem, "png", dpi=100, dist_power=dist_power)
    try:
        from openpyxl import Workbook
        from openpyxl.chart import Reference, ScatterChart, Series
        from openpyxl.chart.shapes import GraphicalProperties
        from openpyxl.formatting.rule import ColorScaleRule
        from openpyxl.styles import Alignment, Font
    except ImportError:
        raise ImportError("writing Excel files needs openpyxl (pip install openpyxl); CSV files work without it") from None
    bold = Font(bold=True)
    wb = Workbook()
    used = set()

    ws = wb.active
    ws.title = _sheet_name("Summary", used)
    for r, (key, value) in enumerate(_summary_pairs(summary, problem, dist_power), 1):
        ws.cell(r, 1, key).font = bold
        ws.cell(r, 2, value)
    ws.column_dimensions["A"].width = 28
    ws.column_dimensions["B"].width = 80

    ws = wb.create_sheet(_sheet_name("Parameters", used))
    ws.append(PARAMETER_HEADER)
    for row in parameter_rows(summary):
        ws.append(row)
    for c in ws[1]:
        c.font = bold
    ws.column_dimensions["A"].width = 26
    for col in "BCDEFGHI":
        ws.column_dimensions[col].width = 15

    for which, title in (("correlation", "Correlation"), ("covariance", "Covariance")):
        rows = _matrix_rows(summary, which)
        if rows is None:
            continue
        ws = wb.create_sheet(_sheet_name(title, used))
        labels = summary.free_labels
        ws.append([""] + labels)
        for lab, row in zip(labels, rows):
            ws.append([lab] + row)
        for c in ws[1]:
            c.font = bold
            c.alignment = Alignment(horizontal="center", wrap_text=True)
        for r in range(2, len(labels) + 2):
            ws.cell(r, 1).font = bold
        ws.column_dimensions["A"].width = 26
        last = ws.cell(len(labels) + 1, len(labels) + 1).coordinate
        rng = f"B2:{last}"
        for row in ws[rng]:
            for c in row:
                c.number_format = "0.00" if which == "correlation" else "0.00E+00"
        if which == "correlation":
            ws.conditional_formatting.add(rng, ColorScaleRule(
                start_type="num", start_value=-1, start_color="F8696B", mid_type="num", mid_value=0, mid_color="FFFFFF",
                end_type="num", end_value=1, end_color="5A8AC6"))  # fmt: skip

    for k, d in enumerate(problem.datasets):
        ws = wb.create_sheet(_sheet_name("Curve" if len(problem.datasets) == 1 else f"Curve {d.name}", used))
        ws.append(CURVE_HEADER)
        for c in ws[1]:
            c.font = bold
        rows = curve_rows(summary, problem, k)
        for row in rows:
            ws.append(row)
        n = len(rows)
        if n:
            chart = ScatterChart()
            chart.title = d.name
            chart.scatterStyle = "lineMarker"
            chart.x_axis.title, chart.y_axis.title = "q", "I(q)"
            chart.x_axis.scaling.logBase = 10
            chart.y_axis.scaling.logBase = 10
            chart.x_axis.delete = False
            chart.y_axis.delete = False
            x = Reference(ws, min_col=1, min_row=2, max_row=n + 1)
            data = Series(Reference(ws, min_col=2, min_row=1, max_row=n + 1), x, title_from_data=True)
            data.marker.symbol = "circle"
            data.marker.size = 5
            data.marker.graphicalProperties = GraphicalProperties(solidFill="1F77B4")
            data.marker.graphicalProperties.line.solidFill = "1F77B4"
            data.graphicalProperties = GraphicalProperties()
            data.graphicalProperties.line.noFill = True
            fit = Series(Reference(ws, min_col=4, min_row=1, max_row=n + 1), x, title_from_data=True)
            fit.marker.symbol = "none"
            fit.smooth = False
            fit.graphicalProperties = GraphicalProperties()
            fit.graphicalProperties.line.solidFill = "D62728"
            fit.graphicalProperties.line.width = 22000  # EMU
            chart.series.extend([data, fit])
            chart.height, chart.width = 11, 18
            ws.add_chart(chart, "G2")
        for col in "ABCDE":
            ws.column_dimensions[col].width = 16

    curves = distribution_curves(problem, dist_power)
    if curves:
        ws = wb.create_sheet(_sheet_name("Distributions", used))
        header = []
        for c in curves:
            header += [f"{c['label']}: x", "N(x)", f"N(x) x^{dist_power:g}", ""]
        ws.append(header)
        for c in ws[1]:
            c.font = bold
        nrow = max(len(c["x"]) for c in curves)
        for r in range(nrow):
            row = []
            for c in curves:
                if r < len(c["x"]):
                    row += [_num(c["x"][r]), _num(c["N"][r]), _num(c["Nxp"][r]), None]
                else:
                    row += [None] * 4
            ws.append(row)
        chart = ScatterChart()
        chart.title = f"size distribution, N(x) x^{dist_power:g}"
        chart.scatterStyle = "lineMarker"
        chart.x_axis.title, chart.y_axis.title = "x", f"N(x) x^{dist_power:g}"
        chart.x_axis.delete = False
        chart.y_axis.delete = False
        if all(c["x"][0] > 0 for c in curves):
            chart.x_axis.scaling.logBase = 10
        palette = ["1F77B4", "FF7F0E", "2CA02C", "D62728", "9467BD", "8C564B"]
        for k, c in enumerate(curves):
            n = len(c["x"])
            ser = Series(Reference(ws, min_col=4 * k + 3, min_row=1, max_row=n + 1),
                         Reference(ws, min_col=4 * k + 1, min_row=2, max_row=n + 1), title_from_data=True)  # fmt: skip
            ser.marker.symbol = "none"
            ser.smooth = False
            ser.graphicalProperties = GraphicalProperties()
            ser.graphicalProperties.line.solidFill = palette[k % len(palette)]
            ser.graphicalProperties.line.width = 22000
            chart.series.append(ser)
        chart.height, chart.width = 11, 18
        ws.add_chart(chart, f"{ws.cell(1, 4 * len(curves) + 2).column_letter}2")
        for k in range(len(curves)):
            for j in range(3):
                ws.column_dimensions[ws.cell(1, 4 * k + 1 + j).column_letter].width = 18

    if png:
        try:
            from openpyxl.drawing.image import Image

            ws = wb.create_sheet(_sheet_name("Plot", used))
            ws.add_image(Image(io.BytesIO(png)), "A1")
        except ImportError:  # Pillow missing: the picture is left out, the rest is still written
            pass

    ws = wb.create_sheet(_sheet_name("Report", used))
    for r, line in enumerate(summary.text().splitlines(), 1):
        ws.cell(r, 1, line).font = Font(name="Courier New")
    ws.column_dimensions["A"].width = 110
    wb.save(str(path))
    return str(path)


# --------------------------------------------------------------------------- summary picture
def summary_figure(summary, problem, title=None, dist_power=DISTRIBUTION_POWER):
    """A one-page matplotlib figure (A4 landscape): data and fit with residuals, parameters with errors, the
    correlation matrix, convergence and warnings for strongly correlated parameters."""
    from matplotlib.figure import Figure

    curves = distribution_curves(problem, dist_power)
    fig = Figure(figsize=(11.7, 8.3), layout="constrained")
    if curves:  # a fourth row: the size distributions below the data, the correlation matrix below them
        gs = fig.add_gridspec(4, 2, width_ratios=[1.5, 1.0], height_ratios=[3.0, 1.0, 2.0, 2.2])
        rows_c, rows_n = 3, slice(2, 4)
    else:
        gs = fig.add_gridspec(3, 2, width_ratios=[1.5, 1.0], height_ratios=[3.0, 1.0, 2.2])
        rows_c, rows_n = 2, 2
    ax = fig.add_subplot(gs[0, 0])
    axr = fig.add_subplot(gs[1, 0], sharex=ax)
    axd = None
    if curves:
        sub = gs[2, 0].subgridspec(1, 2)
        axd = (fig.add_subplot(sub[0, 0]), fig.add_subplot(sub[0, 1]))
    axc = fig.add_subplot(gs[rows_c, 0])
    axt = fig.add_subplot(gs[0:2, 1])
    axn = fig.add_subplot(gs[rows_n, 1])
    fig.suptitle(
        title or f"Fit result: {summary.backend}{'/' + summary.algorithm if summary.algorithm else ''}, "
        f"chi²/dof = {summary.chi2_red:.4g} (dof {summary.dof})", fontsize=12,
    )  # fmt: skip

    single = len(problem.datasets) == 1
    for k, d in enumerate(problem.datasets):
        color = f"C{k}"
        ax.errorbar(d.q, d.I, yerr=d.sigma, fmt="o", ms=3, color=color, ecolor=color, elinewidth=0.6, alpha=0.7, label=d.name)
        q_fit, I_fit = summary.curves[k]
        I_fit = np.asarray(I_fit, float)
        ax.plot(q_fit, I_fit, "-", color="black" if single else color, lw=1.6, label="fit" if single else f"{d.name} fit")
        with np.errstate(invalid="ignore", divide="ignore"):
            axr.plot(d.q, (I_fit - d.I) / d.sigma, "o", ms=2.5, color=color)
    if all(np.all(d.q > 0) for d in problem.datasets):
        ax.set_xscale("log")
    if all(np.all(d.I[np.isfinite(d.I)] > 0) for d in problem.datasets):
        ax.set_yscale("log")
    ax.set_ylabel("I(q)")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(fontsize=8)
    ax.tick_params(labelbottom=False)
    axr.axhline(0, color="0.4", lw=0.8)
    axr.set_ylabel("(fit - data) / σ")
    axr.set_xlabel("q")
    axr.grid(True, which="both", alpha=0.25)

    if axd is not None:  # N(x) and N(x) x^p of the fitted size distributions, side by side
        axd0, axd1 = axd
        for k, c in enumerate(curves):
            lab = c["label"] if len(curves) > 1 else None
            axd0.plot(c["x"], c["N"], "-", lw=1.6, color=f"C{k}", label=lab)
            axd1.plot(c["x"], c["Nxp"], "-", lw=1.6, color=f"C{k}", label=lab)
        for a, yl in ((axd0, "N(x)"), (axd1, f"N(x) x^{dist_power:g}")):
            if all(c["x"][0] > 0 for c in curves):
                a.set_xscale("log")
            a.set_xlabel("x")
            a.set_ylabel(yl)
            a.grid(True, which="both", alpha=0.25)
        axd0.set_title("fitted size distribution", fontsize=9)
        axd1.set_title(f"weighted with x^{dist_power:g} (p set in the explorer)", fontsize=9)
        if len(curves) > 1:
            axd1.legend(fontsize=7)

    # parameter table
    axt.axis("off")
    lines = [f"{'parameter':<22s}{'value':>12s}   error"]
    for p in summary.parameters:
        if p.fixed:
            err = "(fixed)"
        else:
            e = summary.errors.get(p.label)
            err = "n/a" if e is None or not math.isfinite(e) else f"{e:.3g}"
        flag = "  at a bound" if p.label in summary.at_bounds else ""
        lines.append(f"{p.label[:21]:<22s}{p.value:>12.5g}   ± {err}{flag}")
    fs = 9 if len(lines) <= 24 else max(5.0, 9 * 24 / len(lines))
    axt.text(0.0, 1.0, "\n".join(lines), family="monospace", fontsize=fs, va="top", ha="left", transform=axt.transAxes)

    # correlation matrix
    corr = summary.correlation()
    n = len(summary.free_labels)
    strong = []
    if corr is not None and n >= 2 and np.shape(corr) == (n, n) and np.all(np.isfinite(corr)):
        axc.imshow(corr, vmin=-1, vmax=1, cmap="RdBu_r")
        ticks = summary.free_labels if n <= 12 else [str(i + 1) for i in range(n)]
        axc.set_xticks(range(n), ticks, rotation=45, ha="right", fontsize=7)
        axc.set_yticks(range(n), ticks, fontsize=7)
        if n <= 10:
            for i in range(n):
                for j in range(n):
                    axc.text(j, i, f"{corr[i, j]:.2f}", ha="center", va="center", fontsize=7,
                             color="white" if abs(corr[i, j]) > 0.6 else "black")  # fmt: skip
        axc.set_title("correlation of the free parameters", fontsize=9)
        strong = sorted(((abs(corr[i, j]), i, j) for i in range(n) for j in range(i) if abs(corr[i, j]) >= 0.9), reverse=True)
    else:
        axc.axis("off")

    # convergence, data sets, constraints, warnings
    axn.axis("off")
    notes = [f"{'converged' if summary.success else 'NOT converged'}: {summary.message}",
             f"{summary.nfev} evaluations, {summary.seconds:.2f} s"]  # fmt: skip
    for name, npts, c2, c2r in summary.datasets:
        if len(summary.datasets) > 1:
            notes.append(f"data set {name}: {npts} points, chi²/n = {c2r:.4g}")
    for text, g, ok in summary.constraints:
        notes.append(f"constraint {text}: {'ok' if ok else 'VIOLATED'}")
    if strong:
        pairs = ", ".join(f"{summary.free_labels[j]}/{summary.free_labels[i]} ({corr[i, j]:+.2f})" for _r, i, j in strong[:6])
        notes.append("strongly correlated: " + pairs + ". Fix one of them at an independently known value or give one or "
                     "both a prior from independent knowledge.")  # fmt: skip
    wrapped = "\n".join(textwrap.fill(t, 62, subsequent_indent="  ") for t in notes)
    axn.text(0.0, 1.0, wrapped, fontsize=8, va="top", ha="left", transform=axn.transAxes)
    return fig


def figure_bytes(summary, problem, fmt="png", dpi=150, title=None, dist_power=DISTRIBUTION_POWER):
    """The summary figure as bytes of a PNG, PDF or SVG file."""
    buf = io.BytesIO()
    summary_figure(summary, problem, title, dist_power).savefig(buf, format=fmt, dpi=dpi)
    return buf.getvalue()


def save_figure(path, summary, problem, dpi=200, title=None, dist_power=DISTRIBUTION_POWER):
    """Write the summary figure to ``path``; the format follows the suffix (``.pdf``, ``.png`` or ``.svg``)."""
    fmt = os.path.splitext(str(path))[1].lower().lstrip(".") or "png"
    if fmt not in ("pdf", "png", "svg"):
        raise ValueError(f"unknown picture format '.{fmt}' (use .pdf, .png or .svg)")
    summary_figure(summary, problem, title, dist_power).savefig(str(path), format=fmt, dpi=dpi)
    return str(path)
