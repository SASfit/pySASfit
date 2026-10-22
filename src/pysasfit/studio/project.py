"""Save and reload an explorer session (``*.sasstudio``): mixture, parameters, fit selection, data, fit settings.

The file is plain JSON.  The data are stored inside the file, so a project stays usable when the original
data file moves.  The SASfit installation is *not* stored (it is a property of the computer).
"""

from __future__ import annotations

import json
import math
import os

import numpy as np

from .contribution_panel import ROLES

FORMAT = "pysasfit-studio-project"
VERSION = 1
SUFFIX = ".sasstudio"


def _num(x):
    x = float(x)
    return x if math.isfinite(x) else None


def _row(row):
    lo, hi = row.bounds_text()
    return {"value": float(row.value), "fit": bool(row.is_fit()), "lower": lo, "upper": hi}


def _set_row(row, d):
    if row is None or d is None:
        return
    row.lower.setText(str(d.get("lower", "")))
    row.lower._last = row.lower.text()
    row.upper.setText(str(d.get("upper", "")))
    row.upper._last = row.upper.text()
    row.set_value(d["value"], emit=True)
    row.set_fit(d.get("fit", False), emit=True)


def to_dict(ex):
    """The state of explorer ``ex`` as a JSON-able dictionary."""
    fp = ex.fit_panel
    contribs = []
    for p in ex.panels():
        c = {
            "ff": p.names["ff"], "sd": p.names["sd"], "sq": p.names["sq"], "approach": p.approach(),
            "scale": _row(p.scale_row), "parameters": {},
        }  # fmt: skip
        ff_rows = p.param_rows["ff"]
        c["distributed"] = ff_rows[p._dist].name if 0 <= p._dist < len(ff_rows) else None
        for role in ROLES:
            c["parameters"][role] = {r.name: _row(r) for r in p.param_rows[role]}
        contribs.append(c)
    d = {
        "format": FORMAT, "version": VERSION,
        "background": _row(ex.bg_row), "bg_A": _row(ex.bgA_row), "bg_alpha": _row(ex.bgalpha_row),
        "contributions": contribs, "current": ex.tabs.currentIndex(),
        "integration": dict(ex.integration),
        "plot": {
            "qmin": ex.qmin.text(), "qmax": ex.qmax.text(), "points": ex.npoints.value(), "units": ex.units.currentIndex(),
            "logx": ex.logx.isChecked(), "logy": ex.logy.isChecked(), "autoscale": ex.autoscale.isChecked(),
            "mode": ex.mode_combo.currentData(), "parts": ex.show_parts.isChecked(), "sliders": ex.show_sliders,
        },
        "data": None,
        "fit": {
            "q_scale": fp.q_scale.text(), "fit_qmin": fp.fit_qmin.text(), "fit_qmax": fp.fit_qmax.text(),
            "backend": fp.backend.currentData(), "algorithm": fp.algorithm.currentData(), "max_eval": fp.max_eval.value(),
            "polish": fp.polish.isChecked(), "vp": fp.vp.isChecked(), "cache": fp.cache_requested(), "coarse": fp.coarse.isChecked(), "grid": fp.grid.isChecked(), "constraints": fp.constraints.toPlainText(), "priors": fp.priors.toPlainText(),
            "result": fp.result.toPlainText(),
        },
    }  # fmt: skip
    if ex.data is not None:
        dt = ex.data
        d["data"] = {
            "name": dt["name"], "q": [_num(v) for v in dt["q"]], "I": [_num(v) for v in dt["I"]],
            "sigma": [_num(v) for v in dt["sigma"]],
        }  # fmt: skip
    return d


def save(ex, path):
    path = str(path)
    if not os.path.splitext(path)[1]:
        path += SUFFIX
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(to_dict(ex), fh, indent=1)
    return path


def load_dict(ex, d):
    """Rebuild the session of explorer ``ex`` from a dictionary written by :func:`to_dict`."""
    if d.get("format") != FORMAT:
        raise ValueError("not a pySASfit studio project file")
    if int(d.get("version", 1)) > VERSION:
        raise ValueError(f"the project was written by a newer version (format {d['version']})")
    pl = d.get("plot", {})
    if "integration" in d:
        ex.integration = {**ex.integration, **d["integration"]}
    old = ex.panels()
    for c in d["contributions"]:
        p = ex.add_contribution(c["ff"], c.get("sd"), c.get("sq"))
        p.set_approach(c.get("approach", "monodisperse"))
        ff_rows = {r.name: r for r in p.param_rows["ff"]}
        if c.get("distributed") in ff_rows and ff_rows[c["distributed"]].radio.isEnabled():
            ff_rows[c["distributed"]].radio.setChecked(True)
        for role in ROLES:
            rows = {r.name: r for r in p.param_rows[role]}
            for name, rd in c["parameters"].get(role, {}).items():
                _set_row(rows.get(name), rd)
        _set_row(p.scale_row, c.get("scale"))
    for w in old:
        ex.tabs.removeTab(ex.tabs.indexOf(w))
        w.deleteLater()
    ex._structure_changed()
    if 0 <= d.get("current", 0) < ex.tabs.count():
        ex.tabs.setCurrentIndex(d.get("current", 0))
    _set_row(ex.bg_row, d.get("background"))
    _set_row(ex.bgA_row, d.get("bg_A"))
    _set_row(ex.bgalpha_row, d.get("bg_alpha"))
    ex.clear_data()
    dt = d.get("data")
    if dt:
        nan = lambda v: np.array([np.nan if x is None else x for x in v], float)  # noqa: E731
        ex.set_data(nan(dt["q"]), nan(dt["I"]), nan(dt["sigma"]), dt.get("name", "data"))
        ex.fit_panel.data_label.setText(f"{ex.data['name']}: {ex.data['q'].size} points")
    ex.fit_panel._update_buttons()
    # plot settings (after the data: loading data resets the q range)
    if pl:
        ex.qmin.setText(pl.get("qmin", ex.qmin.text()))
        ex.qmax.setText(pl.get("qmax", ex.qmax.text()))
        ex.npoints.setValue(pl.get("points", ex.npoints.value()))
        ex.units.setCurrentIndex(pl.get("units", 0))
        ex.logx.setChecked(pl.get("logx", True))
        ex.logy.setChecked(pl.get("logy", True))
        ex.autoscale.setChecked(pl.get("autoscale", True))
        ex.show_parts.setChecked(pl.get("parts", True))
        i = ex.mode_combo.findData(pl.get("mode"))
        ex.mode_combo.setCurrentIndex(max(i, 0))
        ex.sliders_action.setChecked(bool(pl.get("sliders", False)))
    f = d.get("fit", {})
    fp = ex.fit_panel
    fp.q_scale.setText(f.get("q_scale", "1"))
    fp.fit_qmin.setText(f.get("fit_qmin", ""))
    fp.fit_qmax.setText(f.get("fit_qmax", ""))
    i = fp.backend.findData(f.get("backend"))
    if i >= 0:  # the optimiser may not be installed on this computer
        fp.backend.setCurrentIndex(i)
        j = fp.algorithm.findData(f.get("algorithm"))
        if j >= 0:
            fp.algorithm.setCurrentIndex(j)
    fp.max_eval.setValue(f.get("max_eval", 0))
    fp.polish.setChecked(f.get("polish", True))
    fp.vp.setChecked(bool(f.get("vp", False)))
    fp.cache.setChecked(bool(f.get("cache", False)))
    fp.coarse.setChecked(bool(f.get("coarse", False)))
    fp.grid.setChecked(bool(f.get("grid", False)))
    fp.constraints.setPlainText(f.get("constraints", ""))
    fp.priors.setPlainText(f.get("priors", ""))
    fp.result.setPlainText(f.get("result", ""))
    fp.refresh_table()
    ex._schedule()


def load(ex, path):
    with open(path, encoding="utf-8") as fh:
        d = json.load(fh)
    load_dict(ex, d)
