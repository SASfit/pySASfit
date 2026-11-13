"""Save and reload an explorer session (``*.sasstudio``): mixture, parameters, fit selection, data sets, global
parameters and links, fit settings.

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
VERSION = 1  # several data sets, global parameters and links are additions: older versions read the first data set
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
            "scale": _row(p.scale_row), "parameters": {}, "datasets": p.datasets_text(),
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
            "globals": [[n, v, f, _num(lo), _num(hi)] for n, v, f, lo, hi in fp.global_table_values()],
            "links": fp.links_edit.toPlainText(), "local": sorted(fp.local_addresses()),
        },
        "datasets": [], "current_ds": ex.current_ds,
        "transform": _tr(getattr(ex, "_tr_global", None)),
        "overrides": {str(k): v for k, v in ex.overrides.items()},
    }  # fmt: skip
    ex._store_bg()
    for dt in ex.datasets:
        item = {
            "name": dt["name"], "q": [_num(v) for v in dt["q"]], "I": [_num(v) for v in dt["I"]],
            "sigma": [_num(v) for v in dt["sigma"]], "bg": dict(dt["bg"]),
        }  # fmt: skip
        if dt.get("transform") is not None:  # SESANS / MSAS (additions: older versions ignore them)
            item["transform"] = _tr(dt["transform"])
        ses = dt.get("sesans")
        if ses:
            item["domain"] = "delta"
            item["sesans"] = {
                key: [[_num(v) for v in ses[key][0]], [_num(v) for v in ses[key][1]]] for key in ("G", "P") if ses.get(key)
            }  # fmt: skip
            if ses.get("lam") is not None:
                item["sesans"]["lam"] = [_num(v) for v in ses["lam"]]
            if ses.get("options"):
                item["sesans"]["options"] = dict(ses["options"])
        d["datasets"].append(item)
    if d["datasets"]:
        d["data"] = d["datasets"][0]  # read by version 1
    return d


def _tr(s):
    """Transform settings as JSON (numbers, strings, None)."""
    if not s:
        return None
    return {k: (_num(v) if isinstance(v, float) else v) for k, v in s.items()}


def save(ex, path):
    """Write the session of explorer ``ex`` to ``path`` (``.sasstudio`` is appended if there is no suffix); returns the path."""
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
        p.set_datasets_text(c.get("datasets", "all"))
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
    nan = lambda v: np.array([np.nan if x is None else x for x in v], float)  # noqa: E731
    sets = d.get("datasets") or ([d["data"]] if d.get("data") else [])
    if sets and sets[0].get("bg"):      # the rows hold the background of the data set being added
        b0 = sets[0]["bg"]
        ex.bg_row.set_value(b0["background"], emit=False)
        ex.bgA_row.set_value(b0["bg_A"], emit=False)
        ex.bgalpha_row.set_value(b0["bg_alpha"], emit=False)
    if d.get("transform") and hasattr(ex, "_tr_global"):
        ex._tr_global.update(d["transform"])
    for dt in sets:
        ses = None
        if dt.get("sesans"):
            ses = {key: tuple(nan(v) for v in dt["sesans"][key]) for key in ("G", "P") if dt["sesans"].get(key)}
            ses["lam"] = nan(dt["sesans"]["lam"]) if dt["sesans"].get("lam") else None
            ses["options"] = dt["sesans"].get("options")
        kw = {}
        if dt.get("transform") is not None or ses is not None:
            kw = {"transform": dt.get("transform"), "sesans": ses}
        ex.add_dataset(nan(dt["q"]), nan(dt["I"]), nan(dt["sigma"]), dt.get("name", "data"), bg=dt.get("bg"), **kw)
    if sets:
        ex.current_ds = 0
        if sets[0].get("bg") is None:      # version 1: the background rows hold that of the one data set
            ex._store_bg()
        cur = int(d.get("current_ds", 0))
        if 0 <= cur < len(ex.datasets):
            ex.current_ds = cur
            ex._load_bg(cur)
            ex._refresh_ds_combo()
        if hasattr(ex, "_load_tr"):
            ex._load_tr(ex.current_ds)
    elif hasattr(ex, "_load_tr"):
        ex._load_tr(None)
    ex.overrides = {int(k): {a: float(v) for a, v in ov.items()} for k, ov in (d.get("overrides") or {}).items()}
    ex.fit_panel.refresh_datasets()
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
    inf = lambda v, sgn: sgn * math.inf if v is None else float(v)  # noqa: E731
    fp.set_global_table_values([(n, v, fit, inf(lo, -1), inf(hi, 1)) for n, v, fit, lo, hi in f.get("globals", [])],
                               f.get("links", ""))
    fp.set_local_addresses(f.get("local", []))
    fp.refresh_table()
    ex._schedule()


def load(ex, path):
    """Read a project file and rebuild the session of explorer ``ex`` from it (see :func:`load_dict`)."""
    with open(path, encoding="utf-8") as fh:
        d = json.load(fh)
    load_dict(ex, d)
