# SPDX-License-Identifier: BSD-3-Clause
"""Archive of detector-efficiency maps in one HDF5 file, to follow the detector over time.

Every archived map is one NeXus entry of the archive file, named after the date of its water
measurement, e.g. ``/2026-10-02_SD-8.0m_6.00A_run82280`` with ``efficiency`` (NXdata: the map,
its errors, the filled / undefined / rejected pixels, the row and column profiles of the wires),
``instrument`` (the configuration of the water measurement: detector distance and tilt, pixel size,
beam centre, wavelength and its spread, collimation length, attenuator) and ``process`` (runs,
transmissions, comment, all metadata). The file attribute
``default`` points to the newest map, so viewers (pyTOOLbox) show it first. Summary numbers are
kept as attributes of each entry, so the archive can be listed without reading the maps::

    from pysasfit.sasred.efficiency_log import archive_efficiency, list_maps, load_map, history, plot_history
    archive_efficiency(eff, comment="after HV change")   # archive file: Preferences, or path=...
    list_maps()                     # one dict per map, oldest measurement first
    load_map("2026-10-02_SD-8.0m_6.00A_run82280")        # an Efficiency (default: the newest)
    history()                       # every map against the oldest one of its configuration (or reference=...)
    plot_history().savefig("efficiency_history.png")

The archive file is *Preferences → SANS reduction → efficiency archive file*; empty means
``efficiency_archive.h5`` next to the pySASfit preferences file. Command line::

    python -m pysasfit.sasred.efficiency_log [archive.h5] [--plot history.png]
"""
from __future__ import annotations

import os
import re
from datetime import datetime, timezone
from typing import Dict, List, Optional

import numpy as np

from .core.efficiency import Efficiency

__all__ = ["archive_path", "archive_efficiency", "list_maps", "load_map", "compare", "history", "plot_history"]

_SUMMARY = ("measured", "archived", "instrument", "bank", "detector_distance_m", "SD_m", "wavelength_A", "detector_tilt_deg", "water_runs",
            "empty_cell_runs",
            "n_measured", "n_filled", "n_rejected", "n_undefined", "rms", "min", "max", "median_sigma", "comment")


def archive_path(path: Optional[str] = None) -> str:
    """The archive file: ``path``, else the preference, else ``efficiency_archive.h5`` next to the
    preferences file (its folder is created if needed)."""
    if not path:
        from ..core import settings as prefs
        path = prefs.effective("sasred.efficiency.archive")
    folder = os.path.dirname(os.path.abspath(path))
    os.makedirs(folder, exist_ok=True)
    return str(path)


def _date(meta) -> str:
    """YYYY-MM-DD of the water measurement (today if unknown)."""
    m = re.search(r"\d{4}-\d{2}-\d{2}", str(meta.get("measured_start") or ""))
    return m.group(0) if m else datetime.now().strftime("%Y-%m-%d")


def _stats(eff: Efficiency) -> Dict[str, float]:
    v = eff.value.value[eff.measured]
    s = eff.std()[eff.measured]
    return dict(rms=float(np.std(v)), min=float(np.min(v)), max=float(np.max(v)), median_sigma=float(np.median(s)))


def archive_efficiency(eff: Efficiency, path: Optional[str] = None, instrument: Optional[str] = None,
                       comment: str = "") -> str:
    """Add ``eff`` to the archive file as a new entry named after the date of the water measurement
    (``YYYY-MM-DD_SD-<SD>m_<λ>A_run<first water run>``, suffix _2, _3, ... if the name exists).
    Nothing already in the archive is changed. Returns the entry name."""
    import h5py
    path = archive_path(path)
    meta = eff.meta
    instrument = instrument or {"sans_llb": "SANS-LLB", "sans_i": "SANS-I"}.get(
        str(meta.get("instrument", "")), str(meta.get("instrument", "")) or "SANS-I")
    bank = str(meta.get("bank", "") or "")
    date = _date(meta)
    SD = float(meta.get("SD_m") or 0)
    lam = float(meta.get("wavelength_A") or 0)
    runs = meta.get("water_runs") or [meta.get("water_run")]
    first = runs[0] if runs and runs[0] is not None else "x"
    base = re.sub(r"[^A-Za-z0-9_.-]", "_", f"{date}_SD-{SD:.1f}m_{lam:.2f}A_run{first}" + (f"_{bank}" if bank else ""))
    archived = datetime.now(timezone.utc).isoformat(timespec="seconds")
    eff.meta = dict(meta, instrument=instrument, archived=archived, comment=comment, measured=date)
    with h5py.File(path, "a") as f:
        name, k = base, 2
        while name in f:
            name = f"{base}_{k}"
            k += 1
        g = f.create_group(name)
        eff.write_entry(g, comment)
        tilt = meta.get("detector_tilt_deg")
        summary = dict(measured=date, archived=archived, instrument=instrument, bank=bank,
                       detector_distance_m=float(meta.get("detector_distance_m") or SD), SD_m=SD, wavelength_A=lam,
                       detector_tilt_deg=float("nan") if tilt is None else float(tilt),
                       water_runs=" ".join(map(str, meta.get("water_runs") or [])),
                       empty_cell_runs=" ".join(map(str, meta.get("empty_cell_runs") or [])),
                       n_measured=int(eff.measured.sum()), n_filled=int(eff.filled.sum()),
                       n_rejected=int(eff.rejected.sum()), n_undefined=int(eff.undefined.sum()),
                       comment=comment, **_stats(eff))
        for key, v in summary.items():
            g.attrs[key] = v
        g.attrs["title"] = f"detector efficiency {instrument}{' ' + bank + ' detector' if bank else ''} measured {date}"
        # the newest measurement is the default entry
        newest = max((n for n in f if "measured" in f[n].attrs),
                     key=lambda n: (str(f[n].attrs["measured"]), str(f[n].attrs.get("archived", ""))))
        f.attrs["default"] = newest
        f.attrs["NX_class"] = "NXroot"
        f.attrs["description"] = "pySASred archive of detector-efficiency maps, one entry per map"
    return name


def _attr(v):
    if isinstance(v, bytes):
        return v.decode()
    if isinstance(v, np.generic):
        return v.item()
    return v


def list_maps(path: Optional[str] = None) -> List[dict]:
    """The archived maps (summary of every entry plus ``entry``), oldest measurement first."""
    import h5py
    path = archive_path(path)
    if not os.path.exists(path):
        return []
    out = []
    with h5py.File(path, "r") as f:
        for n in f:
            if "efficiency" not in f[n]:
                continue
            from .legacy import rename_key
            row = {rename_key(k): _attr(v) for k, v in f[n].attrs.items() if rename_key(k) in _SUMMARY}
            row["entry"] = n
            out.append(row)
    return sorted(out, key=lambda r: (str(r.get("measured", "")), str(r.get("archived", ""))))


def load_map(entry: Optional[str] = None, path: Optional[str] = None) -> Efficiency:
    """One archived map (default: the newest measurement)."""
    return Efficiency.load(archive_path(path), entry=entry)


def compare(reference: Efficiency, other: Efficiency, n_sigma: float = 4.0) -> dict:
    """Change of ``other`` against ``reference`` on the pixels measured in both.

    Returns a dict with ``ratio`` (map other / reference, NaN elsewhere), ``rms_change``,
    ``mean_change`` and ``max_change`` (of ratio − 1), ``expected_rms`` (the rms change expected from
    the errors of both maps alone), ``chi2`` (mean squared change in units of the combined standard
    deviation; about 1 if nothing changed but noise), ``n_changed`` (pixels changed by more than
    ``n_sigma`` combined standard deviations), ``row_ratio`` / ``column_ratio`` (ratio of the wire
    profiles) and ``changed_rows`` / ``changed_columns`` (wires whose profile changed by more than
    ``n_sigma`` standard errors)."""
    if reference.shape != other.shape:
        raise ValueError("maps of different detector shapes")
    both = reference.measured & other.measured
    a, b = reference.value.value, other.value.value
    sa, sb = reference.std(), other.std()
    ratio = np.full(a.shape, np.nan)
    ratio[both] = b[both] / a[both]
    sig = np.full(a.shape, np.nan)
    sig[both] = np.hypot(sb[both] / a[both], b[both] * sa[both] / a[both] ** 2)
    with np.errstate(invalid="ignore", divide="ignore"):
        z = np.abs(ratio - 1) / sig
    ra, ca = reference.profiles()
    rb, cb = other.profiles()

    def wire_changes(pa, pb, axis):
        n = both.sum(axis=axis).astype(float)
        with np.errstate(invalid="ignore", divide="ignore"):
            s = np.sqrt(np.nansum(np.where(both, sig, np.nan) ** 2, axis=axis)) / np.maximum(n, 1)
            r = pb / pa
        return r, [int(i) for i in np.flatnonzero((n > 0) & (np.abs(r - 1) > n_sigma * s))]

    row_ratio, rows = wire_changes(ra, rb, 1)
    col_ratio, cols = wire_changes(ca, cb, 0)
    d = ratio[both] - 1
    return dict(ratio=ratio, rms_change=float(np.std(d)), mean_change=float(np.mean(d)),
                expected_rms=float(np.sqrt(np.mean(sig[both] ** 2))), chi2=float(np.mean(z[both] ** 2)),
                max_change=float(np.max(np.abs(d))), n_changed=int(np.sum(z[both] > n_sigma)),
                n_compared=int(both.sum()), row_ratio=row_ratio, column_ratio=col_ratio,
                changed_rows=rows, changed_columns=cols)


def _same_config(a: dict, b: dict) -> bool:
    def f(r, k):
        try:
            return float(r.get(k))
        except (TypeError, ValueError):
            return float("nan")
    SDa, SDb = f(a, "SD_m"), f(b, "SD_m")
    la, lb = f(a, "wavelength_A"), f(b, "wavelength_A")
    ta, tb = f(a, "detector_tilt_deg"), f(b, "detector_tilt_deg")
    if str(a.get("bank", "") or "") != str(b.get("bank", "") or ""):        # SANS-LLB: other detector bank
        return False
    ok = abs(SDa - SDb) <= 0.01 * max(SDa, SDb) and abs(la - lb) <= 0.02 * max(la, lb)
    return bool(ok and (np.isnan(ta) or np.isnan(tb) or abs(ta - tb) <= 0.05))


def history(path: Optional[str] = None, reference: Optional[str] = None, n_sigma: float = 4.0) -> List[dict]:
    """Every archived map compared with a reference: ``reference`` (an entry name) for all maps, or
    by default the oldest map with the same configuration (detector distance, wavelength and tilt),
    since maps measured in other configurations differ also by the geometry corrections.
    Returns the summaries of :func:`list_maps` extended by the results of :func:`compare`."""
    path = archive_path(path)
    rows = list_maps(path)
    if not rows:
        return []
    cache = {}

    def get(name):
        if name not in cache:
            cache[name] = Efficiency.load(path, entry=name)
        return cache[name]

    out = []
    for r in rows:
        same_bank = [x for x in rows if x["entry"] == reference and str(x.get("bank", "") or "") == str(r.get("bank", "") or "")]
        ref_name = same_bank[0]["entry"] if same_bank else next(x["entry"] for x in rows if _same_config(x, r))
        c = compare(get(ref_name), get(r["entry"]), n_sigma)
        out.append(dict(r, reference=ref_name, rms_change=c["rms_change"], mean_change=c["mean_change"],
                        expected_rms=c["expected_rms"], chi2=c["chi2"],
                        max_change=c["max_change"], n_changed=c["n_changed"],
                        changed_rows=c["changed_rows"], changed_columns=c["changed_columns"]))
    return out


def plot_history(path: Optional[str] = None, reference: Optional[str] = None, fig=None):
    """Figure: rms and mean change of every map against the reference over time (top), and the row
    (y wire) and column (x wire) profiles of all maps, coloured by date (bottom)."""
    import matplotlib.dates as mdates
    from matplotlib import cm
    from matplotlib.figure import Figure
    path = archive_path(path)
    h = history(path, reference)
    fig = fig or Figure(figsize=(9, 7), layout="constrained")
    ax0 = fig.add_subplot(2, 1, 1)
    axr = fig.add_subplot(2, 2, 3)
    axc = fig.add_subplot(2, 2, 4)
    if not h:
        ax0.set_title("no efficiency maps archived")
        return fig
    dates = [datetime.strptime(str(r["measured"]), "%Y-%m-%d") for r in h]
    ax0.plot(dates, [100 * r["rms_change"] for r in h], "o-", label="rms change")
    ax0.plot(dates, [100 * r["expected_rms"] for r in h], ":", color="grey", label="expected from the errors")
    ax0.plot(dates, [100 * r["mean_change"] for r in h], "s--", label="mean change")
    ax0.set_ylabel("change against the reference [%]" if reference is None else f"vs {reference} [%]",
                   fontsize="small")
    ax0.legend(fontsize="small")
    ax0.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    ax0.set_title("detector efficiency over time")
    colours = cm.viridis(np.linspace(0, 1, len(h)))
    for r, col in zip(h, colours):
        rows, cols = Efficiency.load(path, entry=r["entry"]).profiles()
        axr.plot(rows, color=col, lw=0.8, label=str(r["measured"]))
        axc.plot(cols, color=col, lw=0.8)
    axr.set_xlabel("row (y wire)")
    axc.set_xlabel("column (x wire)")
    axr.set_ylabel("mean efficiency")
    if len(h) <= 8:
        axr.legend(fontsize="x-small")
    return fig


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(description="List the archived detector-efficiency maps and their change over time.")
    ap.add_argument("archive", nargs="?", default=None, help="archive file (default: from the preferences)")
    ap.add_argument("--reference", default=None,
                    help="entry name of the reference map (default: the oldest of the same configuration)")
    ap.add_argument("--plot", default=None, help="write the history plot to this file (png, pdf, svg)")
    a = ap.parse_args(argv)
    path = archive_path(a.archive)
    h = history(path, a.reference)
    if not h:
        print("no efficiency maps in", path)
        return 0
    print(f"{'entry':36} {'SD/m':>5} {'λ/Å':>5} {'tilt/°':>6} {'Δrms %':>7} {'noise %':>7} {'χ²':>5} "
          f"{'Δmean %':>8} {'n>4σ':>5}  rows / columns changed")
    for r in h:
        print(f"{r['entry'][:36]:36} {float(r['SD_m']):5.1f} {float(r['wavelength_A']):5.2f} "
              f"{float(r.get('detector_tilt_deg', float('nan'))):6.2f} {100 * r['rms_change']:7.3f} "
              f"{100 * r['expected_rms']:7.3f} {r['chi2']:5.2f} {100 * r['mean_change']:8.3f} {r['n_changed']:5d}  "
              f"{r['changed_rows']} / {r['changed_columns']}")
    if a.plot:
        plot_history(path, a.reference).savefig(a.plot, dpi=120)
        print("plot:", a.plot)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
