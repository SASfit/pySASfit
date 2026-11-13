"""The results store of the scout: every fitted candidate and every search is kept on disk and reused.

Layout (plain files, relative paths only: the folder can be copied or moved, two stores are merged by copying)::

    <store>/
      store.json                         format version
      data/<data id>/                    one folder per curve (id: first 16 hex digits of the hash of q, I, sigma)
        data.json                        name, original file, points, q range, first and last use
        data.dat                         the curve itself (q, I, sigma)
        fits/<key>.json                  one fitted candidate: parameters, errors, criteria, warnings, curve
        runs/<UTC time>.json             one search: options, features, notes and the ranked candidates

The ``key`` of a fit is a hash of everything the result depends on: the candidate (functions, fitted and fixed
parameters as registered), the fit options (starts, seed, background, nodes of the size integral, error scale,
start values of a mixture), the functions used (SASfit libraries with their build time, or the Python reference)
and :data:`FIT_VERSION`. A search looks every candidate up before fitting it, so repeating a search, continuing
one that was stopped, or adding models only fits what is new. After an update of SASfit the old fits stay
readable but are not reused.

One file per fit, written atomically: worker processes and jobs on a cluster can write into the same store
without locking. Default location: ``scout_store`` next to the pySASfit preferences file (:func:`default_dir`);
the window uses the folder of Settings -> Preferences -> pySASscout.
"""

from __future__ import annotations

import datetime
import hashlib
import json
import os
import shutil
import time

import numpy as np

__all__ = ["ResultStore", "default_dir", "data_id", "FIT_VERSION"]

FORMAT = 1
#: raise when the way a candidate is fitted changes (start values, bounds, multistart): older fits are then not reused
FIT_VERSION = 1


def default_dir():
    """``scout_store`` next to the pySASfit preferences file (in the user profile)."""
    from ..core import settings as prefs

    return os.path.join(os.path.dirname(prefs.settings_path()), "scout_store")


def data_id(q, I, sigma):
    """Identifier of a curve: the first 16 hex digits of the SHA-256 of its values."""
    a = np.ascontiguousarray(np.vstack([np.asarray(q, float), np.asarray(I, float), np.asarray(sigma, float)]))
    return hashlib.sha256(a.tobytes()).hexdigest()[:16]


def _functions_stamp(reference):
    """What evaluates the models: the Python reference functions, or the SASfit libraries and their build time."""
    if reference:
        return "python reference"
    try:
        from ..studio import paths

        lib = paths.lib_dir()
        if not lib:
            return "sasfit: not installed"
        newest = 0.0
        for root, _dirs, files in os.walk(lib):
            for f in files:
                if f.lower().endswith((".dll", ".so", ".dylib")):
                    newest = max(newest, os.path.getmtime(os.path.join(root, f)))
        return f"sasfit: built {int(newest)}"            # not the path: the store may move to another computer
    except Exception:  # noqa: BLE001
        return "sasfit: unknown"


def _write_json(path, obj):
    """Atomic: a reader never sees a half-written file, two writers of the same key do not mix."""
    tmp = f"{path}.{os.getpid()}.{time.time_ns()}.tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=1)
    os.replace(tmp, path)


def _read_json(path):
    try:
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return None


class ResultStore:
    def __init__(self, root=None):
        self.root = os.path.abspath(root or default_dir())

    # ------------------------------------------------------------------ paths
    def _data_dir(self, did):
        return os.path.join(self.root, "data", did)

    def _ensure(self, did):
        d = self._data_dir(did)
        os.makedirs(os.path.join(d, "fits"), exist_ok=True)
        os.makedirs(os.path.join(d, "runs"), exist_ok=True)
        marker = os.path.join(self.root, "store.json")
        if not os.path.isfile(marker):
            _write_json(marker, {"program": "pySASscout", "format": FORMAT})
        return d

    # ------------------------------------------------------------------ fits
    @staticmethod
    def fit_key(components, options, start=None, error_scale=1.0):
        """Key of the fit of ``components`` with these options (see the module docstring)."""
        from .candidates import DISTRIBUTIONS, FORMS, STRUCTURE_FACTORS

        specs = []
        for c in components:
            specs.append([repr(FORMS[c.form]), repr(DISTRIBUTIONS[c.sd]) if c.sd else None,
                          repr(STRUCTURE_FACTORS[c.sq]) if c.sq else None])  # fmt: skip
        res = options.resolution
        what = {
            "fit_version": FIT_VERSION, "specs": specs,
            "options": [options.starts, options.max_starts, options.seed, bool(options.background), options.n_x,
                        options.n_x_final],
            "error_scale": float(f"{float(error_scale):.12g}"),
            "start": sorted((a, float(f"{v:.12g}")) for a, v in (start or {}).items()),
            "resolution": None if res is None else repr(res),
            "functions": _functions_stamp(options.reference),
        }
        return hashlib.sha256(json.dumps(what, sort_keys=True).encode()).hexdigest()[:24]

    def get_fit(self, did, key):
        """The stored :class:`~pysasfit.scout.search.Candidate` of this key, or None."""
        from .search import candidate_from_dict

        d = _read_json(os.path.join(self._data_dir(did), "fits", key + ".json"))
        if not d:
            return None
        try:
            c = candidate_from_dict(d)
        except Exception:  # noqa: BLE001 -- a file of another version: fit again
            return None
        c.stored = True
        return c

    def put_fit(self, did, key, candidate):
        d = self._ensure(did)
        obj = candidate.as_dict(curves=True)
        obj["key"], obj["saved"] = key, _now()
        _write_json(os.path.join(d, "fits", key + ".json"), obj)

    # ------------------------------------------------------------------ data and runs
    def register_data(self, did, q, I, sigma, name=None, path=None):
        d = self._ensure(did)
        fn = os.path.join(d, "data.json")
        info = _read_json(fn) or {"first_used": _now()}
        info.update({"id": did, "name": name or info.get("name") or "data", "points": int(len(q)),
                     "q_min": float(q[0]), "q_max": float(q[-1]), "last_used": _now()})
        if path:
            info["file"] = path
        _write_json(fn, info)
        dat = os.path.join(d, "data.dat")
        if not os.path.isfile(dat):
            tmp = f"{dat}.{os.getpid()}.tmp"
            np.savetxt(tmp, np.column_stack([q, I, sigma]), header="q I sigma")
            os.replace(tmp, dat)

    def save_run(self, did, result, keep=None):
        """Store a :class:`~pysasfit.scout.search.ScoutResult`; ``keep``: only this many newest runs stay."""
        d = self._ensure(did)
        run = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
        _write_json(os.path.join(d, "runs", run + ".json"), result.as_dict(curves=True))
        if keep:
            for old in self.runs(did)[keep:]:
                try:
                    os.remove(os.path.join(d, "runs", old + ".json"))
                except OSError:
                    pass
        return run

    def runs(self, did):
        """Identifiers (UTC times) of the stored searches of this curve, newest first."""
        try:
            names = os.listdir(os.path.join(self._data_dir(did), "runs"))
        except OSError:
            return []
        return sorted((n[:-5] for n in names if n.endswith(".json")), reverse=True)

    def load_run(self, did, run=None):
        """The stored search ``run`` (default: the newest) as a ScoutResult, or None."""
        from .search import result_from_dict

        runs = self.runs(did)
        if not runs or (run is not None and run not in runs):
            return None
        d = _read_json(os.path.join(self._data_dir(did), "runs", (run or runs[0]) + ".json"))
        if not d:
            return None
        res = result_from_dict(d)
        res.store = {"folder": self.root, "data_id": did, "run": run or runs[0]}
        try:
            a = np.loadtxt(os.path.join(self._data_dir(did), "data.dat"))
            res._data = (a[:, 0], a[:, 1], a[:, 2] * float(d.get("data", {}).get("error_scale", 1.0)))
        except (OSError, ValueError, IndexError):
            pass
        return res

    # ------------------------------------------------------------------ overview and housekeeping
    def datasets(self):
        """``[{id, name, file, points, last_used, fits, runs, bytes}]`` of every curve in the store, newest first."""
        base = os.path.join(self.root, "data")
        out = []
        try:
            ids = os.listdir(base)
        except OSError:
            return out
        for did in ids:
            d = os.path.join(base, did)
            info = _read_json(os.path.join(d, "data.json"))
            if not info:
                continue
            fits = _files(os.path.join(d, "fits"))
            runs = _files(os.path.join(d, "runs"))
            info.update(id=did, fits=len(fits), runs=len(runs),
                        bytes=sum(s for _n, s in fits + runs + _files(d)))
            out.append(info)
        return sorted(out, key=lambda i: i.get("last_used", ""), reverse=True)

    def summary(self):
        """``{folder, datasets, fits, runs, bytes}``."""
        ds = self.datasets()
        return {"folder": self.root, "datasets": len(ds), "fits": sum(d["fits"] for d in ds),
                "runs": sum(d["runs"] for d in ds), "bytes": sum(d["bytes"] for d in ds)}

    def summary_text(self):
        s = self.summary()
        return (f"{s['datasets']} curve{'s' if s['datasets'] != 1 else ''}, {s['fits']} fits, {s['runs']} searches, "
                f"{_size(s['bytes'])}")

    def prune(self, older_than_days=None, did=None):
        """Delete the stored results of one curve (``did``) and / or of all curves not used for
        ``older_than_days``; returns the number of curves removed. With neither argument nothing is deleted."""
        if older_than_days is None and did is None:
            return 0
        limit = None
        if older_than_days is not None:
            limit = (datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(days=older_than_days)).isoformat()
        n = 0
        for info in self.datasets():
            if did is not None and info["id"] != did:
                continue
            if limit is not None and info.get("last_used", "") >= limit:
                continue
            shutil.rmtree(self._data_dir(info["id"]), ignore_errors=True)
            n += 1
        return n


def _now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")


def _files(folder):
    out = []
    try:
        with os.scandir(folder) as it:
            for e in it:
                if e.is_file():
                    out.append((e.name, e.stat().st_size))
    except OSError:
        pass
    return out


def _size(n):
    for unit in ("bytes", "kB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:.0f} {unit}" if unit == "bytes" else f"{n:.1f} {unit}"
        n /= 1024.0
