"""Last found installations of external programs (ATSAS, DENSS), kept between sessions.

Searching for an installation can be slow (DENSS: one Python subprocess per candidate interpreter;
ATSAS: the default install folders). The tools therefore search only when asked (their
"Search again" button, or the first run of a program); what was found is written to
``installations.json`` next to the preferences file and shown at the next start without a search.
"""

from __future__ import annotations

import json
import os
import time


def cache_path():
    from .settings import settings_path
    return os.path.join(os.path.dirname(settings_path()), "installations.json")


def _read():
    try:
        with open(cache_path(), encoding="utf-8") as fh:
            d = json.load(fh)
        return d if isinstance(d, dict) else {}
    except (OSError, ValueError):
        return {}


def get(key):
    """The entry ``key`` (a dict, with ``checked``: time of the search) or None."""
    v = _read().get(key)
    return v if isinstance(v, dict) else None


def put(key, value: dict):
    d = _read()
    d[key] = dict(value, checked=time.strftime("%Y-%m-%d %H:%M"))
    try:
        p = cache_path()
        os.makedirs(os.path.dirname(p), exist_ok=True)
        with open(p, "w", encoding="utf-8") as fh:
            json.dump(d, fh, indent=1)
    except OSError:
        pass


def forget(key):
    d = _read()
    if d.pop(key, None) is not None:
        try:
            with open(cache_path(), "w", encoding="utf-8") as fh:
                json.dump(d, fh, indent=1)
        except OSError:
            pass
