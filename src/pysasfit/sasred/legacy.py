# SPDX-License-Identifier: BSD-3-Clause
"""Names used before 2026-11-04: the collimation length was ``L1`` and the sample-detector distance
``L2``; they are now ``coll`` and ``SD`` throughout pySASfit. Files written before (efficiency maps and
their archive, reduced NXcanSAS / text files, saved reduction tables, the session) still carry the old
names; these helpers translate them when such a file is read."""
from __future__ import annotations

import re

__all__ = ["rename_key", "rename_keys", "rename_config"]


def rename_key(k):
    """'L2_m' -> 'SD_m', 'L1_m' -> 'coll_m', 'L2' -> 'SD', ... (other keys unchanged)."""
    if not isinstance(k, str):
        return k
    k = re.sub(r"^L2(?=_|$)", "SD", k)
    return re.sub(r"^L1(?=_|$)", "coll", k)


def rename_keys(d):
    """A dict (nested dicts and lists too) with the old keys renamed; a key that exists under both names
    keeps the new one."""
    if isinstance(d, dict):
        out = {}
        for k, v in d.items():
            nk = rename_key(k)
            if nk != k and nk in d:
                continue
            out[nk] = rename_keys(v)
        return out
    if isinstance(d, list):
        return [rename_keys(v) for v in d]
    return d


def rename_config(s):
    """A configuration text of the old form 'L2 8.0 m, λ 6.00 Å, L1 8.0 m' (also inside an efficiency
    choice 'geometry:…' or a tab title) in the new form 'SD 8.0 m, λ 6.00 Å, coll 8.0 m'."""
    if not isinstance(s, str):
        return s
    return re.sub(r"\bL1(?= )", "coll", re.sub(r"\bL2(?= )", "SD", s))
