"""Data input: q, I [, sigma] -- three-column files as everywhere in pysasfit, and also
the two-column files SAXSMorph reads (then sigma is None)."""

from __future__ import annotations

import numpy as np

from ..core.io import load_data

__all__ = ["load_curve"]


def load_curve(filename, q_scale=1.0):
    """Returns (q, I, sigma_or_None). Background must already be subtracted."""
    try:
        q, I, s, _dq = load_data(filename, q_scale=q_scale, skip_nonpositive_sigma=True)
        return q, I, s
    except ValueError:
        pass
    rows = []
    with open(filename, "r", errors="replace") as fh:
        for line in fh:
            line = line.strip()
            if not line or line[0] in "#%" or line.startswith("//"):
                continue
            for sep in (None, ",", ";", "\t"):
                parts = line.split(sep) if sep else line.split()
                if len(parts) >= 2:
                    try:
                        rows.append([float(x) for x in parts[:2]])
                    except ValueError:
                        continue
                    break
    if not rows:
        raise ValueError(f"no numeric q, I rows in {filename}")
    a = np.array(rows)
    q, I = a[:, 0] * q_scale, a[:, 1]
    m = np.isfinite(q) & np.isfinite(I) & (q > 0)
    o = np.argsort(q[m])
    return q[m][o], I[m][o], None
