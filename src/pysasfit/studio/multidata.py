"""Several data sets in the model explorer (as in SASfit's multiple data set fit).

* Every **contribution** is assigned to data sets: ``all`` (default) or a list such as ``1, 3`` or ``2-4``
  (data sets numbered from 1 as in the data list). The model curve of data set k is the sum of the contributions
  assigned to it, plus the background of data set k (each data set has its own background).
* **Global parameters** ``P1, P2, …`` are fit parameters of their own. A **link** sets a model parameter from them:
  ``c2.ff.R = 2 * P1`` (the radius of contribution 2 is twice P1), ``c1.sd.mu[2] = 0.5*P2`` (only in data set 2),
  ``background[1,3] = P3``, ``c1.ff.L = P1 + P2``, ``c1.ff.theta = atan(P1)``. The left side is the parameter label of
  the fit table (``ff.R``, ``c1.ff.R``, ``c1.scale``, ``background``, ``bg.A`` …), the right side any expression of the
  global parameters with + - * / ** and the functions sqrt exp log log10 abs min max sin cos tan asin acos atan atan2
  sinh cosh tanh (and pi). Constraints and priors may use the global parameters by name (``P1 + P2 < 100``).
* A fitted parameter is **common** (one value for all data sets) or **separate per data set** (one value each).

The functions here do the parsing and keep the per-data-set values; :mod:`.fit_panel` and :mod:`.explorer` use them.
"""

from __future__ import annotations

import math
import re
from contextlib import contextmanager
from dataclasses import dataclass

__all__ = ["parse_datasets", "format_datasets", "Link", "parse_link", "parse_links", "GlobalParameter",
           "overridden"]  # fmt: skip


def parse_datasets(text, n=None):
    """``"all"``/empty -> None (all data sets); ``"1, 3"``, ``"2-4"`` -> sorted 0-based indices. With ``n`` the
    numbers are checked against the number of data sets."""
    t = str(text or "").strip().lower()
    if t in ("", "all", "*"):
        return None
    out = set()
    for part in re.split(r"[,;\s]+", t):
        if not part:
            continue
        m = re.fullmatch(r"(\d+)(?:-(\d+))?", part)
        if not m:
            raise ValueError(f"data sets '{text}': write 'all' or numbers like 1, 3 or 2-4")
        a, b = int(m.group(1)), int(m.group(2) or m.group(1))
        if a < 1 or b < a:
            raise ValueError(f"data sets '{text}': numbers start at 1")
        out.update(range(a - 1, b))
    if n is not None:
        bad = sorted(k + 1 for k in out if k >= n)
        if bad:
            raise ValueError(f"data sets '{text}': there is no data set {', '.join(map(str, bad))} (have {n})")
    return sorted(out)


def format_datasets(ds):
    if ds is None:
        return "all"
    return ", ".join(str(k + 1) for k in ds)


@dataclass
class Link:
    target: str          # parameter label (fit table), e.g. 'c2.ff.R'
    datasets: object     # None (all) or list of 0-based indices
    factor: object       # float for the simple form 'factor * P', else None
    name: object         # the global parameter of the simple form, else None
    text: str = ""
    expression: str = ""  # right-hand side, e.g. '2 * P1', 'P1 + P2', 'atan(P1)'

    @property
    def names(self):
        """The global parameters the right-hand side uses."""
        from .fitting import _Expression

        return _Expression(self.expression).names

    def value(self, glob):
        """Value of the right-hand side for ``{name: value}`` of the global parameters."""
        from .fitting import _Expression

        return _Expression(self.expression)(glob)

    def describe(self):
        return f"= {self.expression}" + ("" if self.datasets is None else f" in data set {format_datasets(self.datasets)}")


_NUM = r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?"
_SIMPLE = re.compile(
    rf"^\s*(?:(?P<f1>{_NUM})\s*\*?\s*(?P<g1>[A-Za-z]\w*)|(?P<g2>[A-Za-z]\w*)\s*(?:\*\s*(?P<f2>{_NUM}))?|"
    rf"(?P<g3>[A-Za-z]\w*)\s*/\s*(?P<d3>{_NUM}))\s*$"
)
_LHS = re.compile(r"^\s*(?P<t>[A-Za-z][\w.]*)\s*(?:\[(?P<ds>[^\]]*)\])?\s*=(?!=)\s*(?P<rhs>.+?)\s*$")


def parse_link(line, n=None):
    """One link line (see the module docstring) -> :class:`Link`. The right-hand side is ``factor * P`` or any
    expression of global parameters: ``P1 + P2``, ``2*P1 - 1``, ``atan(P1)``, ``sqrt(P1**2 + P2**2)`` (functions:
    sqrt exp log log10 abs min max sin cos tan asin acos atan atan2 sinh cosh tanh, and pi)."""
    from .fitting import _Expression, _FUNCS

    m = _LHS.match(line)
    if not m:
        raise ValueError(f"link '{line.strip()}': write e.g.  c2.ff.R = 2 * P1   or   c1.ff.R = atan(P1)")
    rhs = m.group("rhs")
    factor = name = None
    sm = _SIMPLE.match(rhs)
    if sm and not ((sm.group("g1") or sm.group("g2") or sm.group("g3")) in _FUNCS):
        if sm.group("g1"):
            name, factor = sm.group("g1"), float(sm.group("f1"))
        elif sm.group("g2"):
            name, factor = sm.group("g2"), float(sm.group("f2") or 1.0)
        else:
            d = float(sm.group("d3"))
            if d == 0:
                raise ValueError(f"link '{line.strip()}': division by zero")
            name, factor = sm.group("g3"), 1.0 / d
    try:
        ex = _Expression(rhs)
    except ValueError as exc:
        raise ValueError(f"link '{line.strip()}': {exc}") from None
    if not ex.names:
        raise ValueError(f"link '{line.strip()}': the right-hand side uses no global parameter")
    ds = parse_datasets(m.group("ds"), n) if m.group("ds") is not None else None
    return Link(m.group("t"), ds, factor, name, line.strip(), rhs)


def parse_links(text, n=None):
    """All links of a text (one per line, ``#`` comments)."""
    out = []
    for line in str(text or "").splitlines():
        line = line.split("#", 1)[0]
        if line.strip():
            out.append(parse_link(line, n))
    return out


@dataclass
class GlobalParameter:
    name: str
    value: float = 1.0
    fit: bool = True
    lower: float = -math.inf
    upper: float = math.inf


@contextmanager
def overridden(model, values):
    """Set ``{address: value}`` in ``model`` (see :func:`pysasfit.studio.fitting.locate`) and restore afterwards."""
    from .fitting import locate

    saved = []
    try:
        for a, v in (values or {}).items():
            get, put = locate(model, a)
            saved.append((put, get()))
            put(v)
        yield model
    finally:
        for put, v in reversed(saved):
            put(v)
