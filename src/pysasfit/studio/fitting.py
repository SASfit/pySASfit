r"""Fitting model functions to one or several data sets.

A :class:`FitProblem` couples data sets with models (:class:`~pysasfit.studio.ModelFunction`) and a
table of fit parameters:

* a **common** parameter drives the same model parameter in several (default: all) data sets;
* a **local** parameter belongs to one data set only (``add_local`` creates one per data set);
* a **global** parameter (``add_global``, e.g. ``P1``) is not a model parameter itself: ``link`` makes model parameters
  of any contributions and data sets follow it, each as ``factor * P1`` (SASfit's global parameters with a factor);
  constraints and priors may use it like any other parameter;
* ``fixed`` parameters keep their value, ``lower``/``upper`` are bounds;
* **constraints** are inequalities or equalities of the parameter names (``"ff_R + ff_t < 100"``,
  ``"scale_0_a + scale_0_b == 1"``): the label of a parameter with every run of non-word characters turned into ``_``,
  and a ``c`` in front if it starts with a digit (``1.sd.mu`` -> ``c1_sd_mu``), see :meth:`FitProblem.identifiers`;
* **priors** (``add_prior``, ``add_prior_text``) add a term to chi^2: ``normal`` (mean, sigma), ``lognormal`` (median, factor > 1),
  ``halfnormal`` (sigma, for x >= 0) as extra residuals, and ``loguniform`` (flat in ln x, lower bound > 0 needed) as a scalar
  term ``2 ln x`` that the least-squares backends (``scipy``, ``dfols``) cannot use.

Parameters are addressed like in :class:`~pysasfit.studio.Pipeline`: ``"background"``, ``"scale.<i>"`` and
``"<i>.<ff|sd|sq>.<name>"`` for contribution ``i`` of a model.

    prob = FitProblem()
    prob.add_dataset("a.dat", model_a, name="a")
    prob.add_dataset("b.dat", model_b, name="b")
    prob.add_common("0.ff.R", lower=1, upper=200)       # same radius in both data sets
    prob.add_local("scale.0", lower=0)                  # one scale factor per data set
    prob.add_local("background")
    prob.add_constraint("scale_0_a + scale_0_b == 1")   # labels 'scale.0[a]', 'scale.0[b]'
    result = prob.fit("scipy")                          # or prob.fit("nlopt", "LN_BOBYQA")
    print(result.text())

Backends (see :func:`available_backends` for those installed): ``"scipy"`` (:func:`scipy.optimize.least_squares`),
``"scipy-global"``, ``"nlopt"`` (any NLopt algorithm, chi^2 as scalar objective), ``"iminuit"``, ``"bumps"``,
``"cma"`` and ``"dfols"`` (DFO-LS, derivative-free least squares). More can be registered with
:func:`register_backend`. The residuals are
``(model - data) / sigma``; the uncertainties come from the numerical Jacobian at the optimum for
every backend.

Two options speed up fits of mixtures and size distributions (both off by default, results are the same):

* ``vp=True`` -- **variable projection**: the parameters the intensity is linear in (``background``, ``scale.<i>`` and
  the amplitude ``N`` of a size distribution) are not searched by the optimiser; for every trial of the other
  parameters they are obtained exactly by linear least squares (bounds respected). Linear parameters that appear
  in a constraint or a prior stay with the optimiser. The errors are those of the full problem.
* ``cache=True`` -- **form-factor cache**: the intensity of a contribution and the form factor on the integration
  nodes are remembered, so a change of one contribution does not recompute the others and a change of the size
  distribution does not recompute the form factor. ``snap`` (default 0.05) rounds the automatically found
  integration range outwards to a geometric grid of that relative step, so that the nodes are reused.

Models evaluated for different data sets may be the same object: the values of the parameters of a data
set are set in the model right before it is evaluated. After a fit the models hold the values of the
first data set (``FitProblem.apply(k)`` sets those of data set ``k``).
"""

from __future__ import annotations

import ast
import math
import re
import time
from dataclasses import dataclass, field

import numpy as np

__all__ = [
    "FitProblem", "FitParameter", "FitDataset", "FitSummary", "FitCancelled", "BACKENDS", "SCIPY_ALGORITHMS",
    "nlopt_algorithms", "register_backend", "locate", "available_backends",
]  # fmt: skip

BIG = 1e6  # residual of a point where the model could not be evaluated
CONSTRAINT_TOL = 1e-3  # a constraint g(x) <= 0 counts as satisfied up to this (the scipy penalty leaves a little)
SCIPY_ALGORITHMS = ("trf", "dogbox", "lm")
_NLOPT_ALGORITHMS = {  # name -> short description (only those the installed NLopt has are offered)
    "LN_BOBYQA": "local, derivative-free, bounds",
    "LN_NELDERMEAD": "local, simplex",
    "LN_SBPLX": "local, subplex (robust against noise)",
    "LN_COBYLA": "local, derivative-free, inequality constraints",
    "LN_PRAXIS": "local, principal axis",
    "LD_SLSQP": "local, gradient (finite differences), inequality constraints",
    "LD_LBFGS": "local, gradient (finite differences)",
    "LD_MMA": "local, gradient (finite differences), inequality constraints",
    "GN_DIRECT_L": "global, deterministic (finite bounds needed)",
    "GN_CRS2_LM": "global, controlled random search (finite bounds needed)",
    "GN_ISRES": "global, evolutionary, inequality constraints (finite bounds needed)",
    "GN_ESCH": "global, evolutionary (finite bounds needed)",
}
_NLOPT_CONSTRAINTS = {"LN_COBYLA", "LD_SLSQP", "LD_MMA", "GN_ISRES"}  # others: wrapped in the augmented Lagrangian
_NLOPT_EQUALITY = {"LN_COBYLA", "LD_SLSQP", "GN_ISRES"}  # algorithms that take equality constraints themselves


class FitCancelled(Exception):
    """Raised inside the optimiser when the cancel event is set."""


def nlopt_algorithms():
    """``{name: description}`` of the NLopt algorithms offered here (empty without NLopt)."""
    try:
        import nlopt
    except ImportError:
        return {}
    return {k: v for k, v in _NLOPT_ALGORITHMS.items() if hasattr(nlopt, k)}


# --------------------------------------------------------------------------- addressing
def locate(model, address):
    """``(getter, setter)`` of the model parameter ``address`` (see the module docstring)."""
    from .model import ScatteringContribution

    if address == "background":
        if not np.isscalar(model.background):
            raise ValueError("only a constant background can be fitted")
        return (lambda: float(model.background), lambda v: setattr(model, "background", float(v)))
    if address in ("bg.A", "bg.alpha"):  # power-law background  A * q**(-alpha)
        attr = "bg_" + address[3:]
        return (lambda: float(getattr(model, attr)), lambda v: setattr(model, attr, float(v)))
    parts = address.split(".")
    if parts[0] == "scale":
        i = int(parts[1])
        return (lambda: model.scales[i], lambda v: model.scales.__setitem__(i, float(v)))
    try:
        i, role, key = int(parts[0]), parts[1], ".".join(parts[2:])
        c = model.contributions[i]
    except (ValueError, IndexError):
        raise KeyError(f"bad parameter address '{address}'") from None
    f = c.function_for(role)
    if f is None:
        raise KeyError(f"contribution {i} has no {role}")
    k = ScatteringContribution._index(f, key)
    return (lambda: float(c.vector(role)[k]), lambda v: c.vector(role).__setitem__(k, float(v)))


def _ident(label):
    """Name of a parameter in constraint expressions: the label with ``.`` etc. replaced by ``_``; a label that starts
    with a digit (``1.sd.mu`` of the second contribution) gets a ``c`` in front (``c1_sd_mu``), as the expression
    must be valid Python."""
    name = re.sub(r"\W+", "_", label).strip("_")
    return "c" + name if name[:1].isdigit() else name


# --------------------------------------------------------------------------- data classes
@dataclass
class FitDataset:
    name: str
    q: np.ndarray
    I: np.ndarray
    sigma: np.ndarray
    model: object
    weight: float = 1.0


@dataclass
class FitParameter:
    label: str
    address: str
    datasets: tuple
    value: float
    lower: float = -math.inf
    upper: float = math.inf
    fixed: bool = False
    links: tuple = ()  # global parameter: ((data set, address, factor), ...) -- the model value is factor * value

    @property
    def ident(self):
        return _ident(self.label)

    def targets(self):
        """``[(data set, address, factor)]`` this parameter drives."""
        out = [(k, self.address, 1.0) for k in self.datasets] if self.address else []
        return out + [tuple(t) for t in self.links]


_FUNCS = {"sqrt": math.sqrt, "exp": math.exp, "log": math.log, "log10": math.log10, "abs": abs, "min": min, "max": max,
          "sin": math.sin, "cos": math.cos, "tan": math.tan, "asin": math.asin, "acos": math.acos, "atan": math.atan,
          "atan2": math.atan2, "sinh": math.sinh, "cosh": math.cosh, "tanh": math.tanh, "pi": math.pi}
_NODES = (ast.Expression, ast.BinOp, ast.UnaryOp, ast.Name, ast.Constant, ast.Call, ast.Load, ast.Add, ast.Sub,
          ast.Mult, ast.Div, ast.Pow, ast.Mod, ast.USub, ast.UAdd)  # fmt: skip


class _Constraint:
    """``lhs < rhs`` (or <=, >, >=) as g(x) = lhs - rhs <= 0 (feasible); ``lhs == rhs`` as g(x) = 0."""

    def __init__(self, text):
        self.text = text.strip()
        try:
            tree = ast.parse(self.text, mode="eval").body
        except SyntaxError as exc:
            raise ValueError(f"constraint '{text}': {exc.msg}") from None
        if not (isinstance(tree, ast.Compare) and len(tree.ops) == 1 and isinstance(tree.ops[0], (ast.Lt, ast.LtE, ast.Gt, ast.GtE, ast.Eq))):
            raise ValueError(f"constraint '{text}': write one inequality or equality, e.g. 'ff_R + ff_t < 100'")
        self.equality = isinstance(tree.ops[0], ast.Eq)
        lhs, rhs = tree.left, tree.comparators[0]
        if isinstance(tree.ops[0], (ast.Gt, ast.GtE)):
            lhs, rhs = rhs, lhs
        self.names = set()
        for side in (lhs, rhs):
            for node in ast.walk(side):
                if not isinstance(node, _NODES):
                    raise ValueError(f"constraint '{text}': '{type(node).__name__}' is not allowed")
                if isinstance(node, ast.Call) and not (isinstance(node.func, ast.Name) and node.func.id in _FUNCS):
                    raise ValueError(f"constraint '{text}': only {', '.join(f for f in _FUNCS if f != 'pi')} can be called")
                if isinstance(node, ast.Name) and node.id not in _FUNCS:
                    self.names.add(node.id)
        self._lhs = compile(ast.Expression(lhs), "<constraint>", "eval")
        self._rhs = compile(ast.Expression(rhs), "<constraint>", "eval")

    def violation(self, g):
        """How far ``g`` is from satisfying the constraint (0: satisfied)."""
        return abs(g) if self.equality else max(g, 0.0)

    def g(self, values):
        env = {"__builtins__": {}, **_FUNCS, **values}
        return float(eval(self._lhs, env) - eval(self._rhs, env))  # noqa: S307 -- AST checked above


class _Expression:
    """An arithmetic expression of parameter identifiers (``2*P1``, ``P1 + P2``, ``atan(P1)``, ``sqrt(P1**2 + P2**2)``):
    + - * / ** %, numbers, ``pi`` and the functions of ``_FUNCS``."""

    def __init__(self, text):
        self.text = str(text).strip()
        try:
            tree = ast.parse(self.text, mode="eval")
        except SyntaxError as exc:
            raise ValueError(f"expression '{self.text}': {exc.msg}") from None
        self.names = set()
        for node in ast.walk(tree):
            if not isinstance(node, _NODES):
                raise ValueError(f"expression '{self.text}': '{type(node).__name__}' is not allowed")
            if isinstance(node, ast.Call) and not (isinstance(node.func, ast.Name) and node.func.id in _FUNCS):
                raise ValueError(f"expression '{self.text}': only {', '.join(f for f in _FUNCS if f != 'pi')} can be called")
            if isinstance(node, ast.Name) and node.id not in _FUNCS:
                self.names.add(node.id)
        self._code = compile(tree, "<expression>", "eval")

    def __call__(self, values):
        env = {"__builtins__": {}, **_FUNCS, **values}
        return float(eval(self._code, env))  # noqa: S307 -- AST checked above


@dataclass
class ExpressionLink:
    """Model parameter ``address`` of the data sets ``datasets`` = ``expression`` of global parameters."""

    address: str
    datasets: tuple
    expression: object  # _Expression
    label: str = ""

    @property
    def text(self):
        return self.expression.text


# --------------------------------------------------------------------------- result
@dataclass
class FitSummary:
    backend: str
    algorithm: str
    success: bool
    message: str
    nfev: int
    seconds: float
    parameters: list
    errors: dict
    chi2: float
    chi2_red: float
    dof: int
    datasets: list  # [(name, n points, chi2, chi2_red)]
    curves: list  # [(q, I_fit)] per data set
    covariance: np.ndarray | None = None
    free_labels: list = field(default_factory=list)
    at_bounds: list = field(default_factory=list)
    constraints: list = field(default_factory=list)  # [(text, g, satisfied)]
    asymmetric: dict = field(default_factory=dict)  # label -> (minus, plus): MINOS / posterior 68 % intervals
    derived: list = field(default_factory=list)  # [(label, expression, value)] of the expression links

    @property
    def values(self):
        return {p.label: p.value for p in self.parameters}

    def correlation(self):
        if self.covariance is None:
            return None
        s = np.sqrt(np.diag(self.covariance))
        s[s == 0] = 1.0
        return self.covariance / np.outer(s, s)

    def text(self):
        status = "converged" if self.success else "NOT converged"
        lines = [
            f"{self.backend}{'/' + self.algorithm if self.algorithm else ''}: {status} ({self.message}); "
            f"{self.nfev} evaluations, {self.seconds:.2f} s",
            f"chi2 = {self.chi2:.5g}, chi2/dof = {self.chi2_red:.4g} (dof = {self.dof})",
        ]
        if len(self.datasets) > 1:
            for name, n, c2, c2r in self.datasets:
                lines.append(f"  data set {name}: {n} points, chi2 = {c2:.5g}, chi2/n = {c2r:.4g}")
        for p in self.parameters:
            if p.fixed:
                lines.append(f"  {p.label:24s} = {p.value:.6g}   (fixed)")
            else:
                note = "   at a bound" if p.label in self.at_bounds else ""
                lines.append(f"  {p.label:24s} = {p.value:.6g} ± {self.errors.get(p.label, math.nan):.3g}{note}")
                if p.label in self.asymmetric:
                    lo, hi = self.asymmetric[p.label]
                    lines[-1] += f"   ({lo:+.3g} / {hi:+.3g})"
        for label, expr, v in self.derived:
            lines.append(f"  {label:24s} = {v:.6g}   (= {expr})")
        for text, g, ok in self.constraints:
            lines.append(f"  constraint {text}: {'ok' if ok else 'VIOLATED'} (lhs - rhs = {g:.4g})")
        corr = self.correlation_text()
        if corr:
            lines += ["", corr]
        return "\n".join(lines)

    def correlation_text(self, threshold=0.9):
        """Correlation matrix of the free parameters (lower triangle) and a warning for strongly correlated pairs."""
        c = self.correlation()
        n = len(self.free_labels)
        if c is None or n < 2 or c.shape != (n, n) or not np.all(np.isfinite(c)):
            return ""
        w = max(len(label) for label in self.free_labels)
        lines = ["correlation matrix of the free parameters:", " " * (w + 5) + " ".join(f"{j + 1:>6d}" for j in range(n))]
        for i, label in enumerate(self.free_labels):
            cells = " ".join(f"{c[i, j]:6.2f}" if j < i else "      " for j in range(n))
            lines.append(f"{i + 1:>3d} {label:<{w}s} {cells}")
        strong = sorted(
            ((abs(c[i, j]), i, j) for i in range(n) for j in range(i) if abs(c[i, j]) >= threshold), reverse=True
        )
        if strong:
            lines += ["", f"strongly correlated (|r| >= {threshold:g}):"]
            for _r, i, j in strong:
                lines.append(f"  {self.free_labels[j]} / {self.free_labels[i]}: r = {c[i, j]:+.3f}")
            lines += [
                "  The data do not determine these parameters separately, so their errors are large. Options:",
                "  - fix one of them at a value known independently (e.g. from another measurement);",
                "  - give one or both a Gaussian prior from independent knowledge (Priors tab), not from this data;",
                "  - if only their combination matters (e.g. a product), no action is needed.",
            ]
        return "\n".join(lines)


# --------------------------------------------------------------------------- problem
class FitProblem:
    """Data sets with models, fit parameters, constraints and priors; :meth:`fit` runs an optimiser (see the module docstring)."""

    def __init__(self):
        self.datasets = []
        self.parameters = []
        self.constraints = []
        self.priors = []  # (parameter index, mean, sigma): Gaussian priors
        self.prior_extra = []  # (parameter index, kind, a, b): log-normal and half-normal priors (extra residuals)
        self.log_uniform = []  # parameter indices with a log-uniform prior (scalar term, no residual)
        self._access = {}  # (dataset, address) -> (getter, setter)
        self._base = {}  # (dataset, address) -> value of the model before the fit
        self.expr_links = []  # ExpressionLink: model parameters computed from global parameters
        self.last_error = None

    # ---------------------------------------------------------------- data
    def add_dataset(self, data, model, name=None, q_range=None, q_scale=1.0, weight=1.0):
        """Add a data set: a file name or ``(q, I, sigma)``; ``model`` is evaluated at its q values."""
        if isinstance(data, str):
            from ..core.io import load_data

            q, I, s, *_ = load_data(data, q_scale=q_scale)
            name = name or re.split(r"[\\/]", data)[-1]
        else:
            q, I, s, *_ = data
            q = np.asarray(q, float) * q_scale
        q, I = np.asarray(q, float), np.asarray(I, float)
        s = np.ones_like(I) if s is None else np.asarray(s, float)
        ok = np.isfinite(q) & np.isfinite(I) & np.isfinite(s) & (s > 0)
        if q_range is not None:
            ok &= (q >= q_range[0]) & (q <= q_range[1])
        if not ok.any():
            raise ValueError("no usable data points (finite, sigma > 0, inside the q range)")
        k = len(self.datasets)
        self.datasets.append(FitDataset(name or f"data{k}", q[ok], I[ok], s[ok], model, float(weight)))
        for p in self.parameters:  # remember what the new model has for the addresses already in use
            for a in {p.address} | {t[1] for t in p.links}:
                if a:
                    try:
                        self._remember(k, a)
                    except KeyError:
                        pass
        return k

    def _access_for(self, k, address):
        key = (k, address)
        if key not in self._access:
            self._access[key] = locate(self.datasets[k].model, address)
        return self._access[key]

    def _remember(self, k, address):
        if (k, address) not in self._base:
            self._base[(k, address)] = self._access_for(k, address)[0]()

    # ---------------------------------------------------------- parameters
    def add_parameter(self, address, datasets=None, value=None, lower=-math.inf, upper=math.inf, fixed=False, label=None):
        """A fit parameter driving ``address`` in the given data sets (default: all = common parameter)."""
        if not self.datasets:
            raise ValueError("add the data sets before the parameters")
        ds = tuple(range(len(self.datasets))) if datasets is None else tuple(int(k) for k in datasets)
        for k in ds:
            if not 0 <= k < len(self.datasets):
                raise ValueError(f"no data set {k}")
            self._check_free(k, address)
        for k in range(len(self.datasets)):
            self._remember(k, address)
        if label is None:
            label = re.sub(r"^0\.(?=(ff|sd|sq)\.)", "", address)  # '0.ff.R' -> 'ff.R'
            label = re.sub(r"^scale\.0$", "scale", label)
        if any(p.label == label for p in self.parameters) or any(_ident(p.label) == _ident(label) for p in self.parameters):
            raise ValueError(f"parameter label '{label}' is not unique")
        v = float(value) if value is not None else float(self._base[(ds[0], address)])
        if not lower <= v <= upper:
            v = min(max(v, lower), upper)
        par = FitParameter(label, address, ds, v, float(lower), float(upper), bool(fixed))
        self.parameters.append(par)
        return par

    def _check_free(self, k, address):
        for p in self.parameters:
            for kk, a, _f in p.targets():
                if a == address and kk == k:
                    raise ValueError(f"'{address}' of data set {k} is already driven by parameter '{p.label}'")
        for el in getattr(self, "expr_links", []):
            if el.address == address and k in el.datasets:
                raise ValueError(f"'{address}' of data set {k} is already set by the expression '{el.text}'")

    def link_expression(self, address, expression, datasets=None, label=None):
        """Model parameter ``address`` (in ``datasets``, default all) = ``expression`` of global parameters, e.g.
        ``"P1 + P2"``, ``"atan(P1)"``, ``"2*P1"`` (functions: those of the constraints, plus trigonometric ones)."""
        ex = expression if isinstance(expression, _Expression) else _Expression(expression)
        glob = {p.ident: p for p in self.parameters if not p.address}
        unknown = ex.names - set(glob)
        if unknown:
            raise ValueError(f"expression '{ex.text}': {', '.join(sorted(unknown))} is not a global parameter "
                             f"(known: {', '.join(sorted(glob)) or 'none'})")
        ds = tuple(range(len(self.datasets))) if datasets is None else tuple(int(k) for k in datasets)
        for k in ds:
            if not 0 <= k < len(self.datasets):
                raise ValueError(f"no data set {k}")
            self._access_for(k, address)
            self._check_free(k, address)
        for k in range(len(self.datasets)):
            try:
                self._remember(k, address)
            except KeyError:
                pass
        el = ExpressionLink(address, ds, ex, label or address)
        self.expr_links.append(el)
        return el

    def add_global(self, label, value=0.0, lower=-math.inf, upper=math.inf, fixed=False):
        """A global parameter (e.g. ``"P1"``) that drives model parameters through :meth:`link`."""
        if not label or any(p.label == label or _ident(p.label) == _ident(label) for p in self.parameters):
            raise ValueError(f"parameter label '{label}' is empty or not unique")
        v = min(max(float(value), lower), upper)
        par = FitParameter(label, "", (), v, float(lower), float(upper), bool(fixed), ())
        self.parameters.append(par)
        return par

    def link(self, label, address, datasets=None, factor=1.0):
        """Make ``address`` (in the given data sets, default all) follow the global parameter ``label``:
        model value = ``factor * label``."""
        par = self.parameter(label)
        if par.address:
            raise ValueError(f"'{label}' is not a global parameter (add it with add_global)")
        ds = tuple(range(len(self.datasets))) if datasets is None else tuple(int(k) for k in datasets)
        new = []
        for k in ds:
            if not 0 <= k < len(self.datasets):
                raise ValueError(f"no data set {k}")
            self._access_for(k, address)              # raises for an address the model does not have
            self._check_free(k, address)
            new.append((k, address, float(factor)))
        for k in range(len(self.datasets)):
            try:
                self._remember(k, address)
            except KeyError:
                pass
        par.links = tuple(par.links) + tuple(new)
        return par

    def add_common(self, address, **kw):
        """One parameter for all data sets."""
        return self.add_parameter(address, datasets=None, **kw)

    def add_local(self, address, lower=-math.inf, upper=math.inf, fixed=False, label=None):
        """One independent parameter per data set (labels ``<label>[data set name]``, e.g. ``ff.R[a]``; with a single
        data set just ``label``)."""
        base = label or re.sub(r"^0\.(?=(ff|sd|sq)\.)", "", address)
        if len(self.datasets) == 1:  # a single data set: nothing to distinguish
            return [self.add_parameter(address, [0], lower=lower, upper=upper, fixed=fixed, label=label)]
        return [
            self.add_parameter(address, [k], lower=lower, upper=upper, fixed=fixed, label=f"{base}[{d.name}]")
            for k, d in enumerate(self.datasets)
        ]

    def parameter(self, label):
        for p in self.parameters:
            if p.label == label or p.ident == label:
                return p
        raise KeyError(f"no fit parameter '{label}'")

    def identifiers(self):
        """``{identifier: label}`` for use in constraint expressions."""
        return {p.ident: p.label for p in self.parameters}

    def add_constraint(self, text):
        """An inequality (``<``, ``<=``, ``>``, ``>=``) or equality (``==``) of parameter identifiers, e.g.
        ``"ff_R + ff_t < 100"`` (fixed parameters count as constants; see :meth:`identifiers`)."""
        c = _Constraint(text)
        unknown = c.names - set(self.identifiers())
        if unknown:
            raise ValueError(f"constraint '{text}': unknown {', '.join(sorted(unknown))}; known: {', '.join(self.identifiers())}")
        self.constraints.append(c)
        return c

    PRIOR_USAGE = {
        "normal": "name  mean  sigma", "lognormal": "name  lognormal  median  factor", "halfnormal": "name  halfnormal  sigma",
        "loguniform": "name  loguniform",
    }  # fmt: skip

    def add_prior(self, label, *args, kind="normal"):
        """Add a prior to the parameter ``label``; the term is added to chi^2.

        * ``add_prior(label, mean, sigma)`` -- Gaussian: ``((x - mean) / sigma)^2``;
        * ``add_prior(label, median, factor, kind="lognormal")`` -- ``(ln(x / median) / ln(factor))^2``, factor > 1 (positive
          parameters known to a factor, e.g. 2 = within a factor 2 at one sigma);
        * ``add_prior(label, sigma, kind="halfnormal")`` -- ``(x / sigma)^2``: small values preferred, for x >= 0;
        * ``add_prior(label, kind="loguniform")`` -- flat in ln x (every decade equally likely), adds ``2 ln x``; the lower bound
          of the parameter must be > 0. This term is not a residual: the backends ``scipy`` and ``dfols`` refuse it.
        """
        kind = kind.lower()
        if kind not in self.PRIOR_USAGE:
            raise ValueError(f"unknown prior '{kind}' (use {', '.join(self.PRIOR_USAGE)})")
        need = {"normal": 2, "lognormal": 2, "halfnormal": 1, "loguniform": 0}[kind]
        if len(args) != need:
            raise ValueError(f"write  {self.PRIOR_USAGE[kind]}")
        i = self.parameters.index(self.parameter(label))
        a = [float(v) for v in args]
        if kind == "normal":
            if not a[1] > 0:
                raise ValueError("sigma of a prior must be > 0")
            self.priors.append((i, a[0], a[1]))
        elif kind == "lognormal":
            if not (a[0] > 0 and a[1] > 1):
                raise ValueError("a log-normal prior needs median > 0 and factor > 1")
            self.prior_extra.append((i, kind, a[0], a[1]))
        elif kind == "halfnormal":
            if not a[0] > 0:
                raise ValueError("sigma of a prior must be > 0")
            self.prior_extra.append((i, kind, a[0], 0.0))
        else:
            if not self.parameters[i].lower > 0:
                raise ValueError("a log-uniform prior needs a lower bound > 0 (and a finite upper bound)")
            self.log_uniform.append(i)

    def add_prior_text(self, line):
        """Add a prior from a text line: ``name mean sigma``, ``name lognormal median factor``, ``name halfnormal sigma``
        or ``name loguniform``."""
        parts = [t for t in re.split(r"[\s,;=±]+|\+-", line.strip()) if t]
        kind = "normal"
        if len(parts) > 1 and parts[1].lower() in self.PRIOR_USAGE:
            kind = parts.pop(1).lower()
        if not parts:
            raise ValueError("empty prior")
        try:
            nums = [float(t) for t in parts[1:]]
            self.add_prior(parts[0], *nums, kind=kind)
        except KeyError as exc:
            raise ValueError(f"prior '{line.strip()}': unknown parameter {exc.args[0]}") from None
        except ValueError as exc:
            msg = str(exc)
            if msg.startswith("could not convert"):
                msg = f"write  {self.PRIOR_USAGE[kind]}"
            raise ValueError(f"prior '{line.strip()}': {msg}") from None

    def n_prior_rows(self):
        """Number of residual rows of the priors (the log-uniform prior has none)."""
        return len(self.priors) + len(self.prior_extra)

    def _prior_rows(self, vals):
        rows = [np.array([(vals[i] - mean) / sigma]) for i, mean, sigma in self.priors]
        for i, kind, a, b in self.prior_extra:
            x = vals[i]
            if kind == "lognormal":
                r = (math.log(x / a) / math.log(b)) if x > 0 else BIG
            else:
                r = x / a
            rows.append(np.array([r]))
        return rows

    def prior_penalty(self, vals):
        """Scalar term of the log-uniform priors, ``2 ln x`` (0 without them)."""
        t = 0.0
        for i in self.log_uniform:
            x = vals[i]
            t += 2.0 * math.log(x) if x > 0 else BIG
        return t

    # ------------------------------------------------------------ evaluation
    def apply(self, k, values=None):
        """Set the model of data set ``k`` to the values of the parameters (or to ``values``, a full vector)."""
        vals = np.array([p.value for p in self.parameters]) if values is None else values
        covered = {a for p in self.parameters for kk, a, _f in p.targets() if kk == k}
        covered |= {el.address for el in self.expr_links if k in el.datasets}
        for (kk, address), v in self._base.items():
            if kk == k and address not in covered:
                self._access_for(k, address)[1](v)
        for i, p in enumerate(self.parameters):
            for kk, a, f in p.targets():
                if kk == k:
                    self._access_for(k, a)[1](f * vals[i])
        if self.expr_links:
            env = {p.ident: vals[i] for i, p in enumerate(self.parameters)}
            for el in self.expr_links:
                if k in el.datasets:
                    self._access_for(k, el.address)[1](el.expression(env))

    def derived_values(self, values=None):
        """``[(label, expression, value)]`` of the expression links (per data set where they differ)."""
        vals = np.array([p.value for p in self.parameters]) if values is None else values
        env = {p.ident: vals[i] for i, p in enumerate(self.parameters)}
        out = []
        for el in self.expr_links:
            try:
                v = el.expression(env)
            except (ValueError, ZeroDivisionError, OverflowError):
                v = math.nan
            ds = "" if len(el.datasets) == len(self.datasets) else "[" + ", ".join(self.datasets[k].name for k in el.datasets) + "]"
            out.append((el.label + ds, el.text, v))
        return out

    def _residuals(self, vals):
        out = []
        for k, d in enumerate(self.datasets):
            try:
                self.apply(k, vals)  # an expression link may fail (e.g. sqrt of a negative trial value)
                I = np.asarray(d.model.intensity(d.q), float)
                r = (I - d.I) / d.sigma * math.sqrt(d.weight)
            except FitCancelled:
                raise
            except Exception as exc:  # noqa: BLE001 -- e.g. a radius < 0 during the search
                self.last_error = f"{type(exc).__name__}: {exc}"
                r = np.full(d.q.size, BIG)
            out.append(np.where(np.isfinite(r), r, BIG))
        out.extend(self._prior_rows(vals))
        return np.concatenate(out)

    def _constraint_values(self, vals):
        env = {p.ident: vals[i] for i, p in enumerate(self.parameters)}
        return [c.g(env) for c in self.constraints]

    # ------------------------------------------------------------------ fit
    def linear_parameters(self, candidates=None):
        """Indices of the free parameters that variable projection can eliminate (see the module docstring)."""
        cand = range(len(self.parameters)) if candidates is None else candidates
        used = set().union(*[c.names for c in self.constraints]) if self.constraints else set()
        prior = {i for i, _m, _s in self.priors} | {i for i, _k, _a, _b in self.prior_extra} | set(self.log_uniform)
        kinds = {}
        for i in cand:
            p = self.parameters[i]
            if p.fixed or p.ident in used or i in prior:
                continue
            kind = self._linear_kind(p)
            if kind:
                kinds[i] = kind
        for i, (kind, ci) in list(kinds.items()):  # scale and N of one contribution are redundant: N is eliminated
            if kind == "scale" and any(
                k2 == "N" and c2 == ci and set(self.parameters[j].datasets) & set(self.parameters[i].datasets)
                for j, (k2, c2) in kinds.items()
            ):
                del kinds[i]
        self._lin_kinds = kinds
        return list(kinds)

    def _linear_kind(self, p):
        from .model import ScatteringContribution

        if not p.address or p.links:            # global parameters stay with the optimiser
            return None
        parts = p.address.split(".")
        try:
            if p.address == "background":
                return ("bg", None)
            if p.address == "bg.A":
                return ("bgA", None)
            if parts[0] == "scale":
                return ("scale", int(parts[1]))
            if len(parts) >= 3 and parts[1] == "sd":
                i, key = int(parts[0]), ".".join(parts[2:])
                for k in p.datasets:
                    c = self.datasets[k].model.contributions[i]
                    if c.sd is None or c.amplitude_index() is None or ScatteringContribution._index(c.sd, key) != c.amplitude_index():
                        return None
                return ("N", i)
        except (ValueError, IndexError, KeyError):
            return None
        return None

    def _residuals_vp(self, vals, lin):
        """Residuals with the linear parameters ``lin`` (indices) at their optimum, which is stored in ``vals``.

        Per data set the model is I = offset + sum_j c_j phi_j(q): every contribution is evaluated once (with N = 1
        where N is linear), then all data sets are solved together by (bounded) linear least squares.
        """
        from scipy.optimize import lsq_linear

        nl = len(lin)
        col = {pi: j for j, pi in enumerate(lin)}
        nrows = sum(d.q.size for d in self.datasets)
        total = nrows + self.n_prior_rows()
        Ms, bs = [], []
        for k, d in enumerate(self.datasets):
            try:
                self.apply(k, vals)
            except FitCancelled:
                raise
            except Exception as exc:  # noqa: BLE001 -- an expression link that cannot be evaluated here
                self.last_error = f"{type(exc).__name__}: {exc}"
                return np.full(total, BIG)
            wt = math.sqrt(d.weight) / d.sigma
            sc_cols, n_cols, bg_col, pw_col = {}, {}, None, None
            for pi, j in col.items():
                if k in self.parameters[pi].datasets:
                    kind, ci = self._lin_kinds[pi]
                    if kind == "scale":
                        sc_cols[ci] = j
                    elif kind == "N":
                        n_cols[ci] = j
                    elif kind == "bgA":
                        pw_col = j
                    else:
                        bg_col = j
            try:
                parts, bg = d.model.components(d.q, set(n_cols))
                scales = d.model.scales
                off = np.zeros(d.q.size)
                M = np.zeros((d.q.size, nl))
                for i, part in enumerate(parts):
                    if i in sc_cols:
                        M[:, sc_cols[i]] = part
                    elif i in n_cols:
                        M[:, n_cols[i]] = scales[i] * part
                    else:
                        off = off + scales[i] * part
                pw = d.model.power_term(d.q)  # bg is the constant part plus this power-law term
                bg = bg - pw
                if pw_col is None:
                    off = off + pw
                else:
                    M[:, pw_col] = np.asarray(d.q, float) ** (-d.model.bg_alpha)
                if bg_col is None:
                    off = off + bg
                else:
                    M[:, bg_col] = 1.0
                M = M * wt[:, None]
                b = (d.I - off) * wt
                if not (np.all(np.isfinite(M)) and np.all(np.isfinite(b))):
                    raise FloatingPointError("non-finite model values")
            except FitCancelled:
                raise
            except Exception as exc:  # noqa: BLE001 -- e.g. a radius < 0 during the search
                self.last_error = f"{type(exc).__name__}: {exc}"
                return np.full(total, BIG)
            Ms.append(M)
            bs.append(b)
        M, b = np.vstack(Ms), np.concatenate(bs)
        lo = np.array([self.parameters[i].lower for i in lin])
        hi = np.array([self.parameters[i].upper for i in lin])
        c = np.clip(vals[lin], lo, hi)
        norm = np.linalg.norm(M, axis=0)
        ok = norm > 0
        if ok.any():
            Mn = M[:, ok] / norm[ok]
            lo_s, hi_s = lo[ok] * norm[ok], hi[ok] * norm[ok]
            if np.any(np.isfinite(lo_s)) or np.any(np.isfinite(hi_s)):
                sol = lsq_linear(Mn, b, bounds=(lo_s, hi_s), method="bvls").x
            else:
                sol = np.linalg.lstsq(Mn, b, rcond=None)[0]
            c[ok] = sol / norm[ok]
        vals[lin] = c
        r = [M @ c - b]
        r.extend(self._prior_rows(vals))
        r = np.concatenate(r)
        return np.where(np.isfinite(r), r, BIG)

    def fit(self, backend="scipy", algorithm=None, callback=None, cancel=None, penalty=1e3, scale_errors=True,
            vp=False, cache=False, snap=0.05, coarse=False, coarse_factor=4, grid=False, **options):  # fmt: skip
        """Run the fit; returns a :class:`FitSummary`. Speed-ups (all optional): ``vp`` variable projection,
        ``cache`` form-factor cache, ``coarse`` coarse to fine, ``grid`` fixed logarithmic size grid (implies ``cache``).
        See :meth:`_fit_coarse` and :meth:`_fit_once` for the details and the backend ``options``.
        """
        if not grid:
            return self._fit_coarse(backend, algorithm, callback, cancel, penalty, scale_errors, vp, cache, snap, coarse,
                                    coarse_factor, **options)  # fmt: skip
        cs = [c for c in {id(c): c for d in self.datasets for c in d.model.contributions}.values() if hasattr(c, "fixed_grid")]
        state = [(c, c.fixed_grid, c._grid) for c in cs]
        for c in cs:
            c.fixed_grid, c._grid = True, None
        try:
            return self._fit_coarse(backend, algorithm, callback, cancel, penalty, scale_errors, vp, True, snap, coarse,
                                    coarse_factor, **options)  # fmt: skip
        finally:
            for c, fixed, g in state:
                c.fixed_grid, c._grid = fixed, g

    def _fit_coarse(self, backend="scipy", algorithm=None, callback=None, cancel=None, penalty=1e3, scale_errors=True,
                    vp=False, cache=False, snap=0.05, coarse=False, coarse_factor=4, **options):  # fmt: skip
        """Run the fit, optionally coarse to fine; see :meth:`_fit_once` for the arguments.

        ``coarse=True``: first fit with a cheap size-distribution integral (``n_x`` divided by ``coarse_factor``,
        ``rtol`` at least 1e-3), then repeat from that result with the real settings. A global optimiser is
        replaced by the least-squares ``trf`` in the second stage (the first stage did the global search). The
        evaluations and times of both stages are added in the summary.
        """
        if not coarse:
            return self._fit_once(backend, algorithm, callback, cancel, penalty, scale_errors, vp, cache, snap, **options)
        targets = {id(c): c for d in self.datasets for c in d.model.contributions
                   if getattr(c, "sd", None) is not None and hasattr(c, "n_x")}.values()  # fmt: skip
        if not targets:  # no size distribution: nothing to coarsen
            return self._fit_once(backend, algorithm, callback, cancel, penalty, scale_errors, vp, cache, snap, **options)
        saved = [(c, c.n_x, c.rtol) for c in targets]
        n_coarse = ", ".join(str(max(8, int(c.n_x) // max(int(coarse_factor), 1))) for c in targets)
        try:
            for c in targets:
                c.n_x = max(8, int(c.n_x) // max(int(coarse_factor), 1))
                c.rtol = max(c.rtol, 1e-3)
            first = self._fit_once(backend, algorithm, callback, cancel, penalty, scale_errors, vp, cache, snap, **options)
        finally:
            for c, n_x, rtol in saved:
                c.n_x, c.rtol = n_x, rtol
        if first.message == "cancelled":
            return first
        second_backend, second_alg = (backend, algorithm)
        if not self.log_uniform and (backend in ("scipy-global", "cma") or (backend == "bumps" and algorithm in ("de", "dream"))):
            second_backend, second_alg, options = "scipy", "trf", {k: v for k, v in options.items() if k == "maxeval"}
        elif backend == "scipy":
            options = {k: v for k, v in options.items() if k != "start"}
        second = self._fit_once(second_backend, second_alg, callback, cancel, penalty, scale_errors, vp, cache, snap, **options)
        second.message = (
            f"coarse to fine: stage 1 {first.nfev} evaluations (chi2/dof {first.chi2_red:.4g}, "
            f"n_x {n_coarse}), "
            f"stage 2 {second.nfev} evaluations; {second.message}"
        )
        second.nfev += first.nfev
        second.seconds += first.seconds
        return second

    def _fit_once(self, backend="scipy", algorithm=None, callback=None, cancel=None, penalty=1e3, scale_errors=True,
                  vp=False, cache=False, snap=0.05, **options):  # fmt: skip
        """Run the fit; returns a :class:`FitSummary` (the parameters keep the fitted values).

        ``callback(nfev, chi2_red)`` is called for every evaluation; ``cancel`` is a ``threading.Event`` that stops
        the fit (the best point so far is returned). ``penalty`` weighs the constraints for ``"scipy"``.
        ``vp`` switches variable projection on, ``cache`` the form-factor cache (``snap``: see the module docstring).
        ``options`` go to the backend (``max_nfev``, ``xtol``, ``ftol`` ... for scipy; ``maxeval``, ``maxtime``,
        ``xtol_rel``, ``ftol_rel`` for NLopt; ``rhobeg``, ``rhoend`` for DFO-LS).
        """
        contribs = {id(c): c for d in self.datasets for c in d.model.contributions}.values()
        contribs = [c for c in contribs if hasattr(c, "enable_cache")]  # (simple stand-in models have no cache)
        state = [(c, c.cache, c.snap) for c in contribs]
        if cache:
            for c in contribs:
                c.enable_cache(snap)
        try:
            summary = self._fit(backend, algorithm, callback, cancel, penalty, scale_errors, vp, **options)
            if cache and contribs:
                summary.message += "; cache: " + ", ".join(c.cache.text() for c in contribs if c.cache is not None)
        finally:
            for c, cc, sn in state:
                c.cache, c.snap = cc, sn
        return summary

    def _fit(self, backend, algorithm, callback, cancel, penalty, scale_errors, vp, **options):
        if backend not in BACKENDS:
            raise ValueError(f"unknown backend '{backend}' (available: {', '.join(BACKENDS)})")
        if not self.datasets:
            raise ValueError("no data set")
        if self.log_uniform and backend in ("scipy", "dfols"):
            raise ValueError(
                f"the log-uniform prior is a scalar term that the least-squares backend '{backend}' cannot use: choose another "
                "optimiser (scipy global, NLopt, iminuit, CMA-ES, bumps) or a log-normal prior"
            )
        ctx = _Context(self, cancel, callback, penalty, vp)
        if not ctx.all_free:
            raise ValueError("no free parameter")
        t0 = time.perf_counter()
        try:
            if ctx.free:
                u, message, success = BACKENDS[backend](ctx, algorithm, options)
            else:  # every free parameter is linear: one linear least-squares solution
                u = ctx.u0[:0]
                ctx.residual(u)
                message, success = "all free parameters are linear: solved by linear least squares", True
            if ctx.constraints_violated(u) and ctx.best_u is not None:  # an infeasible end point: best feasible instead
                u, message = ctx.best_u, message + "; end point violated a constraint, best feasible point used"
        except FitCancelled:
            u, message, success = ctx.best_u if ctx.best_u is not None else ctx.u0, "cancelled", False
        seconds = time.perf_counter() - t0
        if ctx.lin:
            message = f"variable projection ({len(ctx.lin)} linear parameter(s) solved exactly); {message}"
        elif vp:
            message = f"variable projection: no linear parameter found; {message}"
        vals = ctx.final_vals(u)
        starts = [p.value for p in self.parameters]  # the values the fit started from (scale of the at-bound test)
        for p, v in zip(self.parameters, vals):
            p.value = float(v)
        self.apply(0)
        r = self._residuals(vals)
        chi2 = float(np.sum(r**2))
        n = sum(d.q.size for d in self.datasets) + self.n_prior_rows()
        dof = max(n - len(ctx.all_free), 1)
        cov, err = ctx.covariance(vals, chi2 / dof if scale_errors else 1.0)
        free_labels = [self.parameters[i].label for i in ctx.all_free]
        errors = dict(zip(free_labels, err))
        at_bounds = []
        for i, x in zip(ctx.all_free, vals[ctx.all_free]):
            lo, hi = self.parameters[i].lower, self.parameters[i].upper
            # Trust-region methods stop slightly inside a bound. 'Slightly' is relative to the size of the parameter
            # (the larger of the result and the start value, so that a parameter driven towards 0 is still caught):
            # 1e-4 of it, or 1e-6 of a finite range. There is no absolute tolerance: parameters of 1e-12 (the
            # amplitude N of a size distribution) are not 'at the bound 0' only because they are small.
            scale = max(abs(x), abs(starts[i]))
            tol = max(1e-4 * scale, 1e-6 * (hi - lo if math.isfinite(hi - lo) else 0.0))
            if x <= lo + tol or x >= hi - tol:
                at_bounds.append(self.parameters[i].label)
        per_set, pos = [], 0
        for d in self.datasets:
            rk = r[pos : pos + d.q.size]
            pos += d.q.size
            c2 = float(np.sum(rk**2))
            per_set.append((d.name, d.q.size, c2, c2 / d.q.size))
        curves = []
        for k, d in enumerate(self.datasets):
            self.apply(k, vals)
            curves.append((d.q, np.asarray(d.model.intensity(d.q), float)))
        self.apply(0)
        asym = {}
        s_err = math.sqrt(chi2 / dof) if scale_errors else 1.0
        for j, (lo, hi) in ctx.asym.items():
            asym[self.parameters[ctx.free[j]].label] = (lo * ctx.scale[j] * s_err, hi * ctx.scale[j] * s_err)
        g = self._constraint_values(vals)
        constraints = [(c.text, gi, c.violation(gi) <= CONSTRAINT_TOL) for c, gi in zip(self.constraints, g)]
        return FitSummary(
            backend, algorithm or "", success, message, ctx.nfev, seconds, [FitParameter(**vars(p)) for p in self.parameters],
            errors, chi2, chi2 / dof, dof, per_set, curves, cov, free_labels, at_bounds, constraints, asym, derived=self.derived_values(vals),
        )  # fmt: skip


class _Context:
    """What a backend needs: scaled free variables, residuals, constraints, bookkeeping."""

    def __init__(self, problem, cancel, callback, penalty, vp=False):
        self.problem, self.cancel, self.callback, self.penalty = problem, cancel, callback, penalty
        ps = problem.parameters
        self.all_free = [i for i, p in enumerate(ps) if not p.fixed]  # everything the fit determines
        self.lin = problem.linear_parameters(self.all_free) if vp else []  # eliminated by variable projection
        self.free = [i for i in self.all_free if i not in self.lin]  # the variables of the optimiser
        self.vals0 = np.array([p.value for p in ps], float)
        x0, self.scale, lo, hi = self._prep(self.free)
        self.scale_all = self._prep(self.all_free)[1]
        self.u0, self.lb, self.ub = x0 / self.scale, lo / self.scale, hi / self.scale
        self.constraints = problem.constraints
        self.nfev = 0
        self.cov_u = None  # covariance in the scaled variables, if the backend has one (DREAM)
        self.asym = {}  # free index -> (minus, plus) interval in the scaled variables
        self.best_chi2, self.best_u = math.inf, None
        self.ndata = sum(d.q.size for d in problem.datasets) + problem.n_prior_rows()
        self.dof = max(self.ndata - len(self.all_free), 1)

    def rescale_to_bounds(self):
        """Global search: scale every variable with finite bounds by its range (a box of width 1).

        The default scale is the start value, which makes the box of a parameter that starts near 0 (or far from the
        middle of its range) huge or tiny; a global optimiser samples the box, so it needs the range as the unit.
        Variables without finite bounds keep their scale."""
        x0, lo, hi = self.u0 * self.scale, self.lb * self.scale, self.ub * self.scale
        width = hi - lo
        ok = np.isfinite(width) & (width > 0)
        scale = np.where(ok, width, self.scale)
        self.scale, self.u0, self.lb, self.ub = scale, x0 / scale, lo / scale, hi / scale

    def _prep(self, idx):
        """Start values (clipped to the bounds), scales and bounds of the parameters ``idx``."""
        ps = self.problem.parameters
        lo = np.array([ps[i].lower for i in idx], float)
        hi = np.array([ps[i].upper for i in idx], float)
        x0 = np.clip(self.vals0[idx], lo, hi)
        width = np.where(np.isfinite(hi - lo), hi - lo, 1.0)
        scale = np.where(np.abs(x0) > 1e-12, np.abs(x0), np.where(width > 0, width, 1.0))
        return x0, scale, lo, hi

    def final_vals(self, u):
        """All parameter values for the optimiser variables ``u`` (the linear ones solved)."""
        v = self.vals(u)
        if self.lin:
            self.problem._residuals_vp(v, self.lin)
        return v

    def vals(self, u):
        v = self.vals0.copy()
        v[self.free] = np.asarray(u, float) * self.scale
        return v

    def g(self, u):
        return self.problem._constraint_values(self.vals(u))

    def constraints_violated(self, u):
        return any(c.violation(g) > CONSTRAINT_TOL for c, g in zip(self.constraints, self.g(u))) if self.constraints else False

    def residual(self, u):
        if self.cancel is not None and self.cancel.is_set():
            raise FitCancelled
        if self.lin:
            r = self.problem._residuals_vp(self.vals(u), self.lin)
        else:
            r = self.problem._residuals(self.vals(u))
        chi2 = float(np.sum(r**2))
        if self.problem.log_uniform:  # the point is judged by chi2 plus the scalar prior term
            chi2 += self.problem.prior_penalty(self.vals(u))
        self.nfev += 1
        if chi2 < self.best_chi2 and not self.constraints_violated(u):
            self.best_chi2, self.best_u = chi2, np.array(u, float)
        if self.callback is not None:
            self.callback(self.nfev, min(self.best_chi2, chi2) / self.dof)
        return r

    def chi2(self, u):
        c = float(np.sum(self.residual(u) ** 2))
        return c + self.problem.prior_penalty(self.vals(u)) if self.problem.log_uniform else c

    def covariance(self, vals, factor):
        """Covariance and standard errors of all free parameters from the numerical Jacobian of the full problem."""
        ps, free = self.problem.parameters, self.all_free
        if self.cov_u is not None and not self.lin:
            cov = np.asarray(self.cov_u, float) * np.outer(self.scale, self.scale) * factor
            return cov, np.sqrt(np.clip(np.diag(cov), 0, None))
        x = np.asarray(vals, float)[free]
        n = len(free)
        try:
            r0 = self.problem._residuals(vals)
            J = np.empty((r0.size, n))
            for j, i in enumerate(free):
                h = 1e-6 * max(abs(x[j]), self.scale_all[j])
                up, dn = x[j] + h <= ps[i].upper, x[j] - h >= ps[i].lower
                v1, v2 = vals.copy(), vals.copy()
                v1[i], v2[i] = (x[j] + h, x[j] - h) if up and dn else ((x[j] + h, x[j]) if up else (x[j], x[j] - h))
                J[:, j] = (self.problem._residuals(v1) - self.problem._residuals(v2)) / (v1[i] - v2[i])
            col = np.linalg.norm(J, axis=0)
            col[col == 0] = 1.0
            Js = J / col
            cov = np.linalg.pinv(Js.T @ Js) / np.outer(col, col) * factor
            return cov, np.sqrt(np.clip(np.diag(cov), 0, None))
        except (np.linalg.LinAlgError, FitCancelled):
            return None, np.full(n, np.nan)


# --------------------------------------------------------------------------- backends
def _run_scipy(ctx, algorithm, options):
    from scipy.optimize import least_squares

    method = algorithm or "trf"
    if method not in SCIPY_ALGORITHMS:
        raise ValueError(f"unknown scipy method '{method}' (use {', '.join(SCIPY_ALGORITHMS)})")
    bounded = np.any(np.isfinite(ctx.lb)) or np.any(np.isfinite(ctx.ub))
    if method == "lm" and bounded:
        raise ValueError("method 'lm' does not support bounds; use 'trf' or 'dogbox' (or remove the bounds)")

    def fun(u):
        r = ctx.residual(u)
        if ctx.constraints:  # penalty rows for violated constraints
            r = np.concatenate([r, ctx.penalty * np.array([c.violation(g) for c, g in zip(ctx.constraints, ctx.g(u))])])
        return r

    kw = {"bounds": (ctx.lb, ctx.ub)} if method != "lm" else {}
    options = dict(options)
    if "maxeval" in options:
        options["max_nfev"] = options.pop("maxeval")
    for key in ("polish", "maxtime"):  # options of the other backends
        options.pop(key, None)
    start = np.clip(options.pop("start", ctx.u0), ctx.lb, ctx.ub)
    options.setdefault("diff_step", 1e-4)  # relative step of the Jacobian: well above the noise of adaptive integrals
    res = least_squares(fun, start, method=method, x_scale=1.0, **kw, **options)
    return res.x, res.message, bool(res.success)


def _run_nlopt(ctx, algorithm, options):
    try:
        import nlopt
    except ImportError:
        raise ImportError("the 'nlopt' backend needs NLopt (pip install nlopt)") from None
    name = algorithm or "LN_BOBYQA"
    if not hasattr(nlopt, name):
        raise ValueError(f"unknown NLopt algorithm '{name}'")
    n = len(ctx.u0)
    if name.startswith("GN_") and not (np.all(np.isfinite(ctx.lb)) and np.all(np.isfinite(ctx.ub))):
        raise ValueError(f"{name} is a global algorithm: set a lower and an upper bound for every free parameter")

    def gradient(f, u, f0, grad):
        # central differences; the step stays well above the noise of adaptive integrals (~1e-6 relative)
        for j in range(n):
            h = 1e-3 * max(1.0, abs(u[j]))
            up, dn = min(u[j] + h, ctx.ub[j]), max(u[j] - h, ctx.lb[j])
            if not up > dn:
                grad[j] = 0.0
                continue
            v, w = u.copy(), u.copy()
            v[j], w[j] = up, dn
            grad[j] = (f(v) - f(w)) / (up - dn)

    norm = max(ctx.chi2(np.clip(ctx.u0, ctx.lb, ctx.ub)), 1e-12)  # objective of order 1 (quasi-Newton methods need it)

    def objective(u, grad):
        u = np.array(u)
        f0 = ctx.chi2(u)
        if grad.size:
            gradient(ctx.chi2, u, f0, grad)
            grad[:] = grad / norm
        return f0 / norm

    def make_constraint(i):
        def fn(u, grad):
            u = np.array(u)
            g0 = ctx.g(u)[i]
            if grad.size:
                gradient(lambda v: ctx.g(v)[i], u, g0, grad)
            return g0

        return fn

    supported = _NLOPT_EQUALITY if any(c.equality for c in ctx.constraints) else _NLOPT_CONSTRAINTS
    wrap = bool(ctx.constraints) and name not in supported
    opt = nlopt.opt(nlopt.AUGLAG if wrap else getattr(nlopt, name), n)
    opt.set_lower_bounds(list(ctx.lb))
    opt.set_upper_bounds(list(ctx.ub))
    width = ctx.ub - ctx.lb  # initial step: 10 % of the value (the default of NLopt is far too large here)
    step = np.minimum(0.1 * np.maximum(np.abs(ctx.u0), 0.1), np.where(np.isfinite(width), 0.25 * width, np.inf))
    opt.set_initial_step(list(np.where(step > 0, step, 0.1)))
    opt.set_min_objective(objective)
    for i, c in enumerate(ctx.constraints):
        if c.equality:
            opt.add_equality_constraint(make_constraint(i), 1e-6)
        else:
            opt.add_inequality_constraint(make_constraint(i), 1e-8)
    if wrap:
        local = nlopt.opt(getattr(nlopt, name), n)
        local.set_xtol_rel(options.get("xtol_rel", 1e-6))
        local.set_ftol_rel(options.get("ftol_rel", 1e-9))
        opt.set_local_optimizer(local)
    opt.set_xtol_rel(options.get("xtol_rel", 1e-6))
    opt.set_ftol_rel(options.get("ftol_rel", 1e-9))
    opt.set_maxeval(int(options.get("maxeval", 2000 * n)))
    if options.get("maxtime"):
        opt.set_maxtime(float(options["maxtime"]))
    best = lambda: ctx.best_u if ctx.best_u is not None else ctx.u0  # noqa: E731
    try:
        u = opt.optimize(np.clip(ctx.u0, ctx.lb, ctx.ub))
        code = opt.last_optimize_result()
        text = {1: "success", 2: "stopval reached", 3: "ftol reached", 4: "xtol reached", 5: "maximum number of evaluations reached",
                6: "maximum time reached"}.get(code, f"NLopt result code {code}")  # fmt: skip
        ok = code in (1, 2, 3, 4)
    except nlopt.RoundoffLimited:
        u, text, ok = best(), "roundoff limited (usually fine)", True
    except FitCancelled:
        raise
    except Exception as exc:  # noqa: BLE001 -- generic NLopt failure (nlopt.runtime_error), e.g. a gradient method on a noisy function
        u, text, ok = best(), f"NLopt stopped ({exc or 'failure'}); best point so far", False
    if options.get("polish") and not ctx.problem.log_uniform:  # finish with a least-squares fit: reaches the minimum, gives the errors
        u2, text2, ok2 = _run_scipy(ctx, "trf", {"start": u})
        u, text, ok = best(), f"{text}; refined with least_squares: {text2}", ok2
    return u, text, ok


# ---------------------------------------------------------------- more backends
def _penalised(ctx):
    """Objective chi^2 / dof plus the (squared) penalty of violated constraints, for optimisers without constraints."""

    def f(u):
        c = ctx.chi2(u)
        if ctx.constraints:
            c += sum((ctx.penalty * cn.violation(g)) ** 2 for cn, g in zip(ctx.constraints, ctx.g(u)))
        return c / ctx.dof

    return f


def _finish(ctx, u, text, ok, options):
    """The best point seen; with ``polish`` a least-squares fit from there (exact minimum and errors)."""
    best = lambda x: ctx.best_u if ctx.best_u is not None else x  # noqa: E731
    if options.get("polish") and not ctx.problem.log_uniform:
        _u, text2, ok2 = _run_scipy(ctx, "trf", {"start": best(u)})
        return best(u), f"{text}; refined with least_squares: {text2}", ok2
    return best(u), text, ok


_GLOBAL_ALGORITHMS = {
    "differential_evolution": "global, population based (finite bounds needed)",
    "multistart": "global, many least-squares fits from spread-out start points, reports the minima found (finite bounds needed)",
    "dual_annealing": "global, simulated annealing (finite bounds needed)",
    "shgo": "global, simplicial homology (finite bounds needed)",
    "basinhopping": "global, random hops + local search (bounds optional)",
}


def _run_scipy_global(ctx, algorithm, options):
    from scipy import optimize as so

    name = algorithm or "differential_evolution"
    if name not in _GLOBAL_ALGORITHMS:
        raise ValueError(f"unknown global method '{name}' (use {', '.join(_GLOBAL_ALGORITHMS)})")
    finite = bool(np.all(np.isfinite(ctx.lb)) and np.all(np.isfinite(ctx.ub)))
    if name != "basinhopping" and not finite:
        raise ValueError(f"{name}: set a lower and an upper bound for every fitted parameter")
    ctx.rescale_to_bounds()
    if name == "multistart":
        return _run_multistart(ctx, options)
    f = _penalised(ctx)
    n = len(ctx.u0)
    bounds = list(zip(ctx.lb, ctx.ub))
    x0 = np.clip(ctx.u0, ctx.lb, ctx.ub)
    maxeval = int(options.get("maxeval") or 0)
    seed = options.get("seed", 1)
    if name == "differential_evolution":
        popsize = int(options.get("popsize", 12))
        maxiter = max(1, maxeval // (popsize * n)) if maxeval else 300
        res = so.differential_evolution(f, bounds, maxiter=maxiter, popsize=popsize, x0=x0, seed=seed, polish=False, tol=1e-8)
    elif name == "dual_annealing":
        res = so.dual_annealing(f, bounds, maxfun=maxeval or 4000 * n, x0=x0, seed=seed, no_local_search=True)
    elif name == "shgo":
        res = so.shgo(f, bounds, n=int(options.get("n", 64)), sampling_method="sobol", options={"maxfev": maxeval or None})
    else:
        import warnings

        warnings.simplefilter("ignore")
        mk = {"method": "Powell", "bounds": bounds} if finite else {"method": "Nelder-Mead"}
        res = so.basinhopping(f, x0, niter=int(options.get("niter", 30)), stepsize=0.2, seed=seed, minimizer_kwargs=mk)
    return _finish(ctx, res.x, str(res.message).strip("[]' ") or name, bool(getattr(res, "success", True)), options)


def _run_multistart(ctx, options):
    """Least-squares fits from ``starts`` (default 20) points spread over the bounds; the best one is returned.

    The start points are a Latin hypercube (uniform in a parameter, log-uniform when its bounds are positive and span
    more than two decades); the first start is the current parameter set. The message lists the distinct minima found
    (chi2/dof, number of starts that ended in each): a minimum found by one start only is a warning sign."""
    from scipy.stats import qmc

    if ctx.problem.log_uniform:
        raise ValueError("multistart uses least-squares fits, which cannot use a log-uniform prior")
    n = len(ctx.u0)
    k = max(int(options.get("starts", 20)), 1)
    seed = options.get("seed", 1)
    lo, hi = ctx.lb * ctx.scale, ctx.ub * ctx.scale
    U = qmc.LatinHypercube(d=n, seed=seed).random(max(k - 1, 0)) if k > 1 else np.empty((0, n))
    X = lo + U * (hi - lo)
    wide = (lo > 0) & (hi > 100 * lo)
    X[:, wide] = np.exp(np.log(lo[wide]) + U[:, wide] * (np.log(hi[wide]) - np.log(lo[wide])))
    starts = [ctx.u0] + [x / ctx.scale for x in X]
    maxeval = int(options.get("maxeval") or 0)
    per = {"maxeval": max(maxeval // k, 50)} if maxeval else {}
    f = _penalised(ctx)
    found = []
    for s in starts:
        try:
            u, _msg, _ok = _run_scipy(ctx, "trf", {"start": s, **per})
        except FitCancelled:
            raise
        except Exception:  # noqa: BLE001 -- a start in a region where the model cannot be evaluated
            continue
        found.append((float(f(u)), np.asarray(u, float)))
    if not found:
        raise ValueError("multistart: no start point gave a usable fit")
    found.sort(key=lambda t: t[0])
    groups = []  # [chi2/dof of the best of the group, count]
    for c, _u in found:
        if groups and c <= groups[-1][0] * (1 + 1e-3) + 1e-12:
            groups[-1][1] += 1
        else:
            groups.append([c, 1])
    shown = ", ".join(f"{c:.4g} (x{m})" for c, m in groups[:4]) + (" ..." if len(groups) > 4 else "")
    text = (f"multistart: {len(found)} of {len(starts)} fits ended, {len(groups)} distinct minima; "
            f"chi2/dof (starts): {shown}")  # fmt: skip
    return found[0][1], text, True


_MINUIT_ALGORITHMS = {
    "migrad": "gradient (variable metric) minimiser",
    "migrad+minos": "as migrad, plus asymmetric MINOS error intervals (slower)",
    "simplex": "derivative-free simplex, then migrad",
}


def _run_iminuit(ctx, algorithm, options):
    from iminuit import Minuit

    name = algorithm or "migrad"
    if name not in _MINUIT_ALGORITHMS:
        raise ValueError(f"unknown MINUIT algorithm '{name}' (use {', '.join(_MINUIT_ALGORITHMS)})")
    labels = [ctx.problem.parameters[i].label for i in ctx.free]
    f = _penalised(ctx)
    m = Minuit(lambda *u: f(np.array(u)), *np.clip(ctx.u0, ctx.lb, ctx.ub), name=labels)
    m.errordef = 1.0 / ctx.dof  # f = chi^2 / dof
    if options.get("precision"):  # relative noise of the model (MINUIT assumes machine precision by default)
        m.precision = float(options["precision"])
    m.errors = list(0.1 * np.maximum(np.abs(ctx.u0), 0.1))
    m.limits = [(None if not np.isfinite(lo) else lo, None if not np.isfinite(hi) else hi) for lo, hi in zip(ctx.lb, ctx.ub)]
    ncall = int(options.get("maxeval") or 0)
    if name == "simplex":
        m.simplex(ncall=ncall)
    for _ in range(3):  # a restart often reaches the EDM goal that a noisy model misses at the first attempt
        m.migrad(ncall=ncall)
        if m.fmin.is_valid:
            break
    # parameter errors stay those of the least-squares Jacobian at the minimum: MINUIT's HESSE is too
    # sensitive to the integration noise of the model.  MINOS (profile) intervals are added on request.
    extra = ""
    if name == "migrad+minos":
        try:
            m.hesse()
            m.migrad(ncall=ncall)  # MINOS needs a valid minimum: restart from the HESSE state
            m.minos()
            ctx.asym = {j: (m.merrors[lab].lower, m.merrors[lab].upper) for j, lab in enumerate(labels)}
        except Exception as exc:  # noqa: BLE001 -- MINOS can fail on a flat direction
            extra = f"; MINOS failed ({exc})"
    fm = m.fmin
    # with a noisy model EDM rarely gets below the tolerance: judge by the progress instead
    ok = bool(fm.is_valid or fm.edm < 1e-2 / ctx.dof)
    text = ("converged" if fm.is_valid else f"stopped, EDM {fm.edm:.2g}") + extra
    return _finish(ctx, np.array(m.values), text, ok, options)


_BUMPS_ALGORITHMS = {
    "lm": "Levenberg-Marquardt",
    "amoeba": "simplex",
    "de": "differential evolution (finite bounds needed)",
    "dream": "DREAM MCMC: best point + posterior errors (finite bounds needed; many evaluations)",
    "newton": "quasi-Newton BFGS (gradient, local minimum only)",
    "pt": "parallel tempering MCMC (experimental; finite bounds needed)",
}

_BUMPS_NEEDS_BOUNDS = ("de", "dream", "pt")  # methods that sample the box between the bounds

# Options of the bumps fitters that can be given to ``problem.fit("bumps", ..., bumps={...})`` or typed into the
# "bumps options" box of the Fit panel (``steps=2000, pop=20, init=lhs``): steps samples burn pop thin init alpha
# outliers trim (dream); steps pop CR F ftol xtol (de); steps starts radius xtol ftol jump (amoeba); steps starts ftol
# xtol jump (lm, newton); steps nT Tmin Tmax CR burn (pt). Unknown keys of the chosen method raise a ValueError
# (the allowed keys come from the installed bumps, see _bumps_settings).


def parse_option_text(text):
    """``"steps=2000, pop=20, init=lhs"`` -> ``{"steps": 2000, "pop": 20, "init": "lhs"}`` (numbers, True/False, else text)."""
    import ast

    out = {}
    for part in re.split(r"[,;\n]+", text or ""):
        part = part.strip()
        if not part:
            continue
        key, sep, val = part.partition("=")
        key, val = key.strip(), val.strip()
        if not sep or not key:
            raise ValueError(f"option '{part}': write it as name=value")
        try:
            out[key] = ast.literal_eval(val)
        except (ValueError, SyntaxError):
            out[key] = val
    return out


def _bumps_settings(name):
    from bumps.fitters import FITTERS

    for f in FITTERS:
        if f.id == name:
            return dict(f.settings)
    return {}


class _BumpsFitness:
    """The fit as a bumps ``Fitness`` object (the free parameters in their scaled variables)."""

    def __init__(self, ctx):
        from bumps.parameter import Parameter

        self.ctx = ctx
        self.pars = []
        for j, (i, u0, lo, hi) in enumerate(zip(ctx.free, ctx.u0, ctx.lb, ctx.ub)):
            # bumps sorts the parameters by name: the index in front keeps them in the order of the fit
            p = Parameter(float(u0), name=f"{j:03d} {ctx.problem.parameters[i].label}")
            p.range(float(lo), float(hi))
            self.pars.append(p)

    def parameters(self):
        return {p.name: p for p in self.pars}

    def update(self):
        pass

    def _residuals(self):
        ctx = self.ctx
        u = np.array([p.value for p in self.pars])
        r = ctx.residual(u)
        if ctx.constraints:
            r = np.concatenate([r, ctx.penalty * np.array([c.violation(g) for c, g in zip(ctx.constraints, ctx.g(u))])])
        return r

    def residuals(self):
        return self._residuals()

    def nllf(self):
        v = 0.5 * float(np.sum(self._residuals() ** 2))
        if self.ctx.problem.log_uniform:
            u = np.array([p.value for p in self.pars])
            v += 0.5 * self.ctx.problem.prior_penalty(self.ctx.vals(u))
        return v

    def numpoints(self):
        return self.ctx.ndata

    def chisq(self):
        return 2.0 * self.nllf() / self.ctx.dof

    def plot(self, **kw):
        pass

    def save(self, basename):
        pass


def _run_bumps(ctx, algorithm, options):
    from bumps.fitproblem import FitProblem
    from bumps.fitters import fit

    name = algorithm or "lm"
    if name not in _BUMPS_ALGORITHMS:
        raise ValueError(f"unknown bumps method '{name}' (use {', '.join(_BUMPS_ALGORITHMS)})")
    if name in _BUMPS_NEEDS_BOUNDS and not (np.all(np.isfinite(ctx.lb)) and np.all(np.isfinite(ctx.ub))):
        raise ValueError(f"bumps {name}: set a lower and an upper bound for every fitted parameter")
    maxeval = int(options.get("maxeval") or 0)
    kw = {}
    if name == "dream":
        pop, burn = int(options.get("pop", 8)), int(options.get("burn", 100))
        kw = {"pop": pop, "burn": burn, "samples": int(options.get("samples", maxeval or 4000 * len(ctx.u0))), "thin": 1}
    elif maxeval:
        kw = {"steps": max(1, maxeval // max(len(ctx.u0), 1))}
    allowed = _bumps_settings(name)
    given = {k: v for k, v in (options.get("bumps") or {}).items()}
    unknown = sorted(set(given) - set(allowed))
    if unknown:
        raise ValueError(f"bumps {name}: unknown option(s) {', '.join(unknown)} (known: {', '.join(allowed)})")
    kw.update(given)  # explicit bumps options win over the defaults above
    problem = FitProblem(_BumpsFitness(ctx))
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = fit(problem, method=name, verbose=False, **kw)
    text, ok = str(res.message or name), bool(res.success)
    if name in ("dream", "pt") and getattr(res, "state", None) is not None:
        try:  # posterior: covariance and 68 % intervals around the best point
            pts = np.asarray(res.state.draw().points, float)
            ctx.cov_u = np.cov(pts.T) if pts.shape[0] > 2 else None
            best = ctx.best_u if ctx.best_u is not None else np.asarray(res.x, float)
            lo, hi = np.percentile(pts, [15.87, 84.13], axis=0)
            ctx.asym = {j: (float(lo[j] - best[j]), float(hi[j] - best[j])) for j in range(pts.shape[1])}
            text = f"{name.upper() if name == 'pt' else 'DREAM'}: {pts.shape[0]} posterior samples"
        except Exception:  # noqa: BLE001 -- a failed state export must not lose the fit
            pass
        ok = True
    return _finish(ctx, np.asarray(res.x, float), text, ok, options)


def _run_cma(ctx, algorithm, options):
    import cma

    ctx.rescale_to_bounds()
    n = len(ctx.u0)
    f = _penalised(ctx)
    x0 = np.clip(ctx.u0, ctx.lb, ctx.ub)
    width = np.where(np.isfinite(ctx.ub - ctx.lb), ctx.ub - ctx.lb, 4.0)
    sigma0 = float(options.get("sigma0", 0.2 * float(np.median(np.minimum(width, 4.0)))))
    opts = {
        "bounds": [np.where(np.isfinite(ctx.lb), ctx.lb, None).tolist(), np.where(np.isfinite(ctx.ub), ctx.ub, None).tolist()],
        "verbose": -9, "seed": int(options.get("seed", 1)), "tolfun": 1e-10, "tolx": 1e-8,
        "maxfevals": int(options.get("maxeval") or 3000 * n),
    }  # fmt: skip
    if options.get("maxtime"):
        opts["timeout"] = float(options["maxtime"])
    es = cma.CMAEvolutionStrategy(list(x0), sigma0, opts)
    es.optimize(f)
    text = ", ".join(f"{k}" for k in es.stop()) or "finished"
    return _finish(ctx, np.asarray(es.result.xbest, float), text, True, options)


_DFOLS_ALGORITHMS = {
    "": "derivative-free Gauss-Newton on the residuals, bounds (fast, few evaluations)",
    "noisy": "the same, for residuals with noise (adaptive integrals): more evaluations, more robust",
}


def _run_dfols(ctx, algorithm, options):
    try:
        import dfols
    except ImportError:
        raise ImportError("the 'dfols' backend needs DFO-LS (pip install DFO-LS)") from None
    if algorithm not in (None, *_DFOLS_ALGORITHMS):
        raise ValueError(f"unknown DFO-LS variant '{algorithm}' (use '' or 'noisy')")
    n = len(ctx.u0)
    big = 1e20  # DFO-LS takes 'no bound' as +-1e20
    lb = np.where(np.isfinite(ctx.lb), ctx.lb, -big)
    ub = np.where(np.isfinite(ctx.ub), ctx.ub, big)
    x0 = np.clip(ctx.u0, lb, ub)
    rhobeg = min(float(options.get("rhobeg", 0.1)), 0.25 * float(np.min(ub - lb)))  # 10 % steps (scaled variables)

    def fun(u):
        r = ctx.residual(u)
        if ctx.constraints:  # penalty rows for violated constraints, as for scipy
            r = np.concatenate([r, ctx.penalty * np.array([c.violation(g) for c, g in zip(ctx.constraints, ctx.g(u))])])
        return r

    kw = {
        "bounds": (lb, ub), "rhobeg": rhobeg, "rhoend": float(options.get("rhoend", 1e-6)),
        "maxfun": int(options.get("maxeval") or 200 * (n + 1)), "objfun_has_noise": algorithm == "noisy",
        "scaling_within_bounds": False,
    }  # fmt: skip
    import logging

    logging.getLogger("DFOLS").setLevel(logging.ERROR)
    try:
        sol = dfols.solve(fun, x0, do_logging=False, **kw)
    except TypeError:  # an older DFO-LS without do_logging
        sol = dfols.solve(fun, x0, **kw)
    ok = sol.flag == getattr(sol, "EXIT_SUCCESS", 0)
    return _finish(ctx, np.asarray(sol.x, float), str(sol.msg).strip() or "finished", ok, options)


BACKENDS = {
    "scipy": _run_scipy, "scipy-global": _run_scipy_global, "nlopt": _run_nlopt, "iminuit": _run_iminuit,
    "bumps": _run_bumps, "cma": _run_cma, "dfols": _run_dfols,
}  # fmt: skip


def available_backends():
    """``{key: (label, {algorithm: description})}`` of the optimisers whose package is installed."""
    from importlib.util import find_spec

    cma_alg = {"": "covariance matrix adaptation evolution strategy (derivative-free, robust against noise)"}
    table = {
        "scipy": ("scipy least_squares", "scipy", {k: "" for k in SCIPY_ALGORITHMS}),
        "scipy-global": ("scipy global", "scipy", _GLOBAL_ALGORITHMS),
        "nlopt": ("NLopt", "nlopt", nlopt_algorithms()),
        "iminuit": ("iminuit (MINUIT)", "iminuit", _MINUIT_ALGORITHMS),
        "bumps": ("bumps", "bumps", _BUMPS_ALGORITHMS),
        "cma": ("CMA-ES", "cma", cma_alg),
        "dfols": ("DFO-LS", "dfols", _DFOLS_ALGORITHMS),
    }
    out = {k: (label, algs) for k, (label, mod, algs) in table.items() if find_spec(mod) is not None and algs}
    for k in BACKENDS:  # backends registered by the user
        out.setdefault(k, (k, {}))
    return out


def register_backend(name, runner):
    """Add an optimiser: ``runner(ctx, algorithm, options) -> (u, message, success)`` (``ctx``: see ``_Context``)."""
    BACKENDS[name] = runner
