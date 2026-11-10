""""Copy Python code": a script that reproduces what a tool's panel shows.

Every tool provides ``python_code() -> str``: a self-contained script that loads the same data, makes the same
calculation with every current setting (also the values from Settings → Preferences, which scripts do not read
by themselves) and plots the result in an optional block at the end. This module has the shared parts:

* the Qt side: :func:`install` adds "Copy Python code" (Ctrl+Shift+C) to the window's File menu and as a
  window-wide shortcut, :func:`button` makes a "Copy Python" push button; both call :func:`copy_to_clipboard`;
* the text side: :class:`Script` collects imports and lines; :func:`lit` writes a value as a Python literal
  (numpy scalars and arrays included), :func:`call` a call with keyword arguments, one per line when long.

The launcher lists the entry of the tool in front in its own File menu.
"""
from __future__ import annotations

import datetime
import math
import os

import numpy as np

__all__ = ["Script", "Code", "lit", "call", "path_lit", "nondefault", "prefs_lines", "install", "button", "copy_to_clipboard",
           "SHORTCUT"]

SHORTCUT = "Ctrl+Shift+C"
WIDTH = 110


# --------------------------------------------------------------------------- literals
def _float(v):
    v = float(v)
    if math.isnan(v):
        return 'float("nan")'
    if math.isinf(v):
        return 'float("inf")' if v > 0 else '-float("inf")'
    return repr(v)


def lit(v, indent=0):
    """Python literal of ``v`` (numbers, strings, None, bool, tuples, lists, dicts, numpy scalars / arrays)."""
    if isinstance(v, (bool, np.bool_)):
        return repr(bool(v))
    if v is None or isinstance(v, str):
        return repr(v)
    if isinstance(v, (int, np.integer)):
        return repr(int(v))
    if isinstance(v, (float, np.floating)):
        return _float(v)
    if isinstance(v, complex):
        return repr(v)
    if isinstance(v, np.ndarray):
        if v.ndim == 0:
            return lit(v.item())
        return f"np.array({lit(v.tolist(), indent)})"
    if isinstance(v, dict):
        items = [f"{lit(k)}: {lit(x, indent + 4)}" for k, x in v.items()]
        one = "{" + ", ".join(items) + "}"
        if len(one) + indent <= WIDTH:
            return one
        pad = " " * (indent + 4)
        return "{\n" + "".join(f"{pad}{i},\n" for i in items) + " " * indent + "}"
    if isinstance(v, (list, tuple)):
        items = [lit(x, indent + 4) for x in v]
        if isinstance(v, tuple):
            one = "(" + ", ".join(items) + ("," if len(items) == 1 else "") + ")"
        else:
            one = "[" + ", ".join(items) + "]"
        if len(one) + indent <= WIDTH:
            return one
        o, c = ("(", ")") if isinstance(v, tuple) else ("[", "]")
        pad = " " * (indent + 4)
        return o + "\n" + "".join(f"{pad}{i},\n" for i in items) + " " * indent + c
    if callable(v) and getattr(v, "__name__", "") != "<lambda>" and getattr(v, "__module__", None):
        return f"{v.__module__}.{v.__name__}"
    raise TypeError(f"no Python literal for {type(v).__name__}: {v!r}")


def nondefault(values, defaults):
    """Entries of ``values`` that differ from ``defaults`` (a dict, or a callable whose keyword defaults count)."""
    if callable(defaults):
        import inspect

        defaults = {k: p.default for k, p in inspect.signature(defaults).parameters.items()
                    if p.default is not inspect.Parameter.empty}
    out = {}
    for k, v in values.items():
        d = defaults.get(k, _MISSING)
        try:
            same = d is not _MISSING and type(d) is type(v) and d == v
            same = same or (d is not _MISSING and isinstance(d, (int, float)) and isinstance(v, (int, float))
                            and not isinstance(d, bool) and not isinstance(v, bool) and float(d) == float(v))
        except Exception:  # noqa: BLE001 -- arrays etc.
            same = False
        if not same:
            out[k] = v
    return out


_MISSING = object()


def prefs_lines(*prefixes, apply=None):
    """Lines that set the preferences under ``prefixes`` that differ from the factory defaults (in memory only, the
    preferences file is not changed), after a reset, so that the script does not depend on the preferences of the
    computer it runs on; ``apply``: a call that pushes them into the library (e.g. ``apply_solver_settings()``)."""
    from . import settings as prefs

    vals = {}
    for p in prefixes:
        for k, v in prefs.values(p).items():
            key = f"{p.rstrip('.')}.{k}"
            if not prefs.is_default(key):
                vals[key] = v
    out = ["from pysasfit.core import settings", "settings.reset()   # factory defaults (in memory only)"]
    if vals:
        out += [f"for key, value in {lit(vals)}.items():", "    settings.set_value(key, value)"]
    if apply:
        out.append(apply)
    return out


def path_lit(path):
    """Raw string literal of a file path (``None`` -> a placeholder the user replaces)."""
    if not path:
        return 'r"your_data_file.dat"'   # no file loaded: the user puts in the name
    p = os.path.abspath(str(path))
    return f'r"{p}"' if '"' not in p and not p.endswith("\\") else repr(p)


def call(func, *args, indent=0, **kw):
    """``func(arg, ..., key=value, ...)``; one argument per line when it does not fit on one line.
    Keyword values that are already code (not literals) are passed as :class:`Code`."""
    parts = [a.text if isinstance(a, Code) else lit(a, indent + 4) for a in args]
    parts += [f"{k}={v.text if isinstance(v, Code) else lit(v, indent + 4)}" for k, v in kw.items()]
    one = f"{func}(" + ", ".join(parts) + ")"
    if len(one) + indent <= WIDTH and "\n" not in one:
        return one
    pad = " " * (indent + 4)
    npos = len(args)
    lines = []
    if npos and all(len(p) < 20 and "\n" not in p for p in parts[:npos]):
        lines.append(", ".join(parts[:npos]))       # short positional arguments (q, I, s) on one line
        parts = parts[npos:]
    lines += parts
    return f"{func}(\n" + "".join(f"{pad}{p},\n" for p in lines) + " " * indent + ")"


def _merge_imports(lines):
    """``import x`` lines first, then one ``from m import a, b`` line per module (names sorted)."""
    plain, frm = set(), {}
    for line in lines:
        line = line.strip()
        if line.startswith("from ") and " import " in line and "(" not in line and " as " not in line.split(" import ")[0]:
            mod, names = line[5:].split(" import ", 1)
            frm.setdefault(mod.strip(), set()).update(n.strip() for n in names.split(","))
        else:
            plain.add(line)
    out = sorted(plain, key=lambda s: (not s.startswith("import "), s))
    for mod in sorted(frm):
        names = sorted(frm[mod], key=lambda n: (n.split(" as ")[0].lower(), n))
        line = f"from {mod} import " + ", ".join(names)
        if len(line) > WIDTH:
            line = f"from {mod} import (\n" + "".join(f"    {n},\n" for n in names) + ")"
        out.append(line)
    return out


class Code:
    """Python source used as an argument of :func:`call` as is (a variable, an expression)."""

    def __init__(self, text):
        self.text = text

    def __repr__(self):
        return self.text


class Script:
    """Imports + body + optional plot block of a generated script."""

    def __init__(self, tool, what=""):
        self.tool = tool
        self.what = what
        self.imports = []
        self.body = []
        self.plot = []

    def imp(self, *lines):
        for line in lines:
            if line not in self.imports:
                self.imports.append(line)
        return self

    def add(self, *lines):
        for line in lines:
            self.body.extend(str(line).split("\n"))
        return self

    def blank(self):
        if self.body and self.body[-1] != "":
            self.body.append("")
        return self

    def comment(self, text):
        import textwrap

        self.blank()
        self.body.extend(textwrap.wrap(text, WIDTH, initial_indent="# ", subsequent_indent="# "))
        return self

    def add_plot(self, *lines):
        for line in lines:
            self.plot.extend(str(line).split("\n"))
        return self

    def text(self):
        stamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
        head = [f"# {self.tool}: {self.what}" if self.what else f"# {self.tool}",
                f"# Generated by pySASfit ({stamp}) from the current settings of the panel, including the values",
                "# of Settings -> Preferences that the calculation uses.", ""]
        imports = _merge_imports(self.imports)
        out = head + imports + [""]
        if self.plot:
            out += ["PLOT = True   # False: calculate only", ""]
        body = list(self.body)
        while body and body[-1] == "":
            body.pop()
        out += body
        if self.plot:
            out += ["", "if PLOT:", "    import matplotlib.pyplot as plt", ""]
            out += [("    " + line) if line else "" for line in self.plot]
            if not any("plt.show()" in line for line in self.plot):
                out += ["    plt.show()"]
        return "\n".join(out) + "\n"


# --------------------------------------------------------------------------- Qt side
def copy_to_clipboard(provider, parent=None, status=None):
    """Put ``provider()`` on the clipboard; problems (no data loaded, ...) are shown, not raised.
    ``status``: callable(text) for a message, e.g. ``statusBar().showMessage``. Returns the code or None."""
    from PySide6 import QtWidgets

    try:
        code = provider()
    except Exception as exc:  # noqa: BLE001 -- show, don't crash
        QtWidgets.QMessageBox.information(parent, "Copy Python code", str(exc) or type(exc).__name__)
        return None
    if not code:
        return None
    QtWidgets.QApplication.clipboard().setText(code)
    msg = f"Python code copied to the clipboard ({code.count(chr(10))} lines)"
    if status is not None:
        try:
            status(msg)
        except Exception:  # noqa: BLE001
            pass
    elif isinstance(parent, QtWidgets.QMainWindow):
        parent.statusBar().showMessage(msg, 8000)
    return code


def install(window, provider=None, status=None, menu=None):
    """Add "Copy Python code" (Ctrl+Shift+C) to ``window``: in its File menu (QMainWindow, or ``menu``) and as a
    shortcut of the window and its children. ``provider`` defaults to ``window.python_code``.
    Returns the QAction (also stored as ``window.copy_python_action``)."""
    from PySide6 import QtCore, QtGui, QtWidgets

    provider = provider or window.python_code
    act = QtGui.QAction("Copy &Python code", window)
    act.setShortcut(QtGui.QKeySequence(SHORTCUT))
    act.setShortcutContext(QtCore.Qt.ShortcutContext.WidgetWithChildrenShortcut)
    act.setToolTip("Copy a Python script that reproduces the current calculation (data, settings, plot)")
    act.triggered.connect(lambda: copy_to_clipboard(provider, window, status))
    if menu is None and isinstance(window, QtWidgets.QMainWindow):
        mb = window.menuBar()
        for a in mb.actions():
            if a.menu() is not None and a.text().replace("&", "") == "File":
                menu = a.menu()
        if menu is None:
            menu = QtWidgets.QMenu("&File", window)
            first = mb.actions()[0] if mb.actions() else None
            mb.insertMenu(first, menu) if first is not None else mb.addMenu(menu)
    if menu is not None:
        if menu.actions():
            menu.addSeparator()
        menu.addAction(act)
    window.addAction(act)
    window.copy_python_action = act
    return act


def button(provider, parent=None, status=None, text="Copy Python"):
    """Push button that copies ``provider()`` to the clipboard."""
    from PySide6 import QtWidgets

    b = QtWidgets.QPushButton(text, parent)
    b.setToolTip(f"Copy a Python script that reproduces the current calculation ({SHORTCUT})")
    b.clicked.connect(lambda: copy_to_clipboard(provider, b.window(), status))
    return b
