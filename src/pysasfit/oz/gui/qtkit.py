"""Small Qt toolkit used by the Ornstein-Zernike GUI.

The OZ GUI was written for Tk. Its tabs keep most of their logic in Tk-style
*variables* (``var.get()``, ``var.set()``, ``var.trace_add("write", cb)``),
and that logic carries over unchanged when the variables keep the same
interface. This module therefore provides:

* :class:`StringVar`, :class:`DoubleVar`, :class:`IntVar`, :class:`BooleanVar`
  -- plain Python objects with the Tk variable interface, bound to Qt widgets
  in both directions by the widget factories below;
* widget factories: :func:`entry`, :func:`combo`, :func:`check`, :func:`radio`,
  :func:`label`, :func:`button`, :func:`separator`, :func:`heading`;
* :mod:`tkinter.messagebox` / :mod:`tkinter.filedialog` replacements
  (:func:`showinfo`, :func:`showerror`, :func:`askyesno`,
  :func:`asksaveasfilename`, :func:`askopenfilename`, :func:`askdirectory`)
  that take Tk's arguments (``filetypes``, ``initialfile``, ...);
* :class:`QueuePoller` -- the worker-thread -> GUI-thread queue that the Tk
  code polled with ``after()``;
* :class:`PlotTabs` -- a tab widget of matplotlib figures with toolbars;
* :func:`on_hover` -- enter/leave callbacks (the hover-help boxes).

Dialogs can be intercepted for automated tests with :func:`set_dialog_hook`.
"""

from __future__ import annotations

import os
import queue as _queue

from PySide6 import QtCore, QtGui, QtWidgets

import matplotlib

if matplotlib.get_backend().lower() not in ("qtagg", "qt5agg", "module://matplotlib_inline.backend_inline", "agg"):
    try:
        matplotlib.use("QtAgg")
    except Exception:  # pragma: no cover - backend already fixed by the host application
        pass
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402

__all__ = [
    "TclError",
    "StringVar",
    "DoubleVar",
    "IntVar",
    "BooleanVar",
    "entry",
    "combo",
    "set_combo_values",
    "check",
    "radio",
    "label",
    "heading",
    "button",
    "separator",
    "hbox",
    "vbox",
    "grid",
    "scroll_area",
    "text_view",
    "set_state",
    "on_hover",
    "after",
    "QueuePoller",
    "PlotTabs",
    "showinfo",
    "showerror",
    "showwarning",
    "askyesno",
    "askokcancel",
    "asksaveasfilename",
    "askopenfilename",
    "askdirectory",
    "set_dialog_hook",
    "clipboard_set",
]

# Code that caught tk.TclError around var.get() catches this instead.
TclError = ValueError


# ---------------------------------------------------------------- variables
class _Var:
    """Tk-compatible variable: get/set/trace_add, bound to widgets."""

    _default = ""

    def __init__(self, master=None, value=None, name=None):
        self._value = self._default if value is None else value
        self._traces = {}
        self._next_id = 0
        self._widget_setters = []  # callables(value) that update bound widgets

    # Tk API -------------------------------------------------------------
    def get(self):
        return self._convert(self._value)

    def set(self, value):
        self._value = value
        for setter in list(self._widget_setters):
            setter(value)
        self._fire()

    def trace_add(self, mode, callback):
        modes = (mode,) if isinstance(mode, str) else tuple(mode)
        self._next_id += 1
        tid = f"trace{self._next_id}"
        if "write" in modes:
            self._traces[tid] = callback
        return tid

    def trace_remove(self, mode, cbname):
        self._traces.pop(cbname, None)

    # legacy Tk spelling used in some code
    def trace(self, mode, callback):  # pragma: no cover
        return self.trace_add("write", callback)

    # internals ----------------------------------------------------------
    def _convert(self, value):
        return value

    def _set_from_widget(self, value, source=None):
        """Called by a bound widget: update value and other widgets, then fire traces."""
        self._value = value
        for setter in list(self._widget_setters):
            if getattr(setter, "_source", None) is not source:
                setter(value)
        self._fire()

    def _fire(self):
        for cb in list(self._traces.values()):
            cb("", "", "write")

    def _raw(self):
        return self._value


class StringVar(_Var):
    _default = ""

    def _convert(self, value):
        return "" if value is None else str(value)


class DoubleVar(_Var):
    _default = 0.0

    def _convert(self, value):
        try:
            return float(value)
        except (TypeError, ValueError):
            raise TclError(f'expected floating-point number but got "{value}"')


class IntVar(_Var):
    _default = 0

    def _convert(self, value):
        try:
            return int(value)
        except (TypeError, ValueError):
            try:
                f = float(value)
                if f == int(f):
                    return int(f)
            except (TypeError, ValueError):
                pass
            raise TclError(f'expected integer but got "{value}"')


class BooleanVar(_Var):
    _default = False

    def _convert(self, value):
        if isinstance(value, str):
            return value.strip().lower() in ("1", "true", "yes", "on")
        return bool(value)


def _as_text(var):
    v = var._raw()
    if isinstance(var, BooleanVar):
        return "1" if var.get() else "0"
    if isinstance(v, float) and isinstance(var, DoubleVar):
        return repr(v) if v != int(v) or abs(v) >= 1e16 else str(v)
    return "" if v is None else str(v)


# ------------------------------------------------------------------ widgets
def _bind(var, widget, setter):
    """Register ``setter`` on ``var`` and drop it again when ``widget`` is destroyed."""
    var._widget_setters.append(setter)

    def unbind(*_):
        if setter in var._widget_setters:
            var._widget_setters.remove(setter)

    widget.destroyed.connect(unbind)


def _char_width(widget, chars):
    fm = widget.fontMetrics()
    return fm.horizontalAdvance("0") * chars + 12


def entry(var=None, width=10, parent=None, on_return=None, on_focus_out=None, readonly=False, tooltip=None):
    """QLineEdit bound to a Tk-style variable (both directions)."""
    w = QtWidgets.QLineEdit(parent)
    if width:
        w.setFixedWidth(_char_width(w, width))
    if var is not None:
        w.setText(_as_text(var))

        def to_widget(value, _w=w):
            text = "" if value is None else (str(value) if not isinstance(var, BooleanVar) else _as_text(var))
            if _w.text() != text:
                _w.blockSignals(True)
                _w.setText(text)
                _w.blockSignals(False)

        to_widget._source = w
        _bind(var, w, to_widget)
        w.textEdited.connect(lambda text: var._set_from_widget(text, source=w))
    if on_return is not None:
        w.returnPressed.connect(on_return)
    if on_focus_out is not None:
        w.editingFinished.connect(on_focus_out)
    if readonly:
        w.setReadOnly(True)
    if tooltip:
        w.setToolTip(tooltip)
    return w


def combo(var=None, values=(), width=None, parent=None, editable=False, on_select=None, tooltip=None):
    """QComboBox bound to a variable. ``on_select`` is called after a user selection
    (Tk's <<ComboboxSelected>>)."""
    w = QtWidgets.QComboBox(parent)
    w.setEditable(bool(editable))
    w.addItems([str(v) for v in values])
    if width:
        w.setMinimumWidth(_char_width(w, width))
    w.setSizeAdjustPolicy(QtWidgets.QComboBox.SizeAdjustPolicy.AdjustToContents)
    if var is not None:

        def to_widget(value, _w=w):
            text = "" if value is None else str(value)
            _w.blockSignals(True)
            i = _w.findText(text)
            if i >= 0:
                _w.setCurrentIndex(i)
            elif _w.isEditable():
                _w.setEditText(text)
            else:
                # Tk lets a readonly combobox show any text set from code
                _w.addItem(text)
                _w.setCurrentIndex(_w.count() - 1)
            _w.blockSignals(False)

        to_widget._source = w
        to_widget(var._raw())
        _bind(var, w, to_widget)

        def from_widget(text, _w=w):
            var._set_from_widget(text, source=_w)

        w.textActivated.connect(from_widget)
        if editable:
            w.editTextChanged.connect(from_widget)
    if on_select is not None:
        w.textActivated.connect(lambda *_: on_select())
    if tooltip:
        w.setToolTip(tooltip)
    return w


def set_combo_values(w, values, var=None):
    """Replace the items of a combobox, keeping the variable's current text selected if present."""
    current = var.get() if var is not None else w.currentText()
    w.blockSignals(True)
    w.clear()
    w.addItems([str(v) for v in values])
    i = w.findText(str(current))
    if i >= 0:
        w.setCurrentIndex(i)
    w.blockSignals(False)


def check(var=None, text="", parent=None, command=None, tooltip=None):
    """QCheckBox bound to a BooleanVar (or any variable holding truthy values)."""
    w = QtWidgets.QCheckBox(text, parent)
    if var is not None:
        w.setChecked(bool(var.get()) if isinstance(var, BooleanVar) else bool(var._raw()))

        def to_widget(value, _w=w):
            state = BooleanVar._convert(None, value) if not isinstance(value, bool) else value
            if _w.isChecked() != state:
                _w.blockSignals(True)
                _w.setChecked(state)
                _w.blockSignals(False)

        to_widget._source = w
        _bind(var, w, to_widget)
        w.toggled.connect(lambda state: var._set_from_widget(bool(state), source=w))
    if command is not None:
        w.toggled.connect(lambda *_: command())
    if tooltip:
        w.setToolTip(tooltip)
    return w


def radio(var, value, text="", parent=None, command=None):
    w = QtWidgets.QRadioButton(text, parent)
    w.setChecked(var._raw() == value)

    def to_widget(v, _w=w):
        _w.blockSignals(True)
        _w.setChecked(v == value)
        _w.blockSignals(False)

    to_widget._source = w
    _bind(var, w, to_widget)

    def toggled(state):
        if state:
            var._set_from_widget(value, source=w)
            if command is not None:
                command()

    w.toggled.connect(toggled)
    return w


def label(text="", var=None, parent=None, color=None, wrap=None, bold=False, small=False, mono=False):
    """QLabel with static text, or showing a variable's value."""
    w = QtWidgets.QLabel(text if var is None else _as_text(var), parent)
    if var is not None:

        def to_widget(value, _w=w):
            _w.setText("" if value is None else str(value))

        _bind(var, w, to_widget)
    style = []
    if color:
        style.append(f"color: {color};")
    if style:
        w.setStyleSheet(" ".join(style))
    if wrap:
        w.setWordWrap(True)
        if isinstance(wrap, int) and wrap > 1:
            w.setMaximumWidth(int(wrap * 1.25))
    f = w.font()
    if bold:
        f.setBold(True)
    if small:
        f.setPointSizeF(max(f.pointSizeF() - 1.5, 6))
    if mono:
        f = QtGui.QFontDatabase.systemFont(QtGui.QFontDatabase.SystemFont.FixedFont)
    w.setFont(f)
    w.setTextInteractionFlags(QtCore.Qt.TextInteractionFlag.TextSelectableByMouse)
    return w


def heading(text, parent=None):
    return label(text, parent=parent, bold=True)


def button(text, command=None, parent=None, enabled=True, tooltip=None):
    w = QtWidgets.QPushButton(text, parent)
    if command is not None:
        w.clicked.connect(lambda *_: command())
    w.setEnabled(enabled)
    if tooltip:
        w.setToolTip(tooltip)
    return w


def separator(parent=None):
    w = QtWidgets.QFrame(parent)
    w.setFrameShape(QtWidgets.QFrame.Shape.HLine)
    w.setFrameShadow(QtWidgets.QFrame.Shadow.Sunken)
    return w


def hbox(*widgets, spacing=4, stretch=True):
    """A QWidget laying out ``widgets`` horizontally (``None`` entries = stretch)."""
    host = QtWidgets.QWidget()
    lay = QtWidgets.QHBoxLayout(host)
    lay.setContentsMargins(0, 0, 0, 0)
    lay.setSpacing(spacing)
    for w in widgets:
        if w is None:
            lay.addStretch(1)
        elif isinstance(w, str):
            lay.addWidget(QtWidgets.QLabel(w))
        else:
            lay.addWidget(w)
    if stretch:
        lay.addStretch(1)
    return host


def vbox(*widgets, spacing=4):
    host = QtWidgets.QWidget()
    lay = QtWidgets.QVBoxLayout(host)
    lay.setContentsMargins(0, 0, 0, 0)
    lay.setSpacing(spacing)
    for w in widgets:
        if w is None:
            lay.addStretch(1)
        else:
            lay.addWidget(w)
    return host


def grid(parent=None, spacing=(6, 3)):
    """A QWidget with a QGridLayout; returns (widget, layout)."""
    host = QtWidgets.QWidget(parent)
    lay = QtWidgets.QGridLayout(host)
    lay.setContentsMargins(0, 0, 0, 0)
    lay.setHorizontalSpacing(spacing[0])
    lay.setVerticalSpacing(spacing[1])
    return host, lay


def scroll_area(inner, min_width=None):
    """Vertical scroll area around ``inner`` (the Tk tabs used a Canvas+Scrollbar)."""
    sa = QtWidgets.QScrollArea()
    sa.setWidgetResizable(True)
    sa.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAsNeeded)
    sa.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
    sa.setWidget(inner)
    if min_width:
        sa.setMinimumWidth(min_width)
    else:
        sa.setMinimumWidth(inner.sizeHint().width() + 24)
    return sa


def text_view(mono=True, wrap=False, readonly=True):
    """Read-only text pane (Tk Text used for logs and summaries)."""
    w = QtWidgets.QPlainTextEdit()
    w.setReadOnly(readonly)
    w.setLineWrapMode(
        QtWidgets.QPlainTextEdit.LineWrapMode.WidgetWidth if wrap else QtWidgets.QPlainTextEdit.LineWrapMode.NoWrap
    )
    if mono:
        w.setFont(QtGui.QFontDatabase.systemFont(QtGui.QFontDatabase.SystemFont.FixedFont))
    return w


def set_state(widget, state):
    """Tk-style state: 'normal', 'disabled' or 'readonly'."""
    if state == "readonly" and isinstance(widget, QtWidgets.QLineEdit):
        widget.setEnabled(True)
        widget.setReadOnly(True)
        return
    if isinstance(widget, QtWidgets.QLineEdit):
        widget.setReadOnly(False)
    widget.setEnabled(state != "disabled")


class _HoverFilter(QtCore.QObject):
    def __init__(self, enter, leave, parent):
        super().__init__(parent)
        self._enter, self._leave = enter, leave

    def eventFilter(self, obj, event):
        t = event.type()
        if t == QtCore.QEvent.Type.Enter and self._enter is not None:
            self._enter()
        elif t == QtCore.QEvent.Type.Leave and self._leave is not None:
            self._leave()
        return False


def on_hover(widget, enter=None, leave=None):
    """Call ``enter()`` / ``leave()`` when the mouse enters / leaves ``widget``."""
    f = _HoverFilter(enter, leave, widget)
    widget.installEventFilter(f)
    return f


def after(ms, callback):
    """Tk ``after``: run ``callback`` on the GUI thread after ``ms`` milliseconds."""
    QtCore.QTimer.singleShot(int(ms), callback)


class QueuePoller(QtCore.QObject):
    """Drain a :class:`queue.Queue` filled by worker threads, on the GUI thread.

    ``handler(item)`` is called for every queued item every ``interval`` ms.
    """

    def __init__(self, parent, q, handler, interval=100):
        super().__init__(parent)
        self.queue = q
        self.handler = handler
        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self.poll)
        self.timer.start(int(interval))

    def poll(self):
        while True:
            try:
                item = self.queue.get_nowait()
            except _queue.Empty:
                return
            self.handler(item)

    def stop(self):
        self.timer.stop()


class PlotTabs(QtWidgets.QTabWidget):
    """Tab widget of matplotlib figures, each with a navigation toolbar.

    ``add_plot(label, key)`` returns the axes; ``self.axes[key]``,
    ``self.canvases[key]`` and ``self.figures[key]`` are kept as in the Tk code.
    """

    def __init__(self, parent=None, figsize=(6, 5), dpi=100):
        super().__init__(parent)
        self.axes, self.canvases, self.figures, self.toolbars = {}, {}, {}, {}
        self._figsize, self._dpi = figsize, dpi

    def add_plot(self, label, key, figure=None, subplot=True):
        page = QtWidgets.QWidget()
        lay = QtWidgets.QVBoxLayout(page)
        lay.setContentsMargins(0, 0, 0, 0)
        fig = figure or Figure(figsize=self._figsize, dpi=self._dpi)
        ax = fig.add_subplot(111) if subplot and not fig.axes else (fig.axes[0] if fig.axes else None)
        canvas = FigureCanvasQTAgg(fig)
        canvas.setSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Expanding)
        toolbar = NavigationToolbar2QT(canvas, page)
        # Toolbar ABOVE the plot, as in the other pySASfit tools (the Tk
        # original packed it below).
        lay.addWidget(toolbar)
        lay.addWidget(canvas, 1)
        self.addTab(page, label)
        self.axes[key], self.canvases[key], self.figures[key], self.toolbars[key] = ax, canvas, fig, toolbar
        return ax

    def add_text(self, label, mono=True, wrap=False):
        w = text_view(mono=mono, wrap=wrap)
        self.addTab(w, label)
        return w

    def draw_all(self):
        for c in self.canvases.values():
            c.draw_idle()


# ------------------------------------------------------------------ dialogs
_DIALOG_HOOK = None


def set_dialog_hook(hook):
    """Intercept dialogs (for tests): ``hook(kind, **kwargs)`` returns the answer.

    kinds: 'info', 'error', 'warning', 'yesno', 'okcancel', 'save', 'open', 'directory'.
    Pass None to restore the real dialogs.
    """
    global _DIALOG_HOOK
    _DIALOG_HOOK = hook


def _parent(parent):
    return parent if parent is not None else QtWidgets.QApplication.activeWindow()


def _message(kind, title, message, parent):
    if _DIALOG_HOOK is not None:
        return _DIALOG_HOOK(kind, title=title, message=message)
    box = {
        "info": QtWidgets.QMessageBox.information,
        "error": QtWidgets.QMessageBox.critical,
        "warning": QtWidgets.QMessageBox.warning,
    }[kind]
    box(_parent(parent), title, str(message))
    return "ok"


def showinfo(title="", message="", parent=None, **kw):
    return _message("info", title, message, parent)


def showerror(title="", message="", parent=None, **kw):
    return _message("error", title, message, parent)


def showwarning(title="", message="", parent=None, **kw):
    return _message("warning", title, message, parent)


def askyesno(title="", message="", parent=None, **kw):
    if _DIALOG_HOOK is not None:
        return bool(_DIALOG_HOOK("yesno", title=title, message=message))
    r = QtWidgets.QMessageBox.question(_parent(parent), title, str(message))
    return r == QtWidgets.QMessageBox.StandardButton.Yes


def askokcancel(title="", message="", parent=None, **kw):
    if _DIALOG_HOOK is not None:
        return bool(_DIALOG_HOOK("okcancel", title=title, message=message))
    r = QtWidgets.QMessageBox.question(
        _parent(parent),
        title,
        str(message),
        QtWidgets.QMessageBox.StandardButton.Ok | QtWidgets.QMessageBox.StandardButton.Cancel,
    )
    return r == QtWidgets.QMessageBox.StandardButton.Ok


def _filter(filetypes):
    """Tk filetypes [("CSV", "*.csv"), ...] -> Qt filter string."""
    parts = []
    for name, pattern in filetypes or []:
        pats = pattern if isinstance(pattern, str) else " ".join(pattern)
        parts.append(f"{name} ({pats})")
    return ";;".join(parts)


def _start(initialdir, initialfile):
    d = initialdir or ""
    if initialfile:
        return os.path.join(d, initialfile) if d else initialfile
    return d


def asksaveasfilename(parent=None, title="Save", defaultextension="", filetypes=None, initialfile=None,
                      initialdir=None, **kw):
    """Tk-compatible save dialog: returns a path ('' when cancelled), adding ``defaultextension``."""
    if _DIALOG_HOOK is not None:
        return _DIALOG_HOOK("save", title=title, defaultextension=defaultextension, filetypes=filetypes,
                            initialfile=initialfile, initialdir=initialdir) or ""
    path, _ = QtWidgets.QFileDialog.getSaveFileName(_parent(parent), title, _start(initialdir, initialfile),
                                                    _filter(filetypes))
    if path and defaultextension and not os.path.splitext(path)[1]:
        path += defaultextension
    return path or ""


def askopenfilename(parent=None, title="Open", filetypes=None, initialdir=None, initialfile=None, **kw):
    if _DIALOG_HOOK is not None:
        return _DIALOG_HOOK("open", title=title, filetypes=filetypes, initialdir=initialdir) or ""
    path, _ = QtWidgets.QFileDialog.getOpenFileName(_parent(parent), title, _start(initialdir, initialfile),
                                                    _filter(filetypes))
    return path or ""


def askdirectory(parent=None, title="Choose folder", initialdir=None, **kw):
    if _DIALOG_HOOK is not None:
        return _DIALOG_HOOK("directory", title=title, initialdir=initialdir) or ""
    return QtWidgets.QFileDialog.getExistingDirectory(_parent(parent), title, initialdir or "") or ""


def clipboard_set(text):
    QtWidgets.QApplication.clipboard().setText(text)
