# -*- coding: utf-8 -*-
"""Settings → Preferences dialog: every setting of :mod:`pysasfit.core.settings`,
one tab per tool, saved to the user's preferences file."""

from __future__ import annotations

import html

from PySide6 import QtCore, QtGui, QtWidgets

from . import settings as prefs

__all__ = ["PreferencesDialog", "open_preferences", "add_preferences_menu"]

_listeners = []


def on_saved(callback):
    """Register callback() to be called after the preferences were saved."""
    _listeners.append(callback)


def _fmt(s, v):
    if s.kind == "bool":
        return "on" if v else "off"
    if s.kind == "choice":
        return dict(s.choices).get(v, str(v))
    if s.kind == "float":
        return f"{v:.6g}" if not s.log else f"{v:.3g}"
    return str(v)


class _Row:
    """Editor for one setting."""

    def __init__(self, s, parent):
        self.s = s
        if s.kind == "bool":
            w = QtWidgets.QCheckBox(parent)
        elif s.kind == "choice":
            w = QtWidgets.QComboBox(parent)
            for val, lab in s.choices:
                w.addItem(lab, val)
        elif s.kind == "int":
            w = QtWidgets.QSpinBox(parent)
            w.setRange(int(s.lo if s.lo is not None else -2**31 + 1), int(s.hi if s.hi is not None else 2**31 - 1))
        else:
            w = QtWidgets.QLineEdit(parent)
            w.setMaximumWidth(140)
        tip = f"<p>{html.escape(s.help or s.label)}</p>"     # rich text: Qt wraps it
        w.setToolTip(tip)
        self.w = w
        self.label = QtWidgets.QLabel(s.label, parent)
        self.label.setToolTip(tip)
        self.label.setWordWrap(True)
        self.hint = QtWidgets.QLabel(f"factory: {_fmt(s, s.default)}", parent)
        self.hint.setStyleSheet("color: gray")
        self.set(prefs.get(s.key))
        sig = (w.toggled if s.kind == "bool" else w.currentIndexChanged if s.kind == "choice"
               else w.valueChanged if s.kind == "int" else w.textChanged)
        sig.connect(lambda *_: self.mark())
        self.mark()

    def set(self, v):
        w, s = self.w, self.s
        if s.kind == "bool":
            w.setChecked(bool(v))
        elif s.kind == "choice":
            w.setCurrentIndex(max(w.findData(v), 0))
        elif s.kind == "int":
            w.setValue(int(v))
        else:
            # repr: shortest text that reads back as exactly the same float
            # (a rounded form could fall outside the allowed range)
            w.setText(repr(float(v)) if s.kind == "float" else str(v))

    def value(self):
        w, s = self.w, self.s
        if s.kind == "bool":
            return w.isChecked()
        if s.kind == "choice":
            return w.currentData()
        if s.kind == "int":
            return w.value()
        return s.coerce(w.text().strip())

    def mark(self):
        """Bold label when the value differs from the factory default; red when invalid."""
        try:
            v = self.value()
            ok = True
        except (ValueError, TypeError):
            v, ok = None, False
        f = self.label.font()
        f.setBold(ok and v != self.s.default)
        self.label.setFont(f)
        if self.s.kind in ("float", "str"):
            self.w.setStyleSheet("" if ok else "background: #ffd0d0")


class _Explain(QtCore.QObject):
    """Event filter: show a row's explanation in the panel when the pointer
    enters it or its editor gets the keyboard focus."""

    def __init__(self, dialog):
        super().__init__(dialog)
        self.dialog = dialog
        self.rows = {}

    def watch(self, row):
        for w in (row.label, row.w, row.hint):
            self.rows[w] = row
            w.installEventFilter(self)

    def eventFilter(self, obj, ev):
        if ev.type() in (QtCore.QEvent.Type.Enter, QtCore.QEvent.Type.FocusIn):
            row = self.rows.get(obj)
            if row is not None:
                self.dialog.explain(row)
        return False


class PreferencesDialog(QtWidgets.QDialog):
    def __init__(self, parent=None, tool=None):
        super().__init__(parent)
        self.setWindowTitle("Preferences — numerical settings")
        self.resize(860, 760)
        self.rows = []
        lay = QtWidgets.QVBoxLayout(self)
        intro = QtWidgets.QLabel(
            "Numerical settings that are not on the tools' main panels. They are read when a "
            "calculation starts. <b>Bold</b> = changed from the factory value (hover for an "
            "explanation, or see the panel below). <i>Save</i> keeps them as your defaults for every later start.")
        intro.setWordWrap(True)
        lay.addWidget(intro)
        self.tabs = QtWidgets.QTabWidget(self)
        lay.addWidget(self.tabs, 1)
        self.explain_box = QtWidgets.QLabel(
            "Point at a setting, or click into its field, to see what it does and when to change it.")
        self.explain_box.setWordWrap(True)
        self.explain_box.setTextFormat(QtCore.Qt.TextFormat.RichText)
        self.explain_box.setAlignment(QtCore.Qt.AlignmentFlag.AlignTop | QtCore.Qt.AlignmentFlag.AlignLeft)
        self.explain_box.setMinimumHeight(96)
        self.explain_box.setFrameShape(QtWidgets.QFrame.Shape.StyledPanel)
        self.explain_box.setMargin(6)
        self.explain_box.setTextInteractionFlags(QtCore.Qt.TextInteractionFlag.TextSelectableByMouse)
        lay.addWidget(self.explain_box)
        self._filter = _Explain(self)
        self.tab_rows = {}
        for t, title in prefs.tools():
            page = QtWidgets.QWidget()
            pl = QtWidgets.QVBoxLayout(page)
            rows = []
            for sec in prefs.sections(t):
                g = QtWidgets.QGroupBox(sec.title, page)
                gl = QtWidgets.QGridLayout(g)
                gl.setColumnStretch(0, 1)
                for i, s in enumerate(sec.settings):
                    r = _Row(s, g)
                    gl.addWidget(r.label, i, 0)
                    gl.addWidget(r.w, i, 1)
                    gl.addWidget(r.hint, i, 2)
                    rows.append(r)
                    self._filter.watch(r)
                pl.addWidget(g)
            pl.addStretch(1)
            sc = QtWidgets.QScrollArea()
            sc.setWidgetResizable(True)
            sc.setWidget(page)
            self.tabs.addTab(sc, title)
            self.tab_rows[t] = rows
            self.rows += rows
            if t == tool:
                self.tabs.setCurrentIndex(self.tabs.count() - 1)
        self.tools = [t for t, _ in prefs.tools()]
        path = QtWidgets.QLabel(f"file: {prefs.settings_path()}")
        path.setStyleSheet("color: gray")
        path.setTextInteractionFlags(QtCore.Qt.TextInteractionFlag.TextSelectableByMouse)
        lay.addWidget(path)
        bb = QtWidgets.QHBoxLayout()
        self.btn_reset_tab = QtWidgets.QPushButton("Factory values (this tab)")
        self.btn_reset_all = QtWidgets.QPushButton("Factory values (all)")
        self.btn_cancel = QtWidgets.QPushButton("Cancel")
        self.btn_save = QtWidgets.QPushButton("Save")
        self.btn_save.setDefault(True)
        self.btn_reset_tab.clicked.connect(self.reset_tab)
        self.btn_reset_all.clicked.connect(lambda: [r.set(r.s.default) for r in self.rows])
        self.btn_cancel.clicked.connect(self.reject)
        self.btn_save.clicked.connect(self.save)
        for b in (self.btn_reset_tab, self.btn_reset_all):
            bb.addWidget(b)
        bb.addStretch(1)
        bb.addWidget(self.btn_cancel)
        bb.addWidget(self.btn_save)
        lay.addLayout(bb)

    def explain(self, row):
        """Show the explanation of one setting in the panel below the tabs."""
        s = row.s
        self.explain_box.setText(
            f"<b>{html.escape(s.label)}</b> &nbsp;<span style='color:gray'><code>{s.key}</code> · "
            f"factory: {html.escape(_fmt(s, s.default))}</span><br>{html.escape(s.help or '')}")
        self.current_row = row

    def reset_tab(self):
        t = self.tools[self.tabs.currentIndex()]
        for r in self.tab_rows[t]:
            r.set(r.s.default)

    def save(self):
        """Validate everything, then apply and write the file."""
        new = {}
        for r in self.rows:
            try:
                new[r.s.key] = r.value()
            except (ValueError, TypeError) as exc:
                self.tabs.setCurrentIndex(self.tools.index(r.s.key.split(".")[0]))
                r.w.setFocus()
                QtWidgets.QMessageBox.warning(self, "Invalid value", f"{r.s.label}:\n{exc}")
                return False
        for k, v in new.items():
            prefs.set_value(k, v)
        try:
            path = prefs.save()
        except OSError as exc:
            QtWidgets.QMessageBox.critical(self, "Not saved", f"The preferences could not be written:\n{exc}")
            return False
        for cb in list(_listeners):
            try:
                cb()
            except Exception:       # a listener must not block saving
                pass
        self.saved_path = path
        self.accept()
        return True


def open_preferences(parent=None, tool=None):
    """Show the dialog modally; returns True when saved."""
    dlg = PreferencesDialog(parent, tool)
    return dlg.exec() == QtWidgets.QDialog.DialogCode.Accepted


def add_preferences_menu(window: QtWidgets.QMainWindow, tool=None):
    """Add 'Settings → Preferences…' to a main window's menu bar."""
    menu = window.menuBar().addMenu("&Settings")
    act = QtGui.QAction("&Preferences…", window)
    act.setShortcut(QtGui.QKeySequence.StandardKey.Preferences)
    act.triggered.connect(lambda: open_preferences(window, tool))
    menu.addAction(act)
    return act
