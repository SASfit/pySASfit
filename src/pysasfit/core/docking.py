# -*- coding: utf-8 -*-
"""Detachable panels for the Qt main windows.

Every panel (input controls, each plot, text output) is a ``QDockWidget``. Inside
the main window it can be moved to another side or stacked as tabs by dragging its
title bar. The detach button in the title bar (or a double-click on the title)
moves the panel into an ordinary, independent window: own task-bar entry,
minimise / maximise, and it does **not** stay on top of the main window (Qt's own
floating docks are tool windows that always stay above their main window).
Closing such a window docks the panel again. Every panel has a border (View →
Panel border: sunken, groove, raised, line or none), docked or detached. Panels
can be hidden and shown from the **View** menu; **View → Reset layout** docks everything in the arrangement
the window started with (optional panels hidden at the start are hidden again). The arrangement, including detached windows and their
positions, is part of the last session (``dock_state``).
"""

from __future__ import annotations

from PySide6 import QtCore, QtGui, QtWidgets

import json

__all__ = ["DockMixin", "PanelWindow", "plot_panel", "PANEL_BORDERS", "DEFAULT_PANEL_BORDER", "adapt_fonts",
           "keep_fonts_adapted"]

#: border drawn around the content of every detachable panel (View → Panel border):
#: name -> (menu label, QFrame shape, QFrame shadow, line width, mid-line width)
PANEL_BORDERS = {
    "sunken": ("Sunken", QtWidgets.QFrame.Shape.Panel, QtWidgets.QFrame.Shadow.Sunken, 2, 0),
    "groove": ("Groove", QtWidgets.QFrame.Shape.Box, QtWidgets.QFrame.Shadow.Sunken, 1, 0),
    "raised": ("Raised", QtWidgets.QFrame.Shape.Panel, QtWidgets.QFrame.Shadow.Raised, 2, 0),
    "line": ("Line", QtWidgets.QFrame.Shape.Box, QtWidgets.QFrame.Shadow.Plain, 1, 0),
    "none": ("None", QtWidgets.QFrame.Shape.NoFrame, QtWidgets.QFrame.Shadow.Plain, 0, 0),
}
DEFAULT_PANEL_BORDER = "sunken"


#: icon size (pixels) of the matplotlib toolbars above the plots
TOOLBAR_ICON_SIZE = 16


def small_toolbar(tb, size=None):
    """Matplotlib navigation toolbar with small icons (default 16 px instead of 24)."""
    s = int(size or TOOLBAR_ICON_SIZE)
    tb.setIconSize(QtCore.QSize(s, s))
    tb.layout().setSpacing(1)
    tb.setMaximumHeight(s + 10)
    return tb


#: figure size (inches) down to which the fonts keep their normal size, and the smallest
#: font scale used for smaller figures
FONT_REF_SIZE = (4.0, 2.9)
FONT_MIN_SCALE = 0.6


def _scaled(t, s):
    b = getattr(t, "_pf_base_size", None)
    if b is None:
        b = t.get_fontsize()
        t._pf_base_size = b
    t.set_fontsize(b * s)


def adapt_fonts(fig):
    """Scale all fonts of ``fig`` with the figure size, so that small panels still show
    every axis label and tick label (constrained layout cannot make room for text
    that is larger than the figure). The size is measured in inches, i.e. it does
    not depend on the screen scaling. Call after drawing and on every resize."""
    w, h = fig.get_size_inches()
    s = max(FONT_MIN_SCALE, min(1.0, w / FONT_REF_SIZE[0], h / FONT_REF_SIZE[1]))
    if fig._suptitle is not None:
        _scaled(fig._suptitle, s)
    for ax in fig.axes:
        for t in (ax.xaxis.label, ax.yaxis.label, ax.title, ax._left_title, ax._right_title,
                  ax.xaxis.get_offset_text(), ax.yaxis.get_offset_text(), *ax.texts):
            _scaled(t, s)
        for name, axis in (("x", ax.xaxis), ("y", ax.yaxis)):
            base = getattr(axis, "_pf_tick_size", None)
            if base is None:
                ticks = axis.get_major_ticks()
                base = ticks[0].label1.get_fontsize() if ticks else 10.0
                axis._pf_tick_size = base
            ax.tick_params(axis=name, which="both", labelsize=base * s)
        lg = ax.get_legend()
        if lg is not None:
            for t in lg.get_texts():
                _scaled(t, s)
            if lg.get_title() is not None:
                _scaled(lg.get_title(), s)
    for lg in fig.legends:
        for t in lg.get_texts():
            _scaled(t, s)
    return s


def keep_fonts_adapted(canvas):
    """Re-scale the fonts of the canvas' figure whenever the canvas is resized."""
    canvas.mpl_connect("resize_event", lambda _e: adapt_fonts(canvas.figure))


def plot_panel(canvas, toolbar_parent, bars=()):
    """Widget with the matplotlib toolbar, optional control bars (QLayouts or
    QWidgets) and the canvas below."""
    from matplotlib.backends.backend_qtagg import NavigationToolbar2QT
    w = QtWidgets.QWidget()
    lay = QtWidgets.QVBoxLayout(w)
    lay.setContentsMargins(2, 2, 2, 2)
    top = QtWidgets.QHBoxLayout()
    top.addWidget(small_toolbar(NavigationToolbar2QT(canvas, toolbar_parent)))
    top.addStretch(1)
    lay.addLayout(top)
    for b in bars:
        if isinstance(b, QtWidgets.QLayout):
            lay.addLayout(b)
        else:
            lay.addWidget(b)
    lay.addWidget(canvas, 1)
    # no large minimum size: a panel that does not fit is clipped by Qt (labels cut off at
    # the edges); instead the canvas shrinks and the fonts follow (adapt_fonts)
    canvas.setMinimumSize(120, 90)
    keep_fonts_adapted(canvas)
    return w


class PanelWindow(QtWidgets.QWidget):
    """Independent top-level window holding a detached panel (no parent: not kept
    above the main window, own task-bar entry). Closing it docks the panel again."""

    def __init__(self, title, on_close):
        super().__init__(None, QtCore.Qt.WindowType.Window)
        self.setWindowTitle(title)
        self._on_close = on_close
        self._closing_app = False
        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)

    def closeEvent(self, event):
        if not self._closing_app:
            self._on_close()
        super().closeEvent(event)


class _TitleBar(QtWidgets.QWidget):
    """Title bar of a docked panel: title, detach button, close button.
    Dragging it moves the panel inside the main window."""

    def __init__(self, dock, on_detach):
        super().__init__(dock)
        self._on_detach = on_detach
        lay = QtWidgets.QHBoxLayout(self)
        lay.setContentsMargins(6, 1, 2, 1)
        lay.setSpacing(2)
        self.label = QtWidgets.QLabel(dock.windowTitle())
        lay.addWidget(self.label)
        lay.addStretch(1)
        st = self.style()
        b = QtWidgets.QToolButton()
        b.setIcon(st.standardIcon(QtWidgets.QStyle.StandardPixmap.SP_TitleBarNormalButton))
        b.setToolTip("Detach into a separate window (closing that window docks it again)")
        b.setAutoRaise(True)
        # clicked(bool) passes 'checked': never forward it as the panel name
        b.clicked.connect(lambda _checked=False: on_detach())
        lay.addWidget(b)
        c = QtWidgets.QToolButton()
        c.setIcon(st.standardIcon(QtWidgets.QStyle.StandardPixmap.SP_TitleBarCloseButton))
        c.setToolTip("Hide this panel (View menu shows it again)")
        c.setAutoRaise(True)
        c.clicked.connect(lambda _checked=False: dock.close())
        lay.addWidget(c)
        self.detach_button, self.close_button = b, c

    def mouseDoubleClickEvent(self, event):
        self._on_detach()


class DockMixin:
    """Mixin for a QMainWindow: :meth:`_add_dock` creates the panels,
    :meth:`_finish_docks` adds the View menu and remembers the start layout."""

    def _add_dock(self, name, title, widget, area=QtCore.Qt.DockWidgetArea.RightDockWidgetArea,
                  scrollable=False):
        if scrollable and not isinstance(widget, QtWidgets.QScrollArea):
            sc = QtWidgets.QScrollArea()
            sc.setWidget(widget)
            sc.setWidgetResizable(True)
            widget = sc
        if isinstance(widget, QtWidgets.QScrollArea):
            widget.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)     # the panel border replaces it
        frame = QtWidgets.QFrame()
        frame.setObjectName(f"panel_{name}")
        lay = QtWidgets.QVBoxLayout(frame)
        lay.setContentsMargins(1, 1, 1, 1)
        lay.addWidget(widget)
        d = QtWidgets.QDockWidget(title, self)
        d.setObjectName(f"dock_{name}")          # needed by saveState/restoreState
        d.setWidget(frame)
        # no Qt floating (tool window, always above the main window): detaching
        # goes to an independent PanelWindow instead
        d.setFeatures(QtWidgets.QDockWidget.DockWidgetFeature.DockWidgetMovable
                      | QtWidgets.QDockWidget.DockWidgetFeature.DockWidgetClosable)
        d.setAllowedAreas(QtCore.Qt.DockWidgetArea.AllDockWidgetAreas)
        d.setTitleBarWidget(_TitleBar(d, lambda n=name: self.detach(n)))
        self.addDockWidget(area, d)
        if not hasattr(self, "docks"):
            self.docks, self.detached, self.view_actions, self.panel_frames = {}, {}, {}, {}
        self.docks[name] = d
        self.panel_frames[name] = frame
        self._apply_border(frame, getattr(self, "panel_border", DEFAULT_PANEL_BORDER))
        d.visibilityChanged.connect(lambda _v, n=name: self._sync_view_action(n))
        return d

    def _finish_docks(self):
        self.setDockNestingEnabled(True)
        self.setDockOptions(QtWidgets.QMainWindow.DockOption.AnimatedDocks
                            | QtWidgets.QMainWindow.DockOption.AllowNestedDocks
                            | QtWidgets.QMainWindow.DockOption.AllowTabbedDocks)
        menu = None
        for a in self.menuBar().actions():
            if a.menu() is not None and a.text().replace("&", "") == "View":
                menu = a.menu()
        if menu is None:
            menu = self.menuBar().addMenu("&View")
        for name, d in self.docks.items():
            act = QtGui.QAction(d.windowTitle(), self)
            act.setCheckable(True)
            act.setChecked(not d.isHidden())
            act.triggered.connect(lambda checked, n=name: self.set_panel_visible(n, checked))
            self.view_actions[name] = act
            menu.addAction(act)
        menu.addSeparator()
        sub = menu.addMenu("Panel border")
        grp = QtGui.QActionGroup(self)
        grp.setExclusive(True)
        self.border_actions = {}
        for key, (label, *_rest) in PANEL_BORDERS.items():
            a = QtGui.QAction(label, self)
            a.setCheckable(True)
            a.setChecked(key == getattr(self, "panel_border", DEFAULT_PANEL_BORDER))
            a.triggered.connect(lambda _c, k=key: self.set_panel_border(k))
            grp.addAction(a)
            sub.addAction(a)
            self.border_actions[key] = a
        menu.addSeparator()
        act = QtGui.QAction("Dock all panels", self)
        act.triggered.connect(self.dock_all)
        menu.addAction(act)
        act = QtGui.QAction("Reset layout", self)
        act.triggered.connect(self.reset_layout)
        menu.addAction(act)
        self.view_menu = menu
        # arrangement at construction (before a session or project can change it);
        # panels hidden at that point (optional panels) stay hidden after Reset layout
        self._start_layout = self.saveState()
        self._start_hidden = {n for n, d in self.docks.items() if d.isHidden()}

    # ---------------------------------------------------------------- borders
    @staticmethod
    def _apply_border(frame, key):
        _lab, shape, shadow, lw, mlw = PANEL_BORDERS.get(key, PANEL_BORDERS[DEFAULT_PANEL_BORDER])
        frame.setFrameShape(shape)
        frame.setFrameShadow(shadow)
        frame.setLineWidth(lw)
        frame.setMidLineWidth(mlw)

    def set_panel_border(self, key):
        """Border of every panel: 'sunken', 'groove', 'raised', 'line' or 'none'."""
        if key not in PANEL_BORDERS:
            key = DEFAULT_PANEL_BORDER
        self.panel_border = key
        for f in getattr(self, "panel_frames", {}).values():
            self._apply_border(f, key)
        a = getattr(self, "border_actions", {}).get(key)
        if a is not None and not a.isChecked():
            a.setChecked(True)

    # ---------------------------------------------------------------- detach / dock
    def detach(self, name, geometry=None):
        """Move panel ``name`` into an independent window."""
        if name in self.detached:
            w = self.detached[name]
            w.raise_()
            w.activateWindow()
            return w
        d = self.docks[name]
        content = d.widget()
        size = content.size() if content.isVisible() else QtCore.QSize(700, 500)
        pos = content.mapToGlobal(QtCore.QPoint(0, 0)) if content.isVisible() else None
        title = d.windowTitle()
        app_title = self.window().windowTitle().split("–")[0].split(" - ")[0].strip() or "pySASfit"
        w = PanelWindow(f"{title} – {app_title}", lambda n=name: self.dock_panel(n))
        w.layout().addWidget(content)            # reparents the content
        content.show()
        if geometry:
            w.setGeometry(*[int(v) for v in geometry])
        else:
            w.resize(max(size.width(), 400), max(size.height(), 300))
            if pos is not None:
                w.move(pos + QtCore.QPoint(30, 30))
        self.detached[name] = w
        d.hide()
        w.show()
        self._sync_view_action(name)
        return w

    def dock_panel(self, name):
        """Put a detached panel back into its dock."""
        w = self.detached.pop(name, None)
        if w is None:
            return
        d = self.docks[name]
        content = w.layout().itemAt(0).widget() if w.layout().count() else None
        if content is not None:
            d.setWidget(content)
        d.show()
        if w.isVisible():
            w._closing_app = True          # avoid a second dock_panel from its closeEvent
            w.close()
        w.deleteLater()
        self._sync_view_action(name)

    def dock_all(self):
        for name in list(self.detached):
            self.dock_panel(name)

    def close_detached(self):
        """Close the detached windows without docking (main window closes)."""
        for w in self.detached.values():
            w._closing_app = True
            w.close()

    def set_panel_visible(self, name, visible):
        if visible:
            if name in self.detached:
                self.detach(name)                # raise it
            else:
                self.docks[name].show()
                self.docks[name].raise_()        # front tab when it shares a place with others
        else:
            if name in self.detached:
                self.dock_panel(name)
            self.docks[name].hide()
        self._sync_view_action(name)

    def _sync_view_action(self, name):
        act = getattr(self, "view_actions", {}).get(name)
        if act is not None:
            act.setChecked(name in self.detached or not self.docks[name].isHidden())

    def reset_layout(self):
        """Back to the arrangement at start: all panels docked and visible."""
        self.dock_all()
        if self._start_layout is not None:
            self.restoreState(self._start_layout)
        for n, d in self.docks.items():
            d.setVisible(n not in getattr(self, "_start_hidden", ()))
            self._sync_view_action(n)

    def closeEvent(self, event):
        self.close_detached()
        super().closeEvent(event)

    # ---------------------------------------------------------------- persistence
    def dock_state(self) -> str:
        geo = {}
        for n, w in self.detached.items():
            g = w.geometry()
            geo[n] = [g.x(), g.y(), g.width(), g.height()]
        return json.dumps({"qt": bytes(self.saveState().toBase64()).decode("ascii"), "detached": geo,
                           "border": getattr(self, "panel_border", DEFAULT_PANEL_BORDER)})

    def restore_dock_state(self, s) -> bool:
        if not s:
            return False
        try:
            st = json.loads(s)
        except (ValueError, TypeError):
            st = {"qt": s, "detached": {}}          # plain Qt state (first version)
        if not isinstance(st, dict):
            return False
        ok = False
        try:
            self.dock_all()
            ok = bool(self.restoreState(QtCore.QByteArray.fromBase64(str(st.get("qt", "")).encode("ascii"))))
        except Exception:  # noqa: BLE001
            pass
        if st.get("border") in PANEL_BORDERS:
            self.set_panel_border(st["border"])
        for n, g in (st.get("detached") or {}).items():
            if n in self.docks:
                self.detach(n, g if isinstance(g, list) and len(g) == 4 else None)
        for n in self.docks:
            self._sync_view_action(n)
        return ok
