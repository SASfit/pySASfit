# -*- coding: utf-8 -*-
"""Detachable panels for the Qt main windows.

Every panel (input controls, each plot, text output) is a ``QDockWidget``. Inside
the main window it can be moved to another side or stacked as tabs by dragging its
title bar. The detach button in the title bar (or a double-click on the title)
moves the panel into an ordinary, independent window: own task-bar entry,
minimise / maximise, and it does **not** stay on top of the main window (Qt's own
floating docks are tool windows that always stay above their main window).
Closing such a window docks the panel again. Every panel has a border (View →
Panel border: raised (default), sunken, groove, line or none), docked or detached; the active
panel (last clicked or holding the keyboard focus) is framed in colour. **View → Arrange panels**
puts the docked panels side by side, one above the other, in a grid or all as tabs, and cascades or
tiles the detached panel windows. Panels
can be hidden and shown from the **View** menu; **View → Reset layout** docks everything in the arrangement
the window started with (optional panels hidden at the start are hidden again). The arrangement, including detached windows and their
positions, is part of the last session (``dock_state``).
"""

from __future__ import annotations

from PySide6 import QtCore, QtGui, QtWidgets

import json

__all__ = ["DockMixin", "PanelWindow", "plot_panel", "PANEL_BORDERS", "DEFAULT_PANEL_BORDER", "adapt_fonts",
           "keep_fonts_adapted", "ACTIVE_PANEL_COLOUR", "DEFAULT_TITLE_COLOUR", "title_colours"]

#: border drawn around the content of every detachable panel (View → Panel border):
#: name -> (menu label, QFrame shape, QFrame shadow, line width, mid-line width)
PANEL_BORDERS = {
    "sunken": ("Sunken", QtWidgets.QFrame.Shape.Panel, QtWidgets.QFrame.Shadow.Sunken, 2, 0),
    "groove": ("Groove", QtWidgets.QFrame.Shape.Box, QtWidgets.QFrame.Shadow.Sunken, 1, 0),
    "raised": ("Raised", QtWidgets.QFrame.Shape.Panel, QtWidgets.QFrame.Shadow.Raised, 2, 0),
    "line": ("Line", QtWidgets.QFrame.Shape.Box, QtWidgets.QFrame.Shadow.Plain, 1, 0),
    "none": ("None", QtWidgets.QFrame.Shape.NoFrame, QtWidgets.QFrame.Shadow.Plain, 0, 0),
}
DEFAULT_PANEL_BORDER = "raised"

#: colour of the frame around the active panel (the one last clicked or holding the keyboard focus)
ACTIVE_PANEL_COLOUR = "#c8246e"


#: title bars of the docked panels, coloured as window title bars under Windows (Windows 10/11 accent blue)
DEFAULT_TITLE_COLOUR = "#0078d7"
TITLE_MODES = ("pale", "all", "active", "off")


def _title_settings():
    """(mode, colour) of the panel title bars from the preferences (defaults without settings)."""
    mode, col = "pale", DEFAULT_TITLE_COLOUR
    try:
        from . import settings as prefs
        mode = prefs.get("appearance.panel_title.mode") or mode
        col = prefs.get("appearance.panel_title.colour") or col
    except Exception:                      # no preferences module / unknown key: defaults
        pass
    if not QtGui.QColor(str(col)).isValid():
        col = DEFAULT_TITLE_COLOUR
    return (mode if mode in TITLE_MODES else "pale"), str(col)


def title_colours(active, mode=None, colour=None):
    """(background, text) of a panel title bar, or None for the system look.

    mode 'pale': active panel in the colour, the others in a pale tint of it (default);
    'all': every panel in the colour; 'active': only the active panel coloured; 'off': system look."""
    m, c = _title_settings()
    mode, colour = mode or m, colour or c
    if mode == "off" or (mode == "active" and not active):
        return None
    bg = QtGui.QColor(colour)
    if mode == "pale" and not active:
        w = 0.22                                       # 22 % of the colour on white
        bg = QtGui.QColor(round(255 - w * (255 - bg.red())), round(255 - w * (255 - bg.green())),
                          round(255 - w * (255 - bg.blue())))
    lum = 0.2126 * bg.redF() + 0.7152 * bg.greenF() + 0.0722 * bg.blueF()
    return bg.name(), ("#ffffff" if lum < 0.55 else "#1a1a1a")


def _glyph_icon(kind, colour, size=14):
    """Flat window-button glyph ('restore' = two squares, 'close' = ×) in the text colour of the bar."""
    pm = QtGui.QPixmap(size, size)
    pm.fill(QtCore.Qt.GlobalColor.transparent)
    p = QtGui.QPainter(pm)
    p.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
    pen = QtGui.QPen(QtGui.QColor(colour), 1.3)
    p.setPen(pen)
    a, b = 3.0, size - 3.0
    if kind == "close":
        p.drawLine(QtCore.QPointF(a, a), QtCore.QPointF(b, b))
        p.drawLine(QtCore.QPointF(a, b), QtCore.QPointF(b, a))
    else:
        o = 2.5
        p.drawRect(QtCore.QRectF(a, a + o, b - a - o, b - a - o))
        p.drawPolyline([QtCore.QPointF(a + o, a + o), QtCore.QPointF(a + o, a), QtCore.QPointF(b, a),
                        QtCore.QPointF(b, b - o), QtCore.QPointF(b - o, b - o)])
    p.end()
    return QtGui.QIcon(pm)


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


class _ActivePanelFilter(QtCore.QObject):
    """Application event filter of a DockMixin window: a click or the keyboard focus inside one of its
    panels makes that panel the active one (coloured frame)."""

    def __init__(self, win):
        super().__init__(win)
        self._win = win

    def eventFilter(self, obj, event):
        if event.type() in (QtCore.QEvent.Type.MouseButtonPress, QtCore.QEvent.Type.FocusIn) \
                and isinstance(obj, QtWidgets.QWidget):
            try:
                n = self._win._panel_of(obj)
                if n is not None:
                    self._win._mark_active(n)
            except RuntimeError:                      # window being deleted
                pass
        return False


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
        self.setObjectName("panelTitleBar")
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_StyledBackground, True)
        self.colours = None
        self.set_active(False)

    def set_active(self, active):
        """Colour of the bar (preferences: appearance.panel_title.*); the system look when switched off."""
        cols = title_colours(active)
        if cols == self.colours:
            return
        self.colours = cols
        if cols is None:
            self.setStyleSheet("")
            st = self.style()
            self.detach_button.setIcon(st.standardIcon(QtWidgets.QStyle.StandardPixmap.SP_TitleBarNormalButton))
            self.close_button.setIcon(st.standardIcon(QtWidgets.QStyle.StandardPixmap.SP_TitleBarCloseButton))
            return
        bg, fg = cols
        hover = "rgba(255,255,255,70)" if fg == "#ffffff" else "rgba(0,0,0,40)"
        self.setStyleSheet(
            f"QWidget#panelTitleBar {{ background-color: {bg}; }}"
            f"QWidget#panelTitleBar QLabel {{ color: {fg}; background: transparent; }}"
            f"QWidget#panelTitleBar QToolButton {{ border: none; background: transparent; padding: 1px 4px; }}"
            f"QWidget#panelTitleBar QToolButton:hover {{ background: {hover}; }}"
            f"QWidget#panelTitleBar QToolButton#closeButton:hover {{ background: #e81123; }}")
        self.detach_button.setIcon(_glyph_icon("restore", fg))
        self.close_button.setIcon(_glyph_icon("close", fg))
        self.close_button.setObjectName("closeButton")
        f = self.label.font()
        f.setBold(active)
        self.label.setFont(f)

    def mouseDoubleClickEvent(self, event):
        self._on_detach()


class DockMixin:
    """Mixin for a QMainWindow: :meth:`_add_dock` creates the panels,
    :meth:`_finish_docks` adds the View menu and remembers the start layout."""

    def _add_dock(self, name, title, widget, area=QtCore.Qt.DockWidgetArea.RightDockWidgetArea,
                  scrollable=False):
        # every panel scrolls: otherwise the widest / tallest panel content fixes the smallest size of the
        # tool, and one tool that of the whole pySASfit window (pysasfit.core.scrolling). Text boxes, tables
        # and trees scroll themselves. ``scrollable`` is kept for existing callers.
        if not isinstance(widget, QtWidgets.QAbstractScrollArea):
            from .scrolling import scrolled
            widget = scrolled(widget)
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
        arr = menu.addMenu("Arrange panels")
        for label, how in (("Side by side (tile horizontally)", "horizontal"),
                           ("One above the other (tile vertically)", "vertical"),
                           ("Grid", "grid"), ("All as tabs (cascade)", "tabs")):
            a = QtGui.QAction(label, self)
            a.triggered.connect(lambda _c=False, h=how: self.arrange_panels(h))
            arr.addAction(a)
        arr.addSeparator()
        for label, how in (("Detached windows: cascade", "cascade"), ("Detached windows: tile", "tile")):
            a = QtGui.QAction(label, self)
            a.triggered.connect(lambda _c=False, h=how: self.arrange_detached(h))
            arr.addAction(a)
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
        # the active panel (last clicked / focused) gets a coloured frame, as the active window in Origin
        self._active_panel = None
        app = QtWidgets.QApplication.instance()
        if app is not None:
            self._active_filter = _ActivePanelFilter(self)      # own object: the window's eventFilter stays its own
            app.installEventFilter(self._active_filter)
        try:                                                    # new title-bar colours after Preferences → Save
            import weakref
            from .settings_dialog import on_saved
            ref = weakref.ref(self)
            on_saved(lambda: ref() is not None and ref().refresh_title_bars())
        except Exception:
            pass

    def refresh_title_bars(self):
        """Colour every panel title bar again (after a change of the preferences)."""
        act = getattr(self, "_active_panel", None)
        for n, d in getattr(self, "docks", {}).items():
            tb = d.titleBarWidget()
            if isinstance(tb, _TitleBar):
                tb.colours = None                                # force: the settings may have changed
                tb.set_active(n == act)

    # ---------------------------------------------------------------- borders
    @staticmethod
    def _apply_border(frame, key):
        _lab, shape, shadow, lw, mlw = PANEL_BORDERS.get(key, PANEL_BORDERS[DEFAULT_PANEL_BORDER])
        frame.setFrameShape(shape)
        frame.setFrameShadow(shadow)
        frame.setLineWidth(lw)
        frame.setMidLineWidth(mlw)

    def set_panel_border(self, key, chosen=True):
        """Border of every panel: 'sunken', 'groove', 'raised', 'line' or 'none'."""
        if key not in PANEL_BORDERS:
            key = DEFAULT_PANEL_BORDER
        self.panel_border = key
        if chosen:
            self._border_chosen = True
        for f in getattr(self, "panel_frames", {}).values():
            self._apply_border(f, key)
        if getattr(self, "_active_panel", None):
            self._mark_active(self._active_panel, force=True)
        a = getattr(self, "border_actions", {}).get(key)
        if a is not None and not a.isChecked():
            a.setChecked(True)

    # ---------------------------------------------------------------- active panel
    def _panel_of(self, w):
        frames = getattr(self, "panel_frames", {})
        rev = {id(f): n for n, f in frames.items()}
        while w is not None:
            n = rev.get(id(w))
            if n is not None:
                return n
            w = w.parentWidget()
        return None

    def _mark_active(self, name, force=False):
        """Coloured frame around panel ``name``; the other panels keep the normal border."""
        if name == getattr(self, "_active_panel", None) and not force:
            return
        frames = getattr(self, "panel_frames", {})
        old = frames.get(getattr(self, "_active_panel", None))
        if old is not None:
            old.setStyleSheet("")
            self._apply_border(old, getattr(self, "panel_border", DEFAULT_PANEL_BORDER))
        f = frames.get(name)
        if f is None:
            return
        docks = getattr(self, "docks", {})
        for n in (getattr(self, "_active_panel", None), name):
            tb = docks[n].titleBarWidget() if n in docks else None
            if isinstance(tb, _TitleBar):
                tb.set_active(n == name)
        self._active_panel = name
        f.setStyleSheet(f"QFrame#{f.objectName()} {{ border: 2px solid {ACTIVE_PANEL_COLOUR}; }}")

    @property
    def active_panel(self):
        return getattr(self, "_active_panel", None)

    # ---------------------------------------------------------------- arrange
    def _arrangeable(self):
        """Docked, visible panels in reading order (left to right, top to bottom)."""
        ds = [(n, d) for n, d in self.docks.items() if n not in self.detached and not d.isHidden()]
        return sorted(ds, key=lambda nd: (nd[1].geometry().x() < 0, nd[1].geometry().x(), nd[1].geometry().y()))

    def arrange_panels(self, how="horizontal"):
        """View → Arrange panels: 'horizontal' (side by side), 'vertical' (one above the other),
        'grid', or 'tabs' (all in one place as tabs, like cascaded windows)."""
        ds = [d for _n, d in self._arrangeable()]
        if len(ds) < 2:
            return
        H, V = QtCore.Qt.Orientation.Horizontal, QtCore.Qt.Orientation.Vertical
        area = QtCore.Qt.DockWidgetArea.LeftDockWidgetArea
        for d in ds:
            self.removeDockWidget(d)
        self.addDockWidget(area, ds[0])
        ds[0].show()
        if how == "tabs":
            for d in ds[1:]:
                self.tabifyDockWidget(ds[0], d)
                d.show()
            ds[0].raise_()
        elif how in ("horizontal", "vertical"):
            for a, b in zip(ds, ds[1:]):
                self.splitDockWidget(a, b, H if how == "horizontal" else V)
                b.show()
        else:                                     # grid: columns first, then the panels below each other
            import math
            ncol = int(math.ceil(math.sqrt(len(ds))))
            cols = [ds[i::ncol] for i in range(ncol)]
            heads = [c[0] for c in cols if c]
            for a, b in zip(heads, heads[1:]):
                self.splitDockWidget(a, b, H)
                b.show()
            for c in cols:
                for a, b in zip(c, c[1:]):
                    self.splitDockWidget(a, b, V)
                    b.show()
        self.hide_stray_tab_bars()

    def arrange_detached(self, how="cascade"):
        """Detached panel windows: 'cascade' (stepped) or 'tile' (side by side on the screen)."""
        ws = [w for w in self.detached.values() if w.isVisible()]
        if not ws:
            return
        scr = (self.screen() or QtGui.QGuiApplication.primaryScreen()).availableGeometry()
        if how == "tile":
            import math
            ncol = int(math.ceil(math.sqrt(len(ws))))
            nrow = int(math.ceil(len(ws) / ncol))
            w0, h0 = scr.width() // ncol, scr.height() // nrow
            for i, w in enumerate(ws):
                w.setGeometry(scr.x() + (i % ncol) * w0, scr.y() + (i // ncol) * h0, w0, h0 - 30)
        else:
            step = 32
            for i, w in enumerate(ws):
                w.setGeometry(scr.x() + 40 + i * step, scr.y() + 40 + i * step,
                              int(scr.width() * 0.6), int(scr.height() * 0.6))
                w.raise_()

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
        self.hide_stray_tab_bars()

    def hide_stray_tab_bars(self):
        """Qt can leave the tab bar of a former tab group visible after ``restoreState`` (with more
        than one group of tabbed panels): a tab bar that does not touch the panel of its current tab
        is hidden. Run again from the event loop, after the layout has settled."""
        def run():
            for tb in self.findChildren(QtWidgets.QTabBar):
                if tb.parent() is not self or not tb.isVisible():
                    continue
                if tb.count() == 0:
                    tb.hide()
                    continue
                title = tb.tabText(tb.currentIndex())
                docks = [d for d in self.docks.values() if d.windowTitle() == title and d.isVisible()
                         and not d.isFloating()]
                g = tb.geometry()
                touch = any(abs(d.geometry().bottom() + 1 - g.top()) <= 6 or abs(g.bottom() + 1 - d.geometry().top()) <= 6
                            for d in docks)
                if not touch:
                    tb.hide()
        run()
        QtCore.QTimer.singleShot(0, run)

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
                           "border": getattr(self, "panel_border", DEFAULT_PANEL_BORDER),
                           "border_chosen": bool(getattr(self, "_border_chosen", False))})

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
        if st.get("border") in PANEL_BORDERS and st.get("border_chosen"):   # older sessions: the default
            self.set_panel_border(st["border"])
        for n, g in (st.get("detached") or {}).items():
            if n in self.docks:
                self.detach(n, g if isinstance(g, list) and len(g) == 4 else None)
        for n in self.docks:
            self._sync_view_action(n)
        self.hide_stray_tab_bars()
        return ok
