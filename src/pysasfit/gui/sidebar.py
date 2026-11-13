"""Side bar of the launcher: one icon per tool, in a column at the left edge of the window.

It replaces the row of tabs, which no longer fits the screen with a dozen and more tools. Hovering an
icon shows the tool's name and what it does; a click brings the tool to the front (and loads it the first
time). The button at the top widens the bar to icon + name. A double click, dragging an icon out of the
bar, or the right mouse button detach the tool into its own window (and attach it again).

With the names shown, icon and name are flush left and the name is written as in the logos: grey *py*,
the core bold in the tool's colour, the ending grey.

A dot at the lower right of an icon marks a tool that is loaded; a ring one that is detached.
"""

from __future__ import annotations

from PySide6 import QtCore, QtGui, QtWidgets

from ..core.tabs import WORDMARK_SCALE

ICON = 28           # icon size (px)
LEFT = 9            # left margin of the icon when the names are shown
GAP = 8             # between icon and name
#: background of the tool in front (light theme, dark theme); Preferences → pySASfit window
SELECTED = ("#FFF8C5", "#2B2B2B")


def contrast(c1, c2):
    """WCAG contrast ratio (1 … 21) of two colours given as #RRGGBB."""
    def lum(h):
        h = h.lstrip("#")
        out = []
        for i in (0, 2, 4):
            c = int(h[i:i + 2], 16) / 255
            out.append(c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4)
        return 0.2126 * out[0] + 0.7152 * out[1] + 0.0722 * out[2]
    a, b = sorted((lum(c1), lum(c2)), reverse=True)
    return (a + 0.05) / (b + 0.05)
DRAG_OUT = 30       # drag an icon this far to detach its tool


class _ToolButton(QtWidgets.QToolButton):
    detach_requested = QtCore.Signal(object)      # global position (QPoint, null: default place)

    def __init__(self, title, icon, colour, segments=None):
        super().__init__()
        self.title = title
        #: the name as in the logo: [(text, colour on light, colour on dark, bold), ...]
        self.segments = [tuple(x) for x in segments] if segments else None
        self.colour = QtGui.QColor(colour)
        self.state = ""            # "", "loaded", "detached"
        self.selected = SELECTED   # background when in front: (light theme, dark theme)
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_Hover, True)
        self.setIcon(icon)
        self.setText(title)
        self.setIconSize(QtCore.QSize(ICON, ICON))
        self.setCheckable(True)
        self.setAutoRaise(True)
        self.setFocusPolicy(QtCore.Qt.FocusPolicy.NoFocus)
        self.setSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Fixed)
        self._press = None

    def set_state(self, state):
        if state != self.state:
            self.state = state
            self.update()

    # ------------------------------------------------------------------ name (shown next to the icon)
    def _seg_font(self, bold):
        f = QtGui.QFont(self.font())
        f.setBold(bool(bold))
        if f.pointSizeF() > 0:
            f.setPointSizeF(f.pointSizeF() * WORDMARK_SCALE)
        return f

    def _segments(self):
        return self.segments or [(self.title, "#202020", "#e8e8e8", False)]

    def name_width(self):
        return sum(QtGui.QFontMetrics(self._seg_font(b)).horizontalAdvance(t) for t, _l, _d, b in self._segments())

    def _names_shown(self):
        return self.toolButtonStyle() != QtCore.Qt.ToolButtonStyle.ToolButtonIconOnly

    def _dark(self):
        return self.palette().color(QtGui.QPalette.ColorRole.Window).lightness() < 128

    def paintEvent(self, e):
        # drawn here, not by the style: the style's "checked" look (Windows: its blue accent colour) makes
        # the coloured names unreadable. The tool in front gets the background of the preferences.
        p = QtGui.QPainter(self)
        dark = self._dark()
        if self.isChecked():
            p.fillRect(self.rect(), QtGui.QColor(self.selected[1 if dark else 0]))
        elif self.isDown() or self.underMouse():
            p.fillRect(self.rect(), QtGui.QColor(("#4A4A4A" if self.isDown() else "#3A3A3A") if dark
                                                 else ("#D4D4D4" if self.isDown() else "#E6E6E6")))
        y = (self.height() - ICON) // 2
        mode = QtGui.QIcon.Mode.Normal if self.isEnabled() else QtGui.QIcon.Mode.Disabled
        if not self._names_shown():
            self.icon().paint(p, QtCore.QRect((self.width() - ICON) // 2, y, ICON, ICON),
                              QtCore.Qt.AlignmentFlag.AlignCenter, mode)
        else:
            # icon and name flush left; the name as in the logo: grey "py", the core in the tool's colour
            # and bold, the ending grey
            self.icon().paint(p, QtCore.QRect(LEFT, y, ICON, ICON), QtCore.Qt.AlignmentFlag.AlignCenter, mode)
            x = LEFT + ICON + GAP
            for text, light, dk, bold in self._segments():
                f = self._seg_font(bold)
                p.setFont(f)
                p.setPen(QtGui.QColor(dk if dark else light))
                w = QtGui.QFontMetrics(f).horizontalAdvance(text)
                p.drawText(QtCore.QRect(x, 0, w + 1, self.height()),
                           int(QtCore.Qt.AlignmentFlag.AlignLeft | QtCore.Qt.AlignmentFlag.AlignVCenter), text)
                x += w
        p.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
        if self.isChecked():                       # tool in front: bar in its colour at the left edge
            p.fillRect(QtCore.QRect(0, 2, 3, self.height() - 4), self.colour)
        if self.state:
            r = 3.2
            only = not self._names_shown()
            right = (self.width() + ICON) / 2 if only else LEFT + ICON     # right edge of the icon
            c = QtCore.QPointF(right + 1, self.height() - 7)
            p.setPen(QtGui.QPen(QtGui.QColor("white"), 3.0))               # halo: visible on the icon too
            p.setBrush(QtCore.Qt.BrushStyle.NoBrush)
            p.drawEllipse(c, r, r)
            p.setPen(QtGui.QPen(self.colour, 1.4))
            p.setBrush(self.colour if self.state == "loaded" else QtCore.Qt.BrushStyle.NoBrush)
            p.drawEllipse(c, r, r)
        p.end()

    def mousePressEvent(self, e):
        self._press = e.position().toPoint() if e.button() == QtCore.Qt.MouseButton.LeftButton else None
        super().mousePressEvent(e)

    def mouseMoveEvent(self, e):
        if self._press is not None and e.buttons() & QtCore.Qt.MouseButton.LeftButton and \
                not self.rect().adjusted(-DRAG_OUT, -DRAG_OUT, DRAG_OUT, DRAG_OUT).contains(e.position().toPoint()):
            self._press = None
            self.setDown(False)
            gp = e.globalPosition().toPoint()
            QtCore.QTimer.singleShot(0, lambda: self.detach_requested.emit(gp))
            return
        super().mouseMoveEvent(e)

    def mouseReleaseEvent(self, e):
        self._press = None
        super().mouseReleaseEvent(e)

    def mouseDoubleClickEvent(self, e):
        self.detach_requested.emit(QtCore.QPoint())


class ToolSideBar(QtWidgets.QWidget):
    """Column of tool icons. Signals: ``activated(title)`` (click), ``detach_requested(title, QPoint)``,
    ``context_requested(title, QPoint)`` (right mouse button, global position), ``expanded_changed(bool)``."""

    activated = QtCore.Signal(str)
    detach_requested = QtCore.Signal(str, object)
    context_requested = QtCore.Signal(str, object)
    expanded_changed = QtCore.Signal(bool)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.buttons = {}
        self._expanded = False
        outer = QtWidgets.QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)
        self.toggle = QtWidgets.QToolButton()
        self.toggle.setAutoRaise(True)
        self.toggle.setFocusPolicy(QtCore.Qt.FocusPolicy.NoFocus)
        self.toggle.setSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Fixed)
        self.toggle.clicked.connect(lambda: self.set_expanded(not self._expanded))
        outer.addWidget(self.toggle)
        inner = QtWidgets.QWidget()
        self._lay = QtWidgets.QVBoxLayout(inner)
        self._lay.setContentsMargins(0, 2, 0, 2)
        self._lay.setSpacing(1)
        self._lay.addStretch(1)
        self.scroll = QtWidgets.QScrollArea()          # a low window: the column scrolls
        self.scroll.setWidget(inner)
        self.scroll.setWidgetResizable(True)
        self.scroll.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        self.scroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        outer.addWidget(self.scroll, 1)
        self.setSizePolicy(QtWidgets.QSizePolicy.Policy.Fixed, QtWidgets.QSizePolicy.Policy.Expanding)
        self._update_width()

    def add_tool(self, title, icon, colour, tooltip, letters="", segments=None):
        """``letters``: drawn on a tile in the tool's colour when the tool has no icon. ``segments``: the
        name as in the logo, ``[(text, colour on light, colour on dark, bold), ...]`` (None: plain)."""
        if icon is None or icon.isNull():
            icon = _letter_icon(letters or title[:3], colour)
        b = _ToolButton(title, icon, colour, segments)
        b.setToolTip(tooltip)
        b.clicked.connect(lambda _c=False, t=title: self.activated.emit(t))
        b.detach_requested.connect(lambda gp, t=title: self.detach_requested.emit(t, gp))
        b.setContextMenuPolicy(QtCore.Qt.ContextMenuPolicy.CustomContextMenu)
        b.customContextMenuRequested.connect(
            lambda pos, t=title, bb=b: self.context_requested.emit(t, bb.mapToGlobal(pos)))
        self._lay.insertWidget(self._lay.count() - 1, b)
        self.buttons[title] = b
        self._style(b)
        self._update_width()
        return b

    def set_selected_colours(self, light, dark):
        """Background of the tool in front on a light / dark theme (#RRGGBB)."""
        for b in self.buttons.values():
            b.selected = (light, dark)
            b.update()

    # ------------------------------------------------------------------ state
    def set_current(self, title):
        """Mark the tool in front (None / unknown: none)."""
        for t, b in self.buttons.items():
            b.setChecked(t == title)

    def current(self):
        return next((t for t, b in self.buttons.items() if b.isChecked()), None)

    def set_state(self, title, state):
        """``""`` not loaded, ``"loaded"``, ``"detached"``."""
        b = self.buttons.get(title)
        if b is not None:
            b.set_state(state)

    def expanded(self):
        return self._expanded

    def set_expanded(self, on):
        on = bool(on)
        if on == self._expanded:
            return
        self._expanded = on
        for b in self.buttons.values():
            self._style(b)
        self._update_width()
        self.expanded_changed.emit(on)

    def _style(self, b):
        b.setToolButtonStyle(QtCore.Qt.ToolButtonStyle.ToolButtonTextBesideIcon if self._expanded
                             else QtCore.Qt.ToolButtonStyle.ToolButtonIconOnly)
        b.setFixedHeight(ICON + 12)
        b.setMinimumWidth(0)

    def _update_width(self):
        self.toggle.setText("«  hide names" if self._expanded else "»")
        self.toggle.setToolTip("Show the icons only" if self._expanded else "Show the tool names next to the icons")
        if self._expanded:
            w = max([b.name_width() for b in self.buttons.values()] + [60]) + LEFT + ICON + GAP + 14
        else:
            w = ICON + 18
        self.setFixedWidth(w)
        for b in self.buttons.values():
            b.setFixedWidth(w)


def _letter_icon(letters, colour):
    icon = QtGui.QIcon()
    for s in (32, 64, 128):
        pm = QtGui.QPixmap(s, s)
        pm.fill(QtCore.Qt.GlobalColor.transparent)
        p = QtGui.QPainter(pm)
        p.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
        p.setPen(QtCore.Qt.PenStyle.NoPen)
        p.setBrush(QtGui.QColor(colour))
        p.drawRoundedRect(QtCore.QRectF(0, 0, s, s), s * 0.16, s * 0.16)
        f = p.font()
        f.setBold(True)
        f.setPixelSize(int(s * (0.42 if len(letters) <= 3 else 0.30)))
        p.setFont(f)
        p.setPen(QtGui.QColor("white"))
        p.drawText(QtCore.QRect(0, 0, s, s), int(QtCore.Qt.AlignmentFlag.AlignCenter), letters)
        p.end()
        icon.addPixmap(pm)
    return icon
