# -*- coding: utf-8 -*-
"""Tab widget whose tabs can be detached into independent windows and attached again.

Used by the pySASfit launcher (one tab per tool) and by tools with tabs of their own
(pyTOOLbox). Any ``QTabWidget`` can be replaced by :class:`DetachableTabWidget`; nothing
else changes for the pages.

Detach a tab by

* dragging it out of the tab bar,
* double-clicking it,
* the tab bar's context menu (right click), or
* :meth:`DetachableTabWidget.detach` / the launcher's View menu.

A detached tab is an ordinary top-level window (own task-bar entry, minimise / maximise,
not kept above the main window). A page that is a ``QMainWindow`` (every pySASfit tool)
becomes the window itself and shows its own menu bar again; any other page gets a
small window around it. The **Attach** button in the window's menu bar (or simply
closing the window) puts the tab back at its original place. The context menu of the
tab bar lists the detached tabs and has "Attach all".

:meth:`DetachableTabWidget.state` / :meth:`restore_state` give the detached tabs and
their window geometries as plain JSON data, for the last session.

A tab can show a coloured *wordmark* instead of its plain text
(:meth:`DetachableTabWidget.set_wordmark`, e.g. the logo labels of the pySASfit
tools: grey "py", bold core in the tool's colour, grey ending). The plain text stays the tab's
name (``tabText``), used for menus, window titles and the last session.
"""

from __future__ import annotations

from PySide6 import QtCore, QtGui, QtWidgets

__all__ = ["DetachableTabWidget", "DragOutTabBar"]

#: how far (pixels) a tab must be dragged away from the tab bar to detach it
DRAG_OUT_DISTANCE = 30

#: wordmark labels are drawn slightly larger than the normal tab text
WORDMARK_SCALE = 1.15


class DragOutTabBar(QtWidgets.QTabBar):
    """Tab bar that reports a tab dragged out of it (and double-clicked tabs)."""

    detach_requested = QtCore.Signal(int, QtCore.QPoint)     # tab index, global cursor position

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMovable(True)          # reorder inside the bar as usual
        self._press_index = -1

    # ---------------------------------------------------------------- wordmarks
    # tabData(i) = [(text, colour on light, colour on dark, bold), ...] draws the label in colour

    def _segments(self, i):
        d = self.tabData(i)
        return d if isinstance(d, (list, tuple)) and d and isinstance(d[0], (list, tuple)) else None

    def _dark(self):
        return self.palette().color(QtGui.QPalette.ColorRole.WindowText).lightness() > 128

    def _seg_font(self, bold):
        f = QtGui.QFont(self.font())
        f.setBold(bool(bold))
        if f.pointSizeF() > 0:
            f.setPointSizeF(f.pointSizeF() * WORDMARK_SCALE)
        return f

    def _seg_width(self, segs):
        return sum(QtGui.QFontMetrics(self._seg_font(b)).horizontalAdvance(t) for t, _l, _d, b in segs)

    def tabSizeHint(self, i):
        s = super().tabSizeHint(i)
        segs = self._segments(i)
        if segs:
            plain = self.fontMetrics().horizontalAdvance(self.tabText(i))
            s.setWidth(s.width() - plain + self._seg_width(segs) + 2)
        return s

    def minimumTabSizeHint(self, i):
        return self.tabSizeHint(i) if self._segments(i) else super().minimumTabSizeHint(i)

    def paintEvent(self, e):
        if not any(self._segments(i) for i in range(self.count())):
            return super().paintEvent(e)
        p = QtWidgets.QStylePainter(self)
        dark = self._dark()
        order = [i for i in range(self.count()) if i != self.currentIndex()] + \
            ([self.currentIndex()] if self.currentIndex() >= 0 else [])     # selected tab on top
        for i in order:
            opt = QtWidgets.QStyleOptionTab()
            self.initStyleOption(opt, i)
            segs = self._segments(i)
            if not segs:
                p.drawControl(QtWidgets.QStyle.ControlElement.CE_TabBarTab, opt)
                continue
            text_rect = self.style().subElementRect(QtWidgets.QStyle.SubElement.SE_TabBarTabText, opt, self)
            opt.text = ""
            p.drawControl(QtWidgets.QStyle.ControlElement.CE_TabBarTab, opt)
            x = text_rect.x() + max(0, (text_rect.width() - self._seg_width(segs)) // 2)
            for text, light, dk, bold in segs:
                f = self._seg_font(bold)
                p.setFont(f)
                p.setPen(QtGui.QColor(dk if dark else light))
                fm = QtGui.QFontMetrics(f)
                w = fm.horizontalAdvance(text)
                p.drawText(QtCore.QRect(x, text_rect.y(), w + 1, text_rect.height()),
                           int(QtCore.Qt.AlignmentFlag.AlignLeft | QtCore.Qt.AlignmentFlag.AlignVCenter), text)
                x += w

    def mousePressEvent(self, e):
        self._press_index = self.tabAt(e.position().toPoint()) \
            if e.button() == QtCore.Qt.MouseButton.LeftButton else -1
        super().mousePressEvent(e)

    def mouseMoveEvent(self, e):
        if self._press_index >= 0 and e.buttons() & QtCore.Qt.MouseButton.LeftButton:
            d = DRAG_OUT_DISTANCE
            if not self.rect().adjusted(-d, -d, d, d).contains(e.position().toPoint()):
                idx, self._press_index = self._press_index, -1
                # end the tab bar's own move first (otherwise it keeps a dragged tab)
                rel = QtGui.QMouseEvent(QtCore.QEvent.Type.MouseButtonRelease, e.position(), e.globalPosition(),
                                        QtCore.Qt.MouseButton.LeftButton, QtCore.Qt.MouseButton.NoButton,
                                        e.modifiers())
                super().mouseReleaseEvent(rel)
                gp = e.globalPosition().toPoint()
                QtCore.QTimer.singleShot(0, lambda: self.detach_requested.emit(idx, gp))
                return
        super().mouseMoveEvent(e)

    def mouseReleaseEvent(self, e):
        self._press_index = -1
        super().mouseReleaseEvent(e)

    def mouseDoubleClickEvent(self, e):
        idx = self.tabAt(e.position().toPoint())
        if idx >= 0:
            self.detach_requested.emit(idx, QtCore.QPoint())
            return
        super().mouseDoubleClickEvent(e)


class _PageWindow(QtWidgets.QMainWindow):
    """Window around a detached page that is not a QMainWindow."""

    def __init__(self, page):
        super().__init__(None, QtCore.Qt.WindowType.Window)
        self.page = page
        self.setCentralWidget(page)


class DetachableTabWidget(QtWidgets.QTabWidget):
    """QTabWidget with detachable tabs (see the module documentation)."""

    #: (page, detached?) after a tab was detached or attached
    detached_changed = QtCore.Signal(QtWidgets.QWidget, bool)

    def __init__(self, parent=None):
        super().__init__(parent)
        bar = DragOutTabBar(self)
        self.setTabBar(bar)
        bar.detach_requested.connect(self._on_detach_requested)
        bar.setContextMenuPolicy(QtCore.Qt.ContextMenuPolicy.CustomContextMenu)
        bar.customContextMenuRequested.connect(self._context_menu)
        bar.tabMoved.connect(lambda _a, _b: self._sync_order())
        self._order = []          # every page in the order of insertion (the place to return to)
        self._info = {}           # page -> (title, icon, tooltip)
        self.detached = {}        # page -> window
        self._saved = {}          # page -> (menu bar visible, corner widget) of a QMainWindow page
        self._closing = False
        self._buttons = {}        # page -> its Attach button (Python must keep a reference)
        self._marks = {}          # page -> wordmark segments of its tab label

    # ---------------------------------------------------------------- bookkeeping
    def tabInserted(self, index):
        super().tabInserted(index)
        w = self.widget(index)
        if w not in self._order:
            before = self.widget(index + 1) if index + 1 < self.count() else None
            self._order.insert(self._order.index(before) if before in self._order else len(self._order), w)
        self._info[w] = (self.tabText(index), self.tabIcon(index), self.tabToolTip(index))
        if w in self._marks:
            self.tabBar().setTabData(index, self._marks[w])

    def set_wordmark(self, index_or_page, segments):
        """Draw the tab label as coloured segments ``[(text, colour on light, colour on dark,
        bold), ...]`` (None: plain text again). The tab's text stays its name."""
        page = self.widget(index_or_page) if isinstance(index_or_page, int) else index_or_page
        if segments:
            self._marks[page] = [tuple(s) for s in segments]
        else:
            self._marks.pop(page, None)
        i = self.indexOf(page)
        if i >= 0:
            self.tabBar().setTabData(i, self._marks.get(page))
            self.tabBar().updateGeometry()
            self.tabBar().update()

    def _sync_order(self):
        """After tabs were reordered by dragging: keep that order, detached pages keep their slots."""
        attached = iter([self.widget(i) for i in range(self.count())])
        self._order = [p if p in self.detached else next(attached) for p in self._order]

    def pages(self):
        """All pages, attached and detached, in their tab order."""
        return list(self._order)

    def title_of(self, page):
        return self._info.get(page, ("", None, ""))[0]

    def page_for(self, title):
        for p in self._order:
            if self.title_of(p) == title:
                return p
        return None

    def is_detached(self, page):
        return page in self.detached

    def show_page(self, page):
        """Bring ``page`` to the front: select its tab, or raise its window."""
        if page in self.detached:
            win = self.detached[page]
            win.showNormal() if win.isMinimized() else win.show()
            win.raise_()
            win.activateWindow()
        elif self.indexOf(page) >= 0:
            self.setCurrentWidget(page)

    # ---------------------------------------------------------------- detach / attach
    def _on_detach_requested(self, index, gpos):
        self.detach(index, pos=None if gpos.isNull() else gpos)

    def detach(self, index_or_page, pos=None, geometry=None):
        """Move a tab (index or page widget) into its own window; returns the window."""
        page = self.widget(index_or_page) if isinstance(index_or_page, int) else index_or_page
        if page is None:
            return None
        if page in self.detached:
            self.show_page(page)
            return self.detached[page]
        idx = self.indexOf(page)
        if idx < 0:
            return None
        self._info[page] = (self.tabText(idx), self.tabIcon(idx), self.tabToolTip(idx))
        title, icon, _tip = self._info[page]
        size = self.currentWidget().size() if self.currentWidget() is not None else QtCore.QSize(1000, 700)
        self.removeTab(idx)
        if isinstance(page, QtWidgets.QMainWindow):
            win = page
            mb = page.menuBar()
            self._saved[page] = (not mb.isHidden(), mb.cornerWidget(QtCore.Qt.Corner.TopRightCorner))
            page.setParent(None)
            page.setWindowFlags(QtCore.Qt.WindowType.Window)
            mb.setVisible(True)
        else:
            win = _PageWindow(page)
            page.show()
        btn = QtWidgets.QToolButton()
        btn.setText("Attach")
        btn.setIcon(self.style().standardIcon(QtWidgets.QStyle.StandardPixmap.SP_TitleBarNormalButton))
        btn.setToolButtonStyle(QtCore.Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        btn.setAutoRaise(True)
        btn.setToolTip("Put this window back as a tab (closing the window does the same)")
        btn.clicked.connect(lambda _c=False, p=page: self.attach(p))
        win.menuBar().setCornerWidget(btn, QtCore.Qt.Corner.TopRightCorner)
        self._buttons[page] = btn
        btn.show()
        owner = self.window().windowTitle() if self.window() is not None else ""
        win.setWindowTitle(f"{title} – {owner}" if owner and owner not in title else title)
        if icon is not None and not icon.isNull():
            win.setWindowIcon(icon)
        win.installEventFilter(self)
        self.detached[page] = win
        if geometry:
            win.setGeometry(*[int(v) for v in geometry])
        else:
            win.resize(max(size.width(), 500), max(size.height(), 350))
            if pos is not None:
                win.move(pos - QtCore.QPoint(60, 15))
            else:
                win.move(self.mapToGlobal(QtCore.QPoint(40, 40)))
        win.show()
        win.raise_()
        self.detached_changed.emit(page, True)
        return win

    def attach(self, page):
        """Put a detached page back as a tab at its original place."""
        win = self.detached.pop(page, None)
        if win is None:
            return
        win.removeEventFilter(self)
        self._buttons.pop(page, None)
        title, icon, tip = self._info[page]
        if win is page:
            vis, corner = self._saved.pop(page, (False, None))
            old = page.menuBar().cornerWidget(QtCore.Qt.Corner.TopRightCorner)
            page.menuBar().setCornerWidget(corner, QtCore.Qt.Corner.TopRightCorner)
            if old is not None and old is not corner:
                old.deleteLater()
            page.hide()
            page.setWindowFlags(QtCore.Qt.WindowType.Widget)
            page.menuBar().setVisible(vis)
        else:
            win.takeCentralWidget()
            win.hide()
            win.deleteLater()
        pos = sum(1 for p in self._order[:self._order.index(page)] if p not in self.detached)
        self.insertTab(pos, page, icon if icon is not None else QtGui.QIcon(), title)
        if tip:
            self.setTabToolTip(pos, tip)
        page.show()
        self.setCurrentWidget(page)
        self.detached_changed.emit(page, False)

    def attach_all(self):
        for p in list(self.detached):
            self.attach(p)

    def close_detached(self):
        """Hide the detached windows without attaching them (the owner window closes)."""
        self._closing = True
        for w in self.detached.values():
            w.hide()

    def eventFilter(self, obj, ev):
        if ev.type() == QtCore.QEvent.Type.Close and not self._closing:
            for page, win in list(self.detached.items()):
                if win is obj:
                    ev.ignore()
                    self.attach(page)
                    return True
        return super().eventFilter(obj, ev)

    # ---------------------------------------------------------------- context menu
    def _context_menu(self, pos):
        m = QtWidgets.QMenu(self)
        idx = self.tabBar().tabAt(pos)
        if idx >= 0:
            m.addAction(f"Detach '{self.tabText(idx)}' into a window", lambda i=idx: self.detach(i))
        if self.detached:
            m.addSeparator()
            for p in self.detached:
                m.addAction(f"Attach '{self.title_of(p)}'", lambda pp=p: self.attach(pp))
            m.addAction("Attach all", self.attach_all)
        if not m.isEmpty():
            m.exec(self.tabBar().mapToGlobal(pos))

    # ---------------------------------------------------------------- persistence
    def state(self):
        """{tab title: [x, y, width, height]} of the detached tabs."""
        out = {}
        for p, w in self.detached.items():
            g = w.geometry()
            out[self.title_of(p)] = [g.x(), g.y(), g.width(), g.height()]
        return out

    def restore_state(self, state):
        """Detach the tabs listed by :meth:`state` (unknown titles are ignored)."""
        for title, geo in (state or {}).items():
            p = self.page_for(title)
            if p is not None and self.indexOf(p) >= 0:
                self.detach(p, geometry=geo if isinstance(geo, list) and len(geo) == 4 else None)
