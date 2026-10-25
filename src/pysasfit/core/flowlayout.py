# -*- coding: utf-8 -*-
"""Layout that places its widgets in a row and wraps them to further rows when the space
is too narrow (like words in a paragraph). Control bars above plots use it, so that they
never force a minimum width onto their panel: a narrow panel just gets two or three
rows of controls.
"""

from __future__ import annotations

from PySide6 import QtCore, QtWidgets

__all__ = ["FlowLayout", "flow_bar"]


class FlowLayout(QtWidgets.QLayout):
    def __init__(self, parent=None, spacing=4):
        super().__init__(parent)
        self._items = []
        self.setContentsMargins(0, 0, 0, 0)
        self.setSpacing(spacing)

    def addItem(self, item):
        self._items.append(item)

    def count(self):
        return len(self._items)

    def itemAt(self, i):
        return self._items[i] if 0 <= i < len(self._items) else None

    def takeAt(self, i):
        return self._items.pop(i) if 0 <= i < len(self._items) else None

    def expandingDirections(self):
        return QtCore.Qt.Orientation(0)

    def hasHeightForWidth(self):
        return True

    def heightForWidth(self, width):
        return self._do_layout(QtCore.QRect(0, 0, width, 0), test=True)

    def setGeometry(self, rect):
        super().setGeometry(rect)
        self._do_layout(rect, test=False)

    def sizeHint(self):
        return self.minimumSize()

    def minimumSize(self):
        s = QtCore.QSize()
        for it in self._items:
            s = s.expandedTo(it.minimumSize())
        m = self.contentsMargins()
        return s + QtCore.QSize(m.left() + m.right(), m.top() + m.bottom())

    def _do_layout(self, rect, test):
        m = self.contentsMargins()
        r = rect.adjusted(m.left(), m.top(), -m.right(), -m.bottom())
        x, y, line_h = r.x(), r.y(), 0
        sp = self.spacing()
        for it in self._items:
            hint = it.sizeHint()
            nx = x + hint.width() + sp
            if nx - sp > r.right() + 1 and line_h > 0:
                x = r.x()
                y = y + line_h + sp
                nx = x + hint.width() + sp
                line_h = 0
            if not test:
                it.setGeometry(QtCore.QRect(QtCore.QPoint(x, y), hint))
            x = nx
            line_h = max(line_h, hint.height())
        return y + line_h - rect.y() + m.bottom()


def flow_bar(*widgets, spacing=4):
    """Widget with ``widgets`` (QWidgets, or str for a label) in a wrapping row."""
    w = QtWidgets.QWidget()
    lay = FlowLayout(w, spacing)
    for x in widgets:
        lay.addWidget(QtWidgets.QLabel(x) if isinstance(x, str) else x)
    sp = w.sizePolicy()
    sp.setHeightForWidth(True)
    sp.setHorizontalPolicy(QtWidgets.QSizePolicy.Policy.Ignored)
    w.setSizePolicy(sp)
    return w
