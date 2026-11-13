"""Scroll bars instead of a minimum window size.

A window cannot be made smaller than the minimum size of its content, and a tab widget or a stack takes
the largest minimum of *all* its pages: one wide page (a long row of controls, a designer canvas) fixes
the smallest size of the whole pySASfit window. Content inside a scroll area does not count: the page
gets scroll bars when the window is smaller than it needs.

    scroll_inside(page)     # the content of ``page`` scrolls; ``page`` itself stays the tab page
    scrolled(widget)        # a scroll area around ``widget``

Used for every panel of the tools (:class:`pysasfit.core.docking.DockMixin`) and for the pages of the
tab widgets of pyTOOLbox and pyOZsolver.
"""

from __future__ import annotations

from PySide6 import QtCore, QtWidgets

__all__ = ["scrolled", "scroll_inside", "MIN_PAGE"]

#: smallest size a scrolling page asks for (below it the scroll bars leave no room for content)
MIN_PAGE = QtCore.QSize(120, 90)


class _Area(QtWidgets.QScrollArea):
    def minimumSizeHint(self):
        return QtCore.QSize(MIN_PAGE)


def scrolled(widget, frame=False):
    """A scroll area around ``widget`` (resized with the area; scroll bars when it gets smaller than the
    widget's minimum size)."""
    sc = _Area()
    sc.setWidgetResizable(True)
    if not frame:
        sc.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
    sc.setWidget(widget)
    return sc


def scroll_inside(page):
    """Put the content of ``page`` (a widget with a layout) into a scroll area inside it. ``page`` keeps its
    identity, so code that refers to it (tab pages, attributes) is not affected. Returns the scroll area
    (None when the page has no layout or scrolls already)."""
    lay = page.layout()
    if lay is None or getattr(page, "_scroll_area", None) is not None:
        return getattr(page, "_scroll_area", None)
    inner = QtWidgets.QWidget()
    inner.setObjectName("scroll_content")
    inner.setLayout(lay)                      # the layout and its widgets move to ``inner``
    sc = scrolled(inner)
    outer = QtWidgets.QVBoxLayout(page)
    outer.setContentsMargins(0, 0, 0, 0)
    outer.addWidget(sc)
    page._scroll_area = sc
    return sc
