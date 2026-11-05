# -*- coding: utf-8 -*-
"""Tab bars with a clearly marked current tab: the selected tab gets a light grey background, a
coloured edge on the side facing its page and bold text; the other tabs stay light. Applied to the
whole application (launcher tabs, tool tabs, pyTOOLbox steps, tabbed docks such as I(q) / Log).
Scroll bars are always drawn at full width with a visible handle (not only while the mouse is over them).

    apply_tab_style(app)
"""
from __future__ import annotations

__all__ = ["tab_style_sheet", "scrollbar_style_sheet", "apply_tab_style"]

_MARK = "/* pysasfit-tab-style */"


def tab_style_sheet(dark: bool = False) -> str:
    if dark:
        normal, selected, hover, border, text, accent = "#2b2b2b", "#4a4a4a", "#3a3a3a", "#555555", "#e8e8e8", "#5ea0e6"
    else:
        normal, selected, hover, border, text, accent = "#ffffff", "#d9d9d9", "#eeeeee", "#c4c4c4", "#202020", "#2a6fb0"
    return f"""{_MARK}
QTabBar::tab {{
    background: {normal}; color: {text}; border: 1px solid {border};
    padding: 4px 12px; margin: 0px 1px 0px 0px;
}}
QTabBar::tab:hover:!selected {{ background: {hover}; }}
QTabBar::tab:selected {{ background: {selected}; font-weight: bold; }}
QTabBar::tab:top {{ border-bottom: none; border-top-left-radius: 3px; border-top-right-radius: 3px; }}
QTabBar::tab:top:selected {{ border-top: 3px solid {accent}; }}
QTabBar::tab:bottom {{ border-top: none; border-bottom-left-radius: 3px; border-bottom-right-radius: 3px; }}
QTabBar::tab:bottom:selected {{ border-bottom: 3px solid {accent}; }}
QTabBar::tab:left:selected {{ border-left: 3px solid {accent}; }}
QTabBar::tab:right:selected {{ border-right: 3px solid {accent}; }}
"""


def scrollbar_style_sheet(dark: bool = False) -> str:
    """Scroll bars always drawn at full width with a visible handle (the Windows 11 style shows only a thin
    line until the mouse is over the bar, which makes it hard to find)."""
    if dark:
        groove, handle, hover, border = "#2b2b2b", "#6a6a6a", "#8a8a8a", "#3a3a3a"
    else:
        groove, handle, hover, border = "#eeeeee", "#a6a6a6", "#7f7f7f", "#d6d6d6"
    return f"""
QScrollBar:vertical {{ width: 13px; background: {groove}; border-left: 1px solid {border}; margin: 0px; }}
QScrollBar:horizontal {{ height: 13px; background: {groove}; border-top: 1px solid {border}; margin: 0px; }}
QScrollBar::handle:vertical {{ background: {handle}; min-height: 28px; margin: 2px; border-radius: 4px; }}
QScrollBar::handle:horizontal {{ background: {handle}; min-width: 28px; margin: 2px; border-radius: 4px; }}
QScrollBar::handle:hover, QScrollBar::handle:pressed {{ background: {hover}; }}
QScrollBar::add-line, QScrollBar::sub-line {{ width: 0px; height: 0px; border: none; background: none; }}
QScrollBar::add-page, QScrollBar::sub-page {{ background: none; }}
"""


def _no_transient_scrollbars(app) -> None:
    """Styles with overlay scroll bars (macOS, some Linux styles) only show them while scrolling: switch
    that off for the whole application."""
    from PySide6 import QtWidgets
    if getattr(app, "_pysasfit_proxy_style", None) is not None:
        return

    class _Style(QtWidgets.QProxyStyle):
        def styleHint(self, hint, option=None, widget=None, returnData=None):
            if hint == QtWidgets.QStyle.StyleHint.SH_ScrollBar_Transient:
                return 0
            return super().styleHint(hint, option, widget, returnData)

    base = app.style()
    try:
        if base.styleHint(QtWidgets.QStyle.StyleHint.SH_ScrollBar_Transient) == 0:
            return                                   # not an overlay style: nothing to do
        st = _Style(base.name())
        app.setStyle(st)
        app._pysasfit_proxy_style = st
    except Exception:  # noqa: BLE001 -- a cosmetic setting
        pass


def apply_tab_style(app=None) -> None:
    """Add the tab style to the application's style sheet (once; light or dark by the palette)."""
    from PySide6 import QtGui, QtWidgets
    app = app or QtWidgets.QApplication.instance()
    if app is None:
        return
    sheet = app.styleSheet() or ""
    if _MARK in sheet:
        return
    dark = app.palette().color(QtGui.QPalette.ColorRole.Window).lightness() < 128
    _no_transient_scrollbars(app)
    app.setStyleSheet(sheet + "\n" + tab_style_sheet(dark) + scrollbar_style_sheet(dark))
