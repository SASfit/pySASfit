# -*- coding: utf-8 -*-
"""Tab bars with a clearly marked current tab: the selected tab gets a light grey background, a
coloured edge on the side facing its page and bold text; the other tabs stay light. Applied to the
whole application (launcher tabs, tool tabs, pyTOOLbox steps, tabbed docks such as I(q) / Log).

    apply_tab_style(app)
"""
from __future__ import annotations

__all__ = ["tab_style_sheet", "apply_tab_style"]

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
    app.setStyleSheet(sheet + "\n" + tab_style_sheet(dark))
