"""Qt widget to choose a SASfit function from the cascading group menu, as in the SASfit GUI.

A button shows the chosen function; clicking it opens the tree of groups built by
:func:`pysasfit.studio.groups.build_menu` (submenus for the groups, the functions as entries).
"Find…" opens a search box over all functions (label, name and group path) for when you know
what you are looking for.

    picker = FunctionPicker(none_label="(none)")
    picker.set_menu(groups.build_menu(entries, ("sd",)))
    picker.changed.connect(lambda name: ...)       # '' when "(none)" was chosen
    picker.set_current("sd_LogNorm")               # silent; False if the function is not in the menu
"""

from __future__ import annotations

from PySide6 import QtCore, QtGui, QtWidgets

from .groups import iter_functions

__all__ = ["FunctionPicker", "esc"]

MAX_HITS = 300  # rows shown in the search list


def esc(text):
    """Escape '&' so that Qt does not read it as a menu mnemonic ("Spheres & Shells")."""
    return text.replace("&", "&&")


class _SearchPopup(QtWidgets.QFrame):
    """Popup with a search field and the list of matching functions."""

    chosen = QtCore.Signal(str)

    def __init__(self, parent):
        super().__init__(parent, QtCore.Qt.WindowType.Popup)
        self.setFrameShape(QtWidgets.QFrame.Shape.StyledPanel)
        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(4, 4, 4, 4)
        self.edit = QtWidgets.QLineEdit()
        self.edit.setPlaceholderText("search label, name or group …")
        self.edit.setClearButtonEnabled(True)
        self.list = QtWidgets.QListWidget()
        lay.addWidget(self.edit)
        lay.addWidget(self.list, 1)
        self.resize(580, 340)
        self._items = []  # (lower-case haystack, display text, name, tooltip)
        self.edit.textChanged.connect(self._filter)
        self.edit.returnPressed.connect(self._accept)
        self.list.itemActivated.connect(self._pick)
        self.edit.installEventFilter(self)

    def set_items(self, items):
        self._items = list(items)

    def popup_below(self, widget):
        self.edit.clear()
        self._filter("")
        self.move(widget.mapToGlobal(QtCore.QPoint(0, widget.height())))
        self.show()
        self.edit.setFocus()

    def _filter(self, text):
        words = text.lower().split()
        self.list.clear()
        for hay, display, name, tip in self._items:
            if all(w in hay for w in words):
                item = QtWidgets.QListWidgetItem(display)
                item.setData(QtCore.Qt.ItemDataRole.UserRole, name)
                item.setToolTip(tip)
                self.list.addItem(item)
                if self.list.count() >= MAX_HITS:
                    break
        if self.list.count():
            self.list.setCurrentRow(0)

    def _accept(self):
        item = self.list.currentItem()
        if item is not None:
            self._pick(item)

    def _pick(self, item):
        name = item.data(QtCore.Qt.ItemDataRole.UserRole)
        self.close()
        self.chosen.emit(name)

    def eventFilter(self, obj, event):
        if obj is self.edit and event.type() == QtCore.QEvent.Type.KeyPress:
            step = {QtCore.Qt.Key.Key_Down: 1, QtCore.Qt.Key.Key_Up: -1}.get(event.key())
            if step and self.list.count():
                row = min(max(self.list.currentRow() + step, 0), self.list.count() - 1)
                self.list.setCurrentRow(row)
                return True
        return super().eventFilter(obj, event)


class FunctionPicker(QtWidgets.QWidget):
    """Button with the cascading menu of function groups, plus a search popup."""

    changed = QtCore.Signal(str)  # name of the chosen function, '' for "(none)"

    def __init__(self, none_label=None, placeholder="(select a function)", parent=None):
        super().__init__(parent)
        self._none_label = none_label
        self._placeholder = placeholder
        self._name = None
        self._info = {}  # name -> (function, path)
        self._actions = {}  # name -> QAction ('' for none)
        self._actgroup = None

        lay = QtWidgets.QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(4)
        self.button = QtWidgets.QPushButton()
        self.button.setSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Fixed)
        self.button.setMinimumWidth(90)
        self.button.setStyleSheet("QPushButton { text-align: left; padding: 3px 22px 3px 8px; }")
        self.find = QtWidgets.QToolButton()
        self.find.setText("Find…")
        self.find.setToolTip("search all functions by label, name or group")
        lay.addWidget(self.button, 1)
        lay.addWidget(self.find)

        self._popup = _SearchPopup(self)
        self._popup.chosen.connect(self._choose)
        self.find.clicked.connect(lambda: self._popup.popup_below(self.button))
        self._update_button()

    # ---------------------------------------------------------------- menu
    def set_menu(self, top):
        """Show the tree ``top`` (from :func:`pysasfit.studio.groups.build_menu`); keeps the current function if present."""
        menu = QtWidgets.QMenu(self)
        menu.setToolTipsVisible(True)
        self._actgroup = QtGui.QActionGroup(menu)
        self._actgroup.setExclusive(True)
        self._actions, self._info = {}, {}
        if self._none_label:
            self._add_leaf(menu, "", self._none_label, "")
        self._fill(menu, top, ())
        items = []
        for path, p in iter_functions(top):
            where = " › ".join(path)
            self._info[p.name] = (p, where)
            display = f"{p.label}   [{p.name}]" + (f"   —   {where}" if where else "")
            items.append((f"{p.label} {p.name} {where}".lower(), display, p.name, f"{p.name}\n{p.brief}".strip()))
        self._popup.set_items(items)
        old = self.button.menu()
        self.button.setMenu(menu)
        if old is not None:
            old.deleteLater()
        if self._name is not None and self._name not in self._actions:
            self._name = None
        self._mark(self._name)
        self._update_button()

    def _fill(self, menu, group, path):
        for g in group.children:
            sub = menu.addMenu(esc(g.label))
            if g.brief:
                sub.menuAction().setToolTip(g.brief)
            self._fill(sub, g, path + (g.label,))
        for p in group.functions:
            self._add_leaf(menu, p.name, p.label, f"{p.name}\n{p.brief}".strip())

    def _add_leaf(self, menu, name, label, tip):
        act = menu.addAction(esc(label))
        act.setCheckable(True)
        act.setToolTip(tip)
        self._actgroup.addAction(act)
        act.triggered.connect(lambda _checked=False, n=name: self._choose(n))
        self._actions[name] = act

    # ------------------------------------------------------------ selection
    def current(self):
        """Name of the chosen function, or None."""
        return self._name

    def set_current(self, name):
        """Select ``name`` (None: nothing / "(none)") without emitting ``changed``; False if not in the menu."""
        name = name or None
        if name is not None and name not in self._actions:
            return False
        self._name = name
        self._mark(name)
        self._update_button()
        return True

    def _choose(self, name):
        changed = (name or None) != self._name
        if self.set_current(name) and changed:
            self.changed.emit(name or "")

    def _mark(self, name):
        act = self._actions.get(name or "")
        if act is not None:
            act.setChecked(True)
        elif self._actgroup is not None:
            checked = self._actgroup.checkedAction()
            if checked is not None:  # an exclusive group cannot uncheck its action
                self._actgroup.setExclusive(False)
                checked.setChecked(False)
                self._actgroup.setExclusive(True)

    # --------------------------------------------------------------- button
    def _update_button(self):
        if self._name is not None and self._name in self._info:
            p, where = self._info[self._name]
            text, tip = p.label, "\n".join(s for s in (p.name, where, p.brief) if s)
        elif self._name is not None:
            text, tip = self._name, self._name
        else:
            text, tip = self._none_label or self._placeholder, ""
        room = max(self.button.width() - 44, 20)
        self.button.setText(esc(self.button.fontMetrics().elidedText(text, QtCore.Qt.TextElideMode.ElideRight, room)))
        self.button.setToolTip(tip)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._update_button()
