# -*- coding: utf-8 -*-
"""Project / last-session support shared by the Qt main windows.

A window that mixes in :class:`ProjectMixin` implements five small hooks:

``_project_columns()``      data as read: dict q, I, sigma[, dq] (or None)
``_project_data_path()``    path of the loaded data file (or None)
``_project_load_data(p)``   load a data file (the normal "Load" action)
``_project_result()``       picklable result state (or None)
``_project_set_result(r)``  show a restored result state

and calls :meth:`_init_project` at the end of ``__init__`` (after all
controls exist) and :meth:`_autosave` after loading data and after a fit.
See :mod:`pysasfit.core.project` for the file format.
"""

from __future__ import annotations

import os

from PySide6 import QtGui, QtWidgets

from . import project as prj


class ProjectMixin:
    _project_tool = None          # "sldprofile" | "inversion"
    _project_exclude = ()         # control names not stored

    # ---------------------------------------------------------------- set-up
    def _project_extra_widgets(self) -> dict:
        """Controls that are not plain attributes (e.g. widgets kept in a dict)."""
        return {}

    def _project_clear(self) -> None:
        """Forget the current result before a project is opened (tool hook)."""

    def _project_note(self, msg: str) -> None:
        try:
            self.statusBar().showMessage(msg, 10000)
        except Exception:  # noqa: BLE001
            pass

    def _init_project(self, menubar=None):
        self._restoring = False
        self.project_path = None
        self._default_controls = prj.control_state(self, self._project_extra_widgets(), self._project_exclude)
        mb = menubar if menubar is not None else self.menuBar()
        file_menu = None
        for a in mb.actions():
            if a.menu() is not None and a.text().replace("&", "") == "File":
                file_menu = a.menu()
        if file_menu is None:
            file_menu = QtWidgets.QMenu("&File", self)
            first = mb.actions()[0] if mb.actions() else None
            mb.insertMenu(first, file_menu) if first is not None else mb.addMenu(file_menu)
        self.open_project_action = QtGui.QAction("&Open project…", self)
        self.open_project_action.setShortcut(QtGui.QKeySequence.StandardKey.Open)
        self.open_project_action.triggered.connect(self.on_open_project)
        self.save_project_action = QtGui.QAction("&Save project…", self)
        self.save_project_action.setShortcut(QtGui.QKeySequence.StandardKey.Save)
        self.save_project_action.triggered.connect(self.on_save_project)
        file_menu.addAction(self.open_project_action)
        file_menu.addAction(self.save_project_action)
        settings_menu = None
        for a in mb.actions():
            if a.menu() is not None and a.text().replace("&", "") == "Settings":
                settings_menu = a.menu()
        if settings_menu is not None:
            self.forget_session_action = QtGui.QAction("Forget last session", self)
            self.forget_session_action.setToolTip("Delete the automatically saved last session of this tool; "
                                                  "the next start begins with default panels and no data")
            self.forget_session_action.triggered.connect(self.forget_session)
            settings_menu.addAction(self.forget_session_action)

    # ---------------------------------------------------------------- state
    def _project_save(self, path):
        cols = self._project_columns()
        prj.save_project(path, self._project_tool, data_path=self._project_data_path(), columns=cols,
                         controls=prj.control_state(self, self._project_extra_widgets(), self._project_exclude),
                         extra=self._project_extra(), result=self._project_result())

    def _project_extra(self) -> dict:
        """Plain values stored with the project: the window layout (detached panels)."""
        ds = getattr(self, "dock_state", None)
        return {"dock_state": ds()} if callable(ds) else {}

    def _project_open(self, path, session=False):
        """Restore a project file; returns the list of notes."""
        m = prj.load_project(path, self._project_tool)
        notes = list(m["notes"])
        extra = self._project_extra_widgets()
        self._restoring = True
        try:
            self._project_clear()
            # 1. factory state: controls the file does not mention keep their defaults
            prj.restore_controls(self, {k: {"kind": kd, "value": v}
                                        for k, (kd, v) in self._default_controls.items()}, extra)
            # 2. data: the original file if unchanged, else the embedded copy
            dp, sig, cols = m["data"].get("path"), m["data"].get("signature"), m["columns"]
            use, keep_result = None, True
            if dp and os.path.isfile(dp) and (sig is None or prj.file_signature(dp) == sig):
                use = dp
            elif cols:
                base = (os.path.splitext(prj.session_path(self._project_tool))[0] if session
                        else os.path.splitext(path)[0])
                name = os.path.splitext(os.path.basename(dp))[0] if dp else "data"
                use = prj.write_data_file(f"{base}_{name}.dat", cols)
                notes.append(("data file changed since the project was saved" if dp and os.path.isfile(dp)
                              else "data file not found") + f"; using the copy stored in the project: {use}")
            elif dp and os.path.isfile(dp):
                use, keep_result = dp, False
                notes.append("data file changed since the project was saved; the stored result is not shown")
            elif dp:
                notes.append(f"data file {dp} not found")
            if use:
                self._project_load_data(use)
            # 3. panel controls
            skipped = prj.restore_controls(self, m["controls"], extra)
            if skipped:
                notes.append(f"{len(skipped)} stored settings not used: {', '.join(sorted(set(skipped))[:8])}")
            # window layout (detached / moved panels): only for the last session --
            # a project opened on another screen keeps the current arrangement
            if session and hasattr(self, "restore_dock_state"):
                self.restore_dock_state(m["extra"].get("dock_state"))
            # 4. result
            if use and keep_result and m["result"] is not None:
                try:
                    self._project_set_result(m["result"])
                except Exception as e:  # noqa: BLE001
                    notes.append(f"the stored result could not be shown ({type(e).__name__}: {e})")
        finally:
            self._restoring = False
        if not session:
            self.project_path = path
        return notes

    # ---------------------------------------------------------------- actions
    def on_save_project(self):
        start = self.project_path or (os.path.splitext(self._project_data_path() or "project")[0] + prj.EXTENSION)
        fn, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save project", start,
                                                      f"pySASfit project (*{prj.EXTENSION})")
        if not fn:
            return
        if not fn.endswith(prj.EXTENSION):
            fn += prj.EXTENSION
        self.save_project_to(fn)

    def save_project_to(self, fn):
        try:
            self._project_save(fn)
        except Exception as e:  # noqa: BLE001
            QtWidgets.QMessageBox.critical(self, "Save project", str(e))
            return False
        self.project_path = fn
        self._project_note(f"project saved: {fn}")
        return True

    def on_open_project(self):
        fn, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Open project", os.path.dirname(self.project_path or ""),
                                                      f"pySASfit project (*{prj.EXTENSION});;All files (*)")
        if fn:
            self.open_project(fn)

    def open_project(self, fn):
        try:
            notes = self._project_open(fn)
        except prj.ProjectError as e:
            QtWidgets.QMessageBox.critical(self, "Open project", str(e))
            return False
        self._project_note(f"project opened: {fn}")
        for n in notes:
            self._project_note(n)
        self._autosave()
        return True

    # ---------------------------------------------------------------- last session
    def _autosave(self, on_close=False):
        if getattr(self, "_restoring", False) or self._project_tool is None:
            return
        if on_close and getattr(self, "_session_forgotten", False):
            return                       # "Forget last session" holds until the next load or fit
        self._session_forgotten = False
        try:
            self._project_save(prj.session_path(self._project_tool))
        except Exception:  # noqa: BLE001 -- never disturb the GUI
            pass

    def restore_session(self):
        """Reopen the last session (no-op when there is none). Returns True if restored."""
        p = prj.session_path(self._project_tool)
        if not os.path.isfile(p):
            return False
        try:
            notes = self._project_open(p, session=True)
        except Exception as e:  # noqa: BLE001
            self._project_note(f"last session not restored ({e})")
            return False
        self._project_note("last session restored (Settings → Forget last session to start empty)")
        for n in notes:
            self._project_note(n)
        return True

    def forget_session(self):
        prj.forget_session(self._project_tool)
        self._session_forgotten = True
        self._project_note("last session deleted; a new one is saved only after the next data load or fit")

    def shutdown(self):
        """Called by the launcher when it closes."""
        self._autosave(on_close=True)

    def closeEvent(self, event):
        self._autosave(on_close=True)
        super().closeEvent(event)
