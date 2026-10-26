"""pyTOOLbox window: HDF5 viewer, metadata lister, merging of reduced curves, 2D data.

    pysasfit-toolbox [file.hdf ...]
    python -m pysasfit.toolbox.gui [file.hdf ...]

Also the "pyTOOLbox" tab of the pySASfit launcher. Files given on the command line (or
dropped on the viewer) are opened in the HDF5 viewer; a double-click on a file name in
the metadata listing opens that file in the viewer. *Merge geometries* groups reduced curves
by sample name (regular expression) and header items and merges the detector distances
(:mod:`pysasfit.toolbox.merge`).
"""
from __future__ import annotations

import json
import os
import sys
import warnings

from PySide6 import QtWidgets

warnings.filterwarnings("ignore", message="constrained_layout not applied")

from ..core import settings as prefs  # noqa: E402
from ..core.settings_dialog import add_preferences_menu  # noqa: E402
from ..core.tabs import DetachableTabWidget  # noqa: E402
from ..resources import tool_icon  # noqa: E402
from .image_gui import ImageTool  # noqa: E402
from .merge_gui import MergeTool  # noqa: E402
from .meta_lister import MetaLister  # noqa: E402
from .viewer import HDF5Viewer  # noqa: E402

__all__ = ["MainWindow", "main"]


def _session_path():
    return os.path.join(os.path.dirname(prefs.settings_path()), "toolbox_session.json")


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("pyTOOLbox – HDF5 viewer, metadata lister, merge geometries, 2D data")
        self.setWindowIcon(tool_icon("pyTOOLbox"))
        self.tabs = DetachableTabWidget()        # viewer and lister can be separate windows
        self.viewer = HDF5Viewer()
        self.lister = MetaLister()
        self.tabs.addTab(self.viewer, "HDF5 viewer")
        self.tabs.addTab(self.lister, "Metadata lister")
        self.merger = MergeTool()
        self.tabs.addTab(self.merger, "Merge geometries")
        self.merger.status.connect(lambda t: self.statusBar().showMessage(t, 10000))
        self.images = ImageTool()
        self.tabs.addTab(self.images, "2D data")
        self.images.status.connect(lambda t: self.statusBar().showMessage(t, 10000))
        self.merger.images_requested.connect(self.show_images)
        self.setCentralWidget(self.tabs)
        self.viewer.status.connect(lambda t: self.statusBar().showMessage(t, 10000))
        self.lister.open_file_requested.connect(self.open_in_viewer)
        m = self.menuBar().addMenu("&File")
        m.addAction("Open HDF5 &files…", self.viewer.on_open)
        m.addAction("&Close HDF5 file", self.viewer.close_current_file)
        m.addSeparator()
        m.addAction("Add &reduced curves (merge)…", lambda: (self.tabs.show_page(self.merger),
                                                             self.merger.on_add_files()))
        add_preferences_menu(self, "toolbox")
        self.resize(1400, 900)

    def show_images(self, imgs):
        self.images.add_images(imgs)
        self.tabs.show_page(self.images)

    def open_in_viewer(self, path):
        self.viewer.open_file(path)
        self.tabs.show_page(self.viewer)

    # ------------------------------------------------------------------ session (launcher hooks)
    def restore_session(self):
        try:
            path = _session_path()
            if not os.path.isfile(path):         # session of the pySASbox era
                path = os.path.join(os.path.dirname(path), "sasbox_session.json")
            with open(path, encoding="utf-8") as fh:
                s = json.load(fh)
        except (OSError, ValueError):
            return False
        for p in s.get("files", []):
            if os.path.isfile(p):
                self.viewer.open_file(p, select=False)
        if s.get("current") and os.path.isfile(s["current"][0]):
            self.viewer.select_path(*s["current"])
        self.merger.restore_state(s.get("merge", {}))
        i = s.get("tab")
        if isinstance(i, int) and 0 <= i < self.tabs.count():
            self.tabs.setCurrentIndex(i)
        return True

    def save_session(self):
        try:
            p = _session_path()
            os.makedirs(os.path.dirname(p), exist_ok=True)
            with open(p, "w", encoding="utf-8") as fh:
                json.dump({"files": self.viewer.files, "current": self.viewer.current,
                           "tab": self.tabs.currentIndex(), "merge": self.merger.state()}, fh, indent=1)
        except OSError:
            pass

    def forget_session(self):
        try:
            os.remove(_session_path())
        except OSError:
            pass

    def may_close(self):
        """Asked by the launcher before it closes (edited values not yet written)."""
        return self.lister.may_close()

    def close_detached(self):
        self.tabs.close_detached()

    def shutdown(self):
        self.save_session()
        self.lister.persist()

    def closeEvent(self, event):
        if not self.lister.may_close():          # edited values not written yet
            event.ignore()
            return
        self.save_session()
        self.close_detached()
        super().closeEvent(event)


def main(argv=None):
    argv = sys.argv if argv is None else argv
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(argv)
    app.setWindowIcon(tool_icon("pyTOOLbox"))
    w = MainWindow()
    w.show()
    files = [a for a in argv[1:] if os.path.isfile(a)]
    if files:
        for f in files:
            w.viewer.open_file(f)
    else:
        w.restore_session()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
