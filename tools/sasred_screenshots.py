r"""Make the screenshots of the pySASred and pyTOOLbox pages (docs/_static/sasred/*.png, docs/_static/toolbox/*.png).

Needs PySide6, matplotlib, h5py and SANS-I raw data (the Vifor runs of September 2026 were used):

    python tools/sasred_screenshots.py <raw-data folder>
    python tools/sasred_screenshots.py <raw-data folder> --only window mask_cartesian
    QT_QPA_PLATFORM=windows python tools/sasred_screenshots.py <folder>    # real fonts and style

In a bash shell (MSYS2, Git Bash) write Windows paths with forward slashes or in quotes:
C:/Users/me/SANS/raw or "C:\Users\me\SANS\raw" -- unquoted, bash removes the backslashes.

Message boxes and dialogs are answered automatically (their text is printed), so the script never waits
for a click; after --timeout seconds (default 600) it stops and prints where it was.

The pictures are rendered by Qt (``QWidget.grab``), no display is needed. Re-run after a change of the GUI;
the pages refer to these names:

    sasred/window          the whole pySASred window: runs, reduction panel, detector, I(q)
    sasred/mask_cartesian  detector panel after Suggest plus a drawn rectangle (cartesian view)
    sasred/mask_polar      the same in the polar (phi, q) view
    sasred/colour_scales   the detector image with linear, log(1+I), symlog and asinh
    sasred/mask_display    masked pixels in the mask colours, on their own grey scale, hidden (first-guess square)
    sasred/iq_live         I(q): a reduced run and the live curve of the selected run
    sasred/budget          error budget of a reduced run
    toolbox/viewer          pyTOOLbox HDF5 viewer with the detector counts of a raw file
    toolbox/lister          pyTOOLbox metadata lister with a listing of a run range
    launcher_tabs          the tab bar of the launcher (tool wordmarks)
"""

import argparse
import faulthandler
import os
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "src"))
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("MPLBACKEND", "QtAgg")
# factory preferences and no session of the user
os.environ["PYSASFIT_SETTINGS"] = os.path.join(tempfile.mkdtemp(prefix="pysasfit-shots-"), "settings.json")

from PySide6 import QtCore, QtGui, QtWidgets  # noqa: E402

OUT = os.path.join(ROOT, "docs", "_static")
NAMES = ["window", "mask_cartesian", "mask_polar", "colour_scales", "mask_display", "iq_live", "budget", "viewer",
         "lister", "launcher_tabs"]
DIRECT_BEAM, SAMPLE, SAMPLE2 = 82173, 82140, 82131      # Vifor, 8 m, 0.6 nm


def _step(text):
    print("...", text, flush=True)


def _no_modal_dialogs():
    """Offscreen there is nobody to click: answer every message box / dialog at once and print it."""
    MB = QtWidgets.QMessageBox

    def _box(kind):
        def f(parent, title, text, *a, **k):
            print(f"[{kind}] {title}: {text}", flush=True)
            return MB.StandardButton.Ok if kind != "question" else MB.StandardButton.No
        return staticmethod(f)

    for kind in ("information", "warning", "critical", "question"):
        setattr(MB, kind, _box(kind))

    def _exec(self, *a, **k):
        text = self.text() if isinstance(self, MB) else ""
        print(f"[dialog skipped] {type(self).__name__} {self.windowTitle()!r} {text}", flush=True)
        return 0

    QtWidgets.QDialog.exec = _exec
    QtWidgets.QDialog.exec_ = _exec
    for name in ("getOpenFileName", "getOpenFileNames", "getSaveFileName", "getExistingDirectory"):
        setattr(QtWidgets.QFileDialog, name,
                staticmethod(lambda *a, _n=name, **k: (print(f"[file dialog skipped] {_n}", flush=True),
                                                       ("" if _n == "getExistingDirectory" else
                                                        ([], "") if _n == "getOpenFileNames" else ("", "")))[1]))


def _settle(app, canvas=None):
    for _ in range(6):
        app.processEvents()
    if canvas is not None:
        canvas.draw()
        app.processEvents()


def _save(pix, sub, name):
    d = os.path.join(OUT, sub) if sub else OUT
    os.makedirs(d, exist_ok=True)
    p = os.path.join(d, name + ".png")
    pix.save(p)
    print("wrote", p)


def _side_by_side(pixmaps):
    w = sum(p.width() for p in pixmaps)
    h = max(p.height() for p in pixmaps)
    out = QtGui.QPixmap(w, h)
    out.fill(QtCore.Qt.GlobalColor.white)
    painter = QtGui.QPainter(out)
    x = 0
    for p in pixmaps:
        painter.drawPixmap(x, 0, p)
        x += p.width()
    painter.end()
    return out


def sasred_shots(app, raw, only):
    from pysasfit.sasred.core import mask as mk
    from pysasfit.sasred.gui import MainWindow
    _step("pySASred: opening the window")
    w = MainWindow()
    w.resize(1500, 900)
    w.show()
    _step(f"pySASred: reading the run table of {raw}")
    w.set_folder(raw)
    _step(f"pySASred: beam centre from run {DIRECT_BEAM}")
    w.select_runs([DIRECT_BEAM])
    w.on_beam_centre()
    _step(f"pySASred: reducing run {SAMPLE2}")
    w.select_runs([SAMPLE2])
    w.on_reduce()
    _step(f"pySASred: mask of run {SAMPLE}")
    w.select_runs([SAMPLE])
    if "mask_display" in only:                           # the untouched start mask: first-guess square
        w.resizeDocks([w.docks["detector"], w.docks["iq"]], [560, 760], QtCore.Qt.Orientation.Horizontal)
        ed = w.mask_editor
        pics = []
        for mode in ("colours", "scale", "hidden"):
            ed.display.setCurrentIndex(ed.display.findData(mode))
            _settle(app, w.det_canvas)
            pics.append(w.docks["detector"].widget().grab())
        _save(_side_by_side(pics), "sasred", "mask_display")
        ed.display.setCurrentIndex(ed.display.findData("colours"))
    w.mask_editor.suggest()
    ed = w.mask_editor
    ed.apply(mk.pixels_in_rectangle(ed.mask.shape, 100, 110, 20, 30), True)
    w.update_preview()
    _settle(app, w.det_canvas)
    if "window" in only:
        _save(w.grab(), "sasred", "window")
    w.resizeDocks([w.docks["detector"], w.docks["iq"]], [720, 600], QtCore.Qt.Orientation.Horizontal)
    _settle(app, w.det_canvas)
    if "mask_cartesian" in only:
        _save(w.docks["detector"].widget().grab(), "sasred", "mask_cartesian")
    if "mask_polar" in only:
        w.det_view.setCurrentIndex(w.det_view.findData("polar"))
        _settle(app, w.det_canvas)
        _save(w.docks["detector"].widget().grab(), "sasred", "mask_polar")
        w.det_view.setCurrentIndex(w.det_view.findData("cartesian"))
    if "colour_scales" in only:
        w.mask_editor.display.setCurrentIndex(w.mask_editor.display.findData("hidden"))
        w.resizeDocks([w.docks["detector"], w.docks["iq"]], [520, 800], QtCore.Qt.Orientation.Horizontal)
        pics = []
        for key in ("linear", "log1p", "symlog", "asinh"):
            w.det_scale.setCurrentIndex(w.det_scale.findData(key))
            _settle(app, w.det_canvas)
            pics.append(w.docks["detector"].widget().grab())
        _save(_side_by_side(pics), "sasred", "colour_scales")
        w.det_scale.setCurrentIndex(w.det_scale.findData("log1p"))
        w.mask_editor.display.setCurrentIndex(w.mask_editor.display.findData("colours"))
    w.resizeDocks([w.docks["detector"], w.docks["iq"]], [500, 800], QtCore.Qt.Orientation.Horizontal)
    if "iq_live" in only:
        w.docks["iq"].raise_()
        _settle(app, w.iq_canvas)
        _save(w.docks["iq"].widget().grab(), "sasred", "iq_live")
    if "budget" in only:
        w.result_combo.setCurrentIndex(0)
        w.docks["budget"].raise_()
        _settle(app, w.budget_canvas)
        _save(w.docks["budget"].widget().grab(), "sasred", "budget")
    w.close()


def toolbox_shots(app, raw, only):
    from pysasfit.toolbox.gui import MainWindow
    _step("pyTOOLbox")
    w = MainWindow()
    w.resize(1400, 820)
    w.show()
    path = os.path.join(raw, f"sans2026n{SAMPLE:06d}.hdf")
    if "viewer" in only:
        w.viewer.select_path(path, "/entry1/SANS/detector/counts")
        _settle(app, w.viewer.canvas)
        _save(w.grab(), "toolbox", "viewer")
    if "lister" in only:
        L = w.lister
        L.folder_edit.setText(raw)
        L.instr_edit.setText("sans")
        L.year_edit.setText("2026")
        L.from_edit.setText(str(SAMPLE - 12))
        L.to_edit.setText(str(SAMPLE + 2))
        L.scan()
        wanted = ("/entry1/sample/name", "/entry1/SANS/detector/counting_time", "/entry1/SANS/monitor1/counts",
                  "/entry1/SANS/detector/distance", "/entry1/SANS/Dornier-VS/lambda", "/entry1/SANS/attenuator/selection",
                  "/entry1/sample/thickness")
        for k in L.entry_state:
            L.entry_state[k] = k in wanted
        L.rebuild_entry_list()
        L.generate()
        w.tabs.setCurrentWidget(L)
        _settle(app)
        _save(w.grab(), "toolbox", "lister")
    w.lister.pending.clear()
    w.close()


def launcher_shot(app):
    from pysasfit.gui.launcher import Launcher
    _step("launcher tab bar (loads all tools)")
    w = Launcher(tools=[("pySASred", "pysasfit.sasred.gui", "MainWindow"),
                        ("pySASstudio", "pysasfit.studio.explorer", "ModelExplorer"),
                        ("pySLDprofile", "pysasfit.sldprofile.gui", "MainWindow"),
                        ("pyINVersion", "pysasfit.inversion.gui.app", "MainWindow"),
                        ("pyOZsolver", "pysasfit.oz.gui.main_window", "OZWidget"),
                        ("pyTOOLbox", "pysasfit.toolbox.gui", "MainWindow")])
    w.resize(1000, 400)
    w.show()
    _settle(app)
    _save(w.grab().copy(0, 0, 1000, 64), "", "launcher_tabs")
    w.tabs.close_detached()
    w.close()


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("raw", help="folder with the SANS-I raw files (sans2026n0821xx.hdf)")
    ap.add_argument("--only", nargs="*", choices=NAMES, default=NAMES)
    ap.add_argument("--timeout", type=float, default=600, help="seconds before giving up (default 600)")
    a = ap.parse_args(argv)
    raw = os.path.abspath(a.raw)
    needed = [os.path.join(raw, f"sans2026n{r:06d}.hdf") for r in (DIRECT_BEAM, SAMPLE, SAMPLE2)]
    missing = [p for p in needed if not os.path.isfile(p)]
    if missing:
        hint = ""
        if ":" in a.raw and "\\" not in a.raw and "/" not in a.raw:
            hint = ("\nThe path has no separators left: in bash write C:/Users/... or put the path in quotes.")
        sys.exit(f"raw-data folder {a.raw!r}: missing " + ", ".join(os.path.basename(p) for p in missing) + hint)
    a.raw = raw
    faulthandler.dump_traceback_later(a.timeout, exit=True)    # never hang silently
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    _no_modal_dialogs()
    only = set(a.only)
    if only & {"window", "mask_cartesian", "mask_polar", "colour_scales", "mask_display", "iq_live", "budget"}:
        sasred_shots(app, a.raw, only)
    if only & {"viewer", "lister"}:
        toolbox_shots(app, a.raw, only)
    if "launcher_tabs" in only:
        launcher_shot(app)
    faulthandler.cancel_dump_traceback_later()
    _step("done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
