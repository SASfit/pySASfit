"""Make the screenshots of the studio manual (docs/_static/studio/*.png).

Needs PySide6, matplotlib and the SASfit libraries (``pysasfit-studio status``). The pictures are rendered by Qt
itself (``QWidget.grab``), so no display is needed:

    python tools/studio_screenshots.py                 # all pictures
    python tools/studio_screenshots.py --only explorer_main fit_result
    QT_QPA_PLATFORM=windows python tools/studio_screenshots.py     # real fonts and style instead of 'offscreen'

Re-run it after a change of the GUI; the manual refers to these file names:

    explorer_main      the explorer: model dock, plot with data and fit, fit dock
    model_dock         the model dock with the two contributions
    parameter_rows     parameter rows with sliders, fit boxes and bounds
    fit_panel          the fit dock before the fit (parameters, algorithm, options)
    fit_result         the Result tab after a fit
    size_distribution  the size-distribution dock (N(x) x^p)
    summary_figure     the exported summary picture (also what "Copy summary image" puts on the clipboard)
"""

import argparse
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "src"))
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("MPLBACKEND", "QtAgg")

from PySide6 import QtCore, QtWidgets  # noqa: E402

from pysasfit.core import load_data  # noqa: E402
from pysasfit.studio import export  # noqa: E402
from pysasfit.studio.explorer import ModelExplorer  # noqa: E402

OUT = os.path.join(ROOT, "docs", "_static", "studio")
DATA = os.path.join(ROOT, "examples", "studio", "data", "SASfitTestData", "test.dat")
NAMES = ["explorer_main", "model_dock", "parameter_rows", "fit_panel", "fit_result", "size_distribution", "summary_figure"]


def wait(ms=600):
    """Let the explorer redraw (it waits a moment after every change) and the GUI settle."""
    end = time.time() + ms / 1000
    while time.time() < end:
        QtWidgets.QApplication.processEvents(QtCore.QEventLoop.AllEvents, 50)
        time.sleep(0.01)


def save(widget, name):
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, name + ".png")
    widget.grab().save(path)
    print("wrote", path)


def build():
    """The bimodal spheres of the SASfit test data, started about 30 % off the result."""
    ex = ModelExplorer("ff_sphere", sd="sd_LogNorm", extra=[("ff_sphere", "sd_LogNorm", None)], qmin=0.003, qmax=1.0)
    ex.resize(1500, 900)
    ex.show()
    wait()
    start = {"0.sd.N": 2.5e-12, "0.sd.s": 0.25, "0.sd.mu": 120, "1.sd.N": 1e-10, "1.sd.s": 0.25, "1.sd.mu": 55}
    for address, value in start.items():
        ex._row(address).set_value(value)
    for address in ("0.ff.eta", "1.ff.eta"):
        ex._row(address).set_value(1.0)
    ex._row("background").set_value(0.07)
    q, I, sigma, *_ = load_data(DATA)
    ex.set_data(q, I, sigma, "test.dat")
    ex.dist_dock.show()
    wait()
    return ex


def fit(ex):
    """Tick the distribution parameters and the background, run the default fit and wait for it."""
    for address in list(start_addresses()) + ["background"]:
        ex._row(address).set_fit(True, emit=True)
    ex.fit_panel.refresh_table()
    ex.fit_panel._fit_or_stop()
    end = time.time() + 300
    while ex.fit_panel._thread is not None and time.time() < end:
        wait(200)
    wait()


def start_addresses():
    return [f"{k}.sd.{n}" for k in (0, 1) for n in ("N", "s", "mu")]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", nargs="+", choices=NAMES, help="only these pictures")
    args = ap.parse_args(argv)
    want = set(args.only or NAMES)

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv[:1])
    ex = build()

    # before the fit: the start values are off, so the curve misses the data
    if "model_dock" in want:
        save(ex.model_dock, "model_dock")
    if "parameter_rows" in want:
        ex.fit_panel.refresh_table()
        save(ex.current_panel(), "parameter_rows")
    for address in start_addresses() + ["background"]:
        ex._row(address).set_fit(True, emit=True)
    ex.fit_panel.refresh_table()
    wait()
    if "fit_panel" in want:
        save(ex.fit_dock, "fit_panel")

    fit(ex)
    if "explorer_main" in want:
        save(ex, "explorer_main")
    if "fit_result" in want:
        ex.fit_panel.tabs.setCurrentWidget(ex.fit_panel.result_page)
        wait(200)
        save(ex.fit_dock, "fit_result")
    if "size_distribution" in want:
        save(ex.dist_dock, "size_distribution")
    if "summary_figure" in want and ex.fit_panel._summary is not None:
        os.makedirs(OUT, exist_ok=True)
        path = os.path.join(OUT, "summary_figure.png")
        export.save_figure(path, ex.fit_panel._summary, ex.fit_panel._problem, dpi=130, dist_power=ex.dist_xpow.value())
        print("wrote", path)
    ex.close()
    app.processEvents()
    return 0


if __name__ == "__main__":
    sys.exit(main())
