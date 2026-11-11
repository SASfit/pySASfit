"""Live view of a running fit: the best point so far (fitting.on_best hook, the 'Progress' tab), and Stop keeping it."""

import itertools
import os
import threading
import time

import numpy as np
import pytest

try:
    from .support import Q, make_data, problem, two_dataset_problem
except ImportError:
    from support import Q, make_data, problem, two_dataset_problem


class Recorder:
    """A fit callback with the live-view hook."""

    def __init__(self, cancel_after=None, cancel=None):
        self.calls, self.best = 0, []
        self.cancel_after, self.cancel = cancel_after, cancel

    def __call__(self, n, chi2_red):
        self.calls += 1

    def on_best(self, ctx):
        self.best.append((ctx.nfev, ctx.best_chi2 / ctx.dof, np.array(ctx.best_vals), ctx.best_curves()))
        if self.cancel_after is not None and len(self.best) >= self.cancel_after:
            self.cancel.set()


# ---------------------------------------------------------------------------- engine
@pytest.mark.parametrize("vp", [False, True])
def test_on_best_reports_improving_points_with_their_model_curve(vp):
    p = problem()
    rec = Recorder()
    s = p.fit("scipy", callback=rec, vp=vp)
    assert s.success and rec.best and rec.calls >= len(rec.best)
    chi = [b[1] for b in rec.best]
    assert all(b <= a for a, b in itertools.pairwise(chi))  # only improvements
    # the curve rebuilt from the residuals is the model at the reported values (linear parameters solved with vp)
    _n, _c, vals, curves = rec.best[-1]
    ((q, I, _sig, Im),) = curves
    d = p.datasets[0]
    assert q is d.q and I is d.I
    p.apply(0, vals)
    assert np.allclose(Im, d.model.intensity(d.q), rtol=1e-9, atol=1e-12)


def test_best_curves_for_several_data_sets():
    p = two_dataset_problem()
    rec = Recorder()
    p.fit("scipy", callback=rec)
    _n, _c, vals, curves = rec.best[-1]
    assert len(curves) == 2
    for k, (q, I, sig, Im) in enumerate(curves):
        p.apply(k, vals)
        assert np.allclose(Im, p.datasets[k].model.intensity(q), rtol=1e-9)


def test_a_plain_callback_still_works():
    seen = []
    problem().fit("scipy", callback=lambda n, c: seen.append(n))
    assert seen and seen == sorted(seen)


def test_stop_keeps_the_best_point_reported_so_far():
    cancel = threading.Event()
    p = problem(start=(1.0, 2.0))
    rec = Recorder(cancel_after=3, cancel=cancel)
    s = p.fit("scipy", callback=rec, cancel=cancel)
    assert "cancelled" in s.message and not s.success
    _n, chi2_red, vals, _curves = rec.best[-1]
    assert np.allclose([q.value for q in p.parameters], vals[: len(p.parameters)])
    assert s.chi2_red == pytest.approx(chi2_red, rel=1e-6)


# ---------------------------------------------------------------------------- panel
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


@pytest.fixture(scope="module")
def app():
    pytest.importorskip("PySide6")
    from PySide6 import QtWidgets

    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


@pytest.fixture()
def panel(app):
    from pysasfit.studio import fit_panel

    try:
        from .test_fit_panel_gui import StubExplorer
    except ImportError:
        from test_fit_panel_gui import StubExplorer
    ex = StubExplorer()
    ex.set_data(*make_data(), name="sample.dat")
    p = fit_panel.FitPanel(ex)
    p.refresh_table()
    yield p
    p.deleteLater()


def _wait(p, timeout=60):
    from PySide6 import QtWidgets

    t0 = time.time()
    while p._thread is not None and time.time() - t0 < timeout:
        QtWidgets.QApplication.processEvents()
        time.sleep(0.005)
    QtWidgets.QApplication.processEvents()
    assert p._thread is None, "the fit did not finish"


def test_the_progress_tab_shows_the_finished_fit(panel):
    assert panel.live.isChecked()
    panel._fit_or_stop()
    assert panel.tabs.currentWidget() is panel.progress_page  # shown while fitting
    _wait(panel)
    lv = panel.progress_page
    assert lv.table.rowCount() == 3
    assert [lv.table.item(r, 0).text() for r in range(3)] == ["ff.A", "ff.k", "background"]
    fitted = {q.label: q.value for q in panel._problem.parameters}
    assert float(lv.table.item(0, 2).text()) == pytest.approx(fitted["ff.A"], rel=1e-5)
    assert "converged" in lv.info.text()
    assert panel.tabs.currentWidget() is panel.result_page  # a finished fit shows its result
    if lv.fig is not None:
        assert lv.draws >= 1 and len(lv._lines) == 1
        x, _y = lv._lines[0].get_data()
        assert np.allclose(x, Q)


def test_snapshots_arrive_while_the_fit_runs(panel, monkeypatch):
    from pysasfit.studio import fit_panel

    monkeypatch.setattr(fit_panel._FitThread, "SNAPSHOT_INTERVAL", 0.0)
    snaps = []
    real = fit_panel._LiveView.update_snapshot
    monkeypatch.setattr(fit_panel._LiveView, "update_snapshot",
                        lambda self, snap, final=None: (snaps.append(final), real(self, snap, final)))
    panel._fit_or_stop()
    _wait(panel)
    assert snaps.count(None) >= 1  # intermediate snapshots (final=None) before the final one
    assert snaps[-1] is not None


def test_live_view_off_sends_no_snapshots(panel):
    panel.live.setChecked(False)
    before = panel.progress_page.draws
    panel._fit_or_stop()
    assert panel._thread.live is False
    _wait(panel)
    assert panel.progress_page.draws == before
    assert panel.tabs.currentWidget() is panel.result_page


def test_stop_keeps_the_best_point_and_says_so(panel, monkeypatch):
    from pysasfit.studio import fit_panel

    # a slow model: the fit is still running when Stop is pressed
    model = panel.explorer.model
    toy = model.contributions[0]
    fast = toy.intensity
    monkeypatch.setattr(toy, "intensity", lambda q: (time.sleep(0.01), fast(q))[1])
    monkeypatch.setattr(fit_panel._FitThread, "SNAPSHOT_INTERVAL", 0.0)
    panel._fit_or_stop()
    from PySide6 import QtWidgets

    t0 = time.time()
    while not panel._thread.trace and time.time() - t0 < 20:
        QtWidgets.QApplication.processEvents()
        time.sleep(0.005)
    assert panel.fit_btn.text() == "Stop"
    panel._fit_or_stop()  # Stop
    _wait(panel)
    assert "stopped" in panel.status.text() and "best point so far kept" in panel.status.text()
    assert "stopped" in panel.progress_page.info.text()
    assert panel.tabs.currentWidget() is panel.progress_page  # stays on the picture of where it stopped
    assert panel.undo_btn.isEnabled()
    assert "cancelled" in panel._summary.message
