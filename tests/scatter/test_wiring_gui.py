"""pySCAtter GUI wiring: every parameter of the calculation has a widget that reaches the calculation, the
widgets that are enabled are those the calculation uses, the combo boxes lead to their parameters, and the
buttons of the Model panel stay outside the scrolling part."""
import os

import pytest

pytest.importorskip("PySide6")

from pysasfit.scatter import params as P  # noqa: E402


@pytest.fixture
def app():
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6 import QtWidgets
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


@pytest.fixture
def panel(app):
    from pysasfit.scatter import gui as g
    return g.ParamPanel()


@pytest.fixture
def win(app, tmp_path, monkeypatch):
    monkeypatch.setenv("PYSASFIT_SETTINGS", str(tmp_path / "settings.json"))
    from pysasfit.scatter import gui as g
    monkeypatch.setattr(g.MainWindow, "_session_path", staticmethod(lambda: str(tmp_path / "scatter_session.json")))
    w = g.MainWindow()
    yield w
    w.shutdown()
    w.close()


SHOWN = [k for k in P.NUM if k not in P.UNUSED]


def _other(k, v):
    _k, _l, _d, lo, hi, dec, _t = P.NUM[k]
    n = v + 1 if dec == 0 else v * 1.25 + 0.05
    return n if n <= hi else (v - 1 if dec == 0 else v * 0.8 - 0.01)


def test_every_parameter_has_one_widget(panel):
    assert set(panel.spin) == set(SHOWN)
    assert set(panel.combo) == set(P.COMBOS)
    for k in SHOWN:                                   # on exactly one toolbox page
        pages = [i for i in range(panel.toolbox.count()) if panel.toolbox.widget(i).isAncestorOf(panel.spin[k])]
        assert len(pages) == 1, k
        assert k in P.LABELS, f"{k}: no readable label"


@pytest.mark.parametrize("k", SHOWN)
def test_spin_box_reaches_params(panel, k):
    seen = []
    panel.changed.connect(lambda: seen.append(panel.params()[k]))
    new = _other(k, panel.spin[k].value())
    panel.spin[k].setValue(new)
    assert seen and seen[-1] == pytest.approx(new, rel=1e-6)
    assert panel.spin[k].minimum() <= P.NUM[k][3] and panel.spin[k].maximum() >= P.NUM[k][4]


@pytest.mark.parametrize("key", list(P.COMBOS))
def test_combo_reaches_params(panel, key):
    cb = panel.combo[key]
    assert [cb.itemText(i) for i in range(cb.count())] == list(P.COMBOS[key])
    for i in range(cb.count()):
        cb.setCurrentIndex(i)
        assert panel.params()[key] == i


def test_width_mode_and_preset_reach_params(panel):
    for i, (ds, para) in enumerate([(True, False), (False, True), (False, False)]):
        panel.width_mode.setCurrentIndex(i)
        p = panel.params()
        assert (p["RadButDebyeScherrer"], p["RadioButtonPara"]) == (ds, para)
    for on in (False, True):
        panel.preset.setChecked(on)
        assert panel.params()["EditQmaxPreset"] is on


def _settings():
    """Combo settings that together switch every parameter on and off."""
    out = []
    for part in range(len(P.PARTICLES)):
        for od in (7, 0, 6, 13):
            out.append(dict(ComboBoxParticle=part, Ordis=od))
    for inter in range(1, len(P.INTERIOR)):
        for part in (0, 1, 3, 5):
            out.append(dict(ComboBoxInterior=inter, ComboBoxParticle=part, ShellProfile=int(inter == 2 and part == 0)))
    for lt in range(len(P.LTYPES)):
        for od in (7, 0, 13):
            out.append(dict(LType=lt, Ordis=od, ComboBoxPeak=(lt + od) % len(P.PEAKS)))
    for pk in range(len(P.PEAKS)):
        out.append(dict(LType=5, ComboBoxPeak=pk))
    for sf in range(len(P.STRUCTURE_FACTOR_NAMES)):
        out.append(dict(StructureFactor=sf))
    out += [dict(EditQmaxPreset=False), dict(LType=5, RadButDebyeScherrer=False),
            dict(LType=5, RadButDebyeScherrer=False, RadioButtonPara=True)]
    return out


def test_enabled_widgets_are_the_relevant_parameters(panel):
    ever_on, ever_off = set(), set()
    for kw in _settings():
        panel.set_params(dict(P.defaults(), **kw), {})
        p = panel.params()
        for k, v in kw.items():
            assert p[k] == v, (kw, k)
        used = P.relevant(p)
        assert used <= set(SHOWN), f"{kw}: used but not shown: {used - set(SHOWN)}"
        for k in SHOWN:
            on = k in used
            assert panel.spin[k].isEnabled() == on and panel.flag[k].isEnabled() == on \
                and panel.label[k].isEnabled() == on, (kw, k)
        ever_on |= used
        ever_off |= set(SHOWN) - used
    assert ever_on == set(SHOWN), f"never enabled: {set(SHOWN) - ever_on}"
    always = {"I0", "Base", "EditQmax", "EditQmin", "EditQsteps", "Editqz", "EditRadius"}
    assert set(SHOWN) - ever_off == always


def test_fit_flags_only_of_enabled_parameters(panel):
    flags = {k: True for k in SHOWN}
    panel.set_params(dict(P.defaults(), ComboBoxParticle=1, Ordis=0), flags)
    got = panel.flags()
    assert set(got) == P.relevant(panel.params())
    assert "EditDbeta" in got and "EditAzi" not in got


def test_orientation_width_is_reachable(panel):
    """Gaussian, Onsager, ...: picking the orientation opens the page with its width, which is enabled for
    particles with an axis; for a lattice the azimuthal domain is on the same page."""
    from pysasfit.scatter import gui as g
    page = panel.toolbox.widget(panel.page_index["Orientation"])
    for k in ("EditDbeta", "EditAzi", "theta", "phi", "ucpsi", "ucn1", "ucn2", "ucn3"):
        assert page.isAncestorOf(panel.spin[k]), k
    for od, name in enumerate(P.ORDIS):
        panel.toolbox.setCurrentIndex(0)
        panel.set_params(dict(P.defaults(), ComboBoxParticle=1, Ordis=od), {})
        panel.combo["Ordis"].activated.emit(od)
        assert panel.toolbox.currentIndex() == panel.page_index["Orientation"]
        assert panel.spin["EditDbeta"].isEnabled() == (od not in (6, 7, 13)), name
    panel.set_params(dict(P.defaults(), LType=5, Ordis=0), {})
    assert panel.spin["EditAzi"].isEnabled() and not panel.spin["EditDbeta"].isEnabled()
    for key, group in g.COMBO_PAGE.items():
        panel.toolbox.setCurrentIndex(panel.toolbox.count() - 1)
        panel.combo[key].activated.emit(0)
        assert panel.toolbox.currentIndex() == panel.page_index[group], key


def test_set_params_roundtrip_of_everything(panel):
    p = P.defaults()
    for k in SHOWN:
        p[k] = _other(k, p[k])
    for key, items in P.COMBOS.items():
        p[key] = len(items) - 1
    p.update(EditQmaxPreset=False, RadButDebyeScherrer=False, RadioButtonPara=True, Comment="x")
    for k in P.UNUSED:                                 # kept for the .ini files although not shown
        p[k] = P.NUM[k][2] + 1
    panel.set_params(p, {})
    q = panel.params()
    assert set(q) == set(p)
    for k, v in p.items():
        assert q[k] == pytest.approx(v, rel=1e-6) if isinstance(v, float) else q[k] == v, k


def test_ini_roundtrip_of_everything(tmp_path):
    p = P.defaults()
    for k in P.NUM:
        p[k] = float(_other(k, p[k]))
    for key, items in P.COMBOS.items():
        p[key] = len(items) - 1
    for k in P.BOOLS:
        p[k] = not p[k]
    f = tmp_path / "all.ini"
    P.write_ini(f, p, {k: True for k in list(P.NUM)[::3]})
    q, fit, _sec = P.read_ini(f)
    for k in list(P.NUM) + list(P.COMBOS) + list(P.BOOLS):
        assert q[k] == pytest.approx(p[k]), k
    assert {k for k, v in fit.items() if v} == set(list(P.NUM)[::3])


def test_model_panel_buttons_do_not_scroll(app, win):
    """Calculate / Abort / Copy Python and the two 'follows' boxes are above the scrolling parameters."""
    from PySide6 import QtWidgets
    sc = win.params_scroll
    assert isinstance(sc, QtWidgets.QScrollArea) and sc.isAncestorOf(win.params_panel)
    buttons = [win.calc_button, win.abort_button, win.auto_1d, win.auto_2d]
    buttons += [b for b in win.model_widget.findChildren(QtWidgets.QPushButton) if "Python" in b.text()]
    assert len(buttons) == 5
    for b in buttons:
        assert not sc.isAncestorOf(b), b.text()
    win.resize(1200, 420)                              # too low for the parameters: they scroll, the buttons stay
    win.show()
    for _ in range(20):
        app.processEvents()
    bar = sc.verticalScrollBar()
    assert bar.maximum() > 0
    bar.setValue(bar.maximum())
    for _ in range(20):
        app.processEvents()
    for b in buttons[:4]:
        assert not b.visibleRegion().isEmpty(), b.text()


def test_gui_parameters_arrive_in_the_calculation(app, win):
    """The parameter set the window hands to the engine is that of the panel."""
    import time
    p = dict(win.params(), ComboBoxParticle=1, EditRadius=4.0, Length=30.0, Ordis=0, EditDbeta=12.0, GridPoints=16)
    win.params_panel.set_params(p, {})
    win.calculate(two_d=True)
    t = time.time()
    while time.time() - t < 120:
        app.processEvents()
        if win.worker is None or not win.worker.isRunning():
            break
        time.sleep(0.02)
    for _ in range(10):
        app.processEvents()
    assert win.result2d_params == win.params()
    assert win.result2d_params["EditDbeta"] == 12.0 and win.result2d[3].shape == (16, 16)


def test_entries_that_are_not_calculated_cannot_be_picked(panel):
    for key, items in P.NOT_AVAILABLE.items():
        cb = panel.combo[key]
        for i in range(cb.count()):
            assert cb.model().item(i).isEnabled() == (i not in items), cb.itemText(i)
    panel.set_params(dict(P.defaults(), LType=13), {})          # an old .ini file can still carry it
    assert panel.params()["LType"] == 13
