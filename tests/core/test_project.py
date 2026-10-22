"""Project files and the automatic last session (pysasfit.core.project)."""
import json
import os
import shutil
import time
import zipfile

import numpy as np
import pytest

pytest.importorskip("PySide6")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6 import QtWidgets  # noqa: E402

from pysasfit.core import project as prj  # noqa: E402

ROOT = os.path.join(os.path.dirname(__file__), "..", "..")
SLD_DATA = os.path.join(ROOT, "examples", "sldprofile", "data", "inhomogeneous_sphere_r5nm.dat")
INV_DATA = os.path.join(ROOT, "tests", "inversion", "data", "test.dat")


def _app():
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


def _wait(app, w):
    t = time.time()
    while (w.worker is not None or w.result is None) and time.time() - t < 120:
        app.processEvents()
        time.sleep(0.02)


def _sld_fit(tmp_path):
    from pysasfit.sldprofile.gui import MainWindow
    app = _app()
    data = str(tmp_path / "sphere.dat")
    shutil.copy(SLD_DATA, data)
    w = MainWindow()
    w.load_file(data)
    w.cb_rep.setCurrentIndex(1)          # cubic B-spline
    w.sp_n.setValue(15)
    w.ck_fix.setChecked(True)
    w.ck_auto.setChecked(False)
    w.sp_rmax.setValue(5.0)
    w.on_run()
    _wait(app, w)
    assert w.result is not None
    return app, w, data


def test_sldprofile_project_roundtrip(tmp_path):
    from pysasfit.sldprofile.gui import MainWindow
    app, w, data = _sld_fit(tmp_path)
    fn = str(tmp_path / "proj.pysasfit")
    assert w.save_project_to(fn)
    chi2 = w.result.chi2_red
    w2 = MainWindow()
    assert w2.open_project(fn)
    assert w2.filename == data
    assert w2.cb_rep.currentIndex() == 1 and w2.sp_n.value() == 15 and w2.ck_fix.isChecked()
    assert w2.result is not None and np.isclose(w2.result.chi2_red, chi2)
    assert getattr(w2.result.best.grid, "is_bspline", False)
    # moved data file: the embedded copy is used, the result is still shown
    os.remove(data)
    w3 = MainWindow()
    assert w3.open_project(fn)
    assert w3.filename != data and os.path.isfile(w3.filename)
    assert w3.result is not None and "copy stored in the project" in w3.log.toPlainText()
    for x in (w, w2, w3):
        x.close()


def test_missing_and_unknown_entries(tmp_path):
    """Entries missing in the file (e.g. parameters added later) take their default
    values; unknown entries and a newer format version are tolerated."""
    from pysasfit.sldprofile.gui import MainWindow
    app, w, data = _sld_fit(tmp_path)
    fn = str(tmp_path / "proj.pysasfit")
    w.save_project_to(fn)
    # edit the manifest: drop sp_n and cb_rep, add an unknown control, raise the version
    with zipfile.ZipFile(fn) as z:
        parts = {n: z.read(n) for n in z.namelist()}
    m = json.loads(parts["project.json"])
    del m["controls"]["sp_n"], m["controls"]["cb_rep"]
    m["controls"]["sp_future_parameter"] = {"kind": "spin", "value": 3.0}
    m["format_version"] = prj.FORMAT_VERSION + 1
    parts["project.json"] = json.dumps(m).encode()
    parts.pop("result.pkl")
    with zipfile.ZipFile(fn, "w") as z:
        for n, b in parts.items():
            z.writestr(n, b)
    w2 = MainWindow()
    w2.sp_n.setValue(33)                                  # must be reset to the default, not kept
    assert w2.open_project(fn)
    assert w2.sp_n.value() == w2._default_controls["sp_n"][1]
    assert w2.cb_rep.currentIndex() == 0                  # default representation
    assert w2.sp_rmax.value() == 5.0 and w2.ck_fix.isChecked()
    log = w2.log.toPlainText()
    assert "newer than this program" in log and "sp_future_parameter" in log
    assert w2.result is None
    with pytest.raises(prj.ProjectError):
        prj.load_project(fn, "inversion")                # project of another tool
    w.close()
    w2.close()


def test_last_session_and_forget(tmp_path):
    from pysasfit.sldprofile.gui import MainWindow
    prj.forget_session()
    app, w, data = _sld_fit(tmp_path)
    w.sp_qmin.setValue(0.2)
    w.close()                                             # saves the session
    assert os.path.isfile(prj.session_path("sldprofile"))
    w2 = MainWindow()
    assert w2.restore_session()
    assert w2.filename == data and abs(w2.sp_qmin.value() - 0.2) < 1e-9
    assert w2.result is not None
    w2.forget_session()
    w2.close()                                            # does not save again after "forget"
    assert not os.path.isfile(prj.session_path("sldprofile"))
    assert not MainWindow().restore_session()


def test_inversion_project_roundtrip(tmp_path):
    from pysasfit.inversion.gui.app import MainWindow
    _app()
    w = MainWindow()
    w.load_file(INV_DATA)
    w.use_fixed_bg.setChecked(True)
    w.fixed_bg_spin.setValue(0.1)
    w.kernel_combo.setCurrentIndex(w.kernel_combo.findData("lamella_pddf"))
    w.solver_combo.setCurrentIndex(w.solver_combo.findData("bayesian_evidence"))
    w.r_n_spin.setValue(40)
    w._on_solve()
    x = np.asarray(w._last_result.x)
    fn = str(tmp_path / "inv.pysasfit")
    assert w.save_project_to(fn)
    w2 = MainWindow()
    assert w2.open_project(fn)
    assert w2.kernel_combo.currentData() == "lamella_pddf"
    assert w2.option_widgets["penalty"][1].currentData() == "hansen_eq19_even"
    assert w2.r_n_spin.value() == 40 and w2.use_fixed_bg.isChecked()
    assert np.allclose(w2._last_result.x, x)
    assert "Lamella" in w2.help_text("result")
    w.close()
    w2.close()
