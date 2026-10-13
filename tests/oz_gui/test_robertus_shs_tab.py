import json

import numpy as np


def test_compute_core_shell_export_and_session(qapp, dialogs, wait, tmp_path):
    from pysasfit.oz.gui.robertus_shs_tab import IMPORT_ERROR, RobertusSHSTab

    assert IMPORT_ERROR is None
    t = RobertusSHSTab()
    t.resize(1200, 800)
    t.ffVar.set("Core-shell")
    t._syncEnabledStates()
    assert all(w.isEnabled() for w in t._entryWidgets.values())
    t.nQVar.set("60")
    t.nbinsVar.set("6")
    t._onCompute()
    wait(lambda: not t.worker.is_alive() and t.result is not None)
    assert t.statusVar.get().startswith("done")
    r = t.result
    assert r.Q.size == 60 and np.all(np.isfinite(r.I_exact)) and r.S.shape == (60, 6, 6)
    assert "sigma_eff" in t.summaryText.toPlainText()

    # export through the shared controls (dialog answered with a path)
    out = tmp_path / "rob.txt"
    dialogs.answers["save"] = str(out)
    t.exportFormatVar.set("ASCII")
    t._onExportSelected()
    lines = out.read_text().splitlines()
    assert lines[0].startswith("# Q\tI_exact") and len(lines) == 61

    sess = tmp_path / "rob.oz"
    dialogs.answers["save"] = str(sess)
    t._onSaveAll()
    assert json.loads(sess.read_text())["format"] == "sasfit_polydisperse_tab_save_v2"

    # invalid input -> error dialog, no worker
    t.QmaxVar.set("abc")
    t._onCompute()
    assert dialogs.kinds()[-1] == "error"
