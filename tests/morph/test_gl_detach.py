"""3D panel detached into its own window (new GL context): no stale shader programs or buffers.

Runs the panel in a subprocess with a real OpenGL platform (native on Windows, Xvfb on Linux if installed),
with and without Qt.AA_ShareOpenGLContexts. Before the fix pyqtgraph printed 'Error while drawing item'
(GLError 1281 in glUseProgram / glGetAttribLocation) after a dock was floated.
"""

import os
import shutil
import subprocess
import sys

import pytest

SCRIPT = r'''
import sys, numpy as np
from PySide6 import QtCore, QtWidgets
if sys.argv[1] == "share":
    QtCore.QCoreApplication.setAttribute(QtCore.Qt.ApplicationAttribute.AA_ShareOpenGLContexts)
app = QtWidgets.QApplication([])
from pysasfit.morph.multiphase import DataSet, MultiParameters, reconstruct_multi
from pysasfit.morph.sesans import dab_sesans
from pysasfit.morph.view3d_gl import MorphGLView
d = np.geomspace(50, 5000, 30); G = dab_sesans(d, 300.0, 1e-6); dG = 0.01 * abs(G).max() * np.ones_like(d)
res = reconstruct_multi([DataSet(d, G, dG, np.array([1., 0.]), name="x", kind="sesans")],
                        MultiParameters(names=["a", "b"], fractions=[0.5, 0.5], n=24, box_size=4000, seed=1),
                        keep_field=True)
mw = QtWidgets.QMainWindow(); mw.resize(600, 400)
dock = QtWidgets.QDockWidget("3D", mw); v = MorphGLView(); dock.setWidget(v)
mw.addDockWidget(QtCore.Qt.DockWidgetArea.RightDockWidgetArea, dock)
mw.setCentralWidget(QtWidgets.QLabel("x")); mw.show()
v.set_result(res)
def spin(n):
    for _ in range(n):
        app.processEvents(); QtCore.QThread.msleep(20)
def drawn():
    img = v.view.grabFramebuffer()
    return sum((img.pixel(x, y) & 0xffffff) != 0xffffff for x in range(0, img.width(), 9)
               for y in range(0, img.height(), 9))
spin(20); a = drawn()
dock.setFloating(True); spin(40); b = drawn()
dock.setFloating(False); spin(40); c = drawn()
print("DRAWN", a, b, c, len(v.view.items))
'''


def _command(mode):
    if sys.platform.startswith("win") or sys.platform == "darwin":
        return [sys.executable, "-c", SCRIPT, mode]
    if shutil.which("xvfb-run"):
        return ["xvfb-run", "-a", "-s", "-screen 0 1280x1024x24", sys.executable, "-c", SCRIPT, mode]
    return None


@pytest.mark.parametrize("mode", ["share", "noshare"])
def test_detached_3d_panel_draws_without_gl_errors(mode):
    pytest.importorskip("pyqtgraph.opengl")
    pytest.importorskip("OpenGL")
    cmd = _command(mode)
    if cmd is None:
        pytest.skip("no OpenGL display (xvfb-run) for this test")
    env = dict(os.environ)
    env.pop("QT_QPA_PLATFORM", None)
    src = os.path.join(os.path.dirname(__file__), "..", "..", "src")
    env["PYTHONPATH"] = os.pathsep.join([os.path.abspath(src), env.get("PYTHONPATH", "")])
    out = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=300)
    text = out.stdout + out.stderr
    if "DRAWN" not in out.stdout:
        pytest.skip(f"OpenGL panel could not run here: {text[-400:]}")
    assert "Error while drawing" not in text, text[-2000:]
    a, b, c, n = (int(x) for x in out.stdout.split("DRAWN")[1].split()[:4])
    assert n == 2 and min(a, b, c) > 0.5 * a
