"""Fixtures for the Qt OZ GUI tests (offscreen; dialogs intercepted)."""

import os
import time

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")


@pytest.fixture(scope="session")
def qapp():
    from PySide6 import QtWidgets

    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


@pytest.fixture
def dialogs():
    """Record dialogs instead of showing them. ``dialogs.answers[kind]`` sets the return value
    (e.g. a file path for 'save'/'open'/'directory', True for 'yesno')."""
    from pysasfit.oz.gui import qtkit

    class Recorder:
        def __init__(self):
            self.calls = []
            self.answers = {}

        def __call__(self, kind, **kw):
            self.calls.append((kind, kw))
            return self.answers.get(kind, "")

        def kinds(self):
            return [k for k, _ in self.calls]

    rec = Recorder()
    qtkit.set_dialog_hook(rec)
    yield rec
    qtkit.set_dialog_hook(None)


@pytest.fixture
def wait(qapp):
    """``wait(predicate, timeout=120)``: process Qt events until ``predicate()`` is true."""
    return lambda predicate, timeout=120.0: wait_until(qapp, predicate, timeout)


def wait_until(qapp, predicate, timeout=120.0):
    """Process Qt events until ``predicate()`` is true (or fail after ``timeout`` s)."""
    t0 = time.time()
    while not predicate():
        qapp.processEvents()
        time.sleep(0.02)
        if time.time() - t0 > timeout:
            raise AssertionError("timed out waiting for the GUI")
    for _ in range(5):
        qapp.processEvents()
