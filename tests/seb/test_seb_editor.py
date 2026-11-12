"""The pySEBpolymers designer widget: palette, linking by mouse, inspector edits, undo, copy/paste, components."""
import os

import numpy as np
import pytest

pytest.importorskip("PySide6")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6 import QtCore, QtWidgets  # noqa: E402
from PySide6.QtTest import QTest  # noqa: E402

from pysasfit.seb import builders  # noqa: E402
from pysasfit.seb.editor import DesignerWidget, LinkItem, NodeItem  # noqa: E402
from pysasfit.seb.graph import Design  # noqa: E402


@pytest.fixture
def ed():
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    w = DesignerWidget()
    w.resize(1100, 650)
    w.show()
    app.processEvents()
    yield w
    w.close()


def _settle():
    app = QtWidgets.QApplication.instance()
    for _ in range(5):
        app.processEvents()


def _port_point(ed, node_id, port):
    it = ed.scene.node_items[node_id].ports[port]
    return ed.view.mapFromScene(it.scene_centre())


def test_build_a_micelle_by_mouse(ed):
    got = []
    ed.changed.connect(lambda d: got.append(d))
    ed.add_from_palette("unit:SolidSphere", QtCore.QPointF(0, 0))
    ed.add_from_palette("pattern:Array", QtCore.QPointF(0, 220))
    _settle()
    d = ed.root
    core, arr = sorted(d.nodes.values(), key=lambda n: n.id)
    ed.view.centerOn(100, 120)
    _settle()
    a, b = _port_point(ed, core.id, "surface"), _port_point(ed, arr.id, "anchor")
    vp = ed.view.viewport()
    QTest.mousePress(vp, QtCore.Qt.MouseButton.LeftButton, QtCore.Qt.KeyboardModifier.NoModifier, a)
    QTest.mouseMove(vp, (a + b) / 2)
    QTest.mouseMove(vp, b)
    QTest.mouseRelease(vp, QtCore.Qt.MouseButton.LeftButton, QtCore.Qt.KeyboardModifier.NoModifier, b)
    _settle()
    assert len(ed.root.links) == 1, ed.status.text()
    assert not ed.root.problems() and got
    s = got[-1].to_structure()
    assert len(s.units()) == 11 and set(s.parameters()) == {"R_sphere1", "beta_sphere1", "Rg_chain", "beta_chain"}
    p = {"R_sphere1": 20.0, "Rg_chain": 8.0, "beta_sphere1": 3.0, "beta_chain": 0.5}
    s_ref = builders.micelle(10)
    p_ref = {"R_core": 20.0, "Rg_chain": 8.0, "beta_core": 3.0, "beta_chain": 0.5}
    q = np.logspace(-3, 0, 50)
    np.testing.assert_allclose(s.form_factor(q, p), s_ref.form_factor(q, p_ref), rtol=1e-12)


def test_loops_are_refused_and_undo_redo(ed):
    d = ed.root
    ed.edit(lambda: (d.add_unit("GaussianPolymer", "a"), d.add_unit("GaussianPolymer", "b")))
    a, b = ed.root.by_name("a"), ed.root.by_name("b")
    ed._on_link(ed.scene.node_items[a.id].ports["end2"], ed.scene.node_items[b.id].ports["end1"])
    assert len(ed.root.links) == 1
    ed._on_link(ed.scene.node_items[a.id].ports["end1"], ed.scene.node_items[b.id].ports["end2"])
    assert len(ed.root.links) == 1 and "loop" in ed.status.text()
    ed.undo()
    assert len(ed.root.links) == 0
    ed.redo()
    assert len(ed.root.links) == 1
    ed.undo()
    ed.undo()
    assert not ed.root.nodes


def test_inspector_pattern_edits(ed):
    d = ed.root
    ed.edit(lambda: d.add_unit("SolidCylinder", "cyl"))
    ed.edit(lambda: d.add_pattern("Array", "GaussianPolymer", name="hairs", n=4))
    cyl, arr = ed.root.by_name("cyl"), ed.root.by_name("hairs")
    ed.edit(lambda: ed.root.link(cyl.id, "hull", arr.id, "anchor"))
    ed.scene.node_items[arr.id].setSelected(True)
    _settle()
    form = ed.inspector.widget()
    spins = form.findChildren(QtWidgets.QSpinBox)
    spins[0].setValue(7)                          # copies
    _settle()
    assert ed.root.by_name("hairs").n == 7
    assert ed.root.count_units() == 8
    # element: a rod attached by its middle
    ed.edit(lambda: ed.set_element(ed.root.by_name("hairs"), "ThinRod", False))
    ed.edit(lambda: setattr(ed.root.by_name("hairs"), "in_ref", "middle"))
    s = ed.root.to_structure()
    assert {k for _n, k, _t in s.units()} == {"SolidCylinder", "ThinRod"}
    # pattern type Chain: the anchor link moves to the chain's start
    ed.edit(lambda: ed.change_pattern_type(ed.root.by_name("hairs"), "Chain"))
    ln = ed.root.links[0]
    assert "start" in (ln.a[1], ln.b[1]) and not ed.root.problems()


def test_labels_of_a_link(ed):
    d = ed.root
    ed.edit(lambda: [d.add_unit("SolidSphere", "core")] + [d.add_unit("GaussianPolymer", f"c{i}") for i in range(2)])
    core = ed.root.by_name("core")
    for i in range(2):
        ci = ed.root.by_name(f"c{i}")
        ed.edit(lambda ci=ci: ed.root.link(core.id, "surface", ci.id, "end1"))
    labels = sorted(e[2] for ln in ed.root.links for e in ln.ends() if e[1] == "surface")
    assert labels == ["p1", "p2"]                     # independent points
    ln = ed.root.links[1]
    ed.edit(lambda: ed.root.set_label(ln, core.id, "p1"))
    s = ed.root.to_structure()
    assert len(s.links()) == 1                        # both chains at one point now


def test_copy_paste_duplicate_delete(ed):
    d = ed.root
    ed.edit(lambda: (d.add_unit("GaussianPolymer", "a"), d.add_unit("ThinRod", "b")))
    a, b = ed.root.by_name("a"), ed.root.by_name("b")
    ed.edit(lambda: ed.root.link(a.id, "end2", b.id, "end1"))
    ed.select_all()
    ed.copy_selection()
    ed.paste()
    assert len(ed.root.nodes) == 4 and len(ed.root.links) == 2
    names = sorted(n.name for n in ed.root.nodes.values())
    assert len(set(names)) == 4
    ed.duplicate()                                    # the pasted pair is selected
    assert len(ed.root.nodes) == 6
    ed.delete_selection()
    assert len(ed.root.nodes) == 4


def test_component_from_selection_and_chain_of_it(ed, monkeypatch):
    d = ed.root
    ed.edit(lambda: (d.add_unit("GaussianPolymer", "bb", tag="bb"), d.add_unit("GaussianPolymer", "sc", tag="sc")))
    bb, sc = ed.root.by_name("bb"), ed.root.by_name("sc")
    ed.edit(lambda: ed.root.link(bb.id, "end2", sc.id, "end1"))
    ed.select_all()
    monkeypatch.setattr(QtWidgets.QInputDialog, "getText", staticmethod(lambda *a, **k: ("seg", True)))
    ed.on_component_from_selection()
    assert ed.design is ed.root.components["seg"]
    c = ed.design
    ed.edit(lambda: c.set_port("in", c.by_name("bb").id, "end1"))
    ed.edit(lambda: c.set_port("out", c.by_name("bb").id, "end2"))
    ed.editing.setCurrentIndex(0)                     # back to the structure
    assert ed.design is ed.root
    root = ed.root
    ed.edit(lambda: [root.remove_node(n) for n in list(root.nodes)])
    ed.add_from_palette("comp:seg", QtCore.QPointF(0, 0))
    arr = next(iter(ed.root.nodes.values()))
    ed.edit(lambda: ed.change_pattern_type(ed.root.nodes[arr.id], "Chain"))
    ed.edit(lambda: setattr(ed.root.nodes[arr.id], "n", 10))
    s = ed.root.to_structure()
    p = {"Rg_bb": 3.0, "Rg_sc": 6.0, "beta_bb": 1.0, "beta_sc": 0.7}
    q = np.logspace(-3, 0, 50)
    np.testing.assert_allclose(s.form_factor(q, p), builders.bottlebrush(10).form_factor(q, p), rtol=1e-12)
    # the component cannot be deleted while it is used
    ed.editing.setCurrentIndex(1)
    ed.on_delete_component()
    assert "seg" in ed.root.components and "used" in ed.status.text()


def test_items_and_inspector_for_every_node_type(ed):
    d = Design.from_dict(Design("x").to_dict())
    ed.set_design(d)
    for k in ("Point", "GaussianPolymer", "GaussianLoop", "ThinRod", "SolidSphere", "ThinSphericalShell",
              "SolidSphericalShell", "ThinCircle", "ThinDisk", "SolidCylinder", "Spheroid"):
        ed.add_from_palette(f"unit:{k}", None)
    for k in ("Array", "Chain", "Generations"):
        ed.add_from_palette(f"pattern:{k}", None)
    _settle()
    for it in list(ed.scene.node_items.values()):
        ed.scene.clearSelection()
        it.setSelected(True)
        _settle()
        assert ed.inspector.widget() is not None
    ed.on_layout()
    _settle()
    img = ed.grab()
    assert not img.isNull()
    assert all(isinstance(i, NodeItem) for i in ed.scene.node_items.values())
    assert not any(isinstance(i, LinkItem) for i in ed.scene.items())
