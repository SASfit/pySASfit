"""Designs (pysasfit.seb.graph): patterns, components, code generation, import, persistence."""
import numpy as np
import pytest

from pysasfit.seb import Structure, builders
from pysasfit.seb import special as sf
from pysasfit.seb import templates as T
from pysasfit.seb.graph import Design

q = np.logspace(-3, 0, 80)


def _same(s1, s2, params, rtol=1e-12):
    np.testing.assert_allclose(s1.form_factor(q, params), s2.form_factor(q, params), rtol=rtol)


def test_micelle_design_equals_builder():
    d = T.design_for("Micelle (core + grafted chains)", {"n": 7})
    p = {"R_core": 20.0, "Rg_chain": 8.0, "beta_core": 3.0, "beta_chain": 0.5}
    _same(d.to_structure(), builders.micelle(7), p)
    d = T.design_for("Micelle (core + grafted chains)", {"n": 5, "same_point": True})
    _same(d.to_structure(), builders.micelle(5, same_point=True), p)


def test_dendrimer_and_brush_designs_equal_builders():
    d = T.design_for("Dendrimer", {"generations": 4, "functionality": 3})
    p = {f"Rg_g{k}": 3.0 + k for k in range(4)} | {f"beta_g{k}": 1.0 + 0.2 * k for k in range(4)}
    _same(d.to_structure(), builders.dendrimer(4, 3), p)
    d = T.design_for("Bottle brush", {"n": 12})
    p = {"Rg_bb": 3.0, "Rg_sc": 6.0, "beta_bb": 1.0, "beta_sc": 0.7}
    _same(d.to_structure(), builders.bottlebrush(12), p)


def test_multiblock_is_a_gaussian_chain():
    d = T.design_for("Linear / multiblock chain", {"n": 3, "alternate": True})
    s = d.to_structure()
    assert len(s.units()) == 6
    F = s.form_factor(q, {"Rg_A": 4.0, "Rg_B": 4.0, "beta_A": 1.0, "beta_B": 1.0})
    np.testing.assert_allclose(F, sf.debye(q**2 * 6 * 16.0), rtol=1e-10)


def test_chain_entered_from_its_end_and_arrays_on_pattern_ports():
    """A sphere linked to the END of a chain of components; an array at the chain's start."""
    d = Design("t")
    c = d.new_component("seg")
    a = c.add_unit("GaussianPolymer", "a", tag="a")
    b = c.add_unit("ThinRod", "b", tag="b")
    c.link(a.id, "end2", b.id, "end1")
    c.set_port("in", a.id, "end1")
    c.set_port("out", b.id, "end2")
    sph = d.add_unit("SolidSphere", "core", tag="core")
    ch = d.add_pattern("Chain", "seg", component=True, name="ch", n=3)
    arr = d.add_pattern("Array", "GaussianPolymer", name="hairs", tag="hair", n=4, in_ref="end1")
    d.link(sph.id, "surface", ch.id, "end", "x")
    d.link(ch.id, "start", arr.id, "anchor")
    s = d.to_structure()
    assert len(s.units()) == 1 + 6 + 4
    # the same structure by hand
    r = Structure()
    r.add("SolidSphere", "core")
    prev = "core.surface#x"
    for i in (2, 1, 0):
        r.link("ThinRod", f"b{i}.end2", prev, tag="b")
        r.link("GaussianPolymer", f"a{i}.end2", f"b{i}.end1", tag="a")
        prev = f"a{i}.end1"
    for k in range(4):
        r.link("GaussianPolymer", f"h{k}.end1", "a0.end1", tag="hair")
    p = {"R_core": 15.0, "Rg_a": 5.0, "L_b": 30.0, "Rg_hair": 4.0, "beta_core": 2.0, "beta_a": 1.0, "beta_b": 0.5,
         "beta_hair": 0.3}
    _same(s, r, p)


def test_nested_components_and_generations_of_components():
    d = Design("t")
    leaf = d.new_component("leaf")             # a rod with 3 loops on its contour
    rod = leaf.add_unit("ThinRod", "rod", tag="rod")
    loops = leaf.add_pattern("Array", "GaussianLoop", name="loops", tag="loop", n=3, in_ref="contour")
    leaf.link(rod.id, "contour", loops.id, "anchor", "q")
    leaf.set_port("in", rod.id, "end1")
    leaf.set_port("out", rod.id, "end2")
    d.add_pattern("Generations", "leaf", component=True, name="tree", f=3, g=2, gen_tags=True)
    s = d.to_structure()
    assert len(s.units()) == d.count_units() == (3 + 6) * 4
    assert set(s.parameters()) == {"L_rod0", "L_rod1", "Rg_loop0", "Rg_loop1", "beta_rod0", "beta_rod1",
                                   "beta_loop0", "beta_loop1"}
    p = {k: (1.0 if k.startswith("beta") else 9.0) for k in s.parameters()}
    F = s.form_factor(q, p)
    np.testing.assert_allclose(F, s.form_factor_pairs(q, p), rtol=1e-10)
    assert abs(F[0] - 1) < 1e-3


def test_links_refuse_loops_and_bad_ports():
    d = Design()
    a = d.add_unit("GaussianPolymer")
    b = d.add_unit("GaussianPolymer")
    c = d.add_unit("SolidSphere")
    d.link(a.id, "end2", b.id, "end1")
    d.link(b.id, "end2", c.id, "surface")
    with pytest.raises(ValueError, match="loop"):
        d.link(c.id, "center", a.id, "end1")
    with pytest.raises(ValueError):
        d.link(a.id, "nonsense", c.id, "center")
    arr = d.add_pattern("Array", "GaussianPolymer", n=3)
    assert "not anchored" in " ".join(d.problems())
    d.link(c.id, "surface", arr.id, "anchor")
    with pytest.raises(ValueError, match="anchored"):
        d.link(a.id, "end1", arr.id, "anchor")
    assert not d.problems()
    d.add_unit("Point")
    assert "not connected" in " ".join(d.problems())


def test_labels_join_points():
    """Two chains on one surface label = same point; different labels = independent points."""
    def micelle(labels):
        d = Design()
        core = d.add_unit("SolidSphere", "core", tag="core")
        for i, lab in enumerate(labels):
            ch = d.add_unit("GaussianPolymer", f"c{i}", tag="chain")
            d.link(core.id, "surface", ch.id, "end1", lab)
        return d.to_structure()
    p = {"R_core": 20.0, "Rg_chain": 8.0, "beta_core": 3.0, "beta_chain": 0.5}
    _same(micelle(["a", "a", "a"]), builders.micelle(3, same_point=True), p)
    _same(micelle(["a", "b", "c"]), builders.micelle(3), p)


def test_json_roundtrip_and_copy():
    d = T.design_for("Bottle brush", {"n": 5})
    d2 = Design.from_json(d.to_json())
    assert d2.to_code() == d.to_code()
    assert d.copy().to_dict() == d.to_dict()


def test_import_from_structure():
    for s in (builders.micelle(6), builders.dendrimer(3), builders.star(4, kind="ThinRod"),
              T.build(T.code_for("Custom"))):
        d = Design.from_structure(s)
        assert not d.problems()
        s2 = d.to_structure()
        p = {k: (1.0 if k.startswith("beta") else 7.0) for k in s.parameters()}
        assert set(s2.parameters()) == set(s.parameters())
        _same(s2, s, p)
    with pytest.raises(ValueError):
        Design.from_structure(builders.bottlebrush(300), max_units=100)


def test_auto_layout_spreads_nodes():
    d = Design.from_structure(builders.dendrimer(3))
    pos = {(round(n.x), round(n.y)) for n in d.nodes.values()}
    assert len(pos) == len(d.nodes)
