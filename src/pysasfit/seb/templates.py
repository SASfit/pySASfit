"""Ready-made structures of the pySEBpolymers panel.

Every template turns a few options into a design (:mod:`pysasfit.seb.graph`, shown in the designer) and thereby into
Python code that builds a :class:`~pysasfit.seb.Structure` named ``s``. :func:`build` runs such code.
"""
from __future__ import annotations

import numpy as np

from .structure import Structure
from . import builders

__all__ = ["TEMPLATES", "Option", "design_for", "code_for", "build", "default_value", "CORE_REF", "CHAIN_REF"]

#: reference point of a core at which chains are grafted (random point on its outer surface)
CORE_REF = {"SolidSphere": "surface", "ThinSphericalShell": "surface", "SolidSphericalShell": "surfaceo",
            "Spheroid": "surface", "SolidCylinder": "surface", "ThinDisk": "surface"}
#: reference point of a chain at which it is grafted
CHAIN_REF = {"GaussianPolymer": "end1", "ThinRod": "end1", "GaussianLoop": "contour"}
LINEAR = ("GaussianPolymer", "ThinRod")


class Option:
    """One option of a template: kind "int", "bool" or "choice"."""

    def __init__(self, key, label, default, kind="int", lo=1, hi=100000, choices=(), tip=""):
        self.key, self.label, self.default, self.kind = key, label, default, kind
        self.lo, self.hi, self.choices, self.tip = lo, hi, tuple(choices), tip


def _micelle(o):
    from .graph import Design
    d = Design("micelle")
    core = d.add_unit(o["core"], "core", tag="core")
    arr = d.add_pattern("Array", o["chain"], name="chains", tag="chain", n=o["n"],
                        mode="same" if o["same_point"] else "independent", in_ref=CHAIN_REF[o["chain"]])
    d.link(core.id, CORE_REF[o["core"]], arr.id, "anchor", "a")
    return d


def _star(o):
    from .graph import Design
    d = Design("star")
    c = d.add_unit("Point", "centre", tag="centre")
    arr = d.add_pattern("Array", o["arm"], name="arms", tag="arm", n=o["n"], mode="same", in_ref="end1")
    d.link(c.id, "point", arr.id, "anchor")
    return d


def _linear(o):
    from .graph import Design
    d = Design("chain")
    if o["alternate"]:
        c = d.new_component("diblock")
        a = c.add_unit(o["block"], "A", tag="A")
        b = c.add_unit(o["block"], "B", tag="B", x=230)
        c.link(a.id, "end2", b.id, "end1")
        c.set_port("in", a.id, "end1")
        c.set_port("out", b.id, "end2")
        d.add_pattern("Chain", "diblock", component=True, name="blocks", n=o["n"])
    else:
        d.add_pattern("Chain", o["block"], name="blocks", tag="block", n=o["n"], in_ref="end1", out_ref="end2")
    return d


def _dendrimer(o):
    from .graph import Design
    d = Design("dendrimer")
    d.add_pattern("Generations", o["kind"], name="dendron", tag="g", f=o["functionality"], g=o["generations"],
                  gen_tags=True, in_ref="end1", out_ref="end2")
    return d


def _brush(o):
    from .graph import Design
    d = Design("bottlebrush")
    c = d.new_component("segment")
    bb = c.add_unit(o["backbone"], "bb", tag="bb")
    sc = c.add_unit(o["side"], "sc", tag="sc", x=230)
    c.link(bb.id, "end2", sc.id, "end1")
    c.set_port("in", bb.id, "end1")
    c.set_port("out", bb.id, "end2")
    d.add_pattern("Chain", "segment", component=True, name="backbone", n=o["n"])
    return d


def _rodcoil(o):
    from .graph import Design
    d = Design("rodcoil")
    rod = d.add_unit("ThinRod", "rod", tag="rod")
    c1 = d.add_unit("GaussianPolymer", "coil1", tag="coil")
    d.link(rod.id, "end1", c1.id, "end1")
    if o["n"] == 2:
        c2 = d.add_unit("GaussianPolymer", "coil2", tag="coil")
        d.link(rod.id, "end2", c2.id, "end1")
    return d


def _vesicle(o):
    from .graph import Design
    d = Design("vesicle")
    v = d.add_unit("ThinSphericalShell", "ves", tag="ves")
    arr = d.add_pattern("Array", "GaussianLoop", name="loops", tag="loop", n=o["n"], in_ref="contour")
    d.link(v.id, "surface", arr.id, "anchor", "p")
    return d


def _custom(_o):
    from .graph import Design
    d = Design("custom")
    cyl = d.add_unit("SolidCylinder", "cyl", tag="cyl")
    top = d.add_unit("GaussianPolymer", "top", tag="chain")
    bot = d.add_unit("GaussianPolymer", "bottom", tag="chain")
    d.link(cyl.id, "top", top.id, "end1")
    d.link(cyl.id, "bottom", bot.id, "end1")
    arr = d.add_pattern("Array", "GaussianPolymer", name="hull", tag="chain", n=6, in_ref="end1")
    d.link(cyl.id, "hull", arr.id, "anchor", "h")
    return d


_CORES = tuple(CORE_REF)
TEMPLATES = {
    "Micelle (core + grafted chains)": (_micelle, [
        Option("n", "chains", 20, lo=1, hi=5000),
        Option("core", "core", "SolidSphere", "choice", choices=_CORES),
        Option("chain", "chains are", "GaussianPolymer", "choice", choices=tuple(CHAIN_REF)),
        Option("same_point", "all chains at one point", False, "bool")]),
    "Star polymer": (_star, [
        Option("n", "arms", 6, lo=1, hi=5000),
        Option("arm", "arms are", "GaussianPolymer", "choice", choices=LINEAR)]),
    "Linear / multiblock chain": (_linear, [
        Option("n", "repeats", 4, lo=1, hi=5000, tip="blocks, or A-B diblocks when alternating"),
        Option("block", "blocks are", "GaussianPolymer", "choice", choices=LINEAR),
        Option("alternate", "alternating A/B blocks", True, "bool")]),
    "Dendrimer": (_dendrimer, [
        Option("generations", "generations", 4, lo=1, hi=12),
        Option("functionality", "functionality", 3, lo=2, hi=8),
        Option("kind", "branches are", "GaussianPolymer", "choice", choices=LINEAR)]),
    "Bottle brush": (_brush, [
        Option("n", "backbone segments", 30, lo=1, hi=5000),
        Option("backbone", "backbone", "GaussianPolymer", "choice", choices=LINEAR),
        Option("side", "side chains", "GaussianPolymer", "choice", choices=LINEAR)]),
    "Rod–coil block copolymer": (_rodcoil, [Option("n", "coils", 1, lo=1, hi=2)]),
    "Vesicle with loops": (_vesicle, [Option("n", "loops", 10, lo=1, hi=5000)]),
    "Custom": (_custom, []),
}


def design_for(template, options=None):
    """Design (pysasfit.seb.graph) of ``template`` with ``options`` (missing ones: defaults)."""
    fn, opts = TEMPLATES[template]
    o = {op.key: op.default for op in opts}
    o.update(options or {})
    d = fn(o)
    d.auto_layout()
    for c in d.components.values():
        c.auto_layout()
    return d


def code_for(template, options=None):
    """Python code of ``template`` with ``options`` (missing ones: defaults)."""
    return design_for(template, options).to_code()


def build(code):
    """Run structure code; it must assign a Structure to ``s``. Returns it."""
    ns = {"Structure": Structure, "builders": builders, "np": np}
    exec(compile(code, "<structure>", "exec"), ns)
    s = ns.get("s")
    if not isinstance(s, Structure):
        raise ValueError("the code must assign a Structure to the variable s")
    s._graph()                       # loops, self-links: report now, not at the first evaluation
    return s


_DEFAULTS = {"beta": 1.0, "Rg": 20.0, "R": 40.0, "L": 100.0, "Ri": 30.0, "Ro": 40.0, "a": 30.0, "c": 60.0}


def default_value(name):
    """Start value of a structure parameter (lengths in the inverse unit of q, Å for q in 1/Å)."""
    return _DEFAULTS.get(name.split("_", 1)[0], 1.0)
