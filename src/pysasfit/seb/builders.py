"""Convenience builders for common polymer architectures."""
from .structure import Structure


def linear_chain(n, kind="GaussianPolymer", tag="block", name="chain"):
    """n sub-units joined end2 -> end1."""
    s = Structure(name)
    s.add(kind, "b0", tag=tag)
    for i in range(1, n):
        s.link(kind, f"b{i}.end1", f"b{i-1}.end2", tag=tag)
    return s


def star(f, kind="GaussianPolymer", tag="arm", name="star"):
    """f arms joined at their end1 (a common point)."""
    s = Structure(name)
    s.add(kind, "arm0", tag=tag)
    for i in range(1, f):
        s.link(kind, f"arm{i}.end1", "arm0.end1", tag=tag)
    return s


def micelle(n_chains, core="SolidSphere", chain="GaussianPolymer",
            same_point=False, name="micelle"):
    """Core with n_chains attached by end1 at independent random surface
    points (or all at one point if same_point)."""
    s = Structure(name)
    s.add(core, "core")
    for i in range(n_chains):
        lab = "a" if same_point else f"a{i}"
        s.link(chain, f"chain{i}.end1", f"core.surface#{lab}", tag="chain")
    return s


def dendrimer(generations, functionality=3, kind="GaussianPolymer", name="dendrimer"):
    """Dendrimer: `functionality` arms from the centre; each further
    generation adds functionality-1 branches at every free end.
    Generation g sub-units have tag 'g{g}'."""
    s = Structure(name)
    free = []
    s.add(kind, "g0_0", tag="g0")
    free.append("g0_0")
    for i in range(1, functionality):
        s.link(kind, f"g0_{i}.end1", "g0_0.end1", tag="g0")
        free.append(f"g0_{i}")
    for g in range(1, generations):
        new, k = [], 0
        for parent in free:
            for _ in range(functionality - 1):
                nm = f"g{g}_{k}"
                s.link(kind, f"{nm}.end1", f"{parent}.end2", tag=f"g{g}")
                new.append(nm)
                k += 1
        free = new
    return s


def bottlebrush(n_backbone, kind_backbone="GaussianPolymer", kind_side="GaussianPolymer",
                name="bottlebrush"):
    """Backbone of n segments, one side chain grafted at every junction
    (end2 of each backbone segment)."""
    s = Structure(name)
    s.add(kind_backbone, "bb0", tag="bb")
    for i in range(1, n_backbone):
        s.link(kind_backbone, f"bb{i}.end1", f"bb{i-1}.end2", tag="bb")
    for i in range(n_backbone):
        s.link(kind_side, f"sc{i}.end1", f"bb{i}.end2", tag="sc")
    return s
