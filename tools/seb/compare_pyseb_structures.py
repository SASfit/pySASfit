"""Compare composite structures with pyseb 0.4.0 (Python 3.12 + pyseb)."""
import sys, time, numpy as np
sys.path.insert(0, "src")
import pyseb
from pysasfit.seb import Structure
from pysasfit.seb.subunits import Spheroid

q = np.logspace(-3, np.log10(0.6), 60)

def build(ops, spheroid_param=True):
    w = pyseb.World(); s = Structure()
    gid = None
    for op in ops:
        if op[0] == "add":
            _, kind, name, tag = op
            g = w.Add(kind, name, tag)
            gid = g if gid is None else gid
            model = Spheroid(surface_measure="parametric", n_mu=200, n_theta=128) if kind == "Spheroid" else None
            s.add(model or kind, name, tag=tag)
        else:
            _, kind, new, old, tag = op
            w.Link(kind, new, old, tag)
            model = Spheroid(surface_measure="parametric", n_mu=200, n_theta=128) if kind == "Spheroid" else None
            s.link(model or kind, new, old, tag=tag)
    w.Add(gid, "S")
    return w, s

def case(name, ops, params):
    w, s = build(ops)
    t = time.perf_counter(); ref = np.array(w.EvaluateFormFactor("S", params, list(q))); t1 = time.perf_counter() - t
    t = time.perf_counter(); mine = s.form_factor(q, params); t2 = time.perf_counter() - t
    r = np.max(np.abs(mine - ref) / np.abs(ref))
    print(f"{name:42s} N={len(s._units):4d}  max rel diff {r:.1e}   pyseb {t1*1e3:7.1f} ms  ours {t2*1e3:6.1f} ms"
          + ("   <-- DIFF" if r > 1e-6 else ""))

L = lambda kind, new, old, tag: ("link", kind, new, old, tag)
A = lambda kind, name, tag: ("add", kind, name, tag)

case("diblock (different beta, Rg)",
     [A("GaussianPolymer", "p1", "p1"), L("GaussianPolymer", "p2.end1", "p1.end2", "p2")],
     {"beta_p1": 1.0, "beta_p2": 0.6, "Rg_p1": 5.0, "Rg_p2": 9.0})

ops = [A("SolidSphere", "core", "core")] + [L("GaussianPolymer", f"c{i}.end1", f"core.surface#a{i}", "ch") for i in range(8)]
case("micelle, 8 chains at random surface points", ops, {"beta_core": 3.0, "beta_ch": 0.4, "R_core": 25.0, "Rg_ch": 10.0})
ops = [A("SolidSphere", "core", "core")] + [L("GaussianPolymer", f"c{i}.end1", "core.surface#x", "ch") for i in range(5)]
case("micelle, 5 chains at one surface point", ops, {"beta_core": 3.0, "beta_ch": 0.4, "R_core": 25.0, "Rg_ch": 10.0})

ops = [A("SolidSphericalShell", "sh", "sh"),
       L("GaussianPolymer", "o1.end1", "sh.surfaceo#a", "o"), L("GaussianPolymer", "o2.end1", "sh.surfaceo#b", "o"),
       L("GaussianPolymer", "i1.end1", "sh.surfacei#a", "i"), L("GaussianPolymer", "s1.end1", "sh.surface#a", "s"),
       L("GaussianLoop", "lp.contour#z", "sh.surfaceo#c", "lp")]
case("solid shell: chains on outer/inner/any surface + loop", ops,
     {"beta_sh": 2.0, "Ri_sh": 15.0, "Ro_sh": 20.0, "beta_o": .5, "Rg_o": 8.0, "beta_i": .3, "Rg_i": 4.0,
      "beta_s": .4, "Rg_s": 6.0, "beta_lp": .7, "Rg_lp": 7.0})

ops = [A("ThinSphericalShell", "ves", "ves")] + [L("GaussianLoop", f"l{i}.contour#a", f"ves.surface#p{i}", "lp") for i in range(4)] \
      + [L("GaussianPolymer", "t.end1", "l0.contour#b", "t")]
case("vesicle + loops + tail on a loop", ops, {"beta_ves": 2.0, "R_ves": 30.0, "beta_lp": 1.0, "Rg_lp": 8.0, "beta_t": .5, "Rg_t": 5.0})

ops = [A("ThinRod", "r0", "rod")] + [L("ThinRod", f"r{i}.end1", "r0.end1", "rod") for i in (1, 2)] \
      + [L("GaussianPolymer", f"g{i}.end1", f"r{i%3}.contour#c{i}", "g") for i in range(6)] \
      + [L("GaussianPolymer", "m.end1", "r1.middle", "g")]
case("rod star + chains on rod contours/middle", ops, {"beta_rod": 1.0, "L_rod": 50.0, "beta_g": .5, "Rg_g": 7.0})

ops = [A("SolidCylinder", "cyl", "cyl"),
       L("GaussianPolymer", "h1.end1", "cyl.hull#a", "g"), L("GaussianPolymer", "h2.end1", "cyl.hull#b", "g"),
       L("GaussianPolymer", "e1.end1", "cyl.ends#a", "g"), L("GaussianPolymer", "s1.end1", "cyl.surface#a", "g"),
       L("GaussianPolymer", "t1.end1", "cyl.top", "g"), L("GaussianPolymer", "b1.end1", "cyl.bottom", "g")]
case("cylinder + chains on hull/ends/surface/top/bottom", ops, {"beta_cyl": 4.0, "R_cyl": 10.0, "L_cyl": 60.0, "beta_g": .5, "Rg_g": 8.0})

ops = [A("ThinDisk", "d", "d"), L("GaussianPolymer", "a.end1", "d.rim#a", "g"), L("GaussianPolymer", "b.end1", "d.rim#b", "g"),
       L("GaussianPolymer", "c.end1", "d.surface#a", "g"), L("GaussianPolymer", "e.end1", "d.center", "g"),
       L("ThinCircle", "ring.contour#a", "a.end2", "ring"), L("GaussianPolymer", "f.end1", "ring.center", "g")]
case("disk (rim/surface/center) + ring", ops, {"beta_d": 2.0, "R_d": 20.0, "beta_g": .5, "Rg_g": 6.0, "beta_ring": 1.0, "R_ring": 12.0})

ops = [A("Spheroid", "sp", "sp"), L("GaussianPolymer", "n.end1", "sp.north", "g"), L("GaussianPolymer", "s.end1", "sp.south", "g"),
       L("GaussianPolymer", "x.end1", "sp.surface#a", "g"), L("GaussianPolymer", "y.end1", "sp.surface#b", "g")]
case("spheroid + chains at poles and surface", ops, {"beta_sp": 3.0, "a_sp": 10.0, "c_sp": 25.0, "beta_g": .5, "Rg_g": 6.0})
ops = [A("Spheroid", "sp", "sp"), L("GaussianPolymer", "n.end1", "sp.center", "g"), L("GaussianPolymer", "x.end1", "sp.surface#a", "g")]
case("spheroid: chain at centre + at surface (Psi c-s)", ops, {"beta_sp": 3.0, "a_sp": 10.0, "c_sp": 25.0, "beta_g": .5, "Rg_g": 6.0})

ops = [A("Point", "pt", "pt")] + [L("GaussianPolymer", f"a{i}.end1", "pt.point", "arm") for i in range(6)]
case("6-arm star on a Point", ops, {"beta_arm": 1.0, "Rg_arm": 10.0})

# dendrimer, 3 generations x functionality 3, generation-dependent tags
ops = [A("GaussianPolymer", "g0x0", "g0")] + [L("GaussianPolymer", f"g0x{i}.end1", "g0x0.end1", "g0") for i in (1, 2)]
free = ["g0x0", "g0x1", "g0x2"]
for g in (1, 2, 3):
    new = []
    for k, par in enumerate(free):
        for j in range(2):
            nm = f"g{g}x{2*k+j}"; ops.append(L("GaussianPolymer", f"{nm}.end1", f"{par}.end2", f"g{g}")); new.append(nm)
    free = new
case("dendrimer f=3, 4 generations", ops, {"beta_g0": 1, "beta_g1": 1, "beta_g2": 1, "beta_g3": 1,
                                             "Rg_g0": 5, "Rg_g1": 4, "Rg_g2": 3, "Rg_g3": 2})

ops = [A("GaussianPolymer", "bb0", "bb")] + [L("GaussianPolymer", f"bb{i}.end1", f"bb{i-1}.end2", "bb") for i in range(1, 30)] \
      + [L("GaussianPolymer", f"sc{i}.end1", f"bb{i}.end2", "sc") for i in range(30)]
case("bottle brush 30 + 30 side chains", ops, {"beta_bb": 1.0, "Rg_bb": 3.0, "beta_sc": 0.8, "Rg_sc": 6.0})
