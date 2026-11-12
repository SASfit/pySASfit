"""pySEBpolymers against pyseb 0.4 (SEB Python bindings).  Skipped unless pyseb is
installed (wheels only for CPython <= 3.12).  Full tables: tools/seb/compare_pyseb_*.py."""
import numpy as np
import pytest

pyseb = pytest.importorskip("pyseb")
pytestmark = pytest.mark.pyseb

from pysasfit.seb import Structure  # noqa: E402

q = np.logspace(-2, np.log10(0.6), 30)


def _both(ops):
    w, s, gid = pyseb.World(), Structure(), None
    for op in ops:
        if op[0] == "add":
            g = w.Add(op[1], op[2], op[3]); gid = g if gid is None else gid
            s.add(op[1], op[2], tag=op[3])
        else:
            w.Link(*op[1:]); s.link(op[1], op[2], op[3], tag=op[4])
    w.Add(gid, "S")
    return w, s


CASES = {
    "diblock": ([("add", "GaussianPolymer", "p1", "p1"), ("link", "GaussianPolymer", "p2.end1", "p1.end2", "p2")],
                {"beta_p1": 1.0, "beta_p2": 0.6, "Rg_p1": 5.0, "Rg_p2": 9.0}),
    "micelle": ([("add", "SolidSphere", "core", "core")]
                + [("link", "GaussianPolymer", f"c{i}.end1", f"core.surface#a{i}", "ch") for i in range(6)],
                {"beta_core": 3.0, "beta_ch": 0.4, "R_core": 25.0, "Rg_ch": 10.0}),
    "shell": ([("add", "SolidSphericalShell", "sh", "sh"),
               ("link", "GaussianPolymer", "o1.end1", "sh.surfaceo#a", "o"),
               ("link", "GaussianPolymer", "i1.end1", "sh.surfacei#a", "o"),
               ("link", "GaussianLoop", "lp.contour#z", "sh.surface#c", "lp")],
              {"beta_sh": 2.0, "Ri_sh": 15.0, "Ro_sh": 20.0, "beta_o": .5, "Rg_o": 8.0, "beta_lp": .7, "Rg_lp": 7.0}),
    "rods": ([("add", "ThinRod", "r0", "rod"), ("link", "ThinRod", "r1.end1", "r0.end1", "rod"),
              ("link", "GaussianPolymer", "g.end1", "r1.contour#c", "g"),
              ("link", "GaussianPolymer", "m.end1", "r0.middle", "g")],
             {"beta_rod": 1.0, "L_rod": 50.0, "beta_g": .5, "Rg_g": 7.0}),
    "cylinder": ([("add", "SolidCylinder", "cyl", "cyl"),
                  ("link", "GaussianPolymer", "h.end1", "cyl.hull#a", "g"),
                  ("link", "GaussianPolymer", "e.end1", "cyl.ends#a", "g"),
                  ("link", "GaussianPolymer", "t.end1", "cyl.top", "g")],
                 {"beta_cyl": 4.0, "R_cyl": 10.0, "L_cyl": 60.0, "beta_g": .5, "Rg_g": 8.0}),
    "disk_ring": ([("add", "ThinDisk", "d", "d"), ("link", "GaussianPolymer", "a.end1", "d.rim#a", "g"),
                   ("link", "ThinCircle", "ring.contour#a", "a.end2", "ring")],
                  {"beta_d": 2.0, "R_d": 20.0, "beta_g": .5, "Rg_g": 6.0, "beta_ring": 1.0, "R_ring": 12.0}),
    "star_point": ([("add", "Point", "pt", "pt")]
                   + [("link", "GaussianPolymer", f"a{i}.end1", "pt.point", "arm") for i in range(4)],
                   {"beta_arm": 1.0, "Rg_arm": 10.0}),
}


@pytest.mark.parametrize("name", list(CASES))
def test_against_pyseb(name):
    ops, params = CASES[name]
    w, s = _both(ops)
    ref = np.array(w.EvaluateFormFactor("S", params, list(q)))
    np.testing.assert_allclose(s.form_factor(q, params), ref, rtol=1e-6)
