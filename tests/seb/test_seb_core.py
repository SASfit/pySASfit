import numpy as np
import pytest
from pysasfit.seb import Structure, builders
from pysasfit.seb import special as sf
from pysasfit.seb.subunits import REGISTRY, ThinDisk, Spheroid, SolidSphere, OrientedBody

q = np.logspace(-4, 0, 150)
SIZES = {"nu": 0.588, "b": 15.0, "Rg": 9.0, "R": 12.0, "L": 40.0, "Ri": 8.0, "Ro": 15.0, "a": 10.0, "c": 22.0}


@pytest.mark.parametrize("kind", [k for k in REGISTRY if k != "Point"])
def test_subunit_limits(kind):
    m = REGISTRY[kind]()
    p = {s: SIZES[s] for s in m.size_params}
    q0 = np.array([1e-7])
    assert abs(m.ff(q0, p)[0] - 1) < 1e-9
    for r in m.references():
        assert abs(m.amp(q0, p, r)[0] - 1) < 1e-9
        for r2 in m.references():
            assert abs(m.psi(q0, p, r, r2)[0] - 1) < 1e-9
            np.testing.assert_allclose(m.psi(q, p, r, r2), m.psi(q, p, r2, r), atol=1e-14)


def test_disk_closed_forms_match_quadrature():
    m, p = ThinDisk(n_mu=400), {"R": 20.0}
    Q, M = m._grid(q)
    np.testing.assert_allclose(m.ff(q, p), m._avg(m._phi(Q, M, p) ** 2), atol=1e-9)
    np.testing.assert_allclose(m.amp(q, p, "center"), m._avg(m._phi(Q, M, p)), atol=1e-9)
    g = m._g("rim", Q, M, p)
    np.testing.assert_allclose(m.psi(q, p, "rim", "rim"), m._avg(g * g), atol=1e-6)


@pytest.mark.parametrize("measure", ["area", "parametric"])
def test_spheroid_sphere_limit(measure):
    sp, s = Spheroid(surface_measure=measure), SolidSphere()
    p, ps = {"a": 15.0, "c": 15.0}, {"R": 15.0}
    np.testing.assert_allclose(sp.ff(q, p), s.ff(q, ps), atol=1e-12)
    np.testing.assert_allclose(sp.amp(q, p, "surface"), s.amp(q, ps, "surface"), atol=1e-10)
    np.testing.assert_allclose(sp.psi(q, p, "surface", "surface"), s.psi(q, ps, "surface", "surface"), atol=1e-10)
    np.testing.assert_allclose(sp.psi(q, p, "center", "surface"), s.psi(q, ps, "center", "surface"), atol=1e-10)


def _pars(s, v=7.0):
    return {k: (1.3 if k.startswith("beta") else v) for k in s.parameters()}


@pytest.mark.parametrize("s", [builders.micelle(7), builders.micelle(4, same_point=True),
                               builders.dendrimer(4), builders.bottlebrush(10), builders.star(6),
                               builders.linear_chain(5, kind="ThinRod")])
def test_tree_equals_pair_sum(s):
    p = _pars(s)
    np.testing.assert_allclose(s.form_factor(q, p), s.form_factor_pairs(q, p), rtol=1e-12)
    assert abs(s.form_factor([1e-9], p)[0] - 1) < 1e-9


def test_linear_gaussian_chain_is_debye():
    s = builders.linear_chain(6)
    F = s.form_factor(q, {"Rg_block": 4.0, "beta_block": 1.0})
    np.testing.assert_allclose(F, sf.debye(q**2 * 6 * 16.0), rtol=1e-10)


def test_star_rg2():
    f, rg = 5, 7.0
    s = builders.star(f)
    assert s.radius_of_gyration2({"Rg_arm": rg, "beta_arm": 1}) == pytest.approx((3 * f - 2) / f * rg**2, rel=1e-6)


def test_freely_jointed_rod_dimer():
    # joints are freely rotating: F = (2 F_rod + 2 (Si(x)/x)^2) / 4
    s = builders.linear_chain(2, kind="ThinRod")
    F = s.form_factor(q, {"L_block": 10.0, "beta_block": 1.0})
    x = q * 10.0
    np.testing.assert_allclose(F, (2 * sf.thin_rod_ff(x) + 2 * sf.si_over_x(x) ** 2) / 4, atol=1e-13)


def test_loops_and_errors():
    s = Structure()
    s.add("GaussianPolymer", "a")
    s.link("GaussianPolymer", "b.end1", "a.end2")
    s.link("b.end2", "a.end1")          # closes a loop
    with pytest.raises(ValueError):
        s.form_factor(q, {"Rg_a": 1, "Rg_b": 1})
    with pytest.raises(KeyError):
        Structure().add("NoSuchThing", "x")
    s2 = Structure(); s2.add("SolidSphere", "c")
    with pytest.raises(KeyError):
        s2.link("GaussianPolymer", "g.end1", "c.top")


def test_include_hierarchy():
    arm = builders.linear_chain(2)
    s = Structure()
    s.include(arm, "A"); s.include(arm, "B")
    s.link("Ab0.end1", "Bb0.end1")
    # two arms of 2 blocks each, joined at the start = linear chain of 4
    F = s.form_factor(q, {"Rg_block": 3.0})
    np.testing.assert_allclose(F, sf.debye(q**2 * 4 * 9.0), rtol=1e-10)
