"""Worm-like and excluded-volume chains, polydisperse and thick sub-units, excluded volume between copies."""
import numpy as np
import pytest

from pysasfit.seb import Structure
from pysasfit.seb import special as sf
from pysasfit.seb import templates as T
from pysasfit.seb.chains import (ChainFromF, ExcludedVolumePolymer, WormLikeChain, WormLikeChainEV,
                                 kholodenko_r2, kholodenko_rg2)
from pysasfit.seb.graph import Design
from pysasfit.seb.ps_worm import sk_wr

q = np.logspace(-4, 0.5, 120)


def _guinier(model, p, q0):
    qq = np.array([q0, q0 / 2])
    g = 3 * (1 - model.ff(qq, p)) / qq**2
    return (4 * g[1] - g[0]) / 3


# ------------------------------------------------------------------ worm-like chain (Kholodenko)
def test_worm_rod_and_coil_limits():
    w = WormLikeChain()
    p = {"L": 100.0, "b": 1e6}
    np.testing.assert_allclose(w.ff(q, p), sf.thin_rod_ff(q * 100), atol=1e-6)
    np.testing.assert_allclose(w.amp(q, p, "end1"), sf.si_over_x(q * 100), atol=1e-6)
    np.testing.assert_allclose(w.amp(q, p, "middle"), sf.si_over_x(q * 50), atol=1e-6)
    np.testing.assert_allclose(w.psi(q, p, "end1", "end2"), sf.sinc(q * 100), atol=1e-6)
    p = {"L": 1e5, "b": 1.0}                       # Gaussian chain, Rg^2 = L b / 6
    x = q**2 * 1e5 / 6
    np.testing.assert_allclose(w.ff(q, p), sf.debye(x), atol=2e-5)
    np.testing.assert_allclose(w.psi(q, p, "end1", "end2"), np.exp(-x), atol=2e-5)


@pytest.mark.parametrize("L,b", [(100.0, 20.0), (1000.0, 10.0), (50.0, 500.0)])
def test_worm_guinier_and_end_to_end(L, b):
    w = WormLikeChain()
    p = {"L": L, "b": b}
    q0 = 1e-4 / max(1.0, np.sqrt(kholodenko_rg2(L, b)) / 10)
    assert _guinier(w, p, q0) == pytest.approx(kholodenko_rg2(L, b), rel=1e-5)
    qq = np.array([q0, q0 / 2])
    e = 6 * (1 - w.psi(qq, p, "end1", "end2")) / qq**2
    assert (4 * e[1] - e[0]) / 3 == pytest.approx(float(kholodenko_r2(L, b)), rel=1e-5)


def test_l_derivative_relations_reproduce_the_two_point_function():
    """ChainFromF applied to Kholodenko's F gives the amplitudes and phases computed from G(s) directly."""
    class KF(ChainFromF):
        kind, size_params = "x", ("L", "b")

        def length(self, p):
            return p["L"]

        def F_of_L(self, q, L, p, ref=None):
            return WormLikeChain().ff(q, {"L": L, "b": p["b"]})
    w, k = WormLikeChain(), KF()
    for p in ({"L": 300.0, "b": 30.0}, {"L": 1000.0, "b": 10.0}):
        for r in ("end1", "middle"):
            np.testing.assert_allclose(k.amp(q, p, r), w.amp(q, p, r), atol=1e-5)
        np.testing.assert_allclose(k.psi(q, p, "end1", "end2"), w.psi(q, p, "end1", "end2"), atol=2e-5)
        np.testing.assert_allclose(k.psi(q, p, "end1", "middle"), w.psi(q, p, "end1", "middle"), atol=2e-5)


# ------------------------------------------------------------------ worm-like chain with excluded volume (PS)
SASMODELS = {   # flexible_cylinder of sasmodels 1.1.0, radius -> 0, normalised at q = 1e-7
    (1000.0, 20.0): [0.998437944945, 0.860184726725, 0.351330523569, 0.048370126872, 0.010740073816, 0.00314350706],
    (300.0, 100.0): [0.998892908676, 0.897826756282, 0.456283646573, 0.105488843074, 0.03490984036, 0.01047198208],
    (50.0, 40.0): [0.999954317409, 0.9954471943, 0.96012149075, 0.629328839109, 0.209549815027, 0.062832164517],
}


@pytest.mark.parametrize("Lb", list(SASMODELS))
def test_pedersen_schurtenberger_equals_sasmodels(Lb):
    qq = np.array([0.001, 0.01, 0.03, 0.1, 0.3, 1.0])
    np.testing.assert_allclose(sk_wr(qq, *Lb), SASMODELS[Lb], rtol=2e-6)
    if Lb[0] / Lb[1] >= WormLikeChainEV.BLEND[1]:      # below: crossover to Kholodenko (PS invalid for L < b)
        np.testing.assert_allclose(WormLikeChainEV().ff(qq, {"L": Lb[0], "b": Lb[1]}), SASMODELS[Lb], rtol=2e-6)


def test_worm_ev_amplitudes_are_sane():
    m = WormLikeChainEV()
    for p in ({"L": 1000.0, "b": 20.0}, {"L": 300.0, "b": 100.0}, {"L": 5000.0, "b": 50.0}):
        A = m.amp(q, p, "end1")
        assert abs(m.amp(np.array([1e-7]), p, "end1")[0] - 1) < 1e-6
        assert np.all(np.diff(A) < 1e-9) and A[-1] > 0
        P = m.psi(q, p, "end1", "end2")
        assert abs(m.psi(np.array([1e-7]), p, "end1", "end2")[0] - 1) < 1e-6
        assert np.all(P > -0.15) and np.all(P <= 1 + 1e-9)


# ------------------------------------------------------------------ excluded-volume chain (des Cloizeaux)
def test_ev_chain_gaussian_limit_and_rg():
    m = ExcludedVolumePolymer()
    p = {"Rg": 10.0, "nu": 0.5}
    np.testing.assert_allclose(m.ff(q, p), sf.debye((q * 10) ** 2), atol=1e-8)
    np.testing.assert_allclose(m.amp(q, p, "end1"), sf.one_minus_exp_over_x((q * 10) ** 2), atol=1e-8)
    np.testing.assert_allclose(m.psi(q, p, "end1", "end2"), np.exp(-(q * 10) ** 2), atol=1e-8)
    for nu in (0.588, 0.7):
        assert _guinier(m, {"Rg": 10.0, "nu": nu}, 1e-3) == pytest.approx(100.0, rel=1e-5)


def test_ev_chain_lies_between_flory_and_saw():
    """At qRg = 3 the des Cloizeaux chain is below the Gaussian-distance (Flory) chain, as self-avoiding walks."""
    from pysasfit.seb.subunits import FloryPolymer
    p = {"Rg": 10.0, "nu": 0.588}
    qq = np.array([0.3])
    assert ExcludedVolumePolymer().ff(qq, p)[0] < FloryPolymer().ff(qq, p)[0]


# ------------------------------------------------------------------ polydisperse and thick
@pytest.mark.parametrize("sigma", [0.05, 0.3, 1.0])
def test_polydisperse_gaussian_chain_is_zimm(sigma):
    s = Structure()
    s.add("GaussianPolymer", "c", poly=True)
    F = s.form_factor(q, {"Rg_c": 10.0, "sigma_c": sigma, "beta_c": 1.0})
    U, x = sigma**2, (q * 10) ** 2
    zimm = 2 * ((1 + U * x) ** (-1 / U) + x - 1) / ((1 + U) * x**2)
    ok = x > 0.05                             # the closed form cancels at small x
    np.testing.assert_allclose(F[ok], zimm[ok], atol=1e-5)


def test_polydisperse_sphere_by_direct_average():
    from scipy.special import gammaln
    s = Structure()
    s.add("SolidSphere", "s", poly=True)
    sig = 0.25
    F = s.form_factor(q, {"R_s": 30.0, "sigma_s": sig, "beta_s": 1.0})
    k = 1 / sig**2
    m = np.linspace(1e-4, 4, 20001)
    pdf = np.exp((k - 1) * np.log(m) - k * m + k * np.log(k) - gammaln(k))
    R = 30.0 * m ** (1 / 3)
    num = np.trapezoid(pdf * m**2 * sf.sphere_amp(np.outer(q, R)) ** 2, m, axis=1)
    np.testing.assert_allclose(F, num / (1 + sig**2), rtol=1e-4, atol=5e-7)


def test_thick_rod_and_options_errors():
    s = Structure()
    s.add("ThinRod", "r", thick=True)
    np.testing.assert_allclose(s.form_factor(q, {"L_r": 100.0, "Rc_r": 5.0}),
                               sf.thin_rod_ff(q * 100) * sf.jinc(q * 5) ** 2, atol=1e-14)
    with pytest.raises(ValueError):
        Structure().add("SolidSphere", "x", thick=True)
    with pytest.raises(ValueError):
        Structure().add("Point", "x", poly=True)


# ------------------------------------------------------------------ excluded volume between copies
def _crowded():
    s = Structure("t")
    s.add("SolidSphere", "core")
    s.interact("chains")
    s.interact("hairs")
    s.interact("worms")
    for i in range(4):
        s.link("FloryPolymer", f"c{i}.end1", f"core.surface#a{i}", tag="chain", thick=True,
               copy=[("chains", i, "chains")])
        for j in range(3):
            s.link("ThinRod", f"c{i}h{j}.end1", f"c{i}.contour#h{j}", tag="hair",
                   copy=[("chains", i, "chains"), (f"c{i}hairs", j, "hairs")])
    for k in range(3):
        s.link("WormLikeChainEV", f"w{k}.end1", "core.center", tag="worm", poly=True, copy=[("worms", k, "worms")])
    s.link("GaussianPolymer", "tail.end1", "c0.end2", tag="tail", poly=True)
    return s


@pytest.mark.parametrize("phi", [0.0, 0.5, 0.99])
def test_copies_correlation_hole_tree_equals_pairs_sum_rule_positive(phi):
    s = _crowded()
    p = {"R_core": 20, "beta_core": 3, "Rg_chain": 8, "nu_chain": 0.6, "Rc_chain": 1.5, "beta_chain": 1,
         "L_hair": 10, "beta_hair": 0.3, "L_worm": 200, "b_worm": 20, "sigma_worm": 0.3, "beta_worm": 0.5,
         "Rg_tail": 5, "sigma_tail": 0.4, "beta_tail": 0.2, "vex_chains": phi, "vex_hairs": phi, "vex_worms": phi}
    F = s.form_factor(q, p)
    np.testing.assert_allclose(F, s.form_factor_pairs(q, p), rtol=1e-9)
    assert s.form_factor([1e-7], p)[0] == pytest.approx(1.0, abs=1e-6)
    assert F.min() > 0
    with pytest.raises(ValueError):
        s.form_factor(q, dict(p, vex_chains=1.0))


def test_correlation_hole_lowers_the_interference():
    s = Structure()
    s.add("SolidSphere", "core")
    s.interact("ch")
    for i in range(40):
        s.link("GaussianPolymer", f"c{i}.end1", f"core.surface#a{i}", tag="ch", copy=[("ch", i, "ch")])
    p = {"R_core": 30.0, "beta_core": 20.0, "Rg_ch": 15.0, "beta_ch": 1.0}
    F0 = s.form_factor(q, dict(p, vex_ch=0.0))
    F1 = s.form_factor(q, dict(p, vex_ch=0.8))
    i = np.argmin(abs(q * 15 - 1.5))
    assert F1[i] < F0[i] and abs(F1[0] - F0[0]) < 1e-6


def test_designer_writes_options_and_nested_excluded_volume():
    d = Design("m")
    c = d.new_component("arm")                   # a chain with 3 interacting rods on its contour
    a = c.add_unit("FloryPolymer", "a", tag="a")
    a.thick = True
    h = c.add_pattern("Array", "ThinRod", name="hairs", tag="hair", n=3, in_ref="end1", vex=True)
    c.link(a.id, "contour", h.id, "anchor", "x")
    c.set_port("in", a.id, "end1")
    core = d.add_unit("SolidSphere", "core", tag="core")
    core.poly = True
    arms = d.add_pattern("Array", "arm", component=True, name="arms", n=4, vex=True)
    d.link(core.id, "surface", arms.id, "anchor", "s")
    code = d.to_code()
    assert "s.interact('arms')" in code and "s.interact('hairs')" in code and "poly=True" in code
    s = d.to_structure()
    assert {"sigma_core", "Rc_a", "vex_arms", "vex_hairs"} <= set(s.parameters())
    p = {k: (0.5 if k.startswith(("vex", "sigma")) else 0.6 if k.startswith("nu") else 1.0 if k.startswith("beta")
             else 2.0 if k.startswith("Rc") else 15.0) for k in s.parameters()}
    np.testing.assert_allclose(s.form_factor(q, p), s.form_factor_pairs(q, p), rtol=1e-9)
    assert s.form_factor([1e-7], p)[0] == pytest.approx(1.0, abs=1e-6)
    d2 = Design.from_json(d.to_json())
    assert d2.to_code() == code


def test_new_templates():
    for name, opt in (("Freely jointed chain", {"poly": True}), ("Worm-like chain", {"thick": True}),
                      ("Worm-like chain", {"kind": "WormLikeChain"}),
                      ("Micelle (core + grafted chains)", {"chain": "WormLikeChainEV", "poly": True, "vex": True}),
                      ("Star polymer", {"arm": "ExcludedVolumePolymer"})):
        s = T.build(T.code_for(name, opt))
        p = {k: T.default_value(k) for k in s.parameters()}
        F = s.form_factor(q, p)
        assert np.all(np.isfinite(F)) and abs(F[0] - 1) < 1e-3, name


def test_ev_worm_short_chains_and_poly_micelle_smooth():
    """PS is invalid for L < b (F > 1): the EV worm crosses over to Kholodenko there; the polydisperse, crowded
    micelle with EV worms has no jumps from branch switches in the L-derivatives."""
    import numpy as np
    from pysasfit.seb.chains import WormLikeChainEV
    from pysasfit.seb import templates as T
    m = WormLikeChainEV()
    q = np.geomspace(1e-4, 3, 600)
    for L in (5.0, 20.0, 60.0, 120.0):
        p = {"L": L, "b": 50.0}
        for v in (m.ff(q, p), m.amp(q, p, "end1"), m.amp(q, p, "middle")):
            assert v.max() <= 1 + 1e-5 and v.min() > -0.3
            assert np.all(np.diff(v) / np.abs(v[:-1]) < 2e-3)
    s = T.build(T.code_for("Micelle (core + grafted chains)", {"chain": "WormLikeChainEV", "poly": True, "vex": True}))
    p = {k: T.default_value(k) for k in s.parameters()}
    qq = np.geomspace(1e-3, 0.5, 300)
    F = s.form_factor(qq, p)
    assert np.all(np.diff(F) / F[:-1] < 1e-3)
