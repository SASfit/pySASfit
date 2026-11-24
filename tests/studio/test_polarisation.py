"""Polarised neutrons in pySASstudio (studio.polarisation, ports.magnetic, ModelFunction.polarisation)."""

from __future__ import annotations

import json
import math

import numpy as np
import pytest

from pysasfit.studio import ModelFunction, ScatteringContribution, function, ports
from pysasfit.studio import polarisation as pol
from pysasfit.studio.fitting import locate
from pysasfit.studio.polarisation import Polarisation
from pysasfit.studio.ports import magnetic

ports.register_aliases()
Q = np.geomspace(0.02, 4.0, 60)


def _p(name, **kw):
    spec = ports.get(name)
    p = np.zeros(50)
    for i, prm in enumerate(spec.params):
        p[i] = prm.default
    for k, v in kw.items():
        p[spec.parameters.index(k)] = v
    return p


# ------------------------------------------------------------------ C transcriptions (libsasfit core)
def _c_sphere_f(q, R, eta):
    x = q * R
    return eta * 4 / 3 * math.pi * R**3 * 3 * (math.sin(x) - x * math.cos(x)) / x**3


def _c_ksh(q, r_out, r_in, d_sh, d_c):
    return _c_sphere_f(q, r_out, d_sh) - _c_sphere_f(q, r_in, d_sh - d_c)


def _c_superparamagnetic(q, p):
    """sasfit_ff_superparamagnetic_ff_psi: SUPER_PARAMAGNETIC_FF_PSI_BODY + SOME_NAME_FCT (GAMMA = 0, SQ_TMP = 1)."""
    R, dR, KNUC, NUC_SH, NUC_MAT, KMAG, R_AV, MU_KT, PSI, POL = p[:10]
    alpha = MU_KT * (R / R_AV) ** 3
    if abs(alpha) < 1e-5:
        LA, L = 1 / 3, alpha / 3
    else:
        L = 1 / math.tanh(alpha) - 1 / alpha
        LA = L / alpha
    kn = _c_ksh(q, R + dR, R, NUC_SH - NUC_MAT, KNUC - NUC_MAT)
    km = _c_ksh(q, R + dR, R, 0.0, KMAG)
    s2 = math.sin(PSI / 180 * math.pi) ** 2
    s4 = s2 * s2
    ipp = (km * L * s2 - kn) ** 2 + km * km * (LA * s2 - (L * L - 1 + 3 * LA) * s4)
    imm = (km * L * s2 + kn) ** 2 + km * km * (LA * s2 - (L * L - 1 + 3 * LA) * s4)
    imp = (s2 - s4) * L * L * km * km + km * km * ((s4 - s2) * (L * L - 1 + 3 * LA) + (2 - s2) * LA)
    ipm = imp
    return (1 - POL) / 2 * (imm + imp) + (1 + POL) / 2 * (ipp + ipm), (ipp, imm, ipm, imp)


def _c_magnetic_shell(q, p):
    R, dR, NC, NS, NM_, MC, MS, MM, PSI, POL = p[:10]
    s2 = math.sin(PSI / 180 * math.pi) ** 2
    kn = _c_sphere_f(q, R + dR, NS - NM_) + _c_sphere_f(q, R, NC - NS)
    km = _c_sphere_f(q, R + dR, MS - MM) + _c_sphere_f(q, R, MC - MS)
    return ((1 - POL) / 2 * (kn * kn + 2 * kn * km * s2 + km * km * s2)
            + (1 + POL) / 2 * (kn * kn - 2 * kn * km * s2 + km * km * s2))  # fmt: skip


@pytest.mark.parametrize("seed", range(6))
def test_superparamagnetic_matches_c(seed):
    rng = np.random.default_rng(seed)
    p = _p("ff_superparamagnetic_ff_psi", R=rng.uniform(3, 20), dR=rng.uniform(0, 5), eta_nuc_core=rng.uniform(-1, 8),
           eta_nuc_shell=rng.uniform(-1, 6), eta_nuc_matrix=rng.uniform(-1, 6), eta_mag_core=rng.uniform(0, 3),
           R_av=rng.uniform(3, 20), mu_kT=rng.choice([1e-7, 0.5, 5.0, 40.0]), psi=rng.uniform(0, 180),
           pol=rng.uniform(-1, 1))  # fmt: skip
    spec = ports.get("ff_superparamagnetic_ff_psi")
    q = Q[5::7]
    got = spec.intensity(q, p)
    ref = np.array([_c_superparamagnetic(x, p)[0] for x in q])
    np.testing.assert_allclose(got, ref, rtol=1e-9)
    # the channels
    T = magnetic.superparamagnetic_terms(q, p)
    ch = np.array([_c_superparamagnetic(x, p)[1] for x in q]).T
    # C (SASfit) labels by the neutron moment: its ipp, imm, ipm, imp are the spin channels mm, pp, mp, pm
    for k, name in enumerate(("mm", "pp", "mp", "pm")):
        np.testing.assert_allclose(pol.combine(T, Polarisation(quantity=name).weights()), ch[k], rtol=1e-9,
                                   atol=1e-12 * np.max(np.abs(ch)))  # fmt: skip


@pytest.mark.parametrize("seed", range(4))
def test_magnetic_shell_matches_c(seed):
    rng = np.random.default_rng(10 + seed)
    vals = dict(zip(magnetic._MS, [rng.uniform(3, 20), rng.uniform(0, 5)] + list(rng.uniform(-1, 6, 6))
                    + [rng.uniform(0, 180), rng.uniform(-1, 1)]))  # fmt: skip
    p = _p("ff_magnetic_shell_psi", **vals)
    q = Q[3::9]
    np.testing.assert_allclose(ports.get("ff_magnetic_shell_psi").intensity(q, p),
                               [_c_magnetic_shell(x, p) for x in q], rtol=1e-9)  # fmt: skip


# ------------------------------------------------------------------ weights
def test_weights_and_relations():
    T = np.random.default_rng(1).normal(size=(5, 7))
    I = {k: pol.combine(T, Polarisation(quantity=k).weights()) for k in pol.QUANTITIES if k not in ("off", "on")}
    np.testing.assert_allclose(I["plus"], I["pp"] + I["pm"])
    np.testing.assert_allclose(I["minus"], I["mm"] + I["mp"])
    np.testing.assert_allclose(I["unpolarised"], (I["plus"] + I["minus"]) / 2)
    np.testing.assert_allclose(I["difference"], I["minus"] - I["plus"])
    a, b = Polarisation(quantity="off", P=0.96, f=0.99, P_f=0.9).mixing()
    r = (1 - 0.9) / -math.log(0.9)
    assert a == pytest.approx((1 + 0.96 * r) / 2) and a + b == pytest.approx(1)
    a2, _ = Polarisation(quantity="on", P=0.96, f=0.99, P_f=0.9).mixing()
    assert a - a2 == pytest.approx(0.99 * 0.96 * r)                       # n+_off - n+_on = f P r
    p = Polarisation(quantity="off", flipper_off_is_plus=False, P=0.96)
    assert p.mixing()[1] == pytest.approx((1 + 0.96) / 2)
    assert Polarisation(quantity="on").needs() == ("P", "f", "P_f") and Polarisation(quantity="plus").needs() == ()
    with pytest.raises(ValueError):
        Polarisation(quantity="up")


def test_spin_convention_moon_riste_koehler():
    """U±± = b ± p M⊥z: spin parallel to field and magnetisation adds the magnetic amplitude."""
    s2, s4 = pol.angular_moments(90.0)
    T = pol.basis_terms(2.0, 1.0, 1.0, 1.0, s2, s4)                       # N = 2, M = 1 along the field
    assert pol.combine(T, Polarisation(quantity="pp").weights()) == pytest.approx((2 + 1) ** 2)
    assert pol.combine(T, Polarisation(quantity="mm").weights()) == pytest.approx((2 - 1) ** 2)
    assert pol.combine(T, Polarisation(quantity="difference").weights()) == pytest.approx(-8.0)   # I- - I+ = -4NM
    assert Polarisation(quantity="plus").effective_pol() == -1.0          # SASfit pol labels the moment


def test_saturated_and_random_limits():
    s2, s4 = pol.angular_moments(90.0)
    T = pol.basis_terms(1.0, 2.0, 1.0, 1.0, s2, s4)
    assert T[2] == pytest.approx(4.0) and T[3] == pytest.approx(0.0)       # ψ = 90°: all non-spin-flip
    s2, s4 = pol.angular_moments(30.0)
    T = pol.basis_terms(0.0, 1.0, 1.0, 1.0, s2, s4)
    assert T[2] + T[3] == pytest.approx(s2)                                 # |M⊥|² = sin²ψ
    T = pol.basis_terms(0.0, 1.0, 0.0, 0.0, s2, s4)
    assert T[2] + T[3] == pytest.approx(2 / 3)                              # random: 2/3 independent of ψ
    assert T[1] == 0.0


def test_azimuthal_average():
    p = _p("ff_superparamagnetic_ff_psi", mu_kT=3.0)
    psi = np.linspace(0, 180, 2001)
    c = ScatteringContribution(function("ff_py_superparamagnetic_ff_psi"), p[:10])
    for quantity in ("plus", "difference", "pm"):
        m = ModelFunction([c], polarisation=Polarisation(quantity=quantity, average=True))
        rows = []
        for ps in psi:
            c.ff_p[8] = ps
            rows.append(ModelFunction([c], polarisation=Polarisation(quantity=quantity)).intensity(Q[::10]))
        c.ff_p[8] = 0.0
        np.testing.assert_allclose(m.intensity(Q[::10]), np.trapezoid(rows, psi, axis=0) / 180, rtol=1e-6)


def test_field_direction():
    c = ScatteringContribution(function("ff_py_magnetic_shell_psi"), _p("ff_magnetic_shell_psi", psi=120.0)[:10])
    a = ModelFunction([c], polarisation=Polarisation(quantity="plus", psi_H=30.0)).intensity(Q)
    c.ff_p[8] = 90.0
    b = ModelFunction([c], polarisation=Polarisation(quantity="plus")).intensity(Q)
    np.testing.assert_allclose(a, b, rtol=1e-12)


# ------------------------------------------------------------------ models
def _models(quantity, **kw):
    mag = ScatteringContribution(function("ff_py_superparamagnetic_ff_psi"), _p("ff_superparamagnetic_ff_psi")[:10])
    nuc = ScatteringContribution(function("ff_py_fuzzysphere"), {"R": 20.0, "sigma": 1.0, "eta_sph": 2.0, "eta_sol": 0.0})
    return ModelFunction([mag, nuc], background=0.1, polarisation=Polarisation(quantity=quantity, **kw)), mag, nuc


def test_nuclear_contribution_weights():
    m, mag, nuc = _models("pm")
    In = nuc.intensity(Q)
    m_np, *_ = _models("plus")
    comps, _ = m.components(Q)
    assert np.all(comps[1] == 0)                                            # no nuclear spin flip
    comps, _ = m_np.components(Q)
    np.testing.assert_allclose(comps[1], In)
    md, *_ = _models("difference")
    assert np.all(md.components(Q)[0][1] == 0)
    # the total is consistent: plus - minus via the model
    mp_, *_ = _models("minus")
    np.testing.assert_allclose(mp_.intensity(Q) - m_np.intensity(Q), md.intensity(Q) - 0.1, rtol=1e-10)


def test_without_setting_the_port_is_sasfit():
    p = _p("ff_superparamagnetic_ff_psi", pol=0.4)
    c = ScatteringContribution(function("ff_py_superparamagnetic_ff_psi"), p[:10])
    a = ModelFunction([c]).intensity(Q)
    a_plus = ModelFunction([c], polarisation=Polarisation(quantity="plus")).intensity(Q)
    a_minus = ModelFunction([c], polarisation=Polarisation(quantity="minus")).intensity(Q)
    np.testing.assert_allclose(a, 0.7 * a_minus + 0.3 * a_plus, rtol=1e-12)     # SASfit pol = +1: spin antiparallel


def test_sanspol_forward_matches_pysasred_correction():
    from pysasfit.sasred.polarisation import PolarisationSetup, sanspol_correct

    kw = dict(P=0.975, f=0.99, P_f=0.88)
    I = {k: _models(k, **kw)[0].intensity(Q) for k in ("off", "on", "plus", "minus", "unpolarised")}
    r = sanspol_correct({"off": I["off"], "on": I["on"]}, PolarisationSetup(**kw))
    np.testing.assert_allclose(r.plus, I["plus"], rtol=1e-10)
    np.testing.assert_allclose(r.minus, I["minus"], rtol=1e-10)
    np.testing.assert_allclose(r.unpolarised, I["unpolarised"], rtol=1e-10)


def test_size_distribution_and_cache(studio_lib):
    sd = {"N": 1.0, "s": 0.2, "p": 1.0, "mu": 6.0}
    c = ScatteringContribution(function("ff_py_superparamagnetic_ff_psi"), _p("ff_superparamagnetic_ff_psi")[:10],
                               size_distribution="sd_LogNorm", sd_params=sd, distributed="R")  # fmt: skip
    ref = {k: ModelFunction([c], polarisation=Polarisation(quantity=k)).intensity(Q) for k in ("plus", "minus")}
    c.enable_cache()
    for _ in range(2):
        for k in ("plus", "minus", "plus"):
            np.testing.assert_allclose(ModelFunction([c], polarisation=Polarisation(quantity=k)).intensity(Q), ref[k],
                                       rtol=1e-12)  # fmt: skip
    c.disable_cache()
    assert not np.allclose(ref["plus"], ref["minus"])


def test_addresses_and_fit_of_P():
    from pysasfit.studio.fitting import FitProblem

    true = _models("off", P=0.95, f=0.99)[0]
    y = true.intensity(Q)
    m, *_ = _models("off", P=0.80, f=0.99)
    g, s = locate(m, "pol.P")
    assert g() == 0.8
    s(0.9)
    assert m.polarisation.P == 0.9
    with pytest.raises(KeyError):
        locate(ModelFunction([]), "pol.P")
    prob = FitProblem()
    prob.add_dataset((Q, y, 0.01 * y), m, name="off")
    prob.add_common("pol.P", lower=0.5, upper=1.0)
    prob.fit("scipy")
    assert m.polarisation.P == pytest.approx(0.95, rel=1e-6)


# ------------------------------------------------------------------ C functions with pol (adapter)
def test_c_function_with_pol_parameter(monkeypatch):
    """A function that only knows SASfit's pol parameter gets pol = n+ - n- of a SANSPOL quantity."""
    from pysasfit.studio import functions, headers

    spec = ports.get("ff_superparamagnetic_ff_psi")

    def legacy(x, p):
        with pol.active(None):
            token = pol._CURRENT.set(None)
            try:
                return spec.intensity(np.asarray(x, float), np.asarray(p, float))
            finally:
                pol._CURRENT.reset(token)

    legacy.vectorized = True
    info = headers.PluginInfo(name="ff_test_pol_legacy", label="legacy", collection="test", group="test",
                              parameters=[headers.ParameterInfo(n, "", 0.0) for n in spec.parameters])  # fmt: skip
    monkeypatch.setitem(functions._PYTHON_FUNCTIONS, "ff_test_pol_legacy", (info, legacy, None, None))
    p = _p("ff_superparamagnetic_ff_psi", pol=0.3)[:10]
    leg = ScatteringContribution(function("ff_test_pol_legacy"), p)
    new = ScatteringContribution(function("ff_py_superparamagnetic_ff_psi"), p)
    for k in ("off", "on", "plus", "unpolarised"):
        sp = Polarisation(quantity=k, P=0.97, f=0.98)
        np.testing.assert_allclose(ModelFunction([leg], polarisation=sp).intensity(Q),
                                   ModelFunction([new], polarisation=sp).intensity(Q), rtol=1e-12)  # fmt: skip
    assert leg.ff_p[9] == 0.3                                                   # restored
    with pytest.raises(ValueError, match="basis terms"):
        ModelFunction([leg], polarisation=Polarisation(quantity="pm")).intensity(Q)


# ------------------------------------------------------------------ files
def test_from_metadata_and_file(tmp_path):
    h5py = pytest.importorskip("h5py")
    p = pol.from_metadata(json.dumps({"output": "minus", "P": 0.97, "f": 0.99, "P_f": 1.0}))
    assert p.quantity == "minus" and p.needs() == ()
    assert pol.from_metadata(None, json.dumps({"wavelength_A": 6.0})) is None     # no state keys known
    from pysasfit.sasred.reduce import load_instrument

    inst = load_instrument("sans_i")
    inst["polarisation"]["metadata_keys"].update(polariser=["pol"], flipper=["flip"])
    p = pol.from_metadata(None, {"pol": "in", "flip": "on", "wavelength_A": 6.0, "coll_m": 6.0}, inst)
    assert p.quantity == "on" and p.P == pytest.approx(0.984) and "Aswal" in p.source["P"]
    path = tmp_path / "x.h5"
    with h5py.File(path, "w") as f:
        e = f.create_group("sasentry01")
        pr = e.create_group("sasprocess01")
        pr["polarisation"] = np.bytes_(json.dumps({"output": "difference", "P": 0.98}).encode())
    assert pol.from_file(str(path)).quantity == "difference"
    assert pol.from_file(str(tmp_path / "missing.h5")) is None
    d = Polarisation(quantity="on", P=0.9, source={"P": "entered"}).to_dict()
    assert Polarisation.from_dict(d) == Polarisation(quantity="on", P=0.9, source={"P": "entered"})


# ------------------------------------------------------------------ GUI
def test_explorer_polarisation_box(tmp_path):
    pytest.importorskip("PySide6")
    from PySide6 import QtWidgets

    from pysasfit.studio import project
    from pysasfit.studio.explorer import ModelExplorer

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])  # noqa: F841
    ex = ModelExplorer(ff="ff_py_superparamagnetic_ff_psi")
    try:
        y = _models("off", P=0.95)[0].intensity(Q)
        ex.add_dataset(Q, y, 0.01 * y, "off", polarisation={"quantity": "off", "P": 0.95, "source": {"P": "test"}})
        ex.add_dataset(Q, y, 0.01 * y, "minus")
        ex.set_polarisation("minus", k=1)
        assert ex.polarisation_for(0).quantity == "off" and ex.polarisation_for(1).quantity == "minus"
        assert ex.model_for(0).polarisation.P == 0.95
        box = ex.polarisation_box
        assert box.settings()["quantity"] == "off"
        assert not box.f.isVisibleTo(box) and box.P.isVisibleTo(box)
        ex.set_polarisation(k=0, P=0.9)
        assert ex.polarisation_for(0).source["P"] == "entered"
        path = tmp_path / "p.json"
        project.save(ex, str(path))
        ex2 = ModelExplorer(ff="ff_py_superparamagnetic_ff_psi")
        try:
            project.load(ex2, str(path))
            assert ex2.polarisation_for(1).quantity == "minus" and ex2.polarisation_for(0).P == 0.9
        finally:
            ex2.close()
        code = ex.python_code()
        assert "polarisation" in code
        ex.set_polarisation("none", k=0)
        assert ex.polarisation_for(0) is None
    finally:
        ex.close()
