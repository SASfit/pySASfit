# SPDX-License-Identifier: BSD-3-Clause
"""SANSPOL correction without spin analysis (pysasfit.sasred.polarisation)."""
import numpy as np
import pytest
import sympy as sy

from pysasfit.sasred.data.uarray import Source, UArray
from pysasfit.sasred import polarisation as pol
from pysasfit.sasred.reduce import load_instrument


def _truth(n=400, seed=1):
    rng = np.random.default_rng(seed)
    Ip = 10 + 5 * rng.random(n)
    Im = Ip * (1.3 + 0.4 * rng.random(n))
    return Ip, Im


def _meas(Ip, Im, setup, states=("off", "on", "unpolarised")):
    return {s: pol.sanspol_forward(Ip, Im, setup, s) for s in states}


def test_depolarisation_factor():
    assert pol.depolarisation_factor(1.0) == 1.0
    assert pol.depolarisation_factor(0.9) == pytest.approx(0.1 / -np.log(0.9))
    assert pol.depolarisation_factor(0.9) == pytest.approx(0.949, abs=1e-3)
    assert pol.depolarisation_factor(0.75) == pytest.approx(0.869, abs=1e-3)
    # series and exact branch agree at the switch
    x = 1 - 1.0001e-4
    assert pol.depolarisation_factor(x) == pytest.approx((1 - x) / -np.log(x), rel=1e-10)
    with pytest.raises(ValueError):
        pol.depolarisation_factor(0.0)


def test_weights_exact_symbolic():
    P, f = sy.symbols("P f")
    a_off = (1 + P) / 2
    b_on = f * a_off + (1 - f) * (1 - a_off)
    assert sy.simplify((a_off - (1 - b_on)) - f * P) == 0          # n+_off - n+_on = f P
    s = pol.PolarisationSetup(P=0.97, f=0.99)
    a, b = pol.state_weights(s, "on")
    assert a + b == pytest.approx(1)
    a0, _ = pol.state_weights(s, "off")
    assert a0 - a == pytest.approx(0.99 * 0.97)


def test_bersans_determinant():
    s = pol.PolarisationSetup(P=0.96, f=0.92, form="bersans")
    (a1, b1), (a2, b2) = pol.state_weights(s, "off"), pol.state_weights(s, "on")
    p, e = 0.98, 0.92
    assert a1 * b2 - a2 * b1 == pytest.approx(p + p * e - 1)        # Keiderling: determinant p + p e - 1


@pytest.mark.parametrize("states", [("off", "on"), ("off", "unpolarised"), ("on", "unpolarised"),
                                    ("off", "on", "unpolarised")])
@pytest.mark.parametrize("form", ["exact", "bersans"])
@pytest.mark.parametrize("plus", [True, False])
def test_roundtrip(states, form, plus):
    Ip, Im = _truth()
    s = pol.PolarisationSetup(P=0.975, f=0.99, P_f=0.85, form=form, flipper_off_is_plus=plus)
    r = pol.sanspol_correct(_meas(Ip, Im, s, states), s)
    np.testing.assert_allclose(r.plus, Ip, rtol=1e-12)
    np.testing.assert_allclose(r.minus, Im, rtol=1e-12)
    np.testing.assert_allclose(r.unpolarised, (Ip + Im) / 2, rtol=1e-12)
    np.testing.assert_allclose(r.difference, Im - Ip, rtol=1e-10, atol=1e-10)
    if len(states) == 3:
        assert np.max(r.chi2) < 1e-15


def test_closed_forms():
    Ip, Im = _truth()
    s = pol.PolarisationSetup(P=0.98, f=0.995, P_f=0.9)
    m = _meas(Ip, Im, s)
    Pe = 0.98 * pol.depolarisation_factor(0.9)
    np.testing.assert_allclose(m["off"] - m["on"], 0.995 * Pe * (Ip - Im), rtol=1e-12)
    np.testing.assert_allclose(m["off"] - (m["off"] - m["on"]) / (2 * 0.995), (Ip + Im) / 2, rtol=1e-12)
    assert pol.flipped_fraction_from_data(m["off"], m["on"], m["unpolarised"]) == pytest.approx(0.995)


def _uarr(x, key, rel=0.02):
    return UArray.from_source(x, Source(key, (rel * x) ** 2, "counting", False), "1/cm/sr")


def test_uncertainty_matches_monte_carlo():
    rng = np.random.default_rng(3)
    Ip, Im = _truth(50)
    s = pol.PolarisationSetup(P=0.97, f=0.98, sigma_P=0.005, sigma_f=0.003)
    m = _meas(Ip, Im, pol.PolarisationSetup(P=0.97, f=0.98), ("off", "on"))
    u = {k: _uarr(v, f"counting:{k}") for k, v in m.items()}
    r = pol.sanspol_correct(u, s)
    assert r.plus.unit == "1/cm/sr"
    assert "polarisation:P" in r.plus.terms and "polarisation:f" in r.plus.terms
    # Monte Carlo
    N = 20000
    out = np.empty((N, 50))
    for i in range(N):
        P = 0.97 + 0.005 * rng.standard_normal()
        f = min(0.98 + 0.003 * rng.standard_normal(), 1.0)
        sm = pol.PolarisationSetup(P=P, f=f)
        mm = {k: v * (1 + 0.02 * rng.standard_normal(v.shape)) for k, v in m.items()}
        out[i] = pol.sanspol_correct(mm, sm).difference
    np.testing.assert_allclose(r.difference.std(), out.std(0), rtol=0.05)
    # the counting part alone: shared sources stay correlated
    assert set(r.plus.terms) >= {"counting:off", "counting:on"}


def test_three_states_weighted_and_chi2():
    Ip, Im = _truth(2000)
    s = pol.PolarisationSetup(P=0.97, f=0.99)
    m = _meas(Ip, Im, s)
    rng = np.random.default_rng(5)
    u = {k: UArray.from_source(v * (1 + 0.01 * rng.standard_normal(v.shape)),
                               Source(f"c:{k}", (0.01 * v) ** 2, "counting", False)) for k, v in m.items()}
    r = pol.sanspol_correct(u, s)
    assert np.median(r.chi2) == pytest.approx(0.455, abs=0.08)        # median of chi²(1)
    # WLS is better than two states only for the unpolarised sum
    r2 = pol.sanspol_correct({k: u[k] for k in ("off", "on")}, s)
    assert np.mean(r.unpolarised.std()) < np.mean(r2.unpolarised.std())
    # wrong P inflates chi² only through the scale: P alone is not identifiable from off/on
    r3 = pol.sanspol_correct({k: u[k] for k in ("off", "on")}, pol.PolarisationSetup(P=0.90, f=0.99))
    np.testing.assert_allclose(r3.unpolarised.value, r2.unpolarised.value, rtol=1e-12)  # sum independent of P


def test_errors():
    s = pol.PolarisationSetup()
    with pytest.raises(ValueError):
        pol.sanspol_correct({"off": np.ones(3)}, s)
    with pytest.raises(ValueError):
        pol.sanspol_correct({"off": np.ones(3), "up": np.ones(3)}, s)
    with pytest.raises(ValueError):
        pol.sanspol_correct({"off": np.ones(3), "on": np.ones(4)}, s)
    with pytest.raises(ValueError):
        pol.PolarisationSetup(f=0.4)


def test_sans_i_table():
    inst = load_instrument("sans_i")
    P, sig, text = pol.beam_polarisation_table(0.6, 6.0, inst)
    assert P == pytest.approx(0.984) and sig == 0.005 and "Aswal" in text and "extrapolated" not in text
    P, _, text = pol.beam_polarisation_table(0.47, 15.0, inst)
    assert P == pytest.approx(0.977)
    P, _, _ = pol.beam_polarisation_table(0.535, 6.0, inst)
    assert P == pytest.approx((0.973 + 0.984) / 2, abs=1e-3)
    P, _, text = pol.beam_polarisation_table(1.0, 15.0, inst)       # only 2 and 3 m measured at 1 nm
    assert "extrapolated" in text and P == pytest.approx(0.984)
    _, _, text = pol.beam_polarisation_table(0.6, 18.0, inst)
    assert "cannot be in" in text
    with pytest.raises(KeyError):
        pol.beam_polarisation_table(0.6, 6.0, load_instrument("sans_llb"))


def test_setup_from_metadata():
    inst = load_instrument("sans_i")
    inst["polarisation"]["metadata_keys"].update(polariser=["entry1/SANS/polariser/status"],
                                                 flipper=["entry1/SANS/flipper/state"])
    meta = {"wavelength_A": 6.0, "coll_m": 6.0, "hdf:entry1/SANS/polariser/status": "in",
            "hdf:entry1/SANS/flipper/state": "off"}
    state, s = pol.setup_from_metadata(meta, inst)
    assert state == "off" and s.P == pytest.approx(0.984) and s.f == 0.995 and s.sigma_P == 0.005
    assert "Aswal" in s.source["P"] and "Quan" in s.source["f"]
    meta["hdf:entry1/SANS/flipper/state"] = "on"
    assert pol.setup_from_metadata(meta, inst)[0] == "on"
    meta["hdf:entry1/SANS/polariser/status"] = "out"
    assert pol.setup_from_metadata(meta, inst)[0] == "unpolarised"
    _, s = pol.setup_from_metadata(meta, inst, P=0.95)
    assert s.P == 0.95 and s.source["P"] == "entered"


def test_correct_files(tmp_path):
    h5py = pytest.importorskip("h5py")
    Ip, Im = _truth(30)
    q = np.linspace(0.1, 1, 30)
    s = pol.PolarisationSetup(P=0.98, f=0.995)
    m = _meas(Ip, Im, s, ("off", "on"))
    paths = {}
    for k, v in m.items():
        p = tmp_path / f"{k}.h5"
        with h5py.File(p, "w") as f:
            f.attrs["default"] = "sasentry01"
            e = f.create_group("sasentry01")
            e.attrs["default"] = "sasdata01"
            e["title"] = np.bytes_(b"sample")
            d = e.create_group("sasdata01")
            d["Q"] = q
            d["I"] = v
            d["I"].attrs["sasred_unit"] = np.bytes_(b"1/cm/sr")
            d["Idev"] = 0.01 * v
            d["I_covariance_statistical"] = np.diag((0.01 * v) ** 2)
            pr = e.create_group("sasprocess01")
            pr["notes"] = np.bytes_(b"reduced")
            pr["metadata"] = np.bytes_(b'{"wavelength_A": 6.0, "coll_m": 6.0}')
        paths[k] = str(p)
    out = pol.correct_files(paths, str(tmp_path / "res"), setup=s)
    with h5py.File(out["plus"], "r") as f:
        d = f["sasentry01/sasdata01"]
        np.testing.assert_allclose(d["I"][()], Ip, rtol=1e-12)
        assert "I_covariance_statistical" not in d and d["Idev"].shape == (30,)
        assert b"I+" in f["sasentry01/title"][()]
        assert b"SANSPOL" in f["sasentry01/sasprocess01/notes"][()]
    # setup from the metadata of the file (SANS-I table at 0.6 nm, 6 m)
    out = pol.correct_files(paths, str(tmp_path / "res2"))
    with h5py.File(out["minus"], "r") as f:
        assert b"Aswal" in f["sasentry01/sasprocess01/polarisation"][()]


def test_correct_reduced_dataclass():
    from dataclasses import dataclass, field

    @dataclass
    class R2:
        label: str
        image: object
        mask: np.ndarray
        notes: list = field(default_factory=list)
        parameters: dict = field(default_factory=dict)

    Ip, Im = _truth(16)
    s = pol.PolarisationSetup(P=0.98, f=0.995)
    m = _meas(Ip.reshape(4, 4), Im.reshape(4, 4), s, ("off", "on"))
    mk = np.zeros((4, 4), np.uint32)
    mk2 = mk.copy(); mk2[0, 0] = 4
    out = pol.correct_reduced(s, off=R2("x", _uarr(m["off"], "c:off"), mk),
                              on=R2("x", _uarr(m["on"], "c:on"), mk2))
    np.testing.assert_allclose(out["plus"].image.value, Ip.reshape(4, 4), rtol=1e-12)
    assert out["minus"].mask[0, 0] == 4 and out["minus"].label == "x [minus]"
    assert out["plus"].parameters["polarisation"]["P"] == 0.98


# ---------------------------------------------------------------- flipper state (not in the raw files)
def test_flipper_from_name():
    from pysasfit.sasred.names import flipper_from_name

    for name, st in [("FeCo flipper on", "on"), ("FeCo_flip_off", "off"), ("sample flipper=ON", "on"),
                     ("Ni_on", "on"), ("Ni-off", "off"), ("Ni (on)", "on"), ("PNIPAM on silica", ""),
                     ("onion", ""), ("", ""), ("water", "")]:
        assert flipper_from_name(name) == st, name


def test_flipper_corrections_and_apply(tmp_path):
    from types import SimpleNamespace

    from pysasfit.sasred import names

    c = names.Corrections(str(tmp_path))
    c.set_flipper(names.alternate_flipper([12, 10, 11, 13], "off"))
    assert c.flippers() == {10: "off", 11: "on", 12: "off", 13: "on"}
    c.exclude([11], "bad")
    assert c[11]["flipper"] == "on"
    c.set([12], name="new")
    assert c[12]["flipper"] == "off"
    c.set_flipper([13], None)
    assert 13 not in c
    with pytest.raises(ValueError):
        c.set_flipper([10], "up")

    def raw(run, pol, name="FeCo"):
        return SimpleNamespace(run=run, sample_name=name, sample_name_in_file=None, role="", excluded="",
                               polariser=pol, flipper="", flipper_source="")

    r = names.apply(raw(10, "in"), c)
    assert (r.flipper, r.flipper_source) == ("off", "set by hand")
    r = names.apply(raw(14, "in", "FeCo flipper on"), c)
    assert (r.flipper, r.flipper_source) == ("on", "from the sample name")
    r = names.apply(raw(15, "in"), c)
    assert r.flipper == ""
    r = names.apply(raw(16, "out", "FeCo_on"), c)          # no polariser: no flipper state
    assert r.flipper == ""
    assert "flipper off" in c.describe()


def test_plan_labels_and_grouping_keys():
    from types import SimpleNamespace

    from pysasfit.sasred.grouping import FIELDS
    from pysasfit.sasred.plan import sample_label
    from pysasfit.sasred.reduction_table import base_sample

    r = SimpleNamespace(sample_name="FeCo", polariser="in", flipper="on")
    assert sample_label(r) == "FeCo [flipper on]" and base_sample(sample_label(r)) == "FeCo"
    assert sample_label(SimpleNamespace(sample_name="FeCo", polariser="in", flipper="")) == "FeCo [flipper ?]"
    assert sample_label(SimpleNamespace(sample_name="FeCo", polariser="out", flipper="")) == "FeCo"
    assert FIELDS["polariser"](r) == "in" and FIELDS["flipper"](r) == "on"
    assert FIELDS["flipper"](SimpleNamespace(sample_name="x")) is None


def test_setup_from_metadata_sans_i_keys():
    state, s = pol.setup_from_metadata({"polariser": "in", "flipper": "on", "flipper_source": "set by hand",
                                        "wavelength_A": 6.0, "coll_m": 6.0})
    assert state == "on" and "set by hand" in s.source["state"]
    state, s = pol.setup_from_metadata({"hdf:entry1/SANS/polarizer/state": "in", "wavelength_A": 6.0, "coll_m": 6.0})
    assert state is None and "unknown" in s.source["state"]
    state, _ = pol.setup_from_metadata({"polariser": "out"})
    assert state == "unpolarised"
