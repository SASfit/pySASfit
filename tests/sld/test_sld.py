"""pySLDcalculator: reference values, consistency of the tables, sequences, exchange, contrast."""
import math

import numpy as np
import pytest

pytest.importorskip("periodictable")

from pysasfit.sld import (Exchange, Material, Probe, contrast, d2o_scan, energy_scan, have_xraydb,  # noqa: E402
                          match_point, water, xray_factors, xray_source_energy)
from pysasfit.sld import entries as E  # noqa: E402
from pysasfit.sld.bio import clean_sequence, residue_table, sequence  # noqa: E402

LYSOZYME = ("KVFGRCELAAAMKRHGLDNYRGYSLGNWVCAAKFESNFNTQATNRNTDGSTDYGILQINSRWWCNDGRTPGSRNLCNIPCSALLSSDITASVNCAKKIVSD"
            "GNGMNAWVAWRNRCKGTDVQAWIRGCRL")
xraydb = pytest.mark.skipif(not have_xraydb(), reason="xraydb is not installed")


@pytest.mark.parametrize("formula, density, sld", [
    ("H2O", 0.99821, -0.560), ("D2O", 1.10536, 6.36), ("SiO2", 2.20, 3.475), ("C7D8", 0.943, 5.66),
    ("C8H8", 1.05, 1.41), ("Si", 2.329, 2.07), ("Au", 19.30, 4.66)])
def test_neutron_sld(formula, density, sld):
    p = Material.from_formula(formula, density=density).properties()
    assert p.n_sld == pytest.approx(sld, abs=0.01)


def test_volume_density_and_formula_density():
    a = Material.from_formula("SiO2", density=2.2)
    b = Material.from_formula("SiO2@2.2")
    c = Material.from_formula("SiO2", volume=a.volume)
    assert a.volume == pytest.approx(60.0843 / (2.2 * 6.02214076e23) * 1e24, rel=1e-4)
    assert b.volume == pytest.approx(a.volume) and c.properties().density == pytest.approx(2.2)
    with pytest.raises(ValueError):
        Material.from_formula("SiO2")


def test_scattering_length_is_sld_times_volume():
    p = Material.from_formula("D2O", density=1.10536).properties()
    assert p.n_b == pytest.approx(2 * 6.671 + 5.803, abs=0.02)            # fm
    assert p.x_b == pytest.approx(2.8179 * p.x_f1, rel=1e-4)
    assert p.electrons == 10 and p.electron_density == pytest.approx(10 / p.volume)


def test_water_cross_sections_and_absorption_scales_with_wavelength():
    m = Material.from_formula("H2O", density=0.99821)
    p6, p12 = m.properties(Probe(neutron_wavelength=6)), m.properties(Probe(neutron_wavelength=12))
    assert p6.n_xs_inc == pytest.approx(5.61, abs=0.02)                    # bound-atom value
    assert p12.n_xs_abs == pytest.approx(2 * p6.n_xs_abs, rel=1e-6)
    assert p6.n_transmission == pytest.approx(math.exp(-0.1 / p6.n_penetration))


def test_xray_water_copper():
    p = Material.from_formula("H2O", density=0.99821).properties(Probe.from_source("Cu", xray_table="henke"))
    assert p.x_sld == pytest.approx(9.45, abs=0.03)
    assert p.x_sld_thomson == pytest.approx(9.40, abs=0.02)
    assert p.x_mu_mass == pytest.approx(9.8, abs=0.3)                      # cm^2/g, photoabsorption
    assert xray_source_energy("Cu") == pytest.approx(8.04, abs=0.01)


@xraydb
def test_chantler_and_henke_agree_away_from_edges():
    for sym in ("C", "O", "Si", "Fe"):
        c1, c2 = xray_factors(sym, 12.0, "chantler")
        h1, h2 = xray_factors(sym, 12.0, "henke")
        assert float(c1) == pytest.approx(float(h1), abs=0.15)
        assert float(c2) == pytest.approx(float(h2), rel=0.08)


@xraydb
def test_energy_scan_shows_the_iron_edge():
    m = Material.from_formula("Fe3O4", density=5.17)
    s = energy_scan(m, np.linspace(6.9, 7.3, 401))
    assert ("Fe", "K", pytest.approx(7.112, abs=0.002)) in [(a, b, c) for a, b, c in s["edges"]]
    k = int(np.argmin(s["sld"]))
    assert abs(s["energy"][k] - 7.112) < 0.01                              # f' has its minimum at the edge
    assert s["sld_imag"][-1] > 3 * s["sld_imag"][0]                        # absorption jumps
    assert s["sld"][k] < s["sld"][0] - 3


def test_henke_out_of_range_is_nan():
    f1, f2 = xray_factors("Fe", 50.0, "henke")
    assert np.isnan(f1) and np.isnan(f2)


def test_deuteration_equals_deuterated_formula():
    a = Material.from_formula("C8H8", density=1.05, deuteration=1.0)
    b = Material.from_formula("C8D8", volume=a.volume)
    assert a.properties().n_sld == pytest.approx(b.properties().n_sld, rel=1e-4)
    half = Material.from_formula("C8H8", density=1.05, deuteration=0.5).properties().n_sld
    assert half == pytest.approx(0.5 * (a.properties().n_sld + Material.from_formula("C8H8", density=1.05)
                                        .properties().n_sld), rel=1e-6)


def test_density_of_the_protonated_material():
    h = Material.from_formula("C7H8", density=0.867)
    d = Material.from_formula("C7D8", density=0.867, natural_density=True)
    assert d.volume == pytest.approx(h.volume, rel=1e-9)
    p = d.properties()
    assert p.density == pytest.approx(0.9427, abs=0.0005) and p.n_sld == pytest.approx(5.66, abs=0.01)
    assert p.n_sld == pytest.approx(Material.from_formula("C7H8", density=0.867, deuteration=1).properties().n_sld,
                                    rel=1e-4)
    wrong = Material.from_formula("C7D8", density=0.867).properties()
    assert wrong.n_sld < p.n_sld - 0.4 and wrong.density == pytest.approx(0.867)
    e = E.new_entry("d-tol", "C7D8", size=0.867, natural_density=True)
    assert E.material(e).volume == pytest.approx(h.volume, rel=1e-9)
    assert Material.from_formula("SiO2", density=2.2, natural_density=True).volume == pytest.approx(
        Material.from_formula("SiO2", density=2.2).volume)


def test_labile_hydrogen_of_a_formula():
    m = Material.from_formula("C2H5OH[1]", density=0.789)
    assert m.n_labile == 1
    p0, p1 = m.properties(d2o=0.0), m.properties(d2o=1.0)
    assert p1.n_b - p0.n_b == pytest.approx(6.671 + 3.739, abs=0.01)       # one H -> D
    assert m.properties(d2o=1.0, exchange=Exchange(side=0.0)).n_b == pytest.approx(p0.n_b)
    assert p1.volume == p0.volume and p1.mass > p0.mass


def test_water_mixture():
    assert water(0).properties().n_sld == pytest.approx(-0.560, abs=0.002)
    assert water(1).properties().n_sld == pytest.approx(6.362, abs=0.01)
    mid = water(0.5).properties().n_sld
    assert mid == pytest.approx(0.5 * (water(0).properties().n_sld + water(1).properties().n_sld), abs=0.02)
    assert match_point(Material.from_formula("SiO2", density=2.2)) == pytest.approx(0.583, abs=0.005)


def test_lysozyme_sequence():
    m = sequence(LYSOZYME)
    assert m.volume == pytest.approx(18021.5, abs=0.5)
    assert m.mass() == pytest.approx(14314, abs=2)
    assert m.n_labile == 264
    assert m.labile == {"backbone": 126.0, "side": 138.0}                  # 129 - 2 Pro - N terminus
    assert m.exchanged_fraction() == pytest.approx(1 - 12.6 / 264)
    full = match_point(m, exchange=Exchange(1, 1))
    part = match_point(m)
    assert 0.44 < part < full < 0.46
    p = m.properties()
    assert p.specific_volume == pytest.approx(0.758, abs=0.003)


def test_sequence_text_and_chains():
    assert clean_sequence(">sp|P00698\nKVFG RCEL\n/\nAAA*XYZ") == ["KVFGRCEL", "AAA"]
    two = sequence("AAA/AAA")
    one = sequence("AAA")
    assert two.volume == pytest.approx(2 * one.volume) and two.n_labile == 2 * one.n_labile
    with pytest.raises(ValueError):
        sequence("AUO")
    with pytest.raises(ValueError):
        sequence("")
    assert sequence("ACGT", kind="dna").n_labile > 0
    assert sequence("ACGU", kind="rna").labile["backbone"] == 0


def test_residue_table():
    rows = {r["code"]: r for r in residue_table("aa")}
    assert rows["G"]["volume"] == pytest.approx(66.4) and rows["A"]["volume"] == pytest.approx(91.5)
    assert rows["R"]["n_labile"] == 6 and rows["P"]["n_labile"] == 0
    assert rows["A"]["match"] == pytest.approx(0.405, abs=0.005)


def test_contrast_and_excess_scattering_length():
    m = Material.from_formula("SiO2", density=2.2)
    c = contrast(m, water(0.0))
    p, w = c["material"], c["matrix"]
    assert c["n_drho"] == pytest.approx(p.n_sld - w.n_sld)
    assert c["n_excess"] == pytest.approx(p.n_b - w.n_sld * p.volume * 0.1)
    assert c["x_excess_electrons"] == pytest.approx(p.x_f1 - w.x_f1 / w.volume * p.volume, rel=1e-6)
    # I(0)/c = M (Δρ v)² / N_A
    v = p.specific_volume
    assert c["n_i0"] == pytest.approx(p.mass * (c["n_drho"] * 1e10 * v) ** 2 / 6.02214076e23, rel=1e-6)
    vac = contrast(m, None)
    assert vac["n_excess"] == pytest.approx(p.n_b) and vac["x_excess"] == pytest.approx(p.x_b)
    x, mat, sol = d2o_scan(m)
    assert x[np.argmin(abs(mat - sol))] == pytest.approx(match_point(m), abs=0.006)


def test_magnetic_sld():
    p = Material.from_formula("Fe", density=7.874, moment=2.2).properties()
    assert p.magnetic_sld == pytest.approx(5.0, abs=0.1)                   # iron: about 5e-6 Å^-2


def test_result_table_and_errors():
    ents = [E.new_entry("a", "SiO2", size=2.2), E.new_entry("bad", "Xx3", size=1.0),
            E.new_entry("lys", LYSOZYME, "aa", "residues", 0.0)]
    rows, cols, errors = E.result_table(ents, "water", d2o=0.4)
    assert cols == ["a", "bad", "lys", "solvent: 40% D2O"] and list(errors) == [1]
    d = {(g, q): v for g, q, u, v in rows}
    assert d[("Neutrons", "SLD")][1] is None and d[("Neutrons", "SLD")][0] == pytest.approx(3.475, abs=0.01)
    assert d[("Contrast to 40% D2O", "neutrons: Δρ")][3] == pytest.approx(0.0)
    text = E.table_text(rows, cols)
    assert text.splitlines()[0].split("\t")[:4] == ["group", "quantity", "unit", "a"]
    rows, cols, errors = E.result_table(ents[:1] + [E.new_entry("tol", "C7D8", size=0.943)], 1)
    assert cols == ["a", "tol"] and not errors
    rows, cols, errors = E.result_table(ents[:1], None)
    assert rows[-1][0] == "Contrast to vacuum"
    dens = E.material(E.new_entry("lys", LYSOZYME, "aa", "density", 1.0 / 0.74))
    assert dens.properties().specific_volume == pytest.approx(0.74, rel=1e-6)


def test_cli(capsys):
    from pysasfit.sld.cli import main
    assert main(["SiO2", "-d", "2.2", "--source", "Mo", "--table", "henke"]) == 0
    out = capsys.readouterr().out
    assert "excess scattering length" in out and "17.4" in out
    assert main(["Xx", "-d", "1"]) == 2
