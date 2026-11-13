"""Material search: text parsing and the sources (the online ones with canned answers of their APIs)."""
import urllib.error

import pytest

pytest.importorskip("periodictable")

from pysasfit.sld import entries as E  # noqa: E402
from pysasfit.sld import lookup as L  # noqa: E402

WIKI_TOLUENE = """{{Short description|Chemical compound}}
{{Chembox
| Name = Toluene
|Section1={{Chembox Identifiers
| CASNo = 108-88-3
| PubChem = 1140
}}
|Section2={{Chembox Properties
| C=7 | H=8
| Appearance = Colorless liquid<ref name="x">ref</ref>
| Density = 0.8623 g/mL (25 °C)<ref name=crc />
| MeltingPtC = -95.0
}}
|Section3={{Chembox Hazards
| NFPA-H = 2
}}
}}
'''Toluene''' is ..."""

WIKI_WATER = """{{Chembox
|Section2={{Chembox Properties
| Formula = {{chem2|H2O}}
| Density = Liquid:<br />0.9998396 g/mL at 0 °C<br />0.9970474 g/mL at 25 °C
0.961893 g/mL at 95 °C
| N = 5
}}
}}"""

WIKI_PS = """{{Chembox
| Section2 = {{Chembox Properties
| Formula = (C<sub>8</sub>H<sub>8</sub>)<sub>''n''</sub>
| Density = 0.96–1.05 g/cm<sup>3</sup>
}}
}}"""


@pytest.mark.parametrize("text, formula", [
    ("C₇H₈", "C7H8"), ("C<sub>7</sub>H<sub>8</sub>", "C7H8"), ("(C<sub>8</sub>H<sub>8</sub>)<sub>''n''</sub>", "C8H8"),
    ("CuSO<sub>4</sub>·5H<sub>2</sub>O", "CuSO4 + 5H2O"), ("{{chem2|SiO2}}", "SiO2"), ("C2H3O2-", "C2H3O2"),
    ("C6HxH5", "C6H[1]H5"), ("H[2]2O", "H[2]2O"), ("junk", ""), ("", "")])
def test_to_formula(text, formula):
    assert L.to_formula(text) == formula


@pytest.mark.parametrize("text, values", [
    ("0.87 g/mL (20 °C)", [0.87]), ("{{convert|0.8669|g/mL|lb/USgal}} at 20 °C", [0.8669]),
    ("2.648 g·cm<sup>−3</sup> (α-quartz)<ref>x</ref>", [2.648]), ("{{val|0.867|u=g/cm3}}", [0.867]),
    ("999.97 kg/m<sup>3</sup>, liquid (4 °C)<br />917 kg/m<sup>3</sup>, solid", [0.99997, 0.917]),
    ("0.8623 g/cu cm at 20 °C", [0.8623]), ("0.96–1.05 g/cm<sup>3</sup>", [1.005]), ("1,05 g/cm3", [1.05]),
    ("Relative density (water = 1): 0.87", []), ("", [])])
def test_parse_densities(text, values):
    assert [v for v, _n in L.parse_densities(text)] == pytest.approx(values)


def test_chembox():
    assert L.chembox(WIKI_TOLUENE) == ("C7H8", [(0.8623, "0.8623 g/mL (25 °C)")])
    f, d = L.chembox(WIKI_WATER)
    assert f == "H2O" and [round(v, 4) for v, _ in d] == [0.9998, 0.997, 0.9619]
    f, d = L.chembox(WIKI_PS)
    assert f == "C8H8" and d[0][0] == pytest.approx(1.005)
    assert L.chembox("just text, no infobox") == ("", [])


def test_offline_sources():
    hits, errors = L.search("toluene", ("builtin",))
    assert not errors and [h.name for h in hits] == ["toluene", "toluene-d8"] and hits[0].density == 0.867
    hits, errors = L.search("SiO2", ("builtin",))
    assert hits and hits[0].formula == "SiO2"
    assert L.search("", ("builtin",)) == ([], {})
    assert "nosuch" in L.search("x", ("nosuch",))[1]
    pytest.importorskip("xraydb")
    hits, errors = L.search("kapton", ("xraydb",))
    assert hits and hits[0].density == pytest.approx(1.42, abs=0.02) and E.material(L.hit_entry(hits[0]))


def _fake(answers):
    def get(url, timeout=0):
        for key, val in answers.items():
            if key in url:
                if isinstance(val, Exception):
                    raise val
                return val
        raise AssertionError(url)
    return get


def test_orso(monkeypatch):
    rows = [{"ID": 227, "name": "Toluene", "formula": "C7H8", "density": 0.8669, "FU_volume": None,
             "description": "liquid", "reference": "CRC", "validated": "2021-01-01", "temperature": 293.15},
            {"ID": 300, "name": "d-Toluene", "formula": "C7D8", "density": None, "FU_volume": 176.4,
             "validated": None},
            {"ID": 301, "name": "broken", "formula": "??", "density": 1.0}]
    monkeypatch.setattr(L, "_get_json", _fake({"slddb.reflectometry.org/api?name=": rows,
                                               "slddb.reflectometry.org/api?formula=": []}))
    hits, errors = L.search("toluene", ("orso",))
    assert not errors and [h.name for h in hits] == ["Toluene", "d-Toluene"]
    assert hits[0].density == pytest.approx(0.8669) and "ORSO validated" in hits[0].note and "293" in hits[0].note
    assert hits[1].density is None and hits[1].volume == pytest.approx(176.4) and "not validated" in hits[1].note
    assert L.hit_entry(hits[1])["size_mode"] == "volume"
    assert E.material(L.hit_entry(hits[1])).properties().n_sld == pytest.approx(5.66, abs=0.01)


def test_wikipedia(monkeypatch):
    pages = {"query": {"pages": [
        {"title": "Toluene (data page)", "index": 2, "revisions": [{"slots": {"main": {"content": "no box"}}}]},
        {"title": "Toluene", "index": 1, "revisions": [{"slots": {"main": {"content": WIKI_TOLUENE}}}]},
        {"title": "Polystyrene", "index": 3, "revisions": [{"slots": {"main": {"content": WIKI_PS}}}]},
        {"title": "No text", "index": 4}]}}
    monkeypatch.setattr(L, "_get_json", _fake({"wikipedia.org": pages}))
    hits, errors = L.search("toluene", ("wikipedia",))
    assert not errors and [(h.name, h.formula) for h in hits] == [("Toluene", "C7H8"), ("Polystyrene", "C8H8")]
    assert hits[0].density == pytest.approx(0.8623) and hits[0].url.endswith("/wiki/Toluene")
    assert "25 °C" in hits[0].note


def test_pubchem(monkeypatch):
    props = {"PropertyTable": {"Properties": [{"CID": 1140, "MolecularFormula": "C7H8", "Title": "Toluene"}]}}
    dens = {"Record": {"RecordNumber": 1140, "Section": [{"TOCHeading": "Chemical and Physical Properties", "Section": [
        {"TOCHeading": "Experimental Properties", "Section": [{"TOCHeading": "Density", "Information": [
            {"ReferenceNumber": 7, "Value": {"StringWithMarkup": [{"String": "0.8623 g/cu cm at 20 °C"}]}},
            {"ReferenceNumber": 9, "Value": {"StringWithMarkup": [{"String": "Relative density (water = 1): 0.87"}]}},
            {"ReferenceNumber": 9, "Value": {"Number": [0.87], "Unit": "g/cm³"}}]}]}]}],
        "Reference": [{"ReferenceNumber": 7, "SourceName": "HSDB"}, {"ReferenceNumber": 9, "SourceName": "ILO-WHO"}]}}
    monkeypatch.setattr(L, "_get_json", _fake({"/property/": props, "pug_view": dens}))
    hits, errors = L.search("toluene", ("pubchem",))
    assert not errors and [round(h.density, 4) for h in hits] == [0.8623, 0.87]
    assert hits[0].formula == "C7H8" and "[HSDB]" in hits[0].note and hits[0].url.endswith("/1140")
    # no density section, unknown name, no network
    nf = urllib.error.HTTPError("u", 404, "Not Found", None, None)
    monkeypatch.setattr(L, "_get_json", _fake({"/property/": props, "pug_view": nf}))
    hits, errors = L.search("toluene", ("pubchem",))
    assert not errors and hits[0].density is None and L.hit_entry(hits[0])["size"] == 0.0
    monkeypatch.setattr(L, "_get_json", _fake({"/property/": nf}))
    assert L.search("nosuchthing", ("pubchem",)) == ([], {})
    monkeypatch.setattr(L, "_get_json", _fake({"pubchem": urllib.error.URLError("offline"),
                                               "wikipedia": {"unexpected": 1}}))
    hits, errors = L.search("toluene", ("builtin", "pubchem", "wikipedia"))
    assert len(hits) == 2 and "no connection" in errors["pubchem"] and "wikipedia" not in errors


def test_cli_search(capsys):
    from pysasfit.sld.cli import main
    assert main(["--search", "toluene", "--sources", "builtin"]) == 0
    assert "C7D8" in capsys.readouterr().out
    assert main(["--search", "nosuchmaterial", "--sources", "builtin"]) == 1
