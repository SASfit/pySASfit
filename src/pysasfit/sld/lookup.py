"""Search for materials by name: composition and mass density from local tables and online databases.

    from pysasfit.sld.lookup import search
    hits, errors = search("toluene")                       # all sources
    hits, errors = search("PMMA", sources=("builtin", "xraydb"))   # offline only
    hits[0].name, hits[0].formula, hits[0].density, hits[0].source, hits[0].note, hits[0].url

Sources (``SOURCES``)

builtin     the small library of this package (:mod:`pysasfit.sld.library`)                      offline
xraydb      the materials table of xraydb (about 90: elements, solvents, minerals, windows)      offline
orso        the ORSO SLD database, slddb.reflectometry.org (curated for reflectometry and SAS;   online
            formula, density, reference, validation flag)
wikipedia   the infobox (Chembox) of the English Wikipedia articles found for the name:          online
            formula and density as written there
pubchem     PubChem: formula of a named compound and its experimental densities (free text       online
            from handbooks, each with temperature and reference)

Every hit carries where it comes from; densities from Wikipedia and PubChem are read from free text
(the text is kept in ``note``), so look at them before use. A hit without a density has ``density`` None.
The online sources need an internet connection; a source that fails is reported in ``errors`` and does
not stop the others.
"""
from __future__ import annotations

import json
import re
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass

__all__ = ["Hit", "search", "SOURCES", "ONLINE", "to_formula", "parse_densities", "hit_entry"]

SOURCES = {"builtin": "built-in library", "xraydb": "xraydb materials", "orso": "ORSO SLD database",
           "wikipedia": "Wikipedia", "pubchem": "PubChem"}
ONLINE = ("orso", "wikipedia", "pubchem")
TIMEOUT = 12.0
USER_AGENT = "pySASfit-pySLDcalculator/0.2 (https://github.com/SASfit/pySASfit)"


@dataclass
class Hit:
    name: str
    formula: str            # in the grammar of periodictable ('' if none could be read)
    density: float | None   # g/cm^3
    source: str             # key of SOURCES
    note: str = ""          # original density text, temperature, reference, validation
    url: str = ""
    volume: float | None = None   # Å^3 per formula unit (ORSO entries given by volume)


def hit_entry(hit):
    """Entry (see :mod:`pysasfit.sld.entries`) of a hit."""
    from .entries import new_entry
    if hit.density:
        return new_entry(hit.name, hit.formula, "formula", "density", float(hit.density))
    if hit.volume:
        return new_entry(hit.name, hit.formula, "formula", "volume", float(hit.volume))
    return new_entry(hit.name, hit.formula, "formula", "density", 0.0)


# ---------------------------------------------------------------------------------------------- text helpers
_SUB = str.maketrans("₀₁₂₃₄₅₆₇₈₉₊₋", "0123456789+-")
_SUP = str.maketrans("⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻", "0123456789+-")


def _valid(formula):
    try:
        import periodictable
        f = periodictable.formula(formula)
        return bool(f.atoms)
    except Exception:  # noqa: BLE001
        return False


def to_formula(text):
    """A formula as written in a database or an infobox -> the grammar of periodictable, '' if it cannot
    be read. Handles unicode and HTML subscripts, wiki markup, hydrates (``·5H2O``), repeat units
    ``(C8H8)n``, charges, and the exchangeable ``Hx`` of the ORSO database (-> ``H[1]``)."""
    s = str(text or "")
    s = re.sub(r"<ref[^>]*>.*?</ref>|<ref[^>]*/>|<!--.*?-->", "", s, flags=re.S)
    s = re.sub(r"\{\{\s*chem2?\s*\|([^{}]*)\}\}", lambda m: "".join(p for p in m.group(1).split("|")
                                                                    if "=" not in p), s, flags=re.I)
    s = re.sub(r"<su[bp]>\s*[+\-−–]?\s*</su[bp]>|<sup>[^<]*</sup>", "", s)       # charges
    s = re.sub(r"</?sub>|</?small>|'''?|\[\[|\]\]|&nbsp;", "", s)
    s = re.sub(r"<[^>]+>", " ", s)
    s = s.translate(_SUB)
    s = re.sub(r"[⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻]+", "", s)
    s = s.replace("−", "-").replace("–", "-")
    s = s.split(" or ")[0].split(";")[0].split(",")[0]
    s = re.sub(r"\s+", "", s)
    s = re.sub(r"\)n$|\]n$", ")", s) if re.search(r"[)\]]n$", s) else s
    s = s.replace("[", "(").replace("]", ")") if not re.search(r"\[\d+\]", s) else s
    s = re.sub(r"Hx(?![a-z])", "H[1]", s)
    s = re.sub(r"(?<=[A-Za-z\d)])[+-]\d*$|[+-]+$", "", s)                      # charges: C2H3O2-, SO4-2
    parts = [p for p in re.split(r"[·•.∙*]\s*(?=\d*[A-Z(])", s) if p]            # hydrates
    s = " + ".join(parts)
    s = re.sub(r"^\((.*)\)$", r"\1", s) if s.count("(") == 1 and s.endswith(")") and s.startswith("(") else s
    return s if s and _valid(s) else ""


_NUM = r"(\d+(?:[.,]\d+)?)"
_UNITS = [
    (r"g\s*(?:/|per|·|\s)\s*(?:cm\s*(?:<sup>)?\s*[3³]|cm\^3|cc|cu\.?\s*cm|cubic\s*cm|m[lL]|mL)|g\s*[·⋅ ]?\s*cm\s*(?:<sup>)?"
     r"\s*[−\-–]\s*3|g\s*[·⋅ ]?\s*m[lL]\s*(?:<sup>)?\s*[−\-–]\s*1|kg\s*/\s*(?:l|L|dm\s*(?:<sup>)?\s*[3³])|"
     r"Mg\s*/\s*m\s*(?:<sup>)?\s*[3³]|g/cm3|g/mL", 1.0),
    (r"kg\s*(?:/|per|·|\s)\s*m\s*(?:<sup>)?\s*[3³]|kg\s*[·⋅ ]?\s*m\s*(?:<sup>)?\s*[−\-–]\s*3|kg/m3|g\s*/\s*(?:l|L)\b|"
     r"g\s*/\s*dm\s*(?:<sup>)?\s*[3³]", 1e-3),
]


def parse_densities(text):
    """Densities in a free text (wikitext of an infobox, a handbook sentence).

    Returns a list of ``(density in g/cm^3, the cleaned piece of text)``, one per line of the text
    (``<br>``, list items); pieces without a readable density are left out. A range gives its middle.
    Gas densities are kept: the note shows what they are.
    """
    out = []
    raw = re.sub(r"<ref[^>]*>.*?</ref>|<ref[^>]*/>|<!--.*?-->", "", str(text or ""), flags=re.S)
    for piece in re.split(r"<br\s*/?>|\n\s*[*#]|\n", raw):
        p = piece.strip(" *#")
        if not p:
            continue
        val = None
        m = re.search(r"\{\{\s*(?:convert|cvt)\s*\|\s*" + _NUM + r"\s*\|\s*([^|}]+)", p, flags=re.I)
        if m:
            unit = m.group(2).strip()
            for pat, fac in _UNITS:
                if re.fullmatch(pat, unit):
                    val = float(m.group(1).replace(",", ".")) * fac
                    break
        if val is None:
            m = re.search(r"\{\{\s*val\s*\|\s*" + _NUM + r"[^}]*?\bu\s*=\s*([^|}]+)", p, flags=re.I)
            if m:
                for pat, fac in _UNITS:
                    if re.fullmatch(pat, m.group(2).strip()):
                        val = float(m.group(1).replace(",", ".")) * fac
                        break
        if val is None:
            for pat, fac in _UNITS:
                m = re.search(_NUM + r"(?:\s*(?:[–\-−]|to)\s*" + _NUM + r")?\s*(?:&nbsp;)?\s*(?:" + pat + ")", p)
                if m:
                    val = float(m.group(1).replace(",", ".")) * fac
                    if m.group(2):                                  # a range: its middle
                        val = 0.5 * (val + float(m.group(2).replace(",", ".")) * fac)
                    break
        if val is None or not val > 0:
            continue
        clean = re.sub(r"\{\{\s*(?:convert|cvt)\s*\|\s*([^|}]+)\|\s*([^|}]+)[^}]*\}\}", r"\1 \2", p, flags=re.I)
        clean = re.sub(r"\{\{[^{}|]*\|([^{}]*)\}\}", lambda mm: " ".join(q for q in mm.group(1).split("|")
                                                                       if "=" not in q), clean)
        clean = re.sub(r"\[\[(?:[^\]|]*\|)?([^\]]*)\]\]", r"\1", clean)
        clean = re.sub(r"<sup>\s*([^<]*)</sup>", lambda mm: mm.group(1).translate(
            str.maketrans("0123456789-−", "⁰¹²³⁴⁵⁶⁷⁸⁹⁻⁻")), clean)
        clean = re.sub(r"<[^>]+>|'''?|&nbsp;", " ", clean)
        out.append((val, " ".join(clean.split())[:160]))
    return out


def _get_json(url, timeout=TIMEOUT):
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT, "Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as fh:
        return json.loads(fh.read().decode("utf-8", "replace"))


def _match(query, *texts):
    q = query.strip().lower()
    return any(q in str(t).lower() for t in texts if t)


# ---------------------------------------------------------------------------------------------- offline
def _builtin(query, limit):
    from .library import LIBRARY
    out = []
    for group, items in LIBRARY.items():
        for name, formula, rho in items:
            if _match(query, name, formula):
                out.append(Hit(name, formula, rho, "builtin", f"{group.lower()}, typical value near 20 °C"))
    return out[:limit]


def _xraydb(query, limit):
    from xraydb import materials
    out = []
    for name, m in materials.get_materials().items():
        if not _match(query, name, m.formula):
            continue
        f = to_formula(m.formula)
        if not f:
            continue
        cats = ", ".join(getattr(m, "categories", None) or [])
        out.append(Hit(name, f, float(m.density), "xraydb", cats))
    out.sort(key=lambda h: (h.name.lower() != query.strip().lower(), len(h.name)))
    return out[:limit]


# ---------------------------------------------------------------------------------------------- ORSO
ORSO_URL = "https://slddb.reflectometry.org/"


def _orso(query, limit):
    rows = []
    seen = set()
    keys = ["name", "formula"] if to_formula(query) else ["name"]
    for key in keys:
        res = _get_json(ORSO_URL + "api?" + urllib.parse.urlencode({key: query}))
        for r in res if isinstance(res, list) else [res] if isinstance(res, dict) else []:
            if isinstance(r, dict) and r.get("ID") not in seen and r.get("formula"):
                seen.add(r.get("ID"))
                rows.append(r)
    out = []
    for r in rows:
        f = to_formula(r.get("formula")) or to_formula(r.get("HR_formula"))
        if not f:
            continue
        note = [str(r[k]) for k in ("description", "physical_state", "reference") if r.get(k)]
        if r.get("temperature"):
            note.insert(0, f"{r['temperature']:g} K")
        note.append("ORSO validated" if r.get("validated") or r.get("ORSO_validated") else "not validated")
        try:
            rho = float(r["density"]) if r.get("density") else None
            vol = float(r["FU_volume"]) if r.get("FU_volume") else None
        except (TypeError, ValueError):
            rho = vol = None
        out.append(Hit(str(r.get("name") or f), f, rho, "orso", "; ".join(" ".join(n.split()) for n in note)[:200],
                       f"{ORSO_URL}material?ID={r.get('ID')}", vol))
    return out[:limit]


# ---------------------------------------------------------------------------------------------- Wikipedia
WIKI_API = "https://en.wikipedia.org/w/api.php"
_ELEMENT = re.compile(r"\|\s*(A[cglmrstu]|B[aehikr]?|C[adeflmnorsu]?|D[bsy]?|E[rsu]|F[elmr]?|G[ade]|H[efgos]?|I[nr]?|"
                      r"Kr?|L[airuv]|M[cdgnot]|N[abdehiop]?|O[gs]?|P[abdmortu]?|R[abefghnu]|S[bcegimnr]?|"
                      r"T[abcehilms]|U|V|W|Xe|Yb?|Z[nr])\s*=\s*(\d+(?:\.\d+)?)\s*(?=[|}\n])")


def _field(wikitext, name):
    """Value of ``| name = value`` in an infobox (up to the next top-level ``|`` line)."""
    m = re.search(r"^\s*\|\s*" + name + r"\s*=[ \t]*(.*(?:\n(?!\s*[|}]).*)*)", wikitext, flags=re.I | re.M)
    return m.group(1).strip() if m else ""


def chembox(wikitext):
    """(formula, [(density, text)]) of the Chembox / infobox of a Wikipedia article; ('', []) if none."""
    if not re.search(r"\{\{\s*(chembox|infobox)", wikitext, flags=re.I):
        return "", []
    formula = to_formula(_field(wikitext, "Formula"))
    if not formula:
        m = re.search(r"\{\{\s*Chembox Properties(.*?)(?=\n\s*\|\s*Section\d+\s*=|\n\}\}\s*\n\}\}|\Z)", wikitext,
                      flags=re.I | re.S)
        block = m.group(1) if m else ""
        counts = {}
        for sym, n in _ELEMENT.findall(block):
            counts.setdefault(sym, n)
        order = sorted(counts, key=lambda s: (s != "C", s != "H" if "C" in counts else False, s))
        formula = to_formula("".join(f"{s}{counts[s]}" for s in order))
    dens = []
    for key in ("Density", "density"):
        dens = parse_densities(_field(wikitext, key))
        if dens:
            break
    return formula, dens


def _wikipedia(query, limit):
    params = {"action": "query", "generator": "search", "gsrsearch": query, "gsrlimit": min(max(limit, 1), 8),
              "prop": "revisions", "rvprop": "content", "rvslots": "main", "redirects": 1, "format": "json",
              "formatversion": 2}
    res = _get_json(WIKI_API + "?" + urllib.parse.urlencode(params))
    pages = sorted((res.get("query") or {}).get("pages") or [], key=lambda p: p.get("index", 0))
    out = []
    for p in pages:
        try:
            text = p["revisions"][0]["slots"]["main"]["content"]
        except (KeyError, IndexError, TypeError):
            continue
        formula, dens = chembox(text)
        if not formula:
            continue
        title = p.get("title", "")
        url = "https://en.wikipedia.org/wiki/" + urllib.parse.quote(title.replace(" ", "_"))
        if dens:
            out += [Hit(title, formula, rho, "wikipedia", note, url) for rho, note in dens]
        else:
            out.append(Hit(title, formula, None, "wikipedia", "no density in the infobox", url))
    return out[:limit]


# ---------------------------------------------------------------------------------------------- PubChem
PUBCHEM = "https://pubchem.ncbi.nlm.nih.gov/rest/"


def _strings(node, out):
    """Collect (text, reference number) of every 'Information' value below a PUG-View node."""
    if isinstance(node, dict):
        for info in node.get("Information", []) or []:
            val = info.get("Value", {}) or {}
            for s in val.get("StringWithMarkup", []) or []:
                if s.get("String"):
                    out.append((s["String"], info.get("ReferenceNumber")))
            if "Number" in val:
                num = val["Number"]
                out.append((f"{num[0] if isinstance(num, list) and num else num} {val.get('Unit', '')}".strip(),
                            info.get("ReferenceNumber")))
        for v in node.values():
            if isinstance(v, (dict, list)):
                _strings(v, out)
    elif isinstance(node, list):
        for v in node:
            _strings(v, out)


def _pubchem(query, limit):
    name = urllib.parse.quote(query.strip(), safe="")
    try:
        res = _get_json(f"{PUBCHEM}pug/compound/name/{name}/property/MolecularFormula,Title/JSON")
    except urllib.error.HTTPError as exc:
        if exc.code == 404:                     # name not known
            return []
        raise
    out = []
    for prop in ((res.get("PropertyTable") or {}).get("Properties") or [])[:2]:
        cid = prop.get("CID")
        f = to_formula(prop.get("MolecularFormula"))
        if not f or cid is None:
            continue
        title = prop.get("Title") or query
        url = f"https://pubchem.ncbi.nlm.nih.gov/compound/{cid}"
        dens = []
        try:
            rec = _get_json(f"{PUBCHEM}pug_view/data/compound/{cid}/JSON?heading=Density")
            texts = []
            _strings(rec.get("Record", {}), texts)
            refs = {r.get("ReferenceNumber"): r.get("SourceName", "") for r in
                    (rec.get("Record", {}).get("Reference") or [])}
            for text, ref in texts:
                for rho, note in parse_densities(text):
                    src = refs.get(ref)
                    dens.append((rho, note + (f"  [{src}]" if src else "")))
        except urllib.error.HTTPError as exc:
            if exc.code != 404:                 # 404: the compound has no density section
                raise
        seen = set()
        for rho, note in dens:
            if (round(rho, 4), note) not in seen:
                seen.add((round(rho, 4), note))
                out.append(Hit(title, f, rho, "pubchem", note, url))
        if not dens:
            out.append(Hit(title, f, None, "pubchem", "no experimental density listed", url))
    return out[:limit]


_PROVIDERS = {"builtin": _builtin, "xraydb": _xraydb, "orso": _orso, "wikipedia": _wikipedia, "pubchem": _pubchem}


def search(query, sources=None, limit=12):
    """Search ``query`` (a name, or a formula for the local tables and ORSO) in ``sources`` (default: all).

    Returns ``(hits, errors)``: the :class:`Hit` s in the order of the sources, and {source: message} of
    the sources that failed (no network, package not installed, ...).
    """
    query = str(query or "").strip()
    hits, errors = [], {}
    if not query:
        return hits, errors
    for src in sources or tuple(SOURCES):
        fn = _PROVIDERS.get(src)
        if fn is None:
            errors[src] = "unknown source"
            continue
        try:
            hits += fn(query, limit)
        except ImportError:
            errors[src] = "the package xraydb is not installed" if src == "xraydb" else "package missing"
        except urllib.error.HTTPError as exc:
            errors[src] = f"the server answered {exc.code} {exc.reason}"
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            errors[src] = f"no connection ({getattr(exc, 'reason', exc)})"
        except Exception as exc:  # noqa: BLE001 -- an unexpected answer of one source must not stop the others
            errors[src] = f"unexpected answer: {exc}"
    return hits, errors
