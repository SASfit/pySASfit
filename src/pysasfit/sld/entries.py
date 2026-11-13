"""Materials as plain dictionaries (what the GUI edits and the session stores) and the table of results.

An entry::

    {"name": "lysozyme", "kind": "aa",            # 'formula', 'aa', 'dna', 'rna'
     "text": "KVFGRCELAA...",                     # formula or sequence
     "size_mode": "residues",                     # 'density' (g/cm^3), 'volume' (Å^3), 'residues' (sequences)
     "size": 0.0, "deuteration": 0.0, "moment": 0.0,
     "natural_density": False}                    # formula + density: the density is that of the compound
                                                  # with natural isotopes (protonated), see Material.from_formula

A matrix (solvent) is ``None`` (vacuum), ``"water"`` (H2O/D2O mixture with the D2O fraction of the
calculation) or the index of an entry.
"""
from __future__ import annotations

import math

from .core import N_A, Exchange, Material, Probe, contrast, match_point, water

__all__ = ["new_entry", "material", "matrix_material", "result_table", "table_text", "DEFAULT_ENTRIES"]

DEFAULT_ENTRIES = [
    {"name": "silica", "kind": "formula", "text": "SiO2", "size_mode": "density", "size": 2.2,
     "deuteration": 0.0, "moment": 0.0},
    {"name": "polystyrene", "kind": "formula", "text": "C8H8", "size_mode": "density", "size": 1.05,
     "deuteration": 0.0, "moment": 0.0},
]


def new_entry(name="material", text="H2O", kind="formula", size_mode="density", size=1.0, deuteration=0.0,
              moment=0.0, natural_density=False):
    return {"name": name, "kind": kind, "text": text, "size_mode": size_mode, "size": size,
            "deuteration": deuteration, "moment": moment, "natural_density": bool(natural_density)}


def material(entry):
    """:class:`Material` of an entry (raises ValueError with a readable message)."""
    kind = entry.get("kind", "formula")
    mode = entry.get("size_mode", "density")
    size = float(entry.get("size") or 0.0)
    deut = float(entry.get("deuteration") or 0.0)
    if not 0 <= deut <= 1:
        raise ValueError("the deuteration must be between 0 and 1")
    name = entry.get("name") or None
    text = str(entry.get("text", "")).strip()
    if not text:
        raise ValueError("no formula" if kind == "formula" else "no sequence")
    try:
        if kind == "formula":
            if mode == "volume":
                m = Material.from_formula(text, volume=size, name=name, deuteration=deut)
            else:
                m = Material.from_formula(text, density=size if size > 0 else None, name=name, deuteration=deut,
                                          natural_density=bool(entry.get("natural_density")))
        else:
            m = Material.from_sequence(text, kind=kind, name=name, deuteration=deut)
            if mode == "volume":
                if not size > 0:
                    raise ValueError("the volume must be positive")
                m = m.with_(volume=size)
            elif mode == "density":
                if not size > 0:
                    raise ValueError("the density must be positive")
                m = m.with_(volume=m.mass() / (size * N_A) * 1e24)
    except ValueError:
        raise
    except Exception as exc:  # noqa: BLE001 -- parser errors of periodictable (pyparsing)
        raise ValueError(f"cannot read '{text[:40]}': {exc}") from None
    return m.with_(moment=float(entry.get("moment") or 0.0))


def matrix_material(matrix, entries, d2o=0.0):
    """:class:`Material` of the matrix, or None for vacuum."""
    if matrix is None or matrix == "vacuum":
        return None
    if matrix == "water":
        return water(d2o)
    return material(entries[int(matrix)])


def result_table(entries, matrix="water", probe=Probe(), d2o=0.0, exchange=Exchange(), thickness=0.1):
    """Table of all quantities of all entries.

    Returns ``(rows, columns, errors)``: ``rows`` is a list of ``(group, quantity, unit, [value per
    column])``; ``columns`` the names (the entries, then the matrix if it is not one of them); ``errors``
    {entry index: message} of entries that could not be calculated (their values are None).
    """
    errors = {}
    mats = []
    for i, e in enumerate(entries):
        try:
            mats.append(material(e))
        except ValueError as exc:
            mats.append(None)
            errors[i] = str(exc)
    try:
        mat = matrix_material(matrix, entries, d2o)
    except (ValueError, IndexError) as exc:
        mat = None
        errors["matrix"] = str(exc)
    columns = [e.get("name") or f"material {i + 1}" for i, e in enumerate(entries)]
    cols = list(mats)
    if matrix == "water":
        columns.append(f"solvent: {mat.name}")
        cols.append(mat)

    data = []
    for i, m in enumerate(cols):
        if m is None:
            data.append(None)
            continue
        try:
            c = contrast(m, mat, probe, d2o, exchange, thickness)
            c["match"] = match_point(m, probe, exchange) if m.kind != "water" else float("nan")
            data.append(c)
        except Exception as exc:  # noqa: BLE001
            data.append(None)
            errors[i if i < len(entries) else "matrix"] = str(exc)

    rows = []
    first = next((c for c in data if c is not None), None)
    if first is None:
        return rows, columns, errors
    template = first["material"].rows()
    per_col = [dict(((g, q), v) for g, q, v, _u in c["material"].rows()) if c else {} for c in data]
    seen = set()
    for c in data:                      # the magnetic rows exist only for materials with a moment
        if c:
            for g, q, _v, u in c["material"].rows():
                if (g, q) not in seen:
                    seen.add((g, q))
                    if (g, q, u) not in [(a, b, d) for a, b, _x, d in template]:
                        template.append((g, q, None, u))
    for g, q, _v, u in template:
        rows.append((g, q, u, [pc.get((g, q)) for pc in per_col]))

    mname = "vacuum" if mat is None else mat.name

    def col(key, scale=1.0):
        return [None if c is None else c[key] * scale for c in data]

    g = f"Contrast to {mname}"
    rows += [
        (g, "neutrons: Δρ", "10⁻⁶ Å⁻²", col("n_drho")),
        (g, "neutrons: (Δρ)²", "10²⁰ cm⁻⁴", col("n_drho2")),
        (g, "neutrons: excess scattering length Δb = b − ρ₀V", "fm", col("n_excess")),
        (g, "neutrons: Δb per mass", "10¹⁰ cm/g", col("n_excess_per_mass", 1e-10)),
        (g, "neutrons: I(0)/c = N_A (Δb)²/M", "cm²/g", col("n_i0")),
        (g, "neutrons: match point (H₂O/D₂O)", "% D₂O", [None if c is None or math.isnan(c["match"])
                                                          else 100 * c["match"] for c in data]),
        (g, "X-rays: Δρ", "10⁻⁶ Å⁻²", col("x_drho")),
        (g, "X-rays: (Δρ)²", "10²⁰ cm⁻⁴", col("x_drho2")),
        (g, "X-rays: excess scattering length Δb", "fm", col("x_excess")),
        (g, "X-rays: excess electrons Δb/r_e", "e", col("x_excess_electrons")),
        (g, "X-rays: Δb per mass", "10¹⁰ cm/g", col("x_excess_per_mass", 1e-10)),
        (g, "X-rays: I(0)/c = N_A (Δb)²/M", "cm²/g", col("x_i0")),
    ]
    return rows, columns, errors


def fmt(v, digits=6):
    if v is None:
        return ""
    if isinstance(v, str):
        return v
    if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
        return "–" if math.isnan(v) else "∞"
    return f"{v:.{digits}g}"


def table_text(rows, columns, sep="\t", digits=6):
    """The result table as text (tab separated by default; ',' gives CSV-like text)."""
    def cell(s):
        s = str(s)
        return '"' + s.replace('"', '""') + '"' if sep != "\t" and (sep in s or '"' in s) else s

    out = [sep.join(cell(c) for c in ["group", "quantity", "unit", *columns])]
    for g, q, u, vals in rows:
        out.append(sep.join(cell(c) for c in [g, q, u, *[fmt(v, digits) for v in vals]]))
    return "\n".join(out) + "\n"
