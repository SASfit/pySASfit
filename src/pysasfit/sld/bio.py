"""Proteins and nucleic acids from their sequence.

Residue formulas, labile hydrogens and residue volumes are those of :mod:`periodictable.fasta`
(Perkins 1988; the amino-acid volumes are the crystal volumes also tabulated by Jacrot 1976; nucleotide
volumes from Buckin 1989 / Durchschlag & Zipper 1997). Asp and Glu are ionised, Lys and Arg protonated.
A chain is closed with H and OH.

Classes of labile hydrogen (:class:`pysasfit.sld.core.Exchange`):

backbone   one amide N-H per residue of a protein, except proline and the first residue (whose N-H
           belongs to the terminal amine); 10 % of them do not exchange by default
side       everything else: side chains, chain ends, all labile H of nucleic acids

Several chains are separated by ``/`` or by new lines; FASTA header lines (``>...``) and everything
after ``*`` are ignored.
"""
from __future__ import annotations

from .core import Material, _is_labile, _pt

__all__ = ["sequence", "residue_table", "clean_sequence", "KINDS"]

KINDS = {"aa": "protein", "dna": "DNA", "rna": "RNA"}


def _tables(kind):
    _pt()
    from periodictable import fasta
    try:
        return {"aa": fasta.AMINO_ACID_CODES, "dna": fasta.DNA_CODES, "rna": fasta.RNA_CODES}[kind]
    except KeyError:
        raise ValueError(f"unknown sequence type {kind!r} (aa, dna, rna)") from None


def clean_sequence(text):
    """Chains of a sequence text: list of upper-case strings (FASTA headers, blanks, digits removed)."""
    chains = []
    cur = []
    for line in str(text).replace("/", "\n/\n").splitlines():
        line = line.strip()
        if line.startswith(">") or line == "/":
            if cur:
                chains.append("".join(cur))
                cur = []
            continue
        line = line.split("*", 1)[0]
        cur.append("".join(ch for ch in line if ch.isalpha() or ch == "-").upper())
    if cur and "".join(cur):
        chains.append("".join(cur))
    return [c for c in chains if c]


def sequence(text, kind="aa", name=None, deuteration=0.0, volume=None):
    """:class:`Material` of a protein ('aa'), DNA ('dna') or RNA ('rna') sequence (one or more chains).

    ``volume`` (Å^3) replaces the sum of the residue volumes, e.g. from a measured partial specific
    volume: V = v̄ M / N_A.
    """
    codes = _tables(kind)
    pt = _pt()
    chains = clean_sequence(text)
    if not chains:
        raise ValueError("empty sequence")
    atoms = {}
    vol = 0.0
    backbone = 0
    for chain in chains:
        for i, c in enumerate(chain):
            try:
                res = codes[c]
            except KeyError:
                raise ValueError(f"unknown {KINDS[kind]} code '{c}'") from None
            for a, n in res.labile_formula.atoms.items():
                atoms[a] = atoms.get(a, 0.0) + n
            vol += res.cell_volume
            if kind == "aa" and i > 0 and c not in "P-" \
                    and any(_is_labile(a) for a in res.labile_formula.atoms):
                backbone += 1
        atoms[pt.H[1]] = atoms.get(pt.H[1], 0.0) + 2          # chain ends: H and OH
        atoms[pt.O] = atoms.get(pt.O, 0.0) + 1
    atoms = {a: n for a, n in atoms.items() if n}
    n_lab = sum(n for a, n in atoms.items() if _is_labile(a))
    if vol <= 0 and volume is None:
        raise ValueError("the sequence has no residue with a volume")
    label = name or f"{KINDS[kind]} ({sum(len(c) for c in chains)} residues)"
    return Material(label, atoms, float(volume if volume is not None else vol),
                    {"backbone": float(backbone), "side": float(n_lab - backbone)},
                    float(deuteration), "/".join(chains), kind)


def residue_table(kind="aa", probe=None, exchange=None):
    """Rows of the residue table: dicts with code, name, formula, volume (Å^3), n_labile, the neutron SLD
    in H2O and in D2O (1e-6 Å^-2, residue inside a chain, all labile H exchanged) and the match point."""
    from .core import Exchange, Probe, match_point
    probe = probe or Probe()
    exchange = exchange or Exchange(backbone=1.0, side=1.0)
    out = []
    for code, res in sorted(_tables(kind).items()):
        if not res.cell_volume or code in "-X":
            continue
        atoms = {a: float(n) for a, n in res.labile_formula.atoms.items()}
        n_lab = sum(n for a, n in atoms.items() if _is_labile(a))
        m = Material(res.name, atoms, res.cell_volume, {"side": n_lab})
        out.append({"code": code, "name": res.name, "formula": str(res.labile_formula),
                    "volume": res.cell_volume, "n_labile": n_lab,
                    "sld_h2o": m.properties(probe, 0.0, exchange).n_sld,
                    "sld_d2o": m.properties(probe, 1.0, exchange).n_sld,
                    "match": match_point(m, probe, exchange)})
    return out
