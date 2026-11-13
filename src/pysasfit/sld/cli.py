"""pySLDcalculator on the command line.

    pysasfit-sld                                  the window (also: python -m pysasfit.sld)
    pysasfit-sld SiO2 -d 2.2                      formula and mass density (g/cm³)
    pysasfit-sld C7D8 -d 0.867 -n                 deuterated formula with the density of the protonated compound
    pysasfit-sld "C8H8@1.05" --matrix "C7D8@0.943"
    pysasfit-sld --aa KVFGRCELAAAMKR... --d2o 0.4  protein sequence in 40 % D2O
    pysasfit-sld Fe3O4 -d 5.17 --energy 7.1 --table henke
    pysasfit-sld --search toluene                 formula and density from the libraries and online databases

Without ``--matrix`` the contrast is that to the H2O/D2O mixture given by ``--d2o`` (default H2O);
``--matrix vacuum`` gives the excess scattering length in vacuum.
"""
from __future__ import annotations

import argparse
import sys

from .core import HC_KEV_A, Exchange, Probe, xray_source_energy
from .entries import fmt, new_entry, result_table

__all__ = ["main"]


def _parser():
    p = argparse.ArgumentParser(prog="pysasfit-sld", description="Scattering length densities for neutrons and "
                                "X-rays, molecular volume, contrast and excess scattering length.")
    p.add_argument("formula", nargs="?", help="chemical formula (periodictable grammar; labile H as H[1]; "
                   "density may be appended as @value)")
    p.add_argument("--aa", metavar="SEQ", help="protein sequence instead of a formula")
    p.add_argument("--dna", metavar="SEQ", help="DNA sequence")
    p.add_argument("--rna", metavar="SEQ", help="RNA sequence")
    p.add_argument("-d", "--density", type=float, help="mass density, g/cm³")
    p.add_argument("-n", "--natural-density", action="store_true",
                   help="the density is that of the compound with natural isotopes (protonated), e.g. "
                        "C7D8 -d 0.867 -n")
    p.add_argument("-V", "--volume", type=float, help="volume of the formula unit, Å³ (instead of the density)")
    p.add_argument("--deuteration", type=float, default=0.0, help="fraction of the non-exchangeable H that is D")
    p.add_argument("--moment", type=float, default=0.0, help="magnetic moment per formula unit, Bohr magnetons")
    p.add_argument("-w", "--wavelength", type=float, default=6.0, help="neutron wavelength, Å (default 6)")
    g = p.add_mutually_exclusive_group()
    g.add_argument("-E", "--energy", type=float, help="X-ray energy, keV (default: Cu K-alpha)")
    g.add_argument("--xray-wavelength", type=float, help="X-ray wavelength, Å")
    g.add_argument("--source", help="X-ray anode element (K-alpha), e.g. Cu, Mo, Ag, Ga")
    p.add_argument("--table", choices=("chantler", "henke"), default="chantler", help="X-ray f', f'' table")
    p.add_argument("--matrix", help="matrix / solvent: a formula with density (e.g. 'C7D8@0.943') or 'vacuum'")
    p.add_argument("--d2o", type=float, default=0.0, help="D2O fraction of the water (0 ... 1)")
    p.add_argument("--exchange-backbone", type=float, default=0.9,
                   help="exchanging fraction of the backbone amide H (default 0.9)")
    p.add_argument("--exchange-side", type=float, default=1.0, help="exchanging fraction of the other labile H")
    p.add_argument("-t", "--thickness", type=float, default=0.1, help="sample thickness for transmissions, cm")
    p.add_argument("--search", metavar="NAME", help="look a material up by name (formula and density) and exit")
    p.add_argument("--sources", default=None, help="sources of --search, comma separated: builtin, xraydb, orso, "
                   "wikipedia, pubchem (default: all)")
    p.add_argument("--gui", action="store_true", help="open the window")
    return p


def main(argv=None):
    argv = sys.argv[1:] if argv is None else list(argv)
    args = _parser().parse_args(argv)
    if args.search:
        from .lookup import SOURCES, search
        hits, errors = search(args.search, args.sources.split(",") if args.sources else None)
        for h in hits:
            size = f"{h.density:.5g} g/cm³" if h.density else f"V={h.volume:.5g} Å³" if h.volume else "-"
            print(f"{h.name[:30]:<31}{h.formula[:28]:<29}{size:<16}{SOURCES.get(h.source, h.source):<19}{h.note}")
        for s, msg in errors.items():
            print(f"{SOURCES.get(s, s)}: {msg}", file=sys.stderr)
        return 0 if hits else 1
    seqs = [(k, getattr(args, k)) for k in ("aa", "dna", "rna") if getattr(args, k)]
    if args.gui or (not args.formula and not seqs):
        from .gui import main as gui_main
        return gui_main([sys.argv[0]])
    if args.energy:
        energy = args.energy
    elif args.xray_wavelength:
        energy = HC_KEV_A / args.xray_wavelength
    else:
        energy = xray_source_energy(args.source or "Cu")
    probe = Probe(args.wavelength, energy, args.table)
    if seqs:
        kind, text = seqs[0]
        mode = "volume" if args.volume else "density" if args.density else "residues"
    else:
        kind, text = "formula", args.formula
        mode = "volume" if args.volume else "density"
    entries = [new_entry(text if kind == "formula" else kind, text, kind, mode,
                         args.volume or args.density or 0.0, args.deuteration, args.moment,
                         args.natural_density)]
    matrix = "water"
    if args.matrix:
        if args.matrix.lower() == "vacuum":
            matrix = None
        else:
            entries.append(new_entry(args.matrix, args.matrix, "formula", "density", 0.0))
            matrix = 1
    rows, columns, errors = result_table(entries, matrix, probe, args.d2o,
                                         Exchange(args.exchange_backbone, args.exchange_side), args.thickness)
    if errors:
        for k, msg in errors.items():
            print(f"error ({'matrix' if k in ('matrix', 1) else 'material'}): {msg}", file=sys.stderr)
        return 2
    w = max(len(q) for _g, q, _u, _v in rows) + 2
    group = None
    print(" " * w + "".join(f"{c[:22]:>24}" for c in columns))
    for g, q, u, vals in rows:
        if g != group:
            group = g
            print(f"\n[{g}]")
        print(f"{q:<{w}}" + "".join(f"{fmt(v)[:22]:>24}" for v in vals) + (f"  {u}" if u else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
