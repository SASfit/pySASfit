# pySLDcalculator

Scattering length densities of materials and biomolecules for neutrons and X-rays, with the quantities
one needs around them: molecular volume, electron density, cross sections and transmissions, the
contrast to a matrix, the **excess scattering length**, and the D₂O match point.

```bash
pysasfit-sld                                   # the window (also a tab of pysasfit-gui)
pysasfit-sld SiO2 -d 2.2                       # values on the command line
pysasfit-sld "C8H8@1.05" --matrix "C7D8@0.943"
pysasfit-sld --aa KVFGRCELAAAMKR... --d2o 0.4
pip install pysasfit[sld]                      # periodictable and xraydb
```

## Input

**Formula** in the grammar of `periodictable` (the NIST calculator): `SiO2`, `C6H5CH3`, `Pd1.3Si`,
`CaCO3 + 6H2O`; isotopes `D2O`, `C7D8`, `Ca[44]CO3`; mixtures `3%wt NaCl // H2O`. Exchangeable
(labile) hydrogen is written `H[1]`: `C2H5OH[1]`.

**Size**: the mass density (g/cm³) of the formula as written, or the volume of one formula unit (Å³).
They are related by V = M / (ρ N_A). The volume is kept for all isotopic variants (deuteration, H/D
exchange); the density then follows from the mass.

Densities are usually known for the protonated material. For a deuterated one either

- write the protonated formula with its density and set the *deuteration* (0 … 1), or
- write the deuterated formula and tick *density is that of the protonated material* (Python:
  `natural_density=True`, command line `-n`, in a formula `C7D8@0.867n`), or
- give the measured density of the deuterated material.

A deuterated formula with the protonated density and without the tick gets a volume that is too large:
for toluene-d8 with 0.867 g/cm³ the SLD would be 5.21 instead of 5.66 · 10⁻⁶ Å⁻².

**Sequences**: protein, DNA or RNA in one-letter codes, chains separated by `/` or new lines, FASTA
headers ignored. The volume is the sum of the residue volumes, or is set by a density (use 1/v̄ for a
known partial specific volume) or a volume.

**Radiation**: neutron wavelength; X-ray energy, wavelength or the Kα line of an anode (Cu, Mo, Ag, Ga,
Cr, Co, Fe, In); X-ray table.

**Matrix**: an H₂O/D₂O mixture, vacuum, or one of the materials.

## Searching a material

*Search…* (Ctrl+F, or `pysasfit-sld --search toluene`) looks a name up and lists formula, density, source
and a note; a double click adds the hit to the materials.

| source | | content |
|---|---|---|
| built-in library | offline | common solvents (also deuterated), polymers, inorganic materials |
| xraydb materials | offline | about 90 entries: elements, solvents, minerals, window materials |
| ORSO SLD database | online | slddb.reflectometry.org: materials collected for reflectometry and SAS, with reference and validation flag; name or formula |
| Wikipedia | online | formula and density of the infobox of the English articles found for the name |
| PubChem | online | formula of a named compound and its experimental densities (handbook values with temperature and source); exact compound names work best |

The densities of Wikipedia and PubChem are read from free text, which is shown in the note: look at it
(temperature, phase, a range) before using the value. A hit without density is added with density 0.
A source that cannot be reached is named below the table and does not stop the others.

```python
from pysasfit.sld.lookup import search, hit_entry
hits, errors = search("toluene", sources=("builtin", "xraydb", "orso"))
```

## Data

| | source | remarks |
|---|---|---|
| neutron scattering lengths, cross sections | `periodictable`: Rauch & Waschkowski (2003), Sears (1992) | per isotope; absorption ∝ wavelength; energy-dependent b of rare-earth isotopes (Lynn & Seeger 1990) |
| X-ray f′, f″ (default) | `xraydb`: Chantler | fine energy grid at the absorption edges, Z ≤ 98 |
| X-ray f₁, f₂ (alternative) | `periodictable`: Henke, Gullikson & Davis (CXRO) | 30 eV – 30 keV, Z ≤ 92 |
| residue formulas and volumes | `periodictable.fasta`: Perkins (1988), volumes as in Jacrot (1976); nucleotides Buckin (1989), Durchschlag & Zipper (1997) | Asp, Glu ionised; Lys, Arg protonated |
| water | 0.99821 (H₂O) and 1.10536 g/cm³ (D₂O) at 20 °C, ideal mixing | |

Both X-ray tables are for free atoms. Within some 10 eV of an edge the real f′, f″ depend on the
chemical state (XANES) and should be measured. The X-ray SLD is the forward value (q = 0, f₀ = Z).
The tabulated bound-atom incoherent cross sections underestimate the scattering of hydrogen for cold
neutrons (H₂O: 5.6 cm⁻¹ calculated, about 7 cm⁻¹ measured), so real transmissions are lower.

## Quantities

With V the volume of the formula unit, b = Σ nᵢ b_c,ᵢ (neutrons) or b = r_e Σ nᵢ (Z + f′ + i f″)ᵢ
(X-rays, r_e = 2.818 fm):

| | |
|---|---|
| SLD | ρ = b / V |
| contrast | Δρ = ρ − ρ₀ (ρ₀: matrix) |
| excess scattering length | Δb = b − ρ₀ V = Δρ V |
| forward scattering | I(0)/c = N_A (Δb)² / M (dilute particles of one formula unit, c in g/cm³) |
| X-rays | δ = λ² Re ρ / 2π, β = λ² Im ρ / 2π, μ = 4πβ/λ (photoabsorption) |
| critical q | q_c = 4 √(π ρ) |
| magnetic SLD | 2.695 fm × (moment in μ_B) / V |

## Hydrogen exchange

In a solvent with the D₂O fraction x a labile hydrogen of class c is deuterium with the probability
x·p_c:

| class | default p | |
|---|---|---|
| backbone amide N-H of proteins | 0.9 | one per residue, except proline and the first residue (Perkins 1986; `DNEXCH` = 0.1 of CRYSON) |
| all other labile H | 1.0 | side-chain and terminal N-H, O-H, S-H; nucleic acids; `H[1]` of formulas |
| H bound to carbon (written `H`) | — | does not exchange; the *deuteration* of the material replaces that fraction by D |

For lysozyme this gives an exchanging fraction of 0.952 of its 264 labile hydrogens, the usual 0.95 of
sequence-based calculators (MULCh, SCOMAP-XD), and a match point of 44.7 % D₂O. SLD of material and
solvent are linear in x, so the match point is exact.

## Python

```python
from pysasfit.sld import Material, Probe, Exchange, water, contrast, match_point, energy_scan

probe = Probe(neutron_wavelength=6.0, xray_energy=8.048, xray_table="chantler")
silica = Material.from_formula("SiO2", density=2.2)
p = silica.properties(probe)
p.volume, p.n_sld, p.x_sld, p.x_mu                 # Å³, 1e-6 Å^-2, 1e-6 Å^-2, 1/cm
c = contrast(silica, water(0.4), probe)
c["n_drho"], c["n_excess"], c["x_excess_electrons"]

lys = Material.from_sequence("KVFGRCELAAAMKRHG...", kind="aa", deuteration=0.0)
match_point(lys, probe, Exchange(backbone=0.9, side=1.0))

import numpy as np
scan = energy_scan(Material.from_formula("Fe3O4", density=5.17), np.linspace(6.9, 7.3, 800))
scan["sld"], scan["sld_imag"], scan["edges"]
```

*Copy Python code* (File menu, Ctrl+Shift+C) writes the script of the current table.
