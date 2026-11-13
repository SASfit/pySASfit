"""Small library of common materials: (name, formula, mass density in g/cm^3 near 20 °C).

Typical literature values, meant as starting points: polymers, glasses and nanoparticles in particular
differ from sample to sample, so check the density of your own material.
"""

__all__ = ["LIBRARY"]

LIBRARY = {
    "Solvents": [
        ("H2O", "H2O", 0.9982),
        ("D2O", "D2O", 1.1054),
        ("toluene", "C7H8", 0.867),
        ("toluene-d8", "C7D8", 0.943),
        ("cyclohexane", "C6H12", 0.779),
        ("cyclohexane-d12", "C6D12", 0.893),
        ("n-decane", "C10H22", 0.730),
        ("n-decane-d22", "C10D22", 0.842),
        ("ethanol", "C2H5OH[1]", 0.789),
        ("tetrahydrofuran", "C4H8O", 0.889),
        ("chloroform", "CHCl3", 1.489),
        ("chloroform-d", "CDCl3", 1.500),
    ],
    "Polymers": [
        ("polystyrene", "C8H8", 1.05),
        ("polystyrene-d8", "C8D8", 1.13),
        ("PMMA", "C5H8O2", 1.18),
        ("PEO / PEG", "C2H4O", 1.13),
        ("PDMS", "C2H6OSi", 0.965),
        ("polyethylene (amorphous)", "C2H4", 0.855),
    ],
    "Inorganic": [
        ("silica (amorphous)", "SiO2", 2.20),
        ("silicon", "Si", 2.329),
        ("gold", "Au", 19.30),
        ("silver", "Ag", 10.49),
        ("magnetite", "Fe3O4", 5.17),
        ("titania (rutile)", "TiO2", 4.23),
        ("alumina (corundum)", "Al2O3", 3.97),
        ("iron", "Fe", 7.874),
        ("nickel", "Ni", 8.908),
        ("aluminium", "Al", 2.699),
        ("vanadium", "V", 6.11),
    ],
}
