"""pySLDcalculator: scattering length densities for neutrons and X-rays.

core      :class:`Material` (formula + volume of the formula unit), :class:`Probe`, :class:`Exchange`,
          :func:`contrast` (contrast, excess scattering length), :func:`match_point`, :func:`d2o_scan`,
          :func:`energy_scan` (anomalous X-ray scattering across absorption edges), :func:`water`
bio       proteins, DNA and RNA from their sequence (residue volumes, labile hydrogens)
library   common solvents, polymers and inorganic materials with typical densities
lookup    search a material by name: this library, the xraydb table, and online the ORSO SLD database,
          Wikipedia (infobox) and PubChem
cli       ``pysasfit-sld`` on the command line
gui       Qt window (``pysasfit-sld`` without arguments and the "pySLDcalculator" launcher tab)

    from pysasfit.sld import Material, Probe, water, contrast
    sio2 = Material.from_formula("SiO2", density=2.2)
    c = contrast(sio2, water(0.0), Probe(neutron_wavelength=6.0, xray_energy=8.048))
    c["material"].n_sld, c["n_drho"], c["n_excess"]

Needs ``periodictable`` (neutron table, Henke X-ray table, formula grammar) and, for Chantler's X-ray
table and the absorption edges, ``xraydb``:  ``pip install pysasfit[sld]``.
"""
from .core import (
    Exchange,
    Material,
    Probe,
    Properties,
    contrast,
    d2o_scan,
    energy_scan,
    have_xraydb,
    match_point,
    water,
    xray_edges,
    xray_factors,
    xray_source_energy,
)

__all__ = ["Material", "Probe", "Exchange", "Properties", "water", "contrast", "match_point", "d2o_scan",
           "energy_scan", "xray_factors", "xray_edges", "xray_source_energy", "have_xraydb"]
