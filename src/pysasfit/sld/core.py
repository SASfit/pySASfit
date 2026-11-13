"""Scattering length densities of materials for neutrons and X-rays.

A :class:`Material` is a chemical formula (or a protein / DNA / RNA sequence, :mod:`pysasfit.sld.bio`)
together with the volume of one formula unit. :meth:`Material.properties` returns, for a
:class:`Probe` (neutron wavelength, X-ray energy, X-ray table):

* molar mass, mass density, molecular volume, number density, electrons, electron density
* neutrons: scattering length of the formula unit, SLD (real, imaginary, incoherent), the macroscopic
  coherent / incoherent / absorption cross sections, 1/e length, transmission
* X-rays: f = sum(Z + f' + i f'') at the energy, SLD (real, imaginary), the Thomson (Z only) SLD,
  delta, beta, linear and mass attenuation, transmission, critical q
* with a matrix (solvent): contrast, contrast squared, excess scattering length, match point

Data
----
Neutrons: :mod:`periodictable` (Rauch & Waschkowski 2003 / Sears 1992 isotope table, absorption
proportional to the wavelength, energy-dependent scattering lengths of the rare-earth isotopes after
Lynn & Seeger 1990).

X-rays: Chantler's f', f'' from :mod:`xraydb` (fine energy grid at the absorption edges, Z = 1 ... 98)
or the Henke / CXRO tables of :mod:`periodictable` (30 eV ... 30 keV, Z = 1 ... 92). Both are tables of
free atoms: close to an edge (XANES region) the real values depend on the chemical state.

Hydrogen exchange
-----------------
Exchangeable (labile) hydrogen is written ``H[1]`` in a formula, as in periodictable and the NIST
calculator: ``C3H4H[1]NO`` is an alanine residue with one labile H. In a solvent with the D2O fraction
x, a labile H of class c is a D with probability ``x * p_c`` (:class:`Exchange`: backbone N-H 0.9, other
N-H / O-H / S-H 1.0, the convention of CRYSON). Hydrogen written ``H`` does not exchange; a
``deuteration`` level of the material replaces that fraction of it by D (perdeuteration). The volume of
the formula unit does not depend on the isotopes.

Units: lengths in Å and fm, SLD in 1e-6 Å^-2 (= 1e10 cm^-2), cross sections in cm^-1.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field, replace

import numpy as np

__all__ = ["Probe", "Exchange", "Material", "Properties", "water", "contrast", "match_point", "d2o_scan",
           "energy_scan", "xray_factors", "xray_edges", "xray_source_energy", "have_xraydb", "XRAY_SOURCES",
           "R_E_FM", "N_A", "HC_KEV_A", "H2O_DENSITY", "D2O_DENSITY", "MAGNETIC_FM_PER_MUB"]

R_E_FM = 2.8179403205          # classical electron radius, fm
N_A = 6.02214076e23
HC_KEV_A = 12.398419843        # E[keV] * lambda[Å]
H2O_DENSITY = 0.99821          # g/cm^3 at 20 °C
D2O_DENSITY = 1.10536          # g/cm^3 at 20 °C
MAGNETIC_FM_PER_MUB = 2.6954   # gamma r_e / 2: magnetic scattering length of one Bohr magneton, fm

#: laboratory X-ray sources (K-alpha of the anode element)
XRAY_SOURCES = ("Cu", "Mo", "Ag", "Ga", "Cr", "Co", "Fe", "In")


def _pt():
    try:
        import periodictable
    except ImportError as exc:          # pragma: no cover
        raise ImportError("pySLDcalculator needs the package 'periodictable' "
                          "(pip install pysasfit[sld]  or  pip install periodictable xraydb)") from exc
    return periodictable


def have_xraydb():
    """True if :mod:`xraydb` (Chantler tables, absorption edges) can be imported."""
    try:
        import xraydb  # noqa: F401
    except Exception:
        return False
    return True


def xray_source_energy(element):
    """K-alpha energy (keV) of an anode element, e.g. ``"Cu"`` -> 8.0416."""
    el = _pt().elements.symbol(element)
    return HC_KEV_A / el.K_alpha


# ---------------------------------------------------------------------------------------------- settings
@dataclass(frozen=True)
class Probe:
    """Radiation: neutron wavelength (Å), X-ray energy (keV), X-ray table ('chantler' | 'henke')."""
    neutron_wavelength: float = 6.0
    xray_energy: float = 8.0478
    xray_table: str = "chantler"

    @property
    def xray_wavelength(self):
        return HC_KEV_A / self.xray_energy

    @classmethod
    def from_xray_wavelength(cls, wavelength, **kw):
        return cls(xray_energy=HC_KEV_A / wavelength, **kw)

    @classmethod
    def from_source(cls, element, **kw):
        return cls(xray_energy=xray_source_energy(element), **kw)

    def table(self):
        """The X-ray table actually used ('henke' if xraydb is not installed)."""
        t = self.xray_table.lower()
        if t not in ("chantler", "henke"):
            raise ValueError(f"unknown X-ray table {self.xray_table!r} (chantler, henke)")
        return t if (t == "henke" or have_xraydb()) else "henke"


@dataclass(frozen=True)
class Exchange:
    """Probability that a labile H is exchanged with the solvent, per class of hydrogen.

    backbone   main-chain amide N-H of a protein (0.9: 10 % do not exchange, Perkins 1986, CRYSON)
    side       all other labile H: side-chain and terminal N-H, O-H, S-H, nucleic acids, sugars, ...
    """
    backbone: float = 0.9
    side: float = 1.0

    def probability(self, cls):
        return self.backbone if cls == "backbone" else self.side


# ---------------------------------------------------------------------------------------------- X-ray tables
def xray_factors(symbol, energy, table="chantler"):
    """(f1, f2) of a free atom at ``energy`` (keV, scalar or array): f1 = Z + f' (forward scattering, q = 0),
    f2 = f''. NaN outside the range of the table."""
    pt = _pt()
    energy = np.asarray(energy, float)
    table = table.lower()
    if table == "chantler" and have_xraydb():
        import xraydb
        z = xraydb.atomic_number(symbol)
        ev = np.atleast_1d(energy) * 1000.0
        f1 = z + np.asarray(xraydb.f1_chantler(symbol, ev), float)
        f2 = np.asarray(xraydb.f2_chantler(symbol, ev), float)
        return f1.reshape(energy.shape), f2.reshape(energy.shape)
    el = pt.elements.symbol(symbol)
    f1, f2 = el.xray.scattering_factors(energy=np.atleast_1d(energy))
    f1 = np.asarray(f1, float)
    f2 = np.asarray(f2, float)
    bad = (np.atleast_1d(energy) < 0.03) | (np.atleast_1d(energy) > 30.0) | (f1 < -9000)
    f1 = np.where(bad, np.nan, f1)
    f2 = np.where(bad, np.nan, f2)
    return f1.reshape(energy.shape), f2.reshape(energy.shape)


def xray_edges(symbol, emin=0.0, emax=1e9):
    """{edge name: energy in keV} of an element between ``emin`` and ``emax`` (keV); needs xraydb (else {})."""
    if not have_xraydb():
        return {}
    import xraydb
    out = {}
    for name, e in xraydb.xray_edges(symbol).items():
        kev = e.energy / 1000.0
        if emin <= kev <= emax:
            out[name] = kev
    return out


# ---------------------------------------------------------------------------------------------- material
def _element_of(atom):
    """Element of an atom / isotope / ion of periodictable."""
    return getattr(atom, "element", atom)


def _is_labile(atom):
    return getattr(atom, "isotope", None) == 1 and _element_of(atom).number == 1


def _is_plain_h(atom):
    return _element_of(atom).number == 1 and getattr(atom, "isotope", None) is None


@dataclass
class Material:
    """A formula unit and its volume.

    name         label
    atoms        {periodictable atom: count}; labile hydrogen as ``H[1]``
    volume       volume of the formula unit, Å^3 (the same for all isotopic variants)
    labile       {class: number of labile H}; classes 'backbone' and 'side' (see :class:`Exchange`);
                 the sum is the number of ``H[1]`` in ``atoms``
    deuteration  fraction of the non-exchangeable H replaced by D (0 ... 1)
    source       text the material was made from (formula or sequence), kind: 'formula', 'aa', 'dna', 'rna'
    moment       magnetic moment per formula unit in Bohr magnetons (optional, for the magnetic SLD)
    """
    name: str
    atoms: dict
    volume: float
    labile: dict = field(default_factory=dict)
    deuteration: float = 0.0
    source: str = ""
    kind: str = "formula"
    moment: float = 0.0

    # ------------------------------------------------------------------ constructors
    @classmethod
    def from_formula(cls, formula, density=None, volume=None, name=None, deuteration=0.0, moment=0.0,
                     natural_density=False):
        """From a chemical formula in the grammar of periodictable.

        ``density`` (g/cm^3) is the mass density of the formula as written (labile ``H[1]`` as H, before
        ``deuteration``); it may also be part of the formula (``"SiO2@2.2"``). ``volume`` (Å^3 per formula
        unit) can be given instead.

        ``natural_density=True``: ``density`` is that of the material with natural isotopes, e.g. the
        density of the protonated compound for a formula written with D (``C7D8`` with the density of
        C7H8). The volume of the formula unit is then that of the natural compound and the density of the
        isotopic one follows from its mass (the ``@0.867n`` of periodictable). Without it a deuterated
        formula with the protonated density gets a volume that is too large and an SLD that is too low.

        Examples: ``H2O``, ``D2O``, ``C6H5CH3``, ``Fe3O4``, ``Ca[44]CO3``, ``C3H4H[1]NO`` (labile H),
        ``3%wt NaCl // H2O`` (mixture: give the density of the mixture; the formula unit is the mixed one).
        """
        pt = _pt()
        f = pt.formula(formula)
        atoms = {a: float(n) for a, n in f.atoms.items() if n}
        if not atoms:
            raise ValueError("empty formula")
        mass = sum(n * a.mass for a, n in atoms.items())
        if volume is None and natural_density and density is not None:
            mass = sum(n * _element_of(a).mass for a, n in atoms.items())      # natural isotopes
        if volume is None:
            rho = density if density is not None else f.density
            if rho is None or not rho > 0:
                raise ValueError(f"'{formula}': give the mass density (g/cm³) or the volume of the formula "
                                 f"unit (Å³)")
            volume = mass / (rho * N_A) * 1e24
        elif not volume > 0:
            raise ValueError("the volume must be positive")
        n_lab = sum(n for a, n in atoms.items() if _is_labile(a))
        return cls(name or str(formula), atoms, float(volume), {"side": n_lab} if n_lab else {},
                   float(deuteration), str(formula), "formula", float(moment))

    @classmethod
    def from_sequence(cls, sequence, kind="aa", name=None, deuteration=0.0, volume=None):
        """From a protein ('aa'), DNA ('dna') or RNA ('rna') sequence; see :func:`pysasfit.sld.bio.sequence`."""
        from .bio import sequence as _seq
        return _seq(sequence, kind=kind, name=name, deuteration=deuteration, volume=volume)

    # ------------------------------------------------------------------ composition
    @property
    def n_labile(self):
        return sum(n for a, n in self.atoms.items() if _is_labile(a))

    def exchanged_fraction(self, exchange=Exchange()):
        """Fraction of the labile H that exchanges in pure D2O (1 if there are none)."""
        n = self.n_labile
        if not n:
            return 1.0
        known = sum(self.labile.values())
        ex = sum(c * exchange.probability(k) for k, c in self.labile.items())
        ex += max(n - known, 0.0) * exchange.side          # labile H without a class
        return min(ex / n, 1.0)

    def variant(self, d2o=0.0, exchange=Exchange()):
        """{atom: count} of the material in a solvent with the D2O fraction ``d2o``: labile H exchanged,
        non-exchangeable H deuterated to ``self.deuteration``."""
        pt = _pt()
        D = pt.D
        x = float(d2o) * self.exchanged_fraction(exchange)
        d = float(self.deuteration)
        out = {}
        for a, n in self.atoms.items():
            if _is_labile(a) and x > 0:
                out[a] = out.get(a, 0.0) + n * (1 - x)
                out[D] = out.get(D, 0.0) + n * x
            elif _is_plain_h(a) and d > 0:
                out[a] = out.get(a, 0.0) + n * (1 - d)
                out[D] = out.get(D, 0.0) + n * d
            else:
                out[a] = out.get(a, 0.0) + n
        return {a: n for a, n in out.items() if n > 0}

    def hill(self, d2o=0.0, exchange=Exchange()):
        """Formula in Hill notation (labile H shown as H[1])."""
        pt = _pt()
        return str(pt.formula(self.variant(d2o, exchange)).hill)

    def mass(self, d2o=0.0, exchange=Exchange()):
        """Molar mass (g/mol) of the variant."""
        return sum(n * a.mass for a, n in self.variant(d2o, exchange).items())

    def with_(self, **kw):
        return replace(self, **kw)

    # ------------------------------------------------------------------ the calculation
    def properties(self, probe=Probe(), d2o=0.0, exchange=Exchange(), thickness=0.1):
        """All quantities of the material (:class:`Properties`). ``d2o``: D2O fraction of the surrounding
        solvent (labile H); ``thickness`` (cm) for the transmissions."""
        pt = _pt()
        from periodictable import nsf

        atoms = self.variant(d2o, exchange)
        V = self.volume
        M = sum(n * a.mass for a, n in atoms.items())
        rho = M / (V * N_A) * 1e24
        f = pt.formula(atoms)
        p = Properties(name=self.name, formula=str(f.hill), mass=M, density=rho, volume=V, d2o=float(d2o),
                       probe=probe, thickness=thickness, n_labile=self.n_labile,
                       exchanged=self.exchanged_fraction(exchange), deuteration=self.deuteration)
        p.n_atoms = sum(atoms.values())
        p.electrons = sum(n * _element_of(a).number for a, n in atoms.items())

        # ---- neutrons
        lam = probe.neutron_wavelength
        sld, xs, pen = nsf.neutron_scattering(f, density=rho, wavelength=lam)
        if sld[0] is None:
            raise ValueError(f"no neutron scattering length for an element of {p.formula}")
        p.n_sld, p.n_sld_imag, p.n_sld_inc = float(sld[0]), float(sld[1]), float(sld[2])
        p.n_xs_coh, p.n_xs_abs, p.n_xs_inc = float(xs[0]), float(xs[1]), float(xs[2])
        p.n_penetration = float(pen)
        p.n_b = p.n_sld * V * 0.1                    # 1e-6 Å^-2 * Å^3 -> fm
        p.n_b_imag = p.n_sld_imag * V * 0.1

        # ---- X-rays
        table = probe.table()
        p.xray_table = table
        E = probe.xray_energy
        lx = probe.xray_wavelength
        f1 = f2 = 0.0
        rows = []
        for a, n in sorted(atoms.items(), key=lambda t: (_element_of(t[0]).number, str(t[0]))):
            el = _element_of(a)
            a1, a2 = xray_factors(el.symbol, E, table)
            f1 += n * float(a1)
            f2 += n * float(a2)
            neu = a.neutron
            bc = getattr(neu, "b_c", None)
            sig_a = getattr(neu, "absorption", None)
            rows.append({"atom": str(a), "Z": el.number, "n": n, "mass": a.mass,
                         "mass_fraction": n * a.mass / M,
                         "b_c": bc, "b_c_imag": getattr(neu, "b_c_i", None),
                         "sigma_inc": getattr(neu, "incoherent", None),
                         "sigma_abs": None if sig_a is None else sig_a * lam / nsf.ABSORPTION_WAVELENGTH,
                         "f1": float(a1), "f2": float(a2)})
        p.elements = rows
        p.x_f1, p.x_f2 = f1, f2
        re_A = R_E_FM * 1e-5
        p.x_sld = re_A * f1 / V * 1e6
        p.x_sld_imag = re_A * f2 / V * 1e6
        p.x_sld_thomson = re_A * p.electrons / V * 1e6
        p.x_b = R_E_FM * f1
        p.x_b_imag = R_E_FM * f2
        p.x_delta = lx ** 2 * p.x_sld * 1e-6 / (2 * math.pi)
        p.x_beta = lx ** 2 * p.x_sld_imag * 1e-6 / (2 * math.pi)
        p.x_mu = 2 * lx * p.x_sld_imag * 1e-6 * 1e8          # 4 pi beta / lambda, cm^-1
        p.magnetic_sld = MAGNETIC_FM_PER_MUB * self.moment / V * 10.0   # fm / Å^3 -> 1e-6 Å^-2
        p.moment = self.moment
        return p


# ---------------------------------------------------------------------------------------------- results
@dataclass
class Properties:
    """Result of :meth:`Material.properties`. SLDs in 1e-6 Å^-2, scattering lengths in fm, cross sections in
    cm^-1, lengths in cm unless stated."""
    name: str
    formula: str
    mass: float                 # g/mol
    density: float              # g/cm^3
    volume: float               # Å^3 per formula unit
    d2o: float
    probe: Probe
    thickness: float            # cm
    n_labile: float = 0.0
    exchanged: float = 1.0
    deuteration: float = 0.0
    n_atoms: float = 0.0
    electrons: float = 0.0
    n_sld: float = 0.0
    n_sld_imag: float = 0.0
    n_sld_inc: float = 0.0
    n_xs_coh: float = 0.0
    n_xs_abs: float = 0.0
    n_xs_inc: float = 0.0
    n_penetration: float = 0.0
    n_b: float = 0.0
    n_b_imag: float = 0.0
    xray_table: str = ""
    x_f1: float = 0.0
    x_f2: float = 0.0
    x_sld: float = 0.0
    x_sld_imag: float = 0.0
    x_sld_thomson: float = 0.0
    x_b: float = 0.0
    x_b_imag: float = 0.0
    x_delta: float = 0.0
    x_beta: float = 0.0
    x_mu: float = 0.0
    magnetic_sld: float = 0.0
    moment: float = 0.0
    elements: list = field(default_factory=list)

    # ---- derived
    @property
    def molar_volume(self):                 # cm^3/mol
        return self.volume * N_A * 1e-24

    @property
    def specific_volume(self):              # cm^3/g
        return 1.0 / self.density

    @property
    def number_density(self):               # formula units per Å^3
        return 1.0 / self.volume

    @property
    def electron_density(self):             # e/Å^3
        return self.electrons / self.volume

    @property
    def n_transmission(self):
        return math.exp(-self.thickness / self.n_penetration) if self.n_penetration > 0 else float("nan")

    @property
    def n_xs_total(self):
        return self.n_xs_coh + self.n_xs_abs + self.n_xs_inc

    @property
    def x_mu_mass(self):                    # cm^2/g
        return self.x_mu / self.density

    @property
    def x_attenuation_length(self):         # cm
        return 1.0 / self.x_mu if self.x_mu > 0 else float("inf")

    @property
    def x_transmission(self):
        return math.exp(-self.x_mu * self.thickness)

    @staticmethod
    def _qc(sld):
        return 4 * math.sqrt(math.pi * sld * 1e-6) if sld > 0 else float("nan")

    @property
    def n_qc(self):                         # Å^-1, total reflection below
        return self._qc(self.n_sld)

    @property
    def x_qc(self):
        return self._qc(self.x_sld)

    @property
    def x_critical_angle(self):             # degrees
        return math.degrees(math.sqrt(2 * self.x_delta)) if self.x_delta > 0 else float("nan")

    def rows(self):
        """[(group, quantity, value, unit)] for tables and printing."""
        pr = self.probe
        r = [
            ("General", "formula (Hill)", self.formula, ""),
            ("General", "molar mass M", self.mass, "g/mol"),
            ("General", "mass density", self.density, "g/cm³"),
            ("General", "volume of the formula unit V", self.volume, "Å³"),
            ("General", "molar volume", self.molar_volume, "cm³/mol"),
            ("General", "specific volume", self.specific_volume, "cm³/g"),
            ("General", "number density", self.number_density * 1e3, "1/nm³"),
            ("General", "atoms per formula unit", self.n_atoms, ""),
            ("General", "electrons per formula unit", self.electrons, ""),
            ("General", "electron density", self.electron_density, "e/Å³"),
            ("General", "labile H", self.n_labile, ""),
            ("General", "exchanging fraction of the labile H", self.exchanged, ""),
            ("General", "deuteration of non-exchangeable H", self.deuteration, ""),
            ("General", "D₂O fraction of the solvent", self.d2o, ""),
            ("Neutrons", "wavelength", pr.neutron_wavelength, "Å"),
            ("Neutrons", "scattering length b = Σ b_c", self.n_b, "fm"),
            ("Neutrons", "SLD", self.n_sld, "10⁻⁶ Å⁻²"),
            ("Neutrons", "SLD imaginary", self.n_sld_imag, "10⁻⁶ Å⁻²"),
            ("Neutrons", "SLD incoherent", self.n_sld_inc, "10⁻⁶ Å⁻²"),
            ("Neutrons", "Σ coherent", self.n_xs_coh, "1/cm"),
            ("Neutrons", "Σ incoherent", self.n_xs_inc, "1/cm"),
            ("Neutrons", "Σ absorption", self.n_xs_abs, "1/cm"),
            ("Neutrons", "Σ total", self.n_xs_total, "1/cm"),
            ("Neutrons", "1/e length", self.n_penetration, "cm"),
            ("Neutrons", f"transmission ({self.thickness:g} cm)", self.n_transmission, ""),
            ("Neutrons", "incoherent background Σ_inc/4π", self.n_xs_inc / (4 * math.pi), "1/cm/sr"),
            ("Neutrons", "critical q", self.n_qc, "1/Å"),
            ("X-rays", "energy", pr.xray_energy, "keV"),
            ("X-rays", "wavelength", pr.xray_wavelength, "Å"),
            ("X-rays", "table", self.xray_table, ""),
            ("X-rays", "f₁ = Σ (Z + f′)", self.x_f1, "e"),
            ("X-rays", "f₂ = Σ f″", self.x_f2, "e"),
            ("X-rays", "scattering length r_e f₁", self.x_b, "fm"),
            ("X-rays", "SLD", self.x_sld, "10⁻⁶ Å⁻²"),
            ("X-rays", "SLD imaginary", self.x_sld_imag, "10⁻⁶ Å⁻²"),
            ("X-rays", "SLD Thomson (Z only)", self.x_sld_thomson, "10⁻⁶ Å⁻²"),
            ("X-rays", "δ", self.x_delta, ""),
            ("X-rays", "β", self.x_beta, ""),
            ("X-rays", "linear attenuation μ", self.x_mu, "1/cm"),
            ("X-rays", "mass attenuation μ/ρ", self.x_mu_mass, "cm²/g"),
            ("X-rays", "attenuation length 1/μ", self.x_attenuation_length * 1e4, "µm"),
            ("X-rays", f"transmission ({self.thickness:g} cm)", self.x_transmission, ""),
            ("X-rays", "critical q", self.x_qc, "1/Å"),
            ("X-rays", "critical angle", self.x_critical_angle, "°"),
        ]
        if self.moment:
            r += [("Magnetic", "moment per formula unit", self.moment, "μ_B"),
                  ("Magnetic", "magnetic SLD", self.magnetic_sld, "10⁻⁶ Å⁻²")]
        return r


# ---------------------------------------------------------------------------------------------- solvent
def water(d2o=0.0, name=None):
    """H2O / D2O mixture at 20 °C with the D2O volume fraction ``d2o`` (ideal mixing; the molar volumes
    of H2O and D2O differ by 0.3 %, so volume and mole fraction are taken as equal)."""
    pt = _pt()
    x = float(d2o)
    if not 0 <= x <= 1:
        raise ValueError("the D2O fraction must be between 0 and 1")
    vh = pt.formula("H2O").mass / (H2O_DENSITY * N_A) * 1e24
    vd = pt.formula("D2O").mass / (D2O_DENSITY * N_A) * 1e24
    atoms = {pt.O: 1.0}
    if x < 1:
        atoms[pt.H] = 2 * (1 - x)
    if x > 0:
        atoms[pt.D] = 2 * x
    if name is None:
        name = "H2O" if x == 0 else "D2O" if x == 1 else f"{100 * x:g}% D2O"
    return Material(name, atoms, (1 - x) * vh + x * vd, source=f"water, D2O fraction {x:g}", kind="water")


# ---------------------------------------------------------------------------------------------- contrast
def contrast(material, matrix, probe=Probe(), d2o=0.0, exchange=Exchange(), thickness=0.1):
    """Contrast of ``material`` in ``matrix`` (a :class:`Material`, or None for vacuum).

    Returns a dict: for neutrons ('n_...') and X-rays ('x_...')

    drho     SLD difference, 1e-6 Å^-2
    drho2    its square, 1e-12 Å^-4 (= 1e20 cm^-4)
    excess   excess scattering length of one formula unit, b - rho_matrix * V, in fm
    excess_per_mass   excess scattering length per mass, cm/g
    i0       forward scattering per mass concentration, I(0)/c = N_A (Δb)² / M, in cm²/g
             (dilute, non-interacting particles of one formula unit each; I(0) in cm^-1 for c in g/cm³)

    plus 'x_excess_electrons' (excess electrons, f1 - rho_e,matrix V), 'material' and 'matrix'
    (:class:`Properties`).
    """
    p = material.properties(probe, d2o, exchange, thickness)
    m = matrix.properties(probe, d2o, exchange, thickness) if matrix is not None else None
    out = {"material": p, "matrix": m}
    for k in ("n", "x"):
        rp = getattr(p, f"{k}_sld")
        rm = getattr(m, f"{k}_sld") if m is not None else 0.0
        d = rp - rm
        ex = d * p.volume * 0.1                              # fm
        out[f"{k}_drho"] = d
        out[f"{k}_drho2"] = d * d
        out[f"{k}_excess"] = ex
        out[f"{k}_excess_per_mass"] = ex * 1e-13 * N_A / p.mass
        out[f"{k}_i0"] = (ex * 1e-13) ** 2 * N_A / p.mass
    out["x_excess_electrons"] = out["x_excess"] / R_E_FM
    return out


def match_point(material, probe=Probe(), exchange=Exchange()):
    """D2O fraction of an H2O/D2O solvent at which the neutron SLD of ``material`` equals that of the
    solvent (labile H exchanged with the solvent). Outside 0 ... 1 the material cannot be matched; NaN if
    the two lines are parallel."""
    p0 = material.properties(probe, 0.0, exchange).n_sld
    p1 = material.properties(probe, 1.0, exchange).n_sld
    s0 = water(0.0).properties(probe).n_sld
    s1 = water(1.0).properties(probe).n_sld
    den = (s1 - s0) - (p1 - p0)
    return (p0 - s0) / den if den else float("nan")


def d2o_scan(material, probe=Probe(), exchange=Exchange(), n=101):
    """Neutron SLD of the material and of the H2O/D2O solvent against the D2O fraction.
    Returns ``(x, sld_material, sld_solvent)``; the contrast is their difference."""
    x = np.linspace(0.0, 1.0, n)
    p0 = material.properties(probe, 0.0, exchange).n_sld
    p1 = material.properties(probe, 1.0, exchange).n_sld
    s0 = water(0.0).properties(probe).n_sld
    s1 = water(1.0).properties(probe).n_sld
    return x, p0 + x * (p1 - p0), s0 + x * (s1 - s0)        # both exactly linear in x


def energy_scan(material, energies, table="chantler", d2o=0.0, exchange=Exchange()):
    """X-ray quantities against the energy (keV, array).

    Returns a dict of arrays: 'energy', 'f1', 'f2' (per formula unit, electrons), 'sld', 'sld_imag'
    (1e-6 Å^-2), 'mu' (1/cm), and 'edges': [(element, edge, keV)] inside the range (xraydb).
    """
    E = np.asarray(energies, float)
    table = Probe(xray_table=table).table()
    atoms = material.variant(d2o, exchange)
    f1 = np.zeros_like(E)
    f2 = np.zeros_like(E)
    edges = []
    per_element = {}
    for a, n in atoms.items():
        sym = _element_of(a).symbol
        per_element[sym] = per_element.get(sym, 0.0) + n
    for sym, n in per_element.items():
        a1, a2 = xray_factors(sym, E, table)
        f1 += n * a1
        f2 += n * a2
        for name, e in xray_edges(sym, float(E.min()), float(E.max())).items():
            edges.append((sym, name, e))
    re_A = R_E_FM * 1e-5
    V = material.volume
    lam = HC_KEV_A / E
    sld_i = re_A * f2 / V * 1e6
    return {"energy": E, "f1": f1, "f2": f2, "sld": re_A * f1 / V * 1e6, "sld_imag": sld_i,
            "mu": 2 * lam * sld_i * 1e-6 * 1e8, "edges": sorted(edges, key=lambda t: t[2]), "table": table}
