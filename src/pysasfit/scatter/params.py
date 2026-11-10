"""Parameter model of the GUI: names, defaults, ranges and combo-box lists follow
CrystalScatter/Source/sc_calc_generic.cpp (slGUI) so that the parameter (.ini) files stay compatible."""
from collections import OrderedDict
from urllib.parse import quote, unquote

LTYPES = ["Lamellae", "Hexagonally packed cylinders (P6/mm)", "Square packed cylinders (P4/mm)",
          "Rectangular centered cylinders (cmm)", "BCC (Im3m)", "FCC (Fm3m)", "HCP (P6/mmc)", "SC (Pm3m)",
          "BCT (I4/mm)", "Gyroid (Ia3d)", "OBDD (Pn3m)", "Plumbers Nightmare (Im3m)", "None", "CP-Layers",
          "2DHex (GISAXS)", "2DSquare (GISAXS)", "1DLam (GISAXS)", "Fd3m", "Orthorombic", "LDQ12"]
LTYPE_KIND = {0: "lamellae", 1: "hexcyl", 2: "squarecyl", 3: "centered_rect", 4: "bcc", 5: "fcc", 6: "hcp",
              7: "sc", 8: "bct", 9: "gyroid", 10: "obdd", 11: "plumbers", 12: None, 13: None,
              14: "hexcyl", 15: "squarecyl", 16: "lamellae", 17: "fd3m", 18: "orthorhombic", 19: None}
PARTICLES = ["sphere", "cylinder", "disk", "vesicle", "cube", "ellipsoid", "triaxial ellipsoid",
             "super ellipsoid, barrel", "superball", "excluded volume chain", "Kratky Porod chain",
             "Teubner-Strey (microemulsion)"]
ORDIS = ["Gaussian", "Exponential", "Onsager", "Maier-Saupe", "Cut-off", "Laguerre", "z-dir", "isotropic",
         "mirrored Gaussian", "mirrored Exponential", "mirrored Onsager", "mirrored Maier-Saupe",
         "mirrored Cut-off", "fiber pattern"]
INTERIOR = ["homogeneous", "core + homogeneous shell", "core + inhomogeneous shell", "multi-shell", "myelin"]
PEAKS = ["Lorentzian", "Gaussian", "mod. 1 Lorentzian", "mod. 2 Lorentzian", "Pseudo-Voigt", "Pearson VII",
         "Gamma", "Anisotropic Gaussian"]
# pySCAtter additions (not in SasScatter2 .ini files; absent keys take the defaults)
STRUCTURE_FACTOR_NAMES = ["none", "hard spheres (Percus-Yevick)", "sticky hard spheres (Baxter)",
                          "mass fractal (Teixeira)"]
SHELL_PROFILES = ["linear gradient", "power law (r/R)^alpha"]
COMBOS = OrderedDict(LType=LTYPES, ComboBoxParticle=PARTICLES, Ordis=ORDIS, ComboBoxInterior=INTERIOR,
                     ComboBoxPeak=PEAKS, StructureFactor=STRUCTURE_FACTOR_NAMES, ShellProfile=SHELL_PROFILES)
COMBO_DEFAULT = dict(LType=12, ComboBoxParticle=0, Ordis=7, ComboBoxInterior=0, ComboBoxPeak=1, StructureFactor=0,
                     ShellProfile=0)

# key, label, default, min, max, decimals, tooltip   (fit flag: every numeric key can carry a FITFLAG_)
NUM = OrderedDict()
GROUPS = OrderedDict()


def _g(group, rows):
    GROUPS[group] = [r[0] for r in rows]
    for r in rows:
        NUM[r[0]] = r


_g("Lattice", [
    ("uca", "uca:", 30, 0.01, 1000, 2, "Unit cell dimension a (nm)"),
    ("ucb", "ucb:", 30, 0.01, 1000, 2, "Unit cell dimension b (nm)"),
    ("ucc", "ucc:", 30, 0.01, 1000, 2, "Unit cell dimension c (nm)"),
    ("ucalpha", "ucalpha:", 90, 0, 360, 3, "Unit cell angle alpha (deg)"),
    ("ucbeta", "ucbeta:", 90, 0, 360, 3, "Unit cell angle beta (deg)"),
    ("ucgamma", "ucgamma:", 90, 0, 360, 3, "Unit cell angle gamma (deg)"),
    ("EditCeff", "TwRatio:", 1, 0, 1, 4, "Effective volume fraction / twin ratio"),
    ("EditCeffcyl", "Ceffcyl:", 0, 0, 999, 4, "Ceffcyl"),
    ("reff", "reff:", 0, -1e6, 1e6, 4, "Effective radius"),
    ("acpl", "acpl:", 0, 0, 1, 5, "Stacking probability A(s)"),
    ("bcpl", "bcpl:", 0, 0, 1, 5, "Stacking probability B(s)"),
    ("ifluc", "ifluc:", 0, 0, 1000, 4, "ifluc"),
    ("rfluc", "rfluc:", 0, 0, 1000, 4, "rfluc"),
])
_g("Particle", [
    ("EditRadius", "EditRadius:", 1.0, 0, 1000, 4, "Inner radius (nm)"),
    ("EditRadiusi", "EditRadiusi:", 0.5, 0, 1000, 4, "Outer radius / shell thickness (nm)"),
    ("EditSigma", "EditSigma:", 0.07, 0, 5, 4, "Relative standard deviation of the radius"),
    ("EditDbeta", "EditDBeta:", 0.4, 0, 360, 4, "Average angle (deg), width of the orientation distribution"),
    ("Length", "Length:", 1, 0, 1000, 4, "Structure length (nm)"),
    ("SigmaL", "SigmaL:", 0.06, 0, 5, 4, "Relative standard deviation of the length"),
    ("Alpha", "Alpha:", 0, -1e6, 1e6, 4, "Internal 'alphash' (superball: exponent p, 0 -> 2)"),
    ("EditRho", "EditRho:", 0, -1e6, 1e6, 4, "Relative contrast of the shell"),
    ("RotAlpha", "RotAlpha:", 0, -1e6, 1e6, 4, "Internal 'alpha'"),
])
_g("Interior", [
    ("ShellN", "shells n:", 3, 1, 200, 0, "multi-shell: number of shells; myelin: number of bilayers"),
    ("ShellThick", "shell thickness:", 2.0, 0, 1000, 4, "multi-shell: thickness of each shell; myelin: exterior "
                                                         "(water) layer thickness (nm)"),
    ("ShellRho2", "rho 2nd shell:", 0.0, -1e6, 1e6, 4, "multi-shell: contrast of every second shell "
                                                        "(the others: EditRho)"),
    ("ProfileAlpha", "profile alpha:", -1.3333, -10, 10, 4, "power-law shell profile rho ~ (r/R_core)^alpha "
                                                             "(star/micelle in good solvent: -4/3)"),
])
_g("Structure factor", [
    ("SFRadius", "SF radius:", 5.0, 1e-3, 1e5, 4, "hard-sphere radius / fractal building-block radius r0 (nm)"),
    ("SFEta", "SF eta:", 0.2, 0, 0.7, 4, "volume fraction"),
    ("SFTau", "SF tau:", 0.5, 1e-4, 1e6, 4, "stickiness parameter tau (large: hard spheres)"),
    ("SFDelta", "SF delta:", 0.01, 1e-4, 0.4, 4, "relative width of the attractive well"),
    ("SFDim", "fractal D:", 2.0, 1.01, 2.99, 4, "fractal dimension"),
    ("SFXi", "fractal xi:", 50.0, 1e-3, 1e6, 4, "fractal cut-off length (nm)"),
])
_g("Peak Shape", [
    ("EditPeakPar", "EditPeakPar:", 0, 0, 1e6, 4, "Peak shape parameter (Pseudo-Voigt: eta, Pearson VII: beta, Gamma: nu)"),
    ("EditDebyeWaller", "EditDebyeWaller:", 2, 0, 100, 3, "Average displacement from ideal lattice point (nm)"),
    ("EditAzi", "EditAzi:", 180, 0, 1e6, 4, "Azimuthal domain size D_psi (nm): reflections spread over an arc of width 4/D_psi on the sphere |q| = g (0: no azimuthal dependence, rings)"),
    ("EditDomainSize", "EditDomainSize:", 400, 0, 5000, 2, "Radial domain size D (nm); peak width delta = 4/D"),
    ("VAx1", "x:", 1, -1000, 1000, 3, "u1 (x)"), ("VAx2", "y:", 0, -1000, 1000, 3, "u1 (y)"),
    ("VAx3", "z:", 0, -1000, 1000, 3, "u1 (z)"),
    ("Ay1", "x:", 0, -1000, 1000, 3, "u2 (x)"), ("Ay2", "y:", 1, -1000, 1000, 3, "u2 (y)"),
    ("Ay3", "z:", 0, -1000, 1000, 3, "u2 (z)"),
    ("Az1", "x:", 0, -1000, 1000, 3, "u3 (x)"), ("Az2", "y:", 0, -1000, 1000, 3, "u3 (y)"),
    ("Az3", "z:", 1, -1000, 1000, 3, "u3 (z)"),
    ("SigX", "Sig x:", 40, 0.01, 9999, 3, "Average domain size X"),
    ("SigY", "Sig y:", 40, 0.01, 9999, 3, "Average domain size Y"),
    ("SigZ", "Sig z:", 40, 0.01, 9999, 3, "Average domain size Z"),
])
_g("Orientation", [
    ("ucpsi", "ucpsi:", 0, 0, 360, 3, "Rotation of the crystal about the zone axis (deg)"),
    ("ucn1", "ucn1:", 1, -1000, 1000, 3, "zone axis u"), ("ucn2", "ucn2:", 1, -1000, 1000, 3, "zone axis v"),
    ("ucn3", "ucn3:", 1, -1000, 1000, 3, "zone axis w"),
    ("theta", "theta:", 90, 0, 360, 3, "polar angle of the director (deg)"),
    ("phi", "phi:", 0, 0, 360, 3, "azimuth of the director (deg)"),
    ("rotTheta", "rotTheta:", 0, 0, 360, 3, "rotTheta"), ("rotPhi", "rotPhi:", 0, 0, 360, 3, "rotPhi"),
])
_g("Control", [
    ("EditBFactor", "EditBFactor:", 0.01, 0, 1e6, 5, "Debye-Waller-like B factor, exp(-B q^2) on the lattice term"),
    ("P1", "P1:", 0, -1e6, 1e6, 4, "P1"),
])
_g("Pixel manipulation", [
    ("I0", "I0:", 1, -1e12, 1e12, 5, "Intensity scale"),
    ("Base", "Base:", 0, -1e12, 1e12, 5, "Constant background"),
    ("iso", "iso:", 0, -1e12, 1e12, 5, "Isotropic (powder-like) additive term, weighted by P(q)"),
])
_g("Calculation", [
    ("EditQmax", "Qmax:", 2, 0, 1e4, 4, "Maximum q (1/nm), used when 'Preset' is selected"),
    ("EditQmin", "Qmin:", 0.01, 0, 1e4, 5, "Minimum q (1/nm), 1D calculation"),
    ("EditQsteps", "number of steps (q):", 200, 2, 100000, 0, "1D calculation: number of q values"),
    ("GridPoints", "Grid points:", 128, 16, 2048, 0, "Grid points of the (quadrant) image"),
    ("HKLmax", "HKLmax:", 3, 1, 9, 0, "Maximum |h|,|k|,|l|"),
    ("Editqz", "Editqz:", 0, -1e4, 1e4, 4, "qz of the detector plane"),
])
_g("Experiment", [
    ("EditWavelength", "Wavelength:", 0.154, 1e-4, 100, 5, "nm"),
    ("EditDet", "EditDet:", 10, 1e-3, 1e4, 4, "Sample-detector distance (m)"),
    ("EditPixelX", "PixelSize X:", 0.0025, 1e-6, 10, 6, "m"),
    ("EditPixelY", "PixelSize Y:", 0.0025, 1e-6, 10, 6, "m"),
    ("EditPixelNoX", "PixelNo. X:", 128, 4, 8192, 0, "number of pixels"),
    ("EditPixelNoY", "PixelNo. Y:", 128, 4, 8192, 0, "number of pixels"),
    ("BeamPosX", "Beam X:", 0, -1e5, 1e5, 3, "beam position offset (pixels)"),
    ("BeamPosY", "Beam Y:", 0, -1e5, 1e5, 3, "beam position offset (pixels)"),
])
# RadButDebyeScherrer: peak width 4/D (on by default in pySCAtter; SasScatter2 files that switch it off get the
# fixed width 1 as in CrystalScatter); RadioButtonPara: paracrystal width 4/D + sigma_DW^2/dist q^2
BOOLS = OrderedDict(CheckBoxTwinned=False, CheckBoxWAXS=False, RadButDebyeScherrer=True, RadioButtonPara=False,
                    EditQmaxData=False, EditQmaxPreset=True, ExpandImage=False)
# 1D / 2D, quadrant factor (1,2,4), new/last image window, centre used
EXTRA = dict(QuadrantFactor=4, NewWindow=True, CenterBeam=True, Dimension=2, Comment="")


def defaults():
    d = {k: r[2] for k, r in NUM.items()}
    d.update(COMBO_DEFAULT)
    d.update(BOOLS)
    d.update(EXTRA)
    return d


def write_ini(path, p, fit=None, section="hypscat"):
    """Write a parameter file (same key=value layout as CrystalScatter ``*.ini``, ``FITFLAG_<key>=NN``)."""
    fit = fit or {}
    with open(path, "w", encoding="utf-8") as f:
        f.write("[%s]\n" % quote(section))
        for k, v in p.items():
            if isinstance(v, bool):
                v = "true" if v else "false"
            f.write("%s=%s\n" % (k, v))
            if k in NUM:
                f.write("FITFLAG_%s=%s\n" % (k, "10" if fit.get(k) else "00"))


def read_ini(path):
    """Read an .ini written by this GUI or by SasScatter2.  Returns (params, fitflags, section)."""
    p, fit, section = defaults(), {}, ""
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith(";"):
                continue
            if line.startswith("["):
                section = unquote(line.strip("[]"))
                continue
            if "=" not in line:
                continue
            k, v = line.split("=", 1)
            if k.startswith("FITFLAG_"):
                fit[k[8:]] = v[:1] == "1"
                continue
            k = {"Alfa": "Alpha", "CalcQmax": None}.get(k, k)
            if k is None:
                continue
            if k in BOOLS:
                p[k] = v.strip().lower() in ("true", "1")
            elif k in COMBOS or k in ("QuadrantFactor", "Dimension"):     # (also the pySCAtter combos)
                p[k] = int(float(v))
            elif k in NUM:
                p[k] = float(v)
            elif k == "Comment":
                p[k] = v
    return p, fit, section


# ----------------------------------------------------------------------------- GUI helpers (pySCAtter)
#: parameters of SasScatter2 .ini files that the pySCAtter calculation does not use (kept in files, not shown)
UNUSED = {"EditCeff", "EditCeffcyl", "reff", "acpl", "bcpl", "ifluc", "rfluc", "RotAlpha",
          "VAx1", "VAx2", "VAx3", "Ay1", "Ay2", "Ay3", "Az1", "Az2", "Az3",
          "rotTheta", "rotPhi", "P1", "iso", "EditBFactor"}

#: readable labels of the combo boxes
COMBO_LABELS = OrderedDict(ComboBoxParticle="particle", ComboBoxInterior="interior", Ordis="orientation",
                           LType="lattice", ComboBoxPeak="peak shape", StructureFactor="structure factor",
                           ShellProfile="shell profile")

#: GUI labels of the numeric parameters (the SasScatter names stay the keys and the .ini names)
LABELS = {
    "EditRadius": "R", "EditRadiusi": "R₂ / shell", "EditSigma": "σ_R (rel.)", "Length": "L", "SigmaL": "σ_L (rel.)",
    "Alpha": "exponent p", "EditRho": "ρ shell", "EditDbeta": "orient. width (°)", "theta": "director θ (°)",
    "phi": "director φ (°)", "ucpsi": "ψ about zone axis (°)", "ucn1": "zone axis u", "ucn2": "zone axis v",
    "ucn3": "zone axis w", "uca": "a", "ucb": "b", "ucc": "c", "ucalpha": "α (°)", "ucbeta": "β (°)",
    "ucgamma": "γ (°)", "EditPeakPar": "peak parameter", "EditDebyeWaller": "Debye-Waller σ_DW",
    "EditDomainSize": "domain size D", "EditAzi": "azimuthal domain D_ψ", "HKLmax": "hkl max", "EditBFactor": "B factor", "I0": "I₀", "Base": "background",
    "EditQmax": "q max", "EditQmin": "q min", "EditQsteps": "q points (1D)", "GridPoints": "grid points (2D)",
    "Editqz": "q offset (vertical)", "EditWavelength": "λ (nm)", "EditDet": "SD (m)", "EditPixelX": "pixel x (m)",
    "EditPixelY": "pixel y (m)", "EditPixelNoX": "pixels x", "EditPixelNoY": "pixels y", "BeamPosX": "beam x (px)",
    "BeamPosY": "beam y (px)", "ShellN": "shells n", "ShellThick": "shell thickness", "ShellRho2": "ρ 2nd shell",
    "ProfileAlpha": "profile exponent α", "SFRadius": "R_HS / r₀", "SFEta": "η (vol. fraction)", "SFTau": "τ (stickiness)",
    "SFDelta": "δ (well width)", "SFDim": "fractal D", "SFXi": "ξ (cut-off)",
    "SigX": "domain size x", "SigY": "domain size y", "SigZ": "domain size z",
}

_ANISO = {1, 2, 4, 5, 6, 7, 8}          # particles with an orientation (cylinder, disk, cube, ellipsoids, ...)


def relevant(p):
    """Numeric parameters that the calculation uses with the combo settings of ``p``."""
    kind, inter = int(p["ComboBoxParticle"]), int(p["ComboBoxInterior"])
    lat = LTYPE_KIND.get(int(p["LType"])) is not None
    ordis = int(p["Ordis"])
    used = {"I0", "Base", "EditQmax", "EditQmin", "EditQsteps", "GridPoints", "Editqz",
            "EditWavelength", "EditDet", "EditPixelX", "EditPixelY", "EditPixelNoX", "EditPixelNoY", "BeamPosX",
            "BeamPosY", "EditRadius"}
    if kind not in (10, 11):
        used.add("EditSigma")
    if kind in (1, 2, 5, 6, 7, 10, 11):
        used.add("Length")
    if kind == 10:
        used.add("SigmaL")
    if kind in (7, 8):
        used.add("Alpha")
    if kind == 3 or kind == 6 or inter in (1, 2, 4):
        used.add("EditRadiusi")
    if kind == 3 or inter in (1, 2, 3, 4):
        used.add("EditRho")
    if inter in (3, 4):
        used |= {"ShellN", "ShellThick"}
    if inter == 3:
        used.add("ShellRho2")
    if inter == 2 and int(p.get("ShellProfile", 0)) == 1:
        used.add("ProfileAlpha")
    if ordis != 7 and kind in _ANISO:
        used |= {"theta", "phi"}
        if ordis not in (6, 13):
            used.add("EditDbeta")
    if lat:
        used |= {"uca", "ucb", "ucc", "ucalpha", "ucbeta", "ucgamma", "EditDomainSize", "EditDebyeWaller",
                 "HKLmax"}
        if int(p["ComboBoxPeak"]) in (4, 5, 6):
            used.add("EditPeakPar")
        if int(p["ComboBoxPeak"]) == 7 and ordis == 7:            # anisotropic Gaussian: domain sizes per axis
            used |= {"SigX", "SigY", "SigZ"}
            used.discard("EditDomainSize")
        elif not (p.get("RadButDebyeScherrer", True) or p.get("RadioButtonPara", False)):
            used.discard("EditDomainSize")                         # fixed width 1, as in CrystalScatter
        if ordis != 7:
            used |= {"ucpsi", "ucn1", "ucn2", "ucn3"}
            if int(p["ComboBoxPeak"]) <= 5 and ordis != 13:          # CrystalScatter form: azimuthal spread
                used.add("EditAzi")
    elif int(p.get("StructureFactor", 0)) > 0:
        from .models_ext import STRUCTURE_FACTORS
        used |= set(STRUCTURE_FACTORS[int(p["StructureFactor"])][2])
    return used
