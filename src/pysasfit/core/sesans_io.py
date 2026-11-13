"""Reading SESANS data (``.ses`` files) for the pySASfit tools.

Two families of files are understood:

* the SasView format (``FileFormatVersion 1.0``): ``key value`` header lines (``Thickness``, ``Thickness_unit``,
  ``SpinEchoLength_unit``, ``Wavelength_unit``, ``Theta_zmax`` ...), then ``BEGIN_DATA``, a line of column names
  (``SpinEchoLength Depolarisation Depolarisation_error SpinEchoLength_error Wavelength Wavelength_error
  Polarisation Polarisation_error``) and the numbers;
* the older Delft files (as in sasfit-collab/data): ``key: value`` or ``key<TAB>value`` header lines
  (``Thickness [cm]: 0.2``, ``Thickness [mm]  2``, ``z-acceptance [nm^-1]  0.3``), one line of column titles with
  units in brackets (``spin echo length [A]``, ``Depolarisation [A-2 cm-1]``, ``wavelength [nm]``,
  ``polarisation``, ``error ...``) and the numbers.

The columns are recognised by their titles. A column without a known title whose values lie between 0 and 1.5 is
taken as the polarisation P/P0 (e.g. ``Sample 5``). Missing quantities are derived:
G(δ) - G(0) = ln(P/P0) / (λ² t) and P/P0 = exp(λ² t [G(δ) - G(0)]) (λ in the length unit, t in cm).

All lengths are converted to ``length_unit`` (``"nm"`` or ``"A"``, also ``"um"``, ``"m"``); G(δ) - G(0) is then in
[length]⁻² cm⁻¹, the unit of the SESANS transform of an intensity in cm⁻¹ with Q in [length]⁻¹. Units printed in a
file can be overridden (``delta_unit``, ``wavelength_unit``, ``thickness_cm``), as some files label them wrongly.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field

import numpy as np

__all__ = ["SESANSData", "load_sesans", "is_sesans_file", "LENGTH_UNITS", "length_factor"]

LENGTH_UNITS = {"m": 1.0, "cm": 1e-2, "mm": 1e-3, "um": 1e-6, "µm": 1e-6, "μm": 1e-6, "micron": 1e-6, "nm": 1e-9,
                "a": 1e-10, "å": 1e-10, "ang": 1e-10, "angstrom": 1e-10}  # fmt: skip


def _unit_key(u):
    u = (u or "").strip().lower().replace("angstroms", "angstrom").replace("Å", "å")
    return {"aa": "a", "ångström": "angstrom", "Å".lower(): "å"}.get(u, u)


def length_factor(src, dst):
    """Factor converting a length from unit ``src`` to unit ``dst`` (names of :data:`LENGTH_UNITS`)."""
    a, b = LENGTH_UNITS.get(_unit_key(src)), LENGTH_UNITS.get(_unit_key(dst))
    if a is None or b is None:
        raise ValueError(f"unknown length unit '{src if a is None else dst}'")
    return a / b


@dataclass
class SESANSData:
    """SESANS data in a common length unit (see the module docstring)."""

    delta: np.ndarray            # spin-echo length
    G: np.ndarray                # G(δ) - G(0) = ln(P/P0) / (λ² t), [length]^-2 cm^-1
    dG: np.ndarray
    P: np.ndarray                # P/P0
    dP: np.ndarray
    wavelength: np.ndarray       # λ per point
    thickness: float             # cm
    length_unit: str = "nm"
    delta_error: np.ndarray | None = None
    q_accept: float | None = None  # acceptance in Q ([length]^-1), from a z-acceptance header line
    theta_max: float | None = None  # acceptance angle (rad), from Theta_zmax: Q_max = 2π sin(θ_max) / λ per point
    header: dict = field(default_factory=dict)
    columns: dict = field(default_factory=dict)  # role -> column title of the file
    file_units: dict = field(default_factory=dict)  # what the file said: delta, wavelength, G

    def observable(self, which="G"):
        """``(y, sigma)`` of ``"G"`` (G(δ) - G(0)) or ``"P"`` (P/P0)."""
        return (self.G, self.dG) if which == "G" else (self.P, self.dP)


def is_sesans_file(path):
    """True for a ``.ses`` file or a text file whose header mentions the spin-echo length."""
    if str(path).lower().endswith(".ses"):
        return True
    try:
        with open(path, "r", errors="replace") as fh:
            head = fh.read(4000).lower()
    except OSError:
        return False
    return "spinecholength" in head or "spin echo length" in head or "spin-echo length" in head


_NUM = re.compile(r"^[\s,;]*[-+]?(\d+\.?\d*|\.\d+)([eE][-+]?\d+)?")


def _numbers(line):
    parts = re.split(r"[\s,;]+", line.strip())
    try:
        return [float(p) for p in parts if p]
    except ValueError:
        return None


def _bracket_unit(title):
    m = re.search(r"\[([^\]]*)\]", title)
    return m.group(1).strip() if m else None


def _role(title):
    t = title.lower().replace("-", " ").replace("_", " ")
    err = "error" in t or t.startswith("err") or " err" in t or t.startswith("d ") or "sigma" in t or "uncert" in t
    if "depol" in t:
        base = "G"
    elif "spin" in t and "echo" in t or "spinecholength" in t.replace(" ", "") or re.search(r"\bsel\b", t) or \
            re.match(r"^(delta|z|δ)\b", t):
        base = "delta"
    elif "wavelength" in t or "lambda" in t or re.search(r"\bλ\b", t):
        base = "lambda"
    elif "polari" in t or "p/p0" in t or re.search(r"\bpol\b", t):
        base = "P"
    else:
        base = None
    if base is None:
        return "error?" if err else None
    return base + "_err" if err else base


def _split_title(line, ncol):
    for sep in ("\t", ";", ","):
        parts = [p.strip() for p in line.split(sep)]
        parts = [p for p in parts if p]
        if len(parts) >= 2:
            return parts
    return line.split()


def _header_value(text):
    m = _NUM.match(text)
    return float(m.group(0).strip(" ,;")) if m else None


def load_sesans(path, length_unit="nm", delta_unit=None, wavelength_unit=None, thickness_cm=None):
    """Read a SESANS file; see the module docstring. Returns a :class:`SESANSData` sorted by δ."""
    with open(path, "r", errors="replace") as fh:
        lines = [ln.rstrip("\r\n") for ln in fh]
    header, titles, rows = {}, None, []
    sasview = any(ln.strip().upper() == "BEGIN_DATA" for ln in lines)
    in_data = False
    for ln in lines:
        s = ln.strip()
        if not s or s.startswith(("#", "%", "//")):
            continue
        if sasview and s.upper() == "BEGIN_DATA":
            in_data = True
            continue
        nums = _numbers(s)
        if nums is not None and len(nums) >= 2 and (in_data or not sasview):
            rows.append(nums)
            continue
        if rows:  # text after the numbers (e.g. END_DATA)
            continue
        if (in_data or not sasview) and titles is None and _looks_like_titles(s):
            titles = _split_title(ln, 0) if not sasview else s.split()
            if sasview or any(_role(t) for t in titles):
                continue
            titles = None
        key, val = _key_value(s)
        if key:
            header[key] = val
    if not rows:
        raise ValueError(f"{path}: no numeric data rows")
    ncol = min(len(r) for r in rows)
    A = np.array([r[:ncol] for r in rows], float)
    titles = list(titles or [])[:ncol]
    titles += [f"column {i + 1}" for i in range(len(titles), ncol)]
    roles = {}
    for i, t in enumerate(titles):
        r = _role(t)
        if r and r != "error?" and r not in roles:
            roles[r] = i
    if "delta" not in roles:
        roles["delta"] = 0
    if "P" not in roles and "G" not in roles:  # an untitled column with values like a polarisation
        for i, t in enumerate(titles):
            if i in roles.values() or _role(t) is not None:
                continue
            v = A[:, i]
            if np.all(np.isfinite(v)) and v.min() > -0.1 and v.max() < 1.5 and np.ptp(v) > 0:
                roles["P"] = i
                if i + 1 < ncol and i + 1 not in roles.values() and _role(titles[i + 1]) in ("error?", None):
                    roles["P_err"] = i + 1
                break
    # units
    hk = {k.lower(): (k, v) for k, v in header.items()}

    def head(*names):
        for n in names:
            for k, (orig, v) in hk.items():
                if k.replace(" ", "").replace("_", "").startswith(n):
                    return orig, v
        return None, None

    d_unit = delta_unit or _bracket_unit(titles[roles["delta"]]) or (head("spinecholengthunit")[1] or "").strip() or "A"
    lam_title = titles[roles["lambda"]] if "lambda" in roles else ""
    l_unit = wavelength_unit or _bracket_unit(lam_title) or (head("wavelengthunit")[1] or "").strip() or d_unit
    g_unit = (_bracket_unit(titles[roles["G"]]) if "G" in roles else None) or (head("depolarisationunit")[1] or "")
    fd = length_factor(d_unit, length_unit)
    fl = length_factor(l_unit, length_unit)
    delta = A[:, roles["delta"]] * fd
    d_err = A[:, roles["delta_err"]] * fd if "delta_err" in roles else None
    # thickness (cm)
    if thickness_cm is None:
        key, val = head("thickness")
        t_unit = "cm"
        if key is not None:
            t_unit = _bracket_unit(key) or (head("thicknessunit")[1] or "cm").strip() or "cm"
            thickness_cm = _header_value(val)
            if thickness_cm is not None:
                thickness_cm *= length_factor(t_unit, "cm")
    thickness = float(thickness_cm) if thickness_cm else math.nan
    # wavelength
    if "lambda" in roles:
        lam = A[:, roles["lambda"]] * fl
    else:
        _k, v = head("wavelength")
        lam = np.full(delta.shape, (_header_value(v or "") or math.nan) * fl)
    # G and P
    lt = lam**2 * thickness
    if "G" in roles:
        gf = _g_factor(g_unit, d_unit, length_unit)
        G = A[:, roles["G"]] * gf
        dG = A[:, roles["G_err"]] * gf if "G_err" in roles else np.full(G.shape, np.nan)
    else:
        G = dG = None
    if "P" in roles:
        P = A[:, roles["P"]]
        dP = A[:, roles["P_err"]] if "P_err" in roles else np.full(P.shape, np.nan)
    else:
        P = dP = None
    if G is None and P is None:
        raise ValueError(f"{path}: neither a depolarisation nor a polarisation column found (titles: {titles})")
    with np.errstate(invalid="ignore", divide="ignore"):
        if G is None:
            G = np.log(P) / lt
            dG = dP / (np.abs(P) * lt)
        if P is None:
            P = np.exp(G * lt)
            dP = P * lt * dG
    # acceptance
    q_acc = None
    key, val = head("z-acceptance", "zacceptance")
    if key is not None and _header_value(val):
        u = _bracket_unit(key) or f"{length_unit}^-1"
        m = re.match(r"\s*([a-zA-Zµμå]+)\s*\^?\s*-1", u)
        q_acc = _header_value(val) / length_factor(m.group(1), length_unit) if m else _header_value(val)
    theta = None
    key, val = head("thetazmax")
    th = _header_value(val or "") if key is not None and not key.lower().replace(" ", "").endswith("unit") else None
    if th:
        unit = (_bracket_unit(key) or (head("thetazmaxunit")[1] or "rad")).strip().lower()
        theta = th * {"mrad": 1e-3, "deg": math.pi / 180, "degree": math.pi / 180, "degrees": math.pi / 180}.get(unit, 1.0)
    ok = np.isfinite(delta) & (delta > 0)
    o = np.argsort(delta[ok])

    def pick(v):
        return None if v is None else np.asarray(v)[ok][o]

    return SESANSData(
        delta=pick(delta), G=pick(G), dG=pick(dG), P=pick(P), dP=pick(dP), wavelength=pick(lam), thickness=thickness,
        length_unit=length_unit, delta_error=pick(d_err), q_accept=q_acc, theta_max=theta, header=header,
        columns={r: titles[i] for r, i in roles.items()},
        file_units={"delta": d_unit, "wavelength": l_unit, "G": g_unit or None},
    )  # fmt: skip


def _looks_like_titles(s):
    words = re.findall(r"[A-Za-z]{3,}", s)
    return len(words) >= 2


def _key_value(s):
    if ":" in s:
        k, v = s.split(":", 1)
        if k.strip():
            return k.strip(), v.strip()
    parts = re.split(r"\t+|\s{2,}", s, maxsplit=1)
    if len(parts) == 2:
        return parts[0].strip(), parts[1].strip()
    parts = s.split(None, 1)
    if len(parts) == 2:
        return parts[0].strip(), parts[1].strip()
    return None, None


def _g_factor(g_unit, d_unit, length_unit):
    """Factor converting the depolarisation column to [length_unit]^-2 cm^-1."""
    u = (g_unit or "").replace(" ", "").lower()
    m = re.match(r"([a-zµμå]+)\^?-2", u)  # "a-2cm-1", "nm^-2cm^-1", "A-2 cm-1"
    src = m.group(1) if m else d_unit
    try:
        return length_factor(length_unit, src) ** 2  # x per src^2  ->  x per dst^2: times (dst/src)^2
    except ValueError:
        return length_factor(length_unit, d_unit) ** 2
