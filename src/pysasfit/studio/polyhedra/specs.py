# SPDX-License-Identifier: GPL-3.0-or-later
"""pySASstudio ports of the hard-polyhedron form factors of BornAgain / libformfactor.

For every shape there are two functions, registered by :mod:`pysasfit.studio.ports` (if this package is installed)
as ``ff_ba_<shape>`` and ``ff_ba_<shape>_oriented`` (and as ``ff_py_ba_...`` aliases):

* ``ff_ba_<shape>``: randomly oriented particles. ``intensity`` = eta^2 <|F|^2> (real), ``amplitude`` = eta <F>,
  the orientation average of the amplitude, which is real (F(-q) = conj F(q) for a real density);
* ``ff_ba_<shape>_oriented``: one particle orientation (Euler angles ZYZ, ``alpha``, ``beta``, ``gamma``, in
  degrees, body -> lab) and scattering vector q = q (cos psi, sin psi, 0) in the detector plane. ``amplitude`` is
  the *complex* amplitude eta F(q, psi), ``intensity`` its modulus squared (real).

The origin of the amplitude phase is the centre of mass of the particle. Lengths are in the unit reciprocal to q,
angles in degrees (as everywhere in pySASstudio), ``eta`` is the scattering length density difference.

Menu: the ``group`` of every spec is ``"menu:<submenu>/<subsubmenu>/..."`` (read by
:func:`pysasfit.studio.groups.build_menu`): "BornAgain" (the non-Platonic BornAgain shapes) and "Polyhedra" with
"Platonic", "Archimedean", "Catalan", "Johnson/J<a>-J<b>", "Kepler-Poinsot" and the prismatic families. The
oriented variants are in a last submenu "oriented". Every Archimedean / Catalan / Johnson / Kepler-Poinsot solid is a
function of its own (``ff_ba_arch_<name>``, ``ff_ba_cat_<name>``, ``ff_ba_j<nn>_<name>``, ``ff_ba_kp_<name>``) with
the parameters (a, eta[, angles]).
"""

from __future__ import annotations

import functools
import math
import re

import numpy as np

from ._catalog import ARCHIMEDEAN_NAMES, CATALAN_NAMES, CATALOG_FACTORIES, JOHNSON_NAMES, KEPLER_POINSOT_NAMES
from ._orient import orientational_average, oriented_amplitude, rotation_zyz
from ._shapes import FACTORIES as _BA_FACTORIES

FACTORIES = {**_BA_FACTORIES, **CATALOG_FACTORIES}

__all__ = ["SPECS", "SHAPES"]

_DEG = math.pi / 180.0
_SOURCE = "libformfactor (BornAgain), J. Wuttke, J. Appl. Cryst. 54, 580 (2021)"
_SOURCE_CAT = "face/edge method of J. Wuttke, J. Appl. Cryst. 54, 580 (2021) (libformfactor), applied to standard polyhedra"


def _len(name, default, doc):
    return (name, dict(default=default, lower=0.0, lower_open=True, kind="length", doc=doc), (2.0, 20.0, "lin"))


def _frac(name, default, doc, lower=0.0):
    return (name, dict(default=default, lower=lower, kind="dimensionless", doc=doc), (0.2, 1.5, "lin"))


def _ang(name, default, doc, upper=180.0):
    return (name, dict(default=default, lower=0.0, lower_open=True, upper=upper, upper_open=upper == 180.0,
                       kind="angle", unit="deg", doc=doc), (40.0, 85.0, "lin"))  # fmt: skip


# key: (label, parameter definitions, conversion of the parameter values to factory arguments)
SHAPES = {
    "Box": ("Box", [_len("L", 12.0, "length (x)"), _len("W", 6.0, "width (y)"), _len("H", 8.0, "height (z)")], lambda v: v),
    "Prism3": ("Triangular prism", [_len("a", 10.0, "base edge"), _len("H", 8.0, "height")], lambda v: v),
    "Prism6": ("Hexagonal prism", [_len("a", 5.0, "base edge"), _len("H", 8.0, "height")], lambda v: v),
    "Tetrahedron": ("Tetrahedron", [_len("a", 10.0, "edge")], lambda v: v),
    "Octahedron": ("Octahedron", [_len("a", 10.0, "edge")], lambda v: v),
    "Dodecahedron": ("Dodecahedron", [_len("a", 5.0, "edge")], lambda v: v),
    "Icosahedron": ("Icosahedron", [_len("a", 6.0, "edge")], lambda v: v),
    "Pyramid2": ("Pyramid, rectangular base", [_len("L", 12.0, "base length"), _len("W", 8.0, "base width"), _len("H", 6.0, "height"),
                 _ang("alpha_L_deg", 70.0, "dihedral angle of the faces tapering the length"),
                 _ang("alpha_W_deg", 70.0, "dihedral angle of the faces tapering the width")],
                 lambda v: (*v[:3], v[3] * _DEG, v[4] * _DEG)),  # fmt: skip
    "Pyramid3": ("Pyramid, triangular base", [_len("a", 10.0, "base edge"), _len("H", 6.0, "height"), _ang("alpha_deg", 70.0, "dihedral angle at the base")],
                 lambda v: (v[0], v[1], v[2] * _DEG)),  # fmt: skip
    "Pyramid4": ("Pyramid, square base", [_len("a", 10.0, "base edge"), _len("H", 6.0, "height"), _ang("alpha_deg", 70.0, "dihedral angle at the base")],
                 lambda v: (v[0], v[1], v[2] * _DEG)),  # fmt: skip
    "Pyramid6": ("Pyramid, hexagonal base", [_len("a", 5.0, "base edge"), _len("H", 6.0, "height"), _ang("alpha_deg", 70.0, "dihedral angle at the base")],
                 lambda v: (v[0], v[1], v[2] * _DEG)),  # fmt: skip
    "Bipyramid4": ("Bipyramid, square", [_len("L", 12.0, "length of the square middle section"), _len("H", 4.0, "height of the lower half"),
                   _frac("ratio", 0.5, "height of the upper half / H"), _ang("alpha_deg", 70.0, "dihedral angle", upper=90.0)],
                   lambda v: (v[0], v[1], v[2], v[3] * _DEG)),  # fmt: skip
    "CantellatedCube": ("Cantellated cube", [_len("L", 10.0, "edge of the cube"), ("d", dict(default=2.0, lower=0.0, kind="length", doc="removed length"), (0.2, 4.0, "lin"))],
                        lambda v: v),  # fmt: skip
    "TruncatedCube": ("Truncated cube", [_len("L", 10.0, "edge of the cube"), ("d", dict(default=2.0, lower=0.0, kind="length", doc="removed length"), (0.2, 4.0, "lin"))],
                      lambda v: v),  # fmt: skip
    # ---- prismatic families (see _catalog.py); the first parameter n selects the solid
    "PrismN": ("Regular n-gonal prism", [("n", dict(default=5, lower=3, upper=1000, kind="dimensionless", doc="number of base edges"), (3, 12, "lin")),
               _len("a", 5.0, "base edge"), _len("H", 8.0, "height")], lambda v: v),  # fmt: skip
    "AntiprismN": ("Regular n-gonal antiprism", [("n", dict(default=5, lower=3, upper=1000, kind="dimensionless", doc="number of base edges"), (3, 12, "lin")),
                   _len("a", 5.0, "base edge"), _len("H", 4.0, "height")], lambda v: v),  # fmt: skip
    "BipyramidN": ("Regular n-gonal bipyramid", [("n", dict(default=5, lower=3, upper=1000, kind="dimensionless", doc="number of base edges"), (3, 12, "lin")),
                   _len("a", 5.0, "base edge"), _len("H", 4.0, "height of each apex above the base plane")], lambda v: v),  # fmt: skip
    "TrapezohedronN": ("n-gonal trapezohedron", [("n", dict(default=5, lower=3, upper=1000, kind="dimensionless", doc="number of vertices of the zig-zag ring / 2"), (3, 12, "lin")),
                       _len("R", 4.0, "radius of the zig-zag ring"), _len("H", 6.0, "distance of the apexes from the centre")], lambda v: v),  # fmt: skip
}

# ---------------------------------------------------------------------------------------- menu and per-solid entries
_MENU = {}  # key -> menu path
_TAG = {}  # key -> text in brackets after the label (default: BornAgain)
_BASENAME = {}  # key -> function name without the prefix ff_ba_
_CATALOG = set()  # keys that are not BornAgain shapes (source: face/edge method)

_PLATONIC = ("Tetrahedron", "Cube", "Octahedron", "Dodecahedron", "Icosahedron")
FACTORIES["Cube"] = lambda a: _BA_FACTORIES["Box"](a, a, a)
SHAPES["Cube"] = ("Cube", [_len("a", 10.0, "edge")], lambda v: v)
SHAPES = {k: SHAPES[k] for k in (*_PLATONIC, *(k for k in SHAPES if k not in _PLATONIC))}  # Platonic solids first
for _k in SHAPES:
    _MENU[_k] = "Polyhedra/Platonic" if _k in _PLATONIC else "BornAgain"
for _k in ("PrismN", "AntiprismN", "BipyramidN", "TrapezohedronN"):
    _MENU[_k] = "Polyhedra/Prisms, antiprisms, bipyramids, trapezohedra"
    _TAG[_k] = "libformfactor method"
    _CATALOG.add(_k)

# Johnson solids are grouped in the submenu by their number J1..J92 (Johnson 1966 / Wikipedia numbering)
_JOHNSON_RANGES = [(1, 6), (7, 17), (18, 25), (26, 48), (49, 57), (58, 61), (62, 64), (65, 71), (72, 83), (84, 92)]


def _slug(text):
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")


def _add_solid(key, family, factory, name, base, menu, edge_doc="edge"):
    FACTORIES[key] = factory
    SHAPES[key] = (name, [_len("a", 5.0, edge_doc)], lambda v: v)
    _MENU[key], _TAG[key], _BASENAME[key] = menu, family, base
    _CATALOG.add(key)


for _n, _name in enumerate(ARCHIMEDEAN_NAMES, 1):
    _add_solid(f"Arch{_n}", "Archimedean", functools.partial(CATALOG_FACTORIES["Archimedean"], _n), _name,
               "arch_" + _slug(_name), "Polyhedra/Archimedean")  # fmt: skip
for _n, _name in enumerate(CATALAN_NAMES, 1):
    _add_solid(f"Cat{_n}", "Catalan", functools.partial(CATALOG_FACTORIES["Catalan"], _n), _name,
               "cat_" + _slug(_name), "Polyhedra/Catalan", "shortest edge")  # fmt: skip
for _n, _name in enumerate(JOHNSON_NAMES, 1):
    _lo, _hi = next(r for r in _JOHNSON_RANGES if r[0] <= _n <= r[1])
    _add_solid(f"J{_n}", "Johnson", functools.partial(CATALOG_FACTORIES["Johnson"], _n), f"J{_n} {_name}",
               f"j{_n:02d}_" + _slug(_name), f"Polyhedra/Johnson/J{_lo}–J{_hi}")  # fmt: skip
for _n, _name in enumerate(KEPLER_POINSOT_NAMES, 1):
    _add_solid(f"KP{_n}", "Kepler-Poinsot", functools.partial(CATALOG_FACTORIES["KeplerPoinsot"], _n), _name,
               "kp_" + _slug(_name), "Polyhedra/Kepler-Poinsot", "edge of the star faces")  # fmt: skip


_FACTORY_ALIAS = {"PrismN": "Prism_n", "AntiprismN": "Antiprism_n", "BipyramidN": "Bipyramid_n",
                  "TrapezohedronN": "Trapezohedron_n"}  # fmt: skip


def _snake(key):
    out = ""
    for i, c in enumerate(key):
        if c.isupper() and i and not key[i - 1].isdigit() and not key[i - 1].isupper():
            out += "_"
        out += c.lower()
    return out


def _build(key, n, convert):
    factory = FACTORIES[_FACTORY_ALIAS.get(key, key)]

    def build(p):
        v = [float(x) for x in p[:n]]
        return factory(*convert(v))

    return build


def _flatten(q, psi):
    q = np.asarray(q, float)
    psi = np.asarray(psi, float)
    return np.broadcast_arrays(q, psi)


def _make(key):
    label, pdefs, convert = SHAPES[key]
    catalog = key in _CATALOG
    source = _SOURCE_CAT if catalog else _SOURCE
    tag = _TAG.get(key, "BornAgain")
    menu = "menu:" + _MENU[key]
    n = len(pdefs)
    names = [d[0] for d in pdefs]
    build = _build(key, n, convert)
    i_eta = n
    base = "ff_ba_" + _BASENAME.get(key, _snake(key))
    sample = {i: d[2] for i, d in enumerate(pdefs)}
    sample[i_eta] = (0.5, 2.0, "lin")

    def eta_of(p):
        return float(p[i_eta])

    # ------------------------------------------------------------------------------------ isotropic
    def _average(q, p):
        q = np.asarray(q, float)
        shape = build(p)
        I, A = orientational_average(shape, np.abs(q).ravel())
        return I.reshape(q.shape), A.reshape(q.shape)

    def intensity(q, p, psi=None):
        """eta^2 <|F|^2> over random orientations."""
        I, _ = _average(q, p)
        return eta_of(p) ** 2 * I

    def amplitude(q, p, psi=None):
        """eta <F>: orientation average of the amplitude (real)."""
        _, A = _average(q, p)
        return eta_of(p) * A

    def volume(x, p, dist=0):
        x = np.asarray(x, float)
        out = np.empty(x.shape)
        pp = np.array(p, float)
        for idx, xi in np.ndenumerate(x):
            pp[dist] = xi
            out[idx] = build(pp).volume
        return out

    volume.vectorized = True

    iso = dict(
        name=base,
        intensity=intensity,
        parameters=names + ["eta"],
        psi_index=None,
        amplitude=amplitude,
        volume=volume,
        sample=dict(sample),
        source=source,
        notes="Randomly oriented particles; <F> is real. Origin of the phase: centre of mass.",
        label=f"{label} ({tag})",
        group=menu,
        c_reference=False,
    )

    # --------------------------------------------------------------------------------------- oriented
    i_a, i_b, i_g, i_psi = n + 1, n + 2, n + 3, n + 4

    def _amp_oriented(q, p, psi):
        psi_rad = p[i_psi] * _DEG if psi is None else psi
        qq, pp = _flatten(q, psi_rad)
        shape = build(p)
        qv = np.stack([qq * np.cos(pp), qq * np.sin(pp), np.zeros_like(qq)], axis=-1)
        rot = rotation_zyz(p[i_a] * _DEG, p[i_b] * _DEG, p[i_g] * _DEG)
        return eta_of(p) * oriented_amplitude(shape, qv, rot)

    def amplitude_oriented(q, p, psi=None):
        """Complex amplitude eta F(q, psi) of one oriented particle (psi in rad; None: from the parameter)."""
        return _amp_oriented(q, np.asarray(p, float), psi)

    def intensity_oriented(q, p, psi=None):
        """|eta F(q, psi)|^2 of one oriented particle."""
        return np.abs(_amp_oriented(q, np.asarray(p, float), psi)) ** 2

    ori = dict(
        name=base + "_oriented",
        intensity=intensity_oriented,
        parameters=names + ["eta", "alpha_deg", "beta_deg", "gamma_deg", "psi"],
        psi_index=i_psi,
        psi_unit="deg",
        amplitude=amplitude_oriented,
        volume=volume,
        sample={**sample, i_a: (0.0, 90.0, "lin"), i_b: (0.0, 90.0, "lin"), i_g: (0.0, 90.0, "lin"), i_psi: (0.0, 90.0, "lin")},
        source=source,
        notes="One particle orientation (Euler angles ZYZ, body -> lab); q in the detector plane at angle psi. "
        "The amplitude is complex.",
        label=f"{label} ({tag}, oriented)",
        group=menu + "/oriented",
        c_reference=False,
    )
    pdict = {names[i]: d[1] for i, d in enumerate(pdefs)}
    pdict["eta"] = dict(default=1.0)
    over_iso = dict(pdict)
    over_ori = dict(pdict)
    over_ori.update(
        alpha_deg=dict(default=0.0, kind="angle", unit="deg", doc="Euler angle alpha (rotation about z)"),
        beta_deg=dict(default=0.0, kind="angle", unit="deg", doc="Euler angle beta (rotation about y)"),
        gamma_deg=dict(default=0.0, kind="angle", unit="deg", doc="Euler angle gamma (rotation about z)"),
        psi=dict(default=0.0, kind="angle", unit="deg", doc="direction of q in the detector plane"),
    )
    return (iso, over_iso), (ori, over_ori)


def _build_specs():
    specs, overrides = [], {}
    for key in SHAPES:
        for d, over in _make(key):
            specs.append(d)
            overrides[d["name"]] = over
    return specs, overrides


SPECS, _OVERRIDES = _build_specs()

# defaults and bounds of the parameters for the parameter tables of pySASstudio (if the host package is there)
try:
    from ..ports import param_defaults as _pd
except ImportError:  # stand-alone use of the polyhedra package
    _pd = None
if _pd is not None:
    for _name, _over in _OVERRIDES.items():
        _pd.OVERRIDES.setdefault(_name, {}).update(_over)
