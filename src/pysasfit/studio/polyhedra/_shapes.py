# SPDX-License-Identifier: GPL-3.0-or-later
#
# Python port of the factory functions of libformfactor (ff/Make.cpp)
#   Copyright (C) Forschungszentrum Juelich GmbH 2022, Scientific Computing Group at MLZ
#   Author of the original: Joachim Wuttke
# Derivative work of libformfactor, licensed under the GNU General Public License, version 3 or later.
"""Factories of the hard polyhedra of BornAgain / libformfactor.

All bodies are centred at their centre of mass. Lengths are in arbitrary but common units, angles in radians
(as in libformfactor; the Pyramid and Bipyramid taper angles are dihedral angles between base and side faces).
"""

from __future__ import annotations

import math

from ._core import Box as _Box
from ._core import Polyhedron, Prism, Topology

__all__ = ["FACTORIES", "box", "prism3", "prism6", "tetrahedron", "octahedron", "dodecahedron", "icosahedron",
           "pyramid2", "pyramid3", "pyramid4", "pyramid6", "bipyramid4", "cantellated_cube", "truncated_cube"]  # fmt: skip

_PI = math.pi
_PI_2 = _PI / 2
_REL_TOL = 1e-14  # matches the relative short-edge cutoff of Face


def _cotangent(angle):
    if angle == _PI_2:
        return 0.0
    return 1 / math.tan(angle)


def _dimension_or_zero(dimension, geometry_scale):
    """Snap a negligible boundary dimension to exactly zero (deterministic topology dispatch)."""
    tol = _REL_TOL * geometry_scale
    if not math.isfinite(dimension) or not math.isfinite(tol):
        return dimension
    return 0.0 if abs(dimension) <= tol else dimension


def _need(cond, msg):
    if not cond:
        raise ValueError(msg)


# --------------------------------------------------------------------------------------------------- prisms
def box(length, width, height):
    _need(length > 0, "Box: invalid parameter: length<=0")
    _need(width > 0, "Box: invalid parameter: width<=0")
    _need(height > 0, "Box: invalid parameter: height<=0")
    return _Box(length, width, height)


def prism3(base_edge, height):
    _need(base_edge > 0, "Prism3: invalid parameter: base_edge<=0")
    _need(height > 0, "Prism3: invalid parameter: height<=0")
    a = base_edge
    as_, ac, ah = a / 2, a / math.sqrt(3) / 2, a / math.sqrt(3)
    return Prism(False, height, [(-ac, as_, 0.0), (-ac, -as_, 0.0), (ah, 0.0, 0.0)])


def prism6(base_edge, height):
    _need(base_edge > 0, "Prism6: invalid parameter: base_edge<=0")
    _need(height > 0, "Prism6: invalid parameter: height<=0")
    a = base_edge
    as_, ac = a * math.sqrt(3) / 2, a / 2
    return Prism(True, height, [(a, 0.0, 0.0), (ac, as_, 0.0), (-ac, as_, 0.0), (-a, 0.0, 0.0), (-ac, -as_, 0.0), (ac, -as_, 0.0)])  # fmt: skip


# --------------------------------------------------------------------------------------------- platonic solids
_TET = Topology([((2, 1, 0), False), ((0, 1, 3), False), ((1, 2, 3), False), ((2, 0, 3), False)], False)


def tetrahedron(edge):
    _need(edge > 0, "Tetrahedron: invalid parameter: edge<=0")
    a = edge
    as_, ac, ah = a / 2, a / math.sqrt(3) / 2, a / math.sqrt(3)
    height = math.sqrt(2.0 / 3) * edge
    zcom = height / 4
    return Polyhedron(_TET, [(-ac, as_, -zcom), (-ac, -as_, -zcom), (ah, 0.0, -zcom), (0.0, 0.0, height - zcom)])


_OCT = Topology(
    [((0, 2, 1), False), ((0, 3, 2), False), ((0, 4, 3), False), ((0, 1, 4), False),
     ((2, 3, 5), False), ((1, 2, 5), False), ((4, 1, 5), False), ((3, 4, 5), False)],
    True)  # fmt: skip


def octahedron(edge):
    _need(edge > 0, "Octahedron: invalid parameter: edge<=0")
    h = edge / math.sqrt(2)
    a = edge / 2
    return Polyhedron(_OCT, [(0, 0, -h), (a, -a, 0), (a, a, 0), (-a, a, 0), (-a, -a, 0), (0, 0, h)])


_DOD = Topology(
    [((0, 4, 3, 2, 1), False),
     ((0, 5, 12, 9, 4), False), ((4, 9, 11, 8, 3), False), ((3, 8, 10, 7, 2), False),
     ((2, 7, 14, 6, 1), False), ((1, 6, 13, 5, 0), False),
     ((8, 11, 16, 15, 10), False), ((9, 12, 17, 16, 11), False), ((5, 13, 18, 17, 12), False),
     ((6, 14, 19, 18, 13), False), ((7, 10, 15, 19, 14), False),
     ((15, 16, 17, 18, 19), False)],
    True)  # fmt: skip


def dodecahedron(edge):
    _need(edge > 0, "Dodecahedron: invalid parameter: edge<=0")
    e = edge
    r1, r2, r3 = 0.2628655560595668 * e, 0.42532540417602 * e, 0.5 * e
    r4, r5, r6 = 0.6881909602355868 * e, 0.8090169943749473 * e, 0.8506508083520399 * e
    r7, r8, r9 = 1.113516364411607 * e, 1.309016994374947 * e, 1.376381920471174 * e
    v = [(r6, 0, -r7), (r1, r5, -r7), (-r4, r3, -r7), (-r4, -r3, -r7), (r1, -r5, -r7),
         (r9, 0, -r1), (r2, r8, -r1), (-r7, r5, -r1), (-r7, -r5, -r1), (r2, -r8, -r1),
         (-r9, 0, r1), (-r2, -r8, r1), (r7, -r5, r1), (r7, r5, r1), (-r2, r8, r1),
         (-r6, 0, r7), (-r1, -r5, r7), (r4, -r3, r7), (r4, r3, r7), (-r1, r5, r7)]  # fmt: skip
    return Polyhedron(_DOD, v)


_ICO = Topology(
    [((0, 2, 1), False),
     ((0, 5, 2), False), ((2, 3, 1), False), ((1, 4, 0), False),
     ((0, 6, 5), False), ((2, 5, 8), False), ((2, 8, 3), False), ((1, 3, 7), False), ((1, 7, 4), False), ((0, 4, 6), False),
     ((3, 8, 9), False), ((5, 11, 8), False), ((5, 6, 11), False), ((4, 10, 6), False), ((4, 7, 10), False), ((3, 9, 7), False),
     ((8, 11, 9), False), ((6, 10, 11), False), ((7, 9, 10), False),
     ((9, 11, 10), False)],
    True)  # fmt: skip


def icosahedron(edge):
    _need(edge > 0, "Icosahedron: invalid parameter: edge<=0")
    e = edge
    s1, s2, s3, s4 = 0.1784110448865449 * e, 0.288675134594813 * e, 0.467086179481358 * e, 0.5 * e
    s5, s6, s7, s8 = 0.5773502691896258 * e, 0.7557613140761708 * e, 0.8090169943749473 * e, 0.9341723589627158 * e
    v = [(s5, 0, -s6), (-s2, s4, -s6), (-s2, -s4, -s6), (-s8, 0, -s1), (s3, s7, -s1), (s3, -s7, -s1),
         (s8, 0, s1), (-s3, s7, s1), (-s3, -s7, s1), (-s5, 0, s6), (s2, s4, s6), (s2, -s4, s6)]  # fmt: skip
    return Polyhedron(_ICO, v)


# ------------------------------------------------------------------------------------------------- pyramids
_P2_FRUSTUM = Topology([((3, 2, 1, 0), True), ((0, 1, 5, 4), False), ((1, 2, 6, 5), False), ((2, 3, 7, 6), False),
                        ((3, 0, 4, 7), False), ((4, 5, 6, 7), True)], False)  # fmt: skip
_P2_LENGTH_RIDGE = Topology([((3, 2, 1, 0), True), ((0, 1, 4), False), ((1, 2, 5, 4), False), ((2, 3, 5), False),
                             ((3, 0, 4, 5), False)], False)  # fmt: skip
_P2_WIDTH_RIDGE = Topology([((3, 2, 1, 0), True), ((0, 1, 5, 4), False), ((1, 2, 5), False), ((2, 3, 4, 5), False),
                            ((3, 0, 4), False)], False)  # fmt: skip
_APEX4 = Topology([((3, 2, 1, 0), True), ((0, 1, 4), False), ((1, 2, 4), False), ((2, 3, 4), False), ((3, 0, 4), False)], False)  # fmt: skip


def pyramid2(length, width, height, alpha_length, alpha_width):
    _need(length > 0, "Pyramid2: invalid parameter: length<=0")
    _need(width > 0, "Pyramid2: invalid parameter: width<=0")
    _need(height > 0, "Pyramid2: invalid parameter: height<=0")
    _need(alpha_length > 0, "Pyramid2: invalid parameter: alpha_length<=0")
    _need(alpha_length < _PI, "Pyramid2: invalid parameter: alpha_length>=pi")
    _need(alpha_width > 0, "Pyramid2: invalid parameter: alpha_width<=0")
    _need(alpha_width < _PI, "Pyramid2: invalid parameter: alpha_width>=pi")
    half_length, half_width = length / 2, width / 2
    length_reduction = height * _cotangent(alpha_length)
    width_reduction = height * _cotangent(alpha_width)
    raw_top_half_length = half_length - length_reduction
    raw_top_half_width = half_width - width_reduction
    length_scale = half_length + abs(length_reduction) + height + abs(raw_top_half_width)
    width_scale = half_width + abs(width_reduction) + height + abs(raw_top_half_length)
    top_hl = _dimension_or_zero(raw_top_half_length, length_scale)
    top_hw = _dimension_or_zero(raw_top_half_width, width_scale)
    _need(top_hl >= 0, "Pyramid2: invalid outcome: top_length<0")
    _need(top_hw >= 0, "Pyramid2: invalid outcome: top_width<0")
    eff_l = half_length - top_hl
    eff_w = half_width - top_hw
    base_q = half_length * half_width
    mixed = half_length * eff_w + half_width * eff_l
    prod = eff_l * eff_w
    zcom = height / 2 * (6 * base_q - 4 * mixed + 3 * prod) / (6 * base_q - 3 * mixed + 2 * prod)
    v = [(-half_length, -half_width, -zcom), (half_length, -half_width, -zcom),
         (half_length, half_width, -zcom), (-half_length, half_width, -zcom)]  # fmt: skip
    zt = height - zcom
    if top_hl == 0 and top_hw == 0:
        return Polyhedron(_APEX4, v + [(0, 0, zt)])
    if top_hl == 0:
        return Polyhedron(_P2_LENGTH_RIDGE, v + [(0, -top_hw, zt), (0, top_hw, zt)])
    if top_hw == 0:
        return Polyhedron(_P2_WIDTH_RIDGE, v + [(-top_hl, 0, zt), (top_hl, 0, zt)])
    return Polyhedron(_P2_FRUSTUM, v + [(-top_hl, -top_hw, zt), (top_hl, -top_hw, zt), (top_hl, top_hw, zt), (-top_hl, top_hw, zt)])  # fmt: skip


_P3_FRUSTUM = Topology([((2, 1, 0), False), ((0, 1, 4, 3), False), ((1, 2, 5, 4), False), ((2, 0, 3, 5), False),
                        ((3, 4, 5), False)], False)  # fmt: skip
_APEX3 = Topology([((2, 1, 0), False), ((0, 1, 3), False), ((1, 2, 3), False), ((2, 0, 3), False)], False)


def pyramid3(base_edge, height, alpha):
    _need(base_edge > 0, "Pyramid3: invalid parameter: base_edge<=0")
    _need(height > 0, "Pyramid3: invalid parameter: height<=0")
    _need(alpha > 0, "Pyramid3: invalid parameter: alpha<=0")
    _need(alpha < _PI, "Pyramid3: invalid parameter: alpha>=pi")
    a = base_edge
    reduction = height * _cotangent(alpha) * 2 * math.sqrt(3.0)
    scale = base_edge + abs(reduction) + height
    b = _dimension_or_zero(base_edge - reduction, scale)
    _need(b >= 0, "Pyramid3: invalid outcome: top_edge < 0")
    as_, ac, ah = a / 2, a * (1.0 / math.sqrt(3) / 2), a * (1.0 / math.sqrt(3))
    bs, bc, bh = b / 2, b * (1.0 / math.sqrt(3) / 2), b * (1.0 / math.sqrt(3))
    zcom = height * (1.0 / 4) * (a * a + 2 * a * b + 3 * b * b) / (a * a + a * b + b * b)
    v = [(-ac, as_, -zcom), (-ac, -as_, -zcom), (ah, 0.0, -zcom)]
    zt = height - zcom
    if b == 0:
        return Polyhedron(_APEX3, v + [(0, 0, zt)])
    return Polyhedron(_P3_FRUSTUM, v + [(-bc, bs, zt), (-bc, -bs, zt), (bh, 0.0, zt)])


def pyramid4(base_edge, height, alpha):
    _need(base_edge > 0, "Pyramid4: invalid parameter: base_edge<=0")
    _need(height > 0, "Pyramid4: invalid parameter: height<=0")
    _need(alpha > 0, "Pyramid4: invalid parameter: alpha<=0")
    _need(alpha < _PI, "Pyramid4: invalid parameter: alpha>=pi")
    a = base_edge / 2
    reduction = height * _cotangent(alpha)
    b = _dimension_or_zero(a - reduction, a + abs(reduction) + height)
    _need(b >= 0, "Pyramid4: invalid outcome: top_edge < 0")
    zcom = height / 4 * (a * a + 2 * a * b + 3 * b * b) / (a * a + a * b + b * b)
    v = [(-a, -a, -zcom), (a, -a, -zcom), (a, a, -zcom), (-a, a, -zcom)]
    zt = height - zcom
    if b == 0:
        return Polyhedron(_APEX4, v + [(0, 0, zt)])
    return Polyhedron(_P2_FRUSTUM, v + [(-b, -b, zt), (b, -b, zt), (b, b, zt), (-b, b, zt)])


_P6_FRUSTUM = Topology([((5, 4, 3, 2, 1, 0), True), ((0, 1, 7, 6), False), ((1, 2, 8, 7), False), ((2, 3, 9, 8), False),
                        ((3, 4, 10, 9), False), ((4, 5, 11, 10), False), ((5, 0, 6, 11), False),
                        ((6, 7, 8, 9, 10, 11), True)], False)  # fmt: skip
_APEX6 = Topology([((5, 4, 3, 2, 1, 0), True), ((0, 1, 6), False), ((1, 2, 6), False), ((2, 3, 6), False),
                   ((3, 4, 6), False), ((4, 5, 6), False), ((5, 0, 6), False)], False)  # fmt: skip


def pyramid6(base_edge, height, alpha):
    _need(base_edge > 0, "Pyramid6: invalid parameter: base_edge<=0")
    _need(height > 0, "Pyramid6: invalid parameter: height<=0")
    _need(alpha > 0, "Pyramid6: invalid parameter: alpha<=0")
    _need(alpha < _PI, "Pyramid6: invalid parameter: alpha>=pi")
    a = base_edge
    reduction = 2 / math.sqrt(3) * _cotangent(alpha) * height
    b = _dimension_or_zero(a - reduction, a + abs(reduction) + height)
    _need(b >= 0, "Pyramid6: invalid outcome: top_edge < 0")
    as_, ac = a / 2, a * math.sqrt(3) / 2
    bs, bc = b / 2, b * math.sqrt(3) / 2
    zcom = height / 4 * (a * a + 2 * a * b + 3 * b * b) / (a * a + a * b + b * b)
    v = [(a, 0.0, -zcom), (as_, ac, -zcom), (-as_, ac, -zcom), (-a, 0.0, -zcom), (-as_, -ac, -zcom), (as_, -ac, -zcom)]
    zt = height - zcom
    if b == 0:
        return Polyhedron(_APEX6, v + [(0, 0, zt)])
    return Polyhedron(_P6_FRUSTUM, v + [(b, 0.0, zt), (bs, bc, zt), (-bs, bc, zt), (-b, 0.0, zt), (-bs, -bc, zt), (bs, -bc, zt)])  # fmt: skip


# -------------------------------------------------------------------------------------------------- bipyramid
_BP_LOWER_PYRAMID = Topology([((0, 2, 1), False), ((0, 3, 2), False), ((0, 4, 3), False), ((0, 1, 4), False),
                              ((1, 2, 3, 4), True)], False)  # fmt: skip
_BP_BOTH_APICES = Topology([((0, 2, 1), False), ((0, 3, 2), False), ((0, 4, 3), False), ((0, 1, 4), False),
                            ((1, 2, 5), False), ((2, 3, 5), False), ((3, 4, 5), False), ((4, 1, 5), False)], False)  # fmt: skip
_BP_LOWER_APEX = Topology([((0, 2, 1), False), ((0, 3, 2), False), ((0, 4, 3), False), ((0, 1, 4), False),
                           ((1, 2, 6, 5), False), ((2, 3, 7, 6), False), ((3, 4, 8, 7), False), ((4, 1, 5, 8), False),
                           ((5, 6, 7, 8), True)], False)  # fmt: skip
_BP_UPPER_APEX = Topology([((3, 2, 1, 0), True), ((0, 1, 5, 4), False), ((1, 2, 6, 5), False), ((2, 3, 7, 6), False),
                           ((3, 0, 4, 7), False), ((4, 5, 8), False), ((5, 6, 8), False), ((6, 7, 8), False),
                           ((7, 4, 8), False)], False)  # fmt: skip
_BP_FULL = Topology([((3, 2, 1, 0), True), ((0, 1, 5, 4), False), ((1, 2, 6, 5), False), ((2, 3, 7, 6), False),
                     ((3, 0, 4, 7), False), ((4, 5, 9, 8), False), ((5, 6, 10, 9), False), ((6, 7, 11, 10), False),
                     ((7, 4, 8, 11), False), ((8, 9, 10, 11), True)], False)  # fmt: skip


def bipyramid4(length, height, height_ratio, alpha):
    _need(length > 0, "Bipyramid4: invalid parameter: length<=0")
    _need(height > 0, "Bipyramid4: invalid parameter: height<=0")
    _need(height_ratio >= 0, "Bipyramid4: invalid parameter: height_ratio<0")
    _need(alpha > 0, "Bipyramid4: invalid parameter: alpha<=0")
    _need(alpha <= _PI_2, "Bipyramid4: invalid parameter: alpha>pi/2")
    half_length = length / 2
    upper_height = _dimension_or_zero(height_ratio * height, height)
    cot_alpha = _cotangent(alpha)
    lower_red = cot_alpha * height
    upper_red = cot_alpha * upper_height
    a = _dimension_or_zero(half_length - lower_red, half_length + abs(lower_red) + height)
    b = half_length
    c = _dimension_or_zero(half_length - upper_red, half_length + abs(upper_red) + upper_height)
    _need(a >= 0 and c >= 0, "Bipyramid4: parameters violate condition 2*height <= length*tan(alpha)*min(1,1/height_ratio)")
    lower_vol = 4 * height / 3 * (a * a + a * b + b * b)
    lower_cen = -height + height / 4 * (a * a + 2 * a * b + 3 * b * b) / (a * a + a * b + b * b)
    upper_vol = 4 * upper_height / 3 * (b * b + b * c + c * c)
    upper_cen = upper_height / 4 * (b * b + 2 * b * c + 3 * c * c) / (b * b + b * c + c * c)
    zcom = (lower_vol * lower_cen + upper_vol * upper_cen) / (lower_vol + upper_vol)
    za, zb, zc = -height - zcom, -zcom, upper_height - zcom
    if upper_height == 0 and a == 0:
        return Polyhedron(_BP_LOWER_PYRAMID, [(0, 0, za), (-b, -b, zb), (b, -b, zb), (b, b, zb), (-b, b, zb)])
    if upper_height == 0:
        return Polyhedron(_P2_FRUSTUM, [(-a, -a, za), (a, -a, za), (a, a, za), (-a, a, za),
                                        (-b, -b, zb), (b, -b, zb), (b, b, zb), (-b, b, zb)])  # fmt: skip
    if a == 0 and c == 0:
        return Polyhedron(_BP_BOTH_APICES, [(0, 0, za), (-b, -b, zb), (b, -b, zb), (b, b, zb), (-b, b, zb), (0, 0, zc)])
    if a == 0:
        return Polyhedron(_BP_LOWER_APEX, [(0, 0, za), (-b, -b, zb), (b, -b, zb), (b, b, zb), (-b, b, zb),
                                           (-c, -c, zc), (c, -c, zc), (c, c, zc), (-c, c, zc)])  # fmt: skip
    if c == 0:
        return Polyhedron(_BP_UPPER_APEX, [(-a, -a, za), (a, -a, za), (a, a, za), (-a, a, za),
                                           (-b, -b, zb), (b, -b, zb), (b, b, zb), (-b, b, zb), (0, 0, zc)])  # fmt: skip
    return Polyhedron(_BP_FULL, [(-a, -a, za), (a, -a, za), (a, a, za), (-a, a, za),
                                 (-b, -b, zb), (b, -b, zb), (b, b, zb), (-b, b, zb),
                                 (-c, -c, zc), (c, -c, zc), (c, c, zc), (-c, c, zc)])  # fmt: skip


# --------------------------------------------------------------------------------------------------- others
_CANT = Topology(
    [((0, 1, 2, 3), True), ((0, 8, 5, 1), True), ((1, 9, 6, 2), True), ((2, 10, 7, 3), True), ((3, 11, 4, 0), True),
     ((0, 4, 8), False), ((1, 5, 9), False), ((2, 6, 10), False), ((3, 7, 11), False),
     ((4, 12, 16, 8), True), ((5, 13, 17, 9), True), ((4, 11, 19, 12), True), ((5, 8, 16, 13), True),
     ((7, 10, 18, 15), True), ((6, 9, 17, 14), True), ((7, 15, 19, 11), True), ((6, 14, 18, 10), True),
     ((13, 21, 17), False), ((12, 20, 16), False), ((15, 23, 19), False), ((14, 22, 18), False),
     ((14, 17, 21, 22), True), ((13, 16, 20, 21), True), ((12, 19, 23, 20), True), ((15, 18, 22, 23), True),
     ((20, 23, 22, 21), True)],
    True)  # fmt: skip


def cantellated_cube(length, removed_length):
    _need(length > 0, "CantellatedCube: invalid parameter: length<=0")
    _need(removed_length >= 0, "CantellatedCube: invalid parameter: removed_length<0")
    _need(removed_length <= length / 2, "CantellatedCube: invalid parameter: removed_length>length/2")
    a = length / 2
    c = a - removed_length
    _need(c >= 0, "CantellatedCube: invalid outcome: c<0")
    v = [(-c, -c, +a), (+c, -c, +a), (+c, +c, +a), (-c, +c, +a), (-a, -c, +c),
         (+c, -a, +c), (+a, +c, +c), (-c, +a, +c), (-c, -a, +c), (+a, -c, +c), (+c, +a, +c), (-a, +c, +c), (-a, -c, -c),
         (+c, -a, -c), (+a, +c, -c), (-c, +a, -c), (-c, -a, -c), (+a, -c, -c), (+c, +a, -c), (-a, +c, -c), (-c, -c, -a),
         (+c, -c, -a), (+c, +c, -a), (-c, +c, -a)]  # fmt: skip
    return Polyhedron(_CANT, v)


_TRUNC = Topology(
    [((0, 1, 7, 6, 9, 10, 4, 3), True), ((0, 2, 1), False), ((3, 4, 5), False), ((9, 11, 10), False), ((6, 7, 8), False),
     ((0, 3, 5, 17, 15, 12, 14, 2), True), ((4, 10, 11, 23, 22, 16, 17, 5), True),
     ((1, 2, 14, 13, 19, 20, 8, 7), True), ((6, 8, 20, 18, 21, 23, 11, 9), True),
     ((15, 17, 16), False), ((12, 13, 14), False), ((18, 20, 19), False), ((21, 22, 23), False),
     ((12, 15, 16, 22, 21, 18, 19, 13), True)],
    True)  # fmt: skip


def truncated_cube(length, removed_length):
    _need(length > 0, "TruncatedCube: invalid parameter: length<=0")
    _need(removed_length >= 0, "TruncatedCube: invalid parameter: removed_length<0")
    _need(removed_length <= length / 2, "TruncatedCube: invalid parameter: removed_length>length/2")
    a = length / 2
    c = a - removed_length
    _need(c >= 0, "TruncatedCube: invalid outcome: c<0")
    v = [(-c, -a, -a), (-a, -c, -a), (-a, -a, -c), (c, -a, -a), (a, -c, -a), (a, -a, -c),
         (-c, a, -a), (-a, c, -a), (-a, a, -c), (c, a, -a), (a, c, -a), (a, a, -c),
         (-c, -a, a), (-a, -c, a), (-a, -a, c), (c, -a, a), (a, -c, a), (a, -a, c),
         (-c, a, a), (-a, c, a), (-a, a, c), (c, a, a), (a, c, a), (a, a, c)]  # fmt: skip
    return Polyhedron(_TRUNC, v)


#: name -> factory; the names are those of libformfactor / BornAgain
FACTORIES = {
    "Box": box, "Prism3": prism3, "Prism6": prism6,
    "Tetrahedron": tetrahedron, "Octahedron": octahedron, "Dodecahedron": dodecahedron, "Icosahedron": icosahedron,
    "Pyramid2": pyramid2, "Pyramid3": pyramid3, "Pyramid4": pyramid4, "Pyramid6": pyramid6,
    "Bipyramid4": bipyramid4, "CantellatedCube": cantellated_cube, "TruncatedCube": truncated_cube,
}  # fmt: skip
