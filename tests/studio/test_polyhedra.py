# SPDX-License-Identifier: GPL-3.0-or-later
"""Tests of pysasfit.studio.polyhedra (port of libformfactor)."""

import json
import math
import pathlib

import numpy as np
import pytest

from pysasfit.studio.polyhedra import (
    FACTORIES,
    oriented_amplitude,
    orientational_average,
    rotation_zyz,
    sphere_rule,
)
from pysasfit.studio.polyhedra import specs

_REFDATA = json.loads((pathlib.Path(__file__).with_name("polyhedra_reference.json")).read_text())
REF = _REFDATA["shapes"]
_DIRS = np.array(_REFDATA["directions"])
_QR = _REFDATA["qR_list"]
SPEC = {s["name"]: s for s in specs.SPECS}


def _p(*values):
    p = np.zeros(50)
    p[: len(values)] = values
    return p


# ------------------------------------------------------------------------- C++ reference (libformfactor 0.5.2)
@pytest.mark.parametrize("name", sorted(REF))
def test_form_factor_against_cpp(name):
    r = REF[name]
    sh = FACTORIES[name](*r["params"])
    assert sh.volume == pytest.approx(r["volume"], rel=1e-12)
    assert sh.radius == pytest.approx(r["radius"], rel=1e-12)
    q = np.array([x / sh.radius * d for x in _QR for d in _DIRS])
    F = sh.ff(q)
    ref = np.array(r["F_re"]) + 1j * np.array(r["F_im"])
    err = np.abs(F - ref) / np.maximum(np.abs(ref), 1e-15 * r["volume"])
    assert err.max() < 1e-9


# ------------------------------------------------------------------------------------ exact / limiting cases
def test_closed_form_volumes():
    a = 3.0
    assert FACTORIES["Tetrahedron"](a).volume == pytest.approx(a**3 / (6 * math.sqrt(2)))
    assert FACTORIES["Octahedron"](a).volume == pytest.approx(math.sqrt(2) / 3 * a**3)
    assert FACTORIES["Dodecahedron"](a).volume == pytest.approx((15 + 7 * math.sqrt(5)) / 4 * a**3)
    assert FACTORIES["Icosahedron"](a).volume == pytest.approx(5 * (3 + math.sqrt(5)) / 12 * a**3)
    assert FACTORIES["Prism6"](a, 2.0).volume == pytest.approx(3 * math.sqrt(3) / 2 * a**2 * 2.0)
    assert FACTORIES["Pyramid4"](4.0, 3.0, math.pi / 2).volume == pytest.approx(48.0)


def test_pyramid4_with_right_angle_is_a_box():
    q = np.random.default_rng(3).normal(size=(30, 3)) * 0.3
    f1 = FACTORIES["Pyramid4"](2.0, 3.0, math.pi / 2).ff(q)
    f2 = FACTORIES["Box"](2.0, 2.0, 3.0).ff(q)
    assert np.abs(f1 - f2).max() < 1e-12


def test_truncated_cube_without_truncation_is_a_box():
    q = np.random.default_rng(4).normal(size=(30, 3)) * 0.3
    f1 = FACTORIES["TruncatedCube"](4.0, 0.0).ff(q)
    f2 = FACTORIES["Box"](4.0, 4.0, 4.0).ff(q)
    assert np.abs(f1 - f2).max() < 1e-11


@pytest.mark.parametrize("name", sorted(REF))
def test_hermitian_symmetry(name):
    sh = FACTORIES[name](*REF[name]["params"])
    q = np.random.default_rng(5).normal(size=(20, 3)) / sh.radius
    assert np.abs(sh.ff(-q) - np.conj(sh.ff(q))).max() < 1e-10 * sh.volume


def test_centrosymmetric_shapes_have_real_amplitude():
    for name in ("Box", "Prism6", "Octahedron", "Dodecahedron", "Icosahedron", "TruncatedCube", "CantellatedCube"):
        sh = FACTORIES[name](*REF[name]["params"])
        q = np.random.default_rng(6).normal(size=(10, 3)) / sh.radius
        assert np.abs(sh.ff(q).imag).max() < 1e-12 * sh.volume


def test_centre_of_mass_is_the_origin():
    for name in ("Tetrahedron", "Pyramid3", "Pyramid2", "Bipyramid4"):
        sh = FACTORIES[name](*REF[name]["params"])
        assert np.abs(sh.center_of_mass()).max() < 1e-12 * sh.radius


def test_invalid_parameters_raise():
    with pytest.raises(ValueError):
        FACTORIES["Box"](1.0, -1.0, 1.0)
    with pytest.raises(ValueError):
        FACTORIES["Pyramid4"](1.0, 10.0, 0.3)  # top edge would be negative


# ---------------------------------------------------------------------------------------- orientation average
@pytest.mark.parametrize("name", sorted(REF))
def test_average_limits(name):
    sh = FACTORIES[name](*REF[name]["params"])
    I0, A0 = orientational_average(sh, [0.0])
    assert I0[0] == pytest.approx(sh.volume**2, rel=1e-12)
    assert A0[0] == pytest.approx(sh.volume, rel=1e-12)


@pytest.mark.parametrize("name", ["Box", "Tetrahedron", "Pyramid2", "TruncatedCube"])
def test_average_converged(name):
    sh = FACTORIES[name](*REF[name]["params"])
    q = np.array([0.3, 2.0, 8.0, 30.0]) / sh.radius
    I1, A1 = orientational_average(sh, q)
    I2, A2 = orientational_average(sh, q, safety=1.6)
    assert np.max(np.abs(I1 - I2) / I2) < 1e-9


@pytest.mark.parametrize("name", sorted(REF))
def test_porod_limit(name):
    """<I> q^4 -> 2 pi S (loose: edge corrections and fringes are of the order of a few percent)."""
    sh = FACTORIES[name](*REF[name]["params"])
    q = np.linspace(20, 40, 60) / sh.radius
    I, _ = orientational_average(sh, q)
    ratio = np.trapezoid(I * q**4, q) / (q[-1] - q[0]) / (2 * math.pi * sh.surface_area())
    assert 0.92 < ratio < 1.08


def test_sphere_rule_weights():
    d, w = sphere_rule(20, 30)
    assert w.sum() == pytest.approx(1.0)
    assert np.allclose(np.linalg.norm(d, axis=1), 1.0)
    assert np.sum(w * d[:, 2] ** 2) == pytest.approx(1 / 3)


# ------------------------------------------------------------------------------------------- orientation
def test_oriented_amplitude_rotation():
    sh = FACTORIES["Tetrahedron"](5.0)
    R = rotation_zyz(0.3, 0.7, -0.4)
    q = np.random.default_rng(7).normal(size=(10, 3)) * 0.2
    assert np.allclose(oriented_amplitude(sh, q @ R.T, R), sh.ff(q))  # q_lab = R q_body
    assert np.allclose(R @ R.T, np.eye(3))


# ------------------------------------------------------------------------------------------ registered specs
def test_spec_count_and_names():
    # 14 BornAgain shapes + cube, 13 Archimedean, 13 Catalan, 92 Johnson, 4 Kepler-Poinsot, 4 families; x 2 (oriented)
    assert len(SPEC) == 2 * (15 + 13 + 13 + 92 + 4 + 4)
    assert all(n.startswith("ff_ba_") for n in SPEC)


def test_menu_paths():
    assert all(s["group"].startswith("menu:") for s in specs.SPECS)
    assert {s["group"][5:].split("/")[0] for s in specs.SPECS} == {"BornAgain", "Polyhedra"}
    g = {n: SPEC[n]["group"] for n in SPEC}
    assert g["ff_ba_box"] == "menu:BornAgain" and g["ff_ba_box_oriented"] == "menu:BornAgain/oriented"
    assert g["ff_ba_icosahedron"] == "menu:Polyhedra/Platonic"
    assert g["ff_ba_arch_truncated_cube"] == "menu:Polyhedra/Archimedean"
    assert g["ff_ba_kp_great_dodecahedron"] == "menu:Polyhedra/Kepler-Poinsot"


def test_isotropic_spec_is_real_with_eta_scaling():
    s = SPEC["ff_ba_prism3"]
    q = np.array([0.0, 0.05, 0.3])
    p = _p(10.0, 6.0, 2.0)  # a, H, eta
    I, A = s["intensity"](q, p), s["amplitude"](q, p)
    assert I.dtype == float and A.dtype == float
    assert I[0] == pytest.approx((2.0 * FACTORIES["Prism3"](10.0, 6.0).volume) ** 2)
    assert np.allclose(s["intensity"](q, _p(10.0, 6.0, 1.0)) * 4, I)


def test_oriented_spec_has_complex_amplitude_and_real_intensity():
    s = SPEC["ff_ba_tetrahedron_oriented"]
    p = _p(10.0, 1.0, 0.0, 30.0, 20.0, 25.0)  # a, eta, alpha, beta, gamma, psi
    q = np.linspace(0.05, 0.6, 8)
    F = s["amplitude"](q, p)
    I = s["intensity"](q, p)
    assert np.iscomplexobj(F) and np.abs(F.imag).max() > 1e-3  # a tetrahedron is not centrosymmetric
    assert I.dtype == float and np.allclose(I, np.abs(F) ** 2)
    # psi in rad overrides the parameter and broadcasts
    Fm = s["amplitude"](q[:, None], p, np.linspace(0, np.pi, 5)[None, :])
    assert Fm.shape == (8, 5)


def test_oriented_spec_matches_direct_form_factor():
    s = SPEC["ff_ba_box_oriented"]
    p = _p(12.0, 6.0, 8.0, 1.5, 10.0, 20.0, 30.0, 40.0)
    q, psi = 0.2, math.radians(40.0)
    qv = q * np.array([math.cos(psi), math.sin(psi), 0.0])
    R = rotation_zyz(*np.radians([10.0, 20.0, 30.0]))
    expect = 1.5 * FACTORIES["Box"](12.0, 6.0, 8.0).ff(qv @ R)[0]
    assert complex(s["amplitude"](q, p)) == pytest.approx(expect)


def test_volume_callable_replaces_the_distributed_parameter():
    s = SPEC["ff_ba_box"]
    p = _p(12.0, 6.0, 8.0, 1.0)
    v = s["volume"](np.array([1.0, 2.0]), p, 1)
    assert np.allclose(v, [12 * 1 * 8, 12 * 2 * 8])
