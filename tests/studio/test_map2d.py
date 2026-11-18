"""2D detector maps of models with ψ-dependent functions (pysasfit.studio.map2d)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from pysasfit.studio import map2d, ports
from pysasfit.studio.model import ModelFunction, ScatteringContribution
from pysasfit.studio.ports import sheared


@pytest.fixture()
def model():
    ports.enable(["ff_sheared_cylinders__maier_saupe_"])
    try:
        c = ScatteringContribution(
            "ff_sheared_cylinders__maier_saupe_", [20, 5, 200, 1, 2, 0, 33, 0, 5, 0, 0]
        )
    finally:
        ports.disable(["ff_sheared_cylinders__maier_saupe_"])
    return ModelFunction([c])


def test_psi_parameter_found(model):
    ((vec, k),) = map2d.anisotropic_parameters(model)
    assert k == 6 and vec[k] == 33


def test_map_matches_direct_evaluation_and_restores_psi(model):
    qy, qz, img = map2d.detector_map(model, 0.1, n=64, n_psi=90, n_q=200)
    assert img.shape == (64, 64)
    assert model.contributions[0].ff_p[6] == 33  # restored
    p = list(model.contributions[0].ff_p[:11])
    for iy, iz in ((50, 20), (10, 40), (32, 60)):
        q = math.hypot(qy[iy], qz[iz])
        p[6] = math.degrees(math.atan2(qz[iz], qy[iy])) % 180.0
        ref = sheared.sheared_intensity(np.array([q]), p, kind="maier_saupe")[0]
        assert img[iz, iy] == pytest.approx(ref, rel=2e-3)


def test_centrosymmetric(model):
    _, _, img = map2d.detector_map(model, 0.1, n=33, n_psi=36)
    np.testing.assert_allclose(img, img[::-1, ::-1], rtol=1e-10)


def test_isotropic_model_is_radial():
    class Iso:
        contributions = ()

        @staticmethod
        def intensity(q):
            return np.exp(-(np.asarray(q) ** 2))

    qy, qz, img = map2d.detector_map(Iso(), 1.0, n=21, n_q=2000)
    Q = np.hypot(*np.meshgrid(qy, qz))
    np.testing.assert_allclose(img, np.exp(-(Q**2)), rtol=2e-3)
