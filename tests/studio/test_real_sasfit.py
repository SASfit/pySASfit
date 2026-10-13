"""Checks against the real SASfit libraries; skipped unless they are installed.

Install them with `pysasfit-studio install-binaries <SASstudio folder>` (or set
PYSASFIT_SASFIT_LIB / SASFIT_LOCATION), then run `pytest tests/studio -v`.
"""

import numpy as np
import pytest

from pysasfit import studio

pytestmark = pytest.mark.skipif(not studio.available(), reason="SASfit libraries not installed")


@pytest.fixture(scope="module")
def lib():
    from pysasfit.studio import library

    library._LIB = None
    return library.get_library()


def test_all_plugin_libraries_initialise(lib):
    n = lib.load_all()
    print(f"{n} plugin libraries loaded, {len(lib.registry)} functions; failed: {lib.failed}")
    assert n >= 0.9 * len(lib.plugin_files())
    cat = studio.catalogue()
    documented = {p.name for p in cat.values() if not p.builtin}
    missing = sorted(documented - set(lib.registry))
    print(f"documented in headers but not exported: {len(missing)}", missing[:20])


def test_builtin_sphere_lognorm_hardsphere(lib):
    q = np.geomspace(1e-3, 0.5, 40)
    R = 50.0
    I = studio.function("ff_sphere")(q, R=R, eta=1.0)
    ref = (4 * np.pi * (np.sin(q * R) - q * R * np.cos(q * R)) / q**3) ** 2
    assert np.allclose(I, ref, rtol=1e-8)
    x = np.linspace(10, 100, 30)
    N = studio.function("sd_LogNorm")(x, N=1, s=0.2, p=1, mu=30)
    assert np.allclose(N, np.exp(-0.5 * (np.log(x / 30) / 0.2) ** 2) / (np.sqrt(2 * np.pi) * 0.2 * x), rtol=1e-10)
    S = studio.function("sq_HardSphere")(q, RHS=R, fp=0.3)
    assert np.isclose(S[0], (1 - 0.3) ** 4 / (1 + 2 * 0.3) ** 2, rtol=1e-3)


def test_plugin_functions(lib):
    q = np.geomspace(1e-3, 0.5, 20)
    I = studio.function("ff_profile__power_law")(q, R=50, alpha=1.0, Deta=1.0)
    assert np.all(np.isfinite(I)) and I[0] > I[-1] > 0
    S = studio.function("sq_hard_sphere__py_")(q, R=50, eta=0.3)
    assert np.all(np.isfinite(S)) and S[0] < 0.5


def test_plugin_matches_builtin_where_both_exist(lib):
    """The plugin PY hard sphere (R, volume fraction eta) and the built-in one (RHS, fp) should agree."""
    q = np.geomspace(1e-3, 0.5, 20)
    a = studio.function("sq_HardSphere")(q, 50, 0.3)
    b = studio.function("sq_hard_sphere__py_")(q, 50, 0.3)
    assert np.allclose(a, b, rtol=1e-6)
