"""Plugin loading via do_init and the common-function table, with mock SASfit libraries."""

import numpy as np
import pytest

from pysasfit import studio


def test_precision_defaults_are_set(studio_lib):
    p = studio_lib.get_precision()
    assert p["nriq"] == 1e-10 and p["int_strategy"] == 5  # P_CUBATURE
    studio_lib.set_precision(nriq=1e-7, int_strategy="GSL_QAGS")
    assert studio_lib.get_precision() == {"nriq": 1e-7, "int_strategy": 9}
    studio_lib.set_precision(nriq=1e-10, int_strategy=5)


def test_builtin_sphere(studio_lib):
    s = studio.function("ff_sphere")
    q = np.array([1e-5, 0.1, 0.3])
    V = 4 / 3 * np.pi * 10**3
    F = s.amplitude(q, R=10, eta=2)
    assert np.isclose(F[0], 2 * V)
    assert np.allclose(s(q, R=10, eta=2), F**2)
    assert np.isclose(s.volume(5.0, R=10, dist=0), 4 / 3 * np.pi * 125)
    assert np.isclose(s.volume(5.0, R=10, dist=1), V)


def test_plugin_uses_common_stubs(studio_lib):
    f = studio.function("ff_mock_scaled")
    assert f.parameter_vector()[0] == 1.5  # default from the header table
    assert np.allclose(f([0.0, 1.0], b=1.0), [3.0, 3.0 * np.exp(-1)])  # 2*A via stubs->twice
    assert np.isclose(f(0.0, 2.0, 0.0), 4.0)  # positional


def test_error_status(studio_lib):
    f = studio.function("ff_mock_scaled")
    with pytest.raises(studio.SASfitError, match="not allowed"):
        f(0.1, A=-1)
    g = studio.function("ff_mock_scaled", errors="ignore")
    assert g(0.1, A=-1) == 0.0
    assert g(0.0, A=1, b=0) == 2.0  # error flag was cleared


def test_dependency_between_plugins(mock_lib):
    from pysasfit.studio import library

    lib = library.SASfitLibrary(mock_lib)  # fresh: nothing loaded
    fi, _, _ = lib.pointers("sq_mock_dep", "sasfit_mockdep")  # needs ff_mock_scaled from another plugin
    assert fi and "ff_mock_scaled" in lib.registry
    assert len(lib.loaded) == 2


def test_size_distribution_plugin(studio_lib):
    box = studio.function("sd_mock_box")
    assert np.allclose(box([0.5, 1.5, 2.5], N=4, xmin=1, xmax=2), [0, 4, 0])


def test_missing_library_message(monkeypatch, tmp_path):
    from pysasfit.studio import library, paths

    monkeypatch.setenv("PYSASFIT_SASFIT_LIB", str(tmp_path))
    monkeypatch.delenv("SASFIT_LOCATION", raising=False)
    monkeypatch.setattr(paths, "DEFAULT_LIB", str(tmp_path))
    with pytest.raises(library.SASfitLibraryNotFound, match="install-binaries"):
        library.SASfitLibrary()


def test_install_binaries(mock_lib, tmp_path):
    from pysasfit.studio.install import install_binaries

    n = install_binaries(mock_lib, str(tmp_path / "dest"), verbose=False)
    assert n == 5  # libsasfit, 2 plugin libraries, 2 headers
    assert install_binaries(mock_lib, str(tmp_path / "dest"), verbose=False) == 0  # up to date
