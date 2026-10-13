import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "mock_sasfit"))


@pytest.fixture(scope="session")
def mock_lib(tmp_path_factory):
    """Mock SASfit installation (libsasfit + 2 plugins) compiled from tests/studio/mock_sasfit."""
    from build import build

    d = build(str(tmp_path_factory.mktemp("sasfit_lib")))
    if d is None:
        pytest.skip("no C compiler for the mock SASfit libraries")
    return d


@pytest.fixture
def studio_lib(mock_lib):
    from pysasfit.studio import library, paths

    old = library._LIB
    paths.set_lib_dir(mock_lib)
    library._LIB = None
    yield library.get_library()
    paths.set_lib_dir(None)
    library._LIB = old
