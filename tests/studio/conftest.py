import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "mock_sasfit"))

# GUI tests run without a display.  The explorer selects the QtAgg backend when it is imported, which fails once
# bumps (used by the fitting tests) has switched matplotlib to a headless backend: import it first.
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
try:
    import pysasfit.studio.explorer  # noqa: F401
except Exception:  # noqa: BLE001 -- no PySide6: test_project is skipped
    pass


@pytest.fixture(scope="session")
def mock_lib(tmp_path_factory):
    """Mock SASfit installation (libsasfit + 2 plugins) compiled from tests/studio/mock_sasfit."""
    from build import build

    d = build(str(tmp_path_factory.mktemp("sasfit_lib")))
    if d is None:
        pytest.skip("no C compiler for the mock SASfit libraries")
    return d


@pytest.fixture
def isolated_settings(monkeypatch):
    """No settings of the user (saved SASfit folder, precision, integration) and nothing written to their file."""
    from pysasfit.studio import paths

    monkeypatch.setattr(paths, "_read_settings", lambda: {})
    monkeypatch.setattr(paths, "_write_settings", lambda data: None)
    monkeypatch.setattr(paths, "_override", None)


@pytest.fixture
def studio_lib(mock_lib, isolated_settings):
    from pysasfit.studio import library, paths

    old = library._LIB
    paths.set_lib_dir(mock_lib)
    library._LIB = None
    yield library.get_library()
    paths.set_lib_dir(None)
    library._LIB = old
