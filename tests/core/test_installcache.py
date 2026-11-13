"""The installations of ATSAS and DENSS are searched only on request and kept between sessions."""
import os
import subprocess
import sys

import pytest

from pysasfit.core import installcache


@pytest.fixture
def fresh_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(installcache, "cache_path", lambda: str(tmp_path / "installations.json"))
    return tmp_path


def test_put_get_forget(fresh_cache):
    assert installcache.get("x") is None
    installcache.put("x", {"a": 1})
    v = installcache.get("x")
    assert v["a"] == 1 and v["checked"]
    installcache.forget("x")
    assert installcache.get("x") is None


def test_atsas_found_once_is_not_searched_again(fresh_cache, tmp_path, monkeypatch):
    from pysasfit.atsas import core as A
    b = tmp_path / "ATSAS-4.1.4" / "bin"
    b.mkdir(parents=True)
    (b / ("autorg" + A.EXE_SUFFIX)).write_text("")
    monkeypatch.setattr(A, "_ATSAS_BIN", None)
    assert A.cached_atsas() is None
    assert A.find_atsas(str(b.parent)) == b              # an explicit folder is not stored
    assert A.cached_atsas() is None
    installcache.put("atsas", {"bin": str(b)})
    monkeypatch.setattr(A, "_default_roots", lambda: pytest.fail("searched although known"))
    monkeypatch.setattr(A.shutil, "which", lambda *_: pytest.fail("searched although known"))
    assert A.find_atsas() == b
    os.remove(b / ("autorg" + A.EXE_SUFFIX))                # uninstalled: the cache is not used
    assert A.cached_atsas() is None


def test_denss_found_once_starts_no_python(fresh_cache, monkeypatch):
    from pysasfit.density import core as D
    monkeypatch.setattr(D, "_INFO", {})
    installcache.put("denss:", {"python": sys.executable, "version": "1.8.7", "location": ""})
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: pytest.fail("a Python was started"))
    i = D.find_denss()
    assert i.python == sys.executable and i.version == "1.8.7"
    assert D.cached_denss("/no/such/python") is None


def test_preferences_show_the_installations_without_searching(fresh_cache, monkeypatch):
    from pysasfit.core import settings_schema as S
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: pytest.fail("a Python was started"))
    assert "Search again" in S._denss_found()
    assert "Search again" in S._atsas_found()
