"""Where the SASfit libraries, headers and plugins are found. Pure Python; the settings file is redirected."""

import json
import os

import pytest

from pysasfit.studio import paths


@pytest.fixture(autouse=True)
def sandbox(tmp_path, monkeypatch):
    """Own settings file, no environment variables, no choice made in the GUI."""
    monkeypatch.setattr(paths, "SETTINGS_FILE", str(tmp_path / "home" / "settings.json"))
    monkeypatch.setattr(paths, "_override", None)
    monkeypatch.setattr(paths, "DEFAULT_LIB", str(tmp_path / "no_bundled_lib"))
    for var in ("PYSASFIT_SASFIT_LIB", "SASFIT_LOCATION", "PYSASFIT_STUDIO_PLUGINS"):
        monkeypatch.delenv(var, raising=False)
    return tmp_path


def make_lib(root, n_plugins=2, plugin_dir=True):
    root.mkdir(parents=True, exist_ok=True)
    (root / f"libsasfit.{paths.LIB_EXT}").write_text("")
    target = root / "plugins" if plugin_dir else root
    target.mkdir(exist_ok=True)
    for i in range(n_plugins):
        (target / f"libsasfit_ff{i}.{paths.LIB_EXT}").write_text("")
    return root


# ---- settings file ------------------------------------------------------------------------------------------


def test_settings_roundtrip_and_removal():
    assert paths.get_setting("a", "dflt") == "dflt"
    paths.set_setting("a", 1)
    paths.set_setting("b", [1, 2])
    assert paths.get_setting("a") == 1 and paths.get_setting("b") == [1, 2]
    paths.set_setting("a", None)
    assert paths.get_setting("a") is None and paths.get_setting("b") == [1, 2]


def test_a_corrupt_settings_file_is_ignored():
    os.makedirs(os.path.dirname(paths.SETTINGS_FILE))
    with open(paths.SETTINGS_FILE, "w") as fh:
        fh.write("{not json")
    assert paths.get_setting("a", 5) == 5
    paths.set_setting("a", 1)  # and is overwritten by the next write
    assert paths.get_setting("a") == 1


def test_a_settings_file_that_is_not_a_dict_is_ignored():
    os.makedirs(os.path.dirname(paths.SETTINGS_FILE))
    with open(paths.SETTINGS_FILE, "w") as fh:
        json.dump([1, 2], fh)
    assert paths.get_setting("a") is None


# ---- finding the library ------------------------------------------------------------------------------------


def test_find_lib_dir_accepts_the_folder_or_a_parent_or_the_plugin_folder(sandbox):
    lib = make_lib(sandbox / "sasfit" / "lib")
    assert paths.find_lib_dir(str(lib)) == str(lib)
    assert paths.find_lib_dir(str(sandbox / "sasfit")) == str(lib)
    assert paths.find_lib_dir(str(lib / "plugins")) == str(lib)


def test_find_lib_dir_rejects_empty_and_wrong_folders(sandbox):
    (sandbox / "empty").mkdir()
    assert paths.find_lib_dir("") is None
    assert paths.find_lib_dir(None) is None
    assert paths.find_lib_dir(str(sandbox / "empty")) is None
    assert paths.find_lib_dir(str(sandbox / "missing")) is None


def test_find_lib_dir_strips_quotes_and_blanks(sandbox):
    lib = make_lib(sandbox / "lib")
    assert paths.find_lib_dir(f'  "{lib}" ') == str(lib)


def test_count_plugin_libs_counts_both_places(sandbox):
    lib = make_lib(sandbox / "lib", n_plugins=3)
    (lib / f"libsasfit_extra.{paths.LIB_EXT}").write_text("")
    (lib / "readme.txt").write_text("")
    assert paths.count_plugin_libs(str(lib)) == 4


def test_no_library_anywhere():
    assert paths.lib_dir() is None
    assert paths.lib_dir_source() is None
    assert paths.plugin_lib_dirs() == []


def test_the_gui_choice_beats_the_environment(sandbox, monkeypatch):
    gui = make_lib(sandbox / "gui")
    env = make_lib(sandbox / "env")
    monkeypatch.setenv("PYSASFIT_SASFIT_LIB", str(env))
    paths.set_lib_dir(str(gui))
    assert paths.lib_dir() == str(gui) and paths.lib_dir_source() == "gui"
    paths.set_lib_dir(None)
    assert paths.lib_dir() == str(env) and paths.lib_dir_source() == "PYSASFIT_SASFIT_LIB"


def test_priority_environment_then_sasfit_location_then_bundled(sandbox, monkeypatch):
    a = make_lib(sandbox / "a")
    b = make_lib(sandbox / "b")
    bundled = make_lib(sandbox / "bundled")
    monkeypatch.setattr(paths, "DEFAULT_LIB", str(bundled))
    assert paths.lib_dir_source() == "bundled"
    monkeypatch.setenv("SASFIT_LOCATION", str(b))
    assert paths.lib_dir_source() == "SASFIT_LOCATION"
    monkeypatch.setenv("PYSASFIT_SASFIT_LIB", str(a))
    assert paths.lib_dir_source() == "PYSASFIT_SASFIT_LIB"


def test_a_wrong_first_choice_falls_through_to_the_next(sandbox, monkeypatch):
    good = make_lib(sandbox / "good")
    monkeypatch.setenv("PYSASFIT_SASFIT_LIB", str(sandbox / "nothing_here"))
    monkeypatch.setenv("SASFIT_LOCATION", str(good))
    assert paths.lib_dir() == str(good)
    rows = {key: (raw, resolved) for key, _, raw, resolved in paths.lib_dir_sources()}
    assert rows["PYSASFIT_SASFIT_LIB"][0] and rows["PYSASFIT_SASFIT_LIB"][1] is None


def test_the_saved_choice_is_used_without_set_lib_dir(sandbox):
    lib = make_lib(sandbox / "saved")
    paths.set_lib_dir(str(lib), save=True)
    paths._override = None  # as after a restart
    assert paths.saved_lib_dir() == str(lib)
    assert paths.lib_dir() == str(lib)
    paths.set_lib_dir("", save=True)
    paths._override = None
    assert paths.saved_lib_dir() is None


def test_set_lib_dir_without_save_leaves_the_file_alone(sandbox):
    paths.set_lib_dir(str(make_lib(sandbox / "x")))
    assert not os.path.exists(paths.SETTINGS_FILE)


def test_describe_marks_the_active_source(sandbox, monkeypatch):
    monkeypatch.setenv("PYSASFIT_SASFIT_LIB", str(sandbox / "broken"))
    monkeypatch.setenv("SASFIT_LOCATION", str(make_lib(sandbox / "ok")))
    text = paths.describe_lib_dir_sources()
    lines = text.splitlines()
    assert len(lines) == 4
    assert lines[2].startswith("->") and "ok" in lines[2]
    assert "bad" in lines[1]
    assert "(not set)" in lines[0]


# ---- plugin and header folders ------------------------------------------------------------------------------


def test_plugin_lib_dirs_lists_plugins_folder_and_lib(sandbox, monkeypatch):
    lib = make_lib(sandbox / "lib")
    monkeypatch.setenv("PYSASFIT_SASFIT_LIB", str(lib))
    assert paths.plugin_lib_dirs() == [str(lib / "plugins"), str(lib)]


def test_header_dirs_include_the_bundled_ones_and_the_library_headers(sandbox, monkeypatch):
    lib = make_lib(sandbox / "lib")
    (lib / "plugins" / "sasfit_ff_sphere.h").write_text("")
    monkeypatch.setenv("PYSASFIT_SASFIT_LIB", str(lib))
    dirs = paths.header_dirs()
    assert dirs[0] == paths.BUNDLED_HEADERS
    assert str(lib / "plugins") in dirs and str(lib) not in dirs


def test_def_dirs_prefer_a_def_subfolder_and_have_no_duplicates(sandbox, monkeypatch):
    plug = sandbox / "myplugins"
    (plug / "def").mkdir(parents=True)
    monkeypatch.setenv("PYSASFIT_STUDIO_PLUGINS", str(plug))
    dirs = paths.def_dirs()
    assert str(plug / "def") in dirs
    assert len(dirs) == len(set(dirs))


def test_python_plugin_dirs_from_the_environment(sandbox, monkeypatch):
    a, b = sandbox / "a", sandbox / "b"
    a.mkdir()
    monkeypatch.setenv("PYSASFIT_STUDIO_PLUGINS", os.pathsep.join([str(a), str(b)]))  # b does not exist
    assert str(a) in paths.python_plugin_dirs()
    assert str(b) not in paths.python_plugin_dirs()
