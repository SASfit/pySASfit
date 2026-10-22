"""The ``pysasfit-studio`` command line: generated files and sub-commands. No SASfit library needed."""

import ast
from types import SimpleNamespace

import pytest

from pysasfit.studio import cli


def fake_info(name="ff_demo", type="ff"):
    """A function description with infinite bounds and a text that contains the letters 'inf'."""
    pars = [
        SimpleNamespace(name="R", default=5.0, bounds=(0.0, float("inf")), description="influence of the radius, info"),
        SimpleNamespace(name="x", default=0.0, bounds=(float("-inf"), float("inf")), description="it's a \"quoted\" text"),
        SimpleNamespace(name="n", default=3, bounds=(1, 10), description=""),
    ]
    return SimpleNamespace(name=name, label="Demo", brief="a demo", type=type, parameters=pars)


@pytest.fixture()
def demo(monkeypatch):
    monkeypatch.setattr(cli.headers, "info", lambda name: fake_info(name, "sq" if name.startswith("sq_") else "ff"))


def namespace_of(path):
    ns = {}
    exec(compile(open(path).read(), path, "exec"), {"__name__": "adapter", "np": None}, ns)  # noqa: S102
    return ns


# ---- new_plugin ---------------------------------------------------------------------------------------------


def test_new_plugin_creates_plugin_and_test_with_the_name_filled_in(tmp_path):
    plugin, test = cli.new_plugin("my_gauss", str(tmp_path))
    assert plugin == str(tmp_path / "sasstudio_my_gauss.py")
    assert test == str(tmp_path / "tests" / "sasstudio_my_gauss_test.py")
    for fn in (plugin, test):
        text = open(fn).read()
        assert "PLUGIN_NAME" not in text and "my_gauss" in text
        ast.parse(text)
    assert "scattering_intensity_my_gauss" in open(plugin).read()


def test_new_plugin_never_overwrites(tmp_path):
    cli.new_plugin("a", str(tmp_path))
    with pytest.raises(FileExistsError):
        cli.new_plugin("a", str(tmp_path))
    plugin, _ = cli.new_plugin("b", str(tmp_path))  # another name is fine
    assert plugin.endswith("sasstudio_b.py")


def test_new_plugin_defaults_to_the_users_plugin_folder(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("USERPROFILE", str(tmp_path))  # Windows
    plugin, _ = cli.new_plugin("zz_demo")
    assert ".pysasfit" in plugin and "studio_plugins" in plugin


# ---- sasview_adapter ----------------------------------------------------------------------------------------


def test_the_adapter_is_valid_python_with_the_parameters(tmp_path, demo):
    fn = cli.sasview_adapter("ff_demo", str(tmp_path))
    ast.parse(open(fn).read())
    ns = namespace_of(fn)
    assert ns["name"] == "ff_demo" and ns["title"] == "Demo" and ns["description"] == "a demo"
    assert ns["category"] == "shape:sasfit" and ns["structure_factor"] is False
    assert [p[0] for p in ns["parameters"]] == ["R", "x", "n"]
    assert ns["parameters"][0][3] == [0.0, float("inf")]
    assert ns["parameters"][1][3] == [float("-inf"), float("inf")]
    assert ns["parameters"][2][3] == [1, 10]


def test_words_containing_inf_are_not_mangled(tmp_path, demo):
    """Only infinite numbers become float('inf'); descriptions such as 'influence' stay as they are."""
    ns = namespace_of(cli.sasview_adapter("ff_demo", str(tmp_path)))
    assert ns["parameters"][0][5] == "influence of the radius, info"
    assert ns["parameters"][1][5] == "it's a \"quoted\" text"


def test_a_structure_factor_gets_its_own_category(tmp_path, demo):
    ns = namespace_of(cli.sasview_adapter("sq_demo", str(tmp_path)))
    assert ns["structure_factor"] is True and ns["category"] == "structure-factor"


def test_the_adapter_never_overwrites(tmp_path, demo):
    cli.sasview_adapter("ff_demo", str(tmp_path))
    with pytest.raises(FileExistsError):
        cli.sasview_adapter("ff_demo", str(tmp_path))


def test_the_adapter_creates_missing_folders(tmp_path, demo):
    fn = cli.sasview_adapter("ff_demo", str(tmp_path / "a" / "b"))
    assert fn.startswith(str(tmp_path / "a" / "b"))


def test_py_only_touches_infinities():
    assert cli._py(float("inf")) == 'float("inf")'
    assert cli._py(float("-inf")) == '-float("inf")'
    assert cli._py(2.5) == "2.5" and cli._py(3) == "3" and cli._py(1e300) == "1e+300"


# ---- sub-commands -------------------------------------------------------------------------------------------


def test_list_shows_the_builtin_functions(capsys):
    assert cli.main(["list", "--type", "sd"]) == 0
    out = capsys.readouterr().out
    assert "sd_LogNorm" in out and "ff_sphere" not in out


def test_search_filters_by_type(capsys):
    cli.main(["search", "hard", "sphere", "--type", "sq"])
    lines = [x for x in capsys.readouterr().out.splitlines() if x.strip()]
    assert lines and all(x.startswith("sq_") for x in lines)


def test_info_of_an_unknown_function_raises_key_error():
    with pytest.raises(KeyError):
        cli.main(["info", "no_such_function_xyz"])


def test_new_plugin_command(tmp_path, capsys):
    cli.main(["new-plugin", "from_cli", "--dir", str(tmp_path)])
    assert "created" in capsys.readouterr().out
    assert (tmp_path / "sasstudio_from_cli.py").is_file()


def test_status_reports_the_folders(capsys):
    cli.main(["status"])
    out = capsys.readouterr().out
    assert "functions documented:" in out and "SASfit libraries:" in out and "group files" in out


def test_tree_lists_the_three_menus(capsys):
    cli.main(["tree"])
    out = capsys.readouterr().out
    headings = [line for line in out.splitlines() if line and not line.startswith(" ") and line.endswith(":")]
    assert len(headings) == 3


def test_tree_of_one_menu_only(capsys):
    cli.main(["tree", "sd"])
    out = capsys.readouterr().out
    headings = [line for line in out.splitlines() if line.endswith(":") and not line.startswith(" ")]
    assert len(headings) == 1


def test_without_a_command_the_help_is_printed(capsys):
    assert cli.main([]) == 0
    assert "usage" in capsys.readouterr().out.lower()
