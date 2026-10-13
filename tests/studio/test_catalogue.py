"""Header catalogue and ctypes layout -- no SASfit libraries needed."""

import ast
import ctypes
import os

import pytest

from pysasfit import studio
from pysasfit.studio import cli, headers
from pysasfit.studio._ctypes import SasfitParam


def test_catalogue_counts():
    cat = studio.catalogue(header_dirs=[studio.paths.BUNDLED_HEADERS])
    n = {t: sum(p.type == t for p in cat.values()) for t in ("ff", "sq", "sd", "peak")}
    assert n["ff"] >= 449 + 50 and n["sq"] >= 138 and n["sd"] >= 42 + 10 and n["peak"] >= 74
    assert all(p.parameters for p in cat.values() if not p.builtin)


def test_plugin_entry():
    p = studio.info("ff_profile__power_law")
    assert p.collection == "sasfit_powerlaw" and p.library == "libsasfit_powerlaw"
    assert p.parameter_names == ["R", "alpha", "Deta"]
    assert p.symbol == "sasfit_ff_profile__power_law"
    assert p.brief == ""  # template placeholder dropped
    assert studio.info("profile__power_law") is p or studio.info("profile__power_law").name == p.name


def test_malformed_tables_are_parsed():
    # ff_boucher_cyl has '</tr></tr>' typos in its table
    p = studio.info("ff_boucher_cyl")
    assert len(p.parameters) >= 6 and "sigma_R0" in p.parameter_names


def test_builtin_functions_listed():
    s = studio.info("ff_sphere")
    assert s.builtin and s.parameter_names[0] == "R" and s.parameter_names[3] == "eta"
    assert studio.info("sd_LogNorm").parameter_names == ["N", "s", "p", "mu"]
    assert studio.info("sq_HardSphere").builtin


def test_search():
    hits = studio.search("hard", "sphere", type="sq")
    assert hits and all(h.type == "sq" for h in hits)
    with pytest.raises(KeyError):
        studio.info("no_such_function_xyz")


def test_param_struct_matches_sasfit_function_h():
    assert ctypes.sizeof(SasfitParam) == 1104
    assert SasfitParam.errStr.offset == 408
    assert SasfitParam.errStatus.offset == 668
    assert SasfitParam.xarr.offset == 672
    assert SasfitParam.more_p.offset == 696
    assert SasfitParam.function.offset == 1096


def test_profiles(tmp_path):
    files = headers.write_profiles(str(tmp_path), header_dirs=[studio.paths.BUNDLED_HEADERS])
    assert len(files) > 600
    txt = (tmp_path / "profile_ff_profile__power_law.txt").read_text()
    assert txt.startswith("R: radius")


def test_sasview_adapter_and_new_plugin(tmp_path):
    fn = cli.sasview_adapter("ff_profile__power_law", str(tmp_path))
    ast.parse(open(fn).read())
    plugin, test = cli.new_plugin("my_guinier", str(tmp_path / "plugins"))
    ast.parse(open(plugin).read())
    assert os.path.isfile(test)


def test_cli_list(capsys):
    cli.main(["search", "lognorm", "--type", "sd"])
    assert "sd_" in capsys.readouterr().out
