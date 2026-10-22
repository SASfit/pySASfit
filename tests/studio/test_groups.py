"""Menu tree from the Doxygen group definitions (*.def). Pure Python, no SASfit library needed."""

import os
import time
from types import SimpleNamespace

import pytest

from pysasfit.studio import groups

DEF = r"""
/** \defgroup FF_top Form factors
 *  \toplvldescr{Form factors,sasfit_ff,ff}
 *  \brief all form factors
 */
/** \defgroup FF_sph Spheres
 *  \ingroup FF_top
 *  \brief Homogeneous \f$ R \f$ spheres
 */
/** \defgroup FF_cyl Cylinders
 *  \ingroup FF_top
 */
/** \defgroup SQ_top Structure factors
 *  \toplvldescr{Structure factors,sasfit_sq,sq}
 */
/* not a group comment */
"""


def entry(name, group="", type="ff", builtin=False, collection="sasfit_ff"):
    return SimpleNamespace(name=name, label=name.upper(), group=group, type=type, builtin=builtin, collection=collection)


@pytest.fixture()
def deffile(tmp_path):
    f = tmp_path / "groups.def"
    f.write_text(DEF)
    return f


@pytest.fixture()
def defs(deffile):
    return groups.load_groups([str(deffile.parent)])


# ---- parsing ------------------------------------------------------------------------------------------------


def test_parse_def_reads_groups_parents_and_toplevel(deffile):
    got = {g.name: g for g in groups.parse_def(str(deffile))}
    assert set(got) == {"FF_top", "FF_sph", "FF_cyl", "SQ_top"}
    assert got["FF_sph"].parent == "FF_top"
    assert got["FF_sph"].label == "Spheres"
    assert got["FF_top"].toplevel == ("Form factors", "sasfit_ff", "ff")
    assert got["FF_sph"].toplevel is None
    assert got["FF_cyl"].source == str(deffile)


def test_the_brief_drops_directives_and_math_markers(deffile):
    got = {g.name: g for g in groups.parse_def(str(deffile))}
    assert got["FF_sph"].brief == "Homogeneous R spheres"
    assert got["FF_top"].brief == "all form factors"
    assert got["FF_cyl"].brief == ""


def test_def_files_lists_only_def_files_sorted(tmp_path):
    for n in ("b.def", "a.DEF", "c.txt"):
        (tmp_path / n).write_text("")
    got = [os.path.basename(f) for f in groups.def_files([str(tmp_path), str(tmp_path / "missing")])]
    assert got == ["a.DEF", "b.def"]


def test_load_groups_collects_all_files(tmp_path):
    (tmp_path / "a.def").write_text(DEF)
    (tmp_path / "b.def").write_text(r"/** \defgroup EXTRA Extra stuff */")
    defs = groups.load_groups([str(tmp_path)])
    assert "EXTRA" in defs and "FF_sph" in defs
    assert len(defs.files) == 2


def test_load_groups_notices_a_changed_file(tmp_path):
    f = tmp_path / "a.def"
    f.write_text(r"/** \defgroup ONE One */")
    assert set(groups.load_groups([str(tmp_path)])) == {"ONE"}
    f.write_text(r"/** \defgroup TWO Two */")
    later = time.time() + 5
    os.utime(f, (later, later))  # the cache key contains the modification time
    assert set(groups.load_groups([str(tmp_path)])) == {"TWO"}


def test_no_def_files_gives_an_empty_mapping(tmp_path):
    assert len(groups.load_groups([str(tmp_path)])) == 0


# ---- menu tree ----------------------------------------------------------------------------------------------


def test_functions_are_filed_under_their_group(defs):
    top = groups.build_menu([entry("sphere", "FF_sph"), entry("cyl", "FF_cyl"), entry("sphere2", "FF_sph")], "ff", defs)
    labels = {c.label: c for c in top.children}
    assert set(labels) == {"Spheres", "Cylinders"}
    assert [p.name for p in labels["Spheres"].functions] == ["sphere", "sphere2"]
    assert top.count() == 3
    assert labels["Spheres"].brief == "Homogeneous R spheres"


def test_groups_are_ordered_as_in_the_def_files(defs):
    top = groups.build_menu([entry("c", "FF_cyl"), entry("s", "FF_sph")], "ff", defs)
    assert [c.label for c in top.children] == ["Spheres", "Cylinders"]


def test_only_the_requested_type_is_listed(defs):
    top = groups.build_menu([entry("a", "FF_sph"), entry("b", "SQ_top", type="sq")], "ff", defs)
    assert top.count() == 1


def test_a_tuple_of_types_is_accepted(defs):
    top = groups.build_menu([entry("a", "FF_sph"), entry("b", "SQ_top", type="sq")], ("ff", "sq"), defs)
    assert top.count() == 2


def test_special_groups_for_builtin_python_and_ungrouped(defs):
    entries = [
        entry("b", "", builtin=True),
        entry("p", "", collection="python"),
        entry("u", "NO_SUCH_GROUP"),
    ]
    top = groups.build_menu(entries, "ff", defs)
    assert [c.label for c in top.children] == ["built-in (libsasfit)", "Python plugins", "(not in a group)"]


def test_a_function_in_a_group_of_another_menu_is_set_aside(defs):
    top = groups.build_menu([entry("odd", "SQ_top"), entry("ok", "FF_sph")], "ff", defs)
    labels = [c.label for c in top.children]
    assert "Spheres" in labels and "(filed in a group of another menu)" in labels


def test_without_definitions_the_list_is_flat():
    top = groups.build_menu([entry("a"), entry("b")], "ff", groups.GroupDefs())
    assert top.children == [] and [p.name for p in top.functions] == ["a", "b"]


def test_a_cyclic_group_chain_terminates():
    d = groups.GroupDefs()
    d["A"] = groups.GroupDef("A", "A", parent="B")
    d["B"] = groups.GroupDef("B", "B", parent="A")
    top = groups.build_menu([entry("x", "A")], "ff", d)
    assert top.count() == 1


def test_iter_functions_yields_the_menu_path(defs):
    top = groups.build_menu([entry("s", "FF_sph")], "ff", defs)
    assert [(path, p.name) for path, p in groups.iter_functions(top)] == [(("Spheres",), "s")]


def test_tree_lines_show_counts_and_functions(defs):
    top = groups.build_menu([entry("s", "FF_sph"), entry("c", "FF_cyl")], "ff", defs)
    lines = groups.tree_lines(top)
    assert "Spheres  [1]" in lines
    assert any(line.strip().startswith("- S") and "(s)" in line for line in lines)
    assert not any("(s)" in line for line in groups.tree_lines(top, functions=False))
