"""The results store of pySASscout: fits and searches are kept on disk and reused (Python reference functions)."""
import json
import os
import shutil

import numpy as np
import pytest

from pysasfit import scout
from pysasfit.scout import Component, reference
from pysasfit.scout.store import ResultStore, data_id, default_dir
from pysasfit.studio.model import ModelFunction, ScatteringContribution

Q = np.geomspace(0.005, 0.4, 60)
FAST = dict(reference=True, starts=3, max_starts=8, n_x=12, n_x_final=48)


@pytest.fixture(autouse=True)
def _reference_functions():
    reference.register()
    yield
    reference.unregister()


@pytest.fixture(scope="module")
def data():
    reference.register()
    c = ScatteringContribution("ff_sphere", {"R": 60, "eta": 1}, size_distribution="sd_LogNorm",
                               sd_params={"N": 1e-12, "s": 0.2, "p": 1, "mu": 60}, distributed="R")  # fmt: skip
    I = ModelFunction([c], background=0.002)(Q)
    s = 0.03 * I + 0.0015 * I[-1]
    return Q, I + np.random.default_rng(0).normal(0, s), s


def test_default_folder_is_next_to_the_preferences_file():
    from pysasfit.core import settings as prefs

    assert default_dir() == os.path.join(os.path.dirname(prefs.settings_path()), "scout_store")
    assert ResultStore().root == os.path.abspath(default_dir())


def test_nothing_is_stored_without_a_store(data, tmp_path):
    r = scout.search(data, forms=["rod"], **FAST)
    assert r.store is None and not any(c.stored for c in r.candidates)


def test_second_search_reuses_the_fits(data, tmp_path):
    root = str(tmp_path / "store")
    first = scout.search(data, forms=["sphere", "rod"], store=root, name="spheres", path="C:/x/spheres.dat", **FAST)
    assert not any(c.stored for c in first.candidates) and first.store["folder"] == os.path.abspath(root)
    did = data_id(*data)
    assert first.store["data_id"] == did and len(did) == 16
    st = ResultStore(root)
    assert os.path.isfile(os.path.join(root, "store.json"))
    assert len(os.listdir(os.path.join(root, "data", did, "fits"))) == 3
    assert np.allclose(np.loadtxt(os.path.join(root, "data", did, "data.dat")), np.column_stack(data), rtol=1e-12)
    again = scout.search(data, forms=["sphere", "rod"], store=root, **FAST)
    assert all(c.stored for c in again.candidates) and again.seconds < first.seconds / 5
    assert any("3 of 3 fits were taken from the results store" in n for n in again.notes)
    for a, b in zip(first.candidates, again.candidates):
        assert a.label == b.label and a.chi2 == b.chi2 and a.parameters == b.parameters and a.warnings == b.warnings
        assert a.weight == pytest.approx(b.weight) and np.array_equal(a.I_fit, b.I_fit)
    assert "(stored)" in again.table() and "(stored)" not in first.table()
    assert st.summary()["datasets"] == 1 and st.summary()["fits"] == 3 and st.summary()["runs"] == 2
    info = st.datasets()[0]
    assert info["name"] == "spheres" and info["file"] == "C:/x/spheres.dat" and info["points"] == Q.size
    assert "1 curve, 3 fits, 2 searches" in st.summary_text()


def test_adding_a_model_fits_only_the_new_one(data, tmp_path):
    root = str(tmp_path / "store")
    scout.search(data, forms=["rod"], store=root, **FAST)
    r = scout.search(data, forms=["rod", "disc"], store=root, **FAST)
    stored = {c.label: c.stored for c in r.candidates}
    assert stored == {"thin rod": True, "thin disc": False}
    assert not any(c.stored for c in scout.search(data, forms=["rod"], store=root, reuse=False, **FAST).candidates)


def test_key_depends_on_what_the_fit_depends_on(data, tmp_path):
    o = scout.ScoutOptions(**FAST)
    comp = (Component("sphere", "lognormal"),)
    key = ResultStore.fit_key(comp, o)
    assert key == ResultStore.fit_key(comp, scout.ScoutOptions(**FAST)) and len(key) == 24
    others = [
        ResultStore.fit_key((Component("sphere"),), o),
        ResultStore.fit_key(comp, scout.ScoutOptions(**{**FAST, "seed": 2})),
        ResultStore.fit_key(comp, scout.ScoutOptions(**{**FAST, "starts": 4})),
        ResultStore.fit_key(comp, scout.ScoutOptions(**{**FAST, "background": False})),
        ResultStore.fit_key(comp, scout.ScoutOptions(**{**FAST, "reference": False})),
        ResultStore.fit_key(comp, o, error_scale=2.0),
        ResultStore.fit_key(comp, o, start={"0.sd.mu": 50.0}),
    ]
    assert len(set(others + [key])) == len(others) + 1
    # options that do not change a fit do not change the key
    same = scout.ScoutOptions(**{**FAST, "criterion": "bic", "max_components": 3, "jobs": 4, "store": "x"})
    assert ResultStore.fit_key(comp, same) == key
    # another q range is another curve
    root = str(tmp_path / "store")
    a = scout.search(data, forms=["rod"], store=root, **FAST)
    b = scout.search(data, forms=["rod"], store=root, q_range=(0.01, 0.3), **FAST)
    assert a.store["data_id"] != b.store["data_id"] and not b.candidates[0].stored


def test_stored_search_can_be_loaded_again(data, tmp_path):
    root = str(tmp_path / "store")
    r = scout.search(data, forms=["sphere", "rod"], store=root, name="spheres", **FAST)
    st = ResultStore(root)
    did = r.store["data_id"]
    assert st.runs(did) == [r.store["run"]]
    back = st.load_run(did)
    assert [c.label for c in back.candidates] == [c.label for c in r.candidates]
    assert back.best.parameters == r.best.parameters and back.best.chi2 == r.best.chi2
    assert np.allclose(back.best.I_fit, r.best.I_fit) and back.features.rg == pytest.approx(r.features.rg)
    assert back.options.seed == 1 and back.data["name"] == "spheres" and back.notes == r.notes
    assert back.text() == r.text().replace(f"in {r.seconds:.1f} s", f"in {back.seconds:.1f} s")
    assert np.allclose(back.model(0)(back.best.q), back.best.I_fit, rtol=1e-6)      # the model can be rebuilt
    assert np.allclose(back._data[1], data[1])
    assert st.load_run("0" * 16) is None and st.load_run(did, "nonsense") is None


def test_store_can_be_moved_and_merged(data, tmp_path):
    a, b = str(tmp_path / "a"), str(tmp_path / "moved" / "b")
    scout.search(data, forms=["rod"], store=a, **FAST)
    shutil.copytree(a, b)
    shutil.rmtree(a)
    r = scout.search(data, forms=["rod"], store=b, **FAST)
    assert r.candidates[0].stored and ResultStore(b).load_run(r.store["data_id"]) is not None


def test_stop_and_time_limit_still_take_stored_fits(data, tmp_path):
    root = str(tmp_path / "store")
    scout.search(data, forms=["rod"], store=root, **FAST)
    r = scout.search(data, forms=["rod", "disc"], store=root, time_limit=0.0, **FAST)
    by = {c.label: c for c in r.candidates}
    assert by["thin rod"].stored and by["thin rod"].success and not by["thin disc"].success


def test_failed_fits_are_not_stored_and_runs_are_limited(data, tmp_path):
    from pysasfit.scout import FormSpec, Par, candidates

    root = str(tmp_path / "store")
    scout.register_form(FormSpec("ghost", "ff_no_such_function", (Par("R"),), label="ghost"))
    try:
        for _ in range(4):
            r = scout.search(data, forms=["rod", "ghost"], store=root, keep_runs=2, **FAST)
    finally:
        del candidates.FORMS["ghost"]
    st = ResultStore(root)
    assert st.summary()["fits"] == 1 and len(st.runs(r.store["data_id"])) == 2
    assert not r.candidates[-1].success and not r.candidates[-1].stored


def test_prune(data, tmp_path):
    root = str(tmp_path / "store")
    r = scout.search(data, forms=["rod"], store=root, **FAST)
    scout.search(data, forms=["rod"], store=root, q_range=(0.01, 0.3), **FAST)
    st = ResultStore(root)
    assert st.summary()["datasets"] == 2 and st.prune() == 0 and st.prune(older_than_days=1) == 0
    assert st.prune(did=r.store["data_id"]) == 1 and st.summary()["datasets"] == 1
    assert st.prune(older_than_days=-1) == 1 and st.summary()["datasets"] == 0 and st.summary()["bytes"] == 0


def test_unusable_store_does_not_stop_the_search(data, tmp_path):
    blocker = tmp_path / "file"
    blocker.write_text("not a folder")
    r = scout.search(data, forms=["rod"], store=str(blocker / "store"), **FAST)
    assert r.best is not None and r.store is None and any("results store not usable" in n for n in r.features.notes)


def test_worker_processes_share_the_store(data, tmp_path):
    root = str(tmp_path / "store")
    scout.search(data, forms=["sphere", "rod"], store=root, jobs=2, **FAST)
    assert ResultStore(root).summary()["fits"] == 3
    assert all(c.stored for c in scout.search(data, forms=["sphere", "rod"], store=root, **FAST).candidates)
