"""Helpers of "Copy Python code" (pysasfit.core.pycode)."""
import numpy as np
import pytest

from pysasfit.core import pycode


def test_literals_round_trip():
    for v in (1, 2.5, -0.0, 1e-300, "a'b\\c", None, True, (1,), [1, 2.0], {"a": (1, 2)}, np.float32(0.1),
              np.int64(3), np.array([[1.0, 2.0], [3.0, 4.0]]), float("inf")):
        txt = pycode.lit(v)
        got = eval(txt, {"np": np})
        if isinstance(v, np.ndarray):
            assert np.array_equal(got, v)
        elif isinstance(v, np.floating):
            assert got == float(v)
        else:
            assert got == v
    assert pycode.lit(float("nan")) == 'float("nan")'
    with pytest.raises(TypeError):
        pycode.lit(lambda x: x)


def test_call_and_script():
    c = pycode.call("f", pycode.Code("q"), 1.0, key="v", other=pycode.Code("x + 1"))
    assert c == "f(q, 1.0, key='v', other=x + 1)"
    long = pycode.call("result = solve", pycode.Code("q"), pycode.Code("I"), **{f"k{i}": i for i in range(30)})
    assert long.startswith("result = solve(\n    q, I,\n    k0=0,") and long.endswith("\n)")
    sc = pycode.Script("tool", "what")
    sc.imp("from os import sep", "from os import path", "import numpy as np")
    sc.add("x = 1")
    sc.add_plot("plt.plot([1], [x])")
    t = sc.text()
    assert "from os import path, sep" in t and t.index("import numpy as np") < t.index("from os import")
    assert "PLOT = True" in t and "plt.show()" in t
    ns = {}
    exec(t.replace("PLOT = True", "PLOT = False"), ns)
    assert ns["x"] == 1
    assert pycode.nondefault({"a": 1, "b": 2.0, "c": "x"}, {"a": 1, "b": 2, "c": "y"}) == {"c": "x"}
    assert pycode.nondefault({"seed": 0, "n": 5}, lambda seed=0, n=3: None) == {"n": 5}


def test_prefs_lines_only_changed_values(tmp_path, monkeypatch):
    from pysasfit.core import settings
    settings.reset()
    lines = pycode.prefs_lines("oz.solvers", apply="go()")
    assert "settings.set_value" not in "\n".join(lines) and lines[-1] == "go()"
    settings.set_value("oz.solvers.max_iterations", 1234)
    try:
        text = "\n".join(pycode.prefs_lines("oz.solvers"))
        assert "'oz.solvers.max_iterations': 1234" in text
    finally:
        settings.reset()
