"""Parameter descriptions of the Python ports: domain check, defaults, suggestion, header round trip."""

from __future__ import annotations

import ctypes
import importlib.util
import math
import os
import sys

import numpy as np
import pytest

from pysasfit.studio import headers, ports
from pysasfit.studio.ports import params as P

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
NAMES = ports.names()


@pytest.mark.parametrize("name", NAMES)
def test_defaults_are_valid_and_finite(name):
    spec = ports.get(name)
    p = spec.defaults()
    assert spec.check(p) == []
    v = np.asarray(spec(np.array([0.01, 0.1, 1.0]), p), float)
    assert np.all(np.isfinite(v))


def test_c_like_messages():
    s = ports.get("ff_sphere")
    p = s.defaults()
    p[0] = -3
    assert s.check(p) == ["R(-3) < 0"]
    with pytest.raises(ValueError, match=r"R\(-3\) < 0"):
        s.validate(p)
    b = ports.get("ff_blob__dendrimer_")
    p = b.defaults()
    p[1] = 0.2
    assert b.check(p) == ["nu(0.2) <= 0.333333"]
    g = ports.get("ff_gauspoly")
    p = g.defaults()
    p[1] = 0.5 * p[2]
    assert "Mw < Mn" in g.check(p)


def test_registered_functions_validate():
    from pysasfit.studio.functions import SASfitFunction

    ports.register_aliases()
    f = SASfitFunction("ff_py_sphere")
    assert f.info.parameters[0].default == 10.0 and f.info.parameters[0].bounds == (
        0.0,
        math.inf,
    )
    with pytest.raises(ValueError, match="R"):
        f(np.array([0.1]), -1.0, 0, 0, 1.0)


def test_generated_table_is_current():
    sys.path.insert(0, os.path.join(ROOT, "tools"))
    import gen_port_params

    from pysasfit.studio.ports.param_table import TABLE

    built = gen_port_params.build(os.path.join(HERE, "port_reference", "sasfit_src"))
    assert set(built) == set(TABLE)
    for name, entry in built.items():
        assert [tuple(x) for x in entry["params"]] == [
            tuple(x) for x in TABLE[name]["params"]
        ], name


def test_suggest_scales_lengths_and_intensity():
    s = ports.get("ff_sphere")
    q = np.geomspace(0.002, 0.2, 100)
    data = 500.0 * s(q, [20.0, 0, 0, 1.0])
    p = s.suggest(q)
    assert p[0] == pytest.approx(2.0 / math.sqrt(0.002 * 0.2))
    p2 = s.suggest(q, data)
    low = q <= np.quantile(q, 0.1)
    assert np.median(s(q[low], p2)) == pytest.approx(np.median(data[low]), rel=1e-12)


def test_kind_guess():
    assert P.guess_kind("R_core") == "length"
    assert P.guess_kind("eta_shell") == "sld"
    assert P.guess_kind("psi") == "angle"
    assert P.guess_kind("dummy") == "dummy"
    assert P.guess_kind("I0") == "intensity"


def test_header_defaults_round_trip(tmp_path):
    sys.path.insert(0, os.path.join(ROOT, "tools"))
    import write_header_defaults as W

    text = (
        "/* ################ start ff_sphere ################ */\r\n/**\r\n * \\defgroup ff_sphere sphere\r\n"
        " * \\ingroup ff_plugins\r\n *\r\n * \\brief sphere\r\n *\r\n * \\par Required parameters:\r\n"
        ' *      <table border="0"><tr>\r\n *       <td>\\b R</td>\r\n *       <td>radius</td>\r\n'
        " *      </tr><tr>\r\n *       <td>\\b dummy</td>\r\n *       <td></td>\r\n *      </tr><tr>\r\n"
        " *       <td>\\b dummy</td>\r\n *       <td></td>\r\n *      </tr><tr>\r\n *       <td>\\b eta</td>\r\n"
        " *       <td>contrast</td>\r\n *      </tr></table>\r\n */\r\n"
        "/* ################ stop ff_sphere ################ */\r\n"
    )
    prm = ports.get("ff_sphere").params
    new, status, notes = W.update_header(text, "ff_sphere", prm, "\r\n")
    assert status == "written" and not notes and "\n" not in new.replace("\r\n", "")
    assert "<td>(default=10)</td>" in new and "<td>[0;+inf)</td>" in new
    again, status2, _ = W.update_header(new, "ff_sphere", prm, "\r\n")
    assert status2 == "unchanged" and again == new
    path = tmp_path / "sasfit_x.h"
    path.write_bytes(new.encode())
    (info,) = headers.parse_header(str(path))
    assert [p.name for p in info.parameters] == ["R", "dummy", "dummy", "eta"]
    assert info.parameters[0].default == 10.0 and info.parameters[0].bounds == (
        0.0,
        math.inf,
    )
    assert (
        info.parameters[3].default == prm[3].default
        and info.parameters[3].description == "contrast"
    )


# ------------------------------------------------------------------------------- the C checks agree
@pytest.fixture(scope="module")
def portref(tmp_path_factory):
    spec = importlib.util.spec_from_file_location(
        "port_reference_build_params", os.path.join(HERE, "port_reference", "build.py")
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    lib = mod.build(str(tmp_path_factory.mktemp("portref_params")))
    if lib is None:
        pytest.skip("no C compiler or GSL for the C reference of the ports")
    return ctypes.CDLL(lib)


class _Param(ctypes.Structure):
    _fields_ = [
        ("p", ctypes.c_double * 50),
        ("errStatus", ctypes.c_int),
        ("errStr", ctypes.c_char * 256),
        (
            "kernelSelector",
            ctypes.c_int,
        ),  # port_reference shim: sasfit_param.kernelSelector
    ]


@pytest.mark.parametrize("name", NAMES)
def test_bounds_reject_what_c_rejects(portref, name):
    spec = ports.get(name)
    try:
        f = getattr(portref, "sasfit_" + name)
    except AttributeError:
        pytest.skip("not in the C reference")
    f.restype = ctypes.c_double
    f.argtypes = [ctypes.c_double, ctypes.POINTER(_Param)]
    base = spec.defaults()
    for i, prm in enumerate(spec.params):
        if prm.kind == "dummy":
            continue
        for bound, is_open, step in (
            (prm.lower, prm.lower_open, -1.0),
            (prm.upper, prm.upper_open, 1.0),
        ):
            if not math.isfinite(bound):
                continue
            for value, valid in (
                (bound + step * (1e-3 + 1e-3 * abs(bound)), False),
                (bound, not is_open),
            ):
                p = base.copy()
                p[i] = value
                s = _Param()
                s.p[:] = [float(v) for v in p]
                f(0.05, ctypes.byref(s))
                c_ok = not s.errStatus
                assert (prm.violation(value) == "") == valid, (prm.name, value)
                assert (prm.violation(value) in spec.check(p)) != valid or valid
                if prm.name in ("sigma",):  # Python-side bound only
                    continue
                if any(not r.test(p) for r in spec.rules):
                    continue  # the value is on the bound but breaks a rule between parameters
                assert c_ok == valid or c_ok, (
                    prm.name,
                    value,
                    s.errStr,
                )  # C must not reject a valid value
