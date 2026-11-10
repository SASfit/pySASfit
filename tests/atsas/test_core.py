"""pyATsas core: finding ATSAS, staging inputs, parsing GNOM output, the call chain (with a fake ATSAS), and --
when ATSAS is installed on this computer -- the self test against the real programs."""
import os
import sys
import textwrap
from pathlib import Path

import numpy as np
import pytest

from pysasfit.atsas import core as A

GNOM_OUT = textwrap.dedent("""\
               ####    G N O M   ---   Version 5.0 (r10552)          ####
       Number of points omitted at the beginning:           5
       Real space range                 :  0.0000 to  10.000
       Real space Rg     : 0.3872E+01 +- 0.1000E-01
       Real space I(0)   : 0.1000E+04 +- 0.2000E+00
       Total Estimate    : 0.92

     ####      Experimental Data and Fit                            ####

          S          J EXP       ERROR       J REG       I REG
      0.0000E+00                                     0.1000E+04
      0.1000E+00  0.9900E+03  0.1000E+01  0.9890E+03  0.9890E+03
      0.2000E+00  0.9500E+03  0.1000E+01  0.9560E+03  0.9560E+03

     ####      Real Space Data                                      ####

               R          P(R)      ERROR
      0.0000E+00  0.0000E+00  0.0000E+00
      0.5000E+01  0.1000E+01  0.1000E-01
      0.1000E+02  0.0000E+00  0.0000E+00
    """)


@pytest.fixture
def no_cache(monkeypatch):
    monkeypatch.setattr(A, "_ATSAS_BIN", None)
    for v in ("PYATSAS_DIR", "ATSAS"):
        monkeypatch.delenv(v, raising=False)
    yield
    A._ATSAS_BIN = None


def _fake_install(root: Path, version="4.1.4"):
    b = root / f"ATSAS-{version}" / "bin"
    b.mkdir(parents=True)
    for exe in ("autorg", "datgnom", "gnom", "primusqt"):
        (b / (exe + A.EXE_SUFFIX)).write_text("")
    return b.parent


def test_parse_gnom_out(tmp_path):
    f = tmp_path / "x.out"
    f.write_text(GNOM_OUT)
    r = A.parse_gnom_out(f)
    assert r.dmax == 10.0 and r.rg_real == pytest.approx(3.872) and r.i0_real == 1000
    assert r.total_estimate == 0.92
    assert np.allclose(r.r, [0, 5, 10]) and np.allclose(r.pr, [0, 1, 0])
    assert np.allclose(r.q, [0.1, 0.2]) and np.allclose(r.i_reg, [989, 956])     # 2-column rows skipped


def test_versions_sort_numerically():
    ps = [Path("ATSAS-4.9.2"), Path("ATSAS-4.10.0"), Path("ATSAS-3.2.1")]
    assert [p.name for p in A._newest(ps)] == ["ATSAS-4.10.0", "ATSAS-4.9.2", "ATSAS-3.2.1"]


def test_find_atsas_from_environment(tmp_path, monkeypatch, no_cache):
    root = _fake_install(tmp_path)
    monkeypatch.setenv("PYATSAS_DIR", str(root))
    assert A.find_atsas() == root / "bin"
    assert A.atsas_root() == root
    assert A.license_file() is None
    (root / "atsas.lic").write_text("x")
    assert A.license_file() == root / "atsas.lic"
    assert A.program_path("primusqt") is not None and A.program_path("chromixs") is None


def test_find_atsas_newest_in_home(tmp_path, monkeypatch, no_cache):
    monkeypatch.setattr(A.Path, "home", classmethod(lambda cls: tmp_path))
    monkeypatch.setattr(A.shutil, "which", lambda _n: None)
    monkeypatch.setattr(A, "_win_registry_atsas", lambda: [])
    if A.IS_WIN:      # do not find a real installation in Program Files
        for v in ("ProgramW6432", "ProgramFiles", "ProgramFiles(x86)", "LOCALAPPDATA"):
            monkeypatch.setenv(v, str(tmp_path / "none"))
    _fake_install(tmp_path, "4.9.2")
    new = _fake_install(tmp_path, "4.10.0")
    assert A.find_atsas() == new / "bin"


def test_explicit_dir_root_or_bin(tmp_path, no_cache):
    root = _fake_install(tmp_path)
    assert A.find_atsas(root) == root / "bin"
    assert A.find_atsas(root / "bin") == root / "bin"
    with pytest.raises(A.AtsasError):
        A.find_atsas(tmp_path / "nothing")


def test_inputs_are_staged_under_short_names(tmp_path):
    src = tmp_path / "my data with spaces ä.dat"
    A.write_dat(src, *A.sphere_curve()[:3])
    work = tmp_path / "run" / "sub"          # not yet existing
    rel = A._as_file(src, work)
    assert not rel.is_absolute() and " " not in str(rel) and (work / rel).is_file()
    rel2 = A._as_file(A.sphere_curve(), work)
    assert str(rel2) == "input.dat" and len(A.read_dat(work / rel2)) == 400


def test_icon_files(tmp_path, no_cache):
    root = _fake_install(tmp_path)
    for n in (16, 64, 256):
        d = root / "share" / "icons" / "hicolor" / f"{n}x{n}" / "apps"
        d.mkdir(parents=True)
        (d / "atsas.png").write_bytes(b"png")
    assert [s for s, _p in A.icon_files("atsas", root)] == [16, 64, 256]
    assert A.icon_files("nothing", root) == []


@pytest.mark.skipif(sys.platform.startswith("win"), reason="fake programs are shell scripts")
def test_call_chain_with_fake_programs(tmp_path, no_cache):
    root = _fake_install(tmp_path)
    b = root / "bin"
    (b / "autorg").write_text("#!/bin/sh\n[ -f \"$1\" ] || exit 1\necho \"3.870 0.010 1000.0 2.0 5 40 0.95 0.0 $1\"\n")
    (tmp_path / "x.out").write_text(GNOM_OUT)
    for exe, pos in (("datgnom", 3), ("gnom", 4)):    # datgnom FILE -o OUT ...; gnom FILE --rmax= -o OUT
        (b / exe).write_text(f'#!/bin/sh\n[ "$ATSAS" = "{root}" ] || exit 2\ncp "{tmp_path / "x.out"}" "${pos}"\n')
    for f in b.iterdir():
        f.chmod(0o755)
    g = A.autorg(A.sphere_curve(), atsas_dir=root, cwd=tmp_path / "w1")
    assert g.rg == pytest.approx(3.87) and (g.first, g.last) == (5, 40)
    p = A.datgnom(A.sphere_curve(), rg=g.rg, atsas_dir=root, cwd=tmp_path / "w2")
    assert p.dmax == 10 and p.run.cmd[1] == "input.dat"
    p2 = A.gnom(A.sphere_curve(), dmax=10, atsas_dir=root, cwd=tmp_path / "w3")
    assert "--force-zero-rmin=Y" in p2.run.cmd and "--rmax=10" in p2.run.cmd


# ----------------------------------------------------------------------- the real ATSAS, when installed
def _real_atsas():
    try:
        b = A.find_atsas(refresh=True)
    except A.AtsasError:
        return None
    finally:
        A._ATSAS_BIN = None
    return b if os.path.getsize(b / ("autorg" + A.EXE_SUFFIX)) > 0 else None


needs_atsas = pytest.mark.skipif(_real_atsas() is None, reason="ATSAS is not installed")


@needs_atsas
@pytest.mark.atsas
def test_real_atsas_sphere():
    """AUTORG, DATGNOM and GNOM of the real installation on a sphere of R = 5 nm (Rg = 3.873, Dmax = 10)."""
    assert A.selftest()
