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


def test_independent_background_fit():
    q = np.linspace(0.5, 3, 200)
    I = 50 * q ** -4 + 0.7
    r = A.fit_power_background(q, I, 0.01 * I)
    assert r.b == pytest.approx(4, abs=1e-3) and r.bg == pytest.approx(0.7, rel=1e-3)
    r = A.fit_power_background(q, I, 0.01 * I, exponent=4)
    assert r.db == 0 and r.bg == pytest.approx(0.7, rel=1e-3) and np.allclose(r.model(q), I, rtol=1e-3)


@pytest.mark.skipif(sys.platform.startswith("win"), reason="fake programs are shell scripts")
@pytest.mark.parametrize("printed_c", [0.7, -0.7])
def test_datporod_sign_is_resolved(tmp_path, no_cache, printed_c):
    """The manual leaves open on which side C is subtracted: whichever reading fits the points is taken."""
    root = _fake_install(tmp_path)
    (root / "bin" / "datporod").write_text(
        f'#!/bin/sh\necho "   0.500000E+02   0.1E+00    4.00000   0.1E-01  {printed_c}  0.01   1.00   $1"\n')
    (root / "bin" / "datporod").chmod(0o755)
    q = np.linspace(0.5, 3, 100)
    I = 50 * q ** -4 + 0.7
    r = A.datporod_exponent((q, I, 0.01 * I), 1, 100, atsas_dir=root, cwd=tmp_path / "w")
    assert r.bg == pytest.approx(0.7) and r.c == printed_c and "--constant" in r.run.cmd
    assert "--first=1" in r.run.cmd and "--last=100" in r.run.cmd


# ----------------------------------------------------------------------- programs that return no result
def _fake(b, name, body):
    p = b / name
    p.write_text("#!/bin/sh\n" + body + "\n")
    p.chmod(0o755)


posix_only = pytest.mark.skipif(sys.platform.startswith("win"), reason="fake programs are shell scripts")


@posix_only
@pytest.mark.parametrize("tool, body, call, match", [
    ("datgnom", "true", lambda d, r: A.datgnom(d, rg=3.9, atsas_dir=r), "wrote no input.out"),
    ("datgnom", 'touch "$3"', lambda d, r: A.datgnom(d, rg=3.9, atsas_dir=r), "empty input.out"),
    ("gnom", 'echo "ERROR: bad input" > "$4"', lambda d, r: A.gnom(d, dmax=10, atsas_dir=r), "no p.r. table"),
    ("autorg", 'echo "nan nan nan nan 0 0 0 0 $1"', lambda d, r: A.autorg(d, atsas_dir=r), "no valid Rg"),
    ("autorg", 'echo "no Guinier region" >&2; exit 1', lambda d, r: A.autorg(d, atsas_dir=r), "no Guinier"),
    ("crysol", 'printf "0.1 1\\n0.2 2\\n0.3 3\\n" > m.int', lambda d, r: A.crysol(__file__, d, atsas_dir=r),
     "no .fit file"),
    ("crysol", "true", lambda d, r: A.crysol(__file__, atsas_dir=r), "no .int/.abs"),
    ("dammif", 'echo "licence expired" >&2; exit 2', lambda d, r: A.dammif(__file__, nruns=2, atsas_dir=r),
     "no bead model"),
    ("datporod", 'echo "nan nan nan nan nan nan nan $1"',
     lambda d, r: A.datporod_exponent(d, 300, 400, atsas_dir=r), "no valid fit"),
])
def test_no_result_raises_atsas_error(tmp_path, no_cache, tool, body, call, match):
    root = _fake_install(tmp_path)
    _fake(root / "bin", tool, body)
    with pytest.raises(A.AtsasError, match=match):
        call(A.sphere_curve(), root)


@posix_only
def test_program_that_cannot_start(tmp_path, no_cache):
    root = _fake_install(tmp_path)
    (root / "bin" / "datgnom").write_text("not a program")
    (root / "bin" / "datgnom").chmod(0o644)
    with pytest.raises(A.AtsasError, match="could not be started"):
        A.datgnom(A.sphere_curve(), rg=3.9, atsas_dir=root)


@posix_only
def test_dammif_keeps_the_good_runs(tmp_path, no_cache):
    root = _fake_install(tmp_path)
    _fake(root / "bin", "dammif", 'case "$4" in *_02) echo boom >&2; exit 3;; esac; p=${4#--prefix=}; '
                                  'echo ATOM > "$p-1.pdb"')
    d = A.dammif(__file__, nruns=3, atsas_dir=root, cwd=tmp_path / "w")
    assert len(d.models) == 2 and len(d.errors) == 1 and "2 of 3" in d.warnings[0]


@posix_only
def test_gnom_warnings(tmp_path, no_cache):
    out = GNOM_OUT.replace("Total Estimate    : 0.92", "Total Estimate    : 0.21")
    out = out.replace("0.5000E+01  0.1000E+01  0.1000E-01", "0.5000E+01  -0.1000E+01  0.1000E-01")
    f = tmp_path / "x.out"
    f.write_text(out)
    w = A.parse_gnom_out(f).warnings
    assert any("total estimate" in x for x in w) and any("negative" in x for x in w)


# ----------------------------------------------------------------------- bead models, envelopes, SASHA
def test_bead_model_pdb_and_cif(tmp_path):
    from pysasfit.atsas import models as M
    pdb = tmp_path / "m-1.pdb"
    pdb.write_text(" DAM packing radius ..................................... : 1.800\n"
                   " Radius of gyration ..................................... : 2.0\n"
                   + "".join(f"ATOM  {i:5d}  CA  ASP    1    {x:8.3f}{y:8.3f}{z:8.3f}  1.00 20.00\n"
                             for i, (x, y, z) in enumerate([(0, 0, 0), (3.6, 0, 0), (0, 3.6, 0)], 1)))
    m = M.read_bead_model(pdb)
    assert m.n == 3 and m.radius == 1.8 and m.radius_from == "header"
    assert m.header["radius of gyration"] == 2.0
    cif = tmp_path / "m-1.cif"
    cif.write_text("data_x\nloop_\n_atom_site.group_PDB\n_atom_site.id\n_atom_site.Cartn_x\n"
                   "_atom_site.Cartn_y\n_atom_site.Cartn_z\n"
                   "ATOM 1 0.0 0.0 0.0\nATOM 2 3.6 0.0 0.0\nATOM 3 0.0 3.6 0.0\n#\n")
    c = M.read_bead_model(cif)
    assert c.n == 3 and c.radius == pytest.approx(1.8) and c.radius_from == "nearest neighbours"
    back = M.read_bead_model(M.write_bead_pdb(c, tmp_path / "back.pdb"))
    assert np.allclose(back.xyz, c.xyz) and back.radius == pytest.approx(1.8, abs=1e-3)
    v, f = M.beads_mesh(m)
    assert f.max() < len(v) and len(v) == 3 * len(M.sphere_mesh()[0])
    with pytest.raises(A.AtsasError, match="no atoms"):
        (tmp_path / "e.pdb").write_text("REMARK nothing\n")
        M.read_bead_model(tmp_path / "e.pdb")


def _ellipsoid_coef(a=30.0, b=20.0, c=15.0, lmax=10):
    from pysasfit.atsas import models as M
    return M.flm_of_shape(lambda T, P: 1 / np.sqrt((np.sin(T) * np.cos(P) / a) ** 2 + (np.sin(T) * np.sin(P) / b) ** 2
                                                   + (np.cos(T) / c) ** 2), lmax)


def test_envelope_is_drawn_only_when_verified(tmp_path):
    """The .flm layout is not documented: an interpretation counts only if V and Rg match the file."""
    from pysasfit.atsas import models as M
    a, b, c = 30.0, 20.0, 15.0
    coef = _ellipsoid_coef(a, b, c)
    V, Rg = 4 / 3 * np.pi * a * b * c, np.sqrt((a * a + b * b + c * c) / 5)
    e = M.read_flm(M.write_flm_reference(tmp_path / "ok.flm", coef, Rg, V))
    assert e.verified and e.rg == pytest.approx(Rg, rel=2e-3) and e.volume == pytest.approx(V, rel=2e-3)
    v, _f = M.envelope_mesh(e)
    assert np.abs(v).max(axis=0) == pytest.approx([a, b, c], rel=0.01)
    assert not M.read_flm(M.write_flm_reference(tmp_path / "x.flm", coef, 1.3 * Rg, V)).verified
    (tmp_path / "n.flm").write_text("\n".join(f"{x.real:.8e} {x.imag:.8e}" for _k, x in sorted(
        coef.items(), key=lambda kv: (kv[0][0], kv[0][1]))) + "\n")
    assert not M.read_flm(tmp_path / "n.flm").verified          # readable, but nothing to check against


def _fake_sasha(root, script):
    leg = root / "Legacy"
    leg.mkdir(exist_ok=True)
    p = leg / "sasha20"
    p.write_text("#!/bin/sh\n" + script)
    p.chmod(0o755)


SASHA_DIALOG = """\
for prompt in "Working mode [ User/Developer ] ........ < User >:" "MONOMER SAS data file name ............. < .dat >:" \\
  "Angular units in the input file : 4*pi*sin(theta)/lambda [1/nm]  (2) .... < 1 >:" \\
  "DIMER SAS data file name ............... < .dat >:" "OUTPUT file name ....................... < .flm >:" \\
  "Smax for the MONOMER fitting region .... < 0.5000 >:" "Enter initial approximation file ...... < .flm >:" \\
  "Symmetry [P1/P2/P222/Pmmm/P3] .......... < P1 >:" "Max. harmonic order for the shape ...... < 4 >:" \\
  "Specify the shape anysometry ........... < O >:" "Enter the molecular weight in kDa .....................:" \\
  "Enter the monomer Rg in angstroms .....................:"; do
  echo "$prompt"; read ans; echo "$ans" >> answers.txt
done
"""


@posix_only
def test_sasha_dialog(tmp_path, no_cache):
    from pysasfit.atsas import models as M
    root = _fake_install(tmp_path)
    ref = tmp_path / "ref.flm"
    M.write_flm_reference(ref, _ellipsoid_coef(), np.sqrt((30.0 ** 2 + 20.0 ** 2 + 15.0 ** 2) / 5),
                          4 / 3 * np.pi * 30 * 20 * 15)
    _fake_sasha(root, SASHA_DIALOG + f'cp "{ref}" sas.flm\nprintf "0.1 10 9.9\\n0.2 8 8.1\\n" > sas.fim\n')
    r = M.sasha(A.sphere_curve(), smax=1.5, mw_kda=14.3, rg_angstrom=14.0, symmetry="p2", lmax=5,
                atsas_dir=root, cwd=tmp_path / "run")
    answers = (tmp_path / "run" / "answers.txt").read_text().splitlines()
    assert answers == ["", "mon.dat", "2", "", "sas", "1.5", "", "P2", "5", "O", "14.3", "14"]
    assert r.envelope.verified and not r.warnings and len(r.fit_q) == 2
    assert (tmp_path / "run" / "sasha_dialog.log").is_file()


@posix_only
def test_sasha_unexpected_dialog_and_no_output(tmp_path, no_cache):
    from pysasfit.atsas import models as M
    root = _fake_install(tmp_path)
    _fake_sasha(root, 'echo "Some other question ..... < 1 >:"; read a\necho "OUTPUT file name ... :"; read b\n')
    with pytest.raises(A.AtsasError, match="wrote no envelope"):
        M.sasha(A.sphere_curve(), smax=1.5, mw_kda=14, rg_angstrom=14, atsas_dir=root, cwd=tmp_path / "r1")
    _fake_sasha(root, 'echo "Something else:"; read a\necho "1 0 1 0" > sas.flm\n')
    r = M.sasha(A.sphere_curve(), smax=1.5, mw_kda=14, rg_angstrom=14, atsas_dir=root, cwd=tmp_path / "r2")
    assert any("dialog differed" in w for w in r.warnings)


def test_sasha_missing(tmp_path, no_cache):
    from pysasfit.atsas import models as M
    root = _fake_install(tmp_path)
    with pytest.raises(A.AtsasError, match="Legacy"):
        M.sasha(A.sphere_curve(), smax=1.5, mw_kda=14, rg_angstrom=14, atsas_dir=root)


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
