"""pyATsas core: a thin wrapper around the ATSAS command-line programs (EMBL Hamburg / BIOSAXS GmbH).

ATSAS itself must be installed and licensed; this module locates the binaries, runs them via subprocess
and parses their output files into numpy arrays / dataclasses.  It has no Qt dependency (the
"pyATsas" panel of the launcher is :mod:`pysasfit.atsas.gui`).

    from pysasfit.atsas import autorg, datgnom, gnom, crysol, dammif
    g = autorg("lysozyme.dat")                 # Guinier Rg, I(0), range, quality
    p = datgnom("lysozyme.dat", rg=g.rg)       # p(r), Dmax, real-space Rg, total estimate

Covered programs
    autorg    Guinier Rg, I(0)                        (--format=ssv)
    datgnom   automatic p(r), Dmax                     (GNOM .out parsed)
    gnom      p(r) with a fixed Dmax, systems 0-6      (GNOM .out parsed)
    datporod  --exponent --constant: flat background of the high-q tail (with fit_power_background,
              an independent fit without ATSAS, as a check)
    crysol    scattering of an atomic model (+ fit); cryson for neutrons
    dammif    ab-initio bead models (n parallel runs)
    launch    start the interactive ATSAS programs (PRIMUS, CHROMIXS, PEAK, ...)

Platforms: Windows (native or MSYS2 Python), Linux, macOS.  The installation is found automatically
(:func:`find_atsas`); check it with ``python -m pysasfit.atsas --selftest``.

Design notes
  * ATSAS output formats change between 3.x and 4.x; parsers are deliberately tolerant
    (header-keyword based, not line-number based).
  * Units: the programs work in the reciprocal unit of the data; lengths come out in its inverse.
    pySASfit curves are in 1/nm, so Rg, Dmax and r are in nm.
  * Every call returns the raw stdout/stderr and the working directory, so the files can always be
    inspected afterwards.
"""
from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional, Sequence

import numpy as np

__all__ = [
    "AtsasError", "find_atsas", "run_tool", "version", "write_dat", "read_dat",
    "autorg", "datgnom", "gnom", "crysol", "dammif", "parse_gnom_out",
    "GuinierResult", "GnomResult", "CrysolResult", "DammifResult", "PorodResult", "datporod_exponent",
    "fit_power_background",
    "inv_A_to_inv_nm", "inv_nm_to_inv_A", "atsas_root", "license_file",
    "selftest", "sphere_curve", "GUI_PROGRAMS", "program_path", "launch", "icon_files", "app_icon_files",
]

_FLOAT = r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eEdD][-+]?\d+)?"


class AtsasError(RuntimeError):
    pass


# ----------------------------------------------------------------------------
# locating & running
# ----------------------------------------------------------------------------
_ATSAS_BIN: Optional[Path] = None
IS_WIN = sys.platform.startswith("win") or os.name == "nt"
EXE_SUFFIX = ".exe" if IS_WIN else ""


def _version_key(p: Path):
    """Sort ATSAS-4.10.0 after ATSAS-4.9.2 (plain string sort gets this wrong)."""
    nums = re.findall(r"\d+", p.as_posix().split("ATSAS")[-1])
    return tuple(int(n) for n in nums[:4])


def _newest(paths: Iterable[Path]) -> list[Path]:
    return sorted(set(paths), key=_version_key, reverse=True)


def _win_registry_atsas() -> list[Path]:
    """The Windows installer stores ATSAS as a (system or user) environment
    variable in the registry.  A Python started from an MSYS2 shell, an IDE or
    a shell opened before the installation may not have it in os.environ."""
    out: list[Path] = []
    try:
        import winreg
    except ImportError:
        return out
    keys = [(winreg.HKEY_CURRENT_USER, r"Environment"),
            (winreg.HKEY_LOCAL_MACHINE,
             r"SYSTEM\CurrentControlSet\Control\Session Manager\Environment")]
    for hive, sub in keys:
        try:
            with winreg.OpenKey(hive, sub) as k:
                val, _ = winreg.QueryValueEx(k, "ATSAS")
                out.append(Path(os.path.expandvars(val)))
        except OSError:
            pass
    return out


def _env_roots() -> list[Path]:
    roots: list[Path] = []
    for var in ("PYATSAS_DIR", "ATSAS"):
        if os.environ.get(var):
            roots.append(Path(os.environ[var]))
    if IS_WIN:
        roots += _win_registry_atsas()
    return roots


def _default_roots() -> list[Path]:
    """ATSAS-x.y.z folders in the default install locations, newest first."""
    roots: list[Path] = []
    home = Path.home()
    globs: list[Path] = [home]                    # 4.x default on Linux & macOS
    if IS_WIN:
        for var, dflt in (("ProgramW6432", r"C:\Program Files"),
                          ("ProgramFiles", r"C:\Program Files"),
                          ("ProgramFiles(x86)", r"C:\Program Files (x86)"),
                          ("LOCALAPPDATA", str(home / "AppData" / "Local"))):
            globs.append(Path(os.environ.get(var, dflt)))
        globs.append(Path(os.environ.get("LOCALAPPDATA",
                                         home / "AppData" / "Local")) / "Programs")
    else:
        globs += [Path("/opt"), Path("/usr/local"), Path("/Applications"),
                  home / "Applications", home / "opt", home / "software"]
    found: list[Path] = []
    for g in globs:
        try:
            found += [p for p in g.glob("ATSAS*") if p.is_dir()]
        except OSError:
            pass
    roots += _newest(found)
    return roots


def find_atsas(atsas_dir: Optional[str | os.PathLike] = None,
               refresh: bool = False) -> Path:
    """Return the ATSAS *bin* directory (cached after the first call).

    Search order
      1. `atsas_dir` argument (root or bin directory)
      2. $PYATSAS_DIR, $ATSAS (and on Windows the registry value of ATSAS)
      3. `autorg` on $PATH (Debian/RPM packages install to /usr/bin)
      4. newest ATSAS-x.y.z folder in the default locations:
         Windows  C:\\Program Files\\ATSAS-x.y.z, %LOCALAPPDATA%\\Programs
         Linux    ~/ATSAS-x.y.z, /opt, /usr/local
         macOS    ~/ATSAS-x.y.z, /Applications, ~/Applications
    A Python started from a macOS GUI does not read ~/.zprofile, so step 4
    matters there even if the installer configured the shell.
    """
    global _ATSAS_BIN
    if _ATSAS_BIN is not None and not refresh and atsas_dir is None:
        return _ATSAS_BIN
    if not refresh and atsas_dir is None:
        b = cached_atsas()                         # found by an earlier session: no search
        if b is not None:
            _ATSAS_BIN = b
            return b
    exe = "autorg" + EXE_SUFFIX
    cands: list[Path] = [Path(atsas_dir)] if atsas_dir else []
    cands += _env_roots()
    w = shutil.which("autorg")
    if w:
        cands.append(Path(w).resolve().parent)
    cands += _default_roots()
    for c in cands:
        for b in (c / "bin", c):
            if (b / exe).is_file():
                if atsas_dir is None:
                    _ATSAS_BIN = b
                    from ..core import installcache
                    installcache.put("atsas", {"bin": str(b)})
                return b
    if atsas_dir is None:
        from ..core import installcache
        installcache.forget("atsas")
    raise AtsasError(
        "ATSAS not found. Pass atsas_dir=..., or set PYATSAS_DIR / ATSAS to the "
        "installation folder (e.g. C:\\Program Files\\ATSAS-4.1.4 or ~/ATSAS-4.1.4).")


def cached_atsas() -> Optional[Path]:
    """The ATSAS *bin* folder found by the last search (kept between sessions), checked with one file test
    only; None if there was none or it is gone."""
    from ..core import installcache
    c = installcache.get("atsas")
    if not c or not c.get("bin"):
        return None
    b = Path(c["bin"])
    return b if (b / ("autorg" + EXE_SUFFIX)).is_file() else None


def atsas_root(atsas_dir=None) -> Path:
    b = find_atsas(atsas_dir)
    return b.parent if b.name.lower() == "bin" else b


def license_file(atsas_dir=None) -> Optional[Path]:
    """ATSAS 4 looks for atsas.lic in the installation root."""
    p = atsas_root(atsas_dir) / "atsas.lic"
    return p if p.is_file() else None


def _exe(tool: str, atsas_dir=None) -> str:
    b = find_atsas(atsas_dir)
    p = b / (tool + EXE_SUFFIX)
    if not p.is_file():
        raise AtsasError(f"{tool} not found in {b}")
    return str(p)


def _child_env(atsas_dir=None) -> dict:
    """Environment for ATSAS child processes: ATSAS points to the root (the
    licence is located through it) and bin/ is first on PATH (DLLs on
    Windows, helper programs called by pipelines such as dammif->damaver)."""
    env = os.environ.copy()
    b = find_atsas(atsas_dir)
    env["ATSAS"] = str(atsas_root(atsas_dir))
    env["PATH"] = str(b) + os.pathsep + env.get("PATH", "")
    env.setdefault("LC_ALL", "C")        # '.' as decimal separator in output
    return env


@dataclass
class RunInfo:
    cmd: list[str]
    cwd: Path
    stdout: str
    stderr: str
    returncode: int


def run_tool(tool: str, args: Sequence[str], cwd=None, atsas_dir=None,
             timeout: Optional[float] = 600, check: bool = True,
             stdin: Optional[str] = None) -> RunInfo:
    """Run an ATSAS program in `cwd` (a fresh temp dir by default)."""
    cmd = [_exe(tool, atsas_dir), *map(str, args)]
    cwd = Path(cwd or tempfile.mkdtemp(prefix=f"pyatsas_{tool}_"))
    cwd.mkdir(parents=True, exist_ok=True)
    kw = {}
    if IS_WIN:   # no console window popping up when called from a Qt GUI
        kw["creationflags"] = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    try:
        p = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True,
                           encoding="utf-8", errors="replace", input=stdin,
                           timeout=timeout, env=_child_env(atsas_dir), **kw)
    except subprocess.TimeoutExpired as e:
        raise AtsasError(f"{tool} timed out after {timeout} s") from e
    except OSError as e:                 # not executable, wrong architecture, blocked by antivirus ...
        raise AtsasError(f"{tool} could not be started ({cmd[0]}): {e}") from e
    info = RunInfo(cmd, cwd, p.stdout or "", p.stderr or "", p.returncode)
    if check and p.returncode != 0:
        msg = (p.stderr or p.stdout or "").strip()
        if "licen" in msg.lower():
            msg += f"\n(licence expected at {atsas_root(atsas_dir) / 'atsas.lic'})"
        raise AtsasError(f"{tool} failed ({p.returncode}):\n{msg}")
    return info


def _tail(run: Optional[RunInfo], n: int = 12) -> str:
    """Last lines of a run's output, for error messages."""
    if run is None:
        return ""
    text = "\n".join(x for x in ((run.stderr or "").strip(), (run.stdout or "").strip()) if x)
    lines = text.splitlines()[-n:]
    return ("\n" + "\n".join(lines)) if lines else " (the program printed nothing)"


def _need_file(path: Path, tool: str, run: Optional[RunInfo]) -> Path:
    """The output file a program should have written; AtsasError (with its output) if missing or empty."""
    path = Path(path)
    if not path.is_file():
        raise AtsasError(f"{tool} finished but wrote no {path.name} (in {path.parent}):{_tail(run)}")
    if path.stat().st_size == 0:
        raise AtsasError(f"{tool} wrote an empty {path.name} (in {path.parent}):{_tail(run)}")
    return path


def _finite(*vals) -> bool:
    try:
        return all(np.isfinite(float(v)) for v in vals)
    except (TypeError, ValueError):
        return False


def version(atsas_dir=None) -> str:
    r = run_tool("autorg", ["--version"], atsas_dir=atsas_dir, check=False)
    m = re.search(r"(\d+\.\d+\.\d+)", r.stdout + r.stderr)
    return m.group(1) if m else (r.stdout + r.stderr).strip()


# ----------------------------------------------------------------------------
# data I/O
# ----------------------------------------------------------------------------
def inv_A_to_inv_nm(q):
    return np.asarray(q) * 10.0


def inv_nm_to_inv_A(q):
    return np.asarray(q) / 10.0


def write_dat(path, q, I, err=None, title="pyatsas export") -> Path:
    """Write a 3-column ATSAS-compatible .dat file."""
    q, I = np.asarray(q, float), np.asarray(I, float)
    err = np.full_like(I, np.nan) if err is None else np.asarray(err, float)
    if np.isnan(err).any():                     # ATSAS needs errors
        err = np.where(np.isnan(err), np.maximum(np.abs(I) * 0.01, 1e-30), err)
    path = Path(path)
    with path.open("w") as f:
        f.write(f"{title}\n")
        for row in zip(q, I, err):
            f.write("%.8e  %.8e  %.8e\n" % row)
    return path


def read_dat(path) -> np.ndarray:
    """Read any whitespace column file, skipping non-numeric lines."""
    rows = []
    with open(path, errors="replace") as f:
        for line in f:
            parts = line.replace(",", " ").split()
            try:
                rows.append([float(x.replace("D", "E")) for x in parts])
            except ValueError:
                continue
    if not rows:
        raise AtsasError(f"no numeric data in {path}")
    n = max(set(map(len, rows)), key=[len(r) for r in rows].count)
    return np.array([r for r in rows if len(r) == n])


def _safe_name(name: str) -> str:
    stem = re.sub(r"[^A-Za-z0-9_.-]", "_", name)
    return stem[-40:] if len(stem) > 40 else stem


def _as_file(data, cwd: Path, name="input.dat") -> Path:
    """Accept a path or a (q, I[, err]) tuple and place it in `cwd`.

    Files are copied into the working directory under a short ASCII name and
    passed as a relative name: some ATSAS programs (Fortran heritage) truncate
    long paths and stumble over spaces or non-ASCII characters - common on
    Windows ("C:\\Users\\...\\OneDrive - PSI\\...") and macOS.
    """
    cwd.mkdir(parents=True, exist_ok=True)
    if isinstance(data, (str, os.PathLike)):
        src = Path(data).expanduser().resolve()
        if not src.is_file():
            raise AtsasError(f"file not found: {src}")
        dst = cwd / _safe_name(src.name)
        if dst.resolve() != src:
            shutil.copy2(src, dst)
        return Path(dst.name)
    write_dat(cwd / name, *data)
    return Path(name)


# ----------------------------------------------------------------------------
# autorg
# ----------------------------------------------------------------------------
@dataclass
class GuinierResult:
    rg: float
    rg_err: float
    i0: float
    i0_err: float
    first: int
    last: int
    quality: float
    aggregated: float
    run: RunInfo = field(repr=False)
    warnings: list = field(default_factory=list)


def autorg(data, mininterval: Optional[int] = None, smaxrg: Optional[float] = None,
           sminrg: Optional[float] = None, minrg: Optional[float] = None,
           maxrg: Optional[float] = None, atsas_dir=None, cwd=None) -> GuinierResult:
    """Guinier analysis. `data` = path or (q, I, err).
    Options left at None use the autorg defaults (mininterval 3, smaxrg 1.3,
    minrg 1.0, no sminrg/maxrg limit)."""
    cwd = Path(cwd or tempfile.mkdtemp(prefix="pyatsas_autorg_"))
    f = _as_file(data, cwd)
    args: list = [f, "--format=ssv"]          # ssv: no header, file name last
    for k, v in (("mininterval", mininterval), ("smaxrg", smaxrg),
                 ("sminrg", sminrg), ("minrg", minrg), ("maxrg", maxrg)):
        if v is not None:
            args.append(f"--{k}={v}")
    r = run_tool("autorg", args, cwd=cwd, atsas_dir=atsas_dir, check=False)
    vals = None
    for line in r.stdout.splitlines():
        toks = line.split()
        nums = []
        for t in toks[:8]:
            try:
                nums.append(float(t))
            except ValueError:
                break
        if len(nums) >= 7:
            vals = nums + [np.nan] * (8 - len(nums))
    if r.returncode != 0 or vals is None:
        raise AtsasError(f"autorg found no Guinier region:\n{r.stderr or r.stdout}")
    if not (_finite(vals[0], vals[2], vals[4], vals[5]) and vals[0] > 0):
        raise AtsasError(f"autorg returned no valid Rg ({' '.join(map(str, vals[:6]))}):{_tail(r)}")
    g = GuinierResult(vals[0], vals[1], vals[2], vals[3], int(vals[4]),
                      int(vals[5]), vals[6], vals[7], r)
    if _finite(g.quality) and g.quality < 0.5:
        g.warnings.append(f"AUTORG quality {g.quality:.2f} < 0.5: the Guinier range is doubtful")
    if _finite(g.aggregated) and g.aggregated > 0:
        g.warnings.append(f"AUTORG flags aggregation ({g.aggregated:g})")
    return g


# ----------------------------------------------------------------------------
# GNOM / DATGNOM
# ----------------------------------------------------------------------------
@dataclass
class GnomResult:
    r: np.ndarray
    pr: np.ndarray
    pr_err: np.ndarray
    q: np.ndarray           # reciprocal-space grid of the fit
    i_exp: np.ndarray
    i_err: np.ndarray
    i_reg: np.ndarray       # regularised (fitted) intensity
    dmax: float
    rg_real: float
    i0_real: float
    total_estimate: float
    out_file: Path
    run: Optional[RunInfo] = field(default=None, repr=False)
    warnings: list = field(default_factory=list)


def _grab(pattern, text, default=np.nan):
    m = re.search(pattern, text, re.I)
    return float(m.group(1).replace("D", "E")) if m else default


def parse_gnom_out(path, run: Optional[RunInfo] = None, tool: str = "GNOM") -> GnomResult:
    """Parse a GNOM .out file (ATSAS 2.8 - 4.x formats).  AtsasError if the file is missing, empty or has no
    p(r) table; questionable values (no Dmax / Rg, negative p(r), poor total estimate) go to ``warnings``."""
    path = _need_file(Path(path), tool, run)
    txt = path.read_text(errors="replace")
    lines = txt.splitlines()

    def numeric_block(start, ncols_min):
        out = []
        for ln in lines[start:]:
            p = ln.split()
            if not p:
                if out:
                    break
                continue
            try:
                vals = [float(x.replace("D", "E")) for x in p]
            except ValueError:
                if out:
                    break
                continue
            if len(vals) >= ncols_min:
                out.append(vals)
            elif out:
                break
        return out

    # --- real space: header line contains R and P(R)
    pr_rows = []
    for i, ln in enumerate(lines):
        u = ln.upper()
        if "P(R)" in u and re.search(r"\bR\b", u):
            pr_rows = numeric_block(i + 1, 3)
            if pr_rows:
                break
    # --- reciprocal space: header contains J EXP / I REG
    rec_rows = []
    for i, ln in enumerate(lines):
        u = ln.upper()
        if ("J EXP" in u or "I EXP" in u) and "REG" in u:
            blk = numeric_block(i + 1, 2)
            rec_rows = blk
            if blk:
                break

    # reciprocal rows: leading rows (below smin of data) may have only 2 cols
    q = iexp = ierr = ireg = np.array([])
    if rec_rows:
        full = [r for r in rec_rows if len(r) >= 5]
        if full:
            a = np.array([r[:5] for r in full])
            q, iexp, ierr, ireg = a[:, 0], a[:, 1], a[:, 2], a[:, 4]

    a = np.array([r[:3] for r in pr_rows]) if pr_rows else np.empty((0, 3))
    dmax = _grab(r"(?:Real space range|Dmax)\s*[:=]?\s*(?:" + _FLOAT +
                 r"\s*(?:to|-)\s*)?(" + _FLOAT + ")", txt)
    if not len(a):
        raise AtsasError(f"{tool}: no p(r) table in {path}:{_tail(run)}")
    if not np.all(np.isfinite(a[:, :2])):
        raise AtsasError(f"{tool}: p(r) in {path} contains NaN/inf:{_tail(run)}")
    if np.isnan(dmax) and len(a):
        dmax = a[-1, 0]
    res = GnomResult(
        r=a[:, 0], pr=a[:, 1], pr_err=a[:, 2],
        q=q, i_exp=iexp, i_err=ierr, i_reg=ireg,
        dmax=dmax,
        rg_real=_grab(r"Real space\s*:?\s*Rg\s*[:=]?\s*(" + _FLOAT + ")", txt,
                      _grab(r"Real space Rg\s*[:=]\s*(" + _FLOAT + ")", txt)),
        i0_real=_grab(r"Real space\s*:?.*?I\(0\)\s*[:=]?\s*(" + _FLOAT + ")", txt,
                      _grab(r"Real space I\(0\)\s*[:=]\s*(" + _FLOAT + ")", txt)),
        total_estimate=_grab(r"Total\s+Estimate\s*[:=]?\s*(" + _FLOAT + ")", txt),
        out_file=path,
        run=run,
    )
    w = res.warnings
    if not len(res.q):
        w.append(f"{tool}: no fitted I(q) found in {path.name}")
    if not _finite(res.rg_real):
        w.append(f"{tool}: no real-space Rg found in {path.name}")
    te = res.total_estimate
    if _finite(te) and te < 0.5:
        w.append(f"{tool}: total estimate {te:.2f} < 0.5 (GNOM rates the solution as poor)")
    pmax = np.nanmax(np.abs(res.pr)) if len(res.pr) else 0
    if pmax > 0 and np.nanmin(res.pr) < -0.05 * pmax:
        w.append(f"{tool}: p(r) is negative by more than 5 % of its maximum (Dmax or background?)")
    return res


def datgnom(data, rg: Optional[float] = None, first: Optional[int] = None,
            last: Optional[int] = None, atsas_dir=None, cwd=None) -> GnomResult:
    """Automatic p(r) (DATGNOM). If rg is None, autorg is run first.
    first/last are 1-based point indices as in ATSAS."""
    cwd = Path(cwd or tempfile.mkdtemp(prefix="pyatsas_datgnom_"))
    f = _as_file(data, cwd)
    if rg is None:
        rg = autorg(cwd / f, atsas_dir=atsas_dir, cwd=cwd).rg
    out = f.stem + ".out"
    args: list = [f, "-o", out, f"--rg={rg}"]
    if first:
        args.append(f"--first={first}")
    if last:
        args.append(f"--last={last}")
    r = run_tool("datgnom", args, cwd=cwd, atsas_dir=atsas_dir)
    return parse_gnom_out(cwd / out, run=r, tool="DATGNOM")


def gnom(data, dmax: float, rmin: float = 0.0, force_zero_rmin: bool = True,
         force_zero_rmax: bool = True, nr: Optional[int] = None,
         alpha: Optional[float] = None, system: int = 0,
         first: Optional[int] = None, last: Optional[int] = None,
         rad56: Optional[float] = None, atsas_dir=None, cwd=None) -> GnomResult:
    """GNOM with a fixed Dmax (= --rmax), ATSAS 3/4 command line.
    system: 0 monodisperse p(r), 1 polydisperse spheres D_V(R), 3 cross-section
    (rods), 4 thickness (lamellae), 5 cylinders, 6 shells (2 is GUI only)."""
    cwd = Path(cwd or tempfile.mkdtemp(prefix="pyatsas_gnom_"))
    f = _as_file(data, cwd)
    out = f.stem + ".out"
    yn = lambda b: "Y" if b else "N"
    args: list = [f, f"--rmax={dmax}", "-o", out, f"--system={system}",
                  f"--force-zero-rmin={yn(force_zero_rmin)}",
                  f"--force-zero-rmax={yn(force_zero_rmax)}"]
    if rmin:
        args.append(f"--rmin={rmin}")
    for k, v in (("first", first), ("last", last), ("nr", nr),
                 ("alpha", alpha), ("rad56", rad56)):
        if v is not None:
            args.append(f"--{k}={v}")
    r = run_tool("gnom", args, cwd=cwd, atsas_dir=atsas_dir)
    return parse_gnom_out(cwd / out, run=r, tool="GNOM")


# ----------------------------------------------------------------------------
# flat background: DATPOROD --exponent --constant, and an independent fit
# ----------------------------------------------------------------------------
@dataclass
class PorodResult:
    """High-q fit I(q) = A q^-B + bg.  ``bg`` is the flat background in the sense I = signal + bg (to be
    subtracted from the data); ``c`` is DATPOROD's C as printed.  ``first``/``last``: 1-based points of the
    data that were fitted."""
    a: float
    da: float
    b: float
    db: float
    bg: float
    dbg: float
    chi2: float
    first: int
    last: int
    c: float = np.nan
    source: str = "datporod"
    note: str = ""
    run: Optional[RunInfo] = field(default=None, repr=False)

    def model(self, q):
        return self.a * np.asarray(q, float) ** (-self.b) + self.bg


def _chi2(q, I, dI, model):
    w = np.where(np.isfinite(dI) & (dI > 0), dI, np.nan)
    r = (I - model) / w
    r = r[np.isfinite(r)]
    return float(np.sum(r ** 2) / max(len(r) - 3, 1)) if len(r) else np.nan


def datporod_exponent(data, first: Optional[int] = None, last: Optional[int] = None,
                      constant: bool = True, atsas_dir=None, cwd=None) -> PorodResult:
    """DATPOROD --exponent [--constant]: I(s) = A s^-B - C over the points first..last (1-based).

    The manual writes the model as A s^-B - C and calls C "a constant to subtract" without saying on which
    side.  Both readings are tested against the fitted points and the one that describes the data is used;
    ``bg`` is then the background to subtract from I (``note`` tells which reading was taken)."""
    cwd = Path(cwd or tempfile.mkdtemp(prefix="pyatsas_datporod_"))
    f = _as_file(data, cwd)
    a = read_dat(cwd / f)
    n = len(a)
    first = int(first or 1)
    last = int(last or n)
    args: list = [f, "--exponent", f"--first={first}", f"--last={last}"]
    if constant:
        args.append("--constant")
    r = run_tool("datporod", args, cwd=cwd, atsas_dir=atsas_dir)
    vals = None
    for line in r.stdout.splitlines():
        nums = []
        for t in line.split():
            try:
                nums.append(float(t.replace("D", "E")))
            except ValueError:
                break
        if len(nums) >= 7:
            vals = nums[:7]
    if vals is None:
        raise AtsasError(f"datporod: no result in its output:\n{r.stdout}\n{r.stderr}")
    A_, dA, B_, dB, C_, dC, chi = vals
    if not _finite(A_, B_, C_):
        raise AtsasError(f"datporod returned no valid fit (A={A_}, B={B_}, C={C_}):{_tail(r)}")
    q, I = a[first - 1:last, 0], a[first - 1:last, 1]
    dI = a[first - 1:last, 2] if a.shape[1] > 2 else np.full(len(q), np.nan)
    base = A_ * q ** (-B_)
    c_minus, c_plus = _chi2(q, I, dI, base - C_), _chi2(q, I, dI, base + C_)
    if C_ == 0 or not (np.isfinite(c_minus) and np.isfinite(c_plus)):
        bg, note = -C_, "I = A q^-B - C (as in the manual)"
    elif c_plus < c_minus:
        bg, note = C_, "C fits as I = A q^-B + C"
    else:
        bg, note = -C_, "C fits as I = A q^-B - C"
    return PorodResult(A_, dA, B_, dB, bg, dC, chi, first, last, c=C_, source="datporod", note=note, run=r)


def fit_power_background(q, I, dI=None, exponent: Optional[float] = None) -> PorodResult:
    """Independent weighted least-squares fit I = A q^-B + bg of the given (tail) points, without ATSAS.
    ``exponent``: B fixed (4 = Porod), else fitted.  Errors from the Jacobian (scaled by the reduced χ²
    when it exceeds 1)."""
    from scipy.optimize import least_squares
    q, I = np.asarray(q, float), np.asarray(I, float)
    dI = np.full_like(I, np.nan) if dI is None else np.asarray(dI, float)
    ok = np.isfinite(q) & np.isfinite(I) & (q > 0)
    q, I, dI = q[ok], I[ok], dI[ok]
    if len(q) < 4:
        raise AtsasError("fewer than 4 points for the background fit")
    w = np.where(np.isfinite(dI) & (dI > 0), dI, np.maximum(np.abs(I) * 0.01, 1e-30))
    k = max(3, len(q) // 5)
    bg0 = float(np.median(I[-k:]))
    B0 = exponent if exponent is not None else 4.0
    A0 = max(float(np.median((I[:k] - bg0) * q[:k] ** B0)), 1e-30)
    fixed = exponent is not None

    def unpack(p):
        return (np.exp(p[0]), exponent, p[1]) if fixed else (np.exp(p[0]), p[1], p[2])

    def res(p):
        A_, B_, bg = unpack(p)
        return (A_ * q ** (-B_) + bg - I) / w

    p0 = [np.log(A0), bg0] if fixed else [np.log(A0), B0, bg0]
    sol = least_squares(res, p0, x_scale="jac")
    A_, B_, bg = unpack(sol.x)
    m = len(q) - len(sol.x)
    chi = float(np.sum(sol.fun ** 2) / max(m, 1))
    try:
        cov = np.linalg.inv(sol.jac.T @ sol.jac) * max(chi, 1.0)
        err = np.sqrt(np.diag(cov))
    except np.linalg.LinAlgError:
        err = np.full(len(sol.x), np.nan)
    dA = A_ * err[0]
    dB = 0.0 if fixed else err[1]
    dbg = err[-1]
    return PorodResult(A_, dA, B_, dB, bg, dbg, chi, 1, len(q), c=np.nan, source="pyATsas fit",
                       note="B fixed" if fixed else "")


# ----------------------------------------------------------------------------
# CRYSOL / CRYSON
# ----------------------------------------------------------------------------
@dataclass
class CrysolResult:
    q: np.ndarray
    i_model: np.ndarray            # model intensity (fitted if data given)
    i_exp: Optional[np.ndarray]
    i_err: Optional[np.ndarray]
    chi2: float
    files: list[Path]
    run: RunInfo = field(repr=False)
    warnings: list = field(default_factory=list)


def crysol(model, data=None, lm: int = 20, ns: int = 101, smax: float = 0.5,
           extra: Iterable[str] = (), atsas_dir=None, cwd=None,
           program: str = "crysol") -> CrysolResult:
    """Compute (or fit) the scattering of a PDB/mmCIF model.
    program="cryson" for neutrons (pass D2O fraction etc. via `extra`)."""
    cwd = Path(cwd or tempfile.mkdtemp(prefix=f"pyatsas_{program}_"))
    m = _as_file(model, cwd)
    args: list = [m, f"--lm={lm}", f"--ns={ns}", f"--smax={smax}", *extra]
    if data is not None:
        args.insert(1, _as_file(data, cwd))
    r = run_tool(program, args, cwd=cwd, atsas_dir=atsas_dir)
    files = sorted(cwd.iterdir())
    fit = sorted(cwd.glob("*.fit"), key=lambda p: p.stat().st_mtime)
    if data is not None:
        if not fit:
            raise AtsasError(f"{program}: a fit to the data was asked for, but no .fit file was written "
                             f"(in {cwd}):{_tail(r)}")
        txt = _need_file(fit[-1], program, r).read_text(errors="replace")
        pat = r"Chi\^?2\s*[:=]\s*(" + _FLOAT + ")"
        chi2 = _grab(pat, txt, _grab(pat, r.stdout))
        a = read_dat(fit[-1])
        if a.shape[1] < 3:
            raise AtsasError(f"{program}: {fit[-1].name} has {a.shape[1]} columns, expected q, I_exp, (σ,) I_fit")
        res = CrysolResult(a[:, 0], a[:, -1], a[:, 1], a[:, 2] if a.shape[1] > 3 else None, chi2, files, r)
        if not _finite(chi2):
            res.warnings.append(f"{program}: no χ² found in {fit[-1].name} or the output")
    else:
        curve = sorted(cwd.glob("*.int")) or sorted(cwd.glob("*.abs"))
        if not curve:
            raise AtsasError(f"{program} produced no .int/.abs file (in {cwd}):{_tail(r)}")
        a = read_dat(_need_file(curve[-1], program, r))
        res = CrysolResult(a[:, 0], a[:, 1], None, None, np.nan, files, r)
    if not np.all(np.isfinite(res.i_model)):
        raise AtsasError(f"{program}: the model curve contains NaN/inf:{_tail(r)}")
    return res


# ----------------------------------------------------------------------------
# DAMMIF (parallel independent runs)
# ----------------------------------------------------------------------------
@dataclass
class DammifResult:
    models: list[Path]
    fits: list[Path]
    workdir: Path
    runs: list[RunInfo] = field(repr=False)
    warnings: list = field(default_factory=list)
    errors: list = field(default_factory=list)     # messages of the runs that failed


def dammif(gnom_out, nruns: int = 10, mode: str = "fast", symmetry: str = "P1",
           max_workers: Optional[int] = None, extra: Iterable[str] = (),
           atsas_dir=None, cwd=None, timeout: float = 3600) -> DammifResult:
    """Run `nruns` independent DAMMIF reconstructions in parallel
    (ATSAS 4 writes mmCIF by default; pass extra=['--model-format=pdb'])."""
    cwd = Path(cwd or tempfile.mkdtemp(prefix="pyatsas_dammif_"))
    g = _as_file(gnom_out, cwd)

    def one(i):
        try:            # one failing run must not lose the others
            return run_tool("dammif", [g, f"--mode={mode}", f"--symmetry={symmetry}",
                                       f"--prefix=dammif_{i:02d}", *extra],
                            cwd=cwd, atsas_dir=atsas_dir, timeout=timeout), None
        except AtsasError as e:
            return None, f"run {i}: {e}"

    n = max_workers or max(1, min(nruns, (os.cpu_count() or 2) - 1))
    with ThreadPoolExecutor(n) as ex:
        out = list(ex.map(one, range(1, nruns + 1)))
    runs = [r for r, _e in out if r is not None]
    errors = [e for _r, e in out if e]
    models = [p for p in sorted(cwd.glob("dammif_*-1.cif")) + sorted(cwd.glob("dammif_*-1.pdb"))
              if p.stat().st_size > 0]
    if not models:
        first = errors[0] if errors else "no model files dammif_NN-1.cif/.pdb" + _tail(runs[0] if runs else None)
        raise AtsasError(f"DAMMIF produced no bead model in {nruns} runs (in {cwd}):\n{first}")
    res = DammifResult(models, sorted(cwd.glob("dammif_*.fir")), cwd, runs, errors=errors)
    if len(models) < nruns:
        res.warnings.append(f"DAMMIF: {len(models)} of {nruns} runs gave a model"
                            + (f"; {len(errors)} failed, first: {errors[0].splitlines()[0]}" if errors else ""))
    return res


# ----------------------------------------------------------------------------
# self test (synthetic sphere; exact Rg = sqrt(3/5) R, Dmax = 2R)
# ----------------------------------------------------------------------------
def sphere_curve(R: float = 5.0, qmin: float = 0.02, qmax: float = 3.0,
                 n: int = 400, noise: float = 0.01, seed: int = 1):
    q = np.linspace(qmin, qmax, n)
    x = q * R
    I = 1e3 * (3 * (np.sin(x) - x * np.cos(x)) / x**3) ** 2 + 1e-4
    err = noise * I + 1e-5
    I = I + np.random.default_rng(seed).normal(0, err)
    return q, I, err


def selftest(atsas_dir=None, R: float = 5.0) -> bool:
    print(f"platform  : {sys.platform}  python {sys.version.split()[0]}")
    b = find_atsas(atsas_dir)
    print(f"ATSAS bin : {b}")
    print(f"version   : {version(atsas_dir)}")
    lic = license_file(atsas_dir)
    print(f"licence   : {lic or 'atsas.lic NOT found in ' + str(atsas_root(atsas_dir))}")
    data = sphere_curve(R)
    rg_true, dmax_true = np.sqrt(0.6) * R, 2 * R
    ok = True
    work = Path(tempfile.mkdtemp(prefix="pyatsas_selftest_"))
    print(f"workdir   : {work}")
    try:
        g = autorg(data, cwd=work, atsas_dir=atsas_dir)
        d = abs(g.rg / rg_true - 1)
        print(f"autorg    : Rg = {g.rg:.3f} +- {g.rg_err:.3f}  (exact {rg_true:.3f}, "
              f"dev {100*d:.1f} %)  points {g.first}-{g.last}")
        ok &= d < 0.03
        p = datgnom(work / "input.dat", rg=g.rg, cwd=work, atsas_dir=atsas_dir)
        print(f"datgnom   : Dmax = {p.dmax:.2f} (exact {dmax_true:.2f})  "
              f"Rg(real) = {p.rg_real:.3f}  TE = {p.total_estimate:.2f}  "
              f"{len(p.r)} r-points")
        ok &= len(p.r) > 5 and abs(p.dmax / dmax_true - 1) < 0.15
        p2 = gnom(work / "input.dat", dmax=dmax_true, cwd=work / "gnom",
                  atsas_dir=atsas_dir)
        d = abs(p2.rg_real / rg_true - 1) if np.isfinite(p2.rg_real) else 1
        print(f"gnom      : Rg(real) = {p2.rg_real:.3f} at Dmax = {dmax_true}  "
              f"(dev {100*d:.1f} %)")
        ok &= d < 0.03
    except Exception as e:                 # report, don't crash the self test
        print(f"FAILED: {type(e).__name__}: {e}")
        ok = False
    print("RESULT    :", "OK" if ok else "PROBLEM - send this output and the workdir files")
    return ok


def _main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(prog="python -m pysasfit.atsas",
                                 description="Thin Python wrapper for ATSAS")
    ap.add_argument("files", nargs="*", help="data files: autorg + datgnom")
    ap.add_argument("--atsas-dir", help="ATSAS installation (root or bin)")
    ap.add_argument("--selftest", action="store_true",
                    help="run autorg/datgnom/gnom on a synthetic sphere")
    a = ap.parse_args(argv)
    try:
        if a.selftest or not a.files:
            return 0 if selftest(a.atsas_dir) else 1
        for fn in a.files:
            g = autorg(fn, atsas_dir=a.atsas_dir)
            p = datgnom(fn, rg=g.rg, atsas_dir=a.atsas_dir)
            print(f"{fn}: Rg = {g.rg:.3f} +- {g.rg_err:.3f}  I0 = {g.i0:.4g}  "
                  f"Dmax = {p.dmax:.3f}  Rg(real) = {p.rg_real:.3f}  "
                  f"TE = {p.total_estimate:.2f}   [{p.out_file}]")
    except AtsasError as e:
        print(e, file=sys.stderr)
        return 1
    return 0


# ----------------------------------------------------------------------------
# interactive ATSAS programs and their icons
# ----------------------------------------------------------------------------
#: (label, executable, icon name in share/icons/hicolor/<n>x<n>/apps, description)
GUI_PROGRAMS = [
    ("ATSAS launcher", "launcher", "atsas", "all ATSAS programs with their input forms"),
    ("PRIMUS", "primusqt", "primus", "subtract, Rg, Dmax, MW, ..."),
    ("CHROMIXS", "chromixs", "chromixs", "SEC-SAS data analysis"),
    ("PEAK", "peakqt", "peak", "partially ordered systems"),
    ("SASPLOT", "sasplotqt", "int2", "plot SAS data"),
    ("GNOMPLOT", "gnomplotqt", "gnom", "plot GNOM .out files"),
    ("SVDPLOT", "svdplot", "svd", "singular value decomposition of data series"),
    ("SASDOC", "sasdoc", "atsas", "ATSAS documentation"),
    ("MASSHA", "massha", "massha", "envelopes, bead and atomic models (legacy, Windows)"),
]


def program_path(name: str, atsas_dir=None) -> Optional[Path]:
    """Path of an ATSAS executable (``bin``, else ``Legacy`` of a Windows installation), or None."""
    try:
        b = find_atsas(atsas_dir)
    except AtsasError:
        return None
    for d in (b, b.parent / "Legacy"):
        p = d / (name + EXE_SUFFIX)
        if p.is_file():
            return p
    return None


def launch(name: str, args: Sequence = (), atsas_dir=None, cwd=None) -> subprocess.Popen:
    """Start an (interactive) ATSAS program without waiting for it, e.g. ``launch("primusqt", [file])``."""
    p = program_path(name, atsas_dir)
    if p is None:
        raise AtsasError(f"{name} is not part of the ATSAS installation in {find_atsas(atsas_dir)}")
    kw = {}
    if IS_WIN:
        kw["creationflags"] = getattr(subprocess, "DETACHED_PROCESS", 0) | \
            getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
    else:
        kw["start_new_session"] = True
    return subprocess.Popen([str(p), *map(str, args)], cwd=str(cwd) if cwd else None,
                            env=_child_env(atsas_dir), stdout=subprocess.DEVNULL,
                            stderr=subprocess.DEVNULL, **kw)


def icon_files(name: str = "atsas", atsas_dir=None) -> list[tuple[int, Path]]:
    """[(size, png)] of an icon of the ATSAS installation (``share/icons/hicolor/<n>x<n>/apps/<name>.png``),
    smallest first; empty without ATSAS.  These are the original ATSAS icons; pySASfit does not ship
    copies of them."""
    try:
        root = atsas_root(atsas_dir)
    except AtsasError:
        return []
    out = []
    base = root / "share" / "icons" / "hicolor"
    try:
        dirs = list(base.iterdir())
    except OSError:
        return []
    for d in dirs:
        m = re.fullmatch(r"(\d+)x\1", d.name)
        f = d / "apps" / f"{name}.png"
        if m and f.is_file():
            out.append((int(m.group(1)), f))
    return sorted(out)


app_icon_files = icon_files


if __name__ == "__main__":
    sys.exit(_main())
