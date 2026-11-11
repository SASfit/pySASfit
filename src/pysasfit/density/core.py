"""pyDENsity core: DENSS (DENsity from Solution Scattering, T. D. Grant, Nature Methods 15 (2018) 191) from pySASfit.

DENSS reconstructs a low-resolution electron-density map on a grid directly from a SAXS/SANS curve (iterative
structure factor retrieval with shrink-wrap support).  Unlike bead models it allows internal contrast and holes, and
unlike spherical-harmonics envelopes it is not limited to star-shaped bodies.

DENSS is **not** part of pySASfit: it is a separate GPL-3.0 package (``pip install denss``, pure Python, needs numpy
and scipy).  pyDENsity runs it as a separate process (``python -m denss.scripts...``) and reads the files it writes;
no DENSS code is imported into pySASfit.

    from pysasfit.density import run_denss, run_denss_all
    r = run_denss((q_nm, I, dI), dmax_nm=10, mode="FAST")     # one map; r.rho, r.voxel, r.rg, r.chi2, r.fit_*
    a = run_denss_all((q_nm, I, dI), dmax_nm=10, nmaps=20, cores=4)   # 20 maps, aligned and averaged, FSC resolution

Units: pySASfit curves have q in 1/nm; DENSS works in 1/Å and Å.  The data are written in 1/Å; results are given in
Å (maps, Rg, volumes) as DENSS reports them, the fitted curve in 1/nm again.

Compatibility: DENSS 1.8.7 (the current release) uses ``np.in1d`` and ``np.trapz``, which NumPy 2.4 removed, and
passes a one-element array to ``struct.pack`` when writing MRC maps (``denss.core.write_mrc``, re-exported as
``denss.write_mrc``), which NumPy 2.x rejects.  The process is
therefore started through a small shim (:data:`SHIM`) that adds the two names (``np.isin``, ``np.trapezoid``) and
writes maps with a plain float box size, without changing the DENSS installation.
"""
from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

__all__ = ["DenssError", "DenssInfo", "find_denss", "DenssResult", "DenssAllResult", "run_denss", "run_denss_all",
           "read_mrc", "write_mrc", "read_fit", "read_stats", "isosurface", "level_for_volume", "SHIM", "sphere_curve",
           "selftest"]

IS_WIN = sys.platform.startswith("win")


class DenssError(RuntimeError):
    pass


#: started as ``python -c SHIM <module> <args>``: NumPy 2 compatibility, then the DENSS script as __main__
SHIM = r'''
import runpy, sys
import numpy as np
if not hasattr(np, "in1d"):
    np.in1d = lambda a, b, assume_unique=False, invert=False: np.isin(np.ravel(a), b, assume_unique=assume_unique,
                                                                      invert=invert)
if not hasattr(np, "trapz"):
    np.trapz = np.trapezoid
try:
    import denss as _d
    import denss.core as _dc
    _orig_write_mrc = _dc.write_mrc
    def _write_mrc(rho, side, filename="map.mrc"):
        s = np.atleast_1d(np.asarray(side, float)).ravel()
        s = (float(s[0]),) * 3 if len(s) == 1 else tuple(float(x) for x in s)
        return _orig_write_mrc(rho, s, filename)
    _dc.write_mrc = _write_mrc
    if getattr(_d, "write_mrc", None) is _orig_write_mrc:     # the package re-exports it (denss-all uses that one)
        _d.write_mrc = _write_mrc
except Exception:
    pass
mod = sys.argv[1]
sys.argv = [mod] + sys.argv[2:]
runpy.run_module(mod, run_name="__main__", alter_sys=True)
'''


# ===================================================================================== finding DENSS
@dataclass
class DenssInfo:
    python: str
    version: str
    location: str


_INFO: dict = {}


def find_denss(python: Optional[str] = None, refresh: bool = False) -> DenssInfo:
    """The Python interpreter that can import ``denss`` (default: the one running pySASfit; else ``python`` /
    ``python3`` on the PATH, else the interpreter of a ``denss`` script on the PATH)."""
    key = python or ""
    if key in _INFO and not refresh:
        return _INFO[key]
    cands = [python] if python else [sys.executable]
    if not python:
        for name in ("python3", "python", "py"):
            w = shutil.which(name)
            if w and w not in cands:
                cands.append(w)
        w = shutil.which("denss")
        if w:                                   # the interpreter of the console script (shebang / launcher)
            try:
                first = Path(w).read_bytes()[:200].split(b"\n")[0]
                if first.startswith(b"#!"):
                    cands.append(first[2:].decode(errors="ignore").strip().split()[0])
            except OSError:
                pass
            for p in (Path(w).parent / "python.exe", Path(w).parent.parent / "python.exe",
                      Path(w).parent / "python3"):
                if p.is_file():
                    cands.append(str(p))
    errors = []
    for py in cands:
        try:
            p = subprocess.run([py, "-c", "import denss, sys; print(denss.__version__); print(denss.__file__)"],
                               capture_output=True, text=True, timeout=60, **_no_window())
        except (OSError, subprocess.TimeoutExpired) as e:
            errors.append(f"{py}: {e}")
            continue
        if p.returncode == 0:
            lines = p.stdout.strip().splitlines()
            info = DenssInfo(py, lines[0] if lines else "?", lines[1] if len(lines) > 1 else "")
            _INFO[key] = info
            return info
        errors.append(f"{py}: {(p.stderr or p.stdout).strip().splitlines()[-1:] or ['import failed']}")
    raise DenssError("DENSS not found. Install it for the Python that runs pySASfit:\n"
                     f'    "{sys.executable}" -m pip install denss\n'
                     "or set Settings → Preferences → pyDENsity → Python with DENSS.\n" + "\n".join(errors))


def _no_window():
    return {"creationflags": getattr(subprocess, "CREATE_NO_WINDOW", 0)} if IS_WIN else {}


def _run(module: str, args: Sequence, cwd: Path, python: Optional[str], timeout: float):
    info = find_denss(python)
    cmd = [info.python, "-c", SHIM, module, *map(str, args)]
    env = os.environ.copy()
    env.setdefault("MPLBACKEND", "Agg")              # DENSS plots to files only
    env["PYTHONIOENCODING"] = "utf-8"
    t0 = time.time()
    try:
        p = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, encoding="utf-8", errors="replace",
                           timeout=timeout, env=env, **_no_window())
    except subprocess.TimeoutExpired as e:
        raise DenssError(f"DENSS ({module}) did not finish within {timeout:.0f} s (in {cwd})") from e
    except OSError as e:
        raise DenssError(f"DENSS could not be started ({info.python}): {e}") from e
    run = {"cmd": ["python", "-c", "<shim>", module, *map(str, args)], "cwd": cwd, "stdout": p.stdout or "",
           "stderr": p.stderr or "", "returncode": p.returncode, "seconds": time.time() - t0, "python": info.python}
    if p.returncode != 0:
        raise DenssError(f"DENSS ({module}) failed ({p.returncode}):{_tail(run)}")
    return run


def _tail(run, n=12):
    """Last lines of the error output (a traceback ends there), else of the normal output."""
    err = (run.get("stderr") or "").strip().splitlines()
    out = (run.get("stdout") or "").strip().splitlines()
    lines = (err[-n:] if err else []) or out[-n:]
    return ("\n" + "\n".join(lines)) if lines else " (no output)"


def _need(path: Path, run) -> Path:
    if not path.is_file() or path.stat().st_size == 0:
        raise DenssError(f"DENSS finished but wrote no {path.name} (in {path.parent}):{_tail(run)}")
    return path


# ===================================================================================== files
def write_dat(path, q, I, err=None, title="pyDENsity"):
    q, I = np.asarray(q, float), np.asarray(I, float)
    err = np.full_like(I, np.nan) if err is None else np.asarray(err, float)
    err = np.where(np.isfinite(err) & (err > 0), err, np.maximum(np.abs(I) * 0.01, 1e-30))
    ok = np.isfinite(q) & np.isfinite(I) & (q > 0)
    with open(path, "w") as f:
        f.write(f"# {title}\n")
        for row in zip(q[ok], I[ok], err[ok]):
            f.write("%.8e  %.8e  %.8e\n" % row)
    return Path(path)


def read_mrc(path):
    """(rho with shape (nx, ny, nz), voxel size (Å, per axis), header dict) of an MRC/CCP4 map (mode 2, float32)."""
    import struct
    path = Path(path)
    data = path.read_bytes()
    if len(data) < 1024:
        raise DenssError(f"{path.name}: not an MRC map (shorter than its header)")
    endian = "<"
    nx, ny, nz, mode = struct.unpack_from("<iiii", data, 0)
    if not (0 < nx < 10000 and 0 < ny < 10000 and 0 < nz < 10000):
        endian = ">"
        nx, ny, nz, mode = struct.unpack_from(">iiii", data, 0)
    if mode != 2:
        raise DenssError(f"{path.name}: MRC mode {mode}, only 2 (32-bit real) is read")
    mx, my, mz = struct.unpack_from(endian + "iii", data, 28)
    a, b, c = struct.unpack_from(endian + "fff", data, 40)
    nsymbt = struct.unpack_from(endian + "i", data, 92)[0]
    start = 1024 + max(nsymbt, 0)
    n = nx * ny * nz
    if len(data) < start + 4 * n:
        raise DenssError(f"{path.name}: truncated map ({len(data)} bytes for {nx}×{ny}×{nz})")
    rho = np.frombuffer(data, dtype=endian + "f4", count=n, offset=start).reshape((nx, ny, nz), order="F")
    voxel = np.array([a / (mx or nx), b / (my or ny), c / (mz or nz)], float)
    return rho.astype(np.float64), voxel, {"cell": (a, b, c), "shape": (nx, ny, nz)}


def write_mrc(path, rho, voxel):
    """Minimal MRC writer (mode 2), compatible with :func:`read_mrc` and DENSS / ChimeraX / PyMOL."""
    import struct
    rho = np.asarray(rho, np.float32)
    nx, ny, nz = rho.shape
    v = np.broadcast_to(np.asarray(voxel, float), (3,))
    h = bytearray(1024)
    struct.pack_into("<iiii", h, 0, nx, ny, nz, 2)
    struct.pack_into("<iii", h, 16, -nx // 2, -ny // 2, -nz // 2)
    struct.pack_into("<iii", h, 28, nx, ny, nz)
    struct.pack_into("<ffffff", h, 40, v[0] * nx, v[1] * ny, v[2] * nz, 90.0, 90.0, 90.0)
    struct.pack_into("<iii", h, 64, 1, 2, 3)
    struct.pack_into("<fff", h, 76, float(rho.min()), float(rho.max()), float(rho.mean()))
    struct.pack_into("<i", h, 88, 1)
    h[208:212] = b"MAP "
    h[212:216] = bytes([0x44, 0x41, 0, 0])
    with open(path, "wb") as f:
        f.write(bytes(h))
        f.write(rho.flatten(order="F").tobytes())
    return Path(path)


def read_fit(path):
    """DENSS ``*_map.fit``: q (1/Å), I_data, error, I_map and χ² from the header."""
    path = Path(path)
    text = path.read_text(errors="replace")
    m = re.search(r"chi2\s*=\s*([-+0-9.eE]+)", text)
    a = np.loadtxt(path, comments="#", ndmin=2)
    if a.shape[1] < 4 or not len(a):
        raise DenssError(f"{path.name}: expected 4 columns q, I, error, I(map)")
    return a[:, 0], a[:, 1], a[:, 2], a[:, 3], (float(m.group(1)) if m else np.nan)


def read_stats(path):
    """DENSS ``*_stats_by_step.dat``: χ², Rg (Å; imaginary values early on → NaN), support volume (Å³) per step."""
    chi, rg, vol = [], [], []
    for ln in Path(path).read_text(errors="replace").splitlines():
        if ln.startswith("#") or not ln.strip():
            continue
        t = ln.split()
        if len(t) < 3:
            continue
        try:
            chi.append(float(t[0]))
            rg.append(np.nan if t[1].endswith("j") else float(t[1]))
            vol.append(float(t[2]))
        except ValueError:
            continue
    return np.array(chi), np.array(rg), np.array(vol)


def _log_values(text):
    out = {}
    for key, pat in (("chi2", r"Final Chi2:\s*([-+0-9.eE]+)"), ("rg", r"Final Rg:\s*([-+0-9.eE]+)"),
                     ("support_volume", r"Final Support Volume:\s*([-+0-9.eE]+)"),
                     ("steps", r"Number of steps:\s*(\d+)"), ("dmax", r"Maximum dimension:\s*([-+0-9.eE]+)"),
                     ("voxel", r"Real space voxel size \(angstroms\):\s*([-+0-9.eE]+)"),
                     ("seed", r"Random seed:\s*(\d+)"), ("mode", r"Mode:\s*(\S+)")):
        m = re.findall(pat, text)
        if m:
            v = m[-1]
            try:
                out[key] = float(v) if key not in ("mode",) else v
            except ValueError:
                out[key] = v
    return out


# ===================================================================================== results
@dataclass
class DenssResult:
    rho: np.ndarray                 # electron density (nx, ny, nz)
    voxel: np.ndarray               # Å per voxel along x, y, z
    support: Optional[np.ndarray]
    fit_q: np.ndarray               # 1/nm
    fit_I: np.ndarray
    fit_err: np.ndarray
    fit_map: np.ndarray             # I of the map, scaled to the data
    chi2: float
    rg: float                       # Å
    support_volume: float           # Å³
    steps: int
    stats: tuple                    # (chi2, Rg, support volume) per step
    map_file: Path
    workdir: Path
    log: dict = field(default_factory=dict)
    run: dict = field(default_factory=dict, repr=False)
    warnings: list = field(default_factory=list)


@dataclass
class DenssAllResult:
    rho: np.ndarray                 # averaged map
    voxel: np.ndarray
    map_file: Path
    maps: list                      # aligned single maps (paths)
    resolution: float               # Å, from the FSC between the maps (NaN if not written)
    resolution_err: float
    fsc: Optional[np.ndarray]       # (1/resolution, FSC)
    rg_mean: float
    rg_sd: float
    fit_q: np.ndarray
    fit_I: np.ndarray
    fit_err: np.ndarray
    fit_map: np.ndarray
    chi2: float
    workdir: Path
    run: dict = field(default_factory=dict, repr=False)
    warnings: list = field(default_factory=list)


def _prepare(data, cwd: Path, units_in: str):
    """Input file for DENSS in cwd: (q [1/nm], I, dI) -> data.dat in 1/Å; a path is copied (GNOM .out or .dat,
    ``units_in`` 'nm' or 'a' as in the file)."""
    cwd.mkdir(parents=True, exist_ok=True)
    if isinstance(data, (str, Path)):
        src = Path(data)
        if not src.is_file():
            raise DenssError(f"file not found: {src}")
        dst = cwd / ("data" + src.suffix.lower())
        shutil.copy2(src, dst)
        return dst.name, ("nm" if units_in == "nm" else "a")
    q, I, dI = (list(data) + [None])[:3]
    write_dat(cwd / "data.dat", np.asarray(q, float) / 10.0, I, dI, "pySASfit curve, q in 1/angstrom")
    return "data.dat", "a"


def _common_args(dmax_nm, mode, nsamples, voxel, oversampling, steps, ncs, ncs_axis, ncs_type, seed, ne, gpu,
                 extra):
    args = ["-m", mode]
    if dmax_nm:
        args += ["-d", f"{10.0 * float(dmax_nm):.6g}"]            # Å
    if nsamples:
        args += ["-n", int(nsamples)]
    if voxel:
        args += ["-v", f"{float(voxel):.6g}"]
    if oversampling:
        args += ["-os", f"{float(oversampling):.6g}"]
    if steps:
        args += ["-s", int(steps)]
    if ncs and int(ncs) > 1:
        args += ["-ncs", int(ncs), "-ncs_axis", str(ncs_axis), "-ncs_type", ncs_type]
    if seed not in (None, ""):
        args += ["--seed", int(seed)]
    if ne:
        args += ["--ne", f"{float(ne):.6g}"]
    if gpu:
        args += ["--gpu"]
    return args + [str(x) for x in (extra or ())]


def _fit_nm(q, I, e, Im):
    return np.asarray(q) * 10.0, np.asarray(I), np.asarray(e), np.asarray(Im)


def run_denss(data, dmax_nm: Optional[float] = None, mode: str = "FAST", nsamples: Optional[int] = None,
              voxel: Optional[float] = None, oversampling: Optional[float] = None, steps: Optional[int] = None,
              ncs: int = 0, ncs_axis: int = 1, ncs_type: str = "C", seed: Optional[int] = None,
              ne: Optional[float] = None, gpu: bool = False, extra: Sequence = (), units_in: str = "nm",
              python: Optional[str] = None, cwd=None, timeout: float = 7200) -> DenssResult:
    """One DENSS reconstruction (``denss``).  ``data``: (q [1/nm], I, dI), or a file (GNOM ``.out`` or ``.dat``, q
    in ``units_in``).  ``dmax_nm`` None: taken from a GNOM file or estimated by DENSS.  ``voxel`` in Å."""
    cwd = Path(cwd or tempfile.mkdtemp(prefix="pydensity_"))
    fname, units = _prepare(data, cwd, units_in)
    args = ["-f", fname, "-u", units, "-o", "denss", "--plot_off", "-q",
            *_common_args(dmax_nm, mode, nsamples, voxel, oversampling, steps, ncs, ncs_axis, ncs_type, seed, ne,
                          gpu, extra)]
    run = _run("denss.scripts.denss_abinitio", args, cwd, python, timeout)
    rho, vox, _h = read_mrc(_need(cwd / "denss.mrc", run))
    sup = None
    if (cwd / "denss_support.mrc").is_file():
        try:
            sup = read_mrc(cwd / "denss_support.mrc")[0] > 0.5
        except DenssError:
            pass
    q, I, e, Im, chi = read_fit(_need(cwd / "denss_map.fit", run))
    stats = read_stats(cwd / "denss_stats_by_step.dat") if (cwd / "denss_stats_by_step.dat").is_file() \
        else (np.array([]),) * 3
    log = _log_values((cwd / "denss.log").read_text(errors="replace")) if (cwd / "denss.log").is_file() else {}
    if not np.all(np.isfinite(rho)):
        raise DenssError(f"DENSS map {cwd / 'denss.mrc'} contains NaN/inf:{_tail(run)}")
    res = DenssResult(rho, vox, sup, *_fit_nm(q, I, e, Im), chi, float(log.get("rg", np.nan)),
                      float(log.get("support_volume", np.nan)), int(log.get("steps", len(stats[0]))), stats,
                      cwd / "denss.mrc", cwd, log, run)
    if not np.isfinite(res.rg):
        res.warnings.append("DENSS reported no real Rg (the map may not have converged)")
    return res


def run_denss_all(data, dmax_nm: Optional[float] = None, nmaps: int = 20, cores: int = 1, mode: str = "FAST",
                  enantiomers: bool = True, nsamples: Optional[int] = None, voxel: Optional[float] = None,
                  oversampling: Optional[float] = None, steps: Optional[int] = None, ncs: int = 0, ncs_axis: int = 1,
                  ncs_type: str = "C", ne: Optional[float] = None, gpu: bool = False, extra: Sequence = (),
                  units_in: str = "nm", python: Optional[str] = None, cwd=None,
                  timeout: float = 4 * 3600) -> DenssAllResult:
    """The full DENSS pipeline (``denss-all``): ``nmaps`` reconstructions, enantiomer selection, alignment,
    averaging and the resolution from the Fourier shell correlation between the maps."""
    cwd = Path(cwd or tempfile.mkdtemp(prefix="pydensity_all_"))
    fname, units = _prepare(data, cwd, units_in)
    before = {p.name for p in cwd.iterdir()}
    args = ["-f", fname, "-u", units, "-o", "denss", "-nm", int(nmaps), "-j", max(1, int(cores)), "--plot_off",
            "-en_on" if enantiomers else "-en_off",
            *_common_args(dmax_nm, mode, nsamples, voxel, oversampling, steps, ncs, ncs_axis, ncs_type, None, ne, gpu,
                          extra)]
    run = _run("denss.scripts.denss_all", args, cwd, python, timeout)
    new = sorted((p for p in cwd.iterdir() if p.is_dir() and p.name not in before and p.name.startswith("denss")),
                 key=lambda p: p.stat().st_mtime)
    if not new:
        raise DenssError(f"denss-all wrote no result folder in {cwd}:{_tail(run)}")
    out = new[-1]
    avg = out / "denss_avg.mrc"
    rho, vox, _h = read_mrc(_need(avg, run))
    fsc, res_, res_sd = None, np.nan, np.nan
    f = out / "denss_fsc.dat"
    if f.is_file():
        txt = f.read_text(errors="replace")
        m = re.search(r"Resolution\s*=\s*([-+0-9.eE]+)\s*\+-\s*([-+0-9.eE]+)", txt)
        if m:
            res_, res_sd = float(m.group(1)), float(m.group(2))
        try:
            fsc = np.loadtxt(f, comments="#", ndmin=2)[:, :2]
        except ValueError:
            pass
    rgm = rgs = chi = np.nan
    for lg in sorted(out.glob("*.log")):
        txt = lg.read_text(errors="replace")
        m = re.search(r"Average \|Rg\|\.*:\s*([-+0-9.eE]+)\s*\+-\s*([-+0-9.eE]+)", txt)
        if m:
            rgm, rgs = float(m.group(1)), float(m.group(2))
        m = re.search(r"Average Chi2\.*:\s*([-+0-9.eE]+)", txt)
        if m:
            chi = float(m.group(1))
    q = I = e = Im = np.array([])
    if (out / "denss_map.fit").is_file():          # q, I, error, I_fit_0 ... I_fit_n-1 (one column per map)
        try:
            a = np.loadtxt(out / "denss_map.fit", comments="#", ndmin=2)
            if a.shape[1] >= 4:
                q, I, e, Im = a[:, 0], a[:, 1], a[:, 2], a[:, 3:].mean(axis=1)
        except ValueError:
            pass
    maps = sorted(out.glob("denss_*_aligned.mrc"))
    r = DenssAllResult(rho, vox, avg, maps, res_, res_sd, fsc, rgm, rgs, *_fit_nm(q, I, e, Im), chi, out, run)
    if not np.isfinite(res_):
        r.warnings.append("no FSC resolution found (denss_fsc.dat)")
    if len(maps) < nmaps:
        r.warnings.append(f"{len(maps)} aligned maps of {nmaps}")
    return r


# ===================================================================================== surfaces
def level_for_volume(rho, voxel, volume):
    """Density level whose isosurface encloses about ``volume`` (Å³): between the n-th largest voxel value and the
    next lower distinct value, so that a surface exists also for maps with flat plateaus (supports, thresholded
    maps)."""
    vv = float(np.prod(np.broadcast_to(np.asarray(voxel, float), (3,))))
    vals = np.sort(np.asarray(rho, float).ravel())[::-1]
    n = int(np.clip(round(volume / vv), 1, vals.size))
    v = vals[n - 1]
    lower = vals[vals < v]
    if not lower.size:                       # constant map: no surface possible, return its value
        return float(v)
    return float(0.5 * (v + lower[0]))


def isosurface(rho, level, voxel=1.0, centre=True):
    """(vertices in Å, faces) of the isosurface rho = level (scikit-image marching cubes if installed, else
    pyqtgraph's)."""
    rho = np.asarray(rho, float)
    vox = np.broadcast_to(np.asarray(voxel, float), (3,))
    pad = np.pad(rho, 1, constant_values=min(float(rho.min()), level) - 1.0)      # closed surfaces at the box edge
    try:
        from skimage.measure import marching_cubes
        v, f, _n, _vals = marching_cubes(pad, level=level, spacing=tuple(vox))
        v = v - vox
    except ImportError:
        import pyqtgraph as pg
        v, f = pg.isosurface(pad, level)
        v = (v - 1.0) * vox
    if centre and len(v):
        v = v - (np.array(rho.shape) - 1) * vox / 2
    return np.asarray(v, float), np.asarray(f, np.int32)


# ===================================================================================== self test
def sphere_curve(R_nm=2.0, n=300, noise=0.01, seed=1):
    """Homogeneous sphere, q in 1/nm from 0.3/R to 15/R (Rg = √(3/5) R)."""
    q = np.linspace(0.3 / R_nm, 15 / R_nm, n)
    x = q * R_nm
    I = 1e3 * (3 * (np.sin(x) - x * np.cos(x)) / x ** 3) ** 2 + 1e-3
    e = noise * I + 2e-3
    return q, I + np.random.default_rng(seed).normal(0, e), e


def selftest(python=None, cwd=None, out=print) -> bool:
    """One FAST map of a sphere R = 2 nm (Rg 15.49 Å); passes within 10 % (a single DENSS map is ~4 % large)."""
    info = find_denss(python)
    out(f"DENSS {info.version} with {info.python}\n  {info.location}")
    r = run_denss(sphere_curve(), dmax_nm=4.0, mode="FAST", seed=1, python=python, cwd=cwd)
    exact = np.sqrt(0.6) * 20.0
    dev = r.rg / exact - 1
    out(f"sphere R = 20 Å: Rg {r.rg:.2f} Å (exact {exact:.2f}, {100 * dev:+.1f} %), χ² {r.chi2:.3g}, "
        f"{r.steps} steps, map {r.rho.shape}, voxel {r.voxel[0]:.2f} Å, {r.run.get('seconds', 0):.1f} s")
    ok = abs(dev) < 0.10
    out("RESULT: " + ("OK" if ok else "PROBLEM (a single map can converge badly: try again or use denss-all)"))
    return ok
