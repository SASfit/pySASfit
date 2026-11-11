"""3D models of the ATSAS programs: bead models (DAMMIF, DAMMIN, MONSA, GASBOR, DAMAVER ...), spherical-harmonics
envelopes (SASHA ``.flm``), and the SASHA run itself.  No Qt here; the 3D panel is :mod:`pysasfit.atsas.view3d`.

Bead models
    PDB (``ATOM``/``HETATM``) or mmCIF (``_atom_site``) files with one dummy atom per bead, coordinates in Å; the
    bead radius is read from the header (DAMMIN/DAMMIF: "DAM packing radius", "Dummy atom radius" ...), else half
    the median nearest-neighbour distance.

Envelopes
    SASHA describes the particle by its boundary r(ω) = Σ_{l≤L, |m|≤l} F_lm Y_lm(ω) with complex coefficients F_lm.
    The layout of the ``.flm`` file is not documented.  :func:`read_flm` therefore tries the plausible layouts and
    normalisations of Y_lm and keeps only an interpretation whose volume and Rg agree with the values SASHA writes
    into the same file (``FlmEnvelope.check``); without such a check the envelope is not drawn.

SASHA (``Legacy/sasha20.exe``, Windows only) is an interactive dialog program; :func:`sasha` answers its documented
dialog (ATSAS 2.x manual) through stdin, keeps the transcript, and checks that the dialog went as expected.
"""
from __future__ import annotations

import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

from . import core as C
from .core import AtsasError, RunInfo

__all__ = ["BeadModel", "read_bead_model", "write_bead_pdb", "FlmEnvelope", "read_flm", "envelope_mesh",
           "sphere_mesh", "beads_mesh", "SashaResult", "sasha", "legacy_path", "SASHA_PROMPTS",
           "flm_of_shape", "write_flm_reference"]


# ===================================================================================== bead models
@dataclass
class BeadModel:
    xyz: np.ndarray                 # (n, 3), Å
    radius: float                   # bead radius, Å
    path: Path
    header: dict = field(default_factory=dict)     # values from the header (Rg, Dmax, volume ...)
    radius_from: str = "header"

    @property
    def n(self):
        return len(self.xyz)

    def rg(self):
        c = self.xyz.mean(axis=0)
        return float(np.sqrt(np.mean(np.sum((self.xyz - c) ** 2, axis=1))))

    def dmax(self):
        x = self.xyz
        if len(x) > 4000:
            x = x[np.random.default_rng(0).choice(len(x), 4000, replace=False)]
        d2 = np.sum(x ** 2, 1)[:, None] + np.sum(x ** 2, 1)[None, :] - 2 * x @ x.T
        return float(np.sqrt(max(d2.max(), 0)))            # between bead centres, as in the ATSAS headers


_RADIUS_KEYS = ("packing radius", "dummy atom radius", "bead radius", "atom radius", "dam radius")


def _header_values(lines):
    out = {}
    for ln in lines:
        m = re.match(r"\s*(?:REMARK\s+\d*\s*)?([A-Za-z][A-Za-z .()/'-]*?)\s*\.{2,}\s*:\s*(" + C._FLOAT + r")", ln)
        if m:
            out[m.group(1).strip().lower()] = float(m.group(2).replace("D", "E"))
    return out


def read_bead_model(path) -> BeadModel:
    """Bead (dummy-atom) model from a PDB or mmCIF file; AtsasError if it has no atoms."""
    path = Path(path)
    text = path.read_text(errors="replace")
    lines = text.splitlines()
    xyz = []
    if "_atom_site." in text:                                 # mmCIF
        cols, i = [], 0
        while i < len(lines):
            if lines[i].strip().startswith("_atom_site."):
                while i < len(lines) and lines[i].strip().startswith("_atom_site."):
                    cols.append(lines[i].strip().split()[0][len("_atom_site."):])
                    i += 1
                ix, iy, iz = (cols.index(k) for k in ("Cartn_x", "Cartn_y", "Cartn_z"))
                while i < len(lines):
                    s = lines[i].strip()
                    if not s or s.startswith(("#", "_", "loop_")):
                        break
                    t = s.split()
                    if len(t) >= len(cols):
                        try:
                            xyz.append((float(t[ix]), float(t[iy]), float(t[iz])))
                        except ValueError:
                            pass
                    i += 1
                break
            i += 1
    else:                                                     # PDB: fixed columns 31-54
        for ln in lines:
            if ln.startswith(("ATOM", "HETATM")):
                try:
                    xyz.append((float(ln[30:38]), float(ln[38:46]), float(ln[46:54])))
                except ValueError:
                    t = ln.split()
                    try:
                        xyz.append(tuple(float(v) for v in t[5:8]))
                    except (ValueError, IndexError):
                        pass
    if not xyz:
        raise AtsasError(f"{path.name}: no atoms (beads) found")
    xyz = np.asarray(xyz, float)
    head = _header_values(lines)
    r, src = None, "header"
    for k, v in head.items():
        if any(key in k for key in _RADIUS_KEYS) and v > 0:
            r = v
            break
    if r is None:                                             # half the median nearest-neighbour distance
        src = "nearest neighbours"
        x = xyz[:3000]
        d2 = np.sum(x ** 2, 1)[:, None] + np.sum(x ** 2, 1)[None, :] - 2 * x @ x.T
        np.fill_diagonal(d2, np.inf)
        r = 0.5 * float(np.sqrt(np.median(np.maximum(d2.min(axis=1), 0)))) if len(x) > 1 else 1.0
    return BeadModel(xyz, float(r), path, head, src)


def write_bead_pdb(model: BeadModel, path) -> Path:
    """Plain PDB with CA atoms (what MASSHA reads: it shows only CA or P atoms)."""
    path = Path(path)
    with path.open("w") as f:
        f.write(f"REMARK 265 DAM packing radius ..................... : {model.radius:.3f}\n")
        for i, (x, y, z) in enumerate(model.xyz, 1):
            f.write(f"ATOM  {i % 100000:5d}  CA  ASP A{(i // 10 + 1) % 10000:4d}    "
                    f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00 20.00           C\n")
        f.write("END\n")
    return path


# ===================================================================================== spherical harmonics
def _ylm(l, m, theta, phi):
    """Complex orthonormal Y_lm (Condon-Shortley), theta polar, phi azimuth (scipy >= 1.15 or older)."""
    import scipy.special as sp
    if hasattr(sp, "sph_harm_y"):
        return sp.sph_harm_y(l, m, theta, phi)
    return sp.sph_harm(m, l, phi, theta)


def _grid(n_theta=48, n_phi=96):
    x, w = np.polynomial.legendre.leggauss(n_theta)
    theta = np.arccos(x)
    phi = np.linspace(0, 2 * np.pi, n_phi, endpoint=False)
    return theta, phi, w * (2 * np.pi / n_phi)          # weights per (theta) row, ∑ = 4π


def _radius(coef, theta, phi, real_basis=False):
    """r(θ, φ) on the grid from {(l, m): value}.  real_basis: real orthonormal harmonics (cos m φ / sin |m| φ)."""
    T, P = np.meshgrid(theta, phi, indexing="ij")
    r = np.zeros(T.shape)
    for (l, m), f in coef.items():
        if real_basis:
            y = _ylm(l, abs(m), T, P)
            if m > 0:
                b = np.sqrt(2) * (-1) ** m * y.real
            elif m < 0:
                b = np.sqrt(2) * (-1) ** m * y.imag
            else:
                b = y.real
            r += float(np.real(f)) * b
        else:
            r += np.real(f * _ylm(l, m, T, P))
    return r


def _shape_moments(r, theta, phi, w):
    """Volume and Rg (about the centroid) of the star-shaped body bounded by r(θ, φ)."""
    T, P = np.meshgrid(theta, phi, indexing="ij")
    W = w[:, None] * np.ones_like(T)
    V = float(np.sum(W * r ** 3) / 3)
    n = np.stack([np.sin(T) * np.cos(P), np.sin(T) * np.sin(P), np.cos(T)], -1)
    c = np.sum((W * r ** 4 / 4)[..., None] * n, axis=(0, 1)) / V
    I2 = float(np.sum(W * r ** 5) / 5) / V
    return V, float(np.sqrt(max(I2 - c @ c, 0))), c


@dataclass
class FlmEnvelope:
    coef: dict                      # (l, m) -> complex (or real for real_basis)
    lmax: int
    real_basis: bool
    layout: str                     # how the file was read
    path: Optional[Path] = None
    rg_file: float = np.nan         # values written by SASHA
    volume_file: float = np.nan
    rg: float = np.nan              # recomputed from the coefficients
    volume: float = np.nan
    check: str = ""                 # "ok: ..." when Rg and V agree with the file, else the reason
    header: dict = field(default_factory=dict)

    @property
    def verified(self):
        return self.check.startswith("ok")

    def radius(self, theta, phi):
        return _radius(self.coef, theta, phi, self.real_basis)


def _complete(coef):
    """Fill m < 0 from m > 0 for a real function: F_l,-m = (-1)^m conj(F_lm)."""
    out = dict(coef)
    for (l, m), f in coef.items():
        if m > 0 and (l, -m) not in out:
            out[(l, -m)] = (-1) ** m * np.conj(f)
    return out


def _candidates(nums, rows):
    """Possible readings of the numbers of an .flm file: [(layout, coef dict, real_basis)]."""
    cands = []
    # explicit indices: rows "l m re [im]"
    idx = [r for r in rows if len(r) >= 3 and float(r[0]).is_integer() and float(r[1]).is_integer()
           and 0 <= r[0] <= 40 and abs(r[1]) <= r[0]]
    if len(idx) >= 4:
        c = {(int(r[0]), int(r[1])): (complex(r[2], r[3]) if len(r) >= 4 else r[2]) for r in idx}
        has_neg = any(m < 0 for _l, m in c)
        cands.append(("rows l m Re Im", c if has_neg else _complete(c), False))
        if all(len(r) == 3 for r in idx):
            cands.append(("rows l m value (real harmonics)", c, True))
    # plain sequences
    vals = np.asarray(nums, float)
    for L in range(0, 30):
        n_full, n_half = (L + 1) ** 2, (L + 1) * (L + 2) // 2
        if len(vals) >= 2 * n_full and len(vals) - 2 * n_full <= 4:
            v = vals[:2 * n_full].reshape(-1, 2)
            c, k = {}, 0
            for l in range(L + 1):
                for m in range(-l, l + 1):
                    c[(l, m)] = complex(*v[k]); k += 1
            cands.append((f"sequence Re Im, l ≤ {L}, m = -l..l", c, False))
        if len(vals) >= 2 * n_half and len(vals) - 2 * n_half <= 4:
            v = vals[:2 * n_half].reshape(-1, 2)
            c, k = {}, 0
            for l in range(L + 1):
                for m in range(0, l + 1):
                    c[(l, m)] = complex(*v[k]); k += 1
            cands.append((f"sequence Re Im, l ≤ {L}, m = 0..l", _complete(c), False))
        if len(vals) >= n_full and len(vals) - n_full <= 4:
            c, k = {}, 0
            for l in range(L + 1):
                for m in range(-l, l + 1):
                    c[(l, m)] = float(vals[k]); k += 1
            cands.append((f"sequence of real coefficients, l ≤ {L}", c, True))
    return cands


def read_flm(path, tol: float = 0.03) -> FlmEnvelope:
    """Read a SASHA/CRYSOL ``.flm`` envelope.  The interpretation is accepted only if the volume and Rg recomputed
    from the coefficients agree within ``tol`` with the values in the file; otherwise ``check`` says why and the
    best candidate is returned unverified (do not draw it as the result)."""
    path = Path(path)
    lines = path.read_text(errors="replace").splitlines()
    head, nums, rows = {}, [], []
    for ln in lines:
        low = ln.lower()
        fl = re.findall(C._FLOAT, ln.replace("D", "E").replace("d", "e"))
        if re.search(r"[a-z]{2,}", low):                      # a text line: maybe "Rg ... 12.3", "volume ... "
            if fl:
                v = float(fl[-1])
                if re.search(r"\brg\b|radius of gyration", low):
                    head.setdefault("rg", v)
                elif "volume" in low:
                    head.setdefault("volume", v)
                elif "distance" in low:
                    head.setdefault("distance", v)
            continue
        if fl:
            vals = [float(x) for x in fl]
            rows.append(vals)
            nums.extend(vals)
    cands = _candidates(nums, rows)
    if not cands:
        raise AtsasError(f"{path.name}: no spherical-harmonics coefficients recognised")
    theta, phi, w = _grid()
    rg_f, v_f = head.get("rg", np.nan), head.get("volume", np.nan)
    best = None
    for layout, coef, real in cands:
        r = _radius(coef, theta, phi, real)
        if np.nanmin(r) <= 0:
            continue
        V, Rg, _c = _shape_moments(r, theta, phi, w)
        err = max(abs(V / v_f - 1) if np.isfinite(v_f) else np.inf,
                  abs(Rg / rg_f - 1) if np.isfinite(rg_f) else np.inf)
        env = FlmEnvelope(coef, max(l for l, _m in coef), real, layout, path, rg_f, v_f, Rg, V, header=head)
        if best is None or err < best[0]:
            best = (err, env)
    if best is None:
        raise AtsasError(f"{path.name}: no reading of the coefficients gives a closed surface (r > 0 everywhere)")
    err, env = best
    if not (np.isfinite(rg_f) and np.isfinite(v_f)):
        env.check = "not verified: the file has no Rg and volume to compare with"
    elif err <= tol:
        env.check = f"ok: Rg {env.rg:.4g} (file {rg_f:.4g}), V {env.volume:.4g} (file {v_f:.4g}), {env.layout}"
    else:
        env.check = (f"not verified: best reading ({env.layout}) gives Rg {env.rg:.4g} / V {env.volume:.4g}, "
                     f"the file says Rg {rg_f:.4g} / V {v_f:.4g}")
    return env


def flm_of_shape(radius_fn, lmax=6, n_theta=64, n_phi=128):
    """Complex coefficients F_lm of r(θ, φ) = radius_fn(θ, φ) (for tests and reference files)."""
    theta, phi, w = _grid(n_theta, n_phi)
    T, P = np.meshgrid(theta, phi, indexing="ij")
    W = w[:, None] * np.ones_like(T)
    r = radius_fn(T, P)
    return {(l, m): complex(np.sum(W * r * np.conj(_ylm(l, m, T, P))))
            for l in range(lmax + 1) for m in range(-l, l + 1)}


def write_flm_reference(path, coef, rg, volume):
    """A test file in the 'rows l m Re Im' layout with Rg and volume (not SASHA's own layout)."""
    with open(path, "w") as f:
        f.write("pyATsas reference envelope\n")
        f.write(f"Radius of gyration ........ : {rg:.6g}\nVolume .................... : {volume:.6g}\n")
        for (l, m), c in sorted(coef.items()):
            f.write(f"{l:4d} {m:4d} {c.real: .8e} {c.imag: .8e}\n")
    return Path(path)


# ===================================================================================== meshes (for the 3D panel)
def envelope_mesh(env: FlmEnvelope, n_theta=40, n_phi=80):
    """(vertices (n, 3), faces (k, 3)) of the envelope surface."""
    theta = np.linspace(0, np.pi, n_theta)
    phi = np.linspace(0, 2 * np.pi, n_phi, endpoint=False)
    r = env.radius(theta, phi)
    T, P = np.meshgrid(theta, phi, indexing="ij")
    v = np.stack([r * np.sin(T) * np.cos(P), r * np.sin(T) * np.sin(P), r * np.cos(T)], -1).reshape(-1, 3)
    faces = []
    for i in range(n_theta - 1):
        for j in range(n_phi):
            a, b = i * n_phi + j, i * n_phi + (j + 1) % n_phi
            c, d = a + n_phi, b + n_phi
            faces += [(a, c, b), (b, c, d)]
    return v, np.asarray(faces, np.int32)


def sphere_mesh(rows=6, cols=10):
    """Unit sphere (vertices, faces), coarse: one per bead."""
    theta = np.linspace(0, np.pi, rows + 1)
    phi = np.linspace(0, 2 * np.pi, cols, endpoint=False)
    T, P = np.meshgrid(theta, phi, indexing="ij")
    v = np.stack([np.sin(T) * np.cos(P), np.sin(T) * np.sin(P), np.cos(T)], -1).reshape(-1, 3)
    f = []
    for i in range(rows):
        for j in range(cols):
            a, b = i * cols + j, i * cols + (j + 1) % cols
            f += [(a, a + cols, b), (b, a + cols, b + cols)]
    return v, np.asarray(f, np.int32)


def beads_mesh(model: BeadModel, scale=1.0, rows=6, cols=10):
    """One mesh with a small sphere per bead (radius × scale)."""
    sv, sf = sphere_mesh(rows, cols)
    n = model.n
    verts = (model.xyz[:, None, :] + scale * model.radius * sv[None, :, :]).reshape(-1, 3)
    faces = (sf[None, :, :] + (np.arange(n) * len(sv))[:, None, None]).reshape(-1, 3)
    return verts, faces.astype(np.int32)


# ===================================================================================== SASHA
#: the dialog of SASHA for monomer data (ATSAS 2.x manual): keyword in the prompt, answer key
SASHA_PROMPTS = [("working mode", "mode"), ("monomer sas data", "data"), ("angular units", "units"),
                 ("dimer sas data", "dimer"), ("output file", "output"), ("smax for the monomer", "smax"),
                 ("initial approximation", "start"), ("symmetry", "symmetry"), ("harmonic order", "lmax"),
                 ("anysometry", "anisometry"), ("molecular weight", "mw"), ("rg in angstroms", "rg")]


@dataclass
class SashaResult:
    envelope: Optional[FlmEnvelope]
    flm: Path
    fit_q: np.ndarray
    fit_exp: np.ndarray
    fit_model: np.ndarray
    workdir: Path
    transcript: str
    run: Optional[RunInfo] = field(default=None, repr=False)
    warnings: list = field(default_factory=list)


def legacy_path(name: str, atsas_dir=None) -> Optional[Path]:
    """An executable of ``<ATSAS>/Legacy`` (Windows installations: SASHA, MASSHA, old GUIs), or None."""
    try:
        p = C.atsas_root(atsas_dir) / "Legacy" / (name + C.EXE_SUFFIX)
    except AtsasError:
        return None
    return p if p.is_file() else None


def sasha(data, smax: float, mw_kda: float, rg_angstrom: float, symmetry: str = "P1", lmax: int = 4,
          anisometry: str = "O", units: int = 2, program: str = "sasha20", atsas_dir=None, cwd=None,
          timeout: float = 600) -> SashaResult:
    """Ab-initio envelope by spherical harmonics (SASHA, legacy).  ``data`` (q, I, dI) or a file, q in the unit
    ``units`` (1: 4π sinθ/λ in 1/Å, 2: in 1/nm -- the pySASfit unit); ``smax`` in the same unit; molecular
    weight in kDa and Rg in Å as SASHA asks.  The answers go to the dialog through stdin; the transcript is kept
    (``sasha_dialog.log``) and checked against the documented order of the prompts."""
    exe = legacy_path(program, atsas_dir)
    if exe is None:
        raise AtsasError(f"{program} is not installed (it is in <ATSAS>/Legacy of the Windows installation only)")
    cwd = Path(cwd or tempfile.mkdtemp(prefix="pyatsas_sasha_"))
    cwd.mkdir(parents=True, exist_ok=True)
    if isinstance(data, (str, Path)):
        shutil.copy2(data, cwd / "mon.dat")
    else:
        C.write_dat(cwd / "mon.dat", *data, title="pyATsas for SASHA")
    sym = symmetry.upper()
    if sym not in ("P1", "P2", "P222", "PMMM", "P3"):
        raise AtsasError(f"SASHA symmetry {symmetry}: one of P1, P2, P222, Pmmm, P3")
    ans = {"mode": "", "data": "mon.dat", "units": str(units), "dimer": "", "output": "sas",
           "smax": f"{smax:.5g}", "start": "", "symmetry": sym, "lmax": str(int(lmax)),
           "anisometry": anisometry[:1].upper(), "mw": f"{mw_kda:.6g}", "rg": f"{rg_angstrom:.6g}"}
    answers = [ans[k] for _p, k in SASHA_PROMPTS] + [""] * 8      # spare returns = defaults of extra prompts
    kw = {}
    if C.IS_WIN:
        kw["creationflags"] = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    try:
        p = subprocess.run([str(exe)], cwd=cwd, input="\n".join(answers) + "\n", capture_output=True, text=True,
                           encoding="utf-8", errors="replace", timeout=timeout, env=C._child_env(atsas_dir), **kw)
    except subprocess.TimeoutExpired as e:
        raise AtsasError(f"SASHA did not finish within {timeout} s (it may wait for input in a window of its own: "
                         f"run it interactively in {cwd})") from e
    except OSError as e:
        raise AtsasError(f"SASHA could not be started ({exe}): {e}") from e
    run = RunInfo([str(exe)], cwd, p.stdout or "", p.stderr or "", p.returncode)
    transcript = run.stdout + ("\n--- stderr ---\n" + run.stderr if run.stderr.strip() else "")
    (cwd / "sasha_dialog.log").write_text("answers:\n" + "\n".join(f"  {k}: {ans[k]!r}" for _p, k in SASHA_PROMPTS)
                                          + "\n\n" + transcript)
    warnings = []
    low = run.stdout.lower()
    pos, missing = 0, []
    for prompt, _k in SASHA_PROMPTS:                 # the documented prompts, in this order?
        i = low.find(prompt, pos)
        if i < 0:
            missing.append(prompt)
        else:
            pos = i
    if missing:
        warnings.append("SASHA's dialog differed from the documented one (not seen in order: "
                        + ", ".join(missing) + "); check sasha_dialog.log")
    flm = cwd / "sas.flm"
    if not flm.is_file() or flm.stat().st_size == 0:
        raise AtsasError(f"SASHA wrote no envelope (sas.flm) in {cwd}; see sasha_dialog.log:{C._tail(run)}")
    fq = fe = fm = np.array([])
    fim = cwd / "sas.fim"
    if fim.is_file():
        try:
            a = C.read_dat(fim)
            fq, fe, fm = a[:, 0], a[:, 1], a[:, -1]
        except AtsasError:
            warnings.append("sas.fim (fit) could not be read")
    env = None
    try:
        env = read_flm(flm)
        if not env.verified:
            warnings.append(f"envelope {env.check}; use MASSHA to look at it")
    except AtsasError as e:
        warnings.append(f"envelope not read ({e}); use MASSHA to look at it")
    return SashaResult(env, flm, fq, fe, fm, cwd, transcript, run, warnings)
