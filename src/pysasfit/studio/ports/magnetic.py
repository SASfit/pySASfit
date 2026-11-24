"""Python ports of SASfit's magnetic core-shell spheres (libsasfit core: magnetic_shell, superparamagnetic_ff_psi).

Each port returns the magnetic **basis terms** (NN, NM, MH, MSF, CH; :mod:`pysasfit.studio.polarisation`) of one
particle at (q, ψ), ψ = angle between q and the field on the detector, field perpendicular to the beam:

* ``ff_magnetic_shell_psi``: core-shell sphere with nuclear and magnetic SLDs of core, shell and matrix,
  magnetisation saturated along the field (S1 = S2 = 1);
* ``ff_superparamagnetic_ff_psi``: core-shell sphere with a magnetic core (magnetic SLD of shell and matrix 0) whose
  moment follows a Langevin distribution with ξ = μH/kT (R/R_av)³ (moment ∝ core volume): S1 = L(ξ),
  S2 = 1 - 3L(ξ)/ξ. Coherent and fluctuation parts as in SASfit.

With a polarisation setting active (a data set of :class:`~pysasfit.studio.ModelFunction` with ``polarisation``)
the intensity is Σ w_k T_k for the measured quantity and the ``pol`` parameter is not used; without it the ports
give the SASfit result with ``pol`` in SASfit's labels (neutron moment): pol = +1 is neutron spin antiparallel to
the field (I- here, -2NM), I = (1+pol)/2 I- + (1-pol)/2 I+. The quantities of a polarisation setting use the
neutron spin (Moon, Riste & Koehler 1969), see :mod:`pysasfit.studio.polarisation`.

C: ``ff_magnetic_shell_cross`` (not ported, the quantity "difference" replaces it) set ψ = π/2 as degrees
(sin²ψ ≈ 7.5e-4 instead of 1) and overwrote the ψ parameter of the caller; fixed 2026-11-23 (ψ = 90°, the
parameter is left alone).
"""

from __future__ import annotations

import math

import numpy as np

from .. import polarisation as pol
from .common import make_spec, phi_sphere

__all__ = ["SPECS", "magnetic_shell_terms", "superparamagnetic_terms", "langevin_moments"]

_SRC = "sasfit_ff/"


def _sph(q, r, eta):
    r = max(float(r), 0.0)
    return eta * 4.0 / 3.0 * math.pi * r**3 * phi_sphere(np.asarray(q, float) * r)


def _ksh(q, r, dr, e_c, e_sh, e_m):
    """Amplitude of a core-shell sphere: (η_sh - η_m) V(R+ΔR) Φ + (η_c - η_sh) V(R) Φ (SASfit Ksh)."""
    return _sph(q, r + dr, e_sh - e_m) + _sph(q, r, e_c - e_sh)


def langevin_moments(xi):
    """(S1, S2) = (L(ξ), 1 - 3L(ξ)/ξ) with SASfit's small-ξ branch (|ξ| < 1e-5: L = ξ/3, L/ξ = 1/3)."""
    xi = float(xi)
    if abs(xi) < 1e-5:
        L, LA = xi / 3.0, 1.0 / 3.0
    else:
        L = 1.0 / math.tanh(xi) - 1.0 / xi
        LA = L / xi
    return L, 1.0 - 3.0 * LA


def _psi_and_mode(p, psi, index=8):
    """ψ relative to the field (deg) and the averaging flag, from the active setting or the parameter."""
    cur = pol.current()
    if psi is None:
        psi = p[index]
    if cur is not None:
        if cur.average:
            return None, True
        return np.asarray(psi, float) - cur.psi_H, False
    return np.asarray(psi, float), False


def magnetic_shell_terms(q, p, psi=None):
    """p = [R, dR, eta_nuc_core, eta_nuc_shell, eta_nuc_matrix, eta_mag_core, eta_mag_shell, eta_mag_matrix,
    psi (deg), pol] -> terms (5, nq)."""
    q = np.asarray(q, float)
    r, dr = p[0], p[1]
    kn = _ksh(q, r, dr, p[2], p[3], p[4])
    km = _ksh(q, r, dr, p[5], p[6], p[7])
    ps, avg = _psi_and_mode(p, psi)
    s2, s4 = pol.angular_moments(ps, avg)
    return pol.basis_terms(kn, km, 1.0, 1.0, s2, s4)


def superparamagnetic_terms(q, p, psi=None):
    """p = [R, dR, eta_nuc_core, eta_nuc_shell, eta_nuc_matrix, eta_mag_core, R_av, mu_kT, psi (deg), pol]."""
    q = np.asarray(q, float)
    r, dr = p[0], p[1]
    kn = _ksh(q, r, dr, p[2], p[3], p[4])
    km = _ksh(q, r, dr, p[5], 0.0, 0.0)
    xi = p[7] * (r / p[6]) ** 3 if p[6] > 0 else p[7]
    S1, S2 = langevin_moments(xi)
    ps, avg = _psi_and_mode(p, psi)
    s2, s4 = pol.angular_moments(ps, avg)
    return pol.basis_terms(kn, km, S1, S2, s2, s4)


def _intensity_from(terms_fn, p_index=9):
    def intensity(q, p, psi=None):
        T = terms_fn(q, p, psi)
        cur = pol.current()
        if cur is not None:
            w = cur.weights()
        else:
            w = pol.Polarisation.sasfit_pol_weights(p[p_index])   # SASfit pol: moment labels
        return pol.combine(T, w)

    return intensity


def _volume_cs(x, p, dist=0):
    """Volume of the whole particle 4π/3 (R + ΔR)³ (x replaces p[dist])."""
    pp = np.array(p, float)
    x = np.asarray(x, float)
    r = np.where(dist == 0, x, pp[0])
    dr = np.where(dist == 1, x, pp[1])
    return 4.0 / 3.0 * math.pi * (np.abs(r) + np.abs(dr)) ** 3


_MS = ["R", "dR", "eta_nuc_core", "eta_nuc_shell", "eta_nuc_matrix", "eta_mag_core", "eta_mag_shell",
       "eta_mag_matrix", "psi", "pol"]  # fmt: skip
_MSS = {0: (5.0, 60.0, "log"), 1: (0.0, 20.0, "lin"), 2: (-1.0, 8.0, "lin"), 3: (-1.0, 6.0, "lin"), 4: (-1.0, 6.0, "lin"),
        5: (0.0, 3.0, "lin"), 6: (0.0, 1.0, "lin"), 7: (0.0, 0.5, "lin"), 8: (0.0, 180.0, "lin"), 9: (-1.0, 1.0, "lin")}
_SP = ["R", "dR", "eta_nuc_core", "eta_nuc_shell", "eta_nuc_matrix", "eta_mag_core", "R_av", "mu_kT", "psi", "pol"]
_SPS = {0: (3.0, 30.0, "log"), 1: (0.0, 10.0, "lin"), 2: (-1.0, 8.0, "lin"), 3: (-1.0, 6.0, "lin"), 4: (-1.0, 6.0, "lin"),
        5: (0.0, 3.0, "lin"), 6: (3.0, 30.0, "log"), 7: (0.0, 20.0, "lin"), 8: (0.0, 180.0, "lin"), 9: (-1.0, 1.0, "lin")}


def _spec(name, terms_fn, pars, sample, source, label, notes):
    d = make_spec(name, _intensity_from(terms_fn), pars, sample, volume=_volume_cs, source=_SRC + source,
                  reduced=lambda q, p: q * (abs(p[0]) + abs(p[1])), reduced_label="q (R+ΔR)", rtol=1e-12,
                  label=label, group="magnetic", notes=notes)  # fmt: skip
    inner = d["intensity"]

    def _int(q, p, psi=None, c_compat=False):
        return _intensity_from(terms_fn)(np.asarray(q, float), np.asarray(p, float), psi)

    _int.__name__, _int.__doc__ = inner.__name__, terms_fn.__doc__
    d["intensity"] = _int
    d["psi_index"] = 8
    d["terms"] = terms_fn
    return d


SPECS = [
    _spec("ff_magnetic_shell_psi", magnetic_shell_terms, _MS, _MSS, "magnetic_shell/sasfit_ff_magnetic_shell_psi.c",
          "Magnetic core-shell sphere (saturated)",
          "magnetisation saturated along the field; ψ = angle between q and the field"),
    _spec("ff_superparamagnetic_ff_psi", superparamagnetic_terms, _SP, _SPS,
          "superparamagnetic_ff_psi/sasfit_ff_superparamagnetic_ff_psi.c",
          "Superparamagnetic core-shell sphere (Langevin)",
          "magnetic core, Langevin statistics with ξ = mu_kT (R/R_av)³"),
]
