# SPDX-License-Identifier: GPL-3.0-or-later
#
# Python/numpy port of libformfactor (https://jugit.fz-juelich.de/mlz/libformfactor)
#   libformfactor: efficient and accurate computation of scattering form factors
#   Copyright (C) Forschungszentrum Juelich GmbH 2022, Scientific Computing Group at MLZ
#   Author of the original: Joachim Wuttke
# The mathematics is described in J. Wuttke, J. Appl. Cryst. 54, 580-587 (2021),
# "Numerically stable form factor of any polygon and polyhedron", doi:10.1107/S1600576721001710.
#
# This file is a derivative work of libformfactor and is therefore licensed under the
# GNU General Public License, version 3 or (at your option) any later version.
"""Form factors (Fourier shape transforms) of polygons, prisms and polyhedra.

Vectorised over arrays of scattering vectors ``q`` of shape ``(N, 3)`` (real or complex). All form factors are
for a body with its origin at the point given by the vertex coordinates (the factories in :mod:`._shapes` put it at
the centre of mass), F(q) = ∫ exp(i q·r) dV, so that F(0) = volume.

The switch-overs between the analytic formula and power series follow the C++ library, including the exact
convergence criterion (three consecutive terms below rounding error).
"""

from __future__ import annotations

import math

import numpy as np

__all__ = ["Face", "Polyhedron", "Prism", "Box", "Topology"]

EPS = 2e-16
Q_LIMIT_SERIES = 1e-2  # polyhedron: q R below which the power series is used
QPA_LIMIT_SERIES = 1e-2  # face: in-plane q * radius below which the power series is used
N_LIMIT_SERIES = 20

# reciprocal factorials, built by the same recursion as in the C++ library
_RF = [1.0]
for _n in range(1, 171):
    _RF.append(_RF[-1] / _n)


def _mag2(a):
    """Σ |a_i|² along the last axis."""
    return np.sum(a.real * a.real + a.imag * a.imag, axis=-1) if np.iscomplexobj(a) else np.sum(a * a, axis=-1)


def _powers(z, m):
    """[z^0, ..., z^m] by repeated multiplication."""
    out = [np.ones_like(z)]
    for _ in range(m):
        out.append(out[-1] * z)
    return out


def _sinc(z):
    """sin(z)/z, exactly 1 at z == 0."""
    zs = np.where(z == 0, 1.0, z)
    return np.where(z == 0, 1.0 + 0j, np.sin(zs) / zs)


def _mul_i(z):
    return 1j * z


class Topology:
    """Face list of a polyhedron: ``faces`` is a list of (vertex_indices, symmetry_S2); ``symmetry_Ci`` says that
    the second half of the faces are the inversion images of the first half and are omitted from computations."""

    def __init__(self, faces, symmetry_Ci):
        self.faces = [(tuple(idx), bool(s2)) for idx, s2 in faces]
        self.symmetry_Ci = bool(symmetry_Ci)


class _Edge:
    __slots__ = ("E", "R")

    def __init__(self, v_low, v_high):
        v_low = np.asarray(v_low, float)
        v_high = np.asarray(v_high, float)
        self.E = (v_high - v_low) / 2  # from the middle of the edge to the upper vertex
        self.R = (v_high + v_low) / 2  # position of the middle of the edge

    def contrib(self, m, qpa, qrperp):
        """Σ_{l=0}^{M/2} u^2l v^(M-2l) / ((2l+1)! (M-2l)!) - v_perp^M / M!, with the l=0 and v_perp terms handled
        as in the C++ code."""
        u = qpa @ self.E
        v2 = qpa @ self.R
        v1 = qrperp
        v = v2 + v1
        pu, pv1, pv2, pv = (_powers(z, m) for z in (u, v1, v2, v))
        # v == 0: only 2l == M contributes
        if m & 1:
            res0 = np.zeros_like(v)
        else:
            res0 = _RF[m] * (pu[m] / (m + 1.0) - pv1[m])
        # l = 0 term minus (qperp.R)^M (the latter cancels in the sum over edges)
        binom = np.zeros_like(v)
        for mm in range(1, m + 1):
            binom = binom + _RF[mm] * _RF[m - mm] * pv2[mm] * pv1[m - mm]
        base = np.where(v1 == 0, _RF[m] * pv2[m], np.where(v2 == 0, 0.0, binom))
        tot = base
        for l in range(1, m // 2 + 1):
            tot = tot + _RF[m - 2 * l] * _RF[2 * l + 1] * pu[2 * l] * pv[m - 2 * l]
        return np.where(v == 0, res0, tot)


class Face:
    """A planar polygon with outward oriented vertex chain; ``symmetry_S2`` marks a polygon with a two-fold
    symmetry axis perpendicular to its plane (only half of the edges are then kept)."""

    def __init__(self, vertices, symmetry_S2=False):
        V = np.asarray(vertices, float)
        self.sym_S2 = bool(symmetry_S2)
        nv = len(V)
        if nv == 0:
            raise ValueError("Invalid polyhedral face: no edges given")
        if nv < 3:
            raise ValueError("Invalid polyhedral face: less than three edges")
        diam = 0.0
        for j in range(nv):
            for jj in range(j + 1, nv):
                diam = max(diam, float(np.linalg.norm(V[j] - V[jj])))
        self.radius_2d = diam / 2
        raw = []
        for j in range(nv):
            jj = (j + 1) % nv
            if np.linalg.norm(V[j] - V[jj]) < 1e-14 * self.radius_2d:
                continue  # edge too short
            raw.append(_Edge(V[j], V[jj]))
        ne = len(raw)
        if ne < 3:
            raise ValueError("Face has less than three non-vanishing edges")
        n = np.zeros(3)
        for j in range(ne):
            n += np.cross(raw[j].E, raw[(j + 1) % ne].E)
        nm = np.linalg.norm(n)
        self.normal = n / nm if nm != 0 else np.zeros(3)
        estimate = float(V[0] @ self.normal)
        s1 = 0.0
        for j in range(1, nv):
            s1 += float(V[j] @ self.normal) - estimate
        self.rperp = estimate + s1 / nv
        area = 0.0
        for j in range(nv):
            area += float(self.normal @ np.cross(V[j], V[(j + 1) % nv])) / 2
        self.area = area
        edges = raw
        if self.sym_S2:
            if ne & 1:
                raise ValueError("Invalid polyhedral face: odd #edges violates symmetry S2")
            half = ne // 2
            fc = self.rperp * self.normal
            for j in range(half):
                if np.sum((edges[j].R + edges[j + half].R - 2 * fc) ** 2) > (2e-12 * self.radius_2d) ** 2:
                    raise ValueError("Invalid polyhedral face: edge centers violate symmetry S2")
                if np.linalg.norm(edges[j].E + edges[j + half].E) > 1e-12 * self.radius_2d:
                    raise ValueError("Invalid polyhedral face: edge vectors violate symmetry S2")
            edges = edges[:half]
        self.edges = edges

    # ---------------------------------------------------------------------------------------------- helpers
    @property
    def pyramidal_volume(self):
        return self.rperp * self.area / 3

    def center_of_polygon(self):
        if self.sym_S2:
            return self.normal * self.rperp
        scaled = (2 / (3.0 * self.area)) * self.normal
        mean_r = sum(e.R for e in self.edges) / len(self.edges)
        res = mean_r.copy()
        for e in self.edges:
            res = res + float(np.cross(e.R - mean_r, e.E) @ scaled) * (e.R - mean_r)
        return res

    def assert_ci(self, other):
        if abs(self.rperp - other.rperp) > abs(1e-15 * (self.rperp + other.rperp)):
            raise ValueError("Invalid polyhedron: faces with different distance from origin violate symmetry Ci")
        if abs(self.area - other.area) > 1e-15 * (self.area + other.area):
            raise ValueError("Invalid polyhedron: faces with different areas violate symmetry Ci")
        if np.linalg.norm(self.normal + other.normal) > 1e-14:
            raise ValueError("Invalid polyhedron: faces do not have opposite orientation, violating symmetry Ci")

    def is_planar(self, tol):
        for e in self.edges:
            for v in (e.R - e.E, e.R + e.E):
                if abs(self.normal @ v - self.rperp) > tol:
                    return False
        return True

    def _decompose(self, q):
        """(q_perp, q_parallel): q_perp = n·q, q_pa = q - q_perp n (refined once)."""
        qperp = q @ self.normal
        qpa = q - qperp[:, None] * self.normal
        qpa = qpa - (qpa @ self.normal)[:, None] * self.normal
        tiny = np.sqrt(_mag2(qpa)) < EPS * np.abs(qperp)
        qpa = np.where(tiny[:, None], 0.0, qpa)
        return qperp, qpa

    # ------------------------------------------------------------------------------------------ series terms
    def _ff_n_core(self, m, qpa, qperp):
        prevec = 2.0 * np.cross(self.normal, qpa)
        qrperp = qperp * self.rperp
        res = np.zeros(len(qpa), complex)
        for e in self.edges:
            vfac = np.conj(prevec) @ e.E
            res = res + vfac * e.contrib(m + 1, qpa, qrperp)
        return res

    def ff_n(self, n, q):
        """Contribution q·n f_n of order q^(n+1) of this face (polyhedron power series)."""
        qn = np.conj(q) @ self.normal
        qmag = np.sqrt(_mag2(q))
        zero = np.abs(qn) < EPS * qmag
        qperp, qpa = self._decompose(q)
        qpa2 = _mag2(qpa)
        core = self._ff_n_core(n, qpa, qperp)
        if self.sym_S2:
            core = core + self._ff_n_core(n, -qpa, qperp)
        main = qn * core / np.where(qpa2 == 0, 1.0, qpa2)
        special = qn * (qperp * self.rperp) ** n * self.area * _RF[n]
        return np.where(zero, 0.0, np.where(qpa2 == 0, special, main))

    def _expansion(self, fac_even, fac_odd, qpa, abslevel):
        """Σ_{n>=1} terms of the q_pa expansion of the 2d form factor, with the C++ stopping rule."""
        N = len(qpa)
        total = np.zeros(N, complex)
        count = np.zeros(N, int)
        done = np.zeros(N, bool)
        n_fac = 1j
        qpa2 = _mag2(qpa)
        for n in range(1, N_LIMIT_SERIES):
            act = ~done
            if not act.any():
                break
            fac = fac_odd if n & 1 else fac_even
            fac = np.broadcast_to(fac, (N,))[act]
            core = self._ff_n_core(n, qpa[act], np.zeros(act.sum(), complex))
            term = n_fac * fac * core / qpa2[act]
            total[act] += term
            conv = (np.abs(term) <= EPS * np.abs(total[act])) | (np.abs(total[act]) < EPS * np.broadcast_to(abslevel, (N,))[act])
            c = np.where(conv, count[act] + 1, 0)
            count[act] = c
            fin = c > 2
            idx = np.flatnonzero(act)
            done[idx[fin]] = True
            n_fac = _mul_i(n_fac)
        if not done.all():
            raise RuntimeError("Numeric error in polyhedral face: series f(q_pa) not converged")
        return total

    def _edge_sum_ff(self, q, qpa, sym_ci):
        prevec = np.cross(self.normal, qpa)
        total = np.zeros(len(qpa), complex)
        vfacsum = np.zeros(len(qpa), complex)
        ne = len(self.edges)
        for i, e in enumerate(self.edges):
            qE = qpa @ e.E
            qR = qpa @ e.R
            if self.sym_S2:
                rfac = np.sin(qR)
            elif sym_ci:
                rfac = np.cos(q @ e.R)
            else:
                rfac = np.exp(1j * qR)
            if self.sym_S2 or i < ne - 1:
                vfac = np.conj(prevec) @ e.E
                vfacsum = vfacsum + vfac
            else:
                vfac = -vfacsum
            total = total + vfac * _sinc(qE) * rfac
        return total

    # ------------------------------------------------------------------------------------------ form factors
    def ff(self, q, sym_ci):
        """Contribution of this face to the polyhedral form factor."""
        qperp, qpa = self._decompose(q)
        qpa_red = self.radius_2d * np.sqrt(_mag2(qpa))
        qr_perp = qperp * self.rperp
        if sym_ci:
            ff0 = 2j * np.sin(qr_perp) * self.area
        else:
            ff0 = np.exp(1j * qr_perp) * self.area
        out = ff0.astype(complex)
        series = (qpa_red != 0) & (qpa_red < QPA_LIMIT_SERIES) & (not self.sym_S2)
        direct = (qpa_red != 0) & ~series
        if series.any():
            if sym_ci:
                fe = 2.0 * _mul_i(np.sin(qr_perp[series]))
                fo = 2.0 * np.cos(qr_perp[series])
            else:
                fe = np.exp(1j * qr_perp[series])
                fo = fe
            out[series] = ff0[series] + self._expansion(fe, fo, qpa[series], np.abs(ff0[series]))
        if direct.any():
            qd, qpad, qr = q[direct], qpa[direct], qr_perp[direct]
            if self.sym_S2:
                prefac = -8.0 * np.sin(qr) if sym_ci else 4.0j * np.exp(1j * qr)
            else:
                prefac = 4.0 + 0j if sym_ci else 2.0 * np.exp(1j * qr)
            out[direct] = prefac * self._edge_sum_ff(qd, qpad, sym_ci) / (1j * _mag2(qpad))
        return out

    def ff_2d(self, qpa):
        """Two-dimensional form factor, for use in a prism."""
        qpa = np.asarray(qpa, complex)
        if np.any(np.abs(np.conj(qpa) @ self.normal) > EPS * np.sqrt(_mag2(qpa))):
            raise RuntimeError("Numeric error in polyhedral formfactor: ff_2D called with perpendicular q component")
        qpa_red = self.radius_2d * np.sqrt(_mag2(qpa))
        out = np.full(len(qpa), self.area, complex)
        series = (qpa_red != 0) & (qpa_red < QPA_LIMIT_SERIES) & (not self.sym_S2)
        direct = (qpa_red != 0) & ~series
        if series.any():
            out[series] = self.area + self._expansion(1.0, 1.0, qpa[series], abs(self.area))
        if direct.any():
            qd = qpa[direct]
            pre = 4.0 if self.sym_S2 else 2.0 / 1j
            out[direct] = pre * self._edge_sum_ff(qd, qd, False) / _mag2(qd)
        return out


class Polyhedron:
    """Convex polyhedron given by vertices and face topology."""

    def __init__(self, topology, vertices):
        self.topology = topology
        self.vertices = np.asarray(vertices, float)
        faces_t = topology.faces
        if len(faces_t) < 4:
            raise ValueError("Invalid polyhedron: less than four non-vanishing faces")
        self.sym_ci = topology.symmetry_Ci
        V = self.vertices
        self.faces = []
        if self.sym_ci:
            nf = len(faces_t)
            if nf & 1:
                raise ValueError("Invalid polyhedron: odd #faces violates symmetry Ci")
            for k in range(nf // 2):
                idx, s2 = faces_t[k]
                face = Face(V[list(idx)], s2)
                idxc, _ = faces_t[nf - 1 - k]
                face.assert_ci(Face(V[list(idxc)], s2))
                self.faces.append(face)
        else:
            for idx, s2 in faces_t:
                self.faces.append(Face(V[list(idx)], s2))
        self.radius = float(np.max(np.linalg.norm(V, axis=1)))
        vol = sum(f.pyramidal_volume for f in self.faces)
        self.volume = 2 * vol if self.sym_ci else vol

    def surface_area(self):
        a = sum(f.area for f in self.faces)
        return 2 * a if self.sym_ci else a

    def center_of_mass(self):
        if self.sym_ci:
            return np.zeros(3)
        res = np.zeros(3)
        for f in self.faces:
            res += f.center_of_polygon() * f.pyramidal_volume
        return res * (3.0 / 4) / self.volume

    def is_convex(self, reltol=1e-12):
        tol = reltol * 2 * self.radius
        for f in self.faces:
            if not f.is_planar(tol):
                return False
            for v in self.vertices:
                if f.normal @ v > f.rperp + tol:
                    return False
                if self.sym_ci and -(f.normal @ v) > f.rperp + tol:
                    return False
        return True

    def _series(self, q, qmag2):
        """Power series of F(q) for q R < 1e-2: volume + Σ_n terms."""
        N = len(q)
        total = np.zeros(N, complex)
        count = np.zeros(N, int)
        done = np.zeros(N, bool)
        result = np.zeros(N, complex)
        n_fac = ((-2.0 if self.sym_ci else -1.0) / qmag2).astype(complex)
        for n in range(2, N_LIMIT_SERIES):
            if self.sym_ci and (n & 1):
                continue
            act = ~done
            if not act.any():
                break
            qa = q[act]
            term = np.zeros(act.sum(), complex)
            for f in self.faces:
                term = term + f.ff_n(n + 1, qa)
            term = term * n_fac[act]
            total[act] += term
            conv = (np.abs(term) <= EPS * np.abs(total[act])) | (np.abs(total[act]) < EPS * self.volume)
            c = np.where(conv, count[act] + 1, 0)
            count[act] = c
            idx = np.flatnonzero(act)
            fin = c > 2
            result[idx[fin]] = self.volume + total[idx[fin]]
            done[idx[fin]] = True
            n_fac = -n_fac if self.sym_ci else _mul_i(n_fac)
        if not done.all():
            raise RuntimeError("Numeric failure in polyhedron: series F(q) not converged")
        return result

    def ff(self, q):
        """F(q) for scattering vectors q of shape (N, 3) (real or complex)."""
        q = np.atleast_2d(np.asarray(q))
        q = q.astype(complex)
        qmag2 = _mag2(q)
        qmag = np.sqrt(qmag2)
        q_red = self.radius * qmag
        out = np.zeros(len(q), complex)
        zero = q_red == 0
        ser = (~zero) & (q_red < Q_LIMIT_SERIES)
        dire = (~zero) & ~ser
        out[zero] = self.volume
        if ser.any():
            out[ser] = self._series(q[ser], qmag2[ser])
        if dire.any():
            qd = q[dire]
            total = np.zeros(len(qd), complex)
            qm = qmag[dire]
            for f in self.faces:
                qn = np.conj(qd) @ f.normal
                skip = np.abs(qn) < EPS * qm
                term = qn * f.ff(qd, self.sym_ci)
                total = total + np.where(skip, 0.0, term)
            out[dire] = total / 1j / qmag2[dire]
        return out

    # BornAgain-like interface shared with Prism and Box
    __call__ = ff


class Prism:
    """Straight prism with polygonal base (in the z=0 plane before centring), height along the base normal."""

    def __init__(self, symmetry_ci, height, base_vertices):
        self.height = float(height)
        self.base_vertices = np.asarray(base_vertices, float)
        self.base = Face(self.base_vertices, symmetry_ci)
        self.radius = math.hypot(self.height / 2, self.base.radius_2d)
        self.volume = self.height * self.base.area
        self.sym_ci = bool(symmetry_ci)

    def surface_area(self):
        per = 0.0
        for e in self.base.edges:
            per += 2 * np.linalg.norm(e.E)
        if self.base.sym_S2:
            per *= 2
        return 2 * self.base.area + per * self.height

    def ff(self, q):
        q = np.atleast_2d(np.asarray(q)).astype(complex)
        qxy = q.copy()
        qxy[:, 2] = 0.0
        return self.height * _sinc(self.height / 2 * q[:, 2]) * self.base.ff_2d(qxy)

    __call__ = ff


class Box:
    """Cuboid with edges along the coordinate axes, centred at the origin."""

    def __init__(self, length, width, height):
        self.length, self.width, self.height = float(length), float(width), float(height)
        self.volume = self.length * self.width * self.height
        self.radius = 0.5 * math.sqrt(self.length**2 + self.width**2 + self.height**2)
        self.sym_ci = True

    def surface_area(self):
        a, b, c = self.length, self.width, self.height
        return 2 * (a * b + b * c + a * c)

    def ff(self, q):
        q = np.atleast_2d(np.asarray(q)).astype(complex)
        return self.volume * _sinc(self.length / 2 * q[:, 0]) * _sinc(self.width / 2 * q[:, 1]) * _sinc(self.height / 2 * q[:, 2])

    __call__ = ff
