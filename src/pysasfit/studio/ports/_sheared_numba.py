"""Compiled (Numba) kernel of the sheared-object ports (:mod:`.sheared`): the triple sum over θ, φ and ν.

The quadrature is the same as in the NumPy path of :func:`.sheared.sheared_intensity`; the kernel only replaces
the array expressions by loops, without temporary (θ, φ, ν) arrays, and runs the q points in parallel on all
cores. Results agree with the NumPy path to rounding (~1e-14).

J₁ is the Cephes rational approximation (as in SciPy's ``scipy.special.j1``, xsf/cephes/j1.h, BSD licence), so
the kernel needs no function pointer into SciPy and can be cached on disk.
"""

from __future__ import annotations

import math

import numpy as np

from pysasfit.core.accel import HAVE_NUMBA

__all__ = ["AVAILABLE", "triple_sum"]

AVAILABLE = HAVE_NUMBA

# Cephes j1 coefficients
_RP = np.array([-8.99971225705559398224e8, 4.52228297998194034323e11, -7.27494245221818276015e13,
                3.68295732863852883286e15])  # fmt: skip
_RQ = np.array([6.20836478118054335476e2, 2.56987256757748830383e5, 8.35146791431949253037e7,
                2.21511595479792499675e10, 4.74914122079991414898e12, 7.84369607876235854894e14,
                8.95222336184627338078e16, 5.32278620332680085395e18])  # fmt: skip
_PP = np.array([7.62125616208173112003e-4, 7.31397056940917570436e-2, 1.12719608129684925192e0,
                5.11207951146807644818e0, 8.42404590141772420927e0, 5.21451598682361504063e0,
                1.00000000000000000254e0])  # fmt: skip
_PQ = np.array([5.71323128072548699714e-4, 6.88455908754495404082e-2, 1.10514232634061696926e0,
                5.07386386128601488557e0, 8.39985554327604159757e0, 5.20982848682361821619e0,
                9.99999999999999997461e-1])  # fmt: skip
_QP = np.array([5.10862594750176621635e-2, 4.98213872951233449420e0, 7.58238284132545283818e1,
                3.66779609360150777800e2, 7.10856304998926107277e2, 5.97489612400613639965e2,
                2.11688757100572135698e2, 2.52070205858023719784e1])  # fmt: skip
_QQ = np.array([7.42373277035675149943e1, 1.05644886038262816351e3, 4.98641058337653607651e3,
                9.56231892404756170795e3, 7.99704160447350683650e3, 2.82619278517639096600e3,
                3.36093607810698293419e2])  # fmt: skip
_Z1 = 1.46819706421238932572e1
_Z2 = 4.92184563216946036703e1
_THPIO4 = 2.35619449019234492885
_SQ2OPI = 7.9788456080286535587989e-1
_SQH = math.sqrt(0.5)

if HAVE_NUMBA:
    import numba as _nb

    _njit = _nb.njit(cache=True, fastmath=False, error_model="numpy")

    @_njit
    def j1(x):
        """Bessel function J₁ (Cephes, polynomials unrolled)."""
        sign = 1.0
        if x < 0.0:
            x = -x
            sign = -1.0
        if x <= 5.0:
            z = x * x
            w = (
                (
                    ((-899971225.7055594) * z + 452228297998.19403) * z
                    + -72749424522181.83
                )
                * z
                + 3682957328638529.0
            ) / (
                (
                    (
                        (
                            (
                                (
                                    (
                                        ((1.0) * z + 620.8364781180543) * z
                                        + 256987.25675774884
                                    )
                                    * z
                                    + 83514679.14319493
                                )
                                * z
                                + 22151159547.97925
                            )
                            * z
                            + 4749141220799.914
                        )
                        * z
                        + 784369607876235.9
                    )
                    * z
                    + 8.952223361846274e16
                )
                * z
                + 5.322786203326801e18
            )
            return sign * w * x * (z - _Z1) * (z - _Z2)
        w = 5.0 / x
        z = w * w
        p = (
            (
                (
                    (
                        (
                            ((0.0007621256162081731) * z + 0.07313970569409176) * z
                            + 1.1271960812968493
                        )
                        * z
                        + 5.112079511468076
                    )
                    * z
                    + 8.424045901417724
                )
                * z
                + 5.214515986823615
            )
            * z
            + 1.0
        ) / (
            (
                (
                    (
                        (
                            ((0.0005713231280725487) * z + 0.06884559087544954) * z
                            + 1.105142326340617
                        )
                        * z
                        + 5.073863861286015
                    )
                    * z
                    + 8.399855543276042
                )
                * z
                + 5.209828486823619
            )
            * z
            + 1.0
        )
        q = (
            (
                (
                    (
                        (
                            (
                                ((0.05108625947501766) * z + 4.982138729512334) * z
                                + 75.82382841325453
                            )
                            * z
                            + 366.7796093601508
                        )
                        * z
                        + 710.8563049989261
                    )
                    * z
                    + 597.4896124006136
                )
                * z
                + 211.68875710057213
            )
            * z
            + 25.207020585802372
        ) / (
            (
                (
                    (
                        (
                            (((1.0) * z + 74.23732770356752) * z + 1056.4488603826283)
                            * z
                            + 4986.410583376536
                        )
                        * z
                        + 9562.318924047562
                    )
                    * z
                    + 7997.041604473507
                )
                * z
                + 2826.1927851763908
            )
            * z
            + 336.0936078106983
        )
        if x < 10.0:
            xn = x - _THPIO4
            p = p * math.cos(xn) - w * q * math.sin(xn)
            return sign * p * _SQ2OPI / math.sqrt(x)
        a = _SQH * (w * q - p)
        b = _SQH * (p + w * q)
        return sign * (a * math.cos(x) + b * math.sin(x)) * _SQ2OPI / math.sqrt(x)

    @_njit
    def _lambda1(x):
        if abs(x) < 1e-4:
            return 1.0 - x * x / 8.0
        return 2.0 * j1(x) / x

    @_njit
    def _j0(x):
        if abs(x) < 1e-4:
            return 1.0 - x * x / 6.0
        return math.sin(x) / x

    @_njit
    def _phi_sph(x):
        if abs(x) < 1e-3:
            x2 = x * x
            return 1.0 - x2 / 10.0 + x2 * x2 / 280.0
        return 3.0 * (math.sin(x) - x * math.cos(x)) / (x * x * x)

    @_njit
    def _nu_sum(shape, cosg, qp, nus, w6):
        """Σ_ν w_ν ν⁶ A²(q, γ; ν=1 lengths scaled by ν) for one direction (qp: precomputed q·lengths)."""
        c2 = cosg * cosg
        s2 = 1.0 - c2
        if shape == 0:  # cylinder: qp = qR, q(R+t), qL/2, C_core, C_shell
            sing = math.sqrt(s2) if s2 > 0.0 else 0.0
            xc, xs, xa = qp[0] * sing, qp[1] * sing, qp[2] * cosg
            out = 0.0
            for v in range(nus.shape[0]):
                nu = nus[v]
                a = _j0(nu * xa) * (
                    qp[3] * _lambda1(nu * xc) + qp[4] * _lambda1(nu * xs)
                )
                out += w6[v] * a * a
            return out
        # spheroid: qp = qa, qb, q(a+t), q(b+t), C_core, C_shell
        xc = math.sqrt(qp[0] * qp[0] * c2 + qp[1] * qp[1] * s2)
        xt = math.sqrt(qp[2] * qp[2] * c2 + qp[3] * qp[3] * s2)
        out = 0.0
        for v in range(nus.shape[0]):
            nu = nus[v]
            a = qp[4] * _phi_sph(nu * xc) + qp[5] * _phi_sph(nu * xt)
            out += w6[v] * a * a
        return out

    @_nb.njit(parallel=True, cache=True, error_model="numpy")
    def _kernel(shape, hp, q, psi, th, wth, offs, nphi, nus, wnu, par, kappa):
        n = q.shape[0]
        out = np.empty(n)
        w6 = wnu * nus**6  # A(ν) = ν³ A(lengths ν) for the scaled arguments
        if hp and kappa != 0.0:
            twophi0 = math.atan(8.0 / kappa)
            c2 = math.cos(twophi0)
        else:
            twophi0 = 0.0
            c2 = 0.0
        if shape == 0:
            r, t, length = par[0], par[1], par[2]
            cc = (par[3] - par[4]) * math.pi * r * r * length
            cs = (par[4] - par[5]) * math.pi * (r + t) * (r + t) * length
        else:
            b, t, a = par[0], par[1], par[2]
            cc = (par[3] - par[4]) * 4.0 / 3.0 * math.pi * a * b * b
            cs = (par[4] - par[5]) * 4.0 / 3.0 * math.pi * (a + t) * (b + t) * (b + t)
        for i in _nb.prange(n):
            qi = q[i]
            qp = np.empty(6)
            if shape == 0:
                qp[0], qp[1], qp[2], qp[3], qp[4] = (
                    qi * r,
                    qi * (r + t),
                    0.5 * qi * length,
                    cc,
                    cs,
                )
            else:
                qp[0], qp[1], qp[2], qp[3], qp[4], qp[5] = (
                    qi * a,
                    qi * b,
                    qi * (a + t),
                    qi * (b + t),
                    cc,
                    cs,
                )
            cps, sps = math.cos(psi[i]), math.sin(psi[i])
            m = nphi[i]
            dphi = math.pi / m
            cphi = np.empty(m + 1)
            for k in range(m + 1):
                cphi[k] = math.cos(k * dphi)
            total = 0.0
            for j in range(offs[i], offs[i + 1]):
                ct, st = math.cos(th[j]), math.sin(th[j])
                acc = 0.0
                if hp:
                    base = sps * ct
                    s2 = st * st
                    for k in range(m):
                        side = cps * st * cphi[k]
                        if kappa == 0.0:
                            dens = 0.25 / math.pi
                        else:
                            den = 1.0 - s2 * c2 * math.cos(
                                2.0 * (k * dphi - 0.5 * twophi0)
                            )
                            dens = (
                                (1.0 - c2)
                                * math.sqrt((1.0 + s2 * c2) ** 3)
                                / (4.0 * math.pi * den * den)
                            )
                        acc += dens * (
                            _nu_sum(shape, base + side, qp, nus, w6)
                            + _nu_sum(shape, base - side, qp, nus, w6)
                        )
                else:
                    for k in range(m + 1):
                        cosg = cps * ct + sps * st * cphi[k]
                        acc += (0.5 if (k == 0 or k == m) else 1.0) * _nu_sum(
                            shape, cosg, qp, nus, w6
                        )
                total += wth[j] * acc * dphi
            out[i] = (2.0 if hp else 4.0) * total
        return out


def triple_sum(shape, hp, q, psi, th, wth, offs, nphi, nus, wnu, par, kappa):
    """Σ_θ wth Σ_φ Σ_ν (...) for all points (see :func:`.sheared.sheared_intensity`).

    ``th``/``wth``: θ nodes and weights of all points concatenated (``offs`` = start indices, length n+1);
    ``wth`` already contains sin θ and, for the axially symmetric distributions, the normalised density."""
    return _kernel(0 if shape == "cylinder" else 1, bool(hp), np.ascontiguousarray(q, float),
                   np.ascontiguousarray(psi, float), np.ascontiguousarray(th, float),
                   np.ascontiguousarray(wth, float), np.ascontiguousarray(offs, np.int64),
                   np.ascontiguousarray(nphi, np.int64), np.ascontiguousarray(nus, float),
                   np.ascontiguousarray(wnu, float), np.ascontiguousarray(par[:6], float), float(par[8]))  # fmt: skip
