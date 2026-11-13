r"""Hankel transforms for SESANS and multiple small-angle scattering (MSAS) in pySASstudio.

This is the Python counterpart of the ``transform`` menu of SASfit (``sasfit_old/sasfit.c``:
``Gztransform``, ``imMSAStransform``, ``integral_IQ_incl_Gztransform``):

=================  =========================================================================================
SASfit             here (:func:`make_transform`)
=================  =========================================================================================
``1``              ``None`` -- the scattering intensity I(Q)
``H{...}/2pi``     ``"sesans"`` -- G(δ) - G(0) = 1/(2π) ∫ I(Q) [J0(Qδ) - 1] Q dQ
``Exp(H{...}/2pi)`` ``"sesans_p"`` -- P/P0 = exp(λ² t [G(δ) - G(0)])  (λ = t = 1 gives SASfit's exp(G(δ) - G(0)))
``MSAS``           ``"msas"`` -- intensity with multiple small-angle scattering
=================  =========================================================================================

Units. SASfit functions carry none, so the user keeps them consistent (as in SASfit): with I(Q) in cm⁻¹ the sample
thickness ``thickness`` is in cm and the wavelength ``wavelength`` is in the reciprocal unit of Q (nm for Q in nm⁻¹,
Å for Q in Å⁻¹). G(δ) - G(0) then has the unit cm⁻¹ · [Q]² (Å⁻²cm⁻¹ for Q in Å⁻¹, the SasView / ``.ses`` convention)
and λ² t G is dimensionless.

SESANS (Rekveldt 1996; Andersson et al., J. Appl. Cryst. 41, 868 (2008)): the measured polarisation is
P(δ)/P0 = exp(λ² t [G(δ) - G(0)]), and G(0) λ² t = Σ t is the total small-angle scattering probability.

MSAS (Schelten & Schmatz, J. Appl. Cryst. 13, 385 (1980); SASfit ``imMSAStransform``). With
φ(r) = λ² t G(r), τ = φ(0) = Σ t and k0 = 2π/λ, the scattered intensity of a slab of thickness t is

    I_MSAS(Q) = c · k0² / (2π t) ∫ [exp(φ(r)) - 1] J0(Q r) r dr,

c = exp(-τ) when the data are normalised to the incident beam (SASfit's MSAS) and c = 1 when they are normalised to
the measured transmission, which already contains the loss exp(-τ) by small-angle scattering (``norm``). The single-
scattering term is split off and evaluated directly,

    exp(φ) - 1 = φ + ψ,   ψ = exp(φ) - 1 - φ,   I_MSAS = c [ I(Q) + k0²/(2π t) ∫ ψ(r) J0(Q r) r dr ],

so that only the (smooth, fast decaying) multiple-scattering part goes through the numerical transforms.

Numerics. Both transforms use the same quadrature: the integrand f(k) k is linear between the nodes of a logarithmic
grid and every interval is integrated *exactly* against J0 (∫J0 by Struve functions, ∫k J0 = k J1 / x). This is a
Filon-type rule: the oscillation of J0 needs no sampling, only I(Q) (or ψ(r)) itself must be resolved by the grid.
The weights depend only on the grid and the output points, so they are computed once and every evaluation during a
fit is one matrix-vector product. Below the first node f is taken as constant (Guinier plateau); above the last node
an optional power-law tail f ∝ k^-p (default p = 4, Porod) completes the zero-order moment ∫ f k dk.
"""

from __future__ import annotations

import math
from collections import OrderedDict

import numpy as np
from scipy import special

__all__ = [
    "hankel_weights", "log_grid", "SESANSTransform", "MSASTransform", "make_transform", "TRANSFORM_KINDS",
    "transform_label", "sphere_sesans",
]  # fmt: skip

TRANSFORM_KINDS = ("none", "sesans", "sesans_p", "msas")
_LABELS = {
    "none": "1   I(Q)",
    "sesans": "H{…}/2π   SESANS  G(δ)−G(0)",
    "sesans_p": "Exp(λ²t·H{…}/2π)   SESANS  P/P0",
    "msas": "MSAS   multiple small-angle scattering",
}


def transform_label(kind):
    return _LABELS.get(kind or "none", str(kind))


# --------------------------------------------------------------------------- quadrature
def _int_j0(u):
    """∫_0^u J0(t) dt (u >= 0)."""
    return special.itj0y0(np.asarray(u, float))[0]


def log_grid(lo, hi, per_decade):
    """Logarithmic grid from ``lo`` to ``hi`` with ``per_decade`` points per decade (at least 8 points)."""
    n = max(int(math.ceil(math.log10(hi / lo) * per_decade)) + 1, 8)
    return np.geomspace(lo, hi, n)


_TAIL_TABLES = {}


def _tail_table(p):
    """∫_1^∞ s^(1-p) J0(s u) ds on a table of u in [0, 60] (computed once per p)."""
    if p not in _TAIL_TABLES:
        from scipy.integrate import quad

        u = np.concatenate([[0.0], np.geomspace(1e-4, 60.0, 600)])
        v = np.empty(u.size)
        v[0] = 1.0 / (p - 2)
        import warnings

        for i, ui in enumerate(u[1:], 1):
            # beyond s = S the rest is bounded by S^(1.5-p) / u^0.5 (negligible for S = 1 + 400/u)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                v[i] = quad(lambda s: s ** (1 - p) * special.j0(s * ui), 1.0, 1.0 + 400.0 / ui, limit=800)[0]
        _TAIL_TABLES[p] = (u, v)
    return _TAIL_TABLES[p]


def _tail_j0(u, p):
    """∫_1^∞ s^(1-p) J0(s u) ds for p > 2 (power-law tail of the integrand, in units of k_N); 0 for u > 60."""
    tu, tv = _tail_table(float(p))
    u = np.asarray(u, float)
    return np.where(u > 60.0, 0.0, np.interp(u, tu, tv))


def hankel_weights(x, k, kernel="j0", tail_low=True, tail_power=None, k_upper=None):
    """Weights ``W`` with ``W @ f(k) ≈ ∫_0^∞ f(k) K(k x) k dk`` for all ``x`` (a matrix len(x) × len(k)).

    ``kernel``: ``"j0"`` (K = J0) or ``"j0m1"`` (K = J0 - 1, the SESANS kernel; finite also for f ∝ k^-p, 2 < p).
    ``f(k) k`` is taken as linear between the nodes ``k`` (increasing, > 0) and each interval is integrated exactly.
    ``tail_low``: f(k) = f(k_0) on [0, k_0]. ``tail_power`` = p > 2: f(k) = f(k_N) (k/k_N)^-p above the last node.
    ``k_upper`` (a number or one per x): the integral of that row ends there (instrument resolution / acceptance of
    SESANS). The limits should be nodes of ``k``: intervals above the limit are dropped, a row with a limit below the
    last node gets no power-law tail.
    """
    x = np.atleast_1d(np.asarray(x, float))
    k = np.asarray(k, float)
    X = x[:, None]
    a, b = k[:-1], k[1:]
    h = b - a
    with np.errstate(divide="ignore", invalid="ignore"):
        # A0 = ∫_a^b J0(kx) dk, A1 = ∫_a^b k J0(kx) dk
        IJ = _int_j0(k[None, :] * X)
        KJ1 = k[None, :] * special.j1(k[None, :] * X)
        A0 = np.where(X > 0, (IJ[:, 1:] - IJ[:, :-1]) / X, h[None, :])
        A1 = np.where(X > 0, (KJ1[:, 1:] - KJ1[:, :-1]) / X, 0.5 * (b**2 - a**2)[None, :])
    # g(k) = f(k) k linear: ∫ g J0 = g_a (b A0 - A1)/h + g_b (A1 - a A0)/h
    wa = (b[None, :] * A0 - A1) / h[None, :]
    wb = (A1 - a[None, :] * A0) / h[None, :]
    if kernel == "j0m1":  # subtract ∫ g dk = h (g_a + g_b) / 2
        wa = wa - 0.5 * h[None, :]
        wb = wb - 0.5 * h[None, :]
    elif kernel != "j0":
        raise ValueError("kernel must be 'j0' or 'j0m1'")
    clipped = np.zeros(x.size, bool)
    if k_upper is not None:
        up = np.broadcast_to(np.asarray(k_upper, float), x.shape)
        keep = b[None, :] <= up[:, None] * (1 + 1e-9)
        wa, wb = wa * keep, wb * keep
        clipped = up < k[-1] * (1 - 1e-9)
    W = np.zeros((x.size, k.size))
    W[:, :-1] += wa * a[None, :]  # the weights act on f, g = f k
    W[:, 1:] += wb * b[None, :]
    if tail_low:
        k0 = k[0]
        with np.errstate(divide="ignore", invalid="ignore"):
            t = np.where(x > 0, k0 * special.j1(k0 * x) / x, 0.5 * k0**2)
        if kernel == "j0m1":
            t = t - 0.5 * k0**2
        W[:, 0] += t
    if tail_power is not None:
        p = float(tail_power)
        if p <= 2:
            raise ValueError("tail_power must be > 2")
        kN = k[-1]
        plain = kN**2 / (p - 2)
        osc = kN**2 * _tail_j0(kN * x, p)
        W[:, -1] += np.where(clipped, 0.0, (osc - plain) if kernel == "j0m1" else osc)
    return W


class _Cache:
    """A few weight matrices, keyed by the output points and the grid settings."""

    def __init__(self, size=8):
        self.size = size
        self.d = OrderedDict()

    def get(self, key, make):
        if key in self.d:
            self.d.move_to_end(key)
            return self.d[key]
        v = make()
        self.d[key] = v
        while len(self.d) > self.size:
            self.d.popitem(last=False)
        return v


def _particles(model, q):
    """Scattering of the particles of a ModelFunction (no background, no resolution)."""
    return np.asarray(model.particles(q), dtype=float)


# --------------------------------------------------------------------------- SESANS
class SESANSTransform:
    """SESANS: G(δ) - G(0) (``observable="G"``, SASfit ``H{...}/2pi``) or P/P0 = exp(λ² t [G(δ) - G(0)])
    (``observable="P"``, SASfit ``Exp(H{...}/2pi)``) at the spin-echo lengths δ given instead of q.

    ``thickness`` t (cm for I in cm⁻¹) and ``wavelength`` λ (unit of δ) are needed for P/P0 only; ``wavelength`` may be
    an array with one value per spin-echo length ``wavelength_delta`` (time-of-flight SESANS), interpolated in log δ.
    Resolution: SESANS measures only neutrons scattered up to the acceptance of the instrument, so both Q integrals
    (of J0 and of the 1 in J0 - 1) end at Q_max -- SASfit takes the resolution column for it. ``q_accept`` is Q_max: a
    number, or one value per spin-echo length ``accept_delta`` (default ``wavelength_delta``), interpolated in log δ.
    ``theta_max`` (rad, e.g. Theta_zmax of a ``.ses`` file) gives Q_max = 2π sin(θ_max) / λ(δ) instead, which varies
    with δ in time-of-flight SESANS. The limits are inserted as grid nodes, so the cut is exact. The Q grid runs from ``q_min_factor`` / δ_max to ``q_max_factor`` / δ_min (or ``q_range``) with
    ``per_decade`` points per decade; above it a Porod tail (``tail_power``, None = off) completes G(0).
    The model background is an offset added to G(δ) - G(0) (zero in an ideal measurement); the power-law term of the
    background is not used.
    """

    def __init__(self, observable="G", thickness=1.0, wavelength=1.0, wavelength_delta=None, q_accept=None,
                 per_decade=64, q_min_factor=1e-2, q_max_factor=60.0, q_range=None, tail_power=4.0, theta_max=None,
                 accept_delta=None):  # fmt: skip
        if observable not in ("G", "P"):
            raise ValueError("observable must be 'G' (G(δ)-G(0)) or 'P' (P/P0)")
        self.observable = observable
        self.thickness = float(thickness)
        self.wavelength = wavelength if np.ndim(wavelength) else float(wavelength)
        self.wavelength_delta = None if wavelength_delta is None else np.asarray(wavelength_delta, float)
        if q_accept is None or (np.ndim(q_accept) == 0 and not q_accept):
            self.q_accept = None
        else:
            self.q_accept = np.asarray(q_accept, float) if np.ndim(q_accept) else float(q_accept)
        self.accept_delta = None if accept_delta is None else np.asarray(accept_delta, float)
        self.theta_max = None if not theta_max else float(theta_max)
        self.per_decade = int(per_decade)
        self.q_min_factor = float(q_min_factor)
        self.q_max_factor = float(q_max_factor)
        self.q_range = None if q_range is None else tuple(map(float, q_range))
        self.tail_power = None if tail_power is None else float(tail_power)
        self._cache = _Cache()

    @property
    def kind(self):  # "sesans" or "sesans_p"
        return "sesans" if self.observable == "G" else "sesans_p"

    @property
    def linear(self):
        """G(δ) - G(0) is linear in I(Q) (variable projection works); P/P0 is not."""
        return self.observable == "G"

    domain = "delta"

    @staticmethod
    def _per_point(values, ref, delta, what):
        values = np.asarray(values, float)
        if ref is None or ref.size != values.size:
            if values.size == np.size(delta):
                return values
            raise ValueError(f"SESANS: {what} per point needs the spin-echo lengths it belongs to")
        o = np.argsort(ref)
        return np.interp(np.log(np.asarray(delta, float)), np.log(ref[o]), values[o])

    def q_max(self, delta):
        """Q_max (acceptance) at each spin-echo length, or None (no limit)."""
        delta = np.asarray(delta, float)
        if self.theta_max:
            return 2 * np.pi * math.sin(self.theta_max) / self.lam(delta)
        if self.q_accept is None:
            return None
        if np.ndim(self.q_accept) == 0:
            return np.full(delta.shape, float(self.q_accept))
        ref = self.accept_delta if self.accept_delta is not None else self.wavelength_delta
        return self._per_point(self.q_accept, ref, delta, "q_accept")

    def grid(self, delta, qmax=None):
        d = np.asarray(delta, float)
        d = d[np.isfinite(d) & (d > 0)]
        if self.q_range is not None:
            lo, hi = self.q_range
        else:
            if d.size == 0:
                raise ValueError("SESANS: no spin-echo length > 0")
            lo, hi = self.q_min_factor / d.max(), self.q_max_factor / d.min()
        tail = self.tail_power
        if qmax is None:
            qmax = self.q_max(delta)
        if qmax is not None:
            hi = float(np.nanmax(qmax))  # the grid ends at the largest acceptance: nothing is measured beyond
            lo = min(lo, float(np.nanmin(qmax)) / 10)
            tail = None
            k = log_grid(lo, hi, self.per_decade)
            extra = np.unique(qmax[(qmax > lo) & (qmax < hi)])
            k = np.unique(np.concatenate([k, extra]))  # the limits are nodes: the cut is exact
            return k, tail
        return log_grid(lo, hi, self.per_decade), tail

    def weights(self, delta):
        delta = np.asarray(delta, float)
        qmax = self.q_max(delta)
        k, tail = self.grid(delta, qmax)
        key = (delta.tobytes(), k[0], k[-1], k.size, tail, None if qmax is None else qmax.tobytes())
        W = self._cache.get(key, lambda: hankel_weights(np.abs(delta), k, "j0m1", True, tail, qmax) / (2 * np.pi))
        return k, W

    def lam(self, delta):
        """λ at the spin-echo lengths ``delta``."""
        if np.ndim(self.wavelength) == 0:
            return np.full(np.shape(delta), float(self.wavelength))
        return self._per_point(self.wavelength, self.wavelength_delta, delta, "wavelength")

    def g_of(self, delta, Ik):
        """G(δ) - G(0) from I on the grid of :meth:`weights`."""
        _k, W = self.weights(delta)
        return W @ Ik

    def finish(self, delta, G):
        """G(δ) - G(0) (+ offset) -> the observable."""
        if self.observable == "G":
            return G
        return np.exp(self.lam(delta) ** 2 * self.thickness * G)

    def evaluate(self, model, delta):
        delta = np.asarray(delta, dtype=float)
        k, W = self.weights(delta)
        bg = getattr(model, "background", 0.0)
        G = W @ _particles(model, k) + (float(bg) if np.isscalar(bg) and bg is not None else 0.0)
        return self.finish(delta, G)

    def parts(self, model, delta, intensities):
        """Linear transform of contribution intensities given on the grid (``observable="G"`` only)."""
        _k, W = self.weights(delta)
        return [W @ np.asarray(v, float) for v in intensities]

    def total_scattering(self, model, delta):
        """Σ t = λ² t G(0) (the scattering probability; λ at the largest δ)."""
        k, tail = self.grid(delta)
        qm = self.q_max(np.atleast_1d(np.max(delta)))
        W0 = hankel_weights([0.0], k, "j0", True, tail, None if qm is None else qm[:1]) / (2 * np.pi)
        G0 = float(W0 @ _particles(model, k))
        lam = float(np.max(self.lam(np.atleast_1d(np.max(delta)))))
        return lam**2 * self.thickness * G0

    def settings(self):
        d = {"kind": self.kind, "thickness": self.thickness, "per_decade": self.per_decade,
             "q_min_factor": self.q_min_factor, "q_max_factor": self.q_max_factor, "tail_power": self.tail_power}  # fmt: skip
        d["wavelength"] = np.asarray(self.wavelength, float).tolist() if np.ndim(self.wavelength) else self.wavelength
        if self.wavelength_delta is not None:
            d["wavelength_delta"] = self.wavelength_delta.tolist()
        if self.q_accept is not None:
            d["q_accept"] = np.asarray(self.q_accept).tolist() if np.ndim(self.q_accept) else self.q_accept
        if self.accept_delta is not None:
            d["accept_delta"] = self.accept_delta.tolist()
        if self.theta_max:
            d["theta_max"] = self.theta_max
        if self.q_range:
            d["q_range"] = list(self.q_range)
        return d

    def __repr__(self):
        lam = "per point" if np.ndim(self.wavelength) else f"{self.wavelength:g}"
        return f"SESANSTransform(observable={self.observable!r}, thickness={self.thickness:g}, wavelength={lam})"


# --------------------------------------------------------------------------- MSAS
class MSASTransform:
    """Multiple small-angle scattering of a slab: the intensity I_MSAS(Q) (see the module docstring).

    ``thickness`` t (cm for I in cm⁻¹), ``wavelength`` λ (reciprocal unit of Q). ``norm``: ``"incident"`` (the data are
    normalised to the incident beam, factor exp(-Σt), as SASfit's MSAS) or ``"transmission"`` (normalised to the
    measured transmission, which contains exp(-Σt)). Q grid: ``q_min_factor`` · Q_min ... ``q_max_factor`` · Q_max
    (or ``q_range``) with ``per_decade`` points per decade (the model is evaluated there); r grid: ``r_min_factor`` /
    k_max ... ``r_max_factor`` / k_min with ``r_per_decade`` points per decade (only exp(φ) is evaluated there, but the
    back transform needs the fine grid for a large dynamic range of I(Q)). The background is added after the multiple-scattering transform.
    """

    kind = "msas"
    domain = "q"
    linear = False

    def __init__(self, thickness=0.1, wavelength=0.6, norm="incident", per_decade=48, r_per_decade=384,
                 q_min_factor=1e-2, q_max_factor=30.0, q_range=None, r_min_factor=0.3, r_max_factor=30.0,
                 tail_power=4.0):  # fmt: skip
        if norm not in ("incident", "transmission"):
            raise ValueError("norm must be 'incident' (SASfit) or 'transmission'")
        self.thickness = float(thickness)
        self.wavelength = float(wavelength)
        self.norm = norm
        self.per_decade = int(per_decade)
        self.r_per_decade = int(r_per_decade)
        self.q_min_factor = float(q_min_factor)
        self.q_max_factor = float(q_max_factor)
        self.q_range = None if q_range is None else tuple(map(float, q_range))
        self.r_min_factor = float(r_min_factor)
        self.r_max_factor = float(r_max_factor)
        self.tail_power = None if tail_power is None else float(tail_power)
        self._cache = _Cache()
        self.last_tau = None  # Σ t of the last evaluation

    def grids(self, q):
        q = np.asarray(q, float)
        qq = q[np.isfinite(q) & (q > 0)]
        if self.q_range is not None:
            klo, khi = self.q_range
        else:
            if qq.size == 0:
                raise ValueError("MSAS: no q > 0")
            klo, khi = self.q_min_factor * qq.min(), self.q_max_factor * qq.max()
        k = log_grid(klo, khi, self.per_decade)
        r = log_grid(self.r_min_factor / khi, self.r_max_factor / klo, self.r_per_decade)
        return k, r

    def weights(self, q):
        q = np.asarray(q, float)
        k, r = self.grids(q)
        tail = self.tail_power

        def make():
            Wkr = hankel_weights(np.concatenate([[0.0], r]), k, "j0", True, tail) / (2 * np.pi)  # I(k) -> G(0), G(r)
            Wrq = hankel_weights(np.abs(q), r, "j0", True, None)  # ψ(r) -> ∫ ψ J0(Q r) r dr
            return Wkr, Wrq

        key = (q.tobytes(), k[0], k[-1], k.size, r[0], r[-1], r.size, tail)
        return k, r, self._cache.get(key, make)

    def multiple(self, q, Ik, I_single):
        """I_MSAS at ``q`` from I on the k grid and the single scattering I(q) at ``q`` (no background)."""
        _k, _r, (Wkr, Wrq) = self.weights(q)
        G = Wkr @ Ik
        lt = self.wavelength**2 * self.thickness
        tau = lt * G[0]
        self.last_tau = float(tau)
        if not np.isfinite(tau) or tau > 100:
            raise ValueError(f"MSAS: Σt = λ²t·G(0) = {tau:.4g} is too large (check thickness, wavelength and the absolute "
                             "scale of I(Q))")
        phi = lt * G[1:]
        psi = np.expm1(phi) - phi
        Iq = I_single + 2 * np.pi / lt * (Wrq @ psi)
        if self.norm == "incident":
            Iq = Iq * math.exp(-tau)
        return Iq

    def evaluate_particles(self, model, q):
        k, _r, _w = self.weights(q)
        return self.multiple(q, _particles(model, k), _particles(model, q))

    def settings(self):
        d = {"kind": "msas", "thickness": self.thickness, "wavelength": self.wavelength, "norm": self.norm,
             "per_decade": self.per_decade, "r_per_decade": self.r_per_decade, "q_min_factor": self.q_min_factor,
             "q_max_factor": self.q_max_factor, "r_min_factor": self.r_min_factor, "r_max_factor": self.r_max_factor, "tail_power": self.tail_power}  # fmt: skip
        if self.q_range:
            d["q_range"] = list(self.q_range)
        return d

    def __repr__(self):
        return f"MSASTransform(thickness={self.thickness:g}, wavelength={self.wavelength:g}, norm={self.norm!r})"


def make_transform(kind=None, **settings):
    """A transform from its kind (:data:`TRANSFORM_KINDS`, also the SASfit names) and settings; ``None`` for I(Q)."""
    if isinstance(kind, dict):
        settings, kind = {**kind, **settings}, kind.get("kind")
    settings = {k: v for k, v in settings.items() if k != "kind"}
    kind = {"1": "none", "H{...}/2pi": "sesans", "Exp(H{...}/2pi)": "sesans_p", "MSAS": "msas", None: "none",
            "": "none"}.get(kind, kind)  # fmt: skip
    if kind == "none":
        return None
    if kind in ("sesans", "sesans_p"):
        keep = ("thickness", "wavelength", "wavelength_delta", "q_accept", "per_decade", "q_min_factor", "q_max_factor",
                "q_range", "tail_power", "theta_max", "accept_delta")  # fmt: skip
        return SESANSTransform("G" if kind == "sesans" else "P", **{k: v for k, v in settings.items() if k in keep})
    if kind == "msas":
        keep = ("thickness", "wavelength", "norm", "per_decade", "r_per_decade", "q_min_factor", "q_max_factor", "q_range",
                "r_min_factor", "r_max_factor", "tail_power")  # fmt: skip
        return MSASTransform(**{k: v for k, v in settings.items() if k in keep})
    raise ValueError(f"unknown transform '{kind}' (one of {', '.join(TRANSFORM_KINDS)})")


# --------------------------------------------------------------------------- reference
def sphere_sesans(delta, R):
    """Normalised projected correlation function G(δ)/G(0) of a homogeneous sphere (Andersson et al. 2008)."""
    xi = np.asarray(delta, float) / (2 * R)
    out = np.zeros_like(xi)
    m = xi < 1
    x = xi[m]
    s = np.sqrt(1 - x**2)
    with np.errstate(divide="ignore", invalid="ignore"):
        lg = np.where(x > 0, np.log(x / (1 + s)), 0.0)
    out[m] = s * (1 + x**2 / 2) + 2 * x**2 * (1 - x**2 / 4) * lg
    return out
