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
    "hankel_weights", "log_grid", "fftlog_grid", "FFTLogPlan", "fast_lengths", "is_fast_length", "fftlog_matrix",
    "fftlog_weights", "qdht_weights", "METHODS", "SESANSTransform", "MSASTransform", "make_transform", "TRANSFORM_KINDS",
    "transform_label", "sphere_sesans", "DEFAULTS", "defaults", "set_defaults", "load_settings", "save_settings",
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


# --------------------------------------------------------------------------- FFTLog and QDHT (as in SASfit)
METHODS = ("filon", "fftlog", "qdht", "sasfit")
METHOD_LABELS = {
    "filon": "exact on linear pieces (default)",
    "fftlog": "FFTLog (Hamilton 2000)",
    "qdht": "QDHT (Guizar-Sicairos 2004)",
    "sasfit": "SASfit sasfit_hankel (strategy of the SASfit library tab)",
}


# Numerical settings of the transforms: defaults for every transform that does not set them itself. The explorer
# edits them in Settings → Integration… (tab "Hankel transform") and remembers them (paths setting "hankel");
# set_defaults() changes them for this session, load_settings()/save_settings() read and write the saved ones.
DEFAULTS = {
    "method": "filon",  # filon | fftlog | qdht
    "sesans_per_decade": 64, "sesans_q_min_factor": 1e-2, "sesans_q_max_factor": 60.0,
    "msas_per_decade": 48, "msas_r_per_decade": 384, "msas_q_min_factor": 1e-2, "msas_q_max_factor": 30.0,
    "msas_r_min_factor": 0.3, "msas_r_max_factor": 30.0,
    "tail_power": 4.0,  # Porod tail above the Q grid (0: none)
    "n_points": 2048,  # FFTLog / QDHT
    "fft_low": 1e-5, "sesans_fft_high": 1e3, "msas_fft_high": 1000.0,  # wide: FFTLog rings at the ends of its grid
    "qdht_margin": 2.0, "qdht_r_factor": 10.0,
    "sasfit_source": "spline",  # sasfit_hankel: I(Q) from a cubic spline on the Q grid, or "direct" (model per point)
    "sasfit_r_per_decade": 32,  # sasfit_hankel MSAS: r points per decade (one adaptive transform each)
}  # fmt: skip
_current = dict(DEFAULTS)
_loaded = False


def defaults():
    """The settings in use (saved ones on first use, then those of :func:`set_defaults`)."""
    global _loaded
    if not _loaded:
        _loaded = True
        try:
            _current.update(load_settings())
        except Exception:  # noqa: BLE001 -- no settings file: the built-in defaults
            pass
    return dict(_current)


def set_defaults(**settings):
    """Change the defaults for transforms made from now on (keys of :data:`DEFAULTS`); ``reset=True`` first
    restores the built-in values. Returns the settings in use."""
    defaults()
    if settings.pop("reset", False):
        _current.clear()
        _current.update(DEFAULTS)
    for k, v in settings.items():
        if k not in DEFAULTS:
            raise KeyError(f"unknown Hankel setting '{k}' (one of {', '.join(DEFAULTS)})")
        _current[k] = _check_value(k, v)
    return dict(_current)


def load_settings():
    """The saved settings (``paths.get_setting("hankel")``), built-in defaults for missing or bad entries."""
    from . import paths

    out = dict(DEFAULTS)
    saved = paths.get_setting("hankel")
    if isinstance(saved, dict):
        for k, d in DEFAULTS.items():
            if k in saved:
                try:
                    out[k] = _check_value(k, saved[k])
                except (TypeError, ValueError):
                    pass
    return out


def save_settings(settings):
    """Remember the settings for later sessions and scripts."""
    from . import paths

    paths.set_setting("hankel", {k: settings.get(k, DEFAULTS[k]) for k in DEFAULTS})


def _check_value(key, v):
    if key == "method":
        return _check_method(v)
    if key == "sasfit_source":
        if v not in ("spline", "direct"):
            raise ValueError("sasfit_source must be 'spline' or 'direct'")
        return v
    return type(DEFAULTS[key])(v)


def _d(value, key):
    """``value`` or, if None, the current default ``key``."""
    if value is not None:
        return value
    v = defaults()[key]
    return None if key == "tail_power" and not v else v


def _check_method(method):
    method = (method or "filon").lower().replace("hankel_", "")
    if method not in METHODS:
        raise ValueError(f"method must be one of {', '.join(METHODS)}")
    return method


def fftlog_grid(klo, khi, n):
    """FFTLog grids: ``(k, r, dln, offset)`` -- n log-spaced k from klo to khi and the reciprocal r grid
    (k_c r_c = exp(offset), the low-ringing offset of :func:`scipy.fft.fhtoffset`)."""
    from scipy import fft

    n = int(n)
    dln = math.log(khi / klo) / (n - 1)
    jc = (n - 1) / 2
    kc = math.sqrt(klo * khi)
    j = np.arange(n) - jc
    offset = float(fft.fhtoffset(dln, mu=0.0))
    return kc * np.exp(j * dln), math.exp(offset) / kc * np.exp(j * dln), dln, offset


def fast_lengths(n):
    """``(previous, next)`` FFT lengths around n that factor into 2, 3 and 5 only (n itself if it is one).
    FFTLog does a real FFT of length n, so these are much faster than e.g. 2047 = 23·89 (OZ solvers prefer 2^m - 1
    because their sine transform needs an FFT of length 2(n+1); FFTLog does not)."""
    from scipy import fft

    n = int(n)
    nxt = fft.next_fast_len(n, real=True)
    prev = n
    while prev > 1 and fft.next_fast_len(prev, real=True) != prev:
        prev -= 1
    return prev, nxt


def is_fast_length(n):
    return fast_lengths(n)[1] == int(n)


_PLANS = OrderedDict()


class FFTLogPlan:
    """The FFTLog transform of one grid, precomputed once: ``(T @ a)[j] = ∫ a(k) J0(k x_j) x_j dk``.

    FFTLog is a circular convolution in ln k: T[j, i] = c[(i + j) mod n], so the n numbers c (the transform of a unit
    vector) define it completely. The plan keeps their spectrum; :meth:`apply` is then two real FFTs of length n
    (no matrix), and :meth:`matrix` builds T from c by indexing (no FFTs). Plans are shared between transforms with
    the same grid. A length n that is not a fast FFT length is used as given; a warning names the nearest fast ones."""

    def __init__(self, n, dln, offset):
        from scipy import fft

        self.n, self.dln, self.offset = int(n), float(dln), float(offset)
        if not is_fast_length(self.n):
            import warnings

            prev, nxt = fast_lengths(self.n)
            warnings.warn(f"FFTLog: N = {self.n} is a slow FFT length; nearest fast lengths {prev} and {nxt}",
                          stacklevel=3)
        e0 = np.zeros(self.n)
        e0[0] = 1.0
        self.c = fft.fht(e0, self.dln, mu=0.0, offset=self.offset)  # T[j, 0] = c[j]
        self.C = fft.rfft(self.c)

    @classmethod
    def get(cls, n, dln, offset):
        key = (int(n), round(float(dln), 15), round(float(offset), 15))
        if key in _PLANS:
            _PLANS.move_to_end(key)
            return _PLANS[key]
        plan = _PLANS[key] = cls(n, dln, offset)
        while len(_PLANS) > 8:
            _PLANS.popitem(last=False)
        return plan

    def apply(self, a):
        """T @ a (a: one vector, or several in the last axis) by FFT."""
        from scipy import fft

        a = np.asarray(a, float)
        ar = np.roll(a[..., ::-1], 1, axis=-1)  # a[-i mod n]
        return fft.irfft(self.C * fft.rfft(ar, axis=-1), self.n, axis=-1)

    def matrix(self):
        i = np.arange(self.n)
        return self.c[(i[None, :] + i[:, None]) % self.n]


def fftlog_matrix(n, dln, offset):
    """T with ``(T @ a)[i] = ∫ a(k) J0(k x_i) x_i dk`` for a on an FFTLog grid, built from the precomputed plan."""
    return FFTLogPlan.get(n, dln, offset).matrix()


def fftlog_weights(x, k, r, dln, offset, kernel="j0", T=None):
    """``W @ f(k) ≈ ∫ f(k) K(k x) k dk`` by FFTLog onto the grid r, then linear interpolation in ln r to x.
    ``kernel="j0m1"`` subtracts ∫ f k dk (trapezoidal rule in ln k on the same grid). Only the rows of the FFTLog
    matrix next to each x are formed (from the plan's kernel, T[j, i] = c[(i + j) mod n])."""
    x = np.asarray(x, float)
    n = k.size
    c = FFTLogPlan.get(n, dln, offset).c
    M = _interp_matrix(r, x, "FFTLog")  # two neighbours per row
    W = np.zeros((x.size, n))
    i = np.arange(n)
    for row in range(x.size):
        for j in np.nonzero(M[row])[0]:
            W[row] += M[row, j] * c[(i + j) % n] * k / r[j]  # F(r_j) = (T @ (f k))_j / r_j
    if np.any(x == 0) or kernel == "j0m1":
        w0 = _trapz_ln(k) * k**2  # ∫ f k dk = ∫ f k² d ln k
        W[x == 0] = w0
        if kernel == "j0m1":
            W = W - w0[None, :]
    return W


def _trapz_ln(k):
    lk = np.log(k)
    w = np.zeros(k.size)
    w[:-1] += 0.5 * np.diff(lk)
    w[1:] += 0.5 * np.diff(lk)
    return w


def _interp_matrix(nodes, x, what):
    """Linear interpolation in ln(node) from increasing ``nodes`` to ``x`` (x = 0 rows are left zero)."""
    x = np.asarray(x, float)
    M = np.zeros((x.size, nodes.size))
    ln = np.log(nodes)
    for i, xi in enumerate(x):
        if xi == 0:
            continue
        if xi < nodes[0] * (1 - 1e-9) or xi > nodes[-1] * (1 + 1e-9):
            raise ValueError(f"{what}: {xi:.4g} is outside the transform grid {nodes[0]:.4g} … {nodes[-1]:.4g}")
        j = int(np.clip(np.searchsorted(ln, math.log(xi)) - 1, 0, nodes.size - 2))
        t = (math.log(xi) - ln[j]) / (ln[j + 1] - ln[j])
        M[i, j], M[i, j + 1] = 1 - t, t
    return M


def _cell_fraction(k, upper, log=False):
    """Part of the cell of every node k (edges halfway between the nodes, in ln k for ``log``) below ``upper``
    (one row per limit): cutting a node sum at a limit between the nodes."""
    u = np.log(k) if log else k
    e = np.empty(k.size + 1)
    e[1:-1] = 0.5 * (u[1:] + u[:-1])
    e[0], e[-1] = u[0] - 0.5 * (u[1] - u[0]), u[-1] + 0.5 * (u[-1] - u[-2])
    lim = np.log(upper) if log else np.asarray(upper, float)
    return np.clip((lim[:, None] - e[None, :-1]) / (e[None, 1:] - e[None, :-1]), 0.0, 1.0)


def _interpolant(x, y, tail_power=None, zero_above=False):
    """f(t) for the adaptive SASfit strategies: cubic spline of y in ln x, constant below the first node, above the
    last node a power law t^-p (``tail_power``), zero (``zero_above``) or constant."""
    from scipy.interpolate import CubicSpline

    lx = np.log(x)
    cs = CubicSpline(lx, np.asarray(y, float))
    x0, xn, y0, yn = float(x[0]), float(x[-1]), float(y[0]), float(y[-1])
    c = cs.c
    brk = lx

    def f(t):
        if t <= x0:
            return y0
        if t >= xn:
            if zero_above:
                return 0.0
            return yn * (xn / t) ** tail_power if tail_power else yn
        u = math.log(t)
        i = min(max(int(np.searchsorted(brk, u)) - 1, 0), brk.size - 2)
        d = u - brk[i]
        return ((c[0, i] * d + c[1, i]) * d + c[2, i]) * d + c[3, i]

    return f


def _sasfit_library():
    from .library import get_library

    lib = get_library()
    if getattr(lib.core, "sasfit_hankel", None) is None:
        raise RuntimeError("this libsasfit has no sasfit_hankel")
    return lib


_ZEROS = {}


def bessel_zeros(n):
    """The first n + 1 zeros of J0 (cached)."""
    if n not in _ZEROS:
        _ZEROS[n] = special.jn_zeros(0, n + 1)
    return _ZEROS[n]


def qdht_weights(x, X, n, kernel="j0"):
    """Quasi-discrete Hankel transform (Fourier-Bessel series): F(x) = ∫_0^K f(k) J0(kx) k dk for F supported on
    [0, X]. Nodes k_m = j_m / X (m = 1..n, K = j_(n+1) / X); returns ``(k, W)`` with ``W @ f(k) = F(x)``.
    The series is evaluated at x itself, no interpolation: W[i, m] = 2 J0(k_m x_i) / (X² J1(j_m)²)."""
    j = bessel_zeros(int(n))
    jm = j[:-1]
    k = jm / X
    x = np.atleast_1d(np.asarray(x, float))
    if np.any(x > X * (1 + 1e-9)):
        raise ValueError(f"QDHT: {x.max():.4g} lies beyond the support X = {X:.4g} (increase the margin)")
    c = 2.0 / (X**2 * special.j1(jm) ** 2)
    K = special.j0(np.outer(x, k))
    if kernel == "j0m1":
        K = K - 1.0
    return k, K * c[None, :]


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
                 per_decade=None, q_min_factor=None, q_max_factor=None, q_range=None, tail_power=None, theta_max=None,
                 accept_delta=None, method=None, n_points=None, fft_range=None, qdht_margin=None,
                 sasfit_source=None):  # fmt: skip
        # None: the defaults of Settings → Integration… → Hankel transform (see DEFAULTS, set_defaults)
        self.sasfit_source = _check_value("sasfit_source", _d(sasfit_source, "sasfit_source"))
        per_decade, q_min_factor = _d(per_decade, "sesans_per_decade"), _d(q_min_factor, "sesans_q_min_factor")
        q_max_factor, method, n_points = _d(q_max_factor, "sesans_q_max_factor"), _d(method, "method"), _d(n_points, "n_points")
        tail_power = _d(tail_power, "tail_power") if tail_power is not False else None
        fft_range = fft_range or (_d(None, "fft_low"), _d(None, "sesans_fft_high"))
        qdht_margin = _d(qdht_margin, "qdht_margin")
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
        self.method = _check_method(method)
        self.n_points = int(n_points)
        self.fft_range = tuple(map(float, fft_range))
        self.qdht_margin = float(qdht_margin)
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
        """G(δ) - G(0) is linear in I(Q) (variable projection works); P/P0 is not. With sasfit_hankel there is no
        weight matrix, so the model is treated as non-linear."""
        return self.observable == "G" and self.method != "sasfit"

    def g_from(self, delta, intensity):
        """G(δ) - G(0) for the function ``intensity(k)`` (vectorised): weights @ I on the nodes, or libsasfit's
        adaptive ``sasfit_hankel`` (method "sasfit")."""
        delta = np.asarray(delta, float)
        if self.method != "sasfit":
            k, W = self.weights(delta)
            return W @ np.asarray(intensity(k), float)
        lib = _sasfit_library()
        if self.sasfit_source == "direct":  # the model at every point the strategy asks for (as SASfit does)
            def f(t):
                return float(np.asarray(intensity(np.array([t])), float)[0])
        else:
            k, tail = self.grid(delta)
            f = _interpolant(k, intensity(k), tail, zero_above=tail is None)
        qmax = self.q_max(delta)
        if qmax is None:
            H = lib.hankel(f, np.abs(delta))
            H0 = lib.integrate(lambda t: t * f(t), 0.0, math.inf)
            return (H - H0) / (2 * np.pi)
        out = np.empty(delta.size)  # resolution: ∫_0^Q_max I (J0 - 1) Q dQ (SASfit's branch with Qres > 0)
        for i, (d, qm) in enumerate(zip(np.abs(delta), qmax)):
            out[i] = lib.integrate(lambda t, d=d: f(t) * t * (special.j0(t * d) - 1.0), 0.0, float(qm))
        return out / (2 * np.pi)

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
        """``(k, W)``: the Q nodes where I is needed and G(δ) - G(0) = W @ I(k) (cached)."""
        delta = np.asarray(delta, float)
        if self.method == "sasfit":
            raise ValueError("sasfit_hankel has no weight matrix (use g_from)")
        qmax = self.q_max(delta)
        if self.method == "filon":
            k, tail = self.grid(delta, qmax)
            key = (delta.tobytes(), k[0], k[-1], k.size, tail, None if qmax is None else qmax.tobytes())
            W = self._cache.get(key, lambda: hankel_weights(np.abs(delta), k, "j0m1", True, tail, qmax) / (2 * np.pi))
            return k, W
        key = (self.method, delta.tobytes(), self.n_points, self.fft_range, self.qdht_margin,
               None if qmax is None else qmax.tobytes())  # fmt: skip
        return self._cache.get(key, lambda: self._grid_weights(delta, np.abs(delta), "j0m1", qmax))

    def _grid_weights(self, delta, x, kernel, qmax):
        """FFTLog / QDHT weights (grid from the spin-echo lengths ``delta``, evaluated at ``x``)."""
        d = delta[np.isfinite(delta) & (delta > 0)]
        if d.size == 0:
            raise ValueError("SESANS: no spin-echo length > 0")
        if self.method == "fftlog":
            lo, hi = self.fft_range
            k, r, dln, off = fftlog_grid(lo / d.max(), hi / d.min(), self.n_points)
            W = fftlog_weights(x, k, r, dln, off, kernel) / (2 * np.pi)
        else:
            X = self.qdht_margin * d.max()
            k, W = qdht_weights(x, X, self.n_points, kernel)
            W = W / (2 * np.pi)
            if k[-1] * d.min() < 10:
                import warnings

                warnings.warn(f"SESANS QDHT: the Q grid ends at {k[-1]:.3g} = {k[-1] * d.min():.3g}/δ_min; more points "
                              "are needed for the smallest spin-echo lengths", stacklevel=2)
        if qmax is not None:  # resolution: I(Q) = 0 above Q_max of each row
            qm = np.asarray(qmax, float)
            qm = np.full(W.shape[0], float(qm)) if qm.ndim == 0 else qm
            W = W * _cell_fraction(k, qm, log=self.method == "fftlog")
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
        bg = getattr(model, "background", 0.0)
        G = self.g_from(delta, lambda k: _particles(model, k)) + (float(bg) if np.isscalar(bg) and bg is not None else 0.0)
        return self.finish(delta, G)

    def parts(self, model, delta, intensities):
        """Linear transform of contribution intensities given on the grid (``observable="G"`` only)."""
        _k, W = self.weights(delta)
        return [W @ np.asarray(v, float) for v in intensities]

    def total_scattering(self, model, delta):
        """Σ t = λ² t G(0) (the scattering probability; λ at the largest δ)."""
        qm = self.q_max(np.atleast_1d(np.max(delta)))
        if self.method == "filon":
            k, tail = self.grid(delta)
            W0 = hankel_weights([0.0], k, "j0", True, tail, None if qm is None else qm[:1]) / (2 * np.pi)
        else:
            k, W0 = self._grid_weights(np.asarray(delta, float), np.array([0.0]), "j0", None if qm is None else qm[:1])
        G0 = float(W0 @ _particles(model, k))
        lam = float(np.max(self.lam(np.atleast_1d(np.max(delta)))))
        return lam**2 * self.thickness * G0

    def settings(self):
        d = {"kind": self.kind, "thickness": self.thickness, "per_decade": self.per_decade,
             "q_min_factor": self.q_min_factor, "q_max_factor": self.q_max_factor, "tail_power": self.tail_power if self.tail_power is not None else False}  # fmt: skip
        d["wavelength"] = np.asarray(self.wavelength, float).tolist() if np.ndim(self.wavelength) else self.wavelength
        if self.wavelength_delta is not None:
            d["wavelength_delta"] = self.wavelength_delta.tolist()
        if self.q_accept is not None:
            d["q_accept"] = np.asarray(self.q_accept).tolist() if np.ndim(self.q_accept) else self.q_accept
        if self.accept_delta is not None:
            d["accept_delta"] = self.accept_delta.tolist()
        if self.theta_max:
            d["theta_max"] = self.theta_max
        d.update(method=self.method, n_points=self.n_points, fft_range=list(self.fft_range),
                 qdht_margin=self.qdht_margin, sasfit_source=self.sasfit_source)  # fmt: skip
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

    def __init__(self, thickness=0.1, wavelength=0.6, norm="incident", per_decade=None, r_per_decade=None,
                 q_min_factor=None, q_max_factor=None, q_range=None, r_min_factor=None, r_max_factor=None,
                 tail_power=None, method=None, n_points=None, fft_range=None, qdht_r_factor=None,
                 sasfit_r_per_decade=None):  # fmt: skip
        # None: the defaults of Settings → Integration… → Hankel transform (see DEFAULTS, set_defaults)
        self.sasfit_r_per_decade = int(_d(sasfit_r_per_decade, "sasfit_r_per_decade"))
        per_decade, r_per_decade = _d(per_decade, "msas_per_decade"), _d(r_per_decade, "msas_r_per_decade")
        q_min_factor, q_max_factor = _d(q_min_factor, "msas_q_min_factor"), _d(q_max_factor, "msas_q_max_factor")
        r_min_factor, r_max_factor = _d(r_min_factor, "msas_r_min_factor"), _d(r_max_factor, "msas_r_max_factor")
        tail_power = _d(tail_power, "tail_power") if tail_power is not False else None
        method, n_points = _d(method, "method"), _d(n_points, "n_points")
        fft_range = fft_range or (_d(None, "fft_low"), _d(None, "msas_fft_high"))
        qdht_r_factor = _d(qdht_r_factor, "qdht_r_factor")
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
        self.method = _check_method(method)
        self.n_points = int(n_points)
        self.fft_range = tuple(map(float, fft_range))
        self.qdht_r_factor = float(qdht_r_factor)
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
        """``(k, r, (Wkr, Wrq))``: I is needed at k; [G(0), G(r)] = Wkr @ I(k); ∫ψ J0(Qr) r dr = Wrq @ ψ(r)."""
        q = np.asarray(q, float)
        if self.method == "sasfit":  # I on the exact method's Q grid (spline), a coarser r grid
            k, r = self.grids(q)
            r = log_grid(r[0], r[-1], self.sasfit_r_per_decade)
            return k, r, ("sasfit",)
        if self.method != "filon":
            key = (self.method, q.tobytes(), self.n_points, self.fft_range, self.qdht_r_factor)
            return self._cache.get(key, lambda: self._grid_weights(q))
        k, r = self.grids(q)
        tail = self.tail_power

        def make():
            Wkr = hankel_weights(np.concatenate([[0.0], r]), k, "j0", True, tail) / (2 * np.pi)  # I(k) -> G(0), G(r)
            Wrq = hankel_weights(np.abs(q), r, "j0", True, None)  # ψ(r) -> ∫ ψ J0(Q r) r dr
            return Wkr, Wrq

        key = (q.tobytes(), k[0], k[-1], k.size, r[0], r[-1], r.size, tail)
        return k, r, self._cache.get(key, make)

    def _grid_weights(self, q):
        qq = q[np.isfinite(q) & (q > 0)]
        if qq.size == 0:
            raise ValueError("MSAS: no q > 0")
        n = self.n_points
        if self.method == "fftlog":  # no matrices: the plan (kernel spectrum) and a sparse interpolation
            lo, hi = self.fft_range
            k, r, dln, off = fftlog_grid(lo * qq.min(), hi * qq.max(), n)
            plan = FFTLogPlan.get(n, dln, off)
            w0 = _trapz_ln(k) * k**2 / (2 * np.pi)  # G(0)
            interp = _interp_matrix(k, np.abs(q), "MSAS FFTLog")
            return k, r, ("fftlog", plan, w0, interp)
        X = self.qdht_r_factor / qq.min()  # support of G(r)
        K = bessel_zeros(n)[-1] / X  # band limit in Q
        if qq.max() > K:
            raise ValueError(f"MSAS QDHT: Q_max of the data {qq.max():.4g} is beyond the band limit {K:.4g}; "
                             "increase the number of points")
        if qq.max() > K / 3:
            import warnings

            warnings.warn(f"MSAS QDHT: the band limit {K:.3g} is less than 3 × Q_max of the data; more points are "
                          "needed for the largest q", stacklevel=2)
        rn = bessel_zeros(n)[:-1] / K
        k, Wf = qdht_weights(np.concatenate([[0.0], rn]), X, n, "j0")
        r, Wrq = qdht_weights(np.abs(q), K, n, "j0")
        return k, r, (Wf / (2 * np.pi), Wrq)

    def multiple(self, q, Ik, I_single):
        """I_MSAS at ``q`` from I on the k grid and the single scattering I(q) at ``q`` (no background)."""
        k, r, w = self.weights(q)
        lib = None
        if isinstance(w[0], str) and w[0] == "sasfit":  # libsasfit's adaptive strategy on a spline of I(Q)
            lib = _sasfit_library()
            fI = _interpolant(k, Ik, self.tail_power, zero_above=self.tail_power is None)
            G0 = lib.integrate(lambda t: t * fI(t), 0.0, math.inf)
            G = np.concatenate([[G0], lib.hankel(fI, r)]) / (2 * np.pi)
            Wrq = None
        elif isinstance(w[0], str):  # FFTLog: two FFTs of length n with the precomputed plan
            _tag, plan, w0, interp = w
            G = np.concatenate([[w0 @ Ik], plan.apply(Ik * k) / r / (2 * np.pi)])
            Wrq = None
        else:
            Wkr, Wrq = w
            G = Wkr @ Ik
        lt = self.wavelength**2 * self.thickness
        tau = lt * G[0]
        self.last_tau = float(tau)
        if not np.isfinite(tau) or tau > 100:
            raise ValueError(f"MSAS: Σt = λ²t·G(0) = {tau:.4g} is too large (check thickness, wavelength and the absolute "
                             "scale of I(Q))")
        phi = lt * G[1:]
        psi = np.expm1(phi) - phi
        if lib is not None:
            back = lib.hankel(_interpolant(r, psi, zero_above=True), np.abs(q))  # ∫ ψ J0(Qr) r dr
        elif Wrq is None:
            back = interp @ (plan.apply(psi * r) / k)  # ∫ ψ J0(Qr) r dr on the k grid, then to the q of the data
        else:
            back = Wrq @ psi
        Iq = I_single + 2 * np.pi / lt * back
        if self.norm == "incident":
            Iq = Iq * math.exp(-tau)
        return Iq

    def evaluate_particles(self, model, q):
        k, _r, _w = self.weights(q)
        return self.multiple(q, _particles(model, k), _particles(model, q))

    def settings(self):
        d = {"kind": "msas", "thickness": self.thickness, "wavelength": self.wavelength, "norm": self.norm,
             "per_decade": self.per_decade, "r_per_decade": self.r_per_decade, "q_min_factor": self.q_min_factor,
             "q_max_factor": self.q_max_factor, "r_min_factor": self.r_min_factor, "r_max_factor": self.r_max_factor, "tail_power": self.tail_power if self.tail_power is not None else False}  # fmt: skip
        d.update(method=self.method, n_points=self.n_points, fft_range=list(self.fft_range),
                 qdht_r_factor=self.qdht_r_factor, sasfit_r_per_decade=self.sasfit_r_per_decade)  # fmt: skip
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
                "q_range", "tail_power", "theta_max", "accept_delta", "method", "n_points", "fft_range", "qdht_margin",
                "sasfit_source")
        return SESANSTransform("G" if kind == "sesans" else "P", **{k: v for k, v in settings.items() if k in keep})
    if kind == "msas":
        keep = ("thickness", "wavelength", "norm", "per_decade", "r_per_decade", "q_min_factor", "q_max_factor", "q_range",
                "method", "n_points", "fft_range", "qdht_r_factor", "sasfit_r_per_decade",
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
