"""Monte Carlo check of the FloryPolymer and ExcludedVolumePolymer sub-units.

1. Gaussian chains with <r_ij^2> ~ |i-j|^(2 nu) (fractional Brownian motion): the model's own statistics -- form
   factor, amplitude about an end and the end-to-end phase factor must agree with the closed forms (up to the
   discreteness of the walk).
2. Self-avoiding walks on the cubic lattice (pivot algorithm): how good are the generalised Gaussian chain
   (FloryPolymer, Gaussian distances) and the des Cloizeaux chain (ExcludedVolumePolymer) with nu = 0.588 for a
   real swollen chain?

    python tools/seb/check_flory_mc.py [N] [samples]
"""
import sys
import time

import numpy as np

sys.path.insert(0, "src")
from pysasfit.seb.subunits import FloryPolymer  # noqa: E402
from pysasfit.seb.chains import ExcludedVolumePolymer  # noqa: E402


def averages(chains, q):
    """F(q), A_end(q), Psi_end-end(q) by orientational averages sin(qr)/(qr) over the samples."""
    F = np.zeros_like(q)
    A = np.zeros_like(q)
    P = np.zeros_like(q)
    rg2 = 0.0
    for X in chains:
        n = len(X)
        d = np.sqrt(((X[:, None, :] - X[None, :, :]) ** 2).sum(-1))
        dd = d[np.triu_indices(n, 1)]
        de = np.concatenate([d[0], d[-1]])
        F += (n + 2 * np.sinc(np.outer(q, dd) / np.pi).sum(1)) / n**2
        A += np.sinc(np.outer(q, de) / np.pi).mean(1)
        P += np.sinc(q * d[0, -1] / np.pi)
        rg2 += np.mean(np.sum((X - X.mean(0)) ** 2, 1))
    k = len(chains)
    return F / k, A / k, P / k, rg2 / k


def fbm_chains(n, nu, m, rng):
    i = np.arange(1, n + 1, dtype=float)
    C = 0.5 * (i[:, None] ** (2 * nu) + i[None, :] ** (2 * nu) - np.abs(i[:, None] - i[None, :]) ** (2 * nu))
    L = np.linalg.cholesky(C)
    return [np.vstack([np.zeros(3), L @ rng.normal(size=(n, 3)) / np.sqrt(3)]) for _ in range(m)]


_OPS = None


def _cubic_ops():
    import itertools
    ops = []
    for perm in itertools.permutations(range(3)):
        for signs in itertools.product((1, -1), repeat=3):
            M = np.zeros((3, 3), int)
            for r, (c, s) in enumerate(zip(perm, signs)):
                M[r, c] = s
            if not np.array_equal(M, np.eye(3, dtype=int)):
                ops.append(M)
    return ops


def saw_chains(n, m, rng, every=40, burn=20000):
    """Self-avoiding walks of n bonds on the cubic lattice by the pivot algorithm."""
    ops = _cubic_ops()
    X = np.zeros((n + 1, 3), int)
    X[:, 0] = np.arange(n + 1)
    out, it = [], 0
    while len(out) < m:
        k = rng.integers(1, n)
        M = ops[rng.integers(len(ops))]
        if rng.uniform() < 0.5:
            new = X[k] + (X[k + 1:] - X[k]) @ M.T
            head = X[:k + 1]
            occ = set(map(tuple, head))
            if not any(tuple(v) in occ for v in new):
                X = np.vstack([head, new])
        else:
            new = X[k] + (X[:k] - X[k]) @ M.T
            tail = X[k:]
            occ = set(map(tuple, tail))
            if not any(tuple(v) in occ for v in new):
                X = np.vstack([new, tail])
        it += 1
        if it > burn and it % every == 0:
            out.append(X.astype(float))
    return out


def compare(name, chains, nu, q, m=None):
    F, A, P, rg2 = averages(chains, q)
    m = m or FloryPolymer()
    p = {"Rg": float(np.sqrt(rg2)), "nu": nu}
    Fm, Am, Pm = m.ff(q, p), m.amp(q, p, "end1"), m.psi(q, p, "end1", "end2")
    x = q * p["Rg"]
    print(f"\n{name}: {len(chains)} chains, Rg = {p['Rg']:.3f}, nu = {nu}")
    print("   qRg      F(MC)   F(model)    A_end(MC) A_end(model)  Psi_ee(MC) Psi_ee(model)")
    for i in range(0, len(q), max(1, len(q) // 10)):
        print(f"{x[i]:7.3f}  {F[i]:9.5f} {Fm[i]:9.5f}    {A[i]:9.5f} {Am[i]:9.5f}     {P[i]:9.5f} {Pm[i]:9.5f}")
    sel = x < 6
    return (np.max(np.abs(F - Fm)[sel]), np.max(np.abs(A - Am)[sel]), np.max(np.abs(P - Pm)[sel]))


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 128
    m = int(sys.argv[2]) if len(sys.argv) > 2 else 800
    rng = np.random.default_rng(2)
    t = time.time()
    for nu in (0.588, 0.4):
        ch = fbm_chains(n, nu, m, rng)
        rg = np.sqrt(averages(ch[:50], np.array([0.1]))[3])
        q = np.linspace(0.05, 6, 30) / rg
        e = compare(f"Gaussian chains with nu = {nu} (fBm, n = {n})", ch, nu, q)
        print("   max |MC - model| for qRg < 6: F %.4f  A_end %.4f  Psi_ee %.4f" % e)
    ch = saw_chains(n, m, rng)
    rg = np.sqrt(averages(ch[:50], np.array([0.1]))[3])
    q = np.linspace(0.05, 6, 30) / rg
    e = compare(f"self-avoiding walks (cubic lattice, pivot, n = {n}) vs FloryPolymer", ch, 0.588, q)
    print("   max |MC - model| for qRg < 6: F %.4f  A_end %.4f  Psi_ee %.4f" % e)
    e = compare("self-avoiding walks vs ExcludedVolumePolymer (des Cloizeaux)", ch, 0.588, q, ExcludedVolumePolymer())
    print("   max |MC - model| for qRg < 6: F %.4f  A_end %.4f  Psi_ee %.4f" % e)
    print(f"\n{time.time() - t:.0f} s")
