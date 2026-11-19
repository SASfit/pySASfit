"""Benchmark: NumPy vs Numba for the Schulz averages of pySCAtter (pysasfit.scatter.schulz_kernel).

    python run_bench.py [--src PATH/pySASfit/src] [--quick] [--threads 1,2,4,...]

Needs numpy, scipy, numba and the pySASfit sources with scatter/schulz_kernel.py (2026-11-18). Works offline.
Prints a table and writes bench_numba_<host>.txt next to this script.
"""
from __future__ import annotations

import argparse
import os
import platform
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=None, help="pySASfit/src folder (if pysasfit is not importable)")
    ap.add_argument("--quick", action="store_true", help="fewer cases")
    ap.add_argument("--threads", default=None, help="thread counts for the scaling test, e.g. 1,2,4,8")
    args = ap.parse_args()
    if args.src:
        sys.path.insert(0, os.path.abspath(args.src))
    sys.path.insert(0, HERE)
    import numpy as np
    import scipy
    import numba
    from pysasfit.scatter import blocks, schulz_kernel as sk
    import schulz_numba as snb

    lines = []

    def out(s=""):
        print(s, flush=True)
        lines.append(s)

    out("Numba benchmark for pysasfit.scatter.schulz_kernel")
    out(f"host {platform.node()}  {platform.platform()}  {platform.processor()}")
    out(f"python {platform.python_version()}  numpy {np.__version__}  scipy {scipy.__version__}  numba {numba.__version__}")
    out(f"cpu_count {os.cpu_count()}  numba threads {numba.get_num_threads()}  threading layer "
        f"{getattr(numba.config, 'THREADING_LAYER', '?')}  SVML {getattr(numba.config, 'USING_SVML', '?')}")
    out()

    def best(f, n=3):
        tb = 1e9
        r = None
        for _ in range(n):
            t = time.perf_counter()
            r = f()
            tb = min(tb, time.perf_counter() - t)
        return tb, r

    t = time.perf_counter()
    k_plain = snb.make(False)
    k_fast = snb.make(True)
    ker = snb.kernels(sk)
    yy = np.logspace(-2, 2, 50)
    for kk in ker.values():  # compile (all kinds)
        snb.value(k_plain, sk, kk, yy, 3.0, 6.0)
        snb.value(k_fast, sk, kk, yy, 3.0, 6.0)
    out(f"compile time {time.perf_counter() - t:.1f} s (both variants)")
    out()

    y1 = np.logspace(-2, 3, 500)
    y2 = np.outer(np.logspace(-2, 2.5, 200), np.linspace(0.2, 1.5, 100)).ravel()
    cases = [(3, 6), (24, 6), (99, 6), (0.5, 3)] if args.quick else [(3, 6), (24, 6), (99, 6), (399, 6), (0.5, 3), (7, 0)]
    blk = [(("P", 3), "P3"), (("F", 3), "F3"), (("P", 2), "P2"), (("F", 2), "F2"), (("P", 1), "P1"), (("L",), "line")]
    out("times in ms; numpy = schulz_kernel direct (no table cache); nb = numba; nb-fm = numba fastmath;")
    out("cache = schulz_kernel with the ln y tables (first call, incl. build); hybrid = old series/asymptotics code")
    out("maxdiff = max |numba - numpy| / max(|numpy|, 1e-300) (rows with |numpy| > 1e-12)")
    out(f"{'z':>6} {'s':>3} {'block':>5} | {'1D numpy':>8} {'nb':>7} {'nb-fm':>7} {'cache':>7} {'hybrid':>7} | "
        f"{'2D numpy':>8} {'nb':>7} {'nb-fm':>7} {'cache':>7} {'hybrid':>7} | maxdiff nb, nb-fm")
    tot = {"numpy": 0.0, "nb": 0.0, "fm": 0.0}
    for z, s in cases:
        for key, name in blk:
            kern = ker[key]
            row = []
            diffs = []
            for y in (y1, y2):
                sk.CACHE = False
                tn, rn = best(lambda: sk.schulz_cos(y, z, s, _sk_kernel(sk, key)), 2 if y is y2 else 3)
                tb, rb = best(lambda: snb.value(k_plain, sk, kern, y, z, s), 2 if y is y2 else 3)
                tf, rf = best(lambda: snb.value(k_fast, sk, kern, y, z, s), 2 if y is y2 else 3)
                sk.CACHE = True

                def cached():
                    sk.clear_cache()
                    return _cached(sk, key, y, z, s)

                tc, _ = best(cached, 1)
                blocks.METHOD = "hybrid"
                th, _ = best(lambda: _hybrid(blocks, key, y, z, s), 1)
                blocks.METHOD = "integral"
                big = np.abs(rn) > 1e-12
                sc = np.maximum(np.abs(rn[big]), 1e-300)
                diffs += [np.max(np.abs(rb - rn)[big] / sc), np.max(np.abs(rf - rn)[big] / sc)]
                row += [tn, tb, tf, tc, th]
                tot["numpy"] += tn
                tot["nb"] += tb
                tot["fm"] += tf
            r = [x * 1e3 for x in row]
            out(f"{z:>6} {s:>3} {name:>5} | {r[0]:8.1f} {r[1]:7.1f} {r[2]:7.1f} {r[3]:7.1f} {r[4]:7.1f} | "
                f"{r[5]:8.1f} {r[6]:7.1f} {r[7]:7.1f} {r[8]:7.1f} {r[9]:7.1f} | "
                f"{max(diffs[0], diffs[2]):.1e}, {max(diffs[1], diffs[3]):.1e}")
    out()
    out(f"sum of times: numpy {tot['numpy']:.2f} s, numba {tot['nb']:.2f} s ({tot['numpy'] / tot['nb']:.2f}x), "
        f"numba fastmath {tot['fm']:.2f} s ({tot['numpy'] / tot['fm']:.2f}x)")
    out()

    # thread scaling (2D, disc and sphere form factor)
    nmax = numba.config.NUMBA_NUM_THREADS
    ths = [int(x) for x in args.threads.split(",")] if args.threads else sorted({1, 2, 4, 8, 16, nmax} & set(range(1, nmax + 1)))
    out("thread scaling, 2D (20000 points), z=24 s=6, times in ms")
    out(f"{'threads':>8} {'P3 nb':>8} {'P3 nb-fm':>9} {'P2 nb':>8} {'P2 nb-fm':>9}")
    for nt in ths:
        numba.set_num_threads(nt)
        a1, _ = best(lambda: snb.value(k_plain, sk, ker[("P", 3)], y2, 24, 6), 2)
        a2, _ = best(lambda: snb.value(k_fast, sk, ker[("P", 3)], y2, 24, 6), 2)
        a3, _ = best(lambda: snb.value(k_plain, sk, ker[("P", 2)], y2, 24, 6), 2)
        a4, _ = best(lambda: snb.value(k_fast, sk, ker[("P", 2)], y2, 24, 6), 2)
        out(f"{nt:>8} {a1 * 1e3:8.1f} {a2 * 1e3:9.1f} {a3 * 1e3:8.1f} {a4 * 1e3:9.1f}")
    numba.set_num_threads(nmax)
    sk.CACHE = False
    a5, _ = best(lambda: sk.schulz_cos(y2, 24, 6, sk.kernel_p(3)), 2)
    a6, _ = best(lambda: sk.schulz_cos(y2, 24, 6, sk.kernel_p(2)), 2)
    sk.CACHE = True
    out(f"{'numpy':>8} {a5 * 1e3:8.1f} {'':>9} {a6 * 1e3:8.1f}")
    fn = os.path.join(HERE, f"bench_numba_{platform.node() or 'host'}.txt")
    with open(fn, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"\nwritten: {fn}")


def _sk_kernel(sk, key):
    if key[0] == "L":
        return sk.kernel_line()
    return sk.kernel_amp(key[1]) if key[0] == "F" else sk.kernel_p(key[1])


def _cached(sk, key, y, z, s):
    if key[0] == "L":
        return sk.line_integral_value(y, z, s)
    return sk.block_value(key[1], y, z, s, key[0])


def _hybrid(blocks, key, y, z, s):
    if key[0] == "L":
        return blocks.BLOCK[1].line_integral(y, z, s)
    return blocks.BLOCK[key[1]].value(y, z, s, key[0])


if __name__ == "__main__":
    main()
