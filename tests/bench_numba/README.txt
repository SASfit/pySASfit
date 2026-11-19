Numba benchmark for pySCAtter's Schulz averages (pysasfit.scatter.schulz_kernel, 2026-11-18)
==========================================================================================

Contents
  run_bench.py     the benchmark (prints a table, writes bench_numba_<host>.txt)
  schulz_numba.py  Numba version of schulz_kernel._piece: the same panels, nodes and weights as the NumPy code,
                   compiled twice (plain, and fastmath=True so that LLVM may vectorise tan/cos/log/exp with SVML)

Requirements (no network needed)
  numpy, scipy, numba, and your pySASfit sources containing src/pysasfit/scatter/schulz_kernel.py

Run (from the unzipped folder)
  python run_bench.py --src C:/Users/kohlbrecher/switchdrive/pySASfit/src
  python run_bench.py --src ... --quick            (4 instead of 6 size distributions)
  python run_bench.py --src ... --threads 1,2,4,8  (thread counts for the scaling test)

  MSYS2: use the same python as pySASfit (e.g. ../SASfitGit/src/.venv/bin/python3) so that numba is found.
  The first lines show numba version, thread count, threading layer and whether SVML is used.

What is measured (times in ms, best of 2-3 runs)
  numpy   schulz_kernel direct (table cache off)        nb, nb-fm  the Numba kernel (plain / fastmath)
  cache   schulz_kernel with the ln y tables, first call including the table build (what pySCAtter uses now)
  hybrid  the old series/asymptotics code (blocks.METHOD = "hybrid")
  1D = 500 q values, 2D = 200 x 100 = 20000 values; blocks P3/F3 (sphere), P2/F2 (disc), P1 (rod), line integral.
  maxdiff: largest relative difference Numba vs NumPy for values > 1e-12. The amplitude averages F decay to tiny
  values where both carry rounding noise of ~1e-16 absolute, so maxdiff up to ~1e-6 there is expected; for the
  form factors (P) it should be <~1e-10.
  At the end: thread scaling for the 2D case and the sum of all times (numpy / numba ratio).

Reference result of the cloud sandbox (2 cores, numba 0.68.1, no SVML): numba 1.2x faster than NumPy in total,
2x for the disc kernel, slower for the cheap polynomial kernels; the cached path is faster than both for 2D.
Please send back bench_numba_<host>.txt.
