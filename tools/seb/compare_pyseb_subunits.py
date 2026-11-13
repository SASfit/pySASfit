"""Compare single sub-unit F, A, Psi with pyseb 0.4.0 (run with Python 3.12 + pyseb)."""
import sys, itertools, numpy as np
sys.path.insert(0, "src")
import pyseb
from pysasfit.seb.subunits import REGISTRY, Spheroid

q = np.logspace(-3, np.log10(0.8), 40)
SIZES = {"Rg": 9.0, "R": 12.0, "L": 40.0, "Ri": 8.0, "Ro": 15.0, "a": 10.0, "c": 22.0}
worst = []
def rel(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return np.max(np.abs(a - b) / np.maximum(np.abs(b), 1e-3))
for kind, cls in REGISTRY.items():
    if kind == "Point":
        continue
    m = Spheroid(surface_measure="parametric", n_mu=256, n_theta=128) if kind == "Spheroid" else cls()
    if hasattr(m, "_mu") and kind != "Spheroid":
        m = cls(n_mu=256)
    w = pyseb.World(); w.Add(kind, "u", "u")
    pp = {f"{s}_u": SIZES[s] for s in cls.size_params}; pp["beta_u"] = 1.0
    p = {s: SIZES[s] for s in cls.size_params}
    res = []
    try:
        ref = w.EvaluateFormFactor("u", pp, list(q))
        res.append(("F", rel(m.ff(q, p), ref)))
    except Exception as e:
        res.append(("F", f"pyseb error {e}"))
    for r in cls.references():
        rr = f"u.{r}" + ("#a" if r in cls.distributed else "")
        try:
            ref = w.EvaluateFormFactorAmplitude(rr, pp, list(q))
            res.append((f"A[{r}]", rel(m.amp(q, p, r), ref)))
        except Exception as e:
            res.append((f"A[{r}]", f"pyseb error {e}"))
    refs = cls.references()
    for r1, r2 in itertools.combinations_with_replacement(refs, 2):
        if r1 == r2 and r1 not in cls.distributed:
            continue
        a = f"u.{r1}" + ("#a" if r1 in cls.distributed else "")
        b = f"u.{r2}" + ("#b" if r2 in cls.distributed else "")
        try:
            ref = w.EvaluatePhaseFactor(a, b, pp, list(q))
            res.append((f"Psi[{r1},{r2}]", rel(m.psi(q, p, r1, r2), ref)))
        except Exception as e:
            res.append((f"Psi[{r1},{r2}]", f"pyseb error {str(e)[:60]}"))
    print(f"== {kind}")
    for k, v in res:
        flag = "" if isinstance(v, str) or v < 1e-6 else "   <-- DIFF"
        print(f"   {k:28s} {v if isinstance(v,str) else f'{v:.2e}'}{flag}")
