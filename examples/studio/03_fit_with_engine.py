"""Fit the bimodal sphere data set of SASfit with the fitting engine: bounds, a constraint, a prior, several optimisers.

    python examples/studio/03_fit_with_engine.py
"""

import os

import matplotlib.pyplot as plt

from pysasfit.studio import ModelFunction, ScatteringContribution, fitting

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data", "SASfitTestData", "test.dat")


# --8<-- [start:problem]
def build():
    """A fresh model and fit problem (the start values are about 30 % off SASfit's result)."""
    big = ScatteringContribution("ff_sphere", {"R": 1, "eta": 1}, size_distribution="sd_LogNorm",
                                 sd_params={"N": 2.5e-12, "s": 0.25, "p": 1, "mu": 120}, distributed="R")
    small = ScatteringContribution("ff_sphere", {"R": 1, "eta": 1}, size_distribution="sd_LogNorm",
                                   sd_params={"N": 1e-10, "s": 0.25, "p": 1, "mu": 55}, distributed="R")
    model = ModelFunction([big, small], background=0.07)

    prob = fitting.FitProblem()
    prob.add_dataset(DATA, model, name="test")                 # a file, or a tuple (q, I, sigma)
    for k in (0, 1):                                           # contribution k: N, mu, s of its distribution
        prob.add_common(f"{k}.sd.N", lower=0)
        prob.add_common(f"{k}.sd.mu", lower=1, upper=1e4)
        prob.add_common(f"{k}.sd.s", lower=0.01, upper=2)
    prob.add_local("background", lower=0)

    names = {label: ident for ident, label in prob.identifiers().items()}   # label -> name usable in a constraint
    prob.add_constraint(f"{names['sd.mu']} > {names['1.sd.mu']}")           # the first population is the larger one
    prob.add_prior("sd.s", 0.3, 0.1)                                        # Gaussian prior: s = 0.3 +- 0.1
    return prob
# --8<-- [end:problem]


# --8<-- [start:fit]
prob = build()
res = prob.fit("scipy")                  # least squares (trf); the default
print(res.text())                        # parameters, errors, chi2/dof, constraints, correlations
# --8<-- [end:fit]

if __name__ == "__main__":                       # not when another script imports this one
    # --8<-- [start:compare]
    available = fitting.available_backends()
    for backend, algorithm in (("scipy", "dogbox"), ("bumps", "lm"), ("bumps", "amoeba")):
        if backend not in available:
            continue
        r = build().fit(backend, algorithm)
        print(f"{backend}/{algorithm}: chi2/dof = {r.chi2_red:.4g}, {r.nfev} evaluations, {r.seconds:.1f} s")
    # --8<-- [end:compare]

    q, I_fit = res.curves[0]
    ds = prob.datasets[0]
    plt.errorbar(ds.q, ds.I, ds.sigma, fmt="o", ms=3, color="0.3", label="data")
    plt.plot(q, I_fit, "r-", label=f"fit, chi2/dof = {res.chi2_red:.3g}")
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("q")
    plt.ylabel("I(q)")
    plt.legend()
    plt.show()
