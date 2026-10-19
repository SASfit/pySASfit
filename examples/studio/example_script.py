"""SASstudio example, rewritten for pysasfit.studio.

Polydisperse spheres (LogNorm radius distribution) with hard-sphere interactions,
compared with the SASfit test data set, then fitted.

    pysasfit-studio install-binaries C:/Users/<you>/switchdrive/SASstudio   # once
    python examples/studio/example_script.py
"""

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
try:
    import pysasfit.studio  # noqa: F401
except ImportError:
    sys.path.insert(0, os.path.join(HERE, "..", "..", "src"))

import matplotlib.pyplot as plt
import numpy as np

from pysasfit import studio
from pysasfit.studio import ModelFunction, Pipeline, ScatteringContribution

print(studio.info("ff_sphere"))
print(studio.info("sd_LogNorm"))

pipe = Pipeline()
d = pipe.add_dataset(os.path.join(HERE, "data", "SASfitTestData", "test.dat"))

# bimodal sphere model of the SASfit test data (data/README.md). SASfit's result
# (data/SASfitTestData/SASfit_test.sas): N = 3.43e-12, s = 0.304, mu = 155.2 and
# N = 7.40e-11, s = 0.200, mu = 69.4, background 0.09995. Start about 30 % off:
big = ScatteringContribution("ff_sphere", {"R": 1, "eta": 1}, size_distribution="sd_LogNorm",
                             sd_params={"N": 2.5e-12, "s": 0.25, "p": 1, "mu": 120}, distributed="R")
small = ScatteringContribution("ff_sphere", {"R": 1, "eta": 1}, size_distribution="sd_LogNorm",
                               sd_params={"N": 1e-10, "s": 0.25, "p": 1, "mu": 55}, distributed="R")
m = pipe.add_model(ModelFunction([big, small], background=0.07), dataset=d)
print(pipe)

q = pipe.datasets[d]["q"]
I0 = pipe.evaluate(m)
res = pipe.fit(m, ["0.sd.N", "0.sd.mu", "0.sd.s", "1.sd.N", "1.sd.mu", "1.sd.s", "background"],
               bounds=([0, 1, 0.01, 0, 1, 0.01, 0], [np.inf, 1e4, 2, np.inf, 1e4, 2, np.inf]))
print(res.summary())

ds = pipe.datasets[d]
plt.errorbar(ds["q"], ds["I"], ds["sigma"], fmt="o", ms=3, color="k", label="data")
plt.plot(q, I0, "--", label="start")
plt.plot(res.q, res.I_fit, "r-", label="fit")
plt.xscale("log")
plt.yscale("log")
plt.xlabel("q")
plt.ylabel("I(q)")
plt.legend()
plt.show()
