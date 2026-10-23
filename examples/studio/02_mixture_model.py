"""Build a model from contributions: polydisperse spheres with a structure factor plus a power-law background.

    python examples/studio/02_mixture_model.py
"""

import matplotlib.pyplot as plt
import numpy as np

from pysasfit.studio import ModelFunction, ScatteringContribution

# --8<-- [start:model]
spheres = ScatteringContribution(
    "ff_sphere", {"R": 1, "eta": 1},                       # form factor and its fixed parameters
    size_distribution="sd_LogNorm",                        # N(x) ...
    sd_params={"N": 1e-11, "s": 0.2, "p": 1, "mu": 70},
    distributed="R",                                       # ... smears the parameter R
    structure_factor="sq_HardSphere",
    sq_params={"RHS": 70, "fp": 0.2},
    approach="lma",                                        # 'monodisperse' | 'decoupling' | 'lma'
)
model = ModelFunction([spheres], background=0.1, bg_A=1e-6, bg_alpha=4.0)   # + 0.1 + A q^-alpha
# --8<-- [end:model]

q = np.geomspace(5e-3, 0.5, 200)
I = model.intensity(q)

# --8<-- [start:variants]
curves = {}
for approach in ("monodisperse", "decoupling", "lma"):
    spheres.set_approach(approach)
    curves[approach] = model.intensity(q)
# --8<-- [end:variants]

plt.figure(figsize=(6, 4))
for approach, y in curves.items():
    plt.loglog(q, y, label=approach)
plt.loglog(q, I, "k:", lw=0.8, label="last setting")
plt.xlabel("q")
plt.ylabel("I(q)")
plt.legend()
plt.show()
