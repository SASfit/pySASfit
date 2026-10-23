"""Evaluate SASfit functions directly: form factor, structure factor and size distribution.

    python examples/studio/01_evaluate_functions.py
"""

import matplotlib.pyplot as plt
import numpy as np

from pysasfit import studio

# --8<-- [start:catalogue]
print(studio.info("ff_sphere"))                       # parameters with descriptions
for hit in studio.search("hard", "sphere", type="sq"):  # search the catalogue (works without the libraries)
    print(hit.name, "-", hit.label)
# --8<-- [end:catalogue]

# --8<-- [start:evaluate]
q = np.geomspace(1e-3, 0.5, 300)

sphere = studio.function("ff_sphere")
I_sphere = sphere(q, R=50, eta=1)                      # named parameters ...
F_sphere = sphere.amplitude(q, R=50, eta=1)            # ... the scattering amplitude F(q)

hard_sphere = studio.function("sq_HardSphere")
S = hard_sphere(q, RHS=50, fp=0.3)                     # structure factor S(q)

lognorm = studio.function("sd_LogNorm")
x = np.linspace(1, 150, 400)
N = lognorm(x, N=1, s=0.2, p=1, mu=50)                 # size distribution N(x)
# --8<-- [end:evaluate]

fig, axes = plt.subplots(1, 3, figsize=(12, 3.5), constrained_layout=True)
axes[0].loglog(q, I_sphere, label="P(q), R = 50")
axes[0].loglog(q, I_sphere * S, label="P(q) S(q)")
axes[0].set(xlabel="q", ylabel="I(q)")
axes[0].legend()
axes[1].semilogx(q, S)
axes[1].set(xlabel="q", ylabel="S(q)", title="hard spheres, fp = 0.3")
axes[2].plot(x, N)
axes[2].set(xlabel="x", ylabel="N(x)", title="log-normal, mu = 50, s = 0.2")
plt.show()
