"""Open the model explorer from a script, with a saved project or with a model of your own.

    python examples/studio/05_open_explorer.py
"""

import os

from pysasfit.studio.explorer import explore

HERE = os.path.dirname(os.path.abspath(__file__))

# --8<-- [start:explore]
# a sphere with a log-normal size distribution and a hard-sphere structure factor
explore("ff_sphere", sd="sd_LogNorm", sq="sq_HardSphere", qmin=0.005, qmax=0.5)

# a mixture: the second contribution is a cylinder with a size distribution
# explore("ff_sphere", extra=[("ff_cylinder", "sd_LogNorm", None)])

# a saved session (File > Save project in the GUI)
# explore(project=os.path.join(HERE, "data", "project.sasstudio"))
# --8<-- [end:explore]
