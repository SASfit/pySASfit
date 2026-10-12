"""SASstudio Python plugin: PLUGIN_NAME

Plugin type (form factor 'ff', structure factor 'sq', size distribution 'sd'): ff
Author name:
Author email address:
Plugin tags (to aid automated model selection, e.g. gaussian):

Save as sasstudio_PLUGIN_NAME.py in a folder of pysasfit.studio.paths.python_plugin_dirs()
(e.g. ~/.pysasfit/studio_plugins) and use it with pysasfit.studio.function("ff_PLUGIN_NAME").
"""

import math

LABEL = "PLUGIN_NAME"

# one entry per parameter, in the order of p[0], p[1], ...: (name, description, default, (lower, upper))
PARAMETERS = [
    ("I0", "forward scattering", 1.0, (0.0, math.inf)),
    ("Rg", "radius of gyration", 10.0, (0.0, math.inf)),
]


def scattering_intensity_PLUGIN_NAME(q, p):
    """I(q); p is the full parameter vector p[0..49]."""
    I0, Rg = p[0], p[1]
    return I0 * math.exp(-((q * Rg) ** 2) / 3.0)


def scattering_amplitude_PLUGIN_NAME(q, p):
    """Scattering amplitude F(q) (optional, needed for the decoupling approximation)."""
    I0, Rg = p[0], p[1]
    return math.sqrt(I0) * math.exp(-((q * Rg) ** 2) / 6.0)


def volume_PLUGIN_NAME(x, p, dist):
    """Particle volume with parameter number `dist` replaced by x (optional, needed for the LMA)."""
    Rg = x if dist == 1 else p[1]
    R = math.sqrt(5.0 / 3.0) * Rg
    return 4.0 / 3.0 * math.pi * R**3
