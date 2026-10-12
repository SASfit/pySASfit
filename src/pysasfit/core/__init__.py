"""Shared infrastructure for all pysasfit tools.

io          ``load_data`` -- q, I, sigma [, dq] text files (pyOZgui ``loadCurve`` conventions)
resolution  pinhole resolution with Rician (default) or Gaussian kernel -- the same
            operator as pyOZgui -- plus slit smearing and user matrices
background  constant, power-law, Ornstein-Zernike and user-function backgrounds,
            linear or nonlinear parameters
"""

from . import background, io, resolution
from .background import Background, BackgroundTerm, Constant, FunctionTerm, OrnsteinZernike, PowerLaw
from .io import load_data
from .resolution import (
    GaussianResolution,
    MatrixResolution,
    NoResolution,
    Resolution,
    RicianResolution,
    SlitResolution,
)

__all__ = [
    "io",
    "resolution",
    "background",
    "load_data",
    "Resolution",
    "RicianResolution",
    "GaussianResolution",
    "SlitResolution",
    "MatrixResolution",
    "NoResolution",
    "Background",
    "BackgroundTerm",
    "Constant",
    "PowerLaw",
    "OrnsteinZernike",
    "FunctionTerm",
]
