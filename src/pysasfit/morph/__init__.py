"""pySASmorph: representative 3D two- and multi-phase morphologies from SAS data (SASmorph).

A clean reimplementation of the SAXSMorph method (Ingham, Li, Allen & Toney,
J. Appl. Cryst. 44 (2011) 221): the measured I(q) is turned into the Debye
correlation function gamma(r), inverted through the level-cut relation of a
clipped Gaussian random field (Berk 1991, Quintanilla 2007) into the field
correlation g(r), and a random field with that correlation is generated on a
voxel grid and cut at the porosity.

Multi-phase (:mod:`.multiphase`): the same field cut at several levels gives ordered phases
(a middle phase is a layer between its neighbours: Berk 1991, Teubner 1991, Roberts 1997,
Arleth, Marcelja & Zemb 2001); one or several curves (contrast series) determine one common g(r),
optionally the volume fractions.

Modules
-------
transforms     exact sine transforms of piecewise-linear functions, analytic Porod tail
extrapolation  Guinier/linear/constant low-q and Porod high-q extrapolation, invariant
correlation    level-cut relation F(g), g(r) from gamma(r), spectral function f(k)
grf            Gaussian random fields (FFT spectral synthesis, Berk's random waves), level cut
statistics     S/V, clusters, percolation, bicontinuity
scattering     back-calculated I(q) of the voxel structure and of the level-cut model
pipeline       two-phase: MorphParameters, reconstruct(), correlations()
multiphase     N phases, contrast series: MultiParameters, DataSet, reconstruct_multi(), correlations_multi()
export         curves (text), voxel matrix (VTK, HDF5, TIFF stack, npy)
surface        interface meshes (marching cubes), largest cluster, STL
view3d_gl      3D panel (pyqtgraph / OpenGL) of the GUI
view3d         optional separate 3D window with PyVista
gui            Qt window (``pysasfit-morph``, launcher tab "pySASmorph")

No Qt in the numerical modules: everything runs headless.
"""

from .correlation import LevelCut, spectral_function, wavevector_distribution
from .extrapolation import ExtrapolatedCurve, extrapolate
from .grf import field_cosine, field_fft, level_cut
from .pipeline import (MorphParameters, MorphResult, correlations, porosity_from_invariant, reconstruct,
                       suggest_box)
from .scattering import contrast_from_invariant, levelcut_intensity, voxel_intensity
from .multiphase import (DataSet, MultiCut, MultiParameters, MultiResult, PhaseSystem, correlations_multi,
                         reconstruct_multi)
from .statistics import morphology_statistics, specific_surface_porod, specific_surface_voxels
from .anisotropic import AnisoParameters, AnisoResult, Data2D, correlations_aniso, load_2d, reconstruct_aniso

__all__ = [
    "MorphParameters", "MorphResult", "reconstruct", "correlations", "suggest_box", "porosity_from_invariant",
    "ExtrapolatedCurve", "extrapolate", "LevelCut", "spectral_function", "wavevector_distribution",
    "field_fft", "field_cosine", "level_cut", "voxel_intensity", "levelcut_intensity", "contrast_from_invariant",
    "morphology_statistics", "specific_surface_porod", "specific_surface_voxels",
    "DataSet", "PhaseSystem", "MultiCut", "MultiParameters", "MultiResult", "correlations_multi", "reconstruct_multi",
    "Data2D", "load_2d", "AnisoParameters", "AnisoResult", "correlations_aniso", "reconstruct_aniso",
]
