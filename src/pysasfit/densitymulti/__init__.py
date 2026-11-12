"""pyDENSSmulti: 3D maps from several SAS contrasts (SANS contrast variation, SANS + SAXS) with a DENSS-type
iteration of its own (numpy/scipy; DENSS itself is not needed).

core   contrasts and components (SLDs, D2O fraction, H/D exchange), contrast matrix and identifiability check,
       the reconstruction (component basis with per-voxel bounded least squares, or DENSS-Multiple-type
       averaging), several maps, alignment and averaging, output files, synthetic test objects, self test
gui    Qt window (``pysasfit-densitymulti`` and the "pyDENSSmulti" tab of the launcher)

The iteration follows T. D. Grant, Nature Methods 15, 191 (2018) (DENSS); the averaging scheme follows
J. Sumner & S. Qian, BBA Advances 2, 100063 (2022) (DENSS-Multiple). Both are re-implemented from the papers.
"""

from .core import (  # noqa: F401
    PRESETS,
    SLD_D2O,
    SLD_H2O,
    SLD_XRAY_WATER,
    AverageResult,
    Component,
    Contrast,
    MultiError,
    MultiResult,
    Settings,
    align_and_average,
    bounded_lsq,
    contrast_matrix,
    core_shell_curves,
    debye_profile,
    dumbbell_curves,
    fit_tail_background,
    identifiability,
    load_result,
    make_grid,
    match_sld,
    number_density,
    project_capped_simplex,
    reconstruct,
    reconstruct_many,
    save_result,
    selftest,
    smooth_curve,
    sphere_curves,
)

__all__ = ["PRESETS", "SLD_D2O", "SLD_H2O", "SLD_XRAY_WATER", "AverageResult", "Component", "Contrast", "MultiError",
           "MultiResult", "Settings", "align_and_average", "bounded_lsq", "contrast_matrix", "core_shell_curves",
           "debye_profile", "dumbbell_curves", "fit_tail_background", "identifiability", "load_result", "make_grid", "match_sld",
           "number_density", "project_capped_simplex", "reconstruct", "reconstruct_many", "save_result", "selftest",
           "smooth_curve", "sphere_curves"]
