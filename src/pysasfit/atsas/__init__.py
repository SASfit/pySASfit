"""pyATsas: the ATSAS programs (EMBL Hamburg / BIOSAXS GmbH) from Python and from the pySASfit launcher.

core   thin wrapper around the ATSAS command-line programs: finds the installation (Windows, Linux,
       macOS), runs autorg, datgnom, gnom, crysol / cryson, dammif and parses their output; starts the
       interactive ATSAS programs (PRIMUS, CHROMIXS, PEAK, ...)
gui    Qt panel (``pysasfit-atsas`` and the "pyATsas" tab of the launcher): Guinier analysis, p(r),
       model curves and bead models of the loaded curves, with plots and a results table

ATSAS is not part of pySASfit: it has to be installed separately, with its own licence
(https://www.embl-hamburg.de/biosaxs/software.html).  The pyATsas icon is the original ATSAS icon, read
from that installation at run time.
"""

from .core import (  # noqa: F401
    AtsasError,
    CrysolResult,
    DammifResult,
    GnomResult,
    GuinierResult,
    GUI_PROGRAMS,
    atsas_root,
    autorg,
    crysol,
    dammif,
    datgnom,
    find_atsas,
    gnom,
    icon_files,
    launch,
    license_file,
    parse_gnom_out,
    program_path,
    read_dat,
    run_tool,
    selftest,
    sphere_curve,
    version,
    write_dat,
)

__all__ = ["AtsasError", "CrysolResult", "DammifResult", "GnomResult", "GuinierResult", "GUI_PROGRAMS",
           "atsas_root", "autorg", "crysol", "dammif", "datgnom", "find_atsas", "gnom", "icon_files", "launch",
           "license_file", "parse_gnom_out", "program_path", "read_dat", "run_tool", "selftest", "sphere_curve",
           "version", "write_dat"]
